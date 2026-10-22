"""Qwen3.5 / Qwen3.6 tool-use solver (Table protocol, first block): Qwen-VL processor at 1 fps, tools declared through
the chat template, at most 6 turns of at most 8,192 tokens, temperature 0.1 / top-p 0.9 / top-k 20 / repetition 1.05.
Used for the Qwen backbones, the TriPhys models (base + LoRA adapter), and on-policy sampling in both training stages
(same loop, sampling temperature set by the caller)."""
import json
import logging
import os
import re

os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')

import torch

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .tools import TOOLS, WebSearch, parse_tool_calls, run_tool

logger = logging.getLogger(__name__)

MAX_ROUNDS = 6
MAX_TOKENS = 8192
EVAL_DECODING = dict(do_sample=True, temperature=0.1, top_p=0.9, top_k=20, repetition_penalty=1.05)
ANSWER_RE = re.compile(r'<answer>.*?</answer>', re.S)


def decoding(temperature=None):
    d = dict(EVAL_DECODING)
    if temperature is not None:
        d['temperature'] = temperature
    return d


def initial_messages(qa, video):
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [{'type': 'video', 'video': video, 'fps': 1.0},
                                         {'type': 'text', 'text': build_user_prompt(qa['question'],
                                                                                    qa['question_type'])}]}]


def split_video_grid(inputs):
    """The Qwen3.5 processor puts a timestamp before every temporal patch, so a video becomes T token segments, while
    video_grid_thw keeps one (T, h, w) row; some transformers releases then run out of grid rows in get_rope_index.
    Splitting the row into T rows of (1, h, w) is exact (the vision encoder already attends within each temporal
    patch) and is applied only when the segment count exceeds the row count."""
    g, tt = inputs.get('video_grid_thw'), inputs.get('mm_token_type_ids')
    if g is None or tt is None:
        return inputs
    segs = int(((tt == 2) & (torch.nn.functional.pad(tt, (1, 0))[:, :-1] != 2)).sum())
    if segs > g.shape[0]:
        g = torch.repeat_interleave(g, g[:, 0], dim=0).clone()
        g[:, 0] = 1
        inputs['video_grid_thw'] = g
    return inputs


class QwenAgent:
    def __init__(self, base_model, adapter=None, device='cuda:0', temperature=None, search=None, device_map=None):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device_map or device)
        if adapter and adapter != 'none':
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.gen = decoding(temperature)
        self.search = search or WebSearch()
        self.im_end = self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self._video_cache = {}

    # ------------------------------------------------------------ inputs
    def video(self, path):
        if path not in self._video_cache:
            from qwen_vl_utils import process_vision_info
            vm = [{'role': 'user', 'content': [{'type': 'video', 'video': path, 'fps': 1.0}]}]
            _, vi, vk = process_vision_info(vm, return_video_kwargs=True)
            if vi:
                n = vi[0].shape[0]
                vk['video_metadata'] = [{'total_num_frames': n, 'fps': 1.0, 'frames_indices': list(range(n))}]
            vk = {k: v if k == 'video_metadata' else (v[0] if isinstance(v, list) and len(v) == 1 else v)
                  for k, v in vk.items()}
            self._video_cache = {path: (vi, vk)}          # one video at a time (questions are grouped by video)
        return self._video_cache[path]

    def _inputs(self, messages, vi, vk):
        text = self.processor.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        return split_video_grid(self.processor(text=[text], images=None, videos=vi, padding=True, return_tensors='pt',
                                               **vk)).to(self.device)

    def _decode(self, ids):
        if self.im_end in ids:
            ids = ids[:ids.index(self.im_end)]
        return self.processor.tokenizer.decode(ids, skip_special_tokens=False).replace('<|im_end|>', '').strip()

    # ------------------------------------------------------------ generation
    def round(self, messages, vi, vk, n=1):
        """n sampled assistant turns for the same context (one generate call)"""
        inputs = self._inputs(messages, vi, vk)
        L = inputs['input_ids'].shape[1]
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, eos_token_id=self.im_end,
                                      num_return_sequences=n, **self.gen)
        return [self._decode(row) for row in out[:, L:].tolist()]

    def tools_step(self, content, messages, turns, tool_log, qid='', rnd=0):
        """execute the calls of one assistant turn; returns the result signature (None if the turn has no call)"""
        calls = parse_tool_calls(content)
        if not calls:
            return None
        messages.append({'role': 'assistant', 'content': content})
        turns.append({'role': 'assistant', 'content': content})
        res = []
        for name, args in calls:
            r = run_tool(name, args, self.search, qid)
            tool_log.append({'round': rnd, 'tool': name, 'args': args, 'result': json.loads(r).get('result', '')[:200]})
            messages.append({'role': 'tool', 'content': r})
            turns.append({'role': 'tool', 'content': r})
            res.append(r)
        return str(res)

    def solve(self, qa, video, qid='', messages=None, rounds_done=0, first=None):
        """the evaluation loop. Each turn either answers (stop), calls tools (results appended as tool messages), or
        does neither (stop); identical tool results in consecutive turns stop the loop. `messages` / `rounds_done`
        continue from a prefix, `first` is an already sampled next turn. Returns (last turn, turns, tool_log)."""
        vi, vk = self.video(video)
        messages = [dict(m) for m in (messages or initial_messages(qa, video))]
        turns, tool_log, prev, content, logged = [], [], None, '', False
        for k in range(MAX_ROUNDS - rounds_done):
            content = first if (k == 0 and first is not None) else self.round(messages, vi, vk)[0]
            if ANSWER_RE.search(content):
                logged = False
                break
            sig = self.tools_step(content, messages, turns, tool_log, qid, rounds_done + k)
            logged = sig is not None
            if sig is None or sig == prev:
                break
            prev = sig
        if not logged:
            turns.append({'role': 'assistant', 'content': content})
        return content, turns, tool_log
