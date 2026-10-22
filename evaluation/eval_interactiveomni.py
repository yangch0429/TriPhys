"""Evaluate InteractiveOmni-4B (Table protocol, third block) through its official chat interface.

  video     model.chat(..., max_patch_num=12, frame=8): 8 uniformly sampled frames, at most 12 image tiles per frame
  prompt    same system prompt and answer-form instruction as the Qwen models
  tools     the official chat() takes no tools argument; its language model is Qwen3-4B, so the tools are declared in
            the system message in the Qwen3 format (<tools>...</tools>, calls as <tool_call>{json}</tool_call>) and
            tool results are returned as a user turn wrapped in <tool_response>...</tool_response>
  budget    at most 6 turns of at most 8,192 tokens, temperature 0.1, top-p 0.9

  python evaluation/eval_interactiveomni.py --model sensenova/InteractiveOmni-4B --out results/interactiveomni4b.jsonl
Requires the model's remote code (trust_remote_code) and decord."""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triphys import data, evalrun
from triphys.prompts import SYSTEM_PROMPT, build_user_prompt
from triphys.tools import TOOLS, WebSearch, parse_tool_calls, run_tool

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MAX_ROUNDS = 6
MAX_TOKENS = 8192
MAX_TILES, N_FRAMES = 12, 8


def qwen3_tool_system(system):
    """the system turn the Qwen3 chat template writes when tools are given"""
    return (system + '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
            'You are provided with function signatures within <tools></tools> XML tags:\n<tools>'
            + ''.join('\n' + json.dumps(t, ensure_ascii=False) for t in TOOLS)
            + '\n</tools>\n\nFor each function call, return a json object with function name and arguments within '
              '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
              '<args-json-object>}\n</tool_call>')


class InteractiveOmniSolver:
    def __init__(self, model_path, search=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.search = search or WebSearch()
        self.system = qwen3_tool_system(SYSTEM_PROMPT)

    def chat(self, messages):
        gen = dict(max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.1, top_p=0.9)
        return self.model.chat(self.tokenizer, gen, messages, MAX_TILES, N_FRAMES, print_flag=False).strip()

    def __call__(self, item):
        qa = item['qa']
        messages = [{'role': 'system', 'content': self.system},
                    {'role': 'user', 'content': [{'type': 'video', 'video': item['video']},
                                                 {'type': 'text', 'text': build_user_prompt(qa['question'],
                                                                                            qa['question_type'])}]}]
        turns, tool_log, prev, content = [], [], None, ''
        for rnd in range(MAX_ROUNDS):
            content = self.chat(messages)
            turns.append({'role': 'assistant', 'content': content})
            calls = parse_tool_calls(content)
            if '<answer>' in content or not calls:
                break
            res = []
            for name, args in calls:
                r = run_tool(name, args, self.search, item['qid'])
                tool_log.append({'round': rnd, 'tool': name, 'args': args,
                                 'result': json.loads(r).get('result', '')[:200]})
                res.append(r)
            reply = '\n'.join('<tool_response>\n%s\n</tool_response>' % r for r in res)
            messages += [{'role': 'assistant', 'content': content}, {'role': 'user', 'content': reply}]
            turns.append({'role': 'tool', 'content': reply})
            if str(res) == prev:                  # identical tool results in consecutive turns stop the loop
                break
            prev = str(res)
        return content, turns, tool_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sensenova/InteractiveOmni-4B')
    ap.add_argument('--subtask', default='both', choices=['MPR', 'KPR', 'both'])
    ap.add_argument('--split', default='test', choices=['test', 'train', 'all'])
    ap.add_argument('--out', required=True, help='output .jsonl (resumable)')
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip_score', action='store_true')
    a = ap.parse_args()

    items = evalrun.shard(data.load(data.subtasks_arg(a.subtask), a.split), a.shard_id, a.n_shards)
    if a.limit:
        items = items[:a.limit]
    solver = InteractiveOmniSolver(a.model)
    evalrun.run(items, solver, a.out, thinking_opened=False, score=not a.skip_score,
                model_name=os.path.basename(a.model.rstrip('/')))


if __name__ == '__main__':
    main()
