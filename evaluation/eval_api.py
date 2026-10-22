"""Evaluate GPT-5.5 and Gemini-3.6-Flash through an OpenAI-compatible chat endpoint (Table protocol, second block).

  GPT-5.5           frames at 2 fps, 8..48 per clip, longer side 768 px (JPEG), each preceded by its timestamp
  Gemini-3.6-Flash  the whole clip as H.264 video (data:video/mp4): MPR 10 fps at 480p, KPR native frame rate at 720p
Same system prompt and answer-form instruction as the Qwen models; web_search / code_executor declared through function
calling; at most 6 turns of at most 8,192 output tokens; temperature 0.1, top-p 0.9 (the endpoints take no top-k /
repetition penalty). Tool results are returned as role "tool" messages.

  export OPENROUTER_API_KEY=... SERPER_API_KEYS=... DEEPSEEK_API_KEY=...
  python evaluation/eval_api.py --model gpt --out results/gpt55.jsonl
  python evaluation/eval_api.py --model gemini --out results/gemini36flash.jsonl
The model ids default to GPT_MODEL / GEMINI_MODEL (OpenRouter ids openai/gpt-5.5 / google/gemini-3.6-flash); set
CHAT_API_BASE / GPT_MODEL / GEMINI_MODEL to use another OpenAI-compatible endpoint."""
import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triphys import config, data, evalrun, llm
from triphys.prompts import SYSTEM_PROMPT, build_user_prompt
from triphys.tools import TOOLS, WebSearch, run_tool

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MAX_ROUNDS = 6
MAX_TOKENS = 8192
EMPTY_RETRIES = 4


# ---------------------------------------------------------------- video input
def ffmpeg_bin():
    if os.environ.get('FFMPEG'):
        return os.environ['FFMPEG']
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def gemini_video(path, subtask, cache_dir):
    """re-encoded H.264 clip (cached): MPR 10 fps / 480p, KPR native fps / 720p"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / (hashlib.md5(path.encode()).hexdigest() + '_%s.mp4' % subtask)
    if not out.exists():
        vf = 'fps=10,scale=-2:480' if subtask == 'MPR' else 'scale=-2:720'
        tmp = out.with_suffix('.tmp.mp4')
        subprocess.run([ffmpeg_bin(), '-y', '-v', 'error', '-i', path, '-vf', vf, '-c:v', 'libx264', '-crf', '28',
                        '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-an', str(tmp)], check=True)
        tmp.replace(out)
    return [{'type': 'image_url', 'image_url': {'url': 'data:video/mp4;base64,' +
                                                       base64.b64encode(out.read_bytes()).decode()}}]


def gpt_frames(path):
    """2 frames per second, 8..48 frames, longer side 768 px, each preceded by its timestamp"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    n_total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30
    n = max(8, min(48, round(2 * n_total / fps)))
    frames = []
    for i in range(n):
        idx = int((i + 0.5) * n_total / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        s = 768 / max(h, w)
        if s < 1:
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        frames.append((idx / fps, base64.b64encode(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()))
    cap.release()
    content = [{'type': 'text', 'text': 'Video (%.1f s) given as %d frames sampled uniformly in time:'
                                        % (n_total / fps, len(frames))}]
    for t, b in frames:
        content += [{'type': 'text', 'text': 't = %.2f s' % t},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b}}]
    return content


# ---------------------------------------------------------------- tool loop
class APISolver:
    def __init__(self, kind, model, temperature, top_p, cache_dir, search=None):
        self.kind, self.model, self.temperature, self.top_p = kind, model, temperature, top_p
        self.cache_dir = Path(cache_dir)
        self.base, self.key = config.CHAT_API_BASE, config.chat_key(model)
        self.search = search or WebSearch()

    def video_content(self, item):
        if self.kind == 'gemini':
            return gemini_video(item['video'], item['subtask'], self.cache_dir)
        return gpt_frames(item['video'])

    def __call__(self, item):
        qa = item['qa']
        user = self.video_content(item) + [{'type': 'text', 'text': build_user_prompt(qa['question'],
                                                                                        qa['question_type'])}]
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
        turns, tool_log, prev, content = [], [], None, ''
        for rnd in range(MAX_ROUNDS):
            for _ in range(EMPTY_RETRIES):          # an empty reply (no text, no call) is re-requested
                resp = asyncio.run(llm.chat(self.base, self.key, self.model, messages, temperature=self.temperature,
                                            max_tokens=MAX_TOKENS, timeout=600.0, retries=6, tools=TOOLS,
                                            top_p=self.top_p))
                msg = resp['choices'][0]['message']
                content = (msg.get('content') or '').strip()
                calls = msg.get('tool_calls') or []
                if content or calls:
                    break
                logger.warning('%s: empty reply in turn %d, retrying', item['qid'], rnd + 1)
            turns.append({'role': 'assistant', 'content': content,
                          'tool_calls': [c['function'] for c in calls] if calls else None})
            if '<answer>' in content or not calls:
                break
            messages.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            res = []
            for c in calls:
                name = c['function']['name']
                try:
                    args = json.loads(c['function'].get('arguments') or '{}')
                except json.JSONDecodeError:
                    args = {}
                r = run_tool(name, args, self.search, item['qid'])
                tool_log.append({'round': rnd, 'tool': name, 'args': args,
                                 'result': json.loads(r).get('result', '')[:200]})
                messages.append({'role': 'tool', 'tool_call_id': c.get('id', ''), 'content': r})
                turns.append({'role': 'tool', 'content': r})
                res.append(r)
            if str(res) == prev:                  # identical tool results in consecutive turns stop the loop
                break
            prev = str(res)
        return content, turns, tool_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['gpt', 'gemini'])
    ap.add_argument('--model_id', default='', help='overrides GPT_MODEL / GEMINI_MODEL')
    ap.add_argument('--subtask', default='both', choices=['MPR', 'KPR', 'both'])
    ap.add_argument('--split', default='test', choices=['test', 'train', 'all'])
    ap.add_argument('--out', required=True, help='output .jsonl (resumable)')
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--video_cache', default='cache/gemini_videos')
    ap.add_argument('--skip_score', action='store_true')
    a = ap.parse_args()

    model = a.model_id or (config.GPT_MODEL if a.model == 'gpt' else config.GEMINI_MODEL)
    items = data.load(data.subtasks_arg(a.subtask), a.split)
    if a.limit:
        items = items[:a.limit]
    solver = APISolver(a.model, model, a.temperature, a.top_p, a.video_cache)
    evalrun.run(items, solver, a.out, thinking_opened=False, score=not a.skip_score, model_name=model,
                workers=a.workers)
    if solver.search.failures:
        logger.warning('%d web searches failed (marked __SERPER_FAILED__)', len(solver.search.failures))


if __name__ == '__main__':
    main()
