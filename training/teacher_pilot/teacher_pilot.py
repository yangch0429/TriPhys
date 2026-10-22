"""10/15 teacher forward-solve pilot (OmniAtlas style): Gemini (OpenAI-compatible endpoint, whole clip as base64 video)
solves f10 train QAs with REAL tools and never sees the GT. Same SYSTEM_PROMPT / user prompt as eval_sft_v9, same tool
protocol (<tool_call>{"name": web_search|code_executor, "arguments": ...}</tool_call>, at most 6 rounds); tool results
come from the eval_sft_v9 Serper search and a python exec with math/numpy. Each round = one API call: the reply is cut
after the last </tool_call> (the teacher must not invent tool output), the tools run, their results go back as a
<tool_response> user message. Scoring = eval_sft_v9 (metrics_final ACC gpt-4o-mini, SCS deepseek-flash).
Output: per trajectory the full turn list (training format), tool log, usage tokens, ACC/SCS.
Keys: OPENROUTER_API_KEY (Gemini, CHAT_API_BASE / GEMINI_MODEL), SERPER_API_KEYS, DEEPSEEK_API_KEY (SCS judge).
Usage (from the work dir $TRIPHYS_WORK): python fc/training/teacher_pilot/teacher_pilot.py OUT.json [K] [N_PER_CLASS]"""
import asyncio, base64, json, logging, os, random, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'scripts'))
from srvpaths import E, EVAL_LIB, N1, EGOEXO_ROOT, BAAI_ROOT  # noqa: E402  machine paths, see srvpaths.py
# 10/16: experiment_right must come first (scripts/metrics_final is the old all-gpt-4o-mini judge; SCS must be Flash)
sys.path.insert(0, EVAL_LIB)
from prompts import SYSTEM_PROMPT, build_user_prompt
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback

# eval_sft_v9 parses argv at import time -> take its Serper block (keys + query_web_search) from the source
# (the block is identical in n1/eval_n1.py, which is the copy shipped here)
_src = open(N1 + '/eval_n1.py').read()
_ns = {}
exec('import json, os, time, logging, requests\nlogger = logging.getLogger("serper")\n'
     + _src[_src.index('SERPER_KEYS = ['):_src.index('class SFTEvaluator')], _ns)
query_web_search = _ns['query_web_search']

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('teacher')

URL = os.environ.get('CHAT_API_BASE', 'https://openrouter.ai/api/v1').rstrip('/') + '/chat/completions'
MODEL, KEY = os.environ.get('GEMINI_MODEL', 'google/gemini-3.6-flash'), os.environ.get('OPENROUTER_API_KEY', '')
try:
    import imageio_ffmpeg
    FFMPEG = os.environ.get('FFMPEG') or imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = os.environ.get('FFMPEG', 'ffmpeg')
VCACHE = Path(E) / 'vid_cache'
PY = sys.executable
CONCURRENCY, MAX_ROUNDS, SEED = 6, 6, 20261015

TOOL_SPEC = """

TOOLS. You have two tools. Call a tool by writing exactly one block per call:
<tool_call>{"name": "web_search", "arguments": {"query": "..."}}</tool_call>   Google search, returns the top snippets
<tool_call>{"name": "code_executor", "arguments": {"code": "..."}}</tool_call>  runs Python (math, numpy available); print() what you need
Protocol for every turn:
- First write your reasoning inside <think>...</think>: what you see in the video / what the last tool result says,
  which value you take from it (quote it) and why, and what you need next.
- Then either call tool(s) and STOP writing (the results arrive in the next message as <tool_response>), or, when
  you are done, give the final <answer> block in the required format.
- Never write a <tool_response> yourself and never assume a search result you have not received.
- Constants, object specifications and material properties must come from web_search (not from memory); if a
  search does not give the value, search again with a better query. Do arithmetic with code_executor.
- You have at most 6 turns in total, so the final answer must come by the 6th turn."""

THINK_REMINDER = ('Every turn must start with <think>...</think> in which you write out your reasoning in full '
                  '(what you observe / what the tool result says, the value you take and why, what to do next), '
                  'then the tool call(s) or the final <answer>.')


def video_b64(path):
    VCACHE.mkdir(parents=True, exist_ok=True)
    out = VCACHE / (re.sub(r'[^A-Za-z0-9]+', '_', path.split('/triworld/')[-1]) + '.mp4')
    if not out.exists():
        vf = 'fps=10,scale=-2:480' if '/takes/' in path else 'scale=-2:720'
        subprocess.run([FFMPEG, '-y', '-v', 'error', '-i', path, '-vf', vf, '-c:v', 'libx264', '-crf', '28',
                        '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-an', str(out)], check=True)
    return base64.b64encode(out.read_bytes()).decode()


def exec_code(code):
    prelude = 'import math\nfrom math import *\nimport numpy as np\nnumpy = np\n'
    try:
        p = subprocess.run([PY, '-c', prelude + code], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return 'Error: timeout'
    if p.returncode:
        return 'Error: ' + (p.stderr.strip().splitlines() or ['unknown'])[-1]
    return p.stdout.strip() or 'No output'


async def call_api(messages):
    for attempt in range(12):
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                r = await client.post(URL, headers={'Authorization': 'Bearer ' + KEY},
                                      json={'model': MODEL, 'messages': messages, 'max_tokens': 8192,
                                            'temperature': 0.7, 'stop': ['<tool_response>'],
                                            'reasoning_effort': 'low'})
            if r.status_code == 200:
                j = r.json()
                c = j.get('choices', [{}])[0].get('message', {}).get('content')
                if c:
                    return c, j.get('usage', {})
                logger.warning('empty content (attempt %d)', attempt + 1)
            else:
                logger.warning('HTTP %d: %s (attempt %d)', r.status_code, r.text[:200], attempt + 1)
                if 'unknown provider' in r.text:
                    await asyncio.sleep(2)
                    continue
        except Exception as e:
            logger.warning('API error %s (attempt %d)', str(e)[:200], attempt + 1)
        await asyncio.sleep(10 * min(attempt + 1, 5))
    return '', {}


def select(n_per_class):
    """stratified by question_type x difficulty, fixed seed"""
    rng = random.Random(SEED)
    out = []
    for cls in 'AB':
        data = json.load(open('%s/data/f10/%sclass_qa_train.json' % (E, cls.lower())))
        pool = defaultdict(list)
        for item in data:
            for qi, qa in enumerate(item['qa_pairs']):
                if qa.get('valid', True):
                    pool[(qa['question_type'], qa.get('difficulty', ''))].append((cls, item, qi, qa))
        cells = sorted(pool)
        for v in pool.values():
            rng.shuffle(v)
        rng.shuffle(cells)
        picked, used = [], set()
        while len(picked) < n_per_class:
            for c in cells:
                while pool[c] and len(picked) < n_per_class:
                    cand = pool[c].pop()
                    vid = cand[1].get('take_uid') or cand[1].get('video_id')
                    if vid not in used:  # one QA per video
                        used.add(vid)
                        picked.append(cand)
                        break
        out += picked
    return out


def video_path(item):
    if item.get('take_uid'):
        return EGOEXO_ROOT + '/%s/best_exo.mp4' % item['take_uid']
    return BAAI_ROOT + '/%s/%s.mp4' % (item['folder_id'], item['video_id'])


async def solve(cls, item, qi, qa, k):
    vpath = video_path(item)
    v64 = await asyncio.to_thread(video_b64, vpath)
    user_text = build_user_prompt(qa['question'], qa['question_type'])
    api_msgs = [{'role': 'system', 'content': SYSTEM_PROMPT + TOOL_SPEC},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:video/mp4;base64,' + v64}},
                                             {'type': 'text', 'text': user_text + THINK_REMINDER}]}]
    turns, tool_log, usage, content, t0, n_retry = [], [], [], '', time.time(), [0]
    for rnd in range(MAX_ROUNDS):
        content, u = await call_api(api_msgs)
        usage.append(u)
        content = re.sub(r'^\s*\w{1,4}\s*(?=<)', '', content)  # stray leading token (seen: "Dan")
        n_retry[0] += 0 if re.match(r'\s*<think>.{80,}?</think>', content, re.DOTALL) else 1  # = turns without visible think
        if not content:
            break
        if re.search(r'<answer>.*?</answer>', content, re.DOTALL) and '<tool_call>' not in content.split('<answer>')[0]:
            turns.append({'role': 'assistant', 'content': content.strip()})
            break
        calls = list(re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL))
        if not calls:
            turns.append({'role': 'assistant', 'content': content.strip()})
            break
        content = content[:calls[-1].end()].strip()  # drop anything after the last call
        turns.append({'role': 'assistant', 'content': content})
        api_msgs.append({'role': 'assistant', 'content': content})
        results = []
        for m in calls:
            try:
                raw = m.group(1).strip()
                try:
                    tc = json.loads(raw, strict=False)
                except Exception:
                    tc = json.loads(re.sub(r'\]\s*$', '', raw), strict=False)
                name, args = tc.get('name', ''), tc.get('arguments', {}) or {}
            except Exception:
                name, args = 'unparsable', {}
            if name == 'web_search':
                res = await asyncio.to_thread(query_web_search, args.get('query', '') or args.get('q', ''), 'pilot')
            elif name == 'code_executor':
                res = json.dumps({'result': await asyncio.to_thread(exec_code, args.get('code', ''))})
            else:
                res = json.dumps({'result': "Tool '%s' unavailable." % name})
            tool_log.append({'round': rnd, 'tool': name, 'args': args, 'result': json.loads(res).get('result', '')})
            turns.append({'role': 'tool', 'content': res})
            results.append(res)
        api_msgs.append({'role': 'user', 'content': '\n'.join('<tool_response>\n%s\n</tool_response>' % r for r in results)})
    final = turns[-1]['content'] if turns and turns[-1]['role'] == 'assistant' else ''
    parsed = parse_response(final)
    rd = {'prediction': parsed.prediction, 'reasoning_chain': parsed.reasoning_chain, 'ground_truth': qa['answer'],
          'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
          'question': qa['question'], 'question_type': qa['question_type'],
          'needs_llm_eval': parsed.needs_llm_eval, 'raw_response': final}
    acc = await calculate_acc_with_fallback(rd)
    scs_res = await calculate_scs_with_fallback(rd)
    scs = scs_res.get('overall', scs_res.get('SCS', 0.0)) if isinstance(scs_res, dict) else float(scs_res or 0)
    uid = item.get('take_uid') or item.get('video_id')
    logger.info('%s %s k%d ACC %s SCS %.2f | %d rounds %d tools | %s', cls, uid[:8], k, acc, scs,
                sum(t['role'] == 'assistant' for t in turns), len(tool_log), str(parsed.prediction)[:60])
    return {'cls': cls, 'qid': '%s|%s|%d' % (cls, uid, qi), 'k': k, 'question_type': qa['question_type'],
            'difficulty': qa.get('difficulty'), 'question': qa['question'], 'ground_truth': qa['answer'],
            'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
            'sources': qa.get('sources', []), 'hidden_cv_chosen': item.get('hidden_cv_chosen'),
            'video': vpath, 'user_text': user_text, 'turns': turns, 'tool_log': tool_log, 'usage': usage,
            'prediction': parsed.prediction, 'ACC': acc, 'SCS': scs, 'seconds': round(time.time() - t0, 1), 'turns_without_think': n_retry[0]}


async def main(out, k, n):
    jobs = [(c, it, qi, qa, j) for (c, it, qi, qa) in select(n) for j in range(k)]
    done = json.load(open(out)) if Path(out).exists() else []
    seen = {(r['qid'], r['k']) for r in done}
    sem, lock, results = asyncio.Semaphore(CONCURRENCY), asyncio.Lock(), list(done)
    logger.info('%d trajectories to run (%d done)', len(jobs), len(done))

    async def one(c, it, qi, qa, j):
        if ('%s|%s|%d' % (c, it.get('take_uid') or it.get('video_id'), qi), j) in seen:
            return
        async with sem:
            try:
                r = await solve(c, it, qi, qa, j)
            except Exception as e:
                logger.exception('solve failed: %s', e)
                return
        async with lock:
            results.append(r)
            json.dump(results, open(out, 'w'), ensure_ascii=False, indent=1)

    await asyncio.gather(*(one(*j) for j in jobs))
    logger.info('DONE %d trajectories', len(results))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2,
                     int(sys.argv[3]) if len(sys.argv) > 3 else 10))
