"""10/16 teacher forward-solve, full run (OmniAtlas style) on all f10 train QAs x K. Same as teacher_pilot (Gemini,
whole clip as base64 video, real Serper + python tools, never sees the GT, scored by metrics_final) with the fixes from
the pilot: (1) visible reasoning in a <reasoning> block every turn (pilot: 3/185 turns had a think block; a "<think>"
prefill is ignored with video input), renamed to <think> at data build; (2) up to MAX_ROUNDS=8 rounds are generated (pilot: only 57% answered within 6), training
keeps <= 6 rounds; (3) K samples per QA, GT filtering happens at data build time.
Output: one JSON list (resumable), same record layout as teacher_pilot.
Usage (from the work dir $TRIPHYS_WORK): python fc/training/teacher_pilot/teacher_full.py OUT.json [K]"""
import asyncio, json, logging, re, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.realpath(__file__)))
import teacher_pilot as tp
from teacher_pilot import (SYSTEM_PROMPT, build_user_prompt, parse_response, calculate_acc_with_fallback,
                           calculate_scs_with_fallback, query_web_search, video_b64, exec_code, call_api, video_path, E)

logger = logging.getLogger('teacher_full')
CONCURRENCY, MAX_ROUNDS = 24, 8
# Gemini will not write <think> in visible text (pilot 3/185, prefill ignored with video input) but does write a
# <reasoning> block; it is renamed to <think> when the SFT data is built.
TOOL_SPEC = tp.TOOL_SPEC.replace('<think>...</think>', '<reasoning>...</reasoning>').replace(
    'You have at most 6 turns in total, so the final answer must come by the 6th turn.',
    'Be efficient: usually 2-4 turns are enough; give the final answer by the 6th turn.') + (
    '\n- Search only for physical properties, object specifications, standards or constants. Never search for the '
    'question wording, the numbers given in the question, the video or a benchmark/solution: they are not online.')
NUDGE_AT = 5  # from this round on the tool-result message (API only, not part of the trajectory) asks for the answer
REMINDER = ('\n\nStart every reply with <reasoning> and write 4-8 sentences: what you see in the video or what the last '
            'tool result says (quote the value you take and say why), which quantity is still missing, and what you do '
            'next. Close with </reasoning>, then the tool call(s) or the final <answer> block.')


def all_qas():
    out = []
    for cls in 'AB':
        for item in json.load(open('%s/data/f10/%sclass_qa_train.json' % (E, cls.lower()))):
            for qi, qa in enumerate(item['qa_pairs']):
                if qa.get('valid', True):
                    out.append((cls, item, qi, qa))
    return out


def parse_calls(content):
    calls = []
    for m in re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        raw = m.group(1).strip()
        try:
            try:
                tc = json.loads(raw, strict=False)
            except Exception:
                tc = json.loads(re.sub(r'\]\s*$', '', raw), strict=False)
            calls.append((tc.get('name', ''), tc.get('arguments', {}) or {}))
        except Exception:
            calls.append(('unparsable', {}))
    return calls


async def solve(cls, item, qi, qa, k):
    vpath = video_path(item)
    v64 = await asyncio.to_thread(video_b64, vpath)
    user_text = build_user_prompt(qa['question'], qa['question_type'])
    api_msgs = [{'role': 'system', 'content': SYSTEM_PROMPT + TOOL_SPEC},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:video/mp4;base64,' + v64}},
                                             {'type': 'text', 'text': user_text + REMINDER}]}]
    turns, tool_log, usage, t0, no_think = [], [], [], time.time(), 0
    for rnd in range(MAX_ROUNDS):
        content, u = await call_api(api_msgs)
        usage.append(u)
        if not content:
            break
        content = re.sub(r'^\s*\w{1,4}\s*(?=<)', '', content).strip()
        # Gemini often leaves <reasoning> unclosed before a tool call: the reasoning is the text before the call/answer
        pre = content.split('</reasoning>')[0] if '</reasoning>' in content else re.split(r'<tool_call>|<answer>', content)[0]
        if len(pre.replace('<reasoning>', '').strip()) < 60:
            no_think += 1
        tail = content.split('</reasoning>')[-1]
        if re.search(r'<answer>.*?</answer>', tail, re.DOTALL) and '<tool_call>' not in tail.split('<answer>')[0]:
            turns.append({'role': 'assistant', 'content': content})
            break
        calls = list(re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL))
        if not calls:
            turns.append({'role': 'assistant', 'content': content})
            break
        content = content[:calls[-1].end()].strip()
        turns.append({'role': 'assistant', 'content': content})
        api_msgs.append({'role': 'assistant', 'content': content})
        results = []
        for name, args in parse_calls(content):
            if name == 'web_search':
                res = await asyncio.to_thread(query_web_search, args.get('query', '') or args.get('q', ''), 'teacher')
            elif name == 'code_executor':
                res = json.dumps({'result': await asyncio.to_thread(exec_code, args.get('code', ''))})
            else:
                res = json.dumps({'result': "Tool '%s' unavailable." % name})
            tool_log.append({'round': rnd, 'tool': name, 'args': args, 'result': json.loads(res).get('result', '')})
            turns.append({'role': 'tool', 'content': res})
            results.append(res)
        msg = '\n'.join('<tool_response>\n%s\n</tool_response>' % r for r in results)
        if rnd + 1 >= NUDGE_AT:
            msg += '\n(%d turns used: give the final <answer> block in your next reply.)' % (rnd + 1)
        api_msgs.append({'role': 'user', 'content': msg})
    final = turns[-1]['content'] if turns and turns[-1]['role'] == 'assistant' else ''
    # parse_response only looks for <answer> after </think>; the reasoning here is in <reasoning>, so parse the tail
    parsed = parse_response('<think>\n</think>\n' + final.split('</reasoning>')[-1])
    rd = {'prediction': parsed.prediction, 'reasoning_chain': parsed.reasoning_chain, 'ground_truth': qa['answer'],
          'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
          'question': qa['question'], 'question_type': qa['question_type'],
          'needs_llm_eval': parsed.needs_llm_eval, 'raw_response': final.split('</reasoning>')[-1]}
    acc = await calculate_acc_with_fallback(rd)
    scs_res = await calculate_scs_with_fallback(rd)
    scs = scs_res.get('overall', scs_res.get('SCS', 0.0)) if isinstance(scs_res, dict) else float(scs_res or 0)
    uid = item.get('take_uid') or item.get('video_id')
    n_asst = sum(t['role'] == 'assistant' for t in turns)
    logger.info('%s %s k%d ACC %.2f SCS %.2f | %d rounds %d tools | %s', cls, uid[:8], k, float(acc), scs, n_asst,
                len(tool_log), str(parsed.prediction)[:60])
    return {'cls': cls, 'qid': '%s|%s|%d' % (cls, uid, qi), 'k': k, 'question_type': qa['question_type'],
            'difficulty': qa.get('difficulty'), 'question': qa['question'], 'ground_truth': qa['answer'],
            'answer_range': qa.get('answer_range'), 'video': vpath, 'user_text': user_text, 'turns': turns,
            'tool_log': tool_log, 'usage': usage, 'prediction': parsed.prediction, 'ACC': acc, 'SCS': scs,
            'rounds': n_asst, 'turns_without_think': no_think, 'seconds': round(time.time() - t0, 1)}


async def main(out, k):
    import os
    jobs = [(c, it, qi, qa, j) for j in range(k) for (c, it, qi, qa) in all_qas()]
    if os.environ.get("LIMIT"):
        n = int(os.environ["LIMIT"]); jobs = jobs[:n // 2] + [j for j in jobs if j[0] == "B"][:n - n // 2]
    done = json.load(open(out)) if Path(out).exists() else []
    seen = {(r['qid'], r['k']) for r in done}
    sem, lock, results = asyncio.Semaphore(CONCURRENCY), asyncio.Lock(), list(done)
    logger.info('%d trajectories total, %d done', len(jobs), len(done))

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
            if len(results) % 10 == 0 or len(results) == len(jobs):
                json.dump(results, open(out, 'w'), ensure_ascii=False)

    await asyncio.gather(*(one(*j) for j in jobs))
    json.dump(results, open(out, 'w'), ensure_ascii=False)
    by = defaultdict(list)
    for r in results:
        by[r['qid']].append(float(r['ACC']) >= .9)
    for c in 'AB':
        q = [v for key, v in by.items() if key[0] == c]
        rs = [r for r in results if r['cls'] == c]
        logger.info('%s: %d traj, ACC %.3f, pass@k %.3f (%d QAs)', c, len(rs),
                    sum(float(r['ACC']) >= .9 for r in rs) / max(1, len(rs)), sum(any(v) for v in q) / max(1, len(q)), len(q))
    logger.info('TEACHER DONE %d trajectories', len(results))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4))
