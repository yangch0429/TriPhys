"""Stage-II DPO data: branch continuations from the Stage-I policy's own trajectory. The functions here are used by
p2_online.py; run standalone, this script branches saved rollouts.
  source   on-policy rollouts of the Stage-I model (comb1_1e4) on the train QAs (standalone: results/<tag>_r*, --tags;
           eval_n1 loop, T=0.7)
  branch   first assistant turn AFTER the first web_search result that uses acquired values: its first code_executor
           call, or the answer turn when no code is called. Prefix = every turn before it (observations, queries,
           returned results); the branch turn's own think (value choice) belongs to the continuation.
  prefix   kept only if a returned search result states a reference value of the QA (number of a web_search source
           key_information, unit conversions allowed) and no Serper failure; <= --per_qa distinct prefixes per QA.
  sample   K continuations per prefix with the same model / T / loop as eval_n1 generate(): rounds continue up to the
           eval limit of 6 assistant rounds in total, identical tool results in consecutive rounds stop the loop.
  score    ACC (GT verifier, same judge as eval) + Flash SCS for answered rows; used_ref = a reference number appears
           in the continuation's code / answer text. Pairing is done by p2_pairs.py.
Output: --out json list of {question, cls, j (prefix id), s, prefix, cont, ACC, SCS, answered, used_ref, ...}
Usage (from the work dir $TRIPHYS_WORK): GEN_TEMP=0.7 python scripts/p2_branch.py --model M --shard i --n_shards n --out results/p2_br/shard_i.json
                [--limit N] [--tags p2_ss]"""
from srvpaths import N1, SCRIPTS, EVAL_LIB  # machine paths, see srvpaths.py
import argparse, asyncio, json, os, re, sys
from collections import defaultdict
sys.path.insert(0, N1)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, EVAL_LIB)
import types
_src = open(N1 + '/eval_n1_sample.py').read().rsplit('asyncio.run(main())', 1)[0]
S = types.ModuleType('eval_n1_sample')
S.__file__ = N1 + '/eval_n1_sample.py'
exec(compile(_src, S.__file__, 'exec'), S.__dict__)
import p1_rft_build as B
from prompts import SYSTEM_PROMPT, build_user_prompt
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback

MAX_ROUNDS = 6
JUDGE_S = 180   # cap per judge call so a judge outage cannot idle the GPU
NUM = r'(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?'
TRIV = {0, 1, 2, 3, 4, 5, 10, 100, 1000, 0.5, 9.8, 9.81}


def nums(s):
    out = []
    for m in re.findall(NUM, str(s or '')):
        try:
            out.append(float(m.replace(',', '')))
        except ValueError:
            pass
    return [x for x in out if abs(x) not in TRIV]


def expand(xs):   # reference numbers under common unit conversion factors
    out = set()
    for y in xs:
        out.add(y)
        for f in (1000, 100, 60, 3600, 3.6, 2.20462, 0.453592, 2.54, 0.3048, 12, 0.0283495, 28.3495, 1e6, 1e-6, 1e3):
            out.add(y * f)
            out.add(y / f)
    return out


def near(x, pool, tol=.02):
    return any(abs(x - y) <= tol * max(abs(x), abs(y), 1e-12) for y in pool)


def refs(qa):
    return expand({x for s in qa.get('sources', []) if s.get('type') == 'web_search' for x in nums(s.get('key_information'))})


def branch(turns):
    """index of the branch assistant turn in turns, or None"""
    seen_search = False
    for i, t in enumerate(turns):
        if t['role'] == 'tool' and seen_search:
            continue
        if t['role'] != 'assistant':
            continue
        c = str(t['content'])
        if seen_search and ('<function=code_executor>' in c or re.search(r'<answer>.*?</answer>', c, re.S)):
            return i
        if '<function=web_search>' in c:
            seen_search = True
    return None


def jobs(tags, per_qa):
    idx = B.train_index()
    by = defaultdict(list)
    for x in B.load_rows(tags):
        q = x['question']
        t = x.get('turns') or []
        if q not in idx or x.get('serper_failed') or not t:
            continue
        b = branch(t)
        if b is None:
            continue
        pre = [{'role': y['role'], 'content': y['content']} for y in t[:b]]
        if any('__SERPER_FAILED__' in y['content'] for y in pre if y['role'] == 'tool'):
            continue
        R = refs(idx[q][2])
        res = ' '.join(y['content'] for y in pre if y['role'] == 'tool')
        if not R or not any(near(v, R) for v in nums(res)):
            continue
        if any(p == pre for p in by[q]):
            continue
        by[q].append(pre)
    return [(q, idx[q], pre) for q in sorted(by) for pre in by[q][:per_qa]]


def continue_from(ev, msgs, vi, vk, n_done):
    """eval_n1 generate() loop from a prefix with n_done assistant rounds already taken"""
    new, prev_sig, content = [], None, ''
    for _ in range(MAX_ROUNDS - n_done):
        content, _n = ev._run_round(msgs, vi, vk, S.MAX_TOKENS)
        new.append({'role': 'assistant', 'content': content})
        if re.search(r'<answer>.*?</answer>', content, re.DOTALL):
            break
        calls = S.parse_tool_calls(content)
        if not calls:
            break
        msgs.append({'role': 'assistant', 'content': content})
        rr = []
        for name, args in calls:
            if name == 'code_executor':
                r = json.dumps({'result': ev._exec_code(args.get('code', ''))})
            elif name == 'web_search':
                r = S.query_web_search(args.get('query', '') or args.get('q', ''), 'p2br')
            else:
                r = json.dumps({'result': "Tool '%s' unavailable." % name})
            new.append({'role': 'tool', 'content': r})
            msgs.append({'role': 'tool', 'content': r})
            rr.append(r)
        if str(rr) == prev_sig:
            break
        prev_sig = str(rr)
    return content, new


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--per_qa', type=int, default=2)
    ap.add_argument('--tags', default='p2_ss')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    J = jobs(a.tags.split(','), a.per_qa)
    S.logger.info('branch prefixes %d on %d QAs (A %d)', len(J), len({j[0] for j in J}), sum(j[1][0] == 'A' for j in J))
    J = J[a.shard::a.n_shards]
    if a.limit:
        J = J[:a.limit]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r['question'], r['j'], r['s']) for r in res}
    ev = S.SFTEvaluator(a.model, 0)
    for j, (q, (cls, vpath, qa), pre) in enumerate(J):
        vi, vk = ev._get_video(vpath)
        base = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [{'type': 'video', 'video': vpath, 'fps': 1.0},
                                             {'type': 'text', 'text': build_user_prompt(q, qa['question_type'])}]}]
        n_done = sum(t['role'] == 'assistant' for t in pre)
        R = refs(qa)
        for s in range(a.K):
            if (q, j, s) in done:
                continue
            content, new = continue_from(ev, base + [dict(m) for m in pre], vi, vk, n_done)
            p = parse_response(content)
            rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': qa['answer'],
                  'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
                  'question': q, 'question_type': qa['question_type'], 'needs_llm_eval': p.needs_llm_eval,
                  'raw_response': content}
            answered = bool(re.search(r'<answer>.*?</answer>', content, re.DOTALL))
            acc, scs = 0.0, 0.0   # set to None if a judge call times out; p2_pairs.py rescores those rows
            try:
                if p.prediction:
                    acc = float(await asyncio.wait_for(calculate_acc_with_fallback(rd), JUDGE_S))
            except asyncio.TimeoutError:
                acc = None
            try:
                if answered:
                    sr = await asyncio.wait_for(calculate_scs_with_fallback(rd), JUDGE_S)
                    scs = float(sr.get('overall', 0.0)) if isinstance(sr, dict) else float(sr or 0)
            except asyncio.TimeoutError:
                scs = None
            txt = ' '.join(' '.join(re.findall(r'<parameter=code>(.*?)</parameter>', t['content'], re.S))
                           for t in new if t['role'] == 'assistant') + ' ' + str(p.reasoning_chain or '')
            used = any(near(v, R) for v in nums(txt))
            res.append({'question': q, 'cls': cls, 'j': j, 's': s, 'video': vpath, 'question_type': qa['question_type'],
                        'prefix': pre, 'cont': new, 'ACC': acc, 'SCS': scs, 'answered': answered, 'used_ref': used,
                        'prediction': str(p.prediction or ''), 'ground_truth': qa['answer'],
                        'serper_failed': any('__SERPER_FAILED__' in t['content'] for t in new if t['role'] == 'tool')})
            S.logger.info('branch %s j%d s%d ACC %s SCS %s ans %s ref %s | %s', cls, j, s, acc, scs, answered, used,
                          str(p.prediction)[:50])
            json.dump(res, open(a.out, 'w'), ensure_ascii=False)
    S.logger.info('BRANCH SHARD DONE %d', len(res))


if __name__ == '__main__':
    asyncio.run(main())
