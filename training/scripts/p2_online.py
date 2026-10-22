"""10/18 13:15 Phase2 DPO data, online version of p2_selfsample.sh + p2_branch.py (user: speed up the DPO stage).
Same method as p2_branch.py (prefix-branch, user-approved); only the sampling schedule changes:
  per train QA, up to --attempts comb1 rollouts (eval_n1 loop, T .7) are generated only UP TO the branch turn
  (= first assistant turn after a web_search result that calls code_executor or answers; p2_branch.branch()).
  A rollout that computes / answers before any search, stops calling tools, or repeats tool results has no prefix
  and is dropped as soon as that is known. A prefix is kept if a returned result states a QA reference number and no
  Serper failure; at most --per_qa prefixes per QA (attempts stop early once reached).
  Continuations per prefix (K total): the branch turn that ended the prefix rollout is itself a policy sample at the
  branch point -> continuation s=0, run on to the end; the other K-1 start with ONE batched generate
  (num_return_sequences, shared prompt / video) for their branch turn, then each continues alone.
  Scoring = p2_branch (ACC rule/LLM judge, Flash SCS, used_ref; judge calls capped, None on timeout -> p2_pairs rescores).
Output rows = p2_branch rows (+ attempt); per-QA attempt log in OUT_attempts.json. Resumable per QA.
Usage: GEN_TEMP=0.7 python scripts/p2_online.py --model M --out results/p2_on/w_i.json --claim_dir results/p2_on/claims
       (or --shard i --n_shards n for a fixed split)"""
from srvpaths import SCRIPTS  # machine paths, see srvpaths.py
import argparse, asyncio, json, os, re, sys
sys.path.insert(0, SCRIPTS)
import torch
import p2_branch as P
S, B = P.S, P.B
from prompts import SYSTEM_PROMPT, build_user_prompt
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback

ANS = re.compile(r'<answer>.*?</answer>', re.DOTALL)


def run_tools(ev, content, msgs, new):
    calls = S.parse_tool_calls(content)
    rr = []
    for name, args in calls:
        if name == 'code_executor':
            r = json.dumps({'result': ev._exec_code(args.get('code', ''))})
        elif name == 'web_search':
            r = S.query_web_search(args.get('query', '') or args.get('q', ''), 'p2on')
        else:
            r = json.dumps({'result': "Tool '%s' unavailable." % name})
        new.append({'role': 'tool', 'content': r})
        msgs.append({'role': 'tool', 'content': r})
        rr.append(r)
    return str(rr)


def to_branch(ev, base, vi, vk):
    """one rollout up to the branch turn -> (prefix turns, branch content, why)"""
    msgs, turns, seen, prev = [dict(m) for m in base], [], False, None
    for _ in range(P.MAX_ROUNDS):
        content, _n = ev._run_round(msgs, vi, vk, S.MAX_TOKENS)
        if seen and ('<function=code_executor>' in content or ANS.search(content)):
            return turns, content, 'ok'
        if ANS.search(content):
            return None, None, 'answer_before_search'
        if not S.parse_tool_calls(content):
            return None, None, 'no_tool_call'
        if '<function=web_search>' in content:
            seen = True
        msgs.append({'role': 'assistant', 'content': content})
        turns.append({'role': 'assistant', 'content': content})
        sig = run_tools(ev, content, msgs, turns)
        if sig == prev:
            return None, None, 'repeat'
        prev = sig
    return None, None, 'round_limit'


def first_rounds(ev, msgs, vi, vk, n):
    """n sampled branch turns from the same prompt in one generate call"""
    text = ev.processor.apply_chat_template(msgs, tools=S.TOOLS, tokenize=False, add_generation_prompt=True)
    inputs = ev.processor(text=[text], images=None, videos=vi, padding=True, return_tensors='pt', **vk).to(ev.device)
    L = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = ev.model.generate(**inputs, max_new_tokens=S.MAX_TOKENS, eos_token_id=ev._im_end_id,
                                num_return_sequences=n, **S.GEN_PARAMS)
    res = []
    for row in out[:, L:].tolist():
        if ev._im_end_id in row:
            row = row[:row.index(ev._im_end_id)]
        res.append(ev.processor.tokenizer.decode(row, skip_special_tokens=False).replace('<|im_end|>', '').strip())
    return res


def finish(ev, msgs, vi, vk, n_done, first):
    """p2_branch.continue_from with the branch turn already sampled"""
    new, prev, content = [], None, first
    for k in range(P.MAX_ROUNDS - n_done):
        if k:
            content, _n = ev._run_round(msgs, vi, vk, S.MAX_TOKENS)
        new.append({'role': 'assistant', 'content': content})
        if ANS.search(content) or not S.parse_tool_calls(content):
            break
        msgs.append({'role': 'assistant', 'content': content})
        sig = run_tools(ev, content, msgs, new)
        if sig == prev:
            break
        prev = sig
    return content, new


async def score(q, qa, content, new, R):
    p = parse_response(content)
    rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': qa['answer'],
          'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
          'question': q, 'question_type': qa['question_type'], 'needs_llm_eval': p.needs_llm_eval,
          'raw_response': content}
    answered = bool(ANS.search(content))
    acc, scs = 0.0, 0.0
    try:
        if p.prediction:
            acc = float(await asyncio.wait_for(calculate_acc_with_fallback(rd), P.JUDGE_S))
    except asyncio.TimeoutError:
        acc = None
    try:
        if answered:
            sr = await asyncio.wait_for(calculate_scs_with_fallback(rd), P.JUDGE_S)
            scs = float(sr.get('overall', 0.0)) if isinstance(sr, dict) else float(sr or 0)
    except asyncio.TimeoutError:
        scs = None
    txt = ' '.join(' '.join(re.findall(r'<parameter=code>(.*?)</parameter>', t['content'], re.S))
                   for t in new if t['role'] == 'assistant') + ' ' + str(p.reasoning_chain or '')
    used = any(P.near(v, R) for v in P.nums(txt))
    return p, acc, scs, answered, used


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--per_qa', type=int, default=2)
    ap.add_argument('--attempts', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--claim_dir', default='', help='work queue: QAs are claimed via lock files (workers can be added)')
    a = ap.parse_args()
    idx = B.train_index()
    Q = sorted(idx)[a.shard::a.n_shards]
    if a.limit:
        Q = Q[:a.limit]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    af = a.out.replace('.json', '_attempts.json')
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    att = json.load(open(af)) if os.path.exists(af) else []
    done = {x['question'] for x in att}
    res = [r for r in res if r['question'] in done]
    ev = S.SFTEvaluator(a.model, 0)
    for q in Q:
        if q in done:
            continue
        if a.claim_dir:
            import hashlib
            c = os.path.join(a.claim_dir, hashlib.md5(q.encode()).hexdigest())
            try:
                fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, a.out.encode())
                os.close(fd)
            except FileExistsError:
                if open(c).read() != a.out:     # claimed by another worker (own claims are resumed)
                    continue
        cls, vpath, qa = idx[q]
        vi, vk = ev._get_video(vpath)
        base = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [{'type': 'video', 'video': vpath, 'fps': 1.0},
                                             {'type': 'text', 'text': build_user_prompt(q, qa['question_type'])}]}]
        R = P.refs(qa)
        whys, j = [], 0
        for t in range(a.attempts):
            if j >= a.per_qa:
                break
            pre, br, why = to_branch(ev, base, vi, vk)
            if pre is not None:
                if any('__SERPER_FAILED__' in y['content'] for y in pre if y['role'] == 'tool'):
                    why = 'serper_failed'
                elif not R or not any(P.near(v, R) for v in P.nums(' '.join(y['content'] for y in pre if y['role'] == 'tool'))):
                    why = 'no_ref_in_prefix'
            whys.append(why)
            S.logger.info('online %s attempt %d: %s', cls, t, why)
            if why != 'ok':
                continue
            n_done = sum(y['role'] == 'assistant' for y in pre)
            firsts = [br] + (first_rounds(ev, base + [dict(m) for m in pre], vi, vk, a.K - 1) if a.K > 1 else [])
            for s, fr in enumerate(firsts):
                content, new = finish(ev, base + [dict(m) for m in pre], vi, vk, n_done, fr)
                p, acc, scs, answered, used = await score(q, qa, content, new, R)
                res.append({'question': q, 'cls': cls, 'j': j, 's': s, 'attempt': t, 'video': vpath,
                            'question_type': qa['question_type'], 'prefix': pre, 'cont': new, 'ACC': acc, 'SCS': scs,
                            'answered': answered, 'used_ref': used, 'prediction': str(p.prediction or ''),
                            'ground_truth': qa['answer'],
                            'serper_failed': any('__SERPER_FAILED__' in y['content'] for y in new if y['role'] == 'tool')})
                S.logger.info('branch %s j%d s%d ACC %s SCS %s ans %s ref %s | %s', cls, j, s, acc, scs, answered,
                              used, str(p.prediction)[:50])
            j += 1
        att.append({'question': q, 'cls': cls, 'attempts': whys, 'prefixes': j})
        json.dump(res, open(a.out, 'w'), ensure_ascii=False)
        json.dump(att, open(af, 'w'), ensure_ascii=False)
        S.logger.info('QA done %s prefixes %d %s | peak mem %.1f GB', cls, j, whys,
                      torch.cuda.max_memory_allocated() / 2**30)
    S.logger.info('ONLINE SHARD DONE %d rows %d QAs', len(res), len(att))


if __name__ == '__main__':
    asyncio.run(main())
