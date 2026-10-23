"""Stage-II preference pairs from the p2_branch.py continuations (shared comb1 prefix, K continuations). Its rescoring
helper is also used by p2_pairs_full.py, which builds the pairs used for DPO.
  chosen    answered, ACC >= .9, a reference value of the QA appears in its code / answer (used_ref), Flash SCS >= .75
  rejected  wrong answer (ACC < .5) or no <answer> within the remaining rounds (label kept in 'rej')
  neither may contain a Serper failure; ACC/SCS = None (judge timed out during sampling) are rescored here first.
  A pair needs both from the SAME prefix; per QA <= 2 pairs, taken from different prefixes first; the best-SCS
  chosen and, alternately, a wrong / no-answer rejected. A/B balance = --class_balance loss weights in train_dpo_p2.
Message lists = [system, user(video)] + prefix + continuation, so chosen and rejected share the prefix and
train_dpo_p2 --mask_common_prefix scores only the model tokens after the branch point.
'tokens' = assistant tokens of the continuation (sets beta_eff for --length_norm).
Usage (from the work dir $TRIPHYS_WORK): python scripts/p2_pairs.py --inp 'results/p2_br/*.json' --out data/p2/dpo_p2.json"""
from srvpaths import BASE_MODEL, SCRIPTS, EVAL_LIB  # machine paths, see srvpaths.py
import argparse, asyncio, glob, json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, EVAL_LIB)
from prompts import SYSTEM_PROMPT, build_user_prompt
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback

ACC_OK, ACC_BAD, SCS_OK = .9, .5, .75


async def rescore(r, qa_by_q):
    qa = qa_by_q[r['question']]
    content = r['cont'][-1]['content'] if r['cont'] else ''
    p = parse_response(content)
    rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': qa['answer'],
          'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
          'question': r['question'], 'question_type': qa['question_type'], 'needs_llm_eval': p.needs_llm_eval,
          'raw_response': content}
    try:
        if r['ACC'] is None:
            r['ACC'] = float(await asyncio.wait_for(calculate_acc_with_fallback(rd), 300)) if p.prediction else 0.0
        if r['SCS'] is None:
            sr = await asyncio.wait_for(calculate_scs_with_fallback(rd), 300)
            r['SCS'] = float(sr.get('overall', 0.0)) if isinstance(sr, dict) else float(sr or 0)
    except asyncio.TimeoutError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default='results/p2_br/*.json')
    ap.add_argument('--out', default='data/p2/dpo_p2.json')
    ap.add_argument('--per_qa', type=int, default=2)
    a = ap.parse_args()
    rows = [r for f in sorted(glob.glob(a.inp)) for r in json.load(open(f))]
    import p1_rft_build as B
    qa_by_q = {q: v[2] for q, v in B.train_index().items()}
    todo = [r for r in rows if r['ACC'] is None or r['SCS'] is None]
    if todo:
        async def go():
            for r in todo:
                await rescore(r, qa_by_q)
        asyncio.run(go())
    rows = [r for r in rows if r['ACC'] is not None and r['SCS'] is not None and not r.get('serper_failed')]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ntok = lambda ms: sum(len(tok(m['content'])['input_ids']) for m in ms if m['role'] == 'assistant')

    def label(r):
        if r['answered'] and r['ACC'] >= ACC_OK and r['used_ref'] and r['SCS'] >= SCS_OK:
            return 'chosen'
        if not r['answered']:
            return 'no_answer'
        if r['ACC'] < ACC_BAD:
            return 'wrong'
        return 'other'

    by_pre = defaultdict(list)
    for r in rows:
        by_pre[(r['question'], r['j'])].append(r)
    stat = Counter()
    by_q = defaultdict(list)
    for (q, j), rs in sorted(by_pre.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        labs = [label(r) for r in rs]
        cls = rs[0]['cls']
        stat[cls + ' prefixes'] += 1
        for l in labs:
            stat[cls + ' ' + l] += 1
        ch = sorted([r for r, l in zip(rs, labs) if l == 'chosen'], key=lambda r: -r['SCS'])
        rj = [r for r, l in zip(rs, labs) if l in ('wrong', 'no_answer')]
        if not ch:
            stat[cls + ' prefix_no_chosen'] += 1
        if not rj:
            stat[cls + ' prefix_no_rejected'] += 1
        if ch and rj:
            rj = sorted(rj, key=lambda r: r['s'])
            by_q[q].append([(c, rj[i % len(rj)]) for i, c in enumerate(ch[:len(rj)])] or [(ch[0], rj[0])])
    pairs = []
    for q, per_prefix in by_q.items():
        take = []
        depth = 0
        while len(take) < a.per_qa and any(depth < len(pp) for pp in per_prefix):
            for pp in per_prefix:
                if depth < len(pp) and len(take) < a.per_qa:
                    take.append(pp[depth])
            depth += 1
        for c, r in take:
            base = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': [{'type': 'video', 'video': c['video'], 'fps': 1.0},
                                                 {'type': 'text', 'text': build_user_prompt(q, c['question_type'])}]}]
            pre = [dict(m) for m in c['prefix']]
            cm, rm = base + pre + c['cont'], base + pre + r['cont']
            pairs.append({'type': 'branch', 'cls': c['cls'], 'question': q, 'j': c['j'], 'rej': label(r),
                          'chosen': cm, 'rejected': rm, 'tokens': [ntok(c['cont']), ntok(r['cont'])],
                          'acc': [c['ACC'], r['ACC']], 'scs': [c['SCS'], r['SCS']]})
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(pairs, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
    rep = {'rows': len(rows), 'stat': dict(sorted(stat.items())), 'pairs': len(pairs),
           'pairs_by_cls': dict(Counter(p['cls'] for p in pairs)), 'qas_by_cls': dict(Counter(p['cls'] for p in {p['question']: p for p in pairs}.values())),
           'rejected_kind': dict(Counter(p['cls'] + ' ' + p['rej'] for p in pairs)),
           'mean_cont_tokens': [sum(p['tokens'][0] for p in pairs) / max(1, len(pairs)),
                                sum(p['tokens'][1] for p in pairs) / max(1, len(pairs))]}
    json.dump(rep, open(a.out.replace('.json', '_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == '__main__':
    main()
