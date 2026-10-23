"""Stage-II preference pairs, full-trajectory version (the DPO data of run_4b.sh): chosen and rejected are whole
trajectories, because whether a rollout ends correct or wrong is mostly decided before the branch point.
Rollouts = p2_online.py rows (comb1, T .7, eval_n1 loop; full trajectory = prefix + cont).
  chosen    1) comb1's own rollout on the QA: answered, ACC >= .9, used_ref, Flash SCS >= .75 (on-policy, preferred)
            2) else a comb1 SFT trajectory of the QA (data/p1/rwf_comb1.json, data/p1/afill_rwf.json: GT-verified base
               self-samples with rewritten answer blocks; system / user replaced by the eval prompt), best SCS_rw first
  rejected  comb1 rollout on the same QA with a wrong answer (ACC < .5) or no <answer>; no Serper failure
  per QA <= --per_qa pairs; distinct rejected rollouts, alternating wrong / no_answer; chosen cycled.
Loss (train_dpo_p2: length-normalised DPO, --mask_common_prefix scores from the first differing token,
init + frozen ref = comb1, A/B --class_balance). 'tokens' = assistant tokens after the shared system/user prompt.
Usage (from the work dir $TRIPHYS_WORK): python scripts/p2_pairs_full.py --inp 'results/p2_on/w_*[0-9].json' --out data/p2/dpo_p2_full1.json"""
from srvpaths import BASE_MODEL, SCRIPTS, EVAL_LIB  # machine paths, see srvpaths.py
import argparse, asyncio, glob, json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, EVAL_LIB)
from prompts import SYSTEM_PROMPT, build_user_prompt
import p2_pairs as P2

ACC_OK, ACC_BAD, SCS_OK = .9, .5, .75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default='results/p2_on/w_*[0-9].json')
    ap.add_argument('--sft', default='data/p1/rwf_comb1.json,data/p1/afill_rwf.json')
    ap.add_argument('--out', default='data/p2/dpo_p2_full1.json')
    ap.add_argument('--per_qa', type=int, default=2)
    a = ap.parse_args()
    import p1_rft_build as B
    idx = B.train_index()
    rows = [r for f in sorted(glob.glob(a.inp)) for r in json.load(open(f))]
    todo = [r for r in rows if r['ACC'] is None or r['SCS'] is None]
    if todo:
        qa_by_q = {q: v[2] for q, v in idx.items()}

        async def go():
            for r in todo:
                await P2.rescore(r, qa_by_q)
        asyncio.run(go())
    rows = [r for r in rows if r['ACC'] is not None and r['SCS'] is not None and not r.get('serper_failed')]

    def base(q):
        cls, vpath, qa = idx[q]
        return [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [{'type': 'video', 'video': vpath, 'fps': 1.0},
                                             {'type': 'text', 'text': build_user_prompt(q, qa['question_type'])}]}]

    def label(r):
        if r['answered'] and r['ACC'] >= ACC_OK and r['used_ref'] and r['SCS'] >= SCS_OK:
            return 'chosen'
        if not r['answered']:
            return 'no_answer'
        if r['ACC'] < ACC_BAD:
            return 'wrong'
        return 'other'

    sft = defaultdict(list)
    for f in a.sft.split(','):
        for x in json.load(open(f)):
            u = x['messages'][1]['content']
            t = u if isinstance(u, str) else ' '.join(z.get('text', '') for z in u if isinstance(z, dict))
            q = next((q for q in idx if q in t), None)
            if q:
                sft[q].append((x['metadata'].get('SCS_rw', x['metadata'].get('SCS', 0)), x['messages'][2:]))
    own, rej = defaultdict(list), defaultdict(list)
    stat = Counter()
    for r in rows:
        l = label(r)
        stat[r['cls'] + ' ' + l] += 1
        traj = [dict(m) for m in r['prefix']] + r['cont']
        if l == 'chosen':
            own[r['question']].append((r['SCS'], traj))
        elif l in ('wrong', 'no_answer'):
            rej[r['question']].append((l, r['ACC'], traj))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ntok = lambda ms: sum(len(tok(m['content'])['input_ids']) for m in ms if m['role'] == 'assistant')
    pairs = []
    for q in sorted(rej):
        cls = idx[q][0]
        if own[q]:
            ch, src = [t for _, t in sorted(own[q], key=lambda z: -z[0])], 'own'
        elif sft[q]:
            ch, src = [t for _, t in sorted(sft[q], key=lambda z: -z[0])], 'sft'
        else:
            stat[cls + ' rejQA_no_chosen'] += 1
            continue
        w = [z for z in rej[q] if z[0] == 'wrong']
        n = [z for z in rej[q] if z[0] == 'no_answer']
        alt = [z for pair in zip(w, n) for z in pair] + w[len(n):] + n[len(w):]
        for i, (l, acc, rt) in enumerate(alt[:a.per_qa]):
            c = ch[i % len(ch)]
            pairs.append({'type': 'full', 'cls': cls, 'question': q, 'src': src, 'rej': l,
                          'chosen': base(q) + c, 'rejected': base(q) + rt, 'tokens': [ntok(c), ntok(rt)]})
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(pairs, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
    rep = {'rows': len(rows), 'stat': dict(sorted(stat.items())), 'pairs': len(pairs),
           'pairs_by_cls': dict(Counter(p['cls'] for p in pairs)),
           'qas_by_cls': dict(Counter(p['cls'] for p in {p['question']: p for p in pairs}.values())),
           'by_src': dict(Counter(p['cls'] + ' ' + p['src'] for p in pairs)),
           'rejected_kind': dict(Counter(p['cls'] + ' ' + p['rej'] for p in pairs)),
           'mean_tokens': [sum(p['tokens'][0] for p in pairs) / max(1, len(pairs)),
                           sum(p['tokens'][1] for p in pairs) / max(1, len(pairs))]}
    json.dump(rep, open(a.out.replace('.json', '_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == '__main__':
    main()
