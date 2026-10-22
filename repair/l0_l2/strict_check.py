# Provenance (TriPhysBench release): fix_code/_v8_work/strict_check.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Strict deterministic recheck of the written benchmark, beyond final_recheck.py (whose code check accepts an output
sharing ANY number with the answer):
  range   answer_range [lo, hi]: lo < hi, and lo and hi both printed by the code (2 %, unit scales)
  value   answer without range (not TYPE_C): its first number printed by the code (2 %, unit scales)
  hidden  the number of every hidden value (after '=') used in a code block (2 %, unit scales; skipped without code)
Output: strict_check.json {qid: [problems]}
Usage: python strict_check.py"""
import json, re
from collections import Counter
from fixlib import WORK, load, iter_qas, sources_of, run_code, text_nums, consistent


def nums_after_eq(h):
    m = re.search(r'=\s*(.*)', h)
    return [x for x in text_nums(m.group(1) if m else h) if x > 0][:1]


def problems(qa):
    codes = [s.get('code') or '' for s in sources_of(qa, 'code_executor')]
    out = '\n'.join(run_code(c)['out'] for c in codes)
    p = []
    ar = qa.get('answer_range')
    if isinstance(ar, list) and len(ar) == 2 and all(isinstance(x, (int, float)) for x in ar):
        lo, hi = ar
        if not lo < hi:
            p.append('range: answer_range %s not increasing' % ar)
        if codes:
            miss = [x for x in (lo, hi) if x and not consistent(out, [abs(x)])]
            if miss:
                p.append('range: %s not printed by the code (answer %r | out %r)' % (miss, qa.get('answer'), out[-300:]))
    elif qa.get('question_type') != 'TYPE_C' and codes:
        a = [x for x in text_nums(qa.get('answer')) if x > 0][:1]
        if a and not consistent(out, a):
            p.append('value: %s not printed by the code (answer %r | out %r)' % (a, qa.get('answer'), out[-300:]))
    if codes:
        code_txt = '\n'.join(codes)
        for h in qa.get('hidden_cv_chosen') or []:
            n = nums_after_eq(h)
            if n and not consistent(code_txt, n):
                p.append('hidden: %r not used in the code' % h)
    return p


def main():
    res = {}
    for split in ('train', 'test'):
        for cls in 'ab':
            for qid, _, qa in iter_qas(load(cls, split), cls, split):
                res[qid] = problems(qa)
    bad = {q: p for q, p in res.items() if p}
    json.dump(bad, open(WORK + '/strict_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('QAs', len(res), '| with problems', len(bad))
    print(Counter(x.split(':')[0] for p in bad.values() for x in p))


if __name__ == '__main__':
    main()
