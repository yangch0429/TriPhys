# Provenance (TriPhysBench release): fix_code/evaluation/fix10/type_check.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""user 10/13: the final answer decides the type - one value -> TYPE_A, an interval -> TYPE_B (TYPE_C yes/no, TYPE_D comparison
are listed separately). Checks every QA of the dataset, the repaired version where one exists.
Usage: python type_check.py out.json [rec1.json ...]   (later record files override earlier ones)"""
import sys, json, re, os
from collections import Counter
ARGV = list(sys.argv)
sys.argv = [sys.argv[0], 'unres151_ids.json']
import fix10 as X  # noqa: E402,F401
from fix10 import R  # noqa: E402

NUM = r'[-+]?\d+(?:\.\d+)?(?:\s*[x×]\s*10\^?[-+]?\d+|[eE][-+]?\d+)?'
INTERVAL = re.compile(r'(?:%s)\s*[^\d\s,;()]{0,12}\s*(?:-|–|—|to|~)\s*(?:%s)|between\s+%s.{0,20}?\s+and\s+%s|\[\s*%s\s*,\s*%s\s*\]'
                      % (NUM, NUM, NUM, NUM, NUM, NUM), re.I)


def form(qa):
    a = str(qa.get('answer') or '')
    rng = qa.get('answer_range')
    ok_rng = isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng) and rng[0] < rng[1]
    a1 = re.sub(r'\([^)]*\)', '', a)  # a parenthesised conversion / remark is not the answer form
    iv = bool(INTERVAL.search(a1.replace('−', '-')))
    if iv or ok_rng:
        return 'interval', iv, ok_rng
    return 'single', iv, ok_rng


recs = {}
for f in ARGV[2:]:
    if os.path.exists(f):
        for k, r in json.load(open(f)).items():
            if isinstance(r, dict) and r.get('status') in ('fixed', 'given') and r.get('qa'):
                recs[k] = r
C, bad = Counter(), {}
for it in R.load():
    k = it['id']
    qa = recs[k]['qa'] if k in recs else it['qa']
    t = qa.get('question_type')
    f, iv, ok = form(qa)
    C[(it['cls'], t, f)] += 1
    if (t == 'TYPE_A' and f == 'interval') or (t == 'TYPE_B' and f == 'single') or (t == 'TYPE_B' and iv != ok):
        bad[k] = dict(type=t, form=f, interval_in_text=iv, answer_range_ok=ok, answer=qa.get('answer'),
                      answer_range=qa.get('answer_range'), repaired=k in recs, split=it['split'])
for x in sorted(C):
    print(x, C[x])
print('mismatch', len(bad), Counter((v['type'], v['form'], v['interval_in_text'], v['answer_range_ok']) for v in bad.values()))
print('by split', Counter((k.split('|')[0], v['split']) for k, v in bad.items()))
json.dump(bad, open(ARGV[1], 'w'), ensure_ascii=False, indent=1)
