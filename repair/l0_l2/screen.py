# Provenance (TriPhysBench release): fix_code/_v8_work/screen.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Post-screen of the ok repairs of L1 / L2 with the shared text checks (l1_rewrite.text_checks) and the manual
review verdicts (review_log.json): prints / writes the qids to redo.
Usage: python screen.py L1|L2   -> redo_<level>.json"""
import json, os, sys
from fixlib import WORK, load, iter_qas
from l1_rewrite import text_checks

level = sys.argv[1]
res = json.load(open(WORK + '/%s_results.json' % level.lower(), encoding='utf-8'))
log = json.load(open(WORK + '/review_log.json', encoding='utf-8')) if os.path.exists(WORK + '/review_log.json') else {}
old = {}
for split in ('train', 'test'):
    for cls in 'ab':
        for qid, vid, qa in iter_qas(load(cls, split, 'L0'), cls, split):
            old[qid] = qa
redo = {}
for qid, r in res.items():
    if r['status'] != 'ok':
        continue
    o, n = old[qid], r['qa']
    errs = text_checks(o, n)
    errs += ['%s changed' % k for k in ('question_type', 'difficulty') if o.get(k) != n.get(k)]
    v = log.get(qid)
    if v and v['level'] == level and v['verdict'] == 'reject':
        errs.append('review: ' + v['notes'])
    if errs:
        redo[qid] = errs
        print(qid, errs)
json.dump(redo, open(WORK + '/redo_%s.json' % level, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(redo), 'to redo')
