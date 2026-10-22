# Provenance (TriPhysBench release): fix_code/_v8_work/review_dump.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Compact old -> new diff of repaired QAs for the one-by-one review (review_log.json).
CONS (cons_results.json) is diffed against the current version without consistency fixes (L0 <- accepted repair).
Usage: python review_dump.py L1|L2|L1E|CONS [qid ...]  -> review_dump_<level>.txt"""
import difflib, json, sys
from fixlib import WORK, load, iter_qas, get_steps, sources_of

level = sys.argv[1]
res = json.load(open(WORK + '/%s_results.json' % level.lower(), encoding='utf-8'))
old = {}
for split in ('train', 'test'):
    for cls in 'ab':
        for qid, vid, qa in iter_qas(load(cls, split, 'L0'), cls, split):
            old[qid] = qa
if level == 'CONS':
    from current import current
    old = {q: qa for q, (qa, lv) in current(pending=False).items()}
todo = sys.argv[2:] or [q for q, r in res.items() if r['status'] == 'ok']
out = []
for qid in todo:
    o, n = old[qid], res[qid]['qa']
    out.append('=' * 100 + '\n%s  [%s %s]' % (qid, o['question_type'], o['difficulty']))
    if n['question'] != o['question']:
        out.append('Q-OLD: ' + o['question'])
    out.append('Q: ' + n['question'])
    out.append('ANS: %s  %s\n  -> %s  %s' % (o['answer'], o.get('answer_range'), n['answer'], n.get('answer_range')))
    if n.get('hidden_cv_chosen') != o.get('hidden_cv_chosen'):
        out.append('HIDDEN: %s\n  -> %s' % (o.get('hidden_cv_chosen'), n.get('hidden_cv_chosen')))
    for s in sources_of(o, 'web_search'):
        out.append('  ws-old: %s | %s' % (s.get('key_information'), s.get('url')))
    for s in sources_of(n, 'web_search'):
        out.append('  ws-new: %s | %s | q=%s\n     quote: %s' % (s.get('key_information'), s.get('url'), s.get('search_query'),
                                                              (s.get('verified_quote') or '')[:300]))
    for s in sources_of(n, 'code_executor'):
        out.append('  code-out: %s' % (s.get('key_information') or '')[:300].replace('\n', ' / '))
    os_, _ = get_steps(o)
    ns, _ = get_steps(n)
    if level == 'L2':
        out += ['  S: ' + x for x in ns]
    else:
        for line in difflib.unified_diff(os_, ns, lineterm='', n=0):
            if not line.startswith(('---', '+++', '@@')):
                out.append('  ' + line[:500])
open(WORK + '/review_dump_%s.txt' % level, 'w', encoding='utf-8').write('\n'.join(out))
print(len(todo))
