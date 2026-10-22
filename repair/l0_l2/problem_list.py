# Provenance (TriPhysBench release): fix_code/_v8_work/problem_list.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Problem-QA list for the two-level fix, keyed on the fix_code data (unfixed version).
problem = any unresolved ws position, or a code block that does not reproduce the answer (after the v8 automatic repair).
Output: _v8_work/problem_qas.json"""
import json, os
from collections import Counter
W = os.path.dirname(os.path.abspath(__file__))
R = W + '/server_reports/'
FC = os.environ.get('TRIPHYS_WORK_DIR') or os.path.dirname(W)
ws = json.load(open(R + 'v8_wsfix_report.json')) + json.load(open(R + 'v8_wsfix_report_test.json'))
code_tr = json.load(open(R + 'v8_code_final.json'))
code_te = {r['qid']: r for r in json.load(open(R + 'v8_test_code_check.json'))}
out, stat = [], Counter()
for split in ('train', 'test'):
    for cls in 'ab':
        data = json.load(open('%s/%sclass_qa_%s.json' % (FC, cls, split), encoding='utf-8'))
        for vid in data:
            vtag = vid.get('take_uid') or '%s/%s' % (vid.get('folder_id'), vid.get('video_id'))
            for i, qa in enumerate(vid['qa_pairs']):
                qid = ('test:' if split == 'test' else '') + '%s|%s|%d' % (cls.upper(), vtag, i)
                stat[split, cls, 'qa'] += 1
                wr = [r for r in ws if r['qid'] == qid]
                n_ws = sum(1 for s in qa.get('sources', []) if s.get('type') == 'web_search')
                if wr and wr[0]['n_ws'] != n_ws:
                    stat['ws_count_mismatch'] += 1
                unres = [r['pos'] for r in wr if r['status'] == 'unresolved']
                if split == 'train':
                    bl = code_tr.get(qid) or []
                    bad = [k for k, b in enumerate(bl) if not b['ok']]
                    verd = [(bl[k].get('diag') or {}).get('verdict') for k in bad]
                else:
                    c = code_te.get(qid)
                    bad = [k for k, b in enumerate(c['blocks']) if b['problem']] if c else []
                    verd = [c.get('verdict')] if bad else []
                if unres or bad:
                    out.append({'qid': qid, 'split': split, 'cls': cls.upper(), 'difficulty': qa.get('difficulty'),
                                'question_type': qa.get('question_type'), 'ws_unresolved_pos': unres,
                                'ws_replaced_pos': [r['pos'] for r in wr if r['status'] == 'replaced'],
                                'code_bad_blocks': bad, 'code_verdicts': verd})
json.dump(out, open(W + '/problem_qas.json', 'w'), ensure_ascii=False, indent=1)
print({'%s' % (k,): v for k, v in stat.items()})
for split in ('train', 'test'):
    for cls in 'AB':
        rs = [r for r in out if r['split'] == split and r['cls'] == cls]
        print(split, cls, len(rs), 'ws_only', sum(1 for r in rs if r['ws_unresolved_pos'] and not r['code_bad_blocks']),
              'code_only', sum(1 for r in rs if r['code_bad_blocks'] and not r['ws_unresolved_pos']),
              'both', sum(1 for r in rs if r['code_bad_blocks'] and r['ws_unresolved_pos']),
              dict(Counter(r['difficulty'] for r in rs)), dict(Counter(v for r in rs for v in r['code_verdicts'])))
