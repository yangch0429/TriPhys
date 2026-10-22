# Provenance (TriPhysBench release): fix_code/_v8_work/final_writeback.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Write the repaired QAs back into the fix_code benchmark files and record every changed QA.
Every QA takes its current version (current.current(pending=False)): L0 (applied to every QA) <- accepted repair of
review_log.json (L1 / L2 mapped onto the original QA's keys / L1E) <- accepted consistency fix (review_cons.json)
<- accepted post repairs (review_post.json: L1T, FIX39, FIX39W, FIX3, CONS2, FIX4; see current.py).
Problem QAs without an accepted repair keep the L0 version and are listed in writeback_report.json.
*class_qa_dataset_cleaned.json is rebuilt from the written train / test files (same order, matched by the original
question text of _v8_work/stage_orig). Checks: counts (A 233 / B 212), difficulty unchanged, question_type unchanged
unless a post repair (L1T / FIX39 / FIX39W / FIX3) relabelled it.
Change tracker changed_qas.json: every QA that differs from stage_orig (the benchmark before the repair):
{qid, split, class, levels, fields, retest (question / answer / answer_range / annotated_solution changed = seen by the
evaluated model or the judge, as build_subset.py), mode (full: question / answer / answer_range / question_type
changed; scs: only annotated_solution changed), answer: [old, new], answer_range: [old, new]}
Usage: python final_writeback.py"""
import json
from collections import Counter
from fixlib import FC, WORK, load, iter_qas
from current import current

problems = [p['qid'] for p in json.load(open(WORK + '/problems.json', encoding='utf-8'))]
EVAL_FIELDS = ('question', 'answer', 'answer_range', 'annotated_solution')
ALL_FIELDS = EVAL_FIELDS + ('question_type', 'sources', 'hidden_cv_chosen')
FULL_FIELDS = ('question', 'answer', 'answer_range', 'question_type')
TYPE_FREE = ('L1T', 'FIX39', 'FIX39W', 'FIX3')     # post repairs that may relabel question_type
POST = TYPE_FREE + ('CONS2', 'FIX4', 'FIX5', 'FIX6', 'FIX7', 'FIX8')


def main():
    cur = current(pending=False)
    review = json.load(open(WORK + '/review_log.json', encoding='utf-8'))
    stat, report, tracker = Counter(), {'not_repaired': [], 'repaired': {}}, []
    written = {}
    for cls in 'ab':
        for split in ('train', 'test'):
            data = load(cls, split, 'L0')
            orig = {qid: qa for qid, _, qa in iter_qas(load(cls, split, 'orig'), cls, split)}
            for qid, vid, qa in iter_qas(data, cls, split):
                o = orig[qid]
                new, levels = cur[qid]
                qa.clear()
                qa.update(new)
                if qid in problems:
                    rl = [lv for lv in levels if lv in ('L1', 'L2', 'L1E') + POST]
                    if rl:
                        report['repaired'][qid] = rl[0]
                        stat['repaired_' + rl[0]] += 1
                    else:
                        report['not_repaired'].append({'qid': qid, 'review': review.get(qid)})
                        stat['not_repaired'] += 1
                if {'CONS', 'CONS2'} & set(levels):
                    stat['consistency_fixed'] += 1
                assert qa['difficulty'] == o['difficulty'], qid
                assert qa['question_type'] == o['question_type'] or set(levels) & set(TYPE_FREE), qid
                for lv in POST:
                    stat[lv] += lv in levels
                fields = [k for k in ALL_FIELDS if o.get(k) != qa.get(k)]
                if fields:
                    rec = {'qid': qid, 'split': split, 'class': cls.upper(), 'levels': levels, 'fields': fields,
                           'retest': any(k in fields for k in EVAL_FIELDS + ('question_type',)),
                           'mode': 'full' if any(k in fields for k in FULL_FIELDS) else
                           'scs' if 'annotated_solution' in fields else None,
                           'answer_changed': o.get('answer') != qa.get('answer'),
                           'answer': [o.get('answer'), qa.get('answer')],
                           'answer_range': [o.get('answer_range'), qa.get('answer_range')]}
                    tracker.append(rec)
                    stat['retest'] += rec['retest']
                    stat['retest_%s' % rec['mode']] += rec['retest']
                written[o['question']] = qa
            json.dump(data, open('%s/%sclass_qa_%s.json' % (FC, cls, split), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
        # rebuild the cleaned file in its own order
        cleaned = load(cls, 'dataset_cleaned', 'orig')
        qlists = [v['qa_pairs'] for v in cleaned] if cls == 'a' else [cleaned['qa_pairs']]
        n = 0
        for ql in qlists:
            for i, qa in enumerate(ql):
                ql[i] = written[qa['question']]
                n += 1
        assert n == {'a': 233, 'b': 212}[cls], n
        if cls == 'b':
            cleaned['total_valid_qa'] = n
            cleaned['difficulty_breakdown'] = dict(Counter(q['difficulty'] for q in cleaned['qa_pairs']))
        json.dump(cleaned, open('%s/%sclass_qa_dataset_cleaned.json' % (FC, cls), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        stat[cls + '_cleaned'] = n
    report['stat'] = dict(stat)
    json.dump(report, open(WORK + '/writeback_report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(tracker, open(WORK + '/changed_qas.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(dict(stat))
    print('changed QAs', len(tracker), '| by levels', Counter('+'.join(r['levels']) for r in tracker),
          '| answer changed', sum(r['answer_changed'] for r in tracker))


if __name__ == '__main__':
    main()
