# Provenance (TriPhysBench release): fix_code/evaluation/scripts/fix10_apply.py
# Role: writes repaired QAs into the dataset files / re-assembles zero-shot results.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 (10/13): write the final fix10 dataset (version 08) into the dataset files.
Input evaluation/fix10/final/final_ids.json {'replace': {id: qa}, 'delete': [id, ...]} (made on the server by
fix10/select_final.py: every needs_fix2 QA whose last repaired / redesigned version passed the strict gpt-5.6 audit,
the 7 lost B videos' redesigns, extra redesigned QAs up to the 410 target; every other needs_fix2 QA is deleted).
  {a,b}class_qa_{train,test}.json   by id (CLS|split|video index|qa index, same order as data/v9)
  {a,b}class_qa_dataset_cleaned.json   located by the original question text (+ answer if the text repeats)
Log entries carry the video key (A take_uid, B folder_id/video_id) for fix10_assemble.py. A video keeps its place (the split is by video and every video keeps >= 1 QA); B cleaned: total_valid_qa and
difficulty_breakdown recomputed. Backups *.before_fix10.json; log evaluation/fix10_log.json.
Usage: python fix10_apply.py [--dry]"""
import os, sys, json, copy, shutil
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix7_apply import FIX, full_qas  # noqa: E402

DRY = '--dry' in sys.argv
GONE = object()


def main():
    fin = json.load(open(os.path.join(FIX, 'evaluation', 'fix10', 'final', 'final_ids.json'), encoding='utf-8'))
    names = {(c, s): '%sclass_qa_%s.json' % (c.lower(), s) for c in 'AB' for s in ('train', 'test', 'dataset_cleaned')}
    D = {k: json.load(open(os.path.join(FIX, f), encoding='utf-8')) for k, f in names.items()}
    n0 = {c: len(full_qas(c, D[(c, 'dataset_cleaned')])) for c in 'AB'}
    log, todo = {}, [(k, 'replace', q) for k, q in fin['replace'].items()] + [(k, 'delete', None) for k in fin['delete']]
    assert not set(fin['replace']) & set(fin['delete'])
    marks = []  # (container list, index) to delete at the end, so the ids (positions) stay valid while editing
    for qid, act, new in sorted(todo):
        cls, sp, vi, qi = qid.split('|')
        tl = D[(cls, sp)][int(vi)]['qa_pairs']
        tq = tl[int(qi)]
        full = D[(cls, 'dataset_cleaned')]
        fl = [v['qa_pairs'] for v in full] if cls == 'A' else [full['qa_pairs']]
        hit = [(l, i) for l in fl for i, q in enumerate(l) if q is not GONE and q['question'] == tq['question']]
        if len(hit) > 1:
            hit = [(l, i) for l, i in hit if l[i]['answer'] == tq['answer']]
        assert len(hit) == 1, (qid, len(hit))
        old = copy.deepcopy(tq)
        vk = D[(cls, sp)][int(vi)]['take_uid'] if cls == 'A' else '%s/%s' % (tq['folder_id'], tq['video_id'])
        if act == 'replace':
            for x in (tq, hit[0][0][hit[0][1]]):
                x.clear()
                x.update(copy.deepcopy(new))
            log[qid] = {'action': 'replace', 'video': vk, 'old_question': old['question'], 'new_question': tq['question'],
                        'old_answer': old['answer'], 'new_answer': tq['answer'], 'old_answer_range': old.get('answer_range'),
                        'answer_range': tq.get('answer_range'), 'old_type': old['question_type'],
                        'question_type': tq['question_type'], 'difficulty': tq.get('difficulty'),
                        'annotated_solution': tq.get('annotated_solution'),
                        'changed': sorted(k for k in set(tq) | set(old) if tq.get(k) != old.get(k))}
        else:
            marks += [(tl, int(qi)), hit[0]]
            log[qid] = {'action': 'delete', 'video': vk, 'old_question': old['question'], 'old_answer': old['answer'],
                        'old_type': old['question_type'], 'difficulty': old.get('difficulty')}
    for l, i in marks:
        l[i] = GONE
    for k, d in D.items():
        for v in (d if isinstance(d, list) else [d]):
            v['qa_pairs'][:] = [q for q in v['qa_pairs'] if q is not GONE]
    # user 10/13: test videos whose QAs stay in the unrepaired v9 version move to the end of train; the B strata
    # re-balance (fix10/rebalance_split.py) adds to_train / to_test moves (after every id edit)
    for src, dst, ks in (('test', 'train', set(fin.get('to_train', []))), ('train', 'test', set(fin.get('to_test', [])))):
        for c in 'AB':
            key = lambda v: v.get('take_uid') or '%s/%s' % (v.get('folder_id') or v['qa_pairs'][0].get('folder_id'),
                                                            v.get('video_id') or v['qa_pairs'][0].get('video_id'))
            S = D[(c, src)]
            mv = [v for v in S if v['qa_pairs'] and key(v) in ks]
            S[:] = [v for v in S if not (v['qa_pairs'] and key(v) in ks)]
            D[(c, dst)].extend(mv)
            for v in mv:
                print('%s: %s video %s -> %s (%d QAs)' % (c, src, key(v), dst, len(v['qa_pairs'])))
    for c in 'AB':
        empty = [i for k in ('train', 'test') for i, v in enumerate(D[(c, k)]) if not v['qa_pairs']]
        if empty:
            print('%s: videos without QAs %s' % (c, empty))
            assert DRY
    B = D[('B', 'dataset_cleaned')]
    B['total_valid_qa'] = len(B['qa_pairs'])
    B['difficulty_breakdown'] = dict(Counter(q.get('difficulty') for q in B['qa_pairs']))
    for c in 'AB':
        sp = sum(len(v['qa_pairs']) for k in ('train', 'test') for v in D[(c, k)])
        fl = len(full_qas(c, D[(c, 'dataset_cleaned')]))
        assert sp == fl, (c, sp, fl)
        print('%s: %d -> %d QAs (train %d / test %d)' % (c, n0[c], fl, sum(len(v['qa_pairs']) for v in D[(c, 'train')]),
                                                     sum(len(v['qa_pairs']) for v in D[(c, 'test')])))
    print('actions:', dict(Counter((k[0], k.split('|')[1], v['action']) for k, v in log.items())))
    if DRY:
        json.dump(log, open(os.path.join(FIX, 'evaluation', 'fix10', 'final', 'fix10_log_dry.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        return
    for k, f in names.items():
        p = os.path.join(FIX, f)
        bak = p.replace('.json', '.before_fix10.json')
        if not os.path.exists(bak):
            shutil.copy(p, bak)
        json.dump(D[k], open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(log, open(os.path.join(FIX, 'evaluation', 'fix10_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written; log evaluation/fix10_log.json')


if __name__ == '__main__':
    main()
