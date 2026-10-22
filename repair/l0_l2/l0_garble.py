# Provenance (TriPhysBench release): fix_code/_v8_work/l0_garble.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L0 text cleanup: garbled tokens (stray CJK / Thai text left by the generator) in QAs outside the ws / code repair set
(the repair-set QAs are rewritten by L1 / L2, whose validators reject garbled text).
Explicit replacements; every restored number is the real stdout of the QA's own code_executor block (run_code).
B|a00652/00305|0 also had arithmetic slips in steps 7-9 / answer (code prints 383558.17 J) -> corrected to the code output.
Applied in place to stage_L0/*class_qa_{train,test}.json; log -> l0_garble_log.json
Usage: python l0_garble.py"""
import json
from fixlib import WORK, load, save, iter_qas, get_steps, set_steps, sources_of, run_code

FIX = {
    'A|ec944f9c-2cd6-4f7d-9883-0bebb1feb0f7|1': {'steps': {4: ('≈认可 3.95 J', '≈ 3.95 J')}},
    'A|aa0ea3a4-9d16-4614-87f4-fd1070727f39|1': {'steps': {5: ('(4.71229)² =自身.', '(4.71229)² = 6.88 J.')}},
    'B|a00242/00872|0': {'steps': {4: ('Step两位ติดปัญหาการทำงานของ code_executor แต่จากการคำนวณด้วยตนเอง: P =',
                                       'Step 5: The code_executor output gives P =')}},
    'B|a00464/00161|0': {'steps': {7: ('× 9.81 m/s² ×数是需要重新计算... Let me compute properly.',
                                       '× 9.81 m/s² × 0.599 m ≈ 32,881 J (code_executor output).')}},
    'B|a00652/00305|0': {'steps': {6: ('= 38949.525 J.', '= 38,995.425 J.'),
                                   7: ('=直播 344,672.5 J.', '= 344,562.75 J.'),
                                   8: ('= 38949.525 J + 344,672.5 J = 383,622.025 J.',
                                       '= 38,995.425 J + 344,562.75 J = 383,558.175 J (code_executor output 383558.17 J).')},
                         'answer': ('383,622 J (or 383.62 kJ)', '383,558 J (or 383.56 kJ)'),
                         'code_ki': True},
}


def main():
    log = []
    for cls in 'ab':
        for split in ('train', 'test'):
            data = load(cls, split, 'L0')
            touched = False
            for qid, vid, qa in iter_qas(data, cls, split):
                f = FIX.get(qid)
                if not f:
                    continue
                steps, kind = get_steps(qa)
                for i, (a, b) in f['steps'].items():
                    assert a in steps[i], (qid, i, steps[i])
                    steps[i] = steps[i].replace(a, b)
                set_steps(qa, steps, kind)
                if 'answer' in f:
                    assert qa['answer'] == f['answer'][0], qa['answer']
                    qa['answer'] = f['answer'][1]
                if f.get('code_ki'):
                    for s in sources_of(qa, 'code_executor'):
                        r = run_code(s.get('code') or '')
                        assert r['ok'], r
                        s.setdefault('key_information_annotated', s.get('key_information'))
                        s['key_information'] = r['out']
                log.append({'qid': qid, 'fix': {str(k): v for k, v in f['steps'].items()}, 'answer': f.get('answer')})
                touched = True
            if touched:
                save(data, cls, split, 'L0')
    assert len(log) == len(FIX), [q for q in FIX if q not in {x['qid'] for x in log}]
    json.dump(log, open(WORK + '/l0_garble_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('fixed', len(log))


if __name__ == '__main__':
    main()
