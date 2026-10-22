# Provenance (TriPhysBench release): fix_code/_v8_work/l0_labels.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L0 text cleanup: malformed step labels ("Step پنج:", "Step135.5:", "Step N:", "Step929Identify ...") left by the
generator in QAs outside the ws / code repair set (repair-set QAs are rewritten by L1 / L2, whose validators reject
malformed labels). The steps of such a QA are renumbered "Step 1:" .. "Step n:"; the step text is unchanged.
Applied in place to stage_L0/*class_qa_{train,test}.json; log -> l0_labels_log.json
Usage: python l0_labels.py"""
import json, re
from fixlib import WORK, load, save, iter_qas, get_steps, set_steps

REPAIR_SET = {x['qid'] for x in json.load(open(WORK + '/problem_qas.json', encoding='utf-8'))}
LABEL = re.compile(r'Step[^:：]{0,30}[:：]\s*|Step\d+(?=[A-Z])')


def bad(steps):
    return any(s.startswith('Step') and not re.match(r'Step \d+:', s) for s in steps)


def main():
    log = []
    for cls in 'ab':
        for split in ('train', 'test'):
            data = load(cls, split, 'L0')
            touched = False
            for qid, vid, qa in iter_qas(data, cls, split):
                steps, kind = get_steps(qa)
                if qid in REPAIR_SET or not bad(steps):
                    continue
                new = []
                for i, s in enumerate(steps):
                    m = LABEL.match(s)
                    assert m, (qid, s[:60])
                    new.append('Step %d: %s' % (i + 1, s[m.end():]))
                set_steps(qa, new, kind)
                log.append({'qid': qid, 'old': [s[:40] for s in steps], 'new': [s[:40] for s in new]})
                touched = True
            if touched:
                save(data, cls, split, 'L0')
    json.dump(log, open(WORK + '/l0_labels_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('relabelled', len(log))


if __name__ == '__main__':
    main()
