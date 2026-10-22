# Provenance (TriPhysBench release): fix_code/_v8_work/l1_textfix.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Cosmetic fixes of L1 repairs that are correct in content but fail a text check (l1_rewrite.text_checks):
malformed step labels kept from the original QA ("Step125:", "Step 6a:", ...) -> renumbered "Step 1:" .. "Step n:";
explicit string replacements for stray tokens / wording. Numbers, sources and answers are unchanged.
Applied in place to l1_results.json; log -> l1_textfix_log.json
Usage: python l1_textfix.py"""
import json, re
from fixlib import WORK, get_steps, set_steps

LABEL = re.compile(r'Step[^:：]{0,30}[:：]\s*')
RELABEL = ['B|a00138/01103|0', 'B|a00079/00862|1', 'test:A|daf66c5f-284b-45da-af6b-e9a45242af7f|0',
           'A|8fc1c385-7529-47ce-a2de-fc42f6e44303|0', 'test:A|9fa3de38-5dfe-47d7-8b53-ebae900dcdb5|0',
           'test:B|a00325/00280|0']
REPLACE = {
    'B|a00479/00073|1': [('=アイレーションは0.156', '= 0.156')],
    'test:B|a00611/02070|0': [('question-supplied m', 'given m'), ('This supplied mass', 'This given mass'),
                              ('supplied in the question', 'given in the question')],
}


def main():
    res = json.load(open(WORK + '/l1_results.json', encoding='utf-8'))
    log = []
    for qid in RELABEL:
        qa = res[qid]['qa']
        steps, kind = get_steps(qa)
        new = []
        for i, s in enumerate(steps):
            m = LABEL.match(s)
            assert m, (qid, s[:60])
            new.append('Step %d: %s' % (i + 1, s[m.end():]))
        set_steps(qa, new, kind)
        log.append({'qid': qid, 'relabel': [(a[:30], b[:30]) for a, b in zip(steps, new) if a[:12] != b[:12]]})
    for qid, reps in REPLACE.items():
        t = json.dumps(res[qid]['qa'], ensure_ascii=False)
        for a, b in reps:
            assert a in t, (qid, a)
            t = t.replace(a, b)
        res[qid]['qa'] = json.loads(t)
        log.append({'qid': qid, 'replace': reps})
    json.dump(res, open(WORK + '/l1_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(log, open(WORK + '/l1_textfix_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('fixed', len(log))


if __name__ == '__main__':
    main()
