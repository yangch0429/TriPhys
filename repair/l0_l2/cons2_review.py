# Provenance (TriPhysBench release): fix_code/_v8_work/cons2_review.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Manual review of cons2_plan.json (consistency round 2) before the rewrite: overrides of the reviewer-LLM plan.
- keep: the QAs name different objects in their questions (not the same property of the same object);
- canonical: a better-fitting verified source of another QA of the video (cons_fix.verified), targets recomputed;
- extra: an additional instruction for the rewriter of one target QA.
Usage: python cons2_review.py   (cons2_plan.json updated in place; run before cons_fix.py fix2)"""
import json
from fixlib import WORK, sources_of
from current import current
from cons_fix import verified, overlap

PLAN = WORK + '/cons2_plan.json'

KEEP = {('A|2a160c63-2acd-463a-8431-3dbf77d6de8e', 'wrench mass/weight'):
        'not the same object: QA 1 asks about a standard adjustable wrench, QA 2 about a typical bicycle wrench'}
CANON = {  # (video, property) -> (qa, web_search index, interval, why)
    ('B|a00385/00050', "mass of chrome steel sphere"):
        ('test:B|a00385/00050|0', 0, [0.009072, 0.009072],
         'the video shows a desk-toy Newton\'s cradle (QA 0: 15 cm strings); a 9.072 g chrome steel ball fits it, '
         'a 0.23 kg industrial or 0.7 kg research sphere does not'),
    ('B|a00385/00050', 'suspension string length (pendulum length)'):
        ('test:B|a00385/00050|2', 1, [0.3, 0.3],
         'desk-toy cradle (QA 0: 15 cm strings): the 30 cm DIY cradle string fits better than the 0.47 m research '
         'pendulum')}
EXTRA = {('A|c9230a3e-6f26-4f2f-bbfa-9daedb658568|0'):
         'The question must no longer ask for "typical adult male body mass and standard biomechanical proportions '
         'for hand mass relative to body mass": rewrite it to use the typical human hand mass, so that question and '
         'solution agree (the body-mass and hand-percentage sources are no longer needed).'}


def main():
    plan = json.load(open(PLAN, encoding='utf-8'))
    qas = {q: qa for q, (qa, lv) in current(pending=True).items()}
    hit = set()
    for p in plan:
        key = (p['video'], p['property'])
        if key in KEEP:
            p.update(verdict='keep', why='manual review: ' + KEEP[key], targets=[], unrepaired_targets=[])
            p.pop('canonical', None)
            hit.add(key)
        if key in CANON:
            q, i, iv, why = CANON[key]
            s = sources_of(qas[q], 'web_search')[i]
            f = verified(s)
            assert f, key
            p.update(verdict='fix', canonical={'qa': q, 'source_idx': i, 'key_information': s.get('key_information'),
                                               **f},
                     canonical_interval=iv, why='manual review: ' + why,
                     targets=[t for t in p['values'] if t != q and not overlap(p['intervals'][t], iv)])
            hit.add(key)
        for t in p.get('targets', []):
            if t in EXTRA:
                p['extra'] = {t: EXTRA[t]}
                hit.add(t)
    assert hit == set(KEEP) | set(CANON) | set(EXTRA), hit
    json.dump(plan, open(PLAN, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for p in plan:
        if (p['video'], p['property']) in CANON:
            print(p['property'], '->', p['canonical']['key_information'][:80], p['targets'])
    print('fix', sum(p['verdict'] == 'fix' for p in plan), '| targets', len({t for p in plan for t in p.get('targets', [])}))


if __name__ == '__main__':
    main()
