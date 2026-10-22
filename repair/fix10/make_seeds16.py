# Provenance (TriPhysBench release): fix_code/evaluation/fix10/make_seeds16.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""user 10/13 type check: QAs outside the 255 whose final answer form does not match the type become seed records (the
original QA + the known defect); audit10 adds the seed defects as major, audit_fix repairs them, report10 counts them.
Also writes unres57_ids.json (given_v15 unresolved)."""
import sys, json
sys.argv = [sys.argv[0], 'unres151_ids.json']
from fix10 import R  # noqa: E402
PCT = ('the answer is one value with an invented +-5% "accepted range"; the type is decided by the final answer: make it '
       'TYPE_A (one value, answer_range null) and remove the +-% from the code and the answer')
SEED = {'A|train|37|1': PCT, 'B|train|0|0': PCT, 'B|train|46|0': PCT,
        'B|train|24|0': 'the answer range 12.5-15.0 kV comes from an invented gap of 4.5-5.5 mm, not from a range input: '
                        'make it TYPE_A with one value for the gap the question states',
        'B|train|74|0': 'the time range comes from the viscosity range 2-4 Pa s (a real range input): keep TYPE_B but the answer '
                        'itself must be the interval "14.4-28.9 s" (not a point value with a range remark)'}
items = {i['id']: i for i in R.load()}
out = {k: dict(id=k, cls=items[k]['cls'], split=items[k]['split'], status='fixed', mode='seed', qa=items[k]['qa'],
               seed_defects=[v]) for k, v in SEED.items()}
json.dump(out, open('seed_v16.json', 'w'), ensure_ascii=False, indent=1)
g = json.load(open('given_v15.json'))
u = sorted(k for k, r in g.items() if '|' in k and r.get('status') == 'unresolved')
json.dump(u, open('unres57_ids.json', 'w'))
print('seeds', len(out), 'unresolved', len(u))
