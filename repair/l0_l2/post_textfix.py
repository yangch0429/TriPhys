# Provenance (TriPhysBench release): fix_code/_v8_work/post_textfix.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Field-level fixes of accepted L1 / L2 repairs found in the one-by-one review (review_log.json):
- answer_range reset to null where the question type has no range (TYPE_C) and the original QA had none;
- B-class hidden_cv_chosen written as "name = value" (the L2 generator listed only the names), values from the QA's
  own web_search sources;
- the flow-regime verdict asked by the question restored in the answer text.
Numbers of answers / sources are unchanged. Applied in place to l1_results.json / l2_results.json; log ->
post_textfix_log.json
Usage: python post_textfix.py"""
import json
from fixlib import WORK

FIX = {
    'l1': {
        'test:B|a00523/01163|1': {'answer_range': None},
        'test:B|a00399/00010|0': {'answer': '1905.0 (dimensionless), indicating laminar flow (Re < 2000)'},
    },
    'l2': {
        'B|a00919/01155|2': {'answer_range': None},
        'B|a00818/01712|1': {'hidden_cv_chosen': ['honey viscosity = 10 Pa·s', 'honey density = 1360 kg/m³',
                                                  'bubble diameter = 2 mm']},
        'test:B|a00523/01163|0': {'hidden_cv_chosen': ['epoxy resin viscosity = 1.6–2.0 Pa·s',
                                                       'laboratory spatula blade = 102 mm × 19 mm']},
    },
}


def main():
    log = []
    for level, fixes in FIX.items():
        path = WORK + '/%s_results.json' % level
        res = json.load(open(path, encoding='utf-8'))
        for qid, f in fixes.items():
            qa = res[qid]['qa']
            for k, v in f.items():
                log.append({'level': level, 'qid': qid, 'field': k, 'old': qa.get(k), 'new': v})
                qa[k] = v
        json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(log, open(WORK + '/post_textfix_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('fixed', len(log))


if __name__ == '__main__':
    main()
