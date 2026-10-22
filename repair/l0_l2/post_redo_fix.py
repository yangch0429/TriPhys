# Provenance (TriPhysBench release): fix_code/_v8_work/post_redo_fix.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Fixes of accepted L1 redo repairs found in the one-by-one review (review_log.json):
- test:B|a00588/00347|1: the car mass must be the quoted value (3,300 lbs = 1,490.6 kg, not a rounded 1500 kg);
  the code is re-run and the answer / steps are taken from its output;
- A|ed2c5dc8|0: meta wording "benchmark uncertainty range" in the answer -> the original answer's wording.
Applied in place to l1_results.json (backup l1_results_before_redofix.json); log -> post_redo_fix_log.json
Usage: python post_redo_fix.py"""
import json, shutil
from fixlib import WORK, get_steps, set_steps, run_code

MASS_Q = 'test:B|a00588/00347|1'
ED = 'A|ed2c5dc8-2878-43e4-9547-8ed8d6715460|0'


def main():
    path = WORK + '/l1_results.json'
    shutil.copy(path, WORK + '/l1_results_before_redofix.json')
    res = json.load(open(path, encoding='utf-8'))
    log = []

    qa = res[MASS_Q]['qa']
    code = next(s for s in qa['sources'] if s.get('type') == 'code_executor')
    assert 'm = 1500\n' in code['code']
    code['code'] = code['code'].replace('m = 1500\n', 'm = 1490.6\n')
    out = run_code(code['code'])
    assert out['ok'], out
    code['key_information'] = out['out'].strip()
    q_kj = code['key_information'].splitlines()[-1].split('=')[-1].strip()      # "48.1 kJ"
    q_j = code['key_information'].splitlines()[-1].split('=')[1].strip()        # "48130 J"
    ws = next(s for s in qa['sources'] if s.get('key_information') == 'NASCAR car mass = 1500 kg')
    ws['key_information'] = 'minimum weight 3,300 lbs (1,490.6 kg); NASCAR car mass = 1490.6 kg'
    qa['hidden_cv_chosen'][0] = 'NASCAR car mass = 1490.6 kg'
    steps, kind = get_steps(qa)
    steps[2] = steps[2].replace('Result: ~1500 kg.', 'Result: 3,300 lbs = 1,490.6 kg.')
    steps[6] = steps[6].replace('0.75 × 1500 kg', '0.75 × 1490.6 kg')
    steps[7] = 'Step 8: Result is Q ≈ %s = %s.' % (q_j, q_kj)
    set_steps(qa, steps, kind)
    log.append({'qid': MASS_Q, 'answer': [qa['answer'], q_kj], 'code_out': code['key_information']})
    qa['answer'] = q_kj

    qa = res[ED]['qa']
    new = qa['answer'].replace(', with the benchmark uncertainty range of 30-70 times.',
                               ' (range: 30-70 times considering measurement uncertainties).')
    assert new != qa['answer']
    log.append({'qid': ED, 'answer': [qa['answer'], new]})
    qa['answer'] = new

    json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(log, open(WORK + '/post_redo_fix_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
