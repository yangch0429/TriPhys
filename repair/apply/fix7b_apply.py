# Provenance (TriPhysBench release): fix_code/evaluation/scripts/fix7b_apply.py
# Role: writes repaired QAs into the dataset files / re-assembles zero-shot results.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Second pass on two fix7 QAs after the question-side search check (10/5): the constant must follow from the question.
Applied to aclass/bclass_qa_test.json and aclass/bclass_qa_dataset_cleaned.json (backups *.before_fix7b.json);
build_fix7_subset.py still compares against *.before_fix7.json, so both QAs stay in the re-evaluation subset.
  A|test|19|1 (A medium, hidden CV = wrist speed, unchanged): reported three-point release velocities differ
              (6.89 sapub, 7.82 Bartell, 8.71/8.76 Slegers 2021) -> range answer 6.89-8.76 m/s -> factor 2.46-3.98;
              redundant ball-mass search removed (mass cancels); stays TYPE_D like A|test|20|1
  B|test|26|0 (B easy, 1 hidden spec = almond piece diameter, unchanged): "size of medium diced almonds" matched
              3.2-8.7 mm (ABC 22/8) and 4.0-6.0 mm -> qualifier "Almond Board of California common specification",
              diameter = mean of the two sieve sizes (5.95 mm); no web value written into the question (B-class rule)
Usage: python fix7b_apply.py [--dry]"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix7_apply import FIX, run, src, code_src, full_qas  # noqa: E402

DRY = '--dry' in sys.argv


def a19_1(x):
    x['question'] = x['question'].replace(
        'Using standard basketball specifications and typical three-point shot release velocity from basketball research,',
        'Using three-point shot release velocities reported in basketball research,')
    x['question'] += ' Provide the answer as a range covering the reported three-point release velocities.'
    x['annotated_solution'] = [
        x['annotated_solution'][0], x['annotated_solution'][1],
        "Step 3: Use web_search(query='typical three point shot release velocity') to retrieve a reported three-point "
        "release velocity: during three-point shooting the initial release velocity of the basketball is 6.89 m/s.",
        "Step 4: Use web_search(query='basketball three-point shot release speed') to retrieve another reported value: "
        "Slegers et al. (2021) report that the three-point shot is released with a velocity of 8.76 m/s.",
        'Step 5: Since the same ball is used for both shots, the mass cancels in the ratio calculation. Compute ratio R = (v_3pt/v_close)².',
        'Step 6: Use code_executor: R_min = (6.89 / 4.3927)² = 2.46 and R_max = (8.76 / 4.3927)² = 3.98.',
        'Step 7: The kinetic energy of a three-point shot exceeds that of the close-range layup by a factor between 2.46 and 3.98.']
    x['total_steps'] = 7
    s = src(x, 'web_search', 0)
    s.clear()
    s.update({'type': 'web_search', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8256521/',
              'key_information': 'three-point shot release velocity: 8.76 m/s (Slegers et al. 2021)',
              'url_annotated': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8256521/',
              'search_query': 'basketball three-point shot release speed',
              'verified_quote': 'the three-point shot is released with a velocity of 8.76 m/s and an angle of 48 degrees'})
    ws = [q for q in x['sources'] if q['type'] == 'web_search']
    x['sources'] = [q for q in x['sources'] if q['type'] == 'video_observation'] + [ws[1], ws[0]]
    x['sources'].append({'type': 'code_executor', 'code': ''})
    out = code_src(x, "v_close = 4.3927\nfor v3 in (6.89, 8.76):\n    print(f'v_3pt = {v3} m/s: KE ratio = {(v3 / v_close)**2:.2f}')")
    x['answer'] = 'The kinetic energy of a three-point shot exceeds that of the close-range layup by a factor between 2.46 and 3.98.'
    x['answer_range'] = [2.46, 3.98]
    return out


def b26_0(x):
    x['question'] = x['question'].replace(
        'the true density of almond kernels and the size of medium diced almonds,',
        'the true density of almond kernels and the Almond Board of California common specification for medium diced '
        'almonds (take the piece diameter as the mean of its two sieve sizes),')
    st = x['annotated_solution']
    st[5] = ("Step 6: Use web_search(query='Almond Board of California medium diced almonds specification') to retrieve the "
             "size specification: Medium 22/8 = 22/64\" & 8/64\" (8.7 & 3.2 mm); piece diameter = mean of the two sieve "
             "sizes = 5.95 mm.")
    st[6] = ("Step 7: Apply Stokes' law: diameter = 5.95 mm → r = 0.002977 m, g = 9.81 m/s², ρ_almond = 900-995 kg/m³, "
             "ρ_honey = 1415 kg/m³, η = 14.095 Pa·s.")
    st[7] = ('Step 8: Use code_executor to compute v_t = (2 × r² × 9.81 × (ρ_almond - 1415)) / (9 × 14.095): '
             '-0.706 mm/s for 900 kg/m³ and -0.576 mm/s for 995 kg/m³. The negative sign means the almond piece rises '
             '(almond kernel density < honey density) at a terminal speed of about 0.576-0.706 mm/s.')
    s = src(x, 'web_search', 3)
    s.update({'key_information': 'medium diced almonds (ABC 22/8): 22/64" & 8/64" (8.7 & 3.2 mm), mean diameter = 5.95 mm',
              'search_query': 'Almond Board of California medium diced almonds specification',
              'verified_quote': 'Medium: 22/8......22/64" & 8/64" (8.7 & 3.2mm)'})
    x['hidden_cv_chosen'] = ['chopped almond piece diameter = 5.95 mm']
    out = code_src(x, "g = 9.81\nrho_h = 1415\neta = 14.095\ndiameter_mm = (22 / 64 + 8 / 64) / 2 * 25.4\n"
                      "r = diameter_mm / 1000.0 / 2.0\nprint(f'diameter = {diameter_mm:.2f} mm')\n"
                      "for rho_a in (900, 995):\n    v_t = (2 * r**2 * g * (rho_a - rho_h)) / (9 * eta)\n"
                      "    print(f'rho_almond = {rho_a} kg/m^3: v_t = {v_t*1000:.3f} mm/s')")
    x['answer'] = 'The almond piece rises (it is less dense than honey) at a terminal speed of about 0.576–0.706 mm/s.'
    x['answer_range'] = [0.576, 0.706]
    return out


EDITS = [('A', 19, 1, a19_1), ('B', 26, 0, b26_0)]


def main():
    files = {k: os.path.join(FIX, f) for k, f in [('At', 'aclass_qa_test.json'), ('Bt', 'bclass_qa_test.json'),
                                                   ('Af', 'aclass_qa_dataset_cleaned.json'), ('Bf', 'bclass_qa_dataset_cleaned.json')]}
    D = {k: json.load(open(f, encoding='utf-8')) for k, f in files.items()}
    logf = os.path.join(FIX, 'evaluation', 'fix7_log.json')
    log = json.load(open(logf, encoding='utf-8'))
    for cls, vi, qi, fn in EDITS:
        qid = '%s|test|%d|%d' % (cls, vi, qi)
        tq = D[cls + 't'][vi]['qa_pairs'][qi]
        fq = [q for q in full_qas(cls, D[cls + 'f']) if q['question'] == tq['question']]
        assert len(fq) == 1, (qid, len(fq))
        outs = [fn(x) for x in (tq, fq[0])]
        assert outs[0] == outs[1] and tq == fq[0]
        log[qid].update({'new_question': tq['question'], 'new_answer': tq['answer'], 'answer_range': tq['answer_range'],
                         'code_output': outs[0], 'fix7b': 'question-side search check: range / qualifier'})
        print('==', qid, '\n   Q:', tq['question'], '\n   answer:', tq['answer'], tq['answer_range'],
              '\n   code:', outs[0].replace('\n', ' | '))
    if DRY:
        return
    for k, f in files.items():
        bak = f.replace('.json', '.before_fix7b.json')
        if not os.path.exists(bak):
            shutil.copy(f, bak)
        json.dump(D[k], open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump(log, open(logf, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written')


if __name__ == '__main__':
    main()
