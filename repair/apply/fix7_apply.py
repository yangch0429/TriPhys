# Provenance (TriPhysBench release): fix_code/evaluation/scripts/fix7_apply.py
# Role: writes repaired QAs into the dataset files / re-assembles zero-shot results.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Fix the 7 test QAs with confirmed annotation errors (manual audit 10/5) and propagate the q14 "amateur" edit to the
full A dataset. Applied to aclass/bclass_qa_test.json and aclass/bclass_qa_dataset_cleaned.json (backups *.before_fix7.json).
Every edited QA is located in the full dataset by its original question text; every new code is executed and its
output stored as key_information. Log: evaluation/fix7_log.json (qid -> changed fields, old/new answer).
  A|test|19|1 three-point KE factor: lower bound 6.67 m/s had no source -> single value from the quote (6.89 m/s), 2.46
  A|test|20|1 vertical jump: quote is about college-aged males, question said basketball player -> question wording
  A|test|6|0  (q13) pedal centripetal force: quote 250-280 g per PAIR -> 0.125-0.140 kg per pedal, range answer
  B|test|17|0 soil slide: 0.3-0.5 is the engineering soil friction factor, not a soil-on-soil coefficient -> wording
  B|test|26|0 almond in honey: 460 kg/m3 is bulk density -> almond kernel true density 900-995 kg/m3 (Aydin 2003)
  B|test|15|1 9-ball torque: quote gives 0.16 kg for numbered balls (0.17 kg = cue ball) -> m = 0.16 kg
  B|test|2|0  marble: mu_roll 0.018 is a DEM calibration value for glass particles, no glass-on-wood value exists ->
              negligible rolling resistance, no web constant
Usage: python fix7_apply.py [--dry]"""
import os, sys, json, copy, shutil, subprocess, difflib

FIX = os.environ.get('TRIPHYS_WORK_DIR') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRY = '--dry' in sys.argv


def run(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding='utf-8',
                          env=dict(os.environ, PYTHONUTF8='1')).stdout.strip()


def src(x, t, i=0):
    return [s for s in x['sources'] if s['type'] == t][i]


def code_src(x, code):
    s = src(x, 'code_executor')
    s['code'] = code
    s['key_information'] = run(code)
    return s['key_information']


# ---------------------------------------------------------------- edits (each takes the QA dict and edits in place)
def a19_1(x):
    x['question'] = x['question'].replace(
        ' Provide the answer as a range accounting for realistic variations in ball mass and three-point release velocity.', '')
    st = x['annotated_solution']
    st[3] = ("Step 4: Use web_search(query='typical three point shot release velocity') to retrieve typical three-point shot "
             "release velocity. Research reports that during three-point shooting the initial release velocity of the "
             "basketball is 6.89 m/s.")
    st[5] = 'Step 6: Calculate the ratio: R = (6.89 / 4.3927)² = 2.46.'
    st[6] = 'Step 7: The kinetic energy of a three-point shot exceeds that of the close-range layup by a factor of about 2.46.'
    del st[7:]
    x['total_steps'] = len(st)
    src(x, 'web_search', 1)['key_information'] = 'ball release velocity for three-point shots: 6.89 m/s (from research literature)'
    x['answer'] = 'The kinetic energy of a three-point shot exceeds that of the close-range layup by a factor of about 2.46.'
    x['answer_range'] = None
    return '%.4f' % ((6.89 / 4.3927) ** 2)


def a20_1(x):
    q = x['question']
    q = q.replace('The typical vertical jump height for a basketball player is obtained via web search.',
                  'The typical vertical jump height for college-aged males is obtained via web search.')
    q = q.replace('to match this typical basketball performance range?', 'to match this typical vertical jump range?')
    x['question'] = q
    st = x['annotated_solution']
    st[2] = ("Step 3: Use web_search(query='average vertical jump college-aged males') to retrieve the typical vertical jump "
             "of college-aged males: 40 to 45 cm (0.40 to 0.45 m).")
    st[5] = ('Step 6: The athlete would need to increase their vertical takeoff speed by a factor of 1.56 to 2.12 to match '
             'the typical vertical jump of college-aged males.')
    src(x, 'web_search')['search_query'] = 'average vertical jump college-aged males'
    x['answer'] = ('The observed jump height is lower than the typical range, so the athlete would need to increase their '
                   'vertical takeoff speed by a factor of 1.56 to 2.12 to match the typical vertical jump of college-aged males.')
    return run(src(x, 'code_executor')['code']).splitlines()[-1]


def a6_0(x):
    x['question'] = x['question'].replace(
        'Use standard bicycle specifications for the crank arm length and pedal mass.',
        'Use standard road bike specifications for the crank arm length and pedal mass, and give the answer as a range '
        'covering typical road pedal weights.')
    x['question_type'] = 'TYPE_B'
    st = x['annotated_solution']
    st[3] = ("Step 4: Use web_search(query='typical road bike pedal mass weight per pedal') to retrieve typical pedal mass. "
             "Search results compare a pair of Eggbeater pedals and Ultegra pedals (280 g vs. about 250 g per pair), "
             "i.e. 0.125-0.140 kg per pedal.")
    st[4] = ('Step 5: Use code_executor to compute centripetal force. Centripetal acceleration a_c = v²/r = (2.4804 m/s)² / '
             '0.170 m = 36.19 m/s². Centripetal force F = m × a_c = 0.125 kg × 36.19 m/s² = 4.52 N to '
             '0.140 kg × 36.19 m/s² = 5.07 N.')
    st[5] = 'Step 6: The centripetal force on the pedal is about 4.52-5.07 N.'
    src(x, 'web_search', 1)['key_information'] = 'typical road pedal weight 250-280 g per pair, i.e. 125-140 g (0.125-0.140 kg) per pedal'
    out = code_src(x, "v = 2.4804; r = 0.170; a = v*v/r\nfor m in (0.125, 0.140):\n    print(f'm = {m:.3f} kg: centripetal force = {m*a:.2f} N')")
    x['answer'] = '4.52–5.07 N'
    x['answer_range'] = [4.52, 5.07]
    return out


def b17_0(x):
    x['question'] = x['question'].replace(
        'Using the typical range for the kinetic friction coefficient between soil and soil surfaces,',
        'Using the typical engineering range of the soil friction factor as the kinetic friction coefficient,')
    st = x['annotated_solution']
    st[1] = st[1].replace('μ is the soil-on-soil kinetic friction coefficient', 'μ is the soil friction factor used as the kinetic friction coefficient')
    st[2] = ("Step 3: Use web_search(query='soil friction factor typical range') to retrieve the typical range of the soil "
             "friction factor. The search result from eng-tips.com states 'We usually see a range of 0.3 to 0.5'.")
    s = src(x, 'web_search')
    s['search_query'] = 'soil friction factor typical range'
    s['verified_quote'] = 'We usually see a range of 0.3 to 0.5'
    x['hidden_cv_chosen'] = ['soil friction factor = 0.3-0.5']
    return run(src(x, 'code_executor')['code'])


def b26_0(x):
    x['question'] = ('In the video observed from 0.0-4.0 seconds, a golden honey stream pours downward at an angle of 90° '
                     'relative to horizontal over chopped nut toppings. Using typical honey properties, the true density of '
                     'almond kernels and the size of medium diced almonds, calculate the terminal speed (in mm/s) of a chopped '
                     'almond piece placed in the honey, and state whether it sinks or rises. Give the speed as a range.')
    x['question_type'] = 'TYPE_B'
    st = x['annotated_solution']
    st[4] = ("Step 5: Use web_search(query='almond kernel true density kg/m3') to retrieve the true density of almond kernels. "
             "From Aydin (2003): the true density of almond kernels increased from 900 to 995 kg/m³.")
    st[6] = ("Step 7: Apply Stokes' law: diameter = 6.0 mm → r = 0.003 m, g = 9.81 m/s², ρ_almond = 900-995 kg/m³, "
             "ρ_honey = 1415 kg/m³, η = 14.095 Pa·s.")
    st[7] = ('Step 8: Use code_executor to compute v_t = (2 × (0.003)² × 9.81 × (ρ_almond - 1415)) / (9 × 14.095): '
             '-0.717 mm/s for 900 kg/m³ and -0.585 mm/s for 995 kg/m³. The negative sign means the almond piece rises '
             '(almond kernel density < honey density) at a terminal speed of about 0.585-0.717 mm/s.')
    s = src(x, 'web_search', 2)
    s.update({'url': 'https://www.scribd.com/document/520134770/aydin2003',
              'key_information': 'true density of almond kernels = 900-995 kg/m³',
              'search_query': 'almond kernel true density kg/m3',
              'verified_quote': 'while that for kernels increased from 900 to 995 kg/m3'})
    out = code_src(x, "g = 9.81\nrho_h = 1415\neta = 14.095\ndiameter_mm = 6.0\nr = diameter_mm / 1000.0 / 2.0\n"
                      "for rho_a in (900, 995):\n    v_t = (2 * r**2 * g * (rho_a - rho_h)) / (9 * eta)\n"
                      "    print(f'rho_almond = {rho_a} kg/m^3: v_t = {v_t*1000:.3f} mm/s')")
    x['answer'] = 'The almond piece rises (it is less dense than honey) at a terminal speed of about 0.585–0.717 mm/s.'
    x['answer_range'] = [0.585, 0.717]
    return out


def b15_1(x):
    st = x['annotated_solution']
    st[4] = ("Step 5: Use web_search(query='regulation billiard ball mass') to retrieve the mass of a standard numbered "
             "billiard ball: the cue ball usually weighs 0.17 kg while all the other balls usually weigh 0.16 kg.")
    st[6] = ('Step 7: Use code_executor with θ = 15°, g = 9.81 m/s², m = 0.16 kg (numbered 9-ball) and the retrieved '
             'diameter to compute τ = (2/7)*m*g*R*sin(θ). The code gives τ = 0.003317 N·m ≈ 0.00332 N·m.')
    s = src(x, 'web_search')
    s['key_information'] = 'numbered billiard ball mass = 0.16 kg (cue ball 0.17 kg)'
    s['verified_quote'] = 'The cue ball usually weigh 0.17 kg (6 ounces) while all the other balls usually weighs 0.16 kg (5.5 ounces).'
    x['hidden_cv_chosen'] = ['billiard ball mass = 0.160 kg', 'billiard ball diameter = 57.15 mm']
    out = code_src(x, src(x, 'code_executor')['code'].replace('m = 0.170', 'm = 0.160'))
    x['answer'] = '0.00332 N·m'
    return out


def b2_0(x):
    x['question'] = ("In the video, a blue marble rolls down a wooden track at an angle of 15° from horizontal. Using a marble "
                     "with mass 5 grams and diameter 16 mm, and assuming rolling without slipping with negligible rolling "
                     "resistance, what is the marble's acceleration down the track?")
    x['annotated_solution'] = [
        'Step 1: Observe the video — the blue marble rolls down the wooden track at an angle of 15° from horizontal.',
        'Step 2: Identify the physical model — a solid sphere rolling without slipping down an inclined plane; rolling resistance is negligible.',
        'Step 3: For a solid sphere, I = (2/5)mR², so I/(mR²) = 0.4 and 1 + I/(mR²) = 1.4.',
        'Step 4: Use the rolling acceleration formula: a = g sinθ/(1 + I/(mR²)) = (5/7) g sinθ.',
        'Step 5: Use code_executor to compute a = 9.81 × sin 15° / 1.4 = 1.813582 m/s².']
    x['total_steps'] = 5
    x['sources'] = [s for s in x['sources'] if s['type'] != 'web_search']
    x['required_external_tools'] = ['code_executor']
    out = code_src(x, "import math\ng = 9.81  # m/s^2\ntheta_deg = 15.0\nm = 0.005  # kg\nd = 0.016  # m\nR = d / 2.0\n"
                      "I = (2.0 / 5.0) * m * R**2\ninertia_factor = I / (m * R**2)\ndenominator = 1.0 + inertia_factor\n"
                      "a = g * math.sin(math.radians(theta_deg)) / denominator\n"
                      "print(f\"inertia factor I/(mR^2): {inertia_factor:.1f}\")\nprint(f\"denominator 1 + I/(mR^2): {denominator:.1f}\")\n"
                      "print(f\"acceleration down the track: {a:.6f} m/s^2\")")
    x['answer'] = "The marble's acceleration down the track is 1.813582 m/s²."
    return out


EDITS = [('A', 19, 1, a19_1), ('A', 20, 1, a20_1), ('A', 6, 0, a6_0),
         ('B', 17, 0, b17_0), ('B', 26, 0, b26_0), ('B', 15, 1, b15_1), ('B', 2, 0, b2_0)]


def full_qas(cls, full):
    return [q for v in full for q in v['qa_pairs']] if cls == 'A' else full['qa_pairs']


def main():
    files = {k: os.path.join(FIX, f) for k, f in [('At', 'aclass_qa_test.json'), ('Bt', 'bclass_qa_test.json'),
                                                   ('Af', 'aclass_qa_dataset_cleaned.json'), ('Bf', 'bclass_qa_dataset_cleaned.json')]}
    D = {k: json.load(open(f, encoding='utf-8')) for k, f in files.items()}
    log = {}
    for cls, vi, qi, fn in EDITS:
        qid = '%s|test|%d|%d' % (cls, vi, qi)
        tq = D[cls + 't'][vi]['qa_pairs'][qi]
        fq = [q for q in full_qas(cls, D[cls + 'f']) if q['question'] == tq['question']]
        assert len(fq) == 1, (qid, len(fq))
        old = copy.deepcopy(tq)
        outs = [fn(x) for x in (tq, fq[0])]
        assert outs[0] == outs[1]
        log[qid] = {'old_question': old['question'], 'new_question': tq['question'], 'old_answer': old['answer'],
                    'new_answer': tq['answer'], 'answer_range': tq.get('answer_range'), 'question_type': tq['question_type'],
                    'code_output': outs[0], 'changed': sorted(k for k in tq if tq.get(k) != old.get(k))}
        print('==', qid, log[qid]['changed'], '\n   answer:', tq['answer'], tq.get('answer_range'), '\n   code:', outs[0].replace('\n', ' | '))
    # q14: the "amateur" edit exists only in the test file -> full dataset
    fa = full_qas('A', D['Af'])
    fset = {q['question'] for q in fa}
    for vi, v in enumerate(D['At']):
        for qi, tq in enumerate(v['qa_pairs']):
            if tq['question'] in fset:
                continue
            best = max(fa, key=lambda q: difflib.SequenceMatcher(None, q['question'], tq['question']).ratio())
            diff = sorted(k for k in set(tq) | set(best) if tq.get(k) != best.get(k))
            print('== A|test|%d|%d not in full set; closest differs in %s' % (vi, qi, diff))
            assert diff == ['question'], diff
            log['A|test|%d|%d' % (vi, qi)] = {'old_question': best['question'], 'new_question': tq['question'], 'changed': ['question'],
                                              'note': 'q14 amateur edit propagated to the full dataset'}
            best['question'] = tq['question']
    if DRY:
        return
    for k, f in files.items():
        bak = f.replace('.json', '.before_fix7.json')
        if not os.path.exists(bak):
            shutil.copy(f, bak)
        json.dump(D[k], open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=2 if k[1] == 'f' else 2)
    json.dump(log, open(os.path.join(FIX, 'evaluation', 'fix7_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written; log evaluation/fix7_log.json')


if __name__ == '__main__':
    main()
