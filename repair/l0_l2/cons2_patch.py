# Provenance (TriPhysBench release): fix_code/_v8_work/cons2_patch.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""CONS2_PATCH: manual review of cons2_results.json (rev_cons2.txt / rev_cons2_ctx.txt), deterministic patches:
- ok rewrites accepted as they are (ACCEPT) or with a patch (source quote / restored source);
- failed rewrites: 4 failed only on the 'old value still used' check, a false positive of near() (checked by hand,
  accepted unchanged); 5 failed because a single evidence value left a zero-width range (or no range) in a range-type
  answer: rebuilt on a range source that contains the canonical value (range of the source, or the two values of two
  sources of the same quantity), so the QA of the video agree and the answer form is kept;
- rewrites that kept a value given in the question text (marble mass, sphere diameter, 4000 rpm): rebuilt on the
  pre-CONS2 version, the inconsistent number removed from the question (mass cancels / value now retrieved);
- REJECT: the reviewer-LLM merged two different quantities (honey: observed 0.5 mm bubbles vs typical aeration bubble;
  epoxy: mixed viscosity vs Part A viscosity): CONS2 rejected, the pre-CONS2 version stands.
Every patched / accepted QA is checked again with final_recheck.one; the review verdicts go to review_post.json (CONS2).
change = 'scs': question and answer unchanged; 'full': question / answer changed (full rerun).
-> cons2_results.json (backup cons2_results.before_patch.json), review_post.json
Usage: python cons2_patch.py"""
import copy, json, os, shutil
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, get_steps, set_steps, sources_of, run_code
from current import current
from final_recheck import one
from fix39 import logged, step_sub, swap, code_sub

FP = 'old-value check false positive (near() scale match); value, steps and code checked by hand'
ACCEPT = {
    'A|72784203-1894-492b-b97d-3854eceae191|0': 'average adult male weight 90.6 kg',
    'A|c9230a3e-6f26-4f2f-bbfa-9daedb658568|0': 'hand mass 0.4 kg (question rewritten to the hand mass)',
    'A|72cfc694-5d6f-42cc-918a-eb1ec0086896|2': 'front wheel 1.2 lb = 0.544 kg',
    'A|7d2a3fcc-a2b1-452b-923d-e819a9e8c84f|0': '327.9 ± 83.9 g -> 2.86-4.83 J',
    'B|a00102/01328|0': '3.7°',
    'B|a00384/00483|1': '64.061 g',
    'B|a00315/00726|0': '22 g propane -> 0.59 kg of air',
    'test:B|a00495/00248|0': '1.5 m',
    'B|a00819/01030|0': '28 mm -> 2.324 L (bottle volume physically implausible, as the original)',
    'test:B|a00385/00050|2': '9.072 g -> 42.42-79.19 N',
    'test:B|a00616/00517|0': '6 ft midpoint -> 0.96 m',
    'test:B|a00209/00620|0': '17.38 N/m'}
FALSE_POS = {'B|a00257/00210|2': 'flagged "10" is mu 0.01; marble 4-5 g',
             'B|a00388/00303|0': 'flagged "10" is not the mass; midpoint 4.5 g -> 0.000867 N',
             'B|a00816/00706|2': 'flagged "0.8" matches the 800 °C temperature; emissivity 0.68',
             'test:B|a00523/01163|1': 'flagged "20" matches 2000; spatula width 19 mm'}
REJECT = {'B|a00818/01712|0': 'QA 0 gives the observed average diameter 0.5 mm of the tiny bubbles in the question; '
                              'QA 1 retrieves a typical (aeration) bubble size: different quantities, pre-CONS2 kept',
          'B|a00722/00179|0': 'QA 0 uses the initial mixed viscosity of the epoxy (0.5-1.5 Pa·s), QA 1 the datasheet '
                              'Part A viscosity (200 Pa·s): different quantities, pre-CONS2 kept'}
PATCH = {}


def P(qid, base, change, note):
    """base: 'cons2' (the CONS2 rewrite) or 'pre' (the version without CONS2)"""
    def deco(f):
        PATCH[qid] = (f, base, change, note)
        return f
    return deco


def out_of(qa):
    return run_code(sources_of(qa, 'code_executor')[0]['code'])['out']


def has(o, *ts):
    for t in ts:
        assert t in o, (t, o)


def add_ws(qa, src):
    """append a copy of web_search source src after the last web_search source"""
    i = max(k for k, s in enumerate(qa['sources']) if s.get('type') == 'web_search')
    qa['sources'].insert(i + 1, copy.deepcopy(src))


def set_code(qa, code, key=None):
    c = sources_of(qa, 'code_executor')[0]
    c['code'] = code
    if key:
        c['key_information'] = key


def new_steps(qa, steps):
    set_steps(qa, steps, get_steps(qa)[1])
    if 'total_steps' in qa:
        qa['total_steps'] = len(steps)


PRE = None   # current(skip=('CONS2',)), set in main


def ws(qid, j):
    return sources_of(PRE[qid][0], 'web_search')[j]


# ---------------------------------------------------------------- accepted rewrites with a patch
@P('B|a00315/00726|1', 'cons2', 'full', 'heat of combustion 46.35 MJ/kg (LHV) -> 4164 °C; source swapped to the '
                                        'Wikipedia table stating 46.35 (the grainger quote said 46.296)')
def _(qa):
    swap(qa, 0, 'propane heat of combustion MJ/kg Wikipedia', 'wikipedia.org/wiki/Heat_of_combustion',
         'Propane, 50.33, 21641, 2219.2, 46.35.',
         key='propane heat of combustion (lower heating value) = 46.35 MJ/kg (HHV 50.33 MJ/kg)')
    sources_of(qa, 'web_search')[0].pop('url_annotated', None)
    assert qa['hidden_cv_chosen'][1].startswith('stoichiometric propane mass fraction')
    qa['hidden_cv_chosen'][1] = 'stoichiometric propane mass fraction in air-propane mixture = 0.0645'


@P('A|73935c70-6d88-4f4d-a4f3-9baf70afa5b1|2', 'cons2', 'full',
   'rear wheel 1.12 kg -> 5.81 J; stature source (quote "1- Height ... 1745") restored from the pre-CONS2 version')
def _(qa):
    s = sources_of(qa, 'web_search')
    old = ws('A|73935c70-6d88-4f4d-a4f3-9baf70afa5b1|2', 1)
    assert old['verified_quote'].startswith('1- Height') and s[2]['key_information'].startswith('50th percentile male stature')
    s[2].clear()
    s[2].update(copy.deepcopy(old))


# ---------------------------------------------------------------- number removed from the question (pre-CONS2 base)
@P('B|a00165/00056|0', 'pre', 'full', 'marble mass 4 g removed from the question (it cancels; QA 1 uses 5 g); '
                                      'answer -0.332 m/s² unchanged')
def _(qa):
    old = 'The marble has a mass of 4 grams and diameter of 1.5 cm'
    assert old in qa['question']
    qa['question'] = qa['question'].replace(old, 'The marble has a diameter of 1.5 cm')
    step_sub(qa, 'marble mass m = 0.004 kg, diameter', 'marble diameter')


@P('B|a00137/00021|0', 'pre', 'full', 'marble mass 5.4 g removed from the question (it cancels; QA 1/2 use 5.71 g); '
                                      'answer unchanged')
def _(qa):
    old = 'The marble has a diameter of 16 mm and a mass of 5.4 grams.'
    assert old in qa['question']
    qa['question'] = qa['question'].replace(old, 'The marble has a diameter of 16 mm.')


@P('test:B|a00385/00050|1', 'pre', 'full',
   'desk-toy cradle: sphere mass 9.072 g (QA 0 source) and string length 0.30 m (QA 2 source); "diameter 6.8 cm" and '
   '"perfectly elastic" removed from the question; PE 0.001944 J, 32.78 % lost')
def _(qa):
    q = qa['question']
    for a, b in ((' with chrome steel spheres of diameter 6.8 cm is shown', ' with chrome steel spheres is shown'),
                 ('Using conservation of energy principles for pendulum motion and assuming perfectly elastic '
                  'collisions between identical spheres, calculate',
                  'Using conservation of energy principles for pendulum motion, treating the spheres as identical, '
                  'and using typical specifications for the sphere mass and the suspension string length, '
                  'calculate')):
        assert q.count(a) == 1, a
        q = q.replace(a, b)
    qa['question'] = q
    m_src, l_src = ws('test:B|a00385/00050|0', 0), ws('test:B|a00385/00050|2', 1)
    assert '9.072 Grams' in m_src['verified_quote'] and '30cm' in l_src['verified_quote']
    s = sources_of(qa, 'web_search')
    for k, src in enumerate((m_src, l_src)):
        s[k].clear()
        s[k].update(copy.deepcopy(src))
    s[0]['key_information'] = 'chrome steel ball weight (each): .32 ounces / 9.072 grams; sphere mass used = 0.009072 kg'
    s[1]['key_information'] = "Newton's cradle thread length = 30 cm; pendulum length used = 0.30 m"
    qa['hidden_cv_chosen'] = ['sphere mass = 0.009072 kg', 'pendulum length = 0.30 m']
    set_code(qa, "import math\n"
                 "m = 0.009072; L = 0.30; g = 9.81\n"
                 "t1 = math.radians(22); t2 = math.radians(18)\n"
                 "h1 = L * (1 - math.cos(t1))\n"
                 "h2 = L * (1 - math.cos(t2))\n"
                 "PE1 = m * g * h1\n"
                 "PE2 = m * g * h2\n"
                 "loss_pct = (PE1 - PE2) / PE1 * 100\n"
                 "print(f'h1 = {h1:.5f} m, h2 = {h2:.5f} m')\n"
                 "print(f'PE1 = {PE1:.6f} J, PE2 = {PE2:.6f} J, Energy loss = {loss_pct:.2f}%')\n",
             key='PE1 = 0.001944 J, PE2 = 0.001307 J, energy loss = 32.78%')
    has(out_of(qa), 'h1 = 0.02184 m, h2 = 0.01468 m', 'PE1 = 0.001944 J, PE2 = 0.001307 J, Energy loss = 32.78%')
    new_steps(qa, [
        'Step 1: Observe the video — the left sphere is deflected at 22° from vertical (4.0–8.0 s), and the right '
        'sphere reaches 18° after the collision sequence.',
        'Step 2: Identify the physical phenomenon — pendulum motion converting gravitational potential energy to '
        'kinetic energy, with the collisions transferring the energy along the row of identical spheres.',
        "Step 3: Use web_search(query='%s') to retrieve the sphere mass: a chrome steel ball weighs .32 ounces / "
        "9.072 grams each, so m = 0.009072 kg." % m_src['search_query'],
        "Step 4: Use web_search(query='%s') to retrieve the suspension string length: 30 cm, so L = 0.30 m."
        % l_src['search_query'],
        'Step 5: The height gained by a pendulum deflected by angle θ is h = L(1 − cosθ). For θ₁ = 22°, '
        'h₁ = 0.30 × (1 − cos22°) = 0.02184 m.',
        'Step 6: Gravitational potential energy PE = mgh: PE₁ = 0.009072 × 9.81 × 0.02184 = 0.001944 J.',
        'Step 7: For the right sphere at θ₂ = 18°, h₂ = 0.30 × (1 − cos18°) = 0.01468 m, so PE₂ = 0.009072 × 9.81 × '
        '0.01468 = 0.001307 J.',
        'Step 8: Use code_executor to compute the energy loss percentage = (PE₁ − PE₂)/PE₁ × 100 = 32.78% (the mass '
        'and length cancel in the percentage).'])
    qa['answer'] = ('The gravitational potential energy of the left sphere at 22° deflection is 0.001944 J. Based on '
                    'the right sphere reaching only 18° instead of 22°, approximately 32.78% of the energy is lost '
                    'during the collision sequence.')


@P('B|a00390/00058|0', 'pre', 'full', '"spins at approximately 4000 rpm" removed from the question; rotation speed '
                                      '150 rpm retrieved (QA 1 source); radius 0.35 m observed (outer wheel, QA 1: rim '
                                      'bead seat 0.311 m); L = 3.85 kg·m²/s')
def _(qa):
    old = 'The wheel has a radius of 0.35 m and spins at approximately 4000 rpm. Using typical bicycle wheel ' \
          'specifications, what'
    assert old in qa['question']
    qa['question'] = qa['question'].replace(old, 'The wheel has a radius of 0.35 m. Using typical bicycle wheel '
                                                 'specifications and a typical wheel rotation speed, what')
    rpm = ws('B|a00390/00058|1', 2)
    assert '150 rpm' in rpm['verified_quote']
    add_ws(qa, rpm)
    sources_of(qa, 'web_search')[0]['key_information'] = 'road bike wheels weigh between 1000 and 2000 g; mass used ' \
                                                         '= 2 kg (upper end, heavy demonstration wheel)'
    set_code(qa, "import math\n"
                 "radius = 0.35\n"
                 "rpm = 150\n"
                 "omega = (rpm * 2 * math.pi) / 60\n"
                 "mass = 2.0\n"
                 "I = mass * radius**2\n"
                 "L = I * omega\n"
                 "print(f'omega = {omega:.3f} rad/s, I = {I:.4f} kg·m²')\n"
                 "print(f'Angular momentum: {L:.2f} kg·m²/s')\n",
             key='omega = 15.708 rad/s, I = 0.245 kg·m², L = 3.85 kg·m²/s')
    has(out_of(qa), 'omega = 15.708 rad/s, I = 0.2450', 'Angular momentum: 3.85')
    new_steps(qa, [
        'Step 1: Observe the video — a bicycle wheel of radius 0.35 m spins around its axle in a gyroscopic '
        'demonstration.',
        'Step 2: Identify the physical phenomenon — rotational motion with angular momentum L = Iω, where I is the '
        'moment of inertia and ω the angular velocity.',
        "Step 3: Use web_search(query='%s') to retrieve the mass of a bicycle wheel: road bike wheels weigh between "
        "1000 and 2000 g; use m = 2 kg for the heavy demonstration wheel." % sources_of(qa, 'web_search')[0]['search_query'],
        "Step 4: Use web_search(query='%s') to retrieve a typical wheel rotation speed: about 150 rpm."
        % rpm['search_query'],
        'Step 5: Compute the moment of inertia with the thin-hoop approximation: I = m r² = 2 kg × (0.35 m)² = '
        '0.245 kg·m².',
        'Step 6: Convert the rotation speed: ω = 150 rev/min × 2π rad/rev ÷ 60 s/min = 15.708 rad/s.',
        'Step 7: Use code_executor to calculate L = I ω = 0.245 kg·m² × 15.708 rad/s = 3.85 kg·m²/s.'])
    qa['answer'] = '3.85 kg·m²/s'


# ---------------------------------------------------------------- range answers rebuilt on a range source
@P('B|a00859/00190|1', 'cons2', 'full', 'eraser mass = 3-5 g as quoted (contains QA 0 5 g); force difference at the '
                                        'same drop height (m_pencil - m_eraser)·v/Δt = 0.000-10.741 N')
def _(qa):
    s = sources_of(qa, 'web_search')[0]
    assert s['verified_quote'] == 'a standard pencil eraser weighs about 3-5 grams'
    s['key_information'] = 'a standard pencil eraser weighs about 3-5 grams; used as 0.003-0.005 kg'
    qa['hidden_cv_chosen'][0] = 'pencil eraser mass = 3-5 g'
    set_code(qa, "import math\n\n"
                 "g = 9.81\nh_min = 0.08\nh_max = 0.12\ndt = 0.001\n\n"
                 "m_eraser_min = 0.003\nm_eraser_max = 0.005\nm_pencil_min = 0.005\nm_pencil_max = 0.010\n\n"
                 "v_min = math.sqrt(2*g*h_min)\nv_max = math.sqrt(2*g*h_max)\n\n"
                 "F_pencil_min = m_pencil_min * v_min / dt\nF_pencil_max = m_pencil_max * v_max / dt\n"
                 "F_eraser_min = m_eraser_min * v_min / dt\nF_eraser_max = m_eraser_max * v_max / dt\n\n"
                 "# same drop height for both: difference = (m_pencil - m_eraser) * v / dt\n"
                 "diff_min = (m_pencil_min - m_eraser_max) * v_min / dt\n"
                 "diff_max = (m_pencil_max - m_eraser_min) * v_max / dt\n\n"
                 "print(f'Impact speed range: {v_min:.4f}-{v_max:.4f} m/s')\n"
                 "print(f'Whole wooden pencil force range: {F_pencil_min:.3f}-{F_pencil_max:.3f} N')\n"
                 "print(f'Loose eraser alone force range: {F_eraser_min:.3f}-{F_eraser_max:.3f} N')\n"
                 "print(f'Difference (pencil - eraser alone) range: {diff_min:.3f}-{diff_max:.3f} N')\n",
             key='pencil 6.264-15.344 N, eraser 3.759-7.672 N, difference 0.000-10.741 N')
    has(out_of(qa), 'Whole wooden pencil force range: 6.264-15.344 N', 'Loose eraser alone force range: 3.759-7.672 N',
        'Difference (pencil - eraser alone) range: 0.000-10.741 N')
    steps, kind = get_steps(qa)
    steps[2] = ("Step 3: Use web_search(query='%s') to retrieve the loose pencil eraser mass. Evidence: a standard "
                "pencil eraser weighs about 3-5 grams; use m_eraser = 0.003-0.005 kg." % s['search_query'])
    steps[5] = ('Step 6: Run code_executor with g = 9.81 m/s², h_min = 0.08 m, h_max = 0.12 m, Δt = 0.001 s, m_pencil '
                '= 0.005-0.010 kg, m_eraser = 0.003-0.005 kg. It computes v_min = 1.2528 m/s, v_max = 1.5344 m/s; whole '
                'wooden pencil force = 6.264-15.344 N; loose eraser alone force = 3.759-7.672 N. For the same drop '
                'height the difference is (m_pencil − m_eraser)·v/Δt, from (0.005 − 0.005)·v_min/Δt = 0.000 N to '
                '(0.010 − 0.003)·v_max/Δt = 10.741 N.')
    steps[6] = ('Step 7: Result: whole wooden pencil force range 6.264-15.344 N; loose eraser alone force range '
                '3.759-7.672 N; the whole pencil produces the larger average impact force (equal only when both '
                'masses are 5 g); difference (whole pencil minus eraser alone) range 0.000-10.741 N.')
    set_steps(qa, steps, kind)
    qa['answer'] = ('Whole wooden pencil force range: 6.264-15.344 N; loose eraser alone force range: 3.759-7.672 N; '
                    'the whole pencil produces the larger average impact force (equal only when both masses are 5 g); '
                    'difference (whole pencil minus eraser alone, same drop height) range: 0.000-10.741 N.')
    qa['answer_range'] = [0.0, 10.741]


@P('A|f1af22f7-699d-4961-bb21-4a9030640f4e|1', 'pre', 'full',
   'player mass = NCAA 91.3 ± 11.1 kg (QA 0 source, 80.2-102.4 kg contains the QA 2 98 kg) -> 387.95-495.34 J')
def _(qa):
    src = ws('A|f1af22f7-699d-4961-bb21-4a9030640f4e|0', 0)
    assert src['verified_quote'] == 'weight, 91.3 + 11.1 kg'
    s = sources_of(qa, 'web_search')[0]
    s.clear()
    s.update(copy.deepcopy(src))
    code_sub(qa, 'mass_range = [62.93, 74.95, 86.97]', 'mass_range = [80.2, 91.3, 102.4]')
    o = out_of(qa)
    has(o, 'Maximum height above takeoff = 0.4931 m', 'PE at mass 80.20 kg = 387.95 J', 'PE at mass 91.30 kg = 441.64 J',
        'PE at mass 102.40 kg = 495.34 J')
    sources_of(qa, 'code_executor')[0]['key_information'] = 'h_max = 0.4931 m; PE = 387.95-495.34 J (441.64 J at 91.3 kg)'
    steps, kind = get_steps(qa)
    steps[2] = ("Step 3: Use web_search(query='%s') to retrieve the typical player mass: NCAA Division I players have "
                "weight = 91.3 + 11.1 kg; interpret the quoted dispersion as ±11.1 kg, giving the mass range "
                "80.2–102.4 kg." % src['search_query'])
    steps[6] = 'Step 7: Result: PE = 387.95–495.34 J (about 441.64 J at the mean mass 91.3 kg).'
    set_steps(qa, steps, kind)
    qa['answer'] = '387.95–495.34 J (with average around 441.64 J)'
    qa['answer_range'] = [387.95, 495.34]


@P('B|a00138/01103|1', 'pre', 'full', 'washer mass = 5-10 g (QA 0 source; contains the QA 2 6.27 g) -> 1.77-3.53 J')
def _(qa):
    src = ws('B|a00138/01103|0', 0)
    assert '5 - 10 grams' in src['verified_quote']
    s = sources_of(qa, 'web_search')[0]
    s.clear()
    s.update(copy.deepcopy(src))
    s['key_information'] = 'small DIN 125 plain washer mass = 5-10 g (0.005-0.010 kg)'
    qa['hidden_cv_chosen'][0] = 'steel washer mass = 5-10 g'
    code_sub(qa, 'mass_min, mass_max = 0.01, 0.02', 'mass_min, mass_max = 0.005, 0.010')
    has(out_of(qa), 'Total mass = 0.09-0.18 kg', 'W_min = 1.77 J', 'W_max = 3.53 J',
        'Final answer range = 1.77-3.53 J (0.00049-0.00098 Wh)')
    sources_of(qa, 'code_executor')[0]['key_information'] = 'total mass 0.09-0.18 kg; energy 1.77-3.53 J (0.00049-0.00098 Wh)'
    steps, kind = get_steps(qa)
    steps[2] = ("Step 3: Use web_search(query='%s') to retrieve washer mass specifications: a small DIN 125 plain "
                "washer weighs around 5-10 grams (0.005-0.010 kg)." % src['search_query'])
    steps[4] = 'Step 5: Calculate the total mass range: minimum mass = 18 × 0.005 kg = 0.09 kg, maximum mass = 18 × 0.010 kg = 0.18 kg.'
    steps[6] = 'Step 7: Minimum work = 0.09 kg × 9.81 m/s² × 2.0 m = 1.77 J.'
    steps[7] = 'Step 8: Maximum work = 0.18 kg × 9.81 m/s² × 2.0 m = 3.53 J.'
    steps[8] = ('Step 9: Assuming 100% efficiency, electrical energy required equals the gravitational work done, so the '
                'answer range is 1.77-3.53 joules (0.00049-0.00098 watt-hours).')
    set_steps(qa, steps, kind)
    qa['answer'] = '1.77-3.53 joules (0.00049-0.00098 watt-hours)'
    qa['answer_range'] = [1.77, 3.53]


@P('test:A|ac259bd8-f9a1-4456-99b3-610f80351c06|1', 'pre', 'full',
   '15-year-old boys 91-182 lbs = 41.28-82.55 kg (same source as QA 2: average 124 lbs) -> 173.6-347.1 J')
def _(qa):
    q = 'average weight 15 year old boy healthy range'
    quote = ('weight measurements for 15 year old boys will be in the range between 91 and 182 lbs. The average '
             'weight for 15 year old boys is 124 lbs')
    swap(qa, 0, q, 'lifemeasure.com/weight-charts/boys/15-years-old', quote,
         key='15-year-old boys weigh 91-182 lbs (41.28-82.55 kg), average 124 lbs (56.25 kg)')
    sources_of(qa, 'web_search')[0].pop('url_annotated', None)
    set_code(qa, "v = 2.9  # m/s (hidden CV value)\n"
                 "LB = 0.45359237\n"
                 "m_min, m_avg, m_max = 91 * LB, 124 * LB, 182 * LB  # kg (from web search, lbs)\n"
                 "ke = lambda m: 0.5 * m * v**2\n"
                 "print(f'Mass range: {m_min:.2f} to {m_max:.2f} kg (average {m_avg:.2f} kg)')\n"
                 "print(f'Kinetic energy range: {ke(m_min):.1f} to {ke(m_max):.1f} J')\n"
                 "print(f'For average mass: KE = {ke(m_avg):.1f} J')\n",
             key='mass 41.28-82.55 kg; KE 173.6-347.1 J (236.5 J at 56.25 kg)')
    has(out_of(qa), 'Mass range: 41.28 to 82.55 kg (average 56.25 kg)', 'Kinetic energy range: 173.6 to 347.1 J',
        'For average mass: KE = 236.5 J')
    steps, kind = get_steps(qa)
    steps[3] = ('Step 4: From search results: 15-year-old boys weigh between 91 and 182 lbs (41.28–82.55 kg), average '
                '124 lbs (56.25 kg). Use this range.')
    steps[4] = ('Step 5: Use code_executor to compute kinetic energy for the mass range 41.28–82.55 kg with speed '
                '2.9 m/s: KE = 0.5 × m × (2.9)².')
    steps[5] = 'Step 6: Result: KE = 173.6–347.1 J (236.5 J for the average mass 56.25 kg).'
    set_steps(qa, steps, kind)
    qa['answer'] = '173.6–347.1 J (range based on athlete mass 41.28–82.55 kg). For average mass 56.25 kg: 236.5 J.'
    qa['answer_range'] = [173.6, 347.1]


@P('B|a00374/00062|2', 'pre', 'full', 'spout diameter = 18.5-25.4 mm from the two sources of the opening (bottle neck '
                                      '18.5/21.5 mm; decanter mouth 1 in = 25.4 mm, QA 1 source) -> 3.0-5.6 s')
def _(qa):
    mouth = ws('B|a00374/00062|1', 0)
    assert 'Mouth is 1' in mouth['verified_quote']
    add_ws(qa, mouth)
    assert qa['hidden_cv_chosen'][2] == 'spout diameter = 18.5-21.5 mm'
    qa['hidden_cv_chosen'][2] = 'spout diameter = 18.5-25.4 mm'
    code_sub(qa, 'neck_diameter_max = 0.0215', 'neck_diameter_max = 0.0254')
    o = out_of(qa)
    has(o, 'Maximum spout area = 0.000507 m^2', 'Minimum emptying time = 3.0 s', 'Maximum emptying time = 5.6 s')
    sources_of(qa, 'code_executor')[0]['key_information'] = 'emptying time 3.0-5.6 s (spout 25.4-18.5 mm)'
    steps, kind = get_steps(qa)
    steps[3] = ("Step 4: Use web_search(query='%s') to retrieve the spirits bottle neck sizes 18.5 mm and 21.5 mm, and "
                "web_search(query='%s') to retrieve the decanter mouth width 1 inch (25.4 mm). Spout diameter range: "
                "18.5–25.4 mm = 0.0185–0.0254 m." % (sources_of(qa, 'web_search')[1]['search_query'],
                                                     mouth['search_query']))
    steps[6] = ('Step 7: Calculate spout area range: a_min = π × (0.0185/2)² = 0.000269 m², a_max = π × (0.0254/2)² = '
                '0.000507 m².')
    steps[9] = ('Step 10: For maximum spout area (25.4 mm): t_min = (2 × 0.0079 × √0.178) / (0.000507 × √(2×9.81)) = '
                '3.0 seconds.')
    set_steps(qa, steps, kind)
    qa['answer'] = 'Approximately 3.0 to 5.6 seconds'
    qa['answer_range'] = [3.0, 5.6]


@P('B|a00323/00577|1', 'pre', 'full', 'toy excavator mass = 0.408-0.92 kg from the two mass sources (Walmart 0.9 lbs, '
                                      'QA 0 source; Bruder 920 g) -> submerged fraction 0.0340-0.0767')
def _(qa):
    wm = ws('B|a00323/00577|0', 0)
    assert wm['verified_quote'] == 'Weight: 0.9 lbs'
    s = sources_of(qa, 'web_search')
    qa['sources'].insert(qa['sources'].index(s[0]), copy.deepcopy(wm))
    s = sources_of(qa, 'web_search')
    s[0]['key_information'] = 'plastic toy excavator weight 0.9 lbs = 0.408 kg'
    qa['hidden_cv_chosen'][0] = 'toy excavator mass = 0.408-0.92 kg'
    set_code(qa, "masses_kg = (0.408, 0.92)\n"
                 "length_m = 0.409\nwidth_m = 0.163\nheight_m = 0.18\nwater_density = 1000\n"
                 "volume_m3 = length_m * width_m * height_m\n"
                 "print(f'Volume: {volume_m3:.4f} m³')\n"
                 "for mass_kg in masses_kg:\n"
                 "    density_kg_m3 = mass_kg / volume_m3\n"
                 "    submerged_fraction = density_kg_m3 / water_density\n"
                 "    print(f'mass {mass_kg} kg: effective density {density_kg_m3:.1f} kg/m³, submerged fraction '\n"
                 "          f'{submerged_fraction:.4f} ({submerged_fraction*100:.2f}%)')\n",
             key='submerged fraction 0.0340-0.0767 (3.40-7.67%)')
    has(out_of(qa), 'mass 0.408 kg: effective density 34.0 kg/m³, submerged fraction 0.0340 (3.40%)',
        'mass 0.92 kg: effective density 76.7 kg/m³, submerged fraction 0.0767 (7.67%)')
    new_steps(qa, [
        'Step 1: Observe the video — the toy excavator is floating in water, indicating its average density is less '
        'than water density.',
        'Step 2: Identify the physics phenomenon — buoyancy equilibrium: buoyant force equals weight, so ρ_fluid × '
        'V_submerged × g = m_object × g.',
        'Step 3: For a floating object with uniform cross-section, the submerged fraction of height equals the density '
        'ratio: h_submerged/h_total = ρ_object/ρ_fluid.',
        "Step 4: Use web_search(query='%s') to retrieve a plastic toy excavator mass: 0.9 lbs = 0.408 kg, and "
        "web_search(query='%s') to retrieve a toy excavator mass of 920 g = 0.92 kg; typical mass range 0.408–0.92 kg."
        % (wm['search_query'], s[1]['search_query']),
        "Step 5: Use web_search(query='%s') to retrieve typical dimensions = 40.9 cm × 16.3 cm × 18 cm (height = 18 "
        "cm = 0.18 m)." % s[2]['search_query'],
        'Step 6: Calculate volume: V = 0.409 m × 0.163 m × 0.18 m = 0.0120 m³.',
        'Step 7: Calculate effective density: ρ_eff = m/V = 0.408–0.92 kg / 0.0120 m³ = 34.0–76.7 kg/m³.',
        'Step 8: Calculate submerged fraction: f_sub = ρ_eff/ρ_water = 0.0340–0.0767.',
        'Step 9: Using code_executor to compute precisely: submerged fraction = 0.0340–0.0767 (3.40–7.67%).'])
    qa['answer'] = '0.0340–0.0767 (3.40–7.67% of total height submerged)'
    qa['answer_range'] = [0.034, 0.0767]


def main():
    global PRE
    path = WORK + '/cons2_results.json'
    bak = WORK + '/cons2_results.before_patch.json'
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    r = json.load(open(bak, encoding='utf-8'))
    PRE = current(skip=('CONS2',))
    assert not set(ACCEPT) & set(FALSE_POS) and set(r) == set(ACCEPT) | set(FALSE_POS) | set(REJECT) | set(PATCH), \
        set(r) ^ (set(ACCEPT) | set(FALSE_POS) | set(REJECT) | set(PATCH))
    notes = {q: 'manual review: accepted (%s)' % n for q, n in ACCEPT.items()}
    for q, n in ACCEPT.items():
        assert r[q]['status'] == 'ok', q
        r[q]['change'] = 'full'
    for q, n in FALSE_POS.items():
        assert r[q]['status'] == 'fail', q
        r[q].update(status='ok', change='full')
        notes[q] = '%s: %s' % (FP, n)
    for q, n in REJECT.items():
        r[q].update(status='reject', change=None)
        notes[q] = 'manual review: rejected, ' + n
    for q, (f, base, change, note) in PATCH.items():
        qa = copy.deepcopy(r[q]['qa'] if base == 'cons2' else PRE[q][0])
        assert qa['question_type'] == PRE[q][0]['question_type'] and qa['difficulty'] == PRE[q][0]['difficulty']
        f(qa)
        r[q].update(status='ok', qa=qa, change=change)
        notes[q] = 'manual patch: ' + note
    todo = [(q, x['qa']) for q, x in r.items() if x['status'] == 'ok']
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, todo))
    for q, x in r.items():
        x['patch'] = notes[q]
        if q in res:
            x['problems'] = res[q]
            if res[q]:
                x['status'] = 'fail'
                print('PROBLEM', q, res[q])
            p = PRE[q][0]
            assert x['qa']['question_type'] == p['question_type'] and x['qa']['difficulty'] == p['difficulty'], q
            assert len(x['qa'].get('hidden_cv_chosen') or []) == len(p.get('hidden_cv_chosen') or []), q
            if q.split(':')[-1].startswith('A|'):
                assert x['qa'].get('hidden_cv_chosen') == p.get('hidden_cv_chosen'), q
    json.dump(r, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    rp = json.load(open(WORK + '/review_post.json', encoding='utf-8'))
    for q, x in r.items():
        if x['status'] == 'ok':
            rp.setdefault(q, {})['CONS2'] = {'verdict': 'accept', 'notes': notes[q]}
        else:
            rp.get(q, {}).pop('CONS2', None)
    json.dump(rp, open(WORK + '/review_post.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(r), 'CONS2,', sum(x['status'] == 'ok' for x in r.values()), 'ok,',
          sum(x['status'] == 'reject' for x in r.values()), 'rejected,', sum(x['status'] == 'fail' for x in r.values()),
          'with problems')


if __name__ == '__main__':
    main()
