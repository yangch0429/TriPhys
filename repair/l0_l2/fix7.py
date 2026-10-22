# Provenance (TriPhysBench release): fix_code/_v8_work/fix7.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""FIX7: manual fixes of the FIX5 / FIX6 items the LLM repair could not do (a new retrieved value was needed, see
"needs_search") or did wrongly; applied after FIX6, question_type / difficulty / number of hidden values unchanged,
A-class hidden_cv_chosen unchanged. New searches are logged in search_log.jsonl (tag fix7, _srch.py).
  a00721/00666|1  the video shows a ping pong ball on the basketball, not a golf ball (ITTF 2.67-2.77 g); fall time
                  0.8 s -> 0.5 s (the 1.2 m drop of QA0)
  2429d435|1      the 62.0 kg lower mass bound was a world adult (both sexes) average -> adult male typical 60 kg
  a00138/01103|1  lift height 2.0 m -> 0.8 m as in QA0 / QA2 (same lift)
  a00087/00122|1  capacitance 0.70 pF was not the code output; smaller sphere = the 50.8 mm standard electrode as in
                  QA0 -> 1.62 pF; arcs as in QA0; garbled hidden entry cleaned
  e6d6f5e3|2      the mass of the same ball cancels in the KE ratio (the 14.4-16.4 range used different masses in
                  numerator and denominator) -> 15.3
  a00087/00028|2  invented ring section / sphere at the rim -> ring section and 15 cm sphere on the axis as in QA1
  a00611/02070|1  fork + box 200 g -> 250 g as QA0 (identical setup)
  a00611/02070|2  solid pine box (0.375 kg) -> given hollow box mass 210 g + steel prongs = 0.249 kg (QA0: 250 g)
  a00673/02402|1  diameter / splash / drop height aligned with QA0; guava diameter hidden value -> water density
  a00100/00705|1, a00588/00347|1, f9afd55f|1, 2a160c63|2   the question states the idealization / object the solution
                  uses (start from rest; contact patches at half the wheelbase; set-shot release height; multitool)
  90e697a5|0, a00818/01712|0, a00576/00194|0   wording the video contradicts (close-range shot; bubbles appear
                  almost stationary; pitcher just above the cup)
  a00611/02070|2  also: the video shows the box placed (not slid) at 12-16 s: the slide is an assumption
  a00588/00347|1  also: unused, disputed spin direction removed
  a00332/00025|1, a00870/00376|1, a00883/01067|0   values the video frames contradict (drop 0.05 m; cup 0.02 m above
                  the rim; ~100 matches)
  aab43780|2, d4a1ea83|1   observation time of the measured approach / of the jump shot
  a00138/01103|0/1, a00879/00373|0/1/2, a00087/00028|1   quantities the video cannot show are stated as assumptions
                  (the clip ends as the magnet takes up the scrap; the bench is near level, 25 deg is the on-screen
                  trajectory angle; the ring's rotation count cannot be read off the uniform ring)
  a00087/00122|0/1   the arcs are seen at 8.0-10.0 s (4.0-8.0 s shows the spheres without arc)
  a00879/00373|2  invented 95-113 rad/s^2 tolerance range -> 103.6 rad/s^2 (the mass cancels); garbled hidden cleaned
-> fix7_results.json {qid: {status, qa, change, note, problems}}
Usage: python fix7.py"""
import copy, json
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, get_steps, set_steps
from current import current
from final_recheck import one
from strict_check import problems as strict_problems
from fix4 import web, code, rebuild, has
from fix39 import step_sub

PATCH = {}


def P(qid, change, note):
    def deco(f):
        PATCH[qid] = (f, change, note)
        return f
    return deco


def q_sub(qa, old, new):
    assert qa['question'].count(old) == 1, (old, qa['question'])
    qa['question'] = qa['question'].replace(old, new)


# ---------------------------------------------------------------- a00721/00666 (basketball + ping pong ball stack)
PP_Q = 'ping pong ball standard mass grams ITTF'
PP_QUOTE = 'any weight between 2.67 and 2.77g is acceptable'


@P('B|a00721/00666|1', 'full', 'golf ball (45.5-45.9 g) -> the ping pong ball the video shows (2.67-2.77 g, '
                               'allabouttabletennis); fall time 0.8 s -> 0.5 s (1.2 m drop of QA0); '
                               '17.918-19.666 -> 8.442-9.232 m')
def _(qa):
    qa['question'] = ('In the video (visible at 4.0–8.0 s), a standard orange basketball with a ping pong ball stacked '
                      'on top drops vertically. The basketball falls for approximately 0.5 seconds before impacting the '
                      'concrete floor. Using standard ball specifications and elastic collision physics, predict the '
                      'range of maximum heights the ping pong ball will reach after the energy transfer.')
    assert len(qa['hidden_cv_chosen']) == 3 and 'golf' in qa['hidden_cv_chosen'][1]
    qa['hidden_cv_chosen'][1] = 'ping pong ball mass = 2.67-2.77 g'
    obs, bb, _, cor, _ = qa['sources']
    obs = dict(obs, key_information='basketball falls vertically for t = 0.5 s with a white ping pong ball on top')
    c = code("g = 9.81\nt = 0.5\nm_b = 0.62\nm_s_min = 0.00267\nm_s_max = 0.00277\ne_min = 0.82\ne_max = 0.88\n\n"
             "v_impact = g * t\nprint(f'Impact speed = {v_impact:.3f} m/s')\n\n"
             "v_s_min = (m_b * (1 + 2 * e_min) - m_s_max) / (m_b + m_s_max) * v_impact\n"
             "v_s_max = (m_b * (1 + 2 * e_max) - m_s_min) / (m_b + m_s_min) * v_impact\n"
             "print(f'Ping pong ball speed range = {v_s_min:.3f}-{v_s_max:.3f} m/s')\n\n"
             "h_min = v_s_min**2 / (2 * g)\nh_max = v_s_max**2 / (2 * g)\nprint(f'Minimum height = {h_min:.3f} m')\n"
             "print(f'Maximum height = {h_max:.3f} m')\nprint(f'Final answer range = {h_min:.3f}-{h_max:.3f} m')")
    has(c['key_information'], 'Final answer range = 8.442-9.232 m', 'speed range = 12.870-13.459 m/s')
    rebuild(qa, [obs, bb, web(PP_Q, 'allabouttabletennis.com', PP_QUOTE,
                              'ping pong ball mass = 2.67-2.77 g (official 2.7 g; 0.00267-0.00277 kg)',
                              ('2.67 and 2.77g', '0.00267-0.00277 kg')), cor, c], [
        'Step 1: Observe the video — the basketball falls vertically for t ≈ 0.5 s with a white ping pong ball stacked '
        'on top before the floor impact.',
        'Step 2: Identify physical phenomena — free fall, floor impact with coefficient of restitution, and 1D elastic '
        'collision between the basketball and the ping pong ball.',
        "Step 3: Use web_search(query='regulation basketball official weight') to retrieve standard basketball mass "
        'm_b = 0.62 kg.',
        "Step 4: Use web_search(query='%s') to retrieve the ping pong ball mass: any weight between 2.67 and 2.77 g is "
        'acceptable, m_s = 0.00267-0.00277 kg.' % PP_Q,
        "Step 5: Use web_search(query='concrete floor basketball impact restitution values') to retrieve "
        'basketball-concrete coefficient of restitution e = 0.82-0.88.',
        'Step 6: Compute basketball impact speed: v = g × t = 9.81 m/s² × 0.5 s = 4.905 m/s.',
        'Step 7: Model the collision: after the basketball rebounds upward at speed e v, the ping pong ball is still '
        'moving downward at v. The elastic collision gives the ping pong ball speed v_s = [m_b(1+2e) - m_s]/(m_b+m_s) '
        '× v. The minimum v_s occurs at e = 0.82 and m_s = 0.00277 kg; the maximum v_s at e = 0.88 and m_s = 0.00267 kg.',
        'Step 8: Run code_executor to compute v_s and the maximum height h = v_s²/(2g). It prints v_s_min = 12.870 m/s, '
        'v_s_max = 13.459 m/s, h_min = 8.442 m, h_max = 9.232 m.',
        'Step 9: Result: the ping pong ball reaches a maximum height range of 8.442–9.232 m.'])
    qa['answer'], qa['answer_range'] = ('The ping pong ball reaches a maximum height range of approximately '
                                        '8.442–9.232 m.', [8.442, 9.232])


# ---------------------------------------------------------------- 2429d435 (mechanic, kinetic energy)
MALE_Q = 'average body weight of adult men worldwide kg'
MALE_QUOTE = "An adult male's average weight typically ranges between 60 kg to 80 kg."


@P('A|2429d435-c4fd-4e57-a25c-d2badd786e60|1', 'full',
   'lower mass bound 62.0 kg was the world adult average of both sexes -> adult male typical lower value 60 kg '
   '(clinicone); 11.16-16.31 -> 10.80-16.31 J')
def _(qa):
    obs, us, _, _ = qa['sources']
    assert '90.6' in us['key_information']
    c = code('mass_min = 60.0\nmass_max = 90.6\npeak_speed = 0.6\n\nke_min = 0.5 * mass_min * peak_speed**2\n'
             'ke_max = 0.5 * mass_max * peak_speed**2\n\nprint(f"Minimum mass: {mass_min:.1f} kg")\n'
             'print(f"Maximum mass: {mass_max:.1f} kg")\nprint(f"Peak speed: {peak_speed:.1f} m/s")\n'
             'print(f"KE_min: {ke_min:.2f} J")\nprint(f"KE_max: {ke_max:.2f} J")\n'
             'print(f"Final answer: {ke_min:.2f}–{ke_max:.2f} J")')
    has(c['key_information'], 'Final answer: 10.80–16.31 J')
    rebuild(qa, [obs, us, web(MALE_Q, 'clinicone.com.np', MALE_QUOTE,
                              'typical adult male weight = 60-80 kg; lower value used: 60 kg', ('60 kg', '60 kg')), c], [
        'Step 1: Observe the video — at t=1.6s, the mechanic reaches a peak speed while working (this speed value must '
        'be measured from the video).',
        'Step 2: Identify the physical phenomenon — kinetic energy of a moving person: KE = ½mv².',
        "Step 3: Use web_search(query='global and American adult male average weight') to retrieve the US adult male "
        'average weight, 199.8 lbs = 90.6 kg (upper typical value).',
        "Step 4: Use web_search(query='%s') to retrieve the typical adult male weight range, 60 kg to 80 kg; its lower "
        'value 60 kg is the lower typical mass, giving a mass range of 60.0–90.6 kg.' % MALE_Q,
        'Step 5: Compute the kinetic energy range at the measured peak speed of approximately 0.6 m/s: '
        'KE_min = ½ × 60.0 kg × (0.6 m/s)² = 10.80 J; KE_max = ½ × 90.6 kg × (0.6 m/s)² = 16.31 J.',
        'Step 6: Run code_executor to confirm: it prints KE_min = 10.80 J and KE_max = 16.31 J.',
        'Step 7: Result: the peak kinetic energy is 10.80–16.31 J.'])
    qa['answer'], qa['answer_range'] = '10.80–16.31 J', [10.8, 16.31]


# ---------------------------------------------------------------- a00138/01103 (electromagnetic lifter)
@P('B|a00138/01103|1', 'full', 'lift height 2.0 m -> 0.8 m stated by QA0 / QA2 for the same lift; '
                               '1.77-3.53 -> 0.71-1.41 J')
def _(qa):
    q_sub(qa, 'to a height of 2.0 meters', 'to a height of 0.8 meters')
    q_sub(qa, "adhering to the electromagnetic lifter's bottom surface during the lifting phase (visible at 4.0-8.0 "
              "seconds)", "adhering to the electromagnetic lifter's bottom surface as it engages the scrap pile "
                          "(visible at 4.0-9.0 seconds)")
    assert qa['sources'][0]['type'] == 'video_observation'
    qa['sources'][0] = dict(qa['sources'][0], time_point='4.0-9.0 s')
    c = code('num_washers = 18\nmass_min, mass_max = 0.005, 0.010\npower_rating = 12.2\ng, h = 9.81, 0.8\n\n'
             'total_mass_min = num_washers * mass_min\ntotal_mass_max = num_washers * mass_max\n'
             'W_min = total_mass_min * g * h\nW_max = total_mass_max * g * h\nenergy_Wh_min = W_min / 3600\n'
             'energy_Wh_max = W_max / 3600\n\nprint(f"Total mass = {total_mass_min:.2f}-{total_mass_max:.2f} kg")\n'
             'print(f"Power rating = {power_rating:.1f} kW (not needed: 100% efficiency)")\n'
             'print(f"W_min = {W_min:.2f} J")\nprint(f"W_max = {W_max:.2f} J")\n'
             'print(f"Electrical energy = {energy_Wh_min:.6f}-{energy_Wh_max:.6f} Wh")\n'
             'print(f"Final answer range = {W_min:.2f}-{W_max:.2f} J ({energy_Wh_min:.6f}-{energy_Wh_max:.6f} Wh)")')
    has(c['key_information'], 'Final answer range = 0.71-1.41 J (0.000196-0.000392 Wh)')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    steps, kind = get_steps(qa)
    steps = [s.replace('h = 2.0 m', 'h = 0.8 m').replace('× 2.0 m = 1.77 J', '× 0.8 m = 0.71 J')
             .replace('× 2.0 m = 3.53 J', '× 0.8 m = 1.41 J')
             .replace('1.77-3.53 joules (0.00049-0.00098 watt-hours)', '0.71-1.41 joules (0.000196-0.000392 '
                                                                       'watt-hours)') for s in steps]
    assert not any(x in s for s in steps for x in ('2.0 m', '1.77', '3.53')), steps
    steps[0] = steps[0].replace('during the lifting phase at 4.0-8.0 seconds.',
                                'as it engages the scrap pile at 4.0-9.0 seconds; the load is to be lifted 0.8 m '
                                '(given).')
    assert '4.0-9.0' in steps[0], steps[0]
    set_steps(qa, steps, kind)
    qa['answer'], qa['answer_range'] = '0.71-1.41 joules (0.000196-0.000392 watt-hours)', [0.71, 1.41]


# ---------------------------------------------------------------- a00087/00122 (spark between brass spheres)
@P('B|a00087/00122|1', 'full', 'capacitance 0.70 pF was not the code output (0.97); smaller sphere = the 50.8 mm '
                               'standard electrode as in QA0 -> 1.62 pF; "arcs" as in QA0; hidden "=-E22 50.8 mm" '
                               'cleaned')
def _(qa):
    q_sub(qa, 'a bright blue-white electric arc discharges between', 'bright blue-white electric arcs discharge between')
    q_sub(qa, 'In the video observed at 4.0–8.0 seconds,', 'In the video observed at 4.0–10.0 seconds,')
    q_sub(qa, 'times the diameter of the smaller sphere.', 'times the diameter of the smaller sphere; take the smaller '
                                                         'sphere as the standard reference electrode.')
    assert qa['hidden_cv_chosen'][1].startswith('typical brass sphere electrode diameter =')
    qa['hidden_cv_chosen'][1] = 'typical brass sphere electrode diameter = 50.8 mm'
    c = code("import math\nd_gap = 0.005  # m, 5 mm gap (video)\nsize_ratio = 1.6  # larger / smaller diameter (video)\n\n"
             "E_breakdown = 3.0e6  # V/m, dielectric strength of air\nsphere_diameter = 0.0508  # m, 50.8 mm standard "
             "(smaller sphere)\nepsilon0 = 8.854e-12  # F/m (8.854×10⁻¹²)\n\nV_min = E_breakdown * d_gap\n"
             "R_small = sphere_diameter / 2\nR_large = R_small * size_ratio\n"
             "C = 4 * math.pi * epsilon0 * (R_large * R_small) / (R_large + R_small + d_gap)\n\n"
             "print(f'Smaller sphere radius: {R_small*1000:.1f} mm')\nprint(f'Larger sphere radius: {R_large*1000:.2f} mm')\n"
             "print(f'Minimum breakdown voltage: {V_min/1000:.1f} kV')\n"
             "print(f'Capacitance between electrodes: {C*1e12:.2f} pF')")
    has(c['key_information'], 'Minimum breakdown voltage: 15.0 kV', 'Capacitance between electrodes: 1.62 pF')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    qa['sources'][0] = dict(qa['sources'][0], time_point='4.0–10.0 s',
                            key_information='brass spheres with size ratio 1.6:1 (4.0-8.0 s), electric arcs across '
                                            'the approximately 5 mm gap (8.0-10.0 s)')
    steps, kind = get_steps(qa)
    steps[0] = ('Step 1: Observe the video — bright electric arcs form between two brass spherical electrodes; the '
                'larger sphere has approximately 1.6 times the diameter of the smaller sphere, separated by '
                'approximately 5 mm; the smaller sphere is the standard reference electrode.')
    assert steps[5].startswith('Step 6:') and steps[6].startswith('Step 7:')
    steps[5] = ('Step 6: Use code_executor to compute: (a) Minimum breakdown voltage V_min = E_breakdown × d_gap = '
                '3.0×10⁶ V/m × 0.005 m = 15.0 kV, and (b) Capacitance using C = 4πε₀(R₁R₂)/(R₁+R₂+d) where R₂ is the '
                'radius of the smaller (standard) sphere (50.8 mm/2 = 25.4 mm), R₁ is the radius of the larger sphere '
                '(25.4 mm × 1.6 = 40.64 mm), and d is the gap distance (5 mm), giving C ≈ 1.62 pF.')
    steps[6] = 'Step 7: Result: minimum voltage 15.0 kV; capacitance between the electrodes 1.62 pF.'
    set_steps(qa, steps, kind)
    qa['answer'] = ('Minimum voltage required to initiate the spark: 15.0 kV. Capacitance between the spherical '
                    'electrodes: 1.62 pF.')


# ---------------------------------------------------------------- e6d6f5e3 (basketball KE ratio)
@P('A|e6d6f5e3-2821-4d8b-ab8f-5608c9dacb98|2', 'full',
   'same ball: the mass cancels in the KE ratio (the 14.4-16.4 range used the lightest and heaviest ball in numerator '
   'and denominator) -> (4.7/1.2)² = 15.3')
def _(qa):
    c = code("v_peak = 4.7\nv_move = 1.2\nm_min = 0.58\nm_max = 0.62\n"
             "for m in (m_min, m_max):\n    ke_peak = 0.5 * m * v_peak**2\n    ke_move = 0.5 * m * v_move**2\n"
             "    print(f'm = {m} kg: KE_peak = {ke_peak:.3f} J, KE_move = {ke_move:.3f} J, ratio = {ke_peak / ke_move:.2f}')\n"
             "print(f'Ratio (mass cancels) = (v_peak / v_move)^2 = {(v_peak / v_move)**2:.1f}')")
    has(c['key_information'], 'ratio = 15.34', 'Ratio (mass cancels) = (v_peak / v_move)^2 = 15.3')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    steps, kind = get_steps(qa)
    assert len(steps) == 5
    steps[3] = ('Step 4: Use code_executor to compute KE_peak = ½ × m × v_peak² and KE_movement = ½ × m × v_movement² '
                'for the same ball: the mass cancels in the ratio, KE_peak / KE_movement = (v_peak / v_movement)², '
                'for any mass in 0.58-0.62 kg.')
    steps[4] = ('Step 5: With v_peak = 4.7 m/s and v_movement = 1.2 m/s, the ratio is (4.7 / 1.2)² ≈ 15.3: the peak '
                'shooting motion has about 15.3 times more kinetic energy.')
    set_steps(qa, steps, kind)
    qa['answer'], qa['answer_range'] = ('The peak shooting motion has about 15.3 times more kinetic energy than the '
                                        'movement motion (ratio ≈ 15.3:1).', None)


# ---------------------------------------------------------------- question states the idealization the solution uses
@P('B|a00100/00705|1', 'question', 'the solution starts the marble from rest at the top of the red ramp (it enters '
                                   'the ramp from a wooden track): stated in the question; answer unchanged')
def _(qa):
    q_sub(qa, 'Using typical marble specifications, calculate:',
          'Assuming the marble starts from rest at the top of the red ramp and rolls without slipping, and using '
          'typical marble specifications, calculate:')


@P('test:B|a00588/00347|1', 'question', 'the solution puts the tire contact patches at half the wheelbase from the '
                                        'spin center (track width ignored): stated in the question; answer unchanged')
def _(qa):
    q_sub(qa, 'spinning approximately 180 degrees counter-clockwise while', 'spinning approximately 180 degrees while')
    q_sub(qa, 'spins about its center, estimate', 'spins about its center, with the tire contact patches treated as '
                                                 'sliding on a circle of radius equal to half the wheelbase, estimate')


@P('A|f9afd55f-08d8-4753-9f76-df55cd8f4e59|1', 'question',
   'the solution releases the ball at the set-shot height (no jump displacement): stated in the question; '
   'answer unchanged')
def _(qa):
    q_sub(qa, 'Using typical basketball jump shot specifications,',
          'Neglecting the jump displacement (the ball is released at the typical set-shot release height above the '
          'head) and using typical basketball shot specifications,')


@P('test:A|2a160c63-2acd-463a-8431-3dbf77d6de8e|2', 'question',
   'the retrieved mass is that of a bicycle multitool: the question names the multitool and states that it is carried '
   'at the walking pace; answer unchanged')
def _(qa):
    q_sub(qa, 'of a typical bicycle wrench when transitioning', 'of a typical bicycle multitool (used as the wrench) '
                                                               'when transitioning')
    q_sub(qa, 'Assume the wrench moves as a point mass at the wrist speed during the rapid motion.',
          'Assume the multitool is carried at the walking pace and moves as a point mass at the wrist speed during '
          'the rapid motion.')
    step_sub(qa, 'used as the typical bicycle wrench mass', 'the typical bicycle multitool used as the wrench')


# ---------------------------------------------------------------- a00087/00028 (kinetic ring sculpture)
@P('B|a00087/00028|2', 'full', 'invented ring section (2 x 5 cm) and 12 cm sphere at r = 0.5 m -> the ring section '
                               '(8 x 3 cm) and the 15 cm brass sphere spinning about its centre as in QA1; '
                               '372.9 -> 163.7 rev')
def _(qa):
    q_sub(qa, 'a wooden ring of approximately 1 m diameter that is part of a kinetic sculpture with a brass sphere '
              'counterweight.',
          'a wooden ring of approximately 1 m outer diameter (radial width 8 cm, thickness 3 cm) that is part of a '
          'kinetic sculpture; its brass sphere counterweight, about 15 cm in diameter and centred on the rotation axis, '
          'turns with the ring.')
    obs, ash, brass, tq, _ = qa['sources']
    obs = dict(obs, key_information='ring outer diameter ~1 m (radius 0.5 m), radial width 8 cm, thickness 3 cm; brass '
                                    'sphere ~15 cm diameter on the axis; ring completes 1/4 revolution in first second '
                                    'after push')
    c = code("import math\n\nR_out = 0.5\nR_in = R_out - 0.08\nthickness = 0.03\nr_sphere = 0.075\n"
             "video_fraction_rev = 1 / 4\nt = 1.0\nwood_density = 710\nbrass_density = 8500\ntau = 0.0013\n\n"
             "omega_0 = (video_fraction_rev * 2 * math.pi) / t\n"
             "ring_volume = math.pi * (R_out**2 - R_in**2) * thickness\nm_ring = wood_density * ring_volume\n"
             "sphere_volume = (4 / 3) * math.pi * r_sphere**3\nm_sphere = brass_density * sphere_volume\n"
             "I_ring = 0.5 * m_ring * (R_out**2 + R_in**2)\nI_sphere = 0.4 * m_sphere * r_sphere**2\n"
             "I = I_ring + I_sphere\nKE = 0.5 * I * omega_0**2\ntheta = KE / tau\nrev = theta / (2 * math.pi)\n\n"
             "print(f'omega_0 = {omega_0:.3f} rad/s')\nprint(f'm_ring = {m_ring:.2f} kg')\n"
             "print(f'm_sphere = {m_sphere:.2f} kg')\nprint(f'I_ring = {I_ring:.4f} kg·m²')\n"
             "print(f'I_sphere = {I_sphere:.4f} kg·m²')\nprint(f'I = {I:.4f} kg·m²')\nprint(f'KE = {KE:.4f} J')\n"
             "print(f'theta = {theta:.1f} rad')\nprint(f'revolutions = {rev:.1f} rev')")
    has(c["key_information"], "revolutions = 163.7 rev", "m_sphere = 15.02 kg")
    o = c['key_information']
    g = lambda k: o.split(k + ' = ')[1].split()[0]
    rev = g('revolutions')
    rebuild(qa, [obs, ash, brass, tq, c], [
        'Step 1: Observe the video — at 12.0–16.0 s, the man spins the wooden ring (outer radius R_out = 0.50 m, radial '
        'width 8 cm so R_in = 0.42 m, thickness 3 cm); the 15 cm brass sphere (r = 0.075 m) on the rotation axis turns '
        'with it. The ring completes 1/4 revolution in t = 1 s, giving ω₀ = (1/4 × 2π rad) / 1 s = π/2 ≈ 1.571 rad/s.',
        'Step 2: Identify the physical phenomenon — rotation slowed by a constant bearing friction torque; the initial '
        'rotational kinetic energy is dissipated as τ_f × θ.',
        "Step 3: Use web_search(query='European ash wood density kg/m³') to retrieve the wood density, 710 kg/m³. Ring "
        'volume V = π(0.50² − 0.42²) × 0.03 ≈ 0.00694 m³, m_ring = %s kg.' % g('m_ring'),
        "Step 4: Use web_search(query='brass density g/cm³') to retrieve the brass density, 8.5 g/cm³ = 8500 kg/m³. "
        'Sphere volume (4/3)π(0.075)³ ≈ 0.001767 m³, m_sphere = %s kg.' % g('m_sphere'),
        "Step 5: Use web_search(query='ball bearing rotational friction torque specification') to retrieve the deep "
        'groove ball bearing friction torque τ_f = 0.0013 N·m.',
        'Step 6: Moment of inertia: annular ring I_ring = ½ m_ring (R_out² + R_in²) = %s kg·m²; solid sphere about its '
        'centre I_sphere = (2/5) m_sphere r² = %s kg·m²; total I = %s kg·m².' % (g('I_ring'), g('I_sphere'), g('I')),
        'Step 7: Initial rotational kinetic energy KE = ½ I ω₀² = %s J.' % g('KE'),
        'Step 8: Work-energy theorem: θ = KE / τ_f = %s rad; revolutions = θ / (2π) = %s.' % (g('theta'), rev),
        'Step 9: Result: approximately %s revolutions before stopping due to bearing friction.' % rev])
    qa['answer'] = 'Approximately %s revolutions' % rev


# ---------------------------------------------------------------- a00611/02070 (tuning forks on resonance boxes)
@P('test:B|a00611/02070|1', 'full', 'fork + box mass 200 g -> 250 g stated by QA0 for the identical setup; '
                                    '0.064434 -> 0.066462 N·s')
def _(qa):
    q_sub(qa, 'has a mass of 200 g', 'has a mass of 250 g')
    c = code("v_mallet = 1.2\nm_mallet = 0.036\nm_fork = 0.250\ne = 0.76\n"
             "v_fork_final = (1 + e) * m_mallet * v_mallet / (m_mallet + m_fork)\nimpulse = m_fork * v_fork_final\n"
             "print(f'Final speed of tuning fork = {v_fork_final:.6f} m/s')\n"
             "print(f'Impulse delivered to tuning fork = {impulse:.6f} N·s')")
    has(c['key_information'], 'Final speed of tuning fork = 0.265846 m/s', 'Impulse delivered to tuning fork = '
                                                                             '0.066462 N·s')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    step_sub(qa, 'initially stationary with mass 200 g', 'initially stationary with mass 250 g')
    step_sub(qa, 'm_fork = 0.200 kg', 'm_fork = 0.250 kg')
    step_sub(qa, 'v_fork_final = 0.322169 m/s and impulse = 0.064434 N·s',
             'v_fork_final = 0.265846 m/s and impulse = 0.066462 N·s')
    qa['answer'] = 'The impulse delivered to the tuning fork is approximately 0.066462 N·s.'


@P('test:B|a00611/02070|2', 'full', 'the box was a solid 15 x 10 x 5 cm pine block (0.375 kg; resonance boxes are '
                                    'hollow) -> the box mass 210 g is given, fork prongs (steel density) as before: '
                                    'total 0.249 kg as QA0 (250 g); pine density search dropped; 169.3% -> 101.9%; '
                                    'the video shows the box placed, not slid: the 0.15 m slide is an assumption')
def _(qa):
    q_sub(qa, 'into its final position. Estimate', 'into its final position. The hollow wooden resonance box alone has '
                                                   'a mass of about 210 g, and each of the fork\'s two steel prongs is '
                                                   'about 10 cm long with a 0.5 cm × 0.5 cm cross-section (neglect the '
                                                   'stem). Estimate')
    q_sub(qa, 'a hand slides a tuning fork box approximately 0.15 m across the table surface into its final position.',
          'a hand places a second tuning fork box onto the table surface. Assume that it is slid approximately 0.15 m '
          'across the table surface into its final position.')
    obs, steel, pine, mu, mallet, _ = qa['sources']
    obs['key_information'] = 'a second tuning fork box is placed onto the table (slide of ~0.15 m assumed)'
    assert 'pine' in pine['search_query']
    c = code("sliding_distance = 0.15\nmallet_speed = 2.0\nmu_k = 0.2\nmallet_mass = 0.036\nbox_mass = 0.210\n"
             "prong_volume = 0.005 * 0.005 * 0.10 * 2\nsteel_density = 7850\nfork_mass = prong_volume * steel_density\n"
             "total_mass = box_mass + fork_mass\ng = 9.81\nnormal_force = total_mass * g\n"
             "friction_force = mu_k * normal_force\nfriction_work = friction_force * sliding_distance\n"
             "ke_mallet = 0.5 * mallet_mass * mallet_speed**2\nfraction = friction_work / ke_mallet\n"
             "print(f'Fork mass: {fork_mass:.5f} kg')\nprint(f'Total mass: {total_mass:.5f} kg')\n"
             "print(f'Normal force: {normal_force:.5f} N')\nprint(f'Friction force: {friction_force:.5f} N')\n"
             "print(f'Friction work: {friction_work:.5f} J')\nprint(f'Mallet KE: {ke_mallet:.3f} J')\n"
             "print(f'Percentage of mallet initial kinetic energy: {fraction * 100:.1f}%')")
    has(c['key_information'], 'Fork mass: 0.03925 kg', 'Total mass: 0.24925 kg', 'Friction work: 0.07335 J',
        'Mallet KE: 0.072 J', 'kinetic energy: 101.9%')
    rebuild(qa, [obs, steel, mu, mallet, c], [
        'Step 1: Observe the video — at 12.0–16.0 s a hand places a second tuning fork box onto the table; as the '
        'question states, assume it is slid approximately 0.15 m into its final position.',
        'Step 2: From the question: the hollow resonance box has a mass of 0.210 kg; the fork has two steel prongs of '
        '10 cm × 0.5 cm × 0.5 cm.',
        "Step 3: Use web_search(query='steel density engineering toolbox') to retrieve steel density = 7850 kg/m³.",
        "Step 4: Use web_search(query='wood on wood kinetic friction coefficient') to retrieve μ_k ≈ 0.2.",
        "Step 5: Use web_search(query='tuning fork mallet technical specifications mass') to retrieve mallet mass = 36 "
        'grams = 0.036 kg.',
        'Step 6: Fork mass: prong volume = (0.005 × 0.005 × 0.10) × 2 = 0.000005 m³; fork mass = 0.000005 × 7850 = '
        '0.03925 kg. Total mass = 0.210 + 0.03925 = 0.24925 kg.',
        'Step 7: Normal force N = 0.24925 × 9.81 = 2.44514 N; friction force = 0.2 × 2.44514 = 0.48903 N; friction work '
        'W = 0.48903 × 0.15 = 0.07335 J.',
        'Step 8: Mallet kinetic energy KE = 0.5 × 0.036 × 2² = 0.072 J.',
        'Step 9: W / KE = 0.07335 / 0.072 ≈ 1.019, or 101.9%: the sliding friction work is slightly greater than the '
        "mallet's initial kinetic energy."])
    qa['answer'] = ("The sliding friction work is about 0.07335 J, slightly greater than the mallet's initial kinetic "
                    'energy of 0.072 J. It would consume approximately 101.9% of that initial kinetic energy.')


# ---------------------------------------------------------------- a00673/02402 (pink object splashing into water)
WD_Q = 'density of water at room temperature kg/m3'
WD_QUOTE = 'At 20 C (room temperature) the density is 998.2 kg/m3.'


@P('B|a00673/02402|1', 'full', 'object diameter 0.08 m / splash 0.24 m / drop height 1.2 m contradicted QA0 (2.5 cm, '
                               '12.5 cm, 0.6 s fall): aligned with QA0; the pink guava diameter (6-10 cm) no longer '
                               'applies -> hidden value replaced by the water density the Weber number uses (998.2 '
                               'kg/m³, princeton.edu); We 25872 -> 11875')
def _(qa):
    qa['question'] = ('In the video (0.0-4.0 s), a pink spherical object of diameter approximately 2.5 cm falls from '
                      'rest into dark water, taking approximately 0.6 s to reach the surface, and the resulting splash '
                      'column rises to about 12.5 cm. Using physical criteria for liquid impact splashing, determine '
                      'whether this impact produces significant splashing by calculating the Weber number and comparing '
                      'it to the critical threshold.')
    assert 'guava' in qa['hidden_cv_chosen'][2]
    qa['hidden_cv_chosen'][2] = 'water density at 20 °C = 998.2 kg/m³'
    obs, st, wc, _, _ = qa['sources']
    obs = dict(obs, key_information='object fall time ≈ 0.6 s from rest, diameter ≈ 0.025 m, splash height ≈ 0.125 m')
    c = code("g = 9.81\nt = 0.6\nD = 0.025\nrho = 998.2\nsigma = 0.0728\nWe_crit = 100\n"
             "v = g * t\nWe = rho * v**2 * D / sigma\nWe_i = int(We)\n"
             "print(f'v = {v:.3f} m/s')\nprint(f'We = {We_i} (dimensionless)')\n"
             "print(f'We_crit = {We_crit} (dimensionless)')\nverdict = 'Yes' if We_i > We_crit else 'No'\n"
             "print(f'Significant splashing? {verdict} - We={We_i} > We_crit={We_crit}')")
    has(c['key_information'], 'v = 5.886 m/s', 'We = 11875', 'Significant splashing? Yes')
    rebuild(qa, [obs, st, wc, web(WD_Q, 'princeton.edu', WD_QUOTE, 'water density at 20 °C = 998.2 kg/m³',
                                  ('998.2 kg/m3', '998.2 kg/m³')), c], [
        'Step 1: Observe the video — the pink object (D ≈ 0.025 m) falls from rest for t ≈ 0.6 s before hitting the '
        'water; the splash column rises to about 0.125 m.',
        'Step 2: Identify the physical phenomenon — liquid impact splashing governed by the Weber number criterion '
        'We = ρv²D/σ > We_crit.',
        "Step 3: Use web_search(query='water surface tension coefficient at room temperature N/m') to retrieve water "
        'surface tension σ = 0.0728 N/m at 20°C.',
        "Step 4: Use web_search(query='critical Weber number for splashing water impact') to retrieve the critical "
        'Weber number We_crit ≈ 100.',
        "Step 5: Use web_search(query='%s') to retrieve the water density at 20 °C, ρ = 998.2 kg/m³." % WD_Q,
        'Step 6: Impact velocity from free fall: v = g t = 9.81 × 0.6 = 5.886 m/s.',
        'Step 7: Run code_executor: We = ρv²D/σ = 998.2 × 5.886² × 0.025 / 0.0728 ≈ 11875.',
        'Step 8: Compare: We ≈ 11875 > We_crit ≈ 100 → the impact produces significant splashing.'])
    qa['answer'] = ('Yes — Weber number ≈ 11875, far above the critical threshold We_crit ≈ 100. The impact produces '
                    'significant splashing.')


# ---------------------------------------------------------------- FIX6 review: wording the video contradicts
@P('A|90e697a5-2cff-4ecc-a580-f4c230a5488c|0', 'question', 'FIX6 "jump shot" -> "close-range shot" (the video shows a '
                                                           'close-range shot; same wording as QA1); answer unchanged')
def _(qa):
    q_sub(qa, 'during a jump shot.', 'during a close-range shot.')
    step_sub(qa, 'during a jump shot.', 'during a close-range shot.')


@P('B|a00818/01712|0', 'question', 'FIX6 "suspended and stationary" contradicted the asked upward terminal velocity -> '
                                   '"appear almost stationary"; answer unchanged')
def _(qa):
    q_sub(qa, 'observed suspended and stationary in golden honey', 'observed suspended in golden honey, appearing '
                                                                  'almost stationary,')
    step_sub(qa, 'air bubbles suspended and stationary in honey', 'air bubbles suspended in honey, appearing almost '
                                                                  'stationary,')


@P('B|a00576/00194|0', 'question', 'pitcher "approximately 15 cm above" the cup -> "just above" as the video shows '
                                   '(the value is not used by the solution); answer unchanged')
def _(qa):
    q_sub(qa, 'held approximately 15 cm above a white ceramic cup', 'held just above a white ceramic cup')
    step_sub(qa, 'from pitcher held ~15 cm above cup', 'from pitcher held just above cup')


# ---------------------------------------------------------------- FIX6 failures fixed by hand (frames checked)
@P('B|a00332/00025|1', 'full', 'drop from the chute onto the yellow slide 0.5 m -> 0.05 m (frames at 4.5-5.5 s: about '
                               'one ball diameter); 0.000932 J -> 0.0000932 J, ratio 0.2317 -> 0.0232')
def _(qa):
    q_sub(qa, 'from an estimated height of 0.5 m at 4.0–8.0 s', 'from an estimated height of 0.05 m at 4.0–8.0 s')
    obs = qa['sources'][0]
    assert '0.5 m' in obs['key_information']
    obs['key_information'] = 'yellow ball drops from estimated height 0.05 m, bounce count = 1'
    c = code('g = 9.81  # m/s^2\nh_drop = 0.05  # m\nm = 0.001  # kg (1.0 g)\nL_cm = 41  # cm (slide length)\n'
             'L = L_cm / 100.0  # m\ne = 0.9\n\nh_bounce = e**2 * h_drop\nE_lost = m * g * (h_drop - h_bounce)\n'
             'E_lift = m * g * L\nratio = E_lost / E_lift\n\nprint(f"Slide length: {L_cm} cm = {L:.2f} m")\n'
             'print(f"Bounce height at e={e}: {h_bounce:.4f} m")\n'
             'print(f"Energy lost in the bounce: {E_lost:.7f} J")\n'
             'print(f"Energy to lift ball by slide length L={L} m: {E_lift:.6f} J")\n'
             'print(f"Ratio (energy lost / lifting energy): {ratio:.4f}")')
    has(c['key_information'], 'Bounce height at e=0.9: 0.0405 m', 'Energy lost in the bounce: 0.0000932 J',
        'L=0.41 m: 0.004022 J', 'lifting energy): 0.0232')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    step_sub(qa, 'from an estimated height of 0.5 m at 4.0–8.0 s', 'from an estimated height of 0.05 m at 4.0–8.0 s')
    step_sub(qa, 'h_bounce = e² × h_drop = 0.81 × 0.5 m = 0.4050 m.', 'h_bounce = e² × h_drop = 0.81 × 0.05 m = 0.0405 m.')
    step_sub(qa, 'ΔE = 0.000932 J, E_lift = 0.004022 J, ratio = 0.2317.',
             'ΔE = 0.0000932 J, E_lift = 0.004022 J, ratio = 0.0232.')
    step_sub(qa, 'is 0.2317 times the lifting energy', 'is 0.0232 times the lifting energy')
    qa['answer'] = ('The energy lost in the bounce (about 0.0000932 J with e = 0.9) is about 0.0232 times the energy '
                    'required to lift the ball by the slide length (about 0.004022 J); thus the lifting energy is larger.')


@P('B|a00870/00376|1', 'full', 'cup held 0.15 m above the rim -> 0.02 m as QA0 / QA2 (FIX6, the video shows the cup '
                               'just above the rim); 5.886 mW -> 0.785 mW')
def _(qa):
    q_sub(qa, 'held approximately 0.15 m above the rim', 'held approximately 0.02 m above the rim')
    obs = qa['sources'][0]
    assert '0.15 m' in obs['key_information']
    obs['key_information'] = 'pouring height ≈ 0.02 m above the rim, stream angle = 270°'
    c = code("import math\nh = 0.02\ng = 9.81\nrho = 1000\nQ = 4e-6\nv = math.sqrt(2*g*h)\nP = 0.5*rho*Q*v**2\n"
             "print(f'v = {v:.6f} m/s')\nprint(f'P = {P:.6f} W = {P*1000:.3f} mW')")
    has(c['key_information'], 'v = 0.626418 m/s', 'P = 0.000785 W = 0.785 mW')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    step_sub(qa, 'from a height of approximately 0.15 m above', 'from a height of approximately 0.02 m above')
    step_sub(qa, 'with h = 0.15 m.', 'with h = 0.02 m.')
    step_sub(qa, 'v = sqrt(2 * 9.81 * 0.15)', 'v = sqrt(2 * 9.81 * 0.02)')
    step_sub(qa, 'v = 1.715517 m/s and P = 0.005886 W = 5.886 mW', 'v = 0.626418 m/s and P = 0.000785 W = 0.785 mW')
    qa['answer'] = 'The kinetic energy flux is approximately 0.000785 W (0.785 mW).'


@P('B|a00883/01067|0', 'full', 'about 10 matchsticks -> about 100 (frames at 0.5-3.5 s: a dense ring of matches '
                               'under the dome); 10-20 kJ -> 100-200 kJ')
def _(qa):
    q_sub(qa, 'approximately 10 wooden matchsticks', 'approximately 100 wooden matchsticks')
    obs = qa['sources'][0]
    obs['key_information'] = 'approximately 100 matchsticks arranged under foil dome'
    c = code('num_matches = 100\nenergy_per_match_min_J = 1000\nenergy_per_match_max_J = 2000\n'
             'total_energy_min_J = num_matches * energy_per_match_min_J\n'
             'total_energy_max_J = num_matches * energy_per_match_max_J\n'
             'total_energy_min_kJ = total_energy_min_J / 1000.0\ntotal_energy_max_kJ = total_energy_max_J / 1000.0\n'
             'print(f"Total minimum energy = {total_energy_min_J} J = {total_energy_min_kJ:g} kJ")\n'
             'print(f"Total maximum energy = {total_energy_max_J} J = {total_energy_max_kJ:g} kJ")\n'
             'print(f"Final answer range: {total_energy_min_kJ:g}-{total_energy_max_kJ:g} kJ")')
    has(c['key_information'], 'Final answer range: 100-200 kJ')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    step_sub(qa, 'approximately 10 matchsticks are visible', 'approximately 100 matchsticks are visible')
    step_sub(qa, 'Result is approximately 10-20 kJ.', 'Result is approximately 100-200 kJ.')
    qa['answer'], qa['answer_range'] = '100-200 kJ', [100, 200]


@P('A|aab43780-401b-4673-b2c2-fc3521e98bea|2', 'solution', 'the speeds are measured at t = 1.4-1.8 s (dribble '
   'approach to the first layup, 0-4 s); the observation was labelled 20.0-22.9 s (end of the drill); answer unchanged')
def _(qa):
    obs = qa['sources'][0]
    assert obs['time_point'] == '20.0–22.9 s'
    obs['time_point'] = '1.4–1.8 s'
    step_sub(qa, 'Step 1: Observe the video segment (20.0–22.9 s) — measure', 'Step 1: Observe the dribble approach to '
             'the first layup (t = 1.4–1.8 s) — measure')


@P('A|d4a1ea83-80ca-4540-95dd-d1e2af40328a|1', 'question', 'the jump shot is at 0-4 s; at 8.0-12.0 s (where the '
   'height is observed) the player takes a set shot: the question no longer puts the jump shot at 8-12 s; answer '
   'unchanged')
def _(qa):
    q_sub(qa, 'In the video at 8.0–12.0 seconds, a basketball player executes a jump shot during practice. Based on',
          "In the video, a basketball player takes shots during practice, including a jump shot; the player's full "
          'height is visible at 8.0–12.0 seconds. Based on')

# ---------------------------------------------------------------- quantities the video cannot show -> assumptions
@P('B|a00138/01103|0', 'question', 'the magnet is lowered onto the pile at 4-8 s and takes up the scrap at 8-9 s; the '
                                   'clip ends (9.2 s) before the 0.8 m lift -> stated as an assumption (supersedes '
                                   'FIX6); answer unchanged')
def _(qa):
    q_sub(qa, 'an industrial electromagnetic lifter picks up approximately 40 small metal hardware pieces (washers, '
              'bolts, rings) from a scrap pile on the factory floor visible at 4.0-8.0 seconds.',
          'an industrial electromagnetic lifter is lowered onto a scrap pile on the factory floor (4.0-8.0 seconds) '
          'and picks up approximately 40 small metal hardware pieces (washers, bolts, rings) at 8.0-9.0 seconds.')
    q_sub(qa, 'The lifter raises the collected scrap', 'Assume the lifter then raises the collected scrap')
    assert qa['sources'][0]['type'] == 'video_observation'
    qa['sources'][0] = dict(qa['sources'][0], time_point='4.0-9.0 s',
                            key_information='electromagnetic lifter lowered onto the scrap pile, picking up metal '
                                            'scrap pieces')
    step_sub(qa, 'picks up metal scrap pieces and lifts them approximately 0.8 meters high.',
             'picks up metal scrap pieces; the lift height is given as approximately 0.8 meters.')


@P('B|a00879/00373|0', 'question', 'the bench is near level and 25 deg is the on-screen trajectory angle -> the incline '
                                   'is stated as a model; answer unchanged')
def _(qa):
    q_sub(qa, 'a red billiard ball is observed rolling down a wooden track at an angle of 25° below horizontal. Assuming',
          'a red billiard ball is observed rolling along a wavy wooden track, its on-screen trajectory pointing 25° '
          'below horizontal. Modelling the path as a straight incline at 25° below horizontal, assuming')
    step_sub(qa, 'the red billiard ball rolls at 25° below horizontal (derived from VLM-measured trajectory angle of '
                 '205° relative to horizontal right).',
             'the red billiard ball rolls along the track with an on-screen trajectory angle of 205° relative to '
             'horizontal right (25° below horizontal); model the path as a 25° incline with the given drop h = 0.5 m.')


@P('B|a00879/00373|1', 'question', 'the near-level bench cannot give a 0.55 m drop; 25 deg is the on-screen trajectory '
                                   'angle -> slope and drop stated as a model; answer unchanged')
def _(qa):
    q_sub(qa, 'a red billiard ball is observed rolling down a wavy wooden track with an estimated vertical drop of '
              'approximately 0.55 meters over a slope inclined at roughly 25° below horizontal.',
          'a red billiard ball is observed rolling along a wavy wooden track. Model its descent as a straight slope '
          'inclined at roughly 25° below horizontal (its on-screen trajectory angle) with a vertical drop of '
          'approximately 0.55 meters.')
    assert qa['sources'][0]['type'] == 'video_observation'
    qa['sources'][0] = dict(qa['sources'][0], key_information='red billiard ball rolling along the wavy track, on-screen '
                                                             'trajectory about 25° below horizontal (h ≈ 0.55 m given)')
    step_sub(qa, 'the red billiard ball rolls down a wavy wooden track. Estimate the vertical drop h ≈ 0.55 m and slope '
                 'angle θ ≈ 25° from the scene geometry.',
             'the red billiard ball rolls along a wavy wooden track; take the given vertical drop h ≈ 0.55 m and the '
             'slope angle θ ≈ 25° (on-screen trajectory angle).')


@P('B|a00879/00373|2', 'full', 'the 205 deg on-screen angle is used as the incline angle: stated as a model; the '
                               'range 95-113 rad/s^2 "for mass/diameter tolerances" was invented (the mass cancels) -> '
                               '103.6 rad/s^2 from the code, mass range carried through the friction torque; garbled '
                               'diameter hidden entry cleaned')
def _(qa):
    q_sub(qa, 'This trajectory corresponds to the ball moving at 25° below horizontal along the incline.',
          'Model this as the ball rolling down a straight incline at 25° below horizontal.')
    assert qa['hidden_cv_chosen'][1].startswith('billiard ball diameter =')
    qa['hidden_cv_chosen'][1] = 'billiard ball diameter = 2.25 inches (57.15 mm)'
    c = code("import math\n\ng = 9.81\ntheta = math.radians(25)  # incline angle (on-screen trajectory angle)\n"
             "R = 0.05715 / 2  # m, 2.25 in (57.15 mm) diameter\n\n"
             "for m in (0.160, 0.170):  # kg, 160-170 g\n"
             "    I = (2 / 5) * m * R**2  # solid sphere\n"
             "    a = m * g * math.sin(theta) / (m + I / R**2)  # rolling without slipping\n"
             "    f = m * g * math.sin(theta) - m * a  # static friction at the contact\n"
             "    alpha = f * R / I\n"
             "    print(f'm = {m*1000:.0f} g: a = {a:.3f} m/s², friction = {f:.4f} N, α = {alpha:.1f} rad/s²')")
    has(c['key_information'], 'm = 160 g: a = 2.961 m/s², friction = 0.1895 N, α = 103.6 rad/s²',
        'm = 170 g: a = 2.961 m/s², friction = 0.2014 N, α = 103.6 rad/s²')
    qa['sources'] = [s for s in qa['sources'] if s['type'] != 'code_executor'] + [c]
    steps, kind = get_steps(qa)
    assert steps[5].startswith('Step 6:') and steps[6].startswith('Step 7:'), steps
    steps[5] = ('Step 6: Use code_executor to compute, with θ = 25°, g = 9.81 m/s² and R = 0.05715/2 = 0.028575 m, for '
                'm = 160 g and 170 g: I = (2/5)mR², a = mg sinθ/(m + I/R²) = (5/7)g sinθ = 2.96 m/s², friction '
                'f = mg sinθ − ma, and angular acceleration α = fR/I = a/R = 103.6 rad/s² (the mass cancels).')
    steps[6] = 'Step 7: Result is approximately 104 rad/s² (103.6 rad/s²) over the whole 160-170 g mass range.'
    set_steps(qa, steps, kind)
    qa['answer'], qa['answer_range'] = 'Approximately 104 rad/s² (103.6 rad/s²; independent of the ball mass)', None


@P('B|a00087/00028|1', 'question', 'the rotation count cannot be read off the uniform ring in the video -> stated as an '
                                   'assumption; answer unchanged')
def _(qa):
    q_sub(qa, 'The ring completes 2 full rotations in the first 4 seconds after being spun.',
          'Assume the ring completes 2 full rotations in the first 4 seconds after being spun.')
    assert qa['sources'][0]['type'] == 'video_observation'
    qa['sources'][0] = dict(qa['sources'][0], key_information='man spins the wooden ring (2 rotations in 4 s given)')
    step_sub(qa, 'and completes 2 rotations in 4 seconds (t=12.0-16.0 s).',
             'and is spun at t=12.0-16.0 s; it completes 2 rotations in 4 seconds (given).')


@P('B|a00087/00122|0', 'question', 'the arcs are seen at 8.0-10.0 s (4.0-8.0 s shows the spheres without arc); VOBS gap '
                                   '"≈ smaller sphere diameter" -> 5 mm as in the question; answer unchanged')
def _(qa):
    q_sub(qa, 'sparking between brass spheres at approximately 4.0-8.0 seconds.',
          'sparking between brass spheres at approximately 4.0-10.0 seconds.')
    assert qa['sources'][0]['type'] == 'video_observation'
    qa['sources'][0] = dict(qa['sources'][0], time_point='4.0-10.0 s',
                            key_information='brass spheres, larger sphere ~1.6× smaller sphere diameter (4.0-8.0 s); '
                                            'electric arcs across a gap ≈ 5 mm (8.0-10.0 s)')
    step_sub(qa, 'discharge between brass spheres from 4.0-8.0 s.', 'discharge between brass spheres from 4.0-10.0 s.')


def main():
    cur = current(skip=('FIX7',))
    out, base = {}, {}
    for qid, (f, change, note) in PATCH.items():
        qa = copy.deepcopy(cur[qid][0])
        base[qid] = copy.deepcopy(qa)
        t, d, h = qa['question_type'], qa['difficulty'], list(qa.get('hidden_cv_chosen') or [])
        f(qa)
        assert (qa['question_type'], qa['difficulty'], len(qa.get('hidden_cv_chosen') or [])) == (t, d, len(h)), qid
        assert not qid.split(':')[-1].startswith('A') or (qa.get('hidden_cv_chosen') or []) == h, qid
        out[qid] = {'status': 'ok', 'qa': qa, 'change': change, 'note': note}
    with ThreadPoolExecutor(8) as ex:
        for qid, p in ex.map(one, [(q, r['qa']) for q, r in out.items()]):
            # strict problems already present before the patch (e.g. A-class CV ranges) are not new
            p = p + [x for x in strict_problems(out[qid]['qa']) if x not in strict_problems(base[qid])]
            out[qid]['problems'] = p
            if p:
                out[qid]['status'] = 'fail'
            print(qid, out[qid]['change'], out[qid]['status'], p or '')
    json.dump(out, open(WORK + '/fix7_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    rp = json.load(open(WORK + '/review_post.json', encoding='utf-8'))
    for q, x in out.items():
        rp.get(q, {}).pop('FIX7', None)
        if x['status'] == 'ok':
            rp.setdefault(q, {})['FIX7'] = {'verdict': 'accept', 'notes': x['note']}
    json.dump(rp, open(WORK + '/review_post.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(out), 'patched,', sum(r['status'] == 'ok' for r in out.values()), 'ok')


if __name__ == '__main__':
    main()
