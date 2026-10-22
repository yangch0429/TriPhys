# Provenance (TriPhysBench release): fix_code/_v8_work/fix4.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""FIX4: fixes of physically wrong / inconsistent QAs found by manual inspection and the deep recheck (deep_audit.py,
video_audit.py); applied after CONS2, question_type / difficulty / number of hidden values unchanged.
  a00722  QA0 spatula speed 0.3 -> 0.15 m/s as in QA1 (same 4-8 s); "ultra-low" 0.5-1.5 Pa·s epoxy -> typical liquid
          epoxy 8-20 Pa·s (thick stringy resin in the video); QA1 Part-A-only 200 Pa·s -> mixed viscosity 10 Pa·s
  a00818  QA1 bubble diameter 2 mm -> the 0.5 mm stated in QA0 (same honey, "tiny" bubbles), honey density from the
          source used by QA0's range, air density retrieved (buoyancy uses rho_honey - rho_air as in QA0)
  a00819  the glass is a short rocks glass ending about half full after a ~2.2 s thin pour; the Torricelli flow through
          the full 28 mm bottle mouth (1.06 L/s, 2.3 L poured) is impossible -> flow = half the glass capacity / 2.2 s
          (QA0 flow rate, QA1 CO2 mass flow, QA2 impact pressure with rocks-glass dimensions)
Every patched QA is checked again with final_recheck.one; accepted fixes go to review_post.json (FIX4).
-> fix4_results.json {qid: {status, qa, change, note, problems}}
Usage: python fix4.py"""
import copy, json
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, sources_of, get_steps, set_steps, run_code
from current import current
from final_recheck import one
from fix39 import logged

PATCH = {}


def P(qid, change, note):
    def deco(f):
        PATCH[qid] = (f, change, note)
        return f
    return deco


def web(query, url_part, quote, key, used=None):
    s = {'type': 'web_search', 'url': logged(query, url_part, quote), 'key_information': key, 'search_query': query,
         'verified_quote': quote}
    if used:
        s['evidence_number_used'] = {'quote_number': used[0], 'value_used': used[1]}
    return s


def code(c):
    o = run_code(c)['out'].strip()
    return {'type': 'code_executor', 'code': c, 'key_information': o, 'key_information_annotated': None}


def rebuild(qa, sources, steps):
    qa['sources'] = sources
    set_steps(qa, steps, get_steps(qa)[1])
    if 'total_steps' in qa:
        qa['total_steps'] = len(steps)


def has(o, *ts):
    for t in ts:
        assert t in o, (t, o)


# ---------------------------------------------------------------- a00722 (epoxy resin mixed on a board)
EPOXY_Q = '"epoxy resin" viscosity "centipoise" typical value mixing'


@P('B|a00722/00179|0', 'full', 'spatula speed 0.3 -> 0.15 m/s (QA1, same 4-8 s); typical liquid epoxy 8,000-20,000 cP '
                               '(ScienceDirect) instead of an ultra-low-viscosity product; 20-60 -> 320-800 Pa')
def _(qa):
    assert 'approximately 0.3 m/s' in qa['question']
    qa['question'] = qa['question'].replace('approximately 0.3 m/s', 'approximately 0.15 m/s')
    obs = qa['sources'][0]
    c = code("mu_min = 8.0  # Pa·s (8,000 cP)\nmu_max = 20.0  # Pa·s (20,000 cP)\ndu_dy = 40  # s⁻¹\n"
             "tau_min = mu_min * du_dy\ntau_max = mu_max * du_dy\nprint(f'Shear stress range: {tau_min:g}–{tau_max:g} Pa')")
    has(c['key_information'], '320–800 Pa')
    rebuild(qa, [obs, web(EPOXY_Q, 'sciencedirect.com/topics/engineering/epoxy-resin-system',
                          'The most widely used liquid epoxy resins have viscosities in the 8,000 to 20,000 centipoise range',
                          'liquid epoxy resin viscosity = 8,000-20,000 cP (converted: 8-20 Pa·s)',
                          ('8,000 to 20,000 centipoise', '8-20 Pa·s')), c], [
        'Step 1: Observe the video — at 4.0-8.0 seconds, a wooden spatula is seen mixing thick, viscous amber-colored '
        'epoxy resin with a circular motion at about 0.15 m/s.',
        'Step 2: Identify the physical phenomenon — viscous shear stress in a Newtonian fluid, given by τ = μ × (du/dy), '
        'where τ is shear stress, μ is dynamic viscosity, and du/dy is the velocity gradient.',
        "Step 3: Use web_search(query='%s') to retrieve the typical viscosity of liquid epoxy resin: the most widely used "
        'liquid epoxy resins have viscosities of 8,000 to 20,000 centipoise, i.e. 8–20 Pa·s.' % EPOXY_Q,
        'Step 4: Use code_executor to compute the shear stress range τ_min = μ_min × (du/dy) and τ_max = μ_max × (du/dy), '
        'with du/dy = 40 s⁻¹.',
        'Step 5: Result is the range of shear stress, 320–800 Pa.'])
    qa['answer'], qa['answer_range'] = '320–800 Pa', [320, 800]


TDS_Q = '5 minute epoxy technical data sheet mixed viscosity'


@P('B|a00722/00179|1', 'full', 'viscosity of Part A alone (200 Pa·s) -> mixed viscosity of a two-part epoxy adhesive '
                               '10,000 cps = 10 Pa·s (data sheet); 19.151 -> 0.958 N')
def _(qa):
    old = 'Use the resin viscosity from the technical datasheet.'
    assert qa['question'].endswith(old)
    qa['question'] = qa['question'][:-len(old)] + ('Use the mixed viscosity from the technical data sheet of a '
                                                   'general-purpose two-part (5-minute) epoxy adhesive.')
    assert qa['hidden_cv_chosen'][0] == 'liquid resin viscosity = 200 Pa·s'
    qa['hidden_cv_chosen'][0] = 'mixed epoxy viscosity = 10 Pa·s'
    obs, _, dims, _ = qa['sources']
    c = code("import math\nmu = 10.0  # Pa*s (10,000 cps mixed)\nv = 0.15  # m/s\nD = 0.0508  # m\nh = 0.003175  # m\n"
             "A = math.pi * (D / 2) ** 2\ntau = mu * v / h\nF = tau * A\nprint(f'Viscous shear force: {F:.3f} N')")
    has(c['key_information'], '0.958 N')
    rebuild(qa, [obs, web(TDS_Q, 'itwperformancepolymers.com', 'VISCOSITY 10,000cps MIXED',
                          'mixed viscosity = 10,000 cps (converted: 10 Pa·s)', ('10,000cps', '10 Pa·s')), dims, c], [
        'Step 1: Observe the video: from 4.0–8.0 s, a craftsperson mixes viscous amber resin adhesive with a wooden '
        'stick; the mixing speed is estimated as v ≈ 0.15 m/s.',
        'Step 2: Identify the physical model: viscous shear stress in a thin resin layer is τ = μ v / h, where h is the '
        'layer thickness. The total viscous force to shear the pool area A = π (D/2)² is F = τ A = μ v π D² / (4 h).',
        "Step 3: Retrieve the mixed viscosity with web_search(query='%s'): the data sheet gives 10,000 cps mixed, "
        'μ = 10 Pa·s.' % TDS_Q,
        "Step 4: Retrieve the typical resin pool diameter with web_search(query='%s'): D ≈ 2 inches = 0.0508 m."
        % dims['search_query'],
        'Step 5: Retrieve the typical casting thickness from the same source: h ≈ 0.125 inches = 0.003175 m.',
        'Step 6: Use code_executor to compute F = μ × v × π × D² / (4 h).',
        'Step 7: The computed viscous shear force is 0.958 N.'])
    qa['answer'] = '0.958 N'


# ---------------------------------------------------------------- a00818 (honey, tiny rising bubbles)
@P('B|a00818/01712|1', 'full', 'bubble diameter 2 mm -> 0.5 mm stated (as in QA0); honey density 1415 kg/m3 '
                               '(kg-m3.com, inside QA0\'s 1380-1450); air density 1.2041 kg/m3 retrieved; '
                               '0.00030 -> 0.0000193 m/s (inside QA0\'s range)')
def _(qa):
    old = ('Assuming spherical bubbles and laminar flow conditions, calculate the terminal upward velocity of a typical '
           'air bubble in the honey using standard honey properties and typical bubble dimensions.')
    assert qa['question'].endswith(old)
    qa['question'] = qa['question'][:-len(old)] + (
        'The bubbles have an average diameter of 0.5 mm. Assuming spherical bubbles, Stokes drag and laminar flow '
        'conditions, calculate the terminal upward velocity of a typical air bubble in the honey using standard honey '
        'properties and the standard density of air at 20 °C.')
    assert len(qa['hidden_cv_chosen']) == 3
    qa['hidden_cv_chosen'] = ['honey viscosity = 10 Pa·s', 'honey density = 1415 kg/m³', 'air density = 1.2041 kg/m³']
    obs, chegg = qa['sources'][:2]
    chegg = dict(chegg, key_information='honey viscosity = 10 Pa·s',
                 verified_quote='The viscosity of liquid honey near room temperature is 10.0Pa∙s')
    logged(chegg['search_query'], 'chegg.com', chegg['verified_quote'])
    hq, aq = 'how much does a cubic metre of honey weigh', 'density of air at 20 °C kg/m3'
    c = code("mu = 10.0  # Pa*s\nrho_honey = 1415.0  # kg/m^3\nrho_air = 1.2041  # kg/m^3\ng = 9.81\nR = 0.25e-3  # m\n"
             "v_t = (2 / 9) * (rho_honey - rho_air) * g * R**2 / mu\n"
             "print(f'v_t = {v_t:.4e} m/s = {v_t * 1000:.4f} mm/s')")
    has(c['key_information'], '1.9263e-05 m/s', '0.0193 mm/s')
    rebuild(qa, [obs, chegg,
                 web(hq, 'kg-m3.com/material/honey', 'Density of Honey is 1415 kg/m3', 'honey density = 1415 kg/m³'),
                 web(aq, 'kg-m3.com/material/air-20c', 'Mass density of Air 20°C is 1.2041 kg/m3',
                     'air density at 20 °C = 1.2041 kg/m³'), c], [
        'Step 1: Observe the video — at 8.0–12.0 seconds, tiny air bubbles (diameter 0.5 mm, R = 0.00025 m) rise '
        'vertically (270°) through honey in the glass container.',
        "Step 2: Identify the physical phenomenon — buoyant bubbles reaching terminal velocity in a viscous fluid, "
        "governed by Stokes' law for small Reynolds numbers: v_t = (2/9) × (ρ_honey − ρ_air) × g × R² / μ.",
        "Step 3: Use web_search(query='%s') to retrieve honey viscosity μ = 10 Pa·s." % chegg['search_query'],
        "Step 4: Use web_search(query='%s') to retrieve honey density ρ_honey = 1415 kg/m³." % hq,
        "Step 5: Use web_search(query='%s') to retrieve air density ρ_air = 1.2041 kg/m³." % aq,
        'Step 6: Use code_executor to compute v_t with g = 9.81 m/s² and R = 0.00025 m.',
        'Step 7: Result is v_t ≈ 1.93 × 10⁻⁵ m/s (0.0193 mm/s).'])
    qa['answer'] = '0.0000193 m/s (0.0193 mm/s)'


# ---------------------------------------------------------------- a00819 (Fentimans poured into a rocks glass)
OF_Q, OF_URL, OF_QUOTE = 'old fashioned glass capacity ml', 'wikipedia.org/wiki/Old_fashioned_glass', \
    'Old fashioned glasses usually hold 180–300 ml'
POUR = {'type': 'video_observation', 'time_point': '4.0-6.2 s',
        'key_information': 'pouring duration = 2.2 seconds; the short old fashioned glass ends about half full'}


@P('B|a00819/01030|0', 'full', 'Torricelli through the full 28 mm bottle mouth (2.324 L poured into a small glass) -> '
                               'average flow rate = half the old fashioned glass capacity (180-300 ml) / 2.2 s = '
                               '40.9-68.2 mL/s')
def _(qa):
    qa['question'] = ('In the video, carbonated liquid is poured from a Fentimans bottle into a short, wide old '
                      'fashioned glass. The pouring lasts approximately 2.2 seconds (visible from 4.0 to 6.2 seconds), '
                      'after which the glass is about half full. Using the typical capacity of an old fashioned glass, '
                      'estimate the range of the average volumetric flow rate of the pour (in mL/s).')
    assert not qa.get('hidden_cv_chosen')
    c = code("V_low = 180.0  # mL, glass capacity\nV_high = 300.0  # mL\nfill = 0.5  # about half full\nt = 2.2  # s\n"
             "Q_low = fill * V_low / t\nQ_high = fill * V_high / t\n"
             "print(f'Average flow rate: {Q_low:.1f}–{Q_high:.1f} mL/s')")
    has(c['key_information'], '40.9–68.2 mL/s')
    rebuild(qa, [dict(POUR), web(OF_Q, OF_URL, OF_QUOTE, 'old fashioned glass capacity = 180–300 ml'), c], [
        'Step 1: Observe the video — the pouring occurs from t = 4.0 s to t = 6.2 s (2.2 s) into a short, wide old '
        'fashioned glass, which ends about half full.',
        'Step 2: Identify the physical relation — the average volumetric flow rate is the poured volume divided by the '
        'pouring time, Q = V / t, with V = half the glass capacity.',
        "Step 3: Use web_search(query='%s') to retrieve the typical capacity: old fashioned glasses usually hold "
        '180–300 ml.' % OF_Q,
        'Step 4: Use code_executor to compute Q = 0.5 × V_glass / 2.2 s for V_glass = 180 ml and 300 ml.',
        'Step 5: The average volumetric flow rate is 40.9–68.2 mL/s.'])
    qa['answer'], qa['answer_range'] = '40.9–68.2 mL/s', [40.9, 68.2]


@P('B|a00819/01030|1', 'full', 'fall height 0.15 m used as a Torricelli head through the full bottle mouth (1.06 L/s) '
                               '-> flow = half the old fashioned glass capacity / 2.2 s; bottle opening diameter -> '
                               'glass capacity (hidden); 6.34-8.45 -> 0.245-0.545 g/s')
def _(qa):
    qa['question'] = ('In the video at 4.0–6.2 s, a carbonated beverage is poured from a bottle into a short old '
                      'fashioned glass, with the liquid stream observed at a trajectory angle of 265° relative to the '
                      'wooden surface plane (0° = rightward along the plank); by the end of the pour the glass is about '
                      'half full. Calculate the range of the average mass flow rate of dissolved CO₂ entering the '
                      'glass, using the typical capacity of an old fashioned glass and typical carbonated beverage CO₂ '
                      'concentration specifications.')
    assert len(qa['hidden_cv_chosen']) == 2 and 'bottle opening' in qa['hidden_cv_chosen'][0]
    qa['hidden_cv_chosen'] = ['old fashioned glass capacity = 180–300 ml',
                              'CO2 concentration in carbonated beverages = 6-8 g/L']
    obs, _, co2, _ = qa['sources']
    obs = dict(obs, key_information='liquid stream trajectory angle = 265°; pouring duration = 2.2 s; '
                                    'the glass ends about half full')
    c = code("V_low = 0.180  # L, glass capacity\nV_high = 0.300  # L\nfill = 0.5\nt = 2.2  # s\n"
             "co2_low = 6.0  # g/L\nco2_high = 8.0  # g/L\n"
             "Q_low = fill * V_low / t\nQ_high = fill * V_high / t\n"
             "print(f'Volumetric flow rate: {Q_low:.5f}–{Q_high:.5f} L/s')\n"
             "print(f'CO2 mass flow rate: {Q_low * co2_low:.3f}–{Q_high * co2_high:.3f} g/s')")
    has(c['key_information'], '0.245–0.545 g/s')
    rebuild(qa, [obs, web(OF_Q, OF_URL, OF_QUOTE, 'old fashioned glass capacity = 180–300 ml'), co2, c], [
        'Step 1: Observe the video — the carbonated beverage is poured from about 4.0 to 6.2 s (2.2 s) with the stream '
        'at 265° relative to the wooden surface; the old fashioned glass ends about half full.',
        'Step 2: Identify the physical relation — the average volumetric flow rate is Q = 0.5 × V_glass / t, and the '
        'mass flow rate of dissolved CO₂ is ṁ = Q × c_CO₂.',
        "Step 3: Use web_search(query='%s') to retrieve the typical capacity: old fashioned glasses usually hold "
        '180–300 ml.' % OF_Q,
        "Step 4: Use web_search(query='%s') to retrieve the CO₂ concentration = 6–8 g/L." % co2['search_query'],
        'Step 5: Compute Q = 0.5 × 0.180 L / 2.2 s ≈ 0.04091 L/s to 0.5 × 0.300 L / 2.2 s ≈ 0.06818 L/s.',
        'Step 6: Compute the CO₂ mass flow range: 0.04091 L/s × 6 g/L ≈ 0.245 g/s to 0.06818 L/s × 8 g/L ≈ 0.545 g/s.',
        'Step 7: Run the calculation code; the average CO₂ mass flow rate is 0.245–0.545 g/s.'])
    qa['answer'], qa['answer_range'] = '0.245–0.545 g/s', [0.245, 0.545]


ROCKS_Q = 'rocks glass dimensions height diameter capacity oz'
ROCKS_QUOTE = ('Rocks Glass has an overall height of 3.25” (8.26 cm) and diameter of 2.625” (6.67 cm). The Single Rocks '
               'Glass holds a volume of 7 oz (20.7 cL).')


@P('B|a00819/01030|2', 'full', 'highball glass (15 cm, typo "250 cm" in the key) -> single rocks glass as seen '
                               '(8.26 cm, 6.67 cm, 20.7 cL); stream through the full 28 mm bottle mouth -> flow = half '
                               'the glass capacity / 2.2 s; 732 -> 21.8 Pa')
def _(qa):
    qa['question'] = ('In the video, a carbonated beverage stream pours into a short rocks glass at an angle of 265° '
                      'relative to the horizontal wooden surface during 4.0–6.2 s, filling the glass to about half its '
                      'capacity. Treating the pour as steady and the stream momentum flux (ρ × Q × v) as spread over '
                      'the glass cross-section, calculate the approximate pressure increase at the bottom of the glass '
                      'due to the stream impact, assuming the stream falls a height equal to 1.5 times the glass height '
                      'before hitting the liquid. Use the standard carbonated beverage density and the typical '
                      'dimensions and capacity of a single rocks glass.')
    assert len(qa['hidden_cv_chosen']) == 4
    qa['hidden_cv_chosen'] = ['rocks glass height = 8.26 cm', 'rocks glass diameter = 6.67 cm',
                              'beverage density = 1040 kg/m³', 'rocks glass capacity = 20.7 cL']
    obs, _, rho, _, _ = qa['sources']
    obs = dict(obs, key_information='liquid stream at 265° angle relative to horizontal; pouring duration = 2.2 s; '
                                    'the glass ends about half full')
    rho = {k: v for k, v in rho.items() if k != 'url_annotated'}
    c = code("import math\nH = 0.0826  # m, glass height\nD = 0.0667  # m, glass diameter\nV = 20.7e-5  # m^3 (20.7 cL)\n"
             "rho = 1040.0  # kg/m^3\ng = 9.81\nt = 2.2  # s\n"
             "Q = 0.5 * V / t\nh = 1.5 * H\nv = math.sqrt(2 * g * h)\nF = rho * Q * v\nA = math.pi * (D / 2) ** 2\n"
             "dP = F / A\nprint(f'Q = {Q:.4e} m^3/s, v = {v:.3f} m/s, F = {F:.4f} N, A = {A:.4e} m^2')\n"
             "print(f'pressure increase = {dP:.1f} Pa')")
    has(c['key_information'], 'pressure increase = 21.8 Pa')
    rebuild(qa, [obs, web(ROCKS_Q, 'dimensions.com/element/single-rocks-glass', ROCKS_QUOTE,
                          'single rocks glass: height 8.26 cm, diameter 6.67 cm, capacity 7 oz = 20.7 cL'), rho, c], [
        'Step 1: Observe the video — the stream pours at 265° relative to horizontal during 4.0–6.2 s (2.2 s) into a '
        'short rocks glass, creating foam upon impact; the glass ends about half full.',
        'Step 2: Identify the physics — the average flow rate is Q = 0.5 × V_glass / t; the stream reaches the liquid '
        'at v = √(2gh) after falling h = 1.5 × H; its momentum flux F = ρ Q v spread over the glass cross-section '
        'A = π (D/2)² gives ΔP = F / A.',
        "Step 3: Use web_search(query='%s') to retrieve the single rocks glass specifications: height 8.26 cm, "
        'diameter 6.67 cm, capacity 7 oz = 20.7 cL.' % ROCKS_Q,
        "Step 4: Use web_search(query='%s') to retrieve beverage density: ρ = 1040 kg/m³." % rho['search_query'],
        'Step 5: Calculate Q = 0.5 × 2.07×10⁻⁴ m³ / 2.2 s ≈ 4.70×10⁻⁵ m³/s.',
        'Step 6: Calculate the fall height h = 1.5 × 0.0826 m ≈ 0.124 m and the impact speed v = √(2 × 9.81 × 0.124) '
        '≈ 1.56 m/s.',
        'Step 7: Calculate the stream force F = ρ Q v ≈ 1040 × 4.70×10⁻⁵ × 1.56 ≈ 0.0763 N.',
        'Step 8: Calculate the glass cross-section A = π (0.03335 m)² ≈ 3.49×10⁻³ m².',
        'Step 9: Calculate ΔP = F / A ≈ 21.8 Pa; run the calculation code to confirm.',
        'Step 10: Final answer is approximately 21.8 Pa (about 2.2 mm of water pressure).'])
    qa['answer'] = '21.8 Pa (approximately 2.2 mm of water pressure)'


def main():
    cur = current(skip=('FIX4',))
    out = {}
    for qid, (f, change, note) in PATCH.items():
        qa = copy.deepcopy(cur[qid][0])
        t, d, n = qa['question_type'], qa['difficulty'], len(qa.get('hidden_cv_chosen') or [])
        f(qa)
        assert (qa['question_type'], qa['difficulty'], len(qa.get('hidden_cv_chosen') or [])) == (t, d, n), qid
        out[qid] = {'status': 'ok', 'qa': qa, 'change': change, 'note': note}
    with ThreadPoolExecutor(8) as ex:
        for qid, p in ex.map(one, [(q, r['qa']) for q, r in out.items()]):
            out[qid]['problems'] = p
            if p:
                out[qid]['status'] = 'fail'
            print(qid, out[qid]['change'], out[qid]['status'], p or '')
    json.dump(out, open(WORK + '/fix4_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    rp = json.load(open(WORK + '/review_post.json', encoding='utf-8'))
    for q, x in out.items():
        if x['status'] == 'ok':
            rp.setdefault(q, {})['FIX4'] = {'verdict': 'accept', 'notes': x['note']}
    json.dump(rp, open(WORK + '/review_post.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(out), 'patched,', sum(r['status'] == 'ok' for r in out.values()), 'ok')


if __name__ == '__main__':
    main()
