# Provenance (TriPhysBench release): fix_code/_v8_work/_l1t_hints.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""writes l1t_hints.json: per-QA guidance for the L1T repair of the 74 not-repaired QAs (n = index in NR, 1-based)"""
import json
from fixlib import WORK

NR = [x['qid'] for x in json.load(open(WORK + '/writeback_report.json', encoding='utf-8'))['not_repaired']]
U, UL = 'ws_unresolved#', 'ws_unlinked#'

# ---------------------------------------------------------------- real search results (search_log.jsonl)
def RPTCOL(key):
    return dict(key=key, url='rptcol2', quote='Upper Arm Length. 0.186H. Forearm Length. 0.146H.',
                value='0.186H; 0.146H', value_converted='upper arm = 0.186 x height; forearm = 0.146 x height',
                query='"forearm length" "percentage of height" anthropometric',
                target='upper-arm and forearm length as fractions of body height')
def SENS(key):
    return dict(key=key, url='sensazioni', quote='the shoulder height of an adult male typically represents about 81% of his total stature',
                value='81%', value_converted='shoulder height = 0.81 x stature',
                query='shoulder height as percentage of total height adult male anthropometry',
                target='adult male shoulder height as a fraction of stature')
def ROY_SH(key):
    return dict(key=key, url='roymech.org/Useful_Tables/Human/Human_sizes',
                quote='3-Shoulder Height, 1340, 1445 ; 4-Elbow Height, 1020, 1100 ; 5-Hip Height, 850, 935',
                value='1445; 935', value_converted='50th percentile male: shoulder height 1445 mm, hip height 935 mm',
                query='ergonomic body dimension chart male shoulder height standing height',
                target='50th percentile male shoulder and hip height')
def ROY_H(key):
    return dict(key=key, url='roymech.org/Useful_Tables/Human/Human_sizes', quote='1- Height, 1630, 1745, 1860, 1510, 1620, 1730',
                value='1745', value_converted='50th percentile male stature 1745 mm',
                query='human shoulder height fraction of body height standard table',
                target='50th percentile male stature')
def ROBS(key):
    return dict(key=key, url='body_part_weights', quote='Forearm 1.87% 1.57% Hand 0.65% 0.50% Forearm & Hand 2.52% 2.07%',
                value='Hand 0.65%; Forearm & Hand 2.52%', value_converted='male hand 0.65%, male forearm+hand 2.52% of body mass',
                query='forearm mass percentage of body weight male', target='male hand and forearm+hand mass fractions')
def ETSY(key, target):
    return dict(key=key, url='1835234453', quote='turn 90°-10×10cm', value='10×10cm', value_converted='0.10 m',
                query='marble run 90 degree corner track radius', target=target)
def MYSIM(key):
    return dict(key=key, url='mysimulator.uk/toys/new_14_marble_run',
                quote='Polished glass-on-wood contact has a fairly low rolling-friction coefficient (roughly 0.01-0.02)',
                value='0.01-0.02', is_range=True, query='"glass on wood" rolling friction coefficient',
                target='glass-on-wood rolling friction coefficient')
def HAUL(key, query):
    return dict(key=key, url='123778137', quote="There's no way you're leaving the ground with less than 220 pounds",
                value='220 pounds', value_converted='99.8 kg', query=query, target='loaded big-wall haul bag mass')
def DABOV(key, query):
    return dict(key=key, url='dabov.us/blog/how-to-perfect-your-pour', quote='Target flow rate: 4 ml/s (approximately 4 g/s)',
                value='4 ml/s', value_converted='4e-6 m³/s', query=query, target='typical pour flow rate')

L = {}  # n -> hint
def H(n, types, notes, drop=(), log=(), qedit=True, qnums=()):
    L[n] = dict(types=types, notes=notes, drop=list(drop), log=list(log), qedit=qedit, qnums=list(qnums))

SINGLE = ('The evidence gives ONE value ({v}): use exactly that value (no invented ranges or ± spreads); the answer '
          'is one value (TYPE_A, or TYPE_B with answer_range = null) unless a video-observed range remains an input.')

# ================================================================ A-class (hidden_cv_chosen = CV values: never change)
H(1, ['TYPE_D'], [
    'Wrist circle radius = forearm length 0.146 x height (elbow rotation sweeps the wrist), elbow circle radius = '
    'upper-arm length 0.186 x height (shoulder rotation sweeps the elbow); use the observed height 1.72 m.',
    'Centripetal accelerations a = v^2 / r with the CV speeds (hidden values); answer the comparison (ratio and both '
    'accelerations). No ranges unless an input is a range.'], log=[RPTCOL(U + '0')])
H(2, ['TYPE_A'], [SINGLE.format(v='front wheel o/a 1.2 lb = 0.544 kg; 700C bead seat diameter 621.95 mm'),
    'Rim speed = the observed wrist speed 4.6115 m/s; thin-hoop model I = m r^2 with r = 621.95/2 mm; KE = 1/2 I w^2. '
    'The question asks for one approximate value.'])
H(3, ['TYPE_D'], ['Keep the moorepants wheel (outer radius 0.336 m, mass 1.55 kg) for the wheel inertia.',
    'Shoulder height = 0.81 x the athlete height (hidden CV value); PE = m g h_shoulder; spin KE from the CV wrist '
    'speed at the rim. Answer which is larger and the factor (single value if all inputs are single).'],
  log=[SENS(U + '1')])
H(4, ['TYPE_A'], [
    'Reword the question: the mechanic turns the handlebar / fork assembly with his hands on the handlebar grips, '
    'hands moving at 0.9 m/s (24.0-28.0 s). Radius = half the handlebar width (42 cm standard) = 0.21 m; '
    'w = v / r. Remove the basket facts entirely.'], drop=[U + '0', UL + '0'])
H(5, ['TYPE_A', 'TYPE_B'], [
    'Body mass cancels in the ratio: remove the body-mass fact. Use the female segment mass fractions from the '
    'verified source (hand 0.56 %, forearm 1.38 %): KE ratio = (m_hand v_wrist^2) / (m_forearm v_elbow^2) with the CV '
    'speeds. The ratio is one value (TYPE_A, or TYPE_B with answer_range = null); the question must ask for the ratio, '
    'not a range.'], drop=[U + '1'])
H(6, ['TYPE_D', 'TYPE_B'], [
    'Shoulder height = 0.81 x athlete height (evidence). Reword if needed to: the percentage of the release '
    'mechanical energy (KE from the CV wrist speed + PE at release height from the verified standing-reach ratio '
    'and the observed height) that is lost by the time the ball is caught at rest at shoulder height (backboard '
    'collision + catch). Do not invent release or hoop heights or a coefficient of restitution; propagate only the '
    'observed height range.'])
H(7, ['TYPE_A', 'TYPE_C'], [
    'The regulation only states the backboard COR must be greater than 0.5 (no typical value). Reword the question to '
    'ask for the MINIMUM impulse magnitude the backboard must deliver: J_min = m v (1 + 0.5) for a normal impact with '
    'the observed speed 0.9 m/s and the verified ball mass. One value (TYPE_A).'])
H(8, ['TYPE_D'], [
    'Reword "waist height" to "hip height". Shoulder and hip heights scale with the observed athlete height 1.81 m '
    'using the 50th-percentile male ratios: shoulder 1445/1745, hip 935/1745 (evidence). Wheel mass 2,242 g (evidence). '
    'Answer: additional potential energy m g (h_shoulder - h_hip), one value (single inputs), plus the comparison.'],
  log=[dict(ROY_SH(U + '1')), ROY_H('ws_new#0')])
H(9, ['TYPE_C'], [
    'Yes/No answer ("No - ..."): KE from the observed 0.8 m/s speed vs the work to lift a rear wheel of 2.4-2.9 lb '
    '(1.09-1.32 kg) to overhead height (from the observed height 1.89 m); give the computed numbers.'])
H(10, ['TYPE_D'], [
    'Wrist motion moves the hand: mass fraction hand 0.65 %; elbow motion moves the forearm & hand: 2.52 % (male, '
    'evidence). Body mass cancels in the ratio: remove it. KE ratio = (0.65 v_wrist^2) / (2.52 v_elbow^2) with the CV '
    'speeds; one value.'], drop=[U + '2'], log=[ROBS(U + '0')])
H(11, ['TYPE_A'], [
    'The question already states the release height as 85 % of the athlete height: the shoulder-height fact needs no '
    'search; remove its web_search step / source and use 0.85 x 1.62 m.'], drop=[U + '1'])
H(12, ['TYPE_B'], [
    'Remove the legs-contribution fact. Reword: the vertical release velocity the ball needs to rise from the release '
    'height (athlete height + 0.305 m, verified) to the regulation rim height 3.05 m (verified) is sqrt(2 g (3.05 - '
    'h_release)); the additional vertical velocity the arms must supply = that value minus the observed body peak '
    'speed (CV). Range only from the CV height range and speed range.'], drop=[U + '2'])
H(13, ['TYPE_D'], ['Shoulder height = 0.81 x athlete height 1.95 m (evidence, single value); margin = KE(release, CV '
    'wrist speed) - m g (3.05 - h_shoulder); one value.'], log=[SENS(U + '2')])
H(14, ['TYPE_A'], [
    'Use the athlete mass exactly as the verified web_search quote states (and its kg conversion).',
    'Reword (a) to the rise of the centre of mass above its takeoff position: h = (v sin(67.8°))^2 / (2 g); keep (b) '
    'KE at takeoff = 1/2 m v^2. Use plain ASCII in the code (no "°" in code).'])
H(15, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='front wheel o/a 1.2 lb = 0.544 kg'),
    'KE before impact = m g h with h = 1.67 m: one value (about 8.9 J).'])
H(16, ['TYPE_D'], [
    'Basketball diameter from dimensions.com (the search step must use the evidence query and url exactly); merge the '
    'unlinked source into that search step. Runnable code (fix any syntax error): integrate the trajectory with and '
    'without quadratic drag using the CV speed and launch angle.'])

# ================================================================ B-class (hidden_cv_chosen = retrieved values, same count)
H(17, ['TYPE_B'], [
    'No evidence for a wheel mass: reword the question to ask for the angular momentum delivered to the paddle wheel '
    'about its axle, L = m v R, with v = sqrt(2 g 0.15) and R = half the HABA paddle-wheel diameter 5.5 in (6.985 cm). '
    'Range from the marble mass 5-10 g.', 'Hidden entries: "toy marble mass = 5-10 g", "paddle wheel radius = 6.985 '
    'cm" (clean text). Code must be real runnable Python.'])
H(18, ['TYPE_B', 'TYPE_A'], [
    'Building block height = 96 mm (evidence); track channel width = 7/8 in = 2.22 cm (evidence). No invented values '
    '(energy-transfer fraction, wheel mass): if the current model needs them, reformulate the question (e.g. the '
    'orange marble speed / kinetic energy at the wheel after dropping 2-3 blocks) so that every input is observed or '
    'retrieved, and use the channel width for a real purpose (e.g. clearance check with the marble).',
    'Hidden entries: marble mass, block height 96 mm, channel width 7/8 in, each as used.'])
H(19, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='steel-gravel friction 0.4 (NAVFAC)')])
H(20, ['TYPE_B', 'TYPE_A'], [SINGLE.format(v='steel-gravel friction 0.4 (NAVFAC)'),
    'The rake mass range stays an input (range answer) if it is a retrieved range.'])
H(21, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='deep groove ball bearing friction torque 0.0013 N·m')])
H(22, ['TYPE_B', 'TYPE_A'], [
    'Wooden ball mass 22.7 g (evidence). Track radius of curvature: use the curved track piece size 10x10 cm -> '
    'radius 0.10 m (evidence). Friction value as verified. Fix any garbled step text (e.g. "Step166").',
    'Hidden entries: wooden ball mass = 22.7 g, wood friction coefficient as used, track curve radius = 0.10 m.'],
  log=[ETSY(U + '2', 'marble run curved track piece radius')])
H(23, ['TYPE_A', 'TYPE_C'], [SINGLE.format(v='friction coefficient 0.4')])
H(24, ['TYPE_B', 'TYPE_A'], [
    'The rolling friction coefficient 0.01-0.02 is given in the question: it needs no search; remove that web_search '
    'step / source.'], drop=[UL + '1'])
H(25, ['TYPE_B'], ['Glass-on-wood rolling friction coefficient 0.01-0.02 (evidence) replaces the old value; range answer.'],
  log=[MYSIM(U + '2')])
H(26, ['TYPE_B'], [
    'Loaded haul bag mass: 220 lb = 99.8 kg (evidence, one value); carabiner strength range stays. Safety factor range '
    'from the carabiner range only.'], log=[HAUL(U + '1', 'typical haul bag weight kg climbing')])
H(27, ['TYPE_B', 'TYPE_A'], ['Fully loaded haul bag mass: 220 lb = 99.8 kg (evidence, one value).'],
  log=[HAUL(U + '0', 'how much does a packed climbing haul bag weigh')])
H(28, ['TYPE_B'], ['Glass-on-wood rolling friction coefficient 0.01-0.02 (evidence); range answer.'], log=[MYSIM(U + '1')])
H(29, ['TYPE_C'], [
    'No droplet friction coefficient exists in the evidence. Reword to a yes/no question: will the droplet slide off '
    'the spoon tilted at 15°? A superhydrophobic surface has a sliding angle below 10° (evidence) -> Yes, since 15° > '
    '10°; justify with numbers (e.g. the frictionless upper bound of the speed at the edge after L = 0.12 m, '
    'sqrt(2 g L sin 15°)).'],
  log=[dict(key=U + '0', url='emerginginvestigators', quote='contact angle of > 150° and sliding angle of < 10° is classified as superhydrophobic',
            value='< 10°', query='water droplet sliding angle on superhydrophobic surface degrees',
            target='sliding angle of water droplets on a superhydrophobic surface')])
H(30, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='glass-on-wood friction 0.299')])
H(31, ['TYPE_A'], [SINGLE.format(v='glass-on-wood friction 0.299')])
H(32, ['TYPE_C'], ['Curve radius of the wooden track: 90° turn piece 10x10 cm -> radius 0.10 m (evidence).'],
  log=[ETSY(U + '1', 'wooden marble track 90-degree curve radius')])
H(33, ['TYPE_B'], [
    'Figure height 4 in = 101.6 mm (4-inch action figure scale, evidence); figure mass 55.79-55.94 g; plastic-plastic '
    'friction 0.20-0.40. No base width is available: reformulate so the three retrieved values are used, e.g. the '
    'tilt angle at which the stacked figures start to SLIDE (arctan mu, range), the friction force on the top figure '
    'at that angle, and the potential energy released when one figure topples (centre of mass at half height).',
    'Remove the garbled text in the answer ("观察到"); all text in English.'],
  log=[dict(key=U + '1', url='Action_figure', quote='4" [101.6 mm], 4-inch action figures', value='4" [101.6 mm]',
            value_converted='0.1016 m', query='action figure base width dimensions', target='action figure height')])
H(34, ['TYPE_B', 'TYPE_A'], [
    'Hook mass 42.0 g (evidence); hook dimensions 27 x 14 x 15 mm (evidence) -> frontal area when falling vertically '
    '= 27 mm x 15 mm; drag coefficient: flat plate Cd = 1.28 (evidence, bluff body). Terminal velocity v = sqrt(2 m g '
    '/ (rho Cd A)); one value (no invented range).',
    'Hidden entries: hook mass = 42.0 g, hook frontal area as used, drag coefficient = 1.28.'],
  log=[dict(key=U + '2', url='shaped3', quote='A flat plate has Cd = 1.28', value='1.28',
            query='drag coefficient of a cube and flat plate table', target='bluff-body (flat plate) drag coefficient')])
H(35, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='marble-ramp friction coefficient 0.017')])
for n, k in ((36, 0), (37, 1), (38, 1), (39, 1), (40, 1), (41, 0)):
    H(n, ['TYPE_B'], ['Glass-on-wood rolling friction coefficient 0.01-0.02 (evidence, range): the answer is a range '
                      'propagated from it (TYPE_B); other retrieved values as their evidence states.'],
      log=[MYSIM(U + str(k))])
H(42, ['TYPE_C'], ['Curved track radius: 90° turn piece 10x10 cm -> 0.10 m (evidence). Remove any Chinese / garbled '
                   'text from the hidden entries and the solution.'], log=[ETSY(U + '0', 'marble run curved track radius')])
H(43, ['TYPE_A'], [SINGLE.format(v='magnetic dipole moment 1.5 A·m²')])
H(44, ['TYPE_A'], [SINGLE.format(v="Newton's cradle string length 4.496 in = 0.1142 m")])
H(45, ['TYPE_B'], [
    'Treat the small white ball as a golf ball (single scenario; no ball-type uncertainty): golf ball mass 45.5-45.9 g '
    '(evidence), basketball-floor COR 0.82-0.88 (evidence), basketball mass as verified. Remove the ping-pong scenario. '
    'Answer one range in English only (remove the Chinese text).',
    'Hidden entries: basketball mass, golf ball mass = 45.5-45.9 g, basketball-concrete COR = 0.82-0.88 (values as used).'])
H(46, ['TYPE_B', 'TYPE_A'], ['Ball-cloth sliding friction 0.15-0.4 (typical 0.2) (evidence).'],
  log=[dict(key=U + '0', url='drdavepoolinfo.com/faq/physics/physical-properties',
            quote='ball-cloth coefficient of sliding friction (μ): 0.15-0.4 (typical value: 0.2)',
            value='0.15-0.4 (typical value: 0.2)', is_range=True,
            query='pool table cloth static friction coefficient typical range', target='ball-cloth friction coefficient')])
H(47, ['TYPE_A', 'TYPE_B'], [
    'A ball rolling without slipping: static friction does no work, so the speed at the bottom is sqrt(2 g L sin20° '
    '/ (1 + k)) with k = 2/5 (solid ball) or as the current model states; the friction value only matters for the '
    'no-slip check (mu >= tan20° k/(1+k)). Reword the question so it is consistent with this (one value answer, or a '
    'range only if a range input is really used). Runnable code.'])
H(48, ['TYPE_A', 'TYPE_B'], [
    'Ball mass 8.5 g (marble-run marble, evidence); impact duration 0.7 ms (evidence); friction as verified. Remove the '
    'garbled text ("亚博体育官方").', 'Hidden entries: wood friction coefficient = 0.2, ball mass = 8.5 g, impact '
    'duration = 0.7 ms.'],
  log=[dict(key=U + '1', url='gravitrax.fandom', quote='The marbles measure 12.8 millimeters in diameter and weigh approx. 8.5 grams.',
            value='8.5 grams', query='plastic marble run balls weight specifications', target='marble run ball mass'),
       dict(key=U + '2', url='drussell/bats/impulse', quote='acts during the 0.7 millisecond contact time',
            value='0.7 millisecond', value_converted='0.0007 s', query='typical ball impact duration milliseconds',
            target='ball impact duration')])
H(49, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='glowing ember emissivity 0.68')])
H(50, ['TYPE_B'], [
    'No evidence for a view factor: remove it (radiation from the ember surface to the surroundings). Emissivity 0.8 '
    '(evidence), ambient 20-32 °C (evidence), Stefan-Boltzmann constant 5.6704e-8 W/(m²·K⁴) (evidence). Net power '
    'P = eps sigma A (T^4 - T_amb^4); range from the observed 800-1000 °C and the ambient range.',
    'Hidden entries: "birch bark emissivity = 0.8", "outdoor ambient temperature = 20-32 °C", '
    '"Stefan-Boltzmann constant = 5.6704e-8 W/(m²·K⁴)".'],
  drop=[U + '2'],
  log=[dict(key='ws_new#0', url='bionumbers.hms.harvard.edu', quote='Stefan-Boltzmann constant Value 5.6704 10^-8 W/(m^2×K^4)',
            value='5.6704 10^-8 W/(m^2×K^4)', value_converted='5.6704e-8 W/(m²·K⁴)',
            query='Stefan-Boltzmann constant value σ', target='Stefan-Boltzmann constant')])
H(51, ['TYPE_B'], ['Angle of repose 30-45° (evidence); soil-metal sliding friction 0.3 (evidence).'])
H(52, ['TYPE_D', 'TYPE_B'], [
    'No evidence for a paint-coating mass: remove it. Reword as a comparison: average impact force when the stamp is '
    'made with the whole wooden pencil (mass 5-10 g, evidence) versus with a loose pencil eraser alone (2-4 g, '
    'evidence) falling through the observed 8-12 cm, both stopped in the contact time 1 ms (evidence). F = m '
    'sqrt(2 g h) / dt; give both force ranges and the comparison.',
    'Hidden entries: "pencil eraser mass = 2-4 g", "fabric contact time = 1 ms", "wooden pencil mass = 5-10 g".'],
  drop=[U + '2'],
  log=[dict(key='ws_new#0', url='62013983', quote='A standard wooden graphite pencil usually weighs between 5 and 10 grams.',
            value='5 and 10 grams', is_range=True, query='how much does a pencil weigh grams', target='wooden pencil mass')])
H(53, ['TYPE_D'], [
    'Ball mass 1.0 g and slide length 41 cm (evidence); COR range 0.5-0.9 verified. The solution must really use the '
    'mass (e.g. the extra energy dissipated / retained between e = 0.5 and e = 0.9, m g h (e_hi^2 - e_lo^2)) and the '
    'slide length (as in the current geometry), besides the bounce-height difference.',
    'Hidden entries: ball mass = 1.0 g, slide length = 41 cm, COR = 0.5-0.9.'])
H(54, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='impact duration 3.5 ms')])
H(55, ['TYPE_A', 'TYPE_B'], [
    'No surface roughness coefficient exists in the evidence. Reword: laminar flow of the mist over the 0.6 m fabric; '
    'compute Re_L = rho U L / mu, the average skin-friction coefficient C_f = 1.328 / sqrt(Re_L) (Blasius, evidence), '
    'and ask for the average wall shear stress tau = C_f 1/2 rho U^2 on the fabric (one value).'],
  log=[dict(key=U + '0', url='sciencedirect.com/topics/engineering/skin-friction-coefficient',
            quote='classical Blasius solution for a laminar boundary layer is used to estimate the skin friction: 100% Laminar BL: (15-18) C f l a m = 1.328',
            value='1.328', value_converted='C_f = 1.328 / sqrt(Re_L)',
            query='Blasius solution skin friction coefficient laminar flat plate',
            target='laminar flat plate average skin-friction coefficient')])
H(56, ['TYPE_B'], ['Griddle temperature 225-375 °F = 107.2-190.6 °C (evidence); range answer.'])
H(57, ['TYPE_A'], [
    'The latent heat 80.4 kJ/kg is given in the question: it needs no search; remove its web_search step / source. '
    'Spoon head mass 13.75 g (evidence). One value.'], drop=[UL + '0'])
H(58, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='propane in the balloon 22 g')])
H(59, ['TYPE_A'], [SINGLE.format(v='resin viscosity 200000 mPa·s = 200 Pa·s; pool diameter about 2 in = 0.0508 m')])
H(60, ['TYPE_A'], [
    'The 0.15 m drop is not a pressure head for the bottle outflow. Reword: liquid leaves the 32 mm spout (evidence) '
    'at a typical pour rate of 4 ml/s (evidence); exit speed v0 = Q / A; speed at the bowl v = sqrt(v0^2 + 2 g 0.15); '
    'ask for the stream diameter just above the bowl (continuity d = d0 sqrt(v0 / v)) - one value.'],
  log=[DABOV('ws_new#0', 'water pouring flow rate from bottle typical mL/s')])
H(61, ['TYPE_A'], [
    'No evidence for the small-cup opening: remove it. Kinetic energy flux of the stream entering the large glass: '
    'P = 1/2 rho Q v^2 with the pour rate Q = 4 ml/s (evidence), v = sqrt(2 g 0.15), water density as verified. One value.',
    'Hidden entries: water density = 1000 kg/m³, pour flow rate = 4 ml/s.'],
  log=[DABOV(U + '1', 'typical pour rate of liquid from a cup')])
H(62, ['TYPE_C'], [
    'Reword to a yes/no question without a torque arm or friction: the stream falls 0.3 m into the large glass; is '
    'the impact force of the stream on the large glass (F = rho Q v, v = sqrt(2 g 0.3), Q = pour rate 4 ml/s, '
    'evidence) larger than the weight of the empty glass tumbler (191 g, evidence)? Answer "No - ..." with F, the '
    'weight and their ratio.',
    'Hidden entries: glass tumbler mass = 191 g, pour flow rate = 4 ml/s, water density = 1000 kg/m³.'],
  log=[DABOV(U + '1', 'typical pour rate of liquid from a cup')])
H(63, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='wet clay angle of repose 15°')])
H(64, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='mallet mass 36 g; rubber-steel COR 0.76')])
H(65, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='rolling friction coefficient 0.018')])
H(66, ['TYPE_B'], ['Rolling friction coefficient 0.01-0.02 (evidence, glass-on-wood); range answer.'], log=[MYSIM(U + '2')])
H(67, ['TYPE_B'], ['Friction on steel 0.3 (evidence, single); conveyor inclination 15-20° (evidence, range); range answer.'])
H(68, ['TYPE_C'], [SINGLE.format(v='glass-on-wood static friction 0.38; funnel angle 40°'),
                   'Yes/No answer with the computed numbers.'])
H(69, ['TYPE_A'], ['Body mass 199.8 lb = 90.63 kg (evidence); stopping distance on landing with bent knees 15 cm '
                   '(evidence). One value.'],
  log=[dict(key=U + '2', url='quora.com/Why-is-it-necessary', quote='your stopping distance will be more like 15 cm',
            value='15 cm', value_converted='0.15 m', query='landing from jump stopping distance knee bend cm',
            target='stopping distance when landing with bent knees')])
H(70, ['TYPE_A'], [SINGLE.format(v='bath bomb mass about 5 oz = 142 g')])
H(71, ['TYPE_D'], [
    'Bath bomb density from the product spec: about 5 oz (141.7 g) and 2.5 in diameter -> rho = m / (pi d^3 / 6) '
    '(evidence); rubber ball density 1.3 g/cm³ (evidence); warm water density 0.995 g/cm³ (evidence) for the '
    'float/sink comparison (no container depth). Ball diameter 7.5 cm (observed) gives each ball mass; keep the '
    'release height as in the current QA. Compare (i) impact KE and (ii) float/sink.',
    'Hidden entries: bath bomb mass and diameter = 5 oz and 2.5 in, rubber ball density = 1.3 g/cm³, warm water '
    'density = 0.995 g/cm³ (clean text, no garbled "162 cm").'],
  log=[dict(key=U + '0', url='facebook.com/groups/1013227208778278', quote='Each bath bomb is approximately 5oz in weight and 2.5" in diame',
            value='5oz; 2.5"', value_converted='141.7 g; 6.35 cm', query='bath bomb diameter and weight',
            target='bath bomb mass and diameter'),
       dict(key=U + '2', url='bodybonbon', quote='Density of Warm Water – 0.995 g/cm³', value='0.995 g/cm³',
            value_converted='0.995 g/cm³', target='warm water density')])
H(72, ['TYPE_B'], [
    "Ball mass 0.23 kg, string length 30 cm, steel ball contact time 75-140 μs (evidence). Hidden entries must state "
    "the values (\"Newton's cradle ball mass = 0.23 kg\", \"suspension string length = 30 cm\", \"steel ball contact "
    "time = 75-140 μs\"), never \"retrieved via web_search\". Range answer from the contact-time range."])
H(73, ['TYPE_B'], ['Metal-on-masonry static friction 0.30-0.70 (evidence, range): range answer (TYPE_B).'],
  log=[dict(key=U + '0', url='static-coefficient', quote='Metal on masonry, 0.30–0.70', value='0.30–0.70', is_range=True,
            query='typical static friction coefficient metal against stone', target='metal-on-masonry static friction')])
H(74, ['TYPE_A', 'TYPE_B'], [SINGLE.format(v='soil-steel friction 0.40; bed length 8 ft = 2.44 m')])

assert sorted(L) == list(range(1, 75)), sorted(set(range(1, 75)) - set(L))
out = {NR[n - 1]: h for n, h in sorted(L.items())}
json.dump(out, open(WORK + '/l1t_hints.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(out), 'hints')
