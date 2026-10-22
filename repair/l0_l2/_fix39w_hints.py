# Provenance (TriPhysBench release): fix_code/_v8_work/_fix39w_hints.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""writes fix39w_hints.json: L1T guidance (l1t.py with L1T_SET=fix39w) for the recheck-problem QAs whose answer must
change because the quote states a different value than the one used (R<n> = numbering of new39.txt)"""
import json, re
from fixlib import WORK

R = {int(n): q for n, q in re.findall(r'^\[R(\d+)\] (\S+)', open(WORK + '/new39.txt', encoding='utf-8').read(), re.M)}
L = {}


def H(n, types, notes, log=(), qedit=True, qnums=(), drop=()):
    L[n] = dict(types=types, notes=notes, drop=list(drop), log=list(log), qedit=qedit, qnums=list(qnums))


def W(key, url, quote, value, current, target, conv=None, query=None, is_range=False, direct=False):
    """replacement of the value of an existing search step (kind ws); direct: url = full url of the existing source
    whose L0-verified quote (page text, not in the search log) states the value"""
    d = dict(key=key, kind='ws', url=url, quote=quote, value=value, value_converted=conv or value, current=current,
             target=target, is_range=is_range, direct=direct)
    if query:
        d['query'] = query
    return d


KEEP = 'Keep the question text exactly as it is unless stated otherwise; the other retrieved facts stay unchanged.'
SINGLE = ('The evidence gives ONE value ({v}): use exactly that value (no invented ranges, midpoints or ± spreads).')

# ---------------------------------------------------------------- A-class (hidden_cv_chosen = CV values: never change)
H(2, ['TYPE_B'], [
    SINGLE.format(v='release 1 foot = 0.3048 m above the top of the head'),
    'Maximum height = observed athlete height (range 1.564-1.667 m from the video) + 0.3048 m + v0^2 sin^2(54°)/(2g); '
    'the range answer comes only from the observed height range.', KEEP],
  log=[W('fix#2', 'inpredictable.com/2015/05/introducing-sharc', 'Typical release height for a free throw is 1 foot '
         'above the top of the shooter\'s head.', '1 foot', '30–60 cm', 'release height above the shooter\'s head',
         conv='0.3048 m', query='basketball release height above head shooter centimeters')])
H(6, ['TYPE_D'], [
    'Mid-range release angle 52-55° (verified source, a range: no midpoint) and three-point release angle 45° '
    '(evidence, single value). Compute H = v^2 sin^2(θ)/(2g) for 52° and 55° and for 45°, and the percentage change '
    'from the mid-range height to the three-point height as a range (both ends); answer_range = [lo, hi] of the '
    'percentage change.', KEEP],
  log=[W('fix#2', 'secretsofshooting.com/physics-based', 'a 3-point shot is approximately 45-degrees', '45-degrees',
         'approximately 45 degrees', 'three-point shot release angle', conv='45°',
         query='three point shot release angle degrees')])
H(7, ['TYPE_A'], [
    SINGLE.format(v='ideal body weight for a 186 cm male = 80.4 kg'),
    'Forearm+hand mass = 0.022 x 80.4 kg (verified source); KE = 0.5 m v^2 with the observed wrist speed.', KEEP],
  log=[W('fix#2', 'calculat.io/en/weight/ideal-weight-metric/186-cm--male', 'Ideal Weight for 186 cm Male is 80.4 kg.',
         '80.4 kg', 'Healthy BMI range 18.5-24.9 kg/m², 64-86 kg', 'ideal body weight for a 186 cm man',
         query='ideal weight for 186 cm male')])
H(8, ['TYPE_A'], [
    SINGLE.format(v='front wheel 950 g = 0.95 kg'),
    'Reword the question minimally so the carried wheel is the detached FRONT wheel; PE = m g h with h = half the '
    'observed height 1.81 m.'],
  log=[W('fix#1', 'https://www.mbr.co.uk/buyers_guide/best-mountain-bike-wheels-360715', 'Weight: 950g front, 1120g rear', '950g',
         '0.85 kg', 'mountain bike front wheel mass', conv='0.95 kg', query='aluminum mountain bike wheelset weight', direct=True)])
H(10, ['TYPE_C'], [
    'Three-point release angle 45° (evidence, single value; no 45-55° range). Release speed = the observed wrist '
    'speed 4.8397 m/s (hidden CV). Do not use an invented release height: the maximum horizontal distance on level '
    'ground at 45° is R = v^2/g; since the rim (3.05 m) is above any release height, the ball cannot reach the hoop '
    'if R < 6.75 m. Answer "No" with the computed range and the shortfall 6.75 m - R. Remove the invented release '
    'height 2.0 m; the hoop height may stay as the argument that the rim is above the release point.',
    'The code must print R and the shortfall; the answer uses exactly the printed values.', KEEP],
  log=[W('fix#3', 'secretsofshooting.com/physics-based', 'a 3-point shot is approximately 45-degrees', '45-degrees',
         '45-55°', 'typical three-point shot release angle', conv='45°',
         query='three point shot release angle degrees')])
H(12, ['TYPE_D'], [
    SINGLE.format(v='mean layup release angle 49.934°'),
    'Ratio R_45 / R_obs = sin(90°) / sin(2 x 49.934°) is independent of speed and mass: one ratio, answer_range = '
    'null. Remove the invented 0.9 speed factor; the release speed (observed wrist speed) may be used to print both '
    'ranges. The ball-mass source is not needed: remove it.',
    'Reword the question minimally: compare the range at 45° to the range at the typical (mean) layup release angle, '
    'answered as one ratio.'],
  drop=(),
  log=[W('fix#2', 'researchgate.net/publication/386015868', 'The mean release angle of unsuccessful layup shots was '
         '49.9340°', '49.9340°', '40-60°', 'mean layup release angle', conv='49.934°',
         query='layup release angle degrees biomechanics')])
H(14, ['TYPE_D'], [
    SINGLE.format(v='average NBA player 217.62 lb = 98.71 kg'),
    'KE = 0.5 m v^2 for the three observed speeds with m = 98.71 kg; answer the comparison (all three energies and '
    'the two ratios) with the printed values.', KEEP],
  log=[W('fix#1', 'reddit.com/r/nba/comments/6yazpf', 'The average NBA player is 6\'7.4" and 217.62lbs.', '217.62lbs',
         '90-100 kg', 'average NBA player mass', conv='98.71 kg', query='typical pro basketball athlete weight')])
H(34, ['TYPE_D'], [
    SINGLE.format(v='average NBA player 217.62 lb = 98.71 kg'),
    'KE_body = 0.5 x 98.71 kg x (3.2 m/s)^2; efficiency = KE_ball / KE_body; compare with 20-25% (verified).', KEEP],
  log=[W('fix#3', 'reddit.com/r/nba/comments/6yazpf', 'The average NBA player is 6\'7.4" and 217.62lbs.', '217.62lbs',
         '98 kg', 'average NBA player mass', conv='98.71 kg', query='average basketball player weight kilograms')])

# ---------------------------------------------------------------- B-class (hidden = retrieved values, count fixed)
H(17, ['TYPE_B'], [
    SINGLE.format(v='golden syrup density 1430 kg/m³'),
    'Viscosity range 2000-10000 cP stays (verified): Re range from the viscosity range only, ρ = 1430 kg/m³. Hidden '
    'entries: "golden syrup density = 1430 kg/m³", "syrup viscosity = 2000-10000 cP".', KEEP],
  log=[W('fix#1', 'https://en.wikipedia.org/wiki/Golden_syrup', 'Its density is approximately 1430 kg·m −3 at room temperature', '1430',
         '1320-1480 kg/m³', 'golden syrup density', conv='1430 kg/m³', query='golden syrup density technical data', direct=True)])
H(18, ['TYPE_C'], [
    'The answer must start with "Yes" or "No" (no "MAYBE"). Reword the question so it is decided by the computed '
    'ranges: ask whether the per-tooth compressive stress is guaranteed to exceed the ice compressive strength for '
    'every combination of shredder torque and ice strength within their typical ranges. Answer: "No" — the stress '
    'range 12.3-24.5 MPa (printed by the code) does not exceed the maximum ice strength 25 MPa, although it exceeds '
    'the lower strengths. answer_range = null. Keep all values, sources and the 8 teeth / 0.3 m given data.',
    'The code prints the stress range and the decision ("No: ...") consistent with the answer.'])
H(21, ['TYPE_B'], [
    SINGLE.format(v='lamp base diameter 8 inch = 0.2032 m'),
    'Lamp mass 1.8-2.7 kg and wood friction 0.25-0.5 stay ranges (verified). Tipping limit F_tip = m g (d/2)/h with '
    'd = 0.2032 m; at each end the controlling (smaller) limit; the range answer from the mass and friction ranges. '
    'Hidden entries: "lamp base mass = 1.8-2.7 kg", "wooden surface static friction coefficient = 0.25-0.5", '
    '"lamp base diameter = 8 in (20.32 cm)".', KEEP],
  log=[W('fix#2', 'amazon.com/lamp-base', 'Metal Base with Diameter 8 inch Table lamp', '8 inch', '15-20 cm',
         'table lamp base diameter', conv='0.2032 m', query='table lamp base diameter inches')])
H(22, ['TYPE_B'], [
    SINGLE.format(v='mixed soup density 1049.999952 kg/m³ ≈ 1050 kg/m³'),
    'Reword "cooking broth density" to "soup density" (the liquid is a soup); PE = ρ V g h, one value, '
    'answer_range = null. hidden_cv_chosen stays empty.'],
  log=[W('fix#0', 'kg-m3.com/material/soup-mixed', 'Mass density of Soup, mixed is 1049.999952 kg/m3.',
         '1049.999952', '1000-1100 kg/m³', 'mixed soup density', conv='1050 kg/m³',
         query='soup mixed mass density')])
H(27, ['TYPE_B'], [
    'Soil density range 1000-1500 kg/m³ (evidence): PE = ρ V g h for both ends; range answer in kJ with '
    'answer_range = [lo, hi]; no "average" value, no stray symbols (remove the "†").', KEEP],
  log=[W('fix#0', 'sciencedirect.com/topics/engineering/soil-density', 'Typical soil density values are within the '
         'range of 1000–1500 kg/m3', '1000–1500', '1000-1600 kg/m³', 'typical soil density',
         query='soil bulk density range kg/m3', is_range=True)])
H(28, ['TYPE_B'], [
    SINGLE.format(v='bottle bore 19.0 mm'),
    'Volume = π (d/2)^2 sqrt(2 g h) t with d = 0.019 m: one value (answer_range = null). hidden_cv_chosen stays '
    'empty.', KEEP],
  log=[W('fix#0', 'https://www.liquorglassbottles.com/spirits-bottle-neck-finish-types-guide.html', 'A bottle from Manufacturer A may have a 19.0mm bore',
         '19.0mm', '18-20 mm', 'beverage bottle inner bore diameter', conv='0.019 m',
         query='bottle neck inner bore diameter standard beverage', direct=True)])
H(29, ['TYPE_B'], [
    SINGLE.format(v='16 mm glass marble weight 5.5 g'),
    'The observed marble diameter is 1.6 cm = 16 mm, so the 16 mm marble mass applies: F = m v^2 / r, one value, '
    'answer_range = null. hidden_cv_chosen stays empty.', KEEP],
  log=[W('fix#0', 'made-in-china.com/product/Wholesale-Glass-Marbles-16mm', '16mm; Weight: 5.5g', '5.5g', '5-20 g',
         '16 mm glass marble mass', conv='0.0055 kg', query='weight of a 16mm glass marble grams')])
H(31, ['TYPE_A'], [
    SINGLE.format(v='1 kJ per match = 1000 J'),
    'ΔT = (20 x 1000 J x 0.01) / (m_foil c_Al). The aluminum specific heat 897 J/(kg·K) is not retrieved: state it '
    'in the question as given data (add exactly "specific heat of aluminum 897 J/(kg·K)").'],
  qnums=[897],
  log=[W('fix#0', 'ieeexplore.ieee.org/document/1383468', '1 kj is equal to one match', '1 kj', '1500 J',
         'chemical energy released by one match', conv='1000 J', query='how much energy does one match produce')])
H(38, ['TYPE_B'], [
    SINGLE.format(v='rubber band spring constant 17.38 N/m'),
    'Ball mass 56.0-59.4 g and rubber-on-wood friction 0.70-0.75 stay ranges (verified): speed range from those, '
    'with k = 17.38 N/m (vf = 0 if the ball stops). Hidden entries: "ball mass = 56.0-59.4 g", "rubber-on-wood '
    'friction coefficient = 0.70-0.75", "rubber band spring constant = 17.38 N/m".', KEEP],
  log=[W('fix#3', 'https://www.wired.com/2012/08/do-rubber-bands-act-like-springs/', 'spring constant of 17.38 N/m', '17.38 N/m',
         '15-25 N/m', 'rubber band spring constant', query='calculate spring constant for rubber band', direct=True)])

out = {R[n]: h for n, h in sorted(L.items())}
json.dump(out, open(WORK + '/fix39w_hints.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(out), 'hints')
