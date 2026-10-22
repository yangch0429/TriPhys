# Provenance (TriPhysBench release): fix_code/_v8_work/fix39.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""FIX39: deterministic fixes of final-recheck problems (R<n> = numbering of new39.txt) on the current version (without
the post layers): search steps aligned with their source's search_query, web_search sources swapped to a real search
result (search_log.jsonl: link and snippet checked) that states the value used, type labels fixed, and the R36 answer
recomputed. Every patched QA is checked again with final_recheck.one (ws quotes / steps / format / codes).
change = 'scs': question and answer unchanged (only the SCS judgement is redone); 'full': question / answer / type
changed (full rerun).
-> fix39_results.json {qid: {status, qa, change, note, problems}}
Usage: python fix39.py"""
import copy, json, re
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, get_steps, set_steps, sources_of
from current import current
from final_recheck import one

R = {int(n): q for n, q in re.findall(r'^\[R(\d+)\] (\S+)', open(WORK + '/new39.txt', encoding='utf-8').read(), re.M)}
_log = None


def logged(query, url_part, quote):
    """full link of the logged result of query whose link contains url_part and whose snippet contains quote"""
    global _log
    if _log is None:
        _log = []
        for line in open(WORK + '/search_log.jsonl', encoding='utf-8'):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            _log += [(d['query'], o) for o in d['raw'].get('organic', [])]
    for q, o in _log:
        if q == query and url_part in (o.get('link') or '') and quote in ' '.join((o.get('snippet') or '').split()):
            return o['link']
    raise ValueError('no logged result: %s | %s | %s' % (query, url_part, quote))


def step_sub(qa, old, new):
    steps, kind = get_steps(qa)
    hits = [i for i, s in enumerate(steps) if old in s]
    assert len(hits) == 1, (old, hits)
    steps[hits[0]] = steps[hits[0]].replace(old, new)
    set_steps(qa, steps, kind)


def requery(qa, old_q, new_q):
    """the search step reads web_search(query='<new_q>')"""
    steps = ' '.join(get_steps(qa)[0])
    for o in ("query='%s'" % old_q, 'query="%s"' % old_q):
        if o in steps:
            step_sub(qa, o, ("query='%s'" if "'" not in new_q else 'query="%s"') % new_q)
            return
    raise ValueError('step query not found: ' + old_q)


def swap(qa, j, query, url_part, quote, key=None):
    """web_search source j -> logged result (query, url, quote); its search step uses query"""
    s = sources_of(qa, 'web_search')[j]
    link = logged(query, url_part, quote)
    requery(qa, s['search_query'], query)
    s.update(url=link, search_query=query, verified_quote=quote)
    if key:
        s['key_information'] = key


def code_sub(qa, old, new, block=0):
    c = sources_of(qa, 'code_executor')[block]
    assert old in c['code'], old
    c['code'] = c['code'].replace(old, new)


PATCH = {}


def P(n, change, note):
    def deco(f):
        PATCH[n] = (f, change, note)
        return f
    return deco


# ---------------------------------------------------------------- search step query = source search_query
@P(4, 'scs', 'search step query aligned with the source query')
def _(qa):
    requery(qa, 'NBA regulation basketball weight mass 22 ounces to kilograms', 'NBA regulation basketball weight in kilograms')

@P(15, 'scs', 'search step query aligned with the source query')
def _(qa):
    requery(qa, 'typical mountain bike wheel radius 29 inch in meters', 'typical mountain bike wheel diameter in meters')

@P(16, 'scs', 'search step query aligned with the source query')
def _(qa):
    requery(qa, 'total arm weight percentage body weight 5.7% male', 'total arm weight percentage of body weight male')

@P(26, 'scs', 'search step query aligned with the source query; garbled characters removed from the video observation')
def _(qa):
    requery(qa, 'surface tension of water 72.8 mN/m at 20°C', 'surface tension of water at room temperature')
    v = sources_of(qa, 'video_observation')[0]
    assert 'object diameter =的开 0.025 m' in v['key_information']
    v['key_information'] = v['key_information'].replace('object diameter =的开 0.025 m', 'object diameter = 0.025 m')

@P(30, 'scs', 'search step query aligned with the source query')
def _(qa):
    requery(qa, 'regulation billiard ball weight 5.5 oz to kg', 'regulation billiard ball weight in kg')

@P(32, 'scs', 'search step query aligned with the source query')
def _(qa):
    requery(qa, 'spring steel Young"s modulus GPa typical', "spring steel Young's modulus GPa typical")

# ---------------------------------------------------------------- value used = value of the verified quote
@P(2, 'full', 'release height above the head = 1 ft (0.3048 m), the value of the verified quote (30-60 cm was '
              'unsupported); the range answer comes only from the observed athlete height range')
def _(qa):
    s = sources_of(qa, 'web_search')[1]
    assert s['verified_quote'].startswith('Typical release height for a free throw is 1 foot')
    s['key_information'] = "typical free throw release height = 1 foot (0.3048 m) above the top of the shooter's head"
    step_sub(qa, "Result: ∼30–60 cm above the shooter's head (source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10436204/).",
             "Result: typically 1 foot (0.3048 m) above the top of the shooter's head (source: %s)." % s['url'])
    step_sub(qa, 'Using the ranges: Minimum = 1.564 m + 0.30 m + 0.617 m = 2.481 m. Maximum = 1.667 m + 0.60 m + 0.617 m '
                 '= 2.884 m.', 'Using the observed height range: Minimum = 1.564 m + 0.3048 m + 0.617 m = 2.486 m. '
                               'Maximum = 1.667 m + 0.3048 m + 0.617 m = 2.589 m.')
    code_sub(qa, 'release_above_head_min = 0.30\nrelease_above_head_max = 0.60\n', 'release_above_head = 0.3048\n')
    code_sub(qa, 'h_min = athlete_height_min + release_above_head_min + h_above\n'
                 'h_max = athlete_height_max + release_above_head_max + h_above\n',
             'h_min = athlete_height_min + release_above_head + h_above\n'
             'h_max = athlete_height_max + release_above_head + h_above\n')
    qa['answer'] = '2.49–2.59 m above ground'
    qa['answer_range'] = [2.49, 2.59]

# ---------------------------------------------------------------- type labels
@P(3, 'full', 'TYPE_C -> TYPE_B: the question asks for a range of takeoff angles (range answer with answer_range)')
def _(qa):
    qa['question_type'] = 'TYPE_B'

@P(5, 'full', 'TYPE_C -> TYPE_A: the question asks for one value (clearance above the rim)')
def _(qa):
    qa['question_type'] = 'TYPE_A'

# ---------------------------------------------------------------- sources swapped to a result stating the value
BB = ('official basketball weight kg size 7', 'youtube.com/watch?v=woyErwkxKBM',
      'Weight: Approximately 22 ounces (624 grams)')

@P(9, 'scs', 'basketball mass source -> result stating 22 oz (624 g); mass 0.624 kg (answer 3.00 J unchanged)')
def _(qa):
    swap(qa, 0, *BB, key='regulation basketball mass = 22 oz (624 g = 0.624 kg)')
    step_sub(qa, 'finding it to be approximately 0.625 kg', 'finding it to be approximately 22 oz (624 g = 0.624 kg)')
    step_sub(qa, 'KE = 0.5 × 0.625 kg × (3.1 m/s)² = 3.00 J', 'KE = 0.5 × 0.624 kg × (3.1 m/s)² = 3.00 J')
    code_sub(qa, 'm = 0.625;', 'm = 0.624;')

@P(13, 'scs', 'basketball mass source -> result stating 22 oz (624 g); mass 0.624 kg (answer 0.45 J unchanged)')
def _(qa):
    swap(qa, 0, *BB, key='regulation basketball mass = 22 oz (624 g = 0.624 kg)')
    step_sub(qa, 'regulation NBA basketball mass is approximately 0.625 kg (source: Quora)',
             'a regulation size 7 basketball weighs approximately 22 oz (624 g = 0.624 kg)')
    step_sub(qa, 'KE = 0.5 * m * v² = 0.5 * 0.625 kg * (1.2 m/s)² = 0.5 * 0.625 * 1.44 = 0.5 * 0.9 = 0.45 J',
             'KE = 0.5 * m * v² = 0.5 * 0.624 kg * (1.2 m/s)² = 0.5 * 0.624 * 1.44 = 0.449 J ≈ 0.45 J')
    m = sources_of(qa, 'manual_calculation')[0]
    m['calculation'] = 'KE = 0.5 * 0.624 * 1.44 = 0.449 J ≈ 0.45 J'

@P(19, 'scs', 'marble diameter source -> result stating 16 mm')
def _(qa):
    swap(qa, 0, 'standard glass marble diameter mm', 'tureks.com/the-dimensions-of-a-marble',
         '16 mm (0.63 inches). Marbles with a 16 mm diameter are commonly used in traditional marble games.',
         key='standard marble diameter = 16 mm (16 mm marbles are commonly used in traditional marble games)')

@P(20, 'scs', 'thread friction source -> result stating 0.11-0.16')
def _(qa):
    swap(qa, 1, 'thread friction coefficient steel bolts typical range', 'lggindustrial.com/coefficient-of-friction',
         'the Coefficient of Friction is on the order of 0.04 LESS THAN the Nut Factor, and runs between 0.11 and 0.16.')

@P(23, 'scs', 'epoxy density source -> result stating 1.15 g/cm³; garbled g value in a step fixed')
def _(qa):
    swap(qa, 1, 'cured epoxy resin density g/cm3', 'analyzing-testing.netzsch.com', 'Density, 1.15 g/cm³.')
    step_sub(qa, 'g =1180.81 m/s²', 'g = 9.81 m/s²')

@P(24, 'scs', 'latent heat source -> result stating 334 kJ/kg')
def _(qa):
    swap(qa, 0, 'latent heat of fusion of ice kJ/kg', 'sciencedirect.com/topics/physics-and-astronomy/heat-of-fusion',
         'The latent heat of fusion of ice is 334 kJ/kg',
         key='latent heat of fusion of ice = 334 kJ/kg (334000 J/kg)')

@P(35, 'scs', 'crank length source -> result stating 170 mm')
def _(qa):
    swap(qa, 0, 'most common road bike crank length', 'reddit.com/r/cycling/comments/zkx14f',
         'most bikes have traditionally come with a 170mm crank',
         key='standard crank arm length: bikes have traditionally come with a 170 mm (0.170 m) crank')

@P(37, 'scs', 'spring constant source -> result stating about 30 N/m (value used unchanged); hidden entry = 30 N/m')
def _(qa):
    swap(qa, 0, 'rubber band spring constant N/m', 'vexforum.com/t/spring-constant-of-vex-rubber-bands',
         'should be ballpark of 30 for large displacements', key='rubber band spring constant k ≈ 30 N/m')
    step_sub(qa, 'to retrieve the spring constant k (typical range 20-40 N/m)',
             'to retrieve the spring constant k (about 30 N/m for large displacements)')
    step_sub(qa, 'with central values k=30 N/m', 'with k=30 N/m')
    assert qa['hidden_cv_chosen'][0] == 'rubber band spring constant = 20-40 N/m'
    qa['hidden_cv_chosen'][0] = 'rubber band spring constant = 30 N/m'

@P(39, 'scs', 'marble mass source -> result stating 4-5 g')
def _(qa):
    swap(qa, 0, 'standard glass marble weight grams', 'homework.study.com/explanation/how-much-does-a-marble-weigh',
         'A marble weighs nearly 4-5 gm.')

@P(36, 'full', 'swapped key_information of the two sources fixed, can-mass source -> result stating 14-15 g; the '
               'answer is recomputed from the 14-15 g range (the old 96.4 J used a rounded speed, 94-99 J was invented)')
def _(qa):
    can, gel = sources_of(qa, 'web_search')
    swap(qa, 0, 'empty aluminum soda can weight grams', 'fxtincan.com/how-many-cans',
         'A typical empty soda can weighs between 14 and 15 grams', key='empty soda can mass = 14-15 grams (0.014-0.015 kg)')
    gel['key_information'] = 'gelatin density = 720 kg/m³'
    steps, kind = get_steps(qa)
    steps[6:10] = [
        'Step 7: Calculate total soda can mass: m_cans = 6 × (0.014–0.015) kg = 0.084–0.090 kg.',
        'Step 8: Calculate total mass: m_total = 6.48 kg + (0.084–0.090) kg = 6.564–6.570 kg.',
        'Step 9: Calculate impact velocity from free fall: v = √(2gh) = √(2 × 9.81 m/s² × 1.5 m) = √(29.43) ≈ 5.42 m/s.',
        'Step 10: Use code_executor to compute KE = ½ × m_total × v² = m_total × g × h for both ends of the can-mass '
        'range: 96.59–96.68 J.']
    set_steps(qa, steps, kind)
    sources_of(qa, 'code_executor')[0]['code'] = (
        "import math\n"
        "h = 1.5; g = 9.81\n"
        "v = math.sqrt(2 * g * h)\n"
        "m_gelatin = 720 * 0.009\n"
        "print(f'Impact velocity: {v:.2f} m/s')\n"
        "for m_can in (0.014, 0.015):\n"
        "    m_total = m_gelatin + 6 * m_can\n"
        "    KE = 0.5 * m_total * v**2\n"
        "    print(f'can mass {m_can*1000:.0f} g: total mass {m_total:.3f} kg, kinetic energy {KE:.2f} J')\n")
    qa['answer'] = '96.59–96.68 J'
    qa['answer_range'] = [96.59, 96.68]


def main():
    cur = current(skip=('FIX39', 'FIX39W'))
    out = {}
    for n, (f, change, note) in sorted(PATCH.items()):
        qid = R[n]
        qa = copy.deepcopy(cur[qid][0])
        f(qa)
        out[qid] = {'status': 'ok', 'qa': qa, 'change': change, 'note': note, 'R': n}
    with ThreadPoolExecutor(8) as ex:
        for qid, p in ex.map(one, [(q, r['qa']) for q, r in out.items()]):
            out[qid]['problems'] = p
            if p:
                out[qid]['status'] = 'fail'
            print('R%d' % out[qid]['R'], qid, out[qid]['change'], out[qid]['status'], p or '')
    json.dump(out, open(WORK + '/fix39_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(out), 'patched,', sum(r['status'] == 'ok' for r in out.values()), 'ok')


if __name__ == '__main__':
    main()
