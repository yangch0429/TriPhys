# Provenance (TriPhysBench release): fix_code/_v8_work/post_l2_fix.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Fixes of L2 results found in the one-by-one review (review_log.json), each value taken from a logged search result
(search_log.jsonl: snippet of the cited url for the cited query):
- test:B|a00588/00347|0: car mass is the quoted "3,300 lbs (1,490.6 kg)", not 1500 kg (also = sibling QA |1);
- test:B|a00642/00628|1: funnel "diameter is 8\"" -> radius 0.1016 m (not a rounded 0.1 m);
- test:B|a00818/01984|0: "10-20 Pa·s, using 15" is not in the cited snippet -> "Honey, at 20 degrees C has a viscosity
  of roughly 14095 mpa - s" (quora, query 'honey density and dynamic viscosity reference values');
- A|f9afd55f|1: release height cited with the query that returned it ("Typical release height for a free throw is
  1 foot above the top of the shooter's head", value-free query, tag postfix);
- B|a00879/00373|0, B|a00721/00666|0, test:B|a00100/00228|1, B|a00816/00706|0: question / hidden wording matched to what the source
  states (ball-cloth rolling resistance; basketball on a hardwood court; ball-bearing rolling resistance; birch
  firewood calorific value);
- B|a00152/00167|2: drop height back to the original question's 1.2 m; restitution 0.75 (samaterials, "Basketball –
  Hardwood court ≈0.75") instead of brainly's 0.8 (an energy ratio of a homework problem); basketball mass from the
  chegg snippet (0.624 kg; the reddit snippet states 588-616 g) — same sources as B|a00721/00666|0.
Every changed code block is re-run and its output checked. Applied in place to l2_results.json (backup
l2_results_before_postfix.json); log -> post_l2_fix_log.json
Already applied fixes (res[qid]['postfix']) are skipped.
Usage: python post_l2_fix.py"""
import json, os, shutil
from fixlib import WORK, get_steps, set_steps, run_code, sources_of


def rep(qa, pairs):
    """replace in every text field of the QA; every old string must occur"""
    s = json.dumps(qa, ensure_ascii=False)
    for old, new in pairs:
        o, n = json.dumps(old, ensure_ascii=False)[1:-1], json.dumps(new, ensure_ascii=False)[1:-1]
        assert o in s, old
        s = s.replace(o, n)
    return json.loads(s)


def set_step(qa, i, text):
    steps, kind = get_steps(qa)
    assert steps[i].startswith('Step %d' % (i + 1)), steps[i]
    steps[i] = text
    set_steps(qa, steps, kind)


def rerun(qa, expect):
    code = sources_of(qa, 'code_executor')[-1]
    out = run_code(code['code'])
    assert out['ok'] and expect in out['out'], out
    code['key_information'] = out['out']
    return out['out']


FIXES = {}


def fix(qid):
    def deco(f):
        FIXES[qid] = f
        return f
    return deco


@fix('test:B|a00588/00347|0')
def _(qa):
    qa = rep(qa, [
        ('NASCAR car mass: approximately 3,300 lbs (1,500 kg) without driver and fuel',
         'NASCAR minimum weight: 3,300 lbs (1,490.6 kg) without driver and fuel'),
        ("'NASCAR vehicles have a base weight of approximately 3,300 lbs (1,500 kg) without the driver and fuel'",
         "'The minimum weight requirement for these vehicles is set at 3,300 lbs (1,490.6 kg) without including the "
         "driver and fuel'"),
        ('NASCAR car mass = 1500 kg', 'NASCAR car mass = 1490.6 kg'),
        ('m = 1500 kg', 'm = 1490.6 kg'),
        ('0.4 × 1500 × 9.81 = 5886 N', '0.4 × 1490.6 × 9.81 = 5849.1 N'),
        ('0.45 × 5886 = 2648.7 N', '0.45 × 5849.1 = 2632.1 N'),
        ('2648.7 N × 3 m = 7946.1 J ≈ 7.95 kJ', '2632.1 N × 3 m = 7896.3 J ≈ 7.90 kJ'),
        ('found 1500 kg from', 'found 3,300 lbs = 1,490.6 kg from'),
        ('0.4 × 1500 × 9.81) × 3 = 7946 J ≈ 7.95 kJ', '0.4 × 1490.6 × 9.81) × 3 = 7896 J ≈ 7.90 kJ'),
        ('m=1500;', 'm=1490.6;'),
        ('Approximately 7.95 kJ', 'Approximately 7.90 kJ'),
    ])
    rerun(qa, '7896.3')
    return qa


@fix('test:B|a00642/00628|1')
def _(qa):
    qa = rep(qa, [
        ('meadow funnel diameter is 8 inches (~20 cm diameter, ~10 cm radius)',
         'meadow funnel diameter is 8" (20.32 cm) -> swirling radius = half the diameter = 10.16 cm'),
        ('funnel swirling radius = 10 cm', 'funnel swirling radius = 10.16 cm'),
        ('Results indicate funnel diameters around 8 inches (~20 cm) → radius r = 0.1 m.',
         "Result from sweetpeatoys.com: 'meadow funnel diameter is 8\"' → diameter 0.2032 m, radius r = 0.1016 m."),
        ('(4.712 rad/s)² × 0.1 m.', '(4.712 rad/s)² × 0.1016 m.'),
        ('F = 0.0111 N or 11.1 mN.', 'F = 0.0113 N or 11.3 mN.'),
        ('indicated typical funnel diameters around 8 inches (~20 cm diameter, ~10 cm radius)',
         'returned a meadow funnel diameter of 8 inches (20.32 cm diameter, 10.16 cm radius)'),
        ('radius = 0.1;', 'radius = 0.1016;'),
    ])
    rerun(qa, '0.01128')
    qa['answer'] = '0.0113 N (11.3 mN)'
    return qa


@fix('test:B|a00818/01984|0')
def _(qa):
    ws = [s for s in sources_of(qa, 'web_search') if 'rawhoneyguide' in s['url']]
    assert len(ws) == 1
    ws[0].update({'url': 'https://www.quora.com/What-is-viscosity-in-water-and-honey',
                  'key_information': 'Honey, at 20 degrees C has a viscosity of roughly 14095 mPa·s = 14.095 Pa·s',
                  'search_query': 'honey density and dynamic viscosity reference values'})
    qa = rep(qa, [
        ("Use web_search(query='honey viscosity Pa s pascal seconds') to retrieve honey viscosity. From literature, "
         "typical honey viscosity at room temperature is 15 Pa·s.",
         "Use web_search(query='honey density and dynamic viscosity reference values') to retrieve honey viscosity. "
         "From quora.com: 'Honey, at 20 degrees C has a viscosity of roughly 14095 mPa·s' → η = 14.095 Pa·s."),
        ('η = 15 Pa·s.', 'η = 14.095 Pa·s.'),
        ('/ (9 × 15) = -1.25 × 10⁻³ m/s.', '/ (9 × 14.095) = -1.33 × 10⁻³ m/s.'),
        ('honey viscosity (15 Pa·s from typical room temperature range)',
         'honey viscosity (14095 mPa·s = 14.095 Pa·s at 20 °C from quora.com)'),
        ('terminal velocity of -1.25 × 10⁻³ m/s', 'terminal velocity of -1.33 × 10⁻³ m/s'),
        ('eta=15;', 'eta=14.095;'),
        ('-1.25 × 10⁻³ m/s', '-1.33 × 10⁻³ m/s'),
    ])
    rerun(qa, '-0.00132')
    return qa


@fix('A|f9afd55f-08d8-4753-9f76-df55cd8f4e59|1')
def _(qa):
    ws = [s for s in sources_of(qa, 'web_search') if 'inpredictable' in s['url']]
    assert len(ws) == 1
    ws[0].update({'key_information': "Typical release height for a free throw is 1 foot above the top of the "
                                     "shooter's head (1 ft = 0.3048 m)",
                  'search_query': 'average basketball player free throw release height above head'})
    set_step(qa, 3, "Step 4: Use web_search(query='average basketball player free throw release height above head') "
                    "to retrieve how high above the head the ball is typically released: 'Typical release height for "
                    "a free throw is 1 foot above the top of the shooter's head' (1 foot = 0.3048 m, from "
                    "https://www.inpredictable.com/2015/05/introducing-sharc-shot-arc-analysis.html).")
    return qa


@fix('B|a00879/00373|0')
def _(qa):
    return rep(qa, [
        ('given the standard rolling friction coefficient for billiard balls on wood surfaces',
         'given the standard ball-cloth rolling resistance coefficient of billiard balls'),
        ('rolling friction coefficient for billiard balls on wood = 0.005-0.015',
         'billiard ball-cloth rolling resistance coefficient = 0.005-0.015'),
        ('rolling resistance coefficient μ_r = 0.005 – 0.015 for billiard balls',
         'ball-cloth coefficient of rolling resistance μ_r = 0.005 – 0.015 for billiard balls'),
        ('retrieve the standard rolling resistance coefficient range for billiard balls on cloth/wood surfaces',
         'retrieve the standard ball-cloth rolling resistance coefficient range for billiard balls'),
        ('web_search found billiard ball rolling friction coefficient range 0.005-0.015',
         'web_search found the billiard ball-cloth rolling resistance coefficient range 0.005-0.015'),
    ])


@fix('B|a00721/00666|0')
def _(qa):
    return rep(qa, [
        ('Using standard basketball and ping pong ball specifications, and assuming',
         'Using standard basketball and ping pong ball specifications and the typical coefficient of restitution '
         'of a basketball on a hard court, and assuming'),
        ('coefficient of restitution for basketball on hard surface ≈ 0.75',
         'Basketball – Hardwood court: coefficient of restitution ≈ 0.75'),
        ('retrieve coefficient of restitution value of 0.75 for basketball on hard surfaces from samaterials.com',
         'retrieve the coefficient of restitution of a basketball on a hardwood court, 0.75, from samaterials.com '
         '(used for the hard concrete floor)'),
    ])


@fix('B|a00152/00167|2')
def _(qa):
    qa['question'] = ('In the video, an orange basketball is observed dropping from approximately 1.2 meters above a '
                      'concrete floor, bouncing once and transferring energy to a small white ball stacked on top of '
                      'it, launching the small ball vertically upward. Assuming a perfectly elastic collision between '
                      'the two balls during the energy transfer, and using standard specifications for both balls '
                      'along with the typical coefficient of restitution of a basketball on a hard court, calculate '
                      'the maximum height above the floor reached by the small white ball.')
    for s in qa['sources']:
        if s['type'] == 'video_observation':
            s['key_information'] = 'basketball drops from ~1.2 m height and bounces once'
        elif 'reddit' in s.get('url', ''):
            s.update({'url': 'https://www.chegg.com/homework-help/questions-and-answers/fully-inflated-basketball-mass-'
                             '0624-kg-volume-708-x-10m--swimmer-takes-basketball-bottom-s-q94370941',
                      'key_information': 'A fully inflated basketball has a mass of 0.624 kg',
                      'search_query': '"basketball mass" kg official weight'})
        elif 'brainly' in s.get('url', ''):
            s.update({'url': 'https://www.samaterials.com/content/coefficient-of-restitution.html',
                      'key_information': 'Basketball – Hardwood court: coefficient of restitution ≈ 0.75',
                      'search_query': 'coefficient of restitution basketball concrete floor'})
    qa['hidden_cv_chosen'] = ['basketball mass = 0.624 kg', 'ping pong ball mass = 0.0027 kg',
                              'basketball-hard court coefficient of restitution = 0.75']
    code = sources_of(qa, 'code_executor')[-1]
    assert 'h=1.1;' in code['code'] and 'e=0.8;' in code['code']
    code['code'] = code['code'].replace('h=1.1;', 'h=1.2;').replace('e=0.8;', 'e=0.75;')
    rerun(qa, '2.68 m')
    set_step(qa, 0, 'Step 1: Observe the video — the orange basketball drops from approximately 1.2 m above the '
                    'concrete floor and bounces once (visible at 0.0–4.0 s).')
    set_step(qa, 2, "Step 3: Use web_search(query='\"basketball mass\" kg official weight') to retrieve basketball "
                    "mass = 0.624 kg ('A fully inflated basketball has a mass of 0.624 kg', source: https://www.chegg"
                    ".com/homework-help/questions-and-answers/fully-inflated-basketball-mass-0624-kg-volume-708-x-10m--"
                    "swimmer-takes-basketball-bottom-s-q94370941).")
    set_step(qa, 4, "Step 5: Use web_search(query='coefficient of restitution basketball concrete floor') to retrieve "
                    "the coefficient of restitution of a basketball on a hard court = 0.75 ('Basketball – Hardwood "
                    "court. ≈0.75', source: https://www.samaterials.com/content/coefficient-of-restitution.html).")
    set_step(qa, 5, 'Step 6: Use code_executor to compute: g = 9.81 m/s², h = 1.2 m, M = 0.624 kg, m = 0.0027 kg, '
                    'e = 0.75; v = √(2×9.81×1.2) = 4.852 m/s; v_reb = 0.75×4.852 = 3.639 m/s; v_small = '
                    '(2×0.624/(0.624+0.0027))×3.639 = 7.247 m/s; H_max = (7.247)²/(2×9.81) = 2.68 m.')
    set_step(qa, 6, 'Step 7: Result is approximately 2.68 meters above the floor.')
    qa['exploration_summary'] = (
        'Narrative: (1) Physical event observed: stacked ball bounce demonstration where the basketball drops from '
        'about 1.2 m, rebounds from the concrete floor and transfers energy to the small white ball. (2) Video-observed '
        'value used: drop height ≈ 1.2 m, one bounce. (3) Latent graph properties: gravitational potential energy to '
        'kinetic energy, coefficient of restitution governs the rebound, momentum transfer in an elastic collision. '
        '(4) Tools called: web_search for basketball mass (0.624 kg, chegg), ping pong ball mass (2.7 g, vsport-tt), '
        'coefficient of restitution of a basketball on a hard court (0.75, samaterials). (5) Computation via '
        'code_executor yields maximum height = 2.68 m.')
    qa['answer'] = '2.68 meters above the floor'
    return qa


@fix('test:B|a00100/00228|1')
def _(qa):
    return rep(qa, [
        ('and the standard rolling friction coefficient for glass marbles on wood surfaces',
         'and the typical rolling resistance coefficient of a ball bearing as an estimate of the marble\'s rolling '
         'friction'),
        ('rolling friction coefficient = 0.002', 'ball bearing rolling resistance coefficient = 0.002'),
    ])


@fix('B|a00816/00706|0')
def _(qa):
    return rep(qa, [
        ('Use typical birch bark calorific values', 'Use typical birch firewood calorific values'),
        ('birch bark calorific value = 16.5-18.3 MJ/kg', 'birch firewood calorific value = 16.5-18.3 MJ/kg'),
    ])


def main():
    path = WORK + '/l2_results.json'
    if not os.path.exists(WORK + '/l2_results_before_postfix.json'):
        shutil.copy(path, WORK + '/l2_results_before_postfix.json')
    res = json.load(open(path, encoding='utf-8'))
    logp = WORK + '/post_l2_fix_log.json'
    log = json.load(open(logp, encoding='utf-8')) if os.path.exists(logp) else []
    for qid, f in FIXES.items():
        if res[qid].get('postfix'):  # applied in an earlier run
            continue
        old = res[qid]['qa']
        new = f(json.loads(json.dumps(old)))
        assert (new['question_type'], new['difficulty']) == (old['question_type'], old['difficulty'])
        res[qid]['qa'] = new
        res[qid]['postfix'] = True
        log.append({'qid': qid, 'answer': [old['answer'], new['answer']],
                    'hidden': [old.get('hidden_cv_chosen'), new.get('hidden_cv_chosen')],
                    'question_changed': old['question'] != new['question']})
    json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(log, open(logp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
