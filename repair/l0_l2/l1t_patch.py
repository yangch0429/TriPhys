# Provenance (TriPhysBench release): fix_code/_v8_work/l1t_patch.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L1T_PATCH: manual review of l1t_results.json (rev_l1t.txt / rev_l1t_fail.txt), deterministic patches on the L1T
rewrites:
- ok rewrites whose code / steps / sources still disagree with the hidden values or the evidence (13, 22, 24, 27, 72);
- failed rewrites: 6 failed only on the 'old value still used' check, a false positive of the scale factors of near()
  (e.g. 0.1 vs d*100, 0.002 vs 2): checked by hand and accepted unchanged; 2 failed on an unsupported value range
  (track length 0.51-0.69 m invented from '24 in'; COR 0.5-0.9 vs the quote 'about 0.9'): rebuilt on the quoted value.
Every patched / accepted QA is checked again with final_recheck.one; the review verdicts go to review_post.json (L1T).
-> l1t_results.json (backup l1t_results.before_patch.json), review_post.json
Usage: python l1t_patch.py"""
import copy, json, os, shutil
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, get_steps, set_steps, sources_of, run_code
from final_recheck import one
from fix39 import logged, step_sub, swap, code_sub, BB

PATCH = {}
FP = 'old-value check false positive (near() scale match); value, steps and code checked by hand'
REMARK = {  # accepted unchanged, with a review remark
    'B|a00165/00056|0': 'manual review: accepted (mu > tan(theta), so the block decelerates; self-consistent)',
    'B|a00257/00210|0': 'manual review: accepted (the mass cancels in the result)',
    'B|a00695/01740|0': 'manual review: accepted (45° incline from the quoted "finish at 45 degrees")',
    'B|a00786/00063|0': 'manual review: accepted (wording only)',
    'test:B|a00541/00012|0': 'manual review: accepted (negative lower bound allowed by the question wording)',
    'test:B|a00495/00248|1': 'manual review: accepted (1.5 m lies within the quoted 4.8-5.2 ft)'}
FALSE_POS = ['B|a00054/00145|0', 'B|a00329/00720|1', 'B|a00816/00706|2', 'B|a00315/00726|2',
             'test:B|a00611/02070|1', 'test:B|a00385/00050|2']


def P(qid, note):
    def deco(f):
        PATCH[qid] = (f, note)
        return f
    return deco


def out_of(qa):
    return run_code(sources_of(qa, 'code_executor')[0]['code'])['out']


@P('A|90e697a5-2cff-4ecc-a580-f4c230a5488c|2', 'code / step 11 use the CV hidden values (height 1.6944-1.7856 m, '
                                               'wrist speed 5.5546 m/s); answer recomputed')
def _(qa):
    code_sub(qa, 'H_min = 1.75\nH_max = 1.80\nv = 5.0\n', 'H_min = 1.6944\nH_max = 1.7856\nv = 5.5546\n')
    step_sub(qa, 'athlete height range 1.75-1.80 m, wrist speed 5.0 m/s',
             'athlete height range 1.6944-1.7856 m, wrist speed 5.5546 m/s')
    o = out_of(qa)
    assert 'Loss min: 63.36%' in o and 'Loss max: 64.13%' in o, o
    steps, kind = get_steps(qa)
    steps.append('Step 12: The code outputs a loss of 63.36% (minimum) and 64.13% (maximum); the basketball mass '
                 'cancels in the ratio, so the range comes from the athlete height range.')
    set_steps(qa, steps, kind)
    qa['answer'] = ('63.36–64.13% of the release mechanical energy is lost by the time the ball is caught at rest at '
                    'shoulder height (backboard collision + catch).')
    qa['answer_range'] = [63.36, 64.13]


@P('test:A|ed511e0d-b9ba-4b7e-91c2-41bf1bdaa994|2', 'basketball mass source -> result stating 22 oz (624 g) '
                                                    '(answer unchanged)')
def _(qa):
    s = sources_of(qa, 'web_search')[1]     # same search_query as source 0: set directly (no swap/requery)
    s.update(url=logged(*BB), search_query=BB[0], verified_quote=BB[2],
             key_information='regulation size 7 basketball mass ≈ 22 oz (624 g = 0.624 kg)')
    s.pop('url_annotated', None)
    step_sub(qa, "Use web_search(query='basketball diameter and weight specifications') to retrieve the basketball "
                 "mass: about 0.624 kg.",
             "Use web_search(query='%s') to retrieve the basketball mass: approximately 22 oz (624 g = 0.624 kg)."
             % BB[0])


@P('B|a00260/00257|0', 'carabiner strength source -> result stating 18-25 kN (20-40 kN was unsupported); hidden '
                       'entry, steps and answer updated')
def _(qa):
    swap(qa, 0, 'carabiner breaking strength kN typical climbing', 'mammut.com/us/en/stories-guides/carabiners',
         'Strength ratings vary depending on the type of carabiner, typically ranging from 18 to 25 kN.',
         key='carabiner breaking strength range = 18-25 kN')
    sources_of(qa, 'web_search')[0].pop('url_annotated', None)
    assert qa['hidden_cv_chosen'][1] == 'carabiner breaking strength range = 20-40 kN'
    qa['hidden_cv_chosen'][1] = 'carabiner breaking strength range = 18-25 kN'
    code_sub(qa, 'strength_low = 20.0', 'strength_low = 18.0')
    code_sub(qa, 'strength_high = 40.0', 'strength_high = 25.0')
    assert 'Safety factor range: 18.39 to 25.53' in out_of(qa)
    step_sub(qa, 'typical carabiner breaking strength range (20-40 kN)',
             'typical carabiner breaking strength range (18-25 kN)')
    step_sub(qa, '(20 kN and 40 kN)', '(18 kN and 25 kN)')
    step_sub(qa, '≈ 20.43 to 40.86', '≈ 18.39 to 25.53')
    qa['answer'] = 'The safety factor range is 18.39 to 25.53 (dimensionless).'
    qa['answer_range'] = [18.39, 25.53]


@P('B|a00257/00210|2', 'marble mass = 10 g as quoted (key / step said 5-10 g, "heaviest typical marble" removed from '
                       'the question)')
def _(qa):
    old = ', assuming it is the heaviest typical marble.'
    assert qa['question'].endswith(old)
    qa['question'] = qa['question'][:-len(old)] + '.'
    sources_of(qa, 'web_search')[0]['key_information'] = 'marble mass ≈ 10 g'
    step_sub(qa, 'to retrieve marble mass range = 5-10 g. For the heaviest typical marble, use m = 10 g = 0.010 kg.',
             'to retrieve the marble mass: about 10 g. Use m = 10 g = 0.010 kg.')
    code_sub(qa, '# kg, 10 g (heaviest typical marble)', '# kg, about 10 g')


@P('test:B|a00642/00628|2', 'marble mass = 5.2 g as quoted (520 g per 100; 5 g was not the quoted value); hidden '
                            'entry, steps and answer numbers updated (answer still No)')
def _(qa):
    sources_of(qa, 'web_search')[0]['key_information'] = '16mm glass marble mass = 5.2 g (a pack of 100 weighs 520 g)'
    assert qa['hidden_cv_chosen'][0] == '16mm glass marble mass = 5 g'
    qa['hidden_cv_chosen'][0] = '16mm glass marble mass = 5.2 g'
    code_sub(qa, 'm = 0.005  # kg', 'm = 0.0052  # kg')
    o = out_of(qa)
    for t in ('Fc = 0.006159 N', 'N = 0.037508 N', 'f_req = 0.035119 N', 'f_max = 0.014253 N', 'mu_req = 0.9363',
              'f_req = -0.020866 N'.replace('f_req', 'f_max - f_req')):
        assert t in o, (t, o)
    step_sub(qa, 'The source gives typical 16mm glass marble mass = 5-6 g. Use m = 0.005 kg.',
             'The source gives 520 g per pack of 100, i.e. 5.2 g per 16mm glass marble. Use m = 0.0052 kg.')
    step_sub(qa, 'F_c = 0.005922 N, N = 0.036065 N, f_req = 0.033768 N, f_max = 0.013705 N, μ_req = 0.9363, '
                 'margin = -0.020063 N', 'F_c = 0.006159 N, N = 0.037508 N, f_req = 0.035119 N, f_max = 0.014253 N, '
                                         'μ_req = 0.9363, margin = -0.020866 N')
    step_sub(qa, 'f_req = 0.033768 N, while f_max = 0.013705 N', 'f_req = 0.035119 N, while f_max = 0.014253 N')
    step_sub(qa, 'insufficient by 0.020063 N', 'insufficient by 0.020866 N')
    a = qa['answer']
    for x, y in (('0.033768', '0.035119'), ('0.013705', '0.014253'), ('−0.020063', '−0.020866')):
        assert a.count(x) == 1, x
        a = a.replace(x, y)
    qa['answer'] = a


@P('test:B|a00100/00228|2', 'track length = 0.61 m (24 in) as quoted (0.51-0.69 m was invented); the speed range '
                            'comes from the rolling-friction range only; hidden entry, steps and answer updated')
def _(qa):
    s = sources_of(qa, 'web_search')[1]
    s['key_information'] = 'wooden marble track length ≈ 24 inches (0.61 m)'
    assert qa['hidden_cv_chosen'][1] == 'wooden track length = 0.51-0.69 m'
    qa['hidden_cv_chosen'][1] = 'wooden track length = 0.61 m'
    code_sub(qa, 'L_min = 0.51\nL_max = 0.69\n', 'L = 0.61\n')
    code_sub(qa, 'print(f"Track length range: {L_min:.2f}-{L_max:.2f} m")', 'print(f"Track length: {L:.2f} m")')
    code_sub(qa, 'v_min = speed(L_min, mu_max)\nv_max = speed(L_max, mu_min)',
             'v_min = speed(L, mu_max)\nv_max = speed(L, mu_min)')
    code_sub(qa, 'print(f"Minimum speed: {v_min:.2f} m/s")\nprint(f"Maximum speed: {v_max:.2f} m/s")\n'
                 'print(f"Speed range: {v_min:.2f}-{v_max:.2f} m/s")',
             'print(f"Minimum speed: {v_min:.3f} m/s")\nprint(f"Maximum speed: {v_max:.3f} m/s")\n'
             'print(f"Speed range: {v_min:.3f}-{v_max:.3f} m/s")')
    assert 'Speed range: 1.431-1.459 m/s' in out_of(qa)
    step_sub(qa, 'Result: wooden marble track length is ≈24 inches (0.61 m), with typical range 20-27 inches '
                 '(0.51-0.69 m). Use L_min = 0.51 m and L_max = 0.69 m.',
             'Result: the track measures approximately 24 inches long (≈0.61 m). Use L = 0.61 m.')
    step_sub(qa, 'Because v increases with L and decreases with μ_r, the minimum speed uses L_min and μ_max, and the '
                 'maximum uses L_max and μ_min.',
             'Because v decreases with μ_r, the minimum speed uses μ_max and the maximum uses μ_min.')
    step_sub(qa, 'L_min = 0.51 m, L_max = 0.69 m', 'L = 0.61 m')
    step_sub(qa, 'Result: speed range = 1.31-1.55 m/s.', 'Result: speed range = 1.431-1.459 m/s.')
    qa['answer'] = '1.431-1.459 m/s'
    qa['answer_range'] = [1.431, 1.459]


@P('B|a00332/00025|1', 'plastic COR = 0.9 as quoted (0.5-0.9 was unsupported); the comparison is rebuilt on the '
                       'quoted value: energy lost in the bounce (e = 0.9) vs energy to lift the ball by the slide '
                       'length (TYPE_D, 3 hidden values kept)')
def _(qa):
    qa['question'] = ('In the video, a yellow ball drops from a chute and impacts a yellow slide from an estimated '
                      'height of 0.5 m at 4.0–8.0 s, with one visible bounce. Using the typical toy ball mass, the '
                      'typical toy slide length, and the typical coefficient of restitution of plastic balls, compare '
                      'the mechanical energy lost in this bounce with the energy required to lift the ball by the '
                      'slide length. Report the ratio and state which energy is larger.')
    s = sources_of(qa, 'web_search')[2]
    s['key_information'] = 'restitution coefficient of PP, acrylic and POM (plastics) ≈ 0.9; used e = 0.9'
    s.pop('url_annotated', None)
    assert qa['hidden_cv_chosen'][2] == 'coefficient of restitution = 0.5-0.9'
    qa['hidden_cv_chosen'][2] = 'coefficient of restitution = 0.9'
    sources_of(qa, 'code_executor')[0]['code'] = (
        "g = 9.81  # m/s^2\n"
        "h_drop = 0.5  # m\n"
        "m = 0.001  # kg (1.0 g)\n"
        "L = 0.41  # m\n"
        "e = 0.9\n"
        "\n"
        "h_bounce = e**2 * h_drop\n"
        "E_lost = m * g * (h_drop - h_bounce)\n"
        "E_lift = m * g * L\n"
        "ratio = E_lost / E_lift\n"
        "\n"
        "print(f\"Bounce height at e={e}: {h_bounce:.4f} m\")\n"
        "print(f\"Energy lost in the bounce: {E_lost:.6f} J\")\n"
        "print(f\"Energy to lift ball by slide length L={L} m: {E_lift:.6f} J\")\n"
        "print(f\"Ratio (energy lost / lifting energy): {ratio:.4f}\")\n")
    o = out_of(qa)
    for t in ('Bounce height at e=0.9: 0.4050 m', 'Energy lost in the bounce: 0.000932 J', 'L=0.41 m: 0.004022 J',
              'lifting energy): 0.2317'):
        assert t in o, (t, o)
    steps, kind = get_steps(qa)
    assert steps[4].startswith('Step 5:') and len(steps) == 12
    steps[4] = ("Step 5: Use web_search(query='polymer-on-polymer collision restitution data') to retrieve the "
                "coefficient of restitution of plastic balls (search yields about 0.9 for PP, acrylic and POM); "
                "use e = 0.9.")
    steps[6:] = [
        'Step 7: Compute the bounce height: h_bounce = e² × h_drop = 0.81 × 0.5 m = 0.4050 m.',
        'Step 8: Compute the mechanical energy lost in the bounce: ΔE = m g (h_drop − h_bounce) = m g h_drop (1 − e²).',
        'Step 9: Compute the energy required to lift the ball by the slide length: E_lift = m g L.',
        'Step 10: Compute the ratio ΔE / E_lift = h_drop (1 − e²) / L.',
        'Step 11: Execute code to compute and print all values: ΔE = 0.000932 J, E_lift = 0.004022 J, '
        'ratio = 0.2317.',
        'Step 12: State the comparison: the energy lost in the bounce is 0.2317 times the lifting energy, so the '
        'lifting energy is larger.']
    set_steps(qa, steps, kind)
    qa['answer'] = ('The energy lost in the bounce (about 0.000932 J with e = 0.9) is about 0.2317 times the energy '
                    'required to lift the ball by the slide length (about 0.004022 J); thus the lifting energy is '
                    'larger.')


@P('A|73935c70-6d88-4f4d-a4f3-9baf70afa5b1|2', 'key_information of the two anthropometric sources swapped back '
                                               '(each now matches its own quote)')
def _(qa):
    s1, s2 = sources_of(qa, 'web_search')[1:3]
    assert s1['verified_quote'].startswith('1- Height') and s2['verified_quote'].startswith('3-Shoulder Height')
    s1['key_information'], s2['key_information'] = s2['key_information'], s1['key_information']


@P('B|a00054/00145|2', 'glass-on-wood static friction = 0.49 as quoted (0.4-0.6 was unsupported); hidden entry, '
                       'steps, code and answer numbers updated (answer still Yes)')
def _(qa):
    s = sources_of(qa, 'web_search')[2]
    s['key_information'] = 'glass-on-wood static friction coefficient = 0.49'
    s.pop('url_annotated', None)
    assert qa['hidden_cv_chosen'][2] == 'glass-on-wood static friction coefficient = 0.4-0.6'
    qa['hidden_cv_chosen'][2] = 'glass-on-wood static friction coefficient = 0.49'
    for a, b in (('mu_min = 0.4\nmu_max = 0.6\n', 'mu = 0.49\n'),
                 ('F_static_min_m_min = mu_min * m_min', 'F_static_m_min = mu * m_min'),
                 ('F_static_min_m_max = mu_min * m_max', 'F_static_m_max = mu * m_max'),
                 ("print(f'Typical static friction coefficient range = {mu_min:.1f}-{mu_max:.1f}')",
                  "print(f'Glass-on-wood static friction coefficient = {mu:.2f}')")):
        code_sub(qa, a, b)
    c = sources_of(qa, 'code_executor')[0]
    c['code'] = c['code'].replace('F_static_min_m_', 'F_static_m_').replace('Minimum available', 'Available')
    o = out_of(qa)
    assert 'at m=0.004 kg = 0.01923 N' in o and 'at m=0.005 kg = 0.02403 N' in o and 'Answer = Yes' in o, o
    step_sub(qa, 'typical glass-on-wood static friction coefficient μ = 0.4-0.6.',
             'the glass-on-wood static friction coefficient μ = 0.49.')
    step_sub(qa, 'minimum available static friction = 0.01570 N; at m = 0.005 kg, centripetal force = 0.00493 N and '
                 'minimum available static friction = 0.01962 N',
             'available static friction = 0.01923 N; at m = 0.005 kg, centripetal force = 0.00493 N and available '
             'static friction = 0.02403 N')
    step_sub(qa, 'μ_req = 0.101 < 0.4, and each centripetal force is below the corresponding minimum available',
             'μ_req = 0.101 < 0.49, and each centripetal force is below the corresponding available')
    a = qa['answer']
    for x, y in (('Typical glass-on-wood static friction is μ = 0.4-0.6.', 'Glass-on-wood static friction is μ = 0.49.'),
                 ('< minimum available static friction 0.01570 N', '< available static friction 0.01923 N'),
                 ('< minimum available static friction 0.01962 N', '< available static friction 0.02403 N')):
        assert a.count(x) == 1, x
        a = a.replace(x, y)
    qa['answer'] = a


@P('test:B|a00495/00248|1', 'sedan height key = the quoted 4.8-5.2 ft (≈1.46-1.58 m), value used 1.5 m (1.4-1.6 m '
                            'was not the quoted range)')
def _(qa):
    sources_of(qa, 'web_search')[0]['key_information'] = ('sedan height = 4.8–5.2 ft (≈1.46–1.58 m); typical roof '
                                                          'height used h = 1.5 m')
    step_sub(qa, 'to retrieve typical sedan roof height specifications.',
             'to retrieve typical sedan roof height specifications: 4.8–5.2 ft (≈1.46–1.58 m), so h = 1.5 m.')


def main():
    path = WORK + '/l1t_results.json'
    bak = WORK + '/l1t_results.before_patch.json'
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    r = json.load(open(bak, encoding='utf-8'))
    notes = dict(REMARK)
    for qid, (f, note) in PATCH.items():
        qa = copy.deepcopy(r[qid]['qa'])
        f(qa)
        r[qid].update(status='ok', qa=qa, question_type=qa['question_type'])
        notes[qid] = 'manual patch: ' + note
    for qid in FALSE_POS:
        assert r[qid]['status'] == 'fail'
        r[qid]['status'] = 'ok'
        notes[qid] = FP
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, [(q, x['qa']) for q, x in r.items()]))
    bad = {q: p for q, p in res.items() if p}
    for q, p in bad.items():
        print('PROBLEM', q, p)
    for q, x in r.items():
        x['patch'] = notes.get(q)
        x['problems'] = res[q]
        if res[q]:
            x['status'] = 'fail'
    json.dump(r, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    rp = json.load(open(WORK + '/review_post.json', encoding='utf-8'))
    for q, x in r.items():
        if x['status'] == 'ok':
            rp.setdefault(q, {})['L1T'] = {'verdict': 'accept', 'notes': notes.get(q, 'manual review: accepted')}
    json.dump(rp, open(WORK + '/review_post.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(r), 'L1T,', sum(x['status'] == 'ok' for x in r.values()), 'ok,', len(bad), 'with problems')


if __name__ == '__main__':
    main()
