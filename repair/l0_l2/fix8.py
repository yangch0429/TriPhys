# Provenance (TriPhysBench release): fix_code/_v8_work/fix8.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""FIX8: ground-truth format fixes for the ACC scorer (experiment_right metrics_final._acc_type_b, which takes the GT
range from the answer text with normalize_range_with_unit and only falls back to answer_range when no 'a-b' range is
found; see evaluation/scripts/gt_parse_check.py); applied after FIX7, question / question_type / difficulty /
hidden values unchanged.
  range text the scorer misreads (comma thousands separator, e-notation, minus sign of the lower bound):
    a00106/00861|1   '38,818–51,448 W' was read as 818–51 -> '38818–51448 W'
    a00137/00035|0, a00695/01740|1   e-notation ranges were read as e.g. 5–6.780185 -> decimal J (+ mJ)
    test:a00541/00012|0   '-0.304–0.590' was read as 0.304–0.590 -> '-0.304 to 0.590' (scorer uses answer_range)
  TYPE_B with a single value and no answer_range (ACC 0 for every response): accepted band = value ±5%, printed by
  the code, stated in the answer, answer_range set
    a00452/02213|0, a00861/00189|0, a00879/00373|2, a00670/00003|0, 1e3b818d|1, 90e697a5|1
  (test:2aac5631|0 'No — ... -5.51 m' is left: the scorer drops the minus sign of predictions; listed in the report)
-> fix8_results.json {qid: {status, qa, change, note, problems}}
Usage: python fix8.py"""
import copy, json, re
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, get_steps, set_steps
from current import current
from final_recheck import one
from strict_check import problems as strict_problems
from fix4 import code, has

PATCH = {}


def P(qid, change, note):
    def deco(f):
        PATCH[qid] = (f, change, note)
        return f
    return deco


def code_src(qa):
    cs = [i for i, s in enumerate(qa['sources']) if s['type'] == 'code_executor']
    assert len(cs) == 1, cs
    return cs[0]


def set_code(qa, c):
    i = code_src(qa)
    qa['sources'][i] = code(c)
    return qa['sources'][i]['key_information']


def last_step(qa, old, new):
    steps, kind = get_steps(qa)
    assert steps[-1].count(old) == 1, (old, steps[-1])
    steps[-1] = steps[-1].replace(old, new)
    set_steps(qa, steps, kind)


# ---------------------------------------------------------------- range text the scorer misreads
@P('B|a00106/00861|1', 'answer', "'38,818–51,448 W' (read as 818–51 by the scorer) -> '38818–51448 W'")
def _(qa):
    assert qa['answer_range'] == [38818, 51448]
    qa['answer'] = ('The flame emits approximately 38818–51448 W (38.8–51.4 kW) of total net radiative power to its '
                    'surroundings.')
    last_step(qa, 'the code gives 38,818–51,448 W.', 'the code gives 38818–51448 W.')


@P('B|a00137/00035|0', 'answer', 'e-notation range (misread by the scorer) -> 0.0000339–0.0000678 J (0.0339–0.0678 mJ)')
def _(qa):
    c = qa['sources'][code_src(qa)]['code']
    old = "print(f'Final energy loss range: {E_lo:.6e}–{E_hi:.6e} J')"
    assert c.count(old) == 1
    o = set_code(qa, c.replace(old, "print(f'Final energy loss range: {E_lo:.7f}–{E_hi:.7f} J "
                                    "({E_lo * 1e3:.4f}–{E_hi * 1e3:.4f} mJ)')"))
    has(o, 'Final energy loss range: 0.0000339–0.0000678 J (0.0339–0.0678 mJ)')
    qa['answer'], qa['answer_range'] = '0.0000339–0.0000678 J (0.0339–0.0678 mJ)', [3.39e-05, 6.78e-05]
    last_step(qa, 'so the energy lost is between 3.390093e-05 J and 6.780185e-05 J.',
              'so the energy lost is between 0.0000339 J and 0.0000678 J (0.0339–0.0678 mJ).')


@P('B|a00695/01740|1', 'answer', 'e-notation range (read reversed by the scorer) -> 0.000155–0.000310 J (0.155–0.310 mJ)')
def _(qa):
    c = qa['sources'][code_src(qa)]['code']
    old = "print(f'Work done against friction range: {W_low:.12e} to {W_high:.12e} J')"
    assert c.count(old) == 1
    o = set_code(qa, c.replace(old, "print(f'Work done against friction range: {W_low:.6f} to {W_high:.6f} J "
                                    "({W_low * 1e3:.3f}–{W_high * 1e3:.3f} mJ)')"))
    has(o, 'Work done against friction range: 0.000155 to 0.000310 J (0.155–0.310 mJ)')
    qa['answer'], qa['answer_range'] = '0.000155–0.000310 J (0.155–0.310 mJ)', [0.000155, 0.00031]
    last_step(qa, 'The computed range is 1.549617360109e-04 J to 3.099234720219e-04 J.',
              'The computed range is 0.000155 J to 0.000310 J (0.155–0.310 mJ).')


@P('test:B|a00541/00012|0', 'answer', "'-0.304–0.590' (read as 0.304–0.590 by the scorer) -> '-0.304 to 0.590', the "
                                      "scorer then uses answer_range [-0.304, 0.59]")
def _(qa):
    assert qa['answer_range'] == [-0.304, 0.59]
    qa['answer'] = '-0.304 to 0.590 m/s²'


# ---------------------------------------------------------------- single-value TYPE_B: accepted band ±5%
def band(qa, var, unit, ans_old=None):
    """append the ±5% band print to the code; the answer states it (after ans_old, else at the end)"""
    c = qa['sources'][code_src(qa)]['code'].rstrip()
    o = set_code(qa, c + "\n# accepted answer band for the single computed value: ±5%%\n"
                         "print(f'Accepted range (±5%%): {%s * 0.95:.3g}–{%s * 1.05:.3g} %s')" % (var, var, unit))
    lo, hi = re.search(r'Accepted range \(±5%\): ([0-9.]+)–([0-9.]+) ', o).groups()
    txt = 'accepted range: %s–%s %s, ±5%%' % (lo, hi, unit)
    a = qa['answer']
    if ans_old:
        assert a.count(ans_old) == 1, (ans_old, a)
        qa['answer'] = a.replace(ans_old, '%s (%s)' % (ans_old, txt))
    elif a.endswith(')'):
        qa['answer'] = a[:-1] + '; %s)' % txt
    else:
        qa['answer'] = '%s (%s)' % (a, txt)
    qa['answer_range'] = [float(lo), float(hi)]


NOTE = 'single value, no answer_range (ACC 0 for every response) -> accepted band ±5% printed by the code'
for _q, _v, _u, _o in (('B|a00452/02213|0', 'PE', 'J', None), ('B|a00861/00189|0', 'F', 'N', None),
                       ('B|a00879/00373|2', 'alpha', 'rad/s²', None), ('B|a00670/00003|0', 'KE', 'J', None),
                       ('A|1e3b818d-7d17-42b4-ab9c-0c04ea9a8a10|1', 'h_max', 'm', None),
                       ('A|90e697a5-2cff-4ecc-a580-f4c230a5488c|1', 'h_max', 'm', 'approximately 2.83 m')):
    P(_q, 'answer', NOTE)(lambda qa, v=_v, u=_u, o=_o: band(qa, v, u, o))


def main():
    cur = current(skip=('FIX8',))
    out, base = {}, {}
    for qid, (f, change, note) in PATCH.items():
        qa = copy.deepcopy(cur[qid][0])
        base[qid] = copy.deepcopy(qa)
        keep = {k: copy.deepcopy(qa.get(k)) for k in ('question', 'question_type', 'difficulty', 'hidden_cv_chosen')}
        f(qa)
        assert {k: qa.get(k) for k in keep} == keep, qid
        out[qid] = {'status': 'ok', 'qa': qa, 'change': change, 'note': note}
    with ThreadPoolExecutor(8) as ex:
        for qid, p in ex.map(one, [(q, r['qa']) for q, r in out.items()]):
            p = p + [x for x in strict_problems(out[qid]['qa']) if x not in strict_problems(base[qid])]
            out[qid]['problems'] = p
            if p:
                out[qid]['status'] = 'fail'
            print(qid, out[qid]['qa']['answer'][:150], out[qid]['qa']['answer_range'], out[qid]['status'], p or '')
    json.dump(out, open(WORK + '/fix8_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    rp = json.load(open(WORK + '/review_post.json', encoding='utf-8'))
    for q, x in out.items():
        rp.get(q, {}).pop('FIX8', None)
        if x['status'] == 'ok':
            rp.setdefault(q, {})['FIX8'] = {'verdict': 'accept', 'notes': x['note']}
    json.dump(rp, open(WORK + '/review_post.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(out), 'patched,', sum(r['status'] == 'ok' for r in out.values()), 'ok')


if __name__ == '__main__':
    main()
