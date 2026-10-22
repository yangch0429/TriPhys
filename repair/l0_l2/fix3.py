# Provenance (TriPhysBench release): fix_code/_v8_work/fix3.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""FIX3: deterministic fixes of the 3 problems of the third full final recheck (final_recheck_run3.log, all on L0
versions) and of the 1 problem of the fourth (final_recheck_run4.log: rounded quoted value made explicit): a verified quote added from the logged snippet of the source's own url, a value aligned with its quote
(22 oz = 0.624 kg) and a source swapped to a real search result stating the range used (0.7-0.8). Every patched QA is
checked again with final_recheck.one; accepted fixes go to review_post.json (FIX3).
-> fix3_results.json {qid: {status, qa, change, note, problems}}
Usage: python fix3.py"""
import copy, json
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, sources_of
from current import current
from final_recheck import one
from fix39 import logged, step_sub, swap, code_sub

PATCH = {}


def P(qid, change, note):
    def deco(f):
        PATCH[qid] = (f, change, note)
        return f
    return deco


@P('A|ec944f9c-2cd6-4f7d-9883-0bebb1feb0f7|2', 'none', 'verified quote = logged snippet of the source url')
def _(qa):
    s = sources_of(qa, 'web_search')[0]
    q = 'For a wheel (thin hoop approximation): I = m × r² (where m = mass, r = radius).'
    assert logged(s['search_query'], 'facebook.com/museumofscience', q) == s['url']
    s['verified_quote'] = q


@P('test:A|0ccfd492-8880-4ef2-90f5-e9f54f2cd080|0', 'full', 'basketball mass = 22 oz = 0.624 kg as quoted (0.625 kg); '
                                                            'answer recomputed 4.56 -> 4.55 J')
def _(qa):
    sources_of(qa, 'web_search')[0]['key_information'] = "standard men's basketball mass = 22 oz (0.624 kg)"
    code_sub(qa, 'm = 0.625\n', 'm = 0.624\n')
    step_sub(qa, 'Typical value is approximately 0.625 kg.', 'Standard weight 22 oz = 0.624 kg.')
    step_sub(qa, 'KE = 0.5 * 0.625 kg', 'KE = 0.5 * 0.624 kg')
    step_sub(qa, 'Result is 4.56 J', 'Result is 4.55 J')
    assert qa['answer'] == '4.56 J'
    qa['answer'] = '4.55 J'


@P('test:B|a00588/00436|0', 'full', 'dry asphalt friction source -> result stating 0.7-0.8 (value used unchanged); '
                                    'garbled characters removed from the answer')
def _(qa):
    swap(qa, 0, 'dry asphalt tire sliding friction coefficient range', 'onspot.com',
         'Dry Asphalt Road: Think of it as a high-five with a friction coefficient of about 0.7 to 0.8.',
         key='dry asphalt friction coefficient = 0.7-0.8')
    sources_of(qa, 'web_search')[0].pop('url_annotated', None)
    assert '§ç' in qa['answer']
    qa['answer'] = qa['answer'].replace('§ç', '')


@P('B|a00456/02183|0', 'scs', 'average adult height = 171.3 cm as quoted, rounded to 1.7 m (stated in the key / step; '
                             'answer 70.0° unchanged)')
def _(qa):
    s = sources_of(qa, 'web_search')[1]
    assert s['verified_quote'] == 'average adult height of people born in 1996 was 171.3 cm'
    s['key_information'] = 'average adult height = 171.3 cm ≈ 1.7 m (rounded)'
    step_sub(qa, "to retrieve typical person height: 1.7 m.",
             "to retrieve typical person height: average adult height 171.3 cm, rounded to 1.7 m.")


def main():
    cur = current(skip=('FIX3', 'CONS2'))   # FIX3 is applied before CONS2
    out = {}
    for qid, (f, change, note) in PATCH.items():
        qa = copy.deepcopy(cur[qid][0])
        f(qa)
        out[qid] = {'status': 'ok', 'qa': qa, 'change': change, 'note': note}
    with ThreadPoolExecutor(8) as ex:
        for qid, p in ex.map(one, [(q, r['qa']) for q, r in out.items()]):
            out[qid]['problems'] = p
            if p:
                out[qid]['status'] = 'fail'
            print(qid, out[qid]['change'], out[qid]['status'], p or '')
    json.dump(out, open(WORK + '/fix3_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    rp = json.load(open(WORK + '/review_post.json', encoding='utf-8'))
    for q, x in out.items():
        if x['status'] == 'ok':
            rp.setdefault(q, {})['FIX3'] = {'verdict': 'accept', 'notes': x['note']}
    json.dump(rp, open(WORK + '/review_post.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(out), 'patched,', sum(r['status'] == 'ok' for r in out.values()), 'ok')


if __name__ == '__main__':
    main()
