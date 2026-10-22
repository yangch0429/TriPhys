# Provenance (TriPhysBench release): fix_code/_v8_work/final_recheck.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Recheck of the written benchmark (fix_code/*class_qa_{train,test}.json after final_writeback.py) with the problem
checks of l0_apply: every code block is executed again (check_codes: runs, prints, reproduces the answer), every
web_search source must carry a verified quote (verified_quote set by L0 / L1 / L1E, or for L2 the snippet of its url in
the logged results of its search_query) containing its value (numbers, else the LLM judges unit conversions / ranges)
and a search_query that an annotated_solution step retrieves; answer format by question_type
(TYPE_B answer_range [lo, hi], TYPE_C Yes/No).
Output: final_recheck.json {qid: [problems]}
Usage: python final_recheck.py"""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, load, iter_qas, get_steps, sources_of, check_codes, text_nums, consistent, call_llm
from cons_fix import logged_snippet

JUDGE = """Does the QUOTE (a real search result) state the VALUE, directly or by a plain unit conversion / rounding
(e.g. "220 lbs" states "99.79 kg"; a range in the quote states a value or sub-range inside it only if the VALUE is
presented as that range or a value from it)? A different number, a different object or a value merely plausible is NO.
VALUE: {v}
QUOTE: {q}
Return ONLY JSON: {{"states": true|false, "why": "<short>"}}"""


def quote_of(s):
    if s.get('verified_quote'):
        return s['verified_quote']
    o = logged_snippet(s.get('search_query'), s.get('url')) if s.get('search_query') else None
    return o and o.get('snippet')


def ws_problems(qa):
    steps = ' '.join(get_steps(qa)[0]).lower()
    out = []
    for j, s in enumerate(sources_of(qa, 'web_search')):
        vq, q = quote_of(s), s.get('search_query')
        nums = [x for x in text_nums(s.get('key_information')) if x > 0]
        if not vq:
            out.append('ws source %d: no verified quote' % j)
        elif nums and not consistent(vq, nums):
            d = call_llm(JUDGE.format(v=s.get('key_information'), q=vq), temperature=0) or {}
            if not d.get('states'):
                out.append('ws source %d: value not in its quote (%s | quote: %s | %s)' % (
                    j, s.get('key_information'), vq[:150], d.get('why')))
        if not q or q.lower() not in steps:
            out.append('ws source %d: no solution step retrieves it' % j)
    return out


def fmt_problems(qa):
    t, ar = qa.get('question_type'), qa.get('answer_range')
    # TYPE_B: a range (answer_range [lo, hi]) or a single value (answer_range null; the eval prompt allows both)
    if t == 'TYPE_B' and ar is not None and not (isinstance(ar, list) and len(ar) == 2):
        return ['TYPE_B answer_range malformed']
    if t == 'TYPE_C' and not str(qa.get('answer', '')).strip().lower().startswith(('yes', 'no')):
        return ['TYPE_C answer not Yes/No']
    return []


def one(item):
    qid, qa = item
    p = ws_problems(qa) + fmt_problems(qa)
    p += ['code block %d: %s' % (i, r['problem']) for i, r in enumerate(check_codes(qa)) if not r['ok']]
    return qid, p


def main():
    items = [(qid, qa) for split in ('train', 'test') for cls in 'ab' for qid, _, qa in iter_qas(load(cls, split), cls, split)]
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, items))
    json.dump({q: p for q, p in res.items() if p}, open(WORK + '/final_recheck.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('QAs', len(res), '| with problems', sum(1 for p in res.values() if p))
    print(Counter(x.split(':', 1)[-1].strip().split(' (')[0].split(' "')[0][:40] for p in res.values() for x in p))


if __name__ == '__main__':
    main()
