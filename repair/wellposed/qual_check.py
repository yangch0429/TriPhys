# Provenance (TriPhysBench release): fix_code/evaluation/wellposed/qual_check.py
# Role: well-posedness / quality check of the 445 QAs of version 07 (10-11); base module of fix9 / fix10.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""10/11 user: per web constant flagged web_unreachable (question-only queries miss the reference value), deepseek-flash
labels EVERY term of the annotator's base query against the QUESTION: stated / implied / generic / added.
  from_question = no "added" term -> the question determines the value; question-only queries just searched worse
                  (search skill, not a question defect)
  added         = the base query narrows the value with something the question does not give -> question underdetermined
  base_misses   = even the base query no longer returns the reference value (basereach.py) -> unreachable for real
Also (user Q3): for web_ambiguous constants, is the accepted alternative value already in the BASE query's own results?
Output qual_check.json + summary.  Usage (fc/evaluation/wellposed): python qual_check.py"""
import sys, json, asyncio
from collections import Counter, defaultdict
sys.argv = ['x']
import wellposed_check as W

QUAL = """A physics benchmark question requires looking up one quantity on the web; the solver sees only the QUESTION text.
QUESTION: %s
Looked-up information (numbers hidden): %s
The annotator's Google query: %s

Split the annotator's query into its terms (single words or short phrases) and label EACH term:
- "stated": the question explicitly states this object / material / condition / population / standard (same meaning,
  possibly other words)
- "implied": not literally stated, but the question clearly implies it, so a careful solver would add it too
- "generic": search wording that does not narrow which value is meant (typical, average, mean, value, range, data, study,
  specifications, units such as kg or m/s, synonyms of the quantity's own name such as weight / mass)
- "added": narrows the value to a specific object, model, brand, size, material, population, condition, standard or
  source that the question does NOT state or clearly imply
verdict = "from_question" if no term is "added", otherwise "added".
Return ONLY JSON: {"terms": [{"term": "...", "label": "stated|implied|generic|added", "evidence": "<question words, or why>"}],
"verdict": "from_question|added"}"""

RANK = ['from_question', 'none', 'added', 'base_misses']


def grp(i):
    c, s = i.split('|')[:2]
    return '%s|%s' % (s, c)


async def one(r, c, kind):
    j = W.pj(await W.llm(QUAL % (r['question'], W.mask(c.get('ref')), c.get('ref_query') or ''))) or {}
    terms = [t for t in (j.get('terms') or []) if isinstance(t, dict)]
    v = j.get('verdict')
    if v not in ('from_question', 'added'):
        v = 'added' if any(t.get('label') == 'added' for t in terms) else ('from_question' if terms else None)
    return {'id': r['id'], 'split': r['split'], 'cls': r['cls'], 'base': kind, 'ref_query': c.get('ref_query'),
            'ref': c.get('ref'), 'qonly_queries': c.get('queries'), 'terms': terms, 'verdict': v}


async def main():
    W.SEM = asyncio.Semaphore(16)
    rs = json.load(open('wellposed.json'))
    kind = {}
    for e in json.load(open('basereach.json')):
        kind.setdefault(e[0], []).append(e[1])
    jobs = []
    for r in rs:
        if 'web_unreachable' not in r['flags']:
            continue
        ks = iter(kind[r['id']])
        for c in r['consts']:
            if not c['reachable']:
                jobs.append(one(r, c, next(ks)))
    out = await asyncio.gather(*jobs)
    qa = {}
    for o in out:
        k = 'base_misses' if o['base'] == 'base_misses' else (o['verdict'] or 'none')
        qa[o['id']] = max(qa.get(o['id'], 'from_question'), k, key=RANK.index)
    print('constants (base reach, verdict):', dict(Counter((o['base'], o['verdict']) for o in out)))
    for k, n in sorted(Counter((grp(i), v) for i, v in qa.items()).items()):
        print('  QA', k, n)
    print('term labels:', dict(Counter(t.get('label') for o in out for t in o['terms'])))
    # user Q3: web_ambiguous - are the accepted alternatives already in the base query's own results?
    amb = []
    for r in rs:
        for c in r.get('consts', []):
            if not c.get('ambiguous'):
                continue
            base_txt = W.search(c['ref_query']) if c.get('ref_query') else None
            alts = [x for x in c['candidates'] if x.get('acc') is not None and x['acc'] < .5
                    and x.get('valid_verdict') in ('both', 'alternative')]
            in_base = [bool(base_txt) and W.quote_ok(x['quote'], base_txt, x['value']) for x in alts]
            keyn = W.AT.nums(c.get('ref'))
            amb.append({'id': r['id'], 'split': r['split'], 'cls': r['cls'], 'ref_query': c.get('ref_query'),
                        'ref': c.get('ref'), 'ref_in_base': bool(base_txt) and bool(keyn) and W.AT.has(keyn, base_txt),
                        'n_alt': len(alts), 'n_alt_in_base': sum(in_base),
                        'alts': [dict(value=x['value'], unit=x.get('unit'), quote=x['quote'], in_base=b,
                                      from_question=x.get('from_question'), alt_answer=x.get('alt_answer'))
                                 for x, b in zip(alts, in_base)]})
    print('web_ambiguous constants by where the accepted alternative was found:')
    for k, n in sorted(Counter((a['split'], a['cls'], 'base_itself_ambiguous' if a['n_alt_in_base'] else
                                'only_other_queries') for a in amb).items()):
        print('  ', k, n)
    qa_amb = defaultdict(set)
    for a in amb:
        qa_amb[a['id']].add('base' if a['n_alt_in_base'] else 'other')
    for k, n in sorted(Counter((grp(i), 'base_itself_ambiguous' if 'base' in s else 'only_other_queries')
                               for i, s in qa_amb.items()).items()):
        print('  QA', k, n)
    json.dump({'unreachable': out, 'unreachable_qa': qa, 'ambiguous': amb}, open('qual_check.json', 'w'),
              ensure_ascii=False, indent=1)
    print(dict(W.STAT))


if __name__ == '__main__':
    asyncio.run(main())
