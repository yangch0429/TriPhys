# Provenance (TriPhysBench release): fix_code/_v8_work/v8_wsfix.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""
Fix ws positions where no query returns the annotated value (train: fixes; test: statistics only).
  valsearch   LLM writes 3 value-bearing queries (object + quantity + expected value + unit) per dead position
              -> Serper -> the same strict validation (stored in p['valsearch'], never used as training queries)
  requery     positions whose value was found online: LLM writes 5 value-free queries modeled on the hit's
              title/snippet wording (no site/brand/author names absent from the question) -> Serper -> validation
              (variants round 4)
  brandcheck  LLM flags valid round-3/4 variants that name a website/brand/product/author not in the question
              (v['brand'] = True; such variants are not used)
  summarize   per position: original_ok | replaced (best valid variant replaces the original query) | unresolved
              (value_found_no_query / value_not_found); writes data/v8_wsfix_report{suffix}.json
dead position = original invalid and no usable valid variant (test split: only the original was searched).
Usage: V8_SPLIT=train|test python v8_wsfix.py stage ...
"""
import json, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from v8_common import DATA, SUFFIX, SPLIT, call_llm
import v8_ws_pipeline as W

_lock = threading.Lock()


def usable(v):
    return v.get('valid') and not v.get('brand')


def is_dead(p):
    return not p['original'].get('valid') and not any(usable(v) for v in p['variants'])


def val_hit(p):
    """first valid value-bearing search: (item, hit result)"""
    for it in p.get('valsearch') or []:
        if it.get('valid'):
            return it, it['results'][it['hit_idx'] - 1]
    return None, None


# ---------------------------------------------------------------- valsearch
VALQ_PROMPT = """For each item, write 3 Google queries that look for the expected fact TOGETHER WITH its value, so that a
page stating it is found (e.g. "tennis ball mass 58 g", "tennis ball weighs 58 grams", "tennis ball 2 oz mass").
Use the same object/material and quantity; include the expected value (or the range) with a unit; vary wording and
unit form (SI / common units). Concise, 3-12 words.

{items}

Return ONLY JSON: {{"1": ["q1", "q2", "q3"], ...}}"""


def valsearch(st):
    dead = [p for p in st if is_dead(p) and not p.get('valsearch')]
    print('[valsearch] %d dead positions' % len(dead), flush=True)

    def one(batch):
        items = '\n'.join('%d. Expected: %s\n   Original query: "%s"\n   Question: %s' % (
            i + 1, W.ref_str(p), p['original']['query'], p['qkey'][:200]) for i, p in enumerate(batch))
        d = call_llm(VALQ_PROMPT.format(items=items), temperature=0.3) or {}
        with _lock:
            for i, p in enumerate(batch):
                qs = [q.strip() for q in d.get(str(i + 1)) or [] if isinstance(q, str) and q.strip()]
                p['valsearch'] = [{'query': q} for q in qs[:3]]

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, [dead[i:i + 8] for i in range(0, len(dead), 8)]))
    W.save(st)
    items = [it for p in dead for it in p['valsearch']]
    W.search_all(st, items, 'valsearch')
    validate_extra(st, 'valsearch')
    n = sum(1 for p in st if val_hit(p)[0])
    print('[valsearch] value found online for %d / %d dead positions' % (n, len(dead)), flush=True)


def validate_extra(st, field):
    """validate items stored outside original/variants via pseudo-positions; persists the real state"""
    pseudo = [{'ref': p['ref'], 'cls': p['cls'], 'original': {'query': '', 'results': None}, 'variants': p[field]}
              for p in st if p.get(field)]
    W.validate(pseudo, persist=lambda _: W.save(st))


# ---------------------------------------------------------------- requery
REQ_PROMPT = """Each item gives a web page that states a fact (title + snippet). Write 5 Google queries WITHOUT any number
whose wording closely follows the page's own title/snippet terms, so that this page (or a similar one) shows the fact
in its search snippet. Target the same quantity for the same object as the expected fact. Do NOT use website, brand,
product, company or author names unless they appear in the question. Concise, 4-12 words.

{items}

Return ONLY JSON: {{"1": ["q1", ...], ...}}"""


def requery(st):
    todo = [p for p in st if is_dead(p) and val_hit(p)[0] and not any(v['round'] == 4 for v in p['variants'])]
    print('[requery] %d positions with the value found online' % len(todo), flush=True)
    added = []

    def one(batch):
        lines = []
        for i, p in enumerate(batch):
            it, hit = val_hit(p)
            lines.append('%d. Expected: %s\n   Question: %s\n   Page title: %s\n   Snippet: %s\n   Quote: "%s"' % (
                i + 1, W.ref_str(p), p['qkey'][:200], hit.get('title', ''), (hit.get('snippet') or '')[:350], it['quote']))
        d = call_llm(REQ_PROMPT.format(items='\n'.join(lines)), temperature=0.5) or {}
        with _lock:
            for i, p in enumerate(batch):
                existing = [p['original']['query']] + [v['query'] for v in p['variants']]
                for q in d.get(str(i + 1)) or []:
                    if isinstance(q, str) and q.strip() and not W.leaks(q, p) and all(W.jacc(q, e) < 0.7 for e in existing):
                        v = {'query': q.strip().strip('"'), 'round': 4}
                        p['variants'].append(v)
                        existing.append(v['query'])
                        added.append(v)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, [todo[i:i + 6] for i in range(0, len(todo), 6)]))
    W.save(st)
    W.search_all(st, added, 'requery')
    W.validate(st)
    print('[requery] %d / %d positions now have a valid value-free query' % (
        sum(1 for p in todo if any(v.get('valid') and v['round'] == 4 for v in p['variants'])), len(todo)), flush=True)


# ---------------------------------------------------------------- brandcheck
BRAND_PROMPT = """For each search query, answer whether it contains a website, brand, product, company, shop, or author
name that does NOT appear in the question (generic words like "wikipedia", "engineering toolbox" also count as site
names). Names of the physical object itself that appear in the question are fine.

{items}

Return ONLY JSON: {{"1": true/false, ...}}  (true = contains such a name)"""


def brandcheck(st):
    todo = [(p, v) for p in st for v in p['variants'] if v.get('valid') and v['round'] >= 3 and 'brand' not in v]
    print('[brandcheck] %d queries' % len(todo), flush=True)

    def one(batch):
        items = '\n'.join('%d. Question: %s\n   Query: "%s"' % (i + 1, p['qkey'][:250], v['query']) for i, (p, v) in enumerate(batch))
        d = call_llm(BRAND_PROMPT.format(items=items)) or {}
        with _lock:
            for i, (p, v) in enumerate(batch):
                if isinstance(d.get(str(i + 1)), bool):
                    v['brand'] = d[str(i + 1)]

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, [todo[i:i + 15] for i in range(0, len(todo), 15)]))
    W.save(st)
    print('[brandcheck] flagged:', [v['query'] for p, v in todo if v.get('brand')], flush=True)


# ---------------------------------------------------------------- summarize
def summarize(st):
    out = []
    for p in st:
        rec = {'qid': p['qid'], 'cls': p['cls'], 'pos': p['pos'], 'n_ws': p['n_ws'], 'query_annotated': p['query_annotated'],
               'expected': W.ref_str(p)}
        if p['original'].get('valid'):
            best = p['original']
            rec['status'] = 'original_ok'
        else:
            cands = [v for v in p['variants'] if usable(v)]
            best = min(cands, key=lambda v: (v['hit_idx'] > 5, -W.jacc(v['query'], p['original']['query']), v['hit_idx'])) if cands else None
            rec['status'] = 'replaced' if best else 'unresolved'
        if best:
            hit = best['results'][best['hit_idx'] - 1]
            rec.update(query=best['query'], hit_idx=best['hit_idx'], quote=best['quote'], url=hit.get('link'))
        else:
            it, hit = val_hit(p)
            rec['reason'] = 'value_found_no_query' if it else 'value_not_found'
            rec['original_why'] = p['original'].get('why')
            if it:
                rec.update(value_query=it['query'], value_quote=it['quote'], value_url=hit.get('link'))
        out.append(rec)
    json.dump(out, open(DATA + '/v8_wsfix_report%s.json' % SUFFIX, 'w'), ensure_ascii=False, indent=1)
    print('[summarize] split=%s' % SPLIT)
    for cls in 'AB':
        rs = [r for r in out if r['cls'] == cls]
        qids = {r['qid'] for r in rs}
        bad_q = {r['qid'] for r in rs if r['status'] == 'unresolved'}
        print('  %s positions=%d %s | reasons %s | QAs with ws=%d, QAs with unresolved ws=%d' % (
            cls, len(rs), dict(Counter(r['status'] for r in rs)),
            dict(Counter(r['reason'] for r in rs if r['status'] == 'unresolved')), len(qids), len(bad_q)), flush=True)


if __name__ == '__main__':
    st = W.load_state()
    for s in sys.argv[1:]:
        {'valsearch': valsearch, 'requery': requery, 'brandcheck': brandcheck, 'summarize': summarize}[s](st)
