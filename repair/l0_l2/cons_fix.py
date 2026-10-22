# Provenance (TriPhysBench release): fix_code/_v8_work/cons_fix.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Consistency repair: QAs of the same video that retrieve the same property of the same object must use the same value.
Only REAL value differences are repaired: the values of a violation (consistency_current.json) are parsed to intervals
in one unit (single value v = [v, v]; range a-b = [a, b]; a ± b = [a-b, a+b]; the value a QA states it uses, when it
names one); a violation is real when two intervals widened by 2 % do not overlap. Precision differences (<= 2 %) and
overlapping ranges are left unchanged.
  plan: for each real violation the canonical value is one web_search source of one of the QAs, whose quote is verified
        in the search log (the snippet of the cited url in the logged results of the cited query) or by L0 (verified_quote),
        chosen by the reviewer LLM (fits the video object; a single value when a QA needs one; otherwise the value
        most QAs already use). QAs not repaired (problem QAs without an accepted repair) are neither canonical nor
        changed. Targets = the other QAs whose interval does not overlap the canonical one (2 %).  -> cons_plan.json
  fix:  every target is rewritten by the L1 rewriter (l1_rewrite: same prompt, validation, independent review; 5
        attempts) from its current version, with the canonical source as evidence (same query / url / quote); all
        properties of one QA in one call.  -> cons_results.json  (reviewed one by one -> review_cons.json)
Round 2 (plan2 / fix2, after the post repairs L1T / FIX39 / FIX39W / FIX3): same procedure on the consistency check
of the final current version; a QA counts as repaired when its current version has any repair level; results go to
cons2_plan.json / cons2_results.json, applied as the last post level CONS2 (current.py, reviewed in review_post.json).
Usage: python cons_fix.py plan | fix [qid ...] | plan2 | fix2 [qid ...]"""
import json, os, sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, call_llm, sources_of, text_nums, video_context
from current import current

TOL = 0.02
ROUND2 = len(sys.argv) > 1 and sys.argv[1] in ('plan2', 'fix2')
PLAN = WORK + ('/cons2_plan.json' if ROUND2 else '/cons_plan.json')
problems = {p['qid'] for p in json.load(open(WORK + '/problems.json', encoding='utf-8'))}
review = json.load(open(WORK + '/review_log.json', encoding='utf-8'))
if ROUND2:
    UNREPAIRED = {q for q, (qa, lv) in current().items() if q in problems and lv == ['L0']}
else:
    UNREPAIRED = {q for q in problems if (review.get(q) or {}).get('verdict') != 'accept'}

PARSE = """Convert the values below, all for the property "{prop}", to numeric intervals in ONE common unit.
- single value v -> [v, v]; range a-b -> [a, b]; "a ± b" -> [a-b, a+b]; "above a" -> [a, null]; "below b" -> [null, b].
- when a text also states the value the QA actually uses (e.g. "567-624 g (using 0.624 kg)", "range ..., typically X",
  "hidden value X"), use that value only.
- percentages of something else (e.g. "0.65% of body weight (0.589 kg)") -> use the absolute value if given.
Values:
{vals}
Return ONLY JSON: {{"unit": "<common unit>", "intervals": {{"<qa label>": [lo, hi], ...}}}}"""

CHOOSE = """QAs about the same video use different values for the same property "{prop}". A benchmark rule requires one
value. Choose the canonical value among the VERIFIED candidate sources below (each is a real search result quote).
Criteria, in order:
1. it must fit the object in the video (same kind of object, size, context; see VIDEO);
2. if a QA needs a single number ({single}), prefer a candidate stating a single value;
3. otherwise prefer the value that most QAs already use (fewest changes) and a more specific / authoritative source.
VIDEO: {video}
VALUES USED BY THE QAs:
{vals}
VERIFIED CANDIDATES:
{cands}
Return ONLY JSON: {{"candidate": <number>, "why": "<one sentence>"}}  or {{"candidate": null, "why": "..."}} if none fits."""


def ukey(u):
    return (u or '').split('?')[0].split('#')[0].rstrip('/').lower().replace('://www.', '://')


_log = None


def logged_snippet(query, url):
    global _log
    if _log is None:
        _log = defaultdict(list)
        for line in open(WORK + '/search_log.jsonl', encoding='utf-8'):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            _log[d['query']].append(d['raw'])
    for raw in _log.get(query, []):
        for o in raw.get('organic', []):
            if ukey(o.get('link')) == ukey(url):
                return o
    return None


def verified(s):
    """{query, url, title, snippet, quote} of a web_search source verified in the search log or by L0 (verified_quote),
    else None"""
    q = s.get('search_query')
    o = logged_snippet(q, s.get('url')) if q else None
    if o is None:
        vq = s.get('verified_quote')
        if not (q and vq and text_nums(vq)):
            return None
        o = {'link': s.get('url'), 'title': None, 'snippet': vq}
    quote = s.get('verified_quote') or o.get('snippet')
    if not quote or not text_nums(quote):
        return None
    return {'query': q, 'url': o.get('link'), 'title': o.get('title'), 'snippet': o.get('snippet'), 'quote': quote}


def widen(iv):
    lo, hi = iv
    lo = -1e300 if lo is None else lo * (1 - TOL) if lo >= 0 else lo * (1 + TOL)
    hi = 1e300 if hi is None else hi * (1 + TOL) if hi >= 0 else hi * (1 - TOL)
    return lo, hi


def overlap(a, b):
    (a0, a1), (b0, b1) = widen(a), widen(b)
    return a0 <= b1 and b0 <= a1


def ok_interval(iv):
    return (isinstance(iv, list) and len(iv) == 2 and all(x is None or isinstance(x, (int, float)) for x in iv)
            and any(x is not None for x in iv) and (None in iv or iv[0] <= iv[1]))


def plan_one(args):
    video, v, qas = args
    labels = [q for q in v['qa_values'] if q in qas]
    rec = {'video': video, 'property': v['object_property'], 'values': {q: v['qa_values'][q] for q in labels}}
    if len(labels) < 2:
        return dict(rec, verdict='skip', why='fewer than 2 known QAs')
    d = call_llm(PARSE.format(prop=v['object_property'], vals='\n'.join('- %s: %s' % (q, v['qa_values'][q])
                                                                          for q in labels)), temperature=0) or {}
    ivs = {q: x for q, x in (d.get('intervals') or {}).items() if q in labels and ok_interval(x)}
    if len(ivs) < len(labels):
        return dict(rec, verdict='manual', why='unparsed values: %s' % [q for q in labels if q not in ivs])
    rec.update(unit=d.get('unit'), intervals=ivs)
    if all(overlap(ivs[a], ivs[b]) for a in labels for b in labels):
        return dict(rec, verdict='keep', why='values overlap within 2 % (precision / range overlap)')
    # canonical: verified sources of repaired / unproblematic QAs
    cands = []
    for q in labels:
        if q in UNREPAIRED:
            continue
        for i, s in enumerate(sources_of(qas[q], 'web_search')):
            f = verified(s)
            if f:
                cands.append({'qa': q, 'source_idx': i, 'key_information': s.get('key_information'), **f})
    single = [q for q in labels if qas[q].get('question_type') == 'TYPE_A' or ivs[q][0] == ivs[q][1]]
    if not cands:
        return dict(rec, verdict='manual', why='no verified candidate source')
    c = call_llm(CHOOSE.format(
        prop=v['object_property'], single=', '.join(single) or 'none', video=(video_context(labels[0]) or '')[:2500],
        vals='\n'.join('- %s: %s  [%s]' % (q, v['qa_values'][q], qas[q].get('question_type')) for q in labels),
        cands='\n'.join('%d. QA %s: "%s" (query: %s; url: %s; quote: "%s")' % (n, x['qa'], x['key_information'],
                                                                             x['query'], x['url'], x['quote'][:300])
                        for n, x in enumerate(cands, 1))), temperature=0) or {}
    n = c.get('candidate')
    if not isinstance(n, int) or not 1 <= n <= len(cands):
        return dict(rec, verdict='manual', why='no canonical chosen: %s' % c.get('why'))
    canon = cands[n - 1]
    civ = ivs[canon['qa']]
    targets = [q for q in labels if q != canon['qa'] and not overlap(ivs[q], civ)]
    skipped = [q for q in targets if q in UNREPAIRED]
    return dict(rec, verdict='fix', canonical=canon, canonical_interval=civ, why=c.get('why'),
                targets=[q for q in targets if q not in UNREPAIRED], unrepaired_targets=skipped)


def merge(video, vs, qas):
    """labels -> qids (the checker writes "QA A|..." and may drop "test:"); pairwise violations of one property merged"""
    ids = {q.split(':')[-1]: q for q in qas if q.split(':')[-1].rsplit('|', 1)[0] == video}
    groups = {}
    for v in vs:
        g = groups.setdefault(v['object_property'].strip().lower(), {'object_property': v['object_property'],
                                                                     'qa_values': {}})
        for lab, val in v['qa_values'].items():
            q = ids.get(lab.replace('QA ', '').strip().split(':')[-1])
            if q:
                g['qa_values'].setdefault(q, val)
    return list(groups.values())


def run_plan():
    viol = json.load(open(WORK + '/consistency_current.json', encoding='utf-8'))
    qas = {q: qa for q, (qa, lv) in current(pending=True).items()}
    items = [(video, v, qas) for video, vs in viol.items() for v in merge(video, vs, qas)]
    with ThreadPoolExecutor(8) as ex:
        plan = list(ex.map(plan_one, items))
    json.dump(plan, open(PLAN, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(Counter(p['verdict'] for p in plan), '| target QAs:', len({q for p in plan for q in p.get('targets', [])}))


def run_fix(only):
    import l1_rewrite as R
    plan = [p for p in json.load(open(PLAN, encoding='utf-8')) if p['verdict'] == 'fix']
    cur = current(pending=False)
    items, notes = defaultdict(dict), defaultdict(list)
    for p in plan:
        c = p['canonical']
        for q in p['targets']:
            k = 'cons#%s' % p['property']
            items[q][k] = {'kind': 'ws_unresolved', 'target': p['property'], 'current': p['values'][q],
                           'unit': p.get('unit'), 'single': cur[q][0].get('question_type') == 'TYPE_A',
                           'evidence': {'found': True, 'query': c['query'], 'rank': None, 'url': c['url'],
                                        'title': c['title'], 'snippet': c['snippet'], 'quote': c['quote'],
                                        'value': c['key_information'], 'value_converted': None,
                                        'is_range': p['canonical_interval'][0] != p['canonical_interval'][1],
                                        'why': 'consistency with %s' % c['qa']}}
            notes[q].append('CONSISTENCY: QA %s of the same video uses %s for "%s" (source %s); this QA uses %s. '
                            'Replace this QA\'s value by the evidence value everywhere (its old web_search source for '
                            'this property is replaced by the evidence source); keep everything else.'
                            % (c['qa'], c['key_information'], p['property'], c['url'], p['values'][q]))
            if (p.get('extra') or {}).get(q):
                notes[q].append(p['extra'][q])
    todo = [q for q in items if not only or q in only]
    R.OUT = WORK + ('/cons2_results.json' if ROUND2 else '/cons_results.json')
    R.res = json.load(open(R.OUT, encoding='utf-8')) if os.path.exists(R.OUT) else {}
    R.evidence = dict(items)
    R.MAX_ATTEMPTS = 5
    R.fmt_code = lambda qid: 'none (but the code must reproduce the new answer)'
    old_verified = R.fmt_verified

    def fmt_verified(qa):  # the facts to be replaced are not "verified, keep unchanged"
        cur_nums = [x for it in items.get(qa.get('_qid'), {}).values() for x in text_nums(it['current']) if x > 0]
        keep = [s for s in sources_of(qa, 'web_search') if s.get('verified_quote') and not any(
            abs(x - y) <= 1e-3 * max(x, y) for x in cur_nums for y in text_nums(s.get('key_information')))]
        return '\n'.join('- query "%s" -> %s (%s)' % (s.get('search_query'), s.get('key_information'), s.get('url'))
                         for s in keep) or 'none'
    R.fmt_verified = fmt_verified
    R.REDO = {}
    for q in todo:
        qa = json.loads(json.dumps(cur[q][0]))
        qa['_qid'] = q
        R.qas[q] = qa
        R.REDO[q] = notes[q]
    _norm = R.normalize

    def normalize(old, d, cls):
        new = _norm(old, d, cls)
        new.pop('_qid', None)
        return new
    R.normalize = normalize
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(R.repair, todo))
    print(Counter(r['status'] for q, r in R.res.items() if q in todo))


if __name__ == '__main__':
    run_plan() if sys.argv[1] in ('plan', 'plan2') else run_fix(sys.argv[2:])
