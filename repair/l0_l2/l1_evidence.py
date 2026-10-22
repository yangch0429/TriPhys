# Provenance (TriPhysBench release): fix_code/_v8_work/l1_evidence.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L1 step 1: real evidence for every web_search fact that could not be verified.
  ws_unresolved  (no value-free query returns the annotated value): the LLM reads ALL real results already collected
                 for the position's value-free queries (original + variants; value-bearing searches excluded) and
                 extracts the value those results actually state for the quantity (not necessarily the annotated one).
  ws_unlinked    (web_search source without a search step): same, with fresh queries only.
  Not found -> the LLM writes 4 new value-free queries (up to 2 rounds) -> Serper -> extract again.
Accepted evidence: quote is verbatim in the result snippet and contains the value; url = that result's real link.
Output: evidence.json  {qid: {"ws_unresolved": [...], "ws_unlinked": [...]}}   (resumable)
Usage: python l1_evidence.py"""
import json, os, re, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fixlib import REP, call_llm, call_serper, squash, text_nums, load, iter_qas

OUT = 'evidence.json'
_lock = threading.Lock()
problems = json.load(open('problems.json', encoding='utf-8'))
state = {}
for f in ('v8_ws_state.json', 'v8_ws_state_test.json'):
    for p in json.load(open(REP + f, encoding='utf-8')):
        state[(p['qid'], p['pos'])] = p
qas = {}
for split in ('train', 'test'):
    for cls in 'ab':
        for qid, vid, qa in iter_qas(load(cls, split, 'L0'), cls, split):
            qas[qid] = qa
ev = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}

EXTRACT = """You extract a physical value from REAL web search results for a physics benchmark.
Target fact: {target}
Question context: {question}
{video}The benchmark's current value ({current}) could NOT be confirmed by search; do not favour it — report what the
results actually state.
{nosearch}
Results (numbered query.result):
{results}

Pick the ONE result that most directly and explicitly states this quantity for the same kind of object / material /
context as in the question (prefer regulation, standard or typical specifications; if a result states a range for a
typical value, keep the range). The value must be stated explicitly in the snippet (unit conversion allowed, no other
computation).{single}
Return ONLY JSON: {{"found": true/false, "ref": "<query.result, e.g. 2.7>", "value": "<value with unit exactly as stated>",
"value_converted": "<same value in the unit the benchmark uses ({unit}), e.g. 0.0226 kg; a range as lo-hi>",
"is_range": true/false, "quote": "<minimal verbatim span copied from that snippet, containing the value, <=200 chars>",
"why": "<short reason>"{nosearch_key}}}"""

NOSEARCH = ('If the fact is a universal unit conversion or mathematical / physical constant that needs no retrieval '
            '(e.g. "1 ounce = 0.02835 kg", "g = 9.81 m/s²"), set "no_search_needed": true.')

SINGLE = ('\nThe question needs ONE number: choose a result that states a single value (e.g. an official / regulation '
          'value or a stated typical/average value), not only a range. If no result states a single value, found=false.')

NEWQ = """Write {n} NEW Google queries to find this fact: {target}
Question context: {question}
Queries already tried (their results did not state it): {tried}
Rules: no numbers at all; no website, brand, product or author names unless they appear in the question; concise
3-12 words; try clearly different angles (spec / dimensions / "how much does ... weigh" / regulation / standard /
typical range / material property table wording).
Return ONLY JSON: {{"queries": ["...", ...]}}"""


def fmt_results(groups):
    lines = []
    for qi, g in enumerate(groups, 1):
        lines.append('Query %d: "%s"' % (qi, g['query']))
        for ri, r in enumerate(g['results'], 1):
            lines.append('  [%d.%d] %s | %s' % (qi, ri, (r.get('title') or '')[:120], (r.get('snippet') or '')[:350]))
    return '\n'.join(lines)


def extract(item, groups, question, unlinked, video='', exclude=()):
    groups = [{'query': g['query'], 'results': [r for r in g['results'] if r.get('link') not in exclude]} for g in groups]
    groups = [g for g in groups if g['results']]
    if not groups:
        return None
    vtxt = ('Video content (the object in the question is THIS one; the value must fit it — same kind, size and '
            'context):\n%s\n' % video) if video else ''
    if exclude:
        vtxt += 'Already rejected as not matching the video object: %s\n' % '; '.join(exclude)
    d = call_llm(EXTRACT.format(target=item['target'], question=question[:1200], current=item['current'], video=vtxt,
                                unit=item['unit'] or 'the unit in the current value', results=fmt_results(groups),
                                nosearch=NOSEARCH if unlinked else '', single=SINGLE if item.get('single') else '',
                                nosearch_key=', "no_search_needed": true/false' if unlinked else '')) or {}
    if unlinked and d.get('no_search_needed'):
        return {'no_search_needed': True, 'why': d.get('why')}
    if not d.get('found'):
        return {'found': False, 'why': d.get('why')}
    if item.get('single') and d.get('is_range'):
        return {'found': False, 'why': 'range, single value required: %s' % d.get('value')}
    m = re.match(r'\s*(\d+)\.(\d+)', str(d.get('ref') or ''))
    quote = (d.get('quote') or '').strip()
    if not m:
        return {'found': False, 'why': 'bad ref: %s' % d.get('ref')}
    qi, ri = int(m.group(1)), int(m.group(2))
    if not (1 <= qi <= len(groups) and 1 <= ri <= len(groups[qi - 1]['results'])):
        return {'found': False, 'why': 'ref out of range: %s' % d.get('ref')}
    g, r = groups[qi - 1], groups[qi - 1]['results'][ri - 1]
    if not quote or squash(quote) not in squash(r.get('snippet')) or not re.search(r'\d', quote):
        return {'found': False, 'why': 'quote not verbatim / no number: %s' % quote[:100]}
    vn = [x for x in text_nums(d.get('value')) if x > 0]
    if vn and not all(any(abs(x - y) <= 1e-9 + 1e-6 * y for y in text_nums(quote)) for x in vn):
        return {'found': False, 'why': 'value %s not in quote' % d.get('value')}
    return {'found': True, 'query': g['query'], 'rank': ri, 'url': r.get('link'), 'title': r.get('title'),
            'snippet': r.get('snippet'), 'quote': quote, 'value': d.get('value'),
            'value_converted': d.get('value_converted'), 'is_range': bool(d.get('is_range')), 'why': d.get('why')}


def existing_groups(p):
    gs = []
    for it in [p['original']] + [v for v in p['variants'] if not v.get('brand')]:
        if it.get('results'):
            gs.append({'query': it['query'], 'results': it['results']})
    return gs


def solve(qid, kind, item, question, video='', exclude=(), extra_groups=()):
    unlinked = kind == 'ws_unlinked'
    groups = (existing_groups(state[(qid, item['pos'])]) if not unlinked else []) + list(extra_groups)
    tried = [g['query'] for g in groups]
    item['searched'] = list(item.get('searched') or [])
    res = None
    # existing results in chunks of 8 queries (prompt size)
    for k in range(0, len(groups), 8):
        res = extract(item, groups[k:k + 8], question, unlinked, video, exclude)
        if res and (res.get('found') or res.get('no_search_needed')):
            return res
    for rnd in range(2):
        d = call_llm(NEWQ.format(n=4, target=item['target'], question=question[:800], tried=json.dumps(tried[-20:])),
                     temperature=0.5) or {}
        qs = [q.strip().strip('"') for q in d.get('queries') or [] if isinstance(q, str) and q.strip()
              and not re.search(r'\d', q)][:4]
        new = []
        for q in qs:
            r = call_serper(q, tag='L1evidence|%s|%s' % (qid, kind))
            if r:
                new.append({'query': q, 'results': r})
            item['searched'].append(q)
            tried.append(q)
        res = extract(item, new, question, unlinked, video, exclude)
        if res and (res.get('found') or res.get('no_search_needed')):
            return res
    return res or {'found': False, 'why': 'no results'}


def unit_of(txt):
    m = re.search(r'\d\s*([A-Za-z°µμ/²³·%]+(?:/[A-Za-z²³]+)?)', txt or '')
    return m.group(1) if m else ''


tasks = []
for p in problems:
    qid, qa = p['qid'], qas[p['qid']]
    for u in p['ws_unresolved']:
        r = u['ref'] or {}
        tasks.append((qid, 'ws_unresolved', {'pos': u['pos'], 'target': '%s (as used in the solution step: "%s")' % (
            r.get('quantity'), state[(qid, u['pos'])]['step'][:300]), 'current': r.get('value'), 'unit': r.get('unit'),
                                              'single': qa.get('question_type') == 'TYPE_A'}))
    for u in p['ws_unlinked']:
        tasks.append((qid, 'ws_unlinked', {'source_idx': u['source_idx'], 'target': u['key_information'],
                                            'current': u['key_information'], 'unit': unit_of(u['key_information']),
                                            'single': qa.get('question_type') == 'TYPE_A'}))


def key(t):
    return '%s#%s' % (t[1], t[2].get('pos', t[2].get('source_idx')))


def run(t):
    qid, kind, item = t
    if key(t) in ev.get(qid, {}):
        return
    item = dict(item)
    item['evidence'] = solve(qid, kind, item, qas[qid]['question'])
    with _lock:
        ev.setdefault(qid, {})[key(t)] = {'kind': kind, **item}
        json.dump(ev, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('%s %s -> %s' % (qid, key(t), 'FOUND ' + str(item['evidence'].get('value')) if item['evidence'].get('found')
                               else 'no-search' if item['evidence'].get('no_search_needed') else 'none'), flush=True)


if __name__ == '__main__':
    print('tasks', len(tasks), flush=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(run, tasks))
    c = Counter()
    for qid, d in ev.items():
        for k, it in d.items():
            e = it['evidence']
            c[(it['kind'], 'found' if e.get('found') else 'no_search' if e.get('no_search_needed') else 'none')] += 1
    print(dict(c))
