# Provenance (TriPhysBench release): fix_code/_v8_work/v8_ws_pipeline.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""
v8 Steps 1-5: web_search originals + regenerated variants with strict LLM value validation.
Stages (state cached in data/v8_ws_state.json, each stage resumable):
  orig_search  Serper top-10 for every position's original query
  gen1         LLM: 5 variant queries per position
  search1      Serper for round-1 variants
  validate     LLM strict check for every searched query not yet validated (original + variants)
  gen2         LLM: 5 new variants for positions with < 2 valid variants (avoid failed phrasings)
  search2      Serper for round-2 variants
  validate     (again)
Usage: python3 v8_ws_pipeline.py [stage ...]   (default: all stages in order)
"""
import json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from v8_common import DATA, SUFFIX, call_llm, call_serper, norm

STATE = DATA + '/v8_ws_state%s.json' % SUFFIX
N_VAR = 5
MIN_VALID = 2
_lock = threading.Lock()


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    pos = json.load(open(DATA + '/v8_ws_positions%s.json' % SUFFIX))
    st = []
    for p in pos:
        st.append({k: p[k] for k in ('qid', 'cls', 'qkey', 'pos', 'n_ws', 'step', 'ref')} | {
            'query_annotated': p['query'],
            'original': {'query': p['query_final'], 'round': 0},
            'variants': [],
        })
    return st


def save(st):
    tmp = STATE + '.tmp'
    json.dump(st, open(tmp, 'w'), ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


def toks(q):
    return set(re.findall(r'[a-z0-9]+', q.lower())) - {'the', 'a', 'an', 'of', 'for', 'in', 'on', 'and', 'to', 'what', 'is'}


def jacc(a, b):
    ta, tb = toks(a), toks(b)
    return len(ta & tb) / max(1, len(ta | tb))


def ref_numbers(p):
    out = set()
    for x in (p.get('ref') or {}).get('numbers') or []:
        try:
            out.add(float(x))
        except (TypeError, ValueError):
            pass
    return {x for x in out if x not in (0, 1, 2, 3)}


def leaks(q, p):
    qn = set(float(x) for x in re.findall(r'(?<![\w.])\d+(?:\.\d+)?', q))
    return bool(qn & ref_numbers(p))


def ref_str(p):
    r = p.get('ref') or {}
    s = '%s = %s' % (r.get('quantity', ''), r.get('value', ''))
    if r.get('used') and norm(r.get('used')) != norm(r.get('value')):
        s += ' (solution uses %s)' % r['used']
    return s


# ---------------- Serper ----------------
def search_all(st, items, label, persist=None):
    """items: list of dicts with 'query' and no 'results'."""
    todo = [it for it in items if it.get('results') is None]
    print('[%s] %d queries to search' % (label, len(todo)), flush=True)
    done = [0]

    def one(it):
        res = call_serper(it['query'])
        with _lock:
            it['results'] = res if res is not None else []
            it['search_failed'] = res is None
            done[0] += 1
            if done[0] % 200 == 0:
                print('  searched %d/%d' % (done[0], len(todo)), flush=True)
                (persist or save)(st)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))
    (persist or save)(st)
    print('[%s] done, empty results: %d' % (label, sum(1 for it in todo if not it['results'])), flush=True)


# ---------------- variant generation ----------------
GEN_PROMPT = """Generate web-search query variants. For each numbered item, produce {n} NEW queries that a person
would type into Google to find the same physical fact (the expected value) as the original query.
Requirements:
- Must target exactly the same quantity for the same object/material/context as the original query.
- Must NOT contain any number from the expected value, and no numeric answer at all (units words are fine).
- Each variant must differ clearly from the original and from each other: use different wording,
  synonyms/terminology, word order, or add/omit helpful context words (e.g. "typical", "standard", "average").
- Concise keyword style, 3-12 words, English.
{extra}
{items}

Return ONLY JSON: {{"1": ["q1", ...], "2": [...]}}"""


def gen_variants(st, targets, rnd):
    extra = ''
    if rnd > 1:
        extra = '- Earlier variants listed under "Failed" did not return the value; avoid those phrasings and try ' \
                'other angles (e.g. the source type that usually states it: spec sheet, engineering table, standard).'
    batches = [targets[i:i + 8] for i in range(0, len(targets), 8)]
    print('[gen%d] %d positions, %d LLM calls' % (rnd, len(targets), len(batches)), flush=True)

    def one(batch):
        lines = []
        for i, p in enumerate(batch):
            lines.append('%d. Original query: "%s"\n   Expected value: %s\n   Question context: %s' % (
                i + 1, p['original']['query'], ref_str(p), p['qkey'][:200]))
            if rnd > 1:
                failed = [v['query'] for v in p['variants'] if not v.get('valid')]
                lines.append('   Failed: ' + json.dumps(failed, ensure_ascii=False))
        d = call_llm(GEN_PROMPT.format(n=N_VAR, extra=extra, items='\n'.join(lines)), temperature=0.7) or {}
        with _lock:
            for i, p in enumerate(batch):
                existing = [p['original']['query']] + [v['query'] for v in p['variants']]
                added = 0
                for q in d.get(str(i + 1)) or []:
                    if not isinstance(q, str) or not q.strip():
                        continue
                    q = q.strip().strip('"')
                    if leaks(q, p) or any(jacc(q, e) >= 0.7 for e in existing):
                        continue
                    p['variants'].append({'query': q, 'round': rnd})
                    existing.append(q)
                    added += 1
                    if added >= N_VAR:
                        break

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, batches))
    save(st)
    print('[gen%d] variants now: %d' % (rnd, sum(len(p['variants']) for p in st)), flush=True)


# ---------------- strict validation ----------------
VAL_PROMPT = """You are a strict data validator. For each item, decide whether the search results EXPLICITLY state
the expected physical fact, so that a reader could extract the expected value from the results alone.
Accept only if a result states the SAME quantity for the SAME kind of object/material, and its value
- equals the expected value, or
- equals it after unit conversion (approximate conversion within 2% is fine, e.g. 22 oz ~ 0.624 kg vs 0.625 kg), or
- is a range whose endpoint or midpoint equals the expected single value, or
- (expected value is a range) states the same range, or a range whose endpoints match within 2%.
Reject when the number belongs to a different quantity/object, only appears coincidentally, the result
merely mentions the topic, or the value must be computed from other facts.

{items}

For each item return {{"ok": true/false, "idx": <result number that contains the value or 0>,
"quote": "<minimal verbatim span copied exactly from that result, containing the value, <=200 chars>",
"why": "<short reason>"}}
Return ONLY JSON: {{"1": {{...}}, "2": {{...}}}}"""


def squash(s):
    return re.sub(r'\s+', ' ', s or '').strip().lower()


def validate(st, persist=None):
    todo = []
    for p in st:
        for it in [p['original']] + p['variants']:
            if it.get('results') is not None and 'valid' not in it:
                if not it['results']:
                    it['valid'] = False
                    it['why'] = 'no results'
                else:
                    todo.append((p, it))
    batches = [todo[i:i + 8] for i in range(0, len(todo), 8)]
    print('[validate] %d items, %d LLM calls' % (len(todo), len(batches)), flush=True)
    done = [0]

    def one(batch):
        lines = []
        for i, (p, it) in enumerate(batch):
            rs = '\n'.join('     [%d] %s' % (k + 1, (r.get('snippet') or '')[:350]) for k, r in enumerate(it['results']))
            lines.append('%d. Expected: %s\n   Query: "%s"\n   Results:\n%s' % (i + 1, ref_str(p), it['query'], rs))
        d = call_llm(VAL_PROMPT.format(items='\n\n'.join(lines))) or {}
        with _lock:
            for i, (p, it) in enumerate(batch):
                v = d.get(str(i + 1))
                if not isinstance(v, dict):
                    continue  # stays unvalidated, retried next run
                ok = bool(v.get('ok'))
                idx = int(v.get('idx') or 0) if str(v.get('idx') or 0).isdigit() else 0
                quote = (v.get('quote') or '').strip()
                if ok:
                    if not (1 <= idx <= len(it['results'])):
                        idx = next((k + 1 for k, r in enumerate(it['results'])
                                    if quote and squash(quote) in squash(r.get('snippet'))), 0)
                    if not idx or not quote or squash(quote) not in squash(it['results'][idx - 1].get('snippet')):
                        ok = False
                        v['why'] = 'quote not verbatim: ' + (v.get('why') or '')
                if ok and not re.search(r'\d', quote):
                    ok = False
                    v['why'] = 'quote has no number: ' + (v.get('why') or '')
                it['valid'] = ok
                it['hit_idx'] = idx if ok else 0
                it['quote'] = quote if ok else ''
                it['why'] = v.get('why', '')
            done[0] += 1
            if done[0] % 25 == 0:
                print('  validated batches %d/%d' % (done[0], len(batches)), flush=True)
                (persist or save)(st)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, batches))
    (persist or save)(st)
    report(st)


def n_valid(p):
    return sum(1 for v in p['variants'] if v.get('valid'))


def report(st):
    for cls in 'AB':
        ps = [p for p in st if p['cls'] == cls]
        o_ok = sum(1 for p in ps if p['original'].get('valid'))
        o_top5 = sum(1 for p in ps if p['original'].get('valid') and p['original'].get('hit_idx', 0) <= 5)
        vs = [v for p in ps for v in p['variants']]
        v_ok = sum(1 for v in vs if v.get('valid'))
        dist = {}
        for p in ps:
            k = min(n_valid(p), 5)
            dist[k] = dist.get(k, 0) + 1
        print('  %s positions=%d | original valid=%d (top5=%d) | variants=%d valid=%d | valid-per-pos=%s' % (
            cls, len(ps), o_ok, o_top5, len(vs), v_ok, dict(sorted(dist.items()))), flush=True)


def main(stages):
    st = load_state()
    for s in stages:
        if s == 'orig_search':
            search_all(st, [p['original'] for p in st], s)
        elif s == 'gen1':
            gen_variants(st, [p for p in st if not any(v['round'] == 1 for v in p['variants'])], 1)
        elif s == 'search1':
            search_all(st, [v for p in st for v in p['variants'] if v['round'] == 1], s)
        elif s == 'validate':
            validate(st)
        elif s == 'gen2':
            gen_variants(st, [p for p in st if n_valid(p) < MIN_VALID and not any(v['round'] == 2 for v in p['variants'])], 2)
        elif s == 'search2':
            search_all(st, [v for p in st for v in p['variants'] if v['round'] == 2], s)
        elif s == 'report':
            report(st)
        else:
            raise ValueError(s)


if __name__ == '__main__':
    main(sys.argv[1:] or ['orig_search', 'gen1', 'search1', 'validate', 'gen2', 'search2', 'validate'])
