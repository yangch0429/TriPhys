# Provenance (TriPhysBench release): fix_code/_v8_work/l1e.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L1E: L1 repair of the QAs that neither L1 nor L2 repaired, with the real search results collected by L2 as additional
evidence (no new searches).
  evidence: for every fact of evidence.json without evidence (need_L2) or whose evidence did not fit the video
            (evidence_rejected, its url excluded), the L1 extractor (l1_evidence.extract: quote verbatim in the snippet,
            value in the quote) reads the pool of this QA = web_search results of its L2 transcripts
            (l2_messages/<qid>.json) + its L1evidence searches (search_log.jsonl). A result is not accepted from a
            query that contains the extracted value.  -> evidence_l1e.json
  rewrite:  l1_rewrite.repair with that evidence (same prompt, validation, independent review; 3 attempts, deepseek-flash); the
            video_observation sources, which must stay unchanged, are restored from the current QA before validation;
            QAs rejected in the one-by-one review start from the review notes, L1 failures from the problems of
            their last L1 attempt.  -> l1e_results.json
Usage: python l1e.py evidence|rewrite [qid ...]"""
import copy, json, os, re, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import l1_evidence as E
import l1_rewrite as R
from fixlib import WORK, text_nums

EV_OUT = WORK + '/evidence_l1e.json'
_lock = threading.Lock()
l1 = json.load(open(WORK + '/l1_results.json', encoding='utf-8'))
review = json.load(open(WORK + '/review_log.json', encoding='utf-8'))
TODO = [q for q in l1 if (review.get(q) or {}).get('verdict') != 'accept']


def pool(qid):
    """[{query, results: [{title, link, snippet}]}] of the L2 transcripts and L1evidence searches of this QA"""
    groups = {}
    p = '%s/l2_messages/%s.json' % (WORK, re.sub(r'[^\w.-]+', '_', qid))
    for msgs in (json.load(open(p, encoding='utf-8')) if os.path.exists(p) else []):
        calls = {}
        for m in msgs:
            if not isinstance(m, dict):
                continue
            for tc in m.get('tool_calls') or []:
                f = tc.get('function') or {}
                if f.get('name') == 'web_search':
                    try:
                        calls[tc.get('id')] = json.loads(f.get('arguments') or '{}').get('query')
                    except json.JSONDecodeError:
                        pass
            if m.get('role') == 'tool' and calls.get(m.get('tool_call_id')):
                try:
                    rs = json.loads(m.get('content') or '[]')
                except json.JSONDecodeError:
                    continue
                if isinstance(rs, list) and rs:
                    groups.setdefault(calls[m['tool_call_id']], [{'title': r.get('title'), 'link': r.get('url'),
                                                                   'snippet': r.get('snippet')} for r in rs
                                                                  if isinstance(r, dict)])
    for line in open(WORK + '/search_log.jsonl', encoding='utf-8'):
        if ('L1evidence|%s|' % qid) not in line:
            continue
        d = json.loads(line)
        if d['tag'].startswith('L1evidence|%s|' % qid):
            groups.setdefault(d['query'], [{'title': o.get('title'), 'link': o.get('link'), 'snippet': o.get('snippet')}
                                           for o in d['raw'].get('organic', [])[:10]])
    return [{'query': q, 'results': r} for q, r in groups.items()]


def rejected_items(qid):
    """{evidence number (1-based): reason} of an L1 evidence_rejected result"""
    r = l1[qid]
    if r['status'] != 'evidence_rejected':
        return {}
    return {int(k): v for k, v in (r['attempts'][-1].get('evidence_rejected') or {}).items() if str(k).isdigit()}


def find(qid, item, groups, exclude):
    """extract in chunks of 8 queries; a hit whose query contains the value is excluded and the chunk re-read"""
    q = R.qas[qid]['question']
    video = R.video_context(qid) or ''
    last = None
    for k in range(0, len(groups), 8):
        chunk = groups[k:k + 8]
        for _ in range(3):
            last = E.extract(item, chunk, q, item['kind'] == 'ws_unlinked', video, tuple(exclude))
            if not (last and last.get('found')):
                break
            vn = [x for x in text_nums(last.get('value')) if x > 0]
            if any(abs(x - y) <= 1e-6 * max(x, y) for x in vn for y in text_nums(last['query'])):
                chunk = [g for g in chunk if g['query'] != last['query']]
                last = {'found': False, 'why': 'value-bearing query dropped'}
                continue
            return last
    return last or {'found': False, 'why': 'empty pool'}


def evidence_one(qid, ev_all):
    items = copy.deepcopy(E.ev.get(qid, {}))
    rej = rejected_items(qid)
    groups = pool(qid)
    log = {}
    for n, (k, it) in enumerate(items.items(), 1):
        e = it['evidence']
        if e.get('no_search_needed') or (e.get('found') and n not in rej):
            continue
        exclude = [e['url']] if n in rej and e.get('url') else []
        new = find(qid, it, groups, exclude)
        new['source'] = 'L2 pool'
        it['evidence'] = new
        log[k] = 'FOUND %s' % new.get('value') if new.get('found') else 'none (%s)' % str(new.get('why'))[:80]
    with _lock:
        ev_all[qid] = items
        json.dump(ev_all, open(EV_OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(qid, 'pool=%d' % len(groups), log, flush=True)


def run_evidence(todo):
    ev_all = json.load(open(EV_OUT, encoding='utf-8')) if os.path.exists(EV_OUT) else {}
    todo = [q for q in todo if q not in ev_all]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda q: evidence_one(q, ev_all), todo))
    c = Counter()
    for q in TODO:
        for it in (ev_all.get(q) or {}).values():
            e = it['evidence']
            c['found' if e.get('found') else 'no_search' if e.get('no_search_needed') else 'none'] += 1
    print(dict(c), '| QAs with all facts evidenced:',
          sum(all(it['evidence'].get('found') or it['evidence'].get('no_search_needed') for it in ev_all.get(q, {}).values())
              for q in TODO if q in ev_all))


_normalize = R.normalize


def normalize(old, d, cls):
    """L1 normalize + the video_observation sources of the current QA (they must stay unchanged)"""
    new = _normalize(old, d, cls)
    ov = [s for s in old['sources'] if s.get('type') == 'video_observation']
    rest = [s for s in new.get('sources') or [] if s.get('type') != 'video_observation']
    new['sources'] = copy.deepcopy(ov) + rest
    return new


def run_rewrite(todo):
    R.evidence = json.load(open(EV_OUT, encoding='utf-8'))
    R.OUT = WORK + '/l1e_results.json'
    R.res = json.load(open(R.OUT, encoding='utf-8')) if os.path.exists(R.OUT) else {}
    R.MAX_ATTEMPTS = 3
    R.normalize = normalize
    R.REDO = {q: [review[q]['notes']] for q in todo if (review.get(q) or {}).get('verdict') == 'reject'
              and review[q]['level'] in ('L1', 'L1E') and 'l1_redo_checks' not in review[q]['notes']}
    R.REDO.update({q: v for q, v in json.load(open(WORK + '/redo_L1.json', encoding='utf-8')).items() if q in todo})
    for q in todo:  # L1 failures: start from the problems of the last L1 attempt
        if q not in R.REDO and l1[q]['status'] == 'fail' and (l1[q]['attempts'][-1].get('errors') or []):
            R.REDO[q] = l1[q]['attempts'][-1]['errors'][:6]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(R.repair, todo))
    print(dict(Counter(r['status'] for q, r in R.res.items() if q in todo)))


if __name__ == '__main__':
    mode, todo = sys.argv[1], sys.argv[2:] or TODO
    print(mode, len(todo), flush=True)
    run_evidence(todo) if mode == 'evidence' else run_rewrite(todo)
