# Provenance (TriPhysBench release): fix_code/_v8_work/l0_apply.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L0: apply the automatic v8 repairs (values unchanged) to the fix_code originals -> _v8_work/stage_L0/.
  ws   original_ok / replaced positions: the web_search source gets url = the real result that states the value,
       search_query, verified_quote (annotated url kept in url_annotated); replaced positions also get the
       validated value-free query in the annotated_solution step.
  code train blocks repaired automatically (repaired code reproduces the unchanged answer): code + real stdout
       (annotated code kept in code_annotated / key_information_annotated).
Then every code block of every QA (train + test) is executed again and the remaining problems are listed:
  ws_unresolved   no value-free query returns the annotated value
  ws_unlinked     web_search source that no annotated_solution search step retrieves
  code            block fails / prints nothing / does not reproduce the answer
Outputs: stage_L0/*.json, l0_log.json (per-QA changes), problems.json
Usage: python l0_apply.py"""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fixlib import REP, load, save, iter_qas, get_steps, set_steps, sources_of, check_codes, consistent

ws = json.load(open(REP + 'v8_wsfix_report.json', encoding='utf-8')) + \
     json.load(open(REP + 'v8_wsfix_report_test.json', encoding='utf-8'))
pos = {(p['qid'], p['pos']): p for f in ('v8_ws_positions.json', 'v8_ws_positions_test.json')
       for p in json.load(open(REP + f, encoding='utf-8'))}
codes = json.load(open(REP + 'v8_code_final.json', encoding='utf-8'))
ws_by = {}
for r in ws:
    ws_by.setdefault(r['qid'], []).append(r)
stat = Counter()
log, problems, jobs = {}, [], []


def map_sources(qid, qa, recs):
    """position -> index into the web_search sources (by the source key_information the position was built from)"""
    srcs = sources_of(qa, 'web_search')
    out, used = {}, set()
    for r in recs:
        ki = pos[(qid, r['pos'])].get('source_ki')
        k = next((j for j, s in enumerate(srcs) if j not in used and ki and s.get('key_information') == ki), None)
        if k is None and len(srcs) == r['n_ws'] and r['pos'] not in used:
            k = r['pos']
        if k is None:  # one search step with several sources: the source stating the position's value
            nums = [float(x) for x in (pos[(qid, r['pos'])]['ref'] or {}).get('numbers') or [] if isinstance(x, (int, float))]
            k = next((j for j, s in enumerate(srcs) if j not in used and nums and consistent(s.get('key_information'), nums)), None)
        if k is not None:
            out[r['pos']] = k
            used.add(k)
    return out


for split in ('train', 'test'):
    for cls in 'ab':
        data = load(cls, split)
        for qid, vid, qa in iter_qas(data, cls, split):
            ch = []
            recs = sorted(ws_by.get(qid, []), key=lambda r: r['pos'])
            srcs = sources_of(qa, 'web_search')
            smap = map_sources(qid, qa, recs)
            steps, kind = get_steps(qa)
            prob = {'qid': qid, 'split': split, 'cls': cls.upper(), 'difficulty': qa.get('difficulty'),
                    'question_type': qa.get('question_type'), 'ws_unresolved': [], 'ws_unlinked': [], 'code': []}
            for r in recs:
                p = pos[(qid, r['pos'])]
                if r['status'] == 'unresolved':
                    prob['ws_unresolved'].append({'pos': r['pos'], 'step_idx': p['step_idx'], 'query': r['query_annotated'],
                                                  'ref': p['ref'], 'source_idx': smap.get(r['pos']), 'reason': r['reason']})
                    continue
                if r['status'] == 'replaced':
                    i = p['step_idx']
                    assert r['query_annotated'] in steps[i], (qid, r['pos'])
                    steps[i] = steps[i].replace(r['query_annotated'], r['query'], 1)
                    ch.append({'kind': 'ws_query', 'step': i, 'old': r['query_annotated'], 'new': r['query']})
                    stat['ws_query_replaced'] += 1
                k = smap.get(r['pos'])
                if k is not None:
                    s = srcs[k]
                    s.setdefault('url_annotated', s.get('url'))
                    s.update(url=r['url'], search_query=r['query'], verified_quote=r['quote'])
                    ch.append({'kind': 'ws_source', 'source': k, 'url': r['url']})
                    stat['ws_source_verified'] += 1
                else:
                    stat['ws_source_unmapped'] += 1
            set_steps(qa, steps, kind)
            for j, s in enumerate(srcs):
                if j not in smap.values():
                    prob['ws_unlinked'].append({'source_idx': j, 'key_information': s.get('key_information'), 'url': s.get('url')})
            if split == 'train':
                blocks = codes.get(qid) or []
                csrc = sources_of(qa, 'code_executor')
                assert not blocks or len(blocks) == len(csrc), qid
                for i, (b, s) in enumerate(zip(blocks, csrc)):
                    if b['fixed'] and b['ok']:
                        s.setdefault('code_annotated', s.get('code'))
                        s.setdefault('key_information_annotated', s.get('key_information'))
                        s['code'], s['key_information'] = b['code'], b['out'][:500]
                        ch.append({'kind': 'code', 'block': i})
                        stat['code_replaced'] += 1
            if ch:
                log[qid] = ch
            jobs.append((qa, prob))
        save(data, cls, split, 'L0')


def recheck(job):
    qa, prob = job
    for i, r in enumerate(check_codes(qa)):
        if not r['ok']:
            prob['code'].append({'block': i, 'out': r['out'][:300], 'problem': r['problem']})


with ThreadPoolExecutor(8) as ex:
    list(ex.map(recheck, jobs))
problems = [p for _, p in jobs if p['ws_unresolved'] or p['ws_unlinked'] or p['code']]
json.dump(log, open('l0_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(problems, open('problems.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(dict(stat), '| QAs changed by L0:', len(log))
for split in ('train', 'test'):
    for cls in 'AB':
        ps = [p for p in problems if p['split'] == split and p['cls'] == cls]
        print('%s %s problem QAs=%d  unresolved=%d unlinked=%d code=%d | %s' % (
            split, cls, len(ps), sum(1 for p in ps if p['ws_unresolved']), sum(1 for p in ps if p['ws_unlinked']),
            sum(1 for p in ps if p['code']), dict(Counter(p['difficulty'] for p in ps))))
