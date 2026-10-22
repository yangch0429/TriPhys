# Provenance (TriPhysBench release): fix_code/_v8_work/v8_writeback.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""
Write v8 fixes back into the benchmark TRAIN files (backup *_qa_train.bak0930.json first):
  ws    replaced positions: the query in the annotated_solution step is replaced by the validated query;
        the matching web_search source (only when #sources == #searches) gets url = page that states the value,
        search_query, verified_quote (old url kept in url_annotated). original_ok positions get the same source fields.
  code  repaired code blocks replace sources[type=code_executor][i].code; key_information = real stdout
        (old values kept in code_annotated / key_information_annotated)
  flags qa['v8_flags']: unresolved ws positions and code blocks whose output disagrees with the answer
Inputs: v8_wsfix_report.json, v8_ws_positions.json (step_idx), v8_code_final.json
Usage: python v8_writeback.py
"""
import ast, json, os, shutil
from collections import Counter
from v8_common import DATA

BAK = '.bak0930.json'
ws = json.load(open(DATA + '/v8_wsfix_report.json'))
step_idx = {(p['qid'], p['pos']): p['step_idx'] for p in json.load(open(DATA + '/v8_ws_positions.json'))}
codes = json.load(open(DATA + '/v8_code_final.json'))
ws_by = {}
for r in ws:
    ws_by.setdefault(r['qid'], []).append(r)
stat = Counter()


def get_steps(qa):
    sol = qa.get('annotated_solution')
    if isinstance(sol, list):
        return list(sol), 'list'
    try:
        v = ast.literal_eval(sol)
        if isinstance(v, list):
            return [str(x) for x in v], 'repr'
    except Exception:
        pass
    return None, 'str'


def fix_ws(qid, qa, flags):
    steps, kind = get_steps(qa)
    srcs = [s for s in qa.get('sources', []) if s.get('type') == 'web_search']
    for r in sorted(ws_by.get(qid, []), key=lambda r: r['pos']):
        if r['status'] == 'unresolved':
            flags.append({'type': 'ws_unresolved', 'pos': r['pos'], 'query': r['query_annotated'], 'expected': r['expected'],
                          'reason': r['reason'], **({'value_url': r['value_url'], 'value_quote': r['value_quote']}
                                                    if r['reason'] == 'value_found_no_query' else {})})
            stat['ws_unresolved'] += 1
            continue
        if r['status'] == 'replaced':
            old, new = r['query_annotated'], r['query']
            if steps is not None:
                i = step_idx[(qid, r['pos'])]
                assert old in steps[i], (qid, r['pos'])
                steps[i] = steps[i].replace(old, new, 1)
            else:
                assert old in qa['annotated_solution'], (qid, r['pos'])
                qa['annotated_solution'] = qa['annotated_solution'].replace(old, new, 1)
            stat['ws_query_replaced'] += 1
        if len(srcs) == r['n_ws']:
            s = srcs[r['pos']]
            s.setdefault('url_annotated', s.get('url'))
            s.update(url=r['url'], search_query=r['query'], verified_quote=r['quote'])
            stat['ws_source_updated'] += 1
    if steps is not None:
        qa['annotated_solution'] = steps if kind == 'list' else repr(steps)


def fix_code(qid, qa, flags):
    blocks = codes.get(qid)
    srcs = [s for s in qa.get('sources', []) if s.get('type') == 'code_executor']
    if not blocks or len(blocks) != len(srcs):
        stat['code_unaligned' if blocks else 'code_not_checked'] += 1
        return
    for i, (b, s) in enumerate(zip(blocks, srcs)):
        if b['fixed']:
            s.setdefault('code_annotated', s.get('code'))
            s.setdefault('key_information_annotated', s.get('key_information'))
            s['code'], s['key_information'] = b['code'], b['out'][:500]
            stat['code_replaced'] += 1
        d = b.get('diag') or {}
        if not b['ok'] and d.get('verdict') == 'answer_inconsistent' and d.get('grounded') and not d['out'].startswith(('Error', 'No output')):
            # code is correct, the benchmark answer is not: store the correct code, flag the answer
            s.setdefault('code_annotated', s.get('code'))
            s.setdefault('key_information_annotated', s.get('key_information'))
            s['code'], s['key_information'] = d['code'], d['out'][:500]
            stat['code_replaced_answer_inconsistent'] += 1
        if not b['ok']:
            f = {'type': 'code_inconsistent', 'block': i, 'output': b['out'][:300], 'answer': str(qa.get('answer'))[:300],
                 'verdict': d.get('verdict'), 'why': d.get('why')}
            if d.get('verdict') == 'answer_inconsistent':
                f.update(correct_answer=d.get('correct_answer'), code_suggested=d.get('code'), output_suggested=d.get('out'))
            flags.append(f)
            stat['code_flagged_%s' % d.get('verdict')] += 1


for cls in 'ab':
    fn = '%s/%sclass_qa_train.json' % (DATA, cls)
    if not os.path.exists(fn[:-5] + BAK):
        shutil.copy(fn, fn[:-5] + BAK)
    data = json.load(open(fn[:-5] + BAK))  # always rebuild from the backup, so reruns are idempotent
    for vid in data:
        vtag = vid.get('take_uid') or '%s/%s' % (vid.get('folder_id'), vid.get('video_id'))
        for i, qa in enumerate(vid['qa_pairs']):
            qid = '%s|%s|%d' % (cls.upper(), vtag, i)
            flags = []
            fix_ws(qid, qa, flags)
            fix_code(qid, qa, flags)
            if flags:
                qa['v8_flags'] = flags
                stat['%s_qa_flagged' % cls.upper()] += 1
    json.dump(data, open(fn, 'w'), ensure_ascii=False, indent=2)
    print('wrote', fn)
print(dict(stat))
