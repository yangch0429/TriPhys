# Provenance (TriPhysBench release): fix_code/evaluation/scripts/final_assemble.py
# Role: writes repaired QAs into the dataset files / re-assembles zero-shot results.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Final result files of the repaired benchmark from all evaluation rounds (runs on the server in
the server evaluation directory fix1001, imports experiment_right's scoring; judge gpt-4o-mini as in all runs).
For every model / class, every entry of the original result file (orig/ = evaluate/results) is matched to its QA of
the original benchmark (video + question) and so to the repaired QA at the same position (final/). Unchanged QAs keep
their entry. For a changed QA:
  response  the newest one to the same (video, question, question_type), the only inputs of the model; pools from
            old to new: orig/, out_scs/, out/ (run1), delta/out_scs/, delta salvage, delta/out*/ (delta round),
            delta/out_gfix/ (the one empty run1 Gemini B response, run again), assembled_prev/ (copy of assembled/
            before fix7, so its already judged entries are reused), fix7/out/ (the fix7 QAs, 10/5)
  SCS       taken from an entry of that response judged against the final annotated_solution, else judged again
  ACC       taken from an entry of that response scored against the final answer / answer_range / type, else scored
            again (calculate_acc_with_fallback, as in the runs)
-> assembled/<model>_<A|B>_class_results.json {'results': [...]}, assembled/assemble_log.json
Usage: python final_assemble.py [--dry]   (--dry: report only, nothing scored)"""
import asyncio, json, os, sys
from collections import Counter

sys.path.insert(0, os.environ.get('EXPERIMENT_RIGHT_DIR', '.'))  # server scoring code (metrics_final), not released
MODELS = ['Qwen3_5_4B', 'Qwen3_5_9B', 'Qwen3_6_27B', 'InteractiveOmni_4B', 'opr_GPT-5.5', 'opr_Gemini-3.6-flash']
POOLS = ['orig', 'out_scs', 'out', 'delta/out_scs', 'delta/out_p1/salvage_', 'delta/out', 'delta/out_p1',
         'delta/out_p2a', 'delta/out_p2b', 'delta/out_gfix', 'assembled_prev', 'fix7/out']
KEEP = ('question_id', 'video_id', 'take_uid', 'video_path')
DROP = ('_pending_scs', '_pending_eval', 'error')
DRY = '--dry' in sys.argv
CONC = 8


def jl(p):
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def sol(s):
    return '\n'.join(s) if isinstance(s, list) else (s or '')


def rng(x):
    return [round(float(v), 9) for v in x] if isinstance(x, (list, tuple)) and len(x) == 2 else None


def vkey(c, r):
    if c == 'A':
        return r['take_uid']
    p = r['video_path'].replace('\\', '/').rsplit('/', 2)
    return '%s/%s' % (p[1], p[2].rsplit('.', 1)[0])


def bench(d, c):
    """[(video key, qa)] in file order"""
    x = jl('%s/%sclass_qa_dataset_cleaned.json' % (d, c.lower()))
    if c == 'A':
        return [(v['take_uid'], q) for v in x for q in v['qa_pairs']]
    return [('%s/%s' % (q['folder_id'], q['video_id']), q) for q in x['qa_pairs']]


def pool_file(p, name):
    return p + name if p.endswith('_') else '%s/%s' % (p, name)


async def main():
    from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback
    os.makedirs('assembled', exist_ok=True)
    log = {}
    sem = asyncio.Semaphore(CONC)
    for c in 'AB':
        orig, fin = bench('orig', c), bench('final', c)
        assert len(orig) == len(fin) and all(a[0] == b[0] for a, b in zip(orig, fin))
        pos = {(v, q['question']): i for i, (v, q) in enumerate(orig)}
        assert len(pos) == len(orig)
        for m in MODELS:
            name = '%s_%s_class_results.json' % (m, c)
            entries = {}       # (video, question, type) -> [entry, ...] old -> new
            for p in POOLS:
                for r in (jl(pool_file(p, name)) or {}).get('results', []):
                    if 'error' in r or not (r.get('raw_response') or '').strip():
                        continue
                    entries.setdefault((vkey(c, r), r['question'], r.get('question_type')), []).append((p, r))
            res = jl('orig/' + name)['results']
            assert len(res) == len(orig), name
            out, todo_acc, todo_scs, src, rows = [], [], [], Counter(), []
            for r in res:
                i = pos[(vkey(c, r), r['question'])]
                o, f = orig[i][1], fin[i][1]
                same = all(o.get(k) == f.get(k) for k in ('question', 'answer', 'answer_range', 'question_type',
                                                          'annotated_solution'))
                if same:
                    out.append(r)
                    continue
                cand = entries.get((orig[i][0], f['question'], f['question_type']))
                if not cand and DRY:
                    src['MISSING'] += 1
                    out.append(r)
                    continue
                assert cand, '%s: no response for %s' % (name, f['question'][:90])
                p, best = cand[-1]
                same_resp = [e for _, e in cand if e['raw_response'] == best['raw_response']]
                fs, fr = sol(f.get('annotated_solution')), rng(f.get('answer_range'))
                scs = [e for e in same_resp if sol(e.get('annotated_solution')) == fs and e.get('scs_score') is not None]
                acc = [e for e in same_resp if e.get('ground_truth') == f['answer'] and rng(e.get('answer_range')) == fr
                       and e.get('acc_score') is not None]
                n = {k: v for k, v in best.items() if k not in DROP}
                n.update({k: r[k] for k in KEEP if k in r})
                n.update(question=f['question'], question_type=f['question_type'], ground_truth=f['answer'],
                         answer_range=f.get('answer_range'), annotated_solution=fs, difficulty=f.get('difficulty'))
                if scs:
                    n['scs_score'], n['scs_details'] = scs[-1]['scs_score'], scs[-1].get('scs_details')
                else:
                    todo_scs.append(n)
                if acc:
                    n['acc_score'] = acc[-1]['acc_score']
                else:
                    todo_acc.append(n)
                src[p.rstrip('_')] += 1
                rows.append({'idx': i, 'from': p.rstrip('_'), 'acc_rescored': not acc, 'scs_rejudged': not scs,
                             'old': [r.get('acc_score'), r.get('scs_score')], 'question': f['question'][:100]})
                out.append(n)
            print('%-45s changed %3d  from %s  ACC rescore %d  SCS rejudge %d' % (
                name, len(rows), dict(src), len(todo_acc), len(todo_scs)), flush=True)
            if not DRY:
                async def a(e):
                    async with sem:
                        e['acc_score'] = await calculate_acc_with_fallback(e)

                async def s(e):
                    async with sem:
                        for _ in range(3):     # as run_scs_subset.py
                            d = await calculate_scs_with_fallback(e)
                            if d.get('details') is not None:
                                break
                        e['scs_score'], e['scs_details'] = d['overall'], d['details']
                await asyncio.gather(*[a(e) for e in todo_acc], *[s(e) for e in todo_scs])
                for e in out:
                    e['ogs_score'] = 0.7 * (e.get('acc_score') or 0) + 0.3 * (e.get('scs_score') or 0)
                json.dump({'results': out}, open('assembled/' + name, 'w', encoding='utf-8'), ensure_ascii=False,
                          indent=1)
            log[name] = {'changed': len(rows), 'from': dict(src), 'acc_rescored': len(todo_acc),
                         'scs_rejudged': len(todo_scs), 'questions': rows}
    json.dump(log, open('assembled/assemble_log%s.json' % ('_dry' if DRY else ''), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    asyncio.run(main())
