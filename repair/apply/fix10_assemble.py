# Provenance (TriPhysBench release): fix_code/evaluation/scripts/fix10_assemble.py
# Role: writes repaired QAs into the dataset files / re-assembles zero-shot results.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 result files (runs on the server in the server evaluation directory fix1001, imports experiment_right's scoring, judge gpt-4o-mini
as in all runs). Base: assembled/ (v9 results, one entry per v9 QA, original file order). fix10/fix10_log.json (from
fix10_apply.py) gives per needs_fix2 QA 'delete' or 'replace' + its video key and old question / answer.
  delete    the entry is dropped (the v10 file keeps the order of the others)
  replace   question and question_type unchanged: the v9 response is kept; else the newest response to the same
            (video, question, question_type) from the fix10 pools (fix10/out_api, out_s*, then round 2 out2_api, out2_s*).
            ACC taken from that response's entry if scored against the final answer / answer_range, else scored again;
            SCS likewise against the final annotated_solution (calculate_*_with_fallback, as in the runs)
-> assembled10/<model>_<A|B>_class_results.json {'results': [...]}, assembled10/assemble_log.json
Usage: python fix10_assemble.py [--dry]"""
import asyncio, json, os, sys
from collections import Counter

sys.path.insert(0, os.environ.get('EXPERIMENT_RIGHT_DIR', '.'))  # server scoring code (metrics_final), not released
MODELS = ['Qwen3_5_4B', 'Qwen3_5_9B', 'Qwen3_6_27B', 'InteractiveOmni_4B', 'opr_GPT-5.5', 'opr_Gemini-3.6-flash']
# out_h*: round-1 9B / InteractiveOmni half-shards (group_fast.sh, 2 processes per GPU); out_s4: 27B B split of s3 (split_s3.sh)
POOLS = (['fix10/out_api'] + ['fix10/out_s%d' % s for s in range(5)] + ['fix10/out_h%d' % s for s in range(8)] +
         ['fix10/out2_api'] + ['fix10/out2_s%d' % s for s in range(4)])
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


async def main():
    from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback
    flog = jl('fix10/fix10_log.json')
    os.makedirs('assembled10', exist_ok=True)
    log, sem = {}, asyncio.Semaphore(CONC)
    for c in 'AB':
        act = {}
        for k, e in flog.items():
            if k.startswith(c + '|'):
                act.setdefault((e['video'], e['old_question']), []).append((k, e))
        for m in MODELS:
            name = '%s_%s_class_results.json' % (m, c)
            entries = {}
            for p in POOLS:
                for r in (jl('%s/%s' % (p, name)) or {}).get('results', []):
                    if 'error' in r or not (r.get('raw_response') or '').strip():
                        continue
                    entries.setdefault((vkey(c, r), r['question'], r.get('question_type')), []).append((p, r))
            res = jl('assembled/' + name)['results']
            out, todo_acc, todo_scs, src, rows, used = [], [], [], Counter(), [], set()
            for r in res:
                es = act.get((vkey(c, r), r['question']), [])
                if len(es) > 1:
                    es = [x for x in es if x[1]['old_answer'] == r['ground_truth']]
                assert len(es) <= 1, (name, r['question'][:80])
                if not es:
                    out.append(r)
                    continue
                k, e = es[0]
                used.add(k)
                if e['action'] == 'delete':
                    src['deleted'] += 1
                    continue
                f = {'question': e['new_question'], 'question_type': e['question_type'], 'answer': e['new_answer'],
                     'answer_range': e['answer_range'], 'annotated_solution': e['annotated_solution'],
                     'difficulty': e['difficulty']}
                key = (vkey(c, r), f['question'], f['question_type'])
                if f['question'] == r['question'] and f['question_type'] == r['question_type']:
                    p, best, cand = 'v9', r, [r]
                elif entries.get(key):
                    p, best = entries[key][-1]
                    cand = [x for _, x in entries[key] if x['raw_response'] == best['raw_response']]
                else:
                    assert DRY, '%s: no response for %s' % (name, f['question'][:90])
                    src['MISSING'] += 1
                    out.append(r)
                    continue
                fs, fr = sol(f['annotated_solution']), rng(f['answer_range'])
                scs = [x for x in cand if sol(x.get('annotated_solution')) == fs and x.get('scs_score') is not None]
                acc = [x for x in cand if x.get('ground_truth') == f['answer'] and rng(x.get('answer_range')) == fr
                       and x.get('acc_score') is not None]
                n = {x: v for x, v in best.items() if x not in DROP}
                n.update({x: r[x] for x in KEEP if x in r})
                n.update(question=f['question'], question_type=f['question_type'], ground_truth=f['answer'],
                         answer_range=f['answer_range'], annotated_solution=fs, difficulty=f['difficulty'])
                if scs:
                    n['scs_score'], n['scs_details'] = scs[-1]['scs_score'], scs[-1].get('scs_details')
                else:
                    todo_scs.append(n)
                if acc:
                    n['acc_score'] = acc[-1]['acc_score']
                else:
                    todo_acc.append(n)
                src[p] += 1
                rows.append({'id': k, 'from': p, 'acc_rescored': not acc, 'scs_rejudged': not scs,
                             'old': [r.get('acc_score'), r.get('scs_score')], 'question': f['question'][:100]})
                out.append(n)
            miss = [k for k in flog if k.startswith(c + '|') and k not in used]
            assert not miss, (name, 'log entries without a v9 result', miss[:5])
            print('%-45s %d -> %d  changed %3d  from %s  ACC rescore %d  SCS rejudge %d' % (
                name, len(res), len(out), len(rows), dict(src), len(todo_acc), len(todo_scs)), flush=True)
            if not DRY:
                async def a(e):
                    async with sem:
                        e['acc_score'] = await calculate_acc_with_fallback(e)

                async def s(e):
                    async with sem:
                        for _ in range(3):  # as run_scs_subset.py
                            d = await calculate_scs_with_fallback(e)
                            if d.get('details') is not None:
                                break
                        e['scs_score'], e['scs_details'] = d['overall'], d['details']
                await asyncio.gather(*[a(e) for e in todo_acc], *[s(e) for e in todo_scs])
                for e in out:
                    e['ogs_score'] = 0.7 * (e.get('acc_score') or 0) + 0.3 * (e.get('scs_score') or 0)
                bad = [e['question'][:80] for e in out if e.get('acc_score') is None or e.get('scs_score') is None]
                assert not bad, (name, 'unscored', bad[:5])
                json.dump({'results': out}, open('assembled10/' + name, 'w', encoding='utf-8'), ensure_ascii=False,
                          indent=1)
            log[name] = {'v9': len(res), 'v10': len(out), 'changed': len(rows), 'from': dict(src),
                         'acc_rescored': len(todo_acc), 'scs_rejudged': len(todo_scs), 'questions': rows}
    json.dump(log, open('assembled10/assemble_log%s.json' % ('_dry' if DRY else ''), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    asyncio.run(main())
