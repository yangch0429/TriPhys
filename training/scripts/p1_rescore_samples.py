"""10/17 Re-score ACC of self-sample shards (results/<tag>_r{0..3}/*_shard*.json) whose ACC < .9 with experiment_right
metrics_final (429/5xx retry): the gpt-4o-mini judge was saturated during sampling ("Step1 extraction failed:
'choices'"), which can turn a correct trajectory into a failure. Rows with ACC >= .9 are kept. Shards are rewritten in
place (ACC_old keeps the first score); a copy of each directory is saved as <dir>_bak1017 first.
10/18: reps r0..r7 (p1_ssA has 8), missing dirs skipped.
Usage (from the work dir $TRIPHYS_WORK): python scripts/p1_rescore_samples.py p1_ss p1_ss2"""
from srvpaths import E, EVAL_LIB  # machine paths, see srvpaths.py
import asyncio, glob, json, os, shutil, sys
sys.path.insert(0, EVAL_LIB)
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_ogs



def qa_index():
    idx = {}
    for cl in ('aclass', 'bclass'):
        for v in json.load(open('%s/data/f10/%s_qa_train.json' % (E, cl))):
            for q in v['qa_pairs']:
                idx[(q['question_type'], q['question'])] = q
    return idx


async def rescore(r, qa, sem):
    resp = r.get('response') or ''
    p = parse_response(resp)
    rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': r['ground_truth'],
          'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
          'question': r['question'], 'question_type': r['question_type'], 'needs_llm_eval': p.needs_llm_eval,
          'raw_response': resp}
    async with sem:
        return float(await calculate_acc_with_fallback(rd))


async def main(tags):
    idx, sem = qa_index(), asyncio.Semaphore(6)
    for tag in tags:
        for r in range(8):
            d = '%s/results/%s_r%d' % (E, tag, r)
            if not os.path.isdir(d):
                continue
            if not os.path.exists(d + '_bak1017'):
                shutil.copytree(d, d + '_bak1017')
            n_re = n_up = 0
            for f in sorted(glob.glob(d + '/*_shard*.json')):
                if 'serper_failures' in f:
                    continue
                j = json.load(open(f))
                rows = [x for x in j.get('results', []) if float(x['ACC']) < .9 and 'ACC_old' not in x]
                new = await asyncio.gather(*(rescore(x, idx[(x['question_type'], x['question'])], sem) for x in rows))
                for x, a in zip(rows, new):
                    x['ACC_old'] = float(x['ACC'])
                    n_up += a >= .9 > x['ACC_old']
                    x['ACC'] = a
                    x['OGS'] = calculate_ogs(a, float(x['SCS']))
                n_re += len(rows)
                json.dump(j, open(f, 'w'), ensure_ascii=False)
            print('%s_r%d: re-scored %d rows, %d became ACC >= .9' % (tag, r, n_re, n_up), flush=True)


asyncio.run(main(sys.argv[1:]))
