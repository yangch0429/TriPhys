"""Re-score ACC of a merged eval (results/u_p1_<name>/merged.json) with metrics_final (judge requests retried on
429/5xx): when the gpt-4o-mini judge is rate-limited during evaluation, Step1 extractions fail and fall back to ACC 0.
SCS (DeepSeek Flash) is kept. Writes merged_rescored.json next to the input and prints old/new ACC per class.
Usage (from the work dir $TRIPHYS_WORK): python scripts/p1_rescore_acc.py NAME [NAME ...]"""
from srvpaths import E, EVAL_LIB  # machine paths, see srvpaths.py
import asyncio, json, sys
sys.path.insert(0, EVAL_LIB)
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_ogs

SEM = None


async def one(r, qa):
    p = parse_response(r['response'])
    rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': r['ground_truth'],
          'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
          'question': r['question'], 'question_type': r['question_type'], 'needs_llm_eval': p.needs_llm_eval,
          'raw_response': r['response']}
    async with SEM:
        acc = float(await calculate_acc_with_fallback(rd))
    return acc


async def main(name):
    global SEM
    SEM = asyncio.Semaphore(8)
    path = '%s/results/u_p1_%s/merged.json' % (E, name)
    d = json.load(open(path))
    for cl in ('aclass', 'bclass'):
        qas = {(q['question_type'], q['question']): q for v in json.load(open('%s/data/f10/%s_qa_test.json' % (E, cl)))
               for q in v['qa_pairs']}
        rows = d[cl]
        new = await asyncio.gather(*(one(r, qas[(r['question_type'], r['question'])]) for r in rows))
        old = [float(r['ACC']) for r in rows]
        for r, a in zip(rows, new):
            r['ACC_old'], r['ACC'] = float(r['ACC']), a
            r['OGS'] = calculate_ogs(a, float(r['SCS']))
        ch = sum(abs(a - b) > 1e-6 for a, b in zip(old, new))
        print('%s %s n=%d ACC old %.4f new %.4f (%d rows changed) SCS %.4f' % (name, cl, len(rows), sum(old) / len(old),
              sum(new) / len(new), ch, sum(float(r['SCS']) for r in rows) / len(rows)), flush=True)
    json.dump(d, open(path.replace('merged.json', 'merged_rescored.json'), 'w'), ensure_ascii=False, indent=1)


for n in sys.argv[1:]:
    asyncio.run(main(n))
