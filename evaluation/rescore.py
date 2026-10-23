"""Score (or re-score) saved responses: rows without scores, rows with judge errors, or all rows (--all).

  python evaluation/rescore.py results/qwen35_4b.jsonl [--all] [--thinking_opened]
Scoring = triphys.metrics.score_response (triphys.parser + triphys.metrics_final, as in training/n1/eval_n1.py)."""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triphys import data, metrics


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--thinking_opened', action='store_true')
    ap.add_argument('--concurrency', type=int, default=8)
    a = ap.parse_args()
    qas = {x['qid']: x['qa'] for x in data.load(('MPR', 'KPR'), 'all')}
    sem = asyncio.Semaphore(a.concurrency)

    async def one(r):
        if not (a.all or 'ACC' not in r or r.get('judge_errors')):
            return r
        async with sem:
            r.update(await metrics.score_response(qas[r['qid']], r.get('response') or '', a.thinking_opened))
        return r

    for f in a.files:
        rows = [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
        rows = await asyncio.gather(*(one(r) for r in rows))
        with open(f, 'w', encoding='utf-8') as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f, metrics.summarize(rows), 'judge errors:', sum(bool(r.get('judge_errors')) for r in rows))


if __name__ == '__main__':
    asyncio.run(main())
