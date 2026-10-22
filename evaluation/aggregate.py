"""Merge result shards and report ACC / SCS / OGS for MPR, KPR and Overall (pooled question-level scores), plus the
difficulty subgroups and the mean SCS components (appendix "Score Aggregation and Reporting").

  python evaluation/aggregate.py results/qwen35_4b_shard*.jsonl --out results/qwen35_4b_summary.json"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triphys import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    rows = {}
    for f in a.files:
        for l in open(f, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                rows[r['qid']] = r
    rows = list(rows.values())
    unscored = [r['qid'] for r in rows if 'ACC' not in r]
    if unscored:
        sys.exit('%d rows are not scored yet (evaluation/rescore.py)' % len(unscored))
    groups = defaultdict(list)
    for r in rows:
        groups[r['subtask']].append(r)
        groups['Overall'].append(r)
        groups['%s/%s' % (r['subtask'], r.get('difficulty'))].append(r)
    out = {g: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.summarize(rs).items()}
           for g, rs in sorted(groups.items())}
    out['_info'] = {'judge_errors': sum(bool(r.get('judge_errors')) for r in rows),
                    'serper_failed': sum(bool(r.get('serper_failed')) for r in rows),
                    'no_tool_call': sum(not r.get('n_tool_calls') for r in rows)}
    for g in ('MPR', 'KPR', 'Overall'):
        if g in out:
            s = out[g]
            print('%-8s n=%3d  ACC %.4f  SCS %.4f  OGS %.4f' % (g, s['n'], s['ACC'], s['SCS'], s['OGS']))
    print(out['_info'])
    if a.out:
        json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
