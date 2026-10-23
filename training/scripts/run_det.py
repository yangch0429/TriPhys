"""Merge of sharded evaluation results. merge(NAME) is what the training scripts use:
  (cd scripts && python -c "import run_det; run_det.merge('NAME')")
collects results/det_<NAME>/*_{aclass,bclass}_shard*.json (Serper failure files excluded) into
results/det_<NAME>/merged.json with a summary per class (A, B, overall: n, ACC, SCS, OGS, no_answer, serper_failed,
ACC by question type, mean SCS components when eval_n1 stored them) and logs one summary line."""
from srvpaths import E  # machine paths, see srvpaths.py
import json, glob, time
from collections import defaultdict

ROOT = E


def log(msg):
    print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + '] ' + msg, flush=True)


def merge(name):
    rd = '%s/results/det_%s' % (ROOT, name)
    out = {c: [r for f in sorted(glob.glob('%s/*_%s_shard*.json' % (rd, c))) if 'serper_failures' not in f
               for r in json.load(open(f)).get('results', [])] for c in ('aclass', 'bclass')}
    summ = {}
    for k, items in (('A', out['aclass']), ('B', out['bclass']), ('overall', out['aclass'] + out['bclass'])):
        if not items:
            continue
        acc = sum(r['ACC'] for r in items) / len(items)
        scs = sum(r['SCS'] for r in items) / len(items)
        bt = defaultdict(list)
        for r in items:
            bt[r['question_type']].append(r['ACC'])
        summ[k] = {'n': len(items), 'ACC': round(acc, 4), 'SCS': round(scs, 4), 'OGS': round(0.7 * acc + 0.3 * scs, 4),
                   'no_answer': sum(1 for r in items if '<answer>' not in (r.get('response') or '')),
                   'serper_failed': sum(1 for r in items if r.get('serper_failed')),
                   'acc_by_type': {t: round(sum(v) / len(v), 4) for t, v in sorted(bt.items())}}
        dd = [r['scs_details'] for r in items if r.get('scs_details')]  # eval_n1 stores the 4 SCS dims
        if dd:
            summ[k]['SCS_dims'] = {d: round(sum(x[d] for x in dd) / len(dd), 4)
                                   for d in ('S_obs', 'S_formula', 'S_derive', 'S_conclude')} | {'n_with_details': len(dd)}
    json.dump({'summary': summ, **out}, open(rd + '/merged.json', 'w'), ensure_ascii=False, indent=1)
    log('%-7s %s' % (name, '  '.join('%s %d ACC %.4f SCS %.4f OGS %.4f noans %d sfail %d' % (
        k, v['n'], v['ACC'], v['SCS'], v['OGS'], v['no_answer'], v['serper_failed']) for k, v in summ.items())))
    return out
