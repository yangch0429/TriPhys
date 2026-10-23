"""Merge of sharded evaluation results. merge(NAME) is what the training scripts use:
  (cd scripts && python -c "import run_det; run_det.merge('NAME')")
collects results/det_<NAME>/*_{aclass,bclass}_shard*.json (Serper failure files excluded) into
results/det_<NAME>/merged.json with a summary per class (A, B, overall: n, ACC, SCS, OGS, no_answer, serper_failed,
ACC by question type, mean SCS components when eval_n1 stored them) and logs one summary line.
The __main__ block (a batch evaluation of the models listed in MODELS) is not used by the training pipeline; it needs
an evaluator script and model directories that are not part of this repository."""
from srvpaths import BASE_MODEL, E  # machine paths, see srvpaths.py
import os, sys, json, glob, signal, subprocess, time
from collections import defaultdict

ROOT = E
PYTHON = sys.executable
LOGS = ROOT + '/logs/det'
GPUS = [0, 1, 3]
M = ROOT + '/outputs/e17_merged'
BASE4B = BASE_MODEL
MODELS = {'base': ('none', BASE4B), 'e17': (ROOT + '/outputs/sft_clean_ab_v9_3ep/checkpoint-183', BASE4B),
          'v4': (ROOT + '/outputs/dpo_v12/final_model', M), 'v5L': (ROOT + '/outputs/dpo_v13/final_model', M),
          'e17_r2': (ROOT + '/outputs/sft_clean_ab_v9_3ep/checkpoint-183', BASE4B)}
WAVES = [['base', 'e17'], ['v4', 'v5L'], ['e17_r2']]
os.makedirs(LOGS, exist_ok=True)


def log(msg):
    print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + '] ' + msg, flush=True)


def stop_holder():
    pf = ROOT + '/logs/hold_gpus.pid'
    if os.path.exists(pf):
        try:
            os.kill(int(open(pf).read().strip()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        os.remove(pf)
        time.sleep(20)


def start_holder():
    log('GPU holder disabled (GPUs are occupied only by real tasks)')
    return
    p = subprocess.Popen([PYTHON, ROOT + '/scripts/hold_idle_gpus.py', '0,1,2,3'], stdout=open(ROOT + '/logs/hold_gpus.log', 'a'),
                         stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    open(ROOT + '/logs/hold_gpus.pid', 'w').write(str(p.pid))
    log('holder restarted PID=%d' % p.pid)


def procs(name):
    path, base = MODELS[name]
    rd = '%s/results/det_%s' % (ROOT, name)
    os.makedirs(rd, exist_ok=True)
    return [subprocess.Popen([PYTHON, ROOT + '/scripts/eval_det_v9.py', '--model_path', path, '--base_model', base,
                              '--class', 'both', '--output_dir', rd, '--gpu', '0', '--n_shards', '3', '--shard_id', str(i)],
                             stdout=open('%s/%s_shard%d.log' % (LOGS, name, i), 'w'), stderr=subprocess.STDOUT, cwd=ROOT,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), JUDGE_TEMP='0'))
            for i, g in enumerate(GPUS)]


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


def same(a, b):
    """determinism check: fraction of QAs with identical response / ACC between two runs"""
    ra = {(c, r['question']): r for c in ('aclass', 'bclass') for r in a[c]}
    rb = {(c, r['question']): r for c in ('aclass', 'bclass') for r in b[c]}
    ks = [k for k in ra if k in rb]
    log('determinism e17 vs e17_r2: %d QAs, identical response %d, identical ACC %d, |dSCS|<=0.05 %d'
        % (len(ks), sum(ra[k]['response'] == rb[k]['response'] for k in ks),
           sum(abs(ra[k]['ACC'] - rb[k]['ACC']) < 1e-9 for k in ks), sum(abs(ra[k]['SCS'] - rb[k]['SCS']) <= .05 for k in ks)))


if __name__ == '__main__':
    stop_holder()
    res = {}
    for wave in WAVES:
        todo = [n for n in wave if not os.path.exists('%s/results/det_%s/merged.json' % (ROOT, n))]
        ps = [p for n in todo for p in procs(n)]
        for p in ps:
            p.wait()
        log('wave %s exit codes %s' % (wave, [p.returncode for p in ps]))
        for n in wave:
            res[n] = merge(n)
    if 'e17' in res and 'e17_r2' in res:
        same(res['e17'], res['e17_r2'])
    start_holder()
    log('run_det finished')
