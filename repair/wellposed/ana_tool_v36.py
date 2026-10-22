# Provenance (TriPhysBench release): fix_code/training/v36/ana_tool_v36.py
# Role: well-posedness / quality check of the 445 QAs of version 07 (10-11); base module of fix9 / fix10.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""v36 (10/11, user: Phase 2 should check whether web_search and code_executor are used correctly; if the searched value
is right and the code output is right, the final answer is probably right; ignore CV values for now).
Tool-use funnel of E17 (checkpoint-183) on TRAIN-set trajectories (results/dpo_samples_*), A and B separately:
  search  the trajectory called web_search
  ret     a key number of EVERY annotated web_search source (sources[type=web_search].key_information) appears in the
          FULL search results the model saw (serper_cache_train holds the exact eval string; tool_log is cut at 200 chars)
  used    each of those key numbers also appears in the model's code_executor code or its final response
  code    code_executor was called;  code_ok: a number printed by the code lies inside answer_range (+-2%)
  acc     ACC >= .5
Stage of the first failure per trajectory: no_search / not_retrieved / not_used / no_code / code_wrong / copy_wrong / ok_but_wrong / correct
Usage (server, fc/training/v36): python ana_tool_v36.py [sample dirs ...]"""
import os, re, sys, ast, json, glob, hashlib
from collections import Counter, defaultdict

ROOT = os.environ.get('TRIPHYS_EXP_ROOT', '.')
CACHE = ROOT + '/results/serper_cache_train'
NUM = r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?'
TRIVIAL = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 0.25, 9.81, 9.8, 9.807, 9.80665, 100.0, 1000.0, 1e6, 1e-3, 1e-6, 60.0,
           3600.0, 180.0, 360.0, 90.0, 3.14159, 3.1416, 3.14, 2.54, 0.0254, 12.0, 0.3048, 0.4536, 0.453592, 2.20462, 1e3,
           6.0, 8.0, 20.0, 50.0, 0.1, 0.01, 0.001, 1.5, 15.0, 30.0, 45.0}
CONV = [1, 1e3, 1e-3, 1e2, 1e-2, 1e6, 1e-6, 1 / 0.0254, 1 / 0.3048, 1 / 0.4536, 0.4536, 0.0254, 0.3048, 1 / 28.35, 1 / 3.6,
        3.6, 1 / 9.81, 9.81, 0.5, 2]
STAGES = ['no_search', 'not_retrieved', 'not_used', 'no_code', 'code_wrong', 'copy_wrong', 'ok_but_wrong', 'correct']


def nums(s):
    out = []
    for m in re.findall(NUM, str(s or '')):
        try:
            out.append(float(m.replace(',', '')))
        except ValueError:
            pass
    return out


def has(key, text):
    t = nums(text)
    return any(any(abs(x - k) <= 0.01 * abs(k) for x in t) for k in key if k != 0.0)


def srcs(r):
    s = r.get('sources')
    s = ast.literal_eval(s) if isinstance(s, str) else (s or [])
    return [x for x in s if isinstance(x, dict) and x.get('type') == 'web_search']


def seen(q):
    f = os.path.join(CACHE, hashlib.sha1(q.strip().encode()).hexdigest() + '.json')
    if not os.path.exists(f):
        return None
    r = json.load(open(f))['r']
    try:
        return json.loads(r).get('result', r)
    except Exception:
        return r


def in_range(vals, rng, gt):
    try:
        lo, hi = float(rng[0]), float(rng[1])
    except Exception:
        g = nums(gt)
        if not g:
            return False
        lo = hi = g[0]
    lo, hi = min(lo, hi), max(lo, hi)
    pad = 0.02 * max(abs(lo), abs(hi))
    return any(lo - pad <= v <= hi + pad for v in vals)


def row(r, cls):
    tl = r.get('tool_log') or []
    tl = [] if isinstance(tl, str) else tl
    ws = [t for t in tl if t.get('tool') == 'web_search']
    ce = [t for t in tl if t.get('tool') == 'code_executor']
    keys = [nums(w.get('key_information')) for w in srcs(r)]
    keys = [k for k in keys if k]
    res = [seen(t['args'].get('query', '')) for t in ws]
    miss = sum(x is None for x in res)
    full = ' | '.join(x or str(t.get('result', '')) for x, t in zip(res, ws))
    code = ' '.join(str(t['args'].get('code', '')) for t in ce)
    cout = [v for t in ce for v in nums(t.get('result', ''))]
    rng, gt = r.get('answer_range'), r.get('ground_truth')
    d = dict(cls=cls, qtype=r.get('question_type'), qid=r['question_id'], nkey=len(keys), nsearch=len(ws), ncode=len(ce),
             cache_miss=miss, acc=float(r.get('ACC') or 0) >= 0.5,
             ret_src=[has(k, full) for k in keys], used_src=[has(k, code + ' ' + str(r.get('response', ''))) for k in keys],
             code_ok=bool(ce) and in_range(cout, rng, gt), ans_ok=in_range(nums(r.get('parsed_answer')), rng, gt))
    outs = [str(t.get('result', '')) for t in ce]
    d['code_err'] = any(o.startswith('Error') or 'Traceback' in o for o in outs)
    d['code_err_last'] = bool(outs) and (outs[-1].startswith('Error') or 'Traceback' in outs[-1])
    d['code_repeat'] = any(outs[i].startswith('Error') and ce[i]['args'].get('code') == ce[i + 1]['args'].get('code')
                           for i in range(len(ce) - 1))
    d['ws_empty'] = sum(not x or 'No results found' in x for x in res)
    # numeric literals in the code that appear nowhere the model could have read them (question, search results,
    # earlier code output) even after a common unit conversion -> made-up constants
    seen_txt = nums(str(r.get('question', '')) + ' ' + full + ' ' + ' '.join(outs))
    lits = [v for v in nums(' '.join(re.sub(r'#.*', '', str(t['args'].get('code', ''))) for t in ce))
            if v not in TRIVIAL and abs(v) >= 1e-12]
    d['nlit'] = len(lits)
    d['unsourced'] = [v for v in lits if not any(abs(v * f - x) <= 0.015 * abs(x) for f in CONV for x in seen_txt if x)]
    d['ret'] = bool(keys) and all(d['ret_src'])
    d['used'] = bool(keys) and all(u for u, x in zip(d['used_src'], d['ret_src']))
    if d['acc']:
        st = 'correct'
    elif not ws:
        st = 'no_search'
    elif keys and not d['ret']:
        st = 'not_retrieved'
    elif keys and not d['used']:
        st = 'not_used'
    elif not ce:
        st = 'no_code'
    elif not d['code_ok']:
        st = 'code_wrong'
    elif not d['ans_ok']:
        st = 'copy_wrong'
    else:
        st = 'ok_but_wrong'
    d['stage'] = st
    return d


def load(dirs):
    out = []
    for dd in dirs:
        for f in sorted(glob.glob('%s/results/%s/*_shard*.json' % (ROOT, dd))):
            if 'fail' in f:
                continue
            for r in json.load(open(f))['results']:
                if r.get('serper_failed'):
                    continue
                d = row(r, 'A' if '_aclass_' in os.path.basename(f) else 'B')
                d['src'] = dd
                out.append(d)
    return out


def m(x):
    x = list(x)
    return '%.3f' % (sum(x) / len(x)) if x else '  -  '


if __name__ == '__main__':
    dirs = sys.argv[1:] or ['dpo_samples_v23', 'dpo_samples_v32']
    R = load(dirs)
    print('trajectories', len(R), 'QAs', len({d['qid'] for d in R}), 'cache-miss queries', sum(d['cache_miss'] for d in R))
    for c in 'AB':
        X = [d for d in R if d['cls'] == c]
        K = [d for d in X if d['nkey']]
        print('\n== %s-class: %d traj, %d QAs, ACC %s' % (c, len(X), len({d['qid'] for d in X}), m(d['acc'] for d in X)))
        print('  search %s | ret %s | used|ret %s | code %s | code_ok %s | code_ok|ret&used %s | ACC|code_ok %s | ACC|!code_ok %s'
              % (m(d['nsearch'] > 0 for d in X), m(d['ret'] for d in K), m(d['used'] for d in K if d['ret']),
                 m(d['ncode'] > 0 for d in X), m(d['code_ok'] for d in X),
                 m(d['code_ok'] for d in K if d['ret'] and d['used']), m(d['acc'] for d in X if d['code_ok']),
                 m(d['acc'] for d in X if not d['code_ok'])))
        print('  ACC | ret %s  !ret %s ; per-source ret %s' % (m(d['acc'] for d in K if d['ret']),
              m(d['acc'] for d in K if not d['ret']), m(x for d in K for x in d['ret_src'])))
        print('  ws_empty %s | code_err %s (last call err %s, same code re-run after err %s) | ACC|code_err %s'
              % (m(d['ws_empty'] > 0 for d in X), m(d['code_err'] for d in X), m(d['code_err_last'] for d in X),
                 m(d['code_repeat'] for d in X), m(d['acc'] for d in X if d['code_err'])))
        print('  code literals %d, unsourced %s | traj with unsourced literal %s | ACC with %s / without %s'
              % (sum(d['nlit'] for d in X), m(x for d in X for x in [1] * len(d['unsourced']) + [0] * (d['nlit'] - len(d['unsourced']))),
                 m(bool(d['unsourced']) for d in X), m(d['acc'] for d in X if d['unsourced']),
                 m(d['acc'] for d in X if not d['unsourced'])))
        for s in ('not_retrieved', 'code_wrong'):
            Y = [d for d in X if d['stage'] == s]
            print('  within %s: code_err %s, unsourced literal %s' % (s, m(d['code_err'] for d in Y), m(bool(d['unsourced']) for d in Y)))
        st = Counter(d['stage'] for d in X)
        print('  first-failure stage: ' + '  '.join('%s %d (%.0f%%)' % (s, st[s], 100 * st[s] / len(X)) for s in STAGES))
        q = defaultdict(list)
        for d in X:
            q[d['qid']].append(d)
        qs = {k: Counter(x['stage'] for x in v).most_common(1)[0][0] for k, v in q.items()}
        print('  per-QA majority stage: ' + '  '.join('%s %d' % (s, n) for s, n in Counter(qs.values()).most_common()))
        print('  QAs ever ret %s / ever code_ok %s / ever correct %s' % (
            m(any(x['ret'] for x in v) for v in q.values()), m(any(x['code_ok'] for x in v) for v in q.values()),
            m(any(x['acc'] for x in v) for v in q.values())))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ana_tool_v36.json'), 'w'))
