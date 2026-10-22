# Provenance (TriPhysBench release): fix_code/_v8_work/current.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""The current version of every QA (one place for consistency_check, cons_fix and final_writeback):
L0 (stage_L0) <- accepted repair of review_log.json (L1: l1_results / L2: l2_results mapped onto the original keys /
L1E: l1e_results) <- accepted consistency fix (cons_results.json + review_cons.json) <- accepted post repairs (POST, in
this order; review_post.json {qid: {level: {verdict, notes}}}): L1T (type-free repair of the not-repaired QAs), FIX39
(deterministic fixes of the final-recheck problems), FIX39W (type-free repair of the final-recheck value mismatches), FIX3 (fixes of the third full recheck),
CONS2 (consistency fix round 2 on the post-repaired version, cons_fix.py fix2), FIX4 (fixes of the deep recheck, fix4.py), FIX5 (LLM repairs of the comprehensive recheck, fix5.py), FIX6 (fix5.py repairs of the video-audit claims), FIX7 (manual fixes, fix7.py), FIX8 (GT format fixes for the ACC scorer, fix8.py)).
pending=True also takes ok repairs never reviewed / consistency fixes not rejected — for checks only, never for the
writeback.
current(pending, skip) -> {qid: (qa, levels)}   levels e.g. ['L0', 'L1E', 'CONS']; skip: POST levels not applied"""
import copy, json, os
from fixlib import WORK, load, iter_qas, get_steps

L2_KEYS = ('question', 'annotated_solution', 'sources', 'answer', 'answer_range', 'physics_category',
           'required_external_tools')
RES_FILE = {'L1': 'l1_results.json', 'L2': 'l2_results.json', 'L1E': 'l1e_results.json'}
POST = (('L1T', 'l1t_results.json'), ('FIX39', 'fix39_results.json'), ('FIX39W', 'fix39w_results.json'),
        ('FIX3', 'fix3_results.json'), ('CONS2', 'cons2_results.json'), ('FIX4', 'fix4_results.json'),
        ('FIX5', 'fix5_results.json'), ('FIX6', 'fix6_results.json'), ('FIX7', 'fix7_results.json'),
        ('FIX8', 'fix8_results.json'))


def _j(name):
    p = '%s/%s' % (WORK, name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def from_l2(old, gen, cls):
    new = dict(old)
    for k in L2_KEYS:
        if gen.get(k) is not None or k == 'answer_range':
            new[k] = gen.get(k)
    if cls == 'b':  # A-class hidden_cv_chosen are CV values of the video: kept
        new['hidden_cv_chosen'] = gen.get('hidden_cv_chosen') or old.get('hidden_cv_chosen')
    new['total_steps'] = len(get_steps(new)[0])
    return new


def repaired(qid, old, lv, res):
    qa = res[lv][qid]['qa']
    return from_l2(old, qa, qid.split(':')[-1][0].lower()) if lv == 'L2' else copy.deepcopy(qa)


def current(pending=False, skip=()):
    review, rcons, rpost = _j('review_log.json'), _j('review_cons.json'), _j('review_post.json')
    post = [(lv, _j(f)) for lv, f in POST if lv not in skip]
    res = {lv: _j(f) for lv, f in RES_FILE.items()}
    cons = _j('cons_results.json')
    out = {}
    for split in ('train', 'test'):
        for cls in 'ab':
            for qid, vid, qa in iter_qas(load(cls, split, 'L0'), cls, split):
                levels = ['L0']
                r = review.get(qid) or {}
                if r.get('verdict') == 'accept':
                    qa = repaired(qid, qa, r['level'], res)
                    levels.append(r['level'])
                elif pending and not r:  # never reviewed (a rejected QA keeps L0)
                    for lv in ('L1E', 'L2', 'L1'):
                        x = res[lv].get(qid)
                        if x and x['status'] == 'ok':
                            qa = repaired(qid, qa, lv, res)
                            levels.append(lv + '(pending)')
                            break
                c, rc = cons.get(qid), rcons.get(qid) or {}
                if c and c['status'] == 'ok' and (rc.get('verdict') == 'accept' or (pending and rc.get('verdict') != 'reject')):
                    qa = copy.deepcopy(c['qa'])
                    levels.append('CONS' if rc.get('verdict') == 'accept' else 'CONS(pending)')
                for lv, res_lv in post:
                    x, v = res_lv.get(qid), ((rpost.get(qid) or {}).get(lv) or {}).get('verdict')
                    if x and x['status'] == 'ok' and (v == 'accept' or (pending and v != 'reject')):
                        new = copy.deepcopy(x['qa'])
                        if qa.get('hidden_cv_chosen') is None and new.get('hidden_cv_chosen') == []:
                            new['hidden_cv_chosen'] = None      # the rewrite returns [] for an absent hidden list
                        qa = new
                        levels.append(lv if v == 'accept' else lv + '(pending)')
                out[qid] = (qa, levels)
    return out
