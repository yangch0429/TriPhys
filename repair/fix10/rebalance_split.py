# Provenance (TriPhysBench release): fix_code/evaluation/fix10/rebalance_split.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 split re-balance (10/13, user: "有严格的层次要求 ... 思考是否需要重新划分"; leak rule: the test set must not
hold a QA the trained models saw). fix10 broke the B strata: B test hard 14 -> 6 QAs (5 deleted, 1 moved with
a00588/00436, 2 changed difficulty), TYPE_B 16 -> 11. A stayed as balanced as the original split (not moved: every
eligible A train video shares >= 3 hidden values with a QA the models trained on).
Original split (paper_material/dataset_processing_tool.stratified_split_by_video): video level, strata = per-video mode of
(TYPE, Difficulty, Physics main), rare (< 3 videos) merged, 7:3, seed 42. The mode breaks ties by set order (hash seed),
so it does not reproduce (5-13 of ~30 test videos shared); the balance is scored here hash-free.
Objective (lower = better), QA level vs the whole class: sum over the TYPE, Difficulty, Physics-main and TYPE x Difficulty
cells of |test_c - 0.27 all_c| (0.27 = the original QA test share), + |test videos - original count|, + the stratum quota
deviation of the original code averaged over 20 random tie-breaks, + 1 per moved video. Greedy over single moves and swaps.
Moves: test -> train any; train -> test only for videos that are not keep_orig, none of whose QAs is identical to a QA of
a training split (versions 01 / 02 / 03) and none of whose hidden_cv_chosen values appears in such a QA.
Input: the split files after fix10_apply without the re-balance ({a,b}class_qa_{train,test}.before_rebalance.json).
Writes final/rebalance_<cls>.json and adds the moves to final/final_ids.json (to_train / to_test, applied by
fix10_apply.py / f10_data.py after the id edits).  Usage: python rebalance_split.py b"""
import json, sys, os, random
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.environ.get('TRIPHYS_WORK_DIR') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.environ.get('DATASET_TOOL_DIR') or os.path.abspath(os.path.join(FIX, '..')))
import logging; logging.disable(logging.CRITICAL)
from dataset_processing_tool import merge_rare_strata, extract_physics_main_category as pm  # noqa: E402
QV = FIX + '/question_versions'
L = lambda p: json.load(open(p, encoding='utf-8'))
C = sys.argv[1]
fin_p = os.path.join(HERE, 'final', 'final_ids.json')
fin = L(fin_p)
keep_orig = set()
vid = (lambda v: v['take_uid']) if C == 'a' else (lambda v: '%s/%s' % (v['folder_id'], v['video_id']))
tr = L('%s/%sclass_qa_train.before_rebalance.json' % (FIX, C))
te = L('%s/%sclass_qa_test.before_rebalance.json' % (FIX, C))
V = {vid(v): v for v in tr + te}
for k in fin['keep_orig']:
    cls, sp, vi, qi = k.split('|')
    if cls.lower() == C:
        q = L('%s/07_fix8_1011/%sclass_qa_%s.json' % (QV, C, sp))[int(vi)]['qa_pairs'][int(qi)]
        keep_orig |= {x for x in V if any(p['question'] == q['question'] for p in V[x]['qa_pairs'])}
test0 = {vid(v) for v in te}
NV = len({vid(v) for v in L('%s/01_cleaned_split_0929/%sclass_qa_test.json' % (QV, C))})
seenq, seenh = defaultdict(set), defaultdict(set)
for ver in ('01_cleaned_split_0929', '02_L0_0930', '03_v9_repaired_1002'):
    for v in L('%s/%s/%sclass_qa_train.json' % (QV, ver, C)):
        for q in v['qa_pairs']:
            seenq[vid(v)].add(q['question'].strip())
            seenh[vid(v)] |= {str(h).strip() for h in q.get('hidden_cv_chosen') or []}
nid = {k: sum(q['question'].strip() in seenq.get(k, ()) for q in V[k]['qa_pairs']) for k in V}
nh = {k: sum(str(h).strip() in seenh.get(k, ()) for q in V[k]['qa_pairs'] for h in q.get('hidden_cv_chosen') or []) for k in V}
elig = {k for k in V if k not in test0 and k not in keep_orig and nid[k] == 0 and nh[k] == 0}
assert not any(nid[k] for k in test0)
qs = [(k, q['question_type'], q['difficulty'], pm(q.get('physics_category', ''))) for k in V for q in V[k]['qa_pairs']]
dims = [lambda x: x[1], lambda x: x[2], lambda x: x[3], lambda x: (x[1], x[2])]
A = [Counter(f(x) for x in qs) for f in dims]
TB = []
for r in range(20):
    rnd = random.Random(r)
    def mode(xs):
        c = Counter(xs); m = max(c.values()); return rnd.choice(sorted(x for x in c if c[x] == m))
    g = defaultdict(list)
    for k in sorted(V):
        qq = V[k]['qa_pairs']
        g[(mode([q['question_type'] for q in qq]), mode([q['difficulty'] for q in qq]),
           mode([pm(q.get('physics_category', '')) for q in qq]))].append(k)
    it = list(g.items()); rnd.shuffle(it)
    TB.append([(vs, min(max(1, int(len(vs) * .3)), len(vs) - 1)) for vs in merge_rare_strata(dict(it), 3).values()])
def parts(T):
    tq = [x for x in qs if x[0] in T]
    qa = sum(sum(abs(Counter(f(x) for x in tq)[c] - .27 * A[i][c]) for c in A[i]) for i, f in enumerate(dims))
    st = sum(sum(abs(sum(k in T for k in vs) - q) for vs, q in tb) for tb in TB) / len(TB)
    return qa, st, abs(len(T) - NV), len(T ^ test0)
score = lambda T: sum(parts(T))
T, best = set(test0), score(test0)
while True:
    cands = [T - {k} for k in T] + [T | {k} for k in elig - T] + [(T - {a}) | {b} for a in T for b in elig - T]
    s, nT = min(((score(x), sorted(x)) for x in cands), key=lambda z: z[0])
    if s >= best - 1e-9:
        break
    best, T = s, set(nT)
rep = ['%s: eligible train -> test %d of %d train videos; keep_orig %s' % (C.upper(), len(elig), len(V) - len(test0), sorted(keep_orig)),
       'before: QA-dev %.1f stratum-dev %.1f |videos-%d| %d moves %d' % ((*parts(test0)[:2], NV) + tuple(parts(test0)[2:])),
       'after : QA-dev %.1f stratum-dev %.1f |videos-%d| %d moves %d' % ((*parts(T)[:2], NV) + tuple(parts(T)[2:]))]
for k in sorted(T ^ test0):
    rep.append('  %s %s %s' % ('train -> test' if k in T else 'test -> train', k,
                               [(q['question_type'], q['difficulty']) for q in V[k]['qa_pairs']]))
for name, i in (('type', 0), ('diff', 1), ('phys', 2)):
    for lab, S in (('before', test0), ('after ', T)):
        t = Counter(dims[i](x) for x in qs if x[0] in S)
        rep.append('  %s %s %s' % (name, lab, {c: '%d/%d' % (t[c], A[i][c]) for c in sorted(A[i])}))
rep.append('test %d videos / %d QAs; train %d / %d' % (len(T), sum(1 for x in qs if x[0] in T), len(V) - len(T),
                                                      sum(1 for x in qs if x[0] not in T)))
print('\n'.join(rep))
mv = {'to_test': sorted(T - test0), 'to_train': sorted(test0 - T)}
json.dump(dict(mv, report=rep), open(os.path.join(HERE, 'final', 'rebalance_%s.json' % C), 'w'), indent=1)
fin['to_train'] = sorted(set(fin.get('to_train', [])) | set(mv['to_train']))
fin['to_test'] = sorted(set(fin.get('to_test', [])) | set(mv['to_test']))
json.dump(fin, open(fin_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
