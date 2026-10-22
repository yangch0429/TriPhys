# Provenance (TriPhysBench release): fix_code/evaluation/fix10/select_final.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 final selection (user 10/13: "先不删…修复 lost videos (8 QAs)，然后多加几个QA，将总题目数控制在410个，200个视频";
"多加的QA从之前未修复好的QA中找"). Keep rule = report10 / build_fix10_subset (strict gpt-5.6 audit pass, sig match).
Every kept non-extra QA stays; extras (ids_extra.json: the redesigned earlier-unresolved QAs) are added greedily until
TARGET QAs, each time the one whose (class, type, difficulty) cell is furthest below its v9 share of TARGET (tie: the
video with fewer QAs). Every other needs_fix2 QA is deleted. Checks: every video keeps >= 1 QA.
-> final/final_ids.json {'replace': {id: qa}, 'delete': [id]}, final/select_report.txt
Usage: python select_final.py audit.json all255.json rec1.json [rec2.json ...]"""
import sys, os, json
from collections import Counter
ARGV = list(sys.argv)
sys.argv = [sys.argv[0], 'unres151_ids.json']
import fix10 as X  # noqa: E402,F401
from fix10 import R  # noqa: E402
from audit10 import sig  # noqa: E402

TARGET = int(os.environ.get('FIX10_TARGET', '410'))
aud = json.load(open(ARGV[1]))
need = set(k for k in json.load(open(ARGV[2])) if '|' in k)
if os.path.exists('seed_v16.json'):
    need |= set(json.load(open('seed_v16.json')))
extra = set(json.load(open('ids_extra.json')))
# user 10/13: the 4 videos whose QAs failed every redesign keep their v9 QA (a00106/00861 keeps 1 of its 2)
orig = set(json.load(open('keep_orig.json'))) if os.path.exists('keep_orig.json') else set()
recs = {}
for f in ARGV[2:]:
    for k, r in json.load(open(f)).items():
        if isinstance(r, dict) and r.get('status') in ('fixed', 'given') and r.get('qa'):
            recs[k] = r
items = R.load()
byid = {it['id']: it for it in items}
# unique video id: A take_uid, B folder_id/video_id
vid = lambda it: it['video'].get('take_uid') or '%s/%s' % (it['video'].get('folder_id'), it['video'].get('video_id'))
cell = lambda it, q: (it['cls'], q.get('question_type'), q.get('difficulty'))
kept = {}
for k in need:
    r = recs.get(k)
    if r and aud.get(k, {}).get('verdict') == 'pass' and aud[k].get('sig') == sig(r['qa']):
        kept[k] = r['qa']
base = {k for k in kept if k not in extra}
cand = {k for k in kept if k in extra}
assert not orig & set(kept), orig & set(kept)
final = {it['id'] for it in items if it['id'] not in need} | base | orig
v9 = Counter(cell(it, it['qa']) for it in items)
cur = Counter(cell(byid[k], kept.get(k, byid[k]['qa'])) for k in final)
nv = Counter(vid(byid[k]) for k in final)
out = []
L = ['v9 %d QAs / %d videos; needs_fix2 %d; kept %d (base %d, extra candidates %d of %d)' % (
    len(items), len({vid(it) for it in items}), len(need), len(kept), len(base), len(cand), len(extra)),
     'kept in the v9 version (keep_orig.json): %s' % ['%s (%s)' % (k, vid(byid[k])) for k in sorted(orig)],
     'before extras: %d QAs / %d videos' % (len(final), len(nv))]
while len(final) < TARGET and cand:
    def score(k):
        c = cell(byid[k], kept[k])
        return (v9[c] * TARGET / len(items) - cur[c], -nv[vid(byid[k])], k)
    k = max(cand, key=score)
    cand.discard(k)
    final.add(k)
    out.append(k)
    cur[cell(byid[k], kept[k])] += 1
    nv[vid(byid[k])] += 1
lost = sorted({vid(it) for it in items} - set(nv))
L += ['extras added %d: %s' % (len(out), ['%s (%s)' % (k, vid(byid[k])) for k in out]), 'final %d QAs / %d videos; videos without QAs: %s' % (len(final), len(nv), lost)]
L.append('cell  v9 -> final (v9 share of %d)' % TARGET)
for c in sorted(v9, key=str):
    L.append('  %s  %d -> %d (%.1f)' % (c, v9[c], cur[c], v9[c] * TARGET / len(items)))
# user 10/13: the kept-v9 QAs are not repaired -> their test videos move to train (test -> train only: no model has
# trained on a test video, so nothing leaks; every other video keeps its split)
to_train = sorted({vid(byid[k]) for k in orig if byid[k]['split'] == 'test'})
mv = lambda k: byid[k]['split'] == 'test' and vid(byid[k]) in to_train
sp = lambda k: 'train' if mv(k) else byid[k]['split']
L.append('test videos moved to train: %s (%d QAs)' % (to_train, sum(mv(k) for k in final)))
for s in ('train', 'test'):
    L.append('%s: ' % s + ', '.join('%s %d -> %d QAs / %d videos' % (
        c, sum(1 for it in items if it['cls'] == c and it['split'] == s),
        sum(1 for k in final if k[0] == c and sp(k) == s), len({vid(byid[k]) for k in final if k[0] == c and sp(k) == s}))
        for c in 'AB'))
os.makedirs('final', exist_ok=True)
json.dump({'replace': {k: kept[k] for k in final if k in kept}, 'delete': sorted(need - final), 'keep_orig': sorted(orig),
           'to_train': to_train},
          open('final/final_ids.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open('final/select_report.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
