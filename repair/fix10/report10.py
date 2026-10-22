# Provenance (TriPhysBench release): fix_code/evaluation/fix10/report10.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""final fix10 report (user 10/12): a needs_fix2 QA stays only if its last repaired version passed the strict gpt-5.6 audit;
the others are unresolved and deleted. Prints the unresolved count and the dataset distribution after the deletion.
Usage: python report10.py audit.json all255.json rec1.json [rec2.json ...]   (later record files override earlier ones)"""
import sys, os, json
from collections import Counter
ARGV = list(sys.argv)
sys.argv = [sys.argv[0], 'unres151_ids.json']
import fix10 as X  # noqa: E402,F401
from fix10 import R, F9  # noqa: E402

aud = json.load(open(ARGV[1]))
need = set(k for k in json.load(open(ARGV[2])) if '|' in k)  # the 255 needs_fix2 ids
SEEDS = 'seed_v16.json'  # QAs outside the 255 whose answer form mismatched the type (user 10/13 type check)
if os.path.exists(SEEDS):
    need |= set(json.load(open(SEEDS)))
recs = {}
for f in ARGV[2:]:
    for k, r in json.load(open(f)).items():
        if isinstance(r, dict) and r.get('status') in ('fixed', 'given') and r.get('qa'):
            recs[k] = r
from audit10 import sig  # noqa: E402
items = R.load()
keep, unres, how = [], [], Counter()
for it in items:
    k = it['id']
    if k not in need:
        keep.append((it, it['qa']))
        continue
    r = recs.get(k)
    if r and aud.get(k, {}).get('verdict') == 'pass' and aud[k].get('sig') == sig(r['qa']):
        keep.append((it, r['qa']))
        how[(it['cls'], 'repair' if r.get('repair') else r['status'])] += 1
    else:
        unres.append(k)
print('needs_fix2 %d: kept %d, UNRESOLVED (deleted) %d' % (len(need), len(need) - len(unres), len(unres)))
print('kept by route:', dict(sorted(how.items())))
print('unresolved by class/split:', dict(sorted(Counter(tuple(k.split('|')[:2]) for k in unres).items())))
print('unresolved reasons:', dict(Counter('no repaired version' if k not in recs else
                                          'audit %s' % aud.get(k, {}).get('verdict', 'missing') for k in unres)))
print('\nDATASET before %d -> after %d' % (len(items), len(keep)))
for name, f in [('class', lambda i, q: i['cls']), ('class/split', lambda i, q: (i['cls'], i['split'])),
                ('class/difficulty', lambda i, q: (i['cls'], q.get('difficulty'))),
                ('class/type', lambda i, q: (i['cls'], q.get('question_type'))),
                ('split', lambda i, q: i['split'])]:
    b = Counter(f(i, i['qa']) for i in items)
    a = Counter(f(i, q) for i, q in keep)
    print('%-17s' % name, ', '.join('%s %d->%d' % ('/'.join(x) if isinstance(x, tuple) else x, b[x], a[x]) for x in sorted(b)))
vids = Counter((i['cls'], i['split'], i['video'].get('take_uid') or i['video'].get('folder_id'), i['video'].get('video_id'))
               for i, q in keep)
allv = Counter((i['cls'], i['split'], i['video'].get('take_uid') or i['video'].get('folder_id'), i['video'].get('video_id'))
               for i in items)
print('videos with >=1 QA: %d -> %d' % (len(allv), len(vids)))
dup = Counter()
for v in vids:
    ds = [q.get('difficulty') for i, q in keep if (i['cls'], i['split'], i['video'].get('take_uid') or i['video'].get('folder_id'),
                                                   i['video'].get('video_id')) == v]
    dup['videos with a repeated difficulty'] += len(ds) != len(set(ds))
print(dict(dup))
json.dump(unres, open('unresolved_final.json', 'w'), indent=0)
