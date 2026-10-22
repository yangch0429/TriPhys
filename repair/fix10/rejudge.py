# Provenance (TriPhysBench release): fix_code/evaluation/fix10/rejudge.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 step 1c (10/12, user): criteria v2 and the final needs_fix list.
User 10/12:
  - W_query_added: ignored (the solver also has the video).
  - W_base_conflict: re-judged WITH the video. The 10/11 VALID judge saw only the question. Here deepseek-flash also sees
    frames and decides whether question + video single out the reference value. A conflict remains only if it says
    "alternative" or "both".
  - M_both_valid: ignored. M_alt_more_standard: fixed (the GT switches to the standard model, or an assumption is stated).
  - V_invented: "typical / standard value" claims are fine if the web value is unique (W covers that). Only claims of a
    specific identity that the frames do not support (brand, league, model, grade, configuration, material) count.
  - V_contradicted / V_absent: counted, after the dense re-check (recheck_ns.py).
Input recount_final.json -> output recount_v2.json (labels2 / needs_fix2 added). The functions are reused by fix10
acceptance."""
import os, json, ast, asyncio
from collections import Counter, defaultdict
import recount as R
from recount import W
import fix10 as F
import recheck_ns as RN

HERE = R.HERE
RSEM = None
COUNT = ('W_base_unreach', 'W_no_base_query', 'W_base_conflict', 'M_alt_more_standard', 'V_contradicted', 'V_absent',
         'V_invented', 'C_inconsistent')

VALIDV = """A physics benchmark question is about the video whose frames are attached. The solver watches the video, reads
the question, and looks up "%s" on the web.
QUESTION: %s
Objects an observer noted in the video: %s

The solver's search returned two values for this quantity. Neither value nor quote is part of the question.
Value 1: %s %s (web quote: "%s")
Value 2: %s %s (web quote: "%s")
Do the QUESTION text together with what the VIDEO shows (the object, its size class, material, setting, condition) make one
value fit and the other not? Reply "1" or "2" if question + video point to that value only. Reply "both" if a careful
solver who watches the video and reads the question could reasonably use either value.
Return ONLY JSON: {"verdict": "1|2|both", "why": "..."}"""

INVKIND = """A physics benchmark question about a video makes these statements, which an auditor found unsupported by the
frames:
%s
Classify each statement:
- "typical": it only says that a typical / standard / average / common value or specification of an object or material
  applies (e.g. "typical pencil specifications apply", "a standard soda can", "the marble is glass"), or it is a general
  material class a careful viewer would accept.
- "identity": it gives a scene object a specific identity, configuration or property that the frames do not show and that
  changes which value applies (a brand, league, professional status, product model, grade, ball number, mechanical
  advantage, a named drill or technique, a specific liquid, a specific bearing type).
Return ONLY JSON: {"items": [{"claim": "...", "kind": "typical|identity"}]}"""


async def rejudge_conflicts(r, item, fr, fkey):
    """re-judge every W conflict of r with the video; sets c['conflicts_video'] per constant"""
    qa = item['qa']
    srcs = ast.literal_eval(qa['sources']) if isinstance(qa['sources'], str) else (qa['sources'] or [])
    srcs = [s for s in srcs if isinstance(s, dict) and s.get('type') == 'web_search']
    objs = '; '.join('%s (%s)' % (o.get('object'), o.get('seen')) for o in (r.get('video') or {}).get('objects') or []
                     if isinstance(o, dict)) or 'n/a'

    async def one(i, c, x):
        vq = srcs[i].get('verified_quote', '') if i < len(srcs) else ''
        o1, o2 = (c.get('ref'), '', vq), (x.get('value'), x.get('unit') or '', x.get('quote') or '')
        flip = W.coin(qa['question'], x.get('quote'), 'v')
        j = W.pj(await F.mllm(VALIDV % ((c.get('quantity'), qa['question'], objs) + (o2 + o1 if flip else o1 + o2)), fr, fkey)) or {}
        x['video_verdict'] = {'1': 'alternative' if flip else 'reference', '2': 'reference' if flip else 'alternative',
                              'both': 'both'}.get(str(j.get('verdict')))
        x['video_why'] = j.get('why')
    cs = r.get('consts') or []
    await asyncio.gather(*[one(i, c, x) for i, c in enumerate(cs) for x in c.get('conflicts') or []])
    for c in cs:
        c['conflicts_video'] = [x for x in c.get('conflicts') or [] if x.get('video_verdict') != 'reference']


async def invented_identity(r):
    v = r.get('video') or {}
    cl = v.get('invented', []) + [s for s in (v.get('recheck') or {}).get('bad', []) if s.get('label') == 'invented']
    if not cl:
        return []
    j = W.pj(await W.llm(INVKIND % '\n'.join('- ' + str(c.get('claim')) for c in cl))) or {}
    kinds = {str(x.get('claim')): x.get('kind') for x in j.get('items') or [] if isinstance(x, dict)}
    return [c for c in cl if kinds.get(str(c.get('claim')), 'identity') == 'identity']


def labels2(r):
    L = set(r['labels'])
    out = {l for l in L if l in COUNT or l.startswith('S_')}
    if 'W_base_conflict' in out and not any(c.get('conflicts_video') for c in r.get('consts') or []):
        out.discard('W_base_conflict')
    if 'V_invented' in out and not r['video'].get('invented_identity'):
        out.discard('V_invented')
    return sorted(out)


async def judge(r, item):
    if 'W_base_conflict' in r['labels']:
        fr, _ = await asyncio.to_thread(R.frames, R.video_path(item), r['question'])
        fkey = '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (fr or [])))
        await rejudge_conflicts(r, item, fr, fkey)
    if 'V_invented' in r['labels']:
        r['video']['invented_identity'] = await invented_identity(r)
    r['labels2'] = labels2(r)
    v = r.get('video') or {}
    if not r['labels2'] and v.get('not_shown') and not v.get('recheck'):
        # clean under v2 but sparse-frame gaps remain -> dense re-check of the not_shown claims (as recheck_ns.py)
        o = await RN.recheck(r, item, RSEM, v['not_shown'])
        o['_v2'] = True
        v['recheck'] = o
        lab = Counter(s.get('label') for s in o.get('bad', []))
        add = ['V_absent'] * bool(lab['absent']) + ['V_contradicted'] * bool(lab['contradicted'])
        if lab['invented']:
            r['labels'] = sorted(set(r['labels'] + ['V_invented']))
            v['invented_identity'] = await invented_identity(r)
            add += ['V_invented'] * bool(v['invented_identity'])
        r['labels'] = sorted(set(r['labels'] + add))
        r['labels2'] = labels2(r)
    r['needs_fix2'] = bool(r['labels2'])


async def main():
    global RSEM
    RSEM = asyncio.Semaphore(16)
    W.SEM, W.SSEM, F.MSEM = asyncio.Semaphore(32), asyncio.Semaphore(4), asyncio.Semaphore(32)
    rs = json.load(open(HERE + '/recount_final.json'))
    items = {i['id']: i for i in R.load()}
    await asyncio.gather(*[judge(r, items[r['id']]) for r in rs])
    json.dump(rs, open(HERE + '/recount_v2.json', 'w'), ensure_ascii=False, indent=1)
    conf = [x for r in rs for c in r.get('consts') or [] for x in c.get('conflicts') or []]
    print('W conflicts re-judged with video:', Counter(x.get('video_verdict') for x in conf))
    print('W_base_conflict QAs: before %d after %d' % (sum('W_base_conflict' in r['labels'] for r in rs),
                                                        sum('W_base_conflict' in r['labels2'] for r in rs)))
    print('V_invented QAs: before %d identity %d' % (sum('V_invented' in r['labels'] for r in rs),
                                                     sum('V_invented' in r['labels2'] for r in rs)))
    G = defaultdict(list)
    for r in rs:
        G[(r['split'], r['cls'])].append(r)
    print('\n== needs_fix (criteria v2) ==')
    for g in sorted(G):
        X = G[g]
        print('%-14s n=%3d  needs_fix %3d  clean %3d' % (g, len(X), sum(r['needs_fix2'] for r in X), sum(not r['needs_fix2'] for r in X)))
    print('all needs_fix', sum(r['needs_fix2'] for r in rs))
    for k, v in sorted(Counter(l for r in rs for l in r['labels2']).items()):
        print('  %-22s %3d  test %3d  train %3d' % (k, v, sum(k in r['labels2'] for r in rs if r['split'] == 'test'),
                                                  sum(k in r['labels2'] for r in rs if r['split'] == 'train')))
    print('by difficulty', dict(sorted(Counter((r['split'], r['difficulty']) for r in rs if r['needs_fix2']).items())))
    print('by type', dict(sorted(Counter((r['split'], r['question_type']) for r in rs if r['needs_fix2']).items())))
    ns = [r for r in rs if ((r.get('video') or {}).get('recheck') or {}).get('_v2')]
    print('dense not_shown re-checks run here: %d, still V-bad %d' % (len(ns), sum(r['needs_fix2'] for r in ns)))
    for r in ns:
        for s in r['video']['recheck'].get('bad', []):
            print('  ', r['id'], s.get('label'), '|', str(s.get('claim'))[:90], '|', str(s.get('evidence'))[:120])
    print('stats', dict(W.STAT))


if __name__ == '__main__':
    asyncio.run(main())
