# Provenance (TriPhysBench release): fix_code/evaluation/fix10/given_convert.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""stated-given conversion of the QAs the search repair (v13 + v15) left unresolved (user 10/12 decisions 1/4):
a web constant no search states uniquely becomes a value the question states; code and answer stay unchanged.
  * A: difficulty = #hidden readings -> unchanged; at least one web constant must stay (the QA still needs a search);
  * B: difficulty = #web constants (1/2/3 -> easy/medium/hard); the new difficulty is allowed only if no other QA of the
    same video has it (hard -> easy included); slots are claimed per video in id order;
  * B hidden_cv_chosen follows the web values: the entry of a stated constant is dropped.
The failing constants come from the last record's stop / history ("web constant(s) [j]"), else anal_const_final.txt.
Every converted QA is then audited as a whole by audit10.py (gpt-5.6-terra).
Usage: python given_convert.py out.json rec1.json [rec2.json ...]   (later files override earlier ones)"""
import sys, json, os, re, ast, asyncio, copy
from collections import Counter, defaultdict
ARGV = list(sys.argv)
sys.argv = [sys.argv[0], 'unres151_ids.json']
import fix10 as X  # noqa: E402
from fix10 import R, W, F9, AT  # noqa: E402

LV = {1: 'easy', 2: 'medium', 3: 'hard'}
FAILRE = re.compile(r'web constant\(s\) \[([\d, ]+)\]')

GIVEN = """A physics video QA needs web constant(s) that no web search states uniquely. Turn exactly these constants into stated
givens: the question states each value as a given input, so the solver no longer searches for it. The code and the answer
stay unchanged.
QUESTION: %s
ANNOTATED SOLUTION:
%s
SOLUTION CODE:
```python
%s
```
WEB CONSTANTS TO STATE (index | search query | key information):
%s
OTHER WEB CONSTANTS (the solver still searches them - never state their value): %s
HIDDEN VALUES (never state them): %s
%sRules:
- For each constant to state, find the code variable that holds it and its value in the code; state exactly that value with
  its unit in one short natural clause or sentence ("Take the density of the brass as 8500 kg/m^3." / "the ball, whose mass
  is 0.62 kg, ..."). Use the code's number (you may convert the unit only if the code converts it the same way).
- A phrase that tells the solver to look this constant up ("using the typical marble mass", "using standard basketball
  specifications" when that is the only constant it covers) now refers to the stated value: drop or reword it so the
  question does not ask for a value it states. Keep the cue phrases of the other web constants.
- Change nothing else in the question: keep every number, the scene description and the asked quantity.
- Never state another web constant's value or a hidden value, never name a tool or searching, never add a method hint.
- Annotated solution: replace the web_search step of each stated constant by a step that reads the value from the question;
  keep every other step.
Return ONLY JSON: {"question": "...", "annotated_solution": ["..."], "stated": [{"index": j, "var": "...", "value": number, "unit": "..."}]}"""


def near(x, y, tol=0.005):
    return abs(x - y) <= tol * max(abs(x), abs(y), 1e-12)


def drop_hidden(hid, dropped, vals):
    """B hidden_cv_chosen mirrors the web values with other wording ("loaded haul bag mass = 99.8 kg" for "... 220 pounds
    (99.8 kg)"): drop the entries whose non-trivial numbers are all numbers of a dropped constant / a stated value"""
    pool = [x for d in dropped for x in AT.nums(str(d.get('key_information') or '')) + AT.nums(str(d.get('verified_quote') or ''))]
    pool += [float(v) for v in vals]
    keep = []
    for h in hid:
        hn = [x for x in AT.nums(str(h)) if not (float(x).is_integer() and abs(x) <= 10)] or AT.nums(str(h))
        if hn and all(any(near(x, p, 0.01) for p in pool) for x in hn):
            continue
        keep.append(h)
    return keep


def failed_consts(rec, anal):
    for t in [rec.get('stop') or ''] + list(reversed(rec.get('history') or [])):
        m = FAILRE.search(str(t))
        if m:
            return sorted({int(x) for x in m.group(1).split(',') if x.strip()})
    return sorted(anal.get(rec['id'], []))


def check(new, old, ws, todo, stated, hid, code):
    P = []
    by = {s.get('index'): s for s in stated if isinstance(s, dict)}
    qn, on = AT.nums(new), AT.nums(old)
    cn = AT.nums(code)
    for j in todo:
        s = by.get(j)
        if not s or not isinstance(s.get('value'), (int, float)):
            P.append('constant [%d] is not in "stated" with a numeric value' % j)
            continue
        v = float(s['value'])
        if not any(near(v, c) for c in cn):
            P.append('the stated value %s of constant [%d] is not a value of the code' % (s['value'], j))
        if not any(near(v, q) for q in qn):
            P.append('the question does not state the value %s of constant [%d]' % (s['value'], j))
    lost = [x for x in on if not any(near(x, y, 1e-9) for y in qn)]
    if lost:
        P.append('the question lost the number(s) %s of the old question' % sorted(set(lost)))
    keep = [j for j in range(len(ws)) if j not in todo]
    sv = [float(s['value']) for s in by.values() if isinstance(s.get('value'), (int, float))]
    # numbers of the kept web constants / the hidden values the new question states (not the stated values themselves,
    # not the hidden entry of a stated constant: B hidden_cv_chosen mirrors the web values)
    hid = drop_hidden(hid, [ws[j] for j in todo], sv)
    bad = lambda x: (not (float(x).is_integer() and abs(x) <= 10) and any(near(x, q) for q in qn)
                     and not any(near(x, o) for o in on) and not any(near(x, v) for v in sv))
    leak = [x for j in keep for x in AT.nums(ws[j].get('key_information')) if bad(x)]
    leak += [x for h in hid for x in AT.nums(h) if bad(x)]
    if leak:
        P.append('the question now states the value(s) %s of a web constant the solver must still search / a hidden value' % sorted(set(leak)))
    if re.search(r'\b(search|google|web|internet|look(?:ed)? up|code|python)\b', new, re.I) and not \
            re.search(r'\b(search|google|web|internet|look(?:ed)? up|code|python)\b', old, re.I):
        P.append('the question names a tool or searching')
    return P


async def convert(item, todo):
    qa = copy.deepcopy(item['qa'])
    if isinstance(qa.get('sources'), str):
        qa['sources'] = ast.literal_eval(qa['sources'])
    hid = qa.get('hidden_cv_chosen') or []
    hid = ast.literal_eval(hid) if isinstance(hid, str) else list(hid)
    ws = F9.wsrc(qa)
    code = F9.code_of(qa)
    tab = '\n'.join('[%d] %s | %s' % (j, ws[j].get('search_query'), ws[j].get('key_information')) for j in todo)
    oth = '; '.join('[%d] %s' % (j, ws[j].get('key_information')) for j in range(len(ws)) if j not in todo) or '(none)'
    fb = ''
    for _ in range(4):
        g = W.pj(await X.mllm(GIVEN % (qa['question'], '\n'.join(qa.get('annotated_solution') or []), code[:6000], tab, oth,
                                       hid or '(none)', fb))) or {}
        if not g.get('question') or not isinstance(g.get('annotated_solution'), list):
            fb = 'YOUR LAST ANSWER WAS NOT VALID JSON WITH question / annotated_solution / stated.\n'
            continue
        P = check(g['question'], qa['question'], ws, todo, g.get('stated') or [], hid, code)
        if P:
            fb = 'YOUR LAST REWRITE "%s" WAS REJECTED: %s. Fix this.\n' % (g['question'], '; '.join(P))
            continue
        new = copy.deepcopy(qa)
        new['question'] = g['question']
        new['annotated_solution'] = g['annotated_solution']
        drop = [ws[j] for j in todo]
        new['sources'] = [s for s in qa['sources'] if not any(s is d for d in drop)]
        if item['cls'] == 'B':
            new['hidden_cv_chosen'] = drop_hidden(hid, drop, [s['value'] for s in g.get('stated') or []
                                                              if isinstance(s, dict) and isinstance(s.get('value'), (int, float))])
        return new, g.get('stated'), None
    return None, None, fb.strip()


async def main():
    W.SEM, W.SSEM, X.MSEM = asyncio.Semaphore(32), asyncio.Semaphore(16), asyncio.Semaphore(24)
    out_f = ARGV[1]
    recs = {}
    for f in ARGV[2:]:
        for k, r in json.load(open(f)).items():
            recs[k] = r
    anal = defaultdict(set)
    for ln in open('anal_const_final.txt', encoding='utf-8'):
        p = ln.rstrip('\n').split('|')
        if len(p) > 6 and p[6].strip().isdigit():
            anal['|'.join(p[:4])].add(int(p[6]))
    items = {i['id']: i for i in R.load()}
    diff = {k: i['qa'].get('difficulty') for k, i in items.items()}
    vid = lambda k: tuple(k.split('|')[:3])
    sibs = defaultdict(list)
    for k in items:
        sibs[vid(k)].append(k)
    for k, r in recs.items():  # an earlier conversion (given_v15.json as a record file) keeps its claimed slot
        if r.get('status') == 'given' and r.get('qa'):
            diff[k] = r['qa'].get('difficulty')
    unres = sorted((k for k, r in recs.items() if r.get('status') not in ('fixed', 'given')),
                   key=lambda k: (k.split('|')[0], k.split('|')[1], int(k.split('|')[2]), int(k.split('|')[3])))
    out, plan = {}, []
    for k in unres:  # slot claims in id order (sequential: a claimed difficulty blocks the video's later QAs)
        it = items[k]
        qa = it['qa']
        srcs = ast.literal_eval(qa['sources']) if isinstance(qa['sources'], str) else qa['sources']
        nws = sum(isinstance(s, dict) and s.get('type') == 'web_search' for s in srcs)
        todo = [j for j in failed_consts(recs[k], anal) if j < nws]
        base = dict(id=k, cls=it['cls'], split=it['split'], old_difficulty=qa.get('difficulty'), failed=todo)
        if not todo:
            out[k] = dict(base, status='unresolved', reason='no failing web constant identified (code / check failure)')
            continue
        left = nws - len(todo)
        if left < 1:
            out[k] = dict(base, status='unresolved', reason='no web constant left (#ws %d, failing %s)' % (nws, todo))
            continue
        nd = qa.get('difficulty') if it['cls'] == 'A' else LV.get(left, 'hard')
        other = [diff[s] for s in sibs[vid(k)] if s != k]
        if nd != qa.get('difficulty') and nd in other:
            out[k] = dict(base, status='unresolved', reason='B %s -> %s but the video already has %s' % (qa.get('difficulty'), nd, nd))
            continue
        diff[k] = nd
        plan.append((k, todo, nd))
    print('convert %d, unresolved without conversion %d' % (len(plan), len(out)), flush=True)

    async def one(k, todo, nd):
        new, stated, err = await convert(items[k], todo)
        base = dict(id=k, cls=items[k]['cls'], split=items[k]['split'], old_difficulty=items[k]['qa'].get('difficulty'),
                    failed=todo)
        if new is None:
            diff[k] = items[k]['qa'].get('difficulty')  # (the slot stays claimed: conservative, the QA is unresolved anyway)
            out[k] = dict(base, status='unresolved', reason='conversion failed: ' + str(err)[:500])
            return
        new['difficulty'] = nd
        out[k] = dict(base, status='given', new_difficulty=nd, stated=stated, qa=new)
    await asyncio.gather(*[one(*p) for p in plan])
    json.dump(out, open(out_f, 'w'), ensure_ascii=False, indent=1)
    C = Counter((r['cls'], r['status'], (r.get('reason') or '').split(' (')[0].split(':')[0][:40]) for r in out.values())
    for x in sorted(C):
        print(x, C[x])
    print('difficulty changes:', Counter((r['old_difficulty'], r['new_difficulty']) for r in out.values()
                                          if r['status'] == 'given' and r['old_difficulty'] != r['new_difficulty']))

if __name__ == '__main__':
    asyncio.run(main())
