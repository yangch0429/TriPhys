# Provenance (TriPhysBench release): fix_code/evaluation/fix10/audit_fix.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""repair of the QAs the strict gpt-5.6 audit (audit10.py) fails (user 10/12: the remaining QAs must be solved, not deleted):
deepseek-flash sees the frames, the whole QA and the auditor's defects and returns the smallest fix; deterministic checks
(the code runs, the answer is its output, no leak, type / readings / area kept, the difficulty slot rule for a web constant
that becomes a stated given); the next audit10.py run re-audits the changed QAs. At most MAXR repairs per QA.
Usage: python audit_fix.py repair.json audit.json rec1.json [rec2.json ...]   (repair.json is also the last record file)"""
import sys, json, os, re, ast, asyncio, copy, hashlib
from collections import Counter, defaultdict
ARGV = list(sys.argv)
sys.argv = [sys.argv[0], 'unres151_ids.json']
import fix10 as X  # noqa: E402
from fix10 import R, W, F9, AT  # noqa: E402
import audit10 as AU  # noqa: E402  (TYPE_NOTE override, type-form checks, audit signature)
from given_convert import near, drop_hidden, LV  # noqa: E402

MAXR = int(os.environ.get('AUDIT_FIX_MAXR', '3'))
TOOLRE = re.compile(r'\b(search|google|internet|look(?:ed)? up|python|code)\b', re.I)

FIX = """You repair one QA of a physics video benchmark (QAs that need the video, a web search and code). A strict auditor
found the defects below. The video frames are attached (timestamps given); trust them.
CLASS %(cls)s - %(clsnote)s
%(note)sQUESTION TYPE %(qt)s - %(typenote)s   DIFFICULTY %(diff)s
QUESTION: %(q)s
ANSWER: %(a)s   (answer_range %(rng)s)
ANNOTATED SOLUTION:
%(sol)s
HIDDEN VALUES (the solver obtains them; never state them): %(hidden)s
WEB CONSTANTS (index | search query | value | verbatim quote) - the solver searches them; you cannot search again:
%(ws)s
SOLUTION CODE:
```python
%(code)s
```
CODE OUTPUT: %(out)s
AUDITOR'S DEFECTS:
%(defects)s
%(fb)sFix EVERY major defect with the smallest change. Keep the asked quantity, the object, the question type, the physics area,
the video readings (class A: the named pose readings; class B: the stated sizes / angles / times) and the difficulty.
Allowed:
- reword the question: describe the scene as the frames show it (fix a wrong object / event / time window), name which
  person / object / phase / reference level is meant, add a modelling assumption WITHOUT a number ("neglect air
  resistance", "take the ball's release speed equal to the athlete's peak wrist speed", "take the floor as zero height");
- an "Assume ..." sentence naming the real material, an official size / grade class or governing standard of a scene
  object, or a physical condition - no number, no brand / model / league / level / profession / population group / study;
- state an input the code uses but nobody gives as a number in the question (class B: a size / angle / time readable from
  the frames; any class: a plain given such as a release height) - never a hidden value or a kept web constant's value;
- if a web constant's quote cannot support its value for this scene and no rewording fixes it: list its index in
  "state_ws" - its value (the code's number) is then stated in the question as a given and its search is dropped (allowed
  only if the difficulty rule permits it; the script checks);
- edit the code (keep every print), then the answer / answer_range / annotated solution must follow the new output.
Never: widen a range or invent a +-%%; change a web constant's value; state a hidden value or a kept web constant's value;
name a tool or searching; give a method hint that removes a step; add a coefficient / efficiency the sources do not give.
TYPE_B: the answer is an interval propagated from range input(s). TYPE_A: one number.
A type_form / seed defect about the answer form: the final answer decides the type - one value -> "question_type":
"TYPE_A" (answer_range null; remove an invented +-%% from the code and the answer); an interval propagated from range
inputs -> "TYPE_B" with the answer itself "low-high unit"; a Yes/No question -> "TYPE_C".
If a defect cannot be fixed this way, return {"unfixable": "<why>"}.
Return ONLY JSON: {"question": "...", "code": "...", "answer": "...", "answer_range": [low, high] or null, "question_type": "...",
"annotated_solution": ["..."], "state_ws": [], "stated": [{"index": j, "value": number, "unit": "..."}], "changes": "..."}"""


def nums_ok(answer, rng, out, qt):
    on = AT.nums(out)
    an = [x for x in AT.nums(str(answer)) if not (float(x).is_integer() and abs(x) <= 10)] or AT.nums(str(answer))
    if not an:
        return 'the answer has no number'
    bad = [x for x in an if not any(near(x, o, 0.006) for o in on)]
    if bad:
        return 'the answer number(s) %s are not in the code output' % bad
    if qt == 'TYPE_B':
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng) and rng[0] < rng[1]):
            return 'TYPE_B needs answer_range [low, high] with low < high'
        if not all(any(near(v, o, 0.006) for o in on) for v in rng):
            return 'answer_range %s is not printed by the code' % rng
    return None


async def repair(item, rec, aud, diff, sibs, lock):
    qa = copy.deepcopy(rec['qa'])
    if isinstance(qa.get('sources'), str):
        qa['sources'] = ast.literal_eval(qa['sources'])
    hid = qa.get('hidden_cv_chosen') or []
    hid = ast.literal_eval(hid) if isinstance(hid, str) else list(hid)
    ws = F9.wsrc(qa)
    code = F9.code_of(qa)
    out0 = await asyncio.to_thread(W.run_code, code)
    fr, _ = await asyncio.to_thread(R.frames, R.video_path(item), qa['question'])
    fkey = '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (fr or [])))
    cls, qt = item['cls'], qa['question_type']
    dtxt = '\n'.join('- [%s, %s] %s' % (d.get('severity'), d.get('criterion'), d.get('detail')) for d in aud.get('defects') or []
                     if isinstance(d, dict))
    fb = ''
    for _ in range(4):
        p = FIX % dict(cls=cls, clsnote=X.CLS_NOTE[cls], note=X.REVNOTE[cls], qt=qt, typenote=X.TYPE_NOTE.get(qt, ''),
                       diff=qa.get('difficulty'), q=qa['question'], a=qa['answer'], rng=qa.get('answer_range'),
                       sol='\n'.join(qa.get('annotated_solution') or []), hidden=hid or '(none)',
                       ws='\n'.join('[%d] %s | %s | "%s"' % (j, s.get('search_query'), s.get('key_information'),
                                                             str(s.get('verified_quote') or '')[:260]) for j, s in enumerate(ws)) or '(none)',
                       code=code[:6000], out=out0[-1500:], defects=dtxt, fb=fb)
        g = W.pj(await X.mllm(p, fr, fkey)) or {}
        if g.get('unfixable'):
            return None, 'unfixable: ' + str(g['unfixable'])[:400]
        if not g.get('question') or not g.get('answer') or not isinstance(g.get('annotated_solution'), list):
            fb = 'YOUR LAST ANSWER WAS NOT VALID JSON WITH question / code / answer / annotated_solution.\n'
            continue
        code2 = g.get('code') or code
        out2 = out0 if code2 == code else await asyncio.to_thread(W.run_code, code2)
        P = []
        if out2.startswith('Error') or not out2.strip():
            P.append('the code fails: ' + out2[-300:])
        st = sorted({int(j) for j in g.get('state_ws') or [] if str(j).isdigit() and int(j) < len(ws)})
        sv = [float(s['value']) for s in g.get('stated') or [] if isinstance(s, dict) and isinstance(s.get('value'), (int, float))]
        new = copy.deepcopy(qa)
        new.update(question=g['question'], answer=g['answer'], annotated_solution=g['annotated_solution'])
        nt = g.get('question_type') if g.get('question_type') in ('TYPE_A', 'TYPE_B', 'TYPE_C') else qt
        if nt == 'TYPE_C' and not (qt == 'TYPE_B' and re.match(r'\W*(yes|no)', str(qa['answer']), re.I)):
            nt = qt  # TYPE_C only for a yes/no question that was mislabelled TYPE_B
        new['question_type'] = nt
        new['answer_range'] = g.get('answer_range') if nt == 'TYPE_B' else None
        drop = [ws[j] for j in st]
        new['sources'] = [s for s in qa['sources'] if not any(s is d for d in drop)]
        for s in new['sources']:
            if isinstance(s, dict) and s.get('type') == 'code_executor':
                s.update(code=code2, key_information=out2)
        if cls == 'B':
            new['hidden_cv_chosen'] = drop_hidden(hid, drop, sv)
        if not P:
            e = nums_ok(new['answer'], new.get('answer_range'), out2, nt)
            if e:
                P.append(e)
            P += [p for p in X.form_problems(new, qa, {}, cls) if not (nt == 'TYPE_C' and p.startswith('Keep the question type'))]
            P += [d['detail'] for d in AU.form_defects(new, out2)]
            qn, on = AT.nums(new['question']), AT.nums(qa['question'])
            for j in st:
                vs = [x for x in AT.nums(F9.code_of(qa)) if any(near(x, v) for v in AT.nums(str(ws[j].get('key_information'))))]
                if not any(any(near(x, q) for q in qn) for x in vs + [v for s in g.get('stated') or [] if isinstance(s, dict)
                                                                       and s.get('index') == j for v in [s.get('value')]
                                                                       if isinstance(v, (int, float))]):
                    P.append('web constant [%d] is in state_ws but the question does not state its value' % j)
            khid = drop_hidden(hid, drop, sv)
            bad = lambda x: (not (float(x).is_integer() and abs(x) <= 10) and any(near(x, q) for q in qn)
                             and not any(near(x, o) for o in on) and not any(near(x, v) for v in sv))
            leak = [x for j in range(len(ws)) if j not in st for x in AT.nums(str(ws[j].get('key_information'))) if bad(x)]
            leak += [x for h in khid for x in AT.nums(str(h)) if bad(x)]
            if leak:
                P.append('the question now states %s, the value of a kept web constant / a hidden value' % sorted(set(leak)))
            if TOOLRE.search(new['question']) and not TOOLRE.search(qa['question']):
                P.append('the question names a tool or searching')
            if re.search(r'\bassum\w*\b[^.]*\d', new['question'], re.I) and not re.search(r'\bassum\w*\b[^.]*\d', qa['question'], re.I):
                P.append('a new "Assume" sentence has a number: state a given input as a plain given ("take ... as ...") '
                         'and keep "Assume" sentences number-free')
        if not P and st:
            async with lock:
                left = len(ws) - len(st)
                nd = qa.get('difficulty') if cls == 'A' else LV.get(left, 'hard')
                other = [diff[s] for s in sibs if s != item['id']]
                if left < 1:
                    P.append('state_ws would leave no web constant')
                elif nd != qa.get('difficulty') and nd in other:
                    P.append('state_ws would change the difficulty to %s, which another QA of this video already has'
                             % nd)
                else:
                    diff[item['id']] = nd
                    new['difficulty'] = nd
        if P:
            fb = 'YOUR LAST FIX WAS REJECTED: %s\nYOUR LAST QUESTION: %s\n' % ('; '.join(P), g['question'])
            continue
        return new, g.get('changes') or ''
    return None, 'checks failed: ' + fb[:600]


async def main():
    W.SEM, W.SSEM, X.MSEM = asyncio.Semaphore(32), asyncio.Semaphore(16), asyncio.Semaphore(24)
    out_f, aud_f = ARGV[1], ARGV[2]
    recs = {}
    for f in ARGV[3:] + [out_f]:
        if os.path.exists(f):
            for k, r in json.load(open(f)).items():
                if r.get('status') in ('fixed', 'given') and r.get('qa'):
                    recs[k] = r
    rep = json.load(open(out_f)) if os.path.exists(out_f) else {}
    aud = json.load(open(aud_f))
    items = {i['id']: i for i in R.load()}
    diff = {k: i['qa'].get('difficulty') for k, i in items.items()}
    for k, r in recs.items():
        diff[k] = r['qa'].get('difficulty')
    sibs = defaultdict(list)
    for k in items:
        sibs[tuple(k.split('|')[:3])].append(k)
    sig = AU.sig
    att = rep.get('_attempts', {})
    todo = [k for k in recs if k in aud and aud[k].get('verdict') == 'fail' and aud[k].get('sig') == sig(recs[k]['qa'])
            and att.get(k, 0) < MAXR]
    print('audit fails %d, repair now %d' % (sum(1 for k in recs if aud.get(k, {}).get('verdict') == 'fail'), len(todo)), flush=True)
    lock = asyncio.Lock()
    C = Counter()

    async def one(k):
        att[k] = att.get(k, 0) + 1
        try:
            new, why = await repair(items[k], recs[k], aud[k], diff, sibs[tuple(k.split('|')[:3])], lock)
        except Exception as e:
            new, why = None, 'exception %s' % e
        if new is None:
            C['not repaired'] += 1
            rep.setdefault('_fail', {})[k] = why
        else:
            C['repaired'] += 1
            st = 'given' if recs[k]['status'] == 'given' or len(F9.wsrc(new)) < len(F9.wsrc(recs[k]['qa'])) else recs[k]['status']
            rep[k] = dict({x: recs[k].get(x) for x in ('id', 'cls', 'split')}, status=st, qa=new, repair=att[k], changes=why,
                          prev_defects=aud[k].get('defects'))
            rep.get('_fail', {}).pop(k, None)
        rep['_attempts'] = att
        json.dump(rep, open(out_f, 'w'), ensure_ascii=False, indent=1)
    await asyncio.gather(*[one(k) for k in todo])
    print(dict(C), flush=True)

if __name__ == '__main__':
    asyncio.run(main())
