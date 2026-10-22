# Provenance (TriPhysBench release): fix_code/evaluation/fix10/redesign17.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""redesign of the QAs whose small repairs failed in every round (user 10/13 "直接做": the remaining QAs are solved, not
deleted). Two kinds are left after v13-v16: (1) a web constant of an everyday object that has no unique value on the web
(hand mass, lamp mass, decanter mouth, pour-stream width, oil contact angle), (2) a question whose design the small edits
cannot fix (the only "web constant" is g, a stated density, a double question, a meaningless answer).
deepseek-flash (frames) writes a NEW version of the QA for the same video that keeps the benchmark design - class, physics
area, question type, difficulty (A: every hidden video reading used, the stated readings kept; B: the same number of web
constants), the video readings - and uses only web constants with a standard value (an official specification, a pure /
named material's property, a governing standard); every constant goes through the same search test as fix10 (const_test:
the base query and its paraphrase return one value, >=2 sites or a standard source; a different unique value replaces the
writer's guess and the answer is recomputed); then the deterministic checks and the strict gpt-5.6 whole-QA audit
(audit10.audit) decide. At most NDES designs per QA, each sees why the earlier ones failed.
Usage: python redesign17.py out.json ids.json rec1.json [rec2.json ...]   (the record files give the failure history)"""
import sys, json, os, re, ast, asyncio, copy
from collections import Counter
ARGV = list(sys.argv)
sys.argv = [sys.argv[0], 'unres151_ids.json']
import audit10 as AU  # noqa: E402  (fix10 + the 10/13 TYPE_NOTE, form checks, audit)
import fix10_v17 as V  # noqa: E402
from fix10 import R, W, F9, AT  # noqa: E402
from audit_fix import nums_ok  # noqa: E402
X = V.X
STAT = V.STAT
NDES = int(os.environ.get('REDESIGN_N', '4'))

REDESIGN = """You redesign one QA of a physics video benchmark (questions that need the video, a web search and Python code). The
QA's small repairs failed (reasons below). Write a NEW version of the QA for the SAME video that keeps the benchmark design
and uses only web constants that a search returns uniquely. The video frames are attached; trust them.
CLASS %(cls)s - %(clsnote)s
OLD QA: type %(qt)s (%(typenote)s), difficulty %(diff)s, physics area "%(area)s"
OLD QUESTION: %(q)s
OLD ANSWER: %(a)s
OLD SOLUTION:
%(sol)s
OLD WEB CONSTANTS (query | value): %(ws)s
%(readings)sOBJECTS SEEN IN THE VIDEO: %(objects)s
WHY THE OLD QA COULD NOT BE REPAIRED (last attempts):
%(hist)s
%(prev)sKeep:
- the same video, physics area and question type: TYPE_A = the answer is ONE number; TYPE_B = the answer is an interval
  that comes from a web constant whose standard value is itself a range (e.g. a regulation mass range); TYPE_C = Yes / No
  plus the deciding number(s); TYPE_D = a comparison (A > B, ratio / difference).
- the difficulty: exactly %(nws)d web constant(s)%(akeep)s.
- a comparable number of reasoning steps (no one-step lookup); keep the old asked quantity and object when you can,
  otherwise ask a closely related quantity of the same object / event.
Each web constant must have ONE value or ONE standard range that any careful search finds:
- an official / regulation specification (official basketball mass, regulation rim height, a standard paper size), a
  governing standard, a property of a pure substance or of a named material the frames show (density of water, specific
  heat of aluminium, latent heat of fusion of ice, speed of sound in air at 20 C), or a physical constant other than g / pi;
- NOT: the mass or size of an everyday product (lamp, decanter, glass, tool, phone, toy), a body mass / height of a
  population, a value from one study, a coefficient that varies widely (plastic-on-metal friction, the contact angle of an
  oil, the density of "wood" without a species), and NOT the constants that failed above.
Rules: the question never states a web-constant value or a hidden video reading's number; no tool / search wording; no
method hint that removes a step; an "Assume ..." sentence only names the real material / official size class / governing
standard of a scene object or a physical condition - no number, brand, model, league, profession, population or study;
nothing the frames contradict; a time window only where the frames show the event; class B: a number the question gives
(size / angle / time) must be readable from the frames, stated as an estimate ("about 30 cm across"); the ball's release
speed is the athlete's peak wrist speed (benchmark convention, no need to say it); never invent a +-%%.
CODE: one single-line assignment per web constant with the value you expect (`m_ball = 0.6  # kg`; a range as two lines
`m_lo = 0.567` and `m_hi = 0.650`), one variable per reading / given input, the calculation, and a print of every
intermediate and of the final result(s). The ANSWER is the final result your code prints.
Return ONLY JSON: {"question": "...", "code": "...", "answer": "...", "answer_range": [low, high] or null,
"question_type": "%(qt)s", "constants": [{"quantity": "<what is looked up>", "var": ["m_ball"], "code_unit": "kg",
"key_information": "<quantity = value unit>", "candidates": [{"query": "...", "paraphrase": "...", "assumption": ""}]}],
"why": "<what you changed and why each constant is unique>"}
Candidates: 4-6 per constant; query = what a solver types (its words are in the question / the assumption, or generic
such as "density", "official", "standard"); paraphrase = the same query worded differently (required); assumption = "" or
one "Assume ..." sentence (rules above) that the question then contains."""


def hist_of(recs, k):
    h = []
    for r in recs:
        x = r.get(k) or {}
        h += [str(t)[:600] for t in (x.get('history') or [])[-3:]]
        if x.get('reason'):
            h.append(str(x['reason'])[:300])
    return '\n'.join(h[-6:]) or '(none)'


async def design_one(item, rc, hist):
    orig = copy.deepcopy(item['qa'])
    if isinstance(orig['sources'], str):
        orig['sources'] = ast.literal_eval(orig['sources'])
    hid = orig.get('hidden_cv_chosen') or []
    hid = ast.literal_eval(hid) if isinstance(hid, str) else list(hid)
    orig['hidden_cv_chosen'] = hid
    cls, qt0 = item['cls'], orig['question_type']
    ws0 = F9.wsrc(orig)
    nws = len(ws0) if cls == 'B' else max(1, len(ws0))
    code0 = F9.code_of(orig)
    fr, _ = await asyncio.to_thread(R.frames, R.video_path(item), orig['question'])
    fkey = '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (fr or [])))
    objects = '; '.join('%s: %s' % (o.get('object'), o.get('seen')) for o in ((rc.get('video') or {}).get('objects') or [])
                        if isinstance(o, dict))[:1500] or '(none)'
    others = [s for s in orig['sources'] if not (isinstance(s, dict) and s.get('type') in ('web_search', 'code_executor'))]
    hn = F9.hidden_nums(orig) if cls == 'A' else []
    readings = ''
    akeep = ''
    if cls == 'A':
        readings = ('VIDEO READINGS the solver measures from the video (hidden: the code uses these numbers, the question names '
                    'the quantity but never the number): %s\nREADINGS THE OLD QUESTION STATES (keep them stated): %s\n'
                    % (hid or '(none)', [s.get('key_information') for s in others]))
        akeep = ('; the code uses every hidden video reading above and the question keeps the readings it states (the number '
                 'of hidden readings is the difficulty)')
    rec = {'id': item['id'], 'split': item['split'], 'cls': cls, 'version': 'redesign17', 'designs': []}
    prev = []
    for k in range(NDES):
        d = dict(cls=cls, clsnote=X.CLS_NOTE[cls], qt=qt0, typenote=X.TYPE_NOTE.get(qt0, ''), diff=orig.get('difficulty'),
                 area=orig.get('physics_category'), q=orig['question'], a=orig['answer'],
                 sol='\n'.join(orig.get('annotated_solution') or [])[:3000],
                 ws='; '.join('%s | %s' % (s.get('search_query'), s.get('key_information')) for s in ws0) or '(none)',
                 readings=readings, objects=objects, hist=hist, nws=nws, akeep=akeep,
                 prev=('EARLIER REDESIGNS WERE REJECTED (do not repeat them):\n' + '\n'.join(prev[-4:]) + '\n') if prev else '')
        g = W.pj(await X.mllm(REDESIGN % d, fr, fkey)) or {}
        STAT['designs'] += 1
        dr = {'design': k + 1, 'question': g.get('question'), 'why': g.get('why')}
        rec['designs'].append(dr)
        cs = [c for c in g.get('constants') or [] if isinstance(c, dict)]
        qt = g.get('question_type') if g.get('question_type') in ('TYPE_A', 'TYPE_B') and qt0 in ('TYPE_A', 'TYPE_B') else qt0
        P = []
        if not g.get('question') or not g.get('code') or not g.get('answer'):
            P.append('no valid JSON with question / code / answer / constants')
        elif len(cs) != nws:
            P.append('it has %d web constants; exactly %d are needed (difficulty)' % (len(cs), nws))
        if P:
            dr['error'] = P
            prev.append('DESIGN %d: %s' % (k + 1, '; '.join(P)))
            continue
        code1 = g['code']
        out1 = await asyncio.to_thread(W.run_code, code1)
        if out1.startswith('Error'):
            P.append('the code fails: ' + out1[-300:])
        else:
            e = nums_ok(g['answer'], g.get('answer_range'), out1, qt)
            if e:
                P.append(e)
            inc = lambda h, c: any(abs(h - x) <= 0.002 * max(abs(h), 1e-9) for x in AT.nums(c))
            miss = [h for h in hn if not inc(h, code1)]
            if miss:
                P.append('the code does not use the hidden video reading(s) %s' % miss)
            for c in cs:
                vs = [v for v in c.get('var') or [] if isinstance(v, str) and v.isidentifier()]
                if not vs or any(V.code_val(code1, v) is None for v in vs):
                    P.append('constant "%s": var %s is not a single-line numeric assignment in the code' % (c.get('quantity'), vs))
                cands = [x for x in c.get('candidates') or [] if isinstance(x, dict) and x.get('query')]
                if len(cands) < 2:
                    P.append('constant "%s" has fewer than 2 query candidates' % c.get('quantity'))
        if P:
            dr['error'] = P
            prev.append('DESIGN %d (question: %s): %s' % (k + 1, g['question'][:400], '; '.join(P)[:600]))
            continue
        qa1 = copy.deepcopy(orig)
        ws1 = [{'type': 'web_search', 'search_query': str(c['candidates'][0]['query']), 'key_information': str(c.get('key_information')),
                'url': None, 'verified_quote': None} for c in cs]
        qa1.update(question=g['question'], answer=str(g['answer']), question_type=qt,
                   answer_range=g.get('answer_range') if qt == 'TYPE_B' or (qt not in ('TYPE_A',) and g.get('answer_range')) else None,
                   sources=others + ws1 + [{'type': 'code_executor', 'code': code1, 'key_information': out1}])
        if cls == 'B':
            qa1['hidden_cv_chosen'] = [w['key_information'] for w in ws1]
        st = [p['text'] for p in F9.static_problems(qa1)]
        if st:
            dr['error'] = st
            prev.append('DESIGN %d (question: %s): %s' % (k + 1, g['question'][:400], '; '.join(st)[:600]))
            continue
        pcs = {j: {'var': c.get('var'), 'code_unit': c.get('code_unit')} for j, c in enumerate(cs)}
        ctx = {'qa': qa1, 'code': code1, 'out': out1, 'objects': objects, 'accc': {}, 'old_si': {j: [] for j in pcs},
               'quant': [str(c.get('quantity')) for c in cs], 'ki': [str(c.get('key_information')) for c in cs],
               'vi': {j: V.var_info(code1, pcs[j], {}) for j in pcs}}
        tests = {j: [dict(query=str(x.get('query') or '').strip(), paraphrase=str(x.get('paraphrase') or '').strip(),
                          assumption=str(x.get('assumption') or '').strip()) for x in c['candidates'] if isinstance(x, dict)][:6]
                 for j, c in enumerate(cs)}
        for j in tests:  # an assumption must be in the question (the writer puts it there)
            tests[j] = [c for c in tests[j] if c['query'] and (not c['assumption'] or c['assumption'].rstrip('.').lower()
                                                                in g['question'].lower())] or tests[j][:0]
        if not all(tests.values()):
            P.append('a constant has no candidate whose assumption sentence is in the question')
            dr['error'] = P
            prev.append('DESIGN %d: %s' % (k + 1, P[0]))
            continue
        res = await asyncio.gather(*[V.const_test(ctx, j, tests[j]) for j in tests])
        if any(r[0] == 'SERPER' for r in res):
            return None
        chosen, nf = {}, []
        for j, (ok, fails) in zip(tests, res):
            if ok:
                chosen[j] = ok[0]
            else:
                nf.append('constant "%s": %s' % (ctx['quant'][j], ' | '.join(f[:220] for f in fails[:4])))
        dr['tested'] = {j: {'pass': [(x['rank'], x['cand']['query'], x['value'], x['unit'], x['sites']) for x in r[0]],
                            'fail': r[1][:6]} for j, r in zip(tests, res)}
        if nf:
            dr['error'] = nf
            prev.append('DESIGN %d (question: %s): no unique search value - %s' % (k + 1, g['question'][:300], '; '.join(nf)[:900]))
            continue
        # a different unique value replaces the writer's guess
        code2, serr = code1, None
        for j, c in sorted(chosen.items()):
            if c['rank'] != 1:
                continue
            vi, G = ctx['vi'].get(j), c['G']
            rng_new = G['si'][0] != G['si'][1]
            if vi and vi['dim'] == G['dim'] and (len(vi['var']) == 2) == rng_new:
                lo, hi = G['si'][0] / vi['f'], G['si'][1] / vi['f']
                code2 = V.put(code2, vi['var'][0], lo) if len(vi['var']) == 1 else V.put(V.put(code2, vi['var'][0], lo), vi['var'][1], hi)
            else:
                c2 = await V.subst_force(code2, ctx['quant'][j], ctx['ki'][j], G['value'], G['unit'], G['quote'])
                if not c2:
                    serr = 'could not put the searched value %s %s of "%s" into the code' % (G['value'], G['unit'], ctx['quant'][j])
                    break
                code2 = c2
        out2 = out1 if code2 == code1 else await asyncio.to_thread(W.run_code, code2)
        if serr or out2.startswith('Error'):
            P.append(serr or 'the code with the searched values fails: ' + out2[-200:])
            dr['error'] = P
            prev.append('DESIGN %d: %s' % (k + 1, P[0]))
            continue
        new_ws = []
        for j, c in sorted(chosen.items()):
            vtxt = '-'.join(map(str, c['value'])) if isinstance(c['value'], list) else str(c['value'])
            new_ws.append({'type': 'web_search', 'search_query': c['cand']['query'], 'url': c['link'], 'url_annotated': c['link'],
                           'verified_quote': c['quote'], 'support': c['sites'],
                           'key_information': '%s = %s %s' % (ctx['quant'][j], vtxt, c['unit'])})
        fin = W.pj(await X.mllm(F9.FINAL % (g['question'], qt, '\n'.join(orig.get('annotated_solution') or []),
                                            g['answer'],
                                            '\n'.join('%s | %s | "%s"' % (s['search_query'], s['key_information'], str(s['verified_quote'])[:200])
                                                      for s in new_ws), code2, out2[:3000], g.get('answer_range')))) or {}
        if not fin.get('answer') or not isinstance(fin.get('annotated_solution'), list):
            prev.append('DESIGN %d: final answer failed' % (k + 1))
            continue
        new = copy.deepcopy(qa1)
        new.update(answer=str(fin['answer']), annotated_solution=[str(x) for x in fin['annotated_solution']],
                   answer_range=fin.get('answer_range') if qt == 'TYPE_B' or (qt in ('TYPE_C', 'TYPE_D') and fin.get('answer_range')) else None,
                   sources=others + new_ws + [{'type': 'code_executor', 'code': code2, 'key_information': out2,
                                               'key_information_annotated': None}])
        new['total_steps'] = len(new['annotated_solution'])
        if cls == 'B':
            new['hidden_cv_chosen'] = [s['key_information'] for s in new_ws]
            new['difficulty'] = orig.get('difficulty')  # #ws unchanged
        info = {'chosen': [{'value': c['value'], 'quote': c['quote']} for j, c in sorted(chosen.items())]}
        P = [p['text'] for p in F9.static_problems(new, orig, info) if not p['text'].startswith(('The original TYPE_', 'Forum / Q&A'))]
        P += [p for p in X.form_problems(new, orig, info, cls) if not p.startswith('The code uses input value')]
        P += [x['detail'] for x in AU.form_defects(new, out2)]
        e = nums_ok(new['answer'], new.get('answer_range'), out2, qt)
        if e:
            P.append(e)
        if re.search(r'\bassum\w*\b[^.]*\d', new['question'], re.I):
            P.append('an "Assume" sentence has a number')
        if P:
            dr['error'] = P
            prev.append('DESIGN %d (question: %s | answer %s): %s' % (k + 1, new['question'][:300], new['answer'], '; '.join(P)[:600]))
            continue
        au = await AU.audit(item, json.loads(json.dumps(new)))
        dr['audit'] = au
        if au.get('verdict') == 'pass':
            rec.update(status='fixed', mode='redesign', qa=new, changes=['redesign: %s' % str(g.get('why'))[:600]],
                       constants=[dict(index=j, rank=c['rank'], query=c['cand']['query'], assumption=c['cand']['assumption'],
                                       value=c['value'], unit=c['unit'], quote=c['quote'], url=c['link'], support=c['sites'])
                                  for j, c in sorted(chosen.items())])
            return rec
        prev.append('DESIGN %d (question: %s | answer %s) strict audit: %s' % (
            k + 1, new['question'][:300], new['answer'],
            ' | '.join('%s: %s' % (x.get('criterion'), str(x.get('detail'))[:220]) for x in au.get('defects') or []
                       if x.get('severity') == 'major')[:900]))
    rec['status'] = 'unresolved'
    rec['history'] = prev
    return rec


async def main():
    W.SEM, W.SSEM, X.MSEM = asyncio.Semaphore(32), asyncio.Semaphore(16), asyncio.Semaphore(24)
    V.SERP_SEM = asyncio.Semaphore(16)
    X.RN_SEM = asyncio.Semaphore(16)
    F9.FSEM = asyncio.Semaphore(16)
    out_f = ARGV[1]
    ids = json.load(open(ARGV[2]))
    hrecs = [json.load(open(f)) for f in ARGV[3:] if os.path.exists(f)]
    rc = {r['id']: r for r in json.load(open(X.HERE + '/recount_v2.json'))}
    items = {i['id']: i for i in R.load()}
    done = json.load(open(out_f)) if os.path.exists(out_f) else {}
    todo = [k for k in ids if k not in done]
    print('redesign %d (todo %d)' % (len(ids), len(todo)), flush=True)
    sem = asyncio.Semaphore(int(os.environ.get('REDESIGN_PAR', '24')))

    async def one(k):
        async with sem:
            try:
                r = await design_one(items[k], rc.get(k) or {}, hist_of(hrecs, k))
            except Exception as e:
                import traceback
                print('EXCEPTION %s %s\n%s' % (k, e, traceback.format_exc()[-1200:]), flush=True)
                return
        if r is None:
            print('SERPER FAILING', k, flush=True)
            return
        done[k] = r
        json.dump(done, open(out_f, 'w'), ensure_ascii=False, indent=1)
        print('progress %d/%d %s %s' % (len(done), len(ids), dict(Counter(x['status'] for x in done.values())), dict(STAT)), flush=True)
    await asyncio.gather(*[one(k) for k in todo])
    json.dump(F9._links, open(F9.LINKS, 'w'), ensure_ascii=False)
    print(dict(Counter((r['cls'], r['split'], r['status']) for r in done.values())), flush=True)

if __name__ == '__main__':
    asyncio.run(main())
