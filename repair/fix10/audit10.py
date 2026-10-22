# Provenance (TriPhysBench release): fix_code/evaluation/fix10/audit10.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""strict whole-QA audit of every repaired QA by gpt-5.6-terra with the video frames (user 10/12: every QA the repair
changed is checked again as a whole, not only its diff). A QA with a major defect counts as unresolved.
Usage: python audit10.py out.json rec1.json [rec2.json ...]   (records with status 'fixed' / 'given' and a 'qa')"""
import sys, json, os, re, ast, asyncio, hashlib
from collections import Counter
ARGV = list(sys.argv)
sys.argv = [sys.argv[0], 'unres151_ids.json']
import fix10 as X  # noqa: E402
from fix10 import R, W, F9, AT  # noqa: E402
# user 10/13: the final answer decides the type - one value = TYPE_A, an interval = TYPE_B; the interval may come from any
# range input (a web constant's range, a range the question states, readings), never from an invented +-% or from
# disagreeing sources
X.TYPE_NOTE = dict(X.TYPE_NOTE,
                   TYPE_A='answer = ONE number with unit (no interval, no "accepted range +-x%").',
                   TYPE_B='answer = an interval "low-high unit" (answer_range [low, high]). The interval comes from '
                   'range input(s) - a web constant stated as a range, or a range the question states - propagated by the code; '
                   'never from an invented +-% or from disagreeing sources.')

X.REVIEWER = os.environ.get('FIX10_REVIEWER', 'gpt-5.6-terra')

NUM = r'[-+]?\d+(?:\.\d+)?(?:\s*[x×]\s*10\^?[-+]?\d+)?'
INTERVAL = re.compile(r'(?<![eE\d.])(?:%s)\s*[^\d\s,;()eE]{0,12}\s*(?:-|–|—|to|~)\s*(?:%s)|between\s+%s.{0,20}?\s+and\s+%s'
                      % (NUM, NUM, NUM, NUM), re.I)
PCT = re.compile(r'±\s*\d+(?:\.\d+)?\s*%|\+/-\s*\d+(?:\.\d+)?\s*%|\baccepted range\b', re.I)


def form(qa):
    """('interval' | 'single', interval stated in the answer text outside parentheses, answer_range is [low < high])"""
    a = re.sub(r'\([^)]*\)', '', str(qa.get('answer') or '')).replace('−', '-')
    rng = qa.get('answer_range')
    ok = isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng) and rng[0] < rng[1]
    iv = bool(INTERVAL.search(a))
    return ('interval' if iv or ok else 'single'), iv, ok


def form_defects(qa, out):
    """deterministic (user 10/13): TYPE_A one value, TYPE_B an interval in the answer + answer_range, no invented +-%"""
    t, (f, iv, ok) = qa.get('question_type'), form(qa)
    D = []
    if t == 'TYPE_A' and f == 'interval':
        D.append('the answer is an interval but the type is TYPE_A (one value): give one number, or make it TYPE_B if the '
                 'interval comes from range inputs')
    if t == 'TYPE_B' and not (iv and ok):
        D.append('TYPE_B: the answer must state the interval "low-high unit" and answer_range must be [low, high]; one '
                 'value (with a remark) is TYPE_A')
    if PCT.search(str(qa.get('answer'))) or PCT.search(out or ''):
        D.append('an invented +-% tolerance / "accepted range": remove it (one value -> TYPE_A; an interval only from range '
                 'inputs)')
    return [{'criterion': 'type_form', 'severity': 'major', 'detail': d} for d in D]


def sig(qa):
    return hashlib.sha1(json.dumps([AUDIT, X.TYPE_NOTE, X.REVNOTE, qa['question'], qa['answer'], qa.get('answer_range'),
                                    F9.code_of(qa)], ensure_ascii=False).encode()).hexdigest()

AUDIT = """You are a strict auditor of one QA of a physics video benchmark (QAs that need the video, a web search and code).
The QA was repaired; audit the WHOLE repaired QA, not only what changed. The video frames are attached; trust them.
CLASS %(cls)s - %(clsnote)s
QUESTION TYPE %(qt)s - %(typenote)s   DIFFICULTY %(diff)s (%(counts)s)
OLD QUESTION (before the repair, for reference): %(oldq)s
OLD ANSWER: %(olda)s
REPAIRED QUESTION: %(q)s
REPAIRED ANSWER: %(a)s   (answer_range %(rng)s)
HIDDEN VALUES (the solver must obtain them; never stated in the question): %(hidden)s
WEB CONSTANTS (base query | value used | verbatim quote | url):
%(ws)s
SOLUTION CODE:
```python
%(code)s
```
CODE OUTPUT: %(out)s
%(note)sReport every MAJOR defect (one that makes the QA wrong, ambiguous, unanswerable or not a fair benchmark item):
- leak: the question states a web-constant value or a hidden value, names a tool / searching, or gives a method hint that
  removes a step.
- scene: the question states something the frames contradict or do not show (object, material, count, motion, identity),
  or a stated given number is implausible for the frames. A time window "a-b s" means the event happens within that
  window (not during all of it); the frames are a sample, so an event between two frames is not a contradiction.
- assumption: an "Assume ..." sentence that selects WHICH value a web constant has (the scenario of a web search) names
  something other than the material, official size / grade class or governing standard of a scene object, or a physical
  condition; or it has a number; or it names a brand / model, a league / level / profession / population group, or one
  study; or its material is not the real material of the scene object. (A stated GIVEN input - a size, mass, distance,
  duration, angle or ratio the question states as a number, "released 0.3 m above her head", "take the mass as 0.62 kg" -
  is NOT an assumption defect; report it only under leak / scene. A modelling assumption without a number - "neglect air
  resistance", "the ball leaves the hand at the peak wrist speed" - is fine.)
- source: a web constant's quote does not state that value for the object / quantity / condition the question (+
  assumption) describes; or the query does not match the value (asks for a range but the value is one number, or names
  another quantity); or the code uses the constant in another role or unit than the quote states.
- physics: the code's model is physically wrong for the scene or for the question, or uses an input the question, the
  frames, the hidden values and the web constants do not give.
- answer: the answer does not match the code output (another number or unit; a difference larger than rounding to 3
  significant figures - more digits are fine), or is another
  quantity than the question asks; TYPE_B needs answer_range [low, high] with low < high that the code prints; TYPE_A one
  number.
- ambiguous: a careful solver could reasonably reach a different answer (an undefined quantity, two plausible web values
  that the question does not separate, an unclear time / phase of the motion).
- unnatural: ungrammatical or contrived wording.
Do NOT report: style preferences, a missing explanation, the old QA's problems that the repair fixed, a "typical" /
"standard" value whose quote states it, a size or angle the solver reads from the video (class B), a pose reading named
without its number (class A).
Return ONLY JSON: {"defects": [{"criterion": "...", "severity": "major" | "minor", "detail": "..."}]}  (empty list if none)"""


def ws_txt(qa):
    return '\n'.join('[%d] %s | %s | "%s" | %s' % (j, s.get('search_query'), s.get('key_information'),
                                                   str(s.get('verified_quote') or '')[:260], s.get('url') or '')
                     for j, s in enumerate(F9.wsrc(qa))) or '(none)'


async def audit(item, qa, seed=None):
    if isinstance(qa.get('sources'), str):
        qa['sources'] = ast.literal_eval(qa['sources'])
    code = F9.code_of(qa)
    out = await asyncio.to_thread(W.run_code, code)
    fr, _ = await asyncio.to_thread(R.frames, R.video_path(item), qa['question'])
    fkey = '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (fr or [])))
    hid = qa.get('hidden_cv_chosen') or []
    hid = ast.literal_eval(hid) if isinstance(hid, str) else hid
    cls = item['cls']
    d = dict(cls=cls, clsnote=X.CLS_NOTE[cls], qt=qa['question_type'], typenote=X.TYPE_NOTE.get(qa['question_type'], ''),
             diff=qa.get('difficulty'), counts='#web constants %d, #hidden %d' % (len(F9.wsrc(qa)), len(hid)),
             oldq=item['qa']['question'], olda=item['qa']['answer'], q=qa['question'], a=qa['answer'],
             rng=qa.get('answer_range'), hidden=hid or '(none)', ws=ws_txt(qa), code=code[:6000], out=out[-1500:],
             note=X.REVNOTE[cls])
    p = AUDIT % d
    rv = W.pj(await X.rmllm(p, fr, fkey))
    if not isinstance(rv, dict):
        await asyncio.sleep(60)
        rv = W.pj(await X.rmllm(p, fr, fkey))
    if not isinstance(rv, dict) or not isinstance(rv.get('defects'), list):
        return {'verdict': 'no_review'}
    D = [x for x in rv['defects'] if isinstance(x, dict)] + form_defects(qa, out)
    D += [{'criterion': 'seed', 'severity': 'major', 'detail': s} for s in seed or []]  # known problems of an unrepaired QA
    major = [x for x in D if x.get('severity') == 'major']
    return {'verdict': 'fail' if major else 'pass', 'defects': D}


async def main():
    W.SEM, W.SSEM, X.MSEM = asyncio.Semaphore(32), asyncio.Semaphore(16), asyncio.Semaphore(24)
    out_f = ARGV[1]
    recs = {}
    for f in ARGV[2:]:
        for k, r in json.load(open(f)).items():
            if r.get('status') in ('fixed', 'given') and r.get('qa'):
                recs[k] = r  # a later file overrides an earlier one
    items = {i['id']: i for i in R.load()}
    done = json.load(open(out_f)) if os.path.exists(out_f) else {}
    todo = [k for k in recs if k not in done or done[k].get('verdict') == 'no_review' or done[k].get('sig') != sig(recs[k]['qa'])]
    print('audit %d (todo %d)' % (len(recs), len(todo)), flush=True)

    async def one(k):
        try:
            r = await audit(items[k], json.loads(json.dumps(recs[k]['qa'])),
                            None if recs[k].get('repair') else recs[k].get('seed_defects'))
        except Exception as e:
            print('EXC', k, e, flush=True)
            return
        r.update(q=recs[k]['qa']['question'], a=recs[k]['qa']['answer'], status=recs[k]['status'], sig=sig(recs[k]['qa']))
        done[k] = r
        json.dump(done, open(out_f, 'w'), ensure_ascii=False, indent=1)
    await asyncio.gather(*[one(k) for k in todo])
    C = Counter((k.split('|')[0], k.split('|')[1], done[k]['verdict']) for k in recs if k in done)
    for x in sorted(C):
        print(x, C[x])
    print('criteria of major defects:', Counter(d.get('criterion') for k in recs if k in done
                                                for d in done[k].get('defects') or [] if d.get('severity') == 'major'))

if __name__ == '__main__':
    asyncio.run(main())
