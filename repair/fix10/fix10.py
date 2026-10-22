# Provenance (TriPhysBench release): fix_code/evaluation/fix10/fix10.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 v7 (user 10/12): query-first repair of a web constant. Per constant 3-5 (assumption, base query) candidates; each
is tested by unique() = the acceptance check itself (recount.base_const with a second object-tagged extraction, substitution
into the code + official ACC, conflicts re-judged with frames) + >= 2 sites or a standard quote + a paraphrase of the query
finds no conflicting value. Order: the old value with a new query (answer unchanged) > one unique new value (answer
recomputed; a range and a single value may replace each other, TYPE_A <-> TYPE_B follows the final answer) > another
property of the same object. No whole rewrite (it drifted to other objects). Reviewer gpt-4o with frames.
fix10 step 2 (10/12): repair of the QAs that need a fix under criteria v2 (rejudge.py -> recount_v2.json needs_fix2).
User 10/12: small edit first, whole rewrite as fallback (edit rounds until a constant has no unique value: repair plan with frames -> re-sourced
constants verified with Serper -> edits applied -> accept; then the rewrite rounds below). Acceptance = recount.check with
Serper on every constant + video re-judge of conflicts + criteria v2 labels + static + fidelity.
Rewrite (10/12 first version): whole rewrite of the QAs the recount flagged (DESIGN.md; user 10/11: rewrite the whole question,
web constant = the annotator's base query reaches it and its value is unique, modelling stated, scene as in the video;
no widened range, no answer set; fixer deepseek-flash; difficulty and type kept).
Per QA, max ROUNDS rounds of
  PLAN    (deepseek-flash, sees video frames) scene, physics, exactly n_ws web constants of objects / materials / standards
          the frames show, each with a base query; modelling assumptions
  VERIFY  each constant BEFORE the question is written: Serper(base query) -> every value of the quantity with a verbatim
          quote; all must agree (else back to PLAN with the conflicting quotes)
  WRITE   question, video observations (B), code, key_information, B hidden_cv_chosen -> code run -> FINAL solution/answer
  ACCEPT  recount.check (W base query / M fresh / V frames + dense re-check of not_shown / S / C) + fix9 static checks
          (leaks, difficulty counts, format, sources) + fidelity review; any problem -> next round with the problems
Kept: class, split, video, question_type, #web constants, #hidden; A keeps its video observations and hidden_cv_chosen.
Output fix10.json {id: {status fixed|unresolved, orig_labels, rounds, qa}}.  Usage: python fix10.py ids.json"""
import os, sys, json, copy, ast, asyncio, hashlib, re
from collections import Counter, defaultdict
import httpx

ARGV = list(sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ['x']
import recount as R  # noqa: E402
import recheck_ns as RN  # noqa: E402
from recount import W  # noqa: E402
import fix9 as F9  # noqa: E402
import units as U  # noqa: E402

AT = W.AT
ROUNDS = 5  # user 10/12; unresolved after 5 rounds -> deleted from version 08
REWRITE = False  # v7: no whole-rewrite fallback
MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash')
REVIEWER = os.environ.get('FIX10_REVIEWER', 'gpt-4o')  # user 10/12: independent reviewer; frames as images (ACC-judge endpoint)
CACHE = HERE + '/fix10_llm_cache.jsonl'
OUT = HERE + '/' + os.environ.get('FIX10_OUT', 'fix10.json')
MSEM = None
_c = {}
if os.path.exists(CACHE):
    for line in open(CACHE):
        try:
            j = json.loads(line)
            _c[j['k']] = j['v']
        except Exception:
            pass


async def mllm(prompt, fr=None, fkey=''):
    """deepseek-flash, optionally with video frames [(t, b64jpeg)]; cached on prompt + frame key"""
    k = hashlib.sha1((MODEL + '|' + fkey + '|' + prompt).encode()).hexdigest()
    if k in _c:
        return _c[k]
    content = prompt
    if fr:
        content = [{'type': 'text', 'text': prompt}]
        for t, b in fr:
            content += [{'type': 'text', 'text': 'Frame at t = %.1f s:' % t},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b}}]
    async with MSEM:
        for a in range(5):
            body = {'model': MODEL, 'temperature': 0.0, 'max_tokens': 32000, 'messages': [{'role': 'user', 'content': content}]}
            if a >= 2:
                body['thinking'] = {'type': 'disabled'}
            try:
                async with httpx.AsyncClient(timeout=1200.0) as cl:
                    r = await cl.post(W.LLM_URL, headers={'Authorization': 'Bearer ' + W.LLM_KEY}, json=body)
                    t = r.json()['choices'][0]['message']['content']
                    if t:
                        _c[k] = t
                        with open(CACHE, 'a') as f:
                            f.write(json.dumps({'k': k, 'v': t}, ensure_ascii=False) + '\n')
                        return t
                    W.STAT['fix_empty'] += 1
            except Exception as e:
                W.STAT['fix_err ' + type(e).__name__] += 1
            await asyncio.sleep(4 * 2 ** a)
    W.STAT['fix_fail'] += 1
    return ''


async def rmllm(prompt, fr=None, fkey=''):
    """the reviewer (gpt-4o on the ACC-judge endpoint), with the same video frames; cached on model + prompt + frame key"""
    k = hashlib.sha1((REVIEWER + '|' + fkey + '|' + prompt).encode()).hexdigest()
    if k in _c:
        return _c[k]
    content = prompt
    if fr:
        content = [{'type': 'text', 'text': prompt}]
        for t, b in fr:
            content += [{'type': 'text', 'text': 'Frame at t = %.1f s:' % t},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b}}]
    async with MSEM:
        for a in range(5):
            body = {'model': REVIEWER, 'temperature': 0.0, 'max_tokens': 4000, 'messages': [{'role': 'user', 'content': content}]}
            try:
                async with httpx.AsyncClient(timeout=600.0) as cl:
                    r = await cl.post(W.JUDGE_API_URL, headers={'Authorization': 'Bearer ' + W.JUDGE_API_KEY}, json=body)
                    t = r.json()['choices'][0]['message']['content']
                    if t:
                        _c[k] = t
                        with open(CACHE, 'a') as f:
                            f.write(json.dumps({'k': k, 'v': t}, ensure_ascii=False) + '\n')
                        return t
                    W.STAT['rev_empty'] += 1
            except Exception as e:
                W.STAT['rev_err ' + type(e).__name__] += 1
            await asyncio.sleep(4 * 2 ** a)
    W.STAT['rev_fail'] += 1
    return ''


# ---------------------------------------------------------------- problems of a recount record, as text
def problems_txt(rc):
    """problems of a recount record under criteria v2 (user 10/12): W_query_added, M both-valid and "typical value" claims
    are not problems; W conflicts count only if the video re-judge (rejudge.py) keeps them"""
    P = []
    for i, c in enumerate(rc.get('consts') or []):
        if c.get('no_base_query'):
            P.append('Web constant [%d] (%s) has no search query.' % (i, c.get('ref')))
            continue
        if not c.get('reach'):
            P.append("Web constant [%d] (%s): its own query '%s' does not return the value." % (i, c.get('ref'), c.get('ref_query')))
        for x in (c.get('conflicts_video', c.get('conflicts')) or [])[:3]:
            P.append("Web constant [%d] (%s): the results of its query '%s' also give %s %s (\"%s\") -> answer %s; question + "
                     "video do not rule it out [%s]" % (
                         i, c.get('ref'), c.get('ref_query'), x.get('value'), x.get('unit') or '', (x.get('quote') or '')[:160],
                         x.get('alt_answer'), (x.get('video_why') or x.get('valid_why') or '')[:160]))
    for m in rc.get('model') or []:
        if m.get('acc') is not None and m['acc'] < .5 and m.get('verdict') == 'X':
            P.append("Modelling '%s': reference = %s; the alternative (judged more standard) = %s (-> %s)" % (
                m.get('aspect'), m.get('reference'), m.get('alternative'), m.get('alt_answer')))
    v = rc.get('video') or {}
    inv = v.get('invented_identity')
    bad = v.get('contradicted', []) + (v.get('invented', []) if inv is None else inv) + [
        s for s in (v.get('recheck') or {}).get('bad', []) if s.get('label') != 'invented']
    for c in bad:
        P.append('Scene: "%s" is %s - %s' % (c.get('claim'), c.get('label'), (c.get('evidence') or '')[:200]))
    for p in rc.get('static') or []:
        P.append('Text: %s %s' % (p['kind'], p.get('values') or p.get('text') or ''))
    if rc.get('consistency') not in (None, 'ok'):
        P.append('The reference code does not reproduce the answer (%s).' % rc['consistency'])
    return '\n'.join('- ' + p for p in P) or '- (none listed)'


# ---------------------------------------------------------------- prompts
COMMON = """CLASS %(cls)s - %(clsnote)s
QUESTION TYPE %(qtype)s - %(typenote)s
DIFFICULTY %(diff)s: exactly %(nws)d web constant(s)%(hidnote)s.
OLD QUESTION: %(question)s
OLD SOLUTION:
%(solution)s
OLD ANSWER: %(answer)s   (range %(range)s)
PROBLEMS OF THE OLD QA (why it is rewritten):
%(problems)s
OBJECTS SEEN IN THE VIDEO (frame audit): %(objects)s
OTHER QUESTIONS ABOUT THIS VIDEO (do not duplicate them): %(siblings)s"""

CLS_NOTE = {'A': 'the solver watches the video and reads motion quantities from it (the CV readings below); it looks up web '
                 'constants with Google and runs Python.',
            'B': 'the solver watches the video (the scene, given numbers in the question) and looks up web constants with Google '
                 'and runs Python.'}
TYPE_NOTE = {'TYPE_A': 'answer = one number with unit; all inputs are single values.',
             'TYPE_B': 'answer = a range "low-high unit". The range must come from a web constant whose STANDARD SPECIFICATION is '
                       'itself a range (a regulation tolerance such as an official ball mass 567-650 g), never from disagreeing '
                       'sources and never from an invented +-%.',
             'TYPE_C': 'answer = yes / no plus the deciding number(s) (e.g. "Yes - clearance 0.18 m"); a feasibility question.',
             'TYPE_D': 'answer = a comparison plus a ratio / difference (e.g. "A > B, ratio 1.3"); a comparison question.'}

PLAN = """You rewrite one QA of a physics video benchmark from scratch, because the old QA is ambiguous or describes the scene
wrongly. The frames of the video are attached (timestamps given). First plan the new QA; the question is written later.
""" + COMMON + """
%(cvnote)s
%(history)s
Plan the new QA:
1 SCENE: only what the frames show. Name no brand, league, model, grade or standard unless a marking or the setting in the
  frames identifies it. Do not claim that an event happens at a given time unless the frame at that time shows it.
2 WEB CONSTANTS: exactly %(nws)d. Each must be a property that reference sources state with ONE accepted value (or, only
  for a TYPE_B range, one standard specified range) for an object / material / standard the frames show or the setting
  implies:
    good: density of fresh water at 20 °C; latent heat of fusion of ice; specific heat of liquid water; speed of sound in
    dry air at 20 °C; density of aluminium; the regulation rim height of a basketball hoop; the official mass range of a
    size 7 basketball; the diameter of a standard pool (pocket billiards) ball.
    not allowed: "typical"/"average" properties of unidentified objects (mass of a wrench, a hand, a marble, a toy car, a
    person), friction coefficients of loosely defined surface pairs, "optimal" angles, research statistics, anything whose
    sources disagree; g, pi or unit conversions. Quantities like these become GIVEN inputs (stated as assumptions in the
    question) or video readings instead.
  A property of an unidentified object becomes unique through ONE explicit assumption sentence in the question that names a
  material / grade / standard / condition plausible for the frames ("Assume the marble is soda-lime glass."); list it in
  "assumptions" and use its words in base_query.
  Keep the old QA's object and asked quantity; never move a property to another object (no table-tennis ball for a block).
  For each constant give base_query: the Google query a solver would write, 3-8 words, no numbers, EVERY word taken from or
  directly implied by the question you will write (the question will name the object / material / condition exactly so).
3 MODELLING: every choice that changes the answer (start from rest, sliding vs rolling, which point / phase of the motion,
  what a ratio is relative to, which quantity is asked) will be stated in the question as an assumption or convention -
  not as a method hint. If the old QA had a modelling problem, use the more standard choice.
4 KEEP: class, the answer form of %(qtype)s, the difficulty counts above, the physics area (%(cat)s) if possible.
Return ONLY JSON: {"scene": "<what the frames show, with timestamps>", "physics_category": "<Area/Subtopic>",
"asked": "<the quantity asked and the reasoning chain in one or two sentences>", "assumptions": ["..."],
"given_inputs": ["<non-looked-up inputs the question will state, with values>"],
"constants": [{"quantity": "<exact quantity with object / material / condition>", "object": "<scene object it belongs to,
frame time>", "why_unique": "<why references agree on one value>", "range_spec": true|false, "base_query": "..."}]}"""

RULES = """Rules for the QUESTION text:
- Natural English, starts from the video scene ("In the video, ..."); describes only what the frames show.
- Names each constant's object / material / condition so that its base query follows from the question wording; never
  writes a web value or a hidden value (no number of a constant, not even rounded or in other units), never mentions tools
  or searching (web search, Google, look up, online, code).
- States the plan's assumptions and given inputs; no method hints beyond them.
- %(qtype)s answer form. Asks one thing (TYPE_C / TYPE_D: the decision / comparison and its deciding number).
- Concise (at most ~90 words). Mention only the objects the physics needs; no inventory of clothing or background.
- Keep the old question's asked quantity and reasoning wherever possible. Assumptions state physical conditions only (start
  from rest, ignore air resistance); never invent physics to make numbers work (e.g. "the two speeds add") and never
  prescribe the formula / model to use. A material / grade / standard the frames do not show is written only as an explicit
  assumption ("Assume ...") that is plausible for the frames, never as a fact of the scene.
Code: Python, uses the given inputs, the video readings%(cvcode)s and the verified constants (convert units in the code); prints
every number the answer states, rounded (f-strings, 3-4 significant figures); TYPE_B prints the low and the high end.
key_information: one sentence per web constant: the quantity, the value as quoted, the value used in SI units.%(bnote)s
Return ONLY JSON: {"question": "...", "physics_category": "...", %(vobs)s"code": "<full code>", "key_information": ["..."]%(hidkey)s}"""


WRITE = """Write the new QA of a physics video benchmark from the plan below.
""" + COMMON + """
%(cvnote)s
PLAN: %(plan)s
VERIFIED WEB CONSTANTS (value found with the base query; the solver must find it the same way):
%(consts)s
%(history)s
""" + RULES

EDIT_PLAN = """You repair one QA of a physics video benchmark with the SMALLEST change that removes the listed problems. The
frames of the video are attached (timestamps given). Keep the scene, the asked quantity, the video readings, the question
type, the difficulty counts and every sentence that has no problem.
""" + COMMON + """
%(cvnote)s
OLD WEB CONSTANTS:
%(oldconsts)s
%(history)s
Decide, per problem:
- A web constant whose base query does not return the value, or returns other values that question + video do not rule out:
  the fix is a NEW BASE QUERY for the SAME quantity of the SAME object that returns ONE value (or one standard range).
  Give 3-5 "candidates", best first, each a base query (3-8 words, no numbers):
  "requery": only the query changes; every word is in the question or shown by the frames;
  "qualify": the query names a specific material / grade / standard / condition, and ONE explicit assumption sentence added
  to the question names it ("Assume the marble is soda-lime glass.", "Assume the ball is a FIBA size 7 basketball.", "Take
  the syrup at 20 °C.", "Assume the rods are copper."). It must be plausible for what the frames show and not contradict
  them; it names no number, formula or method.
  The checker runs each query: if it returns the old value uniquely, only the query (and assumption) changes; if it returns
  one different value uniquely, that value replaces the old one and the answer is recomputed - that is fine. A standard
  range may replace a single value and vice versa. A "typical" quantity is fine if its query returns one value.
  Prefer specific, well-documented properties (a material's density / specific heat, a regulation dimension) and queries
  whose results state one number (a standard, a datasheet, a handbook value) over research statistics.
  "replace": only if no query for the same quantity can be unique: another property of the SAME scene object (e.g. its
  material's density with a size the question gives or the video shows), again with an assumption sentence if needed.
  Never move the constant to another object and never change the asked quantity.
  "keep" for every constant without a problem (no candidates).
- A modelling problem with a more standard alternative: the code uses the standard choice. Add one assumption sentence
  only if no default convention fixes the choice; it states a physical condition only (no formula, no method, no number).
- A scene statement that is contradicted / absent / an invented identity: rewrite only that phrase to what the frames show,
  neutral wording ("a basketball player", not "an NBA player"). If a constant's value depended on that identity, treat the
  constant as above.
- Text leaks or tool wording: remove the value / wording. Code that does not reproduce the answer: the code is fixed and the
  answer recomputed.
Return ONLY JSON: {"edits": ["<old phrase> -> <new phrase>"], "modelling": ["<choice the code makes / assumption sentence>"],
"constants": [{"index": <old index>, "action": "keep|fix", "candidates": [{"action": "requery|qualify|replace",
"assumption": "<sentence added to the question, or empty>", "quantity": "<exact quantity with object / material /
condition>", "base_query": "...", "range_spec": true|false, "why_unique": "..."}]}]}
"constants" has exactly %(nws)d entries, in the old order."""

EDIT_WRITE = """Apply the repair plan to the old QA of a physics video benchmark. Change ONLY what the plan says; copy every
other sentence of the old question verbatim. Each constant marked ADD TO THE QUESTION gets that assumption sentence in the
question (verbatim or merged naturally into the sentence about that object), and the object's other wording must agree
with it. Update the code for the new constant values / modelling choice; keep the rest
of the code.
""" + COMMON + """
%(cvnote)s
OLD CODE:
%(code)s
REPAIR PLAN: %(plan)s
WEB CONSTANTS (kept ones as in the old QA, new ones verified with their base query):
%(consts)s
%(history)s
""" + RULES

FIDEL = """You review a rewritten QA of a physics video benchmark. The old QA was ambiguous; the new one must be a natural,
well-posed question about the same video.
The video frames are attached; trust them. OBJECTS NOTED BY AN EARLIER AUDIT (may be incomplete): %s
OLD QUESTION: %s
NEW QUESTION: %s
NEW ANSWER: %s   (type %s, range %s)
WEB CONSTANTS USED (quantity | value | verbatim quote | page):
%s
CODE:
%s
CODE OUTPUT: %s
%sReport only real defects:
1 surrogate: a scene object is given the property of a different object, or the question is about an object not in the video.
2 source_mismatch: the quote is about a different object, material, size class or standard than the question names.
  A quote that omits a qualifier which does not change the value (room temperature, "pure", the standard's name, a brand
  the quote cites) is NOT a mismatch.
6 invented: the question ASSERTS as a fact of the scene a material, grade or identity the frames do not show ("the ball is a
  FIBA size 7 ball"), or an assumption contradicts the frames / is implausible for them (steel for a visibly plastic toy).
  An explicit assumption ("Assume the marble is soda-lime glass.", "Assume the rods are copper." for metallic rods) that is
  plausible for the frames is NOT a defect.
3 unnatural: contrived or self-contradictory wording, a method hint that gives away the solution, a question that cannot be
  answered from the question + video + the web constants.
4 answer: the answer is not what the code prints, or the code does not implement what the question asks.
5 trivial: a web constant is g, pi or a unit conversion, or the looked-up value does not influence the answer.
7 drift: the new question is about a different object or asks a different quantity than the old question (e.g. a block's
  lifting work became a table-tennis ball's; an athlete's kinetic energy became the ball's).
Return ONLY JSON: {"defects": [{"criterion": "surrogate|source_mismatch|unnatural|answer|trivial|invented|drift", "detail": "..."}]}"""

FIDEL_EDIT = """You review a minimally edited QA of a physics video benchmark. The old QA was ambiguous; the new one must be a natural,
well-posed question about the same video.
The video frames are attached; trust them. OBJECTS NOTED BY AN EARLIER AUDIT (may be incomplete): %s
OLD QUESTION: %s
OLD ANSWER: %s
NEW QUESTION: %s
NEW ANSWER: %s   (type %s, range %s)
CHANGES MADE BY THE REPAIR (web constants re-sourced, modelling fixes): %s
WEB CONSTANTS USED (quantity | value | verbatim quote | page):
%s
CODE:
%s
CODE OUTPUT: %s
%sThe OLD question was reviewed before; judge only what the edit changed. Report a defect only if
the edit INTRODUCED it (a changed or new phrase, a new or re-sourced constant, changed code) - not a design feature the
old question already had and the edit kept. Report only real defects:
1 surrogate: a scene object is given the property of a different object, or the question is about an object not in the video.
2 source_mismatch: the quote is about a different object, material, size class or standard than the question names.
  A quote that omits a qualifier which does not change the value (room temperature, "pure", the standard's name, a brand
  the quote cites) is NOT a mismatch.
6 invented: the question ASSERTS as a fact of the scene a material, grade or identity the frames do not show ("the ball is a
  FIBA size 7 ball"), or an assumption contradicts the frames / is implausible for them (steel for a visibly plastic toy).
  An explicit assumption ("Assume the marble is soda-lime glass.", "Assume the rods are copper." for metallic rods) that is
  plausible for the frames is NOT a defect.
3 unnatural: contrived or self-contradictory wording, an ungrammatical sentence or a sentence fragment left by the edit, a
  method hint that gives away the solution, a question that cannot be answered from the question + video + the web
  constants (e.g. the code uses a size / mass the question does not state).
4 answer: the answer is not what the code prints, or the code does not implement what the question asks.
5 trivial: a web constant is g, pi or a unit conversion, or the looked-up value does not influence the answer.
8 added_physics: the new question or code introduces a coefficient, multiplier, efficiency, extra force or effect that
  neither the old question nor the frames support (e.g. "the release speed is 1.5 times the wrist speed", air buoyancy on a
  concrete block, a motor efficiency of 0.85), or an extra input whose only purpose is to make the numbers work.
9 answer_change: the new answer differs from the old one for a reason other than the CHANGES listed above (a changed web
  value, a modelling fix); e.g. the asked quantity or the reasoning chain changed.
Return ONLY JSON: {"defects": [{"criterion": "surrogate|source_mismatch|unnatural|answer|trivial|invented|added_physics|answer_change",
"detail": "..."}]}"""

REVNOTE = {
    'A': """BENCHMARK CONVENTIONS (not defects): the solver measures body-joint readings from the video with a pose tool, so a
question may name a reading ("the athlete's peak wrist speed") without its number - it is answerable. A stated assumption
that links a reading to the object ("the ball leaves the hand at the peak wrist speed", "the ball moves with the athlete")
is a modelling assumption, not a surrogate. Benchmark convention (user 10/13): the ball's release / launch speed IS the
athlete's peak wrist speed, even if the question does not say so; this is not undefined, not ambiguous and not a missing
input, and the elbow (or another joint) speed is not a competing alternative.
""",
    'B': """BENCHMARK CONVENTION (not a defect): the solver may read sizes, angles and times from the video; a question that
states such an estimate ("the marble is about 1.6 cm across") is fine.
""",
}


# ---------------------------------------------------------------- steps
def same_value(a, b):
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    if len(a) == len(b):
        return all(abs(x - y) <= 0.03 * max(abs(y), 1e-9) for x, y in zip(a, b))
    p, r = (a, b) if len(a) == 1 else (b, a)
    lo, hi = min(r), max(r)
    return lo <= p[0] <= hi and (hi - lo) <= 0.06 * max(abs(hi + lo) / 2, 1e-9)


REQ = """A web search for the quantity "%s" returned snippets without its value. Queries tried: %s
Write ONE other short search query for the same quantity (the words of the quantity, e.g. "air viscosity 20 C Pa s" or
"size 7 basketball weight grams"); no new qualifiers. Return ONLY JSON: {"query": "..."}"""


async def verify_const(scene, c):
    """Serper(base query) -> all values of the quantity; unique -> {value, unit, quote, url, query}; else conflict text"""
    q = (c.get('base_query') or '').strip()
    if not q:
        return None, 'constant "%s": base_query missing' % c.get('quantity')
    tried = []
    for a in range(3):
        if a:
            j = W.pj(await W.llm(REQ % (c.get('quantity'), '; '.join(tried)))) or {}
            q = str(j.get('query') or '').strip()
            if not q or q in tried:
                break
        tried.append(q)
        res = await W.asearch(q)
        if res is None:
            return 'SERPER', None
        e = W.pj(await W.llm(W.EXTRACT % (scene, c.get('quantity'), W.results_block([res])))) or {}
        vals = []
        for v in e.get('values') or []:
            if not (isinstance(v, dict) and v.get('value') is not None and W.quote_ok(v.get('quote'), res, v['value'])):
                continue
            x = U.si(v['value'], v.get('unit'))
            if x:
                vals.append(dict(v, _iv=x[0], _dim=x[1]))
        if vals:
            break
    if not vals:
        return None, 'constant "%s": the results of %s state no value of it. Results: %s' % (
            c.get('quantity'), ' / '.join('"%s"' % t for t in tried), re.sub(r'\s+', ' ', res)[:700])
    dims = Counter(v['_dim'] for v in vals)
    dim = dims.most_common(1)[0][0]
    vals = [v for v in vals if v['_dim'] == dim]  # other dimensions are other quantities picked up by the extractor
    # components of the "agree" relation (SI): points within 5 %, a point inside a range, ranges with ends within 3 %
    comp = list(range(len(vals)))

    def root(i):
        while comp[i] != i:
            i = comp[i]
        return i
    for i in range(len(vals)):
        for j in range(i):
            if U.agree(vals[i]['_iv'], vals[j]['_iv'], 0.12):
                comp[root(i)] = root(j)
    groups = defaultdict(list)
    for i, v in enumerate(vals):
        groups[root(i)].append(v)
    cl = list(groups.values())
    rng = bool(c.get('range_spec'))
    why = None
    if len(cl) > 1:
        why = 'the results disagree'
    else:
        g = cl[0]
        ranges = [v for v in g if U.width(v['_iv']) > 0]
        if rng and not ranges:
            why = 'a range was planned but the results state a single value'
        elif not rng and any(U.width(v['_iv']) > 0.20 for v in g):
            why = 'the results state a wide range, not one value'
    if why:
        return None, 'constant "%s" (query "%s"): %s: %s' % (c.get('quantity'), q, why, ' | '.join(
            '%s %s ("%s")' % (v['value'], v.get('unit') or '', v['quote'][:150]) for v in [g[0] for g in cl][:5]))
    g = cl[0]
    pool = [v for v in g if (U.width(v['_iv']) > 0) == rng] or g
    v = Counter(json.dumps(x['value']) for x in pool).most_common(1)[0][0]
    v = next(x for x in pool if json.dumps(x['value']) == v)
    url = await asyncio.to_thread(find_url, q, v)
    return {'quantity': c.get('quantity'), 'value': v['value'], 'unit': v.get('unit', ''), 'quote': v['quote'], 'query': q,
            'url': url, 'n_agree': len(g)}, None


def find_url(q, v):
    """page of the chosen quote among the raw Serper results: quote prefix, any 25-char window, then the value's number"""
    org = F9.links(q) or []
    norm = lambda t: re.sub(r'\s+', ' ', str(t or '')).strip().lower()
    qt = norm(v['quote'])
    for o in org:
        if qt[:30] in norm(o.get('snippet')):
            return o['link']
    for k in range(0, max(len(qt) - 25, 1), 8):
        w = qt[k:k + 25]
        for o in org:
            if len(w) >= 15 and w in norm(o.get('snippet')):
                return o['link']
    nums = [str(x) for x in (v['value'] if isinstance(v['value'], list) else [v['value']])]
    for o in org:
        sn = norm(o.get('snippet')) + ' ' + norm(o.get('title'))
        if all(re.sub(r'\.0$', '', n) in sn for n in nums):
            return o['link']
    return ''


COND = re.compile(r'(\d+(?:\.\d+)?)\s*(?:°\s*[CF]|degrees? (?:C|F|Celsius|Fahrenheit)|K\b|atm\b)|(?:size|grade|No\.?)\s*(\d+)', re.I)


def cond_nums(q):
    return {float(a or b) for a, b in COND.findall(q)}


RANGE = re.compile(r'\d\s*[a-zA-Z°%/³²]*\s*(?:-|–|—|to)\s*\d')


def area(q):
    return re.sub(r'[^a-z]', '', re.split(r'\s*/\s*', str(q.get('physics_category') or ''))[0].lower())[:5]


COMMONN = [9.81, 9.8, 9.807, 9.80665, 1000, 100, 1e3, 1e-3, 1e-6, 1e6, 0.01, 0.001, 3600, 60, 273.15, 101325, 0.0254, 0.3048,
           0.453592, 0.45359237, 28.3495, 0.0283495, 1609.34, 0.44704, 3.6, 4.184, 2.54, 12, 1e-2, 1e-9, 0.5]


def new_inputs(qa, orig, cls):
    """numeric literals the new code uses that the question, the web constants, the old code and the video readings do not
    give (and that are no common constant / conversion / small integer): inputs the solver cannot know"""
    have = AT.nums(qa['question']) + AT.nums(F9.code_of(orig)) + [x for s in qa['sources'] if isinstance(s, dict)
                                                                  and s.get('type') in ('web_search',)
                                                                  for x in AT.nums(s.get('key_information'))
                                                                  + AT.nums(s.get('verified_quote'))]
    have += F9.hidden_nums(qa) + COMMONN
    if cls == 'A':  # A: the solver measures every CV reading from the video with the pose tool
        have += [x for s in qa['sources'] if isinstance(s, dict) and s.get('type') == 'video_observation'
                 for x in AT.nums(s.get('key_information'))]
    code = re.sub(r'#[^\n]*', '', F9.code_of(qa))
    code = re.sub(r'(f?["\'])(?:\\.|(?!\1).)*\1', '', code)  # strings (print formats) and comments
    near = lambda x, y: abs(x - y) <= 0.005 * max(abs(x), abs(y), 1e-12)
    lit = [float(m) for m in re.findall(r'(?<![\w.])(\d+\.?\d*(?:[eE][-+]?\d+)?)', code)]
    return sorted({x for x in lit if not (x.is_integer() and abs(x) <= 10) and not any(near(x, h) or near(x, h * 1000) or near(x, h / 1000)
                                                                                     or near(x, h * 100) or near(x, h / 100) for h in have)})


def form_problems(qa, orig, info, cls):
    """type / difficulty design kept by a rewrite: #video readings, physics area, the answer form of the question type"""
    P = []
    nv = lambda q: sum(isinstance(x, dict) and x.get('type') == 'video_observation' for x in q['sources'])
    if nv(qa) != nv(orig):
        P.append('The QA must keep exactly %d video reading(s) (has %d) to keep its difficulty.' % (nv(orig), nv(qa)))
    if cls == 'A':  # the readings the old question states must stay stated (#hidden = readings the solver must measure)
        vn = [x for o in orig['sources'] if isinstance(o, dict) and o.get('type') == 'video_observation'
              for x in AT.nums(o.get('key_information'))]
        same = lambda x, y: abs(x - y) <= 1e-9 * max(abs(x), 1e-12)
        lost = [x for x in AT.nums(orig['question']) if any(same(x, v) for v in vn) and not (float(x).is_integer() and abs(x) <= 10)
                and not any(same(x, y) for y in AT.nums(qa['question']))]
        if lost:
            P.append('The old question states the video reading(s) %s; keep them stated (the number of hidden readings '
                     'fixes the difficulty).' % sorted(set(lost)))
    if area(orig) and area(qa) != area(orig):
        P.append('Keep the physics area of the old QA (%s); the new QA is %s.' % (orig.get('physics_category'),
                                                                                qa.get('physics_category')))
    ni = new_inputs(qa, orig, cls)
    if ni:
        P.append('The code uses input value(s) %s that neither the question nor the web constants give - the solver cannot '
                 'know them. State each as a given input in the question (a size estimate "about 1.6 cm across" is fine) '
                 'or do not use it.' % ni)
    t, a = qa['question_type'], str(qa['answer'])
    if t != orig['question_type'] and {t, orig['question_type']} != {'TYPE_A', 'TYPE_B'}:
        P.append('Keep the question type %s.' % orig['question_type'])
    if t == 'TYPE_C' and not re.match(r'\W*(yes|no)\b', a, re.I):
        P.append('TYPE_C: the answer must start with Yes / No and give the deciding number(s).')
    if t == 'TYPE_D' and not re.search(r'[<>%]|greater|less|more|higher|lower|larger|smaller|faster|slower|ratio|times|exceed|than|compar|increase|decrease|'
                                       r'sufficient|differ|change|fraction|percent',
                                       a, re.I):
        P.append('TYPE_D: the answer must state the comparison (A > B) and its ratio / difference.')
    if t == 'TYPE_B' and info.get('chosen') and not any(isinstance(c.get('value'), list) or RANGE.search(str(c.get('value')))
                                                       or RANGE.search(str(c.get('quote'))) for c in info['chosen']):
        P.append('TYPE_B: the range must come from a web constant whose standard specification is itself a range.')
    return P


async def accept(item, orig, info, objects):
    """recount problems of the new QA (+ dense re-check of not_shown claims) + fix9 static + fidelity -> list of texts"""
    r = await R.check(copy.deepcopy(item), None)
    if r.get('serper_failed'):
        return 'SERPER', r
    if 'V_not_shown' in r['labels']:
        o = await RN.recheck(r, item, RN_SEM)
        r['video']['recheck'] = o
        lab = Counter(s.get('label') for s in o.get('bad', []))
        r['labels'] = sorted(set(r['labels'] + ['V_absent'] * bool(lab['absent']) + ['V_contradicted'] * bool(lab['contradicted'])
                                 + ['V_invented'] * bool(lab['invented'])))
    # criteria v2: Serper-found conflicts re-judged with the video, identity vs typical claims (rejudge.py)
    cond = cond_nums(item['qa']['question'])
    for st in r['static']:
        if st.get('values'):
            st['values'] = [x for x in st['values'] if not any(abs(x - c) <= 1e-9 * max(abs(c), 1e-12) for c in cond)]
    r['static'] = [st for st in r['static'] if st.get('values') != []]
    r['labels'] = [l for l in r['labels'] if not l.startswith('S_') or l[2:] in {st['kind'] for st in r['static']}]
    import rejudge as RJ
    RJ.F = sys.modules[__name__]  # use this module's mllm / MSEM (fix10 runs as __main__)
    if 'W_base_conflict' in r['labels']:
        fr, _ = await asyncio.to_thread(R.frames, R.video_path(item), r['question'])
        await RJ.rejudge_conflicts(r, item, fr, '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (fr or []))))
    if 'V_invented' in r['labels']:
        r['video']['invented_identity'] = await RJ.invented_identity(r)
    r['labels2'] = RJ.labels2(r)
    P = problems_txt(r).split('\n') if r['labels2'] else []
    P = [p[2:] for p in P if p != '- (none listed)']
    P += [p['text'] for p in F9.static_problems(item['qa'], orig, info)
          if not p['text'].startswith('The original TYPE_')]  # v7: answer_range is required only for TYPE_B
    P += form_problems(item['qa'], orig, info, item['cls'])
    qa = item['qa']
    if item['cls'] == 'A':  # the CV readings must enter the code
        code = F9.code_of(qa)
        miss = [h for h in F9.hidden_nums(qa) if not any(abs(h - x) <= 0.002 * max(abs(h), 1e-9) for x in AT.nums(code))]
        if miss:
            P.append('The code does not use the video reading(s) %s.' % miss)
    cs = '\n'.join('%s | %s %s | "%s" | %s' % (c['quantity'], c['value'], c.get('unit') or '', c['quote'][:200], c.get('url'))
                   for c in info['chosen']) or '(none)'
    out = next((s.get('key_information') for s in qa['sources'] if s.get('type') == 'code_executor'), '')
    fidel = FIDEL if info.get('mode') != 'edit' else FIDEL_EDIT
    ffr, _ = await asyncio.to_thread(R.frames, R.video_path(item), orig['question'])
    ffk = '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (ffr or [])))
    if fidel is FIDEL_EDIT:
        fa = (objects, orig['question'], orig['answer'], qa['question'], qa['answer'], qa['question_type'], qa.get('answer_range'),
              '; '.join(info.get('changes') or []) or 'none', cs, F9.code_of(qa)[:4000], str(out)[-1500:], REVNOTE[item['cls']])
    else:
        fa = (objects, orig['question'], qa['question'], qa['answer'], qa['question_type'], qa.get('answer_range'), cs,
              F9.code_of(qa)[:4000], str(out)[-1500:], REVNOTE[item['cls']])
    j = W.pj(await rmllm(fidel % fa, ffr, ffk))
    if not isinstance(j, dict) or not isinstance(j.get('defects'), list):
        P.append('Reviewer: no valid review returned.')
        j = {}
    P += ['Reviewer (%s): %s' % (d.get('criterion'), d.get('detail')) for d in (j.get('defects') or [])
          if isinstance(d, dict) and d.get('criterion') in ('surrogate', 'source_mismatch', 'unnatural', 'answer', 'trivial', 'invented',
                                                             'drift', 'added_physics', 'answer_change')]
    return P, r


def context(item, rc, sib):
    qa = item['qa']
    ws = F9.wsrc(qa)
    hid = qa.get('hidden_cv_chosen') or []
    vobs = [s for s in qa['sources'] if isinstance(s, dict) and s.get('type') == 'video_observation']
    objs = '; '.join('%s: %s' % (o.get('object'), o.get('seen')) for o in ((rc.get('video') or {}).get('objects') or [])
                     if isinstance(o, dict))[:1500] or '(none)'
    d = dict(cls=item['cls'], clsnote=CLS_NOTE[item['cls']], qtype=qa['question_type'], typenote=TYPE_NOTE[qa['question_type']],
             diff=qa.get('difficulty'), nws=len(ws), question=qa['question'],
             solution='\n'.join(qa.get('annotated_solution') or []), answer=qa['answer'], range=qa.get('answer_range'),
             problems=problems_txt(rc), objects=objs, siblings=' || '.join(sib) or '(none)',
             cat=qa.get('physics_category'))
    if item['cls'] == 'A':
        d['hidnote'] = ' and exactly %d hidden video reading(s)' % len(hid)
        d['cvnote'] = ('VIDEO READINGS (CV measurements of the video; keep all of them and use them in the new QA): %s. '
                       'HIDDEN ones (the solver must read them from the video; the question asks for / refers to the quantity '
                       'but never states the value): %s. Given ones may be stated in the question as before. Refer to a reading '
                       'by its quantity ("the athlete\'s peak wrist speed during the shot"), not by claiming an event at a time.'
                       % ([s.get('key_information') for s in vobs], hid or 'none'))
        d['cvcode'] = ' (all of them, values as listed)'
        d['bnote'] = ''
        d['vobs'] = ''
        d['hidkey'] = ''
    else:
        d['hidnote'] = (' and %d hidden value(s) = the web constants' % len(hid)) if hid else ''
        d['cvnote'] = 'OLD VIDEO OBSERVATIONS (given scene numbers; keep or restate, they must fit the frames): %s' % (
            [s.get('key_information') for s in vobs])
        d['cvcode'] = ''
        d['bnote'] = ('\nhidden_cv_chosen: exactly %d entries "<short quantity name> = <value> <unit>", one per web constant, '
                      'value as quoted.' % len(hid)) if hid else ''
        d['vobs'] = '"video_observations": [{"time_point": "<a-b s>", "key_information": "<what is observed / given>"}], '
        d['hidkey'] = ', "hidden_cv_chosen": ["..."]' if hid else ''
    return d


def old_consts(item, rc):
    """the QA's web constants as they are (kept constants of a small edit; accept() re-verifies them with Serper)"""
    rcc = rc.get('consts') or []
    out = []
    for i, s in enumerate(F9.wsrc(item['qa'])):
        c = rcc[i] if i < len(rcc) else {}
        out.append({'quantity': c.get('quantity') or W.mask(s.get('key_information')), 'value': s.get('key_information') or '',
                    'unit': '', 'quote': s.get('verified_quote') or '', 'query': s.get('search_query') or '',
                    'url': s.get('url') or '', 'kept': True})
    return out


async def build_accept(item, d, wr, out, chosen, k, hist, rec, plan, mode):
    """FINAL answer from the code output -> new QA -> accept(); returns 'fixed' | 'SERPER' | None (problems -> hist)"""
    orig = item['qa']
    fin = W.pj(await mllm(F9.FINAL % (wr['question'], d.get('qtype_final') or orig['question_type'], d['solution'], orig['answer'],
                                      '\n'.join('%s | %s %s | "%s"' % (c['query'], c['value'], c.get('unit') or '', c['quote'][:200])
                                                for c in chosen), wr['code'], out[:3000], orig.get('answer_range')))) or {}
    if not fin.get('answer') or not isinstance(fin.get('annotated_solution'), list):
        hist.append(mode.upper() + ' ATTEMPT %d: final answer failed.' % (k + 1))
        rec['rounds'].append({'mode': mode, 'plan': plan, 'question': wr['question'], 'error': 'final failed'})
        return None
    new = copy.deepcopy(orig)
    kis = wr.get('key_information') or []
    web = [{'type': 'web_search', 'url': c['url'], 'key_information': kis[i] if i < len(kis) and kis[i] else
            '%s = %s %s' % (c['quantity'], c['value'], c.get('unit') or ''), 'search_query': c['query'], 'verified_quote': c['quote']}
           for i, c in enumerate(chosen)]
    if item['cls'] == 'A' or not isinstance(wr.get('video_observations'), list):
        vobs = [s for s in orig['sources'] if isinstance(s, dict) and s.get('type') == 'video_observation']
    else:
        vobs = [dict(type='video_observation', time_point=str(v.get('time_point', '')), key_information=str(v.get('key_information', '')))
                for v in wr['video_observations'] if isinstance(v, dict)]
    other = [s for s in orig['sources'] if isinstance(s, dict) and s.get('type') not in ('web_search', 'code_executor', 'video_observation')]
    new['sources'] = vobs + other + web + [{'type': 'code_executor', 'code': wr['code'], 'key_information': out,
                                            'key_information_annotated': None}]
    qt = d.get('qtype') or orig['question_type']
    new['question_type'] = qt
    new.update(question=wr['question'], annotated_solution=[str(x) for x in fin['annotated_solution']], answer=str(fin['answer']),
               answer_range=fin.get('answer_range') if qt == 'TYPE_B' or (qt != 'TYPE_A' and orig.get('answer_range') is not None) else None,
               physics_category=orig.get('physics_category') if mode == 'edit' else
               (wr.get('physics_category') or plan.get('physics_category') or orig.get('physics_category')))
    new['total_steps'] = len(new['annotated_solution'])
    new['required_external_tools'] = (['web_search'] if web else []) + ['code_executor']
    if item['cls'] == 'B' and orig.get('hidden_cv_chosen'):  # B hidden values = the web values (memory: follow ws)
        h = wr.get('hidden_cv_chosen')
        if not (isinstance(h, list) and len(h) == len(orig['hidden_cv_chosen'])):
            h = ['%s = %s %s' % (c['quantity'], '-'.join(map(str, c['value'])) if isinstance(c['value'], list) else c['value'],
                                 c.get('unit') or '') for c in chosen][:len(orig['hidden_cv_chosen'])]
        new['hidden_cv_chosen'] = [str(x).strip() for x in h]
    info = {'chosen': chosen, 'quote_checks': [], 'mode': mode, 'changes': d.get('changes') or []}
    P, r = await accept(dict(item, qa=new), orig, info, d['objects'])
    if P == 'SERPER':
        return 'SERPER'
    rec['rounds'].append({'mode': mode, 'plan': plan, 'question': new['question'], 'answer': new['answer'], 'answer_range': new.get('answer_range'),
                          'chosen': chosen, 'labels': r['labels'], 'problems': P, 'qa': new})
    if not P:
        rec.update(status='fixed', mode=mode, qa=new, qtype_change=qt != orig['question_type'],
                   constants=[{k: c.get(k) for k in ('quantity', 'value', 'unit', 'query', 'assumption', 'action', 'rank',
                                                       'old_value', 'support', 'paraphrase', 'cand_fail')} for c in chosen])
        return 'fixed'
    hist.append(mode.upper() + ' ATTEMPT %d question: %s\n  answer: %s\n  problems found: %s' % (k + 1, new['question'], new['answer'],
                                                                                ' | '.join(p[:300] for p in P)))
    # the Serper re-check of the edited QA found no unique value for a constant -> an edit cannot fix it
    return 'noconst' if any(l.startswith('W_') for l in r.get('labels2') or []) else None


# ---------------------------------------------------------------- v7: one uniqueness test (user 10/12)
PARA = """Rewrite this web search query the way another person might type it: the same words in another order, or common
synonyms / spellings ("weight" for "mass", "metre" for "meter"). Keep every qualifier (material, grade, standard, size,
condition); add nothing, drop nothing. Query: %s
Return ONLY JSON: {"query": "..."}"""
STD = re.compile(r'\b(rules?|regulations?|official|standards?|specifications?|ISO|ASTM|EN \d|DIN|FIBA|FIFA|ITTF|NBA|NCAA|UCI|'
                 r'World Athletics|IAAF|NIST|CRC|handbook)\b', re.I)


def link_of(org, quote):
    norm = lambda t: re.sub(r'\s+', ' ', str(t or '')).strip().lower()
    qt = norm(quote)
    if len(qt) < 15:
        return ''
    for o in org:
        if qt[:30] in norm(o.get('snippet')):
            return o['link']
    for k in range(0, max(len(qt) - 25, 1), 8):
        w = qt[k:k + 25]
        for o in org:
            if len(w) >= 15 and w in norm(o.get('snippet')):
                return o['link']
    return ''


def site(link):
    from urllib.parse import urlparse
    h = urlparse(link or '').netloc.lower()
    return h[4:] if h.startswith('www.') else h


def snippet_with(blob, ki):
    ns = AT.nums(ki)
    for p in re.split(r' \| |: ', str(blob or '')):
        if ns and any(AT.has([n], p) for n in ns):
            return p.strip()[:200]
    return ''


async def unique(item, d, qa, src, code, out, quantity, fr, fkey, para=True):
    """the uniqueness test of one web constant = what acceptance checks (recount.base_const with the object-tagged second
    extraction: every value the base query returns is substituted into the code and scored with the official ACC; ACC < .5
    is a conflict unless question + frames single out the reference, re-judged with the frames as in rejudge.py), plus
    >= 2 sites (or one standard / regulation / handbook quote) state the value, plus a paraphrase of the query returns no
    conflicting value. -> 'SERPER' | {'ok', 'why', 'd', 'quote', 'url', 'support', 'paraphrase'}"""
    import rejudge as RJ
    RJ.F = sys.modules[__name__]

    async def run(q):
        dc = await R.base_const(qa, dict(src, search_query=q), code, out, quantity)
        if dc.get('serper_failed'):
            return 'SERPER', None
        if dc.get('conflicts'):
            await RJ.rejudge_conflicts({'consts': [dc], 'video': {'objects': d.get('_objs') or []}},
                                       {'qa': {'question': qa['question'], 'sources': [src]}}, fr, fkey)
        return dc, dc.get('conflicts_video', dc.get('conflicts')) or []
    q = src['search_query']
    dc, conf = await run(q)
    if dc == 'SERPER':
        return 'SERPER'
    res = {'ok': False, 'why': '', 'd': dc, 'quote': '', 'url': '', 'support': []}
    say = lambda xs: '; '.join('%s %s ("%s") -> answer %s' % (x.get('value'), x.get('unit') or '', (x.get('quote') or '')[:120],
                                                            x.get('alt_answer')) for x in xs[:3])
    if not dc.get('reach'):
        res['why'] = 'the query does not return the value "%s"' % str(src.get('key_information'))[:120]
        return res
    if conf:
        res['why'] = 'the query also returns ' + say(conf)
        return res
    agree = [r for r in dc.get('candidates') or [] if r.get('same') or (r.get('acc') or 0) >= .5]
    blob = await W.asearch(q)
    quote = next((r['quote'] for r in agree if r.get('same') and r.get('quote')), '') or snippet_with(blob, src.get('key_information'))
    org = await asyncio.to_thread(F9.links, q) or []
    sites = {site(link_of(org, r.get('quote'))) for r in agree} | {site(link_of(org, quote))}
    sites.discard('')
    res.update(quote=quote, url=link_of(org, quote), support=sorted(sites))
    if len(sites) < 2 and not any(STD.search(str(x)) for x in [quote] + [r.get('quote') for r in agree]):
        res['why'] = 'only %s states the value; need two sites or a standard / handbook quote' % (sorted(sites) or 'one snippet')
        return res
    if para:
        pq = str((W.pj(await W.llm(PARA % q)) or {}).get('query') or '').strip()
        if pq and pq.lower() != q.lower():
            d2, c2 = await run(pq)
            if d2 == 'SERPER':
                return 'SERPER'
            res['paraphrase'] = {'query': pq, 'reach': d2.get('reach'), 'conflicts': len(c2)}
            if c2:
                res['why'] = 'the paraphrased query "%s" returns other values: %s' % (pq, say(c2))
                return res
    res['ok'] = True
    return res


def pick_value(dc, qt):
    """the best-supported group of the values a query returned (other than the old value) -> one row with its code"""
    rows = [r for r in dc.get('candidates') or [] if r.get('code') and r.get('alt_answer') and not str(r.get('out', 'Error')).startswith('Error')]
    groups = []
    for r in rows:
        x = U.si(r['value'], r.get('unit'))
        if not x:
            continue
        for g in groups:
            if g['dim'] == x[1] and U.agree(g['iv'], x[0], 0.03):
                g['rows'].append(r)
                break
        else:
            groups.append({'dim': x[1], 'iv': x[0], 'rows': [r]})
    if not groups:
        return None
    g = max(groups, key=lambda g: len(g['rows']))
    rng = lambda r: isinstance(r['value'], list) and len(set(map(str, r['value']))) > 1
    return next((r for r in g['rows'] if rng(r) == (qt == 'TYPE_B')), g['rows'][0])


async def const_candidate(item, d, j, x, fr, fkey):
    """one (assumption, base query) candidate for the old web constant j -> (dict, None) | (None, why) | ('SERPER', None).
    rank 0: the query returns the old value uniquely (only query / assumption change); rank 1: it returns one different
    value uniquely (value and answer change); rank 2: "replace" (another quantity; reach pre-check, acceptance tests it)"""
    orig = item['qa']
    a = str(x.get('assumption') or '').strip()
    q = str(x.get('base_query') or '').strip()
    quantity = str(x.get('quantity') or (d['_quant'][j] if j < len(d['_quant']) else '') or '')
    if x.get('action') == 'replace':
        v, e = await verify_const(orig['question'] + ' ' + a, x)
        if v == 'SERPER':
            return 'SERPER', None
        return (dict(v, assumption=a, action='replace', rank=2), None) if v else (None, e)
    qa0 = copy.deepcopy(orig)
    qa0['question'] = (orig['question'] + ' ' + a).strip()
    src = dict(F9.wsrc(orig)[j], search_query=q)
    u = await unique(item, d, qa0, src, d['_code'], d['_out'], quantity, fr, fkey)
    if u == 'SERPER':
        return 'SERPER', None
    base = dict(quantity=quantity, query=q, assumption=a, action=x.get('action'))
    if u['ok']:
        return dict(base, value=src.get('key_information') or '', unit='', quote=u['quote'], url=u['url'], rank=0,
                    support=u['support'], paraphrase=u.get('paraphrase')), None
    best = pick_value(u['d'], orig['question_type'])
    if not best:
        return None, 'query "%s": %s' % (q, u['why'])
    qa1 = copy.deepcopy(qa0)
    qa1.update(answer=best['alt_answer'], answer_range=None)
    src1 = dict(src, key_information='%s = %s %s' % (quantity, best['value'], best.get('unit') or ''), verified_quote=best['quote'])
    u1 = await unique(item, d, qa1, src1, best['code'], best['out'], quantity, fr, fkey)
    if u1 == 'SERPER':
        return 'SERPER', None
    if not u1['ok']:
        return None, 'query "%s": old value: %s; its value %s %s: %s' % (q, u['why'], best['value'], best.get('unit') or '', u1['why'])
    return dict(base, value=best['value'], unit=best.get('unit') or '', quote=best['quote'], url=u1['url'] or u['url'], rank=1,
                old_value=src.get('key_information'), new_answer=best['alt_answer'], support=u1['support'],
                paraphrase=u1.get('paraphrase')), None


def new_type(orig, chosen):
    """user 10/12: a range and a single value may replace each other; TYPE_A <-> TYPE_B follows the final answer (a range
    answer needs a web constant that is a range). TYPE_C / TYPE_D keep their type."""
    t = orig['question_type']
    if t not in ('TYPE_A', 'TYPE_B') or not any(c.get('rank') in (1, 2) for c in chosen):
        return t
    rng = any((isinstance(c.get('value'), list) and len(set(map(str, c['value']))) > 1) or RANGE.search(str(c.get('value')))
              for c in chosen)
    return 'TYPE_B' if rng else 'TYPE_A'


async def edit_round(item, rc, d, fr, fkey, k, hist, rec):
    """small edit: repair plan (frames) -> Serper-verify the re-sourced constants -> apply edits -> build_accept"""
    olds = old_consts(item, rc)
    oc = '\n'.join('[%d] %s | key_information: %s | base query: %s' % (i, c['quantity'], c['value'][:200], c['query'])
                   for i, c in enumerate(olds)) or '(none)'
    plan = W.pj(await mllm(EDIT_PLAN % dict(d, oldconsts=oc), fr, fkey)) or {}
    cs = [c for c in plan.get('constants') or [] if isinstance(c, dict)]
    if len(cs) != d['nws']:
        hist.append('EDIT ATTEMPT %d: the repair plan must list exactly %d constant(s) (had %d).' % (k + 1, d['nws'], len(cs)))
        rec['rounds'].append({'mode': 'edit', 'plan': plan, 'error': hist[-1]})
        return None

    async def one(i, c):
        j = c['index'] if isinstance(c.get('index'), int) and 0 <= c['index'] < len(olds) else i
        if c.get('action') == 'keep':
            return olds[j], None
        cand = [x for x in c.get('candidates') or [] if isinstance(x, dict) and x.get('base_query')][:5]
        if not cand:
            return None, 'constant [%d]: no candidate query given' % j
        vs = await asyncio.gather(*[const_candidate(item, d, j, x, fr, fkey) for x in cand])
        if any(v[0] == 'SERPER' for v in vs):
            return 'SERPER', None
        ok = sorted((v['rank'], n) for n, (v, _) in enumerate(vs) if v)  # old value > new value > replace; planner order
        if ok:
            return dict(vs[ok[0][1]][0], cand_fail=[e for _, e in vs if e]), None
        return None, 'constant [%d]: ' % j + ' || '.join(e for _, e in vs if e)
    vr = await asyncio.gather(*[one(i, c) for i, c in enumerate(cs)])
    if any(v[0] == 'SERPER' for v in vr):
        return 'SERPER'
    errs = [e for _, e in vr if e]
    if errs:
        hist.append('EDIT ATTEMPT %d: re-sourced constants failed verification: %s' % (k + 1, ' || '.join(errs)))
        rec['rounds'].append({'mode': 'edit', 'plan': plan, 'error': errs})
        return 'noconst'
    chosen = [v for v, _ in vr]
    tag = lambda c: ('kept' if c.get('kept') else 'same value, new base query' if c.get('rank') == 0 else
                     'NEW VALUE (old: %s) - update the code' % str(c.get('old_value'))[:120] if c.get('rank') == 1 else
                     'NEW QUANTITY - update the code')
    ctxt = '\n'.join('[%d] %s = %s %s | quote: "%s" | base query: %s (%s)%s' % (
        i, c['quantity'], c['value'], c.get('unit') or '', c['quote'][:200], c['query'], tag(c),
        ' | ADD TO THE QUESTION: "%s"' % c['assumption'] if c.get('assumption') else '')
        for i, c in enumerate(chosen)) or '(none)'
    qt = new_type(item['qa'], chosen)
    d = dict(d, qtype=qt, typenote=TYPE_NOTE[qt],
             qtype_final=qt if qt == item['qa']['question_type'] else '%s (changed from %s: the web constant is now %s; answer in '
             'the %s form, not the old form)' % (qt, item['qa']['question_type'], 'a standard range' if qt == 'TYPE_B' else
                                                 'a single value', qt),
             changes=['constant [%d] %s: %s' % (i, c['quantity'], tag(c) if c.get('rank') != 1 else 'value %s -> %s %s' % (
                 str(c.get('old_value'))[:100], c['value'], c.get('unit') or '')) + (' | assumption "%s"' % c['assumption']
                                                                                     if c.get('assumption') else '')
                      for i, c in enumerate(chosen) if not c.get('kept')] + [str(m) for m in plan.get('modelling') or []])
    wr = W.pj(await mllm(EDIT_WRITE % dict(d, code=F9.code_of(item['qa'])[:6000], plan=json.dumps(plan, ensure_ascii=False),
                                           consts=ctxt))) or {}
    if not wr.get('question') or not wr.get('code'):
        hist.append('EDIT ATTEMPT %d: applying the edits failed.' % (k + 1))
        rec['rounds'].append({'mode': 'edit', 'plan': plan, 'error': 'write failed'})
        return None
    out = await asyncio.to_thread(W.run_code, wr['code'])
    if out.startswith('Error'):
        hist.append('EDIT ATTEMPT %d: the code failed: %s' % (k + 1, out[-200:]))
        rec['rounds'].append({'mode': 'edit', 'plan': plan, 'question': wr['question'], 'error': 'code: ' + out[-200:]})
        return None
    return await build_accept(item, d, wr, out, chosen, k, hist, rec, plan, 'edit')


async def fix_one(item, rc, sib):
    orig = item['qa']
    if isinstance(orig['sources'], str):
        orig['sources'] = ast.literal_eval(orig['sources'])
    d = context(item, rc, sib)
    d['_code'] = F9.code_of(orig)
    d['_out'] = await asyncio.to_thread(W.run_code, d['_code'])
    d['_quant'] = [c.get('quantity') for c in rc.get('consts') or []]
    d['_objs'] = (rc.get('video') or {}).get('objects') or []
    fr, _ = await asyncio.to_thread(R.frames, R.video_path(item), orig['question'])
    fkey = '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (fr or [])))
    rec = {'id': item['id'], 'split': item['split'], 'cls': item['cls'], 'orig_labels': rc.get('labels2', rc['labels']),
           'rounds': []}
    hist = []
    mode = 'edit'
    for k in range(ROUNDS):
        d['history'] = ('EARLIER ATTEMPTS FAILED (fix these, do not repeat them):\n' + '\n'.join(hist)) if hist else ''
        if mode == 'edit':  # small edit first; whole rewrite only when a constant has no unique value (user 10/12)
            res = await edit_round(item, rc, d, fr, fkey, k, hist, rec)
            if res in ('SERPER', 'fixed'):
                return None if res == 'SERPER' else rec
            if res == 'noconst' and REWRITE:
                mode = 'rewrite'
            continue
        plan = W.pj(await mllm(PLAN % d, fr, fkey)) or {}
        cs = [c for c in plan.get('constants') or [] if isinstance(c, dict)]
        if not plan.get('scene') or len(cs) != d['nws']:
            hist.append('ATTEMPT %d: the plan must have exactly %d web constant(s) (had %d).' % (k + 1, d['nws'], len(cs)))
            rec['rounds'].append({'error': hist[-1]})
            continue
        vr = await asyncio.gather(*[verify_const(plan['scene'] + ' ' + plan.get('asked', ''), c) for c in cs])
        if any(v[0] == 'SERPER' for v in vr):
            return None
        errs = [e for _, e in vr if e]
        if errs:
            hist.append('ATTEMPT %d plan constants %s failed verification: %s' % (
                k + 1, [c.get('quantity') for c in cs], ' || '.join(errs)))
            rec['rounds'].append({'plan': plan, 'error': errs})
            continue
        chosen = [v for v, _ in vr]
        ctxt = '\n'.join('[%d] %s = %s %s | quote: "%s" | base query: %s' % (i, c['quantity'], c['value'], c.get('unit') or '',
                                                                          c['quote'][:200], c['query']) for i, c in enumerate(chosen)) or '(none)'
        wr = W.pj(await mllm(WRITE % dict(d, plan=json.dumps(plan, ensure_ascii=False), consts=ctxt))) or {}
        if not wr.get('question') or not wr.get('code'):
            hist.append('ATTEMPT %d: writing failed.' % (k + 1))
            rec['rounds'].append({'plan': plan, 'error': 'write failed'})
            continue
        out = await asyncio.to_thread(W.run_code, wr['code'])
        if out.startswith('Error'):
            hist.append('ATTEMPT %d: the code failed: %s' % (k + 1, out[-200:]))
            rec['rounds'].append({'plan': plan, 'question': wr['question'], 'error': 'code: ' + out[-200:]})
            continue
        res = await build_accept(item, d, wr, out, chosen, k, hist, rec, plan, 'rewrite')
        if res in ('SERPER', 'fixed'):
            return None if res == 'SERPER' else rec
    rec['status'] = 'unresolved'
    return rec


async def main():
    global MSEM, RN_SEM
    W.SEM, W.SSEM, MSEM, RN_SEM = asyncio.Semaphore(48), asyncio.Semaphore(6), asyncio.Semaphore(32), asyncio.Semaphore(16)
    R.VSEM = asyncio.Semaphore(24)
    R.TAGGED = True  # v7: the object-tagged second extraction in every base-query check (pre-check and acceptance)
    F9.FSEM = asyncio.Semaphore(16)
    rc = {r['id']: r for r in json.load(open(HERE + '/recount_v2.json'))}  # criteria v2 (rejudge.py)
    items = R.load()
    sib = defaultdict(list)
    for i in items:
        sib[i['id'].rsplit('|', 1)[0]].append((i['id'], i['qa']['question']))
    ids = json.load(open(ARGV[1])) if len(ARGV) > 1 else [i for i, r in rc.items() if r.get('needs_fix2')]
    ids = set(ids)
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [i for i in items if i['id'] in ids and i['id'] not in done]
    print('QAs %d todo %d' % (len(ids), len(todo)), flush=True)
    qsem = asyncio.Semaphore(int(os.environ.get('FIX10_PAR', '48')))  # QAs in flight; all submitted at once
    stop = []

    async def run(i):
        if stop:
            return
        async with qsem:
            if stop:
                return
            r = await fix_one(i, rc[i['id']], [q[:300] for j, q in sib[i['id'].rsplit('|', 1)[0]] if j != i['id']])
        if r is None or W.SERPER_FAILS[0] >= 3:
            stop.append(i['id'])
            return
        done[r['id']] = r
        json.dump(done, open(OUT, 'w'), ensure_ascii=False, indent=1)
        print('progress %d/%d %s serper_fail %d %s' % (len(done), len(ids), dict(Counter(x['status'] for x in done.values())),
                                                    W.SERPER_FAILS[0], dict(W.STAT)), flush=True)
    await asyncio.gather(*[run(i) for i in todo])
    json.dump(F9._links, open(F9.LINKS, 'w'), ensure_ascii=False)
    if stop:
        print('SERPER FAILING - stop', flush=True)
        sys.exit(3)
    C = Counter((r['split'], r['cls'], r['status']) for r in done.values())
    for k in sorted(C):
        print(k, C[k])
    print('rounds used (fixed):', Counter(len(r['rounds']) for r in done.values() if r['status'] == 'fixed'))
    print('problems left (unresolved, last round):', Counter(p.split(':')[0][:40] for r in done.values() if r['status'] == 'unresolved'
                                                             for p in (r['rounds'][-1].get('problems') or [str(r['rounds'][-1].get('error'))[:40]])))


RN_SEM = None
if __name__ == '__main__':
    asyncio.run(main())
