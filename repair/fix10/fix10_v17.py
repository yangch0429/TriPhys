# Provenance (TriPhysBench release): fix_code/evaluation/fix10/fix10_v17.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 v8 (user 10/12): fix only the problems the recount found for each QA (recount_v2: labels2 + their evidence), verify
only what changed, and limit every edit to a diff (v7 re-audited the whole QA each round and let the writer rewrite it: 7 of
the 16 pilot QAs had no web problem but were failed or damaged on their web constants).
Per QA, max ROUNDS rounds of
  PLAN     deepseek-flash with frames, one call: exact question / code span edits for S (remove the leaked value),
           V (the scene phrase as the frames show it), M (the code computes the alternative judged more standard); for each
           web constant that needs a new base query 6-8 candidates (query, paraphrase, optional "Assume ..." sentence:
           material / size or grade class / governing standard / physical condition only)
  SEARCH   all candidate queries and paraphrases at once (one Serper call each, cached)
  EXTRACT  one non-thinking call per constant over all results (numbered, with site): every value of the quantity, what it
           refers to, which scenarios (assumptions) it applies to
  DECIDE   deterministic: every value is put into the code variable of the constant, the code is run, official ACC vs the
           answer. A candidate is unique if every applicable value from its query and its paraphrase agrees (ACC >= .5) with
           the old value (rank 0: answer kept) or with one new value (rank 1: answer recomputed; range <-> single value may
           replace each other, TYPE_A <-> TYPE_B follows); >= 2 sites or a standard / handbook quote state it.
           Order: rank 0 without assumption > rank 0 with > rank 1 without > rank 1 with.
  ACCEPT   deterministic checks (leaks gone, type / difficulty counts, no unknown inputs, the answer changes only through a
           changed value or the modelling fix, M fix = the alternative's answer) + one gpt-5.6-terra review of the diff with
           the frames. A review defect on one constant -> its next candidate, without a new plan.
Unresolved after ROUNDS -> deleted from version 08.  Output as fix10.py.  Usage: FIX10_OUT=x.json python fix10_v9.py ids.json
v9 (pilot v8 analysis, 10/12): a range query is answered only by range values; a V edit re-tests a constant only if it
removed that constant's words; time stamps are deleted, not moved; an unsupported material becomes an assumption, never a
dropped constant (checked: every web constant stays used); M problems "adopt" or "state" (when the alternative needs an
unknown number); object mass may become material density x stated size; degenerate ranges rejected; the old TYPE_B range
and readings the old code did not use are not re-checked; unchanged output keeps answer / range / solution (no FINAL);
failed queries are not re-searched; stop after 3 rounds with no unique value for the same constant.
v10 (pilot i analysis, 10/12): a candidate that passed is offered again in later rounds (v9 re-planned it away into an
implausible assumption); question edits name the web constants their removed words determined (re-tested); a re-tested
constant first tries its old query; modelling may "keep" an alternative that is physically wrong for the scene (reviewed);
checks: a word an edit removed may not survive elsewhere in the question, an edit may not add a number the old phrase did
not have, a query's numbers must be in the question / assumption; the source link prefers a non-forum site; the review does
not judge the sources of constants the repair left unchanged. Unresolved QAs are kept (user 10/12) and analysed.
v11 (pilot j): the old query is exempt from the query-number filter (A|test|6|1 lost its 700c query); an assumption may not
contradict the question's own description (an amber droplet "is water"); the review judges the role of an unchanged constant
when the code edits change its use (a motor efficiency adopted as a drivetrain efficiency); a TYPE_C / TYPE_D tolerance range
the code does not print is rescaled with the answer instead of being dropped.
v12 (pilot k): range values group only if both ends agree within 1 % (U.agree's 3 % put 20-22 oz with 580-620 g, so the old
value lost its own group and became a "new value"); the reported value is the group's own value (v11 pick() could report
another member); forum / Q&A / social pages do not count as supporting sites; an assumption sentence already in the
question is not added again; a re-sourced constant under a new assumption gets its key information from its quote; one
retry of a failed review call (the endpoint returned transient errors).
v13 (pilot l): the removed-word check also reads the answer ("metal rods" in the question but "steel rods" + a steel constant
in the answer passed review), and skips a mention that names another object ("plastic lamp surface" -> "lamp surface" while
"plastic action figure" stays).
v14 (full run, cost): stop after 2 rounds in which every query for the unresolved constant(s) returned results that state no
value of the quantity (51 of 82 "no unique value" QAs: rewording queries never found the number).
v15 (user 10/12, 151 unresolved): a value stated by ONE site / snippet is unique if the query's paraphrase was searched too
and no applicable result of either contradicts it (77 QAs failed only on "stated only by one site"); forum pages count as
sites (no forum check); a range / grade conflict may be resolved by an assumption naming one grade or condition of the SAME
material; the assumption material must be the scene's real material (never another one); a person's sex / age group the
frames show may be used in the query without being written in the question (the solver sees the person); a body quantity
may come from its fraction of a stated body height / mass; "mean +- SD" is the mean; code edits keep every print.
v16 (user 10/13, 57 left): (1) a solution code without any print (6 of the 14 "code failures": every round saw "no output")
gets prints of its top-level results before the repair; (2) the density route also when no unique value was found, and the
size may be added: class B states the size the frames show as an estimate ("about 1.6 cm across"), class A states it as a
given size; (3) benchmark convention (fix10.REVNOTE): the ball's release speed is the athlete's peak wrist speed.
v17 (user 10/13, 48 left): (1) 17 QAs of the data have NO solution code (no code_executor source): every edit failed on
"no output"; the code is now written from the annotated solution and must reproduce the old answer; (2) the substitution
LLM answered "same" for a different value (22 oz vs 580-620 g; 567-650 g "contains" the reference): a forced substitution
then puts the value in (numeric literals and their arithmetic only)."""
import os, sys, json, copy, ast, asyncio, hashlib, re, math, difflib
from collections import Counter, defaultdict
import httpx

ARGV = list(sys.argv)
import fix10 as X  # noqa: E402  (sets sys.argv for the imported modules)
from fix10 import R, W, F9, U, AT  # noqa: E402

ROUNDS = 5
X.REVIEWER = os.environ.get('FIX10_REVIEWER', 'gpt-5.6-terra')  # user 10/12: matched the manual verdicts on 9/11 (gpt-4o ~5/11)
OUT = X.HERE + '/' + os.environ.get('FIX10_OUT', 'fix10_v8.json')
SERP_SEM = None
STAT = W.STAT

# ---------------------------------------------------------------- prompts
PLAN8 = """You repair one QA of a physics video benchmark. Fix ONLY the problems listed below, with the smallest change: every
other word of the question and every other line of the code stays exactly as it is. The video frames are attached.
CLASS %(cls)s - %(clsnote)s
QUESTION TYPE %(qtype)s - %(typenote)s
QUESTION: %(question)s
CODE:
```python
%(code)s
```
CODE OUTPUT: %(out)s
ANSWER: %(answer)s   (range %(range)s)
WEB CONSTANTS (index | quantity | base query | value | status):
%(consts)s
OBJECTS SEEN IN THE VIDEO: %(objects)s
PROBLEMS FOUND BY THE AUDIT (fix each of them, nothing else):
%(problems)s
%(history)s
How to fix each kind of problem:
- Text (a looked-up or hidden value written in the question; tool wording): delete it and keep the sentence grammatical.
  If a deleted value was a web constant, the solver now looks it up: give that constant base-query candidates (below).
- Scene (a phrase the frames contradict or do not show, or an identity they do not show: a league, profession, brand, named
  drill, ...): replace only that phrase with what the frames show, or with neutral wording ("a player", "the ball"). If a
  number in the code came from that phrase (a count, a size), change it in code_edits to what the frames show.
  * a time stamp the frames contradict ("at t=2.1 s"): delete the time and refer to the reading by its quantity ("her
    measured speed of 1.4 m/s"); never move it to another time.
  * a material the frames do not establish ("steel rods") that a web constant depends on: keep the constant and the
    calculation; make the phrase neutral ("metal rods") and give that constant candidates whose assumption names the
    material ("Assume the rods are carbon steel."). Never delete a web constant or the part of the calculation using it.
    Change every mention of that material in the question, not only the first.
  * in each question edit list under "constants" the web constants whose value the removed words determined (removing
    "road" from "road bicycle wheel" affects the wheel-size constant); they are searched again, so give them candidates
    (the assumption may name the size class / type if the frames allow it, e.g. "Assume the bicycle has standard 700c
    wheels.").
  * an assumption must not contradict what the question or the frames say about the object (an amber droplet is not
    "water"; a golden liquid is not "water"). If no plausible material / class gives a unique value, keep the constant as it is.
  * never add a number the old phrase does not have (a count corrected from the frames replaces the old count).
- Modelling (an alternative judged more standard): for each modelling problem choose
  * "adopt": change the code (code_edits) so it computes that alternative; add a question edit only if a reader would
    otherwise not know which model is meant: one sentence stating the physical condition, with no formula, number or method;
  * "state": ONLY if the alternative needs a number that neither the question, the frames nor a web constant gives (e.g. a
    "typical multiplier" between two speeds): keep the code for this item and add one sentence that states the reference's
    choice as a condition ("Assume the ball leaves the hand at the peak wrist speed."), with no number or formula;
  * "keep": ONLY if the alternative is physically wrong for this scene (e.g. it treats a ball in flight as rolling without
    slipping, or adds an effect the scene does not have), or it uses a web constant in a role its quote does not state (a
    motor's efficiency used as a drivetrain / transmission efficiency): code and question stay as they are; say why in "why".
  List your choice per modelling problem in "modelling".
- Web constant marked NEEDS A NEW BASE QUERY: give 6-8 candidates {query, paraphrase, assumption}:
  * query: what a solver would type to look the quantity up. Every word is in the question or in the assumption, or is
    generic ("density", "mass", "typical", "standard", "official").
  * the query must ask for the kind of value it should return: "range" / "tolerance" / "between" only for a quantity whose
    value is a range (a specification interval); a single value otherwise. At most half of the candidates use these
    words: a plain query ("toy marble mass") also finds a stated range.
  * a number in the query must be in the question or the assumption (no product size the question does not state).
  * paraphrase: the same query as another person would type it (other word order, synonyms), with the same qualifiers.
  * assumption: "" for the first 2-3 candidates (the query uses only the question's words); otherwise ONE sentence
    "Assume ...". It may name only the material of a scene object, its official size / grade class or governing standard
    ("Assume the ball is an official size 7 basketball."), or a physical condition (temperature, dry / wet). It must be
    plausible for the frames and contain no number, formula or method. NEVER a product brand or model, a league /
    competition level / profession / nationality / population group of a person, or one particular study.
  * the object and the quantity stay the same.
  * the paraphrase is REQUIRED and must differ from the query: a value stated by only one page is accepted only if the
    paraphrase's results do not contradict it. If the value exists for one size class / specification of the object (an
    M3 washer, an official size 7 ball), a candidate may name that class in the assumption (if the frames allow it).
  * values that differ by grade or condition of the same material (brass alloys, higher vs lower heating value, air-dry vs
    green wood): candidates whose assumption names ONE grade or condition of the SAME material ("Assume the brass is
    cartridge brass.", "Assume the higher heating value applies.", "Assume the wood is air-dried."). The material must be
    the one the question states or the frames clearly show; NEVER replace it by another material.
  * a person: the sex / age group the frames show (an adult man) may be used in the query ("adult male hand mass") but is
    NOT written in the question or the assumption: the solver sees the person. Still never a league / level / profession /
    population group / study.
  * a body segment mass or length published as a fraction of body mass / stature may instead come from that fraction if
    the OLD question states the person's mass / height (never add it yourself): code_edits compute it from a fraction
    variable and the stated value, "quantity": "<segment quantity> as a fraction of <body quantity>", candidates for the
    fraction.
  * an object's mass that varies between products (earlier rounds found disagreeing values or no value) may instead come
    from its material's density and its size: if the OLD question does not state the size, add it with a question edit -
    class B: the size the frames show, as an estimate ("the marble is about 1.6 cm across"); class A: a given size ("a
    1.6 cm marble"); the material must be the real one the frames show. Add code_edits that compute the
    mass from a density variable and that size, give "quantity": "<material> density" and the density's var / code_unit,
    and candidates for the density (the assumption names the material). Nothing else changes.
  The candidates are searched; the first one whose results give one value (or one standard range) is used; if that value
  differs from the old one, the value, code and answer are updated automatically - do not edit them yourself.
- For EVERY web constant give the code variable(s) holding its value and the unit the code uses: "var": ["m"] (["k_low",
  "k_high"] for a range; [] if it is not one plain variable), "code_unit": "kg".
Code edits keep every print statement (an edit that removes the output is rejected).
Never: add a coefficient, multiplier, efficiency, force or input; change the asked quantity or the object; widen a range;
write a looked-up or hidden value into the question; mention tools or searching; edit anything no problem asks for.
Return ONLY JSON:
{"question_edits": [{"old": "<exact text of the QUESTION>", "new": "...", "fixes": "<the problem>", "constants": []}],
 "code_edits": [{"old": "<exact text of the CODE>", "new": "...", "fixes": "<the problem>"}],
 "modelling": [{"problem": "<modelling problem name>", "action": "adopt" | "state" | "keep", "why": "..."}],
 "constants": [{"index": 0, "var": ["..."], "code_unit": "...", "quantity": "<only if changed to a density>",
                "candidates": [{"query": "...", "paraphrase": "...", "assumption": ""}]}]}
(candidates only for the constants that need a new base query; empty lists where nothing is needed; earlier failed
queries are listed under EARLIER ATTEMPTS - give new ones)"""

EXTRACT8 = """A physics question needs the web value of one quantity; the solver searches the web and reads result snippets.
QUESTION: %s
SCENE (objects seen in the video): %s
QUANTITY NEEDED: %s
SCENARIOS (the question as is, or the question plus one assumption sentence):
%s
SEARCH RESULTS (numbered; site in brackets):
%s

List EVERY number in the results that is stated as a value of the QUANTITY NEEDED, for any object; if the same value is in
several results, list it once per result. For each give: the result number; the value (a stated range is one value
[low, high]); the unit; what object / material / size / condition / population it is stated for; and the scenarios it
applies to. A value stated as "mean +- SD" is the mean (one number); a stated min-max range is [low, high]. A value for the
sex / age group of the person the SCENE shows applies to that person. A scenario applies if the value is stated for the
object the question (+ that scenario's assumption)
describes, or for a general class that includes it; it does not apply if the value is for a clearly different object,
size class, grade, material, condition or population. Copy the quote verbatim (max 200 characters, must contain the number).
Return ONLY JSON: {"values": [{"result": 1, "value": <number or [low, high]>, "unit": "...", "refers_to": "...",
"applies": ["S0"], "quote": "..."}]}  (empty list if none)"""

FLIP = """Change this physics solution code so that the web constant "%s" is %s. Change only what this needs and keep every
other line; %s
```python
%s
```
Return ONLY JSON: {"code": "<full code>"}"""

REV8 = """You review a repaired QA of a physics video benchmark. The old QA had the PROBLEMS listed below; the repair may change
only what fixes them. The video frames are attached; trust them.
OBJECTS SEEN (audit): %s
PROBLEMS TO FIX:
%s
OLD QUESTION: %s
NEW QUESTION: %s
OLD ANSWER: %s
NEW ANSWER: %s   (type %s, range %s)
CHANGES MADE: %s
CODE DIFF (old -> new):
%s
WEB CONSTANTS AFTER THE REPAIR (index | base query | value | verbatim quote | sites stating it | change):
%s
NEW CODE OUTPUT: %s
%sWeb-constant problems ("its query also gives ...") are checked automatically: a re-sourced constant passed only because
every value its new base query and its paraphrase return agrees (one site is enough when the paraphrase does not
contradict it). Such a problem is
FIXED; the question does not have to prove the object's size class or grade (a "standard" / "typical" / "regulation" value
whose search is unique is accepted). Judge for web constants only source_mismatch and the assumption sentence.
Judge what the repair changed, and whether each listed problem is fixed. Report only real defects:
- not_fixed: a listed problem is still present.
- assumption: an added "Assume ..." sentence names something other than the material, official size / grade class or
  governing standard of a scene object, or a physical condition (e.g. a product brand or model, a league / competition
  level / profession / population group of a person, one particular study), or it is implausible for the frames, or it
  contradicts the question's own description of the object (e.g. "Assume the droplet is water" for an amber droplet), or
  it names a material that is not the real material of the scene object (the old question states another one, or the
  frames clearly show another one); a grade or condition of the same material is fine.
- source_mismatch: the quote of a re-sourced constant is about a different object, size class, population, condition or
  standard than the question + assumption name; or it is a model input / example of one study rather than the quantity's
  value; or it does not state the value; or the base query does not match the value it returns (it asks for a range /
  tolerance but the value is one number, or it names another quantity); or the code uses the constant in another role
  than its quote states (e.g. a motor's efficiency used as a drivetrain efficiency).
- A modelling problem may be fixed by "state" (one sentence stating the reference's choice as a condition, code kept)
  when the alternative would need a number nobody gives; that is not a defect if the sentence has no number or formula.
  A modelling problem marked "keep" (code unchanged) is correct only if the alternative is really physically wrong for the
  scene; if the alternative is correct and more standard, report not_fixed.
- Constants marked "unchanged" were not repaired: do not judge their sources, but if the code diff changes how one is
  used, judge that role (source_mismatch).
- invented: a changed or new phrase states something about the scene that the frames do not show or contradict.
- added_physics: the change introduces a coefficient, multiplier, efficiency, extra force or effect, or an input that
  neither the old question nor the frames support; or it adopts a model that is physically wrong for the scene (e.g. a
  ball in flight "rolling without slipping").
- answer_change: the answer changes for a reason other than a re-sourced constant value or the modelling fix in PROBLEMS.
- unnatural: an ungrammatical sentence or fragment, contrived wording, a method hint, or the code needs an input the
  question does not give.
- answer: the code does not implement what the new question asks (the printed result is another quantity than the one
  asked), or the answer is physically meaningless (e.g. a negative "how much higher").
Return ONLY JSON: {"defects": [{"criterion": "...", "constant": <index of the web constant concerned or null>,
"detail": "..."}]}"""

CRIT = ('not_fixed', 'assumption', 'source_mismatch', 'invented', 'added_physics', 'answer_change', 'unnatural', 'answer')
GENERIC = set('the a an of for and or to in on at by with from is are was be as its it this that which what how much many typical '
              'standard average official approximate approximately value values range density mass weight size'.split())


# ---------------------------------------------------------------- helpers
async def fllm(prompt):
    """deepseek-flash without thinking (extraction), cached with the fix10 cache"""
    k = hashlib.sha1(('nothink|' + prompt).encode()).hexdigest()
    if k in X._c:
        return X._c[k]
    async with W.SEM:
        for a in range(5):
            body = {'model': W.MODEL, 'temperature': 0.0, 'max_tokens': 16000, 'thinking': {'type': 'disabled'},
                    'messages': [{'role': 'user', 'content': prompt}]}
            try:
                async with httpx.AsyncClient(timeout=600.0) as cl:
                    r = await cl.post(W.LLM_URL, headers={'Authorization': 'Bearer ' + W.LLM_KEY}, json=body)
                    t = r.json()['choices'][0]['message']['content']
                    if t:
                        X._c[k] = t
                        with open(X.CACHE, 'a') as f:
                            f.write(json.dumps({'k': k, 'v': t}, ensure_ascii=False) + '\n')
                        return t
                    STAT['fllm_empty'] += 1
            except Exception as e:
                STAT['fllm_err ' + type(e).__name__] += 1
            await asyncio.sleep(4 * 2 ** a)
    return ''


async def serp(q):
    """raw Serper organic results (link + snippet; the solver sees the snippets) - one call per query, cached"""
    q = q.strip()
    if q not in F9._links and STAT['serper_calls'] >= int(os.environ.get('FIX10_SERPER_MAX', '3000')):
        W.SERPER_FAILS[0] += 3  # budget reached: the run stops, unfinished QAs resume on restart
        return None
    if q not in F9._links:
        async with SERP_SEM:
            if q not in F9._links:
                await asyncio.to_thread(F9.links, q)
                STAT['serper_calls'] += 1
    if q in F9._links:
        return F9._links[q]
    W.SERPER_FAILS[0] += 1
    return None


def apply_edits(text, edits):
    errs = []
    for e in edits:
        if not isinstance(e, dict) or not isinstance(e.get('old'), str) or not isinstance(e.get('new'), str) or not e['old']:
            continue
        n = text.count(e['old'])
        if n != 1:
            errs.append('edit "%s": the old text occurs %d times (must be exactly once, copied verbatim)' % (e['old'][:80], n))
            continue
        text = text.replace(e['old'], e['new'])
    return text, errs


def insert_ass(q, sents):
    """assumption sentences go just before the interrogative sentence"""
    sents = [s.strip() for s in sents if s and s.strip()]
    norm = lambda t: re.sub(r'[^a-z0-9]+', ' ', t.lower()).strip()
    have = {norm(p) for p in re.split(r'(?<=[.!?])\s+', q.strip())}
    out = []
    for x in sents:
        if norm(x) not in have:
            have.add(norm(x))
            out.append(x)
    sents = out
    if not sents:
        return q
    add = ' '.join(s if s[-1] in '.!' else s + '.' for s in sents)
    parts = re.split(r'(?<=[.!?])\s+', q.strip())
    k = max((i for i, p in enumerate(parts) if '?' in p), default=None)
    return q.strip() + ' ' + add if k is None else ' '.join(parts[:k] + [add] + parts[k:])


def same_out(a, b, tol=0.01):
    x, y = AT.nums(a), AT.nums(b)
    return bool(x) and len(x) == len(y) and all(abs(p - q) <= tol * max(abs(p), abs(q)) + 1e-12 for p, q in zip(x, y))


def range_of(rng, out0, out1):
    """the answer_range re-derived from a new code output (same printed positions)"""
    if not (isinstance(rng, list) and len(rng) == 2):
        return None
    idx = W.num_map(' '.join(map(str, rng)), out0)
    on = AT.nums(out1)
    if not idx or len(on) != len(AT.nums(out0)):
        return None
    r = sorted(float('%.4g' % on[i]) for i in idx)
    return r if r[0] < r[1] else None


def _assign(var):
    # one statement per line or several separated by ';'
    return re.compile(r'(^|;)([ \t]*%s[ \t]*=(?!=)[ \t]*)([^\n#;]+?)[ \t]*(?=;|#|$)' % re.escape(var), re.M)


def code_val(code, var):
    m = _assign(var).search(code)
    if not m:
        return None
    try:
        return float(eval(m.group(3), {'__builtins__': {}, 'math': math}))
    except Exception:
        return None


def put(code, var, x):
    m = _assign(var).search(code)
    return code[:m.start(3)] + repr(float('%.6g' % x)) + code[m.end(3):]


def var_info(code, pc, rcc):
    """the code variable(s) of a constant and the code unit -> {'var', 'f', 'dim'} or None (then the LLM substitutes)"""
    vs = [v for v in (pc or {}).get('var') or [] if isinstance(v, str) and v.isidentifier()]
    if len(vs) not in (1, 2) or any(code_val(code, v) is None for v in vs):
        return None
    k = U.key((pc or {}).get('code_unit'))
    raw = sorted(code_val(code, v) for v in vs)
    refs = [U.si(c.get('value'), c.get('unit')) for c in (rcc or {}).get('candidates') or [] if c.get('same')]
    refs = [r for r in refs if r]
    fits = lambda f, dim: not refs or any(r[1] == dim and U.agree(r[0], (raw[0] * f, raw[-1] * f), 0.05) for r in refs)
    f, dim = U._F.get(k, (1, k))
    if not fits(f, dim):
        f, dim = next(((f2, d2) for f2, d2 in sorted(set(U._F.values())) if refs and d2 == refs[0][1] and fits(f2, d2)), (None, None))
        STAT['var_unit_fixed' if f else 'var_unit_mismatch'] += 1
        if f is None:
            return None
    return {'var': vs, 'f': f, 'dim': dim}


def needs_query(c):
    return bool(c.get('no_base_query') or not c.get('reach') or c.get('conflicts_video', c.get('conflicts')))


RANGEQ = re.compile(r'\b(range|ranges|tolerance|tolerances|between|limits?|min(?:imum)?\s*(?:and|/|-)\s*max(?:imum)?|from\s+\w+\s+to)\b', re.I)


def form_mismatch(q, p, si):
    """user 10/12: the query must match the value it returns - a query (or its paraphrase) asking for a range must return a
    range, not a single number"""
    if si[0] == si[1] and (RANGEQ.search(q) or RANGEQ.search(p or '')):
        return 'the query asks for a range but its value is a single number (the query must match the value it returns)'
    return ''


def nums_of(t):
    return {float(x) for x in re.findall(r'(?<![\w.])\d+(?:\.\d+)?', str(t))}


def pick(vs):
    """a value equal to its group's value (the code uses the group's value), preferring a non-forum source"""
    vs = [v for v in vs if close(v['si'], v['g']['si'], 1e-6)] or vs
    return sorted(vs, key=lambda v: bool(QA_RE.search((v.get('link') or '').lower())))[0]


def qwords(t):
    return {w for w in re.findall(r'[a-z][a-z\-]+', str(t).lower()) if len(w) > 3 and w not in GENERIC}


def mentions(w, text, heads):
    """w still describes the edited object: a mention directly before another object's noun does not count"""
    for m in re.finditer(r'\b%s\b(?:\s+([a-z][a-z\-]+))?' % re.escape(w), text):
        nx = m.group(1)
        if nx and qwords(nx) and nx not in heads:  # "plastic action figure" when the edit was about the lamp surface
            continue
        return True
    return False


# ---------------------------------------------------------------- one constant: search -> extract -> decide
async def eval_group(ctx, j, g):
    """put the value group into the code, run it, answer and official ACC vs the current answer"""
    vi, quantity = ctx['vi'].get(j), ctx['quant'][j]
    code = None
    if vi and vi['dim'] == g['dim']:
        lo, hi = g['si'][0] / vi['f'], g['si'][1] / vi['f']
        code = put(ctx['code'], vi['var'][0], (lo + hi) / 2) if len(vi['var']) == 1 else \
            put(put(ctx['code'], vi['var'][0], lo), vi['var'][1], hi)
    if code is None:  # no plain variable / unknown unit: the LLM changes only the literal(s)
        STAT['subst_llm'] += 1
        s = W.pj(await W.llm(W.SUBST % (ctx['code'], quantity, ctx['ki'][j], g['value'], g['unit'], g['quote']))) or {}
        same = s.get('same') and any(U.agree(g['si'], r, 0.01) and (g['si'][0] == g['si'][1]) == (r[0] == r[1]) for r in ctx['old_si'][j])
        code = ctx['code'] if same else s.get('code') if s.get('code') and not s.get('same') and canon2(s['code']) == canon2(ctx['code']) else None
        if code is None and not same:  # v17
            code = await subst_force(ctx['code'], quantity, ctx['ki'][j], g['value'], g['unit'], g['quote'])
    if code is None:
        g['err'] = 'could not put %s %s into the code' % (g['value'], g['unit'])
        return
    out = ctx['out'] if code == ctx['code'] else await asyncio.to_thread(W.run_code, code)
    if out.startswith('Error'):
        g['err'] = 'code with %s %s fails: %s' % (g['value'], g['unit'], out[-120:])
        return
    g['code'], g['out'] = code, out
    g['same'] = same_out(out, ctx['out'])
    g['ans'] = ctx['qa']['answer'] if g['same'] else await W.alt_answer(ctx['qa'], ctx['out'], out)
    g['range'] = ctx['qa'].get('answer_range') if g['same'] else range_of(ctx['qa'].get('answer_range'), ctx['out'], out)
    g['acc'] = 1.0 if g['same'] else (await W.acc(ctx['qa'], g['ans']) if g['ans'] else None)


async def acc_vs(ctx, G, g):
    """ACC of g's answer if G's answer were the reference"""
    if g is G or (g.get('out') and same_out(g['out'], G['out'])):
        return 1.0
    k = (id(G), id(g))
    if k not in ctx['accc']:
        ctx['accc'][k] = await W.acc(dict(ctx['qa'], answer=G['ans'], answer_range=G.get('range')), g['ans']) if g.get('ans') else None
    return ctx['accc'][k]


def close(a, b, tol=0.01):
    """both ends within tol (U.agree allows 3 % for range ends)"""
    return all(abs(x - y) <= tol * max(abs(x), abs(y), 1e-12) for x, y in zip(a, b))


QA_RE = re.compile(F9.FORUM_RE.pattern + r'|study\.com|answers\.com|facebook\.|scribd\.|youtube\.|pinterest\.|tiktok\.|instagram\.')


def support(vs, para=False):
    """sites stating the value. v15 (user 10/12): one site / snippet is enough when the paraphrase was searched as well
    (the caller has already checked that no applicable value of the query or the paraphrase contradicts it)"""
    sites = sorted({v['site'] for v in vs if v['site']})
    if len(sites) >= 2 or any(X.STD.search(v['quote']) for v in vs):
        return sites, True
    if para and vs:
        STAT['single_site'] += 1
        return sites, True
    return sites, False


def say(gs):
    return '; '.join('%s %s ("%s") -> %s' % (g['value'], g['unit'], g['quote'][:110], g.get('ans') or g.get('err')) for g in gs[:3])


async def const_test(ctx, j, cands):
    """all candidates of constant j: search, one extraction, deterministic decision -> (passing list sorted, fail reasons)"""
    qs = []
    for c in cands:
        for q in (c['query'], c.get('paraphrase') or ''):
            if q.strip() and q.strip() not in qs:
                qs.append(q.strip())
    rs = await asyncio.gather(*[serp(q) for q in qs])
    if any(r is None for r in rs):
        return 'SERPER', None
    rows, ids, qids = [], {}, {}
    for q, org in zip(qs, rs):
        qids[q] = []
        for o in org[:10]:
            if not o.get('snippet'):
                continue
            key = (o.get('link'), o['snippet'])
            if key not in ids:
                ids[key] = len(rows) + 1
                rows.append({'n': ids[key], 'link': o.get('link') or '', 'site': X.site(o.get('link')), 'snippet': o['snippet']})
            qids[q].append(ids[key])
    scen = ['']
    for c in cands:
        if c['assumption'] not in scen:
            scen.append(c['assumption'])
    sc_txt = '\n'.join('S%d: %s' % (i, 'the question as is' if not a else 'question + "%s"' % a) for i, a in enumerate(scen))
    block = '\n'.join('[%d] (%s) %s' % (r['n'], r['site'], r['snippet'][:400]) for r in rows)
    ex = W.pj(await fllm(EXTRACT8 % (ctx['qa']['question'], ctx['objects'], ctx['quant'][j], sc_txt, block))) or {}
    vals = []
    for v in ex.get('values') or []:
        if not isinstance(v, dict):
            continue
        n = v.get('result')
        r = rows[n - 1] if isinstance(n, int) and 1 <= n <= len(rows) else None
        if r is None or not W.quote_ok(v.get('quote'), r['snippet'], v.get('value')):
            STAT['value_rejected'] += 1
            continue
        s = U.si(v.get('value'), v.get('unit'))
        if not s:
            continue
        vals.append(dict(n=n, value=v['value'], unit=str(v.get('unit') or ''), refers_to=str(v.get('refers_to') or ''),
                         applies={str(a).strip() for a in v.get('applies') or []}, quote=str(v['quote']), si=s[0], dim=s[1],
                         site=r['site'], link=r['link']))
    groups = []
    for v in vals:
        pt = v['si'][0] == v['si'][1]
        for g in groups:
            if g['dim'] == v['dim'] and (g['si'][0] == g['si'][1]) == pt and close(g['si'], v['si']):
                g['vals'].append(v)
                v['g'] = g
                break
        else:
            g = {'dim': v['dim'], 'si': v['si'], 'value': v['value'], 'unit': v['unit'], 'quote': v['quote'], 'vals': [v]}
            groups.append(g)
            v['g'] = g
    await asyncio.gather(*[eval_group(ctx, j, g) for g in groups])
    ok, fails = [], []
    for order, c in enumerate(cands):
        sid = 'S%d' % scen.index(c['assumption'])
        q, p = c['query'].strip(), (c.get('paraphrase') or '').strip()
        tag = 'query "%s"%s' % (q, ' + "%s"' % c['assumption'] if c['assumption'] else '')
        mine = [v for v in vals if sid in v['applies'] and (v['n'] in qids[q] or (p and v['n'] in qids.get(p, [])))]
        if RANGEQ.search(q) or RANGEQ.search(p):  # a range query is answered only by ranges (a nominal "22 oz" answers another query)
            STAT['rangeq_points_dropped'] += sum(v['si'][0] == v['si'][1] for v in mine)
            mine = [v for v in mine if v['si'][0] != v['si'][1]]
        fromq = [v for v in mine if v['n'] in qids[q]]
        para = bool(p) and p.lower() != q.lower()
        gs = list({id(v['g']): v['g'] for v in mine}.values())
        if not fromq:
            fails.append('%s: its results state no value of the quantity' % tag)
            continue
        if any(g.get('err') or g.get('acc') is None for g in gs):
            fails.append('%s: %s' % (tag, '; '.join(g.get('err') or 'value %s not scorable' % g['value'] for g in gs
                                                  if g.get('err') or g.get('acc') is None)[:300]))
            continue
        old = sorted([v for v in fromq if v['g'].get('same')], key=lambda v: (not any(
            (v['si'][0] == v['si'][1]) == (r[0] == r[1]) and U.agree(v['si'], r, 0.01) for r in ctx['old_si'].get(j, [])), -len(v['g']['vals'])))
        if old:
            bad = [g for g in gs if g['acc'] < .5]
            sites, sup = support([v for v in mine if v['g']['acc'] >= .5], para and not bad)
            if not bad and sup and form_mismatch(q, p, old[0]['si']):
                fails.append('%s: %s' % (tag, form_mismatch(q, p, old[0]['si'])))
                continue
            if not bad and sup:
                v0 = pick([v for v in old if v['g'] is old[0]['g']])
                ok.append(dict(rank=0, order=order, cand=c, G=old[0]['g'], sites=sites, quote=v0['quote'], link=v0['link'],
                               value=v0['value'], unit=v0['unit']))
                continue
            why = ('also returns ' + say(bad)) if bad else 'only %s state(s) the value and the candidate has no paraphrase' % (sites or 'one snippet')
            if bad and not any(not g.get('same') for g in gs):
                fails.append('%s: %s' % (tag, why))
                continue
        # rank 1: one different value, returned uniquely
        cand_g = sorted({id(v['g']): v['g'] for v in fromq if not v['g'].get('same')}.values(),
                        key=lambda g: (-len({v['site'] for v in g['vals']}), -len(g['vals'])))
        if not cand_g:
            fails.append('%s: %s' % (tag, why if old else 'no value'))
            continue
        G = cand_g[0]
        accs = await asyncio.gather(*[acc_vs(ctx, G, g) for g in gs])
        bad = [g for g, a in zip(gs, accs) if a is None or a < .5]
        agree = [v for v in mine if v['g'] not in bad]
        sites, sup = support(agree, para and not bad)
        if bad or not sup:
            fails.append('%s: %s' % (tag, ('the old value: %s; the values disagree: ' % (why if old else 'not returned')) + say([G] + bad)
                                     if bad else 'its value %s %s is stated only by %s' % (G['value'], G['unit'], sites or 'one snippet')))
            continue
        if form_mismatch(q, p, G['si']):
            fails.append('%s: %s' % (tag, form_mismatch(q, p, G['si'])))
            continue
        v0 = pick([v for v in fromq if v['g'] is G])
        ok.append(dict(rank=1, order=order, cand=c, G=G, sites=sites, quote=v0['quote'], link=v0['link'], value=v0['value'],
                       unit=v0['unit']))
    ok.sort(key=lambda x: (x['rank'], bool(x['cand']['assumption']), -len(x['sites']), x['order']))
    return ok, fails


# ---------------------------------------------------------------- one QA
def consts_table(orig, rc, todo):
    ws = F9.wsrc(orig)
    rcc = rc.get('consts') or []
    L = []
    for j, s in enumerate(ws):
        c = rcc[j] if j < len(rcc) else {}
        st = 'OK'
        if j in todo:
            st = 'NEEDS A NEW BASE QUERY: ' + ('no query' if c.get('no_base_query') else 'its query does not return the value'
                                               if not c.get('reach') else 'its results also give other values that change the answer')
        L.append('[%d] %s | %s | %s | %s' % (j, c.get('quantity') or '', s.get('search_query'), str(s.get('key_information'))[:160], st))
    return '\n'.join(L) or '(none)'


SYNTH = """A physics QA lost its solution code. Write the Python code that the annotated solution describes: one variable per
input (video reading, web constant, given), the calculation steps exactly as the solution does them, and a print of every
intermediate result and of the final result(s). Use the solution's input values exactly; no new input, coefficient or model.
QUESTION: %s
ANNOTATED SOLUTION:
%s
SOURCES (video readings / web constants): %s
ANSWER: %s   (answer_range %s)
%sReturn ONLY JSON: {"code": "..."}"""

SUBST2 = """Reference solution code of a physics question:
```python
%s
```
The web-searched quantity "%s" is in this code with the value: %s
It must now use this DIFFERENT value: %s %s (quote: "%s").
Rewrite the code so it uses the new value: change ONLY numeric literals (you may write a unit conversion as arithmetic of
literals, e.g. 22 * 0.0283495); for a range use the midpoint unless the code itself uses a low / high pair (then low and
high). Keep every other character identical.
Return ONLY JSON: {"code": "<full edited code>"}"""


def canon2(code):
    """canon() with arithmetic of numeric literals collapsed: '22 * 0.0283495' counts as one literal"""
    c = W.canon(code)
    prev = None
    while prev != c:
        prev = c
        c = re.sub(r'\(?#\)?(?:\*\*|[*/+\-])\(?#\)?', '#', c)
        c = c.replace('(#)', '#')
    return c


async def subst_force(code, quant, ki, value, unit, quote):
    """v17: the value differs from the code's (the strict check says so): no "same" option, literals only"""
    STAT['subst_forced'] += 1
    for _ in range(2):
        s = W.pj(await W.llm(SUBST2 % (code, quant, ki, value, unit, quote))) or {}
        c = s.get('code')
        if c and c != code and canon2(c) == canon2(code):
            return c
    return None


async def synth_code(orig):
    """v17: write the missing solution code from the annotated solution; it must print the old answer's numbers"""
    srcs = [{k: x.get(k) for k in ('type', 'key_information', 'time_point', 'search_query')} for x in orig['sources']
            if isinstance(x, dict)]
    an = [x for x in AT.nums(str(orig['answer'])) if not (float(x).is_integer() and abs(x) <= 10)] or AT.nums(str(orig['answer']))
    fb = ''
    for _ in range(3):
        g = W.pj(await X.mllm(SYNTH % (orig['question'], '\n'.join(orig.get('annotated_solution') or []),
                                       json.dumps(srcs, ensure_ascii=False)[:4000], orig['answer'], orig.get('answer_range'), fb))) or {}
        c = g.get('code')
        if not c:
            fb = 'YOUR LAST ANSWER HAD NO CODE.\n'
            continue
        out = await asyncio.to_thread(W.run_code, c)
        on = AT.nums(out)
        miss = [x for x in an if not any(abs(x - o) <= 0.015 * max(abs(x), 1e-9) for o in on)]
        if out.startswith('Error') or miss:
            fb = 'YOUR LAST CODE %s. Follow the solution steps exactly.\nYOUR LAST CODE:\n%s\n' % (
                'fails: ' + out[-300:] if out.startswith('Error') else 'does not print the answer number(s) %s (output: %s)'
                % (miss, out[-600:]), c[:3000])
            continue
        STAT['code_synthesized'] += 1
        return c, out
    return None, None


def printify(code):
    """v16: a code that prints nothing prints its top-level results (last expression, assigned names)"""
    import ast as _a
    try:
        t = _a.parse(code)
    except SyntaxError:
        return code
    if any(isinstance(n, _a.Call) and getattr(n.func, 'id', '') == 'print' for n in _a.walk(t)):
        return code
    lines = code.rstrip().split(chr(10))
    if t.body and isinstance(t.body[-1], _a.Expr) and t.body[-1].end_lineno == len(lines):
        last = t.body[-1]
        lines = lines[:last.lineno - 1] + ['print(%s)' % _a.get_source_segment(code, last).strip()]
    names = []
    for n in t.body:
        for tg in (n.targets if isinstance(n, _a.Assign) else [n.target] if isinstance(n, (_a.AnnAssign, _a.AugAssign)) else []):
            for x in ([tg] if isinstance(tg, _a.Name) else getattr(tg, 'elts', [])):
                if isinstance(x, _a.Name) and x.id not in names and not x.id.startswith('_'):
                    names.append(x.id)
    lines += ['print(%r, %s)' % (n + ' =', n) for n in names]
    STAT['printified'] += 1
    return chr(10).join(lines) + chr(10)


async def fix_one(item, rc, sib):
    orig = item['qa']
    if isinstance(orig['sources'], str):
        orig['sources'] = ast.literal_eval(orig['sources'])
    cls, qt0 = item['cls'], orig['question_type']  # (v17 may replace orig by a copy with a synthesized code)
    code0 = F9.code_of(orig)
    if not code0.strip():  # v17: the QA has no solution code
        code0, out0 = await synth_code(orig)
        if code0 is None:
            return {'id': item['id'], 'split': item['split'], 'cls': cls, 'orig_labels': rc.get('labels2', rc['labels']),
                    'rounds': [], 'version': 'v17', 'status': 'unresolved', 'stop': 'no solution code; synthesis failed',
                    'history': ['the QA has no solution code and the synthesized code does not reproduce the answer']}
        orig = copy.deepcopy(orig)
        orig['sources'] = list(orig['sources']) + [{'type': 'code_executor', 'code': code0, 'key_information': out0,
                                                    'key_information_annotated': None, 'synthesized': True}]
    out0 = await asyncio.to_thread(W.run_code, code0)
    if out0.startswith('Error: no output'):
        code0 = printify(code0)
        out0 = await asyncio.to_thread(W.run_code, code0)
    ws0 = F9.wsrc(orig)
    rcc = rc.get('consts') or []
    todo = [j for j in range(len(ws0)) if j < len(rcc) and needs_query(rcc[j])]
    fr, _ = await asyncio.to_thread(R.frames, R.video_path(item), orig['question'])
    fkey = '%s|%s' % (R.video_path(item), ','.join('%.2f' % t for t, _ in (fr or [])))
    objects = '; '.join('%s: %s' % (o.get('object'), o.get('seen')) for o in ((rc.get('video') or {}).get('objects') or [])
                        if isinstance(o, dict))[:1500] or '(none)'
    problems = X.problems_txt(rc)
    malt = [m for m in rc.get('model') or [] if m.get('acc') is not None and m['acc'] < .5 and m.get('verdict') == 'X']
    rec = {'id': item['id'], 'split': item['split'], 'cls': cls, 'orig_labels': rc.get('labels2', rc['labels']), 'rounds': [],
           'version': 'v17'}
    hist, banned, plan = [], set(), None
    prev_ok = defaultdict(list)  # v10: candidates that passed in an earlier round are offered again
    replan = True
    for k in range(ROUNDS):
        rd = {'round': k + 1}
        rec['rounds'].append(rd)
        if replan or plan is None:
            d = dict(cls=cls, clsnote=X.CLS_NOTE[cls], qtype=qt0, typenote=X.TYPE_NOTE[qt0], question=orig['question'],
                     code=code0[:6000], out=out0[-1500:], answer=orig['answer'], range=orig.get('answer_range'),
                     consts=consts_table(orig, rc, todo), objects=objects, problems=problems,
                     history=('EARLIER ATTEMPTS FAILED (do not repeat them):\n' + '\n'.join(hist[-6:]) + '\n' + (
                         'QUERIES ALREADY TRIED (not unique; do not repeat): %s\n' % '; '.join(
                             '[%d] %s' % jq for jq in sorted(banned))[:2500] if banned else '')) if hist else '')
            plan = W.pj(await X.mllm(PLAN8 % d, fr, fkey)) or {}
            STAT['plan_calls'] += 1
        rd['plan'] = plan
        replan = True
        q1, e1 = apply_edits(orig['question'], plan.get('question_edits') or [])
        code1, e2 = apply_edits(code0, plan.get('code_edits') or [])
        if e1 or e2 or not plan:
            hist.append('ROUND %d: %s' % (k + 1, '; '.join(e1 + e2) or 'no plan returned'))
            rd['error'] = hist[-1]
            continue
        out1 = out0 if code1 == code0 else await asyncio.to_thread(W.run_code, code1)
        if out1.startswith('Error'):
            hist.append('ROUND %d: the edited code fails: %s' % (k + 1, out1[-200:]))
            rd['error'] = hist[-1]
            continue
        qa1 = copy.deepcopy(orig)
        qa1['question'] = q1
        if code1 != code0:
            qa1['answer'] = await W.alt_answer(orig, out0, out1)
            qa1['answer_range'] = range_of(orig.get('answer_range'), out0, out1) if orig.get('answer_range') else None
        # constants to (re)test: the flagged ones + any whose query words the edits removed from the question
        pcs = {c['index']: c for c in plan.get('constants') or [] if isinstance(c, dict) and isinstance(c.get('index'), int)}
        test, qnum_fail = {}, []
        # constants whose value the removed words determined (the plan names them per question edit)
        affected = {j for e in plan.get('question_edits') or [] if isinstance(e, dict) for j in (e.get('constants') or [])
                    if isinstance(j, int) and 0 <= j < len(ws0)}
        qn_ok = lambda c: not (nums_of(c['query'] + ' ' + c['paraphrase']) - nums_of(q1 + ' ' + c['assumption']))
        for j, s in enumerate(ws0):
            pc = pcs.get(j) or {}
            cands = [dict(query=str(x.get('query') or '').strip(), paraphrase=str(x.get('paraphrase') or '').strip(),
                          assumption=str(x.get('assumption') or '').strip()) for x in pc.get('candidates') or [] if isinstance(x, dict)]
            for c in cands:  # a number in a query must be in the question or its assumption (no "8x8x16" the question lacks)
                if c['query'] and not qn_ok(c):
                    qnum_fail.append('[%d] "%s"' % (j, c['query']))
                    banned.add((j, c['query']))
            cands = [c for c in cands if c['query'] and (j, c['query']) not in banned][:8]
            seen = {c['query'] for c in cands}
            cands = [c for c in prev_ok[j] if (j, c['query']) not in banned and c['query'] not in seen and qn_ok(c)] + cands
            lost = q1 != orig['question'] and (qwords(s.get('search_query')) & qwords(orig['question'])) - qwords(q1)
            # the reference value was valid only through the question's context: re-test only if the edit removed words of
            # this constant (its query, quantity, value text); a changed time stamp or count removes none
            rcj = rcc[j] if j < len(rcc) else {}
            ctxdep = bool((qwords(orig['question']) - qwords(q1)) & (qwords(s.get('search_query')) | qwords(rcj.get('quantity'))
                                                                     | qwords(s.get('key_information')))) and any(
                x.get('valid_verdict') == 'reference' for x in (rcc[j] if j < len(rcc) else {}).get('candidates') or [])
            oldq = dict(query=str(s.get('search_query') or '').strip(), paraphrase='', assumption='')
            if j not in todo and oldq['query'] and (j, oldq['query']) not in banned \
                    and oldq['query'] not in {c['query'] for c in cands} and (cands or lost or ctxdep or j in affected):
                cands = [oldq] + cands  # its old query was fine: try it first in the new question
            if j in todo or cands:
                test[j] = cands
        if qnum_fail:
            hist.append('ROUND %d: queries with a number the question does not state (rejected): %s' % (k + 1, '; '.join(qnum_fail)[:600]))
        missing = [j for j in sorted(set(todo) | affected) if not test.get(j)]
        if missing:
            hist.append('ROUND %d: no base-query candidates for web constant(s) %s that need a new query' % (k + 1, missing))
            rd['error'] = hist[-1]
            continue
        # a constant re-defined by the plan (object mass -> material density x stated size): no old value to compare with
        redef = {j for j in test if str((pcs.get(j) or {}).get('quantity') or '').strip() and code1 != code0}
        ctx = {'qa': qa1, 'code': code1, 'out': out1, 'objects': objects, 'accc': {},
               'old_si': {j: [] if j in redef else [r[0] for r in (U.si(c.get('value'), c.get('unit')) for c in (rcc[j] if j < len(rcc) else {}).get('candidates') or []
                                                                    if c.get('same')) if r] for j in test},
               'quant': [str(pcs[j]['quantity']).strip() if j in redef else
                         (rcc[j].get('quantity') if j < len(rcc) else '') or str(ws0[j].get('key_information')) for j in range(len(ws0))],
               'ki': [str(s.get('key_information')) for s in ws0],
               'vi': {j: var_info(code1, pcs.get(j), {} if j in redef else rcc[j] if j < len(rcc) else {}) for j in test}}
        res = await asyncio.gather(*[const_test(ctx, j, test[j]) for j in test])
        if any(r[0] == 'SERPER' for r in res):
            return None
        chosen, cfail = {}, {}
        for j, (ok, fails) in zip(test, res):
            cfail[j] = fails
            if ok:
                chosen[j] = ok[0]
                chosen[j]['alts'] = len(ok) - 1
                if ok[0]['cand']['query'] not in {c['query'] for c in prev_ok[j]}:
                    prev_ok[j].append(ok[0]['cand'])
        rd['tested'] = {j: {'pass': [(x['rank'], x['cand']['query'], x['cand']['assumption'], x['value'], x['unit'], x['sites'])
                                     for x in r[0]], 'fail': r[1]} for j, r in zip(test, res)}
        for j, (ok, fails) in zip(test, res):  # queries whose values failed are not searched / extracted again (cost)
            if j in redef:
                continue
            for f in fails:
                m = re.match(r'query "(.*?)"(?: \+ "|: )', f)
                if m and not re.search(r'could not put|fails:|not scorable', f):
                    banned.add((j, m.group(1)))
        nf = [j for j in test if j not in chosen]
        if nf:
            hist.append('ROUND %d: no candidate gives a unique value for web constant(s) %s:\n  %s' % (
                k + 1, nf, '\n  '.join(f[:300] for j in nf for f in cfail[j][:6])))
            rd['error'] = 'no unique value for %s' % nf
            rd['novalue'] = all(cfail[j] and all('state no value' in f for f in cfail[j]) for j in nf)
            if len(rec['rounds']) >= 2 and all(r.get('novalue') for r in rec['rounds'][-2:])                     and len({r.get('error') for r in rec['rounds'][-2:]}) == 1:
                rec['stop'] = 'no search result states web constant(s) %s in 2 rounds' % nf
                STAT['stop_novalue'] += 1
                break
            nu = [r.get('error') for r in rec['rounds'][-3:]]
            if len(nu) == 3 and len(set(nu)) == 1:  # 3 rounds in a row on the same constant(s): the value varies (cost)
                rec['stop'] = 'the value of web constant(s) %s varies: no unique value in 3 rounds' % nf
                break
            continue
        # the repaired QA: assumptions, re-sourced values, code, answer
        q2 = insert_ass(q1, [c['cand']['assumption'] for j, c in sorted(chosen.items())])
        code2, flips, qt, serr = code1, [], qt0, None
        for j, c in sorted(chosen.items()):
            if c['rank'] != 1:
                continue
            vi = ctx['vi'].get(j)
            G = c['G']
            if vi and vi['dim'] == G['dim']:
                lo, hi = G['si'][0] / vi['f'], G['si'][1] / vi['f']
                rng_new, rng_code = G['si'][0] != G['si'][1], len(vi['var']) == 2
                if rng_new and not rng_code:
                    flips.append((j, 'the standard range [%.6g, %.6g] %s' % (lo, hi, pcs[j].get('code_unit')),
                                  'compute the result for the low and for the high end of the range and print both as "low: ..." '
                                  'and "high: ..." after the existing prints', 'TYPE_B'))
                    code2 = put(code2, vi['var'][0], (lo + hi) / 2)
                elif rng_code and not rng_new:
                    code2 = put(put(code2, vi['var'][0], lo), vi['var'][1], lo)
                    flips.append((j, 'the single value %.6g %s' % (lo, pcs[j].get('code_unit')),
                                  'compute one result from this single value and print it (no low / high)', 'TYPE_A'))
                else:
                    code2 = put(code2, vi['var'][0], (lo + hi) / 2) if not rng_code else put(put(code2, vi['var'][0], lo), vi['var'][1], hi)
            else:
                s = W.pj(await W.llm(W.SUBST % (code2, ctx['quant'][j], ctx['ki'][j], G['value'], G['unit'], G['quote']))) or {}
                if s.get('code') and not s.get('same') and canon2(s['code']) == canon2(code2):
                    code2 = s['code']
                else:  # v17: a rank-1 value differs by construction; "same" is not believed
                    c2 = await subst_force(code2, ctx['quant'][j], ctx['ki'][j], G['value'], G['unit'], G['quote'])
                    if not c2:
                        serr = 'ROUND %d: could not put the new value of constant [%d] into the code' % (k + 1, j)
                        break
                    code2 = c2
        if serr:
            hist.append(serr)
            rd['error'] = serr
            continue
        for j, what, how, t in flips:
            fc = W.pj(await X.mllm(FLIP % (ctx['quant'][j], what, how, code2))) or {}
            if fc.get('code'):
                code2 = fc['code']
                if qt0 in ('TYPE_A', 'TYPE_B'):
                    qt = t
        out2 = out1 if code2 == code1 else await asyncio.to_thread(W.run_code, code2)
        if out2.startswith('Error'):
            hist.append('ROUND %d: the code with the new value(s) fails: %s' % (k + 1, out2[-200:]))
            rd['error'] = hist[-1]
            continue
        new = copy.deepcopy(orig)
        new['question'] = q2
        new_ws = []
        for j, s in enumerate(ws0):
            s = dict(s)
            c = chosen.get(j)
            if c:
                vtxt = '-'.join(map(str, c['value'])) if isinstance(c['value'], list) else str(c['value'])
                s.update(search_query=c['cand']['query'], verified_quote=c['quote'], url=c['link'] or s.get('url'))
                if c['rank'] == 0 and c['cand']['assumption']:  # same number, but the old text may name another material
                    s['key_information'] = '%s (%s %s)' % (re.sub(r'\s+', ' ', c['quote']).strip()[:160], vtxt, c['unit'])
                if c['rank'] == 1:
                    s['key_information'] = '%s = %s %s' % (ctx['quant'][j], vtxt, c['unit'])
                    vi = ctx['vi'].get(j)
                    if vi and U.key(c['unit']) != U.key(pcs[j].get('code_unit')):  # the value in the code's unit
                        lo, hi = c['G']['si'][0] / vi['f'], c['G']['si'][1] / vi['f']
                        s['key_information'] += ' (= %s %s)' % ('%.4g' % lo if lo == hi else '%.4g-%.4g' % (lo, hi),
                                                                pcs[j].get('code_unit') or '')
            new_ws.append(s)
        others = [s for s in orig['sources'] if not (isinstance(s, dict) and s.get('type') in ('web_search', 'code_executor'))]
        new['sources'] = others + new_ws + [{'type': 'code_executor', 'code': code2, 'key_information': out2,
                                             'key_information_annotated': None}]
        new['question_type'] = qt
        if cls == 'B' and orig.get('hidden_cv_chosen'):  # B hidden values follow the web values (the entry of the old value)
            h = list(orig['hidden_cv_chosen'])
            for j, c in chosen.items():
                if new_ws[j].get('key_information') == ctx['ki'][j]:
                    continue
                on = AT.nums(ctx['ki'][j])
                i = next((i for i, x in enumerate(h) if str(x).strip() == ctx['ki'][j].strip()), None)
                if i is None:
                    i = next((i for i, x in enumerate(h) if on and AT.nums(x)[:len(on)] == on), None)
                if i is not None:
                    h[i] = new_ws[j]['key_information']
            new['hidden_cv_chosen'] = h
        unchanged = qt == qt0 and same_out(out2, out0) and len(AT.nums(out2)) == len(AT.nums(out0))
        if unchanged and q1 == orig['question']:
            # output and question text unchanged (only an assumption / a new query): keep answer, range and solution
            # (no FINAL call; v8 lost a TYPE_D range here)
            STAT['final_skipped'] += 1
        else:
            qtf = qt if qt == qt0 else '%s (changed from %s: the web constant is now %s)' % (
                qt, qt0, 'a standard range' if qt == 'TYPE_B' else 'a single value')
            fin = W.pj(await X.mllm(F9.FINAL % (q2, qtf, '\n'.join(orig.get('annotated_solution') or []), orig['answer'],
                                                '\n'.join('%s | %s | "%s"' % (s.get('search_query'), s.get('key_information'),
                                                                              str(s.get('verified_quote'))[:200]) for s in new_ws),
                                                code2, out2[:3000], orig.get('answer_range')))) or {}
            if not fin.get('answer') or not isinstance(fin.get('annotated_solution'), list):
                hist.append('ROUND %d: final answer failed' % (k + 1))
                rd['error'] = hist[-1]
                continue
            rng = None
            if qt == 'TYPE_B' or (qt != 'TYPE_A' and orig.get('answer_range') is not None):
                rng = orig.get('answer_range') if unchanged else fin.get('answer_range')
                if not (isinstance(rng, list) and len(rng) == 2) and orig.get('answer_range') and qt == qt0:
                    rng = range_of(orig['answer_range'], out0, out2)
            if rng is None and orig.get('answer_range') and qt == qt0 and qt in ('TYPE_C', 'TYPE_D'):
                # a tolerance range around the old answer that the code does not print: scale it with the answer
                a0, a1 = AT.nums(orig['answer'])[:1], AT.nums(str(fin['answer']))[:1]
                o = [AT.nums(str(x))[:1] for x in orig['answer_range'][:2]]
                o = [x[0] for x in o] if len(o) == 2 and all(o) else [math.inf, -math.inf]
                if a0 and a1 and a0[0] > 0 and a1[0] > 0 and o[0] <= a0[0] <= o[1]:
                    f = a1[0] / a0[0]
                    rng = [float('%.4g' % (o[0] * f)), float('%.4g' % (o[1] * f))]
                    STAT['range_rescaled'] += 1
            new.update(answer=str(fin['answer']), annotated_solution=[str(x) for x in fin['annotated_solution']], answer_range=rng)
            new['total_steps'] = len(new['annotated_solution'])
        changed_val = any(c['rank'] == 1 for c in chosen.values())
        changes = (['question: "%s" -> "%s" (%s)' % (e['old'], e['new'], e.get('fixes')) for e in plan.get('question_edits') or []
                    if isinstance(e, dict) and e.get('old')]
                   + ['code: %s' % e.get('fixes') for e in plan.get('code_edits') or [] if isinstance(e, dict) and e.get('old')]
                   + ['modelling "%s": %s (%s)' % (x.get('problem'), x.get('action'), str(x.get('why') or '')[:200])
                      for x in plan.get('modelling') or [] if isinstance(x, dict) and malt]
                   + ['constant [%d]: %s, base query "%s"%s' % (
                       j, 'same value' if c['rank'] == 0 else 'NEW VALUE %s %s (old: %s)' % (c['value'], c['unit'], ctx['ki'][j][:100]),
                       c['cand']['query'], ' + "%s"' % c['cand']['assumption'] if c['cand']['assumption'] else '')
                      for j, c in sorted(chosen.items())]
                   + (['type %s -> %s' % (qt0, qt)] if qt != qt0 else []))
        # deterministic acceptance
        info = {'chosen': [{'value': (chosen[j]['value'] if j in chosen else s.get('key_information')),
                            'quote': s.get('verified_quote') or ''} for j, s in enumerate(new_ws)]}
        P = [p['text'] for p in F9.static_problems(new, orig, info) if not p['text'].startswith(('The original TYPE_', 'Forum / Q&A'))]
        # the old TYPE_B range already passed construction (its source may be a reading range): check only a new TYPE_B
        P += [p for p in X.form_problems(new, orig, info, cls) if not (p.startswith('TYPE_B: the range') and qt0 == 'TYPE_B')]
        # every web constant stays in the calculation (#ws fixes the difficulty; v8 dropped the steel rods / the friction)
        for j in range(len(ws0)):
            vs = [v for v in ((pcs.get(j) or {}).get('var') or []) if isinstance(v, str) and v.isidentifier()]
            used = lambda c, v: len(re.findall(r'(?<![\w.])%s(?!\w)' % re.escape(v), re.sub(r'#.*', '', c))) >= 2
            if vs and all(code_val(code0, v) is not None and used(code0, v) for v in vs) and j not in redef \
                    and not all(used(code2, v) for v in vs):
                P.append('The repair drops web constant [%d] (%s) from the calculation; every web constant must stay used (the '
                         'number of web constants fixes the difficulty). If a phrase about it was unsupported, keep the constant '
                         'and name the needed material / class in an "Assume ..." sentence.' % (j, ctx['quant'][j]))
        for m in re.finditer(r'(?<![\d.])(\d+(?:\.\d+)?)\s*(?:–|-|—|to)\s*(\d+(?:\.\d+)?)(?![\d.])', new['answer']):
            if float(m.group(1)) == float(m.group(2)) and float(m.group(1)) != 0:
                P.append('The answer gives a degenerate range "%s"; a single value is one number (compute it once).' % m.group(0))
                break
        # v10: a word an edit removed must not survive elsewhere; an edit adds no number its old phrase lacked
        body = ' '.join(x for x in re.split(r'(?<=[.!?])\s+', q2) if not x.strip().lower().startswith(('assume', 'suppose')))
        for e in plan.get('question_edits') or []:
            if not (isinstance(e, dict) and isinstance(e.get('old'), str) and isinstance(e.get('new'), str)):
                continue
            rem = (qwords(e['old']) - qwords(e['new'])) & qwords(e.get('fixes'))
            heads = (qwords(e['old']) | qwords(e['new'])) - rem
            text = (body + ' ' + str(new['answer'])).lower()
            left = sorted(w for w in rem if mentions(w, text, heads))
            if left:
                P.append('The edit removed %s from "%s" but the question or the answer still says %s; change every mention '
                         '(code print labels too). If a web constant needs that material, keep the material and name it in an '
                         '"Assume ..." sentence instead of removing it.' % (left, e['old'][:80], left))
            addn = nums_of(e['new']) - nums_of(e['old']) - nums_of(orig['question'])
            if addn and not nums_of(e['old']):
                P.append('The edit adds the number(s) %s ("%s"); never add a number the old phrase does not have.' % (
                    sorted(addn), e['new'][:100]))
        cond = X.cond_nums(new['question'])
        st = [s for s in R.static(new) if not s.get('values') or [x for x in s['values']
                                                                  if not any(abs(x - c) <= 1e-9 * max(abs(c), 1e-12) for c in cond)]]
        P += ['The question still has %s %s' % (s['kind'], s.get('values') or s.get('text') or '') for s in st]
        if cls == 'A':  # readings the old code used must stay used
            inc = lambda h, c: any(abs(h - x) <= 0.002 * max(abs(h), 1e-9) for x in AT.nums(c))
            miss = [h for h in F9.hidden_nums(new) if inc(h, code0) and not inc(h, code2)]
            if miss:
                P.append('The code does not use the video reading(s) %s.' % miss)
        if not changed_val and code1 == code0:
            a = await W.acc(orig, new['answer'])
            if a is None or a < .5:
                P.append('The answer changed (%s -> %s) although no value and no model changed.' % (orig['answer'], new['answer']))
        if malt:  # per modelling problem: adopt (code computes the alternative) or state (reference choice as a condition)
            acts = {}
            for x in plan.get('modelling') or []:
                if isinstance(x, dict):
                    acts[str(x.get('problem') or '').strip().lower()] = str(x.get('action') or '').strip().lower()
            act = lambda m: next((a for p, a in acts.items() if p and (p in str(m.get('aspect')).lower() or str(m.get('aspect')).lower() in p)),
                                 'adopt')
            adopt = [m for m in malt if act(m) not in ('state', 'keep')]
            nstate = sum(act(m) == 'state' for m in malt)
            if not adopt and code1 != code0:
                P.append('All modelling problems were fixed by "state" / "keep", so the code must stay unchanged.')
            elif len(adopt) == 1 and code1 == code0:
                P.append('Modelling problem "%s" is marked adopt but the code is unchanged.' % adopt[0].get('aspect'))
            elif len(adopt) == 1:  # with several adopted the combined answer is not known: the review judges it
                m0 = adopt[0]
                a = await W.acc(dict(orig, answer=m0.get('alt_answer'), answer_range=AT.nums(m0.get('alt_answer'))[1:3] or None
                                     if orig['question_type'] == 'TYPE_B' else None), qa1['answer'])
                if a is not None and a < .5:
                    P.append('The code edit gives %s, but the standard alternative ("%s") gives %s.' % (qa1['answer'], m0.get('alternative'),
                                                                                                       m0.get('alt_answer')))
            if nstate and q1 == orig['question']:
                P.append('A modelling problem fixed by "state" needs one sentence in the question stating the reference choice.')
        rd.update(question=new['question'], answer=new['answer'], answer_range=new.get('answer_range'), changes=changes, problems=P)
        if P:
            hist.append('ROUND %d (question: %s | answer %s): %s' % (k + 1, q2, new['answer'], ' | '.join(p[:250] for p in P)))
            continue
        # review of the diff (gpt-5.6-terra, frames)
        cdiff = '\n'.join(difflib.unified_diff(code0.splitlines(), code2.splitlines(), lineterm='', n=0))[:3500] or '(code unchanged)'
        ctab = '\n'.join('[%d] %s | %s | "%s" | %s | %s' % (
            j, s.get('search_query'), s.get('key_information'), str(s.get('verified_quote'))[:220],
            ', '.join(chosen[j]['sites']) if j in chosen else '-',
            'unchanged' if j not in chosen else 'same value, new query' if chosen[j]['rank'] == 0 else 'NEW VALUE')
            for j, s in enumerate(new_ws)) or '(none)'
        rprompt = REV8 % (objects, problems, orig['question'], new['question'], orig['answer'], new['answer'], qt,
                          new.get('answer_range'), '; '.join(changes) or 'none', cdiff, ctab, out2[-1200:], X.REVNOTE[cls])
        rv = W.pj(await X.rmllm(rprompt, fr, fkey))
        if not isinstance(rv, dict):  # transient endpoint errors: one more try before spending a round
            STAT['rev_retry'] += 1
            await asyncio.sleep(60)
            rv = W.pj(await X.rmllm(rprompt, fr, fkey))
        if not isinstance(rv, dict) or not isinstance(rv.get('defects'), list):
            hist.append('ROUND %d: no valid review returned' % (k + 1))
            rd['error'] = hist[-1]
            continue
        dfs = [x for x in rv['defects'] if isinstance(x, dict) and x.get('criterion') in CRIT and not (
            x.get('criterion') == 'source_mismatch' and isinstance(x.get('constant'), int) and x['constant'] not in chosen
            and code1 == code0)]
        for x in dfs:  # a flagged re-sourced value is not offered again
            j = x.get('constant')
            if isinstance(j, int) and j in chosen and x.get('criterion') in ('source_mismatch', 'assumption'):
                banned.add((j, chosen[j]['cand']['query']))
                prev_ok[j] = [c for c in prev_ok[j] if c['query'] != chosen[j]['cand']['query']]
        rd['review'] = dfs
        if not dfs:
            rec.update(status='fixed', mode='v17', qa=new, qtype_change=qt != qt0, changes=changes,
                       constants=[dict(index=j, rank=c['rank'], query=c['cand']['query'], assumption=c['cand']['assumption'],
                                       value=c['value'], unit=c['unit'], old_value=ctx['ki'][j], quote=c['quote'], url=c['link'],
                                       support=c['sites'], cand_fail=cfail.get(j)) for j, c in sorted(chosen.items())])
            return rec
        hist.append('ROUND %d (question: %s | answer %s) review: %s' % (
            k + 1, q2, new['answer'], ' | '.join('%s%s: %s' % (x['criterion'], ' [constant %s]' % x['constant']
                                                               if isinstance(x.get('constant'), int) else '', str(x.get('detail'))[:250]) for x in dfs)))
        # a defect of one re-sourced constant only -> its next candidate, same plan
        cj = {x.get('constant') for x in dfs}
        if all(isinstance(j, int) and j in chosen and chosen[j]['alts'] > 0 for j in cj):
            for j in cj:
                banned.add((j, chosen[j]['cand']['query']))
            replan = False
    rec['status'] = 'unresolved'
    rec['history'] = hist
    return rec


async def main():
    global SERP_SEM
    W.SEM, W.SSEM, X.MSEM, SERP_SEM = asyncio.Semaphore(48), asyncio.Semaphore(16), asyncio.Semaphore(32), asyncio.Semaphore(16)
    X.RN_SEM = asyncio.Semaphore(16)
    F9.FSEM = asyncio.Semaphore(16)
    rc = {r['id']: r for r in json.load(open(X.HERE + '/recount_v2.json'))}
    items = R.load()
    ids = set(json.load(open(ARGV[1])) if len(ARGV) > 1 else [i for i, r in rc.items() if r.get('needs_fix2')])
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [i for i in items if i['id'] in ids and i['id'] not in done]
    print('QAs %d todo %d reviewer %s' % (len(ids), len(todo), X.REVIEWER), flush=True)
    qsem = asyncio.Semaphore(int(os.environ.get('FIX10_PAR', '48')))
    stop = []

    async def run(i):
        if stop:
            return
        async with qsem:
            if stop:
                return
            try:
                r = await fix_one(i, rc[i['id']], [])
            except Exception as e:  # not saved: retried when the run is restarted
                import traceback
                STAT['qa_exception'] += 1
                print('EXCEPTION %s %s\n%s' % (i['id'], e, traceback.format_exc()[-1500:]), flush=True)
                return
        if r is None or W.SERPER_FAILS[0] >= 3:
            stop.append(i['id'])
            return
        done[r['id']] = r
        json.dump(done, open(OUT, 'w'), ensure_ascii=False, indent=1)
        print('progress %d/%d %s %s' % (len(done), len(ids), dict(Counter(x['status'] for x in done.values())), dict(STAT)), flush=True)
    await asyncio.gather(*[run(i) for i in todo])
    json.dump(F9._links, open(F9.LINKS, 'w'), ensure_ascii=False)
    if stop:
        print('SERPER FAILING - stop', flush=True)
        sys.exit(3)
    C = Counter((r['split'], r['cls'], r['status']) for r in done.values())
    for k in sorted(C):
        print(k, C[k])

if __name__ == '__main__':
    asyncio.run(main())
