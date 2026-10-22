# Provenance (TriPhysBench release): fix_code/evaluation/fix9/fix9.py
# Role: fix9 pilot (10-11, never applied to the data); imported by fix10 for its static checks.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix9 (10/11, user: fix every problematic QA in one pass and make sure no problem remains; no TYPE_B range covering the
credible values, no acceptable-answer set). Every QA of data/v9 (version 07, train + test, A + B) is checked; the ones that
do not pass are repaired and re-checked with the same checker until they pass (max ROUNDS repair rounds).
Pass = the step-1 criteria (wellposed/wellposed_check.py, checker deepseek-flash, unchanged):
  no annotation_error / web_ambiguous / model_ambiguous; every web constant reachable from question-only queries, or found
  by the annotator's own query whose terms are all stated / implied by the question (qual_check.QUAL from_question,
  basereach); plus static checks: no tool / search wording in the question, no hidden or looked-up value written into the
  question, TYPE_B has a [low, high] range printed by the code, every verified_quote is in the Serper results of its query.
Repair (fixer deepseek-flash, user 10/11; difficulty and type distribution kept):
  PLAN    edited question + per web constant keep / qualify / resource / switch + 2 solver queries
  SOURCE  Serper (eval formatting) on the queries -> the fixer picks the value with a verbatim quote (post-checked)
  APPLY   new code + key_information (+ B hidden_cv_chosen) -> code executed
  FINAL   annotated solution, answer, answer_range from the code output
Allowed: qualifiers / standards, re-sourced values, switch to well-defined constants, explicit modelling assumptions.
Forbidden: web values / tool names in the question; ranges or answer sets widened to cover alternative values.
Kept (user 10/11): difficulty = same number of web constants and of hidden values (A: hidden video readings 0/1/2 = easy /
medium / hard; B: hidden web specs), question_type unchanged (TYPE_B -> TYPE_A only as a last resort in the last round).
Output fix9.json {id: {status clean|fixed|unresolved, orig_problems, rounds, qa}}; caches fix9_llm_cache.jsonl,
serper_links.json.  Usage (server, fc/evaluation/fix9): python fix9.py [ids.json]   (ids.json = list of QA ids)"""
import os, sys, re, json, copy, asyncio, hashlib, time
from collections import Counter
import httpx

ARGV = list(sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'wellposed'))
sys.argv = ['x']
import wellposed_check as W  # noqa: E402
from qual_check import QUAL  # noqa: E402

AT = W.AT
ROUNDS = 4
FMODEL = os.environ.get('FIX_MODEL', 'deepseek-flash')
FCACHE = HERE + '/fix9_llm_cache.jsonl'
OUT = HERE + '/' + os.environ.get('FIX9_OUT', 'fix9.json')
LINKS = HERE + '/serper_links.json'
FSEM = None
TOOL_RE = re.compile(r'web[_ ]?search|code[_ ]?executor|google|search engine|\bsearch(?:ed|ing)?\b|look(?:ed|ing)? up|'
                     r'\bonline\b|internet', re.I)

_fc = {}
if os.path.exists(FCACHE):
    for line in open(FCACHE):
        try:
            j = json.loads(line)
            _fc[j['k']] = j['v']
        except Exception:
            pass
_links = json.load(open(LINKS)) if os.path.exists(LINKS) else {}


async def fllm(prompt):
    k = hashlib.sha1((FMODEL + '|' + prompt).encode()).hexdigest()
    if k in _fc:
        return _fc[k]
    async with FSEM:
        for a in range(5):
            body = {'model': FMODEL, 'temperature': 0.0, 'max_tokens': 32000,
                    'messages': [{'role': 'user', 'content': prompt}]}
            if a >= 2:
                body['thinking'] = {'type': 'disabled'}
            try:
                async with httpx.AsyncClient(timeout=1200.0) as cl:
                    r = await cl.post(W.LLM_URL, headers={'Authorization': 'Bearer ' + W.LLM_KEY}, json=body)
                    t = r.json()['choices'][0]['message']['content']
                    if t:
                        _fc[k] = t
                        with open(FCACHE, 'a') as f:
                            f.write(json.dumps({'k': k, 'v': t}, ensure_ascii=False) + '\n')
                        return t
                    W.STAT['fix_empty'] += 1
            except Exception as e:
                W.STAT['fix_err ' + type(e).__name__] += 1
            await asyncio.sleep(4 * 2 ** a)
    W.STAT['fix_fail'] += 1
    return ''


def links(q):
    """raw Serper organic results [{link, snippet}] (only to attach the page url to a chosen quote)"""
    if q in _links:
        return _links[q]
    if W._SEARCH is None:
        W.search(q)
    g = W._SEARCH.__globals__
    keys, idx = g['SERPER_KEYS'], g['_serper_key_idx']
    for _ in range(len(keys) * 2):
        try:
            r = httpx.post('https://google.serper.dev/search', json={'q': q, 'num': 10},
                           headers={'X-API-KEY': keys[idx[0] % len(keys)]}, timeout=15)
            if r.status_code in (400, 401, 402, 403, 429):
                idx[0] += 1
                continue
            _links[q] = [{'link': o.get('link', ''), 'snippet': o.get('snippet', '')} for o in r.json().get('organic', [])]
            return _links[q]
        except Exception:
            time.sleep(3)
    return []


def wsrc(qa):
    return [s for s in qa['sources'] if isinstance(s, dict) and s.get('type') == 'web_search']


def code_of(qa):
    return '\n'.join(s['code'] for s in qa['sources'] if isinstance(s, dict) and s.get('type') == 'code_executor' and s.get('code'))


def hidden_nums(qa):
    """the value of each hidden entry (first number after '='; timestamps / ranges in the entry are context), minus numbers
    that are also video readings given in a video_observation source (coincidences such as a 45° tilt)"""
    vid = [x for s in qa['sources'] if isinstance(s, dict) and s.get('type') == 'video_observation'
           for x in AT.nums(s.get('key_information'))]
    out = []
    for h in qa.get('hidden_cv_chosen') or []:
        h = str(h)
        n = AT.nums(h.split('=', 1)[1] if '=' in h else h)[:1]
        out += [x for x in n if not any(abs(x - y) <= 1e-9 * max(abs(x), 1e-12) for y in vid)]
    return out


def leaked(question, values):
    qn = AT.nums(question)
    return sorted({v for v in values if not (float(v).is_integer() and abs(v) <= 10)
                   and any(abs(v - x) <= 1e-9 * max(abs(v), 1e-12) for x in qn)})


DESIG = re.compile(r'(?:[A-Za-z#/-]|size|No\.?|ISO|SAE|ASTM|DIN|EN|class|grade|type)\s?$', re.I)


def new_numbers(qa, orig):
    """numbers in the edited question that are not in the original one, without designations (M10, size-64, ISO 7089)"""
    old = AT.nums(orig['question'])
    out = []
    for m in re.finditer(AT.NUM, qa['question']):
        x = float(m.group(0).replace(',', ''))
        if any(abs(x - o) <= 1e-9 * max(abs(o), 1e-12) for o in old) or DESIG.search(qa['question'][max(0, m.start() - 6):m.start()]):
            continue
        out.append(x)
    return out


def spurious(text):
    return [m for m in re.findall(r'\d+\.\d+', str(text)) if len(m.replace('.', '').lstrip('0')) >= 8]


FORUM_RE = re.compile(r'forum|reddit\.|quora\.|answers\.|stackexchange\.|chegg\.|brainly\.|/t/|/threads?/|community\.')
REFS = {'engineering toolbox': 'engineeringtoolbox', 'engineeringtoolbox': 'engineeringtoolbox', 'hyperphysics': 'hyperphysics',
        'wikipedia': 'wikipedia', 'wood handbook': 'fpl.fs.usda', 'matweb': 'matweb', 'nist': 'nist.gov'}


def static_problems(qa, orig=None, info=None):
    P = []
    m = TOOL_RE.search(qa['question'])
    if m:
        P.append({'kind': 'tool_wording', 'text': 'The question mentions tools / searching ("%s"); rephrase without it.'
                  % qa['question'][max(0, m.start() - 40):m.end() + 20]})
    lk = leaked(qa['question'], hidden_nums(qa))
    if lk:
        P.append({'kind': 'value_leak', 'text': 'The question states the hidden value(s) %s; remove them.' % lk})
    if orig is not None:  # numbers introduced by the edit must not be looked-up values
        new = new_numbers(qa, orig)
        webv = []
        for c in (info or {}).get('chosen', []):
            v = c.get('value')
            webv += [float(x) for x in (v if isinstance(v, list) else [v]) if isinstance(x, (int, float))] or AT.nums(v)[:2]
        bad = sorted({x for x in new if any(abs(x - v) <= 0.01 * max(abs(v), 1e-12) for v in webv)})
        if bad:
            P.append({'kind': 'value_leak', 'text': 'The edited question writes looked-up value(s) %s into the question.' % bad})
        triv = [v for v in webv if any(abs(v - t) <= 0.003 * t for t in (9.80665, 9.8, 3.14159, 2.54, 0.3048, 0.4536, 1.609))]
        if triv:
            P.append({'kind': 'trivial_constant', 'text': 'A web constant is g, pi or a unit conversion (%s); the looked-up value '
                      'must stay a physical property of the scene.' % triv[:2]})
        if len(wsrc(qa)) != len(wsrc(orig)):
            P.append({'kind': 'difficulty', 'text': 'The QA must keep exactly %d web constant(s) (has %d) to keep its difficulty.'
                      % (len(wsrc(orig)), len(wsrc(qa)))})
        if len(qa.get('hidden_cv_chosen') or []) != len(orig.get('hidden_cv_chosen') or []):
            P.append({'kind': 'difficulty', 'text': 'The QA must keep exactly %d hidden value(s) (has %d).'
                      % (len(orig.get('hidden_cv_chosen') or []), len(qa.get('hidden_cv_chosen') or []))})
    if qa['question_type'] == 'TYPE_B':
        ar = qa.get('answer_range')
        ok = isinstance(ar, list) and len(ar) == 2 and all(isinstance(x, (int, float)) for x in ar) and ar[0] < ar[1]
        if not ok:
            P.append({'kind': 'format', 'text': 'TYPE_B needs answer_range [low, high] with low < high (got %s).' % ar})
        else:
            on = AT.nums(next((s.get('key_information') for s in qa['sources'] if s.get('type') == 'code_executor'), ''))
            if not all(any(abs(x - o) <= 0.02 * max(abs(x), 1e-9) for o in on) for x in ar):
                P.append({'kind': 'format', 'text': 'answer_range %s is not printed by the code.' % ar})
    sp = spurious(qa['answer']) + [x for h in qa.get('hidden_cv_chosen') or [] for x in spurious(h)]
    if sp:
        P.append({'kind': 'format', 'text': 'Numbers with spurious precision %s in the answer / hidden values; round them as the '
                  'code prints them (3-4 significant figures).' % sp[:3]})
    if orig is not None and orig.get('answer_range') is not None and qa.get('answer_range') is None and qa['question_type'] != 'TYPE_A':
        P.append({'kind': 'format', 'text': 'The original %s answer had answer_range %s; the repaired answer needs one too.'
                  % (orig['question_type'], orig['answer_range'])})
    if orig is not None:  # sources of the repaired web constants: a URL, not a forum / Q&A page, the named reference
        urls = [(s.get('url') or '').lower() for s in wsrc(qa)]
        if any(not u for u in urls):
            P.append({'kind': 'source', 'text': 'A web constant has no source URL; use a value whose page can be identified.'})
        fu = [u for u in urls if FORUM_RE.search(u)]
        if fu:
            P.append({'kind': 'source', 'text': 'Forum / Q&A source %s; use a manufacturer, standard, handbook, reference-table or '
                      'peer-reviewed source instead.' % fu[:1]})
        ql = qa['question'].lower()
        for name, dom in REFS.items():
            if name in ql and not any(dom in u for u in urls):
                P.append({'kind': 'source', 'text': 'The question names "%s" but no web constant comes from %s; name only the '
                          'reference the value is actually taken from.' % (name, dom)})
    for i, ok in (info or {}).get('quote_checks', []):
        if not ok:
            P.append({'kind': 'quote', 'text': 'Web constant [%d]: verified_quote is not in the results of its query.' % i})
    return P


async def verify(item, orig=None, info=None):
    qa = item['qa']
    r = await W.check_qa(item)
    if r.get('serper_failed'):
        return r
    P = []
    if 'annotation_error' in r['flags']:
        P.append({'kind': 'annotation_error', 'text': 'The reference code fails or its output does not reproduce the answer '
                  '(%s); code output: %s' % (r.get('consistency'), r['base_out'][-300:])})
    for i, c in enumerate(r.get('consts', [])):
        if not c['reachable']:
            sq, keyn = c.get('ref_query'), AT.nums(c.get('ref'))
            res = await W.asearch(sq) if sq else None
            if sq and res is None:
                return {'serper_failed': True}
            base = bool(keyn) and res is not None and AT.has(keyn, res)
            base |= any(x.get('same') and not x.get('from_question') for x in c['candidates'])
            j = W.pj(await W.llm(QUAL % (qa['question'], W.mask(c.get('ref')), sq or ''))) or {}
            terms = [t for t in (j.get('terms') or []) if isinstance(t, dict)]
            v = j.get('verdict')
            if v not in ('from_question', 'added'):
                v = 'added' if any(t.get('label') == 'added' for t in terms) else ('from_question' if terms else None)
            added = [t.get('term') for t in terms if t.get('label') == 'added']
            c.update(base_reach=base, qual_verdict=v, qual_added=added)
            if not base or v != 'from_question':
                P.append({'kind': 'web_unreachable', 'const': i, 'text':
                          'Web constant [%d] (%s): the queries a careful solver writes from the question (%s) do not find the '
                          'reference value%s%s.' % (i, c.get('ref'), '; '.join(c.get('queries') or []),
                                                   '' if base else "; the reference's own query '%s' does not return it either" % sq,
                                                   "; the reference's query narrows it with terms the question does not give: %s"
                                                   % added if added else '')})
        if c.get('ambiguous'):
            alts = [x for x in c['candidates'] if x.get('acc') is not None and x['acc'] < .5
                    and x.get('valid_verdict') in ('both', 'alternative')]
            P.append({'kind': 'web_ambiguous', 'const': i, 'text':
                      'Web constant [%d] (%s): from the question a solver can equally find other values: %s.' % (
                          i, c.get('ref'), '; '.join('%s %s ("%s") -> answer %s [auditor: %s]' % (
                              x['value'], x.get('unit', ''), x['quote'][:160], x.get('alt_answer'), (x.get('valid_why') or '')[:200])
                              for x in alts[:4]))})
    for m in r.get('model', []):
        if m.get('acc') is not None and m['acc'] < .5 and m.get('verdict') in ('both', 'X'):
            P.append({'kind': 'model_ambiguous', 'text': "Modelling choice '%s': reference = %s alternative = %s (-> answer %s) "
                      '[auditor: %s]' % (m.get('aspect'), m.get('reference'), m.get('alternative'), m.get('alt_answer'),
                                         (m.get('why') or '')[:250])})
    P += static_problems(qa, orig, info)
    if orig is not None and info is not None:
        P += await fidelity(orig, qa, info)
    r['problems'] = P
    return r


FIDEL = """You review a repair of a physics video benchmark QA. The repair had to make the answer unique (qualifiers, standards,
re-sourced web values, explicit assumptions) while keeping the QA a natural question about the scene in the video.
ORIGINAL QUESTION: %s
ORIGINAL ANSWER: %s
REPAIRED QUESTION: %s
REPAIRED ANSWER: %s   (type %s, range %s)
CHANGES CLAIMED: %s
WEB CONSTANTS NOW USED (quantity | value | verbatim quote from the source):
%s
CODE OUTPUT: %s

Check each criterion; report only real defects:
1 surrogate: a property of an object in the scene is replaced by the property of a different, unrelated object ("take the
  wheel's mass to be that of a shot put", "model the eraser as a coin"), or the question now asks about a peripheral object
  instead of the original one.
2 source_mismatch: a quote does not state the value for exactly the object / condition / statistic the repaired question
  specifies (e.g. the question names a size-64 rubber band but the quote is about an unspecified band; the question says
  mean +- SD but the quote gives a range of means; the question names a product the quote is not about).
3 unnatural: contrived or self-contradictory wording, details that contradict the scene described in the original
  question, method hints that give away the solution beyond stated assumptions.
4 answer: the repaired answer is not supported by the code output or misstates it.
5 drift: what is asked, the physics or the scenario changed more than needed to fix the problems; the asked quantity
  changed (e.g. kinetic energy -> speed); a physical property of the scene was dropped and replaced by a trivial constant
  (g, pi, a unit conversion) as the looked-up value.
ALLOWED, not defects: naming a specific product / model / size / grade / standard / reference table for an object or
material of the scene ("a Shimano WH-RS100 front wheel", "an ISO 7089 M10 washer", "dry oak on oak as tabulated in the
Engineering ToolBox"); stating simplifying modelling assumptions; giving a non-looked-up input (a mass, g) explicitly; small
consistent rephrasings. Report source_mismatch only if the quote clearly is about something else, not for missing
adjectives the question merely adds for clarity.
Return ONLY JSON: {"defects": [{"criterion": "surrogate|source_mismatch|unnatural|answer|drift", "detail": "..."}]}
(empty list if the repair is fine)"""


async def fidelity(orig, qa, info):
    cs = '\n'.join('%s | %s %s | "%s"' % (c.get('quantity'), c.get('value'), c.get('unit') or '', (c.get('quote') or '')[:250])
                   for c in info.get('chosen', [])) or '(none)'
    out = next((s.get('key_information') for s in qa['sources'] if s.get('type') == 'code_executor'), '')
    j = W.pj(await W.llm(FIDEL % (orig['question'], orig['answer'], qa['question'], qa['answer'], qa['question_type'],
                                 qa.get('answer_range'), '; '.join(info.get('changes') or []), cs, str(out)[-1500:]))) or {}
    return [{'kind': 'fidelity_' + str(d.get('criterion')), 'text': 'Reviewer (%s): %s' % (d.get('criterion'), d.get('detail'))}
            for d in (j.get('defects') or []) if isinstance(d, dict) and d.get('criterion') in
            ('surrogate', 'source_mismatch', 'unnatural', 'answer', 'drift')]


# ---------------------------------------------------------------- prompts
PLAN = """You repair a physics video benchmark QA so that its QUESTION text determines the reference answer uniquely.
The solver sees only the QUESTION (class A also sees the video, from which it reads times / distances / speeds; the numbers
already in the question are video readings or givens). It must look up every web constant itself with Google and may run
Python. An audit found the problems listed below.

CLASS: %(cls)s    QUESTION TYPE: %(qtype)s    DIFFICULTY: %(diff)s
QUESTION: %(question)s
REFERENCE SOLUTION:
%(solution)s
WEB CONSTANTS OF THE REFERENCE:
%(consts)s
REFERENCE CODE:
```python
%(code)s
```
CODE OUTPUT: %(out)s
REFERENCE ANSWER: %(answer)s    ANSWER RANGE: %(range)s
HIDDEN VALUES (the solver must obtain them; they must never appear in the question): %(hidden)s

PROBLEMS:
%(problems)s
%(history)s
Repair ALL problems so that a careful solver reading only the edited question arrives at the (repaired) reference answer
and at no other reasonable answer. Allowed means:
 (a) qualify - add to the question the object / material / model / size / population / condition / standard that makes a
     web constant unique and findable (e.g. "a regulation FIBA size 7 men's ball", "hard plastic sliding on dry wood",
     "an ISO 7089 M10 flat washer", "pure water at 20 °C");
 (b) resource - keep the quantity but use another value: the one a search from the edited question finds consistently;
 (c) switch - if a quantity has no unique value anywhere (e.g. "the spring constant of a typical rubber band"), replace it
     by a well-defined quantity with a regulation / standard / handbook value and adapt the scenario minimally, or rewrite
     what is asked so the answer no longer depends on it; keep at least one web constant if the QA had one;
 (d) assumption - for a modelling / interpretation problem state the reference's choice explicitly in the question
     (e.g. "Treat the ball as sliding without rolling.", "Measure the height from the floor.");
 (e) reference - for a material constant whose values differ between sources, name the reference table / standard /
     handbook whose value is used ("as tabulated in the Engineering ToolBox", "per ASTM D1894", "from the Wood Handbook")
     together with the exact material pair / condition.
Rules:
 - NEVER write a looked-up value or a hidden value into the question, not even rounded or in other units; standard names
   and designations (ISO 7089, M10, size 7, SAE 1020) are fine.
 - Never mention tools or searching (web_search, code_executor, Google, search, look up, online, internet).
 - Never widen the answer to a range covering several sources and never accept several answers.
 - Keep the QUESTION TYPE %(qtype)s. A TYPE_B answer is a range: it must come from a constant whose standard specification
   is itself a range (a regulation tolerance band, a handbook range for exactly the stated material), never from several
   conflicting sources.%(typenote)s
 - Keep the difficulty: exactly %(nws)d web constant(s) and the same hidden values (same count), about the same number of
   reasoning steps.
 - Keep the video observations, the given numbers, the physics and what is asked unless (c) requires a change; natural
   English; no method hints beyond the stated assumptions.
 - Wording that a solver could read two ways: fix the wording.
 - Keep what is asked (same quantity). Web constants must stay non-trivial physical properties of the scene; never use g,
   pi or unit conversions as the looked-up value.
 - A web constant must be a property of the object / material actually in the scene (or of the standard it follows).
   NEVER assign an object the property of a different object ("take the wheel's mass to be that of a shot put", "model the
   eraser as a coin") and never move the question to a peripheral object.
 - Name a reference (Engineering ToolBox, Hyperphysics, Wikipedia, a handbook) only if the value will be taken from that very
   site; prefer manufacturer, standard, handbook and reference-table values; never forum or Q&A posts. Qualifiers must read
   naturally (a product, size, material pair, condition), never describe how somebody measured a value.
 - A qualifier must describe exactly what the source states: name a product / size / standard / statistic only if the
   value you will use is stated for exactly that (a "size-64 band" only with a source about size-64 bands).
For every web constant of the repaired QA (reference ones by index, new ones with "ref": null) give the action ("keep" =
unchanged, only for constants without problems) and two different Google queries a careful solver would write from the
EDITED question alone (no numbers).
Return ONLY JSON: {"question": "<full edited question>", "question_type": "TYPE_A|TYPE_B|TYPE_C|TYPE_D",
"changes": ["<short description of each change>"], "constants": [{"ref": <index or null>, "action": "keep|qualify|resource|switch",
"quantity": "<exact quantity with all qualifiers the edited question states>", "queries": ["...", "..."]}]}"""

SOURCE = """A physics benchmark question needs the quantity below, looked up with Google; the solver sees only the QUESTION.
QUESTION: %s
QUANTITY: %s
%s
SEARCH RESULTS (Google snippets):
%s

Choose the value of the QUANTITY that these results state for exactly the object / material / condition / standard the
QUESTION describes. Prefer the value most results agree on, or the regulation / standard / handbook value; keep a range
[low, high] only if the range itself is the standard specification. Copy the quote verbatim from ONE snippet (max 200
characters, it must contain the number as written there). If no snippet states a clearly fitting value, return
{"value": null, "why": "..."}.
Return ONLY JSON: {"value": <number or [low, high]>, "unit": "...", "snippet": "<snippet id, e.g. 1.3>", "quote": "...",
"agreement": "<which other snippets agree / which conflicting values appear>"}"""

APPLY = """Repair the reference code of a physics QA for its edited question and web constants.
EDITED QUESTION: %s
QUESTION TYPE: %s
OLD REFERENCE SOLUTION:
%s
OLD CODE:
```python
%s
```
OLD CODE OUTPUT: %s
OLD ANSWER: %s
CHANGES MADE TO THE QUESTION: %s
WEB CONSTANTS TO USE NOW (in this order):
%s
Write the new code: same structure and video inputs as the old code, the web constants above (convert units in the code),
the modelling the edited question states. It must print every number the final answer will state (including looked-up
values the answer compares against); for TYPE_B print the low and the high end of the answer.
For each web constant give key_information (one sentence: the quantity, the value as quoted, the value used in SI units).
Print numbers rounded (f-strings with 3-4 significant figures), never raw floats.%s
Return ONLY JSON: {"code": "<full code>", "key_information": ["...", ...]%s}"""

FINAL = """Write the reference solution and answer of a physics QA from its code output.
QUESTION: %s
QUESTION TYPE: %s (TYPE_A: one number with unit; TYPE_B: a range "low–high unit"; TYPE_C / TYPE_D: same form as the old answer)
OLD SOLUTION (format to follow):
%s
OLD ANSWER (form to follow): %s
WEB CONSTANTS (query | value | quote):
%s
CODE:
```python
%s
```
CODE OUTPUT:
%s
Write the annotated solution in the step format of the old one ("Step k: ..."; video observation steps kept; one step per
web constant "Use web_search(query='<query>') to retrieve ..." with the quoted value; a step "Use code_executor ..." with the
inputs and the printed results) and the final answer in the same form as the old answer, using only numbers printed by the
code, rounded as in the old answer (3-4 significant figures; never float noise such as 0.0049050000000000005). Give
answer_range [low, high] from the output for TYPE_B, and for TYPE_C / TYPE_D if the old answer had a range (%s); otherwise null.
Return ONLY JSON: {"annotated_solution": ["Step 1: ...", "..."], "answer": "...", "answer_range": [low, high] or null}"""


def consts_txt(qa):
    return '\n'.join('[%d] %s | query: %s | quote: "%s"' % (i, s.get('key_information'), s.get('search_query'),
                                                             (s.get('verified_quote') or '')[:200])
                     for i, s in enumerate(wsrc(qa))) or '(none)'


async def source_const(question, c, old):
    qs = [q for q in (c.get('queries') or []) if isinstance(q, str) and q.strip()][:2]
    if not qs:
        return None, 'no queries for "%s"' % c.get('quantity')
    texts = await asyncio.gather(*[W.asearch(q) for q in qs])
    if any(t is None for t in texts):
        return 'SERPER', None
    blocks, snip = [], {}
    for i, t in enumerate(texts):
        body = re.sub(r'^Search results for "[^"]*":\s*', '', t)
        for j, s in enumerate(body.split(' | ')):
            snip['%d.%d' % (i + 1, j + 1)] = (i, s)
            blocks.append('[%d.%d] %s' % (i + 1, j + 1, s[:400]))
    hint = ''
    if c.get('action') == 'qualify' and old is not None:
        hint = 'The reference used: %s. Keep this value if the results support it for the object of the question.' % old.get('key_information')
    j = W.pj(await fllm(SOURCE % (question, c.get('quantity'), hint, '\n'.join(blocks)))) or {}
    if j.get('value') is None:
        return None, 'no fitting value for "%s" in the results of %s (%s); the results said: %s' % (
            c.get('quantity'), qs, (j.get('why') or '')[:200], ' | '.join(b[:120] for b in blocks[:8]))
    k = next((i for i, t in enumerate(texts) if W.quote_ok(j.get('quote'), t, j['value'])), None)
    if k is None:  # one retry: the quote must be copied character by character from one snippet
        j2 = W.pj(await fllm(SOURCE % (question, c.get('quantity'), hint, '\n'.join(blocks)) + '\n\nYour previous quote "%s" '
                             'is not a verbatim substring of any snippet. Copy the quote character by character from ONE snippet '
                             '(start at a word of that snippet), with the number exactly as written.' % (j.get('quote') or '')[:200])) or {}
        if j2.get('value') is not None:
            j = j2
            k = next((i for i, t in enumerate(texts) if W.quote_ok(j.get('quote'), t, j['value'])), None)
    if k is None:
        return None, 'quote for "%s" not found verbatim in the results' % c.get('quantity')
    q30 = re.sub(r'\s+', ' ', j['quote']).strip()[:30].lower()
    url = ''
    for q in [qs[k]] + [x for x in qs if x != qs[k]]:
        url = next((o['link'] for o in await asyncio.to_thread(links, q) if q30 in re.sub(r'\s+', ' ', o['snippet']).lower()), '')
        if url:
            break
    return {'quantity': c.get('quantity'), 'value': j['value'], 'unit': j.get('unit', ''), 'quote': j['quote'],
            'query': qs[k], 'url': url, 'agreement': j.get('agreement')}, None


async def repair(item, v, hist, last=False):
    qa, cls = item['qa'], item['cls']
    ws = wsrc(qa)
    code = code_of(qa)
    d = dict(cls=cls, qtype=qa['question_type'], diff=qa.get('difficulty'), question=qa['question'],
             solution='\n'.join(qa.get('annotated_solution') or []), consts=consts_txt(qa), code=code, out=v.get('base_out', ''),
             answer=qa['answer'], range=qa.get('answer_range'), hidden=qa.get('hidden_cv_chosen') or [],
             problems='\n'.join('- ' + p['text'] for p in v['problems']),
             history=('PREVIOUS REPAIR ATTEMPTS (do not repeat what failed):\n' + '\n'.join(hist) + '\n') if hist else '',
             nws=len(ws), typenote=' Only if no such range exists at all may it become TYPE_A (single number).'
             if last and qa['question_type'] == 'TYPE_B' else '')
    plan = W.pj(await fllm(PLAN % d)) or {}
    nq, cs = plan.get('question'), [c for c in (plan.get('constants') or []) if isinstance(c, dict)]
    if not nq:
        return None, 'plan failed'
    qtype = plan.get('question_type') if plan.get('question_type') in ('TYPE_A', 'TYPE_B', 'TYPE_C', 'TYPE_D') else qa['question_type']
    if qtype != qa['question_type'] and not (last and qa['question_type'] == 'TYPE_B' and qtype == 'TYPE_A'):
        qtype = qa['question_type']
    chosen = []
    for c in cs:
        ref = c.get('ref') if isinstance(c.get('ref'), int) and 0 <= c.get('ref') < len(ws) else None
        if c.get('action') == 'keep' and ref is not None:
            s = ws[ref]
            chosen.append({'keep': ws[ref], 'quantity': c.get('quantity'), 'query': s.get('search_query'),
                           'value': s.get('key_information'), 'quote': s.get('verified_quote', '')})
            continue
        s, err = await source_const(nq, c, ws[ref] if ref is not None else None)
        if s == 'SERPER':
            return 'SERPER', None
        if err:
            return None, err
        chosen.append(s)
    ctxt = '\n'.join('[%d] %s = %s %s | quote: "%s" | query: %s' % (i, c['quantity'], c['value'], c.get('unit', ''), c['quote'][:200], c['query'])
                     for i, c in enumerate(chosen)) or '(none)'
    hid = ('\nAlso update hidden_cv_chosen %s: exactly the same number of entries in the same form "<quantity> = <value> <unit>"; '
           'entries of web constants that changed get the new value (short quantity name as before, value as quoted / rounded), the others stay.'
           % json.dumps(qa.get('hidden_cv_chosen') or [], ensure_ascii=False), ', "hidden_cv_chosen": ["..."]') \
        if cls == 'B' and qa.get('hidden_cv_chosen') else ('', '')
    ap = W.pj(await fllm(APPLY % (nq, qtype, d['solution'], code, d['out'], qa['answer'], '; '.join(plan.get('changes') or []),
                                  ctxt, hid[0], hid[1]))) or {}
    ncode = ap.get('code')
    if not ncode:
        return None, 'apply failed'
    out = await asyncio.to_thread(W.run_code, ncode)
    if out.startswith('Error'):
        return None, 'new code fails: ' + out[-200:]
    fin = W.pj(await fllm(FINAL % (nq, qtype, d['solution'], qa['answer'],
                                   '\n'.join('%s | %s %s | "%s"' % (c['query'], c['value'], c.get('unit', ''), c['quote'][:200]) for c in chosen),
                                   ncode, out[:3000], qa.get('answer_range')))) or {}
    if not fin.get('answer') or not isinstance(fin.get('annotated_solution'), list):
        return None, 'final failed'
    new = copy.deepcopy(qa)
    kis = ap.get('key_information') or []
    web = []
    for i, c in enumerate(chosen):
        if 'keep' in c:
            web.append(copy.deepcopy(c['keep']))
            continue
        web.append({'type': 'web_search', 'url': c['url'], 'key_information': kis[i] if i < len(kis) and kis[i] else
                    '%s = %s %s' % (c['quantity'], c['value'], c.get('unit', '')), 'search_query': c['query'],
                    'verified_quote': c['quote']})
    other = [s for s in qa['sources'] if isinstance(s, dict) and s.get('type') not in ('web_search', 'code_executor')]
    new['sources'] = other + web + [{'type': 'code_executor', 'code': ncode, 'key_information': out,
                                     'key_information_annotated': None}]
    new.update(question=nq, question_type=qtype, annotated_solution=[str(x) for x in fin['annotated_solution']],
               answer=str(fin['answer']), answer_range=fin.get('answer_range') if qtype == 'TYPE_B' or
               (qtype != 'TYPE_A' and qa.get('answer_range') is not None) else None)
    new['total_steps'] = len(new['annotated_solution'])
    new['required_external_tools'] = (['web_search'] if web else []) + ['code_executor']
    if hid[0] and isinstance(ap.get('hidden_cv_chosen'), list):
        new['hidden_cv_chosen'] = [str(x) for x in ap['hidden_cv_chosen']]
    qc = []
    for i, c in enumerate(chosen):
        if 'keep' in c:
            continue
        t = await W.asearch(c['query'])
        qc.append((i, t is not None and W.quote_ok(c['quote'], t, c['value'])))
    return new, {'changes': plan.get('changes'), 'chosen': [{k: x for k, x in c.items() if k != 'keep'} for c in chosen],
                 'quote_checks': qc}


async def fix_one(item):
    orig = item['qa']
    v = await verify(item)
    if v.get('serper_failed'):
        return None
    rec = {'id': item['id'], 'split': item['split'], 'cls': item['cls'], 'orig_problems': v['problems'], 'rounds': []}
    if not v['problems']:
        rec['status'] = 'clean'
        return rec
    cur, cv, hist = item, v, []
    for k in range(ROUNDS):
        new, info = await repair(cur, cv, hist, last=k == ROUNDS - 1)
        if new == 'SERPER':
            return None
        if new is None:
            rec['rounds'].append({'error': info})
            hist.append('ATTEMPT %d failed before checking: %s' % (k + 1, info))
            continue
        ni = dict(item, qa=new)
        nv = await verify(ni, orig, info)
        if nv.get('serper_failed'):
            return None
        rec['rounds'].append({'question': new['question'], 'answer': new['answer'], 'answer_range': new.get('answer_range'),
                              'question_type': new['question_type'], 'changes': info['changes'], 'chosen': info['chosen'],
                              'problems': nv['problems'], 'qa': new})
        if not nv['problems']:
            rec.update(status='fixed', qa=new)
            return rec
        hist.append('ATTEMPT %d question: %s\n  changes: %s\n  remaining problems: %s' % (
            k + 1, new['question'], info['changes'], ' | '.join(p['text'][:300] for p in nv['problems'])))
        if len(nv['problems']) <= len(cv['problems']):
            cur, cv = ni, nv
    rec['status'] = 'unresolved'
    best = [r for r in rec['rounds'] if 'qa' in r]
    if best:
        rec['qa'] = min(best, key=lambda r: len(r['problems']))['qa']
    return rec


async def main():
    global FSEM
    W.SEM, W.SSEM, FSEM = asyncio.Semaphore(48), asyncio.Semaphore(4), asyncio.Semaphore(16)
    items = W.load()
    if len(ARGV) > 1:
        ids = set(json.load(open(ARGV[1])))
        items = [i for i in items if i['id'] in ids]
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [i for i in items if i['id'] not in done]
    print('QAs %d  todo %d  fixer %s  checker %s' % (len(items), len(todo), FMODEL, W.MODEL), flush=True)
    for b in range(0, len(todo), 24):
        rs = await asyncio.gather(*[fix_one(i) for i in todo[b:b + 24]])
        for r in rs:
            if r:
                done[r['id']] = r
        json.dump(done, open(OUT, 'w'), ensure_ascii=False, indent=1)
        json.dump(_links, open(LINKS, 'w'), ensure_ascii=False)
        print('progress %d/%d  %s  serper_fail %d  %s' % (min(b + 24, len(todo)), len(todo),
              dict(Counter(r['status'] for r in done.values())), W.SERPER_FAILS[0], dict(W.STAT)), flush=True)
        if W.SERPER_FAILS[0] >= 3 or any(r is None for r in rs):
            print('SERPER FAILING - stop; check keys / credits', flush=True)
            sys.exit(3)
    C = Counter((r['split'], r['cls'], r['status']) for r in done.values())
    for k in sorted(C):
        print(k, C[k])
    print('problem kinds before:', dict(Counter(p['kind'] for r in done.values() for p in r['orig_problems'])))
    print('problem kinds left (unresolved):', dict(Counter(p['kind'] for r in done.values() if r['status'] == 'unresolved'
                                                            for p in min((x for x in r['rounds'] if 'problems' in x),
                                                                         key=lambda x: len(x['problems']),
                                                                         default={'problems': r['orig_problems']})['problems'])))


if __name__ == '__main__':
    asyncio.run(main())
