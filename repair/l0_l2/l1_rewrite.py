# Provenance (TriPhysBench release): fix_code/_v8_work/l1_rewrite.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L1 step 2: in-place repair of every problem QA with the real evidence of l1_evidence.py.
The LLM rewrites the QA (search steps + sources + downstream steps + code + answer + answer_range + B hidden_cv_chosen);
question_type / difficulty / video observations are fixed. Every candidate is checked programmatically
(validate) and by an independent LLM review; failures are fed back (max 4 attempts). A QA with a fact that has no
real evidence goes to L2.
Output: l1_results.json {qid: {status: ok|fail|need_L2, qa, attempts: [...], ...}}   (resumable)
Usage: python l1_rewrite.py [qid ...]"""
import copy, json, os, re, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fixlib import REP, video_context, call_llm, load, iter_qas, get_steps, sources_of, run_code, text_nums, consistent, \
    in_answer_range, squash

OUT = 'l1_results.json'
MAX_ATTEMPTS = 4
_lock = threading.Lock()
problems = {p['qid']: p for p in json.load(open('problems.json', encoding='utf-8'))}
evidence = json.load(open('evidence.json', encoding='utf-8'))
code_final = json.load(open(REP + 'v8_code_final.json', encoding='utf-8'))
test_check = {r['qid']: r for r in json.load(open(REP + 'v8_test_code_check.json', encoding='utf-8'))}
qas = {}
for split in ('train', 'test'):
    for cls in 'ab':
        for qid, vid, qa in iter_qas(load(cls, split, 'L0'), cls, split):
            qas[qid] = qa
res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}

FIELDS = ('question_type', 'difficulty', 'question', 'annotated_solution', 'sources', 'answer', 'answer_range',
          'hidden_cv_chosen')
TYPE_RULE = {
    'TYPE_A': 'TYPE_A (quantitative): the answer is ONE number with unit (answer_range = null); it is scored by the '
              'relative error of that single number.',
    'TYPE_B': 'TYPE_B (state prediction): keep the answer form — a range "lo–hi unit" with answer_range=[lo, hi] if the '
              'current answer is a range or if a required input is a range; otherwise one number with unit and null.',
    'TYPE_C': 'TYPE_C (causal / constraint): the answer starts with "Yes" or "No" followed by the quantitative '
              'justification; recompute the verdict with the new values (it may flip; the question stays the same).',
    'TYPE_D': 'TYPE_D (comparative): the answer states the comparison direction and the computed number(s); '
              'answer_range = [lo, hi] only if the answer is a range.',
}

REWRITE = """You repair ONE physics QA of a video benchmark. Some web-retrieved values in it could not be confirmed by
real web search and/or its code does not reproduce its answer. Below is REAL evidence from Google results.
Produce the corrected QA.

HARD RULES
1. question_type = {qtype} and difficulty = {diff} are fixed. {type_rule}
2. Keep the question text EXACTLY unchanged, unless a number stated in it is itself wrong; never put a retrieved value
   into the question. Video observations (values seen in / measured from the video) stay exactly as they are.
3. For each EVIDENCE fact: the corresponding search step must read web_search(query='<evidence query>') with exactly
   that query; its web_search source gets url = evidence url and key_information = the fact as the evidence states it
   (plus the converted value used). Use the evidence value (unit-converted if needed) EVERYWHERE the old value was
   used: downstream steps, code, answer. If the evidence is a range, follow rule 1 for the answer form (TYPE_A needs a
   single value — evidence for TYPE_A is always a single value). If the quote states a number that fits the question
   better than the extracted value (e.g. the quote gives both string length and pendulum length), you may use that
   number instead — report it in "evidence_used". If an evidence value clearly does not fit the object in the video
   (see VIDEO), do not use it: report it in "evidence_rejected" with the reason and return the QA otherwise unchanged.
4. VERIFIED facts stay as they are (same query, url, value).
5. Every web_search source must be retrieved by a web_search step of the solution. UNLINKED sources: if evidence is
   given, add/merge a search step with the evidence query and use the evidence value; if marked "no search needed"
   (unit conversion / constant) remove that web_search source (the conversion may stay inside the computation).
6. Code (code_executor sources): runnable Python (math, numpy available); hardcode exactly the input values the solution
   uses (video observations + retrieved values), never introduce or tune a number to reach an answer; print every
   result with label and unit; the LAST code block prints the final answer value(s) (both ends for a range).
   The code's printed values are the truth: steps after the code and the answer must use exactly those values.
7. CODE PROBLEMS: fix the code if it is wrong; if the solution's own formula and inputs give a different number than
   the stated answer, correct the answer and the downstream steps.
8. {hidden_rule}
9. Keep the step structure and wording otherwise; change only what the new values / fixes require. Keep the
   calculation method and conventions of the current solution (e.g. how a percentage difference is defined) — they
   are part of the benchmark's answer; do not switch to another valid convention.

VIDEO (description of the clip): {video}

CURRENT QA:
{qa}

EVIDENCE (real search results; use these values):
{evidence}

VERIFIED facts (keep unchanged):
{verified}

CODE PROBLEMS:
{code_problems}
{feedback}
Return ONLY JSON:
{{"question": "...", "annotated_solution": ["Step 1: ...", ...], "sources": [{{"type": "video_observation", ...}}, ...],
"answer": "...", "answer_range": null or [lo, hi], "hidden_cv_chosen": [...],
"evidence_used": {{"<evidence number>": {{"quote_number": "<number as written in the quote>", "value_used": "<value with unit as used in the code>"}}, ...}},
"evidence_rejected": {{}} or {{"<evidence number>": "<why it does not fit the video>"}},
"change_summary": "<what changed and why>"}}"""

HIDDEN_A = ('hidden_cv_chosen lists CV/video values hidden from the question — return it unchanged.')
HIDDEN_B = ('hidden_cv_chosen lists the retrieved (web_search) values hidden from the question as "quantity = value"; '
            'update every entry whose value changed to the new value; keep the same number of entries (it is fixed by '
            'the difficulty).')

A_HIDDEN = ('\nNote (A-class): the values of hidden_cv_chosen (e.g. joint speeds) are measured from the video by the '
            'benchmark\'s CV pipeline; the solver has to estimate them from the video, so they need no source — do '
            'not flag them for missing provenance. Video measurements that the question itself states (e.g. "wrist speed is '
            'measured at 4.78 m/s") are given data, not hidden values — they must stay in the question.')

REVIEW = """You are a strict reviewer of a repaired physics QA from a video benchmark (the solver sees the video and can
use web_search and code_executor). Check ONLY for real defects:
- internal consistency: every number used in a step comes from the question, a video observation source, a web_search
  source, or an earlier computation; numbers after a code step equal what the code prints (given below); the answer
  equals the final computed value (answer_range brackets it when present);
- physics: formulas and units are correct and appropriate for the question;
- the question is answerable and unambiguous, and does not state any retrieved or hidden value;
- the answer form fits the type: {type_rule}
- retrieved values match their sources' key_information and the verbatim quotes below (unit conversions correct).
Retrieved values are typical / specification values for the kind of object seen (the exact object is unknown) — that
is acceptable by design; flag a retrieved value only if it is clearly implausible for the object in the video.
Do not ask for a different convention or definition (e.g. of a percentage difference) when the solution's one is a
valid reading of the question.
The benchmark's modelling simplifications are accepted by design — do not flag them: a joint speed (wrist / elbow)
used as the released object's speed, point-mass / rigid-body / no-loss models, stated typical values or assumptions
for quantities the video cannot give. Flag only errors in the solution's own numbers, sources and physics.
Question type {qtype}, difficulty {diff}.

QA:
{qa}

Real code outputs: {outs}
Verbatim quotes of the web_search sources (real search results):
{quotes}

Return ONLY JSON: {{"pass": true/false, "issues": ["<concise, actionable>", ...]}}"""


def view(qa, cls):
    d = {k: qa.get(k) for k in FIELDS if k != 'hidden_cv_chosen' or cls == 'B'}
    d['sources'] = [{k: v for k, v in s.items() if k in ('type', 'time_point', 'url', 'key_information', 'code', 'search_query')}
                    for s in qa.get('sources', [])]
    return json.dumps(d, ensure_ascii=False, indent=1)


def ev_items(qid):
    return list(evidence.get(qid, {}).values())


def fmt_evidence(items):
    out = []
    for i, it in enumerate(items, 1):
        e = it['evidence']
        if it['kind'] == 'ws_unlinked' and e.get('no_search_needed'):
            out.append('%d. UNLINKED source #%d "%s": no search needed (%s) -> remove this web_search source.'
                       % (i, it['source_idx'], it['target'], e.get('why')))
            continue
        what = ('UNLINKED source #%d "%s"' % (it['source_idx'], it['target']) if it['kind'] == 'ws_unlinked'
                else 'search step for "%s" (current value %s)' % (it['target'], it['current']))
        out.append('%d. %s\n   query: %s\n   url: %s\n   page title: %s\n   quote: "%s"\n   value: %s  (converted: %s)'
                   % (i, what, e['query'], e['url'], e.get('title'), e['quote'], e['value'], e.get('value_converted')))
    return '\n'.join(out) or 'none'


def fmt_verified(qa):
    return '\n'.join('- query "%s" -> %s (%s)' % (s.get('search_query'), s.get('key_information'), s.get('url'))
                     for s in sources_of(qa, 'web_search') if s.get('verified_quote')) or 'none'


def fmt_code(qid):
    p = problems[qid]
    lines = []
    for c in p['code']:
        hint = ''
        if qid in code_final:
            d = (code_final[qid][c['block']].get('diag') or {}) if c['block'] < len(code_final[qid]) else {}
            if d:
                hint = ' | earlier diagnosis: %s — %s (answer implied: %s)' % (d.get('verdict'), d.get('why'), d.get('correct_answer'))
        elif qid in test_check and test_check[qid].get('verdict'):
            hint = ' | earlier diagnosis: %s — %s' % (test_check[qid]['verdict'], test_check[qid].get('why'))
        lines.append('- code block %d %s%s' % (c['block'] + 1, c['problem'], hint))
    return '\n'.join(lines) or 'none (but the code must still reproduce the new answer)'


def lits(code):
    import ast
    try:
        return [abs(n.value) for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    except SyntaxError:
        return []


def near(a, b, tol=0.02):
    return any(abs(a * s - b) <= tol * max(abs(a * s), abs(b)) for s in (1, 1e2, 1e-2, 1e3, 1e-3, 1e6, 1e-6))


# stray CJK / kana / Thai / Hangul tokens or words fused with numbers (same rule as agentic_physics_qa.GARBLED)
GARBLED = re.compile(r"[぀-ヿ一-鿿฀-๿가-힯]|\b[a-z]{4,}\d")
META = re.compile(r"\b(?:review(?:er)?|supplied|hidden|original (?:code|qa|question|solution|answer|range)|"
                  r"previous (?:version|answer|qa|solution)|verified (?:source|search|quote|result))\b", re.I)
UNDET = re.compile(r"not determinable|cannot be determined|undetermined|indeterminate|not determined by", re.I)


def text_checks(old, new):
    """text-level checks shared by L1 and the L2 post-screen: garbled text, step labels, meta wording, undeterminable answers, zero-width ranges"""
    errs = []
    steps, _ = get_steps(new)
    ki = [str(s.get('key_information') or '') for s in new.get('sources') or [] if s.get('type') != 'code_executor']
    for t in [new.get('question') or '', str(new.get('answer') or '')] + steps + ki:
        m = GARBLED.search(re.sub(r'https?://\S+|www\.\S+', '', t))
        if m:
            errs.append('garbled text "%s" in: %s — write it cleanly' % (m.group(0), t[:120]))
            break
    bad = [st[:60] for st in steps if st.startswith('Step') and not re.match(r'Step \d+:', st)]
    if bad:
        errs.append('malformed step label(s) %s — every step starts with "Step <n>:"' % bad)
    # the repair must read like a benchmark QA: no references to the repair process
    old_text = json.dumps(old, ensure_ascii=False).lower()
    for t in [new.get('question') or '', str(new.get('answer') or '')] + steps + ki:
        m = META.search(t)
        if m and m.group(0).lower() not in old_text:
            errs.append('meta wording "%s" in: %s — the QA must not refer to reviews, previous versions or '
                        'hidden / supplied values' % (m.group(0), t[:120]))
            break
    if UNDET.search(str(new.get('answer') or '')):
        errs.append('the answer must give the computed result, not declare it undeterminable — solve the question '
                    'as posed with the benchmark\'s modelling simplifications')
    rng = new.get('answer_range')
    if not old.get('answer_range') and rng and old.get('question_type') in ('TYPE_A', 'TYPE_C'):
        errs.append('answer_range must stay null for %s' % old['question_type'])
    if old.get('answer_range') and not rng:
        errs.append('answer_range was dropped — the answer form is fixed by the question type: give the range '
                    '[low, high] of the answer')
    if isinstance(rng, list) and len(rng) == 2 and rng[0] is not None and rng[1] is not None:
        try:
            lo, hi = float(rng[0]), float(rng[1])
            if hi - lo <= 1e-9 * max(1.0, abs(hi)):
                errs.append('answer_range [%s, %s] has zero width (it is scored as a range): the range must come from '
                            'the range / spread of an input (e.g. a min–max or ± of a retrieved value, or two '
                            'evidence values of the same quantity)' % (rng[0], rng[1]))
        except (TypeError, ValueError):
            pass
    return errs


def validate(qid, old, new, items, used_map=None):
    """returns (errors, warnings); fills search_query / verified_quote / key_information(stdout) in new"""
    cls = qid.split('|')[0][-1]
    errs, warns = [], []
    if new.get('question') != old.get('question'):
        on, nn = set(text_nums(old['question'])), set(text_nums(new['question']))
        if nn - on:
            errs.append('question text gained numbers %s — keep the question unchanged' % sorted(nn - on))
        if cls == 'A' and on - nn:
            errs.append('question text lost the given values %s — values stated in the question are given data '
                        '(measured from the video), keep them' % sorted(on - nn))
        warns.append('question text changed')
    ov = [s for s in old['sources'] if s.get('type') == 'video_observation']
    nv = [s for s in new['sources'] if s.get('type') == 'video_observation']
    if [s.get('key_information') for s in ov] != [s.get('key_information') for s in nv]:
        errs.append('video_observation sources must stay exactly unchanged')
    steps, _ = get_steps(new)
    sol = '\n'.join(steps)
    errs += text_checks(old, new)
    nws =sources_of(new, 'web_search')
    old_ws = {s.get('url'): s for s in sources_of(old, 'web_search')}
    # evidence facts
    used_map = used_map or {}
    for n, it in enumerate(items, 1):
        e = it['evidence']
        if it['kind'] == 'ws_unlinked' and e.get('no_search_needed'):
            if any(s.get('key_information') == it['target'] for s in nws):
                errs.append('unlinked source "%s" should be removed' % it['target'])
            continue
        if e['query'] not in sol:
            errs.append("the solution must contain web_search(query='%s')" % e['query'])
        s = next((s for s in nws if s.get('url') == e['url']), None)
        if s is None:
            errs.append('no web_search source with url %s' % e['url'])
            continue
        s['search_query'], s['verified_quote'] = e['query'], e['quote']
        vnums = [x for x in text_nums(e.get('value_converted') or e['value']) if x > 0]
        used = lits(' '.join(c.get('code') or '' for c in sources_of(new, 'code_executor'))) + text_nums(sol)
        u = used_map.get(str(n)) or {}
        qn = [x for x in text_nums(u.get('quote_number')) if x > 0]
        if qn and all(any(abs(x - y) <= 1e-9 + 1e-6 * y for y in text_nums(e['quote'])) for x in qn)                 and u.get('quote_number') not in (e['value'], e.get('value_converted')):
            # the unit conversion quote_number -> value_used is checked by the independent review
            vnums = [x for x in text_nums(u.get('value_used')) if x > 0]
            if not vnums:
                errs.append('evidence_used %s: value_used "%s" has no number' % (n, u.get('value_used')))
            s['evidence_number_used'] = u
        for x in vnums:
            if not any(near(x, y) for y in used):
                errs.append('evidence value %s (%s) is not used in the solution/code' % (x, e.get('value_converted') or e['value']))
        if it['kind'] == 'ws_unresolved':
            onums = [x for x in text_nums(it['current']) if x > 0 and not any(near(x, v) for v in vnums)]
            code_l = lits(' '.join(c.get('code') or '' for c in sources_of(new, 'code_executor')))
            for x in onums:
                if any(near(x, y, 0.001) for y in code_l) and x not in (1, 2, 3, 9.81, 9.8):
                    errs.append('old value %s (%s) is still used in the code' % (x, it['current']))
            if cls == 'B':
                if len(new.get('hidden_cv_chosen') or []) != len(old.get('hidden_cv_chosen') or []):
                    errs.append('hidden_cv_chosen must keep %d entries (difficulty), got %d'
                                % (len(old.get('hidden_cv_chosen') or []), len(new.get('hidden_cv_chosen') or [])))
                for h in new.get('hidden_cv_chosen') or []:
                    if any(near(x, y, 0.001) for x in onums for y in text_nums(h)):
                        errs.append('hidden_cv_chosen still has the old value: "%s"' % h)
    # every web_search source must be verified (evidence, or L0-verified and unchanged)
    for s in nws:
        if s.get('verified_quote'):
            if s.get('search_query') and s['search_query'] not in sol:
                errs.append("the verified query '%s' must stay in the solution" % s['search_query'])
            o = old_ws.get(s.get('url'))
            if o is not None:
                for k in ('url_annotated',):
                    if k in o:
                        s.setdefault(k, o[k])
            continue
        errs.append('web_search source "%s" (%s) has no real evidence — only evidence / verified facts may be sources'
                    % (s.get('key_information'), s.get('url')))
    # code
    ans = str(new.get('answer') or '')
    codes = sources_of(new, 'code_executor')
    if not codes and sources_of(old, 'code_executor'):
        errs.append('code_executor sources were removed')
    outs = []
    for i, s in enumerate(codes):
        r = run_code(s.get('code') or '')
        outs.append(r['out'])
        if not r['ok']:
            errs.append('code block %d fails: %s' % (i + 1, r['out'][:200]))
        elif r['out'] == 'No output':
            errs.append('code block %d prints nothing' % (i + 1))
        else:
            s.setdefault('key_information_annotated', next((o.get('key_information') for o in sources_of(old, 'code_executor')
                                                            if o.get('code') == s.get('code')), None))
            s['key_information'] = r['out'][:500]
    anums = [x for x in text_nums(ans) if x > 0]
    if outs and not outs[-1].startswith('Error') and outs[-1] != 'No output':
        last = outs[-1]
        if anums and not (consistent(last, anums) or in_answer_range(last, ans)):
            errs.append('the last code block prints "%s" but the answer is "%s" — the answer must use the printed value'
                        % (last[:200], ans[:200]))
    qt = old.get('question_type')
    ar = new.get('answer_range')
    if qt == 'TYPE_A' and ar not in (None, [], 'null'):
        errs.append('TYPE_A must have answer_range = null')
    if isinstance(ar, list):
        if len(ar) != 2 or not all(isinstance(x, (int, float)) for x in ar) or ar[0] > ar[1]:
            errs.append('answer_range must be [lo, hi] numbers')
        elif codes and outs and not all(consistent(outs[-1], [abs(x)]) for x in ar if x):
            errs.append('answer_range %s must equal values printed by the last code block ("%s")' % (ar, outs[-1][:200]))
    if qt == 'TYPE_C' and not re.match(r'\W*(yes|no)\b', ans, re.I):
        errs.append('TYPE_C answer must start with Yes or No')
    if qt == 'TYPE_A' and re.search(RANGE_ANS, ans):
        errs.append('TYPE_A answer must be a single value, not a range')
    return errs, warns, outs


RANGE_ANS = r'\d\s*(?:–|—|to|-)\s*\d'


def normalize(old, d, cls):
    new = copy.deepcopy(old)
    for k in ('question', 'annotated_solution', 'sources', 'answer', 'answer_range'):
        if k in d:
            new[k] = d[k]
    if cls == 'B' and isinstance(d.get('hidden_cv_chosen'), list):
        new['hidden_cv_chosen'] = d['hidden_cv_chosen']
    if isinstance(new.get('answer_range'), str):
        try:
            new['answer_range'] = json.loads(new['answer_range'])
        except Exception:
            new['answer_range'] = None
    if not isinstance(new['annotated_solution'], list):
        new['annotated_solution'] = get_steps(new)[0]
    # the model sees only type/url/key_information/code/search_query: re-attach the verification metadata of the
    # unchanged web_search sources (same url)
    old_ws = {s.get('url'): s for s in old.get('sources', []) if s.get('type') == 'web_search'}
    for s in new.get('sources', []):
        o = old_ws.get(s.get('url')) if s.get('type') == 'web_search' else None
        if o is not None and o.get('verified_quote'):
            for k in ('search_query', 'verified_quote', 'url_annotated'):
                if k in o:
                    s[k] = o[k]
    return new


REDO = json.load(open('redo_L1.json', encoding='utf-8')) if os.path.exists('redo_L1.json') else {}


def repair(qid):
    if qid in res:  # resumable: failed QAs go to L2
        return
    old = qas[qid]
    cls = qid.split('|')[0][-1]
    items = ev_items(qid)
    missing = [it for it in items if not it['evidence'].get('found') and not it['evidence'].get('no_search_needed')]
    if missing:
        rec = {'status': 'need_L2', 'reason': 'no real evidence for: ' + '; '.join(it['target'][:120] for it in missing)}
        with _lock:
            res[qid] = rec
            json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(qid, 'need_L2', flush=True)
        return
    qt, diff = old.get('question_type'), old.get('difficulty')
    attempts, feedback, prev = [], '', None
    if qid in REDO:  # rejected in the one-by-one review / text screen: start from the reviewer's issues
        feedback = ('\nA PREVIOUS REWRITE OF THIS QA WAS REJECTED (avoid these problems, keep the rules):\n- %s\n'
                    % '\n- '.join(x.replace(' (l1_redo_checks.json)', '') for x in REDO[qid]))
    video = video_context(qid) or '(no description)'
    status, new = 'fail', None
    for a in range(MAX_ATTEMPTS):
        prompt = REWRITE.format(qtype=qt, diff=diff, type_rule=TYPE_RULE.get(qt, ''), qa=view(old, cls), video=video,
                                evidence=fmt_evidence(items), verified=fmt_verified(old), code_problems=fmt_code(qid),
                                hidden_rule=HIDDEN_B if cls == 'B' else HIDDEN_A, feedback=feedback)
        d = call_llm(prompt, temperature=0.2)
        if not isinstance(d, dict):
            attempts.append({'errors': ['no JSON']})
            continue
        rej = {k: v for k, v in (d.get('evidence_rejected') or {}).items() if v}
        if rej:
            status = 'evidence_rejected'
            attempts.append({'evidence_rejected': rej, 'change_summary': d.get('change_summary')})
            break
        cand = normalize(old, d, cls)
        errs, warns, outs = validate(qid, old, cand, items, d.get('evidence_used') if isinstance(d.get('evidence_used'), dict) else {})
        rv = None
        if not errs:
            rv = call_llm(REVIEW.format(type_rule=TYPE_RULE.get(qt, ''), qtype=qt, diff=diff, qa=view(cand, cls),
                                        outs=json.dumps(outs, ensure_ascii=False),
                                        quotes='\n'.join('- %s: "%s"' % (x.get('url'), x.get('verified_quote'))
                                                          for x in sources_of(cand, 'web_search')) or 'none')
                        + A_HIDDEN * (cls == 'A')) or {}
            if not rv.get('pass', False):
                errs = ['review: ' + x for x in rv.get('issues') or ['review failed without issues']]
        attempts.append({'errors': errs, 'warnings': warns, 'change_summary': d.get('change_summary')})
        if not errs:
            status, new = 'ok', cand
            break
        prev = cand
        feedback = ('\nYOUR PREVIOUS ATTEMPT (fix these problems, keep the rules):\n%s\nPROBLEMS:\n- %s\n'
                    % (view(cand, cls), '\n- '.join(errs)))
    rec = {'status': status, 'attempts': attempts, 'qa': new if new is not None else prev,
           'evidence': [k for k in evidence.get(qid, {})]}
    with _lock:
        res[qid] = rec
        json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(qid, status, 'attempts=%d' % len(attempts), '' if status == 'ok' else (attempts[-1].get('errors') or attempts[-1].get('evidence_rejected')), flush=True)


if __name__ == '__main__':
    todo = sys.argv[1:] or list(problems)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(repair, todo))
    print(dict(Counter(r['status'] for q, r in res.items() if q in problems)))
