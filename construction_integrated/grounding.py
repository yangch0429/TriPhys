"""
grounding.py  —  checks the Stage-3 author agent's QA must pass while it is written

Every generated QA goes through these checks before Stage 3 accepts it (at most 4 attempts per tier; the problems of a
rejected attempt are fed back to the agent):

  * grounding (validate_grounding): every web_search source cites a url that one of the agent's web_search calls
    returned, a verbatim quote of that result that states the value, and the query that returned it (a query that
    contains the value itself is re-run without it); every code block is re-executed, its stdout replaces
    key_information, and the answer (both ends of an answer_range) must be printed by the last block;
  * leaks: no hidden value and no web value in the question (unit-aware), no tool / search wording;
  * answer type and answer (settle_answer, before the checks): the type of a numeric QA is set from its inputs, not
    chosen in advance - TYPE_B when a range input (a web constant stated as a range, or a range the question states) is
    propagated, TYPE_A otherwise; an answer / answer_range that is not printed by the code is rewritten from the output;
  * answer form: TYPE_A one value, TYPE_B an interval propagated from a range input, TYPE_C Yes / No, TYPE_D a
    comparison; every input the code uses is given by the question, the video readings or the web constants;
  * range form: a query that asks for a range / tolerance / limits returns a range;
  * unique-value search test of every web constant (unique()): the Serper results of its query are extracted, every
    value found is substituted into the code and scored with the official ACC; a value that changes the score is a
    conflict unless question + video single out the reference value (Gemini on frames); the value must be stated by
    two sites or a standard / handbook source, and a paraphrase of the query must not return a conflicting value.
    A constant that fails is resolved by a new query for the same quantity ("requery") or by one explicit "Assume ..."
    sentence that names the material / grade / standard ("qualify"); a QA whose constant stays ambiguous is
    regenerated, and after the last attempt the ambiguous constant may become a stated given (stated_given()).
"""

import re
import copy
import json
import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import units as U
from verify_common import (
    llm, vlm, pj, serper, asearch, alinks, results_block, nums, has, near, sci, quote_ok, mask, canon, coin,
    run_code, arun, acc, num_map, alt_answer, range_of, TOOL_RE, sources, wsrc, code_of, hidden_nums, leaked,
    frames, fkey,
)

logger = logging.getLogger("grounding")

HIDDEN = {"easy": 0, "medium": 1, "hard": 2}     # hidden CV readings per tier (A); B hides hidden + 1 web specs
LV = {1: "easy", 2: "medium", 3: "hard"}          # B: difficulty = number of web constants
MAX_ATTEMPTS = 4
MAX_CANDIDATES = 5

# ---------------------------------------------------------------------------
# Prompt block appended to the Stage-3 author prompt
# ---------------------------------------------------------------------------


def grounding_rules(video_class: str, tier: str) -> str:
    n_ws = HIDDEN.get(tier, 0) + 1
    ws_line = ("- This %s KPR QA has exactly %d web_search source(s), one per hidden physical spec.\n" % (tier.upper(), n_ws)
               if video_class == "B" else "- The QA has at least one web_search source.\n")
    return """
## Grounding requirements (checked automatically; a QA that fails is rejected)
""" + ws_line + """- Every web_search source in "sources" has these fields:
  "search_query": the exact query you sent to web_search for this value (the query must not contain the value),
  "url": a url that this web_search call returned,
  "verified_quote": a verbatim passage (max 200 characters) copied from that result's snippet that states the value,
  "key_information": "[object] [property] = [value] [unit]" with the value as the quote states it.
- Each web constant must have ONE value for the scene: its query uses only words the question states or the video
  shows, and returns one value (or one standard range) for the object as the question describes it. Prefer standards,
  regulations, datasheets and handbook values over forum posts and research statistics. If the object's material /
  official size / grade / standard decides the value, name it in one explicit sentence without a number ("Assume the
  ball is a FIBA size 7 basketball.").
- A query that asks for a range / tolerance / limits must return a range. A value the source states as a range
  (e.g. "567-650 g") is used as that range; do not replace it by its midpoint.
- The question never states a web value or a hidden value (in any unit), never mentions tools or searching (search,
  Google, look up, online, internet, code), and gives no method hint.
- The code prints every number the answer states (3-4 significant figures); for an interval it prints the low and the
  high end. The answer uses only numbers the code prints. Every input of the code is given by the question, the video
  readings or the web constants.
- TYPE_A / TYPE_B are not chosen in advance: they follow the inputs. With a range input (a web constant stated as a
  range, or a range the question states) the code propagates it and prints the low and the high end, the answer is
  "low-high unit" and answer_range is [low, high] (TYPE_B); otherwise the answer is one number with unit (TYPE_A).
  Never an invented +-%. TYPE_C: the answer starts with Yes / No and gives the deciding number(s). TYPE_D: the answer
  states the comparison and its ratio / difference.
"""


REJECTED = ("\n## A PREVIOUS ATTEMPT WAS REJECTED\nProblems:\n- %s\nBuild a new QA pair that avoids these problems.\n")

# ---------------------------------------------------------------------------
# Grounding of the agent's sources in its own tool calls
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_SCALES = (1, 1e2, 1e-2, 1e3, 1e-3, 1e6, 1e-6)


def _nums(s: Any) -> List[float]:
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(s or ""))
    return [float(x) for x in _NUM_RE.findall(s)]


def _consistent(text: Any, targets: List[float], tol: float = 0.02) -> bool:
    cand = [abs(x) for x in _nums(text) if abs(x) > 1e-12]
    return any(abs(x * s - y) <= tol * max(x * s, y) for x in cand for y in targets if y > 0 for s in _SCALES)


def tool_evidence(messages: List[Dict[str, Any]], extra: Optional[Dict[str, List[Dict[str, str]]]] = None
                  ) -> Tuple[Dict[str, str], Dict[str, List[str]], List[Dict]]:
    """url -> text the tools returned for it, url -> queries that returned it, executed code runs.
    extra = {query: raw Serper results} of searches run by the checks themselves (requery / qualify)."""
    calls = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls[tc.get("id")] = (fn.get("name"), args)
    texts, queries, runs = {}, {}, []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        name, args = calls.get(m.get("tool_call_id"), (None, {}))
        content = m.get("content") or ""
        if name == "web_search":
            try:
                res = json.loads(content)
            except json.JSONDecodeError:
                res = []
            for r in res if isinstance(res, list) else []:
                u = r.get("url") or r.get("link") if isinstance(r, dict) else None
                if u:
                    texts[u] = texts.get(u, "") + " " + (r.get("title") or "") + " " + (r.get("snippet") or "")
                    queries.setdefault(u, []).append(args.get("query", ""))
        elif name == "page_browser":
            urls = args.get("urls") or []
            for u in [urls] if isinstance(urls, str) else urls:
                texts[u] = texts.get(u, "") + " " + content
        elif name == "code_executor":
            runs.append({"code": args.get("code", ""), "out": content})
    for q, org in (extra or {}).items():
        for o in org or []:
            u = o.get("link") or o.get("url")
            if u:
                texts[u] = texts.get(u, "") + " " + (o.get("title") or "") + " " + (o.get("snippet") or "")
                queries.setdefault(u, []).append(q)
    return texts, queries, runs


# stray CJK / kana / Thai / Hangul tokens or words fused with numbers ("approximately1014") in generated text
GARBLED = re.compile(r"[぀-ヿ一-鿿฀-๿가-힯]|\b[a-z]{4,}\d")

_UNIT_SYN = {"grams": "g", "gram": "g", "kilograms": "kg", "kilogram": "kg", "meters": "m", "meter": "m",
             "metres": "m", "metre": "m", "centimeters": "cm", "centimetres": "cm", "millimeters": "mm",
             "millimetres": "mm", "seconds": "s", "second": "s", "sec": "s", "degrees": "°", "degree": "°",
             "deg": "°", "newtons": "n", "newton": "n", "joules": "j", "joule": "j", "percent": "%"}
_VAL_RE = re.compile(r"(\d+(?:\.\d+)?)((?:\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?)*)\s*([^\s\d,;:()\[\]]*)")


def _valued_numbers(text: str) -> List[Tuple[float, str]]:
    """(number, unit) pairs; every number of a range takes the unit written after the range"""
    out = []
    for m in _VAL_RE.finditer(re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)):
        unit = m.group(3).lower().rstrip(".")
        unit = _UNIT_SYN.get(unit, unit)
        if not re.match(r"^(%|°|[a-zµ/²³·°]+)$", unit) or len(unit) > 8:
            unit = ""
        for x in [m.group(1)] + re.findall(r"\d+(?:\.\d+)?", m.group(2)):
            out.append((float(x), unit))
    return out


def _leaked_hidden_values(question: str, hidden_cv_chosen: List[str]) -> List[str]:
    """hidden entries whose value (same number, same unit) is stated in the question"""
    q = _valued_numbers(question)
    bad = []
    for h in hidden_cv_chosen:
        h = str(h)
        val = h.split("=", 1)[1] if "=" in h else h
        for v, u in _valued_numbers(val):
            if v and any(abs(v - x) <= 1e-9 * max(v, x) and u == ux for x, ux in q):
                bad.append(h)
                break
    return bad


def _strip_values(query: str, values: List[float]) -> str:
    """the query without the retrieved numbers (and their units / quotes)"""
    def repl(m):
        ns = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(0))]
        return " " if any(_consistent(str(n), values, 0.001) for n in ns) else m.group(0)
    q = _VAL_RE.sub(repl, query)
    return re.sub(r"\s+", " ", q.replace('"', " ")).strip(" '")


def _url_key(u: str) -> str:
    return re.sub(r"[?#].*$", "", str(u)).rstrip("/").lower().replace("://www.", "://")


def validate_grounding(qa: Dict[str, Any], messages: List[Dict[str, Any]], difficulty: str, video_class: str,
                       extra: Optional[Dict[str, List[Dict[str, str]]]] = None, requery=None) -> List[str]:
    """Programmatic checks of a written QA; fills search_query from the real queries and code_executor
    key_information with the real stdout."""
    errs = []
    question_type = qa.get("question_type")
    if question_type not in ("TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"):
        errs.append(f"question_type is {question_type}, must be TYPE_A / TYPE_B / TYPE_C / TYPE_D")
    q = qa.get("question") or ""
    if re.search(r"web[_ ]search|search engine|google|code[_ ]executor|\btool", q, re.I):
        errs.append("the question text mentions a tool / search")
    chosen = qa.get("hidden_cv_chosen") or []
    lk = _leaked_hidden_values(q, chosen) if chosen else []
    if lk:
        errs.append("the question text states the hidden value(s) %s: describe them without the number, or keep "
                    "the number in the question and hide another retrieved property instead" % lk)
    n_hide = HIDDEN.get(difficulty, 0) + (1 if video_class == "B" else 0)
    # KPR easy QAs carry no hidden_cv_chosen (0 accepted as well)
    if len(chosen) != n_hide and not (video_class == "B" and difficulty == "easy" and not chosen):
        errs.append(f"hidden_cv_chosen must list exactly {n_hide} value(s) for {difficulty} "
                    f"{'KPR' if video_class == 'B' else 'MPR'}")
    steps = qa.get("annotated_solution") or []
    texts, queries, runs = tool_evidence(messages, extra)
    srcs = sources(qa)
    qa["sources"] = srcs
    ws = [s for s in srcs if s.get("type") == "web_search"]
    if not ws:
        errs.append("no web_search source")
    if video_class == "B" and ws and len(ws) != n_hide:
        errs.append(f"a {difficulty} KPR QA needs exactly {n_hide} web_search source(s) (one per hidden spec), has {len(ws)}")
    qpat = re.compile(r"web_search\(query=(['\"])(.*?)\1\)")
    ws_steps = [i for i, st in enumerate(steps) if qpat.search(str(st))] if isinstance(steps, list) else []
    for j, s in enumerate(ws):
        u, ki = s.get("url"), s.get("key_information") or ""
        if u not in texts:
            errs.append(f"web_search source url {u} was not returned by any web_search / page_browser call")
            continue
        kn = [x for x in _nums(ki) if x > 0]
        if kn and not any(_consistent(texts[u], [x], 0.001) for x in kn):
            errs.append(f"web_search source {u}: no number of \"{ki[:120]}\" appears in the returned text")
        vq = s.get("verified_quote")
        if not vq:
            errs.append(f"web_search source {u}: verified_quote missing - copy the passage of the result that states the value")
        elif not (quote_ok(vq, texts[u], kn[0]) if kn else False) and not (kn and _consistent(sci(vq), kn[:1], 0.001)
                                                                          and quote_ok(vq, texts[u], [])):
            errs.append(f"web_search source {u}: verified_quote \"{str(vq)[:120]}\" is not a verbatim passage of the "
                        f"result that states the value")
        if u not in queries:  # found by page_browser only
            continue
        # the solution cites the query that actually returned u: the query of the step naming u (else the j-th
        # web_search step when steps and sources correspond one to one) is set to the real one; a query stating the
        # retrieved value itself is not accepted
        real = [x for x in dict.fromkeys(queries[u]) if x and not (kn and _consistent(x, kn, 0.001))]
        if not real and requery is not None:
            # search again without the value: accepted when the same page comes back
            for x in dict.fromkeys(queries[u]):
                sq = _strip_values(x, kn)
                if len(sq.split()) < 2 or sq == x:
                    continue
                hits = requery(sq) or []
                if any(_url_key(h.get("url") or h.get("link")) == _url_key(u) for h in hits):
                    real = [sq]
                    queries[u] = queries[u] + [sq]
                    break
        if not real:
            errs.append(f"every web_search query that returned {u} contains the retrieved value; search without it")
            continue
        at = [i for i in ws_steps if u in steps[i]] or ([ws_steps[j]] if len(ws_steps) == len(ws) else [])
        if not at:
            errs.append(f"the solution must contain the web_search(query='...') that returned {u}")
            continue
        cited = qpat.search(steps[at[0]]).group(2)
        sq0 = s.get("search_query")
        qq = cited if cited in real else (sq0 if sq0 in real else real[0])
        steps[at[0]] = qpat.sub(lambda m: "web_search(query='%s')" % qq.replace("'", '"'), steps[at[0]], count=1)
        s["search_query"] = qq
    if ws_steps:
        qa["annotated_solution"] = steps
    codes = [s for s in srcs if s.get("type") == "code_executor"]
    if not codes:
        errs.append("no code_executor source")
    outs = []
    for i, s in enumerate(codes):
        out = run_code(s.get("code") or "", timeout=30)
        outs.append(out)
        if out.startswith("Error") or not out:
            errs.append(f"code block {i + 1} {'fails: ' + out[:150] if out else 'prints nothing'}")
        else:
            s["key_information"] = out[:500]
    ans = str(qa.get("answer") or "")
    an = [x for x in _nums(ans) if x > 0]
    if outs and outs[-1] and not outs[-1].startswith("Error") and an and not _consistent(outs[-1], an):
        errs.append(f"the last code block prints \"{outs[-1][:150]}\" but the answer is \"{ans[:150]}\"")
    ar = qa.get("answer_range")
    if question_type == "TYPE_B" and outs and outs[-1] and not outs[-1].startswith("Error"):
        # the range stated in the answer text, when both endpoints are printed by the last code block
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:–|—|-|to)\s*(\d+(?:\.\d+)?)", re.sub(r"(?<=\d),(?=\d{3}\b)", "", ans))
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if 0 < lo < hi and _consistent(outs[-1], [lo], 0.001) and _consistent(outs[-1], [hi], 0.001) \
                    and ar != [lo, hi]:
                qa["answer_range"] = ar = [lo, hi]
    if question_type == "TYPE_A" and (ar not in (None, [], "null") or re.search(r"\d\s*(?:–|—|to|-)\s*\d", ans)):
        errs.append("TYPE_A: single value answer, answer_range null")
    if question_type == "TYPE_A" and ar in ([], "null"):
        qa["answer_range"] = None
    if isinstance(ar, list) and (len(ar) != 2 or not all(isinstance(x, (int, float)) for x in ar) or ar[0] > ar[1]
                                 or (outs and not all(_consistent(outs[-1], [abs(x)]) for x in ar if x))):
        errs.append(f"answer_range {ar} must be [lo, hi] as printed by the last code block")
    if isinstance(ar, list) and len(ar) == 2 and all(isinstance(x, (int, float)) for x in ar) \
            and not all(_consistent(ans, [abs(x)]) for x in ar if x):
        errs.append(f"the answer text \"{ans[:150]}\" must state the answer_range endpoints {ar}")
    body = [q, ans] + [str(x) for x in (steps if isinstance(steps, list) else [steps])]
    if any(GARBLED.search(re.sub(r"https?://\S+|www\.\S+", "", t)) for t in body):
        errs.append("garbled text in the question / answer / solution (stray characters or fused words); write it cleanly")
    if question_type == "TYPE_C" and not re.match(r"\W*(yes|no)\b", ans, re.I):
        errs.append("TYPE_C answer must start with Yes or No")
    if question_type in ("TYPE_A", "TYPE_B") and (re.match(r"\W*(yes|no)\b", ans, re.I) or not an):
        errs.append(f"{question_type} answer must be a numerical value{' or range' if question_type == 'TYPE_B' else ''}"
                    " with unit, not a Yes/No verdict")
    return errs


def evidence_of(qa: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """raw Serper results of every cited search_query (for checks of a QA without the agent's messages)"""
    return {s["search_query"]: serper(s["search_query"]) or [] for s in wsrc(qa) if s.get("search_query")}


# ---------------------------------------------------------------------------
# Static checks: leaks, tool wording, sources
# ---------------------------------------------------------------------------

DESIG = re.compile(r"(?:[A-Za-z#/-]|size|No\.?|ISO|SAE|ASTM|DIN|EN|class|grade|type)\s?$", re.I)
FORUM_RE = re.compile(r"forum|reddit\.|quora\.|answers\.|stackexchange\.|chegg\.|brainly\.|/t/|/threads?/|community\.")
REFS = {"engineering toolbox": "engineeringtoolbox", "engineeringtoolbox": "engineeringtoolbox", "hyperphysics": "hyperphysics",
        "wikipedia": "wikipedia", "wood handbook": "fpl.fs.usda", "matweb": "matweb", "nist": "nist.gov"}


def spurious(text: Any) -> List[str]:
    return [m for m in re.findall(r"\d+\.\d+", str(text)) if len(m.replace(".", "").lstrip("0")) >= 8]


def static_checks(qa: Dict[str, Any]) -> List[str]:
    P = []
    q = qa.get("question") or ""
    m = TOOL_RE.search(q)
    if m:
        P.append('The question mentions tools / searching ("%s"); rephrase without it.' % q[max(0, m.start() - 40):m.end() + 20])
    lk = leaked(q, hidden_nums(qa))
    if lk:
        P.append("The question states the hidden value(s) %s; remove them." % lk)
    vid = [x for s in sources(qa) if s.get("type") == "video_observation" for x in nums(s.get("key_information"))]
    webv = []
    for s in wsrc(qa):
        ki = str(s.get("key_information") or "")
        used = ki.split("used as", 1)[1] if "used as" in ki else ki
        vals = [x for x in nums(used) if not any(abs(x - y) <= 1e-9 * max(abs(x), 1e-12) for y in vid)]
        webv += vals
        lk = leaked(q, vals)
        if lk:
            P.append("The question writes the web value(s) %s of \"%s\"; the solver must obtain them." % (lk, ki[:120]))
    triv = [v for v in webv if any(abs(v - t) <= 0.003 * t for t in (9.80665, 9.8, 3.14159, 2.54, 0.3048, 0.4536, 1.609))]
    if triv:
        P.append("A web constant is g, pi or a unit conversion (%s); the looked-up value must be a physical property of "
                 "the scene." % triv[:2])
    sp = spurious(qa.get("answer")) + [x for h in qa.get("hidden_cv_chosen") or [] for x in spurious(h)]
    if sp:
        P.append("Numbers with spurious precision %s in the answer / hidden values; round them as the code prints them "
                 "(3-4 significant figures)." % sp[:3])
    urls = [(s.get("url") or "").lower() for s in wsrc(qa)]
    if any(not u for u in urls):
        P.append("A web constant has no source URL; use a value whose page can be identified.")
    fu = [u for u in urls if FORUM_RE.search(u)]
    if fu:
        P.append("Forum / Q&A source %s; use a manufacturer, standard, handbook, reference-table or peer-reviewed source "
                 "instead." % fu[:1])
    ql = q.lower()
    for name, dom in REFS.items():
        if name in ql and not any(dom in u for u in urls):
            P.append('The question names "%s" but no web constant comes from %s; name only the reference the value is '
                     'actually taken from.' % (name, dom))
    return P


# ---------------------------------------------------------------------------
# Answer form and range form
# ---------------------------------------------------------------------------

_NUMI = r"[-+]?\d+(?:\.\d+)?(?:\s*[x×]\s*10\^?[-+]?\d+)?"
INTERVAL = re.compile(r"(?<![eE\d.])(?:%s)\s*[^\d\s,;()eE]{0,12}\s*(?:-|–|—|to|~)\s*(?:%s)|between\s+%s.{0,20}?\s+and\s+%s"
                      % (_NUMI, _NUMI, _NUMI, _NUMI), re.I)
PCT = re.compile(r"±\s*\d+(?:\.\d+)?\s*%|\+/-\s*\d+(?:\.\d+)?\s*%|\baccepted range\b", re.I)
RANGE = re.compile(r"\d\s*[a-zA-Z°%/³²]*\s*(?:-|–|—|to)\s*\d")
RANGEQ = re.compile(r"\b(range|ranges|tolerance|tolerances|between|limits?|min(?:imum)?\s*(?:and|/|-)\s*max(?:imum)?|"
                    r"from\s+\w+\s+to)\b", re.I)
COMMONN = [9.81, 9.8, 9.807, 9.80665, 1000, 100, 1e3, 1e-3, 1e-6, 1e6, 0.01, 0.001, 3600, 60, 273.15, 101325, 0.0254, 0.3048,
           0.453592, 0.45359237, 28.3495, 0.0283495, 1609.34, 0.44704, 3.6, 4.184, 2.54, 12, 1e-2, 1e-9, 0.5]


def answer_form(qa: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """('interval' | 'single', interval stated in the answer text outside parentheses, answer_range is [low < high])"""
    a = re.sub(r"\([^)]*\)", "", str(qa.get("answer") or "")).replace("−", "-")
    rng = qa.get("answer_range")
    ok = isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng) and rng[0] < rng[1]
    iv = bool(INTERVAL.search(a))
    return ("interval" if iv or ok else "single"), iv, ok


def form_defects(qa: Dict[str, Any], out: str) -> List[str]:
    """TYPE_A one value, TYPE_B an interval in the answer + answer_range, no invented +-%"""
    t, (f, iv, ok) = qa.get("question_type"), answer_form(qa)
    D = []
    if t == "TYPE_A" and f == "interval":
        D.append("the answer is an interval but the type is TYPE_A (one value): give one number, or make it TYPE_B if the "
                 "interval comes from range inputs")
    if t == "TYPE_B" and not (iv and ok):
        D.append('TYPE_B: the answer must state the interval "low-high unit" and answer_range must be [low, high]; one '
                 "value (with a remark) is TYPE_A")
    if PCT.search(str(qa.get("answer"))) or PCT.search(out or ""):
        D.append('an invented +-% tolerance / "accepted range": remove it (one value -> TYPE_A; an interval only from range '
                 "inputs)")
    return D


def new_inputs(qa: Dict[str, Any], video_class: str) -> List[float]:
    """numeric literals of the code that the question, the web constants and (MPR) the video readings do not give, and
    that are no common constant / conversion / small integer: inputs the solver cannot know"""
    have = nums(qa.get("question")) + [x for s in wsrc(qa) for x in nums(s.get("key_information")) + nums(s.get("verified_quote"))]
    have += hidden_nums(qa) + COMMONN
    if video_class == "A":  # MPR: the solver measures every CV reading from the video with the pose tool
        have += [x for s in sources(qa) if s.get("type") == "video_observation" for x in nums(s.get("key_information"))]
    code = re.sub(r"#[^\n]*", "", code_of(qa))
    code = re.sub(r"(f?[\"'])(?:\\.|(?!\1).)*\1", "", code)  # strings (print formats) and comments
    lit = [float(m) for m in re.findall(r"(?<![\w.])(\d+\.?\d*(?:[eE][-+]?\d+)?)", code)]
    return sorted({x for x in lit if not (x.is_integer() and abs(x) <= 10) and not any(
        near(x, h) or near(x, h * 1000) or near(x, h / 1000) or near(x, h * 100) or near(x, h / 100) for h in have)})


def nums_ok(answer: Any, rng: Any, out: str, qt: str) -> Optional[str]:
    on = nums(out)
    answer = re.sub(r"\([^)]*\)", "", str(answer))  # a parenthetical restatement ("(≈0.56-0.59 kJ)") is not checked, as in answer_form
    an = [x for x in nums(answer) if not (float(x).is_integer() and abs(x) <= 10)] or nums(answer)
    if not an:
        return "the answer has no number"
    bad = [x for x in an if not any(near(x, o, 0.006) for o in on)]
    if bad:
        return "the answer number(s) %s are not in the code output" % bad
    if qt == "TYPE_B":
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng) and rng[0] < rng[1]):
            return "TYPE_B needs answer_range [low, high] with low < high"
        if not all(any(near(v, o, 0.006) for o in on) for v in rng):
            return "answer_range %s is not printed by the code" % rng
    return None


def _is_range(v: Any) -> bool:
    return (isinstance(v, list) and len(set(map(str, v))) > 1) or bool(RANGE.search(str(v)))


def form_checks(qa: Dict[str, Any], video_class: str, out: Optional[str] = None) -> List[str]:
    """answer form of the question type, range form of every web constant, inputs the solver cannot know"""
    out = run_code(code_of(qa)) if out is None else out
    t, a = qa.get("question_type"), str(qa.get("answer") or "")
    P = form_defects(qa, out)
    if not out.startswith("Error"):
        e = nums_ok(a, qa.get("answer_range"), out, t) if t in ("TYPE_A", "TYPE_B") else None
        if e:
            P.append(e)
    if t == "TYPE_C" and not re.match(r"\W*(yes|no)\b", a, re.I):
        P.append("TYPE_C: the answer must start with Yes / No and give the deciding number(s).")
    if t == "TYPE_D" and not re.search(r"[<>%]|greater|less|more|higher|lower|larger|smaller|faster|slower|ratio|times|exceed|"
                                       r"than|compar|increase|decrease|sufficient|differ|change|fraction|percent", a, re.I):
        P.append("TYPE_D: the answer must state the comparison (A > B) and its ratio / difference.")
    if t == "TYPE_B" and not (any(_is_range(s.get("key_information")) or _is_range(s.get("verified_quote")) for s in wsrc(qa))
                              or RANGE.search(str(qa.get("question") or ""))):
        P.append("TYPE_B: the interval must come from a range input (a web constant stated as a range, or a range the "
                 "question states).")
    for s in wsrc(qa):
        sq, ki = str(s.get("search_query") or ""), str(s.get("key_information") or "")
        used = ki.split("=", 1)[1] if "=" in ki else ki
        rng = _is_range(used)
        if not rng and RANGEQ.search(sq):
            P.append('web constant "%s": the query "%s" asks for a range but its value is a single number (the query must '
                     "match the value it returns)" % (ki[:100], sq))
    ni = new_inputs(qa, video_class)
    if ni:
        P.append("The code uses input value(s) %s that neither the question nor the web constants give - the solver cannot "
                 'know them. State each as a given input in the question (a size estimate "about 1.6 cm across" is fine) '
                 "or do not use it." % ni)
    return P


def range_input(qa: Dict[str, Any]) -> bool:
    """a web constant stated as a range, or a range the question states"""
    return any(_is_range((s.get("key_information") or "").split("=", 1)[-1]) for s in wsrc(qa)) \
        or bool(RANGE.search(str(qa.get("question") or "")))


async def settle_answer(qa: Dict[str, Any]) -> Dict[str, Any]:
    """numeric QAs (TYPE_A / TYPE_B): the type follows the inputs (TYPE_B with a range input, else TYPE_A), not the
    label the agent chose; when the type changes or the answer / answer_range are not the numbers the code prints, the
    answer and the solution are rewritten from the code output (FINAL). Other QAs and failing code are left to the checks."""
    qt = qa.get("question_type")
    code = code_of(qa)
    if qt not in ("TYPE_A", "TYPE_B") or not code:
        return qa
    out = await asyncio.to_thread(run_code, code)
    if not out or out.startswith("Error"):
        return qa
    nt = "TYPE_B" if range_input(qa) else "TYPE_A"
    if nt == qt and not form_defects(qa, out) and nums_ok(qa.get("answer"), qa.get("answer_range"), out, nt) is None:
        return qa
    new = copy.deepcopy(qa)
    new["question_type"] = nt
    consts = "\n".join('%s | %s | "%s"' % (s.get("search_query"), s.get("key_information"), str(s.get("verified_quote"))[:200])
                       for s in wsrc(qa))
    fin = pj(await llm(FINAL % (qa["question"], nt, "\n".join(map(str, qa.get("annotated_solution") or [])), qa.get("answer"),
                                consts, code, out[:3000], qa.get("answer_range")))) or {}
    if fin.get("answer") and isinstance(fin.get("annotated_solution"), list):
        new["annotated_solution"] = [str(x) for x in fin["annotated_solution"]]
        new["answer"] = str(fin["answer"])
        new["answer_range"] = fin.get("answer_range") if nt == "TYPE_B" else None
        new["total_steps"] = len(new["annotated_solution"])
    elif nt == "TYPE_A":
        new["answer_range"] = None
    logger.info(f"  [Stage 3] answer settled from the inputs / code output: {qt} -> {nt}, answer {new.get('answer')!r}")
    return new


async def check_written(qa: Dict[str, Any], messages: List[Dict[str, Any]], difficulty: str, video_class: str,
                        extra: Optional[Dict[str, List[Dict[str, str]]]] = None) -> List[str]:
    """all deterministic Stage-3 checks (grounding, leaks, answer form, range form)"""
    errs = await asyncio.to_thread(validate_grounding, qa, messages, difficulty, video_class, extra, serper)
    errs += static_checks(qa)
    errs += await asyncio.to_thread(form_checks, qa, video_class)
    return list(dict.fromkeys(errs))


def evidence_record(qa: Dict[str, Any], messages: List[Dict[str, Any]],
                    extra: Optional[Dict[str, List[Dict[str, str]]]] = None) -> List[Dict[str, str]]:
    """the search result text behind every web constant (query, url, returned title + snippet / page text)"""
    texts, _, _ = tool_evidence(messages, extra)
    return [{"query": s.get("search_query") or "", "url": s.get("url") or "", "text": texts.get(s.get("url"), "")[:3000]}
            for s in wsrc(qa)]


def sig(qa: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps([qa.get("question"), qa.get("answer"), qa.get("answer_range"), qa.get("question_type"),
                                    code_of(qa), [(s.get("search_query"), s.get("key_information")) for s in wsrc(qa)]],
                                   ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Unique-value search test
# ---------------------------------------------------------------------------

QUERIES = """A student must answer the physics question below and has to look up one quantity with Google.
QUESTION: %s
The reference solution looked it up as (numbers hidden): %s
Reference query (numbers hidden): %s
1. Name the SAME kind of quantity the reference looked up (if it looked up a ratio / percentage / coefficient, name that
   ratio / percentage / coefficient, not a derived absolute value), generically, using ONLY qualifiers (object, material, population, size, condition) that are stated or
   clearly implied by the QUESTION; drop qualifiers that only the reference adds.
2. Write 3 different Google queries a careful student would write from the QUESTION alone for that quantity (no numbers,
   no qualifiers that the question does not state or imply).
Return ONLY JSON: {"quantity": "...", "queries": ["...", "...", "..."]}"""

EXTRACT = """QUESTION: %s
QUANTITY NEEDED: %s
SEARCH RESULTS:
%s

List every value of the QUANTITY NEEDED that the search results state for an object / material / condition consistent with
what the QUESTION describes. A value for a clearly different object or condition does not count; a value for a general
class that includes the question's object counts; a stated range counts as one value. Copy the quote verbatim from the
results (max 200 characters, must contain the number).
Return ONLY JSON: {"values": [{"value": <number or [low, high]>, "unit": "...", "result": <result number>, "quote": "..."}]}
(empty list if none)"""

EXTRACT2 = """QUESTION: %s
QUANTITY NEEDED: %s
SEARCH RESULTS:
%s

List EVERY number in the search results that is stated as a value of the QUANTITY NEEDED (any object), and say which object /
material / condition the result gives it for. same_object = true if that object / material / condition is the one the
QUESTION describes (or a general class that includes it), false if it is a clearly different one (another size, grade,
material, temperature, product). A stated range counts as one value. Copy the quote verbatim (max 200 characters, must
contain the number).
Return ONLY JSON: {"values": [{"value": <number or [low, high]>, "unit": "...", "refers_to": "...", "same_object": true|false,
"quote": "..."}]} (empty list if none)"""

SUBST = """Reference solution code of a physics question:
```python
%s
```
In this code the web-searched quantity "%s" has the reference value: %s
Candidate value from another source: %s %s (quote: "%s")
1. same: is the candidate equal to the reference value after unit conversion (within 2%%; a range counts as equal if it
   contains the reference value)?
2. If not, rewrite the code so it uses the candidate value instead - change ONLY numeric literals (convert units yourself;
   for a range use the midpoint unless the code itself uses a low / high pair). Keep everything else identical.
Return ONLY JSON: {"same": true/false, "code": "<full edited code, or empty if same>"}"""

VALID = """A physics benchmark question requires looking up "%s" on the web; the solver sees only the QUESTION text.
QUESTION: %s

Two values for this quantity were found on two different web pages. NEITHER value nor quote is part of the question.
Value 1: %s %s (web quote: "%s")
Value 2: %s %s (web quote: "%s")
Does the QUESTION text contain a detail (object, model, size, material, population, condition) that makes one value fit and
the other not? Reply "1" or "2" if the question points to that value only, "both" if a careful solver reading only the
question could reasonably use either value.
Return ONLY JSON: {"verdict": "1|2|both", "why": "..."}"""

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

PARA = """Rewrite this web search query the way another person might type it: the same words in another order, or common
synonyms / spellings ("weight" for "mass", "metre" for "meter"). Keep every qualifier (material, grade, standard, size,
condition); add nothing, drop nothing. Query: %s
Return ONLY JSON: {"query": "..."}"""

STD = re.compile(r"\b(rules?|regulations?|official|standards?|specifications?|ISO|ASTM|EN \d|DIN|FIBA|FIFA|ITTF|NBA|NCAA|UCI|"
                 r"World Athletics|IAAF|NIST|CRC|handbook)\b", re.I)


def _num(v: Any) -> Optional[List[float]]:
    try:
        return [float(x) for x in (v if isinstance(v, list) else [v])]
    except Exception:
        return None


def _samev(a: Dict, b: Dict) -> bool:
    a, b = _num(a.get("value")), _num(b.get("value"))
    return bool(a and b and len(a) == len(b) and all(abs(x - y) <= 0.02 * max(abs(y), 1e-9) for x, y in zip(a, b)))


class SerperFailed(Exception):
    pass


async def base_const(qa: Dict[str, Any], src: Dict[str, Any], base_code: str, base_out: str,
                     quantity: Optional[str]) -> Dict[str, Any]:
    """every value of the quantity that the base query returns, substituted into the code and scored with the ACC"""
    ki, sq = src.get("key_information", ""), src.get("search_query", "")
    d = {"ref": ki, "ref_query": sq, "url": src.get("url", "")}
    if not sq:
        d["no_base_query"] = True
        return d
    if not quantity:
        j = pj(await llm(QUERIES % (qa["question"], mask(ki), mask(sq)))) or {}
        quantity = j.get("quantity") or mask(ki)
    d["quantity"] = quantity
    res = await asearch(sq)
    if res is None:
        raise SerperFailed(sq)
    keyn = nums(ki)
    d["reach_numeric"] = bool(keyn) and has(keyn, res)
    e = pj(await llm(EXTRACT % (qa["question"], quantity, results_block([res])))) or {}
    vals = [v for v in e.get("values") or [] if isinstance(v, dict)]
    # second, object-tagged extraction: a missed value would make a constant look unique, a value of another object
    # would make it look ambiguous
    e2 = pj(await llm(EXTRACT2 % (qa["question"], quantity, results_block([res])))) or {}
    v2 = [v for v in e2.get("values") or [] if isinstance(v, dict)]
    other = [v for v in v2 if v.get("same_object") is False]
    d["other_object"] = [{k: v.get(k) for k in ("value", "unit", "refers_to", "quote")} for v in other]
    vals = [v for v in vals if not any(_samev(v, o) for o in other)] + [v for v in v2 if v.get("same_object") is not False]
    uniq = []
    for v in vals:
        if not (v.get("value") is not None and quote_ok(v.get("quote"), res, v.get("value"))):
            continue
        vv = _num(v["value"])
        if not vv:
            continue
        if not any(len(vv) == len(u["_v"]) and all(abs(a - b) <= 0.02 * max(abs(b), 1e-9) for a, b in zip(vv, u["_v"]))
                   and v.get("unit") == u.get("unit") for u in uniq):
            uniq.append(dict(v, _v=vv))
    rows = []
    for c in uniq[:12]:
        s = pj(await llm(SUBST % (base_code, quantity, ki, c["value"], c.get("unit", ""), c["quote"]))) or {}
        row = {k: c.get(k) for k in ("value", "unit", "quote")}
        if s.get("same"):
            row["same"] = True
            rows.append(row)
            continue
        code = s.get("code") or ""
        if not code or canon(code) != canon(base_code):
            row["unchecked"] = "edit_invalid"
            rows.append(row)
            continue
        o = await arun(code)
        row["code"], row["out"] = code, o
        if o.startswith("Error"):
            row["unchecked"] = "run_error"
            rows.append(row)
            continue
        a = await alt_answer(qa, base_out, o)
        row["alt_answer"] = a
        row["acc"] = await acc(qa, a) if a else None
        if row["acc"] is None:
            row["unchecked"] = "no_score"
        elif row["acc"] < 0.5:
            o1, o2 = (ki, "", src.get("verified_quote", "")), (c["value"], c.get("unit", ""), c["quote"])
            flip = coin(qa["question"], c["quote"])
            v = pj(await llm(VALID % ((quantity, qa["question"]) + (o2 + o1 if flip else o1 + o2)))) or {}
            row["valid_verdict"] = {"1": "alternative" if flip else "reference",
                                    "2": "reference" if flip else "alternative", "both": "both"}.get(str(v.get("verdict")))
            row["valid_why"] = v.get("why")
        rows.append(row)
    d["candidates"] = rows
    d["n_values"] = len(rows)
    d["reach"] = d["reach_numeric"] or any(r.get("same") for r in rows)
    d["conflicts"] = [r for r in rows if r.get("acc") is not None and r["acc"] < 0.5
                      and r.get("valid_verdict") in ("both", "alternative")]
    return d


async def rejudge_conflicts(qa: Dict[str, Any], src: Dict[str, Any], dc: Dict[str, Any], fr, fk: str, objects: str) -> List[Dict]:
    """the conflicts of one constant re-judged with the video frames; returns the ones question + video do not resolve"""
    vq = src.get("verified_quote", "")
    for x in dc.get("conflicts") or []:
        o1, o2 = (dc.get("ref"), "", vq), (x.get("value"), x.get("unit") or "", x.get("quote") or "")
        flip = coin(qa["question"], x.get("quote"), "v")
        j = pj(await vlm(VALIDV % ((dc.get("quantity"), qa["question"], objects) + (o2 + o1 if flip else o1 + o2)), fr, fk)) or {}
        x["video_verdict"] = {"1": "alternative" if flip else "reference", "2": "reference" if flip else "alternative",
                              "both": "both"}.get(str(j.get("verdict")))
        x["video_why"] = j.get("why")
    return [x for x in dc.get("conflicts") or [] if x.get("video_verdict") != "reference"]


def link_of(org: List[Dict[str, str]], quote: Any) -> str:
    norm = lambda t: re.sub(r"\s+", " ", str(t or "")).strip().lower()
    qt = norm(quote)
    if len(qt) < 15:
        return ""
    for o in org:
        if qt[:30] in norm(o.get("snippet")):
            return o["link"]
    for k in range(0, max(len(qt) - 25, 1), 8):
        w = qt[k:k + 25]
        for o in org:
            if len(w) >= 15 and w in norm(o.get("snippet")):
                return o["link"]
    return ""


def site(link: str) -> str:
    from urllib.parse import urlparse
    h = urlparse(link or "").netloc.lower()
    return h[4:] if h.startswith("www.") else h


def snippet_with(blob: Any, ki: Any) -> str:
    ns = nums(ki)
    for p in re.split(r" \| |: ", str(blob or "")):
        if ns and any(has([n], p) for n in ns):
            return p.strip()[:200]
    return ""


def objects_text(objects: Optional[List[Dict[str, Any]]]) -> str:
    return "; ".join("%s (%s)" % (o.get("object"), o.get("seen")) for o in objects or [] if isinstance(o, dict)) or "n/a"


async def unique(qa: Dict[str, Any], src: Dict[str, Any], code: str, out: str, quantity: Optional[str], fr, fk: str,
                 objects: str = "n/a", para: bool = True) -> Dict[str, Any]:
    """the unique-value test of one web constant -> {'ok', 'why', 'd', 'quote', 'url', 'support', 'paraphrase'}
    (raises SerperFailed): every value the query returns is substituted into the code and scored with the official
    ACC; ACC < .5 is a conflict unless question + frames single out the reference; >= 2 sites (or one standard /
    regulation / handbook quote) state the value; a paraphrase of the query returns no conflicting value."""
    async def run(q):
        dc = await base_const(qa, dict(src, search_query=q), code, out, quantity)
        conf = await rejudge_conflicts(qa, src, dc, fr, fk, objects) if dc.get("conflicts") else []
        return dc, conf
    q = src.get("search_query") or ""
    dc, conf = await run(q)
    res = {"ok": False, "why": "", "d": dc, "quote": "", "url": "", "support": []}
    say = lambda xs: "; ".join('%s %s ("%s") -> answer %s' % (x.get("value"), x.get("unit") or "", (x.get("quote") or "")[:120],
                                                            x.get("alt_answer")) for x in xs[:3])
    if dc.get("no_base_query"):
        res["why"] = "the web constant has no search query"
        return res
    if not dc.get("reach"):
        res["why"] = 'the query "%s" does not return the value "%s"' % (q, str(src.get("key_information"))[:120])
        return res
    if conf:
        res["why"] = 'the query "%s" also returns %s' % (q, say(conf))
        return res
    agree = [r for r in dc.get("candidates") or [] if r.get("same") or (r.get("acc") or 0) >= .5]
    blob = await asearch(q)
    quote = next((r["quote"] for r in agree if r.get("same") and r.get("quote")), "") or snippet_with(blob, src.get("key_information"))
    org = await alinks(q)
    sites = {site(link_of(org, r.get("quote"))) for r in agree} | {site(link_of(org, quote))}
    sites.discard("")
    res.update(quote=quote, url=link_of(org, quote), support=sorted(sites))
    if len(sites) < 2 and not any(STD.search(str(x)) for x in [quote] + [r.get("quote") for r in agree]):
        res["why"] = 'only %s states the value for the query "%s"; need two sites or a standard / handbook quote' % (
            sorted(sites) or "one snippet", q)
        return res
    if para:
        pq = str((pj(await llm(PARA % q)) or {}).get("query") or "").strip()
        if pq and pq.lower() != q.lower():
            d2, c2 = await run(pq)
            res["paraphrase"] = {"query": pq, "reach": d2.get("reach"), "conflicts": len(c2)}
            if c2:
                res["why"] = 'the paraphrased query "%s" returns other values: %s' % (pq, say(c2))
                return res
    res["ok"] = True
    return res


def pick_value(dc: Dict[str, Any], qt: str) -> Optional[Dict[str, Any]]:
    """the best-supported group of the values a query returned (other than the old value) -> one row with its code"""
    rows = [r for r in dc.get("candidates") or [] if r.get("code") and r.get("alt_answer")
            and not str(r.get("out", "Error")).startswith("Error")]
    groups = []
    for r in rows:
        x = U.si(r["value"], r.get("unit"))
        if not x:
            continue
        for g in groups:
            if g["dim"] == x[1] and U.agree(g["iv"], x[0], 0.03):
                g["rows"].append(r)
                break
        else:
            groups.append({"dim": x[1], "iv": x[0], "rows": [r]})
    if not groups:
        return None
    g = max(groups, key=lambda g: len(g["rows"]))
    rng = lambda r: isinstance(r["value"], list) and len(set(map(str, r["value"]))) > 1
    return next((r for r in g["rows"] if rng(r) == (qt == "TYPE_B")), g["rows"][0])


# ---------------------------------------------------------------------------
# Resolution of a non-unique constant: requery / qualify
# ---------------------------------------------------------------------------

CANDS = """You repair one web constant of a physics video QA whose base query does not return ONE value. The frames of the
video are attached (timestamps given). Keep the scene, the object and the asked quantity.
QUESTION: %s
WEB CONSTANT: %s
BASE QUERY: %s
PROBLEM: %s
OTHER WEB CONSTANTS OF THE QA: %s
The fix is a NEW BASE QUERY for the SAME quantity of the SAME object that returns ONE value (or one standard range).
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
Never move the constant to another object and never change the asked quantity.
Return ONLY JSON: {"candidates": [{"action": "requery|qualify", "assumption": "<sentence added to the question, or empty>",
"quantity": "<exact quantity with object / material / condition>", "base_query": "...", "range_spec": true|false}]}"""

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


def insert_ass(q: str, sents: List[str]) -> str:
    """assumption sentences go just before the interrogative sentence"""
    sents = [s.strip() for s in sents if s and s.strip()]
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    have = {norm(p) for p in re.split(r"(?<=[.!?])\s+", q.strip())}
    out = []
    for x in sents:
        if norm(x) not in have:
            have.add(norm(x))
            out.append(x)
    if not out:
        return q
    add = " ".join(s if s[-1] in ".!" else s + "." for s in out)
    parts = re.split(r"(?<=[.!?])\s+", q.strip())
    k = max((i for i, p in enumerate(parts) if "?" in p), default=None)
    return q.strip() + " " + add if k is None else " ".join(parts[:k] + [add] + parts[k:])


async def const_candidate(qa: Dict[str, Any], j: int, x: Dict[str, Any], code: str, out: str, fr, fk: str,
                          objects: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """one (assumption, base query) candidate for web constant j -> (resolution, '') | (None, why).
    rank 0: the query returns the old value uniquely (only query / assumption change);
    rank 1: it returns one different value uniquely (value and answer change)"""
    a = str(x.get("assumption") or "").strip() if x.get("action") == "qualify" else ""
    q = str(x.get("base_query") or "").strip()
    if not q or nums(q):
        return None, 'candidate query "%s" is empty or has a number' % q
    quantity = str(x.get("quantity") or "") or None
    qa0 = copy.deepcopy(qa)
    qa0["question"] = insert_ass(qa["question"], [a])
    src = dict(wsrc(qa)[j], search_query=q)
    u = await unique(qa0, src, code, out, quantity, fr, fk, objects)
    base = dict(query=q, assumption=a, action=x.get("action"), quantity=quantity)
    if u["ok"] and u["url"]:
        return dict(base, rank=0, quote=u["quote"], url=u["url"], support=u["support"], paraphrase=u.get("paraphrase")), ""
    best = pick_value(u["d"], qa["question_type"])
    if not best:
        return None, 'query "%s": %s' % (q, u["why"] or "no url for the quote")
    qa1 = copy.deepcopy(qa0)
    qa1.update(answer=best["alt_answer"], answer_range=None)
    quant = u["d"].get("quantity") or quantity or ""
    val = "-".join(map(str, best["value"])) if isinstance(best["value"], list) else best["value"]
    src1 = dict(src, key_information="%s = %s %s" % (quant, val, best.get("unit") or ""), verified_quote=best["quote"])
    u1 = await unique(qa1, src1, best["code"], best["out"], quant, fr, fk, objects)
    if not (u1["ok"] and (u1["url"] or u["url"])):
        return None, 'query "%s": old value: %s; its value %s %s: %s' % (q, u["why"], best["value"], best.get("unit") or "",
                                                                          u1["why"] or "no url for the quote")
    return dict(base, rank=1, quote=u1["quote"] or best["quote"], url=u1["url"] or u["url"], support=u1["support"],
                paraphrase=u1.get("paraphrase"), key_information=src1["key_information"], value=best["value"],
                unit=best.get("unit") or "", code=best["code"], out=best["out"], alt_answer=best["alt_answer"]), ""


async def apply_resolution(qa: Dict[str, Any], j: int, r: Dict[str, Any], video_class: str, out0: str) -> Dict[str, Any]:
    """the QA with web constant j re-sourced as resolved by const_candidate"""
    new = copy.deepcopy(qa)
    new["sources"] = sources(new)
    new["question"] = insert_ass(new["question"], [r.get("assumption") or ""])
    w = wsrc(new)[j]
    old_ki = w.get("key_information")
    w.update(search_query=r["query"], url=r["url"], verified_quote=r["quote"])
    steps = new.get("annotated_solution") or []
    qpat = re.compile(r"web_search\(query=(['\"])(.*?)\1\)")
    ws_steps = [i for i, st in enumerate(steps) if qpat.search(str(st))]
    if r["rank"] == 0:
        if len(ws_steps) == len(wsrc(new)):
            i = ws_steps[j]
            steps[i] = qpat.sub(lambda m: "web_search(query='%s')" % r["query"].replace("'", '"'), steps[i], count=1)
        return new
    # rank 1: the value, the code and the answer change; TYPE_A <-> TYPE_B follows the new value
    w["key_information"] = r["key_information"]
    for s in new["sources"]:
        if s.get("type") == "code_executor":
            s.update(code=r["code"], key_information=r["out"][:500])
            break
    qt = new["question_type"]
    if qt in ("TYPE_A", "TYPE_B"):
        qt = "TYPE_B" if any(_is_range((s.get("key_information") or "").split("=", 1)[-1]) for s in wsrc(new)) else "TYPE_A"
    new["question_type"] = qt
    if video_class == "B" and new.get("hidden_cv_chosen"):
        on = [x for x in nums(old_ki) if not (float(x).is_integer() and abs(x) <= 10)] or nums(old_ki)
        hid = list(new["hidden_cv_chosen"])
        k = next((i for i, h in enumerate(hid) if on and any(near(x, y, 0.01) for x in on for y in nums(h))), None)
        if k is not None:
            hid[k] = r["key_information"]
        new["hidden_cv_chosen"] = hid
    consts = "\n".join('%s | %s | "%s"' % (s.get("search_query"), s.get("key_information"), str(s.get("verified_quote"))[:200])
                       for s in wsrc(new))
    fin = pj(await llm(FINAL % (new["question"], qt, "\n".join(map(str, steps)), qa["answer"], consts, r["code"],
                                r["out"][:3000], qa.get("answer_range")))) or {}
    if fin.get("answer") and isinstance(fin.get("annotated_solution"), list):
        new["annotated_solution"] = [str(x) for x in fin["annotated_solution"]]
        new["answer"] = str(fin["answer"])
        new["answer_range"] = fin.get("answer_range") if qt == "TYPE_B" else None
    else:
        new["answer"] = r["alt_answer"]
        new["answer_range"] = range_of(qa.get("answer_range"), out0, r["out"]) if qt == "TYPE_B" else None
    new["total_steps"] = len(new.get("annotated_solution") or [])
    return new


async def ground_constants(qa: Dict[str, Any], video_path: str, video_class: str,
                           objects: Optional[List[Dict[str, Any]]] = None, fr=None) -> Dict[str, Any]:
    """unique-value test of every web constant; a non-unique one is resolved by requery / qualify.
    -> {'qa', 'changed', 'failing': [j], 'why': [..], 'rows': [..], 'extra': {query: results}}"""
    if fr is None:
        fr, _ = await asyncio.to_thread(frames, video_path, qa.get("question") or "")
    fk = fkey(video_path, fr)
    obj = objects_text(objects)
    cur, changed, failing, why, rows, extra = qa, False, [], [], [], {}
    for j in range(len(wsrc(qa))):
        code = code_of(cur)
        out = await arun(code)
        src = wsrc(cur)[j]
        u = await unique(cur, src, code, out, None, fr, fk, obj)
        row = {"index": j, "query": src.get("search_query"), "ok": u["ok"], "why": u["why"], "support": u["support"],
               "paraphrase": u.get("paraphrase")}
        if u["ok"]:
            rows.append(row)
            continue
        others = "; ".join(str(s.get("key_information")) for i, s in enumerate(wsrc(cur)) if i != j) or "(none)"
        c = pj(await vlm(CANDS % (cur["question"], src.get("key_information"), src.get("search_query"), u["why"], others),
                         fr, fk)) or {}
        tried = []
        for x in [x for x in c.get("candidates") or [] if isinstance(x, dict)][:MAX_CANDIDATES]:
            r, e = await const_candidate(cur, j, x, code, out, fr, fk, obj)
            if r is None:
                tried.append(e)
                continue
            cur = await apply_resolution(cur, j, r, video_class, out)
            extra[r["query"]] = await alinks(r["query"])
            changed = True
            row.update(ok=True, resolved={k: r.get(k) for k in ("action", "rank", "query", "assumption", "support")})
            break
        else:
            failing.append(j)
            why.append("web constant [%d] \"%s\": %s%s" % (j, str(src.get("key_information"))[:100], u["why"],
                                                         (" | candidates: " + " | ".join(tried[:3])) if tried else ""))
            row["candidates_failed"] = tried
        rows.append(row)
    return {"qa": cur, "changed": changed, "failing": failing, "why": why, "rows": rows, "extra": extra}


# ---------------------------------------------------------------------------
# Stated-given conversion of a constant no search states uniquely
# ---------------------------------------------------------------------------

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

_TOOLW = re.compile(r"\b(search|google|web|internet|look(?:ed)? up|code|python)\b", re.I)


def drop_hidden(hid: List[str], dropped: List[Dict[str, Any]], vals: List[float]) -> List[str]:
    """KPR hidden_cv_chosen mirrors the web values: drop the entries whose non-trivial numbers are all numbers of a
    dropped constant / a stated value"""
    pool = [x for d in dropped for x in nums(str(d.get("key_information") or "")) + nums(str(d.get("verified_quote") or ""))]
    pool += [float(v) for v in vals]
    keep = []
    for h in hid:
        hn = [x for x in nums(str(h)) if not (float(x).is_integer() and abs(x) <= 10)] or nums(str(h))
        if hn and all(any(near(x, p, 0.01) for p in pool) for x in hn):
            continue
        keep.append(h)
    return keep


def given_check(new: str, old: str, ws: List[Dict[str, Any]], todo: List[int], stated: List[Dict], hid: List[str],
                code: str) -> List[str]:
    P = []
    by = {s.get("index"): s for s in stated if isinstance(s, dict)}
    qn, on, cn = nums(new), nums(old), nums(code)
    for j in todo:
        s = by.get(j)
        if not s or not isinstance(s.get("value"), (int, float)):
            P.append("constant [%d] is not in \"stated\" with a numeric value" % j)
            continue
        v = float(s["value"])
        if not any(near(v, c) for c in cn):
            P.append("the stated value %s of constant [%d] is not a value of the code" % (s["value"], j))
        if not any(near(v, q) for q in qn):
            P.append("the question does not state the value %s of constant [%d]" % (s["value"], j))
    lost = [x for x in on if not any(near(x, y, 1e-9) for y in qn)]
    if lost:
        P.append("the question lost the number(s) %s of the old question" % sorted(set(lost)))
    keep = [j for j in range(len(ws)) if j not in todo]
    sv = [float(s["value"]) for s in by.values() if isinstance(s.get("value"), (int, float))]
    hid = drop_hidden(hid, [ws[j] for j in todo], sv)
    bad = lambda x: (not (float(x).is_integer() and abs(x) <= 10) and any(near(x, q) for q in qn)
                     and not any(near(x, o) for o in on) and not any(near(x, v) for v in sv))
    leak = [x for j in keep for x in nums(ws[j].get("key_information")) if bad(x)]
    leak += [x for h in hid for x in nums(h) if bad(x)]
    if leak:
        P.append("the question now states the value(s) %s of a web constant the solver must still search / a hidden value"
                 % sorted(set(leak)))
    if _TOOLW.search(new) and not _TOOLW.search(old):
        P.append("the question names a tool or searching")
    return P


def given_difficulty(qa: Dict[str, Any], todo: List[int], video_class: str, taken: List[str]) -> Tuple[Optional[str], str]:
    """slot rule: MPR keeps its difficulty and at least one web constant; KPR difficulty = #web constants left, allowed
    only if no other QA of the video has that difficulty"""
    left = len(wsrc(qa)) - len(todo)
    if left < 1:
        return None, "no web constant would be left"
    nd = qa.get("difficulty") if video_class == "A" else LV.get(left, "hard")
    if nd != qa.get("difficulty") and nd in taken:
        return None, "KPR %s -> %s but the video already has a %s QA" % (qa.get("difficulty"), nd, nd)
    return nd, ""


async def stated_given(qa: Dict[str, Any], todo: List[int], video_class: str, new_difficulty: str
                       ) -> Tuple[Optional[Dict[str, Any]], str]:
    """the QA with web constants `todo` stated as givens (code and answer unchanged) -> (qa, '') | (None, why)"""
    qa = copy.deepcopy(qa)
    qa["sources"] = sources(qa)
    hid = list(qa.get("hidden_cv_chosen") or [])
    ws = wsrc(qa)
    code = code_of(qa)
    tab = "\n".join("[%d] %s | %s" % (j, ws[j].get("search_query"), ws[j].get("key_information")) for j in todo)
    oth = "; ".join("[%d] %s" % (j, ws[j].get("key_information")) for j in range(len(ws)) if j not in todo) or "(none)"
    fb = ""
    for _ in range(4):
        g = pj(await llm(GIVEN % (qa["question"], "\n".join(map(str, qa.get("annotated_solution") or [])), code[:6000], tab, oth,
                                  hid or "(none)", fb))) or {}
        if not g.get("question") or not isinstance(g.get("annotated_solution"), list):
            fb = "YOUR LAST ANSWER WAS NOT VALID JSON WITH question / annotated_solution / stated.\n"
            continue
        P = given_check(g["question"], qa["question"], ws, todo, g.get("stated") or [], hid, code)
        if P:
            fb = 'YOUR LAST REWRITE "%s" WAS REJECTED: %s. Fix this.\n' % (g["question"], "; ".join(P))
            continue
        new = copy.deepcopy(qa)
        new["question"] = g["question"]
        new["annotated_solution"] = [str(x) for x in g["annotated_solution"]]
        drop = [ws[j] for j in todo]
        new["sources"] = [s for s in qa["sources"] if not any(s is d for d in drop)]
        new["required_external_tools"] = ["web_search", "code_executor"]
        new["total_steps"] = len(new["annotated_solution"])
        if video_class == "B":
            new["hidden_cv_chosen"] = drop_hidden(hid, drop, [s["value"] for s in g.get("stated") or []
                                                              if isinstance(s, dict) and isinstance(s.get("value"), (int, float))])
            if new_difficulty == "easy":
                new.pop("hidden_cv_chosen", None)
        new["difficulty"] = new_difficulty
        return new, ""
    return None, fb.strip()


# ---------------------------------------------------------------------------
# Grounded generation of one tier
# ---------------------------------------------------------------------------

async def grounded_generate(gen, tier: str, video_class: str, video_path: str, avoid: str = "",
                            max_attempts: int = MAX_ATTEMPTS) -> Dict[str, Any]:
    """gen(extra_instructions) -> Stage-3 result {qa_pairs, messages, action_count}; up to max_attempts attempts, the
    problems of a rejected attempt are fed back. -> {qa | None, attempts, action_count, given, log}; `given` is the last
    attempt that failed only on the uniqueness of some (not all) web constants (a stated-given candidate)."""
    feedback, actions, given, log = "", 0, None, []
    for a in range(max_attempts):
        res = await gen(grounding_rules(video_class, tier) + avoid + feedback)
        actions += res.get("action_count", 0) or 0
        qa = (res.get("qa_pairs") or [None])[-1]
        u = None
        if not isinstance(qa, dict):
            errs = ["no QA pair produced: %s" % res.get("error")]
        else:
            qa["difficulty"] = tier
            msgs = res.get("messages") or []
            try:
                qa = await settle_answer(qa)
                errs = await check_written(qa, msgs, tier, video_class)
                if not errs:
                    u = await ground_constants(qa, video_path, video_class)
                    qa = u["qa"]
                    if u["changed"]:
                        errs = await check_written(qa, msgs, tier, video_class, extra=u["extra"])
                    if not errs and u["failing"]:
                        if len(u["failing"]) < len(wsrc(qa)):
                            given = {"qa": copy.deepcopy(qa), "failing": u["failing"], "why": u["why"],
                                     "evidence": evidence_record(qa, msgs, u["extra"])}
                        errs = u["why"]
            except SerperFailed as e:
                errs = ['the web search failed for "%s"; try again' % e]
        log.append({"attempt": a + 1, "errors": errs, "constants": (u or {}).get("rows")})
        logger.info(f"  [Stage 3] {tier} attempt {a + 1}: {'OK' if not errs else errs[:3]}")
        if not errs:
            qa.setdefault("verification", {})["stage3"] = {"status": "passed", "attempts": a + 1,
                                                           "constants": u["rows"], "sig": sig(qa),
                                                           "evidence": evidence_record(qa, msgs, u["extra"])}
            return {"qa": qa, "attempts": a + 1, "action_count": actions, "given": None, "log": log}
        feedback = REJECTED % "\n- ".join(errs)
    return {"qa": None, "attempts": max_attempts, "action_count": actions, "given": given, "log": log}


async def convert_givens(failed: Dict[str, Dict[str, Any]], filled: List[str], video_class: str) -> List[Dict[str, Any]]:
    """stated-given conversion of the tiers that failed only on uniqueness, in tier order, under the slot rule"""
    out = []
    taken = list(filled)
    for tier in ("easy", "medium", "hard"):
        g = failed.get(tier)
        if not g:
            continue
        nd, why = given_difficulty(g["qa"], g["failing"], video_class, taken)
        if nd is None:
            logger.info(f"  [Stage 3] {tier}: stated-given conversion not allowed ({why})")
            continue
        new, why = await stated_given(g["qa"], g["failing"], video_class, nd)
        if new is None:
            logger.info(f"  [Stage 3] {tier}: stated-given conversion failed ({why[:200]})")
            continue
        P = static_checks(new) + await asyncio.to_thread(form_checks, new, video_class)
        if P:
            logger.info(f"  [Stage 3] {tier}: stated-given QA rejected ({P[:2]})")
            continue
        new.setdefault("verification", {})["stage3"] = {"status": "passed", "stated_given": g["failing"],
                                                        "from_difficulty": tier, "sig": sig(new),
                                                        "evidence": g.get("evidence") or []}
        taken.append(nd)
        out.append(new)
    return out
