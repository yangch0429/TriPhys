"""
verify_qa.py  —  Stage 4b: verification of every QA that Stage 4 checked

Each QA of a video is verified in tier order (easy, medium, hard); later QAs are compared with the QAs of the video
that are already verified. The checks (a QA is verified only if all pass):

  C     code: the solution code runs; for TYPE_A / TYPE_B the answer is a printed number (or scores ACC >= .9 against
        an answer re-derived from the output).
  S     the Stage-3 checks (grounding of every web value in its search results with a verbatim quote, leaks, answer form,
        range form, inputs the solver cannot know).
  V     video consistency: Gemini lists the scene statements of the question on 10-16 frames (visible / plausible /
        contradicted / invented / not_shown); questioned statements are re-judged on up to 48 dense frames
        (absent / contradicted / invented); an invented statement fails if it gives a scene object an identity that
        changes which value applies (not if it only says a typical / standard value applies). Fails on contradicted,
        absent or identity.
  W     well-posedness of every web constant: the unique-value search test of grounding.unique() with the objects V
        saw (the official ACC of triphys/metrics.py decides whether a value found on the web changes the answer).
  M     well-posedness of the modelling: up to 3 modelling / interpretation choices the question leaves open are
        implemented as code edits and scored with the official ACC; a choice with ACC < .5 that a careful solver would
        make instead of the reference (judged from the question alone) fails.
  CONS  cross-question consistency: a property of the same object retrieved by two QAs of the video has the same value
        (the earlier verified QA keeps its value).
  Q     the Stage-4 quality fields (re-checked by the Stage-4 checker if the QA changed) and the Stage-5 rule filter.
  AUDIT strict whole-QA audit by Gemini with the frames (leak, scene, assumption, source, physics, answer, ambiguous,
        unnatural) plus the deterministic answer-form check; any major defect fails.

A failing QA is repaired (repair_qa.py): two rounds of the smallest fix, then one rewrite; it is re-verified after
every round and dropped if it still fails.
"""

import copy
import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import grounding as G
from verify_common import (
    llm, vlm, pj, run_code, arun, acc, num_map, alt_answer, coin, sources, wsrc, code_of, frames, dense_frames, fkey,
)

logger = logging.getLogger("verify")

TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLS_NOTE = {"A": "the solver watches the video and reads motion quantities from it (the CV readings below); it looks up web "
                 "constants with Google and runs Python.",
            "B": "the solver watches the video (the scene, given numbers in the question) and looks up web constants with Google "
                 "and runs Python."}
TYPE_NOTE = {"TYPE_A": 'answer = ONE number with unit (no interval, no "accepted range +-x%").',
             "TYPE_B": 'answer = an interval "low-high unit" (answer_range [low, high]). The interval comes from range input(s) - '
                       "a web constant stated as a range, or a range the question states - propagated by the code; never from "
                       "an invented +-% or from disagreeing sources.",
             "TYPE_C": 'answer = yes / no plus the deciding number(s) (e.g. "Yes - clearance 0.18 m"); a feasibility question.',
             "TYPE_D": 'answer = a comparison plus a ratio / difference (e.g. "A > B, ratio 1.3"); a comparison question.'}
REVNOTE = {
    "A": """BENCHMARK CONVENTIONS (not defects): the solver measures body-joint readings from the video with a pose tool, so a
question may name a reading ("the athlete's peak wrist speed") without its number - it is answerable. A stated assumption
that links a reading to the object ("the ball leaves the hand at the peak wrist speed", "the ball moves with the athlete")
is a modelling assumption, not a surrogate. Benchmark convention: the ball's release / launch speed IS the athlete's peak
wrist speed, even if the question does not say so; this is not undefined, not ambiguous and not a missing input, and the
elbow (or another joint) speed is not a competing alternative.
""",
    "B": """BENCHMARK CONVENTION (not a defect): the solver may read sizes, angles and times from the video; a question that
states such an estimate ("the marble is about 1.6 cm across") is fine.
""",
}

VPROMPT = """You check a physics benchmark question against frames of the video it is about. The solver sees the video and the
question; the question must describe the scene correctly. Frames are sampled from the whole video (timestamps given).
QUESTION: %s

List every statement the question makes about the scene: objects, their identity (brand, model, product, size, regulation,
standard), material, colour, count, setting, positions, actions / events and their timing. NOT statements: numbers
(heights, speeds, angles, masses, distances, durations - given or measured), anything introduced by "assume" / "treat as" /
"consider", hypothetical events, physical interpretations (energy, friction, forces) and what is asked. Label each statement:
- "visible": the frames show it
- "plausible": not directly visible but consistent with the frames and the setting, a careful viewer would accept it
  (e.g. an adult men's game uses a size-7 ball; a dump truck bed is steel)
- "contradicted": the frames show something different
- "invented": a specific identity / material / grade / standard that nothing in the frames supports while other choices
  fit equally well (e.g. a brand or model with no visible marking)
- "not_shown": an object or event the question says happens in the video that appears in none of the frames
Also list each object whose physical property the question needs, with what the frames show about it.
Return ONLY JSON: {"claims": [{"claim": "...", "label": "visible|plausible|contradicted|invented|not_shown", "evidence": "..."}],
"objects": [{"object": "...", "seen": "<material, size class, markings, setting>"}]}"""

RPROMPT = """A physics benchmark question is about the video whose frames follow (about one per second, timestamps given).
An earlier check on sparse frames questioned the statements below. Look through ALL frames and judge each statement.
QUESTION: %s
STATEMENTS:
%s
Labels:
- "not_scene": the statement is a measurement, number, speed / force / energy / centre-of-mass value, assumption or physical
  interpretation, not a statement about what the scene shows (e.g. "the wrist reaches a maximum speed") - drop it
- "visible": some frame shows it (give the timestamp)
- "plausible": not directly visible but consistent with the frames and the setting
- "contradicted": the frames show something different (e.g. a different object / colour / action)
- "invented": a specific identity (brand, league, model, standard) that nothing in the frames supports
- "absent": a scene object or event the question says happens, and no frame shows it
Return ONLY JSON: {"statements": [{"claim": "...", "label": "...", "evidence": "<timestamp + what is seen>"}]}"""

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

MODEL_P = """You audit a physics benchmark QA for ambiguity. The solver sees only the QUESTION (and, for class A, a video
from which it reads times / distances / speeds).
QUESTION: %s
QUESTION TYPE: %s
REFERENCE SOLUTION:
%s
REFERENCE CODE:
```python
%s
```
REFERENCE ANSWER: %s

List up to 3 modelling or interpretation choices made by the reference solution that change the numeric answer and that the
question does NOT state or clearly imply, where a competent physicist reading only the question could reasonably choose
differently (e.g. which formula / model, which quantity is asked for, start from rest or not, rolling vs sliding, which point
of the motion, inclusion of an effect, definition of a ratio). Do NOT list: values read from the video, values looked up on
the web, default textbook conventions every solver would adopt (g = 9.81, point mass, no air resistance unless the
question raises it), or answer-format choices (range vs single value, rounding, sign convention, units). For each choice give the alternative implemented as a minimal edit of the reference code (same
inputs, same printed format, only the modelling changed).
Return ONLY JSON: {"choices": [{"aspect": "<neutral 3-8 word name, no argument>", "reference": "<the reference's option, one
sentence>", "alternative": "<the other option, one sentence>", "code": "<full code>"}]}
(empty list if the question determines the model)"""

MODEL_V = """A physics benchmark question; the solver sees only the QUESTION text (and, for class A, a video for
measurements).
QUESTION: %s

Aspect of the solution: %s
Option 1: %s
Option 2: %s
The option texts were written by an auditor; they are NOT part of the question. Judging only from the QUESTION text, which
option should a careful solver choose? "1" or "2" if the question states or clearly implies that option (or it is the only
standard reading), "both" if the question leaves it open.
Return ONLY JSON: {"verdict": "1|2|both", "why": "..."}"""

CONS = """Below are the externally retrieved values used by different QAs about the SAME video. A benchmark rule says:
when two QAs retrieve the same property of the same object (e.g. the mass of the marble in this video, the friction
coefficient of glass on wood), they must use the same value.

Find every violation: two QAs that use DIFFERENT values for the same property of the same object / material pair.
- Same object means the same physical thing of this video (or the same material pair / same constant), not merely a
  similar word; e.g. "marble mass" and "marble diameter" are different properties; static and kinetic friction are
  different properties; "glass on wood" and "wood on wood" are different material pairs.
- Different values: numbers that differ after unit conversion (5 g vs 5.5 g; 0.3-0.5 vs 0.4), including a range vs a
  single value. Identical values written in other units are NOT violations.
- Ignore physical constants that are the same (g = 9.81 vs 9.80665 is a violation only if both are retrieved values).

{facts}

Return ONLY JSON: {{"violations": [{{"object_property": "...", "qa_values": {{"<qa label>": "<value>", ...}},
"note": "<one sentence>"}}]}}  (empty list if none)"""

AUDIT = """You are a strict auditor of one QA of a physics video benchmark (QAs that need the video, a web search and code).
Audit the WHOLE QA. The video frames are attached; trust them.
CLASS %(cls)s - %(clsnote)s
QUESTION TYPE %(qt)s - %(typenote)s   DIFFICULTY %(diff)s (%(counts)s)
QUESTION: %(q)s
ANSWER: %(a)s   (answer_range %(rng)s)
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
Do NOT report: style preferences, a missing explanation, a "typical" / "standard" value whose quote states it, a size or
angle the solver reads from the video (class B), a pose reading named without its number (class A).
Return ONLY JSON: {"defects": [{"criterion": "...", "severity": "major" | "minor", "detail": "..."}]}  (empty list if none)"""

REBUILD = """Write Python code that implements exactly this reference solution of a physics question and prints the final
result (all intermediate inputs as literals taken from the solution).
QUESTION: %s
REFERENCE SOLUTION:
%s
REFERENCE ANSWER: %s
Return ONLY JSON: {"code": "<full code>"}"""


def ws_txt(qa: Dict[str, Any]) -> str:
    return "\n".join('[%d] %s | %s | "%s" | %s' % (j, s.get("search_query"), s.get("key_information"),
                                                   str(s.get("verified_quote") or "")[:260], s.get("url") or "")
                     for j, s in enumerate(wsrc(qa))) or "(none)"


def hidden_of(qa: Dict[str, Any]) -> List[str]:
    h = qa.get("hidden_cv_chosen") or []
    return [str(x) for x in h] if isinstance(h, list) else [str(h)]


def problem(check: str, detail: str, criterion: str = "") -> Dict[str, str]:
    return {"check": check, "criterion": criterion or check, "severity": "major", "detail": detail}


# ---------------------------------------------------------------------------
# Single checks
# ---------------------------------------------------------------------------

async def check_code(qa: Dict[str, Any]) -> Dict[str, Any]:
    code = code_of(qa)
    if not code:
        j = pj(await llm(REBUILD % (qa["question"], "\n".join(map(str, qa.get("annotated_solution") or [])), qa["answer"]))) or {}
        code = j.get("code") or ""
    out = await arun(code) if code else "Error: no code"
    P = []
    if out.startswith("Error"):
        P.append(problem("C", "the solution code fails: " + out[-300:], "answer"))
    elif qa.get("question_type") in ("TYPE_A", "TYPE_B") and num_map(qa["answer"], out) is None:
        b = await acc(qa, await alt_answer(qa, out, out))
        if b is None or b < 0.9:
            P.append(problem("C", 'the answer "%s" is not what the code prints (%s)' % (qa["answer"], out[-300:]), "answer"))
    return {"code": code, "out": out, "problems": P}


async def check_static(qa: Dict[str, Any], cls: str, out: str) -> List[Dict[str, str]]:
    q = copy.deepcopy(qa)
    ev = (qa.get("verification") or {}).get("stage3", {}).get("evidence") or []
    extra = {}
    for e in ev:
        extra.setdefault(e.get("query") or "", []).append({"link": e.get("url"), "title": "", "snippet": e.get("text") or ""})
    for k, v in (await asyncio.to_thread(G.evidence_of, q)).items():
        extra.setdefault(k, []).extend(v)
    errs = await asyncio.to_thread(G.validate_grounding, q, [], q.get("difficulty") or "easy", cls, extra, None)
    errs += G.static_checks(q) + await asyncio.to_thread(G.form_checks, q, cls, out)
    return [problem("S", e, "leak" if "leak" in e or "states the" in e or "tool" in e else "source")
            for e in dict.fromkeys(errs)]


async def check_video(qa: Dict[str, Any], video_path: str, fr, fk: str) -> Dict[str, Any]:
    j = pj(await vlm(VPROMPT % qa["question"], fr, fk)) or {}
    if not isinstance(j.get("claims"), list):
        return {"objects": [], "problems": [problem("V", "the video check returned no result", "scene")]}
    cl = [c for c in j["claims"] if isinstance(c, dict)]
    objects = [o for o in j.get("objects") or [] if isinstance(o, dict)]
    q = [c for c in cl if c.get("label") in ("contradicted", "invented", "not_shown")]
    res = {"objects": objects, "claims": cl, "problems": []}
    if not q:
        return res
    dfr, _ = await asyncio.to_thread(dense_frames, video_path, qa["question"])
    r = pj(await vlm(RPROMPT % (qa["question"], "\n".join("- " + str(c.get("claim", "")) for c in q)), dfr,
                     fkey(video_path, dfr))) or {}
    if not isinstance(r.get("statements"), list):
        res["problems"].append(problem("V", "the dense video re-check returned no result", "scene"))
        return res
    st = [s for s in r["statements"] if isinstance(s, dict)]
    res["recheck"] = st
    bad = [s for s in st if s.get("label") in ("absent", "contradicted")]
    inv = [s for s in st if s.get("label") == "invented"]
    if inv:
        k = pj(await llm(INVKIND % "\n".join("- " + str(c.get("claim")) for c in inv))) or {}
        kinds = {str(x.get("claim")): x.get("kind") for x in k.get("items") or [] if isinstance(x, dict)}
        bad += [s for s in inv if kinds.get(str(s.get("claim")), "identity") == "identity"]
    res["problems"] = [problem("V", '%s: "%s" (%s)' % (s.get("label"), s.get("claim"), str(s.get("evidence"))[:200]), "scene")
                       for s in bad]
    return res


async def check_web(qa: Dict[str, Any], code: str, out: str, fr, fk: str, objects: List[Dict]) -> Dict[str, Any]:
    rows, P = [], []
    for j, s in enumerate(wsrc(qa)):
        try:
            u = await G.unique(qa, s, code, out, None, fr, fk, G.objects_text(objects))
        except G.SerperFailed as e:
            P.append(problem("W", 'the web search failed for "%s"' % e, "source"))
            continue
        rows.append({"index": j, "query": s.get("search_query"), "ok": u["ok"], "why": u["why"], "support": u["support"]})
        if not u["ok"]:
            P.append(problem("W", 'web constant [%d] "%s": %s' % (j, str(s.get("key_information"))[:120], u["why"]),
                             "ambiguous"))
    return {"rows": rows, "problems": P}


async def check_model(qa: Dict[str, Any], code: str, out: str) -> Dict[str, Any]:
    sol = "\n".join(map(str, qa.get("annotated_solution") or []))
    j = pj(await llm(MODEL_P % (qa["question"], qa["question_type"], sol, code, qa["answer"]))) or {}
    rows, P = [], []
    for c in (j.get("choices") or [])[:3]:
        if not isinstance(c, dict) or not c.get("code"):
            continue
        row = {k: c.get(k) for k in ("aspect", "reference", "alternative")}
        o = await arun(c["code"])
        if o.startswith("Error"):
            row["run_error"] = True
            rows.append(row)
            continue
        a = await alt_answer(qa, out, o)
        row["alt_answer"] = a
        row["acc"] = await acc(qa, a) if a else None
        if row["acc"] is not None and row["acc"] < 0.5:
            flip = coin(qa["question"], c.get("aspect"))
            opts = (c.get("alternative"), c.get("reference")) if flip else (c.get("reference"), c.get("alternative"))
            v = pj(await llm(MODEL_V % ((qa["question"], c.get("aspect")) + opts))) or {}
            row["verdict"] = {"1": "X" if flip else "R", "2": "R" if flip else "X", "both": "both"}.get(str(v.get("verdict")))
            row["why"] = v.get("why")
            if row["verdict"] == "X":
                P.append(problem("M", 'modelling choice "%s": the question points to "%s", the reference uses "%s" (%s)'
                                 % (c.get("aspect"), c.get("alternative"), c.get("reference"), str(row["why"])[:200]),
                                 "physics"))
        rows.append(row)
    return {"rows": rows, "problems": P}


def facts_of(qa: Dict[str, Any], cls: str) -> List[str]:
    lines = ["  - retrieved: %s (query: %s)" % (str(s.get("key_information"))[:300], s.get("search_query") or "-")
             for s in wsrc(qa)]
    if cls == "B":
        lines += ["  - hidden value: %s" % h for h in hidden_of(qa)]
    return lines


async def check_consistency(qa: Dict[str, Any], others: List[Dict[str, Any]], cls: str) -> Dict[str, Any]:
    blocks = []
    for lab, x in [("NEW", qa)] + [("V%d" % (i + 1), o) for i, o in enumerate(others)]:
        f = facts_of(x, cls)
        if f:
            blocks.append("QA %s:\n  question: %s\n%s" % (lab, str(x.get("question"))[:300], "\n".join(f)))
    if len(blocks) < 2 or not facts_of(qa, cls):
        return {"violations": [], "problems": []}
    d = pj(await llm(CONS.format(facts="\n\n".join(blocks)))) or {}
    vio = [v for v in d.get("violations") or [] if isinstance(v, dict) and "NEW" in (v.get("qa_values") or {})]
    P = [problem("CONS", 'another QA of this video uses a different value for "%s": %s (%s)'
                 % (v.get("object_property"), v.get("qa_values"), v.get("note")), "source") for v in vio]
    return {"violations": vio, "problems": P}


async def check_quality(qa: Dict[str, Any], quality_fn: Optional[Callable[[Dict], Awaitable[Dict]]]) -> Dict[str, Any]:
    from filter_valid_qa import is_valid_item
    if quality_fn is not None and qa.get("_stage4_sig") != G.sig(qa):
        try:
            r = await quality_fn(copy.deepcopy(qa))
            r = r.get("multi_hop_qa_pair", r) if isinstance(r, dict) else {}
            for k, v in r.items():
                if k not in qa or k.startswith(("is_", "answerable_", "overall_", "multi_hop_", "quality_")):
                    qa[k] = v
        except Exception as e:
            return {"qa": qa, "problems": [problem("Q", "the Stage-4 quality check failed: %s" % str(e)[:200], "answer")]}
        qa["_stage4_sig"] = G.sig(qa)
    if is_valid_item({"multi_hop_qa_pair": qa, "primary_modality": "video"}):
        return {"qa": qa, "problems": []}
    keys = ("is_question_understandable", "is_answer_correct_and_unique", "answerable_without_visual_and_audio_content",
            "answerable_without_tools", "overall_quality", "multi_hop_reasoning_quality")
    return {"qa": qa, "problems": [problem("Q", "the quality check rejects the QA (%s)%s"
                                           % (", ".join("%s=%s" % (k, qa.get(k)) for k in keys if k in qa),
                                              "; " + str(qa.get("quality_check_error")) if qa.get("quality_check_error") else ""),
                                           "answer")]}


async def check_audit(qa: Dict[str, Any], cls: str, out: str, fr, fk: str) -> Dict[str, Any]:
    hid = hidden_of(qa)
    d = dict(cls=cls, clsnote=CLS_NOTE[cls], qt=qa["question_type"], typenote=TYPE_NOTE.get(qa["question_type"], ""),
             diff=qa.get("difficulty"), counts="#web constants %d, #hidden %d" % (len(wsrc(qa)), len(hid)),
             q=qa["question"], a=qa["answer"], rng=qa.get("answer_range"), hidden=hid or "(none)", ws=ws_txt(qa),
             code=code_of(qa)[:6000], out=out[-1500:], note=REVNOTE[cls])
    p = AUDIT % d
    rv = pj(await vlm(p, fr, fk))
    if not isinstance(rv, dict) or not isinstance(rv.get("defects"), list):
        await asyncio.sleep(30)
        rv = pj(await vlm(p + "\n", fr, fk))
    if not isinstance(rv, dict) or not isinstance(rv.get("defects"), list):
        return {"defects": [], "problems": [problem("AUDIT", "the strict audit returned no review", "answer")]}
    D = [x for x in rv["defects"] if isinstance(x, dict)]
    D += [{"criterion": "type_form", "severity": "major", "detail": x} for x in G.form_defects(qa, out)]
    P = [problem("AUDIT", str(x.get("detail")), str(x.get("criterion"))) for x in D if x.get("severity") == "major"]
    return {"defects": D, "problems": P}


# ---------------------------------------------------------------------------
# One QA
# ---------------------------------------------------------------------------

async def verify_one(qa: Dict[str, Any], video_path: str, cls: str, verified: List[Dict[str, Any]],
                     quality_fn=None) -> Dict[str, Any]:
    """-> {'ok', 'problems': [..], 'checks': {name: summary}, 'qa', 'code', 'out', 'frames'}"""
    qa["sources"] = sources(qa)
    checks: Dict[str, Any] = {}
    c = await check_code(qa)
    P = list(c["problems"])
    if not P:
        P += await check_static(qa, cls, c["out"])
    checks["C"] = "fail" if c["problems"] else "pass"
    checks["S"] = "fail" if any(p["check"] == "S" for p in P) else ("pass" if not c["problems"] else "skipped")
    fr, _ = await asyncio.to_thread(frames, video_path, qa["question"])
    fk = fkey(video_path, fr)
    res = {"ok": False, "problems": P, "checks": checks, "qa": qa, "code": c["code"], "out": c["out"], "frames": (fr, fk)}
    if P:
        return res
    v = await check_video(qa, video_path, fr, fk)
    w = await check_web(qa, c["code"], c["out"], fr, fk, v.get("objects") or [])
    m = await check_model(qa, c["code"], c["out"])
    k = await check_consistency(qa, verified, cls)
    q = await check_quality(qa, quality_fn)
    for name, r in (("V", v), ("W", w), ("M", m), ("CONS", k), ("Q", q)):
        checks[name] = "fail" if r["problems"] else "pass"
        P += r["problems"]
    checks["W_constants"] = w["rows"]
    if P:
        return res
    a = await check_audit(qa, cls, c["out"], fr, fk)
    checks["AUDIT"] = "fail" if a["problems"] else "pass"
    checks["AUDIT_minor"] = [x.get("detail") for x in a["defects"] if x.get("severity") != "major"][:5]
    P += a["problems"]
    res["ok"] = not P
    return res


# ---------------------------------------------------------------------------
# All QAs of one video
# ---------------------------------------------------------------------------

async def verify_video(qa_pairs: List[Dict[str, Any]], video_path: str, cls: str, quality_fn=None, regen=None,
                       max_rounds: int = 3) -> List[Dict[str, Any]]:
    """verify (and repair) every QA in tier order; every returned QA carries
    verification = {status: verified | dropped, checks, rounds, history}"""
    from repair_qa import repair
    order = sorted(range(len(qa_pairs)), key=lambda i: TIER_ORDER.get(qa_pairs[i].get("difficulty"), 3))
    verified: List[Dict[str, Any]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(qa_pairs)
    for i in order:
        qa = copy.deepcopy(qa_pairs[i])
        if not qa.get("quality_check_error") and "is_question_understandable" in qa:
            qa["_stage4_sig"] = G.sig(qa)   # the Stage-4 fields belong to this exact QA
        tier = qa.get("difficulty")
        history = []
        taken = lambda: [x.get("difficulty") for j, x in enumerate(qa_pairs) if j != i and
                         (out[j] is None or out[j]["verification"]["status"] == "verified")]
        r = await verify_one(qa, video_path, cls, verified, quality_fn)
        rnd = 0
        while not r["ok"] and rnd < max_rounds:
            rnd += 1
            history.append({"round": rnd, "problems": [p["check"] + ": " + p["detail"][:300] for p in r["problems"]]})
            logger.info(f"  [Stage 4b] {tier} round {rnd}: {len(r['problems'])} problem(s): "
                        f"{[p['check'] for p in r['problems']]}")
            new, how = await repair(r, cls, rnd, max_rounds, taken(), regen, siblings=verified)
            history[-1]["repair"] = how
            if new is None:
                if rnd < max_rounds - 1:   # the small fix failed: go straight to the rewrite
                    rnd = max_rounds - 1
                continue
            r = await verify_one(new, video_path, cls, verified, quality_fn)
        qa = r["qa"]
        qa.pop("_stage4_sig", None)
        status = "verified" if r["ok"] else "dropped"
        qa.setdefault("verification", {})
        qa["verification"].update(status=status, checks=r["checks"], rounds=rnd, history=history)
        if not r["ok"]:
            qa["verification"]["problems"] = [p["check"] + ": " + p["detail"][:300] for p in r["problems"]]
        logger.info(f"  [Stage 4b] {tier}: {status} after {rnd} repair round(s)")
        if r["ok"]:
            verified.append(qa)
        out[i] = qa
    return [x for x in out if x is not None]
