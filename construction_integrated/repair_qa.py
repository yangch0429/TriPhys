"""
repair_qa.py  —  bounded repair of a QA that fails Stage-4b verification

  rounds 1 .. max_rounds-1  smallest fix: Gemini sees the frames, the whole QA and the problems of the verification and
                            returns the smallest edit (question wording, an "Assume ..." sentence without a number, a stated
                            given input, a code edit with the answer following the new output, the answer type, or a web
                            constant stated as a given). Each fix must pass deterministic checks (up to 4 tries, the
                            rejection is fed back): the code runs, the answer and answer_range are printed by it, no
                            hidden / kept web value or tool wording enters the question, "Assume" sentences carry no
                            number, the video readings, physics area and question type are kept, the answer form matches
                            the type, and a stated web constant respects the difficulty slot rule.
  round max_rounds          rewrite: the Stage-3 author agent writes a new QA of the same tier under the Stage-3 checks,
                            told which problems the failed QA had.
The caller re-verifies after every round and drops the QA if it still fails.
"""

import re
import copy
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import grounding as G
from verify_common import vlm, pj, run_code, nums, near, sources, wsrc, code_of
from verify_qa import CLS_NOTE, TYPE_NOTE, REVNOTE, hidden_of

logger = logging.getLogger("verify")

TOOLRE = re.compile(r"\b(search|google|internet|look(?:ed)? up|python|code)\b", re.I)
ASSUME_NUM = re.compile(r"\bassum\w*\b[^.]*\d", re.I)

FIX = """You repair one QA of a physics video benchmark (QAs that need the video, a web search and code). A strict check
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
DEFECTS:
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
A type_form defect about the answer form: the final answer decides the type - one value -> "question_type":
"TYPE_A" (answer_range null; remove an invented +-%% from the code and the answer); an interval propagated from range
inputs -> "TYPE_B" with the answer itself "low-high unit"; a Yes/No question -> "TYPE_C".
If a defect cannot be fixed this way, return {"unfixable": "<why>"}.
Return ONLY JSON: {"question": "...", "code": "...", "answer": "...", "answer_range": [low, high] or null, "question_type": "...",
"annotated_solution": ["..."], "state_ws": [], "stated": [{"index": j, "value": number, "unit": "..."}], "changes": "..."}"""

REWRITE = """
## A PREVIOUS QA FOR THIS TIER FAILED VERIFICATION
QUESTION: %s
Problems:
- %s
Write a different QA (another quantity or another web constant if the problems concern them) that avoids these problems.
"""


def area(q: Dict[str, Any]) -> str:
    return re.sub(r"[^a-z]", "", re.split(r"\s*/\s*", str(q.get("physics_category") or ""))[0].lower())[:5]


def keep_problems(qa: Dict[str, Any], orig: Dict[str, Any], cls: str) -> List[str]:
    """the design of the QA that a fix keeps: #video readings, stated MPR readings, physics area, question type"""
    P = []
    nv = lambda q: sum(s.get("type") == "video_observation" for s in sources(q))
    if nv(qa) != nv(orig):
        P.append("The QA must keep exactly %d video reading(s) (has %d) to keep its difficulty." % (nv(orig), nv(qa)))
    if cls == "A":
        vn = [x for o in sources(orig) if o.get("type") == "video_observation" for x in nums(o.get("key_information"))]
        same = lambda x, y: abs(x - y) <= 1e-9 * max(abs(x), 1e-12)
        lost = [x for x in nums(orig["question"]) if any(same(x, v) for v in vn) and not (float(x).is_integer() and abs(x) <= 10)
                and not any(same(x, y) for y in nums(qa["question"]))]
        if lost:
            P.append("The old question states the video reading(s) %s; keep them stated (the number of hidden readings "
                     "fixes the difficulty)." % sorted(set(lost)))
    if area(orig) and area(qa) != area(orig):
        P.append("Keep the physics area of the QA (%s); the new QA is %s." % (orig.get("physics_category"),
                                                                             qa.get("physics_category")))
    t = qa["question_type"]
    if t != orig["question_type"] and {t, orig["question_type"]} != {"TYPE_A", "TYPE_B"} and t != "TYPE_C":
        P.append("Keep the question type %s." % orig["question_type"])
    return P


def defects_text(problems: List[Dict[str, str]]) -> str:
    return "\n".join("- [%s, %s] %s" % (p.get("severity", "major"), p.get("criterion"), p.get("detail")) for p in problems)


async def small_fix(r: Dict[str, Any], cls: str, taken: List[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    qa = copy.deepcopy(r["qa"])
    qa["sources"] = sources(qa)
    qa.pop("verification", None)
    hid = hidden_of(qa)
    ws = wsrc(qa)
    code = code_of(qa)
    out0 = r.get("out") or await asyncio.to_thread(run_code, code)
    fr, fk = r["frames"]
    qt = qa["question_type"]
    fb = ""
    for _ in range(4):
        p = FIX % dict(cls=cls, clsnote=CLS_NOTE[cls], note=REVNOTE[cls], qt=qt, typenote=TYPE_NOTE.get(qt, ""),
                       diff=qa.get("difficulty"), q=qa["question"], a=qa["answer"], rng=qa.get("answer_range"),
                       sol="\n".join(map(str, qa.get("annotated_solution") or [])), hidden=hid or "(none)",
                       ws="\n".join('[%d] %s | %s | "%s"' % (j, s.get("search_query"), s.get("key_information"),
                                                             str(s.get("verified_quote") or "")[:260])
                                    for j, s in enumerate(ws)) or "(none)",
                       code=code[:6000], out=out0[-1500:], defects=defects_text(r["problems"]), fb=fb)
        g = pj(await vlm(p, fr, fk)) or {}
        if g.get("unfixable"):
            return None, "unfixable: " + str(g["unfixable"])[:300]
        if not g.get("question") or not g.get("answer") or not isinstance(g.get("annotated_solution"), list):
            fb = "YOUR LAST ANSWER WAS NOT VALID JSON WITH question / code / answer / annotated_solution.\n"
            continue
        code2 = g.get("code") or code
        out2 = out0 if code2 == code else await asyncio.to_thread(run_code, code2)
        P = []
        if out2.startswith("Error") or not out2.strip():
            P.append("the code fails: " + out2[-300:])
        st = sorted({int(j) for j in g.get("state_ws") or [] if str(j).isdigit() and int(j) < len(ws)})
        sv = [float(s["value"]) for s in g.get("stated") or [] if isinstance(s, dict) and isinstance(s.get("value"), (int, float))]
        new = copy.deepcopy(qa)
        new.update(question=str(g["question"]), answer=str(g["answer"]), annotated_solution=[str(x) for x in g["annotated_solution"]])
        nt = g.get("question_type") if g.get("question_type") in ("TYPE_A", "TYPE_B", "TYPE_C") else qt
        if nt == "TYPE_C" and not (qt == "TYPE_B" and re.match(r"\W*(yes|no)", str(qa["answer"]), re.I)):
            nt = qt  # TYPE_C only for a yes/no question labelled TYPE_B
        new["question_type"] = nt
        new["answer_range"] = g.get("answer_range") if nt == "TYPE_B" else None
        drop = [ws[j] for j in st]
        new["sources"] = [s for s in qa["sources"] if not any(s is d for d in drop)]
        for s in new["sources"]:
            if s.get("type") == "code_executor":
                s.update(code=code2, key_information=out2[:500])
                break
        if cls == "B" and "hidden_cv_chosen" in qa:
            new["hidden_cv_chosen"] = G.drop_hidden(hid, drop, sv)
        new["total_steps"] = len(new["annotated_solution"])
        if not P:
            e = G.nums_ok(new["answer"], new.get("answer_range"), out2, nt) if nt in ("TYPE_A", "TYPE_B") else None
            if e:
                P.append(e)
            P += keep_problems(new, qa, cls)
            P += await asyncio.to_thread(G.form_checks, new, cls, out2)
            qn, on = nums(new["question"]), nums(qa["question"])
            for j in st:
                vs = [x for x in nums(code) if any(near(x, v) for v in nums(str(ws[j].get("key_information"))))]
                vs += [s.get("value") for s in g.get("stated") or [] if isinstance(s, dict) and s.get("index") == j
                       and isinstance(s.get("value"), (int, float))]
                if not any(near(x, q) for x in vs for q in qn):
                    P.append("web constant [%d] is in state_ws but the question does not state its value" % j)
            khid = G.drop_hidden(hid, drop, sv)
            bad = lambda x: (not (float(x).is_integer() and abs(x) <= 10) and any(near(x, q) for q in qn)
                             and not any(near(x, o) for o in on) and not any(near(x, v) for v in sv))
            leak = [x for j in range(len(ws)) if j not in st for x in nums(str(ws[j].get("key_information"))) if bad(x)]
            leak += [x for h in khid for x in nums(h) if bad(x)]
            if leak:
                P.append("the question now states %s, the value of a kept web constant / a hidden value" % sorted(set(leak)))
            if TOOLRE.search(new["question"]) and not TOOLRE.search(qa["question"]):
                P.append("the question names a tool or searching")
            if ASSUME_NUM.search(new["question"]) and not ASSUME_NUM.search(qa["question"]):
                P.append('a new "Assume" sentence has a number: state a given input as a plain given ("take ... as ...") '
                         'and keep "Assume" sentences number-free')
        if not P and st:
            nd, why = G.given_difficulty(qa, st, cls, taken)
            if nd is None:
                P.append("state_ws is not allowed: " + why)
            else:
                new["difficulty"] = nd
                if cls == "B" and nd == "easy":
                    new.pop("hidden_cv_chosen", None)
        if P:
            fb = "YOUR LAST FIX WAS REJECTED: %s\nYOUR LAST QUESTION: %s\n" % ("; ".join(P), g["question"])
            continue
        new["verification"] = copy.deepcopy(r["qa"].get("verification") or {})
        return new, "fix: " + str(g.get("changes") or "")[:300]
    return None, "fix checks failed: " + fb[:400]


async def repair(r: Dict[str, Any], cls: str, rnd: int, max_rounds: int, taken: List[str], regen=None,
                 siblings: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """one repair round of a failed verification result r -> (new QA, how) | (None, why)"""
    if rnd < max_rounds:
        return await small_fix(r, cls, taken)
    if regen is None:
        return None, "no rewrite available"
    qa = r["qa"]
    avoid = REWRITE % (qa.get("question"), "\n- ".join(p["check"] + ": " + p["detail"][:300] for p in r["problems"]))
    try:
        new = await regen(qa.get("difficulty") or "easy", avoid, siblings or [])
    except Exception as e:
        logger.warning(f"  [Stage 4b] rewrite failed: {str(e)[:200]}")
        return None, "rewrite failed: %s" % str(e)[:200]
    if not new:
        return None, "rewrite produced no QA that passes the Stage-3 checks"
    return new, "rewrite"
