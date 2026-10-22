# Provenance (TriPhysBench release): fix_code/_v8_work/_patch_gen.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""One-off patch of benchmark_construction/agentic_physics_qa.py (repair copy) for L2:
1. hidden-value leak check compares value AND unit (the old check matched any equal / rounded number: time stamps,
   angles, "0.0015" -> "0");
2. a source whose every query contained the retrieved value is re-searched with the value stripped from the query;
   when the stripped query returns the same url, it becomes the cited query (search logged by call_serper);
3. TYPE_B answer_range taken from the answer text when the two endpoints stated there are printed by the last
   code block;
4. validate_grounding runs in a thread (it may search)."""
P = '../benchmark_construction/agentic_physics_qa.py'
s = open(P, encoding='utf-8').read()

NEW_FUNCS = r'''
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
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(0))]
        return " " if any(_consistent(str(n), values, 0.001) for n in nums) else m.group(0)
    q = _VAL_RE.sub(repl, query)
    return re.sub(r"\s+", " ", q.replace('""', "")).strip(" \"'")


def _url_key(u: str) -> str:
    return re.sub(r"[?#].*$", "", str(u)).rstrip("/").lower().replace("://www.", "://")
'''
anchor = 'def validate_grounding('
assert s.count(anchor) == 1
s = s.replace(anchor, NEW_FUNCS.lstrip('\n') + '\n\n' + anchor)

# signature: optional requery(query) -> [{"url", "snippet", "title"}]
old = '''def validate_grounding(qa: Dict[str, Any], messages: List[Dict[str, Any]], difficulty: str,
                       question_type: str, video_class: str) -> List[str]:'''
assert old in s
s = s.replace(old, '''def validate_grounding(qa: Dict[str, Any], messages: List[Dict[str, Any]], difficulty: str,
                       question_type: str, video_class: str, requery=None) -> List[str]:''')

old = '''    if chosen and _question_leaks_chosen_values(q, chosen):
        errs.append("the question text states a hidden value")'''
assert old in s
s = s.replace(old, '''    leaked = _leaked_hidden_values(q, chosen) if chosen else []
    if leaked:
        errs.append("the question text states the hidden value(s) %s: describe them without the number, or keep "
                    "the number in the question and hide another retrieved property instead" % leaked)''')

old = '''        real = [x for x in dict.fromkeys(queries[u]) if x and not (kn and _consistent(x, kn, 0.001))]
        if not real:'''
assert old in s
s = s.replace(old, '''        real = [x for x in dict.fromkeys(queries[u]) if x and not (kn and _consistent(x, kn, 0.001))]
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
        if not real:''')

old = '''    ar = qa.get("answer_range")
    if question_type == "TYPE_A"'''
assert old in s
s = s.replace(old, '''    ar = qa.get("answer_range")
    if question_type == "TYPE_B" and outs and outs[-1] and not outs[-1].startswith("Error"):
        # the range stated in the answer text, when both endpoints are printed by the last code block
        m = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(?:–|—|-|to)\\s*(\\d+(?:\\.\\d+)?)", re.sub(r"(?<=\\d),(?=\\d{3}\\b)", "", ans))
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if 0 < lo < hi and _consistent(outs[-1], [lo], 0.001) and _consistent(outs[-1], [hi], 0.001) \\
                    and ar != [lo, hi]:
                qa["answer_range"] = ar = [lo, hi]
    if question_type == "TYPE_A"''')

old = '''        errs = (["no QA pair produced: %s" % res.get("error")] if qa is None else
                validate_grounding(qa, res.get("messages") or [], difficulty, question_type, video_class))'''
assert old in s
s = s.replace(old, '''        errs = (["no QA pair produced: %s" % res.get("error")] if qa is None else
                await asyncio.to_thread(validate_grounding, qa, res.get("messages") or [], difficulty,
                                        question_type, video_class, requery))''')

old = '''    temperature: float = 0.6,
    extra_check=None,
) -> Dict[str, Any]:'''
assert old in s
s = s.replace(old, '''    temperature: float = 0.6,
    extra_check=None,
    requery=None,
) -> Dict[str, Any]:''')

open(P, 'w', encoding='utf-8', newline='\n').write(s)
compile(s, P, 'exec')
print('patched')
