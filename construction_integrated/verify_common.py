"""
verify_common.py  —  shared helpers of the grounded Stage 3 and the Stage 4b verification

  * model calls: LLM = DeepSeek-V3.2 (text), VLM = Gemini (video frames), both OpenAI-compatible, cached on disk
  * Serper web search: raw organic results, solver-side text (as the evaluation tool in triphys/tools.py returns it),
    key rotation, every call logged to <work_dir>/search_log.jsonl
  * code execution in a subprocess, answer scoring with the official ACC (triphys/metrics.py)
  * number / unit helpers, leak checks, video frame sampling

The ACC judge of triphys (TYPE_D magnitude and unreadable answers) is pointed at the same DeepSeek endpoint.
"""

import os
import re
import unicodedata
import sys
import json
import time
import base64
import asyncio
import hashlib
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("verify")

# ---------------------------------------------------------------------------
# triphys (official ACC scoring) — package next to this directory, or TRIPHYS_ROOT
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRIPHYS_PARENT = os.environ.get("TRIPHYS_ROOT") or os.path.dirname(_HERE)
if _TRIPHYS_PARENT not in sys.path:
    sys.path.append(_TRIPHYS_PARENT)
from triphys import config as tp_config, llm as tp_llm, metrics as tp_metrics  # noqa: E402
from triphys.tools import SERPER_FAILED  # noqa: E402

# ---------------------------------------------------------------------------
# Config (same variables as aclass_pipeline_core.py)
# ---------------------------------------------------------------------------

LLM_BASE = os.environ.get("DEEPSEEK_BASE", "https://openrouter.ai/api/v1")
LLM_KEY_ENV = "OPENROUTER_API_KEY" if "openrouter.ai" in LLM_BASE else "DEEPSEEK_API_KEY"
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v3.2")
VLM_BASE = os.environ.get("CHAT_API_BASE", "https://openrouter.ai/api/v1")
VLM_KEY_ENV = "VLM_API_KEY" if os.environ.get("VLM_API_KEY") else "OPENROUTER_API_KEY"
VLM_MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-3.6-flash")

# ACC judge = the construction LLM
tp_config.ACC_JUDGE_BASE = LLM_BASE
tp_config.ACC_JUDGE_MODEL = LLM_MODEL
tp_config.ACC_JUDGE_KEY_ENV = LLM_KEY_ENV

MAX_WORKERS = 2          # concurrent API calls (LLM + VLM)
PY = sys.executable

_sem: Optional[asyncio.Semaphore] = None
_work_dir = os.path.join(_HERE, "cache")
_cache: Dict[str, Any] = {}
_cache_loaded_from: Optional[str] = None


def set_work_dir(path: str) -> None:
    """per-video directory for the model cache, the Serper cache and the search log"""
    global _work_dir
    _work_dir = path
    os.makedirs(path, exist_ok=True)
    _load_cache()


def _cache_path() -> str:
    return os.path.join(_work_dir, "verify_cache.jsonl")


def _load_cache() -> None:
    global _cache_loaded_from
    p = _cache_path()
    if _cache_loaded_from == p:
        return
    _cache.clear()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    _cache[j["k"]] = j["v"]
                except Exception:
                    pass
    _cache_loaded_from = p


def _cache_put(k: str, v: Any) -> None:
    _cache[k] = v
    os.makedirs(_work_dir, exist_ok=True)
    with open(_cache_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_WORKERS)
    return _sem


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _frame_content(prompt: str, frames: List[Tuple[float, str]]) -> List[Dict[str, Any]]:
    content = [{"type": "text", "text": prompt}]
    for t, b in frames:
        content += [{"type": "text", "text": "Frame at t = %.1f s:" % t},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}}]
    return content


async def _chat(base: str, key_env: str, model: str, content: Any, max_tokens: int, timeout: float) -> str:
    key = tp_config.key(key_env, "OPENROUTER_API_KEY")
    async with _semaphore():
        for a in range(3):
            try:
                r = await tp_llm.chat(base, key, model, [{"role": "user", "content": content}],
                                      temperature=0.0, max_tokens=max_tokens, timeout=timeout, retries=3)
                t = tp_llm.content(r)
                if t:
                    return t
            except Exception as e:
                logger.warning(f"  [verify] {model} call failed: {str(e)[:160]}")
            await asyncio.sleep(5 * (a + 1))
    return ""


async def llm(prompt: str, max_tokens: int = 4000) -> str:
    """DeepSeek-V3.2, temperature 0, cached on model + prompt"""
    _load_cache()
    k = hashlib.sha1((LLM_MODEL + "|" + prompt).encode()).hexdigest()
    if k in _cache:
        return _cache[k]
    t = await _chat(LLM_BASE, LLM_KEY_ENV, LLM_MODEL, prompt, max_tokens, 300.0)
    if t:
        _cache_put(k, t)
    return t


async def vlm(prompt: str, frames: List[Tuple[float, str]], fkey: str = "", max_tokens: int = 6000) -> str:
    """Gemini with video frames [(t, b64 jpeg)], temperature 0, cached on model + frame key + prompt"""
    _load_cache()
    k = hashlib.sha1((VLM_MODEL + "|" + fkey + "|" + prompt).encode()).hexdigest()
    if k in _cache:
        return _cache[k]
    t = await _chat(VLM_BASE, VLM_KEY_ENV, VLM_MODEL, _frame_content(prompt, frames or []), max_tokens, 600.0)
    if t:
        _cache_put(k, t)
    return t


def pj(t: str) -> Any:
    """first JSON object / array in a model reply"""
    t = re.sub(r"```(?:json)?", "", t or "")
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
            try:  # a quote copied from a LaTeX snippet ({\displaystyle \mu}) has invalid escapes
                return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", m.group(0)))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Serper web search
# ---------------------------------------------------------------------------

def serper_keys() -> List[str]:
    keys = [os.environ.get("SERPER_API_KEY", "")] + os.environ.get("SERPER_API_KEYS", "").split(",")
    return [k.strip() for k in dict.fromkeys(keys) if k and k.strip()]


_key_idx = [0]


def serper(q: str) -> Optional[List[Dict[str, str]]]:
    """raw Serper organic results [{link, title, snippet}] (None if every key failed); cached and logged"""
    _load_cache()
    k = "serper|" + q
    if k in _cache:
        return _cache[k]
    keys = serper_keys()
    if not keys:
        raise RuntimeError("SERPER_API_KEY / SERPER_API_KEYS is not set")
    for a in range(len(keys) * 2):
        try:
            r = httpx.post("https://google.serper.dev/search", json={"q": q, "num": 10},
                           headers={"X-API-KEY": keys[_key_idx[0] % len(keys)]}, timeout=20)
            if r.status_code in (400, 401, 402, 403, 429):
                _key_idx[0] += 1
                continue
            org = [{"link": o.get("link", ""), "title": o.get("title", ""), "snippet": o.get("snippet", "")}
                   for o in r.json().get("organic", [])]
            _cache_put(k, org)
            with open(os.path.join(_work_dir, "search_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "check", "query": q, "organic": org},
                                   ensure_ascii=False) + "\n")
            return org
        except Exception as e:
            logger.warning(f"  [verify] Serper error ({type(e).__name__}), retrying")
            time.sleep(3)
    return None


def search_text(q: str) -> Optional[str]:
    """the text the solver sees for this query (format of the evaluation web_search tool); None if Serper failed"""
    org = serper(q)
    if org is None:
        return None
    snip = [o["snippet"] for o in org if o.get("snippet")]
    if snip:
        return 'Search results for "%s": ' % q + " | ".join(snip[:10])
    return "No results found for: %s" % q


async def asearch(q: str) -> Optional[str]:
    return await asyncio.to_thread(search_text, q)


async def alinks(q: str) -> List[Dict[str, str]]:
    return await asyncio.to_thread(serper, q) or []


def results_block(texts: List[str]) -> str:
    return "\n\n".join("[%d] %s" % (i + 1, t[:3000]) for i, t in enumerate(texts))


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

NUM = r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"


def nums(s: Any) -> List[float]:
    out = []
    for m in re.findall(NUM, str(s or "")):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def has(keys: List[float], text: Any) -> bool:
    """some number of text is within 1 % of a non-zero key"""
    t = nums(text)
    return any(any(abs(x - k) <= 0.01 * abs(k) for x in t) for k in keys if k != 0.0)


def near(x: float, y: float, tol: float = 0.005) -> bool:
    return abs(x - y) <= tol * max(abs(x), abs(y), 1e-12)


_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−–", "0123456789-+--")
_SCI = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[×xX*·]\s*10\s*(?:\^|\*\*)?|10\s*(?:\^|\*\*))\s*\(?\s*([-+]?)\s*(\d+)\)?")


def sci(t: str) -> str:
    """scientific notation as written in snippets (5.670 × 10⁻⁸, 1.789×10 −5, 5.6704 10^-8) -> 5.670e-8"""
    return _SCI.sub(lambda m: "%se%s%s" % m.groups(), t.translate(_SUP))


_GLYPHS = str.maketrans({"−": "-", "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                         "⋅": ".", "·": ".", "∙": ".", "•": "."})


def _glyph_key(s: str) -> str:
    """text compared for the verbatim check: compatibility forms (superscripts), dash and dot variants unified,
    whitespace removed, lower case"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s).translate(_GLYPHS)).lower()


def quote_ok(quote: Any, blob: str, value: Any) -> bool:
    """the quote is in the search results and states the value"""
    q = re.sub(r"\s+", " ", str(quote or "")).strip()
    if len(q) < 4 or _glyph_key(q[:60]) not in _glyph_key(blob or ""):
        return False
    v = value if isinstance(value, list) else [value]
    try:
        return all(has([float(x)], q) or has([float(x)], sci(q)) for x in v)
    except Exception:
        return False


def mask(s: Any) -> str:
    return re.sub(r"\d+(?:[.,]\d+)*", "#", str(s or ""))


def canon(code: str) -> str:
    """code with comments, numbers and whitespace removed — an edit that changes only numeric literals keeps it"""
    c = re.sub(r"#[^\n]*", "", code)
    c = re.sub(NUM, "#", c)
    return re.sub(r"\s+", "", c)


def coin(*a: Any) -> bool:
    """deterministic order randomisation against position bias (cacheable)"""
    return int(hashlib.md5(repr(a).encode()).hexdigest(), 16) % 2 == 1


# ---------------------------------------------------------------------------
# Code execution and the official ACC
# ---------------------------------------------------------------------------

_RUNNER = ("import sys,io,math\ng={'math':math,**{f:getattr(math,f) for f in dir(math) if not f.startswith('_')}}\n"
           "try:\n import numpy as np; g['np']=g['numpy']=np\nexcept ImportError: pass\n"
           "exec(sys.stdin.read(), g)\n")


def run_code(code: str, timeout: int = 60) -> str:
    """stdout of the code, or 'Error: ...'"""
    try:
        p = subprocess.run([PY, "-c", _RUNNER], input=code, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8")
        out = (p.stdout or "").strip()
        return out if p.returncode == 0 and out else "Error: " + ((p.stderr or "").strip()[-300:] or "no output")
    except subprocess.TimeoutExpired:
        return "Error: timeout"


async def arun(code: str) -> str:
    return await asyncio.to_thread(run_code, code)


async def acc(qa: Dict[str, Any], pred: str) -> Optional[float]:
    """official ACC of a predicted answer against the QA's reference"""
    try:
        return float(await tp_metrics.answer_score(pred, qa))
    except Exception as e:
        logger.warning(f"  [verify] ACC failed: {str(e)[:120]}")
        return None


def num_map(answer: str, base_out: str) -> Optional[List[int]]:
    """index (in base_out numbers) of every number of the reference answer, None if one is not printed by the code"""
    on = nums(base_out)
    idx = []
    for a in nums(answer):
        js = [j for j, x in enumerate(on) if abs(x - a) <= 0.02 * max(abs(a), 1e-9)]
        if not js:
            return None
        idx.append(js[-1])
    return idx


ALT = """QUESTION: %s
The reference answer "%s" was derived from this code output:
%s
The code was changed; new output:
%s
Write the final answer derived from the NEW output in exactly the same form (same quantity, units, wording, range form).
Return ONLY JSON: {"answer": "..."}"""


async def alt_answer(qa: Dict[str, Any], base_out: str, alt_out: str) -> str:
    """the reference answer string re-derived from a new code output"""
    idx = num_map(qa["answer"], base_out)
    an, on = nums(qa["answer"]), nums(alt_out)
    if idx and an and len(on) == len(nums(base_out)):
        parts, last = [], 0
        for m, j in zip(re.finditer(NUM, qa["answer"]), idx):
            parts += [qa["answer"][last:m.start()], "%.4g" % on[j]]
            last = m.end()
        return "".join(parts) + qa["answer"][last:]
    j = pj(await llm(ALT % (qa["question"], qa["answer"], base_out[:1500], alt_out[:1500]))) or {}
    return str(j.get("answer") or "")


def range_of(rng: Any, out0: str, out1: str) -> Optional[List[float]]:
    """the answer_range re-derived from a new code output (same printed positions)"""
    if not (isinstance(rng, list) and len(rng) == 2):
        return None
    idx = num_map(" ".join(map(str, rng)), out0)
    on = nums(out1)
    if not idx or len(on) != len(nums(out0)):
        return None
    r = sorted(float("%.4g" % on[i]) for i in idx)
    return r if r[0] < r[1] else None


# ---------------------------------------------------------------------------
# QA fields
# ---------------------------------------------------------------------------

TOOL_RE = re.compile(r"web[_ ]?search|code[_ ]?executor|google|search engine|\bsearch(?:ed|ing)?\b|look(?:ed|ing)? up|"
                     r"\bonline\b|internet", re.I)


def sources(qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    s = qa.get("sources") or []
    if isinstance(s, str):
        import ast
        try:
            s = ast.literal_eval(s)
        except Exception:
            s = []
    return [x for x in s if isinstance(x, dict)]


def wsrc(qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in sources(qa) if s.get("type") == "web_search"]


def code_of(qa: Dict[str, Any]) -> str:
    return "\n".join(s["code"] for s in sources(qa) if s.get("type") == "code_executor" and s.get("code"))


def hidden_nums(qa: Dict[str, Any]) -> List[float]:
    """the value of each hidden entry (first number after '='), minus numbers that are also given video readings"""
    vid = [x for s in sources(qa) if s.get("type") == "video_observation" for x in nums(s.get("key_information"))]
    out = []
    for h in qa.get("hidden_cv_chosen") or []:
        h = str(h)
        n = nums(h.split("=", 1)[1] if "=" in h else h)[:1]
        out += [x for x in n if not any(abs(x - y) <= 1e-9 * max(abs(x), 1e-12) for y in vid)]
    return out


def leaked(question: str, values: List[float]) -> List[float]:
    qn = nums(question)
    return sorted({v for v in values if not (float(v).is_integer() and abs(v) <= 10)
                   and any(abs(v - x) <= 1e-9 * max(abs(v), 1e-12) for x in qn)})


def video_class_of(qa: Dict[str, Any]) -> str:
    return "B" if str(qa.get("_video_class", "")).startswith("B") else "A"


# ---------------------------------------------------------------------------
# Video frames
# ---------------------------------------------------------------------------

WIN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:–|—|-|to)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", re.I)
AT_T = re.compile(r"(?:\bat\b|t\s*=)\s*(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", re.I)


def _read(path: str, ts: List[float], fps: float, width: int, quality: int, cap) -> List[Tuple[float, str]]:
    import cv2
    out = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (width, int(width * fr.shape[0] / fr.shape[1])))
        out.append((t, base64.b64encode(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]).decode()))
    return out


def frames(path: str, question: str, n_uniform: int = 10, width: int = 640) -> Tuple[List[Tuple[float, str]], float]:
    """10 uniform frames + frames inside every time window / time point the question names (max 16)"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    ts = [dur * (i + 0.5) / n_uniform for i in range(n_uniform)]
    for a, b in WIN.findall(question):
        a, b = float(a), float(b)
        if a < b <= dur + 1:
            ts += [a + (b - a) * f for f in (0.1, 0.5, 0.9)]
    for t in AT_T.findall(question):
        if float(t) <= dur + 1:
            ts.append(float(t))
    ts = sorted({round(min(t, max(dur - 0.05, 0)), 2) for t in ts})[:16]
    out = _read(path, ts, fps, width, 80, cap)
    cap.release()
    return out, dur


def dense_frames(path: str, question: str, width: int = 480, cap_n: int = 48) -> Tuple[List[Tuple[float, str]], float]:
    """up to 48 frames over the whole video, never sparser than one per dur/48 s (short videos: up to 5 fps)"""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    step = max(0.2, dur / cap_n)
    ts = [step * (i + 0.5) for i in range(int(dur / step))]
    for a, b in WIN.findall(question):
        a, b = float(a), float(b)
        if a < b <= dur + 1:
            ts += [a + (b - a) * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    for t in AT_T.findall(question):
        if float(t) <= dur + 1:
            ts.append(float(t))
    ts = sorted({round(min(t, max(dur - 0.05, 0)), 2) for t in ts})
    out = _read(path, ts, fps, width, 75, cap)
    cap.release()
    return out, dur


def fkey(path: str, fr: List[Tuple[float, str]]) -> str:
    return "%s|%s" % (os.path.basename(path), ",".join("%.2f" % t for t, _ in (fr or [])))
