"""Scoring (appendix "Answer Scoring", "Solution Judge and Rubric", "Score Aggregation").

ACC  point (TYPE_A)       1 if delta <= .02, exp(-8(delta-.02)) up to .30, else 0; zero reference -> 0
     interval (TYPE_B)    .5 IoU + .5 EQS; predicted interval inside G -> 1; point inside -> 1;
                          point outside -> .5[exp(-5d/w) + exp(-5(d+w)/w)]
     judgment (TYPE_C)    yes/true/correct vs no/false/incorrect
     comparison (TYPE_D)  .5 * 1[direction] + .5 * A_A(magnitude)
     Prediction and reference are converted to SI (Table units); quantities of different physical dimensions score 0.
SCS  .25 S_obs + .25 S_formula + .30 S_derive + .20 S_conclude, DeepSeek-V4-Flash judge (temperature .3, 60 s, retry),
     given only the reference solution and the model solution.
OGS  .7 ACC + .3 SCS.
Model-based evaluation (responses the parser cannot read, and the magnitude of TYPE_D answers) uses the ACC judge
(gpt-4o-mini by default); the extraction step does not see the reference."""
import logging
import math
import re
from typing import Optional

from . import config, llm
from .parser import parse_response

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ units
UNIT_CONVERSIONS = {
    'mm': 0.001, 'cm': 0.01, 'm': 1, 'km': 1000,
    'mg': 0.000001, 'g': 0.001, 'kg': 1,
    'mJ': 0.001, 'J': 1, 'kJ': 1000,
    'N': 1, 'kN': 1000,
    'ms': 0.001, 's': 1, 'min': 60, 'h': 3600,
    'cm/s': 0.01, 'm/s': 1, 'km/h': 0.277778,
}
DIMENSION = {'mm': 'L', 'cm': 'L', 'm': 'L', 'km': 'L', 'mg': 'M', 'g': 'M', 'kg': 'M', 'mJ': 'E', 'J': 'E',
             'kJ': 'E', 'N': 'F', 'kN': 'F', 'ms': 'T', 's': 'T', 'min': 'T', 'h': 'T',
             'cm/s': 'V', 'm/s': 'V', 'km/h': 'V'}
_ALIAS = {}
for _u, _names in {
        'mm': 'millimeter millimeters millimetre millimetres', 'cm': 'centimeter centimeters centimetre centimetres',
        'm': 'meter meters metre metres', 'km': 'kilometer kilometers kilometre kilometres',
        'mg': 'milligram milligrams', 'g': 'gram grams', 'kg': 'kilogram kilograms kgs',
        'mJ': 'millijoule millijoules', 'J': 'joule joules', 'kJ': 'kilojoule kilojoules',
        'N': 'newton newtons', 'kN': 'kilonewton kilonewtons', 'ms': 'millisecond milliseconds msec',
        's': 'second seconds sec secs', 'min': 'minute minutes mins', 'h': 'hour hours hr hrs',
        'm/s': 'm/sec meters/second', 'km/h': 'km/hr kmh kph kmph', 'cm/s': 'cm/sec'}.items():
    for _n in _names.split():
        _ALIAS[_n.lower()] = _u


def canonical_unit(u):
    if not u:
        return None
    u = u.strip().rstrip('.,;:)')
    if u in UNIT_CONVERSIONS:
        return u
    return _ALIAS.get(u.lower())


def same_dimension(u1, u2):
    """False only when both units are recognized and belong to different physical dimensions"""
    a, b = canonical_unit(u1), canonical_unit(u2)
    return not (a and b and DIMENSION[a] != DIMENSION[b])


_SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')


def _clean(text):
    t = str(text).replace('–', '-').replace('—', '-').replace('−', '-')
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)                                     # thousands separators
    t = re.sub(r'([0-9]+\.?[0-9]*)\s*[x×*·]\s*10(?:\s*\^\s*(?:\(\s*([-+]?[0-9]+)\s*\)|([-+]?[0-9]+))|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))',
               lambda m: m.group(1) + 'e' + (m.group(2) or m.group(3) or m.group(4).translate(_SUP)), t)  # 8.9 x 10^-6
    return t


_VNUM = r'(?<![\w.])(-?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)'
_VUNIT = r'\s*([a-zA-Z°%µ][a-zA-Z/°%²³µ]*)?'


def normalize_value_with_unit(text):
    """(SI value, unit) of the first number with a recognized unit, else of the first number; (None, None) if none"""
    if not text:
        return None, None
    hits = [(float(m.group(1)), m.group(2)) for m in re.finditer(_VNUM + _VUNIT, _clean(text))]
    if not hits:
        return None, None
    for v, u in hits:
        cu = canonical_unit(u)
        if cu:
            return v * UNIT_CONVERSIONS[cu], cu
    return hits[0]


_RNUM = r'(?<![\w.])([0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)'
_RSNUM = r'(?<![\w.])(-?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)'
_RUNIT = r'(?!(?:to|and)\b)([a-zA-Z°%][a-zA-Z/°%²³]*)'
_RKNOWN = r'(m/s|km/h|cm/s|kJ|mJ|kN|cm|mm|km|mg|kg|ms|min|J|N|m|g|s|h)'
_RSEP = r'\s*(?:[-~]|\bto\b)\s*'
_RANGE_PATTERNS = [
    (0, _RNUM + _RSEP + _RNUM + r'\s*' + _RKNOWN, 'unit'),                          # X - Y J
    (0, _RSNUM + r'\s+to\s+' + _RSNUM + r'\s*' + _RKNOWN, 'unit'),                  # -0.3 to 0.6 m/s
    (0, _RNUM + r'\s*' + _RUNIT + _RSEP + _RNUM + r'\s*' + _RUNIT, 'both'),          # 440.64 J to 609.12 J
    (0, _RSNUM + r'\s*' + _RUNIT + r'\s+to\s+' + _RSNUM + r'\s*' + _RUNIT, 'both'),  # -2 m/s to 3 m/s
    (1, r'\bbetween\s+' + _RSNUM + r'\s*' + _RKNOWN + r'?\s+and\s+' + _RSNUM + r'\s*' + _RKNOWN + r'?', 'between'),
    (2, _RNUM + _RSEP + _RNUM, 'plain'),                                             # X - Y
    (2, _RSNUM + r'\s+to\s+' + _RSNUM, 'plain'),
]


def _range_from_match(m, kind):
    if kind == 'unit':
        u = m.group(3)
        c = UNIT_CONVERSIONS.get(u, 1)
        return float(m.group(1)) * c, float(m.group(2)) * c, u
    if kind == 'plain':
        return float(m.group(1)), float(m.group(2)), None
    v1, u1, v2, u2 = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    if kind == 'between':
        u1, u2 = u1 or u2, u2 or u1
        return v1 * UNIT_CONVERSIONS.get(u1, 1), v2 * UNIT_CONVERSIONS.get(u2, 1), u2
    c1, c2 = canonical_unit(u1), canonical_unit(u2)
    if c1 and c2:
        if DIMENSION[c1] != DIMENSION[c2]:
            return None
        return v1 * UNIT_CONVERSIONS[c1], v2 * UNIT_CONVERSIONS[c2], c2
    if u1 == u2:
        return v1, v2, u2
    return None


def normalize_range_with_unit(text):
    """(lo, hi, unit) in SI of the earliest interval in the text; at the same position a form with units wins"""
    if not text:
        return None, None, None
    t = _clean(text)
    best = None
    for pri, pat, kind in _RANGE_PATTERNS:
        for m in re.finditer(pat, t, re.I if kind == 'between' else 0):
            if best is not None and (m.start(), pri) >= best[0]:
                break
            r = _range_from_match(m, kind)
            if r is not None:
                best = ((m.start(), pri), r)
                break
    return best[1] if best else (None, None, None)


# ------------------------------------------------------------------ answer scores
def point_score(delta):
    if delta <= 0.02:
        return 1.0
    if delta <= 0.30:
        return math.exp(-8.0 * (delta - 0.02))
    return 0.0


def acc_type_a(pred, gt):
    p, pu = normalize_value_with_unit(pred)
    g, gu = normalize_value_with_unit(gt)
    if p is None or g is None or g == 0 or not same_dimension(pu, gu):
        return 0.0
    return point_score(abs(p - g) / abs(g))


def acc_type_b(pred, gt, gt_range=None):
    lo, hi, gu = normalize_range_with_unit(gt)
    if lo is None:
        if not (gt_range and len(gt_range) == 2):
            return 0.0
        lo, hi = float(gt_range[0]), float(gt_range[1])
        gu = normalize_value_with_unit(gt)[1]
    lo, hi = min(lo, hi), max(lo, hi)
    w = hi - lo
    plo, phi, pu = normalize_range_with_unit(pred)
    if plo is not None:
        if not same_dimension(pu, gu):
            return 0.0
        plo, phi = min(plo, phi), max(plo, phi)
        if lo <= plo and phi <= hi:
            return 1.0
        inter = max(0.0, min(phi, hi) - max(plo, lo))
        span = max(phi, hi) - min(plo, lo)
        iou = inter / span if span > 0 else 0.0
        eqs = max(0.0, 1 - abs((plo + phi) / 2 - (lo + hi) / 2) / w) if w > 0 else 0.0
        return 0.5 * iou + 0.5 * eqs
    p, pu = normalize_value_with_unit(pred)
    if p is None or not same_dimension(pu, gu):
        return 0.0
    if lo <= p <= hi:
        return 1.0
    if w <= 0:
        return 0.0
    d = lo - p if p < lo else p - hi
    return 0.5 * (math.exp(-5.0 * d / w) + math.exp(-5.0 * (d + w) / w))


_POS, _NEG = ('yes', 'true', 'correct'), ('no', 'false', 'incorrect')


def binary_label(text) -> Optional[str]:
    m = re.search(r'\b(yes|no|true|false|correct|incorrect)\b', str(text or ''), re.I)
    if not m:
        return None
    return 'yes' if m.group(1).lower() in _POS else 'no'


def acc_type_c(pred, gt):
    p = binary_label(pred)
    return 1.0 if p is not None and p == binary_label(gt) else 0.0


def comparison_direction(text) -> Optional[str]:
    t = str(text or '').lower()
    if any(k in t for k in ('faster', 'quicker', 'greater', 'larger', 'more', 'higher')):
        return 'greater'
    if any(k in t for k in ('slower', 'less', 'smaller', 'fewer', 'lower')):
        return 'less'
    if any(k in t for k in ('equal', 'same')):
        return 'equal'
    return None


def _has_number(text):
    return re.search(r'\d', re.sub(r'[a-zA-Z°]+', ' ', str(text or ''))) is not None


TYPE_D_PROMPT = """你是物理推理评估专家。请评估学生的比较判断答案。

**问题：**
{question}

**标准答案（Ground Truth）：**
{gt}

**学生答案：**
{pred}

**评分标准：**
标准答案包含方向和数值，评分分为两部分：

1. **比较方向（50%）：**
   - 学生是否正确判断了比较方向（faster/slower, greater/less, more/fewer等）
   - 方向正确：0.5分
   - 方向错误：0分

2. **数值精度（50%）：**
   - 如果学生给出了数值倍数：
     * 相对误差 ≤ 2%：0.5分
     * 相对误差 2%-30%：0.5 × exp(-8 × (误差 - 0.02))
     * 相对误差 > 30%：0分
   - 如果学生只有方向没有数值：数值精度为0分

**返回格式（纯JSON）：**
```json
{{
  "direction_correct": true/false,
  "direction_score": 0.5或0.0,
  "has_numeric": true/false,
  "numeric_error": 误差值（如0.05表示5%误差）或null,
  "numeric_score": 0.0到0.5之间,
  "total_score": 0.0到1.0之间,
  "explanation": "简要说明"
}}
```"""


def _acc_judge_key():
    return config.key(config.ACC_JUDGE_KEY_ENV, 'OPENROUTER_API_KEY')


async def acc_type_d(pred, gt, question):
    """.5 * 1[direction] + .5 * A_A(magnitude). With a magnitude in the reference, the ACC judge reads the direction
    and the relative magnitude error of the answer; the score is computed from them with A_A."""
    if not _has_number(gt):
        d = comparison_direction(pred)
        return 1.0 if d is not None and d == comparison_direction(gt) else 0.0
    r = await llm.chat(config.ACC_JUDGE_BASE, _acc_judge_key(), config.ACC_JUDGE_MODEL,
                       [{'role': 'system', 'content': '你是物理评估专家，严格按照评分标准给分。'},
                        {'role': 'user', 'content': TYPE_D_PROMPT.format(question=question, gt=gt, pred=pred)}],
                       temperature=0.0, max_tokens=800)
    j = llm.json_block(llm.content(r))
    s = 0.5 if j.get('direction_correct') in (True, 'true') else 0.0
    err = j.get('numeric_error')
    if j.get('has_numeric') in (True, 'true') and isinstance(err, (int, float)):
        s += 0.5 * point_score(abs(float(err)))
    return s


async def answer_score(pred, qa):
    t = qa['question_type']
    if t == 'TYPE_A':
        return acc_type_a(pred, qa['answer'])
    if t == 'TYPE_B':
        return acc_type_b(pred, qa['answer'], qa.get('answer_range'))
    if t == 'TYPE_C':
        return acc_type_c(pred, qa['answer'])
    if t == 'TYPE_D':
        return await acc_type_d(pred, qa['answer'], qa['question'])
    raise ValueError('unknown question type %r' % t)


# ------------------------------------------------------------------ model-based extraction (no reference shown)
EXTRACT_PROMPT = """Extract answer and reasoning from physics response.

Question: {question}
Type: {question_type}

Raw Response:
{raw_response}

TASK:
Extract the FINAL numerical answer and the reasoning chain from the raw response.
- For TYPE_A: Extract single numerical value with unit
- For TYPE_B: Extract range (e.g., "4-6 J") or single value
- For TYPE_C: Extract Yes/No
- For TYPE_D: Extract comparison result

DO NOT make up answers. Extract ONLY what is present in the raw response.
If the response is incomplete or has no clear answer, set extracted_answer to null.

Output JSON only (no markdown):
{{
  "extracted_answer": "..." or null,
  "extracted_reasoning": "Step 1: ... Step 2: ...",
  "is_incomplete": false,
  "note": "..."
}}"""


async def llm_extract(question, question_type, raw_response):
    try:
        r = await llm.chat(config.ACC_JUDGE_BASE, _acc_judge_key(), config.ACC_JUDGE_MODEL,
                           [{'role': 'system', 'content': 'Physics answer extractor. Output JSON only.'},
                            {'role': 'user', 'content': EXTRACT_PROMPT.format(
                                question=question, question_type=question_type, raw_response=raw_response)}],
                           temperature=0.0, max_tokens=800, retries=8)
        return llm.json_block(llm.content(r))
    except Exception as e:
        logger.error('extraction failed: %s', e)
        return {'extracted_answer': None, 'extracted_reasoning': None, 'is_incomplete': True,
                'note': 'Extraction error: %s' % e}


# ------------------------------------------------------------------ solution judge
SCS_WEIGHTS = {'S_obs': 0.25, 'S_formula': 0.25, 'S_derive': 0.30, 'S_conclude': 0.20}
SCS_SYSTEM = '你是物理推理评估专家，严格按照评分标准给出客观评分。'
# The judge prompt is in Chinese; the appendix prints its English rendering.
SCS_PROMPT = """你是一位物理推理评估专家。请对比学生的推理链与标准推理链（Ground Truth），评估学生推理链的质量。

**标准推理链（Ground Truth）：**
{gt_solution}

**学生的推理链：**
{reasoning}

---

请对学生的推理链进行四个维度的评分，每个维度使用五档评分（0, 0.25, 0.5, 0.75, 1.0）：

1. **S_obs（物理量观测质量）：**
   - 1.0: 从视频中正确识别并提取了所有关键物理量（速度、质量、高度、角度等），数值有明确来源，与GT一致
   - 0.75: 识别了主要物理量，少数量缺失或来源不够明确
   - 0.5: 只识别了部分关键物理量，或物理量来源存疑
   - 0.25: 仅有少量物理量被提及，且缺乏视频观测依据
   - 0.0: 未从视频中提取任何有效物理量

2. **S_formula（物理公式正确性）：**
   - 1.0: 使用的物理公式完全正确，适用于该问题，与GT公式一致
   - 0.75: 公式基本正确，有小的符号或表述问题
   - 0.5: 公式部分正确，但缺少关键项
   - 0.25: 公式错误但思路有一定合理性
   - 0.0: 未使用公式或公式完全错误

3. **S_derive（推导步骤合理性）：**
   - 1.0: 推导步骤完整、逻辑严密、计算正确，覆盖GT的主要推导步骤
   - 0.75: 推导基本完整，有小的跳步或计算误差
   - 0.5: 推导有明显跳步，但主要逻辑正确
   - 0.25: 推导缺失较多步骤，逻辑链不清晰
   - 0.0: 没有推导过程或推导完全错误

4. **S_conclude（结论逻辑性）：**
   - 1.0: 结论明确、与推导一致、与GT结论匹配
   - 0.75: 结论基本正确，表述略有瑕疵
   - 0.5: 结论部分正确，但不够明确
   - 0.25: 结论不清晰或与推导不完全一致
   - 0.0: 没有结论或结论完全错误

**返回格式（纯JSON，不要其他内容）：**
```json
{{
  "S_obs": 0.75,
  "S_formula": 0.75,
  "S_derive": 1.0,
  "S_conclude": 0.5,
  "explanation": "简要说明评分理由（可选）"
}}
```"""


async def scs_judge(reasoning, gt_solution):
    """four component ratings snapped to {0, .25, .5, .75, 1} and their weighted sum"""
    if isinstance(gt_solution, list):
        gt_solution = '\n'.join(gt_solution)
    if isinstance(reasoning, list):
        reasoning = '\n'.join(reasoning)
    r = await llm.chat(config.SCS_JUDGE_BASE, config.key(config.SCS_JUDGE_KEY_ENV), config.SCS_JUDGE_MODEL,
                       [{'role': 'system', 'content': SCS_SYSTEM},
                        {'role': 'user', 'content': SCS_PROMPT.format(gt_solution=gt_solution, reasoning=reasoning)}],
                       temperature=0.3, max_tokens=600, timeout=60.0, retries=4,
                       thinking={'type': 'disabled'})
    j = llm.json_block(llm.content(r))
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    dims = {k: min(levels, key=lambda x: abs(x - float(j.get(k, 0.0) or 0.0))) for k in SCS_WEIGHTS}
    return {'overall': sum(SCS_WEIGHTS[k] * dims[k] for k in SCS_WEIGHTS), **dims}


def ogs(acc, scs):
    return 0.70 * acc + 0.30 * scs


# ------------------------------------------------------------------ one response
async def score_response(qa, raw_response, thinking_opened=False, with_scs=True):
    """parse + ACC + SCS for one question, scored exactly as the paper runs (training/n1/eval_n1.py and the zero-shot
    tables): response_parser.parse_response, then metrics_final.calculate_acc_with_fallback /
    calculate_scs_with_fallback (two-step LLM evaluation when the parser cannot read the response).
    thinking_opened is kept for the callers; the parser reads the text after </think> in every case.
    Returns a dict with prediction, reasoning, ACC, SCS, OGS, scs_details and parse status."""
    from . import metrics_final as mf           # imported lazily: it reads the judge endpoints at import time
    p = parse_response(raw_response)
    rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': qa['answer'],
          'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
          'question': qa['question'], 'question_type': qa['question_type'],
          'needs_llm_eval': p.needs_llm_eval, 'raw_response': raw_response}
    acc = await mf.calculate_acc_with_fallback(rd)
    scs, details = 0.0, None
    if with_scs:
        r = await mf.calculate_scs_with_fallback(rd)
        scs = float(r.get('overall', 0.0))
        d = r.get('details')
        if isinstance(d, dict) and all(k in d for k in SCS_WEIGHTS):
            details = {'overall': scs, **{k: float(d[k]) for k in SCS_WEIGHTS}}
    return {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'parse_status': p.status,
            'needs_llm_eval': p.needs_llm_eval, 'is_missing': bool(rd.get('is_missing')), 'ACC': float(acc),
            'SCS': scs, 'scs_details': details, 'OGS': mf.calculate_ogs(acc, scs), 'judge_errors': []}


def summarize(rows):
    """mean ACC / SCS / OGS and mean SCS components of a list of scored rows"""
    if not rows:
        return {}
    n = len(rows)
    s = {k: sum(r[k] for r in rows) / n for k in ('ACC', 'SCS', 'OGS')}
    det = [r['scs_details'] for r in rows if r.get('scs_details')]
    s['SCS_dims'] = {k: (sum(d[k] for d in det) / len(det) if det else None) for k in SCS_WEIGHTS}
    s['n'] = n
    return s
