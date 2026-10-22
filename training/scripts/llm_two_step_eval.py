"""两步LLM评测：Step 1提取，Step 2调用规则评分（带单位转换）"""
import httpx
import json
import logging
import math
import os
import re

logger = logging.getLogger(__name__)

# Step 1: ACC judge gpt-4o-mini (endpoint and key from the environment, see metrics_final)
JUDGE_API_URL = os.environ.get("ACC_JUDGE_BASE", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
JUDGE_API_KEY = os.environ.get("ACC_JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = os.environ.get("ACC_JUDGE_MODEL", "openai/gpt-4o-mini")

# 单位转换字典（转换为SI单位）
UNIT_CONVERSIONS = {
    # 长度
    'cm': 0.01, 'mm': 0.001, 'km': 1000, 'm': 1,
    # 质量
    'g': 0.001, 'kg': 1, 'mg': 0.000001,
    # 能量
    'J': 1, 'kJ': 1000, 'mJ': 0.001,
    # 力
    'N': 1, 'kN': 1000,
    # 时间
    's': 1, 'ms': 0.001, 'min': 60, 'h': 3600,
    # 速度
    'm/s': 1, 'km/h': 0.277778, 'cm/s': 0.01,
}


def normalize_value_with_unit(text: str) -> tuple:
    """从文本中提取数值和单位，并转换为SI单位
    返回: (si_value, original_unit)
    """
    if not text:
        return (None, None)

    # 匹配数值和单位，如 "4.25 J", "128 cm", "1.5 m/s"
    patterns = [
        r'([0-9]+\.?[0-9]*)\s*(m/s|km/h|cm/s|kJ|mJ|kN|cm|mm|km|mg|kg|ms|min|J|N|m|g|s|h)',
        r'([0-9]+\.?[0-9]*)\s*([a-zA-Z]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)

            # 转换为SI单位
            if unit in UNIT_CONVERSIONS:
                si_value = value * UNIT_CONVERSIONS[unit]
                return (si_value, unit)
            else:
                return (value, unit)

    # 如果没有单位，尝试提取数值
    match = re.search(r'([0-9]+\.?[0-9]*)', text)
    if match:
        return (float(match.group(1)), None)

    return (None, None)


# 区间解析：数值可带 e 记数法；以 "to" 连接时允许负号（"-" 连接时 "-" 是分隔符）
_RNUM = r'(?<![\w.])([0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)'
_RSNUM = r'(?<![\w.])(-?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)'
_RUNIT = r'(?!(?:to|and)\b)([a-zA-Z°%][a-zA-Z/°%²³]*)'
_RKNOWN = r'(m/s|km/h|cm/s|kJ|mJ|kN|cm|mm|km|mg|kg|ms|min|J|N|m|g|s|h)'
_RSEP = r'\s*(?:[-~]|\bto\b)\s*'
_SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
_RANGE_PATTERNS = [
    # (优先级, 正则, 类型)；同一位置先取带单位的写法
    (0, _RNUM + _RSEP + _RNUM + r'\s*' + _RKNOWN, 'unit'),                         # X - Y J
    (0, _RSNUM + r'\s+to\s+' + _RSNUM + r'\s*' + _RKNOWN, 'unit'),                 # -0.3 to 0.6 m/s
    (0, _RNUM + r'\s*' + _RUNIT + _RSEP + _RNUM + r'\s*' + _RUNIT, 'both'),         # 440.64 J to 609.12 J
    (0, _RSNUM + r'\s*' + _RUNIT + r'\s+to\s+' + _RSNUM + r'\s*' + _RUNIT, 'both'), # -2 m/s to 3 m/s
    (1, r'\bbetween\s+' + _RSNUM + r'\s*' + _RKNOWN + r'?\s+and\s+' + _RSNUM + r'\s*' + _RKNOWN + r'?', 'between'),
    (2, _RNUM + _RSEP + _RNUM, 'plain'),                                            # X - Y
    (2, _RSNUM + r'\s+to\s+' + _RSNUM, 'plain'),
]


def _range_from_match(m, kind):
    if kind == 'unit':
        v1, v2, u = float(m.group(1)), float(m.group(2)), m.group(3)
        c = UNIT_CONVERSIONS.get(u, 1)
        return (v1 * c, v2 * c, u)
    if kind == 'plain':
        return (float(m.group(1)), float(m.group(2)), None)
    v1, u1, v2, u2 = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    if kind == 'between':
        u1, u2 = u1 or u2, u2 or u1
        return (v1 * UNIT_CONVERSIONS.get(u1, 1), v2 * UNIT_CONVERSIONS.get(u2, 1), u2)
    if u1 in UNIT_CONVERSIONS and u2 in UNIT_CONVERSIONS:   # 两端单位分别换算
        return (v1 * UNIT_CONVERSIONS[u1], v2 * UNIT_CONVERSIONS[u2], u2)
    if u1 == u2:
        return (v1, v2, u2)
    return None


def normalize_range_with_unit(text: str) -> tuple:
    """从文本中提取区间并转换为SI单位
    区间两端都带单位（"440.64 J to 609.12 J"、"6.12 J - 6.60 J"）、"X to Y unit"、"between X and Y"、
    负下界（"-0.304 to 0.590"）与 e 记数法都按区间解析。
    取文本中最早出现的区间；同一位置优先带单位的写法"""
    if not text:
        return (None, None, None)
    text_clean = text.replace('–', '-').replace('—', '-').replace('−', '-')
    text_clean = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', text_clean)  # 千分位逗号
    text_clean = re.sub(r'([0-9]+\.?[0-9]*)\s*[x×*·]\s*10(?:\s*\^\s*(?:\(\s*([-+]?[0-9]+)\s*\)|([-+]?[0-9]+))|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))',
                        lambda m: m.group(1) + 'e' + (m.group(2) or m.group(3) or m.group(4).translate(_SUP)), text_clean)  # 8.9 x 10^-6
    best = None
    for pri, pat, kind in _RANGE_PATTERNS:
        for m in re.finditer(pat, text_clean, re.I if kind == 'between' else 0):
            if best is not None and (m.start(), pri) >= best[0]:
                break
            r = _range_from_match(m, kind)
            if r is not None:
                best = ((m.start(), pri), r)
                break
    return best[1] if best else (None, None, None)


async def llm_extract_from_raw_response(
    question: str,
    question_type: str,
    raw_response: str
) -> dict:
    """Step 1: 从raw_response提取答案和推理链（不提供GT，避免污染）"""

    prompt = f"""Extract answer and reasoning from physics response.

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

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                JUDGE_API_URL,
                headers={'Authorization': f'Bearer {JUDGE_API_KEY}'},
                json={
                    'model': JUDGE_MODEL,
                    'messages': [
                        {"role": "system", "content": "Physics answer extractor. Output JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    'temperature': 0.0,
                    'max_tokens': 800
                }
            )

        result = response.json()
        content_text = result['choices'][0]['message']['content']

        logger.info(f"  [Step1-Extract] Response: {content_text[:150]}")

        # 解析JSON
        json_text = content_text.strip()
        if json_text.startswith('```'):
            s = json_text.find('{')
            e = json_text.rfind('}')
            if s != -1 and e != -1:
                json_text = json_text[s:e+1]

        parsed = json.loads(json_text)
        return parsed

    except Exception as e:
        logger.error(f"Step1 extraction failed: {e}")
        return {
            'extracted_answer': None,
            'extracted_reasoning': None,
            'is_incomplete': True,
            'note': f'Extraction error: {str(e)}'
        }


# ==================== Step 2: 规则评分函数（带单位转换）====================

def _acc_type_a(pred: str, gt: str) -> float:
    """TYPE-A: 单数值答案，连续衰减评分（带单位转换）"""
    pred_si, pred_unit = normalize_value_with_unit(pred)
    gt_si, gt_unit = normalize_value_with_unit(gt)

    if pred_si is None or gt_si is None or gt_si == 0:
        return 0.0

    delta = abs(pred_si - gt_si) / abs(gt_si)

    if delta <= 0.02:
        return 1.0
    elif delta <= 0.30:
        return math.exp(-8.0 * (delta - 0.02))
    else:
        return 0.0


def _acc_type_b(pred: str, gt: str, gt_range=None) -> float:
    """TYPE-B: 区间答案，支持单值和区间两种回答（带单位转换）"""

    # 解析GT区间（带单位转换）
    if gt_range and len(gt_range) == 2:
        # gt_range已经是数值，检查gt文本中的单位
        gt_si_min, gt_si_max, gt_unit = normalize_range_with_unit(gt)
        if gt_si_min is None:
            # 如果没有提取到，使用提供的gt_range
            gt_lo, gt_hi = float(gt_range[0]), float(gt_range[1])
        else:
            gt_lo, gt_hi = gt_si_min, gt_si_max
    else:
        gt_si_min, gt_si_max, gt_unit = normalize_range_with_unit(gt)
        if gt_si_min is None:
            return 0.0
        gt_lo, gt_hi = gt_si_min, gt_si_max

    # 尝试提取预测区间（带单位转换）
    pred_si_min, pred_si_max, pred_unit = normalize_range_with_unit(pred)

    if pred_si_min is not None and pred_si_max is not None:
        # 情况1：模型回答区间
        pred_lo, pred_hi = pred_si_min, pred_si_max
        # a range answer lying entirely inside the GT range scores 1.0; partial overlap keeps 0.5*IoU + 0.5*EQS
        if min(gt_lo, gt_hi) <= min(pred_lo, pred_hi) and max(pred_lo, pred_hi) <= max(gt_lo, gt_hi):
            return 1.0

        # 计算IoU
        intersection = max(0, min(pred_hi, gt_hi) - max(pred_lo, gt_lo))
        union = max(pred_hi, gt_hi) - min(pred_lo, gt_lo)
        iou = intersection / union if union > 0 else 0.0

        # 计算EQS（中心校准分）
        pred_center = (pred_lo + pred_hi) / 2
        gt_center = (gt_lo + gt_hi) / 2
        gt_width = gt_hi - gt_lo
        eqs = max(0, 1 - abs(pred_center - gt_center) / gt_width) if gt_width > 0 else 0.0

        return 0.5 * iou + 0.5 * eqs

    else:
        # 情况2：模型回答单值（带单位转换）
        pred_si, pred_unit = normalize_value_with_unit(pred)
        if pred_si is None:
            return 0.0

        gt_width = gt_hi - gt_lo

        if gt_lo <= pred_si <= gt_hi:
            # 单值在区间内：满分
            return 1.0
        else:
            # 单值在区间外：使用EQS距离衰减
            if pred_si < gt_lo:
                lower_error = (gt_lo - pred_si) / gt_width
                upper_error = (gt_hi - pred_si) / gt_width
            else:
                lower_error = (pred_si - gt_lo) / gt_width
                upper_error = (pred_si - gt_hi) / gt_width

            eqs_lower = math.exp(-5.0 * lower_error)
            eqs_upper = math.exp(-5.0 * upper_error)
            eqs = (eqs_lower + eqs_upper) / 2.0
            return eqs


def _acc_type_c(pred: str, gt: str) -> float:
    """TYPE-C: Yes/No答案"""
    pred_clean = pred.strip().lower()
    gt_clean = gt.strip().lower()

    pred_yes = 'yes' in pred_clean or '是' in pred_clean
    pred_no = 'no' in pred_clean or '否' in pred_clean
    gt_yes = 'yes' in gt_clean or '是' in gt_clean
    gt_no = 'no' in gt_clean or '否' in gt_clean

    if (pred_yes and gt_yes) or (pred_no and gt_no):
        return 1.0
    return 0.0


async def _acc_type_d_async(pred: str, gt: str, question: str) -> float:
    """TYPE-D: 比较答案（简单匹配）"""
    pred_clean = pred.strip().lower()
    gt_clean = gt.strip().lower()

    if pred_clean == gt_clean:
        return 1.0

    # 简单的比较符号匹配
    if ('>' in pred_clean and '>' in gt_clean) or \
       ('<' in pred_clean and '<' in gt_clean) or \
       ('equal' in pred_clean and 'equal' in gt_clean):
        return 1.0

    return 0.0


async def llm_evaluate_raw_response_two_step(
    question: str,
    question_type: str,
    ground_truth: str,
    annotated_solution: str,
    raw_response: str,
    answer_range = None
) -> dict:
    """
    两步评测主函数
    Step 1: LLM提取答案和推理链（不提供GT）
    Step 2: 调用原有规则评分函数（带单位转换）
    """

    # Step 1: 提取（不提供GT）
    extraction = await llm_extract_from_raw_response(
        question=question,
        question_type=question_type,
        raw_response=raw_response
    )

    extracted_answer = extraction.get('extracted_answer')
    extracted_reasoning = extraction.get('extracted_reasoning', '')

    # 如果提取失败，继续评估raw_response（fallback逻辑）
    if not extracted_answer or extraction.get('is_incomplete'):
        logger.warning(f"  [Two-Step] Extraction failed, will use raw_response for evaluation")
        pred_text = raw_response
        reasoning_text = raw_response
    else:
        pred_text = extracted_answer
        reasoning_text = extracted_reasoning

    # Step 2: 调用原有规则评分函数（带单位转换）
    try:
        if question_type == 'TYPE_A':
            acc_score = _acc_type_a(pred_text, ground_truth)
        elif question_type == 'TYPE_B':
            acc_score = _acc_type_b(pred_text, ground_truth, gt_range=answer_range)
        elif question_type == 'TYPE_C':
            acc_score = _acc_type_c(pred_text, ground_truth)
        elif question_type == 'TYPE_D':
            acc_score = await _acc_type_d_async(pred_text, ground_truth, question)
        else:
            logger.warning(f"Unknown question type: {question_type}")
            acc_score = 0.0
    except Exception as e:
        logger.error(f"ACC calculation failed: {e}")
        acc_score = 0.0

    # SCS评分：直接调用原有的LLM评分函数
    scs_details = {'S_obs': 0.0, 'S_formula': 0.0, 'S_derive': 0.0, 'S_conclude': 0.0}
    scs_score = 0.0

    if reasoning_text and annotated_solution:
        try:
            # 导入并调用SCS评分函数
            from metrics_final import _llm_judge_reasoning
            scs_result = await _llm_judge_reasoning(reasoning_text, annotated_solution)
            if scs_result:
                scs_details = scs_result
                scs_score = (0.25 * scs_details.get('S_obs', 0) +
                            0.25 * scs_details.get('S_formula', 0) +
                            0.30 * scs_details.get('S_derive', 0) +
                            0.20 * scs_details.get('S_conclude', 0))
        except Exception as e:
            logger.warning(f"SCS evaluation failed: {e}")

    return {
        'acc_score': acc_score,
        'scs_score': scs_score,
        'scs_details': {'overall': scs_score, **scs_details},
        'extracted_answer': extracted_answer,
        'extracted_reasoning': extracted_reasoning,
        'is_missing': not bool(extracted_answer),
        'extraction_note': extraction.get('note', '')
    }

