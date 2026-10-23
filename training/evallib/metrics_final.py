"""
评测指标计算模块：ACC, SCS, TUA, OGS
支持混合评分：规则评分 + LLM-Judge评分

1. TYPE_B支持单值处理（单值在区间内得1.0，区间外用EQS衰减）
2. TYPE_D：GT含数值时由ACC judge判断（方向50% + 数值50%）
3. SCS由SCS judge（DeepSeek-V4-Flash）评估4维度

Endpoints, models and keys are read from the environment (.env.example):
  ACC judge  ACC_JUDGE_BASE / ACC_JUDGE_MODEL (gpt-4o-mini) / ACC_JUDGE_API_KEY or OPENROUTER_API_KEY
  SCS judge  SCS_JUDGE_BASE / SCS_JUDGE_MODEL (deepseek-flash, thinking disabled) / DEEPSEEK_API_KEY
"""
import re
import json
import math
import os
from typing import List, Dict, Any, Optional
import logging
import asyncio
import httpx

logger = logging.getLogger(__name__)


# judge POSTs are retried with backoff on 429 / 5xx (the retry does not affect the scores).
# The retry is installed on httpx.AsyncClient so that llm_two_step_eval (same process) uses it too.
_orig_async_post = httpx.AsyncClient.post
async def _post_retry(self, url, *a, **kw):
    for _i in range(8):
        _r = await _orig_async_post(self, url, *a, **kw)
        if _r.status_code not in (429, 500, 502, 503, 504) or _i == 7:
            return _r
        await asyncio.sleep(min(60, 5 * 2 ** _i))
if not getattr(httpx.AsyncClient.post, "_retry", False):
    _post_retry._retry = True
    httpx.AsyncClient.post = _post_retry
from llm_two_step_eval import llm_evaluate_raw_response_two_step

# ACC judge (TYPE_D with a numerical reference, two-step extraction of nonconforming responses): gpt-4o-mini
JUDGE_API_URL = os.environ.get("ACC_JUDGE_BASE", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
JUDGE_API_KEY = os.environ.get("ACC_JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = os.environ.get("ACC_JUDGE_MODEL", "openai/gpt-4o-mini")
# SCS judge (_llm_judge_reasoning): DeepSeek-V4-Flash, thinking disabled
import re as _re
SCS_API_URL = os.environ.get("SCS_JUDGE_BASE", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
SCS_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
SCS_MODEL = os.environ.get("SCS_JUDGE_MODEL", "deepseek-flash")
SCS_EXTRA = {"thinking": {"type": "disabled"}}


async def calculate_acc_async(results: List[Dict]) -> float:
    """
    计算答案正确性（ACC）- 异步版本
    支持TYPE-A/B/C/D四种题型
    """
    if not results:
        return 0.0

    scores = []
    for result in results:
        if 'error' in result:
            continue

        question_type = result.get('question_type')
        pred = result.get('prediction')
        gt = result.get('ground_truth', '')
        question = result.get('question', '')

        # 如果prediction为None或空，直接记0分
        # 如果prediction为None或空，检查是否需要LLM评测
        if pred is None or pred == '':
            if result.get('needs_llm_eval'):
                try:
                    eval_result = await llm_evaluate_raw_response_two_step(
                        question=result.get('question', ''),
                        question_type=question_type,
                        ground_truth=gt,
                        annotated_solution=result.get('annotated_solution', ''),
                        raw_response=result.get('raw_response', ''),
                        answer_range=result.get('answer_range')
                    )
                    scores.append(eval_result['acc_score'])
                    logger.info(f"  [Two-step] ACC={eval_result['acc_score']:.4f}")
                    continue
                except Exception as e:
                    logger.error(f'Two-step eval failed: {e}')
            scores.append(0.0)
            continue

        if question_type == 'TYPE_A':
            score = _acc_type_a(pred, gt)
        elif question_type == 'TYPE_B':
            score = _acc_type_b(pred, gt, gt_range=result.get('answer_range'))
        elif question_type == 'TYPE_C':
            score = _acc_type_c(pred, gt)
        elif question_type == 'TYPE_D':
            score = await _acc_type_d_async(pred, gt, question)
        else:
            logger.warning(f"Unknown question type: {question_type}")
            continue

        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def calculate_acc(results: List[Dict]) -> float:
    """同步wrapper：calculate_acc_async 的同步版本"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(calculate_acc_async(results))


def _acc_type_a(pred: str, gt: str) -> float:
    """TYPE-A: 单数值答案，连续衰减评分（带单位转换）"""
    pred_si, _ = normalize_value_with_unit(pred)
    gt_si, _ = normalize_value_with_unit(gt)

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
        gt_si_min, gt_si_max, _ = normalize_range_with_unit(gt)
        if gt_si_min is None:
            gt_lo, gt_hi = float(gt_range[0]), float(gt_range[1])
        else:
            gt_lo, gt_hi = gt_si_min, gt_si_max
    else:
        gt_si_min, gt_si_max, _ = normalize_range_with_unit(gt)
        if gt_si_min is None:
            return 0.0
        gt_lo, gt_hi = gt_si_min, gt_si_max

    # 尝试提取预测区间（带单位转换）
    pred_si_min, pred_si_max, _ = normalize_range_with_unit(pred)
    pred_range = (pred_si_min, pred_si_max) if pred_si_min is not None else None

    if pred_range:
        # 情况1：模型回答区间
        pred_lo, pred_hi = pred_range
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
        pred_si, _ = normalize_value_with_unit(pred)
        if pred_si is None:
            return 0.0
        pred_val = pred_si

        gt_width = gt_hi - gt_lo

        if gt_lo <= pred_val <= gt_hi:
            # 单值在区间内：满分
            return 1.0
        else:
            # 单值在区间外：使用EQS距离衰减
            if pred_val < gt_lo:
                lower_error = (gt_lo - pred_val) / gt_width
                upper_error = (gt_hi - pred_val) / gt_width
            else:
                lower_error = (pred_val - gt_lo) / gt_width
                upper_error = (pred_val - gt_hi) / gt_width

            eqs_lower = math.exp(-5.0 * lower_error)
            eqs_upper = math.exp(-5.0 * upper_error)
            eqs = (eqs_lower + eqs_upper) / 2.0

            return eqs  # 直接返回EQS，不打折


def _acc_type_c(pred: str, gt: str) -> float:
    """
    TYPE-C: Yes/No分类，二值评分

    评分逻辑：判断正确得1.0，错误得0.0
    """
    pred_label = _extract_binary_label(pred)
    gt_label = _extract_binary_label(gt)

    return 1.0 if pred_label == gt_label else 0.0


async def _acc_type_d_async(pred: str, gt: str, question: str) -> float:
    """
    TYPE-D: 比较判断，使用LLM评分

    评分逻辑：
    1. GT只包含方向：方向正确1.0，错误0.0
    2. GT包含方向+数值：方向50% + 数值精度50%
       - 方向正确：0.5分
       - 数值精度：最高0.5分（误差≤2%得0.5，2%-30%指数衰减，>30%得0）
       - 若回答只有方向无数值：仅得方向分0.5（数值精度为0）
    """
    # 先判断GT是否包含数值
    gt_has_number = _extract_number(gt) is not None

    if not gt_has_number:
        # GT只有方向，使用简单判断
        pred_direction = _extract_comparison_direction(pred)
        gt_direction = _extract_comparison_direction(gt)
        return 1.0 if pred_direction == gt_direction else 0.0

    # GT包含方向+数值，使用LLM完整评估
    prompt = f"""你是物理推理评估专家。请评估学生的比较判断答案。

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

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            JUDGE_API_URL,
            headers={'Authorization': f'Bearer {JUDGE_API_KEY}'},
            json={
                'model': JUDGE_MODEL,
                'messages': [
                    {"role": "system", "content": "你是物理评估专家，严格按照评分标准给分。"},
                    {"role": "user", "content": prompt}
                ],
                'temperature': 0.0,
                'max_tokens': 800
            }
        )

        result = response.json()

        # Debug: 如果返回格式不对，打印出来
        if 'choices' not in result:
            logger.error(f"API返回格式错误: {result}")
            raise ValueError(f"API返回缺少choices字段: {result}")

        content = result['choices'][0]['message']['content'].strip()

        # 提取JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        parsed = json.loads(content)
        total_score = parsed.get('total_score', 0.0)

        return max(0.0, min(1.0, total_score))


async def calculate_scs(results: List[Dict]) -> float:
    """
    计算推理链覆盖分（SCS）- LLM评分

    评分维度（4维度加权）：
    - S_obs（物理量观测质量）：25%
    - S_formula（物理公式正确性）：25%
    - S_derive（推导步骤合理性）：30%
    - S_conclude（结论逻辑性）：20%

    SCS = 0.25×S_obs + 0.25×S_formula + 0.30×S_derive + 0.20×S_conclude
    """
    if not results:
        return 0.0

    scores = []
    for result in results:
        if 'error' in result:
            continue

        # 无推理链计0分
        if not result.get('reasoning_chain'):
            logger.warning(f"Empty reasoning_chain, SCS=0")
            scores.append(0.0)
            continue

        reasoning = result['reasoning_chain']
        gt_solution = result.get('annotated_solution', '')

        try:
            judge_scores = await _llm_judge_reasoning(reasoning, gt_solution)
            s_obs = judge_scores.get('S_obs', 0.0)
            s_formula = judge_scores.get('S_formula', 0.0)
            s_derive = judge_scores.get('S_derive', 0.0)
            s_conclude = judge_scores.get('S_conclude', 0.0)
        except Exception as e:
            logger.warning(f"LLM-Judge评分失败，SCS置零: {e}")
            s_obs = s_formula = s_derive = s_conclude = 0.0

        scs = 0.25 * s_obs + 0.25 * s_formula + 0.30 * s_derive + 0.20 * s_conclude
        scores.append(scs)

    return sum(scores) / len(scores) if scores else 0.0


async def _llm_judge_reasoning(reasoning: str, gt_solution: str) -> Dict[str, float]:
    """
    使用LLM评估推理链质量

    对比学生的推理链与标准推理链（Ground Truth）

    评估维度（每个维度5档评分：0/0.25/0.5/0.75/1.0）：
    - S_obs: 物理量观测质量
    - S_formula: 物理公式正确性
    - S_derive: 推导步骤合理性
    - S_conclude: 结论逻辑性
    """
    judge_prompt = f"""你是一位物理推理评估专家。请对比学生的推理链与标准推理链（Ground Truth），评估学生推理链的质量。

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

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SCS_API_URL,
                headers={'Authorization': f'Bearer {SCS_API_KEY}'},
                json={
                    'model': SCS_MODEL, **SCS_EXTRA,
                    'messages': [
                        {"role": "system", "content": "你是物理推理评估专家，严格按照评分标准给出客观评分。"},
                        {"role": "user", "content": judge_prompt}
                    ],
                    'temperature': 0.3,
                    'max_tokens': 600
                }
            )

            result = response.json()
            content = result['choices'][0]['message']['content'].strip()

            # 提取JSON（可能包含markdown代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)

            # 验证分数范围
            valid_scores = {0.0, 0.25, 0.5, 0.75, 1.0}
            for key in ['S_obs', 'S_formula', 'S_derive', 'S_conclude']:
                if key not in parsed:
                    parsed[key] = 0.0
                score = parsed[key]
                parsed[key] = min(valid_scores, key=lambda x: abs(x - score))

            # 移除explanation字段
            parsed.pop('explanation', None)

            return parsed

    except json.JSONDecodeError as e:
        logger.error(f"LLM-Judge返回JSON解析失败: {content[:200]}")
        raise
    except Exception as e:
        logger.error(f"LLM-Judge调用失败: {e}")
        raise


def calculate_tua(results: List[Dict]) -> float:
    """
    计算工具调用准确性（TUA†）- 不参与评分，恒返回 0.0
    """
    return 0.0


def calculate_ogs(acc: float, scs: float) -> float:
    """
    计算综合得分（OGS）

    OGS = 0.70 × ACC + 0.30 × SCS
    """
    return 0.70 * acc + 0.30 * scs


# ============ 辅助函数 ============


# 单位转换字典（转换为SI单位）
UNIT_CONVERSIONS = {
    'cm': 0.01, 'mm': 0.001, 'km': 1000, 'm': 1,
    'g': 0.001, 'kg': 1, 'mg': 0.000001,
    'J': 1, 'kJ': 1000, 'mJ': 0.001,
    'N': 1, 'kN': 1000,
    's': 1, 'ms': 0.001, 'min': 60, 'h': 3600,
    'm/s': 1, 'km/h': 0.277778, 'cm/s': 0.01,
}

def normalize_value_with_unit(text: str) -> tuple:
    """从文本中提取数值和单位，并转换为SI单位"""
    if not text:
        return (None, None)
    patterns = [
        r'([0-9]+\.?[0-9]*)\s*(m/s|km/h|cm/s|kJ|mJ|kN|cm|mm|km|mg|kg|ms|min|J|N|m|g|s|h)',
        r'([0-9]+\.?[0-9]*)\s*([a-zA-Z]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit in UNIT_CONVERSIONS:
                si_value = value * UNIT_CONVERSIONS[unit]
                return (si_value, unit)
            else:
                return (value, unit)
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


def _extract_number(text: str) -> Optional[float]:
    """从干净的短文本中提取数值（用于 ground truth）"""
    text = re.sub(r'[a-zA-Z°]+', ' ', text)
    match = re.search(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
    return float(match.group()) if match else None


def _extract_final_number(text: str) -> Optional[float]:
    """
    从模型输出中提取最终答案数值（用于 prediction）

    策略：
    1. 剥离 <think>...</think> 块
    2. 优先查找结论性短语附近的数值
    3. 回退到取文本末尾的最后一个合理数值
    """
    # Step 1: 剥离 <think> 块
    if not text or text is None:
        return None
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not clean:
        clean = text

    # Step 2: 结论性短语紧跟数值
    conclusion_patterns = [
        r'(?:final\s+answer|answer\s+is|result\s+is|equals?|≈|≒)\s*[:\s]*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'approximately\s+\*{0,2}?\$?(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        r'(?:therefore|thus|hence|so)[^.]{0,120}?(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(?:J|joule|kg\s*m|N\b|W\b|m\b|meter)',
        r'\\boxed\{(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\}',
        r'KE\s*[=≈≒]\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
    ]
    for pattern in conclusion_patterns:
        matches = re.findall(pattern, clean, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    # Step 3: 取文本末尾最后一个合理数值
    tail = clean[-600:] if len(clean) > 600 else clean
    nums = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', tail)
    for n in reversed(nums):
        try:
            v = float(n)
            if 1e-4 <= abs(v) <= 1e7 and v != int(v) or (0.1 <= abs(v) <= 9999 and '.' in n):
                return v
        except ValueError:
            continue

    for n in reversed(nums):
        try:
            return float(n)
        except ValueError:
            continue

    return None


def _extract_range(text: str) -> Optional[tuple]:
    """从文本中提取区间 [lo, hi]"""
    # 匹配带括号的标准区间
    m = re.search(r'[\[\(]\s*(-?\d+\.?\d*)\s*[,~–—-]\s*(-?\d+\.?\d*)\s*[\]\)]', text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # "X to Y" 或 "X - Y" 等
    m = re.search(r'(-?\d+\.?\d*)\s*(?:[-,~–—]|\s+to\s+)\s*(-?\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))

    # "between X and Y"
    m = re.search(r'between\s+(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))

    return None


def _extract_binary_label(text: str) -> Optional[str]:
    """提取Yes/No标签"""
    text_lower = text.lower()
    if 'yes' in text_lower or 'true' in text_lower or 'correct' in text_lower:
        return 'yes'
    elif 'no' in text_lower or 'false' in text_lower or 'incorrect' in text_lower:
        return 'no'
    return None


def _extract_comparison_direction(text: str) -> Optional[str]:
    """提取比较方向（greater/less/equal）"""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['faster', 'quicker', 'greater', 'larger', 'more', 'higher']):
        return 'greater'
    elif any(kw in text_lower for kw in ['slower', 'less', 'smaller', 'fewer', 'lower']):
        return 'less'
    elif any(kw in text_lower for kw in ['equal', 'same']):
        return 'equal'
    return None



async def calculate_scs_detailed(results: List[Dict]) -> Dict:
    """
    计算SCS分数并返回详细子分数
    Returns:
        {
            'overall': float,  # 总分
            'details': {       # 子分数
                'S_obs': float,
                'S_formula': float,
                'S_derive': float,
                'S_conclude': float
            }
        }
    """
    if not results:
        return {'overall': 0.0, 'details': None}

    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return {'overall': 0.0, 'details': None}

    result = valid_results[0]  # 单题计算

    # 检查是否有推理链和标准答案
    reasoning = result.get('reasoning_chain', [])
    gt_solution = result.get('annotated_solution', '')

    if not reasoning or not gt_solution:
        return {'overall': 0.0, 'details': None}

    # 如果reasoning_chain是列表，转成字符串
    if isinstance(reasoning, list):
        reasoning = '\n'.join(reasoning)

    try:
        # 调用LLM评分，获取4个维度的分数
        judge_scores = await _llm_judge_reasoning(reasoning, gt_solution)

        # 计算加权总分
        overall = (
            judge_scores['S_obs'] * 0.25 +
            judge_scores['S_formula'] * 0.25 +
            judge_scores['S_derive'] * 0.30 +
            judge_scores['S_conclude'] * 0.20
        )

        return {
            'overall': overall,
            'details': judge_scores
        }
    except Exception as e:
        logger.error(f"SCS detailed calculation failed: {e}")
        return {'overall': 0.0, 'details': None}


async def llm_evaluate_raw_response(
    question: str,
    question_type: str,
    ground_truth: str,
    annotated_solution: str,
    raw_response: str,
    answer_range = None
) -> Dict[str, Any]:
    """
    当模型没有输出<answer>标签时，使用LLM从raw_response评测ACC和SCS（单步版本；评测使用两步版本
    llm_two_step_eval.llm_evaluate_raw_response_two_step）
    """

    type_rules = {
        'TYPE_A': 'Extract LAST numerical value. Soft decay: <=2% → 1.0, 2-30% → exp(-8*(delta-0.02)), >30% → 0.0',
        'TYPE_B': 'Extract range or single value. IoU+EQS scoring.',
        'TYPE_C': 'Extract Yes/No from reasoning.',
        'TYPE_D': 'Extract comparison (greater/less/equal).'
    }

    prompt = f"""Evaluate physics response without <answer> tag.

Question: {question}
Type: {question_type}
Ground Truth: {ground_truth}

Gold Solution:
{annotated_solution}

Raw Response:
{raw_response}

TASK:
1. Extract the LAST numerical answer from raw response
2. Extract reasoning chain if present
3. Evaluate ACC vs ground truth using TYPE-SPECIFIC rules below
4. Evaluate SCS using 5-level scoring (0, 0.25, 0.5, 0.75, 1.0)

**ACC Evaluation Rules by Type:**
- TYPE_A: Soft decay scoring. <=2% error -> 1.0, 2-30% -> exp(-8*(delta-0.02)), >30% -> 0.0
- TYPE_B: IoU+EQS for ranges. Full overlap -> 1.0, partial -> IoU score
- TYPE_C: Yes/No exact match
- TYPE_D: Comparison exact match (greater/less/equal)

IMPORTANT: You MUST use ONLY these exact 5 values: 0, 0.25, 0.5, 0.75, 1.0
Do NOT use negative numbers or any other values!

**SCS Evaluation (5-level: 0, 0.25, 0.5, 0.75, 1.0):**

S_obs (Physical Quantity Observation):
- 1.0: All key quantities identified from video, values match GT
- 0.75: Main quantities identified, minor missing
- 0.5: Some key quantities identified
- 0.25: Few quantities mentioned
- 0.0: No valid quantities

S_formula (Physics Formula):
- 1.0: Formulas completely correct, match GT
- 0.75: Mostly correct, minor notation issues
- 0.5: Partially correct, missing key terms
- 0.25: Wrong but has merit
- 0.0: No formula or completely wrong

S_derive (Derivation):
- 1.0: Complete, logical, matches GT
- 0.75: Mostly sound, minor gaps
- 0.5: Partial with logical issues
- 0.25: Major flaws
- 0.0: No derivation or illogical

S_conclude (Conclusion):
- 1.0: Correct, well-supported, matches GT
- 0.75: Mostly correct, minor issues
- 0.5: Partially correct
- 0.25: Weakly supported
- 0.0: No conclusion or wrong

If no answer/reasoning, set is_missing=true and scores=0.

Output JSON only (no markdown blocks):
{{{{
  "extracted_answer": "..." or null,
  "extracted_reasoning": "..." or null,
  "acc_score": 0.0,
  "scs_scores": {{{{"S_obs": 0.0, "S_formula": 0.0, "S_derive": 0.0, "S_conclude": 0.0}}}},
  "is_missing": false,
  "note": "..."
}}}}"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                JUDGE_API_URL,
                headers={'Authorization': f'Bearer {JUDGE_API_KEY}'},
                json={
                    'model': JUDGE_MODEL,
                    'messages': [
                        {"role": "system", "content": "Expert physics evaluator. Output JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    'temperature': 0.1,
                    'max_tokens': 1000
                }
            )

        result = response.json()
        content_text = result['choices'][0]['message']['content']

        # 解析JSON（处理Markdown）
        import json
        logger.info(f"  [LLM] Response: {content_text[:200]}")
        json_text = content_text.strip()
        if json_text.startswith('```'):
            s = json_text.find('{{')
            e = json_text.rfind('}}')
            if s != -1 and e != -1:
                json_text = json_text[s:e+1]
        parsed = json.loads(json_text)

        scs = parsed.get('scs_scores', {})
        scs_overall = 0.25*scs.get('S_obs',0) + 0.25*scs.get('S_formula',0) + 0.30*scs.get('S_derive',0) + 0.20*scs.get('S_conclude',0)

        return {
            'acc_score': parsed.get('acc_score', 0.0),
            'scs_score': scs_overall,
            'scs_details': {'overall': scs_overall, **scs},
            'extracted_answer': parsed.get('extracted_answer'),
            'extracted_reasoning': parsed.get('extracted_reasoning'),
            'is_missing': parsed.get('is_missing', False),
            'extraction_note': parsed.get('note', '')
        }
    except Exception as e:
        logger.error(f'LLM fallback eval failed: {e}')
        return {
            'acc_score': 0.0,
            'scs_score': 0.0,
            'scs_details': {'overall': 0.0, 'S_obs': 0.0, 'S_formula': 0.0, 'S_derive': 0.0, 'S_conclude': 0.0},
            'is_missing': True
        }


async def calculate_acc_with_fallback(result: Dict) -> float:
    """计算ACC，支持LLM fallback"""
    prediction = result.get('prediction')

    # 有prediction，正常计算
    if prediction:
        try:
            return await calculate_acc_async([result])
        except:
            pass

    # 无prediction，检查是否需要LLM评测
    if result.get('needs_llm_eval'):
        try:
            eval_result = await llm_evaluate_raw_response_two_step(
                question=result.get('question', ''),
                question_type=result.get('question_type', 'TYPE_A'),
                ground_truth=result.get('ground_truth', ''),
                annotated_solution=result.get('annotated_solution', ''),
                raw_response=result.get('raw_response', ''),
                answer_range=result.get('answer_range')
            )

            if eval_result.get('is_missing'):
                result['is_missing'] = True

            return eval_result['acc_score']
        except Exception as e:
            logger.error(f'ACC fallback failed: {e}')

    return 0.0


async def calculate_scs_with_fallback(result: Dict) -> Dict:
    """计算SCS，支持LLM fallback"""
    reasoning = result.get('reasoning_chain')

    # 有reasoning_chain，正常计算
    if reasoning:
        try:
            return await calculate_scs_detailed([result])
        except:
            pass

    # 无reasoning_chain，检查是否需要LLM评测
    if result.get('needs_llm_eval'):
        try:
            eval_result = await llm_evaluate_raw_response_two_step(
                question=result.get('question', ''),
                question_type=result.get('question_type', 'TYPE_A'),
                ground_truth=result.get('ground_truth', ''),
                annotated_solution=result.get('annotated_solution', ''),
                raw_response=result.get('raw_response', ''),
                answer_range=result.get('answer_range')
            )

            return {
                'overall': eval_result['scs_score'],
                'details': eval_result['scs_details']
            }
        except Exception as e:
            logger.error(f'SCS fallback failed: {e}')

    return {'overall': 0.0, 'details': {'S_obs': 0.0, 'S_formula': 0.0, 'S_derive': 0.0, 'S_conclude': 0.0}}
