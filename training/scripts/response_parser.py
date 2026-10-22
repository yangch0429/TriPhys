"""响应解析器 - Final版本
严格规则：只从<answer>标签内提取，无fallback机制
修复：只提取</think>后的<answer>内容，占位符也触发LLM评测
"""
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParsedResponse:
    """解析后的响应"""
    prediction: Optional[str] = None
    reasoning_chain: Optional[str] = None
    status: str = "unknown"
    needs_llm_eval: bool = False
    parse_confidence: float = 0.0


def parse_response(raw_response: str) -> ParsedResponse:
    """
    解析模型响应
    
    规则：
    1. 必须有</think>标签
    2. 只在</think>标签之后查找<answer>标签
    3. 如果有多个<answer>，使用最后一个
    4. 如果无</think>或无<answer>或只有占位符：设置needs_llm_eval=True
    """
    result = ParsedResponse()
    
    # 找到</think>标签的位置
    think_end_match = re.search(r"</think>", raw_response, re.IGNORECASE)
    
    if not think_end_match:
        # 没有</think>标签 - 触发LLM fallback
        result.status = "no_think_end_tag"
        result.needs_llm_eval = True
        result.prediction = None
        result.reasoning_chain = None
        result.parse_confidence = 0.0
        return result
    
    # 只在</think>之后搜索
    search_text = raw_response[think_end_match.end():]
    
    # 查找所有<answer>标签，使用最后一个
    all_matches = list(re.finditer(r"<answer>(.*?)</answer>", search_text, re.DOTALL | re.IGNORECASE))
    
    if not all_matches:
        # 无<answer>标签 - 触发LLM fallback
        result.status = "no_answer_tag"
        result.needs_llm_eval = True
        result.prediction = None
        result.reasoning_chain = None
        result.parse_confidence = 0.0
        return result
    
    # 使用最后一个<answer>标签
    answer_match = all_matches[-1]
    answer_content = answer_match.group(1).strip()
    
    # 按"Final answer:"分割
    final_answer_split = re.split(r"Final answer:", answer_content, flags=re.IGNORECASE)
    
    if len(final_answer_split) > 1:
        # 有"Final answer:"标记
        reasoning_part = final_answer_split[0].strip()
        final_answer_part = final_answer_split[1].strip()
        
        # 提取推理链
        if re.search(r"Reasoning chain:", reasoning_part, re.IGNORECASE):
            # 移除"Reasoning chain:"头部
            reasoning_text = re.sub(r"^.*?Reasoning chain:\s*", "", reasoning_part, flags=re.IGNORECASE | re.DOTALL)
            reasoning_text = reasoning_text.strip()
            if reasoning_text and reasoning_text not in ["...", "Step 1: ...\nStep 2: ...\n..."]:
                result.reasoning_chain = reasoning_text
                result.parse_confidence = 1.0
        
        # 提取最终答案
        if final_answer_part and final_answer_part not in ["[answer]", "[your answer here]", "...", "[value]"]:
            result.prediction = final_answer_part
            result.status = "success"
        else:
            # 占位符 - 触发LLM fallback
            result.status = "placeholder_only"
            result.needs_llm_eval = True
            result.parse_confidence = 0.0
    else:
        # 无"Final answer:"标记 - 整个<answer>内容就是答案
        if answer_content and answer_content not in ["...", "[answer]", "[value]"]:
            result.prediction = answer_content
            result.status = "no_final_answer_marker"
            result.parse_confidence = 0.6
        else:
            # 占位符 - 触发LLM fallback
            result.status = "placeholder_only"
            result.needs_llm_eval = True
            result.parse_confidence = 0.0
    
    return result
