"""Response parser
Strict rule: extract only from inside the <answer> tag, no fallback
Only <answer> content after </think> is extracted; with no </think>, no <answer> or only a placeholder, needs_llm_eval is set (LLM evaluation as fallback)
"""
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParsedResponse:
    """Parsed response"""
    prediction: Optional[str] = None
    reasoning_chain: Optional[str] = None
    status: str = "unknown"
    needs_llm_eval: bool = False
    parse_confidence: float = 0.0


def parse_response(raw_response: str) -> ParsedResponse:
    """
    Parse a model response
    
    Rules:
    1. a </think> tag is required
    2. <answer> tags are searched only after </think>
    3. with several <answer> tags, the last one is used
    4. no </think>, no <answer> or only a placeholder: needs_llm_eval=True
    """
    result = ParsedResponse()
    
    # locate the </think> tag
    think_end_match = re.search(r"</think>", raw_response, re.IGNORECASE)
    
    if not think_end_match:
        # no </think> tag: LLM fallback
        result.status = "no_think_end_tag"
        result.needs_llm_eval = True
        result.prediction = None
        result.reasoning_chain = None
        result.parse_confidence = 0.0
        return result
    
    # search only after </think>
    search_text = raw_response[think_end_match.end():]
    
    # find all <answer> tags, use the last one
    all_matches = list(re.finditer(r"<answer>(.*?)</answer>", search_text, re.DOTALL | re.IGNORECASE))
    
    if not all_matches:
        # no <answer> tag: LLM fallback
        result.status = "no_answer_tag"
        result.needs_llm_eval = True
        result.prediction = None
        result.reasoning_chain = None
        result.parse_confidence = 0.0
        return result
    
    # use the last <answer> tag
    answer_match = all_matches[-1]
    answer_content = answer_match.group(1).strip()
    
    # split on "Final answer:"
    final_answer_split = re.split(r"Final answer:", answer_content, flags=re.IGNORECASE)
    
    if len(final_answer_split) > 1:
        # "Final answer:" marker present
        reasoning_part = final_answer_split[0].strip()
        final_answer_part = final_answer_split[1].strip()
        
        # extract the reasoning chain
        if re.search(r"Reasoning chain:", reasoning_part, re.IGNORECASE):
            # remove the "Reasoning chain:" header
            reasoning_text = re.sub(r"^.*?Reasoning chain:\s*", "", reasoning_part, flags=re.IGNORECASE | re.DOTALL)
            reasoning_text = reasoning_text.strip()
            if reasoning_text and reasoning_text not in ["...", "Step 1: ...\nStep 2: ...\n..."]:
                result.reasoning_chain = reasoning_text
                result.parse_confidence = 1.0
        
        # extract the final answer
        if final_answer_part and final_answer_part not in ["[answer]", "[your answer here]", "...", "[value]"]:
            result.prediction = final_answer_part
            result.status = "success"
        else:
            # placeholder: LLM fallback
            result.status = "placeholder_only"
            result.needs_llm_eval = True
            result.parse_confidence = 0.0
    else:
        # no "Final answer:" marker: the whole <answer> content is the answer
        if answer_content and answer_content not in ["...", "[answer]", "[value]"]:
            result.prediction = answer_content
            result.status = "no_final_answer_marker"
            result.parse_confidence = 0.6
        else:
            # placeholder: LLM fallback
            result.status = "placeholder_only"
            result.needs_llm_eval = True
            result.parse_confidence = 0.0
    
    return result
