import json
import os
import asyncio
import re
import copy
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tqdm import tqdm

from tools.web_tools import web_search, page_browser, web_image_search, save_web_cache
from tools.web_tools import get_openai_function_web_search, get_openai_function_page_browser, get_openai_function_web_image_search
from tools.code_executor import code_executor, get_openai_function_code_executor
from tools.visual_qa import visual_question_answering, get_openai_function_visual_question_answering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8089/v1"
DEFAULT_API_KEY = "empty"
DEFAULT_MODEL = "deepseek-v3.2"
CONCURRENT_LIMIT = 16


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def messages_to_dict(messages: List[Any]) -> List[Dict[str, Any]]:
    serializable = []
    for msg in messages:
        if hasattr(msg, 'model_dump'):
            serializable.append(msg.model_dump())
        elif hasattr(msg, 'dict'):
            serializable.append(msg.dict())
        elif isinstance(msg, dict):
            serializable.append(msg)
        else:
            try:
                serializable.append(dict(msg))
            except Exception:
                serializable.append({"content": str(msg)})
    return serializable


def _build_quality_check_prompt(item: Dict[str, Any]) -> str:
    return f"""You are an expert QA quality evaluator for multimodal benchmark datasets.

Your task is to evaluate the quality of the following multimodal QA item across multiple dimensions.

**IMPORTANT: Output your JSON evaluation directly. Do NOT call any tools or perform web searches.**

## QA Item to Evaluate
```json
{json.dumps(item, indent=2, ensure_ascii=False)}
```

## Evaluation Criteria

### 1. Question Understandability
Is the question clear, grammatically correct, and unambiguous? Can a human understand what is being asked?

### 2. Answer Correctness and Uniqueness
Is the answer factually correct? Is it the ONLY correct answer (no alternative valid answers)?

**IMPORTANT — range answers are valid**: If the question explicitly asks for a range (e.g., "what
range of kinetic energies…") OR the answer is stated as a range (e.g., "101–251 J"), this is
CORRECT and UNIQUE as long as the range bounds are uniquely determinable from the given inputs.
A range answer reflecting physical measurement uncertainty (e.g., ±15% CV calibration error)
is NOT "non-unique" — it IS the unique, honest answer. Mark `is_answer_correct_and_unique: yes`
when the physics and calculation are correct and the answer format matches the question, even if
the answer is a range rather than a single number.

**IMPORTANT — answer_range field**: If the item contains an `answer_range` field (e.g. [0.08, 0.15]),
this explicitly declares that any answer within that interval is considered correct. When checking
correctness, verify that the central value in `answer` falls within `answer_range`, and that the
physics used to derive both bounds is sound. Do NOT penalize for the answer being a range.

### 3. Modality Necessity
- `answerable_without_visual_content`: Can the question be answered WITHOUT looking at any image/video? (yes/no)
- `answerable_without_audio_content`: Can the question be answered WITHOUT listening to any audio? (yes/no)
- `answerable_without_visual_and_audio_content`: Can the question be answered WITHOUT any media at all? (yes/no)

### 4. Tool Necessity
- `answerable_without_tools`: Can the question be answered WITHOUT using any external tools (web search, code execution, etc.)? (yes/no)

### 5. Multi-hop Reasoning Quality
Does the question genuinely require multi-hop reasoning across multiple information sources? Is the reasoning chain in the annotated solution logically sound?

### 6. Overall Quality
Rate the overall quality: excellent / good / fair / poor

## Output Format
Output a JSON object in a markdown code block:
```json
{{
  "is_question_understandable": "yes|no",
  "is_answer_correct_and_unique": "yes|no",
  "answerable_without_visual_content": "yes|no",
  "answerable_without_audio_content": "yes|no",
  "answerable_without_visual_and_audio_content": "yes|no",
  "answerable_without_tools": "yes|no",
  "multi_hop_reasoning_quality": "excellent|good|fair|poor",
  "overall_quality": "excellent|good|fair|poor"
}}
```
"""


async def check_single_item(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    item: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    qa = item.get("multi_hop_qa_pair", item)
    prompt = _build_quality_check_prompt(qa)

    tools = [
        get_openai_function_web_search(),
        get_openai_function_page_browser(),
        get_openai_function_code_executor(),
        get_openai_function_visual_question_answering(),
    ]

    messages = [{"role": "user", "content": prompt}]

    max_turns = 10
    current_turn = 0
    final_result = None

    async with sem:
        while current_turn < max_turns:
            current_turn += 1
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="none",
                    max_tokens=8192,
                    timeout=3600,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                break

            message = response.choices[0].message
            messages.append(message)
            content = message.content or ""

            if "```json" in content:
                try:
                    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if match:
                        result = json.loads(match.group(1))
                        if "is_question_understandable" in result:
                            result.pop("evaluation_notes", None)
                            # CRITICAL: Only keep evaluation fields, discard any other fields
                            # that LLM might have incorrectly copied/modified from the input item
                            allowed_eval_fields = {
                                "is_question_understandable",
                                "is_answer_correct_and_unique",
                                "answerable_without_visual_content",
                                "answerable_without_audio_content",
                                "answerable_without_visual_and_audio_content",
                                "answerable_without_tools",
                                "multi_hop_reasoning_quality",
                                "overall_quality",
                            }
                            final_result = {k: v for k, v in result.items() if k in allowed_eval_fields}
                            break
                except Exception:
                    pass

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception:
                        args = {}

                    tool_result = None
                    try:
                        if func_name == "web_search":
                            tool_result = await web_search(**args)
                        elif func_name == "page_browser":
                            tool_result = await page_browser(**args)
                        elif func_name == "code_executor":
                            tool_result = await code_executor(**args)
                        elif func_name == "visual_question_answering":
                            tool_result = await visual_question_answering(**args)
                        else:
                            tool_result = {"error": f"Unknown tool {func_name}"}
                    except Exception as e:
                        tool_result = {"error": str(e)}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
            else:
                if not final_result:
                    messages.append({
                        "role": "user",
                        "content": "Please output your evaluation in the required JSON format wrapped in ```json ... ```."
                    })

    result_item = copy.deepcopy(item)
    if final_result:
        qa_section = result_item.get("multi_hop_qa_pair", result_item)
        qa_section.update(final_result)
    else:
        qa_section = result_item.get("multi_hop_qa_pair", result_item)
        qa_section["quality_check_error"] = "no_result_after_max_turns"

    return result_item


async def main():
    parser = argparse.ArgumentParser(description="Step 4b: Check QA Quality")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checked_output")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--concurrent", type=int, default=CONCURRENT_LIMIT)
    args = parser.parse_args()

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_url, timeout=3600)
    sem = asyncio.Semaphore(args.concurrent)

    items = load_json(args.input_file)
    if isinstance(items, dict):
        items = [items]

    if args.max_items > 0:
        items = items[:args.max_items]

    logger.info(f"Checking quality for {len(items)} items")

    tasks = [check_single_item(client, sem, item, args.model) for item in items]
    results = []

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Checking QA Quality"):
        try:
            res = await future
            results.append(res)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            results.append({"error": str(e)})

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    out_path = os.path.join(args.output_dir, f"checked_{base_name}_{len(results)}_{timestamp}.json")
    save_json(results, out_path)

    logger.info(f"Saved {len(results)} checked items to {out_path}")

    await client.aclose()
    save_web_cache()


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
