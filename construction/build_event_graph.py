import json
import os
import asyncio
import argparse
import logging
import re
import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_API_URL = "http://localhost:8089/v1"
DEFAULT_API_KEY = "empty"
CONCURRENT_LIMIT = 32

GRAPH_CONSTRUCTION_SYSTEM_PROMPT = """You are an expert in multimodal event reasoning and causal graph construction.
Given structured information extracted from a media item (video, image, or audio), your task is to:
1. Identify key events, actions, objects, and states as graph nodes.
2. Establish causal, temporal, or logical relationships between nodes as directed edges.
3. Verify the graph is internally consistent and grounded in the provided evidence.

Always output valid JSON wrapped in ```json ... ```. Do not include any text outside the code block."""


def _get_graph_construction_prompt(media_info: Dict[str, Any], modality: str) -> str:
    info_json = json.dumps(media_info, indent=2, ensure_ascii=False)

    modality_hint = {
        "video_audio": "This is a video with audio. Use both visual scenes/clips and audio transcripts/events as evidence for nodes and edges.",
        "image_only": "This is an image. Use visual objects, OCR text, faces, and scene context as evidence.",
        "audio_only": "This is an audio clip. Use ASR transcripts, speaker attributes, and sound events as evidence.",
    }.get(modality, "Use all available information as evidence.")

    return f"""You are constructing an Omni-Modal Event Graph from structured media information.

{modality_hint}

## Input Media Information
{info_json}

## Task

### Step 1: Node Extraction
Extract discrete, meaningful units of information as nodes. Each node represents an Event, Action, Object, or State.
- Assign a unique concise `id` (e.g., "event_01", "action_open_door", "state_door_open").
- Provide a clear `content` description.
- Specify `type`: one of `event`, `action`, `object`, `state`.
- Provide `source_info`: reference to the specific clip timestamp, ASR segment, or image region that supports this node.
- Provide `modality`: one of `visual`, `audio`, `cross_modal`.

### Step 2: Edge Construction
Establish directed relationships between nodes. Each edge must be:
- **Objective**: grounded in the provided evidence, not inferred speculatively.
- **Typed**: use one of the following relation types:
  - `causes`: Node A directly leads to Node B.
  - `enables`: Node A creates necessary preconditions for Node B.
  - `triggers`: Node A immediately initiates Node B.
  - `results_in`: Node B is the outcome/state change from Node A.
  - `elaborates`: Node B provides specific details or context for Node A.
  - `follows`: Node B temporally succeeds Node A (use only when no stronger causal link exists).
- Provide `evidence`: a specific quote or reference from the input data.

### Step 3: Graph Verification
Before outputting, verify:
1. Every node has at least one edge (no isolated nodes unless the media is very simple).
2. Every edge references valid node IDs.
3. No contradictory edges exist (e.g., A causes B and B causes A simultaneously without justification).
4. All evidence strings are grounded in the input data.

## Output Format

Output strict JSON in ```json ... ``` format:
```json
{{
    "graph_summary": "A concise description of the overall narrative or logic captured by this event graph.",
    "nodes": [
        {{
            "id": "node_id",
            "content": "Description of the event/action/object/state",
            "type": "event|action|object|state",
            "modality": "visual|audio|cross_modal",
            "source_info": "Reference to specific clip/timestamp/ASR segment/image region"
        }}
    ],
    "edges": [
        {{
            "source": "node_id_1",
            "target": "node_id_2",
            "relation": "causes|enables|triggers|results_in|elaborates|follows",
            "evidence": "Specific quote or reference from input data supporting this relation"
        }}
    ],
    "verification": {{
        "all_nodes_connected": true,
        "all_edges_valid": true,
        "no_contradictions": true,
        "notes": "Any caveats or limitations of the constructed graph"
    }}
}}
```
"""


def clean_json_output(content: str) -> Dict[str, Any]:
    if not content:
        return {"error": "empty_content"}

    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end+1])
            except Exception:
                pass
        return {"error": "json_decode_failed", "raw": content[:500]}


async def construct_event_graph(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    item: Dict[str, Any],
    model: str,
    cv_physics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    media_id = item.get("id", "unknown")
    modality = item.get("modality", "video_audio")

    # Exclude gt_physics (raw joint data, ~500KB) — not needed for graph construction
    llm_input = {k: v for k, v in item.items() if k not in ("id", "paths", "gt_physics")}
    # Inject CV measurements so the graph builder can ground physical quantities
    if cv_physics:
        cv_data = cv_physics.get("clip_cv") or cv_physics.get("cv_measurements")
        if cv_data:
            llm_input["cv_physics_measurements"] = cv_data

    prompt = _get_graph_construction_prompt(llm_input, modality)

    async with sem:
        for attempt in range(3):
            try:
                messages = [
                    {"role": "system", "content": GRAPH_CONSTRUCTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=32768,
                    timeout=3600,
                )
                res_content = completion.choices[0].message.content
                result = clean_json_output(res_content)

                final_output = {
                    "id": media_id,
                    "modality": modality,
                    "paths": item.get("paths"),
                }

                if isinstance(result, dict) and "error" not in result:
                    final_output.update(result)
                    node_count = len(result.get("nodes", []))
                    edge_count = len(result.get("edges", []))
                    logger.info(f"[{media_id}] Graph built: {node_count} nodes, {edge_count} edges")
                else:
                    # Try to repair truncated/malformed JSON via a follow-up LLM call
                    logger.warning(f"[{media_id}] JSON parse failed ({result.get('error')}), attempting repair...")
                    repair_messages = messages + [
                        {"role": "assistant", "content": res_content},
                        {"role": "user", "content": (
                            "Your previous response could not be parsed as valid JSON. "
                            "Please output ONLY the raw JSON object (no markdown fences, no extra text). "
                            "Start directly with { and end with }."
                        )},
                    ]
                    repair_completion = await client.chat.completions.create(
                        model=model,
                        messages=repair_messages,
                        max_tokens=32768,
                        timeout=3600,
                    )
                    repair_content = repair_completion.choices[0].message.content
                    repair_result = clean_json_output(repair_content)
                    if isinstance(repair_result, dict) and "error" not in repair_result:
                        final_output.update(repair_result)
                        node_count = len(repair_result.get("nodes", []))
                        edge_count = len(repair_result.get("edges", []))
                        logger.info(f"[{media_id}] Graph repaired: {node_count} nodes, {edge_count} edges")
                    else:
                        final_output["error"] = result.get("error", "unknown")
                        final_output["raw_response"] = res_content[:2000]
                        logger.warning(f"[{media_id}] Graph construction failed after repair: {result.get('error')}")

                return final_output

            except Exception as e:
                logger.error(f"[{media_id}] Attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {
                        "id": media_id,
                        "modality": modality,
                        "paths": item.get("paths"),
                        "error": str(e)
                    }

    return {"id": media_id, "error": "unknown_failure"}


async def main_pipeline(args):
    client = AsyncOpenAI(base_url=args.api_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrent)

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading {args.input}: {e}")
        return

    if not isinstance(input_data, list):
        input_data = [input_data]

    logger.info(f"Loaded {len(input_data)} items from {args.input}")

    tasks = [construct_event_graph(client, sem, item, args.model) for item in input_data]

    final_results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Building Event Graphs"):
        res = await f
        final_results.append(res)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in final_results if "error" not in r)
    logger.info(f"Saved {len(final_results)} event graphs to {args.output} ({success} successful)")
    await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Construct Omni-Modal Event Graphs")
    parser.add_argument("--input", type=str, required=True, help="Input JSON from step1")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--concurrent", type=int, default=CONCURRENT_LIMIT)

    args = parser.parse_args()

    if args.output is None:
        base = args.input.replace('.json', '')
        args.output = f"{base}_event_graph.json"

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_pipeline(args))
