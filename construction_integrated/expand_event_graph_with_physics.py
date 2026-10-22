"""
Step 2.5b: Expand Event Graph with Physical Knowledge

Reads the event graphs produced by Stage 2 (build_event_graph.py) and the
Physical Event Library produced by Stage 2.5a, then uses an LLM to:
  1. Match event graph nodes to relevant physical knowledge library entries.
  2. Inject new "latent property" leaf nodes into the graph.
  3. Annotate matched nodes with tool-call rules from the library.

The output is a physics-expanded event graph that Stage 3 (agentic_physics_qa.py)
uses to generate physically grounded, tool-dependent QA pairs.

Usage:
    python expand_event_graph_with_physics.py \
        --input  data_output/event_graph.json \
        --library physical_event_library.json \
        --output data_output/physics_expanded_graph.json \
        --model  deepseek-v3.2 \
        --api_url http://localhost:8089/v1
"""

import json
import os
import asyncio
import argparse
import logging
import sys
import re
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_API_URL = "http://localhost:8089/v1"
DEFAULT_API_KEY = "empty"
DEFAULT_CONCURRENT = 16
CHECKPOINT_INTERVAL = 50


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a physics reasoning expert assisting in the construction of a
multimodal benchmark dataset. Given an event graph extracted from a video and a catalogue of
Physical Event Library entries, your job is to:
1. Identify which graph nodes describe physical phenomena that match one or more library entries.
2. For each match, inject new latent-property nodes into the graph and record the tool-call rules
   that will later be used to generate challenging QA questions.

Always output ONLY valid JSON wrapped in ```json ... ```. No text outside the code block."""


def _format_library_catalogue(entries: List[Dict[str, Any]]) -> str:
    """Compact representation of the library for the prompt."""
    lines = []
    for e in entries:
        cat = e.get("Event_Category", "?")
        triggers = ", ".join(e.get("Visual_Triggers", []))
        desc = e.get("Description", "")
        lines.append(f"• [{cat}] triggers=({triggers}) — {desc}")
    return "\n".join(lines)


def _format_graph_nodes(nodes: List[Dict[str, Any]]) -> str:
    lines = []
    for n in nodes:
        lines.append(
            f"  id={n['id']} type={n.get('type','?')} modality={n.get('modality','?')} "
            f"content=\"{n.get('content','')}\""
        )
    return "\n".join(lines)


def _format_graph_edges(edges: List[Dict[str, Any]]) -> str:
    lines = []
    for e in edges:
        src = e.get("from") or e.get("source", "?")
        rel = e.get("relation") or e.get("type", "->")
        tgt = e.get("to") or e.get("target", "?")
        lines.append(f"  {src} --[{rel}]--> {tgt}")
    return "\n".join(lines) if lines else "  (none)"


def make_matching_prompt(
    graph: Dict[str, Any],
    library_entries: List[Dict[str, Any]],
) -> str:
    nodes_str = _format_graph_nodes(graph.get("nodes", []))
    edges_str = _format_graph_edges(graph.get("edges", []))
    catalogue_str = _format_library_catalogue(library_entries)
    graph_summary = graph.get("graph_summary", "")

    return f"""## Event Graph
Graph Summary: {graph_summary}

Nodes:
{nodes_str}

Edges:
{edges_str}

## Physical Event Library Catalogue
{catalogue_str}

## Task 1 — Single-Node Matching
Carefully read every graph node. For each node whose content describes or implies a physical
phenomenon listed in the catalogue above, perform the following:

1. **Match**: identify the best-matching library entry (by Event_Category).
2. **Inject latent nodes**: for each Latent_Property of the matched entry, create a new graph node:
   - id: "phys_<original_node_id>_<short_variable_name>" (snake_case, no spaces)
   - content: the full Latent_Property string from the library
   - type: "latent_property"
   - modality: "cross_modal"
   - source_info: "injected from Physical Event Library: <Event_Category>"
3. **Add injection edges**: connect the original node to each new latent node with relation "requires_quantification".
4. **Annotate**: record the Actionable_Rules and Relevant_Formulas for the matched entry in a
   "physics_annotation" field on the original node.

Only match nodes where the physical connection is CLEAR and GROUNDED in the node content.
Do NOT speculate or over-match. If no node matches any library entry, return an unchanged graph
with "physics_matches": [].

## Task 2 — Adjacent Action Node Pair Matching
Look at the edges listed above. For each pair of action/state nodes connected by an edge,
check if the TRANSITION between them describes an energy-transfer or state-change physical
phenomenon (examples: sprint→jump = kinetic→potential energy conversion; throw→flight =
impulse→projectile motion; swing→impact = rotational→translational energy transfer).

For each such pair where the transition is physically meaningful:
1. Identify the best-matching library entry for the TRANSITION (not the individual nodes).
2. Inject a single bridge latent node:
   - id: "phys_<nodeA_id>_to_<nodeB_id>_<short_name>" (snake_case)
   - content: description of the energy/state transfer between the two nodes
   - type: "latent_property"
   - modality: "cross_modal"
   - source_info: "injected from Physical Event Library: <Event_Category>"
3. Add edges: nodeA --[energy_transfer]--> bridge_node and bridge_node --[enables]--> nodeB
4. Record in "pair_matches" list (see output format).

Only match pairs where the physical transition is CLEAR. Do not over-match. If no adjacent
pair has a meaningful physical transition, return "pair_matches": [].

## Output Format
```json
{{
  "id": "<same as input graph id>",
  "modality": "<same>",
  "paths": "<same>",
  "graph_summary": "<same or enhanced>",
  "nodes": [ /* all original nodes (with physics_annotation added where matched) + new latent nodes */ ],
  "edges": [ /* all original edges + new requires_quantification and energy_transfer edges */ ],
  "physics_matches": [
    {{
      "source_node_id": "original node id",
      "matched_category": "Domain/EventType",
      "injected_node_ids": ["phys_node_id_1", "phys_node_id_2"],
      "actionable_rules": {{ /* copied from library entry */ }},
      "relevant_formulas": [ /* copied from library entry */ ],
      "example_question_sketch": "copied from library entry"
    }}
  ],
  "pair_matches": [
    {{
      "source_node_ids": ["nodeA_id", "nodeB_id"],
      "matched_category": "Domain/EventType",
      "injected_node_ids": ["bridge_node_id"],
      "actionable_rules": {{ /* copied from library entry */ }},
      "relevant_formulas": [ /* copied from library entry */ ],
      "example_question_sketch": "copied from library entry"
    }}
  ]
}}
```
"""


def extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return {"error": "json_decode_failed", "raw": text[:500]}


# ---------------------------------------------------------------------------
# Core async function
# ---------------------------------------------------------------------------

async def expand_one_graph(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    graph: Dict[str, Any],
    library_entries: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    media_id = graph.get("id", "unknown")

    # Skip graphs that already failed in Stage 2
    if "error" in graph and not graph.get("nodes"):
        logger.warning(f"[{media_id}] Skipping: Stage 2 error present.")
        graph["physics_matches"] = []
        return graph

    prompt = make_matching_prompt(graph, library_entries)

    async with sem:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=16384,
                    temperature=0.2,
                    timeout=600,
                )
                content = response.choices[0].message.content
                result = extract_json(content)

                if "error" in result:
                    logger.warning(
                        f"[{media_id}] Parse error attempt {attempt+1}: {result['error']}"
                    )
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Fallback: return original graph with empty physics_matches
                    graph["physics_matches"] = []
                    graph["_physics_expand_error"] = result["error"]
                    return graph

                # Preserve original fields that LLM might not echo back
                result.setdefault("id", media_id)
                result.setdefault("modality", graph.get("modality"))
                result.setdefault("paths", graph.get("paths"))

                # Merge pair_matches into physics_matches
                pair_matches = result.pop("pair_matches", [])
                for pm in pair_matches:
                    pm["is_pair_match"] = True
                result.setdefault("physics_matches", []).extend(pair_matches)

                n_matches = len(result.get("physics_matches", []))
                n_new_nodes = len(
                    [n for n in result.get("nodes", []) if n.get("type") == "latent_property"]
                )

                # Post-process matches:
                # 1. Drop matches with no source node (LLM hallucinated a disconnected match)
                # 2. Deduplicate by matched_category — keep the first (highest-priority) match
                #    per category so Stage 3 sees diverse physics rather than 3× Projectile_Motion
                valid_matches = [m for m in result.get("physics_matches", [])
                                 if m.get("source_node_id")]
                seen_categories: set = set()
                deduped_matches = []
                for m in valid_matches:
                    cat = m.get("matched_category", "?")
                    if cat not in seen_categories:
                        seen_categories.add(cat)
                        deduped_matches.append(m)
                result["physics_matches"] = deduped_matches

                logger.info(
                    f"[{media_id}] Physics expansion: {n_matches} raw → "
                    f"{len(deduped_matches)} deduped matches, "
                    f"{n_new_nodes} latent nodes injected."
                )
                return result

            except Exception as e:
                logger.error(f"[{media_id}] Attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

    graph["physics_matches"] = []
    graph["_physics_expand_error"] = "all_attempts_failed"
    return graph


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def main_pipeline(args):
    # Load event graphs
    with open(args.input, "r", encoding="utf-8") as f:
        graphs = json.load(f)
    if not isinstance(graphs, list):
        graphs = [graphs]

    # Load physical event library
    with open(args.library, "r", encoding="utf-8") as f:
        library_data = json.load(f)
    library_entries = library_data.get("entries", library_data)
    if not isinstance(library_entries, list):
        logger.error("Library file must contain a list under 'entries' key or be a list itself.")
        return

    # Filter out failed library entries
    library_entries = [e for e in library_entries if "_error" not in e]
    logger.info(
        f"Loaded {len(graphs)} event graphs, {len(library_entries)} library entries."
    )

    # Slice
    graphs = graphs[args.start_index :]
    if args.max_items is not None:
        graphs = graphs[: args.max_items]

    client = AsyncOpenAI(base_url=args.api_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrent)

    tasks = [
        expand_one_graph(client, sem, graph, library_entries, args.model)
        for graph in graphs
    ]

    all_results = []
    for i, future in enumerate(
        tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Expanding graphs")
    ):
        result = await future
        all_results.append(result)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved at {i+1} items.")

    await client.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    n_with_matches = sum(
        1 for r in all_results if r.get("physics_matches")
    )
    n_total_matches = sum(
        len(r.get("physics_matches", [])) for r in all_results
    )
    logger.info(
        f"Done. {len(all_results)} graphs processed. "
        f"{n_with_matches} have physics matches ({n_total_matches} total match events). "
        f"Saved to {args.output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2.5b: Expand Event Graphs with Physical Knowledge"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSON: event graphs from Stage 2 (build_event_graph.py)",
    )
    parser.add_argument(
        "--library", type=str, default="physical_event_library.json",
        help="Physical Event Library JSON (from build_physical_event_library.py)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <input>_physics_expanded.json)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)

    args = parser.parse_args()

    if args.output is None:
        base = args.input.replace(".json", "")
        args.output = f"{base}_physics_expanded.json"

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_pipeline(args))
