"""
classify_b1_b2.py  —  Stage 1.5：识别B类视频中可web-search的物体

所有B类视频统一为B（scale-free）。本模块不再区分B1/B2，
而是专注于识别视频中有可查物理规格的物体标签（供Stage 1.5b web search使用）。

触发时机: Stage 1（VLM提取）完成之后，Stage 2（事件图）之前。
输入: stage1_omni_info.json 中的 visual_elements / global_summary
输出: stage2b_class.json
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("classify_b1_b2")

# LLM (DeepSeek-V3.2), default through OpenRouter (model "deepseek/deepseek-v3.2").
# Set DEEPSEEK_BASE (e.g. https://api.deepseek.com) + DEEPSEEK_API_KEY + DEEPSEEK_MODEL to use another endpoint.
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE", "https://openrouter.ai/api/v1")
LLM_API_KEY  = (os.environ.get("OPENROUTER_API_KEY", "") if "openrouter.ai" in LLM_BASE_URL
                else os.environ.get("DEEPSEEK_API_KEY", ""))
LLM_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v3.2")

# Objects that almost always have web-searchable physical specs
_KNOWN_B1_KEYWORDS = {
    # Ball sports
    "basketball", "football", "soccer ball", "baseball", "tennis ball",
    "volleyball", "golf ball", "bowling ball", "rugby ball",
    "cricket ball", "handball",
    # Billiards / snooker / pool
    "snooker ball", "cue ball", "billiard ball", "pool ball",
    "snooker", "billiard", "pool cue", "cue stick",
    # Athletics
    "javelin", "shot put", "discus", "hammer", "hurdle", "barbell",
    # Vehicles
    "bicycle", "motorcycle", "bus", "truck",
    # Other sports equipment
    "arrow", "bow", "puck", "frisbee",
}

# Short keywords that need exact-word matching (prevent substring false positives)
_EXACT_MATCH_KEYWORDS = {kw for kw in _KNOWN_B1_KEYWORDS if len(kw.split()) == 1 and len(kw) <= 5}

# Objects that almost never have standardized specs
_KNOWN_B2_KEYWORDS = {
    "water", "fire", "smoke", "cloud", "wave", "sand", "mud",
    "person", "human", "man", "woman", "child", "athlete",
    "hand", "arm", "leg", "body",
    "tree", "rock", "stone", "grass",
}


def _extract_object_labels(graph: Dict[str, Any]) -> List[str]:
    """Pull unique object/entity labels from Stage 2 event graph nodes."""
    import re
    labels = set()
    for node in graph.get("nodes", []):
        # Only extract from object/entity nodes, not action/event/state nodes
        node_type = node.get("type", "")
        if node_type not in ("object", "entity", ""):
            continue
        # Prefer explicit label/name fields; fall back to first noun phrase in content
        label = node.get("label", "") or node.get("name", "")
        if not label:
            content = node.get("content", "")
            # Take first meaningful noun (up to 3 words) before any verb/adjective clause
            m = re.match(r"^([A-Za-z][\w\s]{1,30}?)(?:\s+(?:wearing|with|on|in|at|that|which|who)\b|$)", content)
            label = m.group(1).strip() if m else content.split(",")[0].strip()
        if label and node_type != "type_fallback":
            labels.add(label.lower().strip())
    # Also pull from physics_matches if already computed
    for pm in graph.get("physics_matches", []):
        for entity in pm.get("participating_entities", []):
            if isinstance(entity, str):
                labels.add(entity.lower().strip())
            elif isinstance(entity, dict):
                lbl = entity.get("label", "") or entity.get("name", "")
                if lbl:
                    labels.add(lbl.lower().strip())
    return sorted(labels)


def _extract_labels_from_omni_info(omni_info: Dict[str, Any]) -> List[str]:
    """Pull object labels from Stage 1 omni_info (visual_elements + global_summary)."""
    import re
    labels = set()

    clips = omni_info.get("mid_level", {}).get("clips", [])
    for clip in clips:
        # visual_elements: list of strings like "Standard orange/brown basketball"
        for ve in clip.get("visual_elements", []):
            # Strip leading descriptors (colours, adjectives) to get the core noun
            # e.g. "Standard orange/brown basketball" → "basketball"
            # Take last 1-3 words as the noun phrase
            words = ve.strip().rstrip(".,").split()
            if len(words) >= 2:
                # Try last 2 words first, then single last word
                for n in (2, 1):
                    candidate = " ".join(words[-n:]).lower().strip("()[]")
                    if len(candidate) > 2 and not candidate[0].isupper():
                        labels.add(candidate)
                        break
            elif words:
                labels.add(words[-1].lower())

        # global_summary: extract noun phrases heuristically
        summary = clip.get("global_summary", "")
        # Find known B1 keywords directly in the summary text
        summary_lower = summary.lower()
        for kw in _KNOWN_B1_KEYWORDS:
            if kw in summary_lower:
                labels.add(kw)

    # Also check high-level summary
    hl_summary = omni_info.get("high_level", {}).get("global_summary", "")
    hl_lower = hl_summary.lower()
    for kw in _KNOWN_B1_KEYWORDS:
        if kw in hl_lower:
            labels.add(kw)

    # Filter out pure person/body labels that never have specs
    labels = {
        lbl for lbl in labels
        if not any(bkw in lbl or lbl in bkw for bkw in _KNOWN_B2_KEYWORDS)
        and len(lbl) > 2
    }
    return sorted(labels)


def _matches_keyword(lbl: str, kw: str) -> bool:
    """Match label against keyword. Short single words require exact match to
    avoid false positives (e.g. 'car' substring-matching unrelated labels)."""
    if kw in _EXACT_MATCH_KEYWORDS:
        # Exact word boundary match: lbl must equal kw or contain kw as a full word
        import re
        return lbl == kw or bool(re.search(r'\b' + re.escape(kw) + r'\b', lbl))
    # Multi-word or longer keywords: substring match in either direction
    return kw in lbl or lbl in kw


def _quick_classify(labels: List[str]) -> Optional[str]:
    """Fast path: check if any label matches a known searchable keyword."""
    for lbl in labels:
        for kw in _KNOWN_B1_KEYWORDS:
            if _matches_keyword(lbl, kw):
                return "B"
    return None  # no known keyword → fall through to LLM for searchable detection


async def _llm_check_object(
    client: Any,
    sem: asyncio.Semaphore,
    label: str,
    model: str,
) -> Dict[str, Any]:
    """Ask LLM whether this object has standardized physical specs findable via web search."""
    prompt = (
        f'Does the object or item "{label}" have standardized physical specifications '
        f"(such as mass, dimensions, material properties) that can be reliably found "
        f"via web search (e.g. from sports governing bodies, ISO standards, manufacturer specs)?\n\n"
        f"Reply in JSON only, no markdown:\n"
        f'{{"has_spec": true_or_false, "example_spec": "brief example or null", '
        f'"confidence": 0_to_1, "reasoning": "one sentence"}}'
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            logger.warning(f"LLM check failed for '{label}': {e}")
            return {"has_spec": False, "confidence": 0.0, "error": str(e)}


async def classify_b1_b2(
    graph: Optional[Dict[str, Any]] = None,
    out_dir: str = "",
    model: str = LLM_MODEL,
    omni_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry: classify a B-class video as B1 or B2.

    Preferred: pass omni_info (Stage 1 output) — runs after Stage 1.5.
    Fallback:  pass graph (Stage 2 event graph) — legacy Stage 2b position.

    Returns dict saved to stage2b_class.json:
    {
        "video_class": "B1" | "B2",
        "labels_checked": [...],
        "searchable_objects": [...],   # labels with has_spec=true
        "specs_found": {label: example_spec, ...},
        "method": "fast_path" | "llm"
    }
    """
    if omni_info is not None:
        labels = _extract_labels_from_omni_info(omni_info)
    elif graph is not None:
        labels = _extract_object_labels(graph)
    else:
        raise ValueError("classify_b1_b2: must provide omni_info or graph")
    logger.info(f"  [B1/B2 classify] Checking {len(labels)} object labels: {labels}")

    if not labels:
        result = {
            "video_class": "B",
            "labels_checked": [],
            "searchable_objects": [],
            "specs_found": {},
            "method": "no_objects",
        }
        logger.info("  [B classify] No objects found → B")
        return result

    # Fast path
    fast = _quick_classify(labels)
    if fast is not None:
        matched_b1 = [
            lbl for lbl in labels
            if any(_matches_keyword(lbl, kw) for kw in _KNOWN_B1_KEYWORDS)
        ]
        result = {
            "video_class": "B",
            "labels_checked": labels,
            "searchable_objects": matched_b1,
            "specs_found": {lbl: "standard specs (known)" for lbl in matched_b1},
            "method": "fast_path",
        }
        logger.info(f"  [B classify] Fast path → B (searchable: {matched_b1})")
        return result

    # LLM path: query each label concurrently
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem    = asyncio.Semaphore(4)

    checks = await asyncio.gather(*[
        _llm_check_object(client, sem, lbl, model) for lbl in labels
    ])
    await client.close()

    searchable = []
    specs_found = {}
    for lbl, chk in zip(labels, checks):
        if chk.get("has_spec") and chk.get("confidence", 0) >= 0.6:
            searchable.append(lbl)
            if chk.get("example_spec"):
                specs_found[lbl] = chk["example_spec"]

    result = {
        "video_class": "B",
        "labels_checked": labels,
        "searchable_objects": searchable,
        "specs_found": specs_found,
        "method": "llm",
        "llm_details": {lbl: chk for lbl, chk in zip(labels, checks)},
    }
    logger.info(f"  [B classify] LLM path → B (searchable: {searchable})")
    return result


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a B-class video as B1 or B2")
    parser.add_argument("--graph", required=True, help="Path to stage2_event_graph.json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args()

    graph = json.load(open(args.graph, encoding="utf-8"))
    if isinstance(graph, list):
        graph = graph[0]

    result = asyncio.run(classify_b1_b2(graph, args.out_dir, args.model))
    out_path = os.path.join(args.out_dir, "stage2b_class.json")
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(result, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Result: {result['video_class']} → {out_path}")
