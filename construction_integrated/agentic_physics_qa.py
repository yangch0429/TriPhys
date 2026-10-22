"""
Step 3 (Physics): Agentic Physics QA Construction

Reads physics-expanded event graphs (from expand_event_graph_with_physics.py) and
uses an agentic LLM loop with tool calls to construct challenging multi-hop QA pairs
that REQUIRE:
  - Physical reasoning over latent variables injected into the event graph
  - At least one code_executor call to compute a physical quantity
  - At least one web_search / page_browser call to retrieve unknown parameters

The output format is compatible with the downstream quality-check pipeline
(check_qa_quality.py, verify_qa_facts.py, filter_valid_qa.py).

Usage:
    python agentic_physics_qa.py \
        --input  data_output/event_graph_physics_expanded.json \
        --output qa_output/physics_qa.json \
        --model  deepseek-v3.2 \
        --api_url http://localhost:8089/v1 \
        --concurrent 16
"""

import os
import json
import re
import time
import random
import asyncio
import argparse
import logging
import sys
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from openai import AsyncOpenAI

from tools.video_search import (
    build_video_index,
    search_related_video_info,
    get_openai_function_search_related_video_info,
)
from tools.web_tools import (
    web_search,
    page_browser,
    web_image_search,
    save_web_cache,
    get_openai_function_web_search,
    get_openai_function_page_browser,
    get_openai_function_web_image_search,
)
from tools.code_executor import (
    code_executor,
    get_openai_function_code_executor,
)
from tools.visual_qa import (
    visual_question_answering,
    get_openai_function_visual_question_answering,
)
import grounding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_BASE_URL = "http://localhost:8089/v1"
DEFAULT_API_KEY = "empty"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 20480
DEFAULT_CONCURRENT_LIMIT = 16
DEFAULT_MAX_ACTIONS = 50


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def get_all_tool_definitions() -> List[Dict]:
    return [
        get_openai_function_search_related_video_info(),
        get_openai_function_web_search(),
        get_openai_function_page_browser(),
        get_openai_function_web_image_search(),
        get_openai_function_code_executor(),
        get_openai_function_visual_question_answering(),
    ]


async def execute_tool(tool_name: str, arguments: Dict[str, Any], current_video_id: str) -> str:
    try:
        if tool_name == "search_related_video_info":
            result = await search_related_video_info(
                query=arguments.get("query", ""),
                current_video_id=current_video_id,
                top_k=5,
                return_full_video=True,
            )
        elif tool_name == "web_search":
            result = await web_search(
                query=arguments.get("query", ""),
                top_k=arguments.get("top_k", 10),
            )
        elif tool_name == "page_browser":
            urls = arguments.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]
            result = await page_browser(urls)
        elif tool_name == "web_image_search":
            result = await web_image_search(
                query=arguments.get("query", ""),
                top_k=arguments.get("top_k", 5),
            )
        elif tool_name == "code_executor":
            result = await code_executor(code=arguments.get("code", ""))
        elif tool_name == "visual_question_answering":
            result = await visual_question_answering(
                file_path=arguments.get("file_path", ""),
                question=arguments.get("question", ""),
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, ensure_ascii=False)
        return str(result)

    except Exception as e:
        logger.error(f"Tool execution error for {tool_name}: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Difficulty tiers and OPQ helpers
# ---------------------------------------------------------------------------

_CONF_THRESHOLD = 0.80       # item-level estimation_confidence threshold (all classes)
_CAL_CONF_THRESHOLD = 0.75  # clip-level: minimum calibration_confidence for metric scale reliability
_CV_METHODS = {"pose_estimation", "yolo_trajectory", "vlm_bbox+calibration"}

_DIFFICULTY_SPECS: Dict[str, Dict] = {
    "easy":   {"label": "EASY",   "hidden_cv": 0, "min_cv": 1, "max_actions": 15},
    "medium": {"label": "MEDIUM", "hidden_cv": 1, "min_cv": 1, "max_actions": 20},
    "hard":   {"label": "HARD",   "hidden_cv": 2, "min_cv": 2, "max_actions": 50},
}


def _collect_all_obs(item: Dict[str, Any]) -> List[Dict]:
    """Extract all observable_physical_quantities entries from an item."""
    all_obs: List[Dict] = []
    high_level = item.get("high_level", {})
    high_ann = (
        high_level.get("annotation", {}) or high_level.get("omni_annotation", {})
        if isinstance(high_level, dict) else {}
    )
    if isinstance(high_ann, dict) and "observable_physical_quantities" in high_ann:
        all_obs.append({"source": "full_video", **high_ann["observable_physical_quantities"]})

    mid_level = item.get("mid_level", {})
    clips = (
        mid_level.get("clips", []) or mid_level.get("vision", {}).get("clips", [])
        if isinstance(mid_level, dict) else []
    )
    for c in clips:
        opq = c.get("observable_physical_quantities", {})
        if opq and (opq.get("subjects") or opq.get("motions")):
            all_obs.append({
                "source": f"{c.get('start_time',0):.1f}–{c.get('end_time',0):.1f} s",
                **opq,
            })
    return all_obs


def _filter_by_confidence(opq_list: List[Dict]) -> List[Dict]:
    """Keep only OPQ entries where calibration_confidence >= _CAL_CONF_THRESHOLD.

    Uses _CAL_CONF_THRESHOLD (clip-level scale reliability) rather than the
    item-level _CONF_THRESHOLD. A clips with moderate calibration quality
    (e.g. 0.75–0.79) pass through; individual items are then filtered by
    estimation_confidence >= _CONF_THRESHOLD in _split_obs_by_method.
    B2 clips have calibration_source="b2_scalefree" and calibration_confidence=0.0
    but should always pass — they are caught by the conf=None path below.
    Entries with no calibration info (e.g. high_level VLM OPQ) are kept.
    """
    result = []
    for opq in opq_list:
        cal = opq.get("calibration", {})
        if not isinstance(cal, dict):
            result.append(opq)
            continue
        # physics_extractor uses "calibration_confidence"; older code used "confidence"
        conf = cal.get("calibration_confidence", cal.get("confidence", None))
        source = cal.get("calibration_source", "")
        if conf is None:
            # No calibration block → high_level VLM entry, keep it
            result.append(opq)
        elif source == "b2_scalefree":
            # B2 clips have scale_px_per_m=None and conf=0.0 by design; always keep
            result.append(opq)
        elif isinstance(conf, (int, float)) and conf >= _CAL_CONF_THRESHOLD:
            result.append(opq)
    return result


def _conf_of(entry: Dict) -> float:
    """Return estimation_confidence of a subject/motion entry (default 0.5)."""
    v = entry.get("estimation_confidence", 0.5)
    return float(v) if isinstance(v, (int, float)) else 0.5


def _is_noisy_cv_item(entry: Dict) -> bool:
    """Return True for CV items that are likely noise and should be discarded."""
    label = (entry.get("label") or entry.get("subject") or "").lower()
    # Generic YOLO class names like "object_13" are noise tracks
    if label.startswith("object_") and label[7:].isdigit():
        return True
    if entry.get("_kind") == "motion":
        speed = entry.get("estimated_speed_ms")
        angle = entry.get("estimated_launch_angle_deg")
        subj  = (entry.get("subject") or "").lower()
        is_human = any(k in subj for k in ("athlete", "runner", "player", "person", "man", "woman"))
        # Implausibly low speed for a moving human (< 0.5 m/s is camera-drift noise)
        if is_human and speed is not None and isinstance(speed, (int, float)) and speed < 0.5:
            return True
        # Near-horizontal angles close to 180° are camera-pan artifacts for human movement
        if is_human and angle is not None and isinstance(angle, (int, float)):
            if abs(angle) > 150 or (90 < abs(angle) < 150):
                return True
    return False


def _dedup_cv_over_vlm(vlm_items: List[Dict], cv_items: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    For each CV motion/subject, suppress the matching VLM entry (same subject, similar event).
    CV-extracted values take precedence over VLM estimates for the same physical quantity.
    """
    def _subj_key(e: Dict) -> str:
        return (e.get("subject") or e.get("label") or "").lower().split()[0]

    cv_subj_keys = {_subj_key(e) for e in cv_items}
    filtered_vlm = []
    for v in vlm_items:
        sk = _subj_key(v)
        # Keep VLM entry only if no CV entry covers the same subject
        if sk not in cv_subj_keys:
            filtered_vlm.append(v)
    return filtered_vlm, cv_items


def _cv_range(entry: Dict) -> Dict:
    """Attach a _measurement_range dict to a CV entry based on measurement_tier or confidence.

    Tier A → ±5%   (independent ref object calibration)
    Tier B → ±12%  (vehicle/person bbox + VLM, or ref_obj mediocre view)
    Tier C → ±25%  (pose-only / weak / circular calibration — qualitative only)
    Tier D → entry should already be filtered out before this call

    Falls back to confidence-based estimate when tier not set.
    """
    tier = entry.get("measurement_tier")
    if tier == "A":
        rel_err = 0.05
    elif tier == "B":
        rel_err = 0.12
    elif tier == "C":
        rel_err = 0.25
    else:
        # Legacy path: no tier assigned yet
        conf = _conf_of(entry)
        if conf >= 0.85:
            rel_err = 0.08
        elif conf >= 0.75:
            rel_err = 0.15
        else:
            rel_err = 0.25

    ranges: Dict[str, Any] = {}
    # mass is never directly measured by CV — it comes from VLM or population tables,
    # so we do NOT generate a range for it (prevents LLM treating it as instrument data)
    for field in ("estimated_height_m", "estimated_speed_ms",
                  "estimated_launch_angle_deg", "estimated_length_m",
                  "estimated_angle_deg", "estimated_duration_s", "estimated_ratio"):
        v = entry.get(field)
        if v is not None and isinstance(v, (int, float)):
            lo = round(v * (1 - rel_err), 2)
            hi = round(v * (1 + rel_err), 2)
            ranges[field] = [lo, hi]
    if ranges:
        entry = dict(entry)
        entry["_measurement_range"] = ranges
        tier_label = f"tier_{tier}" if tier else f"conf={_conf_of(entry):.2f}"
        entry["_range_note"] = f"±{int(rel_err*100)}% ({tier_label})"
    return entry


_HUMAN_LABELS = {"boy", "girl", "man", "woman", "person", "athlete", "runner",
                 "player", "child", "kid", "human", "hurdler", "jumper", "sprinter",
                 "mechanic", "worker", "cyclist", "gymnast", "swimmer", "climber",
                 "goalkeeper", "pitcher", "batter", "boxer", "dancer", "technician"}


def _canonical_label(raw: str) -> str:
    """Normalise subject/motion labels so all human-role variants share a stable dedup key."""
    first = raw.lower().split()[0]
    return "person" if first in _HUMAN_LABELS else first


def _person_role_from_summary(global_summary: str) -> str:
    """Extract a context-appropriate role label from the video global summary.

    Returns the best single word to describe the person (e.g. 'mechanic', 'athlete').
    Falls back to 'athlete' if no specific role is found.
    """
    if not global_summary:
        return "athlete"
    s = global_summary.lower()
    # Priority-ordered role mapping
    for role in ("mechanic", "technician", "worker", "cyclist", "gymnast", "swimmer",
                 "climber", "diver", "pitcher", "batter", "goalkeeper",
                 "boxer", "dancer", "runner", "sprinter", "hurdler",
                 "jumper", "player", "athlete"):
        if role in s:
            return role
    return "athlete"


def _vlm_midpoint_dist(e: Dict) -> float:
    """Distance between an entry's measured value and the midpoint of its VLM-validated range.
    Smaller means better VLM-CV agreement. Returns inf if no validated range."""
    vr = e.get("vlm_validated_range")
    if not vr or len(vr) < 2:
        return float("inf")
    mid = (vr[0] + vr[1]) / 2.0
    val = e.get("estimated_height_m") or e.get("estimated_speed_ms") or 0.0
    return abs(float(val) - mid)


def _dedup_cv_across_clips(cv_items: List[Dict]) -> List[Dict]:
    """Merge duplicate subjects/motions across clips for the same entity.

    All human-athlete label variants (boy/man/athlete/runner/…) are treated as the
    same canonical entity per (method, event) — keeping the highest-confidence entry.
    This prevents the same person appearing multiple times with different labels.

    For subjects, dedup is method-agnostic: pose_estimation and yolo_trajectory both
    measure the same height, so only the highest-confidence reading is kept.

    Tie-breaking (equal confidence): prefer the entry whose measured value falls
    closest to the midpoint of its VLM-validated range — i.e. best VLM-CV agreement.
    """
    # Normalize synonymous object labels so cross-clip duplicates merge correctly.
    # vlm_bbox+calibration can produce different names for the same physical object.
    _LABEL_SYNONYMS: Dict[str, str] = {
        "detached bicycle wheel": "bicycle wheel",
        "front wheel": "bicycle wheel",
        "bicycle front wheel": "bicycle wheel",
        "rear wheel": "bicycle wheel",
        "bicycle rear wheel": "bicycle wheel",
        "bicycle wheel": "bicycle wheel",
    }
    seen: Dict[str, Dict] = {}
    for e in cv_items:
        raw   = (e.get("label") or e.get("subject") or "?")
        label = _canonical_label(raw)
        label = _LABEL_SYNONYMS.get(raw.lower(), label)
        method = e.get("estimation_method", "?")
        event  = (e.get("event") or "").lower()
        kind   = e.get("_kind", "?")
        # For subjects, height is the same physical quantity regardless of CV method.
        # Dedup across methods so only one height estimate survives per canonical label.
        if kind == "subject":
            key = f"subject|{label}|{event}"
        else:
            key = f"{kind}|{label}|{method}|{event}"
        if key not in seen:
            seen[key] = e
        else:
            old_conf = _conf_of(seen[key])
            new_conf = _conf_of(e)
            if new_conf > old_conf:
                seen[key] = e
    return list(seen.values())


def _split_obs_by_method(opq_list: List[Dict], conf_threshold: float = _CONF_THRESHOLD) -> Tuple[List[Dict], List[Dict]]:
    """Returns (vlm_items, cv_items) sorted by confidence descending.

    CV items take priority: VLM entries for the same subject are suppressed when
    a CV measurement exists. Noisy CV tracks (object_N, implausible speeds) are dropped.
    Cross-clip duplicates for the same entity are merged (highest-confidence kept).
    Each CV item is annotated with a _measurement_range reflecting its uncertainty.
    """
    vlm_items: List[Dict] = []
    cv_items: List[Dict] = []
    for opq in opq_list:
        src = opq.get("source", "?")
        for subj in opq.get("subjects", []):
            method = subj.get("estimation_method", "vlm")
            entry = {"_source": src, "_kind": "subject", **subj}
            # Mass cannot be measured by CV — strip it from CV entries to prevent
            # LLM from treating a VLM-inherited mass value as an instrument measurement
            if method in _CV_METHODS:
                entry.pop("estimated_mass_kg", None)
            (cv_items if method in _CV_METHODS else vlm_items).append(entry)
        for motion in opq.get("motions", []):
            method = motion.get("estimation_method", "vlm")
            entry = {"_source": src, "_kind": "motion", **motion}
            (cv_items if method in _CV_METHODS else vlm_items).append(entry)
    # Drop noisy CV items, those below the confidence threshold, and Tier D measurements
    cv_items = [e for e in cv_items
                if not _is_noisy_cv_item(e)
                and _conf_of(e) >= conf_threshold
                and e.get("measurement_tier", "A") != "D"]
    # Merge cross-clip duplicates (same entity, same method) — keep highest confidence
    cv_items = _dedup_cv_across_clips(cv_items)
    # Annotate each CV entry with its measurement uncertainty range
    cv_items = [_cv_range(e) for e in cv_items]
    # Sort by confidence descending so highest-quality values appear first
    cv_items.sort(key=_conf_of, reverse=True)
    vlm_items.sort(key=_conf_of, reverse=True)
    # Suppress VLM entries that have a CV counterpart for the same subject
    vlm_items, cv_items = _dedup_cv_over_vlm(vlm_items, cv_items)
    return vlm_items, cv_items


def _question_leaks_hidden_values(question: str, hidden_cv: List[Dict]) -> bool:
    """Return True if the question text contains any numeric value from hidden_cv.

    Checks exact and rounded forms (1–4 significant decimal places) so near-matches
    like '4.61' (rounded from 4.6115) are also caught.
    """
    import re
    nums_in_question = set(re.findall(r'\d+(?:\.\d+)?', question))
    for item in hidden_cv:
        for k, v in item.items():
            if not isinstance(v, (int, float)):
                continue
            if k in ("conf", "confidence", "tier", "clip_idx", "n_frames"):
                continue
            for precision in (4, 3, 2, 1, 0):
                candidate = str(round(v, precision)) if precision > 0 else str(int(round(v)))
                if candidate in nums_in_question:
                    return True
    return False


def _question_leaks_chosen_values(question: str, hidden_cv_chosen: List[str]) -> bool:
    """Check if question text contains any numeric value declared in hidden_cv_chosen.

    hidden_cv_chosen is a list of strings like ["athlete height = 1.74 m", "GT wrist_speed = 4.6115 m/s"].
    Extracts numbers from those strings and checks if they appear in the question.
    """
    import re
    nums_in_question = set(re.findall(r'\d+(?:\.\d+)?', question))
    for desc in hidden_cv_chosen:
        for num_str in re.findall(r'\d+(?:\.\d+)?', desc):
            try:
                v = float(num_str)
            except ValueError:
                continue
            for precision in (4, 3, 2, 1, 0):
                candidate = str(round(v, precision)) if precision > 0 else str(int(round(v)))
                if candidate in nums_in_question:
                    return True
    return False


def _eligible_difficulties(item: Dict[str, Any]) -> List[str]:
    """Return ordered list of difficulty tiers this item can support.

    A-class: based on n_cv (reliable CV measurements).
    B-class: based on len(web_search_specs) — each hidden spec is one difficulty step.
    """
    video_class = item.get("video_class", "B")

    if video_class == "B":
        raw_specs = item.get("web_search_specs", {})
        n_specs = len(raw_specs) if isinstance(raw_specs, dict) else 0
        eligible = []
        if n_specs >= 1:
            eligible.append("easy")
        if n_specs >= 2:
            eligible.append("medium")
        if n_specs >= 3:
            eligible.append("hard")
        return eligible

    # A-class: use CV measurement count
    all_obs = _collect_all_obs(item)
    filtered = _filter_by_confidence(all_obs)
    _, cv_items = _split_obs_by_method(filtered, conf_threshold=_CONF_THRESHOLD)
    n_cv = len(cv_items)
    eligible = [tier for tier, spec in _DIFFICULTY_SPECS.items() if n_cv >= spec["min_cv"]]
    return eligible


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_physics_matches(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "No physics matches found."
    lines = []
    for m in matches:
        lines.append(f"\n### Match: {m.get('matched_category', '?')}")
        lines.append(f"  Source node: {m.get('source_node_id', '?')}")
        lines.append(f"  Injected latent nodes: {', '.join(m.get('injected_node_ids', []))}")
        rules = m.get("actionable_rules", {})
        if rules:
            lines.append("  Actionable Rules:")
            for k, v in rules.items():
                lines.append(f"    {k}: {v}")
        formulas = m.get("relevant_formulas", [])
        if formulas:
            lines.append("  Relevant Formulas: " + " | ".join(formulas))
        sketch = m.get("example_question_sketch", "")
        if sketch:
            lines.append(f"  Example Sketch: {sketch}")
    return "\n".join(lines)


def _format_graph_for_prompt(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    summary = graph.get("graph_summary", "")

    lines = [f"Graph Summary: {summary}", "", "Nodes:"]
    for n in nodes:
        node_type = n.get("type", "?")
        tag = " [LATENT]" if node_type == "latent_property" else ""
        lines.append(
            f"  [{n['id']}]{tag} ({node_type}, {n.get('modality','?')}) {n['content']}"
        )
        if n.get("source_info"):
            lines.append(f"    Source: {n['source_info']}")
        if n.get("physics_annotation"):
            ann = n["physics_annotation"]
            rules = ann.get("actionable_rules", {})
            if rules:
                lines.append(f"    Physics Rules: {json.dumps(rules, ensure_ascii=False)[:300]}")

    lines.append("\nEdges:")
    for e in edges:
        src = e.get("source") or e.get("from") or e.get("src", "?")
        rel = e.get("relation") or e.get("type") or e.get("label", "->")
        tgt = e.get("target") or e.get("to") or e.get("dst", "?")
        lines.append(f"  {src} --[{rel}]--> {tgt}")
    return "\n".join(lines)


def get_physics_exploration_prompt(item: Dict[str, Any], tool_descriptions: str, difficulty: str = "easy") -> str:
    video_id = item.get("id", "unknown")
    spec = _DIFFICULTY_SPECS.get(difficulty, _DIFFICULTY_SPECS["easy"])

    high_level = item.get("high_level", {})
    omni_annotation = high_level.get("omni_annotation", {}) if isinstance(high_level, dict) else {}
    global_summary = (
        omni_annotation.get("global_summary", "No summary available")
        if isinstance(omni_annotation, dict)
        else "No summary available"
    )

    mid_level = item.get("mid_level", {})
    clips = (
        mid_level.get("clips", [])
        or mid_level.get("vision", {}).get("clips", [])
        if isinstance(mid_level, dict) else []
    )

    # Collect, filter by confidence, then split by estimation method
    video_class = item.get("video_class", "B2")
    _conf_thresh = _CONF_THRESHOLD

    all_physical_obs = _collect_all_obs(item)
    all_physical_obs = _filter_by_confidence(all_physical_obs)
    vlm_items, cv_items = _split_obs_by_method(all_physical_obs, conf_threshold=_conf_thresh)

    # For A-class data, inject GT sensor quantities into the cv_items pool.
    # GT items are the most reliable measurements and are prime candidates to hide.
    gt_physics = item.get("gt_physics", {})
    if gt_physics and isinstance(gt_physics, dict):
        dq = gt_physics.get("derived_quantities", {})
        for field_name, dq_entry in dq.items():
            if not isinstance(dq_entry, dict):
                continue
            val = dq_entry.get("value")
            if val is None:
                continue
            unit = dq_entry.get("unit", "")
            tier = dq_entry.get("reliability_tier", "GT_A")
            # Map field_name to a readable motion entry
            gt_motion: Dict[str, Any] = {
                "_kind": "motion",
                "subject": "mechanic",
                "event": field_name,
                "estimation_method": "sensor_gt",
                "estimation_confidence": float(dq_entry.get("confidence", 0.95)),
                "measurement_tier": "A",
                "gt_source": dq_entry.get("gt_source", "egoexo4d"),
            }
            # Map common GT field names to standard estimated_* keys
            if "speed" in field_name or "velocity" in field_name:
                gt_motion["estimated_speed_ms"] = val
            elif "height" in field_name or "length" in field_name:
                gt_motion["estimated_height_m"] = val
            else:
                gt_motion["gt_value"] = val
                gt_motion["gt_unit"] = unit
            gt_motion = _cv_range(gt_motion)
            cv_items = [gt_motion] + cv_items  # GT items go first (highest confidence)

    # For B-class: VLM observations from _vlm_extract_b_physics are the video-observable
    # given values shown to the LLM in the question. Physical specs (web_search) are hidden.
    # For A-class: VLM observations are fully hidden; CV measurements are the evidence pool.

    # Relabel generic "athlete"/"person" subjects with the context-appropriate role
    # so the LLM uses "mechanic", "gymnast", etc. in the question text.
    _role = _person_role_from_summary(global_summary)

    def _relabel(lst: List[Dict]) -> List[Dict]:
        out = []
        for e in lst:
            lbl = e.get("label") or e.get("subject") or ""
            if lbl.lower() in ("athlete", "person") and _role != "athlete":
                e = dict(e)
                if "label" in e:
                    e["label"] = _role
                if "subject" in e:
                    e["subject"] = _role
            out.append(e)
        return out

    def _strip_internal(lst: List[Dict]) -> List[Dict]:
        """Remove implementation-detail fields from display; rename _source to video_time_segment."""
        result = []
        for e in lst:
            entry = {k: v for k, v in e.items()
                     if k not in ("_range_note", "_kind", "_measurement_range")}
            # Expose clip timing as video_time_segment so LLM knows which segment to cite
            if "_source" in entry:
                entry["video_time_segment"] = entry.pop("_source")
            result.append(entry)
        return result

    def _jdump(lst: List, limit: int = 1200) -> str:
        return json.dumps(_strip_internal(lst), indent=2, ensure_ascii=False)[:limit] if lst else "[]"

    # All CV+GT measurements passed to LLM. LLM selects which to hide per difficulty.
    # For B-class: show VLM observations (video-observable values); CV pool is empty.
    if video_class == "B":
        all_cv = _relabel(vlm_items)
        # Conflict resolution: web_search specs take precedence over VLM for same subject.
        # In practice VLM gives angles/counts and specs give mass/dimensions — rarely overlap.
        all_cv_str = _jdump(all_cv, 2000)
    else:
        all_cv = _relabel(cv_items)
        all_cv_str = _jdump(all_cv, 2000)

    # Web-search physical specs (dimensions + constants) for B2 — hidden from solver
    raw_specs = item.get("web_search_specs", {})
    # For B-class: web_specs_str shows only the un-hidden specs (author context).
    # The hidden_specs are already embedded in difficulty_instructions for B-class.
    # For A-class: all specs are context (full dump).
    if video_class == "B":
        web_specs_str = ""  # B-class specs are handled inside difficulty_instructions
    else:
        web_specs_str = (
            json.dumps(raw_specs, indent=2, ensure_ascii=False)[:1000]
            if raw_specs else ""
        )

    # Strip observable_physical_quantities from clips_desc: metric values are already
    # in all_cv_str. Showing them again causes the LLM to use stale per-clip values
    # instead of the deduplicated canonical measurements.
    clips_stripped = []
    for c in clips:
        c2 = {k: v for k, v in c.items() if k != "observable_physical_quantities"}
        clips_stripped.append(c2)
    clips_desc = json.dumps(clips_stripped, indent=2, ensure_ascii=False)[:4000] if clips_stripped else "[]"
    audio = mid_level.get("audio", {}) if isinstance(mid_level, dict) else {}
    audio_desc = json.dumps(audio, indent=2, ensure_ascii=False)[:2000] if audio else "{}"

    graph_str = _format_graph_for_prompt(item)
    physics_matches_str = _format_physics_matches(item.get("physics_matches", []))

    # Extract the first available clip path for the output template
    _clip_path = next((c.get("clip_path", "") for c in clips if c.get("clip_path")), "")

    n_hide = spec["hidden_cv"]  # 1 for easy, 1 for medium, 2 for hard
    difficulty_label = spec["label"]

    # For B-class, hidden_cv maps to number of web_search specs to hide: easy=1, medium=2, hard=3
    n_hide_specs = n_hide + 1 if video_class == "B" else n_hide

    if video_class == "B":
        # Partition specs: show (n_total - n_hide_specs) in context; hide n_hide_specs
        spec_keys = list(raw_specs.keys())
        shown_spec_keys  = spec_keys[:max(0, len(spec_keys) - n_hide_specs)]
        hidden_spec_keys = spec_keys[max(0, len(spec_keys) - n_hide_specs):]
        shown_specs_str  = (json.dumps(
            {k: raw_specs[k] for k in shown_spec_keys}, indent=2, ensure_ascii=False
        )[:800] if shown_spec_keys else "")
        hidden_specs_str = (json.dumps(
            {k: raw_specs[k] for k in hidden_spec_keys}, indent=2, ensure_ascii=False
        )[:600] if hidden_spec_keys else "")

        if difficulty_label == "EASY":
            difficulty_instructions = (
                "**EASY (B-class)**: The VLM Observations below show all video-observable quantities. "
                "Include them as given values in the question. ONE physical spec is hidden — the solver "
                "must retrieve it via web_search.\n\n"
                "**Given in question**: All VLM-observed values (angles, counts, colors, ratios) "
                "AND the known specs shown in 'Shown physical specs' below.\n\n"
                "**Hidden from solver (1 spec)**: The solver must find these via web_search:\n"
                f"{hidden_specs_str or '(none)'}\n\n"
                "**hidden_cv_chosen**: List exactly 1 string naming the hidden spec, "
                "e.g. `[\"basketball mass = 0.623 kg\"]`.\n\n"
                "**RULE**: Only VLM-observed scale-free quantities (angles, counts, ratios) may appear "
                "as numbers in the question. Absolute metric values (mass, dimensions) are ALWAYS "
                "retrieved via web_search — never stated as numbers.\n\n"
                "**NO secondary derivation**: Do NOT derive one physical quantity from another CV "
                "measurement (e.g. height → arm length via proportion). Each value must come directly "
                "from VLM observation or web_search — never chained."
            )
        elif n_hide_specs == 2:
            difficulty_instructions = (
                "**MEDIUM (B-class)**: The VLM Observations below are all given. TWO physical specs "
                "are hidden — the solver must retrieve BOTH via web_search.\n\n"
                "**Shown specs** (may reference these in question):\n"
                f"{shown_specs_str or '(none)'}\n\n"
                "**Hidden specs (2)** — solver must web_search these:\n"
                f"{hidden_specs_str or '(none)'}\n\n"
                "**hidden_cv_chosen**: List exactly 2 strings, e.g. "
                "`[\"basketball mass\", \"court friction coefficient\"]`.\n\n"
                "**RULE**: Question text may reference shown specs only descriptively "
                "(e.g. 'using standard basketball specifications'). VLM-observed angles/counts "
                "are the only numbers allowed in the question.\n\n"
                "**NO secondary derivation**: Do NOT derive one physical quantity from another "
                "CV measurement via proportion. Each value must come from VLM or web_search."
            )
        else:  # hard, n_hide_specs == 3
            difficulty_instructions = (
                "**HARD (B-class)**: The VLM Observations below are all given. THREE physical specs "
                "are hidden — the solver must retrieve ALL via web_search.\n\n"
                "**No specs shown** — all specifications are hidden:\n"
                f"{hidden_specs_str or '(none)'}\n\n"
                "**hidden_cv_chosen**: List exactly 3 strings naming the hidden specs.\n\n"
                "**RULE**: Question text states ONLY VLM-observed angles/counts/colors/ratios "
                "as numbers. All physical constants and dimensions require web_search.\n\n"
                "**NO secondary derivation**: Do NOT derive one physical quantity from another."
            )
    elif difficulty_label == "EASY":
        difficulty_instructions = (
            "**EASY difficulty**: Include ALL CV measurements as given values directly in "
            "the question text. Do NOT hide any numeric value — the solver receives all "
            "measurements up front. The question tests physics reasoning, NOT video measurement.\n\n"
            "**QUESTION TEXT MUST EXPLICITLY STATE** the measured CV values, e.g.: "
            "'The cue ball moves at an initial angle of 93.6°...', "
            "'The ball-size ratio is 1.07...'\n\n"
            "**PHYSICS COMPLEXITY**: Use a single-step or two-step formula.\n\n"
            "**WHAT GOES IN THE QUESTION** — CV values (video-observed) only:\n"
            "  ✓ Include: angles, speeds, heights, sizes observed from the video\n"
            "  ✗ NEVER include: physical constants (friction coefficient, ball mass,\n"
            "    restitution, dimensions, etc.) — the solver retrieves these via\n"
            "    web_search. Do NOT hardcode μ, m, or any material property in the\n"
            "    question text.\n\n"
            "**hidden_cv_chosen**: Set to an empty list `[]` — nothing is hidden.\n\n"
            "QUESTION TYPE: Choose whichever TYPE (A through E) best fits the physics event "
            "and available evidence. All types are equally valid for EASY difficulty."
        )
    elif n_hide == 1:
        difficulty_instructions = (
            "**MEDIUM difficulty**: From the CV MEASUREMENTS list below, choose exactly "
            "**1 measurement to HIDE** — the solver must observe it from the video. "
            "State all remaining CV values in the question.\n\n"
            "**CRITICAL — HIDDEN VALUE MUST BE FROM CV MEASUREMENTS**: "
            "The hidden value MUST be a numeric value that appears in the CV MEASUREMENTS "
            "list (estimated_height_m, estimated_speed_ms, estimated_angle_deg, gt_value, etc.). "
            "Do NOT hide web-searched physical parameters (mass, friction coefficient, "
            "material property, dimensions from specs) — those are ALWAYS retrieved by the "
            "solver via web_search and are never observable from the video.\n\n"
            "**FORBIDDEN — DO NOT HIDE LENGTH/DISTANCE MEASUREMENTS**: "
            "Do NOT hide length-based measurements such as estimated_height_m, estimated_distance_m, "
            "estimated_radius_m, or any dimensional quantities (arm_length, object_size, etc.). "
            "These are too difficult to observe accurately from video. ONLY hide speed, angle, "
            "or GT sensor values (wrist_speed, elbow_speed) that can be reliably observed or "
            "are provided by high-accuracy sensors.\n\n"
            "**SELECTION RULE**: Choose the hidden quantity to make the most physically "
            "interesting question. Do NOT always pick the highest-confidence item. "
            "Prefer hiding speed or angle measurements rather than length/distance.\n\n"
            "**FORBIDDEN**: The hidden value's number must NOT appear in the question text, "
            "not even approximately.\n\n"
            "**hidden_cv_chosen**: In your JSON output, set `hidden_cv_chosen` to a list "
            "with exactly 1 string describing what you hid, e.g. "
            "`[\"estimated_speed_ms = 4.6 m/s\"]` or `[\"GT wrist_speed = 4.6115 m/s\"]`.\n\n"
            "ANSWER TYPE RULES:\n"
            "  • Answer is INTERVAL if hidden value has measurement uncertainty OR "
            "web_search returns a range. Propagate uncertainty into [lo, hi].\n"
            "  • For categorical questions: compute verdict + margin.\n\n"
            "ROBUSTNESS: verdict must be stable across the full uncertainty range of hidden value.\n\n"
            "**ESSENTIALITY VERIFICATION (CRITICAL)**: Before finalising, mentally double "
            "the hidden CV value. If the answer does not change by ≥20%, the hidden item is "
            "NOT driving the physics — choose a different item to hide, or redesign the "
            "question so the hidden value is a direct numerical input in the formula.\n\n"
            "QUESTION TYPE DIVERSITY — choose any of TYPE_A through TYPE_E."
        )
    else:  # n_hide == 2, hard
        difficulty_instructions = (
            "**HARD difficulty**: From the CV MEASUREMENTS list below, choose exactly "
            "**2 measurements to HIDE** — the solver must observe BOTH from the video. "
            "State NO CV values in the question text.\n\n"
            "**CRITICAL — HIDDEN VALUES MUST BE FROM CV MEASUREMENTS**: "
            "Both hidden values MUST be numeric entries from the CV MEASUREMENTS list "
            "(estimated_height_m, estimated_speed_ms, estimated_angle_deg, gt_value, etc.). "
            "Do NOT hide web-searched physical parameters (mass, friction coefficient, "
            "material properties) — those are ALWAYS retrieved by the solver via web_search.\n\n"
            "**FORBIDDEN — DO NOT HIDE LENGTH/DISTANCE MEASUREMENTS**: "
            "Do NOT hide length-based measurements such as estimated_height_m, estimated_distance_m, "
            "estimated_radius_m, or any dimensional quantities (arm_length, object_size, etc.). "
            "These are too difficult to observe accurately from video. ONLY hide combinations of "
            "speed, angle, or GT sensor values (wrist_speed, elbow_speed) that can be reliably "
            "observed or are provided by high-accuracy sensors.\n\n"
            "**SELECTION RULE**: Choose the 2 hidden quantities so that BOTH are "
            "directly required as numerical inputs in the final formula. "
            "Verify: if either hidden value changed ×2, the answer changes significantly. "
            "Prefer hiding speed + angle, or two different GT speeds.\n\n"
            "**QUESTION TEXT MUST CONTAIN ZERO NUMERIC CV VALUES**: No speeds, heights, "
            "or sizes from the CV list — not even approximately.\n\n"
            "**hidden_cv_chosen**: In your JSON output, set `hidden_cv_chosen` to a list "
            "with exactly 2 strings, e.g. "
            "`[\"estimated_speed_ms = 5.2 m/s\", \"GT wrist_speed = 4.6115 m/s\"]`.\n\n"
            "ANSWER TYPE RULES:\n"
            "  • Default to RANGE answers — two hidden inputs almost always create "
            "observation uncertainty.\n"
            "  • Set answer_range=[lo, hi] whenever the answer is a range.\n\n"
            "HIDDEN CV DISCIPLINE: The CV MEASUREMENTS section shows all values. "
            "Use them ONLY in annotated_solution. NEVER quote them in question text.\n\n"
            "QUESTION TYPE DIVERSITY — choose any of TYPE_A through TYPE_E."
        )

    # B-class scale-free restriction: guide question design for no-calibration videos
    b_class_restriction = ""
    if video_class in ("B", "B2"):
        if difficulty_label == "EASY":
            b_class_restriction = (
                "\n\n**B-CLASS SCALE-FREE (EASY)**:\n"
                "This video has NO pixel-to-metre calibration — no absolute distances or speeds.\n"
                "Scale-free CV measurements ARE available: trajectory angles, deflection angles, "
                "and speed ratios extracted from YOLO tracking (see Observable Physical Quantities below).\n"
                "Use these scale-free CV values as given values in the question text.\n\n"
                "**DO NOT use ball-to-ball size ratio** as a physics input — for same-size "
                "sports (snooker, pool, billiards, tennis) the ratio is a perspective/depth "
                "artifact and does NOT reflect physical size or mass difference.\n\n"
                "**PHYSICAL CONSTANTS** (friction coefficient, ball mass, coefficient of "
                "restitution, etc.) must NOT be given as specific numbers in the question. "
                "The solver must use web_search to retrieve standard values. "
                "WRONG: 'Assume μ = 0.15 for smooth cloth and μ = 0.25 for rough cloth.' "
                "CORRECT: 'Using a typical snooker-cloth rolling friction coefficient found "
                "via web_search, estimate how far the ball travels before stopping.'\n\n"
                "**ANGLE SPECIFICATION (mandatory for every angle in the question)**:\n"
                "  • State the video time segment from the CV source, e.g. '(visible at 8–12 s)'\n"
                "  • Define the coordinate system using the scene_reference from CV data, e.g.\n"
                "    'where 0° is along the horizontal image axis (table long axis), measured "
                "counterclockwise from above' or '0° = pre-collision trajectory direction'"
            )
        else:
            b_class_restriction = (
                "\n\n**B-CLASS SCALE-FREE GUIDANCE (MANDATORY)**:\n"
                "This video has NO pixel-to-metre calibration and NO CV measurements. "
                "The hidden value the solver must observe from the video MUST be "
                "a scale-free observable. "
                "**DO NOT use `clip_visible_duration` as a hidden value** — it is clip-level "
                "tracking duration, not a physical event time, and is unreliable.\n\n"
                "**HIDDEN VALUE MUST BE one of**:\n"
                "  • Deflection angle (degrees) — direction change after collision/bounce (PREFERRED)\n"
                "  • Trajectory angle (degrees) — initial direction of motion\n"
                "  • Phase speed ratio (dimensionless) — second-half/first-half speed ratio\n\n"
                "**DO NOT use ball-to-ball size ratio** as a hidden value — for same-size sports "
                "(snooker, pool, billiards, tennis) it reflects camera perspective/depth, NOT "
                "physical size or mass. It cannot be used to infer mass ratio.\n\n"
                "**ABSOLUTE QUANTITIES (speed, distance, energy) are allowed ONLY as given "
                "assumptions stated in the question text**, e.g.:\n"
                "  ✓ 'Assuming a typical power shot speed of 3.5 m/s, and observing the "
                "deflection angle θ (measure θ from the video), estimate energy transferred.'\n"
                "  ✗ Never ask the solver to derive absolute speed or energy from video alone.\n\n"
                "**ANGLE SPECIFICATION (mandatory for EVERY angle in the question — given AND hidden)**:\n"
                "  • State the video time segment from the CV source, e.g. '(visible at 8–12 s)'\n"
                "  • Define the coordinate system: reference direction + positive rotation sense,\n"
                "    e.g. 'measure the deflection angle θ from the ball's pre-collision "
                "trajectory (0° = no deflection), positive counterclockwise as viewed from above'\n"
                "  • WRONG: '88.6° from horizontal' with no time segment or coordinate definition\n"
                "  • CORRECT: 'initial trajectory angle = 88.6° from the horizontal image axis "
                "(0° = left edge of frame, positive counterclockwise), observed at 8–12 s'\n\n"
                "**PREFERRED question designs for B-class** (in order of quality):\n"
                "  1. Deflection angle geometry: infer collision type / energy transfer from angle change\n"
                "  2. Incidence/reflection at cushion: compare observed vs theoretical 90°-minus rule\n"
                "  3. Initial trajectory angle: decompose velocity vector, estimate component\n"
                "  4. Phase speed ratio: deceleration pattern analysis"
            )

    # B-class universal guidance (appended after difficulty instructions)
    b_class_restriction = ""
    if video_class == "B":
        b_class_restriction = (
            "\n\n**B-CLASS DESIGN PRINCIPLES**:\n"
            "1. VLM observations (angles, counts, colors, ratios) are all given — cite them in the question.\n"
            "2. The question MUST be answerable only if the solver watches the video to understand the scene "
            "context (which object, which collision, which trajectory) even though VLM values are given.\n"
            "3. Physical specs (mass, friction, dimensions) are retrieved via web_search by the solver.\n"
            "4. **NO secondary derivation**: Never chain CV measurements — e.g. do NOT derive arm length "
            "from height, or velocity from angle × another CV value. Each physical quantity comes from "
            "either VLM observation OR web_search directly.\n"
            "5. Angles in the question must specify coordinate system and approximate time segment."
        )

    # Section headers and instructions vary by class and difficulty
    if video_class == "B":
        cv_section_header = "VLM-observed physical quantities (video-observable — all shown to solver):"
        anchor_instruction = (
            "All VLM observations are given in the question as known values. "
            "The solver must retrieve hidden physical specs (listed in difficulty instructions) "
            "via web_search. The question context MUST require watching the video "
            "(the solver needs to identify which object, which event, which trajectory)."
        )
        variable_obfuscation_note = (
            "3. **B-Class: Video-Grounded Observations** — VLM-observed quantities (angles, counts, ratios) "
            "are all given as explicit numbers in the question. Physical specs (mass, friction, dimensions) "
            "are hidden and retrieved via web_search by the solver.\n"
            "   GOOD: 'The ball deflects at θ = 42° after collision. Using standard ball specifications...'\n"
            "   BAD: 'Observe the deflection angle from the video...' (VLM already measured it — state it!)"
        )
    elif difficulty_label == "EASY":
        cv_section_header = "CV-extracted measurements (conf ≥ 0.80) — include ALL of these in the question text:"
        anchor_instruction = (
            "Include ALL CV measurements listed above directly in the question text as explicit given values.\n"
            "The solver receives these numbers up front — they only need to look up physical constants (friction\n"
            "coefficient, ball mass, etc.) via web_search. Do not ask the solver to measure anything from video."
        )
        variable_obfuscation_note = (
            "3. **EASY: No Variable Obfuscation** — All video-observed quantities are given explicitly in the question.\n"
            "   The solver's task is the physics reasoning: look up required constants and compute the answer.\n"
            "   BAD (wrong for EASY): 'The ball moves at θ (measure θ from video)...'\n"
            "   GOOD (correct for EASY): 'The ball moves at θ = 93.6°. Using the ball-cloth friction coefficient...' "
        )
    else:
        cv_section_header = "CV-extracted measurements (conf ≥ 0.80) — choose which to hide per difficulty:"
        anchor_instruction = (
            "Begin by examining the Observable Physical Quantities above. Pick ONE estimated value\n"
            "(angle, speed, height, or object size) to anchor the question to the video. Omit all other\n"
            "numerical values from the question — require the solver to retrieve them via tool calls."
        )
        variable_obfuscation_note = (
            "3. **Variable Obfuscation** (CRITICAL — prevents trivial plug-in questions):\n"
            "   - Keep at most ONE video-observed numerical value explicit in the question.\n"
            "   - All other quantities must be omitted and retrieved via tool calls.\n"
            "   - BAD: \"released at 36°, speed 30 m/s, mass 800g, g=9.8… what is force?\" (all given)\n"
            "   - GOOD: \"released at approximately 36°; using standard men's javelin specifications, what is\n"
            "     the maximum horizontal range?\" (mass, speed, g all require lookup)"
        )

    return f"""You are an expert at constructing high-quality, physically grounded multi-hop QA pairs
for a multimodal benchmark that evaluates world-model and physical reasoning capabilities.

## MANDATORY Requirements for the QA Pair

1. **Physical Reasoning Core**: The question MUST be rooted in a physical phenomenon visible in
   the video. The answer should relate to the physical event observed (e.g., a collision, a sliding
   object, a heat transfer process).

2. **Video-Grounded Question Phrasing** (CRITICAL):
   - The question MUST open with a phrase that references what is observed in the video, e.g.:
     "In the video, an athlete is observed throwing a javelin at approximately X degrees..."
   - Use the **Observable Physical Quantities** section below as the source for these observed values.
   - Select **exactly one** video-estimated quantity to anchor the question to the video.

{variable_obfuscation_note}

4. **No Unsourced Assumptions** (CRITICAL):
   - Every numerical value used in the solution must come from ONE of:
     (a) directly observed in the video (cite the time point),
     (b) retrieved from a web source (cite the URL), or
     (c) computed by code_executor.
   - If a value cannot be sourced this way (e.g. "assumed center-of-mass height = 1.0 m") you
     MUST either (i) look it up via web_search, or (ii) state it explicitly as an assumption IN
     THE QUESTION TEXT — e.g. "...assuming the athlete's center of mass is at 1.0 m height...".
   - Never make silent assumptions in the solution steps.

5. **Web-Searched Values Must NOT Appear in the Question Text** (CRITICAL — A-class and B-class):
   - Physical constants, material properties, and object specifications retrieved via web_search
     (friction coefficient, ball mass, coefficient of restitution, dimensions, standard equipment
     specs, etc.) must NEVER appear as numeric values in the question text.
   - These are for the solver to find independently via web_search.
   - Reference them descriptively in the question: "using the standard friction coefficient for
     snooker baize" or "using typical snooker ball specifications" — never as explicit numbers.
   - Only video-observed quantities (CV measurements, angles, speeds from the video) may be
     stated as numbers in the question.

6. **No Tool Names in Question Text** (CRITICAL):
   - The question text must NEVER mention "web search", "web_search", "Google", "search engine",
     "code_executor", or any specific tool name.
   - WRONG: "...using typical values found via web search..."
   - WRONG: "...retrieve the friction coefficient using web search..."
   - CORRECT: "...using standard steel-on-steel friction coefficient..."
   - CORRECT: "...using typical snooker ball specifications..."
   - The solver knows they have tools available; the question should not prescribe which tools to use.

7. **Hidden CV Must Drive the Final Calculation** (CRITICAL — MEDIUM and HARD):
   - The hidden CV value must appear as a required numerical variable in at least one formula
     in the annotated solution. If you write a solution that never uses the hidden value
     numerically, that hidden CV is decorative — choose a different CV to hide or redesign
     the question.
   - WRONG: hidden_cv_chosen = ["athlete height = 1.74 m"] but the solution formula is
     `W = μ × m × g × d` and height never enters the calculation.
   - WRONG: the question asks for friction work; hidden value is height; height is not
     needed to compute W — the answer is unchanged regardless of height.
   - CORRECT: hidden value is `d` (distance from CV track) and solution is `W = μmgd`.
   - CORRECT: hidden value is deflection angle φ; solution uses `KE_ratio = cos²(φ)`.
   - ESSENTIALITY TEST (mandatory): mentally double the hidden CV value. If the final
     answer does not change by ≥20%, the hidden CV is NOT driving the physics. Fix it
     before submitting.

5. **Answer Consistency with Retrieved Data**:
   - If a web search returns a RANGE (e.g. "release speed 25–30 m/s"), DO NOT silently pick the
     midpoint. Either frame the question so the answer is a range, or find a more precise source.
   - **CV measurement fields** (`_measurement_range`, `measurement_tier`): These are internal
     calibration metadata for the dataset author. Do NOT propagate these ranges into the answer
     unless the question explicitly asks "what range…". The question is authored knowing the
     central CV value; use that central value in the computation.
   - **CRITICAL**: Every numerical value stated in the question text MUST be the SAME value used
     in the annotated_solution steps. Never state speed X in the question and then use speed Y in
     the solution — this makes the answer wrong. Before finalizing, scan every number in the
     question and verify it appears in exactly the same form in the solution.

6. **Question Type — choose one of the following four types**:
   - **[TYPE A] Quantitative Calculation**: Compute a physical quantity from observed + looked-up
     values. (e.g. "what is the maximum height reached?")
   - **[TYPE B] State Prediction**: Given observations from the first N seconds of the video,
     predict the physical state at a later moment — trajectory, landing point, velocity at impact,
     whether an object clears an obstacle, etc. Frame as "Based on the first X seconds of the
     video showing [observed detail], predict [what happens next]."
   - **[TYPE C] Causal / Constraint Explanation**: Ask why a physical outcome is or is not
     possible given the observed setup. (e.g. "Could the athlete have cleared a hurdle 10 cm
     higher given the observed takeoff angle? Justify using physics.")
   - **[TYPE D] Comparative Physics**: Compare two scenarios — one observed in the video and one
     hypothetical — using physics. (e.g. "If the javelin were released at 45° instead of the
     observed ~36°, by how much would the range change?")
   Pick the type that best fits the video content. Use TYPE A only if the other types are not
   applicable; prefer TYPE B, C, or D for more interesting questions.

7. **External Tool Dependency**: Must call at least one tool; answer must not be directly
   readable from the video.

8. **Deterministic Answer**: Specific number with units, a range with units, or an unambiguous
   short phrase. Must match the question type.

9. **Physics Derivation Correctness** (CRITICAL — prevents wrong answers):
   - Speed values in Observable Physical Quantities are **total speeds** (magnitude of the
     velocity vector), NOT horizontal or vertical components. Use them directly as v₀.
   - In projectile motion: v_x = v₀·cos(θ), v_y = v₀·sin(θ). NEVER derive v₀ from a
     horizontal speed using v₀ = v_x/cos(θ) — this is physically wrong for approach speeds
     and produces unrealistically large velocities.
   - Always use code_executor to verify your numerical result. If an intermediate value seems
     physically unreasonable (e.g. v_y > 10 m/s for a human jump, or h_max > 3 m), re-examine
     your formula before proceeding.
   - For human motion, sanity-check against known limits: sprint speed 8–12 m/s, jump height
     0.3–1.0 m, takeoff angle for hurdles 15–25° (not 65–75°).

10. **CV over VLM priority** (CRITICAL):
   - The Observable Physical Quantities section below contains two types of data:
     **CV-extracted** (from pose estimation or YOLO tracking — instrumentally measured) and
     **VLM-estimated** (from visual estimation — subjective and less reliable).
   - When a CV-extracted value and a VLM-estimated value exist for the **same physical quantity**
     (e.g. both give an athlete speed), **always use the CV-extracted value** in your question
     and solution. Ignore the VLM estimate for that quantity.
   - CV-extracted quantities have `estimation_method` = `"pose_estimation"` or `"yolo_trajectory"`.
   - VLM-estimated quantities have `estimation_method` = `"vlm"`.
   - When **multiple CV speed estimates** exist for the same subject, prefer the one with
     event=`"running_speed"` (from YOLO bbox tracking, more reliable in camera-panning shots)
     over event=`"movement"` (from pose-based hip tracking, may underestimate in tracking shots).

## Output Format
When done, output the QA pair in this JSON wrapped in ```json ... ```:
```json
{{
  "multi_hop_qa_pair": {{
    "question_type": "TYPE_A|TYPE_B|TYPE_C|TYPE_D|TYPE_E",
    "difficulty": "{spec['label']}",
    "hidden_cv_chosen": ["description of what was hidden, e.g. 'athlete height = 1.74 m'"],
    "exploration_summary": "Narrative: (1) physical event observed, (2) video-observed value used, (3) latent graph properties, (4) tools called and results, (5) how values combined to reach the answer. Explicitly note if any value was assumed and how it appears in the question.",
    "question": "In the video, [observed detail with one numerical anchor]. [Type-specific framing — prediction/causal/comparative/calculation]. [State any explicit assumptions inline if unavoidable].",
    "physics_category": "Domain/EventType (e.g. Mechanics/Projectile_Motion)",
    "omni_modal_input": [
      {{"type": "video", "id": "{video_id}", "path": "{_clip_path}"}}
    ],
    "annotated_solution": [
      "Step 1: Observe the video — [cite the physical event and the one numerical value used].",
      "Step 2: Identify the physical phenomenon from the event graph.",
      "Step 3: Use web_search(query='...') to retrieve [specific external value, not an unsourced assumption].",
      "Step 4: Use code_executor to compute [quantity] using [formula].",
      "Step N: Result is [answer with units]."
    ],
    "sources": [
      {{"type": "video_observation", "time_point": "Xs", "key_information": "observed value"}},
      {{"type": "web_search", "url": "...", "key_information": "retrieved value (exact, not midpoint of range unless range is the answer)"}},
      {{"type": "code_executor", "code": "...", "key_information": "computed result"}}
    ],
    "answer": "Gold answer. Single value (e.g. '0.13 m') when all inputs are point values. Range answer (e.g. '1.2–1.5 m' or 'Yes — clearance 0.18 m (range 0.12–0.24 m)') when any key input is a range. For categorical/proof: verdict + computed quantity — NEVER just 'yes' or 'no'.",
    "answer_range": "Fill when answer is an interval: [lower_bound, upper_bound] same units as answer. Any evaluator answer within this range is scored correct. Set null when answer is a single deterministic value.",
    "answerable_without_video": "no",
    "answerable_without_audio": "yes|no",
    "answerable_without_tools": "no",
    "total_steps": 0,
    "required_external_tools": ["tools actually used"],
    "task_type": "video w/ audio|video only",
    "category": "Physics"
  }}
}}
```

## Tool Usage Guidelines (use at least one)

### `code_executor` — Physics / math calculation
Run Python code to compute a numerical answer using physics formulas with the observed values.

### `web_search` — External fact retrieval
Search for physical constants, material properties, or coefficients NOT visible in the video.

### `page_browser` — Deep web content access
When search snippets are insufficient, retrieve full page content.

### `visual_question_answering` — Confirm fine-grained video details
Use when an observed quantity needs confirmation (e.g. exact angle, object identity).

---

## Current Video Information

**Video ID:** {video_id}

**Video Summary (authoritative overall description — takes precedence over individual segment descriptions when they conflict):**
{global_summary}

## Difficulty: {spec['label']}

{difficulty_instructions}{b_class_restriction}

**{cv_section_header}**
{all_cv_str}

**IMPORTANT RULES FOR USING CV MEASUREMENTS:**
1. **No tool names in question text**: The question text must NEVER mention "web search", "search", "web_search", "code executor", "API", or any tool name. These are solver activities. The question text should describe observable physics only.
2. **Time references**: When citing when a measurement was observed, use the time range (e.g., "visible at 8.0–12.0 s") — NOT internal labels like "clip 2" or "clip_2".
3. **Angle coordinate systems**: When using a PCA-based scene reference (e.g., "long axis of the snooker/pool table"), describe it in terms of VISIBLE physical objects in the scene, e.g., "0° = along the long edge of the snooker table, measured counterclockwise from above". Do NOT use algorithm names like "PCA axis" or "stationary-objects PCA" in the question text.
4. **GT wrist/elbow speeds**: These are MAXIMUM speeds across ALL GT frames (not necessarily from the action described in the question). Check the `max_at_video_time_s` field — if this time falls outside the clip's time range, the value likely comes from a DIFFERENT action. If uncertain, do NOT use the GT speed as a given CV value; design a question around a CV measurement that is clearly attributable to the relevant action.
5. **Unreliable measurements**: Any CV item whose `estimation_basis` contains "CAUTION: possible_id_switch" is potentially wrong (YOLO tracker switched objects). Do NOT use these as CV inputs or as hidden values.
6. **Range propagation (CRITICAL)**: When a web_search returns a RANGE for any required parameter (e.g., body weight 56–75 kg), you MUST propagate the full range through the formula to produce an interval answer [lo, hi]. Do NOT use the midpoint and report a single value.

**Known physical specifications (dimensions + constants, background context only):**
{web_specs_str}

**CRITICAL — SPECIFICATIONS REQUIRE WEB_SEARCH:**
The specs above are background context for you (the question author). Do NOT pass them
directly to the solver. In annotated_solution, every physical specification (mass,
dimensions, coefficients) MUST be retrieved via a `web_search` step with a real query
and a real URL. NEVER use `"type": "known_specification"` or `"source": "provided dataset"`
in your sources array — replace with actual web_search citations. The solver is assumed
to have NO pre-loaded physical specs.

**Video Content Segments:**
{clips_desc}

**Audio Information:**
{audio_desc}

**Physics-Expanded Event Graph:**
{graph_str}

**Physics Knowledge Matches (use these to guide your question):**
{physics_matches_str}

---

{anchor_instruction}

Choose the most interesting question TYPE (prefer B/C/D over A). If TYPE B (prediction), frame
the question as "Based on the first X seconds showing [detail], predict [future state]."

For every value used in the solution: it must come from (a) video, (b) web search with URL, or
(c) code computation. If a value must be assumed, state it explicitly in the question text.
If a search returns a range, either ask for a range answer or search for a more precise source.

**Answer type — determined ENTIRELY by the input nodes, NOT by difficulty level:**

The answer is a RANGE if and only if any required input is itself a range:
- A web search returns a range for a key parameter (e.g. hurdler mass 75–88 kg, drag coefficient 0.35–0.45)
- A CV measurement enters the computation AND it carries a `_measurement_range`
  (e.g. speed = 4.7 m/s with range [4.0, 5.4] — if this speed is HIDDEN and the solver must observe it,
  or if the question explicitly asks about the range, then the answer is a range)
- The question explicitly asks for a range (e.g. "what plausible range of heights…")

In those cases: propagate the input range through the physics formula, set answer_range=[lo, hi].
Any answer within [lo, hi] is scored correct.

The answer is a SINGLE VALUE when all required inputs are fully specified as point values.
This rule is completely independent of easy/medium/hard difficulty.

**When is a CV `_measurement_range` used?**
- CV value is stated explicitly in the question as "approximately X" → treat as POINT VALUE (the author
  chose the central value; acknowledge as approximate but answer uses that point)
- CV value is HIDDEN (solver must observe from video) OR question explicitly asks "what range…"
  → propagate the _measurement_range through the formula → RANGE answer, set answer_range=[lo, hi]

**Robustness filter for categorical/proof questions:**
Compute the signed decision margin at the central value. If margin stays the same sign across the full
_measurement_range of all uncertain inputs → verdict is stable → question is valid.
If the sign flips → switch to asking for the range of the decision variable instead of a binary verdict.

**Question type diversity targets:**
~50–70% numeric computation (single value or range, as the input nodes decide), ~20–25% categorical/proof with
quantitative justification, ~10–20% comparison/ranking/consistency-check.
Vary: direct vs inverse; kinematics vs energy vs momentum; solve-for-output vs verify-a-claim."""


# ---------------------------------------------------------------------------
# QA extraction
# ---------------------------------------------------------------------------

def extract_qa_pairs(text: str) -> List[Dict]:
    qa_pairs = []
    for match in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL):
        try:
            data = json.loads(match)
            if "multi_hop_qa_pair" in data:
                qa = data["multi_hop_qa_pair"]
                if isinstance(qa, dict) and qa:
                    qa_pairs.append(qa)
        except json.JSONDecodeError:
            continue
    return [qa_pairs[-1]] if qa_pairs else []


def messages_to_dict(messages: List[Any]) -> List[Dict[str, Any]]:
    result = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            result.append(msg.model_dump())
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append({"content": str(msg)})
    return result


# ---------------------------------------------------------------------------
# Agentic generation loop
# ---------------------------------------------------------------------------

async def generate_physics_qa(
    client: AsyncOpenAI,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    difficulty: str = "easy",
    extra_instructions: str = "",
) -> Dict[str, Any]:
    """one agentic QA construction run; extra_instructions (grounding rules, feedback on a rejected attempt) are
    appended to the exploration prompt"""
    video_id = item.get("id", "unknown")

    # Skip items with no physics matches — no physical QA can be constructed
    physics_matches = item.get("physics_matches", [])
    if not physics_matches:
        logger.info(f"[{video_id}] No physics matches, skipping.")
        return {
            "video_id": video_id,
            "qa_pairs": [],
            "action_count": 0,
            "error": "no_physics_matches",
        }

    tools = get_all_tool_definitions()
    tool_descriptions = json.dumps(tools, indent=2)
    user_prompt = get_physics_exploration_prompt(item, tool_descriptions, difficulty=difficulty)
    if extra_instructions:
        user_prompt += "\n" + extra_instructions

    messages = [{"role": "user", "content": user_prompt}]
    result = {
        "video_id": video_id,
        "messages": [],
        "qa_pairs": [],
        "action_count": 0,
        "error": None,
        "physics_category": physics_matches[0].get("matched_category", "") if physics_matches else "",
    }

    action_count = 0
    full_output = ""

    async with semaphore:
        while True:
            if action_count > max_actions + 5:
                break
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=3600,
                )
            except Exception as e:
                logger.error(f"[{video_id}] LLM call failed: {e}")
                result["error"] = str(e)
                break

            if not response or not response.choices:
                logger.error(f"[{video_id}] Empty API response (choices is None/empty)")
                result["error"] = "Empty API response"
                break

            message = response.choices[0].message
            messages.append(message)
            content = message.content or ""
            full_output += content

            if message.tool_calls:
                action_count += len(message.tool_calls)
                tool_results = []
                for tc in message.tool_calls:
                    func_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    logger.info(f"[{video_id}] Tool: {func_name}")
                    tool_result = await execute_tool(func_name, arguments, video_id)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                messages.extend(tool_results)

                if action_count >= max_actions:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have used {action_count} tool calls. "
                            "Please now finalize and output your physics QA pair in the required JSON format."
                        ),
                    })
            else:
                qa_pairs = extract_qa_pairs(full_output)
                if qa_pairs:
                    result["qa_pairs"] = qa_pairs
                    break
                if action_count == 0:
                    break
                messages.append({
                    "role": "user",
                    "content": "Please output your final physics QA pair in the required JSON format wrapped in ```json ... ```.",
                })
                action_count += 1
                if action_count > max_actions + 5:
                    break

    result["action_count"] = action_count
    result["messages"] = messages_to_dict(messages)

    if not result["qa_pairs"]:
        result["qa_pairs"] = extract_qa_pairs(full_output)

    return result


async def generate_physics_qa_tiered(
    client: AsyncOpenAI,
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """Generate up to one QA pair per eligible difficulty tier (easy/medium/hard).

    Anti-collapse: when multiple physics matches exist, rotate through distinct categories
    so each difficulty tier is encouraged to use a different physics category.
    """
    video_id = item.get("id", "unknown")
    eligible = _eligible_difficulties(item)
    logger.info(f"[{video_id}] Eligible difficulty tiers: {eligible}")

    all_qa: List[Dict] = []
    total_actions = 0

    physics_matches = item.get("physics_matches", [])
    # Build a per-tier item with a distinct primary physics match when possible
    # Rotate: easy→match[0], medium→match[1] (if available), hard→match[2] (if available)
    # This distributes distinct physics categories across difficulty tiers.
    used_categories: set = set()
    failed: Dict[str, Dict[str, Any]] = {}   # tier -> stated-given candidate
    log: Dict[str, Any] = {}

    for tier_idx, tier in enumerate(eligible):
        spec = _DIFFICULTY_SPECS[tier]

        # Select the physics match for this tier: prefer one with a new category
        tier_item = item
        if len(physics_matches) > 1:
            preferred = None
            for m in physics_matches:
                cat = m.get("matched_category", "")
                if cat not in used_categories:
                    preferred = m
                    break
            if preferred is not None:
                # Rebuild item with this match first
                remaining = [m for m in physics_matches if m is not preferred]
                tier_item = dict(item)
                tier_item["physics_matches"] = [preferred] + remaining
                used_categories.add(preferred.get("matched_category", ""))
            else:
                used_categories.add(physics_matches[0].get("matched_category", ""))
        else:
            if physics_matches:
                used_categories.add(physics_matches[0].get("matched_category", ""))

        logger.info(f"[{video_id}] Generating {spec['label']} QA …")
        g = await _grounded_tier(client, tier_item, semaphore, model_name, tier,
                                 temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        total_actions += g["action_count"]
        log[tier] = g["log"]
        if g["qa"] is None:
            logger.warning(f"[{video_id}] {spec['label']}: no QA passed the Stage-3 checks in {g['attempts']} attempts")
            if g["given"]:
                failed[tier] = g["given"]
            continue
        all_qa.append(g["qa"])

    # Tiers whose QA failed only on the uniqueness of some web constants: state those constants as givens
    for qa in await grounding.convert_givens(failed, [q["difficulty"] for q in all_qa], "A"):
        logger.info(f"[{video_id}] stated-given QA added ({qa['difficulty']})")
        all_qa.append(qa)

    return {
        "video_id": video_id,
        "qa_pairs": all_qa,
        "action_count": total_actions,
        "error": None,
        "physics_category": physics_matches[0].get("matched_category", "") if physics_matches else "",
        "stage3_log": log,
    }


async def _grounded_tier(client, tier_item, semaphore, model_name, tier, avoid: str = "",
                         temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,
                         max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    """one tier under the Stage-3 checks (grounding.grounded_generate)"""
    video_id = tier_item.get("id", "unknown")

    async def gen(extra: str) -> Dict[str, Any]:
        try:
            r = await generate_physics_qa(client, tier_item, semaphore, model_name, temperature=temperature, top_p=top_p,
                                          max_tokens=max_tokens, max_actions=_DIFFICULTY_SPECS[tier]["max_actions"],
                                          difficulty=tier, extra_instructions=extra)
        except Exception as e:
            logger.error(f"[{video_id}] {tier} QA generation failed: {e}")
            return {"qa_pairs": [], "action_count": 0, "error": str(e)}
        for qa in r.get("qa_pairs", []):
            if isinstance(qa, dict):
                qa["difficulty"] = tier
        return r

    return await grounding.grounded_generate(gen, tier, "A", tier_item.get("video_path", ""), avoid=avoid)


async def regenerate_one(client, item: Dict[str, Any], semaphore, model_name: str, tier: str, avoid: str = "",
                         siblings: List[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """a new QA of one tier under the Stage-3 checks (the rewrite of a QA that failed verification)"""
    if siblings:
        avoid += ("\n## OTHER QAs OF THIS VIDEO (write a different one)\n"
                  + "\n".join("- " + str(q.get("question"))[:300] for q in siblings) + "\n")
    g = await _grounded_tier(client, item, semaphore, model_name, tier, avoid=avoid)
    return g["qa"]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

async def process_batch(
    client: AsyncOpenAI,
    items: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_actions: int,
    output_file: str,
    checkpoint_interval: int = 50,
) -> List[Dict[str, Any]]:
    all_results = []
    tasks = [
        generate_physics_qa(client, item, semaphore, model_name, temperature, top_p, max_tokens, max_actions)
        for item in items
    ]

    for i, future in enumerate(
        tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating Physics QA")
    ):
        try:
            res = await future
            all_results.append(res)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            all_results.append({"error": str(e)})

        if (i + 1) % checkpoint_interval == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved: {i+1} items.")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Step 3 (Physics): Agentic Physics QA Construction")
    parser.add_argument("--input", type=str, required=True,
                        help="Physics-expanded event graphs (from expand_event_graph_with_physics.py)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--video_index_files", type=str, nargs="+", default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT_LIMIT)
    parser.add_argument("--max_actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)

    args = parser.parse_args()

    if args.output is None:
        base = args.input.replace(".json", "")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"{base}_physics_qa_{timestamp}.json"

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    if not isinstance(input_data, list):
        input_data = [input_data]

    input_data = input_data[args.start_index :]
    if args.max_items is not None:
        input_data = input_data[: args.max_items]

    logger.info(f"Loaded {len(input_data)} physics-expanded graphs.")

    index_files = args.video_index_files if args.video_index_files else [args.input]
    logger.info(f"Building video search index from: {index_files}")
    build_video_index(index_files)

    client = AsyncOpenAI(base_url=args.api_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrent)

    try:
        all_results = await process_batch(
            client=client,
            items=input_data,
            semaphore=semaphore,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_actions=args.max_actions,
            output_file=args.output,
        )
    finally:
        await client.close()
        save_web_cache()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total = len(all_results)
    success = sum(1 for r in all_results if r.get("qa_pairs"))
    skipped = sum(1 for r in all_results if r.get("error") == "no_physics_matches")
    logger.info(
        f"Done. {success}/{total} generated QA pairs, {skipped} skipped (no physics match). "
        f"Saved to {args.output}"
    )


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
