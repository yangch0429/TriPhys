"""
aclass_pipeline_core.py  —  per-video construction pipeline + per-folder runner

Stages: 1 extract → 2 event graph → (KPR: 2b class) → 2.5 CV physics / physics library / graph expansion →
3 grounded physics QA (grounding.py) → 4 quality check → 4b verification and bounded repair (verify_qa.py,
repair_qa.py) → 5 filter (verified + quality gates). split_videos.py then splits the kept QAs into train / test.

For each subfolder under Videos/, picks one video at random and runs the full pipeline.

All model calls go through OpenRouter.
Results are saved to ./test_videos_output/<folder_name>/.

Run:
    python aclass_pipeline_core.py
    python aclass_pipeline_core.py --videos_dir ./Videos --out_dir ./test_videos_output
    python aclass_pipeline_core.py --folders archery basketball  # specific folders only
"""

import os
import sys
# ALL_PROXY=socks5://... causes httpx (used by openai SDK) to require socksio.
# Remove it at startup so all OpenAI clients fall back to HTTP_PROXY instead.
for _k in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

import json
import glob
import random
import asyncio
import logging
import time
import argparse
import shutil
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Logging setup (before any imports that also log)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_videos")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR   = os.path.join(SCRIPT_DIR, "Videos")
OUT_DIR      = os.path.join(SCRIPT_DIR, "test_videos_output")
OUT_DIR_V2_B = os.path.join(SCRIPT_DIR, "benchmark_v2_B")

# Endpoints, keys and model ids are read from the environment (see README.md).
# VLM (Gemini) via an OpenAI-compatible endpoint (default OpenRouter).
VLM_BASE_URL = os.environ.get("CHAT_API_BASE", "https://openrouter.ai/api/v1")
VLM_API_KEY  = os.environ.get("VLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
# LLM (DeepSeek-V3.2), default through OpenRouter (model "deepseek/deepseek-v3.2").
# Set DEEPSEEK_BASE (e.g. https://api.deepseek.com) + DEEPSEEK_API_KEY + DEEPSEEK_MODEL to use another endpoint.
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE", "https://openrouter.ai/api/v1")
LLM_API_KEY  = (os.environ.get("OPENROUTER_API_KEY", "") if "openrouter.ai" in LLM_BASE_URL
                else os.environ.get("DEEPSEEK_API_KEY", ""))

# Gemini for vision; DeepSeek V3.2 for reasoning
VLM_MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-3.6-flash")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v3.2")

# ---------------------------------------------------------------------------
# FFmpeg injection (must happen before any imports that call ffmpeg)
# ---------------------------------------------------------------------------

_local_ffmpeg = os.path.join(SCRIPT_DIR, "_ffmpeg", "ffmpeg.exe")
FFMPEG = shutil.which("ffmpeg") or (_local_ffmpeg if os.path.exists(_local_ffmpeg) else None)
if FFMPEG:
    _ffmpeg_dir = os.path.dirname(os.path.abspath(FFMPEG))
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_PATH"] = FFMPEG
else:
    logger.error("FFmpeg not found. Run run_pipeline.py first (auto-downloads) or install manually.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str):
    logger.info("=" * 60)
    logger.info(f"  {msg}")
    logger.info("=" * 60)


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved → {path}  ({os.path.getsize(path) // 1024} KB)")


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pick_random_videos(folder: str, n: int = 1) -> List[str]:
    """Return up to *n* distinct random .mp4 files from *folder*."""
    mp4s = glob.glob(os.path.join(folder, "*.mp4"))
    if not mp4s:
        return []
    return random.sample(mp4s, min(n, len(mp4s)))


def collect_folders(videos_dir: str, filter_names: Optional[List[str]] = None) -> List[str]:
    """Return sorted list of sub-folder paths under videos_dir."""
    folders = sorted([
        d for d in glob.glob(os.path.join(videos_dir, "*"))
        if os.path.isdir(d)
    ])
    if filter_names:
        folders = [f for f in folders if os.path.basename(f) in filter_names]
    return folders


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

async def stage1_extract(video_path: str, out_dir: str, video_class: str = "B1") -> Dict[str, Any]:
    """Stage 1: Extract video info using extract_video_info.py (OpenRouter Gemini)."""
    from extract_video_info import extract_video_info
    return await extract_video_info(
        video_path=video_path,
        out_dir=out_dir,
        concurrency=4,
        video_class=video_class,
    )


async def stage2_graph(
    omni_info: Dict[str, Any],
    out_dir: str,
    cv_physics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stage 2: Build event graph (DeepSeek via OpenRouter)."""
    from openai import AsyncOpenAI
    from build_event_graph import construct_event_graph

    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem    = asyncio.Semaphore(1)
    result = await construct_event_graph(client, sem, omni_info, LLM_MODEL, cv_physics=cv_physics)
    await client.close()
    return result


async def stage25a_library(out_dir: str) -> Dict[str, Any]:
    """Stage 2.5a: Build Physics Event Library (DeepSeek via OpenRouter)."""
    from openai import AsyncOpenAI
    from build_physical_event_library import generate_entry, PHYSICS_ONTOLOGY

    client  = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem     = asyncio.Semaphore(4)
    tasks   = [
        generate_entry(client, sem, domain, event_type, LLM_MODEL)
        for domain, events in PHYSICS_ONTOLOGY.items()
        for event_type in events
    ]
    entries = await asyncio.gather(*tasks)
    await client.close()

    library = {
        "version":       "1.0",
        "total_entries": len([e for e in entries if "_error" not in e]),
        "entries":       [e for e in entries if "_error" not in e],
    }
    logger.info(f"  Physics library: {library['total_entries']} entries")
    return library


async def stage25b_expand(
    graph: Dict[str, Any],
    library: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage 2.5b: Expand event graph with physics nodes (DeepSeek via OpenRouter)."""
    from openai import AsyncOpenAI
    from expand_event_graph_with_physics import expand_one_graph

    client  = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem     = asyncio.Semaphore(1)
    entries = library.get("entries", [])
    result  = await expand_one_graph(client, sem, graph, entries, LLM_MODEL)
    await client.close()
    return result


async def stage3_qa(
    expanded_graph: Dict[str, Any],
    omni_info_path: str,
) -> Dict[str, Any]:
    """Stage 3: Agentic physics QA generation (DeepSeek via OpenRouter)."""
    from openai import AsyncOpenAI
    # MPR (A) and KPR (B) use separate Stage-3 author agents: dispatch on the video class set in
    # run_pipeline_for_video. Both write every tier under the Stage-3 checks of grounding.py.
    if str(expanded_graph.get("video_class", "")).startswith("B"):
        from agentic_physics_qa_kpr import generate_physics_qa_tiered
    else:
        from agentic_physics_qa import generate_physics_qa_tiered
    from tools.video_search import build_video_index

    if os.path.exists(omni_info_path):
        build_video_index([omni_info_path])

    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem    = asyncio.Semaphore(1)
    result = await generate_physics_qa_tiered(
        client,
        expanded_graph,
        sem,
        model_name=LLM_MODEL,
    )
    await client.close()
    return result


async def stage4_check(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 4: Quality-check each QA pair (OpenRouter DeepSeek)."""
    from openai import AsyncOpenAI
    from check_qa_quality import check_single_item

    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem    = asyncio.Semaphore(1)
    checked = []
    for qa in qa_pairs:
        result = await check_single_item(client, sem, qa, LLM_MODEL)
        # check_single_item returns item.get("multi_hop_qa_pair", item) merged with eval fields
        checked.append(result.get("multi_hop_qa_pair", result))
    await client.close()
    return checked


def _author_module(video_class: str):
    if str(video_class).startswith("B"):
        import agentic_physics_qa_kpr as author
    else:
        import agentic_physics_qa as author
    return author


async def stage4b_verify(qa_pairs: List[Dict[str, Any]], expanded_graph: Dict[str, Any],
                         max_rounds: int = 3) -> List[Dict[str, Any]]:
    """Stage 4b: verify every QA (code, grounding / leaks / form, video consistency, unique web values, physical
    model, cross-question consistency, quality gates, strict audit); a failing QA gets up to max_rounds repair
    rounds (smallest fix, then a rewrite by the Stage-3 author agent), re-verified after each; still failing ->
    dropped. Every returned QA carries verification.status = verified | dropped."""
    from openai import AsyncOpenAI
    from check_qa_quality import check_single_item
    from verify_qa import verify_video

    cls = "B" if str(expanded_graph.get("video_class", "")).startswith("B") else "A"
    author = _author_module(cls)
    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem    = asyncio.Semaphore(1)

    async def quality_fn(qa: Dict[str, Any]) -> Dict[str, Any]:
        r = await check_single_item(client, sem, qa, LLM_MODEL)
        return r.get("multi_hop_qa_pair", r)

    async def regen(tier: str, avoid: str, siblings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return await author.regenerate_one(client, expanded_graph, sem, LLM_MODEL, tier, avoid, siblings)

    try:
        return await verify_video(qa_pairs, expanded_graph.get("video_path", ""), cls,
                                  quality_fn=quality_fn, regen=regen, max_rounds=max_rounds)
    finally:
        await client.close()


def stage5_filter(checked_qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stage 5: Keep only QA pairs that Stage 4b verified and that pass all quality gates (no API calls)."""
    from filter_valid_qa import is_valid_item

    valid = []
    for qa in checked_qa_pairs:
        if (qa.get("verification") or {}).get("status") != "verified":
            continue
        # primary_modality="video" skips the stricter visual/audio-only checks in is_valid_item
        if is_valid_item({"multi_hop_qa_pair": qa, "primary_modality": "video"}):
            valid.append(qa)
    return valid


def cleanup_qa_for_export(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove verbose internal fields (no longer needed since Stage 3 doesn't generate them)."""
    # Stage 3 no longer generates exploration_summary or omni_modal_input, but keep this
    # function for backwards compatibility with old cached stage3/stage4 files
    cleaned = []
    for qa in qa_pairs:
        qa_clean = dict(qa)
        qa_clean.pop("exploration_summary", None)
        # Remove omni_modal_input entirely (video_id is sufficient)
        qa_clean.pop("omni_modal_input", None)
        cleaned.append(qa_clean)
    return cleaned



# ---------------------------------------------------------------------------
# Per-video pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline_for_video(
    video_path: str,
    out_dir: str,
    shared_library: Optional[Dict[str, Any]] = None,
    force: bool = False,
    skip_check: bool = False,
    video_class: str = "B1",
    dataset_class: str = "B",
    take_uid: Optional[str] = None,
    gt_annotation_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all pipeline stages for a single video.
    dataset_class: "A" (EgoExo4D with ego_pose GT) or "B" (no GT).
    For B-class, video_class (B1/B2) is auto-determined at Stage 1.5 (after entity extraction, before event graph).
    Stages: extract → event graph → [B: classify B1/B2] → physics library →
            graph expansion → physics QA → quality check → filter.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "s1":       os.path.join(out_dir, "stage1_omni_info.json"),
        "s1_gt":    os.path.join(out_dir, "stage1_gt_physics.json"),
        "s2":       os.path.join(out_dir, "stage2_event_graph.json"),
        "s2b_class": os.path.join(out_dir, "stage2b_class.json"),
        "s25cv":    os.path.join(out_dir, "stage25_cv_physics.json"),
        "s25a":     os.path.join(out_dir, "physics_event_library.json"),
        "s25b":     os.path.join(out_dir, "stage2b_physics_expanded.json"),
        "s3":       os.path.join(out_dir, "stage3_physics_qa.json"),
        "s4":       os.path.join(out_dir, "stage4_qa_checked.json"),
        "s4b":      os.path.join(out_dir, "stage4b_verified.json"),
    }

    summary = {"video": video_path, "out_dir": out_dir, "vlm_failures": {}}
    t0 = time.time()

    # ---- Force: wipe all caches ----
    if force:
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
                logger.info(f"  [Force] Deleted cache {os.path.basename(p)}")

    # ---- Stage 1 ----
    s1_cache_valid = False
    if os.path.exists(paths["s1"]):
        omni_info = load_json(paths["s1"])
        if isinstance(omni_info, list):
            omni_info = omni_info[0]
        # Validate: cached data must be from the same video
        cached_path = omni_info.get("video_path", "")
        if os.path.normcase(os.path.abspath(cached_path)) == os.path.normcase(os.path.abspath(video_path)):
            s1_cache_valid = True
            logger.info(f"  [Stage 1] Cache hit → {paths['s1']}")
        else:
            logger.info(f"  [Stage 1] Cache stale (different video) — re-extracting …")
            # Invalidate all downstream caches
            for p in [paths["s2"], paths["s25b"], paths["s3"], paths["s4"], paths["s4b"]]:
                if os.path.exists(p):
                    os.remove(p)
                    logger.info(f"  [Cache] Invalidated {os.path.basename(p)}")

    if not s1_cache_valid:
        logger.info(f"  [Stage 1] Extracting video info …")
        omni_info = await stage1_extract(video_path, out_dir, video_class=video_class)
        save_json([omni_info], paths["s1"])

    clips = omni_info.get("mid_level", {}).get("clips", [])
    summary["stage1_clips"] = len(clips)

    # Count Stage 1 VLM failures (clips with error field)
    s1_failed_clips = sum(1 for c in clips if c.get("error") or
                          c.get("observable_physical_quantities", {}).get("error"))
    if s1_failed_clips:
        summary["vlm_failures"]["stage1_clips_failed"] = s1_failed_clips
        summary["vlm_failures"]["stage1_clips_total"]  = len(clips)
        logger.warning(f"  [Stage 1] {s1_failed_clips}/{len(clips)} clips had VLM failures")
    logger.info(f"  [Stage 1] {len(clips)} clip annotations")

    # ---- Stage 1.5: B类专用 — 基于Stage 1实体label判定B1/B2 ----
    anchor_specs: Dict[str, Any] = {}  # B1 anchor specs from web_search (empty for A/B2)
    if dataset_class == "B":
        if os.path.exists(paths["s2b_class"]):
            logger.info(f"  [Stage 1.5_classify] Cache hit → {paths['s2b_class']}")
            b_class_result = load_json(paths["s2b_class"])
        else:
            logger.info(f"  [Stage 1.5_classify] Identifying searchable objects for B2 …")
            from classify_b1_b2 import classify_b1_b2
            b_class_result = await classify_b1_b2(
                omni_info=omni_info,
                out_dir=out_dir,
                model=LLM_MODEL,
            )
            save_json(b_class_result, paths["s2b_class"])
        video_class = b_class_result["video_class"]  # "B"
        summary["video_class"] = video_class
        summary["b_class_searchable"] = b_class_result.get("searchable_objects", [])
        logger.info(f"  [Stage 1.5_classify] → {video_class} "
                    f"(searchable: {b_class_result.get('searchable_objects', [])})")

        # Stage 1.5b: B类 — 物理规格由 Stage 3 author-agent 在生成时 web_search 取得。
        # 这里只把 searchable_objects 列表传入扩展图，供 Stage 3 决定难度档和搜索目标。
        anchor_specs: Dict[str, Any] = {}  # 不再预取，Stage 3 自行搜索
    elif dataset_class == "A":
        video_class = "A"
        summary["video_class"] = "A"
        anchor_specs = {}
        # ---- A类: 注入 ego_pose GT ----
        if gt_annotation_path and os.path.exists(gt_annotation_path):
            if os.path.exists(paths["s1_gt"]):
                logger.info(f"  [Stage 1 GT] Cache hit → {paths['s1_gt']}")
                gt_physics = load_json(paths["s1_gt"])
            else:
                logger.info(f"  [Stage 1 GT] Injecting EgoExo4D ego_pose GT …")
                from physics_extractor import load_egoexo_gt, inject_gt_physics
                gt_physics = load_egoexo_gt(
                    gt_annotation_path,
                    video_fps=omni_info.get("meta", {}).get("fps", 30.0),
                )
                omni_info = inject_gt_physics(omni_info, gt_physics)
                gt_physics = omni_info.get("gt_physics", gt_physics)
                save_json(gt_physics, paths["s1_gt"])
                # Re-save omni_info with injected GT so downstream stages can access it
                save_json([omni_info], paths["s1"])
            omni_info["gt_physics"] = gt_physics
            summary["gt_joints_count"] = gt_physics.get("joint_count", 0)
            logger.info(f"  [Stage 1 GT] Injected GT: "
                        f"{gt_physics.get('joint_count', 0)} joint types, "
                        f"{gt_physics.get('frame_count', 0)} frames")
        else:
            logger.warning(f"  [Stage 1 GT] No GT annotation found at {gt_annotation_path!r} — "
                           f"falling back to CV-only for A-class video")

    # ---- Stage 2.5 (CV): CV physics extraction with track_id ----
    # A-class: YOLO+MediaPipe measurements with GT-calibrated scale.
    # B-class: scale-free extraction — trajectory/deflection angles, speed ratios.
    cv_physics = None
    if os.path.exists(paths["s25cv"]):
        logger.info(f"  [Stage 2.5 CV] Cache hit → {paths['s25cv']}")
        cv_physics = load_json(paths["s25cv"])
    else:
        logger.info(f"  [Stage 2.5 CV] Running CV extraction (video_class={video_class}) …")
        from physics_extractor import run_stage25_cv
        cv_physics = run_stage25_cv(
            omni_info=omni_info,
            video_class=video_class,
            anchor_specs=anchor_specs,
        )
        save_json(cv_physics, paths["s25cv"])
    n_clips_cv = len(cv_physics.get("clip_cv", {}))
    logger.info(f"  [Stage 2.5 CV] {n_clips_cv} clips extracted")

    # ---- Stage 2: Build event graph ----
    if os.path.exists(paths["s2"]):
        logger.info(f"  [Stage 2] Cache hit → {paths['s2']}")
        graph = load_json(paths["s2"])
        if isinstance(graph, list):
            graph = graph[0]
    else:
        logger.info(f"  [Stage 2] Building event graph …")
        graph = await stage2_graph(omni_info, out_dir, cv_physics=cv_physics)
        save_json([graph], paths["s2"])

    summary["stage2_nodes"] = len(graph.get("nodes", []))
    summary["stage2_edges"] = len(graph.get("edges", []))
    logger.info(f"  [Stage 2] {summary['stage2_nodes']} nodes, {summary['stage2_edges']} edges")

    # ---- Stage 2.5a ----
    if shared_library is not None:
        # Reuse caller-provided library
        library = shared_library
        logger.info(f"  [Stage 2.5a] Reusing shared physics library ({library.get('total_entries', 0)} entries)")
    elif os.path.exists(paths["s25a"]):
        logger.info(f"  [Stage 2.5a] Cache hit → {paths['s25a']}")
        library = load_json(paths["s25a"])
    else:
        logger.info(f"  [Stage 2.5a] Generating physics event library …")
        library = await stage25a_library(out_dir)
        save_json(library, paths["s25a"])

    summary["stage25a_entries"] = library.get("total_entries", 0)

    # ---- Stage 2.5b ----
    if os.path.exists(paths["s25b"]):
        logger.info(f"  [Stage 2.5b] Cache hit → {paths['s25b']}")
        expanded = load_json(paths["s25b"])
        if isinstance(expanded, list):
            expanded = expanded[0]
    else:
        logger.info(f"  [Stage 2.5b] Expanding graph with physics …")
        expanded = await stage25b_expand(graph, library)
        save_json([expanded], paths["s25b"])

    summary["stage25b_matches"] = len(expanded.get("physics_matches", []))
    logger.info(f"  [Stage 2.5b] {summary['stage25b_matches']} physics matches")

    # Merge stage1 context into expanded graph so stage3 prompt can access OPQ
    expanded["high_level"] = omni_info.get("high_level", {})
    # A-class: inject slim gt_physics into expanded graph for Stage 3
    # Only derived_quantities is used by Stage 3 prompt; joints_per_frame_sample (~450KB) is not needed
    if omni_info.get("gt_physics"):
        gt = omni_info["gt_physics"]
        expanded["gt_physics"] = {
            k: v for k, v in gt.items()
            if k not in ("joints_per_frame_sample", "all_gt_frame_indices", "measurement_frame_indices")
        }
    mid_level_merged = omni_info.get("mid_level", {})
    # Overwrite each clip's OPQ with Stage 2.5 CV measurements (track_id + calibrated values)
    if cv_physics and cv_physics.get("clip_cv"):
        clips_merged = list(mid_level_merged.get("clips", []))
        for i, clip in enumerate(clips_merged):
            cv_clip = cv_physics["clip_cv"].get(f"clip_{i}")
            if cv_clip and "error" not in cv_clip:
                clips_merged[i] = dict(clip)
                clips_merged[i]["observable_physical_quantities"] = {
                    k: v for k, v in cv_clip.items()
                    if k not in ("clip_idx", "clip_path")
                }
        mid_level_merged = dict(mid_level_merged)
        mid_level_merged["clips"] = clips_merged
    expanded["mid_level"] = mid_level_merged
    expanded["video_path"] = omni_info.get("video_path", video_path)
    expanded["video_class"] = video_class  # A / B — used by stage3 threshold logic
    # No upstream stage fills web_search_specs: KPR specs are fetched by the Stage 3 agent via web_search,
    # and KPR tier eligibility is driven by searchable_objects (set below), not by this dict.
    expanded.setdefault("web_search_specs", {})
    # B-class: pass searchable objects so Stage 3 knows which objects to web_search for specs
    if dataset_class == "B":
        expanded["searchable_objects"] = b_class_result.get("searchable_objects", [])
    if not expanded.get("id") or expanded.get("id") == "<same as input graph id>":
        expanded["id"] = omni_info.get("id", os.path.splitext(os.path.basename(video_path))[0])

    # Model cache, Serper cache and search log of the Stage-3 / Stage-4b checks live in the video's output dir
    import verify_common
    from tools import web_tools
    verify_common.set_work_dir(out_dir)
    web_tools.set_search_log(os.path.join(out_dir, "search_log.jsonl"))

    # ---- Stage 3: grounded QA generation ----
    if os.path.exists(paths["s3"]):
        logger.info(f"  [Stage 3] Cache hit → {paths['s3']}")
        qa_result = load_json(paths["s3"])
        if isinstance(qa_result, list):
            qa_result = qa_result[0]
    else:
        logger.info(f"  [Stage 3] Generating physics QA …")
        qa_result = await stage3_qa(expanded, paths["s1"])
        save_json([qa_result], paths["s3"])

    qa_pairs = qa_result.get("qa_pairs", [])
    summary["stage3_qa_pairs"] = len(qa_pairs)
    summary["stage3_stated_given"] = sum(bool((q.get("verification") or {}).get("stage3", {}).get("stated_given"))
                                         for q in qa_pairs)
    logger.info(f"  [Stage 3] {len(qa_pairs)} QA pairs passed the Stage-3 checks")

    # ---- Stage 4: Quality check ----
    if skip_check:
        checked_pairs = qa_pairs
        logger.info(f"  [Stage 4] Skipped (--no_check)")
    elif os.path.exists(paths["s4"]):
        logger.info(f"  [Stage 4] Cache hit → {paths['s4']}")
        checked_pairs = load_json(paths["s4"])
    else:
        logger.info(f"  [Stage 4] Quality-checking {len(qa_pairs)} QA pairs …")
        checked_pairs = await stage4_check(qa_pairs)
        save_json(checked_pairs, paths["s4"])
    summary["stage4_checked"] = len(checked_pairs)

    # ---- Stage 4b: verification + bounded repair ----
    if skip_check:
        verified_pairs = checked_pairs
        logger.info(f"  [Stage 4b] Skipped (--no_check)")
    elif os.path.exists(paths["s4b"]):
        logger.info(f"  [Stage 4b] Cache hit → {paths['s4b']}")
        verified_pairs = load_json(paths["s4b"])
    else:
        logger.info(f"  [Stage 4b] Verifying {len(checked_pairs)} QA pairs …")
        verified_pairs = await stage4b_verify(checked_pairs, expanded)
        save_json(verified_pairs, paths["s4b"])
    if not skip_check:
        st = [(q.get("verification") or {}) for q in verified_pairs]
        summary["stage4b_verified"] = sum(v.get("status") == "verified" for v in st)
        summary["stage4b_dropped"] = sum(v.get("status") == "dropped" for v in st)
        summary["stage4b_repaired"] = sum(v.get("status") == "verified" and (v.get("rounds") or 0) > 0 for v in st)
        logger.info(f"  [Stage 4b] verified {summary['stage4b_verified']} "
                    f"(repaired {summary['stage4b_repaired']}), dropped {summary['stage4b_dropped']}")

    # ---- Stage 5: Filter (no cache — instant) ----
    valid_pairs = stage5_filter(verified_pairs) if not skip_check else verified_pairs
    summary["stage5_valid_qa"] = len(valid_pairs)
    logger.info(f"  [Stage 5] {len(valid_pairs)}/{len(verified_pairs)} QA pairs passed filter")

    # Mark valid/invalid on each item and re-save Stage 4b with valid flags
    if not skip_check:
        valid_id_set = {id(qa) for qa in valid_pairs}
        for qa in verified_pairs:
            qa["valid"] = (id(qa) in valid_id_set)
        verified_pairs = cleanup_qa_for_export(verified_pairs)
        save_json(verified_pairs, paths["s4b"])

    summary["elapsed_min"] = round((time.time() - t0) / 60, 1)

    if valid_pairs:
        q = valid_pairs[0]
        summary["sample_question"]   = q.get("question", "")
        summary["sample_answer"]     = q.get("answer", "")
        summary["physics_category"]  = q.get("physics_category", "")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all(videos_dir: str, out_dir: str, filter_folders: Optional[List[str]] = None, n_per_folder: int = 1, force: bool = False, specific_video: Optional[str] = None, skip_check: bool = False, parallel: int = 1):
    folders = collect_folders(videos_dir, filter_folders)
    if not folders:
        logger.error(f"No subfolders found in {videos_dir}")
        sys.exit(1)

    logger.info(f"Found {len(folders)} folders → picking {n_per_folder} random video(s) each")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Build physics library once, share across all videos ----
    shared_library_path = os.path.join(out_dir, "_shared_physics_library.json")
    if os.path.exists(shared_library_path):
        logger.info(f"Shared physics library: loading from {shared_library_path}")
        shared_library = load_json(shared_library_path)
    else:
        banner("Building shared Physics Event Library (runs once)")
        shared_library = await stage25a_library(out_dir)
        save_json(shared_library, shared_library_path)
    logger.info(f"Physics library: {shared_library.get('total_entries', 0)} entries")

    # ---- Collect all video tasks ----
    all_summaries = []
    all_qa_entries = []
    all_qa_path = os.path.join(out_dir, "all_qa.json")
    qa_lock = asyncio.Lock()  # protects all_qa_entries + file writes

    # Load existing entries so re-runs append rather than overwrite
    if os.path.exists(all_qa_path):
        try:
            all_qa_entries = load_json(all_qa_path)
            if not isinstance(all_qa_entries, list):
                all_qa_entries = []
            logger.info(f"Resuming: loaded {len(all_qa_entries)} existing QA entries from all_qa.json")
        except Exception:
            all_qa_entries = []

    processed_video_ids = {e["video_id"] for e in all_qa_entries if isinstance(e, dict)}

    # Build flat list of (folder_name, video_path) to process
    video_tasks = []
    for folder in folders:
        folder_name = os.path.basename(folder)
        if specific_video:
            matched = [v for v in glob.glob(os.path.join(folder, "*.mp4"))
                       if os.path.splitext(os.path.basename(v))[0] == specific_video
                       or os.path.basename(v) == specific_video]
            if matched:
                video_tasks.append((folder_name, matched[0]))
        else:
            for vp in pick_random_videos(folder, n_per_folder):
                video_tasks.append((folder_name, vp))

    logger.info(f"Total videos to process: {len(video_tasks)}  (parallel={parallel})")

    # Video-level semaphore to cap concurrent pipeline runs
    video_sem = asyncio.Semaphore(parallel)

    async def process_one(folder_name: str, video_path: str):
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_out = os.path.join(out_dir, folder_name, video_basename)
        banner(f"{folder_name}  →  {os.path.basename(video_path)}")
        async with video_sem:
            try:
                summary = await run_pipeline_for_video(
                    video_path, video_out, shared_library,
                    force=force, skip_check=skip_check,
                )
                summary["folder"] = folder_name
                logger.info(
                    f"[{folder_name}] Done in {summary['elapsed_min']} min  "
                    f"| clips={summary['stage1_clips']}  "
                    f"nodes={summary['stage2_nodes']}  "
                    f"qa={summary['stage3_qa_pairs']}  "
                    f"valid={summary.get('stage5_valid_qa','?')}"
                )

                # Collect QA pairs and save incrementally (thread-safe)
                s4b_path = os.path.join(video_out, "stage4b_verified.json")
                s3_path  = os.path.join(video_out, "stage3_physics_qa.json")
                s1_path  = os.path.join(video_out, "stage1_omni_info.json")
                qa_source = s4b_path if os.path.exists(s4b_path) else s3_path
                if os.path.exists(qa_source):
                    qa_data = load_json(qa_source)
                    if isinstance(qa_data, list) and qa_data:
                        qa_data = qa_data[0] if isinstance(qa_data[0], dict) and "qa_pairs" in qa_data[0] else qa_data
                    pairs = qa_data.get("qa_pairs", qa_data) if isinstance(qa_data, dict) else qa_data
                    if isinstance(pairs, list):   # only the QAs kept by Stage 5
                        pairs = [q for q in pairs if isinstance(q, dict) and q.get("valid", True)]
                    if isinstance(pairs, list) and pairs:
                        real_video_id = f"video_{video_basename}"
                        if os.path.exists(s1_path):
                            s1 = load_json(s1_path)
                            if isinstance(s1, list) and s1:
                                s1 = s1[0]
                            real_video_id = s1.get("id", real_video_id)
                        entry = {
                            "video_id": real_video_id,
                            "sport":    folder_name,
                            "qa_pairs": pairs,
                        }
                        async with qa_lock:
                            existing_idx = next(
                                (i for i, e in enumerate(all_qa_entries)
                                 if isinstance(e, dict) and e.get("video_id") == real_video_id),
                                None,
                            )
                            if existing_idx is not None:
                                all_qa_entries[existing_idx] = entry
                            else:
                                all_qa_entries.append(entry)
                            save_json(all_qa_entries, all_qa_path)
                            logger.info(f"  Saved → {all_qa_path}  ({len(all_qa_entries)} videos total)")

                async with qa_lock:
                    all_summaries.append(summary)

            except Exception as exc:
                logger.error(f"[{folder_name}/{video_basename}] Pipeline failed: {exc}", exc_info=True)
                async with qa_lock:
                    all_summaries.append({"folder": folder_name, "video": video_path, "error": str(exc)})

    await asyncio.gather(*[process_one(fn, vp) for fn, vp in video_tasks])

    # ---- Save reports to _reports/ folder ----
    reports_dir = os.path.join(out_dir, "_reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Full pipeline summary (all videos)
    save_json(all_summaries, os.path.join(reports_dir, "pipeline_summary.json"))

    # 2. VLM failure report — videos with any clip-level VLM failures
    failure_entries = []
    for s in all_summaries:
        failures = s.get("vlm_failures", {})
        pipeline_error = s.get("error")
        if failures or pipeline_error:
            entry = {
                "video":             s.get("video", ""),
                "folder":            s.get("folder", ""),
                "pipeline_error":    pipeline_error,
                "vlm_failures":      failures,
                "stage1_clips":      s.get("stage1_clips", 0),
                "stage3_qa_pairs":   s.get("stage3_qa_pairs", 0),
                "stage5_valid_qa":   s.get("stage5_valid_qa", 0),
                "elapsed_min":       s.get("elapsed_min"),
            }
            failure_entries.append(entry)

    save_json(failure_entries, os.path.join(reports_dir, "failures.json"))
    if failure_entries:
        logger.warning(
            f"{len(failure_entries)} videos with failures → "
            f"{reports_dir}/failures.json  (manual inspection needed)"
        )

    # all_qa.json already saved incrementally; final write
    save_json(all_qa_entries, all_qa_path)
    total_qa = sum(len(e["qa_pairs"]) for e in all_qa_entries if isinstance(e, dict))

    banner(f"ALL DONE — {len(all_summaries)} videos processed")
    logger.info(f"Reports → {reports_dir}/")
    logger.info(f"Combined QA → {all_qa_path}  ({len(all_qa_entries)} videos, {total_qa} QA pairs)")

    success = [s for s in all_summaries if "error" not in s]
    failed  = [s for s in all_summaries if "error" in s]
    logger.info(f"  Success: {len(success)}  |  Failed: {len(failed)}")

    if success:
        logger.info("\nPer-video summary:")
        for s in success:
            vlm_fail = s.get("vlm_failures", {})
            fail_tag = f" ⚠{vlm_fail.get('stage1_clips_failed','?')}/{vlm_fail.get('stage1_clips_total','?')}clips" if vlm_fail else ""
            logger.info(
                f"  {s['folder']:<16} clips={s.get('stage1_clips', 0):>2}  "
                f"nodes={s.get('stage2_nodes', 0):>3}  "
                f"matches={s.get('stage25b_matches', 0):>2}  "
                f"qa={s.get('stage3_qa_pairs', 0):>2}  "
                f"valid={s.get('stage5_valid_qa', '?'):>2}  "
                f"({s.get('elapsed_min', '?')} min){fail_tag}"
            )


def main():
    parser = argparse.ArgumentParser(description="Test pipeline on random video from each Videos/ subfolder")
    parser.add_argument("--videos_dir",    default=VIDEOS_DIR,  help="Root folder containing category subfolders")
    parser.add_argument("--out_dir",       default=OUT_DIR_V2_B, help="Output root directory (default: benchmark_v2_B)")
    parser.add_argument("--folders",       nargs="*",           help="Only test these folder names (space-separated)")
    parser.add_argument("--seed",          type=int, default=None, help="Random seed for video selection")
    parser.add_argument("--n_per_folder",  type=int, default=1, help="Number of random videos to pick per folder (default 1)")
    parser.add_argument("--force",         action="store_true",  help="Delete all cached stage outputs and re-run every stage from scratch")
    parser.add_argument("--video",          default=None,        help="Run on a specific video basename")
    parser.add_argument("--no_check",       action="store_true",  help="Skip Stage 4 quality check and Stage 5 filter")
    parser.add_argument("--parallel",       type=int, default=1, help="Number of videos to process in parallel (default 1)")
    parser.add_argument("--legacy_out",     action="store_true",  help="Use legacy output dir (test_videos_output) instead of benchmark_v2_B")
    args = parser.parse_args()

    if args.legacy_out:
        args.out_dir = OUT_DIR

    if args.seed is not None:
        random.seed(args.seed)

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_all(
        args.videos_dir, args.out_dir, args.folders, args.n_per_folder,
        force=args.force, specific_video=args.video,
        skip_check=args.no_check, parallel=args.parallel,
    ))


if __name__ == "__main__":
    main()
