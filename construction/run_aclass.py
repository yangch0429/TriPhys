"""
run_aclass.py — Full pipeline runner for A-class EgoExo4D videos.

Reads:  $EGOEXO_SELECTED_ROOT/manifest.json   (default A_class_Benchmark/egoexo4d_selected)
Output: $MPR_OUTPUT_DIR/{category}/{take_uid}/  (default A_class_Benchmark/output)

Features:
  - dataset_class="A" for all videos (EgoExo4D with ego_pose GT)
  - Parallel video processing (--parallel N, default 8)
  - Serper API key rotation every 64 completed videos
  - Resume support (skips already-completed videos via all_qa.json)
  - Shared physics library across all videos

Run:
    python run_aclass.py
    python run_aclass.py --parallel 8
    python run_aclass.py --take cmu_bike12_4  # single take
    python run_aclass.py --force              # ignore caches
"""

import os
import sys
for _k in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

import json
import asyncio
import logging
import time
import argparse
import shutil
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_aclass")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: extracted EgoExo4D archive
ACLASS_ROOT  = os.environ.get("EGOEXO_SELECTED_ROOT",
                              os.path.join(SCRIPT_DIR, "A_class_Benchmark", "egoexo4d_selected"))
MANIFEST     = os.path.join(ACLASS_ROOT, "manifest.json")

# Output directory
OUT_ROOT     = os.environ.get("MPR_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "A_class_Benchmark", "output"))

# Model endpoints / keys: configured from the environment in aclass_pipeline_core.py.

# Serper API key rotation (64 videos per key)
SERPER_KEYS = [k.strip() for k in os.environ.get("SERPER_API_KEYS", "").split(",") if k.strip()] \
              or [os.environ.get("SERPER_API_KEY", "")]
SERPER_ROTATE_EVERY = 64  # rotate after this many completed videos

# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------
_local_ffmpeg = os.path.join(SCRIPT_DIR, "_ffmpeg", "ffmpeg.exe")
FFMPEG = shutil.which("ffmpeg") or (_local_ffmpeg if os.path.exists(_local_ffmpeg) else None)
if FFMPEG:
    _ffmpeg_dir = os.path.dirname(os.path.abspath(FFMPEG))
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_PATH"] = FFMPEG
else:
    logger.error("FFmpeg not found.")
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
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def set_serper_key(key: str):
    """Hot-swap the Serper API key in tools.web_tools module."""
    import tools.web_tools as wt
    wt.SERPER_API_KEY = key
    logger.info(f"[Serper] Rotated to key ...{key[-8:]}")


# ---------------------------------------------------------------------------
# Import pipeline stages from aclass_pipeline_core.py
# ---------------------------------------------------------------------------
from aclass_pipeline_core import (
    stage1_extract,
    stage2_graph,
    stage25a_library,
    stage25b_expand,
    stage3_qa,
    stage4_check,
    stage5_filter,
    cleanup_qa_for_export,
    run_pipeline_for_video,
)

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
async def run_aclass(
    parallel: int = 8,
    force: bool = False,
    skip_check: bool = False,
    specific_take: Optional[str] = None,
    limit: int = 0,
):
    # ---- Load manifest ----
    if not os.path.exists(MANIFEST):
        logger.error(f"Manifest not found: {MANIFEST}")
        sys.exit(1)

    manifest = load_json(MANIFEST)
    takes = manifest.get("takes", [])
    logger.info(f"Manifest: {len(takes)} takes total")

    # ---- Filter to specific take if requested ----
    if specific_take:
        takes = [t for t in takes if t["take_uid"] == specific_take
                 or t.get("take_name", "") == specific_take]
        if not takes:
            logger.error(f"Take not found in manifest: {specific_take}")
            sys.exit(1)
        logger.info(f"Single-take mode: {takes[0]['take_uid']}")

    if limit > 0:
        takes = takes[:limit]
        logger.info(f"Limit: processing first {limit} takes only")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # ---- Shared physics library ----
    shared_library_path = os.path.join(OUT_ROOT, "_shared_physics_library.json")
    if os.path.exists(shared_library_path):
        logger.info(f"Shared physics library: loading from {shared_library_path}")
        shared_library = load_json(shared_library_path)
    else:
        banner("Building shared Physics Event Library (runs once)")
        shared_library = await stage25a_library(OUT_ROOT)
        save_json(shared_library, shared_library_path)
    logger.info(f"Physics library: {shared_library.get('total_entries', 0)} entries")

    # ---- Resume: load existing all_qa dataset ----
    all_qa_path = os.environ.get("MPR_QA_DATASET",
                                 os.path.join(os.path.dirname(os.path.abspath(OUT_ROOT)), "aclass_qa_dataset.json"))
    all_qa_entries: List[Dict] = []
    if os.path.exists(all_qa_path):
        try:
            all_qa_entries = load_json(all_qa_path)
            if not isinstance(all_qa_entries, list):
                all_qa_entries = []
            logger.info(f"Resume: {len(all_qa_entries)} existing entries in all_qa.json")
        except Exception:
            all_qa_entries = []

    processed_video_ids = {e["video_id"] for e in all_qa_entries if isinstance(e, dict)}

    # ---- Filter already-done takes ----
    pending_takes = []
    for t in takes:
        vid_id = f"video_best_exo_{t['take_uid']}"
        take_out = os.path.join(OUT_ROOT, t.get("category", "unknown"), t["take_uid"])
        s4_path = os.path.join(take_out, "stage4_qa_checked.json")
        s1_path = os.path.join(take_out, "stage1_omni_info.json")
        # Derive real video_id from cached stage1 if available
        if os.path.exists(s1_path):
            try:
                s1 = load_json(s1_path)
                if isinstance(s1, list) and s1:
                    s1 = s1[0]
                vid_id = s1.get("id", vid_id)
            except Exception:
                pass
        if not force and vid_id in processed_video_ids and os.path.exists(s4_path):
            logger.info(f"  SKIP (done): {t['take_uid']}")
            continue
        pending_takes.append(t)

    logger.info(f"Total to process: {len(pending_takes)}  (parallel={parallel})")

    # ---- Serper key rotation state ----
    completed_count = len(all_qa_entries)  # start from already-done count
    completed_lock = asyncio.Lock()
    qa_lock = asyncio.Lock()
    all_summaries: List[Dict] = []

    # Set initial Serper key based on already-completed count
    initial_key_idx = min(completed_count // SERPER_ROTATE_EVERY, len(SERPER_KEYS) - 1)
    set_serper_key(SERPER_KEYS[initial_key_idx])

    # ---- Parallel semaphore ----
    video_sem = asyncio.Semaphore(parallel)

    async def process_one(take: Dict):
        nonlocal completed_count

        take_uid  = take["take_uid"]
        category  = take.get("category", "unknown")
        take_name = take.get("take_name", take_uid)

        video_path = os.path.join(ACLASS_ROOT, take["video_path"])
        gt_path    = os.path.join(ACLASS_ROOT, take["gt_annotation_path"]) \
                     if take.get("gt_annotation_path") else None

        take_out = os.path.join(OUT_ROOT, category, take_uid)

        if not os.path.exists(video_path):
            logger.error(f"  [SKIP] Video not found: {video_path}")
            return

        async with video_sem:
            banner(f"{category}/{take_uid}  ({take.get('best_exo_cam', '?')})")
            try:
                summary = await run_pipeline_for_video(
                    video_path=video_path,
                    out_dir=take_out,
                    shared_library=shared_library,
                    force=force,
                    skip_check=skip_check,
                    video_class="A",
                    dataset_class="A",
                    take_uid=take_uid,
                    gt_annotation_path=gt_path,
                )
                summary["take_uid"]  = take_uid
                summary["category"]  = category
                summary["take_name"] = take_name
                logger.info(
                    f"[{take_uid}] Done {summary['elapsed_min']} min | "
                    f"clips={summary['stage1_clips']} "
                    f"qa={summary['stage3_qa_pairs']} "
                    f"valid={summary.get('stage5_valid_qa','?')}"
                )

                # ---- Collect QA pairs ----
                s4_path = os.path.join(take_out, "stage4_qa_checked.json")
                s3_path = os.path.join(take_out, "stage3_physics_qa.json")
                s1_path = os.path.join(take_out, "stage1_omni_info.json")
                qa_source = s4_path if os.path.exists(s4_path) else s3_path
                if os.path.exists(qa_source):
                    qa_data = load_json(qa_source)
                    if isinstance(qa_data, list) and qa_data:
                        qa_data = qa_data[0] if isinstance(qa_data[0], dict) and "qa_pairs" in qa_data[0] else qa_data
                    pairs = qa_data.get("qa_pairs", qa_data) if isinstance(qa_data, dict) else qa_data
                    if isinstance(pairs, list) and pairs:
                        real_video_id = f"video_best_exo"
                        if os.path.exists(s1_path):
                            s1 = load_json(s1_path)
                            if isinstance(s1, list) and s1:
                                s1 = s1[0]
                            real_video_id = s1.get("id", f"video_{take_uid}")
                        entry = {
                            "video_id":  real_video_id,
                            "take_uid":  take_uid,
                            "take_name": take_name,
                            "category":  category,
                            "qa_pairs":  pairs,
                        }
                        async with qa_lock:
                            idx = next(
                                (i for i, e in enumerate(all_qa_entries)
                                 if isinstance(e, dict) and e.get("take_uid") == take_uid),
                                None,
                            )
                            if idx is not None:
                                all_qa_entries[idx] = entry
                            else:
                                all_qa_entries.append(entry)
                            save_json(all_qa_entries, all_qa_path)

                # ---- Serper key rotation ----
                async with completed_lock:
                    completed_count += 1
                    key_idx = min(completed_count // SERPER_ROTATE_EVERY, len(SERPER_KEYS) - 1)
                    current_key_idx = min((completed_count - 1) // SERPER_ROTATE_EVERY, len(SERPER_KEYS) - 1)
                    if key_idx != current_key_idx:
                        set_serper_key(SERPER_KEYS[key_idx])
                        logger.info(f"[Serper] Rotated after {completed_count} videos → key {key_idx}")

                async with qa_lock:
                    all_summaries.append(summary)

            except Exception as exc:
                logger.error(f"[{take_uid}] Pipeline failed: {exc}", exc_info=True)
                async with qa_lock:
                    all_summaries.append({
                        "take_uid": take_uid,
                        "category": category,
                        "error": str(exc),
                    })

    await asyncio.gather(*[process_one(t) for t in pending_takes])

    # ---- Save reports ----
    reports_dir = os.path.join(OUT_ROOT, "_reports")
    os.makedirs(reports_dir, exist_ok=True)
    save_json(all_summaries, os.path.join(reports_dir, "pipeline_summary.json"))

    total_qa = sum(
        len(e.get("qa_pairs", [])) for e in all_qa_entries if isinstance(e, dict)
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"  ALL DONE: {len(all_qa_entries)} videos, {total_qa} QA pairs")
    logger.info(f"  Output:   {OUT_ROOT}")
    logger.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full A-class EgoExo4D pipeline runner")
    parser.add_argument("--parallel",   type=int, default=8,
                        help="Number of videos to process in parallel (default 8)")
    parser.add_argument("--force",      action="store_true",
                        help="Ignore all caches and reprocess from scratch")
    parser.add_argument("--no_check",   action="store_true",
                        help="Skip Stage 4 quality check")
    parser.add_argument("--take",       default=None,
                        help="Process a single take by take_uid or take_name")
    parser.add_argument("--limit",      type=int, default=0,
                        help="Only process first N takes (for testing)")
    args = parser.parse_args()

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_aclass(
        parallel=args.parallel,
        force=args.force,
        skip_check=args.no_check,
        specific_take=args.take,
        limit=args.limit,
    ))
