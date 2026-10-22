"""
run_aclass_batch.py - batch re-run of the A-class (MPR) pipeline over a list of takes (4 videos in parallel).

It calls the same run_pipeline_for_video() as run_aclass.py; cached Stage 1-2.5 files in each take's output
directory are reused, so delete stage3_physics_qa.json, stage4_qa_checked.json and stage4b_verified.json of a take
to regenerate its QA.

Usage:
    python run_aclass_batch.py --videos rerun_videos.json [--results batch_results.json]
    # rerun_videos.json: {"videos": [{"take_uid": ..., "category": ..., "take_name": ...}, ...]}

Env: EGOEXO_SELECTED_ROOT (packed EgoExo4D dir: takes/<uid>/best_exo.mp4, annotations/<uid>.json),
     MPR_OUTPUT_DIR (default A_class_Benchmark/output; must contain _shared_physics_library.json),
     SERPER_API_KEYS (first key is used), plus the model variables read by aclass_pipeline_core.py.
"""
import os
import sys
for _k in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_run")

sys.path.insert(0, str(Path(__file__).parent))
from aclass_pipeline_core import run_pipeline_for_video, load_json, save_json

OUTPUT_ROOT = Path(os.environ.get("MPR_OUTPUT_DIR", "A_class_Benchmark/output"))
ACLASS_ROOT = Path(os.environ.get("EGOEXO_SELECTED_ROOT", "A_class_Benchmark/egoexo4d_selected"))
SHARED_LIBRARY_PATH = OUTPUT_ROOT / "_shared_physics_library.json"
CONCURRENT_VIDEOS = 4


async def process_single_video(video_info, shared_library, semaphore, progress):
    async with semaphore:
        take_uid = video_info['take_uid']
        category = video_info['category']
        take_name = video_info.get('take_name', take_uid[:16])
        logger.info(f"[{take_name}] Starting...")

        video_path = ACLASS_ROOT / "takes" / take_uid / "best_exo.mp4"
        output_dir = OUTPUT_ROOT / category / take_uid
        # ego_pose GT as laid out by select_videos/pack_egoexo_server.py (same path run_aclass.py uses).
        gt_annotation_path = ACLASS_ROOT / "annotations" / f"{take_uid}.json"

        if not video_path.exists():
            logger.error(f"[{take_name}] Video not found")
            progress['completed'] += 1
            return {'take_uid': take_uid, 'category': category, 'status': 'error',
                    'error': 'Video file not found'}
        if not gt_annotation_path.exists():
            gt_annotation_path = None

        try:
            summary = await run_pipeline_for_video(
                video_path=str(video_path),
                out_dir=str(output_dir),
                shared_library=shared_library,
                video_class="A",
                dataset_class="A",
                take_uid=take_uid,
                gt_annotation_path=str(gt_annotation_path) if gt_annotation_path else None,
                skip_check=False,
                force=False,
            )
            progress['completed'] += 1
            logger.info(f"[{take_name}] Completed: {summary.get('stage5_valid_qa', 0)} valid QA "
                        f"({progress['completed']}/{progress['total']})")
            return {'take_uid': take_uid, 'category': category, 'take_name': take_name,
                    'status': 'success', 'summary': summary}
        except Exception as e:
            progress['completed'] += 1
            logger.error(f"[{take_name}] Failed: {e} ({progress['completed']}/{progress['total']})")
            return {'take_uid': take_uid, 'category': category, 'status': 'error', 'error': str(e)}


async def main(videos_file, results_file):
    with open(videos_file, encoding='utf-8') as f:
        videos = json.load(f)['videos']
    logger.info(f"Total videos: {len(videos)}, Concurrency: {CONCURRENT_VIDEOS}")

    if not SHARED_LIBRARY_PATH.exists():
        logger.error(f"Physics library not found: {SHARED_LIBRARY_PATH} (run run_aclass.py first)")
        return
    shared_library = load_json(str(SHARED_LIBRARY_PATH))
    logger.info(f"Physics library: {shared_library.get('total_entries', 0)} entries")

    keys = [k.strip() for k in os.environ.get("SERPER_API_KEYS", "").split(",") if k.strip()]
    if keys and not os.environ.get("SERPER_API_KEY"):
        os.environ['SERPER_API_KEY'] = keys[0]

    start_time = datetime.now()
    semaphore = asyncio.Semaphore(CONCURRENT_VIDEOS)
    progress = {'completed': 0, 'total': len(videos)}
    results = await asyncio.gather(*[process_single_video(v, shared_library, semaphore, progress)
                                     for v in videos])

    save_json(results, str(results_file))
    ok = sum(1 for r in results if r['status'] == 'success')
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Done: {ok}/{len(results)} succeeded in {elapsed:.1f} min; results -> {results_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help='JSON file {"videos": [{"take_uid","category",...}]}')
    ap.add_argument("--results", default="batch_results.json")
    args = ap.parse_args()
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args.videos, args.results))
