"""
run_baai_160_batch.py - 批量运行160个BAAI B类视频

配置:
- videos 1-80:   Serper key SERPER_API_KEYS[0]
- videos 81-160: Serper key SERPER_API_KEYS[1] (falls back to [0])
- B类超参数: video_class="B", 无GT数据，使用VLM物理提取
- 并行数: 4个视频同时处理（与A类一致）
- 输出目录: $KPR_OUTPUT_DIR (default B_class_Benchmark/outputs/)
- inputs:  $BAAI_ROOT/<folder_id>/<video_id>.mp4, $BAAI_SELECTION (select_videos/select_baai_160.py),
           $SHARED_LIBRARY (Stage 2.5a library built by run_aclass.py)
"""
import json
import asyncio
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# 导入pipeline函数
from aclass_pipeline_core import run_pipeline_for_video, load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Serper API keys
_SERPER_KEYS = [k.strip() for k in os.environ.get("SERPER_API_KEYS", "").split(",") if k.strip()] \
               or [os.environ.get("SERPER_API_KEY", "")]
SERPER_KEY_1 = _SERPER_KEYS[0]
SERPER_KEY_2 = _SERPER_KEYS[1] if len(_SERPER_KEYS) > 1 else _SERPER_KEYS[0]

# 配置
BAAI_ROOT = Path(os.environ.get("BAAI_ROOT", "B_class_Benchmark/BAAI"))
OUTPUT_ROOT = Path(os.environ.get("KPR_OUTPUT_DIR", "B_class_Benchmark/outputs"))
SELECTION_FILE = os.environ.get("BAAI_SELECTION", "baai_selected_160.json")
CONCURRENT_VIDEOS = 4  # 4个视频并行

# B类超参数（与aclass_pipeline_core.py保持一致）
VIDEO_CLASS = "B"


async def process_single_video(video_info: Dict[str, Any], serper_key: str, shared_library: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个视频"""
    video_id = video_info['video_id']
    folder_id = video_info['folder_id']
    video_path = BAAI_ROOT / folder_id / f"{video_id}.mp4"
    output_dir = OUTPUT_ROOT / folder_id / video_id

    logger.info(f"[{video_id}] Starting pipeline...")

    # 设置Serper API key环境变量
    os.environ['SERPER_API_KEY'] = serper_key

    try:
        # 运行pipeline (传入共享物理库)
        summary = await run_pipeline_for_video(
            video_path=str(video_path),
            out_dir=str(output_dir),
            shared_library=shared_library,
            video_class=VIDEO_CLASS,
            dataset_class="B",
            skip_check=False  # 执行quality check
        )

        logger.info(f"[{video_id}] Completed: {summary.get('stage5_valid_qa', 0)} valid QA pairs")
        return {
            'video_id': video_id,
            'folder_id': folder_id,
            'status': 'success',
            'summary': summary
        }

    except Exception as e:
        logger.error(f"[{video_id}] Failed: {e}")
        return {
            'video_id': video_id,
            'folder_id': folder_id,
            'status': 'error',
            'error': str(e)
        }


async def process_batch(videos: List[Dict[str, Any]], serper_key: str, shared_library: Dict[str, Any], batch_name: str):
    """批量处理一组视频（4个并行）"""
    logger.info(f"=== Starting {batch_name}: {len(videos)} videos ===")

    semaphore = asyncio.Semaphore(CONCURRENT_VIDEOS)

    async def process_with_semaphore(video_info):
        async with semaphore:
            return await process_single_video(video_info, serper_key, shared_library)

    results = []
    tasks = [process_with_semaphore(v) for v in videos]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        logger.info(f"Progress: {len(results)}/{len(videos)} completed")

    return results


async def main():
    # 加载选择的160个视频
    if not os.path.exists(SELECTION_FILE):
        logger.error(f"Selection file not found: {SELECTION_FILE}")
        logger.info("Please run: python select_videos/select_baai_160.py")
        return

    selection = load_json(SELECTION_FILE)
    videos = selection['videos']
    logger.info(f"Loaded {len(videos)} selected videos")

    # 分成两批：前80和后80
    batch1_videos = videos[:80]
    batch2_videos = videos[80:160]

    logger.info(f"\nBatch 1: videos 1-80 (Serper key 1)")
    logger.info(f"Batch 2: videos 81-160 (Serper key 2)")

    # 创建输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- 使用A类的共享物理事件库 ----
    aclass_library_path = Path(os.environ.get("SHARED_LIBRARY",
        os.path.join(os.environ.get("MPR_OUTPUT_DIR", "A_class_Benchmark/output"), "_shared_physics_library.json")))

    if aclass_library_path.exists():
        logger.info(f"\nReusing A-class physics library: {aclass_library_path}")
        shared_library = load_json(str(aclass_library_path))
        logger.info(f"Physics library: {shared_library.get('total_entries', 0)} entries\n")
    else:
        logger.error(f"A-class physics library not found: {aclass_library_path}")
        logger.info("Building new shared physics library...")

        # 导入stage25a_library函数
        from aclass_pipeline_core import stage25a_library
        shared_library = await stage25a_library(str(OUTPUT_ROOT))

        # 保存到B类输出目录（备份）
        b_library_path = OUTPUT_ROOT / "_shared_physics_library.json"
        save_json(shared_library, str(b_library_path))
        logger.info(f"Physics library: {shared_library.get('total_entries', 0)} entries\n")

    start_time = time.time()

    # 批处理1：前80个视频
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH 1: Processing videos 1-80")
    logger.info(f"{'='*70}\n")
    batch1_results = await process_batch(batch1_videos, SERPER_KEY_1, shared_library, "Batch 1")

    # 保存batch1结果
    save_json(batch1_results, OUTPUT_ROOT / "batch1_results.json")

    # 批处理2：后80个视频
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH 2: Processing videos 81-160")
    logger.info(f"{'='*70}\n")
    batch2_results = await process_batch(batch2_videos, SERPER_KEY_2, shared_library, "Batch 2")

    # 保存batch2结果
    save_json(batch2_results, OUTPUT_ROOT / "batch2_results.json")

    # 汇总统计
    all_results = batch1_results + batch2_results
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    error_count = len(all_results) - success_count

    total_valid_qa = sum(
        r.get('summary', {}).get('stage5_valid_qa', 0)
        for r in all_results if r['status'] == 'success'
    )

    elapsed_hours = (time.time() - start_time) / 3600

    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH PROCESSING COMPLETED")
    logger.info(f"{'='*70}")
    logger.info(f"Total videos processed: {len(all_results)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Total valid QA pairs: {total_valid_qa}")
    logger.info(f"Elapsed time: {elapsed_hours:.1f} hours")
    logger.info(f"{'='*70}\n")

    # 保存最终统计
    final_stats = {
        'total_videos': len(all_results),
        'success': success_count,
        'errors': error_count,
        'total_valid_qa_pairs': total_valid_qa,
        'elapsed_hours': round(elapsed_hours, 2),
        'batch1_results': batch1_results,
        'batch2_results': batch2_results
    }
    save_json(final_stats, OUTPUT_ROOT / "final_statistics.json")

    if error_count > 0:
        logger.info("Failed videos:")
        for r in all_results:
            if r['status'] == 'error':
                logger.info(f"  - {r['folder_id']}/{r['video_id']}: {r['error']}")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
