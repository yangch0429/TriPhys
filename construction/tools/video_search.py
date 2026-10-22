import json
import os
import hashlib
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    _EMBEDDER_AVAILABLE = True
except ImportError:
    _EMBEDDER_AVAILABLE = False
    logger.warning(
        "sentence_transformers or torch not installed. "
        "search_related_video_info will return empty results. "
        "Install with: pip install sentence-transformers torch"
    )

logger = logging.getLogger(__name__)

BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-large-en-v1.5")
CACHE_DIR = "./cache"

_video_clips_data = None
_video_embeddings = None
_videos_map = None
_embedder = None


def get_embedder() -> "SentenceTransformer":
    if not _EMBEDDER_AVAILABLE:
        raise RuntimeError("sentence_transformers is not installed.")
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(BGE_MODEL_PATH)
    return _embedder


def _cache_id(data_str: str) -> str:
    return hashlib.md5((BGE_MODEL_PATH + data_str).encode()).hexdigest()


def build_video_index(json_file_paths: Union[str, List[str]]) -> None:
    global _video_clips_data, _video_embeddings, _videos_map

    if not _EMBEDDER_AVAILABLE:
        logger.warning("Skipping video index build: sentence_transformers not installed.")
        _video_clips_data = []
        _videos_map = {}
        return

    if isinstance(json_file_paths, str):
        json_file_paths = [json_file_paths]

    _video_clips_data = []
    _videos_map = {}
    corpus = []

    for path in json_file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            continue

        for rec in records:
            vid = rec.get("id", "")
            if vid:
                _videos_map[vid] = rec

            video_path = rec.get("paths", {}).get("video", "")
            high_level = rec.get("high_level", {})
            omni = high_level.get("omni_annotation", {})
            video_summary = omni.get("global_summary", "") if isinstance(omni, dict) else ""

            mid = rec.get("mid_level", {})
            clips = mid.get("vision", {}).get("clips", [])

            for clip in clips:
                entry = {
                    "video_id": vid,
                    "video_path": video_path,
                    "clip_path": clip.get("clip_path", ""),
                    "start_time": clip.get("start_time", 0),
                    "end_time": clip.get("end_time", 0),
                    "global_summary": clip.get("global_summary", ""),
                    "visual_elements": clip.get("visual_elements", []),
                    "audio_elements": clip.get("audio_elements", []),
                    "video_summary": video_summary,
                }
                _video_clips_data.append(entry)

                parts = [clip.get("global_summary", ""), video_summary]
                for ve in clip.get("visual_elements", []):
                    parts.append(ve if isinstance(ve, str) else ve.get("description", ""))
                for ae in clip.get("audio_elements", []):
                    parts.append(ae if isinstance(ae, str) else ae.get("description", ""))
                corpus.append(" ".join(p for p in parts if p))

    if not corpus:
        logger.warning("No video data to index.")
        return

    embedder = get_embedder()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cid = _cache_id("".join(corpus))
    cache_path = os.path.join(CACHE_DIR, f"video_emb_{cid}.pt")

    if os.path.exists(cache_path):
        _video_embeddings = torch.load(cache_path)
    else:
        _video_embeddings = embedder.encode(corpus, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        torch.save(_video_embeddings, cache_path)

    logger.info(f"Video index built: {len(_video_clips_data)} clips.")


async def search_related_video_info(
    query: str,
    current_video_id: str,
    top_k: int = 5,
    return_full_video: bool = False,
) -> List[Dict[str, Any]]:
    global _video_clips_data, _video_embeddings, _videos_map

    if _video_embeddings is None or _video_clips_data is None:
        return [{"error": "Video index not initialized (sentence_transformers may not be installed)."}]
    if not query.strip():
        return [{"error": "Empty query."}]

    embedder = get_embedder()
    q_emb = embedder.encode(query, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(q_emb, _video_embeddings, top_k=100, score_function=util.cos_sim)

    results = []
    total_dur = 0
    MAX_DUR = 12 * 60
    target = 1 if return_full_video else top_k

    for hit in hits[0]:
        if len(results) >= target:
            break
        idx = hit["corpus_id"]
        score = hit["score"]
        clip = _video_clips_data[idx]

        if clip["video_id"] == current_video_id:
            continue

        dur = clip["end_time"] - clip["start_time"]
        if dur > MAX_DUR:
            continue
        if total_dur + dur > MAX_DUR and len(results) >= 1:
            break

        if score > 0:
            res = {
                "video_id": clip["video_id"],
                "video_path": clip["video_path"],
                "clip_path": clip["clip_path"],
                "time_range": f"{clip['start_time']:.2f}s - {clip['end_time']:.2f}s",
                "clip_summary": clip["global_summary"],
                "video_summary": clip["video_summary"],
                "visual_elements": clip["visual_elements"][:3],
                "audio_elements": clip["audio_elements"][:3],
            }
            if return_full_video and _videos_map and clip["video_id"] in _videos_map:
                info_str = json.dumps(_videos_map[clip["video_id"]], ensure_ascii=False)
                res["full_video_info"] = info_str[:30000] + ("...(truncated)" if len(info_str) > 30000 else "")
            results.append(res)
            if return_full_video:
                break
            total_dur += dur

    return results or [{"message": "No related video clips found."}]


def get_openai_function_search_related_video_info() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_related_video_info",
            "description": (
                "Search for related video clips from OTHER videos in the dataset using semantic search. "
                "Automatically excludes clips from the current video. "
                "Returns between 1 and 5 results with total duration not exceeding 12 minutes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query describing the content you are looking for.",
                    }
                },
                "required": ["query"],
            },
        },
    }
