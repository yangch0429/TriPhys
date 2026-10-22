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
        "search_related_image/audio_info will return empty results."
    )

logger = logging.getLogger(__name__)

BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-large-en-v1.5")
CACHE_DIR = "./cache"

_image_data = None
_image_embeddings = None
_audio_data = None
_audio_embeddings = None
_audio_map = None
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


def build_image_index(items: List[Dict[str, Any]]) -> None:
    global _image_data, _image_embeddings

    if not _EMBEDDER_AVAILABLE:
        logger.warning("Skipping image index build: sentence_transformers not installed.")
        _image_data = []
        return

    _image_data = []
    corpus = []

    for item in items:
        image_id = item.get("id", "")
        image_path = item.get("paths", {}).get("image", "")
        mid = item.get("mid_level", {})
        keyframes = mid.get("vision", {}).get("keyframes", [])
        if not keyframes:
            continue
        kf = keyframes[0]
        global_summary = kf.get("global_summary", "")

        parts = [global_summary]
        for obj in kf.get("objects", []):
            parts.append(obj.get("label", ""))
            parts.append(obj.get("detailed_features", ""))
        for txt in kf.get("ocr", []):
            parts.append(txt.get("text", ""))

        entry = {
            "id": image_id,
            "image_path": image_path,
            "global_summary": global_summary,
            "objects": kf.get("objects", []),
            "ocr": kf.get("ocr", []),
        }
        _image_data.append(entry)
        corpus.append(" ".join(p for p in parts if p))

    if not corpus:
        logger.warning("No image data to index.")
        return

    embedder = get_embedder()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cid = _cache_id("".join(corpus))
    cache_path = os.path.join(CACHE_DIR, f"image_emb_{cid}.pt")

    if os.path.exists(cache_path):
        _image_embeddings = torch.load(cache_path)
    else:
        _image_embeddings = embedder.encode(corpus, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        torch.save(_image_embeddings, cache_path)

    logger.info(f"Image index built: {len(_image_data)} entries.")


def build_audio_index(audio_file: str) -> None:
    global _audio_data, _audio_embeddings, _audio_map

    if not _EMBEDDER_AVAILABLE:
        logger.warning("Skipping audio index build: sentence_transformers not installed.")
        _audio_data = []
        return

    _audio_data = []
    _audio_map = {}
    corpus = []

    if not os.path.exists(audio_file):
        logger.warning(f"Audio file not found: {audio_file}")
        return

    with open(audio_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        audio_id = item.get("id", "")
        if audio_id:
            _audio_map[audio_id] = item
        audio_path = item.get("paths", {}).get("audio", "")
        high_level = item.get("high_level", {})
        annotation = high_level.get("audio_annotation", {})
        if not annotation:
            mid = item.get("mid_level", {})
            annotation = mid.get("audio", {})
        if not annotation:
            continue

        asr = annotation.get("asr", [])
        asr_text = " ".join(seg.get("text", "") for seg in asr if seg.get("text")) if isinstance(asr, list) else ""
        global_summary = annotation.get("global_summary", "")
        description = (asr_text + " " + global_summary).strip()
        if not description:
            continue

        duration = 0.0
        if isinstance(asr, list) and asr:
            try:
                duration = max(float(seg.get("end", 0.0)) for seg in asr)
            except Exception:
                pass

        entry = {
            "id": audio_id,
            "audio_path": audio_path,
            "duration": duration,
            "description": description,
            "audio_annotation": annotation,
        }
        _audio_data.append(entry)
        corpus.append(description)

    if not corpus:
        logger.warning("No audio data to index.")
        return

    embedder = get_embedder()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cid = _cache_id("".join(corpus))
    cache_path = os.path.join(CACHE_DIR, f"audio_emb_{cid}.pt")

    if os.path.exists(cache_path):
        _audio_embeddings = torch.load(cache_path)
    else:
        _audio_embeddings = embedder.encode(corpus, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
        torch.save(_audio_embeddings, cache_path)

    logger.info(f"Audio index built: {len(_audio_data)} entries.")


async def search_related_image_info(
    query: str,
    current_image_id: str = "",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    global _image_data, _image_embeddings

    if _image_embeddings is None or _image_data is None:
        return [{"error": "Image index not initialized (sentence_transformers may not be installed)."}]
    if not query.strip():
        return [{"error": "Empty query."}]

    embedder = get_embedder()
    q_emb = embedder.encode(query, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(q_emb, _image_embeddings, top_k=top_k + 10, score_function=util.cos_sim)

    results = []
    for hit in hits[0]:
        if len(results) >= top_k:
            break
        idx = hit["corpus_id"]
        item = _image_data[idx]
        if item["id"] == current_image_id:
            continue
        if hit["score"] > 0:
            results.append({
                "id": item["id"],
                "image_path": item["image_path"],
                "summary": item["global_summary"],
                "key_objects": [o.get("label") for o in item["objects"][:5]],
                "ocr_text": [t.get("text") for t in item["ocr"][:5]],
            })

    return results or [{"message": "No related images found."}]


async def search_related_audio_info(
    query: str,
    top_k: int = 5,
    return_full_audio: bool = False,
) -> List[Dict[str, Any]]:
    global _audio_data, _audio_embeddings, _audio_map

    if _audio_embeddings is None or _audio_data is None:
        return [{"error": "Audio index not initialized (sentence_transformers may not be installed)."}]
    if not query.strip():
        return [{"error": "Empty query."}]

    embedder = get_embedder()
    q_emb = embedder.encode(query, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(q_emb, _audio_embeddings, top_k=20, score_function=util.cos_sim)

    results = []
    total_dur = 0.0
    MAX_DUR = 12 * 60
    target = 2 if return_full_audio else top_k

    for hit in hits[0]:
        if len(results) >= target:
            break
        idx = hit["corpus_id"]
        item = _audio_data[idx]
        dur = item.get("duration", 0.0)
        if dur > MAX_DUR:
            continue
        if total_dur + dur > MAX_DUR and len(results) >= 1:
            break
        if hit["score"] > 0:
            ann = item["audio_annotation"].copy()
            if isinstance(ann.get("asr"), list):
                ann["asr"] = " ".join(
                    f"[{s.get('speaker','?')}, {s.get('start',0)}-{s.get('end',0)}s] {s.get('text','')}"
                    for s in ann["asr"] if s.get("text")
                )
            res = {
                "id": item["id"],
                "audio_path": item["audio_path"],
                "asr": ann.get("asr", ""),
                "speakers": ann.get("speakers", {}),
                "nonspeech_information": ann.get("nonspeech_information", ""),
                "global_summary": ann.get("global_summary", ""),
                "duration": dur,
            }
            if return_full_audio and _audio_map and item["id"] in _audio_map:
                info_str = json.dumps(_audio_map[item["id"]], ensure_ascii=False)
                res["full_audio_info"] = info_str[:30000] + ("...(truncated)" if len(info_str) > 30000 else "")
            results.append(res)
            if return_full_audio:
                break
            total_dur += dur

    return results or [{"message": "No related audio found."}]


def get_openai_function_search_related_image_info() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_related_image_info",
            "description": "Search for related images from the dataset using semantic search. Returns image summaries, objects, and OCR text. Always returns up to 5 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for image content."}
                },
                "required": ["query"],
            },
        },
    }


def get_openai_function_search_related_audio_info() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_related_audio_info",
            "description": "Search for related audio clips using semantic search. Returns audio descriptions, ASR, and speaker info. Total duration does not exceed 12 minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for audio content."}
                },
                "required": ["query"],
            },
        },
    }
