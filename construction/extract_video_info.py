"""
extract_video_info.py  —  Standalone video extractor (OpenRouter edition)

Design:
  - Uniform slicing: each clip >= 4s; last clip kept even if shorter
    e.g. 10s video  →  [0-4s, 4-8s, 8-10s]
  - No audio extraction or analysis
  - All model calls go through OpenRouter
  - asyncio subprocess via thread executor (works on Windows Proactor + Selector)

Standalone usage:
    python extract_video_info.py --video path/to/video.mp4 --out_dir ./output

As a library:
    from extract_video_info import extract_video_info
    result = asyncio.run(extract_video_info("video.mp4", "./output"))
"""

import os
import re
import json
import math
import asyncio
import base64
import mimetypes
import subprocess
import logging
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional

import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Endpoints, keys and model ids are read from the environment (see README.md).
# VLM (Gemini) via an OpenAI-compatible endpoint; the released data used OpenRouter.
VLM_BASE_URL = os.environ.get("CHAT_API_BASE", "https://openrouter.ai/api/v1")
VLM_API_KEY  = os.environ.get("VLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

VLM_MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-3.6-flash")    # vision — via OpenRouter
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v3.2")           # reasoning (unused here) — via OpenRouter

MIN_CLIP_DURATION = 4.0   # seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_video_info")


# ---------------------------------------------------------------------------
# FFmpeg resolution
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> str:
    import shutil
    env_path = os.environ.get("FFMPEG_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ffmpeg", "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    raise RuntimeError(
        "FFmpeg not found. Install FFmpeg and add to PATH, "
        "or place ffmpeg.exe in the _ffmpeg/ subfolder."
    )


FFMPEG_BIN = _resolve_ffmpeg()
logger.info(f"FFmpeg: {FFMPEG_BIN}")


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_meta(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    meta: Dict[str, Any] = {}
    if cap.isOpened():
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        meta = {
            "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":         float(fps),
            "frame_count": int(frames),
            "duration":    round(frames / fps, 3) if fps > 0 else 0.0,
        }
    cap.release()
    return meta


# ---------------------------------------------------------------------------
# Uniform clip computation
# ---------------------------------------------------------------------------

def compute_uniform_clips(
    duration: float,
    min_duration: float = MIN_CLIP_DURATION,
) -> List[Tuple[float, float]]:
    """
    Divide [0, duration] into equal segments of exactly min_duration seconds.
    The final remainder segment is kept if it is > 50 ms, even if shorter than
    min_duration  (e.g. 10s → [(0,4), (4,8), (8,10)]).

    Returns list of (start_sec, end_sec).
    """
    if duration <= 0:
        return []

    n_full = math.floor(duration / min_duration)

    if n_full == 0:
        # Whole video is shorter than min_duration — treat as one clip
        return [(0.0, round(duration, 3))]

    clips: List[Tuple[float, float]] = []
    for i in range(n_full):
        s = round(i * min_duration, 3)
        e = round(s + min_duration, 3)
        clips.append((s, e))

    # Remainder: only keep if >= 1.0s (Gemini API may reject very short clips)
    remainder = duration - n_full * min_duration
    if remainder >= 1.0:
        clips.append((round(n_full * min_duration, 3), round(duration, 3)))

    return clips


# ---------------------------------------------------------------------------
# FFmpeg slicing  (thread-executor — safe on all Windows event loop policies)
# ---------------------------------------------------------------------------

async def slice_video(
    video_path: str,
    clip_dir: str,
    min_duration: float = MIN_CLIP_DURATION,
) -> List[Tuple[float, float, str]]:
    """
    Slice *video_path* into uniform clips.
    Returns list of (start_sec, end_sec, clip_path).
    """
    os.makedirs(clip_dir, exist_ok=True)

    meta     = get_video_meta(video_path)
    duration = meta.get("duration", 0.0)
    if duration <= 0:
        logger.warning(f"Could not determine duration for {video_path}")
        return []

    segments = compute_uniform_clips(duration, min_duration)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    loop     = asyncio.get_event_loop()
    results: List[Tuple[float, float, str]] = []

    logger.info(
        f"Slicing '{os.path.basename(video_path)}' "
        f"({duration:.1f}s) → {len(segments)} clips (min {min_duration}s each)"
    )

    for idx, (start, end) in enumerate(segments):
        clip_path = os.path.join(clip_dir, f"{basename}_clip_{idx:03d}.mp4")
        cmd = [
            FFMPEG_BIN, "-y",
            "-i",  video_path,
            "-ss", f"{start:.3f}",
            "-t",  f"{end - start:.3f}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an",                   # strip audio
            "-loglevel", "error",
            clip_path,
        ]

        def _run(c=cmd):
            return subprocess.run(
                c,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ},
            )

        res = await loop.run_in_executor(None, _run)
        if res.returncode == 0:
            results.append((start, end, clip_path))
            logger.debug(f"  clip {idx:03d}  {start:.1f}s → {end:.1f}s")
        else:
            err = res.stderr.decode("utf-8", errors="replace")
            logger.warning(f"  clip {idx:03d} FFmpeg failed: {err[:300]}")

    logger.info(f"  → {len(results)} clips extracted to {clip_dir}")
    return results


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[Dict]:
    """Extract the first JSON object from LLM output."""
    # Strip <think> blocks (DeepSeek-R1 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    for pattern in [
        r"```(?:json|JSON)\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
    ]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Video → base64 data URL helper
# ---------------------------------------------------------------------------

def _video_to_data_url(path: str, max_height: int = 720) -> str:
    """Convert video to base64 data URL, downscaling to max_height if taller."""
    import subprocess, tempfile, shutil
    mime = "video/mp4"

    # Check if downscaling is needed — use ffmpeg -i (stderr) since ffprobe may not be present
    try:
        import re as _re
        _ffmpeg_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ffmpeg", "ffmpeg.exe")
        if not os.path.exists(_ffmpeg_bin):
            _ffmpeg_bin = "ffmpeg"
        probe = subprocess.run(
            [_ffmpeg_bin, "-i", path],
            capture_output=True, text=True, timeout=10
        )
        # ffmpeg -i prints "Stream ... Video: ... 3840x2160 ..." to stderr
        m = _re.search(r"(\d{3,5})x(\d{3,5})", probe.stderr)
        height = int(m.group(2)) if m else 0
    except Exception:
        height = 0

    src = path
    tmp_dir = None
    if height > max_height:
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "scaled.mp4")
        ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ffmpeg", "ffmpeg.exe")
        if not os.path.exists(ffmpeg):
            ffmpeg = "ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-i", path,
             "-vf", f"scale=-2:{max_height}",
             "-c:v", "libx264", "-crf", "23", "-preset", "fast",
             "-an", tmp_path],
            capture_output=True, timeout=180
        )
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            src = tmp_path

    try:
        with open(src, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# LLM / VLM call via OpenRouter
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an advanced multimodal AI assistant. "
    "Analyze the provided video content with high precision. "
    "Output valid JSON wrapped in a markdown code block (```json ... ```). "
    "Do not include any conversational text outside the code block."
)

PROMPT_FULL_VIDEO = """\
Analyze this video and provide a structured annotation.

Output this JSON structure:
```json
{
    "global_summary": "Highly detailed summary of the entire video",
    "scenes": [
        {
            "start_time": 0.0,
            "end_time": 5.0,
            "description": "Detailed description of everything in this scene"
        }
    ],
    "events": [
        {
            "description": "Detailed description of the event",
            "time_point": 2.5,
            "modality": "visual"
        }
    ]
}
```

Be comprehensive and factual. Output only the JSON code block."""


def _clip_prompt(start: float, end: float, total: float) -> str:
    return f"""\
Analyze this video clip (segment {start:.2f}s–{end:.2f}s of a {total:.2f}s video).

Output this JSON structure:
```json
{{
    "global_summary": "Detailed summary of all actions and events in this clip",
    "visual_elements": [
        "Exhaustive list: every object, action, text, colour, expression, background detail visible"
    ]
}}
```

Be comprehensive and factual. Output only the JSON code block."""


async def _call_vlm(
    client,
    semaphore: asyncio.Semaphore,
    prompt: str,
    video_path: str,
    retries: int = 10,
) -> Dict[str, Any]:
    """Send a video + prompt to the VLM via OpenRouter.
    Retries up to `retries` times with exponential backoff (max 60s wait).
    A clip that fails all retries returns {"error": "vlm_call_failed"} rather
    than raising, so the caller can decide how to handle the gap.
    """
    data_url = _video_to_data_url(video_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": data_url}},
                {"type": "text",      "text":      prompt},
            ],
        },
    ]

    async with semaphore:
        for attempt in range(retries):
            try:
                t0   = time.time()
                resp = await client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=messages,
                    max_tokens=4096,
                    timeout=90,
                    extra_headers={"HTTP-Referer": "https://github.com/OmniGAIA"},
                )
                if not resp or not resp.choices:
                    logger.warning(f"VLM call failed attempt {attempt+1}/{retries}: empty choices in response")
                    continue
                text   = resp.choices[0].message.content
                if not text:
                    logger.warning(f"VLM call failed attempt {attempt+1}/{retries}: null content in response")
                    continue
                parsed = _parse_json(text)
                if parsed is not None:
                    parsed["_latency_s"] = round(time.time() - t0, 2)
                    return parsed
                logger.warning(f"JSON parse failed attempt {attempt+1}/{retries}, snippet: {text[:120]}")
            except Exception as exc:
                clip_name = os.path.basename(video_path)
                logger.warning(f"VLM call failed attempt {attempt+1}/{retries} for {clip_name}: {exc}")
            if attempt < retries - 1:
                wait = min(60, 2 ** attempt)
                await asyncio.sleep(wait)

    logger.error(f"VLM call failed after {retries} attempts for {os.path.basename(video_path)}")
    return {"error": "vlm_call_failed"}


# ---------------------------------------------------------------------------
# Top-level extraction function
# ---------------------------------------------------------------------------

async def extract_video_info(
    video_path: str,
    out_dir: str,
    concurrency: int = 4,
    min_clip_duration: float = MIN_CLIP_DURATION,
    sport_category: str = "",
    video_class: str = "B1",
) -> Dict[str, Any]:
    """
    Extract structured information from a single video.

    Steps:
      1. Read video metadata
      2. Slice into uniform clips (>= min_clip_duration each)
      3. Call VLM on full video (high-level annotation)
      4. Call VLM on each clip in parallel (mid-level annotation)

    Returns a dict with keys: id, video_path, low_level, mid_level, high_level
    """
    from openai import AsyncOpenAI

    os.makedirs(out_dir, exist_ok=True)
    basename  = os.path.splitext(os.path.basename(video_path))[0]
    clip_dir  = os.path.join(out_dir, "clips", basename)

    # --- metadata ---
    meta           = get_video_meta(video_path)
    total_duration = meta.get("duration", 0.0)
    logger.info(
        f"[{basename}] {total_duration:.1f}s  "
        f"{meta.get('width')}x{meta.get('height')}  "
        f"{meta.get('fps'):.1f}fps"
    )

    # --- slice ---
    clips = await slice_video(video_path, clip_dir, min_clip_duration)

    # --- model calls ---
    client = AsyncOpenAI(base_url=VLM_BASE_URL, api_key=VLM_API_KEY)
    sem    = asyncio.Semaphore(concurrency)

    high_level_coro = _call_vlm(client, sem, PROMPT_FULL_VIDEO, video_path)
    clip_coros      = [
        _call_vlm(client, sem, _clip_prompt(s, e, total_duration), path)
        for s, e, path in clips
    ]

    all_results = await asyncio.gather(
        high_level_coro,
        *clip_coros,
        return_exceptions=True,
    )

    await client.close()

    # --- assemble ---
    high_level_data = all_results[0] if not isinstance(all_results[0], Exception) else {"error": str(all_results[0])}

    # Stage 1 is VLM-only. CV physics extraction happens in Stage 2.5 (physics_extractor.run_stage25_cv).
    clip_annotations = []
    for i, (start, end, path) in enumerate(clips):
        res = all_results[i + 1]
        if isinstance(res, Exception):
            logger.error(f"Clip {i} analysis error: {res}")
            res = {"error": str(res)}

        clip_annotations.append({
            "clip_index":      i,
            "start_time":      start,
            "end_time":        end,
            "clip_path":       path,
            "global_summary":  res.get("global_summary", ""),
            "visual_elements": res.get("visual_elements", []),
        })

    logger.info(f"[{basename}] Extracted {len(clip_annotations)} clip annotations")

    return {
        "id":         f"video_{basename}",
        "video_path": video_path,
        "low_level":  {"meta": meta},
        "mid_level":  {"clips": clip_annotations},
        "high_level": {"annotation": high_level_data},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _cli_main():
    parser = argparse.ArgumentParser(description="Extract video info via OpenRouter VLM")
    parser.add_argument("--video",       required=True,        help="Input video file path")
    parser.add_argument("--out_dir",     default="./video_output", help="Output directory")
    parser.add_argument("--concurrency", type=int, default=4,  help="Max concurrent VLM calls")
    parser.add_argument("--min_clip",    type=float, default=MIN_CLIP_DURATION,
                        help="Minimum clip duration in seconds (default 4.0)")
    args = parser.parse_args()

    result   = await extract_video_info(args.video, args.out_dir, args.concurrency, args.min_clip)
    out_path = os.path.join(
        args.out_dir,
        os.path.splitext(os.path.basename(args.video))[0] + "_info.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved → {out_path}")
    clips = result["mid_level"]["clips"]
    logger.info(f"Clips ({len(clips)}):")
    for c in clips:
        logger.info(f"  [{c['clip_index']:03d}] {c['start_time']:.1f}s–{c['end_time']:.1f}s  "
                    f"{os.path.basename(c['clip_path'])}")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(_cli_main())
