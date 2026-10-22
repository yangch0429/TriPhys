import os
import re
import base64
import mimetypes
import asyncio
from typing import Dict, Any
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:8089/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "empty")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-omni-30b-a3b-thinking")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

VQA_SYSTEM_PROMPT = """You are a helpful visual assistant that can analyze images and videos to answer questions.
Provide accurate, detailed, and helpful responses based on what you observe in the visual content.
If you cannot determine something from the visual content, say so clearly.
Focus on providing factual observations rather than speculative interpretations."""


def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("data:"))


def _to_data_url(path: str, default_mime: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or default_mime
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _detect_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        if mime.startswith("image/"):
            return "image"
        elif mime.startswith("video/"):
            return "video"
    return "image"


def _build_media_part(media_type: str, source: str) -> dict:
    default_mimes = {"image": "image/jpeg", "video": "video/mp4"}
    if _is_url(source):
        url = source
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"{media_type} file not found: {source}")
        url = _to_data_url(source, default_mimes[media_type])
    return {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}


def _clean_response(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


async def visual_question_answering(
    file_path: str,
    question: str,
    max_tokens: int = 20480,
    timeout: int = 1200,
) -> Dict[str, Any]:
    result = {"answer": None, "media_type": None, "error": None}

    try:
        media_type = _detect_media_type(file_path)
        result["media_type"] = media_type

        contents = []
        try:
            media_part = _build_media_part(media_type, file_path)
            contents.append(media_part)
        except FileNotFoundError as e:
            result["error"] = str(e)
            return result
        except Exception as e:
            result["error"] = f"Failed to load media file: {str(e)}"
            return result

        contents.append({"type": "text", "text": question})

        messages = [
            {"role": "system", "content": VQA_SYSTEM_PROMPT},
            {"role": "user", "content": contents},
        ]

        client = AsyncOpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)
        try:
            completion = await client.chat.completions.create(
                model=QWEN_MODEL,
                messages=messages,
                stream=False,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            result["answer"] = _clean_response(completion.choices[0].message.content)
        finally:
            await client.close()

    except asyncio.TimeoutError:
        result["error"] = f"Request timed out after {timeout} seconds"
    except Exception as e:
        result["error"] = f"VQA request failed: {str(e)}"
        logger.error(f"VQA error for {file_path}: {e}")

    return result


def get_openai_function_visual_question_answering() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "visual_question_answering",
            "description": (
                "Ask a question about an image or video file and get an answer based on visual analysis. "
                "Use this tool to understand the content of images or video clips, extract specific visual information "
                "(text, objects, scenes, actions, etc.), verify visual details, or analyze downloaded images from web_image_search. "
                "The tool automatically detects whether the input is an image or video based on file extension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image or video file (local path or URL). Supported: JPG, PNG, GIF, WebP, BMP (images); MP4, AVI, MOV, MKV (videos).",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the visual content. Be specific about what you want to know.",
                    },
                },
                "required": ["file_path", "question"],
            },
        },
    }
