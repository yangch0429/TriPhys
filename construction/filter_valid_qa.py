import json
import os
import re
import copy
import argparse
from typing import List, Dict, Any, Union, Tuple, Optional


EXTENSIONS = {
    'video': ['.mp4', '.mkv', '.avi', '.webm'],
    'audio': ['.wav', '.mp3', '.flac', '.m4a'],
    'image': ['.jpg', '.jpeg', '.png']
}

BASE_DIRS = {
    'video': os.getenv("VIDEO_BASE_DIR", "./videos"),
    'audio': os.getenv("AUDIO_BASE_DIR", "./audios"),
    'image': os.getenv("IMAGE_BASE_DIR", "./images"),
}


def parse_source_id(sid: str) -> Tuple[Optional[str], str]:
    if not sid:
        return (None, sid)

    if sid.startswith("video_finevideo_"):
        return ('video', sid[len("video_finevideo_"):])
    if sid.startswith("finevideo_"):
        return ('video', sid[len("finevideo_"):])
    if sid.startswith("video_"):
        return ('video', sid[len("video_"):])

    if sid.startswith("image_coco_"):
        return ('image', sid[len("image_coco_"):])
    if sid.startswith("coco_"):
        return ('image', sid[len("coco_"):])
    if sid.startswith("image_"):
        return ('image', sid[len("image_"):])

    if sid.startswith("audio_audiocaps_"):
        return ('audio', sid[len("audio_audiocaps_"):])
    if sid.startswith("audiocaps_"):
        return ('audio', sid[len("audiocaps_"):])
    if sid.startswith("audio_"):
        return ('audio', sid[len("audio_"):])

    return (None, sid)


def normalize_source_id(sid: str) -> str:
    modality, basename = parse_source_id(sid)
    if modality:
        return f"{modality}_{basename}"
    return sid


def normalize_item_sources(item: Dict[str, Any]) -> Dict[str, Any]:
    item = copy.deepcopy(item)

    if "multi_hop_qa_pair" in item:
        qa = item["multi_hop_qa_pair"]
    else:
        qa = item

    for source in qa.get("sources", []):
        old_id = source.get("id", "")
        if old_id:
            source["id"] = normalize_source_id(old_id)

    if "multi_hop_qa_pair" in item and "sources" in item:
        for source in item.get("sources", []):
            old_id = source.get("id", "")
            if old_id:
                source["id"] = normalize_source_id(old_id)

    top_level_id_fields = ["source_video_id", "source_image_id", "source_audio_id", "id"]
    for obj in ([item, qa] if item is not qa else [item]):
        for field in top_level_id_fields:
            if field in obj and obj[field]:
                obj[field] = normalize_source_id(obj[field])

    def replace_ids_in_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        def sub_func(match):
            content = match.group(1)
            new_id = normalize_source_id(content)
            return f"<{new_id}>"

        return re.sub(r'<([^>]+)>', sub_func, text)

    text_fields = ["question", "answer", "exploration_summary", "question_construction_thought"]
    for field in text_fields:
        if field in qa and qa[field]:
            qa[field] = replace_ids_in_text(qa[field])

    if "annotated_solution" in qa and isinstance(qa["annotated_solution"], list):
        qa["annotated_solution"] = [
            replace_ids_in_text(step) if isinstance(step, str) else step
            for step in qa["annotated_solution"]
        ]

    if "reasoning_chain" in qa and isinstance(qa["reasoning_chain"], list):
        qa["reasoning_chain"] = [
            replace_ids_in_text(step) if isinstance(step, str) else step
            for step in qa["reasoning_chain"]
        ]

    return item


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_file(base_dirs: Union[str, List[str]], basename: str, possible_exts: List[str]) -> bool:
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    for base_dir in base_dirs:
        if os.path.exists(os.path.join(base_dir, basename)):
            return True
        for ext in possible_exts:
            if os.path.exists(os.path.join(base_dir, basename + ext)):
                return True
    return False


def verify_sources_exist(qa: Dict[str, Any]) -> bool:
    sources = qa.get("sources", [])
    if not sources:
        omni_input = qa.get("omni_modal_input", [])
        sources = [{"id": m.get("id", ""), "type": m.get("type", "")} for m in omni_input]

    for source in sources:
        sid = source.get("id", "")
        if not sid:
            continue

        modality, basename = parse_source_id(sid)

        if modality == 'video':
            video_dirs = BASE_DIRS['video']
            if isinstance(video_dirs, str):
                video_dirs = [video_dirs]
            if not check_file(video_dirs, basename, EXTENSIONS['video']):
                return False

        elif modality == 'audio':
            audio_dirs = BASE_DIRS['audio']
            if isinstance(audio_dirs, str):
                audio_dirs = [audio_dirs]
            if not check_file(audio_dirs, basename, EXTENSIONS['audio']):
                return False

        elif modality == 'image':
            image_dirs = BASE_DIRS['image']
            if isinstance(image_dirs, str):
                image_dirs = [image_dirs]
            if not check_file(image_dirs, basename, EXTENSIONS['image']):
                return False

    return True


def _extract_central_value(answer_str: str) -> Optional[float]:
    """Extract the first numeric value from an answer string for range verification."""
    if not answer_str or not isinstance(answer_str, str):
        return None
    # Match first decimal or integer number in the string
    m = re.search(r'[-+]?\d+(?:\.\d+)?', answer_str)
    return float(m.group()) if m else None


def _check_answer_range(qa: Dict[str, Any]) -> bool:
    """If answer_range is present, verify the central answer value falls within it.

    Returns True if check passes (or no answer_range present).
    """
    answer_range = qa.get("answer_range")
    if not answer_range:
        return True
    if not (isinstance(answer_range, list) and len(answer_range) == 2):
        return True  # malformed range — don't discard, let quality check handle it
    try:
        lo, hi = float(answer_range[0]), float(answer_range[1])
    except (TypeError, ValueError):
        return True

    central = _extract_central_value(qa.get("answer", ""))
    if central is None:
        return True  # can't parse — don't penalise
    return lo <= central <= hi


def _has_ambiguous_object_ref(qa: Dict[str, Any]) -> bool:
    """Return True if question or answer contains placeholder object labels like 'object A'."""
    import re
    text = (qa.get("question", "") + " " + qa.get("answer", "")).lower()
    # Match "object A", "object B", "object_1", "object 1" etc.
    return bool(re.search(r'\bobject[\s_][a-z0-9]\b', text))


def is_valid_item(item: Dict[str, Any]) -> bool:
    qa = item.get("multi_hop_qa_pair", item)

    if qa.get("is_question_understandable") != "yes":
        return False

    if qa.get("is_answer_correct_and_unique") != "yes":
        return False

    if qa.get("answerable_without_visual_and_audio_content") != "no":
        return False

    modality = item.get("primary_modality", "")
    if modality != "video":
        if qa.get("answerable_without_visual_content") != "no":
            return False
        if qa.get("answerable_without_audio_content") != "no":
            return False

    if "answerable_without_tools" in qa:
        if qa.get("answerable_without_tools") != "no":
            return False

    if _has_ambiguous_object_ref(qa):
        return False

    if not _check_answer_range(qa):
        return False

    if not verify_sources_exist(qa):
        return False

    # Quality check: reject poor/fair quality
    if qa.get("overall_quality") in ["poor", "fair"]:
        return False
    if qa.get("multi_hop_reasoning_quality") in ["poor", "fair"]:
        return False

    return True


def get_category(item: Dict[str, Any]) -> str:
    qa = item.get("multi_hop_qa_pair", item)
    sources = qa.get("sources", [])

    source_types = set(s.get("type", "") for s in sources)
    if "video" in source_types or "related_video" in source_types:
        return "video"

    for source in sources:
        sid = source.get("id", "")
        modality, _ = parse_source_id(sid)
        if modality == "video":
            return "video"

    omni_input = qa.get("omni_modal_input", [])
    for m in omni_input:
        if m.get("type") == "video":
            return "video"

    primary = item.get("primary_modality", "")
    if "video" in primary:
        return "video"

    return "image_audio"


def main():
    parser = argparse.ArgumentParser(description="Step 4c: Filter Valid QA Items")
    parser.add_argument("--input", nargs="+", required=True, help="Input checked JSON files")
    parser.add_argument("--output_dir", default="./filtered_output")
    parser.add_argument("--video_base_dirs", nargs="+", default=None)
    parser.add_argument("--audio_base_dirs", nargs="+", default=None)
    parser.add_argument("--image_base_dirs", nargs="+", default=None)
    args = parser.parse_args()

    if args.video_base_dirs:
        BASE_DIRS['video'] = args.video_base_dirs
    if args.audio_base_dirs:
        BASE_DIRS['audio'] = args.audio_base_dirs
    if args.image_base_dirs:
        BASE_DIRS['image'] = args.image_base_dirs

    os.makedirs(args.output_dir, exist_ok=True)

    valid_video_items = []
    valid_image_audio_items = []

    total_processed = 0
    total_valid = 0

    for path in args.input:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        print(f"Processing {path}...")
        try:
            items = load_json(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        if not isinstance(items, list):
            items = [items]

        for item in items:
            total_processed += 1

            normalized = normalize_item_sources(item)

            if is_valid_item(normalized):
                total_valid += 1
                category = get_category(normalized)
                if category == "video":
                    valid_video_items.append(normalized)
                else:
                    valid_image_audio_items.append(normalized)

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if valid_video_items:
        out_path = os.path.join(args.output_dir, f"valid_video_{len(valid_video_items)}_{timestamp}.json")
        save_json(valid_video_items, out_path)
        print(f"Saved {len(valid_video_items)} valid video items to {out_path}")

    if valid_image_audio_items:
        out_path = os.path.join(args.output_dir, f"valid_image_audio_{len(valid_image_audio_items)}_{timestamp}.json")
        save_json(valid_image_audio_items, out_path)
        print(f"Saved {len(valid_image_audio_items)} valid image-audio items to {out_path}")

    print(f"\nFilter Summary:")
    print(f"  Total processed: {total_processed}")
    print(f"  Total valid:     {total_valid} ({total_valid/total_processed*100:.1f}% pass rate)")
    print(f"  Video items:     {len(valid_video_items)}")
    print(f"  Image-Audio:     {len(valid_image_audio_items)}")


if __name__ == "__main__":
    main()
