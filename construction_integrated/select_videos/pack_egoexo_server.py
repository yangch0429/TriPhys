"""
Pack selected EgoExo4D videos on CTI server.
Run: EGOEXO_RAW_ROOT=... EGOEXO_SELECTION=... EGOEXO_SELECTED_ROOT=... python pack_egoexo_server.py

Output structure:
  egoexo4d_selected/
    manifest.json
    takes/
      {take_uid}/
        best_exo.mp4
    annotations/
      {take_uid}.json   <- ego_pose GT body annotation
"""

import json
import os
import shutil
import subprocess
import sys

# ── Paths (from the environment) ─────────────────────────────────────────────
EGOEXO_ROOT      = os.environ["EGOEXO_RAW_ROOT"]
SELECTED_JSON    = os.environ.get("EGOEXO_SELECTION", "selected_egoexo200.json")
OUT_DIR          = os.path.abspath(os.environ.get("EGOEXO_SELECTED_ROOT", "egoexo4d_selected"))
ANNO_SRC_ROOT    = os.path.join(EGOEXO_ROOT, "annotations", "ego_pose", "train", "body", "annotation")

# ── Load selected takes ────────────────────────────────────────────────────────
def load_selected():
    with open(SELECTED_JSON, encoding="utf-8") as f:
        return json.load(f)

# ── Find best exo cam for a take ───────────────────────────────────────────────
def pick_best_exo_cam_by_size(take_dir: str):
    """Fallback: return (cam_name, cam_path) for the largest cam*.mp4."""
    frame_aligned = os.path.join(take_dir, "frame_aligned_videos")
    if not os.path.isdir(frame_aligned):
        return None, None
    cams = [
        f for f in os.listdir(frame_aligned)
        if f.startswith("cam") and f.endswith(".mp4")
    ]
    if not cams:
        return None, None
    best = max(cams, key=lambda f: os.path.getsize(os.path.join(frame_aligned, f)))
    return best.replace(".mp4", ""), os.path.join(frame_aligned, best)

def pick_cam(take_dir: str, best_exo_from_json: str):
    """
    Select the best exo camera:
    1. Use best_exo field from selected_egoexo200.json (official EgoExo4D annotation)
    2. Fallback to largest cam*.mp4 by file size
    """
    fa = os.path.join(take_dir, "frame_aligned_videos")
    if best_exo_from_json and os.path.isdir(fa):
        preferred = os.path.join(fa, f"{best_exo_from_json}.mp4")
        if os.path.exists(preferred):
            return best_exo_from_json, preferred
    # Fallback
    return pick_best_exo_cam_by_size(take_dir)

# ── Main packaging ─────────────────────────────────────────────────────────────
def main():
    selected = load_selected()
    print(f"Loaded {len(selected)} selected takes from {SELECTED_JSON}")

    os.makedirs(os.path.join(OUT_DIR, "takes"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "annotations"), exist_ok=True)

    manifest = {
        "version": "1.0",
        "description": "OmniGAIA EgoExo4D selected takes — best exo cam per take + GT ego_pose annotations",
        "takes": []
    }

    skipped = []
    for entry in selected:
        take_uid   = entry["take_uid"]
        take_name  = entry.get("take_name", "")
        category   = entry.get("scenario", entry.get("category", ""))
        best_exo_from_json = entry.get("best_exo", "")

        take_dir = os.path.join(EGOEXO_ROOT, "takes", take_name)

        # ── Find best cam (official best_exo field first, size fallback) ───────
        cam_name, cam_src = pick_cam(take_dir, best_exo_from_json)
        if cam_src is None:
            print(f"  SKIP (no cam): {take_name}")
            skipped.append({"take_uid": take_uid, "reason": "no_cam_file"})
            continue

        # ── Copy video ─────────────────────────────────────────────────────────
        vid_out_dir = os.path.join(OUT_DIR, "takes", take_uid)
        os.makedirs(vid_out_dir, exist_ok=True)
        vid_dst = os.path.join(vid_out_dir, "best_exo.mp4")
        if not os.path.exists(vid_dst):
            shutil.copy2(cam_src, vid_dst)
            size_mb = os.path.getsize(vid_dst) / 1e6
            print(f"  OK  {take_name:40s} {cam_name}  {size_mb:.1f} MB")
        else:
            size_mb = os.path.getsize(vid_dst) / 1e6
            print(f"  SKIP(exists) {take_name:36s} {cam_name}  {size_mb:.1f} MB")

        # ── Copy annotation ───────────────────────────────────────────────────
        anno_src = os.path.join(ANNO_SRC_ROOT, f"{take_uid}.json")
        anno_dst = os.path.join(OUT_DIR, "annotations", f"{take_uid}.json")
        has_anno = False
        if os.path.exists(anno_src) and not os.path.exists(anno_dst):
            shutil.copy2(anno_src, anno_dst)
            has_anno = True
        elif os.path.exists(anno_dst):
            has_anno = True
        else:
            print(f"    WARNING: no GT annotation for {take_uid}")

        manifest["takes"].append({
            "take_uid":          take_uid,
            "take_name":         take_name,
            "category":          category,
            "best_exo_cam":      cam_name,
            "video_path":        f"takes/{take_uid}/best_exo.mp4",
            "gt_annotation_path": f"annotations/{take_uid}.json" if has_anno else None,
        })

    # ── Write manifest ─────────────────────────────────────────────────────────
    manifest["total"]   = len(manifest["takes"])
    manifest["skipped"] = skipped
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written: {manifest_path}")
    print(f"Total packed: {manifest['total']},  Skipped: {len(skipped)}")

    # ── Compute total size ─────────────────────────────────────────────────────
    total_bytes = 0
    for root, dirs, files in os.walk(OUT_DIR):
        for fname in files:
            total_bytes += os.path.getsize(os.path.join(root, fname))
    print(f"Total size: {total_bytes/1e9:.2f} GB")

    # ── Create zip archive ─────────────────────────────────────────────────────
    zip_path = OUT_DIR + ".zip"
    if not os.path.exists(zip_path):
        print(f"\nCreating zip archive: {zip_path}")
        print("(This may take 5-15 minutes for ~14 GB of video)")
        shutil.make_archive(
            base_name=OUT_DIR,
            format="zip",
            root_dir=os.path.dirname(OUT_DIR),
            base_dir=os.path.basename(OUT_DIR),
        )
        zip_size = os.path.getsize(zip_path) / 1e9
        print(f"Archive created: {zip_path}  ({zip_size:.2f} GB)")
    else:
        zip_size = os.path.getsize(zip_path) / 1e9
        print(f"Archive already exists: {zip_path}  ({zip_size:.2f} GB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
