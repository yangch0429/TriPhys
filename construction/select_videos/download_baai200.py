"""
download_baai200.py - download the recordings listed in $BAAI_CANDIDATES from BAAI-DataCube/Physics-aware-videos
into $BAAI_ROOT/<folder_id>/<video_id>.mp4, skipping existing files.
"""
import json
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, HfFileSystem

SEL_JSON = os.environ.get("BAAI_CANDIDATES", "selected_baai200.json")
OUT_DIR   = os.environ.get("BAAI_ROOT", "B_class_Benchmark/BAAI")
REPO_ID   = "BAAI-DataCube/Physics-aware-videos"

os.makedirs(OUT_DIR, exist_ok=True)

with open(SEL_JSON, encoding="utf-8") as f:
    entries = json.load(f)

print(f"Total to download: {len(entries)}")

# 先用 HfFileSystem 确认视频在repo里的路径前缀
# 视频路径格式: a00651/00103.mp4 -> 在repo里可能是 videos/a00651/00103.mp4
fs = HfFileSystem()
# 探测路径前缀
sample_vid = entries[0]["video"]  # e.g. "a00107/00999.mp4"
candidate_prefixes = ["videos", "data", ""]
PREFIX = None
for pfx in candidate_prefixes:
    test_path = f"datasets/{REPO_ID}/{pfx}/{sample_vid}".replace("//", "/")
    try:
        if fs.exists(test_path):
            PREFIX = pfx
            print(f"Video prefix confirmed: '{PREFIX}'")
            break
    except Exception:
        pass

if PREFIX is None:
    # 列出repo顶层目录辅助诊断
    try:
        top = fs.ls(f"datasets/{REPO_ID}", detail=False)
        print(f"Repo top-level: {top[:20]}")
    except Exception as e:
        print(f"ls failed: {e}")
    print("ERROR: cannot determine video prefix. Exiting.")
    sys.exit(1)

ok = 0
skip = 0
fail = 0
fail_list = []

for i, e in enumerate(entries):
    vid = e["video"]  # e.g. "a00651/00103.mp4"
    rel_path = f"{PREFIX}/{vid}" if PREFIX else vid

    # 本地目标路径：保留目录结构
    local_path = os.path.join(OUT_DIR, vid.replace("/", os.sep))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        skip += 1
        if (i + 1) % 20 == 0:
            print(f"[{i+1:03d}/{len(entries)}] skip={skip} ok={ok} fail={fail}")
        continue

    try:
        dl = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=rel_path,
            local_dir=OUT_DIR,
            local_dir_use_symlinks=False,
        )
        # hf_hub_download 会把文件放到 OUT_DIR/rel_path，需要移到我们要的位置
        # 如果 PREFIX 存在，文件在 OUT_DIR/PREFIX/vid，我们想要 OUT_DIR/vid
        if PREFIX:
            src = os.path.join(OUT_DIR, PREFIX, vid.replace("/", os.sep))
            if os.path.exists(src) and src != local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                os.replace(src, local_path)
        ok += 1
        print(f"[{i+1:03d}/{len(entries)}] OK  {vid}")
    except Exception as ex:
        fail += 1
        fail_list.append(vid)
        print(f"[{i+1:03d}/{len(entries)}] FAIL {vid}: {ex}")

print(f"\nDone. ok={ok}  skip={skip}  fail={fail}")
if fail_list:
    print("Failed videos:")
    for v in fail_list:
        print(f"  {v}")
    # 保存失败列表
    with open("baai200_download_fail.txt", "w") as f:
        f.write("\n".join(fail_list))
    print("Saved fail list -> baai200_download_fail.txt")
