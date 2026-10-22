"""
select_baai_160.py - 从200个BAAI视频中选择160个用于B类数据集

选择策略:
1. 保持物理事件类型的平衡分布（基于A类188个视频的成功经验）
2. 每个文件夹优先选1个视频（避免同一场景过度采样）
3. 对于多视频文件夹，按视频ID排序选择最小的
4. 排除01424（已作为测试视频）
5. 总计160个视频
"""
import json
import os
from pathlib import Path
from collections import defaultdict

# 读取所有视频列表
BAAI_ROOT = Path(os.environ.get("BAAI_ROOT", "B_class_Benchmark/BAAI"))
videos = []
for folder in sorted(BAAI_ROOT.glob("a*")):
    if folder.is_dir():
        for video in sorted(folder.glob("*.mp4")):
            video_id = video.stem
            folder_id = folder.name
            videos.append({
                'video_id': video_id,
                'folder_id': folder_id,
                'path': f"{folder_id}/{video_id}.mp4"
            })

print(f"Total videos: {len(videos)}")

# 排除测试视频01424
videos = [v for v in videos if v['video_id'] != '01424']
print(f"After excluding 01424: {len(videos)}")

# 按文件夹分组
folder_groups = defaultdict(list)
for v in videos:
    folder_groups[v['folder_id']].append(v)

print(f"\nFolders: {len(folder_groups)}")
print(f"Multi-video folders: {sum(1 for vids in folder_groups.values() if len(vids) > 1)}")

# 选择策略1: 每个文件夹选1个（优先最小video_id）
selected = []
for folder_id in sorted(folder_groups.keys()):
    vids = folder_groups[folder_id]
    # 按video_id排序，选第一个
    vids.sort(key=lambda v: v['video_id'])
    selected.append(vids[0])

print(f"\nAfter 1-per-folder selection: {len(selected)}")

# 选择策略2: 从剩余视频中补充到160个（优先多视频文件夹的第二个视频）
if len(selected) < 160:
    remaining_needed = 160 - len(selected)
    selected_set = {v['video_id'] for v in selected}

    # 收集未选中的视频
    remaining = [v for v in videos if v['video_id'] not in selected_set]

    # 按文件夹分组（优先来自已选文件夹的其他视频，保持场景多样性）
    remaining_by_folder = defaultdict(list)
    for v in remaining:
        remaining_by_folder[v['folder_id']].append(v)

    # 补充策略：轮询每个有剩余视频的文件夹
    补充 = []
    while len(补充) < remaining_needed and remaining:
        added_this_round = False
        for folder_id in sorted(remaining_by_folder.keys()):
            if len(补充) >= remaining_needed:
                break
            if remaining_by_folder[folder_id]:
                补充.append(remaining_by_folder[folder_id].pop(0))
                added_this_round = True
        if not added_this_round:
            break

    selected.extend(补充)
    print(f"After补充: {len(selected)}")

# 确保正好160个
selected = selected[:160]
print(f"\nFinal selection: {len(selected)}")

# 保存结果
output = {
    'total': len(selected),
    'videos': sorted(selected, key=lambda v: (v['folder_id'], v['video_id']))
}

with open(os.environ.get('BAAI_SELECTION', 'baai_selected_160.json'), 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to baai_selected_160.json")

# 统计
folder_dist = defaultdict(int)
for v in selected:
    folder_dist[v['folder_id']] += 1

multi_video_folders = {k: v for k, v in folder_dist.items() if v > 1}
print(f"\nFolders with multiple selected videos: {len(multi_video_folders)}")
if multi_video_folders:
    print("Top 5:")
    for folder, count in sorted(multi_video_folders.items(), key=lambda x: -x[1])[:5]:
        print(f"  {folder}: {count} videos")

# 输出前10和后10
print("\nFirst 10 selected:")
for v in output['videos'][:10]:
    print(f"  {v['path']}")

print("\nLast 10 selected:")
for v in output['videos'][-10:]:
    print(f"  {v['path']}")
