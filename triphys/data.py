"""Benchmark loading. data/{mpr,kpr}_{all,train,test}.json; MPR files are lists of takes with qa_pairs, KPR train/test
are lists of videos with qa_pairs, KPR all is {"qa_pairs": [...]} with folder_id / video_id on each QA.
Each item: {qid, subtask, video, qa}. qid = "<take_uid>|<i>" (MPR) or "<folder_id>/<video_id>|<i>" (KPR), i = index of
the QA within its video."""
import json
import os
from pathlib import Path

from . import config


def mpr_video(take_uid, root=None):
    return os.path.join(root or config.EGOEXO_ROOT, take_uid, 'best_exo.mp4')


def kpr_video(folder_id, video_id, root=None):
    return os.path.join(root or config.BAAI_ROOT, folder_id, '%s.mp4' % video_id)


def load_split(subtask, split, data_dir=None, only_valid=True):
    """subtask MPR | KPR, split all | train | test"""
    path = Path(data_dir or config.DATA_DIR) / ('%s_%s.json' % (subtask.lower(), split))
    raw = json.load(open(path, encoding='utf-8'))
    items = []
    if subtask == 'MPR':
        for take in raw:
            for i, qa in enumerate(take['qa_pairs']):
                items.append({'qid': '%s|%d' % (take['take_uid'], i), 'subtask': 'MPR',
                              'video': mpr_video(take['take_uid']), 'qa': qa})
    else:
        groups = raw['qa_pairs'] if isinstance(raw, dict) else [q for v in raw for q in v['qa_pairs']]
        cnt = {}
        for qa in groups:
            vid = '%s/%s' % (qa['folder_id'], qa['video_id'])
            cnt[vid] = cnt.get(vid, -1) + 1
            items.append({'qid': '%s|%d' % (vid, cnt[vid]), 'subtask': 'KPR',
                          'video': kpr_video(qa['folder_id'], qa['video_id']), 'qa': qa})
    return [x for x in items if x['qa'].get('valid', True) or not only_valid]


def load(subtasks=('MPR', 'KPR'), split='test', data_dir=None):
    return [x for s in subtasks for x in load_split(s, split, data_dir)]


def subtasks_arg(s):
    """CLI helper: MPR, KPR or both"""
    return ('MPR', 'KPR') if s in ('both', 'all') else (s.upper(),)
