"""Video-level train/test partition, run separately for MPR and KPR.

1. Questions are grouped by source video. Each video gets its modal question type, difficulty and parent physics
   category (ties broken alphabetically); the triple defines its stratum.
2. A stratum with fewer than three videos is merged into a stratum with the same type and difficulty, otherwise into
   one with the same type; if neither exists it keeps its identity.
3. Within each stratum the videos are shuffled with the seed (default 42) and 30% (at least one, at most all but one)
   go to the test side.
4. Greedy exchange: single train/test video swaps are applied while they reduce the deviation of the test side's
   question-level counts (type, difficulty, parent category, type x difficulty) from the test share, keeping at
   least one video on each side of every stratum that has two or more videos.

Usage:
  python split_videos.py mpr aclass_qa_dataset.json   --out_dir splits/      # list of {take_uid, ..., qa_pairs}
  python split_videos.py kpr bclass_qa_dataset.json   --out_dir splits/      # {"qa_pairs": [...]}
Writes <sub>_train.json / <sub>_test.json (MPR: video entries; KPR: [{folder_id, video_id, qa_pairs}]).
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

TEST_SHARE = 0.30
MIN_STRATUM = 3


def physics_parent(cat):
    return (cat or '').split('/')[0].strip() or 'Other'


def mode(xs):
    c = Counter(xs)
    m = max(c.values())
    return sorted(x for x in c if c[x] == m)[0]


def load(sub, path):
    data = json.load(open(path, encoding='utf-8'))
    if sub == 'mpr':
        entries = {v['take_uid']: v for v in data}
        return {k: v['qa_pairs'] for k, v in entries.items()}, entries
    groups = defaultdict(list)
    for q in data['qa_pairs']:
        groups['%s/%s' % (q['folder_id'], q['video_id'])].append(q)
    entries = {k: {'folder_id': k.split('/')[0], 'video_id': k.split('/')[1], 'qa_pairs': v} for k, v in groups.items()}
    return dict(groups), entries


def merge_rare(strata):
    merged = {s: list(v) for s, v in strata.items() if len(v) >= MIN_STRATUM}
    for s, vids in strata.items():
        if len(vids) >= MIN_STRATUM:
            continue
        match = next((c for c in merged if c[:2] == s[:2]), None) or next((c for c in merged if c[0] == s[0]), None)
        merged.setdefault(match or s, []).extend(vids)
    return merged


DIMS = [lambda q: q['question_type'], lambda q: q['difficulty'], lambda q: physics_parent(q.get('physics_category')),
        lambda q: (q['question_type'], q['difficulty'])]


def deviation(V, T, share, totals):
    test = [Counter(f(q) for k in T for q in V[k]) for f in DIMS]
    return sum(abs(test[i][c] - share * totals[i][c]) for i in range(len(DIMS)) for c in totals[i])


def split(V, seed=42, share=TEST_SHARE):
    strata = defaultdict(list)
    for k in sorted(V):
        qs = V[k]
        strata[(mode([q['question_type'] for q in qs]), mode([q['difficulty'] for q in qs]),
                mode([physics_parent(q.get('physics_category')) for q in qs]))].append(k)
    merged = merge_rare(dict(sorted(strata.items())))
    rnd, T, of = random.Random(seed), set(), {}
    for s, vids in sorted(merged.items()):
        vids = sorted(vids)
        for k in vids:
            of[k] = s
        rnd.shuffle(vids)
        n = min(max(1, int(len(vids) * share)), len(vids) - 1) if len(vids) > 1 else 0
        T |= set(vids[:n])
    size = Counter(of.values())
    totals = [Counter(f(q) for k in V for q in V[k]) for f in DIMS]

    def ok(S):
        side = Counter(of[k] for k in S)
        return all(0 < side[s] < size[s] for s in size if size[s] >= 2)

    best = deviation(V, T, share, totals)
    while True:
        cand = None
        for a in sorted(T):
            for b in sorted(set(V) - T):
                S = (T - {a}) | {b}
                d = deviation(V, S, share, totals)
                if d < best - 1e-9 and (cand is None or d < cand[0]) and ok(S):
                    cand = (d, S)
        if cand is None:
            return T
        best, T = cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subtask', choices=['mpr', 'kpr'])
    ap.add_argument('dataset')
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    V, entries = load(a.subtask, a.dataset)
    T = split(V, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, keys in (('train', sorted(set(V) - T)), ('test', sorted(T))):
        with open(os.path.join(a.out_dir, '%s_%s.json' % (a.subtask, name)), 'w', encoding='utf-8') as f:
            json.dump([entries[k] for k in keys], f, ensure_ascii=False, indent=2)
        print('%s %s: %d videos, %d questions' % (a.subtask, name, len(keys), sum(len(V[k]) for k in keys)))


if __name__ == '__main__':
    main()
