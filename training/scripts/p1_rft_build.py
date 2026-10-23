"""Stage-I trajectory data: self-samples on the train QAs (results/<tag>_r{0..7}, eval_n1_sample native tools, T=0.7)
filtered by the GT verifier. System / user turns are rebuilt exactly as eval_n1 builds
them (SYSTEM_PROMPT, video fps 1 + build_user_prompt), assistant / tool turns appended as generated (the chat template
re-renders <think> from the '</think>' in the content).
Keep: ACC >= .9, no Serper failure, final turn has <answer>, at least one tool call; per QA up to --per_qa
trajectories, highest SCS first, SCS >= --min_scs.
Output: data/p1/<name>.json + <name>_report.json; also prints per-QA pass@k stats (A/B).
--tags p1_ss,p1_ss2 pools base-model samples (p1_ss) with samples of the round-2 sampler (p1_ss2); default p1_ss.
Missing reps are skipped; --only FILE (json list of questions) restricts the QAs.
Usage (from the work dir $TRIPHYS_WORK): python scripts/p1_rft_build.py --name rft_k2 --per_qa 2 --min_scs .7 [--tags p1_ss,p1_ss2] [--only data/p1/a_uncovered.json]"""
from srvpaths import BAAI_ROOT, E, EGOEXO_ROOT, EVAL_LIB  # machine paths, see srvpaths.py
import argparse, glob, json, os, sys
from collections import Counter, defaultdict
from statistics import mean
sys.path.insert(0, EVAL_LIB)
from prompts import SYSTEM_PROMPT, build_user_prompt



def train_index():
    """question text -> (cls, video path, qa)"""
    idx = {}
    for it in json.load(open(E + '/data/f10/aclass_qa_train.json')):
        for qa in it['qa_pairs']:
            if qa.get('valid', True):
                idx[qa['question']] = ('A', EGOEXO_ROOT + '/%s/best_exo.mp4' % it['take_uid'], qa)
    for it in json.load(open(E + '/data/f10/bclass_qa_train.json')):
        for qa in it['qa_pairs']:
            if qa.get('valid', True):
                idx[qa['question']] = ('B', BAAI_ROOT + '/%s/%s.mp4' % (it['folder_id'], it['video_id']), qa)
    return idx


def load_rows(tags):
    rows = []
    for tag in tags:
        for r in range(8):
            for f in glob.glob('%s/results/%s_r%d/*_shard*.json' % (E, tag, r)):
                if 'serper_failures' in f:
                    continue
                for x in json.load(open(f)).get('results', []):
                    x['_rep'], x['_src'] = r, tag
                    rows.append(x)
    return rows


def ok(x):
    t = x.get('turns') or []
    return (float(x['ACC']) >= .9 and not x.get('serper_failed') and t and t[-1]['role'] == 'assistant'
            and '<answer>' in t[-1]['content'] and any(y['role'] == 'tool' for y in t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--per_qa', type=int, default=2)
    ap.add_argument('--min_scs', type=float, default=0.0)
    ap.add_argument('--tags', default='p1_ss', help='comma-separated sample dirs results/<tag>_r{0..7}')
    ap.add_argument('--only', default='', help='json list of questions to keep')
    a = ap.parse_args()
    idx, rows = train_index(), load_rows(a.tags.split(','))
    if a.only:
        keep = set(json.load(open(a.only)))
        idx = {k: v for k, v in idx.items() if k in keep}
    by = defaultdict(list)
    for x in rows:
        if x['question'] in idx:
            by[x['question']].append(x)
    rep = Counter(rows=len(rows), qas_sampled=len(by), qas_train=len(idx))
    for c in 'AB':
        q = [v for k, v in by.items() if idx[k][0] == c]
        rep['%s_qas' % c] = len(q)
        if not q:
            continue
        rep['%s_mean_acc' % c] = round(mean(float(x['ACC']) >= .9 for v in q for x in v), 3)
        rep['%s_mean_scs' % c] = round(mean(float(x['SCS']) for v in q for x in v), 3)
        rep['%s_pass@k' % c] = round(mean(any(float(x['ACC']) >= .9 for x in v) for v in q), 3)
        rep['%s_pass@k_kept' % c] = round(mean(any(ok(x) for x in v) for v in q), 3)
    out = []
    for qtext, v in by.items():
        cls, vpath, qa = idx[qtext]
        good = sorted([x for x in v if ok(x) and float(x['SCS']) >= a.min_scs], key=lambda x: -float(x['SCS']))
        for x in good[:a.per_qa]:
            msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': [{'type': 'video', 'video': vpath, 'fps': 1.0},
                                                 {'type': 'text', 'text': build_user_prompt(qtext, qa['question_type'])}]}]
            msgs += [{'role': t['role'], 'content': t['content']} for t in x['turns']]
            out.append({'messages': msgs, 'metadata': {'cls': cls, 'question_type': qa['question_type'],
                        'difficulty': qa.get('difficulty'), 'rep': x['_rep'], 'ACC': float(x['ACC']),
                        'SCS': float(x['SCS']), 'source': x['_src'], 'format': 'n1_native'}})
    for c in 'AB':
        s = [o for o in out if o['metadata']['cls'] == c]
        rep['%s_samples' % c] = len(s)
        rep['%s_qas_kept' % c] = len({json.dumps(o['messages'][1]['content'][1]['text']) for o in s})
        rep['%s_kept_scs' % c] = round(mean(o['metadata']['SCS'] for o in s), 3) if s else 0
        rep['%s_kept_rounds' % c] = round(mean(sum(m['role'] == 'assistant' for m in o['messages']) for o in s), 2) if s else 0
    os.makedirs(E + '/data/p1', exist_ok=True)
    json.dump(out, open('%s/data/p1/%s.json' % (E, a.name), 'w'), ensure_ascii=False, indent=1)
    json.dump(rep, open('%s/data/p1/%s_report.json' % (E, a.name), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
