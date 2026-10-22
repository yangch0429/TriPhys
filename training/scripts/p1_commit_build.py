"""10/17 commit continuations (p1_commit.py, results/p1_commit/shard_*.json) -> SFT samples in the p1_rft_build layout.
Keep ACC >= .9, answered, no leak; up to --per_qa per QA, highest SCS first. turns are stored without the note.
Run p1_rewrite.py (Flash-gated) on the output afterwards.
Usage (from the work dir $TRIPHYS_WORK): python scripts/p1_commit_build.py --name commit_k2 [--per_qa 2]"""
from srvpaths import EVAL_LIB, SCRIPTS  # machine paths, see srvpaths.py
import argparse, glob, json, sys
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, EVAL_LIB)
import p1_rft_build as B
from prompts import SYSTEM_PROMPT, build_user_prompt

E = B.E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--per_qa', type=int, default=2)
    a = ap.parse_args()
    idx = B.train_index()
    by = {}
    for f in glob.glob(E + '/results/p1_commit/shard_*.json'):
        for r in json.load(open(f)):
            if r['ACC'] >= .9 and r['answered'] and not r['leak'] and r['question'] in idx:
                by.setdefault(r['question'], []).append(r)
    out = []
    for q, v in by.items():
        cls, vpath, qa = idx[q]
        for r in sorted(v, key=lambda r: -r['SCS'])[:a.per_qa]:
            msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': [{'type': 'video', 'video': vpath, 'fps': 1.0},
                                                 {'type': 'text', 'text': build_user_prompt(q, qa['question_type'])}]}]
            out.append({'messages': msgs + [{'role': t['role'], 'content': t['content']} for t in r['turns']],
                        'metadata': {'cls': cls, 'question_type': qa['question_type'], 'difficulty': qa.get('difficulty'),
                                     'ACC': r['ACC'], 'SCS': r['SCS'], 'source': 'p1_commit', 'format': 'n1_native'}})
    json.dump(out, open('%s/data/p1/%s.json' % (E, a.name), 'w'), ensure_ascii=False, indent=1)
    print('QAs', len(by), 'samples', len(out), 'A', sum(o['metadata']['cls'] == 'A' for o in out),
          'B', sum(o['metadata']['cls'] == 'B' for o in out))


if __name__ == '__main__':
    main()
