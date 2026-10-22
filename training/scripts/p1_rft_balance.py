"""10/16 Phase1 RFT balanced: subsample B-class to match A count (base self-sampling had many more B successes)."""
import argparse, json, random, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20261016)
    a = ap.parse_args()
    data = json.load(open(a.input))
    rng = random.Random(a.seed)
    by_cls = {'A': [], 'B': []}
    for x in data:
        by_cls[x['metadata']['cls']].append(x)
    n = min(len(by_cls['A']), len(by_cls['B']))
    print(f'Input: {len(by_cls["A"])} A, {len(by_cls["B"])} B → balanced at {n} each', file=sys.stderr)
    for c in 'AB':
        rng.shuffle(by_cls[c])
    out = by_cls['A'][:n] + by_cls['B'][:n]
    rng.shuffle(out)
    json.dump(out, open(a.output, 'w'), ensure_ascii=False, indent=1)
    rep = {}
    for c in 'AB':
        s = [x for x in out if x['metadata']['cls'] == c]
        rep[f'{c}_n'] = len(s)
        rep[f'{c}_scs'] = round(sum(x['metadata']['SCS'] for x in s) / len(s), 3) if s else 0
        rep[f'{c}_qas'] = len({json.dumps(x['messages'][1]['content'][1]['text']) for x in s})
    json.dump(rep, open(a.output.replace('.json', '_report.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))

if __name__ == '__main__':
    main()
