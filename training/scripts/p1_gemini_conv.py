"""Gemini forward-solve trajectories (teacher_full.py, data/teacher_full/gemini_k4.json) -> native Qwen3.5 SFT
samples, ONLY for train QAs that no self-sample solved (p1_ss + p1_ss2); in the paper run this adds 15 A + 16 B QAs.
Per turn: <reasoning> block -> <think>, JSON <tool_call>{"name","arguments"} -> native <function=..><parameter=..>.
Keep: ACC >= .9, <= 6 rounds, >= 1 tool call, final <answer>, no web_search query that copies the question wording
(token overlap with the question > .6). Up to --per_qa per QA, highest SCS first.
Output data/p1/<name>.json in the p1_rft_build layout (run p1_rewrite.py on it afterwards for the Step chain).
Usage (from the work dir $TRIPHYS_WORK): python scripts/p1_gemini_conv.py --name gem_only [--per_qa 2]"""
from srvpaths import EVAL_LIB, SCRIPTS  # machine paths, see srvpaths.py
import argparse, json, re, sys
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, EVAL_LIB)
import p1_rft_build as B
from prompts import SYSTEM_PROMPT, build_user_prompt

E = B.E


def toks(s):
    return set(re.findall(r'[a-z0-9.]+', s.lower()))


def native_call(raw):
    try:
        tc = json.loads(raw, strict=False)
    except Exception:
        tc = json.loads(re.sub(r'\]\s*$', '', raw), strict=False)
    args = tc.get('arguments') or {}
    return '<tool_call>\n<function=%s>\n%s\n</function>\n</tool_call>' % (tc['name'], '\n'.join(
        '<parameter=%s>\n%s\n</parameter>' % (k, v if isinstance(v, str) else json.dumps(v)) for k, v in args.items()))


def convert_turn(c):
    if '</reasoning>' in c:
        th, rest = c.split('</reasoning>', 1)
    else:
        m = re.search(r'<tool_call>|<answer>', c)
        th, rest = (c[:m.start()], c[m.start():]) if m else (c, '')
    th = th.replace('<reasoning>', '').strip()
    calls = re.findall(r'<tool_call>(.*?)</tool_call>', rest, re.DOTALL)
    if calls:
        body = '\n'.join(native_call(x.strip()) for x in calls)
    else:
        a = re.search(r'<answer>.*?</answer>', rest, re.DOTALL)
        body = a.group(0) if a else rest.strip()
    return '<think>\n%s\n</think>\n\n%s' % (th, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--per_qa', type=int, default=2)
    a = ap.parse_args()
    idx = B.train_index()
    solved = {x['question'] for x in B.load_rows(['p1_ss', 'p1_ss2']) if B.ok(x)}
    by = {}
    for r in json.load(open(E + '/data/teacher_full/gemini_k4.json')):
        q = r['question']
        if q in solved or q not in idx or float(r['ACC']) < .9 or r['rounds'] > 6:
            continue
        t = r['turns']
        if not t or t[-1]['role'] != 'assistant' or '<answer>' not in t[-1]['content'] or not r['tool_log']:
            continue
        qt = toks(q)
        if any(x['tool'] == 'web_search' and len(toks(str(x['args'].get('query', ''))) & qt) /
               max(1, len(toks(str(x['args'].get('query', ''))))) > .6 and len(toks(str(x['args'].get('query', '')))) > 8
               for x in r['tool_log']):
            continue
        by.setdefault(q, []).append(r)
    out = []
    for q, v in by.items():
        cls, vpath, qa = idx[q]
        for r in sorted(v, key=lambda r: -float(r['SCS']))[:a.per_qa]:
            try:
                turns = [{'role': t['role'], 'content': convert_turn(t['content']) if t['role'] == 'assistant'
                          else t['content']} for t in r['turns']]
            except Exception as e:
                print('skip', e)
                continue
            msgs = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': [{'type': 'video', 'video': vpath, 'fps': 1.0},
                                                 {'type': 'text', 'text': build_user_prompt(q, qa['question_type'])}]}]
            out.append({'messages': msgs + turns, 'metadata': {
                'cls': cls, 'question_type': qa['question_type'], 'difficulty': qa.get('difficulty'), 'rep': r['k'],
                'ACC': float(r['ACC']), 'SCS': float(r['SCS']), 'source': 'gemini_teacher', 'format': 'n1_native'}})
    json.dump(out, open('%s/data/p1/%s.json' % (E, a.name), 'w'), ensure_ascii=False, indent=1)
    print('QAs', len(by), 'samples', len(out), 'A', sum(o['metadata']['cls'] == 'A' for o in out),
          'B', sum(o['metadata']['cls'] == 'B' for o in out))
    print(out[0]['messages'][2]['content'][-600:] if out else '')


if __name__ == '__main__':
    main()
