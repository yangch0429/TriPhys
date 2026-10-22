# Provenance (TriPhysBench release): fix_code/_v8_work/v8_test_code_check.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""
Statistics only (no data change): do the TEST split's code_executor sources run and reproduce the answer?
Same checks as the v8 train code stage: execute each block (eval-time environment); the last block must print the
answer's value (or a value inside a range stated in the answer), earlier blocks a solution value. Failing QAs get one
LLM diagnosis: code_error (code wrong) vs answer_inconsistent (the solution's own inputs cannot give the answer).
Output: data/v8_test_code_check.json
Usage: python v8_test_code_check.py
"""
import json, re, subprocess, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from v8_common import DATA, load_bench, call_llm

RUNNER = open(__file__.replace('v8_test_code_check.py', 'v8_assemble.py'), encoding='utf-8').read()
RUNNER = re.search(r"RUNNER = r'''(.*?)'''", RUNNER, re.S).group(1)  # identical executor to the train stage
RANGE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)')
SCALES = (1, 1e3, 1e-3, 1e6, 1e-6)


def run_code(code):
    try:
        r = subprocess.run([sys.executable, '-c', RUNNER], input=code, capture_output=True, text=True, timeout=20)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as ex:
        return {'ok': False, 'out': 'Error: %s' % ex}


def text_nums(s):
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s or '')
    return [float(x) for x in re.findall(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', s)]


def consistent(out, nums):
    cand = [abs(x) for x in text_nums(out) if abs(x) > 1e-12]
    return any(abs(x * s - y) <= 0.02 * max(x * s, y) for x in cand for y in nums if y > 0 for s in SCALES)


def in_answer_range(out, answer):
    rngs = [(min(a, b), max(a, b)) for a, b in ((float(x), float(y)) for x, y in RANGE_RE.findall(re.sub(r'(?<=\d),(?=\d{3}\b)', '', answer or '')))
            if a != b]
    cand = [abs(x) for x in text_nums(out) if abs(x) > 1e-12]
    return any(lo * 0.98 <= x * s <= hi * 1.02 for lo, hi in rngs for x in cand for s in SCALES)


def steps(qa):
    sol = qa.get('annotated_solution')
    return [str(x) for x in sol] if isinstance(sol, list) else [str(sol)]


DIAG = """A physics QA's code step does not reproduce the final answer: {problem}.
Decide: "code_error" (the code is wrong: syntax, pseudo-code, wrong formula/inputs, unit slip, missing print) or
"answer_inconsistent" (computing the solution's own formulas with its own input values cannot give the stated answer).
Return ONLY JSON: {{"verdict": "code_error" | "answer_inconsistent", "why": "<1-2 sentences>"}}

Question: {question}
Solution:
{sol}
Final answer: {answer}
Code:
{code}"""


def check(b):
    qa = b['qa']
    ans = str(qa.get('answer', ''))
    anums = [x for x in text_nums(ans) if x > 0]
    snums = text_nums(' '.join(steps(qa)) + ' ' + ans)
    codes = [s.get('code') or '' for s in qa.get('sources', []) if s.get('type') == 'code_executor']
    rec = {'qid': b['qid'], 'cls': b['cls'], 'n_code': len(codes), 'blocks': []}
    for i, c in enumerate(codes):
        is_last = i == len(codes) - 1
        r = run_code(c)
        if not r['ok']:
            prob = 'runtime_error'
        elif r['out'] == 'No output':
            prob = 'no_output'
        elif is_last and anums and not (consistent(r['out'], anums) or in_answer_range(r['out'], ans)):
            prob = 'answer_mismatch'
        elif not is_last and snums and not consistent(r['out'], snums):
            prob = 'solution_mismatch'
        else:
            prob = ''
        rec['blocks'].append({'problem': prob, 'out': r['out'][:200]})
    probs = [x['problem'] for x in rec['blocks'] if x['problem']]
    rec['status'] = 'ok' if not probs else probs[0]
    if probs:
        k = next(i for i, x in enumerate(rec['blocks']) if x['problem'])
        d = call_llm(DIAG.format(problem='%s (output: %s)' % (probs[0], rec['blocks'][k]['out']), question=qa.get('question', '')[:1500],
                                 sol='\n'.join(steps(qa))[:4000], answer=ans, code=codes[k])) or {}
        rec['verdict'], rec['why'] = d.get('verdict'), d.get('why')
    return rec


bench = [b for b in load_bench('test')]
with ThreadPoolExecutor(4) as ex:
    recs = list(ex.map(check, bench))
json.dump(recs, open(DATA + '/v8_test_code_check.json', 'w'), ensure_ascii=False, indent=1)
for cls in 'AB':
    rs = [r for r in recs if r['cls'] == cls]
    print('%s test QAs=%d | with code=%d | status %s | verdicts %s' % (
        cls, len(rs), sum(1 for r in rs if r['n_code']), dict(Counter(r['status'] for r in rs if r['n_code'])),
        dict(Counter(r.get('verdict') for r in rs if r['status'] != 'ok' and r['n_code']))))
