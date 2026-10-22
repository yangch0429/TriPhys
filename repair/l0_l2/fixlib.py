# Provenance (TriPhysBench release): fix_code/_v8_work/fixlib.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Shared helpers for the fix_code two-level repair (L0 / L1 / L2).
Data: fix_code/{a,b}class_qa_{train,test}.json (unrepaired originals, never overwritten until the final write-back).
Every Serper call is appended to _v8_work/search_log.jsonl (raw results kept)."""
import ast, json, os, re, subprocess, sys, threading, time
import requests

WORK = os.environ.get('V8_WORK_DIR') or os.path.dirname(os.path.abspath(__file__))  # repair state files
FC = os.environ.get('TRIPHYS_WORK_DIR') or os.path.dirname(WORK)  # holds {a,b}class_qa_{train,test}.json
REP = WORK + '/server_reports/'

# L0 / L1 / L2 review / first L1E round: gpt-6-astra via an OpenAI-compatible relay (see v8_common.REVIEWER_*);
# from the L1E rerun on (relay quota exhausted): deepseek-flash (DeepSeek-V4.1-Flash, api.deepseek.com)
LLM_URL = os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com') + '/chat/completions'
LLM_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
LLM_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash')
SERPER_KEYS = [k.strip() for k in os.environ.get('SERPER_API_KEYS', '').split(',') if k.strip()]
_lock = threading.Lock()
_key_idx = [0]
_dead = set()
SEARCH_LOG = WORK + '/search_log.jsonl'


# ---------------------------------------------------------------- data
def load(cls, split, stage=None):
    """stage=None: fix_code originals; otherwise _v8_work/stage_<stage>/"""
    d = FC if stage is None else '%s/stage_%s' % (WORK, stage)
    return json.load(open('%s/%sclass_qa_%s.json' % (d, cls, split), encoding='utf-8'))


def save(data, cls, split, stage):
    d = '%s/stage_%s' % (WORK, stage)
    os.makedirs(d, exist_ok=True)
    json.dump(data, open('%s/%sclass_qa_%s.json' % (d, cls, split), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def vtag(vid):
    return vid.get('take_uid') or '%s/%s' % (vid.get('folder_id'), vid.get('video_id'))


def make_qid(cls, split, vid, i):
    return ('test:' if split == 'test' else '') + '%s|%s|%d' % (cls.upper(), vtag(vid), i)


def iter_qas(data, cls, split):
    for vid in data:
        for i, qa in enumerate(vid['qa_pairs']):
            yield make_qid(cls, split, vid, i), vid, qa


def get_steps(qa):
    """(steps list, kind) kind: list | repr | str"""
    sol = qa.get('annotated_solution')
    if isinstance(sol, list):
        return [str(x) for x in sol], 'list'
    try:
        v = ast.literal_eval(sol)
        if isinstance(v, list):
            return [str(x) for x in v], 'repr'
    except Exception:
        pass
    return [str(sol)], 'str'


def set_steps(qa, steps, kind):
    qa['annotated_solution'] = steps if kind == 'list' else repr(steps) if kind == 'repr' else '\n'.join(steps)


def sources_of(qa, t):
    return [s for s in qa.get('sources', []) if s.get('type') == t]


# ---------------------------------------------------------------- LLM
def parse_json_obj(text):
    text = text or ''
    m = re.search(r'```json\s*(\{.*\})\s*```', text, re.S) or re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1) if m.lastindex else m.group(0))
    except json.JSONDecodeError:
        return None


def call_llm(prompt, timeout=300, retries=4, temperature=0, system=None):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    for attempt in range(retries):
        try:
            r = requests.post(LLM_URL, headers={'Authorization': 'Bearer ' + LLM_KEY},
                              json={'model': LLM_MODEL, 'temperature': temperature, 'messages': msgs}, timeout=timeout)
            out = r.json()['choices'][0]['message']['content']
            d = parse_json_obj(out)
            if d is not None:
                return d
            print('  [LLM] unparsable output, retry %d' % attempt, flush=True)
        except Exception as ex:
            print('  [LLM error] %s, retry %d' % (str(ex)[:120], attempt), flush=True)
        time.sleep(10 * (attempt + 1))
    return None


# ---------------------------------------------------------------- Serper
def call_serper(query, tag='', retries=4):
    """organic top-10 [{title, snippet, link}] or None; raw results appended to search_log.jsonl"""
    for attempt in range(retries * len(SERPER_KEYS)):
        with _lock:
            alive = [i for i in range(len(SERPER_KEYS)) if i not in _dead]
            if not alive:
                raise RuntimeError('All Serper keys exhausted')
            i = alive[_key_idx[0] % len(alive)]
            _key_idx[0] += 1
        try:
            r = requests.post('https://google.serper.dev/search',
                              headers={'X-API-KEY': SERPER_KEYS[i], 'Content-Type': 'application/json'},
                              json={'q': query, 'num': 10}, timeout=30)
            if r.status_code in (401, 403) or (r.status_code == 400 and 'credits' in r.text):
                with _lock:
                    _dead.add(i)
                print('  [Serper] key %s dead (%d)' % (SERPER_KEYS[i][:8], r.status_code), flush=True)
                continue
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            raw = r.json()
            res = [{'title': o.get('title', ''), 'snippet': o.get('snippet', ''), 'link': o.get('link', '')}
                   for o in raw.get('organic', [])[:10]]
            with _lock:
                with open(SEARCH_LOG, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': tag, 'query': query,
                                        'raw': raw}, ensure_ascii=False) + '\n')
            return res
        except Exception as ex:
            print('  [Serper error] %s' % str(ex)[:120], flush=True)
            time.sleep(2)
    return None


def squash(s):
    return re.sub(r'\s+', ' ', s or '').strip().lower()


# ---------------------------------------------------------------- code execution
RUNNER = r'''
import sys, io, json, math, contextlib
code = sys.stdin.read()
g = {"__builtins__": __builtins__, "math": math, **{f: getattr(math, f) for f in dir(math) if not f.startswith("_")}}
try:
    import numpy as np
    g["np"] = g["numpy"] = np
except ImportError:
    pass
out = io.StringIO()
try:
    with contextlib.redirect_stdout(out):
        exec(code, g)
    print(json.dumps({"ok": True, "out": out.getvalue().strip() or "No output"}))
except Exception as e:
    print(json.dumps({"ok": False, "out": "Error: %s" % e}))
'''


def run_code(code):
    try:
        r = subprocess.run([sys.executable, '-c', RUNNER], input=code, capture_output=True, text=True, timeout=30,
                           encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as ex:
        return {'ok': False, 'out': 'Error: %s' % ex}


# ---------------------------------------------------------------- numbers
SCALES = (1, 1e3, 1e-3, 1e6, 1e-6)
RANGE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)')


def text_nums(s):
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', str(s or ''))
    return [float(x) for x in re.findall(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', s)]


def consistent(out, nums, tol=0.02):
    cand = [abs(x) for x in text_nums(out) if abs(x) > 1e-12]
    return any(abs(x * s - y) <= tol * max(x * s, y) for x in cand for y in nums if y > 0 for s in SCALES)


def in_answer_range(out, answer):
    rngs = [(min(a, b), max(a, b)) for a, b in ((float(x), float(y)) for x, y in
                                                RANGE_RE.findall(re.sub(r'(?<=\d),(?=\d{3}\b)', '', answer or ''))) if a != b]
    cand = [abs(x) for x in text_nums(out) if abs(x) > 1e-12]
    return any(lo * 0.98 <= x * s <= hi * 1.02 for lo, hi in rngs for x in cand for s in SCALES)


def code_problem(r, target_nums, is_last, answer=''):
    """'' if the block ran and prints a target value (last block: the answer value or a value inside its range)"""
    if not r['ok']:
        return 'fails with "%s"' % r['out'][:150]
    if r['out'] == 'No output':
        return 'prints nothing'
    if target_nums and not consistent(r['out'], target_nums) and not (is_last and in_answer_range(r['out'], answer)):
        return ('prints "%s", which does not contain the final answer value' if is_last else
                'prints "%s", which does not match the numbers in the solution') % r['out'][:200]
    return ''


def check_codes(qa):
    """execute every code_executor source; returns list of {out, ok, problem}"""
    steps, _ = get_steps(qa)
    ans = str(qa.get('answer', ''))
    anums = [x for x in text_nums(ans) if x > 0]
    snums = text_nums(' '.join(steps) + ' ' + ans)
    codes = [s.get('code') or '' for s in sources_of(qa, 'code_executor')]
    res = []
    for i, c in enumerate(codes):
        is_last = i == len(codes) - 1
        r = run_code(c)
        prob = code_problem(r, anums if is_last and anums else snums, is_last, ans)
        res.append({'out': r['out'], 'ok': not prob, 'problem': prob})
    return res


# ---------------------------------------------------------------- video context
ROOT = os.environ.get('TRIPHYS_CONSTRUCTION_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(FC)))  # A_class_Benchmark/output, B_class_Benchmark/outputs


def stage_dir(qid):
    """directory holding the stage1/2/2b/25 outputs of the QA's video"""
    import glob
    tag = qid.split(':')[-1].split('|')[1]
    if qid.split(':')[-1].startswith('A|'):
        g = glob.glob('%s/A_class_Benchmark/output/*/%s' % (ROOT, tag))
        return g[0] if g else None
    d = '%s/B_class_Benchmark/outputs/%s' % (ROOT, tag)
    return d if os.path.isdir(d) else None


def video_context(qid, limit=1500):
    d = stage_dir(qid)
    if not d or not os.path.exists(d + '/stage1_omni_info.json'):
        return ''
    it = json.load(open(d + '/stage1_omni_info.json', encoding='utf-8'))[0]
    hl = (it.get('high_level') or {}).get('omni_annotation') or {}
    s = hl.get('global_summary') if isinstance(hl, dict) else ''
    clips = (it.get('mid_level') or {}).get('clips') or []
    txt = (s or '') + ' ' + ' '.join('[%.0f-%.0fs] %s %s' % (c.get('start_time', 0), c.get('end_time', 0), c.get('global_summary', ''),
                                                              '; '.join(c.get('visual_elements') or [])[:300]) for c in clips)
    return txt.strip()[:limit]
