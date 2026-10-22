# Provenance (TriPhysBench release): fix_code/_v8_work/v8_common.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Shared helpers for the v8 SFT pipeline (LLM batch calls, Serper with key rotation)."""
import json, os, re, time, threading, requests

DATA = os.environ.get('TRIPHYS_EXP_ROOT', '.') + '/data'
SPLIT = os.environ.get('V8_SPLIT', 'train')  # train | test
SUFFIX = '' if SPLIT == 'train' else '_' + SPLIT

REVIEWER_URL = os.environ.get('CHAT_API_BASE', 'https://openrouter.ai/api/v1') + '/chat/completions'  # originally an OpenAI-compatible relay
REVIEWER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
REVIEWER_MODEL = os.environ.get('REVIEWER_MODEL', 'gpt-6-astra')

SERPER_KEYS = [k.strip() for k in os.environ.get('SERPER_API_KEYS', '').split(',') if k.strip()]
_key_lock = threading.Lock()
_key_idx = [0]
_dead = set()


def qkey(text):
    return text.strip().splitlines()[0].strip() if text.strip() else ''


def norm(s):
    return re.sub(r'\s+', ' ', (s or '').strip().lower())


# closing quote must be followed by ')' (or end of sentence for 'Search query:'), so "Newton's" is not cut
SOL_Q_RE = re.compile(r"(?:web_search\(\s*query\s*=\s*(['\"])(.*?)\1\s*\)"
                      r"|Search query:\s*(['\"])(.*?)\3(?=[.,;]?(?:\s|$)))")


def find_sol_queries(step):
    return [m.group(2) if m.group(2) is not None else m.group(4) for m in SOL_Q_RE.finditer(step)]


def sol_steps(qa):
    sol = qa.get('annotated_solution', [])
    return sol if isinstance(sol, list) else [str(sol)]


def load_bench(split=None):
    """Return list of dicts: qid, cls, qkey, qa, video meta. Test qids are prefixed 'test:'."""
    split = split or SPLIT
    out = []
    for cls in 'AB':
        for vid in json.load(open('%s/%sclass_qa_%s.json' % (DATA, cls.lower(), split))):
            vtag = vid.get('take_uid') or '%s/%s' % (vid.get('folder_id'), vid.get('video_id'))
            for i, qa in enumerate(vid['qa_pairs']):
                out.append({'qid': ('' if split == 'train' else split + ':') + '%s|%s|%d' % (cls, vtag, i), 'cls': cls,
                            'qkey': qkey(qa['question']), 'qa': qa})
    return out


def parse_json_obj(text):
    m = re.search(r'\{.*\}', text or '', re.S)
    return json.loads(m.group(0)) if m else None


def call_llm(prompt, timeout=240, retries=4, temperature=0):
    for attempt in range(retries):
        try:
            r = requests.post(REVIEWER_URL, headers={'Authorization': 'Bearer ' + REVIEWER_KEY},
                              json={'model': REVIEWER_MODEL, 'temperature': temperature,
                                    'messages': [{'role': 'user', 'content': prompt}]},
                              timeout=timeout)
            out = r.json()['choices'][0]['message']['content']
            d = parse_json_obj(out)
            if d is not None:
                return d
            print('  [LLM] unparsable output, retry %d' % attempt, flush=True)
        except Exception as ex:
            print('  [LLM error] %s, retry %d' % (str(ex)[:120], attempt), flush=True)
        time.sleep(10 * (attempt + 1))
    return None


def call_serper(query, retries=4):
    """Return list of organic results [{title, snippet, link}] (up to 10), or None on failure."""
    for attempt in range(retries * len(SERPER_KEYS)):
        with _key_lock:
            alive = [i for i in range(len(SERPER_KEYS)) if i not in _dead]
            if not alive:
                raise RuntimeError('All Serper keys exhausted')
            i = alive[_key_idx[0] % len(alive)]
            _key_idx[0] += 1
        try:
            r = requests.post('https://google.serper.dev/search',
                              headers={'X-API-KEY': SERPER_KEYS[i], 'Content-Type': 'application/json'},
                              json={'q': query, 'num': 10}, timeout=30)
            if r.status_code in (401, 403):
                with _key_lock:
                    _dead.add(i)
                print('  [Serper] key %s dead (%d)' % (SERPER_KEYS[i][:8], r.status_code), flush=True)
                continue
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            org = r.json().get('organic', [])
            return [{'title': o.get('title', ''), 'snippet': o.get('snippet', ''), 'link': o.get('link', '')}
                    for o in org[:10]]
        except Exception as ex:
            print('  [Serper error] %s' % str(ex)[:120], flush=True)
            time.sleep(2)
    return None


def snippet_text(results, k=5):
    """Tool-result text as used in training: top-k snippets joined with ' | '."""
    return ' | '.join(r['snippet'] for r in (results or [])[:k] if r.get('snippet'))
