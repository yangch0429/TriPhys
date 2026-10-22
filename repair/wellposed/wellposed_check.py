# Provenance (TriPhysBench release): fix_code/evaluation/wellposed/wellposed_check.py
# Role: well-posedness / quality check of the 445 QAs of version 07 (10-11); base module of fix9 / fix10.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Well-posedness check of every QA (train + test, A + B, current version 06 = data/v9) - 10/11, user: "step 1, check the
whole train and test set". Question: does the QUESTION TEXT determine the reference answer?
Every check ends in executed code scored with the official ACC function (metrics_final.calculate_acc_with_fallback);
an alternative answer with ACC < .5 counts as "marked wrong".
  W  web constants   per web_search source: the checker LLM (deepseek-flash) writes 3 Google queries from the QUESTION ONLY (+ the annotated query)
                     -> Serper (eval request / formatting, cache results/serper_cache_train); it lists every value of the
                     quantity in the results that fits the object / conditions the question states (verbatim quote,
                     post-checked). reachable = the reference value is found by the question-only queries.
                     Each other value: base code minimally edited (only numeric literals may change - checked), run, scored;
                     if marked wrong, a second checker call (sees both quotes + the question) must call the alternative
                     at least as valid as the reference -> web_ambiguous.
  M  model choices   the checker LLM lists modelling / interpretation choices of the reference solution that change the number
                     and are not stated or implied by the question (video readings, web constants and default textbook
                     conventions excluded) with the alternative as edited base code; run, scored; if marked wrong a second
                     call decides from the question text alone: reference / alternative / both -> model_ambiguous unless
                     "reference".
  C  consistency     base code (all code_executor sources in order) runs and its output contains the reference answer
                     numbers (QAs without code: the checker LLM implements the annotated solution, must reproduce the answer).
Flags per QA: annotation_error, web_unreachable, web_ambiguous, model_ambiguous; well_posed = no flag.
A-class video readings are out of scope (user 10/11: ignore CV values for now).
Output: wellposed.json (pilot: wellposed_pilot.json), caches llm_cache.jsonl; summary printed.
Usage (server, fc/evaluation/wellposed): python wellposed_check.py [n_per_group]   n_per_group = pilot QAs per split x class
Stops (exit 3, "SERPER" in log) as soon as Serper calls fail - keys / credits."""
import os, sys, re, json, ast, time, asyncio, hashlib, random, subprocess, logging
from collections import Counter, defaultdict
import httpx

ROOT = os.environ.get('TRIPHYS_EXP_ROOT', '.')  # server experiment dir: data/v9, scripts/, results/
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get('TRIPHYS_PYTHON', sys.executable)
sys.path.insert(0, ROOT + '/scripts')
sys.path.insert(0, ROOT + '/fc/training/v36')
from metrics_final import calculate_acc_with_fallback  # noqa: E402  (server scoring module; cf. triphys/metrics.py)
JUDGE_API_URL = os.environ.get('ACC_JUDGE_BASE', 'https://openrouter.ai/api/v1') + '/chat/completions'  # ACC-judge endpoint (fix10 reviewer)
JUDGE_API_KEY = os.environ.get('ACC_JUDGE_API_KEY') or os.environ.get('OPENROUTER_API_KEY', '')
import ana_tool_v36 as AT  # noqa: E402

# checker LLM: deepseek-flash on api.deepseek.com (user 10/11: use DeepSeek instead of gpt-6-astra; key = fixlib's);
# the official ACC scoring (metrics_final, TYPE_D judge gpt-4o-mini) is unchanged
MODEL = os.environ.get('WP_MODEL', 'deepseek-flash')
LLM_URL = os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com') + '/chat/completions'
LLM_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
LLM_CACHE = HERE + '/llm_cache.jsonl'
SEM = None
SSEM = None
STAT = Counter()
SERPER_FAILS = [0]
random.seed(38)

# ---------------------------------------------------------------- infrastructure
_cache = {}
if os.path.exists(LLM_CACHE):
    for line in open(LLM_CACHE):
        try:
            j = json.loads(line)
            _cache[j['k']] = j['v']
        except Exception:
            pass


async def llm(prompt, T=0.0):
    k = hashlib.sha1((MODEL + '|' + prompt).encode()).hexdigest()
    if k in _cache:
        return _cache[k]
    async with SEM:
        for a in range(5):
            # deepseek-flash thinks by default and can spend the whole budget on reasoning (empty content,
            # finish_reason length): 32k budget, and from the 2nd attempt on thinking disabled
            body = {'model': MODEL, 'temperature': T, 'max_tokens': 32000,
                    'messages': [{'role': 'user', 'content': prompt}]}
            if a:
                body['thinking'] = {'type': 'disabled'}
            try:
                async with httpx.AsyncClient(timeout=900.0) as cl:
                    r = await cl.post(LLM_URL, headers={'Authorization': 'Bearer ' + LLM_KEY}, json=body)
                    t = r.json()['choices'][0]['message']['content']
                    if not t:
                        STAT['llm_empty'] += 1
                    if t:
                        _cache[k] = t
                        with open(LLM_CACHE, 'a') as f:
                            f.write(json.dumps({'k': k, 'v': t}, ensure_ascii=False) + '\n')
                        return t
            except Exception as e:
                STAT['llm_err ' + type(e).__name__] += 1
            await asyncio.sleep(4 * 2 ** a)
    STAT['llm_fail'] += 1
    return ''


def pj(t):
    """first JSON object / array in an LLM reply"""
    t = re.sub(r'```(?:json)?', '', t or '')
    for pat in (r'\{.*\}', r'\[.*\]'):
        m = re.search(pat, t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
            try:  # 10/13: a quote copied from a LaTeX snippet ({\displaystyle \mu}) has invalid escapes
                return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', m.group(0)))
            except Exception:
                pass
    return None


_SEARCH = None


def search(q):
    """the sampler's cached Serper call (same request + formatting as eval_sft_v9, current keys) -> result text or None"""
    global _SEARCH
    if _SEARCH is None:  # exec only the key list + the two search functions (the module runs its sampler on import)
        src = open(ROOT + '/scripts/sample_dpo_v9_cached.py').read()
        heavy = ('torch', 'transformers', 'peft', 'qwen_vl_utils', 'tqdm', 'prompts', 'response_parser', 'metrics_final')
        keep = []
        for n in ast.parse(src).body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                n.names = [a for a in n.names if not (getattr(n, 'module', None) or a.name).split('.')[0] in heavy
                           and a.name.split('.')[0] not in heavy]
                if n.names:
                    keep.append(n)
            elif isinstance(n, ast.Assign) and any(getattr(t, 'id', '').startswith(('SERPER', '_serper')) for t in n.targets)                     or isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', '').startswith(('SERPER', '_serper')):
                keep.append(n)
            elif isinstance(n, ast.FunctionDef) and n.name in ('_query_web_search_live', 'query_web_search'):
                keep.append(n)
        g = {'logger': logging.getLogger('serper')}
        exec(compile(ast.Module(body=keep, type_ignores=[]), 'sample_dpo_v9_cached', 'exec'), g)
        if os.environ.get('SERPER_API_KEYS'):  # release: Serper keys from the environment
            g['SERPER_KEYS'] = [k.strip() for k in os.environ.get('SERPER_API_KEYS', '').split(',') if k.strip()]
        _SEARCH = g['query_web_search']
    for a in range(4):  # transient network timeouts: retry before counting a failure
        r = _SEARCH(q)
        if '__SERPER_FAILED__' not in r:
            break
        STAT['serper_retry'] += 1
        time.sleep(10 * 2 ** a)
    else:
        SERPER_FAILS[0] += 1
        return None
    try:
        r = json.loads(r).get('result', r)
    except Exception:
        pass
    return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


async def asearch(q):
    async with SSEM:
        return await asyncio.to_thread(search, q)


def run_code(code):
    try:
        p = subprocess.run([PY, '-c', code], capture_output=True, text=True, timeout=60)
        out = p.stdout.strip()
        return out if p.returncode == 0 and out else 'Error: ' + (p.stderr.strip()[-300:] or 'no output')
    except subprocess.TimeoutExpired:
        return 'Error: timeout'


def canon(code):
    """code with comments, numbers and whitespace removed - an edit that changes only numeric literals keeps it"""
    c = re.sub(r'#[^\n]*', '', code)
    c = re.sub(AT.NUM, '#', c)
    return re.sub(r'\s+', '', c)


def coin(*a):
    return int(hashlib.md5(repr(a).encode()).hexdigest(), 16) % 2 == 1


def mask(s):
    return re.sub(r'\d+(?:[.,]\d+)*', '#', str(s or ''))


async def acc(qa, pred):
    rd = {'prediction': pred, 'reasoning_chain': '', 'ground_truth': qa['answer'], 'answer_range': qa.get('answer_range'),
          'annotated_solution': qa.get('annotated_solution', []), 'question': qa['question'],
          'question_type': qa['question_type'], 'needs_llm_eval': False, 'raw_response': pred}
    try:
        return float(await calculate_acc_with_fallback(rd))
    except Exception as e:
        STAT['acc_err ' + type(e).__name__] += 1
        return None


def num_map(answer, base_out):
    """index (in base_out numbers) of every number of the reference answer, None if one is not printed by the code"""
    on = AT.nums(base_out)
    idx = []
    for a in AT.nums(answer):
        js = [j for j, x in enumerate(on) if abs(x - a) <= 0.02 * max(abs(a), 1e-9)]
        if not js:
            return None
        idx.append(js[-1])
    return idx


async def alt_answer(qa, base_out, alt_out):
    """the reference answer string re-derived from a new code output"""
    idx = num_map(qa['answer'], base_out)
    an = AT.nums(qa['answer'])
    on = AT.nums(alt_out)
    if idx and an and len(on) == len(AT.nums(base_out)):
        parts, last = [], 0
        ms = list(re.finditer(AT.NUM, qa['answer']))
        for m, j in zip(ms, idx):
            parts += [qa['answer'][last:m.start()], '%.4g' % on[j]]
            last = m.end()
        return ''.join(parts) + qa['answer'][last:]
    t = await llm(ALT % (qa['question'], qa['answer'], base_out[:1500], alt_out[:1500]))
    j = pj(t) or {}
    return str(j.get('answer') or '')


# ---------------------------------------------------------------- prompts
QUERIES = """A student must answer the physics question below and has to look up one quantity with Google.
QUESTION: %s
The reference solution looked it up as (numbers hidden): %s
Reference query (numbers hidden): %s
1. Name the SAME kind of quantity the reference looked up (if it looked up a ratio / percentage / coefficient, name that
   ratio / percentage / coefficient, not a derived absolute value), generically, using ONLY qualifiers (object, material, population, size, condition) that are stated or
   clearly implied by the QUESTION; drop qualifiers that only the reference adds.
2. Write 3 different Google queries a careful student would write from the QUESTION alone for that quantity (no numbers,
   no qualifiers that the question does not state or imply).
Return ONLY JSON: {"quantity": "...", "queries": ["...", "...", "..."]}"""

EXTRACT = """QUESTION: %s
QUANTITY NEEDED: %s
SEARCH RESULTS:
%s

List every value of the QUANTITY NEEDED that the search results state for an object / material / condition consistent with
what the QUESTION describes. A value for a clearly different object or condition does not count; a value for a general
class that includes the question's object counts; a stated range counts as one value. Copy the quote verbatim from the
results (max 200 characters, must contain the number).
Return ONLY JSON: {"values": [{"value": <number or [low, high]>, "unit": "...", "result": <result number>, "quote": "..."}]}
(empty list if none)"""

SUBST = """Reference solution code of a physics question:
```python
%s
```
In this code the web-searched quantity "%s" has the reference value: %s
Candidate value from another source: %s %s (quote: "%s")
1. same: is the candidate equal to the reference value after unit conversion (within 2%%; a range counts as equal if it
   contains the reference value)?
2. If not, rewrite the code so it uses the candidate value instead - change ONLY numeric literals (convert units yourself;
   for a range use the midpoint unless the code itself uses a low / high pair). Keep everything else identical.
Return ONLY JSON: {"same": true/false, "code": "<full edited code, or empty if same>"}"""

VALID = """A physics benchmark question requires looking up "%s" on the web; the solver sees only the QUESTION text.
QUESTION: %s

Two values for this quantity were found on two different web pages. NEITHER value nor quote is part of the question.
Value 1: %s %s (web quote: "%s")
Value 2: %s %s (web quote: "%s")
Does the QUESTION text contain a detail (object, model, size, material, population, condition) that makes one value fit and
the other not? Reply "1" or "2" if the question points to that value only, "both" if a careful solver reading only the
question could reasonably use either value.
Return ONLY JSON: {"verdict": "1|2|both", "why": "..."}"""

MODEL_P = """You audit a physics benchmark QA for ambiguity. The solver sees only the QUESTION (and, for class A, a video
from which it reads times / distances / speeds).
QUESTION: %s
QUESTION TYPE: %s
REFERENCE SOLUTION:
%s
REFERENCE CODE:
```python
%s
```
REFERENCE ANSWER: %s

List up to 3 modelling or interpretation choices made by the reference solution that change the numeric answer and that the
question does NOT state or clearly imply, where a competent physicist reading only the question could reasonably choose
differently (e.g. which formula / model, which quantity is asked for, start from rest or not, rolling vs sliding, which point
of the motion, inclusion of an effect, definition of a ratio). Do NOT list: values read from the video, values looked up on
the web, default textbook conventions every solver would adopt (g = 9.81, point mass, no air resistance unless the
question raises it), or answer-format choices (range vs single value, rounding, sign convention, units). For each choice give the alternative implemented as a minimal edit of the reference code (same
inputs, same printed format, only the modelling changed).
Return ONLY JSON: {"choices": [{"aspect": "<neutral 3-8 word name, no argument>", "reference": "<the reference's option, one
sentence>", "alternative": "<the other option, one sentence>", "code": "<full code>"}]}
(empty list if the question determines the model)"""

MODEL_V = """A physics benchmark question; the solver sees only the QUESTION text (and, for class A, a video for
measurements).
QUESTION: %s

Aspect of the solution: %s
Option 1: %s
Option 2: %s
The option texts were written by an auditor; they are NOT part of the question. Judging only from the QUESTION text, which
option should a careful solver choose? "1" or "2" if the question states or clearly implies that option (or it is the only
standard reading), "both" if the question leaves it open.
Return ONLY JSON: {"verdict": "1|2|both", "why": "..."}"""

REBUILD = """Write Python code that implements exactly this reference solution of a physics question and prints the final
result (all intermediate inputs as literals taken from the solution).
QUESTION: %s
REFERENCE SOLUTION:
%s
REFERENCE ANSWER: %s
Return ONLY JSON: {"code": "<full code>"}"""

ALT = """QUESTION: %s
The reference answer "%s" was derived from this code output:
%s
The code was changed; new output:
%s
Write the final answer derived from the NEW output in exactly the same form (same quantity, units, wording, range form).
Return ONLY JSON: {"answer": "..."}"""


# ---------------------------------------------------------------- checks
def results_block(texts):
    return '\n\n'.join('[%d] %s' % (i + 1, t[:3000]) for i, t in enumerate(texts))


_SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−–', '0123456789-+--')
_SCI = re.compile(r'(\d+(?:\.\d+)?)\s*(?:[×xX*·]\s*10\s*(?:\^|\*\*)?|10\s*(?:\^|\*\*))\s*\(?\s*([-+]?)\s*(\d+)\)?')


def sci(t):
    """10/13: scientific notation as written in snippets (5.670 × 10⁻⁸, 1.789×10 −5, 5.6704 10^-8) -> 5.670e-8"""
    return _SCI.sub(lambda m: '%se%s%s' % m.groups(), t.translate(_SUP))


def quote_ok(quote, blob, value):
    q = re.sub(r'\s+', ' ', str(quote or '')).strip()
    if len(q) < 4 or q[:60].lower() not in re.sub(r'\s+', ' ', blob).lower():
        return False
    v = value if isinstance(value, list) else [value]
    try:
        return all(AT.has([float(x)], q) or AT.has([float(x)], sci(q)) for x in v)
    except Exception:
        return False


async def check_const(qa, src, base_code, base_out):
    ki, sq = src.get('key_information', ''), src.get('search_query', '')
    j = pj(await llm(QUERIES % (qa['question'], mask(ki), mask(sq)))) or {}
    qs = [q for q in (j.get('queries') or []) if isinstance(q, str) and q.strip()][:3]
    quantity = j.get('quantity') or mask(ki)
    if not qs:
        STAT['queries_fail'] += 1
    res_q = await asyncio.gather(*[asearch(q) for q in qs])
    res_a = await asearch(sq) if sq else None
    if any(r is None for r in res_q) or (sq and res_a is None):
        return {'serper_failed': True}
    d = {'quantity': quantity, 'queries': qs, 'ref': ki, 'ref_query': sq}
    texts = list(res_q) + ([res_a] if res_a else [])
    keyn = AT.nums(ki)
    d['reach_numeric'] = bool(keyn) and AT.has(keyn, ' '.join(res_q))
    # extract candidates per result block (question-only queries first)
    cands = []
    for i, t in enumerate(texts):
        e = pj(await llm(EXTRACT % (qa['question'], quantity, results_block([t])))) or {}
        for v in (e.get('values') or []):
            if isinstance(v, dict) and v.get('value') is not None and quote_ok(v.get('quote'), t, v.get('value')):
                cands.append(dict(v, from_question=i < len(res_q)))
            else:
                STAT['cand_rejected'] += 1
    # distinct values (2%), question-only first
    uniq = []
    for c in sorted(cands, key=lambda c: not c['from_question']):
        v = c['value'] if isinstance(c['value'], list) else [c['value']]
        try:
            v = [float(x) for x in v]
        except Exception:
            continue
        if not any(len(v) == len(u['_v']) and all(abs(a - b) <= 0.02 * max(abs(b), 1e-9) for a, b in zip(v, u['_v']))
                   and c.get('unit') == u.get('unit') for u in uniq):
            uniq.append(dict(c, _v=v))
    uniq = uniq[:8]
    d['n_candidates'] = len(uniq)
    out, reach = [], False
    for c in uniq:
        s = pj(await llm(SUBST % (base_code, quantity, ki, c['value'], c.get('unit', ''), c['quote']))) or {}
        row = {k: c[k] for k in ('value', 'unit', 'quote', 'from_question')}
        if s.get('same'):
            row['same'] = True
            reach |= c['from_question']
            out.append(row)
            continue
        code = s.get('code') or ''
        if not code or canon(code) != canon(base_code):
            row['edit_invalid'] = True
            STAT['edit_invalid'] += 1
            out.append(row)
            continue
        o = await asyncio.to_thread(run_code, code)
        row['out'] = o[-300:]
        if o.startswith('Error'):
            row['run_error'] = True
            out.append(row)
            continue
        a = await alt_answer(qa, base_out, o)
        row['alt_answer'] = a
        row['acc'] = await acc(qa, a) if a else None
        if row['acc'] is not None and row['acc'] < 0.5:
            o1, o2 = (ki, '', src.get('verified_quote', '')), (c['value'], c.get('unit', ''), c['quote'])
            flip = coin(qa['question'], c['quote'])  # order randomised (deterministic -> cacheable) against position bias
            v = pj(await llm(VALID % ((quantity, qa['question']) + (o2 + o1 if flip else o1 + o2)))) or {}
            row['valid_verdict'] = {'1': 'alternative' if flip else 'reference',
                                    '2': 'reference' if flip else 'alternative', 'both': 'both'}.get(str(v.get('verdict')))
            row['valid_why'] = v.get('why')
        out.append(row)
    d['reachable'] = reach or d['reach_numeric']
    d['candidates'] = out
    d['ambiguous'] = any(r.get('acc') is not None and r['acc'] < 0.5 and r.get('valid_verdict') in ('both', 'alternative')
                         for r in out)
    return d


async def check_model(qa, base_code, base_out):
    sol = '\n'.join(qa.get('annotated_solution') or [])
    j = pj(await llm(MODEL_P % (qa['question'], qa['question_type'], sol, base_code, qa['answer']))) or {}
    out = []
    for c in (j.get('choices') or [])[:3]:
        if not isinstance(c, dict) or not c.get('code'):
            continue
        row = {k: c.get(k) for k in ('aspect', 'reference', 'alternative')}
        o = await asyncio.to_thread(run_code, c['code'])
        row['out'] = o[-300:]
        if o.startswith('Error'):
            row['run_error'] = True
            out.append(row)
            continue
        a = await alt_answer(qa, base_out, o)
        row['alt_answer'] = a
        row['acc'] = await acc(qa, a) if a else None
        if row['acc'] is not None and row['acc'] < 0.5:
            flip = coin(qa['question'], c.get('aspect'))
            o = (c.get('alternative'), c.get('reference')) if flip else (c.get('reference'), c.get('alternative'))
            v = pj(await llm(MODEL_V % ((qa['question'], c.get('aspect')) + o))) or {}
            row['verdict'] = {'1': 'X' if flip else 'R', '2': 'R' if flip else 'X', 'both': 'both'}.get(str(v.get('verdict')))
            row['why'] = v.get('why')
        out.append(row)
    return out


async def check_qa(item):
    qa = item['qa']
    srcs = qa['sources']
    srcs = ast.literal_eval(srcs) if isinstance(srcs, str) else (srcs or [])
    codes = [s['code'] for s in srcs if isinstance(s, dict) and s.get('type') == 'code_executor' and s.get('code')]
    ws = [s for s in srcs if isinstance(s, dict) and s.get('type') == 'web_search']
    r = {k: item[k] for k in ('id', 'split', 'cls')}
    r.update(question=qa['question'], answer=qa['answer'], answer_range=qa.get('answer_range'),
             question_type=qa['question_type'], flags=[])
    base_code = '\n'.join(codes)
    if not codes:
        j = pj(await llm(REBUILD % (qa['question'], '\n'.join(qa.get('annotated_solution') or []), qa['answer']))) or {}
        base_code = j.get('code') or ''
        r['code_rebuilt'] = True
    base_out = await asyncio.to_thread(run_code, base_code) if base_code else 'Error: no code'
    r['base_out'] = base_out[-500:]
    numeric = qa['question_type'] in ('TYPE_A', 'TYPE_B')
    if base_out.startswith('Error'):
        r['flags'].append('annotation_error')
        r['consistency'] = 'code_error'
        return r
    if numeric and num_map(qa['answer'], base_out) is None:
        base_acc = await acc(qa, await alt_answer(qa, base_out, base_out))
        r['base_acc'] = base_acc
        if base_acc is None or base_acc < 0.9:
            r['flags'].append('annotation_error')
            r['consistency'] = 'code_answer_mismatch'
            return r
    r['consistency'] = 'ok'
    cs = await asyncio.gather(*[check_const(qa, s, base_code, base_out) for s in ws])
    if any(c.get('serper_failed') for c in cs):
        r['serper_failed'] = True
        return r
    r['consts'] = cs
    r['model'] = await check_model(qa, base_code, base_out)
    if any(not c['reachable'] for c in cs):
        r['flags'].append('web_unreachable')
    if any(c['ambiguous'] for c in cs):
        r['flags'].append('web_ambiguous')
    if any(m.get('acc') is not None and m['acc'] < 0.5 and m.get('verdict') in ('both', 'X') for m in r['model']):
        r['flags'].append('model_ambiguous')
    return r


def load():
    items = []
    for c in 'ab':
        for sp in ('train', 'test'):
            for vi, v in enumerate(json.load(open('%s/data/v9/%sclass_qa_%s.json' % (ROOT, c, sp)))):
                for qi, x in enumerate(v['qa_pairs']):
                    items.append({'id': '%s|%s|%d|%d' % (c.upper(), sp, vi, qi), 'split': sp, 'cls': c.upper(), 'qa': x})
    return items


def summary(rs):
    G = defaultdict(list)
    for r in rs:
        G[(r['split'], r['cls'])].append(r)
    for g in sorted(G):
        X = G[g]
        f = Counter(x for r in X for x in r['flags'])
        print('%-5s %s n=%3d  well_posed %3d (%.2f) | annotation_error %d  web_unreachable %d  web_ambiguous %d  '
              'model_ambiguous %d' % (g[0], g[1], len(X), sum(not r['flags'] for r in X),
                                      sum(not r['flags'] for r in X) / len(X), f['annotation_error'],
                                      f['web_unreachable'], f['web_ambiguous'], f['model_ambiguous']), flush=True)
        print('       ambiguous of any kind (web_ambiguous or model_ambiguous or annotation_error): %d (%.2f)' % (
            sum(bool({'web_ambiguous', 'model_ambiguous', 'annotation_error'} & set(r['flags'])) for r in X),
            sum(bool({'web_ambiguous', 'model_ambiguous', 'annotation_error'} & set(r['flags'])) for r in X) / len(X)))


async def main(n):
    global SEM, SSEM
    SEM, SSEM = asyncio.Semaphore(48), asyncio.Semaphore(4)
    items = load()
    print('QAs', len(items), dict(Counter((i['split'], i['cls']) for i in items)), 'model', MODEL, flush=True)
    if n:
        g = defaultdict(list)
        for i in items:
            g[(i['split'], i['cls'])].append(i)
        items = [i for v in g.values() for i in random.sample(v, min(n, len(v)))]
    out = HERE + ('/wellposed_pilot.json' if n else '/wellposed.json')
    done = {}
    if os.path.exists(out):
        done = {r['id']: r for r in json.load(open(out)) if not r.get('serper_failed')}
    todo = [i for i in items if i['id'] not in done]
    print('todo', len(todo), 'done', len(done), flush=True)
    rs = list(done.values())
    for k in range(0, len(todo), 24):
        rs += await asyncio.gather(*[check_qa(i) for i in todo[k:k + 24]])
        json.dump(rs, open(out, 'w'), ensure_ascii=False, indent=1)
        print('progress %d/%d  serper_fail %d  %s' % (len(rs), len(items), SERPER_FAILS[0], dict(STAT)), flush=True)
        if SERPER_FAILS[0] >= 3:
            print('SERPER FAILING - stop; check keys / credits', flush=True)
            sys.exit(3)
    rs = [r for r in rs if not r.get('serper_failed')]
    summary(rs)


if __name__ == '__main__':
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
