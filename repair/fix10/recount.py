# Provenance (TriPhysBench release): fix_code/evaluation/fix10/recount.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 step 1 (10/11, user): recount which QAs need a rewrite under the agreed criteria, one pass, no repeats.
  W  web constant: the annotator's BASE query (search_query) must (a) return the reference value, (b) return no other valid
     value of the quantity that changes the score (all values in the base results are extracted, substituted into the
     code, scored with the official ACC; an alternative with ACC < .5 is a conflict unless the question singles out the
     reference value - VALID verdict), (c) be derivable from the question (QUAL: no "added" term).
     Values that only other queries find are not a problem (user 10/11).
  M  modelling: the 10/11 well-posedness M step (wellposed.json; re-run only for QAs whose question changed since):
     alternatives with ACC < .5, verdict both (reference fine, question silent) or X (alternative judged more standard).
  S  static: tool wording, hidden / web values written into the question.
  V  video: deepseek-flash sees frames of the full evaluation video (uniform + windows the question names) and labels every
     claim of the question about the scene: visible / plausible / contradicted / invented / not_shown.
  C  consistency: base code runs and prints the answer numbers.
Output recount.json + summary.  Usage (server, fc/evaluation/fix10): python recount.py [ids.json]"""
import os, sys, re, json, ast, asyncio, hashlib, base64
from collections import Counter, defaultdict
import httpx

ARGV = list(sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'wellposed'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'fix9'))
sys.argv = ['x']
import wellposed_check as W  # noqa: E402
from qual_check import QUAL  # noqa: E402
from fix9 import TOOL_RE, hidden_nums, leaked  # noqa: E402
import cv2  # noqa: E402

AT = W.AT
EGO = os.environ.get('EGOEXO_ROOT', '')
BAAI = os.environ.get('BAAI_ROOT', '')
VCACHE = HERE + '/recount_vcache.jsonl'
OUT = HERE + '/' + os.environ.get('RECOUNT_OUT', 'recount.json')
VSEM = None
TAGGED = False  # fix10 v7: second, object-tagged extraction (union; values of another object dropped)

EXTRACT2 = """QUESTION: %s
QUANTITY NEEDED: %s
SEARCH RESULTS:
%s

List EVERY number in the search results that is stated as a value of the QUANTITY NEEDED (any object), and say which object /
material / condition the result gives it for. same_object = true if that object / material / condition is the one the
QUESTION describes (or a general class that includes it), false if it is a clearly different one (another size, grade,
material, temperature, product). A stated range counts as one value. Copy the quote verbatim (max 200 characters, must
contain the number).
Return ONLY JSON: {"values": [{"value": <number or [low, high]>, "unit": "...", "refers_to": "...", "same_object": true|false,
"quote": "..."}]} (empty list if none)"""


def _num(v):
    try:
        return [float(x) for x in (v if isinstance(v, list) else [v])]
    except Exception:
        return None


def _samev(a, b):
    a, b = _num(a.get('value')), _num(b.get('value'))
    return bool(a and b and len(a) == len(b) and all(abs(x - y) <= 0.02 * max(abs(y), 1e-9) for x, y in zip(a, b)))

_vc = {}
if os.path.exists(VCACHE):
    for line in open(VCACHE):
        try:
            j = json.loads(line)
            _vc[j['k']] = j['v']
        except Exception:
            pass


# ---------------------------------------------------------------- W: base query
async def base_const(qa, src, base_code, base_out, quantity):
    ki, sq = src.get('key_information', ''), src.get('search_query', '')
    d = {'ref': ki, 'ref_query': sq, 'url': src.get('url', '')}
    if not sq:
        d['no_base_query'] = True
        return d
    if not quantity:
        j = W.pj(await W.llm(W.QUERIES % (qa['question'], W.mask(ki), W.mask(sq)))) or {}
        quantity = j.get('quantity') or W.mask(ki)
    d['quantity'] = quantity
    res = await W.asearch(sq)
    if res is None:
        return {'serper_failed': True}
    keyn = AT.nums(ki)
    d['reach_numeric'] = bool(keyn) and AT.has(keyn, res)
    e = W.pj(await W.llm(W.EXTRACT % (qa['question'], quantity, W.results_block([res])))) or {}
    vals = [v for v in e.get('values') or [] if isinstance(v, dict)]
    if TAGGED:  # a missed value would make a constant look unique; a value of another object would make it look ambiguous
        e2 = W.pj(await W.llm(EXTRACT2 % (qa['question'], quantity, W.results_block([res])))) or {}
        v2 = [v for v in e2.get('values') or [] if isinstance(v, dict)]
        other = [v for v in v2 if v.get('same_object') is False]
        d['other_object'] = [{k: v.get(k) for k in ('value', 'unit', 'refers_to', 'quote')} for v in other]
        vals = [v for v in vals if not any(_samev(v, o) for o in other)] + [v for v in v2 if v.get('same_object') is not False]
    uniq = []
    for v in vals:
        if not (isinstance(v, dict) and v.get('value') is not None and W.quote_ok(v.get('quote'), res, v.get('value'))):
            W.STAT['cand_rejected'] += 1
            continue
        vv = v['value'] if isinstance(v['value'], list) else [v['value']]
        try:
            vv = [float(x) for x in vv]
        except Exception:
            continue
        if not any(len(vv) == len(u['_v']) and all(abs(a - b) <= 0.02 * max(abs(b), 1e-9) for a, b in zip(vv, u['_v']))
                   and v.get('unit') == u.get('unit') for u in uniq):
            uniq.append(dict(v, _v=vv))
    rows = []
    for c in uniq[:12]:
        s = W.pj(await W.llm(W.SUBST % (base_code, quantity, ki, c['value'], c.get('unit', ''), c['quote']))) or {}
        row = {k: c.get(k) for k in ('value', 'unit', 'quote')}
        if s.get('same'):
            row['same'] = True
            rows.append(row)
            continue
        code = s.get('code') or ''
        if not code or W.canon(code) != W.canon(base_code):
            row['unchecked'] = 'edit_invalid'
            rows.append(row)
            continue
        o = await asyncio.to_thread(W.run_code, code)
        if TAGGED:
            row['code'], row['out'] = code, o
        if o.startswith('Error'):
            row['unchecked'] = 'run_error'
            rows.append(row)
            continue
        a = await W.alt_answer(qa, base_out, o)
        row['alt_answer'] = a
        row['acc'] = await W.acc(qa, a) if a else None
        if row['acc'] is None:
            row['unchecked'] = 'no_score'
        elif row['acc'] < 0.5:
            o1, o2 = (ki, '', src.get('verified_quote', '')), (c['value'], c.get('unit', ''), c['quote'])
            flip = W.coin(qa['question'], c['quote'])
            v = W.pj(await W.llm(W.VALID % ((quantity, qa['question']) + (o2 + o1 if flip else o1 + o2)))) or {}
            row['valid_verdict'] = {'1': 'alternative' if flip else 'reference',
                                    '2': 'reference' if flip else 'alternative', 'both': 'both'}.get(str(v.get('verdict')))
            row['valid_why'] = v.get('why')
        rows.append(row)
    d['candidates'] = rows
    d['n_values'] = len(rows)
    d['reach'] = d['reach_numeric'] or any(r.get('same') for r in rows)
    d['conflicts'] = [r for r in rows if r.get('acc') is not None and r['acc'] < 0.5
                      and r.get('valid_verdict') in ('both', 'alternative')]
    q = W.pj(await W.llm(QUAL % (qa['question'], W.mask(ki), sq))) or {}
    terms = [t for t in (q.get('terms') or []) if isinstance(t, dict)]
    d['added_terms'] = [t.get('term') for t in terms if t.get('label') == 'added']
    d['qual'] = 'added' if d['added_terms'] else ('from_question' if terms else None)
    return d


# ---------------------------------------------------------------- V: video frames
def video_path(item):
    v = item['video']
    if item['cls'] == 'A':
        return '%s/%s/best_exo.mp4' % (EGO, v['take_uid'])
    return '%s/%s/%s.mp4' % (BAAI, v['folder_id'], v['video_id'])


WIN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:–|—|-|to)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b', re.I)
AT_T = re.compile(r'(?:\bat\b|t\s*=)\s*(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b', re.I)


def frames(path, question, n_uniform=10, width=640):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = nf / fps
    ts = [dur * (i + 0.5) / n_uniform for i in range(n_uniform)]
    for a, b in WIN.findall(question):
        a, b = float(a), float(b)
        if a < b <= dur + 1:
            ts += [a + (b - a) * f for f in (0.1, 0.5, 0.9)]
    for t in AT_T.findall(question):
        if float(t) <= dur + 1:
            ts.append(float(t))
    ts = sorted({round(min(t, max(dur - 0.05, 0)), 2) for t in ts})[:16]
    out = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (width, int(width * fr.shape[0] / fr.shape[1])))
        out.append((t, base64.b64encode(cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()))
    return out, dur


VPROMPT = """You check a physics benchmark question against frames of the video it is about. The solver sees the video and the
question; the question must describe the scene correctly. Frames are sampled from the whole video (timestamps given).
QUESTION: %s

List every statement the question makes about the scene: objects, their identity (brand, model, product, size, regulation,
standard), material, colour, count, setting, positions, actions / events and their timing. NOT statements: numbers
(heights, speeds, angles, masses, distances, durations - given or measured), anything introduced by "assume" / "treat as" /
"consider", hypothetical events, physical interpretations (energy, friction, forces) and what is asked. Label each statement:
- "visible": the frames show it
- "plausible": not directly visible but consistent with the frames and the setting, a careful viewer would accept it
  (e.g. an adult men's game uses a size-7 ball; a dump truck bed is steel)
- "contradicted": the frames show something different
- "invented": a specific identity / material / grade / standard that nothing in the frames supports while other choices
  fit equally well (e.g. a brand or model with no visible marking)
- "not_shown": an object or event the question says happens in the video that appears in none of the frames
Also list each object whose physical property the question needs, with what the frames show about it.
Return ONLY JSON: {"claims": [{"claim": "...", "label": "visible|plausible|contradicted|invented|not_shown", "evidence": "..."}],
"objects": [{"object": "...", "seen": "<material, size class, markings, setting>"}]}"""


async def vllm(question, fr, key):
    k = hashlib.sha1(('deepseek-flash|' + key + '|' + VPROMPT % question).encode()).hexdigest()
    if k in _vc:
        return _vc[k]
    content = [{'type': 'text', 'text': VPROMPT % question}]
    for t, b in fr:
        content += [{'type': 'text', 'text': 'Frame at t = %.1f s:' % t},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b}}]
    async with VSEM:
        for a in range(5):
            body = {'model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'), 'temperature': 0.0, 'max_tokens': 32000,
                    'messages': [{'role': 'user', 'content': content}]}
            try:
                async with httpx.AsyncClient(timeout=900.0) as cl:
                    r = await cl.post(W.LLM_URL, headers={'Authorization': 'Bearer ' + W.LLM_KEY}, json=body)
                    t = r.json()['choices'][0]['message']['content']
                    if t:
                        _vc[k] = t
                        with open(VCACHE, 'a') as f:
                            f.write(json.dumps({'k': k, 'v': t}, ensure_ascii=False) + '\n')
                        return t
                    W.STAT['vllm_empty'] += 1
            except Exception as e:
                W.STAT['vllm_err ' + type(e).__name__] += 1
            await asyncio.sleep(4 * 2 ** a)
    W.STAT['vllm_fail'] += 1
    return ''


async def check_video(item):
    p = video_path(item)
    fr, dur = await asyncio.to_thread(frames, p, item['qa']['question'])
    if not fr:
        return {'error': 'video not readable: ' + p}
    j = W.pj(await vllm(item['qa']['question'], fr, '%s|%s' % (p, ','.join('%.2f' % t for t, _ in fr)))) or {}
    cl = [c for c in (j.get('claims') or []) if isinstance(c, dict)]
    return {'duration': round(dur, 1), 'n_frames': len(fr), 'claims': cl, 'objects': j.get('objects') or [],
            'contradicted': [c for c in cl if c.get('label') == 'contradicted'],
            'invented': [c for c in cl if c.get('label') == 'invented'],
            'not_shown': [c for c in cl if c.get('label') == 'not_shown']}


# ---------------------------------------------------------------- S: static
def static(qa):
    P = []
    m = TOOL_RE.search(qa['question'])
    if m:
        P.append({'kind': 'tool_wording', 'text': qa['question'][max(0, m.start() - 40):m.end() + 20]})
    lk = leaked(qa['question'], hidden_nums(qa))
    if lk:
        P.append({'kind': 'hidden_leak', 'values': lk})
    vid = [x for s in qa['sources'] if isinstance(s, dict) and s.get('type') == 'video_observation'
           for x in AT.nums(s.get('key_information'))]
    for s in qa['sources']:
        if isinstance(s, dict) and s.get('type') == 'web_search':
            ki = str(s.get('key_information') or '')
            used = ki.split('used as', 1)[1] if 'used as' in ki else ki
            vals = [x for x in AT.nums(used) if not any(abs(x - y) <= 1e-9 * max(abs(x), 1e-12) for y in vid)]
            lk = leaked(qa['question'], vals)
            if lk:
                P.append({'kind': 'web_leak', 'values': lk, 'ref': ki[:160]})
    return P


# ---------------------------------------------------------------- per QA
def load():
    items = []
    for c in 'ab':
        for sp in ('train', 'test'):
            for vi, v in enumerate(json.load(open('%s/data/v9/%sclass_qa_%s.json' % (W.ROOT, c, sp)))):
                for qi, x in enumerate(v['qa_pairs']):
                    items.append({'id': '%s|%s|%d|%d' % (c.upper(), sp, vi, qi), 'split': sp, 'cls': c.upper(), 'qa': x,
                                  'video': {k: v.get(k) for k in ('take_uid', 'folder_id', 'video_id')}})
    return items


async def check(item, wp):
    qa = item['qa']
    srcs = qa['sources']
    srcs = ast.literal_eval(srcs) if isinstance(srcs, str) else (srcs or [])
    qa['sources'] = srcs
    codes = [s['code'] for s in srcs if isinstance(s, dict) and s.get('type') == 'code_executor' and s.get('code')]
    ws = [s for s in srcs if isinstance(s, dict) and s.get('type') == 'web_search']
    r = {k: item[k] for k in ('id', 'split', 'cls')}
    r.update(question=qa['question'], answer=qa['answer'], answer_range=qa.get('answer_range'),
             question_type=qa['question_type'], difficulty=qa.get('difficulty'), n_ws=len(ws),
             n_hidden=len(qa.get('hidden_cv_chosen') or []))
    same = wp is not None and wp.get('question') == qa['question'] and wp.get('answer') == qa['answer']
    base_code = '\n'.join(codes)
    if not codes:
        j = W.pj(await W.llm(W.REBUILD % (qa['question'], '\n'.join(qa.get('annotated_solution') or []), qa['answer']))) or {}
        base_code = j.get('code') or ''
    base_out = await asyncio.to_thread(W.run_code, base_code) if base_code else 'Error: no code'
    r['consistency'] = 'ok'
    if base_out.startswith('Error'):
        r['consistency'] = 'code_error'
    elif qa['question_type'] in ('TYPE_A', 'TYPE_B') and W.num_map(qa['answer'], base_out) is None:
        b = await W.acc(qa, await W.alt_answer(qa, base_out, base_out))
        if b is None or b < 0.9:
            r['consistency'] = 'code_answer_mismatch'
    qn = [c.get('quantity') for c in (wp or {}).get('consts', [])] if same else []
    # 10/11 question-only queries per constant: an "added" base-query term matters only if the question alone did not
    # reach the value uniquely (reachable and not ambiguous -> the extra term is harmless)
    qo = [bool(not c.get('reachable') or c.get('ambiguous')) for c in (wp or {}).get('consts', [])] if same else []
    cs, vd = await asyncio.gather(
        asyncio.gather(*[base_const(qa, s, base_code, base_out, qn[i] if i < len(qn) else None) for i, s in enumerate(ws)]),
        check_video(item))
    if any(c.get('serper_failed') for c in cs):
        r['serper_failed'] = True
        return r
    for i, c in enumerate(cs):
        c['qo_problem'] = qo[i] if i < len(qo) else None
    r['consts'] = cs
    r['video'] = vd
    if same and 'model' in wp:
        r['model'] = wp['model']
    elif r['consistency'] == 'ok':
        r['model'] = await W.check_model(qa, base_code, base_out)
        r['model_rerun'] = True
    else:
        r['model'] = []
    r['static'] = static(qa)
    # problem labels
    L = []
    if r['consistency'] != 'ok':
        L.append('C_inconsistent')
    for c in cs:
        if c.get('no_base_query'):
            L.append('W_no_base_query')
            continue
        if not c['reach']:
            L.append('W_base_unreach')
        if c['conflicts']:
            L.append('W_base_conflict')
        if c['qual'] == 'added' and c.get('qo_problem') is not False:
            L.append('W_query_added')
    bad = [m for m in r['model'] if m.get('acc') is not None and m['acc'] < 0.5 and m.get('verdict') in ('both', 'X')]
    if any(m['verdict'] == 'X' for m in bad):
        L.append('M_alt_more_standard')
    if any(m['verdict'] == 'both' for m in bad):
        L.append('M_both_valid')
    if vd.get('error'):
        L.append('V_error')
    if vd.get('contradicted'):
        L.append('V_contradicted')
    if vd.get('invented'):
        L.append('V_invented')
    if vd.get('not_shown'):
        L.append('V_not_shown')
    L += ['S_' + p['kind'] for p in r['static']]
    r['labels'] = sorted(set(L))
    r['needs_fix'] = any(not l.startswith('V_not_shown') and not l.startswith('V_error') for l in r['labels'])
    return r


def summary(rs):
    G = defaultdict(list)
    for r in rs:
        G[(r['split'], r['cls'])].append(r)
    print('\n== QAs needing a fix (V_not_shown / V_error reported, not counted) ==')
    for g in sorted(G):
        X = G[g]
        print('%-14s n=%3d  needs_fix %3d  clean %3d' % (g, len(X), sum(r['needs_fix'] for r in X), sum(not r['needs_fix'] for r in X)))
    print('all n=%d needs_fix %d' % (len(rs), sum(r['needs_fix'] for r in rs)))
    print('\n== labels (QAs) ==')
    for k, v in sorted(Counter(l for r in rs for l in r['labels']).items()):
        print('  %-24s %3d   test %3d  train %3d' % (k, v, sum(k in r['labels'] for r in rs if r['split'] == 'test'),
                                                    sum(k in r['labels'] for r in rs if r['split'] == 'train')))
    print('\n== needs_fix by difficulty / type ==')
    for key in ('difficulty', 'question_type'):
        print(' ', key, dict(sorted(Counter((r['split'], r[key]) for r in rs if r['needs_fix']).items())))
    print('\nstats', dict(W.STAT))


async def main():
    global VSEM
    W.SEM, W.SSEM, VSEM = asyncio.Semaphore(32), asyncio.Semaphore(4), asyncio.Semaphore(12)
    items = load()
    if len(ARGV) > 1:
        ids = set(json.load(open(ARGV[1])))
        items = [i for i in items if i['id'] in ids]
    wp = {r['id']: r for r in json.load(open(os.path.join(os.path.dirname(HERE), 'wellposed', 'wellposed.json')))}
    done = {}
    if os.path.exists(OUT):
        done = {r['id']: r for r in json.load(open(OUT)) if not r.get('serper_failed')}
    todo = [i for i in items if i['id'] not in done]
    print('QAs', len(items), 'todo', len(todo), flush=True)
    rs = [done[i['id']] for i in items if i['id'] in done]
    for k in range(0, len(todo), 24):
        rs += await asyncio.gather(*[check(i, wp.get(i['id'])) for i in todo[k:k + 24]])
        json.dump(rs, open(OUT, 'w'), ensure_ascii=False, indent=1)
        print('progress %d/%d serper_fail %d %s' % (len(rs), len(items), W.SERPER_FAILS[0], dict(W.STAT)), flush=True)
        if W.SERPER_FAILS[0] >= 3:
            print('SERPER FAILING - stop', flush=True)
            sys.exit(3)
    summary([r for r in rs if not r.get('serper_failed')])


if __name__ == '__main__':
    asyncio.run(main())
