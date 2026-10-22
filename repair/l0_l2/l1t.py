# Provenance (TriPhysBench release): fix_code/_v8_work/l1t.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L1T: L1 repair of the QAs that L1 / L2 / L1E could not repair (writeback_report.json not_repaired), with the
question type free (difficulty fixed) and per-QA guidance from the one-by-one analysis (l1t_hints.json).
  evidence: evidence.json overlaid by evidence_l1e.json; facts listed in "drop" are removed (the QA is reformulated
            so it no longer needs them); "log" items add / replace facts with real search results of search_log.jsonl
            (targeted searches L1T|n): the result whose link contains the url part and whose snippet contains the quote
            verbatim; the query must not contain the value, the quote must contain it. Every fact without evidence
            must be dropped or replaced.  -> evidence_l1t.json
  rewrite:  L1 rewrite prompt with the type choice (allowed types per QA), the question may be reworded (video values
            kept; only the numbers listed in "qnums" may be added), REMOVE facts, guidance = hint notes + the reasons
            earlier repairs failed. Same validation (l1_rewrite.validate, checked against the chosen type) + B-class
            hidden count / garble checks + independent review with the chosen type's rule. 4 attempts, deepseek-flash.
            -> l1t_results.json {qid: {status, attempts, qa, evidence, question_type}}   (resumable)
Usage: python l1t.py evidence | rewrite [qid ...]"""
import copy, json, os, re, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import l1_rewrite as R
from fixlib import WORK, call_llm, get_steps, sources_of, text_nums, squash, video_context
from current import current

# L1T_SET=<name> runs the same repair on another QA set: <name>_hints.json -> evidence_<name>.json, <name>_results.json
SET = os.environ.get('L1T_SET', '')
EV_OUT = WORK + ('/evidence_%s.json' % SET if SET else '/evidence_l1t.json')
OUT = WORK + ('/%s_results.json' % SET if SET else '/l1t_results.json')
MAX_ATTEMPTS = 4
_lock = threading.Lock()
_HF = WORK + ('/%s_hints.json' % SET if SET else '/l1t_hints.json')
HINTS = json.load(open(_HF, encoding='utf-8')) if os.path.exists(_HF) else {}
# l1t_nr.json = writeback_report.json not_repaired at the time of the L1T run (frozen: the writeback after L1T shrinks it)
NR = list(HINTS) if SET else json.load(open(WORK + '/l1t_nr.json', encoding='utf-8'))
TYPES = ('TYPE_A', 'TYPE_B', 'TYPE_C', 'TYPE_D')

TYPE_RULE_T = {
    'TYPE_A': 'TYPE_A (quantitative): the question asks for ONE value; the answer is one number with unit (several '
              'numbers only if the question has sub-questions, each one value); answer_range = null. A range-valued '
              'input cannot be used unless the quote also states a single (typical) value — use that value.',
    'TYPE_B': 'TYPE_B (state prediction): if a required input is a range (an evidence range or a range observed in '
              'the video), the answer is a range "lo–hi unit" with answer_range = [lo, hi] obtained by propagating '
              'exactly those input ranges (no invented spread, no ± margins); otherwise one number with unit and '
              'answer_range = null.',
    'TYPE_C': 'TYPE_C (causal / constraint): the question asks whether something happens / holds; the answer starts '
              'with "Yes" or "No" followed by the quantitative justification (the computed numbers); answer_range = null.',
    'TYPE_D': 'TYPE_D (comparative): the question compares two quantities / scenarios; the answer states the '
              'comparison direction and the computed number(s) (ratio or difference); answer_range = [lo, hi] only if '
              'the answer is a range obtained from input ranges, else null.',
}

REWRITE_T = """You repair ONE physics QA of a video benchmark. Earlier repairs failed because the real web evidence
does not fit the QA in its current form. You may now change the question type and reword the question; the
difficulty is fixed. Produce the corrected QA.

HARD RULES
1. difficulty = {diff} is fixed. question_type: choose one of {types} (current: {qtype}) — the type whose answer form
   the evidence supports exactly. Rules of the allowed types:
{type_rules}
2. QUESTION: {qrule} Never put a retrieved (web) value into the question. The question must ask for exactly what the
   chosen type answers (e.g. TYPE_A asks for one value, not a "range"; TYPE_C asks a yes/no question) and stay about
   the same video event and physical phenomenon.
3. EVIDENCE facts: the corresponding search step must read web_search(query='<evidence query>') with exactly that
   query; its web_search source gets url = evidence url and key_information = the fact as the evidence states it
   (plus the converted value used). Use the evidence value (unit-converted if needed) everywhere the fact is used:
   steps, code, answer. If the quote states a number that fits better than the extracted value, you may use that
   number — report it in "evidence_used". Evidence values are typical values for the kind of object; use them.
4. REMOVE facts: delete their search step and web_search source and every use of their value; reformulate the
   solution (and if needed the question) so the fact is not needed, or use the evidence named in the guidance.
5. VERIFIED facts stay as they are (same query, url, value).
6. Only EVIDENCE and VERIFIED facts may be web_search sources, and every web_search source must be retrieved by a
   web_search step of the solution. A value without evidence must not be used (no invented values, assumptions or
   margins for retrieved quantities). UNLINKED sources: with evidence, add/merge a search step with the evidence
   query; marked "no search needed" -> remove that web_search source.
7. Video observations: the video_observation sources stay exactly unchanged; values seen in / measured from the video
   keep their values.
8. Code (code_executor sources): runnable Python (math, numpy available); hardcode exactly the input values the
   solution uses (video observations + retrieved values), never introduce or tune a number to reach an answer; print
   every result with label and unit; the LAST code block prints the final answer value(s) (both ends for a range).
   The code's printed values are the truth: steps after the code and the answer must use exactly those values.
9. {hidden_rule}
10. Keep the step structure, physics model and conventions of the current solution where they remain valid; change
   only what the evidence, the removals, the type and the guidance require. Write it as a clean benchmark QA: no
   references to repairs, reviews, previous versions or hidden values.

GUIDANCE FOR THIS QA (follow it):
{guidance}

VIDEO (description of the clip): {video}

CURRENT QA:
{qa}

EVIDENCE (real search results; use these values):
{evidence}

REMOVE facts:
{remove}

VERIFIED facts (keep unchanged):
{verified}

CODE PROBLEMS:
{code_problems}
{feedback}
Return ONLY JSON:
{{"question_type": "TYPE_?", "question": "...", "annotated_solution": ["Step 1: ...", ...],
"sources": [{{"type": "video_observation", ...}}, ...], "answer": "...", "answer_range": null or [lo, hi],
"hidden_cv_chosen": [...],
"evidence_used": {{"<evidence number>": {{"quote_number": "<number as written in the quote>", "value_used": "<value with unit as used in the code>"}}, ...}},
"change_summary": "<what changed and why>"}}"""

HIDDEN_B_T = ('hidden_cv_chosen lists the retrieved (web_search) values the solver has to find, as "quantity = value"; '
              'it must list exactly {n} entries (fixed by the difficulty), each a retrieved value the new solution '
              'really uses, with the value as used; write each entry cleanly.')


# ---------------------------------------------------------------- evidence
def _log_rows():
    for line in open(WORK + '/search_log.jsonl', encoding='utf-8'):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        for rank, o in enumerate(d['raw'].get('organic', []), 1):
            yield d['query'], rank, o


def near_any(x, ys, tol=1e-6):
    return any(abs(x - y) <= tol * max(abs(x), abs(y), 1e-12) for y in ys)


def resolve(item):
    """real search result of a log item: link contains item['url'], snippet contains item['quote'] verbatim
    (whitespace-normalised); the query must not contain a value number; item['query'] preferred"""
    vnums = [x for x in text_nums(item.get('value_converted') or item['value']) if x > 0]
    qn = [x for x in text_nums(item['quote'])]
    if not all(near_any(x, qn, 1e-6) for x in text_nums(item['value']) if x > 0):
        return None, 'value %s not in quote' % item['value']
    cands = []
    for query, rank, o in _log_rows():
        sn = ' '.join((o.get('snippet') or '').split())
        if item['url'] not in (o.get('link') or '') or item['quote'] not in sn:
            continue
        if any(near_any(x, text_nums(query)) for x in text_nums(item['value']) if x > 0):
            continue
        cands.append((query != item.get('query'), '"' in query, rank, query, o))
    if not cands:
        return None, 'no logged result (url part %s, quote verbatim, value-free query)' % item['url']
    _, _, rank, query, o = min(cands, key=lambda c: c[:3])
    return {'found': True, 'query': query, 'rank': rank, 'url': o['link'], 'title': o.get('title'),
            'snippet': o.get('snippet'), 'quote': item['quote'], 'value': item['value'],
            'value_converted': item.get('value_converted'), 'is_range': bool(item.get('is_range')),
            'why': 'L1T targeted search'}, None


def build_evidence(qid):
    e1 = json.load(open(WORK + '/evidence.json', encoding='utf-8')).get(qid) or {}
    e2 = json.load(open(WORK + '/evidence_l1e.json', encoding='utf-8')).get(qid)
    items = copy.deepcopy(e2 if e2 else e1)
    h = HINTS[qid]
    errs = []
    for k in h.get('drop', []):
        if k not in items:
            errs.append('drop key %s unknown' % k)
    dropped = {k: items.pop(k) for k in list(items) if k in h.get('drop', [])}
    for li in h.get('log', []):
        if li.get('direct'):  # the L0-verified quote (page text) of the existing source states the value used now
            ev, why = {'found': True, 'query': li['query'], 'rank': 0, 'url': li['url'], 'title': None, 'snippet': None,
                       'quote': li['quote'], 'value': li['value'], 'value_converted': li.get('value_converted'),
                       'is_range': bool(li.get('is_range')), 'why': 'L0-verified quote of the existing source'}, None
        else:
            ev, why = resolve(li)
        if ev is None:
            errs.append('%s: %s' % (li['key'], why))
            continue
        if li['key'] in items:
            items[li['key']]['evidence'] = ev
            if li.get('target'):
                items[li['key']]['target'] = li['target']
        else:
            # kind 'ws': replaces the value of an existing search step (li['current']), else a new fact
            items[li['key']] = {'kind': li.get('kind', 'ws_new'), 'target': li.get('target') or li['key'], 'current': li.get('current', ''),
                                'unit': None, 'single': False, 'evidence': ev}
    for k, it in items.items():
        e = it['evidence']
        if not (e.get('found') or e.get('no_search_needed')):
            errs.append('%s has no evidence and is neither dropped nor replaced' % k)
    return items, dropped, errs


def run_evidence(todo):
    ev_all = json.load(open(EV_OUT, encoding='utf-8')) if os.path.exists(EV_OUT) else {}
    bad = 0
    for q in todo:
        items, dropped, errs = build_evidence(q)
        ev_all[q] = {'items': items, 'dropped': dropped}
        if errs:
            bad += 1
            print('[%d] %s' % (NR.index(q) + 1, q), errs)
    json.dump(ev_all, open(EV_OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(todo), 'QAs,', bad, 'with evidence errors')


# ---------------------------------------------------------------- rewrite
def fmt_evidence(items):
    out = []
    for i, it in enumerate(items, 1):
        e = it['evidence']
        if it['kind'] == 'ws_unlinked' and e.get('no_search_needed'):
            out.append('%d. UNLINKED source #%d "%s": no search needed (%s) -> remove this web_search source.'
                       % (i, it['source_idx'], it['target'], e.get('why')))
            continue
        what = ('UNLINKED source #%d "%s"' % (it['source_idx'], it['target']) if it['kind'] == 'ws_unlinked'
                else 'NEW fact "%s" (add its search step and web_search source)' % it['target'] if it['kind'] == 'ws_new'
                else 'search step for "%s" (current value %s)' % (it['target'], it['current']))
        out.append('%d. %s\n   query: %s\n   url: %s\n   page title: %s\n   quote: "%s"\n   value: %s  (converted: %s)'
                   % (i, what, e['query'], e['url'], e.get('title'), e['quote'], e['value'], e.get('value_converted')))
    return '\n'.join(out) or 'none'


def fmt_remove(dropped):
    return '\n'.join('- "%s" (current value %s)' % (it['target'], it['current']) for it in dropped.values()) or 'none'


def fmt_verified(qa, items, dropped):
    """VERIFIED = L0-verified web_search sources that are not one of the facts being replaced / removed"""
    cur = [x for it in list(items) + list(dropped) for x in text_nums(it.get('current')) if x > 0]
    keep = [s for s in sources_of(qa, 'web_search') if s.get('verified_quote') and not any(
        near_any(x, text_nums(s.get('key_information')), 1e-3) for x in cur)]
    return '\n'.join('- query "%s" -> %s (%s)' % (s.get('search_query'), s.get('key_information'), s.get('url'))
                     for s in keep) or 'none'


def history(qid):
    """why earlier repairs failed: review notes + last errors of L1E / L1 / L2"""
    out = []
    rv = R_REVIEW.get(qid)
    if rv and rv.get('verdict') == 'reject':
        out.append('review (%s): %s' % (rv['level'], rv['notes'][:400]))
    for lv in ('L1E', 'L1', 'L2'):
        r = RES_OLD[lv].get(qid)
        if not r:
            continue
        a = (r.get('attempts') or [{}])[-1]
        info = a.get('errors') or a.get('evidence_rejected')
        if info:
            out.append('%s: %s' % (lv, str(info)[:500]))
    return out


def allowed_nums(new, items):
    nums = list(text_nums(new.get('question') or ''))
    for s in new.get('sources') or []:
        if s.get('type') == 'video_observation' or (s.get('type') == 'web_search' and s.get('verified_quote')):
            nums += text_nums(str(s.get('key_information') or ''))
    for it in items:
        e = it['evidence']
        nums += text_nums(e.get('quote') or '') + text_nums(str(e.get('value') or '')) + \
            text_nums(str(e.get('value_converted') or ''))
    return [x for x in nums if x > 0]


def filter_current(items, allowed):
    out = copy.deepcopy(items)
    for it in out:
        if it['kind'] == 'ws_unresolved':
            keep = [x for x in text_nums(it['current']) if x > 0 and not any(R.near(x, y, 0.001) for y in allowed)]
            it['current'] = ' '.join('%g' % x for x in keep)
    return out


def check_extra(qid, old, new, d, h, dropped, allowed, outs):
    """L1T checks on top of l1_rewrite.validate"""
    cls = qid.split('|')[0][-1]
    errs = []
    qt = new.get('question_type')
    if qt not in h['types']:
        errs.append('question_type must be one of %s' % h['types'])
    code_l = R.lits(' '.join(c.get('code') or '' for c in sources_of(new, 'code_executor')))
    for it in dropped.values():  # removed facts: their old values must be gone from the code
        for x in text_nums(it['current']):
            if x > 0 and x not in (1, 2, 3, 9.81, 9.8) and not any(R.near(x, y, 0.001) for y in allowed) \
                    and any(R.near(x, y, 0.001) for y in code_l):
                errs.append('removed fact "%s": its value %s is still used in the code' % (it['target'][:60], x))
    if cls == 'B':
        n_old, hid = len(old.get('hidden_cv_chosen') or []), new.get('hidden_cv_chosen') or []
        if len(hid) != n_old:
            errs.append('hidden_cv_chosen must have exactly %d entries (difficulty), got %d' % (n_old, len(hid)))
        pool = code_l + text_nums('\n'.join(get_steps(new)[0])) + text_nums(' '.join(outs)) + \
            [x for s in sources_of(new, 'web_search') for x in text_nums(str(s.get('key_information') or ''))]
        for e in hid:
            if R.GARBLED.search(str(e)):
                errs.append('garbled hidden_cv_chosen entry "%s"' % e)
            elif not text_nums(str(e)):
                errs.append('hidden_cv_chosen entry "%s" has no value' % e)
            elif not all(any(R.near(x, y, 0.03) for y in pool) for x in text_nums(str(e)) if x > 0):
                errs.append('hidden_cv_chosen entry "%s" is not a value the solution uses' % e)
    return errs


def normalize(old, d, cls):
    new = R.normalize(old, d, cls)
    ov = [s for s in old['sources'] if s.get('type') == 'video_observation']
    new['sources'] = copy.deepcopy(ov) + [s for s in new.get('sources') or [] if s.get('type') != 'video_observation']
    if d.get('question_type') in TYPES:
        new['question_type'] = d['question_type']
    return new


def repair(qid):
    if qid in res:
        return
    h = HINTS[qid]
    old = copy.deepcopy(CUR[qid])
    cls = qid.split('|')[0][-1]
    ev = EV[qid]
    items, dropped = list(ev['items'].values()), ev['dropped']
    diff = old.get('difficulty')
    if cls == 'B':
        hidden_rule = HIDDEN_B_T.format(n=len(old.get('hidden_cv_chosen') or []))
    else:
        hidden_rule = R.HIDDEN_A
    if h.get('qedit', True):
        qrule = ('you may reword the question (e.g. to ask for the quantity the evidence determines or to fit the '
                 'chosen type), but every value stated in it (video observations / given data) stays in it with the '
                 'same value%s.' % ('; you may add only these values: %s' % h['qnums'] if h.get('qnums') else
                                    '; do not add numbers'))
    else:
        qrule = 'keep the question text exactly unchanged.'
    guidance = '\n'.join('- ' + x for x in h.get('notes', [])) or '- none'
    hist = history(qid)
    if hist:
        guidance += ('\nEarlier repairs of this QA failed for these reasons (the type may now change; avoid the '
                     'mistakes):\n' + '\n'.join('- ' + x for x in hist))
    video = video_context(qid) or '(no description)'
    attempts, feedback, prev, status, new = [], '', None, 'fail', None
    for a in range(MAX_ATTEMPTS):
        prompt = REWRITE_T.format(
            diff=diff, types=h['types'], qtype=old.get('question_type'),
            type_rules='\n'.join('   - ' + TYPE_RULE_T[t] for t in h['types']), qrule=qrule, hidden_rule=hidden_rule,
            guidance=guidance, video=video, qa=R.view(old, cls), evidence=fmt_evidence(items),
            remove=fmt_remove(dropped), verified=fmt_verified(old, items, dropped.values()),
            code_problems=R.fmt_code(qid) if qid in R.problems else 'none (but the code must still reproduce the new answer)',
            feedback=feedback)
        d = call_llm(prompt, temperature=0.2)
        if not isinstance(d, dict):
            attempts.append({'errors': ['no JSON']})
            continue
        cand = normalize(old, d, cls)
        qt = cand.get('question_type')
        # web_search sources with an evidence url: verification metadata (several sources may share one url;
        # L0-verified sources keep their own query / quote)
        for it in items:
            e = it['evidence']
            if e.get('found'):
                for s in sources_of(cand, 'web_search'):
                    if s.get('url') == e['url'] and not s.get('verified_quote'):
                        s['search_query'], s['verified_quote'] = e['query'], e['quote']
        rng = cand.get('answer_range')
        old2 = dict(old, question_type=qt, answer_range=rng if qt in ('TYPE_B', 'TYPE_D') else None)
        allowed = allowed_nums(cand, items)
        used = d.get('evidence_used') if isinstance(d.get('evidence_used'), dict) else {}
        errs, warns, outs = R.validate(qid, old2, cand, filter_current(items, allowed), used)
        fixed = []
        for e in errs:
            if e.startswith('question text gained numbers'):
                gained = [x for x in set(text_nums(cand['question'])) - set(text_nums(old['question'])) if x > 0]
                bad = [x for x in gained if not any(R.near(x, y, 0.001) for y in h.get('qnums', []))]
                if bad:
                    fixed.append('question text gained numbers %s — only these may be added: %s'
                                 % (sorted(bad), h.get('qnums', [])))
                continue
            fixed.append(e)
        errs = fixed + check_extra(qid, old, cand, d, h, dropped, allowed, outs)
        if not errs:
            rv = call_llm(R.REVIEW.format(type_rule=TYPE_RULE_T[qt], qtype=qt, diff=diff, qa=R.view(cand, cls),
                                          outs=json.dumps(outs, ensure_ascii=False),
                                          quotes='\n'.join('- %s: "%s"' % (x.get('url'), x.get('verified_quote'))
                                                           for x in sources_of(cand, 'web_search')) or 'none')
                          + R.A_HIDDEN * (cls == 'A')) or {}
            if not rv.get('pass', False):
                errs = ['review: ' + x for x in rv.get('issues') or ['review failed without issues']]
        attempts.append({'question_type': qt, 'errors': errs, 'warnings': warns, 'change_summary': d.get('change_summary')})
        if not errs:
            status, new = 'ok', cand
            break
        prev = cand
        feedback = ('\nYOUR PREVIOUS ATTEMPT (fix these problems, keep the rules):\n%s\nPROBLEMS:\n- %s\n'
                    % (R.view(cand, cls), '\n- '.join(errs)))
    q = new if new is not None else prev
    rec = {'status': status, 'attempts': attempts, 'qa': q, 'evidence': list(ev['items']),
           'dropped': list(dropped), 'question_type': (q or {}).get('question_type')}
    with _lock:
        res[qid] = rec
        json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[%d]' % (NR.index(qid) + 1), qid, status, (q or {}).get('question_type'), 'attempts=%d' % len(attempts),
          '' if status == 'ok' else str(attempts[-1].get('errors'))[:400], flush=True)


def run_rewrite(todo):
    global res, EV, CUR, R_REVIEW, RES_OLD
    res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    EV = json.load(open(EV_OUT, encoding='utf-8'))
    CUR = {q: qa for q, (qa, lv) in current(skip=('L1T', 'FIX39', 'FIX39W')).items() if q in NR}
    R_REVIEW = json.load(open(WORK + '/review_log.json', encoding='utf-8'))
    RES_OLD = {lv: json.load(open('%s/%s' % (WORK, f), encoding='utf-8')) for lv, f in (
        ('L1', 'l1_results.json'), ('L2', 'l2_results.json'), ('L1E', 'l1e_results.json'))}
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(repair, todo))
    print(dict(Counter(r['status'] for q, r in res.items() if q in todo)))


res = {}
if __name__ == '__main__':
    mode = sys.argv[1]
    args = sys.argv[2:]
    todo = [NR[int(a) - 1] if a.isdigit() else a for a in args] or [q for q in NR if q in HINTS]
    run_evidence(todo) if mode == 'evidence' else run_rewrite(todo)
