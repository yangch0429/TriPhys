# Provenance (TriPhysBench release): fix_code/_v8_work/fix5.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""FIX5: repair of the problems found by the comprehensive recheck of the written benchmark; applied after FIX4,
question_type / difficulty / hidden_cv_chosen unchanged (B-class: garbled hidden entries cleaned, same count), web
evidence unchanged (no new searches; the LLM reports a needed search in "needs_search" for a manual fix).
  collect  fix5_issues.json {qid: [issues]} from strict_check.json (code does not print the answer / range, hidden value
           not used), deep_audit_qa.json, deep_audit_video.json (contradictions between questions of one video) and
           va_out.json (video_audit.py: question claims contradicted by the video)
  repair   per QA the LLM (deepseek-flash) judges every issue (real / false, the benchmark conventions are accepted:
           joint speed used as the released object's speed, typical values, stated idealizations) and, if one is real,
           returns the corrected question / steps / code / answer. Checks: code runs and prints the answer (and both
           range ends), every hidden value used by the code (unless unused already before / A-class justified),
           video_observation sources corrected with the question, final_recheck.one, no new
           number in the question unless declared, independent review (l1_rewrite.REVIEW). 4 attempts.
           -> fix5_results.json {qid: {status: ok|keep|fail, qa, verdicts, change, attempts}}   (resumable)
  accept   accepted results (after the manual review: fix5_reject.json lists the rejected qids) -> review_post.json FIX5
FIX6 (FIXLEVEL=FIX6): the same repair applied after FIX5 to the claims of video_audit.py only (va_out.json); A-class
claims about the magnitude of a CV-measured speed are left out (hidden_cv_chosen are fixed CV values).
Usage: [FIXLEVEL=FIX6] python fix5.py collect | repair [qid ...] | accept"""
import copy, json, os, re, sys, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import l1_rewrite as R
from fixlib import WORK, call_llm, get_steps, set_steps, sources_of, run_code, text_nums, consistent, video_context
from current import current
from final_recheck import one
from strict_check import problems as strict_problems

LEVEL = os.environ.get('FIXLEVEL', 'FIX5')
OUT = WORK + '/%s_results.json' % LEVEL.lower()
ISSUES = WORK + '/%s_issues.json' % LEVEL.lower()
REJECT = WORK + '/%s_reject.json' % LEVEL.lower()
MAX_ATTEMPTS = 4
_lock = threading.Lock()

PROMPT = """You check and, if needed, repair ONE physics QA of a video benchmark. The solver sees the video and the
question, may use web_search and code_executor, and must reproduce the reference answer. Automatic checks raised the
ISSUES below; some are false alarms. Judge every issue, then repair the QA if at least one is real.

ACCEPTED BENCHMARK CONVENTIONS (never real issues): a joint speed (wrist / elbow / body) measured by the CV pipeline
used as the released object's / hand's speed; typical / specification values for the kind of object seen; point-mass,
rigid-body, no-loss models and idealizations the question states; angles "x° relative to ..."; ±% spreads derived from
the hidden value ranges; rounding within 2 %. A hidden value counts as used when the code uses its range
(e.g. 1.5641-1.6671 m for 'height = 1.6 m') or the same value in other units; an A-class hidden value the solution does
not need (e.g. a speed of another phase) is not an issue: list it in "unused_hidden" with the reason. Never add a
value to the code only to "use" it. Audit flags based on the benchmark conventions are false alarms.
REAL issues: the answer (or answer_range) is not what the code prints; the code uses a value different from the one
the question / hidden values / sources give; an absurd result (e.g. kinetic energy of a wrist computed with the whole
body mass; a basketball diameter of 0.3 m used as a height offset); wrong formula / unit / arithmetic; inverted range;
the solution answers a different question; two questions of the same video state different values for the same
quantity of the same event; the question describes an action / object the video does not show at that time.

REPAIR RULES (only if an issue is real)
1. question_type {qtype} and difficulty {diff} are fixed; the answer form must fit the type: {type_rule}
   If the current answer has an answer_range, keep a range of nonzero width obtained by propagating the input ranges
   the solution already uses (no invented ± margins).
2. Keep the web_search sources exactly (same queries, urls, values). Do not add searches. If a fix needs a different
   retrieved value, keep the QA and describe it in "needs_search". If you correct a value / action the question
   states about the video, correct the matching video_observation source too (field "video_observations").
3. hidden_cv_chosen: {hidden_rule}
4. Question: keep it about the same video event and quantity; keep every value it states (you may correct a stated
   value only to resolve a contradiction with another question of the same video / with the video, or a description
   of the action to what the video shows). Never state a hidden or retrieved value. Declare every number you add to
   the question in "added_numbers".
5. Code: self-contained Python that prints every number the answer states (both ends of a range, labelled). Steps:
   "Step n: ..." lines; the web_search steps keep the form web_search(query='<query>'); the numbers in the steps equal
   the code output; the final step states the answer.
6. Fix the real issues with the smallest change; do not touch what is correct.

VIDEO (automatic description): {video}
OTHER QUESTIONS OF THE SAME VIDEO:
{others}

QA ({qid}):
{qa}
hidden_cv_chosen: {hidden}
Verbatim quotes of the web_search sources:
{quotes}
Current code outputs: {outs}

ISSUES:
{issues}
{feedback}
Return ONLY JSON:
{{"verdicts": [{{"issue": <number>, "real": true|false, "why": "<short>"}}],
 "action": "keep" | "fix",
 "needs_search": null | "<what retrieved value is needed and why>",
 "unused_hidden": {{}} | {{"<hidden entry>": "<why the code legitimately does not use it>"}},
 "question": "<question>", "steps": ["Step 1: ...", ...], "codes": ["<code block>", ...],
 "answer": "<answer>", "answer_range": null | [lo, hi], "hidden_cv_chosen": [...],
 "added_numbers": [], "video_observations": [] | [{{"i": <0-based index among the video_observation sources>,
 "time_point": "<...>", "key_information": "<corrected>"}}], "change": "<what changed and why>"}}
(for "keep" only verdicts / action / needs_search / unused_hidden are needed)"""


def video_of(qid):
    return qid.split(':')[-1].rsplit('|', 1)[0]


def collect():
    cur = current()
    iss = defaultdict(list)
    if LEVEL == 'FIX6':
        for q, cs in json.load(open(WORK + '/va_out.json', encoding='utf-8')).items():
            for c in cs or []:
                if q.split(':')[-1].startswith('A') and re.search(r'speed|m/s|velocity', c.get('claim') or '', re.I):
                    continue
                iss[q].append('[video] the question claims "%s" but the video shows: %s' % (c.get('claim'), c.get('video')))
        iss = {q: v for q, v in iss.items() if v and q in cur}
        json.dump(iss, open(ISSUES, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('QAs with issues', len(iss), Counter(q.split(':')[-1][0] for q in iss))
        return
    for q, ps in json.load(open(WORK + '/strict_check.json', encoding='utf-8')).items():
        iss[q] += ['[strict] ' + p for p in ps]
    for q, ps in json.load(open(WORK + '/deep_audit_qa.json', encoding='utf-8')).items():
        for p in ps or []:
            iss[q].append('[audit:%s] %s (hint: %s)' % (p.get('category'), p.get('evidence'), p.get('fix_hint')))
    for v, cs in json.load(open(WORK + '/deep_audit_video.json', encoding='utf-8')).items():
        for c in cs or []:
            qs = [q for q in c.get('qas') or [] if q in cur]
            for q in qs:
                iss[q].append('[video-consistency] %s: %s — %s (questions %s)' % (
                    c.get('quantity'), c.get('values'), c.get('why'), ', '.join(qs)))
    va = WORK + '/va_out.json'
    for q, cs in (json.load(open(va, encoding='utf-8')) if os.path.exists(va) else {}).items():
        for c in cs or []:
            iss[q].append('[video] the question claims "%s" but the video shows: %s' % (c.get('claim'), c.get('video')))
    iss = {q: v for q, v in iss.items() if v and q in cur}
    json.dump(iss, open(ISSUES, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('QAs with issues', len(iss), Counter(x.split(']')[0][1:].split(':')[0] for v in iss.values() for x in v))


def outs_of(qa):
    return [run_code(s.get('code') or '')['out'][-1500:] for s in sources_of(qa, 'code_executor')]


def quotes_of(qa):
    return '\n'.join('- %s: "%s"' % (s.get('url'), s.get('verified_quote'))
                     for s in sources_of(qa, 'web_search')) or 'none'


def build(old, d):
    new = copy.deepcopy(old)
    new['question'] = d.get('question') or old['question']
    new['answer'] = d.get('answer') or old['answer']
    new['answer_range'] = d.get('answer_range')
    if isinstance(d.get('hidden_cv_chosen'), list) and old['qid_cls'] == 'B':
        new['hidden_cv_chosen'] = d['hidden_cv_chosen']
    codes = [c for c in d.get('codes') or [] if isinstance(c, str) and c.strip()]
    srcs = [s for s in old['sources'] if s.get('type') != 'code_executor']
    for c in codes:
        o = run_code(c)['out'].strip()
        srcs.append({'type': 'code_executor', 'code': c, 'key_information': o, 'key_information_annotated': None})
    vo = [s for s in srcs if s.get('type') == 'video_observation']
    for e in d.get('video_observations') or []:
        if isinstance(e, dict) and isinstance(e.get('i'), int) and 0 <= e['i'] < len(vo) and e.get('key_information'):
            vo[e['i']]['key_information'] = str(e['key_information'])
            if e.get('time_point'):
                vo[e['i']]['time_point'] = str(e['time_point'])
    new['sources'] = srcs
    steps = [str(s) for s in d.get('steps') or []]
    set_steps(new, steps, get_steps(old)[1])
    if 'total_steps' in new:
        new['total_steps'] = len(steps)
    new.pop('qid_cls', None)
    return new


def validate(qid, old, new, d):
    cls = qid.split('|')[0][-1]
    errs = []
    for k in ('question_type', 'difficulty'):
        if new.get(k) != old.get(k):
            errs.append('%s changed' % k)
    oh, nh = old.get('hidden_cv_chosen') or [], new.get('hidden_cv_chosen') or []
    if cls == 'A' and nh != oh:
        errs.append('A-class hidden_cv_chosen must stay unchanged')
    if len(nh) != len(oh):
        errs.append('hidden_cv_chosen must keep %d entries' % len(oh))
    for e in nh:
        if R.GARBLED.search(str(e)) or re.search(r'=\s*[^\d\s.(≈~-]{1,4}\d|=\s*-[A-Z]', str(e)):
            errs.append('garbled hidden entry "%s"' % e)
    if [(s.get('search_query'), s.get('url')) for s in sources_of(new, 'web_search')] != \
            [(s.get('search_query'), s.get('url')) for s in sources_of(old, 'web_search')]:
        errs.append('web_search sources changed')
    ar = new.get('answer_range')
    if old.get('answer_range') and not (isinstance(ar, list) and len(ar) == 2 and ar[0] < ar[1]):
        errs.append('answer_range must be [lo, hi] with lo < hi (the original QA has a range)')
    if new['question_type'] == 'TYPE_C' and not str(new['answer']).strip().lower().startswith(('yes', 'no')):
        errs.append('TYPE_C answer must start with Yes / No')
    for i, s in enumerate(sources_of(new, 'code_executor')):
        r = run_code(s['code'])
        if not r['ok'] or r['out'] == 'No output':
            errs.append('code block %d fails / prints nothing: %s' % (i, r['out'][-300:]))
    unused = d.get('unused_hidden') if isinstance(d.get('unused_hidden'), dict) else {}
    pre = {p for p in strict_problems(old) if p.startswith('hidden:')}   # hidden value unused already before: not required
    for p in strict_problems(new):
        if p.startswith('hidden:') and (p in pre or cls == 'A' and any(h in p for h in unused)):
            continue
        errs.append(p)
    added = [x for x in set(text_nums(new['question'])) - set(text_nums(old['question'])) if x > 0]
    decl = [x for v in d.get('added_numbers') or [] for x in text_nums(str(v))]
    bad = [x for x in added if not any(R.near(x, y, 0.001) for y in decl)]
    if bad:
        errs.append('question gained undeclared numbers %s' % sorted(bad))
    hid = [x for h in nh for x in text_nums(str(h).split('=', 1)[-1])[:1] if x > 0]
    web = [x for s in sources_of(new, 'web_search') for x in text_nums(str(s.get('key_information')))[:2] if x > 0]
    leak = [x for x in added if any(R.near(x, y, 0.001) for y in hid + web)]
    if leak:
        errs.append('question states hidden / retrieved values %s' % leak)
    errs += one((qid, new))[1]
    return errs


def others_of(qid, cur):
    v = video_of(qid)
    return '\n'.join('[%s] %s' % (q, qa['question']) for q, (qa, _) in cur.items()
                     if q != qid and video_of(q) == v) or 'none'


def repair(qid):
    if qid in res:
        return
    old = copy.deepcopy(CUR[qid][0])
    cls = qid.split('|')[0][-1]
    issues = ISS[qid]
    hidden_rule = (R.HIDDEN_A if cls == 'A' else 'retrieved values the solver must find, "quantity = value"; keep the '
                   'same entries (fix only garbled text, e.g. "=ат 2.25 inches" -> "= 2.25 inches"); same count.')
    base = dict(qid=qid, qtype=old['question_type'], diff=old['difficulty'], type_rule=R.TYPE_RULE[old['question_type']],
                hidden_rule=hidden_rule, video=video_context(qid) or '(none)', others=others_of(qid, CUR),
                qa=R.view(old, 'A'), hidden=json.dumps(old.get('hidden_cv_chosen'), ensure_ascii=False),
                quotes=quotes_of(old), outs=json.dumps(outs_of(old), ensure_ascii=False),
                issues='\n'.join('%d. %s' % (i + 1, x) for i, x in enumerate(issues)))
    attempts, feedback, status, new, d = [], '', 'fail', None, {}
    for a in range(MAX_ATTEMPTS):
        d = call_llm(PROMPT.format(feedback=feedback, **base), temperature=0.2)
        if not isinstance(d, dict):
            attempts.append({'errors': ['no JSON']})
            continue
        if d.get('action') == 'keep':
            strict = [p for p in strict_problems(old) if not p.startswith('hidden:')]
            if strict:
                errs = ['action "keep" is not possible: the deterministic check still fails: ' + '; '.join(strict)]
            else:
                status = 'keep'
                attempts.append({'action': 'keep', 'errors': []})
                break
        else:
            cand = build(dict(old, qid_cls=cls), d)
            errs = validate(qid, old, cand, d)
            if not errs:
                outs = outs_of(cand)
                rv = call_llm(R.REVIEW.format(type_rule=R.TYPE_RULE[cand['question_type']], qtype=cand['question_type'],
                                              diff=cand['difficulty'], qa=R.view(cand, cls),
                                              outs=json.dumps(outs, ensure_ascii=False), quotes=quotes_of(cand))
                              + R.A_HIDDEN * (cls == 'A')) or {}
                if not rv.get('pass', False):
                    errs = ['review: ' + x for x in rv.get('issues') or ['review failed without issues']]
            if not errs:
                status, new = 'ok', cand
                attempts.append({'action': 'fix', 'errors': [], 'change': d.get('change')})
                break
            feedback = ('\nYOUR PREVIOUS ATTEMPT (fix these problems, keep the rules):\n%s\nPROBLEMS:\n- %s\n'
                        % (json.dumps({k: d.get(k) for k in ('question', 'steps', 'codes', 'answer', 'answer_range')},
                                      ensure_ascii=False, indent=1), '\n- '.join(errs)))
        attempts.append({'action': d.get('action'), 'errors': errs, 'change': d.get('change')})
        if d.get('action') == 'keep':
            feedback = '\nYOUR PREVIOUS ANSWER was "keep" but: %s\n' % '; '.join(errs)
    rec = {'status': status, 'qa': new, 'verdicts': d.get('verdicts'), 'needs_search': d.get('needs_search'),
           'unused_hidden': d.get('unused_hidden'), 'change': d.get('change'), 'issues': issues, 'attempts': attempts}
    with _lock:
        res[qid] = rec
        json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(qid, status, len(attempts), '' if status != 'fail' else str(attempts[-1].get('errors'))[:300], flush=True)


def accept():
    rej = json.load(open(REJECT, encoding='utf-8')) if os.path.exists(REJECT) else {}
    r = json.load(open(OUT, encoding='utf-8'))
    rp = json.load(open(WORK + '/review_post.json', encoding='utf-8'))
    n = 0
    for q, x in r.items():
        rp.get(q, {}).pop(LEVEL, None)
        if x['status'] == 'ok' and q not in rej:
            rp.setdefault(q, {})[LEVEL] = {'verdict': 'accept', 'notes': x.get('change')}
            n += 1
    json.dump(rp, open(WORK + '/review_post.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(LEVEL, 'accepted', n)


res = {}
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'collect':
        collect()
    elif mode == 'accept':
        accept()
    else:
        res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
        ISS = json.load(open(ISSUES, encoding='utf-8'))
        CUR = current(skip=(LEVEL,))
        todo = sys.argv[2:] or list(ISS)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(repair, todo))
        print(dict(Counter(res[q]['status'] for q in todo if q in res)))
