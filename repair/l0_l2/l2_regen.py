# Provenance (TriPhysBench release): fix_code/_v8_work/l2_regen.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L2: regenerate the QAs that L1 could not repair (no real evidence for a fact / L1 failed) with the benchmark's own
Stage 3 generator (fix_code/benchmark_construction/agentic_physics_qa.generate_single_qa):
same video (cached stage1 / 2b / 2.5 outputs; physics matches from _shared_physics_library.json via stage2b),
same difficulty, same question_type; generator model deepseek-v3.2 (as in the original construction), tools
web_search (live Serper keys, raw results in search_log.jsonl) / page_browser / code_executor; every source must be a
real tool output (validate_grounding); up to 4 attempts, then an independent LLM review (as in L1); a review
rejection triggers one more generation round with the review issues as feedback.
Output: l2_results.json {qid: {status: ok|fail, qa, attempts: [{errors, action_count}], review}}  (resumable);
        l2_messages/<qid>.json  full agent transcripts of every attempt
Usage: python l2_regen.py [qid ...]      (default: every need_L2 / fail QA of l1_results.json)"""
import asyncio, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.environ.get('CONSTRUCTION_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmark_construction'))
from openai import AsyncOpenAI
import agentic_physics_qa as G
from fixlib import WORK, call_llm, call_serper, load, iter_qas, stage_dir, get_steps, sources_of
from l1_rewrite import REVIEW, TYPE_RULE, view, text_checks

OUT = 'l2_results.json'
MSG_DIR = WORK + '/l2_messages'
GEN_MODEL = os.environ.get('DEEPSEEK_GEN_MODEL', 'deepseek-v3.2')
# deepseek-v3.2 (the construction / original judge model), called through an OpenAI-compatible relay
GEN_KEY = os.environ.get('OPENROUTER_API_KEY', '')
CONCURRENT = 10
REVIEW_ROUNDS = 2
res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
qas = {}
for split in ('train', 'test'):
    for cls in 'ab':
        for qid, vid, qa in iter_qas(load(cls, split, 'L0'), cls, split):
            qas[qid] = qa


async def _web_search(query, top_k=10):
    try:
        r = await asyncio.to_thread(call_serper, query, 'L2')
    except RuntimeError as ex:  # all Serper keys out of credits: stop instead of producing search-less failures
        print('ABORT:', ex, flush=True)
        os._exit(3)
    return [{'id': i + 1, 'title': x['title'], 'url': x['link'], 'snippet': x['snippet']} for i, x in enumerate(r or [])][:top_k]


G.web_search = _web_search  # execute_tool resolves web_search from the module namespace


def _requery(query):
    """validate_grounding: re-search a query with the retrieved value stripped (logged as tag L2-requery)"""
    try:
        return [{'url': x['link'], 'title': x['title'], 'snippet': x['snippet']} for x in call_serper(query, 'L2-requery') or []]
    except RuntimeError as ex:
        print('ABORT:', ex, flush=True)
        os._exit(3)


A_HIDDEN = ('\nNote (A-class): the values of hidden_cv_chosen are measured from the video by the benchmark\'s CV pipeline '
            '(e.g. joint speeds); the solver has to estimate them from the video, so they need no source — do not flag '
            'them for missing provenance.')


REVIEW_LOG = json.load(open(WORK + '/review_log.json', encoding='utf-8')) if os.path.exists(WORK + '/review_log.json') else {}


def hidden_check(qa, cls):
    if cls != 'B':
        return []
    bad = [h for h in qa.get('hidden_cv_chosen') or [] if not re.search(r'=\s*\S*\d', str(h))]
    return ['hidden_cv_chosen entries %s must be written "name = value unit" with the retrieved value' % bad] if bad else []


def avoid_text(old):
    steps, _ = get_steps(old)
    return ('\n## PREVIOUS QA OF THIS SLOT (to be rebuilt)\nThe previous QA of this video for this slot could not be '
            'grounded in real search results:\nquestion: "%s"\nanswer: "%s"\nsolution:\n%s\n'
            'Rebuild it: keep the question as close to it as possible (same observed event, same asked quantity, same '
            'answer form, same kind of hidden values), but search for every retrieved value yourself and use only '
            'values your web_search / page_browser calls actually returned (do not assume a value you have not seen '
            'in a result); recompute everything with code_executor. If a hidden value cannot be found, replace it by '
            'another retrieved property of the same object that the calculation needs, and adjust the question '
            'minimally. Keep the values the question states (they are given data) and the answer form of the question '
            'type (TYPE_B: a range [low, high]; TYPE_A / TYPE_C: answer_range null); every number the solution uses '
            'must come from the question, the video or a search result — do not assume unsourced values (speeds, '
            'sizes, efficiencies). B-class hidden_cv_chosen entries are written "name = value unit". Search for the '
            'PROPERTY, never for a value: a web_search query must not contain the number you expect (e.g. search '
            '"glass marble mass grams", not "glass marble mass 5 g"); copy the value exactly as the returned snippet / '
            'page states it, and cite only urls your calls returned.\n'
            % (old.get('question', ''), old.get('answer', ''), '\n'.join(steps)[:3000]))


async def regen(qid, client, sem):
    if qid in res and res[qid]['status'] == 'ok' and (REVIEW_LOG.get(qid) or {}).get('verdict') != 'reject':
        return
    old = qas[qid]
    cls = qid.split(':')[-1][0]
    d = stage_dir(qid)
    if d is None:
        res[qid] = {'status': 'fail', 'reason': 'no stage outputs'}
        return
    item = G.load_stage3_item(d, cls)
    qt, diff = old['question_type'], old['difficulty']
    all_attempts, feedback = [], ''
    rev = REVIEW_LOG.get(qid) or {}
    if rev.get('level') == 'L2' and rev.get('verdict') == 'reject':  # rejected in the one-by-one review
        feedback = ('\n## THE LAST QA WAS REJECTED BY THE REVIEWER\nIssues:\n- ' + rev['notes'] +
                    '\nBuild the QA again so that these issues are fixed.\n')
    async with sem:
        # up to 2 rounds: a QA that passes the programmatic checks but not the review is regenerated once with the
        # review issues as feedback
        for rnd in range(REVIEW_ROUNDS):
            out = await G.generate_single_qa(client, item, diff, qt, model_name=GEN_MODEL,
                                             avoid=avoid_text(old) + feedback,
                                             extra_check=lambda qa: text_checks(old, qa) + hidden_check(qa, cls),
                                             requery=_requery)
            all_attempts += out['attempts']
            qa, review = out['qa'], None
            if qa is None:
                break
            outs = [s.get('key_information') for s in sources_of(qa, 'code_executor')]
            quotes = '\n'.join('- %s: %s' % (s.get('url'), s.get('key_information')) for s in sources_of(qa, 'web_search'))
            review = await asyncio.to_thread(call_llm, REVIEW.format(
                type_rule=TYPE_RULE.get(qt, ''), qtype=qt, diff=diff, qa=view(qa, cls.upper()),
                outs=json.dumps(outs, ensure_ascii=False), quotes=quotes or 'none').replace(
                '- retrieved values match', '- every retrieved value is needed (the answer changes when it changes);\n'
                '- retrieved values match') + A_HIDDEN * (cls == 'A')) or {}
            if review.get('pass'):
                break
            issues = review.get('issues') or ['review failed']
            all_attempts[-1]['errors'] = ['review: ' + x for x in issues]
            qa = None
            feedback = ('\n## THE LAST QA WAS REJECTED BY THE REVIEWER\nIssues:\n- ' + '\n- '.join(issues) +
                        '\nBuild the QA again so that these issues are fixed.\n')
    out['attempts'] = all_attempts
    os.makedirs(MSG_DIR, exist_ok=True)
    json.dump([a['messages'] for a in out['attempts']], open('%s/%s.json' % (MSG_DIR, re.sub(r'[^\w.-]+', '_', qid)), 'w',
                                                              encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    rec = {'status': 'ok' if qa is not None else 'fail', 'qa': qa if qa is not None else out['attempts'][-1]['qa'],
           'attempts': [{'errors': a['errors'], 'action_count': a['action_count']} for a in out['attempts']],
           'review': review}
    res[qid] = rec
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(qid, rec['status'], [a['errors'][:2] for a in rec['attempts']], flush=True)


async def main(todo):
    client = AsyncOpenAI(base_url=os.environ.get('CHAT_API_BASE', 'https://openrouter.ai/api/v1'), api_key=GEN_KEY)
    sem = asyncio.Semaphore(CONCURRENT)
    await asyncio.gather(*(regen(q, client, sem) for q in todo))
    await client.close()
    print(dict(Counter(r['status'] for q, r in res.items() if q in todo)))


if __name__ == '__main__':
    todo = sys.argv[1:]
    if not todo:
        l1 = json.load(open('l1_results.json', encoding='utf-8'))
        todo = [q for q, r in l1.items() if r['status'] != 'ok']
    unknown = [q for q in todo if q not in qas]
    assert not unknown, 'unknown qids: %s' % unknown
    asyncio.run(main(todo))
