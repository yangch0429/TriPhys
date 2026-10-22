# Provenance (TriPhysBench release): fix_code/_v8_work/deep_audit.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Deep recheck of the written benchmark (fix_code/*class_qa_{train,test}.json), beyond final_recheck.py (quotes, steps,
answer format, code reruns). The reviewer LLM (deepseek-flash) audits
  qa     every QA on its own: physical plausibility of inputs / result for the scene of the video (stage1 description),
         validity of the physical model and the computation, question <-> solution agreement (the solution computes what
         is asked, with the values the question gives), well-posedness, answer <-> code output agreement;
  video  all QAs of a video together: the values / descriptions the QUESTIONS state about the same object or event must
         not contradict each other (consistency_check.py covers the retrieved values only).
Only major issues (the reference answer is wrong / meaningless or the question cannot be answered as posed) are listed;
every flag is then reviewed manually.
Output: deep_audit_qa.json {qid: [issues]}, deep_audit_video.json {video: [issues]}   (resumable)
Usage: python deep_audit.py qa|video"""
import json, os, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, load, iter_qas, get_steps, sources_of, run_code, call_llm, video_context

QA_PROMPT = """You audit one question of a video-based physics reasoning benchmark. The model being evaluated sees the
video and the question, may use web search and a code executor, and must reproduce the reference answer. Values the
question calls "typical / standard / specified" are retrieved by web search (the WEB SOURCES); values it states are
given; values observed in the video come from the video.

VIDEO (automatic description): {video}

QUESTION ({qtype}, {difficulty}): {question}
REFERENCE ANSWER: {answer}   (answer_range: {answer_range})
HIDDEN VALUES (must be retrieved / measured by the model): {hidden}
WEB SOURCES:
{sources}
SOLUTION STEPS:
{steps}
CODE (and its actual output when executed):
{codes}

Find MAJOR problems only, i.e. problems that make the reference answer wrong or physically meaningless, or the question
unanswerable as posed:
 implausible  an input or the result is physically impossible / absurd for the scene (e.g. more liquid poured than the
              container holds, a speed or mass off by a large factor for the object seen); a typical value that clearly
              concerns a different kind / size of object than the one in the video
 model        wrong formula, wrong physics, unit / conversion error, the model applied far outside its validity so that
              the number is meaningless, arithmetic error
 mismatch     the solution answers a different question; a value stated in the question differs from the one used; the
              answer's quantity / unit differs from what is asked; the answer text disagrees with the code output
 ill_posed    a needed input is neither given, observable in the video, nor retrievable as a typical value; or the
              question admits clearly different readings leading to very different answers
Do NOT report: style, rounding within 2 %, idealizations the question states (e.g. "assuming ...", Torricelli,
no air resistance), typical-value choices that are reasonable, the use of hidden values, video angles given as
"x° relative to ..." (a fixed benchmark convention), missing significant-figure discussion.
Return ONLY JSON: {{"issues": [{{"category": "implausible|model|mismatch|ill_posed", "evidence": "<concrete, with numbers>",
"fix_hint": "<short>"}}]}}  (empty list if none)"""

VIDEO_PROMPT = """Below are all questions of a video-based physics benchmark about the SAME video (automatic video
description first). Each question may state values or descriptions of what is seen (sizes, speeds, durations, masses,
heights, counts, object identities, colours, materials, time windows of an event).

Find every CONTRADICTION between two questions: they state different values / descriptions for the same quantity of the
same object or event (e.g. one says the stick moves at 0.3 m/s, another at 0.15 m/s during the same 4-8 s; one calls the
ball a tennis ball, another a baseball). Different time windows, different objects, different quantities, or a value
given in one question and asked for / assumed differently in another as a stated hypothetical ("assuming ...") are NOT
contradictions. Identical values in other units are not contradictions.

VIDEO: {video}

{questions}

Return ONLY JSON: {{"contradictions": [{{"qas": ["<id>", "<id>"], "quantity": "<what>", "values": "<value in each>",
"why": "<short>"}}]}}  (empty list if none)"""


def items():
    out = []
    for split in ('train', 'test'):
        for cls in 'ab':
            for qid, vid, qa in iter_qas(load(cls, split), cls, split):
                out.append((qid, qa))
    return out


def fmt_sources(qa):
    lines = []
    for j, s in enumerate(sources_of(qa, 'web_search')):
        lines.append('  [%d] query=%r | key=%r | quote=%r' % (j, s.get('search_query') or s.get('query'),
                                                             s.get('key_information'), s.get('verified_quote')))
    return '\n'.join(lines) or '  (none)'


def fmt_codes(qa):
    out = []
    for i, s in enumerate(sources_of(qa, 'code_executor')):
        c = s.get('code') or ''
        out.append('--- block %d\n%s\n--- output\n%s' % (i, c, run_code(c)['out'][:1500]))
    return '\n'.join(out) or '(none)'


def audit_qa(item):
    qid, qa = item
    p = QA_PROMPT.format(video=video_context(qid) or '(none)', qtype=qa.get('question_type'),
                         difficulty=qa.get('difficulty'), question=qa.get('question'), answer=qa.get('answer'),
                         answer_range=qa.get('answer_range'), hidden=qa.get('hidden_cv_chosen'),
                         sources=fmt_sources(qa), steps='\n'.join(get_steps(qa)[0]), codes=fmt_codes(qa))
    d = call_llm(p, temperature=0) or {'issues': None}
    return qid, d.get('issues')


def video_of(qid):
    return qid.split(':')[-1].rsplit('|', 1)[0]


def audit_video(arg):
    v, qs = arg
    txt = '\n\n'.join('[%s] (%s, %s) %s' % (q, qa.get('question_type'), qa.get('difficulty'), qa.get('question'))
                      for q, qa in qs)
    d = call_llm(VIDEO_PROMPT.format(video=video_context(qs[0][0]) or '(none)', questions=txt), temperature=0) or {}
    return v, d.get('contradictions')


def run(mode):
    path = WORK + '/deep_audit_%s.json' % mode
    res = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    lock = threading.Lock()
    if mode == 'qa':
        todo = [it for it in items() if res.get(it[0]) is None]
        fn = audit_qa
    else:
        by = defaultdict(list)
        for q, qa in items():
            by[video_of(q)].append((q, qa))
        todo = [(v, qs) for v, qs in by.items() if len(qs) > 1 and res.get(v) is None]
        fn = audit_video
    print(mode, 'todo', len(todo), flush=True)

    def work(it):
        k, r = fn(it)
        with lock:
            res[k] = r
            json.dump(res, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(k, 'FAILED' if r is None else len(r), flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, todo))
    print('done', len(res), '| flagged', sum(1 for r in res.values() if r), '| failed',
          sum(1 for r in res.values() if r is None))


if __name__ == '__main__':
    run(sys.argv[1])
