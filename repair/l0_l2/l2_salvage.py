# Provenance (TriPhysBench release): fix_code/_v8_work/l2_salvage.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Re-check of failed L2 attempts with the corrected checks (no new generation): every attempt of a failed QA
(l2_messages/<qid>.json, latest attempt first) is re-extracted from its messages and validated again with the patched
validate_grounding (unit-aware hidden-value leak check, value-free re-search of queries that stated the retrieved
value, TYPE_B range from the code output) + text_checks + hidden_check; an attempt passing all checks goes through the
same reviewer as l2_regen. Passing QAs are stored as status ok with salvaged = attempt index.
Backup l2_results_before_salvage.json; log -> l2_salvage_log.json
Usage: python l2_salvage.py [qid ...]"""
import json, os, re, shutil, sys
from concurrent.futures import ThreadPoolExecutor
import l2_regen as R
from l2_regen import G, qas, text_checks, hidden_check, _requery, REVIEW, TYPE_RULE, view, A_HIDDEN, REVIEW_LOG
from fixlib import WORK, call_llm, sources_of


def qa_of(messages):
    full = ''.join(m.get('content') or '' for m in messages if isinstance(m, dict) and m.get('role') == 'assistant')
    q = G.extract_qa_pairs(full)
    return q[-1] if q else None


def salvage(qid):
    old, cls = qas[qid], qid.split(':')[-1][0]
    qt, diff = old['question_type'], old['difficulty']
    p = '%s/%s.json' % (R.MSG_DIR, re.sub(r'[^\w.-]+', '_', qid))
    if not os.path.exists(p):
        return qid, None, 'no messages'
    runs = json.load(open(p, encoding='utf-8'))
    last = 'no attempt'
    for i in reversed(range(len(runs))):
        qa = qa_of(runs[i])
        if qa is None:
            continue
        errs = G.validate_grounding(qa, runs[i], diff, qt, 'B' if cls == 'B' else 'A', _requery)
        errs += text_checks(old, qa) + hidden_check(qa, cls)
        if errs:
            last = errs[:2]
            continue
        outs = [s.get('key_information') for s in sources_of(qa, 'code_executor')]
        quotes = '\n'.join('- %s: %s' % (s.get('url'), s.get('key_information')) for s in sources_of(qa, 'web_search'))
        rv = call_llm(REVIEW.format(
            type_rule=TYPE_RULE.get(qt, ''), qtype=qt, diff=diff, qa=view(qa, cls.upper()),
            outs=json.dumps(outs, ensure_ascii=False), quotes=quotes or 'none').replace(
            '- retrieved values match', '- every retrieved value is needed (the answer changes when it changes);\n'
            '- retrieved values match') + A_HIDDEN * (cls == 'A')) or {}
        if rv.get('pass'):
            qa['difficulty'] = diff
            return qid, {'qa': qa, 'review': rv, 'attempt': i}, 'ok'
        last = ['review: ' + x for x in (rv.get('issues') or ['review failed'])][:2]
    return qid, None, last


def main(todo):
    res = json.load(open(R.OUT, encoding='utf-8'))
    shutil.copy(R.OUT, WORK + '/l2_results_before_salvage.json')
    todo = todo or [q for q, r in res.items() if r['status'] != 'ok'
                    or (REVIEW_LOG.get(q, {}).get('level') == 'L2' and REVIEW_LOG[q]['verdict'] == 'reject')]
    log = {}
    with ThreadPoolExecutor(6) as ex:
        for qid, got, why in ex.map(salvage, todo):
            log[qid] = {'result': 'ok' if got else 'fail', 'why': why}
            print(qid, 'ok' if got else 'fail', str(why)[:200], flush=True)
            if got:
                res[qid].update({'status': 'ok', 'qa': got['qa'], 'review': got['review'], 'salvaged': got['attempt']})
    json.dump(res, open(R.OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(log, open(WORK + '/l2_salvage_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('salvaged', sum(v['result'] == 'ok' for v in log.values()), 'of', len(log))


if __name__ == '__main__':
    main(sys.argv[1:])
