# Provenance (TriPhysBench release): fix_code/_v8_work/cons2_check.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Review dump of cons2_results.json: structural checks (difficulty, question_type, hidden count, A-class CV hidden
values unchanged), the plan item(s) of every target, the field diffs against the current version and a
final_recheck.one run. -> rev_cons2.txt, cons2_check.json {qid: problems}
Usage: python cons2_check.py"""
import json
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, get_steps, sources_of
from current import current
from final_recheck import one

R = json.load(open(WORK + '/cons2_results.json', encoding='utf-8'))
PLAN = json.load(open(WORK + '/cons2_plan.json', encoding='utf-8'))


def main():
    cur = current(skip=('CONS2',))
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(one, [(q, x['qa']) for q, x in R.items()]))
    out = []
    for q, x in R.items():
        o, n = cur[q][0], x['qa']
        st = []
        for k in ('difficulty', 'question_type'):
            if o[k] != n[k]:
                st.append('%s changed %s -> %s' % (k, o[k], n[k]))
        if len(o.get('hidden_cv_chosen') or []) != len(n.get('hidden_cv_chosen') or []):
            st.append('hidden count changed')
        if 'A|' in q and o.get('hidden_cv_chosen') != n.get('hidden_cv_chosen'):
            st.append('A-class CV hidden changed')
        out.append('=' * 100 + '\n%s  status=%s  %s\nSTRUCT %s\nRECHECK %s' % (q, x['status'], x.get('attempts'), st, res[q]))
        for p in PLAN:
            if q in p.get('targets', []):
                out.append('PLAN %s | canon %s | %s' % (p['property'], p['canonical_interval'],
                                                     (p['canonical'].get('key_information') or '')[:150]))
        if x.get('evidence'):
            out.append('EVIDENCE %s' % x['evidence'])
        for k in ('question', 'answer', 'answer_range', 'hidden_cv_chosen'):
            if o.get(k) != n.get(k):
                out.append('-%s: %s\n+%s: %s' % (k, o.get(k), k, n.get(k)))
        so, sn = sources_of(o, 'web_search'), sources_of(n, 'web_search')
        for i in range(max(len(so), len(sn))):
            a = so[i] if i < len(so) else {}
            b = sn[i] if i < len(sn) else {}
            if (a.get('key_information'), a.get('url')) != (b.get('key_information'), b.get('url')):
                out.append('-ws%d: %s | %s\n+ws%d: %s | %s | %s' % (i, a.get('key_information'), a.get('url'), i,
                           b.get('key_information'), b.get('url'), b.get('verified_quote')))
        a, b = get_steps(o)[0], get_steps(n)[0]
        for i in range(max(len(a), len(b))):
            if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None):
                out.append('+%s' % (b[i] if i < len(b) else '<removed> ' + a[i]))
        co, cn = sources_of(o, 'code_executor'), sources_of(n, 'code_executor')
        if [c.get('code') for c in co] != [c.get('code') for c in cn]:
            out.append('+CODE:\n%s' % cn[0]['code'] if cn else '+CODE removed')
    open(WORK + '/rev_cons2.txt', 'w', encoding='utf-8').write('\n'.join(out))
    json.dump(res, open(WORK + '/cons2_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for q in R:
        print(q, R[q]['status'], 'RECHECK-PROBLEMS' if res[q] else '')


if __name__ == '__main__':
    main()
