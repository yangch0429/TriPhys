# Provenance (TriPhysBench release): fix_code/_v8_work/consistency_check.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Cross-QA consistency of retrieved values: QAs of the same video that retrieve a property of the same object must
use the same value. Per video (train + test QAs together), the retrieved facts of every QA (web_search sources:
query + key_information; B-class hidden_cv_chosen) are compared by the reviewer LLM, which lists every pair of QAs
using different values for the same property of the same object.
Versions: orig = stage_orig; current = current.current(pending=True) (L0 <- accepted / pending repair <- consistency fix).
Usage: python consistency_check.py orig|current  -> consistency_<mode>.json"""
import json, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, load, iter_qas, sources_of, call_llm

PROMPT = """Below are the externally retrieved values used by different QAs about the SAME video. A benchmark rule says:
when two QAs retrieve the same property of the same object (e.g. the mass of the marble in this video, the friction
coefficient of glass on wood), they must use the same value.

Find every violation: two QAs that use DIFFERENT values for the same property of the same object / material pair.
- Same object means the same physical thing of this video (or the same material pair / same constant), not merely a
  similar word; e.g. "marble mass" and "marble diameter" are different properties; static and kinetic friction are
  different properties; "glass on wood" and "wood on wood" are different material pairs.
- Different values: numbers that differ after unit conversion (5 g vs 5.5 g; 0.3-0.5 vs 0.4), including a range vs a
  single value. Identical values written in other units are NOT violations.
- Ignore physical constants that are the same (g = 9.81 vs 9.80665 is a violation only if both are retrieved values).

{facts}

Return ONLY JSON: {{"violations": [{{"object_property": "...", "qa_values": {{"<qa label>": "<value>", ...}},
"note": "<one sentence>"}}]}}  (empty list if none)"""


def versions(mode):
    if mode == 'current':
        from current import current
        return {q: qa for q, (qa, lv) in current(pending=True).items()}
    out = {}
    for split in ('train', 'test'):
        for cls in 'ab':
            for qid, vid, qa in iter_qas(load(cls, split, 'orig'), cls, split):
                out[qid] = qa
    return out


def facts_of(qid, qa):
    lines = []
    for s in sources_of(qa, 'web_search'):
        lines.append('  - retrieved: %s (query: %s)' % (str(s.get('key_information'))[:300], s.get('search_query') or '-'))
    if qid.split(':')[-1].startswith('B|'):
        for h in qa.get('hidden_cv_chosen') or []:
            lines.append('  - hidden value: %s' % h)
    return lines


def check(item):
    video, qids, qas = item
    blocks = []
    for q in qids:
        f = facts_of(q, qas[q])
        if f:
            blocks.append('QA %s:\n  question: %s\n%s' % (q, qas[q]['question'][:300], '\n'.join(f)))
    if len(blocks) < 2:
        return video, []
    d = call_llm(PROMPT.format(facts='\n\n'.join(blocks)), temperature=0) or {}
    return video, d.get('violations') or []


def main(mode):
    qas = versions(mode)
    by_video = defaultdict(list)
    for q in qas:
        by_video[q.split(':')[-1].rsplit('|', 1)[0]].append(q)
    items = [(v, qs, qas) for v, qs in by_video.items() if len(qs) >= 2]
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(check, items))
    res = {v: x for v, x in res.items() if x}
    json.dump(res, open(WORK + '/consistency_%s.json' % mode, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(mode, 'videos checked', len(items), '| with violations', len(res),
          '| violations', sum(len(x) for x in res.values()))


if __name__ == '__main__':
    main(sys.argv[1])
