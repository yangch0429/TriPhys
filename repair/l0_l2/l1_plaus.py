# Provenance (TriPhysBench release): fix_code/_v8_work/l1_plaus.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""L1 step 1b: plausibility check of every evidence of evidence.json against the VIDEO (stage1 description).
  found evidence   -> LLM judges whether the retrieved value fits the object / situation seen in the video
                      (typical values for that kind of object are fine; flagged only when the source clearly concerns
                      a different kind / size / context or is off by a large factor).
                      implausible -> re-extract with the video description, excluding the rejected urls (max 2 rounds).
  TYPE_A with range evidence -> re-extract with the single-value rule.
  not found        -> one more extraction + new-query round with the video description.
Rejected evidence is kept in item['rejected']; item['plaus'] = {plausible, why}.
Output: evidence.json updated in place (resumable: items with 'plaus' are skipped)
Usage: python l1_plaus.py"""
import json, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fixlib import call_llm, video_context, SEARCH_LOG
import l1_evidence as L

_lock = threading.Lock()
ev = L.ev
logged = {}
for line in open(SEARCH_LOG, encoding='utf-8'):
    r = json.loads(line)
    logged[(r['tag'], r['query'])] = [{'title': o.get('title', ''), 'snippet': o.get('snippet', ''), 'link': o.get('link', '')}
                                      for o in r['raw'].get('organic', [])[:10]]

PLAUS = """A physics benchmark question is about the video described below. A value for one of its inputs was retrieved
from the web. Judge whether the retrieved value is appropriate for the object / situation in THIS video.
Typical or regulation values for that kind of object are acceptable by design (the exact object is unknown). Flag it
ONLY if the source clearly concerns a different kind of object, a different size class or context (e.g. a laboratory
apparatus vs. a desktop toy, a professional vs. a children's ball), a different quantity, or the value is physically
implausible for what the video shows.

Video: {video}
Question: {question}
Needed input: {target}
Retrieved: value {value} (converted {conv}); page "{title}" ({url}); quote: "{quote}"

Return ONLY JSON: {{"plausible": true/false, "why": "<short reason>"}}"""


def extra_groups(qid, it):
    tag = 'L1evidence|%s|%s' % (qid, it['kind'])
    return [{'query': q, 'results': logged[(tag, q)]} for q in it.get('searched') or [] if (tag, q) in logged]


def judge(qid, it, video):
    e = it['evidence']
    return call_llm(PLAUS.format(video=video or '(no description)', question=L.qas[qid]['question'][:1200],
                                 target=it['target'], value=e.get('value'), conv=e.get('value_converted'),
                                 title=e.get('title'), url=e.get('url'), quote=e.get('quote'))) or {}


def check(args):
    qid, k = args
    it = ev[qid][k]
    if 'plaus' in it or it['evidence'].get('no_search_needed'):
        return
    video = video_context(qid)
    question = L.qas[qid]['question']
    it.setdefault('rejected', [])
    it.setdefault('single', L.qas[qid].get('question_type') == 'TYPE_A')
    note = []
    for rnd in range(3):
        e = it['evidence']
        if e.get('found') and it.get('single') and e.get('is_range'):
            it['rejected'].append({'url': e['url'], 'value': e['value'], 'why': 'range for a single-value question'})
            note.append('range->redo')
        elif e.get('found'):
            j = judge(qid, it, video)
            if j.get('plausible', True):
                it['plaus'] = {'plausible': True, 'why': j.get('why'), 'rounds': rnd, 'note': note}
                break
            it['rejected'].append({'url': e['url'], 'value': e['value'], 'why': j.get('why')})
            note.append('implausible')
        elif rnd > 0:
            it['plaus'] = {'plausible': False, 'why': e.get('why'), 'rounds': rnd, 'note': note}
            break
        else:
            note.append('none->retry')
        if rnd == 2:
            it['evidence'] = {'found': False, 'why': 'all evidence rejected: ' + '; '.join(r['why'] or '' for r in it['rejected'])}
            it['plaus'] = {'plausible': False, 'why': 'rejected', 'rounds': rnd, 'note': note}
            break
        item = {x: it[x] for x in ('pos', 'source_idx', 'target', 'current', 'unit', 'single', 'searched') if x in it}
        it['evidence'] = L.solve(qid, it['kind'], item, question, video,
                                 exclude=tuple(r['url'] for r in it['rejected']), extra_groups=extra_groups(qid, it))
        it['searched'] = item['searched']
    with _lock:
        json.dump(ev, open(L.OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        e = it['evidence']
        print('%s %s %s -> %s %s' % (qid, k, it['plaus'].get('note'), 'FOUND ' + str(e.get('value')) if e.get('found') else 'none',
                                     (it['plaus'].get('why') or '')[:100]), flush=True)


if __name__ == '__main__':
    todo = [(q, k) for q, d in ev.items() for k in d]
    print('items', len(todo), flush=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(check, todo))
    c = Counter()
    for q, d in ev.items():
        for k, it in d.items():
            c['no_search' if it['evidence'].get('no_search_needed') else
              'found' if it['evidence'].get('found') else 'none'] += 1
            c['rejected_%d' % len(it.get('rejected', []))] += 1
    print(dict(c))
