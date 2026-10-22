# Provenance (TriPhysBench release): fix_code/evaluation/fix10/recheck_ns.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""fix10 step 1b (10/12): dense re-check of V_not_shown claims for QAs whose only label is V_not_shown.
The recount sampled 10-16 frames; events between frames were reported as not_shown. Here only those claims are re-judged
on ~1 fps frames (max 48, plus the windows the question names). Claims that are measurements / numbers / physics
interpretations are dropped (not scene statements). absent / contradicted / invented -> needs_fix.
Output recount_final.json (= recount.json with recheck results merged).  Usage: python recheck_ns.py"""
import os, json, asyncio, base64
from collections import Counter
import recount as R
from recount import W, cv2

HERE = R.HERE


def dense_frames(path, question, width=480, cap_n=48):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    step = max(0.2, dur / cap_n)  # short B videos: up to 5 fps, never sparser than the recount
    ts = [step * (i + 0.5) for i in range(int(dur / step))]
    for a, b in R.WIN.findall(question):
        a, b = float(a), float(b)
        if a < b <= dur + 1:
            ts += [a + (b - a) * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    for t in R.AT_T.findall(question):
        if float(t) <= dur + 1:
            ts.append(float(t))
    ts = sorted({round(min(t, max(dur - 0.05, 0)), 2) for t in ts})
    out = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (width, int(width * fr.shape[0] / fr.shape[1])))
        out.append((t, base64.b64encode(cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]).decode()))
    return out, dur


RPROMPT = """A physics benchmark question is about the video whose frames follow (about one per second, timestamps given).
An earlier check on sparse frames questioned the statements below. Look through ALL frames and judge each statement.
QUESTION: %s
STATEMENTS:
%s
Labels:
- "not_scene": the statement is a measurement, number, speed / force / energy / centre-of-mass value, assumption or physical
  interpretation, not a statement about what the scene shows (e.g. "the wrist reaches a maximum speed") - drop it
- "visible": some frame shows it (give the timestamp)
- "plausible": not directly visible but consistent with the frames and the setting
- "contradicted": the frames show something different (e.g. a different object / colour / action)
- "invented": a specific identity (brand, league, model, standard) that nothing in the frames supports
- "absent": a scene object or event the question says happens, and no frame shows it
Return ONLY JSON: {"statements": [{"claim": "...", "label": "...", "evidence": "<timestamp + what is seen>"}]}"""


async def recheck(r, item, sem, cl=None):
    p = R.video_path(item)
    fr, dur = await asyncio.to_thread(dense_frames, p, r['question'])
    if not fr:
        return {'error': 'video not readable'}
    claims = '\n'.join('- ' + c.get('claim', '') for c in (r['video']['not_shown'] if cl is None else cl))
    content = [{'type': 'text', 'text': RPROMPT % (r['question'], claims)}]
    for t, b in fr:
        content += [{'type': 'text', 'text': 'Frame at t = %.1f s:' % t},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b}}]
    async with sem:
        for a in range(5):
            try:
                import httpx
                async with httpx.AsyncClient(timeout=900.0) as cl:
                    res = await cl.post(W.LLM_URL, headers={'Authorization': 'Bearer ' + W.LLM_KEY},
                                        json={'model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'), 'temperature': 0.0, 'max_tokens': 32000,
                                              'messages': [{'role': 'user', 'content': content}]})
                    t = res.json()['choices'][0]['message']['content']
                    j = W.pj(t) or {}
                    if j.get('statements') is not None:
                        st = [s for s in j['statements'] if isinstance(s, dict)]
                        return {'n_frames': len(fr), 'statements': st,
                                'bad': [s for s in st if s.get('label') in ('absent', 'contradicted', 'invented')]}
            except Exception as e:
                W.STAT['recheck_err ' + type(e).__name__] += 1
            await asyncio.sleep(4 * 2 ** a)
    return {'error': 'llm failed'}


async def main():
    """MODE=notshown (default): QAs whose only label is V_not_shown, their not_shown claims; input recount.json.
    MODE=contra: QAs with V_contradicted not yet re-checked, their contradicted + invented + not_shown claims; the V labels
    of these QAs are replaced by the dense result; input/output recount_final.json."""
    mode = os.environ.get('MODE', 'notshown')
    src = HERE + ('/recount_final.json' if mode == 'contra' else '/recount.json')
    rs = json.load(open(src))
    items = {i['id']: i for i in R.load()}
    if mode == 'contra':
        todo = [r for r in rs if 'V_contradicted' in r.get('labels', []) and not r['video'].get('recheck')]
    else:
        todo = [r for r in rs if r.get('labels') == ['V_not_shown']]
    print('recheck', mode, len(todo), flush=True)
    sem = asyncio.Semaphore(8)

    def claims(r):
        v = r['video']
        return v['not_shown'] if mode != 'contra' else v.get('contradicted', []) + v.get('invented', []) + v.get('not_shown', [])
    out = await asyncio.gather(*[recheck(r, items[r['id']], sem, claims(r)) for r in todo])
    for r, o in zip(todo, out):
        r['video']['recheck'] = o
        lab = Counter(s.get('label') for s in o.get('bad', []))
        new = ['V_absent'] * bool(lab['absent']) + ['V_contradicted'] * bool(lab['contradicted']) +               ['V_invented'] * bool(lab['invented'])
        keep = [l for l in r['labels'] if not (mode == 'contra' and l in ('V_contradicted', 'V_invented', 'V_not_shown'))]
        r['labels'] = sorted(set(keep + new + (['V_error'] if o.get('error') else [])))
        r['needs_fix'] = any(not l.startswith('V_not_shown') and not l.startswith('V_error') for l in r['labels'])
    json.dump(rs, open(HERE + '/recount_final.json', 'w'), ensure_ascii=False, indent=1)
    print('still V-bad among rechecked:', sum(bool(r['video']['recheck'].get('bad')) for r in todo), '/', len(todo))
    print(Counter(s.get('label') for r in todo for s in r['video']['recheck'].get('statements', [])))
    for r in todo:
        for s in r['video']['recheck'].get('bad', []):
            print(r['id'], s.get('label'), '|', str(s.get('claim'))[:90], '|', str(s.get('evidence'))[:140])


if __name__ == '__main__':
    asyncio.run(main())
