# Provenance (TriPhysBench release): fix_code/_v8_work/video_audit.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Video grounding check of every QA (server): Gemini (OpenAI-compatible endpoint CHAT_API_BASE, the clip re-encoded and sent as base64 as in
run_api_subset.py) compares what the QUESTION and the solution's video observations state about the video
(objects, actions, time windows, colours, counts, rough sizes / speeds / durations) with the video itself and lists
clear contradictions only. Usage: python video_audit.py IN_JSON OUT_JSON"""
import asyncio, json, os, re, sys
import httpx
sys.path.insert(0, os.environ.get('FIX1001_DIR', '.'))  # server eval dir with run_api_subset.py (not released)
from run_api_subset import video_b64  # noqa: E402
URL = os.environ.get('CHAT_API_BASE', 'https://openrouter.ai/api/v1') + '/chat/completions'

PROMPT = """You check a question of a video-based physics benchmark against the video (attached). The question and the
annotated video observations below describe what is seen in the video.

QUESTION: {question}
VIDEO OBSERVATIONS (from the reference solution): {obs}

Watch the video and list every CLEAR contradiction between these descriptions and the video: an object / person /
material that is not there or is a different kind of thing, an action or event that does not happen (at all or in the
stated time window, allow +-2 s), a wrong colour or count, a stated duration / size / speed that is off by more than a
factor of 2 from what is plainly visible. Angles written as "x deg relative to ..." are measured by a CV tool: ignore them.
Do not report things that are merely not verifiable, assumptions the question states ("assuming ...", "typical ..."),
or small estimation differences.
Return ONLY JSON: {{"contradictions": [{{"claim": "<text>", "video": "<what the video shows>"}}]}}  (empty list if none)"""


async def ask(item):
    mid, key = os.environ.get('GEMINI_MODEL', 'google/gemini-3.6-flash'), os.environ.get('OPENROUTER_API_KEY', '')
    content = [{'type': 'image_url', 'image_url': {'url': 'data:video/mp4;base64,' + video_b64(item['path'])}},
               {'type': 'text', 'text': PROMPT.format(question=item['question'],
                                                      obs=json.dumps(item['observations'], ensure_ascii=False))}]
    for attempt in range(15):
        try:
            async with httpx.AsyncClient(timeout=600.0) as c:
                r = await c.post(URL, headers={'Authorization': 'Bearer ' + key},
                                 json={'model': mid, 'messages': [{'role': 'user', 'content': content}],
                                       'max_tokens': 4096, 'temperature': 0})
            if r.status_code == 200:
                t = r.json()['choices'][0]['message'].get('content') or ''
                m = re.search(r'\{.*\}', t, re.S)
                if m:
                    return json.loads(m.group(0)).get('contradictions')
        except Exception as e:
            print('err', str(e)[:120], flush=True)
        await asyncio.sleep(3 if attempt < 5 else 15)
    return None


async def main(inp, outp):
    items = json.load(open(inp, encoding='utf-8'))
    res = json.load(open(outp, encoding='utf-8')) if os.path.exists(outp) else {}
    sem = asyncio.Semaphore(4)

    async def one(it):
        async with sem:
            r = await ask(it)
            res[it['qid']] = r
            json.dump(res, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(it['qid'], 'FAILED' if r is None else len(r), flush=True)

    await asyncio.gather(*(one(it) for it in items if res.get(it['qid']) is None))
    print('done', len(res), 'flagged', sum(1 for r in res.values() if r), 'failed', sum(1 for r in res.values() if r is None))


if __name__ == '__main__':
    asyncio.run(main(*sys.argv[1:3]))
