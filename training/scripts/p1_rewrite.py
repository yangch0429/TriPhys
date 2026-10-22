"""10/17 Phase1 SCS fix: rewrite only the final <answer> block of RFT trajectories into a full Step-by-step reasoning
chain. Think text, tool calls and tool results stay exactly as the model generated them; the final answer string stays
the model's own (ACC >= .9 already). Why: 22% of A samples in rft_k3_bal end with a bare '<answer>1.28 m</answer>'
(SCS 0), and even correct eval answers get S_obs ~.56 because the chain does not name the observed/retrieved quantities.
Rewriter (Gemini-3.6-Flash, text only, not the SCS judge) sees the question, the trajectory, the final answer and the
train-set gold solution (only as a guide to which quantities/formula to name; values must come from the trajectory).
10/17 (user: annotated-solution variants are accepted by a DeepSeek Flash score, as gpt-4o did before): every rewrite is
scored by the eval SCS judge (metrics_final.calculate_scs_detailed = deepseek-flash, same parse as eval) against the gold
annotated solution and kept only if SCS >= max(MIN_SCS, the sample's own SCS); up to 3 tries, else the original sample is
kept when its own (Flash) SCS >= MIN_SCS, otherwise dropped. --prev OLD_OUT reuses earlier rewrites as the first candidate.
10/17: env RW_TRIES (default 3) sets the number of tries (A refill uses 6).
Usage (from the work dir $TRIPHYS_WORK, training/README.md): [RW_TRIES=6] python scripts/p1_rewrite.py IN_NAME OUT_NAME [--prev OLD_OUT]"""
from srvpaths import EVAL_LIB, SCRIPTS  # machine paths, see srvpaths.py
import asyncio, json, logging, os, re, sys
import httpx
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, EVAL_LIB)
import p1_rft_build as B
from response_parser import parse_response
from metrics_final import calculate_scs_detailed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('rw')
logging.getLogger('httpx').setLevel(logging.WARNING)
URL = os.environ.get('CHAT_API_BASE', 'https://openrouter.ai/api/v1').rstrip('/') + '/chat/completions'
MODEL, KEY = os.environ.get('GEMINI_MODEL', 'google/gemini-3.6-flash'), os.environ['OPENROUTER_API_KEY']
E = B.E
MIN_SCS = 0.75
TRIES = int(os.environ.get('RW_TRIES', 3))

PROMPT = """A solver answered a physics question about a video using tools. Rewrite ONLY its final reasoning chain.

Question: {q}

Solver trajectory (its own thinking, tool calls and tool results):
{traj}

Solver's final answer (fixed, do not change): {ans}

Reference solution by an expert (a guide to WHICH quantities, formula and steps matter; do not copy values that the
solver did not have):
{gold}

Write 4-6 lines, each starting with "Step k:", in this style:
Step 1: Observe the video - what is seen and the quantities read from the video or the question, with values.
Step 2: Identify the physical phenomenon - the principle and the formula in symbols.
Step 3: The values obtained from tools (e.g. web_search for '...' gave ...; say which value is taken and why).
Step 4: The computation with numbers substituted and the result.
Step 5: Conclusion - state the final answer and why it answers the question.
Rules: every number must appear in the question or in the solver trajectory (its thinking or tool results); never
mention a reference/expert/gold solution; plain text, no markdown; the chain must end at exactly the final answer.
Output only the Step lines."""


def traj_text(msgs):
    out = []
    for m in msgs[2:]:
        c = m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])
        if m['role'] == 'assistant':
            th = c.split('</think>')[0].replace('<think>', '').strip() if '</think>' in c else ''
            rest = c.split('</think>')[-1]
            calls = re.findall(r'<tool_call>.*?</tool_call>', rest, re.DOTALL)
            out.append('[thinking] ' + (th[:1500] + ' ... ' + th[-1500:] if len(th) > 3000 else th))
            out += ['[tool call] ' + x[:600] for x in calls]
        elif m['role'] == 'tool':
            out.append('[tool result] ' + c[:1500])
    return '\n'.join(out)


async def call(prompt):
    for a in range(10):
        try:
            async with httpx.AsyncClient(timeout=300.0) as cl:
                r = await cl.post(URL, headers={'Authorization': 'Bearer ' + KEY},
                                  json={'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                                        'max_tokens': 4096, 'temperature': 0.3, 'reasoning_effort': 'low'})
            if r.status_code == 200:
                c = (r.json().get('choices') or [{}])[0].get('message', {}).get('content')
                if c:
                    return c
            else:
                log.warning('HTTP %d %s', r.status_code, r.text[:150])
        except Exception as e:
            log.warning('err %s', str(e)[:150])
        await asyncio.sleep(5 * (a + 1))
    return ''


def steps_of(text):
    lines = [l.strip() for l in text.strip().splitlines() if re.match(r'\s*Step \d+:', l)]
    bad = re.search(r'reference|expert|gold solution', text, re.I)
    return lines if 4 <= len(lines) <= 8 and not bad else None


async def flash_scs(text, q, qa):
    p = parse_response(text)
    if not p.reasoning_chain:
        return 0.0, None
    gold = qa.get('annotated_solution', [])
    r = await calculate_scs_detailed([{'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain,
                                       'annotated_solution': gold, 'question': q, 'question_type': qa['question_type'],
                                       'ground_truth': qa['answer']}])
    return float(r.get('overall') or 0.0), r.get('details')


def key_of(o):
    return o['messages'][-1]['content'].split('<answer>')[0].rstrip()


async def main(inp, outn, prev=None):
    idx = B.train_index()
    data = json.load(open('%s/data/p1/%s.json' % (E, inp)))
    old = {key_of(o): o['messages'][-1]['content'] for o in json.load(open('%s/data/p1/%s.json' % (E, prev)))} if prev else {}
    sem, jsem = asyncio.Semaphore(16), asyncio.Semaphore(8)
    stats = {'rw_pass': 0, 'kept_orig': 0, 'drop': 0, 'tries': 0}

    async def one(o):
        msgs = o['messages']
        q = next(k for k in idx if k in msgs[1]['content'][1]['text'])
        qa = idx[q][2]
        last = msgs[-1]['content']
        ans = parse_response(last).prediction
        own = float(o['metadata'].get('SCS', 0.0))
        bar = max(MIN_SCS, own)
        head = last.split('<answer>')[0].rstrip()
        best = (-1.0, None, None)
        if ans:
            gold = qa.get('annotated_solution', [])
            gold = '\n'.join(gold) if isinstance(gold, list) else str(gold)
            p = PROMPT.format(q=q, traj=traj_text(msgs[:-1] + [{'role': 'assistant', 'content': head}]), ans=ans, gold=gold)
            for t in range(TRIES):
                if t == 0 and key_of(o) in old:
                    cand = old[key_of(o)]
                else:
                    async with sem:
                        st = steps_of(await call(p))
                    stats['tries'] += 1
                    if not st:
                        continue
                    cand = (head + '\n\n<answer>\nReasoning chain:\n' + '\n'.join(st) + '\n\nFinal answer: ' + ans
                            + '\n</answer>')
                async with jsem:
                    sc, dims = await flash_scs(cand, q, qa)
                if sc > best[0]:
                    best = (sc, cand, dims)
                if sc >= bar:
                    break
        if best[1] is not None and best[0] >= bar:
            stats['rw_pass'] += 1
            new = dict(o, messages=msgs[:-1] + [{'role': 'assistant', 'content': best[1]}])
            new['metadata'] = dict(o['metadata'], rewritten=True, SCS_rw=best[0], SCS_rw_dims=best[2])
            return new
        if own >= MIN_SCS:
            stats['kept_orig'] += 1
            return dict(o, metadata=dict(o['metadata'], rewritten=False, SCS_rw=max(best[0], 0.0)))
        stats['drop'] += 1
        return None

    res = [r for r in await asyncio.gather(*(one(o) for o in data)) if r]
    json.dump(res, open('%s/data/p1/%s.json' % (E, outn), 'w'), ensure_ascii=False, indent=1)
    sc = lambda c: [r['metadata'].get('SCS_rw') if r['metadata'].get('rewritten') else r['metadata']['SCS']
                    for r in res if r['metadata']['cls'] == c]
    log.info('REWRITE DONE %s -> %s: %s, A %d (Flash SCS %.3f) B %d (%.3f)', inp, outn, stats,
             len(sc('A')), sum(sc('A')) / max(1, len(sc('A'))), len(sc('B')), sum(sc('B')) / max(1, len(sc('B'))))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == '--prev' else None))
