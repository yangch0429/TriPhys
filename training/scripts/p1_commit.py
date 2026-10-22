"""10/17 Phase1 no-answer fix: "commit" continuations. 26/110 test rows of rftrw_bal_1e4 give no answer, 17 of them because
the model keeps re-searching a value it cannot find until the round limit; the RFT pool has almost no successful
trajectory that shows "search failed -> commit to an estimate -> answer" (A 7/272).
Source: stuck self-samples (results/p1_ss*, no <answer>, >= 2 tool rounds) on f10 TRAIN QAs. Prefix = the model's own
first K rounds (assistant + tool turns, K = 2). The last tool message gets a one-off note asking it to commit to an
estimate and answer; the model (Phase1 SFT model, T=0.7) writes the next turn(s) (up to 3 rounds; tools run, each tool result carries the note). Kept only if ACC >= .9 (GT verifier),
the reply has <answer>, and its think does not mention the note. The note is removed from the stored trajectory, so the
SFT sample is: own failed searches -> own committed answer. Eval is unchanged (no note there).
Usage (from the work dir $TRIPHYS_WORK): python scripts/p1_commit.py --model M --shard i --n_shards n --out results/p1_commit/shard_i.json"""
from srvpaths import N1, SCRIPTS  # machine paths, see srvpaths.py
import argparse, asyncio, json, os, re, sys
from collections import defaultdict
sys.path.insert(0, N1)
sys.path.insert(0, SCRIPTS)
import types
# eval_n1_sample.py ends with a bare asyncio.run(main()); load it without that line
_src = open(N1 + '/eval_n1_sample.py').read().rsplit('asyncio.run(main())', 1)[0]
S = types.ModuleType('eval_n1_sample')
S.__file__ = N1 + '/eval_n1_sample.py'
exec(compile(_src, S.__file__, 'exec'), S.__dict__)
import p1_rft_build as B
SYSTEM_PROMPT, build_user_prompt = S.SYSTEM_PROMPT, S.build_user_prompt
from prompts import SYSTEM_PROMPT, build_user_prompt
from response_parser import parse_response
from metrics_final import calculate_acc_with_fallback, calculate_scs_with_fallback

NOTE = ('\n\n[Note: the needed value may not be available online. Do not search for it again. Commit to a reasonable '
        'estimate from the video, the question and the results you already have, and give the final <answer> now.]')
LEAK = re.compile(r'\bnote\b|told to|instruct|commit', re.I)


def jobs(k=2, per_qa=2):
    idx = B.train_index()
    by = defaultdict(list)
    for x in B.load_rows(['p1_ss', 'p1_ss2']):
        t = x.get('turns') or []
        if x['question'] not in idx or not t or '<answer>' in t[-1]['content'] or x.get('serper_failed'):
            continue
        if sum(y['role'] == 'tool' for y in t) < k:
            continue
        by[x['question']].append(x)
    out = []
    for q in sorted(by):
        for x in by[q][:per_qa]:
            t, pre, n_asst = x['turns'], [], 0
            for y in t:
                if y['role'] == 'assistant':
                    if n_asst == k:
                        break
                    n_asst += 1
                pre.append({'role': y['role'], 'content': y['content']})
            out.append((q, idx[q], pre))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    J = jobs()[a.shard::a.n_shards]
    ev = S.SFTEvaluator(a.model, 0)
    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r['question'], r['j'], r['s']) for r in res}
    for j, (q, (cls, vpath, qa), pre) in enumerate(J):
        vi, vk = ev._get_video(vpath)
        base = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [{'type': 'video', 'video': vpath, 'fps': 1.0},
                                             {'type': 'text', 'text': build_user_prompt(q, qa['question_type'])}]}]
        for s in range(a.samples):
            if (q, j, s) in done:
                continue
            msgs = base + [dict(m) for m in pre]
            msgs[-1] = dict(msgs[-1], content=msgs[-1]['content'] + NOTE)
            new_turns = []
            for _ in range(3):   # the model may still compute (or search once more) before answering
                content, _n = ev._run_round(msgs, vi, vk, S.MAX_TOKENS)
                new_turns.append({'role': 'assistant', 'content': content})
                if re.search(r'<answer>.*?</answer>', content, re.DOTALL):
                    break
                calls = S.parse_tool_calls(content)
                if not calls:
                    break
                msgs.append({'role': 'assistant', 'content': content})
                for name, args in calls:
                    if name == 'code_executor':
                        r = json.dumps({'result': ev._exec_code(args.get('code', ''))})
                    elif name == 'web_search':
                        r = S.query_web_search(args.get('query', '') or args.get('q', ''), 'commit')
                    else:
                        r = json.dumps({'result': "Tool '%s' unavailable." % name})
                    new_turns.append({'role': 'tool', 'content': r})
                    msgs.append({'role': 'tool', 'content': r})
                msgs[-1] = dict(msgs[-1], content=msgs[-1]['content'] + NOTE)
            th = ' '.join(t['content'].split('</think>')[0] for t in new_turns if t['role'] == 'assistant')
            p = parse_response(content)
            rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': qa['answer'],
                  'answer_range': qa.get('answer_range'), 'annotated_solution': qa.get('annotated_solution', []),
                  'question': q, 'question_type': qa['question_type'], 'needs_llm_eval': p.needs_llm_eval,
                  'raw_response': content}
            acc = await calculate_acc_with_fallback(rd) if p.prediction else 0.0
            scs = (await calculate_scs_with_fallback(rd)).get('overall', 0.0) if p.prediction else 0.0
            res.append({'question': q, 'cls': cls, 'j': j, 's': s, 'video': vpath, 'question_type': qa['question_type'],
                        'turns': pre + new_turns, 'ACC': float(acc), 'SCS': float(scs),
                        'answered': bool(p.prediction), 'leak': bool(LEAK.search(th))})
            S.logger.info('commit %s j%d s%d ACC %.2f SCS %.2f leak %s | %s', cls, j, s, float(acc), float(scs),
                          res[-1]['leak'], str(p.prediction)[:50])
            json.dump(res, open(a.out, 'w'), ensure_ascii=False)
    S.logger.info('COMMIT SHARD DONE %d', len(res))


if __name__ == '__main__':
    asyncio.run(main())
