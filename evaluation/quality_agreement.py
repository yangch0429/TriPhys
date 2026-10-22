"""Model-judge agreement on question quality (appendix Table "agreement").

GPT-5.5 and Gemini-3.6-Flash rate every question on is_question_understandable and is_answer_correct_and_unique from
text only (question, reference answer, reference solution, video-measured and retrieved hidden values); the third judge
is the label stored in the dataset (the construction-time quality check). A request that keeps failing (API or JSON
error) is recorded as an error and excluded, never counted as "no".

  export OPENROUTER_API_KEY=...
  python evaluation/quality_agreement.py run   --out results/quality_agreement.jsonl
  python evaluation/quality_agreement.py stats --out results/quality_agreement.jsonl"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triphys import config, data, llm

FIELDS = ('is_question_understandable', 'is_answer_correct_and_unique')

SYSTEM = '你是专业的物理教育评估专家。严格按照标准给出客观评价。注意题目中的隐藏值（hidden values）是设计的一部分。'

PROMPT = """你是物理教育质量评估专家。请评估以下视频物理题的两个维度。

**重要说明**：
本题是视频分析物理题，解题会用到三类物理量：
1. 题面给出的量：可以直接使用。
2. 视频测量值（隐藏值）：由视频CV测量获得，题面不直接给出，但在标准解答步骤中使用。
3. 检索得到的外部物理量（隐藏值）：如物体规格、材料常数、典型参数，题面不直接给出，由解题者用web_search检索获得，并在标准解答步骤中使用。
视频测量值和检索得到的外部物理量都是题目的特意设计，不属于"缺少关键信息"。**每个视频测量值都有唯一确定的数值；每个检索得到的外部物理量用标准解答中的检索query都能返回唯一的值（或唯一的标准区间）。**
你只能看到文本，看不到视频；不要因为看不到视频而判定题目不可理解或答案不唯一。

**本题的视频测量值**：
{video_text}

**本题检索得到的外部物理量**：
{web_text}

**题目**:
{question}

**标准答案**:
{answer}

**标准解答步骤**:
{solution}

---

请评估以下两个维度，每个维度回答"yes"或"no"：

1. **is_question_understandable**: 题目是否表述清晰、逻辑完整、可理解？
   - yes: 题目表述清晰，任何具有高中物理知识的人都能理解题目要求，知道需要求什么。注意：视频测量值和检索得到的外部物理量的存在不影响题目可理解性，它们都是唯一确定的。
   - no: 题目表述模糊、逻辑混乱、存在歧义、或无法理解题目意图（与隐藏值无关的问题）

2. **is_answer_correct_and_unique**: 答案是否正确且唯一？
   - yes: 根据题面给出的量、标准解答步骤、视频测量值和检索得到的外部物理量（都是唯一确定的），答案在物理上正确，且在合理误差范围内是唯一确定的；区间答案以该区间本身作为唯一答案
   - no: 答案错误、或解答步骤有明显物理错误、或即使给定上述各值仍存在多个同样合理的答案

**返回格式（纯JSON，不要markdown代码块标记）**:
{{
  "is_question_understandable": "yes",
  "is_answer_correct_and_unique": "yes",
  "reasoning": "简要说明你的判断依据（2-3句话）"
}}"""


def build_prompt(sub, q):
    """MPR hidden_cv_chosen = video-measured values; KPR hidden_cv_chosen = retrieved values"""
    hidden = q.get('hidden_cv_chosen') or []
    video = hidden if sub == 'MPR' else []
    web = ['  - %s' % v for v in hidden] if sub == 'KPR' else []
    web += ['  - 检索query: %s → 检索结果: %s' % (s.get('search_query', ''), s.get('key_information', ''))
            for s in q.get('sources', []) if s.get('type') == 'web_search']
    sol = q.get('annotated_solution', [])
    return PROMPT.format(video_text='\n'.join('  - %s' % v for v in video) or '  无', web_text='\n'.join(web) or '  无',
                         question=q.get('question', ''), answer=q.get('answer') or q.get('answer_range', ''),
                         solution='\n'.join(sol) if isinstance(sol, list) else str(sol))


def parse(text):
    d = llm.json_block(text)
    res = {}
    for f in FIELDS:
        v = str(d.get(f, '')).strip().lower()
        if v not in ('yes', 'no'):
            raise ValueError('bad %s: %r' % (f, v))
        res[f] = v
    res['reasoning'] = d.get('reasoning', '')
    return res


async def judge(model, prompt, sem):
    last = ''
    async with sem:
        for attempt in range(4):
            try:
                resp = await llm.chat(config.CHAT_API_BASE, config.chat_key(model), model,
                                      [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
                                      temperature=0.0, max_tokens=4000, timeout=300.0, retries=4)
                return parse(llm.content(resp))
            except Exception as e:                       # API or parse failure -> retry, then record as error
                last = '%s: %s' % (type(e).__name__, str(e)[:200])
    return {'error': last}


async def run(a):
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding='utf-8'):
            r = json.loads(l)
            if 'error' not in r['result']:
                done.add((r['qid'], r['model']))
    items = [x for s in ('MPR', 'KPR') for x in data.load_split(s, 'all', only_valid=False)]
    if a.limit:
        items = items[:a.limit]
    jobs = [(x, m) for x in items for m in a.models.split(',') if (x['qid'], m) not in done]
    print('questions %d, jobs %d (cached %d)' % (len(items), len(jobs), len(done)))
    sem = asyncio.Semaphore(a.concurrency)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    async def one(x, m):
        res = await judge(m, build_prompt(x['subtask'], x['qa']), sem)
        with open(a.out, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'qid': x['qid'], 'subtask': x['subtask'], 'model': m, 'result': res},
                               ensure_ascii=False) + '\n')
    await asyncio.gather(*(one(*j) for j in jobs))


def stats(a):
    models = a.models.split(',')
    res, sub = {}, {}
    for l in open(a.out, encoding='utf-8'):
        r = json.loads(l)
        if 'error' not in r['result']:
            res.setdefault(r['qid'], {})[r['model']] = r['result']
            sub[r['qid']] = r['subtask']
    stored = {x['qid']: x['qa'] for s in ('MPR', 'KPR') for x in data.load_split(s, 'all', only_valid=False)}
    out = {}
    for s in ('MPR', 'KPR', 'All'):
        ids = [i for i in res if all(m in res[i] for m in models) and (s == 'All' or sub[i] == s)]
        st = {'assessed': len(ids)}
        for f in FIELDS:
            g, m = ([res[i][mm][f] for i in ids] for mm in models[:2])
            ds = [str(stored[i].get(f, '')).lower() for i in ids]
            st[f] = {'all_three_agree': sum(x == y == z for x, y, z in zip(g, m, ds)),
                     'negative_%s' % models[0]: g.count('no'), 'negative_%s' % models[1]: m.count('no'),
                     'negative_both': sum(x == y == 'no' for x, y in zip(g, m)),
                     'pairwise_agreement': round(sum(x == y for x, y in zip(g, m)) / max(1, len(ids)), 4)}
        out[s] = st
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'stats'])
    ap.add_argument('--out', default='results/quality_agreement.jsonl')
    ap.add_argument('--models', default='%s,%s' % (config.GPT_MODEL, config.GEMINI_MODEL))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--concurrency', type=int, default=6)
    a = ap.parse_args()
    asyncio.run(run(a)) if a.cmd == 'run' else stats(a)


if __name__ == '__main__':
    main()
