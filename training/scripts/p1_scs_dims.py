"""10/17 Per-dimension SCS (S_obs / S_formula / S_derive / S_conclude, Flash judge) of a merged eval
(results/u_p1_<name>/merged.json), re-judged with experiment_right calculate_scs_with_fallback on the stored response.
Writes results/u_p1_<name>/scs_dims.json: question -> {overall, dims, answered}. Usage: python p1_scs_dims.py NAME"""
from srvpaths import E, EVAL_LIB  # machine paths, see srvpaths.py
import asyncio, json, sys
sys.path.insert(0, EVAL_LIB)
from response_parser import parse_response
from metrics_final import calculate_scs_with_fallback



async def main(name):
    d = json.load(open('%s/results/u_p1_%s/merged.json' % (E, name)))
    sem, out = asyncio.Semaphore(8), {}

    async def one(r):
        p = parse_response(r['response'])
        rd = {'prediction': p.prediction, 'reasoning_chain': p.reasoning_chain, 'ground_truth': r['ground_truth'],
              'answer_range': r.get('answer_range'), 'annotated_solution': r.get('annotated_solution', []),
              'question': r['question'], 'question_type': r['question_type'], 'needs_llm_eval': p.needs_llm_eval,
              'raw_response': r['response']}
        if isinstance(rd['annotated_solution'], str) and rd['annotated_solution'].startswith('['):
            try:
                rd['annotated_solution'] = json.loads(rd['annotated_solution'])
            except Exception:
                pass
        async with sem:
            s = await calculate_scs_with_fallback(rd)
        det = s.get('details') or {k: v for k, v in s.items() if k.startswith('S_')}
        out[r['question']] = {'cls': cls, 'overall': s.get('overall'), 'dims': det, 'answered': bool(p.prediction),
                              'stored_SCS': float(r['SCS'])}
    for cls in ('aclass', 'bclass'):
        await asyncio.gather(*(one(r) for r in d[cls]))
    json.dump(out, open('%s/results/u_p1_%s/scs_dims.json' % (E, name), 'w'), indent=1)
    for cls in ('aclass', 'bclass'):
        v = [x for x in out.values() if x['cls'] == cls]
        dims = ['S_obs', 'S_formula', 'S_derive', 'S_conclude']
        print(name, cls, 'n', len(v), 'overall %.3f (stored %.3f)' % (sum(x['overall'] or 0 for x in v) / len(v),
              sum(x['stored_SCS'] for x in v) / len(v)),
              ' '.join('%s %.3f' % (k, sum(float((x['dims'] or {}).get(k, 0)) for x in v) / len(v)) for k in dims))


asyncio.run(main(sys.argv[1]))
