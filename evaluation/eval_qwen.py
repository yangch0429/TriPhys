"""Evaluate Qwen3.5-4B / Qwen3.5-9B / Qwen3.6-27B and the TriPhys models (base weights + LoRA adapter).

  python evaluation/eval_qwen.py --base_model Qwen/Qwen3.5-4B --out results/qwen35_4b.jsonl
  python evaluation/eval_qwen.py --base_model Qwen/Qwen3.5-4B --adapter outputs/stage1_4b/final_model --out ...
  # Stage-II model: base = merged Stage-I model, adapter = DPO adapter
  python evaluation/eval_qwen.py --base_model outputs/stage2_4b/merged_sft --adapter outputs/stage2_4b/final_model ...
Shard over GPUs with --shard_id i --n_shards n (one process per GPU, CUDA_VISIBLE_DEVICES=i); merge with
evaluation/aggregate.py. Use device_map auto (--device_map auto) when the 27B model does not fit one GPU."""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triphys import data, evalrun
from triphys.qwen_agent import QwenAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', required=True)
    ap.add_argument('--adapter', default='none', help='LoRA adapter directory, or none for the untrained backbone')
    ap.add_argument('--subtask', default='both', choices=['MPR', 'KPR', 'both'])
    ap.add_argument('--split', default='test', choices=['test', 'train', 'all'])
    ap.add_argument('--out', required=True, help='output .jsonl (resumable)')
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--device_map', default=None)
    ap.add_argument('--skip_score', action='store_true', help='only generate; score later with evaluation/rescore.py')
    ap.add_argument('--temperature', type=float, default=None,
                    help='sampling temperature (Stage-II rollouts: 0.7; one output file per rollout); default: '
                         'evaluation decoding')
    a = ap.parse_args()

    items = evalrun.shard(data.load(data.subtasks_arg(a.subtask), a.split), a.shard_id, a.n_shards)
    if a.limit:
        items = items[:a.limit]
    agent = QwenAgent(a.base_model, a.adapter, a.device, temperature=a.temperature, device_map=a.device_map)
    name = os.path.basename(a.base_model.rstrip('/')) + ('' if a.adapter == 'none' else '+' + a.adapter)
    evalrun.run(items, lambda it: agent.solve(it['qa'], it['video'], it['qid']), a.out,
                thinking_opened=True, score=not a.skip_score, model_name=name)


if __name__ == '__main__':
    main()
