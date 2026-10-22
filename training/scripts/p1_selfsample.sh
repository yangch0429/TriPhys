#!/bin/bash
# 10/16 base self-sampling on f10 train (300 QAs): untrained base, native tools (eval_n1 = eval_sft_v9 framework),
# T=0.7, k=4 repeats x 16 shards, 2 slots per GPU on GPU0-7 -> results/p1_ss_r{0..3}
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
L=logs/p1_selfsample.log
for r in 0 1 2 3; do for i in $(seq 0 15); do echo "$r $i"; done; done | \
  xargs -P 16 -n 2 sh -c "mkdir -p results/p1_ss_r\$0; GEN_TEMP=0.7 CUDA_VISIBLE_DEVICES=\$((\$1 % 8)) $PY fc/training/n1/eval_n1_sample.py --model_path none --base_model ${BASE_MODEL:-Qwen/Qwen3.5-4B} --class both --aclass_data data/f10/aclass_qa_train.json --bclass_data data/f10/bclass_qa_train.json --output_dir results/p1_ss_r\$0 --gpu 0 --n_shards 16 --shard_id \$1 > logs/p1_ss_r\$0_s\$1.log 2>&1"
echo "$(date "+%F %T") SELFSAMPLE DONE" >> $L
