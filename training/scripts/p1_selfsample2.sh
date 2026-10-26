#!/bin/bash
# Stage-I round-2 self-sampling: rft_k3bal_1e4 (LoRA SFT on the balanced round-1 samples) on the train split (300 QAs),
# native tools (eval_n1_sample), T=0.7, k=4 repeats x 16 shards; SLOTS (default 2) fixed slots per GPU on GPU0-7
# -> results/p1_ss2_r{0..3}. Each shard is capped at 4 h (a shard can hang in a long tool-call loop).
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
M=${M:-outputs/phase1/rft_k3bal_1e4/final_model}
NS=$((8 * ${SLOTS:-2}))
L=logs/p1_selfsample2.log
echo "$(date "+%F %T") SELFSAMPLE2 start model $M" >> $L
slot() {
  for k in $(seq $1 $NS 63); do
    r=$((k / 16)); i=$((k % 16))
    mkdir -p results/p1_ss2_r$r
    GEN_TEMP=0.7 CUDA_VISIBLE_DEVICES=$(($1 % 8)) timeout 4h $PY fc/training/n1/eval_n1_sample.py --model_path $M \
      --base_model ${BASE_MODEL:-Qwen/Qwen3.5-4B} --class both --aclass_data data/f10/aclass_qa_train.json \
      --bclass_data data/f10/bclass_qa_train.json --output_dir results/p1_ss2_r$r --gpu 0 --n_shards 16 --shard_id $i \
      > logs/p1_ss2_r${r}_s$i.log 2>&1
  done
}
for j in $(seq 0 $((NS - 1))); do slot $j & done
wait
echo "$(date "+%F %T") SELFSAMPLE2 DONE $(ls results/p1_ss2_r*/*.json | wc -l) files" >> $L
