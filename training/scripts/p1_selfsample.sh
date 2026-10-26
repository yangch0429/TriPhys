#!/bin/bash
# Stage-I base self-sampling on the train split (300 QAs): untrained base model, native tools (eval_n1_sample),
# T=0.7, k=4 repeats x 16 shards; SLOTS (default 2) fixed slots per GPU on GPU0-7 -> results/p1_ss_r{0..3}
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
NS=$((8 * ${SLOTS:-2}))
L=logs/p1_selfsample.log
slot() {
  for k in $(seq $1 $NS 63); do
    r=$((k / 16)); i=$((k % 16))
    mkdir -p results/p1_ss_r$r
    GEN_TEMP=0.7 CUDA_VISIBLE_DEVICES=$(($1 % 8)) $PY fc/training/n1/eval_n1_sample.py --model_path none \
      --base_model ${BASE_MODEL:-Qwen/Qwen3.5-4B} --class both --aclass_data data/f10/aclass_qa_train.json \
      --bclass_data data/f10/bclass_qa_train.json --output_dir results/p1_ss_r$r --gpu 0 --n_shards 16 --shard_id $i \
      > logs/p1_ss_r${r}_s$i.log 2>&1
  done
}
for j in $(seq 0 $((NS - 1))); do slot $j & done
wait
echo "$(date "+%F %T") SELFSAMPLE DONE" >> $L
