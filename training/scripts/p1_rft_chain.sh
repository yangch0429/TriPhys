#!/bin/bash
# Stage-I SFT on a trajectory set data/p1/DATA.json (built by p1_rft_build.py / p1_rewrite.py), then evaluation.
# Native tool format in training and evaluation: train_sft_n1.py (tools=n1_tools.TOOLS in the template), LoRA r32,
# 4 GPUs x accum 4, checkpoint per epoch; then eval_n1 (native tools) on the test split (data/f10/*_test.json),
# 8 shards, merge, ACC re-score.
# Usage: p1_rft_chain.sh NAME DATA GPUS PORT   (env EPOCHS, LR; defaults 3 / 2e-4)
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}
PY=${PY:-python}
TR=${TR:-torchrun}
H=$E/fc/training/n1
NAME=$1; DATA=$2; GPUS=$3; PORT=$4
G=(${GPUS//,/ })
OUT=$E/outputs/phase1/$NAME
R=results/u_p1_$NAME
L=$E/logs/p1_chain_$NAME.log
cd $E
step() { echo "$(date "+%F %T") $*" >> $L; }
# optional pause: training waits while the file logs/HOLD_COMB3 exists
if [ -f logs/HOLD_COMB3 ]; then step "paused before training (logs/HOLD_COMB3 exists)"; while [ -f logs/HOLD_COMB3 ]; do sleep 120; done; step "resumed"; fi
N=$($PY -c "import json;print(len(json.load(open(\"data/p1/$DATA.json\"))))")
SPE=$(( (N + 15) / 16 ))
step "train start: $N samples, $SPE steps/epoch, epochs ${EPOCHS:-3}, lr ${LR:-2e-4}, GPUs $GPUS"
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=${#G[@]} --master_port=$PORT \
  $H/train_sft_n1.py --data data/p1/$DATA.json --output $OUT --base_model ${BASE_MODEL:-Qwen/Qwen3.5-4B} \
  --epochs ${EPOCHS:-3} --lr ${LR:-2e-4} --lora_r 32 --max_length 16384 --grad_accum 4 --save_steps $SPE \
  > logs/train_p1_$NAME.log 2>&1
[ -f $OUT/final_model/adapter_model.safetensors ] || { step "TRAIN FAILED"; exit 1; }
step "train done; eval on GPUs $GPUS"
mkdir -p $R
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${G[$((i % 4))]} $PY $H/eval_n1.py --model_path $OUT/final_model --base_model ${BASE_MODEL:-Qwen/Qwen3.5-4B} \
    --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
    --output_dir $R --gpu 0 --n_shards 8 --shard_id $i > logs/u_p1_${NAME}_s$i.log 2>&1 &
done
wait
ln -sfn $E/$R results/det_u_p1_$NAME
(cd scripts && $PY -c "import run_det; run_det.merge(\"u_p1_$NAME\")") >> $L 2>&1
$PY scripts/p1_rescore_acc.py $NAME >> $L 2>&1
step "P1 $NAME DONE"
