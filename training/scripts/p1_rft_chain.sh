#!/bin/bash
# Stage-I SFT on a trajectory set data/p1/DATA.json (built by p1_rft_build.py / p1_rewrite.py), then evaluation.
# Native tool format in training and evaluation: train_sft_n1.py (tools=n1_tools.TOOLS in the template), LoRA r32,
# effective batch 16 (GPUs x accum), checkpoint per epoch; then eval_n1 (native tools) on the test split
# (data/f10/*_test.json), 8 shards (1 per GPU), merge, ACC re-score.
# MP=1: train_sft_n1_mp.py, one process with the model split over GPUS, accum 16; retries resume from the last checkpoint.
# Usage: p1_rft_chain.sh NAME DATA GPUS PORT   (env BASE_MODEL, MP, EPOCHS, LR; defaults Qwen/Qwen3.5-4B / 0 / 3 / 1e-4)
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}
PY=${PY:-python}
TR=${TR:-torchrun}
H=$E/fc/training/n1
BM=${BASE_MODEL:-Qwen/Qwen3.5-4B}
NAME=$1; DATA=$2; GPUS=$3; PORT=$4
G=(${GPUS//,/ })
OUT=$E/outputs/phase1/$NAME
R=results/u_p1_$NAME
L=$E/logs/p1_chain_$NAME.log
cd $E
step() { echo "$(date "+%F %T") $*" >> $L; }
N=$($PY -c "import json;print(len(json.load(open(\"data/p1/$DATA.json\"))))")
SPE=$(( (N + 15) / 16 ))
ARGS="--data data/p1/$DATA.json --output $OUT --base_model $BM --epochs ${EPOCHS:-3} --lr ${LR:-1e-4} --lora_r 32
  --max_length 16384 --save_steps $SPE"
step "train start: $N samples, $SPE steps/epoch, epochs ${EPOCHS:-3}, lr ${LR:-1e-4}, GPUs $GPUS, MP ${MP:-0}"
if [ "${MP:-0}" = 1 ]; then
  for try in 1 2 3; do
    [ -f $OUT/final_model/adapter_model.safetensors ] && break
    RES=""; ls -d $OUT/checkpoint-* >/dev/null 2>&1 && RES="--resume"
    CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $H/train_sft_n1_mp.py $ARGS \
      --grad_accum 16 $RES >> logs/train_p1_$NAME.log 2>&1
  done
else
  CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=${#G[@]} \
    --master_port=$PORT $H/train_sft_n1.py $ARGS --grad_accum $((16 / ${#G[@]})) > logs/train_p1_$NAME.log 2>&1
fi
[ -f $OUT/final_model/adapter_model.safetensors ] || { step "TRAIN FAILED"; exit 1; }
step "train done; eval"
mkdir -p $R
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i $PY $H/eval_n1.py --model_path $OUT/final_model --base_model $BM \
    --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
    --output_dir $R --gpu 0 --n_shards 8 --shard_id $i > logs/u_p1_${NAME}_s$i.log 2>&1 &
done
wait
ln -sfn $E/$R results/det_u_p1_$NAME
(cd scripts && $PY -c "import run_det; run_det.merge(\"u_p1_$NAME\")") >> $L 2>&1
$PY scripts/p1_rescore_acc.py $NAME >> $L 2>&1
step "P1 $NAME DONE"
