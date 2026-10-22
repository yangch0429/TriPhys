#!/bin/bash
# 10/20 user: re-evaluate the default 4B DPO p2_full1 once more (same model, same eval_n1, no retraining) to measure
# eval noise. 16 shards, 2 per GPU on GPU0-7. Output results/u_p2n1_p2_full1_r2; ACC rescored locally (OpenRouter).
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
H=$E/fc/training/n1
MERGED=$E/outputs/phase1/comb1_1e4/merged
N=u_p2n1_p2_full1_r2; R=results/$N; L=logs/p2_full1_rerun.log
step() { echo "$(date "+%F %T") $*" >> $L; }
step "eval_n1 16 shards"
mkdir -p $R
for i in $(seq 0 15); do
  CUDA_VISIBLE_DEVICES=$((i % 8)) $PY $H/eval_n1.py --model_path outputs/phase2/p2_full1/final_model --base_model $MERGED \
    --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
    --output_dir $R --gpu 0 --n_shards 16 --shard_id $i > logs/${N}_s$i.log 2>&1 &
done
wait
ln -sfn $E/$R results/det_$N
(cd scripts && $PY -c "import run_det; run_det.merge(\"$N\")") >> $L 2>&1
step "RERUN DONE"
