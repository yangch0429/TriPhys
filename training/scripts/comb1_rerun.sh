#!/bin/bash
# 10/20 user: re-evaluate 4B comb1_1e4 (stage-1 SFT) once more, same eval_n1 as u_p1_comb1_1e4, no retraining, to
# measure eval noise. 8 shards, 1 per GPU, alongside p2_full1_rerun.sh. Output results/u_p1_comb1_1e4_r2.
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
H=$E/fc/training/n1
N=u_p1_comb1_1e4_r2; R=results/$N; L=logs/comb1_rerun.log
step() { echo "$(date "+%F %T") $*" >> $L; }
step "eval_n1 8 shards"
mkdir -p $R
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i $PY $H/eval_n1.py --model_path outputs/phase1/comb1_1e4/final_model --base_model ${BASE_MODEL:-Qwen/Qwen3.5-4B} \
    --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
    --output_dir $R --gpu 0 --n_shards 8 --shard_id $i > logs/${N}_s$i.log 2>&1 &
done
wait
ln -sfn $E/$R results/det_$N
(cd scripts && $PY -c "import run_det; run_det.merge(\"$N\")") >> $L 2>&1
step "RERUN DONE"
