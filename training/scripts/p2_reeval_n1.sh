#!/bin/bash
# Stage-II evaluation: the DPO adapter outputs/phase2/NAME/final_model on top of the merged Stage-I model, evaluated
# with eval_n1 (native tools, same harness as Stage I) on the test split, 8 shards (1 per GPU), then merge.
# Usage: p2_reeval_n1.sh [NAME]   (default p2_full1)
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
H=$E/fc/training/n1
NAME=${1:-p2_full1}
OUT=outputs/phase2/$NAME
MERGED=$E/outputs/phase1/comb1_1e4/merged
R=results/u_p2n1_$NAME
L=logs/p2_reeval_$NAME.log
step() { echo "$(date "+%F %T") $*" >> $L; }
step "eval_n1 8 shards"
mkdir -p $R
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i $PY $H/eval_n1.py --model_path $OUT/final_model --base_model $MERGED \
    --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
    --output_dir $R --gpu 0 --n_shards 8 --shard_id $i > logs/u_p2n1_${NAME}_s$i.log 2>&1 &
done
wait
ln -sfn $E/$R results/det_u_p2n1_$NAME
(cd scripts && $PY -c "import run_det; run_det.merge(\"u_p2n1_$NAME\")") >> $L 2>&1
step "P2N1 $NAME DONE"
