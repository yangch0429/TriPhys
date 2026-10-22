#!/bin/bash
# 10/18 one p2_online.py worker on the shared QA queue (results/p2_on/claims); resumes its own claims, retries twice.
# Usage: p2_worker.sh ID GPU
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
mkdir -p results/p2_on/claims
for t in 1 2 3; do
  GEN_TEMP=0.7 CUDA_VISIBLE_DEVICES=$2 $PY scripts/p2_online.py --model outputs/phase1/comb1_1e4/final_model \
    --out results/p2_on/w_$1.json --claim_dir results/p2_on/claims --K 4 --per_qa 2 --attempts 4 >> logs/p2_on_w$1.log 2>&1
  rc=$?
  echo "$(date "+%F %T") worker $1 gpu $2 try $t exit $rc" >> logs/p2_online.log
  [ $rc -eq 0 ] && break
  sleep 30
done
