#!/bin/bash
# TriPhys, Stage I (SFT, comb1_1e4) and Stage II (DPO, p2_full1) for one backbone: all commands, in order.
# 8 GPUs (80 GB). Run training/setup_work.sh first, with one work directory ($TRIPHYS_WORK) per backbone.
# env: EGOEXO_ROOT, BAAI_ROOT, SERPER_API_KEYS, OPENROUTER_API_KEY (Gemini rewriter / teacher, gpt-4o-mini ACC judge),
#      DEEPSEEK_API_KEY (SCS judge); optional PY, TR and
#   BASE_MODEL  Qwen/Qwen3.5-4B (default), Qwen/Qwen3.5-9B, Qwen/Qwen3.6-27B, or a local directory
#   MP          1: model-parallel training (train_sft_n1_mp.py, train_dpo_p2_mp.py); default 1 for 27B, else 0
#   SLOTS       sampling processes per GPU; default 1 for 27B, else 2
#   PREBUILT    directory with an existing Stage-I set (rwf_comb1.json) and pair set (dpo_p2_full1.json): steps 1-7 and
#               9-11 are skipped and both stages are trained on these files
set -e
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(readlink -f "$0")")")/work}; cd $E
export PY=${PY:-python} TR=${TR:-torchrun}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-4B}
case $BASE_MODEL in *27B*|*27b*) MP=${MP:-1}; SLOTS=${SLOTS:-1};; esac
export MP=${MP:-0} SLOTS=${SLOTS:-2}

if [ -z "$PREBUILT" ]; then
# ---------------- Stage I ----------------
# 1. base self-sampling on the 300 train QAs: T 0.7, 4 repeats x 16 shards -> results/p1_ss_r{0..3}
bash scripts/p1_selfsample.sh
# 2. GT-verified samples (ACC >= .9, answered, no leak), <= 3 per QA, class/type balanced -> data/p1/rft_k3_bal.json
$PY scripts/p1_rft_build.py --name rft_k3 --per_qa 3 --min_scs 0.0
$PY scripts/p1_rft_balance.py --input data/p1/rft_k3.json --output data/p1/rft_k3_bal.json
# 3. round-2 sampler: SFT on rft_k3_bal, then it samples the train QAs again -> results/p1_ss2_r{0..3}
LR=1e-4 bash scripts/p1_rft_chain.sh rft_k3bal_1e4 rft_k3_bal 0,1,2,3 29610
bash scripts/p1_selfsample2.sh
# 4. Gemini teacher (never sees the GT, real tools), 4 samples per QA; correct trajectories of QAs no self-sample
#    solved -> gem_only
$PY fc/training/teacher_pilot/teacher_full.py data/teacher_full/gemini_k4.json 4
$PY scripts/p1_gemini_conv.py --name gem_only
# 5. ACC re-score of self-sample rows below .9 (judge errors during sampling)
$PY scripts/p1_rescore_samples.py p1_ss p1_ss2
# 6. answer-block rewrite (Gemini; kept only if the SCS judge scores it >= max(.75, own SCS), 3 tries)
$PY scripts/p1_rewrite.py rft_k3_bal rft_k3_bal_rwf
$PY scripts/p1_rewrite.py gem_only gem_only_rwf
LR=1e-4 bash scripts/p1_rft_chain.sh rftrwf_bal_1e4 rft_k3_bal_rwf 0,1,2,3 29612
# 7. commit continuations: own stuck prefix (2 tool rounds) + one-off note -> the intermediate SFT model commits to an
#    answer
mkdir -p results/p1_commit
for i in $(seq 0 7); do
  GEN_TEMP=0.7 CUDA_VISIBLE_DEVICES=$i $PY scripts/p1_commit.py --model outputs/phase1/rftrwf_bal_1e4/final_model \
    --shard $i --n_shards 8 --out results/p1_commit/shard_$i.json > logs/p1_commit_s$i.log 2>&1 &
done
wait
$PY scripts/p1_commit_build.py --name commit_k2
$PY scripts/p1_rewrite.py commit_k2 commit_k2_rwf
# 8a. Stage-I data
$PY - <<'PYEOF'
import json
d = []
for n in ["rft_k3_bal_rwf", "gem_only_rwf", "commit_k2_rwf"]:
    d += json.load(open("data/p1/%s.json" % n))
json.dump(d, open("data/p1/rwf_comb1.json", "w"), ensure_ascii=False, indent=1)
print('rwf_comb1', len(d))
PYEOF
else
  cp $PREBUILT/rwf_comb1.json data/p1/rwf_comb1.json
  cp $PREBUILT/dpo_p2_full1.json data/p2/dpo_p2_full1.json
fi
# 8b. Stage-I model comb1_1e4 (3 ep, lr 1e-4, effective batch 16), evaluated on the 110 test QAs
LR=1e-4 bash scripts/p1_rft_chain.sh comb1_1e4 rwf_comb1 0,1,2,3 29616

# ---------------- Stage II ----------------
if [ -z "$PREBUILT" ]; then
# 9. MPR refill pool data/p1/afill_rwf.json (Stage-II fallback chosen)
bash fc/training/p1_afill_data.sh
# 10. comb1_1e4 rollouts (p2_online: prefix branching, K 4, per_qa 2, T 0.7), workers on a shared QA queue
for w in $(seq 0 $((8 * SLOTS - 1))); do bash scripts/p2_worker.sh $w $((w % 8)) & done
wait
# 11. full-trajectory pairs, <= 2 per QA -> data/p2/dpo_p2_full1.json
$PY scripts/p2_pairs_full.py --inp "results/p2_on/w_*[0-9].json" --out data/p2/dpo_p2_full1.json
fi
# 12. DPO from comb1_1e4 (merged = init and frozen reference), effective batch 16 pairs
S1=outputs/phase1/comb1_1e4
DPO_ARGS="--base_model $BASE_MODEL --sft_ckpt $S1/final_model --merged_dir $S1/merged --data data/p2/dpo_p2_full1.json
  --output outputs/phase2/p2_full1 --max_length 16384 --epochs 2 --lr 5e-6 --beta 0.1 --sft_alpha 0
  --mask_common_prefix --length_norm --class_balance"
if [ $MP = 1 ]; then
  # 2 ranks, each holding one replica split over 4 GPUs; 2 x accum 8
  GPUS_PER_RANK=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $TR --nproc_per_node=2 --master_port=29631 scripts/train_dpo_p2_mp.py $DPO_ARGS --grad_accum 8 \
    > logs/train_p2_full1.log 2>&1
else
  # 8 GPUs x accum 2
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $TR --nproc_per_node=8 --master_port=29631 scripts/train_dpo_p2.py $DPO_ARGS --grad_accum 2 \
    > logs/train_p2_full1.log 2>&1
fi
# 13. evaluation of the DPO model on the 110 test QAs (eval_n1, 8 shards, merge)
bash scripts/p2_reeval_n1.sh p2_full1
