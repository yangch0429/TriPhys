#!/bin/bash
# Qwen3.6-27B (dense, 52 GB bf16): Stage I + Stage II with the 4B data and hyperparameters (as p9_chain.sh).
#  1. SFT: data/p1/rwf_comb1.json (209), LoRA r32, 3 ep, lr 1e-4, effective batch 16 (= 4B comb1_1e4).
#     A full 27B replica per GPU (DDP) runs out of memory at 16k tokens, so train_sft_n1_mp.py runs one process with
#     the model split over GPUs 0-3 (device_map auto), accum 16. Retries resume from the last checkpoint.
#  2. DPO: data/p2/dpo_p2_full1.json (143 4B pairs), same flags as p2_full1; train_dpo_p2_mp.py: 2 ranks x 4 GPUs
#     (one split replica per rank), accum 8 (ranks x accum = 16 as the 4B / 9B runs).
#  3. eval_n1 on the test split, 8 shards, one 27B process per GPU (SFT shard then DPO shard), run_det merge.
# Outputs under $D; outputs/phase{1,2}/*_27b are symlinks.
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
TR=${TR:-torchrun}
H=$E/fc/training/n1
B27=${B27:-Qwen/Qwen3.6-27B}
D=${D27:-$E/store_27b}
SFT=$D/phase1/comb1_27b; MERGED=$SFT/merged; DPO=$D/phase2/p2_full1_27b
L=logs/p27_chain.log
step() { echo "$(date "+%F %T") $*" >> $L; }
mkdir -p $D/phase1 $D/phase2
ln -sfn $SFT outputs/phase1/comb1_27b; ln -sfn $DPO outputs/phase2/p2_full1_27b
for try in 1 2 3; do
  [ -f $SFT/final_model/adapter_model.safetensors ] && break
  R=""; ls -d $SFT/checkpoint-* >/dev/null 2>&1 && R="--resume"
  step "27B SFT (MP) train try $try $R (209 samples, GPUs 0-3 split, accum 16, 3 ep, lr 1e-4)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
    $H/train_sft_n1_mp.py --data data/p1/rwf_comb1.json --output $SFT --base_model $B27 \
    --epochs 3 --lr 1e-4 --lora_r 32 --max_length 16384 --grad_accum 16 --save_steps 3 $R >> logs/train_p1_comb1_27b.log 2>&1
done
[ -f $SFT/final_model/adapter_model.safetensors ] || { step "SFT TRAIN FAILED"; exit 1; }
step "27B SFT done; DPO (MP) train start (143 pairs, 2 ranks x 4 GPUs, accum 8)"
GPUS_PER_RANK=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=2 --master_port=29662 \
  scripts/train_dpo_p2_mp.py --base_model $B27 --sft_ckpt $SFT/final_model --merged_dir $MERGED --data data/p2/dpo_p2_full1.json \
  --output $DPO --max_length 16384 --epochs 2 --lr 5e-6 --beta 0.1 --sft_alpha 0 --grad_accum 8 \
  --mask_common_prefix --length_norm --class_balance > logs/train_p2_full1_27b.log 2>&1
if [ -f $DPO/final_model/adapter_model.safetensors ]; then step "DPO done; evals"; DOK=1; else step "DPO TRAIN FAILED; SFT eval only"; DOK=0; fi
mkdir -p results/u_p1_comb1_27b results/u_p2n1_p2_full1_27b
for i in $(seq 0 7); do
  ( CUDA_VISIBLE_DEVICES=$i $PY $H/eval_n1.py --model_path $SFT/final_model --base_model $B27 \
      --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
      --output_dir results/u_p1_comb1_27b --gpu 0 --n_shards 8 --shard_id $i > logs/u_p1_comb1_27b_s$i.log 2>&1
    [ $DOK = 1 ] && CUDA_VISIBLE_DEVICES=$i $PY $H/eval_n1.py --model_path $DPO/final_model --base_model $MERGED \
      --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
      --output_dir results/u_p2n1_p2_full1_27b --gpu 0 --n_shards 8 --shard_id $i > logs/u_p2n1_p2_full1_27b_s$i.log 2>&1 ) &
done
wait
for n in u_p1_comb1_27b u_p2n1_p2_full1_27b; do
  ln -sfn $E/results/$n results/det_$n
  (cd scripts && $PY -c "import run_det; run_det.merge(\"$n\")") >> $L 2>&1
done
step "P27 DONE"
