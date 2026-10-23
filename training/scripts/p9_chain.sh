#!/bin/bash
# Qwen3.5-9B: Stage I + Stage II with the 4B data and hyperparameters.
#  1. SFT: data/p1/rwf_comb1.json (209 = comb1_1e4 data), train_sft_n1 LoRA r32, 3 ep, lr 1e-4, 8 GPUs x accum 2
#     (effective batch 16 = 4B's 4 GPUs x accum 4)
#  2. DPO: data/p2/dpo_p2_full1.json (143 4B pairs), train_dpo_p2 init + frozen ref = 9B SFT (merged), same flags as p2_full1
#  3. eval_n1 on the test split for both (8 shards each, 2 processes per GPU), run_det merge. ACC re-scoring with the
#     gpt-4o-mini judge (p1_rescore_acc.py) is run separately.
# Large outputs under $D; outputs/phase{1,2}/*_9b are symlinks.
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
TR=${TR:-torchrun}
H=$E/fc/training/n1
B9=${B9:-Qwen/Qwen3.5-9B}
D=${D9:-$E/store_9b}
SFT=$D/phase1/comb1_9b; MERGED=$SFT/merged; DPO=$D/phase2/p2_full1_9b
L=logs/p9_chain.log
step() { echo "$(date "+%F %T") $*" >> $L; }
mkdir -p $D/phase1 $D/phase2
ln -sfn $SFT outputs/phase1/comb1_9b; ln -sfn $DPO outputs/phase2/p2_full1_9b
step "9B SFT train start (209 samples, 8 GPUs x accum 2, 3 ep, lr 1e-4)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=8 --master_port=29651 \
  $H/train_sft_n1.py --data data/p1/rwf_comb1.json --output $SFT --base_model $B9 \
  --epochs 3 --lr 1e-4 --lora_r 32 --max_length 16384 --grad_accum 2 --save_steps 14 > logs/train_p1_comb1_9b.log 2>&1
[ -f $SFT/final_model/adapter_model.safetensors ] || { step "SFT TRAIN FAILED"; exit 1; }
step "9B SFT done; DPO train start (143 pairs, 8 GPUs)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TR --nproc_per_node=8 --master_port=29652 \
  scripts/train_dpo_p2.py --base_model $B9 --sft_ckpt $SFT/final_model --merged_dir $MERGED --data data/p2/dpo_p2_full1.json \
  --output $DPO --max_length 16384 --epochs 2 --lr 5e-6 --beta 0.1 --sft_alpha 0 --grad_accum 2 \
  --mask_common_prefix --length_norm --class_balance > logs/train_p2_full1_9b.log 2>&1
if [ -f $DPO/final_model/adapter_model.safetensors ]; then step "DPO done; evals"; DOK=1; else step "DPO TRAIN FAILED; SFT eval only"; DOK=0; fi
mkdir -p results/u_p1_comb1_9b results/u_p2n1_p2_full1_9b
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i $PY $H/eval_n1.py --model_path $SFT/final_model --base_model $B9 \
    --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
    --output_dir results/u_p1_comb1_9b --gpu 0 --n_shards 8 --shard_id $i > logs/u_p1_comb1_9b_s$i.log 2>&1 &
  [ $DOK = 1 ] && CUDA_VISIBLE_DEVICES=$i $PY $H/eval_n1.py --model_path $DPO/final_model --base_model $MERGED \
    --class both --aclass_data data/f10/aclass_qa_test.json --bclass_data data/f10/bclass_qa_test.json \
    --output_dir results/u_p2n1_p2_full1_9b --gpu 0 --n_shards 8 --shard_id $i > logs/u_p2n1_p2_full1_9b_s$i.log 2>&1 &
done
wait
for n in u_p1_comb1_9b u_p2n1_p2_full1_9b; do
  ln -sfn $E/results/$n results/det_$n
  (cd scripts && $PY -c "import run_det; run_det.merge(\"$n\")") >> $L 2>&1
done
step "P9 DONE"
