#!/bin/bash
# 10/17 Phase1 A refill (user confirmed "启动A类补采样"). A covers only 65/167 train QAs in rwf_comb2 (B ~124/133).
#  1) uncovered = A train QAs with no sample in rwf_comb2 -> data/p1/a_uncovered.json, data/f10/aclass_qa_train_uncov.json
#  2) old correct samples of those QAs (p1_ss + p1_ss2, mostly failed the Flash gate) -> afill_old, rewrite with 6 tries
#  3) rftrwf_bal_1e4 samples the uncovered QAs 8 more times (T=0.7, eval_n1 native tools) -> results/p1_ssA_r{0..7}
#     8 reps x 8 shards = 64 jobs; slot j runs jobs j, j+16, ... on GPU j%8; slots 8-15 start once both running
#     chains (comb2_1e4, rftrwf_bal_5e5) are DONE/FAILED (training there needs the memory)
#  4) afill_new (p1_ssA, same filter, top-3 SCS) -> rewrite 6 tries; afill_rwf = old + new, <= 3 per QA
#  5) source scripts/p1_afill_train.sh (editable while this runs: which mix / LRs to train)
E=${TRIPHYS_WORK:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/work}; cd $E
PY=${PY:-python}
M=outputs/phase1/rftrwf_bal_1e4/final_model
L=logs/p1_afill.log
step() { echo "$(date "+%F %T") $*" >> $L; }
step "AFILL start model $M"
$PY - <<'EOF' >> $L
import json
E = __import__('os').getcwd()
have = {o['messages'][1]['content'][1]['text'] for o in json.load(open(E + '/data/p1/rwf_comb2.json'))}
items, unc = [], []
for it in json.load(open(E + '/data/f10/aclass_qa_train.json')):
    qs = [qa for qa in it['qa_pairs'] if qa.get('valid', True) and not any(qa['question'] in t for t in have)]
    if qs:
        items.append(dict(it, qa_pairs=qs))
        unc += [qa['question'] for qa in qs]
json.dump(unc, open(E + '/data/p1/a_uncovered.json', 'w'), ensure_ascii=False, indent=1)
json.dump(items, open(E + '/data/f10/aclass_qa_train_uncov.json', 'w'), ensure_ascii=False, indent=1)
print('uncovered A QAs', len(unc), 'videos', len(items))
EOF
$PY scripts/p1_rft_build.py --name afill_old --per_qa 3 --tags p1_ss,p1_ss2 --only data/p1/a_uncovered.json >> $L 2>&1
(RW_TRIES=6 $PY scripts/p1_rewrite.py afill_old afill_old_rwf > logs/p1_afill_rw_old.log 2>&1; step "rewrite old done") &

slot() {
  local j=$1
  if [ $j -ge 8 ]; then
    until grep -qE "DONE|FAILED" logs/p1_chain_comb2_1e4.log && grep -qE "DONE|FAILED" logs/p1_chain_rftrwf_bal_5e5.log; do sleep 120; done
  fi
  for k in $(seq $j 16 63); do
    r=$((k / 8)); i=$((k % 8))
    mkdir -p results/p1_ssA_r$r
    GEN_TEMP=0.7 CUDA_VISIBLE_DEVICES=$((j % 8)) timeout 5h $PY fc/training/n1/eval_n1_sample.py --model_path $M \
      --base_model ${BASE_MODEL:-Qwen/Qwen3.5-4B} --class A --aclass_data data/f10/aclass_qa_train_uncov.json \
      --bclass_data data/f10/bclass_qa_train.json --output_dir results/p1_ssA_r$r --gpu 0 --n_shards 8 --shard_id $i \
      > logs/p1_ssA_r${r}_s$i.log 2>&1
    step "sample r$r s$i exit $?"
  done
}
for j in $(seq 0 15); do slot $j & done
wait
step "SAMPLING DONE $(ls results/p1_ssA_r*/*_shard*.json | wc -l) files"
$PY scripts/p1_rft_build.py --name afill_new --per_qa 3 --tags p1_ssA --only data/p1/a_uncovered.json >> $L 2>&1
RW_TRIES=6 $PY scripts/p1_rewrite.py afill_new afill_new_rwf > logs/p1_afill_rw_new.log 2>&1
$PY - <<'EOF' >> $L
import json
E = __import__('os').getcwd()
by = {}
for n in ('afill_old_rwf', 'afill_new_rwf'):
    for o in json.load(open('%s/data/p1/%s.json' % (E, n))):
        by.setdefault(o['messages'][1]['content'][1]['text'], []).append(o)
sc = lambda o: o['metadata'].get('SCS_rw') if o['metadata'].get('rewritten') else o['metadata']['SCS']
out = [o for v in by.values() for o in sorted(v, key=lambda o: -sc(o))[:3]]
json.dump(out, open(E + '/data/p1/afill_rwf.json', 'w'), ensure_ascii=False, indent=1)
print('afill_rwf: new A QAs', len(by), 'samples', len(out))
EOF
step "AFILL DATA DONE"
source scripts/p1_afill_train.sh
