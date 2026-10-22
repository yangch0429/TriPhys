# 10/17 sourced by p1_afill.sh after the A refill data is built (edit before then to change the mix / LRs).
# 10/18: first re-score p1_ssA ACC (gpt-4o-mini judge returned 400/'choices' during sampling), rebuild afill_new,
# rewrite again reusing the earlier rewrites (--prev), and re-merge afill_rwf.
$PY scripts/p1_rescore_samples.py p1_ssA > logs/p1_afill_rescore.log 2>&1
step "rescore p1_ssA done: $(grep -c . logs/p1_afill_rescore.log) log lines"
cp data/p1/afill_new_rwf.json data/p1/afill_new_rwf_pre.json
$PY scripts/p1_rft_build.py --name afill_new --per_qa 3 --tags p1_ssA --only data/p1/a_uncovered.json >> $L 2>&1
RW_TRIES=6 $PY scripts/p1_rewrite.py afill_new afill_new_rwf --prev afill_new_rwf_pre > logs/p1_afill_rw_new2.log 2>&1
$PY - <<'EOF2' >> $L
import json
E = __import__('os').getcwd()
by = {}
for n in ('afill_old_rwf', 'afill_new_rwf'):
    for o in json.load(open('%s/data/p1/%s.json' % (E, n))):
        by.setdefault(o['messages'][1]['content'][1]['text'], []).append(o)
sc = lambda o: o['metadata'].get('SCS_rw') if o['metadata'].get('rewritten') else o['metadata']['SCS']
out = [o for v in by.values() for o in sorted(v, key=lambda o: -sc(o))[:3]]
json.dump(out, open(E + '/data/p1/afill_rwf.json', 'w'), ensure_ascii=False, indent=1)
print('afill_rwf (after rescore): new A QAs', len(by), 'samples', len(out))
EOF2
# Default: comb3 = rwf_comb2 + afill_rwf, trained at 1e-4 (GPU0-3) and 5e-5 (GPU4-7) - LR-insensitivity check.
$PY - <<'EOF' >> $L
import json
E = __import__('os').getcwd()
out = json.load(open(E + '/data/p1/rwf_comb2.json')) + json.load(open(E + '/data/p1/afill_rwf.json'))
json.dump(out, open(E + '/data/p1/rwf_comb3.json', 'w'), ensure_ascii=False, indent=1)
print('rwf_comb3', len(out), 'A', sum(o['metadata']['cls'] == 'A' for o in out), 'B', sum(o['metadata']['cls'] == 'B' for o in out))
EOF
LR=1e-4 bash scripts/p1_rft_chain.sh comb3_1e4 rwf_comb3 0,1,2,3 29620 &
LR=5e-5 bash scripts/p1_rft_chain.sh comb3_5e5 rwf_comb3 4,5,6,7 29621 &
wait
step "AFILL TRAIN DONE"
