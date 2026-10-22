#!/bin/bash
# Provenance (TriPhysBench release): fix_code/evaluation/fix10/fix10_chain4.sh
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
# user 10/13: v16 on the 57 -> stated-given -> gpt-5.6 audit (wrist convention, type-form checks, seeds) ->
# 3 x (audit-driven repair + re-audit) -> type check -> report
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export FIX10_REVIEWER=gpt-5.6-terra
$PY make_seeds16.py
echo "v16 $(date)"
FIX10_OUT=fix10_v16.json FIX10_SERPER_MAX=6000 $PY -u fix10_v16.py unres57_ids.json > full_v16.log 2>&1
echo "given $(date)"
$PY -u given_convert.py given_v16.json fix10_full_v13.json fix10_v15.json given_v15.json fix10_v16.json > given_v16.log 2>&1
R="fix10_full_v13.json fix10_v15.json given_v15.json fix10_v16.json given_v16.json seed_v16.json"
echo "audit $(date)"
$PY -u audit10.py audit_v16.json $R > audit_v16.log 2>&1
for i in 1 2 3; do
  echo "repair round $i $(date)"
  $PY -u audit_fix.py repair_v16.json audit_v16.json $R >> repair_v16.log 2>&1
  $PY -u audit10.py audit_v16.json $R repair_v16.json >> audit_v16.log 2>&1
done
$PY -u report10.py audit_v16.json fix10_full_v13.json $R repair_v16.json > report10.txt 2>&1
echo "CHAIN4 DONE $(date)"
