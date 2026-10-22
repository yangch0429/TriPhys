#!/bin/bash
# Provenance (TriPhysBench release): fix_code/evaluation/fix10/fix10_chain5.sh
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
# 10/13 "直接做": v17 (code synthesis + forced substitution) on the 11 pipeline-bug QAs and the redesign of the 37
# no-unique-value / design-flaw QAs run now (in parallel with chain4); the v17 failures get the redesign too; after
# CHAIN4 DONE: audit with the new records, 3 repair rounds, report
cd "$(dirname "$0")"
P=${PYTHON:-python}
H="fix10_full_v13.json fix10_v15.json fix10_v16.json"
FIX10_OUT=fix10_v17.json FIX10_SERPER_MAX=3000 $P -u fix10_v17.py ids11.json > v17.log 2>&1 &
FIX10_SERPER_MAX=6000 $P -u redesign17.py redesign17.json ids37.json $H > redesign17.log 2>&1 &
wait
$P -c "import json;d=json.load(open('fix10_v17.json'));json.dump([k for k in json.load(open('ids11.json')) if (d.get(k) or {}).get('status')!='fixed'],open('ids_v17fail.json','w'))"
FIX10_SERPER_MAX=3000 $P -u redesign17.py redesign17b.json ids_v17fail.json $H fix10_v17.json > redesign17b.log 2>&1
echo "V17+REDESIGN DONE" >> fix10_chain5.log
while ! grep -q "CHAIN4 DONE" fix10_chain4.log; do sleep 60; done
R="fix10_full_v13.json fix10_v15.json given_v15.json fix10_v16.json given_v16.json seed_v16.json fix10_v17.json redesign17.json redesign17b.json"
$P -u audit10.py audit_v16.json $R > audit_v17.log 2>&1
for i in 1 2 3; do
  $P -u audit_fix.py repair_v16.json audit_v16.json $R >> repair_v17.log 2>&1
  $P -u audit10.py audit_v16.json $R repair_v16.json >> audit_v17.log 2>&1
done
$P -u report10.py audit_v16.json fix10_full_v13.json $R repair_v16.json > report10_v17.txt 2>&1
echo "CHAIN5 DONE" >> fix10_chain5.log
