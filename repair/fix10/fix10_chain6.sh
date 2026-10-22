#!/bin/bash
# Provenance (TriPhysBench release): fix_code/evaluation/fix10/fix10_chain6.sh
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
# 10/13 user: ">=400 QAs must remain". redesign18 (mechanical checks of redesign17 fixed) on the 30 redesign17 failures
# now; after chain5: redesign18 on every QA still unresolved (audit fails after the repairs included, with the audit's
# defects as history); then audit / 3 repair rounds / report; one more redesign pass if > 45 are unresolved
cd "$(dirname "$0")"
P=${PYTHON:-python}
H="fix10_full_v13.json fix10_v15.json fix10_v16.json fix10_v17.json redesign17.json redesign17b.json"
$P -c "import json;d=json.load(open('redesign17.json'));json.dump([k for k,r in d.items() if r['status']!='fixed'],open('ids_r18a.json','w'))"
FIX10_SERPER_MAX=6000 $P -u redesign18.py redesign18.json ids_r18a.json $H > redesign18.log 2>&1
echo "R18A DONE $(date)" >> fix10_chain6.log
while ! grep -q "CHAIN5 DONE" fix10_chain5.log; do sleep 60; done
R="fix10_full_v13.json fix10_v15.json given_v15.json fix10_v16.json given_v16.json seed_v16.json fix10_v17.json redesign17.json redesign17b.json repair_v16.json redesign18.json"
$P -c "
import json
u=json.load(open('unresolved_final.json')); a=json.load(open('ids_r18a.json')); d=json.load(open('redesign18.json'))
json.dump([k for k in u if k not in a and (d.get(k) or {}).get('status')!='fixed'],open('ids_r18b.json','w'))"
FIX10_SERPER_MAX=6000 $P -u redesign18.py redesign18b.json ids_r18b.json $H audit_v16.json > redesign18b.log 2>&1
R="$R redesign18b.json"
$P merge_raudit.py audit_v16.json redesign18.json redesign18b.json >> fix10_chain6.log
$P -u audit10.py audit_v16.json $R > audit_v18.log 2>&1
for i in 1 2 3; do
  $P -u audit_fix.py repair_v18.json audit_v16.json $R >> repair_v18.log 2>&1
  $P -u audit10.py audit_v16.json $R repair_v18.json >> audit_v18.log 2>&1
done
$P -u report10.py audit_v16.json fix10_full_v13.json $R repair_v18.json > report10_v18.txt 2>&1
echo "PASS1 DONE $(date)" >> fix10_chain6.log
N=$($P -c "import json;print(len(json.load(open('unresolved_final.json'))))")
if [ "$N" -gt 45 ]; then
  cp unresolved_final.json ids_r18c.json
  FIX10_SERPER_MAX=6000 REDESIGN_N=8 $P -u redesign18.py redesign18c.json ids_r18c.json $H redesign18.json redesign18b.json audit_v16.json > redesign18c.log 2>&1
  R="$R repair_v18.json redesign18c.json"
  $P merge_raudit.py audit_v16.json redesign18c.json >> fix10_chain6.log
  for i in 1 2; do
    $P -u audit_fix.py repair_v18c.json audit_v16.json $R >> repair_v18.log 2>&1
    $P -u audit10.py audit_v16.json $R repair_v18c.json >> audit_v18.log 2>&1
  done
  $P -u report10.py audit_v16.json fix10_full_v13.json $R repair_v18c.json > report10_v18.txt 2>&1
fi
echo "CHAIN6 DONE $(date)" >> fix10_chain6.log
