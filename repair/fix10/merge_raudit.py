# Provenance (TriPhysBench release): fix_code/evaluation/fix10/merge_raudit.py
# Role: fix10 round (10-11 .. 10-13) that produced version 08 (410 QAs / 200 videos).
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""put the strict audit that a redesign already passed into the audit file (same sig: audit10 does not repeat it)
Usage: python merge_raudit.py audit.json redesign1.json [...]"""
import sys, json
aud = json.load(open(sys.argv[1]))
n = 0
for f in sys.argv[2:]:
    for k, r in json.load(open(f)).items():
        if r.get('status') == 'fixed' and r.get('audit') and r.get('sig'):
            aud[k] = dict(r['audit'], q=r['qa']['question'], a=r['qa']['answer'], status='fixed', sig=r['sig'])
            n += 1
json.dump(aud, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('merged', n)
