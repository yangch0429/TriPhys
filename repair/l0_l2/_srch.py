# Provenance (TriPhysBench release): fix_code/_v8_work/_srch.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
import sys
from fixlib import call_serper
for q in sys.argv[1:]:
    print('=== Q:', q)
    for o in call_serper(q, tag='fix7') or []:
        print(' -', o.get('link')); print('   ', ' '.join((o.get('snippet') or '').split())[:400])
