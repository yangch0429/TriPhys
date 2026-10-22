# Provenance (TriPhysBench release): fix_code/_v8_work/_fix39_search.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Targeted Serper searches for the recheck-problem QAs (tag FIX39|R<n>); snippets -> fix39_search.txt"""
from concurrent.futures import ThreadPoolExecutor
from fixlib import call_serper

Q = [
    ('7', 'ideal body weight for a 6 ft 1 in man kg'), ('7', 'ideal weight for 186 cm male'),
    ('8', 'how much does a mountain bike wheel weigh'), ('8', 'average weight of a 29er mountain bike front wheel'),
    ('12', 'layup release angle degrees biomechanics'), ('12', 'basketball layup shot release angle study'),
    ('19', 'standard glass marble diameter mm'), ('19', 'most common marble size mm'),
    ('23', 'cured epoxy resin density g/cm3'), ('23', 'epoxy resin specific gravity typical'),
    ('29', 'weight of a 16mm glass marble grams'), ('29', 'standard glass marble weight grams'),
    ('31', 'energy released by burning a match head joules'), ('31', 'how much energy does one match produce'),
    ('35', 'most common road bike crank length'), ('35', 'standard crank arm length road bike'),
]


def one(x):
    n, q = x
    return n, q, call_serper(q, 'FIX39|R' + n) or []


with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(one, Q))
out = []
for n, q, rs in res:
    out.append('### [R%s] %s' % (n, q))
    for i, r in enumerate(rs, 1):
        out.append('  %d. %s | %s\n     %s' % (i, r['title'][:80], r['link'][:110], r['snippet']))
open('fix39_search.txt', 'w', encoding='utf-8').write('\n'.join(out))
print(len(out))
