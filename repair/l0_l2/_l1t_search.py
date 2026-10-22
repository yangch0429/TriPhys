# Provenance (TriPhysBench release): fix_code/_v8_work/_l1t_search.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
"""Targeted Serper searches for the facts the 74 unrepaired QAs still lack (tag L1T|<n>); snippets -> l1t_search.txt"""
import json
from concurrent.futures import ThreadPoolExecutor
from fixlib import WORK, call_serper

Q = [
    ('8', 'hip height as percentage of stature anthropometry'), ('8', 'trochanteric height proportion of body height'),
    ('17', 'HABA marble run water wheel weight grams'), ('17', 'marble run paddle wheel piece weight'),
    ('22', 'wooden marble run curved track radius'), ('22', 'marble run curve piece radius cm'),
    ('42', 'wooden marble run corner piece radius inches'), ('42', 'marble run 90 degree corner track radius'),
    ('26', 'empty haul bag weight kg big wall'), ('26', 'loaded haul bag weight big wall climbing'),
    ('33', 'action figure weight grams'), ('33', 'action figure base width height dimensions cm'),
    ('34', 'Bruder toy crane hook dimensions'), ('34', 'drag coefficient irregular bluff body free fall'),
    ('48', 'wooden marble run ball weight grams'), ('48', 'ball collision contact time milliseconds wood'),
    ('52', 'eraser impact contact time milliseconds'), ('52', 'rubber impact duration on fabric milliseconds'),
    ('55', 'woven fabric skin friction coefficient airflow'), ('55', 'woven fabric surface roughness height mm'),
    ('69', 'landing from jump stopping distance knee bend cm'), ('69', 'jump landing deceleration distance meters grass'),
    ('73', 'steel on stone static friction coefficient'), ('73', 'metal on masonry coefficient of friction'),
    ('46', 'billiard ball cloth static friction coefficient'), ('46', 'cue ball felt coefficient of friction'),
    ('29', 'water droplet superhydrophobic surface sliding friction coefficient'),
    ('71', 'plastic storage bin water depth bath bomb'), ('71', 'bath bomb density g/cm3'),
]


def one(x):
    n, q = x
    return n, q, call_serper(q, 'L1T|' + n) or []


with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(one, Q))
out = []
for n, q, rs in res:
    out.append('### [%s] %s' % (n, q))
    for i, r in enumerate(rs, 1):
        out.append('  %d. %s | %s\n     %s' % (i, r['title'][:80], r['link'][:110], r['snippet']))
open(WORK + '/l1t_search.txt', 'w', encoding='utf-8').write('\n'.join(out))
print(len(res), 'queries')
