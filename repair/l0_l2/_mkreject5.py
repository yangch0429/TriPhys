# Provenance (TriPhysBench release): fix_code/_v8_work/_mkreject5.py
# Role: versions 01 -> 03 (09-30 .. 10-02): L0 text fixes, L1 evidence repair, L2 regeneration and post layers.
# Release edits only: API keys, endpoints, model ids and machine paths come from environment variables
# (see repair/README.md); logic unchanged. Provenance record, not guaranteed to run end-to-end; data/ is canonical.
import json
r=json.load(open('fix5_results.json',encoding='utf-8'))
REJ={
 # cosmetic: value added to the code only to "use" a hidden value
 'A|25299d42-173e-45a8-b114-8ddee0e7a7e4|1':'cosmetic','A|aab43780-401b-4673-b2c2-fc3521e98bea|2':'cosmetic','6e44dfad-517d-4ef7-8999-938f5c66f8b7|1':'cosmetic','A|90e697a5-2cff-4ecc-a580-f4c230a5488c|2':'cosmetic',
 'A|b813c971-0f52-4d13-a985-2741f6ff064a|0':'cosmetic','daf66c5f-284b-45da-af6b-e9a45242af7f|1':'cosmetic','a00128/00907|0':'cosmetic',
 # status ok but nothing changed
 '2429d435-c4fd-4e57-a25c-d2badd786e60|1':'no-op','a00242/00872|2':'no-op','a00721/00666|1':'no-op','a00588/00347|1':'no-op','a00280/00387|0':'no-op',
 'a00870/00376|1':'no-op',
 # questionable repairs
 'A|7b078956-9843-49dd-be3c-39591bde70fe|2':'carry -> drag the wheel: not what the video shows',
 'A|e6d6f5e3-2821-4d8b-ab8f-5608c9dacb98|2':'artificial rounding range [15.3, 15.4]',
 'A|f1af22f7-699d-4961-bb21-4a9030640f4e|2':'invented ±10% uncertainty',
 'a00640/00012|0':'invented 4th figure mass; |2 resolves the count conflict instead',
 'a00138/01103|1':'invented height range 0.8-1.5 m',
 'A|90e697a5-2cff-4ecc-a580-f4c230a5488c|1':'asked quantity changed; release height is stated in the question (false alarm)',
 'A|4f1a4a10-b34c-41a3-9544-86622f7881ab|0':'ratio -> contrived difference',
 'a00100/00705|1':'asked quantity changed to angular momentum',
 'A|b813c971-0f52-4d13-a985-2741f6ff064a|1':'verdict flipped by an unfalsifiable flexed-arm model',
 'a00372/00088|0':'spoon head (47 g) heavier than the whole spoon (13.75 g)',
 'a00137/00035|1':'verdict flipped by a trivial N>=0 criterion',
 'test:B|a00315/01016|2':'swaps the dimensions with |0 (still inconsistent); |0 kept',
 'a00215/00045|2':'removes the rolling description and time window; |0 kept',
 'a00079/00862|1':'adopts 30° while |2 makes the blade horizontal; original "horizontally" is consistent',
 'A|6badc0af-4a97-4dd4-b89d-b540f6f34002|1':'discards the retrieved release velocity',
 'a00087/00122|1':'drops the voltage part (dielectric strength hidden value becomes unused)',
}
out={}
for k,why in REJ.items():
    m=[q for q in r if k in q]
    assert len(m)==1,(k,m)
    assert r[m[0]]['status']=='ok',(k,r[m[0]]['status'])
    out[m[0]]=why
json.dump(out,open('fix5_reject.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(len(out))
