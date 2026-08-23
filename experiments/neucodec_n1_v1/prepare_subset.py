"""Build the deterministic nested ~4 min train/~1 min validation panel."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).parents[2]; OUT=ROOT/'experiments/neucodec_n1_v1/data'; OUT.mkdir(parents=True,exist_ok=True)
def load(name):
    rows=[json.loads(x) for x in (ROOT/'data/spicor_eng_m_spk001_v1/manifests'/name).read_text().splitlines() if x.strip()]
    return sorted(rows,key=lambda r:r['source_id'])
def take(rows,target):
    out=[]; total=0.0
    for r in rows:
        d=float(r['source_duration_seconds'])
        if out and total+d > target and total >= target-15: break
        wav = f"data/spicor_eng_m_spk001_v1/audio_24k/{r['source_id']}.wav"
        out.append({'utterance_id':r['source_id'],'source_wav':wav,'source_text':r['source_text'],'training_text':r['training_text'],'duration_seconds':d,'split':r['split'],'speaker_id':'ENG_M_SPK001','language':'en-IN'})
        total += d
    return out,total
train,td=take(load('debug_30min_train.jsonl'),240.0)
val,vd=take(load('debug_30min_val.jsonl'),60.0)
for name,rows in [('train_manifest.jsonl',train),('val_manifest.jsonl',val)]:
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
print('train',len(train),td,'val',len(val),vd)
