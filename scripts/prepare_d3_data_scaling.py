#!/usr/bin/env python3
"""Freeze nested D3 data-scaling rung manifests without training."""
from __future__ import annotations
import json, hashlib, subprocess, os, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALIGN=ROOT/'experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl'
BASE=ROOT/'experiments/neucodec_n1_v1/data'
OUT=ROOT/'experiments/swara_speech_poc_v1/reports/d3_rungs'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def phon(w):
    p=subprocess.run(['/opt/homebrew/bin/espeak-ng','-q','--ipa=3','-v','en-us','--',w],capture_output=True,text=True,check=True,env={**os.environ,'ESPEAK_DATA_PATH':'/opt/homebrew/Cellar/espeak-ng/1.52.0/share/espeak-ng-data'})
    return p.stdout.strip()
def main():
    rows=[json.loads(x) for x in ALIGN.read_text().splitlines() if x.strip()]
    tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']
    frozen32=[json.loads(x)['utterance_id'] for x in (BASE/'train_manifest.jsonl').read_text().splitlines() if x.strip()]
    frozenval=[json.loads(x)['utterance_id'] for x in (BASE/'val_manifest.jsonl').read_text().splitlines() if x.strip()]
    by={r['utterance_id']:r for r in tr}; ordered=frozen32+[r['utterance_id'] for r in sorted(tr,key=lambda x:x['utterance_id']) if r['utterance_id'] not in set(frozen32)]
    assert len(ordered)==267 and len(set(ordered))==267
    assert set(frozenval).isdisjoint(ordered) and len(frozenval)==8
    all_words={t for r in rows for u in r['units'] if u['token_kind']=='grapheme' for t in [u['token_value'].strip().lower()]}
    failures=[]
    for w in sorted(all_words):
        if not phon(w): failures.append(w)
    OUT.mkdir(parents=True,exist_ok=True)
    rungs=[]
    for n in (32,64,128,267):
        ids=ordered[:n]; payload={'rung':n,'train_ids':ids,'validation_ids':frozenval,'train_rows':n,'validation_rows':8,'nested_parent':None if n==32 else (32 if n==64 else 64 if n==128 else 128),'train_frames':sum(by[i]['neucodec_frames'] for i in ids),'validation_frames':sum(next(r for r in va if r['utterance_id']==i)['neucodec_frames'] for i in frozenval)}
        path=OUT/f'{n}.json'; path.write_text(json.dumps(payload,indent=2)+'\n'); rungs.append(payload)
    report={'schema_version':'swara.d3.data_scaling_manifest.v1','seed':20260824,'phonemizer':{'tool':'eSpeak NG','version':'1.52.0','voice':'en-us','command':'espeak-ng -q --ipa=3 -v en-us -- WORD','coverage_success':len(all_words)-len(failures),'coverage_total':len(all_words),'failures':failures},'alignment_manifest_sha256':sha(ALIGN),'c1_train_manifest_sha256':sha(BASE/'train_manifest.jsonl'),'c1_val_manifest_sha256':sha(BASE/'val_manifest.jsonl'),'train_rows':267,'validation_rows':8,'train_validation_overlap':sorted(set(ordered)&set(frozenval)),'rungs':rungs,'training_exposure':{'reference':'D2 32-item full-batch protocol','d2_optimizer_steps':500,'d2_effective_epochs':500,'planned_optimizer_steps':{str(n):500 for n in (32,64,128,267)},'planned_effective_epochs':{str(n):500 for n in (32,64,128,267)}}}
    (ROOT/'experiments/swara_speech_poc_v1/reports/swara_d3_data_scaling_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
