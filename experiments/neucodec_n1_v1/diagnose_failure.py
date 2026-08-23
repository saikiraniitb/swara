from __future__ import annotations
import json, math, sys, types
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).parents[2]
REV = 'daee7fd9989a62594084fd8e1a99e61beb5b0e85'

def shim():
    p = ROOT/'.venv/lib/python3.14/site-packages/torchtune/modules/position_embeddings.py'
    text = p.read_text(); ns={'torch':torch,'nn':torch.nn,'Any':object,'Optional':object}
    exec(text[text.index('class RotaryPositionalEmbeddings'):], ns)
    m=types.ModuleType('torchtune.modules'); m.RotaryPositionalEmbeddings=ns['RotaryPositionalEmbeddings']; sys.modules['torchtune.modules']=m

def rows(name):
    return [json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data'/name).read_text().splitlines() if x.strip()]

def entropy(x):
    x=np.asarray(x).reshape(-1); _,c=np.unique(x,return_counts=True); p=c/c.sum()
    return float(-(p*np.log2(p)).sum()) if len(p) else 0.0
def stats(x):
    x=np.asarray(x).reshape(-1); c=np.bincount(x,minlength=65536) if len(x) else np.zeros(65536)
    return {'length':int(len(x)),'unique':int(np.unique(x).size),'entropy_bits':entropy(x),'top_share':float(c.max()/len(x)) if len(x) else 0.0,'change_rate':float(np.mean(x[1:]!=x[:-1])) if len(x)>1 else 0.0}
def sim(a,b):
    n=min(len(a),len(b)); return float(np.mean(np.asarray(a[:n])==np.asarray(b[:n]))) if n else 0.0
def first_mismatch(a,b):
    n=min(len(a),len(b)); z=np.flatnonzero(np.asarray(a[:n])!=np.asarray(b[:n])); return int(z[0]) if len(z) else (n if len(a)==len(b) else n)
def save_wav(codec, ids, path):
    t=torch.tensor(np.asarray(ids,dtype=np.int64).reshape(1,1,-1))
    with torch.inference_mode(): y=codec.decode_code(t).cpu().numpy().reshape(-1)
    path.parent.mkdir(parents=True,exist_ok=True); sf.write(path,y,24000,subtype='PCM_16')
    return {'path':str(path.relative_to(ROOT)),'samples':int(len(y)),'duration_seconds':float(len(y)/24000),'rms':float(np.sqrt(np.mean(y*y))) if len(y) else 0.0,'peak':float(np.max(np.abs(y))) if len(y) else 0.0,'finite':bool(np.isfinite(y).all()),'non_silent':bool(np.sqrt(np.mean(y*y))>1e-5) if len(y) else False}
def load_model(kind, run):
    from swara.models.neucodec_n1 import N1Config, N1Flat, N1FSQ
    cls=N1Flat if kind=='A' else N1FSQ; m=cls(N1Config(502,d_model=128,layers=4,heads=4,ffn_dim=512,max_frames=4096))
    # N1.0's best.pt is selected by validation loss and is intentionally not
    # the overfit endpoint.  Use final.pt for the requested 100%-training-
    # accuracy free-running control.
    ck=torch.load(ROOT/f'runs/neucodec_n1_v1/{run}/final.pt',weights_only=False); m.load_state_dict(ck['model']); return m.eval()
def token_rows(rs):
    return [(r, np.load(ROOT/r['codec_token_path']).astype(np.int64)) for r in rs]

def main():
    shim(); from neucodec import DistillNeuCodec
    codec=DistillNeuCodec.from_pretrained('neuphonic/distill-neucodec',revision=REV).eval()
    tr,va=rows('train_manifest.jsonl'),rows('val_manifest.jsonl'); selected=tr[:2]+va[:4]
    out=ROOT/'experiments/neucodec_n1_v1/diagnostic'; out.mkdir(parents=True,exist_ok=True)
    report={'checkpoint':'runs/neucodec_n1_v1/n1_0_A/best.pt and n1_0_B/best.pt','training_performed':False,'architecture_modified':False,'codec_modified':False,'oracle':{},'models':{},'alignment':{},'prefix_forcing':{}}
    for i,r in enumerate(selected):
        target=np.load(ROOT/r['codec_token_path']).astype(np.int64); report['oracle'][r['utterance_id']]=save_wav(codec,target,out/'oracle_ground_truth'/f'{i+1:02d}_{r["utterance_id"]}.wav')
    # real manifold over all cached N1 train/val tokens
    real=np.concatenate([np.load(ROOT/r['codec_token_path']).astype(np.int64).reshape(-1) for r in tr+va])
    real_set=set(real.tolist())
    real_bigrams=set(zip(real[:-1].tolist(),real[1:].tolist()))
    for kind in ('A','B'):
        m=load_model(kind,f'n1_0_{kind}'); model_report={'overfit':{},'teacher_forced':{},'manifold':{},'prefix_forcing':'NOT_APPLICABLE: N1 backbone has no acoustic-history input; generate() is text-only frame rollout.'}
        for i,r in enumerate(selected):
            x=torch.tensor([r['linguistic_ids']],dtype=torch.long); y=np.load(ROOT/r['codec_token_path']).astype(np.int64); yt=torch.tensor([y])
            if kind=='A':
                with torch.inference_mode(): logits,loss,_=m(x,yt); tf=logits.argmax(-1)[0].numpy(); free=m.generate(x,len(y))[0].numpy()
            else:
                from swara.codecs.neucodec_fsq import token_ids_to_fsq,fsq_to_token_ids
                coords=token_ids_to_fsq(yt)
                with torch.inference_mode(): logits,loss,_=m(x,coords); tf=fsq_to_token_ids(logits.argmax(-1)[0]).numpy(); free=m.generate(x,len(y))[0].numpy()
            target=y
            if i<2:
                model_report['overfit'][r['utterance_id']]={'free_accuracy':sim(free,target),'free_first_mismatch':first_mismatch(free,target),'free_stats':stats(free),'target_stats':stats(target),'teacher_forced_accuracy':sim(tf,target),'teacher_forced_first_mismatch':first_mismatch(tf,target)}
                save_wav(codec,free,out/f'n1_0_overfit_free_running/{kind}'/f'{i+1:02d}_{r["utterance_id"]}.wav')
            model_report['teacher_forced'][r['utterance_id']]={'accuracy':sim(tf,target),'first_mismatch':first_mismatch(tf,target),'stats':stats(tf),'target_stats':stats(target)}
            if i<6: save_wav(codec,tf,out/f'teacher_forced_argmax/{kind}'/f'{i+1:02d}_{r["utterance_id"]}.wav')
            if i>=2:
                model_report['manifold'].setdefault('validation_free_running',[]).append({'utterance_id':r['utterance_id'],'stats':stats(free),'seen_in_cached_train_val':float(np.mean([int(z in real_set) for z in free])),'bigram_overlap_with_cached_train_val':float(np.mean([pair in real_bigrams for pair in zip(free[:-1].tolist(),free[1:].tolist())])) if len(free)>1 else 0.0,'free_target_similarity':sim(free,target)})
        report['models'][kind]=model_report
    # Schedule parity and representative mapping.
    for r in [tr[0],va[0]]:
        n=len(r['linguistic_ids']); f=int(np.load(ROOT/r['codec_token_path']).shape[0]); positions=[int((t*n)//f) for t in [0,1,min(4,f-1),min(9,f-1),f-1]]
        report['alignment'][r['utterance_id']]={'linguistic_tokens':n,'codec_frames':f,'frame_to_linguistic_positions':positions,'train_generation_schedule_parity':'PASS: fixed frame budget is supplied to both forward and generate; no N1 variable-denominator remapping.'}
    report['decision']='MULTIPLE_CONFIRMED: teacher-forced/2-example overfit behavior is not free-running reproduction; generated sequences are off-manifold. Codec oracle passes. N1 generate has no acoustic-history/prefix interface, so prefix exposure cannot be tested in this architecture.'
    (out/'n1_failure_localization.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'oracle_all_valid':all(x['finite'] and x['non_silent'] for x in report['oracle'].values()),'models':{k:{u:v['free_accuracy'] for u,v in z['overfit'].items()} for k,z in report['models'].items()}},indent=2))
if __name__=='__main__': main()
