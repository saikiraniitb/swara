"""Bounded N1-A/N1-B training and token-only evaluation."""
from __future__ import annotations
import json, math, time, random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from swara.codecs.neucodec_fsq import token_ids_to_fsq
from swara.models.neucodec_n1 import N1Config, N1Flat, N1FSQ

ROOT=Path(__file__).parents[2]; SEED=20260823
def seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def load(name): return [json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data'/name).read_text().splitlines() if x.strip()]
def entropy(a, classes=None):
    a=np.asarray(a); _,c=np.unique(a,return_counts=True); p=c/c.sum(); return float(-(p*np.log2(p)).sum())
def sim(a,b):
    n=min(len(a),len(b)); return float((np.asarray(a[:n])==np.asarray(b[:n])).mean()) if n else 0.0
def stats(ids):
    x=np.asarray(ids); return {'length':int(x.size),'unique':int(np.unique(x).size),'entropy_bits':entropy(x),'most_frequent_share':float(np.bincount(x).max()/x.size) if x.size else 0.0}
def make_config(vocab): return N1Config(vocab,d_model=128,layers=4,heads=4,ffn_dim=512,max_frames=4096)
def data(rows):
    return [(torch.tensor([r['linguistic_ids']],dtype=torch.long),torch.tensor([np.load(ROOT/r['codec_token_path'])],dtype=torch.long),r['utterance_id']) for r in rows]
def eval_model(model, items, kind):
    model.eval(); out=[]; losses=[]
    with torch.no_grad():
      for x,y,uid in items:
        if kind=='A': logits,loss,_=model(x,y); pred=logits.argmax(-1)[0].cpu().numpy(); target=y[0].cpu().numpy(); ce=float(loss)
        else:
          coords=token_ids_to_fsq(y); logits,loss,_=model(x,coords); pred=logits.argmax(-1)[0]; target=coords[0]; ce=float(loss); pred=__import__('swara.codecs.neucodec_fsq',fromlist=['fsq_to_token_ids']).fsq_to_token_ids(pred).cpu().numpy(); target= y[0].cpu().numpy()
        out.append({'utterance_id':uid,'ce':ce,'stats':stats(pred),'target_stats':stats(target),'accuracy':float((pred==target).mean()),'pred':pred.tolist(),'target':target.tolist()})
    return out
def train(kind, rows, val_rows, steps, root):
    seed(); vocab=502; model=(N1Flat if kind=='A' else N1FSQ)(make_config(vocab)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); train_items=data(rows); val_items=data(val_rows); root.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'config':asdict(make_config(vocab))},root/'initial.pt'); curve=[]; best=float('inf'); best_step=0; t0=time.perf_counter()
    points={1,50,100,200,250,300,500,750,1000};
    for step in range(1,steps+1):
      x,y,_=train_items[(step-1)%len(train_items)]; opt.zero_grad(); target= y if kind=='A' else token_ids_to_fsq(y); _,loss,_=model(x,target); loss.backward(); opt.step()
      if step in points or step==steps:
        tr=eval_model(model,train_items,kind); va=eval_model(model,val_items,kind); val_loss=float(np.mean([z['ce'] for z in va])); row={'step':step,'train_loss':float(np.mean([z['ce'] for z in tr])),'val_loss':val_loss,'train_accuracy':float(np.mean([z['accuracy'] for z in tr])),'val_accuracy':float(np.mean([z['accuracy'] for z in va]))}
        if kind=='B': row['train_joint_nll']=row['train_loss']*8; row['val_joint_nll']=row['val_loss']*8
        curve.append(row); print(kind,row,flush=True)
        if val_loss<best: best=val_loss; best_step=step; torch.save({'model':model.state_dict(),'config':asdict(make_config(vocab))},root/'best.pt')
    torch.save({'model':model.state_dict(),'config':asdict(make_config(vocab))},root/'final.pt')
    return model,{'kind':kind,'steps':steps,'best_step':best_step,'curve':curve,'wall_seconds':time.perf_counter()-t0,'train_final':eval_model(model,train_items,kind),'val_final':eval_model(model,val_items,kind),'best_val_loss':best}
def free_run(model, rows, kind):
    model.eval(); items=data(rows); generated=[]
    with torch.no_grad():
      for x,y,uid in items:
        pred=model.generate(x,y.shape[1])[0].cpu().numpy(); generated.append({'utterance_id':uid,'generated':pred.tolist(),'target':y[0].cpu().numpy().tolist(),'stats':stats(pred)})
    sims=[]
    for i,a in enumerate(generated):
      for j,b in enumerate(generated):
        if i<j: sims.append(sim(a['generated'],b['generated']))
    swaps=[]
    for i,(x,y,uid) in enumerate(items):
      for j,(xx,yy,uid2) in enumerate(items):
        if i!=j:
          p1=generated[i]['generated']; p2=model.generate(xx,y.shape[1])[0].cpu().numpy().tolist(); swaps.append({'from':uid,'to':uid2,'changed_ratio':1-sim(p1,p2)})
          break
    return {'generated':generated,'max_nonself_similarity':max(sims) if sims else 0.0,'pairwise_mean_similarity':float(np.mean(sims)) if sims else 0.0,'text_swaps':swaps,'min_text_swap_change':min(x['changed_ratio'] for x in swaps) if swaps else 0.0}
def main():
    seed(); tr=load('train_manifest.jsonl'); va=load('val_manifest.jsonl'); two=tr[:2]
    base=ROOT/'runs/neucodec_n1_v1'; base.mkdir(parents=True,exist_ok=True)
    results={}
    for kind in ('A','B'):
      _,r=train(kind,two,va,300,base/f'n1_0_{kind}'); results[kind]=r
      (ROOT/f'experiments/neucodec_n1_v1/reports/n1_{kind.lower()}_n10_metrics.json').write_text(json.dumps(r,indent=2))
    learned=any(r['curve'][-1]['train_accuracy']>0.02 for r in results.values())
    if not learned: print('N1.0 BOTH_FAILED'); return
    for kind in ('A','B'):
      _,r=train(kind,tr,va,1000,base/f'n1_1_{kind}')
      # Reload explicitly for a clean evaluation-only pass.
      m=(N1Flat if kind=='A' else N1FSQ)(make_config(502)); m.load_state_dict(torch.load(base/f'n1_1_{kind}/best.pt',weights_only=False)['model']); r['free_running']=free_run(m,va,kind)
      (ROOT/f'experiments/neucodec_n1_v1/reports/n1_{kind.lower()}_metrics.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__': main()
