from __future__ import annotations
import json, math, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from swara.models.neucodec_n2 import N2Config,N2Model,parameter_count

ROOT=Path(__file__).parents[2]; SEED=20260823
def seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def load(name): return [json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data'/name).read_text().splitlines() if x.strip()]
def item(r): return torch.tensor([r['linguistic_ids']],dtype=torch.long), torch.tensor([np.load(ROOT/r['codec_token_path'])],dtype=torch.long), r['utterance_id']
def stats(a):
 x=np.asarray(a); _,c=np.unique(x,return_counts=True); p=c/c.sum(); return {'length':int(x.size),'unique':int(np.unique(x).size),'entropy_bits':float(-(p*np.log2(p)).sum()),'top_share':float(c.max()/x.size),'change_rate':float(np.mean(x[1:]!=x[:-1])) if x.size>1 else 0.0}
def sim(a,b): n=min(len(a),len(b)); return float(np.mean(np.asarray(a[:n])==np.asarray(b[:n])))
def eval_tf(m,rs):
 m.eval(); out=[]
 with torch.no_grad():
  for r in rs:
   x,y,u=item(r); inp=torch.cat([torch.full((1,1),m.config.bos_id,dtype=torch.long),y[:,:-1]],1); logits,l,_=m(x,inp,y); p=logits.argmax(-1)[0].numpy(); t=y[0].numpy(); out.append({'utterance_id':u,'ce':float(l),'accuracy':sim(p,t),'pred_stats':stats(p),'target_stats':stats(t),'pred':p.tolist(),'target':t.tolist()})
 return out
def eval_free(m,rs):
 m.eval(); out=[]
 with torch.no_grad():
  for r in rs:
   x,y,u=item(r); p=m.generate(x,y.shape[1])[0].numpy(); t=y[0].numpy(); out.append({'utterance_id':u,'accuracy':sim(p,t),'stats':stats(p),'target_stats':stats(t),'generated':p.tolist(),'target':t.tolist()})
 return out
def train(rows,val,steps,run):
 seed(); m=N2Model(N2Config(502)); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); run.mkdir(parents=True,exist_ok=True); torch.save({'model':m.state_dict(),'config':asdict(m.config)},run/'initial.pt'); curve=[];best=1e30;best_step=0;t0=time.perf_counter(); points={1,50,100,200,250,300,500,750,1000}
 for step in range(1,steps+1):
  x,y,_=item(rows[(step-1)%len(rows)]); inp=torch.cat([torch.full((1,1),m.config.bos_id,dtype=torch.long),y[:,:-1]],1);opt.zero_grad();_,l,_=m(x,inp,y);l.backward();opt.step()
  if step in points or step==steps:
   tr=eval_tf(m,rows);va=eval_tf(m,val);vl=float(np.mean([z['ce'] for z in va])); row={'step':step,'train_ce':float(np.mean([z['ce'] for z in tr])),'val_ce':vl,'train_accuracy':float(np.mean([z['accuracy'] for z in tr])),'val_accuracy':float(np.mean([z['accuracy'] for z in va]))};curve.append(row);print(row,flush=True)
   if vl<best:best=vl;best_step=step;torch.save({'model':m.state_dict(),'config':asdict(m.config)},run/'best.pt')
 torch.save({'model':m.state_dict(),'config':asdict(m.config)},run/'final.pt'); return {'steps':steps,'best_step':best_step,'best_val_ce':best,'wall_seconds':time.perf_counter()-t0,'curve':curve,'train_final':eval_tf(m,rows),'val_final':eval_tf(m,val)}
def main():
 tr,va=load('train_manifest.jsonl'),load('val_manifest.jsonl'); root=ROOT/'runs/neucodec_n2_history_v1';root.mkdir(parents=True,exist_ok=True);results={}
 # N2.0 uses final checkpoints for the explicit memorization control.
 r=train(tr[:2],va,300,root/'n2_0'); results['n2_0']=r; m=N2Model(N2Config(502));m.load_state_dict(torch.load(root/'n2_0/final.pt',weights_only=False)['model']);results['n2_0']['free']=eval_free(m,tr[:2]);
 if min(z['accuracy'] for z in results['n2_0']['free'])<0.9: print('N2.0 FAILED'); return
 r=train(tr,va,1000,root/'n2_1');results['n2_1']=r;m=N2Model(N2Config(502));m.load_state_dict(torch.load(root/'n2_1/best.pt',weights_only=False)['model']);results['n2_1']['free']=eval_free(m,va);(ROOT/'experiments/neucodec_n2_history_v1/reports').mkdir(parents=True,exist_ok=True);(ROOT/'experiments/neucodec_n2_history_v1/reports/n2_metrics.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__': main()
