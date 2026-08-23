from __future__ import annotations
import json,random,time,math
from pathlib import Path
import numpy as np,torch
from swara.models.neucodec_n2 import N2Config,N2Model
ROOT=Path(__file__).parents[2];SEED=20260823
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def load(name):return [json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data'/name).read_text().splitlines() if x.strip()]
def item(r):return torch.tensor([r['linguistic_ids']],dtype=torch.long),torch.tensor([np.load(ROOT/r['codec_token_path'])],dtype=torch.long),r['utterance_id']
def sim(a,b):n=min(len(a),len(b));return float(np.mean(np.asarray(a[:n])==np.asarray(b[:n])))
def stats(a):
 x=np.asarray(a);_,c=np.unique(x,return_counts=True);p=c/c.sum();return {'length':int(len(x)),'unique':int(np.unique(x).size),'entropy_bits':float(-(p*np.log2(p)).sum()),'top_share':float(c.max()/len(x)),'change_rate':float(np.mean(x[1:]!=x[:-1])) if len(x)>1 else 0.0}
def prob(step):
 if step<=100:return 1.0
 if step<=250:return .9
 if step<=400:return .75
 if step<=600:return .5
 return .25
def mixed(m,x,y,p):
 with torch.no_grad():
  base=torch.cat([torch.full((1,1),m.config.bos_id,dtype=torch.long),y[:,:-1]],1);pred=m(x,base,y)[0].argmax(-1)
 inp=base.clone()
 if p<1:inp[:,1:]=torch.where(torch.rand_like(inp[:,1:].float())<p,y[:,:-1],pred[:,:-1]).detach()
 return inp
def evaluate_tf(m,rs):
 m.eval();out=[]
 with torch.no_grad():
  for r in rs:
   x,y,u=item(r);inp=torch.cat([torch.full((1,1),m.config.bos_id,dtype=torch.long),y[:,:-1]],1);log,l,_=m(x,inp,y);p=log.argmax(-1)[0].numpy();t=y[0].numpy();out.append({'utterance_id':u,'ce':float(l),'accuracy':sim(p,t),'pred_stats':stats(p),'target_stats':stats(t)})
 return out
def evaluate_free(m,rs):
 m.eval();out=[]
 with torch.no_grad():
  for r in rs:
   x,y,u=item(r);p=m.generate(x,y.shape[1])[0].numpy();out.append({'utterance_id':u,'generated':p.tolist(),'target':y[0].numpy().tolist(),'stats':stats(p),'target_stats':stats(y[0].numpy()),'target_similarity':sim(p,y[0].numpy())})
 return out
def main():
 seed();tr,va=load('train_manifest.jsonl'),load('val_manifest.jsonl');m=N2Model(N2Config(502));opt=torch.optim.AdamW(m.parameters(),lr=1e-3);run=ROOT/'runs/neucodec_n2_5min_v1';run.mkdir(parents=True,exist_ok=True);torch.save({'model':m.state_dict(),'config':m.config.__dict__},run/'initial.pt');curve=[];best=1e30;best_step=0;t0=time.perf_counter()
 for step in range(1,1501):
  x,y,_=item(tr[(step-1)%len(tr)]);opt.zero_grad();_,loss,_=m(x,mixed(m,x,y,prob(step)),y);loss.backward();opt.step()
  if step in {1,100,250,500,750,1000,1250,1500}:
   a=evaluate_tf(m,tr);b=evaluate_tf(m,va);row={'step':step,'teacher_forcing_probability':prob(step),'train_ce':float(np.mean([z['ce'] for z in a])),'train_accuracy':float(np.mean([z['accuracy'] for z in a])),'val_ce':float(np.mean([z['ce'] for z in b])),'val_accuracy':float(np.mean([z['accuracy'] for z in b])),'train':a,'val':b};curve.append(row);print({k:row[k] for k in ('step','teacher_forcing_probability','train_ce','train_accuracy','val_ce','val_accuracy')},flush=True)
   if row['val_ce']<best:best=row['val_ce'];best_step=step;torch.save({'model':m.state_dict(),'config':m.config.__dict__},run/'best.pt')
 torch.save({'model':m.state_dict(),'config':m.config.__dict__},run/'final.pt');bestm=N2Model(N2Config(502));bestm.load_state_dict(torch.load(run/'best.pt',weights_only=False)['model']);free=evaluate_free(bestm,va);result={'params':sum(p.numel() for p in bestm.parameters()),'steps':1500,'schedule':'1-100:1.0; 101-250:0.9; 251-400:0.75; 401-600:0.5; 601-1500:0.25','best_step':best_step,'curve':curve,'free_validation':free,'wall_seconds':time.perf_counter()-t0};(ROOT/'experiments/neucodec_n2_history_v1/reports/n2_5min_metrics.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
