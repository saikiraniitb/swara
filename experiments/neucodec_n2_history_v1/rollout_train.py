from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np, torch
from swara.models.neucodec_n2 import N2Config, N2Model

ROOT=Path(__file__).parents[2]; SEED=20260823
def seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def rows(): return [json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data/train_manifest.jsonl').read_text().splitlines() if x.strip()][:2]
def item(r): return torch.tensor([r['linguistic_ids']],dtype=torch.long), torch.tensor([np.load(ROOT/r['codec_token_path'])],dtype=torch.long), r['utterance_id']
def sim(a,b): n=min(len(a),len(b)); return float(np.mean(np.asarray(a[:n])==np.asarray(b[:n])))
def stats(a):
    x=np.asarray(a); _,c=np.unique(x,return_counts=True); p=c/c.sum(); return {'length':int(len(x)),'unique':int(np.unique(x).size),'entropy_bits':float(-(p*np.log2(p)).sum()),'top_share':float(c.max()/len(x))}
def divergence(a,b):
    n=min(len(a),len(b)); z=np.flatnonzero(np.asarray(a[:n])!=np.asarray(b[:n])); i=int(z[0]) if len(z) else n
    return {'first_mismatch':i,'correct_prefix':i,'target_at_mismatch':int(b[i]) if i<n else None,'predicted_at_mismatch':int(a[i]) if i<n else None,'next20_accuracy':sim(np.asarray(a[i:i+20]),np.asarray(b[i:i+20])) if i<n else 1.0,'overall_accuracy':sim(a,b)}
def free(m,rs):
    out=[]
    for r in rs:
        x,y,u=item(r); p=m.generate(x,y.shape[1])[0].numpy(); t=y[0].numpy(); z=divergence(p,t); z.update({'utterance_id':u,'stats':stats(p)}); out.append(z)
    return out
def tf(m,rs):
    out=[]
    with torch.no_grad():
        for r in rs:
            x,y,u=item(r); inp=torch.cat([torch.full((1,1),m.config.bos_id,dtype=torch.long),y[:,:-1]],1); log,l,_=m(x,inp,y); p=log.argmax(-1)[0].numpy(); out.append({'utterance_id':u,'ce':float(l),'accuracy':sim(p,y[0].numpy())})
    return out
def probability(step):
    if step<=50:return 1.0
    if step<=100:return .9
    if step<=150:return .75
    if step<=200:return .5
    return .25
def mixed_inputs(m,x,y,teacher_p):
    with torch.no_grad():
        base=torch.cat([torch.full((1,1),m.config.bos_id,dtype=torch.long),y[:,:-1]],1); pred=m(x,base,y)[0].argmax(-1)
    inp=base.clone()
    if teacher_p<1:
        mask=torch.rand_like(inp[:,1:].float())<teacher_p; inp[:,1:]=torch.where(mask,y[:,:-1],pred[:,:-1]).detach()
    return inp
def main():
    seed(); rs=rows(); m=N2Model(N2Config(502)); m.load_state_dict(torch.load(ROOT/'runs/neucodec_n2_history_v1/n2_0/final.pt',weights_only=False)['model']); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); run=ROOT/'runs/neucodec_n2_history_v1/n2_rollout'; run.mkdir(parents=True,exist_ok=True); torch.save({'model':m.state_dict(),'config':m.config.__dict__},run/'initial.pt')
    metrics={'baseline':{'free':free(m,rs)},'schedule':{},'steps':300,'training_method':'sequence-level detached self-conditioning; predicted previous tokens replace teacher-forced history according to schedule'}; t0=time.perf_counter()
    for step in range(1,301):
        x,y,_=item(rs[(step-1)%2]); opt.zero_grad(); inp=mixed_inputs(m,x,y,probability(step)); _,loss,_=m(x,inp,y); loss.backward(); opt.step()
        if step in {1,50,100,150,200,250,300}:
            metrics['schedule'][str(step)]={'teacher_forcing_probability':probability(step),'train_tf':tf(m,rs),'free':free(m,rs)}; torch.save({'model':m.state_dict(),'config':m.config.__dict__},run/f'step_{step}.pt'); print(step,metrics['schedule'][str(step)],flush=True)
    torch.save({'model':m.state_dict(),'config':m.config.__dict__},run/'final.pt'); metrics['wall_seconds']=time.perf_counter()-t0; (ROOT/'experiments/neucodec_n2_history_v1/reports/n2_rollout_metrics.json').write_text(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
