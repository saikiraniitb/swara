import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).parents[2];sys.path.insert(0,str(ROOT/'experiments/neucodec_n2_history_v1'))
import train_5min as t
from swara.models.neucodec_n2 import N2Config,N2Model
def js(a,b):
 p=a/a.sum();q=b/b.sum();m=(p+q)/2; z=0
 for x,y,w in zip(p,q,m):
  if x:z+=.5*x*np.log2(x/w)
  if y:z+=.5*y*np.log2(y/w)
 return float(z)
def main():
 tr=t.load('train_manifest.jsonl');va=t.load('val_manifest.jsonl');m=N2Model(N2Config(502));m.load_state_dict(torch.load(ROOT/'runs/neucodec_n2_5min_v1/best.pt',weights_only=False)['model']);free=t.evaluate_free(m,va)
 real=np.concatenate([np.load(ROOT/r['codec_token_path']).reshape(-1) for r in tr]);real_ids=set(real.tolist());bigrams=set(zip(real[:-1].tolist(),real[1:].tolist()));hist=np.bincount(real,minlength=65536).astype(float)
 for z in free:
  x=np.asarray(z['generated']);c=np.bincount(x,minlength=65536).astype(float);pairs=list(zip(x[:-1].tolist(),x[1:].tolist()));z['ids_seen_in_train']=float(np.mean([int(v in real_ids) for v in x]));z['unigram_js_bits']=js(c,hist);z['real_bigram_overlap']=float(np.mean([p in bigrams for p in pairs])) if pairs else 0.;z['transition_entropy_bits']=float(-(np.bincount(x[1:][x[1:]!=x[:-1]],minlength=65536)/max(1,np.sum(x[1:]!=x[:-1]))*np.log2(np.maximum(1,np.bincount(x[1:][x[1:]!=x[:-1]],minlength=65536))/max(1,np.sum(x[1:]!=x[:-1])))).sum())
 # same fixed text swap: each item gets the next validation text and its own duration.
 swaps=[]
 for i,r in enumerate(va):
  wrong=va[(i+1)%len(va)];x,_y,_=t.item(r);wx,_wy,_=t.item(wrong);p=m.generate(x,int(np.load(ROOT/r['codec_token_path']).shape[0]))[0].numpy();q=m.generate(wx,int(np.load(ROOT/r['codec_token_path']).shape[0]))[0].numpy();swaps.append({'from':r['utterance_id'],'to':wrong['utterance_id'],'changed_ratio':1-t.sim(p,q)})
 out=json.loads((ROOT/'experiments/neucodec_n2_history_v1/reports/n2_5min_metrics.json').read_text());out['free_validation']=free;out['free_running_summary']={'max_nonself_similarity':max(t.sim(a['generated'],b['generated']) for i,a in enumerate(free) for b in free[i+1:]),'pairwise_mean_similarity':float(np.mean([t.sim(a['generated'],b['generated']) for i,a in enumerate(free) for b in free[i+1:]])),'text_swaps':swaps,'min_text_swap_change':min(z['changed_ratio'] for z in swaps)}; (ROOT/'experiments/neucodec_n2_history_v1/reports/n2_5min_metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['free_running_summary'],indent=2))
if __name__=='__main__':main()
