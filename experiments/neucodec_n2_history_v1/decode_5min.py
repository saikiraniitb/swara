import sys,types,json
from pathlib import Path
import numpy as np,torch,soundfile as sf
ROOT=Path(__file__).parents[2]
def shim():
 p=ROOT/'.venv/lib/python3.14/site-packages/torchtune/modules/position_embeddings.py';t=p.read_text();ns={'torch':torch,'nn':torch.nn,'Any':object,'Optional':object};exec(t[t.index('class RotaryPositionalEmbeddings'):],ns);m=types.ModuleType('torchtune.modules');m.RotaryPositionalEmbeddings=ns['RotaryPositionalEmbeddings'];sys.modules['torchtune.modules']=m
def main():
 shim();from neucodec import DistillNeuCodec;from swara.models.neucodec_n2 import N2Config,N2Model
 rows=[json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data/train_manifest.jsonl').read_text().splitlines() if x.strip()][:2]+[json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data/val_manifest.jsonl').read_text().splitlines() if x.strip()];codec=DistillNeuCodec.from_pretrained('neuphonic/distill-neucodec',revision='daee7fd9989a62594084fd8e1a99e61beb5b0e85').eval();m=N2Model(N2Config(502));m.load_state_dict(torch.load(ROOT/'runs/neucodec_n2_5min_v1/best.pt',weights_only=False)['model']);m.eval();out=ROOT/'evaluations/neucodec_n2_5min_v1';out.mkdir(parents=True,exist_ok=True)
 for i,r in enumerate(rows):
  x=torch.tensor([r['linguistic_ids']]);frames=int(np.load(ROOT/r['codec_token_path']).shape[0]);ids=m.generate(x,frames)[0].numpy()
  with torch.inference_mode():y=codec.decode_code(torch.tensor(ids.reshape(1,1,-1))).cpu().numpy().reshape(-1)
  p=out/f'{i+1:02d}_{r["utterance_id"]}_generated.wav';sf.write(p,y,24000,subtype='PCM_16');print(p,float(np.sqrt(np.mean(y*y))),float(np.max(np.abs(y))))
  if i>=1: (out/'listening').mkdir(exist_ok=True);sf.write(out/'listening'/f'{r["utterance_id"]}.wav',y,24000,subtype='PCM_16');(out/'listening'/f'{r["utterance_id"]}.txt').write_text(r.get('training_text',r.get('source_text',r['utterance_id'])))
if __name__=='__main__':main()
