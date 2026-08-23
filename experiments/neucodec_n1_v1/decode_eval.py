from __future__ import annotations
import json, sys, types
from pathlib import Path
import numpy as np, soundfile as sf, torch
ROOT=Path(__file__).parents[2]
def shim():
 p=ROOT/'.venv/lib/python3.14/site-packages/torchtune/modules/position_embeddings.py'; text=p.read_text(); ns={'torch':torch,'nn':torch.nn,'Any':object,'Optional':object}; exec(text[text.index('class RotaryPositionalEmbeddings'):],ns); m=types.ModuleType('torchtune.modules'); m.RotaryPositionalEmbeddings=ns['RotaryPositionalEmbeddings']; sys.modules['torchtune.modules']=m
def main():
 shim(); from neucodec import DistillNeuCodec
 from swara.models.neucodec_n1 import N1Config, N1Flat, N1FSQ
 model=DistillNeuCodec.from_pretrained('neuphonic/distill-neucodec',revision='daee7fd9989a62594084fd8e1a99e61beb5b0e85').eval()
 train_rows=[json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data/train_manifest.jsonl').read_text().splitlines() if x.strip()][:2]
 val_rows=[json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data/val_manifest.jsonl').read_text().splitlines() if x.strip()][:4]
 rows=train_rows+val_rows
 report={}
 for k in ('a','b'):
  cfg=N1Config(502,d_model=128,layers=4,heads=4,ffn_dim=512,max_frames=4096)
  cls=N1Flat if k=='a' else N1FSQ
  m=cls(cfg); ck=torch.load(ROOT/f'runs/neucodec_n1_v1/n1_1_{k.upper()}/best.pt',weights_only=False); m.load_state_dict(ck['model']); m.eval()
  out=ROOT/f'evaluations/neucodec_n1/{k.upper()}'; out.mkdir(parents=True,exist_ok=True); report[k]={}
  for i,row in enumerate(rows):
   x=torch.tensor([row['linguistic_ids']],dtype=torch.long); frames=int(np.load(ROOT/row['codec_token_path']).shape[0])
   with torch.inference_mode(): ids=m.generate(x,frames)[0].cpu().numpy()
   c=torch.tensor(np.asarray(ids,dtype=np.int64).reshape(1,1,-1))
   with torch.inference_mode(): y=model.decode_code(c).cpu().numpy().reshape(-1)
   path=out/f'{i+1:02d}_{row["utterance_id"]}_generated.wav'; sf.write(path,y,24000,subtype='PCM_16')
   report[k][row['utterance_id']]={'path':str(path.relative_to(ROOT)),'frames':frames,'samples':int(y.size),'duration_seconds':float(y.size/24000),'rms':float(np.sqrt(np.mean(y*y))) if y.size else 0.0,'peak':float(np.max(np.abs(y))) if y.size else 0.0,'finite':bool(np.isfinite(y).all()),'non_silent':bool(np.sqrt(np.mean(y*y))>1e-5) if y.size else False}
 (ROOT/'evaluations/neucodec_n1/decode_metrics.json').write_text(json.dumps(report,indent=2))
 print('decoded 2 train + 4 validation per model')
if __name__=='__main__': main()
