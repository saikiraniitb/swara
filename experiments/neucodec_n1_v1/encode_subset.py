from __future__ import annotations
import json, sys, types
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
ROOT=Path(__file__).parents[2]
def shim():
 p=ROOT/'.venv/lib/python3.14/site-packages/torchtune/modules/position_embeddings.py'; text=p.read_text(); ns={'torch':torch,'nn':torch.nn,'Any':object,'Optional':object}; exec(text[text.index('class RotaryPositionalEmbeddings'):],ns); m=types.ModuleType('torchtune.modules'); m.RotaryPositionalEmbeddings=ns['RotaryPositionalEmbeddings']; sys.modules['torchtune.modules']=m
def rows(name): return [json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data'/name).read_text().splitlines() if x.strip()]
def main():
 from swara.contracts import build_plain_text_request
 from swara.frontend import Frontend
 from swara.models.linguistic import LinguisticVocabulary
 allrows=rows('train_manifest.jsonl')+rows('val_manifest.jsonl'); seqs=tuple(Frontend().compile(build_plain_text_request(r['training_text'],default_language='en-IN',speaker_id='ENG_M_SPK001')) for r in allrows); vocab=LinguisticVocabulary.build(seqs); cfg=ROOT/'experiments/neucodec_n1_v1/configs'; cfg.mkdir(parents=True,exist_ok=True); vocab.save(cfg/'linguistic_vocab.json')
 shim(); from neucodec import DistillNeuCodec
 model=DistillNeuCodec.from_pretrained('neuphonic/distill-neucodec',revision='daee7fd9989a62594084fd8e1a99e61beb5b0e85').eval()
 tokdir=ROOT/'experiments/neucodec_n1_v1/tokens'; tokdir.mkdir(exist_ok=True)
 for r,s in zip(allrows,seqs):
  ids=list(vocab.encode(s).ids); r['linguistic_ids']=ids; wav,sr=sf.read(ROOT/r['source_wav'],dtype='float32',always_2d=False); wav=np.asarray(wav).reshape(-1); x=torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0),int(sr),16000).unsqueeze(0)
  with torch.inference_mode(): c=model.encode_code(x).detach().cpu().numpy().astype(np.int64).reshape(-1)
  np.save(tokdir/(r['utterance_id']+'.npy'),c); r.update({'codec_token_path':f'experiments/neucodec_n1_v1/tokens/{r["utterance_id"]}.npy','codec_frames':int(c.size),'codec_frame_rate_hz':float(c.size/r['duration_seconds']),'codec_token_min':int(c.min()),'codec_token_max':int(c.max()),'codec_token_unique':int(np.unique(c).size)})
 for name,split in [('train_manifest.jsonl',allrows[:len(rows('train_manifest.jsonl'))]),('val_manifest.jsonl',allrows[len(rows('train_manifest.jsonl')):])]: (ROOT/'experiments/neucodec_n1_v1/data'/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in split))
 print('vocab',vocab.size,'encoded',len(allrows))
if __name__=='__main__': main()
