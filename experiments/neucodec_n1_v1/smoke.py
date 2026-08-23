from __future__ import annotations
import json
from pathlib import Path
import torch
from swara.codecs.neucodec_fsq import CARDINALITY, DIMENSIONS, fsq_to_token_ids, token_ids_to_fsq
from swara.models.neucodec_n1 import N1Config, N1Flat, N1FSQ, parameter_counts

ROOT=Path(__file__).parents[2]
def main():
    torch.manual_seed(20260823)
    ids=list(range(CARDINALITY)); coords=token_ids_to_fsq(torch.tensor(ids)); assert torch.equal(fsq_to_token_ids(coords),torch.tensor(ids)); assert coords.min()==0 and coords.max()==3 and torch.unique(coords,dim=0).shape[0]==CARDINALITY
    train=[json.loads(x) for x in (ROOT/'experiments/neucodec_n1_v1/data/train_manifest.jsonl').read_text().splitlines() if x.strip()]; r=train[0]
    ling=torch.tensor([r['linguistic_ids']],dtype=torch.long); target=torch.tensor([__import__('numpy').load(ROOT/r['codec_token_path'])],dtype=torch.long)
    cfg=N1Config(linguistic_vocab_size=502); a=N1Flat(cfg); b=N1FSQ(cfg)
    assert parameter_counts(a)[0]==parameter_counts(b)[0]
    la,lossa,_=a(ling,target); fsq_target=token_ids_to_fsq(target); lb,lossb,_=b(ling,fsq_target)
    lossa.backward(); lossb.backward(); ga=a.generate(ling,target.shape[1]); gb=b.generate(ling,target.shape[1]); assert ga.shape==target.shape and gb.shape==target.shape and ga.min()>=0 and ga.max()<CARDINALITY and gb.min()>=0 and gb.max()<CARDINALITY
    ac,ah,at=parameter_counts(a); bc,bh,bt=parameter_counts(b)
    report={'fsq_bijection':'PASS','n1_a':{'backbone_params':ac,'head_params':ah,'total_params':at,'loss':float(lossa.detach()),'token_accuracy':float((ga==target).float().mean())},'n1_b':{'backbone_params':bc,'head_params':bh,'total_params':bt,'loss':float(lossb.detach()),'per_dimension_accuracy':[float((lb[:,:,i].argmax(-1)==fsq_target[:,:,i]).float().mean()) for i in range(DIMENSIONS)],'exact_full_token_accuracy':float((gb==target).float().mean())},'head_parameter_ratio':ah/bh}
    (ROOT/'experiments/neucodec_n1_v1/reports').mkdir(exist_ok=True); (ROOT/'experiments/neucodec_n1_v1/reports/smoke.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
