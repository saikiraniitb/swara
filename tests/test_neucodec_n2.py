import unittest, torch
from swara.models.neucodec_n2 import N2Config,N2Model,parameter_count

class N2Tests(unittest.TestCase):
    def test_shapes_and_special_ids(self):
        m=N2Model(N2Config(32,d_model=32,heads=4,ffn_dim=64,max_frames=16)); x=torch.randint(0,32,(1,5)); y=torch.randint(0,65536,(1,7)); logits,loss,_=m(x,torch.cat([torch.full((1,1),65536),y[:,:-1]],1),y); self.assertEqual(logits.shape,(1,7,65536)); self.assertTrue(torch.isfinite(loss))
    def test_generation_range(self):
        m=N2Model(N2Config(32,d_model=32,heads=4,ffn_dim=64,max_frames=8)); z=m.generate(torch.randint(0,32,(1,4)),4); self.assertTrue(bool(((z>=0)&(z<65536)).all()))
    def test_backward(self):
        m=N2Model(N2Config(32,d_model=32,heads=4,ffn_dim=64,max_frames=8)); x=torch.randint(0,32,(1,4)); y=torch.randint(0,65536,(1,4)); _,l,_=m(x,torch.cat([torch.full((1,1),65536),y[:,:-1]],1),y); l.backward(); self.assertTrue(torch.isfinite(l))
