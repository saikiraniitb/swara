import unittest
import torch

from swara.codecs.neucodec_fsq import CARDINALITY, fsq_to_token_ids, token_ids_to_fsq
from swara.models.neucodec_n1 import N1Config, N1Flat, N1FSQ, parameter_counts


class NeuCodecN1Tests(unittest.TestCase):
    def test_full_fsq_bijection(self):
        ids = torch.arange(CARDINALITY)
        coords = token_ids_to_fsq(ids)
        self.assertEqual(tuple(coords.shape), (CARDINALITY, 8))
        self.assertTrue(torch.equal(fsq_to_token_ids(coords), ids))
        self.assertEqual(torch.unique(coords, dim=0).shape[0], CARDINALITY)

    def test_heads_and_shapes(self):
        c = N1Config(32, d_model=32, layers=1, heads=4, ffn_dim=64, max_frames=64)
        x = torch.randint(0, 32, (2, 7)); y = torch.randint(0, 65536, (2, 11))
        a, la, _ = N1Flat(c)(x, y); b, lb, _ = N1FSQ(c)(x, token_ids_to_fsq(y))
        self.assertEqual(tuple(a.shape), (2, 11, 65536)); self.assertEqual(tuple(b.shape), (2, 11, 8, 4))
        self.assertTrue(torch.isfinite(la)); self.assertTrue(torch.isfinite(lb))

    def test_same_backbone(self):
        c = N1Config(32, d_model=32, layers=1, heads=4, ffn_dim=64)
        self.assertEqual(parameter_counts(N1Flat(c))[0], parameter_counts(N1FSQ(c))[0])

    def test_generation_ranges(self):
        c = N1Config(32, d_model=32, layers=1, heads=4, ffn_dim=64, max_frames=16)
        x = torch.randint(0, 32, (1, 5)); a = N1Flat(c).generate(x, 8); b = N1FSQ(c).generate(x, 8)
        self.assertEqual(tuple(a.shape), (1, 8)); self.assertEqual(tuple(b.shape), (1, 8))
        self.assertGreaterEqual(int(a.min()), 0); self.assertLess(int(a.max()), 65536)
        self.assertGreaterEqual(int(b.min()), 0); self.assertLess(int(b.max()), 65536)
