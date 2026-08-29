import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from swara.diagnostics.continuous_targets import (
    channel_statistics,
    deterministic_order,
    deterministic_seed,
    ensure_output_path,
    official_fsq_from_projected,
    perturb_representation,
)


class _FakeLayer(torch.nn.Module):
    def bound(self, x):
        return x

    def forward(self, x):
        coordinates = torch.round(x).clamp(-2, 1) / 2
        indices = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
        return coordinates, indices


class _FakeQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_quantizers = 1
        self.project_in = torch.nn.Identity()
        self.project_out = torch.nn.Identity()
        self.layers = torch.nn.ModuleList([_FakeLayer()])

    def forward(self, x):
        projected = self.project_in(x)
        emb, ids, _ = official_fsq_from_projected(self, projected)
        return emb, ids.unsqueeze(-1)


class ContinuousTargetTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.arange(60, dtype=np.float32).reshape(20, 3) / 10
        self.std = np.array([0.5, 1.0, 2.0], dtype=np.float64)

    def test_deterministic_panel_order(self):
        rows = [{"utterance_id": x} for x in ("c", "a", "b")]
        self.assertEqual(deterministic_order(rows, namespace="panel"), deterministic_order(rows, namespace="panel"))
        self.assertEqual(deterministic_seed("a"), deterministic_seed("a"))

    def test_frozen_panel_identity(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "experiments/swara_continuous_target_bakeoff_v1/panel.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["seed"], 20260823)
        self.assertTrue(payload["frozen_before_analysis"])
        self.assertEqual(len(payload["items"]), 20)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "26a727a6287c9ad88a36bb9ae66fd937d4090ded2269d93dfadef72094e95f97",
        )

    def test_no_perturbation_at_zero(self):
        result = perturb_representation(self.clean, self.std, sigma=0, seed=1, family="iid")
        self.assertTrue(np.array_equal(result, self.clean))
        self.assertIsNot(result, self.clean)

    def test_deterministic_channel_scaled_iid(self):
        a = perturb_representation(self.clean, self.std, sigma=.1, seed=4, family="iid")
        b = perturb_representation(self.clean, self.std, sigma=.1, seed=4, family="iid")
        self.assertTrue(np.array_equal(a, b))
        normalized = (a - self.clean) / self.std[None, :]
        self.assertGreater(float(normalized.std()), .04)

    def test_smooth_noise_rms_matches_iid_draw(self):
        iid = perturb_representation(self.clean, self.std, sigma=.1, seed=9, family="iid") - self.clean
        smooth = perturb_representation(self.clean, self.std, sigma=.1, seed=9, family="smooth") - self.clean
        self.assertTrue(np.allclose(np.sqrt(np.mean(iid ** 2, axis=0)), np.sqrt(np.mean(smooth ** 2, axis=0)), rtol=2e-5))
        self.assertLess(float(np.mean(np.abs(np.diff(smooth, axis=0)))), float(np.mean(np.abs(np.diff(iid, axis=0)))))

    def test_channel_statistics_orientation(self):
        stats = channel_statistics([self.clean[:10], self.clean[10:]])
        self.assertEqual(stats.channels, 3)
        self.assertEqual(stats.frame_count, 20)

    def test_target_b_clean_equivalence_helper(self):
        q = _FakeQuantizer()
        x = torch.randn(2, 5, 8)
        standard_emb, standard_ids = q(x)
        extracted = q.project_in(x)
        rebuilt_emb, rebuilt_ids, _ = official_fsq_from_projected(q, extracted)
        self.assertTrue(torch.equal(standard_ids[..., 0], rebuilt_ids))
        self.assertTrue(torch.equal(standard_emb, rebuilt_emb))

    def test_actual_target_b_and_c_equivalence_evidence(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "experiments/swara_speech_poc_v1/reports/continuous_target_bakeoff_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        target_b = payload["targets"]["target_b_prefsq"]
        target_c = payload["targets"]["target_c_decoder_latent"]
        self.assertEqual(len(target_b["items"]), 20)
        self.assertTrue(target_b["clean_exact_id_equivalence"])
        self.assertEqual(target_b["clean_waveform_equivalence"]["max_absolute_difference"], 0.0)
        self.assertEqual(len(target_c["items"]), 20)
        self.assertEqual(target_c["clean_waveform_equivalence"]["max_absolute_difference"], 0.0)
        self.assertEqual(target_c["clean_waveform_equivalence"]["mean_absolute_difference"], 0.0)

    def test_output_path_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = ensure_output_path(root, "target_a_mel", "clean", 0, "u1")
            noisy = ensure_output_path(root, "target_a_mel", "iid", .05, "u1")
            self.assertEqual(clean.relative_to(root).as_posix(), "target_a_mel/clean/u1.wav")
            self.assertEqual(noisy.relative_to(root).as_posix(), "target_a_mel/iid/sigma_005/u1.wav")


if __name__ == "__main__":
    unittest.main()
