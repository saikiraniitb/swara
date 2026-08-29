"""Lightweight device-ownership tests for the Stage2B.4B loss path."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swara.training.stage2b_pronunciation import (
    compute_qwen_split_preservation_kl,
    compute_qwen_split_target_ce,
    masked_codebook_cross_entropy,
    masked_logits_kl,
    residual_native_norm_diagnostic,
    build_stage2b_frame_masks,
)


def available_devices() -> tuple[torch.device, ...]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return tuple(devices)


class Stage2B4BCudaPortabilityTests(unittest.TestCase):
    def test_loss_helpers_move_cpu_targets_and_masks_to_logits_device(self):
        for device in available_devices():
            main = torch.randn(1, 3, 13, device=device)
            residual = torch.randn(1, 3, 3, 7, device=device)
            native_main = torch.randn(1, 3, 13, device=device)
            native_residual = torch.randn(1, 3, 3, 7, device=device)
            codes = torch.randint(0, 7, (1, 3, 4), device="cpu")
            mask = torch.tensor([[True, True, False]], device="cpu")

            target = compute_qwen_split_target_ce(main, residual, codes, mask)
            preserve = compute_qwen_split_preservation_kl(
                main, native_main, residual, native_residual, mask
            )
            generic_target = masked_codebook_cross_entropy(
                residual, codes[:, :, 1:], mask, codebooks=(0, 1, 2)
            )
            generic_kl = masked_logits_kl(
                residual, native_residual, mask
            )

            for value in (target, preserve, generic_target, generic_kl):
                self.assertEqual(value.device.type, device.type)
                if device.type == "cuda":
                    self.assertEqual(value.device.index, device.index or torch.cuda.current_device())
                self.assertTrue(torch.isfinite(value))

    def test_residual_norm_diagnostic_accepts_cpu_masks_for_compute_states(self):
        for device in available_devices():
            masks = build_stage2b_frame_masks(
                batch_size=1,
                total_frames=3,
                target_ranges=(((0, 1),),),
            )
            native = torch.ones(1, 3, 4, device=device)
            residual = torch.full((1, 3, 4), 0.01, device=device)
            result = residual_native_norm_diagnostic(native, residual, masks)
            self.assertGreater(result.ratio_target, 0.0)
            self.assertGreater(result.ratio_non_target, 0.0)

    def test_runner_split_ce_diagnostics_accepts_cpu_targets_for_compute_logits(self):
        from run_stage2b4b_pronunciation import split_ce_diagnostics

        for device in available_devices():
            main = torch.randn(1, 3, 13, device=device)
            residual = torch.randn(1, 3, 3, 7, device=device)
            codes = torch.randint(0, 7, (1, 3, 4), device="cpu")
            mask = torch.tensor([[True, True, False]], device="cpu")
            result = split_ce_diagnostics(main, residual, codes, mask)
            self.assertEqual(set(result), {"q0_ce", "q1_ce", "q2_ce", "q3_ce"})
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
