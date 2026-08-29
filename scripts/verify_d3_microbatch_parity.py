#!/usr/bin/env python3
"""One-step, no-artifact parity audit for D3's exact-loss microbatch path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_swara_c0_decoder_latent as c0  # noqa: E402
import run_swara_c1_decoder_latent as c1  # noqa: E402
import run_swara_d2_phoneme_ablation as d2  # noqa: E402
from run_continuous_target_bakeoff import load_neucodec  # noqa: E402
from run_swara_d3_data_scaling import extract_canonical_cached_targets  # noqa: E402
from run_swara_d3_microbatch import logical_microbatch_step  # noqa: E402
from swara.models.c0_decoder_latent import normalized_decoder_latent_loss  # noqa: E402
from swara.models.d2_phoneme_ablation import SwaraD2PhonemeModel  # noqa: E402
from swara.models.phoneme_composer import PhonemeComposerVocabulary  # noqa: E402
from swara.training.speech_poc_dataset import load_duration_supervision, select_examples  # noqa: E402


MANIFEST = ROOT / "experiments/swara_speech_poc_v1/reports/d3_rungs/32.json"
ALIGNMENT = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
MICROBATCH_ROWS = 16


def max_gradient_difference(left, right) -> float:
    values = []
    for a, b in zip(left.parameters(), right.parameters()):
        if a.grad is None or b.grad is None:
            if a.grad is not b.grad:
                raise RuntimeError("gradient presence differs")
            continue
        values.append(float((a.grad - b.grad).abs().max()))
    return max(values, default=0.0)


def max_parameter_difference(left, right) -> float:
    return max(float((a - b).abs().max().detach()) for a, b in zip(left.parameters(), right.parameters()))


def main() -> None:
    config = json.loads(MANIFEST.read_text())
    all_train = load_duration_supervision(ALIGNMENT, split="train")
    all_val = load_duration_supervision(ALIGNMENT, split="val")
    train = select_examples(all_train, config["train_ids"])
    validation = select_examples(all_val, config["validation_ids"])
    if len(train) != 32 or len(validation) != 8:
        raise RuntimeError("D3 rung-32 parity requires frozen 32/8 membership")

    # Eval mode intentionally disables stochastic dropout, isolating the exact
    # full-rung denominator/gradient algebra rather than RNG-mask ordering.
    d2.c0.seed_everything()
    mapping, audit = d2.coverage(list(train) + list(validation))
    if audit["failure_count"] or audit["empty_outputs"]:
        raise RuntimeError("phonemizer gate failed")
    vocabulary = PhonemeComposerVocabulary.from_sequences(tuple(x.sequence for x in list(train) + list(validation)), mapping)
    seed_model = SwaraD2PhonemeModel(vocabulary, mapping).cpu().eval()
    initial_state = seed_model.state_dict()
    full_model = SwaraD2PhonemeModel(vocabulary, mapping).cpu().eval()
    micro_model = SwaraD2PhonemeModel(vocabulary, mapping).cpu().eval()
    full_model.load_state_dict(initial_state)
    micro_model.load_state_dict(initial_state)
    full_optimizer, micro_optimizer = c0.optimizer_for(full_model), c0.optimizer_for(micro_model)

    codec = load_neucodec()
    targets = extract_canonical_cached_targets(codec, train)
    stats = np.load(ROOT / "runs/swara_c1_decoder_latent_v1/target_normalization.npz")
    mean, std = torch.from_numpy(stats["mean"]), torch.from_numpy(stats["std"])

    full_target, full_norm, full_padding = c1.build_target_batch(train, targets, mean, std, torch.device("cpu"))
    del full_target
    full_optimizer.zero_grad(set_to_none=True)
    prediction, aligned = full_model(
        [x.sequence for x in train], [x.alignment_units for x in train], [x.target_total_frames for x in train]
    )
    if not torch.equal(aligned.padding_mask, full_padding):
        raise RuntimeError("full-rung frame mask mismatch")
    full_loss = normalized_decoder_latent_loss(prediction, full_norm, aligned.padding_mask).total
    full_loss.backward()
    torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
    full_optimizer.step()

    micro_loss, _, maximum_autograd_rows = logical_microbatch_step(
        micro_model, micro_optimizer, train, targets, mean, std, torch.device("cpu"), MICROBATCH_ROWS
    )
    gradient_difference = max_gradient_difference(full_model, micro_model)
    parameter_difference = max_parameter_difference(full_model, micro_model)
    loss_difference = abs(float(full_loss.detach()) - micro_loss)
    loss_gradient_tolerance = 2e-6
    # Adam's post-update values amplify order-level gradient differences by a
    # few ulps; this remains a tight ~1e-5 absolute parameter tolerance.
    parameter_tolerance = 1e-5
    passed = (
        loss_difference <= loss_gradient_tolerance
        and gradient_difference <= loss_gradient_tolerance
        and parameter_difference <= parameter_tolerance
    )
    result = {
        "D3_MICROBATCH_RUNG32_PARITY": "PASS" if passed else "FAIL",
        "mode": "eval (dropout disabled only for deterministic algebra audit)",
        "microbatch_rows": MICROBATCH_ROWS,
        "pre_update_full_loss": float(full_loss.detach()),
        "pre_update_microbatch_loss": micro_loss,
        "pre_update_loss_difference": loss_difference,
        "max_gradient_difference": gradient_difference,
        "max_post_update_parameter_difference": parameter_difference,
        "max_autograd_forward_batch": maximum_autograd_rows,
        "loss_gradient_tolerance": loss_gradient_tolerance,
        "post_update_parameter_tolerance": parameter_tolerance,
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("D3 rung-32 microbatch parity failed")


if __name__ == "__main__":
    main()
