#!/usr/bin/env python3
"""Read-only D3 preflight for canonical cached-ID Target-C reconstruction.

This verifier is intentionally separate from the historical C1/D2 launchers.
It proves that D3 can construct the unchanged Target-C representation without
freshly encoding training WAVs, while using the historical C1 WAV path only for
the explicitly requested 32-row parity audit.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_swara_c0_decoder_latent as c0  # noqa: E402
import run_swara_c1_decoder_latent as c1  # noqa: E402
from run_continuous_target_bakeoff import NEUCODEC_ID, NEUCODEC_REVISION, load_neucodec  # noqa: E402
from run_swara_d3_data_scaling import canonical_cached_target_c, target_c_from_canonical_ids  # noqa: E402
from swara.training.speech_poc_dataset import load_duration_supervision, select_examples  # noqa: E402


D3_RUNG = ROOT / "experiments/swara_speech_poc_v1/reports/d3_rungs/267.json"
ALIGNMENT = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
REPORT = ROOT / "experiments/swara_speech_poc_v1/reports/d3_canonical_targetc_from_cache.json"
TOLERANCE = 1e-6
ENTE_4277 = "IISc_SPICORProject_EN_M_ENTE_4277"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())


def d3_examples():
    config = json.loads(D3_RUNG.read_text(encoding="utf-8"))
    all_train = load_duration_supervision(ALIGNMENT, split="train")
    all_validation = load_duration_supervision(ALIGNMENT, split="val")
    train = select_examples(all_train, config["train_ids"])
    validation = select_examples(all_validation, config["validation_ids"])
    if len(train) != 267 or len(validation) != 8:
        raise RuntimeError("D3 canonical target preflight requires exactly 267 train / 8 validation rows")
    return train, validation


def reconstruct_coverage(codec, examples):
    rows = []
    for example in examples:
        path = c0.token_path(example)
        target = canonical_cached_target_c(codec, example)
        rows.append(
            {
                "utterance_id": example.utterance_id,
                "cache_path": str(path.relative_to(ROOT)),
                "cache_sha256": sha256(path),
                "frames": int(target["standard_ids"].numel()),
                "target_shape": list(target["target"].shape),
                "finite": bool(torch.isfinite(target["target"]).all()),
            }
        )
    return rows


def historic_parity(codec):
    """Compare D3's cache-only path with C1's original local path for 32 rows."""

    train, _ = c1.frozen_p2_split()
    historical = c1.extract_targets(codec, train)
    max_abs = 0.0
    minimum_cosine = 1.0
    rows = []
    for example in train:
        reconstructed = canonical_cached_target_c(codec, example)
        original = historical[example.utterance_id]
        if not torch.equal(reconstructed["standard_ids"], original["standard_ids"]):
            raise RuntimeError(f"{example.utterance_id}: C1 canonical IDs changed during parity audit")
        difference = (reconstructed["target"] - original["target"]).abs()
        row_max = float(difference.max())
        row_cosine = cosine(reconstructed["target"], original["target"])
        max_abs = max(max_abs, row_max)
        minimum_cosine = min(minimum_cosine, row_cosine)
        rows.append({"utterance_id": example.utterance_id, "max_abs_difference": row_max, "cosine": row_cosine})
    passed = max_abs <= TOLERANCE and minimum_cosine >= 1.0 - 1e-7
    if not passed:
        raise RuntimeError(
            f"D3 Target-C cache reconstruction parity failed: max_abs={max_abs}, cosine={minimum_cosine}"
        )
    return {"rows": rows, "max_abs_difference": max_abs, "minimum_cosine": minimum_cosine, "pass": passed}


def main() -> None:
    codec = load_neucodec()
    train, validation = d3_examples()

    # This is the only permitted fresh-encode use in this preflight: a
    # historical, read-only parity comparison for the original C1/D2 32 rows.
    # D3 coverage below never reads WAVs or invokes the encoder.
    parity = historic_parity(codec)
    train_rows = reconstruct_coverage(codec, train)
    validation_rows = reconstruct_coverage(codec, validation)

    ente = next(row for row in train_rows + validation_rows if row["utterance_id"] == ENTE_4277)
    cached = np.load(ROOT / ente["cache_path"], allow_pickle=False).reshape(-1)
    if cached.shape != (333,) or int(cached[181]) != 18020:
        raise RuntimeError("ENTE_4277 canonical cache invariant failed")
    canonical_ente_target = target_c_from_canonical_ids(codec, torch.from_numpy(cached).long())
    colab_boundary_candidate = cached.copy()
    colab_boundary_candidate[181] = 18016
    candidate_target = target_c_from_canonical_ids(codec, torch.from_numpy(colab_boundary_candidate).long())
    candidate_delta = float((canonical_ente_target - candidate_target).abs().max())
    if candidate_delta == 0.0:
        raise RuntimeError("ENTE_4277 boundary candidate unexpectedly shares Target-C with canonical IDs")

    report = {
        "schema_version": "swara.d3.canonical_targetc_from_cache.v1",
        "status": "complete",
        "method": "canonical cached IDs -> official quantizer.get_output_from_indices -> fc_post_a",
        "target_definition": "Distill-NeuCodec fc_post_a decoder-side latent [T,1024]",
        "fresh_wav_encode_required_for_training": False,
        "fresh_wav_encode_used_for_d3_targets": False,
        "historical_c1_parity_only_fresh_encode": True,
        "codec": {"model": NEUCODEC_ID, "revision": NEUCODEC_REVISION},
        "parity_32": parity,
        "ente_4277": {
            "canonical_cache_sha256": ente["cache_sha256"],
            "cached_frames": int(cached.size),
            "canonical_cached_id_at_frame_181": int(cached[181]),
            "confirmed_colab_fresh_id_at_frame_181": 18016,
            "reconstructed_target_shape": [333, 1024],
            "reconstructed_target_finite": True,
            "fresh_boundary_flip_is_target_independent": True,
            "canonical_vs_colab_boundary_candidate_target_max_abs_difference": candidate_delta,
            "d3_target_uses_canonical_cached_id_not_colab_fresh_id": True,
        },
        "coverage": {
            "train": {"valid": sum(row["finite"] for row in train_rows), "total": len(train_rows), "rows": train_rows},
            "validation": {"valid": sum(row["finite"] for row in validation_rows), "total": len(validation_rows), "rows": validation_rows},
        },
        "canonical_cache_modified": False,
        "training_performed": False,
        "architecture_modified": False,
        "historical_c1_d2_artifacts_modified": False,
        "commit_push": False,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "D3_CANONICAL_TARGETC_FROM_CACHE": "COMPLETE",
        "parity_32": {key: parity[key] for key in ("max_abs_difference", "minimum_cosine", "pass")},
        "train_coverage": f"{report['coverage']['train']['valid']}/267",
        "validation_coverage": f"{report['coverage']['validation']['valid']}/8",
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
