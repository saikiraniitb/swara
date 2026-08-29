#!/usr/bin/env python3
"""D3 Colab launcher: run one frozen nested rung with the D2 formulation.

D3's Target-C source is deliberately the frozen canonical codec-ID cache, not
a fresh cross-platform WAV encoding.  A fresh encode is sensitive to harmless
FSQ boundary variation (documented for ENTE_4277), while the cache is the
provenance-approved acoustic target.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swara_d2_phoneme_ablation as d2
import run_swara_c1_decoder_latent as c1
import run_swara_c0_decoder_latent as c0
from swara.training.speech_poc_dataset import load_duration_supervision, select_examples


@torch.inference_mode()
def target_c_from_canonical_ids(codec, cached_ids: torch.Tensor) -> torch.Tensor:
    """Apply the official frozen NeuCodec ID-to-Target-C path for D3."""

    if cached_ids.ndim != 1:
        raise ValueError("D3 canonical IDs must be a one-dimensional [T] tensor")
    if not torch.all((cached_ids >= 0) & (cached_ids < 65_536)):
        raise ValueError("D3 canonical IDs must be in [0, 65535]")
    # Official NeuCodec inverse-FSQ / project-out route.  The shape is [B,T,Q]
    # with the one official residual FSQ quantizer in the final dimension.
    indices = cached_ids.to(codec.device)[None, :, None]
    decoder_embedding = codec.generator.quantizer.get_output_from_indices(indices)
    return codec.fc_post_a(decoder_embedding)[0].detach().cpu().float()


@torch.inference_mode()
def canonical_cached_target_c(codec, example) -> dict[str, Any]:
    """Build D3 Target-C from the canonical cached NeuCodec IDs only.

    This is the exact frozen decoder-side path used by
    ``DistillNeuCodec.decode_code`` before it invokes the decoder:

    ``cached IDs -> quantizer.get_output_from_indices -> fc_post_a``.

    It intentionally does not read or encode the source WAV.  WAV hashes are
    checked by the D3 provenance gate, but WAV content cannot replace the
    canonical discrete cache as the training target source.
    """

    path = c0.token_path(example)
    if not path.is_file():
        raise FileNotFoundError(f"{example.utterance_id}: canonical codec cache missing: {path}")
    cached_ids = torch.from_numpy(np.load(path, allow_pickle=False)).long().reshape(-1)
    if cached_ids.numel() != example.target_total_frames:
        raise RuntimeError(
            f"{example.utterance_id}: cached IDs {cached_ids.numel()} != "
            f"GT expansion {example.target_total_frames}"
        )
    if not torch.all((cached_ids >= 0) & (cached_ids < 65_536)):
        raise RuntimeError(f"{example.utterance_id}: canonical cached IDs outside [0, 65535]")

    target_latent = target_c_from_canonical_ids(codec, cached_ids)
    expected_shape = (example.target_total_frames, 1024)
    if tuple(target_latent.shape) != expected_shape:
        raise RuntimeError(
            f"{example.utterance_id}: reconstructed Target-C {tuple(target_latent.shape)} != "
            f"{expected_shape}"
        )
    if not torch.isfinite(target_latent).all():
        raise RuntimeError(f"{example.utterance_id}: reconstructed Target-C is non-finite")
    return {"target": target_latent, "standard_ids": cached_ids}


def extract_canonical_cached_targets(codec, examples: Sequence) -> dict[str, dict[str, Any]]:
    """D3-only batch target provider; never fresh-encodes source WAVs."""

    return {example.utterance_id: canonical_cached_target_c(codec, example) for example in examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, choices=(32, 64, 128, 267), required=True)
    ap.add_argument("--drive-root", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--microbatch-rows", type=int, default=None)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    # Rung 267 must never enter D2's full-rung autograd graph.  The only
    # executable route is the D3 microbatch runner below; D2's legacy main is
    # retained solely for the already-completed smaller historical rungs.
    if args.rung == 267:
        from run_swara_d3_microbatch import run_rung267

        drive_root = args.drive_root if args.drive_root is not None else ROOT / "runs/d3_data_scaling"
        run_rung267(
            drive_root=drive_root,
            requested_microbatch_rows=args.microbatch_rows,
            resume=args.resume,
            smoke_only=args.smoke_only,
        )
        return

    manifest = ROOT / "experiments/swara_speech_poc_v1/reports/d3_rungs" / f"{args.rung}.json"
    cfg = json.loads(manifest.read_text())
    all_train = load_duration_supervision(
        ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl", split="train"
    )
    all_val = load_duration_supervision(
        ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl", split="val"
    )
    train = select_examples(all_train, cfg["train_ids"])
    val = select_examples(all_val, cfg["validation_ids"])

    # Process-local override only: D2/C1 historical source files and results
    # remain unchanged.  D2's unchanged training orchestration calls this D3
    # provider when it builds the Target-C batch.
    d2.c1.frozen_p2_split = lambda: (train, val)
    d2.c1.extract_targets = extract_canonical_cached_targets

    root = (args.drive_root / str(args.rung)) if args.drive_root else ROOT / f"runs/d3_data_scaling/{args.rung}"
    root = Path(root)
    d2.RUN_ROOT = root / "run"
    d2.CHECKPOINT = d2.RUN_ROOT / "best.pt"
    d2.EVAL_ROOT = root / "evaluations"
    d2.REPORT = root / "d3_metrics.json"
    d2.DOC = root / "D3_RESULT.md"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_state.json").write_text(
        json.dumps(
            {
                "rung": args.rung,
                "train_rows": len(train),
                "validation_rows": len(val),
                "status": "started",
                "max_steps": 500,
                "target_source": "canonical_cached_neucodec_ids_v1",
                "fresh_wav_encode_required_for_training": False,
            },
            indent=2,
        )
        + "\n"
    )
    if args.resume:
        os.environ["D3_RESUME"] = "1"
    d2.main()
    (root / "run_state.json").write_text(
        json.dumps(
            {
                "rung": args.rung,
                "train_rows": len(train),
                "validation_rows": len(val),
                "status": "complete",
                "max_steps": 500,
                "target_source": "canonical_cached_neucodec_ids_v1",
                "fresh_wav_encode_required_for_training": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
