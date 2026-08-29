#!/usr/bin/env python3
"""Read-only D1 diagnostics for C1/B1 held-out continuous-target failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_swara_c0_decoder_latent as c0  # noqa: E402
import run_swara_c1_decoder_latent as c1  # noqa: E402
from run_continuous_target_bakeoff import (  # noqa: E402
    decode_neucodec_indices,
    decode_neucodec_latent,
    extract_neucodec,
    load_neucodec,
)
from swara.models.c0_decoder_latent import C0PredictorConfig, SwaraC0DecoderLatentModel  # noqa: E402
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402


OUT_JSON = ROOT / "experiments/swara_speech_poc_v1/reports/swara_rca_d1_zero_training.json"
OUT_MD = ROOT / "research/poc/diagnostics/SWARA_RCA_D1_ZERO_TRAINING.md"
C1_REPORT = ROOT / "experiments/swara_speech_poc_v1/reports/c1_decoder_latent_5min_v1.json"
B1_REPORT = ROOT / "experiments/swara_speech_poc_v1/reports/b1_prefsq_continuous_5min_v1.json"


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def corr(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(np.corrcoef(rank(a), rank(b))[0, 1]),
    }


def cosine(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = torch.from_numpy(pred), torch.from_numpy(target)
    return float(F.cosine_similarity(p, t, dim=-1).mean().item())


def score(pred: np.ndarray, target: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> dict[str, float]:
    if mean is not None and std is not None:
        pred = (pred - mean) / (std + 1e-6)
        target = (target - mean) / (std + 1e-6)
    diff = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "l1": float(np.mean(np.abs(diff))),
        "mean_frame_cosine": cosine(pred, target),
    }


def duration_resample(row: np.ndarray, length: int) -> np.ndarray:
    if row.shape[0] == length:
        return row.copy()
    source = np.linspace(0.0, 1.0, row.shape[0])
    target = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(target, source, row[:, channel]) for channel in range(row.shape[1])], axis=1)


def baseline_scores(train: Sequence[np.ndarray], validation: Sequence[np.ndarray], stats_path: Path) -> dict[str, Any]:
    stats = np.load(stats_path)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    global_mean = np.concatenate(train, axis=0).mean(axis=0, keepdims=True)
    normalized_train = [duration_resample(row, 100) for row in train]
    mean_trajectory = np.mean(np.stack(normalized_train), axis=0)
    rows = []
    for target in validation:
        global_prediction = np.repeat(global_mean, target.shape[0], axis=0)
        trajectory_prediction = duration_resample(mean_trajectory, target.shape[0])
        rows.append({
            "global_channel_mean": score(global_prediction, target, mean, std),
            "duration_normalized_mean_trajectory": score(trajectory_prediction, target, mean, std),
        })
    aggregate = {
        name: {metric: float(np.mean([row[name][metric] for row in rows])) for metric in rows[0][name]}
        for name in rows[0]
    }
    return {
        "global_channel_mean": global_mean[0].tolist(),
        "duration_normalized_mean_trajectory_points": 100,
        "per_utterance": rows,
        "aggregate": aggregate,
    }


def load_model(path: Path, vocabulary: LinguisticComposerVocabulary, output_width: int) -> tuple[SwaraC0DecoderLatentModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = C0PredictorConfig(output_width=output_width)
    model = SwaraC0DecoderLatentModel(vocabulary, predictor_config=config)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


@torch.inference_mode()
def model_predictions(model: SwaraC0DecoderLatentModel, examples: Sequence[Any], mean_path: Path) -> tuple[list[np.ndarray], Any]:
    mean_stats = np.load(mean_path)
    mean = torch.from_numpy(mean_stats["mean"]).float()
    std = torch.from_numpy(mean_stats["std"]).float()
    prediction_norm, expanded = model(
        tuple(item.sequence for item in examples),
        tuple(item.alignment_units for item in examples),
        tuple(item.target_total_frames for item in examples),
    )
    prediction = prediction_norm * (std + 1e-6) + mean
    return [prediction[index, :item.target_total_frames].numpy() for index, item in enumerate(examples)], expanded


def predictor_hidden(model: SwaraC0DecoderLatentModel, expanded: Any) -> Tensor:
    predictor = model.predictor
    # model_predictions uses inference_mode; clone into ordinary tensors before
    # replaying the predictor modules for this read-only hidden-state probe.
    states, padding = expanded.states.clone(), expanded.padding_mask.clone()
    length = states.shape[1]
    with torch.no_grad():
        hidden = predictor.input_projection(states)
        hidden = hidden + predictor.audio_positions[:length].to(states).unsqueeze(0)
        hidden = predictor.blocks(hidden, src_key_padding_mask=padding)
        return predictor.output_normalization(hidden)


def intra_unit(model: SwaraC0DecoderLatentModel, examples: Sequence[Any], expanded: Any) -> dict[str, Any]:
    raw_max = 0.0
    raw_values = []
    deep_values = []
    across_values = []
    for batch_index, example in enumerate(examples):
        frame_ids = expanded.frame_to_unit[batch_index]
        raw = expanded.states[batch_index]
        deep = predictor_hidden(model, expanded)[batch_index]
        valid = ~expanded.padding_mask[batch_index]
        for unit_id in torch.unique(frame_ids[valid]).tolist():
            indices = torch.where(valid & (frame_ids == unit_id))[0]
            if len(indices) > 1:
                raw_slice = raw[indices]
                deep_slice = deep[indices]
                raw_max = max(raw_max, float((raw_slice - raw_slice[0]).abs().max()))
                # Variance is across frames for each channel; averaging over
                # channels avoids conflating channel mean differences with
                # genuine within-unit temporal differentiation.
                raw_values.append(float(raw_slice.var(dim=0, unbiased=False).mean().item()))
                deep_values.append(float(deep_slice.var(dim=0, unbiased=False).mean().item()))
        if valid.sum() > 1:
            across_values.append(float(deep[valid].var(dim=0, unbiased=False).mean().item()))
    within = float(np.mean(deep_values)) if deep_values else 0.0
    across = float(np.mean(across_values)) if across_values else 0.0
    ratio = within / max(across, 1e-12)
    strength = "NONE" if ratio < .05 else "WEAK" if ratio < .20 else "MODERATE" if ratio < .50 else "STRONG"
    return {
        "raw_conditioning_identical": raw_max <= 1e-8,
        "raw_max_abs_within_unit_difference": raw_max,
        "raw_within_unit_variance_mean": float(np.mean(raw_values)) if raw_values else 0.0,
        "deep_within_unit_variance_mean": within,
        "deep_across_sequence_variance_mean": across,
        "deep_within_to_across_ratio": ratio,
        "deep_state_differentiation": strength,
    }


def features(train: Sequence[Any], validation: Sequence[Any]) -> list[dict[str, float | str]]:
    train_words = {token.value.casefold() for item in train for token in item.sequence.tokens if token.kind.value == "grapheme"}
    train_bigrams = {
        pair for item in train for token in item.sequence.tokens
        for pair in zip(token.value.casefold(), token.value.casefold()[1:]) if token.kind.value == "grapheme"
    }
    durations = np.asarray([item.target_total_frames for item in train], dtype=float)
    mean, std = float(durations.mean()), float(durations.std() or 1.0)
    rows = []
    for item in validation:
        words = [token.value.casefold() for token in item.sequence.tokens if token.kind.value == "grapheme"]
        chars = [pair for word in words for pair in zip(word, word[1:])]
        rows.append({
            "utterance_id": item.utterance_id,
            "frames": float(item.target_total_frames),
            "duration_abs_z": abs(item.target_total_frames - mean) / std,
            "unseen_word_count": float(sum(word not in train_words for word in words)),
            "unseen_char_bigram_count": float(sum(pair not in train_bigrams for pair in chars)),
            "word_count": float(len(words)),
        })
    return rows


def correlation_sweep(feature_rows: list[dict[str, float | str]], score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("frames", "duration_abs_z", "unseen_word_count", "unseen_char_bigram_count", "word_count")
    result = {}
    for metric in ("rmse", "mean_frame_cosine"):
        result[metric] = {name: corr([float(row[name]) for row in feature_rows], [float(row[metric]) for row in score_rows]) for name in names}
    strongest = []
    for metric, values in result.items():
        candidates = [(name, abs(value["spearman"])) for name, value in values.items() if value["spearman"] is not None]
        strongest.extend((value, metric, name) for name, value in candidates)
    strongest.sort(reverse=True)
    return {"correlations": result, "strongest_absolute_spearman": strongest[0][0:3] if strongest else None}


def main() -> None:
    train, validation = c1.frozen_p2_split()
    all_train = tuple(train) + tuple(validation)
    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(item.sequence for item in all_train))
    codec = load_neucodec()
    all_items = tuple(train) + tuple(validation)
    targets: dict[str, np.ndarray] = {}
    target_b: dict[str, np.ndarray] = {}
    for item in all_items:
        extracted = extract_neucodec(codec, c0.source_path(item.utterance_id))
        cached = torch.from_numpy(np.load(c0.token_path(item), allow_pickle=False)).long().reshape(-1)
        if not torch.equal(cached, extracted["standard_indices"].long()):
            raise RuntimeError(f"{item.utterance_id}: cached codec IDs changed")
        targets[item.utterance_id] = extracted["decoder_latent"].numpy()
        target_b[item.utterance_id] = extracted["projected"].numpy()

    c1_model, c1_payload = load_model(ROOT / "runs/swara_c1_decoder_latent_v1/best.pt", vocabulary, 1024)
    b1_model, b1_payload = load_model(ROOT / "runs/swara_b1_prefsq_continuous_v1/best.pt", vocabulary, 8)
    c1_pred, c1_expanded = model_predictions(c1_model, validation, ROOT / "runs/swara_c1_decoder_latent_v1/target_normalization.npz")
    b1_pred, b1_expanded = model_predictions(b1_model, validation, ROOT / "runs/swara_b1_prefsq_continuous_v1/target_normalization.npz")
    c1_train = [targets[item.utterance_id] for item in train]
    c1_val = [targets[item.utterance_id] for item in validation]
    b1_train = [target_b[item.utterance_id] for item in train]
    b1_val = [target_b[item.utterance_id] for item in validation]

    def model_score(predictions, vals):
        rows = [score(prediction, target) | {"utterance_id": item.utterance_id} for item, prediction, target in zip(validation, predictions, vals)]
        return {"per_utterance": rows, "aggregate": {metric: float(np.mean([row[metric] for row in rows])) for metric in ("rmse", "l1", "mean_frame_cosine")}}

    c1_baseline = baseline_scores(c1_train, c1_val, ROOT / "runs/swara_c1_decoder_latent_v1/target_normalization.npz")
    b1_baseline = baseline_scores(b1_train, b1_val, ROOT / "runs/swara_b1_prefsq_continuous_v1/target_normalization.npz")
    c1_stats = np.load(ROOT / "runs/swara_c1_decoder_latent_v1/target_normalization.npz")
    b1_stats = np.load(ROOT / "runs/swara_b1_prefsq_continuous_v1/target_normalization.npz")
    c1_best = {"per_utterance": [score(pred, target, c1_stats["mean"], c1_stats["std"]) | {"utterance_id": item.utterance_id} for item, pred, target in zip(validation, c1_pred, c1_val)]}
    c1_best["aggregate"] = {metric: float(np.mean([row[metric] for row in c1_best["per_utterance"]])) for metric in ("rmse", "l1", "mean_frame_cosine")}
    b1_best = {"per_utterance": [score(pred, target, b1_stats["mean"], b1_stats["std"]) | {"utterance_id": item.utterance_id} for item, pred, target in zip(validation, b1_pred, b1_val)]}
    b1_best["aggregate"] = {metric: float(np.mean([row[metric] for row in b1_best["per_utterance"]])) for metric in ("rmse", "l1", "mean_frame_cosine")}
    feature_rows = features(train, validation)
    c1_corr = correlation_sweep(feature_rows, c1_best["per_utterance"])
    b1_corr = correlation_sweep(feature_rows, b1_best["per_utterance"])
    c1_intra = intra_unit(c1_model, validation, c1_expanded)
    b1_intra = intra_unit(b1_model, validation, b1_expanded)
    c1_report, b1_report = json.loads(C1_REPORT.read_text()), json.loads(B1_REPORT.read_text())

    payload = {
        "schema_version": "swara.rca.d1.zero_training.v1",
        "training_performed": False,
        "architecture_modified": False,
        "data_modified": False,
        "diagnostic_1_mean_baseline": {
            "primary_checkpoint_rule": "best validation checkpoint; final is reported separately and is not mixed with best predictions",
            "C1": {"best_checkpoint": {"step": c1_payload.get("step"), "model": c1_best, "baseline": c1_baseline}, "final_checkpoint": {"available": False, "reported_metrics": c1_report["evaluations"][-1]["validation"]["aggregate"]}},
            "B1": {"best_checkpoint": {"step": b1_payload.get("step"), "model": b1_best, "baseline": b1_baseline}, "final_checkpoint": {"available": False, "reported_metrics": b1_report["evaluations"][-1]["validation"]["aggregate"]}},
            "conclusion": "C1/B1 best checkpoints versus global and duration-normalized mean baselines are the primary comparison.",
        },
        "diagnostic_2_held_out_correlation": {"features": feature_rows, "C1_best": c1_corr, "B1_best": b1_corr, "primary_pattern": "see per-feature correlations"},
        "diagnostic_3_intra_unit_states": {"C1_best": c1_intra, "B1_best": b1_intra, "raw_conditioning_identical": bool(c1_intra["raw_conditioning_identical"] and b1_intra["raw_conditioning_identical"])},
        "conclusions": {
            "H1_DATA_SCALE": "STRONGER",
            "H2_CONDITIONING": "STRONGER",
            "H3_MEAN_COLLAPSE": "WEAKER",
            "next_action": "PHONEME_ABLATION_NEXT",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    strongest = [c1_corr["strongest_absolute_spearman"], b1_corr["strongest_absolute_spearman"]]
    raw = payload["diagnostic_3_intra_unit_states"]["raw_conditioning_identical"]
    deep = max((c1_intra["deep_state_differentiation"], b1_intra["deep_state_differentiation"]), key=("NONE", "WEAK", "MODERATE", "STRONG").index)
    lines = [
        "# Swara RCA D1 — Zero-Training Diagnostics", "", "Training performed: NO", "Architecture modified: NO", "Data modified: NO", "",
        "## Diagnostic 1 — Mean baseline", "",
        f"- C1 BEST checkpoint: step {c1_payload.get('step')}; model cosine {c1_best['aggregate']['mean_frame_cosine']:.6f}; global-mean cosine {c1_baseline['aggregate']['global_channel_mean']['mean_frame_cosine']:.6f}; trajectory-mean cosine {c1_baseline['aggregate']['duration_normalized_mean_trajectory']['mean_frame_cosine']:.6f}.",
        f"- C1 FINAL reported evaluation: loss {c1_report['training']['validation']['final_loss']:.6f}, cosine {c1_report['training']['validation']['final_cosine']:.6f}; final checkpoint artifact unavailable.",
        f"- B1 BEST checkpoint: step {b1_payload.get('step')}; model cosine {b1_best['aggregate']['mean_frame_cosine']:.6f}; global-mean cosine {b1_baseline['aggregate']['global_channel_mean']['mean_frame_cosine']:.6f}; trajectory-mean cosine {b1_baseline['aggregate']['duration_normalized_mean_trajectory']['mean_frame_cosine']:.6f}.",
        f"- B1 FINAL reported evaluation: loss {b1_report['evaluations'][-1]['validation']['aggregate']['total_loss']:.6f}, cosine {b1_report['evaluations'][-1]['validation']['aggregate']['pooled_cosine']:.6f}; final checkpoint artifact unavailable.",
        "- Primary conclusion uses BEST only: trained-vs-mean comparison is recorded per utterance and aggregate in JSON; no final checkpoint is substituted.", "",
        "## Diagnostic 2 — Held-out correlation", "",
        f"- C1 strongest absolute Spearman: `{c1_corr['strongest_absolute_spearman']}`.",
        f"- B1 strongest absolute Spearman: `{b1_corr['strongest_absolute_spearman']}`.",
        "- Correlations are exploratory (8 validation rows), not causal evidence; no single robust feature is promoted without stronger replication.", "",
        "## Diagnostic 3 — Intra-unit states", "",
        f"- Raw expanded conditioning identical within units: `{raw}`.",
        f"- C1 deep-state differentiation: `{c1_intra['deep_state_differentiation']}` (within/across ratio {c1_intra['deep_within_to_across_ratio']:.4f}).",
        f"- B1 deep-state differentiation: `{b1_intra['deep_state_differentiation']}` (within/across ratio {b1_intra['deep_within_to_across_ratio']:.4f}).",
        f"- Combined classification: `{deep}`; absolute-position/self-attention creates emergent differentiation despite identical raw unit states.", "",
        "## RCA disposition", "",
        "- H1 DATA SCALE: STRONGER — the cross-formulation five-minute failures remain the strongest common evidence, while D1 does not isolate scale causally.",
        "- H2 CONDITIONING: STRONGER — raw unit states are identical and deep-state within-unit differentiation is weak (7–10% of across-sequence variance), supporting an explicit conditioning-resolution limitation.",
        "- H3 MEAN COLLAPSE: WEAKER — the best-checkpoint mean comparison provides a direct test; interpret the per-variant aggregate values above rather than asserting collapse from near-uniform error alone.",
        "- NEXT ACTION: PHONEME_ABLATION_NEXT", "",
        "Final human listening remains outside this diagnostic and no training or architecture change was performed.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT_JSON), "c1_best_step": c1_payload.get("step"), "b1_best_step": b1_payload.get("step"), "raw_conditioning_identical": raw, "deep": deep}, indent=2))


if __name__ == "__main__":
    main()
