#!/usr/bin/env python3
"""Run Swara B0: predict the NeuCodec pre-FSQ continuous latent [T,8].

B0 is a 2-utterance memorization experiment asking whether R0's Target B
(the ``ResidualFSQ.project_in`` output, immediately before the official FSQ
bounding/quantization path) is easier to learn than the 1024-D decoder
latent used by C0/C0b (Target C).

Everything downstream of "predict [T,8]" is the frozen, already-validated R0
target path: official FSQ, ``fc_post_a``, and the frozen decoder.  Only the
acoustic predictor's final output width changes relative to C0b (1024 -> 8);
the linguistic side, monotonic expansion, loss, and optimizer are reused
unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swara_c0_decoder_latent as c0  # noqa: E402
from run_continuous_target_bakeoff import (  # noqa: E402
    NEUCODEC_ID,
    NEUCODEC_REVISION,
    decode_neucodec_indices,
    decode_neucodec_projected,
    extract_neucodec,
    load_neucodec,
)
from swara.diagnostics.continuous_targets import audio_integrity, quantization_diagnostics  # noqa: E402
from swara.models.c0_decoder_latent import C0PredictorConfig, SwaraC0DecoderLatentModel, normalized_decoder_latent_loss  # noqa: E402
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402
from swara.training.speech_poc_dataset import load_duration_supervision, select_examples  # noqa: E402


MANIFEST = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
EVAL_ROOT = ROOT / "evaluations/swara_b0_prefsq_continuous_v1"
RUN_ROOT = ROOT / "runs/swara_b0_prefsq_continuous_v1"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/b0_prefsq_continuous_v1.json"
RESEARCH_PATH = ROOT / "research/poc/diagnostics/B0_PREFSQ_CONTINUOUS_V1.md"
STATS_PATH = RUN_ROOT / "target_normalization.npz"
CHECKPOINT_PATH = RUN_ROOT / "best.pt"

SEED = 20260824
SELECTED_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_2140",
    "IISc_SPICORProject_EN_M_AGRI_7084",
)
EXPECTED_TRANSCRIPTS = (
    "This isn't the right time to check into the Lemon Tree stock",
    "Among these, trees of Ashok, Kachnaar, Amaltaash, Neem, Australian Babul, Kaner, Sheesham, Sagaun, Mango, Pomegranate, Papaya can also be found",
)
MAX_STEPS = 500
EVALUATION_STEPS = (100, 200, 500)
RUNTIME_BUDGET_SECONDS = 10 * 60


def per_dimension_stats(rows: Sequence[Tensor]) -> dict[str, Any]:
    merged = torch.cat(tuple(row.float() for row in rows), dim=0)
    mean = merged.mean(dim=0)
    std = merged.std(dim=0, unbiased=False)
    minimum = merged.min(dim=0).values
    maximum = merged.max(dim=0).values
    p01 = torch.quantile(merged, 0.01, dim=0)
    p99 = torch.quantile(merged, 0.99, dim=0)
    return {
        "mean": mean,
        "std": std,
        "report": {
            "channels": int(merged.shape[1]),
            "frames": int(merged.shape[0]),
            "global_mean": float(merged.mean().item()),
            "global_std": float(merged.std(unbiased=False).item()),
            "global_min": float(merged.min().item()),
            "global_max": float(merged.max().item()),
            "per_dimension_mean": [float(x) for x in mean],
            "per_dimension_std": [float(x) for x in std],
            "per_dimension_min": [float(x) for x in minimum],
            "per_dimension_max": [float(x) for x in maximum],
            "per_dimension_p01": [float(x) for x in p01],
            "per_dimension_p99": [float(x) for x in p99],
        },
    }


def masked_pooled_cosine(prediction: Tensor, target: Tensor, padding_mask: Tensor) -> float:
    valid = ~padding_mask
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    return float(cosine[valid].mean().item())


@torch.inference_mode()
def evaluate(
    model: SwaraC0DecoderLatentModel,
    codec,
    examples,
    target_raw: dict[str, dict[str, Any]],
    target: Tensor,
    target_norm: Tensor,
    mean: Tensor,
    std: Tensor,
    step: int,
    folder: Path,
) -> dict[str, Any]:
    model.eval()
    prediction_norm, aligned = c0.forward_batch(model, examples)
    if aligned.padding_mask.shape != target.shape[:2]:
        raise RuntimeError("B0 evaluation target/alignment geometry differs")
    aggregate = normalized_decoder_latent_loss(prediction_norm, target_norm, aligned.padding_mask)
    prediction = prediction_norm * (std + 1e-6) + mean
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        frames = example.target_total_frames
        predicted = prediction[index, :frames]
        truth = target[index, :frames]
        normalized_error = F.smooth_l1_loss(prediction_norm[index, :frames], target_norm[index, :frames])
        difference = predicted - truth
        cosine = F.cosine_similarity(predicted, truth, dim=-1).mean()
        waveform, predicted_indices, predicted_coordinates = decode_neucodec_projected(
            codec, predicted.detach().cpu().numpy()
        )
        integrity = c0.save_wave(folder / f"{example.utterance_id}.wav", waveform)
        target_indices = target_raw[example.utterance_id]["standard_indices"]
        target_coordinates = target_raw[example.utterance_id]["coordinates"]
        fsq = quantization_diagnostics(target_indices, predicted_indices, target_coordinates, predicted_coordinates)
        rows.append({
            "utterance_id": example.utterance_id,
            "frames": frames,
            "normalized_smooth_l1": float(normalized_error.item()),
            "latent_rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
            "latent_cosine": float(cosine.item()),
            "fsq": {
                "frame_token_match_rate": fsq["exact_token_retention"],
                "coordinate_quantization_match_rate": 1.0 - fsq["coordinate_boundary_crossing_rate"],
                "self_transition_rate": fsq["self_transition_rate"],
                "exact_bigram_retention": fsq["exact_bigram_retention"],
            },
            "waveform": integrity,
        })
    return {
        "step": step,
        "aggregate": {
            "normalized_smooth_l1": float(aggregate.latent.item()),
            "normalized_delta_smooth_l1": float(aggregate.delta.item()),
            "total_loss": float(aggregate.total.item()),
        },
        "utterances": rows,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_research(report: dict[str, Any]) -> None:
    lines = [
        "# B0 Pre-FSQ Continuous Target — 2-Utterance Memorization", "",
        "Status: machine run complete; human listening is primary for the success gate.", "",
        "## Frozen scope", "",
        f"- Seed: `{SEED}`",
        f"- Utterances: `{SELECTED_IDS[0]}`, `{SELECTED_IDS[1]}`",
        "- Target: R0 Target B, `ResidualFSQ.project_in` output `[T,8]`, immediately before official FSQ bounding/quantization",
        "- Reused unchanged: linguistic encoder, GT duration monotonic expansion, C0b temporal predictor architecture (output width 1024->8 only), loss, optimizer",
        "- Downstream of prediction: official frozen FSQ -> fc_post_a -> frozen decoder (unchanged)",
        "- Autoregressive feedback / categorical 8x4 heads / Target-C prediction / flow / diffusion: none", "",
        "## Target-B normalization (train-derived, both utterances)", "",
    ]
    stats = report["target"]["statistics"]
    lines.append(f"- Global mean/std: `{stats['global_mean']:.6f}` / `{stats['global_std']:.6f}`")
    lines.append(f"- Global min/max: `{stats['global_min']:.6f}` / `{stats['global_max']:.6f}`")
    lines.append(f"- Per-dimension mean: `{[round(x, 4) for x in stats['per_dimension_mean']]}`")
    lines.append(f"- Per-dimension std: `{[round(x, 4) for x in stats['per_dimension_std']]}`")
    lines.append("")
    lines += ["## Equivalence", "", ]
    for row in report["utterances"]:
        lines.append(
            f"- `{row['utterance_id']}`: {row['frames']} frames; cached-ID equivalence PASS; "
            f"oracle waveform max difference vs standard cached-ID decode `{row['oracle_equivalence_max_abs']:.3g}`."
        )
    training = report["training"]
    lines += ["", "## Runtime and training", "",
        f"- Device: `{training['device']}`",
        f"- New predictor parameters: `{report['parameters']['new_acoustic_predictor']:,}`",
        f"- Total trainable parameters: `{report['parameters']['total_trainable']:,}`",
        f"- Steps completed: `{training['steps_completed']}` / `{training['maximum_steps']}`",
        f"- Wall time: `{training['wall_seconds']:.2f}` seconds",
        f"- Stop reason: `{training['stop_reason']}`",
        f"- Initial loss: `{training['initial_loss']:.6f}`",
        f"- Best loss: `{training['best_loss']:.6f}` at step `{training['best_step']}`",
        f"- Final loss: `{training['final_loss']:.6f}`", "",
    ]
    if training.get("step100_inspection") is not None:
        lines += ["## Step-100 inspection", "", training["step100_inspection"]["note"], ""]
    lines += ["## Per-utterance final (step %d) metrics" % report["evaluations"][-1]["step"], "", ]
    for row in report["evaluations"][-1]["utterances"]:
        lines.append(
            f"- `{row['utterance_id']}`: latent cosine `{row['latent_cosine']:.4f}`; "
            f"FSQ frame token match `{row['fsq']['frame_token_match_rate']:.4f}`; "
            f"self-transition `{row['fsq']['self_transition_rate']:.4f}`; "
            f"audio finite={row['waveform']['finite']} non_silent={row['waveform']['non_silent']} "
            f"rms={row['waveform']['rms']}"
        )
    lines += ["", "## Listening gate", "",
        "Machine checks establish finite/non-silent audio, continuous-latent fit, and FSQ token retention only. "
        "They do not establish intelligibility or transcript match.",
        f"Listen under `{EVAL_ROOT.relative_to(ROOT)}` and classify both utterances against the B0 success gate "
        "(PASS / PARTIAL / FAIL). No generalization claim is authorized from this 2-utterance memorization run.", "",
        f"B0 remains `{report['machine_classification']}` until that review is supplied.", "",
        "Training performed: YES (bounded B0 only)  ",
        "Codec modified: NO  ",
        "Commit/push: NO", "",
    ]
    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    c0.seed_everything()
    device = torch.device("cpu")

    all_examples = load_duration_supervision(MANIFEST, split=None)
    examples = select_examples(all_examples, SELECTED_IDS)
    for example, expected in zip(examples, EXPECTED_TRANSCRIPTS):
        if example.sequence.normalized_text != expected:
            raise RuntimeError(f"{example.utterance_id}: authoritative transcript drift")
        if not c0.source_path(example.utterance_id).is_file():
            raise FileNotFoundError(c0.source_path(example.utterance_id))

    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(e.sequence for e in all_examples))
    predictor_config = C0PredictorConfig(output_width=8)
    model = SwaraC0DecoderLatentModel(vocabulary, predictor_config=predictor_config).to(device)
    predictor_parameters = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    codec = load_neucodec()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    target_raw: dict[str, dict[str, Any]] = {}
    target_rows: list[Tensor] = []
    oracle_rows: list[dict[str, Any]] = []
    for example in examples:
        extracted = extract_neucodec(codec, c0.source_path(example.utterance_id))
        target_latent = extracted["projected"].float()
        cached_ids = torch.from_numpy(np.load(c0.token_path(example), allow_pickle=False)).long().reshape(-1)
        standard_ids = extracted["standard_indices"].long().reshape(-1)
        if not torch.equal(cached_ids, standard_ids):
            raise RuntimeError(f"{example.utterance_id}: Target-B extraction differs from frozen cached codec IDs")
        if target_latent.shape != (example.target_total_frames, 8):
            raise RuntimeError(
                f"{example.utterance_id}: Target-B {tuple(target_latent.shape)} != "
                f"GT expansion ({example.target_total_frames}, 8); refusing to interpolate"
            )
        oracle_waveform, oracle_indices, oracle_coordinates = decode_neucodec_projected(
            codec, target_latent.numpy()
        )
        if not torch.equal(oracle_indices.long(), standard_ids):
            raise RuntimeError(f"{example.utterance_id}: Target-B oracle FSQ indices differ from cached codec IDs")
        reference_waveform = decode_neucodec_indices(codec, standard_ids)
        if oracle_waveform.shape != reference_waveform.shape:
            raise RuntimeError(f"{example.utterance_id}: Target-B oracle waveform shape mismatch")
        maximum = float(np.max(np.abs(oracle_waveform - reference_waveform)))
        if maximum > 1e-6:
            raise RuntimeError(f"{example.utterance_id}: Target-B clean waveform equivalence regression ({maximum})")
        oracle_audio = c0.save_wave(EVAL_ROOT / "oracle" / f"{example.utterance_id}.wav", oracle_waveform)
        target_raw[example.utterance_id] = {
            "target": target_latent,
            "standard_indices": standard_ids,
            "coordinates": extracted["coordinates"].float(),
        }
        target_rows.append(target_latent)
        oracle_rows.append({
            "utterance_id": example.utterance_id,
            "transcript": example.sequence.normalized_text,
            "frames": example.target_total_frames,
            "cached_codec_ids_exact": True,
            "oracle_equivalence_max_abs": maximum,
            "oracle_audio": oracle_audio,
        })

    target, target_padding = c0.pad_targets(target_rows, device)
    stats = per_dimension_stats(target_rows)
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATS_PATH, mean=mean.cpu().numpy(), std=std.cpu().numpy(), epsilon=np.array(1e-6, np.float32))
    stats_report = dict(stats["report"])
    stats_report.update({"path": str(STATS_PATH.relative_to(ROOT)), "sha256": c0.sha256(STATS_PATH)})

    model.train()
    prediction, aligned = c0.forward_batch(model, examples)
    if not torch.equal(aligned.padding_mask, target_padding):
        raise RuntimeError("B0 GT expanded linguistic frame mask differs from Target-B frame mask")
    target_norm = ((target - mean) / (std + 1e-6)).masked_fill(target_padding.unsqueeze(-1), 0.0)
    initial_losses = normalized_decoder_latent_loss(prediction, target_norm, target_padding)
    initial_losses.total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("B0 preflight backward produced non-finite gradients")
    model.zero_grad(set_to_none=True)

    benchmark_started = time.perf_counter()
    prediction, aligned = c0.forward_batch(model, examples)
    benchmark_loss = normalized_decoder_latent_loss(prediction, target_norm, aligned.padding_mask).total
    benchmark_loss.backward()
    benchmark_seconds = time.perf_counter() - benchmark_started
    model.zero_grad(set_to_none=True)
    estimated_seconds = benchmark_seconds * MAX_STEPS
    if estimated_seconds > RUNTIME_BUDGET_SECONDS:
        raise RuntimeError(
            f"B0_BLOCKED: estimated runtime {estimated_seconds / 60:.1f} minutes exceeds the 10-minute budget; "
            "stopping before training as required."
        )
    print(f"B0_PREFLIGHT: PASS frames={[e.target_total_frames for e in examples]} predictor_params={predictor_parameters}")
    print(f"B0_RUNTIME_ESTIMATE_SECONDS: {estimated_seconds:.1f}", flush=True)

    optimizer = c0.optimizer_for(model)
    evaluations: list[dict[str, Any]] = []
    initial_loss = float(initial_losses.total.item())
    best_loss, best_step = initial_loss, 0
    stop_reason = "maximum_steps"
    step100_inspection: dict[str, Any] | None = None

    training_started = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, aligned = c0.forward_batch(model, examples)
        losses = normalized_decoder_latent_loss(prediction, target_norm, aligned.padding_mask)
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"B0 non-finite loss at optimizer step {step}")
        losses.total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"B0 non-finite gradient at optimizer step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step in EVALUATION_STEPS or step == MAX_STEPS:
            result = evaluate(model, codec, examples, target_raw, target, target_norm, mean, std, step, EVAL_ROOT / f"step_{step:03d}")
            evaluations.append(result)
            current = result["aggregate"]["total_loss"]
            print(f"B0 step={step} loss={current:.6f}", flush=True)
            if current < best_loss:
                best_loss, best_step = current, step
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema_version": "swara.b0.prefsq_continuous.v1",
                    "seed": SEED,
                    "step": step,
                    "selected_ids": SELECTED_IDS,
                    "model": model.state_dict(),
                    "normalization_path": str(STATS_PATH.relative_to(ROOT)),
                }, CHECKPOINT_PATH)

            if step == 100:
                both_non_speech = all(
                    (not row["waveform"]["finite"]) or (not row["waveform"]["non_silent"])
                    for row in result["utterances"]
                )
                if both_non_speech:
                    step100_inspection = {
                        "step": 100,
                        "both_non_speech": True,
                        "note": (
                            "Step-100 inspection: both utterances failed the finite/non-silent audio "
                            "integrity gate. Per B0's fail-fast contract this is a one-time inspection, "
                            "not a hard stop -- training continued to the maximum step budget."
                        ),
                    }
                    print("B0_STEP100_INSPECTION: both utterances non-speech at step 100", flush=True)

        if step == 50 and step not in EVALUATION_STEPS:
            with torch.inference_mode():
                probe_prediction, probe_aligned = c0.forward_batch(model.eval(), examples)
                probe_loss = normalized_decoder_latent_loss(probe_prediction, target_norm, probe_aligned.padding_mask).total
            model.train()
            if float(probe_loss.item()) >= initial_loss * 0.95:
                stop_reason = "no_clear_loss_improvement_by_step_50"
                break

    wall_seconds = time.perf_counter() - training_started
    steps_completed = evaluations[-1]["step"]
    final_step_rows = evaluations[-1]["utterances"]
    final_non_speech = all(
        (not row["waveform"]["finite"]) or (not row["waveform"]["non_silent"]) for row in final_step_rows
    )
    machine_classification = "FAIL" if final_non_speech else "HUMAN_REVIEW_REQUIRED"

    report = {
        "schema_version": "swara.b0.prefsq_continuous.v1",
        "status": "human_listening_required",
        "seed": SEED,
        "utterances": oracle_rows,
        "target": {
            "description": "NeuCodec pre-FSQ continuous latent (ResidualFSQ.project_in output), before official FSQ bounding/quantization",
            "shape": "[B,T,8]",
            "codec_model": NEUCODEC_ID,
            "codec_revision": NEUCODEC_REVISION,
            "clean_id_equivalence": "PASS (reused from R0)",
            "clean_waveform_equivalence": "PASS (reused from R0)",
            "statistics": stats_report,
            "normalization": "per-dimension train standardization: (z - mean_dim) / (std_dim + eps)",
        },
        "model": {
            "reused_from": "C0b temporal predictor architecture (output width 1024->8 only)",
            "non_autoregressive": True,
            "ground_truth_durations_only": True,
            "previous_acoustic_state": False,
            "categorical_codec_prediction": False,
            "flow_matching": False,
            "architecture": "3 non-causal Transformer encoder blocks; width 256; 4 heads; FFN 1024; output 8",
        },
        "parameters": {
            "new_acoustic_predictor": predictor_parameters,
            "total_trainable": total_parameters,
        },
        "training": {
            "device": str(device),
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "delta_loss_weight": 0.1,
            "steps_completed": steps_completed,
            "maximum_steps": MAX_STEPS,
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "best_step": best_step,
            "final_loss": evaluations[-1]["aggregate"]["total_loss"],
            "estimated_seconds_before_training": estimated_seconds,
            "wall_seconds": wall_seconds,
            "stop_reason": stop_reason,
            "step100_inspection": step100_inspection,
        },
        "evaluations": evaluations,
        "oracle_audio_generated": True,
        "human_listening_required": True,
        "machine_classification": machine_classification,
        "autoregression": False,
        "categorical_codec_prediction": False,
        "flow_matching": False,
        "codec_modified": False,
        "commit_push": False,
    }
    write_json(REPORT_PATH, report)
    write_research(report)
    print(f"B0_COMPLETE steps={steps_completed} best_step={best_step} wall_seconds={wall_seconds:.1f}")


if __name__ == "__main__":
    main()
