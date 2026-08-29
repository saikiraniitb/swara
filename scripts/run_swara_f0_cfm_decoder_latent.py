#!/usr/bin/env python3
"""Run Swara F0: conditional flow matching toward Target-C [T,1024], 2 utterances.

F0 tests a new generation objective -- straight-line conditional flow
matching from Gaussian noise -- after deterministic regression toward both
Target-C (C1) and Target-B (B1) failed to generalize.  This is a bounded
2-utterance memorization experiment only; no generalization claim is made.

Reused unchanged: the R0/C0 Target-C extraction path (frozen `fc_post_a`
output, decoded via `CodecDecoderVocos(vq=False)`), the accepted linguistic
encoder / GT-duration monotonic expansion, and the same AdamW-with-decay-
grouping optimizer helper C0 uses.  New: `swara.models.f0_cfm`, a small
non-causal flow-matching velocity predictor and its training/inference
utilities.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    decode_neucodec_latent,
    extract_neucodec,
    load_neucodec,
)
from swara.models.f0_cfm import (  # noqa: E402
    F0PredictorConfig,
    SwaraF0CFMModel,
    euler_integrate,
    masked_velocity_loss,
    sample_flow_matching_batch,
)
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402
from swara.training.speech_poc_dataset import load_duration_supervision, select_examples  # noqa: E402


MANIFEST = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
EVAL_ROOT = ROOT / "evaluations/swara_f0_cfm_decoder_latent_v1"
RUN_ROOT = ROOT / "runs/swara_f0_cfm_decoder_latent_v1"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/f0_cfm_decoder_latent_v1.json"
RESEARCH_PATH = ROOT / "research/poc/diagnostics/F0_CFM_DECODER_LATENT_V1.md"
MANIFEST_PATH = EVAL_ROOT / "LISTENING_MANIFEST.md"
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
INFERENCE_SEEDS = {
    "IISc_SPICORProject_EN_M_AGRI_2140": 2026082401,
    "IISc_SPICORProject_EN_M_AGRI_7084": 2026082402,
}
MULTI_SEED_EXTRA = (2026082401, 2026082411, 2026082421)  # 3 distinct seeds for the sanity check
MAX_STEPS = 1000
EVALUATION_STEPS = (50, 100, 200, 400, 600, 1000)
RUNTIME_BUDGET_SECONDS = 15 * 60
CHECKPOINT_EULER_STEPS = 8
BEST_EULER_STEPS = (4, 8, 16)
WARMUP_BENCH_STEPS = 5
MEASURED_BENCH_STEPS = 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tensor(tensor: Tensor) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def codec_state_hash(codec) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(codec.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@torch.no_grad()
def generate_and_decode(
    model: SwaraF0CFMModel,
    codec,
    examples,
    mean: Tensor,
    std: Tensor,
    device: torch.device,
    num_steps: int,
    seeds: dict[str, int],
    folder: Path,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for example in examples:
        aligned = model.align((example.sequence,), (example.alignment_units,), (example.target_total_frames,))
        frames = aligned.states.shape[1]
        generator = torch.Generator(device="cpu").manual_seed(seeds[example.utterance_id])
        x0 = torch.randn(1, frames, model.velocity_predictor.config.latent_width, generator=generator).to(device)
        x1_pred_norm = euler_integrate(model.velocity_predictor, aligned, x0, num_steps)
        x1_pred = x1_pred_norm * (std + 1e-6) + mean
        valid = x1_pred[0, : example.target_total_frames]
        waveform = decode_neucodec_latent(codec, valid.cpu().numpy())
        integrity = c0.save_wave(folder / f"{example.utterance_id}.wav", waveform)
        rows.append({
            "utterance_id": example.utterance_id,
            "euler_steps": num_steps,
            "seed": seeds[example.utterance_id],
            "generated_latent_sha256": sha256_tensor(valid),
            "waveform": integrity,
        })
    return rows


def compare_generated_to_real(generated: Tensor, real: Tensor) -> dict[str, float]:
    difference = generated - real
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    cosine = float(F.cosine_similarity(generated, real, dim=-1).mean().item())
    if generated.shape[0] > 1:
        delta_gen = generated[1:] - generated[:-1]
        delta_real = real[1:] - real[:-1]
        temporal_derivative_error = float(torch.sqrt(torch.mean((delta_gen - delta_real).square())).item())
    else:
        temporal_derivative_error = 0.0
    scale = float(torch.sqrt(torch.mean(real.square())).item()) + 1e-8
    return {
        "rmse": rmse,
        "normalized_rmse": rmse / scale,
        "latent_cosine": cosine,
        "temporal_derivative_error": temporal_derivative_error,
    }


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
    predictor_config = F0PredictorConfig()  # hidden=256, layers=4, heads=4, ffn=1024, latent_width=1024
    model = SwaraF0CFMModel(vocabulary, predictor_config=predictor_config).to(device)
    flow_parameters = sum(p.numel() for p in model.velocity_predictor.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    codec = load_neucodec()
    codec_hash_before = codec_state_hash(codec)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    target_rows: list[Tensor] = []
    oracle_rows: list[dict[str, Any]] = []
    for example in examples:
        extracted = extract_neucodec(codec, c0.source_path(example.utterance_id))
        target_latent = extracted["decoder_latent"].float()
        cached_ids = torch.from_numpy(np.load(c0.token_path(example), allow_pickle=False)).long().reshape(-1)
        standard_ids = extracted["standard_indices"].long().reshape(-1)
        if not torch.equal(cached_ids, standard_ids):
            raise RuntimeError(f"{example.utterance_id}: Target-C extraction differs from frozen cached codec IDs")
        if target_latent.shape != (example.target_total_frames, 1024):
            raise RuntimeError(
                f"{example.utterance_id}: Target-C {tuple(target_latent.shape)} != "
                f"GT expansion ({example.target_total_frames}, 1024); refusing to interpolate"
            )
        oracle_direct = decode_neucodec_latent(codec, target_latent.numpy())
        oracle_standard = decode_neucodec_indices(codec, standard_ids)
        if oracle_direct.shape != oracle_standard.shape:
            raise RuntimeError(f"{example.utterance_id}: Target-C oracle shape mismatch")
        maximum = float(np.max(np.abs(oracle_direct - oracle_standard)))
        if maximum > 1e-6:
            raise RuntimeError(f"{example.utterance_id}: Target-C clean equivalence regression ({maximum})")
        oracle_audio = c0.save_wave(EVAL_ROOT / "oracle" / f"{example.utterance_id}.wav", oracle_direct)
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
    stats = c0.normalization_statistics(target_rows)
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATS_PATH, mean=mean.cpu().numpy(), std=std.cpu().numpy(), epsilon=np.array(1e-6, np.float32))
    stats_report = {k: v for k, v in stats.items() if k not in {"mean", "std"}}
    stats_report.update({
        "channels": 1024,
        "frames": int(sum(row.shape[0] for row in target_rows)),
        "path": str(STATS_PATH.relative_to(ROOT)),
        "sha256": c0.sha256(STATS_PATH),
        "derived_from": "train_only (both fixed utterances)",
    })
    target_norm_full = ((target - mean) / (std + 1e-6)).masked_fill(target_padding.unsqueeze(-1), 0.0)

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    def forward_and_loss():
        aligned = model.align(
            tuple(e.sequence for e in examples),
            tuple(e.alignment_units for e in examples),
            tuple(e.target_total_frames for e in examples),
        )
        padding_device = aligned.padding_mask
        if not torch.equal(padding_device.cpu(), target_padding):
            raise RuntimeError("F0 GT expanded linguistic frame mask differs from Target-C frame mask")
        # x0/t are always sampled on CPU (the generator's device); move to the
        # active training device only for the model forward/loss call.
        xt, t, v_target = sample_flow_matching_batch(target_norm_full, target_padding, generator)
        v_pred = model.velocity(xt.to(padding_device.device), t.to(padding_device.device), aligned)
        losses = masked_velocity_loss(v_pred, v_target.to(padding_device.device), padding_device)
        return losses

    model.train()
    preflight_losses = forward_and_loss()
    preflight_losses.mse.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("F0 preflight backward produced non-finite gradients")
    model.zero_grad(set_to_none=True)

    print(f"F0_PREFLIGHT: PASS flow_params={flow_parameters} total_params={total_parameters}")

    optimizer = c0.optimizer_for(model)

    def run_one_step() -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = forward_and_loss()
        if not torch.isfinite(losses.mse):
            raise RuntimeError("F0 non-finite loss during benchmark/training")
        losses.mse.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("F0 non-finite gradient during benchmark/training")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(losses.mse.item())

    for _ in range(WARMUP_BENCH_STEPS):
        run_one_step()
    bench_started = time.perf_counter()
    for _ in range(MEASURED_BENCH_STEPS):
        run_one_step()
    bench_seconds = (time.perf_counter() - bench_started) / MEASURED_BENCH_STEPS
    estimated_1000_step_seconds = bench_seconds * MAX_STEPS
    print(f"F0_BENCHMARK: sec_per_step={bench_seconds:.4f} estimated_1000_steps={estimated_1000_step_seconds:.1f}s", flush=True)

    device_used = "cpu"
    if estimated_1000_step_seconds > RUNTIME_BUDGET_SECONDS:
        print("F0_BENCHMARK: CPU estimate exceeds budget; probing MPS as the one allowed lightweight fallback", flush=True)
        if torch.backends.mps.is_available():
            try:
                mps_model = SwaraF0CFMModel(vocabulary, predictor_config=predictor_config).to("mps")
                mps_model.load_state_dict(model.state_dict())
                mps_optimizer = c0.optimizer_for(mps_model)
                mps_generator = torch.Generator(device="cpu").manual_seed(SEED)
                mps_padding = target_padding.to("mps")

                def mps_step() -> None:
                    mps_model.train()
                    mps_optimizer.zero_grad(set_to_none=True)
                    aligned = mps_model.align(
                        tuple(e.sequence for e in examples),
                        tuple(e.alignment_units for e in examples),
                        tuple(e.target_total_frames for e in examples),
                    )
                    # Sample x0/t on CPU (generator device must match tensor device), then move to MPS.
                    xt, t, v_target = sample_flow_matching_batch(target_norm_full, target_padding, mps_generator)
                    v_pred = mps_model.velocity(xt.to("mps"), t.to("mps"), aligned)
                    losses = masked_velocity_loss(v_pred, v_target.to("mps"), mps_padding)
                    losses.mse.backward()
                    mps_optimizer.step()

                for _ in range(5):
                    mps_step()
                torch.mps.synchronize()
                mps_started = time.perf_counter()
                for _ in range(10):
                    mps_step()
                torch.mps.synchronize()
                mps_seconds = (time.perf_counter() - mps_started) / 10
                mps_estimate = mps_seconds * MAX_STEPS
                print(f"F0_MPS_PROBE: sec_per_step={mps_seconds:.4f} estimated_1000_steps={mps_estimate:.1f}s", flush=True)
                if mps_estimate <= RUNTIME_BUDGET_SECONDS:
                    device_used = "mps"
                    device = torch.device("mps")
                    model = mps_model
                    optimizer = mps_optimizer
                    # target_norm_full/target_padding stay CPU-resident: the
                    # seeded generator is CPU-only, and forward_and_loss()
                    # moves sampled xt/t/v_target to `device` per call.
                    mean, std = mean.to("mps"), std.to("mps")
                    bench_seconds = mps_seconds
                    estimated_1000_step_seconds = mps_estimate
            except Exception as error:  # noqa: BLE001 - single lightweight compatibility attempt only
                print(f"F0_MPS_PROBE: failed ({type(error).__name__}: {error}); staying on CPU", flush=True)

        if estimated_1000_step_seconds > RUNTIME_BUDGET_SECONDS:
            blocked_reason = (
                f"Estimated 1000-step runtime {estimated_1000_step_seconds/60:.1f} minutes exceeds the "
                f"{RUNTIME_BUDGET_SECONDS/60:.0f}-minute local budget on {device_used}, measured from "
                f"{MEASURED_BENCH_STEPS} real steps after {WARMUP_BENCH_STEPS} warmup steps "
                f"({bench_seconds:.3f}s/step)."
            )
            report = {
                "schema_version": "swara.f0.cfm_decoder_latent.v1",
                "status": "blocked",
                "seed": SEED,
                "utterances": oracle_rows,
                "parameters": {"flow_params": flow_parameters, "total_trainable": total_parameters},
                "benchmark": {"seconds_per_step": bench_seconds, "estimated_1000_step_seconds": estimated_1000_step_seconds, "device": device_used},
                "blocked_reason": blocked_reason,
                "machine_classification": "HUMAN_REVIEW_REQUIRED",
                "generalization_tested": False,
                "flow_matching": True,
                "autoregression": False,
                "fsq": False,
                "codec_modified": False,
                "commit_push": False,
            }
            write_json(REPORT_PATH, report)
            RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESEARCH_PATH.write_text(
                f"# F0 CFM Decoder Latent V1\n\nStatus: BLOCKED\n\n{blocked_reason}\n\n"
                "No training step beyond the runtime benchmark was executed.\n",
                encoding="utf-8",
            )
            print(f"F0_BLOCKED: {blocked_reason}")
            return

    print(f"F0_TRAINING: device={device_used} steps<= {MAX_STEPS}", flush=True)
    initial_loss = None
    best_loss, best_step = float("inf"), 0
    evaluations: list[dict[str, Any]] = []
    stop_reason = "maximum_steps"

    training_started = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        loss_value = run_one_step()
        if initial_loss is None:
            initial_loss = loss_value
        best_loss = min(best_loss, loss_value)

        if step in EVALUATION_STEPS:
            folder = EVAL_ROOT / f"step_{step:03d}"
            rows = generate_and_decode(model, codec, examples, mean, std, device, CHECKPOINT_EULER_STEPS, INFERENCE_SEEDS, folder)
            comparisons = []
            model.eval()
            with torch.no_grad():
                for example, row, real_row in zip(examples, rows, target_rows):
                    aligned = model.align((example.sequence,), (example.alignment_units,), (example.target_total_frames,))
                    gen = torch.Generator(device="cpu").manual_seed(row["seed"])
                    x0 = torch.randn(1, aligned.states.shape[1], predictor_config.latent_width, generator=gen).to(device)
                    x1_pred_norm = euler_integrate(model.velocity_predictor, aligned, x0, CHECKPOINT_EULER_STEPS)
                    x1_pred = (x1_pred_norm * (std + 1e-6) + mean)[0, : example.target_total_frames]
                    comparison = compare_generated_to_real(x1_pred, real_row.to(device))
                    comparisons.append({"utterance_id": example.utterance_id, **comparison})
            evaluations.append({"step": step, "loss": loss_value, "generation": rows, "comparison": comparisons})
            print(f"F0 step={step} loss={loss_value:.6f} " + " ".join(f"{c['utterance_id']}:cos={c['latent_cosine']:.3f}" for c in comparisons), flush=True)
        if step == 100 and initial_loss is not None and loss_value >= initial_loss * 0.95:
            stop_reason = "no_meaningful_improvement_by_step_100"
            break

    wall_seconds = time.perf_counter() - training_started
    steps_completed = evaluations[-1]["step"] if evaluations else 0
    if evaluations:
        best_eval = min(evaluations, key=lambda e: e["loss"])
        best_step = best_eval["step"]
    else:
        best_eval = None

    best_dir = EVAL_ROOT / "best"
    best_generation: dict[int, list[dict[str, Any]]] = {}
    if best_eval is not None:
        for num_steps in BEST_EULER_STEPS:
            folder = best_dir / f"euler_{num_steps:02d}"
            best_generation[num_steps] = generate_and_decode(model, codec, examples, mean, std, device, num_steps, INFERENCE_SEEDS, folder)

    multi_seed_rows: list[dict[str, Any]] = []
    if best_eval is not None:
        sanity_example = examples[0]
        sanity_folder = best_dir / "multi_seed"
        aligned = model.align((sanity_example.sequence,), (sanity_example.alignment_units,), (sanity_example.target_total_frames,))
        latents = []
        for seed in MULTI_SEED_EXTRA:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            x0 = torch.randn(1, aligned.states.shape[1], predictor_config.latent_width, generator=gen).to(device)
            with torch.no_grad():
                x1_pred_norm = euler_integrate(model.velocity_predictor, aligned, x0, CHECKPOINT_EULER_STEPS)
            x1_pred = (x1_pred_norm * (std + 1e-6) + mean)[0, : sanity_example.target_total_frames]
            waveform = decode_neucodec_latent(codec, x1_pred.cpu().numpy())
            path = sanity_folder / f"seed_{seed}.wav"
            integrity = c0.save_wave(path, waveform)
            latents.append(x1_pred)
            multi_seed_rows.append({
                "seed": seed,
                "utterance_id": sanity_example.utterance_id,
                "latent_sha256": sha256_tensor(x1_pred),
                "wav_sha256": sha256_file(ROOT / integrity["path"]),
                "waveform": integrity,
            })
        pairwise = []
        for i in range(len(latents)):
            for j in range(i + 1, len(latents)):
                diff = float(torch.sqrt(torch.mean((latents[i] - latents[j]).square())).item())
                pairwise.append({"seeds": [MULTI_SEED_EXTRA[i], MULTI_SEED_EXTRA[j]], "latent_rmse_difference": diff})
        multi_seed_pass = all(p["latent_rmse_difference"] > 1e-4 for p in pairwise) and len({r["latent_sha256"] for r in multi_seed_rows}) == len(multi_seed_rows)
    else:
        pairwise = []
        multi_seed_pass = False

    codec_hash_after = codec_state_hash(codec)
    if codec_hash_after != codec_hash_before:
        raise RuntimeError("F0 frozen NeuCodec decoder weights changed during this run")

    listening_rows = []
    for example, oracle_row in zip(examples, oracle_rows):
        row = {
            "source_id": example.utterance_id,
            "transcript": example.sequence.normalized_text,
            "source": str(c0.source_path(example.utterance_id).relative_to(ROOT)),
            "oracle": oracle_row["oracle_audio"]["path"],
        }
        for ev in evaluations:
            step = ev["step"]
            gen_row = next(g for g in ev["generation"] if g["utterance_id"] == example.utterance_id)
            row[f"step{step}"] = gen_row["waveform"]["path"]
        if best_eval is not None:
            for num_steps in BEST_EULER_STEPS:
                gen_row = next(g for g in best_generation[num_steps] if g["utterance_id"] == example.utterance_id)
                row[f"best_euler_{num_steps}"] = gen_row["waveform"]["path"]
        listening_rows.append(row)

    final_generation = evaluations[-1]["generation"] if evaluations else []
    both_final_non_speech = bool(final_generation) and all(
        (not row["waveform"]["finite"]) or (not row["waveform"]["non_silent"]) for row in final_generation
    )
    machine_classification = "FAIL" if both_final_non_speech else "HUMAN_REVIEW_REQUIRED"

    report = {
        "schema_version": "swara.f0.cfm_decoder_latent.v1",
        "status": "human_listening_required",
        "seed": SEED,
        "utterances": oracle_rows,
        "target": {
            "description": "Distill-NeuCodec fc_post_a output consumed by CodecDecoderVocos(vq=False)",
            "shape": "[B,T,1024]",
            "codec_model": NEUCODEC_ID,
            "codec_revision": NEUCODEC_REVISION,
            "statistics": stats_report,
            "normalization": "per-channel train-only standardization: (x - mean_c) / (std_c + eps)",
        },
        "flow": {
            "formulation": "straight-line conditional flow matching",
            "xt": "(1-t)*x0 + t*x1",
            "v_target": "x1 - x0",
            "loss": "MSE(v_pred, v_target), padded frames excluded",
        },
        "model": {
            "hidden_width": predictor_config.hidden_width,
            "layers": predictor_config.layers,
            "heads": predictor_config.heads,
            "ffn_dim": predictor_config.ffn_dim,
            "additive_conditioning_only": True,
            "causal": False,
            "cross_attention": False,
            "adaln_zero": False,
            "cfg": False,
        },
        "parameters": {"flow_params": flow_parameters, "total_trainable": total_parameters},
        "benchmark": {
            "seconds_per_step": bench_seconds,
            "estimated_1000_step_seconds": estimated_1000_step_seconds,
            "warmup_steps": WARMUP_BENCH_STEPS,
            "measured_steps": MEASURED_BENCH_STEPS,
            "device": device_used,
        },
        "training": {
            "device": device_used,
            "steps_completed": steps_completed,
            "maximum_steps": MAX_STEPS,
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "best_step": best_step,
            "final_loss": evaluations[-1]["loss"] if evaluations else None,
            "wall_seconds": wall_seconds,
            "stop_reason": stop_reason,
        },
        "evaluations": evaluations,
        "best_checkpoint": {
            "step": best_step,
            "selection_rule": "minimum training flow-velocity MSE among evaluated checkpoints "
                               "(operational selection only -- not a success criterion)",
            "euler_generations": best_generation,
        },
        "multi_seed_sanity": {
            "utterance_id": examples[0].utterance_id if best_eval is not None else None,
            "seeds": list(MULTI_SEED_EXTRA),
            "rows": multi_seed_rows,
            "pairwise_latent_rmse_difference": pairwise,
            "result": "PASS" if multi_seed_pass else ("FAIL" if best_eval is not None else "NOT_RUN"),
        },
        "oracle_audio_generated": True,
        "codec_frozen_hash_before": codec_hash_before,
        "codec_frozen_hash_after": codec_hash_after,
        "codec_weights_unchanged": codec_hash_after == codec_hash_before,
        "listening_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "human_listening_required": True,
        "machine_classification": machine_classification,
        "generalization_tested": False,
        "flow_matching": True,
        "autoregression": False,
        "fsq": False,
        "categorical_prediction": False,
        "adversarial": False,
        "codec_modified": False,
        "commit_push": False,
    }
    write_json(REPORT_PATH, report)

    manifest_lines = [
        "# F0 CFM Decoder Latent Listening Manifest", "",
        "| source_id | transcript | source | oracle | " + " | ".join(f"step{s}" for s in EVALUATION_STEPS)
        + " | best_euler_4 | best_euler_8 | best_euler_16 |",
        "|" + "---|" * (4 + len(EVALUATION_STEPS) + 3),
    ]
    for row in listening_rows:
        cells = [row.get(f"step{s}", "MISSING") for s in EVALUATION_STEPS]
        cells += [row.get(f"best_euler_{n}", "MISSING") for n in BEST_EULER_STEPS]
        manifest_lines.append(
            f"| `{row['source_id']}` | {row['transcript']} | `{row['source']}` | `{row['oracle']}` | "
            + " | ".join(f"`{c}`" for c in cells) + " |"
        )
    manifest_lines.append("")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(manifest_lines), encoding="utf-8")

    research_lines = [
        "# F0 CFM Decoder Latent V1", "",
        "Status: machine run complete; human listening is primary.", "",
        "## Frozen scope", "",
        f"- Seed: `{SEED}`",
        f"- Utterances: `{SELECTED_IDS[0]}`, `{SELECTED_IDS[1]}` (2-utterance memorization only; no generalization claim)",
        "- Target: frozen Distill-NeuCodec `fc_post_a` output `[T,1024]`, decoded via `CodecDecoderVocos(vq=False)`",
        "- Flow: straight-line conditional flow matching, MSE(v_pred, x1-x0), padded frames excluded",
        f"- Model: {predictor_config.layers} non-causal Transformer blocks, hidden={predictor_config.hidden_width}, "
        f"heads={predictor_config.heads}, FFN={predictor_config.ffn_dim}, additive conditioning only",
        "- No FSQ, no autoregression, no CFG, no adversarial/diffusion loss, no speaker/style conditioning", "",
        "## Runtime", "",
        f"- Benchmark: `{bench_seconds:.4f}`s/step ({WARMUP_BENCH_STEPS} warmup + {MEASURED_BENCH_STEPS} measured, real forward/loss/backward/step)",
        f"- Estimated 1000-step runtime: `{estimated_1000_step_seconds/60:.2f}` minutes on `{device_used}`",
        f"- Device used for training: `{device_used}`",
        f"- Steps completed: `{steps_completed}` / `{MAX_STEPS}`; wall time `{wall_seconds:.1f}`s",
        f"- Stop reason: `{stop_reason}`", "",
        "## Flow loss", "",
        f"- Initial: `{initial_loss:.6f}`" if initial_loss is not None else "- Initial: n/a",
        f"- Best: `{best_loss:.6f}` at step `{best_step}`",
        f"- Final: `{evaluations[-1]['loss']:.6f}`" if evaluations else "- Final: n/a", "",
        "## Generated-vs-real latent comparison (final evaluated checkpoint)", "",
    ]
    if evaluations:
        for c in evaluations[-1]["comparison"]:
            research_lines.append(
                f"- `{c['utterance_id']}`: cosine `{c['latent_cosine']:.4f}`, normalized RMSE `{c['normalized_rmse']:.4f}`, "
                f"temporal derivative error `{c['temporal_derivative_error']:.4f}`"
            )
    research_lines += ["", "## Multi-seed sanity check", "",
        f"Result: `{report['multi_seed_sanity']['result']}`",
        f"Utterance: `{report['multi_seed_sanity']['utterance_id']}`, seeds `{list(MULTI_SEED_EXTRA)}`", ""]
    for p in pairwise:
        research_lines.append(f"- seeds {p['seeds']}: latent RMSE difference `{p['latent_rmse_difference']:.6f}`")
    research_lines += ["", f"Frozen decoder weights unchanged across the run: `{report['codec_weights_unchanged']}`", "",
        "## Listening gate", "",
        "Machine checks establish finite/non-silent audio and generated-vs-real latent similarity only. "
        "They do not establish intelligibility or transcript match, and latent similarity is explicitly NOT "
        "the success criterion. Human listening decides F0 PASS/PARTIAL/FAIL.",
        f"Listen under `{EVAL_ROOT.relative_to(ROOT)}` using `{MANIFEST_PATH.relative_to(ROOT)}`.", "",
        f"F0 remains `{machine_classification}` until that review is supplied.", "",
        "Generalization tested: NO  ",
        "Codec modified: NO  ",
        "Commit/push: NO", "",
    ]
    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PATH.write_text("\n".join(research_lines), encoding="utf-8")

    status_word = "STOPPED_EARLY" if stop_reason != "maximum_steps" else "COMPLETE"
    print(f"F0_{status_word} steps={steps_completed} best_step={best_step} wall_seconds={wall_seconds:.1f}")


if __name__ == "__main__":
    main()
