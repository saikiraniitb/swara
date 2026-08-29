#!/usr/bin/env python3
"""Run the bounded Stage2D.4 Colab dry-run contract.

This entry point intentionally has no training mode. ``--dry-run`` loads the
frozen Qwen model and Stage2B step025 bridge/gate, executes three bounded
teacher-forced probes (one positive, one targeted native, one general native),
performs one real mixed-loss backward, optionally performs one disposable
optimizer step, restores the probe state, and writes only compact JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import soundfile as sf
import torch
from torch import Tensor, nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT", str(REPO_ROOT))).resolve()
MODEL_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_MODEL_ROOT", str(BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base"))).resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_codec import Qwen12HzCodecAdapter
from swara.adapters.qwen_stage2b_training import build_qwen_teacher_forced_schedule, run_qwen_teacher_forced_schedule
from swara.adapters.qwen_tts import QwenFoundationTTS
from swara.frontend import Frontend
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation
from swara.training.stage2b_pronunciation import qwen_acoustic_tokens_tensor
from swara.training.stage2d4_training import (
    BASE_STEP025_SHA256, Stage2D4Dataset, build_mixed_batch, compute_stage2d4_v1_loss,
    build_deterministic_epoch_batches, compute_trajectory_metrics, load_step025_initialization, qwen_parameters_frozen,
    run_graph_dry_run, save_stage2d4_checkpoint, set_trainable_phase,
)

RUN_ID = "stage2d4_v1_medium_dry_run"
DESIGN_ROOT = REPO_ROOT / "artifacts/stage2d/stage2d4_training_design"
CONFIG_PATH = REPO_ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_training_config.json"
INVENTORY_PATH = REPO_ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
CACHE_ROOT = REPO_ROOT / "data/stage2d_spicor_selected_audio"
DEFAULT_CHECKPOINT = BUNDLE_ROOT / "run_artifacts/stage2b4b_pronunciation_v0/checkpoints/step025.pt"
REFERENCE_AUDIO_NAME = "IISc_SPICORProject_EN_M_AGRI_116.wav"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_waveform(path: Path) -> tuple[Tensor, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim != 1 or int(sample_rate) != 24000:
        raise RuntimeError(f"dry-run audio must be mono 24 kHz: {path}")
    return torch.from_numpy(samples), int(sample_rate)


def sample_audio_path(sample: Any) -> Path:
    path = Path(sample.audio_resolver_path)
    if path.is_file():
        return path
    candidate = CACHE_ROOT / "archive" / path.name
    if candidate.is_file():
        return candidate
    candidate = REPO_ROOT / path
    if candidate.is_file():
        return candidate
    raise RuntimeError(f"dry-run audio path is not materialized: {sample.utterance_id} ({sample.audio_resolver_path})")


def build_sample_representation(sample: Any) -> Any:
    overrides: tuple[PronunciationOverride, ...] = ()
    if sample.is_positive:
        assert sample.target_char_span is not None and sample.phone_sequence is not None
        start, end = sample.target_char_span
        overrides = (PronunciationOverride(start, end, "swara-phones-v0", sample.phone_sequence, "en-IN", source="lexicon"),)
    request = SynthesisRequest(Content(sample.transcript, "en-IN"), SpeakerRef("stage2b4b-frozen-speaker"), PronunciationInput(overrides=overrides))
    return build_stage2b_representation(Frontend().compile(request))


def target_frame_range(sample: Any, total_frames: int) -> tuple[int, int]:
    """Make a non-empty dry-run target mask from the immutable text span.

    Stage2D.4 rows carry text spans, not invented acoustic alignments. This
    bounded proxy validates the real CE graph without changing production data.
    """
    if not sample.is_positive:
        return (0, 0)
    assert sample.target_char_span is not None
    start_char, end_char = sample.target_char_span
    text_length = max(1, len(sample.transcript))
    start = min(total_frames - 1, max(0, int(start_char * total_frames / text_length)))
    end = min(total_frames, max(start + 1, int((end_char * total_frames + text_length - 1) / text_length)))
    return (start, end)


def pad_first_dim(values: list[Tensor], max_frames: int) -> Tensor:
    padded = []
    for value in values:
        amount = max_frames - value.shape[1]
        if value.ndim == 3:
            padded.append(F.pad(value, (0, 0, 0, amount)))
        elif value.ndim == 4:
            padded.append(F.pad(value, (0, 0, 0, 0, 0, amount)))
        else:
            raise RuntimeError(f"unsupported dry-run tensor rank: {value.ndim}")
    return torch.cat(padded, dim=0)


def environment_report(device: torch.device, dtype_name: str, output_dir: Path) -> dict[str, Any]:
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": __import__("transformers").__version__, "numpy": __import__("numpy").__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda,
        "dtype": dtype_name, "output_dir": str(output_dir), "model_root": str(MODEL_ROOT), "bundle_root": str(BUNDLE_ROOT),
    }


def source_git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable_in_bundle"


def run_full_training(
    *, args: argparse.Namespace, config: dict[str, Any], dataset: Stage2D4Dataset, model: Any, codec: Any,
    bridge: nn.Module, gate: nn.Parameter, optimizer: torch.optim.Optimizer, qwen_parameters: tuple[nn.Parameter, ...],
    qwen_device: torch.device, speaker_condition: Tensor, source_git_commit: str, evaluation_contract_sha256: str,
) -> dict[str, Any]:
    """Run the frozen 4-epoch/64-step schedule when explicitly requested."""

    if config["sampling"] != {
        "batch_size": 8, "positive_oversampling": False,
        "policy": "deterministic structured batches; positives are distributed round-robin across epoch batches",
        "epochs": 4, "warmup_optimizer_steps": 5, "estimated_optimizer_steps": 64,
    }:
        raise RuntimeError("Stage2D.4 V1 MEDIUM sampling schedule was mutated")
    all_samples = tuple(dataset.train_samples)
    representations = {sample.sample_id: build_sample_representation(sample) for sample in all_samples}
    tensorizers = {
        sample.sample_id: Stage2BLinguisticTensorizer.from_representations((representations[sample.sample_id],)).to(qwen_device).eval()
        for sample in all_samples
    }
    target_codes: dict[str, Tensor] = {}
    for sample in all_samples:
        audio, rate = load_waveform(sample_audio_path(sample))
        waveform = type("Waveform", (), {"samples": tuple(float(x) for x in audio.tolist()), "sample_rate_hz": rate})()
        target_codes[sample.sample_id] = qwen_acoustic_tokens_tensor(codec.encode(waveform), codec.spec).to(device=qwen_device, dtype=torch.long)
        del audio, waveform
    optimizer.zero_grad(set_to_none=True)
    set_trainable_phase(bridge, gate, "gate_warmup")
    checkpoint_steps = {0, 5, 32, 64}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    save_stage2d4_checkpoint(
        checkpoint_dir / "step000.pt", step=0, bridge=bridge, gate=gate, optimizer=optimizer,
        dataset_sha256=dataset.dataset_sha256, config=config, source_git_commit=source_git_commit,
        evaluation_contract_sha256=evaluation_contract_sha256,
    )
    losses: list[dict[str, Any]] = []
    for step in range(1, 65):
        epoch = (step - 1) // 16
        batch_index = (step - 1) % 16
        epoch_batches = build_deterministic_epoch_batches(all_samples, batch_size=8, seed=int(config["seed"]), epoch=epoch)
        batch_samples = epoch_batches[batch_index]
        batch_contract = build_mixed_batch(batch_samples)
        selected = list(batch_samples)
        codes = [target_codes[sample.sample_id] for sample in selected]
        max_frames = max(int(value.shape[0]) for value in codes)
        conditioned_main: list[Tensor] = []
        native_main: list[Tensor] = []
        conditioned_residual: list[Tensor] = []
        native_residual: list[Tensor] = []
        target_ranges: list[tuple[tuple[int, int], ...]] = []
        for sample, sample_codes in zip(selected, codes):
            target = sample_codes.unsqueeze(0)
            representation = representations[sample.sample_id]
            native_schedule = build_qwen_teacher_forced_schedule(model._model, text=representation.source_text, language="English", speaker_condition=speaker_condition, target_acoustic_codes=target)
            with torch.no_grad():
                native_output = run_qwen_teacher_forced_schedule(getattr(model._model, "model", model._model).talker, native_schedule)
            tensorized = tensorizers[sample.sample_id]((representation,))
            conditioned_schedule = build_qwen_teacher_forced_schedule(
                model._model, text=representation.source_text, language="English", speaker_condition=speaker_condition,
                target_acoustic_codes=target, stage2b_representation=representation, stage2b_tensorized=tensorized,
                stage2b_bridge=bridge, gate=gate,
            )
            conditioned_output = run_qwen_teacher_forced_schedule(getattr(model._model, "model", model._model).talker, conditioned_schedule)
            native_main.append(native_output.main_logits)
            conditioned_main.append(conditioned_output.main_logits)
            native_residual.append(native_output.residual_logits)
            conditioned_residual.append(conditioned_output.residual_logits)
            target_ranges.append((target_frame_range(sample, int(sample_codes.shape[0])),) if sample.is_positive else ())
        padded_targets = [F.pad(value, (0, 0, 0, max_frames - value.shape[0])).unsqueeze(0) for value in codes]
        valid_masks = [torch.cat((torch.ones(value.shape[0], dtype=torch.bool, device=qwen_device), torch.zeros(max_frames - value.shape[0], dtype=torch.bool, device=qwen_device))).unsqueeze(0) for value in codes]
        losses_value = compute_stage2d4_v1_loss(
            pad_first_dim(conditioned_main, max_frames), pad_first_dim(native_main, max_frames),
            pad_first_dim(conditioned_residual, max_frames), pad_first_dim(native_residual, max_frames),
            torch.cat(padded_targets, dim=0), batch_contract, target_ranges, valid_acoustic_mask=torch.cat(valid_masks, dim=0),
            lambda_preserve=float(config["loss"]["lambda_preserve"]), lambda_eos=float(config["loss"]["lambda_eos"]),
        )
        optimizer.zero_grad(set_to_none=True)
        losses_value.total.backward()
        if not qwen_parameters_frozen(qwen_parameters):
            raise RuntimeError(f"Qwen received gradients at optimizer step {step}")
        if not any(parameter.grad is not None for parameter in (gate, *bridge.parameters())):
            raise RuntimeError(f"no trainable gradient at optimizer step {step}")
        optimizer.step()
        losses.append({"step": step, "epoch": epoch, "batch": batch_index, "target_ce": float(losses_value.target_ce.detach().item()), "preservation_kl": float(losses_value.preservation_kl.detach().item()), "total": float(losses_value.total.detach().item()), "phase": "gate_warmup" if step <= 5 else "bridge_and_gate"})
        if step == 5:
            set_trainable_phase(bridge, gate, "bridge_and_gate")
        if step in checkpoint_steps:
            save_stage2d4_checkpoint(
                checkpoint_dir / f"step{step:03d}.pt", step=step, bridge=bridge, gate=gate, optimizer=optimizer,
                dataset_sha256=dataset.dataset_sha256, config=config, source_git_commit=source_git_commit,
                evaluation_contract_sha256=evaluation_contract_sha256,
            )
        del conditioned_main, native_main, conditioned_residual, native_residual, padded_targets, valid_masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {"status": "PASS", "training_performed": True, "optimizer_steps": 64, "checkpoint_steps": [0, 5, 32, 64], "dataset_sha256": dataset.dataset_sha256, "source_git_commit": source_git_commit, "evaluation_contract_sha256": evaluation_contract_sha256, "step025_sha256": BASE_STEP025_SHA256, "qwen_weights_included": False, "losses": losses}
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true", help="execute the frozen 64-step schedule")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("SWARA_STAGE2D4_OUTPUT_DIR", str(BUNDLE_ROOT / "run_artifacts" / RUN_ID))))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_ROOT)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--archive", type=Path, default=Path("/nonexistent/spicor.tar.gz"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--probe-max-frames", type=int, default=24)
    parser.add_argument("--no-disposable-step", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run == args.train:
        raise SystemExit("pass exactly one of --dry-run or --train")
    if args.probe_max_frames <= 0:
        raise SystemExit("--probe-max-frames must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("mask_mode") != "target_context_1":
        raise SystemExit("Stage2D.4 dry-run requires frozen mask_mode=target_context_1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(int(os.environ.get("SWARA_TORCH_THREADS", "2")))
    torch.manual_seed(int(config["seed"]))
    dtype_name = os.environ.get("SWARA_STAGE2B4B_DTYPE", "float32")
    if dtype_name != "float32":
        raise SystemExit("Stage2D.4 dry-run is float32-only; set SWARA_STAGE2B4B_DTYPE=float32")

    dataset = Stage2D4Dataset.from_design(
        args.design_dir, repo_root=REPO_ROOT, inventory_path=args.inventory, archive_path=args.archive,
        cache_root=args.cache_root, training_only=True,
    )
    targeted = next(sample for sample in dataset.native_train_samples if sample.supervision_type == "NATIVE_PRESERVATION_TARGETED")
    general = next(sample for sample in dataset.native_train_samples if sample.supervision_type == "NATIVE_PRESERVATION")
    batches = build_mixed_batch((dataset.positive_train_samples[0], targeted, general))
    selected_samples = list(batches.samples)
    if len(selected_samples) != 3:
        raise RuntimeError("dry-run sample plan must contain exactly three samples")
    if not args.checkpoint.is_file():
        raise RuntimeError(f"frozen step025 checkpoint is required but was not found: {args.checkpoint}")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != str(config["base_checkpoint"]["sha256"]) or checkpoint_sha != BASE_STEP025_SHA256:
        raise RuntimeError(f"step025 SHA256 mismatch: {checkpoint_sha}")

    device_map = os.environ.get("SWARA_STAGE2B4B_DEVICE_MAP", "cuda:0" if torch.cuda.is_available() else "cpu")
    model = QwenFoundationTTS.from_local_path(MODEL_ROOT, reference_audio=str(BUNDLE_ROOT / "data" / "source_audio" / REFERENCE_AUDIO_NAME), device_map=device_map, dtype=torch.float32)
    native_model = getattr(model._model, "model", model._model)
    talker = native_model.talker
    native_model.eval()
    qwen_parameters = tuple(native_model.parameters())
    for parameter in qwen_parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    qwen_device = next(talker.parameters()).device
    codec = Qwen12HzCodecAdapter.from_local_path(MODEL_ROOT / "speech_tokenizer")
    reference_audio = str(BUNDLE_ROOT / "data" / "source_audio" / REFERENCE_AUDIO_NAME)
    speaker_prompt = model._model.create_voice_clone_prompt(ref_audio=reference_audio, x_vector_only_mode=True)
    speaker_condition = speaker_prompt[0].ref_spk_embedding
    if speaker_condition.ndim == 1:
        speaker_condition = speaker_condition.unsqueeze(0)
    speaker_condition = speaker_condition.detach().to(device=qwen_device, dtype=torch.float32)
    representations = tuple(build_sample_representation(sample) for sample in selected_samples)
    tensorizer = Stage2BLinguisticTensorizer.from_representations(representations).to(qwen_device).eval()
    for parameter in tensorizer.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    hidden_size = int(talker.config.hidden_size)
    bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, hidden_size, initialization_seed=int(config["seed"]))).to(qwen_device)
    gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=qwen_device))
    initialization = load_step025_initialization(args.checkpoint, bridge, gate)
    phase = "bridge_and_gate"
    trainable_names = set_trainable_phase(bridge, gate, phase)
    bridge.train()
    optimizer = torch.optim.AdamW([
        {"params": [gate], "lr": float(config["optimizer"]["gate_learning_rate"]), "weight_decay": float(config["optimizer"]["weight_decay"])},
        {"params": list(bridge.parameters()), "lr": float(config["optimizer"]["bridge_learning_rate"]), "weight_decay": float(config["optimizer"]["weight_decay"])},
    ])
    target_codes: list[Tensor] = []
    target_ranges: list[tuple[tuple[int, int], ...]] = []
    for sample in selected_samples:
        audio, rate = load_waveform(sample_audio_path(sample))
        waveform = type("Waveform", (), {"samples": tuple(float(x) for x in audio.tolist()), "sample_rate_hz": rate})()
        codes = qwen_acoustic_tokens_tensor(codec.encode(waveform), codec.spec)[:args.probe_max_frames].to(device=qwen_device, dtype=torch.long)
        target_codes.append(codes)
        target_ranges.append((target_frame_range(sample, int(codes.shape[0])),) if sample.is_positive else ())

    if args.train:
        evaluation_contract_path = REPO_ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_evaluation_contract.json"
        run_full_training(
            args=args, config=config, dataset=dataset, model=model, codec=codec, bridge=bridge, gate=gate,
            optimizer=optimizer, qwen_parameters=qwen_parameters, qwen_device=qwen_device, speaker_condition=speaker_condition,
            source_git_commit=source_git_sha(), evaluation_contract_sha256=sha256_file(evaluation_contract_path),
        )
        return 0

    trace_holder: dict[str, Any] = {}

    def forward_loss() -> Tensor:
        conditioned_main: list[Tensor] = []
        native_main: list[Tensor] = []
        conditioned_residual: list[Tensor] = []
        native_residual: list[Tensor] = []
        evaluation_pairs: list[tuple[Tensor, Tensor]] = []
        max_frames = max(int(codes.shape[0]) for codes in target_codes)
        for sample, representation, codes in zip(selected_samples, representations, target_codes):
            target = codes.unsqueeze(0)
            native_schedule = build_qwen_teacher_forced_schedule(model._model, text=representation.source_text, language="English", speaker_condition=speaker_condition, target_acoustic_codes=target)
            with torch.no_grad():
                native_output = run_qwen_teacher_forced_schedule(talker, native_schedule)
            if sample.is_positive:
                tensorized = tensorizer((representation,))
                conditioned_schedule = build_qwen_teacher_forced_schedule(
                    model._model, text=representation.source_text, language="English", speaker_condition=speaker_condition,
                    target_acoustic_codes=target, stage2b_representation=representation, stage2b_tensorized=tensorized,
                    stage2b_bridge=bridge, gate=gate,
                )
                conditioned_output = run_qwen_teacher_forced_schedule(talker, conditioned_schedule)
            else:
                conditioned_output = native_output
            conditioned_main.append(conditioned_output.main_logits)
            native_main.append(native_output.main_logits)
            conditioned_residual.append(conditioned_output.residual_logits)
            native_residual.append(native_output.residual_logits)
            evaluation_pairs.append((native_output.main_logits.detach(), conditioned_output.main_logits.detach()))
        padded_target = []
        valid_masks = []
        for codes in target_codes:
            amount = max_frames - codes.shape[0]
            padded_target.append(F.pad(codes, (0, 0, 0, amount)).unsqueeze(0))
            valid_masks.append(torch.cat((torch.ones(codes.shape[0], dtype=torch.bool, device=qwen_device), torch.zeros(amount, dtype=torch.bool, device=qwen_device))).unsqueeze(0))
        mixed_losses = compute_stage2d4_v1_loss(
            pad_first_dim(conditioned_main, max_frames), pad_first_dim(native_main, max_frames),
            pad_first_dim(conditioned_residual, max_frames), pad_first_dim(native_residual, max_frames),
            torch.cat(padded_target, dim=0), batches, target_ranges, valid_acoustic_mask=torch.cat(valid_masks, dim=0),
            lambda_preserve=float(config["loss"]["lambda_preserve"]), lambda_eos=float(config["loss"]["lambda_eos"]),
        )
        trace_holder.update({"losses": mixed_losses, "max_frames": max_frames, "target_ranges": [list(item[0]) if item else [] for item in target_ranges], "teacher_forced_calls": 4, "evaluation_pairs": evaluation_pairs})
        return mixed_losses.total

    graph = run_graph_dry_run(forward_loss, trainable_parameters=(gate, *bridge.parameters()), qwen_parameters=qwen_parameters, optimizer=optimizer, perform_disposable_step=not args.no_disposable_step)
    if not qwen_parameters_frozen(qwen_parameters):
        raise RuntimeError("Qwen freeze contract failed after dry-run")
    losses = trace_holder["losses"]
    evaluation_rows = []
    for sample, (native_q0, conditioned_q0) in zip(selected_samples, trace_holder["evaluation_pairs"]):
        evaluation_rows.append({"sample_id": sample.sample_id, "supervision_type": sample.supervision_type, **compute_trajectory_metrics(native_q0, conditioned_q0), "trajectory_not_generated": True})
    report_base = {
        "run_id": RUN_ID, "status": "PASS", "elapsed_seconds": time.monotonic() - started, "dry_run": True,
        "training_performed": False, "qwen_generation_performed": False, "persistent_checkpoint_written": False,
        "optimizer_step_count": 0, "disposable_optimizer_step_reverted": graph["disposable_optimizer_step_reverted"],
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha, "initialized": initialization},
        "sample_plan": [{"sample_id": sample.sample_id, "utterance_id": sample.utterance_id, "supervision_type": sample.supervision_type, "audio": str(sample_audio_path(sample)), "frames": int(codes.shape[0])} for sample, codes in zip(selected_samples, target_codes)],
        "real_qwen_teacher_forced_call_count": trace_holder["teacher_forced_calls"], "real_backward_count": 1,
    }
    (args.output_dir / "dry_run_status.json").write_text(json.dumps(report_base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "dry_run_dataset_summary.json").write_text(json.dumps({"dataset_sha256": dataset.dataset_sha256, "train_count": len(dataset.train_samples), "positive_count": len(dataset.positive_train_samples), "targeted_native_count": sum(sample.supervision_type == "NATIVE_PRESERVATION_TARGETED" for sample in dataset.native_train_samples), "general_native_count": sum(sample.supervision_type == "NATIVE_PRESERVATION" for sample in dataset.native_train_samples), "gold_excluded": not any(sample.human_gold_reference for sample in dataset.train_samples), "mixed_batch_sample_ids": [sample.sample_id for sample in selected_samples]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "dry_run_loss_report.json").write_text(json.dumps({"target_ce": float(losses.target_ce.detach().item()), "preservation_kl": float(losses.preservation_kl.detach().item()), "eos_preservation": float(losses.eos_preservation.detach().item()), "total": float(losses.total.detach().item()), "lambda_preserve": float(config["loss"]["lambda_preserve"]), "lambda_eos": float(config["loss"]["lambda_eos"]), "target_codebooks": list(config["loss"]["target_codebooks"]), "target_ranges": trace_holder["target_ranges"], "valid_frames_padded_to": trace_holder["max_frames"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "dry_run_gradient_report.json").write_text(json.dumps({"phase": phase, "trainable_names": list(trainable_names), "trainable_gradient_count": graph["trainable_gradient_count"], "trainable_gradient_norm": graph["trainable_gradient_norm"], "qwen_gradient_norm": graph["qwen_gradient_norm"], "qwen_gradients_absent": graph["qwen_gradient_norm"] == 0.0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "dry_run_qwen_freeze_report.json").write_text(json.dumps({"qwen_parameters": len(qwen_parameters), "qwen_parameter_count": sum(parameter.numel() for parameter in qwen_parameters), "all_requires_grad_false": all(not parameter.requires_grad for parameter in qwen_parameters), "all_gradients_absent": qwen_parameters_frozen(qwen_parameters), "dtype": dtype_name, "device": str(qwen_device), "adapter_trace_api": {"build_schedule": True, "run_schedule": True}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "dry_run_evaluation_report.json").write_text(json.dumps({"evaluation_executed": True, "generation_executed": False, "trajectory_metrics": evaluation_rows, "note": "Tiny teacher-forced q0 evaluation plumbing only; no autoregressive Qwen generation is performed in this dry-run."}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "dry_run_environment.json").write_text(json.dumps(environment_report(qwen_device, dtype_name, args.output_dir), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report_base, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
