"""Bounded four-example real token memorization experiment for M3B only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch

from swara.adapters.qwen_codec import Qwen12HzCodecAdapter
from swara.contracts import AudioTokenSequence, AudioTokenSpec, AudioWaveform, GenerationOptions
from swara.frontend.spans import TextSpan
from swara.frontend.tokenizer import LinguisticSequence, LinguisticToken, LinguisticTokenKind
from swara.models.generator import GeneratorConfig, LearnedSpeakerConditioner, SwaraSpeechGenerator
from swara.models.linguistic import LinguisticVocabulary
from swara.models.training import compute_token_losses


TRAINING_IDS = ("001", "005", "006", "014")
SEED = 20260822


def load_examples(dataset_root: Path) -> list[dict[str, Any]]:
    records = {
        record["example_id"]: record
        for record in (json.loads(line) for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    }
    if set(TRAINING_IDS) - set(records):
        raise ValueError("selected M3B examples are missing from manifest")
    examples: list[dict[str, Any]] = []
    for example_id in TRAINING_IDS:
        record = records[example_id]
        tokens = np.load(dataset_root / record["codec_token_path"], allow_pickle=False)
        if tokens.ndim != 2 or tokens.shape[1] != 16 or tokens.size == 0:
            raise ValueError(f"invalid cached codec tokens for {example_id}")
        if int(tokens.min()) < 0 or int(tokens.max()) > 2047:
            raise ValueError(f"codec tokens out of range for {example_id}")
        linguistic_tokens = []
        for item in record["linguistic_tokens"]:
            source_range = item["source_range"]
            source_span = None if source_range is None else TextSpan(source_range[0], source_range[1])
            linguistic_tokens.append(
                LinguisticToken(
                    LinguisticTokenKind(item["kind"]),
                    item["value"],
                    item["language"],
                    source_span,
                    source_span,
                    item["override_id"],
                )
            )
        sequence = LinguisticSequence(
            record["linguistic_schema_version"], record["transcript"], record["transcript"], tuple(linguistic_tokens), ()
        )
        examples.append({"record": record, "sequence": sequence, "targets": torch.from_numpy(tokens.astype(np.int64))})
    speakers = {example["record"]["speaker_id"] for example in examples}
    if len(speakers) != 1:
        raise ValueError("M3B selected examples must have one speaker")
    return examples


def build_checkpoint(model: SwaraSpeechGenerator, config: GeneratorConfig, vocabulary: LinguisticVocabulary, speaker_ids: tuple[str, ...]) -> dict[str, Any]:
    config_data = asdict(config)
    config_data["audio_spec"] = asdict(config.audio_spec)
    return {
        "format": "swara.m3b-real-overfit.v0",
        "model_state": model.module.state_dict(),
        "config": config_data,
        "vocabulary": vocabulary.to_dict(),
        "speaker_ids": speaker_ids,
    }


def restore_checkpoint(path: Path) -> SwaraSpeechGenerator:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config_data = dict(checkpoint["config"])
    config_data["audio_spec"] = AudioTokenSpec(**config_data["audio_spec"])
    config = GeneratorConfig(**config_data)
    vocabulary = LinguisticVocabulary.from_dict(checkpoint["vocabulary"])
    conditioner = LearnedSpeakerConditioner(tuple(checkpoint["speaker_ids"]))
    model = SwaraSpeechGenerator(config, vocabulary, conditioner)
    model.module.load_state_dict(checkpoint["model_state"])
    return model.eval()


def evaluate(model: SwaraSpeechGenerator, examples: list[dict[str, Any]], speaker_index: int = 0) -> dict[str, float]:
    model.eval()
    primary_loss = residual_loss = 0.0
    primary_correct = residual_correct = primary_total = residual_total = 0
    with torch.no_grad():
        for example in examples:
            targets = example["targets"].unsqueeze(0)
            text_ids = model.encode_linguistic(example["sequence"])
            speakers = torch.tensor([speaker_index], dtype=torch.long)
            inputs = model.teacher_forcing_inputs(targets)
            primary, residual, _ = model.forward(text_ids, speakers, inputs, primary_tokens_for_residual=targets[:, :, 0])
            losses = compute_token_losses(primary, residual, targets)
            primary_loss += float(losses.primary)
            residual_loss += float(losses.residual)
            primary_prediction = primary.argmax(dim=-1)
            residual_prediction = residual.argmax(dim=-1)
            primary_correct += int((primary_prediction == targets[:, :, 0]).sum())
            residual_correct += int((residual_prediction == targets[:, :, 1:]).sum())
            primary_total += targets.shape[0] * targets.shape[1]
            residual_total += targets.shape[0] * targets.shape[1] * (targets.shape[2] - 1)
    primary_accuracy = primary_correct / primary_total
    residual_accuracy = residual_correct / residual_total
    return {
        "primary_loss": primary_loss / len(examples),
        "residual_loss": residual_loss / len(examples),
        "total_loss": (primary_loss + residual_loss) / len(examples),
        "primary_accuracy": primary_accuracy,
        "residual_accuracy": residual_accuracy,
        "overall_token_accuracy": (primary_correct + residual_correct) / (primary_total + residual_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/m3_real_speech_v0"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/m3b_real_overfit_v0"))
    parser.add_argument("--codec-path", type=Path, default=Path("models/qwen3-tts-tokenizer-12hz"))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1000:
        raise ValueError("max steps must be within 1..1000")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("run directory must be absent or empty; refusing to overwrite artifacts")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    start = time.monotonic()
    examples = load_examples(args.dataset)
    info = json.loads((args.dataset / "dataset_info.json").read_text(encoding="utf-8"))
    spec = AudioTokenSpec(**{
        "version": info["codec"]["spec_version"],
        "codebook_count": info["codec"]["codebook_count"],
        "vocabulary_size": info["codec"]["vocabulary_size"],
        "frame_rate_hz": info["codec"]["frame_rate_hz"],
    })
    vocabulary = LinguisticVocabulary.build(tuple(example["sequence"] for example in examples))
    speaker_ids = (info["speaker_id"],)
    conditioner = LearnedSpeakerConditioner(speaker_ids)
    # M2B's architecture is unchanged. Only its four-frame smoke bound is
    # expanded to the real selected clips' known maximum target length.
    config = GeneratorConfig(
        linguistic_vocab_size=vocabulary.size,
        speaker_count=1,
        audio_spec=spec,
        model_dim=256,
        layers=4,
        heads=4,
        ffn_dim=512,
        max_text_tokens=max(len(example["sequence"].tokens) for example in examples),
        max_audio_frames=max(int(example["targets"].shape[0]) for example in examples),
    )
    model = SwaraSpeechGenerator(config, vocabulary, conditioner).train()
    args.run_dir.mkdir(parents=True)
    initial_checkpoint = args.run_dir / "initial.pt"
    best_checkpoint = args.run_dir / "best.pt"
    final_checkpoint = args.run_dir / "final.pt"
    torch.save(build_checkpoint(model, config, vocabulary, speaker_ids), initial_checkpoint)
    initial = evaluate(model, examples)
    best = initial
    torch.save(build_checkpoint(model, config, vocabulary, speaker_ids), best_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    step = 0
    stop_reason = "maximum step cap reached"
    for step in range(1, args.max_steps + 1):
        example = examples[(step - 1) % len(examples)]
        targets = example["targets"].unsqueeze(0)
        inputs = model.teacher_forcing_inputs(targets)
        primary, residual, _ = model.forward(
            model.encode_linguistic(example["sequence"]),
            torch.tensor([0], dtype=torch.long),
            inputs,
            primary_tokens_for_residual=targets[:, :, 0],
        )
        losses = compute_token_losses(primary, residual, targets)
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        optimizer.step()
        if step % 25 == 0 or step == args.max_steps:
            metrics = evaluate(model, examples)
            if metrics["total_loss"] < best["total_loss"]:
                best = metrics
                torch.save(build_checkpoint(model, config, vocabulary, speaker_ids), best_checkpoint)
            if metrics["primary_accuracy"] >= 0.95 and metrics["residual_accuracy"] >= 0.90 and metrics["overall_token_accuracy"] >= 0.90:
                stop_reason = "all M3B overfit accuracy thresholds met"
                break
    final = evaluate(model, examples)
    torch.save(build_checkpoint(model, config, vocabulary, speaker_ids), final_checkpoint)
    if not (final["total_loss"] < initial["total_loss"] and final["primary_accuracy"] >= 0.95 and final["residual_accuracy"] >= 0.90 and final["overall_token_accuracy"] >= 0.90):
        raise RuntimeError(json.dumps({"initial": initial, "final": final, "steps": step}, sort_keys=True))

    # Serialization/load gate: reload the best checkpoint before inference.
    restored = restore_checkpoint(best_checkpoint)
    serialization_metrics = evaluate(restored, examples)
    codec = Qwen12HzCodecAdapter.from_local_path(args.codec_path)
    if codec.spec != spec:
        raise RuntimeError("M2A codec spec is incompatible with this M3B run")
    generated_dir = args.run_dir / "generated"
    generated_dir.mkdir()
    listening = []
    speaker = conditioner.resolve(info["speaker_id"])
    for example in examples:
        record = example["record"]
        target_frames = int(example["targets"].shape[0])
        max_duration_ms = math.ceil(target_frames / spec.frame_rate_hz * 1000)
        generated = restored.generate(
            example["sequence"], speaker, generation=GenerationOptions(seed=SEED, deterministic=True, max_duration_ms=max_duration_ms)
        )
        generated.validate_against(spec)
        waveform = codec.decode(generated)
        samples = np.asarray(waveform.samples, dtype=np.float32)
        if samples.size == 0 or not np.isfinite(samples).all() or waveform.sample_rate_hz != 24000:
            raise RuntimeError(f"invalid decoded waveform for {record['example_id']}")
        output_path = generated_dir / f"{record['example_id']}_generated.wav"
        sf.write(output_path, samples, waveform.sample_rate_hz, subtype="PCM_16")
        listening.append({
            "example_id": record["example_id"],
            "transcript": record["transcript"],
            "target_wav": f"../../data/m3_real_speech_v0/{record['audio_path']}",
            "generated_wav": f"generated/{output_path.name}",
            "target_token_frames": target_frames,
            "generated_token_frames": len(generated.frames),
            "training_set_status": "training",
            "generation_stop": "known target frame length for bounded M3B overfit only",
        })
    elapsed = time.monotonic() - start
    summary = {
        "schema_version": "swara.m3b-real-overfit.v0",
        "seed": SEED,
        "training_ids": list(TRAINING_IDS),
        "model_config": {**asdict(config), "audio_spec": asdict(spec)},
        "parameter_count": model.parameter_count,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "steps": step,
        "max_steps": args.max_steps,
        "initial_losses": initial,
        "final_losses": final,
        "best_losses": best,
        "serialization_reload_losses": serialization_metrics,
        "early_stop_reason": stop_reason,
        "wall_clock_seconds": elapsed,
        "checkpoint_files": [initial_checkpoint.name, best_checkpoint.name, final_checkpoint.name],
        "duration_handling": "Known target frame length is used only for M3B autoregressive reconstruction; no duration model exists.",
    }
    (args.run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.run_dir / "listening_manifest.json").write_text(json.dumps(listening, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"final": final, "initial": initial, "parameter_count": model.parameter_count, "steps": step, "wall_clock_seconds": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
