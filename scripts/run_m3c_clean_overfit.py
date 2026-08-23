"""One bounded all-20 M3C memorization run using corrected text-prefix conditioning."""

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
from swara.contracts import AudioTokenSpec, AudioWaveform, GenerationOptions
from swara.frontend.tokenizer import LinguisticSequence, LinguisticToken, LinguisticTokenKind
from swara.models.generator import GeneratorConfig, LearnedSpeakerConditioner, SwaraSpeechGenerator
from swara.models.linguistic import LinguisticVocabulary
from swara.models.training import compute_token_losses


SEED = 20260822
CONDITIONING_IDS = ("001", "005", "006", "014", "019")


def load_examples(dataset: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_ids = [f"{index:03d}" for index in range(1, 21)]
    if [row["example_id"] for row in rows] != expected_ids:
        raise ValueError("M3C manifest must contain ordered IDs 001 through 020")
    examples: list[dict[str, Any]] = []
    for row in rows:
        array = np.load(dataset / row["codec_token_path"], allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 16 or array.size == 0 or int(array.min()) < 0 or int(array.max()) > 2047:
            raise ValueError(f"invalid cached codec tokens: {row['example_id']}")
        linguistic = tuple(
            LinguisticToken(LinguisticTokenKind(item["kind"]), item["value"], item["language"], None, None, item["override_id"])
            for item in row["linguistic_tokens"]
        )
        sequence = LinguisticSequence("swara.linguistic.v0", row["transcript"], row["transcript"], linguistic, ())
        examples.append({"record": row, "sequence": sequence, "targets": torch.from_numpy(array.astype(np.int64))})
    if len({example["record"]["speaker_id"] for example in examples}) != 1:
        raise ValueError("M3C run requires one speaker/session ID")
    return examples


def checkpoint_payload(model: SwaraSpeechGenerator, config: GeneratorConfig, vocabulary: LinguisticVocabulary, speaker_ids: tuple[str, ...]) -> dict[str, Any]:
    config_data = asdict(config)
    config_data["audio_spec"] = asdict(config.audio_spec)
    return {"format": "swara.m3c-clean-overfit.v0", "model_state": model.module.state_dict(), "config": config_data, "vocabulary": vocabulary.to_dict(), "speaker_ids": speaker_ids}


def restore(path: Path) -> SwaraSpeechGenerator:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config_data = dict(payload["config"])
    config_data["audio_spec"] = AudioTokenSpec(**config_data["audio_spec"])
    model = SwaraSpeechGenerator(GeneratorConfig(**config_data), LinguisticVocabulary.from_dict(payload["vocabulary"]), LearnedSpeakerConditioner(tuple(payload["speaker_ids"])))
    model.module.load_state_dict(payload["model_state"])
    return model.eval()


def metrics_for(model: SwaraSpeechGenerator, sequence: LinguisticSequence, targets: torch.Tensor) -> dict[str, float]:
    target = targets.unsqueeze(0)
    with torch.no_grad():
        primary, residual, _ = model.forward(model.encode_linguistic(sequence), torch.tensor([0]), model.teacher_forcing_inputs(target), primary_tokens_for_residual=target[:, :, 0])
        losses = compute_token_losses(primary, residual, target)
        primary_accuracy = float((primary.argmax(dim=-1) == target[:, :, 0]).float().mean())
        residual_accuracy = float((residual.argmax(dim=-1) == target[:, :, 1:]).float().mean())
    return {"primary_loss": float(losses.primary), "residual_loss": float(losses.residual), "total_loss": float(losses.total), "primary_accuracy": primary_accuracy, "residual_accuracy": residual_accuracy, "overall_token_accuracy": (primary_accuracy + 15 * residual_accuracy) / 16}


def evaluate(model: SwaraSpeechGenerator, examples: list[dict[str, Any]]) -> dict[str, float]:
    model.eval()
    rows = [metrics_for(model, example["sequence"], example["targets"]) for example in examples]
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def conditioning_sensitivity(model: SwaraSpeechGenerator, by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for index, example_id in enumerate(CONDITIONING_IDS):
        wrong_id = CONDITIONING_IDS[(index + 1) % len(CONDITIONING_IDS)]
        target = by_id[example_id]["targets"]
        correct = metrics_for(model, by_id[example_id]["sequence"], target)["total_loss"]
        wrong = metrics_for(model, by_id[wrong_id]["sequence"], target)["total_loss"]
        report[example_id] = {"wrong_text_id": wrong_id, "correct_text_loss": correct, "wrong_text_loss": wrong, "margin": wrong - correct, "positive": correct < wrong}
    return {"items": report, "all_positive": all(item["positive"] for item in report.values())}


def similarity(generated: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    frames = min(generated.shape[0], target.shape[0])
    return {
        "primary_token_similarity": float((generated[:frames, 0] == target[:frames, 0]).float().mean()),
        "full_token_similarity": float((generated[:frames] == target[:frames]).float().mean()),
        "compared_frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/m3c_clean_speech_v0"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/m3c_clean_20_v0"))
    parser.add_argument("--codec-path", type=Path, default=Path("models/qwen3-tts-tokenizer-12hz"))
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1500:
        raise ValueError("M3C hard cap is 1..1500 steps")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("run directory must be empty; refusing to overwrite evidence")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    started = time.monotonic()
    examples = load_examples(args.dataset)
    by_id = {example["record"]["example_id"]: example for example in examples}
    info = json.loads((args.dataset / "dataset_info.json").read_text(encoding="utf-8"))
    spec = AudioTokenSpec(info["codec"]["spec_version"], info["codec"]["codebook_count"], info["codec"]["vocabulary_size"], info["codec"]["frame_rate_hz"])
    vocabulary = LinguisticVocabulary.build(tuple(example["sequence"] for example in examples))
    conditioner = LearnedSpeakerConditioner((info["speaker_id"],))
    config = GeneratorConfig(vocabulary.size, 1, spec, model_dim=256, layers=4, heads=4, ffn_dim=512, max_text_tokens=max(len(example["sequence"].tokens) for example in examples), max_audio_frames=max(int(example["targets"].shape[0]) for example in examples))
    model = SwaraSpeechGenerator(config, vocabulary, conditioner).train()
    args.run_dir.mkdir(parents=True)
    initial_path, best_path, final_path = (args.run_dir / name for name in ("initial.pt", "best.pt", "final.pt"))
    torch.save(checkpoint_payload(model, config, vocabulary, (info["speaker_id"],)), initial_path)
    initial = evaluate(model, examples)
    initial_sensitivity = conditioning_sensitivity(model, by_id)
    best = initial
    torch.save(checkpoint_payload(model, config, vocabulary, (info["speaker_id"],)), best_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    stop_reason = "maximum step cap reached"
    for step in range(1, args.max_steps + 1):
        example = examples[(step - 1) % len(examples)]  # batch size 1: no cross-example padding/alignment risk.
        targets = example["targets"].unsqueeze(0)
        primary, residual, _ = model.forward(model.encode_linguistic(example["sequence"]), torch.tensor([0]), model.teacher_forcing_inputs(targets), primary_tokens_for_residual=targets[:, :, 0])
        losses = compute_token_losses(primary, residual, targets)
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.max_steps:
            current = evaluate(model, examples)
            if current["total_loss"] < best["total_loss"]:
                best = current
                torch.save(checkpoint_payload(model, config, vocabulary, (info["speaker_id"],)), best_path)
            sensitivity = conditioning_sensitivity(model, by_id)
            if current["primary_accuracy"] >= 0.97 and current["residual_accuracy"] >= 0.92 and current["overall_token_accuracy"] >= 0.92 and sensitivity["all_positive"]:
                stop_reason = "token thresholds and text-conditioning sensitivity met"
                break
    final = evaluate(model, examples)
    final_sensitivity = conditioning_sensitivity(model, by_id)
    torch.save(checkpoint_payload(model, config, vocabulary, (info["speaker_id"],)), final_path)
    metrics_pass = final["primary_accuracy"] >= 0.97 and final["residual_accuracy"] >= 0.92 and final["overall_token_accuracy"] >= 0.92 and final_sensitivity["all_positive"]

    restored = restore(best_path)
    serialization_metrics = evaluate(restored, examples)
    codec = Qwen12HzCodecAdapter.from_local_path(args.codec_path)
    if codec.spec != spec:
        raise RuntimeError("M2A codec spec mismatch")
    generated_dir = args.run_dir / "generated"
    generated_dir.mkdir()
    nearest, matrix, listening = {}, {}, []
    generated_by_id: dict[str, torch.Tensor] = {}
    speaker = restored.speaker_conditioner.resolve(info["speaker_id"])
    for example in examples:
        target_frames = int(example["targets"].shape[0])
        generated = restored.generate(example["sequence"], speaker, generation=GenerationOptions(seed=SEED, deterministic=True, max_duration_ms=math.ceil(target_frames / spec.frame_rate_hz * 1000)))
        generated.validate_against(spec)
        generated_tensor = torch.tensor(generated.frames, dtype=torch.long)
        generated_by_id[example["record"]["example_id"]] = generated_tensor
        waveform = codec.decode(generated)
        samples = np.asarray(waveform.samples, dtype=np.float32)
        if samples.size == 0 or not np.isfinite(samples).all() or waveform.sample_rate_hz != 24000:
            raise RuntimeError(f"invalid decoded waveform: {example['record']['example_id']}")
        output = generated_dir / f"{example['record']['example_id']}_generated.wav"
        sf.write(output, samples, waveform.sample_rate_hz, subtype="PCM_16")
    for example in examples:
        example_id = example["record"]["example_id"]
        scores = {target_id: similarity(generated_by_id[example_id], target_example["targets"]) for target_id, target_example in by_id.items()}
        closest = max(scores, key=lambda target_id: scores[target_id]["full_token_similarity"])
        matrix[example_id] = scores
        nearest[example_id] = {"closest_target_id": closest, **scores[closest]}
        listening.append({"example_id": example_id, "transcript": example["record"]["transcript"], "target_prepared_wav": f"../../data/m3c_clean_speech_v0/{example['record']['prepared_audio_path']}", "generated_wav": f"generated/{example_id}_generated.wav", "target_frames": int(example["targets"].shape[0]), "generated_frames": len(generated_by_id[example_id]), "closest_target_id": closest, "similarity": scores[closest], "generation_stop": "Known target frame length for bounded M3C memorization only; no duration model."})
    nearest_count = sum(example_id == report["closest_target_id"] for example_id, report in nearest.items())
    elapsed = time.monotonic() - started
    summary = {"schema_version": "swara.m3c-clean-overfit.v0", "seed": SEED, "training_examples": len(examples), "model_config": {**asdict(config), "audio_spec": asdict(spec)}, "parameter_count": model.parameter_count, "optimizer": "AdamW", "learning_rate": args.learning_rate, "batch_size": 1, "steps": step, "max_steps": args.max_steps, "wall_clock_seconds": elapsed, "initial_losses": initial, "final_losses": final, "best_losses": best, "initial_conditioning_sensitivity": initial_sensitivity, "final_conditioning_sensitivity": final_sensitivity, "serialization_reload_losses": serialization_metrics, "early_stop_reason": stop_reason, "checkpoint_files": [initial_path.name, best_path.name, final_path.name], "nearest_target_accuracy": {"correct": nearest_count, "total": len(examples)}, "temporary_duration_handling": "Target frame length is used only as M3C reconstruction stop constraint."}
    (args.run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.run_dir / "listening_manifest.json").write_text(json.dumps(listening, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.run_dir / "nearest_target_report.json").write_text(json.dumps(nearest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.run_dir / "similarity_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_pass": metrics_pass, "nearest_target_correct": nearest_count, "steps": step, "wall_clock_seconds": elapsed, "final": final, "conditioning_positive": final_sensitivity["all_positive"]}, sort_keys=True))


if __name__ == "__main__":
    main()
