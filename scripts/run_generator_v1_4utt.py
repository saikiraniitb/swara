"""One bounded Swara Generator v1 four-utterance validation run."""

from __future__ import annotations

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
from swara.contracts import AudioTokenSpec, GenerationOptions
from swara.frontend.spans import TextSpan
from swara.frontend.tokenizer import LinguisticSequence, LinguisticToken, LinguisticTokenKind
from swara.models.generator import GeneratorConfig, LearnedSpeakerConditioner, SwaraSpeechGenerator
from swara.models.linguistic import LinguisticVocabulary
from swara.models.training import compute_token_losses


IDS = ("001", "005", "006", "014")
SEED = 20260822


def load_examples(root: Path) -> list[dict[str, Any]]:
    rows = {row["example_id"]: row for row in (json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    examples = []
    for example_id in IDS:
        row = rows[example_id]
        array = np.load(root / row["codec_token_path"], allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 16 or array.size == 0 or int(array.min()) < 0 or int(array.max()) > 2047:
            raise ValueError(f"invalid codec tokens: {example_id}")
        tokens = tuple(LinguisticToken(LinguisticTokenKind(item["kind"]), item["value"], item.get("language"),
                                       None if item.get("source_range") is None else TextSpan(*item["source_range"]),
                                       None if item.get("normalized_range") is None else TextSpan(*item["normalized_range"]), item.get("override_id"))
                       for item in row["linguistic_tokens"])
        sequence = LinguisticSequence(row.get("linguistic_schema_version", "swara.linguistic.v0"), row["transcript"], row["transcript"], tokens, ())
        examples.append({"id": example_id, "row": row, "sequence": sequence, "targets": torch.from_numpy(array.astype(np.int64))})
    if len({e["row"]["speaker_id"] for e in examples}) != 1:
        raise ValueError("selected examples must have one speaker")
    return examples


def payload(model: SwaraSpeechGenerator, config: GeneratorConfig, vocabulary: LinguisticVocabulary, speakers: tuple[str, ...]) -> dict[str, Any]:
    data = asdict(config)
    data["audio_spec"] = asdict(config.audio_spec)
    return {"format": "swara.generator-v1.4utt.v0", "model_state": model.module.state_dict(), "config": data, "vocabulary": vocabulary.to_dict(), "speaker_ids": speakers}


def restore(path: Path) -> SwaraSpeechGenerator:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    data = dict(saved["config"])
    data["audio_spec"] = AudioTokenSpec(**data["audio_spec"])
    model = SwaraSpeechGenerator(GeneratorConfig(**data), LinguisticVocabulary.from_dict(saved["vocabulary"]), LearnedSpeakerConditioner(tuple(saved["speaker_ids"])))
    model.module.load_state_dict(saved["model_state"])
    return model.eval()


def metrics(model: SwaraSpeechGenerator, examples: list[dict[str, Any]]) -> dict[str, float]:
    rows = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            target = example["targets"].unsqueeze(0)
            primary, residual, _ = model.forward(model.encode_linguistic(example["sequence"]), torch.tensor([0]), model.teacher_forcing_inputs(target), primary_tokens_for_residual=target[:, :, 0], residual_history_inputs=model.teacher_forcing_frame_history(target), residual_targets_for_prediction=target[:, :, 1:])
            loss = compute_token_losses(primary, residual, target)
            pa = (primary.argmax(-1) == target[:, :, 0]).float().mean()
            ra = (residual.argmax(-1) == target[:, :, 1:]).float().mean()
            rows.append((float(loss.primary), float(loss.residual), float(loss.total), float(pa), float(ra)))
    values = np.asarray(rows)
    return {"primary_loss": float(values[:, 0].mean()), "residual_loss": float(values[:, 1].mean()), "total_loss": float(values[:, 2].mean()), "primary_accuracy": float(values[:, 3].mean()), "residual_accuracy": float(values[:, 4].mean()), "overall_token_accuracy": float((values[:, 3].mean() + 15 * values[:, 4].mean()) / 16)}


def sensitivity(model: SwaraSpeechGenerator, examples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for index, example in enumerate(examples):
        wrong = examples[(index + 1) % len(examples)]
        target = example["targets"].unsqueeze(0)
        with torch.no_grad():
            correct_logits = model.forward(model.encode_linguistic(example["sequence"]), torch.tensor([0]), model.teacher_forcing_inputs(target), primary_tokens_for_residual=target[:, :, 0], residual_history_inputs=model.teacher_forcing_frame_history(target), residual_targets_for_prediction=target[:, :, 1:])
            wrong_logits = model.forward(model.encode_linguistic(wrong["sequence"]), torch.tensor([0]), model.teacher_forcing_inputs(target), primary_tokens_for_residual=target[:, :, 0], residual_history_inputs=model.teacher_forcing_frame_history(target), residual_targets_for_prediction=target[:, :, 1:])
            c = float(compute_token_losses(correct_logits[0], correct_logits[1], target).primary)
            w = float(compute_token_losses(wrong_logits[0], wrong_logits[1], target).primary)
        result[example["id"]] = {"wrong_text_id": wrong["id"], "correct_primary_loss": c, "wrong_primary_loss": w, "margin": w - c, "positive": c < w}
    return {"items": result, "all_positive": all(item["positive"] for item in result.values())}


def main() -> None:
    root = Path("data/m3c_clean_speech_v0")
    run = Path("runs/m3b_generator_v2_4utt_v0")
    if run.exists() and any(run.iterdir()):
        raise ValueError("refusing to overwrite a non-empty run directory")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    start = time.monotonic()
    examples = load_examples(root)
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    spec = AudioTokenSpec(info["codec"]["spec_version"], info["codec"]["codebook_count"], info["codec"]["vocabulary_size"], info["codec"]["frame_rate_hz"])
    vocabulary = LinguisticVocabulary.build(tuple(e["sequence"] for e in examples))
    speaker_ids = (examples[0]["row"]["speaker_id"],)
    config = GeneratorConfig(vocabulary.size, 1, spec, model_dim=384, layers=4, heads=6, ffn_dim=1536, max_text_tokens=max(len(e["sequence"].tokens) for e in examples), max_audio_frames=max(int(e["targets"].shape[0]) for e in examples), residual_dim=192, primary_history_dropout=0.0)
    model = SwaraSpeechGenerator(config, vocabulary, LearnedSpeakerConditioner(speaker_ids)).train()
    run.mkdir(parents=True)
    initial_path, best_path, final_path = (run / name for name in ("initial.pt", "best.pt", "final.pt"))
    torch.save(payload(model, config, vocabulary, speaker_ids), initial_path)
    initial, initial_sensitivity = metrics(model, examples), sensitivity(model, examples)
    best = initial
    torch.save(payload(model, config, vocabulary, speaker_ids), best_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    stop_reason = "maximum step cap reached"
    for step in range(1, 801):
        example = examples[(step - 1) % len(examples)]
        target = example["targets"].unsqueeze(0)
        primary, residual, _ = model.forward(model.encode_linguistic(example["sequence"]), torch.tensor([0]), model.teacher_forcing_inputs(target), primary_tokens_for_residual=target[:, :, 0], residual_history_inputs=model.teacher_forcing_frame_history(target), residual_targets_for_prediction=target[:, :, 1:])
        loss = compute_token_losses(primary, residual, target).total
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == 800:
            current = metrics(model, examples)
            if current["total_loss"] < best["total_loss"]:
                best = current
                torch.save(payload(model, config, vocabulary, speaker_ids), best_path)
            sens = sensitivity(model, examples)
            if current["primary_accuracy"] >= 0.95 and current["residual_accuracy"] >= 0.90 and current["overall_token_accuracy"] >= 0.90 and sens["all_positive"]:
                stop_reason = "functional teacher-forced thresholds and positive text sensitivity"
                break
    final = metrics(model, examples)
    final_sensitivity = sensitivity(model, examples)
    torch.save(payload(model, config, vocabulary, speaker_ids), final_path)
    restored = restore(best_path)
    codec = Qwen12HzCodecAdapter.from_local_path(Path("models/qwen3-tts-tokenizer-12hz"))
    generated_dir = run / "generated"
    generated_dir.mkdir()
    generated, nearest, listening = {}, {}, []
    speaker = restored.speaker_conditioner.resolve(speaker_ids[0])
    for example in examples:
        frames = int(example["targets"].shape[0])
        out = restored.generate(example["sequence"], speaker, generation=GenerationOptions(seed=SEED, deterministic=True, max_duration_ms=math.ceil(frames / spec.frame_rate_hz * 1000)))
        out.validate_against(spec)
        generated[example["id"]] = torch.tensor(out.frames, dtype=torch.long)
        waveform = codec.decode(out)
        samples = np.asarray(waveform.samples, dtype=np.float32)
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError(f"invalid decoded waveform: {example['id']}")
        path = generated_dir / f"{example['id']}_generated.wav"
        sf.write(path, samples, waveform.sample_rate_hz, subtype="PCM_16")
    for example in examples:
        scores = {}
        for target in examples:
            n = min(len(generated[example["id"]]), len(target["targets"]))
            scores[target["id"]] = {"primary_similarity": float((generated[example["id"]][:n, 0] == target["targets"][:n, 0]).float().mean()), "full_similarity": float((generated[example["id"]][:n] == target["targets"][:n]).float().mean())}
        closest = max(scores, key=lambda key: scores[key]["full_similarity"])
        nearest[example["id"]] = {"closest_target_id": closest, **scores[closest]}
        listening.append({"example_id": example["id"], "transcript": example["row"]["transcript"], "target_wav": f"../../data/m3c_clean_speech_v0/{example['row']['prepared_audio_path']}", "generated_wav": f"generated/{example['id']}_generated.wav", "target_frames": int(example["targets"].shape[0]), "generated_frames": len(generated[example["id"]]), **nearest[example["id"]]})
    elapsed = time.monotonic() - start
    summary = {"schema_version": "swara.generator-v1.4utt.v0", "seed": SEED, "model_config": {**asdict(config), "audio_spec": asdict(spec)}, "parameter_count": model.parameter_count, "optimizer": "AdamW", "learning_rate": 0.001, "steps": step, "wall_clock_seconds": elapsed, "initial_losses": initial, "final_losses": final, "best_losses": best, "initial_text_sensitivity": initial_sensitivity, "final_text_sensitivity": final_sensitivity, "early_stop_reason": stop_reason, "checkpoint_files": ["initial.pt", "best.pt", "final.pt"], "nearest_target": nearest, "temporary_duration_handling": "Known target frame length is used only for this bounded reconstruction."}
    (run / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run / "listening_manifest.json").write_text(json.dumps(listening, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"parameter_count": model.parameter_count, "steps": step, "initial": initial, "final": final, "nearest": nearest, "wall_clock_seconds": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
