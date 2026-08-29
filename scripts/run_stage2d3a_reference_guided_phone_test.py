#!/usr/bin/env python3
"""Run the bounded Stage2D.3A reference-guided synthesis panel.

The script is intended for the existing model-equipped Stage2B/Stage2C
environment (for example the browser-Colab bundle).  It does not train or
modify any model/checkpoint.  It is not run by the local preparation pass when
the Qwen foundation is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT", str(REPO_ROOT))).resolve()
OUTPUT_ROOT = BUNDLE_ROOT / "artifacts/stage2d/stage2d3_reference_guided_phone_test"
MODEL_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_MODEL_ROOT", str(BUNDLE_ROOT / "models/qwen3_tts_0_6b_base"))).resolve()
DEFAULT_CHECKPOINT = BUNDLE_ROOT / "run_artifacts/stage2b4b_pronunciation_v0/checkpoints/step025.pt"
SPEC_PATH = Path(os.environ.get("SWARA_STAGE2D3A_SPEC", str(OUTPUT_ROOT / "stage2d3a_candidate_spec.json"))).resolve()
LEXICON_PATH = Path(os.environ.get("SWARA_STAGE2D3A_REFERENCE_LEXICON", str(BUNDLE_ROOT / "artifacts/stage2d/stage2d2_dataset_design/human_acoustic_reference_lexicon_v0_1.json"))).resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara.adapters.qwen_stage2b import QwenStage2BAdapter, QwenStage2BConditioningConfig
from swara.adapters.qwen_tts import QwenFoundationTTS
from swara.frontend import Frontend
from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation


def samples_and_rate(value: Any) -> tuple[list[float], int]:
    if isinstance(value, tuple) and len(value) == 2 and hasattr(value[0], "samples"):
        value = value[0]
    if not hasattr(value, "samples") or not hasattr(value, "sample_rate_hz"):
        raise TypeError("unsupported Qwen waveform result")
    samples = value.samples
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu()
        if samples.ndim == 2 and samples.shape[0] == 1:
            samples = samples[0]
        if samples.ndim != 1:
            raise ValueError("generated waveform must be mono")
        samples = samples.tolist()
    values = [float(x) for x in samples]
    rate = int(value.sample_rate_hz)
    if not values or rate <= 0 or not all(math.isfinite(x) for x in values):
        raise ValueError("generated waveform is invalid")
    return values, rate


def extract_acoustic_trace(wrapper: Any) -> Any:
    """Unwrap the current native/conditioned adapter result to its trace.

    Native diagnostics return ``QwenStage2BNativeTrace`` while conditioned
    diagnostics return ``QwenStage2BConditioningResult``.  Both expose the
    actual ``QwenAcousticGenerationTrace`` through ``acoustic_trace``; the
    helper also accepts the trace itself for small, dependency-free tests.
    """

    trace = wrapper
    if hasattr(trace, "acoustic_trace"):
        trace = trace.acoustic_trace
    if trace is None or not all(hasattr(trace, name) for name in ("generated_frame_count", "eos_index")):
        raise TypeError("adapter result does not contain a QwenAcousticGenerationTrace")
    return trace


def classify_trajectory(trace: Any) -> str:
    """Classify a completed trace without judging pronunciation quality.

    Stage2C.2A separated ordinary outputs (well below ten seconds) from
    EOS-completed long paths.  Qwen's returned acoustic frames exclude its
    boundary EOS frame, so a 511-frame result at ``max_new_tokens=512`` is
    treated as a max-length result when the trace exposes that limit.
    """

    if trace is None:
        return "FAILED"
    generated_frames = int(getattr(trace, "generated_frame_count", 0))
    max_new_tokens = getattr(trace, "max_new_tokens", None)
    if bool(getattr(trace, "max_generation_hit", False)):
        return "MAX_LENGTH_TRAJECTORY"
    if isinstance(max_new_tokens, int) and max_new_tokens > 0 and generated_frames >= max_new_tokens - 1:
        return "MAX_LENGTH_TRAJECTORY"
    if getattr(trace, "eos_index", None) is None:
        return "LONG_TRAJECTORY"
    sample_rate = getattr(trace, "sample_rate_hz", None)
    sample_count = getattr(trace, "waveform_sample_count", None)
    if isinstance(sample_rate, (int, float)) and sample_rate > 0 and isinstance(sample_count, int):
        if sample_count / sample_rate > 10.0:
            return "LONG_TRAJECTORY"
    # Qwen's 12.5 Hz codec frame rate gives the same ten-second boundary when
    # waveform metadata is absent from a lightweight trace.
    if generated_frames > 125:
        return "LONG_TRAJECTORY"
    return "NORMAL_TRAJECTORY"


def state_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def representation(text: str, word: str | None, phones: list[str] | None):
    overrides = ()
    if word is not None:
        start = text.index(word)
        overrides = (PronunciationOverride(start, start + len(word), "swara-phones-v0", tuple(phones or ()), "en-IN"),)
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("stage2b4b-frozen-speaker"),
        pronunciation=PronunciationInput(overrides=overrides),
    )
    return build_stage2b_representation(Frontend().compile(request))


def load_checkpoint(path: Path) -> tuple[dict[str, Any], Stage2BLinguisticBridge]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_id") != "stage2b4b_pronunciation_v0" or int(payload.get("step", -1)) != 25:
        raise ValueError("Stage2D.3A requires frozen Stage2B step025")
    state = payload.get("bridge_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks bridge_state_dict")
    projection = state.get("projection.weight")
    if not isinstance(projection, torch.Tensor) or tuple(projection.shape)[1] != 160:
        raise ValueError("checkpoint bridge is not compatible with D_ling=160")
    bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, int(projection.shape[0]), initialization_seed=20260829))
    bridge.load_state_dict(state, strict=True)
    bridge.eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return payload, bridge


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(OUTPUT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_html(spec: dict[str, Any], rows: list[dict[str, Any]], blind_map: dict[str, Any]) -> None:
    by_word: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_word.setdefault(row["word"], []).append(row)
    sections = []
    for case in spec["cases"]:
        word = case["word"]
        ref_word = (OUTPUT_ROOT / "audio" / word.lower() / "reference" / "word.wav").resolve()
        ref_context = (OUTPUT_ROOT / "audio" / word.lower() / "reference" / "context.wav").resolve()
        audio = [f'<p>Human reference word-only</p><audio controls src="{html.escape(os.path.relpath(ref_word, OUTPUT_ROOT))}"></audio>', f'<p>Human reference context</p><audio controls src="{html.escape(os.path.relpath(ref_context, OUTPUT_ROOT))}"></audio>']
        for row in by_word[word]:
            audio.append(f'<div><p><strong>Review label {html.escape(row["review_label"])}</strong></p><audio controls src="{html.escape(os.path.relpath(Path(row["waveform_path"]), OUTPUT_ROOT))}"></audio></div>')
        sections.append(f'<section><h2>{html.escape(word)}</h2><p>{html.escape(case["synthesis_sentence"])}</p>{"".join(audio)}<p>Which generated version pronounces the target closest to the reference? A / B / C / D / NONE</p><p>Is the difference mainly pronunciation or surrounding voice/prosody?</p></section>')
    (OUTPUT_ROOT / "human_review.html").write_text("<!doctype html><meta charset='utf-8'><title>Stage2D.3A blinded review</title><h1>Stage2D.3A reference-guided phone test</h1>" + "".join(sections), encoding="utf-8")


def main() -> int:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_dir.resolve()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["diagnostic_word_count"] != 5 or spec["runtime"]["mask_mode"] != "target_context_1":
        raise ValueError("Stage2D.3A candidate spec is not the frozen five-word plan")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SPEC_PATH, OUTPUT_ROOT / "candidate_spec.json")
    payload, bridge = load_checkpoint(args.checkpoint.resolve())
    foundation = QwenFoundationTTS.from_local_path(
        MODEL_ROOT,
        reference_audio=os.environ["SWARA_STAGE2B4B_REFERENCE_AUDIO"],
        device_map=os.environ.get("SWARA_STAGE2B4B_DEVICE_MAP", "cuda"),
        dtype=getattr(torch, os.environ.get("SWARA_STAGE2B4B_DTYPE", "float32")),
    )
    native = foundation._model.model
    hidden_size = int(native.talker.config.hidden_size)
    if bridge.backbone_dim != hidden_size:
        raise ValueError("checkpoint bridge width does not match Qwen Talker")
    qwen_before = state_hash(native)
    settings = dict(spec["generation_settings"])
    rows = []
    blind_map = {}
    review_root = BUNDLE_ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review"
    if not review_root.is_dir():
        review_root = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review"
    for case in spec["cases"]:
        reference_dir = OUTPUT_ROOT / "audio" / case["word"].lower() / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        for source_key, output_name in (("reference_word_audio_path", "word.wav"), ("reference_context_audio_path", "context.wav")):
            source = (review_root / case[source_key]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"missing human reference clip: {source}")
            shutil.copy2(source, reference_dir / output_name)
    for case in spec["cases"]:
        rep = representation(case["synthesis_sentence"], None, None)
        native_tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).eval()
        with torch.no_grad():
            native_batch = native_tensorizer((rep,))
        conditions = [("native", None, "native")]
        conditions.extend((candidate["candidate_id"], candidate["phone_sequence"], "candidate") for candidate in case["candidates"])
        labels = [f"{case['word'].lower()}_A", f"{case['word'].lower()}_B", f"{case['word'].lower()}_C", f"{case['word'].lower()}_D"]
        order = list(range(len(conditions)))
        # Stable deterministic blinding: reverse the condition order within each word.
        order.reverse()
        for display_index, condition_index in enumerate(order):
            condition_id, phones, kind = conditions[condition_index]
            display_label = labels[display_index]
            if kind == "native":
                adapter = QwenStage2BAdapter(foundation, bridge, QwenStage2BConditioningConfig(160, hidden_size, gate=float(payload["gate"]), mask_mode="full"))
                output, native_result = adapter.diagnostic_native_generation(text=case["synthesis_sentence"], **settings)
                trace = extract_acoustic_trace(native_result)
                active = []
            else:
                rep = representation(case["synthesis_sentence"], case["word"], phones)
                tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).eval()
                with torch.no_grad():
                    batch = tensorizer((rep,))
                adapter = QwenStage2BAdapter(foundation, bridge, QwenStage2BConditioningConfig(160, hidden_size, gate=float(payload["gate"]), mask_mode="target_context_1", strict_equivalence=False))
                output, conditioned_result = adapter.diagnostic_conditioned_generation(rep, batch, **settings)
                trace = extract_acoustic_trace(conditioned_result)
                active = list(conditioned_result.active_residual_positions)
            samples, rate = samples_and_rate(output)
            candidate_dir = "native" if kind == "native" else "_".join(condition_id.split("_")[-2:])
            word_dir = OUTPUT_ROOT / "audio" / case["word"].lower() / candidate_dir
            word_dir.mkdir(parents=True, exist_ok=True)
            wav = word_dir / f"{display_label}.wav"
            sf.write(wav, samples, rate, subtype="PCM_16")
            row = {
                "word": case["word"], "review_label": display_label, "actual_condition": condition_id,
                "kind": kind, "synthesis_sentence": case["synthesis_sentence"], "phone_sequence": phones,
                "mask_mode": "native" if kind == "native" else "target_context_1", "active_residual_positions": active,
                "generated_frame_count": trace.generated_frame_count if trace else None,
                "eos_index": trace.eos_index if trace else None, "eos_reason": trace.termination_reason if trace else None,
                "duration_seconds": len(samples) / rate, "trajectory_class": classify_trajectory(trace),
                "waveform_path": str(wav), "qwen_hash_before": qwen_before, "qwen_hash_after": state_hash(native),
            }
            rows.append(row)
            blind_map[display_label] = {"word": case["word"], "actual_condition": condition_id, "phone_sequence": phones}
    (OUTPUT_ROOT / "stage2d3a_manifest.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "stage2d3a_report.json").write_text(json.dumps({"rows": rows, "training_performed": False, "qwen_modified": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "generation_manifest.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trajectory_report = {
        "rows": rows,
        "normal_trajectories": sum(row["trajectory_class"] == "NORMAL_TRAJECTORY" for row in rows),
        "long_trajectories": sum(row["trajectory_class"] == "LONG_TRAJECTORY" for row in rows),
        "failed_generations": sum(row["trajectory_class"] == "FAILED" for row in rows),
    }
    (OUTPUT_ROOT / "trajectory_report.json").write_text(json.dumps(trajectory_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "stage2d3a_blinding_map.json").write_text(json.dumps(blind_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html(spec, rows, blind_map)
    (OUTPUT_ROOT / "run_status.json").write_text(json.dumps({"status": "SUCCESS", "rows": len(rows), "training_performed": False, "qwen_modified": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "rows": len(rows), "output_dir": str(OUTPUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
