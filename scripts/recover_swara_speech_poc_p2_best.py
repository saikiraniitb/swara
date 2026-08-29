"""Evaluation-only listening recovery for the frozen P2 step-100 checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swara_speech_poc_p1 as p1
import run_swara_speech_poc_p2 as p2

from swara.models.linguistic_composer import LinguisticComposerVocabulary
from swara.models.speech_poc_acoustic import SwaraSpeechPoCV1


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/swara_speech_poc_v1/p2_five_minute/best.pt"
OUTPUT = ROOT / "evaluations/swara_speech_poc_v1/p2_five_minute/best_step_100"
RECOVERY_METRICS = OUTPUT / "recovery_metrics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_rows() -> dict[str, dict]:
    path = ROOT / "experiments/neucodec_n1_v1/data/val_manifest.jsonl"
    return {row["utterance_id"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def valid_wav(path: Path) -> dict:
    stats = p2.wav_stats(path)
    if stats["sample_rate"] != 24_000 or not stats["finite"] or not stats["non_silent"] or stats["samples"] <= 0:
        raise RuntimeError(f"invalid recovered audio: {path}")
    return stats


def main() -> None:
    started = time.perf_counter()
    p1.seed_everything()
    all_train, train, validation = p2.frozen_panel()
    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(example.sequence for example in all_train))
    model = SwaraSpeechPoCV1(vocabulary)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("step") != 100 or payload.get("seed") != p2.SEED:
        raise RuntimeError("P2 best checkpoint metadata is not frozen step 100")
    metadata = payload["metadata"]
    if metadata["initialization_sha256"] != "007aa71ece76a4aa56f22b865bbdb6f3fc060fbf194b3fd4c4413a3a1fd64215":
        raise RuntimeError("P2 initialization provenance changed")
    if metadata["config_sha256"] != "b89bfec80abbcb06cb3c968b5548a60f4528e5986624eaf5c4f3e450dc1ce590":
        raise RuntimeError("P2 configuration provenance changed")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("P2 recovery model is not wholly in eval mode")

    with torch.inference_mode():
        teacher = p2.teacher_forced_metrics(model, validation)
        free, arrays = p2.free_running_evaluation(model, validation)
        manifold = {
            "ground_truth_duration": p2.manifold_metrics(train, arrays, "ground_truth_duration"),
            "full_pipeline": p2.manifold_metrics(train, arrays, "full_pipeline"),
        }

    codec = p1.load_codec()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = source_rows()
    listening = []
    audio_validation = {"codec_oracle": 0, "ground_truth_duration": 0, "full_pipeline": 0}
    by_id = {example.utterance_id: example for example in validation}
    for utterance_id in metadata["config"]["validation_ids"]:
        example = by_id[utterance_id]
        source = ROOT / sources[utterance_id]["source_wav"]
        if not source.exists():
            raise RuntimeError(f"missing source WAV: {source}")

        oracle_output = OUTPUT / "codec_oracle" / f"{utterance_id}.wav"
        existing = p2.find_existing_oracle(utterance_id)
        oracle_output.parent.mkdir(parents=True, exist_ok=True)
        if existing is not None:
            shutil.copy2(existing, oracle_output)
            oracle_reused = True
        else:
            p1.decode_tokens(codec, p2.token_array(example), oracle_output)
            oracle_reused = False
        oracle_stats = valid_wav(oracle_output)
        audio_validation["codec_oracle"] += 1

        generated_stats = {}
        for mode in ("ground_truth_duration", "full_pipeline"):
            output = OUTPUT / mode / f"{utterance_id}.wav"
            generated_stats[mode] = p1.decode_tokens(codec, arrays[utterance_id][mode], output)
            valid_wav(output)
            audio_validation[mode] += 1

        duration = next(row["duration"] for row in free["rows"] if row["utterance_id"] == utterance_id)
        listening.append({
            "utterance_id": utterance_id,
            "authoritative_transcript": example.sequence.source_text,
            "source_wav": str(source.relative_to(ROOT)),
            "codec_oracle": str(oracle_output.relative_to(ROOT)),
            "codec_oracle_reused": oracle_reused,
            "ground_truth_duration_generated_wav": generated_stats["ground_truth_duration"]["path"],
            "full_pipeline_generated_wav": generated_stats["full_pipeline"]["path"],
            "target_frames": duration["target_total_frames"],
            "predicted_frames": duration["predicted_total_frames"],
            "relative_duration_error": duration["relative_length_error"],
            "classification": None,
            "notes": {
                "omissions": None,
                "repetitions": None,
                "loops": None,
                "gross_timing_problems": None,
                "codec_oracle_shared_artifacts": None
            }
        })

    listening_manifest = {
        "schema_version": "swara.speech_poc.p2.best_listening.v1",
        "status": "human_listening_required",
        "p2_overall_machine_status": "fail",
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "sha256": sha256(CHECKPOINT),
            "step": payload["step"]
        },
        "codec": {"model": p1.CODEC_MODEL, "revision": p1.CODEC_REVISION},
        "classification_options": ["RECOGNIZABLE", "PARTIAL", "NOT RECOGNIZABLE"],
        "codec_artifact_policy": "Artifacts shared by codec-oracle and generated audio are not automatically attributed to Swara.",
        "items": listening
    }
    manifest_path = OUTPUT / "listening_manifest.json"
    manifest_path.write_text(json.dumps(listening_manifest, indent=2) + "\n")

    recovery = {
        "schema_version": "swara.speech_poc.p2.best_recovery.v1",
        "status": "complete",
        "training_performed": False,
        "optimizer_steps": 0,
        "model_eval_mode": True,
        "p2_overall_machine_status": "fail",
        "checkpoint": listening_manifest["checkpoint"] | {
            "embedded_seed": payload["seed"],
            "initialization_sha256": metadata["initialization_sha256"],
            "config_sha256": metadata["config_sha256"],
            "frozen_validation_loss": 11.472201234893873,
            "frozen_validation_acoustic_ce": 11.387405395507812,
            "frozen_duration_median_relative_error": 0.11583042879144037,
            "frozen_duration_p90_relative_error": 0.27743749999999995
        },
        "recomputed_step_100": {
            "validation": teacher,
            "free_running": free,
            "manifold": manifold
        },
        "audio_validation": audio_validation,
        "listening_manifest": str(manifest_path.relative_to(ROOT)),
        "wall_seconds": time.perf_counter() - started,
        "architecture_modified": False,
        "codec_modified": False,
        "p3_started": False
    }
    RECOVERY_METRICS.write_text(json.dumps(recovery, indent=2) + "\n")
    print(json.dumps({
        "checkpoint": recovery["checkpoint"],
        "duration": teacher["duration"],
        "acoustic": {key: value for key, value in teacher["acoustic"].items() if key != "rows"},
        "free": {key: value for key, value in free.items() if key not in {"rows", "pairwise", "text_swaps"}},
        "manifold_full": manifold["full_pipeline"],
        "audio": audio_validation,
        "manifest": str(manifest_path)
    }, indent=2))


if __name__ == "__main__":
    main()
