#!/usr/bin/env python3
"""Prepare the bounded Stage2D.2F pilot without installing/downloading models.

If Allosaurus is later approved and installed, this manifest is the exact input
set for one recognizer run. No acoustic output is invented here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

PILOT_WORDS = ["srinagar", "hyderabad", "bengaluru", "chandigarh", "chhattisgarh", "banerjee", "nagpur", "gorakhpur", "jamshedpur", "udhampur"]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = root / "artifacts/stage2d/stage2d2_dataset_design/batch1_acoustic_phone_pilot"
    review = root / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review/batch1_human_review_index.json"
    index = read(review)
    by_word = {row["normalized_word"]: row for row in index["words"]}
    assert list(by_word.get(word, {}) and word for word in PILOT_WORDS) == PILOT_WORDS
    entries = []
    for word in PILOT_WORDS:
        row = by_word[word]
        for entry in row["entries"]:
            clip = (review.parent / entry["word_only_audio_path"]).resolve()
            if not clip.is_file():
                raise FileNotFoundError(clip)
            entries.append({
                "word": row["word"], "normalized_word": word,
                "utterance_id": entry["utterance_id"], "role": entry["role"],
                "transcript": entry["transcript"], "preceding_word": entry["preceding_word"], "following_word": entry["following_word"],
                "audio_path": str(clip), "audio_duration_seconds": entry["clip_metadata"]["word_only"]["duration_seconds"],
                "source_audio_path": entry["source_audio_path"], "context_audio_path": str((review.parent / entry["context_audio_path"]).resolve()),
                "alignment_start_seconds": entry["aligned_start_seconds"], "alignment_end_seconds": entry["aligned_end_seconds"],
                "source_provenance": "Stage2D.2 Batch-1 human-review word-only clip; CTC is segmentation-only",
                "recognition_status": "NOT_RUN_MISSING_SELECTED_RECOGNIZER",
                "raw_phone_sequence": None, "timestamps": None, "confidences": None,
            })
    allosaurus = shutil.which("allosaurus") or shutil.which("python")
    allosaurus_module = bool(__import__("importlib.util").util.find_spec("allosaurus"))
    allophant_module = bool(__import__("importlib.util").util.find_spec("allophant"))
    try:
        version = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        version = None
    audit = {
        "schema_version": "stage2d2f-recognizer-audit.v1",
        "selected_recognizer": "Allosaurus",
        "selection_status": "SELECTED_BUT_NOT_INSTALLED",
        "selection_reason": "Simplest first bounded universal-phone pilot: documented WAV input, IPA inventory, CPU mode, and optional approximate timestamps.",
        "allosaurus": {"available_executable": shutil.which("allosaurus"), "python_module_available": allosaurus_module, "pretrained_model": "uni2005/default latest", "model_available_locally": False, "requires_install_or_download": True, "license": "GPL-3.0", "runtime_note": "CPU interface is documented; local Python 3.14 compatibility was not tested because package is absent.", "timestamp_support": "Documented approximate start/duration output."},
        "allophant": {"available_executable": None, "python_module_available": allophant_module, "pretrained_model": "kgnlp/allophant or local checkpoint", "model_available_locally": False, "requires_install_or_download": True, "license": "MIT", "runtime_note": "Repository documents Python >=3.10, tested on 3.12; local Python 3.14 compatibility was not tested.", "timestamp_support": "Not established for this pilot."},
        "other_local_sources": {"espeak_ng": version, "cmudict": False, "g2p_en": False, "phonemizer": False, "epitran": False},
        "decision": "Do not install/download in this task; pilot remains pending approval of one isolated recognizer environment.",
        "ctc_policy": "CTC alignment metadata is segmentation-only and is never treated as phone evidence.",
    }
    write(out / "stage2d2f_recognizer_audit.json", audit)
    with (out / "stage2d2f_raw_acoustic_phone_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    common = {"status": "NOT_RESOLVED_RECOGNIZER_UNAVAILABLE", "reason": "No pretrained acoustic recognizer was installed or run; repeatability is not measurable."}
    write(out / "stage2d2f_repeatability.json", {"schema_version": "stage2d2f-repeatability.v1", "words": [{"word": word.title(), "normalized_word": word, "sample_count": 5, "usable_prediction_count": 0, "classification": "RECOGNIZER_UNRELIABLE", **common} for word in PILOT_WORDS], "metrics": "Not computed without acoustic predictions."})
    write(out / "stage2d2f_espeak_acoustic_comparison.json", {"schema_version": "stage2d2f-espeak-acoustic-comparison.v1", "words": [{"word": word.title(), "normalized_word": word, "espeak_sources": ["espeak_ng_en_us", "espeak_ng_en_gb"], "acoustic_prediction_count": 0, "comparison": "NOT_RESOLVED_RECOGNIZER_UNAVAILABLE"} for word in PILOT_WORDS], "provenance_note": "eSpeak configurations and acoustic recognizer are not three independent systems."})
    write(out / "stage2d2f_inventory_evidence.json", {"schema_version": "stage2d2f-inventory-evidence.v1", "status": "NOT_RESOLVED", "production_inventory_modified": False, "evidence": {"SCHWA": "NOT_RESOLVED", "aspiration": "NOT_RESOLVED", "retroflex": "NOT_RESOLVED", "W_V": "NOT_RESOLVED", "NG": "NOT_RESOLVED", "SH_affricate": "NOT_RESOLVED", "vowel_length": "NOT_RESOLVED"}, "reason": "No acoustic recognizer predictions available."})
    write(out / "stage2d2f_family_analysis.json", {"schema_version": "stage2d2f-family-analysis.v1", "status": "NOT_RESOLVED", "families": {"PUR": ["Nagpur", "Udhampur"], "GARH": ["Chandigarh", "Chhattisgarh", "Gorakhpur"], "NAGAR": ["Srinagar"], "other": ["Hyderabad", "Bengaluru", "Banerjee", "Jamshedpur"]}, "conclusion": "No acoustic family result without recognizer output; no morphological rule inferred."})
    write(out / "stage2d2f_minimal_human_questions.json", {"schema_version": "stage2d2f-minimal-human-questions.v1", "question_count": 0, "status": "NOT_CREATED_PENDING_ACOUSTIC_EVIDENCE", "questions": []})
    write(out / "stage2d2f_summary.json", {"schema_version": "stage2d2f-summary.v1", "status": "INSUFFICIENT_PHONETIC_SOURCES", "pilot_words": [word.title() for word in PILOT_WORDS], "clip_count": len(entries), "recognition_success_count": 0, "recognition_failure_count": len(entries), "recognition_failure_reason": "Allosaurus and Allophant are absent and no pretrained model is local; installation/download was intentionally not performed.", "repeatability_class_counts": {"ACOUSTIC_PHONE_PATTERN_STABLE": 0, "MOSTLY_STABLE": 0, "UNSTABLE": 0, "RECOGNIZER_UNRELIABLE": 0}, "human_review_question_count": 0, "training_performed": False, "qwen_loaded": False, "swara_phones_v0_modified": False})
    print(json.dumps({"pilot_words": len(PILOT_WORDS), "clips": len(entries), "status": "INSUFFICIENT_PHONETIC_SOURCES", "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
