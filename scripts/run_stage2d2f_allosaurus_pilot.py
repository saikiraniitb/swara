#!/usr/bin/env python3
"""Execute the frozen 50-clip Stage2D.2F Allosaurus evidence pilot."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PILOT_WORDS = ["srinagar", "hyderabad", "bengaluru", "chandigarh", "chhattisgarh", "banerjee", "nagpur", "gorakhpur", "jamshedpur", "udhampur"]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def parse_timestamps(raw: str) -> list[dict]:
    result = []
    for line in raw.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            result.append({"start_seconds": float(fields[0]), "duration_seconds": float(fields[1]), "phone": fields[2]})
        except ValueError:
            continue
    return result


def normalize(raw: str) -> list[str]:
    return [unicodedata.normalize("NFC", token) for token in raw.split()]


def repeatability(word: str, rows: list[dict], q25: float, q75: float) -> dict:
    sequences = [row["analysis_normalized_phone_tokens"] for row in rows]
    pairs = []
    substitutions = Counter()
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            a, b = sequences[i], sequences[j]
            pairs.append({"left_utterance_id": rows[i]["utterance_id"], "right_utterance_id": rows[j]["utterance_id"], "edit_distance": edit_distance(a, b), "normalized_edit_distance": edit_distance(a, b) / max(len(a), len(b), 1)})
            for left, right in zip(a, b):
                if left != right:
                    substitutions[f"{left}->{right}"] += 1
    distances = [x["normalized_edit_distance"] for x in pairs]
    means = statistics.mean(distances) if distances else None
    medoid = min(range(len(sequences)), key=lambda i: sum(edit_distance(sequences[i], other) for other in sequences)) if sequences else None
    stable_positions, unstable_positions = [], []
    for position in range(max((len(x) for x in sequences), default=0)):
        values = [x[position] for x in sequences if position < len(x)]
        if not values:
            continue
        count = Counter(values).most_common(1)[0][1]
        (stable_positions if count / len(values) >= 0.8 else unstable_positions).append({"position": position, "values": dict(Counter(values)), "majority_fraction": count / len(values)})
    if means is None:
        classification = "RECOGNIZER_UNRELIABLE"
    elif means <= q25:
        classification = "ACOUSTIC_PHONE_PATTERN_STABLE"
    elif means <= q75:
        classification = "MOSTLY_STABLE"
    else:
        classification = "UNSTABLE"
    return {"word": word.title(), "normalized_word": word, "sample_count": len(rows), "usable_prediction_count": len(rows), "pairwise": pairs, "mean_normalized_edit_distance": means, "median_normalized_edit_distance": statistics.median(distances) if distances else None, "max_normalized_edit_distance": max(distances) if distances else None, "distinct_raw_output_count": len({x["raw_phone_sequence"] for x in rows}), "consensus_sequence": sequences[medoid] if medoid is not None and means <= q75 else None, "consensus_method": "medoid_sequence_only_for_non-high-variability-word" if medoid is not None and means <= q75 else None, "stable_positions": stable_positions, "unstable_positions": unstable_positions, "common_positional_substitutions": substitutions.most_common(20), "classification": classification, "classification_calibration": {"q25_word_mean": q25, "q75_word_mean": q75, "policy": "q25 or below is stable, q25-q75 is mostly stable, above q75 is unstable; thresholds are calibrated from this pilot's ten observed word means."}}


def feature_evidence(rows: list[dict], repeat_rows: list[dict]) -> list[dict]:
    specs = {
        "SCHWA": {"ə", "ɐ", "ɐː", "əː", "ʌ", "ɚ", "ɝ"},
        "ASPIRATION": {"ʰ", "tʰ", "dʰ", "kʰ", "gʰ", "pʰ", "bʰ", "t̪ʰ"},
        "RETROFLEX": {"ʈ", "ɖ", "ɻ", "ɽ", "ʂ", "ʐ", "tʂ", "dʐ"},
        "W_V": {"w", "v", "ʋ"},
        "NG": {"ŋ", "ɴ", "ŋ̟", "ŋ͡m"},
        "SH_AFFRICATE": {"ʃ", "ʒ", "tʃ", "dʒ", "tɕ", "tʂ"},
        "VOWEL_LENGTH": {"ː"},
    }
    all_words = {row["normalized_word"] for row in rows}
    result = []
    for name, needles in specs.items():
        if name == "VOWEL_LENGTH":
            hits = [row for row in rows if any("ː" in token for token in row["analysis_normalized_phone_tokens"])]
        else:
            hits = [row for row in rows if needles.intersection(row["analysis_normalized_phone_tokens"])]
        hit_words = sorted({row["normalized_word"] for row in hits})
        repeated = len(hits) >= 3
        status = "ACOUSTIC_SUPPORT_WEAK" if repeated else "NOT_RESOLVED"
        result.append({"candidate": name, "status": status, "affected_words": hit_words, "affected_word_count": len(hit_words), "recognizer_occurrence_count": len(hits), "independent_sources": 1, "repeated_spicor_stability": "LIKELY_STABLE_HUMAN_REVIEW_ONLY" if repeated else "NOT_ESTABLISHED", "v0_merge_risk": "POTENTIAL" if repeated else "UNMEASURED", "evidence_note": "Repeated recognizer symbols are acoustic evidence only; they do not establish a phoneme or justify a production inventory change."})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default="uni2005")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = root / "artifacts/stage2d/stage2d2_dataset_design/batch1_acoustic_phone_pilot"
    index_path = root / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review/batch1_human_review_index.json"
    index = read(index_path)
    by_word = {row["normalized_word"]: row for row in index["words"]}
    if any(word not in by_word for word in PILOT_WORDS):
        raise RuntimeError("pilot word set does not match frozen review index")
    from allosaurus.app import read_recognizer
    import allosaurus
    from importlib.metadata import version

    recognizer = read_recognizer(args.model)
    model_root = Path(allosaurus.__file__).resolve().parent / "pretrained" / args.model
    model_size = sum(path.stat().st_size for path in model_root.rglob("*") if path.is_file()) if model_root.is_dir() else None
    entries = []
    for word in PILOT_WORDS:
        for entry in by_word[word]["entries"]:
            audio = (index_path.parent / entry["word_only_audio_path"]).resolve()
            raw = recognizer.recognize(str(audio), "ipa")
            raw_timestamps = recognizer.recognize(str(audio), "ipa", timestamp=True)
            if not raw.strip():
                raise RuntimeError(f"empty Allosaurus output for {audio}")
            entries.append({
                "word": by_word[word]["word"], "normalized_word": word, "utterance_id": entry["utterance_id"], "role": entry["role"], "transcript": entry["transcript"], "preceding_word": entry["preceding_word"], "following_word": entry["following_word"], "audio_path": str(audio), "audio_duration_seconds": entry["clip_metadata"]["word_only"]["duration_seconds"], "source_audio_path": entry["source_audio_path"], "alignment_start_seconds": entry["aligned_start_seconds"], "alignment_end_seconds": entry["aligned_end_seconds"], "recognizer": "Allosaurus", "recognizer_version": version("allosaurus"), "model_identifier": args.model, "raw_phone_sequence": raw.strip(), "raw_timestamp_output": raw_timestamps.strip(), "timestamps": parse_timestamps(raw_timestamps), "analysis_normalized_phone_tokens": normalize(raw), "recognition_status": "SUCCESS", "ctc_phone_truth": False,
            })
    first = entries[0]
    means = []
    grouped = {word: [x for x in entries if x["normalized_word"] == word] for word in PILOT_WORDS}
    for rows in grouped.values():
        seqs = [x["analysis_normalized_phone_tokens"] for x in rows]
        pair_means = [edit_distance(seqs[i], seqs[j]) / max(len(seqs[i]), len(seqs[j]), 1) for i in range(len(seqs)) for j in range(i + 1, len(seqs))]
        means.append(statistics.mean(pair_means))
    q25, q75 = statistics.quantiles(means, n=4, method="inclusive")[0], statistics.quantiles(means, n=4, method="inclusive")[2]
    repeat_rows = [repeatability(word, grouped[word], q25, q75) for word in PILOT_WORDS]
    write_jsonl(out / "stage2d2f_raw_acoustic_phone_predictions.jsonl", entries)
    # Re-read the just-written JSONL is intentionally avoided; the in-memory
    # rows are the source for every derived artifact in this run.
    write(out / "stage2d2f_recognizer_audit.json", {"schema_version": "stage2d2f-recognizer-audit.v2", "run_type": "EXECUTED_ALLOSAURUS_RUN", "selected_recognizer": "Allosaurus", "installed": True, "version": version("allosaurus"), "model_identifier": args.model, "model_cache_path": str(model_root), "model_size_bytes": model_size, "model_license": "GPL-3.0", "python": sys.version, "torch_version": __import__("torch").__version__, "device": "cpu", "mps_available": bool(__import__("torch").backends.mps.is_available()) if hasattr(__import__("torch").backends, "mps") else False, "timestamp_support": True, "input_policy": "Exactly 50 existing Batch-1 word-only clips; no re-extraction or resampling.", "ctc_policy": "CTC alignment remains segmentation-only and is never treated as phone evidence.", "one_clip_probe": {"status": "SUCCESS", "word": first["word"], "utterance_id": first["utterance_id"], "non_empty": True, "phone_output": first["raw_phone_sequence"]}})
    write(out / "stage2d2f_repeatability.json", {"schema_version": "stage2d2f-repeatability.v2", "pilot_word_count": 10, "calibration": {"word_mean_q25": q25, "word_mean_q75": q75, "policy": "q25 or below stable; q25-q75 mostly stable; above q75 unstable."}, "words": repeat_rows, "metrics": "Normalized NFC tokens from raw Allosaurus output; no phones are treated as canonical."})
    espeak = read(root / "artifacts/stage2d/stage2d2_dataset_design/batch1_phonetic_hypotheses/batch1_normalized_hypotheses.json")
    espeak_by_word = {row["normalized_word"]: row for row in espeak["words"]}
    comparisons = []
    for word in PILOT_WORDS:
        acoustic = grouped[word]
        source_rows = []
        for source_id, source_tokens in espeak_by_word[word]["sources"].items():
            distances = [edit_distance(source_tokens, row["analysis_normalized_phone_tokens"]) / max(len(source_tokens), len(row["analysis_normalized_phone_tokens"]), 1) for row in acoustic]
            source_rows.append({"source_id": source_id, "mean_normalized_edit_distance": statistics.mean(distances), "median_normalized_edit_distance": statistics.median(distances), "comparison": "BROAD_PATTERN_AGREEMENT" if statistics.mean(distances) <= 0.5 else "CONFLICT", "note": "Cross-inventory sequence comparison only; not a phoneme-level correctness judgment."})
        comparisons.append({"word": word.title(), "normalized_word": word, "eSpeak_sources": list(espeak_by_word[word]["sources"]), "Allosaurus_prediction_count": len(acoustic), "comparisons": source_rows, "provenance": {"orthographic_g2p": "eSpeak NG", "acoustic_phone_recognizer": "Allosaurus", "human_listening": "SPICOR repeated-word stability"}})
    write(out / "stage2d2f_espeak_acoustic_comparison.json", {"schema_version": "stage2d2f-espeak-acoustic-comparison.v2", "rows": comparisons})
    pressure = feature_evidence(entries, repeat_rows)
    write(out / "stage2d2f_inventory_evidence.json", {"schema_version": "stage2d2f-inventory-evidence.v2", "production_inventory_modified": False, "candidates": pressure, "decision": "No production phone is supported to freeze; repeated recognizer symbols are weak acoustic evidence only."})
    family_specs = {"PUR": ["nagpur", "udhampur"], "GARH": ["chandigarh", "chhattisgarh", "gorakhpur"], "NAGAR": ["srinagar"], "OTHER_DIAGNOSTIC": ["hyderabad", "bengaluru", "banerjee", "jamshedpur"]}
    family_rows = []
    for name, members in family_specs.items():
        family_rows.append({"family": name, "words": [by_word[w]["word"] for w in members], "recognizer_observations": {w: [row["analysis_normalized_phone_tokens"][-3:] for row in grouped[w]] for w in members}, "conclusion": "Descriptive suffix comparison only; no morphological rule or shared canonical phone sequence is inferred.", "production_rule_created": False})
    write(out / "stage2d2f_family_analysis.json", {"schema_version": "stage2d2f-family-analysis.v2", "families": family_rows})
    questions = []
    question_specs = [("SCHWA", "Srinagar", "Do you hear an 'uh'-like central vowel in the target, consistently across the selected contexts?"), ("ASPIRATION", "Chhattisgarh", "Do you hear an extra breath after a stop consonant?"), ("RETROFLEX", "Chandigarh", "Does any tongue-stop sound noticeably different from ordinary English T or D?"), ("W_V", "Bengaluru", "Does the relevant consonant sound closer to V or W?"), ("VOWEL_LENGTH", "Jamshedpur", "Does the vowel sound consistently short or long, beyond speaking-rate differences?"), ("SH_AFFRICATE", "Banerjee", "Does the consonant sound like SH or like an SH-plus-stop affricate?")]
    pressure_by_name = {row["candidate"]: row for row in pressure}
    for feature, word, prompt in question_specs:
        if pressure_by_name[feature]["status"] == "ACOUSTIC_SUPPORT_WEAK":
            questions.append({"feature": feature, "word": word, "plain_language_question": prompt, "evidence_status": pressure_by_name[feature]["status"], "phone_assignment_requested": False})
    write(out / "stage2d2f_minimal_human_questions.json", {"schema_version": "stage2d2f-minimal-human-questions.v2", "question_count": len(questions), "questions": questions, "policy": "Only repeated weak/strong acoustic signals with potential v0 loss are presented; no IPA or Swara label is requested."})
    write(out / "stage2d2f_summary.json", {"schema_version": "stage2d2f-summary.v2", "status": "EXECUTED_ALLOSAURUS_RUN", "pilot_words": [word.title() for word in PILOT_WORDS], "clip_count": len(entries), "recognition_success_count": len(entries), "recognition_failure_count": 0, "repeatability_class_counts": {key: sum(row["classification"] == key for row in repeat_rows) for key in ("ACOUSTIC_PHONE_PATTERN_STABLE", "MOSTLY_STABLE", "UNSTABLE", "RECOGNIZER_UNRELIABLE")}, "human_review_question_count": len(questions), "training_performed": False, "qwen_loaded": False, "swara_phones_v0_modified": False, "canonical_lexicon_modified": False})
    print(json.dumps({"status": "EXECUTED_ALLOSAURUS_RUN", "clips": len(entries), "success": len(entries), "failure": 0, "model": args.model, "repeatability": {key: sum(row["classification"] == key for row in repeat_rows) for key in ("ACOUSTIC_PHONE_PATTERN_STABLE", "MOSTLY_STABLE", "UNSTABLE", "RECOGNIZER_UNRELIABLE")}, "human_questions": len(questions)}, indent=2))


if __name__ == "__main__":
    main()
