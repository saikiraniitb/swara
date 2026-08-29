#!/usr/bin/env python3
"""Build the frozen Stage2D.2 Batch-1 human-listening package.

This script only creates review derivatives and metadata.  It does not modify
the corpus, acoustic-analysis artifacts, pronunciation labels, or models.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from swara.alignment.ctc_forced import (
    ALIGNER_MODEL_ID,
    ALIGNER_REVISION,
    Wav2Vec2ExactTranscriptAligner,
)
from swara.contracts import build_plain_text_request
from swara.data.spicor_audio import SpicorAudioResolver
from swara.frontend.pipeline import Frontend


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review"
DEFAULT_BATCH = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design/stage2d2_review_batch1.json"
DEFAULT_OCCURRENCES = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design/stage2d2_review_occurrence_manifest.jsonl"
DEFAULT_INDEX = REPO_ROOT / "artifacts/stage2d/pronunciation_atlas_v0_1/occurrence_index.jsonl"
DEFAULT_ALIGNMENT = REPO_ROOT / "artifacts/stage2d/pronunciation_atlas_v0_1/acoustic_consistency/stage2d1b_alignment_report.json"
DEFAULT_ALIGNER = REPO_ROOT / "models/alignment/facebook-wav2vec2-base-960h"
DEFAULT_INVENTORY = REPO_ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
DEFAULT_ARCHIVE = Path("/Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz")
DEFAULT_SELECTED_CACHE = REPO_ROOT / "data/stage2d_spicor_selected_audio"

SPECIAL_QUESTIONS = {
    "hyderabad": "Is the middle consonant/vowel realization stable across contexts?",
    "bengaluru": "Is the final 'luru/luru-like' portion stable across occurrences?",
    "chandigarh": "Is the Chandigarh ending stable, including the final consonant realization?",
    "chhattisgarh": "Do you hear one stable consonant pattern, or clearly different recurring realizations?",
    "banerjee": "Is the initial/middle vowel pattern stable?",
    "ahmedabad": "Is the Ahmedabad internal vowel/consonant realization stable?",
    "chatterjee": "Is the '-jee' ending acoustically consistent?",
    "mukherjee": "Is the '-jee' ending acoustically consistent?",
    "srinagar": "Does 'nagar' behave consistently both independently and inside Srinagar?",
    "nagar": "Does 'nagar' behave consistently both independently and inside Srinagar?",
}
PUR_ENDING = {"nagpur", "jaipur", "raipur", "udhampur", "sultanpur", "bilaspur"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"source audio must be finite mono audio: {path}")
    return waveform, int(sample_rate)


def write_clip(source_path: Path, output_path: Path, start: float, end: float, padding: float) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(source_path)
    first = max(0, int(np.floor((start - padding) * info.samplerate)))
    last = min(info.frames, int(np.ceil((end + padding) * info.samplerate)))
    if not 0 <= first < last <= info.frames:
        raise ValueError(f"invalid clipped interval for {source_path}: {start}, {end}, {padding}")
    samples, sample_rate = sf.read(source_path, start=first, stop=last, dtype="float32", always_2d=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, samples, sample_rate, subtype="PCM_16")
    return {
        "sample_rate_hz": int(sample_rate),
        "frame_count": int(last - first),
        "duration_seconds": float((last - first) / sample_rate),
        "source_start_sample": first,
        "source_end_sample": last,
        "padding_seconds": padding,
    }


def find_unit_index(sequence: Any, occurrence: Mapping[str, Any]) -> int:
    start = int(occurrence["source_span_start"])
    end = int(occurrence["source_span_end"])
    for index, token in enumerate(sequence.tokens):
        if token.kind.value == "grapheme" and token.source_span is not None and token.source_span.start == start and token.source_span.end == end:
            return index
    for index, token in enumerate(sequence.tokens):
        if token.kind.value == "grapheme" and token.source_span is not None and token.source_span.start <= start < token.source_span.end:
            return index
    raise ValueError(f"target span does not map to a frontend grapheme: {occurrence['occurrence_id']}")


def lexical_span(lexical_spans: Any, unit_index: int) -> Any:
    for span in lexical_spans:
        if span.linguistic_unit_index == unit_index:
            return span
    raise ValueError(f"aligner returned no lexical span for frontend unit {unit_index}")


def listening_question(word: str) -> tuple[str, str | None]:
    common = "Across these full utterances, does this word have the same underlying pronunciation, with differences limited to natural prosody, coarticulation, stress or speaking rate?"
    normalized = word.casefold()
    if normalized in PUR_ENDING:
        return common, "Is the '-pur' portion consistently realized?"
    return common, SPECIAL_QUESTIONS.get(normalized)


def relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def highlight_transcript(transcript: str, start: int, end: int) -> str:
    if not 0 <= start < end <= len(transcript):
        return html.escape(transcript)
    return html.escape(transcript[:start]) + "<mark>" + html.escape(transcript[start:end]) + "</mark>" + html.escape(transcript[end:])


def role_for_index(index: int) -> str:
    return ("MEDOID", "NEAR", "FAR", "DIVERSE_CONTEXT", "DIVERSE_CONTEXT")[min(index, 4)]


def build_package(
    *,
    batch_path: Path = DEFAULT_BATCH,
    occurrence_manifest_path: Path = DEFAULT_OCCURRENCES,
    occurrence_index_path: Path = DEFAULT_INDEX,
    alignment_path: Path = DEFAULT_ALIGNMENT,
    output_dir: Path = DEFAULT_OUTPUT,
    repo_root: Path = REPO_ROOT,
    aligner_model: Path = DEFAULT_ALIGNER,
    inventory_path: Path = DEFAULT_INVENTORY,
    archive_path: Path = DEFAULT_ARCHIVE,
    selected_cache_root: Path = DEFAULT_SELECTED_CACHE,
) -> dict[str, Any]:
    batch = read_json(batch_path)
    batch_words = [dict(row) for row in batch["words"]]
    occurrence_rows = read_jsonl(occurrence_manifest_path)
    occurrence_by_id = {row["occurrence_id"]: row for row in occurrence_rows}
    inventory_occurrences: dict[str, dict[str, Any]] = {}
    for inventory_row in read_jsonl(occurrence_index_path):
        for lexical in inventory_row.get("lexical_occurrences", []):
            word_index, surface, normalized, span, preceding, following, _flags = lexical
            occurrence_id = f"{inventory_row['utterance_id']}:word:{int(word_index):04d}"
            inventory_occurrences[occurrence_id] = {
                "occurrence_id": occurrence_id,
                "utterance_id": inventory_row["utterance_id"],
                "transcript": inventory_row["full_transcript"],
                "word_index": int(word_index),
                "surface_form": surface,
                "normalized_word": normalized,
                "source_span_start": int(span[0]),
                "source_span_end": int(span[1]),
                "preceding_word": preceding,
                "following_word": following,
            }
    existing_alignment = read_json(alignment_path)
    alignment_by_id = {row["occurrence_id"]: row for row in existing_alignment.get("rows", []) if row.get("status") == "ALIGNED"}
    inventory = {row["source_id"]: row for row in read_jsonl(inventory_path)}
    resolver = SpicorAudioResolver(inventory, repo_root=repo_root, archive_path=archive_path, selected_cache_root=selected_cache_root)

    intended_words = [str(row["normalized_word"]) for row in batch_words]
    if len(intended_words) != 25 or len(set(intended_words)) != 25:
        raise ValueError("Batch-1 must contain exactly 25 distinct words")
    selected_ids = [occurrence_id for word in batch_words for occurrence_id in word["occurrence_ids"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Batch-1 contains duplicate occurrence IDs")
    missing_rows = sorted(set(selected_ids) - set(occurrence_by_id))
    if missing_rows:
        raise ValueError(f"Batch-1 occurrence manifest is missing IDs: {missing_rows[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frontend = Frontend()
    aligner: Wav2Vec2ExactTranscriptAligner | None = None
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    alignment_sources: Counter[str] = Counter()

    for batch_word in batch_words:
        word = str(batch_word["word"])
        normalized = str(batch_word["normalized_word"])
        for position, occurrence_id in enumerate(batch_word["occurrence_ids"]):
            review = occurrence_by_id[occurrence_id]
            occurrence = inventory_occurrences.get(occurrence_id)
            if occurrence is None:
                failures.append({"word": word, "occurrence_id": occurrence_id, "error": "missing occurrence index row"})
                continue
            source_resolution = resolver.resolve(str(occurrence["utterance_id"]))
            if source_resolution.selected_audio_path is None:
                raise FileNotFoundError(f"resolver did not resolve selected audio: {occurrence['utterance_id']}")
            source_path = source_resolution.selected_audio_path.resolve()
            source_value = str(review["audio_resolver_path"])
            try:
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                import soundfile as sf

                source_info = sf.info(source_path)
                source_duration = float(source_info.frames / source_info.samplerate)
                alignment = alignment_by_id.get(occurrence_id)
                if alignment is not None:
                    start = float(alignment["word_start_seconds"])
                    end = float(alignment["word_end_seconds"])
                    confidence = float(alignment["alignment_confidence"])
                    alignment_method = "existing_stage2d1b_exact_transcript_ctc"
                    alignment_sources[alignment_method] += 1
                else:
                    if aligner is None:
                        aligner = Wav2Vec2ExactTranscriptAligner(aligner_model, device="cpu")
                    waveform, sample_rate = load_audio(source_path)
                    if sample_rate != 16_000:
                        import librosa

                        waveform_16khz = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16_000)
                    else:
                        waveform_16khz = waveform
                    sequence = frontend.compile(build_plain_text_request(str(occurrence["transcript"])))
                    unit_index = find_unit_index(sequence, occurrence)
                    _, lexical = aligner.align(waveform_16khz, sequence)
                    span = lexical_span(lexical, unit_index)
                    start = float(span.start_seconds)
                    end = float(span.end_seconds)
                    confidence = float(span.confidence)
                    alignment_method = "stage2d2_batch1_local_exact_transcript_ctc"
                    alignment_sources[alignment_method] += 1
                if not 0 <= start < end <= source_duration + 0.05:
                    raise ValueError(f"aligned interval outside source duration: {start}, {end}, {source_duration}")
                word_dir = output_dir / safe_name(normalized)
                stem = f"{safe_name(normalized)}__{safe_name(str(occurrence['utterance_id']))}__{role_for_index(position).lower()}"
                word_meta = write_clip(source_path, word_dir / f"{stem}__word.wav", start, end, 0.075)
                context_meta = write_clip(source_path, word_dir / f"{stem}__context.wav", start, end, 0.300)
                full_reference = relative_path(source_path, output_dir)
                word_reference = relative_path(word_dir / f"{stem}__word.wav", output_dir)
                context_reference = relative_path(word_dir / f"{stem}__context.wav", output_dir)
                common_question, special_question = listening_question(word)
                entries.append({
                    "word": word,
                    "normalized_word": normalized,
                    "role": role_for_index(position),
                    "occurrence_id": occurrence_id,
                    "utterance_id": occurrence["utterance_id"],
                    "corpus_recurrence": int(review["recurrence_count"]),
                    "source_audio_duration_seconds": source_duration,
                    "selected_review_occurrence_count": len(batch_word["occurrence_ids"]),
                    "transcript": occurrence["transcript"],
                    "preceding_word": occurrence.get("preceding_word"),
                    "following_word": occurrence.get("following_word"),
                    "target_word_index": occurrence.get("word_index"),
                    "target_char_span": [int(occurrence["source_span_start"]), int(occurrence["source_span_end"])],
                    "aligned_start_seconds": start,
                    "aligned_end_seconds": end,
                    "alignment_confidence": confidence,
                    "alignment_model": ALIGNER_MODEL_ID,
                    "alignment_revision": ALIGNER_REVISION,
                    "alignment_method": alignment_method,
                    "source_audio_path": source_value,
                    "resolved_audio_source_type": source_resolution.source_type,
                    "full_audio_path": full_reference,
                    "context_audio_path": context_reference,
                    "word_only_audio_path": word_reference,
                    "clip_metadata": {"word_only": word_meta, "context": context_meta},
                    "current_swara_v0_representability": review.get("representability"),
                    "current_pronunciation_candidate": None,
                    "candidate_provenance": "Stage2D.1 Tier-2 review queue; no automatic phone assignment",
                    "primary_listening_question": common_question,
                    "special_listening_question": special_question,
                })
            except Exception as exc:
                failures.append({"word": word, "occurrence_id": occurrence_id, "error_type": type(exc).__name__, "error": str(exc)})

    if failures:
        raise RuntimeError(json.dumps({"batch1_clip_generation_failures": failures}, ensure_ascii=False))

    entries.sort(key=lambda row: (int(next(item["review_rank"] for item in occurrence_rows if item["occurrence_id"] == row["occurrence_id"])), row["occurrence_id"]))
    by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_word[entry["normalized_word"]].append(entry)

    words: list[dict[str, Any]] = []
    for batch_word in batch_words:
        normalized = str(batch_word["normalized_word"])
        word_entries = by_word[normalized]
        words.append({
            "word": batch_word["word"],
            "normalized_word": normalized,
            "corpus_recurrence": int(batch_word["occurrence_count"]),
            "usable_audio_recurrence": int(batch_word["occurrence_count"]),
            "selected_review_occurrence_count": len(word_entries),
            "acoustic_class_from_stage2d1b": "NOT_AVAILABLE_FOR_BATCH1_WORD",
            "relative_variability_from_stage2d1b": None,
            "primary_listening_question": word_entries[0]["primary_listening_question"],
            "special_listening_question": word_entries[0]["special_listening_question"],
            "entries": word_entries,
        })

    index = {
        "schema_version": "stage2d2-batch1-human-review.v1",
        "source_batch_manifest": relative_path(batch_path, output_dir),
        "source_occurrence_manifest": relative_path(occurrence_manifest_path, output_dir),
        "package_root": ".",
        "audio_view_order": ["FULL_UTTERANCE", "CONTEXT", "WORD_ONLY"],
        "word_only_padding_seconds": 0.075,
        "context_padding_seconds": 0.300,
        "phone_assignment": "NOT_REQUESTED_IN_BATCH1_REVIEW; all phone fields remain null",
        "words": words,
        "entry_count": len(entries),
        "alignment_success_count": len(entries),
        "alignment_failure_count": 0,
        "alignment_sources": dict(sorted(alignment_sources.items())),
    }
    write_json(output_dir / "batch1_human_review_index.json", index)
    write_json(output_dir / "batch1_human_review_summary.json", {
        "schema_version": "stage2d2-batch1-human-review-summary.v1",
        "words": [row["word"] for row in words],
        "word_count": len(words),
        "total_review_occurrences": len(entries),
        "words_with_5_reviewed_samples": [row["word"] for row in words if row["selected_review_occurrence_count"] >= 5],
        "words_with_3_to_4_reviewed_samples": [row["word"] for row in words if 3 <= row["selected_review_occurrence_count"] <= 4],
        "words_with_fewer_than_3_reviewed_samples": [row["word"] for row in words if row["selected_review_occurrence_count"] < 3],
        "alignment_success_count": len(entries),
        "alignment_failure_count": 0,
        "audio_generation_success_count": len(entries) * 2,
        "audio_generation_failure_count": 0,
        "full_utterance_audio": "REFERENCED_FROM_SELECTED_SPICOR_AUDIO; not duplicated",
        "analysis_conclusions_changed": False,
        "training_performed": False,
        "qwen_loaded": False,
        "swara_phones_v0_modified": False,
    })
    write_markdown(output_dir / "batch1_human_review_listening_index.md", words)
    write_html(output_dir / "human_review.html", words)
    write_decision_template(output_dir / "batch1_human_review_decisions_template.md", words)
    return {
        "word_count": len(words),
        "entry_count": len(entries),
        "word_only_clip_count": len(entries),
        "context_clip_count": len(entries),
        "full_utterance_reference_count": len(entries),
        "alignment_sources": dict(sorted(alignment_sources.items())),
        "failures": failures,
    }


def write_markdown(path: Path, words: list[dict[str, Any]]) -> None:
    lines = ["# Stage2D.2 Batch-1 Human Listening Index", "", "Listen in this order for every occurrence: **FULL UTTERANCE**, **CONTEXT**, **WORD ONLY**.", "", "No phone transcription is requested in this review.", ""]
    for word in words:
        lines.extend([f"## {word['word']}", "", f"Corpus recurrence: {word['corpus_recurrence']}", f"Usable audio recurrence: {word['usable_audio_recurrence']}", f"Review occurrences: {word['selected_review_occurrence_count']}", f"Primary question: {word['primary_listening_question']}"])
        if word["special_listening_question"]:
            lines.append(f"Special question: {word['special_listening_question']}")
        lines.append("")
        for entry in word["entries"]:
            lines.extend([
                f"### {entry['role']} — {entry['utterance_id']}", "",
                f"Transcript: {entry['transcript']}",
                f"Preceding/following: {entry['preceding_word']} / {entry['following_word']}",
                f"Aligned interval: {entry['aligned_start_seconds']:.6f}–{entry['aligned_end_seconds']:.6f} s",
                f"Alignment confidence: {entry['alignment_confidence']:.6f}",
                f"Full utterance: [{entry['full_audio_path']}]({entry['full_audio_path']})",
                f"Context: [{entry['context_audio_path']}]({entry['context_audio_path']})",
                f"Word only: [{entry['word_only_audio_path']}]({entry['word_only_audio_path']})", "",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, words: list[dict[str, Any]]) -> None:
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Stage2D.2 Batch-1 Human Review</title><style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto;line-height:1.4}section{border-top:2px solid #888;margin-top:2rem;padding-top:1rem}article{border:1px solid #ccc;padding:1rem;margin:1rem 0}audio{width:min(100%,700px)}.question{background:#f4f4f4;padding:.6rem}mark{background:#ffeb80}</style></head><body>", "<h1>Stage2D.2 Batch-1 Human Listening Review</h1>", "<p>Listen in order: <strong>FULL UTTERANCE</strong>, <strong>CONTEXT</strong>, <strong>WORD ONLY</strong>. Do not assign phones in this stage.</p>"]
    for word in words:
        parts.append(f"<section><h2>{html.escape(word['word'])}</h2><p>Corpus recurrence: {word['corpus_recurrence']} · Usable audio recurrence: {word['usable_audio_recurrence']} · Review occurrences: {word['selected_review_occurrence_count']}</p><p class='question'><strong>Primary question:</strong> {html.escape(word['primary_listening_question'])}</p>")
        if word["special_listening_question"]:
            parts.append(f"<p class='question'><strong>Special question:</strong> {html.escape(word['special_listening_question'])}</p>")
        for entry in word["entries"]:
            transcript = highlight_transcript(entry["transcript"], entry["target_char_span"][0], entry["target_char_span"][1])
            parts.append(f"<article><h3>{html.escape(entry['role'])} — {html.escape(entry['utterance_id'])}</h3><p><strong>Transcript:</strong> {transcript}</p><p>Preceding: {html.escape(str(entry['preceding_word']))} · Following: {html.escape(str(entry['following_word']))}</p><p>Aligned: {entry['aligned_start_seconds']:.6f}–{entry['aligned_end_seconds']:.6f} s · confidence {entry['alignment_confidence']:.6f}</p><p><strong>Full utterance</strong></p><audio controls preload='none' src='{html.escape(entry['full_audio_path'])}'></audio><p><strong>Context</strong></p><audio controls preload='none' src='{html.escape(entry['context_audio_path'])}'></audio><p><strong>Word only</strong></p><audio controls preload='none' src='{html.escape(entry['word_only_audio_path'])}'></audio></article>")
        parts.append("</section>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_decision_template(path: Path, words: list[dict[str, Any]]) -> None:
    lines = ["# Stage2D.2 Batch-1 Human Review Decisions", "", "Allowed per-word verdicts: `CANONICAL_STABLE`, `LIKELY_STABLE`, `VARIANT_PRESENT`, `INSUFFICIENT_EVIDENCE`, `UNSURE`.", "", "Do not enter phone strings in this stage.", ""]
    for word in words:
        lines.extend([f"## {word['word']}", "", "| word | utterance_id | role | occurrence_decision | notes |", "|---|---|---|---|---|"])
        for entry in word["entries"]:
            lines.append(f"| {entry['word']} | {entry['utterance_id']} | {entry['role']} |  |  |")
        lines.extend(["", "**FINAL VERDICT:** ", "", "**NOTES:**", "", "**OBSERVED DISTINCTIONS:**", "", "---", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--occurrences", type=Path, default=DEFAULT_OCCURRENCES)
    parser.add_argument("--occurrence-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--aligner-model", type=Path, default=DEFAULT_ALIGNER)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--selected-cache-root", type=Path, default=DEFAULT_SELECTED_CACHE)
    args = parser.parse_args()
    print(json.dumps(build_package(batch_path=args.batch.resolve(), occurrence_manifest_path=args.occurrences.resolve(), occurrence_index_path=args.occurrence_index.resolve(), alignment_path=args.alignment.resolve(), output_dir=args.output_dir.resolve(), repo_root=args.repo_root.resolve(), aligner_model=args.aligner_model.resolve(), inventory_path=args.inventory.resolve(), archive_path=args.archive.resolve(), selected_cache_root=args.selected_cache_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
