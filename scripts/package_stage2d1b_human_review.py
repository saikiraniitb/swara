#!/usr/bin/env python3
"""Create pointer-preserving Stage2D.1B human-listening clips and indexes."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


ROLE_MAP = {
    "medoid_typical": "medoid",
    "nearest_to_medoid": "near",
    "farthest_from_medoid": "far",
}
DEFAULT_ROOT = Path("artifacts/stage2d/pronunciation_atlas_v0_1/acoustic_consistency")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value or "unknown"


def _question(word: str) -> str:
    common = "Across these full utterances, does the target word have the same underlying pronunciation, with only natural changes in prosody, speaking rate, stress, and coarticulation?"
    return {
        "singh": "Does the final realization sound consistently the same, or do some utterances contain a clearly different final release / h-like ending?",
        "kumar": "Is the core vowel/consonant pronunciation stable across full sentence contexts?",
        "mumbai": "Is the final vowel/diphthong realization stable across full sentence contexts?",
        "agrawal": "Do the available occurrences sound like the same underlying pronunciation or a meaningful pronunciation variant?",
    }.get(word.casefold(), common)


def _clip_bounds(start: float, end: float, padding: float, frames: int, sample_rate: int) -> tuple[int, int]:
    first = max(0, int((start - padding) * sample_rate))
    last = min(frames, int((end + padding) * sample_rate + 0.999999))
    if last <= first:
        raise ValueError("clip interval is empty after source-bound clipping")
    return first, last


def _write_clip(source_path: Path, output_path: Path, start: float, end: float, padding: float) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(source_path)
    first, last = _clip_bounds(start, end, padding, info.frames, info.samplerate)
    samples, sample_rate = sf.read(source_path, start=first, stop=last, dtype="float32", always_2d=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, samples, sample_rate, subtype="PCM_16")
    return {"sample_rate_hz": int(sample_rate), "frame_count": int(last - first), "duration_seconds": float((last - first) / sample_rate), "source_start_sample": first, "source_end_sample": last, "padding_seconds": padding}


def _copy_full_utterance(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Copy one selected canonical WAV byte-for-byte for browser review."""
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    info = sf.info(output_path)
    return {
        "sample_rate_hz": int(info.samplerate),
        "frame_count": int(info.frames),
        "duration_seconds": float(info.frames / info.samplerate),
        "byte_size": int(output_path.stat().st_size),
        "copied_byte_for_byte": True,
    }


def build_package(root: str | Path = DEFAULT_ROOT, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    repo_root = Path(repo_root).resolve()
    clip_root = root / "human_review_clips"
    clip_root.mkdir(parents=True, exist_ok=True)
    review_rows = _read_jsonl(root / "human_review_manifest.jsonl")
    sample_rows = {row["occurrence_id"]: row for row in _read_jsonl(root / "stage2d1b_occurrence_sample.jsonl")}
    alignment_rows = {row["occurrence_id"]: row for row in _read_json(root / "stage2d1b_alignment_report.json")["rows"]}
    consistency_rows = {row["normalized_word"]: row for row in _read_json(root / "stage2d1b_word_consistency.json")["words"]}
    target_rows = {row["normalized_word"]: row for row in _read_json(root / "stage2d1b_target_set.json")["selection"]}

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for review in sorted(review_rows, key=lambda row: row["review_item_id"]):
        occurrence_id = review["occurrence_id"]
        occurrence = sample_rows.get(occurrence_id)
        alignment = alignment_rows.get(occurrence_id)
        if occurrence is None or alignment is None:
            failures.append({"review_item_id": review["review_item_id"], "occurrence_id": occurrence_id, "error": "missing occurrence or alignment metadata"})
            continue
        source_path = Path(review["audio_path"])
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if not source_path.is_file():
            failures.append({"review_item_id": review["review_item_id"], "occurrence_id": occurrence_id, "error": f"missing source WAV: {source_path}"})
            continue
        role = ROLE_MAP.get(review["role"], "other")
        word_dir = clip_root / _safe_name(review["normalized_word"])
        stem = f"{_safe_name(review['normalized_word'])}__{_safe_name(occurrence['utterance_id'])}__{role}"
        try:
            full_meta = _copy_full_utterance(source_path, word_dir / f"{stem}__full.wav")
            word_meta = _write_clip(source_path, word_dir / f"{stem}__word.wav", float(alignment["word_start_seconds"]), float(alignment["word_end_seconds"]), 0.075)
            context_meta = _write_clip(source_path, word_dir / f"{stem}__context.wav", float(alignment["word_start_seconds"]), float(alignment["word_end_seconds"]), 0.300)
        except Exception as exc:
            failures.append({"review_item_id": review["review_item_id"], "occurrence_id": occurrence_id, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        rel_full = (word_dir / f"{stem}__full.wav").relative_to(root).as_posix()
        rel_word = (word_dir / f"{stem}__word.wav").relative_to(root).as_posix()
        rel_context = (word_dir / f"{stem}__context.wav").relative_to(root).as_posix()
        entries.append({
            "review_item_id": review["review_item_id"],
            "target_word": review["target_word"],
            "normalized_word": review["normalized_word"],
            "role": role,
            "utterance_id": occurrence["utterance_id"],
            "occurrence_id": occurrence_id,
            "transcript": occurrence["full_transcript"],
            "preceding_word": occurrence["preceding_word"],
            "following_word": occurrence["following_word"],
            "source_audio_path": review["audio_path"],
            "source_audio_path_resolved": str(source_path),
            "target_source_span": [occurrence["source_span_start"], occurrence["source_span_end"]],
            "aligned_start_seconds": alignment["word_start_seconds"],
            "aligned_end_seconds": alignment["word_end_seconds"],
            "alignment_confidence": alignment["alignment_confidence"],
            "alignment_model": alignment["alignment_model"],
            "alignment_revision": alignment["alignment_revision"],
            "clip_paths": {"full_utterance": rel_full, "context": rel_context, "word_only": rel_word},
            "full_utterance_audio_path": rel_full,
            "context_clip_path": rel_context,
            "word_only_clip_path": rel_word,
            "clip_metadata": {"full_utterance": full_meta, "word_only": word_meta, "context": context_meta},
        })

    by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_word[entry["normalized_word"]].append(entry)
    word_sections: list[dict[str, Any]] = []
    for word in sorted(by_word):
        consistency = consistency_rows[word]
        target = target_rows[word]
        word_sections.append({
            "target_word": target["target_word"],
            "normalized_word": word,
            "corpus_occurrence_count": target["corpus_occurrence_count"],
            "analyzed_sample_count": consistency["usable_aligned_occurrence_count"],
            "acoustic_class": consistency["classification"],
            "relative_variability": consistency["relative_variability_score"],
            "listening_question": _question(word),
            "selected_review_entries": sorted(by_word[word], key=lambda item: item["review_item_id"]),
        })
    index = {
        "schema_version": "stage2d1b-human-review.v1",
        "source_review_manifest": str((root / "human_review_manifest.jsonl").as_posix()),
        "clip_directory": "human_review_clips",
        "word_only_padding_seconds": 0.075,
        "context_padding_seconds": 0.300,
        "default_listening_question": "Across these full utterances, does the target word have the same underlying pronunciation, with only natural changes in prosody, speaking rate, stress, and coarticulation?",
        "words": word_sections,
        "entry_count": len(entries),
        "failure_count": len(failures),
    }
    _dump_json(root / "human_review_listening_index.json", index)
    _write_markdown(root / "human_review_listening_index.md", word_sections)
    _write_html(root / "human_review.html", word_sections)
    _write_decision_template(root / "human_review_decisions_template.md", word_sections)
    _dump_json(root / "human_review_clip_generation_report.json", {"schema_version": "stage2d1b-human-review-clips.v1", "selected_entry_count": len(review_rows), "successful_entry_count": len(entries), "word_only_clip_count": len(entries), "context_clip_count": len(entries), "failures": failures, "source_audio_modified": False})
    return {"selected_entries": len(review_rows), "successful_entries": len(entries), "word_only_clips": len(entries), "context_clips": len(entries), "failures": failures, "review_words": [section["target_word"] for section in word_sections]}


def _dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, sections: list[dict[str, Any]]) -> None:
    lines = ["# Stage2D.1B Human Listening Index", "", "Review only the extracted clips. The canonical SPICOR WAVs are unchanged.", ""]
    for section in sections:
        lines.extend([f"## {section['target_word']}", "", f"Corpus occurrences: {section['corpus_occurrence_count']}", f"Analyzed sample: {section['analyzed_sample_count']}", f"Acoustic class: `{section['acoustic_class']}`", f"Relative variability: `{section['relative_variability']}`", "", f"Listening question: {section['listening_question']}", ""])
        for entry in section["selected_review_entries"]:
            lines.extend([f"### {entry['role']} — {entry['utterance_id']}", "", f"Transcript: {entry['transcript']}", f"Preceding/following: {entry['preceding_word']} / {entry['following_word']}", f"Aligned interval: {entry['aligned_start_seconds']:.6f}–{entry['aligned_end_seconds']:.6f} s", f"Confidence: {entry['alignment_confidence']:.6f}", f"Full utterance: [{entry['clip_paths']['full_utterance']}]({entry['clip_paths']['full_utterance']})", f"Context: [{entry['clip_paths']['context']}]({entry['clip_paths']['context']})", f"Word-only: [{entry['clip_paths']['word_only']}]({entry['clip_paths']['word_only']})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(path: Path, sections: list[dict[str, Any]]) -> None:
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Stage2D.1B Human Review</title></head><body>", "<h1>Stage2D.1B Human Listening Review</h1>", "<p>Review-only clips. Canonical SPICOR audio and analysis conclusions are unchanged.</p>"]
    for section in sections:
        parts.append(f"<section><h2>{html.escape(section['target_word'])}</h2><p>Corpus occurrences: {section['corpus_occurrence_count']} · Analyzed: {section['analyzed_sample_count']} · Class: {html.escape(section['acoustic_class'])}</p><p>{html.escape(section['listening_question'])}</p>")
        for entry in section["selected_review_entries"]:
            parts.append(f"<article><h3>{html.escape(entry['role'])} — {html.escape(entry['utterance_id'])}</h3><p>{_highlight_transcript(entry)}</p><p>{entry['aligned_start_seconds']:.6f}–{entry['aligned_end_seconds']:.6f} s · confidence {entry['alignment_confidence']:.6f}</p><p>Full utterance</p><audio controls preload='none' src='{html.escape(entry['clip_paths']['full_utterance'])}'></audio><p>Context</p><audio controls preload='none' src='{html.escape(entry['clip_paths']['context'])}'></audio><p>Word-only</p><audio controls preload='none' src='{html.escape(entry['clip_paths']['word_only'])}'></audio></article>")
        parts.append("</section>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _highlight_transcript(entry: dict[str, Any]) -> str:
    transcript = entry["transcript"]
    start, end = entry["target_source_span"]
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(transcript)):
        return html.escape(transcript)
    return html.escape(transcript[:start]) + "<mark>" + html.escape(transcript[start:end]) + "</mark>" + html.escape(transcript[end:])


def _write_decision_template(path: Path, sections: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage2D.1B Human Review Decisions",
        "",
        "Review the full utterance first, then context and word-only clips. Do not assign phone labels here.",
        "",
    ]
    for section in sections:
        lines.extend([f"## {section['target_word']}", "", "| word | utterance_id | role | decision | notes |", "|---|---|---|---|---|"])
        for entry in section["selected_review_entries"]:
            lines.append(f"| {entry['target_word']} | {entry['utterance_id']} | {entry['role']} |  |  |")
        lines.extend(["", "Per-word verdict: ", "", "Allowed verdicts: `CANONICAL_STABLE`, `VARIANT_PRESENT`, `INSUFFICIENT_EVIDENCE`, `UNSURE`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(build_package(args.root, args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
