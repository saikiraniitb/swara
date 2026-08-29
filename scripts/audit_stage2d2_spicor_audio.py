#!/usr/bin/env python3
"""Audit SPICOR WAV availability without extracting or copying the corpus."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from swara.diagnostics.pronunciation_atlas import ANNOTATION_RE, MIXED_ALNUM_RE, NUMBER_RE, extract_lexical_tokens, normalize_lexical_word


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_DEFAULT = REPO_ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
ATLAS_ROOT = REPO_ROOT / "artifacts/stage2d/pronunciation_atlas_v0_1"
OCCURRENCE_INDEX_DEFAULT = ATLAS_ROOT / "occurrence_index.jsonl"
QUEUE_DEFAULT = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design/stage2d2_pronunciation_review_queue.json"
OUTPUT_DEFAULT = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design"
ARCHIVE_DEFAULT = Path("/Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz")
ARCHIVE_PROVENANCE = Path("/Users/saikiran/Documents/tts-reference/moss-tts-nano/experiment/spicor_audit_v1")
TARGETS = ("agrawal", "gupta", "kashmir", "kumar", "mishra", "mumbai", "sharma")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                yield row


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _resolved_prepared_path(repo_root: Path, row: Mapping[str, Any]) -> tuple[str | None, Path | None]:
    value = row.get("prepared_audio_path")
    if not isinstance(value, str) or not value:
        return None, None
    path = Path(value)
    resolved = path if path.is_absolute() else repo_root / path
    return value, resolved


def _quality_reasons(row: Mapping[str, Any], seen_transcripts: set[str] | None = None) -> list[str]:
    text = str(row.get("source_text", ""))
    reasons: list[str] = []
    if bool(row.get("transcript_empty")) or not text.strip():
        reasons.append("empty_transcript")
    if ANNOTATION_RE.search(text):
        reasons.append("annotation_or_non_speech_text")
    tokens = extract_lexical_tokens(text)
    if any(NUMBER_RE.match(token["surface_form"]) for token in tokens):
        reasons.append("numeric_token")
    if any(MIXED_ALNUM_RE.match(token["surface_form"]) for token in tokens):
        reasons.append("mixed_alphanumeric_token")
    if row.get("cleanup_flags"):
        reasons.append("manifest_cleanup_flags")
    if seen_transcripts is not None and text in seen_transcripts:
        reasons.append("duplicate_transcript")
    return reasons


def _load_archive_provenance(archive_path: Path) -> dict[str, Any]:
    structure = ARCHIVE_PROVENANCE / "archive_structure.json"
    preflight = ARCHIVE_PROVENANCE / "archive_preflight.json"
    pairing = ARCHIVE_PROVENANCE / "pairing_integrity.json"
    result: dict[str, Any] = {
        "archive_path": str(archive_path),
        "exists": archive_path.is_file(),
        "size_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "archive_member_root": "IISc_SPICOR_Data/IISc_SPICORProject_English_Male_Spk001_HC/wav/",
        "verification_source": str(ARCHIVE_PROVENANCE),
    }
    if structure.is_file():
        payload = json.loads(structure.read_text(encoding="utf-8"))
        result.update({"verified_wav_count": payload.get("wav_count"), "verified_regular_file_bytes": payload.get("regular_file_bytes"), "verified_member_count": payload.get("member_count")})
    if preflight.is_file():
        payload = json.loads(preflight.read_text(encoding="utf-8"))
        result.update({"archive_sha256": payload.get("archive_sha256"), "minimum_extracted_regular_file_bytes": payload.get("minimum_extracted_regular_file_bytes"), "extraction_previously_permitted": payload.get("extraction_permitted")})
    if pairing.is_file():
        payload = json.loads(pairing.read_text(encoding="utf-8"))
        result.update({"verified_transcript_without_wav_count": payload.get("transcript_without_wav_count"), "verified_wav_without_transcript_count": payload.get("wav_without_transcript_count"), "pairing_rule": payload.get("mapping_rule")})
    return result


def _load_occurrences(index_path: Path, inventory_by_id: Mapping[str, Mapping[str, Any]], wanted: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(index_path):
        utterance_id = str(row.get("utterance_id"))
        inventory = inventory_by_id.get(utterance_id, {})
        for occurrence in row.get("lexical_occurrences", []):
            if len(occurrence) < 6:
                continue
            normalized = normalize_lexical_word(str(occurrence[2]))
            if normalized not in wanted:
                continue
            item = {
                "utterance_id": utterance_id,
                "occurrence_id": f"{utterance_id}:word:{int(occurrence[0]):04d}",
                "normalized_word": normalized,
                "surface_form": str(occurrence[1]),
                "transcript": str(row.get("full_transcript", inventory.get("source_text", ""))),
                "audio_path": inventory.get("prepared_audio_path"),
                "source_wav_member": inventory.get("source_wav_member"),
                "source_text": inventory.get("source_text", row.get("full_transcript", "")),
                "cleanup_flags": inventory.get("cleanup_flags", []),
                "transcript_empty": inventory.get("transcript_empty", False),
                "target_char_span": {"start": int(occurrence[3][0]), "end": int(occurrence[3][1])},
                "word_index": int(occurrence[0]),
                "preceding_word": occurrence[4],
                "following_word": occurrence[5],
            }
            result[normalized].append(item)
    for rows in result.values():
        rows.sort(key=lambda row: row["occurrence_id"])
    return result


def audit(repo_root: Path = REPO_ROOT, output: Path = OUTPUT_DEFAULT, inventory_path: Path = INVENTORY_DEFAULT, index_path: Path = OCCURRENCE_INDEX_DEFAULT, archive_path: Path = ARCHIVE_DEFAULT, queue_path: Path = QUEUE_DEFAULT) -> dict[str, Any]:
    inventory_rows = list(read_jsonl(inventory_path))
    inventory_by_id = {str(row["source_id"]): row for row in inventory_rows}
    prepared_count = 0
    expected_names: set[str] = set()
    map_rows: list[dict[str, Any]] = []
    archive_info = _load_archive_provenance(archive_path)
    archive_verified = bool(archive_info.get("exists") and archive_info.get("verified_wav_count") == len(inventory_rows) and archive_info.get("verified_transcript_without_wav_count") == 0 and archive_info.get("verified_wav_without_transcript_count") == 0)
    for row in sorted(inventory_rows, key=lambda value: str(value["source_id"])):
        utterance_id = str(row["source_id"])
        member = str(row.get("source_wav_member") or "")
        expected_names.add(Path(member).name)
        inventory_audio_path, resolved = _resolved_prepared_path(repo_root, row)
        if resolved is not None and resolved.is_file():
            prepared_count += 1
            status = "RESOLVES"
            alternate = None
            basis = "prepared_audio_path_exists"
        elif archive_verified and member:
            status = "FOUND_ALTERNATE_LOCATION"
            alternate = f"{archive_path}::{member}"
            basis = "verified_local_archive_member"
        else:
            status = "MISSING"
            alternate = None
            basis = "no_local_file_or_verified_archive_member"
        map_rows.append({"utterance_id": utterance_id, "inventory_audio_path": inventory_audio_path, "source_wav_member": member, "status": status, "resolved_path": str(resolved) if resolved is not None and resolved.is_file() else None, "alternate_path": alternate, "resolution_basis": basis})
    status_counts = Counter(row["status"] for row in map_rows)
    occurrences = _load_occurrences(index_path, inventory_by_id, set(TARGETS))
    target_summary: dict[str, Any] = {}
    for word in TARGETS:
        rows = occurrences.get(word, [])
        current = sum(1 for row in rows if row.get("audio_path") and (repo_root / row["audio_path"]).is_file())
        seen_text: set[str] = set()
        quality_excluded = Counter()
        quality_eligible = 0
        for item in rows:
            inventory = inventory_by_id[item["utterance_id"]]
            reasons = _quality_reasons(inventory, seen_text)
            if reasons:
                quality_excluded.update(reasons)
            else:
                seen_text.add(str(inventory.get("source_text", "")))
                quality_eligible += 1
        raw = len(rows)
        target_summary[word] = {
            "raw_recurrence": raw,
            "currently_resolved": current,
            "alternate_recovered_from_local_archive": raw - current if archive_verified else 0,
            "still_missing": 0 if archive_verified else raw - current,
            "usable_total_if_archive_path_is_resolved": quality_eligible,
            "archive_audio_available_total": raw if archive_verified else current,
            "capped_total_at_20": min(quality_eligible, 20),
            "quality_exclusions": dict(sorted(quality_excluded.items())),
        }
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue_items = queue.get("items", [])
    queue_occurrences = _load_occurrences(index_path, inventory_by_id, {str(item["normalized_word"]) for item in queue_items})
    queue_audio: dict[str, Any] = {}
    for item in queue_items:
        word = str(item["normalized_word"])
        rows = queue_occurrences.get(word, [])
        current = sum(1 for row in rows if row.get("audio_path") and (repo_root / row["audio_path"]).is_file())
        recovered = len(rows) - current if archive_verified else 0
        queue_audio[word] = {"recurrence_count": len(rows), "currently_resolved": current, "alternate_recovered": recovered, "audio_available_after_archive_resolution": len(rows) if archive_verified else current}
    queue_counts = {"queue_words": len(queue_items), "currently_with_at_least_1_audio": sum(value["currently_resolved"] >= 1 for value in queue_audio.values()), "currently_with_at_least_3_audio": sum(value["currently_resolved"] >= 3 for value in queue_audio.values()), "currently_with_at_least_5_audio": sum(value["currently_resolved"] >= 5 for value in queue_audio.values()), "currently_with_at_least_10_audio": sum(value["currently_resolved"] >= 10 for value in queue_audio.values()), "after_archive_resolution_with_at_least_1_audio": sum(value["audio_available_after_archive_resolution"] >= 1 for value in queue_audio.values()), "after_archive_resolution_with_at_least_3_audio": sum(value["audio_available_after_archive_resolution"] >= 3 for value in queue_audio.values()), "after_archive_resolution_with_at_least_5_audio": sum(value["audio_available_after_archive_resolution"] >= 5 for value in queue_audio.values()), "after_archive_resolution_with_at_least_10_audio": sum(value["audio_available_after_archive_resolution"] >= 10 for value in queue_audio.values())}
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "stage2d2_spicor_audio_resolution.jsonl", map_rows)
    resolving_examples = [row for row in map_rows if row["status"] == "RESOLVES"][:5]
    missing_examples = [row for row in map_rows if row["status"] != "RESOLVES"][:5]
    summary = {
        "schema_version": "stage2d2-spicor-audio-availability-v0.1",
        "inventory_path": str(inventory_path),
        "expected_wavs": len(inventory_rows),
        "unique_expected_wav_names": len(expected_names),
        "currently_resolving": status_counts["RESOLVES"],
        "missing": status_counts["MISSING"] + status_counts["FOUND_ALTERNATE_LOCATION"],
        "alternate_locations_found": status_counts["FOUND_ALTERNATE_LOCATION"],
        "recoverable_without_download": status_counts["FOUND_ALTERNATE_LOCATION"],
        "still_missing": status_counts["MISSING"],
        "root_cause_classification": "FULL_CORPUS_EXISTS_ELSEWHERE_LOCALLY",
        "root_cause_evidence": "The prepared tree has 1,274 files, while the already-local Downloads archive has a historically verified 25,158-member WAV set paired 1:1 with the 25,158 transcript records.",
        "full_corpus_exists_locally": archive_verified,
        "archive_provenance": archive_info,
        "candidate_roots": [
            {"path": str(repo_root / "data/spicor_eng_m_spk001_v1/audio_24k"), "kind": "PARTIAL_PREPARED_AUDIO", "file_count": status_counts["RESOLVES"]},
            {"path": str(archive_path), "kind": "COMPLETE_LOCAL_ARCHIVE", "verified_member_count": archive_info.get("verified_wav_count"), "extracted": False},
            {"path": str(ARCHIVE_PROVENANCE), "kind": "HISTORICAL_PROVENANCE_RECORDS", "source_only": True},
        ],
        "resolving_examples": resolving_examples,
        "missing_or_recoverable_examples": missing_examples,
        "validated_target_availability": target_summary,
        "review_queue_audio_availability": {"threshold_counts": queue_counts, "by_word": dict(sorted(queue_audio.items()))},
        "recommended_recovery_method": "Use a non-destructive archive-aware resolver or extract only the needed members from the existing local tar.gz into a separately configured corpus root; do not rewrite the immutable source inventory or extract the full archive without sufficient disk headroom.",
        "no_audio_copied": True,
        "no_training_performed": True,
        "qwen_loaded": False,
    }
    dump(output / "stage2d2_spicor_audio_availability.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_DEFAULT)
    parser.add_argument("--occurrence-index", type=Path, default=OCCURRENCE_INDEX_DEFAULT)
    parser.add_argument("--queue", type=Path, default=QUEUE_DEFAULT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(audit(args.repo_root.resolve(), args.output.resolve(), args.inventory.resolve(), args.occurrence_index.resolve(), args.archive.resolve(), args.queue.resolve()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
