"""Small, fail-closed validation helpers for the Stage2B.4B data manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_ID, PRONUNCIATION_ALPHABET_V0


class Stage2BManifestError(ValueError):
    """Raised when a pronunciation candidate or split is unsafe to train."""


STATUSES = frozenset(
    {
        "PENDING_HUMAN_REVIEW",
        "HUMAN_AUDIO_VERIFIED_PHONE_PENDING",
        "UNSUPPORTED_ALPHABET_VARIANT",
        "VERIFIED",
        "REJECTED",
        "UNSUPPORTED_ALPHABET",
    }
)
REQUIRED_FIELDS = frozenset(
    {
        "candidate_id", "audio_path", "transcript", "target_text", "source_span_start", "source_span_end",
        "language", "audio_start_seconds", "audio_end_seconds", "alignment_confidence", "alignment_model",
        "alignment_revision", "codec_frame_start", "codec_frame_end", "codec_total_frames",
        "pronunciation_system", "proposed_phone_sequence", "phone_sequence_source", "verification_status",
        "verification_note",
    }
)


def _fail(message: str) -> None:
    raise Stage2BManifestError(message)


def validate_candidate_record(record: Mapping[str, object], *, repository_root: str | Path = ".") -> None:
    """Validate one candidate without requiring human verification."""

    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        _fail(f"candidate is missing fields: {sorted(missing)}")
    candidate_id = record["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id:
        _fail("candidate_id must be a non-empty string")
    transcript = record["transcript"]
    target_text = record["target_text"]
    if not isinstance(transcript, str) or not isinstance(target_text, str) or not target_text:
        _fail("transcript and target_text must be non-empty strings")
    start, end = record["source_span_start"], record["source_span_end"]
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(transcript):
        _fail("source span is not a valid Unicode code-point half-open range")
    if transcript[start:end] != target_text:
        _fail("target_text does not match transcript[source_span_start:source_span_end]")

    audio_path = record["audio_path"]
    if not isinstance(audio_path, str) or not (Path(repository_root) / audio_path).is_file():
        _fail(f"candidate audio does not exist: {audio_path}")
    numeric = ("audio_start_seconds", "audio_end_seconds", "alignment_confidence")
    values = [record[key] for key in numeric]
    if any(not isinstance(value, (int, float)) for value in values):
        _fail("alignment seconds and confidence must be numeric")
    audio_start, audio_end, confidence = (float(value) for value in values)
    if audio_start < 0 or audio_end <= audio_start or not 0.0 <= confidence <= 1.0:
        _fail("alignment interval/confidence is invalid")
    frame_start, frame_end, frame_total = (
        record["codec_frame_start"], record["codec_frame_end"], record["codec_total_frames"]
    )
    if not all(isinstance(value, int) for value in (frame_start, frame_end, frame_total)):
        _fail("codec frame geometry must be integer")
    if frame_start < 0 or frame_end <= frame_start or frame_end > frame_total or frame_total <= 0:
        _fail("codec frame range is invalid")
    if record["pronunciation_system"] != PRONUNCIATION_ALPHABET_ID:
        _fail("candidate pronunciation system is not swara-phones-v0")
    phones = record["proposed_phone_sequence"]
    if phones is not None:
        if not isinstance(phones, (list, tuple)) or not phones or any(phone not in PRONUNCIATION_ALPHABET_V0 for phone in phones):
            _fail("proposed phone sequence contains unsupported or empty symbols")
    status = record["verification_status"]
    if status not in STATUSES:
        _fail(f"unknown verification status: {status}")
    if status == "VERIFIED" and (not phones or record.get("override_id") in (None, "")):
        _fail("VERIFIED candidates require an override_id and verified phone sequence")


def validate_candidate_manifest(records: Sequence[Mapping[str, object]], *, repository_root: str | Path = ".") -> None:
    seen: set[str] = set()
    for record in records:
        validate_candidate_record(record, repository_root=repository_root)
        candidate_id = str(record["candidate_id"])
        if candidate_id in seen:
            _fail(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)


def validate_accepted_manifest(records: Sequence[Mapping[str, object]], *, repository_root: str | Path = ".") -> None:
    validate_candidate_manifest(records, repository_root=repository_root)
    for record in records:
        if record["verification_status"] != "VERIFIED":
            _fail("accepted_manifest cannot contain non-VERIFIED records")


def validate_disjoint_splits(train: Iterable[Mapping[str, object]], evaluation: Iterable[Mapping[str, object]]) -> None:
    train_keys = {(record.get("source_id"), record.get("source_span_start"), record.get("source_span_end")) for record in train}
    overlap = train_keys & {
        (record.get("source_id"), record.get("source_span_start"), record.get("source_span_end"))
        for record in evaluation
    }
    if overlap:
        _fail(f"train/eval audio occurrence overlap: {sorted(overlap)}")


def validate_transfer_texts(training_transcripts: Iterable[str], transfer_texts: Iterable[str]) -> None:
    training = set(training_transcripts)
    overlap = training & set(transfer_texts)
    if overlap:
        _fail(f"transfer text duplicates training transcript: {sorted(overlap)}")


__all__ = [
    "PRONUNCIATION_ALPHABET_ID",
    "Stage2BManifestError",
    "validate_accepted_manifest",
    "validate_candidate_manifest",
    "validate_candidate_record",
    "validate_disjoint_splits",
    "validate_transfer_texts",
]
