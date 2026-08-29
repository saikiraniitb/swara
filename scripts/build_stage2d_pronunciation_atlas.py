#!/usr/bin/env python3
"""Build the deterministic Stage2D.1 pronunciation atlas.

The scanner reads transcript metadata and existing curated phone-review JSON.
It never loads audio, Qwen, an acoustic model, or a training checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from swara.diagnostics.pronunciation_atlas import (
    ATLAS_SCHEMA_VERSION,
    AtlasOccurrence,
    build_consistency_report,
    build_data_quality_report,
    build_holdout_plan,
    build_training_candidates,
    build_vocabulary,
    json_dump,
    load_curated_phone_review,
    load_jsonl,
    make_extension_proposals,
    recurrence_buckets,
    scan_manifest,
)


DEFAULT_MANIFEST = "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
DEFAULT_CURATED = "data/stage2b_pronunciation/lexical_phone_review.json"
DEFAULT_FIXTURES = "data/stage2b_pronunciation/evaluation_fixtures.json"
DEFAULT_OUTPUT = "artifacts/stage2d/pronunciation_atlas_v0_1"


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _occurrences_for_word(occurrences: list[AtlasOccurrence], word: str) -> list[AtlasOccurrence]:
    return [item for item in occurrences if item.normalized_word == word]


def _write_grouped_occurrence_index(path: Path, occurrences: list[AtlasOccurrence]) -> None:
    """Write each full transcript once, with all lexical occurrences nested."""

    grouped: dict[str, list[AtlasOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence.utterance_id].append(occurrence)
    with path.open("w", encoding="utf-8") as handle:
        for utterance_id in sorted(grouped):
            items = sorted(grouped[utterance_id], key=lambda item: item.word_index)
            first = items[0]
            handle.write(
                json.dumps(
                    {
                        "utterance_id": utterance_id,
                        "full_transcript": first.full_transcript,
                        "audio_path": first.audio_path,
                        "source_wav_member": first.source_wav_member,
                        "split": first.split,
                        "domain": first.domain,
                        "lexical_occurrences": [
                            [
                                item.word_index,
                                item.surface_form,
                                item.normalized_word,
                                [item.source_span_start, item.source_span_end],
                                item.preceding_word,
                                item.following_word,
                                list(item.interest_signals),
                            ]
                            for item in items
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def build_outputs(manifest_path: str | Path, curated_path: str | Path, fixtures_path: str | Path | None, output_dir: str | Path) -> dict[str, Any]:
    occurrences = scan_manifest(manifest_path)
    manifest_rows = load_jsonl(manifest_path)
    curated = load_curated_phone_review(curated_path)
    fixtures = _load_optional_json(fixtures_path)
    vocabulary = build_vocabulary(occurrences, curated)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _write_grouped_occurrence_index(output / "occurrence_index.jsonl", occurrences)
    json_dump(
        output / "vocabulary.json",
        {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "source_manifest": str(manifest_path),
            "words": vocabulary,
        },
    )
    json_dump(
        output / "top_recurrent_words.json",
        {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "top_n": 30,
            "words": vocabulary[:30],
        },
    )
    json_dump(
        output / "consistency_report.json",
        {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "levels": {
                "A": "lexical consistency",
                "B": "canonical mapping consistency",
                "C": "curated pronunciation consistency",
                "D": "acoustic realization consistency; UNMEASURED in this stage",
            },
            "words": build_consistency_report(vocabulary),
        },
    )

    anchors: list[dict[str, Any]] = []
    for normalized, variants in sorted(curated.items()):
        matched = _occurrences_for_word(occurrences, normalized)
        anchors.append(
            {
                "normalized_word": normalized,
                "display_words": sorted({item.surface_form for item in matched}) or sorted({item["target_text"] for item in variants}),
                "corpus_occurrence_count": len(matched),
                "corpus_utterance_count": len({item.utterance_id for item in matched}),
                "surface_forms": dict(sorted(__import__("collections").Counter(item.surface_form for item in matched).items())),
                "occurrence_sample": [item.to_dict() for item in matched[:20]],
                "curated_variants": variants,
                "acoustic_realization_consistency": "UNMEASURED",
                "anchor_usefulness": "high" if matched else "external_or_absent_corpus_anchor",
            }
        )
    json_dump(
        output / "curated_anchor_analysis.json",
        {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "source": str(curated_path),
            "anchors": anchors,
            "note": "Curated records are human evidence; absence of acoustic realization inference is intentional.",
        },
    )

    dasharatha_count = len(_occurrences_for_word(occurrences, "dasharatha"))
    extensions = make_extension_proposals(vocabulary, curated, dasharatha_count)
    json_dump(
        output / "candidate_phone_extensions.json",
        {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "production_inventory_modified": False,
            "current_inventory_id": "swara-phones-v0",
            "proposals": extensions,
        },
    )
    json_dump(
        output / "swara_phones_v1_proposal.json",
        {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "status": "PROPOSAL_ONLY",
            "production_inventory_modified": False,
            "current_inventory_id": "swara-phones-v0",
            "recommended_extensions": [item for item in extensions if item["include_in_v1_recommendation"]],
            "other_candidates": [item for item in extensions if not item["include_in_v1_recommendation"]],
            "decision_rule": "No extension is production-approved without human/acoustic validation.",
        },
    )
    training_candidates = build_training_candidates(vocabulary, curated, fixtures)
    json_dump(output / "training_pronunciation_candidates.json", training_candidates)
    json_dump(output / "holdout_plan.json", build_holdout_plan(curated, fixtures))

    quality = build_data_quality_report(occurrences, manifest_path)
    quality["transcript_count"] = len(manifest_rows)
    quality["empty_transcript_count"] = sum(
        1 for row in manifest_rows if not str(row.get("source_text", row.get("training_text", ""))).strip()
    )
    json_dump(output / "data_quality_report.json", quality)

    thresholds = {threshold: sum(1 for row in vocabulary if row["occurrence_count"] >= threshold) for threshold in (2, 5, 10, 25, 50, 100)}
    interest_words = [row for row in vocabulary if row.get("pronunciation_interest")]
    verified_variants = [item for values in curated.values() for item in values if item.get("verification_status") == "VERIFIED"]
    valid_verified = [item for item in verified_variants if not item.get("invalid_symbols")]
    atlas_summary = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "atlas_version": "pronunciation_atlas_v0_1",
        "corpus_source": {
            "manifest": str(manifest_path),
            "selection_policy": "master_inventory.jsonl only; overlapping train/val/test manifests are not rescanned",
        },
        "occurrence_index_record_format": {
            "record": "one per utterance; full_transcript appears once",
            "lexical_occurrence_tuple": ["word_index", "surface_form", "normalized_word", "[span_start, span_end]", "preceding_word", "following_word", "interest_signals"],
            "occurrence_id": "derived as utterance_id + ':word:' + zero-padded word_index",
        },
        "transcript_count": len(manifest_rows),
        "total_lexical_token_count": len(occurrences),
        "unique_normalized_word_count": len(vocabulary),
        "recurring_word_counts": thresholds,
        "recurrence_buckets": recurrence_buckets(vocabulary),
        "pronunciation_interest_unique_word_count": len(interest_words),
        "dasharatha": {"occurrence_count": dasharatha_count, "status": "IN_SPICOR" if dasharatha_count else "EXTERNAL_UNSEEN_PROBE"},
        "swara_phones_v0": {
            "inventory_id": "swara-phones-v0",
            "curated_variant_count": len([item for values in curated.values() for item in values]),
            "curated_representable_variant_count": len(valid_verified),
            "curated_representable_variant_rate": len(valid_verified) / max(1, len(verified_variants) + sum(1 for values in curated.values() for item in values if item.get("verification_status") == "UNSUPPORTED_ALPHABET_VARIANT")),
            "automatic_g2p_available": False,
            "ordinary_vocabulary_automatic_coverage_rate": 0.0,
        },
        "evidence_policy": {
            "canonical_phone_candidate": "existing curated human review only",
            "occurrence_realization_evidence": "metadata only; acoustic phone realization UNMEASURED",
            "training_performed": False,
            "qwen_loaded": False,
            "audio_loaded": False,
            "production_phone_inventory_modified": False,
        },
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    json_dump(output / "atlas_summary.json", atlas_summary)
    return atlas_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--curated", default=DEFAULT_CURATED)
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_outputs(args.manifest, args.curated, args.fixtures, args.output_dir)
    print(json.dumps({key: summary[key] for key in ("transcript_count", "total_lexical_token_count", "unique_normalized_word_count", "recurring_word_counts", "dasharatha")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
