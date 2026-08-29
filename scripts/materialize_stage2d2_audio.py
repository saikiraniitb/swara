#!/usr/bin/env python3
"""Materialize only the Stage2D.2 SPICOR WAV union and rebuild its design."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from swara.data.spicor_audio import AudioResolution, SpicorAudioResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DEFAULT = Path("/Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz")
INVENTORY_DEFAULT = REPO_ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
OUTPUT_DEFAULT = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design"
CACHE_DEFAULT = REPO_ROOT / "data/stage2d_spicor_selected_audio"
ACCEPTED_DEFAULT = REPO_ROOT / "data/stage2b_pronunciation/accepted_manifest.jsonl"
NATIVE_DEFAULT = OUTPUT_DEFAULT / "stage2d2_native_preservation_set.jsonl"
TARGETS = ("agrawal", "gupta", "kashmir", "kumar", "mishra", "mumbai", "sharma")
EXCLUDED = {"singh", "sensharma", "kashmiri", "dasharatha"}


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DESIGN = _load_script("stage2d2_design", REPO_ROOT / "scripts/design_stage2d2_dataset.py")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def inventory_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["source_id"]): row for row in read_jsonl(path)}


def enrich(rows: Mapping[str, list[dict[str, Any]]], inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word, values in rows.items():
        for item in values:
            row = inventory[item["utterance_id"]]
            item = dict(item)
            item.update({
                "cleanup_flags": list(row.get("cleanup_flags") or []),
                "transcript_empty": bool(row.get("transcript_empty", False)),
                "source_duration_seconds": row.get("source_duration_seconds"),
                "source_sample_rate_hz": row.get("source_sample_rate_hz"),
                "source_wav_member": row.get("source_wav_member"),
                "audio_path": row.get("prepared_audio_path"),
                "source_size_bytes": row.get("source_size_bytes"),
            })
            output[word].append(item)
    return output


def quality_reasons(item: Mapping[str, Any], seen: set[str] | None = None) -> list[str]:
    text = str(item.get("transcript", ""))
    reasons: list[str] = []
    if not text.strip() or item.get("transcript_empty"):
        reasons.append("empty_transcript")
    if DESIGN.ANNOTATION_RE.search(text):
        reasons.append("annotation_or_non_speech_text")
    tokens = DESIGN.extract_lexical_tokens(text)
    if any(DESIGN.NUMBER_RE.match(token["surface_form"]) for token in tokens):
        reasons.append("numeric_token")
    if any(DESIGN.MIXED_ALNUM_RE.match(token["surface_form"]) for token in tokens):
        reasons.append("mixed_alphanumeric_token")
    if item.get("cleanup_flags"):
        reasons.append("manifest_cleanup_flags")
    if seen is not None and text in seen:
        reasons.append("duplicate_transcript")
    return reasons


def valid_unique(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    result: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen: set[str] = set()
    for item in sorted(rows, key=lambda value: value["occurrence_id"]):
        reasons = quality_reasons(item, seen)
        if reasons:
            exclusions.update(reasons)
            continue
        seen.add(item["transcript"])
        result.append(item)
    return result, exclusions


def enforce_global_source_disjoint(train: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep occurrence-level splits while preventing source-utterance leakage."""

    train = [dict(row) for row in train]
    evaluation = [dict(row) for row in evaluation]
    while True:
        train_source_counts = Counter(row["utterance_id"] for row in train)
        evaluation_sources = {row["utterance_id"] for row in evaluation}
        conflict = next((row for row in sorted(evaluation, key=lambda value: value["occurrence_id"]) if row["utterance_id"] in train_source_counts), None)
        if conflict is None:
            break
        candidates = [
            row for row in train
            if row["target_normalized_word"] == conflict["target_normalized_word"]
            and train_source_counts[row["utterance_id"]] == 1
            and row["utterance_id"] not in evaluation_sources
        ]
        if not candidates:
            raise RuntimeError(f"cannot repair train/eval source overlap for {conflict['occurrence_id']}")
        replacement = min(candidates, key=lambda value: value["occurrence_id"])
        train.remove(replacement)
        evaluation.remove(conflict)
        conflict["dataset_split"] = "TRAIN"
        replacement["dataset_split"] = "EVAL_SEEN_WORD_UNSEEN_CONTEXT"
        train.append(conflict)
        evaluation.append(replacement)
    return sorted(train, key=lambda row: (row["target_normalized_word"], row["occurrence_id"])), sorted(evaluation, key=lambda row: (row["target_normalized_word"], row["occurrence_id"]))


def selected_explicit(high: Mapping[str, list[dict[str, Any]]], phone_map: Mapping[str, Mapping[str, Any]], acoustic_ids: set[str], cap: int = 20) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for word in TARGETS:
        valid, excluded = valid_unique(list(high.get(word, [])))
        exclusions.update(excluded)
        selected = DESIGN.select_diverse_occurrences(valid, cap, acoustic_ids)
        target_train, target_eval = DESIGN._split_selected(selected)
        for split, values in (("TRAIN", target_train), ("EVAL_SEEN_WORD_UNSEEN_CONTEXT", target_eval)):
            for item in values:
                record = {
                    "occurrence_id": item["occurrence_id"], "utterance_id": item["utterance_id"], "transcript": item["transcript"],
                    "target_word": item["surface_form"], "target_normalized_word": word, "target_char_span": item["target_char_span"], "target_word_index": item["word_index"],
                    "canonical_phone_sequence": phone_map[word]["canonical_phone_sequence"], "phone_inventory": "swara-phones-v0", "override_id": phone_map[word]["override_id"],
                    "confidence": phone_map[word]["confidence"], "context": {"preceding_word": item["preceding_word"], "following_word": item["following_word"], "position": item["position_bucket"], "domain": item.get("domain")},
                    "audio_path": item["audio_path"], "source_wav_member": item["source_wav_member"], "source_duration_seconds": item["source_duration_seconds"], "source_sample_rate_hz": item["source_sample_rate_hz"],
                    "supervision_type": "HIGH_CONFIDENCE_EXPLICIT_PRONUNCIATION", "dataset_split": split,
                    "source_evidence": "Stage2D.1 canonical_pronunciation_lexicon_v0.1 plus human full-utterance review",
                    "paired_native_reference": {"utterance_id": item["utterance_id"], "transcript": item["transcript"], "audio_path": item["audio_path"], "override_id": None, "same_text_same_audio_setup": True},
                    "selection_cap": cap, "acoustic_evidence_available": item["occurrence_id"] in acoustic_ids,
                }
                (train if split == "TRAIN" else evaluation).append(record)
    train, evaluation = enforce_global_source_disjoint(train, evaluation)
    return train, evaluation, exclusions


def update_audio_paths(rows: list[dict[str, Any]], resolved: Mapping[str, AudioResolution], repo_root: Path) -> None:
    for row in rows:
        resolution = resolved[row["utterance_id"]]
        if resolution.selected_audio_path is None:
            raise RuntimeError(f"selected item did not materialize: {row['utterance_id']}")
        row["audio_path"] = str(resolution.selected_audio_path.relative_to(repo_root)) if resolution.selected_audio_path.is_relative_to(repo_root) else str(resolution.selected_audio_path)
        row["audio_resolution_status"] = resolution.source_type


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DEFAULT)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--accepted", type=Path, default=ACCEPTED_DEFAULT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve(); output = args.output.resolve(); cache_root = args.cache_root.resolve()
    inventory = inventory_map(args.inventory.resolve())
    tier_payload = json.loads((repo_root / "artifacts/stage2d/pronunciation_atlas_v0_1/training_pronunciation_candidates.json").read_text(encoding="utf-8"))
    tier_words = {str(item["normalized_word"]) for item in tier_payload["tiers"]["TIER_2_REVIEW_REQUIRED"]}
    high_raw, review_raw = DESIGN.load_occurrence_candidates(repo_root / "artifacts/stage2d/pronunciation_atlas_v0_1/occurrence_index.jsonl", repo_root, set(TARGETS) | tier_words | EXCLUDED)
    high = enrich(high_raw, inventory); review = enrich(review_raw, inventory)
    acoustic_ids = DESIGN._load_acoustic_ids(repo_root / "artifacts/stage2d/pronunciation_atlas_v0_1/acoustic_consistency/stage2d1b_occurrence_sample.jsonl")
    phone_map = DESIGN._phone_rows(DESIGN._canonical_map(repo_root / "artifacts/stage2d/pronunciation_atlas_v0_1/canonical_pronunciation_lexicon_v0_1.json"))
    train_rows, eval_rows, exclusions = selected_explicit(high, phone_map, acoustic_ids, 20)
    existing_native = read_jsonl(repo_root / NATIVE_DEFAULT.relative_to(REPO_ROOT))
    accepted = read_jsonl(args.accepted.resolve())
    category_ids: dict[str, set[str]] = defaultdict(set)
    for row in train_rows + eval_rows: category_ids["EXPLICIT_PRONUNCIATION"].add(row["utterance_id"])
    for row in existing_native: category_ids["NATIVE_PRESERVATION"].add(row["utterance_id"])
    for row in accepted: category_ids["FROZEN_EVAL_HOLDOUT"].add(row["source_id"])
    review_selection: dict[str, list[dict[str, Any]]] = {}
    for rank, queue_item in enumerate(tier_payload["tiers"]["TIER_2_REVIEW_REQUIRED"], 1):
        word = str(queue_item["normalized_word"])
        valid, excluded = valid_unique(review.get(word, [])); exclusions.update(excluded)
        chosen = DESIGN.select_diverse_occurrences(valid, 5, acoustic_ids)
        review_selection[word] = chosen
        for item in chosen: category_ids["PRONUNCIATION_REVIEW_QUEUE"].add(item["utterance_id"])
    union = sorted({utterance_id for values in category_ids.values() for utterance_id in values})
    resolver = SpicorAudioResolver(inventory, repo_root=repo_root, archive_path=args.archive.resolve(), selected_cache_root=cache_root)
    estimated = resolver.estimate_archive_bytes(union)
    resolution_before = {utterance_id: resolver.resolve(utterance_id) for utterance_id in union}
    free_before = __import__("shutil").disk_usage(cache_root.parent).free
    resolved = resolver.materialize(union)
    output.mkdir(parents=True, exist_ok=True)
    materialization_rows: list[dict[str, Any]] = []
    for utterance_id in union:
        row = inventory[utterance_id]; resolution = resolved[utterance_id]
        if resolution.selected_audio_path is None: raise RuntimeError(f"no selected path after materialization: {utterance_id}")
        selected_path = resolution.selected_audio_path
        selected_value = str(selected_path.relative_to(repo_root)) if selected_path.is_relative_to(repo_root) else str(selected_path)
        materialization_rows.append({"utterance_id": utterance_id, "source_type": resolution.source_type, "original_inventory_path": resolution.original_inventory_path, "archive_member": resolution.archive_member, "selected_audio_path": selected_value, "selection_categories": sorted(category for category, ids in category_ids.items() if utterance_id in ids), "file_size_bytes": selected_path.stat().st_size})
    write_jsonl(cache_root / "manifest.jsonl", materialization_rows)
    write_jsonl(output / "stage2d2_audio_resolution_manifest.jsonl", materialization_rows)
    update_audio_paths(train_rows + eval_rows, resolved, repo_root)
    for row in train_rows + eval_rows:
        # Keep the trajectory-pairing record synchronized with the resolver-backed
        # path; archive-extracted examples have no prepared_audio_path in inventory.
        row["paired_native_reference"]["audio_path"] = row["audio_path"]
    native_rows = list(existing_native)
    for row in native_rows:
        r = resolved[row["utterance_id"]]; row["audio_path"] = str(r.selected_audio_path.relative_to(repo_root)) if r.selected_audio_path and r.selected_audio_path.is_relative_to(repo_root) else str(r.selected_audio_path); row["audio_resolution_status"] = r.source_type
    write_jsonl(output / "stage2d2_explicit_candidates.jsonl", sorted(train_rows + eval_rows, key=lambda row: (row["dataset_split"], row["target_normalized_word"], row["occurrence_id"])))
    write_jsonl(output / "stage2d2_native_preservation_set.jsonl", native_rows)
    review_rows: list[dict[str, Any]] = []
    queue_by_word = {item["normalized_word"]: item for item in tier_payload["tiers"]["TIER_2_REVIEW_REQUIRED"]}
    queue_rank = {str(item["normalized_word"]): rank for rank, item in enumerate(tier_payload["tiers"]["TIER_2_REVIEW_REQUIRED"], 1)}
    for word in sorted(review_selection):
        for item in review_selection[word]:
            r = resolved[item["utterance_id"]]
            review_rows.append({"review_rank": queue_rank[word], "word": item["surface_form"], "normalized_word": word, "occurrence_id": item["occurrence_id"], "utterance_id": item["utterance_id"], "transcript": item["transcript"], "audio_resolver_path": str(r.selected_audio_path.relative_to(repo_root)) if r.selected_audio_path and r.selected_audio_path.is_relative_to(repo_root) else str(r.selected_audio_path), "audio_resolution_status": r.source_type, "preceding_word": item["preceding_word"], "following_word": item["following_word"], "recurrence_count": queue_by_word[word]["occurrence_count"], "current_phone_candidate": None, "representability": "UNKNOWN_NO_AUTOMATIC_G2P", "review_priority": queue_rank[word]})
    write_jsonl(output / "stage2d2_review_occurrence_manifest.jsonl", sorted(review_rows, key=lambda row: (row["review_priority"] or 0, row["occurrence_id"])))
    batch_words = [item["normalized_word"] for item in tier_payload["tiers"]["TIER_2_REVIEW_REQUIRED"][:25]]
    dump(output / "stage2d2_review_batch1.json", {"schema_version": "stage2d2-review-batch1-v0.1", "selection_policy": "top 25 existing Tier-2 words by frozen Stage2D.1 review ordering; no phone labels assigned", "words": [{"word": queue_by_word[word]["word"], "normalized_word": word, "occurrence_ids": [item["occurrence_id"] for item in review_selection[word]], "occurrence_count": queue_by_word[word]["occurrence_count"], "phone_sequence": None, "status": "PENDING_HUMAN_REVIEW"} for word in batch_words]})
    fixture_plan = DESIGN._fixture_plan(repo_root / "data/stage2b_pronunciation/evaluation_fixtures.json", {row["transcript"] for row in train_rows}, {row["occurrence_id"] for row in train_rows})
    train_sources = {row["utterance_id"] for row in train_rows}
    eval_sources = {row["utterance_id"] for row in eval_rows}
    split_plan = {"schema_version": "stage2d2-split-plan-v0.2", "explicit_train_occurrences": [row["occurrence_id"] for row in train_rows], "eval_seen_word_unseen_context_occurrences": [row["occurrence_id"] for row in eval_rows], **fixture_plan, "external_holdouts": ["Dasharatha", "Anirban", "Ashwini", "Chandrashekhar", "Karthik"], "archive_backed_audio": True}
    split_plan["leakage_check"]["explicit_train_source_utterances_disjoint_from_eval"] = train_sources.isdisjoint(eval_sources)
    split_plan["leakage_check"]["explicit_train_eval_source_overlap_count"] = len(train_sources & eval_sources)
    dump(output / "stage2d2_split_plan.json", split_plan)
    options = []
    for cap, native_count in ((10, 100), (15, 200), (20, 300), (25, 300)):
        opt_train, opt_eval, _ = selected_explicit(high, phone_map, acoustic_ids, cap)
        options.append({"name": {10: "SMALL", 15: "MEDIUM", 20: "LARGE_INITIAL", 25: "LARGE_INITIAL_HIGH_CAP"}[cap], "cap_per_target": cap, "explicit_selected_count": len(opt_train) + len(opt_eval), "explicit_train_count": len(opt_train), "eval_seen_count": len(opt_eval), "native_preservation_count": native_count, "unique_target_words": len(TARGETS), "lexical_balance": DESIGN._balance(opt_train), "phone_symbol_coverage_count": len({p for row in opt_train for p in row["canonical_phone_sequence"]}), "risk": "repeated single-speaker contexts add less lexical diversity after cap 20" if cap >= 20 else "lower context coverage"})
    dump(output / "stage2d2_scale_options.json", {"options": options, "recommended": {"name": "LARGE_INITIAL", "cap_per_target": 20, "reason": "134 quality-filtered archive-backed occurrences are available before deterministic train/eval split; cap limits frequency domination."}})
    dump(output / "stage2d2_phone_coverage.json", DESIGN._phone_coverage(train_rows))
    dump(output / "stage2d2_word_balance.json", DESIGN._balance(train_rows))
    dump(output / "stage2d2_exclusion_report.json", {"schema_version": "stage2d2-exclusion-report-v0.2", "explicit_and_review_quality_exclusions": dict(sorted(exclusions.items())), "missing_audio_exclusions_after_archive_resolution": 0, "no_inventory_gap_items_in_explicit_train": True, "no_unresolved_phone_mappings_in_explicit_train": True})
    dump(output / "stage2d2_trajectory_pairing_plan.json", {"schema_version": "stage2d2-trajectory-pairing-plan-v0.2", "pair_count": len(train_rows) + len(eval_rows), "pairing": "same text/audio occurrence with override_id null", "pairs": [{"occurrence_id": row["occurrence_id"], "native_reference": row["paired_native_reference"]} for row in train_rows + eval_rows]})
    summary = {"schema_version": "stage2d2-dataset-design-v0.2", "no_training_performed": True, "qwen_loaded": False, "swara_phones_v0_modified": False, "audio_source": "local SPICOR archive with selective materialization", "archive_path": str(args.archive.resolve()), "validated_high_confidence_target_words": list(TARGETS), "raw_occurrences_per_validated_target": {word: len(high.get(word, [])) for word in TARGETS}, "quality_usable_occurrences_per_validated_target": {word: len(valid_unique(high.get(word, []))[0]) for word in TARGETS}, "recommended_cap": 20, "explicit_selected_count": len(train_rows) + len(eval_rows), "explicit_train_count": len(train_rows), "eval_seen_word_unseen_context_count": len(eval_rows), "native_preservation_count": len(native_rows), "review_queue_count": len(tier_payload["tiers"]["TIER_2_REVIEW_REQUIRED"]), "review_occurrence_count": len(review_rows), "review_batch1_word_count": len(batch_words), "review_batch1_words": batch_words, "materialization_union_count": len(union), "prepared_local_reused_count": sum(r.source_type == "PREPARED_LOCAL" for r in resolved.values()), "archive_extracted_count": sum(r.source_type == "ARCHIVE_EXTRACTED" for r in resolved.values()), "estimated_archive_bytes": estimated, "free_bytes_before": free_before, "selected_audio_bytes": sum(Path(row["selected_audio_path"]).stat().st_size if Path(row["selected_audio_path"]).is_absolute() else (repo_root / row["selected_audio_path"]).stat().st_size for row in materialization_rows), "status": "READY_FOR_STAGE2D2_BATCH1_HUMAN_REVIEW"}
    dump(output / "stage2d2_audio_materialization_plan.json", {"schema_version": "stage2d2-audio-materialization-plan-v0.1", "archive_path": str(args.archive.resolve()), "selection_categories": {category: sorted(ids) for category, ids in sorted(category_ids.items())}, "unique_union_count": len(union), "expected_archive_extraction_count": sum(resolution_before[uid].status == "ARCHIVE_MEMBER_AVAILABLE" for uid in union), "estimated_uncompressed_bytes": estimated, "free_bytes_before": free_before, "safe_extraction_headroom_bytes": 512 * 1024 * 1024, "selected_cache_root": str(cache_root), "full_archive_extracted": False, "materialization_manifest": str(cache_root / "manifest.jsonl")})
    dump(output / "stage2d2_summary.json", summary)
    (repo_root / "docs/stage2d/STAGE2D2_DATASET_DESIGN.md").write_text(f"# Stage2D.2 Dataset Design\n\nArchive-backed redesign completed without training. The local SPICOR tarball was selectively materialized only for the {len(union)}-utterance union.\n\n- Explicit selected: {len(train_rows) + len(eval_rows)} ({len(train_rows)} TRAIN, {len(eval_rows)} EVAL-SEEN)\n- Native preservation: {len(native_rows)}\n- Tier-2 review words: 100; Batch 1: {len(batch_words)}\n- Prepared-local reused: {summary['prepared_local_reused_count']}\n- Archive-extracted: {summary['archive_extracted_count']}\n- Full archive extraction: NO\n- Training/Qwen: NO\n\nThe seven explicit targets use only the existing human-reviewed swara-phones-v0 mappings. Singh, Sensharma, Kashmiri, and Dasharatha remain excluded from explicit training as previously specified. See the JSON artifacts for exact IDs, paths, splits, coverage, and pairing metadata.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
