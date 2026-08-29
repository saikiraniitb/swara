#!/usr/bin/env python3
"""Create the design-only Stage2D.4 bounded intervention dataset.

This script reads SPICOR transcript metadata and existing Stage2D evidence.  It
does not load audio, Qwen, checkpoints, or an optimizer, and it never writes
to the production pronunciation inventory.  Archive-backed paths are recorded
through :class:`SpicorAudioResolver`; no audio is materialized here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from swara.data.spicor_audio import SpicorAudioResolver
from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_ID, PRONUNCIATION_ALPHABET_V0


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_DEFAULT = REPO_ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
OUTPUT_DEFAULT = REPO_ROOT / "artifacts/stage2d/stage2d4_training_design"
CACHE_DEFAULT = REPO_ROOT / "data/stage2d_spicor_selected_audio"
ARCHIVE_DEFAULT = Path("/Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz")
POLICY_DEFAULT = REPO_ROOT / "artifacts/stage2d/stage2d3_reference_guided_phone_test/stage2d3_pronunciation_intervention_policy_v0_1.json"
NATIVE_DEFAULT = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design/stage2d2_native_preservation_set.jsonl"

POSITIVE_WORDS = ("Jamshedpur", "Chandigarh", "Nagpur")
TARGETED_NATIVE_WORDS = ("Nagar", "Banerjee")
POSITIVE_PHONES = {
    "Jamshedpur": ("J", "A", "M", "SH", "I", "D", "P", "U"),
    "Chandigarh": ("CH", "A", "N", "D", "I", "G", "AA"),
    "Nagpur": ("N", "A", "G", "P", "U", "R"),
}
GOLD_REFERENCES = {
    "Jamshedpur": "IISc_SPICORProject_EN_M_AGRI_3841",
    "Chandigarh": "IISc_SPICORProject_EN_M_WEAT_288",
    "Nagpur": "IISc_SPICORProject_EN_M_ENTE_3545",
}
EXTERNAL_HOLDOUTS = ("Dasharatha", "Anirban", "Ashwini", "Chandrashekhar", "Karthik")
PHONE_CONTRAST_FIXTURES = ("Singh", "Mumbai", "Kumar")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NUMBER_RE = re.compile(r"\d")
MIXED_ALNUM_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def lexical_occurrences(text: str, target: str | None = None) -> list[dict[str, Any]]:
    matches = list(TOKEN_RE.finditer(text))
    result: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        surface = match.group(0)
        if target is not None and surface.casefold() != target.casefold():
            continue
        result.append({
            "word_index": index,
            "surface_form": surface,
            "normalized_word": surface.casefold(),
            "target_char_span": {"start": match.start(), "end": match.end()},
            "preceding_word": matches[index - 1].group(0) if index else None,
            "following_word": matches[index + 1].group(0) if index + 1 < len(matches) else None,
            "token_count": len(matches),
        })
    return result


def make_resolver(rows: Mapping[str, Mapping[str, Any]], *, archive: Path, cache: Path) -> SpicorAudioResolver:
    resolver_rows: dict[str, dict[str, Any]] = {}
    for utterance_id, row in rows.items():
        prepared = row.get("prepared_audio_path")
        if not prepared:
            prepared = f"data/spicor_eng_m_spk001_v1/audio_24k/{utterance_id}.wav"
        resolver_rows[utterance_id] = {**row, "prepared_audio_path": prepared}
    return SpicorAudioResolver(resolver_rows, repo_root=REPO_ROOT, archive_path=archive, selected_cache_root=cache)


def load_inventory(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["source_id"]): row for row in read_jsonl(path)}


def target_rows(inventory: Mapping[str, Mapping[str, Any]], targets: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    wanted = {word.casefold(): word for word in targets}
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for utterance_id, source in inventory.items():
        transcript = str(source.get("source_text", ""))
        for occurrence in lexical_occurrences(transcript):
            canonical = wanted.get(occurrence["normalized_word"])
            if canonical is None:
                continue
            output[canonical].append({
                "word": canonical,
                "normalized_word": canonical.casefold(),
                "utterance_id": utterance_id,
                "transcript": transcript,
                "target_word_index": occurrence["word_index"],
                "target_char_span": occurrence["target_char_span"],
                "preceding_word": occurrence["preceding_word"],
                "following_word": occurrence["following_word"],
                "token_count": occurrence["token_count"],
                "position": "INITIAL" if occurrence["word_index"] == 0 else "FINAL" if occurrence["word_index"] == occurrence["token_count"] - 1 else "MEDIAL",
                "domain": source.get("domain"),
                "duration_seconds": source.get("source_duration_seconds"),
                "source_wav_member": source.get("source_wav_member"),
                "source_size_bytes": source.get("source_size_bytes"),
                "cleanup_flags": list(source.get("cleanup_flags") or []),
                "header_error": source.get("header_error"),
                "transcript_empty": bool(source.get("transcript_empty", not transcript.strip())),
                "duplicate_text_group": source.get("duplicate_text_group"),
            })
    return output


def quality_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    transcript = str(row.get("transcript", ""))
    if not transcript.strip() or row.get("transcript_empty"):
        reasons.append("empty_transcript")
    if row.get("header_error"):
        reasons.append("header_error")
    if row.get("cleanup_flags"):
        reasons.append("manifest_cleanup_flags")
    if row.get("duplicate_text_group"):
        reasons.append("duplicate_text_group")
    if NUMBER_RE.search(transcript):
        reasons.append("numeric_content")
    if MIXED_ALNUM_RE.search(transcript):
        reasons.append("mixed_alphanumeric_content")
    if not row.get("source_wav_member"):
        reasons.append("missing_archive_member")
    return reasons


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    # Stable coverage proxy.  This is metadata diversity, not an acoustic score.
    duration = row.get("duration_seconds")
    duration_bucket = "UNKNOWN" if duration is None else "SHORT" if float(duration) < 5 else "MEDIUM" if float(duration) < 9 else "LONG"
    return (
        str(row.get("position")),
        str(row.get("domain") or ""),
        str(row.get("preceding_word") or "").casefold(),
        str(row.get("following_word") or "").casefold(),
        duration_bucket,
        str(row["utterance_id"]),
        int(row["target_word_index"]),
    )


def select_unique(rows: Iterable[Mapping[str, Any]], *, limit: int, excluded_ids: set[str] | None = None) -> tuple[list[dict[str, Any]], Counter[str]]:
    excluded_ids = excluded_ids or set()
    selected: list[dict[str, Any]] = []
    reasons = Counter()
    seen_transcripts: set[str] = set()
    seen_utterances: set[str] = set()
    for original in sorted(rows, key=selection_key):
        row = dict(original)
        if row["utterance_id"] in excluded_ids:
            reasons["excluded_requested_reference"] += 1
            continue
        bad = quality_reasons(row)
        if bad:
            reasons.update(bad)
            continue
        if row["transcript"] in seen_transcripts:
            reasons["duplicate_transcript"] += 1
            continue
        # One occurrence per source utterance is the safe default for context transfer.
        if row["utterance_id"] in seen_utterances:
            reasons["duplicate_source_utterance"] += 1
            continue
        selected.append(row)
        seen_transcripts.add(row["transcript"])
        seen_utterances.add(row["utterance_id"])
        if len(selected) >= limit:
            break
    return selected, reasons


def resolver_record(row: Mapping[str, Any], resolver: SpicorAudioResolver) -> dict[str, Any]:
    resolution = resolver.resolve(str(row["utterance_id"]))
    if resolution.selected_audio_path is not None:
        audio_path = str(resolution.selected_audio_path.relative_to(REPO_ROOT)) if resolution.selected_audio_path.is_relative_to(REPO_ROOT) else str(resolution.selected_audio_path)
    elif resolution.archive_member:
        audio_path = f"spicor://archive/{Path(resolution.archive_member).name}"
    else:
        audio_path = None
    return {
        "audio_resolver_path": audio_path,
        "audio_resolution_status": resolution.status,
        "source_type": resolution.source_type,
        "original_inventory_path": resolution.original_inventory_path,
        "source_wav_member": resolution.archive_member,
    }


def split_positive(rows: list[dict[str, Any]], *, word: str, cap: int, resolver: SpicorAudioResolver) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    gold_id = GOLD_REFERENCES[word]
    gold_candidates = [row for row in rows if row["utterance_id"] == gold_id]
    if len(gold_candidates) != 1:
        raise ValueError(f"expected exactly one gold occurrence for {word}, found {len(gold_candidates)}")
    usable, exclusions = select_unique(rows, limit=max(0, cap - 1), excluded_ids={gold_id})
    # Keep the user-selected gold reference in a separate evaluation split.
    gold = dict(gold_candidates[0])
    gold["split"] = "HUMAN_GOLD_REFERENCE"
    gold["is_human_gold_reference"] = True
    gold["training_eligible"] = False
    gold["canonical_experimental_phone_sequence"] = list(POSITIVE_PHONES[word])
    gold["phone_inventory"] = PRONUNCIATION_ALPHABET_ID
    gold["evidence_status"] = "EXPLICIT_OVERRIDE_SUPPORTED"
    gold["human_reference_utterance_id"] = gold_id
    gold["intervention_required"] = True
    gold["supervision_type"] = "POSITIVE_INTERVENTION_HUMAN_GOLD_REFERENCE"
    gold["source_evidence"] = "Stage2D.3A/B blinded reference-guided human comparison; held out from training"
    gold.update(resolver_record(gold, resolver))
    eval_count = max(1, math.ceil(len(usable) * 0.25)) if usable else 0
    eval_rows = usable[-eval_count:] if eval_count else []
    train_rows = usable[:-eval_count] if eval_count else usable
    def decorate(row: Mapping[str, Any], split: str) -> dict[str, Any]:
        result = dict(row)
        result.update({
            "canonical_experimental_phone_sequence": list(POSITIVE_PHONES[word]),
            "phone_inventory": PRONUNCIATION_ALPHABET_ID,
            "evidence_status": "EXPLICIT_OVERRIDE_SUPPORTED",
            "human_reference_utterance_id": gold_id,
            "is_human_gold_reference": False,
            "training_eligible": split == "TRAIN",
            "split": split,
            "supervision_type": "POSITIVE_INTERVENTION",
            "intervention_required": True,
            "source_evidence": "Stage2D.3A/B blinded reference-guided human comparison",
            "selection_cap_including_gold": cap,
            "selection_basis": "deterministic metadata-diversity ordering; one source utterance and transcript at most",
            "context": {
                "preceding_word": row["preceding_word"],
                "following_word": row["following_word"],
                "position": row["position"],
                "domain": row["domain"],
            },
            "paired_native_reference": {
                "utterance_id": row["utterance_id"],
                "transcript": row["transcript"],
                "audio_resolver_path": resolver_record(row, resolver)["audio_resolver_path"],
                "override_id": None,
                "same_text_same_audio_setup": True,
            },
        })
        result.update(resolver_record(row, resolver))
        return result
    return [decorate(row, "TRAIN") for row in train_rows], [decorate(row, "EVAL_SEEN_WORD_UNSEEN_CONTEXT") for row in eval_rows], [gold], exclusions


def native_row(row: Mapping[str, Any], resolver: SpicorAudioResolver | None = None) -> dict[str, Any]:
    result = dict(row)
    result.update({
        "word": None,
        "normalized_word": None,
        "phone_sequence": None,
        "phone_inventory": None,
        "override_id": None,
        "intervention_required": False,
        "supervision_type": "NATIVE_PRESERVATION",
        "source_evidence": "Stage2D.2 native-preservation design; no pronunciation override",
        "training_eligible": True,
        "split": "TRAIN",
    })
    if resolver is not None and result.get("utterance_id"):
        result.update(resolver_record({"utterance_id": result["utterance_id"]}, resolver))
    return result


def build_targeted_native(targets: Mapping[str, list[dict[str, Any]]], resolver: SpicorAudioResolver, limit: int = 5) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    exclusions = Counter()
    for word in TARGETED_NATIVE_WORDS:
        selected, excluded = select_unique(targets.get(word, []), limit=limit)
        exclusions.update({f"{word}:{key}": value for key, value in excluded.items()})
        for row in selected:
            result = dict(row)
            result.update(resolver_record(row, resolver))
            result.update({
                "word": word,
                "normalized_word": word.casefold(),
                "phone_sequence": None,
                "phone_inventory": None,
                "override_id": None,
                "intervention_required": False,
                "supervision_type": "NATIVE_PRESERVATION_TARGETED",
                "source_evidence": "Stage2D.3B native-preferred intervention policy",
                "training_eligible": True,
                "split": "TRAIN",
                "selection_basis": "deterministic metadata-diversity ordering; no phone supervision",
            })
            output.append(result)
    return sorted(output, key=lambda row: (str(row["word"]), str(row["utterance_id"]))), exclusions


def build_general_native(path: Path, *, excluded_ids: set[str], count: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = read_jsonl(path)
    candidates: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: str(value.get("utterance_id", ""))):
        if str(row.get("utterance_id")) in excluded_ids:
            continue
        if not str(row.get("transcript", "")).strip():
            continue
        candidates.append(row)
    selected: list[dict[str, Any]] = []
    seen_transcripts: set[str] = set()
    exclusions = Counter()
    for row in candidates:
        text = str(row.get("transcript", ""))
        if text in seen_transcripts:
            exclusions["duplicate_transcript"] += 1
            continue
        selected.append(native_row(row))
        seen_transcripts.add(text)
        if len(selected) >= count:
            break
    return selected, exclusions


def phone_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    words_by_phone: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("split") != "TRAIN":
            continue
        word = str(row["word"])
        for phone in row["canonical_experimental_phone_sequence"]:
            counts[phone] += 1
            words_by_phone[phone].add(word)
    absent = sorted(PRONUNCIATION_ALPHABET_V0 - set(counts))
    return {
        "inventory": PRONUNCIATION_ALPHABET_ID,
        "train_utterance_count": sum(row.get("split") == "TRAIN" for row in rows),
        "phone_occurrence_counts": dict(sorted(counts.items())),
        "lexical_targets_by_phone": {phone: sorted(words_by_phone[phone]) for phone in sorted(words_by_phone)},
        "absent_phones": absent,
        "underrepresented_phones_threshold_lt_2": sorted(phone for phone, count in counts.items() if count < 2),
        "count_unit_note": "Counts are lexical phone occurrences in distinct training utterances; codec frames are not counted as phonetic evidence.",
    }


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    return (2 * sum((index + 1) * value for index, value in enumerate(ordered)) - (n + 1) * sum(ordered)) / (n * sum(ordered))


def word_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["word"]) for row in rows if row.get("split") == "TRAIN")
    return {
        "metric": "Gini coefficient over positive TRAIN occurrence counts",
        "examples_per_target": dict(sorted(counts.items())),
        "fractions": {word: counts[word] / sum(counts.values()) for word in sorted(counts)} if counts else {},
        "min": min(counts.values()) if counts else 0,
        "max": max(counts.values()) if counts else 0,
        "gini": gini(list(counts.values())),
    }


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("split")) for row in rows).items()))


def build_scale_options(all_targets: Mapping[str, list[dict[str, Any]]], resolver: SpicorAudioResolver, native_path: Path) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for name, cap, native_count, targeted_count in (("SMALL", 5, 50, 3), ("MEDIUM", 10, 100, 5)):
        positive_rows: list[dict[str, Any]] = []
        train_count = eval_count = gold_count = 0
        per_word: dict[str, dict[str, int]] = {}
        for word in POSITIVE_WORDS:
            train, evaluation, gold, _ = split_positive(all_targets[word], word=word, cap=cap, resolver=resolver)
            positive_rows.extend(train + evaluation + gold)
            per_word[word] = {"train": len(train), "eval": len(evaluation), "gold": len(gold)}
            train_count += len(train)
            eval_count += len(evaluation)
            gold_count += len(gold)
        targeted, _ = build_targeted_native(all_targets, resolver, limit=targeted_count)
        general, _ = build_general_native(native_path, excluded_ids={row["utterance_id"] for row in positive_rows + targeted}, count=native_count)
        options.append({
            "name": name,
            "positive_intervention": {"train": train_count, "eval_seen_word_unseen_context": eval_count, "human_gold_reference": gold_count, "per_word": per_word},
            "targeted_native_preservation": len(targeted),
            "general_native_preservation": len(general),
            "training_example_count": train_count + len(targeted) + len(general),
            "evaluation_example_count_excluding_gold": eval_count,
            "unique_positive_words": len(POSITIVE_WORDS),
            "lexical_balance": word_balance([row for row in positive_rows if row.get("split") == "TRAIN"]),
            "human_review_burden": "No new human review; uses frozen Stage2D.3 evidence and existing native set.",
            "risk": "SMALL has only three positive train contexts per word where recurrence permits; MEDIUM remains single-speaker and needs native/Qwen preservation checks.",
        })
    return options


def build_evaluation_matrix(split: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage2d4-evaluation-matrix-v0.1",
        "rows": [
            {"set": "POSITIVE_SEEN_WORD_TRANSFER", "words": list(POSITIVE_WORDS), "conditions": ["native", "pre_training_supported_v0", "post_training_explicit"], "checks": ["held-out contexts", "human gold reference", "trajectory class"]},
            {"set": "NEGATIVE_NATIVE_PREFERRED", "words": list(TARGETED_NATIVE_WORDS), "conditions": ["native", "post_training_no_intervention"], "checks": ["same text", "localized residual no-op", "q0 preservation"]},
            {"set": "GENERAL_NATIVE_PRESERVATION", "source": "stage2d2_native_preservation_set.jsonl", "checks": ["ordinary transcript diversity", "q0 KL", "duration/trajectory regression"]},
            {"set": "PHONE_CONTRAST_FIXTURES", "words": list(PHONE_CONTRAST_FIXTURES), "training_usage": "evaluation only", "checks": ["existing causal contrast remains intact"]},
            {"set": "EXTERNAL_UNSEEN", "words": list(EXTERNAL_HOLDOUTS), "training_usage": "evaluation only", "expected": "no guaranteed improvement; monitor unintended behavior"},
        ],
        "split_reference": {
            "positive_train_occurrences": split.get("positive_train_occurrences", []),
            "positive_eval_occurrences": split.get("positive_eval_occurrences", []),
            "human_gold_occurrences": split.get("human_gold_occurrences", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DEFAULT)
    parser.add_argument("--policy", type=Path, default=POLICY_DEFAULT)
    parser.add_argument("--native", type=Path, default=NATIVE_DEFAULT)
    args = parser.parse_args()

    inventory = load_inventory(args.inventory)
    resolver = make_resolver(inventory, archive=args.archive, cache=args.cache_root)
    targets = target_rows(inventory, (*POSITIVE_WORDS, *TARGETED_NATIVE_WORDS))
    policy = read_json(args.policy)
    policy_by_word = {str(row["word"]): row for row in policy["entries"]}
    if any(policy_by_word[word]["classification"] != "EXPLICIT_OVERRIDE_SUPPORTED" for word in POSITIVE_WORDS):
        raise ValueError("frozen Stage2D.3 policy does not support all three positive words")
    if any(policy_by_word[word]["classification"] != "NATIVE_PREFERRED" for word in TARGETED_NATIVE_WORDS):
        raise ValueError("frozen Stage2D.3 policy does not mark native targets correctly")
    if any(phone not in PRONUNCIATION_ALPHABET_V0 for sequence in POSITIVE_PHONES.values() for phone in sequence):
        raise ValueError("positive experimental sequence contains a non-v0 phone")

    positive_rows: list[dict[str, Any]] = []
    positive_exclusions: Counter[str] = Counter()
    split = {"positive_train_occurrences": [], "positive_eval_occurrences": [], "human_gold_occurrences": []}
    raw_counts: dict[str, int] = {}
    usable_counts: dict[str, int] = {}
    per_word_counts: dict[str, dict[str, int]] = {}
    for word in POSITIVE_WORDS:
        raw_counts[word] = len(targets[word])
        valid = [row for row in targets[word] if not quality_reasons(row)]
        usable_counts[word] = sum(resolver.resolve(row["utterance_id"]).status != "MISSING" for row in valid)
        train, evaluation, gold, exclusions = split_positive(targets[word], word=word, cap=10, resolver=resolver)
        positive_rows.extend(train + evaluation + gold)
        positive_exclusions.update({f"{word}:{key}": value for key, value in exclusions.items()})
        per_word_counts[word] = {"raw_recurrence": raw_counts[word], "quality_usable_archive_backed": usable_counts[word], "selected_non_gold": len(train) + len(evaluation), "train": len(train), "eval_seen_word_unseen_context": len(evaluation), "human_gold": len(gold)}
        split["positive_train_occurrences"].extend(row["utterance_id"] + f":word:{int(row['target_word_index']):04d}" for row in train)
        split["positive_eval_occurrences"].extend(row["utterance_id"] + f":word:{int(row['target_word_index']):04d}" for row in evaluation)
        split["human_gold_occurrences"].extend(row["utterance_id"] + f":word:{int(row['target_word_index']):04d}" for row in gold)
    positive_rows.sort(key=lambda row: (str(row.get("split")), str(row["word"]), str(row["utterance_id"])))

    targeted_native, targeted_exclusions = build_targeted_native(targets, resolver, limit=5)
    positive_and_targeted_ids = {row["utterance_id"] for row in positive_rows + targeted_native}
    general_native, general_exclusions = build_general_native(args.native, excluded_ids=positive_and_targeted_ids, count=100)

    output = args.output
    write_jsonl(output / "stage2d4_positive_interventions.jsonl", positive_rows)
    write_jsonl(output / "stage2d4_targeted_native_preservation.jsonl", targeted_native)
    write_jsonl(output / "stage2d4_general_native_preservation.jsonl", general_native)

    split["targeted_native_occurrences"] = [row["utterance_id"] for row in targeted_native]
    split["general_native_occurrences"] = [row["utterance_id"] for row in general_native]
    split["external_holdouts"] = list(EXTERNAL_HOLDOUTS)
    split["phone_contrast_fixtures"] = list(PHONE_CONTRAST_FIXTURES)
    split["gold_excluded_from_training"] = True
    split["transcript_leakage_policy"] = "No duplicate transcript across positive train/eval/gold; general native candidates exclude selected source IDs and duplicate transcripts."
    write_json(output / "stage2d4_split_plan.json", split)

    all_positive_non_gold = [row for row in positive_rows if not row["is_human_gold_reference"]]
    write_json(output / "stage2d4_phone_coverage.json", phone_coverage(all_positive_non_gold))
    write_json(output / "stage2d4_word_balance.json", word_balance(all_positive_non_gold))
    write_json(output / "stage2d4_exclusion_report.json", {
        "schema_version": "stage2d4-exclusion-report-v0.1",
        "positive_exclusions": dict(sorted(positive_exclusions.items())),
        "targeted_native_exclusions": dict(sorted(targeted_exclusions.items())),
        "general_native_exclusions": dict(sorted(general_exclusions.items())),
        "gold_references_excluded_from_training": list(GOLD_REFERENCES.values()),
        "unresolved_phone_mapping_exclusions": {"Nagar": "native-preferred; no phone label", "Banerjee": "explicit override unsafe; no phone label"},
        "policy": "No unsupported or unresolved symbolic phone sequence is included in POSITIVE_INTERVENTION rows.",
    })

    pairing = [{
        "occurrence_id": row["utterance_id"] + f":word:{int(row['target_word_index']):04d}",
        "word": row["word"],
        "split": row["split"],
        "intervention_audio": row["audio_resolver_path"],
        "native_reference": row["paired_native_reference"],
        "same_text_same_audio_setup": True,
        "teacher_forced_history_requirement": "future native and conditioned comparison must share target acoustic history",
    } for row in all_positive_non_gold]
    write_json(output / "stage2d4_trajectory_pairing_plan.json", {
        "schema_version": "stage2d4-trajectory-pairing-plan-v0.1",
        "pair_count": len(pairing),
        "pairs": pairing,
        "gold_reference_ids_not_training_pairs": list(GOLD_REFERENCES.values()),
    })

    write_json(output / "stage2d4_loss_design.json", {
        "schema_version": "stage2d4-loss-design-v0.1",
        "qwen_frozen": True,
        "current_stage2b_objective": {
            "positive_target": "masked codebook CE on target frames, default q0-q3",
            "non_target_preservation": "masked logits KL against frozen native logits",
            "eos_preservation": "optional masked EOS-logit KL",
            "schedule": "gate-only warmup followed by bridge plus gate in the established implementation",
            "available_helpers": ["compute_qwen_split_target_ce", "compute_qwen_split_preservation_kl", "masked_logits_kl"],
            "not_present": ["residual-energy penalty", "q0 trajectory loss", "learned word-level gate"],
        },
        "support_audit": {
            "positive_intervention": "SUPPORTED by current target CE plus non-target native preservation KL.",
            "targeted_native_preservation": "The no-override path is already an exact localized no-op. Records carry no phone labels; use as preservation/evaluation controls. They do not create a gradient for a nonexistent intervention.",
            "general_native_preservation": "Use the existing native KL/evaluation path; do not invent a word-level gate.",
        },
        "recommended_minimal_formulation": {
            "name": "V1_POSITIVE_PLUS_NATIVE_PRESERVATION",
            "positive_loss": "existing target CE + non-target native KL; retain optional EOS preservation",
            "native_loss": "native-preservation KL only when a conditioned-vs-native teacher-forced path is explicitly constructed; otherwise exact no-op is the contract",
            "residual_penalty": "defer; it could suppress a needed local intervention",
            "learned_intervention_gate": "not required now; explicit override-present vs no-override path already selects intervention",
        },
        "future_loss_change_requires_pilot": True,
    })

    write_json(output / "stage2d4_trajectory_metrics.json", {
        "schema_version": "stage2d4-trajectory-metrics-v0.1",
        "purpose": "trajectory-safety measurement; not a duration cap and not initially a training loss",
        "teacher_forced_metrics": [
            {"metric": "q0_kl_per_step", "scope": "native vs conditioned under shared target acoustic history"},
            {"metric": "q0_top1_divergence_count", "scope": "all valid q0 decoding positions"},
            {"metric": "first_q0_divergent_step", "scope": "early trajectory risk"},
            {"metric": "eos_logit_divergence", "scope": "termination preservation"},
            {"metric": "native_conditioned_margin_at_risk_points", "scope": "decision sensitivity"},
        ],
        "generation_metrics": ["output_codec_frame_count", "decoded_duration_seconds", "eos_frame", "trajectory_class"],
        "initial_role": "evaluation gate for every positive and native-preservation comparison",
        "loss_recommendation": "evaluation-only in first bounded experiment; add a q0 term only after observing a reproducible regression and an ablation plan",
        "pathology_policy": "do not treat a long trajectory alone as phone failure; report NORMAL/LONG/MAX_LENGTH/FAILED separately",
    })

    variants = [
        {"id": "V0", "name": "POSITIVE_ONLY_BASELINE", "train_sets": ["POSITIVE_INTERVENTION"], "loss": "current target CE + non-target native KL (+ optional EOS KL)", "isolates": "whether validated interventions can be learned at all", "native_preservation": "evaluation only"},
        {"id": "V1", "name": "POSITIVE_PLUS_NATIVE_PRESERVATION", "train_sets": ["POSITIVE_INTERVENTION", "NATIVE_PRESERVATION_TARGETED", "NATIVE_PRESERVATION_GENERAL"], "loss": "V0 loss plus existing native-preservation KL orchestration; no new architecture", "isolates": "whether intervention learning coexists with native behavior preservation", "recommended": True},
        {"id": "V2", "name": "V1_WITH_TRAJECTORY_SAFETY_ABLATION", "train_sets": ["POSITIVE_INTERVENTION", "NATIVE_PRESERVATION_TARGETED", "NATIVE_PRESERVATION_GENERAL"], "loss": "V1; trajectory metrics added as an eval gate, optional q0 term only as a later ablation", "isolates": "whether measured early q0 divergence predicts or prevents trajectory regressions", "recommended_for_first_run": False},
    ]
    write_json(output / "stage2d4_experiment_variants.json", {"schema_version": "stage2d4-experiment-variants-v0.1", "variants": variants, "architecture_expansion": False})

    scales = build_scale_options(targets, resolver, args.native)
    medium = next(option for option in scales if option["name"] == "MEDIUM")
    write_json(output / "stage2d4_scale_options.json", {"schema_version": "stage2d4-scale-options-v0.1", "options": scales, "recommended": {"name": "MEDIUM", "reason": f"SMALL provides only three positive training contexts per word; MEDIUM provides {medium['positive_intervention']['train']} positive train examples for the current quality-filtered inventory while remaining bounded and fully held out from the three gold references."}})
    write_json(output / "stage2d4_evaluation_matrix.json", build_evaluation_matrix(split))
    write_json(output / "stage2d4_success_criteria.json", {
        "schema_version": "stage2d4-success-criteria-v0.1",
        "frozen_before_training": True,
        "positive": ["validated intervention improves blinded pronunciation relative to native/pre-training baseline on held-out contexts", "improvement is present on human gold references not used in training", "no new LONG or MAX_LENGTH trajectory regression attributable to intervention"],
        "negative": ["Nagar no-intervention output remains native-equivalent", "Banerjee remains native-preferred and explicit override is not introduced", "no-override localized residual remains exact/no-op"],
        "general": ["ordinary native set has bounded preservation KL", "no systematic SPICOR speaker/accent takeover", "duration and trajectory distributions do not regress materially"],
        "mechanism": ["Singh/Mumbai/Kumar causal contrast fixtures remain usable", "Qwen stays frozen", "swara-phones-v0 remains unchanged"],
        "failure_policy": "training CE alone cannot declare success; any positive gain with native or trajectory regression is not a successful intervention experiment",
    })

    training_rows = [row for row in positive_rows if row["split"] == "TRAIN"] + targeted_native + general_native
    summary = {
        "schema_version": "stage2d4-training-design-v0.1",
        "stage": "STAGE2D.4",
        "design_only": True,
        "training_performed": False,
        "qwen_loaded": False,
        "checkpoint_modified": False,
        "swara_phones_v0_modified": False,
        "source_inventory": str(args.inventory),
        "archive_backed_design": args.archive.is_file(),
        "positive_intervention_words": list(POSITIVE_WORDS),
        "targeted_native_preservation_words": list(TARGETED_NATIVE_WORDS),
        "raw_recurrence": raw_counts,
        "quality_usable_archive_backed_occurrences": usable_counts,
        "selected_positive_counts": per_word_counts,
        "positive_train_count": sum(row.get("split") == "TRAIN" for row in positive_rows),
        "positive_eval_count": sum(row.get("split") == "EVAL_SEEN_WORD_UNSEEN_CONTEXT" for row in positive_rows),
        "human_gold_count": sum(row.get("split") == "HUMAN_GOLD_REFERENCE" for row in positive_rows),
        "targeted_native_count": len(targeted_native),
        "general_native_count": len(general_native),
        "total_train_examples_recommended": len(training_rows),
        "recommended_scale": "MEDIUM",
        "gold_references": GOLD_REFERENCES,
        "gold_references_excluded_from_training": all(row["split"] != "TRAIN" for row in positive_rows if row["utterance_id"] in GOLD_REFERENCES.values()),
        "no_phone_labels_on_native_records": all(row.get("phone_sequence") is None and not row.get("intervention_required") for row in targeted_native + general_native),
        "no_universal_pur_rule": True,
        "status": "READY_FOR_STAGE2D4_TRAINING_IMPLEMENTATION",
    }
    write_json(output / "stage2d4_summary.json", summary)

    doc = f"""# Stage2D.4 bounded pronunciation intervention training design\n\nThis is a design checkpoint only. It reads SPICOR metadata and frozen Stage2D.3 evidence; it does not load Qwen, train, synthesize, materialize audio, or modify `swara-phones-v0`.\n\n## Frozen intervention policy\n\n- Jamshedpur: `J A M SH I D P U`\n- Chandigarh: `CH A N D I G AA`\n- Nagpur: `N A G P U R`\n- Nagar: native preferred; no phone label\n- Banerjee: native preferred; explicit override unsafe; no phone label\n\nThe three positive words have raw recurrence {raw_counts}; archive-backed quality-usable counts {usable_counts}. With a cap of 10 selected occurrences including the human gold reference, the design contains {summary['positive_train_count']} positive train examples, {summary['positive_eval_count']} held-out seen-word/unseen-context examples, and three gold references.\n\n## Recommended scale\n\n`MEDIUM` is recommended: {summary['positive_train_count']} positive train examples, {summary['targeted_native_count']} targeted native-preservation examples, and {summary['general_native_count']} general native-preservation examples ({summary['total_train_examples_recommended']} total training examples). The three human reference utterances are excluded from training. `SMALL` is retained as a lower-risk option but has only three positive train contexts per word where recurrence permits.\n\n## Loss and gate policy\n\nThe current Stage2B implementation provides target codebook CE, non-target native-logit KL, and optional EOS-preservation KL; it has no residual-energy, q0-trajectory, or learned word-level gate loss. Use the existing target CE/KL objective for positive interventions. Native records carry no phone labels and represent no-intervention preservation controls. The localized no-override path is already an exact no-op, so a new learned intervention gate is not justified at this stage.\n\nTrajectory metrics—q0 KL per step, q0 top-1 divergence, first divergent step, EOS-logit divergence, and trajectory class—are evaluation gates in the first experiment. They are not replaced by a crude duration cap and are not initially added as a loss.\n\n## Leakage and pairing\n\nEach positive record contains its experimental sequence, target span, human gold ID, split, and resolver path. Gold references are held out. Each non-gold positive has a same-text/same-audio native pairing record for future teacher-forced comparisons. Native-preservation rows have `phone_sequence: null` and `intervention_required: false`.\n\nThe evaluation matrix retains held-out positive transfer, Nagar/Banerjee native preference, general native preservation, Singh/Mumbai/Kumar mechanism fixtures, and external names including Dasharatha. Success is not declared from CE alone.\n"""
    (REPO_ROOT / "docs/stage2d/STAGE2D4_BOUNDED_TRAINING_DESIGN.md").parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "docs/stage2d/STAGE2D4_BOUNDED_TRAINING_DESIGN.md").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
