import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "design_stage2d2_dataset.py"
SPEC = importlib.util.spec_from_file_location("stage2d2_design", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _item(identifier, preceding="a", following="b", position="MEDIAL", length="MEDIUM_7_14", duration="MEDIUM_3_7S"):
    return {
        "occurrence_id": identifier,
        "preceding_word": preceding,
        "following_word": following,
        "position_bucket": position,
        "length_bucket": length,
        "duration_bucket": duration,
        "domain": "TEST",
        "token_count": 10,
        "transcript": f"A unique transcript {identifier}",
    }


def test_diverse_selection_is_bounded_and_deterministic():
    rows = [_item(f"utt{i:02d}", preceding=f"p{i}", following=f"f{i}", position="INITIAL" if i == 0 else "MEDIAL") for i in range(8)]
    first = MODULE.select_diverse_occurrences(rows, 4)
    second = MODULE.select_diverse_occurrences(list(reversed(rows)), 4)
    assert [row["occurrence_id"] for row in first] == [row["occurrence_id"] for row in second]
    assert len(first) == 4


def test_split_is_disjoint_and_repeatable():
    rows = [_item(f"utt{i:02d}") for i in range(20)]
    train_a, eval_a = MODULE._split_selected(rows)
    train_b, eval_b = MODULE._split_selected(list(reversed(rows)))
    assert {row["occurrence_id"] for row in train_a}.isdisjoint({row["occurrence_id"] for row in eval_a})
    assert train_a == train_b
    assert eval_a == eval_b


def test_phone_coverage_counts_unique_phone_per_utterance():
    rows = [
        {"target_normalized_word": "one", "canonical_phone_sequence": ["A", "A", "B"]},
        {"target_normalized_word": "two", "canonical_phone_sequence": ["B", "C"]},
    ]
    coverage = MODULE._phone_coverage(rows)
    assert coverage["phone_utterance_coverage"]["A"] == 1
    assert coverage["phone_utterance_coverage"]["B"] == 2


def test_native_row_has_no_override_supervision():
    row = MODULE._native_row({
        "utterance_id": "u1", "transcript": "Clean ordinary sentence", "audio_path": "a.wav",
        "source_wav_member": None, "source_duration_seconds": 1.0, "source_sample_rate_hz": 44100,
        "domain": "TEST", "split": "train",
    })
    assert row["supervision_type"] == "NATIVE_PRESERVATION"
    assert row["override_id"] is None
    assert row["target_words"] == []


def test_review_queue_never_invents_phone_sequence():
    payload = {"tiers": {"TIER_2_REVIEW_REQUIRED": [{"normalized_word": "nagar", "word": "Nagar", "occurrence_count": 2}]}}
    queue = MODULE._build_review_queue(payload, {})
    assert queue[0]["proposed_canonical_phone_candidate"] is None
    assert queue[0]["training_eligible"] is False


def test_generated_artifacts_preserve_training_boundaries():
    root = Path(__file__).parents[1] / "artifacts/stage2d/stage2d2_dataset_design"
    explicit = [json.loads(line) for line in (root / "stage2d2_explicit_candidates.jsonl").read_text().splitlines() if line]
    assert explicit
    assert {row["target_normalized_word"] for row in explicit} == {
        "agrawal", "gupta", "kashmir", "kumar", "mishra", "mumbai", "sharma",
    }
    assert all(row["canonical_phone_sequence"] for row in explicit)
    assert all(set(row["canonical_phone_sequence"]).issubset(MODULE.PRONUNCIATION_ALPHABET_V0) for row in explicit)
    split = json.loads((root / "stage2d2_split_plan.json").read_text())
    assert set(split["explicit_train_occurrences"]).isdisjoint(set(split["eval_seen_word_unseen_context_occurrences"]))
    assert split["leakage_check"]["explicit_train_source_utterances_disjoint_from_eval"] is True
    assert split["leakage_check"]["explicit_train_eval_source_overlap_count"] == 0
    assert split["leakage_check"]["transfer_text_exact_match_with_training_transcript_count"] == 0


def test_external_fixture_words_are_explicit_and_deterministic(tmp_path):
    fixture = tmp_path / "fixtures.json"
    fixture.write_text(json.dumps({"transfer": {}, "unseen_name": [
        "Anirban joined the review panel.",
        "The committee invited Ashwini to the meeting.",
        "Chandrashekhar submitted the final report.",
        "The researchers met Karthik after the seminar.",
    ]}))
    plan = MODULE._fixture_plan(fixture, set(), set())
    assert [row["word"] for row in plan["external_unseen"]] == [
        "dasharatha", "anirban", "ashwini", "chandrashekhar", "karthik",
    ]
