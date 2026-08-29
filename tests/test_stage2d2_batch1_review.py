import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
BATCH_ROOT = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review"
BATCH = ROOT / "artifacts/stage2d/stage2d2_dataset_design/stage2d2_review_batch1.json"
EXPECTED_WORDS = [
    "nagar", "srinagar", "hyderabad", "bengaluru", "chandigarh", "chhattisgarh",
    "banerjee", "ahmedabad", "jee", "nagpur", "dimapur", "jaipur", "manipur",
    "raipur", "chatterjee", "gorakhpur", "mukherjee", "sambalpur", "aligarh",
    "allahabad", "jamshedpur", "udhampur", "azamgarh", "sultanpur", "bilaspur",
]


def _index():
    return json.loads((BATCH_ROOT / "batch1_human_review_index.json").read_text(encoding="utf-8"))


def test_batch1_words_and_selection_are_frozen():
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    assert [row["normalized_word"] for row in batch["words"]] == EXPECTED_WORDS
    index = _index()
    assert [row["normalized_word"] for row in index["words"]] == EXPECTED_WORDS
    assert index["entry_count"] == 124
    assert all(3 <= len(row["entries"]) <= 5 for row in index["words"])


def test_review_audio_paths_and_alignment_geometry_are_valid():
    index = _index()
    entries = [entry for word in index["words"] for entry in word["entries"]]
    assert len({entry["occurrence_id"] for entry in entries}) == len(entries)
    assert all(0 <= entry["aligned_start_seconds"] < entry["aligned_end_seconds"] <= entry["source_audio_duration_seconds"] + 0.05 for entry in entries)
    for entry in entries:
        for key in ("full_audio_path", "context_audio_path", "word_only_audio_path"):
            path = (BATCH_ROOT / entry[key]).resolve()
            assert not Path(entry[key]).is_absolute()
            assert path.is_file(), path


def test_html_has_three_ordered_audio_views_and_no_phone_assignment():
    html = (BATCH_ROOT / "human_review.html").read_text(encoding="utf-8")
    assert html.count("<audio controls") == 124 * 3
    assert html.count("<strong>Full utterance</strong>") == 124
    assert html.count("<strong>Context</strong>") == 124
    assert html.count("<strong>Word only</strong>") == 124
    assert "file://" not in html
    assert "S I NG" not in html


def test_decision_template_verdicts_remain_unfilled():
    template = (BATCH_ROOT / "batch1_human_review_decisions_template.md").read_text(encoding="utf-8")
    assert template.count("**FINAL VERDICT:** \n") == 25
    assert "CANONICAL_STABLE /" not in template
    assert "VARIANT_PRESENT /" not in template


def test_summary_reports_no_failures_and_expected_sample_buckets():
    summary = json.loads((BATCH_ROOT / "batch1_human_review_summary.json").read_text(encoding="utf-8"))
    assert summary["word_count"] == 25
    assert summary["total_review_occurrences"] == 124
    assert len(summary["words_with_5_reviewed_samples"]) == 24
    assert summary["words_with_fewer_than_3_reviewed_samples"] == []
    assert summary["alignment_failure_count"] == 0
    assert summary["audio_generation_failure_count"] == 0
