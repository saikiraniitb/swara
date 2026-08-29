import importlib.util
from pathlib import Path

import numpy as np

from swara.diagnostics.acoustic_consistency import (
    classify_consistency,
    dtw_distance,
    extract_features,
    pairwise_distances,
)
from swara.diagnostics.pronunciation_atlas import AtlasOccurrence


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_stage2d1b_acoustic_consistency.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("stage2d1b_runner", _SCRIPT_PATH)
assert _SCRIPT_SPEC and _SCRIPT_SPEC.loader
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)


def _occurrence(source_id: str, index: int, word: str = "Kumar") -> AtlasOccurrence:
    return AtlasOccurrence(
        occurrence_id=f"{source_id}:word:{index:04d}",
        utterance_id=source_id,
        word_index=index,
        surface_form=word,
        normalized_word=word.casefold(),
        full_transcript=f"Before {word} after {source_id}",
        source_span_start=7,
        source_span_end=7 + len(word),
        preceding_word="Before",
        following_word="after",
        audio_path=f"data/{source_id}.wav",
        source_wav_member=None,
        split="train",
        domain="test",
        source_duration_seconds=1.0,
        source_sample_rate_hz=24000,
        interest_signals=(),
    )


def test_target_selection_is_deterministic_and_bounded():
    rows = []
    for i in range(7):
        rows.append(_occurrence(f"id{i:03d}", 1))
    for row in rows:
        path = Path(row.audio_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    try:
        selected, absent = _SCRIPT.select_target_set(rows, max_occurrences=3, repo_root=".")
        again, absent_again = _SCRIPT.select_target_set(rows, max_occurrences=3, repo_root=".")
    finally:
        for row in rows:
            Path(row.audio_path).unlink(missing_ok=True)
    assert selected == again
    assert absent == absent_again
    kumar = next(item for item in selected if item["normalized_word"] == "kumar")
    assert kumar["sampled_occurrence_count"] == 3
    assert kumar["corpus_occurrence_count"] == 7


def test_dtw_and_pairwise_distance_are_finite():
    first = np.zeros((4, 3), dtype=np.float32)
    second = np.ones((4, 3), dtype=np.float32)
    assert dtw_distance(first, first) == 0.0
    assert dtw_distance(first, second) > 0.0


def test_feature_extraction_and_observation_pairwise_geometry():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = np.sin(2 * np.pi * 180 * time).astype(np.float32)
    features = extract_features(waveform, sample_rate, 0.25, 0.5)
    assert features["mfcc_frames"].ndim == 2
    assert np.isfinite(features["mfcc_frames"]).all()


def test_classification_does_not_claim_multimodality_without_evidence():
    assert classify_consistency(usable_count=2, relative_variability=None, outlier_count=0, context_effect_present=False) == "INSUFFICIENT_EVIDENCE"
    assert classify_consistency(usable_count=5, relative_variability=1.0, outlier_count=0, context_effect_present=False) == "ACOUSTICALLY_STABLE"
    assert classify_consistency(usable_count=5, relative_variability=2.0, outlier_count=0, context_effect_present=False) == "CONTEXT_VARIANT"
