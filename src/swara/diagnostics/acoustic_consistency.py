"""Bounded repeated-word acoustic consistency analysis.

This module uses the repository's pinned exact-transcript CTC aligner only to
obtain word boundaries.  It never assigns phoneme labels to audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping

import numpy as np


class AcousticConsistencyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AcousticObservation:
    occurrence_id: str
    normalized_word: str
    word_start_seconds: float
    word_end_seconds: float
    alignment_confidence: float
    utterance_duration_seconds: float
    lexical_token_count: int
    word_duration_seconds: float
    rms_mean: float
    spectral_centroid_mean: float
    f0_mean_hz: float | None
    f0_std_hz: float | None
    mfcc_mean: tuple[float, ...]
    mfcc_std: tuple[float, ...]
    context_window_seconds: float
    _mfcc_frames: Any = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("_mfcc_frames", None)
        return result


def _finite_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def extract_features(waveform: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float, *, context_seconds: float = 0.1) -> AcousticObservation | dict[str, Any]:
    """Extract compact descriptors and retain MFCC frames only in memory."""

    import librosa

    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise AcousticConsistencyError("waveform must be finite mono audio")
    if not 0 <= start_seconds < end_seconds:
        raise AcousticConsistencyError("word interval must be positive and in range")
    start = max(0, int(round((start_seconds - context_seconds) * sample_rate)))
    end = min(waveform.size, int(round((end_seconds + context_seconds) * sample_rate)))
    if end <= start:
        raise AcousticConsistencyError("context interval is empty")
    segment = waveform[start:end].astype(np.float32, copy=False)
    minimum = 400
    if segment.size < minimum:
        segment = np.pad(segment, (0, minimum - segment.size))
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13, n_fft=400, win_length=400, hop_length=160, center=True)
    mfcc = np.asarray(mfcc, dtype=np.float32).T
    mfcc_mean = np.nan_to_num(mfcc.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    mfcc_std = np.nan_to_num(mfcc.std(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    normalized_mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-5)
    rms = librosa.feature.rms(y=segment, frame_length=400, hop_length=160, center=True)[0]
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate, n_fft=400, hop_length=160, center=True)[0]
    f0 = librosa.yin(segment, fmin=60.0, fmax=400.0, sr=sample_rate, frame_length=512, hop_length=160)
    valid_f0 = f0[np.isfinite(f0)]
    return {
        "mfcc_frames": normalized_mfcc.astype(np.float32),
        "mfcc_mean": tuple(_finite_float(value) for value in mfcc_mean),
        "mfcc_std": tuple(_finite_float(value) for value in mfcc_std),
        "rms_mean": _finite_float(np.mean(rms)),
        "spectral_centroid_mean": _finite_float(np.mean(centroid)),
        "f0_mean_hz": _finite_float(np.mean(valid_f0)) if valid_f0.size else None,
        "f0_std_hz": _finite_float(np.std(valid_f0)) if valid_f0.size else None,
    }


def make_observation(occurrence: Mapping[str, Any], alignment: Mapping[str, Any], features: Mapping[str, Any]) -> AcousticObservation:
    return AcousticObservation(
        occurrence_id=str(occurrence["occurrence_id"]),
        normalized_word=str(occurrence["normalized_word"]),
        word_start_seconds=float(alignment["start_seconds"]),
        word_end_seconds=float(alignment["end_seconds"]),
        alignment_confidence=float(alignment["confidence"]),
        utterance_duration_seconds=float(occurrence["utterance_duration_seconds"]),
        lexical_token_count=int(occurrence["lexical_token_count"]),
        word_duration_seconds=float(alignment["end_seconds"] - alignment["start_seconds"]),
        rms_mean=float(features["rms_mean"]),
        spectral_centroid_mean=float(features["spectral_centroid_mean"]),
        f0_mean_hz=features["f0_mean_hz"],
        f0_std_hz=features["f0_std_hz"],
        mfcc_mean=tuple(features["mfcc_mean"]),
        mfcc_std=tuple(features["mfcc_std"]),
        context_window_seconds=0.1,
        _mfcc_frames=features["mfcc_frames"],
    )


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return length-normalized Euclidean DTW distance."""

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1] or not first.size or not second.size:
        raise AcousticConsistencyError("DTW inputs must be non-empty [frames, features] arrays")
    costs = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    accumulated = np.full((first.shape[0] + 1, second.shape[0] + 1), np.inf, dtype=np.float64)
    lengths = np.zeros_like(accumulated, dtype=np.int32)
    accumulated[0, 0] = 0.0
    for i in range(1, first.shape[0] + 1):
        for j in range(1, second.shape[0] + 1):
            candidates = ((accumulated[i - 1, j], lengths[i - 1, j]), (accumulated[i, j - 1], lengths[i, j - 1]), (accumulated[i - 1, j - 1], lengths[i - 1, j - 1]))
            best_cost, best_length = min(candidates, key=lambda item: (item[0], item[1]))
            accumulated[i, j] = best_cost + float(costs[i - 1, j - 1])
            lengths[i, j] = best_length + 1
    return float(accumulated[-1, -1] / max(1, lengths[-1, -1]))


def pairwise_distances(observations: Iterable[AcousticObservation]) -> list[dict[str, Any]]:
    items = sorted(observations, key=lambda item: item.occurrence_id)
    result: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            acoustic = dtw_distance(left._mfcc_frames, right._mfcc_frames)
            duration = abs(math.log(max(1e-6, left.word_duration_seconds) / max(1e-6, right.word_duration_seconds)))
            composite = acoustic + 0.25 * duration
            result.append({"left_occurrence_id": left.occurrence_id, "right_occurrence_id": right.occurrence_id, "dtw_mfcc_distance": acoustic, "log_duration_distance": duration, "composite_distance": composite})
    return result


def median_pairwise_distance(distances: Iterable[Mapping[str, Any]]) -> float | None:
    values = [float(item["composite_distance"]) for item in distances]
    return float(np.median(values)) if values else None


def classify_consistency(*, usable_count: int, relative_variability: float | None, outlier_count: int, context_effect_present: bool, multimodal_supported: bool = False) -> str:
    if usable_count < 3:
        return "INSUFFICIENT_EVIDENCE"
    if outlier_count > 0 and outlier_count >= max(1, usable_count // 4):
        return "LIKELY_DATA_OR_ALIGNMENT_OUTLIER"
    if multimodal_supported:
        return "MULTIMODAL_CANDIDATE"
    if context_effect_present:
        return "CONTEXT_VARIANT"
    if relative_variability is not None and relative_variability > 1.5:
        return "CONTEXT_VARIANT"
    return "ACOUSTICALLY_STABLE"
