"""Deterministic utilities for the continuous acoustic-target bake-off.

This module contains no model training code.  Model-specific extraction remains
thin and uses the frozen model's own projection, quantizer, and decoder modules.
Representations exposed to the generic perturbation helpers are time-major
``(T, C)`` arrays.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


BASE_SEED = 20260823
FSQ_ROUNDING_BOUNDARIES = (-1.5, -0.5, 0.5)


def deterministic_seed(*parts: object, base_seed: int = BASE_SEED) -> int:
    """Return a stable 63-bit seed independent of Python hash randomization."""

    payload = "\0".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def deterministic_order(
    rows: Sequence[dict], *, namespace: str, base_seed: int = BASE_SEED
) -> list[dict]:
    """Deterministically order rows by their utterance IDs."""

    return sorted(
        rows,
        key=lambda row: (
            deterministic_seed(namespace, row["utterance_id"], base_seed=base_seed),
            row["utterance_id"],
        ),
    )


@dataclass(frozen=True)
class ChannelStats:
    mean: np.ndarray
    std: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    p01: np.ndarray
    p99: np.ndarray
    frame_count: int

    @property
    def channels(self) -> int:
        return int(self.mean.shape[0])


def channel_statistics(representations: Iterable[np.ndarray]) -> ChannelStats:
    arrays = [np.asarray(rep, dtype=np.float64) for rep in representations]
    if not arrays or any(array.ndim != 2 for array in arrays):
        raise ValueError("representations must be non-empty (T, C) arrays")
    channels = arrays[0].shape[1]
    if any(array.shape[1] != channels for array in arrays):
        raise ValueError("all representations must have the same channel count")
    joined = np.concatenate(arrays, axis=0)
    return ChannelStats(
        mean=joined.mean(axis=0),
        std=joined.std(axis=0),
        minimum=joined.min(axis=0),
        maximum=joined.max(axis=0),
        p01=np.percentile(joined, 1, axis=0),
        p99=np.percentile(joined, 99, axis=0),
        frame_count=int(joined.shape[0]),
    )


def _smooth_time(noise: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 != 1:
        raise ValueError("smooth window must be a positive odd integer")
    if window == 1:
        return noise.copy()
    radius = window // 2
    padded = np.pad(noise, ((radius, radius), (0, 0)), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack(
        [np.convolve(padded[:, channel], kernel, mode="valid") for channel in range(noise.shape[1])],
        axis=1,
    )


def perturb_representation(
    clean: np.ndarray,
    channel_std: np.ndarray,
    *,
    sigma: float,
    seed: int,
    family: str,
    smooth_window: int = 9,
) -> np.ndarray:
    """Apply channel-aware IID or time-smoothed Gaussian perturbation.

    Smoothed noise is independently RMS-normalized per channel to the raw IID
    draw before multiplying by ``sigma * channel_std``.  No smoothing occurs
    across channels.
    """

    clean = np.asarray(clean)
    std = np.asarray(channel_std, dtype=np.float64)
    if clean.ndim != 2 or std.shape != (clean.shape[1],):
        raise ValueError("clean must be (T, C) and channel_std must be (C,)")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0:
        return clean.copy()
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(clean.shape)
    if family == "iid":
        noise = raw
    elif family == "smooth":
        noise = _smooth_time(raw, smooth_window)
        raw_rms = np.sqrt(np.mean(raw * raw, axis=0))
        smooth_rms = np.sqrt(np.mean(noise * noise, axis=0))
        noise = noise * (raw_rms / np.maximum(smooth_rms, 1e-12))[None, :]
    else:
        raise ValueError("family must be 'iid' or 'smooth'")
    perturbed = clean.astype(np.float64) + sigma * std[None, :] * noise
    return perturbed.astype(clean.dtype, copy=False)


def audio_integrity(waveform: np.ndarray, sample_rate: int) -> dict:
    wave = np.asarray(waveform, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(wave).all())
    if not finite or wave.size == 0:
        return {
            "sample_rate": int(sample_rate), "samples": int(wave.size),
            "duration_seconds": float(wave.size / sample_rate), "finite": finite,
            "non_silent": False, "rms": None, "peak": None, "clipping_count": None,
        }
    rms = float(np.sqrt(np.mean(wave * wave)))
    peak = float(np.max(np.abs(wave)))
    return {
        "sample_rate": int(sample_rate), "samples": int(wave.size),
        "duration_seconds": float(wave.size / sample_rate), "finite": True,
        "non_silent": bool(rms > 1e-6), "rms": rms, "peak": peak,
        "clipping_count": int(np.count_nonzero(np.abs(wave) >= 0.999)),
    }


def representation_distance(clean: np.ndarray, perturbed: np.ndarray) -> dict:
    clean64 = np.asarray(clean, dtype=np.float64)
    delta = np.asarray(perturbed, dtype=np.float64) - clean64
    if clean64.shape != delta.shape or clean64.ndim != 2:
        raise ValueError("representations must have the same (T, C) shape")
    derivative = np.diff(delta, axis=0)
    scale_l1 = np.mean(np.abs(clean64)) + 1e-12
    scale_l2 = np.sqrt(np.mean(clean64 * clean64)) + 1e-12
    return {
        "l1": float(np.mean(np.abs(delta))),
        "l2": float(np.sqrt(np.mean(delta * delta))),
        "normalized_l1": float(np.mean(np.abs(delta)) / scale_l1),
        "normalized_l2": float(np.sqrt(np.mean(delta * delta)) / scale_l2),
        "temporal_derivative_deviation": float(
            np.sqrt(np.mean(derivative * derivative)) if derivative.size else 0.0
        ),
    }


def waveform_spectral_metrics(clean: np.ndarray, noisy: np.ndarray, sample_rate: int) -> dict:
    """Dependency-light waveform degradation relative to the target's clean reconstruction."""

    import librosa

    a = torch.as_tensor(np.asarray(clean), dtype=torch.float32).flatten()
    b = torch.as_tensor(np.asarray(noisy), dtype=torch.float32).flatten()
    length = min(a.numel(), b.numel())
    a, b = a[:length], b[:length]
    window = torch.hann_window(1024)
    sa = torch.stft(a, 1024, 256, 1024, window, return_complex=True).abs()
    sb = torch.stft(b, 1024, 256, 1024, window, return_complex=True).abs()
    spectral_convergence = torch.linalg.vector_norm(sa - sb) / torch.clamp(
        torch.linalg.vector_norm(sa), min=1e-12
    )
    mel_filter = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=1024, n_mels=80, fmin=0, fmax=sample_rate / 2)
    ).to(dtype=sa.dtype)
    ma = mel_filter @ sa
    mb = mel_filter @ sb
    log_distance = torch.mean(torch.abs(torch.log(ma.clamp_min(1e-5)) - torch.log(mb.clamp_min(1e-5))))
    return {
        "spectral_convergence": float(spectral_convergence),
        "log_mel_waveform_distance": float(log_distance),
        "stoi": None,
        "pesq": None,
    }


def official_fsq_from_projected(quantizer, projected: torch.Tensor):
    """Continue from ResidualFSQ.project_in output using official FSQ modules."""

    if quantizer.num_quantizers != 1:
        raise ValueError("bake-off contract requires exactly one FSQ quantizer")
    layer = quantizer.layers[0]
    residual = layer.bound(projected)
    coordinates, indices = layer(residual)
    decoder_embedding = quantizer.project_out(coordinates)
    return decoder_embedding, indices, coordinates


def fsq_rounding_margin(quantizer, projected: torch.Tensor) -> torch.Tensor:
    """Distance to the nearest decision boundary in the actual rounding domain.

    ResidualFSQ first bounds ``projected`` to form its residual.  FSQ then calls
    ``bound`` again immediately before rounding.  The four-level rounding
    boundaries in that confirmed domain are -1.5, -0.5, and 0.5.
    """

    layer = quantizer.layers[0]
    rounding_domain = layer.bound(layer.bound(projected))
    boundaries = torch.tensor(
        FSQ_ROUNDING_BOUNDARIES, device=rounding_domain.device, dtype=rounding_domain.dtype
    )
    return torch.amin(torch.abs(rounding_domain.unsqueeze(-1) - boundaries), dim=-1)


def quantization_diagnostics(clean_indices: torch.Tensor, noisy_indices: torch.Tensor,
                             clean_coordinates: torch.Tensor, noisy_coordinates: torch.Tensor) -> dict:
    ci = clean_indices.reshape(-1).long()
    ni = noisy_indices.reshape(-1).long()
    cc = clean_coordinates.reshape(-1, clean_coordinates.shape[-1])
    nc = noisy_coordinates.reshape(-1, noisy_coordinates.shape[-1])
    coord_changes = cc.ne(nc)
    frame_changes = ci.ne(ni)
    changed_frames = int(frame_changes.sum())
    clean_bigrams = torch.stack([ci[:-1], ci[1:]], dim=1) if ci.numel() > 1 else ci.new_empty((0, 2))
    noisy_bigrams = torch.stack([ni[:-1], ni[1:]], dim=1) if ni.numel() > 1 else ni.new_empty((0, 2))
    return {
        "coordinate_boundary_crossing_rate": float(coord_changes.float().mean()),
        "frame_token_change_rate": float(frame_changes.float().mean()),
        "per_dimension_change_rate": [float(x) for x in coord_changes.float().mean(dim=0)],
        "mean_changed_dimensions_per_changed_frame": float(
            coord_changes.sum(dim=1)[frame_changes].float().mean() if changed_frames else 0.0
        ),
        "exact_token_retention": float((~frame_changes).float().mean()),
        "exact_bigram_retention": float(clean_bigrams.eq(noisy_bigrams).all(dim=1).float().mean())
        if clean_bigrams.numel() else 1.0,
        "self_transition_rate": float(ni[:-1].eq(ni[1:]).float().mean()) if ni.numel() > 1 else 0.0,
    }


def ensure_output_path(root: Path, target: str, condition: str, sigma: float, utterance_id: str) -> Path:
    if condition == "clean":
        folder = root / target / "clean"
    else:
        sigma_name = f"sigma_{int(round(sigma * 100)):03d}"
        folder = root / target / condition / sigma_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{utterance_id}.wav"
