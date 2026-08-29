#!/usr/bin/env python3
"""Run Swara R0 without training any model.

The script is deliberately explicit about the two third-party source roots so
that it cannot silently substitute another vocoder or codec implementation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import types
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from swara.diagnostics.continuous_targets import (
    audio_integrity,
    channel_statistics,
    deterministic_seed,
    ensure_output_path,
    fsq_rounding_margin,
    official_fsq_from_projected,
    perturb_representation,
    quantization_diagnostics,
    representation_distance,
    waveform_spectral_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "experiments/swara_continuous_target_bakeoff_v1/panel.json"
EVAL_ROOT = ROOT / "evaluations/swara_continuous_target_bakeoff_v1"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/continuous_target_bakeoff_v1.json"
NEUCODEC_ID = "neuphonic/distill-neucodec"
NEUCODEC_REVISION = "daee7fd9989a62594084fd8e1a99e61beb5b0e85"
BIGVGAN_ID = "nvidia/bigvgan_v2_24khz_100band_256x"
BIGVGAN_REVISION = "c329ede9e9bbc100ddf5c91e2330a61921262370"
SIGMAS = (0.01, 0.05, 0.10, 0.20)
FAMILIES = ("iid", "smooth")
SMOOTH_WINDOW = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_wave(path: Path, waveform: np.ndarray, sample_rate: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    sf.write(path, wave, sample_rate, subtype="PCM_16")
    result = audio_integrity(wave, sample_rate)
    result["path"] = str(path.relative_to(ROOT))
    result["bytes"] = path.stat().st_size
    return result


def stats_json(stats) -> dict:
    aggregate = lambda array: {
        "channel_mean": float(np.mean(array)), "channel_median": float(np.median(array)),
        "channel_min": float(np.min(array)), "channel_max": float(np.max(array)),
    }
    return {
        "orientation": "time_major_[T,C]", "frames": stats.frame_count,
        "channels": stats.channels, "mean": aggregate(stats.mean), "std": aggregate(stats.std),
        "min": aggregate(stats.minimum), "max": aggregate(stats.maximum),
        "p01": aggregate(stats.p01), "p99": aggregate(stats.p99),
        "per_channel": {
            "mean": stats.mean.tolist(), "std": stats.std.tolist(),
            "min": stats.minimum.tolist(), "max": stats.maximum.tolist(),
            "p01": stats.p01.tolist(), "p99": stats.p99.tolist(),
        },
    }


def load_panel() -> list[dict]:
    payload = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    rows = payload["items"]
    if len(rows) != 20 or payload["seed"] != 20260823:
        raise RuntimeError("frozen panel contract mismatch")
    for row in rows:
        for key in ("source_wav", "cached_codec_token_path"):
            if not (ROOT / row[key]).is_file():
                raise FileNotFoundError(ROOT / row[key])
    return rows


def load_neucodec():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # NeuCodec needs only this position-embedding class.  Loading torchtune's
    # package root imports optional torchao GPU extensions that are unavailable
    # on this CPU Mac, so use the same narrow import shim as the accepted P1/P2
    # codec-decode path.
    source = Path(torch.__file__).parent.parent / "torchtune/modules/position_embeddings.py"
    if not source.exists():
        source = ROOT / ".venv/lib/python3.14/site-packages/torchtune/modules/position_embeddings.py"
    text = source.read_text(encoding="utf-8")
    namespace = {"torch": torch, "nn": torch.nn, "Any": object, "Optional": object}
    exec(text[text.index("class RotaryPositionalEmbeddings"):], namespace)
    module = types.ModuleType("torchtune.modules")
    module.RotaryPositionalEmbeddings = namespace["RotaryPositionalEmbeddings"]
    sys.modules["torchtune.modules"] = module
    from neucodec import DistillNeuCodec

    model = DistillNeuCodec.from_pretrained(
        NEUCODEC_ID, revision=NEUCODEC_REVISION, local_files_only=True, map_location="cpu"
    )
    return model.eval()


@torch.inference_mode()
def extract_neucodec(model, audio_path: Path) -> dict:
    """Source-equivalent DistillNeuCodec encode with confirmed boundaries exposed."""
    # torchaudio 2.13 delegates file I/O to optional TorchCodec.  Read losslessly
    # with SoundFile, then retain NeuCodec's official torchaudio resampler and
    # private padding routine; this changes no signal-processing semantics.
    wave, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(wave.mean(axis=1))[None, None, :]
    if sample_rate != 16_000:
        mono = Resample(sample_rate, 16_000)(mono)
    y = model._prepare_audio(mono)
    semantic_inputs = []
    for index in range(y.size(0)):
        values = model.feature_extractor(
            F.pad(y[index, :].cpu(), (160, 160)), sampling_rate=16_000, return_tensors="pt"
        ).input_values.to(model.device).squeeze(0)
        semantic_inputs.append(values)
    semantic_features = torch.vstack(semantic_inputs)
    acoustic = model.fc_sq_prior(model.codec_encoder(y.to(model.device))).transpose(1, 2)
    semantic = model.semantic_model(semantic_features).last_hidden_state.transpose(1, 2)
    semantic = model.SemanticEncoder_module(semantic)
    if acoustic.shape[-1] != semantic.shape[-1]:
        length = min(acoustic.shape[-1], semantic.shape[-1])
        acoustic, semantic = acoustic[..., :length], semantic[..., :length]
    generator_input = model.fc_prior(torch.cat([semantic, acoustic], dim=1).transpose(1, 2)).transpose(1, 2)
    time_major_2048 = generator_input.transpose(1, 2)
    quantizer = model.generator.quantizer
    standard_embedding, standard_indices = quantizer(time_major_2048)
    projected = quantizer.project_in(time_major_2048)
    rebuilt_embedding, rebuilt_indices, coordinates = official_fsq_from_projected(quantizer, projected)
    decoder_embedding = quantizer.get_output_from_indices(standard_indices)
    decoder_latent = model.fc_post_a(decoder_embedding)
    return {
        "projected": projected[0].cpu(),
        "coordinates": coordinates[0].cpu(),
        "standard_indices": standard_indices[0, :, 0].cpu(),
        "rebuilt_indices": rebuilt_indices[0].cpu(),
        "standard_embedding": standard_embedding[0].cpu(),
        "rebuilt_embedding": rebuilt_embedding[0].cpu(),
        "decoder_embedding": decoder_embedding[0].cpu(),
        "decoder_latent": decoder_latent[0].cpu(),
        "rounding_margin": fsq_rounding_margin(quantizer, projected)[0].cpu(),
    }


@torch.inference_mode()
def decode_neucodec_indices(model, indices: torch.Tensor) -> np.ndarray:
    codes = indices.long()[None, None, :].to(model.device)
    return model.decode_code(codes)[0, 0].cpu().numpy()


@torch.inference_mode()
def decode_neucodec_projected(model, projected_tc: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    projected = torch.as_tensor(projected_tc, dtype=torch.float32, device=model.device)[None]
    embedding, indices, coordinates = official_fsq_from_projected(model.generator.quantizer, projected)
    latent = model.fc_post_a(embedding)
    waveform = model.generator(latent, vq=False)[0][0, 0]
    return waveform.cpu().numpy(), indices[0].cpu(), coordinates[0].cpu()


@torch.inference_mode()
def decode_neucodec_latent(model, latent_tc: np.ndarray) -> np.ndarray:
    latent = torch.as_tensor(latent_tc, dtype=torch.float32, device=model.device)[None]
    return model.generator(latent, vq=False)[0][0, 0].cpu().numpy()


def run_neucodec(rows: list[dict], report: dict) -> None:
    started = time.time()
    model = load_neucodec()
    extracted = {}
    for index, row in enumerate(rows, 1):
        print(f"NeuCodec extract {index}/20 {row['utterance_id']}", flush=True)
        data = extract_neucodec(model, ROOT / row["source_wav"])
        cached = torch.from_numpy(np.load(ROOT / row["cached_codec_token_path"])).long().reshape(-1)
        standard = data["standard_indices"].reshape(-1)
        if not torch.equal(standard, data["rebuilt_indices"].reshape(-1)):
            raise RuntimeError(f"Target-B exact interception failed for {row['utterance_id']}")
        if not torch.equal(standard, cached):
            raise RuntimeError(f"fresh/cached codec IDs differ for {row['utterance_id']}")
        if not torch.equal(data["standard_embedding"], data["rebuilt_embedding"]):
            raise RuntimeError(f"Target-B projected embedding differs for {row['utterance_id']}")
        extracted[row["utterance_id"]] = data

    stats_b = channel_statistics([x["projected"].numpy() for x in extracted.values()])
    stats_c = channel_statistics([x["decoder_latent"].numpy() for x in extracted.values()])
    all_margins = torch.cat([x["rounding_margin"].reshape(-1) for x in extracted.values()]).numpy()
    target_b = {
        "status": "valid", "clean_exact_id_equivalence": True,
        "representation": {"orientation": "[T,8]", "boundary": "ResidualFSQ.project_in output"},
        "statistics": stats_json(stats_b),
        "decision_margin_rounding_domain": {
            "formula": "min(abs(bound(bound(projected)) - {-1.5,-0.5,0.5}))",
            "mean": float(np.mean(all_margins)), "median": float(np.median(all_margins)),
            "p10": float(np.percentile(all_margins, 10)), "p90": float(np.percentile(all_margins, 90)),
        }, "items": [],
    }
    target_c = {
        "status": "valid", "representation": {
            "orientation": "[T,1024]", "boundary": "fc_post_a output consumed by CodecDecoderVocos(vq=False)"
        }, "statistics": stats_json(stats_c), "items": [],
    }
    c_clean_max, c_clean_mean = [], []
    b_clean_max, b_clean_mean = [], []
    for index, row in enumerate(rows, 1):
        uid, data = row["utterance_id"], extracted[row["utterance_id"]]
        print(f"NeuCodec decode {index}/20 {uid}", flush=True)
        standard_wave = decode_neucodec_indices(model, data["standard_indices"])
        b_wave, b_ids, b_coords = decode_neucodec_projected(model, data["projected"].numpy())
        c_wave = decode_neucodec_latent(model, data["decoder_latent"].numpy())
        for actual, collector_max, collector_mean, name in (
            (b_wave, b_clean_max, b_clean_mean, "B"), (c_wave, c_clean_max, c_clean_mean, "C")
        ):
            if actual.shape != standard_wave.shape:
                raise RuntimeError(f"Target-{name} clean waveform shape mismatch for {uid}")
            collector_max.append(float(np.max(np.abs(actual - standard_wave))))
            collector_mean.append(float(np.mean(np.abs(actual - standard_wave))))
        b_clean_info = save_wave(
            ensure_output_path(EVAL_ROOT, "target_b_prefsq", "clean", 0, uid), b_wave, 24_000
        )
        c_clean_info = save_wave(
            ensure_output_path(EVAL_ROOT, "target_c_decoder_latent", "clean", 0, uid), c_wave, 24_000
        )
        item_b = {"utterance_id": uid, "frames": int(data["projected"].shape[0]), "clean_audio": b_clean_info, "conditions": []}
        item_c = {"utterance_id": uid, "frames": int(data["decoder_latent"].shape[0]), "clean_audio": c_clean_info, "conditions": []}
        for family in FAMILIES:
            for sigma in SIGMAS:
                seed_b = deterministic_seed("target_b_prefsq", uid, sigma, family)
                noisy_b = perturb_representation(data["projected"].numpy(), stats_b.std, sigma=sigma, seed=seed_b, family=family, smooth_window=SMOOTH_WINDOW)
                b_noisy_wave, noisy_ids, noisy_coords = decode_neucodec_projected(model, noisy_b)
                b_condition = {
                    "family": family, "sigma": sigma, "seed": seed_b,
                    "representation": representation_distance(data["projected"].numpy(), noisy_b),
                    "quantization": quantization_diagnostics(data["standard_indices"], noisy_ids, data["coordinates"], noisy_coords),
                    "audio": save_wave(ensure_output_path(EVAL_ROOT, "target_b_prefsq", family, sigma, uid), b_noisy_wave, 24_000),
                    "waveform_degradation_vs_clean": waveform_spectral_metrics(b_wave, b_noisy_wave, 24_000),
                }
                item_b["conditions"].append(b_condition)
                seed_c = deterministic_seed("target_c_decoder_latent", uid, sigma, family)
                noisy_c = perturb_representation(data["decoder_latent"].numpy(), stats_c.std, sigma=sigma, seed=seed_c, family=family, smooth_window=SMOOTH_WINDOW)
                c_noisy_wave = decode_neucodec_latent(model, noisy_c)
                c_condition = {
                    "family": family, "sigma": sigma, "seed": seed_c,
                    "representation": representation_distance(data["decoder_latent"].numpy(), noisy_c),
                    "audio": save_wave(ensure_output_path(EVAL_ROOT, "target_c_decoder_latent", family, sigma, uid), c_noisy_wave, 24_000),
                    "waveform_degradation_vs_clean": waveform_spectral_metrics(c_wave, c_noisy_wave, 24_000),
                }
                item_c["conditions"].append(c_condition)
        target_b["items"].append(item_b)
        target_c["items"].append(item_c)
        report["targets"]["target_b_prefsq"] = target_b
        report["targets"]["target_c_decoder_latent"] = target_c
        write_json(REPORT_PATH, report)
    target_b["clean_waveform_equivalence"] = {"max_absolute_difference": max(b_clean_max), "mean_absolute_difference": float(np.mean(b_clean_mean))}
    target_c["clean_waveform_equivalence"] = {"max_absolute_difference": max(c_clean_max), "mean_absolute_difference": float(np.mean(c_clean_mean))}
    target_b["wall_seconds"] = time.time() - started
    target_c["wall_seconds_in_shared_run"] = time.time() - started
    report["targets"]["target_b_prefsq"] = target_b
    report["targets"]["target_c_decoder_latent"] = target_c
    write_json(REPORT_PATH, report)
    del model, extracted
    gc.collect()


def load_bigvgan(source: Path, snapshot: Path):
    sys.path.insert(0, str(source))
    import bigvgan
    from meldataset import get_mel_spectrogram

    model = bigvgan.BigVGAN.from_pretrained(str(snapshot), use_cuda_kernel=False)
    model.remove_weight_norm()
    return model.eval(), get_mel_spectrogram


def run_bigvgan(rows: list[dict], report: dict, source: Path, snapshot: Path) -> None:
    started = time.time()
    model, get_mel = load_bigvgan(source, snapshot)
    mels = {}
    for index, row in enumerate(rows, 1):
        print(f"BigVGAN Mel {index}/20 {row['utterance_id']}", flush=True)
        wave, _ = librosa.load(ROOT / row["source_wav"], sr=24_000, mono=True)
        with torch.inference_mode():
            mel = get_mel(torch.from_numpy(wave).float()[None], model.h)[0].transpose(0, 1).cpu().numpy()
        mels[row["utterance_id"]] = mel
    stats = channel_statistics(mels.values())
    fresh_target = {
        "status": "valid", "model": BIGVGAN_ID, "revision": BIGVGAN_REVISION,
        "license": "MIT", "official_code_revision": "7d2b454564a6c7d014227f635b7423881f14bdac",
        "checkpoint_sha256": sha256(snapshot / "bigvgan_generator.pt"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mel_frontend": {
            "sample_rate": 24000, "n_fft": 1024, "win_length": 1024, "hop_length": 256,
            "n_mels": 100, "f_min": 0, "f_max": 12000, "mel_scale": "Slaney (librosa default)",
            "mel_normalization": "Slaney (librosa default)", "magnitude": "sqrt(real^2+imag^2+1e-9)",
            "log": "natural log after clamp(min=1e-5)", "center": False,
            "padding": "manual reflect, 384 samples each side", "stft_normalized": False,
        }, "statistics": stats_json(stats), "items": [],
    }
    target = report.get("targets", {}).get("target_a_mel", fresh_target)
    completed_ids = {item["utterance_id"] for item in target.get("items", [])}
    for index, row in enumerate(rows, 1):
        uid, clean = row["utterance_id"], mels[row["utterance_id"]]
        if uid in completed_ids:
            print(f"BigVGAN decode {index}/20 {uid} (persisted; skip)", flush=True)
            continue
        print(f"BigVGAN decode {index}/20 {uid}", flush=True)
        definitions = [("clean", 0.0, None, clean)]
        for family in FAMILIES:
            for sigma in SIGMAS:
                seed = deterministic_seed("target_a_mel", uid, sigma, family)
                noisy = perturb_representation(
                    clean, stats.std, sigma=sigma, seed=seed, family=family,
                    smooth_window=SMOOTH_WINDOW,
                )
                definitions.append((family, sigma, seed, noisy))
        decoded = []
        # A one-condition batch is faster for BigVGAN's grouped anti-aliasing
        # convolutions on this CPU runtime and keeps memory use predictable.
        with torch.inference_mode():
            for offset in range(0, len(definitions), 1):
                batch = np.stack([entry[3].T for entry in definitions[offset:offset + 1]])
                decoded.extend(model(torch.from_numpy(batch).float())[:, 0].cpu().numpy())
        clean_wave = decoded[0]
        clean_audio = save_wave(ensure_output_path(EVAL_ROOT, "target_a_mel", "clean", 0, uid), clean_wave, 24_000)
        if not clean_audio["finite"] or not clean_audio["non_silent"]:
            raise RuntimeError(f"Target-A clean baseline invalid for {uid}")
        item = {"utterance_id": uid, "frames": int(clean.shape[0]), "clean_audio": clean_audio, "conditions": []}
        for (family, sigma, seed, noisy), noisy_wave in zip(definitions[1:], decoded[1:]):
            item["conditions"].append({
                    "family": family, "sigma": sigma, "seed": seed,
                    "representation": representation_distance(clean, noisy),
                    "audio": save_wave(ensure_output_path(EVAL_ROOT, "target_a_mel", family, sigma, uid), noisy_wave, 24_000),
                    "waveform_degradation_vs_clean": waveform_spectral_metrics(clean_wave, noisy_wave, 24_000),
            })
        target["items"].append(item)
        report["targets"]["target_a_mel"] = target
        write_json(REPORT_PATH, report)
    target["wall_seconds"] = time.time() - started
    report["targets"]["target_a_mel"] = target
    write_json(REPORT_PATH, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bigvgan-source", type=Path, required=True)
    parser.add_argument("--bigvgan-snapshot", type=Path, required=True)
    parser.add_argument("--only", choices=("neucodec", "mel", "all"), default="all")
    args = parser.parse_args()
    rows = load_panel()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "panel.json").write_bytes(PANEL_PATH.read_bytes())
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {
        "schema_version": 1, "experiment": "swara_continuous_target_bakeoff_v1",
        "training_performed": False, "panel_path": str(PANEL_PATH.relative_to(ROOT)),
        "panel_sha256": sha256(PANEL_PATH), "panel_size": len(rows), "seed": 20260823,
        "perturbations": {"sigmas": list(SIGMAS), "families": list(FAMILIES), "smooth_window_frames": SMOOTH_WINDOW},
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "neucodec": importlib.metadata.version("neucodec")},
        "codec": {"model": NEUCODEC_ID, "revision": NEUCODEC_REVISION, "license": "Apache-2.0", "modified": False},
        "targets": {},
    }
    if args.only in ("neucodec", "all"):
        run_neucodec(rows, report)
    if args.only in ("mel", "all"):
        run_bigvgan(rows, report, args.bigvgan_source, args.bigvgan_snapshot)
    report["completed"] = True
    write_json(REPORT_PATH, report)


if __name__ == "__main__":
    main()
