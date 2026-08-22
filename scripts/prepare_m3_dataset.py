"""Prepare the fixed self-recorded M3A dataset without modifying its sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from swara.adapters.qwen_codec import Qwen12HzCodecAdapter
from swara.contracts import AudioWaveform, build_plain_text_request
from swara.frontend import compile_request


SPEAKER_ID = "m3_speaker_001"
TARGET_SAMPLE_RATE = 24_000
FRAME_RATE_HZ = 12.5
EXPECTED_IDS = tuple(f"{number:03d}" for number in range(1, 21))
ENTRY_PATTERN = re.compile(r"^(?P<id>\d{3})[ \t]+(?P<text>\S.*)$")


def parse_transcripts(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        match = ENTRY_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"invalid transcript entry on line {line_number}")
        example_id, transcript = match.group("id"), match.group("text")
        if example_id in entries:
            raise ValueError(f"duplicate transcript ID: {example_id}")
        entries[example_id] = transcript
    if tuple(entries) != EXPECTED_IDS:
        raise ValueError("sample.txt must contain exactly ordered IDs 001 through 020")
    return entries


def source_clipping_stats(samples: np.ndarray) -> tuple[int, int]:
    saturated = np.abs(samples) >= 0.999969
    count = int(saturated.sum())
    longest = run = 0
    for value in saturated:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return count, longest


def normalize_audio(source: Path, destination: Path) -> dict[str, Any]:
    samples, source_rate = sf.read(source, dtype="float32", always_2d=True)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"invalid source waveform: {source.name}")
    source_peak = float(np.abs(samples).max())
    source_rms = float(np.sqrt(np.mean(np.square(samples))))
    saturated_samples, max_saturated_run = source_clipping_stats(samples.reshape(-1))
    if max_saturated_run > 8:
        raise ValueError(f"sustained full-scale clipping in {source.name}")
    mono = samples.mean(axis=1)
    waveform = torch.from_numpy(mono).unsqueeze(0)
    if source_rate != TARGET_SAMPLE_RATE:
        waveform = audio_functional.resample(waveform, source_rate, TARGET_SAMPLE_RATE)
    output = waveform.squeeze(0).numpy()
    if output.size == 0 or not np.isfinite(output).all():
        raise ValueError(f"normalization produced invalid audio: {source.name}")
    pre_guard_peak = float(np.abs(output).max())
    # Resampling can create a small inter-sample overshoot. Guard only the
    # prepared copy so PCM16 writing cannot introduce clipping; this is not
    # denoising, trimming, or mastering and never changes the source file.
    peak_guard_scale = 1.0
    if pre_guard_peak > 0.999:
        peak_guard_scale = 0.999 / pre_guard_peak
        output = output * peak_guard_scale
    sf.write(destination, output, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return {
        "source_sample_rate_hz": int(source_rate),
        "source_channels": int(samples.shape[1]),
        "source_duration_seconds": len(samples) / source_rate,
        "source_peak": source_peak,
        "source_rms": source_rms,
        "source_saturated_sample_count": saturated_samples,
        "source_max_saturated_run": max_saturated_run,
        "prepared_sample_rate_hz": TARGET_SAMPLE_RATE,
        "prepared_channels": 1,
        "prepared_duration_seconds": len(output) / TARGET_SAMPLE_RATE,
        "prepared_peak": float(np.abs(output).max()),
        "prepared_rms": float(np.sqrt(np.mean(np.square(output)))),
        "prepared_pre_guard_peak": pre_guard_peak,
        "prepared_peak_guard_scale": peak_guard_scale,
    }


def serialize_linguistic(sequence: Any) -> list[dict[str, Any]]:
    return [
        {
            "kind": token.kind.value,
            "value": token.value,
            "language": token.language,
            "source_range": None if token.source_span is None else [token.source_span.start, token.source_span.end],
            "override_id": token.override_id,
        }
        for token in sequence.tokens
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec-path", type=Path, required=True)
    args = parser.parse_args()
    transcripts = parse_transcripts(args.transcripts)
    sources = {example_id: args.source_audio / f"{example_id}.wav" for example_id in EXPECTED_IDS}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing expected source recordings: {missing}")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be absent or empty; refusing to overwrite dataset")
    audio_dir = args.output / "audio"
    encoded_dir = args.output / "encoded"
    audio_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir.mkdir(parents=True, exist_ok=True)
    codec = Qwen12HzCodecAdapter.from_local_path(args.codec_path)
    expected_spec = codec.spec
    if expected_spec.codebook_count != 16 or expected_spec.vocabulary_size != 2048 or expected_spec.frame_rate_hz != FRAME_RATE_HZ:
        raise ValueError(f"unexpected M2A codec geometry: {expected_spec}")

    records: list[dict[str, Any]] = []
    for example_id in EXPECTED_IDS:
        destination = audio_dir / f"{example_id}.wav"
        audio_stats = normalize_audio(sources[example_id], destination)
        prepared, prepared_rate = sf.read(destination, dtype="float32", always_2d=True)
        sequence = compile_request(build_plain_text_request(transcripts[example_id], default_language="en-IN", speaker_id=SPEAKER_ID))
        tokens = codec.encode(AudioWaveform(tuple(float(value) for value in prepared[:, 0]), int(prepared_rate)))
        tokens.validate_against(expected_spec)
        token_array = np.asarray(tokens.frames, dtype=np.int16)
        token_path = encoded_dir / f"{example_id}.npy"
        np.save(token_path, token_array, allow_pickle=False)
        frame_count = int(token_array.shape[0])
        record = {
            "example_id": example_id,
            "transcript": transcripts[example_id],
            "speaker_id": SPEAKER_ID,
            "language": "en-IN",
            "audio_path": f"audio/{example_id}.wav",
            "audio_duration_seconds": audio_stats["prepared_duration_seconds"],
            "codec_token_path": f"encoded/{example_id}.npy",
            "codec_spec_version": expected_spec.version,
            "codec_frame_count": frame_count,
            "codec_codebook_count": expected_spec.codebook_count,
            "codec_min_token_id": int(token_array.min()),
            "codec_max_token_id": int(token_array.max()),
            "codec_duration_seconds": frame_count / expected_spec.frame_rate_hz,
            "linguistic_schema_version": sequence.schema_version,
            "linguistic_token_count": len(sequence.tokens),
            "linguistic_tokens": serialize_linguistic(sequence),
            "pronunciation_overrides_used": False,
            "source_audio_path": str(sources[example_id]),
            "provenance_source": "self-recorded M3A source supplied by project user",
            "selection_reason": "one-to-one authoritative transcript mapping; valid finite mono source and bounded duration",
            "audio_stats": audio_stats,
        }
        records.append(record)

    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    info = {
        "schema_version": "swara.m3-real-speech.v0",
        "speaker_id": SPEAKER_ID,
        "speaker_count": 1,
        "utterance_count": len(records),
        "total_duration_seconds": sum(record["audio_duration_seconds"] for record in records),
        "audio_format": {"sample_rate_hz": TARGET_SAMPLE_RATE, "channels": 1, "encoding": "PCM_16_WAV"},
        "codec": {"spec_version": expected_spec.version, "frame_rate_hz": expected_spec.frame_rate_hz, "codebook_count": expected_spec.codebook_count, "vocabulary_size": expected_spec.vocabulary_size},
        "transcript_source": str(args.transcripts),
        "source_audio_directory": str(args.source_audio),
        "provenance": "Self-recorded audio supplied and authorized by the project user for this M3A experiment.",
        "preprocessing": "Copied into Swara data as mono 24 kHz PCM16 WAV; no denoising, mastering, trimming, or transcript rewriting.",
    }
    (args.output / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"utterance_count": len(records), "total_duration_seconds": info["total_duration_seconds"], "total_codec_frames": sum(record["codec_frame_count"] for record in records)}, sort_keys=True))


if __name__ == "__main__":
    main()
