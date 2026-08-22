"""Prepare the fixed self-recorded M3C AAC/M4A session without source edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import soundfile as sf

from swara.adapters.qwen_codec import Qwen12HzCodecAdapter
from swara.contracts import AudioWaveform, build_plain_text_request
from swara.frontend import compile_request


SPEAKER_ID = "m3_speaker_002"
TARGET_SAMPLE_RATE = 24_000
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


def probe(source: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name,duration:stream=codec_name,codec_type,sample_rate,channels", "-of", "json", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    audio_streams = [stream for stream in report.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(audio_streams) != 1:
        raise ValueError(f"expected exactly one audio stream: {source.name}")
    stream = audio_streams[0]
    duration = float(report["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"empty source audio: {source.name}")
    return {
        "container": report["format"].get("format_name"),
        "codec": stream.get("codec_name"),
        "source_sample_rate_hz": int(stream["sample_rate"]),
        "source_channels": int(stream["channels"]),
        "source_duration_seconds": duration,
    }


def convert(source: Path, destination: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )


def silence_seconds(samples: np.ndarray, sample_rate: int, threshold: float = 0.01) -> tuple[float, float]:
    active = np.abs(samples) > threshold
    if not active.any():
        raise ValueError("prepared clip contains no signal above silence threshold")
    leading = int(np.argmax(active))
    trailing = int(np.argmax(active[::-1]))
    return leading / sample_rate, trailing / sample_rate


def serialize_linguistic(sequence: Any) -> list[dict[str, Any]]:
    return [
        {"kind": token.kind.value, "value": token.value, "language": token.language, "override_id": token.override_id}
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
    sources = {example_id: args.source_audio / f"{example_id}.m4a" for example_id in EXPECTED_IDS}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing expected M4A source recordings: {missing}")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be absent or empty; refusing to overwrite dataset")
    audio_dir = args.output / "audio"
    encoded_dir = args.output / "encoded"
    audio_dir.mkdir(parents=True)
    encoded_dir.mkdir()
    codec = Qwen12HzCodecAdapter.from_local_path(args.codec_path)
    spec = codec.spec
    if (spec.codebook_count, spec.vocabulary_size, spec.frame_rate_hz) != (16, 2048, 12.5):
        raise ValueError(f"unexpected M2A codec geometry: {spec}")

    records: list[dict[str, Any]] = []
    for example_id in EXPECTED_IDS:
        source = sources[example_id]
        source_info = probe(source)
        destination = audio_dir / f"{example_id}.wav"
        convert(source, destination)
        audio, rate = sf.read(destination, dtype="float32", always_2d=True)
        if rate != TARGET_SAMPLE_RATE or audio.shape[1] != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"invalid converted WAV: {example_id}")
        mono = audio[:, 0]
        leading_silence, trailing_silence = silence_seconds(mono, rate)
        sequence = compile_request(build_plain_text_request(transcripts[example_id], default_language="en-IN", speaker_id=SPEAKER_ID))
        tokens = codec.encode(AudioWaveform(tuple(float(value) for value in mono), rate))
        tokens.validate_against(spec)
        array = np.asarray(tokens.frames, dtype=np.int16)
        token_path = encoded_dir / f"{example_id}.npy"
        np.save(token_path, array, allow_pickle=False)
        records.append({
            "example_id": example_id,
            "transcript": transcripts[example_id],
            "speaker_id": SPEAKER_ID,
            "language": "en-IN",
            "source_m4a_path": str(source),
            "prepared_audio_path": f"audio/{example_id}.wav",
            "duration_seconds": len(mono) / rate,
            "source_audio": source_info,
            "prepared_audio": {
                "sample_rate_hz": rate,
                "channels": 1,
                "encoding": "PCM_16_WAV",
                "peak": float(np.abs(mono).max()),
                "leading_silence_seconds": leading_silence,
                "trailing_silence_seconds": trailing_silence,
            },
            "linguistic_token_count": len(sequence.tokens),
            "linguistic_tokens": serialize_linguistic(sequence),
            "pronunciation_overrides_used": False,
            "codec_token_path": f"encoded/{example_id}.npy",
            "codec_frames": int(array.shape[0]),
            "codec_min_token_id": int(array.min()),
            "codec_max_token_id": int(array.max()),
            "codec_spec_version": spec.version,
            "provenance": "PROJECT-OWNED / SELF-RECORDED FOR SWARA; clean M3C recording session supplied by project user",
        })

    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    total_duration = sum(record["duration_seconds"] for record in records)
    info = {
        "schema_version": "swara.m3c-clean-speech.v0",
        "speaker_id": SPEAKER_ID,
        "speaker_count": 1,
        "utterance_count": len(records),
        "total_duration_seconds": total_duration,
        "audio_format": {"sample_rate_hz": TARGET_SAMPLE_RATE, "channels": 1, "encoding": "PCM_16_WAV"},
        "codec": {"spec_version": spec.version, "frame_rate_hz": spec.frame_rate_hz, "codebook_count": spec.codebook_count, "vocabulary_size": spec.vocabulary_size},
        "provenance": "PROJECT-OWNED / SELF-RECORDED FOR SWARA. New clean M4A recording session; raw sources remain outside Git.",
        "preprocessing": "FFmpeg conversion to mono 24 kHz PCM WAV only. No denoising, EQ, compression, trimming, or aggressive normalization.",
    }
    (args.output / "dataset_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    unusual = [record["example_id"] for record in records if record["prepared_audio"]["leading_silence_seconds"] > 0.5]
    print(json.dumps({"utterance_count": len(records), "total_duration_seconds": total_duration, "total_codec_frames": sum(record["codec_frames"] for record in records), "leading_silence_over_500ms": unusual}, sort_keys=True))


if __name__ == "__main__":
    main()
