"""Validate the M3C clean self-recorded dataset without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from swara.contracts import build_plain_text_request
from swara.frontend import compile_request


EXPECTED_IDS = [f"{number:03d}" for number in range(1, 21)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if [record.get("example_id") for record in records] != EXPECTED_IDS:
        raise ValueError("dataset must contain exactly ordered IDs 001 through 020")
    speakers = {record.get("speaker_id") for record in records}
    if speakers != {"m3_speaker_002"} or info.get("speaker_id") != "m3_speaker_002":
        raise ValueError("M3C dataset must contain exactly its new session speaker ID")
    total_duration = total_frames = 0
    for record in records:
        transcript = record.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"invalid transcript: {record['example_id']}")
        compile_request(build_plain_text_request(transcript, default_language="en-IN", speaker_id="m3_speaker_002"))
        source = Path(record["source_m4a_path"])
        audio_path = root / record["prepared_audio_path"]
        tokens_path = root / record["codec_token_path"]
        if not source.is_file() or not audio_path.is_file() or not tokens_path.is_file():
            raise ValueError(f"missing source, audio, or token file: {record['example_id']}")
        audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if rate != 24000 or audio.shape[1] != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"invalid prepared audio: {record['example_id']}")
        tokens = np.load(tokens_path, allow_pickle=False)
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] != 16:
            raise ValueError(f"invalid codec geometry: {record['example_id']}")
        if not np.issubdtype(tokens.dtype, np.integer) or int(tokens.min()) < 0 or int(tokens.max()) > 2047:
            raise ValueError(f"invalid codec token IDs: {record['example_id']}")
        duration = len(audio) / rate
        if abs(tokens.shape[0] / 12.5 - duration) > 0.25:
            raise ValueError(f"implausible codec-frame duration: {record['example_id']}")
        total_duration += duration
        total_frames += int(tokens.shape[0])
    if total_duration > 300:
        raise ValueError("dataset exceeds five-minute limit")
    print(json.dumps({"dataset_validation": "PASS", "utterance_count": len(records), "speaker_count": len(speakers), "total_duration_seconds": total_duration, "total_codec_frames": total_frames}, sort_keys=True))


if __name__ == "__main__":
    main()
