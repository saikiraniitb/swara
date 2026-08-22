"""Validate a prepared M3A dataset without invoking any model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from swara.contracts import build_plain_text_request
from swara.frontend import compile_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 10 <= len(records) <= 30:
        raise ValueError("dataset must contain 10 through 30 utterances")
    ids = [record.get("example_id") for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example IDs")
    speakers = {record.get("speaker_id") for record in records}
    if speakers != {info.get("speaker_id")} or len(speakers) != 1:
        raise ValueError("dataset must have exactly one consistent speaker ID")
    total_duration = 0.0
    total_frames = 0
    for record in records:
        transcript = record.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"invalid transcript for {record.get('example_id')}")
        compile_request(build_plain_text_request(transcript, default_language=record.get("language"), speaker_id=record["speaker_id"]))
        audio_path = root / record["audio_path"]
        token_path = root / record["codec_token_path"]
        if not audio_path.is_file() or not token_path.is_file():
            raise ValueError(f"missing prepared audio or tokens for {record['example_id']}")
        audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if rate != 24000 or audio.shape[1] != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"invalid prepared audio for {record['example_id']}")
        tokens = np.load(token_path, allow_pickle=False)
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] != 16:
            raise ValueError(f"invalid token geometry for {record['example_id']}")
        if not np.issubdtype(tokens.dtype, np.integer) or int(tokens.min()) < 0 or int(tokens.max()) > 2047:
            raise ValueError(f"token IDs out of range for {record['example_id']}")
        duration = len(audio) / rate
        if abs(tokens.shape[0] / 12.5 - duration) > 0.25:
            raise ValueError(f"codec-frame duration mismatch for {record['example_id']}")
        if int(record["codec_frame_count"]) != tokens.shape[0]:
            raise ValueError(f"manifest frame mismatch for {record['example_id']}")
        total_duration += duration
        total_frames += int(tokens.shape[0])
    if total_duration > 300.0:
        raise ValueError("dataset exceeds the five-minute M3A limit")
    print(json.dumps({"dataset_validation": "PASS", "utterance_count": len(records), "speaker_count": len(speakers), "total_duration_seconds": total_duration, "total_codec_frames": total_frames}, sort_keys=True))


if __name__ == "__main__":
    main()
