#!/usr/bin/env python3
"""Stream SPICOR metadata and prepare deterministic experimental subsets."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import struct
import tarfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

SEED = 20250822


def wav_header(data: bytes, member_size: int):
    if len(data) < 44 or data[:4] != b"RIFF":
        raise ValueError("invalid RIFF header")
    pos = 12
    rate = channels = bits = byte_rate = data_size = 0
    while pos + 8 <= len(data):
        tag = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        pos += 8
        if tag == b"fmt " and pos + 16 <= len(data):
            _, channels, rate, byte_rate, _, bits = struct.unpack_from("<HHIIHH", data, pos)
        if tag == b"data":
            data_size = size
            break
        pos += size + (size & 1)
    if not all((rate, channels, bits, byte_rate)):
        raise ValueError("incomplete fmt chunk")
    return rate, channels, bits, (data_size or member_size - 44) / byte_rate


def normalize_text(source: str):
    flags = []
    text = unicodedata.normalize("NFKC", source).strip()
    clean = re.sub(r"\s+", " ", text)
    if clean != source:
        flags.append("whitespace_or_unicode_normalized")
    if not clean:
        flags.append("empty_transcript")
    if re.search(r"[a-z][A-Z]", clean) or re.search(r"[A-Za-z]{18,}", clean):
        flags.append("suspicious_concatenation_or_long_token")
    return clean, flags


def stream_inventory(archive: Path, transcripts: dict):
    rows = []
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".wav"):
                continue
            source_id = Path(member.name).stem
            if source_id not in transcripts:
                raise RuntimeError(f"missing transcript: {source_id}")
            f = tar.extractfile(member)
            try:
                rate, channels, bits, duration = wav_header(f.read(4096), member.size)
                error = None
            except Exception as exc:
                rate = channels = bits = 0
                duration = 0.0
                error = str(exc)
            source = str(transcripts[source_id].get("Transcript", ""))
            clean, flags = normalize_text(source)
            rows.append({"source_id": source_id, "source_wav_member": member.name,
                         "source_text": source, "training_text": clean,
                         "domain": transcripts[source_id].get("Domain", "UNKNOWN"),
                         "source_duration_seconds": round(duration, 6),
                         "source_sample_rate_hz": rate, "source_channels": channels,
                         "source_bit_depth": bits, "source_size_bytes": member.size,
                         "header_error": error, "transcript_empty": not bool(clean),
                         "cleanup_flags": flags})
    if len(rows) != len(transcripts):
        raise RuntimeError(f"audio/transcript mismatch: {len(rows)} vs {len(transcripts)}")
    counts = Counter(r["training_text"] for r in rows if r["training_text"])
    for r in rows:
        text = r["training_text"]
        r["duplicate_text_group"] = hashlib.sha1(text.encode()).hexdigest()[:12] if text and counts[text] > 1 else None
        if r["duplicate_text_group"]:
            r["cleanup_flags"].append("duplicate_text")
    return rows


def split_rows(rows):
    evaluation = [r for r in rows if r["domain"] == "EVALUATION" and not r["transcript_empty"]]
    pool = [r for r in rows if r["domain"] != "EVALUATION" and not r["transcript_empty"] and not r["header_error"]]
    groups = defaultdict(list)
    for r in pool:
        groups[r["duplicate_text_group"] or r["source_id"]].append(r)
    by_domain = defaultdict(list)
    for group in groups.values():
        by_domain[group[0]["domain"]].append(group)
    train, val, test = [], [], []
    rng = random.Random(SEED)
    for domain in sorted(by_domain):
        groups = by_domain[domain]
        rng.shuffle(groups)
        for i, group in enumerate(groups):
            (train if i % 20 < 18 else val if i % 20 == 18 else test).extend(group)
    for name, bucket in (("train", train), ("validation", val), ("test", test), ("evaluation_holdout", evaluation)):
        for r in bucket:
            r["split"] = name
    return train, val, test, evaluation


def choose(rows, target_seconds, tag, include=()):
    selected = list(include)
    ids = {r["source_id"] for r in selected}
    total = sum(r["source_duration_seconds"] for r in selected)
    rng = random.Random(f"{SEED}:{tag}")
    buckets = defaultdict(list)
    for r in rows:
        if r["source_id"] not in ids:
            buckets[r["domain"]].append(r)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    domains = sorted(buckets)
    while total < target_seconds and any(buckets.values()):
        for domain in domains:
            if buckets[domain]:
                r = buckets[domain].pop()
                selected.append(r)
                total += r["source_duration_seconds"]
                if total >= target_seconds:
                    break
    return selected


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in sorted(rows, key=lambda x: x["source_id"]):
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_audio(archive: Path, rows, root: Path):
    wanted = {r["source_wav_member"]: r for r in rows}
    outdir = root / "audio_24k"
    outdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            row = wanted.get(member.name)
            if row is None:
                continue
            data = tar.extractfile(member).read()
            samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if samples.ndim != 1:
                samples = samples.mean(axis=1)
            if rate != 24000:
                samples = scipy.signal.resample_poly(samples, 160, 294)
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            if peak > 0.999:
                samples *= 0.999 / peak
            sf.write(outdir / f"{row['source_id']}.wav", samples, 24000, subtype="PCM_16")
            del wanted[member.name]
            row["prepared_audio_path"] = str((outdir / f"{row['source_id']}.wav").relative_to(root.parent.parent))
            row["prepared_sample_rate_hz"] = 24000
            row["prepared_channels"] = 1
            row["prepared_subtype"] = "PCM_16"
    if wanted:
        raise RuntimeError(f"missing prepared files: {len(wanted)}")


def compile_frontend(rows):
    from swara.contracts import build_plain_text_request
    from swara.frontend import compile_request
    failures = 0
    for r in rows:
        try:
            seq = compile_request(build_plain_text_request(r["training_text"], default_language="en-IN"))
            r["frontend_status"] = "pass"
            r["linguistic_token_count"] = len(seq.tokens)
        except Exception as exc:
            r["frontend_status"] = f"fail:{type(exc).__name__}:{exc}"
            failures += 1
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-audio", action="store_true", help="reuse already prepared subset audio")
    a = ap.parse_args()
    transcripts = json.loads(a.transcripts.read_text())["Transcripts"]
    rows = stream_inventory(a.archive, transcripts)
    train, val, test, evaluation = split_rows(rows)
    debug_train = choose(train, 1800, "debug_train")
    debug_val = choose(val, 300, "debug_val")
    train_2h = choose(train, 7200, "train_2h", debug_train)
    val_2h = choose(val, 720, "val_2h", debug_val)
    test_2h = choose(test, 720, "test_2h")
    prepared = {r["source_id"]: r for r in debug_train + debug_val + train_2h + val_2h + test_2h}
    failures = compile_frontend(list(prepared.values()))
    if a.skip_audio:
        for r in prepared.values():
            r["prepared_audio_path"] = str((a.output / "audio_24k" / f"{r['source_id']}.wav").relative_to(a.output.parent.parent))
            r["prepared_sample_rate_hz"] = 24000
            r["prepared_channels"] = 1
            r["prepared_subtype"] = "PCM_16"
    else:
        prepare_audio(a.archive, list(prepared.values()), a.output)
    man = a.output / "manifests"
    write_jsonl(man / "master_inventory.jsonl", rows)
    for name, bucket in (("full_train", train), ("full_val", val), ("full_test", test), ("evaluation_holdout", evaluation),
                         ("debug_30min_train", debug_train), ("debug_30min_val", debug_val), ("train_2h", train_2h),
                         ("val_2h", val_2h), ("test_2h", test_2h)):
        write_jsonl(man / f"{name}.jsonl", bucket)
    def stats(bucket):
        return {"count": len(bucket), "duration_seconds": round(sum(r["source_duration_seconds"] for r in bucket), 3),
                "domains": dict(Counter(r["domain"] for r in bucket))}
    info = {"schema_version": "swara.spicor.dataset.v1", "seed": SEED,
            "source": {"corpus": "SPICOR TTS 1.0 English Male High-Confidence", "archive": str(a.archive),
                        "archive_size_bytes": a.archive.stat().st_size, "license": "CC-BY-4.0",
                        "speaker_id": "ENG_M_SPK001", "speaker_tag": "Spk0001"},
            "original_records": len(rows), "excluded_empty": sum(r["transcript_empty"] for r in rows),
            "clean_records": len(train) + len(val) + len(test) + len(evaluation),
            "full": {"train": stats(train), "validation": stats(val), "test": stats(test), "evaluation_holdout": stats(evaluation)},
            "subsets": {"debug_30min_train": stats(debug_train), "debug_30min_val": stats(debug_val),
                        "train_2h": stats(train_2h), "val_2h": stats(val_2h), "test_2h": stats(test_2h)},
            "audio": {"observed_source": {"sample_rate_hz": 44100, "channels": 1, "bit_depth": 16},
                      "prepared": {"sample_rate_hz": 24000, "channels": 1, "subtype": "PCM_16"}},
            "frontend": {"language": "en-IN", "selected_records": len(prepared), "failures": failures},
            "codec": {"status": "debug_subset_pending"}}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"full_train": stats(train), "full_val": stats(val), "full_test": stats(test), "evaluation": stats(evaluation),
                      "debug_train": stats(debug_train), "debug_val": stats(debug_val), "train_2h": stats(train_2h),
                      "val_2h": stats(val_2h), "test_2h": stats(test_2h), "prepared": len(prepared), "frontend_failures": failures}, indent=2))


if __name__ == "__main__":
    main()
