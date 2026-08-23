"""Deterministically select the 20-clip codec bake-off panel."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/spicor_eng_m_spk001_v1"
OUT = ROOT / "experiments/codec_bakeoff_v1"

def load_rows():
    rows = []
    for p in [DATA / "manifests/full_train.jsonl", DATA / "manifests/full_val.jsonl"]:
        rows.extend(json.loads(x) for x in p.read_text().splitlines())
    return [r for r in rows if not r.get("transcript_empty") and r.get("prepared_audio_path")]

def words(text):
    return re.findall(r"[A-Za-z][A-Za-z'’-]*", text)

def path_for(row):
    p = row.get("prepared_audio_path") or row.get("audio_path")
    if p:
        return ROOT / p
    matches = sorted((DATA / "audio_24k").glob(row["source_id"] + "*.wav"))
    return matches[0] if matches else DATA / "audio_24k" / (row["source_id"] + ".wav")

rows = load_rows()
used = set()
selected = []

def choose(category, predicate, score=lambda r: (len(words(r["training_text"])), r["source_id"]), reverse=True):
    candidates = [r for r in rows if r["source_id"] not in used and predicate(r) and path_for(r).exists()]
    candidates.sort(key=score, reverse=reverse)
    for r in candidates[:5]:
        used.add(r["source_id"])
        selected.append({
            "utterance_id": r["source_id"],
            "source_wav": str(path_for(r).relative_to(ROOT)),
            "source_text": r["training_text"],
            "duration_seconds": r["source_duration_seconds"],
            "category": category,
            "domain": r.get("domain"),
        })

indian = re.compile(r"\b(hyderabad|bengaluru|bangalore|visakhapatnam|thiruvananthapuram|madhapur|rajahmundry|india|indian|delhi|mumbai|chennai|kolkata|telangana|andhra|kerala|karnataka|rupee|crore|lakh)\b", re.I)
choose("A1_ordinary_english", lambda r: r.get("domain") in {"LJSPEECH", "ENTERTAINMENT", "OTHERS"} and not indian.search(r["training_text"]) and 5 <= len(words(r["training_text"])) <= 20, score=lambda r: (len(words(r["training_text"])), r["source_id"]), reverse=False)
# Use a broad deterministic fallback if those domains have fewer than five eligible rows.
if len(selected) < 5:
    for r in sorted([x for x in rows if x["source_id"] not in used and x.get("domain") in {"LJSPEECH", "ENTERTAINMENT", "OTHERS"} and not indian.search(x["training_text"])], key=lambda x: (len(words(x["training_text"])), x["source_id"])):
        if len(selected) >= 5: break
        used.add(r["source_id"]); selected.append({"utterance_id":r["source_id"],"source_wav":str(path_for(r).relative_to(ROOT)),"source_text":r["training_text"],"duration_seconds":r["source_duration_seconds"],"category":"A1_ordinary_english","domain":r.get("domain")})
choose("A2_indian_names_locations", lambda r: bool(indian.search(r["training_text"])), score=lambda r: (len(words(r["training_text"])), r["source_id"]))
choose("A3_pronunciation_challenging", lambda r: len(words(r["training_text"])) >= 28 or any(len(w) >= 15 for w in words(r["training_text"])), score=lambda r: (max(map(len, words(r["training_text"])), default=0), len(words(r["training_text"])), r["source_id"]))
choose("A4_long_prosodic", lambda r: True, score=lambda r: (r["source_duration_seconds"], len(words(r["training_text"])), r["source_id"]))

# The prepared subset may contain only a fraction of the source inventory. Fill any
# remaining slots deterministically from the longest remaining prepared files.
if len(selected) < 20:
    remaining = [r for r in rows if r["source_id"] not in used and path_for(r).exists()]
    remaining.sort(key=lambda r: (r["source_duration_seconds"], r["source_id"]), reverse=True)
    for r in remaining[: 20 - len(selected)]:
        used.add(r["source_id"])
        selected.append({"utterance_id":r["source_id"],"source_wav":str(path_for(r).relative_to(ROOT)),"source_text":r["training_text"],"duration_seconds":r["source_duration_seconds"],"category":"A4_long_prosodic","domain":r.get("domain")})

assert len(selected) == 20, len(selected)
assert len({x["utterance_id"] for x in selected}) == 20
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "original").mkdir(exist_ok=True)
for i, item in enumerate(selected, 1):
    src = ROOT / item["source_wav"]
    dst = OUT / "original" / f"{i:02d}_{item['utterance_id']}.wav"
    shutil.copy2(src, dst)
    item["copied_audio"] = str(dst.relative_to(ROOT))
(OUT / "manifest.json").write_text(json.dumps({"seed": 20260823, "categories": {c:5 for c in ["A1_ordinary_english","A2_indian_names_locations","A3_pronunciation_challenging","A4_long_prosodic"]}, "clips": selected}, indent=2) + "\n")
print(json.dumps(selected, indent=2))
