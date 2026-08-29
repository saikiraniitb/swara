#!/usr/bin/env python3
"""Generate source-preserving, non-production phonetic hypotheses for Batch-1."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

WORDS = [
    "nagar", "srinagar", "hyderabad", "bengaluru", "chandigarh", "chhattisgarh",
    "banerjee", "ahmedabad", "jee", "nagpur", "dimapur", "jaipur", "manipur",
    "raipur", "chatterjee", "gorakhpur", "mukherjee", "sambalpur", "aligarh",
    "allahabad", "jamshedpur", "udhampur", "azamgarh", "sultanpur", "bilaspur",
]
VOICE_SOURCES = (("espeak_ng_en_us", "en-us"), ("espeak_ng_en_gb", "en-gb"))
V0 = {"A", "AA", "E", "EE", "I", "II", "O", "OO", "U", "UU", "AI", "AU", "K", "G", "T", "D", "N", "P", "B", "M", "Y", "R", "L", "V", "S", "H", "SH", "CH", "J", "NG"}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ipa_output(executable: str, voice: str, word: str) -> str:
    result = subprocess.run([executable, "-q", "--ipa=3", "-v", voice, "--", word], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def ipa_tokens(raw: str) -> list[str]:
    text = raw.replace("\u200d", "").replace("͡", "").replace(" ", "")
    for mark in ("ˈ", "ˌ"):
        text = text.replace(mark, "")
    compounds = ("tʃ", "dʒ", "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "ɪə", "eə", "ʊə", "iː", "uː", "ɑː", "ɔː", "ɜː", "ɐː")
    tokens: list[str] = []
    i = 0
    while i < len(text):
        match = next((item for item in compounds if text.startswith(item, i)), None)
        if match:
            tokens.append(match)
            i += len(match)
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def map_to_v0(tokens: list[str]) -> dict[str, Any]:
    mapping = {
        "ə": ("A", "schwa approximated as A"), "ɐ": ("A", "central vowel approximated as A"), "ʌ": ("A", "STRUT approximated as A"), "ɚ": ("A", "r-colored schwa loses rhotic vowel detail"), "ɝ": ("AA", "r-colored vowel loses vowel-quality detail"),
        "a": ("A", None), "ɑ": ("AA", None), "ɑː": ("AA", "length/quality may be coarse"), "æ": ("A", "TRAP approximated as A"), "ɛ": ("E", None), "e": ("E", None), "ɪ": ("I", None), "ᵻ": ("I", "reduced vowel approximated as I"), "i": ("I", None), "iː": ("II", "length is represented coarsely"), "ʊ": ("U", None), "ʊə": ("U", "centering diphthong loses its second element"), "u": ("U", None), "uː": ("UU", "length is represented coarsely"), "ɔ": ("O", None), "ɔː": ("OO", "length is represented coarsely"), "ɒ": ("O", None), "o": ("O", None), "ɜ": ("A", "NURSE quality approximated as A"), "ɜː": ("AA", "NURSE quality approximated as AA"), "aɪ": ("AI", "diphthong is represented by coarse AI"), "aʊ": ("AU", "diphthong is represented by coarse AU"), "eɪ": ("AI", "diphthong is approximated as AI"), "oʊ": ("AU", "diphthong is approximated as AU"), "ɔɪ": ("AI", "diphthong is approximated as AI"),
        "p": ("P", None), "b": ("B", None), "t": ("T", None), "d": ("D", None), "k": ("K", None), "g": ("G", None), "ɡ": ("G", None), "m": ("M", None), "n": ("N", None), "ŋ": ("NG", None), "ɹ": ("R", "English rhotic realization approximated as R"), "r": ("R", None), "l": ("L", None), "v": ("V", None), "w": (None, "W is absent from swara-phones-v0"), "j": ("Y", None), "s": ("S", None), "z": ("S", "voicing distinction approximated as S"), "ʃ": ("SH", None), "ʒ": ("SH", "ZH approximated as SH"), "h": ("H", None), "ɾ": ("T", "flap approximated as T"), "tʃ": ("CH", None), "dʒ": ("J", None),
        "θ": (None, "voiceless dental fricative absent from swara-phones-v0"), "ð": (None, "voiced dental fricative absent from swara-phones-v0"), "x": (None, "velar fricative absent from swara-phones-v0"), "ʰ": (None, "aspiration distinction absent from swara-phones-v0"),
    }
    seq: list[str] = []
    unsupported: list[str] = []
    losses: list[str] = []
    for token in tokens:
        value = mapping.get(token)
        if value is None:
            unsupported.append(token)
            losses.append(f"{token} has no documented v0 mapping")
        else:
            symbol, loss = value
            if symbol is None:
                unsupported.append(token)
            else:
                seq.append(symbol)
            if loss:
                losses.append(loss)
    return {"v0_sequence": seq or None, "unsupported_phones": sorted(set(unsupported)), "distinctions_lost": sorted(set(losses)), "mapping_status": "LOSSY" if losses else "LOSSLESS" if seq else "UNREPRESENTABLE"}


def compare(a: list[str], b: list[str]) -> tuple[str, list[dict[str, Any]]]:
    if a == b:
        return "EXACT_AGREEMENT", []
    strip_length = lambda xs: [x.replace("ː", "") for x in xs]
    if strip_length(a) == strip_length(b):
        return "NEAR_AGREEMENT", [{"type": "vowel_length_or_quality", "source_a": a, "source_b": b}]
    diffs = []
    for i in range(max(len(a), len(b))):
        left, right = (a[i] if i < len(a) else None), (b[i] if i < len(b) else None)
        if left != right:
            diffs.append({"position": i, "source_a": left, "source_b": right})
    return "MEANINGFUL_DISAGREEMENT", diffs


def source_inventory(executable: str) -> dict[str, Any]:
    version = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True).stdout.strip().splitlines()[0]
    return {
        "schema_version": "stage2d2-phonetic-source-inventory.v1",
        "sources": [
            {"source_id": sid, "tool": "eSpeak NG", "voice": voice, "executable": executable, "version": version, "independent_system": False, "usable": True, "note": "Distinct eSpeak voice configurations; not independent pronunciation systems."}
            for sid, voice in VOICE_SOURCES
        ] + [
            {"source_id": "cmudict", "tool": "CMUdict", "usable": False, "reason": "Not installed in the local environment."},
            {"source_id": "repository_curated", "tool": "Existing Swara curated evidence", "usable": True, "usable_for_batch1": False, "reason": "No exact trusted curated phone mapping exists for these 25 words."},
        ],
        "analysis_policy": "Source output is preserved verbatim; source configurations are not counted as independent systems.",
    }


def minimal_panel(root: Path, source_rows: dict[str, dict[str, Any]], agreement: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = ["srinagar", "hyderabad", "bengaluru", "chandigarh", "chhattisgarh", "banerjee", "nagpur", "gorakhpur", "jamshedpur", "udhampur"]
    index = read(root / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review/batch1_human_review_index.json")
    by_word = {row["normalized_word"]: row for row in index["words"]}
    questions = {
        "srinagar": "Does the 'nagar' portion sound like the separate Nagar examples, allowing for natural context?",
        "hyderabad": "Do the central vowels sound like 'uh' or another central vowel rather than a clear full vowel?",
        "bengaluru": "Is the final 'luru/luru-like' portion stable, including any r/w-like realization?",
        "chandigarh": "Do you hear any extra breath or a meaningfully different final consonant quality?",
        "chhattisgarh": "Do you hear one stable consonant pattern, or a recurring breathy/aspirated distinction?",
        "banerjee": "Is the initial/middle vowel pattern stable across contexts?",
        "nagpur": "Does the '-pur' portion have a stable vowel and r-like ending?",
        "gorakhpur": "Does the '-pur' portion match the other -pur words, and is the middle kh-like sound stable?",
        "jamshedpur": "Does the '-pur' portion have a stable realization across this context?",
        "udhampur": "Does the '-pur' portion remain stable, and is the dh-like consonant distinct?",
    }
    result = []
    for word in wanted:
        row = by_word[word]
        entry = next((e for e in row["entries"] if e["role"] == "MEDOID"), row["entries"][0])
        result.append({
            "word": row["word"], "normalized_word": word, "human_status": "LIKELY_STABLE",
            "utterance_id": entry["utterance_id"], "role": entry["role"],
            "full_audio_path": entry["full_audio_path"], "competing_hypotheses": source_rows[word]["sources"],
            "source_agreement": agreement[word]["agreement_class"], "plain_language_question": questions[word],
            "phone_assignment": "NOT_REQUESTED; this panel does not ask the reviewer to choose symbols.",
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    executable = shutil.which("espeak-ng")
    if not executable:
        raise SystemExit("INSUFFICIENT_PHONETIC_SOURCES: espeak-ng is unavailable")
    out = root / "artifacts/stage2d/stage2d2_dataset_design/batch1_phonetic_hypotheses"
    source_rows: dict[str, dict[str, Any]] = {}
    normalized_rows = []
    agreement_rows = []
    loss_rows = []
    for word in WORDS:
        sources = {}
        for sid, voice in VOICE_SOURCES:
            raw = ipa_output(executable, voice, word)
            tokens = ipa_tokens(raw)
            sources[sid] = {"raw_output": raw, "analysis_tokens": tokens, "v0_mapping": map_to_v0(tokens)}
        a, b = sources[VOICE_SOURCES[0][0]]["analysis_tokens"], sources[VOICE_SOURCES[1][0]]["analysis_tokens"]
        agreement_class, differences = compare(a, b)
        source_rows[word] = {"word": word.title(), "normalized_word": word, "sources": sources}
        normalized_rows.append({"word": word.title(), "normalized_word": word, "sources": {k: v["analysis_tokens"] for k, v in sources.items()}, "analysis_inventory": "stage2d_phonetic_analysis_inventory_v0"})
        candidate_status = "INVENTORY_GAP" if any(value["v0_mapping"]["unsupported_phones"] for value in sources.values()) else "AMBIGUOUS" if agreement_class == "MEANINGFUL_DISAGREEMENT" else "PLAUSIBLE"
        agreement_rows.append({"word": word.title(), "normalized_word": word, "candidate_source_count": 2, "independent_source_count": 0, "agreement_class": agreement_class, "differences": differences, "stress_only_difference": False, "aggregate_candidate_status": candidate_status, "promotion_status": "NOT_PROMOTED"})
        for sid, value in sources.items():
            loss_rows.append({"word": word.title(), "normalized_word": word, "source_id": sid, **value["v0_mapping"]})

    write(out / "phonetic_source_inventory.json", source_inventory(executable))
    write(out / "batch1_source_pronunciations.json", {
        "schema_version": "stage2d2-batch1-source-pronunciations.v1", "level": "LEVEL_B_PHONETIC_HYPOTHESIS", "production_mapping": False,
        "words": [{"word": source_rows[w]["word"], "normalized_word": w, "sources": {sid: val["raw_output"] for sid, val in source_rows[w]["sources"].items()}, "repository_curated": None, "cmudict": None} for w in WORDS],
    })
    write(out / "stage2d_phonetic_analysis_inventory_v0.json", {
        "schema_version": "stage2d-phonetic-analysis-inventory-v0", "production_inventory": False,
        "units": ["IPA segment/token", "stress removed for comparison", "vowel length retained", "affricates grouped"],
        "normalization_rules": ["remove primary/secondary stress marks", "remove whitespace, zero-width joiners, and tie marks", "group common affricates/diphthongs and long vowels", "retain vowel length and segment identity", "do not map IPA tokens to v0 in this analysis layer"],
        "not_swara_phones_v1": True,
    })
    write(out / "batch1_normalized_hypotheses.json", {"schema_version": "stage2d2-batch1-normalized-hypotheses.v1", "words": normalized_rows})
    agreement = {row["normalized_word"]: row for row in agreement_rows}
    write(out / "batch1_source_agreement.json", {"schema_version": "stage2d2-batch1-source-agreement.v1", "independent_system_count": 0, "rows": agreement_rows, "counts": {k: sum(row["agreement_class"] == k for row in agreement_rows) for k in ("EXACT_AGREEMENT", "NEAR_AGREEMENT", "MEANINGFUL_DISAGREEMENT", "SINGLE_SOURCE_ONLY", "NO_CANDIDATE")}})
    write(out / "batch1_v0_loss_analysis.json", {"schema_version": "stage2d2-batch1-v0-loss-analysis.v1", "inventory": "swara-phones-v0", "production_inventory_modified": False, "rows": loss_rows})

    pressure_specs = [
        ("SCHWA", {"ə", "ɐ", "ʌ", "ɚ", "ɝ", "ɜ", "ɜː"}, "PROMISING_BUT_UNPROVEN", "Batch-1 source hypotheses contain central/r-colored vowel tokens, but no independent system or trusted phone label confirms a new schwa category."),
        ("TH", {"θ", "ð", "ʰ"}, "PROMISING_BUT_UNPROVEN", "No independent source or trusted occurrence-level aspiration label establishes TH; eSpeak spellings such as kh are not treated as phonetic ground truth."),
        ("T_RETROFLEX", set(), "NOT_TESTABLE", "No source output or trusted label provides a retroflex place distinction."),
        ("D_RETROFLEX", set(), "NOT_TESTABLE", "No source output or trusted label provides a retroflex place distinction."),
        ("W", {"w"}, "NOT_TESTABLE", "A W-like token is not a repeated independent finding and v0 has no W symbol."),
    ]
    pressure = []
    for symbol, tokens, status, why in pressure_specs:
        affected = sorted({w for w in WORDS for sid in source_rows[w]["sources"] if tokens.intersection(source_rows[w]["sources"][sid]["analysis_tokens"])})
        pressure.append({"candidate": symbol, "status": status, "affected_words": affected, "affected_word_count": len(affected), "independent_sources": 0, "repeated_spicor_stability": "LIKELY_STABLE_HUMAN_REVIEW_ONLY", "v0_merge_risk": "UNMEASURED", "evidence": why})
    write(out / "batch1_inventory_pressure.json", {"schema_version": "stage2d2-batch1-inventory-pressure.v1", "production_inventory_modified": False, "new_phone_candidates": [], "candidates": pressure})

    family_specs = {"PUR": ["nagpur", "jaipur", "manipur", "raipur", "sambalpur", "udhampur", "sultanpur", "bilaspur"], "NAGAR": ["nagar", "srinagar"], "JEE": ["jee", "banerjee", "chatterjee", "mukherjee"], "GARH": ["chandigarh", "chhattisgarh"], "ABAD": ["hyderabad", "ahmedabad", "allahabad"]}
    family_rows = []
    for name, members in family_specs.items():
        family_rows.append({"family": name, "words": [source_rows[w]["word"] for w in members], "source_observations": {w: {sid: v["analysis_tokens"] for sid, v in source_rows[w]["sources"].items()} for w in members}, "conclusion": "Source hypotheses may be compared as a lexical family, but no production morphological rule or shared Swara suffix sequence is created.", "shared_phone_representation": None, "human_stability": "LIKELY_STABLE", "independent_evidence": False})
    write(out / "batch1_family_phonetic_analysis.json", {"schema_version": "stage2d2-batch1-family-phonetic-analysis.v1", "families": family_rows, "production_rules_created": False})
    panel = minimal_panel(root, source_rows, agreement)
    write(out / "batch1_minimal_phone_review_panel.json", {"schema_version": "stage2d2-batch1-minimal-phone-review-panel.v1", "panel_size": len(panel), "bounded_maximum": 12, "phone_assignment_requested": False, "entries": panel})

    statuses = [row["aggregate_candidate_status"] for row in agreement_rows]
    print(json.dumps({"words": len(WORDS), "independent_sources": 0, "agreement_counts": {k: sum(x == k for x in [r["agreement_class"] for r in agreement_rows]) for k in ("EXACT_AGREEMENT", "NEAR_AGREEMENT", "MEANINGFUL_DISAGREEMENT", "SINGLE_SOURCE_ONLY", "NO_CANDIDATE")}, "candidate_status_counts": {k: statuses.count(k) for k in ("HIGH_CONFIDENCE_CANDIDATE", "PLAUSIBLE", "AMBIGUOUS", "INVENTORY_GAP", "NO_EVIDENCE")}, "panel_size": len(panel), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
