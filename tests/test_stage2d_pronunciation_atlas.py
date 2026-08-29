import json
import importlib.util
from pathlib import Path

from swara.diagnostics.pronunciation_atlas import (
    PRONUNCIATION_ALPHABET_V0,
    build_consistency_report,
    build_vocabulary,
    extract_lexical_tokens,
    normalize_lexical_word,
    scan_manifest,
)


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_stage2d_pronunciation_atlas.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("stage2d_atlas_builder", _SCRIPT_PATH)
assert _SCRIPT_SPEC and _SCRIPT_SPEC.loader
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
build_outputs = _SCRIPT_MODULE.build_outputs


def _write_fixture_files(root: Path) -> tuple[Path, Path, Path]:
    manifest = root / "master_inventory.jsonl"
    rows = [
        {"source_id": "u002", "source_text": "Kumar met Kumar."},
        {"source_id": "u001", "source_text": "Dasharatha, Kumar!"},
        {"source_id": "u003", "source_text": "kumar met a friend."},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    curated = root / "curated.json"
    curated.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_text": "Kumar",
                        "variants": [
                            {
                                "variant_id": "Kumar-A",
                                "candidate_ids": ["C1"],
                                "verified_phone_sequence": ["K", "UU", "M", "AA", "R"],
                                "status": "VERIFIED",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fixtures = root / "fixtures.json"
    fixtures.write_text(json.dumps({"transfer": {"Kumar": ["Kumar returned today."]}}), encoding="utf-8")
    return manifest, curated, fixtures


def test_extract_tokens_preserves_unicode_codepoint_spans():
    text = "P. Kumar’s meeting"
    tokens = extract_lexical_tokens(text)
    assert [item["surface_form"] for item in tokens] == ["P", "Kumar’s", "meeting"]
    for item in tokens:
        assert text[item["source_span_start"] : item["source_span_end"]] == item["surface_form"]


def test_normalization_is_conservative():
    assert normalize_lexical_word("KUMAR") == "kumar"
    assert normalize_lexical_word("Kumar’s") == "kumar’s"
    assert normalize_lexical_word("Kumar's") != normalize_lexical_word("Kumars")


def test_scan_and_recurrence_count_are_deterministic(tmp_path):
    manifest, _, _ = _write_fixture_files(tmp_path)
    occurrences = scan_manifest(manifest)
    assert [item.utterance_id for item in occurrences] == ["u001", "u001", "u002", "u002", "u002", "u003", "u003", "u003", "u003"]
    vocabulary = build_vocabulary(occurrences)
    assert vocabulary[0]["normalized_word"] == "kumar"
    assert vocabulary[0]["occurrence_count"] == 4
    assert vocabulary[0]["canonical_phone_candidates"] == []


def test_curated_phone_aggregation_and_consistency_levels(tmp_path):
    manifest, curated, fixtures = _write_fixture_files(tmp_path)
    output = tmp_path / "out"
    build_outputs(manifest, curated, fixtures, output)
    vocabulary = json.loads((output / "vocabulary.json").read_text(encoding="utf-8"))["words"]
    kumar = next(item for item in vocabulary if item["normalized_word"] == "kumar")
    dasharatha = next(item for item in vocabulary if item["normalized_word"] == "dasharatha")
    assert kumar["canonical_phone_candidates"][0]["verified_phone_sequence"] == ["K", "UU", "M", "AA", "R"]
    assert dasharatha["canonical_phone_candidates"] == []
    consistency = json.loads((output / "consistency_report.json").read_text(encoding="utf-8"))["words"]
    assert all(item["acoustic_realization_consistency"] == "UNMEASURED" for item in consistency)


def test_unrepresentable_curated_variant_is_not_reported_consistent():
    rows = [
        {
            "normalized_word": "agrawal",
            "occurrence_count": 2,
            "surface_forms": {"Agrawal": 2},
            "canonical_phone_candidates": [
                {"verification_status": "VERIFIED", "verified_phone_sequence": ["A"]},
                {"verification_status": "UNSUPPORTED_ALPHABET_VARIANT", "verified_phone_sequence": None},
            ],
        }
    ]
    assert build_consistency_report(rows)[0]["curated_consistency"] == "VARIANT_UNREPRESENTABLE"


def test_dasharatha_external_probe_and_extension_schema(tmp_path):
    manifest, curated, fixtures = _write_fixture_files(tmp_path)
    output = tmp_path / "out"
    summary = build_outputs(manifest, curated, fixtures, output)
    assert summary["dasharatha"]["status"] == "IN_SPICOR"
    extensions = json.loads((output / "candidate_phone_extensions.json").read_text(encoding="utf-8"))["proposals"]
    assert {item["symbol"] for item in extensions} >= {"SCHWA", "TH"}
    for item in extensions:
        assert set(item) >= {
            "symbol", "distinction", "why_v0_is_insufficient", "supporting_words",
            "corpus_occurrence_evidence", "curated_evidence", "external_probe_evidence",
            "confidence", "include_in_v1_recommendation",
        }


def test_inventory_constant_is_frozen_and_outputs_are_repeatable(tmp_path):
    manifest, curated, fixtures = _write_fixture_files(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_outputs(manifest, curated, fixtures, first)
    build_outputs(manifest, curated, fixtures, second)
    assert sorted(path.name for path in first.iterdir()) == sorted(path.name for path in second.iterdir())
    assert (first / "occurrence_index.jsonl").read_text() == (second / "occurrence_index.jsonl").read_text()
    assert set(["A", "AA", "E", "EE", "I", "II", "O", "OO", "U", "UU", "AI", "AU"]).issubset(PRONUNCIATION_ALPHABET_V0)
