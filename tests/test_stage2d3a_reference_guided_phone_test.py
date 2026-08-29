import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_stage2d3a_reference_guided_phone_test import (  # noqa: E402
    classify_trajectory,
    extract_acoustic_trace,
    samples_and_rate,
)


ROOT = Path(__file__).parents[1]
OUT = ROOT / "artifacts/stage2d/stage2d3_reference_guided_phone_test"
V0 = {"A", "AA", "E", "EE", "I", "II", "O", "OO", "U", "UU", "AI", "AU", "K", "G", "T", "D", "N", "P", "B", "M", "Y", "R", "L", "V", "S", "H", "SH", "CH", "J", "NG"}


def read_spec():
    return json.loads((OUT / "stage2d3a_candidate_spec.json").read_text(encoding="utf-8"))


def test_exact_five_words_and_bounded_candidates():
    spec = read_spec()
    assert [x["word"] for x in spec["cases"]] == ["Jamshedpur", "Banerjee", "Chandigarh", "Nagar", "Nagpur"]
    assert all(1 <= len(x["candidates"]) <= 3 for x in spec["cases"])


def test_candidates_are_v0_and_not_canonical():
    spec = read_spec()
    assert all(set(c["phone_sequence"]) <= V0 for case in spec["cases"] for c in case["candidates"])
    assert all(c["canonical"] is False for case in spec["cases"] for c in case["candidates"])


def test_reference_ids_and_native_baselines():
    spec = read_spec()
    lexicon = json.loads((ROOT / "artifacts/stage2d/stage2d2_dataset_design/human_acoustic_reference_lexicon_v0_1.json").read_text())
    refs = {x["word"]: x["reference"]["resolved_utterance_id"] for x in lexicon["entries"]}
    expected = {"Jamshedpur": "IISc_SPICORProject_EN_M_AGRI_3841", "Banerjee": "IISc_SPICORProject_EN_M_WEAT_433", "Chandigarh": "IISc_SPICORProject_EN_M_WEAT_288", "Nagar": "IISc_SPICORProject_EN_M_ENTE_6968", "Nagpur": "IISc_SPICORProject_EN_M_ENTE_3545"}
    assert {x["word"]: x["reference_utterance_id"] for x in spec["cases"]} == expected
    assert all(x["native_baseline_required"] for x in spec["cases"])


def test_frozen_runtime_and_no_production_change():
    spec = read_spec()
    assert spec["runtime"] == {"mask_mode": "target_context_1", "dtype": "float32", "deterministic": True, "qwen_frozen": True, "training": False}
    assert spec["execute_now"] is False


def test_current_native_and_conditioned_trace_wrappers_are_unwrapped():
    trace = SimpleNamespace(generated_frame_count=7, eos_index=6)
    native_wrapper = SimpleNamespace(acoustic_trace=trace)
    conditioned_wrapper = SimpleNamespace(acoustic_trace=trace, active_residual_positions=(3, 4))

    assert extract_acoustic_trace(native_wrapper) is trace
    assert extract_acoustic_trace(conditioned_wrapper) is trace
    assert extract_acoustic_trace(trace) is trace
    assert trace.generated_frame_count == 7
    assert trace.eos_index == 6
    assert list(conditioned_wrapper.active_residual_positions) == [3, 4]


def test_waveform_and_trajectory_helpers_serialize_current_metadata():
    samples, rate = samples_and_rate((SimpleNamespace(samples=torch.tensor([[0.0, 0.25, -0.25]]), sample_rate_hz=24000), 0.1))
    assert samples == [0.0, 0.25, -0.25]
    assert rate == 24000

    normal = SimpleNamespace(generated_frame_count=3, eos_index=2)
    long = SimpleNamespace(generated_frame_count=3, eos_index=None)
    rows = [{"generated_frame_count": normal.generated_frame_count, "eos_index": normal.eos_index, "trajectory_class": classify_trajectory(normal)}]
    encoded = json.dumps({"rows": rows})
    assert json.loads(encoded)["rows"][0] == {
        "generated_frame_count": 3,
        "eos_index": 2,
        "trajectory_class": "NORMAL_TRAJECTORY",
    }
    assert classify_trajectory(long) == "LONG_TRAJECTORY"


def test_trajectory_classifier_separates_long_and_max_length_paths():
    assert classify_trajectory(SimpleNamespace(
        generated_frame_count=100, eos_index=99, max_new_tokens=512,
        max_generation_hit=False, sample_rate_hz=24000, waveform_sample_count=120000,
    )) == "NORMAL_TRAJECTORY"
    assert classify_trajectory(SimpleNamespace(
        generated_frame_count=165, eos_index=164, max_new_tokens=512,
        max_generation_hit=False, sample_rate_hz=12500, waveform_sample_count=165000,
    )) == "LONG_TRAJECTORY"
    assert classify_trajectory(SimpleNamespace(
        generated_frame_count=511, eos_index=None, max_new_tokens=512,
        max_generation_hit=False, sample_rate_hz=12500, waveform_sample_count=511000,
    )) == "MAX_LENGTH_TRAJECTORY"


def test_human_results_and_intervention_policy_preserve_blind_decode():
    results = json.loads((OUT / "stage2d3a_human_results.json").read_text())
    by_word = {item["word"]: item for item in results["entries"]}
    assert by_word["Jamshedpur"]["selected_review_label"] == "A"
    assert by_word["Jamshedpur"]["decoded_condition"] == "jamshedpur_candidate_02"
    assert by_word["Jamshedpur"]["winning_phone_sequence"] == ["J", "A", "M", "SH", "I", "D", "P", "U"]
    assert by_word["Chandigarh"]["winning_phone_sequence"] == ["CH", "A", "N", "D", "I", "G", "AA"]
    assert by_word["Nagpur"]["winning_phone_sequence"] == ["N", "A", "G", "P", "U", "R"]
    assert by_word["Banerjee"]["decoded_condition"] == "native"
    assert by_word["Nagar"]["decoded_condition"] == "native"

    policy = json.loads((OUT / "stage2d3_pronunciation_intervention_policy_v0_1.json").read_text())
    policy_by_word = {item["word"]: item for item in policy["entries"]}
    assert policy_by_word["Banerjee"]["secondary_classification"] == "EXPLICIT_OVERRIDE_UNSAFE"
    assert policy_by_word["Nagar"]["classification"] == "NATIVE_PREFERRED"
    assert policy["universal_pur_rule"] == "NOT_JUSTIFIED"
    assert policy["swara_phones_v1_freeze"] == "DEFERRED"
