import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_baseline_evaluation_plan.json"
CONTRACT = ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_evaluation_contract.json"
RUNNER = ROOT / "scripts/run_stage2d4_baseline_evaluation.py"


def test_baseline_fixture_groups_are_exact_and_deterministic():
    from scripts.run_stage2d4_baseline_evaluation import build_fixtures

    fixtures = build_fixtures()
    counts = {}
    for fixture in fixtures:
        counts[fixture["group"]] = counts.get(fixture["group"], 0) + 1
    assert counts == {
        "POSITIVE_HELD_OUT_CONTEXT": 7,
        "HUMAN_GOLD_REFERENCE": 3,
        "TARGETED_NATIVE": 10,
        "GENERAL_NATIVE": 100,
        "MECHANISM_REGRESSION": 3,
        "EXTERNAL_HOLDOUT": 5,
    }
    assert len(fixtures) == 128
    assert [x["utterance_id"] for x in fixtures if x["group"] == "HUMAN_GOLD_REFERENCE"] == [
        "IISc_SPICORProject_EN_M_AGRI_3841", "IISc_SPICORProject_EN_M_WEAT_288", "IISc_SPICORProject_EN_M_ENTE_3545"
    ]


def test_baseline_is_evaluation_only_without_optimizer_or_backward():
    source = RUNNER.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "save_stage2d4_checkpoint" not in source
    assert 'if not args.baseline:' in source
    assert '"optimizer_created": False' in source
    assert '"backward_executed": False' in source


def test_baseline_contract_and_comparison_keys_are_frozen():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert plan["checkpoint"]["sha256"] == "2d4bb1896f17f96c38e11ce78bafb0eab060d044fcb2cd0796ba4a93d5e6969a"
    assert plan["runtime"] == {
        "dtype": "float32", "mask_mode": "target_context_1", "deterministic": True,
        "qwen_frozen": True, "optimizer_created": False, "backward_executed": False,
    }
    assert plan["counts"] == {"fixtures": 128, "generation_records": 251}
    assert contract["trajectory_metrics"] == [
        "q0_kl_per_step", "mean_q0_kl", "max_q0_kl", "top1_divergence_count",
        "first_divergent_q0_step", "eos_logit_divergence", "generated_frame_count", "eos_index", "trajectory_class",
    ]
    assert "waveform_sha256" in plan["before_after_comparison_keys"]
    assert "step025_sha256" in plan["before_after_comparison_keys"]


def test_schedule_and_checkpoint_contract_remain_frozen():
    config = json.loads((ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_training_config.json").read_text())
    checkpoint = json.loads((ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_checkpoint_contract.json").read_text())
    assert config["sampling"] == {
        "batch_size": 8, "positive_oversampling": False,
        "policy": "deterministic structured batches; positives are distributed round-robin across epoch batches",
        "epochs": 4, "warmup_optimizer_steps": 5, "estimated_optimizer_steps": 64,
    }
    assert config["optimizer"]["name"] == "AdamW"
    assert config["optimizer"]["gate_learning_rate"] == 1e-3
    assert config["optimizer"]["bridge_learning_rate"] == 1e-4
    assert checkpoint["checkpoint_steps"] == [0, 5, 32, 64]
    assert "Qwen model weights" in checkpoint["excluded"]


def test_dry_run_evidence_is_compact_and_metadata_semantics_are_clear():
    root = ROOT / "artifacts/stage2d/stage2d4_training_implementation/dry_run"
    status = json.loads((root / "dry_run_status.json").read_text())
    assert status["real_graph_dry_run"] is True
    assert status["real_qwen_teacher_forced_call_count"] == 4
    assert status["metadata_clarification"] == {
        "qwen_weights_loaded_from_checkpoint": False,
        "runtime_qwen_loaded": True,
        "meaning": "The historical field refers to Qwen weights not being loaded from the step025 bridge/gate checkpoint; the Colab runtime did instantiate and exercise frozen Qwen.",
    }
    assert json.loads((root / "dry_run_loss_report.json").read_text())["total"] == 3.11146879196167
