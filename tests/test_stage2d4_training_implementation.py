import json
from pathlib import Path

import torch
from torch import nn

from swara.training.stage2d4_training import (
    Stage2D4Dataset,
    build_deterministic_epoch_batches,
    build_mixed_batch,
    classify_trajectory,
    compute_stage2d4_v1_loss,
    compute_trajectory_metrics,
    sampling_exposure,
    save_stage2d4_checkpoint,
    run_graph_dry_run,
    set_trainable_phase,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "artifacts/stage2d/stage2d4_training_design"
INVENTORY = ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
ARCHIVE = Path("/Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz")
CACHE = ROOT / "data/stage2d_spicor_selected_audio"


def dataset():
    return Stage2D4Dataset.from_design(DESIGN, repo_root=ROOT, inventory_path=INVENTORY, archive_path=ARCHIVE, cache_root=CACHE)


def test_exact_v1_counts_and_gold_exclusion():
    value = dataset()
    assert len(value.train_samples) == 124
    assert len(value.positive_train_samples) == 14
    assert len(value.native_train_samples) == 110
    assert not any(sample.human_gold_reference for sample in value.train_samples)
    assert {sample.word if hasattr(sample, "word") else sample.raw.get("word") for sample in value.positive_train_samples} == {"Jamshedpur", "Chandigarh", "Nagpur"}


def test_native_rows_have_no_phone_supervision_and_positive_rows_do():
    value = dataset()
    assert all(sample.phone_sequence is not None and sample.intervention_required for sample in value.positive_train_samples)
    assert all(sample.phone_sequence is None and not sample.intervention_required for sample in value.native_train_samples)


def test_mixed_loss_has_zero_native_ce_and_native_preservation_kl():
    value = dataset()
    positive, native = value.positive_train_samples[0], value.native_train_samples[0]
    mixed = build_mixed_batch((positive, native))
    shape = (2, 4)
    main = torch.randn(*shape, 8, requires_grad=True)
    native_main = torch.randn(*shape, 8)
    residual = torch.randn(2, 4, 3, 8, requires_grad=True)
    native_residual = torch.randn(2, 4, 3, 8)
    codes = torch.randint(0, 8, (2, 4, 4))
    losses = compute_stage2d4_v1_loss(main, native_main, residual, native_residual, codes, mixed, (((1, 3),), ()))
    assert losses.target_ce.item() > 0
    assert losses.preservation_kl.item() >= 0
    native_only = build_mixed_batch((native,))
    native_loss = compute_stage2d4_v1_loss(main[:1], native_main[:1], residual[:1], native_residual[:1], codes[:1], native_only, ((),))
    assert native_loss.target_ce.item() == 0.0
    assert native_loss.preservation_kl.item() >= 0
    native_loss.total.backward()


def test_deterministic_sampler_guarantees_positive_exposure_without_oversampling():
    value = dataset()
    first = build_deterministic_epoch_batches(value.train_samples, batch_size=8, seed=20260829, epoch=0)
    second = build_deterministic_epoch_batches(value.train_samples, batch_size=8, seed=20260829, epoch=0)
    assert [[sample.sample_id for sample in batch] for batch in first] == [[sample.sample_id for sample in batch] for batch in second]
    exposure = sampling_exposure(first)
    assert exposure["examples_per_epoch"]["POSITIVE_INTERVENTION"] == 14
    assert exposure["positive_batches"] == 14
    assert exposure["oversampling"] is False


def test_phase_parameter_contract():
    bridge = nn.Linear(3, 4)
    gate = nn.Parameter(torch.tensor(0.0))
    assert set_trainable_phase(bridge, gate, "gate_warmup") == ("gate",)
    assert all(not parameter.requires_grad for parameter in bridge.parameters())
    names = set_trainable_phase(bridge, gate, "bridge_and_gate")
    assert "gate" in names and "bridge.weight" in names and "bridge.bias" in names
    assert all(parameter.requires_grad for parameter in bridge.parameters())


def test_checkpoint_contract_excludes_qwen(tmp_path):
    bridge = nn.Linear(3, 4)
    gate = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([gate, *bridge.parameters()], lr=1e-3)
    path = tmp_path / "step000.pt"
    save_stage2d4_checkpoint(path, step=0, bridge=bridge, gate=gate, optimizer=optimizer, dataset_sha256="dataset", config={"x": 1}, source_git_commit="dirty", evaluation_contract_sha256="evaluation")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["qwen_weights_included"] is False
    assert payload["evaluation_contract_sha256"] == "evaluation"
    assert "qwen_state_dict" not in payload
    assert "bridge_state_dict" in payload and "gate" in payload


def test_corrected_trajectory_classifier_and_q0_metrics():
    assert classify_trajectory(generated_frame_count=10, duration_seconds=1.0, eos_index=9) == "NORMAL_TRAJECTORY"
    assert classify_trajectory(generated_frame_count=165, duration_seconds=13.2, eos_index=164) == "LONG_TRAJECTORY"
    assert classify_trajectory(generated_frame_count=511, duration_seconds=40.88, eos_index=510, max_new_tokens=512) == "MAX_LENGTH_TRAJECTORY"
    assert classify_trajectory(generated_frame_count=0, duration_seconds=None, eos_index=None, failed=True) == "FAILED"
    native = torch.zeros(1, 3, 5)
    conditioned = native.clone()
    metrics = compute_trajectory_metrics(native, conditioned)
    assert metrics["mean_q0_kl"] == 0.0
    assert metrics["top1_divergence_count"] == 0


def test_graph_dry_run_checks_gradients_and_reverts_disposable_step():
    bridge = nn.Linear(2, 2)
    gate = nn.Parameter(torch.tensor(0.25))
    qwen = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    optimizer = torch.optim.AdamW([gate, *bridge.parameters()], lr=1e-2)
    before = [parameter.detach().clone() for parameter in (gate, *bridge.parameters())]
    result = run_graph_dry_run(
        lambda: (bridge.weight.square().mean() + bridge.bias.square().mean() + gate.square()),
        trainable_parameters=(gate, *bridge.parameters()),
        qwen_parameters=(qwen,),
        optimizer=optimizer,
        perform_disposable_step=True,
    )
    assert result["forward_backward_executed"] is True
    assert result["qwen_gradient_norm"] == 0.0
    assert result["disposable_optimizer_step_reverted"] is True
    for parameter, original in zip((gate, *bridge.parameters()), before):
        assert torch.equal(parameter, original)
