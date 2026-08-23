#!/usr/bin/env python3
"""Bounded Swara v3 SPICOR debug trainer/evaluator."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from swara.contracts import AudioTokenSpec, SpeakerCondition, build_plain_text_request
from swara.frontend import compile_request
from swara.models.generator_v3_3 import GeneratorV3Config, SwaraSpeechGeneratorV33
from swara.models.linguistic import LinguisticVocabulary

SEED = 20250822
SPEAKER = "ENG_M_SPK001"
PANEL_TRAIN_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_1143", "IISc_SPICORProject_EN_M_AGRI_1222",
    "IISc_SPICORProject_EN_M_AGRI_1826", "IISc_SPICORProject_EN_M_AGRI_1832",
    "IISc_SPICORProject_EN_M_AGRI_2140",
)
PANEL_VAL_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_116", "IISc_SPICORProject_EN_M_AGRI_256",
    "IISc_SPICORProject_EN_M_AGRI_2592", "IISc_SPICORProject_EN_M_AGRI_4068",
    "IISc_SPICORProject_EN_M_AGRI_7084", "IISc_SPICORProject_EN_M_ENTE_157",
    "IISc_SPICORProject_EN_M_ENTE_191", "IISc_SPICORProject_EN_M_ENTE_560",
    "IISc_SPICORProject_EN_M_ENTE_6952", "IISc_SPICORProject_EN_M_ENTE_7315",
)


def read_rows(root: Path, name: str):
    return [json.loads(x) for x in (root / "manifests" / f"{name}.jsonl").read_text().splitlines()]


def make_items(rows):
    items = []
    for row in rows:
        seq = compile_request(build_plain_text_request(row["training_text"], default_language="en-IN"))
        tokens = np.load(row["codec_token_path"], allow_pickle=False)
        items.append((row, seq, torch.from_numpy(tokens).unsqueeze(0).long()))
    return items


def metrics(model, items):
    total = primary = residual = pa = ra = 0.0
    with torch.no_grad():
        model.eval()
        for row, seq, target in items:
            out = model.forward(model.encode_linguistic(seq), model.speaker_tensor(SPEAKER), target.to(model.device))
            vals = model.losses(out, target.to(model.device))
            total += float(vals[0]); primary += float(vals[1]); residual += float(vals[2]); pa += float(vals[3]); ra += float(vals[4])
    n = max(1, len(items))
    return {"total_loss": total / n, "primary_loss": primary / n, "residual_loss": residual / n, "primary_accuracy": pa / n, "residual_accuracy": ra / n}


def checkpoint_model(checkpoint, vocab, device):
    spec = AudioTokenSpec("swara.audio.qwen12hz.v0", 16, 2048, 12.5)
    config = GeneratorV3Config(vocab.size, 1, spec, model_dim=384, layers=8, heads=6, ffn_dim=1536, max_text_tokens=160, max_audio_frames=256)
    model = SwaraSpeechGeneratorV33(config, vocab, (SPEAKER,))
    model.module.load_state_dict(checkpoint["state_dict"])
    model.module.to(device).eval()
    return model


def panel_items(train_items, val_items):
    train = {row["source_id"]: item for item in train_items for row in [item[0]]}
    val = {row["source_id"]: item for item in val_items for row in [item[0]]}
    return [train[x] for x in PANEL_TRAIN_IDS], [val[x] for x in PANEL_VAL_IDS]


def primary_similarity(a, b):
    n = min(len(a), len(b))
    return float(sum(x == y for x, y in zip(a[:n], b[:n])) / max(n, 1))


def evaluate_checkpoint(model, train_items, val_items, run):
    train_panel, val_panel = panel_items(train_items, val_items)
    diag = run / "diagnostics"; primary_dir = diag / "primary_tokens"; primary_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    generated_tokens = {}
    targets = {}
    for row, seq, target in train_panel + val_panel:
        out = model.generate(seq, SpeakerCondition("speaker_id", SPEAKER), int(target.shape[1]))
        values = np.asarray([frame[0] for frame in out.frames], dtype=np.int64)
        generated[row["source_id"]] = values; generated_tokens[row["source_id"]] = out; targets[row["source_id"]] = target[0, :, 0].cpu().numpy()
        np.save(primary_dir / f"{row['source_id']}_generated_primary.npy", values)
    report = {"panel_train_ids": list(PANEL_TRAIN_IDS), "panel_validation_ids": list(PANEL_VAL_IDS), "nearest_targets": {}, "text_swap": {}}
    val_max_nonself = 0.0
    for row, _, _ in val_panel:
        sid = row["source_id"]; scores = {other: primary_similarity(generated[sid], targets[other]) for other in PANEL_VAL_IDS if other != sid}
        report["nearest_targets"][sid] = {"self_similarity": primary_similarity(generated[sid], targets[sid]), "max_nonself_similarity": max(scores.values()), "closest_nonself": max(scores, key=scores.get)}
        val_max_nonself = max(val_max_nonself, max(scores.values()))
    for i, (row, _, target) in enumerate(val_panel[:5]):
        wrong = val_panel[(i + 1) % len(val_panel)]
        swapped = model.generate(wrong[1], SpeakerCondition("speaker_id", SPEAKER), int(target.shape[1]))
        n = min(len(generated[row["source_id"]]), len(swapped.frames)); changed = sum(generated[row["source_id"]][j] != swapped.frames[j][0] for j in range(n)) / max(n, 1)
        report["text_swap"][row["source_id"]] = {"wrong_text_id": wrong[0]["source_id"], "changed_primary_ratio": float(changed)}
    report["max_nonself_validation_similarity"] = val_max_nonself
    report["text_swap_gate"] = all(v["changed_primary_ratio"] >= 0.25 for v in report["text_swap"].values())
    report["trajectory_gate"] = val_max_nonself < 0.90
    report["token_gates_pass"] = report["text_swap_gate"] and report["trajectory_gate"]
    report["wav_decoding"] = "not_run"
    codec_path = getattr(evaluate_checkpoint, "codec_path", None)
    if report["token_gates_pass"] and codec_path is not None:
        import soundfile as sf
        from swara.adapters.qwen_codec import Qwen12HzCodecAdapter
        codec = Qwen12HzCodecAdapter.from_local_path(codec_path)
        wav_dir = run / "decoded_samples"; wav_dir.mkdir(exist_ok=True)
        selected = list(PANEL_TRAIN_IDS[:3]) + list(PANEL_VAL_IDS[:5])
        for sid in selected:
            waveform = codec.decode(generated_tokens[sid])
            sf.write(wav_dir / f"{sid}.wav", np.asarray(waveform.samples, dtype=np.float32), waveform.sample_rate_hz, subtype="PCM_16")
        report["wav_decoding"] = "pass"
    (diag / "v3_1_token_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/spicor_eng_m_spk001_v1"))
    ap.add_argument("--run", type=Path, default=Path("runs/generator_v3_3_spicor_30min_v0"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--codec-path", type=Path)
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke_only and not args.evaluate_only:
        raise RuntimeError(
            "The bounded v3 corpus run requires a CUDA/T4 device; local CPU is "
            "supported only for --smoke-only. Run this same script on a T4."
        )
    train_items = make_items(read_rows(args.dataset, "debug_30min_train"))
    val_items = make_items(read_rows(args.dataset, "debug_30min_val"))
    all_sequences = tuple([x[1] for x in train_items + val_items])
    vocab = LinguisticVocabulary.build(all_sequences)
    spec = AudioTokenSpec("swara.audio.qwen12hz.v0", 16, 2048, 12.5)
    config = GeneratorV3Config(vocab.size, 1, spec, model_dim=384, layers=8, heads=6, ffn_dim=1536, max_text_tokens=160, max_audio_frames=256)
    if args.evaluate_only:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required with --evaluate-only")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        vocab = LinguisticVocabulary.from_dict(checkpoint["vocabulary"])
        model = checkpoint_model(checkpoint, vocab, device)
    else:
        model = SwaraSpeechGeneratorV33(config, vocab, (SPEAKER,)); model.module.to(device)
    args.run.mkdir(parents=True, exist_ok=True); (args.run / "diagnostics").mkdir(exist_ok=True)
    vocab.save(args.run / "diagnostics" / "linguistic_vocabulary.json")
    panel = {"seed": SEED, "train_ids": list(PANEL_TRAIN_IDS), "validation_ids": list(PANEL_VAL_IDS)}
    (args.run / "diagnostics" / "evaluation_panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    if args.evaluate_only:
        evaluate_checkpoint.codec_path = args.codec_path
        report = evaluate_checkpoint(model, train_items, val_items, args.run)
        print(json.dumps({"mode": "evaluate-only", "parameter_count": model.parameter_count, "token_gates_pass": report["token_gates_pass"]}, indent=2), flush=True)
        return
    torch.save({"config": config.__dict__ if hasattr(config, "__dict__") else {"model_dim": config.model_dim, "layers": config.layers, "heads": config.heads, "ffn_dim": config.ffn_dim, "vocab_size": vocab.size}, "state_dict": model.module.state_dict(), "vocabulary": vocab.to_dict()}, args.run / "initial.pt")
    smoke = metrics(model, train_items[:1]); print(json.dumps({"device": str(device), "parameter_count": model.parameter_count, "smoke": smoke}, indent=2), flush=True)
    if args.smoke_only:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    best = float("inf"); best_metrics = None; start = time.time(); history = []
    rng = random.Random(SEED); order = list(range(len(train_items))); step = 0
    max_steps = min(args.steps, 1500)
    eval_steps = {1, 250, 500, 750, 1000, 1250, 1500}
    while step < max_steps:
        if not order: order = list(range(len(train_items))); rng.shuffle(order)
        row, seq, target = train_items[order.pop()]; target = target.to(device)
        model.train(); opt.zero_grad(set_to_none=True); vals = model.losses(model.forward(model.encode_linguistic(seq), model.speaker_tensor(SPEAKER), target), target); vals[0].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        if step in eval_steps or step == max_steps:
            tr = metrics(model, train_items[: min(64, len(train_items))]); va = metrics(model, val_items); entry = {"step": step, "train": tr, "validation": va, "gates": model.gate_values}; history.append(entry); print(json.dumps(entry), flush=True)
            if va["total_loss"] < best:
                best = va["total_loss"]; best_metrics = entry; torch.save({"state_dict": model.module.state_dict(), "vocabulary": vocab.to_dict(), "config": {"model_dim": config.model_dim, "layers": config.layers, "heads": config.heads, "ffn_dim": config.ffn_dim, "vocab_size": vocab.size}}, args.run / "best.pt")
    torch.save({"state_dict": model.module.state_dict(), "vocabulary": vocab.to_dict(), "config": {"model_dim": config.model_dim, "layers": config.layers, "heads": config.heads, "ffn_dim": config.ffn_dim, "vocab_size": vocab.size}}, args.run / "final.pt")
    summary = {"seed": SEED, "device": str(device), "parameter_count": model.parameter_count, "steps": step, "wall_clock_seconds": time.time() - start, "history": history, "best": best_metrics, "final_gates": model.gate_values}
    (args.run / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
