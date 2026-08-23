#!/usr/bin/env python3
"""Per-codebook residual diagnostics for a trained Swara v3.2 checkpoint.

Checkpoint-only: no optimizer, no weight writes, no audio decoding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from swara.contracts import AudioTokenSpec, build_plain_text_request
from swara.frontend import compile_request
from swara.models.generator_v3_2 import GeneratorV3Config, SwaraSpeechGeneratorV32
from swara.models.linguistic import LinguisticVocabulary

SPEAKER = "ENG_M_SPK001"
PANEL_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_116", "IISc_SPICORProject_EN_M_AGRI_256",
    "IISc_SPICORProject_EN_M_AGRI_2592", "IISc_SPICORProject_EN_M_AGRI_4068",
    "IISc_SPICORProject_EN_M_AGRI_7084", "IISc_SPICORProject_EN_M_ENTE_157",
    "IISc_SPICORProject_EN_M_ENTE_191", "IISc_SPICORProject_EN_M_ENTE_560",
    "IISc_SPICORProject_EN_M_ENTE_6952", "IISc_SPICORProject_EN_M_ENTE_7315",
)


def load_rows(root: Path):
    rows = []
    for name in ("debug_30min_train", "debug_30min_val"):
        rows.extend(json.loads(x) for x in (root / "manifests" / f"{name}.jsonl").read_text().splitlines())
    return rows


def load_items(root: Path):
    rows = load_rows(root)
    seqs = tuple(compile_request(build_plain_text_request(r["training_text"], default_language="en-IN")) for r in rows)
    vocab = LinguisticVocabulary.build(seqs)
    return {r["source_id"]: (r, s, torch.from_numpy(np.load(r["codec_token_path"], allow_pickle=False)).long().unsqueeze(0)) for r, s in zip(rows, seqs)}, vocab


def model_from_checkpoint(path: Path, vocab: LinguisticVocabulary, device):
    checkpoint = torch.load(path, map_location="cpu")
    spec = AudioTokenSpec("swara.audio.qwen12hz.v0", 16, 2048, 12.5)
    config = GeneratorV3Config(vocab.size, 1, spec, model_dim=384, layers=8, heads=6, ffn_dim=1536, max_text_tokens=160, max_audio_frames=256)
    model = SwaraSpeechGeneratorV32(config, vocab, (SPEAKER,))
    model.module.load_state_dict(checkpoint["state_dict"])
    model.module.to(device).eval()
    return model


def entropy(values):
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts.astype(np.float64) / values.size
    return float(-(p * np.log2(p)).sum())


def diversity(values):
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return {"unique": 0, "entropy_bits": 0.0, "most_frequent_share": 0.0}
    _, counts = np.unique(values, return_counts=True)
    return {"unique": int(len(counts)), "entropy_bits": entropy(values), "most_frequent_share": float(counts.max() / values.size)}


def stage_metrics(logits, target):
    pred = logits.argmax(-1)
    return {
        "accuracy": float((pred == target).float().mean()),
        "cross_entropy": float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))),
        "predicted": diversity(pred.detach().cpu().numpy().reshape(-1)),
        "target": diversity(target.detach().cpu().numpy().reshape(-1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/spicor_eng_m_spk001_v1"))
    ap.add_argument("--output", type=Path, default=Path("diagnostics/generator_v3_2_residual_diagnostic.json"))
    ap.add_argument("--full-validation", action="store_true", help="Use all 45 validation rows; otherwise use the frozen panel.")
    args = ap.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items, vocab = load_items(args.dataset)
    model = model_from_checkpoint(args.checkpoint, vocab, device)
    ids = [r["source_id"] for r in load_rows(args.dataset) if (args.full_validation and r.get("split") == "validation") or (not args.full_validation and r["source_id"] in PANEL_IDS)]
    per = {str(i): {"teacher_forced": [], "free_running": [], "exposure": []} for i in range(1, 16)}
    with torch.no_grad():
        for sid in ids:
            row, seq, target = items[sid]; target = target.to(device)
            text = model.encode_linguistic(seq); speaker = model.speaker_tensor(SPEAKER)
            primary, residual_tf, hidden = model.forward(text, speaker, target, schedule_frames=target.shape[1])
            residual_free = model.module.residual_logits(hidden, target[:, :, 0], targets=None)
            # True-history and generated-history logits are both produced by the
            # same residual module; only within-frame previous residual inputs differ.
            for i in range(15):
                cb = i + 1
                tf = residual_tf[:, :, i]
                fr = residual_free[:, :, i]
                tgt = target[:, :, cb]
                true_stage = stage_metrics(tf, tgt)
                free_stage = stage_metrics(fr, tgt)
                p_true = F.log_softmax(tf, -1); p_gen = F.log_softmax(fr, -1)
                q_true = p_true.exp()
                kl = (q_true * (p_true - p_gen)).sum(-1).mean()
                per[str(cb)]["teacher_forced"].append(true_stage)
                per[str(cb)]["free_running"].append(free_stage)
                per[str(cb)]["exposure"].append({
                    "true_history_accuracy": true_stage["accuracy"],
                    "generated_history_accuracy": free_stage["accuracy"],
                    "accuracy_degradation": true_stage["accuracy"] - free_stage["accuracy"],
                    "kl_true_vs_generated_history": float(kl),
                })
    def mean_metric(entries, key):
        return float(np.mean([x[key] for x in entries])) if entries else 0.0
    summary = {}
    for cb, values in per.items():
        tf, fr, ex = values["teacher_forced"], values["free_running"], values["exposure"]
        summary[cb] = {
            "teacher_forced": {"accuracy": mean_metric(tf, "accuracy"), "cross_entropy": mean_metric(tf, "cross_entropy"), "predicted_unique": mean_metric([x["predicted"] for x in tf], "unique"), "target_unique": mean_metric([x["target"] for x in tf], "unique"), "predicted_entropy_bits": mean_metric([x["predicted"] for x in tf], "entropy_bits"), "target_entropy_bits": mean_metric([x["target"] for x in tf], "entropy_bits")},
            "free_running": {"accuracy": mean_metric(fr, "accuracy"), "generated_unique": mean_metric([x["predicted"] for x in fr], "unique"), "generated_entropy_bits": mean_metric([x["predicted"] for x in fr], "entropy_bits"), "most_frequent_token_share": mean_metric([x["predicted"] for x in fr], "most_frequent_share")},
            "exposure": {"accuracy_degradation": mean_metric(ex, "accuracy_degradation"), "kl_true_vs_generated_history": mean_metric(ex, "kl_true_vs_generated_history")},
        }
    first = None
    for cb in range(1, 16):
        x = summary[str(cb)]
        if x["free_running"]["generated_unique"] <= 4 or x["teacher_forced"]["accuracy"] < 0.1:
            first = cb; break
    result = {"status": "complete", "checkpoint": str(args.checkpoint), "device": str(device), "scope": "full_validation" if args.full_validation else "frozen_panel", "rows": len(ids), "codebooks": summary, "first_collapse_codebook": first, "collapse_before_exposure": bool(first == 1 or (first is not None and summary[str(first)]["teacher_forced"]["predicted_unique"] <= 4)), "shared_residual_cell_and_head": True, "training_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
