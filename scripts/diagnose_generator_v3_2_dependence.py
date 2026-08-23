#!/usr/bin/env python3
"""Checkpoint-only v3.1 text/history dependence diagnostics.

This script never performs an optimizer step and never writes model weights.
It requires the v3.1 best checkpoint and the existing debug manifests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from swara.contracts import AudioTokenSpec, build_plain_text_request
from swara.frontend import compile_request
from swara.models.generator_v3_2 import GeneratorV3Config, SwaraSpeechGeneratorV32
from swara.models.linguistic import LinguisticVocabulary

SPEAKER = "ENG_M_SPK001"
PANEL_TRAIN = (
    "IISc_SPICORProject_EN_M_AGRI_1143", "IISc_SPICORProject_EN_M_AGRI_1222",
    "IISc_SPICORProject_EN_M_AGRI_1826", "IISc_SPICORProject_EN_M_AGRI_1832",
    "IISc_SPICORProject_EN_M_AGRI_2140",
)
PANEL_VAL = (
    "IISc_SPICORProject_EN_M_AGRI_116", "IISc_SPICORProject_EN_M_AGRI_256",
    "IISc_SPICORProject_EN_M_AGRI_2592", "IISc_SPICORProject_EN_M_AGRI_4068",
    "IISc_SPICORProject_EN_M_AGRI_7084", "IISc_SPICORProject_EN_M_ENTE_157",
    "IISc_SPICORProject_EN_M_ENTE_191", "IISc_SPICORProject_EN_M_ENTE_560",
    "IISc_SPICORProject_EN_M_ENTE_6952", "IISc_SPICORProject_EN_M_ENTE_7315",
)


def rows(root: Path, name: str):
    return [json.loads(line) for line in (root / "manifests" / f"{name}.jsonl").read_text().splitlines()]


def load_items(root: Path):
    all_rows = rows(root, "debug_30min_train") + rows(root, "debug_30min_val")
    sequences = tuple(compile_request(build_plain_text_request(r["training_text"], default_language="en-IN")) for r in all_rows)
    vocab = LinguisticVocabulary.build(sequences)
    items = {}
    for r, seq in zip(all_rows, sequences):
        tok = torch.from_numpy(np.load(r["codec_token_path"], allow_pickle=False)).long().unsqueeze(0)
        items[r["source_id"]] = (r, seq, tok)
    return items, vocab


def make_model(checkpoint: dict, vocab: LinguisticVocabulary, device):
    spec = AudioTokenSpec("swara.audio.qwen12hz.v0", 16, 2048, 12.5)
    cfg = GeneratorV3Config(vocab.size, 1, spec, model_dim=384, layers=8, heads=6, ffn_dim=1536, max_text_tokens=160, max_audio_frames=256)
    model = SwaraSpeechGeneratorV32(cfg, vocab, (SPEAKER,))
    model.module.load_state_dict(checkpoint["state_dict"])
    model.module.to(device).eval()
    return model


def compare(a, b):
    n = min(a[0].shape[1], b[0].shape[1])
    pa, pb = a[0][:, :n], b[0][:, :n]
    ha, hb = a[2][:, :n], b[2][:, :n]
    qa, qb = torch.softmax(pa, -1), torch.softmax(pb, -1)
    kl = (qa * (qa.clamp_min(1e-8).log() - qb.clamp_min(1e-8).log())).sum(-1).mean()
    changed = (pa.argmax(-1) != pb.argmax(-1)).float().mean()
    l2 = (ha - hb).pow(2).sum(-1).sqrt().mean()
    cosine = 1.0 - torch.nn.functional.cosine_similarity(ha, hb, dim=-1).mean()
    return {"argmax_changed_ratio": float(changed), "mean_kl": float(kl), "hidden_l2": float(l2), "hidden_cosine_distance": float(cosine)}


def aligned_history(source: torch.Tensor, length: int, primary_only: bool = False):
    out = torch.zeros((1, length, 16), dtype=torch.long, device=source.device)
    n = min(length, source.shape[1])
    out[:, :n] = source[:, :n]
    if primary_only:
        out[:, :, 1:] = 0
    return out


def component_norms(model, text_ids, frames, schedule):
    module = model.module
    text = module.text_memory(text_ids)
    n = frames.shape[1]
    trailing = module.text_schedule(text, n, schedule)
    acoustic = []
    positional = []
    for t in range(n):
        if t == 0:
            a = module.control[3].view(1, 1, -1).expand(1, 1, -1)
        else:
            a = sum(module.frame_codebooks[k](frames[:, t-1:t, k]) for k in range(16))
        acoustic.append(a)
        positional.append(module.audio_pos.weight[min(t, module.audio_pos.num_embeddings - 1)].view(1, 1, -1) + module.modality.weight[1].view(1, 1, -1))
    def stats(values):
        x = torch.cat(values, 1)
        return {"mean_l2": float(x.pow(2).sum(-1).sqrt().mean()), "rms": float(x.pow(2).mean().sqrt())}
    return {"acoustic_history": stats(acoustic), "aligned_linguistic": stats([trailing]), "position_plus_modality": stats(positional)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/spicor_eng_m_spk001_v1"))
    ap.add_argument("--output", type=Path, default=Path("diagnostics/generator_v3_1_dependence_diagnostic.json"))
    args = ap.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items, vocab = load_items(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = make_model(checkpoint, vocab, device)
    ids = list(PANEL_TRAIN + PANEL_VAL)
    text_swap, history_swap, ablation, norms = {}, {}, {}, {}
    with torch.no_grad():
        for i, sid in enumerate(ids):
            row, seq, target = items[sid]; target = target.to(device)
            text = model.encode_linguistic(seq); spk = model.speaker_tensor(SPEAKER); schedule = target.shape[1]
            normal = model.module(text, spk, target, schedule_frames=schedule)
            wrong_id = ids[(i + 1) % len(ids)]
            wrong_text = model.encode_linguistic(items[wrong_id][1])
            swapped = model.module(wrong_text, spk, target, schedule_frames=schedule)
            text_swap[sid] = compare(normal, swapped)
            other = items[ids[(i + 1) % len(ids)]][2].to(device)
            swapped_history = model.module(text, spk, aligned_history(other, target.shape[1]), schedule_frames=schedule)
            history_swap[sid] = compare(normal, swapped_history)
            zero_history = torch.zeros_like(target)
            primary_only = aligned_history(target, target.shape[1], primary_only=True)
            ablation[sid] = {"zero_history": compare(normal, model.module(text, spk, zero_history, schedule_frames=schedule)), "primary_only_history": compare(normal, model.module(text, spk, primary_only, schedule_frames=schedule))}
            if len(norms) < 5:
                norms[sid] = component_norms(model, text, target, schedule)
    def avg(table):
        keys = next(iter(table.values())).keys(); return {k: float(np.mean([v[k] for v in table.values()])) for k in keys}
    text_avg, history_avg = avg(text_swap), avg(history_swap)
    result = {
        "status": "complete", "checkpoint": str(args.checkpoint), "device": str(device), "architecture_modified": False,
        "text_swap": {"per_id": text_swap, "mean": text_avg},
        "acoustic_history_swap": {"per_id": history_swap, "mean": history_avg},
        "history_to_text_sensitivity_ratio": history_avg["mean_kl"] / max(text_avg["mean_kl"], 1e-12),
        "history_ablation": ablation, "component_norms": norms,
        "diagnosis": "acoustic-history domination" if history_avg["mean_kl"] > 2 * text_avg["mean_kl"] else "inconclusive",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
