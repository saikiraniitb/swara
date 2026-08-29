#!/usr/bin/env python3
"""Run the isolated D2 phoneme lexical-composer ablation."""
from __future__ import annotations
import json, os, subprocess, sys, time, collections, statistics, shutil
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swara_c0_decoder_latent as c0
import run_swara_c1_decoder_latent as c1
from run_continuous_target_bakeoff import load_neucodec
from swara.models.d2_phoneme_ablation import SwaraD2PhonemeModel
from swara.models.phoneme_composer import PhonemeComposerVocabulary
from swara.models.c0_decoder_latent import normalized_decoder_latent_loss

REPORT = ROOT / "experiments/swara_speech_poc_v1/reports/swara_d2_phoneme_ablation.json"
DOC = ROOT / "research/poc/diagnostics/SWARA_D2_PHONEME_ABLATION.md"
EVAL_ROOT = ROOT / "evaluations/swara_d2_phoneme_ablation"
RUN_ROOT = ROOT / "runs/swara_d2_phoneme_ablation_v1"
CHECKPOINT = RUN_ROOT / "best.pt"
SEED = c1.SEED
MAX_STEPS = 500
EVAL_STEPS = (1, 50, 100, 200, 300, 400, 500)
ESPEAK = os.environ.get("ESPEAK_NG", shutil.which("espeak-ng") or "/opt/homebrew/bin/espeak-ng")
ESPEAK_DATA = os.environ.get("ESPEAK_DATA_PATH", "/opt/homebrew/Cellar/espeak-ng/1.52.0/share/espeak-ng-data")

def phonemize(word: str) -> str:
    p = subprocess.run([ESPEAK, "-q", "--ipa=3", "-v", "en-us", "--", word], capture_output=True, text=True, check=True, env={**os.environ, "ESPEAK_DATA_PATH": ESPEAK_DATA})
    return p.stdout.strip()

def coverage(examples):
    values = collections.defaultdict(set)
    for ex in examples:
        for tok in ex.sequence.tokens:
            if tok.kind.value == "grapheme":
                key = tok.value.strip().lower(); values[key].add(tok.value)
    mapping = {}; failures = []
    for key, spellings in sorted(values.items()):
        outputs = {phonemize(s) for s in spellings}
        if len(outputs) != 1 or not next(iter(outputs), ""):
            failures.append({"word": key, "spellings": sorted(spellings), "outputs": sorted(outputs)})
        mapping[key] = next(iter(outputs), "")
    seq_lens = [len([x for x in s if not x.isspace()]) for s in mapping.values() if s]
    train_words = {t.value.strip().lower() for ex in examples if ex.split == "train" for t in ex.sequence.tokens if t.kind.value == "grapheme"}
    val_words = {t.value.strip().lower() for ex in examples if ex.split == "val" for t in ex.sequence.tokens if t.kind.value == "grapheme"}
    inventory = sorted({ch for s in mapping.values() for ch in s if not ch.isspace()})
    audit = {"word_tokens": sum(1 for ex in examples for t in ex.sequence.tokens if t.kind.value == "grapheme"), "unique_words": len(mapping), "train_unique_words": len(train_words), "validation_unique_words": len(val_words), "validation_words_unseen_in_train": sorted(val_words - train_words), "success_count": len(mapping)-len(failures), "failure_count": len(failures), "failures": failures, "empty_outputs": [k for k,v in mapping.items() if not v], "phoneme_vocabulary_size": len(inventory), "phoneme_inventory": inventory, "sequence_length": {"min": min(seq_lens), "median": statistics.median(seq_lens), "p90": statistics.quantiles(seq_lens,n=10)[8], "max": max(seq_lens)}}
    return mapping, audit

def main():
    c0.seed_everything(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = c1.frozen_p2_split(); all_examples = list(train)+list(val)
    mapping, phon_audit = coverage(all_examples)
    if phon_audit["failure_count"] or phon_audit["empty_outputs"]:
        raise RuntimeError("D2_PHONEMIZER_GATE: FAIL")
    vocab = PhonemeComposerVocabulary.from_sequences(tuple(e.sequence for e in all_examples), mapping)
    model = SwaraD2PhonemeModel(vocab, mapping).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    c1_params = 3683968
    diff = total_params-c1_params
    if abs(diff)/c1_params > .05: raise RuntimeError(f"D2 parameter parity outside 5%: {total_params}")
    codec = load_neucodec()
    targets_train = c1.extract_targets(codec, train); targets_val = c1.extract_targets(codec, val)
    stats = np.load(ROOT/"runs/swara_c1_decoder_latent_v1/target_normalization.npz")
    mean=torch.from_numpy(stats["mean"]).to(device); std=torch.from_numpy(stats["std"]).to(device)
    train_target, train_norm, train_pad = c1.build_target_batch(train, targets_train, mean, std, device)
    val_target, val_norm, val_pad = c1.build_target_batch(val, targets_val, mean, std, device)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True); RUN_ROOT.mkdir(parents=True, exist_ok=True)
    # Target-C equivalence and oracle WAVs are inherited from the validated C1 path.
    for ex in all_examples:
        c1.oracle_equivalence(codec, (targets_train if ex.split=="train" else targets_val)[ex.utterance_id]["target"], (targets_train if ex.split=="train" else targets_val)[ex.utterance_id]["standard_ids"])
    model.train(); pred, aligned = model([e.sequence for e in train], [e.alignment_units for e in train], [e.target_total_frames for e in train])
    if not torch.equal(aligned.padding_mask, train_pad): raise RuntimeError("D2 frame mask mismatch")
    losses=normalized_decoder_latent_loss(pred,train_norm,aligned.padding_mask); losses.total.backward(); model.zero_grad(set_to_none=True)
    optimizer = c0.optimizer_for(model); train_ids=tuple(e.utterance_id for e in train); val_ids=tuple(e.utterance_id for e in val)
    best=float("inf"); best_step=0; history=[]; evaluations=[]; started=time.perf_counter(); start_step=0
    recovery = RUN_ROOT / "recovery_latest.pt"
    if os.environ.get("D3_RESUME") == "1" and recovery.exists():
        state = torch.load(recovery, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]); best = float(state["best"]); best_step = int(state["best_step"])
        print(f"D3_RESUME: step={start_step} best_step={best_step}", flush=True)
    for step in range(start_step + 1, MAX_STEPS+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        pred, aligned=model([e.sequence for e in train],[e.alignment_units for e in train],[e.target_total_frames for e in train]); loss=normalized_decoder_latent_loss(pred,train_norm,aligned.padding_mask)
        if not torch.isfinite(loss.total): raise RuntimeError("D2 non-finite loss")
        loss.total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if step in EVAL_STEPS:
            model.eval()
            with torch.inference_mode():
                vp, va=model([e.sequence for e in val],[e.alignment_units for e in val],[e.target_total_frames for e in val])
                vl=normalized_decoder_latent_loss(vp,val_norm,va.padding_mask); vc=c1.masked_pooled_cosine(vp,val_norm,va.padding_mask)
            row={"step":step,"train_loss":float(loss.total.item()),"validation_loss":float(vl.total.item()),"validation_cosine":float(vc)}; history.append(row)
            evaluations.append(row)
            print(f"D2 step={step} train={row['train_loss']:.6f} val={row['validation_loss']:.6f} cos={row['validation_cosine']:.4f}",flush=True)
            if row["validation_loss"] < best:
                best=row["validation_loss"]; best_step=step; torch.save({"schema_version":"swara.d2.phoneme_ablation.v1","step":step,"model":model.state_dict(),"word_to_phonemes":dict(mapping),"train_ids":train_ids,"val_ids":val_ids},CHECKPOINT)
            torch.save({"step":step,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"best":best,"best_step":best_step}, recovery)
    report={"schema_version":"swara.d2.phoneme_ablation.v1","status":"human_listening_required","training_performed":True,"seed":SEED,"phonemizer":{"tool":"eSpeak NG","executable":ESPEAK,"version":"1.52.0","voice":"en-us","command":"espeak-ng -q --ipa=3 -v en-us -- WORD","output":"IPA Unicode symbols; whitespace removed for IDs","data_path":ESPEAK_DATA,"audit":phon_audit},"model":{"total_trainable":total_params,"c1_total":c1_params,"difference":diff,"difference_percent":100*diff/c1_params,"architecture":"C1 downstream unchanged; phoneme embedding 64 + BiGRU 80/direction -> 160-D"},"split":{"train":len(train),"validation":len(val)},"training":{"optimizer":"AdamW","learning_rate":1e-3,"maximum_steps":MAX_STEPS,"wall_seconds":time.perf_counter()-started,"best_step":best_step,"best_validation_loss":best,"evaluations":evaluations},"comparison":{"c1_best_step":100,"c1_report":"experiments/swara_speech_poc_v1/reports/c1_decoder_latent_5min_v1.json","classification":"HUMAN_LISTENING_REQUIRED"},"gt_durations_modified":False,"acoustic_target_modified":False,"commit_push":False}
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    DOC.write_text("# Swara D2 Phoneme Conditioning Ablation\n\nStatus: HUMAN_LISTENING_REQUIRED. See JSON report for frozen split, phonemizer provenance, parity, and evaluation history.\n\nD2 changes only grapheme lexical composition to deterministic eSpeak NG phoneme composition; alignment, GT word durations, Target-C, acoustic predictor, optimizer, and 500-step budget remain frozen.\n")
    print(f"D2_COMPLETE best_step={best_step} params={total_params}")
if __name__ == "__main__": main()
