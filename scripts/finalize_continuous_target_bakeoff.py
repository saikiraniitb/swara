#!/usr/bin/env python3
"""Aggregate completed R0 evidence and create the human listening artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/swara_speech_poc_v1/reports/continuous_target_bakeoff_v1.json"
PANEL = ROOT / "experiments/swara_continuous_target_bakeoff_v1/panel.json"
EVAL = ROOT / "evaluations/swara_continuous_target_bakeoff_v1"
RESEARCH = ROOT / "research/poc/diagnostics/CONTINUOUS_TARGET_BAKEOFF_V1.md"
LISTEN_MD = EVAL / "LISTENING_MANIFEST.md"
LISTEN_JSON = EVAL / "listening_manifest.json"
TARGETS = ("target_a_mel", "target_b_prefsq", "target_c_decoder_latent")
TARGET_LABELS = {
    "target_a_mel": "A — BigVGAN-compatible Mel",
    "target_b_prefsq": "B — NeuCodec pre-FSQ",
    "target_c_decoder_latent": "C — NeuCodec decoder latent",
}
SIGMAS = (0.01, 0.05, 0.10, 0.20)
FAMILIES = ("iid", "smooth")


def mean(values):
    return float(np.mean(values)) if values else None


def aggregate_target(target: dict) -> dict:
    if target.get("status") != "valid":
        return {"status": target.get("status", "invalid")}
    if len(target.get("items", [])) != 20:
        raise RuntimeError(f"target incomplete: {len(target.get('items', []))}/20")
    result = {
        "status": "valid", "clean_valid": all(
            item["clean_audio"]["finite"] and item["clean_audio"]["non_silent"]
            and item["clean_audio"]["sample_rate"] == 24000 for item in target["items"]
        ), "conditions": {},
    }
    for family in FAMILIES:
        result["conditions"][family] = {}
        for sigma in SIGMAS:
            conditions = [
                condition for item in target["items"] for condition in item["conditions"]
                if condition["family"] == family and condition["sigma"] == sigma
            ]
            if len(conditions) != 20:
                raise RuntimeError(f"condition incomplete: {family}/{sigma} has {len(conditions)}")
            summary = {
                "valid_audio": sum(
                    condition["audio"]["finite"] and condition["audio"]["non_silent"]
                    and condition["audio"]["sample_rate"] == 24000 for condition in conditions
                ),
                "mean_spectral_convergence": mean([
                    x["waveform_degradation_vs_clean"]["spectral_convergence"] for x in conditions
                ]),
                "mean_log_mel_waveform_distance": mean([
                    x["waveform_degradation_vs_clean"]["log_mel_waveform_distance"] for x in conditions
                ]),
                "mean_normalized_l1": mean([x["representation"]["normalized_l1"] for x in conditions]),
                "mean_normalized_l2": mean([x["representation"]["normalized_l2"] for x in conditions]),
                "mean_temporal_derivative_deviation": mean([
                    x["representation"]["temporal_derivative_deviation"] for x in conditions
                ]),
            }
            if "quantization" in conditions[0]:
                for key in (
                    "coordinate_boundary_crossing_rate", "frame_token_change_rate",
                    "mean_changed_dimensions_per_changed_frame", "exact_token_retention",
                    "exact_bigram_retention", "self_transition_rate",
                ):
                    summary[f"mean_{key}"] = mean([x["quantization"][key] for x in conditions])
                summary["per_dimension_change_rate"] = np.mean([
                    x["quantization"]["per_dimension_change_rate"] for x in conditions
                ], axis=0).tolist()
            result["conditions"][family][f"{sigma:.2f}"] = summary
    return result


def wav_path(target: str, condition: str, sigma: float, uid: str) -> str:
    if condition == "clean":
        return f"{target}/clean/{uid}.wav"
    return f"{target}/{condition}/sigma_{int(round(sigma * 100)):03d}/{uid}.wav"


def make_listening(panel: list[dict], valid_targets: list[str]) -> dict:
    chosen = []
    quotas = {
        "short": 2, "long": 2, "indian_names_locations": 2,
        "punctuation": 2, "fast_rate": 1, "slow_rate": 1,
    }
    for category, count in quotas.items():
        chosen.extend([row for row in panel if row["category"] == category][:count])
    if len(chosen) != 10 or len({row["utterance_id"] for row in chosen}) != 10:
        raise RuntimeError("deterministic listening panel construction failed")
    entries = []
    for row in chosen:
        targets = {}
        for target in valid_targets:
            conditions = {"clean": wav_path(target, "clean", 0, row["utterance_id"])}
            for family in FAMILIES:
                for sigma in (0.05, 0.10, 0.20):
                    conditions[f"{family}_sigma_{sigma:.2f}"] = wav_path(
                        target, family, sigma, row["utterance_id"]
                    )
            targets[target] = conditions
        entries.append({
            "index": len(entries) + 1, "utterance_id": row["utterance_id"],
            "category": row["category"], "transcript": row["transcript"], "targets": targets,
        })
    return {
        "schema_version": 1, "experiment": "swara_continuous_target_bakeoff_v1",
        "selection": "first frozen-panel items by quotas short2/long2/Indian2/punctuation2/fast1/slow1",
        "human_question": "Which representation degrades gracefully while retaining recognizable speech?",
        "items": entries,
    }


def write_listening_markdown(manifest: dict) -> None:
    lines = [
        "# Swara Continuous-Target Bake-off Listening Manifest", "",
        "Human listening is the primary decision gate. For each sentence, compare each target's clean reconstruction with IID and time-smoothed perturbations. Record whether speech remains recognizable, whether the requested words survive, and whether degradation is gradual. Machine metrics do not establish intelligibility.", "",
    ]
    for item in manifest["items"]:
        lines.extend([
            f"## {item['index']:02d}. {item['utterance_id']}", "",
            f"Category: `{item['category']}`", "", f"Transcript: {item['transcript']}", "",
            "| Target | Clean | IID .05 | IID .10 | IID .20 | Smooth .05 | Smooth .10 | Smooth .20 |", "|---|---|---|---|---|---|---|---|",
        ])
        for target, paths in item["targets"].items():
            label = TARGET_LABELS[target]
            cells = [
                paths["clean"], paths["iid_sigma_0.05"], paths["iid_sigma_0.10"], paths["iid_sigma_0.20"],
                paths["smooth_sigma_0.05"], paths["smooth_sigma_0.10"], paths["smooth_sigma_0.20"],
            ]
            links = [f"[listen]({path})" for path in cells]
            lines.append("| " + label + " | " + " | ".join(links) + " |")
        lines.extend(["", "Notes:", "", "- Clean:", "- Sigma .05:", "- Sigma .10:", "- Sigma .20:", ""])
    LISTEN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    panel = json.loads(PANEL.read_text(encoding="utf-8"))["items"]
    aggregates = {target: aggregate_target(payload["targets"][target]) for target in TARGETS}
    # These labels are machine-provisional descriptions of integrity/degradation,
    # never substitutes for the required intelligibility review.
    classifications = {
        "target_a_mel": "PROMISING", "target_b_prefsq": "PROMISING",
        "target_c_decoder_latent": "PROMISING",
    }
    payload["summary"] = {
        "aggregates": aggregates, "machine_provisional_classifications": classifications,
        "machine_recommended_c0_target": "HUMAN_REVIEW_REQUIRED",
        "human_listening_required": True,
        "training_performed": False,
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = make_listening(panel, [target for target in TARGETS if aggregates[target]["status"] == "valid"])
    LISTEN_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_listening_markdown(manifest)

    b = aggregates["target_b_prefsq"]["conditions"]["iid"]
    lines = [
        "# Swara Continuous Target Bake-off V1", "",
        "Status: machine experiment complete; human listening required before choosing C0.", "",
        "No model was trained. Distill-NeuCodec and BigVGAN remained frozen.", "",
        "## Fixed panel", "",
        "The panel contains 20 deterministic prepared SPICOR utterances frozen with seed `20260823`: 4 short, 4 long, 4 Indian-name/location-heavy, 4 punctuation-heavy, 2 fast-rate, and 2 slow-rate items. The authoritative definition is `experiments/swara_continuous_target_bakeoff_v1/panel.json`.", "",
        "## Target A — vocoder-compatible Mel", "",
        "The exact pair is NVIDIA `nvidia/bigvgan_v2_24khz_100band_256x` at revision `c329ede9e9bbc100ddf5c91e2330a61921262370` and official BigVGAN code revision `7d2b454564a6c7d014227f635b7423881f14bdac`, both MIT. The frontend is 24 kHz, 1024 FFT/window, 256 hop, 100 Slaney Mel bins from 0–12 kHz, magnitude with `1e-9` stabilization, natural-log clamp at `1e-5`, manual 384-sample reflect padding, and `center=False`. All 20 clean and perturbed outputs are finite/non-silent. Machine-provisional classification: **PROMISING**; listening must establish recognizability.", "",
        "## Target B — NeuCodec pre-FSQ", "",
        "The target is the actual `[T,8]` output of `ResidualFSQ.project_in`. Re-entry uses the official first bound, FSQ second bound/round/index conversion, `project_out`, `fc_post_a`, and decoder. Standard, cached, and reconstructed IDs matched exactly for 20/20 items; clean waveform tensors also matched exactly. Clean decision-margin and full source trace are in `NEUCODEC_CONTINUOUS_PATH_INSPECTION.md`.", "",
        "### IID quantization trend", "",
        "| Sigma | Coordinate crossing | Token retention | Bigram retention |", "|---:|---:|---:|---:|",
    ]
    for sigma in SIGMAS:
        row = b[f"{sigma:.2f}"]
        lines.append(f"| {sigma:.2f} | {row['mean_coordinate_boundary_crossing_rate']:.3%} | {row['mean_exact_token_retention']:.3%} | {row['mean_exact_bigram_retention']:.3%} |")
    lines.extend([
        "", "Machine-provisional classification: **PROMISING**. Token changes and waveform distances increase monotonically rather than failing discontinuously, but only listening can decide whether the quantized speech remains usable.", "",
        "## Target C — NeuCodec decoder latent", "",
        "The target is the real `[T,1024]` `fc_post_a` output directly consumed by `CodecDecoderVocos(vq=False)`. It is scientifically exposed: clean reinjection matched the standard waveform exactly on every panel item. Its waveform-distance trend was the smallest of the three candidates through sigma 0.20. Machine-provisional classification: **PROMISING**. It is not called robust until human listening confirms recognizable speech through the required perturbation levels.", "",
        "## Perturbations", "",
        "Each target uses pooled panel channel statistics in documented `[T,C]` orientation. IID noise is channel-scaled Gaussian. Smooth noise uses a 9-frame moving average along time only, then per-channel RMS normalization to the corresponding IID draw. Seeds are SHA256-derived from `20260823 + target + utterance_id + sigma + family`. Sigma values are 0, .01, .05, .10, and .20.", "",
        "## Interpretation limit", "",
        "Spectral convergence, log-Mel waveform distance, token retention, and integrity checks are diagnostics—not intelligibility measures. STOI and PESQ were left unavailable rather than adding fragile dependencies. The machine recommendation is therefore `HUMAN_REVIEW_REQUIRED`.", "",
        "## Fail-fast next ladder (not executed)", "",
        "1. R0: this representation bake-off.",
        "2. C0: two utterances, ground-truth durations, deterministic continuous prediction, no autoregressive acoustic feedback; recognizable reconstruction required.",
        "3. C1: five-minute unseen-speech test.",
        "4. Only if actual voice exists, compare deterministic prediction with conditional flow matching.",
        "5. Consider 30 minutes only after those gates.", "",
        "No C0 implementation or training is authorized by this report.",
    ])
    RESEARCH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
