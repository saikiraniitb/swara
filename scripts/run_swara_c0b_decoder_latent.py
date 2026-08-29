#!/usr/bin/env python3
"""Run C0b with the fixed clean-oracle AGRI_7084 replacement pair."""

from __future__ import annotations

import sys
from pathlib import Path

# The C0b experiment is intentionally a configuration-only reuse of the
# accepted C0 implementation.  No model, loss, optimizer, or codec path is
# duplicated or changed here.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swara_c0_decoder_latent as c0  # noqa: E402


c0.SELECTED_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_2140",
    "IISc_SPICORProject_EN_M_AGRI_7084",
)
c0.EXPECTED_TRANSCRIPTS = (
    "This isn't the right time to check into the Lemon Tree stock",
    "Among these, trees of Ashok, Kachnaar, Amaltaash, Neem, Australian Babul, Kaner, Sheesham, Sagaun, Mango, Pomegranate, Papaya can also be found",
)
c0.EVAL_ROOT = ROOT / "evaluations/swara_c0b_decoder_latent_v1"
c0.RUN_ROOT = ROOT / "runs/swara_c0b_decoder_latent_v1"
c0.REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/c0b_decoder_latent_v1.json"
c0.RESEARCH_PATH = ROOT / "research/poc/diagnostics/C0B_DECODER_LATENT_V1.md"
c0.STATS_PATH = c0.RUN_ROOT / "target_normalization.npz"
c0.CHECKPOINT_PATH = c0.RUN_ROOT / "best.pt"
c0.EVALUATION_STEPS = (100, 200, 500)
c0.DATA_SPLIT = None


if __name__ == "__main__":
    c0.main()
