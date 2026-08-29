"""Run the read-only Stage2B.3C local-Qwen trajectory panel.

This script only observes a locally loaded Qwen model through the Swara-owned
Stage2B adapter. It does not save model state, audio files, or hidden tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_stage2b import QwenStage2BAdapter
from swara.adapters.qwen_tts import QwenFoundationTTS
from swara.frontend import Frontend
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation


MODEL_PATH = ROOT / "models" / "qwen3-tts-12hz-0.6b-base"
REFERENCE_AUDIO = ROOT / "data" / "spicor_eng_m_spk001_v1" / "audio_24k" / "IISc_SPICORProject_EN_M_AGRI_116.wav"
PANEL = (
    "The meeting begins tomorrow.",
    "Kolkata hosted the conference.",
    "Ajinkya travelled to Bengaluru.",
    "Wait... Really?!",
    "Ravi met Ravi after the meeting.",
)


def representation(text: str, overrides=()):
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("trajectory-panel-speaker"),
        pronunciation=PronunciationInput(overrides=tuple(overrides)),
    )
    return build_stage2b_representation(Frontend().compile(request))


def batch_for(rep):
    tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,))
    tensorizer.eval()
    for parameter in tensorizer.parameters():
        parameter.requires_grad_(False)
    return tensorizer((rep,))


def compare(native, integrated):
    nt = native.acoustic_trace
    it = integrated.acoustic_trace
    token_equal = torch.equal(nt.acoustic_tokens, it.acoustic_tokens)
    codec_equal = torch.equal(nt.codec_input_tokens, it.codec_input_tokens)
    waveform_delta = (nt.waveform - it.waveform).abs() if nt.waveform is not None and it.waveform is not None else None
    return {
        "shape_equal": tuple(nt.acoustic_tokens.shape) == tuple(it.acoustic_tokens.shape),
        "native_shape": list(nt.acoustic_tokens.shape),
        "integrated_shape": list(it.acoustic_tokens.shape),
        "native_frame_count": nt.generated_frame_count,
        "integrated_frame_count": it.generated_frame_count,
        "codebook_count_equal": nt.codebook_count == it.codebook_count,
        "token_equal": token_equal,
        "differing_token_count": int((nt.acoustic_tokens != it.acoustic_tokens).sum().item()) if nt.acoustic_tokens.shape == it.acoustic_tokens.shape else None,
        "codec_input_equal": codec_equal,
        "native_eos_index": nt.eos_index,
        "integrated_eos_index": it.eos_index,
        "native_termination_reason": nt.termination_reason,
        "integrated_termination_reason": it.termination_reason,
        "eos_equal": nt.eos_index == it.eos_index,
        "waveform_shape_equal": nt.waveform_shape == it.waveform_shape,
        "waveform_max_abs_diff": float(waveform_delta.max().item()) if waveform_delta is not None else None,
        "waveform_mean_abs_diff": float(waveform_delta.mean().item()) if waveform_delta is not None else None,
        "waveform_rms_diff": float(torch.sqrt((waveform_delta.square()).mean()).item()) if waveform_delta is not None else None,
        "native_token_sha256": nt.acoustic_token_sha256,
        "integrated_token_sha256": it.acoustic_token_sha256,
        "native_codec_input_sha256": nt.codec_input_sha256,
        "integrated_codec_input_sha256": it.codec_input_sha256,
        "native_waveform_sha256": nt.waveform_sha256,
        "integrated_waveform_sha256": it.waveform_sha256,
    }


def main() -> None:
    torch.set_num_threads(2)
    foundation = QwenFoundationTTS.from_local_path(MODEL_PATH, reference_audio=str(REFERENCE_AUDIO))
    adapter = QwenStage2BAdapter.from_foundation(foundation, initialization_seed=23)
    rows = []
    for text in PANEL:
        _, native = adapter.diagnostic_native_generation(
            text=text, x_vector_only_mode=True, do_sample=False, subtalker_dosample=False, max_new_tokens=2
        )
        rep = representation(text)
        _, integrated = adapter.diagnostic_conditioned_generation(
            rep, batch_for(rep), x_vector_only_mode=True, do_sample=False, subtalker_dosample=False, max_new_tokens=2
        )
        rows.append({"text": text, "comparison": compare(native, integrated), "native": native.acoustic_trace.to_summary(), "integrated": integrated.acoustic_trace.to_summary()})

    negative_text = "Kolkata hosted the conference."
    override = PronunciationOverride(0, 7, "swara-phones-v0", ("K", "O", "L"), "en-IN")
    _, no_override = adapter.diagnostic_conditioned_generation(
        representation(negative_text), batch_for(representation(negative_text)), x_vector_only_mode=True, do_sample=False, subtalker_dosample=False, max_new_tokens=2
    )
    overridden_rep = representation(negative_text, (override,))
    _, with_override = adapter.diagnostic_conditioned_generation(
        overridden_rep, batch_for(overridden_rep), x_vector_only_mode=True, do_sample=False, subtalker_dosample=False, max_new_tokens=2
    )
    rows.append({"text": negative_text, "case": "override_gate_zero_negative_control", "comparison": compare(no_override, with_override)})

    _, zero = adapter.diagnostic_conditioned_generation(
        representation(negative_text), batch_for(representation(negative_text)), x_vector_only_mode=True, do_sample=False, subtalker_dosample=False, max_new_tokens=2
    )
    adapter.config = type(adapter.config)(
        stage2b_input_dim=adapter.config.stage2b_input_dim,
        qwen_conditioning_dim=adapter.config.qwen_conditioning_dim,
        gate=0.001,
        strict_equivalence=False,
    )
    _, nonzero = adapter.diagnostic_conditioned_generation(
        representation(negative_text), batch_for(representation(negative_text)), x_vector_only_mode=True, do_sample=False, subtalker_dosample=False, max_new_tokens=2
    )
    zero_tokens = zero.acoustic_trace.acoustic_tokens
    nonzero_tokens = nonzero.acoustic_trace.acoustic_tokens
    rows.append({
        "text": negative_text,
        "case": "manual_nonzero_gate_0.001",
        "token_shape_equal": tuple(zero_tokens.shape) == tuple(nonzero_tokens.shape),
        "differing_token_count": int((zero_tokens != nonzero_tokens).sum().item()) if zero_tokens.shape == nonzero_tokens.shape else None,
        "zero": zero.acoustic_trace.to_summary(),
        "nonzero": nonzero.acoustic_trace.to_summary(),
    })
    print(json.dumps({"model_path": str(MODEL_PATH), "reference_audio": str(REFERENCE_AUDIO), "do_sample": False, "max_new_tokens": 2, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
