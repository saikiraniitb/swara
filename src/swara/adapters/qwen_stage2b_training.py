"""Read-only differentiable teacher-forcing seam for the local Qwen Talker.

The official top-level Qwen model exposes generation, not a speech-training
forward method. This Swara-owned helper uses the already-local Talker
decoder, codec head, and ``forward_sub_talker_finetune`` APIs with a supplied
native mixed schedule and a shared target acoustic history. It does not alter
Qwen source or instantiate an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor

from swara.models.stage2b_linguistic import (
    Stage2BLinguisticRepresentation,
    Stage2BTensorizedBatch,
)

from .qwen_stage2b import (
    QwenStage2BAlignment,
    aligned_swara_states,
    build_qwen_stage2b_alignment,
)


class QwenTeacherForcedError(ValueError):
    """Raised when a prepared Qwen teacher-forcing batch is invalid."""


@dataclass(frozen=True, slots=True)
class QwenTeacherForcedOutput:
    """Logits for all target frames under one shared teacher-forced history."""

    main_logits: Tensor
    residual_logits: Tensor
    target_codes: Tensor
    history_shared: bool


@dataclass(frozen=True, slots=True)
class QwenTeacherForcedSchedule:
    """The graph-connected, native Qwen mixed schedule before frame forcing."""

    inputs_embeds: Tensor
    native_inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    text_position_mask: Tensor
    trailing_text_hidden: Tensor
    tts_pad_embed: Tensor
    trailing_text_position_mask: Tensor
    acoustic_position_mask: Tensor
    native_text_token_ids: Tensor
    target_acoustic_history: Tensor
    alignment: QwenStage2BAlignment | None
    schedule_version: str
    native_prompt_text: str
    effective_gate: float

    def __post_init__(self) -> None:
        if self.inputs_embeds.ndim != 3 or self.native_inputs_embeds.shape != self.inputs_embeds.shape:
            raise QwenTeacherForcedError("Qwen schedule inputs must share [B,S,D] geometry")
        batch_size, sequence_length, hidden_size = self.inputs_embeds.shape
        if self.attention_mask.shape != (batch_size, sequence_length):
            raise QwenTeacherForcedError("Qwen schedule attention mask must be [B,S]")
        if self.position_ids.shape != (3, batch_size, sequence_length):
            raise QwenTeacherForcedError("Qwen schedule position IDs must be [3,B,S]")
        if self.text_position_mask.shape != (batch_size, sequence_length):
            raise QwenTeacherForcedError("Qwen text position mask must be [B,S]")
        if self.trailing_text_hidden.ndim != 3 or self.trailing_text_hidden.shape[0] != batch_size or self.trailing_text_hidden.shape[2] != hidden_size:
            raise QwenTeacherForcedError("Qwen trailing text geometry is inconsistent")
        if self.trailing_text_position_mask.shape != self.trailing_text_hidden.shape[:2]:
            raise QwenTeacherForcedError("Qwen trailing text mask must match trailing text states")
        if self.tts_pad_embed.ndim != 3 or self.tts_pad_embed.shape[-1] != self.inputs_embeds.shape[-1]:
            raise QwenTeacherForcedError("Qwen tts_pad_embed must have shape [B|1,1,D]")
        if self.tts_pad_embed.shape[0] not in (1, batch_size) or self.tts_pad_embed.shape[1] != 1:
            raise QwenTeacherForcedError("Qwen tts_pad_embed must have shape [B|1,1,D]")
        if self.acoustic_position_mask.shape != (batch_size, sequence_length):
            raise QwenTeacherForcedError("Qwen acoustic position mask must be [B,S]")
        if self.native_text_token_ids.ndim != 2 or self.native_text_token_ids.shape[0] != batch_size:
            raise QwenTeacherForcedError("native text token IDs must be [B,S]")
        if self.target_acoustic_history.ndim != 3 or self.target_acoustic_history.shape[0] != batch_size:
            raise QwenTeacherForcedError("target acoustic history must be [B,T,Q]")
        if not torch.isfinite(self.inputs_embeds).all() or not torch.isfinite(self.trailing_text_hidden).all():
            raise QwenTeacherForcedError("Qwen schedule contains non-finite states")


def _unwrap_qwen(model: Any) -> tuple[Any, Any, Any]:
    foundation = model
    wrapper = getattr(foundation, "_model", foundation)
    native_model = getattr(wrapper, "model", wrapper)
    processor = getattr(wrapper, "processor", None)
    if native_model is None or processor is None:
        raise QwenTeacherForcedError("Qwen model and processor are required")
    talker = getattr(native_model, "talker", None)
    if talker is None:
        raise QwenTeacherForcedError("Qwen model does not expose a Talker")
    return native_model, processor, talker


def _tokenize_prompt(processor: Any, prompt: str) -> Tensor:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise QwenTeacherForcedError("Stage2B teacher schedule currently supports one text sample")
    return input_ids


def _speaker_embedding(speaker_condition: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    if speaker_condition is None:
        return None
    value = speaker_condition
    if isinstance(value, Mapping):
        value = value.get("ref_spk_embedding", value.get("speaker_embedding"))
        if isinstance(value, (list, tuple)):
            value = value[0]
    elif isinstance(value, (list, tuple)):
        value = value[0]
    if not isinstance(value, Tensor):
        raise QwenTeacherForcedError("speaker_condition must be a Tensor or a voice-prompt mapping")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] != 1:
        raise QwenTeacherForcedError("speaker condition must have shape [1,D]")
    return value.to(device=device, dtype=dtype)


def build_qwen_teacher_forced_schedule(
    model: Any,
    *,
    text: str,
    language: str = "English",
    speaker_condition: Any = None,
    target_acoustic_codes: Tensor,
    stage2b_representation: Stage2BLinguisticRepresentation | None = None,
    stage2b_tensorized: Stage2BTensorizedBatch | None = None,
    stage2b_bridge: Any = None,
    gate: float | Tensor = 0.0,
    require_x_vector_only_mode: bool = True,
) -> QwenTeacherForcedSchedule:
    """Compile Qwen's native x-vector mixed schedule with optional Swara residual.

    This function reproduces the non-ICL schedule assembly from
    ``Qwen3TTSForConditionalGeneration.generate``. It does not reproduce
    sampling or autoregressive generation. Text embeddings, projection, and
    all conditioned additions remain under autograd; only frozen parameters
    are expected to have ``requires_grad=False`` at the caller.
    """

    if not isinstance(text, str) or not text:
        raise QwenTeacherForcedError("teacher schedule requires non-empty text")
    if target_acoustic_codes.ndim != 3 or target_acoustic_codes.shape[0] != 1:
        raise QwenTeacherForcedError("target acoustic history must have shape [1,T,Q]")
    if target_acoustic_codes.shape[1] <= 0:
        raise QwenTeacherForcedError("target acoustic history must contain frames")
    native_model, processor, talker = _unwrap_qwen(model)
    if require_x_vector_only_mode and speaker_condition is not None and isinstance(speaker_condition, Mapping):
        if speaker_condition.get("icl_mode", False):
            raise QwenTeacherForcedError("ICL/reference-text schedule is outside the Stage2B.4A contract")
    prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    try:
        device = next(talker.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    input_ids = _tokenize_prompt(processor, prompt).to(device=device)
    if input_ids.shape[1] < 9:
        raise QwenTeacherForcedError("Qwen prompt is too short for the native assistant schedule slices")
    talker_config = talker.config
    top_config = native_model.config
    hidden_size = int(talker_config.hidden_size)
    if target_acoustic_codes.shape[2] != int(talker_config.num_code_groups):
        raise QwenTeacherForcedError("target codebook count does not match Qwen Talker")
    if stage2b_representation is not None:
        if stage2b_representation.source_text != text:
            raise QwenTeacherForcedError("Stage2B source text does not match native Qwen text")
        if stage2b_tensorized is None or stage2b_bridge is None:
            raise QwenTeacherForcedError("conditioned schedule requires tensorized input and bridge")
        if stage2b_tensorized.features.shape[0] != 1 or stage2b_tensorized.provenance[0] != stage2b_representation.units:
            raise QwenTeacherForcedError("Stage2B tensorized provenance is malformed")
        if stage2b_bridge.backbone_dim != hidden_size:
            raise QwenTeacherForcedError("Stage2B bridge width does not match Qwen Talker")
        bridge_result = stage2b_bridge(stage2b_tensorized)
        alignment = build_qwen_stage2b_alignment(stage2b_representation, processor)
        aligned = aligned_swara_states(
            stage2b_representation,
            alignment,
            bridge_result.bridge_output[:, :len(stage2b_representation.units)],
        )
    else:
        aligned = None
        alignment = None
    gate_value = float(gate.detach().item()) if isinstance(gate, Tensor) else float(gate)
    if not math.isfinite(gate_value):
        raise QwenTeacherForcedError("conditioning gate must be finite")

    text_embedding = talker.get_text_embeddings()
    text_projection = talker.text_projection
    tts_bos, tts_eos, tts_pad = text_projection(
        text_embedding(torch.tensor([[top_config.tts_bos_token_id, top_config.tts_eos_token_id, top_config.tts_pad_token_id]], device=input_ids.device, dtype=input_ids.dtype))
    ).chunk(3, dim=1)
    speaker_embed = _speaker_embedding(speaker_condition, device=input_ids.device, dtype=tts_pad.dtype)
    language_key = language.lower()
    language_ids = getattr(talker_config, "codec_language_id", {})
    if language_key == "auto":
        language_id = None
    elif language_key not in language_ids:
        raise QwenTeacherForcedError(f"Qwen language is not available: {language}")
    else:
        language_id = int(language_ids[language_key])
    if language_id is None:
        codec_prefill = [[talker_config.codec_nothink_id, talker_config.codec_think_bos_id, talker_config.codec_think_eos_id]]
    else:
        codec_prefill = [[talker_config.codec_think_id, talker_config.codec_think_bos_id, language_id, talker_config.codec_think_eos_id]]
    codec0 = talker.get_input_embeddings()(torch.tensor(codec_prefill, device=input_ids.device, dtype=input_ids.dtype))
    codec1 = talker.get_input_embeddings()(torch.tensor([[talker_config.codec_pad_id, talker_config.codec_bos_id]], device=input_ids.device, dtype=input_ids.dtype))
    codec_input = torch.cat([codec0, codec1], dim=1) if speaker_embed is None else torch.cat([codec0, speaker_embed.view(1, 1, -1), codec1], dim=1)
    role_ids = input_ids[:, :3]
    role = text_projection(text_embedding(role_ids))
    control = torch.cat((tts_pad.expand(-1, codec_input.shape[1] - 2, -1), tts_bos), dim=1) + codec_input[:, :-1]
    native_first = text_projection(text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
    native_initial = torch.cat((role, control, native_first), dim=1)
    trailing_native = torch.cat((text_projection(text_embedding(input_ids[:, 4:-5])), tts_eos), dim=1)
    text_mask = torch.zeros((1, native_initial.shape[1]), dtype=torch.bool, device=input_ids.device)
    text_mask[:, native_initial.shape[1] - 1] = True
    trailing_mask = torch.zeros(trailing_native.shape[:2], dtype=torch.bool, device=input_ids.device)
    if trailing_mask.shape[1] > 1:
        trailing_mask[:, :-1] = True
    conditioned_initial = native_initial
    conditioned_trailing = trailing_native
    if aligned is not None and alignment is not None:
        # Native token index 3 is the first user token in this exact Qwen
        # template. Remaining content tokens are trailing text states.
        for native_position in alignment.conditioned_native_positions:
            edges = alignment.edges_for_native(native_position)
            residual = aligned[:, native_position:native_position + 1]
            if native_position == 3:
                conditioned_initial = conditioned_initial.clone()
                conditioned_initial[:, -1:] = _apply_gate(native_initial[:, -1:], residual, gate, gate_value)
            elif 4 <= native_position < input_ids.shape[1] - 5:
                trailing_index = native_position - 4
                if trailing_index < conditioned_trailing.shape[1] - 1:
                    if conditioned_trailing is trailing_native:
                        conditioned_trailing = conditioned_trailing.clone()
                    conditioned_trailing[:, trailing_index:trailing_index + 1] = _apply_gate(
                        trailing_native[:, trailing_index:trailing_index + 1], residual, gate, gate_value
                    )
            elif edges:
                raise QwenTeacherForcedError("attempted to condition a native control/special position")
    attention_mask = torch.ones(native_initial.shape[:2], dtype=torch.long, device=input_ids.device)
    if not callable(getattr(talker, "get_rope_index", None)):
        raise QwenTeacherForcedError("Qwen Talker does not expose get_rope_index")
    position_ids, _ = talker.get_rope_index(attention_mask)
    acoustic_mask = torch.zeros_like(text_mask)
    return QwenTeacherForcedSchedule(
        inputs_embeds=conditioned_initial,
        native_inputs_embeds=native_initial,
        attention_mask=attention_mask,
        position_ids=position_ids,
        text_position_mask=text_mask,
        trailing_text_hidden=conditioned_trailing,
        tts_pad_embed=tts_pad,
        trailing_text_position_mask=trailing_mask,
        acoustic_position_mask=acoustic_mask,
        native_text_token_ids=input_ids,
        target_acoustic_history=target_acoustic_codes.to(device=input_ids.device),
        alignment=alignment,
        schedule_version="swara.stage2b.qwen.teacher-schedule.xvector.v0",
        native_prompt_text=prompt,
        effective_gate=gate_value,
    )


def _apply_gate(native: Tensor, residual: Tensor, gate: float | Tensor, gate_value: float) -> Tensor:
    if native.shape != residual.shape:
        raise QwenTeacherForcedError("Qwen residual geometry does not match native text state")
    # Keep a tensor-valued gate in the autograd graph even at exactly zero.
    # ``native + 0 * residual`` is tensor-identical to ``native`` for finite
    # values, while the derivative with respect to the gate remains defined.
    # Fixed scalar inference retains the certified zero fast path.
    if gate_value == 0.0 and not isinstance(gate, Tensor):
        return native
    gate_tensor = gate.to(device=native.device, dtype=native.dtype) if isinstance(gate, Tensor) else gate
    return native + residual.to(device=native.device, dtype=native.dtype) * gate_tensor


def _position_ids(talker: Any, attention_mask: Tensor) -> Tensor:
    if not callable(getattr(talker, "get_rope_index", None)):
        raise QwenTeacherForcedError("Qwen Talker does not expose get_rope_index")
    positions, _ = talker.get_rope_index(attention_mask)
    if positions.ndim != 3:
        raise QwenTeacherForcedError("Qwen position IDs must have shape [3,B,S]")
    return positions


def _frame_embedding(talker: Any, codec_ids: Tensor) -> Tensor:
    """Reproduce the native Talker frame embedding sum without sampling."""

    if codec_ids.ndim != 2:
        raise QwenTeacherForcedError("codec frame must have shape [B,Q]")
    parts = [talker.get_input_embeddings()(codec_ids[:, :1])]
    residual_embeddings = talker.code_predictor.get_input_embeddings()
    for codebook in range(1, codec_ids.shape[1]):
        parts.append(residual_embeddings[codebook - 1](codec_ids[:, codebook:codebook + 1]))
    return torch.cat(parts, dim=1).sum(dim=1, keepdim=True)


def run_qwen_teacher_forced(
    talker: Any,
    *,
    mixed_talker_inputs: Tensor,
    attention_mask: Tensor,
    target_codes: Tensor,
    trailing_text_hidden: Tensor | None = None,
    tts_pad_embed: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> QwenTeacherForcedOutput:
    """Run native Qwen Talker decoder/code-predictor teacher forcing.

    ``mixed_talker_inputs`` is the native prompt schedule. A conditioned
    schedule is made by the caller by adding a Swara residual only at already
    existing text positions. The same ``target_codes`` are then fed to both
    calls, so native and conditioned logits have identical acoustic history.

    Main logits are aligned to target frames: the prefill last state predicts
    frame zero, and each subsequent decoder state predicts the next frame.
    Residual logits use Qwen's existing ``forward_sub_talker_finetune`` for
    each target frame and have shape ``[B,T,Q-1,V]``.
    """

    if mixed_talker_inputs.ndim != 3 or attention_mask.shape != mixed_talker_inputs.shape[:2]:
        raise QwenTeacherForcedError("mixed schedule must be [B,S,D] and mask [B,S]")
    if target_codes.ndim != 3 or target_codes.shape[0] != mixed_talker_inputs.shape[0]:
        raise QwenTeacherForcedError("target_codes must be [B,T,Q]")
    if attention_mask.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        raise QwenTeacherForcedError("attention_mask must be boolean or integer")
    config = getattr(talker, "config", None)
    q = getattr(config, "num_code_groups", None)
    hidden = getattr(config, "hidden_size", None)
    if not isinstance(q, int) or not isinstance(hidden, int) or target_codes.shape[2] != q:
        raise QwenTeacherForcedError("target codebook count does not match Qwen Talker config")
    if mixed_talker_inputs.shape[-1] != hidden:
        raise QwenTeacherForcedError("mixed schedule width does not match Qwen Talker hidden size")
    if not torch.isfinite(mixed_talker_inputs).all():
        raise QwenTeacherForcedError("mixed schedule contains non-finite values")
    compute_device = mixed_talker_inputs.device
    target_codes = target_codes.to(device=compute_device, dtype=torch.long)
    if trailing_text_hidden is not None:
        trailing_text_hidden = trailing_text_hidden.to(device=compute_device, dtype=mixed_talker_inputs.dtype)
    if tts_pad_embed is not None:
        tts_pad_embed = tts_pad_embed.to(device=compute_device, dtype=mixed_talker_inputs.dtype)

    batch_size, prompt_length = mixed_talker_inputs.shape[:2]
    full_mask = attention_mask.to(device=compute_device, dtype=torch.long)
    initial_positions = (
        _position_ids(talker, full_mask)
        if position_ids is None
        else position_ids.to(device=compute_device)
    )
    if initial_positions.shape != (3, batch_size, prompt_length):
        raise QwenTeacherForcedError("provided Qwen position IDs do not match the mixed schedule")
    decoder = getattr(talker, "model", None)
    codec_head = getattr(talker, "codec_head", None)
    if decoder is None or not callable(decoder) or codec_head is None:
        raise QwenTeacherForcedError("Qwen Talker decoder/model and codec_head are required")
    prefill = decoder(
        inputs_embeds=mixed_talker_inputs,
        attention_mask=full_mask,
        position_ids=initial_positions,
        use_cache=True,
        cache_position=torch.arange(prompt_length, device=mixed_talker_inputs.device),
    )
    current_hidden = prefill.last_hidden_state[:, -1, :]
    main = [codec_head(current_hidden.unsqueeze(1))]
    residual = []
    for frame_index in range(target_codes.shape[1]):
        residual_logits, _ = talker.forward_sub_talker_finetune(
            target_codes[:, frame_index, :], current_hidden
        )
        residual.append(residual_logits.unsqueeze(1))
        if frame_index + 1 >= target_codes.shape[1]:
            break
        frame = _frame_embedding(talker, target_codes[:, frame_index, :])
        if trailing_text_hidden is not None and frame_index < trailing_text_hidden.shape[1]:
            frame = frame + trailing_text_hidden[:, frame_index:frame_index + 1]
        elif tts_pad_embed is not None:
            frame = frame + tts_pad_embed
        full_mask = torch.cat(
            (full_mask, torch.ones((batch_size, 1), dtype=full_mask.dtype, device=full_mask.device)), dim=1
        )
        positions = _position_ids(talker, full_mask)[:, :, -1:]
        step = decoder(
            inputs_embeds=frame,
            attention_mask=full_mask,
            position_ids=positions,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            cache_position=torch.tensor([prompt_length + frame_index], device=frame.device),
        )
        prefill = step
        current_hidden = step.last_hidden_state[:, -1, :]
        main.append(codec_head(current_hidden.unsqueeze(1)))
    main_logits = torch.cat(main, dim=1)
    residual_logits = torch.cat(residual, dim=1)
    if main_logits.shape[1] != target_codes.shape[1] or residual_logits.shape[1] != target_codes.shape[1]:
        raise QwenTeacherForcedError("teacher-forced logits did not preserve target frame length")
    if not torch.isfinite(main_logits).all() or not torch.isfinite(residual_logits).all():
        raise QwenTeacherForcedError("Qwen teacher-forced logits are non-finite")
    return QwenTeacherForcedOutput(main_logits, residual_logits, target_codes, history_shared=True)


def run_qwen_teacher_forced_schedule(talker: Any, schedule: QwenTeacherForcedSchedule) -> QwenTeacherForcedOutput:
    """Run the decoder on a compiled native or conditioned schedule."""

    if not isinstance(schedule, QwenTeacherForcedSchedule):
        raise TypeError("expected QwenTeacherForcedSchedule")
    return run_qwen_teacher_forced(
        talker,
        mixed_talker_inputs=schedule.inputs_embeds,
        attention_mask=schedule.attention_mask,
        target_codes=schedule.target_acoustic_history,
        trailing_text_hidden=schedule.trailing_text_hidden,
        tts_pad_embed=schedule.tts_pad_embed,
        position_ids=schedule.position_ids,
    )


__all__ = [
    "QwenTeacherForcedError",
    "QwenTeacherForcedOutput",
    "QwenTeacherForcedSchedule",
    "build_qwen_teacher_forced_schedule",
    "run_qwen_teacher_forced",
    "run_qwen_teacher_forced_schedule",
]
