"""Small, original, staged causal Swara speech generator for M2B.

This architecture-validation model predicts the primary codec stream at the
audio-frame rate, then predicts each residual codebook from the resulting main
hidden state and primary token. It intentionally has no Qwen/Dia generator
imports, code, checkpoints, or text-token dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from swara.contracts import AudioTokenSequence, AudioTokenSpec, GenerationOptions, PerformancePlan, SpeakerCondition
from swara.frontend.tokenizer import LinguisticSequence

from .linguistic import LinguisticVocabulary


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    linguistic_vocab_size: int
    speaker_count: int
    audio_spec: AudioTokenSpec
    model_dim: int = 256
    layers: int = 4
    heads: int = 4
    ffn_dim: int = 512
    max_text_tokens: int = 128
    max_audio_frames: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.linguistic_vocab_size < 2 or self.speaker_count < 1:
            raise ValueError("generator vocabulary and speaker count must be positive")
        if self.model_dim <= 0 or self.layers < 1 or self.heads < 1 or self.ffn_dim < self.model_dim:
            raise ValueError("generator dimensions are invalid")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.max_text_tokens < 1 or self.max_audio_frames < 1:
            raise ValueError("maximum sequence lengths must be positive")

    @property
    def primary_codec_vocabulary_size(self) -> int:
        return self.audio_spec.vocabulary_size

    @property
    def residual_codebook_count(self) -> int:
        return self.audio_spec.codebook_count - 1


class LearnedSpeakerConditioner:
    """M2B table-ID conditioner; future encoders can preserve this boundary."""

    condition_kind = "speaker_id_table.v0"

    def __init__(self, speaker_ids: tuple[str, ...]) -> None:
        if not speaker_ids or len(set(speaker_ids)) != len(speaker_ids):
            raise ValueError("speaker IDs must be non-empty and unique")
        self._indices = {speaker_id: index for index, speaker_id in enumerate(speaker_ids)}

    def resolve_id(self, speaker_id: str) -> int:
        try:
            return self._indices[speaker_id]
        except KeyError as error:
            raise ValueError(f"unknown speaker ID: {speaker_id}") from error

    def resolve(self, speaker_id: str) -> SpeakerCondition:
        self.resolve_id(speaker_id)
        return SpeakerCondition(condition_kind=self.condition_kind, reference_id=speaker_id)


class SwaraSpeechGenerator:  # PyTorch is purposefully isolated to this optional module.
    """Causal primary-token transformer plus a parallel residual predictor."""

    def __init__(self, config: GeneratorConfig, vocabulary: LinguisticVocabulary, speaker_conditioner: LearnedSpeakerConditioner) -> None:
        import torch
        from torch import nn

        if vocabulary.size != config.linguistic_vocab_size:
            raise ValueError("generator config and linguistic vocabulary size differ")
        self.torch = torch
        self.nn = nn
        self.config = config
        self.vocabulary = vocabulary
        self.speaker_conditioner = speaker_conditioner
        self.module = _GeneratorModule(config, nn)

    @property
    def device(self) -> Any:
        return next(self.module.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters())

    def train(self) -> "SwaraSpeechGenerator":
        self.module.train()
        return self

    def eval(self) -> "SwaraSpeechGenerator":
        self.module.eval()
        return self

    def parameters(self) -> Any:
        return self.module.parameters()

    def encode_linguistic(self, sequence: LinguisticSequence) -> Any:
        encoded = self.vocabulary.encode(sequence)
        return self.torch.tensor([encoded.ids], dtype=self.torch.long, device=self.device)

    def forward(
        self,
        text_ids: Any,
        speaker_ids: Any,
        primary_inputs: Any,
        text_padding_mask: Any | None = None,
        primary_tokens_for_residual: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        """Teacher-forced logits: primary `(B,T,V)`, residual `(B,T,Q-1,V)`."""
        return self.module(text_ids, speaker_ids, primary_inputs, text_padding_mask, primary_tokens_for_residual)

    def teacher_forcing_inputs(self, target_frames: Any) -> Any:
        """Shift primary targets right and prepend the dedicated BOS token."""
        if target_frames.ndim != 3 or target_frames.shape[-1] != self.config.audio_spec.codebook_count:
            raise ValueError("target frames must have shape (batch, frames, codebooks)")
        batch, frames, _ = target_frames.shape
        result = self.torch.full((batch, frames), self.config.audio_spec.vocabulary_size, dtype=self.torch.long, device=target_frames.device)
        if frames > 1:
            result[:, 1:] = target_frames[:, :-1, 0]
        return result

    def generate(
        self,
        sequence: LinguisticSequence,
        speaker: SpeakerCondition,
        performance: PerformancePlan | None = None,
        generation: GenerationOptions | None = None,
        cache: object | None = None,
    ) -> AudioTokenSequence:
        """Autoregressively emit bounded token frames; M2B deliberately has no KV cache."""
        del performance, cache  # Neutral performance only; cache is reserved by the public boundary.
        if speaker.condition_kind != self.speaker_conditioner.condition_kind:
            raise ValueError("speaker condition is incompatible with the M2B table conditioner")
        generation = generation or GenerationOptions(deterministic=True)
        frame_limit = self.config.max_audio_frames
        if generation.max_duration_ms is not None:
            frame_limit = min(frame_limit, max(1, math.ceil(generation.max_duration_ms * self.config.audio_spec.frame_rate_hz / 1000)))
        text_ids = self.encode_linguistic(sequence)
        speaker_index = self.speaker_conditioner.resolve_id(speaker.reference_id)
        speaker_ids = self.torch.tensor([speaker_index], dtype=self.torch.long, device=self.device)
        generator = None
        if generation.seed is not None:
            generator = self.torch.Generator(device=self.device)
            generator.manual_seed(generation.seed)
        primary_inputs = self.torch.full((1, 1), self.config.audio_spec.vocabulary_size, dtype=self.torch.long, device=self.device)
        frames: list[tuple[int, ...]] = []
        self.module.eval()
        with self.torch.no_grad():
            for _ in range(frame_limit):
                primary_logits, _, hidden = self.forward(text_ids, speaker_ids, primary_inputs)
                primary = self._select(primary_logits[0, -1], generation.deterministic, generator)
                current_primary = self.torch.tensor([[primary]], dtype=self.torch.long, device=self.device)
                residual_frame = self.module.residual_logits(hidden[:, -1:], current_primary)[0, 0]
                residual = [self._select(logits, generation.deterministic, generator) for logits in residual_frame]
                frames.append((primary, *residual))
                primary_inputs = self.torch.cat((primary_inputs, self.torch.tensor([[primary]], dtype=self.torch.long, device=self.device)), dim=1)
        output = AudioTokenSequence(frames=tuple(frames), spec_version=self.config.audio_spec.version)
        output.validate_against(self.config.audio_spec)
        return output

    def _select(self, logits: Any, deterministic: bool, generator: Any | None) -> int:
        if deterministic:
            return int(logits.argmax().item())
        probabilities = self.torch.softmax(logits, dim=-1)
        return int(self.torch.multinomial(probabilities, 1, generator=generator).item())


class _GeneratorModule:  # composition keeps the public wrapper free of torch subclassing details
    def __new__(cls, config: GeneratorConfig, nn: Any) -> Any:
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                dim = config.model_dim
                self.text_embedding = nn.Embedding(config.linguistic_vocab_size, dim, padding_idx=0)
                self.speaker_embedding = nn.Embedding(config.speaker_count, dim)
                self.audio_embedding = nn.Embedding(config.audio_spec.vocabulary_size + 1, dim)
                self.text_position_embedding = nn.Embedding(config.max_text_tokens, dim)
                self.position_embedding = nn.Embedding(config.max_audio_frames, dim)
                layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=config.heads,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
                self.final_norm = nn.LayerNorm(dim)
                self.primary_head = nn.Linear(dim, config.audio_spec.vocabulary_size)
                self.primary_embedding = nn.Embedding(config.audio_spec.vocabulary_size, dim)
                self.residual_codebook_embedding = nn.Embedding(config.audio_spec.codebook_count - 1, dim)
                self.residual_heads = nn.ModuleList(
                    [nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.audio_spec.vocabulary_size)) for _ in range(config.audio_spec.codebook_count - 1)]
                )

            def forward(
                self,
                text_ids: Any,
                speaker_ids: Any,
                primary_inputs: Any,
                text_padding_mask: Any | None = None,
                primary_tokens_for_residual: Any | None = None,
            ) -> tuple[Any, Any, Any]:
                if text_ids.ndim != 2 or primary_inputs.ndim != 2 or speaker_ids.ndim != 1:
                    raise ValueError("text IDs, speaker IDs, and primary inputs must be rank 2, 1, and 2")
                batch, frames = primary_inputs.shape
                if frames > config.max_audio_frames or text_ids.shape[1] > config.max_text_tokens:
                    raise ValueError("input exceeds configured sequence length")
                text_length = text_ids.shape[1]
                speaker = self.speaker_embedding(speaker_ids).unsqueeze(1)
                text_positions = self.text_position_embedding(self.position_ids(text_length, text_ids.device)).unsqueeze(0)
                audio_positions = self.position_embedding(self.position_ids(frames, primary_inputs.device)).unsqueeze(0)
                # Text is a causal prefix, so every audio frame can attend to
                # every linguistic kind/language/value token directly. This
                # replaces the M2B smoke model's single pooled text vector,
                # which a teacher-forced audio path could effectively ignore.
                text_prefix = self.text_embedding(text_ids) + text_positions + speaker
                audio_stream = self.audio_embedding(primary_inputs) + audio_positions + speaker
                conditioned = self.cat((text_prefix, audio_stream), dim=1)
                causal_mask = self.causal_mask(text_length + frames, primary_inputs.device)
                if text_padding_mask is None:
                    prefix_padding = text_ids == 0
                else:
                    prefix_padding = text_padding_mask
                audio_padding = self.zeros((batch, frames), dtype=self.bool, device=primary_inputs.device)
                padding_mask = self.cat((prefix_padding, audio_padding), dim=1)
                hidden = self.final_norm(self.transformer(conditioned, mask=causal_mask, src_key_padding_mask=padding_mask))[:, text_length:]
                primary_logits = self.primary_head(hidden)
                residual_tokens = primary_tokens_for_residual if primary_tokens_for_residual is not None else primary_inputs.clamp_max(config.audio_spec.vocabulary_size - 1)
                return primary_logits, self.residual_logits(hidden, residual_tokens), hidden

            def residual_logits(self, hidden: Any, primary_tokens: Any) -> Any:
                """Second-stage logits for supplied teacher-forced or sampled primary tokens."""
                if primary_tokens.shape != hidden.shape[:2]:
                    raise ValueError("primary tokens for residual prediction must align to hidden frames")
                # Every residual head receives the selected/teacher-forced
                # primary token and the frame hidden state; no upstream
                # residual implementation is used.
                residual_base = hidden + self.primary_embedding(primary_tokens)
                residual_logits = []
                for index, head in enumerate(self.residual_heads):
                    codebook = self.residual_codebook_embedding.weight[index].view(1, 1, -1)
                    residual_logits.append(head(residual_base + codebook))
                return self.stack(residual_logits, dim=2)

            @staticmethod
            def position_ids(frames: int, device: Any) -> Any:
                import torch
                return torch.arange(frames, device=device)

            @staticmethod
            def causal_mask(frames: int, device: Any) -> Any:
                import torch
                return torch.triu(torch.ones((frames, frames), dtype=torch.bool, device=device), diagonal=1)

            @staticmethod
            def stack(values: list[Any], dim: int) -> Any:
                import torch
                return torch.stack(values, dim=dim)

            @staticmethod
            def cat(values: tuple[Any, ...], dim: int) -> Any:
                import torch
                return torch.cat(values, dim=dim)

            @staticmethod
            def zeros(*shape: Any, **kwargs: Any) -> Any:
                import torch
                return torch.zeros(*shape, **kwargs)

            @property
            def bool(self) -> Any:
                import torch
                return torch.bool

        return Module()
