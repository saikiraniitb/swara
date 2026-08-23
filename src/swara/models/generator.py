"""Swara-owned Qwen-Talker-parity generator (v2)."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    model_dim: int = 384
    layers: int = 4
    heads: int = 6
    ffn_dim: int = 1536
    max_text_tokens: int = 128
    max_audio_frames: int = 256
    dropout: float = 0.0
    encoder_layers: int | None = None
    decoder_layers: int | None = None
    residual_dim: int | None = None
    primary_history_dropout: float = 0.0
    talker_version: str = "v2"

    def __post_init__(self) -> None:
        if self.linguistic_vocab_size < 2 or self.speaker_count < 1:
            raise ValueError("generator vocabulary and speaker count must be positive")
        if self.model_dim <= 0 or self.layers < 1 or self.heads < 1 or self.ffn_dim < self.model_dim:
            raise ValueError("generator dimensions are invalid")
        if self.model_dim % self.heads or self.residual_width % self.heads:
            raise ValueError("model and residual dimensions must divide evenly by heads")
        if self.max_text_tokens < 1 or self.max_audio_frames < 1:
            raise ValueError("maximum sequence lengths must be positive")
        if not 0.0 <= self.primary_history_dropout < 1.0:
            raise ValueError("primary history dropout must be in [0, 1)")
        if self.talker_version != "v2":
            raise ValueError("only Swara Talker v2 is active")
        if self.encoder_depth < 1 or self.decoder_depth < 1:
            raise ValueError("encoder and decoder must each have at least one layer")

    @property
    def encoder_depth(self) -> int:
        return self.encoder_layers if self.encoder_layers is not None else self.layers

    @property
    def decoder_depth(self) -> int:
        return self.decoder_layers if self.decoder_layers is not None else self.layers

    @property
    def residual_width(self) -> int:
        return self.residual_dim if self.residual_dim is not None else self.model_dim

    @property
    def primary_codec_vocabulary_size(self) -> int:
        return self.audio_spec.vocabulary_size

    @property
    def residual_codebook_count(self) -> int:
        return self.audio_spec.codebook_count - 1


class LearnedSpeakerConditioner:
    """Stable logical speaker ID resolution for the existing public boundary."""

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


class SwaraSpeechGenerator:
    """PyTorch is isolated here; public inputs and outputs stay Swara-native."""

    def __init__(self, config: GeneratorConfig, vocabulary: LinguisticVocabulary, speaker_conditioner: LearnedSpeakerConditioner) -> None:
        import torch
        from torch import nn

        if vocabulary.size != config.linguistic_vocab_size:
            raise ValueError("generator config and linguistic vocabulary size differ")
        self.torch = torch
        self.config = config
        self.vocabulary = vocabulary
        self.speaker_conditioner = speaker_conditioner
        kinds, languages, kind_count, language_count = _feature_lookups(vocabulary)
        self.module = _GeneratorModule(config, nn, kinds, languages, kind_count, language_count)

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
        return self.torch.tensor([self.vocabulary.encode(sequence).ids], dtype=self.torch.long, device=self.device)

    def forward(
        self,
        text_ids: Any,
        speaker_ids: Any,
        primary_inputs: Any,
        text_padding_mask: Any | None = None,
        primary_tokens_for_residual: Any | None = None,
        residual_history_inputs: Any | None = None,
        residual_targets_for_prediction: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        return self.module(
            text_ids, speaker_ids, primary_inputs, text_padding_mask,
            primary_tokens_for_residual, residual_history_inputs, residual_targets_for_prediction,
        )

    def teacher_forcing_inputs(self, target_frames: Any) -> Any:
        self._validate_target_frames(target_frames)
        batch, frames, _ = target_frames.shape
        result = self.torch.full((batch, frames), self.config.audio_spec.vocabulary_size, dtype=self.torch.long, device=target_frames.device)
        if frames > 1:
            result[:, 1:] = target_frames[:, :-1, 0]
        return result

    def teacher_forcing_frame_history(self, target_frames: Any) -> Any:
        self._validate_target_frames(target_frames)
        result = self.torch.zeros_like(target_frames)
        if target_frames.shape[1] > 1:
            result[:, 1:] = target_frames[:, :-1]
        return result

    def _validate_target_frames(self, target_frames: Any) -> None:
        if target_frames.ndim != 3 or target_frames.shape[-1] != self.config.audio_spec.codebook_count:
            raise ValueError("target frames must have shape (batch, frames, codebooks)")

    def generate(
        self,
        sequence: LinguisticSequence,
        speaker: SpeakerCondition,
        performance: PerformancePlan | None = None,
        generation: GenerationOptions | None = None,
        cache: object | None = None,
    ) -> AudioTokenSequence:
        """Generate one fresh bounded utterance. Cache is reserved by protocol."""
        del performance, cache
        if speaker.condition_kind != self.speaker_conditioner.condition_kind:
            raise ValueError("speaker condition is incompatible with the table conditioner")
        generation = generation or GenerationOptions(deterministic=True)
        limit = self.config.max_audio_frames
        if generation.max_duration_ms is not None:
            limit = min(limit, max(1, math.ceil(generation.max_duration_ms * self.config.audio_spec.frame_rate_hz / 1000)))
        text_ids = self.encode_linguistic(sequence)
        speakers = self.torch.tensor([self.speaker_conditioner.resolve_id(speaker.reference_id)], dtype=self.torch.long, device=self.device)
        random = None
        if generation.seed is not None:
            random = self.torch.Generator(device=self.device)
            random.manual_seed(generation.seed)
        primary_inputs = self.torch.full((1, 1), self.config.audio_spec.vocabulary_size, dtype=self.torch.long, device=self.device)
        histories = self.torch.zeros((1, 1, self.config.audio_spec.codebook_count), dtype=self.torch.long, device=self.device)
        frames: list[tuple[int, ...]] = []
        self.module.eval()
        with self.torch.no_grad():
            for _ in range(limit):
                primary_logits, _, hidden = self.forward(text_ids, speakers, primary_inputs, residual_history_inputs=histories)
                primary = self._select(primary_logits[0, -1], generation.deterministic, random)
                primary_tensor = self.torch.tensor([[primary]], dtype=self.torch.long, device=self.device)
                residual_logits = self.module.residual_logits(hidden[:, -1:], primary_tensor)
                residual = [self._select(logits, generation.deterministic, random) for logits in residual_logits[0, 0]]
                frame = (primary, *residual)
                frames.append(frame)
                primary_inputs = self.torch.cat((primary_inputs, primary_tensor), dim=1)
                histories = self.torch.cat((histories, self.torch.tensor([[frame]], dtype=self.torch.long, device=self.device)), dim=1)
        output = AudioTokenSequence(frames=tuple(frames), spec_version=self.config.audio_spec.version)
        output.validate_against(self.config.audio_spec)
        return output

    def _select(self, logits: Any, deterministic: bool, random: Any | None) -> int:
        if deterministic:
            return int(logits.argmax().item())
        return int(self.torch.multinomial(self.torch.softmax(logits, dim=-1), 1, generator=random).item())


def _feature_lookups(vocabulary: LinguisticVocabulary) -> tuple[list[int], list[int], int, int]:
    """Keep kind and language independently learnable without changing M1."""
    kind_ids = {"<special>": 0, "grapheme": 1, "pronunciation": 2, "punctuation": 3, "boundary": 4}
    language_ids = {"<none>": 0}
    kinds: list[int] = []
    languages: list[int] = []
    for symbol in vocabulary.to_dict()["symbols"]:
        if symbol.startswith("<"):
            kind, language = "<special>", "<none>"
        else:
            kind, language, _ = json.loads(symbol)
            if kind not in kind_ids:
                kind_ids[kind] = len(kind_ids)
            if language not in language_ids:
                language_ids[language] = len(language_ids)
        kinds.append(kind_ids[kind])
        languages.append(language_ids[language])
    return kinds, languages, len(kind_ids), len(language_ids)


class _GeneratorModule:
    def __new__(cls, config: GeneratorConfig, nn: Any, kinds: list[int], languages: list[int], kind_count: int, language_count: int) -> Any:
        import torch
        import torch.nn.functional as functional

        class RotaryAttention(nn.Module):
            def __init__(self, dim: int, heads: int, dropout: float) -> None:
                super().__init__()
                self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
                if self.head_dim % 2:
                    raise ValueError("rotary attention head dimension must be even")
                self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))

            def forward(self, query: Any, key_value: Any, key_padding_mask: Any | None = None, causal: bool = False, rotary: bool = False) -> Any:
                batch, q_len, dim = query.shape
                k_len = key_value.shape[1]
                q = self.q_proj(query).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(key_value).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(key_value).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
                if rotary:
                    q, k = self._rotate(q), self._rotate(k)
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if causal:
                    scores = scores.masked_fill(torch.triu(torch.ones((q_len, k_len), dtype=torch.bool, device=query.device), diagonal=1), float("-inf"))
                if key_padding_mask is not None:
                    scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
                weights = functional.dropout(functional.softmax(scores, dim=-1), p=self.dropout, training=self.training)
                return self.out_proj((weights @ v).transpose(1, 2).reshape(batch, q_len, dim))

            def _rotate(self, values: Any) -> Any:
                length, half = values.shape[-2], self.head_dim // 2
                frequencies = 1.0 / (10000 ** (torch.arange(half, device=values.device, dtype=values.dtype) / half))
                angles = torch.outer(torch.arange(length, device=values.device, dtype=values.dtype), frequencies)
                cos, sin = angles.cos()[None, None], angles.sin()[None, None]
                first, second = values[..., :half], values[..., half:]
                return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

        class EncoderLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_self = nn.LayerNorm(config.model_dim)
                self.self_attention = RotaryAttention(config.model_dim, config.heads, config.dropout)
                self.norm_ffn = nn.LayerNorm(config.model_dim)
                self.ffn = nn.Sequential(nn.Linear(config.model_dim, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, config.model_dim))

            def forward(self, states: Any, padding: Any) -> Any:
                normal = self.norm_self(states)
                states = states + self.self_attention(normal, normal, padding, rotary=True)
                return states + self.ffn(self.norm_ffn(states))

        class DecoderLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_self, self.norm_cross, self.norm_ffn = (nn.LayerNorm(config.model_dim) for _ in range(3))
                self.self_attention = RotaryAttention(config.model_dim, config.heads, config.dropout)
                self.cross_attention = RotaryAttention(config.model_dim, config.heads, config.dropout)
                self.ffn = nn.Sequential(nn.Linear(config.model_dim, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, config.model_dim))
                self.speaker_projection = nn.Linear(config.model_dim, config.model_dim, bias=False)

            def forward(self, states: Any, memory: Any, padding: Any, speaker: Any) -> Any:
                normal = self.norm_self(states + self.speaker_projection(speaker).unsqueeze(1))
                states = states + self.self_attention(normal, normal, causal=True, rotary=True)
                states = states + self.cross_attention(self.norm_cross(states), memory, padding)
                return states + self.ffn(self.norm_ffn(states))

        class ResidualPredictor(nn.Module):
            """Codebooks 1--15 are generated causally with selected earlier groups."""
            def __init__(self) -> None:
                super().__init__()
                width = config.residual_width
                self.primary_embedding = nn.Embedding(config.audio_spec.vocabulary_size, width)
                self.previous_embedding = nn.Embedding(config.audio_spec.vocabulary_size + 1, width)
                self.codebook_embedding = nn.Embedding(config.residual_codebook_count, width)
                self.hidden_projection = nn.Linear(config.model_dim, width)
                self.cell = nn.GRUCell(width, width)
                self.output = nn.Linear(width, config.audio_spec.vocabulary_size)
                self.bos_id = config.audio_spec.vocabulary_size

            def forward(self, hidden: Any, primary: Any, targets: Any | None = None) -> Any:
                if primary.shape != hidden.shape[:2]:
                    raise ValueError("primary tokens and decoder state must align")
                if targets is not None and targets.shape != (*hidden.shape[:2], config.residual_codebook_count):
                    raise ValueError("residual targets must align to frames and codebooks")
                batch, frames, _ = hidden.shape
                state = torch.tanh(self.hidden_projection(hidden) + self.primary_embedding(primary))
                previous = torch.full((batch, frames), self.bos_id, dtype=torch.long, device=hidden.device)
                logits: list[Any] = []
                for index in range(config.residual_codebook_count):
                    codebook = self.codebook_embedding.weight[index].view(1, 1, -1)
                    state = self.cell((self.previous_embedding(previous) + codebook).reshape(-1, config.residual_width), state.reshape(-1, config.residual_width)).view(batch, frames, config.residual_width)
                    current = self.output(state + codebook)
                    logits.append(current)
                    previous = targets[:, :, index] if targets is not None else current.argmax(dim=-1)
                return torch.stack(logits, dim=2)

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                dim = config.model_dim
                self.symbol_embedding = nn.Embedding(config.linguistic_vocab_size, dim, padding_idx=0)
                self.kind_embedding, self.language_embedding = nn.Embedding(kind_count, dim), nn.Embedding(language_count, dim)
                self.speaker_embedding = nn.Embedding(config.speaker_count, dim)
                self.primary_embedding = nn.Embedding(config.audio_spec.vocabulary_size + 1, dim)
                self.history_residual_embeddings = nn.ModuleList([nn.Embedding(config.audio_spec.vocabulary_size, dim) for _ in range(config.residual_codebook_count)])
                self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(config.encoder_depth)])
                self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(config.decoder_depth)])
                self.encoder_norm, self.decoder_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
                self.step_text_projection = nn.Linear(dim, dim, bias=False)
                self.codec_bos_embedding = nn.Parameter(torch.zeros(dim))
                self.codec_eos_embedding = nn.Parameter(torch.zeros(dim))
                self.primary_head = nn.Linear(dim, config.audio_spec.vocabulary_size)
                self.residual_predictor = ResidualPredictor()
                self.register_buffer("kind_lookup", torch.tensor(kinds, dtype=torch.long), persistent=True)
                self.register_buffer("language_lookup", torch.tensor(languages, dtype=torch.long), persistent=True)

            def forward(self, text_ids: Any, speaker_ids: Any, primary_inputs: Any, text_padding_mask: Any | None = None, primary_tokens_for_residual: Any | None = None, residual_history_inputs: Any | None = None, residual_targets_for_prediction: Any | None = None) -> tuple[Any, Any, Any]:
                if text_ids.ndim != 2 or primary_inputs.ndim != 2 or speaker_ids.ndim != 1:
                    raise ValueError("text IDs, speaker IDs, and primary inputs must be rank 2, 1, and 2")
                batch, frames = primary_inputs.shape
                if text_ids.shape[0] != batch or speaker_ids.shape[0] != batch:
                    raise ValueError("batch dimensions must agree")
                if frames > config.max_audio_frames or text_ids.shape[1] > config.max_text_tokens:
                    raise ValueError("input exceeds configured sequence length")
                padding = text_ids == 0 if text_padding_mask is None else text_padding_mask
                if residual_history_inputs is None:
                    residual_history_inputs = torch.zeros((batch, frames, config.audio_spec.codebook_count), dtype=torch.long, device=primary_inputs.device)
                if residual_history_inputs.shape != (batch, frames, config.audio_spec.codebook_count):
                    raise ValueError("residual history must have shape (batch, frames, codebooks)")
                memory = self.symbol_embedding(text_ids) + self.kind_embedding(self.kind_lookup[text_ids]) + self.language_embedding(self.language_lookup[text_ids])
                for layer in self.encoder_layers:
                    memory = layer(memory, padding)
                memory = self.encoder_norm(memory)
                audio_inputs = primary_inputs
                if self.training and config.primary_history_dropout:
                    drop = torch.rand(primary_inputs.shape, device=primary_inputs.device) < config.primary_history_dropout
                    drop[:, :1] = False
                    audio_inputs = primary_inputs.masked_fill(drop, config.audio_spec.vocabulary_size)
                audio = self.primary_embedding(audio_inputs)
                for codebook, embedding in enumerate(self.history_residual_embeddings, start=1):
                    audio = audio + embedding(residual_history_inputs[:, :, codebook])
                # Qwen Talker parity: the projected linguistic state remains
                # active at every speech step. States are aligned monotonically
                # across the utterance and padded with the final state.
                text_length = memory.shape[1]
                frame_positions = torch.arange(frames, device=audio.device)
                if text_length == 1:
                    aligned = memory[:, :1].expand(batch, frames, -1)
                else:
                    indices = torch.div(frame_positions * text_length, max(frames, 1), rounding_mode="floor").clamp_max(text_length - 1)
                    aligned = memory[:, indices]
                audio = audio + self.step_text_projection(aligned)
                if frames:
                    audio[:, 0] = audio[:, 0] + self.codec_bos_embedding
                speaker = self.speaker_embedding(speaker_ids)
                for layer in self.decoder_layers:
                    audio = layer(audio, memory, padding, speaker)
                hidden = self.decoder_norm(audio)
                primary_logits = self.primary_head(hidden)
                primary = primary_tokens_for_residual if primary_tokens_for_residual is not None else primary_inputs.clamp_max(config.audio_spec.vocabulary_size - 1)
                return primary_logits, self.residual_predictor(hidden, primary, residual_targets_for_prediction), hidden

            def residual_logits(self, hidden: Any, primary: Any, targets: Any | None = None) -> Any:
                return self.residual_predictor(hidden, primary, targets)

        return Module()
