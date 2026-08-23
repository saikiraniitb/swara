"""Swara Generator v3.2 with normalized gated history/text fusion.

This is an independently written debug model.  It mirrors the important Qwen
Talker relationship (prefilled text/control sequence, then one full codec-frame
embedding plus trailing text state per generation step) without importing Qwen
generator code or IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from swara.contracts import AudioTokenSequence, AudioTokenSpec, SpeakerCondition
from swara.frontend.tokenizer import LinguisticSequence
from .linguistic import LinguisticVocabulary


@dataclass(frozen=True, slots=True)
class GeneratorV3Config:
    linguistic_vocab_size: int
    speaker_count: int
    audio_spec: AudioTokenSpec
    model_dim: int = 384
    layers: int = 4
    heads: int = 6
    ffn_dim: int = 1536
    max_text_tokens: int = 160
    max_audio_frames: int = 256
    dropout: float = 0.0
    talker_version: str = "v3"

    def __post_init__(self) -> None:
        if self.model_dim % self.heads or self.linguistic_vocab_size < 2 or self.speaker_count < 1:
            raise ValueError("invalid v3 dimensions or vocabulary")
        if self.audio_spec.codebook_count != 16 or self.audio_spec.vocabulary_size != 2048:
            raise ValueError("v3 debug model requires the Swara 16x2048 codec spec")


class SwaraSpeechGeneratorV32:
    """Public Swara-native wrapper around the v3 PyTorch module."""

    def __init__(self, config: GeneratorV3Config, vocabulary: LinguisticVocabulary, speaker_ids: tuple[str, ...]):
        import torch
        from torch import nn
        if vocabulary.size != config.linguistic_vocab_size:
            raise ValueError("vocabulary size mismatch")
        self.torch, self.config, self.vocabulary = torch, config, vocabulary
        self.speaker_ids = speaker_ids
        self.speaker_to_id = {s: i for i, s in enumerate(speaker_ids)}
        kinds, languages, kind_count, language_count = self._lookups(vocabulary)
        self.module = _V32Module(config, nn, kinds, languages, kind_count, language_count)

    @staticmethod
    def _lookups(vocab: LinguisticVocabulary):
        import json
        kind_ids = {"<special>": 0, "grapheme": 1, "pronunciation": 2, "punctuation": 3, "boundary": 4}
        lang_ids = {"<none>": 0}
        kinds, langs = [], []
        for symbol in vocab.to_dict()["symbols"]:
            if symbol.startswith("<"):
                kind, lang = "<special>", "<none>"
            else:
                kind, lang, _ = json.loads(symbol)
                kind_ids.setdefault(kind, len(kind_ids)); lang_ids.setdefault(lang, len(lang_ids))
            kinds.append(kind_ids[kind]); langs.append(lang_ids[lang])
        return kinds, langs, len(kind_ids), len(lang_ids)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    @property
    def device(self):
        return next(self.module.parameters()).device

    @property
    def gate_values(self) -> dict[str, float]:
        return {
            "acoustic_gate": float(self.module.acoustic_gate.detach().cpu()),
            "linguistic_gate": float(self.module.linguistic_gate.detach().cpu()),
        }

    def train(self):
        self.module.train(); return self

    def eval(self):
        self.module.eval(); return self

    def parameters(self):
        return self.module.parameters()

    def encode_linguistic(self, sequence: LinguisticSequence):
        encoded = self.vocabulary.encode(sequence)
        return self.torch.tensor([encoded.ids], dtype=self.torch.long, device=self.device)

    def speaker_tensor(self, speaker_id: str):
        if speaker_id not in self.speaker_to_id:
            raise ValueError(f"unknown speaker: {speaker_id}")
        return self.torch.tensor([self.speaker_to_id[speaker_id]], dtype=self.torch.long, device=self.device)

    def forward(self, text_ids, speaker_ids, target_frames, schedule_frames=None):
        return self.module(text_ids, speaker_ids, target_frames, schedule_frames=schedule_frames)

    @staticmethod
    def losses(outputs, targets):
        import torch.nn.functional as F
        primary, residual = outputs[0], outputs[1]
        primary_target = targets[:, :, 0]
        primary_loss = F.cross_entropy(primary.reshape(-1, primary.shape[-1]), primary_target.reshape(-1))
        residual_loss = F.cross_entropy(residual.reshape(-1, residual.shape[-1]), targets[:, :, 1:].reshape(-1))
        total = primary_loss + residual_loss
        with __import__("torch").no_grad():
            pa = (primary.argmax(-1) == primary_target).float().mean()
            ra = (residual.argmax(-1) == targets[:, :, 1:]).float().mean()
        return total, primary_loss, residual_loss, pa, ra

    def generate(self, sequence: LinguisticSequence, speaker: SpeakerCondition, max_frames: int) -> AudioTokenSequence:
        if speaker.reference_id not in self.speaker_to_id:
            raise ValueError("unknown speaker")
        text = self.encode_linguistic(sequence)
        sid = self.speaker_tensor(speaker.reference_id)
        frames: list[tuple[int, ...]] = []
        self.eval()
        with self.torch.no_grad():
            for _ in range(min(max_frames, self.config.max_audio_frames)):
                if frames:
                    previous = self.torch.tensor([frames], dtype=self.torch.long, device=self.device)
                    # Include one placeholder position for the frame currently
                    # being predicted.  Its frame input is derived only from
                    # the preceding frame, while the placeholder gives the
                    # Transformer a speech position whose logits we can read.
                    target = self.torch.cat(
                        (previous, self.torch.zeros((1, 1, 16), dtype=self.torch.long, device=self.device)),
                        dim=1,
                    )
                else:
                    target = self.torch.zeros((1, 1, 16), dtype=self.torch.long, device=self.device)
                outputs = self.module(text, sid, target, generation=True, schedule_frames=max_frames)
                primary = int(outputs[0][0, -1].argmax().item())
                hidden = outputs[2][:, -1:]
                residual = self.module.generate_residual(hidden, self.torch.tensor([[primary]], device=self.device))
                frames.append((primary, *[int(x) for x in residual[0, 0].tolist()]))
        out = AudioTokenSequence(tuple(frames), self.config.audio_spec.version)
        out.validate_against(self.config.audio_spec)
        return out


class _V32Module:
    def __new__(cls, config, nn, kinds, languages, kind_count, language_count):
        import torch
        import torch.nn.functional as F

        class Module(nn.Module):
            def __init__(self):
                super().__init__(); d = config.model_dim
                self.symbol = nn.Embedding(config.linguistic_vocab_size, d, padding_idx=0)
                self.kind = nn.Embedding(kind_count, d); self.language = nn.Embedding(language_count, d)
                self.text_pos = nn.Embedding(config.max_text_tokens, d); self.audio_pos = nn.Embedding(config.max_audio_frames, d)
                self.modality = nn.Embedding(2, d); self.speaker = nn.Embedding(config.speaker_count, d)
                self.control = nn.Parameter(torch.randn(4, d) * 0.02)
                self.acoustic_norm = nn.LayerNorm(d)
                self.linguistic_norm = nn.LayerNorm(d)
                self.acoustic_gate = nn.Parameter(torch.tensor(0.3))
                self.linguistic_gate = nn.Parameter(torch.tensor(1.0))
                self.frame_codebooks = nn.ModuleList([nn.Embedding(config.audio_spec.vocabulary_size, d) for _ in range(16)])
                layer = nn.TransformerEncoderLayer(d, config.heads, config.ffn_dim, config.dropout, batch_first=True, norm_first=True, activation="gelu")
                self.decoder = nn.TransformerEncoder(layer, config.layers); self.norm = nn.LayerNorm(d)
                self.primary_head = nn.Linear(d, config.audio_spec.vocabulary_size)
                self.residual_codebook = nn.Embedding(15, d); self.residual_prev = nn.Embedding(config.audio_spec.vocabulary_size + 1, d)
                self.residual_primary = nn.Embedding(config.audio_spec.vocabulary_size, d)
                self.residual_cell = nn.GRUCell(d, d); self.residual_head = nn.Linear(d, config.audio_spec.vocabulary_size)
                self.trailing = nn.Linear(d, d, bias=False)
                self.register_buffer("kind_lookup", torch.tensor(kinds), persistent=True); self.register_buffer("language_lookup", torch.tensor(languages), persistent=True)

            def text_memory(self, ids):
                b, n = ids.shape; pos = torch.arange(n, device=ids.device).clamp_max(config.max_text_tokens - 1)
                return self.symbol(ids) + self.kind(self.kind_lookup[ids]) + self.language(self.language_lookup[ids]) + self.text_pos(pos)[None] + self.modality.weight[0]

            def text_schedule(self, text, frame_count, schedule_frames):
                tl = text.shape[1]
                if tl == 0 or frame_count == 0:
                    return torch.zeros((text.shape[0], frame_count, config.model_dim), device=text.device)
                tpos = torch.arange(frame_count, device=text.device)
                idx = torch.div(tpos * tl, max(int(schedule_frames), 1), rounding_mode="floor").clamp_max(tl - 1)
                return self.trailing(text[:, idx])

            def frame_inputs(self, text, frames, schedule_frames=None):
                b, n, _ = frames.shape; tl = text.shape[1]
                schedule_frames = int(schedule_frames if schedule_frames is not None else n)
                trailing = self.text_schedule(text, n, schedule_frames)
                states = []
                for t in range(n):
                    if t == 0:
                        acoustic = self.control[3].view(1, 1, -1).expand(b, 1, -1)
                    else:
                        acoustic = sum(self.frame_codebooks[k](frames[:, t-1:t, k]) for k in range(16))
                    fused = (self.acoustic_gate * self.acoustic_norm(acoustic)
                             + self.linguistic_gate * self.linguistic_norm(trailing[:, t:t+1]))
                    states.append(fused + self.audio_pos.weight[min(t, config.max_audio_frames-1)].view(1,1,-1) + self.modality.weight[1])
                return torch.cat(states, dim=1) if states else torch.empty((b, 0, config.model_dim), device=text.device)

            def sequence(self, text_ids, speaker_ids, frames, schedule_frames=None):
                text = self.text_memory(text_ids); b = text.shape[0]
                controls = self.control.view(1, 4, -1).expand(b, -1, -1) + self.speaker(speaker_ids).unsqueeze(1)
                seq = torch.cat((controls, text, self.frame_inputs(text, frames, schedule_frames)), dim=1)
                mask = torch.full((seq.shape[1], seq.shape[1]), float("-inf"), device=seq.device); mask = torch.triu(mask, diagonal=1)
                hidden = self.norm(self.decoder(seq, mask=mask))
                return hidden, 4 + text.shape[1]

            def forward(self, text_ids, speaker_ids, frames, generation=False, schedule_frames=None):
                hidden, start = self.sequence(text_ids, speaker_ids, frames, schedule_frames)
                speech = hidden[:, start:]
                primary = self.primary_head(speech)
                primary_ids = frames[:, :, 0] if frames.shape[1] else torch.empty((frames.shape[0],0),dtype=torch.long,device=frames.device)
                residual = self.residual_logits(speech, primary_ids, frames[:, :, 1:] if frames.shape[1] else None)
                return primary, residual, speech

            def residual_logits(self, hidden, primary, targets=None):
                b,n,d=hidden.shape
                state=torch.tanh(hidden + self.residual_primary(primary))
                previous=torch.full((b,n), config.audio_spec.vocabulary_size, dtype=torch.long, device=hidden.device); out=[]
                for i in range(15):
                    cb=self.residual_codebook.weight[i].view(1,1,-1); state=self.residual_cell((self.residual_prev(previous)+cb).reshape(-1,d), state.reshape(-1,d)).view(b,n,d); out.append(self.residual_head(state+cb)); previous=targets[:,:,i] if targets is not None else out[-1].argmax(-1)
                return torch.stack(out,2)

            def generate_residual(self, hidden, primary):
                return self.residual_logits(hidden, primary).argmax(-1)
        return Module()
