"""Small, fair N1-A/N1-B token-prediction models.

Both heads share the exact same causal frame backbone.  This is a falsification
model, not a production speech generator.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from swara.codecs.neucodec_fsq import fsq_to_token_ids


@dataclass(frozen=True)
class N1Config:
    linguistic_vocab_size: int
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    ffn_dim: int = 512
    max_text_tokens: int = 512
    max_frames: int = 4096
    token_cardinality: int = 65536


class N1Backbone(nn.Module):
    def __init__(self, config: N1Config):
        super().__init__(); self.config = config
        self.token_embedding = nn.Embedding(config.linguistic_vocab_size, config.d_model)
        self.frame_position = nn.Embedding(config.max_frames, config.d_model)
        layer = nn.TransformerEncoderLayer(config.d_model, config.heads, config.ffn_dim, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers)
        self.norm = nn.LayerNorm(config.d_model)

    def frame_inputs(self, linguistic_ids: torch.Tensor, frames: int) -> torch.Tensor:
        b, n = linguistic_ids.shape
        if frames > self.config.max_frames: raise ValueError("frame budget exceeds max_frames")
        pos = torch.arange(frames, device=linguistic_ids.device)
        text_pos = torch.div(pos * n, frames, rounding_mode="floor").clamp_max(n - 1)
        aligned = self.token_embedding(linguistic_ids[:, text_pos])
        return aligned + self.frame_position(pos)[None, :, :]

    def forward(self, linguistic_ids: torch.Tensor, frames: int) -> torch.Tensor:
        x = self.frame_inputs(linguistic_ids, frames)
        mask = torch.triu(torch.ones(frames, frames, device=x.device, dtype=torch.bool), diagonal=1)
        return self.norm(self.transformer(x, mask=mask))


class N1Flat(nn.Module):
    def __init__(self, config: N1Config):
        super().__init__(); self.backbone = N1Backbone(config); self.head = nn.Linear(config.d_model, config.token_cardinality)
    def forward(self, linguistic_ids, targets=None):
        frames = targets.shape[1] if targets is not None else 1
        hidden = self.backbone(linguistic_ids, frames); logits = self.head(hidden)
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss, hidden
    def generate(self, linguistic_ids, frames):
        with torch.no_grad(): return self.head(self.backbone(linguistic_ids, frames)).argmax(-1)


class N1FSQ(nn.Module):
    def __init__(self, config: N1Config):
        super().__init__(); self.backbone = N1Backbone(config); self.heads = nn.ModuleList(nn.Linear(config.d_model, 4) for _ in range(8))
    def forward(self, linguistic_ids, targets=None):
        frames = targets.shape[1] if targets is not None else 1
        hidden = self.backbone(linguistic_ids, frames); logits = torch.stack([head(hidden) for head in self.heads], dim=2)
        loss = None if targets is None else torch.stack([F.cross_entropy(logits[:,:,i,:].reshape(-1,4), targets[:,:,i].reshape(-1)) for i in range(8)]).mean()
        return logits, loss, hidden
    def generate(self, linguistic_ids, frames):
        with torch.no_grad():
            hidden = self.backbone(linguistic_ids, frames)
            coords = torch.stack([head(hidden).argmax(-1) for head in self.heads], dim=-1)
            return fsq_to_token_ids(coords)


def parameter_counts(model: nn.Module) -> tuple[int, int, int]:
    head = model.head if hasattr(model, "head") else model.heads
    hp = sum(p.numel() for p in head.parameters()); total = sum(p.numel() for p in model.parameters())
    return total - hp, hp, total
