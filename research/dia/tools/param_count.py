"""
Static parameter-count estimator for the Dia 1.6B architecture.

Derives parameter counts directly from dia/config.py default values and the
module shapes defined in dia/layers.py (DenseGeneral, RMSNorm, nn.Embedding).
Does NOT download or load any model weights -- it only replicates the shape
arithmetic that the real modules perform at __init__ time.

Run: python param_count.py
"""

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    vocab_size: int = 256


@dataclass
class DecoderConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    cross_hidden_size: int = 1024
    cross_num_attention_heads: int = 16
    cross_num_key_value_heads: int = 16
    cross_head_dim: int = 128
    vocab_size: int = 1028
    num_channels: int = 9


def dense_general(in_features: int, *out_features: int) -> int:
    """weight shape = in_shapes + out_features, no bias (DenseGeneral has no bias param)."""
    n = in_features
    for f in out_features:
        n *= f
    return n


def self_attn_params(q_dim: int, kv_dim: int, n_q: int, n_kv: int, head_dim: int, out_dim: int) -> int:
    q = dense_general(q_dim, n_q, head_dim)
    k = dense_general(kv_dim, n_kv, head_dim)
    v = dense_general(kv_dim, n_kv, head_dim)
    o = dense_general(n_q * head_dim, out_dim)
    return q + k + v + o


def mlp_params(embed_dim: int, intermediate_dim: int) -> int:
    wi_fused = dense_general(embed_dim, 2, intermediate_dim)
    wo = dense_general(intermediate_dim, embed_dim)
    return wi_fused + wo


def rmsnorm_params(dim: int) -> int:
    return dim


def encoder_params(cfg: EncoderConfig) -> dict:
    embed = cfg.vocab_size * cfg.hidden_size

    attn_per_layer = self_attn_params(
        q_dim=cfg.hidden_size,
        kv_dim=cfg.hidden_size,
        n_q=cfg.num_attention_heads,
        n_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        out_dim=cfg.hidden_size,
    )
    mlp_per_layer = mlp_params(cfg.hidden_size, cfg.intermediate_size)
    norm_per_layer = 2 * rmsnorm_params(cfg.hidden_size)  # pre_sa_norm, post_sa_norm

    per_layer = attn_per_layer + mlp_per_layer + norm_per_layer
    all_layers = per_layer * cfg.num_hidden_layers
    final_norm = rmsnorm_params(cfg.hidden_size)

    total = embed + all_layers + final_norm
    return {
        "embedding": embed,
        "attn_per_layer": attn_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "norm_per_layer": norm_per_layer,
        "per_layer_total": per_layer,
        "num_layers": cfg.num_hidden_layers,
        "all_layers_total": all_layers,
        "final_norm": final_norm,
        "total": total,
    }


def decoder_params(cfg: DecoderConfig) -> dict:
    embed = cfg.num_channels * cfg.vocab_size * cfg.hidden_size

    self_attn_per_layer = self_attn_params(
        q_dim=cfg.hidden_size,
        kv_dim=cfg.hidden_size,
        n_q=cfg.num_attention_heads,
        n_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        out_dim=cfg.hidden_size,
    )
    cross_attn_per_layer = self_attn_params(
        q_dim=cfg.hidden_size,
        kv_dim=cfg.cross_hidden_size,
        n_q=cfg.cross_num_attention_heads,
        n_kv=cfg.cross_num_key_value_heads,
        head_dim=cfg.cross_head_dim,
        out_dim=cfg.hidden_size,
    )
    mlp_per_layer = mlp_params(cfg.hidden_size, cfg.intermediate_size)
    norm_per_layer = 3 * rmsnorm_params(cfg.hidden_size)  # pre_sa, pre_ca, pre_mlp

    per_layer = self_attn_per_layer + cross_attn_per_layer + mlp_per_layer + norm_per_layer
    all_layers = per_layer * cfg.num_hidden_layers
    final_norm = rmsnorm_params(cfg.hidden_size)
    logits_dense = dense_general(cfg.hidden_size, cfg.num_channels, cfg.vocab_size)

    total = embed + all_layers + final_norm + logits_dense
    return {
        "embedding_9_channels": embed,
        "self_attn_per_layer": self_attn_per_layer,
        "cross_attn_per_layer": cross_attn_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "norm_per_layer": norm_per_layer,
        "per_layer_total": per_layer,
        "num_layers": cfg.num_hidden_layers,
        "all_layers_total": all_layers,
        "final_norm": final_norm,
        "logits_dense": logits_dense,
        "total": total,
    }


def fmt(n: int) -> str:
    return f"{n:,}  ({n / 1e6:.2f}M)"


if __name__ == "__main__":
    enc = encoder_params(EncoderConfig())
    dec = decoder_params(DecoderConfig())
    total = enc["total"] + dec["total"]

    print("=" * 70)
    print("DIA PARAMETER COUNT (derived from dia/config.py defaults + dia/layers.py shapes)")
    print("=" * 70)

    print("\n--- ENCODER (12 layers, hidden=1024) ---")
    for k, v in enc.items():
        print(f"  {k:22s}: {fmt(v) if isinstance(v, int) and k != 'num_layers' else v}")

    print("\n--- DECODER (18 layers, hidden=2048) ---")
    for k, v in dec.items():
        print(f"  {k:22s}: {fmt(v) if isinstance(v, int) and k != 'num_layers' else v}")

    print("\n--- TOTALS ---")
    print(f"  Encoder total         : {fmt(enc['total'])}")
    print(f"  Decoder total         : {fmt(dec['total'])}")
    print(f"  MODEL TOTAL           : {fmt(total)}")
    print(f"  Encoder share         : {100 * enc['total'] / total:.1f}%")
    print(f"  Decoder share         : {100 * dec['total'] / total:.1f}%")

    print("\n--- DECODER BREAKDOWN BY SUBSYSTEM ---")
    dec_self_attn_total = dec["self_attn_per_layer"] * dec["num_layers"]
    dec_cross_attn_total = dec["cross_attn_per_layer"] * dec["num_layers"]
    dec_mlp_total = dec["mlp_per_layer"] * dec["num_layers"]
    dec_norm_total = dec["norm_per_layer"] * dec["num_layers"] + dec["final_norm"]
    print(f"  Self-attention (all layers)  : {fmt(dec_self_attn_total)}  [{100*dec_self_attn_total/total:.1f}% of model]")
    print(f"  Cross-attention (all layers) : {fmt(dec_cross_attn_total)}  [{100*dec_cross_attn_total/total:.1f}% of model]")
    print(f"  MLP / FFN (all layers)       : {fmt(dec_mlp_total)}  [{100*dec_mlp_total/total:.1f}% of model]")
    print(f"  Per-codebook embeddings (9x) : {fmt(dec['embedding_9_channels'])}  [{100*dec['embedding_9_channels']/total:.1f}% of model]")
    print(f"  Logits projection (9x1028)   : {fmt(dec['logits_dense'])}  [{100*dec['logits_dense']/total:.1f}% of model]")
    print(f"  Norms                        : {fmt(dec_norm_total)}  [{100*dec_norm_total/total:.2f}% of model]")

    print("\n--- ENCODER BREAKDOWN BY SUBSYSTEM ---")
    enc_attn_total = enc["attn_per_layer"] * enc["num_layers"]
    enc_mlp_total = enc["mlp_per_layer"] * enc["num_layers"]
    print(f"  Self-attention (all layers)  : {fmt(enc_attn_total)}  [{100*enc_attn_total/total:.1f}% of model]")
    print(f"  MLP / FFN (all layers)       : {fmt(enc_mlp_total)}  [{100*enc_mlp_total/total:.1f}% of model]")
    print(f"  Text embedding                : {fmt(enc['embedding'])}  [{100*enc['embedding']/total:.2f}% of model]")

    print("\nNOTE: This counts only dia/layers.py DiaModel parameters (encoder+decoder).")
    print("It EXCLUDES the DAC (Descript Audio Codec) model, which is a separate")
    print("pretrained network loaded at runtime and is not part of the 1.6B figure.")
