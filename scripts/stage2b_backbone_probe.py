"""Local, read-only Stage2B.3A capability probe.

This probe inspects the checked-in Qwen checkpoint metadata and, optionally,
the local fast tokenizer.  It never downloads, loads model weights into a
neural module, generates audio, or changes repository state.  The bakeoff
artifact is kept separate from production adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parameter_count(path: Path) -> int | None:
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    total = 0
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            count = 1
            for dimension in tensors.get_slice(key).get_shape():
                count *= dimension
            total += count
    return total


def probe_qwen_local(model_dir: str | Path, include_tokenizer: bool = False) -> dict[str, Any]:
    """Return concise metadata for an already-local Qwen3-TTS Base asset."""

    root = Path(model_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    talker = config["talker_config"]
    predictor = talker["code_predictor_config"]
    required = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
    )
    result: dict[str, Any] = {
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "local_path": str(root),
        "local_asset_complete": all((root / item).exists() for item in required),
        "missing_files": [item for item in required if not (root / item).exists()],
        "config": {
            "architectures": config.get("architectures"),
            "tts_model_type": config.get("tts_model_type"),
            "tts_model_size": config.get("tts_model_size"),
            "tokenizer_type": config.get("tokenizer_type"),
            "speaker_encoder_dim": config["speaker_encoder_config"]["enc_dim"],
            "speaker_encoder_sample_rate_hz": config["speaker_encoder_config"]["sample_rate"],
            "talker_hidden_size": talker["hidden_size"],
            "talker_text_hidden_size": talker["text_hidden_size"],
            "talker_text_vocab_size": talker["text_vocab_size"],
            "talker_layers": talker["num_hidden_layers"],
            "talker_attention_heads": talker["num_attention_heads"],
            "talker_kv_heads": talker["num_key_value_heads"],
            "talker_num_code_groups": talker["num_code_groups"],
            "talker_position_id_per_seconds": talker["position_id_per_seconds"],
            "talker_max_position_embeddings": talker["max_position_embeddings"],
            "code_predictor_hidden_size": predictor["hidden_size"],
            "code_predictor_layers": predictor["num_hidden_layers"],
            "code_predictor_vocab_size": predictor["vocab_size"],
        },
        "model_tensor_parameter_count": _parameter_count(root / "model.safetensors"),
        "probe_scope": "metadata-only; no model weights loaded and no network access",
    }

    if include_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, use_fast=True)
        examples = []
        for text in ("Kolkata hosted the conference.", "Ajinkya travelled to Bengaluru."):
            encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
            examples.append(
                {
                    "text": text,
                    "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
                    "offsets": [list(offset) for offset in encoded["offset_mapping"]],
                }
            )
        result["tokenizer_probe"] = {
            "class": type(tokenizer).__name__,
            "offset_unit": "fast-tokenizer character offsets; validated against Python strings for this probe",
            "examples": examples,
            "current_qwen_inference_helper_retains_offsets": False,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tokenizer", action="store_true")
    args = parser.parse_args()
    print(json.dumps(probe_qwen_local(args.model_dir, include_tokenizer=args.tokenizer), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
