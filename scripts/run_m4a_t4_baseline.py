"""Inference-only Qwen3-TTS 0.6B Base baseline runner for a Colab T4."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

SENTENCES = {
    "001": "The morning train arrived exactly on time.",
    "002": "I travelled from Hyderabad to Visakhapatnam last weekend.",
    "003": "Saikiran walked through the streets of Madhapur after dinner.",
    "004": "The family planned a journey from Bengaluru to Thiruvananthapuram.",
    "005": "For a moment, nobody spoke, and the entire room became completely silent in Rajahmundry.",
}

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/T4 is required for M4A_T4")
    print("CUDA", torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
    out = Path("/content/drive/MyDrive/swara/m4a_qwen_foundation_baseline")
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path("/content/qwen3-tts-12hz-0.6b-base")
    reference = Path("/content/m3c_reference_001.wav")
    load_start = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(str(model_path), local_files_only=True, device_map="cuda", dtype=torch.float16)
    load_seconds = time.monotonic() - load_start
    rows = []
    for sid, text in SENTENCES.items():
        started = time.monotonic()
        wavs, sample_rate = model.generate_voice_clone(text=text, language="English", ref_audio=str(reference), ref_text=SENTENCES["001"], x_vector_only_mode=False, do_sample=False, max_new_tokens=512)
        samples = np.asarray(wavs[0], dtype=np.float32)
        if samples.size == 0 or not np.isfinite(samples).all() or int(sample_rate) != 24000:
            raise RuntimeError(f"invalid output {sid}")
        path = out / f"{sid}.wav"
        sf.write(path, samples, int(sample_rate), subtype="PCM_16")
        rows.append({"sentence_id": sid, "text": text, "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "revision": "main", "generation_settings": {"language": "English", "do_sample": False, "max_new_tokens": 512}, "reference_voice": "m3c_reference_001.wav", "output_path": str(path), "duration_seconds": float(samples.size / sample_rate), "generation_seconds": time.monotonic() - started})
        print(sid, rows[-1]["generation_seconds"], rows[-1]["duration_seconds"])
    (out / "baseline_manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary = {"cuda": torch.cuda.get_device_name(0), "model_dtype": "float16", "model_load_seconds": load_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(), "training": False}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("M4A_T4 PASS", summary)

if __name__ == "__main__":
    main()
