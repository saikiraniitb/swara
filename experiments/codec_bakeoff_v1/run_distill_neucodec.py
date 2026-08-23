"""Run the authenticated Distill-NeuCodec roundtrip for the fixed bake-off panel."""
from __future__ import annotations

import json, random, sys, time, types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def install_rotary_import_shim() -> None:
    """Avoid torchtune's optional torchao import on the project's torch build."""
    import importlib.util
    source = Path(torch.__file__).parent.parent / "torchtune/modules/position_embeddings.py"
    if not source.exists():
        source = Path(__file__).parents[2] / ".venv/lib/python3.14/site-packages/torchtune/modules/position_embeddings.py"
    text = source.read_text()
    ns = {"torch": torch, "nn": torch.nn, "Any": object, "Optional": object}
    exec(text[text.index("class RotaryPositionalEmbeddings"):], ns)
    mod = types.ModuleType("torchtune.modules")
    mod.RotaryPositionalEmbeddings = ns["RotaryPositionalEmbeddings"]
    sys.modules["torchtune.modules"] = mod


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(x).all())
    if not finite or x.size == 0:
        return {"samples": int(x.size), "finite": finite}
    return {
        "samples": int(x.size), "finite": True,
        "duration_seconds": float(x.size / 24000),
        "rms": float(np.sqrt(np.mean(x * x))),
        "peak": float(np.max(np.abs(x))),
        "clipping_count": int(np.sum(np.abs(x) >= 0.999)),
        "dc_offset": float(np.mean(x)),
    }


def sisdr(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b)); a = a[:n].astype(np.float64); b = b[:n].astype(np.float64)
    if n == 0: return None
    a -= a.mean(); b -= b.mean()
    den = float(np.dot(a, a))
    if den <= 1e-12: return None
    target = float(np.dot(b, a) / den) * a
    noise = b - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def main() -> None:
    root = Path(__file__).parents[2]
    panel = root / "experiments/codec_bakeoff_v1/manifest.json"
    out_dir = root / "experiments/codec_bakeoff_v1/distill_neucodec"
    listening = root / "experiments/codec_bakeoff_v1/listening_distill_neucodec"
    reports = root / "experiments/codec_bakeoff_v1/reports"
    out_dir.mkdir(exist_ok=True); listening.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    manifest = json.loads(panel.read_text())

    install_rotary_import_shim()
    from neucodec import DistillNeuCodec
    from huggingface_hub import model_info
    info = model_info("neuphonic/distill-neucodec", revision="daee7fd9989a62594084fd8e1a99e61beb5b0e85")
    model = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec", revision=info.sha).eval()
    model.to("cpu")
    params = sum(p.numel() for p in model.parameters())
    q = model.generator.quantizer
    qinfo = {k: str(v) for k, v in vars(q).items() if k in ("levels", "dim", "codebook_size")}
    print("loaded", info.sha, "params", params, "sample_rate", model.sample_rate, "hop", model.hop_length, "quantizer", qinfo)

    rows=[]; t_all=time.perf_counter()
    for i, item in enumerate(manifest["clips"], 1):
        src = root / item["copied_audio"]
        wav, sr = sf.read(src, dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        t0=time.perf_counter()
        # The tensor API intentionally assumes 16 kHz; perform the same
        # conversion explicitly while avoiding torchaudio's optional file
        # backend on this environment.
        audio16 = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), int(sr), 16000)
        with torch.inference_mode(): codes = model.encode_code(audio16.unsqueeze(0))
        enc_t=time.perf_counter()-t0
        t1=time.perf_counter()
        with torch.inference_mode(): recon = model.decode_code(codes)
        dec_t=time.perf_counter()-t1
        c = codes.detach().cpu().numpy().astype(np.int64)
        y = recon.detach().cpu().numpy().reshape(-1).astype(np.float32)
        out = out_dir / f"{i:02d}_{item['utterance_id']}.wav"; sf.write(out, y, model.sample_rate, subtype="PCM_16")
        src_st=stats(wav); out_st=stats(y)
        n=min(len(wav),len(y)); corr=float(np.corrcoef(wav[:n],y[:n])[0,1]) if n>1 else None
        rows.append({"index":i,"utterance_id":item["utterance_id"],"category":item["category"],"source_path":str(src.relative_to(root)),"output_path":str(out.relative_to(root)),"input_sample_rate":int(sr),"output_sample_rate":int(model.sample_rate),"source_stats":src_st,"output_stats":out_st,"encoded_shape":list(c.shape),"frames":int(c.shape[-1]),"codebooks":int(c.shape[1]),"token_min":int(c.min()),"token_max":int(c.max()),"token_unique":int(np.unique(c).size),"encode_seconds":enc_t,"decode_seconds":dec_t,"encode_rtf":enc_t/max(src_st.get("duration_seconds",1),1e-9),"decode_rtf":dec_t/max(src_st.get("duration_seconds",1),1e-9),"aligned_correlation":corr,"si_sdr_db":sisdr(wav,y)})
        print(i, c.shape, enc_t, dec_t, flush=True)

    random.seed(20260823); mapping={};
    for i in range(1,21):
        src = (root / manifest["clips"][i-1]["copied_audio"]); rec = out_dir / f"{i:02d}_{manifest['clips'][i-1]['utterance_id']}.wav"
        pair=[("original",src), ("distill_neucodec",rec)]; random.shuffle(pair)
        mapping[f"sample_{i:02d}"]={"A":pair[0][0],"B":pair[1][0]}
        for label, path in [("A",pair[0][1]),("B",pair[1][1])]:
            import shutil; shutil.copy2(path, listening/f"sample_{i:02d}_{label}.wav")
        (listening/f"sample_{i:02d}.txt").write_text(manifest["clips"][i-1]["source_text"]+"\n")
    (reports/"distill_neucodec_blind_mapping.json").write_text(json.dumps({"seed":20260823,"status":"HIDDEN_MAPPING","mapping":mapping},indent=2)+"\n")
    report={"status":"complete","model_id":"neuphonic/distill-neucodec","revision":info.sha,"license":"apache-2.0","parameter_count":params,"sample_rate":model.sample_rate,"hop_length":model.hop_length,"quantizer":qinfo,"clips":rows,"total_seconds":sum(r["source_stats"]["duration_seconds"] for r in rows),"wall_seconds":time.perf_counter()-t_all}
    (reports/"distill_neucodec_metrics.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__ == "__main__": main()
