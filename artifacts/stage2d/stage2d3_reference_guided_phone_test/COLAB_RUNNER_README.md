# Stage2D.3A Colab runner

Local Mac generation was not run because the frozen Qwen foundation is not
available locally. The prepared runner uses the existing Stage2B/Stage2C
runtime and must be run only where the bundle model is already present.

```python
import os, sys, subprocess
ROOT = "/content/stage2b4b_colab_bundle_v2"
os.environ["SWARA_STAGE2B4B_BUNDLE_ROOT"] = ROOT
os.environ["SWARA_STAGE2B4B_MODEL_ROOT"] = ROOT + "/models/qwen3_tts_0_6b_base"
os.environ["SWARA_STAGE2B4B_REFERENCE_AUDIO"] = ROOT + "/data/source_audio/IISc_SPICORProject_EN_M_AGRI_116.wav"
os.environ["SWARA_STAGE2B4B_DEVICE_MAP"] = "cuda"
os.environ["SWARA_STAGE2B4B_DTYPE"] = "float32"
os.environ["PYTHONPATH"] = ROOT + "/repo/src"
subprocess.run([
    sys.executable,
    ROOT + "/repo/scripts/run_stage2d3a_reference_guided_phone_test.py",
    "--checkpoint",
    ROOT + "/run_artifacts/stage2b4b_pronunciation_v0/checkpoints/step025.pt",
    "--output-dir",
    ROOT + "/artifacts/stage2d/stage2d3_reference_guided_phone_test",
], check=True)
```

The runner writes native and candidate WAVs, `stage2d3a_manifest.json`,
`stage2d3a_report.json`, `stage2d3a_blinding_map.json`, and
`human_review.html`. It uses `target_context_1`, float32, deterministic
decoding, and the frozen step025 bridge. It does not train or write a
checkpoint.
