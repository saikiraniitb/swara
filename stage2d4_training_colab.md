# Stage2D.4 final training overlay

This overlay is applied over `stage2b4b_colab_bundle_v2.zip`. It contains the
frozen Stage2D.4 source, exact normalized training audio, baseline evaluator,
and training runner. The Qwen/model bundle and `step025.pt` remain external.

## 1. Mount Drive and extract the existing bundle

```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive'
!rm -rf /content/stage2b4b_colab_bundle_v2
!unzip -q "$DRIVE_ROOT/swara/stage2b4b_colab_bundle_v2.zip" -d /content
BUNDLE_ROOT = '/content/stage2b4b_colab_bundle_v2'
```

## 2. Extract the final Stage2D.4 overlay

```python
!unzip -q "$DRIVE_ROOT/swara/stage2d4_training_colab_inputs_v1.zip" -d "$BUNDLE_ROOT"
REPO_ROOT = f'{BUNDLE_ROOT}/repo'
```

## 3. Pin the known-good environment

```python
%pip install -q transformers==4.57.3 tokenizers==0.22.2 huggingface_hub==0.36.2 numpy==2.2.6 numba==0.61.2 librosa==0.11.0 onnxruntime==1.29.0 soundfile==0.13.1
!apt-get update -qq && apt-get install -y -qq sox
%pip install -q -e "$REPO_ROOT/vendor/qwen3-tts"
import os, sys
os.environ['SWARA_STAGE2B4B_BUNDLE_ROOT'] = BUNDLE_ROOT
os.environ['SWARA_STAGE2B4B_MODEL_ROOT'] = f'{BUNDLE_ROOT}/models/qwen3_tts_0_6b_base'
os.environ['SWARA_STAGE2B4B_DEVICE_MAP'] = 'cuda:0'
os.environ['SWARA_STAGE2B4B_DTYPE'] = 'float32'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, REPO_ROOT + '/src')
```

## 4. Verify step025 and run preflight

```python
import hashlib, pathlib
STEP025 = pathlib.Path(f'{DRIVE_ROOT}/swara/stage2b_reference/swara_stage2b_reference/checkpoint/step025.pt')
assert hashlib.sha256(STEP025.read_bytes()).hexdigest() == '2d4bb1896f17f96c38e11ce78bafb0eab060d044fcb2cd0796ba4a93d5e6969a'
%cd $REPO_ROOT
!python scripts/preflight_stage2d4_overlay.py --checkpoint "$STEP025" --archive /nonexistent/spicor.tar.gz
```

## 5. Generate the untouched step025 baseline

```python
BASELINE_OUT = f'{DRIVE_ROOT}/swara/stage2d4_runs/stage2d4_v1_medium_baseline_step025'
!python scripts/run_stage2d4_baseline_evaluation.py --baseline --checkpoint "$STEP025" --output-dir "$BASELINE_OUT"
```

This mode is evaluation-only: no optimizer is created, no backward is called,
and no trainable checkpoint is written. It records 128 frozen fixtures with
teacher-forced q0 metrics, autoregressive trajectory metrics, and deterministic
PCM16 waveform outputs.

## 6. Run the frozen 64-step training only when explicitly authorized

```python
TRAIN_OUT = f'{DRIVE_ROOT}/swara/stage2d4_runs/stage2d4_v1_medium_training'
!python scripts/run_stage2d4_bounded_training.py --train --checkpoint "$STEP025" --output-dir "$TRAIN_OUT" --archive /nonexistent/spicor.tar.gz
```

The frozen schedule is batch size 8, four epochs, 64 optimizer steps, gate-only
steps 1–5, then bridge+gate, with checkpoints at steps 000, 005, 032, and 064.
Each checkpoint excludes Qwen weights and includes the dataset, evaluation
contract, source, and step025 provenance hashes.
