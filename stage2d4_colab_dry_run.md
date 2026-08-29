# Stage2D.4 Colab dry-run

This package is an overlay for `stage2b4b_colab_bundle_v2.zip`. It does not
contain Qwen weights or `step025.pt`, and it has no training mode. The frozen
`step025.pt` must already be available in Drive at the path used below.

## 1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive'
```

## 2. Extract the existing Stage2B.4B bundle

```python
!rm -rf /content/stage2b4b_colab_bundle_v2
!unzip -q "$DRIVE_ROOT/swara/stage2b4b_colab_bundle_v2.zip" -d /content
BUNDLE_ROOT = '/content/stage2b4b_colab_bundle_v2'
```

## 3. Extract this Stage2D.4 overlay over the bundle

```python
!unzip -q "$DRIVE_ROOT/swara/stage2d4_colab_inputs_v2.zip" -d "$BUNDLE_ROOT"
REPO_ROOT = f'{BUNDLE_ROOT}/repo'
```

## 4. Pin the known-good dependencies

```python
%pip install -q transformers==4.57.3 tokenizers==0.22.2 huggingface_hub==0.36.2 numpy==2.2.6 numba==0.61.2 librosa==0.11.0 onnxruntime==1.29.0 soundfile==0.13.1
!apt-get update -qq && apt-get install -y -qq sox
%pip install -q -e "$REPO_ROOT/vendor/qwen3-tts"
```

## 5. Configure the frozen float32 environment

```python
import os, sys
os.environ['SWARA_STAGE2B4B_BUNDLE_ROOT'] = BUNDLE_ROOT
os.environ['SWARA_STAGE2B4B_MODEL_ROOT'] = f'{BUNDLE_ROOT}/models/qwen3_tts_0_6b_base'
os.environ['SWARA_STAGE2B4B_DEVICE_MAP'] = 'cuda:0'
os.environ['SWARA_STAGE2B4B_DTYPE'] = 'float32'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['SWARA_SPICOR_ARCHIVE'] = '/nonexistent/spicor.tar.gz'
sys.path.insert(0, REPO_ROOT + '/src')
```

## 6. Verify the frozen step025 SHA256

```python
import hashlib, pathlib
STEP025 = pathlib.Path(f'{DRIVE_ROOT}/swara/stage2b_reference/swara_stage2b_reference/checkpoint/step025.pt')
actual = hashlib.sha256(STEP025.read_bytes()).hexdigest()
expected = '2d4bb1896f17f96c38e11ce78bafb0eab060d044fcb2cd0796ba4a93d5e6969a'
assert actual == expected, (actual, expected)
print('step025 SHA256 OK:', actual)
```

## 7. Run Stage2D.4 preflight

```python
%cd $REPO_ROOT
!python scripts/preflight_stage2d4_overlay.py --checkpoint "$STEP025" --archive /nonexistent/spicor.tar.gz --output "$DRIVE_ROOT/swara/stage2d4_runs/stage2d4_v2_medium_dry_run/preflight.json"
```

## 8. Execute the bounded dry-run

```python
RUN_OUT = f'{DRIVE_ROOT}/swara/stage2d4_runs/stage2d4_v2_medium_dry_run'
!mkdir -p "$RUN_OUT"
!python scripts/run_stage2d4_bounded_training.py --dry-run --checkpoint "$STEP025" --output-dir "$RUN_OUT" --archive /nonexistent/spicor.tar.gz --probe-max-frames 24
```

The probe uses one positive, one targeted-native, and one general-native
training sample; four real teacher-forced Qwen calls (one native call for each
sample plus one conditioned call for the positive) feed one padded mixed loss.
It performs one backward and one disposable optimizer step, then restores the
original bridge/gate state. No autoregressive Qwen generation and no persistent
checkpoint are written.

## 9. Print the compact dry-run report

```python
import json, pathlib
out = pathlib.Path(RUN_OUT)
for name in ('dry_run_status.json', 'dry_run_dataset_summary.json',
             'dry_run_loss_report.json', 'dry_run_gradient_report.json',
             'dry_run_qwen_freeze_report.json', 'dry_run_evaluation_report.json',
             'dry_run_environment.json'):
    print(f'\n=== {name} ===')
    print(json.dumps(json.loads((out / name).read_text()), indent=2, sort_keys=True))
assert not list(out.glob('*.pt')), 'dry-run must not create a checkpoint'
```
