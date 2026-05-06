# Mobile Gaze Estimation Pipeline

A research pipeline for offline gaze estimation from mobile face recordings.
Built on top of [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) (L2CS-Net style ResNet/MobileNet models).

## Overview

This project has two components:

1. **Data collection** — a React Native app (`GazeAppRN/`, Android) that displays calibration and validation dots on screen while recording the front camera.
2. **Offline pipeline** — Python scripts that take a recorded session, run face-based gaze inference, fit a per-user polynomial calibration, and report accuracy.

---

## Project Structure

```
gaze-estimation/
├── pipeline/                   # Mobile gaze pipeline (active development)
│   ├── gaze_utils.py           # Shared utilities: gaze model, face detector, calibration fitting
│   ├── run_calibration.py      # Phase 1: fit calibration model, output residual plot
│   ├── run_validation.py       # Phase 2: re-fit calibration + evaluate on held-out points
│   └── visualizer.py           # Debug viewer: play back session video with event overlay
│
├── tools/                      # Standalone desktop utilities
│   ├── inference.py            # Webcam / video gaze inference (real-time display)
│   ├── onnx_export.py          # Export PyTorch weights → ONNX
│   ├── onnx_inference.py       # ONNX runtime inference
│   ├── calibration_demo.py     # Quick single-image calibration demo
│   ├── convert_to_coreml.py    # Convert model → Core ML (iOS deployment)
│   └── convert_resnet34_coreml.py
│
├── models/                     # Model architectures (ResNet, MobileNet, MobileOne)
├── utils/                      # Dataset loaders and training helpers
├── weights/                    # Model weight files (*.pt, *.onnx — not in git)
├── data/                       # Training datasets (not in git)
│   ├── Gaze360/
│   └── MPIIFaceGaze/
│
├── GazeAppRN/                  # React Native data-collection app (Android)
├── GazeData/                   # Recorded sessions (not in git — excluded by .gitignore)
│
├── main.py                     # Training entry point
├── evaluate.py                 # Dataset evaluation
├── run.sh                      # Pipeline launcher (recommended entry point)
├── requirements.txt
└── MobileDataPipeline.markdown # Detailed pipeline design document
```

> `GazeData/` is excluded from git (`.gitignore`) — session recordings are large and local-only.
> `GazeAppRN/` is tracked by git and can be cloned normally.

---

## Quick Start — Pipeline

All pipeline commands go through `run.sh`:

```bash
# 1. Verify timestamp alignment before running inference
bash run.sh viz --session GazeData/session_xxx

# 2. Calibration only (inspect residual plot)
bash run.sh cal --session GazeData/session_xxx --weight weights/resnet34.pt

# 3. Full calibration + validation (recommended)
bash run.sh val --session GazeData/session_xxx --weight weights/resnet34.pt
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--session` | (required) | Path to session folder |
| `--weight` | `weights/resnet34.pt` | Path to model weights |
| `--dataset` | `gaze360` | Dataset config: `gaze360` or `mpiigaze` |
| `--stride` | `1` | Process every Nth frame (e.g. `3` = 3× faster) |
| `--skip-ms` | `1000` | ms to skip after each point_start (saccade + settling) |
| `--end-trim-ms` | `1000` | ms to trim before each point_end (anticipatory saccade) |
| `--verbose` | off | Enable DEBUG logging |

### Example with MPIIFaceGaze model

```bash
bash run.sh val \
    --session GazeData/session_xxx \
    --weight  weights/resnet34_mpiigaze.pt \
    --dataset mpiigaze
```

---

## Session Folder Format

Each recorded session from `GazeAppRN` produces:

```
GazeData/session_<timestamp>_<id>/
├── metadata.json               # Device info, screen size, camera fps
├── calibration.mp4             # Face video during calibration phase
├── calibration_events.csv      # Timestamps + target positions for each calibration dot
├── validation.mp4              # Face video during validation phase
└── validation_events.csv       # Timestamps + target positions for each validation dot
```

The pipeline uses actual video FPS from the container (not metadata), because mobile encoders often deliver a lower frame rate than the camera target (e.g. Galaxy A10 targets 30 fps but encodes at ~8.6 fps).

---

## Visualizer

```bash
bash run.sh viz --session GazeData/session_xxx
```

Opens an interactive OpenCV window that plays back the face recording for each calibration point, with overlaid information to verify that video timestamps align with gaze events before running inference.

**What you see on screen:**

- Top-left overlay: current point index, target position (normalized + pixels), time window, playback state
- Bottom-right: current frame timestamp in milliseconds (session-relative)
- Bottom-left: mini schematic of the phone screen with a dot showing where the target was
- **Blue tint overlay (start)**: "skip zone" — covers the first `skip_ms` (default 1000 ms) after each point starts. Excluded from inference because the subject's eyes are still moving from the previous target (saccade + settling).
- **Blue tint overlay (end)**: "end trim zone" — covers the last `end_trim_ms` (default 1000 ms) before each point ends. Excluded from inference because the subject's eyes start drifting toward the next target (anticipatory saccade). The clear frames between the two blue zones are what the model actually uses.

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Play / pause |
| `[` or `←` | Previous point |
| `]` or `→` | Next point |
| `0`–`9` | Jump directly to point 0–9 |
| `a`–`f` | Jump directly to point 10–15 |
| `Q` | Quit |

**How to use it:**

1. Run `viz` first before any inference.
2. Press `]` to step through each calibration point.
3. For each point: press `SPACE` to play. The clear window between the two blue zones is what gets used for inference. If the eyes are still moving when the start blue ends, increase `--skip-ms`. If the eyes start drifting before the end blue begins, increase `--end-trim-ms`.
4. If the video jumps to a completely wrong moment, there is an FPS mismatch — check that `metadata.json` matches what `get_video_fps()` reads from the container.

---

## Training

Models are trained with `main.py`. To train ResNet-34 on MPIIFaceGaze (recommended for screen-gaze scenarios):

```bash
python main.py \
    --data data/MPIIFaceGaze \
    --dataset mpiigaze \
    --arch resnet34 \
    --num-epochs 50 \
    --batch-size 64
```

To resume from a checkpoint:

```bash
python main.py --data data/MPIIFaceGaze --dataset mpiigaze --arch resnet34 \
    --checkpoint output/epoch_30.pt
```

### Dataset setup

```
data/
├── Gaze360/
│   ├── Image/
│   └── Label/
└── MPIIFaceGaze/
    ├── Image/
    └── Label/
```

- **Gaze360**: https://gaze360.csail.mit.edu/download.php — ±180° range, good for general gaze; coarse bins for phone screen use.
- **MPIIFaceGaze**: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation — trained on screen-reading scenarios, recommended for this pipeline.

Pre-processing scripts: https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/

---

## Pre-trained Weights (Gaze360)

| Model | PyTorch | ONNX | MAE (°) |
|-------|---------|------|---------|
| ResNet-18 | [resnet18.pt](https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18.pt) | [resnet18_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18_gaze.onnx) | 12.84 |
| ResNet-34 | [resnet34.pt](https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34.pt) | [resnet34_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34_gaze.onnx) | 11.33 |
| ResNet-50 | [resnet50.pt](https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet50.pt) | [resnet50_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet50_gaze.onnx) | 11.34 |
| MobileNet V2 | [mobilenetv2.pt](https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2.pt) | [mobilenetv2_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2_gaze.onnx) | 13.07 |

MAE = Mean Absolute Error in degrees on Gaze360 test set.

---

## Environment

```bash
conda activate DeepLearning
# or directly:
/opt/anaconda3/envs/DeepLearning/bin/python pipeline/run_validation.py ...
```

---

## Upstream

This project extends [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation), which is built on [L2CS-Net](https://github.com/Ahmednull/L2CS-Net).
Face detection uses [uniface](https://github.com/yakhyo/uniface) (RetinaFace).
