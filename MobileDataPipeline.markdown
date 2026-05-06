# Mobile Gaze Estimation — Offline Data Pipeline

## 1. Project Overview

This pipeline processes face video and event logs recorded by a mobile React Native app to run offline gaze estimation on a desktop machine. The mobile app (data collection tool, already complete) records the front camera during three phases: calibration, validation, and experiment. Desktop inference uses a ResNet34 gaze model to predict pitch/yaw angles from face crops, then maps them to screen coordinates via polynomial regression.

**Current scope: Phases 1 & 2 (Calibration + Validation) only. Experiment phase is deferred.**

High-level flow:

```
Session folder
  ├── metadata.json
  ├── calibration.mp4 + calibration_events.csv  →  fit polynomial gaze model
  ├── validation.mp4  + validation_events.csv    →  evaluate model accuracy
  └── experiment.mp4  + experiment_events.csv    →  [deferred]
```

---

## 2. Input Data Reference

### 2.1 Session Folder Layout

Each recording session produces a folder `session_YYYYMMDD_HHMMSS_XXXX/` under `GazeData/` containing 7 files:

```
GazeData/
└── session_20260504_210116_THH0FFPA/
    ├── metadata.json
    ├── calibration.mp4
    ├── calibration_events.csv
    ├── validation.mp4
    ├── validation_events.csv
    ├── experiment.mp4
    └── experiment_events.csv
```

### 2.2 metadata.json

Device and session metadata recorded at the start of each session.

```json
{
  "session_id": "THH0FFPA",
  "platform": "android",
  "os_version": "28",
  "camera_resolution": "1920x1080",
  "camera_fps": 30,
  "screen_width_px": 720,
  "screen_height_px": 1520,
  "screen_scale": 1.75,
  "created_at": "2026-05-05T01:01:16.037Z"
}
```

| Field | Description |
|---|---|
| `session_id` | Unique session identifier |
| `camera_resolution` | Hardware camera resolution in landscape notation (`1920x1080`). The actual face video is stored **portrait** (1080 wide × 1920 tall). No rotation needed before processing. |
| `camera_fps` | Frame rate (30 fps) |
| `screen_width_px`, `screen_height_px` | **Total physical screen** dimensions, including the Android navigation bar. **Do not use for gaze mapping.** See §2.5. |

### 2.3 calibration_events.csv / validation_events.csv

Shared format. Columns:

| Column | Description |
|---|---|
| `elapsed_ms` | Milliseconds since session start (the moment `markStart()` was called in the app). Used to align with video frames. |
| `event_type` | `session_start` / `point_start` / `session_end` |
| `point_index` | Index of the calibration/validation point; `-1` for session-level events |
| `target_x_norm`, `target_y_norm` | Target dot position in **normalized coordinates** (0–1). Primary training target. |
| `target_x_px`, `target_y_px` | Target dot position in screen pixels (within the drawable area) |
| `screen_w_px`, `screen_h_px` | **Drawable area dimensions** used when rendering dots. This is the canonical screen size for gaze mapping. |
| `fixation_letter` | Random letter shown on the dot to prevent mind-wandering. Ignore in offline processing. |

- **Calibration:** 16 points in a fixed spatial layout (corners, edges, center cluster).
- **Validation:** 9 points in a 3×3 uniform grid.
- Each point occupies approximately **10.8 seconds** → ~324 raw frames at 30 fps.

Example rows from `calibration_events.csv`:

```
elapsed_ms, event_type,  point_index, target_x_norm, target_y_norm, target_x_px, target_y_px, screen_w_px, screen_h_px, ...
85,          point_start, 0,           0.0500,        0.0500,        36,          69,          720,         1382,        ...
10877,       point_start, 1,           0.9500,        0.0500,        684,         69,          720,         1382,        ...
```

### 2.4 experiment_events.csv

Different format; only `stimulus_start` and `stimulus_end` events are relevant. Only video frames between these two timestamps should be used for gaze inference. **This phase is deferred and not covered in the current pipeline.**

### 2.5 Important Data Quirks

#### Screen height discrepancy

`metadata.json` reports `screen_height_px: 1520` (total physical screen, including Android soft navigation bar ≈ 138 px).

The event CSV reports `screen_h_px: 1382` (the actual drawable area the app used when rendering calibration dots).

Verification: Point 0 has `target_y_norm = 0.05` and `target_y_px = 69`.
- Using CSV value: `1382 × 0.05 = 69.1 px` ✓ matches
- Using metadata value: `1520 × 0.05 = 76 px` ✗ does not match

**Rule: always read `screen_w_px` and `screen_h_px` from the event CSV, not from `metadata.json`.**

#### Video orientation

The phone is held portrait during recording. The face video is stored upright — faces appear vertically normal when opened. No rotation pre-processing is required before running face detection.

The hardware camera reports its sensor resolution as `1920×1080` (landscape notation), but the stored video dimensions are portrait: 1080 px wide × 1920 px tall.

#### Timestamp alignment

`elapsed_ms = 0` corresponds exactly to video frame 0. To convert a timestamp to a frame index:

```python
frame_index = int(elapsed_ms / 1000.0 * camera_fps)
```

---

## 3. Pipeline Architecture

```
[Session Folder]
       │
       ▼
[Step 1]  Load metadata.json + parse event CSV
           → per-point time windows (ms range per point)
       │
       ▼
[Step 2]  Extract face frames from MP4 per time window
           → list of BGR frames per calibration point
       │
       ▼
[Step 3]  Face detection (RetinaFace) + Gaze inference (ResNet34)
           → (pitch_deg, yaw_deg) per valid frame
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[Step 4 — Calibration]                   [Debug Visualizer]
  Average (pitch, yaw) per point          Interactive viewer to verify
  → polynomial regression fit             timestamp alignment between
  → calibration quality report            video frames and CSV events
       │
       ▼
[Step 5 — Validation]
  Predict gaze for each validation point
  → compare to ground truth
  → compute error metrics + scatter plot
```

---

## 4. Phase 1 — Calibration

### 4.1 Event Parsing & Frame Window Extraction

1. Load `calibration_events.csv`.
2. Filter rows where `event_type == "point_start"`, sort ascending by `elapsed_ms`. This gives 16 rows.
3. Read canonical screen dimensions from the first `point_start` row: `screen_w_px`, `screen_h_px`.
4. For each calibration point `i` (0–15):

```
t_window_start = elapsed_ms[i] + 1000   # skip first 1 second (saccade + settling)
t_window_end   = elapsed_ms[i+1]        # next point's start timestamp
                                         # (for the last point: use session_end timestamp)

frame_start = int(t_window_start / 1000.0 * fps)
frame_end   = int(t_window_end   / 1000.0 * fps)
```

5. Open the video, seek to `frame_start`, read frames until `frame_end`.

**Expected yield per point:** ~9.8 s × 30 fps ≈ **294 frames** after the 1-second skip.

> Why skip the first second? After a new dot appears, the eye must saccade from the previous dot's position. Saccades take 50–200 ms, followed by a settling period. Frames during this transition are labeled "looking at the new dot" but the eye is physically in motion — these are wrong-label frames. Skipping 1 second removes ~30 frames (9% of the window) while eliminating the noisiest portion. The remaining ~294 frames are clean fixation data suitable for regression.

### 4.2 Face Detection & Gaze Inference

For each extracted frame:

1. Run **RetinaFace** face detector on the frame.
2. If no face is detected → skip this frame (do not include in aggregation).
3. If one or more faces detected → select the face with the largest bounding box area.
4. Clamp bounding box to frame boundaries. Crop the face region.
5. Run **ResNet34 gaze model** on the 448×448 normalized face crop (same transform pipeline as `calibration_demo.py`):
   - Resize to 448, normalize with ImageNet mean/std.
   - Decode soft-argmax output → `pitch_deg`, `yaw_deg`.

Reference implementation: `calibration_demo.py` functions `get_face_crop()` and `predict_gaze()`.

### 4.3 Per-Point Feature Aggregation

For each calibration point:

1. Collect all valid `(pitch_deg, yaw_deg)` predictions (frames where face was detected).
2. Compute the mean: `pitch_mean = mean(pitches)`, `yaw_mean = mean(yaws)`.
3. Pair with the ground truth: `(target_x_norm, target_y_norm)` from the CSV.

This produces 16 training samples:

```
(pitch_mean_i, yaw_mean_i) → (target_x_norm_i, target_y_norm_i)   for i = 0..15
```

### 4.4 Polynomial Regression Fitting

**Input:** 16 samples of `(pitch_mean, yaw_mean)` → `(target_x_norm, target_y_norm)`.

**Procedure:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

feats   = np.column_stack([pitch_means, yaw_means])          # shape (16, 2)
targets = np.array([[x_norm, y_norm], ...])                  # shape (16, 2)

poly  = PolynomialFeatures(degree=2, include_bias=True)
X     = poly.fit_transform(feats)                            # shape (16, 6)
reg_x = Ridge(alpha=1.0).fit(X, targets[:, 0])
reg_y = Ridge(alpha=1.0).fit(X, targets[:, 1])
```

**Calibration model:** the tuple `(poly, reg_x, reg_y)`.

**Prediction function:**

```python
def predict_norm(pitch, yaw, poly, reg_x, reg_y):
    feat = poly.transform([[pitch, yaw]])
    return reg_x.predict(feat)[0], reg_y.predict(feat)[0]

# Convert to screen pixels:
x_px = x_norm * screen_w_px
y_px = y_norm * screen_h_px
```

**Model persistence:** Re-fit each run. Polynomial regression on 16 points completes in under 1 ms; no serialization needed.

### 4.5 Calibration Quality Assessment (In-Sample Fit)

After fitting, evaluate on the 16 training points:

1. Predict `(x_norm_pred, y_norm_pred)` for each of the 16 calibration inputs.
2. Compute per-point Euclidean error in pixels:
   ```
   dx = (x_norm_pred - x_norm_true) × screen_w_px
   dy = (y_norm_pred - y_norm_true) × screen_h_px
   error_px = sqrt(dx² + dy²)
   ```
3. Report: `RMSE_x`, `RMSE_y`, `mean_euclidean_error_px`.
4. Generate a **residual scatter plot** on a schematic of the phone screen:
   - Blue dots: ground truth positions of all 16 calibration points.
   - Red dots: predicted positions.
   - Arrows from true → predicted, showing direction and magnitude of residual error.

> Note: This is in-sample fit quality. A low training RMSE confirms the polynomial has enough capacity to represent the mapping, but does not measure generalization. Validation MAE (Phase 2) is the true accuracy measure.

---

## 5. Phase 2 — Validation

Same frame extraction and gaze inference pipeline as Phase 1. Apply the calibration model fitted in Phase 1 to the validation video.

**Input:** `validation_events.csv` — 9 points in a 3×3 uniform grid.

### 5.1 Procedure

1. Fit the calibration model from Phase 1 (16 calibration points).
2. Parse `validation_events.csv`, extract 9 `point_start` windows.
3. For each validation point `j` (0–8):
   - Extract frames with the same 1-second skip rule.
   - For each valid frame: detect face → predict `(pitch, yaw)` → apply calibration model → get `(x_norm_pred, y_norm_pred)`.
   - Average predictions across all valid frames: `(x_norm_mean, y_norm_mean)`.
4. Compare to ground truth `(target_x_norm_j, target_y_norm_j)`.

### 5.2 Error Metrics

| Metric | Description |
|---|---|
| Per-point error (px) | `sqrt(dx² + dy²)` using screen dimensions from CSV |
| Mean error (px) | Average Euclidean error across all 9 validation points |
| Median error (px) | Median Euclidean error (robust to outliers) |
| Max error (px) | Worst-case point |
| Mean error (norm) | Device-independent; useful for cross-session comparison |

### 5.3 Validation Scatter Plot

On a schematic of the phone screen:
- Blue dots: 9 ground truth validation point positions.
- Red dots: 9 predicted gaze positions.
- Arrows and per-point error labels.

This plot is the primary deliverable for evaluating calibration quality.

---

## 6. Debug Visualization Tool

### 6.1 Purpose

Before trusting any pipeline output, verify that the time alignment between the event CSV and the face video is correct. A misalignment of even a few hundred milliseconds would pair gaze predictions with the wrong calibration targets, silently corrupting the model.

This tool is a **diagnostic/debugging aid only** — no gaze inference runs inside it.

### 6.2 UI Description

An OpenCV-based interactive viewer:

```
┌──────────────────────────────────────────────────────┐
│  Point 3 | target: (0.95, 0.95)        [Q: quit]    │
│                                                      │
│                                                      │
│              [ face video frame ]                    │
│                                                      │
│                                                      │
│  ┌─────────────┐                    107802 ms        │
│  │  ·          │                                     │
│  │             │   ← mini screen diagram             │
│  │             │     dot shows target position       │
│  │          •  │                                     │
│  └─────────────┘                                     │
└──────────────────────────────────────────────────────┘
```

**Controls:**

| Key | Action |
|---|---|
| `0`–`9`, `a`–`f` or arrow keys | Select calibration point (0–15) |
| `SPACE` | Play / pause |
| `Q` | Quit |

**Overlays:**

- **Top-left:** `Point {index} | target: ({x_norm:.2f}, {y_norm:.2f})` — identifies which calibration point is being reviewed.
- **Bottom-right:** `{elapsed_ms} ms` — timestamp relative to session start, same reference frame as the CSV. Updated every ~500 ms of video time to avoid visual clutter.
- **Bottom-left inset:** A small rectangle (~100×180 px) representing the phone screen schematically, with a filled dot at `(x_norm, y_norm)`. Provides immediate spatial context for which screen region the subject should be looking at.

### 6.3 Implementation Notes

- Seek to `frame_index = int(t_window_start / 1000.0 * fps)` when a new point is selected.
- Compute and display `elapsed_ms = int(current_frame_index / fps * 1000)` for the current frame.
- The mini screen inset: draw a filled rectangle, then a circle at `(x_norm * inset_w, y_norm * inset_h)`.
- Video plays at the original frame rate (30 fps target using `cv2.waitKey(33)`).
- If `frame_end` is reached, stop and wait for user input (do not loop).

---

## 7. Design Decisions & Rationale

### 7.1 Frame Selection: Skip First 1 Second Per Point

**Decision:** Skip the first 1000 ms after each `point_start` event; average all remaining frames in the window.

**Rationale:**
- After a new calibration dot appears, the eye saccades from the previous position to the new target. This movement takes 50–200 ms, and the eye requires additional time to stabilize (total ≈ 300–700 ms). The 1-second cutoff is a conservative buffer covering even slow reactions.
- Frames during the saccade carry the label "looking at target X" but the gaze is physically aimed elsewhere. These are mislabeled samples that introduce systematic bias into the regression.
- Skipping 1 s removes only 30 frames (9% of the ~324-frame window), leaving ~294 clean fixation frames — more than sufficient for a robust mean estimate.
- Averaging all remaining frames provides a stable, noise-reduced feature vector per calibration point.

### 7.2 Coordinate System: Normalized (0–1)

**Decision:** Train the polynomial regression using `(target_x_norm, target_y_norm)` as targets; convert to pixels only for display.

**What normalized means:** A point at screen coordinates `(x_px, y_px)` on a `(W, H)` drawable area is represented as `(x_px / W, y_px / H)`. Example: `(360px, 691px)` on a `720×1382` screen → `(0.50, 0.50)`.

**Rationale:**
- The event CSV already provides `target_x_norm` / `target_y_norm` as first-class fields.
- A model trained in normalized space generalizes across devices and screen resolutions without re-fitting.
- Error metrics can be reported in both normalized units (device-independent) and pixels (intuitive) by simple multiplication.

### 7.3 Canonical Screen Dimensions

**Decision:** Use `screen_w_px` / `screen_h_px` from the event CSV, not `metadata.json`.

**Rationale:** `metadata.json` records the total physical screen height (1520 px on the test device), which includes the Android soft navigation bar (≈ 138 px). The app rendered all calibration and validation dots within the drawable area (1382 px), and computed normalized coordinates using that value. Using the metadata height would introduce a systematic vertical error in all gaze predictions.

Verification: `target_y_norm = 0.05` → `1382 × 0.05 = 69.1 px` ✓ (matches `target_y_px = 69` in CSV).

---

## 8. Deliverables

| File | Description |
|---|---|
| `pipeline/gaze_utils.py` | Shared utilities: face detection, gaze inference, frame extraction, coordinate conversion |
| `pipeline/run_calibration.py` | Full calibration pipeline: parse events → inference → polynomial fit → quality report + residual plot |
| `pipeline/run_validation.py` | Validation pipeline: load calibration model → predict → compute error metrics + scatter plot |
| `pipeline/visualizer.py` | Debug visualization tool: interactive point selector + face video playback with overlays |

All scripts accept a `--session` argument pointing to the session folder path.

---

## 9. Out of Scope (Current Phase)

| Topic | Notes |
|---|---|
| **Experiment phase** | Running gaze inference on `experiment.mp4` during `stimulus_start` → `stimulus_end`. Deferred until calibration + validation are validated end-to-end. |
| **AOI analysis** | Mapping gaze predictions to Areas of Interest in the stimulus video. Part of the experiment phase. |
| **Temporal smoothing** | Moving average or Kalman filter on frame-level predictions. May be added in the experiment phase. |
| **On-device inference** | Out of scope by design. The mobile app is a data collection tool only. |
| **Multi-session aggregation** | Each session is processed independently for now. |
