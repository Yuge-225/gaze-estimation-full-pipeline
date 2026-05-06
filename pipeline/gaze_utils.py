"""
Shared utilities for the mobile gaze estimation offline pipeline.

All other pipeline scripts import from here. Nothing in this file
is runnable as a standalone script.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ---------------------------------------------------------------------------
# Project root on sys.path so config / utils / models are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import data_config
from uniface import RetinaFace
from utils.helpers import get_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PointRecord:
    """One calibration or validation point parsed from the event CSV."""

    index: int
    t_start_ms: float  # elapsed_ms of point_start event
    t_end_ms: float    # elapsed_ms of next point_start (or session_end)
    x_norm: float      # normalised target x in [0, 1]
    y_norm: float      # normalised target y in [0, 1]
    x_px: int          # target x in drawable-area pixels
    y_px: int          # target y in drawable-area pixels
    screen_w: int      # drawable-area width  (from CSV — NOT metadata.json)
    screen_h: int      # drawable-area height (from CSV — NOT metadata.json)


@dataclass
class PointInference:
    """Gaze inference result aggregated over one calibration/validation point."""

    record: PointRecord
    pitch_mean: float  # mean pitch across all valid frames (degrees)
    yaw_mean: float    # mean yaw   across all valid frames (degrees)
    n_valid: int       # frames where a face was successfully detected
    n_total: int       # total frames in the inference window


@dataclass
class CalibrationModel:
    """Fitted polynomial gaze calibration model."""

    poly: object   # sklearn PolynomialFeatures transformer
    reg_x: object  # Ridge regressor for x_norm
    reg_y: object  # Ridge regressor for y_norm
    degree: int
    screen_w: int  # drawable-area width used during calibration
    screen_h: int  # drawable-area height used during calibration


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


def load_session_meta(session_dir: Path) -> dict:
    """
    Load and basic-validate metadata.json from a session folder.

    Raises FileNotFoundError / ValueError on missing file or required fields.
    """
    import json

    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {session_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    for required_key in ("session_id", "camera_fps"):
        if required_key not in meta:
            raise ValueError(
                f"metadata.json is missing required field '{required_key}' "
                f"in {meta_path}"
            )

    logger.info(
        "Session %s | platform=%s | fps=%s | camera_res=%s",
        meta.get("session_id"),
        meta.get("platform"),
        meta.get("camera_fps"),
        meta.get("camera_resolution"),
    )
    return meta


# ---------------------------------------------------------------------------
# Event CSV parsing
# ---------------------------------------------------------------------------


def parse_event_csv(csv_path: Path) -> list[PointRecord]:
    """
    Parse calibration_events.csv or validation_events.csv.

    Returns PointRecord list sorted by point_index.
    Screen dimensions come from the CSV (drawable area), not metadata.json.
    The last point's t_end_ms is the session_end timestamp; if that row is
    absent, we estimate it from the median inter-point gap.
    """
    import csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Event CSV not found: {csv_path}")

    point_rows: list[dict] = []
    session_end_ms: Optional[float] = None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            etype = row.get("event_type", "").strip()

            if etype == "point_start":
                try:
                    point_rows.append(
                        {
                            "elapsed_ms": float(row["elapsed_ms"]),
                            "index": int(row["point_index"]),
                            "x_norm": float(row["target_x_norm"]),
                            "y_norm": float(row["target_y_norm"]),
                            "x_px": int(row["target_x_px"]),
                            "y_px": int(row["target_y_px"]),
                            "screen_w": int(row["screen_w_px"]),
                            "screen_h": int(row["screen_h_px"]),
                        }
                    )
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "Skipping malformed point_start row in %s: %s — %s",
                        csv_path.name,
                        dict(row),
                        exc,
                    )

            elif etype == "session_end":
                try:
                    session_end_ms = float(row["elapsed_ms"])
                except ValueError:
                    logger.warning(
                        "Cannot parse session_end elapsed_ms in %s", csv_path.name
                    )

    if not point_rows:
        raise ValueError(
            f"No valid point_start events found in {csv_path}. "
            "Check that the file is not empty and the CSV header matches the expected format."
        )

    point_rows.sort(key=lambda r: r["elapsed_ms"])

    # Compute t_end for each point
    records: list[PointRecord] = []
    for i, ps in enumerate(point_rows):
        if i + 1 < len(point_rows):
            t_end = point_rows[i + 1]["elapsed_ms"]
        elif session_end_ms is not None:
            t_end = session_end_ms
        else:
            # Estimate from median gap between consecutive points
            gaps = [
                point_rows[j + 1]["elapsed_ms"] - point_rows[j]["elapsed_ms"]
                for j in range(len(point_rows) - 1)
            ]
            median_gap = float(np.median(gaps)) if gaps else 10_000.0
            t_end = ps["elapsed_ms"] + median_gap
            logger.warning(
                "session_end event missing in %s — estimating last point t_end "
                "as %.0f ms (median gap = %.0f ms)",
                csv_path.name,
                t_end,
                median_gap,
            )

        records.append(
            PointRecord(
                index=ps["index"],
                t_start_ms=ps["elapsed_ms"],
                t_end_ms=t_end,
                x_norm=ps["x_norm"],
                y_norm=ps["y_norm"],
                x_px=ps["x_px"],
                y_px=ps["y_px"],
                screen_w=ps["screen_w"],
                screen_h=ps["screen_h"],
            )
        )

    logger.info(
        "Parsed %d points from %s | screen=%dx%d (drawable area)",
        len(records),
        csv_path.name,
        records[0].screen_w,
        records[0].screen_h,
    )
    return records


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def ms_to_frame(elapsed_ms: float, fps: float) -> int:
    """Convert a session-relative timestamp (ms) to a video frame index."""
    return int(elapsed_ms / 1000.0 * fps)


def get_video_fps(video_path: Path, fallback_fps: float = 30.0) -> float:
    """
    Read the actual frame rate from the video file via OpenCV.

    Mobile recordings often report a target fps (e.g. 30) in app metadata
    while the actual encoded fps is much lower. This function reads from
    the video container, which is what OpenCV uses when seeking by frame index.

    Falls back to `fallback_fps` if the video cannot be opened or reports
    an invalid fps.
    """
    if not video_path.exists():
        logger.warning(
            "Cannot read fps from missing file %s — using fallback %.1f fps",
            video_path.name,
            fallback_fps,
        )
        return fallback_fps

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(
            "Cannot open %s to read fps — using fallback %.1f fps",
            video_path.name,
            fallback_fps,
        )
        return fallback_fps

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if video_fps <= 0:
        logger.warning(
            "%s reported invalid fps=%.2f — using fallback %.1f fps",
            video_path.name,
            video_fps,
            fallback_fps,
        )
        return fallback_fps

    if abs(video_fps - fallback_fps) > 1.0:
        logger.warning(
            "%s: actual fps=%.2f differs significantly from metadata fps=%.1f "
            "(total_frames=%d). Using actual video fps for frame-index math. "
            "This is common on mobile — metadata reports target fps, not achieved fps.",
            video_path.name,
            video_fps,
            fallback_fps,
            total_frames,
        )

    return video_fps


def iter_frames(
    video_path: Path,
    frame_start: int,
    frame_end: int,
    stride: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_index, BGR frame) for frame indices in [frame_start, frame_end)
    at the given stride.

    Uses direct seeking to frame_start — efficient for windowed access.
    Releases the VideoCapture even if the caller abandons the generator early.

    Raises:
        FileNotFoundError: if video_path does not exist.
        IOError:           if OpenCV cannot open the file.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(
            f"OpenCV cannot open video: {video_path}. "
            "Make sure the codec is supported and the file is not corrupted."
        )

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_start >= total:
            logger.warning(
                "frame_start=%d >= total_frames=%d in %s — nothing to yield",
                frame_start,
                total,
                video_path.name,
            )
            return

        clipped_end = min(frame_end, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_start))

        for idx in range(frame_start, clipped_end, stride):
            # Seek is only needed when stride > 1
            if stride > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))

            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    "Unexpected read failure at frame %d (expected up to %d) in %s",
                    idx,
                    clipped_end,
                    video_path.name,
                )
                break
            yield idx, frame

    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_transform() -> transforms.Compose:
    """Standard preprocessing transform matching the training pipeline."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_gaze_model(
    weight_path: Path,
    model_arch: str = "resnet34",
    dataset: str = "gaze360",
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load the gaze model weights and return
    (model, idx_tensor, binwidth, angle, device).

    Raises:
        FileNotFoundError: if weight_path does not exist.
        ValueError:        if dataset is not in data_config.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Gaze model weights not found: {weight_path}. "
            "Run download.sh or check the --weight argument."
        )

    if dataset not in data_config:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available options: {list(data_config.keys())}"
        )

    cfg = data_config[dataset]
    bins, binwidth, angle = cfg["bins"], cfg["binwidth"], cfg["angle"]

    logger.info(
        "Loading model '%s' from %s on %s ...", model_arch, weight_path, device
    )
    model = get_model(model_arch, bins, inference_mode=True)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.info("Model ready.")

    idx_tensor = torch.arange(bins, dtype=torch.float32, device=device)
    return model, idx_tensor, binwidth, angle, device


def load_face_detector() -> RetinaFace:
    """Instantiate the RetinaFace detector. Logs timing."""
    logger.info("Initialising RetinaFace detector ...")
    detector = RetinaFace()
    logger.info("RetinaFace ready.")
    return detector


# ---------------------------------------------------------------------------
# Single-frame inference
# ---------------------------------------------------------------------------


def get_face_crop(
    frame: np.ndarray,
    detector: RetinaFace,
) -> Optional[np.ndarray]:
    """
    Detect faces in frame and return the largest face crop (BGR), or None.

    The largest face by bounding-box area is selected to handle cases where
    a second person (or reflection) is partially visible.
    Bounding-box coordinates are clamped to frame boundaries before cropping.
    """
    faces = detector.detect(frame)
    if not faces:
        return None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    x1, y1, x2, y2 = map(int, face.bbox[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None  # degenerate box after clamping

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def predict_gaze(
    face_crop: np.ndarray,
    model,
    idx_tensor: torch.Tensor,
    binwidth: float,
    angle: float,
    device: torch.device,
    transform: transforms.Compose,
) -> tuple[float, float]:
    """
    Run the gaze model on a BGR face crop.

    Returns (pitch_deg, yaw_deg).
    The model outputs (yaw_logits, pitch_logits); decoded via soft-argmax.
    """
    img = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        yaw_raw, pitch_raw = model(img)

    yaw_prob = F.softmax(yaw_raw, dim=1)
    pitch_prob = F.softmax(pitch_raw, dim=1)

    yaw_deg = (torch.sum(yaw_prob * idx_tensor, dim=1) * binwidth - angle).item()
    pitch_deg = (torch.sum(pitch_prob * idx_tensor, dim=1) * binwidth - angle).item()

    return pitch_deg, yaw_deg


# ---------------------------------------------------------------------------
# Per-point pipeline
# ---------------------------------------------------------------------------


def infer_point(
    video_path: Path,
    record: PointRecord,
    fps: float,
    face_detector: RetinaFace,
    gaze_model,
    idx_tensor: torch.Tensor,
    binwidth: float,
    angle: float,
    device: torch.device,
    transform: transforms.Compose,
    skip_ms: float = 1000.0,
    end_trim_ms: float = 500.0,
    stride: int = 1,
) -> Optional[PointInference]:
    """
    Extract and process all frames for one calibration/validation point.

    Skips the first `skip_ms` milliseconds (saccade + settling) and the last
    `end_trim_ms` milliseconds (anticipatory saccade to next target).
    Returns None if no face is detected in any frame of the window.

    Args:
        video_path:    path to the face-recording video.
        record:        PointRecord describing the time window and ground truth.
        fps:           video frame rate.
        skip_ms:       milliseconds to skip after point_start (default 1000 ms).
        end_trim_ms:   milliseconds to trim before point_end (default 500 ms).
        stride:        process every Nth frame (default 1 = every frame).
    """
    t_infer_start = record.t_start_ms + skip_ms
    t_infer_end = record.t_end_ms - end_trim_ms

    if t_infer_start >= t_infer_end:
        logger.warning(
            "Point %d: usable window [%.0f, %.0f ms] is empty after "
            "skip_ms=%.0f + end_trim_ms=%.0f — skipping",
            record.index,
            t_infer_start,
            t_infer_end,
            skip_ms,
            end_trim_ms,
        )
        return None

    frame_start = ms_to_frame(t_infer_start, fps)
    frame_end = ms_to_frame(t_infer_end, fps)

    if frame_start >= frame_end:
        logger.warning(
            "Point %d: empty frame range [%d, %d) after skip — skipping",
            record.index,
            frame_start,
            frame_end,
        )
        return None

    pitches: list[float] = []
    yaws: list[float] = []
    n_total = 0
    log_every = max(1, (frame_end - frame_start) // stride // 5)  # log ~5 times per point

    for i, (frame_idx, frame) in enumerate(
        iter_frames(video_path, frame_start, frame_end, stride=stride)
    ):
        n_total += 1

        crop = get_face_crop(frame, face_detector)
        if crop is None:
            continue

        try:
            pitch, yaw = predict_gaze(
                crop, gaze_model, idx_tensor, binwidth, angle, device, transform
            )
        except Exception as exc:
            logger.debug(
                "Point %d frame %d: gaze inference error — %s", record.index, frame_idx, exc
            )
            continue

        pitches.append(pitch)
        yaws.append(yaw)

        if i > 0 and i % log_every == 0:
            logger.debug(
                "  Point %d: %d/%d frames processed, %d valid so far",
                record.index,
                n_total,
                (frame_end - frame_start) // stride,
                len(pitches),
            )

    n_valid = len(pitches)
    detection_rate = n_valid / n_total if n_total > 0 else 0.0

    if n_valid == 0:
        logger.warning(
            "Point %02d | target=(%.2f, %.2f) | NO valid face detections "
            "in %d frames (window %.0f–%.0f ms) — point excluded",
            record.index,
            record.x_norm,
            record.y_norm,
            n_total,
            t_infer_start,
            t_infer_end,
        )
        return None

    logger.info(
        "Point %02d | target=(%.3f, %.3f) | frames=%d/%d (%.0f%%) "
        "| pitch=%+.2f°  yaw=%+.2f°",
        record.index,
        record.x_norm,
        record.y_norm,
        n_valid,
        n_total,
        detection_rate * 100,
        np.mean(pitches),
        np.mean(yaws),
    )

    return PointInference(
        record=record,
        pitch_mean=float(np.mean(pitches)),
        yaw_mean=float(np.mean(yaws)),
        n_valid=n_valid,
        n_total=n_total,
    )


# ---------------------------------------------------------------------------
# Calibration model fitting
# ---------------------------------------------------------------------------


def fit_calibration_model(
    inferences: list[PointInference],
    degree: int = 2,
) -> CalibrationModel:
    """
    Fit a polynomial regression: (pitch_deg, yaw_deg) → (x_norm, y_norm).

    Uses Ridge regression to handle near-collinear features that arise from
    limited spatial coverage of calibration points.

    Raises ValueError if fewer valid points are provided than the number of
    polynomial features (model would be under-determined).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures

    # Minimum points needed: number of terms in a degree-d 2D polynomial
    min_points = (degree + 1) * (degree + 2) // 2
    if len(inferences) < min_points:
        raise ValueError(
            f"Need at least {min_points} valid calibration points for "
            f"a degree-{degree} polynomial, but only {len(inferences)} are available. "
            "Check the calibration video for face-detection failures."
        )

    feats = np.array([[p.pitch_mean, p.yaw_mean] for p in inferences])
    x_norms = np.array([p.record.x_norm for p in inferences])
    y_norms = np.array([p.record.y_norm for p in inferences])
    screen_w = inferences[0].record.screen_w
    screen_h = inferences[0].record.screen_h

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X = poly.fit_transform(feats)
    reg_x = Ridge(alpha=1.0).fit(X, x_norms)
    reg_y = Ridge(alpha=1.0).fit(X, y_norms)

    # In-sample RMSE (reported in pixels for intuitiveness)
    x_pred = reg_x.predict(X)
    y_pred = reg_y.predict(X)
    rmse_x_px = float(np.sqrt(np.mean((x_pred - x_norms) ** 2)) * screen_w)
    rmse_y_px = float(np.sqrt(np.mean((y_pred - y_norms) ** 2)) * screen_h)
    mean_euc_px = float(
        np.mean(
            np.sqrt(
                ((x_pred - x_norms) * screen_w) ** 2
                + ((y_pred - y_norms) * screen_h) ** 2
            )
        )
    )

    logger.info(
        "Polynomial fit (degree=%d, %d points) | "
        "RMSE_x=%.1f px  RMSE_y=%.1f px  mean_euclidean=%.1f px  [in-sample]",
        degree,
        len(inferences),
        rmse_x_px,
        rmse_y_px,
        mean_euc_px,
    )

    return CalibrationModel(
        poly=poly,
        reg_x=reg_x,
        reg_y=reg_y,
        degree=degree,
        screen_w=screen_w,
        screen_h=screen_h,
    )


def apply_calibration(
    pitch: float,
    yaw: float,
    model: CalibrationModel,
) -> tuple[float, float]:
    """
    Map (pitch_deg, yaw_deg) → (x_norm, y_norm) using the fitted model.

    Predictions are NOT clamped to [0, 1] — callers decide whether to clip
    to screen bounds, since out-of-range values are useful diagnostics.
    """
    feat = model.poly.transform([[pitch, yaw]])
    x_norm = float(model.reg_x.predict(feat)[0])
    y_norm = float(model.reg_y.predict(feat)[0])
    return x_norm, y_norm
