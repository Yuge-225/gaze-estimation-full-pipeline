"""
visualizer.py — Debug / alignment-verification tool.

Plays the face-recording video for each calibration or validation point.
Overlays the session-relative timestamp (ms) and a mini screen diagram
showing where the target dot was, so you can verify that time alignment
between the event CSV and the video is correct before trusting inference.

Controls:
    SPACE       Play / pause
    [ or LEFT   Previous point
    ] or RIGHT  Next point
    0–9         Jump to point 0–9 directly
    a–f         Jump to point 10–15 directly
    Q           Quit

The "skip zone" (first skip_ms of each window) is highlighted with a blue
tint — this is the saccade / settling period excluded from gaze inference.

Usage:
    python pipeline/visualizer.py \\
        --session GazeData/session_20260504_210116_THH0FFPA

    python pipeline/visualizer.py \\
        --session GazeData/session_20260504_210116_THH0FFPA \\
        --phase validation
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaze_utils import PointRecord, get_video_fps, load_session_meta, ms_to_frame, parse_event_csv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Mini screen inset dimensions (px in the display window)
INSET_W = 110
INSET_H = 190
INSET_PAD = 12
INSET_DOT_R = 6


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _text_bg(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    pad: int = 4,
) -> None:
    """Draw text with a dark backing rectangle for readability."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_top_info(
    frame: np.ndarray,
    record: PointRecord,
    pt_idx: int,
    n_total: int,
    paused: bool,
    phase: str,
) -> None:
    """Top-left overlay: point metadata and playback state."""
    lines = [
        (
            f"{phase.upper()}  Point {record.index}  ({pt_idx + 1}/{n_total})",
            (80, 200, 255),
            0.58,
        ),
        (
            f"target (norm): ({record.x_norm:.3f}, {record.y_norm:.3f})",
            (180, 180, 180),
            0.50,
        ),
        (
            f"target  (px) : ({record.x_px}, {record.y_px})",
            (180, 180, 180),
            0.50,
        ),
        (
            f"window : {record.t_start_ms:.0f} - {record.t_end_ms:.0f} ms",
            (180, 180, 180),
            0.50,
        ),
        (
            "[ PAUSED ]  SPACE=play  [ / ]  ←/→  0-9/a-f  Q=quit"
            if paused
            else "SPACE=pause  [ / ]  ←/→  0-9/a-f  Q=quit",
            (120, 120, 120),
            0.44,
        ),
    ]
    y = 28
    for text, color, scale in lines:
        _text_bg(frame, text, (10, y), scale, color)
        y += int(scale * 45 + 6)


def draw_timestamp(frame: np.ndarray, elapsed_ms: int) -> None:
    """Bottom-right: session-relative timestamp of the current video frame."""
    h, w = frame.shape[:2]
    text = f"{elapsed_ms:,} ms"
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.60, 1)
    x = w - tw - 12
    y = h - baseline - 10
    _text_bg(frame, text, (x, y), 0.60, (80, 210, 255), pad=5)


def draw_excluded_overlay(
    frame: np.ndarray,
    is_excluded: bool,
    label: str,
) -> None:
    """Blue tint + label while inside an excluded zone (skip or end-trim)."""
    if not is_excluded:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (120, 60, 0), -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    _text_bg(frame, label, (10, h // 2), 0.52, (100, 130, 255))


def draw_screen_inset(
    frame: np.ndarray,
    x_norm: float,
    y_norm: float,
) -> None:
    """
    Bottom-left: small rectangle schematically representing the phone screen,
    with a dot at the normalised target position.
    """
    h = frame.shape[0]
    x0 = INSET_PAD
    y0 = h - INSET_H - INSET_PAD

    # Background + border
    cv2.rectangle(frame, (x0, y0), (x0 + INSET_W, y0 + INSET_H), (35, 35, 55), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + INSET_W, y0 + INSET_H), (90, 90, 130), 1)

    # "screen" label
    cv2.putText(
        frame, "screen",
        (x0 + 3, y0 - 5),
        FONT, 0.36, (110, 110, 160), 1, cv2.LINE_AA,
    )

    # Target dot — clamped so it never overflows the inset border
    dot_x = int(np.clip(x0 + x_norm * INSET_W, x0 + INSET_DOT_R, x0 + INSET_W - INSET_DOT_R))
    dot_y = int(np.clip(y0 + y_norm * INSET_H, y0 + INSET_DOT_R, y0 + INSET_H - INSET_DOT_R))
    cv2.circle(frame, (dot_x, dot_y), INSET_DOT_R, (80, 210, 255), -1)
    cv2.circle(frame, (dot_x, dot_y), INSET_DOT_R + 2, (80, 210, 255), 1)


# ---------------------------------------------------------------------------
# Video state helpers
# ---------------------------------------------------------------------------


def seek_and_read(
    cap: cv2.VideoCapture,
    frame_idx: int,
    total_frames: int,
) -> tuple[np.ndarray | None, int]:
    """
    Seek cap to frame_idx and read one frame.
    Returns (frame_or_None, actual_frame_idx_read).
    actual_frame_idx_read may differ from frame_idx due to keyframe rounding
    in compressed video — we log the discrepancy if large.
    """
    clamped = min(frame_idx, max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(clamped))
    ret, frame = cap.read()
    if not ret:
        logger.warning("cap.read() failed after seek to frame %d", clamped)
        return None, clamped

    actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if abs(actual - clamped) > 5:
        logger.debug(
            "Seek rounding: requested frame %d, got frame %d "
            "(compressed-video keyframe snap)",
            clamped,
            actual,
        )
    return frame, actual


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------


def run_viewer(
    records: list[PointRecord],
    video_path: Path,
    fps: float,
    skip_ms: float,
    end_trim_ms: float,
    phase: str,
) -> None:
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Use the actual video fps for all frame-index ↔ timestamp math.
    # The metadata fps is the camera's target rate; mobile encoders often
    # achieve a lower rate. Frame indices are wrong if we use the wrong fps.
    fps = video_fps if video_fps > 0 else fps
    logger.info(
        "Video: %s | actual_fps=%.2f | %d frames",
        video_path.name,
        fps,
        total_frames,
    )

    window_title = (
        "Gaze Debug Visualizer  |  SPACE=play/pause  [ ]=prev/next  0-9/a-f=jump  Q=quit"
    )
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    # Start with a portrait aspect hint; user can resize freely
    sample_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sample_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_h = min(900, sample_h)
    display_w = int(sample_w * display_h / sample_h)
    cv2.resizeWindow(window_title, display_w, display_h)

    # ── State ────────────────────────────────────────────────────────────
    current_pt = 0
    paused = True

    # Load initial frame
    start_frame = ms_to_frame(records[current_pt].t_start_ms, fps)
    cached_frame, current_frame_idx = seek_and_read(cap, start_frame, total_frames)
    if cached_frame is None:
        cached_frame = np.zeros((sample_h, sample_w, 3), dtype=np.uint8)

    logger.info(
        "Viewer ready. %d points loaded. Starting at point %d.",
        len(records),
        records[current_pt].index,
    )

    def jump_to_point(pt_idx: int) -> tuple[np.ndarray, int]:
        rec = records[pt_idx]
        frame_idx = ms_to_frame(rec.t_start_ms, fps)
        frame, actual_idx = seek_and_read(cap, frame_idx, total_frames)
        if frame is None:
            frame = np.zeros((sample_h, sample_w, 3), dtype=np.uint8)
        logger.info(
            "→ Point %d | frames [%d, %d) | target=(%.3f, %.3f)",
            rec.index,
            frame_idx,
            ms_to_frame(rec.t_end_ms, fps),
            rec.x_norm,
            rec.y_norm,
        )
        return frame, actual_idx

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        rec = records[current_pt]
        frame_end_idx = ms_to_frame(rec.t_end_ms, fps)
        skip_end_frame = ms_to_frame(rec.t_start_ms + skip_ms, fps)
        trim_start_frame = ms_to_frame(rec.t_end_ms - end_trim_ms, fps)

        # Advance video when playing
        if not paused:
            if current_frame_idx >= frame_end_idx:
                # Reached end of this point's window — pause in place
                logger.info("End of point %d window — pausing.", rec.index)
                paused = True
            else:
                ret, frame = cap.read()
                if ret:
                    cached_frame = frame
                    current_frame_idx += 1
                else:
                    logger.warning(
                        "Unexpected read failure at frame %d — pausing.",
                        current_frame_idx,
                    )
                    paused = True

        # Build display frame
        display = (
            cached_frame.copy()
            if cached_frame is not None
            else np.zeros((sample_h, sample_w, 3), dtype=np.uint8)
        )

        elapsed_ms = int(current_frame_idx / fps * 1000)

        draw_excluded_overlay(
            display,
            current_frame_idx < skip_end_frame,
            "skip zone  (saccade + settling - excluded from inference)",
        )
        draw_excluded_overlay(
            display,
            current_frame_idx >= trim_start_frame,
            "end trim zone  (anticipatory saccade - excluded from inference)",
        )
        draw_top_info(display, rec, current_pt, len(records), paused, phase)
        draw_screen_inset(display, rec.x_norm, rec.y_norm)
        draw_timestamp(display, elapsed_ms)

        cv2.imshow(window_title, display)

        # waitKey: ~30 ms when playing for smooth video; 50 ms when paused (saves CPU)
        delay_ms = 33 if not paused else 50
        key = cv2.waitKey(delay_ms) & 0xFF

        # ── Key handling ─────────────────────────────────────────────────
        if key == ord("q"):
            logger.info("Quit requested.")
            break

        elif key == ord(" "):
            paused = not paused
            if paused:
                logger.debug("Paused at frame %d (%d ms)", current_frame_idx, elapsed_ms)

        elif key in (ord("["), 81, 2):
            # Previous point  ([ key, Linux left-arrow=81, macOS left-arrow & 0xFF = 2)
            if current_pt > 0:
                current_pt -= 1
                cached_frame, current_frame_idx = jump_to_point(current_pt)
                paused = True
            else:
                logger.debug("Already at first point.")

        elif key in (ord("]"), 83, 3):
            # Next point  (] key, Linux right-arrow=83, macOS right-arrow & 0xFF = 3)
            if current_pt < len(records) - 1:
                current_pt += 1
                cached_frame, current_frame_idx = jump_to_point(current_pt)
                paused = True
            else:
                logger.debug("Already at last point.")

        elif ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < len(records):
                current_pt = idx
                cached_frame, current_frame_idx = jump_to_point(current_pt)
                paused = True
            else:
                logger.debug("Point index %d out of range (max %d).", idx, len(records) - 1)

        elif ord("a") <= key <= ord("f"):
            idx = 10 + (key - ord("a"))
            if idx < len(records):
                current_pt = idx
                cached_frame, current_frame_idx = jump_to_point(current_pt)
                paused = True
            else:
                logger.debug("Point index %d out of range (max %d).", idx, len(records) - 1)

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive visualizer for gaze session alignment verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--session", required=True,
        help="Path to session folder",
    )
    p.add_argument(
        "--phase", default="calibration",
        choices=["calibration", "validation"],
        help="Which recording phase to visualize",
    )
    p.add_argument(
        "--skip-ms", type=float, default=1000.0,
        help="Duration of skip zone shown as overlay (ms)",
    )
    p.add_argument(
        "--end-trim-ms", type=float, default=500.0,
        help="Duration of end trim zone shown as overlay (ms)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
    )

    session_dir = Path(args.session).resolve()
    if not session_dir.is_dir():
        logger.error("Session folder not found: %s", session_dir)
        sys.exit(1)

    csv_path = session_dir / f"{args.phase}_events.csv"
    video_path = session_dir / f"{args.phase}.mp4"

    for path in (csv_path, video_path):
        if not path.exists():
            logger.error("Required file not found: %s", path)
            sys.exit(1)

    meta = load_session_meta(session_dir)
    fps = float(meta["camera_fps"])

    records = parse_event_csv(csv_path)
    logger.info(
        "Loaded %d %s points from %s",
        len(records),
        args.phase,
        csv_path.name,
    )

    run_viewer(records, video_path, fps, skip_ms=args.skip_ms, end_trim_ms=args.end_trim_ms, phase=args.phase)


if __name__ == "__main__":
    main()
