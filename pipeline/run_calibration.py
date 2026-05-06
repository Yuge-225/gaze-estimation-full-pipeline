"""
run_calibration.py — Phase 1 of the mobile gaze pipeline.

Parses calibration_events.csv, runs gaze inference on each calibration point's
face-video window, fits a polynomial regression model, and outputs a quality
report + residual scatter plot.

Usage:
    python pipeline/run_calibration.py \\
        --session GazeData/session_20260504_210116_THH0FFPA \\
        --weight  weights/resnet34.pt

    # Process every 3rd frame (faster, still robust):
    python pipeline/run_calibration.py --session ... --stride 3
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make project root importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaze_utils import (
    CalibrationModel,
    PointInference,
    apply_calibration,
    build_transform,
    fit_calibration_model,
    get_video_fps,
    infer_point,
    load_face_detector,
    load_gaze_model,
    load_session_meta,
    parse_event_csv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_residuals(
    inferences: list[PointInference],
    cal_model: CalibrationModel,
    out_path: Path,
) -> None:
    """
    Save a residual scatter plot: ground truth vs in-sample predictions
    on a schematic of the phone screen. Arrows show direction and magnitude
    of each calibration point's residual error.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed — skipping residual plot. "
            "Install with: pip install matplotlib"
        )
        return

    sw, sh = cal_model.screen_w, cal_model.screen_h
    fig, ax = plt.subplots(figsize=(5, 9))

    ax.set_xlim(-sw * 0.05, sw * 1.05)
    ax.set_ylim(sh * 1.05, -sh * 0.05)  # y-axis flipped to match screen coords
    ax.set_aspect("equal")
    ax.set_facecolor("#12121e")
    fig.patch.set_facecolor("#12121e")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_title(
        "Calibration residuals  (in-sample)",
        color="white",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("screen x (px)", color="#aaa", fontsize=9)
    ax.set_ylabel("screen y (px)", color="#aaa", fontsize=9)

    # Screen border
    border = plt.Rectangle(
        (0, 0), sw, sh,
        linewidth=1, edgecolor="#555", facecolor="none",
    )
    ax.add_patch(border)

    for inf in inferences:
        xp_norm, yp_norm = apply_calibration(
            inf.pitch_mean, inf.yaw_mean, cal_model
        )
        xt_px = inf.record.x_norm * sw
        yt_px = inf.record.y_norm * sh
        xp_px = xp_norm * sw
        yp_px = yp_norm * sh

        # Ground truth
        ax.plot(xt_px, yt_px, "o", color="#4cc9f0", markersize=7, zorder=4)
        # Predicted
        ax.plot(
            xp_px, yp_px, "x",
            color="#f72585", markersize=7, markeredgewidth=2, zorder=4,
        )
        # Arrow: true → predicted
        ax.annotate(
            "",
            xy=(xp_px, yp_px),
            xytext=(xt_px, yt_px),
            arrowprops=dict(arrowstyle="-|>", color="#ffd166", lw=1.2),
            zorder=3,
        )
        # Point index label
        ax.text(
            xt_px + sw * 0.015, yt_px,
            str(inf.record.index),
            color="white", fontsize=7, va="center", zorder=5,
        )

    legend_handles = [
        mpatches.Patch(color="#4cc9f0", label="Ground truth"),
        mpatches.Patch(color="#f72585", label="Predicted (in-sample)"),
    ]
    ax.legend(
        handles=legend_handles,
        facecolor="#222",
        labelcolor="white",
        fontsize=8,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Residual plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# Core calibration logic (exported for use by run_validation.py)
# ---------------------------------------------------------------------------


def build_calibration_model(
    session_dir: Path,
    gaze_model,
    idx_tensor,
    binwidth: float,
    angle: float,
    device,
    transform,
    face_detector,
    degree: int = 2,
    skip_ms: float = 1000.0,
    end_trim_ms: float = 500.0,
    stride: int = 1,
) -> tuple[list[PointInference], CalibrationModel]:
    """
    Full calibration pipeline for one session:
      1. Parse calibration_events.csv
      2. Run gaze inference per point
      3. Fit polynomial regression model

    Returns (inferences, calibration_model).
    Raises RuntimeError if too few valid points are available.
    """
    csv_path = session_dir / "calibration_events.csv"
    video_path = session_dir / "calibration.mp4"

    if not video_path.exists():
        raise FileNotFoundError(
            f"Calibration video not found: {video_path}. "
            "Ensure the session folder contains calibration.mp4."
        )

    meta = load_session_meta(session_dir)
    fps = get_video_fps(video_path, fallback_fps=float(meta["camera_fps"]))
    records = parse_event_csv(csv_path)

    logger.info(
        "Running inference on %d calibration points "
        "(skip_ms=%.0f, end_trim_ms=%.0f, stride=%d) ...",
        len(records),
        skip_ms,
        end_trim_ms,
        stride,
    )

    inferences: list[PointInference] = []
    skipped_indices: list[int] = []

    for rec in records:
        result = infer_point(
            video_path,
            rec,
            fps,
            face_detector,
            gaze_model,
            idx_tensor,
            binwidth,
            angle,
            device,
            transform,
            skip_ms=skip_ms,
            end_trim_ms=end_trim_ms,
            stride=stride,
        )
        if result is None:
            skipped_indices.append(rec.index)
        else:
            inferences.append(result)

    if skipped_indices:
        logger.warning(
            "%d calibration point(s) had no valid face detections and were "
            "excluded from fitting: %s",
            len(skipped_indices),
            skipped_indices,
        )

    if not inferences:
        raise RuntimeError(
            "All calibration points failed face detection. "
            "Check video quality, lighting, and that the face is visible."
        )

    logger.info(
        "Fitting degree-%d polynomial on %d / %d points ...",
        degree,
        len(inferences),
        len(records),
    )
    cal_model = fit_calibration_model(inferences, degree=degree)
    return inferences, cal_model


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_calibration_summary(
    inferences: list[PointInference],
    cal_model: CalibrationModel,
) -> list[float]:
    """Log per-point residuals and aggregate stats. Returns list of errors (px)."""
    sw, sh = cal_model.screen_w, cal_model.screen_h
    sep = "─" * 72

    logger.info(sep)
    logger.info("CALIBRATION SUMMARY  (in-sample fit)")
    logger.info(sep)
    logger.info(
        "  %-5s  %-15s  %-15s  %-10s  %-12s",
        "Pt",
        "Target (norm)",
        "Pred  (norm)",
        "Error (px)",
        "Frames",
    )
    logger.info(sep)

    errors_px: list[float] = []
    for inf in inferences:
        xp, yp = apply_calibration(inf.pitch_mean, inf.yaw_mean, cal_model)
        dx = (xp - inf.record.x_norm) * sw
        dy = (yp - inf.record.y_norm) * sh
        err = float(np.sqrt(dx**2 + dy**2))
        errors_px.append(err)

        logger.info(
            "  %-5d  (%.3f, %.3f)    (%.3f, %.3f)    %7.1f px   %d/%d",
            inf.record.index,
            inf.record.x_norm,
            inf.record.y_norm,
            xp,
            yp,
            err,
            inf.n_valid,
            inf.n_total,
        )

    logger.info(sep)
    logger.info("  Points used       : %d / %d", len(inferences), len(inferences) + sum(1 for _ in []))
    logger.info("  Mean  error (px)  : %.1f", np.mean(errors_px))
    logger.info("  Median error (px) : %.1f", np.median(errors_px))
    logger.info("  Max   error (px)  : %.1f", np.max(errors_px))
    logger.info(sep)

    return errors_px


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibration phase: fit gaze model from mobile session data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--session", required=True,
        help="Path to session folder (e.g. GazeData/session_20260504_210116_THH0FFPA)",
    )
    p.add_argument(
        "--weight", default="weights/resnet34.pt",
        help="Path to gaze model weights (.pt file)",
    )
    p.add_argument("--model",   default="resnet34", help="Model architecture")
    p.add_argument("--dataset", default="gaze360",  help="Dataset config key")
    p.add_argument(
        "--degree", type=int, default=2,
        help="Polynomial degree for the gaze-to-screen mapping",
    )
    p.add_argument(
        "--skip-ms", type=float, default=1000.0,
        help="Milliseconds to skip after each point_start (saccade settling time)",
    )
    p.add_argument(
        "--end-trim-ms", type=float, default=500.0,
        help="Milliseconds to trim before each point_end (anticipatory saccade)",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Process every Nth frame (1 = every frame; 3 = ~3× faster)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    session_dir = Path(args.session).resolve()
    if not session_dir.is_dir():
        logger.error("Session folder not found: %s", session_dir)
        sys.exit(1)

    weight_path = Path(args.weight).resolve()

    # Load heavy resources once
    gaze_model, idx_tensor, binwidth, angle, device = load_gaze_model(
        weight_path, args.model, args.dataset
    )
    transform = build_transform()
    face_detector = load_face_detector()

    inferences, cal_model = build_calibration_model(
        session_dir,
        gaze_model,
        idx_tensor,
        binwidth,
        angle,
        device,
        transform,
        face_detector,
        degree=args.degree,
        skip_ms=args.skip_ms,
        end_trim_ms=args.end_trim_ms,
        stride=args.stride,
    )

    print_calibration_summary(inferences, cal_model)

    plot_path = session_dir / "calibration_residuals.png"
    plot_residuals(inferences, cal_model, plot_path)


if __name__ == "__main__":
    main()
