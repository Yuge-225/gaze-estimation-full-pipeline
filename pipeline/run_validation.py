"""
run_validation.py — Phase 2 of the mobile gaze pipeline.

Re-fits the calibration model from calibration_events.csv, then runs gaze
inference on each validation point and computes accuracy metrics.

Usage:
    python pipeline/run_validation.py \\
        --session GazeData/session_20260504_210116_THH0FFPA \\
        --weight  weights/resnet34.pt

    # Same stride flag available as in run_calibration.py:
    python pipeline/run_validation.py --session ... --stride 3
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaze_utils import (
    CalibrationModel,
    PointInference,
    apply_calibration,
    build_transform,
    get_video_fps,
    infer_point,
    load_face_detector,
    load_gaze_model,
    load_session_meta,
    parse_event_csv,
)
from run_calibration import (
    build_calibration_model,
    parse_args as _cal_parse_args,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_validation(
    val_inferences: list[PointInference],
    cal_model: CalibrationModel,
    out_path: Path,
) -> None:
    """
    Save a scatter plot of predicted vs. ground-truth gaze positions for all
    validation points on a schematic of the phone screen.
    Per-point pixel error is annotated next to each arrow.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed — skipping validation plot. "
            "Install with: pip install matplotlib"
        )
        return

    sw, sh = cal_model.screen_w, cal_model.screen_h
    fig, ax = plt.subplots(figsize=(5, 9))

    ax.set_xlim(-sw * 0.05, sw * 1.05)
    ax.set_ylim(sh * 1.05, -sh * 0.05)
    ax.set_aspect("equal")
    ax.set_facecolor("#12121e")
    fig.patch.set_facecolor("#12121e")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_title(
        "Validation: predicted vs. ground truth",
        color="white",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("screen x (px)", color="#aaa", fontsize=9)
    ax.set_ylabel("screen y (px)", color="#aaa", fontsize=9)

    border = plt.Rectangle(
        (0, 0), sw, sh,
        linewidth=1, edgecolor="#555", facecolor="none",
    )
    ax.add_patch(border)

    for inf in val_inferences:
        xp_norm, yp_norm = apply_calibration(
            inf.pitch_mean, inf.yaw_mean, cal_model
        )
        xt_px = inf.record.x_norm * sw
        yt_px = inf.record.y_norm * sh
        xp_px = xp_norm * sw
        yp_px = yp_norm * sh
        err_px = float(
            np.sqrt((xp_px - xt_px) ** 2 + (yp_px - yt_px) ** 2)
        )

        # Ground truth dot
        ax.plot(xt_px, yt_px, "o", color="#4cc9f0", markersize=9, zorder=4)
        # Predicted cross
        ax.plot(
            xp_px, yp_px, "x",
            color="#f72585", markersize=9, markeredgewidth=2.5, zorder=4,
        )
        # Arrow: true → predicted
        ax.annotate(
            "",
            xy=(xp_px, yp_px),
            xytext=(xt_px, yt_px),
            arrowprops=dict(arrowstyle="-|>", color="#ffd166", lw=1.5),
            zorder=3,
        )
        # Error label
        ax.text(
            xt_px + sw * 0.015, yt_px - sh * 0.015,
            f"{err_px:.0f} px",
            color="#ffd166", fontsize=7, va="center", zorder=5,
        )

    legend_handles = [
        mpatches.Patch(color="#4cc9f0", label="Ground truth"),
        mpatches.Patch(color="#f72585", label="Predicted"),
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
    logger.info("Validation plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_and_log_metrics(
    val_inferences: list[PointInference],
    cal_model: CalibrationModel,
    n_total_points: int,
) -> None:
    """Compute and log validation accuracy metrics."""
    sw, sh = cal_model.screen_w, cal_model.screen_h

    errors_px: list[float] = []
    errors_norm: list[float] = []

    sep = "─" * 72
    logger.info(sep)
    logger.info("VALIDATION RESULTS")
    logger.info(sep)
    logger.info(
        "  %-5s  %-15s  %-15s  %-10s",
        "Pt",
        "Target (norm)",
        "Pred  (norm)",
        "Error (px)",
    )
    logger.info(sep)

    for inf in val_inferences:
        xp, yp = apply_calibration(inf.pitch_mean, inf.yaw_mean, cal_model)
        dx_px = (xp - inf.record.x_norm) * sw
        dy_px = (yp - inf.record.y_norm) * sh
        err_px = float(np.sqrt(dx_px**2 + dy_px**2))
        err_norm = float(
            np.sqrt(
                (xp - inf.record.x_norm) ** 2 + (yp - inf.record.y_norm) ** 2
            )
        )
        errors_px.append(err_px)
        errors_norm.append(err_norm)

        logger.info(
            "  %-5d  (%.3f, %.3f)    (%.3f, %.3f)    %7.1f px",
            inf.record.index,
            inf.record.x_norm,
            inf.record.y_norm,
            xp,
            yp,
            err_px,
        )

    logger.info(sep)
    logger.info(
        "  Points evaluated  : %d / %d",
        len(val_inferences),
        n_total_points,
    )
    logger.info("  Mean   error (px) : %.1f", np.mean(errors_px))
    logger.info("  Median error (px) : %.1f", np.median(errors_px))
    logger.info("  Std    error (px) : %.1f", np.std(errors_px))
    logger.info("  Max    error (px) : %.1f", np.max(errors_px))
    logger.info(
        "  Mean   error (norm): %.4f  (device-independent)",
        np.mean(errors_norm),
    )
    logger.info(sep)

    # Flag if any point exceeds a rough 20% screen-width threshold
    bad = [
        inf.record.index
        for inf, e in zip(val_inferences, errors_px)
        if e > sw * 0.20
    ]
    if bad:
        logger.warning(
            "Points with error > 20%% screen width (%.0f px): %s — "
            "consider re-running calibration.",
            sw * 0.20,
            bad,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validation phase: evaluate calibrated gaze model accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--session", required=True, help="Path to session folder")
    p.add_argument("--weight",  default="weights/resnet34.pt")
    p.add_argument("--model",   default="resnet34")
    p.add_argument("--dataset", default="gaze360")
    p.add_argument("--degree",  type=int, default=2)
    p.add_argument(
        "--skip-ms", type=float, default=1000.0,
        help="ms to skip after each point_start (applied to both cal. and val.)",
    )
    p.add_argument(
        "--end-trim-ms", type=float, default=500.0,
        help="ms to trim before each point_end (applied to both cal. and val.)",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Process every Nth frame (applied to both calibration and validation)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    session_dir = Path(args.session).resolve()
    if not session_dir.is_dir():
        logger.error("Session folder not found: %s", session_dir)
        sys.exit(1)

    val_video = session_dir / "validation.mp4"
    val_csv = session_dir / "validation_events.csv"
    for path in (val_video, val_csv):
        if not path.exists():
            logger.error("Required file not found: %s", path)
            sys.exit(1)

    weight_path = Path(args.weight).resolve()

    # Load model and detector once — shared across calibration + validation
    gaze_model, idx_tensor, binwidth, angle, device = load_gaze_model(
        weight_path, args.model, args.dataset
    )
    transform = build_transform()
    face_detector = load_face_detector()

    # ── Phase 1: Calibration ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1 — Calibration")
    logger.info("=" * 60)

    cal_inferences, cal_model = build_calibration_model(
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

    # ── Phase 2: Validation ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — Validation")
    logger.info("=" * 60)

    meta = load_session_meta(session_dir)
    fps = get_video_fps(val_video, fallback_fps=float(meta["camera_fps"]))
    val_records = parse_event_csv(val_csv)

    logger.info(
        "Running inference on %d validation points "
        "(skip_ms=%.0f, end_trim_ms=%.0f, stride=%d) ...",
        len(val_records),
        args.skip_ms,
        args.end_trim_ms,
        args.stride,
    )

    val_inferences: list[PointInference] = []
    skipped_indices: list[int] = []

    for rec in val_records:
        result = infer_point(
            val_video,
            rec,
            fps,
            face_detector,
            gaze_model,
            idx_tensor,
            binwidth,
            angle,
            device,
            transform,
            skip_ms=args.skip_ms,
            end_trim_ms=args.end_trim_ms,
            stride=args.stride,
        )
        if result is None:
            skipped_indices.append(rec.index)
        else:
            val_inferences.append(result)

    if skipped_indices:
        logger.warning(
            "%d validation point(s) had no valid face detections and were "
            "excluded from metrics: %s",
            len(skipped_indices),
            skipped_indices,
        )

    if not val_inferences:
        logger.error(
            "No valid validation inferences. Cannot compute metrics. "
            "Check validation.mp4 for face visibility."
        )
        sys.exit(1)

    compute_and_log_metrics(val_inferences, cal_model, len(val_records))

    plot_path = session_dir / "validation_scatter.png"
    plot_validation(val_inferences, cal_model, plot_path)


if __name__ == "__main__":
    main()
