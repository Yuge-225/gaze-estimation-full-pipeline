"""
preprocess_mpiifacegaze.py — Convert raw MPIIFaceGaze to Image/ + Label/ format.

Raw layout (per participant):
    pXX/
        dayYY/NNNNN.jpg   (face images)
        pXX.txt           (28-column annotation, one line per image)

Output layout (expected by utils/datasets.py MPIIGaze class):
    <out>/
        Image/pXX/dayYY/NNNNN.jpg
        Label/pXX.label   (header + one line per image)

Label line columns (space-separated):
    0  image_path          pXX/dayYY/NNNNN.jpg  (relative to Image/)
    1  person              pXX
    2  day                 dayYY
    3  filename            NNNNN.jpg
    4  screen_x            gaze screen x (px)
    5  screen_y            gaze screen y (px)
    6  gaze3d              gx,gy,gz  (normalised gaze direction)
    7  gaze2d              yaw,pitch (radians)  ← read by MPIIGaze dataset class

Usage:
    python tools/preprocess_mpiifacegaze.py \\
        --src  data/MPIIFaceGaze \\
        --dst  data/MPIIFaceGaze_processed

    # Dry run (no file copying, just generate labels):
    python tools/preprocess_mpiifacegaze.py --src data/MPIIFaceGaze --dst /tmp/test --dry-run
"""

import argparse
import logging
import math
import shutil
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PARTICIPANTS = [f"p{i:02d}" for i in range(15)]  # p00 – p14
LABEL_HEADER = "image_path person day filename screen_x screen_y gaze3d gaze2d\n"


def gaze3d_to_angles(fc: list[float], gt: list[float]) -> tuple[float, float]:
    """
    Convert face-centre fc and gaze-target gt (both in camera coordinates)
    to (yaw, pitch) in radians.

    Convention matches Gazehub / MPIIFaceGaze preprocessing:
        gaze_dir = gt - fc  (unnormalised)
        normalise to unit vector [dx, dy, dz]
        pitch = arcsin(-dy)
        yaw   = arctan2(-dx, -dz)
    """
    dx = gt[0] - fc[0]
    dy = gt[1] - fc[1]
    dz = gt[2] - fc[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-9:
        return 0.0, 0.0
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    pitch = math.asin(-dy)
    yaw = math.atan2(-dx, -dz)
    return yaw, pitch


def parse_annotation_line(line: str) -> dict | None:
    """
    Parse one line of pXX.txt (28 whitespace-separated tokens).
    Returns None on malformed lines.
    """
    parts = line.strip().split()
    if len(parts) < 27:
        return None
    try:
        return {
            "rel_path": parts[0],               # e.g. day01/0005.jpg
            "screen_x": parts[1],
            "screen_y": parts[2],
            "fc": [float(parts[21]), float(parts[22]), float(parts[23])],
            "gt": [float(parts[24]), float(parts[25]), float(parts[26])],
        }
    except (ValueError, IndexError):
        return None


def process_participant(
    person: str,
    src_root: Path,
    dst_root: Path,
    dry_run: bool,
) -> int:
    src_person = src_root / person
    ann_file = src_person / f"{person}.txt"

    if not ann_file.exists():
        logger.warning("Annotation file not found: %s — skipping %s", ann_file, person)
        return 0

    dst_image_person = dst_root / "Image" / person
    dst_label_file = dst_root / "Label" / f"{person}.label"

    if not dry_run:
        dst_image_person.mkdir(parents=True, exist_ok=True)
        (dst_root / "Label").mkdir(parents=True, exist_ok=True)

    lines_out: list[str] = [LABEL_HEADER]
    n_ok = 0
    n_skip = 0

    with open(ann_file, encoding="utf-8") as f:
        raw_lines = f.readlines()

    for raw in raw_lines:
        parsed = parse_annotation_line(raw)
        if parsed is None:
            n_skip += 1
            continue

        rel_path = parsed["rel_path"]          # dayYY/NNNNN.jpg
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) != 2:
            n_skip += 1
            continue
        day, fname = parts

        src_img = src_person / day / fname
        if not src_img.exists():
            logger.debug("Image not found: %s", src_img)
            n_skip += 1
            continue

        yaw, pitch = gaze3d_to_angles(parsed["fc"], parsed["gt"])
        gaze2d_str = f"{yaw:.6f},{pitch:.6f}"
        fc, gt = parsed["fc"], parsed["gt"]
        dx = gt[0] - fc[0]; dy = gt[1] - fc[1]; dz = gt[2] - fc[2]
        norm = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
        gaze3d_str = f"{dx/norm:.6f},{dy/norm:.6f},{dz/norm:.6f}"

        img_rel = f"{person}/{day}/{fname}"   # relative to Image/
        label_line = (
            f"{img_rel} {person} {day} {fname} "
            f"{parsed['screen_x']} {parsed['screen_y']} "
            f"{gaze3d_str} {gaze2d_str}\n"
        )
        lines_out.append(label_line)

        if not dry_run:
            dst_img = dst_image_person / day / fname
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)

        n_ok += 1

    if not dry_run:
        with open(dst_label_file, "w", encoding="utf-8") as f:
            f.writelines(lines_out)

    logger.info(
        "%s: %d images processed, %d skipped%s",
        person, n_ok, n_skip,
        " [dry-run, no files written]" if dry_run else f" → {dst_label_file.name}",
    )
    return n_ok


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess raw MPIIFaceGaze → Image/ + Label/ for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, help="Path to raw MPIIFaceGaze folder (contains p00-p14)")
    p.add_argument("--dst", required=True, help="Output folder (will be created)")
    p.add_argument("--dry-run", action="store_true", help="Parse only, do not copy images or write labels")
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.is_dir():
        logger.error("Source folder not found: %s", src)
        raise SystemExit(1)

    logger.info("Source : %s", src)
    logger.info("Output : %s", dst)
    if args.dry_run:
        logger.info("DRY RUN — no files will be written")

    total = 0
    for person in PARTICIPANTS:
        total += process_participant(person, src, dst, dry_run=args.dry_run)

    logger.info("Done. Total images: %d", total)
    if not args.dry_run:
        logger.info("Image/ → %s", dst / "Image")
        logger.info("Label/ → %s", dst / "Label")


if __name__ == "__main__":
    main()
