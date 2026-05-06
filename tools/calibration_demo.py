"""
Gaze Calibration Demo
=====================
Phase 0: Face preview — ensure your face is visible, press SPACE to continue
Phase 1: 25-point fixed grid (5×5), 30-frame average per point
Phase 2: degree-2 polynomial fit f(pitch, yaw) -> (screen_x, screen_y)
Phase 3: Live smoothed cursor

Usage:
    python calibration_demo.py --model mobilenetv2 --weight weights/mobilenetv2.pt --source 1
"""

import argparse
import random
import string
import warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import data_config
from utils.helpers import get_model
from uniface import RetinaFace

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Calibration canvas constants
# ---------------------------------------------------------------------------

WINDOW         = "Gaze Calibration"
DOT_RADIUS     = 14
DOT_COLOR      = (0, 200, 255)
DOT_DONE_COLOR = (0, 160, 0)
CURSOR_COLOR   = (0, 0, 255)

# ---------------------------------------------------------------------------
# Gaze inference
# ---------------------------------------------------------------------------

def build_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_gaze(face_crop, model, idx_tensor, binwidth, angle, device, transform):
    img = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        yaw_raw, pitch_raw = model(img)
    yaw_prob   = F.softmax(yaw_raw,   dim=1)
    pitch_prob = F.softmax(pitch_raw, dim=1)
    yaw_deg   = (torch.sum(yaw_prob   * idx_tensor, dim=1) * binwidth - angle).item()
    pitch_deg = (torch.sum(pitch_prob * idx_tensor, dim=1) * binwidth - angle).item()
    return pitch_deg, yaw_deg


def get_face_crop(frame, face_detector):
    faces = face_detector.detect(frame)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

# ---------------------------------------------------------------------------
# Calibration grid
# ---------------------------------------------------------------------------

def make_grid_points(sw, sh, grid=5, margin=0.08):
    xs = [margin + (1 - 2*margin) * i / (grid-1) for i in range(grid)]
    ys = [margin + (1 - 2*margin) * i / (grid-1) for i in range(grid)]
    return [(int(x*sw), int(y*sh)) for y in ys for x in xs]

# ---------------------------------------------------------------------------
# Screen drawing
# ---------------------------------------------------------------------------

def draw_face_preview(canvas, frame, face_detector, sw, sh):
    """Draw camera feed with face detection status. Returns True if face detected."""
    canvas[:] = 20

    # Scale and center camera frame on canvas
    cam_h, cam_w = frame.shape[:2]
    scale = min(sw / cam_w, sh / cam_h) * 0.7
    new_w, new_h = int(cam_w * scale), int(cam_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    x0 = (sw - new_w) // 2
    y0 = (sh - new_h) // 2

    # Detect face and draw bounding box
    faces = face_detector.detect(frame)
    face_found = bool(faces)
    if face_found:
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        sx1, sy1 = int(x1 * scale), int(y1 * scale)
        sx2, sy2 = int(x2 * scale), int(y2 * scale)
        cv2.rectangle(resized, (sx1, sy1), (sx2, sy2), (0, 200, 255), 2)

    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    status = "Face detected - press SPACE to start calibration" if face_found \
             else "No face detected - adjust your position"
    color  = (0, 200, 100) if face_found else (0, 100, 255)
    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
    cv2.putText(canvas, status, ((sw - tw) // 2, y0 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Q to quit",
                (sw - 120, sh - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
    return face_found


def draw_calibration_screen(canvas, points, current_idx, sw, sh,
                             fixation_char="",
                             recording=False, frames_done=0, frames_total=30):
    canvas[:] = 25
    for pt in points[:current_idx]:
        cv2.circle(canvas, pt, 5, DOT_DONE_COLOR, -1)
    if current_idx < len(points):
        pt = points[current_idx]
        cv2.circle(canvas, pt, DOT_RADIUS, DOT_COLOR, -1)
        # Draw a random letter in the center so the eye has a precise target to fixate on
        if fixation_char:
            (tw, th), _ = cv2.getTextSize(fixation_char, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, fixation_char, (pt[0] - tw//2, pt[1] + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)
        if recording:
            angle_end = int(360 * frames_done / frames_total)
            cv2.ellipse(canvas, pt, (DOT_RADIUS+10, DOT_RADIUS+10),
                        -90, 0, angle_end, (255, 255, 255), 3)
        else:
            cv2.circle(canvas, pt, DOT_RADIUS+5, DOT_COLOR, 2)
    msg   = (f"Hold still... ({frames_done}/{frames_total})" if recording
             else "Stare at the dot, then press SPACE to record.")
    color = (180, 220, 255) if recording else (180, 180, 180)
    cv2.putText(canvas, msg, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Point {min(current_idx+1, len(points))} / {len(points)}",
                (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)


def draw_live_screen(canvas, gx, gy, sw, sh):
    canvas[:] = 20
    cv2.putText(canvas, "LIVE  |  Press Q to quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cx = int(np.clip(gx, 0, sw-1))
    cy = int(np.clip(gy, 0, sh-1))
    cv2.circle(canvas, (cx, cy), 18, CURSOR_COLOR, -1)
    cv2.circle(canvas, (cx, cy), 22, (255, 255, 255), 2)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def refit(collected_pitch, collected_yaw, collected_xy, degree):
    """Fit polynomial on all collected points so far; return (poly, rx, ry, rmse)."""
    feats   = np.column_stack([collected_pitch, collected_yaw])
    targets = np.array(collected_xy, dtype=float)
    poly    = PolynomialFeatures(degree=degree, include_bias=True)
    X       = poly.fit_transform(feats)
    rx      = Ridge(alpha=1.0).fit(X, targets[:, 0])
    ry      = Ridge(alpha=1.0).fit(X, targets[:, 1])
    rmse    = float(np.mean([
        np.sqrt(np.mean((rx.predict(X) - targets[:, 0])**2)),
        np.sqrt(np.mean((ry.predict(X) - targets[:, 1])**2)),
    ]))
    return poly, rx, ry, rmse

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   type=str, default="resnet34")
    p.add_argument("--weight",  type=str, default="resnet34.pt")
    p.add_argument("--source",  type=str, default="0")
    p.add_argument("--dataset", type=str, default="gaze360")
    p.add_argument("--grid",    type=int, default=5,
                   help="Grid size: 5 = 5×5 = 25 points (default: 5)")
    p.add_argument("--frames",  type=int, default=30,
                   help="Frames to average per calibration point (default: 30)")
    p.add_argument("--degree",  type=int, default=2,
                   help="Polynomial degree (default: 2)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dataset not in data_config:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    cfg = data_config[args.dataset]
    bins, binwidth, angle = cfg["bins"], cfg["binwidth"], cfg["angle"]

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_tensor = torch.arange(bins, dtype=torch.float32, device=device)
    transform  = build_transform()

    print(f"Loading model {args.model} from {args.weight} ...")
    gaze_model = get_model(args.model, bins, inference_mode=True)
    gaze_model.load_state_dict(torch.load(args.weight, map_location=device))
    gaze_model.to(device).eval()

    face_detector = RetinaFace()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {args.source}")

    # Screen size
    import tkinter as tk
    _r = tk.Tk(); _r.withdraw()
    sw, sh = _r.winfo_screenwidth(), _r.winfo_screenheight()
    _r.destroy()

    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    points = make_grid_points(sw, sh, grid=args.grid)
    n_pts  = len(points)

    # ── Phase 0: Face preview ─────────────────────────────────────────────
    print("Face preview: position yourself so your face is visible, then press SPACE.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        face_found = draw_face_preview(canvas, frame, face_detector, sw, sh)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return
        if key == ord(' ') and face_found:
            break

    # ── Phase 1: Calibration ──────────────────────────────────────────────
    collected_pitch, collected_yaw, collected_xy = [], [], []
    rmse_history  = []
    current_idx   = 0
    fixation_char = random.choice(string.ascii_uppercase)
    fixation_tick = 0  # increments every frame; char rotates every 3 frames

    def tick_fixation():
        nonlocal fixation_char, fixation_tick
        fixation_tick += 1
        if fixation_tick % 3 == 0:
            fixation_char = random.choice(string.ascii_uppercase)

    print(f"\n--- Calibration: {n_pts} points ({args.grid}×{args.grid}), "
          f"{args.frames} frames/point ---")
    print("Stare at each dot, press SPACE to record. Q to abort.\n")

    while current_idx < n_pts:
        tick_fixation()
        draw_calibration_screen(canvas, points, current_idx, sw, sh,
                                fixation_char=fixation_char)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Aborted.")
            cap.release(); cv2.destroyAllWindows(); return

        if key == ord(' '):
            pitches, yaws = [], []

            while len(pitches) < args.frames:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)

                crop = get_face_crop(frame, face_detector)
                if crop is None: continue

                p, y = predict_gaze(crop, gaze_model, idx_tensor,
                                    binwidth, angle, device, transform)
                pitches.append(p); yaws.append(y)

                tick_fixation()
                draw_calibration_screen(canvas, points, current_idx, sw, sh,
                                        fixation_char=fixation_char,
                                        recording=True,
                                        frames_done=len(pitches),
                                        frames_total=args.frames)
                cv2.imshow(WINDOW, canvas)
                cv2.waitKey(1)

            # Store point
            p_avg = float(np.mean(pitches))
            y_avg = float(np.mean(yaws))
            collected_pitch.append(p_avg)
            collected_yaw.append(y_avg)
            collected_xy.append(points[current_idx])

            # Intermediate refit to track RMSE
            if len(collected_pitch) >= 2:
                _, _, _, rmse = refit(
                    collected_pitch, collected_yaw, collected_xy, args.degree)
                rmse_history.append(rmse)

            print(f"  [{current_idx+1:2d}/{n_pts}] "
                  f"screen=({points[current_idx][0]:4d},{points[current_idx][1]:4d})  "
                  f"pitch={p_avg:+.2f}°  yaw={y_avg:+.2f}°"
                  + (f"  RMSE={rmse_history[-1]:.1f}px" if rmse_history else ""))
            current_idx += 1

    # ── Phase 2: Final fit ────────────────────────────────────────────────
    print(f"\n--- Final fit on {n_pts} points ---")
    poly, reg_x, reg_y, _ = refit(collected_pitch, collected_yaw, collected_xy, args.degree)

    feats  = np.column_stack([collected_pitch, collected_yaw])
    X      = poly.transform(feats)
    tgts   = np.array(collected_xy, dtype=float)
    rmse_x = np.sqrt(np.mean((reg_x.predict(X) - tgts[:, 0])**2))
    rmse_y = np.sqrt(np.mean((reg_y.predict(X) - tgts[:, 1])**2))
    print(f"Calibration RMSE: x={rmse_x:.1f}px  y={rmse_y:.1f}px")

    # ── Phase 3: Live ─────────────────────────────────────────────────────
    print("Starting live mode. Press Q to quit.")
    smooth_n = 8
    gx_buf, gy_buf = [], []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        crop = get_face_crop(frame, face_detector)
        if crop is not None:
            p, y = predict_gaze(crop, gaze_model, idx_tensor,
                                binwidth, angle, device, transform)
            feat = poly.transform([[p, y]])
            gx_buf.append(reg_x.predict(feat)[0])
            gy_buf.append(reg_y.predict(feat)[0])
            if len(gx_buf) > smooth_n: gx_buf.pop(0); gy_buf.pop(0)

        gx = int(np.mean(gx_buf)) if gx_buf else sw//2
        gy = int(np.mean(gy_buf)) if gy_buf else sh//2

        draw_live_screen(canvas, gx, gy, sw, sh)
        cv2.imshow(WINDOW, canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
