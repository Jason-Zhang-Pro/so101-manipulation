#!/usr/bin/env python3
"""Calibrate real top-camera red-cube pixels to SO101 base-frame XY.

Place the red cube at each prompted robot-base coordinate, then press ``s`` to
sample. The script fits an affine mapping:

    [x, y] = M @ [u, v, 1]

and saves it as JSON for ``classic_grasp_cube.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def detect_red_cube(bgr, args):
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    rgb_mask = (r > args.red_min) & (r > g * args.red_ratio) & (r > b * args.red_ratio)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hsv_mask = ((h <= args.hsv_red_high) | (h >= args.hsv_red_low)) & (
        s >= args.hsv_sat_min
    ) & (value >= args.hsv_value_min)

    if args.red_mask_mode == "strict":
        mask = rgb_mask & hsv_mask
    elif args.red_mask_mode == "hsv":
        mask = hsv_mask
    elif args.red_mask_mode == "rgb":
        mask = rgb_mask
    else:
        mask = rgb_mask | hsv_mask

    total = int(np.count_nonzero(mask))
    if total < args.min_red_pixels:
        return None, total, mask

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = []
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if args.min_red_pixels <= area <= args.max_red_pixels:
            candidates.append((area, label))
    if not candidates:
        return None, total, mask

    _, selected = max(candidates)
    ys, xs = np.nonzero(labels == selected)
    return (float(xs.mean()), float(ys.mean()), xs, ys), int(len(xs)), mask


def draw_text(frame, lines):
    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def parse_points(raw):
    points = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        x, y = [float(v.strip()) for v in item.split(",")]
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("Need at least 3 calibration points.")
    return points


def fit_affine(samples):
    pixels = np.array([[s["u"], s["v"], 1.0] for s in samples], dtype=np.float64)
    xy = np.array([[s["x"], s["y"]] for s in samples], dtype=np.float64)
    matrix_t, *_ = np.linalg.lstsq(pixels, xy, rcond=None)
    matrix = matrix_t.T
    pred = pixels @ matrix_t
    err = pred - xy
    rmse = np.sqrt(np.mean(np.sum(err * err, axis=1)))
    return matrix, float(rmse), pred


def open_camera(args):
    api = cv2.CAP_V4L2 if args.opencv_api == "v4l2" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, api)
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.camera}")
    for _ in range(args.warmup_frames):
        cap.read()
    return cap


def save_calibration(args, samples):
    matrix, rmse, pred = fit_affine(samples)
    payload = {
        "type": "affine_pixel_to_robot_xy",
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "fourcc": args.fourcc,
        "cube_center_z": args.cube_center_z,
        "matrix_2x3": matrix.tolist(),
        "rmse_m": rmse,
        "samples": samples,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved calibration: {output}")
    print(f"RMSE: {rmse * 1000:.1f} mm")
    for sample, point in zip(samples, pred):
        print(
            f"  pixel=({sample['u']:.1f},{sample['v']:.1f}) "
            f"target=({sample['x']:.4f},{sample['y']:.4f}) "
            f"pred=({point[0]:.4f},{point[1]:.4f})"
        )


def run(args):
    points = parse_points(args.points)
    cap = open_camera(args)
    samples = []
    point_index = 0
    current_detection = None

    print("Controls: s=sample current point, b=remove last, q=save+quit, Esc=quit without saving")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read camera frame.")

        detection, count, mask = detect_red_cube(frame, args)
        view = frame.copy()
        if mask is not None:
            overlay = view.copy()
            overlay[mask] = (0, 0, 255)
            view = cv2.addWeighted(overlay, 0.18, view, 0.82, 0.0)

        current_detection = None
        if detection is not None:
            u, v, xs, ys = detection
            current_detection = (u, v)
            cv2.rectangle(view, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 2)
            cv2.drawMarker(view, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)

        target_x, target_y = points[min(point_index, len(points) - 1)]
        lines = [
            f"camera: {args.camera} red_pixels={count}",
            f"sample {point_index + 1}/{len(points)}: place cube at x={target_x:.3f}, y={target_y:.3f} m",
            "press s when cube is placed and green cross is stable",
            f"collected: {len(samples)}",
        ]
        if current_detection is not None:
            lines.append(f"pixel u={current_detection[0]:.1f} v={current_detection[1]:.1f}")
        else:
            lines.append("cube: NOT DETECTED")
        draw_text(view, lines)
        cv2.imshow("SO101 real top camera calibration", view)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Quit without saving.")
            break
        if key == ord("b") and samples:
            removed = samples.pop()
            point_index = max(0, point_index - 1)
            print("Removed:", removed)
        if key == ord("s"):
            if current_detection is None:
                print("No red cube detected; not sampled.")
                continue
            u, v = current_detection
            sample = {"u": float(u), "v": float(v), "x": float(target_x), "y": float(target_y)}
            samples.append(sample)
            print("Sampled:", sample)
            point_index = min(point_index + 1, len(points) - 1)
            if len(samples) >= len(points):
                save_calibration(args, samples)
                break
        if key == ord("q"):
            if len(samples) >= 3:
                save_calibration(args, samples)
            else:
                print("Need at least 3 samples to save.")
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video2")
    parser.add_argument("--opencv-api", choices=["v4l2", "any"], default="v4l2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="YUYV")
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--cube-center-z", type=float, default=0.015)
    parser.add_argument("--output", default="config/real_top_camera_calib.json")
    parser.add_argument(
        "--points",
        default="0.30,0.00;0.30,0.05;0.30,-0.05;0.25,0.00;0.35,0.00",
        help="Semicolon-separated robot-base XY points in meters.",
    )

    parser.add_argument("--red-mask-mode", choices=["strict", "hsv", "rgb", "either"], default="strict")
    parser.add_argument("--red-min", type=int, default=60)
    parser.add_argument("--red-ratio", type=float, default=2.0)
    parser.add_argument("--hsv-red-low", type=int, default=165)
    parser.add_argument("--hsv-red-high", type=int, default=12)
    parser.add_argument("--hsv-sat-min", type=int, default=50)
    parser.add_argument("--hsv-value-min", type=int, default=25)
    parser.add_argument("--min-red-pixels", type=int, default=20)
    parser.add_argument("--max-red-pixels", type=int, default=30000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
