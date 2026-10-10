#!/usr/bin/env python3
"""Live red-cube visualizer for real OpenCV cameras or ROS image topics.

This tool never commands the robot. It only shows where the vision pipeline
thinks the red cube is.
"""

import argparse
import math
import threading
import time

import cv2
import numpy as np


try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
except ImportError:
    rclpy = None
    Node = object
    CameraInfo = None
    Image = None


ROBOT_BASE_IN_GAZEBO = np.array([0.2, 0.0, 0.75], dtype=np.float64)
DEFAULT_TOP_CAMERA_POSE = np.array([0.728, 0.0, 1.0, 3.1416, 1.1868, 3.1416], dtype=np.float64)


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def draw_text(image, lines):
    x, y = 12, 24
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def detect_red_cube(bgr, args):
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    rgb_mask = (r > args.red_min) & (r > g * args.red_ratio) & (r > b * args.red_ratio)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    value = hsv[:, :, 2]
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

    total_count = int(np.count_nonzero(mask))
    if total_count < args.min_red_pixels:
        return None, total_count

    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if args.min_red_pixels <= area <= args.max_red_pixels:
            candidates.append((area, label))

    if not candidates:
        return ("ambiguous", total_count, mask), total_count

    count, selected_label = max(candidates)
    ys, xs = np.nonzero(labels == selected_label)
    return (float(xs.mean()), float(ys.mean()), xs, ys, mask), count


def pixel_to_plane_xy(u, v, args):
    """Simple real-camera tabletop calibration: pixel center plus meters-per-pixel."""
    x = args.origin_x_m + (v - args.origin_v_px) * args.meter_per_pixel_y
    y = args.origin_y_m + (u - args.origin_u_px) * args.meter_per_pixel_x
    return np.array([x, y, args.cube_center_z], dtype=np.float64)


def image_from_ros_msg(msg):
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
    }
    encoding = msg.encoding.lower()
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")

    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if encoding in ("rgb8", "rgba8"):
        arr = arr[:, :, [2, 1, 0] + ([3] if channels == 4 else [])]
    return arr[:, :, :3].copy()


def gazebo_from_ros_pixel(u, v, image_width, image_height, camera_info, args):
    if camera_info is not None and camera_info.k[0] > 0.0 and camera_info.k[4] > 0.0:
        fx, fy, cx, cy = float(camera_info.k[0]), float(camera_info.k[4]), float(camera_info.k[2]), float(camera_info.k[5])
    else:
        fx = (image_width / 2.0) / math.tan(args.horizontal_fov / 2.0)
        fy, cx, cy = fx, (image_width - 1.0) / 2.0, (image_height - 1.0) / 2.0

    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    ray_camera = np.array([1.0, -x_norm, -y_norm], dtype=np.float64)
    ray_camera /= np.linalg.norm(ray_camera)

    cam_pose = np.asarray(args.top_camera_pose, dtype=np.float64)
    cam_pos = cam_pose[:3]
    ray_world = rpy_to_matrix(*cam_pose[3:]) @ ray_camera
    if abs(ray_world[2]) < 1e-6:
        return None
    scale = (args.cube_center_z - cam_pos[2]) / ray_world[2]
    return cam_pos + scale * ray_world


def render_frame(frame, source_label, args, ros_camera_info=None):
    detection, count = detect_red_cube(frame, args)
    lines = [source_label, f"red_pixels: {count}"]

    if detection is None:
        lines.append("cube: NOT DETECTED")
        draw_text(frame, lines)
        return frame, None

    if detection[0] == "ambiguous":
        _, total_count, mask = detection
        overlay = frame.copy()
        overlay[mask] = (0, 0, 255)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0.0)
        lines.append("cube: AMBIGUOUS RED AREA")
        lines.append(f"allowed component: {args.min_red_pixels}-{args.max_red_pixels}")
        draw_text(frame, lines)
        return frame, None

    u, v, xs, ys, mask = detection
    overlay = frame.copy()
    overlay[mask] = (0, 0, 255)
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0)
    cv2.rectangle(frame, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 2)
    cv2.drawMarker(frame, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(frame, (int(args.origin_u_px), int(args.origin_v_px)), 5, (255, 0, 255), -1)

    if args.source == "ros":
        gazebo = gazebo_from_ros_pixel(u, v, frame.shape[1], frame.shape[0], ros_camera_info, args)
        if gazebo is None:
            lines.append(f"pixel: u={u:.1f} v={v:.1f}")
            lines.append("ray: parallel to plane")
            draw_text(frame, lines)
            return frame, None
        moveit = gazebo - np.asarray(args.robot_base_in_gazebo, dtype=np.float64)
    else:
        moveit = pixel_to_plane_xy(u, v, args)
        gazebo = moveit + np.asarray(args.robot_base_in_gazebo, dtype=np.float64)

    grasp = moveit + np.asarray(args.grasp_target_offset, dtype=np.float64)
    lines.extend(
        [
            f"pixel: u={u:.1f} v={v:.1f}",
            f"cube moveit: x={moveit[0]:.4f} y={moveit[1]:.4f} z={moveit[2]:.4f}",
            f"grasp target: x={grasp[0]:.4f} y={grasp[1]:.4f} z={grasp[2]:.4f}",
            f"cube gazebo: x={gazebo[0]:.4f} y={gazebo[1]:.4f} z={gazebo[2]:.4f}",
        ]
    )
    draw_text(frame, lines)
    return frame, (u, v, moveit, grasp, gazebo)


def run_opencv(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.camera}")
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    for _ in range(args.warmup_frames):
        cap.read()

    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame from camera: {args.camera}")

        frame, result = render_frame(frame, f"opencv: {args.camera}", args)
        now = time.time()
        if result is not None and now - last_print > args.print_period:
            u, v, moveit, grasp, gazebo = result
            print(
                f"pixel=({u:.1f},{v:.1f}) "
                f"moveit=[{moveit[0]:.4f},{moveit[1]:.4f},{moveit[2]:.4f}] "
                f"grasp=[{grasp[0]:.4f},{grasp[1]:.4f},{grasp[2]:.4f}] "
                f"gazebo=[{gazebo[0]:.4f},{gazebo[1]:.4f},{gazebo[2]:.4f}]"
            )
            last_print = now

        cv2.imshow("SO101 real cube vision calibrator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


class RosImageViewer(Node):
    def __init__(self, args):
        super().__init__("cube_vision_calibrator")
        self.args = args
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_camera_info = None
        self.create_subscription(Image, args.image_topic, self.image_callback, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, 10)

    def image_callback(self, msg):
        with self.lock:
            self.latest_image = msg

    def camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def spin_view(self):
        last_print = 0.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)
            with self.lock:
                image_msg = self.latest_image
                camera_info = self.latest_camera_info
            if image_msg is None:
                continue

            frame = image_from_ros_msg(image_msg)
            frame, result = render_frame(frame, f"ros: {self.args.image_topic}", self.args, camera_info)
            now = time.time()
            if result is not None and now - last_print > self.args.print_period:
                u, v, moveit, grasp, gazebo = result
                self.get_logger().info(
                    f"pixel=({u:.1f},{v:.1f}) "
                    f"moveit=[{moveit[0]:.4f},{moveit[1]:.4f},{moveit[2]:.4f}] "
                    f"grasp=[{grasp[0]:.4f},{grasp[1]:.4f},{grasp[2]:.4f}] "
                    f"gazebo=[{gazebo[0]:.4f},{gazebo[1]:.4f},{gazebo[2]:.4f}]"
                )
                last_print = now

            cv2.imshow("SO101 ROS cube vision calibrator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["opencv", "ros"], default="opencv")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--warmup-frames", type=int, default=10)

    parser.add_argument("--image-topic", default="/top_camera/image_raw")
    parser.add_argument("--camera-info-topic", default="/top_camera/camera_info")
    parser.add_argument("--horizontal-fov", type=float, default=1.745329)
    parser.add_argument("--top-camera-pose", nargs=6, type=float, default=DEFAULT_TOP_CAMERA_POSE.tolist())
    parser.add_argument("--robot-base-in-gazebo", nargs=3, type=float, default=ROBOT_BASE_IN_GAZEBO.tolist())

    parser.add_argument("--cube-center-z", type=float, default=0.015)
    parser.add_argument("--origin-u-px", type=float, default=320.0)
    parser.add_argument("--origin-v-px", type=float, default=240.0)
    parser.add_argument("--origin-x-m", type=float, default=0.30)
    parser.add_argument("--origin-y-m", type=float, default=0.0)
    parser.add_argument("--meter-per-pixel-x", type=float, default=0.001)
    parser.add_argument("--meter-per-pixel-y", type=float, default=0.001)

    parser.add_argument("--grasp-target-offset", nargs=3, type=float, default=[-0.035, 0.0, 0.0])
    parser.add_argument("--red-mask-mode", choices=["strict", "hsv", "rgb", "either"], default="strict")
    parser.add_argument("--red-min", type=int, default=145)
    parser.add_argument("--red-ratio", type=float, default=1.75)
    parser.add_argument("--hsv-red-low", type=int, default=165)
    parser.add_argument("--hsv-red-high", type=int, default=12)
    parser.add_argument("--hsv-sat-min", type=int, default=100)
    parser.add_argument("--hsv-value-min", type=int, default=70)
    parser.add_argument("--min-red-pixels", type=int, default=60)
    parser.add_argument("--max-red-pixels", type=int, default=30000)
    parser.add_argument("--print-period", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source == "opencv":
        run_opencv(args)
        return

    if rclpy is None:
        raise RuntimeError("ROS mode needs rclpy. Use /usr/bin/python3 after sourcing ROS.")
    rclpy.init(args=None)
    node = RosImageViewer(args)
    try:
        node.spin_view()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
