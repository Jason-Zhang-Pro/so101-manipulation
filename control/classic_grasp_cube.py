#!/usr/bin/env python3
"""Simple vision + MoveIt grasp script for the SO101 Gazebo cube world.

Pipeline:
- read /top_camera/image_raw and /top_camera/camera_info
- detect the red cube
- project the pixel onto the cube center height
- call MoveIt /compute_ik directly
- send joint targets to /forward_position_controller/commands
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_LOWER_LIMITS = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER_LIMITS = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])
CUBE_GRID_CENTERS = [(x, y) for x in (0.432, 0.512, 0.592) for y in (-0.274, -0.214, -0.154, -0.094, -0.034)]

# Real folded initial pose mapped to sim radians.
HOME_CLOSED = np.array([0.038, -1.739, 1.679, 1.249, 0.068, -0.054], dtype=np.float64)
ROBOT_BASE_IN_GAZEBO = np.array([0.2, -0.15, 0.75], dtype=np.float64)
DEFAULT_TOP_CAMERA_POSE = np.array([0.62, -0.155, 1.05, 3.1416, 1.308997, 3.1416], dtype=np.float64)
DEFAULT_PLACE_POSITION = np.array([0.1, -0.35, 0.055], dtype=np.float64)


def parse_duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def angle_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def clamp_angle_near(value: float, reference: float, max_delta: float) -> float:
    signed_delta = math.atan2(math.sin(value - reference), math.cos(value - reference))
    signed_delta = max(-max_delta, min(max_delta, signed_delta))
    return reference + signed_delta


def lerobot_norm_to_sim_rad(joint: str, value: float) -> float:
    low = JOINT_LOWER_LIMITS[JOINT_NAMES.index(joint)]
    high = JOINT_UPPER_LIMITS[JOINT_NAMES.index(joint)]
    if joint == "gripper":
        alpha = max(0.0, min(100.0, value)) / 100.0
    else:
        alpha = (max(-100.0, min(100.0, value)) + 100.0) / 200.0
    return low + alpha * (high - low)


def sim_rad_to_lerobot_norm(joint: str, value: float) -> float:
    low = JOINT_LOWER_LIMITS[JOINT_NAMES.index(joint)]
    high = JOINT_UPPER_LIMITS[JOINT_NAMES.index(joint)]
    alpha = (float(value) - low) / (high - low)
    if joint == "gripper":
        return max(0.0, min(100.0, alpha * 100.0))
    return max(-100.0, min(100.0, alpha * 200.0 - 100.0))


class SimpleVisionMoveItGrasper(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_simple_vision_grasper")
        self.args = args
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_camera_info = None
        self.latest_joint_state = None
        self.image_seq = 0
        self.real_robot = None
        self.opencv_camera = None
        self.real_camera_calib = self.load_real_camera_calib()

        self.cmd_pub = self.create_publisher(Float64MultiArray, args.command_topic, 10)
        if args.backend == "sim":
            self.create_subscription(JointState, args.joint_states_topic, self.joint_state_callback, 10)
        if args.image_source == "ros":
            self.create_subscription(Image, args.image_topic, self.image_callback, 10)
            self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, 10)
        self.ik_client = self.create_client(GetPositionIK, args.ik_service)
        if args.image_source == "opencv":
            self.open_opencv_camera()
        if args.backend == "real" and not args.real_connect_after_detection:
            self.connect_real_robot()

    def connect_real_robot(self):
        if self.real_robot is not None and self.real_robot.is_connected:
            return
        self.real_robot = SO101Follower(
            SO101FollowerConfig(
                id=self.args.real_robot_id,
                port=self.args.real_port,
                cameras={},
                max_relative_target=None,
                use_degrees=False,
            )
        )
        self.real_robot.connect()
        self.get_logger().info(f"connected real follower on {self.args.real_port}")

    def release_opencv_camera(self):
        if self.opencv_camera is not None:
            self.opencv_camera.release()
            self.opencv_camera = None
            self.get_logger().info("released OpenCV camera before connecting real follower")

    def ensure_execution_ready(self):
        if self.args.backend == "real" and self.real_robot is None:
            if self.args.real_release_camera_before_connect:
                self.release_opencv_camera()
            self.connect_real_robot()

    def open_opencv_camera(self):
        api = cv2.CAP_V4L2 if self.args.opencv_api == "v4l2" else cv2.CAP_ANY
        self.opencv_camera = cv2.VideoCapture(self.args.opencv_camera, api)
        if self.args.opencv_fourcc:
            self.opencv_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.args.opencv_fourcc))
        self.opencv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.opencv_width)
        self.opencv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.opencv_height)
        self.opencv_camera.set(cv2.CAP_PROP_FPS, self.args.opencv_fps)
        if not self.opencv_camera.isOpened():
            raise RuntimeError(f"Failed to open OpenCV camera: {self.args.opencv_camera}")
        actual_w = int(self.opencv_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.opencv_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.opencv_camera.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f"opened OpenCV camera {self.args.opencv_camera}: {actual_w}x{actual_h} "
            f"fps={actual_fps:.1f} fourcc={self.args.opencv_fourcc or 'default'} api={self.args.opencv_api}"
        )
        for _ in range(max(0, self.args.opencv_warmup_frames)):
            self.opencv_camera.read()

    def joint_state_callback(self, msg: JointState):
        with self.lock:
            self.latest_joint_state = msg

    def image_callback(self, msg: Image):
        with self.lock:
            self.latest_image = msg
            self.image_seq += 1

    def camera_info_callback(self, msg: CameraInfo):
        with self.lock:
            self.latest_camera_info = msg

    def wait_ready(self):
        self.get_logger().info(f"waiting for {self.args.ik_service}")
        if not self.ik_client.wait_for_service(timeout_sec=self.args.wait_timeout):
            raise RuntimeError("MoveIt IK service is not available. Start move_group first.")

        start = time.time()
        while time.time() - start < self.args.wait_timeout:
            with self.lock:
                image_ready = self.args.image_source == "opencv" or self.latest_image is not None
                joint_ready = self.args.backend == "real" or self.latest_joint_state is not None
                ready = image_ready and joint_ready
            if ready:
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("Timed out waiting for camera image and joint state.")

    def image_stamp_key(self, image):
        if image is None:
            return None
        return (int(image.header.stamp.sec), int(image.header.stamp.nanosec))

    def wait_for_fresh_image(self):
        if self.args.image_source == "opencv":
            return
        with self.lock:
            old_seq = self.image_seq
            old_stamp = self.image_stamp_key(self.latest_image)
        deadline = time.time() + self.args.fresh_image_timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            with self.lock:
                image = self.latest_image
                seq = self.image_seq
                stamp = self.image_stamp_key(image)
            if image is not None and seq > old_seq and (stamp is None or stamp != old_stamp):
                return
        self.get_logger().warning("Timed out waiting for fresh image; using latest frame.")

    def set_cube_position(self, position: np.ndarray):
        req = (
            'name: "cube" '
            f"position {{ x: {float(position[0])} y: {float(position[1])} z: {float(position[2])} }} "
            "orientation { w: 1.0 }"
        )
        cmd = [
            "ign",
            "service",
            "-s",
            f"/world/{self.args.world_name}/set_pose",
            "--reqtype",
            "ignition.msgs.Pose",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            req,
        ]
        completed = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        if completed.returncode != 0 or "data: true" not in completed.stdout:
            raise RuntimeError(
                f"Failed to set cube pose: rc={completed.returncode} "
                f"stdout={completed.stdout.strip()!r} stderr={completed.stderr.strip()!r}"
            )

    def maybe_randomize_cube(self):
        if not self.args.randomize_cube or self.args.backend == "real":
            return
        cube_x, cube_y = random.choice(CUBE_GRID_CENTERS)
        cube = np.array([cube_x, cube_y, self.args.cube_spawn_z], dtype=np.float64)
        self.get_logger().info(f"randomizing cube gazebo_world=[{cube[0]:.4f}, {cube[1]:.4f}, {cube[2]:.4f}]")
        self.set_cube_position(cube)
        time.sleep(self.args.cube_settle_s)
        self.wait_for_fresh_image()

    def current_joint_vector(self) -> np.ndarray:
        if self.args.backend == "real":
            raw = self.real_robot.bus.sync_read("Present_Position")
            return np.array([lerobot_norm_to_sim_rad(name, float(raw[name])) for name in JOINT_NAMES], dtype=np.float64)
        with self.lock:
            msg = self.latest_joint_state
        if msg is None:
            return HOME_CLOSED.copy()
        positions = dict(zip(msg.name, msg.position))
        return np.array([float(positions.get(name, HOME_CLOSED[i])) for i, name in enumerate(JOINT_NAMES)])

    def image_array(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        channels_by_encoding = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
        channels = channels_by_encoding.get(enc)
        if channels is None:
            raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")
        row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if enc in ("bgr8", "bgra8"):
            arr = arr[:, :, [2, 1, 0] + ([3] if channels == 4 else [])]
        return arr[:, :, :3]

    def read_opencv_image(self):
        ok, frame = self.opencv_camera.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame from {self.args.opencv_camera}")
        mean = float(frame.mean())
        max_value = int(frame.max())
        self.get_logger().info(f"OpenCV frame stats: mean={mean:.1f} max={max_value}")
        if mean < self.args.opencv_black_mean:
            cv2.imwrite("/tmp/so101_opencv_black_debug.jpg", frame)
            raise RuntimeError(
                f"OpenCV camera frame looks black: mean={mean:.1f} max={max_value}. "
                f"Saved /tmp/so101_opencv_black_debug.jpg"
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb, SimpleNamespace(width=rgb.shape[1], height=rgb.shape[0])

    def image_intrinsics(self, image: Image):
        with self.lock:
            info = self.latest_camera_info
        if info is not None and info.k[0] > 0.0 and info.k[4] > 0.0:
            return float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])
        fx = (image.width / 2.0) / math.tan(self.args.horizontal_fov / 2.0)
        return fx, fx, (image.width - 1.0) * 0.5, (image.height - 1.0) * 0.5

    def load_real_camera_calib(self):
        path = Path(self.args.real_camera_calib).expanduser()
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        matrix = np.asarray(payload["matrix_2x3"], dtype=np.float64)
        if matrix.shape != (2, 3):
            raise RuntimeError(f"Invalid real camera calibration matrix in {path}")
        self.get_logger().info(
            f"loaded real camera calibration {path} rmse={float(payload.get('rmse_m', -1.0)) * 1000:.1f}mm"
        )
        return payload

    def real_pixel_to_moveit(self, u: float, v: float) -> np.ndarray:
        if self.real_camera_calib is None:
            raise RuntimeError(
                f"Missing real camera calibration: {self.args.real_camera_calib}. "
                "Run tools/real_top_camera_calibrate.py first."
            )
        matrix = np.asarray(self.real_camera_calib["matrix_2x3"], dtype=np.float64)
        xy = matrix @ np.array([u, v, 1.0], dtype=np.float64)
        z = float(self.real_camera_calib.get("cube_center_z", self.args.real_cube_center_z))
        point = np.array([xy[0], xy[1], z], dtype=np.float64)
        point += np.asarray(self.args.grasp_offset, dtype=np.float64)
        self.get_logger().info(f"real calibrated cube moveit_frame=[{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
        return point

    def show_detection(self, rgb, mask, xs, ys, u, v, lines):
        if not self.args.display_vision:
            return
        frame = rgb[:, :, ::-1].copy()
        if mask is not None:
            overlay = frame.copy()
            overlay[mask] = (0, 0, 255)
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0)
        if xs is not None and len(xs) > 0:
            cv2.rectangle(frame, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 2)
        if u is not None and v is not None:
            cv2.drawMarker(frame, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
        y_text = 24
        for line in lines:
            cv2.putText(frame, line, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            y_text += 24
        cv2.imshow(self.args.vision_window_name, frame)
        key = cv2.waitKey(self.args.vision_wait_ms) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt

    def detect_cube_pixel(self):
        if self.args.image_source == "opencv":
            rgb, image = self.read_opencv_image()
        else:
            self.wait_for_fresh_image()
            with self.lock:
                image = self.latest_image
            if image is None:
                raise RuntimeError("No camera image received.")
            rgb = self.image_array(image)
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        rgb_mask = (r > self.args.red_min) & (r > g * self.args.red_ratio) & (r > b * self.args.red_ratio)

        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        hsv_mask = ((h <= self.args.hsv_red_high) | (h >= self.args.hsv_red_low)) & (
            s >= self.args.hsv_sat_min
        ) & (value >= self.args.hsv_value_min)

        if self.args.red_mask_mode == "strict":
            mask = rgb_mask & hsv_mask
        elif self.args.red_mask_mode == "hsv":
            mask = hsv_mask
        elif self.args.red_mask_mode == "rgb":
            mask = rgb_mask
        else:
            mask = rgb_mask | hsv_mask

        count = int(np.count_nonzero(mask))
        red_score = r - np.maximum(g, b)
        best_y, best_x = np.unravel_index(int(np.argmax(red_score)), red_score.shape)
        best_info = (
            f"best=({best_x},{best_y}) "
            f"rgb=({int(r[best_y,best_x])},{int(g[best_y,best_x])},{int(b[best_y,best_x])}) "
            f"hsv=({int(h[best_y,best_x])},{int(s[best_y,best_x])},{int(value[best_y,best_x])}) "
            f"score={int(red_score[best_y,best_x])}"
        )
        if count < self.args.min_red_pixels:
            cv2.imwrite("/tmp/so101_red_debug.jpg", rgb[:, :, ::-1])
            self.show_detection(rgb, mask, None, None, None, None, ["cube: NOT DETECTED", f"red_pixels={count}", best_info])
            raise RuntimeError(f"Red cube not detected; red_pixels={count}; {best_info}")

        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        candidates = []
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if self.args.min_red_pixels <= area <= self.args.max_red_pixels:
                candidates.append((area, label))
        if not candidates:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if label_count > 1 else 0
            cv2.imwrite("/tmp/so101_red_debug.jpg", rgb[:, :, ::-1])
            raise RuntimeError(f"Red detection ambiguous; total={count}; largest_component={largest}")

        _, selected = max(candidates)
        ys, xs = np.nonzero(labels == selected)
        u = float(xs.mean())
        v = float(ys.mean())
        self.get_logger().info(f"cube pixel u={u:.1f} v={v:.1f} red_pixels={len(xs)} {best_info}")
        self.show_detection(
            rgb,
            mask,
            xs,
            ys,
            u,
            v,
            ["cube: DETECTED", f"pixel u={u:.1f} v={v:.1f}", f"red_pixels={len(xs)}", best_info],
        )
        return u, v, image

    def pixel_to_gazebo_world(self, u: float, v: float, image: Image) -> np.ndarray:
        fx, fy, cx, cy = self.image_intrinsics(image)
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy

        ray_camera = np.array([1.0, -x_norm, -y_norm], dtype=np.float64)
        ray_camera /= np.linalg.norm(ray_camera)

        cam_pose = np.asarray(self.args.top_camera_pose, dtype=np.float64)
        cam_pos = cam_pose[:3]
        ray_world = rpy_to_matrix(*cam_pose[3:]) @ ray_camera
        if abs(ray_world[2]) < 1e-9:
            raise RuntimeError("Camera ray is parallel to cube plane.")
        t = (self.args.cube_center_z - cam_pos[2]) / ray_world[2]
        if t <= 0:
            raise RuntimeError(f"Projected cube is behind the camera; t={t:.4f}")

        point = cam_pos + t * ray_world
        self.get_logger().info(f"vision cube gazebo_world=[{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
        return point

    def gazebo_to_moveit(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64) - np.asarray(self.args.robot_base_in_gazebo, dtype=np.float64)
        point += np.asarray(self.args.grasp_offset, dtype=np.float64)
        self.get_logger().info(f"vision cube moveit_frame=[{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
        return point

    def make_pose(self, xyz: np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.args.planning_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        quat = np.asarray(self.args.tip_orientation_xyzw, dtype=np.float64)
        quat /= max(float(np.linalg.norm(quat)), 1e-9)
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        return pose

    def wrist_roll_reference(self, seed_q: np.ndarray) -> float:
        if self.args.wrist_roll_reference == "home":
            return float(HOME_CLOSED[4])
        if self.args.wrist_roll_reference == "zero":
            return 0.0
        return float(seed_q[4])

    def wrist_flex_reference(self, seed_q: np.ndarray) -> float:
        if self.args.wrist_flex_reference == "home":
            return float(HOME_CLOSED[3])
        if self.args.wrist_flex_reference == "zero":
            return 0.0
        if self.args.wrist_flex_reference == "target":
            return float(self.args.wrist_flex_target)
        return float(seed_q[3])

    def call_ik_once(self, target_xyz: np.ndarray, seed: np.ndarray, label: str):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.args.group
        req.ik_request.ik_link_name = self.args.tip_link
        req.ik_request.pose_stamped = self.make_pose(target_xyz)
        req.ik_request.timeout = parse_duration(self.args.ik_timeout)
        req.ik_request.avoid_collisions = self.args.avoid_collisions
        req.ik_request.robot_state.joint_state.name = JOINT_NAMES
        req.ik_request.robot_state.joint_state.position = [float(x) for x in seed]

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.ik_timeout + 1.0)
        if future.result() is None:
            raise RuntimeError(f"MoveIt IK timed out: {label}")
        res = future.result()
        if res.error_code.val != res.error_code.SUCCESS:
            return None, f"error_code={res.error_code.val}"

        solution = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        q = np.array([float(solution.get(name, seed[i])) for i, name in enumerate(JOINT_NAMES)], dtype=np.float64)
        return np.clip(q, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS), None

    def compute_ik(self, target_xyz: np.ndarray, seed_q: np.ndarray, gripper: float, label: str) -> np.ndarray:
        seed = np.asarray(seed_q, dtype=np.float64).copy()
        seed[5] = gripper

        wrist_flex_ref = self.wrist_flex_reference(seed)
        wrist_roll_ref = self.wrist_roll_reference(seed)
        candidates = []
        errors = []
        flex_seeds = [wrist_flex_ref, seed[3], HOME_CLOSED[3], self.args.wrist_flex_target]
        roll_seeds = [wrist_roll_ref, seed[4], HOME_CLOSED[4], 0.0]
        for wrist_flex in dict.fromkeys(float(x) for x in flex_seeds):
            for wrist_roll in dict.fromkeys(float(x) for x in roll_seeds):
                trial_seed = seed.copy()
                trial_seed[3] = np.clip(wrist_flex, JOINT_LOWER_LIMITS[3], JOINT_UPPER_LIMITS[3])
                trial_seed[4] = np.clip(wrist_roll, JOINT_LOWER_LIMITS[4], JOINT_UPPER_LIMITS[4])
                q, err = self.call_ik_once(target_xyz, trial_seed, label)
                if q is None:
                    errors.append(err)
                    continue
                q[5] = gripper
                flex_delta = abs(float(q[3] - wrist_flex_ref))
                roll_delta = angle_delta(float(q[4]), wrist_roll_ref)
                joint_delta = float(np.linalg.norm(q[:5] - seed[:5]))
                candidates.append((flex_delta, roll_delta, joint_delta, q))

        if not candidates:
            raise RuntimeError(f"MoveIt IK failed: {label}, errors={errors}")

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        flex_delta, roll_delta, joint_delta, q = candidates[0]
        if flex_delta > self.args.max_wrist_flex_delta:
            self.get_logger().warning(
                f"IK {label}: wrist_flex_delta={flex_delta:.3f} exceeds preferred "
                f"limit={self.args.max_wrist_flex_delta:.3f}; using best available IK."
            )
        if roll_delta > self.args.max_wrist_roll_delta:
            limited = clamp_angle_near(float(q[4]), wrist_roll_ref, self.args.max_wrist_roll_delta)
            self.get_logger().warning(
                f"IK {label}: limiting wrist_roll from {q[4]:.3f} to {limited:.3f} "
                f"around ref={wrist_roll_ref:.3f} limit={self.args.max_wrist_roll_delta:.3f} "
                f"(raw_delta={roll_delta:.3f})."
            )
            q[4] = np.clip(limited, JOINT_LOWER_LIMITS[4], JOINT_UPPER_LIMITS[4])
            roll_delta = angle_delta(float(q[4]), wrist_roll_ref)

        q[5] = gripper
        q[4] = np.clip(wrist_roll_ref, JOINT_LOWER_LIMITS[4], JOINT_UPPER_LIMITS[4])
        roll_delta = angle_delta(float(q[4]), wrist_roll_ref)
        q = np.clip(q, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)
        self.get_logger().info(
            f"IK {label}: wrist_flex_delta={flex_delta:.3f} wrist_roll_delta={roll_delta:.3f} "
            f"joint_delta={joint_delta:.3f} "
            + " ".join(f"{name}={q[i]:.3f}" for i, name in enumerate(JOINT_NAMES))
        )
        return q

    def send_joint_command(self, q: np.ndarray):
        if self.args.backend == "real":
            action = {f"{name}.pos": sim_rad_to_lerobot_norm(name, q[i]) for i, name in enumerate(JOINT_NAMES)}
            self.real_robot.send_action(action)
            return
        msg = Float64MultiArray()
        msg.data = [float(x) for x in np.asarray(q, dtype=np.float64)]
        self.cmd_pub.publish(msg)

    def wait_control_step(self):
        time.sleep(self.args.control_dt)

    def move_joints(self, target_q: np.ndarray, duration: float, label: str):
        rclpy.spin_once(self, timeout_sec=0.0)
        start_q = self.current_joint_vector()
        target_q = np.clip(np.asarray(target_q, dtype=np.float64), JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)
        base_steps = max(2, int(duration / self.args.control_dt))
        max_step = np.array([self.args.max_joint_speed * self.args.control_dt] * 5 + [self.args.max_gripper_speed * self.args.control_dt])
        speed_steps = int(np.max(np.ceil(np.abs(target_q - start_q) / max_step))) + 1
        steps = max(base_steps, speed_steps)

        self.get_logger().info(f"moving {label}: duration={duration:.2f}s steps={steps}")
        for alpha in np.linspace(0.0, 1.0, steps):
            smooth = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)
            q = (1.0 - smooth) * start_q + smooth * target_q
            self.send_joint_command(q)
            rclpy.spin_once(self, timeout_sec=0.0)
            self.wait_control_step()
        for _ in range(self.args.hold_steps):
            self.send_joint_command(target_q)
            rclpy.spin_once(self, timeout_sec=0.0)
            self.wait_control_step()

        actual = self.current_joint_vector()
        error = np.abs(actual - target_q)
        self.get_logger().info(
            f"joint follow error {label}: max={float(error.max()):.4f} "
            f"arm_max={float(error[:5].max()):.4f} gripper={float(error[5]):.4f}"
        )

    def hold_joints(self, q: np.ndarray, duration: float):
        steps = max(1, int(duration / self.args.control_dt))
        q = np.asarray(q, dtype=np.float64)
        for _ in range(steps):
            self.send_joint_command(q)
            rclpy.spin_once(self, timeout_sec=0.0)
            self.wait_control_step()

    def open_gripper(self, position: float, duration: float):
        q = self.current_joint_vector()
        q[5] = position
        self.move_joints(q, duration, "open_gripper")

    def close_gripper(self):
        q = self.current_joint_vector()
        q[5] = self.args.closed_gripper
        self.move_joints(q, self.args.gripper_duration, "close_gripper")
        self.hold_joints(q, self.args.gripper_hold_s)

    def grasp_once(self):
        self.wait_ready()
        self.maybe_randomize_cube()

        delayed_real = self.args.backend == "real" and self.args.real_connect_after_detection

        if self.args.move_home and not delayed_real:
            home = HOME_CLOSED.copy()
            home[5] = self.args.open_gripper
            self.move_joints(home, self.args.move_duration, "home_open")

        u, v, image = self.detect_cube_pixel()
        if self.args.backend == "real" and self.args.use_real_camera_calib:
            cube = self.real_pixel_to_moveit(u, v)
        else:
            cube = self.gazebo_to_moveit(self.pixel_to_gazebo_world(u, v, image))

        grasp = cube + np.asarray(self.args.grasp_target_offset, dtype=np.float64)
        cube_top_z = cube[2] + self.args.cube_size * 0.5
        grasp[2] = max(grasp[2] + self.args.grasp_z_offset, cube_top_z + self.args.grasp_top_clearance)
        if self.args.hover_above == "grasp":
            hover = grasp.copy()
            hover[2] = cube[2] + self.args.hover_height
        else:
            hover = cube + np.array([0.0, 0.0, self.args.hover_height])
        lift = grasp.copy()
        lift[2] = cube[2] + self.args.lift_height

        self.get_logger().info(
            f"grasp target=[{grasp[0]:.4f}, {grasp[1]:.4f}, {grasp[2]:.4f}] "
            f"hover=[{hover[0]:.4f}, {hover[1]:.4f}, {hover[2]:.4f}] "
            f"cube_center=[{cube[0]:.4f}, {cube[1]:.4f}, {cube[2]:.4f}] cube_top_z={cube_top_z:.4f}"
        )

        self.ensure_execution_ready()

        if self.args.move_home and delayed_real:
            home = HOME_CLOSED.copy()
            home[5] = self.args.open_gripper
            self.move_joints(home, self.args.move_duration, "home_open")

        seed = self.current_joint_vector()
        if self.args.solve_grasp_first:
            q_grasp = self.compute_ik(grasp, seed, self.args.approach_gripper, "grasp")
            q_hover = self.compute_ik(hover, q_grasp, self.args.open_gripper, "hover")
        else:
            q_hover = self.compute_ik(hover, seed, self.args.open_gripper, "hover")
            q_grasp = self.compute_ik(grasp, q_hover, self.args.approach_gripper, "grasp")

        self.move_joints(q_hover, self.args.move_duration, "hover")

        self.move_joints(q_grasp, self.args.descend_duration, "grasp")

        self.close_gripper()

        q_lift = self.compute_ik(lift, q_grasp, self.args.closed_gripper, "lift")
        self.move_joints(q_lift, self.args.lift_duration, "lift")

        if self.args.place:
            place = np.asarray(self.args.place_position, dtype=np.float64)
            place_above = place + np.array([0.0, 0.0, self.args.place_hover_height])
            q_place_above = self.compute_ik(place_above, q_lift, self.args.closed_gripper, "place_above")
            self.move_joints(q_place_above, self.args.place_move_duration, "place_above")
            self.hold_joints(q_place_above, self.args.place_settle_duration)
            q_place = self.compute_ik(place, q_place_above, self.args.closed_gripper, "place")
            self.move_joints(q_place, self.args.place_descend_duration, "place")
            self.hold_joints(q_place, self.args.place_before_open_duration)
            self.open_gripper(self.args.place_open_gripper, self.args.place_open_duration)

        if self.args.return_home:
            home = HOME_CLOSED.copy()
            home[5] = self.args.closed_gripper
            self.move_joints(home, self.args.return_home_duration, "return_home")

        self.get_logger().info("done")

    def close(self):
        if self.args.display_vision:
            cv2.destroyWindow(self.args.vision_window_name)
        if self.opencv_camera is not None:
            self.opencv_camera.release()
        if self.real_robot is not None and self.real_robot.is_connected:
            self.real_robot.disconnect()

    def camera_monitor_loop(self):
        self.get_logger().info("camera-only monitor mode; press q or Ctrl+C to stop")
        while rclpy.ok():
            try:
                u, v, image = self.detect_cube_pixel()
                point = self.pixel_to_gazebo_world(u, v, image)
                self.gazebo_to_moveit(point)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.get_logger().warning(str(exc))
                time.sleep(0.05)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--world-name", default="grasp_world")
    p.add_argument("--backend", choices=["sim", "real", "camera"], default="sim")
    p.add_argument("--real-port", default="/dev/ttyACM1")
    p.add_argument("--real-robot-id", default="my_awesome_follower_arm")
    p.add_argument("--real-connect-after-detection", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--real-release-camera-before-connect", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-real-camera-calib", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--real-camera-calib", default="config/real_top_camera_calib.json")
    p.add_argument("--real-cube-center-z", type=float, default=0.015)
    p.add_argument("--group", default="arm")
    p.add_argument("--tip-link", default="gripper_frame_link")
    p.add_argument("--planning-frame", default="world")
    p.add_argument("--ik-service", default="/compute_ik")
    p.add_argument("--avoid-collisions", action="store_true")
    p.add_argument("--joint-states-topic", default="/joint_states")
    p.add_argument("--command-topic", default="/forward_position_controller/commands")

    p.add_argument("--image-topic", default="/top_camera/image_raw")
    p.add_argument("--camera-info-topic", default="/top_camera/camera_info")
    p.add_argument("--image-source", choices=["ros", "opencv"], default="ros")
    p.add_argument("--opencv-camera", default="/dev/video2")
    p.add_argument("--opencv-api", choices=["v4l2", "any"], default="v4l2")
    p.add_argument("--opencv-width", type=int, default=1280)
    p.add_argument("--opencv-height", type=int, default=720)
    p.add_argument("--opencv-fps", type=int, default=30)
    p.add_argument("--opencv-fourcc", default="MJPG")
    p.add_argument("--opencv-warmup-frames", type=int, default=15)
    p.add_argument("--opencv-black-mean", type=float, default=5.0)
    p.add_argument("--horizontal-fov", type=float, default=1.745329)
    p.add_argument("--top-camera-pose", nargs=6, type=float, default=DEFAULT_TOP_CAMERA_POSE.tolist())
    p.add_argument("--robot-base-in-gazebo", nargs=3, type=float, default=ROBOT_BASE_IN_GAZEBO.tolist())
    p.add_argument("--cube-center-z", type=float, default=0.7655)

    p.add_argument("--randomize-cube", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cube-x-range", nargs=2, type=float, default=[0.49, 0.51])
    p.add_argument("--cube-y-range", nargs=2, type=float, default=[-0.03, 0.03])
    p.add_argument("--cube-spawn-z", type=float, default=0.7655)
    p.add_argument("--cube-settle-s", type=float, default=0.6)
    p.add_argument("--fresh-image-timeout", type=float, default=1.0)

    p.add_argument("--grasp-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    p.add_argument("--grasp-target-offset", nargs=3, type=float, default=[-0.035, 0.0, 0.0])
    p.add_argument("--cube-size", type=float, default=0.03)
    p.add_argument("--hover-height", type=float, default=0.10)
    p.add_argument("--hover-above", choices=["grasp", "cube"], default="grasp")
    p.add_argument("--grasp-top-clearance", type=float, default=-0.006)
    p.add_argument("--grasp-z-offset", type=float, default=0.0)
    p.add_argument("--lift-height", type=float, default=0.12)

    p.add_argument("--red-mask-mode", choices=["strict", "hsv", "rgb", "either"], default="strict")
    p.add_argument("--red-min", type=int, default=145)
    p.add_argument("--red-ratio", type=float, default=1.75)
    p.add_argument("--hsv-red-low", type=int, default=165)
    p.add_argument("--hsv-red-high", type=int, default=12)
    p.add_argument("--hsv-sat-min", type=int, default=100)
    p.add_argument("--hsv-value-min", type=int, default=70)
    p.add_argument("--min-red-pixels", type=int, default=60)
    p.add_argument("--max-red-pixels", type=int, default=30000)
    p.add_argument("--display-vision", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vision-window-name", default="SO101 top camera detection")
    p.add_argument("--vision-wait-ms", type=int, default=1)

    # MoveIt IK uses this as the desired tip orientation; [1, 0, 0, 0] keeps the gripper pointing down here.
    p.add_argument("--tip-orientation-xyzw", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0])
    p.add_argument("--wrist-flex-target", type=float, default=float(HOME_CLOSED[3]))
    p.add_argument("--max-wrist-flex-delta", type=float, default=0.45)
    p.add_argument("--wrist-flex-reference", choices=["seed", "home", "zero", "target"], default="target")
    p.add_argument("--max-wrist-roll-delta", type=float, default=0.15)
    p.add_argument("--wrist-roll-reference", choices=["seed", "home", "zero"], default="home")
    p.add_argument("--solve-grasp-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ik-timeout", type=float, default=1.0)
    p.add_argument("--wait-timeout", type=float, default=10.0)

    p.add_argument("--open-gripper", type=float, default=0.9)
    p.add_argument("--approach-gripper", type=float, default=0.9)
    p.add_argument("--closed-gripper", type=float, default=0.03)
    p.add_argument("--place-open-gripper", type=float, default=0.55)

    p.add_argument("--move-home", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--place", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--place-position", nargs=3, type=float, default=DEFAULT_PLACE_POSITION.tolist())
    p.add_argument("--place-hover-height", type=float, default=0.08)
    p.add_argument("--return-home", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--control-dt", type=float, default=0.02)
    p.add_argument("--hold-steps", type=int, default=8)
    p.add_argument("--max-joint-speed", type=float, default=0.45)
    p.add_argument("--max-gripper-speed", type=float, default=0.55)
    p.add_argument("--move-duration", type=float, default=1.8)
    p.add_argument("--descend-duration", type=float, default=2.2)
    p.add_argument("--lift-duration", type=float, default=1.8)
    p.add_argument("--gripper-duration", type=float, default=1.2)
    p.add_argument("--gripper-hold-s", type=float, default=0.4)
    p.add_argument("--place-move-duration", type=float, default=2.2)
    p.add_argument("--place-settle-duration", type=float, default=0.8)
    p.add_argument("--place-descend-duration", type=float, default=1.4)
    p.add_argument("--place-before-open-duration", type=float, default=0.8)
    p.add_argument("--place-open-duration", type=float, default=1.4)
    p.add_argument("--return-home-duration", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SimpleVisionMoveItGrasper(args)
    try:
        if args.backend == "camera":
            node.camera_monitor_loop()
        else:
            node.grasp_once()
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
