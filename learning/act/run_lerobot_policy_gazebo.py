#!/usr/bin/env python3
"""Run a LeRobot ACT policy against the SO101 Gazebo ROS2 controller."""

from __future__ import annotations

import argparse
import os
from collections import deque
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if os.environ.get("LEROBOT_SRC"):
    sys.path.insert(0, str(Path(os.environ["LEROBOT_SRC"]).expanduser()))

import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Limits from worlds/so101.sdf. LeRobot SO101 arm motors are normalized to
# [-100, 100], while the gripper is normalized to [0, 100].
LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986, -100.0, 100.0),
    "shoulder_lift": (-1.74533, 1.74533, -100.0, 100.0),
    "elbow_flex": (-1.69, 1.69, -100.0, 100.0),
    "wrist_flex": (-1.65806, 1.65806, -100.0, 100.0),
    "wrist_roll": (-2.74385, 2.84121, -100.0, 100.0),
    "gripper": (-0.174533, 1.74533, 0.0, 100.0),
}

# Measured zero-point correction between the real LeRobot calibration and the
# reachable Gazebo folded pose. Apply it in both directions so observations and
# commands remain in the same coordinate system as the training dataset.
SIM_CALIBRATION_OFFSET = {
    "shoulder_pan": 0.0184,
    "shoulder_lift": 0.0,
    "elbow_flex": -0.0887,
    "wrist_flex": 0.0202,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
CUBE_GRID_CENTERS = [(x, y) for x in (0.432, 0.512, 0.592) for y in (-0.274, -0.214, -0.154, -0.094, -0.034)]

# RGB statistics measured over the first frame of all 150 real demonstrations.
TRAIN_IMAGE_STATS = {
    "fixed": (
        np.array([43.3143, 38.2112, 41.9451], dtype=np.float32),
        np.array([41.0665, 39.0702, 38.1991], dtype=np.float32),
    ),
    "handeye": (
        np.array([21.1151, 18.1680, 22.2743], dtype=np.float32),
        np.array([53.3007, 49.4228, 50.1857], dtype=np.float32),
    ),
}


def lerobot_to_rad(name: str, value: float) -> float:
    lo, hi, nlo, nhi = LIMITS[name]
    value = float(np.clip(value, nlo, nhi))
    nominal = lo + (value - nlo) * (hi - lo) / (nhi - nlo)
    return float(np.clip(nominal + SIM_CALIBRATION_OFFSET[name], lo, hi))


def rad_to_lerobot(name: str, value: float) -> float:
    lo, hi, nlo, nhi = LIMITS[name]
    value = float(np.clip(value - SIM_CALIBRATION_OFFSET[name], lo, hi))
    return nlo + (value - lo) * (nhi - nlo) / (hi - lo)


def image_msg_to_rgb(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()

    if enc in ("rgb8", "bgr8"):
        arr = arr.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    if enc in ("rgba8", "bgra8"):
        arr = arr.reshape(h, msg.step)[:, : w * 4].reshape(h, w, 4)
        code = cv2.COLOR_RGBA2RGB if enc == "rgba8" else cv2.COLOR_BGRA2RGB
        return cv2.cvtColor(arr, code)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def match_image_moments(image: np.ndarray, camera: str) -> np.ndarray:
    """Match global RGB moments to the real training camera distribution."""
    target_mean, target_std = TRAIN_IMAGE_STATS[camera]
    pixels = image.astype(np.float32)
    source_mean = pixels.mean(axis=(0, 1))
    source_std = pixels.std(axis=(0, 1))
    scale = np.clip(target_std / np.maximum(source_std, 1.0), 0.35, 2.5)
    matched = (pixels - source_mean) * scale + target_mean
    return np.clip(matched, 0, 255).astype(np.uint8)


class GazeboLeRobotRunner(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("gazebo_lerobot_policy_runner")
        self.args = args
        self.fixed_image = None
        self.handeye_image = None
        self.fixed_frames = deque(maxlen=20)
        self.handeye_frames = deque(maxlen=20)
        self.fixed_stamp = None
        self.handeye_stamp = None
        self.fixed_received_at = None
        self.handeye_received_at = None
        self.observation_stamp = None
        self.joint_pos = {}

        self.create_subscription(JointState, args.joint_states_topic, self.on_joint_states, 10)
        self.create_subscription(Image, args.fixed_image_topic, self.on_fixed_image, 10)
        self.create_subscription(Image, args.handeye_image_topic, self.on_handeye_image, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, args.command_topic, 10)

        cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        cfg.pretrained_path = args.policy_path
        cfg.device = args.device
        ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
        self.policy = make_policy(cfg, ds_meta=ds_meta)
        self.policy.reset()
        self.device = get_safe_torch_device(cfg.device)
        self.use_amp = cfg.use_amp

        self.get_logger().info(f"Loaded policy: {args.policy_path}")
        self.get_logger().info(f"Using dataset stats: {ds_meta.root}")

    def on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINTS:
                self.joint_pos[name] = float(pos)

    def on_fixed_image(self, msg: Image):
        self.fixed_image = image_msg_to_rgb(msg)
        self.fixed_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.fixed_received_at = time.monotonic()
        self.fixed_frames.append((self.fixed_stamp, self.fixed_received_at, self.fixed_image))

    def on_handeye_image(self, msg: Image):
        self.handeye_image = image_msg_to_rgb(msg)
        self.handeye_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.handeye_received_at = time.monotonic()
        self.handeye_frames.append((self.handeye_stamp, self.handeye_received_at, self.handeye_image))

    def ready(self) -> bool:
        return (
            all(j in self.joint_pos for j in JOINTS)
            and self.fixed_image is not None
            and self.handeye_image is not None
        )

    def observation(self) -> dict[str, np.ndarray]:
        if not self.fixed_frames or not self.handeye_frames:
            raise RuntimeError("Camera frame buffers are empty")
        pairs = [(fixed, handeye) for fixed in self.fixed_frames for handeye in self.handeye_frames]
        synchronized = [
            pair for pair in pairs if abs(pair[0][0] - pair[1][0]) <= self.args.max_camera_skew_s
        ]
        if synchronized:
            fixed_frame, handeye_frame = max(
                synchronized, key=lambda pair: min(pair[0][0], pair[1][0])
            )
        else:
            fixed_frame, handeye_frame = min(
                pairs, key=lambda pair: abs(pair[0][0] - pair[1][0])
            )
        fixed_stamp, fixed_received_at, fixed_image = fixed_frame
        handeye_stamp, handeye_received_at, handeye_image = handeye_frame
        now = time.monotonic()
        ages = (now - fixed_received_at, now - handeye_received_at)
        if max(ages) > self.args.image_disconnect_s:
            raise RuntimeError(f"Stale camera input: fixed_age={ages[0]:.3f}s handeye_age={ages[1]:.3f}s")
        if max(ages) > self.args.max_image_age_s:
            self.get_logger().warning(
                f"Camera callback is late: fixed_age={ages[0]:.3f}s handeye_age={ages[1]:.3f}s"
            )
        skew = abs(fixed_stamp - handeye_stamp)
        self.observation_stamp = min(fixed_stamp, handeye_stamp)
        if skew > self.args.max_camera_skew_s:
            self.get_logger().warning(
                f"Camera timestamps differ by {skew:.3f}s (limit={self.args.max_camera_skew_s:.3f}s)"
            )
        state = np.array([rad_to_lerobot(j, self.joint_pos[j]) for j in JOINTS], dtype=np.float32)
        fixed = cv2.resize(fixed_image, (640, 480), interpolation=cv2.INTER_AREA)
        handeye = cv2.resize(handeye_image, (640, 480), interpolation=cv2.INTER_AREA)
        if self.args.match_training_photometry:
            fixed = match_image_moments(fixed, "fixed")
            handeye = match_image_moments(handeye, "handeye")
        return {
            "observation.state": state,
            "observation.images.fixed": fixed,
            "observation.images.handeye": handeye,
        }

    def publish_action(self, action: np.ndarray):
        target = [lerobot_to_rad(j, action[i]) for i, j in enumerate(JOINTS)]

        if self.args.max_delta_rad is not None:
            current = np.array([self.joint_pos[j] for j in JOINTS], dtype=np.float64)
            target_np = np.array(target, dtype=np.float64)
            target_np = current + np.clip(target_np - current, -self.args.max_delta_rad, self.args.max_delta_rad)
            target = target_np.tolist()

        if self.args.dry_run:
            pretty = " ".join(f"{j}={action[i]:7.2f}" for i, j in enumerate(JOINTS))
            self.get_logger().info(f"dry-run policy action: {pretty}")
            return

        msg = Float64MultiArray()
        msg.data = target
        self.cmd_pub.publish(msg)

    def publish_rad_target(self, target: list[float]):
        if self.args.dry_run:
            pretty = " ".join(f"{j}={rad_to_lerobot(j, target[i]):7.2f}" for i, j in enumerate(JOINTS))
            self.get_logger().info(f"dry-run home target: {pretty}")
            return
        msg = Float64MultiArray()
        msg.data = [float(x) for x in target]
        self.cmd_pub.publish(msg)

    def set_cube_position(self, position: np.ndarray):
        req = (
            'name: "cube" '
            f'position {{ x: {float(position[0])} y: {float(position[1])} z: {float(position[2])} }} '
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
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3.0,
        )
        ok = completed.returncode == 0 and "data: true" in completed.stdout
        if not ok:
            raise RuntimeError(
                "Failed to set cube pose: "
                f"returncode={completed.returncode} "
                f"stdout={completed.stdout.strip()!r} "
                f"stderr={completed.stderr.strip()!r}"
            )

    def maybe_randomize_cube(self):
        if not self.args.randomize_cube:
            return
        cube_x, cube_y = random.choice(CUBE_GRID_CENTERS)
        cube_position = np.array([cube_x, cube_y, self.args.cube_spawn_z], dtype=np.float64)
        self.get_logger().info(
            f"randomizing cube gazebo_world=[{cube_position[0]:.4f}, "
            f"{cube_position[1]:.4f}, {cube_position[2]:.4f}]"
        )
        self.set_cube_position(cube_position)
        time.sleep(self.args.cube_settle_s)

    def move_to_real_home(self):
        if not self.args.move_home:
            return
        home_norm = np.array([float(x) for x in self.args.home_norm.split(",")], dtype=np.float64)
        if home_norm.shape != (len(JOINTS),):
            raise ValueError(f"--home-norm must contain {len(JOINTS)} comma-separated values")

        start = np.array([self.joint_pos[j] for j in JOINTS], dtype=np.float64)
        goal = np.array([lerobot_to_rad(j, home_norm[i]) for i, j in enumerate(JOINTS)], dtype=np.float64)
        steps = max(2, int(self.args.home_duration_s * self.args.fps))
        self.get_logger().info(
            "Moving sim to real initial pose: "
            + " ".join(f"{j}={home_norm[i]:.2f}" for i, j in enumerate(JOINTS))
        )
        for k in range(steps):
            rclpy.spin_once(self, timeout_sec=0.0)
            alpha = (k + 1) / steps
            # Smoothstep avoids a little snap at start/end.
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target = (1.0 - alpha) * start + alpha * goal
            self.publish_rad_target(target.tolist())
            time.sleep(1.0 / self.args.fps)

        deadline = time.monotonic() + self.args.home_settle_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_rad_target(goal.tolist())
            rclpy.spin_once(self, timeout_sec=0.02)
            actual = np.array([self.joint_pos[j] for j in JOINTS], dtype=np.float64)
            error = np.abs(actual - goal)
            if float(error.max()) <= self.args.home_tolerance_rad:
                break
            time.sleep(1.0 / self.args.fps)
        else:
            actual_norm = np.array(
                [rad_to_lerobot(j, self.joint_pos[j]) for j in JOINTS], dtype=np.float64
            )
            error_norm = np.abs(actual_norm - home_norm)
            raise RuntimeError(
                "Simulation failed to reach the training initial pose. "
                f"target={np.round(home_norm, 2).tolist()} "
                f"actual={np.round(actual_norm, 2).tolist()} "
                f"max_normalized_error={float(error_norm.max()):.2f}"
            )

    def run(self):
        deadline = time.time() + self.args.wait_timeout
        while rclpy.ok() and not self.ready() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not self.ready():
            raise RuntimeError(
                "Timed out waiting for /joint_states and camera images. "
                "Start Gazebo, controllers, and camera bridge first."
            )

        self.maybe_randomize_cube()
        self.move_to_real_home()

        # The first CUDA forwards are slower due to kernel/library setup. Warm
        # them up before timing the control loop, then clear policy history.
        for _ in range(self.args.warmup_steps):
            rclpy.spin_once(self, timeout_sec=0.02)
            predict_action(
                self.observation(),
                self.policy,
                self.device,
                self.use_amp,
                task=self.args.task,
                robot_type="so101_follower",
            )
        self.policy.reset()
        rclpy.spin_once(self, timeout_sec=0.05)

        period = 1.0 / self.args.fps
        first_obs = self.observation()
        start_sim_stamp = self.observation_stamp
        last_step_stamp = None
        start_wall = time.monotonic()
        wall_deadline = start_wall + max(self.args.wait_timeout, self.args.duration_s * 10.0)
        self.get_logger().info(
            f"Running policy for {self.args.duration_s:.1f} simulation seconds at {self.args.fps} fps "
            f"dry_run={self.args.dry_run}"
        )

        frame_index = 0
        obs = first_obs
        while rclpy.ok():
            if time.monotonic() >= wall_deadline:
                raise RuntimeError(
                    "Timed out while waiting for Gazebo camera time to advance; "
                    "check that the simulation is unpaused and both camera bridges are running."
                )
            if frame_index > 0:
                rclpy.spin_once(self, timeout_sec=0.01)
                obs = self.observation()
            stamp = self.observation_stamp
            if stamp is None:
                continue
            if stamp - start_sim_stamp >= self.args.duration_s:
                break
            if last_step_stamp is not None and stamp - last_step_stamp < period * 0.9:
                continue
            action = predict_action(
                obs,
                self.policy,
                self.device,
                self.use_amp,
                task=self.args.task,
                robot_type="so101_follower",
            ).numpy()
            if frame_index == 0 and self.args.max_initial_action_delta is not None:
                initial_delta = np.abs(action - obs["observation.state"])
                arm_delta = float(initial_delta[:-1].max())
                gripper_delta = float(initial_delta[-1])
                if (
                    arm_delta > self.args.max_initial_action_delta
                    or gripper_delta > self.args.max_initial_gripper_delta
                ):
                    raise RuntimeError(
                        "Unsafe first policy action; refusing to command the robot. "
                        f"arm_max_delta={arm_delta:.2f} "
                        f"gripper_delta={gripper_delta:.2f} "
                        f"state={np.round(obs['observation.state'], 2).tolist()} "
                        f"action={np.round(action, 2).tolist()} "
                        f"delta={np.round(initial_delta, 2).tolist()}"
                    )
            if frame_index % self.args.log_every == 0:
                state = obs["observation.state"]
                self.get_logger().info(
                    f"step={frame_index} "
                    f"state={np.round(state, 2).tolist()} "
                    f"action={np.round(action, 2).tolist()}"
                )
            self.publish_action(action)

            if self.args.display:
                bgr = cv2.cvtColor(obs["observation.images.fixed"], cv2.COLOR_RGB2BGR)
                cv2.imshow("fixed -> ACT", bgr)
                if cv2.waitKey(1) == 27:
                    break

            last_step_stamp = stamp
            frame_index += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", required=True)
    p.add_argument("--dataset-repo-id", default="jasonz/so101_pick_red_v3")
    p.add_argument("--dataset-root", required=True, help="Local dataset directory matching the checkpoint")
    p.add_argument("--task", default="Pick up the red block and place it in the target area")
    p.add_argument("--device", default="cuda")
    p.add_argument("--duration-s", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--log-every", type=int, default=15)
    p.add_argument("--wait-timeout", type=float, default=10.0)
    p.add_argument("--max-image-age-s", type=float, default=0.25)
    p.add_argument("--image-disconnect-s", type=float, default=2.0)
    p.add_argument("--max-camera-skew-s", type=float, default=0.10)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--world-name", default="grasp_world")
    p.add_argument("--joint-states-topic", default="/joint_states")
    p.add_argument("--command-topic", default="/forward_position_controller/commands")
    p.add_argument("--fixed-image-topic", default="/top_camera/image_raw")
    p.add_argument("--handeye-image-topic", default="/gripper_camera/image_raw")
    p.add_argument("--max-delta-rad", type=float, default=0.08)
    p.add_argument("--max-initial-action-delta", type=float, default=15.0)
    p.add_argument(
        "--max-initial-gripper-delta",
        type=float,
        default=35.0,
        help="Maximum allowed first-action delta for the gripper in LeRobot units.",
    )
    p.add_argument("--move-home", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--home-norm",
        default="-0.30,-99.06,99.08,72.10,-3.28,1.48",
        help="Mean initial pose of the 15-point training dataset in LeRobot normalized units.",
    )
    p.add_argument("--home-duration-s", type=float, default=3.0)
    p.add_argument("--home-settle-timeout-s", type=float, default=5.0)
    p.add_argument("--home-tolerance-rad", type=float, default=0.04)
    p.add_argument("--randomize-cube", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cube-x-range", nargs=2, type=float, default=[0.48, 0.52])
    p.add_argument("--cube-y-range", nargs=2, type=float, default=[-0.04, 0.04])
    p.add_argument("--cube-spawn-z", type=float, default=0.7655)
    p.add_argument("--cube-settle-s", type=float, default=0.4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--display", action="store_true")
    p.add_argument(
        "--match-training-photometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match simulated RGB mean/std to the 150 real training demonstrations.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = GazeboLeRobotRunner(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
