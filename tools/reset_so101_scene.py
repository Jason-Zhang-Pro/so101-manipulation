#!/usr/bin/env python3
"""Reset the simulated SO-101 and place the cube on a 3x5 grid center."""

from __future__ import annotations

import argparse
import random
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Mean initial state of the 150-episode ACT training dataset, converted to the
# current Gazebo joint convention. This is the relevant reset for policy tests.
HOME_POSITION = np.array([0.0126, -1.7289, 1.5858, 1.2156, -0.0429, -0.1461], dtype=np.float64)

# Cell 0-4 are the top row from left to right, 5-9 are the middle row,
# and 10-14 are the bottom row from left to right.
GRID_CENTERS = [
    (x, y, 0.7655)
    for x in (0.592, 0.512, 0.432)
    for y in (-0.034, -0.094, -0.154, -0.214, -0.274)
]


class SceneReset(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_scene_reset")
        self.args = args
        self.joint_positions: dict[str, float] = {}
        self.command_pub = self.create_publisher(Float64MultiArray, args.command_topic, 10)
        self.create_subscription(JointState, args.joint_states_topic, self._on_joint_states, 10)

    def _on_joint_states(self, msg: JointState) -> None:
        self.joint_positions = dict(zip(msg.name, msg.position))

    def wait_for_joint_states(self) -> np.ndarray:
        deadline = time.monotonic() + self.args.wait_timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(name in self.joint_positions for name in JOINT_NAMES):
                return np.array([self.joint_positions[name] for name in JOINT_NAMES], dtype=np.float64)
        raise RuntimeError(
            f"Timed out waiting for {self.args.joint_states_topic}. "
            "Start Gazebo and its controllers first."
        )

    def publish_target(self, target: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = target.astype(float).tolist()
        self.command_pub.publish(msg)

    def move_home(self, start: np.ndarray) -> None:
        steps = max(2, round(self.args.home_duration * self.args.rate))
        self.get_logger().info(f"moving to folded home pose over {self.args.home_duration:.2f}s")
        for step in range(1, steps + 1):
            alpha = step / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self.publish_target(start + smooth * (HOME_POSITION - start))
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / self.args.rate)

        for _ in range(10):
            self.publish_target(HOME_POSITION)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / self.args.rate)

        deadline = time.monotonic() + self.args.settle_timeout
        while time.monotonic() < deadline:
            self.publish_target(HOME_POSITION)
            rclpy.spin_once(self, timeout_sec=0.02)
            actual = np.array([self.joint_positions[name] for name in JOINT_NAMES], dtype=np.float64)
            error = np.abs(actual - HOME_POSITION)
            if float(error.max()) <= self.args.home_tolerance:
                self.get_logger().info(f"home reached; max error={float(error.max()):.4f} rad")
                return
            time.sleep(1.0 / self.args.rate)
        raise RuntimeError(
            "Failed to reach ACT training home pose: "
            f"target={np.round(HOME_POSITION, 4).tolist()} "
            f"actual={np.round(actual, 4).tolist()} "
            f"error={np.round(error, 4).tolist()}"
        )

    def place_cube(self, cell_index: int) -> None:
        x, y, z = GRID_CENTERS[cell_index]
        request = (
            'name: "cube" '
            f"position {{ x: {x} y: {y} z: {z} }} "
            "orientation { w: 1.0 }"
        )
        command = [
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
            request,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3.0, check=False)
        if result.returncode != 0 or "data: true" not in result.stdout:
            raise RuntimeError(
                f"Failed to place cube: stdout={result.stdout.strip()!r} "
                f"stderr={result.stderr.strip()!r}"
            )
        self.get_logger().info(f"cube placed at cell {cell_index}: x={x:.3f}, y={y:.3f}, z={z:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, choices=range(15), help="Cell 0-14; random when omitted")
    parser.add_argument("--world-name", default="grasp_world")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--command-topic", default="/forward_position_controller/commands")
    parser.add_argument("--home-duration", type=float, default=2.5)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    parser.add_argument("--settle-timeout", type=float, default=5.0)
    parser.add_argument("--home-tolerance", type=float, default=0.04)
    parser.add_argument("--no-move-home", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = SceneReset(args)
    try:
        start = node.wait_for_joint_states()
        if not args.no_move_home:
            node.move_home(start)
        cell_index = args.cell if args.cell is not None else random.randrange(len(GRID_CENTERS))
        node.place_cube(cell_index)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
