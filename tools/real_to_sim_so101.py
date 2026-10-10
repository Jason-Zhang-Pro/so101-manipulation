#!/usr/bin/env python3
"""Mirror a real SO-101 arm into the Gazebo SO-101 model.

This script only reads the real arm's present position. It publishes mapped
joint targets to the Gazebo forward_position_controller. Use ``--source leader``
to control sim by hand without moving the real follower.
"""

from __future__ import annotations

import argparse
import signal
import time
from typing import Sequence

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Must match the Gazebo URDF/controller joint order.
SIM_LOWER = {
    "shoulder_pan": -1.91986,
    "shoulder_lift": -1.74533,
    "elbow_flex": -1.69,
    "wrist_flex": -1.65806,
    "wrist_roll": -2.74385,
    "gripper": -0.174533,
}
SIM_UPPER = {
    "shoulder_pan": 1.91986,
    "shoulder_lift": 1.74533,
    "elbow_flex": 1.69,
    "wrist_flex": 1.65806,
    "wrist_roll": 2.84121,
    "gripper": 1.74533,
}


def lerobot_norm_to_sim_rad(joint: str, value: float) -> float:
    """Map LeRobot normalized joint values to Gazebo joint radians."""
    low = SIM_LOWER[joint]
    high = SIM_UPPER[joint]
    if joint == "gripper":
        alpha = max(0.0, min(100.0, value)) / 100.0
    else:
        alpha = (max(-100.0, min(100.0, value)) + 100.0) / 200.0
    return low + alpha * (high - low)


def parse_vector(raw: str | None, default: Sequence[float]) -> list[float]:
    if raw is None:
        return list(default)
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} comma-separated values, got {len(vals)}")
    return vals


def make_ros_mirror(topic: str):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray

    class RealToSimMirror(Node):
        def __init__(self, topic: str):
            super().__init__("so101_real_to_sim_mirror")
            self.publisher = self.create_publisher(Float64MultiArray, topic, 10)

        def publish(self, q: Sequence[float]) -> None:
            msg = Float64MultiArray()
            msg.data = [float(x) for x in q]
            self.publisher.publish(msg)

    rclpy.init(args=None)
    return rclpy, RealToSimMirror(topic)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["follower", "leader"], default="follower")
    parser.add_argument("--port", default=None)
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--topic", default="/forward_position_controller/commands")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="Low-pass target smoothing. 1.0 disables smoothing; lower values are smoother but slower.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print mapped values without publishing to sim.")
    parser.add_argument("--no-calibrate", action="store_true", help="Do not prompt/write calibration on connect.")
    parser.add_argument(
        "--signs",
        default=None,
        help="Optional comma-separated per-joint signs, e.g. 1,-1,1,1,1,1.",
    )
    parser.add_argument(
        "--offsets",
        default=None,
        help="Optional comma-separated radian offsets added after mapping.",
    )
    args = parser.parse_args()

    signs = parse_vector(args.signs, [1.0] * len(JOINT_NAMES))
    offsets = parse_vector(args.offsets, [0.0] * len(JOINT_NAMES))
    smoothing_alpha = max(0.0, min(1.0, args.smoothing_alpha))

    default_port = "/dev/ttyACM0" if args.source == "leader" else "/dev/ttyACM1"
    default_id = "my_awesome_leader_arm" if args.source == "leader" else "my_awesome_follower_arm"
    port = args.port or default_port
    device_id = args.robot_id or default_id

    if args.source == "leader":
        device = SO101Leader(
            SO101LeaderConfig(
                id=device_id,
                port=port,
                use_degrees=False,
            )
        )
    else:
        device = SO101Follower(
            SO101FollowerConfig(
                id=device_id,
                port=port,
                cameras={},
                max_relative_target=None,
                use_degrees=False,
            )
        )

    node = None
    should_stop = False

    def handle_sigint(signum, frame):
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        device.connect(calibrate=not args.no_calibrate)
        rclpy_module = None
        if not args.dry_run:
            rclpy_module, node = make_ros_mirror(args.topic)

        period = 1.0 / args.rate
        print(f"Mirroring real SO-101 {args.source} to sim. Press Ctrl+C to stop.")
        print(f"port: {port}, id: {device_id}")
        print(f"smoothing_alpha: {smoothing_alpha}")
        print("joint order:", ", ".join(JOINT_NAMES))
        smoothed_q = None
        next_tick = time.perf_counter()
        while not should_stop:
            raw = device.bus.sync_read("Present_Position")
            sim_q = []
            norm_q = []
            for i, name in enumerate(JOINT_NAMES):
                norm = float(raw[name])
                mapped = lerobot_norm_to_sim_rad(name, norm)
                mapped = signs[i] * mapped + offsets[i]
                mapped = max(SIM_LOWER[name], min(SIM_UPPER[name], mapped))
                norm_q.append(norm)
                sim_q.append(mapped)

            if smoothed_q is None:
                smoothed_q = list(sim_q)
            elif smoothing_alpha >= 1.0:
                smoothed_q = list(sim_q)
            else:
                smoothed_q = [
                    (1.0 - smoothing_alpha) * old + smoothing_alpha * new
                    for old, new in zip(smoothed_q, sim_q)
                ]

            if args.dry_run:
                norm_text = " ".join(f"{name}={val:7.2f}" for name, val in zip(JOINT_NAMES, norm_q))
                sim_text = " ".join(f"{name}={val: .3f}" for name, val in zip(JOINT_NAMES, smoothed_q))
                print(f"real_norm: {norm_text}")
                print(f"sim_rad:   {sim_text}\n")
            else:
                node.publish(smoothed_q)

            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
    finally:
        if node is not None:
            node.destroy_node()
            rclpy_module.shutdown()
        if device.is_connected:
            device.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
