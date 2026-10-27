#!/usr/bin/python3
"""Publish the Gazebo goal-region centre once in the world frame."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from goal_region_pose_core import (
    PoseSample,
    parse_pose_block,
    transform_point,
)


class GoalRegionPosePublisher(Node):
    """Resolve one Gazebo pose and retain it for late ROS subscribers."""

    def __init__(self) -> None:
        super().__init__("goal_region_pose_publisher")

        self.declare_parameter(
            "world_pose_topic",
            "/world/empty/pose/info",
        )
        self.declare_parameter(
            "table_model_name",
            "lab_table",
        )
        self.declare_parameter(
            "goal_link_name",
            "goal_region_plate_link",
        )
        self.declare_parameter(
            "output_topic",
            "/goal_region/center",
        )

        self.world_pose_topic = str(
            self.get_parameter("world_pose_topic").value
        )
        self.table_model_name = str(
            self.get_parameter("table_model_name").value
        )
        self.goal_link_name = str(
            self.get_parameter("goal_link_name").value
        )
        self.output_topic = str(
            self.get_parameter("output_topic").value
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._publisher = self.create_publisher(
            PointStamped,
            self.output_topic,
            qos,
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._table_pose: Optional[PoseSample] = None
        self._goal_pose: Optional[PoseSample] = None
        self._pending_point: Optional[
            Tuple[float, float, float]
        ] = None
        self._published = False

        self._pose_process: Optional[subprocess.Popen] = None
        self._pose_thread: Optional[threading.Thread] = None

        self._publish_timer = self.create_timer(
            0.05,
            self._publish_if_ready,
        )

        self._start_pose_reader()

        self.get_logger().info(
            "Waiting once for Gazebo goal-region pose: "
            f"table={self.table_model_name!r}, "
            f"link={self.goal_link_name!r}, "
            f"source={self.world_pose_topic!r}"
        )

    def _start_pose_reader(self) -> None:
        command = [
            "ign",
            "topic",
            "-e",
            "-t",
            self.world_pose_topic,
        ]

        try:
            self._pose_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
        except OSError as error:
            raise RuntimeError(
                f"Failed to start Gazebo pose reader: {error}"
            ) from error

        self._pose_thread = threading.Thread(
            target=self._pose_reader_loop,
            name="goal_region_pose_reader",
            daemon=True,
        )
        self._pose_thread.start()

    def _pose_reader_loop(self) -> None:
        process = self._pose_process

        if process is None or process.stdout is None:
            return

        block_lines = []
        brace_depth = 0
        collecting_pose = False

        while not self._stop_event.is_set():
            line = process.stdout.readline()

            if line == "":
                if process.poll() is not None:
                    break

                time.sleep(0.01)
                continue

            stripped = line.strip()

            if not collecting_pose:
                if stripped == "pose {":
                    collecting_pose = True
                    block_lines = [line]
                    brace_depth = 1
                continue

            block_lines.append(line)
            brace_depth += line.count("{")
            brace_depth -= line.count("}")

            if brace_depth == 0:
                self._process_pose_block(
                    "".join(block_lines)
                )
                collecting_pose = False
                block_lines = []

        if (
            not self._stop_event.is_set()
            and not self._published
        ):
            return_code = process.poll()
            self.get_logger().error(
                "Gazebo pose reader stopped before the "
                "goal-region centre was resolved; "
                f"return code={return_code}"
            )

    def _process_pose_block(self, block: str) -> None:
        try:
            sample = parse_pose_block(block)
        except ValueError:
            return

        with self._lock:
            if sample.name == self.table_model_name:
                self._table_pose = sample
            elif sample.name == self.goal_link_name:
                self._goal_pose = sample
            else:
                return

            if (
                self._pending_point is not None
                or self._published
                or self._table_pose is None
                or self._goal_pose is None
            ):
                return

            self._pending_point = transform_point(
                self._table_pose,
                self._goal_pose,
            )

    def _publish_if_ready(self) -> None:
        with self._lock:
            point = self._pending_point

            if point is None or self._published:
                return

            self._published = True
            self._pending_point = None

        message = PointStamped()
        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.header.frame_id = "world"
        message.point.x = point[0]
        message.point.y = point[1]
        message.point.z = point[2]

        self._publisher.publish(message)

        self.get_logger().info(
            "Goal-region centre cached from Gazebo and "
            "published once: "
            f"frame=world, "
            f"x={point[0]:.6f}, "
            f"y={point[1]:.6f}, "
            f"z={point[2]:.6f}"
        )

        # The ROS node remains alive so TRANSIENT_LOCAL keeps the
        # one static message available to subscribers that start later.
        self._stop_event.set()
        self._stop_pose_reader()

    def _stop_pose_reader(self) -> None:
        process = self._pose_process

        if process is not None and process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

        thread = self._pose_thread

        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=1.0)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._stop_pose_reader()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalRegionPosePublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
