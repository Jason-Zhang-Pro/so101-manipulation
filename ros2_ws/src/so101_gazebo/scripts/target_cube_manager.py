#!/usr/bin/env python3

import json
import math
import os
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class CubeState:
    name: str
    entity_id: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    last_seen: float
    source_sim_time_s: Optional[float] = None


class TargetCubeManager(Node):
    def __init__(self) -> None:
        super().__init__("target_cube_manager")

        self.declare_parameter("world_name", "empty")
        self.declare_parameter(
            "dynamic_pose_topic",
            "/world/empty/dynamic_pose/info",
        )

        self.declare_parameter("model_name_prefix", "target_cube_")
        self.declare_parameter("name_digits", 4)
        self.declare_parameter("model_sdf_path", "")

        self.declare_parameter("cube_size", 0.030)
        self.declare_parameter("table_surface_z", 0.750)
        self.declare_parameter("spawn_clearance", 0.003)

        self.declare_parameter("spawn_center_x", 0.0)
        self.declare_parameter("spawn_center_y", 0.0)
        self.declare_parameter("spawn_inner_radius", 0.10)
        self.declare_parameter("spawn_outer_radius", 0.40)
        self.declare_parameter("spawn_angle_min", -math.pi / 2.0)
        self.declare_parameter("spawn_angle_max", math.pi / 2.0)

        self.declare_parameter("minimum_center_distance", 0.050)
        self.declare_parameter("maximum_spawn_attempts", 100)
        self.declare_parameter("maximum_cube_count", 20)
        self.declare_parameter("state_publish_rate", 30.0)

        self.world_name = str(
            self.get_parameter("world_name").value
        )
        self.dynamic_pose_topic = str(
            self.get_parameter("dynamic_pose_topic").value
        )

        self.model_name_prefix = str(
            self.get_parameter("model_name_prefix").value
        )
        self.name_digits = int(
            self.get_parameter("name_digits").value
        )

        configured_model_path = str(
            self.get_parameter("model_sdf_path").value
        )

        if configured_model_path:
            self.model_sdf_path = Path(configured_model_path)
        else:
            package_share = Path(
                get_package_share_directory("so101_gazebo")
            )
            self.model_sdf_path = (
                package_share
                / "models"
                / "target_cube"
                / "model.sdf"
            )

        self.cube_size = float(
            self.get_parameter("cube_size").value
        )
        self.table_surface_z = float(
            self.get_parameter("table_surface_z").value
        )
        self.spawn_clearance = float(
            self.get_parameter("spawn_clearance").value
        )

        self.spawn_center_x = float(
            self.get_parameter("spawn_center_x").value
        )
        self.spawn_center_y = float(
            self.get_parameter("spawn_center_y").value
        )
        self.spawn_inner_radius = float(
            self.get_parameter("spawn_inner_radius").value
        )
        self.spawn_outer_radius = float(
            self.get_parameter("spawn_outer_radius").value
        )
        self.spawn_angle_min = float(
            self.get_parameter("spawn_angle_min").value
        )
        self.spawn_angle_max = float(
            self.get_parameter("spawn_angle_max").value
        )

        self.minimum_center_distance = float(
            self.get_parameter("minimum_center_distance").value
        )
        self.maximum_spawn_attempts = int(
            self.get_parameter("maximum_spawn_attempts").value
        )
        self.maximum_cube_count = int(
            self.get_parameter("maximum_cube_count").value
        )
        self.state_publish_rate = float(
            self.get_parameter("state_publish_rate").value
        )

        self.spawn_z = (
            self.table_surface_z
            + self.cube_size / 2.0
            + self.spawn_clearance
        )

        self._validate_parameters()

        # Use the operating system random source instead of the
        # deterministic Mersenne Twister generator.
        self._random = random.SystemRandom()

        self._states: Dict[str, CubeState] = {}
        self._deleted_until: Dict[str, float] = {}
        self._deleted_pose_ignore_duration = 2.0
        self._pose_publishers = {}
        self._state_lock = threading.Lock()
        self._state_sequence = 0

        self._pose_process: Optional[subprocess.Popen] = None
        self._pose_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._spawn_service = self.create_service(
            Trigger,
            "/target_cube/spawn_random",
            self._spawn_random_callback,
        )

        self._delete_all_service = self.create_service(
            Trigger,
            "/target_cube/delete_all",
            self._delete_all_callback,
        )

        self._delete_name_subscription = self.create_subscription(
            String,
            "/target_cube/delete_name",
            self._delete_name_callback,
            10,
        )

        self._states_publisher = self.create_publisher(
            String,
            "/target_cube/states",
            10,
        )

        self._delete_result_publisher = self.create_publisher(
            String,
            "/target_cube/delete_result",
            10,
        )

        timer_period = 1.0 / max(self.state_publish_rate, 0.1)
        self._publish_timer = self.create_timer(
            timer_period,
            self._publish_states,
        )

        self._start_pose_reader()

        self.get_logger().info(
            "Target cube manager started. "
            f"Spawn region: radius "
            f"{self.spawn_inner_radius:.3f}-"
            f"{self.spawn_outer_radius:.3f} m, "
            f"angle {math.degrees(self.spawn_angle_min):.1f}-"
            f"{math.degrees(self.spawn_angle_max):.1f} deg, "
            f"spawn Z={self.spawn_z:.3f} m"
        )

    def _validate_parameters(self) -> None:
        if not self.model_name_prefix:
            raise ValueError("model_name_prefix cannot be empty")

        if self.name_digits < 1:
            raise ValueError("name_digits must be at least 1")

        if self.cube_size <= 0.0:
            raise ValueError("cube_size must be positive")

        if not (
            0.0
            < self.spawn_inner_radius
            < self.spawn_outer_radius
        ):
            raise ValueError(
                "Require 0 < spawn_inner_radius "
                "< spawn_outer_radius"
            )

        if self.spawn_angle_min >= self.spawn_angle_max:
            raise ValueError(
                "spawn_angle_min must be less than "
                "spawn_angle_max"
            )

        if self.minimum_center_distance < self.cube_size:
            raise ValueError(
                "minimum_center_distance must be at least "
                "cube_size"
            )

        if self.maximum_spawn_attempts < 1:
            raise ValueError(
                "maximum_spawn_attempts must be positive"
            )

        if self.maximum_cube_count < 1:
            raise ValueError(
                "maximum_cube_count must be positive"
            )

    def _start_pose_reader(self) -> None:
        command = [
            "ign",
            "topic",
            "-e",
            "-t",
            self.dynamic_pose_topic,
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
                f"Failed to start ign topic: {error}"
            ) from error

        self._pose_thread = threading.Thread(
            target=self._pose_reader_loop,
            daemon=True,
        )
        self._pose_thread.start()

    def _pose_reader_loop(self) -> None:
        process = self._pose_process

        if process is None or process.stdout is None:
            return

        block_lines = []
        brace_depth = 0
        block_kind = None
        source_sim_time_s = None

        while not self._stop_event.is_set():
            line = process.stdout.readline()

            if line == "":
                if process.poll() is not None:
                    break

                time.sleep(0.01)
                continue

            stripped = line.strip()

            if block_kind is None:
                if stripped in ("header {", "pose {"):
                    block_kind = stripped.split()[0]
                    block_lines = [line]
                    brace_depth = 1
                continue

            block_lines.append(line)
            brace_depth += line.count("{")
            brace_depth -= line.count("}")

            if brace_depth == 0:
                block = "".join(block_lines)
                if block_kind == "header":
                    source_sim_time_s = self._pose_header_timestamp_s(block)
                else:
                    self._process_pose_block(
                        block, source_sim_time_s=source_sim_time_s
                    )
                block_kind = None
                block_lines = []

        if not self._stop_event.is_set():
            return_code = process.poll()
            self.get_logger().error(
                "Gazebo dynamic pose reader stopped "
                f"unexpectedly with return code {return_code}"
            )

    @staticmethod
    def _pose_header_timestamp_s(block: str) -> Optional[float]:
        """Read the Pose_V simulation stamp, never a callback/ROS wall time."""
        stamp = TargetCubeManager._extract_nested_block(block, "stamp")
        if stamp is None:
            return None
        seconds = re.search(r"\bsec:\s*(-?\d+)", stamp)
        nanoseconds = re.search(r"\bnsec:\s*(-?\d+)", stamp)
        sec = 0 if seconds is None else int(seconds.group(1))
        nsec = 0 if nanoseconds is None else int(nanoseconds.group(1))
        if sec < 0 or not 0 <= nsec < 1_000_000_000:
            return None
        return sec + nsec * 1e-9

    def _process_pose_block(
        self, block: str, *, source_sim_time_s: Optional[float] = None
    ) -> None:
        name_match = re.search(
            r'\bname:\s*"([^"]+)"',
            block,
        )
        id_match = re.search(
            r"\bid:\s*(\d+)",
            block,
        )

        if name_match is None or id_match is None:
            return

        name = name_match.group(1)

        if not name.startswith(self.model_name_prefix):
            return

        now = time.monotonic()

        # Ignore pose messages that were already queued before deletion.
        with self._state_lock:
            deleted_until = self._deleted_until.get(name)

            if deleted_until is not None:
                if now < deleted_until:
                    return

                self._deleted_until.pop(name, None)

        position_block = self._extract_nested_block(
            block,
            "position",
        )
        orientation_block = self._extract_nested_block(
            block,
            "orientation",
        )

        if position_block is None or orientation_block is None:
            return

        try:
            state = CubeState(
                name=name,
                entity_id=int(id_match.group(1)),
                x=self._extract_float(position_block, "x"),
                y=self._extract_float(position_block, "y"),
                z=self._extract_float(position_block, "z"),
                qx=self._extract_float(
                    orientation_block,
                    "x",
                    default=0.0,
                ),
                qy=self._extract_float(
                    orientation_block,
                    "y",
                    default=0.0,
                ),
                qz=self._extract_float(
                    orientation_block,
                    "z",
                    default=0.0,
                ),
                qw=self._extract_float(
                    orientation_block,
                    "w",
                    default=1.0,
                ),
                last_seen=time.monotonic(),
                source_sim_time_s=source_sim_time_s,
            )
        except ValueError:
            return

        with self._state_lock:
            self._states[name] = state

    @staticmethod
    def _extract_nested_block(
        text: str,
        field_name: str,
    ) -> Optional[str]:
        match = re.search(
            rf"\b{re.escape(field_name)}\s*\{{(.*?)\}}",
            text,
            re.DOTALL,
        )

        if match is None:
            return None

        return match.group(1)

    @staticmethod
    def _extract_float(
        text: str,
        field_name: str,
        default: Optional[float] = None,
    ) -> float:
        match = re.search(
            rf"\b{re.escape(field_name)}:\s*"
            rf"({FLOAT_PATTERN})",
            text,
        )

        if match is None:
            if default is not None:
                return default

            raise ValueError(
                f"Missing numeric field {field_name}"
            )

        return float(match.group(1))

    def _sample_position(self) -> Optional[Tuple[float, float]]:
        with self._state_lock:
            occupied_positions = [
                (state.x, state.y)
                for state in self._states.values()
            ]

        for _ in range(self.maximum_spawn_attempts):
            theta = self._random.uniform(
                self.spawn_angle_min,
                self.spawn_angle_max,
            )

            radius_squared = self._random.uniform(
                self.spawn_inner_radius ** 2,
                self.spawn_outer_radius ** 2,
            )
            radius = math.sqrt(radius_squared)

            x = (
                self.spawn_center_x
                + radius * math.cos(theta)
            )
            y = (
                self.spawn_center_y
                + radius * math.sin(theta)
            )

            separated = all(
                math.hypot(x - old_x, y - old_y)
                >= self.minimum_center_distance
                for old_x, old_y in occupied_positions
            )

            if separated:
                return x, y

        return None

    def _next_available_name(self) -> str:
        now = time.monotonic()

        with self._state_lock:
            expired_names = [
                name
                for name, deleted_until
                in self._deleted_until.items()
                if deleted_until <= now
            ]

            for name in expired_names:
                self._deleted_until.pop(name, None)

            existing_names = (
                set(self._states)
                | set(self._deleted_until)
            )

        index = 1

        while True:
            name = (
                f"{self.model_name_prefix}"
                f"{index:0{self.name_digits}d}"
            )

            if name not in existing_names:
                return name

            index += 1

    def _spawn_random_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        with self._state_lock:
            current_count = len(self._states)

        if current_count >= self.maximum_cube_count:
            response.success = False
            response.message = (
                "Maximum managed cube count reached"
            )
            return response

        if not self.model_sdf_path.is_file():
            response.success = False
            response.message = (
                f"Cube model not found: "
                f"{self.model_sdf_path}"
            )
            return response

        sampled_position = self._sample_position()

        if sampled_position is None:
            response.success = False
            response.message = (
                "Could not find a collision-free random "
                "spawn position"
            )
            return response

        x, y = sampled_position
        name = self._next_available_name()

        # Sample an independent horizontal orientation for every spawn.
        # SystemRandom.random() is backed by the operating-system random source.
        # Mapping [0, 1) -> [-pi, pi) gives a uniform full-circle yaw.
        spawn_yaw = (
            2.0 * math.pi * self._random.random()
        ) - math.pi
        spawn_half_yaw = 0.5 * spawn_yaw
        spawn_qz = math.sin(spawn_half_yaw)
        spawn_qw = math.cos(spawn_half_yaw)

        request_text = (
            f'sdf_filename: "{self.model_sdf_path}" '
            f'name: "{name}" '
            "allow_renaming: false "
            "pose: { "
            "position: { "
            f"x: {x:.9f} "
            f"y: {y:.9f} "
            f"z: {self.spawn_z:.9f} "
            "} "
            f"orientation: {{ x: 0.0 y: 0.0 "
            f"z: {spawn_qz:.12f} w: {spawn_qw:.12f} }} "
            "}"
        )

        success, command_output = self._call_ign_service(
            service=f"/world/{self.world_name}/create",
            request_type="ignition.msgs.EntityFactory",
            request_text=request_text,
        )

        response.success = success

        if not success:
            response.message = (
                f"Failed to spawn {name}: {command_output}"
            )
            return response

        entity_id = self._wait_for_entity(name, timeout=2.0)

        result = {
            "name": name,
            "entity_id": entity_id,
            "requested_position": {
                "x": x,
                "y": y,
                "z": self.spawn_z,
            },
        }

        response.message = json.dumps(result)
        return response

    def _delete_name_callback(self, message: String) -> None:
        name = message.data.strip()

        if not name:
            self._publish_delete_result(
                success=False,
                name="",
                message="Cube name is empty",
            )
            return

        if name.lower() == "all":
            deleted, failed = self._delete_all_cubes()
            self._publish_delete_result(
                success=not failed,
                name="all",
                message=(
                    f"Deleted {deleted} cube(s); "
                    f"failed {failed}"
                ),
            )
            return

        with self._state_lock:
            state = self._states.get(name)

        if state is None:
            self._publish_delete_result(
                success=False,
                name=name,
                message="Cube is not present in the registry",
            )
            return

        success, command_output = self._remove_entity(
            state.entity_id
        )

        if success:
            self._remove_local_state(name)

        self._publish_delete_result(
            success=success,
            name=name,
            message=command_output,
        )

    def _delete_all_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        deleted, failed = self._delete_all_cubes()

        response.success = failed == 0
        response.message = (
            f"Deleted {deleted} cube(s); failed {failed}"
        )
        return response

    def _delete_all_cubes(self) -> Tuple[int, int]:
        with self._state_lock:
            states = list(self._states.values())

        deleted = 0
        failed = 0

        for state in states:
            success, _ = self._remove_entity(
                state.entity_id
            )

            if success:
                deleted += 1
                self._remove_local_state(state.name)
            else:
                failed += 1

        return deleted, failed

    def _remove_entity(
        self,
        entity_id: int,
    ) -> Tuple[bool, str]:
        return self._call_ign_service(
            service=f"/world/{self.world_name}/remove",
            request_type="ignition.msgs.Entity",
            request_text=f"id: {entity_id}",
        )

    def _call_ign_service(
        self,
        service: str,
        request_type: str,
        request_text: str,
    ) -> Tuple[bool, str]:
        command = [
            "ign",
            "service",
            "-s",
            service,
            "--reqtype",
            request_type,
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            request_text,
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            return False, "ign command was not found"
        except subprocess.TimeoutExpired:
            return False, "ign service command timed out"

        output = (
            completed.stdout.strip()
            or completed.stderr.strip()
        )

        success = (
            completed.returncode == 0
            and re.search(
                r"\bdata:\s*true\b",
                output,
            )
            is not None
        )

        return success, output

    def _wait_for_entity(
        self,
        name: str,
        timeout: float,
    ) -> Optional[int]:
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            with self._state_lock:
                state = self._states.get(name)

            if state is not None:
                return state.entity_id

            time.sleep(0.02)

        return None

    def _remove_local_state(self, name: str) -> None:
        with self._state_lock:
            self._states.pop(name, None)
            self._deleted_until[name] = (
                time.monotonic()
                + self._deleted_pose_ignore_duration
            )

        publisher = self._pose_publishers.pop(
            name,
            None,
        )

        if publisher is not None:
            self.destroy_publisher(publisher)

    def _publish_delete_result(
        self,
        success: bool,
        name: str,
        message: str,
    ) -> None:
        result_message = String()
        result_message.data = json.dumps(
            {
                "success": success,
                "name": name,
                "message": message,
            }
        )
        self._delete_result_publisher.publish(result_message)

    def _publish_states(self) -> None:
        with self._state_lock:
            states = list(self._states.values())

        if self._pose_process is not None and self._pose_process.poll() is not None:
            # Do not keep refreshing subscribers after the pose reader dies.
            return
        if any(state.source_sim_time_s is None for state in states):
            # Missing source stamps must fail closed, not masquerade as now.
            return

        self._state_sequence += 1
        now = self.get_clock().now()
        # A timer publication is not a new physical observation. Keep the
        # actual Gazebo source stamp even if the reader stalls. Empty registry
        # messages still follow the simulation clock for reset/spawn barriers.
        sim_time_s = (
            min(state.source_sim_time_s for state in states)
            if states
            else now.nanoseconds * 1e-9
        )
        serialized_states = []

        for state in states:
            publisher = self._pose_publishers.get(
                state.name
            )

            if publisher is None:
                publisher = self.create_publisher(
                    PoseStamped,
                    f"/target_cube/pose/{state.name}",
                    10,
                )
                self._pose_publishers[state.name] = publisher

            pose_message = PoseStamped()
            pose_message.header.stamp = now.to_msg()
            source_ns = round(state.source_sim_time_s * 1_000_000_000)
            source_sec, source_nsec = divmod(source_ns, 1_000_000_000)
            pose_message.header.stamp.sec = source_sec
            pose_message.header.stamp.nanosec = source_nsec
            pose_message.header.frame_id = "world"

            pose_message.pose.position.x = state.x
            pose_message.pose.position.y = state.y
            pose_message.pose.position.z = state.z

            pose_message.pose.orientation.x = state.qx
            pose_message.pose.orientation.y = state.qy
            pose_message.pose.orientation.z = state.qz
            pose_message.pose.orientation.w = state.qw

            publisher.publish(pose_message)

            serialized_states.append(
                {
                    "name": state.name,
                    "entity_id": state.entity_id,
                    "position": {
                        "x": state.x,
                        "y": state.y,
                        "z": state.z,
                    },
                    "orientation": {
                        "x": state.qx,
                        "y": state.qy,
                        "z": state.qz,
                        "w": state.qw,
                    },
                }
            )

        states_message = String()
        states_message.data = json.dumps(
            {
                "frame_id": "world",
                "sequence": self._state_sequence,
                "sim_time_s": sim_time_s,
                "cube_count": len(serialized_states),
                "cubes": serialized_states,
            }
        )
        self._states_publisher.publish(states_message)

    def destroy_node(self) -> bool:
        self._stop_event.set()

        if (
            self._pose_process is not None
            and self._pose_process.poll() is None
        ):
            self._pose_process.terminate()

            try:
                self._pose_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._pose_process.kill()

        if self._pose_thread is not None:
            self._pose_thread.join(timeout=2.0)

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetCubeManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
