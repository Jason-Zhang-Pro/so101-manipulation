"""ROS 2 I/O node for the SO-101 Gazebo reaching environment.

The node receives measured robot, cube, and camera states and stores them in
the environment's thread-safe caches. Controller commands are published only
when ``publish_joint_targets()`` is explicitly called.

This module intentionally does not use cv_bridge.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, String

from .image_cache import RgbImageCache
from .gazebo_grasp_attachment import (
    GRASP_ATTACHMENT_STATE_TOPIC, GraspAttachmentStateCache, GazeboGraspAttachmentError,
)
from .contact_state import (
    FingerContactCache, FIXED_FINGER_CONTACT_TOPIC, MOVING_FINGER_CONTACT_TOPIC,
)
from .ros_image_conversion import ros_stamp_to_seconds
from .policy_action import (
    POLICY_JOINT_ACTION_SIZE,
    POLICY_JOINT_NAMES,
)
from .ros_message_adapter import (
    update_joint_cache_from_message,
    update_rgb_image_cache_from_message,
)
from .simulation_clock import SimulationClockCache
from .state_cache import (
    GoalRegionCache,
    JointStateCache,
    TargetCubeCollectionCache,
    TargetCubeStateCache,
)
from .target_cube_parser import (
    ParsedTargetCube,
    ParsedTargetCubeStates,
    parse_target_cube_states_json,
    select_target_cube,
)


CLOCK_TOPIC = "/clock"
JOINT_STATE_TOPIC = "/joint_states"
TARGET_CUBE_STATES_TOPIC = "/target_cube/states"
GOAL_REGION_CENTER_TOPIC = "/goal_region/center"
TOP_CAMERA_TOPIC = "/top_camera/image_raw"
GRIPPER_CAMERA_TOPIC = "/gripper_camera/image_raw"
CONTROLLER_COMMAND_TOPIC = "/position_controller/commands"


def controller_command_message(
    joint_targets: Sequence[float],
) -> Float64MultiArray:
    """Create a six-joint position-controller command message."""

    command = np.asarray(
        joint_targets,
        dtype=float,
    )

    if command.shape != (POLICY_JOINT_ACTION_SIZE,):
        raise ValueError(
            "Controller command must have shape "
            f"({POLICY_JOINT_ACTION_SIZE},)."
        )

    if not np.all(np.isfinite(command)):
        raise ValueError(
            "Controller command must contain only finite values."
        )

    message = Float64MultiArray()
    message.data = command.tolist()
    return message


def _select_runtime_target_cube(
    states: ParsedTargetCubeStates,
    *,
    configured_target_name: str | None,
    current_target_name: str | None,
) -> ParsedTargetCube:
    """Select one stable reward target from a non-empty cube collection.

    An explicitly configured target remains strict. Without one, keep the
    currently selected target while it exists; otherwise choose the
    lexicographically smallest available name for deterministic behavior.
    """

    if configured_target_name is not None:
        return select_target_cube(
            states,
            target_name=configured_target_name,
        )

    available_names = {
        cube.name
        for cube in states.cubes
    }

    if not available_names:
        raise ValueError(
            "At least one target cube is required for runtime selection."
        )

    selected_name = (
        current_target_name
        if current_target_name in available_names
        else min(available_names)
    )

    return select_target_cube(
        states,
        target_name=selected_name,
    )


class So101ReachRosNode(Node):
    """Collect ROS feedback and explicitly publish joint targets."""

    def __init__(self, *, enable_task_contacts: bool = False,
                 enable_grasp_attachment: bool = False) -> None:
        super().__init__("so101_reach_environment")

        self.declare_parameter("target_cube_name", "")
        self.declare_parameter("top_image_height", 600)
        self.declare_parameter("top_image_width", 800)
        self.declare_parameter("gripper_image_height", 480)
        self.declare_parameter("gripper_image_width", 640)
        self.declare_parameter("image_history_size", 8)

        configured_target_name = str(
            self.get_parameter("target_cube_name").value
        ).strip()
        self._target_cube_name = (
            configured_target_name
            if configured_target_name
            else None
        )

        top_height = int(
            self.get_parameter("top_image_height").value
        )
        top_width = int(
            self.get_parameter("top_image_width").value
        )
        gripper_height = int(
            self.get_parameter("gripper_image_height").value
        )
        gripper_width = int(
            self.get_parameter("gripper_image_width").value
        )
        image_history_size = int(
            self.get_parameter("image_history_size").value
        )

        self.simulation_clock_cache = (
            SimulationClockCache()
        )
        self.joint_state_cache = JointStateCache()
        self.contact_cache = FingerContactCache() if enable_task_contacts else None
        self.grasp_attachment_state_cache = GraspAttachmentStateCache() if enable_grasp_attachment else None
        self._grasp_attachment_subscription = None
        if enable_grasp_attachment:
            self._grasp_attachment_subscription = self.create_subscription(
                String, GRASP_ATTACHMENT_STATE_TOPIC, self._grasp_attachment_callback, 100)
        self._contact_subscriptions = []
        if enable_task_contacts:
            from ros_gz_interfaces.msg import Contacts
            for side, topic in (("left", FIXED_FINGER_CONTACT_TOPIC),
                                ("right", MOVING_FINGER_CONTACT_TOPIC)):
                self._contact_subscriptions.append(self.create_subscription(
                    Contacts, topic,
                    lambda message, side=side: self._contact_callback(side, message), 100,
                ))
        self.target_cube_collection_cache = (
            TargetCubeCollectionCache()
        )
        self.target_cube_cache = TargetCubeStateCache()
        self.goal_region_cache = GoalRegionCache()
        self.top_image_cache = RgbImageCache(
            max_history=image_history_size,
            expected_shape=(
                top_height,
                top_width,
                3,
            ),
        )
        self.gripper_image_cache = RgbImageCache(
            max_history=image_history_size,
            expected_shape=(
                gripper_height,
                gripper_width,
                3,
            ),
        )

        self._clock_subscription = self.create_subscription(
            Clock,
            CLOCK_TOPIC,
            self._clock_callback,
            100,
        )
        self._joint_state_subscription = self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._joint_state_callback,
            10,
        )
        self._target_cube_subscription = self.create_subscription(
            String,
            TARGET_CUBE_STATES_TOPIC,
            self._target_cube_states_callback,
            10,
        )
        goal_region_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._goal_region_subscription = (
            self.create_subscription(
                PointStamped,
                GOAL_REGION_CENTER_TOPIC,
                self._goal_region_center_callback,
                goal_region_qos,
            )
        )
        self._top_camera_subscription = self.create_subscription(
            Image,
            TOP_CAMERA_TOPIC,
            self._top_camera_callback,
            10,
        )
        self._gripper_camera_subscription = self.create_subscription(
            Image,
            GRIPPER_CAMERA_TOPIC,
            self._gripper_camera_callback,
            10,
        )

        self._command_publisher = self.create_publisher(
            Float64MultiArray,
            CONTROLLER_COMMAND_TOPIC,
            10,
        )

        self.get_logger().info(
            "SO-101 reach ROS I/O initialized. "
            f"Controller order: {', '.join(POLICY_JOINT_NAMES)}"
        )

    @property
    def target_cube_name(self) -> str | None:
        return self._target_cube_name

    def set_target_cube_name(
        self,
        name: str | None,
    ) -> None:
        """Set the cube selected from `/target_cube/states`."""

        if name is None:
            self._target_cube_name = None
            return

        normalized_name = str(name).strip()

        if not normalized_name:
            raise ValueError(
                "Target cube name must not be empty."
            )

        self._target_cube_name = normalized_name

    def joint_target_subscription_count(self) -> int:
        """Return matched command subscribers; feedback alone is not readiness."""

        return self._command_publisher.get_subscription_count()

    def publish_joint_targets(
        self,
        joint_targets: Sequence[float],
    ) -> None:
        """Publish six absolute joint-position targets."""

        self._command_publisher.publish(
            controller_command_message(joint_targets)
        )

    def _clock_callback(
        self,
        message: Clock,
    ) -> None:
        try:
            self.simulation_clock_cache.update(
                ros_stamp_to_seconds(message.clock)
            )
        except ValueError as error:
            self.get_logger().error(
                f"Rejected /clock message: {error}"
            )

    def _joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        try:
            update_joint_cache_from_message(
                self.joint_state_cache,
                message,
            )
        except ValueError as error:
            self.get_logger().error(
                f"Rejected /joint_states message: {error}"
            )

    def _grasp_attachment_callback(self, message: String) -> None:
        try:
            self.grasp_attachment_state_cache.update(message.data)
        except GazeboGraspAttachmentError as error:
            # Cache retains the error, so the main collector cannot reuse an
            # earlier attached message after a corrupt/backwards status.
            self.get_logger().error(f"Rejected grasp attachment state: {error}")

    def _contact_callback(self, side: str, message) -> None:
        try:
            self.contact_cache.update_from_message(side, message)
        except ValueError as error:
            self.get_logger().warning(f"Rejected {side} finger contacts: {error}")

    def _target_cube_states_callback(
        self,
        message: String,
    ) -> None:
        try:
            parsed = parse_target_cube_states_json(
                message.data
            )

            self.target_cube_collection_cache.update(
                cube_names=tuple(
                    cube.name
                    for cube in parsed.cubes
                ),
                source_sequence=parsed.source_sequence,
                source_sim_time_s=(
                    parsed.source_sim_time_s
                ),
            )

            # Empty state is expected during reset or before spawning.
            if parsed.cube_count == 0:
                return

            current_target = self.target_cube_cache.latest()
            selected = _select_runtime_target_cube(
                parsed,
                configured_target_name=self._target_cube_name,
                current_target_name=(
                    None
                    if current_target is None
                    else current_target.name
                ),
            )

            self.target_cube_cache.update(
                name=selected.name,
                position=selected.position,
                quaternion_xyzw=(
                    selected.quaternion_xyzw
                ),
                source_sequence=parsed.source_sequence,
                source_sim_time_s=parsed.source_sim_time_s,
            )
        except ValueError as error:
            self.get_logger().warning(
                "Rejected /target_cube/states message: "
                f"{error}"
            )

    def _goal_region_center_callback(
        self,
        message: PointStamped,
    ) -> None:
        try:
            self.goal_region_cache.update_once(
                frame_id=message.header.frame_id,
                x=message.point.x,
                y=message.point.y,
                z=message.point.z,
            )
        except ValueError as error:
            self.get_logger().warning(
                "Rejected /goal_region/center message: "
                f"{error}"
            )

    def _top_camera_callback(
        self,
        message: Image,
    ) -> None:
        try:
            update_rgb_image_cache_from_message(
                self.top_image_cache,
                message,
            )
        except ValueError as error:
            self.get_logger().error(
                f"Rejected top-camera frame: {error}"
            )

    def _gripper_camera_callback(
        self,
        message: Image,
    ) -> None:
        try:
            update_rgb_image_cache_from_message(
                self.gripper_image_cache,
                message,
            )
        except ValueError as error:
            self.get_logger().error(
                f"Rejected gripper-camera frame: {error}"
            )


def main(args: list[str] | None = None) -> None:
    """Run the ROS I/O node without starting a training loop."""

    rclpy.init(args=args)
    node = So101ReachRosNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
