"""Forward kinematics for the SO-101 grasp reference point.

The grasp point is fixed relative to ``gripper_frame_link``. Joint positions
must come from measured feedback, such as ROS 2 ``/joint_states``, rather than
from commanded target positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ACTIVE_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# Calibrated relative to gripper_frame_link.
GRASP_POINT_OFFSET = np.array([-0.018, 0.0, -0.008], dtype=float)

# Gazebo task-world position of base_link.
DEFAULT_WORLD_FROM_BASE_TRANSLATION = np.array(
    [0.0, 0.0, 0.7524],
    dtype=float,
)

# Each entry contains URDF joint origin xyz and rpy.
# Joint axis is +Z for all five active arm joints.
_JOINT_ORIGINS = (
    (
        np.array([0.0388353, -8.97657e-09, 0.0624], dtype=float),
        np.array([3.14159, 4.18253e-17, -3.14159], dtype=float),
    ),
    (
        np.array([-0.0303992, -0.0182778, -0.0542], dtype=float),
        np.array([-1.5708, -1.5708, 0.0], dtype=float),
    ),
    (
        np.array([-0.11257, -0.028, 1.73763e-16], dtype=float),
        np.array([-3.63608e-16, 8.74301e-16, 1.5708], dtype=float),
    ),
    (
        np.array([-0.1349, 0.0052, 3.62355e-17], dtype=float),
        np.array([4.02456e-15, 8.67362e-16, -1.5708], dtype=float),
    ),
    (
        np.array([5.55112e-17, -0.0611, 0.0181], dtype=float),
        np.array([1.5708, 0.0486795, 3.14159], dtype=float),
    ),
)

# Fixed joint: gripper_link -> gripper_frame_link.
_GRIPPER_FRAME_ORIGIN_XYZ = np.array(
    [-0.0079, -0.000218121, -0.0981274],
    dtype=float,
)
_GRIPPER_FRAME_ORIGIN_RPY = np.array(
    [0.0, 3.14159, 0.0],
    dtype=float,
)


@dataclass(frozen=True)
class GraspPointPose:
    """World-frame pose of the calibrated grasp reference point."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray
    transform: np.ndarray


def _rotation_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Return URDF fixed-axis RPY rotation: Rz(yaw) Ry(pitch) Rx(roll)."""

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )

    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )

    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return rotation_z @ rotation_y @ rotation_x


def _transform_from_xyz_rpy(
    xyz: Sequence[float],
    rpy: Sequence[float],
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_from_rpy(*map(float, rpy))
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _rotation_about_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return transform


def _translation_transform(
    translation: Sequence[float],
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _quaternion_xyzw_from_rotation(
    rotation: np.ndarray,
) -> np.ndarray:
    """Convert a 3x3 rotation matrix to normalized [x, y, z, w]."""

    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))

        if diagonal_index == 0:
            scale = 2.0 * np.sqrt(
                1.0
                + matrix[0, 0]
                - matrix[1, 1]
                - matrix[2, 2]
            )
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale

        elif diagonal_index == 1:
            scale = 2.0 * np.sqrt(
                1.0
                + matrix[1, 1]
                - matrix[0, 0]
                - matrix[2, 2]
            )
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale

        else:
            scale = 2.0 * np.sqrt(
                1.0
                + matrix[2, 2]
                - matrix[0, 0]
                - matrix[1, 1]
            )
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def joint_positions_from_names(
    names: Sequence[str],
    positions: Sequence[float],
) -> dict[str, float]:
    """Extract and reorder measured positions using joint names."""

    if len(names) != len(positions):
        raise ValueError(
            "Joint name and position arrays must have the same length."
        )

    positions_by_name = {
        name: float(position)
        for name, position in zip(names, positions)
    }

    missing = [
        name
        for name in ACTIVE_JOINT_NAMES
        if name not in positions_by_name
    ]

    if missing:
        raise ValueError(
            f"Missing required joint positions: {', '.join(missing)}"
        )

    return {
        name: positions_by_name[name]
        for name in ACTIVE_JOINT_NAMES
    }


def compute_grasp_point_pose(
    joint_positions: Mapping[str, float],
    *,
    world_from_base: np.ndarray | None = None,
    grasp_point_offset: Sequence[float] = GRASP_POINT_OFFSET,
) -> GraspPointPose:
    """Compute the grasp-point world pose from measured arm joint positions.

    Extra entries such as the measured ``gripper`` joint are accepted but do
    not affect the grasp-point pose.
    """

    missing = [
        name
        for name in ACTIVE_JOINT_NAMES
        if name not in joint_positions
    ]

    if missing:
        raise ValueError(
            f"Missing required joint positions: {', '.join(missing)}"
        )

    if world_from_base is None:
        world_from_base_matrix = _translation_transform(
            DEFAULT_WORLD_FROM_BASE_TRANSLATION
        )
    else:
        world_from_base_matrix = np.asarray(
            world_from_base,
            dtype=float,
        )

        if world_from_base_matrix.shape != (4, 4):
            raise ValueError(
                "world_from_base must be a 4x4 homogeneous transform."
            )

        if not np.all(np.isfinite(world_from_base_matrix)):
            raise ValueError(
                "world_from_base must contain only finite values."
            )

    base_from_link = np.eye(4, dtype=float)

    for joint_name, (origin_xyz, origin_rpy) in zip(
        ACTIVE_JOINT_NAMES,
        _JOINT_ORIGINS,
    ):
        base_from_link = (
            base_from_link
            @ _transform_from_xyz_rpy(origin_xyz, origin_rpy)
            @ _rotation_about_z(float(joint_positions[joint_name]))
        )

    base_from_gripper_frame = (
        base_from_link
        @ _transform_from_xyz_rpy(
            _GRIPPER_FRAME_ORIGIN_XYZ,
            _GRIPPER_FRAME_ORIGIN_RPY,
        )
    )

    gripper_frame_from_point = _translation_transform(
        grasp_point_offset
    )

    world_from_point = (
        world_from_base_matrix
        @ base_from_gripper_frame
        @ gripper_frame_from_point
    )

    return GraspPointPose(
        position=world_from_point[:3, 3].copy(),
        quaternion_xyzw=_quaternion_xyzw_from_rotation(
            world_from_point[:3, :3]
        ),
        transform=world_from_point.copy(),
    )
