import numpy as np
import pytest

from so101_capstone.robot.so101_forward_kinematics import (
    ACTIVE_JOINT_NAMES,
    GRASP_POINT_OFFSET,
    compute_grasp_point_pose,
    joint_positions_from_names,
)


def zero_joint_positions():
    return {
        name: 0.0
        for name in ACTIVE_JOINT_NAMES
    }


def test_calibrated_offset_is_preserved():
    np.testing.assert_allclose(
        GRASP_POINT_OFFSET,
        [-0.018, 0.0, -0.008],
        atol=0.0,
    )


def test_zero_pose_matches_current_xacro_geometry():
    result = compute_grasp_point_pose(
        zero_joint_positions()
    )

    np.testing.assert_allclose(
        result.position,
        [
            0.383361314,
            -0.000885224,
            0.996848312,
        ],
        atol=1e-8,
    )

    np.testing.assert_allclose(
        np.linalg.norm(result.quaternion_xyzw),
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.transform[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=0.0,
    )


def test_gripper_feedback_does_not_change_reference_point():
    closed = zero_joint_positions()
    closed["gripper"] = -0.174533

    open_ = zero_joint_positions()
    open_["gripper"] = 1.74533

    closed_result = compute_grasp_point_pose(closed)
    open_result = compute_grasp_point_pose(open_)

    np.testing.assert_allclose(
        closed_result.transform,
        open_result.transform,
        atol=0.0,
    )


def test_joint_feedback_is_reordered_by_name():
    names = [
        "gripper",
        "wrist_roll",
        "shoulder_pan",
        "elbow_flex",
        "shoulder_lift",
        "wrist_flex",
    ]

    positions = [
        0.9,
        0.5,
        0.1,
        0.3,
        0.2,
        0.4,
    ]

    ordered = joint_positions_from_names(
        names,
        positions,
    )

    assert ordered == {
        "shoulder_pan": 0.1,
        "shoulder_lift": 0.2,
        "elbow_flex": 0.3,
        "wrist_flex": 0.4,
        "wrist_roll": 0.5,
    }


def test_missing_joint_feedback_is_rejected():
    incomplete = zero_joint_positions()
    del incomplete["wrist_roll"]

    with pytest.raises(
        ValueError,
        match="wrist_roll",
    ):
        compute_grasp_point_pose(incomplete)


def test_nonzero_pose_matches_tf_reference():
    """Verify FK against the independently observed ROS TF result."""

    measured_joint_positions = {
        "shoulder_pan": 0.2,
        "shoulder_lift": -0.3,
        "elbow_flex": 0.4,
        "wrist_flex": -0.2,
        "wrist_roll": 0.0,
    }

    result = compute_grasp_point_pose(
        measured_joint_positions
    )

    expected_position = np.array(
        [
            0.340610413,
            -0.062076310,
            1.001648498,
        ],
        dtype=float,
    )

    expected_quaternion = np.array(
        [
            0.08319973,
            0.66570024,
            -0.05604920,
            0.73944539,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        result.position,
        expected_position,
        atol=1e-8,
        rtol=0.0,
    )

    actual_quaternion = (
        result.quaternion_xyzw
        / np.linalg.norm(result.quaternion_xyzw)
    )
    expected_quaternion = (
        expected_quaternion
        / np.linalg.norm(expected_quaternion)
    )

    # q and -q represent the same physical orientation.
    orientation_similarity = abs(
        float(
            np.dot(
                actual_quaternion,
                expected_quaternion,
            )
        )
    )

    np.testing.assert_allclose(
        orientation_similarity,
        1.0,
        atol=1e-8,
        rtol=0.0,
    )
