from __future__ import annotations

import math

import numpy as np
import pytest

from so101_capstone.rewards.reach_reward import (
    ReachRewardConfig,
    ReachRewardInput,
    RewardState,
    StatefulReachReward,
)


HOME = (
    0.0,
    -1.710422666954443,
    1.605702911834783,
    1.134464013796314,
    0.0,
    -0.087266462599716,
)
GOAL = (10.0, 10.0, 0.0)


def _target_for_angle_deg(
    angle_deg: float,
    *,
    length_m: float = 0.10,
) -> tuple[float, float, float]:
    theta = math.radians(angle_deg)
    return (
        length_m * math.sin(theta),
        0.0,
        -0.04 + length_m * math.cos(theta),
    )


def _step(
    tracker: StatefulReachReward,
    *,
    target: tuple[float, float, float],
    rotation: np.ndarray | None = None,
):
    if rotation is None:
        rotation = np.eye(3, dtype=float)
    return tracker.step(
        ReachRewardInput(
            joint_positions_rad=HOME,
            grasp_position_world_m=(0.0, 0.0, 0.0),
            grasp_rotation_world=rotation,
            goal_region_center_world_m=GOAL,
            target_name="target_cube_0001",
            target_position_world_m=target,
        )
    )


def test_state2_local_z_target_config_defaults() -> None:
    config = ReachRewardConfig()
    assert config.state2_local_z_target_reference_offset_m == pytest.approx(
        0.04
    )
    assert config.state2_local_z_target_threshold_deg == pytest.approx(3.0)
    assert (
        config.state2_local_z_target_misaligned_penalty_per_step
        == pytest.approx(0.1)
    )
    assert config.state2_local_z_target_success_bonus == pytest.approx(5.0)


def test_state2_local_z_penalizes_each_misaligned_step() -> None:
    tracker = StatefulReachReward()
    tracker.reset(joint_positions_rad=HOME)

    first = _step(tracker, target=_target_for_angle_deg(10.0))
    second = _step(tracker, target=_target_for_angle_deg(10.0))

    assert first.state is RewardState.TARGET_APPROACH
    assert second.state is RewardState.TARGET_APPROACH
    assert first.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(-0.1)
    assert second.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(-0.1)


def test_success_once_and_penalty_resumes_after_exit() -> None:
    tracker = StatefulReachReward()
    tracker.reset(joint_positions_rad=HOME)

    misaligned = _step(tracker, target=_target_for_angle_deg(10.0))
    aligned_first = _step(tracker, target=_target_for_angle_deg(2.0))
    aligned_hold = _step(tracker, target=_target_for_angle_deg(1.0))
    misaligned_again = _step(tracker, target=_target_for_angle_deg(8.0))
    aligned_again = _step(tracker, target=_target_for_angle_deg(2.0))

    assert misaligned.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(-0.1)

    assert aligned_first.components[
        "state2_local_z_target_success"
    ] == pytest.approx(5.0)
    assert aligned_first.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(0.0)

    assert aligned_hold.components[
        "state2_local_z_target_success"
    ] == pytest.approx(0.0)
    assert aligned_hold.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(0.0)

    assert misaligned_again.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(-0.1)

    assert aligned_again.components[
        "state2_local_z_target_success"
    ] == pytest.approx(0.0)
    assert aligned_again.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(0.0)


def test_four_cm_inside_reference_handles_target_at_compare_point() -> None:
    tracker = StatefulReachReward()
    tracker.reset(joint_positions_rad=HOME)

    result = _step(tracker, target=(0.0, 0.0, 0.0))

    assert result.state is RewardState.TARGET_APPROACH
    assert result.components[
        "state2_local_z_target_success"
    ] == pytest.approx(5.0)


def test_uses_gripper_local_z_not_world_z() -> None:
    tracker = StatefulReachReward()
    tracker.reset(joint_positions_rad=HOME)

    rotation = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ),
        dtype=float,
    )

    result = _step(
        tracker,
        target=(0.06, 0.0, 0.0),
        rotation=rotation,
    )

    assert result.state is RewardState.TARGET_APPROACH
    assert result.components[
        "state2_local_z_target_success"
    ] == pytest.approx(5.0)
    assert result.components[
        "state2_local_z_target_misaligned"
    ] == pytest.approx(0.0)


def test_target_switch_rearms_success_bonus() -> None:
    tracker = StatefulReachReward()
    tracker.reset(joint_positions_rad=HOME)

    first = _step(tracker, target=_target_for_angle_deg(2.0))
    assert first.components[
        "state2_local_z_target_success"
    ] == pytest.approx(5.0)

    second = tracker.step(
        ReachRewardInput(
            joint_positions_rad=HOME,
            grasp_position_world_m=(0.0, 0.0, 0.0),
            grasp_rotation_world=np.eye(3, dtype=float),
            goal_region_center_world_m=GOAL,
            target_name="target_cube_0002",
            target_position_world_m=_target_for_angle_deg(2.0),
        )
    )

    assert second.state is RewardState.TARGET_APPROACH
    assert second.components[
        "state2_local_z_target_success"
    ] == pytest.approx(5.0)
