import math

import numpy as np
import pytest

from so101_capstone.rewards import (
    ReachRewardConfig,
    ReachRewardInput,
    RewardState,
    StatefulReachReward,
    compute_reach_reward,
)


HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
IDENTITY = np.eye(3)
GOAL_FAR = (1.0, 1.0, 0.0)


def rad(degrees):
    return math.radians(degrees)


def quiet_config(**overrides):
    values = {
        "home_joint_positions": HOME,
        "state1_stationary_reward": 0.0,
        "state1_motion_penalty_per_deg": 0.0,
        "early_approach_bonus": 0.0,
        "distance_penalty_weight": 0.0,
        "distance_improvement_weight": 0.0,
        "state2_distance_increase_penalty_per_m": 0.0,
        "state2_local_z_target_misaligned_penalty_per_step": 0.0,
        "state2_local_z_target_success_bonus": 0.0,
        "state2_counterclockwise_wrist_roll_penalty_per_deg": 0.0,
        "gripper_opening_reward_per_deg": 0.0,
        "gripper_opening_reward_cap": 0.0,
        "gripper_still_reward": 0.0,
        "gripper_motion_penalty": 0.0,
        "gripper_target_opening_penalty_per_deg": 0.0,
        "alignment_progress_reward_per_bucket": 0.0,
        "alignment_success_bonus": 0.0,
        "alignment_exit_penalty": 0.0,
        "target_contact_bonus": 0.0,
        "target_contact_exit_penalty": 0.0,
        "state3_hold_reward": 0.0,
        "gripper_closing_reward_per_deg": 0.0,
        "gripper_closing_reward_cap": 0.0,
        "target_lift_bonus": 0.0,
        "target_drop_penalty": 0.0,
        "goal_entry_bonus": 0.0,
        "goal_exit_penalty": 0.0,
        "state4_gripper_open_reward_per_deg": 0.0,
        "state4_gripper_open_reward_cap": 0.0,
        "state4_not_home_penalty": 0.0,
        "success_bonus": 0.0,
    }
    values.update(overrides)
    return ReachRewardConfig(**values)


def observation(
    *,
    joints=HOME,
    grasp=(0.0, 0.0, 0.0),
    rotation=IDENTITY,
    goal=GOAL_FAR,
    target_name=None,
    target=None,
):
    return ReachRewardInput(
        joint_positions_rad=joints,
        grasp_position_world_m=grasp,
        grasp_rotation_world=rotation,
        goal_region_center_world_m=goal,
        target_name=target_name,
        target_position_world_m=target,
    )


def reset_tracker(
    config=None,
    *,
    joints=HOME,
    grasp=(0.0, 0.0, 0.0),
    rotation=IDENTITY,
    goal=GOAL_FAR,
    target_name=None,
    target=None,
):
    tracker = StatefulReachReward(config or quiet_config())
    tracker.reset(
        joint_positions_rad=joints,
        target_name=target_name,
        target_position_world_m=target,
        grasp_position_world_m=(grasp if target_name is not None else None),
        grasp_rotation_world=(rotation if target_name is not None else None),
        goal_region_center_world_m=goal,
    )
    return tracker


def rotation_with_x_ground_error(angle_deg):
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=float,
    )


def test_step_requires_reset():
    tracker = StatefulReachReward(quiet_config())

    with pytest.raises(RuntimeError, match="reset"):
        tracker.step(observation())


def test_state1_stationary_reward():
    tracker = reset_tracker(
        quiet_config(state1_stationary_reward=0.001)
    )

    result = tracker.step(observation())

    assert result.state is RewardState.TARGET_ABSENT
    assert result.reward == pytest.approx(0.001)
    assert result.components["state1_stationary"] == pytest.approx(0.001)
    assert result.distance_m is None
    assert result.distance_improvement_m is None
    assert result.events == ("state1_stationary",)


def test_state1_home_tolerance_gives_maximum_reward():
    tracker = reset_tracker(
        quiet_config(
            state1_stationary_reward=0.1,
            state1_home_tolerance_deg=1.0,
            state1_home_error_scale_deg=20.0,
            state1_home_negative_start_deg=30.0,
        )
    )
    offsets_deg = (1.0, -1.0, 0.5, -0.5, 1.0, -1.0)
    joints = tuple(
        home + rad(offset)
        for home, offset in zip(HOME, offsets_deg)
    )

    result = tracker.step(observation(joints=joints))

    assert result.state is RewardState.TARGET_ABSENT
    assert result.reward == pytest.approx(0.1)
    assert result.components["state1_stationary"] == pytest.approx(0.1)
    assert result.components["state1_home_error"] == pytest.approx(0.0)
    assert result.events == ("state1_stationary",)


def test_state1_home_error_uses_squared_shaping_after_tolerance():
    tracker = reset_tracker(
        quiet_config(
            state1_stationary_reward=0.1,
            state1_home_tolerance_deg=1.0,
            state1_home_error_scale_deg=20.0,
            state1_home_negative_start_deg=30.0,
        )
    )
    joints = tuple(
        home + rad(11.0)
        for home in HOME
    )

    result = tracker.step(observation(joints=joints))

    # Effective error on every joint is 10 deg, so mean = 10 deg:
    # 0.1 * (1 - (10 / 20)^2) = 0.075.
    assert result.reward == pytest.approx(0.075)
    assert result.components["state1_home_error"] == pytest.approx(0.075)
    assert result.components["state1_motion"] == pytest.approx(0.0)
    assert result.events == ("state1_home_error",)


def test_state1_home_error_becomes_negative_when_far_from_home():
    tracker = reset_tracker(
        quiet_config(
            state1_stationary_reward=0.5,
            state1_home_tolerance_deg=1.0,
            state1_home_error_scale_deg=10.0,
            state1_home_negative_start_deg=20.0,
        )
    )
    joints = tuple(
        home + rad(41.0)
        for home in HOME
    )

    result = tracker.step(observation(joints=joints))

    # Effective mean error = 40 deg:
    # -0.5 * ((40 - 20) / 10)^2 = -2.0.
    assert result.reward == pytest.approx(-2.0)


def test_state1_home_error_is_zero_between_positive_and_negative_regions():
    tracker = reset_tracker(
        quiet_config(
            state1_stationary_reward=0.5,
            state1_home_tolerance_deg=1.0,
            state1_home_error_scale_deg=10.0,
            state1_home_negative_start_deg=20.0,
        )
    )
    joints = tuple(
        home + rad(16.0)
        for home in HOME
    )

    result = tracker.step(observation(joints=joints))

    # Raw error = 16 deg; after the 1 deg deadband,
    # effective mean HOME error = 15 deg, which is in the zero plateau.
    assert result.reward == pytest.approx(0.0)
    assert result.components["state1_home_error"] == pytest.approx(0.0)
    assert result.events == ("state1_home_error",)


def test_state1_home_error_scale_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        ReachRewardConfig(state1_home_error_scale_deg=0.0)



def test_four_states_are_mutually_exclusive_and_goal_uses_xy_only():
    tracker = reset_tracker(quiet_config())

    absent = tracker.step(observation())

    state2 = tracker.step(
        observation(
            target_name="cube",
            target=(0.2, 0.0, 5.0),
            goal=(0.0, 0.0, 0.0),
        )
    )

    state3 = tracker.step(
        observation(
            target_name="cube",
            target=(0.2, 0.0, 5.0),
            grasp=(0.2, 0.0, 5.0),
            goal=(0.0, 0.0, 0.0),
        )
    )

    state4 = tracker.step(
        observation(
            target_name="cube",
            target=(0.02, -0.02, 50.0),
            grasp=(0.2, 0.0, 5.0),
            goal=(0.0, 0.0, 0.0),
        )
    )

    assert absent.state is RewardState.TARGET_ABSENT
    assert state2.state is RewardState.TARGET_APPROACH
    assert state3.state is RewardState.TARGET_GRASP_TRANSPORT
    assert state4.state is RewardState.TARGET_RELEASE_RETURN_HOME
    assert state4.target_in_goal_region is True

def test_target_activation_has_zero_distance_progress():
    tracker = reset_tracker(quiet_config())

    result = tracker.step(
        observation(target_name="cube", target=(0.0, 0.0, 0.2))
    )

    assert result.state is RewardState.TARGET_OUTSIDE_GOAL
    assert result.target_age_steps == 1
    assert result.distance_m == pytest.approx(0.2)
    assert result.distance_improvement_m == pytest.approx(0.0)
    assert "target_activated" in result.events
    assert result.components["distance_improvement"] == pytest.approx(0.0)


def test_early_approach_is_disabled_and_record_distance_reward_remains():
    tracker = reset_tracker(
        quiet_config(
            early_approach_bonus=1.0,
            distance_improvement_weight=10.0,
        ),
        target_name="cube",
        target=(0.0, 0.0, 0.10),
    )

    first = tracker.step(
        observation(
            grasp=(0.0, 0.0, 0.01),
            target_name="cube",
            target=(0.0, 0.0, 0.10),
        )
    )
    away = tracker.step(
        observation(
            grasp=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.10),
        )
    )
    new_record = tracker.step(
        observation(
            grasp=(0.0, 0.0, 0.02),
            target_name="cube",
            target=(0.0, 0.0, 0.10),
        )
    )

    assert first.components["early_approach"] == pytest.approx(0.0)
    assert "early_approach" not in first.events
    assert first.distance_improvement_m == pytest.approx(0.01)
    assert first.components["distance_improvement"] == pytest.approx(0.001)

    assert away.components["early_approach"] == pytest.approx(0.0)
    assert away.distance_improvement_m == pytest.approx(0.0)
    assert away.components["distance_improvement"] == pytest.approx(0.0)

    assert new_record.components["early_approach"] == pytest.approx(0.0)
    assert new_record.distance_improvement_m == pytest.approx(0.01)
    assert new_record.components["distance_improvement"] == pytest.approx(0.001)



def test_state2_counterclockwise_wrist_roll_penalty_repeats_each_step():
    tracker = reset_tracker(
        quiet_config(
            state2_counterclockwise_wrist_roll_penalty_per_deg=1.0,
        ),
        target_name="cube",
        target=(0.2, 0.0, 0.0),
    )
    counterclockwise_joints = (
        0.0,
        0.0,
        0.0,
        0.0,
        rad(15.0),
        0.0,
    )

    first = tracker.step(
        observation(
            joints=counterclockwise_joints,
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    second = tracker.step(
        observation(
            joints=counterclockwise_joints,
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )

    assert first.state is RewardState.TARGET_OUTSIDE_GOAL
    assert first.components[
        "state2_counterclockwise_wrist_roll"
    ] == pytest.approx(-15.0)
    assert first.reward == pytest.approx(-15.0)

    assert second.components[
        "state2_counterclockwise_wrist_roll"
    ] == pytest.approx(-15.0)
    assert second.reward == pytest.approx(-15.0)
    assert (
        "state2_counterclockwise_wrist_roll"
        in second.events
    )


def test_state2_wrist_roll_penalty_ignores_clockwise_and_other_states():
    tracker = reset_tracker(
        quiet_config(
            state2_counterclockwise_wrist_roll_penalty_per_deg=1.0,
        )
    )
    clockwise_joints = (
        0.0,
        0.0,
        0.0,
        0.0,
        rad(-15.0),
        0.0,
    )
    counterclockwise_joints = (
        0.0,
        0.0,
        0.0,
        0.0,
        rad(15.0),
        0.0,
    )

    state1 = tracker.step(
        observation(joints=counterclockwise_joints)
    )
    clockwise_state2 = tracker.step(
        observation(
            joints=clockwise_joints,
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    state3 = tracker.step(
        observation(
            joints=counterclockwise_joints,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )

    assert state1.state is RewardState.TARGET_ABSENT
    assert state1.components[
        "state2_counterclockwise_wrist_roll"
    ] == pytest.approx(0.0)

    assert clockwise_state2.state is RewardState.TARGET_OUTSIDE_GOAL
    assert clockwise_state2.components[
        "state2_counterclockwise_wrist_roll"
    ] == pytest.approx(0.0)

    assert state3.state is RewardState.TARGET_INSIDE_GOAL
    assert state3.components[
        "state2_counterclockwise_wrist_roll"
    ] == pytest.approx(0.0)


def test_distance_penalty_starts_on_first_state2_step():
    tracker = reset_tracker(
        quiet_config(distance_penalty_weight=1.0),
        target_name="cube",
        target=(0.0, 0.0, 0.2),
    )

    first = tracker.step(
        observation(target_name="cube", target=(0.0, 0.0, 0.2))
    )
    second = tracker.step(
        observation(target_name="cube", target=(0.0, 0.0, 0.2))
    )

    assert first.target_age_steps == 1
    assert first.components["distance_penalty"] == pytest.approx(-0.04)
    assert second.target_age_steps == 2
    assert second.components["distance_penalty"] == pytest.approx(-0.04)


def test_home_based_gripper_opening_reward_has_45_degree_cap():
    tracker = reset_tracker(
        quiet_config(
            gripper_opening_reward_per_deg=0.1,
            gripper_opening_reward_cap=4.5,
        ),
        target_name="cube",
        target=(0.2, 0.0, 0.0),
    )

    ten_deg = tracker.step(
        observation(
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, rad(10.0)),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    fifty_deg = tracker.step(
        observation(
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, rad(50.0)),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    repeated = tracker.step(
        observation(
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, rad(60.0)),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )

    assert ten_deg.components["gripper_opening"] == pytest.approx(1.0)
    assert fifty_deg.components["gripper_opening"] == pytest.approx(3.5)
    assert repeated.components["gripper_opening"] == pytest.approx(0.0)



def test_state2_gripper_penalizes_opening_error_from_45_degrees():
    config = quiet_config(
        gripper_target_opening_deg=45.0,
        gripper_target_opening_penalty_per_deg=1.0,
    )
    tracker = reset_tracker(
        config,
        target_name="cube",
        target=(0.2, 0.0, 0.0),
    )

    at_target = list(HOME)
    at_target[-1] = HOME[-1] + rad(45.0)
    exact = tracker.step(
        observation(
            joints=tuple(at_target),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )

    below_target = list(HOME)
    below_target[-1] = HOME[-1] + rad(30.0)
    below = tracker.step(
        observation(
            joints=tuple(below_target),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )

    above_target = list(HOME)
    above_target[-1] = HOME[-1] + rad(60.0)
    above = tracker.step(
        observation(
            joints=tuple(above_target),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )

    assert exact.state is RewardState.TARGET_APPROACH
    assert exact.components["gripper_stability"] == pytest.approx(0.0)
    assert below.components["gripper_stability"] == pytest.approx(-15.0)
    assert above.components["gripper_stability"] == pytest.approx(-15.0)

def test_compare_x_ground_parallel_reward_is_record_based_and_penalizes_exits():
    tracker = reset_tracker(
        quiet_config(
            alignment_progress_bucket_deg=1.0,
            alignment_progress_reward_per_bucket=0.1,
            alignment_success_threshold_deg=3.0,
            alignment_success_bonus=1.0,
            alignment_exit_threshold_deg=3.0,
            alignment_exit_penalty=2.0,
        ),
        rotation=rotation_with_x_ground_error(10.0),
        target_name="cube",
        target=(0.2, 0.0, 0.0),
    )

    nine = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(9.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    two = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(2.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    one = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(1.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    four = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(4.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    still_outside = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(6.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    back_inside = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(2.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    leave_again = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(4.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )

    assert nine.alignment_error_deg == pytest.approx(9.0)
    assert nine.components["alignment_progress"] == pytest.approx(0.1)
    assert two.components["alignment_progress"] == pytest.approx(0.7)
    assert two.components["alignment_success"] == pytest.approx(1.0)
    assert one.components["alignment_progress"] == pytest.approx(0.1)
    assert four.components["alignment_exit"] == pytest.approx(-2.0)
    assert still_outside.components["alignment_exit"] == pytest.approx(0.0)
    assert back_inside.components["alignment_success"] == pytest.approx(0.0)
    assert leave_again.components["alignment_exit"] == pytest.approx(-2.0)


def test_contact_bonus_is_once_per_episode_even_after_reentry_and_target_switch():
    tracker = reset_tracker(
        quiet_config(
            target_contact_bonus=5.0,
            target_contact_exit_penalty=10.0,
        ),
        target_name="cube1",
        target=(0.0, 0.0, 0.03),
    )

    enter = tracker.step(
        observation(target_name="cube1", target=(0.0, 0.0, 0.005))
    )
    leave = tracker.step(
        # 3 cm is strictly outside the current inclusive 2 cm contact radius.
        observation(target_name="cube1", target=(0.0, 0.0, 0.03))
    )
    reenter = tracker.step(
        observation(target_name="cube1", target=(0.0, 0.0, 0.005))
    )
    switch_inside = tracker.step(
        observation(target_name="cube2", target=(0.0, 0.0, 0.005))
    )

    assert enter.state is RewardState.TARGET_GRASP_TRANSPORT
    assert leave.state is RewardState.TARGET_APPROACH
    assert reenter.state is RewardState.TARGET_GRASP_TRANSPORT
    assert enter.components["target_contact"] == pytest.approx(5.0)
    assert leave.components["target_contact_exit"] == pytest.approx(-10.0)
    assert reenter.components["target_contact"] == pytest.approx(0.0)
    assert switch_inside.components["target_contact"] == pytest.approx(0.0)


def test_closing_inside_contact_radius_is_incremental_and_capped_per_episode():
    tracker = reset_tracker(
        quiet_config(
            gripper_closing_reward_per_deg=0.1,
            gripper_closing_reward_cap=4.0,
        ),
        target_name="cube",
        target=(0.0, 0.0, 0.02),
    )
    tracker.step(
        observation(target_name="cube", target=(0.0, 0.0, 0.005))
    )

    ten = tracker.step(
        observation(
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, rad(-10.0)),
            target_name="cube",
            target=(0.0, 0.0, 0.005),
        )
    )
    fifty = tracker.step(
        observation(
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, rad(-50.0)),
            target_name="cube",
            target=(0.0, 0.0, 0.005),
        )
    )
    repeated = tracker.step(
        observation(
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, rad(-60.0)),
            target_name="cube",
            target=(0.0, 0.0, 0.005),
        )
    )

    assert ten.components["gripper_closing"] == pytest.approx(1.0)
    assert fifty.components["gripper_closing"] == pytest.approx(3.0)
    assert repeated.components["gripper_closing"] == pytest.approx(0.0)



def test_lift_bonus_is_state3_only_once_per_target_and_each_lift_arms_drop():
    tracker = reset_tracker(
        quiet_config(
            target_lift_bonus=2.0,
            target_drop_penalty=2.0,
        ),
        target_name="cube",
        target=(0.2, 0.0, 0.765),
    )

    enter_state3 = tracker.step(
        observation(
            target_name="cube",
            target=(0.2, 0.0, 0.765),
            grasp=(0.2, 0.0, 0.765),
        )
    )
    assert enter_state3.state is RewardState.TARGET_GRASP_TRANSPORT
    assert enter_state3.components["target_lift"] == pytest.approx(0.0)

    first_lift = tracker.step(
        observation(
            target_name="cube",
            target=(0.2, 0.0, 0.785),
            grasp=(0.2, 0.0, 0.785),
        )
    )
    first_drop = tracker.step(
        observation(
            target_name="cube",
            target=(0.2, 0.0, 0.765),
            grasp=(0.2, 0.0, 0.765),
        )
    )
    second_lift = tracker.step(
        observation(
            target_name="cube",
            target=(0.2, 0.0, 0.785),
            grasp=(0.2, 0.0, 0.785),
        )
    )
    second_drop = tracker.step(
        observation(
            target_name="cube",
            target=(0.2, 0.0, 0.765),
            grasp=(0.2, 0.0, 0.765),
        )
    )

    assert first_lift.state is RewardState.TARGET_GRASP_TRANSPORT
    assert first_lift.components["target_lift"] == pytest.approx(2.0)
    assert first_drop.components["target_drop"] == pytest.approx(-2.0)
    assert second_lift.components["target_lift"] == pytest.approx(0.0)
    assert second_drop.components["target_drop"] == pytest.approx(-2.0)


def test_goal_entry_is_state3_once_per_target_and_goal_exit_can_repeat():
    non_home = (rad(10.0), 0.0, 0.0, 0.0, 0.0, 0.0)
    tracker = reset_tracker(
        quiet_config(goal_entry_bonus=10.0, goal_exit_penalty=11.0),
        joints=non_home,
        target_name="cube",
        target=(0.2, 0.0, 0.765),
        goal=(0.0, 0.0, 0.0),
    )

    enter_state3 = tracker.step(
        observation(
            joints=non_home,
            grasp=(0.2, 0.0, 0.765),
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.2, 0.0, 0.765),
        )
    )
    assert enter_state3.state is RewardState.TARGET_GRASP_TRANSPORT

    enter = tracker.step(
        observation(
            joints=non_home,
            grasp=(0.0, 0.0, 0.765),
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    assert enter.state is RewardState.TARGET_RELEASE_RETURN_HOME
    assert enter.components["goal_entry"] == pytest.approx(10.0)

    exit_one = tracker.step(
        observation(
            joints=non_home,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.2, 0.0, 0.765),
        )
    )
    assert exit_one.state is RewardState.TARGET_APPROACH
    assert exit_one.components["goal_exit"] == pytest.approx(-11.0)

    enter_state3_again = tracker.step(
        observation(
            joints=non_home,
            grasp=(0.2, 0.0, 0.765),
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.2, 0.0, 0.765),
        )
    )
    assert (
        enter_state3_again.state
        is RewardState.TARGET_GRASP_TRANSPORT
    )

    reenter = tracker.step(
        observation(
            joints=non_home,
            grasp=(0.0, 0.0, 0.765),
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    assert reenter.state is RewardState.TARGET_RELEASE_RETURN_HOME
    assert reenter.components["goal_entry"] == pytest.approx(0.0)

    exit_two = tracker.step(
        observation(
            joints=non_home,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.2, 0.0, 0.765),
        )
    )
    assert exit_two.components["goal_exit"] == pytest.approx(-11.0)

def test_target_switch_does_not_create_false_goal_exit():
    non_home = (rad(10.0), 0.0, 0.0, 0.0, 0.0, 0.0)
    tracker = reset_tracker(
        quiet_config(goal_exit_penalty=11.0),
        joints=non_home,
        goal=(0.0, 0.0, 0.0),
        target_name="inside",
        target=(0.0, 0.0, 0.765),
    )

    result = tracker.step(
        observation(
            joints=non_home,
            goal=(0.0, 0.0, 0.0),
            target_name="outside",
            target=(0.2, 0.0, 0.765),
        )
    )

    assert result.state is RewardState.TARGET_OUTSIDE_GOAL
    assert result.components["goal_exit"] == pytest.approx(0.0)
    assert "target_switched" in result.events


def test_state4_gripper_opening_reward_is_disabled():
    non_home_closed = (rad(10.0), 0.0, 0.0, 0.0, 0.0, rad(-20.0))
    tracker = reset_tracker(
        quiet_config(
            # Keep the legacy fields non-zero to prove they no longer
            # affect State4 reward.
            state4_gripper_open_reward_per_deg=0.2,
            state4_gripper_open_reward_cap=3.0,
        ),
        joints=non_home_closed,
        goal=(0.0, 0.0, 0.0),
        target_name="cube",
        target=(0.2, 0.0, 0.765),
    )

    results = []
    for gripper_deg in (-20.0, -10.0, 0.0, 10.0):
        results.append(
            tracker.step(
                observation(
                    joints=(
                        rad(10.0),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        rad(gripper_deg),
                    ),
                    goal=(0.0, 0.0, 0.0),
                    target_name="cube",
                    target=(0.0, 0.0, 0.765),
                )
            )
        )

    assert all(
        result.state is RewardState.TARGET_RELEASE_RETURN_HOME
        for result in results
    )
    assert all(
        result.components["state4_gripper_opening"]
        == pytest.approx(0.0)
        for result in results
    )

def test_state4_requires_consecutive_home_hold_before_success():
    config = quiet_config(
        state1_stationary_reward=0.5,
        state1_home_tolerance_deg=1.0,
        state1_home_error_scale_deg=10.0,
        state1_home_negative_start_deg=20.0,
        success_home_hold_steps=3,
        success_bonus=50.0,
    )
    non_home = tuple(
        home + rad(7.0)
        for home in HOME
    )
    tracker = reset_tracker(
        config,
        joints=non_home,
        goal=(0.0, 0.0, 0.0),
        target_name="cube",
        target=(0.0, 0.0, 0.765),
    )

    not_home = tracker.step(
        observation(
            joints=non_home,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    home_one = tracker.step(
        observation(
            joints=HOME,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    home_two = tracker.step(
        observation(
            joints=HOME,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    reset = tracker.step(
        observation(
            joints=non_home,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    tracker.step(
        observation(
            joints=HOME,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    tracker.step(
        observation(
            joints=HOME,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )
    success = tracker.step(
        observation(
            joints=HOME,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )

    # Raw 7 deg joint error minus the 1 deg deadband gives a mean
    # effective HOME error of 6 deg:
    # 0.5 * (1 - (6 / 10)^2) = 0.32.
    assert not_home.state is RewardState.TARGET_RELEASE_RETURN_HOME
    assert not_home.success is False
    assert not_home.components["state4_not_home"] == pytest.approx(0.32)
    assert not_home.success_home_hold_count == 0

    assert home_one.success is False
    assert home_one.components["state4_not_home"] == pytest.approx(0.5)
    assert home_one.success_home_hold_count == 1

    assert home_two.success is False
    assert home_two.success_home_hold_count == 2

    assert reset.success is False
    assert reset.components["state4_not_home"] == pytest.approx(0.32)
    assert reset.success_home_hold_count == 0

    assert success.success is True
    assert success.components["state4_not_home"] == pytest.approx(0.5)
    assert success.components["success"] == pytest.approx(50.0)
    assert success.success_home_hold_count == 3


@pytest.mark.parametrize(
    ("raw_error_deg", "expected_reward"),
    (
        (1.0, 0.5),
        (6.0, 0.375),
        (11.0, 0.0),
        (16.0, 0.0),
        (21.0, 0.0),
        (26.0, -0.125),
        (31.0, -0.5),
        (41.0, -2.0),
    ),
)
def test_state4_home_reward_matches_state1_piecewise_shaping(
    raw_error_deg,
    expected_reward,
):
    config = quiet_config(
        state1_stationary_reward=0.5,
        state1_home_tolerance_deg=1.0,
        state1_home_error_scale_deg=10.0,
        state1_home_negative_start_deg=20.0,
        success_home_hold_steps=999,
        success_bonus=50.0,
    )
    joints = tuple(
        home + rad(raw_error_deg)
        for home in HOME
    )
    tracker = reset_tracker(
        config,
        joints=joints,
        goal=(0.0, 0.0, 0.0),
        target_name="cube",
        target=(0.0, 0.0, 0.765),
    )

    result = tracker.step(
        observation(
            joints=joints,
            goal=(0.0, 0.0, 0.0),
            target_name="cube",
            target=(0.0, 0.0, 0.765),
        )
    )

    assert result.state is RewardState.TARGET_RELEASE_RETURN_HOME
    assert result.components["state4_not_home"] == pytest.approx(
        expected_reward
    )
    assert result.components["state4_gripper_opening"] == pytest.approx(0.0)

def test_default_compare_x_axis_is_parallel_to_ground_with_identity_rotation():
    tracker = reset_tracker(
        quiet_config(),
        target_name="cube",
        target=(0.2, 0.0, 0.0),
    )

    horizontal = tracker.step(
        observation(
            rotation=IDENTITY,
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )
    vertical = tracker.step(
        observation(
            rotation=rotation_with_x_ground_error(90.0),
            target_name="cube",
            target=(0.2, 0.0, 0.0),
        )
    )

    assert horizontal.alignment_error_deg == pytest.approx(0.0)
    assert vertical.alignment_error_deg == pytest.approx(90.0)


def test_compute_reach_reward_delegates_to_stateful_tracker():
    tracker = reset_tracker(
        quiet_config(state1_stationary_reward=0.001)
    )

    result = compute_reach_reward(tracker, observation())

    assert result.reward == pytest.approx(0.001)
    with pytest.raises(TypeError, match="StatefulReachReward"):
        compute_reach_reward(object(), observation())


def test_input_arrays_are_not_modified():
    joints = np.zeros(6, dtype=float)
    grasp = np.zeros(3, dtype=float)
    rotation = np.eye(3, dtype=float)
    goal = np.array(GOAL_FAR, dtype=float)
    target = np.array([0.0, 0.0, 0.2], dtype=float)
    originals = [value.copy() for value in (joints, grasp, rotation, goal, target)]
    tracker = reset_tracker(quiet_config())

    tracker.step(
        observation(
            joints=joints,
            grasp=grasp,
            rotation=rotation,
            goal=goal,
            target_name="cube",
            target=target,
        )
    )

    for actual, expected in zip((joints, grasp, rotation, goal, target), originals):
        np.testing.assert_array_equal(actual, expected)


def test_invalid_configuration_and_input_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        ReachRewardConfig(target_contact_radius_m=0.0)
    with pytest.raises(ValueError, match="greater than or equal"):
        ReachRewardConfig(
            alignment_success_threshold_deg=4.0,
            alignment_exit_threshold_deg=3.0,
        )
    with pytest.raises(ValueError, match="positive"):
        ReachRewardConfig(success_home_hold_steps=0)
    with pytest.raises(ValueError, match="six"):
        ReachRewardConfig(home_joint_positions=(0.0,) * 5)

    tracker = reset_tracker(quiet_config())
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        tracker.step(observation(grasp=(0.0, 0.0)))
    with pytest.raises(ValueError, match="required"):
        tracker.step(
            observation(target_name="cube", target=None)
        )

def test_state2_distance_increase_penalty_uses_delta_distance():
    config = quiet_config(
        state2_distance_increase_penalty_per_m=10.0,
    )
    tracker = reset_tracker(config)

    first = tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(0.0, 0.0, 0.0),
        )
    )
    assert first.state is RewardState.TARGET_APPROACH
    assert first.components["distance_increase"] == pytest.approx(0.0)

    farther = tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(-0.01, 0.0, 0.0),
        )
    )
    assert farther.state is RewardState.TARGET_APPROACH
    assert farther.components["distance_increase"] == pytest.approx(-0.1)
    assert "distance_increase" in farther.events

    closer = tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(0.01, 0.0, 0.0),
        )
    )
    assert closer.state is RewardState.TARGET_APPROACH
    assert closer.components["distance_increase"] == pytest.approx(0.0)


def test_state2_distance_increase_does_not_double_penalize_state3_exit():
    config = quiet_config(
        target_contact_exit_penalty=10.0,
        state2_distance_increase_penalty_per_m=10.0,
    )
    tracker = reset_tracker(config)

    tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(0.0, 0.0, 0.0),
        )
    )

    contact = tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(0.195, 0.0, 0.0),
        )
    )
    assert contact.state is RewardState.TARGET_GRASP_TRANSPORT

    lost = tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            # Leave the inclusive 2 cm contact region by a clear margin.
            grasp=(0.17, 0.0, 0.0),
        )
    )
    assert lost.state is RewardState.TARGET_APPROACH
    assert lost.components["target_contact_exit"] == pytest.approx(-10.0)
    assert lost.components["distance_increase"] == pytest.approx(0.0)


def test_state3_hold_reward_applies_every_state3_step():
    config = quiet_config(
        state3_hold_reward=0.1,
    )
    tracker = reset_tracker(config)

    tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(0.0, 0.0, 0.0),
        )
    )

    entered = tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(0.195, 0.0, 0.0),
        )
    )
    held = tracker.step(
        observation(
            target_name="cube",
            target=(0.20, 0.0, 0.0),
            grasp=(0.195, 0.0, 0.0),
        )
    )

    assert entered.state is RewardState.TARGET_GRASP_TRANSPORT
    assert held.state is RewardState.TARGET_GRASP_TRANSPORT
    assert entered.components["state3_hold"] == pytest.approx(0.1)
    assert held.components["state3_hold"] == pytest.approx(0.1)
    assert "state3_hold" in entered.events
    assert "state3_hold" in held.events
