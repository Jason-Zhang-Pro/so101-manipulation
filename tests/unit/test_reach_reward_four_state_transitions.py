# Focused tests for the four-state SO-101 reach reward machine.

import numpy as np

from so101_capstone.rewards.reach_reward import (
    ReachRewardConfig,
    ReachRewardInput,
    RewardState,
    StatefulReachReward,
)


IDENTITY_ROTATION = np.eye(3)


def _observation(
    *,
    joints,
    target_name="target_cube_0001",
    target=(0.20, 0.0, 0.765),
    compare_point=(0.0, 0.0, 0.765),
    goal=(0.0, 0.20, 0.0),
):
    return ReachRewardInput(
        joint_positions_rad=joints,
        grasp_position_world_m=compare_point,
        grasp_rotation_world=IDENTITY_ROTATION,
        goal_region_center_world_m=goal,
        target_name=target_name,
        target_position_world_m=(
            None if target_name is None else target
        ),
    )


def test_reward_state_values_and_legacy_aliases():
    assert RewardState.TARGET_ABSENT.value == 1
    assert RewardState.TARGET_APPROACH.value == 2
    assert RewardState.TARGET_GRASP_TRANSPORT.value == 3
    assert RewardState.TARGET_RELEASE_RETURN_HOME.value == 4

    assert (
        RewardState.TARGET_OUTSIDE_GOAL
        is RewardState.TARGET_APPROACH
    )
    assert (
        RewardState.TARGET_INSIDE_GOAL
        is RewardState.TARGET_RELEASE_RETURN_HOME
    )


def test_four_state_transition_contract():
    config = ReachRewardConfig()
    joints = tuple(config.home_joint_positions)
    tracker = StatefulReachReward(config)
    tracker.reset(joint_positions_rad=joints)

    # Target refresh always enters State 2.
    state2 = tracker.step(
        _observation(
            joints=joints,
            compare_point=(0.20, 0.0, 0.765),
        )
    )
    assert state2.state is RewardState.TARGET_APPROACH

    # Same target + compare point in range -> State 3.
    state3 = tracker.step(
        _observation(
            joints=joints,
            compare_point=(0.20, 0.0, 0.765),
        )
    )
    assert state3.state is RewardState.TARGET_GRASP_TRANSPORT

    # Compare point leaves range -> State 2.
    back_to_state2 = tracker.step(
        _observation(
            joints=joints,
            compare_point=(0.15, 0.0, 0.765),
        )
    )
    assert back_to_state2.state is RewardState.TARGET_APPROACH

    # Re-enter compare-point range -> State 3.
    state3_again = tracker.step(
        _observation(
            joints=joints,
            compare_point=(0.20, 0.0, 0.765),
        )
    )
    assert state3_again.state is RewardState.TARGET_GRASP_TRANSPORT

    # Cube enters goal -> State 4.
    state4 = tracker.step(
        _observation(
            joints=joints,
            target=(0.0, 0.20, 0.765),
            compare_point=(0.0, 0.20, 0.765),
        )
    )
    assert (
        state4.state
        is RewardState.TARGET_RELEASE_RETURN_HOME
    )

    # Cube exits goal -> State 2 even if compare point is still in range.
    state2_after_goal_exit = tracker.step(
        _observation(
            joints=joints,
            target=(0.20, 0.0, 0.765),
            compare_point=(0.20, 0.0, 0.765),
        )
    )
    assert state2_after_goal_exit.state is RewardState.TARGET_APPROACH

    # Staying in compare range on the next step -> State 3.
    state3_after_goal_exit = tracker.step(
        _observation(
            joints=joints,
            target=(0.20, 0.0, 0.765),
            compare_point=(0.20, 0.0, 0.765),
        )
    )
    assert (
        state3_after_goal_exit.state
        is RewardState.TARGET_GRASP_TRANSPORT
    )

    # Target disappearance from any phase -> State 1.
    state1 = tracker.step(
        _observation(
            joints=joints,
            target_name=None,
        )
    )
    assert state1.state is RewardState.TARGET_ABSENT


def test_target_switch_always_restarts_at_state2():
    config = ReachRewardConfig()
    joints = tuple(config.home_joint_positions)
    tracker = StatefulReachReward(config)
    tracker.reset(joint_positions_rad=joints)

    tracker.step(_observation(joints=joints))
    tracker.step(
        _observation(
            joints=joints,
            compare_point=(0.20, 0.0, 0.765),
        )
    )

    switched = tracker.step(
        _observation(
            joints=joints,
            target_name="target_cube_0002",
            target=(0.20, 0.0, 0.765),
            compare_point=(0.20, 0.0, 0.765),
        )
    )

    assert switched.state is RewardState.TARGET_APPROACH
    assert tracker.active_target_name == "target_cube_0002"
