import numpy as np
import pytest

from so101_capstone.environments import (
    JointStateCache,
    ReachEnvironmentConfig,
    ReachEnvironmentStateError,
    So101ReachEnvironmentCore,
    TargetCubeStateCache,
)
from so101_capstone.rewards import ReachRewardConfig
from so101_capstone.robot import compute_grasp_point_pose


JOINT_NAMES = [
    "gripper",
    "wrist_roll",
    "shoulder_pan",
    "elbow_flex",
    "shoulder_lift",
    "wrist_flex",
]


def joint_snapshot(
    cache,
    *,
    shoulder_pan=0.0,
    velocities=True,
    received_time=10.0,
):
    positions_by_name = {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.5,
    }
    velocities_by_name = {
        "shoulder_pan": 0.1,
        "shoulder_lift": 0.2,
        "elbow_flex": 0.3,
        "wrist_flex": 0.4,
        "wrist_roll": 0.5,
        "gripper": 0.6,
    }

    positions = [
        positions_by_name[name]
        for name in JOINT_NAMES
    ]

    velocity_values = (
        [
            velocities_by_name[name]
            for name in JOINT_NAMES
        ]
        if velocities
        else []
    )

    return cache.update(
        JOINT_NAMES,
        positions,
        velocity_values,
        received_monotonic_s=received_time,
    )


def cube_snapshot(
    cache,
    position,
    *,
    name="target_cube_0001",
    source_sequence=None,
    source_sim_time_s=None,
    received_time=10.0,
):
    return cache.update(
        name,
        position,
        [0.0, 0.0, 0.0, 1.0],
        source_sequence=source_sequence,
        source_sim_time_s=source_sim_time_s,
        received_monotonic_s=received_time,
    )


def test_reset_builds_ordered_policy_and_privileged_state():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    joints = joint_snapshot(joint_cache)
    grasp = compute_grasp_point_pose(
        joints.positions
    ).position
    cube = cube_snapshot(
        cube_cache,
        grasp + np.array([0.2, 0.0, 0.0]),
    )

    env = So101ReachEnvironmentCore()
    observation, info = env.reset_from_feedback(
        joints,
        cube,
        now_monotonic_s=10.0,
    )

    np.testing.assert_allclose(
        observation.joint_positions,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    )
    np.testing.assert_allclose(
        observation.joint_velocities,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )

    assert info["episode_step"] == 0
    assert info["control_hz"] == 30.0
    assert info["control_period_s"] == pytest.approx(
        1.0 / 30.0
    )
    assert info["privileged_state"].distance_m == pytest.approx(
        0.2
    )


def test_step_uses_new_feedback_and_rewards_progress():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(
        joint_cache,
        shoulder_pan=0.0,
    )
    second_joints = joint_snapshot(
        joint_cache,
        shoulder_pan=0.2,
    )

    first_grasp = compute_grasp_point_pose(
        first_joints.positions
    ).position
    second_grasp = compute_grasp_point_pose(
        second_joints.positions
    ).position

    direction = second_grasp - first_grasp
    direction /= np.linalg.norm(direction)
    target_position = second_grasp + 0.2 * direction

    first_cube = cube_snapshot(
        cube_cache,
        target_position,
    )
    second_cube = cube_snapshot(
        cube_cache,
        target_position,
    )

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        first_joints,
        first_cube,
        now_monotonic_s=10.0,
    )

    _, _, terminated, truncated, info = (
        env.step_from_feedback(
            second_joints,
            second_cube,
            now_monotonic_s=10.0,
        )
    )

    assert info["progress_m"] > 0.0
    assert info["reward_components"]["distance_improvement"] > 0.0
    assert terminated is False
    assert truncated is False
    assert info["cube_state_reused"] is False


def test_success_terminates_after_required_home_hold():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)
    third_joints = joint_snapshot(joint_cache)

    grasp = compute_grasp_point_pose(
        first_joints.positions
    ).position

    initial_cube = cube_snapshot(
        cube_cache,
        grasp + np.array([0.1, 0.0, 0.0]),
    )
    success_cube_one = cube_snapshot(
        cube_cache,
        grasp,
    )
    success_cube_two = cube_snapshot(
        cube_cache,
        grasp,
    )

    env = So101ReachEnvironmentCore(
        reward_config=ReachRewardConfig(
            success_home_hold_steps=2,
            home_joint_positions=tuple(
                first_joints.positions[name]
                for name in (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                )
            ),
        ),
        goal_region_center_world_m=tuple(grasp),
    )

    env.reset_from_feedback(
        first_joints,
        initial_cube,
        now_monotonic_s=10.0,
    )

    _, _, first_terminated, first_truncated, first_info = (
        env.step_from_feedback(
            second_joints,
            success_cube_one,
            now_monotonic_s=10.0,
        )
    )

    assert first_terminated is False
    assert first_truncated is False
    assert first_info["success"] is False
    assert first_info["success_home_hold_count"] == 1
    assert first_info["success_home_hold_required_steps"] == 2
    assert first_info["success_home_hold_remaining_steps"] == 1
    assert env.episode_active is True

    _, reward, terminated, truncated, info = (
        env.step_from_feedback(
            third_joints,
            success_cube_two,
            now_monotonic_s=10.0,
        )
    )

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["target_in_goal_region"] is True
    assert info["robot_at_home"] is True
    assert info["success_home_hold_count"] == 2
    assert info["success_home_hold_remaining_steps"] == 0
    assert info["reward_components"]["success"] == pytest.approx(50.0)
    assert reward == pytest.approx(
        sum(info["reward_components"].values())
    )
    assert env.episode_active is False


def test_step_limit_truncates_episode():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    grasp = compute_grasp_point_pose(
        first_joints.positions
    ).position
    cube = cube_snapshot(
        cube_cache,
        grasp + np.array([0.3, 0.0, 0.0]),
    )

    env = So101ReachEnvironmentCore(
        config=ReachEnvironmentConfig(
            max_episode_steps=1,
        )
    )

    env.reset_from_feedback(
        first_joints,
        cube,
        now_monotonic_s=10.0,
    )

    _, _, terminated, truncated, _ = (
        env.step_from_feedback(
            second_joints,
            cube,
            now_monotonic_s=10.0,
        )
    )

    assert terminated is False
    assert truncated is True
    assert env.episode_active is False


def test_stale_feedback_is_rejected():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    joints = joint_snapshot(
        joint_cache,
        received_time=1.0,
    )
    cube = cube_snapshot(
        cube_cache,
        [0.2, 0.0, 0.765],
        received_time=1.0,
    )

    env = So101ReachEnvironmentCore(
        config=ReachEnvironmentConfig(
            max_joint_state_age_s=0.1,
            max_cube_state_age_s=0.1,
        )
    )

    with pytest.raises(
        ReachEnvironmentStateError,
        match="stale",
    ):
        env.reset_from_feedback(
            joints,
            cube,
            now_monotonic_s=2.0,
        )


def test_target_cube_can_switch_during_episode():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    first_cube = cube_snapshot(
        cube_cache,
        [0.25, 0.0, 0.765],
        name="target_cube_0001",
    )
    second_cube = cube_snapshot(
        cube_cache,
        [0.25, 0.0, 0.765],
        name="target_cube_0002",
    )

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        first_joints,
        first_cube,
        now_monotonic_s=10.0,
    )

    _, _, terminated, truncated, info = (
        env.step_from_feedback(
            second_joints,
            second_cube,
            now_monotonic_s=10.0,
        )
    )

    assert terminated is False
    assert truncated is False
    assert env.episode_active is True
    assert env.target_active is True
    assert info["target_transition"] == "switched"
    assert info["target_cube_name"] == "target_cube_0002"
    assert info["progress_m"] == pytest.approx(0.0)
    assert info["reward_target_age_steps"] == 1
    assert "target_switched" in info["reward_events"]


def test_each_step_requires_new_joint_feedback():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    joints = joint_snapshot(joint_cache)
    cube = cube_snapshot(
        cube_cache,
        [0.25, 0.0, 0.765],
    )

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        joints,
        cube,
        now_monotonic_s=10.0,
    )

    with pytest.raises(
        ReachEnvironmentStateError,
        match="newer",
    ):
        env.step_from_feedback(
            joints,
            cube,
            now_monotonic_s=10.0,
        )


def test_policy_observation_requires_joint_velocities():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    joints = joint_snapshot(
        joint_cache,
        velocities=False,
    )
    cube = cube_snapshot(
        cube_cache,
        [0.25, 0.0, 0.765],
    )

    env = So101ReachEnvironmentCore()

    with pytest.raises(
        ReachEnvironmentStateError,
        match="velocities",
    ):
        env.reset_from_feedback(
            joints,
            cube,
            now_monotonic_s=10.0,
        )



def test_cube_source_sequence_controls_reuse():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    target = [0.30, 0.0, 0.765]

    first_cube = cube_snapshot(
        cube_cache,
        target,
        source_sequence=100,
        source_sim_time_s=5.0,
    )
    second_cube = cube_snapshot(
        cube_cache,
        target,
        source_sequence=100,
        source_sim_time_s=5.0,
    )

    assert second_cube.sequence != first_cube.sequence

    env = So101ReachEnvironmentCore()

    _, reset_info = env.reset_from_feedback(
        first_joints,
        first_cube,
        now_monotonic_s=10.0,
    )

    _, _, _, _, step_info = env.step_from_feedback(
        second_joints,
        second_cube,
        now_monotonic_s=10.0,
    )

    assert reset_info["cube_source_sequence"] == 100
    assert reset_info["cube_source_sim_time_s"] == pytest.approx(
        5.0
    )
    assert step_info["cube_sequence"] == 2
    assert step_info["cube_source_sequence"] == 100
    assert step_info["cube_state_reused"] is True


def test_cube_source_sequence_cannot_move_backwards():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    first_cube = cube_snapshot(
        cube_cache,
        [0.30, 0.0, 0.765],
        source_sequence=10,
        source_sim_time_s=1.0,
    )
    second_cube = cube_snapshot(
        cube_cache,
        [0.30, 0.0, 0.765],
        source_sequence=9,
        source_sim_time_s=1.033,
    )

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        first_joints,
        first_cube,
        now_monotonic_s=10.0,
    )

    with pytest.raises(
        ReachEnvironmentStateError,
        match="source sequence moved backwards",
    ):
        env.step_from_feedback(
            second_joints,
            second_cube,
            now_monotonic_s=10.0,
        )


def test_cube_simulation_time_cannot_move_backwards():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    first_cube = cube_snapshot(
        cube_cache,
        [0.30, 0.0, 0.765],
        source_sequence=10,
        source_sim_time_s=2.0,
    )
    second_cube = cube_snapshot(
        cube_cache,
        [0.30, 0.0, 0.765],
        source_sequence=11,
        source_sim_time_s=1.9,
    )

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        first_joints,
        first_cube,
        now_monotonic_s=10.0,
    )

    with pytest.raises(
        ReachEnvironmentStateError,
        match="simulation time moved backwards",
    ):
        env.step_from_feedback(
            second_joints,
            second_cube,
            now_monotonic_s=10.0,
        )


def test_reset_can_start_without_target_cube():
    joint_cache = JointStateCache()
    joints = joint_snapshot(joint_cache)

    env = So101ReachEnvironmentCore()

    observation, info = env.reset_from_feedback(
        joints,
        None,
        now_monotonic_s=10.0,
    )

    assert observation.joint_positions.shape == (6,)
    assert observation.joint_velocities.shape == (6,)
    assert env.episode_active is True
    assert env.target_active is False

    assert info["target_active"] is False
    assert info["target_cube_name"] is None
    assert info["cube_sequence"] is None
    assert info["cube_source_sequence"] is None
    assert info["cube_source_sim_time_s"] is None
    assert info["cube_state_age_s"] is None
    assert info["privileged_state"] is None


def test_policy_step_without_cube_uses_state1_stationary_reward():
    joint_cache = JointStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        first_joints,
        None,
        now_monotonic_s=10.0,
    )

    (
        _observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step_from_feedback(
        second_joints,
        None,
        now_monotonic_s=10.0,
    )

    assert reward == pytest.approx(
        sum(info["reward_components"].values())
    )
    assert terminated is False
    assert truncated is False
    assert info["target_active"] is False
    assert info["distance_m"] is None
    assert info["progress_m"] is None
    assert info["success"] is False
    assert info["reward_components"]["state1_home_error"] == pytest.approx(
        reward
    )
    assert info["reward_state_name"] == "TARGET_ABSENT"


def test_timeout_can_end_episode_before_cube_spawns():
    joint_cache = JointStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    env = So101ReachEnvironmentCore(
        config=ReachEnvironmentConfig(
            max_episode_steps=1,
        )
    )

    env.reset_from_feedback(
        first_joints,
        None,
        now_monotonic_s=10.0,
    )

    _, reward, terminated, truncated, info = (
        env.step_from_feedback(
            second_joints,
            None,
            now_monotonic_s=10.0,
        )
    )

    assert reward == pytest.approx(
        sum(info["reward_components"].values())
    )
    assert terminated is False
    assert truncated is True
    assert info["episode_step"] == 1
    assert info["reward_components"]["state1_home_error"] == pytest.approx(
        reward
    )
    assert env.episode_active is False


def test_first_cube_state_activates_target_without_progress_jump():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    grasp = compute_grasp_point_pose(
        second_joints.positions
    ).position

    cube = cube_snapshot(
        cube_cache,
        grasp + np.array([0.2, 0.0, 0.0]),
        source_sequence=100,
        source_sim_time_s=5.0,
    )

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        first_joints,
        None,
        now_monotonic_s=10.0,
    )

    _, reward, terminated, truncated, info = (
        env.step_from_feedback(
            second_joints,
            cube,
            now_monotonic_s=10.0,
        )
    )

    assert env.target_active is True
    assert info["target_active"] is True
    assert info["target_cube_name"] == "target_cube_0001"
    assert info["target_transition"] == "activated"
    assert info["progress_m"] == pytest.approx(0.0)
    assert info["distance_m"] == pytest.approx(0.2)
    assert info["reward_target_age_steps"] == 1
    assert info["cube_state_reused"] is False
    assert info["reward_components"]["distance_improvement"] == pytest.approx(0.0)

    assert reward == pytest.approx(
        sum(info["reward_components"].values())
    )
    assert terminated is False
    assert truncated is False


def test_active_target_can_disappear_mid_episode():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    first_joints = joint_snapshot(joint_cache)
    second_joints = joint_snapshot(joint_cache)

    cube = cube_snapshot(
        cube_cache,
        [0.25, 0.0, 0.765],
    )

    env = So101ReachEnvironmentCore()

    env.reset_from_feedback(
        first_joints,
        cube,
        now_monotonic_s=10.0,
    )

    _, reward, terminated, truncated, info = (
        env.step_from_feedback(
            second_joints,
            None,
            now_monotonic_s=10.0,
        )
    )

    assert reward == pytest.approx(
        sum(info["reward_components"].values())
    )
    assert terminated is False
    assert truncated is False
    assert env.episode_active is True
    assert env.target_active is False
    assert info["target_transition"] == "removed"
    assert info["target_active"] is False
    assert info["target_cube_name"] is None
    assert info["distance_m"] is None
    assert info["progress_m"] is None
    assert info["reward_components"]["state1_home_error"] == pytest.approx(
        reward
    )
    assert "target_removed" in info["reward_events"]


def test_reset_with_target_in_goal_uses_tracker_state2_baseline():
    env = So101ReachEnvironmentCore()
    joints = joint_snapshot(JointStateCache())
    cube = cube_snapshot(
        TargetCubeStateCache(), env.goal_region_center_world_m
    )
    _, info = env.reset_from_feedback(joints, cube, now_monotonic_s=10.0)
    assert info["reward_state"] == 2
    # Goal transition belongs to the first actual reward step, not reset.
    next_joints = joint_snapshot(JointStateCache(), received_time=10.0)
    from dataclasses import replace
    next_joints = replace(next_joints, sequence=joints.sequence + 1)
    _, _, _, _, info = env.step_from_feedback(
        next_joints, cube, now_monotonic_s=10.0
    )
    assert info["reward_state"] == 4


@pytest.mark.parametrize("step_end_sim_time_s, accepted", [
    (None, False),  # Standalone callers retain the original wall-age guard.
    (1.033, True),  # A long pause did not age the physical observation.
    (1.301, False),  # Advancing simulation really did make the pose stale.
])
def test_source_age_is_used_only_with_verified_simulation_clock(
    step_end_sim_time_s, accepted
):
    env = So101ReachEnvironmentCore()
    joint_cache = JointStateCache()
    first = joint_snapshot(joint_cache, received_time=10.0)
    cube = cube_snapshot(
        TargetCubeStateCache(), [0.3, 0.0, 0.765],
        source_sequence=1, source_sim_time_s=1.0, received_time=10.0,
    )
    env.reset_from_feedback(first, cube, now_monotonic_s=10.0)
    second = joint_snapshot(joint_cache, received_time=1010.0)
    kwargs = dict(now_monotonic_s=1010.0, step_end_sim_time_s=step_end_sim_time_s)
    if not accepted:
        with pytest.raises(ReachEnvironmentStateError, match="Target cube feedback is stale"):
            env.step_from_feedback(second, cube, **kwargs)
    else:
        _, _, _, _, info = env.step_from_feedback(second, cube, **kwargs)
        assert cube.received_monotonic_s == 10.0
        assert info["cube_state_age_s"] == pytest.approx(0.033)
        assert info["cube_received_monotonic_age_s"] == pytest.approx(1000.0)


def test_simulation_time_freshness_requires_a_pose_source_stamp():
    env = So101ReachEnvironmentCore()
    joint_cache = JointStateCache()
    first = joint_snapshot(joint_cache)
    cube = cube_snapshot(TargetCubeStateCache(), [0.3, 0.0, 0.765])
    env.reset_from_feedback(first, cube, now_monotonic_s=10.0)
    second = joint_snapshot(joint_cache)
    with pytest.raises(ReachEnvironmentStateError, match="source timestamp is required"):
        env.step_from_feedback(
            second, cube, now_monotonic_s=10.0, step_end_sim_time_s=1.033
        )
