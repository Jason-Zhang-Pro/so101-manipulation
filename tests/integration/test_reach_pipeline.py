import json

import numpy as np
import pytest

from so101_capstone.environments import (
    JointStateCache,
    RgbImageCache,
    So101ReachEnvironmentCore,
    TargetCubeStateCache,
    build_policy_observation,
    parse_policy_action,
    select_synchronized_camera_frames,
    update_target_cube_cache_from_json,
)
from so101_capstone.rewards import ReachRewardConfig
from so101_capstone.robot import compute_grasp_point_pose


JOINT_MESSAGE_ORDER = (
    "gripper",
    "wrist_roll",
    "shoulder_pan",
    "elbow_flex",
    "shoulder_lift",
    "wrist_flex",
)


def update_joint_feedback(
    cache,
    *,
    shoulder_pan,
    received_time,
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

    return cache.update(
        JOINT_MESSAGE_ORDER,
        [
            positions_by_name[name]
            for name in JOINT_MESSAGE_ORDER
        ],
        [
            velocities_by_name[name]
            for name in JOINT_MESSAGE_ORDER
        ],
        received_monotonic_s=received_time,
    )


def cube_payload(
    position,
    *,
    name="target_cube_0001",
    entity_id=101,
):
    return json.dumps(
        {
            "frame_id": "world",
            "cube_count": 1,
            "cubes": [
                {
                    "name": name,
                    "entity_id": entity_id,
                    "position": {
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                }
            ],
        }
    )


def rgb_image(value):
    return np.full(
        (240, 320, 3),
        value,
        dtype=np.uint8,
    )


def test_complete_reach_pipeline_uses_measured_feedback():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()
    top_cache = RgbImageCache()
    gripper_cache = RgbImageCache()

    initial_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.0,
        received_time=100.000,
    )
    next_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.2,
        received_time=100.030,
    )

    initial_grasp = compute_grasp_point_pose(
        initial_joints.positions
    ).position
    next_grasp = compute_grasp_point_pose(
        next_joints.positions
    ).position

    motion = next_grasp - initial_grasp
    motion_direction = motion / np.linalg.norm(motion)
    target_position = next_grasp + 0.2 * motion_direction

    cube = update_target_cube_cache_from_json(
        cube_cache,
        cube_payload(target_position),
        received_monotonic_s=100.000,
    )

    environment = So101ReachEnvironmentCore()

    initial_state, reset_info = environment.reset_from_feedback(
        initial_joints,
        cube,
        now_monotonic_s=100.010,
    )

    top_cache.update(
        rgb_image(10),
        received_monotonic_s=100.000,
        source_timestamp_s=500.000,
    )
    gripper_cache.update(
        rgb_image(20),
        received_monotonic_s=100.001,
        source_timestamp_s=500.005,
    )

    initial_frames = select_synchronized_camera_frames(
        top_cache,
        gripper_cache,
        now_monotonic_s=100.010,
    )
    initial_observation = build_policy_observation(
        initial_state,
        initial_frames,
    )

    assert initial_observation.joint_positions.shape == (6,)
    assert initial_observation.joint_velocities.shape == (6,)
    assert initial_observation.top_image.shape == (240, 320, 3)
    assert initial_observation.gripper_image.shape == (240, 320, 3)
    assert reset_info["episode_step"] == 0

    policy_action = parse_policy_action(
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.5]
    )
    np.testing.assert_allclose(
        policy_action.controller_command(),
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.5],
    )

    top_cache.update(
        rgb_image(30),
        received_monotonic_s=100.030,
        source_timestamp_s=500.033,
    )
    gripper_cache.update(
        rgb_image(40),
        received_monotonic_s=100.031,
        source_timestamp_s=500.035,
    )

    next_state, reward, terminated, truncated, step_info = (
        environment.step_from_feedback(
            next_joints,
            cube,
            now_monotonic_s=100.040,
        )
    )

    next_frames = select_synchronized_camera_frames(
        top_cache,
        gripper_cache,
        now_monotonic_s=100.040,
    )
    next_observation = build_policy_observation(
        next_state,
        next_frames,
    )

    assert step_info["progress_m"] > 0.0
    assert step_info["distance_m"] == pytest.approx(0.2)
    assert step_info["cube_state_reused"] is True
    assert step_info["reward_components"]["distance_improvement"] > 0.0
    assert reward == pytest.approx(
        sum(step_info["reward_components"].values())
    )
    assert terminated is False
    assert truncated is False
    assert np.all(next_observation.top_image == 30)
    assert np.all(next_observation.gripper_image == 40)


def test_measured_success_terminates_without_policy_done_signal():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    initial_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.0,
        received_time=200.000,
    )
    next_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.0,
        received_time=200.030,
    )

    grasp_position = compute_grasp_point_pose(
        initial_joints.positions
    ).position

    initial_cube = update_target_cube_cache_from_json(
        cube_cache,
        cube_payload(grasp_position + np.array([0.1, 0.0, 0.0])),
        received_monotonic_s=200.000,
    )

    environment = So101ReachEnvironmentCore(
        reward_config=ReachRewardConfig(
            success_home_hold_steps=1,
            home_joint_positions=tuple(
                initial_joints.positions[name]
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
        goal_region_center_world_m=tuple(grasp_position),
    )

    environment.reset_from_feedback(
        initial_joints,
        initial_cube,
        now_monotonic_s=200.010,
    )

    policy_action = parse_policy_action(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    )
    assert policy_action is not None

    success_cube = update_target_cube_cache_from_json(
        cube_cache,
        cube_payload(grasp_position),
        received_monotonic_s=200.030,
    )

    _, reward, terminated, truncated, info = (
        environment.step_from_feedback(
            next_joints,
            success_cube,
            now_monotonic_s=200.040,
        )
    )

    assert info["success"] is True
    assert info["target_in_goal_region"] is True
    assert info["robot_at_home"] is True
    assert info["reward_components"]["success"] == pytest.approx(50.0)
    assert reward == pytest.approx(
        sum(info["reward_components"].values())
    )
    assert terminated is True
    assert truncated is False


def test_target_removal_does_not_end_policy_episode():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    initial_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.0,
        received_time=300.000,
    )
    next_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.0,
        received_time=300.030,
    )

    grasp_position = compute_grasp_point_pose(
        initial_joints.positions
    ).position
    initial_cube = update_target_cube_cache_from_json(
        cube_cache,
        cube_payload(grasp_position + np.array([0.2, 0.0, 0.0])),
        received_monotonic_s=300.000,
    )

    environment = So101ReachEnvironmentCore()
    environment.reset_from_feedback(
        initial_joints,
        initial_cube,
        now_monotonic_s=300.010,
    )

    _, reward, terminated, truncated, info = (
        environment.step_from_feedback(
            next_joints,
            None,
            now_monotonic_s=300.040,
        )
    )

    assert reward == pytest.approx(0.1)
    assert info["reward_components"]["state1_stationary"] == pytest.approx(0.1)
    assert terminated is False
    assert truncated is False
    assert info["target_active"] is False
    assert info["target_cube_name"] is None
    assert info["target_transition"] == "removed"
    assert "target_removed" in info["reward_events"]


def test_selected_target_can_switch_without_restarting_episode():
    joint_cache = JointStateCache()
    cube_cache = TargetCubeStateCache()

    initial_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.0,
        received_time=400.000,
    )
    next_joints = update_joint_feedback(
        joint_cache,
        shoulder_pan=0.0,
        received_time=400.030,
    )

    grasp_position = compute_grasp_point_pose(
        initial_joints.positions
    ).position
    initial_cube = update_target_cube_cache_from_json(
        cube_cache,
        cube_payload(
            grasp_position + np.array([0.2, 0.0, 0.0]),
            name="target_cube_0001",
            entity_id=101,
        ),
        received_monotonic_s=400.000,
    )

    environment = So101ReachEnvironmentCore()
    environment.reset_from_feedback(
        initial_joints,
        initial_cube,
        now_monotonic_s=400.010,
    )

    switched_cube = update_target_cube_cache_from_json(
        cube_cache,
        cube_payload(
            grasp_position + np.array([0.3, 0.0, 0.0]),
            name="target_cube_0002",
            entity_id=102,
        ),
        received_monotonic_s=400.030,
    )

    _, _, terminated, truncated, info = (
        environment.step_from_feedback(
            next_joints,
            switched_cube,
            now_monotonic_s=400.040,
        )
    )

    assert terminated is False
    assert truncated is False
    assert info["target_active"] is True
    assert info["target_cube_name"] == "target_cube_0002"
    assert info["target_transition"] == "switched"
    assert info["progress_m"] == pytest.approx(0.0)
    assert info["reward_target_age_steps"] == 1
    assert "target_switched" in info["reward_events"]
