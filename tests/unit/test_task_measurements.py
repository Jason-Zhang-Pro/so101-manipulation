from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments.contact_state import FingerContactCache
from so101_capstone.environments.ros_message_adapter import update_joint_cache_from_message
from so101_capstone.environments.state_cache import JointStateCache, JointStateSnapshot, MEASURED_JOINT_NAMES, TargetCubeSnapshot, TargetCubeStateCache
from so101_capstone.environments.task_measurements import TaskMeasurementBuilder, TaskMeasurementError, interpolate_joints, quaternion_angular_velocity
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig

HOME = PickPlaceRewardConfig().home_joint_positions_rad
GOAL = (.078, -.300, .7515)


def joint(t, sequence=1, offset=0):
    return JointStateSnapshot(dict(zip(MEASURED_JOINT_NAMES, np.asarray(HOME)+offset)),
                              dict.fromkeys(MEASURED_JOINT_NAMES, 0.0), sequence, 0.0, t)


def cube(t, sequence=1, x=.2, name="target_cube_1"):
    return TargetCubeSnapshot(name, np.array([x, 0, .765]), np.array([0, 0, 0, 1.0]), sequence, 0.0, sequence, t)


def contacts(*times):
    result = FingerContactCache()
    for t in times:
        for side in ("left", "right"):
            result.update(side, sim_time_s=t)
    return result


def test_joint_header_and_bounded_history_preserve_source_time():
    cache = JointStateCache()
    message = SimpleNamespace(name=MEASURED_JOINT_NAMES, position=HOME, velocity=[0]*6,
                              header=SimpleNamespace(stamp=SimpleNamespace(sec=4, nanosec=10_000_000)))
    snapshot = update_joint_cache_from_message(cache, message, received_monotonic_s=9)
    assert snapshot.source_sim_time_s == pytest.approx(4.01)
    assert cache.history()[0].source_sim_time_s == pytest.approx(4.01)
    assert snapshot.received_monotonic_s == 9


def test_target_republication_does_not_create_physical_history_sample():
    cache = TargetCubeStateCache()
    for seq in (1, 2, 3):
        cache.update("target_cube_1", [0, 0, .765], [0, 0, 0, 1], source_sequence=seq, source_sim_time_s=2)
    assert len(cache.history()) == 1
    cache.update("target_cube_2", [0, 0, .765], [0, 0, 0, 1], source_sim_time_s=3)
    assert len(cache.history()) == 1
    assert cache.history()[0].name == "target_cube_2"


def test_joint_interpolation_uses_brackets_without_extrapolation():
    q, _ = interpolate_joints([joint(0), joint(.1, offset=.2)], .05, .12)
    np.testing.assert_allclose(q, np.asarray(HOME)+.1)
    assert interpolate_joints([joint(0)], .01, .12) is None
    assert interpolate_joints([joint(0), joint(1)], .5, .12) is None


def test_quaternion_sign_change_is_not_rotation_and_short_arc_is_used():
    np.testing.assert_allclose(quaternion_angular_velocity([0, 0, 0, 1], [0, 0, 0, -1], .1), 0)
    actual = quaternion_angular_velocity([0, 0, 0, 1], [0, 0, np.sin(.05), np.cos(.05)], .1)
    np.testing.assert_allclose(actual, [0, 0, 1], atol=1e-9)


def test_builder_aligns_fk_to_object_source_and_estimates_actual_dt():
    builder = TaskMeasurementBuilder()
    streams = contacts(0, .05, .1)
    builder.update(joint(0), cube(0), control_sim_time_s=0, contact_cache=streams, goal_center_world_m=GOAL)
    measured = builder.update(joint(.1, 2, offset=.2), cube(.05, 2, x=.21),
        control_sim_time_s=.1, joint_history=[joint(0), joint(.1, 2, offset=.2)],
        contact_cache=streams, goal_center_world_m=GOAL)
    assert measured.evidence_valid and measured.target_motion_valid
    np.testing.assert_allclose(measured.joint_positions_rad, np.asarray(HOME)+.1)
    np.testing.assert_allclose(measured.target_linear_velocity_m_s, [.2, 0, 0])
    assert measured.evidence_sim_time_s == .05


def test_first_pose_motion_unknown_and_repeat_does_not_advance_evidence():
    builder = TaskMeasurementBuilder()
    streams = contacts(0, .05)
    first = builder.update(joint(0), cube(0), control_sim_time_s=0, contact_cache=streams, goal_center_world_m=GOAL)
    repeated = builder.update(joint(.05, 2), cube(0, 2), control_sim_time_s=.05,
        joint_history=[joint(0), joint(.05, 2)], contact_cache=streams, goal_center_world_m=GOAL)
    assert not first.target_motion_valid and not repeated.target_motion_valid
    assert repeated.evidence_sim_time_s == first.evidence_sim_time_s


def test_formal_mode_never_silently_falls_back_without_contact_streams():
    builder = TaskMeasurementBuilder()
    first = builder.update(joint(0), None, control_sim_time_s=0, goal_center_world_m=GOAL)
    assert not first.evidence_valid and not first.contact_valid
    with pytest.raises(TaskMeasurementError, match="evidence unavailable"):
        builder.update(joint(.31, 2), None, control_sim_time_s=.31, goal_center_world_m=GOAL)


def test_missing_joint_stamp_is_rejected_and_clock_rollback_requires_reset():
    builder = TaskMeasurementBuilder()
    with pytest.raises(TaskMeasurementError, match="JointState.header"):
        builder.update(replace(joint(0), source_sim_time_s=None), None, control_sim_time_s=0, goal_center_world_m=GOAL)
    builder.reset()
    builder.update(joint(1), None, control_sim_time_s=1, contact_cache=contacts(1), goal_center_world_m=GOAL)
    with pytest.raises(TaskMeasurementError, match="backwards"):
        builder.update(joint(0, 2), None, control_sim_time_s=0, goal_center_world_m=GOAL)


def test_spawn_with_lagging_pose_does_not_move_evidence_clock_backwards():
    builder = TaskMeasurementBuilder()
    streams = contacts(.1, .15, .2)
    builder.update(joint(.1), None, control_sim_time_s=.1, contact_cache=streams, goal_center_world_m=GOAL)
    result = builder.update(joint(.2, 2), cube(.05), control_sim_time_s=.2,
        joint_history=[joint(.1), joint(.2, 2)], contact_cache=streams, goal_center_world_m=GOAL,
        camera_source_times_s=(0, .1))
    assert result.evidence_sim_time_s >= .1
    assert not result.target_measurement_valid
    assert builder.last_info["camera_observed_target"] == [False, True]


def test_repeated_sparse_pose_arrivals_cannot_reset_motion_unavailable_watchdog():
    builder = TaskMeasurementBuilder()
    streams = contacts(*np.arange(0, .501, 1/30))
    poses, joints = [], []
    with pytest.raises(TaskMeasurementError, match="evidence unavailable"):
        for step in range(16):
            t = step / 30
            joints.append(joint(t, step+1))
            if step % 5 == 0:
                poses.append(cube(t, step+1))
            result = builder.update(joints[-1], poses[-1], control_sim_time_s=t,
                joint_history=joints, cube_history=poses, contact_cache=streams,
                goal_center_world_m=GOAL)
            assert not result.target_motion_valid
