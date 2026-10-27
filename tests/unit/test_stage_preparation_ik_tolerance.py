"""Relaxed preparation IK accepts a verified approximation without stalling.

The carry waypoint is copied from a live failure, so the regression does not
depend on transient report files. FK is real; these are not Gazebo tests.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments.stage_preparation import StagePreparationController
from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS
from so101_capstone.rewards import PickPlaceRewardConfig
from so101_capstone.robot.so101_forward_kinematics import (
    ACTIVE_JOINT_NAMES, compute_grasp_point_pose,
)
from so101_capstone.robot.so101_inverse_kinematics import (
    IKConfig, InverseKinematicsError, solve_grasp_point_ik,
)


CARRY_SEED = (
    1.2560831806988089, .7825842314754266, -1.1416860261764168,
    1.639212708854286, .004842393413483218,
)
CARRY_WAYPOINT = (.13258099929570094, -.2941625310529171, .8397643793970044)
CARRY_MEASURED = (.13351660888385788, -.2913488502607737, .8392617195354416)
CARRY_TRANSLATION = (-.0033626918859557264, -.01164210425824086, .002381937810260859)


def pose(joints):
    return compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, joints)))


def relaxed_config(**kwargs):
    return IKConfig(position_tolerance_m=.01, preferred_position_tolerance_m=.0005,
                    multistart=False, **kwargs)


def assert_verified(result, waypoint, desired_axis, config):
    actual = pose(result.joint_positions_rad)
    distance = np.linalg.norm(actual.position - waypoint)
    angle = math.acos(float(np.clip(actual.transform[:3, 2] @ desired_axis, -1., 1.)))
    assert distance <= config.position_tolerance_m
    assert angle <= config.axis_tolerance_rad
    assert result.position_error_m == pytest.approx(distance)
    assert result.axis_error_rad == pytest.approx(angle)
    for q, limit in zip(result.joint_positions_rad, SIM_JOINT_CALIBRATIONS):
        assert limit.gazebo_lower_rad <= q <= limit.gazebo_upper_rad
    return actual


def test_live_carry_failure_accepts_one_cm_after_attempting_preferred_precision():
    strict = IKConfig(position_tolerance_m=.0005, multistart=False)
    with pytest.raises(InverseKinematicsError, match="best position error=0.000605 m"):
        solve_grasp_point_ik(CARRY_WAYPOINT, CARRY_SEED, config=strict)
    config = relaxed_config()
    result = solve_grasp_point_ik(CARRY_WAYPOINT, CARRY_SEED, config=config)
    actual = assert_verified(result, CARRY_WAYPOINT, np.array((0., 0., -1.)), config)
    # The seed itself is already only 2 mm away. Returning it immediately
    # would satisfy 1 cm but leave each preparation control tick stationary.
    assert .0005 < result.position_error_m < .001
    assert np.linalg.norm(actual.position - pose(CARRY_SEED).position) > .001
    assert result.iterations > 1


def test_preparation_uses_relaxed_solver_on_the_failed_live_carry_step():
    controller = StagePreparationController(PickPlaceRewardConfig())
    controller.command = np.array((*CARRY_SEED, -.0015730476396023998))
    controller.snapshot = SimpleNamespace(measurements=SimpleNamespace(
        grasp_position_world_m=CARRY_MEASURED))
    commands = []
    controller._tick = lambda command, **kwargs: commands.append(np.asarray(command))
    controller._move_step(CARRY_TRANSLATION, np.array((0., 0., -1.)), require_grasp=True)
    assert len(commands) == 1
    actual = pose(commands[0][:5])
    assert np.linalg.norm(actual.position - CARRY_WAYPOINT) < .001
    assert np.linalg.norm(actual.position - pose(CARRY_SEED).position) > .001
    assert commands[0][-1] == controller.command[-1]


def test_two_mm_waypoint_still_moves_instead_of_accepting_seed_inside_one_cm():
    seed = (0., -.8, .7, 1.1, -.3)
    initial = pose(seed)
    waypoint = initial.position + np.array((.002, 0., 0.))
    config = relaxed_config()
    result = solve_grasp_point_ik(waypoint, seed,
        desired_axis_world=initial.transform[:3, 2], config=config)
    actual = assert_verified(result, waypoint, initial.transform[:3, 2], config)
    assert result.position_error_m <= config.preferred_position_tolerance_m
    assert np.linalg.norm(actual.position - initial.position) >= .0015


def test_no_preferred_precision_preserves_previous_first_acceptable_solution():
    seed = (0., -.8, .7, 1.1, -.3)
    initial = pose(seed)
    result = solve_grasp_point_ik(initial.position + np.array((.002, 0., 0.)), seed,
        desired_axis_world=initial.transform[:3, 2],
        config=IKConfig(position_tolerance_m=.01, multistart=False))
    assert result.joint_positions_rad == seed
    assert result.iterations == 1


def test_one_cm_fallback_does_not_accept_position_when_axis_is_invalid():
    seed = (0., -.8, .7, 1.1, -.3)
    initial = pose(seed)
    with pytest.raises(InverseKinematicsError):
        solve_grasp_point_ik(initial.position, seed,
            desired_axis_world=-initial.transform[:3, 2],
            config=relaxed_config(max_iterations=1))


def test_one_cm_fallback_does_not_accept_position_outside_hard_tolerance():
    with pytest.raises(InverseKinematicsError):
        solve_grasp_point_ik((10., 0., 0.), CARRY_SEED,
            config=relaxed_config(max_iterations=15))


def test_one_cm_fallback_does_not_relax_calibrated_joint_limits():
    seed = np.asarray(CARRY_SEED).copy()
    seed[0] = SIM_JOINT_CALIBRATIONS[0].gazebo_upper_rad + .001
    with pytest.raises(ValueError, match="exceeds calibrated joint limits"):
        solve_grasp_point_ik(CARRY_WAYPOINT, seed, config=relaxed_config())


@pytest.mark.parametrize("preferred", [True, 0., -.0001, math.inf, math.nan, .011])
def test_preferred_precision_must_be_positive_finite_and_within_hard_tolerance(preferred):
    with pytest.raises(ValueError, match="preferred_position_tolerance_m"):
        IKConfig(position_tolerance_m=.01, preferred_position_tolerance_m=preferred)


def test_preferred_precision_may_equal_hard_tolerance():
    assert IKConfig(position_tolerance_m=.01,
                    preferred_position_tolerance_m=.01).preferred_position_tolerance_m == .01
