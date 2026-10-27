import math

import numpy as np
import pytest

from so101_capstone.robot.so101_forward_kinematics import ACTIVE_JOINT_NAMES, compute_grasp_point_pose
from so101_capstone.robot.so101_inverse_kinematics import IKConfig, InverseKinematicsError, solve_grasp_point_ik
from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig


@pytest.mark.parametrize("q", [(0, -.8, .7, 1.1, -.3), (.4, -1, 1.2, .9, -1.4),
                              (-.7, .1, -.3, .9, .8), (.3, -.5, .8, -.6, 1.1)])
def test_bounded_ik_roundtrip_uses_existing_fk_and_recovers_perturbed_seed(q):
    pose = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q)))
    seed = np.asarray(q) + np.array((.035, -.04, .03, -.03, .04))
    config = IKConfig(position_tolerance_m=1e-4, axis_tolerance_rad=.005, multistart=False)
    result = solve_grasp_point_ik(pose.position, seed, desired_axis_world=pose.transform[:3, 2], config=config)
    recovered = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, result.joint_positions_rad)))
    assert np.linalg.norm(recovered.position - pose.position) <= config.position_tolerance_m
    assert result.axis_error_rad <= config.axis_tolerance_rad
    for value, limits in zip(result.joint_positions_rad, SIM_JOINT_CALIBRATIONS):
        assert limits.gazebo_lower_rad <= value <= limits.gazebo_upper_rad
    assert result == solve_grasp_point_ik(pose.position, seed, desired_axis_world=pose.transform[:3, 2], config=config)


def test_unreachable_goal_is_an_error_not_best_effort_success():
    with pytest.raises(InverseKinematicsError, match="No verified bounded"):
        solve_grasp_point_ik((10, 0, 0), (0,) * 5, config=IKConfig(max_iterations=15, multistart=False))


@pytest.mark.parametrize("position", [(.078, -.300, .833), (.078, -.300, .813),
                                     (.078, -.300, .779), (.18, 0, .771),
                                     (.15, .08, .771), (.20, -.08, .771)])
def test_real_goal_center_and_representative_source_samples_have_verified_solutions(position):
    # The model's lab_table goal center is (.078, -.300, .7515), with
    # its support surface at .753. This is an IK check, not live contact proof.
    c = IKConfig()
    result = solve_grasp_point_ik(position, PickPlaceRewardConfig().home_joint_positions_rad, config=c)
    pose = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, result.joint_positions_rad)))
    assert np.linalg.norm(pose.position - position) <= c.position_tolerance_m
    assert result.axis_error_rad <= c.axis_tolerance_rad


@pytest.mark.parametrize("kwargs", [dict(position_world_m=(0, 0, math.nan)),
                                   dict(seed_joint_positions_rad=(10,) * 5),
                                   dict(desired_axis_world=(0, 0, 0)),
                                   dict(seed_joint_positions_rad=(0,) * 4)])
def test_invalid_request_cannot_issue_a_solution(kwargs):
    args = dict(position_world_m=(.2, 0, .85), seed_joint_positions_rad=(0,) * 5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        solve_grasp_point_ik(**args)


@pytest.mark.parametrize("kwargs", [dict(max_iterations=True), dict(max_iterations=0),
                                   dict(axis_tolerance_rad=math.pi), dict(multistart=1),
                                   dict(position_tolerance_m=math.inf), dict(axis_weight_m=-1)])
def test_invalid_ik_configuration(kwargs):
    with pytest.raises(ValueError):
        IKConfig(**kwargs)
