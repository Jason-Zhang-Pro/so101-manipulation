"""Schema-4 HOME base cost guides correction beyond the old 0.5-rad plateau."""
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import (
    PickPlaceReward, PickPlaceRewardConfig, PickPlaceRewardInput,
    TaskMeasurements, TaskStateTracker,
)


def home_transition(*, error_rad=0., velocity_rad_s=0., schema=4,
                    dt=1/30, mode="idle_empty", valid=True):
    cfg = PickPlaceRewardConfig(reward_schema_version=schema, shaping_enabled=False)
    q, velocity = list(cfg.home_joint_positions_rad), [0.] * 6
    q[-1] += error_rad
    velocity[-1] = velocity_rad_s
    measured = TaskMeasurements(
        0., q, velocity, (0., 0., .8), np.eye(3), evidence_valid=valid,
        joint_measurement_valid=valid,
    )
    tracker = TaskStateTracker(cfg)
    previous = tracker.reset(measured, task_mode=mode)
    current = tracker.update(replace(measured, sim_time_s=dt))
    reward = PickPlaceReward(cfg)
    reward.reset(previous)
    return cfg, reward.step(PickPlaceRewardInput(previous, current, dt))


def test_schema4_observed_78_degree_gripper_improves_with_two_and_five_degree_corrections():
    costs = [home_transition(error_rad=math.radians(degrees))[1].components["home_position"]
             for degrees in (80., 78., 76., 73.)]
    assert all(left < right < 0. for left, right in zip(costs, costs[1:]))
    # Potential shaping is disabled: improvement is in the actual base objective.
    results = [home_transition(error_rad=math.radians(degrees))[1] for degrees in (78., 76., 73.)]
    assert all(item.components["potential_shaping"] == 0. for item in results)
    assert all(item.components["remaining_task"] == 0. for item in results)
    assert results[0].reward < results[1].reward < results[2].reward


@pytest.mark.parametrize("error", [-1.7, -.8, -.1, .1, .8, 1.7, 1e200])
def test_schema4_home_cost_matches_bounded_formula_without_large_error_square_overflow(error):
    cfg, result = home_transition(error_rad=error)
    excess = max(abs(error) - cfg.home_joint_tolerance_rad, 0.)
    expected = -cfg.home_position_cost_per_step / 6 * (excess / math.hypot(cfg.home_position_scales_rad[-1], excess))**2
    value = result.components["home_position"]
    assert math.isfinite(value)
    assert -cfg.home_position_cost_per_step / 6 <= value <= 0.
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("fraction", [-.99, -.5, 0., .5, .99])
def test_home_tolerance_has_zero_position_cost(fraction):
    cfg = PickPlaceRewardConfig()
    _, result = home_transition(error_rad=fraction * cfg.home_joint_tolerance_rad)
    assert result.components["home_position"] == 0.


def test_schema4_retains_velocity_penalty_and_physical_time_scaling():
    _, stationary = home_transition(error_rad=1.36)
    _, moving = home_transition(error_rad=1.36, velocity_rad_s=.5)
    _, longer = home_transition(error_rad=1.36, velocity_rad_s=.5, dt=.1)
    assert moving.components["home_position"] == stationary.components["home_position"]
    assert stationary.components["home_velocity"] == 0.
    assert moving.components["home_velocity"] < 0.
    assert moving.reward < stationary.reward
    assert longer.components["home_position"] == pytest.approx(3 * moving.components["home_position"])
    assert longer.components["home_velocity"] == pytest.approx(3 * moving.components["home_velocity"])


def test_unknown_measurements_do_not_create_home_position_or_velocity_cost():
    _, result = home_transition(error_rad=1.36, velocity_rad_s=.5, valid=False)
    assert result.components["home_position"] == 0.
    assert result.components["home_velocity"] == 0.


def test_schema3_complete_idle_result_and_old_far_error_plateau_are_unchanged():
    _, result = home_transition(error_rad=math.radians(78.), velocity_rad_s=.5, schema=3)
    # These are exactly the pre-v4 cost operations, not the new rational score.
    position = -.05 * float(np.mean(np.clip(np.array([0.] * 5 + [math.radians(78.) - math.radians(1.)]) / .5, 0, 1)**2))
    velocity = -.02 * float(np.mean(np.clip(np.array([0.] * 5 + [.5 - .03]), 0, 1)**2))
    assert result.to_info() == {
        "reward": position + velocity,
        "reward_components": {
            "potential_shaping": 0., "success": 0., "failure": 0., "active_time": 0.,
            "home_position": position, "home_velocity": velocity,
            "moving_velocity": 0., "command_change": 0.,
        },
        "potential_before": 0., "potential_after": 0., "terminated": False, "truncated": False,
    }
    old_costs = [home_transition(error_rad=math.radians(degrees), schema=3)[1].components["home_position"]
                 for degrees in (78., 76., 73., 40.)]
    assert all(value == old_costs[0] for value in old_costs)
