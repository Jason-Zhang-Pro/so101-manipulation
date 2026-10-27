"""Offline distribution checks for physical, distance-scaled ACT exploration."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.act.bounded_actions import bound_action_means

from so101_capstone.policies.so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS,
    policy_positions_to_gazebo,
)
from so101_capstone.training.act_ppo import (
    ActPpoConfig,
    ActPpoExploration,
    evaluate_bounded_policy_actions,
    sample_bounded_policy_actions,
)
from so101_capstone.training.act_ppo_exploration_profile import (
    final_good_act_exploration_offset,
    final_good_act_exploration_signature,
    validate_final_good_act_exploration_bounds,
)


HALF_SPANS = np.asarray([
    (joint.gazebo_upper_rad - joint.gazebo_lower_rad) / 2.
    for joint in SIM_JOINT_CALIBRATIONS
])


def _exploration():
    return ActPpoExploration(ActPpoConfig(
        initial_log_std=-3., minimum_log_std=-9., maximum_log_std=-2.5,
        gripper_maximum_log_std=-1.5,
    ))


def _offset(info, stage="moving", initial=-3.):
    return final_good_act_exploration_offset(
        info, stage_name=stage, initial_log_std=initial,
    )


def _rad_stds(info, stage="moving"):
    return _exploration().with_offset(_offset(info, stage)).detach().double().exp().numpy() * HALF_SPANS


@pytest.mark.parametrize("distance,expected", [
    (0., .05), (.079999, .05), (.08, .05), (.080001, .08),
    (.15, .08), (.150001, .12), (1., .12),
])
def test_distance_boundaries_apply_same_physical_std_to_all_six_joints(distance, expected):
    np.testing.assert_allclose(
        _rad_stds({"task_phase": "approach", "target_distance_m": distance}),
        np.full(6, expected), rtol=3e-7,
    )


@pytest.mark.parametrize("phase,stage", [
    ("APPROACH", "moving"), ("GRASP_LIFT", "manipulation"),
    ("TRANSPORT", "moving"), ("PLACE_RELEASE", "manipulation"),
])
def test_target_phases_use_target_distance_not_home_error(phase, stage):
    np.testing.assert_allclose(_rad_stds({
        "task_phase_name": phase, "target_distance_m": .2, "home_error_rad": 0.,
    }, stage), [.12] * 6, rtol=3e-7)


@pytest.mark.parametrize("phase,stage", [
    ("WAIT_HOME", "idle"), ("RETURN_HOME", "moving"), ("HOME_HOLD", "home_hold"),
])
@pytest.mark.parametrize("error", [0., .01, .02, .03, .08, .080001, .15, .150001, .3])
def test_home_taper_uses_error_in_radians_and_the_active_reward_tolerance(phase, stage, error):
    tolerance = .02
    bin_std = .12 if error > .15 else .08 if error > .08 else .05
    expected = min(bin_std, .001 + .5 * max(error - tolerance, 0.))
    np.testing.assert_allclose(_rad_stds({
        "task_phase": phase, "home_error_rad": error,
        "home_joint_tolerance_rad": tolerance, "target_distance_m": 1.,
    }, stage), [expected] * 6, rtol=5e-7)


def test_home_tolerance_is_not_a_hardcoded_one_degree():
    base = {"task_phase": "WAIT_HOME", "home_error_rad": .03}
    np.testing.assert_allclose(_rad_stds({**base, "home_joint_tolerance_rad": .04}, "idle"),
                               [.001] * 6, rtol=5e-7)
    np.testing.assert_allclose(_rad_stds({**base, "home_joint_tolerance_rad": .01}, "idle"),
                               [.011] * 6, rtol=5e-7)


def test_stage_evidence_takes_precedence_without_changing_phase_selection():
    np.testing.assert_allclose(_rad_stds({
        "task_phase": "APPROACH", "stage_reach_distance_m": .02, "target_distance_m": 1.,
    }), [.05] * 6, rtol=3e-7)
    np.testing.assert_allclose(_rad_stds({
        "task_phase": "RETURN_HOME", "stage_home_error_rad": .01,
        "home_error_rad": 1., "home_joint_tolerance_rad": .02,
    }), [.001] * 6, rtol=5e-7)


@pytest.mark.parametrize("info,stage,match", [
    ({}, "moving", "valid task phase"),
    ({"task_phase": "unknown"}, "moving", "valid task phase"),
    ({"task_phase": "APPROACH"}, "moving", "stage_reach_distance_m"),
    ({"task_phase": "WAIT_HOME", "home_error_rad": 0.}, "idle", "home_joint_tolerance_rad"),
    ({"task_phase": "WAIT_HOME", "home_joint_tolerance_rad": .02}, "idle", "stage_home_error_rad"),
    ({"task_phase": "WAIT_HOME", "home_error_rad": 0., "home_joint_tolerance_rad": 0.}, "idle", "positive"),
    ({"task_phase": "RETURN_HOME", "home_error_rad": .02, "home_joint_tolerance_rad": .02}, "idle", "does not match"),
])
def test_missing_or_mismatched_required_evidence_is_rejected(info, stage, match):
    with pytest.raises(ValueError, match=match):
        _offset(info, stage)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1., True, "0.1", complex(1., 0.)])
@pytest.mark.parametrize("key,info,stage", [
    ("target_distance_m", {"task_phase": "APPROACH"}, "moving"),
    ("home_error_rad", {"task_phase": "WAIT_HOME", "home_joint_tolerance_rad": .02}, "idle"),
    ("home_joint_tolerance_rad", {"task_phase": "WAIT_HOME", "home_error_rad": .02}, "idle"),
])
def test_invalid_measurements_are_not_treated_as_zero(value, key, info, stage):
    with pytest.raises(ValueError, match=key):
        _offset({**info, key: value}, stage)


def test_invalid_stage_measurement_does_not_fall_back_to_valid_full_task_measurement():
    with pytest.raises(ValueError, match="stage_reach_distance_m"):
        _offset({"task_phase": "APPROACH", "stage_reach_distance_m": None, "target_distance_m": .2})


@pytest.mark.parametrize("distance,expected", [(.02, .05), (.1, .08), (.2, .12)])
def test_real_sampler_is_bidirectional_physical_std_and_bounded_samples_replay_exactly(distance, expected):
    exploration = _exploration()
    offset = _offset({"task_phase": "APPROACH", "target_distance_m": distance})
    logs = exploration.with_offset(offset)
    means = torch.tensor([0.] * 5 + [50.], dtype=torch.float64).expand(4096, 6)
    torch.manual_seed(732)
    actions, old_log_prob, latent = sample_bounded_policy_actions(means, logs, return_latent=True)
    replay, _ = evaluate_bounded_policy_actions(means, actions, logs, policy_action_latents=latent)
    torch.testing.assert_close(replay, old_log_prob, rtol=0., atol=0.)
    assert bool((latent.abs() < 1.).all())
    commands = np.stack([policy_positions_to_gazebo(row) for row in actions.detach().numpy()])
    deviations = commands - policy_positions_to_gazebo(means[0].numpy())
    assert np.all((deviations < 0.).any(axis=0))
    assert np.all((deviations > 0.).any(axis=0))
    np.testing.assert_allclose(np.sqrt(np.mean(deviations ** 2, axis=0)), [expected] * 6, rtol=.04)


def test_learned_base_keeps_offset_fixed_and_changes_replayed_distribution_with_gradients():
    exploration = _exploration()
    info = {"task_phase": "APPROACH", "target_distance_m": .1}
    offset = _offset(info)
    saved_offset = offset.copy()
    before = exploration.with_offset(offset).detach().clone()
    means = torch.tensor([[0.] * 5 + [50.]])
    torch.manual_seed(400)
    actions, old_log_prob, latent = sample_bounded_policy_actions(means, before, return_latent=True)
    with torch.no_grad():
        exploration.log_std.add_(.1)
    after = exploration.with_offset(saved_offset)
    torch.testing.assert_close(after, before + .1)
    np.testing.assert_allclose(after.detach().double().exp().numpy() * HALF_SPANS,
                               [.08 * math.exp(.1)] * 6, rtol=4e-7)
    new_log_prob, _ = evaluate_bounded_policy_actions(means, actions, after, policy_action_latents=latent)
    assert not torch.equal(new_log_prob, old_log_prob)
    new_log_prob.sum().backward()
    assert torch.isfinite(exploration.log_std.grad).all()
    assert torch.count_nonzero(exploration.log_std.grad) == 6
    np.testing.assert_array_equal(saved_offset, _offset(info))


def test_outside_raw_predictions_are_bounded_before_exploration_and_replay_matches():
    exploration = _exploration()
    logs = exploration.with_offset(_offset({"task_phase": "APPROACH", "target_distance_m": .2}))
    raw_means = torch.tensor([110., -110., 110., -110., 110., -10.]).expand(256, 6)
    config = SimpleNamespace(action_bounds_lower=[-100.] * 5 + [0.],
                             action_bounds_upper=[100.] * 6, action_bound_softness=.02)
    means = bound_action_means(raw_means, config)
    torch.manual_seed(95)
    actions, old_log_prob, latent = sample_bounded_policy_actions(means, logs, return_latent=True)
    assert bool((latent.abs() <= 1.).all())
    assert bool(((means >= torch.tensor(config.action_bounds_lower))
                 & (means <= torch.tensor(config.action_bounds_upper))).all())
    # Boundary exploration retains inward variation rather than folding an
    # unbounded tail into repeated limit commands.
    outward_sign = torch.tensor([1., -1., 1., -1., 1., -1.])
    assert bool((((actions - means) * outward_sign < 0).float().mean(dim=0) > .4).all())
    assert bool((actions.std(dim=0) > 0.1).all())
    replay, _ = evaluate_bounded_policy_actions(means, actions, logs, policy_action_latents=latent)
    torch.testing.assert_close(replay, old_log_prob, rtol=0., atol=0.)
    commands = np.stack([policy_positions_to_gazebo(row) for row in actions.detach().numpy()])
    assert np.all(commands >= np.asarray([j.gazebo_lower_rad for j in SIM_JOINT_CALIBRATIONS]) - 1e-12)
    assert np.all(commands <= np.asarray([j.gazebo_upper_rad for j in SIM_JOINT_CALIBRATIONS]) + 1e-12)


def test_valid_bounds_cover_the_home_floor_and_every_distance_bin():
    validate_final_good_act_exploration_bounds(
        initial_log_std=-3., minimum_log_std=-9., maximum_log_std=-2.5,
        gripper_maximum_log_std=-1.5,
    )


@pytest.mark.parametrize("overrides,match", [
    ({"minimum_log_std": -6.}, "0.001 rad"),
    ({"maximum_log_std": -3.}, "0.12 rad"),
    ({"gripper_maximum_log_std": -3.}, "gripper"),
    ({"gripper_maximum_log_std": None}, "gripper"),
    ({"initial_log_std": 0.}, "outside configured bounds"),
    ({"minimum_log_std": float("nan")}, "finite"),
    ({"maximum_log_std": True}, "finite"),
])
def test_bounds_cannot_silently_replace_the_requested_physical_profile(overrides, match):
    kwargs = dict(initial_log_std=-3., minimum_log_std=-9., maximum_log_std=-2.5,
                  gripper_maximum_log_std=-1.5)
    with pytest.raises(ValueError, match=match):
        validate_final_good_act_exploration_bounds(**{**kwargs, **overrides})


def test_signature_is_fresh_json_with_units_calibration_and_taper_contract():
    signature = final_good_act_exploration_signature()
    assert json.loads(json.dumps(signature, allow_nan=False)) == signature
    assert signature["profile"] == "final_good_act"
    assert signature["distance_bins"]["reach_distance_units"] == "metres"
    assert signature["distance_bins"]["home_error_units"] == "radians_max_joint_error"
    np.testing.assert_array_equal(signature["normalized_half_span_rad"], HALF_SPANS)
    signature["home"]["minimum_initial_std_rad"] = 99.
    assert final_good_act_exploration_signature()["home"]["minimum_initial_std_rad"] == .001
