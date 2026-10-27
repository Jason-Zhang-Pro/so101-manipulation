from types import SimpleNamespace

import pytest

from so101_capstone.policies.act_temporal_config import validate_act_temporal_config


def config(*, chunk=15, steps=1, coefficient=.01):
    return SimpleNamespace(chunk_size=chunk, n_action_steps=steps, temporal_ensemble_coeff=coefficient)


@pytest.mark.parametrize("coefficient", [None, .01, 0., -.01])
def test_valid_temporal_schedule_metadata(coefficient):
    metadata = validate_act_temporal_config(config(coefficient=coefficient))
    assert metadata == {
        "chunk_size": 15, "n_action_steps": 1,
        "temporal_ensemble_coeff": coefficient,
        "chunk_schedule": "fixed_n_action_steps" if coefficient is None else "temporal_ensemble_every_step",
    }


def test_fixed_schedule_retains_multistep_chunk_execution():
    assert validate_act_temporal_config(config(steps=5, coefficient=None))["n_action_steps"] == 5


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), -float("inf"), True, ".01"])
def test_invalid_temporal_coefficient_rejected(coefficient):
    with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
        validate_act_temporal_config(config(coefficient=coefficient))


@pytest.mark.parametrize("chunk,steps", [(0, 1), (True, 1), (15., 1), (15, 0), (15, True), (15, 1.), (15, 16)])
def test_invalid_action_schedule_rejected(chunk, steps):
    with pytest.raises(ValueError, match="chunk_size|n_action_steps"):
        validate_act_temporal_config(config(chunk=chunk, steps=steps))


def test_temporal_ensemble_requires_fresh_inference_every_step():
    with pytest.raises(ValueError, match="requires n_action_steps=1"):
        validate_act_temporal_config(config(steps=5))


@pytest.mark.parametrize("coefficient,chunk", [(-1., 100), (-.0885, 1000), (1e300, 15), (-1e300, 15)])
def test_finite_coefficients_with_overflowing_float32_weights_rejected(coefficient, chunk):
    with pytest.raises(ValueError, match="weights.*finite float32"):
        validate_act_temporal_config(config(coefficient=coefficient, chunk=chunk))


def test_temporal_reward_home_hold_still_has_three_second_floor():
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    reward = PickPlaceRewardConfig()
    minimum, maximum = reward.resolve_home_hold_bounds(n_action_steps=1)
    assert reward.control_hz == 30
    assert minimum >= 3.
    assert maximum > minimum
    assert (minimum, maximum) == reward.resolve_home_hold_bounds(n_action_steps=5)
