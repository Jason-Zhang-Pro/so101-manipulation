"""Schema 4 fixes wide-opening shaping without changing historical v3 runs."""

import numpy as np
import pytest

from so101_capstone.rewards import (
    PickPlaceReward, PickPlaceRewardConfig, PickPlaceRewardInput,
    TaskMeasurements, TaskStateTracker,
)


def state(config, *, distance=.03, opening=1.8, time=0.):
    q = list(config.home_joint_positions_rad)
    q[-1] += opening
    measured = TaskMeasurements(
        time, q, (0.,) * 6, (0., 0., .765+distance), np.diag([1., -1., -1.]),
        target_name="cube", target_position_world_m=(0., 0., .765),
        target_motion_valid=True, contact_valid=True,
    )
    return TaskStateTracker(config).reset(measured)


def test_real_run_schema3_resolved_config_hash_is_unchanged():
    # config.json execution_signature.reward_config_hash from the actual run
    # act_ppo_20260927_153051_from_0005000. Do not load untracked run artifacts
    # in this test: a fresh checkout must verify the compatibility contract.
    legacy = PickPlaceRewardConfig(reward_schema_version=3)
    assert legacy.config_hash == "3aa8fbbb178c56fbe675320fe8d4ab2971bb9db86ae02ab54f9303bd865b762e"
    assert PickPlaceRewardConfig.from_dict(legacy.to_dict()).config_hash == legacy.config_hash
    assert PickPlaceRewardConfig().reward_schema_version == 4
    assert PickPlaceRewardConfig().config_hash != legacy.config_hash


@pytest.mark.parametrize("distance,opening,opening_score,closure_score,potential", [
    (.028759266169284724, 1.6525244887227628, .0593443890965466, 0., .8216185999812644),
    (.03, 1.8, 0., 0., .7971745196419153),
    (.03, 1.75, 0., 0., .7971745196419153),
    (.005, 1.8, 0., 0., 1.3632733307483953),
    (.005, 1.0, .875, 0., 1.3632733307483953),
    (.03, .7, 1., .22222222222222232, .9110565938764745),
])
def test_schema3_keeps_pre_change_scores_including_its_historical_plateaus(
    distance, opening, opening_score, closure_score, potential,
):
    # Frozen from the pre-change implementation with an aligned tool. The first
    # pair uses the run's best observed distance/opening, not its exact attitude.
    cfg = PickPlaceRewardConfig(reward_schema_version=3)
    snapshot = state(cfg, distance=distance, opening=opening)
    assert snapshot.approach_opening_score == pytest.approx(opening_score, abs=1e-14)
    assert snapshot.grasp_closure_score == pytest.approx(closure_score, abs=1e-14)
    assert PickPlaceReward(cfg).potential(snapshot) == pytest.approx(potential, abs=1e-14)


def test_schema4_narrowing_above_old_opening_cutoff_has_signal():
    cfg = PickPlaceRewardConfig()
    reward = PickPlaceReward(cfg)
    wider, narrower = state(cfg, opening=1.80), state(cfg, opening=1.75)
    assert wider.grasp_close_gate == narrower.grasp_close_gate == 0
    assert 0 < wider.approach_opening_score < narrower.approach_opening_score < 1
    assert reward.potential(narrower) > reward.potential(wider)


def test_schema4_at_full_close_gate_narrowing_is_informative_across_physical_opening_range():
    cfg, reward = PickPlaceRewardConfig(), PickPlaceReward(PickPlaceRewardConfig())
    # Calibrated max q 1.7441249748 minus HOME q -0.0872664626.
    openings = np.linspace(1.8313914374101695, 0., 101)
    snapshots = [state(cfg, distance=.005, opening=float(opening)) for opening in openings]
    values = [reward.potential(s) for s in snapshots]
    assert all(s.grasp_close_gate == pytest.approx(1.) for s in snapshots)
    assert all(right > left for left, right in zip(values, values[1:]))
    assert all(not s.grasp_valid and not s.lifted_once and not s.success_event for s in snapshots)


def test_schema4_keeps_legitimate_opening_interval_and_home_reference():
    legacy, current = PickPlaceRewardConfig(reward_schema_version=3), PickPlaceRewardConfig()
    assert current.home_joint_positions_rad == legacy.home_joint_positions_rad
    assert current.success_definition_version == legacy.success_definition_version == 2
    assert current.pregrasp_distance_tolerance_m == legacy.pregrasp_distance_tolerance_m
    assert current.pregrasp_orientation_tolerance_rad == legacy.pregrasp_orientation_tolerance_rad
    assert current.approach_opening_min_rad == legacy.approach_opening_min_rad
    assert current.approach_opening_max_rad == legacy.approach_opening_max_rad
    for opening in (.45, .6, .9):
        assert state(current, opening=opening).approach_opening_score == pytest.approx(1.)


@pytest.mark.parametrize("schema", [3, 4])
def test_versioned_pregrasp_potential_remains_bounded_and_closing_loop_cannot_farm_reward(schema):
    cfg = PickPlaceRewardConfig(reward_schema_version=schema)
    reward = PickPlaceReward(cfg)
    for distance in np.linspace(0., .4, 11):
        for opening in np.linspace(-.0861, 1.8314, 17):
            assert 0 <= reward.potential(state(cfg, distance=float(distance), opening=float(opening))) <= 2
    initial = state(cfg, distance=.005, opening=1.80)
    reward.reset(initial)
    previous, discounted = initial, 0.
    for index, opening in enumerate([1.4, .7, .2, .7, 1.4, 1.80]):
        current = state(cfg, distance=.005, opening=opening, time=(index+1)/30)
        shaped = reward.step(PickPlaceRewardInput(previous, current, 1/30)).components["potential_shaping"]
        discounted += cfg.gamma**index * shaped
        previous = current
    assert discounted == pytest.approx((cfg.gamma**6-1)*reward.potential(initial))
    assert discounted < 0
