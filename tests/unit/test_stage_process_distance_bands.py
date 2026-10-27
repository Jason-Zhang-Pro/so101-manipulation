"""Distance-band credit is continuous, bounded and conditional on real evidence."""

from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards.stage_process_reward import StageProcessRewardConfig, StageReachReward
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_stage_process_reward import DT, RewardTrace
from test_stage_process_progress_reward import pose_at_distance


EDGES = (.08, .075, .07, .065, .06, .055, .05, .045, .04, .035, .03, .025, .02)
BONUSES = (1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 7., 8.)
BAND_KEYS = tuple(f"s1_distance_band_{index:02d}" for index in range(1, 13))
V2_VALUES = dict(schema_version=2, s1_shaping_scale=.1,
                 s1_distance_cost_per_s=.4, s1_best_distance_bonus=8.)
# Recorded from the resolved schema-2 configuration of the 20260930_132641 run.
V2_CONFIG_HASH = "7279879690482889d51d857e61e27d689b5898eb9ac4996a2020c65ab13936e1"


def config(**overrides):
    return StageProcessRewardConfig(**{
        **V2_VALUES, "schema_version": 3, "s1_time_cost_per_s": .3,
        "s1_distance_band_edges_m": EDGES, "s1_distance_band_bonuses": BONUSES,
        **overrides,
    })


def trace(**kwargs):
    return RewardTrace(config=config(), **kwargs)


def band_rewards(result):
    return np.array([result.components[key] for key in BAND_KEYS])


def reset_at_distance(tr, distance):
    """Independent measured geometry, with the HOME joints required at reset."""
    pose = pose_at_distance(tr, distance, opening=0., tilt=70.)
    tr.trace.snapshot = tr.trace.tracker.reset(
        tr.trace.measurement(**pose), "pick_place", task_instance_id="bands-reset",
    )
    tr.trace.info = tr.trace.evaluator.reset(tr.trace.snapshot)
    tr.reward.reset(tr.trace.snapshot, tr.trace.info)


def discounted_return(results, gamma):
    return sum(gamma**index * result.reward for index, result in enumerate(results))


def test_schema_three_roundtrip_and_history_shape_without_changing_old_hashes():
    new = config()
    payload = new.to_dict()
    assert payload["s1_distance_band_edges_m"] == list(EDGES)
    assert payload["s1_distance_band_bonuses"] == list(BONUSES)
    assert StageProcessRewardConfig.from_dict(payload) == new
    assert new.context_dim == 14 and new.context_schema_version == 3
    assert sum(new.s1_distance_band_bonuses) == 48
    old1, old2 = StageProcessRewardConfig(), StageProcessRewardConfig(**V2_VALUES)
    assert old1.config_hash == "b0479c1b9ea4182fff60b22979cec9a41646656ebb2697dd05668d523a9a1cd9"
    assert old2.config_hash == V2_CONFIG_HASH
    for old in (old1, old2):
        assert not any("distance_band" in name for name in old.to_dict())
        assert StageProcessRewardConfig.from_dict(old.to_dict()) == old
        assert old.config_hash != new.config_hash
    assert old1.context_dim == 12 and old2.context_dim == 14
    stages = replace(StageTrainingConfig(), process_reward=new)
    assert StageTrainingConfig.from_dict(stages.to_dict()) == stages


@pytest.mark.parametrize("field,value", [
    ("s1_distance_band_edges_m", None),
    ("s1_distance_band_bonuses", None),
    ("s1_distance_band_edges_m", ()),
    ("s1_distance_band_edges_m", EDGES[:-1]),
    ("s1_distance_band_edges_m", (.08, .08) + EDGES[2:]),
    ("s1_distance_band_edges_m", (.07, .08) + EDGES[2:]),
    ("s1_distance_band_edges_m", EDGES[:-1] + (0.,)),
    ("s1_distance_band_edges_m", (math.nan,) + EDGES[1:]),
    ("s1_distance_band_edges_m", (math.inf,) + EDGES[1:]),
    ("s1_distance_band_edges_m", (True,) + EDGES[1:]),
    ("s1_distance_band_bonuses", BONUSES[:-1]),
    ("s1_distance_band_bonuses", (0.,) + BONUSES[1:]),
    ("s1_distance_band_bonuses", (-1.,) + BONUSES[1:]),
    ("s1_distance_band_bonuses", (math.nan,) + BONUSES[1:]),
    ("s1_distance_band_bonuses", (math.inf,) + BONUSES[1:]),
    ("s1_distance_band_bonuses", (True,) + BONUSES[1:]),
])
def test_malformed_distance_bands_are_rejected(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


@pytest.mark.parametrize("version", [1, 2])
def test_band_fields_cannot_silently_change_old_schema(version):
    kwargs = {} if version == 1 else V2_VALUES
    with pytest.raises(ValueError, match="schema_version=3"):
        StageProcessRewardConfig(**kwargs, s1_distance_band_edges_m=EDGES,
                                 s1_distance_band_bonuses=BONUSES)


def test_band_endpoint_must_equal_the_actual_stage_success_distance():
    bad = config(s1_distance_band_edges_m=EDGES[:-1] + (.019,))
    with pytest.raises(ValueError, match="distance"):
        replace(StageTrainingConfig(), process_reward=bad)
    tr = trace()
    with pytest.raises(ValueError, match="distance"):
        StageReachReward(bad, tr.trace.config, tr.trace.reward)


@pytest.mark.parametrize("index", range(12))
def test_each_five_mm_band_pays_proportionally_for_measured_partial_progress(index):
    tr = trace()
    upper, lower, bonus = EDGES[index], EDGES[index + 1], BONUSES[index]
    tr.step(**pose_at_distance(tr, upper, tilt=70.))
    for fraction in (.25, .5, .75, 1.):
        result = tr.step(**pose_at_distance(tr, upper - fraction * (upper - lower), tilt=70.))
        expected = np.zeros(12)
        expected[index] = bonus / 4
        np.testing.assert_allclose(band_rewards(result), expected, atol=2e-12, rtol=0)
        assert result.components["s1_best_distance"] > 0


def test_micrometre_progress_is_not_delayed_until_a_distance_threshold():
    tr = trace()
    tr.step(**pose_at_distance(tr, .077, tilt=70.))
    result = tr.step(**pose_at_distance(tr, .077 - 1e-6, tilt=70.))
    assert result.components[BAND_KEYS[0]] == pytest.approx(1. / .005 * 1e-6)
    np.testing.assert_allclose(band_rewards(result)[1:], 0, atol=1e-12)


def test_a_step_crossing_multiple_bands_pays_full_and_partial_intervals_once():
    tr = trace()
    tr.step(**pose_at_distance(tr, .0854, tilt=70.))
    result = tr.step(**pose_at_distance(tr, .063, tilt=70.))
    expected = np.zeros(12)
    expected[:3] = BONUSES[:3]
    expected[3] = .4 * BONUSES[3]
    np.testing.assert_allclose(band_rewards(result), expected, atol=2e-12, rtol=0)
    assert result.components["s1_best_distance"] > 0
    repeat = tr.step(**pose_at_distance(tr, .063, tilt=70.))
    np.testing.assert_allclose(band_rewards(repeat), 0, atol=1e-12)


def test_starting_inside_a_band_sets_a_baseline_without_awarding_prior_progress():
    tr = trace()
    reset_at_distance(tr, .0775)
    stationary = tr.step(**pose_at_distance(tr, .0775, opening=0., tilt=70.))
    np.testing.assert_allclose(band_rewards(stationary), 0, atol=1e-12)
    assert stationary.components["s1_best_distance"] == 0
    improved = tr.step(**pose_at_distance(tr, .075, opening=0., tilt=70.))
    assert band_rewards(improved)[0] == pytest.approx(.5)
    np.testing.assert_allclose(band_rewards(improved)[1:], 0, atol=1e-12)


def test_approach_retreat_and_recovery_cannot_repeat_bands_or_exceed_48():
    tr = trace()
    accumulated = np.zeros(12)
    for index, distance in enumerate(EDGES[1:]):
        result = tr.step(**pose_at_distance(tr, distance, tilt=70.))
        accumulated += band_rewards(result)
        assert accumulated.sum() == pytest.approx(sum(BONUSES[:index + 1]))
        for repeated_distance in (.09, distance, distance):
            result = tr.step(**pose_at_distance(tr, repeated_distance, tilt=70.))
            np.testing.assert_allclose(band_rewards(result), 0, atol=1e-12)
    result = tr.step(**pose_at_distance(tr, .01, tilt=70.))
    np.testing.assert_allclose(band_rewards(result), 0, atol=1e-12)
    np.testing.assert_allclose(accumulated, BONUSES, atol=2e-12)
    assert accumulated.sum() == pytest.approx(48)
    # Being near is insufficient: all the original orientation/opening/XY
    # conditions still govern StageEvaluator, independently of distance credit.
    assert not tr.trace.info["stage_success"]


@pytest.mark.parametrize("kind", ["duplicate", "gap", "contact"])
def test_missing_or_unqualified_evidence_cannot_be_repaid_after_recovery(kind):
    tr = trace()
    tr.step(**pose_at_distance(tr, .0854, tilt=70.))
    pose = pose_at_distance(tr, .065, tilt=70.)
    overrides = {"evidence_sim_time_s": tr.trace.snapshot.evidence_sim_time_s} if kind == "duplicate" else (
        {"dt": .2} if kind == "gap" else {"left_contact": True})
    blocked = tr.step(**pose, **overrides)
    np.testing.assert_allclose(band_rewards(blocked), 0, atol=1e-12)
    assert tr.reward.critic_context()[-1] == 0
    recovered = tr.step(**pose)
    np.testing.assert_allclose(band_rewards(recovered), 0, atol=1e-12)
    tr.step(**pose_at_distance(tr, .09, tilt=70.))
    revisit = tr.step(**pose)
    np.testing.assert_allclose(band_rewards(revisit), 0, atol=1e-12)
    new_progress = tr.step(**pose_at_distance(tr, .064, tilt=70.))
    expected = np.zeros(12)
    expected[3] = .2 * BONUSES[3]
    np.testing.assert_allclose(band_rewards(new_progress), expected, atol=2e-12, rtol=0)


def test_success_is_a_bootstrapped_stage_cut_and_does_not_repeat_distance_credit():
    tr = trace()
    results = []
    while not tr.trace.info["stage_cut"]:
        results.append(tr.step(**tr.pose(x=0., z=.019, opening=.7)))
    assert len(results) <= 7
    assert tr.trace.info["stage_result"] == "SUCCESS"
    assert results[-1].truncated and not results[-1].terminated
    assert tr.trace.snapshot.outcome == "running"
    assert results[-1].potential_after > 0
    assert results[-1].components["s1_success"] == 10
    assert sum(result.components["s1_success"] for result in results) == 10
    assert sum(band_rewards(result).sum() for result in results) == pytest.approx(48)
    np.testing.assert_allclose(band_rewards(results[-1]), 0, atol=1e-12)
    assert all(result.components["failure"] == 0 for result in results)


def test_true_failure_cannot_collect_bands_even_when_distance_improves():
    tr = trace()
    tr.step(**pose_at_distance(tr, .0854, tilt=70.))
    result = tr.step(**pose_at_distance(tr, .019, tilt=70.), failure_reason="collision")
    np.testing.assert_allclose(band_rewards(result), 0, atol=1e-12)
    assert result.components["s1_best_distance"] == 0
    assert result.components["failure"] == -10
    assert result.components["s1_success"] == 0
    assert result.terminated and not result.truncated and result.potential_after == 0


def test_stationary_time_cost_is_ten_times_schema_two_without_changing_other_terms():
    old, new = RewardTrace(config=StageProcessRewardConfig(**V2_VALUES)), trace()
    for tr in (old, new):
        tr.step(**pose_at_distance(tr, .0854, tilt=70.))
    old_result = old.step(**pose_at_distance(old, .0854, tilt=70.))
    new_result = new.step(**pose_at_distance(new, .0854, tilt=70.))
    assert new_result.components["s1_time"] == pytest.approx(-.01)
    assert new_result.components["s1_time"] == pytest.approx(10 * old_result.components["s1_time"])
    for key, value in old_result.components.items():
        if key != "s1_time":
            assert new_result.components[key] == pytest.approx(value)
    np.testing.assert_allclose(band_rewards(new_result), 0, atol=1e-12)
    assert new_result.reward < old_result.reward < 0


def run_near_target_outcome(outcome, *, delay=0):
    tr = trace()
    results = [tr.step(**pose_at_distance(tr, distance, tilt=70.)) for distance in EDGES]
    near_pose = pose_at_distance(tr, .02, tilt=70.)
    if outcome == "failure":
        results.append(tr.step(**near_pose, failure_reason="collision"))
    else:
        for _ in range(delay):
            results.append(tr.step(**near_pose))
        while not tr.trace.info["stage_cut"]:
            pose = tr.pose(x=0., z=.019, opening=.7) if outcome == "success" else near_pose
            results.append(tr.step(**pose))
    return tr, results


def test_same_approach_prefix_prefers_prompt_success_then_delayed_success_then_waiting_to_failure():
    fast, fast_results = run_near_target_outcome("success")
    slow, slow_results = run_near_target_outcome("success", delay=60)
    wait, wait_results = run_near_target_outcome("timeout")
    fail, fail_results = run_near_target_outcome("failure")
    returns = [discounted_return(results, fast.trace.reward.gamma) for results in (
        fast_results, slow_results, wait_results, fail_results)]
    assert returns[0] > returns[1] > returns[2] > returns[3]
    assert [tr.trace.info["stage_result"] for tr in (fast, slow, wait, fail)] == [
        "SUCCESS", "SUCCESS", "TIMEOUT", "FAILURE"]
    assert len(wait_results) == 300
    # These are bounded measured trajectories, not a claim that raw truncated
    # returns rank every policy; the PPO critic also bootstraps stage cuts.
    for results in (fast_results, slow_results, wait_results, fail_results):
        assert sum(band_rewards(result).sum() for result in results) == pytest.approx(48)


def test_largest_ten_second_running_cost_does_not_prefer_immediate_failure():
    wait, fail = trace(), trace()
    pose = pose_at_distance(wait, .0854, tilt=70.)
    wait_results = []
    while not wait.trace.info["stage_cut"]:
        wait_results.append(wait.step(**pose, left_contact=True))
    failure = fail.step(**pose, left_contact=True, failure_reason="collision")
    running_keys = ("s1_time", "s1_remaining", "s1_distance_remaining")
    costs = [sum(result.components[key] for key in running_keys) for result in wait_results]
    assert len(costs) == 300
    assert sum(costs) == pytest.approx(-10)
    gamma = wait.trace.reward.gamma
    assert -sum(gamma**index * cost for index, cost in enumerate(costs)) < 10
    assert discounted_return(wait_results, gamma) > failure.reward
    assert all(band_rewards(result).sum() == 0 for result in wait_results)


def test_context_reuses_best_quality_and_eligibility_and_reset_discards_episode_claims():
    tr = trace()
    initial = tr.reward.critic_context()
    assert initial.shape == (14,)
    result = tr.step(**pose_at_distance(tr, .05, tilt=70.))
    assert band_rewards(result).sum() == pytest.approx(sum(BONUSES[:6]))
    progressed = tr.reward.critic_context()
    assert progressed[-2] > initial[-2] and progressed[-1] == 1
    progressed[:] = math.nan
    assert np.isfinite(tr.reward.critic_context()).all()
    reset_at_distance(tr, .0854)
    result = tr.step(**pose_at_distance(tr, .05, tilt=70.))
    assert band_rewards(result).sum() == pytest.approx(sum(BONUSES[:6]))
