"""Schema-2 task credit must survive PBRS without rewarding repeated approach."""

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.rewards.stage_process_reward import StageProcessRewardConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_stage_process_reward import COMPONENTS, DT, RewardTrace, progress
from test_training_stages import StageTrace


V1_CONFIG_HASH = "b0479c1b9ea4182fff60b22979cec9a41646656ebb2697dd05668d523a9a1cd9"
V2_VALUES = dict(schema_version=2, s1_shaping_scale=.1,
                 s1_distance_cost_per_s=.4, s1_best_distance_bonus=8.)


def config(**overrides):
    return StageProcessRewardConfig(**{**V2_VALUES, **overrides})


def trace(**kwargs):
    return RewardTrace(config=config(), **kwargs)


def pose_at_distance(tr, distance, *, opening=.959, tilt=25.):
    """Physical target/grasp separation, near the observed 8.54 cm plateau."""
    height = .0046
    assert distance >= height
    return tr.pose(x=math.sqrt(distance**2 - height**2), z=height,
                   opening=opening, tilt=tilt)


def best_reward(result):
    return result.components["s1_best_distance"]


def test_version_one_serialization_and_recorded_hash_remain_exact():
    old = StageProcessRewardConfig()
    payload = old.to_dict()
    assert old.schema_version == 1
    assert not any(name in payload for name in V2_VALUES if name != "schema_version")
    # Captured from the original schema-1 training execution signature, not
    # recomputed using a second copy of the new serialization implementation.
    assert old.config_hash == V1_CONFIG_HASH
    assert StageProcessRewardConfig.from_dict(payload) == old
    assert old.context_dim == 12 and old.context_schema_version == 1


def test_new_profile_json_roundtrips_and_changes_the_resume_contract():
    path = Path(__file__).resolve().parents[2] / "config" / "act_ppo_stages_s1_progress_v2.json"
    stages = StageTrainingConfig.load(path)
    new = stages.process_reward
    assert new == config()
    assert StageProcessRewardConfig.from_dict(new.to_dict()) == new
    assert new.config_hash != V1_CONFIG_HASH
    assert new.context_dim == 14 and new.context_schema_version == 2


@pytest.mark.parametrize("field", ["s1_shaping_scale", "s1_distance_cost_per_s", "s1_best_distance_bonus"])
@pytest.mark.parametrize("value", [None, 0., -1., math.nan, math.inf, True])
def test_schema_two_requires_explicit_finite_positive_new_parameters(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


@pytest.mark.parametrize("field,value", [("s1_shaping_scale", .1),
                                          ("s1_distance_cost_per_s", .4),
                                          ("s1_best_distance_bonus", 8.)])
def test_new_parameters_cannot_silently_change_schema_one(field, value):
    with pytest.raises(ValueError, match="requires.*schema_version=2"):
        StageProcessRewardConfig(**{field: value})


def test_every_measured_millimetre_from_observed_plateau_gets_new_distance_credit():
    tr = trace()
    tr.step(**pose_at_distance(tr, .0854))
    last = tr.trace.info["stage_reach_distance_m"]
    for distance in np.arange(.0844, .0684, -.001):
        result = tr.step(**pose_at_distance(tr, float(distance)))
        measured = tr.trace.info["stage_reach_distance_m"]
        assert last - measured == pytest.approx(.001, abs=1e-12)
        assert best_reward(result) > 0
        assert result.reward > 0
        last = measured


def test_staying_still_receives_no_new_best_credit():
    tr = trace()
    pose = pose_at_distance(tr, .0854)
    tr.step(**pose)
    for _ in range(4):
        result = tr.step(**pose)
        assert best_reward(result) == 0
        assert all(progress(result, name) == pytest.approx(0) for name in COMPONENTS)
        assert result.reward < 0


def test_retreat_then_recovery_has_pbrs_progress_without_duplicate_best_bonus():
    tr = trace()
    tr.step(**pose_at_distance(tr, .0854))
    first = tr.step(**pose_at_distance(tr, .08))
    assert best_reward(first) > 0
    retreat = tr.step(**pose_at_distance(tr, .09))
    recovery = tr.step(**pose_at_distance(tr, .08))
    assert best_reward(retreat) == best_reward(recovery) == 0
    assert progress(retreat, "distance_coarse") < 0
    assert progress(recovery, "distance_coarse") > 0


def test_repeated_approach_cannot_exceed_the_single_episode_distance_budget():
    tr = trace()
    initial_distance = tr.trace.info["stage_reach_distance_m"]
    # The maximum budget follows from a bounded [0,1] distance-quality record.
    initial_quality = .1 / (.1 + max(initial_distance - .02, 0))
    limit = 8. * (1. - initial_quality)
    total = best_reward(tr.step(**pose_at_distance(tr, .0854)))
    total += best_reward(tr.step(**pose_at_distance(tr, .08)))
    for _ in range(8):
        retreat = tr.step(**pose_at_distance(tr, .0854))
        revisit = tr.step(**pose_at_distance(tr, .08))
        assert best_reward(retreat) == best_reward(revisit) == 0
    for distance in (.07, .06, .05, .04, .03, .019):
        # Deliberately wrong orientation avoids prematurely finishing S1 while
        # verifying that distance credit is bounded independently of success.
        total += best_reward(tr.step(**pose_at_distance(tr, distance, tilt=70.)))
        assert total <= limit + 1e-12
    assert total == pytest.approx(limit)
    assert not tr.trace.info["stage_success"]


@pytest.mark.parametrize("kind", ["duplicate", "gap", "contact"])
def test_unobserved_or_unqualified_approach_cannot_be_reclaimed_after_recovery(kind):
    tr = trace()
    tr.step(**pose_at_distance(tr, .0854))
    original_best = tr.reward.critic_context()[-2]
    pose = pose_at_distance(tr, .07)
    if kind == "duplicate":
        blocked = tr.step(**pose, evidence_sim_time_s=tr.trace.snapshot.evidence_sim_time_s)
    elif kind == "gap":
        blocked = tr.step(dt=.2, **pose)
    else:
        blocked = tr.step(**pose, left_contact=True)
    assert best_reward(blocked) == 0
    assert tr.reward.critic_context()[-2] >= original_best
    recovered = tr.step(**pose)
    assert best_reward(recovered) == 0
    recovered_best = tr.reward.critic_context()[-2]
    tr.step(**pose_at_distance(tr, .08))
    assert tr.reward.critic_context()[-2] >= recovered_best
    assert best_reward(tr.step(**pose)) == 0
    assert best_reward(tr.step(**pose_at_distance(tr, .069))) > 0


def test_repeated_evidence_cannot_reduce_distance_cost_by_changing_geometry():
    tr = trace()
    before = tr.step(**pose_at_distance(tr, .0854))
    repeated = tr.step(**pose_at_distance(tr, .03),
                       evidence_sim_time_s=tr.trace.snapshot.evidence_sim_time_s)
    assert repeated.components["s1_distance_remaining"] == pytest.approx(
        before.components["s1_distance_remaining"])
    assert best_reward(repeated) == 0


def test_better_opening_cannot_erase_distance_cost_and_nearer_pose_reduces_it():
    tr = trace()
    first = tr.step(**pose_at_distance(tr, .0854, opening=1.2))
    opened = tr.step(**pose_at_distance(tr, .0854, opening=.7))
    closer = tr.step(**pose_at_distance(tr, .0754, opening=.7))
    assert progress(opened, "opening") > 0
    assert best_reward(opened) == 0
    assert first.components["s1_distance_remaining"] == pytest.approx(
        opened.components["s1_distance_remaining"])
    assert opened.components["s1_distance_remaining"] < closer.components["s1_distance_remaining"] < 0


def test_shaping_reduction_does_not_scale_away_the_existing_dense_objective():
    old, new = RewardTrace(), trace()
    for kwargs in ({"x": .09, "z": .05}, {"x": .08, "z": .045}):
        old_result = old.step(**old.pose(**kwargs))
        new_result = new.step(**new.pose(**kwargs))
        assert new_result.potential_after == pytest.approx(.1 * old_result.potential_after)
        for name in COMPONENTS:
            assert progress(new_result, name) == pytest.approx(.1 * progress(old_result, name))
        assert new_result.components["s1_remaining"] == pytest.approx(old_result.components["s1_remaining"])
        assert new_result.components["s1_time"] == pytest.approx(old_result.components["s1_time"])


def test_real_failure_does_not_pay_distance_credit_and_zeroes_potential():
    tr = trace()
    tr.step(**pose_at_distance(tr, .0854))
    result = tr.step(**pose_at_distance(tr, .03), failure_reason="collision")
    assert result.terminated and not result.truncated
    assert result.potential_after == 0
    assert best_reward(result) == 0
    assert result.components["failure"] == -tr.trace.reward.failure_cost
    assert result.components["s1_success"] == 0


@pytest.mark.parametrize("succeed", [False, True])
def test_sampling_cuts_preserve_bootstrap_potential_and_real_task_outcome(succeed):
    tr = trace(stages=replace(StageTrainingConfig(), s1_budget_s=.2))
    pose = tr.pose(x=0., z=.01, opening=.7) if succeed else pose_at_distance(tr, .0854)
    while not tr.trace.info["stage_cut"]:
        result = tr.step(**pose)
    assert result.truncated and not result.terminated
    assert result.potential_after > 0
    assert tr.trace.snapshot.outcome == "running"
    assert tr.trace.info["stage_result"] == ("SUCCESS" if succeed else "TIMEOUT")
    assert result.components["s1_success"] == (10. if succeed else 0.)
    assert result.components["failure"] == 0


def test_default_non_potential_cost_budget_is_less_than_real_failure_cost():
    tr = trace()
    result = tr.step(**pose_at_distance(tr, .0854), left_contact=True)
    cost = sum(result.components[name] for name in (
        "s1_time", "s1_remaining", "s1_distance_remaining"))
    assert cost == pytest.approx(-.73 * DT)
    assert -.73 * DT <= cost + 1e-12 <= 0
    ten_seconds_budget = .73 * tr.trace.config.s1_budget_s
    assert ten_seconds_budget == pytest.approx(7.3)
    assert ten_seconds_budget < tr.trace.reward.failure_cost


def test_schema_two_critic_exposes_best_record_and_eligibility_and_resets_them():
    tr = trace()
    initial = tr.reward.critic_context()
    assert initial.shape == (14,)
    assert np.isfinite(initial).all()
    assert initial[-1] == 1
    tr.step(**pose_at_distance(tr, .0854))
    progressed = tr.reward.critic_context()
    assert progressed[-2] > initial[-2]
    tr.step(**pose_at_distance(tr, .08), left_contact=True)
    ineligible = tr.reward.critic_context()
    assert ineligible[-1] == 0
    assert ineligible[-2] >= progressed[-2]
    ineligible[:] = math.nan
    assert np.isfinite(tr.reward.critic_context()).all()
    fresh = StageTrace()
    tr.reward.reset(fresh.snapshot, fresh.info)
    np.testing.assert_array_equal(tr.reward.critic_context(), initial)
