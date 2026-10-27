"""Measured S1 trajectories exercise incremental rewards and evidence guards."""

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import PickPlaceRewardInput
from so101_capstone.rewards.stage_process_reward import (
    StageProcessRewardConfig,
    StageReachReward,
)
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_training_stages import StageTrace


DT = 1 / 30
COMPONENTS = (
    "distance_coarse", "distance_fine", "xy", "height", "bearing", "axis",
    "opening", "combined", "hold",
)


class RewardTrace:
    """Use the existing measurement/state/stage path, not reward-private scores."""

    def __init__(self, *, config=None, stages=None, reward_config=None):
        self.trace = StageTrace(config=stages, reward_config=reward_config)
        self.config = config or StageProcessRewardConfig()
        self.reward = StageReachReward(self.config, self.trace.config, self.trace.reward)
        self.reward.reset(self.trace.snapshot, self.trace.info)

    def pose(self, *, x=.08, y=0., z=.05, tilt=0., opening=.7):
        angle = math.radians(tilt)
        rotation_y = np.array([[math.cos(angle), 0., math.sin(angle)],
                               [0., 1., 0.],
                               [-math.sin(angle), 0., math.cos(angle)]])
        joints = list(self.trace.reward.home_joint_positions_rad)
        joints[-1] += opening
        return {
            "grasp_position_world_m": (x, y, .765 + z),
            "grasp_rotation_world": rotation_y @ np.diag((1., -1., -1.)),
            "joint_positions_rad": joints,
        }

    def step(self, *, dt=DT, **measurements):
        previous = self.trace.snapshot
        info = self.trace.step(dt=dt, **measurements)
        transition = PickPlaceRewardInput(
            previous, self.trace.snapshot, dt,
            terminated=self.trace.snapshot.outcome != "running",
            truncated=bool(info["stage_cut"]),
        )
        return self.reward.step(transition, info)


def progress(result, name):
    return result.components[f"s1_{name}_progress"]


def test_config_is_frozen_and_has_finite_positive_reference_scales():
    config = StageProcessRewardConfig()
    assert config.schema_version == 1
    assert config.s1_potential_weights == (20, 20, 15, 10, 5, 5, 10, 10, 5)
    assert config.s1_time_cost_per_s == pytest.approx(.03)
    assert config.s1_remaining_cost_per_s == pytest.approx(.3)
    assert config.s1_success_bonus == pytest.approx(10)
    with pytest.raises(FrozenInstanceError):
        config.s1_time_cost_per_s = 1


@pytest.mark.parametrize("values", [
    {"schema_version": 2}, {"schema_version": True},
    {"s1_distance_coarse_scale_m": 0}, {"s1_distance_fine_scale_m": -1},
    {"s1_xy_scale_m": math.nan}, {"s1_height_scale_m": math.inf},
    {"s1_orientation_scale_rad": False}, {"s1_opening_scale_rad": 0},
    {"s1_soft_gate_floor": -1}, {"s1_soft_gate_floor": 1.01},
    {"s1_time_cost_per_s": -.01}, {"s1_success_bonus": math.inf},
    {"s1_remaining_cost_per_s": -.01}, {"s1_remaining_cost_per_s": math.nan},
    {"s1_remaining_cost_per_s": math.inf}, {"s1_remaining_cost_per_s": True},
    {"s1_potential_weights": (1,) * 8},
    {"s1_potential_weights": (0,) * 9},
    {"s1_potential_weights": (-1,) + (1,) * 8},
    {"s1_potential_weights": (math.nan,) + (1,) * 8},
])
def test_invalid_config_is_rejected(values):
    with pytest.raises(ValueError):
        StageProcessRewardConfig(**values)


@pytest.mark.parametrize("component,before,after", [
    ("distance_coarse", {"x": .20}, {"x": .20 - 1e-6}),
    ("distance_fine", {"x": .20}, {"x": .20 - 1e-6}),
    ("xy", {"x": .20}, {"x": .20 - 1e-6}),
    ("height", {"x": .08, "z": .08}, {"x": .08, "z": .08 - 1e-6}),
    ("bearing", {"x": 0., "z": .08, "tilt": 50.},
                {"x": 0., "z": .08, "tilt": 50. - 1e-4}),
    ("axis", {"x": 0., "z": .08, "tilt": 50.},
             {"x": 0., "z": .08, "tilt": 50. - 1e-4}),
    ("opening", {"x": .25, "opening": 1.2},
                {"x": .25, "opening": 1.2 - 1e-6}),
    ("combined", {"x": .20}, {"x": .20 - 1e-6}),
])
def test_measured_tiny_improvement_has_continuous_signal(component, before, after):
    moving, stationary = RewardTrace(), RewardTrace()
    moving.step(**moving.pose(**before))
    stationary.step(**stationary.pose(**before))
    changed = moving.step(**moving.pose(**after))
    unchanged = stationary.step(**stationary.pose(**before))
    assert progress(changed, component) > 0
    assert progress(unchanged, component) == pytest.approx(0, abs=1e-12)
    assert changed.reward > unchanged.reward


def test_far_target_does_not_disable_height_or_opening_guidance():
    trace = RewardTrace()
    trace.step(**trace.pose(x=1., z=.20, opening=1.5))
    improved = trace.step(**trace.pose(x=1., z=.1999, opening=1.4999))
    assert progress(improved, "height") > 0
    assert progress(improved, "opening") > 0


def test_stationary_state_has_no_progress_or_positive_reward():
    trace = RewardTrace()
    trace.step(**trace.pose())
    result = trace.step(**trace.pose())
    assert all(progress(result, name) == pytest.approx(0) for name in COMPONENTS)
    assert result.components["s1_discount_correction"] <= 0
    assert result.components["s1_time"] == pytest.approx(-.03 * DT)
    assert result.components["s1_success"] == 0
    assert result.reward <= 0


def test_remaining_cost_prefers_better_partial_reach_even_without_motion():
    costs = []
    for distance in (.25, .12, .04):
        trace = RewardTrace()
        pose = trace.pose(x=distance, z=.04, opening=.7)
        trace.step(**pose)
        stationary = trace.step(**pose)
        assert all(progress(stationary, name) == pytest.approx(0) for name in COMPONENTS)
        assert stationary.components["s1_success"] == 0
        costs.append(stationary.components["s1_remaining"])
    assert costs[0] < costs[1] < costs[2] <= 0


def test_remaining_cost_does_not_charge_for_unfinished_hold_or_reward_hold_twice():
    trace = RewardTrace()
    reached = trace.pose(x=0., z=.01, opening=.7)
    arrived = trace.step(**reached)
    assert trace.trace.info["stage_reach_hold_s"] == 0
    held = trace.step(**reached)
    assert trace.trace.info["stage_reach_hold_s"] > 0
    assert progress(held, "hold") > 0
    assert held.potential_after > arrived.potential_after
    assert arrived.components["s1_remaining"] == pytest.approx(0)
    assert held.components["s1_remaining"] == pytest.approx(0)


@pytest.mark.parametrize("case", ["far", "near", "contact", "outside_source"])
def test_non_potential_process_cost_has_a_small_explicit_bound(case):
    trace = RewardTrace()
    pose = trace.pose(x=.25 if case == "far" else .02)
    if case == "contact":
        pose["left_contact"] = True
    elif case == "outside_source":
        pose["target_position_world_m"] = (.08, 0., .765)
    result = trace.step(**pose)
    cost = result.components["s1_time"] + result.components["s1_remaining"]
    assert -(.03 + .3) * DT - 1e-12 <= cost <= -.03 * DT + 1e-12
    if case in ("contact", "outside_source"):
        assert result.potential_after == 0
        assert cost == pytest.approx(-.33 * DT)


def test_remaining_cost_can_be_explicitly_disabled():
    trace = RewardTrace(config=replace(StageProcessRewardConfig(), s1_remaining_cost_per_s=0))
    result = trace.step(**trace.pose(x=.3))
    assert result.components["s1_remaining"] == 0


def test_returning_from_regression_receives_progress_without_beating_old_best():
    trace = RewardTrace()
    trace.step(**trace.pose(x=.10))
    backward = trace.step(**trace.pose(x=.12))
    recovery = trace.step(**trace.pose(x=.11))
    assert progress(backward, "xy") < 0
    assert progress(recovery, "xy") > 0


@pytest.mark.parametrize("opening,expected_sign", [(1.1, 1), (.7, 0), (.2, -1)])
def test_opening_moves_toward_interval_and_does_not_reward_closing_past_it(opening, expected_sign):
    trace = RewardTrace()
    baseline = 1.2 if opening == 1.1 else .8
    trace.step(**trace.pose(x=.03, z=.015, opening=baseline))
    result = trace.step(**trace.pose(x=.03, z=.015, opening=opening))
    value = progress(result, "opening")
    if expected_sign:
        assert expected_sign * value > 0
    else:
        assert value == pytest.approx(0)


def test_opening_interval_uses_active_stage_config():
    trace = RewardTrace(stages=replace(StageTrainingConfig(),
                                      s1_opening_min_rad=1.0, s1_opening_max_rad=1.2))
    trace.step(**trace.pose(opening=1.1))
    result = trace.step(**trace.pose(opening=.7))
    assert progress(result, "opening") < 0


@pytest.mark.parametrize("bad", [
    {"left_contact": True}, {"right_contact": True},
    {"target_linear_velocity_m_s": (.1, 0., 0.)},
    {"target_angular_velocity_rad_s": (0., 0., 1.)},
    {"target_position_world_m": (.08, 0., .765)},
    {"target_position_world_m": (0., 0., .80)},
    {"goal_center_world_m": (0., 0., .753)},
])
def test_unqualified_target_or_contact_cannot_collect_reach_potential(bad):
    trace = RewardTrace()
    trace.step(**trace.pose())
    result = trace.step(**{**trace.pose(), **bad})
    assert result.potential_before > 0
    assert result.potential_after == 0
    assert result.components["s1_success"] == 0


@pytest.mark.parametrize("repetitions", [1, 3, 8])
def test_discounted_closed_loop_cannot_farm_progress(repetitions):
    trace = RewardTrace(config=replace(StageProcessRewardConfig(), s1_time_cost_per_s=0))
    initial = trace.pose(x=.12, z=.06, opening=.7)
    improved = trace.pose(x=.08, z=.05, opening=.7)
    trace.step(**initial)
    rewards = []
    for _ in range(repetitions):
        rewards.append(trace.step(**improved).reward)
        rewards.append(trace.step(**improved).reward)
        rewards.append(trace.step(**initial).reward)
    discounted = sum(trace.trace.reward.gamma**i * value for i, value in enumerate(rewards))
    assert discounted <= 1e-10


def test_fresh_hold_progress_and_stage_success_are_measured_and_awarded_once():
    trace = RewardTrace()
    reached = trace.pose(x=0., z=.01, opening=.7)
    first = trace.step(**reached)
    assert progress(first, "hold") == 0
    held = trace.step(**reached)
    assert progress(held, "hold") > 0
    assert held.components["s1_success"] == 0
    while not trace.trace.info["stage_cut"]:
        result = trace.step(**reached)
    assert trace.trace.info["stage_result"] == "SUCCESS"
    assert result.components["s1_success"] == trace.config.s1_success_bonus
    assert result.truncated and not result.terminated
    assert result.potential_after == pytest.approx(sum(trace.config.s1_potential_weights))
    assert trace.trace.snapshot.outcome == "running"
    previous = trace.trace.snapshot
    later = replace(previous, measurements=replace(
        previous.measurements, sim_time_s=previous.sim_time_s + DT,
        evidence_sim_time_s=previous.evidence_sim_time_s + DT,
    ))
    with pytest.raises(RuntimeError, match="reset required"):
        trace.reward.step(PickPlaceRewardInput(previous, later, DT), trace.trace.info)


def test_timeout_preserves_potential_and_does_not_pay_success_or_failure():
    stages = replace(StageTrainingConfig(), s1_budget_s=.2)
    trace = RewardTrace(stages=stages)
    while not trace.trace.info["stage_cut"]:
        result = trace.step(**trace.pose())
    assert trace.trace.info["stage_result"] == "TIMEOUT"
    assert result.truncated and not result.terminated
    assert result.potential_after > 0
    assert result.components["s1_success"] == 0
    assert result.components.get("failure", 0) == 0


def test_real_failure_zeroes_terminal_potential_and_keeps_failure_cost():
    trace = RewardTrace()
    trace.step(**trace.pose())
    result = trace.step(**trace.pose(), failure_reason="collision")
    assert trace.trace.info["stage_result"] == "FAILURE"
    assert result.terminated and not result.truncated
    assert result.potential_after == 0
    assert result.components["failure"] == -trace.trace.reward.failure_cost
    assert result.components["s1_success"] == 0


def test_certified_grasp_and_lift_keep_strong_downstream_success():
    trace = RewardTrace()
    jaws = trace.pose(x=0., z=0., opening=.27)
    contact = dict(left_contact=True, right_contact=True, contacts_simultaneous=True)
    for _ in range(4):
        trace.step(**jaws, **contact)
    lifted = {**jaws, "target_position_world_m": (0., 0., .805),
              "grasp_position_world_m": (0., 0., .805)}
    for _ in range(4):
        result = trace.step(**lifted, **contact)
        if trace.trace.info["stage_success"]:
            break
    assert trace.trace.info["stage_downstream_grasp_lift"]
    assert trace.trace.info["stage_result"] == "SUCCESS"
    assert result.components["s1_success"] == trace.config.s1_success_bonus
    assert result.potential_after == pytest.approx(sum(trace.config.s1_potential_weights))


def test_duplicate_evidence_freezes_potential_despite_changed_geometry():
    trace = RewardTrace()
    previous = trace.step(**trace.pose(x=.20))
    evidence_time = trace.trace.snapshot.evidence_sim_time_s
    result = trace.step(**trace.pose(x=.03), evidence_sim_time_s=evidence_time)
    assert not trace.trace.info["stage_evidence_fresh"]
    assert result.potential_after == pytest.approx(previous.potential_after)
    assert result.components["s1_remaining"] == pytest.approx(previous.components["s1_remaining"])
    assert all(progress(result, name) == pytest.approx(0) for name in COMPONENTS)
    assert result.components["s1_success"] == 0
    assert result.reward <= 0


def test_evidence_gap_rebaselines_without_paying_for_unseen_improvement():
    trace = RewardTrace()
    trace.step(**trace.pose(x=.20))
    after_gap = trace.step(dt=.2, **trace.pose(x=.08))
    assert after_gap.potential_after > after_gap.potential_before
    assert after_gap.components["s1_evidence_guard"] < 0
    assert after_gap.reward <= 0
    unchanged = trace.step(**trace.pose(x=.08))
    assert all(progress(unchanged, name) == pytest.approx(0) for name in COMPONENTS)
    improved = trace.step(**trace.pose(x=.0799))
    assert progress(improved, "xy") > 0


@pytest.mark.parametrize("flag", ["evidence_valid", "joint_measurement_valid",
                                 "target_measurement_valid", "target_motion_valid", "contact_valid"])
def test_invalid_measurement_is_rejected_at_reward_boundary(flag):
    trace = RewardTrace()
    previous = trace.trace.snapshot
    invalid = replace(previous, measurements=replace(previous.measurements,
                        sim_time_s=DT, evidence_sim_time_s=DT, **{flag: False}))
    # Call the reward boundary directly: StageEvaluator also rejects this, but
    # reward must not trust a caller merely because it supplies valid old info.
    with pytest.raises(ValueError):
        trace.reward.step(PickPlaceRewardInput(previous, invalid, DT), trace.trace.info)


def test_critic_context_is_finite_independent_and_resets_without_history_leak():
    trace = RewardTrace()
    initial = trace.reward.critic_context().copy()
    assert initial.shape == (12,)
    assert np.isfinite(initial).all()
    trace.step(**trace.pose(x=.08))
    updated = trace.reward.critic_context()
    assert updated.shape == (12,)
    assert np.isfinite(updated).all()
    assert not np.array_equal(initial, updated)
    updated[:] = math.nan
    assert np.isfinite(trace.reward.critic_context()).all()
    new_episode = StageTrace()
    trace.reward.reset(new_episode.snapshot, new_episode.info)
    np.testing.assert_array_equal(trace.reward.critic_context(), initial)
