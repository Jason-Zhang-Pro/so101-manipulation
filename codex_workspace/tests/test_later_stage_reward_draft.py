"""CPU-only draft reward checks; importing this file never starts simulation.

Keep these auxiliary checks outside the formal training/test integration until
S2--S5 have a measured stage adapter. Synthetic core tests isolate accounting;
PhysicalTrace uses the real TaskStateTracker for contact/lift/release evidence.
"""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.rewards.later_stage_reward import (
    EpisodeRewardLedger,
    LaterRewardTuning,
    LaterStageReward,
    Milestone,
    StageSignals,
    live_carry,
    live_grasp,
)
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig, PickPlaceRewardInput
from so101_capstone.rewards.s2_grasp_lift_reward import S2GraspLiftReward
from so101_capstone.rewards.s3_transport_reward import S3TransportReward
from so101_capstone.rewards.s4_place_release_reward import S4PlaceReleaseReward
from so101_capstone.rewards.s5_return_home_reward import S5ReturnHomeReward
from so101_capstone.rewards.task_state import TaskMeasurements, TaskPhase, TaskSnapshot, TaskStateTracker
from so101_capstone.rewards.training_stages import StageTrainingConfig


DT = 1 / 30


def resolved_stage_config():
    return StageTrainingConfig.load(
        Path(__file__).resolve().parents[2] / "config" / "act_ppo_stages_s1_lift_completion.json")


class ToyReward(LaterStageReward):
    """Known quality and contact predicates isolate the shared accounting."""
    def __init__(self, *, stage="S2", hold_s=.1, reward_config=None, tuning=None):
        self.stage = stage
        super().__init__(reward_config or PickPlaceRewardConfig(), weights={"progress": 1.},
                         milestones=(Milestone("grasp", 3.), Milestone("finish", 5., hold_s)),
                         completion_goal="finish", tuning=tuning)

    def signals(self, snapshot):
        return StageSignals("toy", {"progress": snapshot.approach_distance_score},
                            {"grasp": live_grasp(snapshot), "finish": live_carry(snapshot)},
                            {"distance_m": snapshot.target_distance_m})


def snapshot(t=0., *, value=.1, grasp=False, lifted=False, attempt=1,
             evidence_time=None, outcome="running", events=(), identity="episode", **measurements):
    config = PickPlaceRewardConfig()
    m = TaskMeasurements(
        sim_time_s=t, evidence_sim_time_s=evidence_time,
        joint_positions_rad=config.home_joint_positions_rad, joint_velocities_rad_s=(0.,) * 6,
        grasp_position_world_m=(0., 0., .765), grasp_rotation_world=np.eye(3),
        target_name="cube", target_position_world_m=(0., 0., .765),
        target_motion_valid=True, contact_valid=True, **measurements,
    )
    return TaskSnapshot(m, identity, "pick_place", TaskPhase.GRASP_LIFT,
                        outcome=outcome, events=events, grasp_attempt_id=attempt,
                        grasp_attempt_active=grasp, grasp_valid=grasp,
                        attempt_lifted=lifted, lifted_once=lifted,
                        approach_distance_score=value)


def score(reward, before, after, **flags):
    flags.setdefault("terminated", after.outcome != "running")
    return reward.step(PickPlaceRewardInput(before, after, after.sim_time_s - before.sim_time_s, **flags))


def record_credit(result, stage="s2"):
    return sum(value for key, value in result.components.items() if key.startswith(f"{stage}_progress_band_"))


def test_subband_progress_is_immediate_but_retreat_and_revisit_cannot_pay_twice():
    reward = ToyReward()
    first = snapshot()
    reward.reset(first)
    improved = snapshot(DT, value=.1001)
    result = score(reward, first, improved)
    assert record_credit(result) == pytest.approx(.001)
    assert sum(value > 0 for key, value in result.components.items() if "_band_" in key) == 1
    back = snapshot(2 * DT, value=.05)
    retreat = score(reward, improved, back)
    assert record_credit(retreat) == 0.
    assert retreat.components["s2_potential_progress"] < 0.
    assert retreat.components["s2_regression"] < 0.
    revisit = snapshot(3 * DT, value=.1001)
    assert record_credit(score(reward, back, revisit)) == 0.
    newer = snapshot(4 * DT, value=.1002)
    assert record_credit(score(reward, revisit, newer)) == pytest.approx(.001)


def test_crossing_many_bands_only_pays_new_fraction_once():
    reward = ToyReward()
    first = snapshot(value=.1)
    reward.reset(first)
    last = snapshot(DT, value=.95)
    assert record_credit(score(reward, first, last)) == pytest.approx(8.5)
    unchanged = snapshot(2 * DT, value=.95)
    assert record_credit(score(reward, last, unchanged)) == 0.


def test_stage_completion_is_once_only_and_does_not_choose_termination():
    reward = ToyReward()
    previous = snapshot()
    reward.reset(previous)
    grasped = snapshot(DT, grasp=True)
    assert score(reward, previous, grasped).components["s2_grasp_complete"] == 3.
    previous = grasped
    completions = []
    for index in range(2, 8):
        current = snapshot(index * DT, grasp=True, lifted=True)
        result = score(reward, previous, current)
        completions.append(result)
        assert not result.terminated and not result.truncated
        assert current.outcome == "running"
        previous = current
    assert sum(row.components["s2_finish_complete"] for row in completions) == 5.
    assert sum(row.components["s2_stage_complete"] for row in completions) == 20.
    assert sum(row.stage_success_event for row in completions) == 1
    assert completions[-1].stage_complete
    assert not completions[-1].stage_success_event
    assert completions[-1].to_info()["reward_stage_complete"]


def test_same_physical_episode_reset_and_shared_ledger_preserve_paid_bonuses():
    reward = ToyReward(hold_s=0.)
    initial = snapshot(value=0.)
    ledger = EpisodeRewardLedger.for_snapshot(initial)
    reward.reset(initial, ledger=ledger)
    reached = snapshot(DT, value=.9, grasp=True, lifted=True)
    assert score(reward, initial, reached).stage_success_event
    stored_paid, stored_best = set(ledger.paid), dict(ledger.best)
    # Require an explicit ledger choice even when worlds reuse an episode name.
    dropped = snapshot(2 * DT, value=.2, attempt=1)
    with pytest.raises(ValueError, match="explicit ledger"):
        reward.reset(dropped)
    reward.reset(dropped, ledger=ledger)
    assert reward.ledger is ledger
    recovered = snapshot(3 * DT, value=.9, grasp=True, lifted=True, attempt=2)
    replayed = score(reward, dropped, recovered)
    assert record_credit(replayed) == 0.
    assert not replayed.stage_success_event
    assert replayed.components["s2_grasp_complete"] == 0.
    assert replayed.components["s2_finish_complete"] == 0.
    # A separate reward object for a stage revisit receives the same ledger.
    replacement = ToyReward(hold_s=0.)
    replacement.reset(dropped, ledger=ledger)
    assert not score(replacement, dropped, recovered).stage_success_event
    assert ledger.paid == stored_paid and ledger.best == stored_best


def test_drop_penalty_shared_across_stages_and_regrasp_does_not_repeat_grasp_bonus():
    reward = ToyReward()
    before = snapshot()
    ledger = EpisodeRewardLedger.for_snapshot(before)
    reward.reset(before, ledger=ledger)
    held = snapshot(DT, grasp=True, lifted=True)
    assert score(reward, before, held).components["s2_grasp_complete"] == 3.
    dropped = snapshot(2 * DT, events=("grasp_lost",))
    first_drop = score(reward, held, dropped)
    assert first_drop.components["s2_grasp_loss"] == -reward.tuning.drop_cost
    repeated = snapshot(3 * DT, events=("grasp_lost",))
    assert score(reward, dropped, repeated).components["s2_grasp_loss"] == 0.
    later_stage = ToyReward(stage="S3")
    later_stage.reset(held, ledger=ledger)
    assert score(later_stage, held, dropped).components["s3_grasp_loss"] == 0.
    regrasp = snapshot(4 * DT, grasp=True, attempt=2)
    assert score(reward, repeated, regrasp).components["s2_grasp_complete"] == 0.


def test_legal_release_does_not_pay_drop_penalty():
    reward = ToyReward()
    before = snapshot(grasp=True, lifted=True)
    reward.reset(before)
    after = snapshot(DT, events=("grasp_lost", "valid_release"))
    assert score(reward, before, after).components["s2_grasp_loss"] == 0.


@pytest.mark.parametrize("field", ["evidence_valid", "joint_measurement_valid", "target_measurement_valid",
                                   "target_motion_valid", "contact_valid"])
def test_invalid_sensor_evidence_rejects_transition_without_changing_ledger(field):
    reward = ToyReward()
    before = snapshot(grasp=True, lifted=True)
    reward.reset(before)
    state = reward.reward_state()
    valid = snapshot(DT, events=("grasp_lost",))
    invalid = replace(valid, measurements=replace(valid.measurements, **{field: False}))
    with pytest.raises(ValueError, match="invalid sensor evidence"):
        score(reward, before, invalid)
    assert reward.reward_state() == state
    # The same real transition is still accepted when its evidence is corrected.
    assert score(reward, before, valid).components["s2_grasp_loss"] == -reward.tuning.drop_cost


def test_proxy_and_stale_evidence_are_not_treated_as_real_drops():
    with pytest.raises(ValueError, match="real contact"):
        ToyReward(reward_config=PickPlaceRewardConfig(grasp_evidence_mode="proxy"))
    reward = ToyReward()
    before = snapshot()
    reward.reset(before)
    proxy = snapshot(DT, grasp_evidence_mode="proxy")
    with pytest.raises(ValueError, match="invalid sensor evidence"):
        score(reward, before, proxy)
    stale = snapshot(.2, evidence_time=0.)
    with pytest.raises(ValueError, match="stale"):
        score(reward, before, stale)


def test_repeated_sensor_timestamp_does_not_accumulate_dwell_or_records():
    reward = ToyReward(hold_s=.1)
    before = snapshot()
    reward.reset(before)
    candidate = snapshot(DT, grasp=True, lifted=True)
    score(reward, before, candidate)
    repeated = snapshot(2 * DT, evidence_time=DT, grasp=True, lifted=True, value=.9)
    duplicate_result = score(reward, candidate, repeated)
    assert duplicate_result.metrics["finish_hold_s"] == 0.
    assert record_credit(duplicate_result) == 0.
    fresh = snapshot(3 * DT, grasp=True, lifted=True)
    resumed = score(reward, repeated, fresh)
    assert resumed.metrics["finish_hold_s"] == pytest.approx(DT)
    assert not resumed.stage_complete


def test_pre_stage_evidence_cannot_add_hold_and_small_new_hold_slices_pay_once():
    reward = ToyReward(hold_s=.1)
    before = snapshot(1., evidence_time=.94, grasp=True, lifted=True)
    reward.reset(before)
    pre_stage = snapshot(1. + DT, evidence_time=.94 + DT, grasp=True, lifted=True)
    first = score(reward, before, pre_stage)
    assert first.metrics["finish_hold_s"] == 0.
    assert first.components["s2_finish_dwell"] == 0.
    first_current = snapshot(1. + 2 * DT, evidence_time=1. + DT, grasp=True, lifted=True)
    earned = score(reward, pre_stage, first_current)
    assert earned.metrics["finish_hold_s"] == pytest.approx(DT)
    assert earned.components["s2_finish_dwell"] == pytest.approx(reward.tuning.hold_progress_bonus / 3.)
    lost = snapshot(1. + 3 * DT, grasp=False, lifted=False)
    score(reward, first_current, lost)
    regained = snapshot(1. + 4 * DT, grasp=True, lifted=True)
    assert score(reward, lost, regained).components["s2_finish_dwell"] == 0.
    held = snapshot(1. + 5 * DT, grasp=True, lifted=True)
    assert score(reward, regained, held).components["s2_finish_dwell"] == pytest.approx(0., abs=1e-12)


def test_evidence_gap_is_new_baseline_without_deferred_records_or_instant_goal_bonus():
    reward = ToyReward(hold_s=0.)
    before = snapshot(value=.1)
    reward.reset(before)
    unseen = snapshot(.2, value=.8, grasp=True, lifted=True)
    gap_result = score(reward, before, unseen)
    assert record_credit(gap_result) == 0.
    assert gap_result.components["s2_grasp_complete"] == 0.
    assert gap_result.components["s2_finish_complete"] == 0.
    assert not gap_result.stage_success_event
    after = snapshot(.2 + DT, value=.8, grasp=True, lifted=True)
    result = score(reward, unseen, after)
    assert record_credit(result) == 0.
    assert result.components["s2_grasp_complete"] == 0.
    assert result.components["s2_finish_complete"] == 0.
    # Unseen completion must not get laundered into a later fresh-frame bonus.
    assert result.components["s2_stage_complete"] == 0.


def test_assisted_complete_reset_is_baseline_without_completion_bonus():
    reward = ToyReward(hold_s=0.)
    before = snapshot(grasp=True, lifted=True, value=.9)
    reward.reset(before)
    result = score(reward, before, snapshot(DT, grasp=True, lifted=True, value=.9))
    assert result.stage_complete
    assert not result.stage_success_event
    assert result.components["s2_stage_complete"] == 0.
    assert result.components["s2_finish_complete"] == 0.


@pytest.mark.parametrize("outcome", ["task_failure", "success"])
def test_real_terminal_zeroes_potential_and_closes_reward(outcome):
    reward = ToyReward()
    before = snapshot(value=.6)
    reward.reset(before)
    terminal = snapshot(DT, value=.8, outcome=outcome)
    result = score(reward, before, terminal)
    assert result.terminated and not result.truncated
    assert result.potential_after == 0.
    assert result.components["s2_potential_progress"] == -result.potential_before
    if outcome == "task_failure":
        assert result.components["s2_failure"] == -reward.tuning.failure_cost
        assert record_credit(result) == 0.
        assert not result.stage_complete
    with pytest.raises(RuntimeError, match="reset"):
        score(reward, terminal, snapshot(2 * DT))


def test_sampling_cut_retains_potential_without_converting_to_physical_success():
    reward = ToyReward()
    before = snapshot(value=.3)
    reward.reset(before)
    after = snapshot(DT, value=.4)
    result = score(reward, before, after, truncated=True)
    assert result.truncated and not result.terminated
    assert result.potential_after == pytest.approx(4.)
    assert after.outcome == "running"
    with pytest.raises(RuntimeError, match="reset"):
        score(reward, after, snapshot(2 * DT))


def test_identity_time_and_outcome_contracts_fail_before_accounting():
    reward = ToyReward()
    before = snapshot()
    reward.reset(before)
    state = reward.reward_state()
    with pytest.raises(ValueError, match="physical task identity"):
        score(reward, before, snapshot(DT, identity="other"))
    with pytest.raises(ValueError, match="dt must match"):
        reward.step(PickPlaceRewardInput(before, snapshot(DT), DT * 2))
    with pytest.raises(ValueError, match="termination must match"):
        score(reward, before, snapshot(DT), terminated=True)
    with pytest.raises(ValueError, match="terminated and truncated"):
        score(reward, before, snapshot(DT, outcome="success"), terminated=True, truncated=True)
    with pytest.raises(ValueError, match="different physical episode"):
        reward.reset(before, ledger=EpisodeRewardLedger(("wrong", "pick_place", "cube")))
    assert reward.reward_state() == state


def test_critic_reward_state_is_a_copy_and_includes_holds_best_records_and_paid_milestones():
    reward = ToyReward()
    before = snapshot()
    reward.reset(before)
    after = snapshot(DT, grasp=True, value=.3)
    score(reward, before, after)
    exported = reward.reward_state()
    assert exported["best"]["s2_progress"] == .3
    assert "s2_grasp_complete" in exported["paid"]
    exported["best"]["s2_progress"] = 999.
    exported["dwell_s"]["finish"] = 999.
    exported["paid"].clear()
    assert reward.reward_state()["best"]["s2_progress"] == .3
    assert reward.reward_state()["dwell_s"]["finish"] == 0.
    assert "s2_grasp_complete" in reward.reward_state()["paid"]


@pytest.mark.parametrize("changes", [{"quality_bands": 2.5}, {"quality_bands": True},
                                      {"time_cost_per_s": -1.}, {"potential_max": math.nan}])
def test_invalid_core_tuning_is_rejected(changes):
    with pytest.raises(ValueError):
        LaterRewardTuning(**changes)


class PhysicalTrace:
    """Minimal measured trace; no imports from the formal test suite or ROS."""
    def __init__(self, config=None):
        self.config = config or PickPlaceRewardConfig()
        self.t = 0.
        self.position = (0., 0., .765)
        self.tracker = TaskStateTracker(self.config)
        self.snapshot = self.tracker.reset(self.measurement(), "pick_place", task_instance_id="physical_trace")

    def measurement(self, **overrides):
        values = dict(sim_time_s=self.t, joint_positions_rad=self.config.home_joint_positions_rad,
                      joint_velocities_rad_s=(0.,) * 6, grasp_position_world_m=(0., 0., 1.1),
                      grasp_rotation_world=np.eye(3), target_name="cube", target_position_world_m=self.position,
                      target_motion_valid=True, contact_valid=True,
                      goal_center_world_m=(.25, 0., .753), goal_half_extents_xy_m=(.05, .05),
                      grasp_evidence_mode="contact")
        values.update(overrides)
        return TaskMeasurements(**values)

    def step(self, position=None, *, contact=False, at_home=False, dt=DT, **overrides):
        if position is not None:
            self.position = position
        self.t += dt
        q = list(self.config.home_joint_positions_rad)
        if not at_home:
            q[0] += .3
        values = dict(grasp_position_world_m=self.position if contact else (0., 0., 1.1),
                      left_contact=contact, right_contact=contact, contacts_simultaneous=contact,
                      joint_positions_rad=q)
        values.update(overrides)
        self.snapshot = self.tracker.update(self.measurement(**values))
        return self.snapshot


def test_real_tracker_contact_lift_drop_trace_preserves_once_only_grasp_accounting():
    trace = PhysicalTrace()
    reward = ToyReward(hold_s=0.)
    previous = trace.snapshot
    reward.reset(previous)
    results = []
    for index in range(10):
        after = trace.step((0., 0., .765 if index < 5 else .805), contact=True)
        results.append(score(reward, previous, after))
        previous = after
    assert trace.snapshot.grasp_valid and trace.snapshot.attempt_lifted
    assert sum(result.components["s2_grasp_complete"] for result in results) == 3.
    assert sum(result.stage_success_event for result in results) == 1
    dropped = trace.step(contact=False)
    loss = score(reward, previous, dropped)
    assert "grasp_lost" in dropped.events
    assert loss.components["s2_grasp_loss"] == -reward.tuning.drop_cost
    previous = dropped
    for _ in range(5):
        after = trace.step(contact=True)
        result = score(reward, previous, after)
        assert result.components["s2_grasp_complete"] == 0.
        assert result.components["s2_stage_complete"] == 0.
        previous = after


def test_actual_s2_s3_s4_s5_physical_trace_and_one_shared_episode_ledger():
    trace = PhysicalTrace()
    ledger = EpisodeRewardLedger.for_snapshot(trace.snapshot)

    def run(reward, rows):
        previous = trace.snapshot
        reward.reset(previous, ledger=ledger)
        results = []
        for kwargs in rows:
            after = trace.step(**kwargs)
            results.append(score(reward, previous, after))
            previous = after
        assert reward.ledger is ledger
        assert sum(result.stage_success_event for result in results) == 1
        assert results[-1].stage_complete
        assert all(math.isfinite(result.reward) for result in results)
        assert all(result.reward == pytest.approx(sum(result.components.values())) for result in results)
        return results

    s2 = S2GraspLiftReward(trace.config)
    grasp_lift = run(s2, [dict(position=(0., 0., .765), contact=True)] * 5
                        + [dict(position=(0., 0., .815), contact=True)] * 16)
    assert trace.snapshot.grasp_valid and trace.snapshot.attempt_lifted
    assert sum(row.components["s2_grasped_complete"] for row in grasp_lift) == s2.config.grasp_bonus
    assert sum(row.components["s2_lifted_complete"] for row in grasp_lift) == s2.config.lift_bonus
    assert all(not row.terminated and not row.truncated for row in grasp_lift)

    s3 = S3TransportReward(trace.config)
    transported = run(s3, [dict(position=(x, 0., .805), contact=True)
                          for x in (.05, .10, .15, .20, .25)]
                         + [dict(contact=True)] * 12)
    assert trace.snapshot.carried_into_goal and trace.snapshot.grasp_valid
    assert sum(row.components["s3_arrived_complete"] for row in transported) == s3.config.arrival_bonus
    assert transported[-1].subphase == "HOLD_INSIDE_GOAL"

    s4 = S4PlaceReleaseReward(trace.config)
    placed = run(s4, [dict(position=(.25, 0., .788), contact=True),
                     dict(position=(.25, 0., .775), contact=True),
                     dict(position=(.25, 0., .768), contact=True), dict(contact=False)]
                    + [dict(contact=False)] * 15)
    assert trace.snapshot.valid_release and trace.snapshot.delivery_valid and trace.snapshot.placed_stable
    assert sum(row.components["s4_released_complete"] for row in placed) == s4.config.released_bonus
    assert sum(row.components["s4_grasp_loss"] for row in placed) == 0.
    assert all(not row.terminated for row in placed)

    s5 = S5ReturnHomeReward(trace.config, stage_config=resolved_stage_config())
    s5.reset(trace.snapshot, ledger=ledger)
    previous = trace.snapshot
    returning = []
    for error in (.2, .1, .05, .03, .01):
        positions = list(trace.config.home_joint_positions_rad)
        positions[0] += error
        current = trace.step(joint_positions_rad=positions)
        returning.append(score(s5, previous, current))
        previous = current
    for _ in range(120):
        current = trace.step(at_home=True)
        returning.append(score(s5, previous, current))
        previous = current
        if current.outcome != "running":
            break
    assert trace.snapshot.outcome == "success" and trace.snapshot.success_event
    assert returning[-1].stage_complete and returning[-1].terminated
    assert returning[-1].potential_after == 0.
    assert sum(row.stage_success_event for row in returning) == 1
    assert sum(row.components["s5_home_held_complete"] for row in returning) == s5.config.home_held_bonus
    assert all(f"s{stage}_stage_complete" in ledger.paid for stage in range(2, 6))


@pytest.mark.parametrize("reward_type", [S3TransportReward, S4PlaceReleaseReward, S5ReturnHomeReward])
def test_pushed_into_goal_object_without_carry_provenance_cannot_unlock_later_goals(reward_type):
    trace = PhysicalTrace()
    kwargs = {"stage_config": resolved_stage_config()} if reward_type is S5ReturnHomeReward else {}
    reward = reward_type(trace.config, **kwargs)
    before = trace.snapshot
    reward.reset(before)
    result = None
    for _ in range(15):
        current = trace.step(position=(.25, 0., .768), contact=False, at_home=True)
        result = score(reward, before, current)
        assert not result.stage_complete and not result.stage_success_event
        assert not any(reward.signals(current).goals.values())
        before = current
    assert result is not None
    assert not trace.snapshot.delivery_valid and not trace.snapshot.valid_release


def test_s5_joint_velocity_does_not_break_position_only_home_hold():
    trace = PhysicalTrace()
    for _ in range(5):
        trace.step(contact=True)
    for _ in range(5):
        trace.step(position=(0., 0., .805), contact=True)
    for _ in range(3):
        trace.step(position=(.25, 0., .805), contact=True)
    trace.step(position=(.25, 0., .768), contact=True)
    for _ in range(16):
        trace.step(contact=False)
    assert trace.snapshot.placed_stable
    reward = S5ReturnHomeReward(trace.config, stage_config=resolved_stage_config())
    previous = trace.snapshot
    reward.reset(previous)
    for _ in range(15):
        current = trace.step(at_home=True)
        result = score(reward, previous, current)
        previous = current
    held_before = result.metrics["home_held_hold_s"]
    assert held_before > 0.
    velocities = [0.] * 6
    velocities[-1] = .1
    current = trace.step(at_home=True, joint_velocities_rad_s=velocities)
    moving = score(reward, previous, current)
    assert current.robot_at_home and not current.robot_static
    assert current.home_stable_duration_s == 0.
    assert moving.metrics["home_held_hold_s"] == pytest.approx(held_before + DT)
    assert moving.subphase == "HOME_HOLD"
    assert not moving.stage_complete
