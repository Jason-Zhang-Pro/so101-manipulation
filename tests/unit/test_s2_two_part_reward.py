"""Offline S2 CLOSE/LIFT contracts using measured TaskStateTracker evidence."""
from dataclasses import replace

import numpy as np
import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig, PickPlaceRewardInput
from so101_capstone.rewards.s2_grasp_lift_reward import S2GraspLiftReward, S2RewardConfig
from so101_capstone.rewards.task_state import TaskMeasurements, TaskStateTracker


DT = 1 / 30


class S2Trace:
    def __init__(self, *, initial_height=.765, reward_config=None, stage_config=None):
        self.config = reward_config or PickPlaceRewardConfig()
        self.time = 0.
        self.height = initial_height
        self.tracker = TaskStateTracker(self.config)
        self.snapshot = self.tracker.reset(self.measurement(offset=(0., 0., .018)),
                                           task_instance_id="s2_two_part")
        self.reward = S2GraspLiftReward(self.config, stage_config)
        self.reward.reset(self.snapshot)

    def measurement(self, *, offset=(0., 0., 0.), opening=0., contact=False, **overrides):
        joints = list(self.config.home_joint_positions_rad)
        joints[-1] += opening
        target = np.asarray((0., 0., self.height))
        values = dict(sim_time_s=self.time, joint_positions_rad=joints,
                      joint_velocities_rad_s=(0.,) * 6,
                      grasp_position_world_m=target + np.asarray(offset),
                      grasp_rotation_world=np.diag((1., -1., -1.)),
                      target_name="cube", target_position_world_m=target,
                      target_motion_valid=True, contact_valid=True,
                      left_contact=contact, right_contact=contact, contacts_simultaneous=contact)
        values.update(overrides)
        return TaskMeasurements(**values)

    def step(self, *, height=None, dt=DT, **measurements):
        previous = self.snapshot
        self.time += dt
        if height is not None:
            self.height = height
        self.snapshot = self.tracker.update(self.measurement(**measurements))
        return self.reward.step(PickPlaceRewardInput(
            previous, self.snapshot, dt, terminated=self.snapshot.outcome != "running"))

    def close(self):
        return [self.step(contact=True) for _ in range(3)]

    def lift(self, *, height=.815, count=4, **measurements):
        return [self.step(height=height, contact=True, **measurements) for _ in range(count)]


def test_close_fine_alignment_contact_and_lift_are_two_parts_with_separate_bonuses():
    tr = S2Trace()
    aligned = tr.step(offset=(0., 0., .01), opening=.6)
    assert aligned.subphase == "CLOSE" and aligned.components["s2_aligned_complete"] == 3.
    closing = tr.close()
    assert [row.subphase for row in closing] == ["CLOSE", "CLOSE", "LIFT"]
    assert closing[0].components["s2_dual_contact_complete"] == 2.
    assert closing[-1].components["s2_grasped_complete"] == 5.
    assert closing[-1].metrics["close_complete"] == 1.
    assert all(not row.stage_complete for row in closing)
    # The first qualifying height sample starts the stable interval. Three
    # further 30 Hz intervals finish it, without a prior tracker-lift delay.
    lifting = tr.lift()
    assert all(row.subphase == "LIFT" for row in lifting)
    assert all(not row.stage_complete for row in lifting[:-1])
    assert lifting[-1].stage_complete and lifting[-1].stage_success_event
    assert lifting[0].metrics["lift_height_confirmed"] == 1.
    assert lifting[1].metrics["stable_hold_hold_s"] == pytest.approx(DT)
    assert lifting[-1].metrics["stable_hold_hold_s"] == pytest.approx(.1)
    rows = [aligned, *closing, *lifting]
    for key, expected in (("grasped", 5.), ("lifted", 5.), ("stable_hold", 5.), ("stage", 20.)):
        assert sum(row.components[f"s2_{key}_complete"] for row in rows) == expected
    assert not any(row.terminated or row.truncated for row in rows)


@pytest.mark.parametrize("evidence", [
    {},
    {"left_contact": True},
    {"left_contact": True, "right_contact": True, "contacts_simultaneous": False},
])
def test_closed_angle_or_unconfirmed_contact_cannot_finish_close(evidence):
    tr = S2Trace()
    for _ in range(20):
        row = tr.step(opening=0., **evidence)
        assert row.subphase == "CLOSE"
        assert not tr.snapshot.grasp_valid and not tr.snapshot.grasp_attempt_active
        assert row.metrics["close_complete"] == 0.
        assert row.components["s2_grasped_complete"] == row.components["s2_stage_complete"] == 0.
        assert not row.stage_complete


@pytest.mark.parametrize("condition", [
    {"offset": (.0051, 0., 0.)},
    {"target_linear_velocity_m_s": (.0151, 0., 0.)},
    {"target_angular_velocity_rad_s": (0., 0., .1501)},
])
def test_lift_requires_slip_and_speed_limits_for_the_entire_hold(condition):
    tr = S2Trace()
    tr.close()
    rows = tr.lift(count=18, **condition)
    assert tr.snapshot.grasp_valid and tr.snapshot.attempt_lifted
    assert all(row.subphase == "LIFT" and not row.stage_complete for row in rows)
    assert rows[-1].metrics["stable_hold_hold_s"] == 0.
    recovered = tr.lift(count=4)
    assert not any(row.stage_complete for row in recovered[:-1])
    assert recovered[-1].stage_complete


def test_exact_stability_thresholds_are_allowed():
    tr = S2Trace()
    tr.close()
    row = tr.lift(offset=(.005, 0., 0.),
                  target_linear_velocity_m_s=(.015, 0., 0.),
                  target_angular_velocity_rad_s=(0., 0., .15))[-1]
    assert row.stage_complete and row.metrics["lift_stable_candidate"] == 1.


def test_current_object_height_and_new_attempt_rise_are_both_required():
    tr = S2Trace(initial_height=.795)
    tr.close()
    for row in tr.lift(height=.805, count=18):
        assert row.subphase == "LIFT" and not row.stage_complete
        assert row.metrics["object_clearance_m"] > .03
        assert row.metrics["attempt_rise_m"] < .05
    assert tr.lift(height=.845)[-1].stage_complete
    # A latched lift history cannot excuse losing either current height gate.
    lowered = tr.step(height=.794, contact=True)
    assert tr.snapshot.attempt_lifted and lowered.subphase == "LIFT"
    assert not lowered.stage_complete and lowered.metrics["lift_stable_candidate"] == 0.


def test_drop_returns_to_close_and_recovery_cannot_repeat_part_or_stage_bonuses():
    tr = S2Trace()
    first = [*tr.close(), *tr.lift()]
    assert first[-1].stage_complete
    old_attempt = tr.snapshot.grasp_attempt_id
    loss = tr.step(contact=False)
    assert loss.subphase == "CLOSE" and not loss.stage_complete
    assert not tr.snapshot.grasp_valid and not tr.snapshot.attempt_lifted
    assert loss.components["s2_grasp_loss"] == -5.
    assert tr.step(contact=False).components["s2_grasp_loss"] == 0.
    recovered_close = tr.close()
    assert tr.snapshot.grasp_attempt_id > old_attempt
    assert recovered_close[-1].subphase == "LIFT"
    assert not tr.snapshot.attempt_lifted  # Raised pose alone is insufficient.
    held_at_old_height = tr.lift(count=15)
    assert not any(row.stage_complete for row in held_at_old_height)
    recovered_lift = tr.lift(height=.865)
    assert recovered_lift[-1].stage_complete
    for row in [*recovered_close, *held_at_old_height, *recovered_lift]:
        assert not row.stage_success_event
        for goal in ("grasped", "lifted", "stable_hold", "stage"):
            assert row.components[f"s2_{goal}_complete"] == 0.


def test_prepared_grasp_and_lift_are_baselines_not_paid_policy_achievements():
    tr = S2Trace()
    tr.close()
    tr.lift()
    # A fresh reward beginning after assisted preparation gets no backdated
    # grasp/lift bonus; its own fresh .1 s stable dwell remains to be proved.
    tr.reward = S2GraspLiftReward(tr.config)
    tr.reward.reset(tr.snapshot)
    assert tr.reward.reward_state()["phase"] == "LIFT"
    assert tr.reward.reward_state()["dwell_s"]["stable_hold"] == 0.
    rows = tr.lift(count=3)
    for row in rows:
        for goal in ("aligned", "dual_contact", "grasped", "lifted"):
            assert row.components[f"s2_{goal}_complete"] == 0.
    assert not any(row.stage_complete for row in rows[:-1])
    assert rows[-1].stage_complete  # Earned by post-preparation fresh holding.


def test_close_retains_twenty_continuous_progress_bands_without_revisit_credit():
    tr = S2Trace()
    assert tr.reward.tuning.quality_bands == 20
    better = tr.step(offset=(0., 0., .017999), opening=.6)
    keys = [key for key in better.components if key.startswith("s2_centering_band_")]
    assert len(keys) == 20
    assert 0 < sum(better.components[key] for key in keys) < .01
    tr.step(offset=(0., 0., .02), opening=.6)
    revisit = tr.step(offset=(0., 0., .017999), opening=.6)
    assert sum(revisit.components[key] for key in keys) == 0.


@pytest.mark.parametrize("height,complete", [(.814999, False), (.815, True), (.815001, True)])
def test_five_centimetre_attempt_rise_boundary(height, complete):
    tr = S2Trace()
    tr.close()
    rows = tr.lift(height=height, count=5)
    assert rows[-1].metrics["object_clearance_m"] > .03
    assert rows[-1].stage_complete is complete


@pytest.mark.parametrize("height,complete", [(.794999, False), (.795, True), (.795001, True)])
def test_three_centimetre_clearance_is_an_independent_boundary(height, complete):
    # Isolate the absolute gate with an acquisition anchor below the support
    # plane. This is a predicate fixture, not a proposed simulation start.
    tr = S2Trace(initial_height=.744)
    tr.close()
    rows = tr.lift(height=height, count=5)
    assert rows[-1].metrics["attempt_rise_m"] > .05
    assert rows[-1].stage_complete is complete


def test_s2_does_not_wait_for_shared_tracker_lift_confirmation_or_threshold():
    config = replace(PickPlaceRewardConfig(), lift_clearance_m=.08, lift_confirm_s=.5)
    tr = S2Trace(reward_config=config)
    tr.close()
    rows = tr.lift()
    assert not tr.snapshot.attempt_lifted
    assert rows[0].metrics["lift_stable_candidate"] == 1.
    assert rows[1].metrics["stable_hold_hold_s"] == pytest.approx(DT)
    assert rows[-1].stage_complete


def test_lift_progress_uses_separately_normalized_clearance_and_attempt_rise():
    tr = S2Trace()
    tr.close()
    tr.step(height=.78, contact=True)  # 1.5 cm: half the clearance, 30% of rise.
    signals = tr.reward.signals(tr.snapshot)
    assert signals.metrics["object_clearance_fraction"] == pytest.approx(.5)
    assert signals.metrics["attempt_rise_fraction"] == pytest.approx(.3)
    assert signals.qualities["lift"] == pytest.approx(.4)
    assert not signals.goals["lifted"]
    tr.step(height=.795, contact=True)
    signals = tr.reward.signals(tr.snapshot)
    assert signals.metrics["object_clearance_fraction"] == pytest.approx(1.)
    assert signals.metrics["attempt_rise_fraction"] == pytest.approx(.6)
    assert signals.qualities["lift"] == pytest.approx(.8)
    assert not signals.goals["lifted"]


@pytest.mark.parametrize("interruption", [
    {"height": .814},
    {"offset": (.0051, 0., 0.)},
    {"target_linear_velocity_m_s": (.0151, 0., 0.)},
    {"target_angular_velocity_rad_s": (0., 0., .1501)},
])
def test_any_failed_stability_condition_restarts_the_continuous_tenth_second(interruption):
    tr = S2Trace()
    tr.close()
    partial = tr.lift(count=2)
    assert partial[-1].metrics["stable_hold_hold_s"] == pytest.approx(DT)
    broken = tr.step(contact=True, **interruption)
    assert broken.metrics["stable_hold_hold_s"] == 0.
    after = tr.lift()
    assert not any(row.stage_complete for row in after[:-1])
    assert after[-1].stage_complete


def test_duplicate_evidence_cannot_fill_or_later_backfill_stable_hold():
    tr = S2Trace()
    tr.close()
    tr.lift(count=2)
    stamp = tr.snapshot.evidence_sim_time_s
    duplicate = tr.step(contact=True, evidence_sim_time_s=stamp)
    assert duplicate.metrics["stable_hold_hold_s"] == pytest.approx(DT)
    assert not duplicate.stage_complete
    fresh = tr.step(contact=True)
    assert fresh.metrics["stable_hold_hold_s"] == pytest.approx(2 * DT)
    assert not fresh.stage_complete
    assert tr.step(contact=True).stage_complete


@pytest.mark.parametrize("name", ["lift_clearance_m", "lift_rise_m", "stable_hold_s"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_independent_s2_thresholds_reject_invalid_values(name, value):
    with pytest.raises(ValueError):
        S2RewardConfig(**{name: value})
