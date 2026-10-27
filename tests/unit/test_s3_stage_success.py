"""S3 success is current carry + complete footprint for 0.1 measured second."""
import math

import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardInput
from so101_capstone.rewards.s3_transport_reward import S3RewardConfig, S3TransportReward

from test_task_state import Trace


DT = 1 / 30


class S3Trace:
    def __init__(self, *, prepare=True, config=None):
        self.trace = Trace()
        if prepare:
            self.trace.acquire_and_lift()
        self.reward = S3TransportReward(self.trace.config, config=config)
        self.reward.reset(self.trace.snapshot)

    def step(self, position=None, *, contact=True, **measurements):
        before = self.trace.snapshot
        after = self.trace.step(position, contact=contact, at_home=False, dt=DT, **measurements)
        return self.reward.step(PickPlaceRewardInput(
            before, after, DT, terminated=after.outcome != "running"))

    def hold(self, position=(.25, 0., .805), *, count=4, **measurements):
        return [self.step(position, **measurements) for _ in range(count)]


def test_whole_footprint_hold_completes_after_point_one_second_once():
    tr = S3Trace()
    rows = tr.hold()
    assert all(not row.stage_complete for row in rows[:-1])
    assert rows[-1].stage_complete and rows[-1].stage_success_event
    assert rows[-1].subphase == "HOLD_INSIDE_GOAL"
    assert rows[-1].metrics["transported_hold_s"] == pytest.approx(.1)
    assert sum(row.components["s3_transported_complete"] for row in rows) == 5.
    assert sum(row.components["s3_stage_complete"] for row in rows) == 20.
    revisit = tr.step()
    assert revisit.stage_complete and not revisit.stage_success_event
    assert revisit.components["s3_transported_complete"] == 0.


def test_success_does_not_require_centering_hover_height_or_slow_motion():
    tr = S3Trace()
    # Still within real grasp tolerance (1.5 cm), but beyond the old 5 mm
    # stability threshold. Hover height, center and speeds all miss old gates.
    rows = tr.hold((.28, 0., .90),
                   grasp_position_world_m=(.274, 0., .90),
                   target_linear_velocity_m_s=(.2, 0., 0.),
                   target_angular_velocity_rad_s=(0., 0., 1.))
    assert rows[-1].stage_complete
    assert tr.trace.snapshot.grasp_valid and tr.trace.snapshot.attempt_lifted
    assert rows[-1].metrics["object_goal_xy_m"] > .01
    assert rows[-1].metrics["goal_clearance_m"] > .05
    assert rows[-1].metrics["stable_hover_candidate"] == 0.


def test_current_height_below_old_transport_gate_can_complete_with_valid_lift_history():
    tr = S3Trace()
    row = tr.hold((.25, 0., .775))[-1]
    assert tr.trace.snapshot.grasp_valid and tr.trace.snapshot.attempt_lifted
    assert not tr.trace.snapshot.carried_into_goal  # Shared tracker keeps old height rule.
    assert row.metrics["goal_clearance_m"] < .02
    assert row.stage_complete


def test_exact_region_edge_allowed_without_extra_two_millimeter_margin():
    tr = S3Trace()
    row = tr.hold((.285, 0., .805))[-1]
    assert row.stage_complete and row.metrics["object_inside_goal"] == 1.
    assert not tr.trace.snapshot.carried_into_goal  # Tracker's margin is deliberately independent.
    assert row.metrics["footprint_overflow_m"] == pytest.approx(.002)


@pytest.mark.parametrize("position", [(.28501, 0., .805), (.25, .03501, .805)])
def test_center_inside_but_footprint_over_edge_cannot_complete(position):
    tr = S3Trace()
    rows = tr.hold(position, count=10)
    assert all(not row.stage_complete for row in rows)
    assert rows[-1].metrics["object_inside_goal"] == 0.


def test_rotated_footprint_checked_instead_of_center_or_unrotated_size():
    tr = S3Trace()
    rotated = dict(target_quaternion_xyzw=(0., 0., math.sin(math.pi / 8), math.cos(math.pi / 8)))
    assert not tr.hold((.28, 0., .805), **rotated)[-1].stage_complete
    edge_x = .25 + .05 - .015 * math.sqrt(2)
    assert tr.hold((edge_x, 0., .805), **rotated)[-1].stage_complete


def test_leaving_region_breaks_continuity_without_clearing_previous_carry():
    tr = S3Trace()
    first = tr.hold(count=3)
    assert first[-1].metrics["transported_hold_s"] == pytest.approx(2 * DT)
    outside = tr.step((.20, 0., .805))
    assert tr.trace.snapshot.carried_into_goal and not outside.stage_complete
    assert outside.metrics["transported_hold_s"] == 0.
    returned = tr.hold()
    assert all(not row.stage_complete for row in returned[:-1])
    assert returned[-1].stage_complete


def test_drop_breaks_success_even_after_historical_valid_arrival():
    tr = S3Trace()
    assert tr.hold()[-1].stage_complete
    lost = tr.step(contact=False)
    assert tr.trace.snapshot.lifted_once and not tr.trace.snapshot.grasp_valid
    assert not lost.stage_complete and lost.subphase == "RECOVER_GRASP"
    assert lost.components["s3_grasp_loss"] == -5.
    assert all(not row.stage_complete for row in tr.hold(count=6, contact=False))


def test_pushed_object_then_grasped_in_goal_cannot_launder_source_provenance():
    tr = S3Trace(prepare=False)
    tr.step((.25, 0., .768), contact=False)
    tr.hold((.25, 0., .768), count=5)
    rows = tr.hold((.25, 0., .85), count=10)
    assert not tr.trace.snapshot.grasp_valid and not tr.trace.snapshot.attempt_lifted
    assert all(not row.stage_complete for row in rows)


def test_prepared_inside_state_starts_its_own_hold_without_backdated_bonus():
    tr = S3Trace()
    tr.trace.step((.25, 0., .805), contact=True, at_home=False, repeat=5)
    tr.reward = S3TransportReward(tr.trace.config)
    tr.reward.reset(tr.trace.snapshot)
    rows = tr.hold(count=3)
    assert all(not row.stage_complete for row in rows[:-1])
    assert rows[-1].stage_complete
    assert all(row.components["s3_arrived_complete"] == 0. for row in rows)


@pytest.mark.parametrize("value", [0., -.1, math.nan, math.inf, True])
def test_transport_hold_setting_requires_positive_finite_number(value):
    with pytest.raises(ValueError):
        S3RewardConfig(transport_hold_s=value)
