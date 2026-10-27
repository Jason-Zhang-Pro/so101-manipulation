"""CPU sensor traces for LOWER -> RELEASE -> empty-gripper RAISE.

Grasp, carried entry, release and placement flags come from TaskStateTracker.
No Gazebo or robot is started by these tests.
"""
from dataclasses import replace

import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardInput
from so101_capstone.rewards.s4_place_release_reward import (
    S4PlaceReleaseReward, S4PlaceReleaseRewardConfig,
)
from test_task_state import Trace


DT = 1 / 30
GOAL_Z = .753
PLACED_CENTER = (.25, 0., GOAL_Z + .015)


class S4Trace:
    def __init__(self, config=None):
        self.trace = Trace()
        self.trace.acquire_and_lift()
        self.trace.step((.25, 0., .805), contact=True, repeat=3, at_home=False, dt=DT)
        assert self.trace.snapshot.grasp_valid and self.trace.snapshot.carried_into_goal
        self.reward = S4PlaceReleaseReward(self.trace.config, config)
        self.reward.reset(self.trace.snapshot)

    def step(self, position=None, *, contact=True, tool_z=None, tool_x=.25, **measurements):
        previous = self.trace.snapshot
        if tool_z is not None:
            measurements["grasp_position_world_m"] = (tool_x, 0., tool_z)
        current = self.trace.step(position, contact=contact, at_home=False, dt=DT, **measurements)
        return self.reward.step(PickPlaceRewardInput(
            previous, current, current.sim_time_s - previous.sim_time_s,
            terminated=current.outcome != "running", truncated=False,
        ))

    def lower(self):
        return self.step(PLACED_CENTER, contact=True)

    def release(self, *, tool_z=GOAL_Z + .03, **measurements):
        return self.step(contact=False, tool_z=tool_z, **measurements)

    def settle(self, *, tool_z=GOAL_Z + .081, tool_x=.25, count=20):
        return [self.step(contact=False, tool_z=tool_z, tool_x=tool_x) for _ in range(count)]


def test_three_parts_reward_raise_but_success_requires_stable_separated_placement():
    t = S4Trace()
    assert t.reward.signals(t.trace.snapshot).phase == "LOWER"
    lowered = t.lower()
    assert lowered.subphase == "RELEASE"
    assert lowered.components["s4_lowered_complete"] == t.reward.config.lowered_bonus
    assert not lowered.stage_complete
    released = t.release()
    assert released.subphase == "RAISE"
    assert released.components["s4_released_complete"] == t.reward.config.released_bonus
    assert released.components["s4_grasp_loss"] == 0
    assert not released.stage_complete
    results = t.settle()
    assert results[-1].stage_complete
    assert sum(r.stage_success_event for r in results) == 1
    assert sum(r.components["s4_raised_complete"] for r in results) == t.reward.config.raised_bonus
    assert sum(r.components["s4_stage_complete"] for r in results) == t.reward.tuning.stage_completion_bonus
    assert results[-1].metrics["raised_hold_s"] == pytest.approx(.1)
    assert results[-1].metrics["placed_hold_s"] == pytest.approx(.1)
    assert t.trace.snapshot.placed_stable and t.trace.snapshot.target_supported
    assert all(not r.terminated and not r.truncated for r in results)


def test_sideways_withdrawal_can_complete_stage_without_empty_gripper_height():
    t = S4Trace()
    t.lower()
    t.release()
    results = t.settle(tool_z=GOAL_Z + .03, tool_x=.35)
    assert t.trace.snapshot.placed_stable
    assert sum(r.components["s4_placed_complete"] for r in results) == t.reward.config.placed_bonus
    assert results[-1].stage_complete
    assert sum(r.stage_success_event for r in results) == 1
    assert all(r.components["s4_raised_complete"] == 0 for r in results)
    assert results[-1].metrics["raise_candidate"] == 0


def test_raising_while_still_holding_object_pays_no_empty_raise_progress():
    t = S4Trace()
    raised = t.step((.25, 0., .87), contact=True)
    assert t.trace.snapshot.grasp_valid
    assert raised.subphase == "LOWER"
    assert not raised.stage_complete
    assert t.reward.signals(t.trace.snapshot).qualities["empty_raise"] == 0
    assert not any(v for k, v in raised.components.items() if k.startswith("s4_empty_raise_band_"))


def test_one_finger_contact_cannot_complete_release_or_enable_raise():
    t = S4Trace()
    t.lower()
    one_finger = t.release(left_contact=True)
    assert t.trace.snapshot.valid_release  # Tracker permits legal partial detachment.
    assert one_finger.subphase == "RELEASE"
    assert one_finger.components["s4_released_complete"] == 0
    assert t.reward.signals(t.trace.snapshot).qualities["empty_raise"] == 0
    cleared = t.release()
    assert cleared.subphase == "RAISE"
    assert cleared.components["s4_released_complete"] == t.reward.config.released_bonus


def test_object_brought_up_with_gripper_invalidates_placement_and_blocks_success():
    t = S4Trace()
    t.lower()
    t.release()
    lifted = t.step((.25, 0., .82), contact=False, tool_z=GOAL_Z + .10)
    assert not t.trace.snapshot.delivery_valid
    assert lifted.components["s4_placement_loss"] == -t.reward.tuning.placement_loss_cost
    assert lifted.metrics["recovery_required"] == 1
    assert not lifted.stage_complete
    assert t.reward.signals(t.trace.snapshot).qualities["empty_raise"] == 0


def test_small_upward_steps_pay_immediately_but_revisiting_height_cannot_pay_again():
    t = S4Trace()
    t.lower()
    t.release(tool_z=GOAL_Z + .03)
    improvement = t.release(tool_z=GOAL_Z + .0301)
    credit = lambda r: sum(v for k, v in r.components.items() if k.startswith("s4_empty_raise_band_"))
    assert credit(improvement) > 0
    retreat = t.release(tool_z=GOAL_Z + .029)
    assert credit(retreat) == 0
    assert retreat.components["s4_regression"] < 0
    assert credit(t.release(tool_z=GOAL_Z + .0301)) == 0


def test_height_subgoal_hold_resets_without_revoking_placement_success():
    t = S4Trace()
    t.lower()
    t.release()
    # Establish stable placement by withdrawing sideways without reaching height.
    placed = t.settle(tool_z=GOAL_Z + .04, tool_x=.35)
    assert sum(r.stage_success_event for r in placed) == 1
    first = t.release(tool_z=GOAL_Z + .08)
    assert first.metrics["raised_hold_s"] == 0
    partial = t.release(tool_z=GOAL_Z + .08)
    assert 0 < partial.metrics["raised_hold_s"] < .1
    broken = t.release(tool_z=GOAL_Z + .079)
    assert broken.metrics["raised_hold_s"] == 0
    results = t.settle(tool_z=GOAL_Z + .08, count=5)
    assert results[-1].stage_complete
    assert sum(r.components["s4_raised_complete"] for r in results) == t.reward.config.raised_bonus
    assert sum(r.stage_success_event for r in results) == 0
    assert t.release(tool_z=GOAL_Z + .079).stage_complete
    repeated = t.settle(tool_z=GOAL_Z + .08, count=5)
    assert repeated[-1].stage_complete
    assert all(r.components["s4_stage_complete"] == 0 for r in repeated)
    assert all(r.components["s4_raised_complete"] == 0 for r in repeated)


def test_configurable_height_and_hold_use_the_goal_support_plane():
    t = S4Trace(S4PlaceReleaseRewardConfig(raise_clearance_m=.10, raise_hold_s=.2))
    t.lower()
    t.release()
    placed = t.settle(tool_z=GOAL_Z + .081)
    assert placed[-1].stage_complete
    assert not any(r.components["s4_raised_complete"] for r in placed)
    results = t.settle(tool_z=GOAL_Z + .10, count=8)
    assert results[-1].stage_complete
    assert results[-1].metrics["raised_hold_s"] == pytest.approx(.2)


def test_five_cm_and_point_one_second_succeed_without_legacy_placement_certificate():
    t = S4Trace()
    t.lower()
    t.release()
    # Exactly 5 cm horizontally from the object center, with no 8 cm raise.
    rows = t.settle(tool_z=PLACED_CENTER[2], tool_x=.30, count=4)
    assert [r.stage_complete for r in rows] == [False, False, False, True]
    assert rows[-1].metrics["placed_hold_s"] == pytest.approx(.1)
    assert rows[-1].metrics["withdrawal_distance_m"] == pytest.approx(.05)
    assert not t.trace.snapshot.placed_stable
    assert t.trace.snapshot.place_stable_duration_s == 0
    assert rows[-1].components["s4_raised_complete"] == 0
    assert rows[-1].components["s4_placed_complete"] == t.reward.config.placed_bonus
    assert rows[-1].components["s4_stage_complete"] == t.reward.tuning.stage_completion_bonus


@pytest.mark.parametrize("broken", [
    {"tool_x": .2999},
    {"left_contact": True},
    {"right_contact": True},
    {"target_linear_velocity_m_s": (.0151, 0., 0.)},
    {"target_angular_velocity_rad_s": (0., 0., .1501)},
    {"position": (.25, 0., GOAL_Z + .022)},
])
def test_each_placement_condition_breaks_continuous_hold(broken):
    t = S4Trace()
    t.lower()
    t.release()
    partial = t.settle(tool_z=PLACED_CENTER[2], tool_x=.30, count=3)
    assert 0 < partial[-1].metrics["placed_hold_s"] < .1
    kwargs = dict(contact=False, tool_z=PLACED_CENTER[2], tool_x=.30)
    kwargs.update(broken)
    interrupted = t.step(**kwargs)
    assert not interrupted.stage_complete
    assert interrupted.metrics["placed_hold_s"] == 0
    # Restore measured geometry; previous partial dwell cannot carry over.
    restored = t.step(PLACED_CENTER, contact=False, tool_z=PLACED_CENTER[2], tool_x=.30)
    assert restored.metrics["placed_hold_s"] == 0
    rows = t.settle(tool_z=PLACED_CENTER[2], tool_x=.30, count=3)
    assert [r.stage_complete for r in rows] == [False, False, True]


def test_leaving_goal_invalidates_release_and_cannot_complete_by_being_pushed_back():
    t = S4Trace()
    t.lower()
    t.release()
    outside = t.step((.4, 0., PLACED_CENTER[2]), contact=False,
                     tool_z=PLACED_CENTER[2], tool_x=.5)
    assert not t.trace.snapshot.delivery_valid
    assert not outside.stage_complete
    pushed = t.step(PLACED_CENTER, contact=False, tool_z=PLACED_CENTER[2], tool_x=.30)
    assert not pushed.stage_complete
    assert not any(r.stage_complete for r in t.settle(count=10))


@pytest.mark.parametrize("name", ["raise_clearance_m", "raise_hold_s", "raised_bonus",
                                  "release_separation_m", "place_hold_s"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_new_config_is_rejected(name, value):
    with pytest.raises(ValueError):
        S4PlaceReleaseRewardConfig(**{name: value})


def test_invalid_feedback_does_not_change_raise_progress_or_paid_records():
    t = S4Trace()
    t.lower()
    t.release()
    previous = t.trace.snapshot
    before = t.reward.reward_state()
    invalid = replace(previous, measurements=replace(previous.measurements,
        sim_time_s=previous.sim_time_s + DT, contact_valid=False))
    with pytest.raises(ValueError, match="invalid sensor evidence"):
        t.reward.step(PickPlaceRewardInput(previous, invalid, DT))
    assert t.reward.reward_state() == before
