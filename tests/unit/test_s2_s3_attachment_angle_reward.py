"""S2/S3 learn closed commands while a certified rigid grasp stays attached."""
from dataclasses import replace
import math

import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardInput
from so101_capstone.rewards.s2_grasp_lift_reward import S2GraspLiftReward, S2RewardConfig
from so101_capstone.rewards.s3_transport_reward import S3TransportReward, S3RewardConfig

from test_task_state_grasp_attachment import AssistedTrace
from test_task_state import Trace


DT = 1 / 30
REFERENCE = 0.04


class AngleTrace:
    def __init__(self, reward_type):
        self.trace = AssistedTrace()
        self.trace.assisted_lift()
        self.reward = reward_type(self.trace.config)
        self.reward.reset(self.trace.snapshot)
        self.key = f"{self.reward.stage.lower()}_attachment_gripper_command"

    def step(self, command=REFERENCE - math.radians(2), *, dt=DT, **overrides):
        before = self.trace.snapshot
        targets = (*before.measurements.joint_positions_rad[:-1], command)
        measurements = dict(executed_command_joint_targets_rad=targets,
                            grasp_attachment_sim_time_s=self.trace.t + dt)
        measurements.update(overrides)
        after = self.trace.held_step(dt=dt, **measurements)
        return self.reward.step(PickPlaceRewardInput(
            before, after, dt, terminated=after.outcome != "running"))


@pytest.fixture(params=[S2GraspLiftReward, S3TransportReward])
def trace(request):
    return AngleTrace(request.param)


@pytest.mark.parametrize("margin_deg,rate", [
    (1e-7, .05 * 1e-7 / 2), (.5, .0125), (1., .025), (2., .05), (10., .05),
    (0., -.05), (-.5, -.0875), (-1., -.125), (-2., -.2), (-10., -.2),
])
def test_strict_command_boundary_and_bounded_margin_guidance(trace, margin_deg, rate):
    row = trace.step(REFERENCE - math.radians(margin_deg))
    assert row.components[trace.key] == pytest.approx(rate * DT)
    assert row.metrics["attachment_grasp_reference_rad"] == REFERENCE
    assert row.metrics["attachment_gripper_closing_margin_rad"] == pytest.approx(math.radians(margin_deg))
    assert row.metrics["attachment_gripper_command_reward_time_s"] == pytest.approx(DT)
    assert trace.trace.snapshot.grasp_valid
    assert trace.trace.snapshot.measurements.grasp_attachment_active


def test_reference_is_frozen_physical_angle_not_current_measurement_or_attach_command(trace):
    # Attach command is -0.01, frozen actual angle +0.04. The +0.02 command
    # must be rewarded, even if collision leaves the current actual angle high.
    joints = (*trace.trace.snapshot.measurements.joint_positions_rad[:-1], .9)
    row = trace.step(.02, joint_positions_rad=joints)
    assert row.components[trace.key] > 0
    assert row.metrics["attachment_grasp_reference_rad"] == .04
    assert trace.trace.snapshot.measurements.grasp_attachment_command_rad == -.01
    assert trace.trace.snapshot.measurements.joint_positions_rad[-1] == .9
    assert trace.trace.snapshot.measurements.executed_command_joint_targets_rad[-1] == .02


def test_angle_term_is_separate_from_potential_and_does_not_change_other_progress(trace):
    control = AngleTrace(type(trace.reward))
    close = trace.step(REFERENCE - .01)
    opened = control.step(REFERENCE + .01)
    assert close.components[trace.key] > 0 > opened.components[trace.key]
    assert close.potential_after == opened.potential_after
    assert close.potential_before == opened.potential_before
    assert {k: v for k, v in close.components.items() if k != trace.key} == {
        k: v for k, v in opened.components.items() if k != trace.key}
    assert close.stage_complete == opened.stage_complete


def test_continuing_close_has_dense_credit_but_cannot_pay_to_stall(trace):
    rows = [trace.step() for _ in range(90)]
    assert all(row.components[trace.key] == pytest.approx(.05 * DT) for row in rows)
    assert all(row.reward < 0 for row in rows)
    assert sum(row.components[trace.key] for row in rows) == pytest.approx(.15)
    assert all(not row.stage_complete for row in rows)
    # The entire configured 10-second stage can at most earn +0.5 for grip
    # commands, compared with a +20 stage completion bonus.
    assert trace.reward.config.attachment_close_reward_per_s * 10 < trace.reward.tuning.stage_completion_bonus


def test_duplicate_feedback_pays_zero_and_next_frame_does_not_backfill(trace):
    first = trace.step()
    stamp = trace.trace.snapshot.evidence_sim_time_s
    duplicate = trace.step(REFERENCE, evidence_sim_time_s=stamp)
    assert duplicate.components[trace.key] == 0
    assert duplicate.metrics["attachment_gripper_command_reward_time_s"] == 0
    fresh = trace.step()
    assert fresh.components[trace.key] == pytest.approx(first.components[trace.key])


def test_fresh_sensor_frame_can_lag_current_plugin_status(trace):
    row = trace.step(evidence_sim_time_s=trace.trace.t + DT - .008)
    assert row.components[trace.key] == pytest.approx(.05 * (DT - .008))
    assert row.metrics["attachment_gripper_command_valid"] == 1
    next_row = trace.step(evidence_sim_time_s=trace.trace.t + DT - .008)
    assert next_row.components[trace.key] == pytest.approx(.05 * DT)


def test_no_executed_command_has_no_inferred_angle_reward(trace):
    row = trace.step(executed_command_joint_targets_rad=None)
    assert row.components[trace.key] == 0
    assert row.metrics["attachment_gripper_command_valid"] == 0


@pytest.mark.parametrize("reward_type", [S2GraspLiftReward, S3TransportReward])
def test_unassisted_real_grasp_does_not_invent_attachment_reference(reward_type):
    tr = Trace()
    tr.acquire_and_lift()
    reward = reward_type(tr.config)
    reward.reset(tr.snapshot)
    before = tr.snapshot
    after = tr.step(contact=True, at_home=False, dt=DT,
                    executed_command_joint_targets_rad=(*before.measurements.joint_positions_rad[:-1], -.5))
    row = reward.step(PickPlaceRewardInput(before, after, DT))
    assert after.grasp_valid and not after.measurements.grasp_attachment_enabled
    assert row.components[f"{reward.stage.lower()}_attachment_gripper_command"] == 0
    assert row.metrics["attachment_gripper_command_valid"] == 0


@pytest.mark.parametrize("change", ["inactive", "wrong_attempt", "invalid_grasp", "stale_status"])
def test_attachment_provenance_cannot_be_laundered(trace, change):
    trace.step()
    s = trace.trace.snapshot
    if change == "inactive":
        s = replace(s, measurements=replace(s.measurements, grasp_attachment_active=False))
    elif change == "wrong_attempt":
        s = replace(s, grasp_attempt_id=s.grasp_attempt_id + 1)
    elif change == "invalid_grasp":
        s = replace(s, grasp_valid=False)
    else:
        s = replace(s, measurements=replace(s.measurements, grasp_attachment_sim_time_s=s.sim_time_s - DT))
    components, metrics = trace.reward._command_reward(s, covered_s=DT, fresh=True, failed=False)
    assert components[trace.key] == 0
    assert metrics["attachment_gripper_command_valid"] == 0


def test_wrong_attachment_target_is_rejected_before_scoring(trace):
    trace.step()
    with pytest.raises(ValueError, match="matching target"):
        replace(trace.trace.snapshot.measurements, grasp_attachment_target_name="other")


@pytest.mark.parametrize("fresh,failed", [(False, False), (True, True)])
def test_unknown_interval_or_terminal_failure_cannot_earn_grip_credit(trace, fresh, failed):
    trace.step()
    components, metrics = trace.reward._command_reward(
        trace.trace.snapshot, covered_s=DT, fresh=fresh, failed=failed)
    assert components[trace.key] == 0
    assert metrics["attachment_gripper_command_reward_time_s"] == 0


@pytest.mark.parametrize("config_type", [S2RewardConfig, S3RewardConfig])
@pytest.mark.parametrize("field", ["attachment_close_margin_scale_rad", "attachment_close_reward_per_s", "attachment_open_cost_per_s"])
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True])
def test_angle_settings_require_positive_finite_numbers(config_type, field, value):
    with pytest.raises(ValueError):
        config_type(**{field: value})
