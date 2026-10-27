"""Task certification is tested with sensor traces, independently of PPO/ROS."""
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import (
    PickPlaceRewardConfig, TaskMeasurements, TaskPhase, TaskStateTracker,
    TASK_CONTEXT_DIM, encode_task_context,
)
from so101_capstone.rewards.task_state import object_geometry


class Trace:
    def __init__(self, *, mode="pick_place", target=True, config=None):
        self.config = config or PickPlaceRewardConfig()
        self.t = 0.0
        self.position = (0.25 if mode == "idle_completed" else 0.0, 0.0, 0.768 if mode == "idle_completed" else 0.765)
        self.target = target
        self.tracker = TaskStateTracker(self.config)
        self.snapshot = self.tracker.reset(self.measurement(), mode, task_instance_id="trace")

    def measurement(self, **overrides):
        kwargs = dict(sim_time_s=self.t, joint_positions_rad=self.config.home_joint_positions_rad,
                      joint_velocities_rad_s=(0,) * 6, grasp_position_world_m=(0, 0, 1.1),
                      grasp_rotation_world=np.eye(3), target_name="cube" if self.target else None,
                      target_position_world_m=self.position if self.target else None,
                      target_motion_valid=self.target, contact_valid=True,
                      goal_center_world_m=(0.25, 0, 0.753), goal_half_extents_xy_m=(0.05, 0.05),
                      grasp_evidence_mode=self.config.grasp_evidence_mode)
        kwargs.update(overrides)
        return TaskMeasurements(**kwargs)

    def step(self, position=None, *, contact=False, at_home=True, repeat=1, dt=0.05, **overrides):
        if position is not None:
            self.position = position
        for _ in range(repeat):
            self.t += dt
            q = list(self.config.home_joint_positions_rad)
            if not at_home:
                q[0] += 0.3
            values = dict(grasp_position_world_m=self.position if contact else (0, 0, 1.1),
                          left_contact=contact, right_contact=contact, contacts_simultaneous=contact,
                          joint_positions_rad=q)
            values.update(overrides)
            self.snapshot = self.tracker.update(self.measurement(**values))
        return self.snapshot

    def acquire_and_lift(self, *, x=0.0, bottom=0.75):
        self.step((x, 0, bottom+0.015), contact=True, repeat=3, at_home=False)
        return self.step((x, 0, bottom+0.055), contact=True, repeat=3, at_home=False)

    def carry_and_release(self):
        self.step((0.25, 0, 0.808), contact=True, repeat=2, at_home=False)
        self.step((0.25, 0, 0.768), contact=True, at_home=False)
        return self.step(contact=False, at_home=False)


def test_complete_grasp_place_then_all_six_home_and_measured_hold():
    trace = Trace()
    assert trace.snapshot.phase is TaskPhase.APPROACH
    lifted = trace.acquire_and_lift()
    assert lifted.grasp_valid and lifted.attempt_lifted
    assert lifted.phase is TaskPhase.TRANSPORT
    released = trace.carry_and_release()
    assert released.valid_release and released.delivery_valid
    assert "valid_release" in released.events
    placed = trace.step(at_home=False, repeat=10)
    assert placed.placed_stable and placed.phase is TaskPhase.RETURN_HOME
    q = list(trace.config.home_joint_positions_rad)
    q[-1] += 0.1
    assert trace.step(joint_positions_rad=q, repeat=65).outcome == "running"
    assert trace.snapshot.home_stable_duration_s == 0
    assert trace.step(repeat=59).outcome == "running"
    success = trace.step()
    assert success.outcome == "success" and success.success_event
    assert success.to_info()["formal_success"] is True
    assert trace.step().success_event is False


def test_stationary_requires_real_velocity_and_resets_hold():
    trace = Trace()
    trace.acquire_and_lift()
    trace.carry_and_release()
    trace.step(repeat=20)
    assert trace.snapshot.home_stable_duration_s > 0
    moving = trace.step(joint_velocities_rad_s=(0.1, 0, 0, 0, 0, 0))
    assert moving.robot_at_home and not moving.robot_static
    assert moving.phase is TaskPhase.HOME_HOLD
    assert moving.home_stable_duration_s == 0
    assert trace.step(repeat=59).outcome == "running"
    assert trace.step().outcome == "success"


@pytest.mark.parametrize("mode", ["pick_place", "idle_empty"])
def test_empty_home_never_earns_pick_place_success(mode):
    trace = Trace(mode=mode, target=False)
    s = trace.step(repeat=80)
    assert s.phase is TaskPhase.WAIT_HOME and s.robot_at_home and s.robot_static
    assert s.outcome == "running" and not s.success_event


def test_idle_completed_cannot_earn_delivery_and_disturbance_fails():
    trace = Trace(mode="idle_completed")
    s = trace.step(repeat=80)
    assert s.outcome == "running" and not s.delivery_valid
    disturbed = trace.step(target_linear_velocity_m_s=(0.1, 0, 0))
    assert disturbed.outcome == "task_failure"
    assert disturbed.failure_reason == "completed_object_disturbed"


def test_spawn_has_physical_boundary_and_target_identity_is_fixed():
    trace = Trace(target=False)
    trace.step(repeat=3)
    trace.target = True
    s = trace.step()
    assert s.target_name == "cube" and s.phase is TaskPhase.APPROACH
    with pytest.raises(ValueError, match="identity"):
        trace.step(target_name="another")


def test_pushing_object_into_goal_then_gripping_and_lifting_is_not_delivery():
    trace = Trace()
    trace.step((0.25, 0, 0.768), contact=False)
    trace.step(contact=True, repeat=4)
    s = trace.step((0.25, 0, 0.808), contact=True, repeat=4)
    assert not s.grasp_valid and not s.attempt_lifted and not s.carried_into_goal
    trace.step((0.25, 0, 0.768), contact=False)
    assert trace.step(repeat=80).outcome == "running"


def test_lift_once_drop_push_regrip_does_not_launder_history():
    trace = Trace()
    trace.acquire_and_lift()
    lost = trace.step((0.21, 0, 0.765), contact=False)
    assert lost.lifted_once and not lost.attempt_lifted
    # This is within the local recovery radius, but already inside goal.
    trace.step((0.23, 0, 0.768), contact=True, repeat=4)
    s = trace.step((0.23, 0, 0.808), contact=True, repeat=4)
    assert not s.grasp_valid and not s.arrived_via_valid_carry
    trace.step((0.23, 0, 0.768), contact=False)
    assert trace.step(repeat=80).outcome == "running"


def test_recovery_after_real_lift_outside_goal_requires_new_lift():
    trace = Trace()
    trace.acquire_and_lift()
    trace.step((0.10, 0, 0.765), contact=False)
    s = trace.step(contact=True, repeat=3)
    assert s.grasp_attempt_id == 2 and s.grasp_valid and not s.attempt_lifted
    s = trace.step((0.10, 0, 0.805), contact=True, repeat=3)
    assert s.attempt_lifted
    trace.carry_and_release()
    assert trace.step(repeat=80).outcome == "success"


def test_goal_recovery_only_after_valid_carried_arrival():
    trace = Trace()
    trace.acquire_and_lift()
    trace.step((0.25, 0, 0.808), contact=True)
    lost = trace.step(contact=False)
    assert lost.arrived_via_valid_carry and not lost.delivery_valid
    trace.step((0.25, 0, 0.768), contact=False)
    trace.step(contact=True, repeat=3)
    assert not trace.snapshot.attempt_lifted
    trace.step((0.25, 0, 0.808), contact=True, repeat=3)
    trace.step((0.25, 0, 0.768), contact=True)
    assert trace.step().valid_release
    assert trace.step(repeat=80).outcome == "success"


def test_goal_recovery_token_revokes_and_cannot_be_restored_by_pushing_back():
    trace = Trace()
    trace.acquire_and_lift()
    trace.step((0.25, 0, 0.808), contact=True)
    trace.step(contact=False)
    assert trace.snapshot.goal_recovery_authorized
    trace.step((0.4, 0, 0.765), contact=False)
    assert trace.snapshot.arrived_via_valid_carry and not trace.snapshot.goal_recovery_authorized
    trace.step((0.25, 0, 0.768), contact=False)
    trace.step(contact=True, repeat=3)
    trace.step((0.25, 0, 0.808), contact=True, repeat=3)
    assert not trace.snapshot.grasp_valid and not trace.snapshot.carried_into_goal
    assert not trace.snapshot.goal_recovery_authorized


@pytest.mark.parametrize("velocity", [
    {"target_linear_velocity_m_s": (0.2, 0, 0)},
    {"target_angular_velocity_rad_s": (0, 0, 1.0)},
])
def test_throw_then_eventual_settling_is_not_legal_placement(velocity):
    trace = Trace()
    trace.acquire_and_lift()
    trace.step((0.25, 0, 0.808), contact=True)
    trace.step((0.25, 0, 0.768), contact=True)
    s = trace.step(contact=False, **velocity)
    assert not s.valid_release and not s.delivery_valid
    assert trace.step(repeat=80).outcome == "running"


def test_placement_confirmation_requires_both_fingers_released():
    trace = Trace()
    trace.acquire_and_lift()
    trace.carry_and_release()
    s = trace.step(left_contact=True, repeat=80)
    assert s.valid_release and not s.placed_stable and s.place_stable_duration_s == 0
    assert s.outcome == "running"
    assert trace.step(repeat=80).outcome == "success"


def test_contact_without_lift_does_not_authorize_recovery_ratchet():
    trace = Trace()
    trace.step(contact=True, repeat=3)
    trace.step((0.1, 0, 0.765), contact=False)
    assert not trace.snapshot.local_recovery_authorized
    s = trace.step(contact=True, repeat=3)
    assert not s.grasp_valid and s.grasp_attempt_id == 1


def test_relative_slip_invalidates_attempt_and_can_reacquire():
    trace = Trace()
    trace.step(contact=True, repeat=3)
    slipped = trace.step(contact=True, grasp_position_world_m=(0.02, 0, 0.765))
    assert not slipped.grasp_valid and not slipped.grasp_attempt_active
    s = trace.step(contact=True, repeat=3)
    assert s.grasp_valid and s.grasp_attempt_id == 2


def test_only_simultaneous_both_fingers_certify_grasp():
    trace = Trace()
    trace.step(contact=True, contacts_simultaneous=False, repeat=5)
    assert not trace.snapshot.grasp_valid
    trace.step(contact=True, right_contact=False, repeat=5)
    assert not trace.snapshot.grasp_valid


def test_duplicate_evidence_freezes_timers_and_unknown_clears_them():
    trace = Trace()
    trace.acquire_and_lift()
    trace.carry_and_release()
    trace.step(repeat=20)
    before = trace.snapshot.home_stable_duration_s
    old_evidence = trace.snapshot.evidence_sim_time_s
    s = trace.step(evidence_sim_time_s=old_evidence)
    assert s.home_stable_duration_s == before
    s = trace.step(evidence_valid=False)
    assert s.home_stable_duration_s == 0 and s.place_stable_duration_s == 0
    assert trace.step(repeat=60).outcome == "running"


def test_unknown_carry_interval_requires_new_attempt():
    trace = Trace()
    trace.acquire_and_lift()
    s = trace.step(contact=True, evidence_valid=False)
    assert not s.grasp_attempt_active and not s.attempt_lifted
    s = trace.step((0.25, 0, 0.808), contact=True, repeat=5)
    assert not s.carried_into_goal


def test_warmup_unknown_motion_is_not_stationary_and_large_gap_does_not_count():
    trace = Trace(mode="idle_completed")
    s = trace.step(target_motion_valid=False)
    assert not s.target_static and s.home_stable_duration_s == 0
    trace.step(repeat=3)
    s = trace.step(dt=1.0)
    assert s.home_stable_duration_s == 0 and s.outcome == "running"


def test_whole_rotated_footprint_and_physical_bottom():
    trace = Trace()
    # Center is inside, a corner is outside: 45-degree cube width is larger.
    m = trace.measurement(target_position_world_m=(0.280, 0, 0.768),
                          target_quaternion_xyzw=(0, 0, math.sin(math.pi/8), math.cos(math.pi/8)))
    inside, bottom, _ = object_geometry(m, margin_m=0.002)
    assert not inside and bottom == pytest.approx(0.753)
    assert object_geometry(replace(m, target_position_world_m=(0.25, 0, 0.768)))[0]
    tilted = replace(m, target_quaternion_xyzw=(math.sin(math.pi/4), 0, 0, math.cos(math.pi/4)), target_half_extents_m=(.01,.02,.03))
    assert object_geometry(tilted)[1] == pytest.approx(0.748)


def test_already_placed_target_requires_idle_completed_mode():
    trace = Trace(mode="idle_completed")
    with pytest.raises(ValueError, match="outside goal"):
        TaskStateTracker(trace.config).reset(trace.measurement(), "pick_place")


def test_proxy_remains_labelled_and_never_claims_formal_success():
    trace = Trace(config=PickPlaceRewardConfig(grasp_evidence_mode="proxy"))
    trace.acquire_and_lift()
    trace.carry_and_release()
    s = trace.step(repeat=80)
    assert s.outcome == "success" and s.to_info()["formal_success"] is False
    assert s.to_info()["grasp_evidence_mode"] == "proxy"


def test_proxy_can_evaluate_without_claiming_contacts_are_measured():
    trace = Trace(config=PickPlaceRewardConfig(grasp_evidence_mode="proxy"))
    trace.step(contact=True, contact_valid=False, repeat=3)
    s = trace.step((0, 0, .805), contact=True, contact_valid=False, repeat=3)
    assert s.grasp_valid and s.attempt_lifted
    assert s.to_info()["contact_measurement_valid"] is False


def test_context_is_fixed_finite_and_includes_attempt_history_anchors():
    trace = Trace()
    before = encode_task_context(trace.snapshot)
    trace.acquire_and_lift()
    context = encode_task_context(trace.snapshot)
    assert context.shape == (TASK_CONTEXT_DIM,) and context.dtype == np.float32
    assert np.isfinite(context).all() and not np.array_equal(before, context)
    assert np.array_equal(context, trace.snapshot.to_info()["critic_task_context"])
    assert trace.snapshot.to_info()["task_phase_name"] == "TRANSPORT"
    assert trace.snapshot.source_position_xy_m == (0, 0)


def test_command_context_records_actual_last_executed_absolute_command():
    trace = Trace()
    baseline = encode_task_context(trace.snapshot)
    q_command = tuple(q+.01 for q in trace.config.home_joint_positions_rad)
    s = trace.step(executed_command_joint_targets_rad=q_command)
    context = encode_task_context(s)
    np.testing.assert_allclose(context[-7:-1], q_command, atol=1e-7)
    assert baseline[-1] == 0 and context[-1] == 1


def test_invalid_measurements_and_time_reversal_are_errors():
    trace = Trace()
    with pytest.raises(ValueError, match="finite"):
        trace.measurement(joint_velocities_rad_s=(math.nan,) * 6)
    with pytest.raises(ValueError, match="increase"):
        trace.tracker.update(trace.measurement())
    with pytest.raises(ValueError, match="rotation"):
        trace.measurement(grasp_rotation_world=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="future"):
        trace.measurement(evidence_sim_time_s=1)
