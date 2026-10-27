"""A confirmed rigid grasp may sustain real acquisition, never fabricate it."""
from dataclasses import fields, replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import TASK_CONTEXT_DIM, encode_task_context
from so101_capstone.rewards.task_state import GRASP_ATTACHMENT_CONTEXT_DIM
from test_task_state import Trace


class AssistedTrace(Trace):
    def __init__(self):
        self.attached = False
        self.ever_attached = False
        self.attached_attempt = 0
        self.attached_time = None
        super().__init__()

    def measurement(self, **overrides):
        values = dict(
            grasp_attachment_enabled=True,
            grasp_attachment_active=self.attached,
            grasp_attachment_ever_used=self.ever_attached,
            grasp_attachment_target_name="cube" if self.ever_attached else None,
            grasp_attachment_attempt_id=self.attached_attempt,
            grasp_attachment_command_rad=-0.01 if self.ever_attached else None,
            grasp_attachment_measured_rad=0.04 if self.ever_attached else None,
            grasp_attachment_sim_time_s=self.attached_time,
        )
        values.update(overrides)
        return super().measurement(**values)

    def attach(self, attempt=None):
        self.attached = self.ever_attached = True
        self.attached_attempt = self.snapshot.grasp_attempt_id if attempt is None else attempt
        self.attached_time = self.t

    def held_step(self, position=None, **overrides):
        values = dict(contact=False, at_home=False,
                      grasp_position_world_m=self.position if position is None else position)
        values.update(overrides)
        return self.step(position, **values)

    def acquire_and_attach(self):
        acquired = self.step(contact=True, at_home=False, repeat=3)
        assert acquired.grasp_valid and acquired.grasp_attempt_id == 1
        self.attach()
        return acquired

    def assisted_lift(self):
        self.acquire_and_attach()
        lifted = self.held_step((0, 0, 0.805), repeat=3)
        assert lifted.grasp_valid and lifted.attempt_lifted
        return lifted

    def assisted_carry(self):
        self.assisted_lift()
        carried = self.held_step((0.25, 0, 0.808), repeat=2)
        assert carried.carried_into_goal
        return carried


@pytest.mark.parametrize("contacts", [
    {"left_contact": False}, {"right_contact": False},
    {"contacts_simultaneous": False}, {"contact_valid": False},
])
def test_assistance_enabled_still_requires_real_bilateral_acquisition(contacts):
    trace = AssistedTrace()
    snapshot = trace.step(contact=True, repeat=5, **contacts)
    assert not snapshot.grasp_valid
    assert snapshot.grasp_attempt_id == 0
    assert "grasp_confirmed" not in snapshot.events


def test_constraint_alone_cannot_invent_a_grasp_attempt():
    trace = AssistedTrace()
    trace.attach(attempt=1)  # A backend claim without tracker-certified contact.
    snapshot = trace.held_step(repeat=5)
    assert not snapshot.grasp_valid
    assert not snapshot.grasp_attempt_active
    assert snapshot.grasp_attempt_id == 0
    assert not snapshot.attempt_lifted


def test_same_attempt_remains_grasped_and_carries_when_weld_suppresses_contact():
    trace = AssistedTrace()
    snapshot = trace.assisted_carry()
    assert snapshot.grasp_valid and snapshot.grasp_attempt_active
    assert snapshot.attempt_lifted and snapshot.carried_into_goal
    assert snapshot.grasp_attempt_id == 1
    assert snapshot.measurements.contact_valid
    assert not snapshot.measurements.left_contact
    assert not snapshot.measurements.right_contact
    assert "grasp_lost" not in snapshot.events


def test_active_constraint_cannot_masquerade_as_release_or_placement():
    trace = AssistedTrace()
    trace.assisted_carry()
    snapshot = trace.held_step((0.25, 0, 0.768), repeat=20)
    assert snapshot.grasp_valid
    assert not snapshot.valid_release
    assert not snapshot.delivery_valid
    assert not snapshot.placed_stable
    assert "valid_release" not in snapshot.events


def test_detach_allows_real_release_and_labels_eventual_success_as_assisted():
    trace = AssistedTrace()
    trace.assisted_carry()
    trace.held_step((0.25, 0, 0.768))
    trace.attached = False
    released = trace.step(contact=False, at_home=False)
    assert released.valid_release and released.delivery_valid
    assert "valid_release" in released.events
    success = trace.step(repeat=80)
    assert success.outcome == "success"
    info = success.to_info()
    assert info["formal_success"] is True
    assert info["unassisted_success"] is False
    assert info["success_physics"] == "rigid_grasp_assisted"
    assert info["grasp_attachment_ever_used"] is True
    assert info["grasp_attachment_active"] is False
    assert info["grasp_attachment_command_rad"] == -0.01
    assert info["grasp_attachment_measured_rad"] == 0.04


def test_detach_cannot_use_the_last_welded_empty_contact_frame_as_release():
    trace = AssistedTrace()
    trace.assisted_carry()
    trace.held_step((0.25, 0, 0.768), evidence_sim_time_s=trace.t + .02)
    trace.attached = False
    trace.attached_time = trace.t  # paused-boundary detach acknowledgement
    pending = trace.step(contact=False, at_home=False, evidence_sim_time_s=trace.t)
    assert not pending.valid_release and not pending.delivery_valid
    assert not pending.grasp_valid
    assert pending.grasp_attempt_active and pending.carried_into_goal
    assert "valid_release" not in pending.events and "grasp_lost" not in pending.events
    released = trace.step(contact=False, at_home=False)
    assert released.valid_release and released.delivery_valid
    assert "valid_release" in released.events


@pytest.mark.parametrize("unknown", [
    {"contact_valid": False}, {"evidence_valid": False},
    {"target_measurement_valid": False}, {"target_motion_valid": False},
    {"joint_measurement_valid": False}, {"dt": 1.0},
])
def test_constraint_does_not_launder_unknown_physical_evidence(unknown):
    trace = AssistedTrace()
    trace.assisted_lift()
    snapshot = trace.held_step(**unknown)
    assert not snapshot.grasp_valid
    assert not snapshot.grasp_attempt_active
    assert not snapshot.attempt_lifted
    assert not snapshot.carried_into_goal
    assert not snapshot.valid_release
    # Subsequent backend status cannot restore the lost attempt by itself.
    snapshot = trace.held_step(repeat=3)
    assert not snapshot.grasp_valid
    assert snapshot.grasp_attempt_id == 1


def test_wrong_attempt_does_not_sustain_grasp():
    trace = AssistedTrace()
    trace.assisted_lift()
    snapshot = trace.held_step(grasp_attachment_attempt_id=2)
    assert not snapshot.grasp_valid
    assert not snapshot.grasp_attempt_active
    assert "grasp_lost" in snapshot.events


def test_wrong_attachment_target_is_rejected_before_tracker_update():
    trace = AssistedTrace()
    trace.assisted_lift()
    with pytest.raises(ValueError, match="matching target"):
        trace.held_step(grasp_attachment_target_name="other_cube")


def test_constraint_does_not_hide_measured_relative_slip():
    trace = AssistedTrace()
    trace.assisted_lift()
    snapshot = trace.held_step(grasp_position_world_m=(0.1, 0, 0.805))
    assert not snapshot.grasp_valid
    assert not snapshot.grasp_attempt_active
    assert "grasp_lost" in snapshot.events


def test_enabled_but_unused_attachment_does_not_relabel_natural_success():
    trace = AssistedTrace()
    trace.acquire_and_lift()
    trace.carry_and_release()
    success = trace.step(repeat=80)
    info = success.to_info()
    assert success.outcome == "success"
    assert info["formal_success"] and info["unassisted_success"]
    assert info["success_physics"] == "unassisted"
    assert not info["grasp_attachment_ever_used"]


def test_assisted_context_records_hidden_constraint_and_angle_state():
    trace = AssistedTrace()
    empty = encode_task_context(trace.snapshot)
    assert empty.shape == (TASK_CONTEXT_DIM + GRASP_ATTACHMENT_CONTEXT_DIM,)
    trace.assisted_lift()
    context = encode_task_context(trace.snapshot)
    assert np.isfinite(context).all()
    np.testing.assert_allclose(context[-6:], [1, 1, -0.01, 0.04, math.radians(2), 0.01])
    assert trace.snapshot.to_info()["critic_task_context"] == context.tolist()
    natural = Trace()
    assert encode_task_context(natural.snapshot).shape == (TASK_CONTEXT_DIM,)


def test_reset_cannot_erase_active_attachment_history():
    trace = AssistedTrace()
    trace.assisted_lift()
    with pytest.raises(ValueError, match="reset.*attachment"):
        trace.tracker.reset(trace.measurement())


def test_attachment_mode_cannot_change_mid_episode():
    trace = AssistedTrace()
    with pytest.raises(ValueError, match="mode cannot change"):
        trace.step(grasp_attachment_enabled=False)


def test_paused_attach_annotation_preserves_every_physical_fact_and_timer():
    trace = AssistedTrace()
    before = trace.acquire_and_attach()
    metadata = {field.name: getattr(trace.measurement(), field.name)
                for field in fields(before.measurements)
                if field.name.startswith("grasp_attachment_")}
    annotated = trace.tracker.annotate_attachment_measurements(
        replace(before.measurements, **metadata))
    assert annotated.measurements.grasp_attachment_active
    assert replace(annotated, measurements=before.measurements) == before
    snapshot = trace.held_step()
    assert snapshot.grasp_valid and snapshot.grasp_attempt_id == 1


@pytest.mark.parametrize("alteration", [
    {"sim_time_s": 1.0}, {"left_contact": False},
    {"joint_positions_rad": (0,) * 6},
])
def test_paused_annotation_cannot_rewrite_sensor_evidence(alteration):
    trace = AssistedTrace()
    before = trace.acquire_and_attach()
    with pytest.raises(ValueError, match="cannot alter sensor"):
        trace.tracker.annotate_attachment_measurements(
            replace(before.measurements, **alteration))


def test_paused_annotation_cannot_certify_a_nonexistent_attempt():
    trace = AssistedTrace()
    before = trace.snapshot
    trace.attach(attempt=1)
    metadata = {field.name: getattr(trace.measurement(), field.name)
                for field in fields(before.measurements)
                if field.name.startswith("grasp_attachment_")}
    with pytest.raises(ValueError, match="contact-certified grasp attempt"):
        trace.tracker.annotate_attachment_measurements(
            replace(before.measurements, **metadata))


def test_paused_detach_annotation_does_not_issue_release_event_before_physics():
    trace = AssistedTrace()
    trace.assisted_carry()
    before = trace.held_step((0.25, 0, 0.768))
    detached = trace.tracker.annotate_attachment_measurements(
        replace(before.measurements, grasp_attachment_active=False))
    assert replace(detached, measurements=before.measurements) == before
    assert not detached.valid_release and "valid_release" not in detached.events
    trace.attached = False
    assert trace.step(contact=False).valid_release


@pytest.mark.parametrize("invalid", [
    {"grasp_attachment_enabled": False}, {"grasp_attachment_ever_used": False},
    {"grasp_attachment_attempt_id": 0}, {"grasp_attachment_command_rad": None},
    {"grasp_attachment_measured_rad": None}, {"grasp_attachment_sim_time_s": None},
    {"grasp_attachment_sim_time_s": 100.0},
    {"grasp_attachment_command_rad": float("nan")},
    {"grasp_attachment_measured_rad": float("inf")},
    {"grasp_attachment_release_delta_rad": 0.0},
])
def test_active_attachment_measurement_requires_complete_finite_evidence(invalid):
    trace = AssistedTrace()
    trace.assisted_lift()
    with pytest.raises(ValueError):
        replace(trace.snapshot.measurements, **invalid)
