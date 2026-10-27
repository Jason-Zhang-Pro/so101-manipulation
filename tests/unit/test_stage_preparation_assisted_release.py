"""Assisted release may need empty-tool withdrawal to clear a fixed contact."""
from dataclasses import replace

import pytest

from so101_capstone.environments.stage_preparation import (
    StagePreparationConfig, StagePreparationController, StagePreparationError,
)
from test_stage_preparation import PhysicsTrace
from test_task_state_grasp_attachment import AssistedTrace


def released_snapshot():
    trace = AssistedTrace()
    trace.assisted_carry()
    trace.held_step((0.25, 0, 0.768))
    trace.attached = False
    snapshot = trace.step(contact=False, left_contact=True, at_home=False)
    controller = StagePreparationController(trace.config)
    q = tuple(snapshot.measurements.joint_positions_rad[:-1]) + (controller.open_gripper,)
    snapshot = replace(snapshot, measurements=replace(snapshot.measurements, joint_positions_rad=q))
    assert snapshot.valid_release and snapshot.delivery_valid
    return controller, snapshot


def test_open_detached_released_tool_may_withdraw_despite_fixed_finger_contact():
    controller, snapshot = released_snapshot()
    assert snapshot.measurements.left_contact
    assert controller._release_ready_to_raise(snapshot)
    assert not controller._s4_complete(snapshot)


@pytest.mark.parametrize("field", ["valid_release", "delivery_valid"])
def test_invalid_release_cannot_proceed_to_withdrawal(field):
    controller, snapshot = released_snapshot()
    assert not controller._release_ready_to_raise(replace(snapshot, **{field: False}))


def test_attached_tool_cannot_proceed_even_with_release_flags():
    controller, snapshot = released_snapshot()
    assert not controller._release_ready_to_raise(replace(snapshot,
        measurements=replace(snapshot.measurements, grasp_attachment_active=True)))


def test_opening_command_without_measured_opening_cannot_proceed():
    controller, snapshot = released_snapshot()
    m = snapshot.measurements
    q = m.joint_positions_rad[:-1] + (controller.open_gripper - 2 * controller.config.joint_tolerance_rad,)
    assert not controller._release_ready_to_raise(replace(snapshot,
        measurements=replace(m, joint_positions_rad=q)))


def test_measured_opening_uses_existing_preparation_joint_tolerance():
    controller, snapshot = released_snapshot()
    m = snapshot.measurements
    q = m.joint_positions_rad[:-1] + (controller.open_gripper - .99 * controller.config.joint_tolerance_rad,)
    assert controller._release_ready_to_raise(replace(snapshot,
        measurements=replace(m, joint_positions_rad=q)))


def test_natural_mode_retains_no_contact_requirement_before_withdrawal():
    controller, snapshot = released_snapshot()
    natural = replace(snapshot, measurements=replace(snapshot.measurements, grasp_attachment_enabled=False))
    assert not controller._release_ready_to_raise(natural)
    clear = replace(natural, measurements=replace(natural.measurements, left_contact=False))
    assert controller._release_ready_to_raise(clear)


class LingeringContactTrace(PhysicsTrace):
    def __init__(self, *, assisted=True, never_clear=False):
        self.assisted = assisted
        self.never_clear = never_clear
        self.fixed_contact_at_raise = None
        super().__init__(config=StagePreparationConfig(phase_timeout_s=4))

    def measurement(self):
        m = super().measurement()
        phase = getattr(self.controller, "phase", None)
        lingering = self.q[-1] > .35 and (phase == "RELEASE_OBJECT" or
            (self.never_clear and phase in ("RAISE_EMPTY_GRIPPER", "VERIFY_PLACEMENT")))
        return replace(m, grasp_attachment_enabled=self.assisted,
                       left_contact=m.left_contact or lingering)

    def advance(self, command):
        if self.controller.phase == "RAISE_EMPTY_GRIPPER" and self.fixed_contact_at_raise is None:
            self.fixed_contact_at_raise = self.controller.snapshot.measurements.left_contact
            assert self.controller.snapshot.valid_release
            assert self.controller.snapshot.delivery_valid
            assert not self.controller.snapshot.measurements.grasp_attachment_active
            assert abs(self.q[-1] - self.controller.open_gripper) <= self.controller.config.joint_tolerance_rad
        return super().advance(command)


def test_fixed_contact_clears_only_after_raise_and_full_s5_preparation_finishes():
    # Actual motion/contact generation is a semantic fixture, not Gazebo proof.
    trace = LingeringContactTrace()
    report = trace.run("S5")
    assert trace.fixed_contact_at_raise is True
    assert "RAISE_EMPTY_GRIPPER" in report.phases
    assert "VERIFY_PLACEMENT" in report.phases
    assert report.certificate["s4_complete"]
    assert not report.final_snapshot.measurements.left_contact
    assert not report.final_snapshot.measurements.right_contact
    assert report.final_snapshot.target_distance_m >= trace.controller.s4.release_separation_m


def test_contact_remaining_after_raise_still_blocks_final_preparation_certificate():
    trace = LingeringContactTrace(never_clear=True)
    with pytest.raises(StagePreparationError) as error:
        trace.run("S5")
    assert trace.fixed_contact_at_raise is True
    assert error.value.phase == "VERIFY_PLACEMENT"


def test_natural_trace_still_waits_for_contact_clearance_before_raise():
    trace = LingeringContactTrace(assisted=False)
    with pytest.raises(StagePreparationError) as error:
        trace.run("S5")
    assert error.value.phase == "RELEASE_OBJECT"
    assert trace.fixed_contact_at_raise is None
