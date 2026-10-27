"""Formal S2/S3 weld persistence; fake physics does not validate Gazebo."""
from dataclasses import replace
import math

import pytest

from so101_capstone.environments.gazebo_grasp_attachment import GazeboGraspAttachmentError
from so101_capstone.rewards.training_stages import StageTrainingConfig

from test_gazebo_grasp_attachment import COMMAND, controller, snapshot
from test_grasp_attachment_environment_integration import AttachmentHarness


@pytest.mark.parametrize("stage", ["S2", "S3"])
@pytest.mark.parametrize("opening", [0., .15, math.pi])
def test_formal_s2_s3_never_detach_for_any_measured_or_commanded_opening(stage, opening):
    client, backend, _ = controller()
    client.configure_training_stage(stage)
    initial = client.after_step(snapshot(), COMMAND)
    command = COMMAND[:-1] + (opening,)
    fresh = snapshot(command=command, measured=opening)
    for _ in range(3):
        client.before_step(command, fresh)
        result = client.after_step(fresh, command)
    assert result["active"] and result["release_mode"] == "stage_persistent"
    assert result["training_stage"] == stage
    assert not result["release_allowed"] and not result["release_requested"]
    assert result["confirmed_measured_rad"] == initial["confirmed_measured_rad"]
    assert backend.attachments == 1 and backend.detachments == 0
    assert len(backend.calls) == 1


@pytest.mark.parametrize("stage", [None, "S0", "S1", "S4", "S5"])
def test_default_and_other_stages_retain_measured_opening_release(stage):
    client, backend, _ = controller()
    client.configure_training_stage(stage)
    client.after_step(snapshot(), COMMAND)
    command = COMMAND[:-1] + (math.pi,)
    result = client.before_step(command, snapshot(command=command, measured=math.pi))
    assert not result["active"] and result["release_mode"] == "measured_opening"
    assert result["release_allowed"] and backend.detachments == 1


def test_s2_s3_handoff_keeps_same_constraint_and_s4_restores_normal_release():
    client, backend, _ = controller()
    client.configure_training_stage("S2")
    client.after_step(snapshot(), COMMAND)
    original = client.metadata()
    command = COMMAND[:-1] + (math.pi,)
    fresh = snapshot(command=command, measured=math.pi)
    client.configure_training_stage("S3")
    client.before_step(command, fresh)
    assert client.metadata()["joint_entity"] == original["joint_entity"]
    assert client.metadata()["confirmed_measured_rad"] == original["confirmed_measured_rad"]
    client.configure_training_stage("S4")
    assert backend.detachments == 0  # Configuration never executes a service.
    client.before_step(command, snapshot(command=command))
    assert client.metadata()["active"]  # Still wait for actual jaw clearance.
    client.after_step(fresh, command)
    assert not client.metadata()["active"] and backend.detachments == 1
    client.configure_training_stage("S2")
    client.after_step(snapshot(), COMMAND)
    assert backend.attachments == 1  # Same old attempt cannot be reattached.


@pytest.mark.parametrize("stage", ["S2", "S3"])
def test_persistence_still_requires_contact_certificate_and_fresh_constraint_status(stage):
    client, backend, _ = controller()
    client.configure_training_stage(stage)
    s = snapshot()
    invalid = replace(s, measurements=replace(s.measurements, right_contact=False))
    client.after_step(invalid, COMMAND)
    assert not backend.calls
    client.after_step(s, COMMAND)
    with pytest.raises(GazeboGraspAttachmentError, match="unexpectedly changed"):
        client.observe_status({**backend.status, "state": "detached", "joint_entity": 0})


def test_release_policy_is_independent_between_worlds():
    held, held_backend, _ = controller(world_name="held")
    other, other_backend, _ = controller(world_name="other")
    held.configure_training_stage("S2")
    for client in (held, other):
        client.after_step(snapshot(), COMMAND)
        command = COMMAND[:-1] + (math.pi,)
        client.before_step(command, snapshot(command=command, measured=math.pi))
    assert held_backend.detachments == 0
    assert other_backend.detachments == 1


@pytest.mark.parametrize("stage", ["S2", "S3"])
@pytest.mark.parametrize("boundary", ["preparation", "clear_stage", "full_task"])
def test_core_turns_off_persistence_at_preparation_or_full_task_boundary(stage, boundary):
    h = AttachmentHarness()
    h.core.configure_training_stage(stage, StageTrainingConfig(enable_later_stage_context=True))
    assert h.client.metadata()["release_mode"] == "stage_persistent"
    if boundary == "preparation":
        h.core.begin_stage_preparation()
    elif boundary == "clear_stage":
        h.core.configure_training_stage(None)
    else:
        h.core.configure_task("pick_place")
    assert h.client.metadata()["release_mode"] == "measured_opening"
    assert h.client.metadata()["training_stage"] is None


def test_preparation_can_release_even_if_s3_was_configured_before_it():
    h = AttachmentHarness()
    h.core.configure_training_stage("S3", StageTrainingConfig(enable_later_stage_context=True))
    h.core.begin_stage_preparation()
    h.attach()
    command = h.command[:-1] + (h.q[-1] + h.client.release_delta_rad,)
    h.q = h.q[:-1] + (command[-1],)
    result = h.step(command)
    assert result.info["preparation_only"]
    assert result.info["grasp_attachment"]["release_mode"] == "measured_opening"
    assert not result.info["grasp_attachment_active"]


def test_measured_s2_handoff_keeps_attachment_and_actions_free_without_extra_steps():
    h = AttachmentHarness(preparing=True)
    h.contact = False
    h.step()
    h.step()
    config = StageTrainingConfig(enable_later_stage_context=True)
    _, info = h.coordinator.handoff_training_stage("S2", config, command_joint_targets_rad=h.command)
    assert info["grasp_attachment"]["release_mode"] == "stage_persistent"
    h.contact = True
    h.attach()
    h.contact = False
    command = h.command[:-1] + (1.2,)
    h.q = h.q[:-1] + (1.2,)
    count = len(h.steps)
    result = h.step(command)
    assert h.published[-1] == command
    assert h.core.task_snapshot.measurements.executed_command_joint_targets_rad == command
    assert len(h.steps) == count + 1
    assert h.backend.attachments == 1 and h.backend.detachments == 0
    assert result.info["grasp_attachment_active"]
    assert h.core.task_snapshot.grasp_valid
