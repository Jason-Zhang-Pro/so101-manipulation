"""Offline protocol/state-machine tests; no Gazebo or physical grasp claims."""
from dataclasses import replace
import json
import math
import subprocess
from types import SimpleNamespace

import pytest

from so101_capstone.environments.gazebo_grasp_attachment import (
    GazeboGraspAttachmentController, GazeboGraspAttachmentError, GraspAttachmentStateCache,
)
from so101_capstone.rewards.task_state import TaskMeasurements, TaskPhase, TaskSnapshot


COMMAND = (0., 0., 0., 0., 0., math.radians(1.91))


def snapshot(*, attempt=1, time=1., command=COMMAND, measured=math.radians(7.08), **changes):
    m = TaskMeasurements(sim_time_s=time, joint_positions_rad=(0.,) * 5 + (measured,),
                         joint_velocities_rad_s=(0.,) * 6, grasp_position_world_m=(.25, 0., .78),
                         grasp_rotation_world=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                         target_name="target_cube_0", target_position_world_m=(.25, 0., .78),
                         target_motion_valid=True, contact_valid=True,
                         left_contact=True, right_contact=True, contacts_simultaneous=True,
                         executed_command_joint_targets_rad=command)
    s = TaskSnapshot(m, "episode_1", "pick_place", TaskPhase.GRASP_LIFT,
                     grasp_attempt_id=attempt, grasp_attempt_active=True, grasp_valid=True,
                     events=("grasp_confirmed",))
    return replace(s, **changes)


class FakeClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeBackend:
    def __init__(self, *, queued=0):
        self.calls = []
        self.queued = queued
        self.pending = 0
        self.attachments = 0
        self.detachments = 0
        self.status = dict(schema=1, request_id="", state="detached", target="", attempt=-1,
                           joint_entity=0, sim_time_s=1., attach_sim_time_s=None,
                           command_rad=0., measured_rad=0., error="")

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        assert isinstance(command, list)
        assert "shell" not in kwargs
        assert kwargs["timeout"] > 0
        assert command[command.index("--reqtype") + 1] == "ignition.msgs.StringMsg"
        assert command[command.index("--reptype") + 1] == "ignition.msgs.StringMsg"
        service = command[command.index("-s") + 1]
        payload = json.loads(command[command.index("--req") + 1][len("data: "):])
        if service.endswith("/command"):
            fields = dict(line.split("=", 1) for line in payload.splitlines())
            self.status.update(request_id=fields["request_id"], target=fields["target"],
                               attempt=int(fields["attempt"]), command_rad=float(fields["command_rad"]),
                               measured_rad=float(fields["measured_rad"]))
            self.final_state = "attached" if fields["action"] == "attach" else "detached"
            self.pending = self.queued
            self.status["state"] = "queued" if self.pending else self.final_state
            if self.final_state == "attached":
                self.attachments += 1
                self.status.update(joint_entity=100 + self.attachments, attach_sim_time_s=self.status["sim_time_s"])
            else:
                self.detachments += 1
                self.status["joint_entity"] = 0
        else:
            assert service.endswith("/status") and payload == ""
            if self.pending:
                self.pending -= 1
                if not self.pending:
                    self.status["state"] = self.final_state
        return self.response(self.status)

    @staticmethod
    def response(status):
        return SimpleNamespace(returncode=0, stdout="data: " + json.dumps(json.dumps(status)) + "\n", stderr="")


def controller(backend=None, **kwargs):
    backend = backend or FakeBackend()
    clock = FakeClock()
    counter = iter(range(1, 100))
    client = GazeboGraspAttachmentController(runner=backend, monotonic=clock.monotonic,
                                            sleep=clock.sleep, request_id_factory=lambda: f"request_{next(counter)}",
                                            **kwargs)
    return client, backend, clock


def test_certified_grasp_records_executed_command_and_actual_angle_separately():
    client, backend, _ = controller()
    result = client.after_step(snapshot(), COMMAND)
    assert result["active"] and result["ever_used"]
    assert result["confirmed_command_rad"] == COMMAND[-1]
    assert result["confirmed_measured_rad"] == pytest.approx(math.radians(7.08))
    assert result["target"] == "target_cube_0" and result["attempt"] == 1
    assert result["state"] == "attached" and result["joint_entity"] == 101
    assert result["events"] == ["grasp_attachment_attached"]
    assert backend.attachments == 1


@pytest.mark.parametrize("field,value", [("grasp_valid", False), ("grasp_attempt_active", False),
                                        ("grasp_attempt_id", 0), ("outcome", "task_failure"),
                                        ("outcome", "success")])
def test_invalid_grasp_certificate_does_not_request_attachment(field, value):
    client, backend, _ = controller()
    client.after_step(snapshot(**{field: value}), COMMAND)
    assert not backend.calls


@pytest.mark.parametrize("field,value", [("evidence_valid", False), ("joint_measurement_valid", False),
                                        ("target_measurement_valid", False), ("target_motion_valid", False),
                                        ("contact_valid", False), ("left_contact", False), ("right_contact", False),
                                        ("contacts_simultaneous", False), ("grasp_evidence_mode", "proxy"),
                                        ("evidence_sim_time_s", .7)])
def test_only_fresh_real_bilateral_contact_can_attach(field, value):
    client, backend, _ = controller()
    s = snapshot()
    client.after_step(replace(s, measurements=replace(s.measurements, **{field: value})), COMMAND)
    assert not backend.calls


def test_attachment_waits_for_matching_backend_confirmation():
    client, backend, clock = controller(FakeBackend(queued=3))
    client.after_step(snapshot(), COMMAND)
    assert client.metadata()["active"]
    assert clock.now == pytest.approx(.15)
    assert len(backend.calls) == 4


def test_release_requires_confirmation_measured_angle_plus_two_degrees_in_command_and_feedback():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    old_threshold = COMMAND[:-1] + (COMMAND[-1] + math.radians(2),)
    client.before_step(old_threshold, snapshot())
    assert backend.detachments == 0
    threshold = math.radians(7.08 + 2.)
    opened = COMMAND[:-1] + (threshold,)
    client.before_step(opened, snapshot())
    assert backend.detachments == 0 and client.metadata()["release_requested"]
    client.after_step(snapshot(command=opened, measured=threshold - 1e-6), opened)
    assert backend.detachments == 0
    result = client.after_step(snapshot(command=opened, measured=threshold), opened)
    assert backend.detachments == 1 and not result["active"] and result["ever_used"]
    assert result["confirmed_command_rad"] == COMMAND[-1]
    assert result["release_reference"] == "measured_confirmation"
    assert result["release_threshold_rad"] == pytest.approx(threshold)
    assert result["events"] == ["grasp_attachment_detached"]


def test_detach_does_not_depend_on_grasp_still_valid_and_never_reattaches_same_attempt():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    opened = COMMAND[:-1] + (math.radians(7.08 + 2.),)
    client.before_step(opened, snapshot(measured=opened[-1], grasp_attempt_active=False, grasp_valid=False))
    client.after_step(snapshot(command=opened), opened)
    client.before_step(COMMAND, snapshot())
    client.after_step(snapshot(), COMMAND)
    assert backend.attachments == backend.detachments == 1


def test_live_penetration_trace_keeps_constraint_until_actual_finger_clearance():
    client, backend, _ = controller()
    caught = COMMAND[:-1] + (math.radians(.397),)
    client.after_step(snapshot(command=caught, measured=math.radians(10.138)), caught)
    tightened = COMMAND[:-1] + (math.radians(-1.603),)
    client.before_step(tightened, snapshot(command=caught, measured=math.radians(10.138)))
    client.after_step(snapshot(command=tightened, measured=math.radians(-1.603)), tightened)
    premature = COMMAND[:-1] + (math.radians(2.397),)
    client.before_step(premature, snapshot(command=tightened, measured=math.radians(-1.603)))
    client.after_step(snapshot(command=premature, measured=math.radians(-.525)), premature)
    assert backend.detachments == 0
    assert client.metadata()["release_threshold_rad"] == pytest.approx(math.radians(12.138))
    clear = COMMAND[:-1] + (math.radians(12.138),)
    client.before_step(clear, snapshot(command=premature, measured=math.radians(-.525)))
    client.after_step(snapshot(command=clear, measured=math.radians(11.9)), clear)
    assert backend.detachments == 0
    client.after_step(snapshot(command=clear, measured=clear[-1]), clear)
    assert backend.detachments == 1


def test_new_closing_command_cancels_release_intent_despite_lagged_measured_opening():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    opened = COMMAND[:-1] + (math.radians(9.08),)
    client.before_step(opened, snapshot())
    assert client.metadata()["release_requested"] and client.metadata()["active"]
    client.before_step(COMMAND, snapshot())
    assert not client.metadata()["release_requested"]
    client.after_step(snapshot(command=COMMAND, measured=opened[-1]), COMMAND)
    assert client.metadata()["active"] and backend.detachments == 0
    client.before_step(opened, snapshot(command=COMMAND, measured=opened[-1]))
    assert backend.detachments == 1


@pytest.mark.parametrize("field,value", [("joint_measurement_valid", False), ("evidence_valid", False),
                                        ("evidence_sim_time_s", .7)])
def test_invalid_or_stale_measured_opening_does_not_detach(field, value):
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    opened = COMMAND[:-1] + (math.radians(9.08),)
    s = snapshot(command=opened, measured=opened[-1])
    s = replace(s, measurements=replace(s.measurements, **{field: value}))
    client.before_step(opened, s)
    client.after_step(s, opened)
    assert backend.detachments == 0 and client.metadata()["active"]


def test_opening_command_without_measured_snapshot_does_not_detach():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    client.before_step(COMMAND[:-1] + (math.radians(90),))
    assert client.metadata()["release_requested"] and backend.detachments == 0


def test_new_real_grasp_attempt_may_attach_again():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    opened = COMMAND[:-1] + (math.radians(7.08 + 3.),)
    client.before_step(opened, snapshot(measured=opened[-1]))
    client.after_step(snapshot(attempt=2), COMMAND)
    assert backend.attachments == 2 and client.metadata()["attempt"] == 2


def test_metadata_is_copied_and_events_clear_at_next_before_step():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    metadata = client.metadata()
    metadata["events"].clear()
    assert client.metadata()["events"] == ["grasp_attachment_attached"]
    client.before_step(COMMAND, snapshot())
    client.after_step(snapshot(), COMMAND)
    assert client.metadata()["events"] == [] and len(backend.calls) == 1


def test_measurement_fields_are_distinct_from_contacts_and_keep_real_status_time():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    fields = client.measurement_fields(1.033, "target_cube_0")
    assert fields["grasp_attachment_enabled"] is True
    assert fields["grasp_attachment_release_delta_rad"] == pytest.approx(math.radians(2.))
    assert fields["grasp_attachment_active"]
    assert fields["grasp_attachment_ever_used"]
    assert fields["grasp_attachment_sim_time_s"] == 1.
    assert not ({"left_contact", "right_contact", "contact_valid"} & fields.keys())
    assert len(backend.calls) == 1


@pytest.mark.parametrize("changes", [{"target": "other_cube"}, {"attempt": 9}, {"request_id": "foreign"},
                                     {"command_rad": 9.}, {"measured_rad": 2.}, {"sim_time_s": float("nan")},
                                     {"schema": True}, {"joint_entity": 0}, {"attach_sim_time_s": None}])
def test_corrupt_or_cross_world_confirmation_is_rejected(changes):
    backend = FakeBackend()
    def runner(command, **kwargs):
        response = backend(command, **kwargs)
        return backend.response({**backend.status, **changes})
    client, _, _ = controller(runner)
    with pytest.raises(GazeboGraspAttachmentError):
        client.after_step(snapshot(), COMMAND)
    assert client.metadata()["state"] == "error"
    assert not client.metadata()["active"]


def test_pending_request_timeout_is_not_treated_as_success():
    client, backend, clock = controller(FakeBackend(queued=999), timeout_s=.11)
    with pytest.raises(GazeboGraspAttachmentError, match="Timed out"):
        client.after_step(snapshot(), COMMAND)
    assert not client.metadata()["active"] and client.metadata()["state"] == "error"
    assert clock.now == pytest.approx(.11)
    with pytest.raises(GazeboGraspAttachmentError):
        client.measurement_fields(1., "target_cube_0")


@pytest.mark.parametrize("failure", ["returncode", "timeout", "malformed", "no_response", "backend_error"])
def test_transport_and_backend_failures_poison_collection(failure):
    backend = FakeBackend()
    def runner(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        response = backend(command, **kwargs)
        if failure == "returncode":
            return SimpleNamespace(returncode=1, stdout="", stderr="failed")
        if failure == "malformed":
            return SimpleNamespace(returncode=0, stdout='data: "not-json"', stderr="")
        if failure == "no_response":
            return SimpleNamespace(returncode=0, stdout="Service unavailable", stderr="")
        return backend.response({**backend.status, "state": "error", "error": "no target model"})
    client, _, _ = controller(runner)
    with pytest.raises(GazeboGraspAttachmentError):
        client.after_step(snapshot(), COMMAND)
    with pytest.raises(GazeboGraspAttachmentError):
        client.before_step(COMMAND, snapshot())


def test_observed_constraint_loss_cannot_silently_preserve_attachment():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    with pytest.raises(GazeboGraspAttachmentError, match="unexpectedly changed"):
        client.observe_status({**backend.status, "state": "detached", "joint_entity": 0})
    with pytest.raises(GazeboGraspAttachmentError):
        client.measurement_fields(1., "target_cube_0")


def test_refresh_is_explicit_and_checks_joint_identity():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    backend.status["sim_time_s"] = 1.033
    client.refresh_status()
    assert client.metadata()["status_sim_time_s"] == 1.033
    backend.status["joint_entity"] = 999
    with pytest.raises(GazeboGraspAttachmentError, match="joint identity"):
        client.refresh_status()


def test_initial_refresh_checks_no_unowned_attachment_exists():
    client, backend, _ = controller()
    client.refresh_status()
    assert not client.metadata()["active"]
    backend.status.update(state="attached", joint_entity=77, attach_sim_time_s=1.)
    with pytest.raises(GazeboGraspAttachmentError, match="unowned"):
        client.refresh_status()


def test_target_switch_and_future_status_are_rejected():
    client, _, _ = controller()
    client.after_step(snapshot(), COMMAND)
    with pytest.raises(GazeboGraspAttachmentError, match="newer"):
        client.measurement_fields(.9, "target_cube_0")
    client, _, _ = controller()
    client.after_step(snapshot(), COMMAND)
    with pytest.raises(GazeboGraspAttachmentError, match="target differs"):
        client.measurement_fields(1., "other_cube")


def test_two_world_controllers_do_not_share_attempt_or_attachment_state():
    first, first_backend, _ = controller(world_name="first")
    second, second_backend, _ = controller(world_name="second")
    first.after_step(snapshot(), COMMAND)
    assert not second.metadata()["active"]
    second.after_step(snapshot(), COMMAND)
    assert first_backend.attachments == second_backend.attachments == 1
    assert first_backend.calls[0][0][3] == "/world/first/so101/grasp_attachment/command"
    assert second_backend.calls[0][0][3] == "/world/second/so101/grasp_attachment/command"


def test_executed_command_disagreement_rejected_before_backend_call():
    client, backend, _ = controller()
    with pytest.raises(GazeboGraspAttachmentError, match="executed command"):
        client.after_step(snapshot(), COMMAND[:-1] + (0.5,))
    assert not backend.calls


def test_new_episode_requires_new_client_even_after_successful_detachment():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    opened = COMMAND[:-1] + (math.radians(7.08 + 3.),)
    client.before_step(opened, snapshot(measured=opened[-1]))
    with pytest.raises(GazeboGraspAttachmentError, match="new physical episode"):
        client.after_step(snapshot(attempt=2, task_instance_id="episode_2"), COMMAND)
    assert backend.attachments == 1


@pytest.mark.parametrize("change", [{"task_instance_id": "episode_2"}, {"grasp_attempt_id": 2}])
def test_active_attachment_identity_cannot_change(change):
    client, _, _ = controller()
    client.after_step(snapshot(), COMMAND)
    with pytest.raises(GazeboGraspAttachmentError, match="without a confirmed release"):
        client.before_step(COMMAND, snapshot(**change))


def test_stale_status_cannot_move_simulation_clock_backwards():
    client, backend, _ = controller()
    client.after_step(snapshot(), COMMAND)
    with pytest.raises(GazeboGraspAttachmentError, match="backwards"):
        client.observe_status({**backend.status, "sim_time_s": .9, "attach_sim_time_s": .9})


def test_streamed_status_avoids_cli_processes_and_requires_completed_step_time():
    cache = GraspAttachmentStateCache()
    client, backend, _ = controller(status_provider=cache.wait_for_sim_time)
    client.after_step(snapshot(), COMMAND)
    before = len(backend.calls)
    cache.update({**backend.status, "sim_time_s": 1.033})
    result = client.refresh_status(minimum_sim_time_s=1.033)
    assert result["active"] and result["status_sim_time_s"] == 1.033
    assert len(backend.calls) == before


@pytest.mark.parametrize("provider", [lambda minimum, timeout: None,
                                      lambda minimum, timeout: FakeBackend().status])
def test_missing_or_old_stream_never_falls_back_to_cli_or_cached_active(provider):
    client, backend, _ = controller(status_provider=provider)
    client.after_step(snapshot(), COMMAND)
    before = len(backend.calls)
    with pytest.raises(GazeboGraspAttachmentError):
        client.refresh_status(minimum_sim_time_s=1.033)
    assert len(backend.calls) == before
    assert client.metadata()["state"] == "error"


def test_source_time_cache_never_relabels_old_status_as_new():
    cache = GraspAttachmentStateCache()
    status = FakeBackend().status
    cache.update(status)
    assert cache.wait_for_sim_time(1.033, 0) is None
    result = cache.wait_for_sim_time(1., 0)
    assert result["sim_time_s"] == 1.
    result["state"] = "bad"
    assert cache.wait_for_sim_time(1., 0)["state"] == "detached"


@pytest.mark.parametrize("invalid", ["bad JSON", {**FakeBackend().status, "sim_time_s": .9}])
def test_corrupt_or_backwards_stream_poison_cache_even_after_later_valid_message(invalid):
    cache = GraspAttachmentStateCache()
    status = FakeBackend().status
    cache.update(status)
    with pytest.raises(GazeboGraspAttachmentError):
        cache.update(invalid)
    with pytest.raises(GazeboGraspAttachmentError):
        cache.wait_for_sim_time(1., 0)
    with pytest.raises(GazeboGraspAttachmentError):
        cache.update({**status, "sim_time_s": 2.})


def test_cache_notifies_waiter_when_new_source_time_arrives():
    from threading import Thread
    cache = GraspAttachmentStateCache()
    results = []
    thread = Thread(target=lambda: results.append(cache.wait_for_sim_time(2., 1.)))
    thread.start()
    cache.update({**FakeBackend().status, "sim_time_s": 2.})
    thread.join(timeout=1.)
    assert not thread.is_alive() and results[0]["sim_time_s"] == 2.


@pytest.mark.parametrize("kwargs", [{"world_name": "empty\nother"}, {"world_name": ""},
                                    {"release_delta_rad": 0}, {"timeout_s": -1},
                                    {"poll_period_s": float("nan")}, {"max_evidence_age_s": True}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        controller(**kwargs)


@pytest.mark.parametrize("command", [(0.,) * 5, (float("nan"),) * 6, (float("inf"),) * 6])
def test_invalid_actions_rejected(command):
    client, backend, _ = controller()
    with pytest.raises(ValueError):
        client.before_step(command)
    assert not backend.calls
