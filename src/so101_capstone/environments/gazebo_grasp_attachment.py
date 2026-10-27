"""Explicit, contact-certified rigid grasp assistance for a single world.

This client never creates contact evidence or advances physics. The Gazebo
plugin confirms constraint creation/removal while the world remains paused.
One controller belongs to one physical episode/world, including preparation
and stage handoffs. A service failure invalidates that world for collection.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from threading import Condition
import time
import uuid
from typing import Any, Callable, Mapping, Sequence

from so101_capstone.rewards.task_state import TaskSnapshot


class GazeboGraspAttachmentError(RuntimeError):
    """Attachment state cannot be certified; do not continue the rollout."""


GRASP_ATTACHMENT_STATE_TOPIC = "/so101/grasp_attachment_state"


class GraspAttachmentStateCache:
    """Source-time constraint status from the world plugin, independently of ROS.

    Invalid or backwards status poisons this world's cache. Waiting for the
    completed physical step never treats an old ``attached`` message as fresh.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._status: dict[str, Any] | None = None
        self._error: str | None = None

    def update(self, value: Mapping[str, Any] | str) -> None:
        try:
            status = GazeboGraspAttachmentController._decode_status(value)
        except GazeboGraspAttachmentError as error:
            with self._condition:
                self._error = str(error)
                self._condition.notify_all()
            raise
        with self._condition:
            if self._error is not None:
                raise GazeboGraspAttachmentError(self._error)
            if self._status is not None and status["sim_time_s"] < self._status["sim_time_s"] - 1e-9:
                self._error = "Attachment stream simulation time moved backwards"
                self._condition.notify_all()
                raise GazeboGraspAttachmentError(self._error)
            self._status = status
            self._condition.notify_all()

    def wait_for_sim_time(self, minimum_sim_time_s: float, timeout_s: float) -> dict[str, Any] | None:
        for name, value in (("minimum_sim_time_s", minimum_sim_time_s), ("timeout_s", timeout_s)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise GazeboGraspAttachmentError(self._error)
                if self._status is not None and self._status["sim_time_s"] + 1e-9 >= minimum_sim_time_s:
                    return dict(self._status)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


class GazeboGraspAttachmentController:
    """Attach only a measured valid grasp, with stage-specific release rules.

    Opening by ``release_delta_rad`` from the measured angle which confirmed
    grasp requests detachment only once actual measured opening reaches that
    threshold too. A grasp attempt can attach only once; a later true
    regrasp with a new attempt id may attach again. Actual measured aperture is
    recorded separately from the executed command, since contact limits motion.
    Formal S2/S3 segments keep the constraint regardless of finger commands;
    preparation and other stages retain the measured-opening release rule.
    """

    def __init__(self, *, world_name: str = "empty",
                 release_delta_rad: float = math.radians(2.0),
                 timeout_s: float = 5.0, poll_period_s: float = 0.05,
                 max_evidence_age_s: float = 0.12,
                 runner: Callable[..., Any] | None = None,
                 status_provider: Callable[[float, float], Mapping[str, Any] | None] | None = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 request_id_factory: Callable[[], str] | None = None) -> None:
        if not isinstance(world_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", world_name):
            raise ValueError("world_name must be a nonempty scoped service token")
        self.world_name = world_name
        self.release_delta_rad = _positive(release_delta_rad, "release_delta_rad")
        self.timeout_s = _positive(timeout_s, "timeout_s")
        self.poll_period_s = _positive(poll_period_s, "poll_period_s")
        self.max_evidence_age_s = _positive(max_evidence_age_s, "max_evidence_age_s")
        self.command_service = f"/world/{world_name}/so101/grasp_attachment/command"
        self.status_service = f"/world/{world_name}/so101/grasp_attachment/status"
        self._runner = subprocess.run if runner is None else runner
        if status_provider is not None and not callable(status_provider):
            raise TypeError("status_provider must be callable")
        self._status_provider = status_provider
        self._monotonic, self._sleep = monotonic, sleep
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)
        self._active = self._ever_used = False
        self._state = "detached"
        self._target: str | None = None
        self._attempt = 0
        self._instance: str | None = None
        self._confirmed_command: float | None = None
        self._confirmed_measured: float | None = None
        self._request_id: str | None = None
        self._joint_entity = 0
        self._status_time: float | None = None
        self._attach_time: float | None = None
        self._events: list[str] = []
        self._used_attempts: set[tuple[str, str, int]] = set()
        self._error: str | None = None
        self._release_requested = False
        self._training_stage: str | None = None

    def configure_training_stage(self, stage: str | None) -> None:
        """Select release behavior without changing a physical constraint.

        ``None`` denotes full-task control or physical preparation. Changing
        stage never detaches immediately; the next measured control step uses
        the selected rule. Constraints still fail on missing/invalid feedback.
        """
        if stage not in {None, "S0", "S1", "S2", "S3", "S4", "S5"}:
            raise ValueError("attachment training stage must be None or S0 through S5")
        if stage != self._training_stage:
            self._training_stage = stage
            self._release_requested = False

    @property
    def release_allowed(self) -> bool:
        return self._training_stage not in {"S2", "S3"}

    @staticmethod
    def _command(command: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in command)
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError("command must contain six finite executed joint targets")
        return values

    def _fail(self, reason: str) -> None:
        self._state, self._error = "error", str(reason)
        raise GazeboGraspAttachmentError(str(reason))

    def _healthy(self) -> None:
        if self._error is not None:
            raise GazeboGraspAttachmentError(self._error)

    @staticmethod
    def _decode_status(value: Mapping[str, Any] | str) -> dict[str, Any]:
        try:
            result = json.loads(value) if isinstance(value, str) else dict(value)
        except (TypeError, ValueError) as error:
            raise GazeboGraspAttachmentError("Malformed attachment JSON status") from error
        if not isinstance(result, dict):
            raise GazeboGraspAttachmentError("Attachment status must be a JSON object")
        if type(result.get("schema")) is not int or result["schema"] != 1:
            raise GazeboGraspAttachmentError("Unsupported attachment status schema")
        if result.get("state") not in {"queued", "attached", "detached", "error"}:
            raise GazeboGraspAttachmentError("Unknown attachment backend state")
        for key in ("request_id", "target"):
            if not isinstance(result.get(key), str):
                raise GazeboGraspAttachmentError(f"Attachment status is missing {key}")
        for key in ("attempt", "joint_entity"):
            if type(result.get(key)) is not int:
                raise GazeboGraspAttachmentError(f"Attachment {key} must be an integer")
        if result["joint_entity"] < 0:
            raise GazeboGraspAttachmentError("Attachment joint entity cannot be negative")
        for key in ("sim_time_s", "command_rad", "measured_rad"):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise GazeboGraspAttachmentError(f"Attachment {key} must be finite")
        if result["sim_time_s"] < 0:
            raise GazeboGraspAttachmentError("Attachment simulation time cannot be negative")
        attach_time = result.get("attach_sim_time_s")
        if attach_time is not None and (isinstance(attach_time, bool)
                or not isinstance(attach_time, (int, float)) or not math.isfinite(attach_time)
                or not 0 <= attach_time <= result["sim_time_s"] + 1e-9):
            raise GazeboGraspAttachmentError("Invalid attachment creation time")
        if result["state"] == "attached" and (result["joint_entity"] <= 0 or attach_time is None):
            raise GazeboGraspAttachmentError("Attached status lacks a confirmed joint entity/time")
        return result

    def _call(self, service: str, payload: str, deadline: float) -> dict[str, Any]:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            self._fail("Timed out confirming Gazebo grasp attachment state")
        command = ["ign", "service", "-s", service, "--reqtype", "ignition.msgs.StringMsg",
                   "--reptype", "ignition.msgs.StringMsg", "--timeout",
                   str(max(1, math.ceil(remaining * 1000))), "--req", f"data: {json.dumps(payload)}"]
        try:
            completed = self._runner(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, check=False, timeout=remaining)
        except (OSError, subprocess.SubprocessError) as error:
            self._fail(f"Gazebo grasp attachment service failed: {error}")
        if completed.returncode != 0:
            self._fail(f"Gazebo grasp attachment service failed ({completed.returncode}): "
                       f"{str(completed.stderr or '').strip()}")
        # ign prints protobuf TextFormat, whose data string contains JSON.
        # Never execute a shell or treat other console lines as response data.
        matches = re.findall(r'^\s*data:\s*("(?:\\.|[^"\\])*")\s*$',
                             completed.stdout or "", re.MULTILINE)
        if len(matches) != 1:
            self._fail("Gazebo attachment service returned no unique StringMsg response")
        try:
            return self._decode_status(json.loads(matches[0]))
        except (ValueError, GazeboGraspAttachmentError) as error:
            self._fail(str(error))

    def _check_identity(self, status: Mapping[str, Any]) -> None:
        if (status["request_id"] != self._request_id or status["target"] != self._target
                or status["attempt"] != self._attempt):
            self._fail("Attachment response request/target/attempt does not match this world transaction")
        if self._status_time is not None and status["sim_time_s"] < self._status_time - 1e-9:
            self._fail("Attachment status simulation time moved backwards")
        if (not math.isclose(status["command_rad"], self._confirmed_command, abs_tol=1e-9)
                or not math.isclose(status["measured_rad"], self._confirmed_measured, abs_tol=1e-9)):
            self._fail("Attachment response changed the recorded confirmation angles")

    def _store_status(self, status: Mapping[str, Any]) -> None:
        self._status_time = float(status["sim_time_s"])
        self._attach_time = status.get("attach_sim_time_s")
        self._joint_entity = status["joint_entity"]
        self._state = status["state"]

    def _transact(self, action: str) -> None:
        request_id = self._request_id_factory()
        if (not isinstance(request_id, str) or len(request_id) > 128
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id)):
            self._fail("Invalid attachment request id")
        self._request_id = request_id
        payload = "\n".join((f"request_id={request_id}", f"action={action}", f"target={self._target}",
                             f"attempt={self._attempt}", f"command_rad={self._confirmed_command:.17g}",
                             f"measured_rad={self._confirmed_measured:.17g}"))
        deadline = self._monotonic() + self.timeout_s
        self._state = f"{action}_pending"
        status = self._call(self.command_service, payload, deadline)
        expected = "attached" if action == "attach" else "detached"
        while True:
            self._check_identity(status)
            if status["state"] == "error":
                self._fail(f"Gazebo rejected grasp {action}: {status.get('error') or 'unspecified error'}")
            if status["state"] == expected:
                self._store_status(status)
                self._active = action == "attach"
                self._ever_used = self._ever_used or self._active
                self._events.append(f"grasp_attachment_{expected}")
                return
            if status["state"] != "queued":
                self._fail(f"Unexpected attachment state {status['state']!r} while confirming {action}")
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                self._fail(f"Timed out confirming grasp {action}; world must be discarded")
            self._sleep(min(self.poll_period_s, remaining))
            status = self._call(self.status_service, "", deadline)

    def before_step(self, command: Sequence[float], snapshot: TaskSnapshot | None = None) -> dict[str, Any]:
        """Release only where allowed and measured fingers are already clear."""
        self._healthy()
        values = self._command(command)
        self._events.clear()
        self._release_requested = False
        if self._active:
            if snapshot is not None and (snapshot.target_name != self._target
                    or snapshot.task_instance_id != self._instance
                    or snapshot.grasp_attempt_id != self._attempt):
                self._fail("Attached target/attempt changed without a confirmed release")
            self._release_when_clear(values, snapshot)
        return self.metadata()

    def _release_when_clear(self, command: Sequence[float], snapshot: TaskSnapshot | None) -> None:
        # Formal grasp/lift and carry training cannot drop the target merely
        # by opening the fingers. Keep the certified physical constraint and
        # leave commands untouched so the reward can teach aperture control.
        if not self.release_allowed:
            self._release_requested = False
            return
        # The fixed constraint can suppress collision feedback while the
        # moving jaw closes into the object. Re-enabling collisions before the
        # jaw actually reopens can eject it. Never substitute a commanded
        # angle for measured aperture, and never carry an old opening intent
        # across a new closing action.
        threshold = self._confirmed_measured + self.release_delta_rad
        self._release_requested = bool(command[-1] >= threshold - 1e-12)
        if not self._release_requested or snapshot is None:
            return
        m = snapshot.measurements
        measured_clear = bool(m.joint_measurement_valid and m.evidence_valid
                              and -1e-9 <= m.sim_time_s - m.evidence_time <= self.max_evidence_age_s
                              and m.joint_positions_rad[-1] >= threshold - 1e-12)
        if measured_clear:
            self._transact("detach")

    def after_step(self, snapshot: TaskSnapshot, command: Sequence[float]) -> dict[str, Any]:
        """Certify and attach a new grasp only after real synchronized feedback."""
        self._healthy()
        values = self._command(command)
        executed = snapshot.measurements.executed_command_joint_targets_rad
        if executed is not None and any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(executed, values)):
            self._fail("Attachment command differs from the measured transition's executed command")
        if self._active:
            if (snapshot.target_name != self._target or snapshot.task_instance_id != self._instance
                    or snapshot.grasp_attempt_id != self._attempt):
                self._fail("Attached target/attempt changed without a confirmed release")
            self._release_when_clear(values, snapshot)
            return self.metadata()
        m = snapshot.measurements
        key = (snapshot.task_instance_id, snapshot.target_name, snapshot.grasp_attempt_id)
        if key in self._used_attempts:
            return self.metadata()
        age = m.sim_time_s - m.evidence_time
        certified = (snapshot.outcome == "running" and snapshot.grasp_valid
                     and snapshot.grasp_attempt_active and snapshot.grasp_attempt_id > 0
                     and snapshot.target_name is not None and m.grasp_evidence_mode == "contact"
                     and m.evidence_valid and m.joint_measurement_valid and m.target_measurement_valid
                     and m.target_motion_valid and m.contact_valid and m.contacts_simultaneous
                     and m.left_contact and m.right_contact and -1e-9 <= age <= self.max_evidence_age_s)
        if not certified:
            return self.metadata()
        if len(snapshot.target_name) > 128 or not re.fullmatch(r"target_cube_[A-Za-z0-9_.-]+", snapshot.target_name):
            self._fail("Target name is invalid for the attachment protocol")
        if self._instance is not None and self._instance != snapshot.task_instance_id:
            self._fail("Create a new attachment controller for a new physical episode")
        self._target, self._attempt, self._instance = snapshot.target_name, snapshot.grasp_attempt_id, snapshot.task_instance_id
        self._confirmed_command, self._confirmed_measured = values[-1], float(m.joint_positions_rad[-1])
        self._release_requested = False
        self._used_attempts.add(key)
        self._transact("attach")
        return self.metadata()

    def observe_status(self, value: Mapping[str, Any] | str) -> dict[str, Any]:
        """Accept a plugin status message without inventing a new timestamp."""
        self._healthy()
        try:
            status = self._decode_status(value)
        except GazeboGraspAttachmentError as error:
            self._fail(str(error))
        if self._request_id is None:
            if status["state"] != "detached" or status["joint_entity"] != 0:
                self._fail("World already contains an unowned grasp attachment")
            self._store_status(status)
            return self.metadata()
        self._check_identity(status)
        if status["state"] != self._state:
            self._fail(f"Grasp attachment unexpectedly changed from {self._state} to {status['state']}")
        if self._active and status["joint_entity"] != self._joint_entity:
            self._fail("Active grasp attachment joint identity changed")
        self._store_status(status)
        return self.metadata()

    def refresh_status(self, *, minimum_sim_time_s: float | None = None) -> dict[str, Any]:
        """Verify fresh streamed status; services remain for initialization.

        The coordinator supplies the completed step's simulation time. A
        missing/stale stream fails the step; there is no service fallback that
        silently conceals a broken bridge. Standalone clients without a stream
        can still perform an explicit service query.
        """
        self._healthy()
        if minimum_sim_time_s is not None:
            if (isinstance(minimum_sim_time_s, bool) or not math.isfinite(float(minimum_sim_time_s))
                    or minimum_sim_time_s < 0):
                raise ValueError("minimum_sim_time_s must be finite and nonnegative")
        if self._status_provider is not None and minimum_sim_time_s is not None:
            try:
                status = self._status_provider(minimum_sim_time_s, self.timeout_s)
            except GazeboGraspAttachmentError as error:
                self._fail(str(error))
            if status is None:
                self._fail(f"Timed out waiting for grasp attachment stream at simulation time {minimum_sim_time_s:.9f}")
        else:
            status = self._call(self.status_service, "", self._monotonic() + self.timeout_s)
        result = self.observe_status(status)
        if minimum_sim_time_s is not None and result["status_sim_time_s"] + 1e-9 < minimum_sim_time_s:
            self._fail("Attachment status is older than the completed simulation step")
        return result

    def measurement_fields(self, sim_time_s: float, target_name: str | None) -> dict[str, Any]:
        """Return separate attachment evidence, never fabricated contact flags."""
        self._healthy()
        if isinstance(sim_time_s, bool) or not math.isfinite(float(sim_time_s)) or sim_time_s < 0:
            raise ValueError("sim_time_s must be finite and non-negative")
        if self._active and target_name != self._target:
            self._fail("Active attachment target differs from measured target")
        if self._status_time is not None and self._status_time > sim_time_s + 1e-9:
            self._fail("Attachment status is newer than the completed simulation step")
        return {"grasp_attachment_enabled": True,
                "grasp_attachment_release_delta_rad": self.release_delta_rad,
                "grasp_attachment_active": self._active,
                "grasp_attachment_ever_used": self._ever_used,
                "grasp_attachment_target_name": self._target,
                "grasp_attachment_attempt_id": self._attempt,
                "grasp_attachment_command_rad": self._confirmed_command,
                "grasp_attachment_measured_rad": self._confirmed_measured,
                "grasp_attachment_sim_time_s": self._status_time}

    def metadata(self) -> dict[str, Any]:
        return {"active": self._active, "ever_used": self._ever_used,
                "target": self._target, "attempt": self._attempt,
                "confirmed_command_rad": self._confirmed_command,
                "confirmed_measured_rad": self._confirmed_measured,
                "request_id": self._request_id, "events": list(self._events),
                "state": self._state, "joint_entity": self._joint_entity,
                "status_sim_time_s": self._status_time, "attach_sim_time_s": self._attach_time,
                "release_delta_rad": self.release_delta_rad,
                "training_stage": self._training_stage,
                "release_mode": "measured_opening" if self.release_allowed else "stage_persistent",
                "release_allowed": self.release_allowed,
                "release_reference": "measured_confirmation",
                "release_threshold_rad": (None if self._confirmed_measured is None
                                          else self._confirmed_measured + self.release_delta_rad),
                "release_requested": self._release_requested, "error": self._error}
