"""Measured task semantics for pick/place; independent of ROS and the actor.

The environment aligns sensors before constructing ``TaskMeasurements``.  The
tracker consumes evidence time, never wall time, and never commands the robot.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .pick_place_reward import PickPlaceRewardConfig


class TaskPhase(str, Enum):
    WAIT_HOME = "wait_home"
    APPROACH = "approach"
    GRASP_LIFT = "grasp_lift"
    TRANSPORT = "transport"
    PLACE_RELEASE = "place_release"
    RETURN_HOME = "return_home"
    HOME_HOLD = "home_hold"


TASK_MODES = ("pick_place", "idle_empty", "idle_completed")
TASK_CONTEXT_SCHEMA_VERSION = 2
GRASP_ATTACHMENT_CONTEXT_DIM = 6


def _vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(v) for v in a)


def quaternion_rotation(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(_vector(quaternion, 4, "quaternion"))
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("quaternion must be nonzero")
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def bounded_score(error: float, scale: float) -> float:
    return float(1.0 - np.clip(error / scale, 0.0, 1.0))


@dataclass(frozen=True)
class TaskMeasurements:
    sim_time_s: float
    joint_positions_rad: Sequence[float]
    joint_velocities_rad_s: Sequence[float]
    grasp_position_world_m: Sequence[float]
    grasp_rotation_world: Sequence[Sequence[float]]
    target_name: str | None = None
    target_position_world_m: Sequence[float] | None = None
    target_quaternion_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0)
    target_linear_velocity_m_s: Sequence[float] = (0.0, 0.0, 0.0)
    target_angular_velocity_rad_s: Sequence[float] = (0.0, 0.0, 0.0)
    target_motion_valid: bool = False
    target_measurement_valid: bool = True
    joint_measurement_valid: bool = True
    evidence_valid: bool = True
    evidence_sim_time_s: float | None = None
    contact_valid: bool = False
    left_contact: bool = False
    right_contact: bool = False
    contacts_simultaneous: bool = False
    goal_center_world_m: Sequence[float] = (0.25, 0.0, 0.753)
    goal_half_extents_xy_m: Sequence[float] = (0.05, 0.05)
    goal_yaw_rad: float = 0.0
    goal_support_z_m: float = 0.753
    target_half_extents_m: Sequence[float] = (0.015, 0.015, 0.015)
    source_support_z_m: float = 0.750
    failure_reason: str | None = None
    grasp_evidence_mode: str = "contact"
    executed_command_joint_targets_rad: Sequence[float] | None = None
    grasp_attachment_enabled: bool = False
    grasp_attachment_active: bool = False
    grasp_attachment_ever_used: bool = False
    grasp_attachment_target_name: str | None = None
    grasp_attachment_attempt_id: int = 0
    grasp_attachment_command_rad: float | None = None
    grasp_attachment_measured_rad: float | None = None
    grasp_attachment_sim_time_s: float | None = None
    grasp_attachment_release_delta_rad: float = math.radians(2.0)

    def __post_init__(self) -> None:
        for name, size in (("joint_positions_rad", 6), ("joint_velocities_rad_s", 6),
                           ("grasp_position_world_m", 3), ("target_quaternion_xyzw", 4),
                           ("target_linear_velocity_m_s", 3), ("target_angular_velocity_rad_s", 3),
                           ("goal_center_world_m", 3), ("goal_half_extents_xy_m", 2),
                           ("target_half_extents_m", 3)):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))
        r = np.asarray(self.grasp_rotation_world, dtype=float)
        if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-3) or np.linalg.det(r) < 0.99:
            raise ValueError("grasp_rotation_world must be a rotation matrix")
        object.__setattr__(self, "grasp_rotation_world", tuple(tuple(float(v) for v in row) for row in r))
        if (self.target_name is None) != (self.target_position_world_m is None):
            raise ValueError("target identity and position must be supplied together")
        if self.target_name is not None:
            if not str(self.target_name).strip():
                raise ValueError("target_name must not be empty")
            object.__setattr__(self, "target_position_world_m", _vector(self.target_position_world_m, 3, "target_position_world_m"))
        quaternion_rotation(self.target_quaternion_xyzw)
        if self.executed_command_joint_targets_rad is not None:
            object.__setattr__(self, "executed_command_joint_targets_rad", _vector(self.executed_command_joint_targets_rad, 6, "executed_command_joint_targets_rad"))
        for name in ("sim_time_s", "goal_yaw_rad", "goal_support_z_m", "source_support_z_m"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.evidence_sim_time_s is not None and (not math.isfinite(self.evidence_sim_time_s) or self.evidence_sim_time_s > self.sim_time_s + 1e-6):
            raise ValueError("evidence time must be finite and not in the future")
        if min(self.goal_half_extents_xy_m) <= 0 or min(self.target_half_extents_m) <= 0:
            raise ValueError("geometry extents must be positive")
        if self.grasp_evidence_mode not in ("contact", "proxy"):
            raise ValueError("unknown grasp_evidence_mode")
        for name in ("grasp_attachment_command_rad", "grasp_attachment_measured_rad", "grasp_attachment_sim_time_s"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.grasp_attachment_release_delta_rad) or self.grasp_attachment_release_delta_rad <= 0:
            raise ValueError("grasp attachment release delta must be positive and finite")
        if self.grasp_attachment_active:
            if (not self.grasp_attachment_enabled or not self.grasp_attachment_ever_used
                    or self.grasp_attachment_target_name != self.target_name or self.target_name is None
                    or self.grasp_attachment_attempt_id < 1
                    or self.grasp_attachment_command_rad is None or self.grasp_attachment_measured_rad is None
                    or self.grasp_attachment_sim_time_s is None
                    or self.grasp_attachment_sim_time_s > self.sim_time_s + 1e-6):
                raise ValueError("active attachment requires matching target, certified attempt and confirmed angles/time")

    @property
    def evidence_time(self) -> float:
        return self.sim_time_s if self.evidence_sim_time_s is None else self.evidence_sim_time_s


def object_geometry(m: TaskMeasurements, margin_m: float = 0.0) -> tuple[bool, float, float]:
    """Entire oriented footprint membership, bottom height, goal XY distance."""
    if m.target_position_world_m is None:
        return False, 0.0, 0.0
    center = np.asarray(m.target_position_world_m)
    rotation = quaternion_rotation(m.target_quaternion_xyzw)
    half = np.asarray(m.target_half_extents_m)
    cy, sy = math.cos(m.goal_yaw_rad), math.sin(m.goal_yaw_rad)
    goal_rotation = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    relative = goal_rotation.T @ (center - np.asarray(m.goal_center_world_m))
    projected_half = np.abs(goal_rotation.T @ rotation) @ half
    inside = bool(np.all(np.abs(relative[:2]) + projected_half[:2] + margin_m <= np.asarray(m.goal_half_extents_xy_m) + 1e-9))
    bottom = float(center[2] - np.abs(rotation[2]) @ half)
    return inside, bottom, float(np.linalg.norm(relative[:2]))


@dataclass(frozen=True)
class TaskSnapshot:
    measurements: TaskMeasurements
    task_instance_id: str
    task_mode: str
    phase: TaskPhase
    outcome: str = "running"
    failure_reason: str | None = None
    success_event: bool = False
    events: tuple[str, ...] = ()
    grasp_attempt_id: int = 0
    grasp_attempt_active: bool = False
    grasp_valid: bool = False
    lifted_once: bool = False
    attempt_lifted: bool = False
    carried_into_goal: bool = False
    valid_release: bool = False
    delivery_valid: bool = False
    arrived_via_valid_carry: bool = False
    local_recovery_authorized: bool = False
    goal_recovery_authorized: bool = False
    placed_stable: bool = False
    target_in_goal: bool = False
    target_supported: bool = False
    target_static: bool = False
    robot_at_home: bool = False
    robot_static: bool = False
    pregrasp_ready: bool = False
    pregrasp_ready_duration_s: float = 0.0
    contact_duration_s: float = 0.0
    lift_duration_s: float = 0.0
    place_stable_duration_s: float = 0.0
    home_stable_duration_s: float = 0.0
    home_hold_required_s: float = 3.0
    home_error_rad: float = 0.0
    max_joint_velocity_rad_s: float = 0.0
    target_distance_m: float = 0.0
    target_bottom_z_m: float = 0.0
    approach_score: float = 0.0
    approach_distance_score: float = 0.0
    approach_orientation_score: float = 0.0
    approach_opening_score: float = 0.0
    grasp_close_gate: float = 0.0
    grasp_closure_score: float = 0.0
    contact_score: float = 0.0
    lift_score: float = 0.0
    transport_score: float = 0.0
    support_score: float = 0.0
    separation_score: float = 0.0
    home_score: float = 0.0
    source_position_xy_m: tuple[float, ...] | None = None
    recovery_position_xy_m: tuple[float, ...] | None = None
    grasp_relative_anchor_m: tuple[float, ...] | None = None
    grasp_origin_bottom_z_m: float = 0.0

    @property
    def sim_time_s(self) -> float:
        return self.measurements.sim_time_s

    @property
    def evidence_sim_time_s(self) -> float:
        return self.measurements.evidence_time

    @property
    def target_name(self) -> str | None:
        return self.measurements.target_name

    def to_info(self) -> dict[str, object]:
        result = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "measurements"}
        result.update(task_phase=self.phase.value, task_phase_name=self.phase.name,
                      task_outcome=self.outcome, phase=self.phase.value,
                      sim_time_s=self.sim_time_s, evidence_sim_time_s=self.evidence_sim_time_s,
                      target_name=self.target_name, target_motion_valid=self.measurements.target_motion_valid,
                      contact_measurement_valid=self.measurements.contact_valid,
                      grasp_evidence_mode=self.measurements.grasp_evidence_mode,
                      formal_success=self.outcome == "success" and self.measurements.grasp_evidence_mode == "contact",
                      physical_target_present=self.target_name is not None,
                      goal_center_world_m=self.measurements.goal_center_world_m,
                      goal_half_extents_xy_m=self.measurements.goal_half_extents_xy_m,
                      goal_yaw_rad=self.measurements.goal_yaw_rad,
                      target_half_extents_m=self.measurements.target_half_extents_m,
                      source_support_z_m=self.measurements.source_support_z_m,
                      task_context_schema_version=TASK_CONTEXT_SCHEMA_VERSION)
        result["critic_task_context"] = encode_task_context(self).tolist()
        result.update({f.name: getattr(self.measurements, f.name) for f in fields(self.measurements)
                       if f.name.startswith("grasp_attachment_")})
        # Stage success remains a valid result for this explicitly assisted task;
        # it must never be presented as proof of an unassisted friction grasp.
        result["unassisted_success"] = bool(result["formal_success"] and not self.measurements.grasp_attachment_ever_used)
        result["success_physics"] = "rigid_grasp_assisted" if self.measurements.grasp_attachment_ever_used else "unassisted"
        return result


_CONTEXT_FLAGS = ("grasp_valid", "grasp_attempt_active", "lifted_once", "attempt_lifted", "carried_into_goal", "valid_release",
                  "delivery_valid", "arrived_via_valid_carry", "local_recovery_authorized", "goal_recovery_authorized", "placed_stable",
                  "target_in_goal", "target_supported", "target_static", "robot_at_home", "robot_static", "pregrasp_ready")
_POTENTIAL_FEATURES = ("approach_score", "contact_score", "lift_score", "transport_score",
                       "support_score", "separation_score", "home_score", "approach_distance_score",
                       "approach_orientation_score", "approach_opening_score", "grasp_close_gate", "grasp_closure_score")
TASK_CONTEXT_DIM = 7 + 3 + len(_CONTEXT_FLAGS) + 7 + 4 + 3 * 4 + 4 + 11 + 6 + 3 + len(_POTENTIAL_FEATURES) + 7


def encode_task_context(snapshot: TaskSnapshot) -> np.ndarray:
    """Fixed schema; actor observations are deliberately unchanged."""
    s, m = snapshot, snapshot.measurements
    target = np.asarray(m.target_position_world_m or m.grasp_position_world_m)
    values = [float(s.phase is p) for p in TaskPhase] + [float(s.task_mode == mode) for mode in TASK_MODES]
    values += [float(getattr(s, name)) for name in _CONTEXT_FLAGS]
    values += [s.contact_duration_s, s.lift_duration_s, s.place_stable_duration_s,
               s.home_stable_duration_s, s.home_hold_required_s, min(s.grasp_attempt_id, 100) / 100,
               s.pregrasp_ready_duration_s]
    values += [float(m.evidence_valid), float(m.joint_measurement_valid), float(m.target_motion_valid), float(m.contact_valid)]
    values += list(target - np.asarray(m.grasp_position_world_m))
    values += list(target - np.asarray(m.goal_center_world_m))
    values += list(m.target_linear_velocity_m_s) + list(m.target_angular_velocity_rad_s)
    values += list(m.goal_half_extents_xy_m) + [m.goal_support_z_m, max(0.0, m.sim_time_s-m.evidence_time)]
    for anchor, size in ((s.source_position_xy_m, 2), (s.recovery_position_xy_m, 2), (s.grasp_relative_anchor_m, 3)):
        values += [float(anchor is not None)] + list((0.0,) * size if anchor is None else anchor)
    values += [s.grasp_origin_bottom_z_m]
    values += list(m.joint_velocities_rad_s)
    values += [float(m.left_contact), float(m.right_contact), float(m.contacts_simultaneous)]
    values += [float(getattr(s, name)) for name in _POTENTIAL_FEATURES]
    values += list(m.joint_positions_rad if m.executed_command_joint_targets_rad is None else m.executed_command_joint_targets_rad)
    values += [float(m.executed_command_joint_targets_rad is not None)]
    if m.grasp_attachment_enabled:
        values += [float(m.grasp_attachment_active), float(m.grasp_attachment_ever_used),
                   m.grasp_attachment_command_rad or 0.0, m.grasp_attachment_measured_rad or 0.0,
                   m.grasp_attachment_release_delta_rad, min(m.grasp_attachment_attempt_id, 100) / 100.0]
    result = np.asarray(values, dtype=np.float32)
    expected_dim = TASK_CONTEXT_DIM + (GRASP_ATTACHMENT_CONTEXT_DIM if m.grasp_attachment_enabled else 0)
    if result.shape != (expected_dim,) or not np.isfinite(result).all():
        raise ValueError("invalid critic task context")
    return result


class TaskStateTracker:
    """Ordered grasp/carry/release certification and measured dwell timers."""

    def __init__(self, config: PickPlaceRewardConfig):
        self.config = config
        self._initialized = False

    def reset(self, measurements: TaskMeasurements, task_mode: str = "pick_place", *,
              task_instance_id: str = "episode", home_hold_required_s: float | None = None) -> TaskSnapshot:
        self._initialized = False
        if measurements.grasp_attachment_active:
            raise ValueError("cannot reset task semantics while a grasp attachment is active")
        if task_mode not in TASK_MODES:
            raise ValueError("unknown task_mode")
        if not task_instance_id:
            raise ValueError("task_instance_id must not be empty")
        self.mode, self.instance = task_mode, task_instance_id
        self.required_hold = self.config.home_hold_min_s if home_hold_required_s is None else float(home_hold_required_s)
        if not math.isfinite(self.required_hold) or self.required_hold < self.config.home_hold_min_s:
            raise ValueError("home_hold_required_s is below configured minimum")
        self._target = measurements.target_name
        self._source_xy = None if measurements.target_position_world_m is None else np.asarray(measurements.target_position_world_m[:2])
        self._recovery_xy = None
        self._relative_anchor = None
        self._grasp_origin_bottom = 0.0
        self._attempt = 0
        self._attempt_active = self._attempt_lifted = self._lifted_once = False
        self._carried = self._arrived = self._release = self._delivery = False
        self._goal_recovery_token = False
        self._contact_s = self._lift_s = self._place_s = self._home_s = 0.0
        self._pregrasp_s = 0.0
        self._outcome, self._failure = "running", None
        self._previous_measurement = measurements
        self._last_evidence = measurements.evidence_time
        if measurements.grasp_evidence_mode != self.config.grasp_evidence_mode:
            raise ValueError("measurement and configured grasp evidence modes differ")
        inside, bottom, _ = object_geometry(measurements, self.config.goal_margin_m)
        if task_mode == "idle_empty" and measurements.target_name is not None:
            raise ValueError("idle_empty must start without a target")
        if task_mode == "idle_completed" and (measurements.target_name is None or not inside or abs(bottom-measurements.goal_support_z_m) > self.config.support_height_tolerance_m):
            raise ValueError("idle_completed requires an initially supported object in goal")
        if task_mode == "pick_place" and measurements.target_name is not None and inside:
            raise ValueError("pick_place must start with target outside goal")
        self._snapshot = self._evaluate(measurements, 0.0, fresh=False)
        self._initialized = True
        return self._snapshot

    @property
    def snapshot(self) -> TaskSnapshot:
        if not self._initialized:
            raise RuntimeError("reset must be called first")
        return self._snapshot

    def annotate_attachment_measurements(self, measurements: TaskMeasurements) -> TaskSnapshot:
        """Record a paused-boundary acknowledgement without advancing evidence.

        All sensor values, reward events and dwell timers stay unchanged. Only
        the new constraint metadata is allowed to enter the current snapshot.
        """
        previous = self.snapshot.measurements
        for field in fields(previous):
            if not field.name.startswith("grasp_attachment_") and getattr(previous, field.name) != getattr(measurements, field.name):
                raise ValueError("attachment annotation cannot alter sensor measurements")
        if measurements.grasp_attachment_enabled != previous.grasp_attachment_enabled:
            raise ValueError("attachment annotation cannot change task physics mode")
        if measurements.grasp_attachment_active and (not self._snapshot.grasp_valid
                or not self._attempt_active or measurements.grasp_attachment_attempt_id != self._attempt):
            raise ValueError("attachment annotation requires the same contact-certified grasp attempt")
        self._snapshot = replace(self._snapshot, measurements=measurements)
        self._previous_measurement = measurements
        return self._snapshot

    def update(self, measurements: TaskMeasurements, task_mode: str | None = None, *,
               allow_success: bool = True) -> TaskSnapshot:
        """Advance physical evidence; optionally leave success to a stage judge.

        Assisted preparation and S2--S5 retain all contact/carry/release facts
        but do not latch the separate full-task HOME success. Physical failures
        remain terminal regardless of this flag. Existing callers keep True.
        """
        if not self._initialized:
            raise RuntimeError("reset must be called first")
        if task_mode is not None and task_mode != self.mode:
            raise ValueError("task mode cannot change within an episode")
        if measurements.grasp_evidence_mode != self.config.grasp_evidence_mode:
            raise ValueError("grasp evidence mode cannot change within an episode")
        if measurements.grasp_attachment_enabled != self._previous_measurement.grasp_attachment_enabled:
            raise ValueError("grasp attachment mode cannot change within an episode")
        if measurements.sim_time_s <= self._previous_measurement.sim_time_s:
            raise ValueError("control simulation time must increase; reset after a clock reset")
        if measurements.evidence_time < self._last_evidence-1e-9:
            raise ValueError("evidence time moved backwards")
        if self._target is not None and measurements.target_name != self._target:
            raise ValueError("active target disappeared or changed identity")
        if self.mode == "idle_empty" and measurements.target_name is not None:
            raise ValueError("idle_empty cannot acquire a target")
        if self._target is None and measurements.target_name is not None:
            if object_geometry(measurements, self.config.goal_margin_m)[0]:
                raise ValueError("pick_place target must appear outside goal")
            self._target = measurements.target_name
            self._source_xy = np.asarray(measurements.target_position_world_m[:2])
        dt = measurements.evidence_time-self._last_evidence
        valid = measurements.evidence_valid and measurements.joint_measurement_valid
        if measurements.target_name is not None:
            contact_known = measurements.contact_valid or self.config.grasp_evidence_mode == "proxy"
            valid = valid and measurements.target_measurement_valid and contact_known and measurements.target_motion_valid
        fresh = bool(valid and 0 < dt <= self.config.max_evidence_gap_s)
        if not valid or dt > self.config.max_evidence_gap_s:
            self._contact_s = self._lift_s = self._place_s = self._home_s = 0.0
            self._pregrasp_s = 0.0
            # Contact endpoints either side of an unknown interval do not prove
            # continuous carrying. A later attempt must certify a fresh lift.
            if self._attempt_active:
                if self._attempt_lifted and self._previous_measurement.target_position_world_m is not None:
                    self._recovery_xy = np.asarray(self._previous_measurement.target_position_world_m[:2]).copy()
                self._attempt_active = self._attempt_lifted = self._carried = False
            self._relative_anchor = None
        previous_snapshot = self._snapshot
        self._snapshot = self._evaluate(measurements, dt if fresh else 0.0, fresh=fresh,
                                        allow_success=allow_success)
        if not valid or dt > self.config.max_evidence_gap_s:
            # Unknown data cannot assert a new physical phase or geometric
            # progress. Dwell/continuity still reset, since no covered interval
            # can be certified. This is not a task failure.
            old = previous_snapshot
            self._snapshot = replace(self._snapshot, phase=old.phase,
                                     **{name: getattr(old, name) for name in _POTENTIAL_FEATURES})
        self._last_evidence = measurements.evidence_time
        self._previous_measurement = measurements
        return self._snapshot

    def _evaluate(self, m: TaskMeasurements, dt: float, *, fresh: bool,
                  allow_success: bool = True) -> TaskSnapshot:
        cfg = self.config
        events: list[str] = []
        qerr = np.abs(np.asarray(m.joint_positions_rad)-np.asarray(cfg.home_joint_positions_rad))
        qvel = np.abs(m.joint_velocities_rad_s)
        at_home = bool(m.joint_measurement_valid and np.all(qerr <= cfg.home_joint_tolerance_rad+1e-9))
        static = bool(m.joint_measurement_valid and np.max(qvel) <= cfg.home_velocity_tolerance_rad_s+1e-9)
        effective_error = np.maximum(qerr-cfg.home_joint_tolerance_rad, 0)
        # Retain guidance well outside the HOME tolerance, including when a
        # non-worst joint improves while another joint is still far away.
        home_joint_scores = np.exp(-effective_error/np.asarray(cfg.home_position_scales_rad))
        home_score = float(0.5 * np.mean(home_joint_scores) + 0.5 * np.min(home_joint_scores))
        has_target = m.target_name is not None
        inside, bottom, goal_distance = object_geometry(m, cfg.goal_margin_m)
        supported = bool(has_target and abs(bottom-m.goal_support_z_m) <= cfg.support_height_tolerance_m)
        target_static = bool(has_target and m.target_motion_valid and np.linalg.norm(m.target_linear_velocity_m_s) <= cfg.object_linear_velocity_tolerance_m_s and np.linalg.norm(m.target_angular_velocity_rad_s) <= cfg.object_angular_velocity_tolerance_rad_s)
        distance = 0.0 if not has_target else float(np.linalg.norm(np.asarray(m.target_position_world_m)-m.grasp_position_world_m))
        dual = bool(has_target and m.contact_valid and m.contacts_simultaneous and m.left_contact and m.right_contact and distance <= cfg.grasp_candidate_distance_m)
        if cfg.grasp_evidence_mode == "proxy":
            # Explicit diagnostic mode, never reported as formal contact success.
            opening = m.joint_positions_rad[-1]-cfg.home_joint_positions_rad[-1]
            dual = bool(has_target and distance <= cfg.grasp_candidate_distance_m and opening <= cfg.approach_opening_min_rad)
        relative = None if not has_target else np.asarray(m.grasp_rotation_world).T @ (np.asarray(m.target_position_world_m)-m.grasp_position_world_m)
        source_allowed = bool(has_target and self._source_xy is not None and np.linalg.norm(np.asarray(m.target_position_world_m[:2])-self._source_xy) <= cfg.source_adjustment_tolerance_m)
        if fresh and has_target:
            # Recovery permissions are revocable tokens, not permanent history.
            if self._recovery_xy is not None and np.linalg.norm(np.asarray(m.target_position_world_m[:2])-self._recovery_xy) > cfg.source_adjustment_tolerance_m:
                self._recovery_xy = None
            if goal_distance > cfg.goal_recovery_radius_m:
                self._goal_recovery_token = False
        recovery_allowed = bool(has_target and self._recovery_xy is not None and np.linalg.norm(np.asarray(m.target_position_world_m[:2])-self._recovery_xy) <= cfg.source_adjustment_tolerance_m)
        if self._goal_recovery_token and has_target:
            recovery_allowed = recovery_allowed or goal_distance <= cfg.goal_recovery_radius_m
        relative_stable = bool(relative is not None and (self._relative_anchor is None or np.linalg.norm(relative-self._relative_anchor) <= cfg.relative_grasp_tolerance_m))
        attached = bool(m.grasp_attachment_active and self._attempt_active
                        and m.grasp_attachment_attempt_id == self._attempt
                        and m.grasp_evidence_mode == "contact")
        # A weld may suppress parent/child collision contacts. It can sustain
        # only the already certified attempt, never fabricate acquisition.
        held = bool((dual or attached) and relative_stable)
        grasp = bool(held and self._contact_s+1e-9 >= cfg.grasp_confirm_s and self._attempt_active)
        if fresh and self.mode == "pick_place" and has_target:
            if held:
                if self._relative_anchor is None:
                    self._relative_anchor = relative.copy()
                    self._grasp_origin_bottom = bottom
                self._contact_s = min(self._contact_s+dt, cfg.grasp_confirm_s)
                # A pushed-in object cannot become a delivery by being lifted
                # inside the goal. Local goal recovery requires carried entry.
                acquisition_allowed = (source_allowed or recovery_allowed) and (not inside or self._goal_recovery_token)
                if dual and not self._attempt_active and self._contact_s+1e-9 >= cfg.grasp_confirm_s and acquisition_allowed:
                    self._attempt += 1
                    self._attempt_active = True
                    self._attempt_lifted = self._carried = self._release = self._delivery = False
                    self._lift_s = self._place_s = self._home_s = 0.0
                    events.append("grasp_confirmed")
            else:
                self._contact_s = 0.0
            grasp = bool(held and self._contact_s+1e-9 >= cfg.grasp_confirm_s and self._attempt_active)
            # Legal release is evaluated BEFORE generic loss-of-grasp rollback.
            release_height_ok = -cfg.support_height_tolerance_m <= bottom-m.goal_support_z_m <= cfg.release_max_clearance_m
            release_speed_ok = bool(m.target_motion_valid and np.linalg.norm(m.target_linear_velocity_m_s) <= cfg.release_linear_velocity_max_m_s and np.linalg.norm(m.target_angular_velocity_rad_s) <= cfg.release_angular_velocity_max_rad_s)
            # The last welded contact frame can be empty because DART filters
            # connected bodies. It cannot certify release after a same-boundary
            # detach. Wait for genuinely later sensor evidence, retaining only
            # attempt provenance until that evidence arrives (not grasp_valid).
            release_feedback_pending = bool(m.grasp_attachment_enabled and m.grasp_attachment_ever_used
                and not m.grasp_attachment_active and m.grasp_attachment_attempt_id == self._attempt
                and m.grasp_attachment_sim_time_s is not None
                and m.evidence_time <= m.grasp_attachment_sim_time_s + 1e-9)
            legal_release = bool(not m.grasp_attachment_active and not release_feedback_pending
                and self._attempt_active and self._attempt_lifted and self._carried and not dual
                and inside and release_height_ok and release_speed_ok)
            if legal_release:
                self._attempt_active = False
                self._release = self._delivery = True
                events.append("valid_release")
            elif self._attempt_active and not held and not release_feedback_pending:
                if self._attempt_lifted:
                    self._recovery_xy = np.asarray(m.target_position_world_m[:2]).copy()
                self._attempt_active = self._attempt_lifted = self._carried = self._delivery = self._release = False
                self._lift_s = self._place_s = self._home_s = 0.0
                self._contact_s = 0.0
                self._relative_anchor = None
                events.append("grasp_lost")
            if not held and not release_feedback_pending:
                self._relative_anchor = None
            # Absolute height alone cannot certify regrasping an already raised
            # or already pushed-in object: this attempt must show actual lift.
            if grasp and bottom-m.source_support_z_m+1e-9 >= cfg.lift_clearance_m and bottom-self._grasp_origin_bottom+1e-9 >= cfg.lift_clearance_m:
                self._lift_s = min(self._lift_s+dt, cfg.lift_confirm_s)
                if self._lift_s+1e-9 >= cfg.lift_confirm_s:
                    if not self._attempt_lifted:
                        events.append("lift_confirmed")
                    self._attempt_lifted = self._lifted_once = True
            elif not self._attempt_lifted:
                self._lift_s = 0.0
            if grasp and self._attempt_lifted and inside and bottom-m.goal_support_z_m >= cfg.transport_clearance_m:
                if not self._carried:
                    events.append("carried_into_goal")
                self._carried = self._arrived = True
                self._goal_recovery_token = True
            if self._delivery and (not inside or dual or not release_height_ok):
                self._delivery = self._release = False
                self._place_s = self._home_s = 0.0
                events.append("placement_invalidated")
        separated = bool(has_target and distance >= cfg.release_separation_m)
        released_contacts = (not dual) if cfg.grasp_evidence_mode == "proxy" else not (m.left_contact or m.right_contact)
        place_now = bool(not m.grasp_attachment_active and self._delivery and inside and supported and target_static and separated and released_contacts)
        if self.mode == "idle_completed":
            place_now = bool(inside and supported and target_static and not dual)
        if fresh:
            self._place_s = min(self._place_s+dt, cfg.place_confirm_s) if place_now else 0.0
        placed = bool(place_now and self._place_s+1e-9 >= cfg.place_confirm_s)
        if fresh:
            hold_allowed = ((not has_target or placed) if self.mode == "pick_place" else (not has_target or place_now))
            self._home_s = min(self._home_s+dt, self.required_hold) if hold_allowed and at_home and static else 0.0
        phase = TaskPhase.WAIT_HOME
        if self.mode == "pick_place" and has_target:
            if self._delivery:
                phase = TaskPhase.HOME_HOLD if placed and at_home else (TaskPhase.RETURN_HOME if placed else TaskPhase.PLACE_RELEASE)
            elif self._carried and inside:
                phase = TaskPhase.PLACE_RELEASE
            elif self._attempt_lifted and grasp:
                phase = TaskPhase.TRANSPORT
            elif self._attempt_active or distance <= cfg.grasp_candidate_distance_m:
                phase = TaskPhase.GRASP_LIFT
            else:
                phase = TaskPhase.APPROACH
        failure = m.failure_reason
        if has_target and fresh and bottom < m.source_support_z_m-cfg.drop_below_support_m:
            failure = failure or "object_below_workspace"
        if self.mode == "idle_completed" and fresh and (not inside or not supported or not target_static or dual):
            failure = failure or "completed_object_disturbed"
        success_event = False
        if self._outcome == "running":
            if failure:
                self._outcome, self._failure = "task_failure", str(failure)
                events.append("task_failure")
            elif allow_success and self.mode == "pick_place" and placed and self._home_s+1e-9 >= self.required_hold:
                self._outcome, success_event = "success", True
                events.append("success")
        orientation_score, orientation_error = 0.0, math.pi
        if has_target:
            # Bearing alone becomes perfect for every tool orientation when
            # target and grasp point coincide. Keep an independent world-axis
            # requirement so sideways/upside-down tools cannot pass by merely
            # reaching the position. This candidate FK-reference orientation
            # does not assert fingertip geometry or certify actual grasping.
            tool_axis = np.asarray(m.grasp_rotation_world)[:, 2]
            tool_axis = tool_axis / np.linalg.norm(tool_axis)
            direction = np.asarray(m.target_position_world_m)-np.asarray(m.grasp_position_world_m)+cfg.approach_reference_offset_m*tool_axis
            length = float(np.linalg.norm(direction))
            bearing_error = (math.pi if length < 1e-9 else
                             math.acos(float(np.clip(np.dot(tool_axis, direction/length), -1, 1))))
            axis_error = math.acos(float(np.clip(np.dot(tool_axis, cfg.desired_grasp_axis_world), -1, 1)))
            orientation_error = max(bearing_error, axis_error)
            orientation_score = math.exp(-orientation_error/cfg.approach_orientation_scale_rad)
        opening = float(m.joint_positions_rad[-1]-cfg.home_joint_positions_rad[-1])
        aperture_error = max(cfg.approach_opening_min_rad-opening, opening-cfg.approach_opening_max_rad, 0.0)
        if cfg.reward_schema_version == 3:
            # Preserve the exact reward used by existing runs and paired
            # evaluations. Never silently reinterpret a resolved v3 config.
            opening_score = bounded_score(aperture_error, cfg.approach_opening_scale_rad) if has_target else 0.0
        else:
            # A very wide jaw must still improve when it narrows toward the
            # opening interval; the v3 linear cutoff was flat above 1.7 rad.
            opening_score = math.exp(-aperture_error/cfg.approach_opening_scale_rad) if has_target else 0.0
        distance_score = (0.35 * math.exp(-distance/cfg.approach_distance_scale_m)
                          + 0.65 * math.exp(-distance/cfg.approach_fine_distance_scale_m)) if has_target else 0.0
        geometry_valid = bool(has_target and m.evidence_valid and m.joint_measurement_valid
                              and m.target_measurement_valid and m.target_motion_valid)
        close_gate = 0.0
        if geometry_valid:
            close_gate = float(np.clip((cfg.pregrasp_distance_tolerance_m-distance)
                                       / (cfg.pregrasp_distance_tolerance_m-cfg.grasp_close_full_distance_m), 0, 1))
            close_gate *= bounded_score(orientation_error, cfg.pregrasp_orientation_tolerance_rad)
        if cfg.reward_schema_version == 3:
            closure_score = (bounded_score(max(opening, 0.0), cfg.approach_opening_max_rad)
                             if geometry_valid else 0.0)
        else:
            # At close_gate=1 the opening term vanishes. This score must retain
            # signal above the opening interval too, otherwise a well-aligned
            # but wide-open gripper receives no incentive to begin closing.
            closure_score = (math.exp(-max(opening, 0.0)/cfg.approach_opening_max_rad)
                             if geometry_valid else 0.0)
        # The coarse scale guides reaching; the fine scale makes millimetre
        # improvements visible. Smoothly hand opening guidance to closure once
        # aligned. The maximum raw value is 1.8 (centered, aligned, closed).
        approach = (distance_score * (1 + 0.4 * orientation_score
                    + 0.2 * (1-close_gate) * opening_score)
                    + 0.4 * close_gate * closure_score) / 1.8 if has_target else 0.0
        pregrasp_candidate = bool(self.mode == "pick_place" and geometry_valid
                                 and not self._attempt_active and not self._lifted_once
                                 and not inside and distance <= cfg.pregrasp_distance_tolerance_m
                                 and orientation_error <= cfg.pregrasp_orientation_tolerance_rad
                                 and cfg.approach_opening_min_rad <= opening <= cfg.approach_opening_max_rad)
        if fresh:
            self._pregrasp_s = min(self._pregrasp_s+dt, cfg.pregrasp_confirm_s) if pregrasp_candidate else 0.0
        pregrasp_ready = bool(pregrasp_candidate and self._pregrasp_s+1e-9 >= cfg.pregrasp_confirm_s)
        lift_score = float(np.clip((bottom-m.source_support_z_m)/cfg.lift_clearance_m, 0, 1)) if grasp else 0.0
        transport = math.exp(-goal_distance/cfg.transport_distance_scale_m)*float(np.clip((bottom-m.goal_support_z_m)/cfg.transport_clearance_m, 0, 1)) if self._attempt_lifted and grasp else 0.0
        return TaskSnapshot(
            measurements=m, task_instance_id=self.instance, task_mode=self.mode, phase=phase,
            outcome=self._outcome, failure_reason=self._failure, success_event=success_event, events=tuple(events),
            grasp_attempt_id=self._attempt, grasp_attempt_active=self._attempt_active, grasp_valid=grasp, lifted_once=self._lifted_once,
            attempt_lifted=self._attempt_lifted, carried_into_goal=self._carried,
            valid_release=self._release, delivery_valid=self._delivery, arrived_via_valid_carry=self._arrived,
            local_recovery_authorized=recovery_allowed, placed_stable=placed, target_in_goal=inside,
            goal_recovery_authorized=self._goal_recovery_token,
            target_supported=supported, target_static=target_static, robot_at_home=at_home, robot_static=static,
            pregrasp_ready=pregrasp_ready, pregrasp_ready_duration_s=self._pregrasp_s,
            contact_duration_s=self._contact_s, lift_duration_s=self._lift_s,
            place_stable_duration_s=self._place_s, home_stable_duration_s=self._home_s,
            home_hold_required_s=self.required_hold, home_error_rad=float(np.max(qerr)),
            max_joint_velocity_rad_s=float(np.max(qvel)), target_distance_m=distance, target_bottom_z_m=bottom,
            approach_score=approach, contact_score=float(np.clip(self._contact_s/cfg.grasp_confirm_s, 0, 1)),
            approach_distance_score=distance_score, approach_orientation_score=orientation_score,
            approach_opening_score=opening_score, grasp_close_gate=close_gate, grasp_closure_score=closure_score,
            lift_score=lift_score, transport_score=transport,
            support_score=math.exp(-abs(bottom-m.goal_support_z_m)/cfg.support_score_scale_m) if has_target else 0,
            separation_score=float(np.clip(distance/cfg.release_separation_m, 0, 1)) if has_target else 0,
            home_score=home_score,
            source_position_xy_m=None if self._source_xy is None else tuple(self._source_xy),
            recovery_position_xy_m=None if self._recovery_xy is None else tuple(self._recovery_xy),
            grasp_relative_anchor_m=None if self._relative_anchor is None else tuple(self._relative_anchor),
            grasp_origin_bottom_z_m=self._grasp_origin_bottom,
        )
