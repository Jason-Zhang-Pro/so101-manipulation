"""Build physical task evidence on simulation time, without ROS or rewards.

Joint poses are interpolated to a measured object/contact time. Object motion
uses distinct source samples; neither timer republications nor wall time are
physical measurements. Unobserved intervals never become stability evidence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from so101_capstone.robot import compute_grasp_point_pose
from .contact_state import ContactEvidence, FingerContactCache
from .state_cache import MEASURED_JOINT_NAMES, JointStateSnapshot, TargetCubeSnapshot


class TaskMeasurementError(RuntimeError):
    """Physical evidence cannot be formed; do not train on invented values."""


@dataclass(frozen=True)
class TaskMeasurementConfig:
    max_evidence_age_s: float = 0.12
    max_sample_gap_s: float = 0.12
    max_unknown_s: float = 0.30
    goal_half_extents_xy_m: tuple[float, float] = (0.05, 0.05)
    goal_yaw_rad: float = 0.0
    goal_support_offset_m: float = 0.0015
    source_support_z_m: float = 0.750
    target_half_extents_m: tuple[float, float, float] = (0.015, 0.015, 0.015)
    grasp_evidence_mode: str = "contact"

    def __post_init__(self) -> None:
        for name in ("max_evidence_age_s", "max_sample_gap_s", "max_unknown_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name, size in (("goal_half_extents_xy_m", 2), ("target_half_extents_m", 3)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f"{name} must contain {size} positive finite values.")
        for name in ("goal_yaw_rad", "goal_support_offset_m", "source_support_z_m"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")
        if self.grasp_evidence_mode not in {"contact", "proxy"}:
            raise ValueError("grasp_evidence_mode must be contact or proxy.")


def _joint_vectors(snapshot: JointStateSnapshot) -> tuple[np.ndarray, np.ndarray]:
    if snapshot.velocities is None:
        raise TaskMeasurementError("Measured joint velocities are required.")
    positions = np.asarray([snapshot.positions[name] for name in MEASURED_JOINT_NAMES])
    velocities = np.asarray([snapshot.velocities[name] for name in MEASURED_JOINT_NAMES])
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise TaskMeasurementError("Joint feedback must be finite.")
    return positions, velocities


def interpolate_joints(history: Sequence[JointStateSnapshot], time_s: float,
                       max_gap_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate measured bounded revolute positions; never wrap or extrapolate."""
    by_time = {item.source_sim_time_s: item for item in history
               if item.source_sim_time_s is not None}
    before = after = None
    for stamp, sample in sorted(by_time.items()):
        if abs(stamp - time_s) <= 1e-8:
            return _joint_vectors(sample)
        if stamp < time_s:
            before = sample
        else:
            after = sample
            break
    if before is None or after is None:
        return None
    gap = after.source_sim_time_s - before.source_sim_time_s
    if gap > max_gap_s:
        return None
    fraction = (time_s - before.source_sim_time_s) / gap
    p0, v0 = _joint_vectors(before)
    p1, v1 = _joint_vectors(after)
    return p0 + fraction * (p1 - p0), v0 + fraction * (v1 - v0)


def quaternion_angular_velocity(previous: Sequence[float], current: Sequence[float],
                                dt_s: float) -> np.ndarray:
    """Shortest relative rotation in world axes, treating q and -q identically."""
    q0, q1 = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    if q0.shape != (4,) or q1.shape != (4,) or not np.all(np.isfinite([q0, q1])):
        raise ValueError("Quaternions must be finite xyzw 4-vectors.")
    if not math.isfinite(dt_s) or dt_s <= 0 or min(np.linalg.norm(q0), np.linalg.norm(q1)) <= 1e-12:
        raise ValueError("Quaternion motion needs nonzero quaternions and positive dt.")
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    vector = q0[3] * q1[:3] - q1[3] * q0[:3] - np.cross(q1[:3], q0[:3])
    scalar = q1[3] * q0[3] + np.dot(q1[:3], q0[:3])
    length = np.linalg.norm(vector)
    if length < 1e-12:
        return np.zeros(3)
    return vector * (2.0 * math.atan2(length, max(0.0, scalar)) / (length * dt_s))


class TaskMeasurementBuilder:
    def __init__(self, config: TaskMeasurementConfig | None = None) -> None:
        self.config = config or TaskMeasurementConfig()
        self.reset()

    def reset(self, *, target_history: Sequence[TargetCubeSnapshot] = ()) -> None:
        """Clear task history, optionally seed measured motion at a stage handoff.

        Only a recent, continuous same-object preparation history is allowed.
        This restores no grasp events, rewards, or controller transitions.
        Ordinary episode resets deliberately provide no prehistory.
        """
        self._target_name: str | None = None
        self._target_first_time: float | None = None
        self._poses: deque[TargetCubeSnapshot] = deque(maxlen=256)
        self._last_control_time: float | None = None
        self._last_evidence_time: float | None = None
        self._unknown_since: float | None = None
        self.last_info: dict[str, object] = {}
        if target_history:
            name = target_history[0].name
            for item in target_history:
                stamp = item.source_sim_time_s
                if item.name != name or stamp is None or not math.isfinite(stamp):
                    raise TaskMeasurementError("Stage motion prehistory needs one target and finite source times.")
                if self._poses:
                    gap = stamp - self._poses[-1].source_sim_time_s
                    if gap <= 0 or gap > self.config.max_sample_gap_s + 1e-9:
                        raise TaskMeasurementError("Stage motion prehistory must be ordered and continuous.")
                self._poses.append(item)
            self._target_name = name
            self._target_first_time = self._poses[0].source_sim_time_s

    def update(self, joint_state: JointStateSnapshot, cube_state: TargetCubeSnapshot | None,
               *, control_sim_time_s: float, joint_history: Sequence[JointStateSnapshot] = (),
               cube_history: Sequence[TargetCubeSnapshot] = (),
               contact_cache: FingerContactCache | None = None,
               goal_center_world_m: Sequence[float],
               camera_source_times_s: tuple[float | None, float | None] = (None, None),
               command_joint_targets_rad: Sequence[float] | None = None):
        from so101_capstone.rewards.task_state import TaskMeasurements

        now = float(control_sim_time_s)
        if not math.isfinite(now) or now < 0:
            raise TaskMeasurementError("Control simulation time must be finite and non-negative.")
        if self._last_control_time is not None and now < self._last_control_time - 1e-9:
            raise TaskMeasurementError("Simulation time moved backwards within an episode.")
        self._last_control_time = now
        if joint_state.source_sim_time_s is None:
            raise TaskMeasurementError("pick_place_v2 requires JointState.header simulation time.")
        if joint_state.source_sim_time_s > now + 1e-6:
            raise TaskMeasurementError("Joint feedback is newer than the completed step.")
        q, qdot = _joint_vectors(joint_state)
        name = None if cube_state is None else cube_state.name
        if name != self._target_name:
            self._target_name = name
            self._poses.clear()
            self._target_first_time = None if cube_state is None else cube_state.source_sim_time_s
        if cube_state is not None:
            if cube_state.source_sim_time_s is None:
                raise TaskMeasurementError("pick_place_v2 requires target pose source time.")
            if cube_state.source_sim_time_s > now + 1e-6:
                raise TaskMeasurementError("Target feedback is newer than the completed step.")
            samples = (*cube_history, cube_state)
            unique = {item.source_sim_time_s: item for item in samples
                      if item.name == name and item.source_sim_time_s is not None
                      and item.source_sim_time_s >= self._target_first_time}
            for stamp, item in sorted(unique.items()):
                if not self._poses or stamp > self._poses[-1].source_sim_time_s + 1e-9:
                    self._poses.append(item)

        evidence_time = joint_state.source_sim_time_s
        selected = cube_state
        contact_latest = None
        if contact_cache is not None:
            history = contact_cache.history()
            if history["left"] and history["right"]:
                contact_latest = min(history["left"][-1].sim_time_s, history["right"][-1].sim_time_s)
                evidence_time = min(evidence_time, contact_latest)
        if selected is not None:
            possible = [item for item in self._poses if item.source_sim_time_s <= evidence_time + 1e-8]
            selected = possible[-1] if possible else cube_state
            evidence_time = min(evidence_time, selected.source_sim_time_s)

        # A newly appearing object's delayed source pose may precede the last
        # idle evidence. Keep the evidence cursor monotonic, explicitly mark
        # the mismatched target unknown, and wait for a later real sample.
        if self._last_evidence_time is not None:
            evidence_time = max(evidence_time, self._last_evidence_time)

        aligned = interpolate_joints((*joint_history, joint_state), evidence_time, self.config.max_sample_gap_s)
        joint_valid = aligned is not None
        if aligned is not None:
            q, qdot = aligned
        age = now - evidence_time
        target_valid = selected is None or abs(selected.source_sim_time_s - evidence_time) <= 1e-8
        evidence_valid = joint_valid and target_valid and -1e-6 <= age <= self.config.max_evidence_age_s

        contact = ContactEvidence(False, sim_time_s=evidence_time, reason="no_contact_stream")
        if contact_cache is not None and contact_latest is not None:
            start = self._last_evidence_time
            # A long missing interval is never filled in retrospectively.
            if start is None or evidence_time - start > self.config.max_sample_gap_s:
                start = evidence_time
            if evidence_time >= start:
                contact = contact_cache.evidence(target_name=name, end_time_s=evidence_time,
                                                start_time_s=start, max_gap_s=self.config.max_sample_gap_s)
        if self.config.grasp_evidence_mode == "contact":
            evidence_valid = evidence_valid and contact.valid
        self._last_evidence_time = evidence_time

        linear, angular, motion_valid = np.zeros(3), np.zeros(3), False
        if selected is not None:
            before = [item for item in self._poses if item.source_sim_time_s < selected.source_sim_time_s - 1e-9]
            if before:
                prior = before[-1]
                dt = selected.source_sim_time_s - prior.source_sim_time_s
                if dt <= self.config.max_sample_gap_s:
                    linear = (selected.position - prior.position) / dt
                    angular = quaternion_angular_velocity(prior.quaternion_xyzw, selected.quaternion_xyzw, dt)
                    motion_valid = evidence_valid

        # A recent pose alone is insufficient: persistently sparse poses can
        # never establish object speed, even if each arrival refreshes its age.
        # Give the first pose the same bounded startup grace as other unknown
        # evidence, then reject the rollout instead of training on a frozen task.
        task_evidence_available = evidence_valid and (selected is None or motion_valid)
        if not task_evidence_available:
            if self._unknown_since is None:
                self._unknown_since = now
            if now - self._unknown_since > self.config.max_unknown_s:
                raise TaskMeasurementError("Task evidence unavailable beyond allowed simulation-time gap.")
        else:
            self._unknown_since = None
        pose = compute_grasp_point_pose(dict(zip(MEASURED_JOINT_NAMES, q)))
        goal = np.asarray(goal_center_world_m, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise TaskMeasurementError("Goal center must be a finite world 3-vector.")
        self.last_info = {
            "physical_target_present": name is not None,
            "target_appeared_sim_time_s": self._target_first_time,
            "joint_source_sim_time_s": joint_state.source_sim_time_s,
            "target_source_sim_time_s": None if selected is None else selected.source_sim_time_s,
            "evidence_sim_time_s": evidence_time,
            "evidence_age_s": age,
            "evidence_valid": evidence_valid,
            "task_evidence_available": task_evidence_available,
            "camera_source_times_s": list(camera_source_times_s),
            "camera_observed_target": [name is not None and stamp is not None
                                       and stamp >= self._target_first_time
                                       for stamp in camera_source_times_s],
            "contact_evidence_reason": contact.reason,
            "contact_overlap_duration_s": contact.simultaneous_duration_s,
        }
        return TaskMeasurements(
            sim_time_s=now, joint_positions_rad=tuple(q), joint_velocities_rad_s=tuple(qdot),
            grasp_position_world_m=tuple(pose.position), grasp_rotation_world=tuple(map(tuple, pose.transform[:3, :3])),
            target_name=name, target_position_world_m=None if selected is None else tuple(selected.position),
            target_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0) if selected is None else tuple(selected.quaternion_xyzw),
            target_linear_velocity_m_s=tuple(linear), target_angular_velocity_rad_s=tuple(angular),
            target_motion_valid=motion_valid, target_measurement_valid=target_valid,
            joint_measurement_valid=joint_valid, evidence_valid=evidence_valid,
            evidence_sim_time_s=evidence_time, contact_valid=contact.valid,
            left_contact=contact.left_contact, right_contact=contact.right_contact,
            contacts_simultaneous=contact.contacts_simultaneous,
            goal_center_world_m=tuple(goal), goal_half_extents_xy_m=self.config.goal_half_extents_xy_m,
            goal_yaw_rad=self.config.goal_yaw_rad, goal_support_z_m=goal[2] + self.config.goal_support_offset_m,
            target_half_extents_m=self.config.target_half_extents_m,
            source_support_z_m=self.config.source_support_z_m,
            grasp_evidence_mode=self.config.grasp_evidence_mode,
            executed_command_joint_targets_rad=command_joint_targets_rad,
        )
