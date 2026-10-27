"""Versioned pick/place objective, shaping, and measured process rewards.

All numeric defaults are calibration candidates, not tuned robot parameters.
The actor is never called here. True terminals have zero potential; externally
truncated samples retain their physical potential and require value bootstrap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .task_state import TaskPhase, TaskSnapshot, _vector


# These fields exist only in resolved schema-5 configurations. Older serialized
# configurations keep exactly their original keys and hashes.
_SCHEMA5_DEFAULTS = {
    "reach_distance_progress_weight": 100.0,
    "reach_stagnation_distance_weight": 100.0,
    "reach_step_cost": 0.5,
    "reach_alignment_step_deg": 0.5,
    "reach_alignment_progress_bonus": 0.1,
    "reach_alignment_regression_cost": 0.5,
    "reach_z_perfect_threshold_deg": 3.0,
    "reach_z_perfect_bonus": 5.0,
    "reach_close_alignment_distance_m": 0.06,
    "reach_close_alignment_threshold_deg": 10.0,
    "reach_close_alignment_cost": 5.0,
    "reach_distance_thresholds_m": (0.22, 0.19, 0.15, 0.12, 0.09, 0.06, 0.03, 0.02),
    "reach_distance_threshold_bonuses": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    "reach_multi_threshold_multiplier": 1.5,
    "reach_gripper_progress_weight": 5.0,
    "reach_gripper_regression_cost": 5.0,
    "home_progress_weight": 10.0,
    "home_stagnation_cost": 0.5,
    "home_regression_cost": 2.0,
    "home_unsettled_step_cost": 0.5,
}

REWARD_HISTORY_CONTEXT_SCHEMA_VERSION = 1
# reach initialized; six best/latest geometry values; perfect flag; eight
# threshold flags; seven opening/closure values; two four-value HOME histories;
# and the two schema-4 dense-cost cache values. All entries are finite scalars.
REWARD_HISTORY_CONTEXT_DIM = 33


@dataclass(frozen=True)
class PickPlaceRewardConfig:
    reward_schema_version: int = 4
    success_definition_version: int = 2
    calibration_status: str = "candidate_unvalidated"
    gamma: float = 0.997
    control_hz: float = 30.0
    grasp_evidence_mode: str = "contact"
    success_bonus: float = 100.0
    failure_cost: float = 10.0
    potential_max: float = 10.0
    shaping_enabled: bool = True
    active_time_cost_per_s: float = 0.3
    # Schema 4 adds a genuine dense objective, not potential-based shaping.
    # None resolves to 0.3/s in v4; v3 must leave it None and omits the field
    # from serialization so historical resolved-config hashes remain identical.
    remaining_task_cost_per_s: float | None = None
    home_position_cost_per_step: float = 0.05
    home_velocity_cost_per_step: float = 0.02
    moving_velocity_cost_per_step: float = 0.0
    command_change_cost_per_step: float = 0.0
    home_joint_positions_rad: tuple[float, ...] = (0.0, -1.710422666954443, 1.605702911834783, 1.134464013796314, 0.0, -0.087266462599716)
    home_joint_tolerance_rad: float = math.radians(1.0)
    home_velocity_tolerance_rad_s: float = 0.03
    home_position_scales_rad: tuple[float, ...] = (0.5,) * 6
    velocity_scales_rad_s: tuple[float, ...] = (1.0,) * 6
    command_change_scales_rad: tuple[float, ...] = (0.1,) * 6
    home_hold_min_s: float = 3.0
    home_hold_max_s: float = 5.0
    home_reset_jitter_rad: float = math.radians(1.0)
    grasp_confirm_s: float = 0.1
    lift_confirm_s: float = 0.1
    place_confirm_s: float = 0.4
    max_evidence_gap_s: float = 0.12
    grasp_candidate_distance_m: float = 0.045
    relative_grasp_tolerance_m: float = 0.015
    source_adjustment_tolerance_m: float = 0.035
    goal_recovery_radius_m: float = 0.10
    lift_clearance_m: float = 0.020
    transport_clearance_m: float = 0.020
    release_max_clearance_m: float = 0.012
    release_linear_velocity_max_m_s: float = 0.05
    release_angular_velocity_max_rad_s: float = 0.40
    release_separation_m: float = 0.060
    support_height_tolerance_m: float = 0.006
    object_linear_velocity_tolerance_m_s: float = 0.015
    object_angular_velocity_tolerance_rad_s: float = 0.15
    drop_below_support_m: float = 0.080
    goal_half_extents_xy_m: tuple[float, ...] = (0.05, 0.05)
    goal_yaw_rad: float = 0.0
    goal_support_offset_m: float = 0.0015
    target_half_extents_m: tuple[float, ...] = (0.015, 0.015, 0.015)
    source_support_z_m: float = 0.750
    goal_margin_m: float = 0.002
    approach_distance_scale_m: float = 0.3
    approach_fine_distance_scale_m: float = 0.025
    approach_orientation_scale_rad: float = math.pi / 2
    # Candidate tabletop direction for the FK gripper-reference local +Z.
    # This is not a measured fingertip pose or grasp certification.
    desired_grasp_axis_world: tuple[float, ...] = (0.0, 0.0, -1.0)
    approach_reference_offset_m: float = 0.04
    approach_opening_min_rad: float = 0.45
    approach_opening_max_rad: float = 0.90
    approach_opening_scale_rad: float = 0.80
    pregrasp_distance_tolerance_m: float = 0.020
    pregrasp_orientation_tolerance_rad: float = math.radians(20.0)
    pregrasp_confirm_s: float = 0.15
    grasp_close_full_distance_m: float = 0.008
    transport_distance_scale_m: float = 0.3
    support_score_scale_m: float = 0.05

    reach_distance_progress_weight: float | None = None
    reach_stagnation_distance_weight: float | None = None
    reach_step_cost: float | None = None
    reach_alignment_step_deg: float | None = None
    reach_alignment_progress_bonus: float | None = None
    reach_alignment_regression_cost: float | None = None
    reach_z_perfect_threshold_deg: float | None = None
    reach_z_perfect_bonus: float | None = None
    reach_close_alignment_distance_m: float | None = None
    reach_close_alignment_threshold_deg: float | None = None
    reach_close_alignment_cost: float | None = None
    reach_distance_thresholds_m: tuple[float, ...] | None = None
    reach_distance_threshold_bonuses: tuple[float, ...] | None = None
    reach_multi_threshold_multiplier: float | None = None
    reach_gripper_progress_weight: float | None = None
    reach_gripper_regression_cost: float | None = None
    home_progress_weight: float | None = None
    home_stagnation_cost: float | None = None
    home_regression_cost: float | None = None
    home_unsettled_step_cost: float | None = None

    def __post_init__(self) -> None:
        vector_sizes = {"home_joint_positions_rad": 6, "home_position_scales_rad": 6,
                        "velocity_scales_rad_s": 6, "command_change_scales_rad": 6,
                        "goal_half_extents_xy_m": 2, "target_half_extents_m": 3,
                        "desired_grasp_axis_world": 3}
        for name, size in vector_sizes.items():
            value = _vector(getattr(self, name), size, name)
            if name not in ("home_joint_positions_rad", "desired_grasp_axis_world") and min(value) <= 0:
                raise ValueError(f"{name} values must be positive")
            if name == "desired_grasp_axis_world":
                norm = math.hypot(*value)
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("desired_grasp_axis_world must be a nonzero finite direction")
                value = tuple(component/norm for component in value)
            object.__setattr__(self, name, value)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(field.default, (int, float)) and not isinstance(field.default, bool):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{field.name} must be a finite number")
        if not isinstance(self.calibration_status, str) or not self.calibration_status.strip():
            raise ValueError("calibration_status must be a nonempty string")
        if self.reward_schema_version not in (3, 4, 5) or self.success_definition_version != 2:
            raise ValueError("unsupported reward or success definition version")
        for name, default in _SCHEMA5_DEFAULTS.items():
            value = getattr(self, name)
            if self.reward_schema_version < 5:
                if value is not None:
                    raise ValueError(f"{name} requires reward schema 5")
                continue
            value = default if value is None else value
            if isinstance(default, tuple):
                value = _vector(value, len(default), name)
                if min(value) < 0:
                    raise ValueError(f"{name} must be non-negative")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.reward_schema_version == 5:
            if self.reach_alignment_step_deg <= 0:
                raise ValueError("reach_alignment_step_deg must be positive")
            thresholds = self.reach_distance_thresholds_m
            if min(thresholds) <= 0 or any(a <= b for a, b in zip(thresholds, thresholds[1:])):
                raise ValueError("reach_distance_thresholds_m must be positive and strictly decreasing")
            if self.reach_multi_threshold_multiplier < 1:
                raise ValueError("reach_multi_threshold_multiplier must be at least one")
            if self.reach_z_perfect_threshold_deg > self.reach_close_alignment_threshold_deg or self.reach_close_alignment_threshold_deg > 180:
                raise ValueError("reach angular thresholds must be ordered within [0, 180]")
        if self.reward_schema_version == 3:
            if self.remaining_task_cost_per_s is not None:
                raise ValueError("schema 3 does not support remaining_task_cost_per_s; leave it None")
        else:
            cost = .3 if self.remaining_task_cost_per_s is None else self.remaining_task_cost_per_s
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                raise ValueError("remaining_task_cost_per_s must be finite and non-negative")
            object.__setattr__(self, "remaining_task_cost_per_s", float(cost))
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if self.grasp_evidence_mode not in ("contact", "proxy"):
            raise ValueError("grasp_evidence_mode must be contact or proxy")
        if not isinstance(self.shaping_enabled, bool):
            raise ValueError("shaping_enabled must be boolean")
        positive = ("control_hz", "home_joint_tolerance_rad", "home_velocity_tolerance_rad_s",
                    "home_hold_min_s", "home_hold_max_s", "grasp_confirm_s", "lift_confirm_s",
                    "place_confirm_s", "max_evidence_gap_s", "grasp_candidate_distance_m",
                    "relative_grasp_tolerance_m", "source_adjustment_tolerance_m", "goal_recovery_radius_m",
                    "lift_clearance_m", "transport_clearance_m", "release_max_clearance_m",
                    "release_linear_velocity_max_m_s", "release_angular_velocity_max_rad_s",
                    "release_separation_m", "support_height_tolerance_m", "object_linear_velocity_tolerance_m_s",
                    "object_angular_velocity_tolerance_rad_s", "drop_below_support_m", "approach_distance_scale_m",
                    "approach_orientation_scale_rad", "approach_opening_scale_rad", "transport_distance_scale_m",
                    "support_score_scale_m", "approach_fine_distance_scale_m",
                    "pregrasp_distance_tolerance_m", "pregrasp_orientation_tolerance_rad",
                    "pregrasp_confirm_s", "grasp_close_full_distance_m")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        nonnegative = ("success_bonus", "failure_cost", "potential_max", "active_time_cost_per_s",
                       "home_position_cost_per_step", "home_velocity_cost_per_step", "moving_velocity_cost_per_step",
                       "command_change_cost_per_step", "goal_margin_m", "approach_reference_offset_m")
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.home_reset_jitter_rad < 0:
            raise ValueError("home_reset_jitter_rad must be nonnegative")
        if self.home_hold_min_s < 3 or self.home_hold_max_s < self.home_hold_min_s:
            raise ValueError("HOME hold must satisfy 3 <= min <= max seconds")
        if not 0 < self.approach_opening_min_rad <= self.approach_opening_max_rad:
            raise ValueError("opening interval must be positive and ordered")
        if not self.grasp_close_full_distance_m < self.pregrasp_distance_tolerance_m <= self.grasp_candidate_distance_m:
            raise ValueError("close distances must satisfy full < pregrasp <= grasp candidate")
        if self.approach_fine_distance_scale_m >= self.approach_distance_scale_m:
            raise ValueError("fine approach distance scale must be below the coarse scale")
        if self.pregrasp_orientation_tolerance_rad > math.pi:
            raise ValueError("pregrasp orientation tolerance must not exceed pi")
        if self.release_max_clearance_m >= self.transport_clearance_m:
            raise ValueError("release clearance must be below transport clearance")
        if self.support_height_tolerance_m >= self.lift_clearance_m:
            raise ValueError("support tolerance must be below lift clearance")
        if self.grasp_candidate_distance_m >= self.release_separation_m:
            raise ValueError("release separation must exceed grasp candidate distance")
        if min(self.goal_half_extents_xy_m) <= max(self.target_half_extents_m[:2]) + self.goal_margin_m:
            raise ValueError("goal must accommodate the entire target footprint")

    @classmethod
    def from_dict(cls, values: Mapping[str, object], *, gamma: float | None = None) -> PickPlaceRewardConfig:
        values = dict(values)
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown pick/place reward fields: {sorted(unknown)}")
        if gamma is not None:
            if "gamma" in values and not math.isclose(float(values["gamma"]), float(gamma), rel_tol=0, abs_tol=1e-12):
                raise ValueError("reward gamma must equal PPO gamma")
            values["gamma"] = gamma
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path | None = None, *, gamma: float | None = None) -> PickPlaceRewardConfig:
        values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("reward configuration must be a JSON object")
        return cls.from_dict(values, gamma=gamma)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.reward_schema_version == 3:
            result.pop("remaining_task_cost_per_s")
        if self.reward_schema_version < 5:
            for name in _SCHEMA5_DEFAULTS:
                result.pop(name)
        return json.loads(json.dumps(result, allow_nan=False))

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def resolve_home_hold_bounds(self, n_action_steps: int) -> tuple[float, float]:
        if isinstance(n_action_steps, bool) or int(n_action_steps) != n_action_steps or n_action_steps <= 0:
            raise ValueError("n_action_steps must be a positive integer")
        minimum = max(self.home_hold_min_s, 3.0, 2 * n_action_steps / self.control_hz)
        width = max(1 / self.control_hz, self.home_hold_max_s-self.home_hold_min_s)
        return minimum, max(minimum + width, self.home_hold_max_s)


@dataclass(frozen=True)
class PickPlaceRewardInput:
    prev_snapshot: TaskSnapshot
    next_snapshot: TaskSnapshot
    dt_s: float
    terminated: bool = False
    truncated: bool = False
    command_joint_targets_rad: Sequence[float] | None = None
    previous_command_joint_targets_rad: Sequence[float] | None = None
    termination_reason: str | None = None


@dataclass(frozen=True)
class PickPlaceRewardResult:
    reward: float
    components: dict[str, float]
    potential_before: float
    potential_after: float
    terminated: bool
    truncated: bool

    def to_info(self) -> dict[str, object]:
        return {"reward": self.reward, "reward_components": dict(self.components),
                "potential_before": self.potential_before, "potential_after": self.potential_after,
                "terminated": self.terminated, "truncated": self.truncated}


class PickPlaceReward:
    def __init__(self, config: PickPlaceRewardConfig):
        self.config = config
        self._previous: TaskSnapshot | None = None
        self._closed = False
        self._dense_target_seen = False
        self._dense_last_valid_progress = 0.0
        self._reset_process_history()

    def _reset_process_history(self) -> None:
        self._reach_initialized = False
        self._reach_best_distance = self._reach_last_distance = self.config.approach_distance_scale_m
        self._reach_best_x = self._reach_last_x = 0.0
        self._reach_best_z = self._reach_last_z = 0.0
        self._reach_perfect_claimed = False
        self._reach_threshold_claimed = [False] * 8
        self._reach_open_initialized = self._reach_close_initialized = False
        self._reach_best_open = self._reach_best_close = 0.0
        self._reach_last_open = self._reach_last_close = self._reach_last_gate = 0.0
        # Waiting/idle HOME and post-delivery HOME have separate records. Once
        # created, neither record resets on a phase regression or re-entry.
        self._home_initialized = [False, False]
        self._home_best = [0.0, 0.0]
        self._home_last = [1.0, 1.0]
        self._home_last_unsettled = [True, True]

    def critic_history_context(self) -> np.ndarray:
        """Expose all reward memory used by schema 5, after reset/step.

        Schema 3/4 callers receive an empty vector so their critic contract is
        unchanged. Schema 5 appends this vector to the measured task context.
        """
        if self.config.reward_schema_version < 5:
            return np.empty(0, dtype=np.float32)
        values = [float(self._reach_initialized), self._reach_best_distance,
                  self._reach_last_distance, self._reach_best_x, self._reach_last_x,
                  self._reach_best_z, self._reach_last_z, float(self._reach_perfect_claimed)]
        values += [float(value) for value in self._reach_threshold_claimed]
        values += [float(self._reach_open_initialized), self._reach_best_open,
                   float(self._reach_close_initialized), self._reach_best_close,
                   self._reach_last_open, self._reach_last_close, self._reach_last_gate]
        for index in range(2):
            values += [float(self._home_initialized[index]), self._home_best[index],
                       self._home_last[index], float(self._home_last_unsettled[index])]
        values += [float(self._dense_target_seen), self._dense_last_valid_progress]
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (REWARD_HISTORY_CONTEXT_DIM,) or not np.isfinite(result).all():
            raise ValueError("invalid reward history context")
        return result

    def _valid_process_measurement(self, snapshot: TaskSnapshot) -> bool:
        m = snapshot.measurements
        valid = (m.evidence_valid and m.joint_measurement_valid
                 and 0 <= m.sim_time_s-m.evidence_time <= self.config.max_evidence_gap_s)
        if snapshot.target_name is not None:
            valid = (valid and m.target_measurement_valid and m.target_motion_valid
                     and (m.contact_valid or self.config.grasp_evidence_mode == "proxy"))
        return bool(valid)

    @staticmethod
    def _reach_active(snapshot: TaskSnapshot) -> bool:
        return bool(snapshot.task_mode == "pick_place" and snapshot.target_name is not None
                    and snapshot.phase in (TaskPhase.APPROACH, TaskPhase.GRASP_LIFT)
                    and not snapshot.grasp_attempt_active and not snapshot.grasp_valid)

    @staticmethod
    def _home_history_index(snapshot: TaskSnapshot) -> int | None:
        if snapshot.task_mode != "pick_place" or snapshot.phase is TaskPhase.WAIT_HOME:
            return 0
        if snapshot.phase in (TaskPhase.RETURN_HOME, TaskPhase.HOME_HOLD):
            return 1
        return None

    def _reach_geometry(self, snapshot: TaskSnapshot) -> tuple[float, float, float]:
        """Keep the current FK reference point and 4 cm orientation geometry.

        The historic project's 20 cm ray origin and gripper offset are not
        transplanted. Z guidance also retains the current downward-axis check.
        """
        m = snapshot.measurements
        rotation = np.asarray(m.grasp_rotation_world)
        x_axis = rotation[:, 0] / np.linalg.norm(rotation[:, 0])
        x_error = math.degrees(math.asin(float(np.clip(abs(x_axis[2]), 0, 1))))
        axis = rotation[:, 2] / np.linalg.norm(rotation[:, 2])
        direction = (np.asarray(m.target_position_world_m)-m.grasp_position_world_m
                     + self.config.approach_reference_offset_m*axis)
        length = float(np.linalg.norm(direction))
        bearing = (math.pi if length < 1e-9 else
                   math.acos(float(np.clip(np.dot(axis, direction/length), -1, 1))))
        downward = math.acos(float(np.clip(np.dot(axis, self.config.desired_grasp_axis_world), -1, 1)))
        return snapshot.target_distance_m, x_error, math.degrees(max(bearing, downward))

    def _seed_reach_history(self, snapshot: TaskSnapshot) -> None:
        distance, x_error, z_error = self._reach_geometry(snapshot)
        self._reach_initialized = True
        self._reach_best_distance = self._reach_last_distance = distance
        self._reach_best_x = self._reach_last_x = x_error
        self._reach_best_z = self._reach_last_z = z_error
        self._reach_perfect_claimed = z_error <= self.config.reach_z_perfect_threshold_deg
        self._reach_threshold_claimed = [distance < t for t in self.config.reach_distance_thresholds_m]
        self._cache_gripper_scores(snapshot)
        if self._reach_last_gate < 1:
            self._reach_open_initialized = True
            self._reach_best_open = self._reach_last_open
        if self._reach_last_gate > 0:
            self._reach_close_initialized = True
            self._reach_best_close = self._reach_last_close

    def _cache_gripper_scores(self, snapshot: TaskSnapshot) -> None:
        self._reach_last_open = snapshot.approach_opening_score
        self._reach_last_close = snapshot.grasp_closure_score
        self._reach_last_gate = snapshot.grasp_close_gate

    def _home_error(self, snapshot: TaskSnapshot) -> float:
        q = np.asarray(snapshot.measurements.joint_positions_rad)
        error = np.maximum(np.abs(q-self.config.home_joint_positions_rad)
                           - self.config.home_joint_tolerance_rad, 0)
        return float(np.sqrt(np.mean((error/self.config.home_position_scales_rad)**2)))

    def _seed_home_history(self, snapshot: TaskSnapshot, index: int) -> None:
        self._home_initialized[index] = True
        self._home_best[index] = self._home_last[index] = self._home_error(snapshot)
        self._home_last_unsettled[index] = not (snapshot.robot_at_home and snapshot.robot_static)

    def _process_components(self, prev: TaskSnapshot, nxt: TaskSnapshot,
                            dt_s: float) -> dict[str, float]:
        cfg = self.config
        components = dict.fromkeys((
            "reach_progress", "reach_stagnation", "reach_step",
            "reach_x_alignment_progress", "reach_x_alignment_regression",
            "reach_z_alignment_progress", "reach_z_alignment_regression",
            "reach_z_alignment_perfect", "reach_close_alignment", "reach_threshold",
            "reach_gripper_progress", "reach_gripper_regression", "home_progress",
            "home_stagnation", "home_regression", "home_unsettled",
        ), 0.0)
        scale = dt_s * 30.0
        fresh = (0 < nxt.evidence_sim_time_s-prev.evidence_sim_time_s <= cfg.max_evidence_gap_s
                 and self._valid_process_measurement(nxt) and nxt.outcome == "running")
        if self._reach_active(nxt):
            components["reach_step"] = -cfg.reach_step_cost * scale
            previous_distance = self._reach_last_distance
            if fresh and not self._reach_initialized:
                # The target may appear after reset. Its first valid sample is
                # a baseline, never a free distance/orientation/threshold prize.
                self._seed_reach_history(nxt)
                previous_distance = self._reach_last_distance
            distance, x_error, z_error = (self._reach_last_distance,
                                         self._reach_last_x, self._reach_last_z)
            distance_improved = False
            if fresh:
                distance, x_error, z_error = self._reach_geometry(nxt)
                if distance < self._reach_best_distance:
                    components["reach_progress"] = cfg.reach_distance_progress_weight * (self._reach_best_distance-distance)
                    self._reach_best_distance = distance
                    distance_improved = True
                for axis_name, error in (("x", x_error), ("z", z_error)):
                    best = getattr(self, f"_reach_best_{axis_name}")
                    if error < best:
                        old_level = math.floor((best+1e-10)/cfg.reach_alignment_step_deg)
                        level = math.floor((error+1e-10)/cfg.reach_alignment_step_deg)
                        components[f"reach_{axis_name}_alignment_progress"] = cfg.reach_alignment_progress_bonus * max(0, old_level-level)
                        setattr(self, f"_reach_best_{axis_name}", error)
                if not self._reach_perfect_claimed and z_error <= cfg.reach_z_perfect_threshold_deg:
                    components["reach_z_alignment_perfect"] = cfg.reach_z_perfect_bonus
                    self._reach_perfect_claimed = True
                crossed = []
                for i, (threshold, bonus) in enumerate(zip(cfg.reach_distance_thresholds_m,
                                                          cfg.reach_distance_threshold_bonuses)):
                    if not self._reach_threshold_claimed[i] and previous_distance >= threshold > distance:
                        self._reach_threshold_claimed[i] = True
                        crossed.append(bonus)
                components["reach_threshold"] = sum(crossed) * (cfg.reach_multi_threshold_multiplier if len(crossed) >= 2 else 1)
                self._cache_gripper_scores(nxt)
                gate = self._reach_last_gate
                for name, score, weight in (("open", self._reach_last_open, 1-gate),
                                             ("close", self._reach_last_close, gate)):
                    if weight <= 0:
                        continue
                    if not getattr(self, f"_reach_{name}_initialized"):
                        setattr(self, f"_reach_{name}_initialized", True)
                        setattr(self, f"_reach_best_{name}", score)
                    best = getattr(self, f"_reach_best_{name}")
                    if score > best:
                        components["reach_gripper_progress"] += cfg.reach_gripper_progress_weight * weight * (score-best)
                        setattr(self, f"_reach_best_{name}", score)
                self._reach_last_distance, self._reach_last_x, self._reach_last_z = distance, x_error, z_error
            if not distance_improved:
                components["reach_stagnation"] = -cfg.reach_stagnation_distance_weight * distance**2 * scale
            for name, error in (("x", x_error), ("z", z_error)):
                best = getattr(self, f"_reach_best_{name}")
                levels = max(0, math.floor((error+1e-10)/cfg.reach_alignment_step_deg)
                             - math.floor((best+1e-10)/cfg.reach_alignment_step_deg))
                components[f"reach_{name}_alignment_regression"] = -cfg.reach_alignment_regression_cost * levels * scale
            if distance <= cfg.reach_close_alignment_distance_m and z_error > cfg.reach_close_alignment_threshold_deg:
                components["reach_close_alignment"] = -cfg.reach_close_alignment_cost * scale
            gate = self._reach_last_gate
            for name, score, weight in (("open", self._reach_last_open, 1-gate),
                                         ("close", self._reach_last_close, gate)):
                if getattr(self, f"_reach_{name}_initialized"):
                    regression = max(0.0, getattr(self, f"_reach_best_{name}")-score)
                    components["reach_gripper_regression"] -= cfg.reach_gripper_regression_cost * weight * regression**2 * scale
        home_index = self._home_history_index(nxt)
        if home_index is not None:
            i = home_index
            if not self._home_initialized[i] and self._valid_process_measurement(prev):
                self._seed_home_history(prev, i)
            improved = False
            if fresh:
                if not self._home_initialized[i]:
                    self._seed_home_history(nxt, i)
                error = self._home_error(nxt)
                if error < self._home_best[i]:
                    components["home_progress"] = cfg.home_progress_weight * (self._home_best[i]-error)
                    self._home_best[i] = error
                    improved = True
                self._home_last[i] = error
                self._home_last_unsettled[i] = not (nxt.robot_at_home and nxt.robot_static)
            error = self._home_last[i]
            if not improved:
                components["home_stagnation"] = -cfg.home_stagnation_cost * error**2 * scale
            if self._home_initialized[i]:
                components["home_regression"] = -cfg.home_regression_cost * max(0, error-self._home_best[i])**2 * scale
            if self._home_last_unsettled[i]:
                components["home_unsettled"] = -cfg.home_unsettled_step_cost * scale
        return components

    def potential(self, snapshot: TaskSnapshot, *, terminal: bool = False) -> float:
        if terminal or not self.config.shaping_enabled:
            return 0.0
        return self.physical_stage_progress(snapshot) * self.config.potential_max / 10

    def physical_stage_progress(self, snapshot: TaskSnapshot) -> float:
        """Physical stage score in [0, 10], independent of shaping switches.

        Reuse the ordered task scores without normalizing every phase back to
        zero. Perfect reaching still leaves grasp, carry, place and HOME work.
        True terminals are not artificially zeroed here as they are for PBRS.
        """
        s, cfg = snapshot, self.config
        if s.task_mode != "pick_place" or s.phase is TaskPhase.WAIT_HOME:
            raw = 2 * s.home_score
        elif s.phase in (TaskPhase.APPROACH, TaskPhase.GRASP_LIFT):
            # Keep near-target geometry informative before the first contact.
            # The same expression across the candidate-distance boundary
            # avoids a free phase-entry bonus. Measured dual-contact dwell
            # hands off to grasp/lift; closing alone never certifies a grasp.
            pregrasp = 2 * s.approach_score
            grasp_lift = 2 + 2 * (0.3 + 0.7 * s.lift_score)
            raw = (1-s.contact_score) * pregrasp + s.contact_score * grasp_lift
        elif s.phase is TaskPhase.TRANSPORT:
            raw = 4 + 2 * s.transport_score
        elif s.phase is TaskPhase.PLACE_RELEASE:
            p = min(s.place_stable_duration_s / cfg.place_confirm_s, 1)
            raw = 6 + 2 * (0.35 * s.support_score + 0.20 * float(s.valid_release)
                           + 0.15 * s.separation_score + 0.30 * p)
        elif s.phase in (TaskPhase.RETURN_HOME, TaskPhase.HOME_HOLD):
            b = min(s.home_stable_duration_s / s.home_hold_required_s, 1)
            raw = 8 + 2 * (0.7 * s.home_score + 0.3 * b)
        else:
            raise ValueError("unknown task phase")
        if not math.isfinite(raw) or raw < -1e-9 or raw > 10 + 1e-9:
            raise ValueError("snapshot has invalid potential features")
        return float(raw)

    def _valid_dense_progress(self, snapshot: TaskSnapshot) -> bool:
        m = snapshot.measurements
        return bool(
            snapshot.target_name is not None and snapshot.phase is not TaskPhase.WAIT_HOME
            and m.evidence_valid and m.joint_measurement_valid
            and m.target_measurement_valid and m.target_motion_valid
            and (m.contact_valid or self.config.grasp_evidence_mode == "proxy")
            and 0 <= m.sim_time_s-m.evidence_time <= self.config.max_evidence_gap_s
        )

    def reset(self, initial_snapshot: TaskSnapshot) -> None:
        if initial_snapshot.outcome != "running":
            raise ValueError("reward reset requires a running initial task")
        if initial_snapshot.measurements.grasp_evidence_mode != self.config.grasp_evidence_mode:
            raise ValueError("measurement and reward grasp evidence mode mismatch")
        self.potential(initial_snapshot)
        self._previous, self._closed = initial_snapshot, False
        self._dense_target_seen = initial_snapshot.target_name is not None
        self._dense_last_valid_progress = (
            self.physical_stage_progress(initial_snapshot)
            if self.config.reward_schema_version >= 4 and self.config.remaining_task_cost_per_s > 0
            and self._valid_dense_progress(initial_snapshot) else 0.0
        )
        self._reset_process_history()
        if self.config.reward_schema_version >= 5 and self._valid_process_measurement(initial_snapshot):
            if self._reach_active(initial_snapshot):
                self._seed_reach_history(initial_snapshot)
            home_index = self._home_history_index(initial_snapshot)
            if home_index is not None:
                self._seed_home_history(initial_snapshot, home_index)

    def step(self, transition: PickPlaceRewardInput) -> PickPlaceRewardResult:
        if self._previous is None or self._closed:
            raise RuntimeError("reward reset required before a transition or after episode end")
        prev, nxt, cfg = transition.prev_snapshot, transition.next_snapshot, self.config
        if prev != self._previous:
            raise ValueError("reward transition does not continue previous snapshot")
        if prev.task_instance_id != nxt.task_instance_id or prev.task_mode != nxt.task_mode:
            raise ValueError("task identity and mode must not change in a transition")
        if not math.isfinite(transition.dt_s) or transition.dt_s <= 0 or not math.isclose(nxt.sim_time_s-prev.sim_time_s, transition.dt_s, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("dt must match positive control simulation time advance")
        if transition.terminated and transition.truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        terminal_outcome = nxt.outcome in ("success", "task_failure")
        if bool(transition.terminated) != terminal_outcome:
            raise ValueError("true termination must agree with the task outcome")
        if nxt.outcome not in ("running", "success", "task_failure"):
            raise ValueError("unknown task outcome")
        if nxt.measurements.grasp_evidence_mode != cfg.grasp_evidence_mode:
            raise ValueError("grasp evidence mode changed within episode")
        before = self.potential(prev)
        after = self.potential(nxt, terminal=transition.terminated)
        # Per-step discount is shared with PPO even when the measured dt varies.
        components = {"potential_shaping": cfg.gamma * after - before,
                      "success": 0.0, "failure": 0.0, "active_time": 0.0,
                      "home_position": 0.0, "home_velocity": 0.0,
                      "moving_velocity": 0.0, "command_change": 0.0}
        if nxt.outcome == "success":
            if nxt.task_mode != "pick_place" or not nxt.success_event:
                raise ValueError("success requires one pick/place success event")
            components["success"] = cfg.success_bonus
        elif nxt.outcome == "task_failure":
            components["failure"] = -cfg.failure_cost
        if prev.task_mode == "pick_place" and prev.target_name is not None:
            components["active_time"] = -cfg.active_time_cost_per_s * transition.dt_s
        dense_target_seen = self._dense_target_seen or nxt.target_name is not None
        dense_progress = self._dense_last_valid_progress
        if cfg.reward_schema_version >= 4:
            components["remaining_task"] = 0.0
            fresh = 0 < nxt.evidence_sim_time_s-prev.evidence_sim_time_s <= cfg.max_evidence_gap_s
            if cfg.remaining_task_cost_per_s > 0 and fresh and self._valid_dense_progress(nxt):
                dense_progress = self.physical_stage_progress(nxt)
            if prev.task_mode == "pick_place" and dense_target_seen and nxt.outcome != "success":
                # Unknown data cannot improve the score or waive an active
                # task's cost. Carry forward the last measured progress until
                # evidence returns; valid physical regression increases cost.
                remaining = float(np.clip(1-dense_progress/10, 0, 1))
                components["remaining_task"] = -float(cfg.remaining_task_cost_per_s) * transition.dt_s * remaining
        # Cost units are a 30 Hz reference budget, irrespective of control rate.
        step_scale = transition.dt_s * 30.0
        holding = nxt.task_mode != "pick_place" or nxt.phase in (TaskPhase.WAIT_HOME, TaskPhase.HOME_HOLD)
        m = nxt.measurements
        qerr = np.maximum(np.abs(np.asarray(m.joint_positions_rad)-cfg.home_joint_positions_rad)-cfg.home_joint_tolerance_rad, 0)
        speed = np.maximum(np.abs(m.joint_velocities_rad_s)-cfg.home_velocity_tolerance_rad_s, 0)
        speed_cost = float(np.mean(np.clip(speed/np.asarray(cfg.velocity_scales_rad_s), 0, 1)**2))
        if holding and m.joint_measurement_valid and m.evidence_valid:
            if cfg.reward_schema_version == 3:
                # Keep historical base rewards bit-for-bit, including the
                # clipped plateau beyond each joint's configured scale.
                position_cost = float(np.mean(np.clip(qerr/np.asarray(cfg.home_position_scales_rad), 0, 1)**2))
            else:
                # x²/(1+x²) retains a correction signal even at the observed
                # ~78° idle gripper error. qerr/hypot(scale,qerr) avoids
                # overflowing x² for large but finite position errors.
                normalized_error = qerr / np.hypot(np.asarray(cfg.home_position_scales_rad), qerr)
                position_cost = float(np.mean(normalized_error**2))
            components["home_position"] = -cfg.home_position_cost_per_step * step_scale * position_cost
            components["home_velocity"] = -cfg.home_velocity_cost_per_step * step_scale * speed_cost
        elif not holding and m.joint_measurement_valid and m.evidence_valid:
            components["moving_velocity"] = -cfg.moving_velocity_cost_per_step * step_scale * speed_cost
        current_command = transition.command_joint_targets_rad
        previous_command = transition.previous_command_joint_targets_rad
        if current_command is not None:
            current_command = np.asarray(_vector(current_command, 6, "command_joint_targets_rad"))
        if previous_command is not None:
            previous_command = np.asarray(_vector(previous_command, 6, "previous_command_joint_targets_rad"))
        if cfg.command_change_cost_per_step > 0 and (current_command is None or previous_command is None):
            raise ValueError("enabled command change cost requires actual current and previous commands")
        if current_command is not None and previous_command is not None:
            change = np.clip(np.abs(current_command-previous_command)/np.asarray(cfg.command_change_scales_rad), 0, 1)
            components["command_change"] = -cfg.command_change_cost_per_step * step_scale * float(np.mean(change**2))
        if cfg.reward_schema_version >= 5:
            components.update(self._process_components(prev, nxt, transition.dt_s))
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise ValueError("nonfinite reward")
        self._previous, self._closed = nxt, bool(transition.terminated or transition.truncated)
        self._dense_target_seen, self._dense_last_valid_progress = dense_target_seen, dense_progress
        return PickPlaceRewardResult(total, components, before, after, transition.terminated, transition.truncated)
