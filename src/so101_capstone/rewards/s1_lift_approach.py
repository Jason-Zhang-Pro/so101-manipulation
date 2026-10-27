"""Measured lift-then-approach guidance inside one S1 sampling segment.

This module has no action or world-control interface. Heights use the measured
table support plane and the existing grasp reference point; wrist rotation is
the physical wrist_roll joint relative to HOME, not a wrapped visual heading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from numbers import Real
from typing import Mapping

from .pick_place_reward import PickPlaceRewardConfig
from .task_state import TaskSnapshot


@dataclass(frozen=True)
class S1PostureRegressionConfig:
    """Time-based costs for losing previously demonstrated S1 alignment.

    The four independent channels share these rates. The cap keeps the cost
    budget finite; it does not turn the cost into a one-time record penalty.
    """

    slack_rad: float = math.radians(0.5)
    error_scale_rad: float = math.radians(10.0)
    parallel_tolerance_rad: float = math.radians(3.0)
    cost_per_s: float = 0.02

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"posture regression {field.name} must be positive and finite")
        if self.slack_rad >= self.error_scale_rad:
            raise ValueError("posture regression slack must be smaller than error scale")
        if self.parallel_tolerance_rad >= math.pi / 2:
            raise ValueError("parallel tolerance must be smaller than pi/2")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> S1PostureRegressionConfig:
        if not isinstance(values, Mapping):
            raise ValueError("posture regression configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown posture regression fields: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True)
class S1LiftApproachConfig:
    lift_clearance_m: float = 0.08
    lift_hold_s: float = 0.1
    lift_band_step_m: float = 0.005
    lift_band_bonus: float = 2.0
    # One measured LIFT -> APPROACH event per episode; zero preserves old profiles.
    lift_completion_bonus: float = 0.0
    # Disable only wrist-roll / finger-parallel shaping, not target alignment.
    # Historical profiles omit True to preserve their saved reward contracts.
    wrist_rotation_reward_enabled: bool = True
    # Looking from the wrist toward the fingers, negative wrist_roll is CW.
    roll_target_rad: float = -math.pi / 2.0
    roll_tolerance_rad: float = math.radians(15.0)
    roll_band_step_rad: float = math.radians(10.0)
    roll_band_bonus: float = 2.0
    approach_outer_xy_m: float = 0.06
    approach_inner_xy_m: float = 0.01
    approach_hover_clearance_m: float = 0.08
    approach_final_dz_m: float = 0.01
    corridor_tolerance_m: float = 0.005
    lift_potential_weight: float = 20.0
    roll_potential_weight: float = 20.0
    parallel_potential_weight: float = 10.0
    lift_remaining_cost_per_s: float = 0.4
    roll_remaining_cost_per_s: float = 0.3
    corridor_cost_per_s: float = 0.5
    # Keep a whole S1 window's dense costs below the physical failure cost.
    dense_cost_failure_fraction: float = 0.9
    # Omitted in historical configurations, including their serialized hashes.
    posture_regression: S1PostureRegressionConfig | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "wrist_rotation_reward_enabled":
                if type(value) is not bool:
                    raise ValueError("wrist_rotation_reward_enabled must be a boolean")
                continue
            if field.name == "posture_regression":
                if value is not None and not isinstance(value, S1PostureRegressionConfig):
                    raise ValueError("posture_regression must be a posture regression configuration")
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite real number")
            if field.name == "lift_completion_bonus":
                if value < 0:
                    raise ValueError("lift_completion_bonus must be nonnegative")
                continue
            if field.name != "roll_target_rad" and value <= 0:
                raise ValueError(f"{field.name} must be positive")
        if not 0 < abs(self.roll_target_rad) <= math.pi:
            raise ValueError("roll_target_rad must be nonzero and within [-pi, pi]")
        if self.roll_tolerance_rad >= abs(self.roll_target_rad):
            raise ValueError("roll tolerance must be smaller than the requested rotation")
        if self.roll_band_step_rad > abs(self.roll_target_rad):
            raise ValueError("roll band step must not exceed the requested rotation")
        if self.lift_band_step_m > self.lift_clearance_m:
            raise ValueError("lift band step must not exceed the lift clearance")
        if self.approach_inner_xy_m >= self.approach_outer_xy_m:
            raise ValueError("approach inner XY must be smaller than outer XY")
        if self.approach_final_dz_m >= self.approach_hover_clearance_m:
            raise ValueError("final approach dz must be smaller than hover clearance")
        if self.corridor_tolerance_m >= self.approach_hover_clearance_m:
            raise ValueError("corridor tolerance must be smaller than hover clearance")
        if self.dense_cost_failure_fraction >= 1:
            raise ValueError("dense cost failure fraction must be strictly below one")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> S1LiftApproachConfig:
        if not isinstance(values, Mapping):
            raise ValueError("S1 lift approach configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown S1 lift approach fields: {sorted(unknown)}")
        values = dict(values)
        if values.get("posture_regression") is not None:
            values["posture_regression"] = S1PostureRegressionConfig.from_dict(values["posture_regression"])
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        if self.posture_regression is None:
            values.pop("posture_regression")
        if self.lift_completion_bonus == 0:
            values.pop("lift_completion_bonus")
        if self.wrist_rotation_reward_enabled:
            values.pop("wrist_rotation_reward_enabled")
        return values


class S1LiftApproachTracker:
    """Latch an evidence-qualified lift once; never reset the S1 clock.

    StageEvaluator validates timestamps and supplies the covered fresh control
    interval. Reused evidence cannot add dwell; a long gap or invalid lift
    endpoint clears it. Once lifted, low reach positions do not undo the latch.
    """

    def __init__(self, config: S1LiftApproachConfig, reward_config: PickPlaceRewardConfig):
        self.config, self.reward_config = config, reward_config
        self.phase = "LIFT"
        self.lift_hold_s = 0.0
        self._last_lift_ready = False

    @staticmethod
    def _eligible(snapshot: TaskSnapshot, metrics: Mapping[str, object]) -> bool:
        m = snapshot.measurements
        return bool(
            m.target_name is not None
            and metrics["stage_target_at_source"]
            and metrics["stage_target_source_supported"]
            and metrics["stage_target_static"]
            and not metrics["stage_target_in_goal"]
            and not (m.left_contact or m.right_contact or snapshot.grasp_valid
                     or snapshot.grasp_attempt_active or snapshot.lifted_once)
        )

    def _lift_ready(self, snapshot: TaskSnapshot, metrics: Mapping[str, object]) -> bool:
        m = snapshot.measurements
        clearance = m.grasp_position_world_m[2] - m.source_support_z_m
        return self._eligible(snapshot, metrics) and clearance + 1e-9 >= self.config.lift_clearance_m

    def reset(self, snapshot: TaskSnapshot, metrics: Mapping[str, object]) -> dict[str, object]:
        self.phase, self.lift_hold_s = "LIFT", 0.0
        self._last_lift_ready = self._lift_ready(snapshot, metrics)
        return self.metrics(snapshot, metrics)

    def update(self, snapshot: TaskSnapshot, metrics: Mapping[str, object], *,
               fresh: bool, covered_s: float, evidence_gap: bool) -> dict[str, object]:
        if not math.isfinite(covered_s) or covered_s < 0:
            raise ValueError("covered lift evidence time must be finite and nonnegative")
        ready = self._lift_ready(snapshot, metrics)
        if self.phase == "LIFT":
            if evidence_gap or not ready:
                self.lift_hold_s = 0.0
            elif fresh:
                self.lift_hold_s = self.lift_hold_s + covered_s if self._last_lift_ready else 0.0
            if fresh and ready and self.lift_hold_s + 1e-9 >= self.config.lift_hold_s:
                self.phase = "APPROACH"
                self.lift_hold_s = self.config.lift_hold_s
        self._last_lift_ready = ready
        return self.metrics(snapshot, metrics)

    def metrics(self, snapshot: TaskSnapshot, metrics: Mapping[str, object]) -> dict[str, object]:
        m, c = snapshot.measurements, self.config
        clearance = float(m.grasp_position_world_m[2] - m.source_support_z_m)
        # The SO-101 joint order is shoulder_pan/lift, elbow, wrist_flex/roll,
        # gripper. Do not wrap: rotating the other way is a different motion.
        roll = float(m.joint_positions_rad[4] - self.reward_config.home_joint_positions_rad[4])
        roll_error = abs(roll - c.roll_target_rad)
        parallel_error = math.asin(min(1.0, abs(float(m.grasp_rotation_world[2][0]))))
        xy = float(metrics["stage_reach_xy_m"])
        target_clearance = float(m.target_position_world_m[2] - m.source_support_z_m)
        fraction = min(1.0, max(0.0, (xy - c.approach_inner_xy_m)
                                   / (c.approach_outer_xy_m - c.approach_inner_xy_m)))
        final_clearance = target_clearance + c.approach_final_dz_m
        required = final_clearance + fraction * (c.approach_hover_clearance_m - final_clearance)
        deficit = max(0.0, required - clearance)
        return {
            "stage_s1_phase": self.phase,
            "stage_s1_lift_completed": self.phase == "APPROACH",
            "stage_s1_lift_hold_s": self.lift_hold_s,
            "stage_s1_clearance_m": clearance,
            "stage_s1_roll_error_rad": roll_error,
            "stage_s1_parallel_error_rad": parallel_error,
            "stage_s1_wrist_rotation_reward_enabled": c.wrist_rotation_reward_enabled,
            "stage_s1_required_clearance_m": required,
            "stage_s1_corridor_deficit_m": deficit,
            "stage_s1_corridor_safe": deficit <= c.corridor_tolerance_m + 1e-9,
        }
