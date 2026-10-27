"""Continuous process guidance for the measured S1 reach objective.

S0 retains PickPlaceReward. S1 uses the same geometry and tolerances as its
StageEvaluator, including the open-jaw interval rather than a grasp-close gate.
This module never commands the robot or changes success / terminal semantics.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from .pick_place_reward import PickPlaceRewardConfig, PickPlaceRewardInput, PickPlaceRewardResult
from .task_state import TaskSnapshot
from .s1_lift_approach import S1LiftApproachConfig

if TYPE_CHECKING:
    from .training_stages import StageTrainingConfig


S1_POTENTIAL_NAMES = (
    "distance_coarse", "distance_fine", "xy", "height", "bearing", "axis",
    "opening", "combined", "hold",
)
STAGE_PROCESS_CONTEXT_SCHEMA_VERSION = 1
STAGE_PROCESS_CONTEXT_DIM = 12


@dataclass(frozen=True)
class StageProcessRewardConfig:
    schema_version: int = 1
    s1_distance_coarse_scale_m: float = 0.1
    s1_distance_fine_scale_m: float = 0.02
    s1_xy_scale_m: float = 0.03
    s1_height_scale_m: float = 0.02
    s1_orientation_scale_rad: float = 0.35
    s1_opening_scale_rad: float = 0.2
    s1_soft_gate_floor: float = 0.25
    s1_potential_weights: tuple[float, ...] = (20.0, 20.0, 15.0, 10.0, 5.0, 5.0, 10.0, 10.0, 5.0)
    s1_time_cost_per_s: float = 0.03
    s1_remaining_cost_per_s: float = 0.3
    s1_success_bonus: float = 10.0
    # Version 2 adds a bounded task signal that does not telescope with PBRS.
    # None preserves the exact version-1 serialized/checkpoint contract.
    s1_shaping_scale: float | None = None
    s1_distance_cost_per_s: float | None = None
    s1_best_distance_bonus: float | None = None
    # Version 3 adds continuous first-visit credit within distance bands.
    # The existing best-quality record fully determines claimed band credit.
    s1_distance_band_edges_m: tuple[float, ...] | None = None
    s1_distance_band_bonuses: tuple[float, ...] | None = None
    # Version 4 sequences lift and approach inside the same S1 episode.
    s1_lift_approach: S1LiftApproachConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version not in (1, 2, 3, 4):
            raise ValueError("unsupported stage process reward schema_version")
        if self.schema_version == 4:
            if not isinstance(self.s1_lift_approach, S1LiftApproachConfig):
                raise ValueError("schema 4 requires s1_lift_approach configuration")
        elif self.s1_lift_approach is not None:
            raise ValueError("s1_lift_approach requires schema_version=4")
        v2_fields = ("s1_shaping_scale", "s1_distance_cost_per_s", "s1_best_distance_bonus")
        for name in v2_fields:
            value = getattr(self, name)
            if self.schema_version == 1:
                if value is not None:
                    raise ValueError(f"{name} requires stage process reward schema_version=2")
            elif (isinstance(value, bool) or not isinstance(value, Real)
                  or not math.isfinite(value) or value <= 0):
                raise ValueError(f"schema {self.schema_version} requires positive finite {name}")
        band_fields = ("s1_distance_band_edges_m", "s1_distance_band_bonuses")
        if self.schema_version < 3:
            if any(getattr(self, name) is not None for name in band_fields):
                raise ValueError("distance bands require stage process reward schema_version=3")
        else:
            try:
                edges = tuple(self.s1_distance_band_edges_m)
                bonuses = tuple(self.s1_distance_band_bonuses)
            except TypeError as error:
                raise ValueError("schema 3 requires distance band edges and bonuses") from error
            if len(edges) < 2 or len(bonuses) != len(edges) - 1:
                raise ValueError("distance bands require one bonus per interval")
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(value) or value <= 0 for value in (*edges, *bonuses)):
                raise ValueError("distance band edges and bonuses must be positive finite numbers")
            if any(upper <= lower for upper, lower in zip(edges, edges[1:])):
                raise ValueError("distance band edges must be strictly decreasing")
            if not math.isfinite(sum(bonuses)):
                raise ValueError("sum of distance band bonuses must be finite")
            object.__setattr__(self, "s1_distance_band_edges_m", tuple(float(value) for value in edges))
            object.__setattr__(self, "s1_distance_band_bonuses", tuple(float(value) for value in bonuses))
        for field in fields(self):
            if field.name in ("schema_version", "s1_potential_weights", "s1_lift_approach", *v2_fields, *band_fields):
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if value < 0 or (value == 0 and field.name not in (
                "s1_time_cost_per_s", "s1_remaining_cost_per_s", "s1_success_bonus",
            )):
                raise ValueError(f"{field.name} must be positive (cost and bonus may be zero)")
        if not 0 < self.s1_soft_gate_floor <= 1:
            raise ValueError("s1_soft_gate_floor must be in (0, 1]")
        if self.s1_distance_fine_scale_m >= self.s1_distance_coarse_scale_m:
            raise ValueError("fine distance scale must be smaller than coarse distance scale")
        try:
            weights = tuple(self.s1_potential_weights)
        except TypeError as error:
            raise ValueError("s1_potential_weights must contain nine positive finite weights") from error
        if len(weights) != len(S1_POTENTIAL_NAMES) or any(
            isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0 for value in weights
        ):
            raise ValueError("s1_potential_weights must contain nine positive finite weights")
        if not math.isfinite(sum(weights)):
            raise ValueError("sum of potential weights must be finite")
        object.__setattr__(self, "s1_potential_weights", tuple(float(value) for value in weights))

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> StageProcessRewardConfig:
        if not isinstance(values, Mapping):
            raise ValueError("stage process reward configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown stage process reward fields: {sorted(unknown)}")
        values = dict(values)
        if values.get("s1_lift_approach") is not None:
            values["s1_lift_approach"] = S1LiftApproachConfig.from_dict(values["s1_lift_approach"])
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        if self.schema_version < 4:
            values.pop("s1_lift_approach")
        else:
            values["s1_lift_approach"] = self.s1_lift_approach.to_dict()
        if self.schema_version == 1:
            for name in ("s1_shaping_scale", "s1_distance_cost_per_s", "s1_best_distance_bonus"):
                values.pop(name)
        if self.schema_version < 3:
            for name in ("s1_distance_band_edges_m", "s1_distance_band_bonuses"):
                values.pop(name)
        return json.loads(json.dumps(values, allow_nan=False))

    @property
    def context_dim(self) -> int:
        if self.schema_version == 4:
            # Append records and carried per-channel costs; keep the old 26
            # slots intact for historical configurations and diagnostics.
            return 34 if self.s1_lift_approach.posture_regression is not None else 26
        return STAGE_PROCESS_CONTEXT_DIM + (2 if self.schema_version >= 2 else 0)

    @property
    def context_schema_version(self) -> int:
        if self.schema_version == 4 and self.s1_lift_approach.posture_regression is not None:
            return 5
        return self.schema_version

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric(info: Mapping[str, object], name: str) -> float:
    value = info.get(name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"stage process reward requires finite {name}")
    return float(value)


def _flag(info: Mapping[str, object], name: str) -> bool:
    value = info.get(name)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"stage process reward requires boolean {name}")
    return bool(value)


def _interval_error(value: float, low: float, high: float) -> float:
    return max(low - value, value - high, 0.0)


class StageReachReward:
    """S1-only reward; use StageEvaluator info from the same sensor snapshot.

    Each logged progress component is the signed change in a continuous,
    bounded potential. A separate discount correction makes their sum exactly
    gamma * Phi(next) - Phi(previous). Thus recovering from a bad step earns
    progress even without a new historical best, while closed loops cannot
    produce positive discounted shaping returns.
    """

    def __init__(self, config: StageProcessRewardConfig, stage_config: StageTrainingConfig,
                 reward_config: PickPlaceRewardConfig):
        if config.schema_version == 4 and type(self) is StageReachReward:
            raise ValueError("schema 4 requires S1LiftApproachReward")
        self.config, self.stage_config, self.reward_config = config, stage_config, reward_config
        if config.schema_version >= 3 and not math.isclose(
            config.s1_distance_band_edges_m[-1], stage_config.s1_reach_distance_m,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("distance bands must end at the S1 reach distance tolerance")
        self._previous: TaskSnapshot | None = None
        self._potentials = np.zeros(len(S1_POTENTIAL_NAMES), dtype=np.float64)
        self._elapsed_fraction = 0.0
        self._closed = False
        self._best_distance_quality = 0.0
        self._previous_eligible = False

    def _eligible(self, snapshot: TaskSnapshot, info: Mapping[str, object]) -> bool:
        m = snapshot.measurements
        return (all(_flag(info, name) for name in (
            "stage_target_at_source", "stage_target_source_supported", "stage_target_static",
        )) and not _flag(info, "stage_target_in_goal")
            and not (m.left_contact or m.right_contact or snapshot.grasp_valid
                     or snapshot.grasp_attempt_active or snapshot.lifted_once))

    def _distance_quality(self, info: Mapping[str, object]) -> float:
        distance = _metric(info, "stage_reach_distance_m")
        if distance < 0:
            raise ValueError("invalid nonnegative reach distance")
        excess = max(distance - self.stage_config.s1_reach_distance_m, 0.0)
        scale = self.config.s1_distance_coarse_scale_m
        return scale / (scale + excess)

    def _distance_band_credit(self, quality: float) -> np.ndarray:
        """Total credit reached at a distance, independent of how it was reached.

        Only positive differences above the historical record are paid. Each
        interval is linear, so sub-millimetre progress has no threshold jump.
        The record also consumes ineligible/gap endpoints, as in schema 2.
        """
        edges = np.asarray(self.config.s1_distance_band_edges_m, dtype=np.float64)
        bonuses = np.asarray(self.config.s1_distance_band_bonuses, dtype=np.float64)
        if quality <= 0:
            return np.zeros_like(bonuses)
        distance = (self.stage_config.s1_reach_distance_m
                    + self.config.s1_distance_coarse_scale_m * (1.0 / quality - 1.0))
        fractions = np.clip((edges[:-1] - distance) / (edges[:-1] - edges[1:]), 0.0, 1.0)
        return bonuses * fractions

    @property
    def _shaping_scale(self) -> float:
        return 1.0 if self.config.schema_version == 1 else float(self.config.s1_shaping_scale)

    def _validate_snapshot(self, snapshot: TaskSnapshot, info: Mapping[str, object]) -> None:
        if info.get("training_stage") != "S1":
            raise ValueError("StageReachReward requires S1 stage measurements")
        m = snapshot.measurements
        if not all((m.evidence_valid, m.joint_measurement_valid, m.target_measurement_valid,
                    m.target_motion_valid, m.contact_valid)) or m.target_name is None:
            raise ValueError("invalid stage sensor evidence")
        if m.sim_time_s - m.evidence_time > self.reward_config.max_evidence_gap_s + 1e-9:
            raise ValueError("stale stage sensor evidence")
        if m.grasp_evidence_mode != "contact" or self.reward_config.grasp_evidence_mode != "contact":
            raise ValueError("stage process reward requires measured contact evidence")

    def _score_terms(self, snapshot: TaskSnapshot, info: Mapping[str, object]) -> np.ndarray:
        c, s = self.config, self.stage_config
        weights = np.asarray(c.s1_potential_weights, dtype=np.float64)
        # This flag is certified by StageEvaluator from real simultaneous
        # contact, grasp and lift evidence; a premature close is not enough.
        if _flag(info, "stage_downstream_grasp_lift"):
            return weights.copy()
        if not self._eligible(snapshot, info):
            return np.zeros_like(weights)

        def quality(error: float, scale: float) -> float:
            return scale / (scale + max(error, 0.0))

        def gate(score: float) -> float:
            return c.s1_soft_gate_floor + (1.0 - c.s1_soft_gate_floor) * score

        distance = _metric(info, "stage_reach_distance_m")
        xy = _metric(info, "stage_reach_xy_m")
        bearing = _metric(info, "stage_bearing_error_rad")
        axis = _metric(info, "stage_axis_error_rad")
        hold_s = _metric(info, "stage_reach_hold_s")
        if min(distance, xy, bearing, axis, hold_s) < 0 or max(bearing, axis) > math.pi + 1e-9:
            raise ValueError("invalid nonnegative reach geometry or hold time")
        coarse = quality(distance - s.s1_reach_distance_m, c.s1_distance_coarse_scale_m)
        fine = quality(distance - s.s1_reach_distance_m, c.s1_distance_fine_scale_m)
        horizontal = quality(xy - s.s1_reach_xy_m, c.s1_xy_scale_m)
        height = quality(_interval_error(_metric(info, "stage_reach_dz_m"),
                                         s.s1_reach_dz_min_m, s.s1_reach_dz_max_m), c.s1_height_scale_m)
        direction = quality(bearing - s.s1_orientation_tolerance_rad, c.s1_orientation_scale_rad)
        downward = quality(axis - s.s1_orientation_tolerance_rad, c.s1_orientation_scale_rad)
        opening = quality(_interval_error(_metric(info, "stage_opening_rad"),
                                          s.s1_opening_min_rad, s.s1_opening_max_rad), c.s1_opening_scale_rad)
        # A soft sequence leaves a nonzero signal for all objectives at any
        # distance; no fixed milestone can permanently disable a later skill.
        alignment_gate = gate(horizontal) * gate(height)
        combined = math.prod((fine, horizontal, height, direction, downward, opening)) ** (1.0 / 6.0)
        # Instant distance success has no dwell objective or dwell bonus.
        hold_score = (0.0 if s.s1_success_mode == "lift_then_distance" else
                      min(hold_s / s.s1_reach_hold_s, 1.0))
        scores = np.asarray((coarse, fine, horizontal, gate(horizontal) * height,
                             alignment_gate * direction, alignment_gate * downward,
                             alignment_gate * opening, combined,
                             hold_score))
        return weights * scores

    def _elapsed(self, info: Mapping[str, object]) -> float:
        elapsed = _metric(info, "stage_elapsed_s")
        if elapsed < 0:
            raise ValueError("stage elapsed time must be nonnegative")
        return min(elapsed / self.stage_config.s1_budget_s, 1.0)

    def reset(self, snapshot: TaskSnapshot, stage_info: Mapping[str, object]) -> None:
        self._validate_snapshot(snapshot, stage_info)
        if snapshot.outcome != "running" or not stage_info.get("start_valid"):
            raise ValueError("stage reward reset requires a valid running start certificate")
        potentials = self._score_terms(snapshot, stage_info) * self._shaping_scale
        elapsed = self._elapsed(stage_info)
        self._previous, self._potentials = snapshot, potentials
        self._elapsed_fraction, self._closed = elapsed, False
        if self.config.schema_version >= 2:
            self._best_distance_quality = self._distance_quality(stage_info)
            self._previous_eligible = self._eligible(snapshot, stage_info)

    def critic_context(self) -> np.ndarray:
        # S0 supplies an all-zero vector of this same size at the core boundary.
        values = (1.0, float(self._previous is not None), *self._potentials, self._elapsed_fraction)
        if self.config.schema_version >= 2:
            values += (self._best_distance_quality, float(self._previous_eligible))
        return np.asarray(values, dtype=np.float32)

    def step(self, transition: PickPlaceRewardInput,
             stage_info: Mapping[str, object]) -> PickPlaceRewardResult:
        if self._previous is None or self._closed:
            raise RuntimeError("stage reward reset required before a transition or after episode end")
        prev, nxt = transition.prev_snapshot, transition.next_snapshot
        if prev != self._previous:
            raise ValueError("reward transition does not continue previous snapshot")
        if (prev.task_instance_id, prev.task_mode, prev.target_name) != (nxt.task_instance_id, nxt.task_mode, nxt.target_name):
            raise ValueError("stage task identity must not change")
        if not math.isfinite(transition.dt_s) or transition.dt_s <= 0 or not math.isclose(
            nxt.sim_time_s - prev.sim_time_s, transition.dt_s, rel_tol=1e-6, abs_tol=1e-6,
        ):
            raise ValueError("dt must match positive control simulation time advance")
        if transition.terminated and transition.truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        if nxt.outcome not in ("running", "success", "task_failure"):
            raise ValueError("unknown task outcome")
        if bool(transition.terminated) != (nxt.outcome in ("success", "task_failure")):
            raise ValueError("true termination must agree with the task outcome")
        self._validate_snapshot(nxt, stage_info)
        elapsed = self._elapsed(stage_info)
        evidence_dt = nxt.evidence_sim_time_s - prev.evidence_sim_time_s
        if evidence_dt < -1e-9:
            raise ValueError("stage evidence time moved backwards")
        fresh = (0 < evidence_dt <= self.reward_config.max_evidence_gap_s + 1e-9
                 and _flag(stage_info, "stage_evidence_fresh"))
        before = self._potentials.copy()
        if transition.terminated:
            after = np.zeros_like(before)
        elif evidence_dt > 0:
            after = self._score_terms(nxt, stage_info) * self._shaping_scale
        else:
            # A callback repeat cannot create progress, even if a caller has
            # changed the metrics without advancing the underlying evidence.
            after = before.copy()
        deltas = after - before
        components = {f"s1_{name}_progress": float(value) for name, value in zip(S1_POTENTIAL_NAMES, deltas)}
        components["s1_discount_correction"] = (self.reward_config.gamma - 1.0) * float(after.sum())
        # Rebase on a measured endpoint after a long evidence gap, but forfeit
        # positive changes across the missing interval. Negative changes stay.
        components["s1_evidence_guard"] = -float(np.maximum(deltas, 0).sum()) if not fresh else 0.0
        components["s1_time"] = -self.config.s1_time_cost_per_s * transition.dt_s
        # A bounded dense objective also values better partial reach states,
        # even before a rollout contains a successful reach. This is distinct
        # from PBRS; it never accumulates debt against an old best angle.
        quality = float(after[:-1].sum()) / (sum(self.config.s1_potential_weights[:-1]) * self._shaping_scale)
        components["s1_remaining"] = -self.config.s1_remaining_cost_per_s * transition.dt_s * max(0.0, 1.0 - quality)
        if self.config.schema_version >= 2:
            eligible = self._eligible(nxt, stage_info)
            distance_quality = self._distance_quality(stage_info)
            components["s1_best_distance"] = 0.0
            if self.config.schema_version == 3:
                for index in range(len(self.config.s1_distance_band_bonuses)):
                    components[f"s1_distance_band_{index + 1:02d}"] = 0.0
            # Pay only for new measured progress across two qualified endpoints.
            # Keep the high-water mark across contacts/gaps; recovery or repeated
            # evidence cannot repeatedly collect the same distance improvement.
            if evidence_dt > 0:
                if fresh and eligible and self._previous_eligible and not transition.terminated:
                    components["s1_best_distance"] = float(self.config.s1_best_distance_bonus) * max(
                        0.0, distance_quality - self._best_distance_quality,
                    )
                    if self.config.schema_version == 3:
                        credits = np.maximum(self._distance_band_credit(distance_quality)
                                             - self._distance_band_credit(self._best_distance_quality), 0.0)
                        for index, value in enumerate(credits):
                            components[f"s1_distance_band_{index + 1:02d}"] = float(value)
                self._best_distance_quality = max(self._best_distance_quality, distance_quality)
            self._previous_eligible = eligible and fresh
            # Distance is an explicit objective: better opening/orientation alone
            # cannot remove the cost of remaining far from the target.
            if not eligible and not _flag(stage_info, "stage_downstream_grasp_lift"):
                distance_quality = 0.0
            elif evidence_dt <= 0:
                # Reused evidence cannot claim a newly improved dense state cost.
                distance_quality = float(before[0]) / (self.config.s1_potential_weights[0] * self._shaping_scale)
            components["s1_distance_remaining"] = -float(self.config.s1_distance_cost_per_s) * transition.dt_s * (1.0 - distance_quality)
        components["s1_success"] = (self.config.s1_success_bonus if fresh and not transition.terminated
                                     and _flag(stage_info, "stage_success") else 0.0)
        components["failure"] = -self.reward_config.failure_cost if nxt.outcome == "task_failure" else 0.0
        components["success"] = self.reward_config.success_bonus if nxt.outcome == "success" and nxt.success_event else 0.0
        reward = float(sum(components.values()))
        if not math.isfinite(reward):
            raise ValueError("non-finite stage process reward")
        self._previous, self._potentials, self._elapsed_fraction = nxt, after, elapsed
        self._closed = bool(transition.terminated or transition.truncated)
        return PickPlaceRewardResult(reward, components, float(before.sum()), float(after.sum()),
                                     transition.terminated, transition.truncated)
