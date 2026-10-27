"""Measured stage boundaries and optional early-training reward configuration.

S0 and S1 certify measured behaviour; they never command the robot or change
physical task outcomes. The environment selects an optional S1 process reward;
SUCCESS and TIMEOUT remain external sampling cuts requiring value bootstrap.
A physical task failure retains its own terminal semantics. All thresholds
are calibration candidates, not validated settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .pick_place_reward import PickPlaceRewardConfig
from .s0_home_success import S0HomeSuccess
from .task_state import TaskSnapshot, object_geometry

if TYPE_CHECKING:
    from .stage_process_reward import StageProcessRewardConfig


@dataclass(frozen=True)
class StageTrainingConfig:
    stage_schema_version: int = 1
    calibration_status: str = "candidate_unvalidated"
    s0_budget_s: float = 5.0
    s0_home_hold_s: float = 3.0
    # Stage-only success tolerances; reset certification and task rewards keep
    # using the calibrated reward HOME contract. None preserves older runs.
    s0_home_joint_tolerance_rad: float | None = None
    s0_home_velocity_tolerance_rad_s: float | None = None
    s0_success_mode: str = "window_end"
    s1_budget_s: float = 10.0
    s2_budget_s: float = 10.0
    s3_budget_s: float = 10.0
    s4_budget_s: float = 10.0
    enable_later_stage_context: bool = False
    stage_preparation: dict | None = None
    s1_reach_distance_m: float = 0.020
    s1_reach_xy_m: float = 0.010
    s1_reach_dz_min_m: float = 0.0
    s1_reach_dz_max_m: float = 0.020
    s1_orientation_tolerance_rad: float = math.radians(20.0)
    s1_opening_min_rad: float = 0.45
    s1_opening_max_rad: float = 0.90
    s1_reach_hold_s: float = 0.15
    # Preserve historical success contracts unless a profile explicitly opts in.
    s1_success_mode: str = "qualified_reach"
    process_reward: StageProcessRewardConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stage_schema_version, bool) or self.stage_schema_version != 1:
            raise ValueError("unsupported stage_schema_version")
        if not isinstance(self.calibration_status, str) or not self.calibration_status.strip():
            raise ValueError("calibration_status must be a nonempty string")
        if not isinstance(self.enable_later_stage_context, bool):
            raise ValueError("enable_later_stage_context must be boolean")
        if self.stage_preparation is not None:
            if not isinstance(self.stage_preparation, dict):
                raise ValueError("stage_preparation must be a JSON object")
            # The runtime validates named controller fields before moving.
            # Configuration files must nevertheless remain finite JSON data.
            try:
                json.dumps(self.stage_preparation, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("stage_preparation must contain finite JSON values") from error
        if not isinstance(self.s0_success_mode, str) or self.s0_success_mode not in {
            "window_end", "home_hold", "position_hold",
        }:
            raise ValueError("unsupported s0_success_mode")
        if not isinstance(self.s1_success_mode, str) or self.s1_success_mode not in {
            "qualified_reach", "lift_then_distance",
        }:
            raise ValueError("unsupported s1_success_mode")
        if self.process_reward is not None:
            from .stage_process_reward import StageProcessRewardConfig
            if not isinstance(self.process_reward, StageProcessRewardConfig):
                raise ValueError("process_reward must be a StageProcessRewardConfig or None")
        for field in fields(self):
            if field.name in ("stage_schema_version", "calibration_status", "process_reward",
                              "s0_success_mode", "s1_success_mode", "enable_later_stage_context", "stage_preparation"):
                continue
            value = getattr(self, field.name)
            if value is None and field.name in {"s0_home_joint_tolerance_rad", "s0_home_velocity_tolerance_rad_s"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            allow_zero = (field.name == "s1_reach_dz_min_m" or
                          (field.name == "s1_reach_hold_s" and self.s1_success_mode == "lift_then_distance"))
            if value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f"{field.name} must be positive (dz minimum may be zero)")
        if self.s1_success_mode == "lift_then_distance":
            if self.process_reward is None or self.process_reward.schema_version != 4:
                raise ValueError("lift_then_distance requires schema 4 lift/approach process reward")
            if self.s1_reach_hold_s != 0:
                raise ValueError("lift_then_distance requires s1_reach_hold_s=0; unset or set S1_REACH_HOLD_S=0")
        if self.s0_home_hold_s > self.s0_budget_s:
            raise ValueError("S0 hold must fit within the S0 budget")
        if self.s1_reach_hold_s > self.s1_budget_s:
            raise ValueError("S1 hold must fit within the S1 budget")
        if self.s1_reach_xy_m > self.s1_reach_distance_m:
            raise ValueError("S1 XY tolerance must not exceed its 3D tolerance")
        if not self.s1_reach_dz_min_m <= self.s1_reach_dz_max_m <= self.s1_reach_distance_m:
            raise ValueError("S1 dz bounds must be ordered and within the 3D tolerance")
        if self.s1_opening_min_rad > self.s1_opening_max_rad:
            raise ValueError("S1 opening interval must be ordered")
        if self.s1_orientation_tolerance_rad > math.pi:
            raise ValueError("S1 orientation tolerance must not exceed pi")
        if self.process_reward is not None and self.process_reward.schema_version >= 3:
            if not math.isclose(self.process_reward.s1_distance_band_edges_m[-1],
                                self.s1_reach_distance_m, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("distance bands must end at the S1 reach distance tolerance")
        if self.process_reward is not None and self.process_reward.schema_version == 4:
            lift = self.process_reward.s1_lift_approach
            if lift.lift_hold_s + self.s1_reach_hold_s > self.s1_budget_s:
                raise ValueError("S1 lift and reach holds must fit within the single S1 budget")
            if not self.s1_reach_dz_min_m <= lift.approach_final_dz_m <= self.s1_reach_dz_max_m:
                raise ValueError("final approach dz must fit within the S1 reach dz interval")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> StageTrainingConfig:
        if not isinstance(values, Mapping):
            raise ValueError("stage configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown stage configuration fields: {sorted(unknown)}")
        values = dict(values)
        if values.get("process_reward") is not None:
            from .stage_process_reward import StageProcessRewardConfig
            values["process_reward"] = StageProcessRewardConfig.from_dict(values["process_reward"])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path | None = None) -> StageTrainingConfig:
        return cls.from_dict({} if path is None else json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        for name in ("s2_budget_s", "s3_budget_s", "s4_budget_s"):
            if values[name] == 10.0:
                values.pop(name)
        if not self.enable_later_stage_context:
            values.pop("enable_later_stage_context")
        if self.stage_preparation is None:
            values.pop("stage_preparation")
        if self.s0_success_mode == "window_end":
            values.pop("s0_success_mode")
        for name in ("s0_home_joint_tolerance_rad", "s0_home_velocity_tolerance_rad_s"):
            if values[name] is None:
                values.pop(name)
        if self.s1_success_mode == "qualified_reach":
            values.pop("s1_success_mode")
        if self.process_reward is None:
            # Keep historical stage configurations and resume hashes unchanged.
            values.pop("process_reward")
        else:
            values["process_reward"] = self.process_reward.to_dict()
        return json.loads(json.dumps(values, allow_nan=False))

    def stage_budget_s(self, stage: str) -> float:
        if stage not in {"S0", "S1", "S2", "S3", "S4", "S5"}:
            raise ValueError("training stage must be S0 through S5")
        # S5 delegates the complete HOME contract, including its deadline.
        return float(getattr(self, f"{('S0' if stage == 'S5' else stage).lower()}_budget_s"))

    @property
    def config_hash(self) -> str:
        value = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class StageEvaluator:
    """Evaluate one S0/S1 segment from fresh, aligned sensor snapshots.

    ``reset`` raises ValueError for an invalid initial state. Invalid sensor
    evidence during ``update`` also raises ValueError: callers must discard the
    affected segment rather than label missing measurements a policy failure.
    A long evidence gap resets dwell; repeated evidence never adds dwell.
    """

    def __init__(self, stage: str, config: StageTrainingConfig,
                 reward_config: PickPlaceRewardConfig):
        if stage not in ("S0", "S1"):
            raise ValueError("training stage must be S0 or S1")
        self.stage, self.config, self.reward_config = stage, config, reward_config
        self._s0_home = S0HomeSuccess(config, reward_config) if stage == "S0" else None
        self._s1_lift_approach = None
        if stage == "S1" and config.process_reward is not None and config.process_reward.schema_version == 4:
            from .s1_lift_approach import S1LiftApproachTracker
            self._s1_lift_approach = S1LiftApproachTracker(config.process_reward.s1_lift_approach, reward_config)
        self._initialized = False

    @property
    def budget_s(self) -> float:
        return self.config.s0_budget_s if self.stage == "S0" else self.config.s1_budget_s

    def _validate_evidence(self, snapshot: TaskSnapshot) -> None:
        m = snapshot.measurements
        required = (m.evidence_valid, m.joint_measurement_valid,
                    m.target_measurement_valid, m.contact_valid)
        if not all(required) or (m.target_name is not None and not m.target_motion_valid):
            raise ValueError("invalid stage sensor evidence; discard the segment")
        if m.sim_time_s - m.evidence_time > self.reward_config.max_evidence_gap_s + 1e-9:
            raise ValueError("stale stage sensor evidence; discard the segment")
        if m.grasp_evidence_mode != "contact" or self.reward_config.grasp_evidence_mode != "contact":
            raise ValueError("training stages require measured contact evidence, not proxy mode")

    def _metrics(self, snapshot: TaskSnapshot) -> dict[str, object]:
        m, cfg = snapshot.measurements, self.reward_config
        home_error = float(np.max(np.abs(np.asarray(m.joint_positions_rad) - cfg.home_joint_positions_rad)))
        velocity = float(np.max(np.abs(m.joint_velocities_rad_s)))
        joint_tolerance = cfg.home_joint_tolerance_rad
        velocity_tolerance = cfg.home_velocity_tolerance_rad_s
        if self._s0_home is not None:
            measured_home = self._s0_home.measure(snapshot)
            home_error = measured_home.home_error_rad
            velocity = measured_home.max_joint_velocity_rad_s
            joint_tolerance = self._s0_home.joint_tolerance_rad
            velocity_tolerance = self._s0_home.velocity_tolerance_rad_s
            at_home, robot_static = measured_home.at_home, measured_home.robot_static
        else:
            at_home = home_error <= joint_tolerance + 1e-9
            robot_static = velocity <= velocity_tolerance + 1e-9
        result: dict[str, object] = {
            "stage_home_error_rad": home_error,
            "stage_max_joint_velocity_rad_s": velocity,
            "stage_robot_at_home": at_home, "stage_robot_static": robot_static,
            "stage_opening_rad": float(m.joint_positions_rad[-1] - cfg.home_joint_positions_rad[-1]),
            "stage_target_present": m.target_name is not None,
            "stage_reach_distance_m": None, "stage_reach_xy_m": None,
            "stage_reach_dz_m": None, "stage_orientation_error_rad": None,
            "stage_bearing_error_rad": None, "stage_axis_error_rad": None,
            "stage_target_source_distance_m": None, "stage_target_at_source": False,
            "stage_target_source_supported": False, "stage_target_static": False,
            "stage_target_in_goal": False, "stage_target_bottom_z_m": None,
        }
        if self.stage == "S0":
            result.update(stage_s0_home_joint_tolerance_rad=joint_tolerance,
                          stage_s0_home_velocity_tolerance_rad_s=velocity_tolerance,
                          stage_s0_home_hold_required_s=self._s0_home.hold_required_s,
                          stage_s0_success_mode=self.config.s0_success_mode,
                          stage_s0_home_velocity_required=self._s0_home.velocity_required,
                          stage_home_hold_candidate=self._home_hold_candidate(result))
        if m.target_name is None:
            return result
        delta = np.asarray(m.grasp_position_world_m) - m.target_position_world_m
        axis = np.asarray(m.grasp_rotation_world)[:, 2]
        axis = axis / np.linalg.norm(axis)
        bearing = -delta + cfg.approach_reference_offset_m * axis
        length = float(np.linalg.norm(bearing))
        bearing_error = math.pi if length < 1e-9 else math.acos(float(np.clip(np.dot(axis, bearing / length), -1, 1)))
        axis_error = math.acos(float(np.clip(np.dot(axis, cfg.desired_grasp_axis_world), -1, 1)))
        inside, bottom, _ = object_geometry(m, cfg.goal_margin_m)
        source = snapshot.source_position_xy_m
        source_distance = None
        if source is not None:
            value = np.asarray(source, dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("invalid source position in stage snapshot")
            source_distance = float(np.linalg.norm(np.asarray(m.target_position_world_m[:2]) - value))
        result.update(
            stage_reach_distance_m=float(np.linalg.norm(delta)),
            stage_reach_xy_m=float(np.linalg.norm(delta[:2])), stage_reach_dz_m=float(delta[2]),
            stage_orientation_error_rad=max(bearing_error, axis_error),
            stage_bearing_error_rad=bearing_error, stage_axis_error_rad=axis_error,
            stage_target_source_distance_m=source_distance,
            stage_target_at_source=source_distance is not None and source_distance <= cfg.source_adjustment_tolerance_m + 1e-9,
            stage_target_source_supported=abs(bottom - m.source_support_z_m) <= cfg.support_height_tolerance_m + 1e-9,
            stage_target_static=bool(np.linalg.norm(m.target_linear_velocity_m_s) <= cfg.object_linear_velocity_tolerance_m_s + 1e-9
                                     and np.linalg.norm(m.target_angular_velocity_rad_s) <= cfg.object_angular_velocity_tolerance_rad_s + 1e-9),
            stage_target_in_goal=inside, stage_target_bottom_z_m=bottom,
        )
        return result

    def _home_hold_candidate(self, metrics: Mapping[str, object]) -> bool:
        if self._s0_home is not None:
            return self._s0_home.candidate(bool(metrics["stage_robot_at_home"]),
                                           bool(metrics["stage_robot_static"]))
        return bool(metrics["stage_robot_at_home"] and metrics["stage_robot_static"])

    def _lift_approach_ready(self, metrics: Mapping[str, object]) -> bool:
        if self._s1_lift_approach is None:
            return True
        sequence = self._s1_lift_approach.config
        return bool(metrics["stage_s1_lift_completed"] and metrics["stage_s1_corridor_safe"]
                    and (not sequence.wrist_rotation_reward_enabled
                         or metrics["stage_s1_roll_error_rad"] <= sequence.roll_tolerance_rad + 1e-9))

    def _reach_candidate(self, snapshot: TaskSnapshot, metrics: Mapping[str, object]) -> bool:
        c, m = self.config, snapshot.measurements
        if c.s1_success_mode == "lift_then_distance":
            # Position is measured at the grasp reference, not the EE origin.
            # Posture, opening and approach corridor remain reward guidance.
            return bool(m.target_name is not None and metrics["stage_s1_lift_completed"]
                        and metrics["stage_reach_distance_m"] < c.s1_reach_distance_m)
        if m.target_name is None or not self._lift_approach_ready(metrics):
            return False
        return bool(
            metrics["stage_reach_distance_m"] <= c.s1_reach_distance_m + 1e-9
            and metrics["stage_reach_xy_m"] <= c.s1_reach_xy_m + 1e-9
            and c.s1_reach_dz_min_m - 1e-9 <= metrics["stage_reach_dz_m"] <= c.s1_reach_dz_max_m + 1e-9
            and metrics["stage_orientation_error_rad"] <= c.s1_orientation_tolerance_rad + 1e-9
            and c.s1_opening_min_rad - 1e-9 <= metrics["stage_opening_rad"] <= c.s1_opening_max_rad + 1e-9
            and metrics["stage_target_at_source"] and metrics["stage_target_source_supported"]
            and metrics["stage_target_static"] and not metrics["stage_target_in_goal"]
            and not (m.left_contact or m.right_contact or snapshot.grasp_valid
                     or snapshot.grasp_attempt_active or snapshot.lifted_once)
        )

    def _strong_downstream(self, snapshot: TaskSnapshot, metrics: Mapping[str, object]) -> bool:
        """Current physical grasp/lift can exceed the open-jaw reach milestone."""
        if self.config.s1_success_mode == "lift_then_distance":
            # This profile has exactly one success path: lifted, then <2 cm.
            return False
        m, c = snapshot.measurements, self.reward_config
        bottom = metrics["stage_target_bottom_z_m"]
        return bool(
            self._lift_approach_ready(metrics)
            and bottom is not None and snapshot.grasp_valid and snapshot.grasp_attempt_active
            and snapshot.attempt_lifted and m.left_contact and m.right_contact and m.contacts_simultaneous
            and snapshot.contact_duration_s + 1e-9 >= c.grasp_confirm_s
            and snapshot.lift_duration_s + 1e-9 >= c.lift_confirm_s
            and metrics["stage_reach_distance_m"] <= c.grasp_candidate_distance_m + 1e-9
            and bottom - m.source_support_z_m + 1e-9 >= c.lift_clearance_m
            and bottom - snapshot.grasp_origin_bottom_z_m + 1e-9 >= c.lift_clearance_m
        )

    def _record_measured_sample(self, snapshot: TaskSnapshot, metrics: Mapping[str, object]) -> None:
        """Sample RMS uses distinct evidence stamps, never callback repeats.

        Maxima also include valid endpoints after a gap. These are actual
        measurements, but the missing interval contributes no stable time.
        """
        m = snapshot.measurements
        error = np.asarray(m.joint_positions_rad) - self.reward_config.home_joint_positions_rad
        velocity = np.asarray(m.joint_velocities_rad_s)
        self._sample_count += 1
        self._home_error_square_sum += error * error
        self._velocity_square_sum += velocity * velocity
        self._home_error_max = np.maximum(self._home_error_max, np.abs(error))
        self._velocity_max = np.maximum(self._velocity_max, np.abs(velocity))
        for name, attribute in (("stage_reach_distance_m", "_min_reach_distance"),
                                ("stage_reach_xy_m", "_min_reach_xy")):
            value = metrics[name]
            if value is not None:
                previous = getattr(self, attribute)
                setattr(self, attribute, float(value) if previous is None else min(previous, float(value)))

    def _window_statistics(self, elapsed_s: float) -> dict[str, object]:
        return {
            "stage_measurement_sample_count": self._sample_count,
            "stage_home_error_max_per_joint_rad": self._home_error_max.tolist(),
            "stage_home_error_sample_rms_per_joint_rad": np.sqrt(self._home_error_square_sum / self._sample_count).tolist(),
            "stage_joint_velocity_max_per_joint_rad_s": self._velocity_max.tolist(),
            "stage_joint_velocity_sample_rms_per_joint_rad_s": np.sqrt(self._velocity_square_sum / self._sample_count).tolist(),
            "stage_home_static_evidence_s": self._home_static_evidence_s,
            "stage_home_static_fraction": self._home_static_evidence_s / elapsed_s if elapsed_s > 0 else 0.0,
            "stage_fresh_evidence_s": self._fresh_evidence_s,
            "stage_evidence_coverage_fraction": self._fresh_evidence_s / elapsed_s if elapsed_s > 0 else 0.0,
            "stage_min_reach_distance_m": self._min_reach_distance,
            "stage_min_reach_xy_m": self._min_reach_xy,
            "stage_reach_candidate_evidence_s": self._reach_candidate_evidence_s,
        }

    def reset(self, snapshot: TaskSnapshot) -> dict[str, object]:
        self._initialized = False
        self._validate_evidence(snapshot)
        metrics = self._metrics(snapshot)
        m = snapshot.measurements
        checks = {
            "evidence_valid": True,
            # A relaxed stage success threshold must never relax the prepared
            # initial-state certificate or the S1 starting-state contract.
            "home_position_valid": metrics["stage_home_error_rad"] <= self.reward_config.home_joint_tolerance_rad + 1e-9,
            "home_velocity_valid": metrics["stage_max_joint_velocity_rad_s"] <= self.reward_config.home_velocity_tolerance_rad_s + 1e-9,
            "task_running": snapshot.outcome == "running" and not snapshot.failure_reason and not m.failure_reason,
            "no_prior_grasp": not (snapshot.grasp_valid or snapshot.grasp_attempt_active
                                   or snapshot.lifted_once or snapshot.attempt_lifted
                                   or snapshot.carried_into_goal or snapshot.valid_release
                                   or snapshot.delivery_valid or snapshot.placed_stable
                                   or snapshot.grasp_attempt_id or m.left_contact or m.right_contact),
        }
        if self.stage == "S0":
            checks["target_absent"] = m.target_name is None
            checks["task_mode_valid"] = snapshot.task_mode in ("idle_empty", "pick_place")
        else:
            checks.update(
                target_present=m.target_name is not None,
                task_mode_valid=snapshot.task_mode == "pick_place",
                target_at_source=bool(metrics["stage_target_at_source"]),
                target_source_supported=bool(metrics["stage_target_source_supported"]),
                target_static=bool(metrics["stage_target_static"]),
                target_outside_goal=not metrics["stage_target_in_goal"],
            )
        if not all(checks.values()):
            failed = ", ".join(name for name, valid in checks.items() if not valid)
            raise ValueError(f"invalid {self.stage} initial state: {failed}")
        if self._s1_lift_approach is not None:
            metrics.update(self._s1_lift_approach.reset(snapshot, metrics))
        self._start_certificate = {"start_valid": True, "training_stage": self.stage,
                                   "sim_time_s": m.sim_time_s, "evidence_sim_time_s": m.evidence_time,
                                   "checks": checks}
        self._start_time = self._last_sim_time = m.sim_time_s
        self._last_evidence_time = m.evidence_time
        self._instance, self._target = snapshot.task_instance_id, m.target_name
        self._home_hold = self._reach_hold = 0.0
        self._last_home = bool(metrics["stage_robot_at_home"] and metrics["stage_robot_static"])
        self._last_home_hold_candidate = self._home_hold_candidate(metrics)
        self._last_reach = self._reach_candidate(snapshot, metrics)
        self._sample_count = 0
        self._home_error_square_sum = np.zeros(6)
        self._velocity_square_sum = np.zeros(6)
        self._home_error_max = np.zeros(6)
        self._velocity_max = np.zeros(6)
        self._home_static_evidence_s = self._fresh_evidence_s = self._reach_candidate_evidence_s = 0.0
        self._min_reach_distance = self._min_reach_xy = None
        self._record_measured_sample(snapshot, metrics)
        self._result, self._reason = "RUNNING", None
        if self._s0_home is not None:
            self._s0_home.reset(snapshot)
        self._initialized = True
        return self._info(snapshot, metrics, fresh=False, downstream=False)

    def update(self, snapshot: TaskSnapshot) -> dict[str, object]:
        if not self._initialized:
            raise RuntimeError("reset must be called before stage update")
        if self._result != "RUNNING":
            raise RuntimeError("stage segment is closed; reset before another update")
        self._validate_evidence(snapshot)
        m = snapshot.measurements
        if snapshot.task_instance_id != self._instance or m.target_name != self._target:
            raise ValueError("stage task instance or target identity changed; discard the segment")
        if m.sim_time_s <= self._last_sim_time:
            raise ValueError("stage simulation time must increase")
        dt = m.evidence_time - self._last_evidence_time
        if dt < -1e-9:
            raise ValueError("stage evidence time moved backwards")
        metrics = self._metrics(snapshot)
        fresh = 0 < dt <= self.reward_config.max_evidence_gap_s + 1e-9
        # Never credit reused evidence time or pre-stage preparation time.
        covered_s = (min(dt, m.sim_time_s - self._last_sim_time,
                         max(0.0, m.evidence_time - self._start_time)) if fresh else 0.0)
        if self._s1_lift_approach is not None:
            metrics.update(self._s1_lift_approach.update(
                snapshot, metrics, fresh=fresh, covered_s=covered_s,
                evidence_gap=dt > self.reward_config.max_evidence_gap_s + 1e-9,
            ))
        home = bool(metrics["stage_robot_at_home"] and metrics["stage_robot_static"])
        home_hold_candidate = self._home_hold_candidate(metrics)
        reach = self._reach_candidate(snapshot, metrics)
        if dt > 0:
            self._record_measured_sample(snapshot, metrics)
        if fresh:
            # Cap by this control interval: a later fresh stamp cannot credit
            # time already logged as reused evidence. Do not count preparation
            # evidence before the stage clock starts either.
            self._fresh_evidence_s += covered_s
            if home and self._last_home:
                self._home_static_evidence_s += covered_s
            if reach and self._last_reach:
                self._reach_candidate_evidence_s += covered_s
        if dt > self.reward_config.max_evidence_gap_s + 1e-9:
            self._home_hold = self._reach_hold = 0.0
        elif fresh:
            if self._s0_home is None:
                self._home_hold = (self._home_hold + dt
                                   if home_hold_candidate and self._last_home_hold_candidate else 0.0)
            self._reach_hold = self._reach_hold + dt if reach and self._last_reach else 0.0
        # Changed conditions can invalidate a dwell even without a new stamp.
        if not home_hold_candidate:
            self._home_hold = 0.0
        if not reach:
            self._reach_hold = 0.0
        if self._s0_home is not None:
            self._s0_home.update(snapshot)
            self._home_hold = self._s0_home.hold_s
        downstream = bool(fresh and self._strong_downstream(snapshot, metrics))
        elapsed = m.sim_time_s - self._start_time
        if snapshot.outcome == "task_failure" or snapshot.failure_reason or m.failure_reason:
            self._result = "FAILURE"
            self._reason = snapshot.failure_reason or m.failure_reason or "task_failure"
        elif self.stage == "S0":
            self._result, self._reason = self._s0_home.decision()
        elif self.stage == "S1":
            # A measurement after the budget cannot retroactively prove that
            # the milestone was reached before the collection deadline.
            within_budget = elapsed <= self.budget_s + 1e-9
            distance_only = self.config.s1_success_mode == "lift_then_distance"
            ready = (reach if distance_only else
                     downstream or (reach and self._reach_hold + 1e-9 >= self.config.s1_reach_hold_s))
            if within_budget and fresh and ready:
                self._result = "SUCCESS"
                self._reason = ("reach_distance" if distance_only else
                                "grasp_lift" if downstream else "reach_hold")
            elif elapsed + 1e-9 >= self.budget_s:
                self._result, self._reason = "TIMEOUT", "reach_not_completed"
        self._last_sim_time, self._last_evidence_time = m.sim_time_s, m.evidence_time
        self._last_home, self._last_reach = home, reach
        self._last_home_hold_candidate = home_hold_candidate
        return self._info(snapshot, metrics, fresh=fresh, downstream=downstream)

    def _info(self, snapshot: TaskSnapshot, metrics: Mapping[str, object], *,
              fresh: bool, downstream: bool) -> dict[str, object]:
        all_metrics = {**metrics, **self._window_statistics(snapshot.sim_time_s - self._start_time)}
        return {
            "training_stage": self.stage,
            "stage_result": self._result,
            "stage_success": self._result == "SUCCESS",
            "stage_timeout": self._result == "TIMEOUT",
            "stage_cut": self._result in ("SUCCESS", "TIMEOUT"),
            "stage_result_reason": self._reason,
            "stage_elapsed_s": snapshot.sim_time_s - self._start_time,
            "stage_budget_s": self.budget_s,
            "start_valid": True,
            "start_certificate": {**self._start_certificate, "checks": dict(self._start_certificate["checks"])},
            "assisted_initialization": False,
            "full_unassisted_success": False,
            "stage_home_hold_s": self._home_hold,
            "stage_reach_hold_s": self._reach_hold,
            "stage_reach_candidate": self._reach_candidate(snapshot, metrics),
            "stage_downstream_grasp_lift": downstream,
            "stage_evidence_fresh": fresh,
            "stage_metrics": all_metrics,
            **all_metrics,
        }
