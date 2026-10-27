"""S0's reusable measured HOME contract, independent of stage termination.

S5 may require valid placement as an additional eligibility condition, but it
must use the same resolved S0 configuration, position comparisons and dwell
clock. This helper does not command joints, reward behaviour or end episodes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .pick_place_reward import PickPlaceRewardConfig
from .task_state import TaskSnapshot

if TYPE_CHECKING:
    from .training_stages import StageTrainingConfig


@dataclass(frozen=True)
class S0HomeMeasurement:
    home_error_rad: float
    max_joint_velocity_rad_s: float
    at_home: bool
    robot_static: bool
    candidate: bool


class S0HomeSuccess:
    """Share S0's HOME success predicate and continuous evidence timer.

    ``eligible=False`` breaks continuity immediately, including on repeated
    evidence. A later eligible sample starts a new interval; elapsed time
    while ineligible is never credited. ``decision`` shares S0's mode and
    deadline semantics, but never closes the caller's episode or latches a
    result; physical failure and stage termination remain with the caller.
    """

    def __init__(self, config: StageTrainingConfig,
                 reward_config: PickPlaceRewardConfig):
        self.config = config
        self.reward_config = reward_config
        self._initialized = False
        self.hold_s = 0.0
        self.fresh = False
        self.measurement: S0HomeMeasurement | None = None
        self._eligible = False

    @property
    def joint_tolerance_rad(self) -> float:
        value = self.config.s0_home_joint_tolerance_rad
        return self.reward_config.home_joint_tolerance_rad if value is None else value

    @property
    def velocity_tolerance_rad_s(self) -> float:
        value = self.config.s0_home_velocity_tolerance_rad_s
        return self.reward_config.home_velocity_tolerance_rad_s if value is None else value

    @property
    def hold_required_s(self) -> float:
        return self.config.s0_home_hold_s

    @property
    def velocity_required(self) -> bool:
        return self.config.s0_success_mode != "position_hold"

    def candidate(self, at_home: bool, robot_static: bool) -> bool:
        return bool(at_home and (robot_static or not self.velocity_required))

    def measure(self, snapshot: TaskSnapshot) -> S0HomeMeasurement:
        m = snapshot.measurements
        joints = np.asarray(m.joint_positions_rad)
        home = np.asarray(self.reward_config.home_joint_positions_rad)
        error = float(np.max(np.abs(joints - home)))
        velocity = float(np.max(np.abs(m.joint_velocities_rad_s)))
        if self.config.s0_success_mode == "position_hold":
            # Comparing open intervals directly keeps an exact +/- boundary
            # outside despite roundoff when subtracting the HOME position.
            at_home = bool(np.all((joints > home - self.joint_tolerance_rad)
                                  & (joints < home + self.joint_tolerance_rad)))
        else:
            at_home = error <= self.joint_tolerance_rad + 1e-9
        robot_static = velocity <= self.velocity_tolerance_rad_s + 1e-9
        return S0HomeMeasurement(error, velocity, at_home, robot_static,
                                 self.candidate(at_home, robot_static))

    def _validate_evidence(self, snapshot: TaskSnapshot) -> None:
        m = snapshot.measurements
        if not m.evidence_valid or not m.joint_measurement_valid:
            raise ValueError("invalid S0 HOME sensor evidence; discard the segment")
        if m.sim_time_s - m.evidence_time > self.reward_config.max_evidence_gap_s + 1e-9:
            raise ValueError("stale S0 HOME sensor evidence; discard the segment")

    def reset(self, snapshot: TaskSnapshot, *, eligible: bool = True) -> S0HomeMeasurement:
        self._validate_evidence(snapshot)
        measured = self.measure(snapshot)
        m = snapshot.measurements
        self._start_time = self._last_sim_time = m.sim_time_s
        self._last_evidence_time = m.evidence_time
        self._instance, self._target = snapshot.task_instance_id, m.target_name
        self._eligible = bool(eligible)
        self._last_candidate = measured.candidate and self._eligible
        self.hold_s, self.fresh = 0.0, False
        self.measurement = measured
        self._initialized = True
        return measured

    def update(self, snapshot: TaskSnapshot, *, eligible: bool = True) -> S0HomeMeasurement:
        if not self._initialized:
            raise RuntimeError("reset must be called before S0 HOME update")
        self._validate_evidence(snapshot)
        m = snapshot.measurements
        if snapshot.task_instance_id != self._instance or m.target_name != self._target:
            raise ValueError("S0 HOME task instance or target identity changed; discard the segment")
        if m.sim_time_s <= self._last_sim_time:
            raise ValueError("S0 HOME simulation time must increase")
        dt = m.evidence_time - self._last_evidence_time
        if dt < -1e-9:
            raise ValueError("S0 HOME evidence time moved backwards")
        measured = self.measure(snapshot)
        fresh = 0 < dt <= self.reward_config.max_evidence_gap_s + 1e-9
        candidate = measured.candidate and bool(eligible)
        if dt > self.reward_config.max_evidence_gap_s + 1e-9:
            self.hold_s = 0.0
        elif fresh:
            # Preserve legacy window_end timing. Opt-in early modes count
            # only this measured control interval, excluding preparation and
            # time already consumed by callbacks with reused evidence.
            home_dt = (min(dt, m.sim_time_s - self._last_sim_time,
                           max(0.0, m.evidence_time - self._start_time))
                       if self.config.s0_success_mode in {"home_hold", "position_hold"}
                       else dt)
            self.hold_s = self.hold_s + home_dt if candidate and self._last_candidate else 0.0
        # Invalid conditions reset immediately even if the stamp repeats.
        if not candidate:
            self.hold_s = 0.0
        self.measurement, self.fresh = measured, fresh
        self._eligible = bool(eligible)
        self._last_candidate = candidate
        self._last_sim_time, self._last_evidence_time = m.sim_time_s, m.evidence_time
        return measured

    @property
    def ready(self) -> bool:
        return bool(self._initialized and self.fresh and self._eligible
                    and self.measurement is not None and self.measurement.candidate
                    and self.hold_s + 1e-9 >= self.hold_required_s)

    @property
    def elapsed_s(self) -> float:
        return self._last_sim_time - self._start_time if self._initialized else 0.0

    def decision(self) -> tuple[str, str | None]:
        """Use the exact S0 mode/deadline contract without ending a stage."""
        if not self._initialized:
            return "RUNNING", None
        mode, budget = self.config.s0_success_mode, self.config.s0_budget_s
        if mode in {"home_hold", "position_hold"}:
            if self.elapsed_s <= budget + 1e-9 and self.ready:
                return "SUCCESS", mode
            if self.elapsed_s + 1e-9 >= budget:
                return "TIMEOUT", f"{mode}_not_completed"
        elif self.elapsed_s + 1e-9 >= budget:
            return (("SUCCESS", "home_hold") if self.ready else
                    ("TIMEOUT", "home_hold_not_completed"))
        return "RUNNING", None

    def state_dict(self) -> dict[str, object]:
        """Expose all mutable clock state for reward/critic diagnostics."""
        return {
            "initialized": self._initialized,
            "start_time": getattr(self, "_start_time", None),
            "last_sim_time": getattr(self, "_last_sim_time", None),
            "last_evidence_time": getattr(self, "_last_evidence_time", None),
            "task_instance_id": getattr(self, "_instance", None),
            "target_name": getattr(self, "_target", None),
            "last_candidate": getattr(self, "_last_candidate", False),
            "eligible": self._eligible,
            "hold_s": self.hold_s,
            "fresh": self.fresh,
            "ready": self.ready,
            "elapsed_s": self.elapsed_s,
            "decision": self.decision(),
            "measurement": asdict(self.measurement) if self.measurement is not None else None,
        }
