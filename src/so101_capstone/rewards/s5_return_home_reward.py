"""S5 reward for leaving a placed object undisturbed.

The physical task tracker must retain the real grasp/carry/release provenance.
An idle_completed scene is not a substitute for completing the pick/place
task. This file provides measurement-only reward signals; it neither moves
the robot nor prepares its start; later_stage_training handles the stage cut.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace

import numpy as np

from .later_stage_reward import (
    LaterRewardTuning,
    LaterStageReward,
    EpisodeRewardLedger,
    LaterStageRewardResult,
    Milestone,
    StageSignals,
    placement_valid,
    quality,
    unit,
    validate_numbers,
)
from .pick_place_reward import PickPlaceRewardConfig, PickPlaceRewardInput
from .s0_home_success import S0HomeSuccess
from .task_state import TaskSnapshot, object_geometry
from .training_stages import StageTrainingConfig


@dataclass(frozen=True)
class S5ReturnHomeRewardConfig:
    position_scale_rad: float = 0.250
    velocity_scale_rad_s: float = 0.100
    release_separation_m: float = 0.050
    withdrawn_bonus: float = 3.0
    home_position_bonus: float = 5.0
    home_static_bonus: float = 3.0
    home_held_bonus: float = 5.0

    def __post_init__(self) -> None:
        validate_numbers(self)


class S5ReturnHomeReward(LaterStageReward):
    """Valid placement plus S0's shared HOME success judgment.

    All HOME guidance is gated by an undisturbed, certified placement. Changes
    to any joint can earn small progress even if a different joint remains the
    worst error. Velocity guidance is weighted by that joint's position
    quality. The caller must supply the same resolved StageTrainingConfig used
    for S0, including any command-line overrides. HOME tolerances, velocity
    requirement, dwell, mode and budget come only from S0HomeSuccess. There is
    no S5 copy of the HOME rules or second timer. Placement loss clears the
    shared judge's dwell. Stage cuts belong to the training adapter.
    """

    stage = "S5"

    def _signals_without_new_evidence(self, measured: StageSignals) -> StageSignals:
        # Match S0 position_hold: a changed out-of-bounds joint or disturbed
        # placement cancels dwell even when the evidence timestamp repeats.
        # Keep the old potential/qualities: no progress or completion may be
        # credited until new evidence arrives (enforced by the shared step).
        return replace(self._signals, goals=measured.goals,
                       phase=measured.phase, metrics=measured.metrics)

    def __init__(self, reward_config: PickPlaceRewardConfig,
                 config: S5ReturnHomeRewardConfig | None = None,
                 tuning: LaterRewardTuning | None = None, *,
                 stage_config: StageTrainingConfig):
        if not isinstance(stage_config, StageTrainingConfig):
            raise TypeError("S5 requires the resolved S0 StageTrainingConfig")
        self.config = config or S5ReturnHomeRewardConfig()
        self.home_success = S0HomeSuccess(stage_config, reward_config)
        c = self.config
        weights = {f"joint_{i + 1}_position": 1.0 for i in range(6)}
        weights.update({f"joint_{i + 1}_velocity": 0.4 for i in range(6)})
        weights.update(withdrawal=0.7, home_readiness=1.8)
        super().__init__(
            reward_config,
            weights=weights,
            milestones=(
                Milestone("withdrawn", c.withdrawn_bonus),
                Milestone("home_position", c.home_position_bonus),
                Milestone("home_static", c.home_static_bonus),
                Milestone("home_held", c.home_held_bonus, self.home_success.hold_required_s),
            ),
            completion_goal="home_held",
            tuning=tuning,
        )

    def reset(self, snapshot: TaskSnapshot, *, ledger: EpisodeRewardLedger | None = None) -> None:
        previous = self.home_success
        self.home_success = copy(previous)
        try:
            self._validate_snapshot(snapshot)
            self.home_success.reset(snapshot, eligible=placement_valid(
                snapshot, self.reward_config, separation_m=self.config.release_separation_m))
            super().reset(snapshot, ledger=ledger)
        except Exception:
            self.home_success = previous
            raise

    def step(self, transition: PickPlaceRewardInput) -> LaterStageRewardResult:
        # Advance the sole HOME judge transactionally: invalid transitions
        # must not consume hold time or disturb subsequent valid feedback.
        previous = self.home_success
        self.home_success = copy(previous)
        try:
            self._validate_snapshot(transition.next_snapshot)
            eligible = placement_valid(transition.next_snapshot, self.reward_config,
                                       separation_m=self.config.release_separation_m)
            eligible = eligible and (transition.prev_snapshot.grasp_attempt_id
                                     == transition.next_snapshot.grasp_attempt_id)
            self.home_success.update(transition.next_snapshot, eligible=eligible)
            return super().step(transition)
        except Exception:
            self.home_success = previous
            raise

    def _milestone_progress(self, name: str, milestone: Milestone, current: StageSignals,
                            **timing) -> tuple[float, bool]:
        if name == "home_held":
            # Use S0's complete decision, including window/deadline semantics.
            # The base class only pays the associated dwell and completion.
            return self.home_success.hold_s, bool(
                current.goals[name] and self.home_success.decision()[0] == "SUCCESS")
        return super()._milestone_progress(name, milestone, current, **timing)

    def reward_state(self) -> dict[str, object]:
        state = super().reward_state()
        state["home_success"] = self.home_success.state_dict()
        return state

    def signals(self, s: TaskSnapshot) -> StageSignals:
        c, r, m = self.config, self.reward_config, s.measurements
        _, bottom, _ = object_geometry(m, r.goal_margin_m)
        placed = placement_valid(s, r, separation_m=0.0)
        withdrawn = placement_valid(s, r, separation_m=c.release_separation_m)
        joints = np.asarray(m.joint_positions_rad)
        home = np.asarray(r.home_joint_positions_rad)
        errors = np.abs(joints - home)
        speeds = np.abs(np.asarray(m.joint_velocities_rad_s))
        home_measurement = self.home_success.measure(s)
        at_home = bool(withdrawn and home_measurement.at_home)
        quiet = bool(at_home and home_measurement.robot_static)
        home_candidate = bool(withdrawn and home_measurement.candidate)
        qualities: dict[str, float] = {}
        metrics: dict[str, float] = {}
        for index, (error, speed) in enumerate(zip(errors, speeds), start=1):
            position = (quality(max(float(error) - self.home_success.joint_tolerance_rad, 0.0),
                                c.position_scale_rad) if withdrawn else 0.0)
            velocity = (position * quality(max(float(speed) - self.home_success.velocity_tolerance_rad_s, 0.0),
                                           c.velocity_scale_rad_s) if withdrawn else 0.0)
            qualities[f"joint_{index}_position"] = position
            qualities[f"joint_{index}_velocity"] = velocity
            metrics[f"joint_{index}_home_error_rad"] = float(error)
            metrics[f"joint_{index}_speed_rad_s"] = float(speed)
        qualities["withdrawal"] = (unit(s.target_distance_m / c.release_separation_m)
                                   if placed else 0.0)
        # Position readiness is stateless. Actual continuous hold progress is
        # paid from the shared S0 judge via the home_held milestone.
        qualities["home_readiness"] = min(qualities[f"joint_{i}_position"] for i in range(1, 7))
        metrics.update(home_max_error_rad=float(np.max(errors)),
                       home_mean_error_rad=float(np.mean(errors)),
                       max_joint_speed_rad_s=float(np.max(speeds)),
                       home_position_tolerance_rad=self.home_success.joint_tolerance_rad,
                       home_hold_required_s=self.home_success.hold_required_s,
                       home_hold_candidate=float(home_candidate),
                       home_velocity_required=float(self.home_success.velocity_required),
                       home_budget_s=self.home_success.config.s0_budget_s,
                       object_bottom_clearance_m=bottom - m.goal_support_z_m,
                       withdrawal_distance_m=s.target_distance_m,
                       placement_valid=float(placed))
        phase = ("HOME_HOLD" if home_candidate else "SETTLE_HOME" if at_home else "RETURN_HOME" if withdrawn else
                 "WITHDRAW" if placed else "PLACEMENT_RECOVERY_REQUIRED")
        return StageSignals(
            phase=phase,
            qualities=qualities,
            goals={"withdrawn": withdrawn, "home_position": at_home,
                   "home_static": quiet, "home_held": home_candidate},
            metrics=metrics,
        )
