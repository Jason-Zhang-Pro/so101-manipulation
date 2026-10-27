"""S2--S5 reward primitives used by the later_stage_training adapter.

These rewards consume existing contact-certified TaskSnapshot objects. They
never command/reset a robot, create a world, change task outcomes or decide
terminated/truncated. The stage adapter owns those operations.
Select ONE active stage reward per transition; do not add its reward to the
full-task PickPlaceReward or score all four stages on the same transition.

Keep one EpisodeRewardLedger for the entire physical episode, including stage
handoffs and recovery visits. Resetting that ledger after a drop would allow
bonus farming. Preparing a later stage must preserve TaskStateTracker history.
Each world/new physical episode needs a NEW ledger: task_instance_id alone is
not globally unique across workers or rebuilt worlds. Pass the SAME ledger
explicitly for every revisit within that episode; reset never infers identity.
The adapter exports reward_state() to the critic: dwell, previous qualities,
records and paid milestones are part of the reward state, not actor inputs.
All new numeric settings are calibration candidates, not live-validated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Real
from typing import Mapping

import numpy as np

from .pick_place_reward import PickPlaceRewardConfig, PickPlaceRewardInput, PickPlaceRewardResult
from .task_state import TaskSnapshot, object_geometry


def unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def quality(error: float, scale: float) -> float:
    """Smooth guidance remains nonzero outside the final tolerance."""
    return float(scale / (scale + max(0.0, error)))


def validate_numbers(config: object, *, zero_allowed: tuple[str, ...] = ()) -> None:
    for name, value in asdict(config).items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
        if value < 0 or (value == 0 and name not in zero_allowed):
            raise ValueError(f"{name} must be positive")


def live_grasp(s: TaskSnapshot) -> bool:
    return bool(s.grasp_valid and s.grasp_attempt_active)


def live_carry(s: TaskSnapshot) -> bool:
    return live_grasp(s) and s.attempt_lifted


def placement_valid(s: TaskSnapshot, reward_config: PickPlaceRewardConfig, *,
                    separation_m: float) -> bool:
    """Current placement evidence, without the full-task dwell/distance gates.

    Legal grasp/carry/release provenance still comes from TaskStateTracker.
    Each stage owns its own continuous hold, so the old ``placed_stable``
    certificate (0.4 s at 6 cm by default) must not delay S4/S5 completion.
    """
    m = s.measurements
    inside, _, _ = object_geometry(m, reward_config.goal_margin_m)
    return bool(s.valid_release and s.delivery_valid and inside
                and s.target_supported and s.target_static
                and not (m.left_contact or m.right_contact)
                and not s.grasp_valid and not s.grasp_attempt_active
                and s.target_distance_m + 1e-9 >= separation_m)


def relative_slip_m(s: TaskSnapshot) -> float:
    if s.grasp_relative_anchor_m is None:
        return math.inf
    m = s.measurements
    relative = np.asarray(m.grasp_rotation_world).T @ (
        np.asarray(m.target_position_world_m) - m.grasp_position_world_m)
    return float(np.linalg.norm(relative - s.grasp_relative_anchor_m))


def object_speed(s: TaskSnapshot) -> tuple[float, float]:
    m = s.measurements
    return (float(np.linalg.norm(m.target_linear_velocity_m_s)),
            float(np.linalg.norm(m.target_angular_velocity_rad_s)))


def attachment_command_reward(
        s: TaskSnapshot, *, stage: str, margin_scale_rad: float,
        close_reward_per_s: float, open_cost_per_s: float,
        covered_s: float, fresh: bool, failed: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score the command against the frozen physical angle at attachment.

    This dense term is separate from PBRS: continuing to command a tighter
    grasp remains positive, while equality and opening are negative. It does
    not modify the command or certify a grasp. Only matching, current rigid
    attachment evidence can enable it; the caller pays fresh measured time.
    """
    m = s.measurements
    reference = m.grasp_attachment_measured_rad
    command = m.executed_command_joint_targets_rad
    valid = bool(m.grasp_attachment_enabled and m.grasp_attachment_active
                 and live_grasp(s)
                 and s.target_name == m.target_name == m.grasp_attachment_target_name
                 and m.grasp_attachment_attempt_id == s.grasp_attempt_id
                 and reference is not None and command is not None
                 and m.grasp_attachment_sim_time_s is not None
                 # This is the plugin STATUS stamp, not the attachment event
                 # stamp. Synchronized sensor evidence may precede it slightly.
                 and math.isclose(s.sim_time_s, m.grasp_attachment_sim_time_s,
                                  rel_tol=0.0, abs_tol=1e-6))
    command_angle = float(command[-1]) if command is not None else 0.0
    reference_angle = float(reference) if reference is not None else 0.0
    margin = reference_angle - command_angle
    rate = 0.0
    if valid:
        if margin > 0:
            rate = close_reward_per_s * unit(margin / margin_scale_rad)
        else:
            # Equality is explicitly penalized, without requiring an arbitrary
            # extra-close target angle. Further opening increases the cost.
            rate = -open_cost_per_s * (0.25 + 0.75 * unit(-margin / margin_scale_rad))
    paid_s = covered_s if valid and fresh and not failed else 0.0
    key = f"{stage.lower()}_attachment_gripper_command"
    return ({key: rate * paid_s}, dict(
        attachment_gripper_command_valid=float(valid),
        attachment_grasp_reference_rad=reference_angle,
        attachment_gripper_command_rad=command_angle,
        attachment_gripper_closing_margin_rad=margin if valid else 0.0,
        attachment_gripper_command_reward_rate=rate,
        attachment_gripper_command_reward_time_s=paid_s))


@dataclass(frozen=True)
class LaterRewardTuning:
    potential_max: float = 10.0
    record_bonus_total: float = 10.0
    hold_progress_bonus: float = 2.0
    quality_bands: int = 20
    time_cost_per_s: float = 0.1
    regression_cost_per_s: float = 0.2
    grasp_loss_cost: float = 1.0
    drop_cost: float = 5.0
    placement_loss_cost: float = 5.0
    failure_cost: float = 10.0
    stage_completion_bonus: float = 20.0

    def __post_init__(self) -> None:
        validate_numbers(self, zero_allowed=("record_bonus_total", "hold_progress_bonus", "time_cost_per_s",
                         "regression_cost_per_s", "grasp_loss_cost", "drop_cost",
                         "placement_loss_cost"))
        if not isinstance(self.quality_bands, int):
            raise ValueError("quality_bands must be an integer")


@dataclass(frozen=True)
class Milestone:
    name: str
    bonus: float
    hold_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or any(not math.isfinite(v) or v < 0 for v in (self.bonus, self.hold_s)):
            raise ValueError("invalid milestone")


@dataclass(frozen=True)
class StageSignals:
    phase: str
    qualities: Mapping[str, float]
    goals: Mapping[str, bool]
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass
class EpisodeRewardLedger:
    """One shared ledger per physical episode, not per grasp attempt or wave."""
    identity: tuple[str, str, str | None]
    paid: set[str] = field(default_factory=set)
    best: dict[str, float] = field(default_factory=dict)

    @classmethod
    def for_snapshot(cls, s: TaskSnapshot) -> EpisodeRewardLedger:
        return cls((s.task_instance_id, s.task_mode, s.target_name))


@dataclass(frozen=True)
class LaterStageRewardResult(PickPlaceRewardResult):
    stage: str
    subphase: str
    stage_complete: bool
    stage_success_event: bool
    completed_subgoals: tuple[str, ...]
    metrics: dict[str, float]

    def to_info(self) -> dict[str, object]:
        return dict(super().to_info(), reward_stage=self.stage,
                    reward_subphase=self.subphase, reward_stage_complete=self.stage_complete,
                    reward_stage_success_event=self.stage_success_event,
                    reward_completed_subgoals=list(self.completed_subgoals),
                    reward_stage_metrics=dict(self.metrics))


class LaterStageReward:
    """Small progress bands + signed potential + one-time certified goals.

    Subclasses supply geometry and milestone predicates only. Continuous
    first-visit credit is linear within each of 20 quality bands (no dead zone
    between thresholds). Signed potential and a bounded time-based best-record
    regression cost discourage retreat. Changing attempt never clears records.
    """
    stage: str

    def __init__(self, reward_config: PickPlaceRewardConfig, *,
                 weights: Mapping[str, float], milestones: tuple[Milestone, ...],
                 completion_goal: str, tuning: LaterRewardTuning | None = None):
        if reward_config.grasp_evidence_mode != "contact":
            raise ValueError("S2--S5 rewards require real contact evidence, not proxy grasp")
        self.reward_config = reward_config
        self.tuning = tuning or LaterRewardTuning()
        if not weights or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("quality weights must be positive")
        self.weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        self.milestones = {m.name: m for m in milestones}
        if len(self.milestones) != len(milestones) or completion_goal not in self.milestones:
            raise ValueError("unique milestones and a valid completion goal are required")
        self.completion_goal = completion_goal
        self._previous: TaskSnapshot | None = None
        self._closed = False

    def signals(self, s: TaskSnapshot) -> StageSignals:
        raise NotImplementedError

    def _command_reward(self, snapshot: TaskSnapshot, *, covered_s: float,
                        fresh: bool, failed: bool) -> tuple[dict[str, float], dict[str, float]]:
        """Optional action guidance, outside geometric potential accounting."""
        return {}, {}

    def _signals_without_new_evidence(self, measured: StageSignals) -> StageSignals:
        """Default: reused feedback cannot advance measured stage signals."""
        return self._signals

    def _milestone_progress(self, name: str, milestone: Milestone, current: StageSignals, *,
                            gap: bool, fresh: bool, same_attempt: bool,
                            covered_s: float) -> tuple[float, bool]:
        """Return measured dwell and certification; permit a shared external judge.

        S5 delegates HOME timing and success to S0. All other milestones keep
        the existing stage-local dwell and once-only reward accounting.
        """
        hold = self._holds[name]
        if gap or not current.goals[name] or not same_attempt:
            hold = 0.0
        elif fresh:
            hold = (min(milestone.hold_s, hold + covered_s)
                    if self._signals.goals[name] else 0.0)
        return hold, bool(current.goals[name] and hold + 1e-9 >= milestone.hold_s)

    def _validate_snapshot(self, s: TaskSnapshot) -> None:
        m = s.measurements
        if s.task_mode != "pick_place" or s.target_name is None:
            raise ValueError("later stages require a pick_place task with its original target")
        if m.grasp_evidence_mode != "contact" or not all((m.evidence_valid,
                m.joint_measurement_valid, m.target_measurement_valid,
                m.target_motion_valid, m.contact_valid)):
            raise ValueError("invalid sensor evidence; discard data, do not score as a drop")
        if not math.isfinite(s.evidence_sim_time_s) or not math.isfinite(s.sim_time_s):
            raise ValueError("non-finite snapshot time")
        age = s.sim_time_s - s.evidence_sim_time_s
        if age < -1e-9 or age > self.reward_config.max_evidence_gap_s + 1e-9:
            raise ValueError("stale or future sensor evidence")

    def _checked_signals(self, s: TaskSnapshot) -> StageSignals:
        signals = self.signals(s)
        if set(signals.qualities) != set(self.weights) or set(signals.goals) != set(self.milestones):
            raise ValueError("stage signal keys differ from reward contract")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in signals.qualities.values()):
            raise ValueError("qualities must be finite in [0, 1]")
        if any(not isinstance(v, (bool, np.bool_)) for v in signals.goals.values()):
            raise ValueError("milestone predicates must be boolean")
        if any(not math.isfinite(v) for v in signals.metrics.values()):
            raise ValueError("stage metrics must be finite")
        return signals

    def _key(self, name: str) -> str:
        return f"{self.stage.lower()}_{name}"

    def reset(self, snapshot: TaskSnapshot, *, ledger: EpisodeRewardLedger | None = None) -> None:
        self._validate_snapshot(snapshot)
        if snapshot.outcome != "running":
            raise ValueError("reward must start from a running task")
        identity = (snapshot.task_instance_id, snapshot.task_mode, snapshot.target_name)
        if ledger is None and self._previous is not None:
            raise ValueError("reset requires an explicit ledger: reuse for recovery, new for a new episode")
        ledger = ledger or EpisodeRewardLedger.for_snapshot(snapshot)
        if ledger.identity != identity:
            raise ValueError("ledger belongs to a different physical episode")
        initial_signals = self._checked_signals(snapshot)
        self.ledger = ledger
        self._previous, self._closed = snapshot, False
        self._stage_started_sim_time_s = snapshot.sim_time_s
        self._signals = initial_signals
        self._holds = {name: 0.0 for name in self.milestones}
        for name, value in self._signals.qualities.items():
            key = self._key(name)
            self.ledger.best[key] = max(value, self.ledger.best.get(key, 0.0))
        for name, milestone in self.milestones.items():
            if self._signals.goals[name] and milestone.hold_s == 0:
                # An assisted start is a baseline, not policy progress.
                self.ledger.paid.add(self._key(f"{name}_complete"))
                if name == self.completion_goal:
                    self.ledger.paid.add(self._key("stage_complete"))
            if milestone.hold_s > 0:
                self.ledger.best.setdefault(self._key(f"{name}_dwell"), 0.0)
        self._potential = self._phi(self._signals)

    def _phi(self, signals: StageSignals) -> float:
        return self.tuning.potential_max * sum(
            self.weights[k] * v for k, v in signals.qualities.items())

    def reward_state(self) -> dict[str, object]:
        """Future critic/checkpoint adapter must include this history explicitly."""
        if self._previous is None:
            raise RuntimeError("reset required")
        return dict(schema_version=1, stage=self.stage, phase=self._signals.phase,
                    qualities=dict(self._signals.qualities), potential=self._potential,
                    previous_goals=dict(self._signals.goals),
                    dwell_s=dict(self._holds), paid=sorted(self.ledger.paid),
                    best=dict(self.ledger.best), identity=list(self.ledger.identity),
                    stage_started_sim_time_s=self._stage_started_sim_time_s,
                    evidence_sim_time_s=self._previous.evidence_sim_time_s)

    def step(self, transition: PickPlaceRewardInput) -> LaterStageRewardResult:
        if self._previous is None or self._closed:
            raise RuntimeError("reset required before step or after an episode cut")
        prev, nxt = transition.prev_snapshot, transition.next_snapshot
        if prev != self._previous:
            raise ValueError("transition does not continue the previous snapshot")
        if (nxt.task_instance_id, nxt.task_mode, nxt.target_name) != self.ledger.identity:
            raise ValueError("physical task identity changed")
        dt = transition.dt_s
        if not math.isfinite(dt) or dt <= 0 or not math.isclose(
                nxt.sim_time_s - prev.sim_time_s, dt, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("dt must match positive simulation time advance")
        if transition.terminated and transition.truncated:
            raise ValueError("transition cannot be terminated and truncated together")
        if nxt.outcome not in ("running", "success", "task_failure") or bool(
                transition.terminated) != (nxt.outcome != "running"):
            raise ValueError("termination must match the physical task outcome")
        self._validate_snapshot(nxt)
        evidence_dt = nxt.evidence_sim_time_s - prev.evidence_sim_time_s
        if evidence_dt < -1e-9:
            raise ValueError("evidence time moved backwards")
        fresh = bool(0 < evidence_dt <= self.reward_config.max_evidence_gap_s + 1e-9
                     and nxt.evidence_sim_time_s > self._stage_started_sim_time_s + 1e-9)
        # Evidence at/before the stage boundary also establishes only a
        # baseline; a preparation action cannot earn a zero-dwell milestone.
        gap = evidence_dt > 0 and not fresh
        measured = self._checked_signals(nxt)
        current = measured if evidence_dt > 0 else self._signals_without_new_evidence(measured)
        failed = nxt.outcome == "task_failure"
        before = self._potential
        after = 0.0 if transition.terminated else self._phi(current)
        prefix, tuning = self.stage.lower(), self.tuning
        components = {f"{prefix}_time": -tuning.time_cost_per_s * dt}
        # Missing-interval geometry is a new baseline; it earns no progress.
        shaping_before = after if gap and not transition.terminated else before
        components[f"{prefix}_potential_progress"] = self.reward_config.gamma * after - shaping_before
        gained: list[str] = []
        same_attempt = prev.grasp_attempt_id == nxt.grasp_attempt_id
        covered_s = min(dt, max(0.0, evidence_dt),
                        max(0.0, nxt.evidence_sim_time_s - self._stage_started_sim_time_s))
        achieved_goals: dict[str, bool] = {}
        for name, milestone in self.milestones.items():
            self._holds[name], achieved = self._milestone_progress(
                name, milestone, current, gap=gap, fresh=fresh,
                same_attempt=same_attempt, covered_s=covered_s)
            achieved_goals[name] = achieved
            key = self._key(f"{name}_complete")
            components[key] = 0.0
            if fresh and not failed and achieved and key not in self.ledger.paid:
                components[key] = milestone.bonus
                self.ledger.paid.add(key)
                gained.append(name)
            elif gap and current.goals[name] and milestone.hold_s == 0:
                self.ledger.paid.add(key)  # Never pay a jump through unseen evidence.
                if name == self.completion_goal:
                    self.ledger.paid.add(self._key("stage_complete"))
            if milestone.hold_s > 0:
                hold_key = self._key(f"{name}_dwell")
                fraction = unit(self._holds[name] / milestone.hold_s)
                best_hold = self.ledger.best[hold_key]
                # Even a small newly evidenced slice of a hold gives feedback.
                # Losing and rebuilding a hold cannot farm this progress credit.
                components[hold_key] = (tuning.hold_progress_bonus * max(0.0, fraction - best_hold)
                                        if fresh and not failed else 0.0)
                self.ledger.best[hold_key] = max(best_hold, fraction)
        regression = 0.0
        for name, value in current.qualities.items():
            key = self._key(name)
            record = self.ledger.best[key]
            for index in range(tuning.quality_bands):
                band_value = unit(value * tuning.quality_bands - index)
                band_record = unit(record * tuning.quality_bands - index)
                components[f"{key}_band_{index + 1:02d}"] = (
                    tuning.record_bonus_total * self.weights[name] / tuning.quality_bands
                    * max(0.0, band_value - band_record) if fresh and not failed else 0.0)
            regression += self.weights[name] * max(0.0, record - value)
            if evidence_dt > 0:
                self.ledger.best[key] = max(record, value)
        components[f"{prefix}_regression"] = -tuning.regression_cost_per_s * dt * regression
        components[f"{prefix}_grasp_loss"] = 0.0
        components[f"{prefix}_placement_loss"] = 0.0
        if fresh and "grasp_lost" in nxt.events and "valid_release" not in nxt.events:
            key = f"grasp_lost_attempt_{nxt.grasp_attempt_id}"
            if key not in self.ledger.paid:
                components[f"{prefix}_grasp_loss"] = -(tuning.drop_cost if prev.attempt_lifted
                                                       else tuning.grasp_loss_cost)
                self.ledger.paid.add(key)
        if fresh and "placement_invalidated" in nxt.events:
            components[f"{prefix}_placement_loss"] = -tuning.placement_loss_cost
        complete = bool(fresh and not failed and achieved_goals[self.completion_goal])
        completion_key = self._key("stage_complete")
        success_event = complete and completion_key not in self.ledger.paid
        components[completion_key] = tuning.stage_completion_bonus if success_event else 0.0
        if success_event:
            self.ledger.paid.add(completion_key)
        components[f"{prefix}_failure"] = -tuning.failure_cost if failed else 0.0
        command_components, command_metrics = self._command_reward(
            nxt, covered_s=covered_s, fresh=fresh, failed=failed)
        components.update(command_components)
        reward = float(sum(components.values()))
        if not math.isfinite(reward):
            raise ValueError("non-finite reward")
        self._previous, self._signals, self._potential = nxt, current, after
        self._closed = transition.terminated or transition.truncated
        metrics = dict(current.metrics, **{f"{k}_hold_s": v for k, v in self._holds.items()})
        metrics.update(command_metrics)
        return LaterStageRewardResult(reward, components, before, after,
            transition.terminated, transition.truncated, self.stage, current.phase,
            complete, success_event, tuple(gained), metrics)
