"""Measured lift-then-approach guidance within a single S1 collection window.

Height and wrist-turn bands pay continuously for new measured records. The
approach objective starts only after the evaluator certifies the lift, with
its own baseline. No commands, inverse kinematics or extra reset are used.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .pick_place_reward import PickPlaceRewardInput, PickPlaceRewardResult
from .stage_process_reward import (
    S1_POTENTIAL_NAMES, StageReachReward, _flag, _interval_error, _metric,
)


NAMES = (*S1_POTENTIAL_NAMES, "lift", "roll", "parallel", "corridor")
APPROACH_INDICES = (0, 1, 2, 3, 7, 8)
POSTURE_NAMES = ("bearing", "axis", "parallel", "wrist_roll")
POSTURE_METRICS = ("stage_bearing_error_rad", "stage_axis_error_rad",
                   "stage_s1_parallel_error_rad", "stage_s1_roll_error_rad")


def _quality(error: float, scale: float) -> float:
    return scale / (scale + max(error, 0.0))


def _band_credit(progress: float, total: float, step: float, bonus: float) -> np.ndarray:
    """Linear within each band, including a possibly shorter final interval."""
    inner = np.arange(math.ceil(total / step)) * step
    # Decimal divisions such as .07/.01 can round just above an integer.
    # Drop a numerically identical end point before appending the exact end.
    inner = inner[(inner < total) & ~np.isclose(inner, total, rtol=1e-12, atol=0.0)]
    edges = np.r_[inner, total]
    return bonus * np.clip((progress - edges[:-1]) / np.diff(edges), 0.0, 1.0)


class S1LiftApproachReward(StageReachReward):
    def __init__(self, config, stage_config, reward_config):
        if config.schema_version != 4:
            raise ValueError("S1LiftApproachReward requires process schema 4")
        super().__init__(config, stage_config, reward_config)
        self.sequence = config.s1_lift_approach
        self._potentials = np.zeros(len(NAMES), dtype=np.float64)
        c = self.sequence
        state_budget = (c.dense_cost_failure_fraction * reward_config.failure_cost
                        / stage_config.s1_budget_s - config.s1_time_cost_per_s)
        if state_budget <= 0:
            raise ValueError("S1 fixed time cost exhausts the dense cost failure budget")
        # Keep the conservative original upper bound when rotation is off:
        # removing that cost must not silently amplify distance/lift costs.
        maximum_state_rate = config.s1_remaining_cost_per_s + max(
            c.lift_remaining_cost_per_s + c.roll_remaining_cost_per_s,
            config.s1_distance_cost_per_s + c.roll_remaining_cost_per_s + c.corridor_cost_per_s)
        self.state_cost_scale = min(1.0, state_budget / maximum_state_rate)
        if c.posture_regression is not None:
            maximum_rate = (config.s1_time_cost_per_s + self.state_cost_scale * maximum_state_rate
                            + len(POSTURE_NAMES) * c.posture_regression.cost_per_s)
            if maximum_rate * stage_config.s1_budget_s >= reward_config.failure_cost:
                raise ValueError("posture regression exhausts the remaining dense cost failure budget")

    def _weights(self):
        c = self.sequence
        return np.asarray((*self.config.s1_potential_weights, c.lift_potential_weight,
                           c.roll_potential_weight if c.wrist_rotation_reward_enabled else 0.0,
                           c.parallel_potential_weight if c.wrist_rotation_reward_enabled else 0.0,
                           c.lift_potential_weight), dtype=np.float64)

    def _score_terms(self, snapshot, info):
        c, s, p = self.sequence, self.stage_config, self.config
        if _flag(info, "stage_downstream_grasp_lift"):
            return self._weights()
        if not self._eligible(snapshot, info):
            return np.zeros(len(NAMES))
        approach = info["stage_s1_phase"] == "APPROACH"
        # During lift, target distance/height must not penalize moving upwards.
        values = (super()._score_terms(snapshot, info) if approach
                  else np.zeros(len(S1_POTENTIAL_NAMES)))
        # Alignment is active at every height, independent of distance gates.
        values[4] = p.s1_potential_weights[4] * _quality(
            _metric(info, "stage_bearing_error_rad") - s.s1_orientation_tolerance_rad,
            p.s1_orientation_scale_rad)
        values[5] = p.s1_potential_weights[5] * _quality(
            _metric(info, "stage_axis_error_rad") - s.s1_orientation_tolerance_rad,
            p.s1_orientation_scale_rad)
        values[6] = p.s1_potential_weights[6] * _quality(
            _interval_error(_metric(info, "stage_opening_rad"), s.s1_opening_min_rad,
                            s.s1_opening_max_rad), p.s1_opening_scale_rad)
        deficit = _metric(info, "stage_s1_corridor_deficit_m") if approach else 0.0
        corridor = _quality(deficit, p.s1_height_scale_m)
        if approach:
            # A low lateral shortcut loses approach quality continuously; the
            # first-visit distance bonus also requires the certified corridor.
            values[list(APPROACH_INDICES)] *= corridor
        lift = (1.0 if approach else _quality(
            c.lift_clearance_m - _metric(info, "stage_s1_clearance_m"), p.s1_height_scale_m))
        roll = (_quality(_metric(info, "stage_s1_roll_error_rad") - c.roll_tolerance_rad,
                         p.s1_orientation_scale_rad) if c.wrist_rotation_reward_enabled else 0.0)
        parallel = (_quality(_metric(info, "stage_s1_parallel_error_rad"), p.s1_orientation_scale_rad)
                    if c.wrist_rotation_reward_enabled else 0.0)
        return np.r_[values, c.lift_potential_weight * lift, c.roll_potential_weight * roll,
                     c.parallel_potential_weight * parallel, c.lift_potential_weight * corridor]

    def _height_record(self, info):
        return float(np.clip(_metric(info, "stage_s1_clearance_m"), 0, self.sequence.lift_clearance_m))

    def _roll_record(self, info):
        total = abs(self.sequence.roll_target_rad)
        return float(np.clip(total - _metric(info, "stage_s1_roll_error_rad"), 0, total))

    def reset(self, snapshot, stage_info):
        self._validate_snapshot(snapshot, stage_info)
        if snapshot.outcome != "running" or not stage_info.get("start_valid"):
            raise ValueError("stage reward reset requires a valid running start certificate")
        if stage_info.get("stage_s1_phase") != "LIFT":
            raise ValueError("schema 4 S1 must begin in LIFT")
        self._previous = snapshot
        self._potentials = self._score_terms(snapshot, stage_info) * self._shaping_scale
        self._elapsed_fraction, self._closed = self._elapsed(stage_info), False
        self._info = dict(stage_info)
        self._approach = False
        self._best_distance_quality = 0.0
        self._distance_quality_last = 0.0
        self._best_height = self._height_record(stage_info)
        self._best_roll = self._roll_record(stage_info)
        self._previous_eligible = self._eligible(snapshot, stage_info)
        self._previous_corridor_safe = _flag(stage_info, "stage_s1_corridor_safe")
        # The start certificate is not a fresh transition. Records are first
        # initialized by an eligible measured endpoint, not by reset geometry.
        self._best_posture_errors = np.full(len(POSTURE_NAMES), -1.0)
        self._posture_cost_fractions = np.zeros(len(POSTURE_NAMES))

    def critic_context(self):
        c, info = self.sequence, self._info
        values = (1.0, float(self._previous is not None), *self._potentials, self._elapsed_fraction,
                  self._best_distance_quality, float(self._previous_eligible), float(self._approach),
                  min(_metric(info, "stage_s1_lift_hold_s") / c.lift_hold_s, 1.0),
                  self._best_height / c.lift_clearance_m, self._best_roll / abs(c.roll_target_rad),
                  float(self._previous_corridor_safe),
                  _metric(info, "stage_s1_clearance_m") / c.lift_clearance_m,
                  _metric(info, "stage_s1_roll_error_rad") / math.pi, self._distance_quality_last)
        if c.posture_regression is not None:
            # -1 explicitly means no qualified record yet. Carried fractions
            # expose the costs frozen across invalid/reused observations.
            records = np.where(self._best_posture_errors < 0, -1.0,
                               self._best_posture_errors / math.pi)
            values = (*values, *records, *self._posture_cost_fractions)
        return np.asarray(values, dtype=np.float32)

    def _posture_regression_costs(self, info, *, fresh, eligible, terminated, dt_s):
        c = self.sequence.posture_regression
        if c is None:
            return {}
        if fresh and not terminated and _flag(info, "stage_downstream_grasp_lift"):
            # A certified successful grasp/lift legitimately changes reach
            # posture; do not charge retreat from a pre-grasp orientation.
            self._posture_cost_fractions.fill(0.0)
        elif fresh and eligible and not terminated:
            errors = np.asarray([_metric(info, name) for name in POSTURE_METRICS])
            if np.any(errors < 0):
                raise ValueError("posture errors must be nonnegative")
            self._best_posture_errors = np.where(
                self._best_posture_errors < 0, errors, np.minimum(self._best_posture_errors, errors))
            tolerances = np.asarray((self.stage_config.s1_orientation_tolerance_rad,
                                     self.stage_config.s1_orientation_tolerance_rad,
                                     c.parallel_tolerance_rad, self.sequence.roll_tolerance_rad))
            excess = errors - np.maximum(self._best_posture_errors, tolerances) - c.slack_rad
            self._posture_cost_fractions = np.clip(excess / c.error_scale_rad, 0.0, 1.0)
        if not self.sequence.wrist_rotation_reward_enabled:
            # Neither freshly measured nor carried stale costs may ask the
            # actor to restore a wrist angle or finger/table parallelism.
            self._posture_cost_fractions[2:] = 0.0
            self._best_posture_errors[2:] = -1.0
        # Bad evidence cannot establish a new best or erase an existing cost.
        # The last valid fractions continue charging every control step.
        return {f"s1_{name}_regression": -c.cost_per_s * dt_s * float(fraction)
                for name, fraction in zip(POSTURE_NAMES, self._posture_cost_fractions)}

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
        dt = nxt.evidence_sim_time_s - prev.evidence_sim_time_s
        if dt < -1e-9:
            raise ValueError("stage evidence time moved backwards")
        fresh = 0 < dt <= self.reward_config.max_evidence_gap_s + 1e-9 and _flag(stage_info, "stage_evidence_fresh")
        # Repeated callbacks cannot change measured progress or reduce costs.
        info = dict(stage_info) if dt > 0 else self._info
        approach = info["stage_s1_phase"] == "APPROACH"
        if self._approach and not approach:
            raise ValueError("S1 approach cannot return to lift")
        switched = approach and not self._approach
        before = self._potentials.copy()
        after = (np.zeros_like(before) if transition.terminated else
                 self._score_terms(nxt, info) * self._shaping_scale if dt > 0 else before.copy())
        deltas = after - before
        effective_deltas = deltas.copy()
        if switched:
            effective_deltas[list(APPROACH_INDICES)] = 0.0
        components = {f"s1_{name}_progress": float(value) for name, value in zip(NAMES, deltas)}
        # Changing the objective is not physical progress. Log its cancellation
        # explicitly; subsequent approach steps use this new measured baseline.
        components["s1_phase_rebase"] = -float((deltas - effective_deltas).sum())
        components["s1_discount_correction"] = (self.reward_config.gamma - 1.0) * float(after.sum())
        components["s1_evidence_guard"] = -float(np.maximum(effective_deltas, 0).sum()) if not fresh else 0.0
        components["s1_time"] = -self.config.s1_time_cost_per_s * transition.dt_s
        eligible = self._eligible(nxt, info)
        safe = _flag(info, "stage_s1_corridor_safe")
        can_pay = fresh and eligible and self._previous_eligible and not transition.terminated
        c = self.sequence
        height, roll = self._height_record(info), self._roll_record(info)
        for name, current, record, total, step, bonus, active in (
            ("lift", height, self._best_height, c.lift_clearance_m, c.lift_band_step_m, c.lift_band_bonus, not self._approach),
            ("roll", roll, self._best_roll, abs(c.roll_target_rad), c.roll_band_step_rad, c.roll_band_bonus, c.wrist_rotation_reward_enabled),
        ):
            credit = _band_credit(current, total, step, bonus)
            gains = np.maximum(credit - _band_credit(record, total, step, bonus), 0)
            for index, gain in enumerate(gains):
                components[f"s1_{name}_band_{index + 1:02d}"] = float(gain) if can_pay and active else 0.0
        components["s1_best_distance"] = 0.0
        gains = np.zeros(len(self.config.s1_distance_band_bonuses))
        distance_quality = self._distance_quality(info) if approach else 0.0
        if dt > 0:
            # Consume ineligible or missing-interval endpoints too: a retreat
            # and revisit may not reclaim unobserved/unsafe record credit.
            if not self._approach:
                self._best_height = max(self._best_height, height)
            self._best_roll = max(self._best_roll, roll)
            if switched:
                self._best_distance_quality = distance_quality
            elif approach:
                if can_pay and safe and self._previous_corridor_safe:
                    components["s1_best_distance"] = float(self.config.s1_best_distance_bonus) * max(
                        distance_quality - self._best_distance_quality, 0.0)
                    gains = np.maximum(self._distance_band_credit(distance_quality)
                                       - self._distance_band_credit(self._best_distance_quality), 0)
                self._best_distance_quality = max(self._best_distance_quality, distance_quality)
            self._distance_quality_last = distance_quality
        else:
            distance_quality = self._distance_quality_last
        for index, gain in enumerate(gains):
            components[f"s1_distance_band_{index + 1:02d}"] = float(gain)
        downstream = _flag(info, "stage_downstream_grasp_lift")
        lift_quality = 1.0 if approach else _quality(c.lift_clearance_m - _metric(info, "stage_s1_clearance_m"),
                                                   self.config.s1_height_scale_m)
        components["s1_lift_remaining"] = -c.lift_remaining_cost_per_s * transition.dt_s * (1 - lift_quality)
        roll_deficit = max(_metric(info, "stage_s1_roll_error_rad") - c.roll_tolerance_rad, 0) / abs(c.roll_target_rad)
        components["s1_roll_remaining"] = (-c.roll_remaining_cost_per_s * transition.dt_s * min(roll_deficit, 1.0)
                                            if c.wrist_rotation_reward_enabled else 0.0)
        corridor_deficit = _metric(info, "stage_s1_corridor_deficit_m") if approach else 0.0
        components["s1_corridor_remaining"] = -c.corridor_cost_per_s * transition.dt_s * min(
            corridor_deficit / c.lift_clearance_m, 1.0)
        if not eligible and not downstream:
            distance_quality = 0.0
        components["s1_distance_remaining"] = (-float(self.config.s1_distance_cost_per_s) * transition.dt_s
                                                * (1.0 - distance_quality)) if approach else 0.0
        quality = float(after.sum()) / float(self._weights().sum() * self._shaping_scale)
        components["s1_remaining"] = -self.config.s1_remaining_cost_per_s * transition.dt_s * max(0., 1. - quality)
        # A fixed, configuration-derived scale preserves every gradient and
        # requires no hidden running budget. Dense costs alone cannot make
        # waiting the whole window more costly than immediate physical failure.
        for name in ("s1_remaining", "s1_distance_remaining", "s1_lift_remaining",
                     "s1_roll_remaining", "s1_corridor_remaining"):
            components[name] *= self.state_cost_scale
        components.update(self._posture_regression_costs(
            info, fresh=fresh, eligible=eligible, terminated=transition.terminated, dt_s=transition.dt_s))
        if c.lift_completion_bonus > 0:
            # The evaluator certifies the height and continuous dwell. Its
            # latched phase already exposes payment history to the critic.
            # Keep this event separate from potential rebase/discount terms.
            within_budget = _metric(stage_info, "stage_elapsed_s") <= self.stage_config.s1_budget_s + 1e-9
            components["s1_lift_completed"] = (c.lift_completion_bonus
                if switched and fresh and not transition.terminated and within_budget else 0.0)
        components["s1_success"] = (self.config.s1_success_bonus if fresh and not transition.terminated
                                     and _flag(stage_info, "stage_success") else 0.0)
        components["failure"] = -self.reward_config.failure_cost if nxt.outcome == "task_failure" else 0.0
        components["success"] = self.reward_config.success_bonus if nxt.outcome == "success" and nxt.success_event else 0.0
        reward = float(sum(components.values()))
        if not math.isfinite(reward):
            raise ValueError("non-finite stage process reward")
        self._previous, self._potentials = nxt, after
        self._elapsed_fraction, self._info = self._elapsed(stage_info), dict(info)
        self._approach = approach
        self._previous_eligible = eligible and fresh
        self._previous_corridor_safe = safe
        self._closed = bool(transition.terminated or transition.truncated)
        return PickPlaceRewardResult(reward, components, float(before.sum()), float(after.sum()),
                                     transition.terminated, transition.truncated)
