"""Whole-wave stage selection using accepted on-policy transition counts.

The scheduler does not pad, split, or discard trajectories. Call ``observe``
only after complete valid segments are accepted, and call ``next_stage`` only
at a wave boundary. The policy must remain fixed until the batch is ready.
Batch counters deliberately do not belong in checkpoint collection state:
without the corresponding rollout, restoring them would invent experience.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import math


TRAINING_STAGE_SELECTIONS = {
    **{f"s{index}": (f"S{index}",) for index in range(6)},
    "s0_s1": ("S0", "S1"),
    "s0_s5": tuple(f"S{index}" for index in range(6)),
}


def selected_training_stages(selection: str) -> tuple[str, ...]:
    try:
        return TRAINING_STAGE_SELECTIONS[selection]
    except (KeyError, TypeError) as error:
        raise ValueError("Stage selection must be s0..s5, s0_s1 or s0_s5.") from error


def uses_later_training_stages(selection: str) -> bool:
    return selection != "full" and any(int(stage[1:]) >= 2 for stage in selected_training_stages(selection))


def parse_stage_step_weights(value: str) -> tuple[float, ...]:
    """Parse positive S0:S1:S2:S3:S4:S5 accepted-step weights.

    Weights are not episode counts. Skipping an already successful stage is a
    separate operation; zero weights must not silently disable a selected stage.
    """
    message = "stage_step_weights must contain six positive finite numbers in S0:S1:S2:S3:S4:S5 order."
    if not isinstance(value, str) or len(value.split(":")) != 6:
        raise ValueError(message)
    try:
        weights = tuple(float(part.strip()) for part in value.split(":"))
    except (ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError(message)
    return weights


class StageWaveScheduler:
    """Balance S0/S1 steps while keeping each collection wave in one stage."""

    SCHEMA = "fixed_stage_wave_actual_steps_v2_variable_world_count"
    _MAX_PLAN_ENUMERATION = 100_000

    def __init__(self, *, selection: str, rollout_steps: int,
                 s0_step_fraction: float = .4, ratio_tolerance: float = .05,
                 max_rollout_multiplier: float = 4):
        self.validate(selection=selection, rollout_steps=rollout_steps,
                      s0_step_fraction=s0_step_fraction, ratio_tolerance=ratio_tolerance,
                      max_rollout_multiplier=max_rollout_multiplier)
        self.selection = selection
        self.rollout_steps = rollout_steps
        self.s0_step_fraction = (1.0 if selection == "s0" else
                                 0.0 if selection == "s1" else float(s0_step_fraction))
        self.ratio_tolerance = float(ratio_tolerance)
        self.max_rollout_multiplier = float(max_rollout_multiplier)
        self.max_steps_before_next_wave = math.ceil(rollout_steps * max_rollout_multiplier)
        self.reset()

    @staticmethod
    def validate(*, selection: str, rollout_steps: int,
                 s0_step_fraction: float = .4, ratio_tolerance: float = .05,
                 max_rollout_multiplier: float = 4) -> None:
        if selection not in ("s0", "s1", "s0_s1"):
            raise ValueError("Stage selection must be s0, s1 or s0_s1.")
        if isinstance(rollout_steps, bool) or not isinstance(rollout_steps, int) or rollout_steps <= 0:
            raise ValueError("rollout_steps must be a positive integer.")
        for name, value in (("s0_step_fraction", s0_step_fraction),
                            ("ratio_tolerance", ratio_tolerance),
                            ("max_rollout_multiplier", max_rollout_multiplier)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if not 0 <= s0_step_fraction <= 1:
            raise ValueError("s0_step_fraction must be in [0, 1].")
        if not 0 <= ratio_tolerance <= .5:
            raise ValueError("ratio_tolerance must be in [0, 0.5].")
        if max_rollout_multiplier < 1:
            raise ValueError("max_rollout_multiplier must be at least 1.")

    def reset(self) -> None:
        """Start an empty batch after its rollout has been consumed/discarded."""
        self._counts = {"S0": 0, "S1": 0}
        self._episode_counts = {"S0": 0, "S1": 0}

    @property
    def total_steps(self) -> int:
        return self._counts["S0"] + self._counts["S1"]

    @property
    def _required_stages_present(self) -> bool:
        return ((self.s0_step_fraction == 0 or self._counts["S0"] > 0)
                and (self.s0_step_fraction == 1 or self._counts["S1"] > 0))

    @property
    def _ratio_error(self) -> float | None:
        if self.total_steps == 0:
            return None
        return abs(self._counts["S0"] / self.total_steps - self.s0_step_fraction)

    @property
    def _ratio_met(self) -> bool:
        error = self._ratio_error
        return error is not None and error <= self.ratio_tolerance + 1e-12

    @property
    def ready(self) -> bool:
        return (self.total_steps >= self.rollout_steps
                and self._required_stages_present and self._ratio_met)

    def observe(self, stage: str, steps: int) -> None:
        """Count actual accepted transitions; never replace a short success.

        Do not check readiness or the cap here: all complete segments from an
        already-started wave must be included, even if an earlier segment made
        the running total temporarily ready or crossed the collection cap.
        """
        if stage not in ("S0", "S1"):
            raise ValueError("Observed stage must be S0 or S1.")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("Observed steps must be a positive integer.")
        if (stage == "S0" and self.s0_step_fraction == 0
                or stage == "S1" and self.s0_step_fraction == 1):
            raise ValueError(f"{stage} is not a target stage for this collection contract.")
        self._counts[stage] += steps
        self._episode_counts[stage] += 1

    def next_stage(self) -> str:
        """Select one stage for the entire next wave, including its retries."""
        if self.ready:
            raise RuntimeError("Stage rollout is ready for PPO; reset after consuming it before collecting another wave.")
        if self.total_steps >= self.max_steps_before_next_wave:
            actual = self._counts["S0"] / self.total_steps
            raise RuntimeError(
                "Stage rollout step ratio is not ready at the collection cap: "
                f"S0={self._counts['S0']}, S1={self._counts['S1']}, total={self.total_steps}, "
                f"S0 fraction={actual:.6f}, target={self.s0_step_fraction:.6f} "
                f"+/- {self.ratio_tolerance:.6f}, cap={self.max_steps_before_next_wave}. "
                "Do not update an imbalanced batch or pad successful episodes; "
                "review the ratio tolerance, wave size or collection cap."
            )
        if self.s0_step_fraction == 1:
            return "S0"
        if self.s0_step_fraction == 0:
            return "S1"
        if self.total_steps:
            actual = self._counts["S0"] / self.total_steps
            if actual < self.s0_step_fraction:
                return "S0"
            if actual > self.s0_step_fraction:
                return "S1"
        return "S0" if self.s0_step_fraction >= .5 else "S1"

    def plan_wave(self, max_worlds: int, *, s0_episode_steps: int,
                  s1_episode_steps: int) -> dict[str, object]:
        """Plan a bounded whole-episode tail, then start only its next wave.

        Estimates are this batch's observed episode means, or the supplied
        budgets for an unseen stage. Each call to ``observe`` must therefore
        describe one complete episode. Replan after every completed wave;
        forecasts never satisfy the actual-step readiness test on their own.
        """
        for name, value in (("max_worlds", max_worlds),
                            ("s0_episode_steps", s0_episode_steps),
                            ("s1_episode_steps", s1_episode_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        preferred_stage = self.next_stage()  # Preserve the actual ready/cap guards.
        budgets = {"S0": s0_episode_steps, "S1": s1_episode_steps}
        estimates = {
            stage: (self._counts[stage] / self._episode_counts[stage]
                    if self._episode_counts[stage] else float(budgets[stage]))
            for stage in ("S0", "S1")
        }
        # Enumerate the more expensive episode dimension. For each integer
        # count, total steps and the upper share bound give an analytic lower
        # bound for the other count; any larger count costs more predicted
        # steps. This is linear, not a Cartesian search over both stages.
        x_stage = "S0" if estimates["S0"] >= estimates["S1"] else "S1"
        y_stage = "S1" if x_stage == "S0" else "S0"
        x_share = self.s0_step_fraction if x_stage == "S0" else 1 - self.s0_step_fraction
        low_share = max(0.0, x_share - self.ratio_tolerance)
        high_share = min(1.0, x_share + self.ratio_tolerance)
        x_steps, y_steps = self._counts[x_stage], self._counts[y_stage]
        x_estimate, y_estimate = estimates[x_stage], estimates[y_stage]
        search_limit = self.max_steps_before_next_wave + max_worlds * max(estimates.values())
        min_x = int(x_share > 0 and x_steps == 0)
        max_x = math.floor(max(0.0, search_limit - self.total_steps) / x_estimate)
        if x_share == 0:
            max_x = 0
        min_y = int(x_share < 1 and y_steps == 0)
        best: tuple[float, float, int, dict[str, int]] | None = None
        search_skipped = max_x - min_x + 1 > self._MAX_PLAN_ENUMERATION
        if not search_skipped:
            for extra_x in range(min_x, max_x + 1):
                predicted_x = x_steps + extra_x * x_estimate
                required_y = max(float(min_y),
                                 (self.rollout_steps - predicted_x - y_steps) / y_estimate)
                if high_share == 0:
                    if predicted_x > 0:
                        continue
                else:
                    required_y = max(required_y,
                                     ((1 - high_share) * predicted_x - high_share * y_steps)
                                     / (high_share * y_estimate))
                extra_y = max(0, math.ceil(required_y - 1e-12))
                if x_share == 1 and extra_y:
                    continue
                predicted_y = y_steps + extra_y * y_estimate
                predicted_total = predicted_x + predicted_y
                if predicted_total + 1e-9 < self.rollout_steps or predicted_total > search_limit + 1e-9:
                    continue
                if ((x_share > 0 and predicted_x <= 0)
                        or (x_share < 1 and predicted_y <= 0)):
                    continue
                predicted_share = predicted_x / predicted_total
                if not low_share - 1e-12 <= predicted_share <= high_share + 1e-12:
                    continue
                error = abs(predicted_share - x_share)
                episodes = extra_x + extra_y
                counts = {x_stage: extra_x, y_stage: extra_y}
                if (best is None or predicted_total < best[0] - 1e-9
                        or (abs(predicted_total - best[0]) <= 1e-9
                            and (error < best[1] - 1e-12
                                 or (abs(error - best[1]) <= 1e-12 and episodes < best[2])))):
                    best = (predicted_total, error, episodes, counts)

        fallback = best is None
        if fallback:
            # Infeasible forecasts are not proof of infeasible future actual
            # durations. Collect a bounded next wave and measure again.
            stage = preferred_stage
            remaining = max(0, self.rollout_steps - self.total_steps)
            world_count = (min(max_worlds, max(1, math.ceil(remaining / estimates[stage])))
                           if remaining else max_worlds)
            additional = {"S0": 0, "S1": 0}
            additional[stage] = world_count
        else:
            additional = best[3]
            stage = preferred_stage if additional[preferred_stage] else (
                "S1" if preferred_stage == "S0" else "S0")
            world_count = min(max_worlds, additional[stage])
        if world_count <= 0:
            raise RuntimeError("Stage tail planning did not produce a nonempty next wave.")
        final_s0 = self._counts["S0"] + additional["S0"] * estimates["S0"]
        final_s1 = self._counts["S1"] + additional["S1"] * estimates["S1"]
        return {
            "stage": stage,
            "world_count": world_count,
            "planned_additional_episodes": dict(additional),
            "estimated_final_steps": final_s0 + final_s1,
            "estimated_final_s0_fraction": final_s0 / (final_s0 + final_s1),
            "estimated_episode_steps": dict(estimates),
            "planning_fallback": fallback,
            "planning_status": ("fallback_search_bound" if search_skipped else
                                "fallback_no_feasible_tail" if fallback else "estimated_tail"),
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "s0_steps": self._counts["S0"],
            "s1_steps": self._counts["S1"],
            "s0_episode_count": self._episode_counts["S0"],
            "s1_episode_count": self._episode_counts["S1"],
            "total_steps": self.total_steps,
            "target_s0_step_fraction": self.s0_step_fraction,
            "actual_s0_step_fraction": self._counts["S0"] / self.total_steps if self.total_steps else None,
            "ratio_error": self._ratio_error,
            "ratio_tolerance": self.ratio_tolerance,
            "ratio_met": self._ratio_met,
            "required_stages_present": self._required_stages_present,
            "ready": self.ready,
            "rollout_step_threshold": self.rollout_steps,
            "max_steps_before_next_wave": self.max_steps_before_next_wave,
        }

    def contract(self) -> dict[str, object]:
        """Immutable collection semantics for resume compatibility checks."""
        return {
            "schema": self.SCHEMA,
            "selection": self.selection,
            "target_s0_step_fraction": self.s0_step_fraction,
            "ratio_tolerance": self.ratio_tolerance,
            "max_rollout_multiplier": self.max_rollout_multiplier,
            "rollout_step_threshold": self.rollout_steps,
            "world_count_semantics": "maximum_available_worlds_per_wave",
            "tail_planning": "minimum_estimated_complete_episode_steps_actual_counts_decide_readiness",
        }


class BalancedStageWaveScheduler:
    """Configurable accepted-step shares; never insert preparation steps.

    Skipped stages are removed before constructing a new empty batch. Whole
    episodes remain intact, and forecasts only choose the next wave size.
    """

    SCHEMA = "equal_stage_wave_actual_steps_v1"
    WEIGHTED_SCHEMA = "weighted_stage_wave_actual_steps_v1"

    def __init__(self, *, selection: str, rollout_steps: int,
                 active_stages: frozenset[str] | None = None,
                 stage_step_weights: str = "1:1:1:1:1:1",
                 ratio_tolerance: float = .05, max_rollout_multiplier: float = 4):
        configured = selected_training_stages(selection)
        self.stage_step_weights = parse_stage_step_weights(stage_step_weights)
        StageWaveScheduler.validate(selection="s0_s1", rollout_steps=rollout_steps,
            ratio_tolerance=ratio_tolerance, max_rollout_multiplier=max_rollout_multiplier)
        self.selection = selection
        self.stages = tuple(stage for stage in configured
                            if active_stages is None or stage in active_stages)
        if not self.stages or (active_stages is not None and not active_stages <= set(configured)):
            raise ValueError("Active stages must be a nonempty subset of the selected stages.")
        weights = {stage: self.stage_step_weights[int(stage[1:])] for stage in self.stages}
        self._equal_shares = len(set(weights.values())) == 1
        if self._equal_shares:
            # Preserve the exact old contract and planner for equal shares.
            self.target_stage_fractions = {stage: 1 / len(self.stages) for stage in self.stages}
        else:
            # Normalize decimal ratios before rounding to floats. Equivalent
            # 1:3 and .7:2.1 weights must not differ in the resume contract just
            # because binary division rounds them in opposite directions.
            with localcontext() as context:
                context.prec = 40
                decimal_weights = {stage: Decimal(str(weight)) for stage, weight in weights.items()}
                total_weight = sum(decimal_weights.values())
                self.target_stage_fractions = {
                    stage: float(weight / total_weight) for stage, weight in decimal_weights.items()
                }
            if any(fraction <= 0 for fraction in self.target_stage_fractions.values()):
                raise ValueError("stage_step_weights differ too much to represent positive stage fractions.")
        self.rollout_steps = rollout_steps
        self.ratio_tolerance = float(ratio_tolerance)
        self.max_rollout_multiplier = float(max_rollout_multiplier)
        self.max_steps_before_next_wave = math.ceil(rollout_steps * max_rollout_multiplier)
        self.s0_step_fraction = self.target_stage_fractions.get("S0", 0.0)
        self.reset()

    def reset(self) -> None:
        self._counts = {stage: 0 for stage in self.stages}
        self._episode_counts = dict(self._counts)

    @property
    def total_steps(self) -> int:
        return sum(self._counts.values())

    @property
    def ready(self) -> bool:
        return (self.total_steps >= self.rollout_steps and all(self._counts.values())
                and max(abs(count / self.total_steps - self.target_stage_fractions[stage])
                        for stage, count in self._counts.items()) <= self.ratio_tolerance + 1e-12)

    def observe(self, stage: str, steps: int) -> None:
        if stage not in self.stages:
            raise ValueError(f"{stage} is not active in this stage batch.")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("Observed steps must be a positive integer.")
        self._counts[stage] += steps
        self._episode_counts[stage] += 1

    def next_stage(self) -> str:
        if self.ready:
            raise RuntimeError("Stage rollout is ready for PPO; consume it before collecting another wave.")
        if self.total_steps >= self.max_steps_before_next_wave:
            raise RuntimeError(f"Stage rollout step ratio is not ready at the collection cap: {self._counts}; "
                               f"cap={self.max_steps_before_next_wave}. No imbalanced update will run.")
        return min(self.stages, key=lambda stage: self._counts[stage] / self.target_stage_fractions[stage])

    def plan_wave(self, max_worlds: int, *, s0_episode_steps: int,
                  s1_episode_steps: int, stage_episode_steps: dict[str, int] | None = None) -> dict[str, object]:
        if isinstance(max_worlds, bool) or not isinstance(max_worlds, int) or max_worlds < 1:
            raise ValueError("max_worlds must be a positive integer.")
        budgets = {"S0": s0_episode_steps, "S1": s1_episode_steps,
                   **(stage_episode_steps or {})}
        if any(isinstance(budgets.get(stage), bool) or not isinstance(budgets.get(stage), int)
               or budgets[stage] <= 0 for stage in self.stages):
            raise ValueError("Every active stage requires a positive episode step budget.")
        stage = self.next_stage()
        estimates = {key: (self._counts[key] / self._episode_counts[key]
                           if self._episode_counts[key] else float(budgets[key])) for key in self.stages}
        # Catch up to the most represented stage after dividing by its share,
        # or to each stage's minimum requested share of the rollout threshold.
        # Bound the final wave to complete episodes; actual counts decide readiness.
        if self._equal_shares:
            target = max(self.rollout_steps / len(self.stages), max(self._counts.values()))
            episodes = math.ceil((target - self._counts[stage]) / estimates[stage])
        else:
            target_total = max(self.rollout_steps,
                               max(self._counts[key] / self.target_stage_fractions[key] for key in self.stages))
            target = target_total * self.target_stage_fractions[stage]
            episodes = math.ceil((target - self._counts[stage]) / estimates[stage] - 1e-12)
        worlds = min(max_worlds, max(1, episodes))
        return {"stage": stage, "world_count": worlds,
                "planned_additional_episodes": {key: worlds if key == stage else 0 for key in self.stages},
                "estimated_episode_steps": estimates,
                "planning_fallback": False,
                "planning_status": ("equal_share_complete_episode_wave" if self._equal_shares
                                    else "weighted_share_complete_episode_wave")}

    def diagnostics(self) -> dict[str, object]:
        total = self.total_steps
        shares = {stage: self._counts[stage] / total if total else None for stage in self.stages}
        error = max(abs(value - self.target_stage_fractions[stage])
                    for stage, value in shares.items()) if total else None
        return {
            **{f"s{i}_steps": self._counts.get(f"S{i}", 0) for i in range(6)},
            **{f"s{i}_episode_count": self._episode_counts.get(f"S{i}", 0) for i in range(6)},
            "stage_steps": dict(self._counts), "stage_fractions": shares,
            "target_stage_fractions": dict(self.target_stage_fractions),
            "active_stages": list(self.stages), "total_steps": total,
            "target_s0_step_fraction": self.s0_step_fraction,
            "actual_s0_step_fraction": self._counts.get("S0", 0) / total if total else None,
            "ratio_error": error, "ratio_tolerance": self.ratio_tolerance,
            "ratio_met": error is not None and error <= self.ratio_tolerance + 1e-12,
            "required_stages_present": all(self._counts.values()), "ready": self.ready,
            "rollout_step_threshold": self.rollout_steps,
            "max_steps_before_next_wave": self.max_steps_before_next_wave,
        }

    def contract(self) -> dict[str, object]:
        return {"schema": self.SCHEMA if self._equal_shares else self.WEIGHTED_SCHEMA,
                "selection": self.selection, "stages": list(self.stages),
                "target_step_fractions": dict(self.target_stage_fractions),
                "ratio_tolerance": self.ratio_tolerance,
                "max_rollout_multiplier": self.max_rollout_multiplier,
                "rollout_step_threshold": self.rollout_steps,
                "world_count_semantics": "maximum_available_worlds_per_wave",
                "tail_planning": ("least_actual_steps_complete_episode_wave_actual_counts_decide_readiness"
                                  if self._equal_shares else
                                  "least_normalized_actual_steps_complete_episode_wave_actual_counts_decide_readiness")}
