"""One collection-batch skip after a stage meets its episode success target.

This controller uses completed, valid episodes and the formal ``stage_success``
flag, never reward magnitude, transition count, or a physical-phase label.
Finish a batch only after every episode accepted into it has been observed.
The caller filters *all* transitions of ``exclude_update_stages`` from that
batch, then collects the next full rollout from the remaining active stages.

The controller does not edit a rollout, start a world, invoke PPO or increment
an optimizer-update counter. If every configured stage is skipped, explicitly
consume one empty collection batch and resume normal collection afterwards.
Only batch-boundary state is checkpointable; partial episode statistics are
never saved/restored without the rollout to which they belong.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType

from .act_ppo_stage_schedule import TRAINING_STAGE_SELECTIONS


@dataclass(frozen=True)
class StageSuccessSkipDecision:
    """A completed collection batch and the stages skipped in the next one."""

    batch_index: int
    episode_counts: Mapping[str, int]
    success_counts: Mapping[str, int]
    success_rates: Mapping[str, float | None]
    exclude_update_stages: frozenset[str]
    next_skip_stages: frozenset[str]
    empty_batch: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe copy, with stable ordering for logs."""
        return {
            "batch_index": self.batch_index,
            "episode_counts": dict(sorted(self.episode_counts.items())),
            "success_counts": dict(sorted(self.success_counts.items())),
            "success_rates": dict(sorted(self.success_rates.items())),
            "exclude_update_stages": sorted(self.exclude_update_stages),
            "next_skip_stages": sorted(self.next_skip_stages),
            "empty_batch": self.empty_batch,
        }


class StageSuccessSkipController:
    """Exclude a mastered stage now and skip its next collection batch once.

    ``collection_batch_index`` starts at zero and counts completed collection
    batches, including explicit empty skip batches. It is independent of PPO
    update numbering. An observed invalid episode is ignored in both the
    success numerator and denominator, but still opens the batch boundary.
    """

    SCHEMA = 1
    _STAGES = {selection: frozenset(stages) for selection, stages in TRAINING_STAGE_SELECTIONS.items()}
    _STATE_KEYS = frozenset({
        "schema", "selection", "threshold", "current_skipped_stages",
        "collection_batch_index",
    })

    def __init__(self, selection: str, threshold: float = .8):
        if not isinstance(selection, str) or selection not in self._STAGES:
            raise ValueError("Stage selection must be s0..s5, s0_s1 or s0_s5.")
        self._validate_threshold(threshold)
        self.selection = selection
        self.threshold = float(threshold)
        self.configured_stages = self._STAGES[selection]
        self._current_skipped_stages: frozenset[str] = frozenset()
        self.collection_batch_index = 0
        self._clear_counts()

    @staticmethod
    def _validate_threshold(value: object) -> None:
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError("Success threshold must be finite and in (0, 1].")

    def _clear_counts(self) -> None:
        self._episode_counts = {stage: 0 for stage in sorted(self.configured_stages)}
        self._success_counts = dict(self._episode_counts)
        self._batch_open = False

    @property
    def current_skipped_stages(self) -> frozenset[str]:
        return self._current_skipped_stages

    @property
    def active_stages(self) -> frozenset[str]:
        return self.configured_stages - self._current_skipped_stages

    @property
    def episode_counts(self) -> dict[str, int]:
        return dict(self._episode_counts)

    @property
    def success_counts(self) -> dict[str, int]:
        return dict(self._success_counts)

    def observe(self, record: Mapping[str, object]) -> bool:
        """Observe one completed episode; return whether it was counted.

        The caller owns episode completion/acceptance. A truthy string or an
        unrelated ``success`` field cannot stand in for formal stage success.
        Stage membership is checked even for invalid episodes, since sampling
        a skipped stage is a collection-contract error, not a failed attempt.
        """
        if not isinstance(record, Mapping):
            raise ValueError("Episode record must be a mapping.")
        stage = record.get("training_stage")
        if not isinstance(stage, str) or stage not in self.configured_stages:
            raise ValueError("Episode training_stage is outside the configured selection.")
        if stage in self._current_skipped_stages:
            raise ValueError(f"{stage} is skipped in this collection batch.")
        valid = record.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("Episode valid must be an explicit boolean.")
        if not valid:
            self._batch_open = True
            return False
        success = record.get("stage_success")
        if not isinstance(success, bool):
            raise ValueError("A valid episode requires a boolean stage_success.")
        self._episode_counts[stage] += 1
        self._success_counts[stage] += int(success)
        self._batch_open = True
        return True

    def _decision(self, next_skip: frozenset[str], *, empty: bool = False) -> StageSuccessSkipDecision:
        rates = {stage: (self._success_counts[stage] / total if total else None)
                 for stage, total in self._episode_counts.items()}
        return StageSuccessSkipDecision(
            batch_index=self.collection_batch_index,
            episode_counts=MappingProxyType(dict(self._episode_counts)),
            success_counts=MappingProxyType(dict(self._success_counts)),
            success_rates=MappingProxyType(rates),
            exclude_update_stages=next_skip,
            next_skip_stages=next_skip,
            empty_batch=empty,
        )

    def finish_batch(self) -> StageSuccessSkipDecision:
        """Close an observed batch and establish exactly one batch of skips.

        All valid completed episodes of a stage are included, even ones which
        fail; the caller must not pre-filter its denominator by reward/length.
        No-data stages, including currently skipped ones, cannot requalify.
        """
        if not self.active_stages:
            raise RuntimeError("All stages are skipped; use consume_empty_batch().")
        if not self._batch_open:
            raise RuntimeError("Cannot finish an unobserved collection batch.")
        qualified = frozenset(
            stage for stage, total in self._episode_counts.items()
            if total > 0 and self._success_counts[stage] / total >= self.threshold
        )
        decision = self._decision(qualified)
        self._current_skipped_stages = qualified
        self.collection_batch_index += 1
        self._clear_counts()
        return decision

    def consume_empty_batch(self) -> StageSuccessSkipDecision:
        """Consume an all-stage skip without sampling, worlds or a PPO update."""
        if self.active_stages or self._batch_open:
            raise RuntimeError("An empty skip batch requires every configured stage to be skipped.")
        decision = self._decision(frozenset(), empty=True)
        self._current_skipped_stages = frozenset()
        self.collection_batch_index += 1
        self._clear_counts()
        return decision

    def _require_boundary(self) -> None:
        if self._batch_open:
            raise RuntimeError("Skip-controller state is available only at a safe batch boundary.")

    def state_dict(self) -> dict[str, object]:
        self._require_boundary()
        return {
            "schema": self.SCHEMA,
            "selection": self.selection,
            "threshold": self.threshold,
            "current_skipped_stages": sorted(self._current_skipped_stages),
            "collection_batch_index": self.collection_batch_index,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Atomically restore boundary state after verifying this contract."""
        self._require_boundary()
        if not isinstance(state, Mapping) or set(state) != self._STATE_KEYS:
            raise ValueError("Skip-controller state has missing or unexpected fields.")
        schema = state["schema"]
        if isinstance(schema, bool) or not isinstance(schema, int) or schema != self.SCHEMA:
            raise ValueError("Unsupported skip-controller state schema.")
        if state["selection"] != self.selection:
            raise ValueError("Checkpoint stage selection differs from this controller.")
        threshold = state["threshold"]
        self._validate_threshold(threshold)
        if float(threshold) != self.threshold:
            raise ValueError("Checkpoint success threshold differs from this controller.")
        index = state["collection_batch_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("collection_batch_index must be a nonnegative integer.")
        skipped = state["current_skipped_stages"]
        if (not isinstance(skipped, list)
                or any(not isinstance(stage, str) or stage not in self.configured_stages
                       for stage in skipped)
                or len(set(skipped)) != len(skipped)):
            raise ValueError("current_skipped_stages must be a unique list of configured stages.")
        if index == 0 and skipped:
            raise ValueError("An initial collection batch cannot already contain skipped stages.")
        self._current_skipped_stages = frozenset(skipped)
        self.collection_batch_index = index
        self._clear_counts()
