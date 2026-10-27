"""Durable accounting of episode failures and unused final PPO transitions.

An environment step is counted only when its successful result reached the
trainer. A request interrupted by a simulator crash is deliberately recorded
as unconfirmed, even if the simulator might have executed it. Planned world
rebuilds after saved episodes are not failure events.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from so101_capstone.training.act_ppo_episode_records import sync_record


_FAILURE_COUNTERS = (
    "discarded_environment_steps",
    "discarded_transition_count",
    "gazebo_crash_discarded_environment_steps",
    "gazebo_crash_discarded_transition_count",
    "gazebo_crash_episode_count",
    "failure_episode_count",
    "unconfirmed_step_count",
    "gazebo_crash_unconfirmed_step_count",
)

_ABORT_DEFAULTS = {
    "aborted_rollout_discarded_transition_count": 0,
    "aborted_rollout_reason": "training_aborted_partial_rollout",
    "interrupted_update_transition_count": 0,
    "interrupted_update_reason": "training_aborted_during_ppo_update",
}


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _empty_counters() -> dict[str, int]:
    return dict.fromkeys(_FAILURE_COUNTERS, 0)


def _failure_event(record: dict[str, Any]) -> dict[str, Any]:
    world = _nonnegative_integer(record["world_index"], "world_index")
    attempt = _nonnegative_integer(record["episode_attempt_index"], "episode_attempt_index")
    completed = _nonnegative_integer(
        record.get("discarded_environment_steps", record.get("failed_step")),
        "discarded_environment_steps",
    )
    buffered = _nonnegative_integer(
        record["discarded_transition_count"], "discarded_transition_count",
    )
    if buffered > completed:
        raise ValueError("Discarded transitions cannot exceed confirmed completed steps.")
    unconfirmed = _nonnegative_integer(record.get("unconfirmed_step_count", 0), "unconfirmed_step_count")
    if unconfirmed > 1:
        raise ValueError("An episode can have at most one unconfirmed step request.")
    category = record.get("failure_category", "unknown")
    if not isinstance(category, str) or not category:
        raise ValueError("failure_category must be a nonempty string.")
    diagnostics = record.get("gazebo_crash") or {}
    if not isinstance(diagnostics, dict):
        raise ValueError("gazebo_crash must be a dictionary when provided.")
    confirmed_crash = diagnostics.get("confirmed") is True
    return {
        "event_id": f"world_{world}:attempt_{attempt}",
        "world_index": world,
        "episode_attempt_index": attempt,
        "failure_category": category,
        "failure_record_path": str(record.get(
            "failure_record_path",
            f"episode_failures/attempt_{attempt:06d}_world_{world:02d}.json",
        )),
        "gazebo_crash_confirmed": confirmed_crash,
        "discarded_environment_steps": completed,
        "discarded_transition_count": buffered,
        "gazebo_crash_discarded_environment_steps": completed if confirmed_crash else 0,
        "gazebo_crash_discarded_transition_count": buffered if confirmed_crash else 0,
        "gazebo_crash_episode_count": int(confirmed_crash),
        "failure_episode_count": 1,
        "unconfirmed_step_count": unconfirmed,
        "gazebo_crash_unconfirmed_step_count": unconfirmed if confirmed_crash else 0,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Persist before reporting success, including directory metadata syncing."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_record(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DiscardedStepStats:
    """Persist ``discarded_steps.json`` beside the training's other outputs.

    ``discarded_environment_steps`` covers confirmed steps in failed episodes.
    ``discarded_transition_count`` additionally includes unused PPO transitions
    at normal or abnormal shutdown; their episodes remain in the episode logs.
    Transitions involved in an interrupted update have uncertain usage and are
    reported separately, without adding them to the confirmed discard count.
    Crash subtotals require explicit evidence (``gazebo_crash.confirmed=True``).
    Failure events retain references to the detailed records for reconstruction.
    """

    def __init__(self, output_directory: Path, world_indices: Iterable[int] = ()) -> None:
        self.path = Path(output_directory) / "discarded_steps.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        worlds = sorted({_nonnegative_integer(index, "world_index") for index in world_indices})
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("schema_version") != 1:
                raise ValueError("Unsupported discarded-step accounting schema.")
            self._state = saved
            self._events = {event["event_id"]: event for event in saved["failure_events"]}
            if len(self._events) != len(saved["failure_events"]):
                raise ValueError("Duplicate failure events in discarded-step accounting.")
            if any(key not in saved for key in _ABORT_DEFAULTS):
                candidate = self.snapshot()
                for key, default in _ABORT_DEFAULTS.items():
                    candidate.setdefault(key, default)
                self._commit(candidate)
        else:
            self._events: dict[str, dict[str, Any]] = {}
            self._state = {
                "schema_version": 1,
                "updated_at_utc": None,
                **_empty_counters(),
                "failure_discarded_transition_count": 0,
                "final_partial_rollout_discarded_transition_count": 0,
                "final_partial_rollout_reason": "training_end_partial_rollout",
                **_ABORT_DEFAULTS,
                "by_world": {str(index): _empty_counters() for index in worlds},
                "by_failure_category": {},
                "failure_events": [],
            }
            self._commit(deepcopy(self._state))

    def snapshot(self) -> dict[str, Any]:
        """Return a detached copy; callers cannot change accumulated counts."""
        return deepcopy(self._state)

    def _commit(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate["discarded_transition_count"] = (
            candidate["failure_discarded_transition_count"]
            + candidate["final_partial_rollout_discarded_transition_count"]
            + candidate["aborted_rollout_discarded_transition_count"]
        )
        candidate["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(self.path, candidate)
        # A storage failure must not advance in-memory totals or deduplication.
        self._state = candidate
        self._events = {event["event_id"]: event for event in candidate["failure_events"]}
        return self.snapshot()

    def record_failure(self, record: dict[str, Any]) -> dict[str, Any]:
        event = _failure_event(record)
        previous = self._events.get(event["event_id"])
        if previous is not None:
            if previous != event:
                raise ValueError(f"Conflicting failure event: {event['event_id']}.")
            return self.snapshot()
        candidate = self.snapshot()
        aggregates = (
            candidate,
            candidate["by_world"].setdefault(str(event["world_index"]), _empty_counters()),
            candidate["by_failure_category"].setdefault(event["failure_category"], _empty_counters()),
        )
        for aggregate in aggregates:
            for key in _FAILURE_COUNTERS:
                aggregate[key] += event[key]
        candidate["failure_discarded_transition_count"] += event["discarded_transition_count"]
        candidate["failure_events"].append(event)
        return self._commit(candidate)

    def record_final_partial(self, count: int) -> dict[str, Any]:
        """Set (rather than increment) the final unused rollout's length.

        This is separate from failed episodes, has no crash attribution, and is
        idempotent so repeated finalization cannot double-count transitions.
        """
        count = _nonnegative_integer(count, "final_partial_rollout_discarded_transition_count")
        if self._state["final_partial_rollout_discarded_transition_count"] == count:
            return self.snapshot()
        candidate = self.snapshot()
        candidate["final_partial_rollout_discarded_transition_count"] = count
        return self._commit(candidate)

    def record_aborted_rollout(
        self, count: int, *, update_in_progress: bool = False,
    ) -> dict[str, Any]:
        """Record buffered transitions affected by abnormal training shutdown.

        Without an active PPO update these transitions are confirmed unused.
        During an update some optimizer steps may already have used them, so
        only a separate uncertain-usage count is recorded. Both are idempotent
        final counts, independent of crash attribution and individual worlds.
        """
        if not isinstance(update_in_progress, bool):
            raise ValueError("update_in_progress must be a boolean.")
        key = (
            "interrupted_update_transition_count" if update_in_progress
            else "aborted_rollout_discarded_transition_count"
        )
        count = _nonnegative_integer(count, key)
        if self._state[key] == count:
            return self.snapshot()
        candidate = self.snapshot()
        candidate[key] = count
        return self._commit(candidate)
