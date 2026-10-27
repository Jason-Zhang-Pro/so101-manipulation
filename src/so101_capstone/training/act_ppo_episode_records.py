"""Validate and durably record an episode before its Gazebo world is rebuilt.

The JSONL is an audit record of each accepted transition, not an image dataset
or a replacement for the in-memory PPO buffer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from so101_capstone.training.act_ppo import (
    ACT_PPO_SAMPLING_SCHEMA_VERSION,
    CRITIC_POLICY_CONTEXT_SCHEMA_VERSION,
    ActPpoRolloutBuffer,
)


_IMAGE_FIELDS = frozenset({"top_images", "gripper_images"})


def validate_complete_episode(
    buffer: ActPpoRolloutBuffer,
    completed_steps: int,
) -> None:
    """Reject incomplete trajectories before accepting or persisting them."""
    if not isinstance(buffer, ActPpoRolloutBuffer):
        raise TypeError("buffer must be an ActPpoRolloutBuffer.")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
        raise ValueError("completed_steps must be a positive integer.")
    if completed_steps <= 0:
        raise ValueError("completed_steps must be a positive integer.")
    for name, values in vars(buffer).items():
        if isinstance(values, list) and name != "actor_observations":
            if len(values) != completed_steps:
                raise ValueError(
                    f"Incomplete episode: {name} has {len(values)} entries; "
                    f"expected {completed_steps}."
                )
    for index, reference in enumerate(buffer.actor_refs, start=1):
        if (
            isinstance(reference, (bool, np.bool_))
            or not isinstance(reference, (int, np.integer))
            or not 0 <= reference < len(buffer.actor_observations)
        ):
            raise ValueError(f"Invalid actor reference at step {index}: {reference}.")
    if not buffer.episode_end[-1]:
        raise ValueError("The final transition must mark episode_end.")
    if any(buffer.episode_end[:-1]) or any(buffer.terminated[:-1]):
        raise ValueError("An episode cannot contain transitions after its end.")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize episode value of type {type(value).__name__}.")


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_record(path: Path) -> None:
    """Flush a closed record and its directory; propagate any storage failure."""
    path = Path(path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def persist_episode_steps(
    path: Path,
    buffer: ActPpoRolloutBuffer,
    completed_steps: int,
) -> dict[str, Any]:
    """Atomically persist every step's non-image PPO data and sync it to disk.

    A raised exception means the caller must not proceed to a planned world
    rebuild. In particular, a directory-sync error can occur after replacement;
    even then this function does not claim the write completed durably.
    """
    validate_complete_episode(buffer, completed_steps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        name: values
        for name, values in vars(buffer).items()
        if isinstance(values, list)
        and name != "actor_observations"
        and name not in _IMAGE_FIELDS
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for index in range(completed_steps):
                record = {
                    "step_index": index + 1,
                    "sampling_schema_version": ACT_PPO_SAMPLING_SCHEMA_VERSION,
                    "critic_policy_context_schema_version": CRITIC_POLICY_CONTEXT_SCHEMA_VERSION,
                }
                record.update({name: values[index] for name, values in fields.items()})
                record["actor_policy_state"] = buffer.actor_observations[
                    buffer.actor_refs[index]
                ].policy_state
                json.dump(record, stream, default=_json_default, allow_nan=False)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "step_count": completed_steps, "images_included": False}
