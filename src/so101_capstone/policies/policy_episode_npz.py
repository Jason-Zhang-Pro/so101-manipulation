"""Persist one completed policy episode as a compressed NPZ archive.

Images are intentionally excluded. The archive contains joint state,
controller actions, rewards, termination flags, and JSON metadata only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .policy_episode_runner import (
    PolicyEpisodeResult,
    PolicyTransition,
)


POLICY_EPISODE_NPZ_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "NPZ metadata contains a non-finite float."
            )
        return value

    if isinstance(value, np.generic):
        return _jsonable(value.item())

    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "NPZ metadata dictionary keys must be strings."
                )
            result[key] = _jsonable(item)

        return result

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    raise TypeError(
        "Unsupported NPZ metadata value type: "
        f"{type(value).__name__}."
    )


def _json_text(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stack_vectors(
    transitions: tuple[PolicyTransition, ...],
    *,
    attribute: str,
    use_next: bool = False,
) -> np.ndarray:
    rows = []

    for transition in transitions:
        observation = (
            transition.next_observation
            if use_next
            else transition.observation
        )
        rows.append(
            np.asarray(
                getattr(observation, attribute),
                dtype=np.float64,
            )
        )

    result = np.stack(rows, axis=0)

    if result.shape != (len(transitions), 6):
        raise ValueError(
            f"{attribute} data must have shape "
            f"({len(transitions)}, 6)."
        )

    return result


def _episode_arrays(
    episode: PolicyEpisodeResult,
    *,
    metadata: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    transitions = episode.transitions
    actions = np.stack(
        [
            transition.action.joint_targets
            for transition in transitions
        ],
        axis=0,
    ).astype(np.float64, copy=False)

    if actions.shape != (len(transitions), 6):
        raise ValueError(
            "Transition actions must have shape "
            f"({len(transitions)}, 6)."
        )

    return {
        "schema_version": np.asarray(
            POLICY_EPISODE_NPZ_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "step_index": np.asarray(
            [item.step_index for item in transitions],
            dtype=np.int64,
        ),
        "observation_joint_positions": _stack_vectors(
            transitions,
            attribute="joint_positions",
        ),
        "observation_joint_velocities": _stack_vectors(
            transitions,
            attribute="joint_velocities",
        ),
        "next_observation_joint_positions": _stack_vectors(
            transitions,
            attribute="joint_positions",
            use_next=True,
        ),
        "next_observation_joint_velocities": _stack_vectors(
            transitions,
            attribute="joint_velocities",
            use_next=True,
        ),
        "action_joint_targets": actions,
        "reward": np.asarray(
            [item.reward for item in transitions],
            dtype=np.float64,
        ),
        "terminated": np.asarray(
            [item.terminated for item in transitions],
            dtype=np.bool_,
        ),
        "truncated": np.asarray(
            [item.truncated for item in transitions],
            dtype=np.bool_,
        ),
        "done": np.asarray(
            [item.done for item in transitions],
            dtype=np.bool_,
        ),
        "initial_joint_positions": np.asarray(
            episode.initial_observation.joint_positions,
            dtype=np.float64,
        ).copy(),
        "initial_joint_velocities": np.asarray(
            episode.initial_observation.joint_velocities,
            dtype=np.float64,
        ).copy(),
        "final_joint_positions": np.asarray(
            episode.final_observation.joint_positions,
            dtype=np.float64,
        ).copy(),
        "final_joint_velocities": np.asarray(
            episode.final_observation.joint_velocities,
            dtype=np.float64,
        ).copy(),
        "total_reward": np.asarray(
            episode.total_reward,
            dtype=np.float64,
        ),
        "episode_terminated": np.asarray(
            episode.terminated,
            dtype=np.bool_,
        ),
        "episode_truncated": np.asarray(
            episode.truncated,
            dtype=np.bool_,
        ),
        "reached_max_steps": np.asarray(
            episode.reached_max_steps,
            dtype=np.bool_,
        ),
        "initial_info_json": np.asarray(
            _json_text(episode.initial_info),
        ),
        "transition_info_json": np.asarray(
            [_json_text(item.info) for item in transitions]
        ),
        "final_info_json": np.asarray(
            _json_text(episode.final_info),
        ),
        "metadata_json": np.asarray(
            _json_text({} if metadata is None else metadata),
        ),
    }


def write_policy_episode_npz(
    episode: PolicyEpisodeResult,
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one completed episode without images."""

    if not isinstance(episode, PolicyEpisodeResult):
        raise TypeError(
            "episode must be a PolicyEpisodeResult."
        )

    destination = Path(output_path).expanduser().resolve()

    if destination.suffix.lower() != ".npz":
        raise ValueError(
            "output_path must end with .npz."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = _episode_arrays(
        episode,
        metadata=metadata,
    )
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".npz",
            prefix=destination.stem + ".",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, **arrays)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


class PolicyEpisodeNpzSink:
    """Buffer transitions and write the finalized episode."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_path = Path(
            output_path
        ).expanduser().resolve()
        self.metadata = (
            {}
            if metadata is None
            else dict(metadata)
        )
        self._transitions: list[PolicyTransition] = []
        self._finalized = False
        self._written_path: Path | None = None

    def reset(self) -> None:
        self._transitions.clear()
        self._finalized = False
        self._written_path = None

    def append(
        self,
        transition: PolicyTransition,
    ) -> None:
        if self._finalized:
            raise RuntimeError(
                "Cannot append after NPZ episode finalization."
            )

        if not isinstance(transition, PolicyTransition):
            raise TypeError(
                "transition must be a PolicyTransition."
            )

        expected = len(self._transitions)

        if transition.step_index != expected:
            raise ValueError(
                "Transition step_index must be contiguous: "
                f"expected {expected}, "
                f"received {transition.step_index}."
            )

        self._transitions.append(transition)

    def finalize(
        self,
        episode: PolicyEpisodeResult,
    ) -> None:
        if self._finalized:
            raise RuntimeError(
                "NPZ episode sink is already finalized."
            )

        if len(self._transitions) != len(
            episode.transitions
        ):
            raise ValueError(
                "Buffered transition count does not match "
                "the finalized episode."
            )

        if not all(
            buffered is finalized
            for buffered, finalized in zip(
                self._transitions,
                episode.transitions,
                strict=True,
            )
        ):
            raise ValueError(
                "Buffered transitions do not match "
                "the finalized episode."
            )

        self._written_path = write_policy_episode_npz(
            episode,
            self.output_path,
            metadata=self.metadata,
        )
        self._finalized = True

    @property
    def transitions(self) -> tuple[PolicyTransition, ...]:
        return tuple(self._transitions)

    def release_transitions(self) -> None:
        """Release archived history while keeping completion and path metadata."""

        self._transitions.clear()

    @property
    def written_path(self) -> Path | None:
        return self._written_path
