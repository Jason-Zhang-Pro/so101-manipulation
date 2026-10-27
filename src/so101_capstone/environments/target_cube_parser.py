"""Parse target-cube manager JSON messages.

The ROS topic ``/target_cube/states`` publishes ``std_msgs/msg/String``.
This module parses the message's ``data`` field without depending on ROS.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .state_cache import (
    TargetCubeSnapshot,
    TargetCubeStateCache,
)


def _readonly_array(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _require_object(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")

    return value


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> float:
    if key not in mapping:
        raise ValueError(
            f"{context} is missing field {key}."
        )

    raw_value = mapping[key]

    if isinstance(raw_value, bool):
        raise ValueError(
            f"{context}.{key} must be numeric."
        )

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context}.{key} must be numeric."
        ) from error

    if not np.isfinite(value):
        raise ValueError(
            f"{context}.{key} must be finite."
        )

    return value


@dataclass(frozen=True)
class ParsedTargetCube:
    """One cube parsed from the manager's JSON message."""

    name: str
    entity_id: int
    position: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class ParsedTargetCubeStates:
    """Validated contents of one `/target_cube/states` message."""

    frame_id: str
    cubes: tuple[ParsedTargetCube, ...]
    source_sequence: int | None = None
    source_sim_time_s: float | None = None

    @property
    def cube_count(self) -> int:
        return len(self.cubes)



def _parse_source_metadata(
    payload: str,
) -> tuple[int | None, float | None]:
    """Parse optional Gazebo sequence and simulation time."""

    try:
        document = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Target cube states payload must be valid JSON."
        ) from error

    if not isinstance(document, dict):
        raise ValueError(
            "Target cube states payload must be a JSON object."
        )

    sequence_value = document.get("sequence")
    sim_time_value = document.get("sim_time_s")

    if sequence_value is None:
        source_sequence = None
    else:
        if (
            isinstance(sequence_value, bool)
            or not isinstance(sequence_value, int)
            or sequence_value < 0
        ):
            raise ValueError(
                "sequence must be a non-negative integer."
            )

        source_sequence = sequence_value

    if sim_time_value is None:
        source_sim_time_s = None
    else:
        if isinstance(sim_time_value, bool):
            raise ValueError(
                "sim_time_s must be a finite "
                "non-negative number."
            )

        try:
            source_sim_time_s = float(sim_time_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "sim_time_s must be a finite "
                "non-negative number."
            ) from error

        if (
            not np.isfinite(source_sim_time_s)
            or source_sim_time_s < 0.0
        ):
            raise ValueError(
                "sim_time_s must be a finite "
                "non-negative number."
            )

    return source_sequence, source_sim_time_s


def parse_target_cube_states_json(
    payload: str,
) -> ParsedTargetCubeStates:
    """Parse and validate `/target_cube/states` String.data."""

    source_sequence, source_sim_time_s = (
        _parse_source_metadata(payload)
    )

    if not isinstance(payload, str) or not payload.strip():
        raise ValueError(
            "Target cube state payload must be a non-empty string."
        )

    try:
        root_value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Target cube state payload is not valid JSON."
        ) from error

    root = _require_object(
        root_value,
        name="Target cube state payload",
    )

    frame_id = root.get("frame_id")

    if frame_id != "world":
        raise ValueError(
            "Target cube frame_id must be 'world'."
        )

    declared_count = root.get("cube_count")

    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count < 0
    ):
        raise ValueError(
            "cube_count must be a non-negative integer."
        )

    raw_cubes = root.get("cubes")

    if not isinstance(raw_cubes, list):
        raise ValueError("cubes must be a JSON array.")

    if declared_count != len(raw_cubes):
        raise ValueError(
            "cube_count does not match the number of cubes."
        )

    parsed_cubes: list[ParsedTargetCube] = []
    seen_names: set[str] = set()

    for index, raw_cube in enumerate(raw_cubes):
        context = f"cubes[{index}]"
        cube = _require_object(
            raw_cube,
            name=context,
        )

        raw_name = cube.get("name")

        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                f"{context}.name must be a non-empty string."
            )

        name = raw_name.strip()

        if name in seen_names:
            raise ValueError(
                f"Duplicate target cube name: {name}."
            )

        seen_names.add(name)

        entity_id = cube.get("entity_id")

        if (
            isinstance(entity_id, bool)
            or not isinstance(entity_id, int)
            or entity_id < 0
        ):
            raise ValueError(
                f"{context}.entity_id must be a "
                "non-negative integer."
            )

        position_object = _require_object(
            cube.get("position"),
            name=f"{context}.position",
        )
        orientation_object = _require_object(
            cube.get("orientation"),
            name=f"{context}.orientation",
        )

        position = _readonly_array(
            [
                _finite_float(
                    position_object,
                    "x",
                    context=f"{context}.position",
                ),
                _finite_float(
                    position_object,
                    "y",
                    context=f"{context}.position",
                ),
                _finite_float(
                    position_object,
                    "z",
                    context=f"{context}.position",
                ),
            ]
        )

        quaternion = np.asarray(
            [
                _finite_float(
                    orientation_object,
                    "x",
                    context=f"{context}.orientation",
                ),
                _finite_float(
                    orientation_object,
                    "y",
                    context=f"{context}.orientation",
                ),
                _finite_float(
                    orientation_object,
                    "z",
                    context=f"{context}.orientation",
                ),
                _finite_float(
                    orientation_object,
                    "w",
                    context=f"{context}.orientation",
                ),
            ],
            dtype=float,
        )

        quaternion_norm = float(
            np.linalg.norm(quaternion)
        )

        if quaternion_norm <= 1e-12:
            raise ValueError(
                f"{context}.orientation quaternion "
                "must have non-zero norm."
            )

        quaternion = quaternion / quaternion_norm
        quaternion.setflags(write=False)

        parsed_cubes.append(
            ParsedTargetCube(
                name=name,
                entity_id=entity_id,
                position=position,
                quaternion_xyzw=quaternion,
            )
        )

    return ParsedTargetCubeStates(
        frame_id=frame_id,
        cubes=tuple(parsed_cubes),
        source_sequence=source_sequence,
        source_sim_time_s=source_sim_time_s,
    )


def select_target_cube(
    states: ParsedTargetCubeStates,
    target_name: str | None = None,
) -> ParsedTargetCube:
    """Select one episode target from parsed cube states.

    When ``target_name`` is omitted, exactly one cube must exist.
    """

    if target_name is None:
        if states.cube_count != 1:
            raise ValueError(
                "Exactly one target cube is required when "
                "target_name is not specified."
            )

        return states.cubes[0]

    requested_name = str(target_name).strip()

    if not requested_name:
        raise ValueError(
            "target_name must not be empty."
        )

    for cube in states.cubes:
        if cube.name == requested_name:
            return cube

    raise ValueError(
        f"Target cube not found: {requested_name}."
    )


def update_target_cube_cache_from_json(
    cache: TargetCubeStateCache,
    payload: str,
    *,
    target_name: str | None = None,
    received_monotonic_s: float | None = None,
) -> TargetCubeSnapshot:
    """Parse one message and store the selected target in the cache."""

    states = parse_target_cube_states_json(payload)
    cube = select_target_cube(
        states,
        target_name=target_name,
    )

    return cache.update(
        name=cube.name,
        position=cube.position,
        quaternion_xyzw=cube.quaternion_xyzw,
        source_sequence=states.source_sequence,
        source_sim_time_s=states.source_sim_time_s,
        received_monotonic_s=received_monotonic_s,
    )
