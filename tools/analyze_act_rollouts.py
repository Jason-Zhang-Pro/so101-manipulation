#!/usr/bin/env python3
"""Diagnose saved SO-101 ACT rollout NPZ files without changing runtime code.

The analyzer is deliberately schema-tolerant. It inventories every NPZ field,
parses per-step JSON metadata when present, resolves known aliases, and clearly
marks conclusions that cannot be established from image-free rollout archives.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


DEFAULT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_ARRAY_ALIASES: dict[str, tuple[str, ...]] = {
    "step_index": (
        "step_index",
        "episode_step",
    ),
    "joint_positions": (
        "observation_joint_positions",
        "observation_joint_position",
        "joint_positions",
        "joint_position",
        "observation_state",
    ),
    "joint_velocities": (
        "observation_joint_velocities",
        "observation_joint_velocity",
        "joint_velocities",
        "joint_velocity",
    ),
    "next_joint_positions": (
        "next_observation_joint_positions",
        "next_observation_joint_position",
        "next_joint_positions",
        "next_joint_position",
    ),
    "next_joint_velocities": (
        "next_observation_joint_velocities",
        "next_observation_joint_velocity",
        "next_joint_velocities",
        "next_joint_velocity",
    ),
    "executed_action": (
        "action_joint_targets",
        "executed_action",
        "controller_command_rad",
        "controller_targets_rad",
        "proposed_controller_command_rad",
        "action",
    ),
    "raw_policy_action": (
        "raw_policy_positions",
        "raw_policy_action",
        "policy_raw_action",
        "raw_action",
    ),
    "policy_bound_clipped_action": (
        "clipped_policy_positions",
        "clipped_policy_action",
        "policy_clipped_action",
    ),
    "mapped_gazebo_action": (
        "mapped_gazebo_positions_rad",
        "mapped_gazebo_action_rad",
        "mapped_action_rad",
    ),
    "reward": (
        "reward",
        "rewards",
    ),
    "terminated": (
        "terminated",
    ),
    "truncated": (
        "truncated",
    ),
    "done": (
        "done",
    ),
}

_SERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "distance_m": (
        "distance_m",
        "privileged_state.distance_m",
        "reward_result.distance_m",
    ),
    "progress_m": (
        "progress_m",
    ),
    "z_error": (
        "z_target_error_deg",
        "z_axis_error_deg",
        "z_error_deg",
        "z_error",
    ),
    "gripper_open_norm": (
        "gripper_open_norm",
        "gripper_open_normalized",
        "gripper_opening_norm",
    ),
    "success_ready": (
        "success_ready_now",
        "success_ready",
    ),
    "success": (
        "success",
        "success_ever",
    ),
    "target_active": (
        "target_active",
        "target_spawned",
    ),
    "camera_pair_reused": (
        "camera_pair_reused",
    ),
    "consecutive_camera_reuse_steps": (
        "consecutive_camera_reuse_steps",
        "camera_reuse_run_length",
    ),
    "camera_age_s": (
        "camera_lag_from_step_end_s",
        "camera_age_s",
        "image_age_s",
        "max_image_age_s",
    ),
    "camera_callback_grace_used": (
        "camera_callback_grace_used",
    ),
    "max_camera_lag_from_step_end_s": ("max_camera_lag_from_step_end_s",),
    "max_camera_reuse_lag_s": ("max_camera_reuse_lag_s",),
    "top_image_sequence": (
        "top_image_sequence",
    ),
    "gripper_image_sequence": (
        "gripper_image_sequence",
    ),
    "camera_skew_s": (
        "camera_skew_s",
    ),
    "raw_policy_action": _ARRAY_ALIASES["raw_policy_action"],
    "policy_bound_clipped_action": _ARRAY_ALIASES[
        "policy_bound_clipped_action"
    ],
    "mapped_gazebo_action": _ARRAY_ALIASES["mapped_gazebo_action"],
    "executed_action": _ARRAY_ALIASES["executed_action"],
}

_JSON_FIELD_HINTS = (
    "json",
    "info",
    "metadata",
)


@dataclass(frozen=True)
class ResolvedSeries:
    """One resolved diagnostic series and its source field."""

    name: str
    source: str
    values: np.ndarray


@dataclass
class EpisodeArchive:
    """Loaded NPZ contents plus parsed JSON-derived series."""

    path: Path
    arrays: dict[str, np.ndarray]
    json_values: dict[str, Any]
    per_step_series: dict[str, np.ndarray]
    metadata: dict[str, Any]
    step_count: int


@dataclass(frozen=True)
class MatrixStats:
    """Per-column statistics for a time-by-joint matrix."""

    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    value_range: np.ndarray
    std: np.ndarray
    total_variation: np.ndarray
    net_change: np.ndarray
    unchanged_ratio: np.ndarray
    longest_unchanged_run: np.ndarray


class DiagnosticError(RuntimeError):
    """Raised when a rollout cannot be safely diagnosed."""


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _scalar_text(value: np.ndarray) -> str | None:
    array = np.asarray(value)

    if array.shape == ():
        item = array.item()
    elif array.shape == (1,):
        item = array[0]
    else:
        return None

    if isinstance(item, bytes):
        return item.decode("utf-8")

    if isinstance(item, str):
        return item

    return None


def _parse_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_json_array(array: np.ndarray) -> Any:
    scalar = _scalar_text(array)

    if scalar is not None:
        return _parse_json_text(scalar)

    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        return None

    parsed = []

    for item in array:
        if isinstance(item, bytes):
            text = item.decode("utf-8")
        else:
            text = str(item)

        value = _parse_json_text(text)

        if value is None:
            return None

        parsed.append(value)

    return parsed


def _flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_key = str(key)
            path = child_key if not prefix else f"{prefix}.{child_key}"
            result.update(_flatten(child, prefix=path))
        return result

    result[prefix] = value
    return result


def _coerce_series(values: Sequence[Any]) -> np.ndarray | None:
    non_missing = [value for value in values if value is not None]

    if not non_missing:
        return None

    if all(isinstance(value, (bool, np.bool_)) for value in non_missing):
        return np.array(
            [False if value is None else bool(value) for value in values],
            dtype=bool,
        )

    scalar_numeric = all(
        isinstance(value, (bool, int, float, np.number))
        for value in non_missing
    )

    if scalar_numeric:
        return np.array(
            [np.nan if value is None else float(value) for value in values],
            dtype=float,
        )

    shapes: set[tuple[int, ...]] = set()
    converted: list[np.ndarray | None] = []

    for value in values:
        if value is None:
            converted.append(None)
            continue

        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None

        if array.ndim == 0 or not np.all(np.isfinite(array)):
            return None

        shapes.add(array.shape)
        converted.append(array)

    if len(shapes) != 1:
        return None

    shape = next(iter(shapes))
    result = np.full((len(values),) + shape, np.nan, dtype=float)

    for index, array in enumerate(converted):
        if array is not None:
            result[index] = array

    return result


def _extract_per_step_series(
    parsed_json: Mapping[str, Any],
    *,
    step_count: int,
) -> dict[str, np.ndarray]:
    flattened_rows: list[dict[str, Any]] = []

    for field_name, value in parsed_json.items():
        if not isinstance(value, list) or len(value) != step_count:
            continue

        if not all(isinstance(row, Mapping) for row in value):
            continue

        if not flattened_rows:
            flattened_rows = [{} for _ in range(step_count)]

        for row_index, row in enumerate(value):
            for path, child in _flatten(row).items():
                full_path = f"{field_name}.{path}" if path else field_name
                flattened_rows[row_index][full_path] = child

    if not flattened_rows:
        return {}

    paths = sorted(
        {
            path
            for row in flattened_rows
            for path in row
        }
    )
    result: dict[str, np.ndarray] = {}

    for path in paths:
        values = [row.get(path) for row in flattened_rows]
        series = _coerce_series(values)

        if series is not None:
            result[path] = series

    return result


def _infer_step_count(arrays: Mapping[str, np.ndarray]) -> int:
    preferred = (
        "step_index",
        "reward",
        "action_joint_targets",
        "observation_joint_positions",
    )

    for key in preferred:
        if key in arrays and arrays[key].ndim >= 1:
            return int(arrays[key].shape[0])

    candidates = [
        int(array.shape[0])
        for array in arrays.values()
        if array.ndim >= 1 and array.shape[0] > 1
    ]

    if not candidates:
        raise DiagnosticError(
            "Could not infer a transition count from the NPZ arrays."
        )

    counts, frequencies = np.unique(candidates, return_counts=True)
    return int(counts[int(np.argmax(frequencies))])


def _merge_metadata(parsed_json: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for field_name, value in parsed_json.items():
        normalised = _normalise_name(field_name)

        if "metadata" in normalised and isinstance(value, Mapping):
            metadata.update(value)

    return metadata


def load_episode_archive(path: str | Path) -> EpisodeArchive:
    """Load one rollout archive without enabling object-pickle loading."""

    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"NPZ file not found: {resolved}")

    with np.load(resolved, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}

    if not arrays:
        raise DiagnosticError(f"NPZ contains no fields: {resolved}")

    step_count = _infer_step_count(arrays)
    parsed_json: dict[str, Any] = {}

    for key, array in arrays.items():
        normalised = _normalise_name(key)
        hinted = any(hint in normalised for hint in _JSON_FIELD_HINTS)

        if not hinted and array.dtype.kind not in {"U", "S"}:
            continue

        value = _parse_json_array(array)

        if value is not None:
            parsed_json[key] = value

    return EpisodeArchive(
        path=resolved,
        arrays=arrays,
        json_values=parsed_json,
        per_step_series=_extract_per_step_series(
            parsed_json,
            step_count=step_count,
        ),
        metadata=_merge_metadata(parsed_json),
        step_count=step_count,
    )


def _candidate_match(
    available: Iterable[str],
    aliases: Sequence[str],
) -> str | None:
    available_list = list(available)
    normalised_available = {
        name: _normalise_name(name)
        for name in available_list
    }

    for alias in aliases:
        alias_normalised = _normalise_name(alias)

        for name, normalised in normalised_available.items():
            if normalised == alias_normalised:
                return name

    for alias in aliases:
        alias_normalised = _normalise_name(alias)

        for name, normalised in normalised_available.items():
            if normalised.endswith(alias_normalised):
                return name

    return None


def resolve_array(
    episode: EpisodeArchive,
    logical_name: str,
) -> ResolvedSeries | None:
    aliases = _ARRAY_ALIASES[logical_name]
    key = _candidate_match(episode.arrays, aliases)

    if key is not None:
        array = np.asarray(episode.arrays[key])

        if array.ndim >= 1 and array.shape[0] == episode.step_count:
            return ResolvedSeries(logical_name, key, array)

    series_key = _candidate_match(episode.per_step_series, aliases)

    if series_key is not None:
        return ResolvedSeries(
            logical_name,
            series_key,
            episode.per_step_series[series_key],
        )

    return None


def resolve_info_series(
    episode: EpisodeArchive,
    logical_name: str,
) -> ResolvedSeries | None:
    aliases = _SERIES_ALIASES[logical_name]
    key = _candidate_match(episode.per_step_series, aliases)

    if key is not None:
        return ResolvedSeries(
            logical_name,
            key,
            episode.per_step_series[key],
        )

    array_key = _candidate_match(episode.arrays, aliases)

    if array_key is not None:
        array = np.asarray(episode.arrays[array_key])

        if array.ndim >= 1 and array.shape[0] == episode.step_count:
            return ResolvedSeries(logical_name, array_key, array)

    return None


def _as_time_matrix(
    series: ResolvedSeries | None,
    *,
    expected_width: int | None = None,
) -> np.ndarray | None:
    if series is None:
        return None

    array = np.asarray(series.values, dtype=float)

    if array.ndim == 1:
        array = array[:, None]

    if array.ndim != 2:
        return None

    if expected_width is not None and array.shape[1] != expected_width:
        return None

    return array


def _as_scalar_series(series: ResolvedSeries | None) -> np.ndarray | None:
    if series is None:
        return None

    array = np.asarray(series.values)

    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]

    if array.ndim != 1:
        return None

    return array


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0

    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def matrix_stats(
    matrix: np.ndarray,
    *,
    unchanged_tolerance: float,
) -> MatrixStats:
    """Compute robust per-column motion and constancy statistics."""

    values = np.asarray(matrix, dtype=float)

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("matrix must have shape (steps, columns).")

    finite = np.isfinite(values)
    safe = np.where(finite, values, np.nan)
    differences = np.diff(safe, axis=0)
    absolute_differences = np.abs(differences)
    unchanged = absolute_differences <= unchanged_tolerance

    with np.errstate(all="ignore"):
        minimum = np.nanmin(safe, axis=0)
        maximum = np.nanmax(safe, axis=0)
        mean = np.nanmean(safe, axis=0)
        std = np.nanstd(safe, axis=0)
        total_variation = np.nansum(absolute_differences, axis=0)

    if values.shape[0] >= 2:
        net_change = safe[-1] - safe[0]
        unchanged_ratio = np.nanmean(unchanged.astype(float), axis=0)
        longest = np.array(
            [
                _longest_true_run(unchanged[:, index])
                for index in range(values.shape[1])
            ],
            dtype=int,
        )
    else:
        net_change = np.zeros(values.shape[1], dtype=float)
        unchanged_ratio = np.ones(values.shape[1], dtype=float)
        longest = np.zeros(values.shape[1], dtype=int)

    return MatrixStats(
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        value_range=maximum - minimum,
        std=std,
        total_variation=total_variation,
        net_change=net_change,
        unchanged_ratio=unchanged_ratio,
        longest_unchanged_run=longest,
    )


def _stats_dict(stats: MatrixStats, names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for index, name in enumerate(names):
        result[name] = {
            "min": float(stats.minimum[index]),
            "max": float(stats.maximum[index]),
            "mean": float(stats.mean[index]),
            "range": float(stats.value_range[index]),
            "std": float(stats.std[index]),
            "total_variation": float(stats.total_variation[index]),
            "net_change": float(stats.net_change[index]),
            "unchanged_ratio": float(stats.unchanged_ratio[index]),
            "longest_unchanged_run": int(
                stats.longest_unchanged_run[index]
            ),
        }

    return result


def _find_joint_names(
    episode: EpisodeArchive,
    width: int,
    configured: Sequence[str],
) -> tuple[str, ...]:
    flattened = _flatten(episode.metadata)

    for key, value in flattened.items():
        normalised = _normalise_name(key)

        if not (
            normalised.endswith("jointnames")
            or normalised.endswith("jointorder")
        ):
            continue

        if isinstance(value, Sequence) and not isinstance(value, str):
            names = tuple(str(item) for item in value)

            if len(names) == width:
                return names

    if len(configured) == width:
        return tuple(configured)

    return tuple(f"joint_{index}" for index in range(width))


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]

    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "final": None,
        }

    final_finite = finite[-1]
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "final": float(final_finite),
    }


def _phase_summary(
    start: int,
    stop: int,
    *,
    reward: np.ndarray | None,
    joint_positions: np.ndarray | None,
    executed_action: np.ndarray | None,
    distance: np.ndarray | None,
    gripper_open: np.ndarray | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "start_step": int(start),
        "stop_step_exclusive": int(stop),
        "step_count": int(max(0, stop - start)),
    }

    if reward is not None:
        phase = reward[start:stop]
        result["reward_sum"] = float(np.nansum(phase))
        result["reward_mean"] = float(np.nanmean(phase))

    if joint_positions is not None and stop > start:
        phase = joint_positions[start:stop]
        result["joint_total_variation"] = np.nansum(
            np.abs(np.diff(phase, axis=0)),
            axis=0,
        ).tolist()
        result["joint_range"] = (
            np.nanmax(phase, axis=0) - np.nanmin(phase, axis=0)
        ).tolist()

    if executed_action is not None and stop > start:
        phase = executed_action[start:stop]
        result["executed_action_std"] = np.nanstd(
            phase,
            axis=0,
        ).tolist()
        result["executed_action_range"] = (
            np.nanmax(phase, axis=0) - np.nanmin(phase, axis=0)
        ).tolist()

    if distance is not None:
        result["distance_m"] = _finite_summary(distance[start:stop])

        finite_phase = distance[start:stop]
        finite_phase = finite_phase[np.isfinite(finite_phase)]

        if finite_phase.size >= 2:
            result["distance_change_m"] = float(
                finite_phase[-1] - finite_phase[0]
            )

    if gripper_open is not None:
        result["gripper_open_norm"] = _finite_summary(
            gripper_open[start:stop]
        )

    return result


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)

    if np.count_nonzero(mask) < 3:
        return None

    a = a[mask]
    b = b[mask]

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def _clipping_report(
    *,
    joint_names: Sequence[str],
    current_positions: np.ndarray | None,
    executed_action: np.ndarray | None,
    raw_policy: np.ndarray | None,
    policy_bound_clipped: np.ndarray | None,
    mapped_gazebo: np.ndarray | None,
    max_joint_delta_rad: float,
    tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "max_joint_delta_rad": float(max_joint_delta_rad),
        "exact_delta_clipping_available": False,
        "method": "unavailable",
    }

    if current_positions is None or executed_action is None:
        result["reason"] = (
            "Current joint positions or executed controller actions are "
            "missing."
        )
        return result

    if mapped_gazebo is not None and mapped_gazebo.shape == executed_action.shape:
        clipped = np.abs(mapped_gazebo - executed_action) > tolerance
        result["exact_delta_clipping_available"] = True
        result["method"] = "mapped_gazebo_vs_executed"
    else:
        delta = np.abs(executed_action - current_positions)
        clipped = np.isclose(
            delta,
            max_joint_delta_rad,
            atol=max(tolerance, 1e-6),
            rtol=0.0,
        )
        result["method"] = "executed_delta_boundary_inference"
        result["warning"] = (
            "This is a strong boundary-hit inference, not an exact proof. "
            "Exact max-delta clipping requires the pre-delta mapped Gazebo "
            "target for every step."
        )

    counts = np.sum(clipped, axis=0)
    ratios = np.mean(clipped, axis=0)
    result["per_joint"] = {
        name: {
            "count": int(counts[index]),
            "ratio": float(ratios[index]),
        }
        for index, name in enumerate(joint_names)
    }
    result["any_joint_step_count"] = int(np.count_nonzero(np.any(clipped, axis=1)))
    result["any_joint_step_ratio"] = float(np.mean(np.any(clipped, axis=1)))

    if raw_policy is not None and policy_bound_clipped is not None:
        if raw_policy.shape == policy_bound_clipped.shape:
            bound_clipped = np.abs(raw_policy - policy_bound_clipped) > tolerance
            result["policy_bound_clipping"] = {
                "per_joint": {
                    name: {
                        "count": int(np.sum(bound_clipped[:, index])),
                        "ratio": float(np.mean(bound_clipped[:, index])),
                    }
                    for index, name in enumerate(joint_names)
                },
                "any_joint_step_ratio": float(
                    np.mean(np.any(bound_clipped, axis=1))
                ),
            }

    return result


def _archived_camera_limit(episode: EpisodeArchive, name: str) -> np.ndarray:
    """Use recorded settings; an old archive cannot prove today's limits."""
    limits = np.full(episode.step_count, np.nan, dtype=float)
    metadata = _flatten(episode.metadata)
    key = _candidate_match(metadata, (name,))
    if key is not None:
        try:
            value = float(metadata[key])
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value) and value >= 0:
            limits.fill(value)
    series = _as_scalar_series(resolve_info_series(episode, name))
    if series is not None:
        values = np.asarray(series, dtype=float)
        valid = np.isfinite(values) & (values >= 0)
        limits[valid] = values[valid]
    return limits


def _camera_report(episode: EpisodeArchive) -> dict[str, Any]:
    reused_series = _as_scalar_series(
        resolve_info_series(episode, "camera_pair_reused")
    )
    consecutive_series = _as_scalar_series(
        resolve_info_series(episode, "consecutive_camera_reuse_steps")
    )
    age_series = _as_scalar_series(resolve_info_series(episode, "camera_age_s"))
    grace_series = _as_scalar_series(
        resolve_info_series(episode, "camera_callback_grace_used")
    )

    result: dict[str, Any] = {}

    if reused_series is not None:
        reused = np.asarray(reused_series, dtype=bool)
        result["reuse_count"] = int(np.count_nonzero(reused))
        result["reuse_ratio"] = float(np.mean(reused))
        result["computed_max_consecutive_reuse"] = _longest_true_run(reused)

    if consecutive_series is not None:
        finite = np.asarray(consecutive_series, dtype=float)
        finite = finite[np.isfinite(finite)]
        result["reported_max_consecutive_reuse"] = (
            int(np.max(finite)) if finite.size else None
        )

    if age_series is not None:
        result["image_age_s"] = _finite_summary(age_series)
        ages = np.asarray(age_series, dtype=float)
        limits = np.full(ages.shape, np.nan)
        if reused_series is not None:
            reuse_flags = np.asarray(reused_series, dtype=float)
            fresh_limits = _archived_camera_limit(
                episode, "max_camera_lag_from_step_end_s"
            )
            reuse_limits = _archived_camera_limit(episode, "max_camera_reuse_lag_s")
            limits[reuse_flags == 0] = fresh_limits[reuse_flags == 0]
            limits[reuse_flags == 1] = reuse_limits[reuse_flags == 1]
        checked = np.isfinite(ages) & np.isfinite(limits)
        result["age_limit_checked_step_count"] = int(np.count_nonzero(checked))
        result["age_over_limit_count"] = int(np.count_nonzero(
            checked & (ages > limits + 1e-9)
        ))
        result["age_limit_check_available"] = bool(np.any(checked))
        if not np.any(checked):
            result["age_limit_check_reason"] = (
                "Archived camera limits or reuse flags are missing; "
                "no configured-limit violation can be established."
            )

    if grace_series is not None:
        grace = np.asarray(grace_series, dtype=bool)
        result["callback_grace_used_count"] = int(np.count_nonzero(grace))
        result["callback_grace_used_ratio"] = float(np.mean(grace))

    if not result:
        result["available"] = False
        result["reason"] = "No camera reuse or image-age fields were found."
    else:
        result["available"] = True

    return result


def _action_behaviour_report(
    matrix: np.ndarray | None,
    names: Sequence[str],
    *,
    unchanged_tolerance: float,
) -> dict[str, Any]:
    if matrix is None:
        return {
            "available": False,
            "reason": "Action field not present in the archive.",
        }

    stats = matrix_stats(matrix, unchanged_tolerance=unchanged_tolerance)
    nearly_constant = (
        stats.unchanged_ratio >= 0.95
    ) | (stats.value_range <= max(unchanged_tolerance * 10.0, 1e-4))

    return {
        "available": True,
        "per_joint": _stats_dict(stats, names),
        "nearly_constant_joints": [
            names[index]
            for index, value in enumerate(nearly_constant)
            if value
        ],
        "all_joints_nearly_constant": bool(np.all(nearly_constant)),
    }


def _build_anomalies(
    *,
    step_count: int,
    joint_positions: np.ndarray | None,
    executed_action: np.ndarray | None,
    reward: np.ndarray | None,
    clipping: Mapping[str, Any],
    action_behaviour: Mapping[str, Any],
    distance: np.ndarray | None,
    success_ready: np.ndarray | None,
    success: np.ndarray | None,
    done: np.ndarray | None,
    camera: Mapping[str, Any],
) -> list[str]:
    anomalies: list[str] = []

    for label, array in (
        ("joint_positions", joint_positions),
        ("executed_action", executed_action),
        ("reward", reward),
    ):
        if array is not None and not np.all(np.isfinite(array)):
            anomalies.append(f"{label} contains non-finite values.")

    if action_behaviour.get("all_joints_nearly_constant"):
        anomalies.append(
            "Executed ACT commands are nearly constant across all joints."
        )

    clipping_ratio = clipping.get("any_joint_step_ratio")

    if isinstance(clipping_ratio, (float, int)) and clipping_ratio >= 0.5:
        anomalies.append(
            "At least one joint hits the 0.03 rad delta boundary on 50% or "
            "more of policy steps."
        )

    if distance is not None:
        finite = distance[np.isfinite(distance)]

        if finite.size >= 2 and finite[-1] >= finite[0] - 1e-4:
            anomalies.append(
                "Target distance does not show net improvement over the "
                "available target-visible interval."
            )

    if success_ready is not None and done is not None:
        ready = np.asarray(success_ready, dtype=bool)
        done_bool = np.asarray(done, dtype=bool)

        if np.any(ready & ~done_bool):
            anomalies.append(
                "success-ready is true on at least one non-done transition."
            )

    if success is not None and done is not None:
        success_bool = np.asarray(success, dtype=bool)
        done_bool = np.asarray(done, dtype=bool)

        if np.any(success_bool & ~done_bool):
            anomalies.append(
                "success is true on at least one transition where done is false."
            )

    if camera.get("age_over_limit_count", 0) > 0:
        anomalies.append(
            "At least one archived image age exceeds its recorded fresh-frame "
            "or reused-frame limit."
        )

    max_reuse = camera.get("reported_max_consecutive_reuse")

    if max_reuse is None:
        max_reuse = camera.get("computed_max_consecutive_reuse")

    if isinstance(max_reuse, int) and max_reuse > 6:
        anomalies.append(
            "Archived camera reuse exceeds the configured six-step limit."
        )

    if step_count == 0:
        anomalies.append("The episode contains zero transitions.")

    return anomalies


def _classify_failure(
    *,
    clipping: Mapping[str, Any],
    action_behaviour: Mapping[str, Any],
    phase_comparison: Mapping[str, Any],
    distance: np.ndarray | None,
    success_ready: np.ndarray | None,
    success: np.ndarray | None,
    done: np.ndarray | None,
    raw_policy_available: bool,
    image_data_present: bool,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(category: str, level: str, evidence: str) -> None:
        findings.append(
            {
                "category": category,
                "assessment": level,
                "evidence": evidence,
            }
        )

    add(
        "observation_format_mismatch",
        "not_demonstrated",
        "The NPZ can validate archived joint-state shape and finiteness, but "
        "cannot prove that live ACT normalization/calibration matched the "
        "training dataset.",
    )

    if image_data_present:
        add(
            "image_order_or_preprocessing_mismatch",
            "inspectable",
            "Image-like fields exist and should be checked separately.",
        )
    else:
        add(
            "image_order_or_preprocessing_mismatch",
            "not_assessable_from_npz",
            "The rollout archive contains no images, so camera ordering, RGB "
            "channel order, crop/distortion, and normalization cannot be "
            "verified from this file.",
        )

    if raw_policy_available:
        add(
            "action_semantics_mismatch",
            "inspect_raw_mapping",
            "Raw policy outputs are available; compare policy-unit bounds, "
            "mapped Gazebo targets, and executed radian commands.",
        )
    else:
        add(
            "action_semantics_mismatch",
            "not_fully_assessable",
            "Only executed controller targets were found. Raw policy-unit "
            "outputs are required to validate the complete absolute-action "
            "mapping.",
        )

    clipping_ratio = clipping.get("any_joint_step_ratio")

    if isinstance(clipping_ratio, (float, int)):
        if clipping_ratio >= 0.5:
            level = "strong_candidate"
        elif clipping_ratio >= 0.2:
            level = "possible_contributor"
        else:
            level = "not_supported"

        add(
            "action_clipping_too_strong",
            level,
            f"Any-joint 0.03 rad boundary/clipping ratio is "
            f"{float(clipping_ratio):.1%}; method={clipping.get('method')}.",
        )
    else:
        add(
            "action_clipping_too_strong",
            "not_assessable",
            "Required current-state/action fields are missing.",
        )

    post = phase_comparison.get("post_target", {})
    distance_change = post.get("distance_change_m")

    if isinstance(distance_change, (float, int)):
        if distance_change < -0.01:
            generalisation_level = "policy_responds_but_task_incomplete"
        else:
            generalisation_level = "strong_candidate"
        distance_text = f"post-target distance change={distance_change:.6f} m"
    else:
        generalisation_level = "possible"
        distance_text = "post-target distance series unavailable"

    if action_behaviour.get("all_joints_nearly_constant"):
        generalisation_level = "strong_candidate_or_upstream_input_issue"

    add(
        "policy_generalisation_insufficient",
        generalisation_level,
        distance_text
        + "; executed action constancy="
        + str(action_behaviour.get("all_joints_nearly_constant")),
    )

    pre = phase_comparison.get("pre_target", {})
    pre_variation = np.asarray(pre.get("joint_total_variation", []), dtype=float)
    post_variation = np.asarray(post.get("joint_total_variation", []), dtype=float)

    if pre_variation.size and post_variation.size:
        pre_total = float(np.nansum(pre_variation))
        post_total = float(np.nansum(post_variation))
        if pre_total > 0.05 and post_total <= pre_total * 0.75:
            timing_level = "plausible_contributor"
        else:
            timing_level = "not_proven"
        timing_evidence = (
            f"pre-target joint variation={pre_total:.6f}, "
            f"post-target variation={post_total:.6f}."
        )
    else:
        timing_level = "not_assessable"
        timing_evidence = "Phase-specific joint motion is unavailable."

    add(
        "target_appearance_timing_distribution_shift",
        timing_level,
        timing_evidence
        + " Training-time target-visibility timing is not stored in the NPZ, "
        "so a distribution shift cannot be confirmed from rollout data alone.",
    )

    reward_issue = False
    evidence_parts: list[str] = []

    if success_ready is not None and done is not None:
        mismatch = np.asarray(success_ready, dtype=bool) & ~np.asarray(done, dtype=bool)
        if np.any(mismatch):
            reward_issue = True
            evidence_parts.append("success-ready occurred without done")

    if success is not None and done is not None:
        mismatch = np.asarray(success, dtype=bool) & ~np.asarray(done, dtype=bool)
        if np.any(mismatch):
            reward_issue = True
            evidence_parts.append("success occurred without done")

    if distance is None:
        evidence_parts.append("distance unavailable")

    add(
        "reward_or_success_definition_problem",
        "strong_candidate" if reward_issue else "not_supported_by_available_fields",
        "; ".join(evidence_parts) if evidence_parts else (
            "No archived success/done inconsistency was detected."
        ),
    )

    return findings


def _inventory(episode: EpisodeArchive) -> list[dict[str, Any]]:
    return [
        {
            "name": key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        for key, array in sorted(episode.arrays.items())
    ]


def _contains_image_field(episode: EpisodeArchive) -> bool:
    return any("image" in key.lower() for key in episode.arrays)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def analyze_episode(
    episode: EpisodeArchive,
    *,
    target_delay_steps: int,
    max_joint_delta_rad: float,
    joint_names: Sequence[str],
    unchanged_tolerance: float,
    comparison_tolerance: float,
) -> dict[str, Any]:
    """Analyze one loaded episode and return a JSON-serialisable report."""

    joint_series = resolve_array(episode, "joint_positions")
    joint_positions = _as_time_matrix(joint_series)

    if joint_positions is None:
        width = 6
    else:
        width = int(joint_positions.shape[1])

    names = _find_joint_names(episode, width, joint_names)
    executed_series = resolve_array(episode, "executed_action")
    executed_action = _as_time_matrix(executed_series, expected_width=width)
    raw_series = resolve_array(episode, "raw_policy_action")
    raw_policy = _as_time_matrix(raw_series, expected_width=width)
    bound_series = resolve_array(episode, "policy_bound_clipped_action")
    policy_bound_clipped = _as_time_matrix(bound_series, expected_width=width)
    mapped_series = resolve_array(episode, "mapped_gazebo_action")
    mapped_gazebo = _as_time_matrix(mapped_series, expected_width=width)

    reward_series = resolve_array(episode, "reward")
    reward = _as_scalar_series(reward_series)

    done = _as_scalar_series(resolve_array(episode, "done"))
    terminated = _as_scalar_series(resolve_array(episode, "terminated"))
    truncated = _as_scalar_series(resolve_array(episode, "truncated"))

    if done is None and terminated is not None and truncated is not None:
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)

    distance = _as_scalar_series(resolve_info_series(episode, "distance_m"))
    z_error = _as_scalar_series(resolve_info_series(episode, "z_error"))
    gripper_open = _as_scalar_series(
        resolve_info_series(episode, "gripper_open_norm")
    )
    success_ready = _as_scalar_series(
        resolve_info_series(episode, "success_ready")
    )
    success = _as_scalar_series(resolve_info_series(episode, "success"))

    clipping = _clipping_report(
        joint_names=names,
        current_positions=joint_positions,
        executed_action=executed_action,
        raw_policy=raw_policy,
        policy_bound_clipped=policy_bound_clipped,
        mapped_gazebo=mapped_gazebo,
        max_joint_delta_rad=max_joint_delta_rad,
        tolerance=comparison_tolerance,
    )

    joint_report: dict[str, Any]
    if joint_positions is None:
        joint_report = {
            "available": False,
            "reason": "No time-by-joint observation position field found.",
        }
    else:
        joint_report = {
            "available": True,
            "source": joint_series.source if joint_series else None,
            "per_joint": _stats_dict(
                matrix_stats(
                    joint_positions,
                    unchanged_tolerance=unchanged_tolerance,
                ),
                names,
            ),
        }

    raw_action_report = _action_behaviour_report(
        raw_policy,
        names,
        unchanged_tolerance=unchanged_tolerance,
    )
    if raw_series is not None:
        raw_action_report["source"] = raw_series.source

    executed_action_report = _action_behaviour_report(
        executed_action,
        names,
        unchanged_tolerance=unchanged_tolerance,
    )
    if executed_series is not None:
        executed_action_report["source"] = executed_series.source

    gripper_report: dict[str, Any] = {
        "joint_name": names[-1] if names else "gripper",
    }

    if joint_positions is not None:
        state = joint_positions[:, -1]
        gripper_report["measured_state"] = _finite_summary(state)
        gripper_report["measured_state_net_change"] = float(state[-1] - state[0])
        gripper_report["measured_state_total_variation"] = float(
            np.nansum(np.abs(np.diff(state)))
        )

    if executed_action is not None:
        action = executed_action[:, -1]
        gripper_report["executed_action"] = _finite_summary(action)
        gripper_report["executed_action_total_variation"] = float(
            np.nansum(np.abs(np.diff(action)))
        )

        if joint_positions is not None:
            gripper_report["action_state_correlation"] = _safe_correlation(
                action,
                joint_positions[:, -1],
            )

    if raw_policy is not None:
        gripper_report["raw_policy_action"] = _finite_summary(
            raw_policy[:, -1]
        )

    if gripper_open is not None:
        gripper_report["gripper_open_norm"] = _finite_summary(gripper_open)

    split = min(max(int(target_delay_steps), 0), episode.step_count)
    phase_comparison = {
        "target_delay_steps": int(target_delay_steps),
        "pre_target": _phase_summary(
            0,
            split,
            reward=reward,
            joint_positions=joint_positions,
            executed_action=executed_action,
            distance=distance,
            gripper_open=gripper_open,
        ),
        "post_target": _phase_summary(
            split,
            episode.step_count,
            reward=reward,
            joint_positions=joint_positions,
            executed_action=executed_action,
            distance=distance,
            gripper_open=gripper_open,
        ),
    }

    camera = _camera_report(episode)
    anomalies = _build_anomalies(
        step_count=episode.step_count,
        joint_positions=joint_positions,
        executed_action=executed_action,
        reward=reward,
        clipping=clipping,
        action_behaviour=executed_action_report,
        distance=distance,
        success_ready=success_ready,
        success=success,
        done=done,
        camera=camera,
    )

    reward_report: dict[str, Any]
    if reward is None:
        reward_report = {
            "available": False,
            "reason": "No reward field found.",
        }
    else:
        reward_report = {
            "available": True,
            "source": reward_series.source if reward_series else None,
            "per_step": reward.tolist(),
            "cumulative": np.nancumsum(reward).tolist(),
            "sum": float(np.nansum(reward)),
            "mean": float(np.nanmean(reward)),
            "min": float(np.nanmin(reward)),
            "max": float(np.nanmax(reward)),
        }

    optional_fields = {
        "distance_m": None if distance is None else {
            "source": resolve_info_series(episode, "distance_m").source,
            "summary": _finite_summary(distance),
            "values": distance.tolist(),
        },
        "z_error": None if z_error is None else {
            "source": resolve_info_series(episode, "z_error").source,
            "summary": _finite_summary(z_error),
            "values": z_error.tolist(),
        },
        "gripper_open_norm": None if gripper_open is None else {
            "source": resolve_info_series(
                episode,
                "gripper_open_norm",
            ).source,
            "summary": _finite_summary(gripper_open),
            "values": gripper_open.tolist(),
        },
        "success_ready": None if success_ready is None else {
            "source": resolve_info_series(episode, "success_ready").source,
            "true_count": int(np.count_nonzero(success_ready)),
            "values": np.asarray(success_ready, dtype=bool).tolist(),
        },
        "success": None if success is None else {
            "source": resolve_info_series(episode, "success").source,
            "true_count": int(np.count_nonzero(success)),
            "values": np.asarray(success, dtype=bool).tolist(),
        },
    }

    diagnosis = _classify_failure(
        clipping=clipping,
        action_behaviour=executed_action_report,
        phase_comparison=phase_comparison,
        distance=distance,
        success_ready=success_ready,
        success=success,
        done=done,
        raw_policy_available=raw_policy is not None,
        image_data_present=_contains_image_field(episode),
    )

    return _json_safe(
        {
            "file": str(episode.path),
            "step_count": episode.step_count,
            "joint_names": list(names),
            "field_inventory": _inventory(episode),
            "parsed_json_fields": sorted(episode.json_values),
            "resolved_sources": {
                "joint_positions": None if joint_series is None else joint_series.source,
                "executed_action": None if executed_series is None else executed_series.source,
                "raw_policy_action": None if raw_series is None else raw_series.source,
                "policy_bound_clipped_action": None if bound_series is None else bound_series.source,
                "mapped_gazebo_action": None if mapped_series is None else mapped_series.source,
            },
            "joint_positions": joint_report,
            "raw_policy_action": raw_action_report,
            "executed_action": executed_action_report,
            "max_joint_delta_clipping": clipping,
            "gripper": gripper_report,
            "reward": reward_report,
            "phase_comparison": phase_comparison,
            "optional_fields": optional_fields,
            "camera": camera,
            "anomalies": anomalies,
            "failure_assessment": diagnosis,
            "archive_limitations": [
                "The NPZ intentionally may omit RGB images; image ordering, "
                "crop/distortion, channel order, and normalization therefore "
                "cannot be verified unless image fields are actually present.",
                "Exact 0.03 rad delta clipping requires per-step pre-delta "
                "mapped Gazebo targets. When unavailable, the report labels "
                "0.03 rad command-state differences as boundary-hit inference.",
                "Training-distribution details are not encoded in the rollout, "
                "so target-delay distribution shift can be judged only as a "
                "plausible contributor, not proven from NPZ alone.",
            ],
        }
    )


def _format_number(value: Any) -> str:
    if value is None:
        return "unavailable"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def _append_joint_table(
    lines: list[str],
    title: str,
    report: Mapping[str, Any],
) -> None:
    lines.append(title)

    if not report.get("available"):
        lines.append(f"  unavailable: {report.get('reason')}")
        return

    lines.append(
        "  joint | min | max | mean | range | std | total_variation | "
        "unchanged_ratio | longest_unchanged_run"
    )

    for joint, values in report.get("per_joint", {}).items():
        lines.append(
            "  "
            + " | ".join(
                [
                    joint,
                    _format_number(values.get("min")),
                    _format_number(values.get("max")),
                    _format_number(values.get("mean")),
                    _format_number(values.get("range")),
                    _format_number(values.get("std")),
                    _format_number(values.get("total_variation")),
                    f"{float(values.get('unchanged_ratio', 0.0)):.1%}",
                    str(values.get("longest_unchanged_run")),
                ]
            )
        )


def render_text_report(report: Mapping[str, Any]) -> str:
    """Render a compact but complete human-readable diagnostic report."""

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(f"ACT rollout diagnostic: {report['file']}")
    lines.append(f"step_count={report['step_count']}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("1. NPZ field inventory")

    for field in report["field_inventory"]:
        lines.append(
            f"  {field['name']}: shape={tuple(field['shape'])}, "
            f"dtype={field['dtype']}"
        )

    lines.append("")
    _append_joint_table(lines, "2. Joint position statistics", report["joint_positions"])
    lines.append("")
    _append_joint_table(lines, "3. Raw policy action statistics", report["raw_policy_action"])
    lines.append("")
    _append_joint_table(lines, "4. Executed action statistics", report["executed_action"])
    lines.append("")
    lines.append("5. max_joint_delta_rad clipping")
    clipping = report["max_joint_delta_clipping"]
    lines.append(
        "  method="
        f"{clipping.get('method')}, exact="
        f"{clipping.get('exact_delta_clipping_available')}"
    )

    if clipping.get("warning"):
        lines.append(f"  warning={clipping['warning']}")

    for joint, values in clipping.get("per_joint", {}).items():
        lines.append(
            f"  {joint}: count={values['count']}, ratio={values['ratio']:.1%}"
        )

    if "any_joint_step_ratio" in clipping:
        lines.append(
            "  any_joint: count="
            f"{clipping['any_joint_step_count']}, "
            f"ratio={clipping['any_joint_step_ratio']:.1%}"
        )

    lines.append("")
    lines.append("6. Saturation / constancy")
    lines.append(
        "  raw nearly_constant_joints="
        f"{report['raw_policy_action'].get('nearly_constant_joints')}"
    )
    lines.append(
        "  executed nearly_constant_joints="
        f"{report['executed_action'].get('nearly_constant_joints')}"
    )
    lines.append("")
    lines.append("7. Gripper")

    for key, value in report["gripper"].items():
        lines.append(f"  {key}={value}")

    lines.append("")
    lines.append("8. Reward")
    reward = report["reward"]

    if reward.get("available"):
        lines.append(
            f"  sum={reward['sum']:.6f}, mean={reward['mean']:.6f}, "
            f"min={reward['min']:.6f}, max={reward['max']:.6f}"
        )
        lines.append(f"  cumulative_final={reward['cumulative'][-1]:.6f}")
    else:
        lines.append(f"  unavailable: {reward.get('reason')}")

    lines.append("")
    lines.append("9. Pre-target vs post-target")

    for phase_name in ("pre_target", "post_target"):
        phase = report["phase_comparison"][phase_name]
        lines.append(f"  {phase_name}: {phase}")

    lines.append("")
    lines.append("10-11. Optional distance/orientation/gripper/success fields")

    for key, value in report["optional_fields"].items():
        lines.append(f"  {key}={value if value is not None else 'unavailable'}")

    lines.append("")
    lines.append("12. Camera reuse / image age")

    for key, value in report["camera"].items():
        lines.append(f"  {key}={value}")

    lines.append("")
    lines.append("13. Key anomalies")

    if report["anomalies"]:
        for anomaly in report["anomalies"]:
            lines.append(f"  - {anomaly}")
    else:
        lines.append("  - No rule-based anomaly was detected in available fields.")

    lines.append("")
    lines.append("14. Failure assessment")

    for finding in report["failure_assessment"]:
        lines.append(
            f"  - {finding['category']}: {finding['assessment']} — "
            f"{finding['evidence']}"
        )

    lines.append("")
    lines.append("Archive limitations")

    for limitation in report["archive_limitations"]:
        lines.append(f"  - {limitation}")

    return "\n".join(lines) + "\n"


def _write_curve_csv(
    report: Mapping[str, Any],
    destination: Path,
) -> None:
    step_count = int(report["step_count"])
    reward = report["reward"].get("per_step") if report["reward"].get("available") else None
    cumulative = report["reward"].get("cumulative") if report["reward"].get("available") else None
    distance_info = report["optional_fields"].get("distance_m")
    z_info = report["optional_fields"].get("z_error")
    gripper_info = report["optional_fields"].get("gripper_open_norm")
    rows = ["step,reward,cumulative_reward,distance_m,z_error,gripper_open_norm"]

    for index in range(step_count):
        values = [
            index,
            "" if reward is None else reward[index],
            "" if cumulative is None else cumulative[index],
            "" if distance_info is None else distance_info["values"][index],
            "" if z_info is None else z_info["values"][index],
            "" if gripper_info is None else gripper_info["values"][index],
        ]
        rows.append(",".join(str(value) for value in values))

    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _save_default_plots(report: Mapping[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    created: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    step_count = int(report["step_count"])
    steps = np.arange(step_count)

    plot_specs: list[tuple[str, str, Sequence[float] | None, str]] = []
    reward = report["reward"]

    if reward.get("available"):
        plot_specs.extend(
            [
                ("reward_curve.png", "Reward per step", reward["per_step"], "reward"),
                (
                    "cumulative_reward_curve.png",
                    "Cumulative reward",
                    reward["cumulative"],
                    "cumulative reward",
                ),
            ]
        )

    optional = report["optional_fields"]

    if optional.get("distance_m") is not None:
        plot_specs.append(
            (
                "distance_curve.png",
                "Grasp point to target distance",
                optional["distance_m"]["values"],
                "distance (m)",
            )
        )

    if optional.get("gripper_open_norm") is not None:
        plot_specs.append(
            (
                "gripper_open_norm_curve.png",
                "Gripper open norm",
                optional["gripper_open_norm"]["values"],
                "gripper open norm",
            )
        )

    for filename, title, values, ylabel in plot_specs:
        if values is None:
            continue

        figure = plt.figure()
        axis = figure.add_subplot(111)
        axis.plot(steps, np.asarray(values, dtype=float))
        axis.axvline(
            int(report["phase_comparison"]["target_delay_steps"]),
            linestyle="--",
        )
        axis.set_title(title)
        axis.set_xlabel("policy step")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(str(path))

    return created


def analyze_paths(
    paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    target_delay_steps: int,
    max_joint_delta_rad: float,
    joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
    unchanged_tolerance: float = 1e-4,
    comparison_tolerance: float = 1e-8,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Analyze multiple NPZ files and persist text, JSON, CSV, and plots."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    episode_reports: list[dict[str, Any]] = []

    for path in paths:
        episode = load_episode_archive(path)
        report = analyze_episode(
            episode,
            target_delay_steps=target_delay_steps,
            max_joint_delta_rad=max_joint_delta_rad,
            joint_names=joint_names,
            unchanged_tolerance=unchanged_tolerance,
            comparison_tolerance=comparison_tolerance,
        )
        stem = episode.path.stem
        text_path = destination / f"{stem}_diagnostic.txt"
        json_path = destination / f"{stem}_diagnostic.json"
        csv_path = destination / f"{stem}_curves.csv"
        text_path.write_text(render_text_report(report), encoding="utf-8")
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_curve_csv(report, csv_path)
        report["output_files"] = {
            "text": str(text_path),
            "json": str(json_path),
            "csv": str(csv_path),
            "plots": _save_default_plots(report, destination / stem)
            if make_plots
            else [],
        }
        episode_reports.append(report)
        print(render_text_report(report), end="")

    combined = {
        "episode_count": len(episode_reports),
        "episodes": episode_reports,
    }
    combined_path = destination / "combined_diagnostic.json"
    combined_path.write_text(
        json.dumps(_json_safe(combined), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"combined_report={combined_path}")
    return combined


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose image-free SO-101 ACT rollout NPZ files. The script "
            "does not modify policy, reward, cameras, or training settings."
        )
    )
    parser.add_argument("npz", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-delay-steps", type=int, default=90)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.03)
    parser.add_argument(
        "--joint-names",
        default=",".join(DEFAULT_JOINT_NAMES),
        help="Comma-separated controller/policy joint order.",
    )
    parser.add_argument("--unchanged-tolerance", type=float, default=1e-4)
    parser.add_argument("--comparison-tolerance", type=float, default=1e-8)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    names = tuple(
        name.strip()
        for name in args.joint_names.split(",")
        if name.strip()
    )

    if args.target_delay_steps < 0:
        raise SystemExit("--target-delay-steps must be non-negative.")

    if not math.isfinite(args.max_joint_delta_rad) or args.max_joint_delta_rad <= 0:
        raise SystemExit("--max-joint-delta-rad must be positive and finite.")

    if not math.isfinite(args.unchanged_tolerance) or args.unchanged_tolerance < 0:
        raise SystemExit("--unchanged-tolerance must be finite and non-negative.")

    analyze_paths(
        args.npz,
        output_dir=args.output_dir,
        target_delay_steps=args.target_delay_steps,
        max_joint_delta_rad=args.max_joint_delta_rad,
        joint_names=names,
        unchanged_tolerance=args.unchanged_tolerance,
        comparison_tolerance=args.comparison_tolerance,
        make_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
