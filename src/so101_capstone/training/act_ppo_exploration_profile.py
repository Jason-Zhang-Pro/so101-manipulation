"""Distance-scaled physical exploration for the existing absolute ACT policy.

The historical ``final_good_act`` controller multiplied a unit Gaussian by a
distance-dependent joint delta. Here the physical scales are reused, with an
S1-specific near-target boost and an opt-in continuous near-target taper. ACT
still commands absolute positions; the sampler and likelihood remain the
existing bounded-mean truncated Gaussian.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS


_NEAR_THRESHOLD = 0.08
_FAR_THRESHOLD = 0.15
_NEAR_STD_RAD = 0.05
_MIDDLE_STD_RAD = 0.08
_FAR_STD_RAD = 0.12
# S1-only near-target boost. Keep the original bins for HOME and full-task
# grasp/transport so increasing reach exploration does not disrupt holding.
_S1_NEAR_STD_RAD = 0.12
# Optional S1 taper: (measured target distance in metres, initial scale in rad).
# The scale is the underlying Gaussian's parameter; truncation at joint limits
# can make the sampled command's measured standard deviation smaller.
_S1_NEAR_TARGET_CONTROL_POINTS = (
    (0.02, 0.012), (0.04, 0.025), (0.06, 0.05), (0.08, 0.08), (0.15, 0.12),
)
_HOME_MINIMUM_STD_RAD = 0.001
_HOME_EXCESS_ERROR_GAIN = 0.5
_PHASE_STAGES = {
    "WAIT_HOME": "idle",
    "APPROACH": "moving",
    "GRASP_LIFT": "manipulation",
    "TRANSPORT": "moving",
    "PLACE_RELEASE": "manipulation",
    "RETURN_HOME": "moving",
    "HOME_HOLD": "home_hold",
}
_HOME_PHASES = frozenset(("WAIT_HOME", "RETURN_HOME", "HOME_HOLD"))


def _half_spans_rad() -> np.ndarray:
    return np.asarray([
        (joint.gazebo_upper_rad - joint.gazebo_lower_rad) / 2.0
        for joint in SIM_JOINT_CALIBRATIONS
    ], dtype=np.float64)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    return result


def _measurement(info: Mapping[str, Any], primary: str, fallback: str | None = None) -> float:
    # An invalid supplied primary measurement must not silently fall back to a
    # different value. Missing stage-only fields can use the full-task evidence.
    key = primary if primary in info else fallback
    if key is None or key not in info:
        raise ValueError(f"final_good_act exploration requires {primary}"
                         + (f" or {fallback}." if fallback else "."))
    result = _finite_number(info[key], name=key)
    if result < 0.0:
        raise ValueError(f"{key} must be non-negative.")
    return result


def _distance_std_rad(distance: float) -> float:
    if distance > _FAR_THRESHOLD:
        return _FAR_STD_RAD
    if distance > _NEAR_THRESHOLD:
        return _MIDDLE_STD_RAD
    return _NEAR_STD_RAD


def _exploration_task_phase(info: Mapping[str, Any]) -> str:
    """A prepared local stage can precede the corresponding physical phase."""
    stage = info.get("training_stage")
    if stage in {"S2", "S3", "S4", "S5"}:
        if stage == "S5":
            return "HOME_HOLD" if bool(info.get("stage_robot_at_home", info.get("robot_at_home", False))) else "RETURN_HOME"
        return {"S2": "GRASP_LIFT", "S3": "TRANSPORT", "S4": "PLACE_RELEASE"}[stage]
    value = info.get("task_phase_name", info.get("task_phase"))
    if not isinstance(value, str) or value.upper() not in _PHASE_STAGES:
        raise ValueError("final_good_act exploration requires a valid task phase.")
    return value.upper()


def final_good_act_exploration_offset(
    info: Mapping[str, Any], *, stage_name: str, initial_log_std: float,
) -> np.ndarray:
    """Return six normalized log-std offsets for the current state.

    Reach thresholds use metres. HOME thresholds deliberately use the measured
    maximum joint error in radians, followed by a taper to 0.001 rad at HOME.
    S1 target phases use 0.12 rad at distances up to 0.08 m, including the
    GRASP_LIFT phase entered by proximity alone. HOME keeps its original bins.
    ``home_joint_tolerance_rad`` must come from the active reward configuration;
    this module does not invent a HOME tolerance. The shared learned base still
    multiplies these initial physical scales by ``exp(base - initial_base)``.
    """
    initial = _finite_number(initial_log_std, name="initial_log_std")
    phase = _exploration_task_phase(info)
    if stage_name != _PHASE_STAGES[phase]:
        raise ValueError(f"Exploration stage {stage_name!r} does not match task phase {phase}.")

    if phase in _HOME_PHASES:
        error = _measurement(info, "stage_home_error_rad", "home_error_rad")
        tolerance = _measurement(info, "home_joint_tolerance_rad")
        if tolerance <= 0.0:
            raise ValueError("home_joint_tolerance_rad must be positive.")
        sigma = min(
            _distance_std_rad(error),
            _HOME_MINIMUM_STD_RAD + _HOME_EXCESS_ERROR_GAIN * max(error - tolerance, 0.0),
        )
    else:
        distance = _measurement(info, "stage_reach_distance_m", "target_distance_m")
        sigma = _distance_std_rad(distance)
        if info.get("training_stage") == "S1" and distance <= _NEAR_THRESHOLD:
            sigma = _S1_NEAR_STD_RAD

    return (np.log(sigma / _half_spans_rad()) - initial).astype(np.float32)


def final_good_act_near_target_exploration_offset(
    info: Mapping[str, Any], *, stage_name: str, initial_log_std: float,
) -> np.ndarray:
    """Taper S1 target exploration continuously while retaining old HOME/full behavior.

    Reuse the existing evidence and phase validation first, including rejection
    of a supplied invalid stage measurement rather than falling back silently.
    This changes exploration only, never the absolute ACT action mean.
    """
    original = final_good_act_exploration_offset(
        info, stage_name=stage_name, initial_log_std=initial_log_std,
    )
    phase = info.get("task_phase_name", info.get("task_phase")).upper()
    if info.get("training_stage") != "S1" or phase in _HOME_PHASES:
        return original
    distance = _measurement(info, "stage_reach_distance_m", "target_distance_m")
    distances, scales = zip(*_S1_NEAR_TARGET_CONTROL_POINTS)
    # np.interp clamps both ends: <=2 cm stays at .012 rad, >=15 cm at .12 rad.
    sigma = float(np.interp(distance, distances, scales))
    return (np.log(sigma / _half_spans_rad()) - float(initial_log_std)).astype(np.float32)


def validate_final_good_act_exploration_bounds(
    *, initial_log_std: float, minimum_log_std: float, maximum_log_std: float,
    gripper_maximum_log_std: float | None = None,
) -> None:
    """Reject bounds that would silently clip the profile's initial scales.

    The distance bins, S1 near-target override and HOME floor bound every scale
    selected by both profiles, including intermediate HOME and S1 taper values.
    Every optional S1 control point lies inside this same [.001, .12] rad
    envelope, so checking the extrema also covers its continuous interpolation.
    Learning can still hit the existing configured bounds after initialization.
    """
    initial = _finite_number(initial_log_std, name="initial_log_std")
    minimum = _finite_number(minimum_log_std, name="minimum_log_std")
    maximum = _finite_number(maximum_log_std, name="maximum_log_std")
    if not minimum <= initial <= maximum:
        raise ValueError("initial_log_std is outside configured bounds.")
    caps = np.full(len(SIM_JOINT_CALIBRATIONS), maximum, dtype=np.float64)
    if gripper_maximum_log_std is not None:
        caps[-1] = _finite_number(gripper_maximum_log_std, name="gripper_maximum_log_std")
    if np.any(caps < minimum):
        raise ValueError("Effective exploration maximum must not be below minimum_log_std.")

    for sigma in (_HOME_MINIMUM_STD_RAD, _NEAR_STD_RAD, _MIDDLE_STD_RAD, _FAR_STD_RAD, _S1_NEAR_STD_RAD):
        values = np.log(sigma / _half_spans_rad())
        invalid = (values < minimum) | (values > caps)
        if np.any(invalid):
            names = ", ".join(joint.name for joint, outside in zip(SIM_JOINT_CALIBRATIONS, invalid) if outside)
            raise ValueError(
                f"final_good_act exploration bounds would clip the initial {sigma:g} rad "
                f"standard deviation for {names}; widen the configured log-std bounds."
            )


def final_good_act_exploration_signature() -> dict[str, Any]:
    """Return a fresh JSON-compatible contract for exact resume validation."""
    # Preserve this historical profile contract exactly for existing runs.
    # Its preclip label predates the truncated sampler; the enclosing trainer
    # execution signature records the actual distribution and checkpoint schema.
    return {
        "profile": "final_good_act",
        "schema_version": 2,
        "action_mean": "unchanged_absolute_act",
        "sample_distribution": "existing_preclip_gaussian",
        "distance_bins": {
            "near_upper_inclusive": _NEAR_THRESHOLD,
            "middle_upper_inclusive": _FAR_THRESHOLD,
            "reach_distance_units": "metres",
            "home_error_units": "radians_max_joint_error",
            "initial_joint_std_rad": [_NEAR_STD_RAD, _MIDDLE_STD_RAD, _FAR_STD_RAD],
        },
        "s1_near_override": {
            "training_stage": "S1",
            "phases": sorted(set(_PHASE_STAGES) - _HOME_PHASES),
            "distance_upper_inclusive_m": _NEAR_THRESHOLD,
            "initial_joint_std_rad": _S1_NEAR_STD_RAD,
        },
        "home": {
            "phases": sorted(_HOME_PHASES),
            "minimum_initial_std_rad": _HOME_MINIMUM_STD_RAD,
            "excess_error_gain": _HOME_EXCESS_ERROR_GAIN,
            "rule": "min(distance_bin_std, floor + gain * max(error - tolerance, 0))",
            "tolerance_source": "active_reward_home_joint_tolerance_rad",
        },
        "evidence": {
            "phase": ["task_phase_name", "task_phase"],
            "reach_distance": ["stage_reach_distance_m", "target_distance_m"],
            "home_error": ["stage_home_error_rad", "home_error_rad"],
            "invalid_measurement": "reject",
        },
        "joint_names": [joint.name for joint in SIM_JOINT_CALIBRATIONS],
        "normalized_half_span_rad": _half_spans_rad().tolist(),
        "learned_scale": "exp(base_log_std - initial_log_std)",
        "effective_bounds": "existing_configured_log_std_limits",
    }


def final_good_act_near_target_exploration_signature() -> dict[str, Any]:
    """Describe the opt-in taper without changing the historical profile hash."""
    signature = final_good_act_exploration_signature()
    signature.update(
        profile="final_good_act_near_target", schema_version=1,
        sample_distribution="bounded_mean_truncated_gaussian",
    )
    del signature["s1_near_override"]
    signature["s1_target_taper"] = {
        "training_stage": "S1",
        "phases": sorted(set(_PHASE_STAGES) - _HOME_PHASES),
        "control_points": [
            {"distance_m": distance, "initial_joint_std_rad": scale}
            for distance, scale in _S1_NEAR_TARGET_CONTROL_POINTS
        ],
        "interpolation": "linear_in_physical_radians_constant_beyond_endpoints",
        "scale_semantics": "underlying_gaussian_before_truncation",
        "other_stages_and_home": "unchanged_final_good_act_schema_2",
    }
    return signature
