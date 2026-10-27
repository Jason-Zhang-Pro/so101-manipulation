"""Offline diagnostics for the initial-pose → HOME → target evaluation.

Distances and angles follow ``StageEvaluator._metrics``. This monitor does not
declare formal S0/S1 success: its starting pose and continuous timeline differ
from the training-stage protocol. Durations use fresh evidence timestamps, not
wall time, callback counts, or commanded joint positions.
"""
from __future__ import annotations

import math
from collections import Counter


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    result = [_number(item) for item in value]
    return result if all(item is not None for item in result) else None


def _norm(value):
    return math.sqrt(sum(item * item for item in value))


def _angle(left, right):
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))))


def _rotation_axis(rotation):
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        return None
    rows = [_vector(row, 3) for row in rotation]
    if any(row is None for row in rows):
        return None
    for i in range(3):
        for j in range(3):
            if abs(sum(rows[k][i] * rows[k][j] for k in range(3)) - (i == j)) > 1e-3:
                return None
    a, b, c = rows
    determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    return [row[2] for row in rows] if determinant >= 0.99 else None


def measure(info: dict, reward_config: dict) -> dict:
    """Read serialized aligned ``TaskMeasurements`` supplied by the runtime.

    Missing/invalid evidence yields explicit invalid flags and nullable values.
    Raw finite measurements may remain visible when stale, but summaries never
    use those measurements as evidence of progress or stable HOME dwell.
    """
    m = info.get("evaluation_measurements") or {}
    snapshot = info.get("task_snapshot") or info
    home = _vector(reward_config.get("home_joint_positions_rad"), 6)
    if home is None:
        raise ValueError("reward_config requires six finite HOME joint positions")
    sim_time = _number(m.get("sim_time_s"))
    evidence_time = _number(m.get("evidence_sim_time_s"))
    if m.get("evidence_sim_time_s") is None:
        evidence_time = sim_time
    age = None if sim_time is None or evidence_time is None else sim_time - evidence_time
    joints = _vector(m.get("joint_positions_rad"), 6)
    velocities = _vector(m.get("joint_velocities_rad_s"), 6)
    errors = None if joints is None else [abs(q - h) for q, h in zip(joints, home)]
    home_error = None if errors is None else max(errors)
    velocity = None if velocities is None else max(abs(value) for value in velocities)
    evidence_valid = bool(
        m.get("evidence_valid", False) and m.get("joint_measurement_valid", False)
        and joints is not None and velocities is not None and age is not None
        and -1e-6 <= age <= float(reward_config["max_evidence_gap_s"]) + 1e-9
    )
    result = {
        "evidence_valid": evidence_valid,
        "evidence_sim_time_s": evidence_time, "sim_time_s": sim_time,
        "evidence_age_s": age, "home_error_rad": home_error,
        "home_joint_errors_rad": errors, "max_joint_velocity_rad_s": velocity,
        "at_home": bool(evidence_valid and home_error <= float(reward_config["home_joint_tolerance_rad"]) + 1e-9),
        "robot_static": bool(evidence_valid and velocity <= float(reward_config["home_velocity_tolerance_rad_s"]) + 1e-9),
        "opening_rad": None if joints is None else joints[-1] - home[-1],
        "target_name": m.get("target_name"), "target_present": m.get("target_name") is not None,
        "target_evidence_valid": False, "target_motion_valid": bool(m.get("target_motion_valid", False)),
        "distance_m": None, "xy_distance_m": None, "dz_m": None,
        "bearing_error_deg": None, "axis_error_deg": None, "orientation_error_deg": None,
        "contact_valid": bool(m.get("contact_valid", False)),
        **{key: bool(m.get(key, False)) for key in ("left_contact", "right_contact", "contacts_simultaneous")},
        "task_phase": snapshot.get("task_phase", snapshot.get("phase")),
        **{key: bool(snapshot.get(key, False)) for key in (
            "grasp_valid", "grasp_attempt_active", "lifted_once", "attempt_lifted",
            "target_supported", "target_static", "target_in_goal")},
    }
    target = _vector(m.get("target_position_world_m"), 3)
    grasp = _vector(m.get("grasp_position_world_m"), 3)
    axis = _rotation_axis(m.get("grasp_rotation_world"))
    if not result["target_present"] or target is None or grasp is None or axis is None or _norm(axis) < 1e-9:
        return result
    axis = [value / _norm(axis) for value in axis]
    desired = _vector(reward_config.get("desired_grasp_axis_world"), 3)
    if desired is None or _norm(desired) < 1e-9:
        raise ValueError("reward_config requires a finite nonzero desired grasp axis")
    delta = [g - t for g, t in zip(grasp, target)]
    bearing = [-d + float(reward_config["approach_reference_offset_m"]) * a for d, a in zip(delta, axis)]
    bearing_length = _norm(bearing)
    bearing_error = 180.0 if bearing_length < 1e-9 else _angle(axis, [value / bearing_length for value in bearing])
    axis_error = _angle(axis, desired)
    result.update(
        target_evidence_valid=bool(evidence_valid and m.get("target_measurement_valid", False)
                                   and m.get("target_motion_valid", False)),
        distance_m=_norm(delta), xy_distance_m=_norm(delta[:2]), dz_m=delta[2],
        bearing_error_deg=bearing_error, axis_error_deg=axis_error,
        orientation_error_deg=max(bearing_error, axis_error),
    )
    return result


def summarize(samples: list[dict], *, initial: dict, reward_config: dict,
              target_delay_s: float = 3.0) -> dict:
    """Summarize an initial measurement and control-clock labelled samples.

    Each sample contains ``step_index``, ``elapsed_s``, and ``diagnostics``.
    Distinct monotonically advancing evidence contributes one sample. A stable
    interval needs qualifying endpoints and a gap no larger than the configured
    evidence gap. Duplicate evidence adds neither dwell nor sample count;
    invalid/backwards evidence and long gaps break stable dwell.
    """
    max_gap = float(reward_config["max_evidence_gap_s"])
    # The policy timeline starts at the initial control clock, not the older
    # aligned camera evidence. Keep the evidence fallback for saved fixtures.
    origin = _number(initial.get("sim_time_s"))
    if origin is None:
        origin = _number(initial.get("evidence_sim_time_s"))
    rows = [{"step_index": 0, "elapsed_s": 0.0, "diagnostics": initial}, *samples]
    unique, before = [], []
    last_time, previous_home_time, stable_duration, longest_stable = None, None, 0.0, 0.0
    counts = Counter()
    first_home = first_stable = None
    for row in rows:
        d, elapsed = row["diagnostics"], float(row["elapsed_s"])
        time = _number(d.get("evidence_sim_time_s"))
        valid = bool(d.get("evidence_valid") and time is not None)
        if not valid:
            counts["invalid_samples"] += 1
            previous_home_time, stable_duration = None, 0.0
            continue
        if last_time is not None and time <= last_time:
            counts["duplicate_samples" if time == last_time else "backwards_samples"] += 1
            if time < last_time:
                previous_home_time, stable_duration = None, 0.0
            continue
        if last_time is not None and time - last_time > max_gap + 1e-9:
            counts["long_evidence_gaps"] += 1
            previous_home_time, stable_duration = None, 0.0
        last_time = time
        unique.append(row)
        if elapsed > target_delay_s + 1e-9:
            continue
        before.append(row)
        relative_time = None if origin is None else max(0.0, time - origin)
        if d.get("at_home") and first_home is None:
            first_home = relative_time
        stable = bool(d.get("at_home") and d.get("robot_static"))
        if stable:
            if first_stable is None:
                first_stable = relative_time
            if previous_home_time is not None and time - previous_home_time <= max_gap + 1e-9:
                interval_start = previous_home_time if origin is None else max(previous_home_time, origin)
                stable_duration += max(0.0, time - interval_start)
            else:
                stable_duration = 0.0
            previous_home_time = time
            longest_stable = max(longest_stable, stable_duration)
        else:
            previous_home_time, stable_duration = None, 0.0
    boundary = next((row for row in reversed(rows) if abs(float(row["elapsed_s"]) - target_delay_s) <= 1e-6), None)
    boundary_d = {} if boundary is None else boundary["diagnostics"]
    boundary_valid = bool(boundary_d.get("evidence_valid"))
    after = [row for row in unique if float(row["elapsed_s"]) > target_delay_s + 1e-9]
    target_rows = [row for row in after if row["diagnostics"].get("target_evidence_valid")
                   and _number(row["diagnostics"].get("distance_m")) is not None]
    nearest = min(target_rows, key=lambda row: row["diagnostics"]["distance_m"], default=None)
    final = rows[-1]
    final_d = final["diagnostics"]
    final_target_valid = bool(float(final["elapsed_s"]) > target_delay_s + 1e-9
                              and final_d.get("target_evidence_valid"))
    tail = [row for row in target_rows if float(row["elapsed_s"]) >= float(final["elapsed_s"]) - 2.0 - 1e-9]

    def compact(row):
        if row is None:
            return None
        d = row["diagnostics"]
        return {"step_index": row["step_index"], "elapsed_s": row["elapsed_s"],
                **{key: d.get(key) for key in ("evidence_sim_time_s", "distance_m", "xy_distance_m", "dz_m",
                                                "bearing_error_deg", "axis_error_deg", "orientation_error_deg", "opening_rad")}}

    home_errors = [row["diagnostics"]["home_error_rad"] for row in before
                   if _number(row["diagnostics"].get("home_error_rad")) is not None]
    first_target = target_rows[0] if target_rows else None
    return {
        "protocol": "initial_pose_home_then_target_monitor",
        "formal_stage_success_evaluated": False,
        "distance_threshold_counts_are_successes": False,
        "evidence": {"total_samples_including_initial": len(rows), "valid_unique_samples": len(unique),
                     **{key: counts[key] for key in ("invalid_samples", "duplicate_samples", "backwards_samples", "long_evidence_gaps")}},
        "before_target": {
            "duration_s": target_delay_s, "first_at_home_evidence_elapsed_s": first_home,
            "first_home_and_static_evidence_elapsed_s": first_stable,
            "longest_home_and_static_duration_s": longest_stable,
            "minimum_home_error_rad": min(home_errors) if home_errors else None,
            "boundary_measurement_available": boundary is not None,
            "boundary_evidence_valid": boundary_valid,
            "at_home_at_target_request": bool(boundary_d.get("at_home")) if boundary_valid else None,
            "home_and_static_at_target_request": bool(boundary_d.get("at_home") and boundary_d.get("robot_static")) if boundary_valid else None,
            "home_error_at_target_request_rad": boundary_d.get("home_error_rad") if boundary_valid else None,
            "unexpected_target_sample_count": sum(bool(row["diagnostics"].get("target_present")) for row in before),
        },
        "after_target": {
            "first_valid_target_feedback_elapsed_s": None if first_target is None else first_target["elapsed_s"],
            "first_valid_target_evidence_elapsed_s": None if first_target is None or origin is None else first_target["diagnostics"]["evidence_sim_time_s"] - origin,
            "valid_unique_target_samples": len(target_rows),
            "minimum_distance_m": None if nearest is None else nearest["diagnostics"]["distance_m"],
            "nearest": compact(nearest),
            "final_distance_m": final_d.get("distance_m") if final_target_valid else None,
            "final": compact(final) if final_target_valid else None,
            "tail_window_s": 2.0, "tail_valid_unique_samples": len(tail),
            "tail_mean_distance_m": sum(row["diagnostics"]["distance_m"] for row in tail) / len(tail) if tail else None,
            "distance_threshold_sample_counts": {f"le_{cm}_cm": sum(row["diagnostics"]["distance_m"] <= cm / 100 + 1e-9 for row in target_rows) for cm in (8, 6, 4, 2)},
            "task_phase_sample_counts": dict(Counter(row["diagnostics"].get("task_phase") or "unknown" for row in after)),
            **{f"observed_{flag}": any(row["diagnostics"].get("contact_valid") and row["diagnostics"].get(flag) for row in after)
               for flag in ("grasp_valid", "grasp_attempt_active", "lifted_once", "attempt_lifted", "left_contact", "right_contact", "contacts_simultaneous")},
        },
    }
