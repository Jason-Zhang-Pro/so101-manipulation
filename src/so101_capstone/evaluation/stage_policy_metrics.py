"""Pure summaries of stage-policy evaluation, without redefining success.

Only a matching stage evaluator's terminal response supplies a result. Geometry
and telemetry below are diagnostics, never alternative success predicates.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math


_STAGES = tuple(f"S{i}" for i in range(6))
_RESULTS = {"SUCCESS", "TIMEOUT", "FAILURE"}


def _stage(value):
    value = str(value).upper()
    if value not in _STAGES:
        raise ValueError(f"Unknown training stage: {value}")
    return value


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _vector(value, size=6):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    values = [_number(item) for item in value]
    return values if all(item is not None for item in values) else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _mean(values):
    return sum(values) / len(values) if values else None


def _mean_vector(values):
    return [_mean(list(column)) for column in zip(*values)] if values else None


def _diagnostics(row, home, max_gap):
    info = _mapping(row.get("info"))
    m = _mapping(info.get("evaluation_measurements"))
    snapshot = _mapping(info.get("task_snapshot")) or info
    clock = _number(m.get("sim_time_s", info.get("sim_time_s")))
    stamp = _number(m.get("evidence_sim_time_s", info.get("evidence_sim_time_s", clock)))
    age = None if clock is None or stamp is None else clock - stamp
    valid = bool(m.get("evidence_valid", info.get("evidence_valid")) and age is not None
                 and -1e-6 <= age <= max_gap + 1e-9)
    joints = _vector(m.get("joint_positions_rad", row.get("joint_positions_after_rad")))
    speed = _vector(m.get("joint_velocities_rad_s", row.get("joint_velocities_after_rad_s")))
    # The standard worker's evidence_valid already includes joint validity.
    joint_valid = bool(valid and m.get("joint_measurement_valid", True) and joints is not None)
    errors = None if not joint_valid else [math.degrees(abs(q - h)) for q, h in zip(joints, home)]
    velocities = [abs(item) for item in speed] if joint_valid and speed is not None else None
    target = _vector(m.get("target_position_world_m"), 3)
    grasp = _vector(m.get("grasp_position_world_m"), 3)
    metrics = _mapping(info.get("stage_metrics"))
    target_present = bool(m.get("target_name", info.get("target_name")) is not None
                          or info.get("physical_target_present"))
    target_valid = bool(valid and target_present and m.get("target_measurement_valid", True)
                        and m.get("target_motion_valid", info.get("target_motion_valid",
                                  snapshot.get("target_motion_valid", False)))
                        and info.get("task_evidence_available", True))
    distance = None
    if target_valid:
        distance = (math.sqrt(sum((a - b) ** 2 for a, b in zip(target, grasp)))
                    if target is not None and grasp is not None else
                    _number(info.get("distance_m", metrics.get("stage_reach_distance_m",
                                 info.get("stage_reach_distance_m", snapshot.get("target_distance_m"))))))
        if distance is not None and distance < 0:
            distance = None
    return {
        "stamp": stamp, "valid": valid, "joint_valid": joint_valid,
        "elapsed_s": _number(info.get("stage_elapsed_s")),
        "errors": errors, "velocities": velocities, "distance_m": distance,
        "lift_completed": bool(metrics.get("stage_s1_lift_completed", info.get("stage_s1_lift_completed"))),
        "lift_hold_s": _number(metrics.get("stage_s1_lift_hold_s", info.get("stage_s1_lift_hold_s"))),
        "clearance_m": _number(metrics.get("stage_s1_clearance_m", info.get("stage_s1_clearance_m"))),
        "snapshot": snapshot, "measurements": m, "info": info,
    }


def summarize_episode(records, *, stage, initial_info, reward_config, error=None,
                      policy_started=True) -> dict:
    """Summarize confirmed policy steps and preserve the evaluator's decision.

    Diagnostics use the standard worker's evidence flags and measured joint
    arrays. Serialized aligned ``evaluation_measurements`` are also supported.
    Initial/preparation data are not counted as policy steps, rewards, event
    counts, or progress. Invalid, duplicate, and backwards evidence cannot
    improve the reported minimum target distance.
    """
    stage = _stage(stage)
    records = list(records)
    initial_info = _mapping(initial_info)
    home = _vector(reward_config.get("home_joint_positions_rad"))
    max_gap = _number(reward_config.get("max_evidence_gap_s"))
    if home is None or max_gap is None or max_gap <= 0:
        raise ValueError("reward_config requires six finite HOME positions and a positive max_evidence_gap_s")
    final_row = records[-1] if records else {}
    final_info = _mapping(final_row.get("info")) if records else initial_info
    terminal = bool(final_row.get("terminated") or final_row.get("truncated"))
    matching = final_info.get("training_stage") == stage
    declared = final_info.get("stage_result")
    failure = deepcopy(error)
    if failure is None:
        if not policy_started:
            failure = "Policy sampling did not start."
        elif not records:
            failure = "No completed policy steps were recorded."
        elif not matching:
            failure = f"Final response stage {final_info.get('training_stage')!r} does not match {stage}."
        elif not terminal or declared not in _RESULTS:
            failure = "No authoritative terminal stage result was recorded."
    evaluated = failure is None
    result = declared if evaluated else "ERROR"
    samples = [_diagnostics(row, home, max_gap) for row in records]
    unique = []
    evidence_counts = Counter()
    last_stamp = None
    for sample in samples:
        stamp = sample["stamp"]
        sample["chronological_valid"] = sample["valid"] and (last_stamp is None or stamp >= last_stamp)
        if not sample["valid"]:
            evidence_counts["invalid"] += 1
        elif last_stamp is not None and stamp <= last_stamp:
            evidence_counts["duplicate" if stamp == last_stamp else "backwards"] += 1
        else:
            if last_stamp is not None and stamp - last_stamp > max_gap + 1e-9:
                evidence_counts["long_gaps"] += 1
            last_stamp = stamp
            unique.append(sample)
    final_sample = samples[-1] if samples else {}
    final_elapsed = _number(final_info.get("stage_elapsed_s"))
    joint_samples = [sample for sample in unique if sample["errors"] is not None]
    tail_samples = [sample for sample in joint_samples
                    if final_elapsed is not None and sample["elapsed_s"] is not None
                    and sample["elapsed_s"] >= final_elapsed - 1.0 - 1e-9]
    target_samples = [sample for sample in unique if sample["distance_m"] is not None]
    nearest = min(target_samples, key=lambda sample: sample["distance_m"], default={})
    last_target = target_samples[-1] if target_samples else {}
    final_valid = final_sample.get("chronological_valid", False)
    final_errors = final_sample.get("errors") if final_valid else None
    # A duplicate valid final frame is valid final telemetry, but is not an
    # additional sample in means, minima, or event counts.
    final_distance = final_sample.get("distance_m") if final_valid else None
    lift_samples = [sample for sample in unique if sample["info"].get("training_stage") == "S1"]
    first_lift = next((sample for sample in lift_samples if sample["lift_completed"]), None)
    events, subgoals = Counter(), []
    observed = Counter()
    dropped_events = 0
    was_dropped = False
    for sample in unique:
        info, snapshot, m = sample["info"], sample["snapshot"], sample["measurements"]
        names = set()
        for source in (info.get("reward_events"), snapshot.get("events"),
                       _mapping(info.get("grasp_attachment")).get("events")):
            if isinstance(source, (list, tuple)):
                names.update(name for name in source if isinstance(name, str))
        events.update(names)
        for name in ("grasp_valid", "grasp_attempt_active", "valid_release", "placed_stable"):
            observed[name] += bool(snapshot.get(name))
        observed["attachment_active"] += bool(m.get("grasp_attachment_active",
            snapshot.get("grasp_attachment_active", _mapping(info.get("grasp_attachment")).get("active"))))
        dropped = bool("dropped" in names or "target_dropped" in names
                       or snapshot.get("failure_reason") == "object_below_workspace")
        dropped_events += bool(dropped and not was_dropped)
        was_dropped = dropped
    # These are authoritative reward-ledger events rather than reconstructed
    # geometry. Keep them even if a terminal sample has missing diagnostics.
    for row in records:
        goals = _mapping(row.get("info")).get("reward_completed_subgoals")
        if isinstance(goals, (list, tuple)):
            subgoals.extend(name for name in goals if isinstance(name, str) and name not in subgoals)
    final_snapshot = _mapping(final_info.get("task_snapshot")) or final_info
    return {
        "stage": stage, "stage_result": result, "reported_stage_result": declared,
        "policy_started": bool(policy_started), "evaluated": evaluated,
        "success": result == "SUCCESS", "error": failure,
        "completed_steps": len(records),
        "total_reward": sum(_number(row.get("reward")) or 0.0 for row in records),
        "stage_elapsed_s": final_elapsed,
        "stage_result_reason": final_info.get("stage_result_reason"),
        "terminated": bool(final_row.get("terminated")), "truncated": bool(final_row.get("truncated")),
        "initial_info": deepcopy(initial_info), "final_info": deepcopy(final_info),
        "stage_metrics": deepcopy(_mapping(final_info.get("stage_metrics"))),
        "reward_completed_subgoals": subgoals,
        "evidence": {
            "recorded_samples": len(samples), "valid_unique_samples": len(unique),
            **{name: evidence_counts[name] for name in ("invalid", "duplicate", "backwards", "long_gaps")},
        },
        "home": {
            "joint_count": 6, "valid_unique_samples": len(joint_samples),
            "mean_mae_deg": _mean([_mean(sample["errors"]) for sample in joint_samples]),
            "mean_joint_abs_error_deg": _mean_vector([sample["errors"] for sample in joint_samples]),
            "mean_joint_abs_velocity_rad_s": _mean_vector([sample["velocities"] for sample in joint_samples
                                                         if sample["velocities"] is not None]),
            "final_mae_deg": _mean(final_errors) if final_errors is not None else None,
            "final_max_abs_error_deg": max(final_errors) if final_errors else None,
            "final_joint_abs_error_deg": final_errors,
            "final_joint_abs_velocity_rad_s": final_sample.get("velocities") if final_valid else None,
            "tail_window_s": 1.0, "tail_valid_unique_samples": len(tail_samples),
            "tail_mean_mae_deg": _mean([_mean(sample["errors"]) for sample in tail_samples]),
            "tail_mean_joint_abs_error_deg": _mean_vector([sample["errors"] for sample in tail_samples]),
            "tail_mean_joint_abs_velocity_rad_s": _mean_vector([sample["velocities"] for sample in tail_samples
                                                              if sample["velocities"] is not None]),
        },
        "target": {
            "valid_unique_samples": len(target_samples),
            "minimum_distance_m": nearest.get("distance_m"), "nearest_elapsed_s": nearest.get("elapsed_s"),
            "final_distance_m": final_distance,
            "last_valid_distance_m": last_target.get("distance_m"),
            "last_valid_elapsed_s": last_target.get("elapsed_s"),
        },
        "s1_lift": {
            "completed": first_lift is not None,
            "first_completion_elapsed_s": None if first_lift is None else first_lift["elapsed_s"],
            "maximum_clearance_m": max((sample["clearance_m"] for sample in lift_samples
                                       if sample["clearance_m"] is not None), default=None),
            "maximum_hold_s": max((sample["lift_hold_s"] for sample in lift_samples
                                  if sample["lift_hold_s"] is not None), default=None),
        },
        "state_events": {
            "event_counts": dict(events), "grasp_confirmed_events": events["grasp_confirmed"],
            "grasp_lost_events": events["grasp_lost"],
            "attachment_attached_events": events["grasp_attachment_attached"],
            "attachment_detached_events": events["grasp_attachment_detached"],
            "valid_release_events": events["valid_release"],
            "dropped_events": dropped_events, "dropped_observed": dropped_events > 0,
            **{f"{name}_samples": observed[name] for name in (
                "grasp_valid", "grasp_attempt_active", "attachment_active", "valid_release", "placed_stable")},
            **{f"observed_{name}": observed[name] > 0 for name in (
                "grasp_valid", "attachment_active", "valid_release", "placed_stable")},
            "final_grasp_valid": bool(final_snapshot.get("grasp_valid")),
            "final_attachment_active": bool(final_snapshot.get("grasp_attachment_active",
                _mapping(final_info.get("grasp_attachment")).get("active"))),
            "final_valid_release": bool(final_snapshot.get("valid_release")),
            "success_physics": final_info.get("success_physics"),
        },
    }


def aggregate_results(episodes, stages) -> dict:
    """Return per-stage rates with explicit evaluated and attempted denominators."""
    if isinstance(stages, str):
        stages = [stages]
    stages = tuple(dict.fromkeys(_stage(stage) for stage in stages))
    episodes = list(episodes)
    result = {}
    for stage in stages:
        selected = [episode for episode in episodes if str(episode.get("stage", "")).upper() == stage]
        evaluated = [episode for episode in selected
                     if episode.get("evaluated") and episode.get("stage_result") in _RESULTS
                     and episode.get("error") is None]
        counts = Counter(episode["stage_result"] for episode in evaluated)
        result[stage] = {
            "attempted": len(selected), "evaluated": len(evaluated),
            "success": counts["SUCCESS"], "timeout": counts["TIMEOUT"], "failure": counts["FAILURE"],
            "errors": len(selected) - len(evaluated),
            "preparation_errors": sum(not episode.get("policy_started") for episode in selected),
            "success_rate_evaluated": counts["SUCCESS"] / len(evaluated) if evaluated else None,
            "success_rate_all_attempts": counts["SUCCESS"] / len(selected) if selected else None,
        }
    return result
