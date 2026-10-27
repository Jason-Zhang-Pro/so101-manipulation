#!/usr/bin/env python3
"""Read-only analysis of isolated simulation benchmark trials; launches nothing.

The default input is trials/, never pilot/. Each planned repeat is an independent
experimental unit. Failed/incomplete measurements remain in the CSV and matrix,
but only complete, validated trials contribute to comparative plots and means.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import struct
import tempfile
import textwrap
from typing import Any
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "so101_benchmark_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


HERE = Path(__file__).resolve().parent
PHASES = ("stepped", "free_run")
RESOURCES = ("cpu_process_percent", "cpu_host_percent", "process_pss_mib",
             "process_rss_mib", "gpu_percent", "gpu_vram_mib", "gpu_power_w",
             "gpu_memory_controller_percent", "gpu_temperature_c", "system_available_mib")
CPU_RESOURCES = {"cpu_process_percent", "cpu_host_percent"}
METRICS = {
    "per_world_rtf_mean": ("每世界实时因子", "Mean per-world real-time factor", "×"),
    "aggregate_sim_seconds_per_wall_s": ("总仿真吞吐", "Aggregate simulated-time throughput", "sim s / wall s"),
    "aggregate_control_steps_per_wall_s": ("总控制步吞吐", "Aggregate control throughput", "steps / wall s"),
    "cycle_service_p95_ms": ("整组步进服务延迟 P95", "Group step service latency P95", "ms"),
    "worker_step_p95_ms": ("世界步进服务延迟 P95", "Pooled world step service latency P95", "ms"),
    "selected_pair_skew_mean_ms": ("选中图像对时间差均值", "Selected camera-pair mean skew", "ms"),
    "selected_pair_skew_p95_ms": ("选中图像对时间差 P95", "Selected camera-pair skew P95", "ms"),
    "selected_pair_cache_arrival_gap_mean_ms": ("选中相机对缓存接收间隔均值", "Selected pair cache-arrival gap mean", "ms"),
    "selected_pair_cache_arrival_gap_p95_ms": ("选中相机对缓存接收间隔 P95", "Selected pair cache-arrival gap P95", "ms"),
    "selected_pair_reuse_percent": ("选中图像对复用比例", "Selected camera-pair reuse", "%"),
    "selected_pair_lag_p95_ms": ("选中图像对落后步尾 P95", "Selected camera-pair lag P95", "ms"),
    "top_raw_wall_fps_mean": ("顶部相机原始墙钟帧率", "Top raw wall-clock frame rate", "frames / wall s"),
    "gripper_raw_wall_fps_mean": ("手眼相机原始墙钟帧率", "Gripper raw wall-clock frame rate", "frames / wall s"),
    "top_raw_sim_fps_mean": ("顶部相机原始仿真帧率", "Top raw simulation-time frame rate", "frames / sim s"),
    "gripper_raw_sim_fps_mean": ("手眼相机原始仿真帧率", "Gripper raw simulation-time frame rate", "frames / sim s"),
    "cpu_process_percent_mean": ("受管进程 CPU 合计", "Managed process-tree CPU", "% (1 core = 100%)"),
    "process_pss_mib_mean": ("受管进程 PSS 内存合计", "Managed process-tree PSS RAM", "MiB"),
    "gpu_percent_mean": ("整卡 GPU 利用率", "Whole-device GPU utilization", "%"),
    "gpu_vram_mib_mean": ("整卡显存使用", "Whole-device VRAM used", "MiB"),
}
PLOT_GROUPS = {
    "performance_stepped": ("stepped", ["per_world_rtf_mean", "aggregate_control_steps_per_wall_s", "cycle_service_p95_ms", "worker_step_p95_ms"]),
    "performance_free_run": ("free_run", ["per_world_rtf_mean", "aggregate_sim_seconds_per_wall_s"]),
    "resources_stepped": ("stepped", ["cpu_process_percent_mean", "process_pss_mib_mean", "gpu_percent_mean", "gpu_vram_mib_mean"]),
    "resources_free_run": ("free_run", ["cpu_process_percent_mean", "process_pss_mib_mean", "gpu_percent_mean", "gpu_vram_mib_mean"]),
    "camera_stepped": ("stepped", ["top_raw_wall_fps_mean", "gripper_raw_wall_fps_mean", "top_raw_sim_fps_mean", "gripper_raw_sim_fps_mean"]),
    "camera_free_run": ("free_run", ["top_raw_wall_fps_mean", "gripper_raw_wall_fps_mean", "top_raw_sim_fps_mean", "gripper_raw_sim_fps_mean"]),
    "camera_pair_quality": ("stepped", ["selected_pair_skew_mean_ms", "selected_pair_skew_p95_ms", "selected_pair_reuse_percent", "selected_pair_lag_p95_ms"]),
}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def numbers(values):
    return [float(value) for value in values if finite(value)]


def mean(values):
    vals = numbers(values)
    return statistics.fmean(vals) if vals else None


def quantile(values, q):
    vals = numbers(values)
    return float(np.quantile(vals, q)) if vals else None


def ratio(a, b):
    return float(a) / float(b) if finite(a) and finite(b) and b > 0 else None


def safe_json(path: Path, warnings: list[str], default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        warnings.append(f"{path}: {type(error).__name__}: {error}")
        return default


def jsonl(path: Path, warnings: list[str]):
    if not path.exists():
        return []
    rows = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Expected JSON object")
            rows.append(value)
        except ValueError as error:
            warnings.append(f"{path}:{index}: {error}")
    return rows


def plan(session):
    args = session["arguments"]
    variants = ["baseline", "latest"] if args["variant"] == "both" else [args["variant"]]
    guis = [False, True] if args["gui"] == "both" else [args["gui"] == "on"]
    result = []
    for variant, world_count, gui, repeat in itertools.product(variants, args["worlds"], guis, range(1, int(args["repeats"]) + 1)):
        result.append(dict(case_id=f"{variant}_w{world_count}_gui{int(gui)}_r{repeat}",
                           variant=variant, world_count=int(world_count), gui=bool(gui), repeat=repeat))
    return result


def weighted_quantile(values, weights, q):
    ordered = sorted(zip(values, weights))
    target = sum(weights) * q
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def resource_metrics(records, begin, end):
    """Integrate only timestamps inside the root's phase interval.

    CPU samples describe the previous sampler interval. Other metrics use a
    piecewise-constant left sample, capped at the next in-phase timestamp/end.
    We never invent a value before the first in-phase sample. CPU intervals
    straddling the start are excluded because their average includes warmup.
    Coverage is retained separately for each metric, and peaks are sampled peaks.
    """
    result = {}
    if not finite(begin) or not finite(end) or end <= begin:
        return result
    ordered = sorted((r for r in records if finite(r.get("monotonic_s"))), key=lambda r: r["monotonic_s"])
    selected = [r for r in ordered if begin <= r["monotonic_s"] <= end]
    result["resource_sample_count"] = len(selected)
    result["memory_missing_process_sample_count"] = sum(int(r.get("memory_sample_missing_processes", 0) or 0) > 0 for r in selected)
    for metric in RESOURCES:
        vals, weights = [], []
        sample_vals = []
        for index, row in enumerate(selected):
            value = row.get(metric)
            if not finite(value):
                continue
            if metric in {"process_pss_mib", "process_rss_mib"} and row.get("memory_sample_missing_processes", 0):
                # An incomplete process sum is not comparable to a full sum.
                continue
            if metric in CPU_RESOURCES:
                if index == 0:
                    continue
                duration = row["monotonic_s"] - selected[index - 1]["monotonic_s"]
            else:
                right = selected[index + 1]["monotonic_s"] if index + 1 < len(selected) else end
                duration = min(right - row["monotonic_s"], 3.0)
            sample_vals.append(float(value))
            # Refuse to carry a stale sampler value over an unexpectedly long gap.
            duration = max(0.0, duration)
            if duration > 0:
                vals.append(float(value))
                weights.append(duration)
        duration_sum = sum(weights)
        result[metric + "_mean"] = sum(v*w for v, w in zip(vals, weights)) / duration_sum if duration_sum else None
        result[metric + "_p95"] = weighted_quantile(vals, weights, .95) if duration_sum else None
        result[metric + "_peak"] = max(sample_vals) if sample_vals else None
        result[metric + "_coverage_fraction"] = duration_sum / (end - begin)
    return result


def camera_gate_events(data):
    """Distinct rejected reuse-gate events, never fabricated selected frames."""
    events = set()

    def add(error, domain=None):
        if not isinstance(error, dict):
            return
        message = str(error.get("error_message", ""))
        if "camera frame reuse limit exceeded" not in message.lower():
            return
        events.add((error.get("domain_id", domain), error.get("operation"),
                    error.get("error_type"), message,
                    (error.get("clock") or {}).get("sim_time_s")))

    for phase in data.get("measurements", {}).values():
        for measurement in phase.get("end_responses", []):
            for error in measurement.get("errors", []):
                add(error, measurement.get("domain_id"))
    for measurement in data.get("failure_diagnostics", []):
        if measurement.get("type") == "measure_result":
            for error in measurement.get("errors", []):
                add(error, measurement.get("domain_id"))
    try:
        add(json.loads(data.get("error", "")))
    except (ValueError, TypeError):
        pass
    return len(events)


def partial_measurements(data):
    """Only explicit end_measure replies; pending step replies are not added."""
    phases = defaultdict(list)
    seen = set()
    for response in data.get("failure_diagnostics", []):
        if response.get("type") != "measure_result":
            continue
        identity = (response.get("worker_pid"), response.get("domain_id"),
                    response.get("wall_start_monotonic_s"), response.get("wall_end_monotonic_s"))
        if identity in seen:
            continue
        seen.add(identity)
        phase = "free_run" if response.get("world_running_at_start") else "stepped"
        phases[phase].append(response)
    return phases


def summarize_world(measurement, meta, phase, world_index):
    row = dict(meta, phase=phase, world_index=world_index, domain_id=measurement.get("domain_id"),
               worker_pid=measurement.get("worker_pid"), wall_seconds=measurement.get("wall_duration_s"),
               sim_seconds=measurement.get("sim_duration_s"), successful_steps=measurement.get("successful_steps"),
               attempted_steps=measurement.get("attempted_steps"), failed_steps=measurement.get("failed_steps"),
               worker_error_count=len(measurement.get("errors", [])))
    row["camera_reuse_gate_event_count"] = sum("camera frame reuse limit exceeded" in str(e.get("error_message", "")).lower() for e in measurement.get("errors", []))
    row["rtf"] = ratio(row["sim_seconds"], row["wall_seconds"])
    steps = measurement.get("steps", [])
    for camera in ("top", "gripper"):
        raw = measurement.get("raw_frames", {}).get(camera, {})
        count = raw.get("count")
        row[camera + "_raw_count"] = count
        sources = numbers(raw.get("source_timestamps_s", []))
        arrivals = numbers(raw.get("received_monotonic_s", []))
        # Unique source timestamps identify distinct delivered sensor frames.
        # A callback is not assumed to be a newly rendered frame.
        unique_count = len(set(sources)) if sources or count == 0 else None
        row[camera + "_unique_source_frame_count"] = unique_count
        row[camera + "_raw_wall_fps"] = ratio(unique_count, row["wall_seconds"])
        row[camera + "_raw_sim_fps"] = ratio(unique_count, row["sim_seconds"])
        row[camera + "_received_callback_wall_fps"] = ratio(count, row["wall_seconds"])
        row[camera + "_received_callback_sim_fps"] = ratio(count, row["sim_seconds"])
        row[camera + "_source_timestamp_count"] = len(sources)
        row[camera + "_source_duplicates"] = len(sources) - len(set(sources))
        row[camera + "_duplicate_timestamp_percent"] = 100*(len(sources)-len(set(sources)))/len(sources) if sources else None
        row[camera + "_missing_source_timestamp_count"] = max(0, int(count)-len(sources)) if finite(count) else None
        row[camera + "_source_backwards_count"] = sum(b < a for a, b in zip(sources, sources[1:]))
        row[camera + "_source_interval_p95_ms"] = quantile([1000*(b-a) for a, b in zip(sources, sources[1:])], .95)
        row[camera + "_arrival_interval_p95_ms"] = quantile([1000*(b-a) for a, b in zip(arrivals, arrivals[1:])], .95)
    step_ms = [1000*s["step_wall_duration_s"] for s in steps if finite(s.get("step_wall_duration_s"))]
    skews, reuses, lags, sim_errors, arrival_gaps = [], [], [], [], []
    for step in steps:
        info = step.get("info", {})
        if finite(info.get("camera_skew_s")):
            skews.append(abs(info["camera_skew_s"])*1000)
        top_received, gripper_received = info.get("top_image_received_monotonic_s"), info.get("gripper_image_received_monotonic_s")
        if finite(top_received) and finite(gripper_received):
            arrival_gaps.append(abs(top_received-gripper_received)*1000)
        if isinstance(info.get("camera_pair_reused"), bool):
            reuses.append(info["camera_pair_reused"])
        if finite(info.get("camera_lag_from_step_end_s")):
            lags.append(info["camera_lag_from_step_end_s"]*1000)
        if finite(info.get("sim_step_error_s")):
            sim_errors.append(abs(info["sim_step_error_s"])*1000)
    row.update(worker_step_mean_ms=mean(step_ms), worker_step_p95_ms=quantile(step_ms, .95),
               selected_pair_count=len(steps), selected_pair_skew_sample_count=len(skews),
               selected_pair_skew_mean_ms=mean(skews), selected_pair_skew_p95_ms=quantile(skews, .95),
               selected_pair_skew_max_ms=max(skews) if skews else None,
               selected_pair_cache_arrival_gap_sample_count=len(arrival_gaps),
               selected_pair_cache_arrival_gap_mean_ms=mean(arrival_gaps),
               selected_pair_cache_arrival_gap_p95_ms=quantile(arrival_gaps,.95),
               selected_pair_reuse_known_count=len(reuses), selected_pair_reuse_count=sum(reuses),
               selected_pair_reuse_percent=100*sum(reuses)/len(reuses) if reuses else None,
               selected_pair_lag_p95_ms=quantile(lags, .95),
               absolute_sim_step_error_max_ms=max(sim_errors) if sim_errors else None)
    return row, {"step_ms": step_ms, "skews": skews, "reuses": reuses, "lags": lags,"arrival_gaps":arrival_gaps}


def analyze(folder, session, warnings):
    trials, worlds, cases = [], [], []
    planned = plan(session)
    planned_ids = {item["case_id"] for item in planned}
    extras = sorted(path.parent.name for path in folder.glob("*/result.json") if path.parent.name not in planned_ids)
    if extras:
        warnings.append("Directories outside the session plan were excluded: " + ", ".join(extras))
    phases = (["stepped"] if session["arguments"].get("steps", 0) > 0 else []) + (["free_run"] if session["arguments"].get("free_seconds", 0) > 0 else [])
    for meta in planned:
        case_dir = folder / meta["case_id"]
        path = case_dir / "result.json"
        data = safe_json(path, warnings, {}) if path.exists() else {}
        if not isinstance(data, dict):
            warnings.append(f"{path}: not a JSON object")
            data = {}
        status = data.get("status", "missing" if not path.exists() else "invalid_json")
        identity_ok = all(data.get(key) == value for key, value in meta.items()) if data else False
        if data and not identity_ok:
            warnings.append(f"{path}: identity differs from planned case")
            status = "identity_mismatch"
        if data.get("cleanup_error"):
            status = "cleanup_failed"
        case = dict(meta, status=status, error_type=data.get("error_type", ""),
                    error=data.get("error", ""), cleanup_error=data.get("cleanup_error", ""),
                    started_at_utc=data.get("started_at_utc", ""), ended_at_utc=data.get("ended_at_utc", ""),
                    camera_reuse_gate_event_count=camera_gate_events(data))
        for camera in ("top", "gripper"):
            shapes = {tuple(r[f"{camera}_policy_shape"]) for r in data.get("startup_responses", [])
                      if isinstance(r,dict) and isinstance(r.get(f"{camera}_policy_shape"),list)}
            case[f"{camera}_policy_shapes"] = " / ".join("×".join(str(v) for v in shape) for shape in sorted(shapes))
        cases.append(case)
        resources = jsonl(case_dir / "resources.jsonl", warnings)
        cycles = jsonl(case_dir / "steps.jsonl", warnings)
        partial = partial_measurements(data)
        domain_indices = {spec.get("domain_id"): index for index, spec in enumerate(data.get("world_specs", []))}
        for phase in phases:
            measurement = data.get("measurements", {}).get(phase, {})
            source_kind = "complete_phase" if measurement else "failed_partial" if partial.get(phase) else "missing"
            row = dict(case, phase=phase, phase_recorded=bool(measurement or partial.get(phase)), included=False,
                       measurement_source=source_kind,
                       root_wall_seconds=measurement.get("wall_seconds"),
                       root_begin_monotonic_s=measurement.get("begin_monotonic_s"),
                       root_end_monotonic_s=measurement.get("end_monotonic_s"),
                       completed_group_steps=measurement.get("completed_steps"),
                       planned_group_steps=data.get("planned_steps", session["arguments"].get("steps")))
            # Failed prefixes never invent parent timing/throughput. Their
            # worker-window intersection can independently define an explicitly
            # inferred resource diagnostic window, recorded below.
            ending = measurement.get("end_responses", []) if measurement else partial.get(phase, [])
            rows, raw = [], defaultdict(list)
            for index, response in enumerate(ending):
                if response.get("type") != "measure_result":
                    warnings.append(f"{meta['case_id']}/{phase}: invalid end response {index}")
                    continue
                world_index = domain_indices.get(response.get("domain_id"), index)
                wr, arrays = summarize_world(response, case, phase, world_index)
                wr["measurement_source"] = source_kind
                rows.append(wr)
                for name, values in arrays.items():
                    raw[name].extend(values)
            valid = (status == "completed" and identity_ok and len(rows) == meta["world_count"]
                     and finite(row["root_wall_seconds"]) and row["root_wall_seconds"] > 0
                     and all(finite(r["wall_seconds"]) and r["wall_seconds"] > 0
                             and finite(r["sim_seconds"]) and r["sim_seconds"] > 0
                             and not r["worker_error_count"] and not r["failed_steps"] for r in rows))
            if phase == "stepped":
                valid = valid and row["completed_group_steps"] == row["planned_group_steps"] and all(
                    r["successful_steps"] == row["planned_group_steps"] for r in rows)
            row["included"] = bool(valid)
            if status == "completed" and not valid:
                warnings.append(f"{meta['case_id']}/{phase}: completed result failed phase validation")
            for wr in rows:
                wr["included"] = row["included"]
                worlds.append(wr)
            row["measured_world_count"] = len(rows)
            row["observed_successful_world_steps"] = sum(numbers(r["successful_steps"] for r in rows)) if rows else None
            row["observed_attempted_world_steps"] = sum(numbers(r["attempted_steps"] for r in rows)) if rows else None
            row["observed_failed_world_steps"] = sum(numbers(r["failed_steps"] for r in rows)) if rows else None
            row["selected_pair_reuse_known_count"] = len(raw["reuses"])
            row["selected_pair_reuse_count"] = sum(raw["reuses"])
            row["selected_pair_skew_max_ms"] = max(raw["skews"]) if raw["skews"] else None
            row["per_world_rtf_mean"] = mean(r["rtf"] for r in rows)
            row["per_world_rtf_min"] = min(numbers(r["rtf"] for r in rows), default=None)
            row["per_world_rtf_max"] = max(numbers(r["rtf"] for r in rows), default=None)
            row["observed_world_wall_seconds_mean"] = mean(r["wall_seconds"] for r in rows)
            row["aggregate_sim_seconds_per_wall_s"] = ratio(sum(numbers(r["sim_seconds"] for r in rows)), row["root_wall_seconds"]) if rows else None
            row["aggregate_control_steps_per_wall_s"] = ratio(sum(numbers(r["successful_steps"] for r in rows)), row["root_wall_seconds"]) if phase == "stepped" and rows else None
            for camera in ("top", "gripper"):
                for basis in ("wall", "sim"):
                    metric = f"{camera}_raw_{basis}_fps"
                    row[metric + "_mean"] = mean(r[metric] for r in rows)
                    row[metric + "_min"] = min(numbers(r[metric] for r in rows), default=None)
            row.update(worker_step_p95_ms=quantile(raw["step_ms"], .95),
                       selected_pair_skew_mean_ms=mean(raw["skews"]),
                       selected_pair_skew_p95_ms=quantile(raw["skews"], .95),
                       selected_pair_cache_arrival_gap_sample_count=len(raw["arrival_gaps"]),
                       selected_pair_cache_arrival_gap_mean_ms=mean(raw["arrival_gaps"]),
                       selected_pair_cache_arrival_gap_p95_ms=quantile(raw["arrival_gaps"],.95),
                       selected_pair_lag_p95_ms=quantile(raw["lags"], .95),
                       selected_pair_reuse_percent=100*sum(raw["reuses"])/len(raw["reuses"]) if raw["reuses"] else None,
                       selected_pair_count=sum(r["selected_pair_count"] for r in rows))
            if phase == "stepped":
                begin, end = row["root_begin_monotonic_s"], row["root_end_monotonic_s"]
                selected = [r for r in cycles if finite(begin) and finite(end) and finite(r.get("monotonic_s")) and begin <= r["monotonic_s"] <= end]
                row["cycle_service_p95_ms"] = quantile([1000*r["cycle_service_s"] for r in selected if finite(r.get("cycle_service_s"))], .95)
                row["cycle_record_count"] = len(selected)
            resource_begin, resource_end = row["root_begin_monotonic_s"], row["root_end_monotonic_s"]
            resource_source = "parent_phase"
            interval_count = 0
            if not (finite(resource_begin) and finite(resource_end) and resource_end > resource_begin):
                resource_begin, resource_end = None, None
                resource_source = "unavailable"
                intervals = [(response.get("wall_start_monotonic_s"), response.get("wall_end_monotonic_s"))
                    for response in ending if response.get("type") == "measure_result"]
                intervals = [(a,b) for a,b in intervals if finite(a) and finite(b) and b > a]
                interval_count = len(intervals)
                if source_kind == "failed_partial" and interval_count == meta["world_count"]:
                    left, right = max(a for a,b in intervals), min(b for a,b in intervals)
                    if right > left:
                        resource_begin, resource_end = left, right
                        resource_source = "worker_intersection_inferred"
            row.update(resource_window_source=resource_source,
                resource_window_begin_monotonic_s=resource_begin,
                resource_window_end_monotonic_s=resource_end,
                resource_window_wall_seconds=resource_end-resource_begin if finite(resource_begin) and finite(resource_end) else None,
                resource_window_worker_interval_count=interval_count)
            row.update(resource_metrics(resources, resource_begin, resource_end))
            trials.append(row)
    return cases, trials, worlds


def reliability(cases, trials):
    """One row per planned attempt, including the failed measurement prefixes."""
    by_case = defaultdict(list)
    for trial in trials:
        by_case[trial["case_id"]].append(trial)
    rows = []
    for case in cases:
        segments = by_case[case["case_id"]]
        known = sum(r.get("selected_pair_reuse_known_count", 0) for r in segments)
        reused = sum(r.get("selected_pair_reuse_count", 0) for r in segments)
        failed = numbers(r.get("observed_failed_world_steps") for r in segments)
        measured = any(r.get("measured_world_count", 0) for r in segments)
        rows.append(dict(case,
            trial_completed=int(case["status"] == "completed"),
            trial_failed=int(case["status"] not in {"completed", "missing", "starting"}),
            trial_missing=int(case["status"] == "missing"),
            observed_measurement_segments=sum(r.get("measured_world_count", 0) for r in segments),
            returned_selected_pair_count=sum(r.get("selected_pair_count", 0) for r in segments),
            selected_pair_reuse_known_count=known, selected_pair_reuse_count=reused,
            selected_pair_reuse_percent=100*reused/known if known else None,
            selected_pair_skew_max_ms=max(numbers(r.get("selected_pair_skew_max_ms") for r in segments), default=None),
            observed_successful_world_steps=sum(numbers(r.get("observed_successful_world_steps") for r in segments)) if measured else None,
            observed_failed_world_steps=sum(failed) if failed else None,
            failed_partial_segments=sum(r.get("measurement_source") == "failed_partial" for r in segments),
            rule="Includes returned selected steps from successful and failed attempts; rejected steps have no selected frame. Failed partial performance is diagnostic only."))
    return rows


def aggregate(trials):
    groups = defaultdict(list)
    for row in trials:
        groups[(row["variant"], row["world_count"], row["gui"], row["phase"])].append(row)
    summary = []
    for (variant, count, gui, phase), rows in sorted(groups.items()):
        for metric in METRICS:
            vals = numbers(r.get(metric) for r in rows if r["included"])
            summary.append(dict(variant=variant, world_count=count, gui=gui, phase=phase, metric=metric,
                n_planned=len(rows), n_complete=sum(r["included"] for r in rows), n_metric=len(vals),
                mean=statistics.fmean(vals) if vals else None,
                sd=statistics.stdev(vals) if len(vals) > 1 else None,
                minimum=min(vals) if vals else None, maximum=max(vals) if vals else None,
                individual_values=json.dumps(vals)))
    comparisons = []
    keyed = {(r["variant"], r["world_count"], r["gui"], r["phase"], r["repeat"]): r for r in trials}
    for count, gui, phase in sorted({(r["world_count"], r["gui"], r["phase"]) for r in trials}):
        repeats = sorted({r["repeat"] for r in trials})
        for metric in METRICS:
            paired = []
            for repeat in repeats:
                a, b = keyed.get(("baseline", count, gui, phase, repeat)), keyed.get(("latest", count, gui, phase, repeat))
                if a and b and a["included"] and b["included"]:
                    value = ratio(b.get(metric), a.get(metric))
                    if value is not None:
                        paired.append(value)
            comparisons.append(dict(world_count=count, gui=gui, phase=phase, metric=metric,
                n_pairs=len(paired), latest_over_baseline_mean=statistics.fmean(paired) if paired else None,
                latest_over_baseline_sd=statistics.stdev(paired) if len(paired) > 1 else None,
                mean_percent_change=100*(statistics.fmean(paired)-1) if paired else None,
                individual_pair_ratios=json.dumps(paired)))
    return summary, comparisons


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def configure_plots(language):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.dpi": 110,
                         "savefig.dpi": 180, "svg.fonttype": "path", "pdf.fonttype": 42})
    if language == "zh":
        font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams["axes.unicode_minus"] = False
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"


def label(metric, language):
    item = METRICS[metric]
    return item[0 if language == "zh" else 1] + "\n(" + item[2] + ")"


def plot_label(metric, language):
    item = METRICS[metric]
    title = item[0] if language == "zh" else textwrap.fill(item[1], width=24)
    return title + "\n(" + item[2] + ")"


def plot_all(trials, output, language, prefix=""):
    configure_plots(language)
    files = []
    worlds = sorted({r["world_count"] for r in trials})
    guis = sorted({r["gui"] for r in trials})
    for name, (phase, metrics) in PLOT_GROUPS.items():
        if not any(r["phase"] == phase for r in trials):
            continue
        fig, axes = plt.subplots(len(metrics), len(guis), figsize=(6.1*len(guis), 2.75*len(metrics)), squeeze=False)
        for rindex, metric in enumerate(metrics):
            for cindex, gui in enumerate(guis):
                ax = axes[rindex, cindex]
                any_data = False
                for variant, color, shift in (("baseline", "#52677e", -.065), ("latest", "#087f8c", .065)):
                    xs, ys = [], []
                    for world in worlds:
                        rows = [r for r in trials if r["included"] and r["variant"] == variant and r["world_count"] == world and r["gui"] == gui and r["phase"] == phase]
                        vals = numbers(r.get(metric) for r in rows)
                        if not vals:
                            continue
                        any_data = True
                        x, y = world + shift, statistics.fmean(vals)
                        xs.append(x); ys.append(y)
                        spread = np.linspace(-.026, .026, len(vals)) if len(vals) > 1 else [0]
                        ax.scatter([x+s for s in spread], vals, color=color, alpha=.6, s=22, zorder=3)
                        ax.errorbar(x, y, yerr=statistics.stdev(vals) if len(vals) > 1 else None,
                                    color=color, capsize=4, marker="D", markersize=4, zorder=4)
                    title = ("历史资产基线" if variant == "baseline" else "当前资产") if language == "zh" else ("Historical-asset baseline" if variant == "baseline" else "Current assets")
                    ax.plot(xs, ys, color=color, lw=1.5, label=title)
                ax.set_xticks(worlds)
                ax.set_xlim(min(worlds)-.4, max(worlds)+.4)
                ax.set_xlabel("并行世界数" if language == "zh" else "Concurrent worlds")
                ax.set_ylabel(plot_label(metric, language))
                ax.set_title(("GUI 开启" if gui else "GUI 关闭") if language == "zh" else ("GUI on" if gui else "GUI off"))
                if not any_data:
                    ax.text(.5, .5, "无有效试验" if language == "zh" else "No valid trials", transform=ax.transAxes, ha="center", color="#667085")
                if any_data and ax.get_ylim()[0] > 0:
                    ax.set_ylim(bottom=0)
                if rindex == 0:
                    ax.legend(fontsize=8, loc="best")
        phase_label = ("确定性步进（含 30 Hz 墙钟限速）" if phase == "stepped" else "自由运行（配置 RTF=1）") if language == "zh" else ("Deterministic stepping (30 wall-Hz pacing)" if phase == "stepped" else "Free run (configured RTF=1)")
        subtitle = "点：单次试验；菱形：均值；误差棒：重复间样本标准差；缺失不补零" if language == "zh" else "Dots: individual trials; diamonds: mean; bars: sample SD across repeats; missing values are not zero"
        fig.suptitle(phase_label + "\n" + subtitle, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .955))
        basename = f"{prefix}{name}_{language}"
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / "figures" / f"{basename}.{extension}", bbox_inches="tight")
        plt.close(fig)
        files.append((name, output / "figures" / f"{basename}.png"))
    return files


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}" if finite(value) else "—"


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+html.escape(str(h))+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def high_load_failure(case):
    return (case.get("status") == "failed" and case.get("variant") == "baseline"
            and case.get("world_count", 0) >= 3 and case.get("camera_reuse_gate_event_count", 0) > 0)


def completion_label(case, language):
    zh = language == "zh"
    if high_load_failure(case):
        return "高负载下无法完成（旧帧复用超限）" if zh else "Unable to complete under high load (stale-frame reuse limit)"
    labels = {"completed":("完成","Completed"),"failed":("失败","Failed"),
        "missing":("缺失","Missing"),"starting":("进行中","In progress"),
        "cleanup_failed":("清理失败","Cleanup failed"),
        "identity_mismatch":("标识不匹配","Identity mismatch"),"invalid_json":("JSON无效","Invalid JSON")}
    return labels.get(case.get("status"),(case.get("status","unknown"),)*2)[0 if zh else 1]


def brief_error(case):
    raw = case.get("error") or case.get("cleanup_error") or ""
    try:
        parsed = json.loads(raw) if isinstance(raw,str) else raw
    except (ValueError,TypeError):
        parsed = None
    if isinstance(parsed,dict):
        return " · ".join(f"{key}={parsed[key]}" for key in ("operation","domain_id","error_message") if parsed.get(key) is not None)
    return str(raw).split("Traceback (most recent call last):",1)[0].strip()[:600]


def asset_reproducibility(assets, warnings):
    """Count collision instances in the archived asset inputs, excluding camera links."""
    arm_links = {r["link"] for r in assets.get("original_arm_geometry_proof",{}).get("instances",[])}
    result = {}
    for variant, definition in assets.get("variants",{}).items():
        if not arm_links:
            continue
        try:
            relative = "share/so101_description/urdf/so101.urdf.xacro"
            # Archived manifests retain acquisition-time paths after relocation.
            current_source = HERE / "assets" / variant / relative
            source = current_source if current_source.exists() else Path(definition["prefix"]) / relative
            content = source.read_bytes()
            expected_hash = definition["files_sha256"][relative]
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError("URDF no longer matches archived manifest hash")
            mesh_lookup = {r["uri"]:r for r in definition["asset_validation"]["meshes"]}
            mesh_counts = {}
            instance_count = triangle_count = 0
            for link in ET.fromstring(content).findall("link"):
                if link.get("name") not in arm_links:
                    continue
                for mesh in link.findall("collision/geometry/mesh"):
                    uri = mesh.get("filename")
                    if uri not in mesh_counts:
                        entry = mesh_lookup[uri]
                        stl = Path(entry["resolved"]).read_bytes()
                        if hashlib.sha256(stl).hexdigest() != entry["sha256"]:
                            raise ValueError("collision STL no longer matches archived manifest hash")
                        count = struct.unpack("<I",stl[80:84])[0]
                        if len(stl) != 84 + 50*count:
                            raise ValueError("collision mesh is not a validated binary STL")
                        mesh_counts[uri] = count
                    instance_count += 1
                    triangle_count += mesh_counts[uri]
            result[variant] = dict(arm_collision_instances=instance_count,
                arm_collision_instance_triangles=triangle_count, source=str(source),
                hashes_match_manifest=True, excludes_camera_links=True)
        except (KeyError,ValueError,OSError,ET.ParseError,struct.error) as exc:
            warnings.append(f"{variant}: collision complexity metadata unavailable: {exc}")
    return result


def measured_findings(cases, supplement, language):
    """Describe observed outcomes; use only the independent cohort for its table."""
    t = lambda a,b: a if language == "zh" else b
    completed = sum(r["status"] == "completed" for r in cases)
    failed = sum(r["status"] == "failed" for r in cases)
    baseline_low = [r for r in cases if r["variant"] == "baseline" and r["world_count"] in (1,2)]
    baseline_high = [r for r in cases if r["variant"] == "baseline" and r["world_count"] in (3,4)]
    latest = [r for r in cases if r["variant"] == "latest"]
    passed_low = sum(r["status"] == "completed" for r in baseline_low)
    failed_high = sum(high_load_failure(r) for r in baseline_high)
    passed_latest = sum(r["status"] == "completed" for r in latest)
    parts = ['<section class="findings"><h2>'+t("实测结论", "Measured findings")+'</h2>',
        '<p>'+t(f'主矩阵计划 {len(cases)} 次，完成 {completed} 次、失败 {failed} 次。历史资产（baseline）的 1/2-world 条件完成 {passed_low}/{len(baseline_low)} 次；3/4-world 条件中有 {failed_high}/{len(baseline_high)} 次因旧帧复用超限而无法完成受控采样。当前资产（latest）完成 {passed_latest}/{len(latest)} 次。',
                 f'The main matrix has {len(cases)} planned trials: {completed} completed and {failed} failed. Historical assets (baseline) completed {passed_low}/{len(baseline_low)} trials with 1/2 worlds; {failed_high}/{len(baseline_high)} trials with 3/4 worlds could not complete controlled sampling because of the stale-frame reuse limit. Current assets (latest) completed {passed_latest}/{len(latest)} trials.')+'</p>']
    if failed and sum(high_load_failure(r) for r in cases) == failed:
        parts.append('<p>'+t("这些短时试验中，记录到的失败是相机旧帧复用门槛触发，记录未提供进程崩溃的证据；这也不是长期稳定性保证。主矩阵失败条件不用于计算完整运行性能改善。",
            "In these short trials, the recorded failures were camera stale-frame reuse gate events; the records do not establish a process crash. This is not a long-duration stability guarantee. Failed main-matrix conditions are excluded from complete-run performance comparisons.")+'</p>')
    if supplement is not None:
        supplement_cases = supplement["cases"]
        complete_supplement = sum(r["status"] == "completed" for r in supplement_cases)
        parts.append('<p>'+t(f'独立自由运行补充矩阵完成 {complete_supplement}/{len(supplement_cases)} 次。以下四世界结果仅来自该补充矩阵，与主矩阵分开统计；表内为重复试验均值 ± 样本标准差，顺序为 baseline → latest。',
            f'The independent free-run supplement completed {complete_supplement}/{len(supplement_cases)} trials. The four-world results below come only from that separate cohort. Values are repeat-level mean ± sample standard deviation, shown as baseline → latest.')+'</p>')
        index = {(r["variant"],r["world_count"],r["gui"],r["phase"],r["metric"]):r for r in supplement["summary"]}
        metrics = [("per_world_rtf_mean",3),("cpu_process_percent_mean",1),
                   ("process_pss_mib_mean",1),("gpu_percent_mean",1)]
        rows = []
        for gui in (False,True):
            if not any((v,4,gui,"free_run",metrics[0][0]) in index for v in ("baseline","latest")):
                continue
            row = ["on" if gui else "off"]
            for metric,digits in metrics:
                entries = [index.get((variant,4,gui,"free_run",metric),{}) for variant in ("baseline","latest")]
                values = []
                for entry in entries:
                    value = fmt(entry.get("mean"),digits)
                    if finite(entry.get("sd")):
                        value += " ± " + fmt(entry["sd"],digits)
                    values.append(value)
                row.append(" → ".join(values) + f' (n={entries[0].get("n_metric",0)}/{entries[1].get("n_metric",0)})')
            rows.append(row)
        if rows:
            parts.append(table(["GUI",t("四世界：每 world RTF","Four worlds: per-world RTF"),
                t("CPU 进程合计 %（单核 100%）","Summed process CPU % (one core = 100%)"),
                t("PSS 内存 MiB","PSS RAM MiB"),t("整卡 GPU 使用率 %","Whole-device GPU utilization %")],rows))
        parts.append('<p>'+t("CPU 是管理进程的使用率之和，可超过 100%；PSS 表示这些进程的比例内存总和；GPU 是整卡读数，不能归因于某一个 world。补充自由运行采用不同前置负载，其通过不覆盖主矩阵的失败。",
            "CPU is summed across managed processes and may exceed 100%; PSS is their total proportional memory. GPU utilization is measured for the whole device and cannot be attributed to one world. Independent free run has a different preceding workload; passing it does not replace the main-matrix failures.")+'</p>')
    parts.append('<p>'+t("相机配置的 60/45 Hz 不是墙钟 FPS，也不能直接等同于实测仿真 FPS；本报告分别用有效源时间戳帧数和各 world 的实测时间计算帧率。",
        "The configured camera rates of 60/45 Hz are not wall-clock FPS and need not equal measured simulation-time FPS. This report calculates rates from distinct valid source timestamps and each world's measured time intervals.")+'</p></section>')
    return ''.join(parts)


def supplement_section(supplement, language):
    """A separately analyzed cohort; never append its rows to the main trials."""
    if supplement is None:
        return ""
    zh = language == "zh"
    t = lambda a,b: a if zh else b
    args = supplement["session"]["arguments"]
    cases, trials = supplement["cases"], supplement["trials"]
    statuses = Counter(r["status"] for r in cases)
    title = t("附加实验：独立自由运行（单独统计）", "Supplement: independent free run, analyzed separately")
    parts = ['<section class="supplement"><h2>'+title+'</h2>',
        '<div class="notice">'+t("这一补充矩阵不替换失败的主试验，也不与主矩阵合并均值、标准差或提升比值；主试验完成数保持原值。", "This supplemental matrix neither replaces failed main trials nor shares their means, standard deviations or improvement ratios. Main-trial completion counts remain unchanged.")+'</div>',
        '<p>'+t(f'补充方案从新世界开始，执行 {args.get("warmup_steps", "—")} 个 warmup 控制步，再自由运行预热 3 秒，然后测量 {args.get("free_seconds", "—")} 秒；正式计分 step 数为 {args.get("steps", "—")}。主矩阵则先完成长步进相位才进入自由运行，二者具有不同前置历史，不能混为同一条件。', f'The supplemental protocol starts fresh worlds, performs {args.get("warmup_steps", "—")} warmup control steps, then 3 s of free-run warmup before a {args.get("free_seconds", "—")} s measurement; measured deterministic steps are {args.get("steps", "—")}. Main-matrix free run follows the long stepped phase. Their preceding workloads differ and the cohorts must not be pooled.')+'</p>',
        '<p>'+t("独立自由运行是不同负载模式的补充测量，不是对失败的 600-step 确定性步进条件反复重试；自由运行成功也不能改写主矩阵的高负载失败判定。", "Independent free run is a supplemental measurement of a different load mode, not another retry of the failed 600-step deterministic condition. A successful free-run trial does not reverse the high-load failure recorded in the main matrix.")+'</p>',
        '<p>'+t(f'补充计划 {len(cases)} 次，完成 {statuses["completed"]} 次，失败 {sum(v for k,v in statuses.items() if k not in {"completed","missing","starting"})} 次，缺失 {statuses["missing"]} 次；仅有效补充试验参与本节统计。', f'Supplemental trials: {len(cases)} planned, {statuses["completed"]} completed, {sum(v for k,v in statuses.items() if k not in {"completed","missing","starting"})} failed, {statuses["missing"]} missing. Only valid supplemental trials enter this section’s statistics.')+'</p>',
        '<p>'+t("数据目录：", "Data directory: ")+html.escape(str(supplement["source"]))+'</p>']
    grouped = defaultdict(list)
    for row in cases:
        grouped[(row["variant"],row["world_count"],row["gui"])].append(row)
    matrix = [[variant,world,"on" if gui else "off",' · '.join(f'R{r["repeat"]}: {completion_label(r,language)}' for r in sorted(rows,key=lambda r:r["repeat"]))]
              for (variant,world,gui),rows in sorted(grouped.items())]
    parts.append(table([t("补充资产","Supplement assets"),t("世界数","Worlds"),"GUI",t("补充各次重复","Supplemental repeats")],matrix))
    metrics = ["per_world_rtf_mean","top_raw_wall_fps_mean","gripper_raw_wall_fps_mean",
               "cpu_process_percent_mean","process_pss_mib_mean","gpu_percent_mean","gpu_vram_mib_mean"]
    index = {(r["variant"],r["world_count"],r["gui"],r["phase"],r["metric"]):r for r in supplement["summary"]}
    values = []
    for key in sorted({(r["variant"],r["world_count"],r["gui"],r["phase"]) for r in supplement["summary"]}):
        row = [key[0],key[1],"on" if key[2] else "off",key[3]]
        for metric in metrics:
            result = index[(*key,metric)]
            value = fmt(result["mean"],2)
            if finite(result["sd"]):
                value += " ± " + fmt(result["sd"],2)
            if result["n_metric"]:
                value += f' (n={result["n_metric"]})'
            row.append(value)
        values.append(row)
    parts.append(table([t("补充资产","Supplement assets"),t("世界数","Worlds"),"GUI",t("相位","Phase")]+
                       [label(metric,language).replace("\n"," ") for metric in metrics],values))
    for name, path in supplement.get("figures",{}).get(language,[]):
        name_text = {"performance_free_run":t("独立自由运行性能","Independent free-run performance"),
                     "resources_free_run":t("独立自由运行资源使用","Independent free-run resources"),
                     "camera_free_run":t("独立自由运行相机帧率","Independent free-run camera rates")}.get(name,name)
        parts.append('<figure><h3>'+name_text+'</h3><img alt="'+name_text+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"></figure>')
    comparison_rows = [[r["world_count"],"on" if r["gui"] else "off",METRICS[r["metric"]][0 if zh else 1],
        r["n_pairs"],fmt(r["latest_over_baseline_mean"]),fmt(r["mean_percent_change"],1)]
        for r in supplement["comparisons"] if r["metric"] in {"per_world_rtf_mean","cpu_process_percent_mean","process_pss_mib_mean","gpu_percent_mean","gpu_vram_mib_mean"}]
    parts.append('<h3>'+t("仅补充组内的配对比值", "Paired ratios within the supplemental cohort only")+'</h3>')
    parts.append(table([t("世界数","Worlds"),"GUI",t("指标","Metric"),"n",t("当前/历史","Current/baseline"),t("变化 %","Change %")],comparison_rows))
    parts.append('<p>'+t("本节图表与 CSV 使用 supplement_ 前缀，独立配对只使用同一补充组内的资产、世界数、GUI 状态和重复编号。", "This section’s figures and CSV use the supplement_ prefix. Pairing is restricted to matching world count, GUI state and repeat index within this supplemental cohort.")+'</p>')
    if supplement["warnings"]:
        parts.append('<details open><summary>'+t("补充数据完整性提示","Supplemental integrity notes")+'</summary><ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in supplement["warnings"])+'</ul></details>')
    parts.append('</section>')
    return ''.join(parts)


def report(language, output, session, assets, cases, trials, summary, comparisons, figures, warnings, downloads, input_folder, reliability_rows, supplement=None):
    zh = language == "zh"
    t = lambda a, b: a if zh else b
    title = t("SO-101 仿真性能对比", "SO-101 simulation performance benchmark")
    statuses = Counter(r["status"] for r in cases)
    count = len(cases)
    complete = statuses["completed"]
    available = sum(row["included"] for row in trials)
    body = [f'<header><div class="eyebrow">SO-101 · GAZEBO FORTRESS · 2026-10-02</div><h1>{title}</h1>',
        '<p class="lead">'+t("匹配工作负载的历史资产基线与当前资产对比；共同使用当前运行时，不代表历史软件栈复现。", "Matched historical-asset baseline versus current assets, using a common current runtime; this is not a reproduction of the historical software stack.")+'</p></header>',
        '<div class="cards">'+''.join(f'<div class="card"><strong>{number}</strong><span>{text}</span></div>' for number, text in [(count,t("计划试验","Planned trials")),(complete,t("完成试验","Completed trials")),(statuses["missing"],t("尚无结果","Missing results")),(available,t("有效相位测量","Valid phase measurements"))])+'</div>']
    if input_folder.name != "trials":
        body.append('<div class="notice">'+t("仅为 pilot／诊断数据预览，不属于正式 benchmark 结果。", "PILOT / DIAGNOSTIC PREVIEW ONLY — excluded from the formal benchmark.")+'</div>')
    if complete < count:
        body.append('<div class="notice">'+t("这是尚未完成或包含失败试验的测量报告；空白表示未获得有效测量，不表示零负载或零帧率。", "This report is incomplete or contains failed trials. Blank entries mean no valid measurement, not zero load or zero frame rate.")+'</div>')
    if not any(r["included"] for r in trials):
        body.append('<div class="notice">'+t("目前没有可纳入对比的有效试验，因此不能得出任何性能提升结论。", "There are currently no valid trials for comparison; no performance improvement can be inferred.")+'</div>')
    body.append(measured_findings(cases,supplement,language))
    body += ['<h2>'+t("测试范围与实验条件", "Scope and conditions")+'</h2>',
        '<p>'+t("测试采用 HOME 附近固定的小幅肩部转动和腕部转动，不生成 target，不运行 ACT 或 PPO；确定性步进包含 30 Hz 墙钟限速，自由运行设置 RTF=1，因此两者都不能当作无限速极限吞吐测试。", "The workload uses fixed small shoulder-pan and wrist-roll motions near HOME, creates no target, and runs no ACT or PPO. The stepped phase is paced at 30 wall-Hz and free run is configured for RTF=1; neither is an uncapped maximum-throughput test.")+'</p>',
        '<p>'+t("历史资产使用原始碰撞网格、高分辨率 60 Hz 相机和带阴影照明；当前资产使用简化碰撞体、较低相机分辨率/频率和无阴影照明，因此整体差异不能归因于某一个单独优化。", "Historical assets use original collision meshes, higher-resolution 60 Hz cameras and shadowed lighting; current assets use simplified collision geometry, reduced camera resolution/rates and shadowless lighting. The overall difference cannot be attributed to one optimization alone.")+'</p>',
        '<p>'+t("两组均配置相同接触记录插件与两个传感器，但历史组绑定完整夹爪、当前组绑定功能指面；此无目标小幅运动工作负载不能验证真实抓取碰撞求解收益或任务成功率。", "Both variants have the same contact-evidence plugin and two sensors, but the baseline observes whole jaws while current assets observe functional finger surfaces. This no-target small-motion workload does not establish grasp-contact solver savings or task success.")+'</p>']
    exclusions = session.get("benchmark_exclusions", {})
    excluded_ids = exclusions.get("excluded_cases", [])
    if excluded_ids:
        body.append('<p>'+t(f'最早 {len(excluded_ids)} 次 preliminary 试验与报告绘图验证并行，因额外计算负载而排除并在无并行分析时重测；这是按预先识别的运行干扰处理，不是按性能或成功/失败结果筛选。原数据保留在 excluded_preliminary，原因记录在 exclusions.json；运行失败试验没有因此删除，正式主矩阵仍按原计划每条件 3 次重复。', f'The earliest {len(excluded_ids)} preliminary trials overlapped report-plot validation and were excluded for concurrent compute interference, then repeated without concurrent analysis. This is an interference-based exclusion, not selection by performance or success/failure outcome. Originals remain in excluded_preliminary with the reason in exclusions.json; runtime-failure trials were not removed. The formal main matrix retains the original three repeats per condition.')+'</p>')
        body.append('<details><summary>'+t("被替换的 preliminary 试验", "Replaced preliminary trials")+'</summary><p>'+html.escape(", ".join(str(x) for x in excluded_ids))+'</p><pre>'+html.escape(json.dumps(exclusions,indent=2,ensure_ascii=False))+'</pre></details>')
    args = session["arguments"]
    metadata = [(t("历史资产提交", "Historical asset revision"), assets.get("historical_asset_revision", "—")),
                (t("当前资产提交", "Current asset revision"), assets.get("current_revision", session.get("head", "—"))),
                (t("正式数据目录", "Input trial directory"), str(input_folder)),
                (t("重复次数", "Planned repeats"), args.get("repeats")),
                (t("步进相位长度", "Stepped phase length"), str(args.get("steps"))+" steps"),
                (t("自由运行长度", "Free-run length"), str(args.get("free_seconds"))+" wall s"),
                (t("CPU 型号", "CPU model"), session.get("cpu_model", "—")),
                (t("CPU 物理核/逻辑核", "CPU physical/logical cores"), f'{session.get("cpu_physical_count", "—")} / {session.get("cpu_logical_count", "—")}'),
                (t("系统内存", "System RAM"), fmt(session.get("ram_bytes", 0)/2**30, 1)+" GiB"),
                (t("GPU、驱动、显存", "GPU, driver, VRAM"), session.get("gpu_details", "—")),
                ("Gazebo", session.get("gazebo_version", "—")),
                (t("渲染器预检", "Renderer preflight"), " · ".join(session.get("renderer_preflight",[]))),
                (t("每个 GUI 窗口尺寸", "Each GUI window size"), f'{assets.get("gui",{}).get("width","—")}×{assets.get("gui",{}).get("height","—")}')]
    body.append(table([t("项目", "Item"),t("值", "Value")], metadata))
    configuration_rows = []
    for variant, definition in assets.get("variants",{}).items():
        validation = definition.get("asset_validation",{})
        sensors = {s["name"]:s for s in validation.get("sensors",[])}
        row = [variant]
        for camera in ("top","gripper"):
            sensor = sensors.get(f"{camera}_camera_sensor",{})
            row.append(f'{sensor.get("width","—")}×{sensor.get("height","—")} @ {sensor.get("hz","—")} Hz')
        for camera in ("top","gripper"):
            shapes = sorted({r.get(f"{camera}_policy_shapes") for r in cases if r["variant"] == variant and r.get(f"{camera}_policy_shapes")})
            row.append(" / ".join(shapes) or "—")
        row.append(validation.get("scene_shadows","—"))
        row.append(assets.get("report_collision_complexity",{}).get(variant,{}).get("arm_collision_instance_triangles","—"))
        configuration_rows.append(row)
    body.append(table([t("资产","Assets"),t("原始 top 相机配置","Raw top camera configuration"),
        t("原始 gripper 相机配置","Raw gripper camera configuration"),
        t("top 输出 H×W×C","Top output H×W×C"),t("gripper 输出 H×W×C","Gripper output H×W×C"),
        t("场景阴影","Scene shadows"),t("机械臂碰撞实例三角形（不含相机）","Arm collision-instance triangles (excluding cameras)")],configuration_rows))
    proof = assets.get("original_arm_geometry_proof",{})
    if proof.get("all_mesh_bytes_identical_to_upstream_and_1eb21fc"):
        body.append('<p>'+t(f'基线的 {proof.get("unique_original_arm_meshes")} 个独立机械臂 STL 与上游原始文件及历史提交逐字节一致；碰撞 link、origin、scale 与顺序也经清单核验。碰撞三角形按实际实例累计，排除相机和相机支架；当前数量由清单所指 Xacro 与 STL 离线统计，并验证 SHA-256 未变化。相机输出尺寸来自各试验启动记录；480×640×3 表示 640×480 RGB 图像。',
            f'The baseline’s {proof.get("unique_original_arm_meshes")} distinct arm STLs are byte-identical to the upstream originals and historical revision; collision links, origins, scales and ordering were also verified in the manifest. Triangle counts sum actual collision instances, excluding cameras and their mount. Current counts are computed offline from the manifest-referenced Xacro and STLs after verifying their SHA-256 hashes. Camera output shapes come from trial startup records; 480×640×3 denotes a 640×480 RGB image.')+'</p>')
        body.append('<details><summary>'+t("原始网格一致性证据", "Original-mesh identity evidence")+'</summary><pre>'+html.escape(json.dumps(proof,indent=2,ensure_ascii=False))+'</pre></details>')
    body += ['<h2>'+t("完整试验矩阵", "Complete trial matrix")+'</h2>',
             '<p>'+t("保留所有计划重复，包括失败、未运行及清理失败；性能图表和对比均值只使用通过完整性校验的已完成试验，失败前的部分测量仍保留在 CSV 和下方可靠性表中。", "All planned repeats are retained, including failures, missing runs and cleanup failures. Performance plots and comparative means use completed, validated trials only; measured prefixes of failed trials remain in the CSV and reliability table below.")+'</p>']
    if any(high_load_failure(r) for r in cases):
        body.append('<div class="notice">'+t("历史资产在 3/4 world 的确定性步进条件下多次触发旧帧复用超限，标记为“高负载下无法完成”。原计划的三次重复测量完成后，不再对已反复失败的受控采样条件追加尝试；独立 free-run 是另一工作负载，其通过不覆盖原失败。直接证据是相机复用门槛，而不是已证实的 Gazebo 崩溃、CPU 100% 或 OOM。", "Historical assets repeatedly exceeded the stale-frame reuse limit during deterministic stepping with 3/4 worlds and are marked unable to complete under high load. After the three prespecified repetitions were completed, no additional attempts were made for repeatedly failing controlled-sampling conditions. Independent free run is a different workload; passing it does not replace the original failures. The direct evidence is the camera-reuse gate, not a proven Gazebo crash, CPU at 100%, or OOM.")+'</div>')
    grouped = defaultdict(list)
    for row in cases:
        grouped[(row["variant"],row["world_count"],row["gui"])].append(row)
    matrix = []
    for (variant, world, gui), rows in sorted(grouped.items()):
        matrix.append([variant, world, "on" if gui else "off", ' · '.join(f'R{r["repeat"]}: {completion_label(r,language)}' for r in sorted(rows,key=lambda r:r["repeat"]))])
    body.append(table([t("资产", "Assets"),t("世界数","Worlds"),"GUI",t("各次重复","Repeats")],matrix))
    failure_rows = [[r["case_id"], completion_label(r,language),r["error_type"], brief_error(r)] for r in cases if r["status"] not in ("completed","missing","starting")]
    if failure_rows:
        body.append(table([t("试验", "Trial"),t("完成判定","Completion assessment"),t("错误类型","Error type"),t("操作、domain 与错误摘要","Operation, domain and error summary")],failure_rows))
        body.append('<p>'+t("完整错误原文和 traceback 保留在可下载的 trial_status.csv 及各试验原始 result.json 中，可据此复核。", "Complete original errors and tracebacks remain in the downloadable trial_status.csv and each trial’s raw result.json for verification.")+'</p>')
    body.append('<h2>'+t("所有尝试的可靠性（包括失败前片段）", "Reliability across all attempts, including failed prefixes")+'</h2>')
    body.append('<p>'+t("这张表同时纳入成功和失败试验已返回的有效 step；复用率分母是具有已知复用标记的选中图像对数量，失败时被门槛拒绝的 step 没有合法选帧，不能计作新帧或未复用帧。失败 step 与 camera reuse gate 事件单独计数；部分性能仅供诊断，不进入上方完整试验均值。", "This table includes valid steps returned by both successful and failed attempts. Reuse is divided by selected pairs with a known reuse flag. A step rejected by the camera gate has no valid selected frame and is never counted as a new or non-reused frame. Failed steps and camera-reuse gate events are counted separately. Partial performance is diagnostic only and never enters complete-trial means.")+'</p>')
    reliability_groups = defaultdict(list)
    for row in reliability_rows:
        reliability_groups[(row["variant"],row["world_count"],row["gui"])].append(row)
    reliability_table = []
    for (variant,world,gui), entries in sorted(reliability_groups.items()):
        known = sum(r["selected_pair_reuse_known_count"] for r in entries)
        reused = sum(r["selected_pair_reuse_count"] for r in entries)
        failures = numbers(r["observed_failed_world_steps"] for r in entries)
        reliability_table.append([variant,world,"on" if gui else "off",
            f'{sum(r["trial_completed"] for r in entries)}/{len(entries)}',
            sum(r["trial_failed"] for r in entries),sum(r["trial_missing"] for r in entries),
            f'{reused}/{known}',fmt(100*reused/known if known else None,2),
            fmt(max(numbers(r["selected_pair_skew_max_ms"] for r in entries),default=None)),
            str(int(sum(failures))) if failures else "—",
            sum(r["camera_reuse_gate_event_count"] for r in entries)])
    body.append(table([t("资产","Assets"),t("世界数","Worlds"),"GUI",
        t("完成/计划","Complete/planned"),t("失败试验","Failed trials"),t("缺失试验","Missing trials"),
        t("复用/已知选帧","Reused/known pairs"),t("复用率 %","Reuse %"),
        t("最大时间差 ms","Maximum skew ms"),t("测量内失败步","Measured failed steps"),
        t("复用门槛事件","Reuse-gate events")],reliability_table))
    body.append('<p>'+t("失败前片段取自 failure_diagnostics 中明确的 measure_result，不重复加入 pending 响应；若失败相位没有父进程起止时间，各 world 的观测窗口仍可用于诊断，并可用所有 world 窗口的共同交集定义明确标注的资源诊断窗口；这不等于恢复了精确父进程窗口，没有成功回收的片段仍属未知。", "Failed prefixes are taken only from explicit measure_result entries in failure_diagnostics; pending replies are not appended again. Without parent timing bounds, worker windows still support diagnostics, and their common intersection can define an explicitly inferred resource window. This does not reconstruct the exact parent window; unrecovered prefixes remain unknown.")+'</p>')
    diagnostic_rows = [r for r in trials if r.get("measurement_source") == "failed_partial"]
    if diagnostic_rows:
        body.append('<h2>'+t("失败前性能诊断（部分时窗，不进入性能提升比较）", "Performance before failure — partial-window diagnostics only")+'</h2>')
        body.append('<div class="notice">'+t("下表保留高负载失败条件已测得的资源和相机信息；这些数值既不纳入完整试验均值，也不用于计算性能提升百分比。有效步数是所有 world 已返回的 step 总数，不含被相机门槛拒绝的动作。", "The table retains observed resource and camera information from failed high-load conditions. These values enter neither complete-trial means nor improvement percentages. Valid steps are the sum returned by all worlds and exclude camera-gate rejected actions.")+'</div>')
        body.append('<p>'+t("资源窗口取所有 worker 测量窗口的交集 [最晚开始，最早结束]，CSV 标记 resource_window_source=worker_intersection_inferred；只有全部预期 world 的有效时间边界齐全时才推定。CPU/GPU/PSS 在该交集内统计，RTF 和两路 unique wall FPS 仍分别使用各 world 自身测量窗口后取世界均值，不推算父进程吞吐。", "The resource window is the common worker intersection [latest start, earliest end], marked resource_window_source=worker_intersection_inferred in the CSV and defined only when all expected world bounds are available. CPU/GPU/PSS use that intersection. RTF and both unique wall-FPS values use each world's own interval, then average across worlds. Parent throughput is not inferred.")+'</p>')
        diagnostic_table = []
        for r in diagnostic_rows:
            coverage = [r.get(k+"_coverage_fraction") for k in ("cpu_process_percent","gpu_percent","process_pss_mib")]
            diagnostic_table.append([r["case_id"],r["phase"],
                str(int(r["observed_successful_world_steps"])) if finite(r.get("observed_successful_world_steps")) else "—",
                fmt(r.get("resource_window_wall_seconds"),2),
                r.get("resource_window_source","unavailable"),
                fmt(r.get("per_world_rtf_mean")),
                fmt(r.get("cpu_process_percent_mean"),1),fmt(r.get("gpu_percent_mean"),1),
                fmt(r.get("process_pss_mib_mean"),1),
                fmt(r.get("top_raw_wall_fps_mean"),2),fmt(r.get("gripper_raw_wall_fps_mean"),2),
                fmt(r.get("selected_pair_reuse_percent"),2),
                fmt(r.get("selected_pair_cache_arrival_gap_p95_ms"),2),
                "/".join(fmt(100*v if finite(v) else None,1) for v in coverage)])
        body.append(table([t("试验","Trial"),t("相位","Phase"),t("有效 world-step 总数","Valid world-steps"),
            t("资源窗口 s","Resource window s"),t("时窗来源","Window source"),t("世界 RTF 均值","Mean world RTF"),
            "CPU %","GPU %","PSS MiB",t("顶部 unique wall FPS","Top unique wall FPS"),
            t("手眼 unique wall FPS","Gripper unique wall FPS"),t("复用率 %","Reuse %"),
            t("缓存接收间隔 P95 ms","Cache-arrival gap P95 ms"),
            t("覆盖率 CPU/GPU/PSS %","Coverage CPU/GPU/PSS %")],diagnostic_table))
    body.append('<h2>'+t("主要指标数值", "Key numerical results")+'</h2>')
    summary_index = {(r["variant"],r["world_count"],r["gui"],r["phase"],r["metric"]):r for r in summary}
    result_columns = ["per_world_rtf_mean", "top_raw_wall_fps_mean", "gripper_raw_wall_fps_mean", "cpu_process_percent_mean", "process_pss_mib_mean", "gpu_percent_mean", "gpu_vram_mib_mean"]
    for phase in PHASES:
        keys = sorted({(r["variant"],r["world_count"],r["gui"],r["phase"]) for r in summary if r["phase"] == phase})
        if not keys:
            continue
        body.append('<h3>'+t("确定性步进" if phase == "stepped" else "自由运行", "Deterministic stepping" if phase == "stepped" else "Free run")+'</h3>')
        value_rows = []
        for key in keys:
            values = [key[0],key[1],"on" if key[2] else "off"]
            for metric in result_columns:
                entry = summary_index[(*key,metric)]
                value = fmt(entry["mean"],2)
                if finite(entry["sd"]):
                    value += " ± " + fmt(entry["sd"],2)
                if entry["n_metric"]:
                    value += f' (n={entry["n_metric"]})'
                values.append(value)
            value_rows.append(values)
        body.append(table([t("资产","Assets"),t("世界数","Worlds"),"GUI"]+[label(metric,language).replace("\n"," ") for metric in result_columns],value_rows))
    quality_keys = sorted({(r["variant"],r["world_count"],r["gui"],r["phase"]) for r in summary if r["phase"] == "stepped"})
    if quality_keys:
        body.append('<h3>'+t("相机源时间与缓存接收时间", "Camera source time versus cache-arrival time")+'</h3>')
        body.append('<p>'+t("源时间差（source skew）衡量选中帧的仿真源时间戳差，缓存接收间隔（cache arrival gap）是两帧 received_monotonic 时间戳的绝对差；源时间差为零不代表无延迟。落后步尾时间反映另一项时效性指标。本表仅完整有效试验：先逐次计算均值/P95，再报告重复间均值 ± 标准差；失败前数据见上方诊断表和 CSV。", "Source skew is the difference between selected frames’ simulation-source timestamps. Cache-arrival gap is the absolute difference between their received_monotonic timestamps. Zero source skew does not imply zero latency; lag relative to step end is a separate freshness measure. For complete valid trials, means/P95s are computed within each trial and then summarized as mean ± SD across repeats. Failed-prefix values appear in the diagnostic table and CSV.")+'</p>')
        quality_metrics = ["selected_pair_skew_mean_ms","selected_pair_skew_p95_ms",
            "selected_pair_cache_arrival_gap_mean_ms","selected_pair_cache_arrival_gap_p95_ms",
            "selected_pair_lag_p95_ms","selected_pair_reuse_percent"]
        quality_table = []
        for key in quality_keys:
            values = [key[0],key[1],"on" if key[2] else "off"]
            for metric in quality_metrics:
                result = summary_index[(*key,metric)]
                value = fmt(result["mean"],2)
                if finite(result["sd"]):
                    value += " ± " + fmt(result["sd"],2)
                if result["n_metric"]:
                    value += f' (n={result["n_metric"]})'
                values.append(value)
            quality_table.append(values)
        body.append(table([t("资产","Assets"),t("世界数","Worlds"),"GUI"]+
            [label(metric,language).replace("\n"," ") for metric in quality_metrics],quality_table))
    body += ['<h2>'+t("图表", "Figures")+'</h2>', '<p>'+t("每个小点是一整次试验，菱形为均值，误差棒为重复间样本标准差；只有一个有效重复时不画误差棒，未运行的条件不补零。", "Each dot is one full trial; diamonds are means and error bars are sample SD across repeats. A single valid repeat has no error bar; missing conditions are never replaced by zero.")+'</p>']
    plot_names = {"performance_stepped":t("确定性步进性能","Deterministic-step performance"),"performance_free_run":t("自由运行性能","Free-run performance"),"resources_stepped":t("步进相位资源使用","Resources during stepping"),"resources_free_run":t("自由运行资源使用","Resources during free run"),"camera_stepped":t("步进相位原始相机帧率","Raw camera rates during stepping"),"camera_free_run":t("自由运行原始相机帧率","Raw camera rates during free run"),"camera_pair_quality":t("被策略观测选中的相机对","Camera pairs selected for observations")}
    for name, path in figures:
        encoded = base64.b64encode(path.read_bytes()).decode()
        body.append('<figure><h3>'+plot_names[name]+'</h3><img alt="'+plot_names[name]+'" src="data:image/png;base64,'+encoded+'"></figure>')
    body += ['<h2>'+t("同条件重复配对比较", "Matched-condition repeat comparisons")+'</h2>',
        '<p>'+t("比值为同一世界数、GUI 状态和重复编号的“当前/历史”指标，表中是这些比值的均值；1 表示相同，CPU/内存/延迟更低通常更好，吞吐更高通常更好，FPS 的配置目标不同，不能直接解释为掉帧或改进幅度。", "Each ratio is current/baseline for the same world count, GUI state and repeat index; the table reports the mean of those ratios. One means equal. Lower CPU/memory/latency is usually preferable, while higher throughput is usually preferable. Camera configurations have different rate targets, so their FPS ratios do not directly imply frame drops or quality improvements.")+'</p>']
    compare_metrics = {"per_world_rtf_mean", "cycle_service_p95_ms", "cpu_process_percent_mean", "process_pss_mib_mean", "gpu_percent_mean", "gpu_vram_mib_mean"}
    comparison_rows = [[r["world_count"], "on" if r["gui"] else "off", r["phase"], METRICS[r["metric"]][0 if zh else 1], r["n_pairs"],fmt(r["latest_over_baseline_mean"]),fmt(r["mean_percent_change"],1)+("%" if finite(r["mean_percent_change"]) else "")] for r in comparisons if r["metric"] in compare_metrics and (r["phase"] == "stepped" or r["metric"] != "cycle_service_p95_ms")]
    body.append(table([t("世界数","Worlds"),"GUI",t("相位","Phase"),t("指标","Metric"),"n",t("当前/历史","Current/baseline"),t("变化","Change")],comparison_rows))
    body.append(supplement_section(supplement,language))
    body.append('<h2>'+t("统计口径与限制", "Definitions and limitations")+'</h2>')
    definitions = [
        t("相机 raw wall FPS = 该 world 实测窗口收到的不同源时间戳帧数 / 该 world 的墙钟窗口长度；raw sim FPS 使用同一窗口的仿真时长作为分母；回调总数、回调 FPS、重复时间戳比例及缺失时间戳数另存 CSV，不能把重复回调当成新帧。", "Raw wall FPS is the number of distinct received source-timestamp frames divided by that world's measured wall interval; raw sim FPS uses its simulated interval. Total callbacks, callback FPS, duplicate-timestamp percentage and missing-timestamp counts are separately retained in the CSV; repeated callbacks are not assumed to be new frames."),
        t("配置 60 Hz 或 45 Hz 不等于实测仿真 FPS：1 ms 物理步的传感器调度可能出现约 16 ms 或 22 ms 周期，因此不能直接以标称频率为基准计算丢帧率。", "Configured 60 Hz or 45 Hz is not the measured simulation-time FPS: sensor scheduling on 1 ms physics steps can yield approximately 16 ms or 22 ms periods. Nominal rates therefore cannot directly serve as a frame-drop denominator."),
        t("双相机选帧时间差只统计 coordinator 实际选中的图像对；复用率为复用 step 数 / 有复用标记的 step 数，自由运行没有策略选帧，故不提供该指标；完整试验性能图与所有尝试可靠性表使用不同纳入范围，后者保留失败前片段。", "Pair skew concerns the actual image pairs selected by the coordinator. Reuse is reused steps / steps with a known reuse flag. Free run has no observation selection. Complete-trial performance plots and all-attempt reliability tables have different inclusion rules: the latter retain failed prefixes."),
        t("缓存接收间隔是选中两帧到达缓存的墙钟时间差，不是源时间差，也不是完整端到端或网络延迟；源时间戳一致仍可能伴随不同到达时间和明显的步尾落后。", "Cache-arrival gap is the wall-clock separation of the selected frames’ cache-receive timestamps; it is neither source skew nor full end-to-end/network latency. Equal source timestamps can coexist with different arrival times and substantial step-end lag."),
        t("每世界 RTF 使用各 worker 的仿真时长/墙钟时长，总吞吐使用各 world 仿真时长或控制步数之和/父进程相位墙钟时长。", "Per-world RTF uses each worker's sim duration / wall duration; aggregate throughput sums simulated durations or control steps and divides by the parent's phase wall duration."),
        t("完整试验资源按父进程相位起止时间裁切，失败部分时窗诊断可用明确标注的 worker 窗口交集；CPU 按上一采样区间加权且剔除跨入时窗的首段，其他指标按当前采样值保持至下一采样点；不填充首采样前空窗，不让瞬时值跨越超过 3 秒缺口，各指标覆盖率保留在 CSV。", "Complete-trial resources are clipped to the parent's phase interval; failed-prefix diagnostics may use the explicitly marked worker-window intersection. CPU is weighted by the preceding sampler interval, excluding the first boundary-straddling interval; other values are held until the next sample. No value is filled before the first sample or held over a gap longer than 3 s. Per-metric coverage is retained in the CSV."),
        t("CPU 是受管进程树合计，一个逻辑核占满为 100%，多核合计可超过 100%；PSS 是同采样时刻进程比例驻留集之和，RSS 另存但会重复计算共享页；存在进程内存缺失的采样不进入内存统计，缺失计数另存。", "CPU is summed across managed process trees: one fully occupied logical core is 100%, so totals can exceed 100%. PSS sums proportional resident memory at each sample. RSS is retained separately but double-counts shared pages; memory samples with missing processes are excluded and missing counts are retained."),
        t("GPU 利用率与显存是整卡读数，包含桌面和其他后台负载，不能归因到单个 world；采样峰值也不是保证捕获的瞬时峰值。", "GPU utilization and VRAM are whole-device readings, including desktop and other background activity; they cannot be attributed to one world. Sampled peaks are not guaranteed instantaneous peaks."),
        t("三次重复属于描述性测量，step 数和 world 数不会增加独立重复样本量，不据此宣称统计显著；失败条件的幸存试验可能产生选择偏差。", "Three repeats support descriptive measurement. Steps and worlds do not increase the number of independent experimental repeats; no statistical-significance claim is made. Surviving trials in failure-prone conditions may be selected samples."),
        t("照明、碰撞体、相机同时变化，且当前工作负载没有方块接触，因此不能拆出各优化贡献，不能推断 ACT/PPO 训练速度或抓放成功率。", "Lighting, collision geometry and cameras change together, and the workload has no cube contact. Individual contributions, ACT/PPO training speed and task success cannot be inferred."),
    ]
    body.append('<ul>'+''.join('<li>'+d+'</li>' for d in definitions)+'</ul>')
    body.append('<h2>'+t("下载与复核", "Downloads and audit")+'</h2><p>')
    for name, content in downloads.items():
        body.append('<a class="download" download="'+name+'" href="data:text/csv;base64,'+base64.b64encode(content).decode()+'">'+name+'</a> ')
    body.append('</p><p>'+t("HTML 已内嵌图像和 CSV，可独立打开；同目录 figures 提供 PNG、SVG 和 PDF，原始试验 JSON/JSONL 保留在输入目录。", "Images and CSV downloads are embedded in this standalone HTML. The figures directory also contains PNG, SVG and PDF exports; original trial JSON/JSONL remain in the input directory.")+'</p>')
    if warnings:
        body.append('<details open><summary>'+t("数据完整性提示", "Data-integrity notes")+f' ({len(warnings)})</summary><ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in warnings)+'</ul></details>')
    body.append('<details><summary>'+t("完整会话配置", "Full session configuration")+'</summary><pre>'+html.escape(json.dumps(session,indent=2,ensure_ascii=False))+'</pre></details>')
    body.append('<footer>'+t("生成时间", "Generated")+' '+datetime.now(timezone.utc).isoformat()+'</footer>')
    css = """*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:#162739;font:15px/1.65 system-ui,'Noto Sans CJK SC',sans-serif}main{max-width:1440px;margin:auto;padding:44px 4vw 60px}header{padding:24px 0 20px}h1{font-size:38px;letter-spacing:-1px;margin:6px 0 14px}h2{margin-top:40px;border-bottom:1px solid #cbd5e1;padding-bottom:8px}h3{margin:0 0 12px}.eyebrow{font-size:12px;letter-spacing:2px;color:#087f8c;font-weight:700}.lead{font-size:19px;max-width:1050px;color:#43556b}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{padding:20px;background:white;border:1px solid #dbe3ea;border-radius:10px}.card strong{display:block;font-size:32px}.card span{color:#52677e}.notice{padding:16px 20px;background:#fff4db;border-left:4px solid #bd8006;margin:20px 0}figure{margin:24px 0;background:white;padding:20px;border:1px solid #dbe3ea;border-radius:10px}figure img{width:100%;height:auto}.table-wrap{overflow:auto;background:white;border:1px solid #dbe3ea;border-radius:8px;margin:18px 0}table{border-collapse:collapse;min-width:100%;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e5ebf0;vertical-align:top}th{background:#e7eef3;white-space:nowrap}td{max-width:900px;overflow-wrap:anywhere}li{margin-bottom:10px}pre{overflow:auto;background:#e7eef3;padding:18px;font-size:12px}details{margin:20px 0}summary{cursor:pointer;font-weight:600}.download{display:inline-block;background:#087f8c;color:white;padding:8px 12px;margin:4px;text-decoration:none;border-radius:5px}footer{color:#667085;margin-top:40px;font-size:12px}@media(max-width:700px){main{padding:20px}h1{font-size:28px}.cards{grid-template-columns:repeat(2,1fr)}figure{padding:8px}}"""
    document = '<!doctype html><html lang="'+('zh-CN' if zh else 'en')+'"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>'
    (output / f"report_{language}.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--result-set", default="trials", help="Default excludes pilot data; explicit alternate sets are diagnostic only.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--supplement-folder", type=Path,
        help="Independent free-run cohort; default: root/independent_free_run if present. An explicit relative path is relative to cwd.")
    parser.add_argument("--no-supplement", action="store_true", help="Disable automatic independent-free-run discovery.")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / args.result_set
    warnings = []
    session = safe_json(source / "session.json", warnings)
    if not isinstance(session, dict) or not isinstance(session.get("arguments"), dict):
        raise SystemExit("A valid result-set/session.json is required; no trial plan is inferred from directory names.")
    if args.result_set == "trials" and (root / "exclusions.json").exists():
        session = {**session, "benchmark_exclusions":safe_json(root / "exclusions.json",warnings,{})}
    output = (args.output_dir or root / "reports").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    assets = safe_json(root / "asset_manifest.json", warnings, {})
    assets["report_collision_complexity"] = asset_reproducibility(assets,warnings)
    cases, trials, worlds = analyze(source, session, warnings)
    summary, comparisons = aggregate(trials)
    reliability_rows = reliability(cases, trials)
    for collection in (cases,trials,worlds,summary,comparisons,reliability_rows):
        for row in collection:
            row["dataset"] = source.name
    downloads = {"trial_status.csv":csv_bytes(cases), "trial_metrics.csv":csv_bytes(trials),
                 "world_metrics.csv":csv_bytes(worlds), "summary.csv":csv_bytes(summary),
                 "paired_comparisons.csv":csv_bytes(comparisons),
                 "reliability_all_attempts.csv":csv_bytes(reliability_rows)}
    supplement = None
    supplement_folder = args.supplement_folder.expanduser().resolve() if args.supplement_folder else root / "independent_free_run"
    if args.supplement_folder and args.no_supplement:
        raise SystemExit("--supplement-folder and --no-supplement cannot be combined.")
    if not args.no_supplement and (args.supplement_folder is not None or supplement_folder.exists()):
        if supplement_folder.resolve() == source.resolve():
            raise SystemExit("Supplement must be a separate cohort, not the main input folder.")
        supplement_warnings = []
        supplement_session = safe_json(supplement_folder / "session.json", supplement_warnings)
        if isinstance(supplement_session, dict) and isinstance(supplement_session.get("arguments"), dict):
            if supplement_session["arguments"].get("steps") != 0:
                raise SystemExit("Independent free-run supplement must have session.arguments.steps=0.")
            sc, st, sw = analyze(supplement_folder,supplement_session,supplement_warnings)
            ss, sp = aggregate(st)
            sr = reliability(sc,st)
            for collection in (sc,st,sw,ss,sp,sr):
                for row in collection:
                    row["dataset"] = supplement_folder.name
            supplement = dict(source=supplement_folder,session=supplement_session,cases=sc,trials=st,
                              worlds=sw,summary=ss,comparisons=sp,reliability=sr,
                              warnings=supplement_warnings,figures={})
            for name, rows in (("trial_status",sc),("trial_metrics",st),("world_metrics",sw),
                               ("summary",ss),("paired_comparisons",sp),("reliability_all_attempts",sr)):
                downloads[f"supplement_{name}.csv"] = csv_bytes(rows)
        else:
            warnings.extend(supplement_warnings)
            warnings.append("Supplement was not included because its session manifest is absent or invalid.")
    for name, content in downloads.items():
        (output / name).write_bytes(content)
    for language in ("zh", "en"):
        figures = plot_all(trials, output, language)
        if supplement is not None:
            supplement["figures"][language] = plot_all(supplement["trials"],output,language,prefix="supplement_")
        report(language, output, session, assets, cases, trials, summary, comparisons, figures, warnings, downloads, source, reliability_rows,supplement)
    manifest = dict(schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    result_set=args.result_set, session=str(source / "session.json"),
                    planned_trials=len(cases), statuses=dict(Counter(r["status"] for r in cases)),
                    included_phase_measurements=sum(r["included"] for r in trials), warnings=warnings,
                    collision_complexity=assets.get("report_collision_complexity",{}))
    if session.get("benchmark_exclusions"):
        manifest["preliminary_exclusions"] = session["benchmark_exclusions"]
    if supplement is not None:
        manifest["supplement"] = dict(source=str(supplement["source"]),
            planned_trials=len(supplement["cases"]),
            statuses=dict(Counter(r["status"] for r in supplement["cases"])),
            included_phase_measurements=sum(r["included"] for r in supplement["trials"]),
            warnings=supplement["warnings"],pooled_with_main=False)
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(manifest, ensure_ascii=False))
    print(f"Reports: {output / 'report_zh.html'} | {output / 'report_en.html'}")


if __name__ == "__main__":
    main()
