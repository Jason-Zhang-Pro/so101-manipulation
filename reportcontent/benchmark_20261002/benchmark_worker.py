"""Isolated camera / deterministic-step benchmark worker; never launches Gazebo.

Entry point: ``worker_main(multiprocessing_connection, spec)``. The parent owns
the Gazebo launch and shutdown. This module imports ROS only inside the spawned
worker, after selecting its ROS domain and Gazebo transport partition.

Protocol (one request outstanding per worker):
  Startup -> ready, or error and worker exit.
  {op: step, targets: [six radians]} -> step_result
  {op: begin_measure, label?: str} -> measure_started
  {op: end_measure} -> measure_result (all scalar frame records; no images)
  {op: resume} -> resumed (free-running phase; no later stepped phase allowed)
  {op: pause} -> paused
  {op: status} -> status
  {op: close} -> closed and worker exit

An optional request_id is echoed in replies. An operation error returns error,
traceback, and the latest simulation clock. A failed step invalidates subsequent
step requests but permits measurement collection and orderly close. There are
no automatic retries, world resets, target creation/deletion, or threshold
changes. A measurement must be explicitly ended before another is started.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import sys
from threading import Lock
from time import monotonic, time
import traceback
from typing import Any


_STEP_INFO = (
    "physics_steps", "scheduled_sim_time_s", "schedule_timing_error_s",
    "step_start_sim_time_s", "step_end_sim_time_s", "expected_sim_delta_s",
    "actual_sim_delta_s", "sim_step_error_s", "top_camera_lag_from_step_end_s",
    "gripper_camera_lag_from_step_end_s", "camera_lag_from_step_end_s",
    "camera_pair_reused", "consecutive_camera_reuse_steps",
    "max_camera_reuse_steps", "max_camera_reuse_lag_s",
    "max_camera_lag_from_step_end_s", "max_camera_callback_grace_s",
    "top_image_sequence", "gripper_image_sequence", "top_image_source_timestamp_s",
    "gripper_image_source_timestamp_s", "camera_skew_s",
    "target_collection_sequence", "target_collection_source_sequence",
    "target_collection_source_sim_time_s", "target_collection_cube_count",
    "target_collection_updated",
)


def _clock_snapshot(runtime: Any) -> dict[str, Any] | None:
    if runtime is None:
        return None
    try:
        snapshot = runtime.io_node.simulation_clock_cache.latest()
        if snapshot is None:
            return None
        return {"sim_time_s": float(snapshot.sim_time_s),
                "sequence": int(snapshot.sequence),
                "received_monotonic_s": float(snapshot.received_monotonic_s),
                "age_wall_s": max(0.0, monotonic() - snapshot.received_monotonic_s)}
    except Exception as error:
        return {"snapshot_error": f"{type(error).__name__}: {error}"}


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        result = value.item()
        if result is None or isinstance(result, (str, bool, int, float)):
            return result
    raise TypeError(f"Expected scalar diagnostic; got {type(value).__name__}")


class _FrameRecorder:
    """Record metadata from existing cache updates, without extra image copies."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active = False
        self._start_wall = 0.0
        self._label = ""
        self._raw: dict[str, list[tuple[int, float | None, float]]] = {
            "top": [], "gripper": []}
        self._original_updates: list[tuple[Any, Any]] = []

    def install(self, camera: str, cache: Any) -> None:
        original = cache.update

        def observed_update(*args: Any, **kwargs: Any) -> Any:
            snapshot = original(*args, **kwargs)
            received = float(snapshot.received_monotonic_s)
            with self._lock:
                # An update may begin before begin_measure and finish after it.
                if self._active and received >= self._start_wall:
                    self._raw[camera].append((int(snapshot.sequence),
                        None if snapshot.source_timestamp_s is None
                        else float(snapshot.source_timestamp_s), received))
            return snapshot

        self._original_updates.append((cache, original))
        cache.update = observed_update

    def restore(self) -> None:
        with self._lock:
            self._active = False
        for cache, original in self._original_updates:
            cache.update = original
        self._original_updates.clear()

    def begin(self, label: str) -> float:
        with self._lock:
            if self._active:
                raise RuntimeError("A measurement is already active.")
            self._raw = {"top": [], "gripper": []}
            self._label = str(label)
            self._start_wall = monotonic()
            self._active = True
            return self._start_wall

    def end(self) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                raise RuntimeError("No active measurement.")
            end_wall = monotonic()
            self._active = False
            arrays = self._raw
            self._raw = {"top": [], "gripper": []}
            return {"label": self._label, "wall_start_monotonic_s": self._start_wall,
                    "wall_end_monotonic_s": end_wall,
                    "wall_duration_s": end_wall - self._start_wall,
                    "raw_frames": {
                        name: {"count": len(rows),
                               "sequences": [row[0] for row in rows],
                               "source_timestamps_s": [row[1] for row in rows],
                               "received_monotonic_s": [row[2] for row in rows]}
                        for name, rows in arrays.items()}}


def _validate_targets(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError("targets must be a list of six absolute joint positions in radians.")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("targets must contain only finite values.")
    return result


def _configure_process(spec: dict[str, Any]) -> Path:
    required = ("domain_id", "partition", "variant", "raw_width", "raw_height",
                "crop_xyxy", "output_dir", "project_root")
    missing = [name for name in required if name not in spec]
    if missing:
        raise ValueError(f"Missing benchmark worker fields: {missing}")
    domain = int(spec["domain_id"])
    if not 0 <= domain <= 232:
        raise ValueError("domain_id must be between 0 and 232.")
    if not str(spec["partition"]):
        raise ValueError("partition must not be empty.")
    root = Path(spec["project_root"]).resolve()
    if not (root / "src/so101_capstone").is_dir():
        raise ValueError(f"Current project source is unavailable: {root}")
    os.environ.update(ROS_DOMAIN_ID=str(domain), GZ_PARTITION=str(spec["partition"]),
                      IGN_PARTITION=str(spec["partition"]), PYTHONNOUSERSITE="1",
                      ROS_LOCALHOST_ONLY="1",
                      SO101_ENVIRONMENT_RANDOMIZATION_ENABLED="false",
                      SO101_IMAGE_AUGMENTATION_ENABLED="false")
    # Do not inject thread caps or pacing differences between variants.
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)
    return root


def worker_main(conn: Any, spec: dict[str, Any]) -> None:
    """Run inside multiprocessing.spawn; Gazebo is owned by the parent."""
    runtime = None
    recorder = _FrameRecorder()
    measurement: dict[str, Any] | None = None
    successful_steps = 0
    attempted_steps = 0
    failed_steps = 0
    episode_ended = False
    step_failed = False
    free_run_started = False
    world_running = False

    def reply(payload: dict[str, Any], request: dict[str, Any] | None = None) -> None:
        payload.setdefault("worker_pid", os.getpid())
        payload.setdefault("domain_id", spec.get("domain_id"))
        payload.setdefault("variant", spec.get("variant"))
        if request is not None and "request_id" in request:
            payload["request_id"] = request["request_id"]
        conn.send(payload)

    def report_error(error: BaseException, operation: str,
                     request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"type": "error", "operation": operation,
                   "error_type": type(error).__name__, "error_message": str(error),
                   "traceback": traceback.format_exc(), "clock": _clock_snapshot(runtime),
                   "successful_steps": successful_steps, "attempted_steps": attempted_steps,
                   "failed_steps": failed_steps, "episode_ended": episode_ended,
                   "step_failed": step_failed, "world_running": world_running}
        if measurement is not None:
            measurement["errors"].append(dict(payload))
        print(f"[benchmark worker {spec.get('domain_id')}] {operation} failed: "
              f"{type(error).__name__}: {error}", flush=True)
        reply(payload, request)
        return payload

    try:
        try:
            root = _configure_process(spec)
            from so101_capstone.environments.episode_reset import EpisodeResetConfig
            from so101_capstone.environments.reach_env_core import ReachEnvironmentConfig
            from so101_capstone.environments.simulation_runtime import (
                SimulationRuntimeConfig, create_ros_simulation_runtime,
            )
            from so101_capstone.environments.top_camera_processing import (
                default_simulation_top_camera_processing_config,
            )

            width, height = int(spec["raw_width"]), int(spec["raw_height"])
            crop = tuple(int(value) for value in spec["crop_xyxy"])
            if width <= 0 or height <= 0 or len(crop) != 4:
                raise ValueError("Invalid raw camera size / crop_xyxy.")
            top_config = replace(default_simulation_top_camera_processing_config(),
                                 input_width=width, input_height=height, fixed_crop_xyxy=crop)
            reset_config = EpisodeResetConfig()
            if "home_joint_positions_rad" in spec:
                reset_config = replace(reset_config, home_joint_positions=
                    _validate_targets(spec["home_joint_positions_rad"]))
            runtime = create_ros_simulation_runtime(
                runtime_config=SimulationRuntimeConfig(
                    startup_feedback_timeout_s=float(spec.get("startup_timeout_s", 90.0)),
                    resume_world_on_close=False),
                environment_config=ReachEnvironmentConfig(
                    max_episode_steps=int(spec.get("max_episode_steps", 1_000_000))),
                reset_config=reset_config, top_camera_config=top_config,
                environment_randomization_enabled=False, image_augmentation_enabled=False,
                reward_profile="legacy",
                ros_args=["--ros-args", "-p", f"top_image_width:={width}",
                          "-p", f"top_image_height:={height}"],
            )
            runtime.wait_until_ready()
            # Standard prepare_episode moves from spawn ZERO to fixed HOME.
            # The special ZERO evaluation path would instead hold ZERO, so it
            # is intentionally not used. Confirm the same real controller first.
            controller_ready = runtime._wait_for_evaluation_controller()
            prepared = runtime.prepare_episode()
            recorder.install("top", runtime.io_node.top_image_cache)
            recorder.install("gripper", runtime.io_node.gripper_image_cache)
            print(f"[benchmark worker {spec['domain_id']}] ready: {spec['variant']}, "
                  f"raw top={width}x{height}, fixed HOME", flush=True)
            reply({"type": "ready", "project_root": str(root),
                   "output_dir": str(Path(spec["output_dir"]).resolve()),
                   "partition": spec["partition"], "clock": _clock_snapshot(runtime),
                   "home_targets_rad": list(reset_config.home_joint_positions),
                   "measured_joint_positions_rad": prepared.observation.joint_positions.tolist(),
                   "top_policy_shape": list(prepared.observation.top_image.shape),
                   "gripper_policy_shape": list(prepared.observation.gripper_image.shape),
                   "controller_readiness": controller_ready,
                   "reset": asdict(prepared.home_reset),
                   "frame0_info": prepared.info,
                   "top_camera_processing_config": asdict(top_config),
                   "coordinator_config": asdict(runtime.coordinator.config),
                   "policy_control_hz": runtime.coordinator.schedule.control_hz,
                   "physics_step_s": runtime.coordinator.schedule.physics_step_s,
                   "reward_profile": "legacy", "target_policy": "no targets created",
                   "sampling_model": "fixed absolute joint targets; no ACT or PPO"})
        except BaseException as error:
            report_error(error, "startup")
            return

        while True:
            try:
                request = conn.recv()
            except EOFError:
                break
            operation = request.get("op") if isinstance(request, dict) else None
            try:
                if operation == "close":
                    if measurement is not None:
                        raise RuntimeError("End the active measurement before closing the worker.")
                    recorder.restore()
                    if world_running:
                        runtime.world_controller.pause()
                        world_running = False
                    runtime.close()
                    runtime = None
                    reply({"type": "closed"}, request)
                    return
                if operation == "begin_measure":
                    if measurement is not None:
                        raise RuntimeError("A measurement is already active.")
                    start_clock = _clock_snapshot(runtime)
                    wall_start = recorder.begin(str(request.get("label", "measurement")))
                    measurement = {"clock_start": start_clock,
                                   "wall_start_unix_s": time(),
                                   "successful_steps_start": successful_steps,
                                   "attempted_steps_start": attempted_steps,
                                   "failed_steps_start": failed_steps,
                                   "world_running_at_start": world_running,
                                   "steps": [], "errors": []}
                    reply({"type": "measure_started", "label": str(request.get("label", "measurement")),
                           "clock": start_clock, "wall_start_monotonic_s": wall_start,
                           "world_running": world_running}, request)
                elif operation == "end_measure":
                    if measurement is None:
                        raise RuntimeError("No active measurement.")
                    clock_end = _clock_snapshot(runtime)
                    raw = recorder.end()
                    details = measurement
                    measurement = None
                    payload = {"type": "measure_result", **raw, **details,
                               "clock_end": clock_end, "world_running_at_end": world_running,
                               "successful_steps": successful_steps - details["successful_steps_start"],
                               "attempted_steps": attempted_steps - details["attempted_steps_start"],
                               "failed_steps": failed_steps - details["failed_steps_start"],
                               "episode_ended": episode_ended, "step_failed": step_failed}
                    start_sim = (details["clock_start"] or {}).get("sim_time_s")
                    end_sim = (clock_end or {}).get("sim_time_s")
                    payload["sim_duration_s"] = (None if start_sim is None or end_sim is None
                                                 else end_sim - start_sim)
                    reply(payload, request)
                elif operation == "step":
                    if episode_ended:
                        raise RuntimeError("Episode terminated/truncated; stepping cannot continue.")
                    if step_failed:
                        raise RuntimeError("An earlier step failed; a new world is required.")
                    if free_run_started:
                        raise RuntimeError("Cannot resume deterministic stepping after free-run.")
                    targets = _validate_targets(request.get("targets"))
                    attempted_steps += 1
                    started = monotonic()
                    try:
                        result = runtime.step(targets)
                    except BaseException:
                        failed_steps += 1
                        step_failed = True
                        raise
                    ended = monotonic()
                    successful_steps += 1
                    episode_ended = bool(result.terminated or result.truncated)
                    info = {key: _scalar(result.info[key]) for key in _STEP_INFO if key in result.info}
                    info.update(top_image_received_monotonic_s=float(result.camera_frames.top.received_monotonic_s),
                                gripper_image_received_monotonic_s=float(result.camera_frames.gripper.received_monotonic_s),
                                camera_pair_used_source_timestamps=bool(result.camera_frames.used_source_timestamps))
                    payload = {"type": "step_result", "step_index": successful_steps,
                               "wall_start_monotonic_s": started, "wall_end_monotonic_s": ended,
                               "step_wall_duration_s": ended - started,
                               "reward": float(result.reward), "terminated": bool(result.terminated),
                               "truncated": bool(result.truncated), "info": info}
                    if measurement is not None:
                        measurement["steps"].append(dict(payload))
                    reply(payload, request)
                elif operation == "resume":
                    runtime.world_controller.resume()
                    free_run_started = True
                    world_running = True
                    reply({"type": "resumed", "clock": _clock_snapshot(runtime),
                           "wall_monotonic_s": monotonic()}, request)
                elif operation == "pause":
                    runtime.world_controller.pause()
                    world_running = False
                    reply({"type": "paused", "clock": _clock_snapshot(runtime),
                           "wall_monotonic_s": monotonic()}, request)
                elif operation == "status":
                    reply({"type": "status", "clock": _clock_snapshot(runtime),
                           "successful_steps": successful_steps, "attempted_steps": attempted_steps,
                           "failed_steps": failed_steps, "episode_ended": episode_ended,
                           "step_failed": step_failed, "free_run_started": free_run_started,
                           "world_running": world_running, "measurement_active": measurement is not None}, request)
                else:
                    raise ValueError(f"Unknown worker operation: {operation!r}")
            except BaseException as error:
                report_error(error, str(operation), request if isinstance(request, dict) else None)
    finally:
        recorder.restore()
        if runtime is not None:
            try:
                # The parent remains responsible for terminating Gazebo. If
                # the protocol ended while free-running, leave it paused.
                if world_running:
                    runtime.world_controller.pause()
                runtime.close()
            except BaseException as error:
                print(f"[benchmark worker {spec.get('domain_id')}] cleanup failed: "
                      f"{type(error).__name__}: {error}\n{traceback.format_exc()}", flush=True)
                try:
                    reply({"type": "error", "operation": "cleanup",
                           "error_type": type(error).__name__, "error_message": str(error),
                           "traceback": traceback.format_exc(), "clock": _clock_snapshot(runtime)})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        conn.close()
