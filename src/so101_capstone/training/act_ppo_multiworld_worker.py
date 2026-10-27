"""Process-isolated Gazebo workers for multi-world Direct ACT PPO.

Each worker owns exactly one ROS 2 domain, one Gazebo transport partition,
one ros2 launch process, and one SO-101 simulation runtime. Policy inference
and PPO optimization stay in the parent process. Two RGB observations are
published through shared memory so full camera frames are not pickled through
the control pipe on every policy step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import numpy as np

from so101_capstone.environments.policy_observation import PolicyObservation

from .act_ppo_gazebo_crash import read_gazebo_crash_diagnostics


_POLICY_IMAGE_SHAPE = (480, 640, 3)
_POLICY_IMAGE_DTYPE = np.uint8
_RUNTIME_SHUTDOWN_TIMEOUT_S = 5.0
_LAUNCH_KILL_TIMEOUT_S = 2.0
_WORLD_CLEANUP_MAX_SCANS = 8
_WORLD_CLEANUP_SCAN_PAUSE_S = 0.25
_WORLD_STOP_RESPONSE_MARGIN_S = 10.0


def _environment_randomization_enabled(*, include_images: bool = True) -> bool:
    """Whether preparation needs a deterministic episode randomization key."""
    enabled = []
    names = ("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "SO101_IMAGE_AUGMENTATION_ENABLED")
    for name in (names if include_images else names[:1]):
        value = os.environ.get(name, "false").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            enabled.append(True)
        elif value in {"0", "false", "no", "off", ""}:
            enabled.append(False)
        else:
            raise ValueError(f"{name} must be a boolean value.")
    return any(enabled)


_POLICY_IMAGE_BYTES = int(np.prod(_POLICY_IMAGE_SHAPE))


class GazeboWorkerCommunicationError(RuntimeError):
    """Raised when a worker process stops responding."""

    def __init__(
        self,
        message: str,
        *,
        gazebo_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.gazebo_diagnostics = gazebo_diagnostics


@dataclass(frozen=True)
class GazeboWorkerConfig:
    world_index: int
    ros_domain_id: int
    gz_partition: str
    gazebo_gui: bool
    project_root: Path
    output_directory: Path
    startup_feedback_timeout_s: float
    startup_max_attempts: int = 3
    restart_max_attempts: int = 3
    retry_delay_s: float = 1.0
    launch_shutdown_timeout_s: float = 10.0
    max_episode_steps: int = 600
    reward_profile: str = "legacy"
    reward_config: dict[str, Any] | None = None
    n_action_steps: int = 1
    evaluation_mode: bool = False
    observe_after_success_s: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "world_index",
            "ros_domain_id",
            "startup_max_attempts",
            "restart_max_attempts",
            "max_episode_steps",
            "n_action_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.world_index < 0:
            raise ValueError("world_index must be non-negative.")
        if not 0 <= self.ros_domain_id <= 232:
            raise ValueError("ros_domain_id must be within [0, 232].")
        if self.startup_max_attempts < 1:
            raise ValueError("startup_max_attempts must be positive.")
        if self.restart_max_attempts < 1:
            raise ValueError("restart_max_attempts must be positive.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
        if self.n_action_steps < 1 or self.reward_profile not in {"legacy", "pick_place_v2"}:
            raise ValueError("Invalid reward profile or ACT execution length.")
        if self.reward_profile == "pick_place_v2" and self.reward_config is None:
            raise ValueError("Workers require the complete resolved pick-place configuration.")
        if not isinstance(self.evaluation_mode, bool):
            raise ValueError("evaluation_mode must be a bool.")
        if self.evaluation_mode and self.reward_profile != "pick_place_v2":
            raise ValueError("Evaluation workers require pick_place_v2.")
        if (isinstance(self.observe_after_success_s, bool)
                or not math.isfinite(float(self.observe_after_success_s))
                or float(self.observe_after_success_s) < 0):
            raise ValueError("observe_after_success_s must be finite and non-negative.")
        if not self.evaluation_mode and self.observe_after_success_s != 0:
            raise ValueError("Success monitoring requires an evaluation worker.")
        if not isinstance(self.gazebo_gui, bool):
            raise ValueError("gazebo_gui must be a bool.")
        if not str(self.gz_partition).strip():
            raise ValueError("gz_partition must not be empty.")
        if float(self.startup_feedback_timeout_s) <= 0.0:
            raise ValueError("startup_feedback_timeout_s must be positive.")
        if float(self.retry_delay_s) < 0.0:
            raise ValueError("retry_delay_s must be non-negative.")
        if float(self.launch_shutdown_timeout_s) <= 0.0:
            raise ValueError("launch_shutdown_timeout_s must be positive.")
        object.__setattr__(self, "project_root", Path(self.project_root).expanduser().resolve())
        object.__setattr__(
            self,
            "output_directory",
            Path(self.output_directory).expanduser().resolve(),
        )

    @property
    def world_label(self) -> str:
        return f"world_{self.world_index:02d}"

    @property
    def log_path(self) -> Path:
        return self.output_directory / "worlds" / self.world_label / "gazebo.log"

    def payload(self) -> dict[str, Any]:
        return {
            "world_index": self.world_index,
            "ros_domain_id": self.ros_domain_id,
            "gz_partition": self.gz_partition,
            "gazebo_gui": self.gazebo_gui,
            "project_root": str(self.project_root),
            "output_directory": str(self.output_directory),
            "startup_feedback_timeout_s": self.startup_feedback_timeout_s,
            "startup_max_attempts": self.startup_max_attempts,
            "restart_max_attempts": self.restart_max_attempts,
            "retry_delay_s": self.retry_delay_s,
            "launch_shutdown_timeout_s": self.launch_shutdown_timeout_s,
            "max_episode_steps": self.max_episode_steps,
            "reward_profile": self.reward_profile,
            "reward_config": self.reward_config,
            "n_action_steps": self.n_action_steps,
            "evaluation_mode": self.evaluation_mode,
            "observe_after_success_s": self.observe_after_success_s,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GazeboWorkerConfig":
        return cls(**payload)


def gazebo_launch_command(config: GazeboWorkerConfig) -> list[str]:
    randomization = (
        "true" if _environment_randomization_enabled(include_images=False) else "false"
    )
    return [
        "ros2",
        "launch",
        "so101_bringup",
        "sim.launch.py",
        f"gazebo_gui:={'true' if config.gazebo_gui else 'false'}",
        f"environment_randomization:={randomization}",
    ]


def _response_from_service(response: Any) -> dict[str, Any]:
    return {
        "success": bool(getattr(response, "success", True)),
        "message": str(getattr(response, "message", "")),
    }


def _runtime_error_types() -> tuple[type[BaseException], ...]:
    from so101_capstone.environments.episode_reset import EpisodeResetError
    from so101_capstone.environments.gazebo_world_control import (
        GazeboWorldControlError,
    )
    from so101_capstone.environments.reach_env_core import (
        ReachEnvironmentStateError,
    )
    from so101_capstone.environments.simulation_runtime import (
        SimulationRuntimeError,
    )
    from so101_capstone.environments.simulation_step_coordinator import (
        SimulationStepCoordinatorError,
    )
    from so101_capstone.environments.target_cube_control import (
        TargetCubeControlError,
    )

    return (
        EpisodeResetError,
        GazeboWorldControlError,
        ReachEnvironmentStateError,
        SimulationRuntimeError,
        SimulationStepCoordinatorError,
        TargetCubeControlError,
    )


def _error_payload(error: BaseException, *, operation: str) -> dict[str, Any]:
    payload = {
        "type": "operation_error",
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    if operation == "start_training_stage":
        payload["failure_category"] = "stage_initialization"
        current: BaseException | None = error
        while current is not None:
            report = getattr(current, "stage_initialization", None)
            if isinstance(report, dict):
                payload["stage_initialization"] = dict(report)
                break
            current = current.__cause__
    return payload


def _world_stop_timeout_s(config: GazeboWorkerConfig) -> float:
    """Allow all bounded host cleanup waits before treating it as unresponsive."""

    cleanup_timeout_s = min(5.0, float(config.launch_shutdown_timeout_s))
    # Each scoped cleanup can wait for TERM, then KILL, then a 0.25 s
    # settling window. The host also waits between successive scans.
    cleanup_scan_timeout_s = (
        cleanup_timeout_s
        + min(1.0, cleanup_timeout_s)
        + 0.25
        + _WORLD_CLEANUP_SCAN_PAUSE_S
    )
    return (
        2.0 * _RUNTIME_SHUTDOWN_TIMEOUT_S
        + float(config.launch_shutdown_timeout_s)
        + _LAUNCH_KILL_TIMEOUT_S
        + _WORLD_CLEANUP_MAX_SCANS * cleanup_scan_timeout_s
        + _WORLD_STOP_RESPONSE_MARGIN_S
    )


def _cleanup_reserved_world_resources(
    config: GazeboWorkerConfig,
    *,
    cleanup_fn: Any | None = None,
    sleep_fn: Any = time.sleep,
    quiet_scans_required: int = 2,
    max_scans: int = _WORLD_CLEANUP_MAX_SCANS,
) -> dict[str, int]:
    """Clear stale managed processes for exactly one worker resource set."""

    if cleanup_fn is None:
        from .act_ppo_multiworld_cleanup import (
            cleanup_stale_multiworld_processes,
        )

        cleanup_fn = cleanup_stale_multiworld_processes

    required = int(quiet_scans_required)
    scans_limit = int(max_scans)

    if required < 1:
        raise ValueError("quiet_scans_required must be positive.")
    if scans_limit < required:
        raise ValueError(
            "max_scans must be at least quiet_scans_required."
        )

    quiet_scans = 0
    total_matched = 0
    total_terminated = 0
    total_killed = 0

    cleanup_timeout_s = min(
        5.0,
        float(config.launch_shutdown_timeout_s),
    )

    for scan_index in range(1, scans_limit + 1):
        report = cleanup_fn(
            target_ros_domain_ids=(config.ros_domain_id,),
            target_gz_partitions=(config.gz_partition,),
            timeout_s=cleanup_timeout_s,
            current_pid=os.getpid(),
        )

        matched = len(report.matched_processes)
        total_matched += matched
        total_terminated += len(report.terminated_processes)
        total_killed += len(report.killed_processes)

        if matched == 0:
            quiet_scans += 1
            if quiet_scans >= required:
                return {
                    "scan_count": scan_index,
                    "matched": total_matched,
                    "terminated": total_terminated,
                    "killed": total_killed,
                }
        else:
            quiet_scans = 0

        # Leave a short observation window so late-exiting / detached
        # descendants can appear before the next empty-state confirmation.
        sleep_fn(_WORLD_CLEANUP_SCAN_PAUSE_S)

    raise RuntimeError(
        "World resource cleanup did not reach a stable empty state: "
        f"world={config.world_index}, "
        f"ROS_DOMAIN_ID={config.ros_domain_id}, "
        f"partition={config.gz_partition}, "
        f"scans={scans_limit}, "
        f"matched={total_matched}, "
        f"terminated={total_terminated}, "
        f"killed={total_killed}."
    )


class _WorkerRuntimeHost:
    def __init__(
        self,
        config: GazeboWorkerConfig,
        *,
        top_buffer: Any,
        gripper_buffer: Any,
    ) -> None:
        self.config = config
        self.runtime: Any | None = None
        # A world is consumed as soon as episode preparation starts. The
        # parent must persist its episode records before requesting a restart;
        # reusing this world would require removing live target bodies.
        self._fresh_world_available = False
        self.launch_process: subprocess.Popen[str] | None = None
        self.log_stream: Any | None = None
        self.launch_log_offset: int | None = None
        self.last_cleanup_summary: dict[str, int] = {
            "scan_count": 0,
            "matched": 0,
            "terminated": 0,
            "killed": 0,
        }
        self.top_array = np.frombuffer(
            top_buffer, dtype=_POLICY_IMAGE_DTYPE
        ).reshape(_POLICY_IMAGE_SHAPE)
        self.gripper_array = np.frombuffer(
            gripper_buffer, dtype=_POLICY_IMAGE_DTYPE
        ).reshape(_POLICY_IMAGE_SHAPE)

    def _write_log_header(self, text: str) -> None:
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.log_path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + "=" * 80 + "\n")
            stream.write(text + "\n")
            stream.write("=" * 80 + "\n")

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(self.config.ros_domain_id)
        env["GZ_PARTITION"] = self.config.gz_partition
        env["IGN_PARTITION"] = self.config.gz_partition
        return env

    def gazebo_diagnostics(self) -> dict[str, Any]:
        return read_gazebo_crash_diagnostics(
            self.config.log_path,
            launch_log_offset=self.launch_log_offset,
        )

    def _terminate_launch_process(self) -> None:
        process = self.launch_process
        self.launch_process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.config.launch_shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=_LAUNCH_KILL_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    pass

    def stop_world(self) -> None:
        self._fresh_world_available = False
        runtime = self.runtime
        self.runtime = None
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as error:
                self._write_log_header(
                    "Runtime close error: "
                    f"{type(error).__name__}: {error}"
                )

        # Fast path: stop the ros2 launch process group.
        self._terminate_launch_process()

        if self.log_stream is not None:
            try:
                self.log_stream.close()
            except BaseException:
                pass
            self.log_stream = None

        # Strong path: launch-process exit alone is not proof that all
        # Gazebo / ROS descendants are gone. Clear exactly this worker's
        # ROS domain / Gazebo partition, then require two consecutive
        # scans with no managed process before any replacement may start.
        self.last_cleanup_summary = (
            _cleanup_reserved_world_resources(self.config)
        )

        summary = self.last_cleanup_summary
        if summary["matched"] > 0 or summary["killed"] > 0:
            message = (
                "[WORLD CLEANUP] "
                f"world={self.config.world_index} "
                f"ROS_DOMAIN_ID={self.config.ros_domain_id} "
                f"partition={self.config.gz_partition} "
                f"scans={summary['scan_count']} "
                f"matched={summary['matched']} "
                f"terminated={summary['terminated']} "
                f"killed={summary['killed']}"
            )
            print(message, flush=True)
            self._write_log_header(message)

    def _launch_once(self, *, reason: str, attempt: int) -> None:
        from so101_capstone.environments.reach_env_core import (
            ReachEnvironmentConfig,
        )
        from so101_capstone.environments.simulation_runtime import (
            SimulationRuntimeConfig,
            create_ros_simulation_runtime,
        )

        if self.config.evaluation_mode and _environment_randomization_enabled():
            raise ValueError("Evaluation workers require environment and image randomization disabled.")
        self._fresh_world_available = False
        self.stop_world()
        self.launch_log_offset = (
            self.config.log_path.stat().st_size
            if self.config.log_path.exists()
            else 0
        )
        self._write_log_header(
            f"{reason}: launch attempt {attempt}; "
            f"ROS_DOMAIN_ID={self.config.ros_domain_id}; "
            f"partition={self.config.gz_partition}; "
            f"gui={self.config.gazebo_gui}"
        )
        self.log_stream = self.config.log_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        command = gazebo_launch_command(self.config)
        self.launch_process = subprocess.Popen(
            command,
            cwd=self.config.project_root,
            env=self._environment(),
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        runtime_config = SimulationRuntimeConfig(
            startup_feedback_timeout_s=self.config.startup_feedback_timeout_s,
            shutdown_timeout_s=_RUNTIME_SHUTDOWN_TIMEOUT_S,
            resume_world_on_close=False,
        )
        runtime_kwargs: dict[str, Any] = {
            "runtime_config": runtime_config,
            "environment_config": ReachEnvironmentConfig(
                max_episode_steps=self.config.max_episode_steps,
            ),
        }
        if _environment_randomization_enabled():
            runtime_kwargs["environment_randomization_world_index"] = (
                self.config.world_index
            )
        if self.config.reward_profile == "pick_place_v2":
            from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
            runtime_kwargs.update(
                reward_profile=self.config.reward_profile,
                reward_config=PickPlaceRewardConfig.from_dict(self.config.reward_config),
            )
        if self.config.evaluation_mode:
            runtime_kwargs["observe_after_success_s"] = self.config.observe_after_success_s
        self.runtime = create_ros_simulation_runtime(**runtime_kwargs)
        self.runtime.wait_until_ready(
            timeout_s=self.config.startup_feedback_timeout_s,
        )
        if self.launch_process.poll() is not None:
            raise RuntimeError(
                "ros2 launch exited before the Gazebo worker became ready."
            )
        self._fresh_world_available = True

    def start_world(
        self,
        *,
        max_attempts: int,
        reason: str,
    ) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            try:
                self._launch_once(reason=reason, attempt=attempt)
                assert self.launch_process is not None
                return {
                    "type": "ready",
                    "world_index": self.config.world_index,
                    "attempts_used": attempt,
                    "ros_domain_id": self.config.ros_domain_id,
                    "gz_partition": self.config.gz_partition,
                    "gazebo_gui": self.config.gazebo_gui,
                    "launch_pid": self.launch_process.pid,
                    "log_path": str(self.config.log_path),
                    "launch_log_offset": self.launch_log_offset,
                }
            except BaseException as error:
                failures.append(
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "gazebo_diagnostics": self.gazebo_diagnostics(),
                    }
                )
                self._write_log_header(
                    f"{reason}: attempt {attempt} failed: "
                    f"{type(error).__name__}: {error}"
                )
                self.stop_world()
                if attempt < max_attempts and self.config.retry_delay_s > 0.0:
                    time.sleep(self.config.retry_delay_s)
        return {
            "type": "startup_failed" if reason == "initial_startup" else "restart_failed",
            "world_index": self.config.world_index,
            "attempts_used": max_attempts,
            "failures": failures,
            "log_path": str(self.config.log_path),
            "launch_log_offset": self.launch_log_offset,
            "gazebo_diagnostics": (
                failures[-1]["gazebo_diagnostics"] if failures else self.gazebo_diagnostics()
            ),
            "launch_pid": None,
        }

    def _require_runtime(self) -> Any:
        if self.runtime is None:
            raise RuntimeError("Gazebo worker runtime is not available.")
        return self.runtime

    def publish_observation(self, observation: PolicyObservation) -> dict[str, Any]:
        top = np.asarray(observation.top_image)
        gripper = np.asarray(observation.gripper_image)
        if top.shape != _POLICY_IMAGE_SHAPE or top.dtype != _POLICY_IMAGE_DTYPE:
            raise RuntimeError(
                "Worker top observation does not match the policy image contract."
            )
        if (
            gripper.shape != _POLICY_IMAGE_SHAPE
            or gripper.dtype != _POLICY_IMAGE_DTYPE
        ):
            raise RuntimeError(
                "Worker gripper observation does not match the policy image contract."
            )
        np.copyto(self.top_array, top)
        np.copyto(self.gripper_array, gripper)
        return {
            "joint_positions": tuple(
                float(value) for value in observation.joint_positions
            ),
            "joint_velocities": tuple(
                float(value) for value in observation.joint_velocities
            ),
        }

    def prepare_episode(
        self,
        *,
        episode_attempt_index: int | None = None,
        task_mode: str = "pick_place",
        scene_seed: int | None = None,
    ) -> dict[str, Any]:
        runtime = self._require_runtime()
        if not self._fresh_world_available:
            raise RuntimeError(
                "Episode preparation requires a freshly started Gazebo world. "
                "Persist the previous episode records and restart the world first."
            )
        # Consume before any scene/reset work: a partially prepared scene is
        # also unsafe to reuse, even if preparation raises an exception.
        self._fresh_world_available = False
        if self.config.evaluation_mode and task_mode != "pick_place":
            raise ValueError("Evaluation workers require task_mode='pick_place'.")
        if self.config.reward_profile == "pick_place_v2":
            runtime.configure_task(task_mode, seed=scene_seed, n_action_steps=self.config.n_action_steps)
        if self.config.evaluation_mode:
            if _environment_randomization_enabled():
                raise ValueError("Evaluation workers require environment and image randomization disabled.")
            prepared = runtime.prepare_episode(evaluation_initial_pose="zero")
        elif _environment_randomization_enabled():
            if (
                isinstance(episode_attempt_index, bool)
                or not isinstance(episode_attempt_index, int)
                or episode_attempt_index < 0
            ):
                raise ValueError(
                    "episode_attempt_index must be a non-negative integer "
                    "when environment randomization is enabled."
                )
            prepared = runtime.prepare_episode(
                environment_randomization_episode_key=episode_attempt_index,
            )
        else:
            prepared = runtime.prepare_episode()
        return {
            "type": "episode_prepared",
            "observation": self.publish_observation(prepared.observation),
            "info": self._response_info(prepared.info),
            "cleanup": {
                "success": True,
                "message": "A fresh Gazebo world has no previous episode targets.",
                "requested": False,
                "strategy": "fresh_world",
            },
            ("initial_pose_reset" if self.config.evaluation_mode else "home_reset"): {
                "control_steps": int(prepared.home_reset.control_steps),
                "physics_steps": int(prepared.home_reset.physics_steps),
                "stable_sample_count": int(prepared.home_reset.stable_sample_count),
                "max_position_error_rad": float(
                    prepared.home_reset.max_position_error_rad
                ),
                "max_velocity_rad_s": float(
                    prepared.home_reset.max_velocity_rad_s
                ),
            },
        }

    def _response_info(self, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        if self.config.evaluation_mode:
            result["evaluation_measurements"] = self._require_runtime().evaluation_measurements()
        return result

    def start_training_stage(
        self, *, training_stage: str, stage_config: dict[str, Any],
    ) -> dict[str, Any]:
        if self.config.evaluation_mode:
            raise ValueError("Evaluation workers cannot initialize HOME training stages.")
        prepared = self._require_runtime().start_training_stage(
            training_stage=training_stage, stage_config=stage_config,
        )
        return {
            "type": "stage_started",
            "observation": self.publish_observation(prepared.observation),
            "info": dict(prepared.info),
            "initialization": dict(prepared.info["stage_initialization"]),
        }

    def step(self, joint_targets: Any) -> dict[str, Any]:
        runtime = self._require_runtime()
        result = runtime.step(joint_targets)
        return {
            "type": "step_result",
            "observation": self.publish_observation(result.observation),
            "reward": float(result.reward),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "info": self._response_info(result.info),
        }

def _worker_main(
    connection: Connection,
    config_payload: dict[str, Any],
    top_buffer: Any,
    gripper_buffer: Any,
) -> None:
    config = GazeboWorkerConfig.from_payload(config_payload)
    os.environ["ROS_DOMAIN_ID"] = str(config.ros_domain_id)
    os.environ["GZ_PARTITION"] = config.gz_partition
    os.environ["IGN_PARTITION"] = config.gz_partition
    host = _WorkerRuntimeHost(
        config,
        top_buffer=top_buffer,
        gripper_buffer=gripper_buffer,
    )
    runtime_errors = _runtime_error_types()

    try:
        connection.send(
            host.start_world(
                max_attempts=config.startup_max_attempts,
                reason="initial_startup",
            )
        )
        while True:
            request = connection.recv()
            command = str(request.get("command", ""))
            try:
                if command == "prepare_episode":
                    connection.send(
                        host.prepare_episode(
                            episode_attempt_index=request.get(
                                "episode_attempt_index"
                            ),
                            task_mode=request.get("task_mode", "pick_place"),
                            scene_seed=request.get("scene_seed"),
                        )
                    )
                elif command == "start_training_stage":
                    connection.send(host.start_training_stage(
                        training_stage=request["training_stage"],
                        stage_config=request["stage_config"],
                    ))
                elif command == "step":
                    connection.send(host.step(request["joint_targets"]))
                elif command == "spawn_random_target":
                    response = host._require_runtime().spawn_random_target()
                    connection.send(
                        {
                            "type": "target_spawned",
                            **_response_from_service(response),
                        }
                    )
                elif command == "spawn_target":
                    response = host._require_runtime().spawn_target(
                        request["position_world_m"], request["quaternion_xyzw"])
                    connection.send({"type": "target_spawned", **_response_from_service(response)})
                elif command == "delete_all_targets":
                    raise RuntimeError(
                        "Target deletion is disabled in managed PPO worlds. "
                        "Persist the episode records and restart the world instead."
                    )
                elif command == "stop_world":
                    host.stop_world()
                    connection.send(
                        {
                            "type": "world_stopped",
                            "world_index": config.world_index,
                            "launch_pid": None,
                        }
                    )
                elif command == "restart_world":
                    reason = request.get("reason", "runtime_restart")
                    if reason not in {"runtime_restart", "episode_complete"}:
                        raise ValueError(f"Invalid world restart reason: {reason!r}")
                    connection.send(
                        host.start_world(
                            max_attempts=config.restart_max_attempts,
                            reason=reason,
                        )
                    )
                elif command == "shutdown":
                    host.stop_world()
                    connection.send(
                        {
                            "type": "shutdown_complete",
                            "world_index": config.world_index,
                            "launch_pid": None,
                        }
                    )
                    return
                else:
                    raise ValueError(f"Unknown worker command: {command!r}")
            except runtime_errors as error:
                payload = _error_payload(error, operation=command)
                payload["recoverable"] = True
                payload["launch_log_offset"] = host.launch_log_offset
                payload["gazebo_diagnostics"] = host.gazebo_diagnostics()
                connection.send(payload)
            except (ValueError, FloatingPointError) as error:
                payload = _error_payload(error, operation=command)
                payload["recoverable"] = True
                payload["launch_log_offset"] = host.launch_log_offset
                payload["gazebo_diagnostics"] = host.gazebo_diagnostics()
                connection.send(payload)
            except BaseException as error:
                payload = _error_payload(error, operation=command)
                payload["recoverable"] = False
                payload["launch_log_offset"] = host.launch_log_offset
                payload["gazebo_diagnostics"] = host.gazebo_diagnostics()
                connection.send(payload)
    except (EOFError, BrokenPipeError):
        return
    finally:
        try:
            host.stop_world()
        finally:
            connection.close()


class GazeboWorkerClient:
    """Parent-side control handle for one isolated Gazebo worker."""

    def __init__(
        self,
        config: GazeboWorkerConfig,
        *,
        context: mp.context.BaseContext | None = None,
    ) -> None:
        self.config = config
        self.context = mp.get_context("spawn") if context is None else context
        self.top_buffer = self.context.RawArray("B", _POLICY_IMAGE_BYTES)
        self.gripper_buffer = self.context.RawArray("B", _POLICY_IMAGE_BYTES)
        self.top_array = np.frombuffer(
            self.top_buffer, dtype=_POLICY_IMAGE_DTYPE
        ).reshape(_POLICY_IMAGE_SHAPE)
        self.gripper_array = np.frombuffer(
            self.gripper_buffer, dtype=_POLICY_IMAGE_DTYPE
        ).reshape(_POLICY_IMAGE_SHAPE)
        self.parent_connection: Connection | None = None
        self.process: mp.Process | None = None
        self.launch_pid: int | None = None
        self.launch_log_offset: int | None = None
        self.last_start_response: dict[str, Any] | None = None
        self._closed = False

    @property
    def world_index(self) -> int:
        return self.config.world_index

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def _spawn_process(self) -> None:
        if self.alive:
            raise RuntimeError("Gazebo worker process is already alive.")
        if self.parent_connection is not None:
            self.parent_connection.close()
        self._mark_launch_log_boundary()
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(
            target=_worker_main,
            args=(
                child,
                self.config.payload(),
                self.top_buffer,
                self.gripper_buffer,
            ),
            name=f"so101_gazebo_worker_{self.world_index:02d}",
            daemon=False,
        )
        process.start()
        child.close()
        self.parent_connection = parent
        self.process = process
        self.launch_pid = None

    def _mark_launch_log_boundary(self) -> None:
        try:
            self.launch_log_offset = self.config.log_path.stat().st_size
        except FileNotFoundError:
            self.launch_log_offset = 0
        except OSError:
            self.launch_log_offset = None

    def gazebo_diagnostics(self) -> dict[str, Any]:
        """Read current-launch evidence without asking an unresponsive worker."""

        return read_gazebo_crash_diagnostics(
            self.config.log_path,
            launch_log_offset=self.launch_log_offset,
        )

    def _hard_kill(self) -> None:
        if self.launch_pid is not None:
            try:
                os.killpg(self.launch_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        process = self.process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        self.launch_pid = None

    def _receive(self, *, timeout_s: float) -> dict[str, Any]:
        connection = self.parent_connection
        if connection is None:
            raise GazeboWorkerCommunicationError(
                "Worker pipe is not available.",
                gazebo_diagnostics=self.gazebo_diagnostics(),
            )
        try:
            response_available = connection.poll(timeout_s)
        except (OSError, EOFError) as error:
            diagnostics = self.gazebo_diagnostics()
            self._hard_kill()
            raise GazeboWorkerCommunicationError(
                f"World {self.world_index} worker pipe could not be polled.",
                gazebo_diagnostics=diagnostics,
            ) from error
        if not response_available:
            diagnostics = self.gazebo_diagnostics()
            self._hard_kill()
            raise GazeboWorkerCommunicationError(
                f"World {self.world_index} timed out after {timeout_s:.1f} s.",
                gazebo_diagnostics=diagnostics,
            )
        try:
            response = connection.recv()
        except (EOFError, OSError) as error:
            diagnostics = self.gazebo_diagnostics()
            self._hard_kill()
            raise GazeboWorkerCommunicationError(
                f"World {self.world_index} worker process disconnected.",
                gazebo_diagnostics=diagnostics,
            ) from error
        if "launch_log_offset" in response:
            self.launch_log_offset = response["launch_log_offset"]
        pid = response.get("launch_pid")
        if pid is not None:
            self.launch_pid = int(pid)
        elif response.get("type") in {
            "startup_failed",
            "restart_failed",
            "shutdown_complete",
        }:
            self.launch_pid = None
        return response

    def begin_start(self) -> None:
        self._spawn_process()

    def finish_start(self) -> dict[str, Any]:
        timeout = (
            self.config.startup_max_attempts
            * (self.config.startup_feedback_timeout_s + 20.0)
            + 10.0
        )
        response = self._receive(timeout_s=timeout)
        self.last_start_response = response
        return response

    def start(self) -> dict[str, Any]:
        self.begin_start()
        return self.finish_start()

    def _send(self, payload: dict[str, Any]) -> None:
        connection = self.parent_connection
        if connection is None or not self.alive:
            raise GazeboWorkerCommunicationError(
                f"World {self.world_index} worker process is not alive.",
                gazebo_diagnostics=self.gazebo_diagnostics(),
            )
        try:
            connection.send(payload)
        except (BrokenPipeError, EOFError, OSError) as error:
            diagnostics = self.gazebo_diagnostics()
            self._hard_kill()
            raise GazeboWorkerCommunicationError(
                f"World {self.world_index} worker command could not be sent.",
                gazebo_diagnostics=diagnostics,
            ) from error

    def request_async(self, command: str, **payload: Any) -> None:
        if command == "restart_world":
            # Ready will replace this provisional boundary with the exact
            # launch offset. Exclude the prior episode even if restart fails.
            self._mark_launch_log_boundary()
        self._send({"command": command, **payload})

    def receive(self, *, timeout_s: float) -> dict[str, Any]:
        return self._receive(timeout_s=timeout_s)

    def begin_restart(self, *, reason: str = "runtime_restart") -> str:
        if reason not in {"runtime_restart", "episode_complete"}:
            raise ValueError(f"Invalid world restart reason: {reason!r}")
        if not self.alive:
            self.begin_start()
            return "process_respawn"
        self.request_async("restart_world", reason=reason)
        return "runtime_restart"

    def finish_restart(self, mode: str) -> dict[str, Any]:
        attempts = (
            self.config.startup_max_attempts
            if mode == "process_respawn"
            else self.config.restart_max_attempts
        )
        timeout = (
            attempts
            * (self.config.startup_feedback_timeout_s + 20.0)
            + 10.0
        )
        response = self.receive(timeout_s=timeout)
        self.last_start_response = response
        return response

    def restart(self) -> dict[str, Any]:
        mode = self.begin_restart()
        return self.finish_restart(mode)

    def begin_stop(self) -> None:
        """Dispatch cleanup without waiting, keeping the worker reusable."""

        if not self.alive:
            raise GazeboWorkerCommunicationError(
                f"World {self.world_index} worker is not alive; "
                "world cleanup cannot be confirmed.",
                gazebo_diagnostics=self.gazebo_diagnostics(),
            )
        self.request_async("stop_world")

    def finish_stop(self) -> dict[str, Any]:
        """Require this world's cleanup acknowledgement before proceeding."""

        response = self.receive(timeout_s=_world_stop_timeout_s(self.config))
        if (
            not isinstance(response, dict)
            or response.get("type") != "world_stopped"
            or response.get("world_index") != self.world_index
        ):
            raise GazeboWorkerCommunicationError(
                f"World {self.world_index} cleanup was not confirmed: {response!r}",
                gazebo_diagnostics=self.gazebo_diagnostics(),
            )
        self.launch_pid = None
        return response

    def stop_world(self) -> dict[str, Any]:
        self.begin_stop()
        return self.finish_stop()

    def prepare_episode_async(
        self,
        episode_attempt_index: int | None = None,
        *,
        task_mode: str | None = None,
        scene_seed: int | None = None,
    ) -> None:
        if task_mode is not None:
            self.request_async("prepare_episode", episode_attempt_index=episode_attempt_index,
                               task_mode=task_mode, scene_seed=scene_seed)
            return
        if episode_attempt_index is None:
            self.request_async("prepare_episode")
            return
        self.request_async(
            "prepare_episode",
            episode_attempt_index=int(episode_attempt_index),
        )

    def start_training_stage_async(
        self, training_stage: str, stage_config: dict[str, Any],
    ) -> None:
        self.request_async(
            "start_training_stage", training_stage=training_stage,
            stage_config=stage_config,
        )

    def step_async(self, joint_targets: Any) -> None:
        self.request_async("step", joint_targets=np.asarray(joint_targets, dtype=float))

    def spawn_random_target_async(self) -> None:
        self.request_async("spawn_random_target")

    def spawn_target_async(self, position_world_m: Any, quaternion_xyzw: Any) -> None:
        self.request_async("spawn_target", position_world_m=list(position_world_m),
                           quaternion_xyzw=list(quaternion_xyzw))

    def delete_all_targets_async(self) -> None:
        self.request_async("delete_all_targets")

    def observation_from_metadata(self, metadata: dict[str, Any]) -> PolicyObservation:
        return PolicyObservation(
            joint_positions=np.asarray(metadata["joint_positions"], dtype=float),
            joint_velocities=np.asarray(metadata["joint_velocities"], dtype=float),
            top_image=self.top_array.copy(),
            gripper_image=self.gripper_array.copy(),
        )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.alive:
                try:
                    self.request_async("shutdown")
                    self.receive(timeout_s=_world_stop_timeout_s(self.config))
                except BaseException:
                    self._hard_kill()
        finally:
            if self.parent_connection is not None:
                try:
                    self.parent_connection.close()
                except BaseException:
                    pass
            if self.process is not None:
                self.process.join(timeout=2.0)
            self._hard_kill()

    def __enter__(self) -> "GazeboWorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.shutdown()
        return False
