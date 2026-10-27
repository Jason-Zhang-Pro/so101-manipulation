"""Multi-world Direct ACT PPO finetuning for deterministic SO-101 Gazebo.

One parent process owns the trainable ACT, critic, exploration parameters,
optimizer, rollout buffer, checkpoints, and TensorBoard. Each Gazebo world is
isolated in its own process, ROS_DOMAIN_ID, and Gazebo transport partition.
Worlds execute complete episodes concurrently; PPO updates only happen after a
complete wave of episodes. Optional stage-success gating excludes mastered
stages from the update and skips their next collection batch once.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
from .act_ppo_checkpoint_contract import actor_forward_contract
from .act_ppo_curriculum import ReachCurriculum
from .act_ppo_gazebo import (
    _configure_act_replanning, _make_curriculum, _sync_curriculum_state,
    _sample_training_scene, _resolved_stage_config, _episode_step_limit,
    _accumulate_task_metrics,
)
from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
from so101_capstone.policies.act_temporal_config import validate_act_temporal_config

from .act_ppo import (
    ActPpoRolloutBuffer,
    ActPpoTemporalHistory,
    ActPpoTrainer,
    ActPpoValueNetwork,
    actor_observation_from_batch,
    actor_observation_tensors,
    predict_action_chunk_with_grad,
    sample_bounded_policy_actions,
)
from .act_ppo_gazebo import (
    ActPpoGazeboConfig,
    RecoverableEpisodeError,
    _act_ppo_config,
    _actor_update_enabled,
    _append_jsonl,
    _critic_tensors,
    _critic_context, _critic_value, _resolved_reward, _execution_signature,
    _reward_cli_kwargs, _episode_scene, _episode_training_stage, _accumulate_action_diagnostics, _finalize_action_diagnostics,
    _default_output_directory,
    _copy_reward_config_for_export,
    _exploration_enabled,
    _exploration_stage,
    _map_policy_positions_for_episode,
    _parser as _single_world_parser,
    _set_seed,
    _should_save_update,
    _write_json,
    _DISABLED_DELTA_SENTINEL_RAD,
)
from .act_ppo_multiworld_cleanup import (
    PreflightCleanupReport,
    cleanup_stale_multiworld_processes,
)
from .act_ppo_multiworld_worker import (
    GazeboWorkerClient,
    GazeboWorkerCommunicationError,
    GazeboWorkerConfig,
    _cleanup_reserved_world_resources,
)
from .act_ppo_tensorboard import DirectActPpoTensorBoard
from .act_ppo_episode_records import (
    persist_episode_steps,
    sync_record,
    validate_complete_episode,
)
from .act_ppo_discard_stats import DiscardedStepStats
from .act_ppo_stage_schedule import (StageWaveScheduler, BalancedStageWaveScheduler,
                                     selected_training_stages, uses_later_training_stages,
                                     parse_stage_step_weights)
from .act_ppo_stage_success_skip import StageSuccessSkipController
from .residual_ppo import compress_policy_observation, model_state_sha256


DEFAULT_WORLD_COUNT = 1
DEFAULT_BASE_ROS_DOMAIN_ID = 20
DEFAULT_GZ_PARTITION_PREFIX = "gz_env_"
DEFAULT_WORLD_STARTUP_MAX_ATTEMPTS = 3
DEFAULT_WORLD_RESTART_MAX_ATTEMPTS = 3
DEFAULT_WORLD_DEFERRED_RESTART_MAX_ATTEMPTS = 3
DEFAULT_PREFLIGHT_CLEANUP_TIMEOUT_S = 5.0



# _TERMINAL_PROGRESS_LOGGING_V1
_TERMINAL_RULE = "=" * 80
_TERMINAL_STRONG_RULE = "#" * 80
_TERMINAL_HEARTBEAT_STEPS = 100
_CURRENT_WAVE_INDEX = 0


def _terminal_print_block(
    title: str,
    lines: list[str],
    *,
    strong: bool = False,
) -> None:
    rule = _TERMINAL_STRONG_RULE if strong else _TERMINAL_RULE
    print(rule, flush=True)
    print(f" {title}", flush=True)
    print(rule, flush=True)
    for line in lines:
        print(line, flush=True)
    print(rule, flush=True)


def _terminal_world_list(slots: list["_WorldSlot"]) -> str:
    if not slots:
        return "none"
    return " ".join(
        f"world{slot.world_index}"
        for slot in sorted(slots, key=lambda item: item.world_index)
    )


def _terminal_elapsed(start_monotonic_s: float) -> str:
    elapsed = max(0, int(time.monotonic() - start_monotonic_s))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _terminal_stage(
    info: dict[str, Any],
    base_config: ActPpoGazeboConfig,
) -> str:
    try:
        stage, _, _ = _exploration_stage(info, base_config)
        return str(stage)
    except BaseException:
        return "unknown"


# _TERMINAL_TARGET_PRESENCE_V1
def _terminal_target_present(info: dict[str, Any]) -> bool:
    # target_active comes from the same current environment feedback used
    # to resolve the reward state for this step.
    return bool(info.get("target_active", False))


def _terminal_checkpoint_lines(payload: dict[str, Any]) -> list[str]:
    if not bool(payload.get("checkpoint_saved", False)):
        return []
    lines = ["checkpoint     : saved"]
    for key in sorted(payload):
        value = payload[key]
        lowered = key.lower()
        if (
            isinstance(value, (str, Path))
            and (
                "checkpoint" in lowered
                or "training_state" in lowered
                or lowered.endswith("_path")
            )
        ):
            lines.append(f"{key:<14} : {value}")
    return lines


def _terminal_failure_block(record: dict[str, Any]) -> None:
    _terminal_print_block(
        (
            f"WAVE {_CURRENT_WAVE_INDEX:05d} "
            f"WORLD {int(record['world_index'])} EPISODE FAILED"
        ),
        [
            f"attempt        : {record.get('episode_attempt_index')}",
            f"category       : {record.get('failure_category')}",
            f"step           : {record.get('failed_step')}",
            f"wasted steps   : {record.get('discarded_environment_steps', record.get('failed_step'))} confirmed completed",
            f"unconfirmed    : {record.get('unconfirmed_step_count', 0)} step attempts (not included above)",
            f"Gazebo crash   : {bool(record.get('gazebo_crash', {}).get('confirmed', False))}",
            (
                "discarded      : "
                f"{record.get('discarded_transition_count')} transitions"
            ),
            (
                "consecutive    : "
                f"{record.get('consecutive_episode_failures')}/"
                f"{record.get('maximum_consecutive_episode_failures')}"
            ),
            f"reason         : {record.get('failure_message')}",
            *([
                "wasted total   : " + str(record["discarded_steps_cumulative"]["discarded_environment_steps"]),
                "crash wasted   : " + str(record["discarded_steps_cumulative"]["gazebo_crash_discarded_environment_steps"]),
            ] if "discarded_steps_cumulative" in record else []),
        ],
        strong=True,
    )


def _rollout_threshold_reached(transition_count: int, threshold: int) -> bool:
    if transition_count < 0 or threshold < 1:
        raise ValueError("Invalid rollout threshold inputs.")
    return transition_count >= threshold


def _episode_minimum_reached(completed_count: int, minimum_count: int) -> bool:
    if completed_count < 0 or minimum_count < 1:
        raise ValueError("Invalid episode-count inputs.")
    return completed_count >= minimum_count


def _deferred_world_restart_enabled(config: "MultiWorldConfig") -> bool:
    return config.world_restart_max_attempts == 0


def _worker_restart_attempt_limit(config: "MultiWorldConfig") -> int:
    if _deferred_world_restart_enabled(config):
        return config.world_deferred_restart_max_attempts
    return config.world_restart_max_attempts


def _validate_target_lifecycle(final_target_absent_s: float) -> None:
    if final_target_absent_s != 0.0:
        raise ValueError(
            "Managed PPO worlds keep targets until the episode is recorded "
            "and the world is rebuilt. Set --final-target-absent-s=0; "
            "in-episode target deletion is no longer supported."
        )


@dataclass(frozen=True)
class MultiWorldConfig:
    world_count: int = DEFAULT_WORLD_COUNT
    base_ros_domain_id: int = DEFAULT_BASE_ROS_DOMAIN_ID
    gz_partition_prefix: str = DEFAULT_GZ_PARTITION_PREFIX
    gazebo_gui: bool = True
    world_startup_max_attempts: int = DEFAULT_WORLD_STARTUP_MAX_ATTEMPTS
    world_restart_max_attempts: int = DEFAULT_WORLD_RESTART_MAX_ATTEMPTS
    world_deferred_restart_max_attempts: int = (
        DEFAULT_WORLD_DEFERRED_RESTART_MAX_ATTEMPTS
    )
    preflight_cleanup_enabled: bool = True
    preflight_cleanup_timeout_s: float = DEFAULT_PREFLIGHT_CLEANUP_TIMEOUT_S
    s0_step_fraction: float = 0.4
    stage_ratio_tolerance: float = 0.05
    stage_max_rollout_multiplier: float = 4.0
    stage_success_skip_enabled: bool = False
    stage_success_skip_threshold: float = 0.8
    stage_step_weights: str = "1:1:1:1:1:1"

    def __post_init__(self) -> None:
        parse_stage_step_weights(self.stage_step_weights)
        for name in (
            "world_count",
            "base_ros_domain_id",
            "world_startup_max_attempts",
            "world_restart_max_attempts",
            "world_deferred_restart_max_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.world_count < 1:
            raise ValueError("world_count must be positive.")
        if not 0 <= self.base_ros_domain_id <= 232:
            raise ValueError("base_ros_domain_id must be within [0, 232].")
        if self.base_ros_domain_id + self.world_count - 1 > 232:
            raise ValueError(
                "The requested world_count exceeds the ROS domain ID range."
            )
        if self.world_startup_max_attempts < 1:
            raise ValueError("world_startup_max_attempts must be positive.")
        if self.world_restart_max_attempts < 0:
            raise ValueError(
                "world_restart_max_attempts must be non-negative; "
                "0 defers recovery to wave boundaries."
            )
        if self.world_deferred_restart_max_attempts < 1:
            raise ValueError(
                "world_deferred_restart_max_attempts must be positive."
            )
        if not isinstance(self.gazebo_gui, bool):
            raise ValueError("gazebo_gui must be a bool.")
        if not isinstance(self.preflight_cleanup_enabled, bool):
            raise ValueError("preflight_cleanup_enabled must be a bool.")
        if not isinstance(self.stage_success_skip_enabled, bool):
            raise ValueError("stage_success_skip_enabled must be a bool.")
        threshold = self.stage_success_skip_threshold
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 < threshold <= 1):
            raise ValueError("stage_success_skip_threshold must be within (0, 1].")
        if float(self.preflight_cleanup_timeout_s) <= 0.0:
            raise ValueError("preflight_cleanup_timeout_s must be positive.")
        if not str(self.gz_partition_prefix):
            raise ValueError("gz_partition_prefix must not be empty.")
        for name, low, high in (
            ("s0_step_fraction", 0.0, 1.0),
            ("stage_ratio_tolerance", 0.0, 0.5),
            ("stage_max_rollout_multiplier", 1.0, math.inf),
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{name} must be finite and within [{low}, {high}].")

    def domain_id(self, world_index: int) -> int:
        return self.base_ros_domain_id + world_index

    def partition(self, world_index: int) -> str:
        return f"{self.gz_partition_prefix}{self.domain_id(world_index)}"

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["worlds"] = [
            {
                "world_index": index,
                "ros_domain_id": self.domain_id(index),
                "gz_partition": self.partition(index),
            }
            for index in range(self.world_count)
        ]
        return result


@dataclass
class _WorldSlot:
    client: GazeboWorkerClient
    available: bool = False
    quarantined: bool = False
    consecutive_episode_failures: int = 0
    successful_episode_count: int = 0  # legacy name: valid collected episodes
    task_success_count: int = 0
    failed_episode_count: int = 0
    restart_count: int = 0
    episode_restart_count: int = 0
    pending_episode_restart: bool = False
    startup_response: dict[str, Any] | None = None

    @property
    def world_index(self) -> int:
        return self.client.world_index


@dataclass
class _EpisodeSession:
    slot: _WorldSlot
    buffer: ActPpoRolloutBuffer
    observation: Any
    last_info: dict[str, Any]
    cleanup: dict[str, Any]
    home_reset: dict[str, Any]
    attempt_index: int
    completed_steps: int = 0
    step_in_flight: bool = False
    episode_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    spawn: dict[str, Any] = field(
        default_factory=lambda: {
            "requested": False,
            "request_completed_step": None,
            "service_success": None,
            "service_message": None,
        }
    )
    tail_target_cleanup: dict[str, Any] = field(
        default_factory=lambda: {
            "requested": False,
            "request_completed_step": None,
            "service_success": None,
            "service_message": None,
        }
    )
    exploration_enabled_steps: int = 0
    exploration_disabled_steps: int = 0
    actor_update_enabled_steps: int = 0
    actor_update_disabled_steps: int = 0
    exploration_mode_switches: int = 0
    sampled_actions: np.ndarray | None = None
    sampled_action_latents: np.ndarray | None = None
    sampled_log_probabilities: np.ndarray | None = None
    sampled_action_log_std: np.ndarray | None = None
    chunk_means: torch.Tensor | None = None
    task_mode: str = "pick_place"
    max_steps: int | None = None
    stage_initialization: dict[str, Any] = field(default_factory=dict)
    training_stage: str | None = None
    spawn_delay_steps: int | None = None
    curriculum_scene: dict[str, Any] = field(default_factory=dict)
    reward_totals: dict[str, float] = field(default_factory=dict)
    action_diagnostics: dict[str, Any] = field(default_factory=dict)
    stage_reached: dict[str, float] = field(default_factory=dict)
    chunk_actor_reference: int | None = None
    chunk_stage: str | None = None
    chunk_exploration_enabled: bool = True
    chunk_actor_update_enabled: bool = True
    chunk_log_std_offset: float | np.ndarray = 0.0
    chunk_action_index: int = 0
    chunk_executable: int = 0
    temporal_history: ActPpoTemporalHistory | None = None
    chunk_temporal_actor_references: tuple[int, ...] | None = None

    def clear_chunk(self) -> None:
        self.chunk_means = None
        self.sampled_actions = None
        self.sampled_action_latents = None
        self.sampled_log_probabilities = None
        self.sampled_action_log_std = None
        self.chunk_actor_reference = None
        self.chunk_stage = None
        self.chunk_action_index = 0
        self.chunk_executable = 0
        self.chunk_temporal_actor_references = None


@dataclass(frozen=True)
class _EpisodeFailure:
    slot: _WorldSlot
    attempt_index: int
    completed_steps: int
    discarded_transition_count: int
    category: str
    error_type: str
    message: str
    unconfirmed_step_count: int = 0
    stage_initialization: dict[str, Any] = field(default_factory=dict)
    gazebo_diagnostics: dict[str, Any] = field(default_factory=dict)


class _WorldShutdownError(RuntimeError):
    """Fatal cleanup failure, carrying completed episodes until they are saved."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.completed_sessions: list[_EpisodeSession] = []


def _gazebo_failure_diagnostics(client: Any, source: Any = None) -> dict[str, Any]:
    """Prefer evidence captured before timeout cleanup over a later log read."""
    captured = (source.get("gazebo_diagnostics") if isinstance(source, dict)
                else getattr(source, "gazebo_diagnostics", None))
    if isinstance(captured, dict) and captured:
        return dict(captured)
    reader = getattr(client, "gazebo_diagnostics", None)
    if callable(reader):
        try:
            return dict(reader())
        except Exception as error:
            return {"confirmed": False, "status": "unknown", "evidence": [],
                    "diagnostic_error": str(error)}
    return {"confirmed": False, "status": "unknown", "evidence": []}


def _discard_counters(snapshot: dict[str, Any]) -> dict[str, Any]:
    # Individual events already have their own durable failure files. Avoid
    # appending their whole growing history to metrics at every new event.
    return {key: value for key, value in snapshot.items() if key != "failure_events"}


def _record_aborted_rollout(
    rollout: ActPpoRolloutBuffer | None,
    *,
    discard_stats: DiscardedStepStats,
    output_directory: Path,
    tensorboard: DirectActPpoTensorBoard,
    update_in_progress: bool,
) -> None:
    if rollout is None or len(rollout) == 0:
        return
    counters = _discard_counters(discard_stats.record_aborted_rollout(
        len(rollout), update_in_progress=update_in_progress,
    ))
    _append_jsonl(output_directory / "metrics.jsonl", {
        "type": "interrupted_update_rollout" if update_in_progress else "aborted_rollout_discarded",
        "transition_count": len(rollout),
        "reason": "training_aborted",
        "discarded_steps_cumulative": counters,
    })
    sync_record(output_directory / "metrics.jsonl")
    tensorboard.log_discarded_steps(counters, event_index=int(counters["failure_episode_count"]) + 1)


def _worker_failure_category(response: dict[str, Any]) -> str:
    error_type = str(response.get("error_type", "worker_error"))
    message = str(response.get("error_message", "")).lower()
    if error_type == "EpisodeResetError":
        return "episode_reset"
    if error_type == "TargetCubeControlError":
        return "target_service"
    if error_type == "GazeboWorldControlError":
        return "gazebo_control_or_timing"
    if error_type == "ReachEnvironmentStateError":
        return "environment_feedback"
    if error_type == "SimulationRuntimeError":
        return "simulation_runtime"
    if error_type == "SimulationStepCoordinatorError":
        if "camera" in message or "image" in message:
            return "camera"
        if "joint" in message:
            return "joint_feedback"
        if "clock" in message or "simulation" in message or "physics" in message:
            return "gazebo_deterministic_timing"
        return "simulation_step"
    if error_type == "GazeboWorkerCommunicationError":
        return "worker_communication"
    return "worker_operation"


def _is_ready_response(response: dict[str, Any]) -> bool:
    return response.get("type") == "ready"


def _spawn_payload_from_worker(
    response: dict[str, Any],
    *,
    completed_step: int,
) -> dict[str, Any]:
    if response.get("type") != "target_spawned" or not bool(
        response.get("success", False)
    ):
        error = RecoverableEpisodeError(
            "target_service",
            str(response.get("error_message") or response.get("message") or response),
        )
        error.gazebo_diagnostics = response.get("gazebo_diagnostics")
        raise error
    return {
        "requested": True,
        "request_completed_step": completed_step,
        "service_success": True,
        "service_message": str(response.get("message", "")),
    }



def _build_base_config(args: argparse.Namespace) -> ActPpoGazeboConfig:
    _validate_target_lifecycle(args.final_target_absent_s)
    return ActPpoGazeboConfig(
        act_checkpoint=args.act_checkpoint,
        **_reward_cli_kwargs(args),
        device=args.device,
        episode_count=args.episode_count,
        max_consecutive_episode_failures=args.max_consecutive_episode_failures,
        max_steps_per_episode=args.max_steps_per_episode,
        target_delay_s=args.target_delay_s,
        final_target_absent_s=args.final_target_absent_s,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        critic_image_height=args.critic_image_height,
        critic_image_width=args.critic_image_width,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_learning_rate=args.critic_learning_rate,
        exploration_learning_rate=args.exploration_learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        kl_backoff_max_retries=args.kl_backoff_max_retries,
        kl_backoff_factor=args.kl_backoff_factor,
        kl_backoff_min_scale=args.kl_backoff_min_scale,
        value_loss_coefficient=args.value_loss_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        mean_bounds_coefficient=args.mean_bounds_coefficient,
        raw_mean_bounds_coefficient=args.raw_mean_bounds_coefficient,
        max_gradient_norm=args.max_gradient_norm,
        critic_max_gradient_norm=args.critic_max_gradient_norm,
        exploration_max_gradient_norm=args.exploration_max_gradient_norm,
        ppo_epochs=args.ppo_epochs,
        initial_log_std=args.initial_log_std,
        minimum_log_std=args.minimum_log_std,
        maximum_log_std=args.maximum_log_std,
        state1_exploration_log_std=args.state1_exploration_log_std,
        state2_exploration_log_std=args.state2_exploration_log_std,
        state3_exploration_log_std=args.state3_exploration_log_std,
        state4_not_home_exploration_log_std=(
            args.state4_not_home_exploration_log_std
        ),
        state4_home_exploration_log_std=args.state4_home_exploration_log_std,
        state1_exploration_enabled=args.state1_exploration_enabled,
        state2_exploration_enabled=args.state2_exploration_enabled,
        state3_exploration_enabled=args.state3_exploration_enabled,
        state4_not_home_exploration_enabled=(
            args.state4_not_home_exploration_enabled
        ),
        state4_home_exploration_enabled=args.state4_home_exploration_enabled,
        state1_actor_update_enabled=args.state1_actor_update_enabled,
        state2_actor_update_enabled=args.state2_actor_update_enabled,
        state3_actor_update_enabled=args.state3_actor_update_enabled,
        state4_not_home_actor_update_enabled=(
            args.state4_not_home_actor_update_enabled
        ),
        state4_home_actor_update_enabled=args.state4_home_actor_update_enabled,
        actor_learning_rate=args.actor_learning_rate,
        actor_backbone_learning_rate=args.actor_backbone_learning_rate,
        checkpoint_every_updates=args.checkpoint_every_updates,
        tensorboard_enabled=args.tensorboard,
        tensorboard_auto_open=args.open_tensorboard,
        tensorboard_port=args.tensorboard_port,
        disable_max_joint_delta_limit=args.disable_max_joint_delta_limit,
        startup_feedback_timeout_s=args.startup_feedback_timeout_s,
        seed=args.seed,
        output_directory=args.output_directory,
        resume_checkpoint=args.resume_checkpoint,
    )


def _parser() -> argparse.ArgumentParser:
    def stage_step_weights_argument(value: str) -> str:
        try:
            parse_stage_step_weights(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        return value

    parser = _single_world_parser()
    parser.description = (
        "Use one PPO learner with multiple isolated Gazebo worlds and export "
        "a standalone ACT. Each world is rebuilt after its complete episode "
        "has been recorded; targets are never deleted in a running world."
    )
    parser.add_argument("--s0-step-fraction", type=float, default=0.4,
                        help="Target S0 fraction of actual PPO rollout steps in s0_s1; replaces episode-probability sampling.")
    parser.add_argument("--stage-step-weights", type=stage_step_weights_argument, default="1:1:1:1:1:1",
                        help="Positive accepted-step weights for S0:S1:S2:S3:S4:S5 in s0_s5; skipped stages are removed and remaining weights renormalized.")
    parser.add_argument("--stage-ratio-tolerance", type=float, default=0.05,
                        help="Maximum absolute error in each active stage's step fraction before PPO (0.05 means five percentage points).")
    parser.add_argument("--stage-max-rollout-multiplier", type=float, default=4.0,
                        help="Stop before another wave if an unbalanced stage rollout reaches this multiple of rollout-steps.")
    parser.add_argument("--stage-success-skip", action=argparse.BooleanOptionalAction, default=False,
                        help="Exclude successful stages from this update and skip their next complete collection batch once.")
    parser.add_argument("--stage-success-skip-threshold", type=float, default=0.8,
                        help="Minimum per-stage valid-episode success fraction to trigger one-batch skipping (0 < value <= 1).")
    parser.add_argument(
        "--world-count",
        type=int,
        default=DEFAULT_WORLD_COUNT,
        help="Maximum concurrent Gazebo worlds; stage tail waves may launch fewer to meet rollout size and ratio.",
    )
    parser.add_argument(
        "--base-ros-domain-id",
        type=int,
        default=DEFAULT_BASE_ROS_DOMAIN_ID,
        help="ROS_DOMAIN_ID used by world 0; later worlds increment by one.",
    )
    parser.add_argument(
        "--gz-partition-prefix",
        default=DEFAULT_GZ_PARTITION_PREFIX,
        help="Gazebo transport partition prefix; domain ID is appended.",
    )
    parser.add_argument(
        "--gazebo-gui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open one Gazebo GUI for every world; disable for server-only -s.",
    )
    parser.add_argument(
        "--world-startup-max-attempts",
        type=int,
        default=DEFAULT_WORLD_STARTUP_MAX_ATTEMPTS,
        help="Maximum consecutive launch attempts for initial world startup.",
    )
    parser.add_argument(
        "--world-restart-max-attempts",
        type=int,
        default=DEFAULT_WORLD_RESTART_MAX_ATTEMPTS,
        help=(
            "Immediate same-wave rebuild attempts; 0 disables "
            "same-wave rebuild and defers recovery to a wave boundary."
        ),
    )
    parser.add_argument(
        "--world-deferred-restart-max-attempts",
        type=int,
        default=DEFAULT_WORLD_DEFERRED_RESTART_MAX_ATTEMPTS,
        help=(
            "Maximum rebuild attempts at each wave boundary when "
            "--world-restart-max-attempts=0."
        ),
    )
    parser.add_argument(
        "--preflight-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before workers start, terminate stale managed SO-101/Gazebo "
            "processes in the ROS domains and Gazebo partitions reserved "
            "by this finetune."
        ),
    )
    parser.add_argument(
        "--preflight-cleanup-timeout-s",
        type=float,
        default=DEFAULT_PREFLIGHT_CLEANUP_TIMEOUT_S,
        help="Wall-clock grace after SIGTERM before stale processes receive SIGKILL.",
    )
    return parser


def _checkpoint_paths(
    trainer: ActPpoTrainer,
    *,
    output_directory: Path,
    name: str,
    base_config: ActPpoGazeboConfig,
    multiworld_config: MultiWorldConfig,
    metrics: dict[str, Any],
) -> dict[str, str]:
    directory = output_directory / "checkpoints" / name
    directory.mkdir(parents=True, exist_ok=False)
    training_state = trainer.save_checkpoint(
        directory / "training_state.pt",
        extra={
            "act_checkpoint": str(base_config.act_checkpoint.resolve()),
            "gazebo_config": base_config.to_json_dict(),
            "collection_state": getattr(trainer, "collection_state", {}),
            "multiworld_config": multiworld_config.to_json_dict(),
            "metrics": metrics,
        },
    )
    standalone = directory / "pretrained_model"
    trainer.policy.save_pretrained(standalone)
    _copy_reward_config_for_export(output_directory, standalone)
    return {
        "training_state": str(training_state),
        "standalone_act": str(standalone),
    }


def _export_final(
    trainer: ActPpoTrainer,
    *,
    output_directory: Path,
    base_config: ActPpoGazeboConfig,
    multiworld_config: MultiWorldConfig,
    metrics: dict[str, Any] | None,
) -> tuple[Path, Path]:
    standalone = output_directory / "pretrained_model"
    trainer.policy.save_pretrained(standalone)
    _copy_reward_config_for_export(output_directory, standalone)
    state = trainer.save_checkpoint(
        output_directory / "final_training_state.pt",
        extra={
            "act_checkpoint": str(base_config.act_checkpoint.resolve()),
            "gazebo_config": base_config.to_json_dict(),
            "collection_state": getattr(trainer, "collection_state", {}),
            "multiworld_config": multiworld_config.to_json_dict(),
            "metrics": metrics,
            "final_standalone_act": str(standalone),
        },
    )
    return standalone, state


def _log_world_event(
    output_directory: Path,
    payload: dict[str, Any],
) -> None:
    _append_jsonl(
        output_directory / "metrics.jsonl",
        {"type": "world_event", **payload},
    )


def _start_clients_concurrently(
    slots: list[_WorldSlot],
    *,
    output_directory: Path,
) -> None:
    for slot in slots:
        slot.client.begin_start()
    for slot in slots:
        try:
            response = slot.client.finish_start()
        except GazeboWorkerCommunicationError as error:
            response = {
                "type": "startup_failed",
                "world_index": slot.world_index,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        slot.startup_response = response
        slot.available = _is_ready_response(response)
        slot.quarantined = not slot.available
        _log_world_event(
            output_directory,
            {
                "event": "initial_ready" if slot.available else "initial_failed",
                "world_index": slot.world_index,
                "response": response,
            },
        )
    # Startup can fail after spawning descendants. Unavailable slots must not
    # leave a simulator running beside the healthy worlds or the later PPO.
    _stop_worlds_concurrently(
        [slot for slot in slots if not slot.available],
        output_directory=output_directory,
    )


def _recover_slots_concurrently(
    slots: list[_WorldSlot],
    *,
    output_directory: Path,
    reset_episode_failure_budget: bool,
    quarantine_on_failure: bool = True,
    reason: str = "runtime_failure",
) -> None:
    pending: list[tuple[_WorldSlot, str, str, bool, bool]] = []
    failed: list[_WorldSlot] = []
    for slot in slots:
        if slot.available and not slot.quarantined:
            continue
        restart_reason = "episode_complete" if slot.pending_episode_restart else reason
        planned_restart = restart_reason == "episode_complete"
        quarantine = False if planned_restart else bool(quarantine_on_failure)
        reset_budget = bool(reset_episode_failure_budget and not planned_restart)
        slot.pending_episode_restart = False
        try:
            mode = (
                slot.client.begin_restart(reason="episode_complete")
                if planned_restart
                else slot.client.begin_restart()
            )
            pending.append((slot, mode, restart_reason, quarantine, reset_budget))
        except GazeboWorkerCommunicationError as error:
            slot.available = False
            slot.quarantined = quarantine
            failed.append(slot)
            _log_world_event(
                output_directory,
                {
                    "event": "restart_begin_failed",
                    "reason": restart_reason,
                    "world_index": slot.world_index,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "quarantine_on_failure": quarantine,
                },
            )

    for slot, mode, restart_reason, quarantine, reset_budget in pending:
        try:
            response = slot.client.finish_restart(mode)
        except GazeboWorkerCommunicationError as error:
            response = {
                "type": "restart_failed",
                "world_index": slot.world_index,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        slot.available = _is_ready_response(response)
        if slot.available:
            slot.quarantined = False
            slot.restart_count += 1
            if restart_reason == "episode_complete":
                slot.episode_restart_count += 1
            if reset_budget:
                slot.consecutive_episode_failures = 0
        else:
            # Deferred recovery failures stay pending for the next wave.
            slot.quarantined = quarantine
            failed.append(slot)

        _log_world_event(
            output_directory,
            {
                "event": "restart_ready" if slot.available else "restart_failed",
                "reason": restart_reason,
                "world_index": slot.world_index,
                "response": response,
                "reset_episode_failure_budget": reset_budget,
                "quarantine_on_failure": quarantine,
            },
        )
        print(
            "[WORLD RECOVERY] "
            f"world={slot.world_index} "
            f"result={'READY' if slot.available else 'FAILED'} "
            f"mode={mode} "
            f"reason={restart_reason} "
            f"attempts_used={response.get('attempts_used', 'n/a')}",
            flush=True,
        )
    _stop_worlds_concurrently(failed, output_directory=output_directory)


def _prepare_worlds_for_wave(
    slots: list[_WorldSlot], *, output_directory: Path,
    multiworld_config: MultiWorldConfig, starting_new_rollout: bool,
) -> None:
    """Create replacements only when the next operation is episode collection."""
    deferred = _deferred_world_restart_enabled(multiworld_config)
    pending = [
        slot for slot in slots
        if slot.pending_episode_restart
        or (starting_new_rollout and (not slot.available or slot.quarantined))
        or (deferred and not slot.available and not slot.quarantined)
    ]
    if not pending:
        return
    _terminal_print_block(
        "PREPARE WORLDS FOR NEXT WAVE",
        ["worlds           : " + _terminal_world_list(pending)],
    )
    _recover_slots_concurrently(
        pending, output_directory=output_directory,
        reset_episode_failure_budget=starting_new_rollout,
        quarantine_on_failure=not deferred,
    )


def _prepare_stage_worlds_for_wave(
    slots: list[_WorldSlot], *, world_count: int, output_directory: Path,
    multiworld_config: MultiWorldConfig, starting_new_rollout: bool, selection_offset: int = 0,
) -> list[_WorldSlot]:
    """Start only the planned number of worlds; keep unused worlds closed.

    Rotate by the saved scene-attempt counter so partial waves do not always
    consume world zero's stream or need another checkpoint counter. A failed
    launch can be replaced by an unused slot
    without increasing concurrent simulation count above the wave plan.
    """
    candidates = sorted(
        (slot for slot in slots if not slot.quarantined),
        key=lambda slot: (slot.world_index - selection_offset) % multiworld_config.world_count,
    )
    ready: list[_WorldSlot] = []
    while candidates and len(ready) < world_count:
        count = world_count - len(ready)
        chosen, candidates = candidates[:count], candidates[count:]
        fresh = [slot for slot in chosen if slot.startup_response is None]
        if fresh:
            _start_clients_concurrently(fresh, output_directory=output_directory)
        existing = [slot for slot in chosen if all(slot is not item for item in fresh)]
        _prepare_worlds_for_wave(
            existing, output_directory=output_directory,
            multiworld_config=multiworld_config, starting_new_rollout=starting_new_rollout,
        )
        ready.extend(slot for slot in chosen if slot.available and not slot.quarantined)
    return ready


def _stop_worlds_concurrently(
    slots: list[_WorldSlot], *, output_directory: Path | None = None,
) -> None:
    """Stop all worlds and confirm cleanup before allowing PPO or another wave."""
    pending: list[_WorldSlot] = []
    fallback: list[tuple[_WorldSlot, Exception]] = []
    for slot in slots:
        slot.available = False
        try:
            slot.client.begin_stop()
            pending.append(slot)
        except GazeboWorkerCommunicationError as error:
            fallback.append((slot, error))
    for slot in pending:
        try:
            response = slot.client.finish_stop()
            if output_directory is not None:
                _log_world_event(output_directory, {
                    "event": "world_stopped", "world_index": slot.world_index,
                    "response": response,
                })
        except GazeboWorkerCommunicationError as error:
            fallback.append((slot, error))
    errors = []
    for slot, error in fallback:
        # A dead/unresponsive worker cannot acknowledge cleanup. Terminate it
        # first, then reuse the same domain/partition cleanup barrier in the
        # parent. A failed barrier aborts instead of training beside old worlds.
        try:
            slot.client._hard_kill()
            cleanup = _cleanup_reserved_world_resources(slot.client.config)
        except Exception as cleanup_error:
            errors.append(f"world{slot.world_index}: {cleanup_error}")
            if output_directory is not None:
                _log_world_event(output_directory, {
                    "event": "world_stop_failed", "world_index": slot.world_index,
                    "error_message": str(cleanup_error),
                })
        else:
            if output_directory is not None:
                _log_world_event(output_directory, {
                    "event": "world_stopped", "world_index": slot.world_index,
                    "forced_cleanup": True, "cleanup": cleanup,
                    "worker_error": str(error),
                })
    if errors:
        raise _WorldShutdownError("Gazebo world shutdown could not be confirmed: " + "; ".join(errors))


def _stop_failed_worlds(failures: list[_EpisodeFailure]) -> None:
    unique = {failure.slot.world_index: failure.slot for failure in failures}
    _stop_worlds_concurrently(list(unique.values()))


def _environment_randomization_enabled() -> bool:
    """Whether preparation needs a deterministic episode randomization key."""
    enabled = []
    for name in ("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "SO101_IMAGE_AUGMENTATION_ENABLED"):
        value = os.environ.get(name, "false").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            enabled.append(True)
        elif value in {"0", "false", "no", "off", ""}:
            enabled.append(False)
        else:
            raise ValueError(f"{name} must be a boolean value.")
    return any(enabled)


def _spawn_session_target(session: _EpisodeSession) -> None:
    if session.curriculum_scene:
        session.slot.client.spawn_target_async(session.curriculum_scene["target_position_world_m"],
                                                session.curriculum_scene["target_quaternion_xyzw"])
    else:
        session.slot.client.spawn_random_target_async()


def _prepare_sessions(
    slots: list[_WorldSlot],
    *,
    ppo_config: Any,
    base_config: ActPpoGazeboConfig,
    attempt_counter: list[int],
    on_failure: Callable[[_EpisodeFailure], None] | None = None,
    curriculum: ReachCurriculum | None = None,
) -> tuple[list[_EpisodeSession], list[_EpisodeFailure]]:
    pending_attempts: list[tuple[_WorldSlot, int]] = []
    failures: list[_EpisodeFailure] = []

    def report_failure(failure: _EpisodeFailure) -> None:
        if on_failure is not None:
            on_failure(failure)
        failures.append(failure)

    randomization_enabled = _environment_randomization_enabled()
    for slot in slots:
        attempt_counter[0] += 1
        attempt_index = attempt_counter[0]
        if curriculum is not None:
            curriculum.attempt_count = attempt_index
        try:
            if base_config.reward_profile == "pick_place_v2":
                mode, delay, seed = _episode_scene(base_config, attempt_index, slot.world_index)
                slot.client.prepare_episode_async(attempt_index, task_mode=mode, scene_seed=seed)
            elif randomization_enabled:
                slot.client.prepare_episode_async(attempt_index)
            else:
                slot.client.prepare_episode_async()
            pending_attempts.append((slot, attempt_index))
        except GazeboWorkerCommunicationError as error:
            report_failure(_EpisodeFailure(
                slot=slot, attempt_index=attempt_index, completed_steps=0,
                discarded_transition_count=0, category="worker_communication",
                error_type=type(error).__name__, message=str(error),
                gazebo_diagnostics=_gazebo_failure_diagnostics(slot.client, error),
            ))

    sessions: list[_EpisodeSession] = []
    for slot, attempt_index in pending_attempts:
        try:
            response = slot.client.receive(timeout_s=180.0)
        except GazeboWorkerCommunicationError as error:
            report_failure(
                _EpisodeFailure(
                    slot=slot,
                    attempt_index=attempt_index,
                    completed_steps=0,
                    discarded_transition_count=0,
                    category="worker_communication",
                    error_type=type(error).__name__,
                    message=str(error),
                    gazebo_diagnostics=_gazebo_failure_diagnostics(slot.client, error),
                )
            )
            continue
        if response.get("type") != "episode_prepared":
            report_failure(
                _EpisodeFailure(
                    slot=slot,
                    attempt_index=attempt_index,
                    completed_steps=0,
                    discarded_transition_count=0,
                    category=_worker_failure_category(response),
                    error_type=str(response.get("error_type", "WorkerError")),
                    message=str(response.get("error_message", response)),
                    gazebo_diagnostics=_gazebo_failure_diagnostics(slot.client, response),
                )
            )
            continue
        observation = slot.client.observation_from_metadata(response["observation"])
        session = _EpisodeSession(
            slot=slot,
            buffer=ActPpoRolloutBuffer(ppo_config),
            observation=observation,
            last_info=dict(response["info"]),
            cleanup=dict(response["cleanup"]),
            home_reset=dict(response["home_reset"]),
            attempt_index=attempt_index,
        )
        session.task_mode, session.spawn_delay_steps, _ = _episode_scene(base_config, attempt_index, slot.world_index)
        if base_config.training_stages != "full":
            session.training_stage = _episode_training_stage(base_config, attempt_index, slot.world_index)
        session.max_steps = _episode_step_limit(base_config, session.task_mode, session.training_stage)
        session.curriculum_scene = _sample_training_scene(base_config, curriculum,
            seed=_episode_scene(base_config, attempt_index, slot.world_index)[2],
            task_mode=session.task_mode, info=session.last_info)
        sessions.append(session)
        print(
            f"[WAVE {_CURRENT_WAVE_INDEX:05d}]"
            f"[world{slot.world_index}] "
            f"episode attempt={attempt_index} START | "
            f"max_steps={session.max_steps} | "
            f"target_delay_steps={session.spawn_delay_steps}",
            flush=True,
        )

    immediate = [s for s in sessions if s.task_mode == "pick_place" and s.spawn_delay_steps == 0]
    if immediate:
        sent: list[_EpisodeSession] = []
        for session in immediate:
            try:
                _spawn_session_target(session)
                sent.append(session)
            except GazeboWorkerCommunicationError as error:
                report_failure(_EpisodeFailure(
                    slot=session.slot, attempt_index=session.attempt_index, completed_steps=0,
                    discarded_transition_count=0, category="worker_communication",
                    error_type=type(error).__name__, message=str(error),
                    gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, error),
                ))
        kept: list[_EpisodeSession] = [s for s in sessions if all(s is not item for item in immediate)]
        for session in sent:
            try:
                response = session.slot.client.receive(timeout_s=10.0)
                session.spawn = _spawn_payload_from_worker(response, completed_step=0)
                kept.append(session)
            except (GazeboWorkerCommunicationError, RecoverableEpisodeError) as error:
                report_failure(
                    _EpisodeFailure(
                        slot=session.slot,
                        attempt_index=session.attempt_index,
                        completed_steps=0,
                        discarded_transition_count=0,
                        category=(
                            "worker_communication"
                            if isinstance(error, GazeboWorkerCommunicationError)
                            else error.category
                        ),
                        error_type=type(error).__name__,
                        message=str(error),
                        gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, error),
                    )
                )
        sessions = kept

    stage_config = _resolved_stage_config(base_config)
    if stage_config is not None:
        sessions, stage_failures = _start_training_stage_sessions(
            sessions, stage_config=stage_config.to_dict(), on_failure=on_failure,
            initialization_timeout_s=base_config.resolved_stage_initialization_timeout_s,
        )
        failures.extend(stage_failures)
    return sessions, failures


def _start_training_stage_sessions(
    sessions: list[_EpisodeSession], *, stage_config: dict[str, Any],
    initialization_timeout_s: float = 180.0,
    on_failure: Callable[[_EpisodeFailure], None] | None = None,
) -> tuple[list[_EpisodeSession], list[_EpisodeFailure]]:
    """Cross the physical frame-zero barrier before any ACT sampling begins."""
    pending: list[tuple[_EpisodeSession, str]] = []
    ready: list[_EpisodeSession] = []
    failures: list[_EpisodeFailure] = []

    def failed(session: _EpisodeSession, error: Exception, source: Any = None) -> None:
        initialization = (dict(source.get("stage_initialization", {}))
                          if isinstance(source, dict) else {})
        if not initialization:
            # A dead/timed-out worker cannot certify how much preparation it
            # completed. Keep that unknown, independently of zero PPO steps.
            initialization = {
                "valid": False,
                "training_stage": session.training_stage or (
                    "S0" if session.task_mode == "idle_empty" else "S1"),
                "phase": "worker_unavailable",
                "preparation_control_steps": None,
                "policy_steps": 0, "reason": str(error),
            }
        failure = _EpisodeFailure(
            slot=session.slot, attempt_index=session.attempt_index,
            completed_steps=0, discarded_transition_count=0,
            category="stage_initialization", error_type=type(error).__name__,
            message=str(error),
            stage_initialization=initialization,
            gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, source or error),
        )
        failures.append(failure)
        if on_failure is not None:
            on_failure(failure)

    for session in sessions:
        stage = session.training_stage or ("S0" if session.task_mode == "idle_empty" else "S1")
        try:
            session.slot.client.start_training_stage_async(stage, stage_config)
            pending.append((session, stage))
        except GazeboWorkerCommunicationError as error:
            failed(session, error)
    for session, stage in pending:
        response = None
        try:
            response = session.slot.client.receive(timeout_s=initialization_timeout_s)
            if response.get("type") != "stage_started":
                raise RecoverableEpisodeError(
                    "stage_initialization", str(response.get("error_message", response)),
                )
            info = dict(response["info"])
            if str(info.get("training_stage", "")).upper() != stage:
                raise RecoverableEpisodeError(
                    "stage_initialization", f"Expected {stage} frame zero, received {info.get('training_stage')!r}.",
                )
            session.observation = session.slot.client.observation_from_metadata(response["observation"])
            session.last_info = info
            session.stage_initialization = {
                key: info[key] for key in (
                    "stage_initialization", "stage_start_certificate", "start_certificate",
                ) if key in info
            }
            # No action, reward, or actor reference from the controller's
            # discarded preparation steps is admitted into this new segment.
            session.clear_chunk()
            session.temporal_history = None
            ready.append(session)
            print(
                f"[world{session.slot.world_index}] {stage} READY | "
                f"attempt={session.attempt_index} | max_steps={session.max_steps} | "
                "ACT steps=0",
                flush=True,
            )
        except (GazeboWorkerCommunicationError, RecoverableEpisodeError) as error:
            failed(session, error, response)
    return ready, failures


def _refresh_chunk(
    session: _EpisodeSession, *, adapter: ActPolicyAdapter, policy: Any,
    trainer: ActPpoTrainer, device: torch.device, loaded_config: Any,
    base_config: ActPpoGazeboConfig, ppo_config: Any,
    store_actor_observation: bool = True,
) -> None:
    actor_observation = actor_observation_from_batch(adapter.build_batch(session.observation))
    session.chunk_actor_reference = (
        session.buffer.add_actor_observation(actor_observation) if store_actor_observation else None)
    with torch.no_grad():
        session.chunk_means = predict_action_chunk_with_grad(
            policy, actor_observation_tensors(actor_observation, device=device))
        coefficient = getattr(loaded_config, "temporal_ensemble_coeff", None)
        if coefficient is not None:
            if session.temporal_history is None:
                validate_act_temporal_config(loaded_config)
                session.temporal_history = ActPpoTemporalHistory(coefficient, int(loaded_config.chunk_size))
            session.chunk_means = session.temporal_history.push(
                session.chunk_actor_reference, session.chunk_means)
            session.chunk_temporal_actor_references = (
                session.temporal_history.actor_references if store_actor_observation else None
            )
    session.chunk_action_index = 0
    session.chunk_executable = int(loaded_config.n_action_steps)
    # Noise is sampled at each control step; only deterministic ACT means are cached.
    if base_config.reward_profile == "legacy":
        _sample_session_action(session, trainer=trainer, base_config=base_config, ppo_config=ppo_config)


def _sample_session_action(session: _EpisodeSession, *, trainer: Any,
                           base_config: ActPpoGazeboConfig, ppo_config: Any) -> None:
    stage, _, offset = _exploration_stage(session.last_info, base_config)
    session.chunk_stage = stage
    session.chunk_exploration_enabled = _exploration_enabled(session.last_info, base_config)
    session.chunk_actor_update_enabled = _actor_update_enabled(session.last_info, base_config)
    session.chunk_log_std_offset = np.asarray(offset, dtype=np.float32).copy()
    std = trainer.exploration.with_offset(offset)
    assert session.chunk_means is not None
    means = session.chunk_means
    if base_config.reward_profile == "pick_place_v2":
        means = means[:, session.chunk_action_index:session.chunk_action_index + 1]
    with torch.no_grad():
        sampled, probabilities, latents = sample_bounded_policy_actions(
            means, std, deterministic=not session.chunk_exploration_enabled, return_latent=True)
    session.sampled_actions = sampled[0].detach().cpu().double().numpy()
    session.sampled_action_latents = latents[0].detach().cpu().numpy()
    session.sampled_log_probabilities = probabilities[0].detach().cpu().double().numpy()
    session.sampled_action_log_std = std.detach().cpu().numpy().copy()


def _run_sessions(
    sessions: list[_EpisodeSession],
    *,
    adapter: ActPolicyAdapter,
    policy: Any,
    critic: Any,
    trainer: ActPpoTrainer,
    device: torch.device,
    loaded_config: Any,
    base_config: ActPpoGazeboConfig,
    ppo_config: Any,
    on_failure: Callable[[_EpisodeFailure], None] | None = None,
) -> tuple[list[_EpisodeSession], list[_EpisodeFailure]]:
    _validate_target_lifecycle(base_config.final_target_absent_s)
    active = {session.slot.world_index: session for session in sessions}
    for session in sessions:
        if session.max_steps is None:
            if base_config.training_stages != "full":
                session.training_stage = session.training_stage or _episode_training_stage(
                    base_config, session.attempt_index, session.slot.world_index)
            session.max_steps = _episode_step_limit(base_config, session.task_mode, session.training_stage)
    completed: list[_EpisodeSession] = []
    failures: list[_EpisodeFailure] = []

    def report_failure(failure: _EpisodeFailure) -> None:
        if on_failure is not None:
            on_failure(failure)
        failures.append(failure)

    while active:
        step_context: dict[int, dict[str, Any]] = {}
        failed_before_send: list[int] = []
        for world_index, session in list(active.items()):
            try:
                if session.chunk_means is None:
                    _refresh_chunk(
                        session,
                        adapter=adapter,
                        policy=policy,
                        trainer=trainer,
                        device=device,
                        loaded_config=loaded_config,
                        base_config=base_config,
                        ppo_config=ppo_config,
                    )
                if base_config.reward_profile == "pick_place_v2":
                    _sample_session_action(session, trainer=trainer, base_config=base_config, ppo_config=ppo_config)
                assert session.sampled_actions is not None
                assert session.sampled_log_probabilities is not None
                assert session.sampled_action_log_std is not None
                assert session.chunk_actor_reference is not None
                action_index = session.chunk_action_index
                compact = compress_policy_observation(
                    session.observation,
                    image_height=ppo_config.critic_image_height,
                    image_width=ppo_config.critic_image_width,
                )
                contexts = _critic_context(session.last_info, session.chunk_means, action_index,
                                           int(loaded_config.n_action_steps), base_config)
                value = _critic_value(critic, compact, contexts, device)
                sample_index = 0 if base_config.reward_profile == "pick_place_v2" else action_index
                sampled_policy_action = session.sampled_actions[sample_index]
                _accumulate_action_diagnostics(session.action_diagnostics, session.chunk_stage,
                    session.chunk_means[:, action_index:action_index+1], sampled_policy_action)
                policy_result = _map_policy_positions_for_episode(
                    adapter,
                    session.observation,
                    sampled_policy_action,
                )
                step_context[world_index] = {
                    "compact": compact,
                    "contexts": contexts,
                    "actor_reference": session.chunk_actor_reference,
                    "temporal_actor_references": session.chunk_temporal_actor_references,
                    "latent": session.sampled_action_latents[sample_index].copy(),
                    # Freeze sampling parameters before bootstrap refreshes the chunk.
                    "policy_action_mean": session.chunk_means[0, action_index].detach().cpu().numpy().copy(),
                    "policy_action_log_std": session.sampled_action_log_std.copy(),
                    "actor_update_enabled": session.chunk_actor_update_enabled,
                    "std_offset": session.chunk_log_std_offset,
                    "value": float(value.item()),
                    "action_index": action_index,
                    "policy_action": sampled_policy_action,
                    "log_probability": float(
                        session.sampled_log_probabilities[sample_index]
                    ),
                }
                session.step_in_flight = True
                session.slot.client.step_async(
                    policy_result.action.controller_command()
                )
            except (RecoverableEpisodeError, GazeboWorkerCommunicationError) as error:
                report_failure(
                    _EpisodeFailure(
                        slot=session.slot,
                        attempt_index=session.attempt_index,
                        completed_steps=session.completed_steps,
                        discarded_transition_count=len(session.buffer),
                        category=(
                            error.category
                            if isinstance(error, RecoverableEpisodeError)
                            else "worker_communication"
                        ),
                        error_type=type(error).__name__,
                        message=str(error),
                        unconfirmed_step_count=int(session.step_in_flight),
                        gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, error),
                    )
                )
                failed_before_send.append(world_index)
        for world_index in failed_before_send:
            active.pop(world_index, None)

        responses: dict[int, dict[str, Any]] = {}
        for world_index in list(step_context):
            session = active.get(world_index)
            if session is None:
                continue
            try:
                responses[world_index] = session.slot.client.receive(timeout_s=15.0)
            except GazeboWorkerCommunicationError as error:
                report_failure(
                    _EpisodeFailure(
                        slot=session.slot,
                        attempt_index=session.attempt_index,
                        completed_steps=session.completed_steps,
                        discarded_transition_count=len(session.buffer),
                        category="worker_communication",
                        error_type=type(error).__name__,
                        message=str(error),
                        unconfirmed_step_count=int(session.step_in_flight),
                        gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, error),
                    )
                )
                active.pop(world_index, None)

        spawn_worlds: list[int] = []
        for world_index, response in list(responses.items()):
            session = active.get(world_index)
            if session is None:
                continue
            if response.get("type") != "step_result":
                report_failure(
                    _EpisodeFailure(
                        slot=session.slot,
                        attempt_index=session.attempt_index,
                        completed_steps=session.completed_steps,
                        discarded_transition_count=len(session.buffer),
                        category=_worker_failure_category(response),
                        error_type=str(response.get("error_type", "WorkerError")),
                        message=str(response.get("error_message", response)),
                        unconfirmed_step_count=int(session.step_in_flight),
                        gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, response),
                    )
                )
                active.pop(world_index, None)
                continue
            session.step_in_flight = False
            session.completed_steps += 1
            session.episode_reward += float(response["reward"])
            if session.chunk_exploration_enabled:
                session.exploration_enabled_steps += 1
            else:
                session.exploration_disabled_steps += 1
            if session.chunk_actor_update_enabled:
                session.actor_update_enabled_steps += 1
            else:
                session.actor_update_disabled_steps += 1
            if (
                session.task_mode == "pick_place"
                and not bool(response["terminated"]) and not bool(response["truncated"])
                and not session.spawn["requested"]
                and session.completed_steps == (base_config.target_delay_steps if session.spawn_delay_steps is None else session.spawn_delay_steps)
            ):
                try:
                    _spawn_session_target(session)
                    spawn_worlds.append(world_index)
                except GazeboWorkerCommunicationError as error:
                    report_failure(
                        _EpisodeFailure(
                            slot=session.slot,
                            attempt_index=session.attempt_index,
                            completed_steps=session.completed_steps,
                            discarded_transition_count=len(session.buffer),
                            category="worker_communication",
                            error_type=type(error).__name__,
                            message=str(error),
                            unconfirmed_step_count=int(session.step_in_flight),
                            gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, error),
                        )
                    )
                    active.pop(world_index, None)

        spawn_failed: set[int] = set()
        for world_index in spawn_worlds:
            session = active.get(world_index)
            if session is None:
                continue
            try:
                response = session.slot.client.receive(timeout_s=10.0)
                session.spawn = _spawn_payload_from_worker(
                    response,
                    completed_step=session.completed_steps,
                )
            except (GazeboWorkerCommunicationError, RecoverableEpisodeError) as error:
                report_failure(
                    _EpisodeFailure(
                        slot=session.slot,
                        attempt_index=session.attempt_index,
                        completed_steps=session.completed_steps,
                        discarded_transition_count=len(session.buffer),
                        category=(
                            "worker_communication"
                            if isinstance(error, GazeboWorkerCommunicationError)
                            else error.category
                        ),
                        error_type=type(error).__name__,
                        message=str(error),
                        unconfirmed_step_count=int(session.step_in_flight),
                        gazebo_diagnostics=_gazebo_failure_diagnostics(session.slot.client, error),
                    )
                )
                spawn_failed.add(world_index)
                active.pop(world_index, None)

        for world_index, response in responses.items():
            if world_index in spawn_failed:
                continue
            session = active.get(world_index)
            if session is None:
                continue
            context = step_context[world_index]
            next_observation = session.slot.client.observation_from_metadata(
                response["observation"]
            )
            next_compact = compress_policy_observation(
                next_observation,
                image_height=ppo_config.critic_image_height,
                image_width=ppo_config.critic_image_width,
            )
            session.terminated = bool(response["terminated"])
            session.truncated = bool(response["truncated"])
            reached_limit = session.completed_steps >= session.max_steps
            episode_end = session.terminated or session.truncated or reached_limit
            old_stage = session.chunk_stage
            session.observation = next_observation
            session.last_info = dict(response["info"])
            next_stage, _, _ = _exploration_stage(session.last_info, base_config)
            exploration_mode_changed = next_stage != old_stage
            if exploration_mode_changed:
                session.exploration_mode_switches += 1
            session.chunk_action_index += 1
            legacy_refresh = base_config.reward_profile == "legacy" and (
                exploration_mode_changed or session.completed_steps == base_config.target_delay_steps)
            if not session.terminated and (session.chunk_action_index >= session.chunk_executable or legacy_refresh):
                _refresh_chunk(session, adapter=adapter, policy=policy, trainer=trainer, device=device,
                    loaded_config=loaded_config, base_config=base_config, ppo_config=ppo_config,
                    store_actor_observation=not episode_end)
            next_value = 0.0
            if not session.terminated:
                next_contexts = _critic_context(session.last_info, session.chunk_means, session.chunk_action_index,
                                                int(loaded_config.n_action_steps), base_config)
                next_value = float(_critic_value(critic, next_compact, next_contexts, device).item())
            session.buffer.add(
                context["compact"], actor_observation_index=context["actor_reference"],
                temporal_actor_references=context["temporal_actor_references"],
                action_index=context["action_index"], policy_action=context["policy_action"],
                policy_action_latent=context["latent"], log_probability=context["log_probability"],
                policy_action_mean=context["policy_action_mean"],
                policy_action_log_std=context["policy_action_log_std"],
                actor_update_enabled=context["actor_update_enabled"], exploration_log_std_offset=context["std_offset"],
                value=context["value"], reward=float(response["reward"]), next_value=next_value,
                terminated=session.terminated, episode_end=episode_end, **context["contexts"])
            for key, component in session.last_info.get("reward_components", {}).items():
                if isinstance(component, (int, float)):
                    session.reward_totals[key] = session.reward_totals.get(key, 0.0) + float(component)
            _accumulate_task_metrics(session.stage_reached, session.last_info)
            if (
                session.completed_steps % _TERMINAL_HEARTBEAT_STEPS == 0
            ):
                print(
                    f"[W{_CURRENT_WAVE_INDEX:05d}]"
                    f"[world{world_index}]"
                    f"[attempt{session.attempt_index}] "
                    f"step {session.completed_steps}/"
                    f"{session.max_steps} | "
                    f"training_stage={session.last_info.get('training_stage', 'FULL')} | "
                    f"state={_terminal_stage(session.last_info, base_config)} | "
                    f"stage_result={session.last_info.get('stage_result', 'n/a')} | "
                    f"reward={float(response['reward']):+.4f} | "
                    f"return={session.episode_reward:+.4f} | "
                    "target="
                    f"{'yes' if _terminal_target_present(session.last_info) else 'no'}",
                    flush=True,
                )
            if episode_end:
                completed.append(session)
                active.pop(world_index, None)

    return completed, failures


def _episode_record(
    session: _EpisodeSession,
    *,
    episode_index: int,
    base_config: ActPpoGazeboConfig,
    rollout_size_after_episode: int,
) -> dict[str, Any]:
    stage_fields = {
        key: value for key, value in session.last_info.items()
        if key == "training_stage" or key.startswith("stage_")
        or key in {"start_certificate", "start_valid", "assisted_initialization", "full_unassisted_success"}
    }
    return {
        "episode_index": episode_index,
        "episode_attempt_index": session.attempt_index,
        **session.curriculum_scene,
        "world_index": session.slot.world_index,
        "valid": True,
        "step_count": session.completed_steps,
        "total_reward": session.episode_reward,
        "terminated": session.terminated,
        "truncated": session.truncated,
        "reached_max_steps": (
            session.completed_steps >= (
                session.max_steps if session.max_steps is not None
                else _episode_step_limit(base_config, session.task_mode)
            )
            and not session.terminated
            and not session.truncated
        ),
        "target": session.spawn,
        "tail_target_cleanup": session.tail_target_cleanup,
        "cleanup_before_episode": session.cleanup,
        "home_reset": session.home_reset,
        "final_info": session.last_info,
        "task_mode": session.task_mode,
        "task_success": session.last_info.get("task_outcome") == "success",
        "stage_reached": session.stage_reached,
        "reward_components_total": session.reward_totals,
        "action_diagnostics": _finalize_action_diagnostics(session.action_diagnostics),
        "rollout_buffer_size_after_episode": rollout_size_after_episode,
        "exploration_enabled_step_count": session.exploration_enabled_steps,
        "exploration_disabled_step_count": session.exploration_disabled_steps,
        "exploration_mode_switch_count": session.exploration_mode_switches,
        "actor_update_enabled_step_count": session.actor_update_enabled_steps,
        "actor_update_disabled_step_count": session.actor_update_disabled_steps,
        **session.stage_initialization,
        **stage_fields,
    }


def _training_stage_summary(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep local milestones separate from the physical full-task success count."""
    result = {}
    for stage in ("S0", "S1", "S2", "S3", "S4", "S5"):
        rows = [row for row in episodes if str(row.get("training_stage", "")).upper() == stage]
        if not rows:
            continue
        successes = sum(bool(row.get("stage_success", False)) for row in rows)
        result[stage] = {
            "valid_episode_count": len(rows),
            "collected_steps": sum(int(row["step_count"]) for row in rows),
            "stage_success_count": successes,
            "stage_success_rate": successes / len(rows),
            "stage_timeout_count": sum(str(row.get("stage_result", "")).upper() == "TIMEOUT" for row in rows),
            "task_failure_count": sum(row.get("final_info", {}).get("task_outcome") == "task_failure" for row in rows),
        }
    return result


def _initial_attempt_count(
    config: ActPpoGazeboConfig, curriculum: ReachCurriculum | None,
    resume_payload: dict[str, Any] | None,
) -> int:
    """Restore scene-seed progression without involving the spatial curriculum."""
    if config.training_stages == "full":
        return 0 if curriculum is None else curriculum.attempt_count
    if resume_payload is None:
        return 0
    saved = resume_payload.get("extra", {}).get("collection_state", {})
    if "stage_attempt_count" not in saved:
        raise ValueError(
            "Stage PPO resume is missing stage_attempt_count; "
            "use ACT weights-only warm-start instead of replaying old scene seeds."
        )
    count = saved["stage_attempt_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("stage_attempt_count must be a non-negative integer.")
    return count


def _sync_stage_attempt_count(
    trainer: Any, config: ActPpoGazeboConfig, attempt_count: int,
) -> None:
    if config.training_stages != "full":
        trainer.collection_state = {
            **getattr(trainer, "collection_state", {}),
            "stage_attempt_count": attempt_count,
        }


def _complete_episode_wave(
    sessions: list[_EpisodeSession],
    *,
    rollout: ActPpoRolloutBuffer,
    base_config: ActPpoGazeboConfig,
    output_directory: Path,
    first_episode_index: int,
    tensorboard: DirectActPpoTensorBoard,
) -> list[dict[str, Any]]:
    """Record every complete trajectory, then close its world without recreating it.

    The parent owns copied rollout observations and transitions. A replacement
    Gazebo process never owns or resets that data or the learner's optimizer.
    A validation or persistence error aborts this boundary without a rebuild.
    """
    ordered = sorted(sessions, key=lambda item: item.slot.world_index)
    for session in ordered:
        validate_complete_episode(session.buffer, session.completed_steps)

    records: list[dict[str, Any]] = []
    for offset, session in enumerate(ordered):
        episode_index = first_episode_index + offset
        before = len(rollout)
        appended = rollout.extend(session.buffer)
        if appended != session.completed_steps or len(rollout) - before != appended:
            raise RuntimeError("Complete episode was not fully appended to the PPO rollout.")

        steps_path = output_directory / "episode_steps" / f"episode_{episode_index:06d}.jsonl"
        step_records = persist_episode_steps(steps_path, session.buffer, session.completed_steps)
        step_records["path"] = str(steps_path.relative_to(output_directory))
        record = _episode_record(
            session, episode_index=episode_index, base_config=base_config,
            rollout_size_after_episode=len(rollout),
        )
        record["step_records"] = step_records
        record["recorded_transition_count"] = appended
        record["wave_index"] = _CURRENT_WAVE_INDEX
        record["world_lifecycle"] = "stop_after_recording_start_before_next_wave"
        episode_path = output_directory / "episodes" / f"episode_{episode_index:06d}.json"
        _write_json(episode_path, record)
        sync_record(episode_path)
        metrics_path = output_directory / "metrics.jsonl"
        _append_jsonl(metrics_path, {"type": "episode", **record})
        sync_record(metrics_path)
        tensorboard.log_episode(record)
        records.append(record)

        _terminal_print_block(
            f"WAVE {_CURRENT_WAVE_INDEX:05d} WORLD {session.slot.world_index} EPISODE RECORDED",
            [
                f"attempt        : {session.attempt_index}",
                f"steps          : {session.completed_steps}",
                f"recorded       : {step_records['step_count']} steps (files synced)",
                f"step log       : {step_records['path']}",
                f"return         : {session.episode_reward:+.4f}",
                f"final state    : {_terminal_stage(session.last_info, base_config)}",
                *([
                    f"training stage : {record['training_stage']}",
                    f"stage result   : {record.get('stage_result', 'RUNNING')}",
                    f"stage time     : {record.get('stage_elapsed_s', 0):.3f} / {record.get('stage_budget_s', 0):.3f} s",
                ] if "training_stage" in record else []),
                f"terminated     : {session.terminated}",
                f"truncated      : {session.truncated}",
                f"accepted       : +{appended} transitions",
                f"rollout        : {before} -> {len(rollout)} / {base_config.rollout_steps}",
            ],
        )

    if ordered:
        # All successful worlds have crossed the recording barrier. Failures
        # must abort before PPO if cleanup cannot be confirmed. Replacements
        # are launched at the next wave boundary, after any pending PPO update.
        for session in ordered:
            _log_world_event(output_directory, {
                "event": "episode_stop_begin",
                "world_index": session.slot.world_index,
                "episode_attempt_index": session.attempt_index,
                "recorded_step_count": session.completed_steps,
            })
            session.slot.available = False
            session.slot.quarantined = False
        _stop_worlds_concurrently(
            [session.slot for session in ordered],
            output_directory=output_directory,
        )
        for session in ordered:
            session.slot.pending_episode_restart = True
        _terminal_print_block(
            "EPISODE WORLDS CLOSED",
            ["worlds           : " + _terminal_world_list([session.slot for session in ordered]),
             "next startup     : immediately before the next collection wave"],
        )
    return records


def _record_failure(
    failure: _EpisodeFailure,
    *,
    output_directory: Path,
    maximum_consecutive_failures: int,
    discard_stats: DiscardedStepStats | None = None,
    tensorboard: DirectActPpoTensorBoard | None = None,
) -> dict[str, Any]:
    slot = failure.slot
    slot.failed_episode_count += 1
    slot.consecutive_episode_failures += 1
    diagnostics = failure.gazebo_diagnostics or _gazebo_failure_diagnostics(slot.client)
    confirmed_crash = diagnostics.get("confirmed") is True
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_index": slot.world_index,
        "episode_attempt_index": failure.attempt_index,
        "valid": False,
        "failed_step": failure.completed_steps,
        "discarded_environment_steps": failure.completed_steps,
        "discarded_transition_count": failure.discarded_transition_count,
        "unconfirmed_step_count": failure.unconfirmed_step_count,
        "failure_category": "gazebo_crash" if confirmed_crash else failure.category,
        "reported_failure_category": failure.category,
        "gazebo_crash": diagnostics,
        "failure_type": failure.error_type,
        "failure_message": failure.message,
        **({"stage_initialization": failure.stage_initialization} if failure.stage_initialization else {}),
        "consecutive_episode_failures": slot.consecutive_episode_failures,
        "maximum_consecutive_episode_failures": maximum_consecutive_failures,
    }
    failure_path = (
        output_directory
        / "episode_failures"
        / f"attempt_{failure.attempt_index:06d}_world_{slot.world_index:02d}.json"
    )
    record["failure_record_path"] = str(failure_path.relative_to(output_directory))
    _write_json(failure_path, record)
    sync_record(failure_path)
    _append_jsonl(
        output_directory / "metrics.jsonl",
        {"type": "episode_failure", **record},
    )
    sync_record(output_directory / "metrics.jsonl")
    if discard_stats is not None:
        counters = _discard_counters(discard_stats.record_failure(record))
        record["discarded_steps_cumulative"] = counters
        _append_jsonl(output_directory / "metrics.jsonl", {"type": "discarded_steps", **counters})
        if tensorboard is not None:
            tensorboard.log_discarded_steps(counters, event_index=int(counters["failure_episode_count"]))
    _terminal_failure_block(record)
    return record


def _run_wave(
    slots: list[_WorldSlot],
    *,
    adapter: ActPolicyAdapter,
    policy: Any,
    critic: Any,
    trainer: ActPpoTrainer,
    device: torch.device,
    loaded_config: Any,
    base_config: ActPpoGazeboConfig,
    multiworld_config: MultiWorldConfig,
    ppo_config: Any,
    output_directory: Path,
    attempt_counter: list[int],
    discard_stats: DiscardedStepStats | None = None,
    tensorboard: DirectActPpoTensorBoard | None = None,
    curriculum: ReachCurriculum | None = None,
    wave_training_stage: str | None = None,
) -> tuple[list[_EpisodeSession], list[dict[str, Any]], int]:
    if base_config.training_stages != "full":
        configured_stages = selected_training_stages(base_config.training_stages)
        if wave_training_stage is None and len(configured_stages) == 1:
            wave_training_stage = configured_stages[0]
        if wave_training_stage not in configured_stages:
            raise ValueError("Stage waves require one explicit stage within the selected training stages.")
        # Freeze scene selection for the entire wave, including every retry.
        # The attempt/world-based target and image seeds still advance normally.
        base_config = replace(base_config, training_stages=wave_training_stage.lower())
    elif wave_training_stage is not None:
        raise ValueError("A wave stage requires stage training.")
    pending = [slot for slot in slots if slot.available and not slot.quarantined]
    successes: list[_EpisodeSession] = []
    failure_records: list[dict[str, Any]] = []
    discarded_environment_steps = 0
    recorded_failures: set[tuple[int, int]] = set()

    def record_failure_now(failure: _EpisodeFailure) -> None:
        nonlocal discarded_environment_steps
        key = (failure.slot.world_index, failure.attempt_index)
        if key in recorded_failures:
            return
        record = _record_failure(
            failure, output_directory=output_directory,
            maximum_consecutive_failures=base_config.max_consecutive_episode_failures,
            discard_stats=discard_stats, tensorboard=tensorboard,
        )
        recorded_failures.add(key)
        failure_records.append(record)
        discarded_environment_steps += failure.completed_steps

    while pending:
        sessions, prepare_failures = _prepare_sessions(
            pending,
            ppo_config=ppo_config,
            base_config=base_config,
            attempt_counter=attempt_counter,
            on_failure=record_failure_now,
            **({"curriculum": curriculum} if curriculum is not None else {}),
        )
        completed, run_failures = _run_sessions(
            sessions,
            adapter=adapter,
            policy=policy,
            critic=critic,
            trainer=trainer,
            device=device,
            loaded_config=loaded_config,
            base_config=base_config,
            ppo_config=ppo_config,
            on_failure=record_failure_now,
        )
        successes.extend(completed)
        for session in completed:
            session.slot.successful_episode_count += 1
            session.slot.task_success_count += int(session.last_info.get("task_outcome") == "success")
            session.slot.consecutive_episode_failures = 0

        failures = [*prepare_failures, *run_failures]
        if not failures:
            break

        for failure in failures:
            record_failure_now(failure)

        try:
            _stop_failed_worlds(failures)
        except _WorldShutdownError as error:
            error.completed_sessions = successes
            raise

        retry_slots: list[_WorldSlot] = []
        seen: set[int] = set()
        for failure in failures:
            slot = failure.slot
            if slot.world_index in seen:
                continue
            seen.add(slot.world_index)
            if (
                slot.consecutive_episode_failures
                >= base_config.max_consecutive_episode_failures
            ):
                slot.available = False
                slot.quarantined = True
                _log_world_event(
                    output_directory,
                    {
                        "event": "quarantined_for_current_rollout",
                        "world_index": slot.world_index,
                        "consecutive_episode_failures": (
                            slot.consecutive_episode_failures
                        ),
                    },
                )
            else:
                retry_slots.append(slot)

        if not retry_slots:
            break

        if _deferred_world_restart_enabled(multiworld_config):
            # Do not rebuild or rerun failed worlds inside this wave.
            for slot in retry_slots:
                slot.available = False
                slot.quarantined = False
                _log_world_event(
                    output_directory,
                    {
                        "event": "restart_deferred_to_next_wave",
                        "world_index": slot.world_index,
                        "deferred_restart_max_attempts": (
                            multiworld_config.world_deferred_restart_max_attempts
                        ),
                    },
                )
            break

        # Positive-N mode keeps the old same-wave behavior.
        try:
            _recover_slots_concurrently(
                retry_slots,
                output_directory=output_directory,
                reset_episode_failure_budget=False,
            )
        except _WorldShutdownError as error:
            error.completed_sessions = successes
            raise
        pending = [
            slot for slot in retry_slots
            if slot.available and not slot.quarantined
        ]

    return successes, failure_records, discarded_environment_steps


def _stage_sampling_counts(sampling: dict[str, Any]) -> str:
    return ", ".join(f"S{i}={sampling[f's{i}_steps']}" for i in range(6) if f"s{i}_steps" in sampling)


def _stage_sampling_fractions(sampling: dict[str, Any]) -> str:
    if "stage_fractions" in sampling:
        return f"shares={sampling['stage_fractions']} target={sampling['target_stage_fractions']} +/- {sampling['ratio_tolerance']:.4f}"
    return (f"S0 share={sampling['actual_s0_step_fraction']} "
            f"target={sampling['target_s0_step_fraction']:.4f} +/- {sampling['ratio_tolerance']:.4f}")


def _perform_update(
    rollout: ActPpoRolloutBuffer,
    *,
    trainer: ActPpoTrainer,
    output_directory: Path,
    base_config: ActPpoGazeboConfig,
    multiworld_config: MultiWorldConfig,
    tensorboard: DirectActPpoTensorBoard,
    rollout_index: int,
    stage_scheduler: StageWaveScheduler | None = None,
    stage_success_skip: StageSuccessSkipController | None = None,
) -> dict[str, Any]:
    if base_config.training_stages != "full":
        if stage_scheduler is None or stage_scheduler.total_steps != len(rollout) or not stage_scheduler.ready:
            raise RuntimeError("Stage PPO update requires a complete rollout with the requested actual step ratio.")
    stage_sampling = None if stage_scheduler is None else stage_scheduler.diagnostics()
    collected_count = len(rollout)
    skip_decision = None
    if stage_success_skip is not None:
        if stage_scheduler is None:
            raise RuntimeError("Stage success skipping requires a complete stage sampling batch.")
        decision = stage_success_skip.finish_batch()
        skip_decision = decision.to_dict()
        excluded = decision.exclude_update_stages
        if excluded:
            rollout = rollout.without_stage_groups({int(stage[1:]) for stage in excluded})
        expected = collected_count - sum(stage_sampling[f"{stage.lower()}_steps"] for stage in excluded)
        if len(rollout) != expected:
            raise RuntimeError("Stage success filtering disagrees with collected stage counts.")
        skip_decision.update({
            "collected_transition_count": collected_count,
            "update_transition_count": len(rollout),
            "excluded_transition_count": collected_count - len(rollout),
        })
        _sync_stage_success_skip(trainer, stage_success_skip)
        _append_jsonl(output_directory / "metrics.jsonl", {
            "type": "stage_success_skip_decision", "rollout_index": rollout_index,
            "update_index_before": trainer.update_index, **skip_decision,
        })
        tensorboard.log_stage_success_skip(skip_decision)
        _terminal_print_block("STAGE SUCCESS GATE", [
            f"success threshold  : {multiworld_config.stage_success_skip_threshold:.0%}",
            f"success rates      : {skip_decision['success_rates']}",
            f"excluded this PPO  : {sorted(excluded)}",
            f"skip next sampling : {sorted(decision.next_skip_stages)} (one batch)",
            f"collected / update : {collected_count} / {len(rollout)} transitions",
        ])
        if not len(rollout):
            payload = {
                "type": "stage_update_skipped", "rollout_index": rollout_index,
                "update_index": trainer.update_index, "checkpoint_saved": False,
                "collected_rollout_transition_count": collected_count,
                "actual_rollout_transition_count": 0, "stage_sampling": stage_sampling,
                "stage_success_skip": skip_decision,
                "reason": "all_collected_stages_met_success_threshold",
            }
            _append_jsonl(output_directory / "metrics.jsonl", payload)
            print("[PPO SKIP] All sampled stages met the success threshold; no optimizer step or update increment.", flush=True)
            return payload
    pending_update_index = trainer.update_index + 1
    _terminal_print_block(
        f"PPO UPDATE {pending_update_index:06d} START",
        [
            f"rollout index      : {rollout_index}",
            f"transitions        : {len(rollout)}",
            *([] if stage_sampling is None else [
                f"collected stages   : {_stage_sampling_counts(stage_sampling)}",
                f"collection balance : {_stage_sampling_fractions(stage_sampling)}",
            ]),
            f"epochs             : {trainer.config.ppo_epochs}",
            f"policy KL limit    : {trainer.config.target_kl}",
            ("KL rejection       : restore epoch; at most "
             f"{trainer.config.kl_backoff_max_retries} smaller-step retries"),
            f"minibatch size     : {trainer.config.minibatch_size}",
            f"actor LR           : {trainer.actor_learning_rate:.6g}",
            (
                "backbone LR        : "
                f"{trainer.actor_backbone_learning_rate:.6g}"
            ),
            f"critic LR          : {trainer.config.critic_learning_rate:.6g}",
            f"raw bounds weight  : {trainer.config.raw_mean_bounds_coefficient:.6g}",
            (
                "log std mean       : "
                f"{float(trainer.exploration.value().detach().mean().cpu()):.6f}"
            ),
        ],
        strong=True,
    )
    metrics = trainer.update(rollout)
    payload = {
        "type": "ppo_update",
        "update_index": trainer.update_index,
        "rollout_index": rollout_index,
        "partial_rollout": False,
        "configured_rollout_step_threshold": base_config.rollout_steps,
        "actual_rollout_transition_count": len(rollout),
        **({"collected_rollout_transition_count": collected_count,
            "success_gated_transition_count": collected_count - len(rollout),
            "stage_success_skip": skip_decision} if skip_decision is not None else {}),
        **({"stage_sampling": stage_sampling} if stage_sampling is not None else {}),
        **metrics.to_dict(),
    }
    if _should_save_update(
        trainer.update_index,
        base_config.checkpoint_every_updates,
    ):
        payload["checkpoint_saved"] = True
        payload.update(
            _checkpoint_paths(
                trainer,
                output_directory=output_directory,
                name=f"update_{trainer.update_index:06d}",
                base_config=base_config,
                multiworld_config=multiworld_config,
                metrics=payload,
            )
        )
    else:
        payload["checkpoint_saved"] = False
    tensorboard.log_update(
        payload,
        total_environment_steps=trainer.total_environment_steps,
    )
    _append_jsonl(output_directory / "metrics.jsonl", payload)
    complete_lines = [
        f"rollout index   : {rollout_index}",
        f"transitions     : {metrics.transition_count}",
        (
            "epochs          : "
            f"{metrics.completed_ppo_epochs}/"
            f"{metrics.configured_ppo_epochs}"
        ),
        f"early stopped   : {metrics.early_stopped}",
        f"policy KL limit : {metrics.target_kl:.6f}",
        f"KL retries      : {metrics.kl_backoff_retry_count}",
        f"KL rejections   : {metrics.kl_backoff_rejection_count}",
        f"policy LR scale : {metrics.policy_learning_rate_scale:.8g}",
        f"at LR minimum   : {metrics.kl_backoff_at_min_scale}",
        (
            "precheck KL     : "
            f"{metrics.pre_update_pre_clip_policy_kl:.8g} (analytic truncated)"
        ),
        (
            "precheck clip   : "
            f"{metrics.pre_update_clip_fraction:.6f}"
        ),
        (
            "precheck max|lr|: "
            f"{metrics.pre_update_max_abs_log_ratio:.8g}"
        ),
        (
            "policy KL       : "
            f"{metrics.pre_clip_policy_kl:.8g} (guard; old||new)"
        ),
        f"policy KL p95   : {metrics.pre_clip_policy_kl_p95:.8g}",
        f"policy KL max   : {metrics.pre_clip_policy_kl_max:.8g}",
        f"sampled final KL: {metrics.final_policy_approximate_kl:.8g} (diagnostic)",
        (
            "final policy clip: "
            f"{metrics.final_policy_clip_fraction:.6f}"
        ),
        f"policy loss     : {metrics.policy_loss:+.6f}",
        f"value loss      : {metrics.value_loss:.6f}",
        *([] if metrics.raw_mean_bounds_diagnostic_count == 0 else [
            f"raw bounds loss : {metrics.raw_mean_bounds_loss:.6g} (unweighted; accepted minibatches)",
            f"raw overflow    : {metrics.raw_mean_out_of_bounds_fraction:.2%} of joint predictions",
            f"raw overflow avg: {metrics.raw_mean_overflow_normalized:.6g} (half-range units)",
            f"bound derivative: {metrics.raw_mean_bound_derivative_mean:.6g} (mean)",
        ]),
        f"truncated entropy: {metrics.entropy:.6f}",
        f"sampled mean KL : {metrics.approximate_kl:.6f} (diagnostic)",
        f"clip fraction   : {metrics.clip_fraction:.6f}",
        f"gradient norm   : {metrics.gradient_norm:.6f}",
        f"log std mean    : {metrics.log_std_mean:.6f}",
        f"env steps total : {trainer.total_environment_steps}",
        *_terminal_checkpoint_lines(payload),
    ]
    _terminal_print_block(
        f"PPO UPDATE {trainer.update_index:06d} COMPLETE",
        complete_lines,
        strong=True,
    )
    return payload


def _stage_wave_scheduler(base_config: ActPpoGazeboConfig, multiworld_config: MultiWorldConfig,
                          skipped_stages: frozenset[str] = frozenset()):
    if base_config.training_stages == "full":
        return None
    selection = base_config.training_stages
    if uses_later_training_stages(selection):
        active = frozenset(selected_training_stages(selection)) - skipped_stages
        if not active:
            raise RuntimeError("Consume an all-stage empty skip before scheduling worlds.")
        return BalancedStageWaveScheduler(
            selection=selection, rollout_steps=base_config.rollout_steps, active_stages=active,
            stage_step_weights=multiworld_config.stage_step_weights if selection == "s0_s5" else "1:1:1:1:1:1",
            ratio_tolerance=multiworld_config.stage_ratio_tolerance,
            max_rollout_multiplier=multiworld_config.stage_max_rollout_multiplier,
        )
    if skipped_stages:
        if skipped_stages == {"S0"}:
            selection = "s1"
        elif skipped_stages == {"S1"}:
            selection = "s0"
        else:
            raise RuntimeError("Consume an all-stage empty skip before scheduling worlds.")
    return StageWaveScheduler(
        selection=selection, rollout_steps=base_config.rollout_steps,
        s0_step_fraction=multiworld_config.s0_step_fraction,
        ratio_tolerance=multiworld_config.stage_ratio_tolerance,
        max_rollout_multiplier=multiworld_config.stage_max_rollout_multiplier,
    )


def _stage_success_skip_controller(base_config: ActPpoGazeboConfig, multiworld_config: MultiWorldConfig):
    if not multiworld_config.stage_success_skip_enabled:
        return None
    if base_config.training_stages == "full":
        raise ValueError("Stage success skipping requires stage training.")
    stages = _resolved_stage_config(base_config)
    if not stages.enable_later_stage_context and (stages.process_reward is None or stages.process_reward.schema_version < 2):
        raise ValueError(
            "Stage success skipping requires process reward schema >= 2 with binary stage context; "
            "use the current STAGE_CONFIG or disable stage success skipping for legacy configurations."
        )
    scheduler = _stage_wave_scheduler(base_config, multiworld_config)
    selection = (base_config.training_stages if uses_later_training_stages(base_config.training_stages) else
                 "s0" if scheduler.s0_step_fraction == 1 else
                 "s1" if scheduler.s0_step_fraction == 0 else "s0_s1")
    return StageSuccessSkipController(selection, threshold=multiworld_config.stage_success_skip_threshold)


def _sync_stage_success_skip(trainer: Any, controller: StageSuccessSkipController) -> None:
    trainer.collection_state = {
        **getattr(trainer, "collection_state", {}), "stage_success_skip": controller.state_dict(),
    }


def _multiworld_execution_signature(base_config: ActPpoGazeboConfig, reward: Any,
                                   multiworld_config: MultiWorldConfig) -> dict[str, Any]:
    signature = _execution_signature(base_config, reward, world_count=multiworld_config.world_count)
    scheduler = _stage_wave_scheduler(base_config, multiworld_config)
    if scheduler is not None:
        # The legacy per-episode draw is unused by homogeneous stage waves.
        signature["stage_training"].pop("s0_episode_probability", None)
        signature["stage_training"]["wave_sampling"] = {
            **scheduler.contract(), "world_count": multiworld_config.world_count,
            "finish_current_batch_after_episode_minimum": True,
        }
    controller = _stage_success_skip_controller(base_config, multiworld_config)
    if controller is not None:
        signature["stage_training"]["success_skip"] = {
            "schema": "valid_episode_rate_skip_update_and_one_sampling_batch_v1",
            "threshold": multiworld_config.stage_success_skip_threshold,
            "excluded_objectives": ["actor", "critic", "exploration", "auxiliary"],
            "next_batch_steps": "fill_rollout_steps_with_remaining_stages",
            "all_stages_skipped": "consume_one_empty_batch_without_worlds_or_update",
        }
    return signature


def run_act_ppo_multiworld_gazebo(
    base_config: ActPpoGazeboConfig,
    multiworld_config: MultiWorldConfig,
) -> dict[str, Any]:
    global _CURRENT_WAVE_INDEX
    _validate_target_lifecycle(base_config.final_target_absent_s)
    reward = _resolved_reward(base_config)
    stage_config = _resolved_stage_config(base_config)
    stage_scheduler = _stage_wave_scheduler(base_config, multiworld_config)
    stage_success_skip = _stage_success_skip_controller(base_config, multiworld_config)
    execution_signature = _multiworld_execution_signature(base_config, reward, multiworld_config)
    _set_seed(base_config.seed)
    output_directory = (
        _default_output_directory(base_config.act_checkpoint)
        if base_config.output_directory is None
        else base_config.output_directory
    ).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    discard_stats = DiscardedStepStats(output_directory, world_indices=range(multiworld_config.world_count))

    preflight_cleanup_report = PreflightCleanupReport(
        enabled=False,
        target_ros_domain_ids=tuple(
            multiworld_config.domain_id(index)
            for index in range(multiworld_config.world_count)
        ),
        target_gz_partitions=tuple(
            multiworld_config.partition(index)
            for index in range(multiworld_config.world_count)
        ),
        matched_processes=(),
        terminated_processes=(),
        killed_processes=(),
    )
    if multiworld_config.preflight_cleanup_enabled:
        print(
            "[PREFLIGHT CLEANUP] reserved ROS domains="
            + ",".join(
                str(value)
                for value in preflight_cleanup_report.target_ros_domain_ids
            )
            + "; Gazebo partitions="
            + ",".join(preflight_cleanup_report.target_gz_partitions)
        )
        preflight_cleanup_report = cleanup_stale_multiworld_processes(
            target_ros_domain_ids=preflight_cleanup_report.target_ros_domain_ids,
            target_gz_partitions=preflight_cleanup_report.target_gz_partitions,
            timeout_s=multiworld_config.preflight_cleanup_timeout_s,
        )
        print(
            "[PREFLIGHT CLEANUP] matched="
            f"{len(preflight_cleanup_report.matched_processes)} "
            "terminated="
            f"{len(preflight_cleanup_report.terminated_processes)} "
            "killed="
            f"{len(preflight_cleanup_report.killed_processes)}"
        )
    else:
        print("[PREFLIGHT CLEANUP] disabled by configuration")
    _write_json(
        output_directory / "preflight_cleanup.json",
        preflight_cleanup_report.to_json_dict(),
    )

    tensorboard = DirectActPpoTensorBoard.disabled()
    clients: list[GazeboWorkerClient] = []
    rollout: ActPpoRolloutBuffer | None = None
    update_in_progress = False
    try:
        if base_config.tensorboard_enabled:
            tensorboard = DirectActPpoTensorBoard.start(
                output_directory / "tensorboard",
                preferred_port=base_config.tensorboard_port,
                auto_open=base_config.tensorboard_auto_open,
            )

        device = torch.device(base_config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")

        loaded = load_act_checkpoint(base_config.act_checkpoint, device=device)
        _configure_act_replanning(loaded, base_config.act_max_action_steps)
        policy = loaded.policy
        action_metadata = validate_act_temporal_config(loaded.config)
        if action_metadata["temporal_ensemble_coeff"] is not None:
            execution_signature.update(action_metadata)

        policy.eval()
        policy.reset()

        adapter = ActPolicyAdapter(
            policy,
            device=loaded.device,
            max_joint_delta_rad=_DISABLED_DELTA_SENTINEL_RAD,
        )
        ppo_config = _act_ppo_config(base_config, int(loaded.config.n_action_steps))
        critic = ActPpoValueNetwork(ppo_config).to(device)
        trainer = ActPpoTrainer(policy, critic, config=ppo_config,
                                **({"execution_signature": execution_signature} if reward else {}))
        resume_payload = None
        if base_config.resume_checkpoint is not None:
            resume_payload = trainer.load_checkpoint(
                base_config.resume_checkpoint,
                map_location=device,
            )
        curriculum = _make_curriculum(base_config, resume_payload)
        _sync_curriculum_state(trainer, curriculum)
        attempt_counter = [_initial_attempt_count(base_config, curriculum, resume_payload)]
        _sync_stage_attempt_count(trainer, base_config, attempt_counter[0])
        if stage_success_skip is not None:
            if resume_payload is not None:
                saved_skip = resume_payload.get("extra", {}).get("collection_state", {}).get("stage_success_skip")
                if saved_skip is None:
                    raise ValueError("Stage-skip resume is missing collection state; use an ACT weights-only start.")
                stage_success_skip.load_state_dict(saved_skip)
            _sync_stage_success_skip(trainer, stage_success_skip)
        initial_fingerprint = model_state_sha256(policy)

        if reward is not None:
            _write_json(output_directory / "resolved_reward.json", reward.to_dict())
        if stage_config is not None:
            _write_json(output_directory / "resolved_stages.json", stage_config.to_dict())
        _write_json(
            output_directory / "config.json",
            {
                "gazebo": base_config.to_json_dict(),
                "multiworld": multiworld_config.to_json_dict(),
                "preflight_cleanup": preflight_cleanup_report.to_json_dict(),
                "ppo": asdict(ppo_config),
                "resolved_reward": None if reward is None else reward.to_dict(),
                "resolved_stages": None if stage_config is None else stage_config.to_dict(),
                "execution_signature": execution_signature,
                "checkpoint_act_config": {
                    **action_metadata,
                    "action_bounds_lower": loaded.config.action_bounds_lower,
                    "action_bounds_upper": loaded.config.action_bounds_upper,
                    "action_bound_softness": loaded.config.action_bound_softness,
                    "optimizer_lr": loaded.config.optimizer_lr,
                    "optimizer_lr_backbone": loaded.config.optimizer_lr_backbone,
                    "optimizer_weight_decay": loaded.config.optimizer_weight_decay,
                    "use_vae": loaded.config.use_vae,
                    "vision_backbone": loaded.config.vision_backbone,
                },
                "rollout_semantics": {
                    "threshold_is_minimum": True,
                    "complete_episode_is_atomic": True,
                    "all_available_worlds_start_each_wave": stage_scheduler is None,
                    "world_count_is_concurrency_cap": stage_scheduler is not None,
                    "overshoot_is_consumed_in_one_update": stage_success_skip is None,
                    "successful_stage_exclusion_enabled": stage_success_skip is not None,
                    "final_partial_rollout_is_discarded": stage_scheduler is None,
                    "finish_stage_batch_after_episode_minimum": stage_scheduler is not None,
                    **({"stage_wave_sampling": stage_scheduler.contract()} if stage_scheduler is not None else {}),
                    "world_rebuilt_after_episode_recorded": True,
                    "world_stopped_after_episode_recorded": True,
                    "world_started_only_before_collection_wave": True,
                    "gazebo_worlds_running_during_ppo": False,
                    "in_episode_target_deletion": False,
                    "step_records_include_images": False,
                },
                "discarded_step_statistics": "discarded_steps.json",
                "deployment_artifact": "standalone ACT pretrained_model",
                "critic_is_training_only": True,
                "exploration_is_training_only": True,
                "max_joint_delta_limit_enabled": False,
                "absolute_joint_limits_enabled": True,
                "target_randomness": ("seeded fixed central source sector for stage initialization" if stage_config is not None else
                                      "seeded reach curriculum" if curriculum is not None
                                      else "independent target_cube_manager SystemRandom per world"),
            },
        )

        slots: list[_WorldSlot] = []
        for world_index in range(multiworld_config.world_count):
            worker_config = GazeboWorkerConfig(
                max_episode_steps=(base_config.max_steps_per_episode if stage_config is None else
                                   max(_episode_step_limit(base_config, mode) for mode in ("idle_empty", "pick_place"))),
                reward_profile=base_config.reward_profile,
                reward_config=None if reward is None else reward.to_dict(),
                n_action_steps=int(loaded.config.n_action_steps),
                world_index=world_index,
                ros_domain_id=multiworld_config.domain_id(world_index),
                gz_partition=multiworld_config.partition(world_index),
                gazebo_gui=multiworld_config.gazebo_gui,
                project_root=Path.cwd(),
                output_directory=output_directory,
                startup_feedback_timeout_s=base_config.startup_feedback_timeout_s,
                startup_max_attempts=multiworld_config.world_startup_max_attempts,
                restart_max_attempts=_worker_restart_attempt_limit(
                    multiworld_config
                ),
            )
            client = GazeboWorkerClient(worker_config)
            clients.append(client)
            slots.append(_WorldSlot(client=client))

        if stage_scheduler is None:
            _start_clients_concurrently(slots, output_directory=output_directory)
            if not any(slot.available for slot in slots):
                raise RuntimeError(
                    "No Gazebo world reached ready state during initial startup."
                )

        rollout = ActPpoRolloutBuffer(ppo_config)
        episodes: list[dict[str, Any]] = []
        failed_episodes: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        stage_skipped_update_count = 0
        success_gated_transition_count = 0
        total_collected_steps = 0
        discarded_environment_steps = 0
        discarded_final_partial_transitions = 0
        completed_episode_count = 0
        rollout_index = 0
        starting_new_rollout = False
        wave_index = 0
        training_started_monotonic = time.monotonic()

        _terminal_print_block(
            "ACT PPO MULTI-WORLD FINETUNE",
            [
                f"worlds             : {multiworld_config.world_count}",
                f"episodes target    : {base_config.episode_count}",
                (f"max steps/episode  : {base_config.max_steps_per_episode}" if stage_config is None else
                 "stage sim budgets : " + "; ".join(f"{stage} {stage_config.stage_budget_s(stage):g}s"
                    for stage in selected_training_stages(base_config.training_stages))),
                f"training stages    : {base_config.training_stages}",
                *([] if stage_scheduler is None else [
                    f"stage wave sampling: {stage_scheduler.contract()}",
                    "episode probability: unused; stage selection balances actual rollout steps",
                ]),
                f"rollout target     : {base_config.rollout_steps}",
                f"PPO epochs         : {ppo_config.ppo_epochs}",
                f"ACT action queue   : {loaded.config.n_action_steps} steps",
                f"Reach curriculum   : {curriculum.diagnostics() if curriculum is not None else 'disabled'}",
                f"policy KL limit    : {ppo_config.target_kl}",
                ("KL rejection       : restore epoch; at most "
                 f"{ppo_config.kl_backoff_max_retries} smaller-step retries"),
                f"minibatch size     : {ppo_config.minibatch_size}",
                f"device             : {device}",
                f"checkpoint every   : {base_config.checkpoint_every_updates} updates",
                f"seed               : {base_config.seed}",
                f"pretrained         : {base_config.act_checkpoint}",
                f"resume checkpoint  : {base_config.resume_checkpoint or 'none'}",
                "world lifecycle    : record every step, close worlds, PPO if ready, then start worlds for collection",
                f"output directory   : {output_directory}",
                f"TensorBoard        : {tensorboard.url or 'disabled'}",
                "world mapping      : "
                + ", ".join(
                    f"world{slot.world_index}=domain"
                    f"{slot.client.config.ros_domain_id}/"
                    f"{slot.client.config.gz_partition}"
                    for slot in sorted(slots, key=lambda item: item.world_index)
                ),
            ],
        )

        while (not _episode_minimum_reached(completed_episode_count, base_config.episode_count)
               or (stage_scheduler is not None and len(rollout) > 0)):
            if stage_success_skip is not None and not len(rollout):
                if not stage_success_skip.active_stages:
                    empty = stage_success_skip.consume_empty_batch().to_dict()
                    _sync_stage_success_skip(trainer, stage_success_skip)
                    _append_jsonl(output_directory / "metrics.jsonl", {
                        "type": "stage_sampling_skipped", "update_index": trainer.update_index, **empty,
                    })
                    tensorboard.log_stage_success_skip(empty)
                    print("[STAGE SAMPLING SKIP] All configured stages skipped once; no worlds or PPO started. Resume sampling all stages now.", flush=True)
                stage_scheduler = _stage_wave_scheduler(
                    base_config, multiworld_config, stage_success_skip.current_skipped_stages)
            # Check the balance limit while worlds are closed, before starting
            # another wave. No samples cross a policy update or are duplicated.
            wave_plan = None
            wave_training_stage = None
            if stage_scheduler is not None:
                maximum_worlds = sum(not slot.quarantined for slot in slots)
                if maximum_worlds == 0:
                    raise RuntimeError("All Gazebo worlds are quarantined before the stage batch is ready.")
                wave_plan = stage_scheduler.plan_wave(
                    maximum_worlds,
                    s0_episode_steps=math.ceil(stage_config.s0_budget_s * base_config.policy_hz),
                    s1_episode_steps=math.ceil(stage_config.s1_budget_s * base_config.policy_hz),
                    **({"stage_episode_steps": {stage: math.ceil(stage_config.stage_budget_s(stage) * base_config.policy_hz)
                                                for stage in selected_training_stages(base_config.training_stages)}}
                       if isinstance(stage_scheduler, BalancedStageWaveScheduler) else {}),
                )
                wave_training_stage = wave_plan["stage"]
                available_slots = _prepare_stage_worlds_for_wave(
                    slots, world_count=wave_plan["world_count"], output_directory=output_directory,
                    multiworld_config=multiworld_config, starting_new_rollout=starting_new_rollout,
                    selection_offset=attempt_counter[0],
                )
            else:
                _prepare_worlds_for_wave(
                    slots, output_directory=output_directory,
                    multiworld_config=multiworld_config,
                    starting_new_rollout=starting_new_rollout,
                )
                available_slots = [slot for slot in slots if slot.available and not slot.quarantined]
            starting_new_rollout = False
            if not available_slots:
                raise RuntimeError(
                    "All Gazebo worlds are unavailable before the current "
                    "rollout reached its update threshold. Partial rollout "
                    "data has been left un-updated."
                )

            wave_index += 1
            _CURRENT_WAVE_INDEX = wave_index
            wave_rollout_before = len(rollout)
            pending_slots = [
                slot
                for slot in slots
                if not slot.available and not slot.quarantined and slot.startup_response is not None
            ]
            _terminal_print_block(
                f"WAVE {wave_index:05d} START",
                [
                    "active worlds      : "
                    + _terminal_world_list(available_slots),
                    "closed other worlds: "
                    + _terminal_world_list(pending_slots),
                    f"rollout            : {len(rollout)} / {base_config.rollout_steps}",
                    *([] if stage_scheduler is None else [
                        f"wave training stage: {wave_training_stage} (all selected worlds and retries)",
                        f"planned world count: {wave_plan['world_count']} / {multiworld_config.world_count}",
                        f"wave plan          : {wave_plan}",
                        f"stage batch        : {stage_scheduler.diagnostics()}",
                    ]),
                    f"PPO update         : {trainer.update_index}",
                    f"episodes complete  : {completed_episode_count}",
                    f"episode failures   : {len(failed_episodes)}",
                ],
            )

            policy.reset()
            try:
                successes, wave_failures, discarded_steps = _run_wave(
                    available_slots,
                    adapter=adapter,
                    policy=policy,
                    critic=critic,
                    trainer=trainer,
                    device=device,
                    loaded_config=loaded.config,
                    base_config=base_config,
                    multiworld_config=multiworld_config,
                    ppo_config=ppo_config,
                    output_directory=output_directory,
                    attempt_counter=attempt_counter,
                    discard_stats=discard_stats,
                    tensorboard=tensorboard,
                    **({"wave_training_stage": wave_training_stage} if stage_scheduler is not None else {}),
                    **({"curriculum": curriculum} if curriculum is not None else {}),
                )
            except _WorldShutdownError as error:
                # A failed world's cleanup must not lose other worlds' already
                # completed trajectories. Save them before aborting; never PPO.
                _complete_episode_wave(
                    error.completed_sessions, rollout=rollout,
                    base_config=base_config, output_directory=output_directory,
                    first_episode_index=completed_episode_count,
                    tensorboard=tensorboard,
                )
                raise
            finally:
                # Scene attempts consumed by initialization failures and
                # aborted waves must not be replayed after a saved update.
                _sync_stage_attempt_count(trainer, base_config, attempt_counter[0])
            failed_episodes.extend(wave_failures)
            discarded_environment_steps += discarded_steps

            records = _complete_episode_wave(
                successes,
                rollout=rollout,
                base_config=base_config,
                output_directory=output_directory,
                first_episode_index=completed_episode_count,
                tensorboard=tensorboard,
            )
            total_collected_steps += sum(record["recorded_transition_count"] for record in records)
            completed_episode_count += len(records)
            episodes.extend(records)
            if stage_scheduler is not None:
                for record in records:
                    if record.get("training_stage") != wave_training_stage:
                        raise RuntimeError("A completed episode disagrees with its wave's training stage.")
                    stage_scheduler.observe(record["training_stage"], record["recorded_transition_count"])
                    if stage_success_skip is not None:
                        stage_success_skip.observe(record)
                if stage_scheduler.total_steps != len(rollout):
                    raise RuntimeError("Stage step accounting disagrees with the PPO rollout.")
                _append_jsonl(output_directory / "metrics.jsonl", {
                    "type": "stage_wave_sampling", "wave_index": wave_index,
                    "wave_training_stage": wave_training_stage,
                    "wave_plan": wave_plan,
                    "active_world_count": len(available_slots),
                    "next_update_index": trainer.update_index + 1,
                    **stage_scheduler.diagnostics(),
                })
            if curriculum is not None:
                for record in sorted(records, key=lambda item: item["episode_attempt_index"]):
                    curriculum.observe(record)
                _sync_curriculum_state(trainer, curriculum)
                _append_jsonl(output_directory / "metrics.jsonl",
                    {"type": "curriculum", "wave_index": wave_index, **curriculum.diagnostics()})

            failed_worlds = sorted(
                {
                    int(record["world_index"])
                    for record in wave_failures
                }
            )
            _terminal_print_block(
                f"WAVE {wave_index:05d} END",
                [
                    "completed worlds   : "
                    + _terminal_world_list(
                        [session.slot for session in successes]
                    ),
                    (
                        "failed worlds      : "
                        + (
                            " ".join(f"world{index}" for index in failed_worlds)
                            if failed_worlds
                            else "none"
                        )
                    ),
                    (
                        "accepted this wave : "
                        f"+{len(rollout) - wave_rollout_before} transitions"
                    ),
                    f"discarded this wave: {discarded_steps} environment steps",
                    (
                        "rollout            : "
                        f"{wave_rollout_before} -> {len(rollout)} / "
                        f"{base_config.rollout_steps}"
                    ),
                ],
            )
            print(
                "[TRAIN] "
                f"wave={wave_index} | "
                f"update={trainer.update_index} | "
                f"rollout={len(rollout)}/{base_config.rollout_steps} | "
                f"completed={completed_episode_count} | "
                f"failed={len(failed_episodes)} | "
                f"wasted_steps={discard_stats.snapshot()['discarded_environment_steps']} | "
                f"crash_wasted_steps={discard_stats.snapshot()['gazebo_crash_discarded_environment_steps']} | "
                f"wall={_terminal_elapsed(training_started_monotonic)}",
                flush=True,
            )
            if stage_scheduler is not None:
                balance = stage_scheduler.diagnostics()
                print(
                    f"[STAGE BATCH] {_stage_sampling_counts(balance)} | "
                    f"{_stage_sampling_fractions(balance)} | "
                    f"PPO ready={stage_scheduler.ready}", flush=True,
                )

            if (stage_scheduler.ready if stage_scheduler is not None else
                    _rollout_threshold_reached(len(rollout), base_config.rollout_steps)):
                _terminal_print_block(
                    "ROLLOUT READY",
                    [
                        f"collected      : {len(rollout)} transitions",
                        f"target         : {base_config.rollout_steps}",
                        f"next PPO update: {trainer.update_index + 1}",
                    ],
                    strong=True,
                )
                rollout_index += 1
                update_in_progress = True
                payload = _perform_update(
                    rollout,
                    trainer=trainer,
                    output_directory=output_directory,
                    base_config=base_config,
                    multiworld_config=multiworld_config,
                    tensorboard=tensorboard,
                    rollout_index=rollout_index,
                    **({"stage_scheduler": stage_scheduler} if stage_scheduler is not None else {}),
                    **({"stage_success_skip": stage_success_skip} if stage_success_skip is not None else {}),
                )
                if payload.get("type") == "stage_update_skipped":
                    stage_skipped_update_count += 1
                else:
                    updates.append(payload)
                if stage_success_skip is not None:
                    success_gated_transition_count += payload["stage_success_skip"]["excluded_transition_count"]
                consumed_rollout_transitions = len(rollout)
                rollout.clear()
                if stage_scheduler is not None:
                    stage_scheduler.reset()
                update_in_progress = False
                print(
                    "[ROLLOUT] "
                    f"collected {consumed_rollout_transitions}, "
                    f"used {payload.get('actual_rollout_transition_count', consumed_rollout_transitions)} transitions "
                    f"for PPO; update index {trainer.update_index}; "
                    f"buffer cleared {consumed_rollout_transitions} -> 0",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for slot in slots:
                    slot.quarantined = False
                    if stage_scheduler is not None:
                        # A partial first wave may not select this slot until
                        # later in the batch. Its retry budget still starts fresh.
                        slot.consecutive_episode_failures = 0
                starting_new_rollout = (
                    not _episode_minimum_reached(
                        completed_episode_count, base_config.episode_count
                    )
                )

            if (_episode_minimum_reached(completed_episode_count, base_config.episode_count)
                    and (stage_scheduler is None or len(rollout) == 0)):
                break

        if len(rollout) > 0:
            discarded_final_partial_transitions = len(rollout)
            payload = {
                "type": "final_partial_rollout_discarded",
                "transition_count": len(rollout),
                "configured_rollout_step_threshold": base_config.rollout_steps,
                "completed_episode_count": completed_episode_count,
            }
            _append_jsonl(output_directory / "metrics.jsonl", payload)
            counters = _discard_counters(discard_stats.record_final_partial(len(rollout)))
            rollout.clear()
            _append_jsonl(output_directory / "metrics.jsonl", {"type": "discarded_steps", **counters})
            tensorboard.log_discarded_steps(counters, event_index=int(counters["failure_episode_count"]) + 1)

        final_fingerprint = model_state_sha256(policy)
        if final_fingerprint == initial_fingerprint and not (stage_skipped_update_count and not updates):
            raise RuntimeError(
                "Direct ACT PPO completed without a full PPO update changing "
                "ACT parameters. The final partial rollout is intentionally "
                "not used for training, so no standalone finetuned ACT was exported."
            )

        final_act, final_training_state = _export_final(
            trainer,
            output_directory=output_directory,
            base_config=base_config,
            multiworld_config=multiworld_config,
            metrics=updates[-1] if updates else None,
        )
        reloaded = load_act_checkpoint(final_act, device="cpu")
        if (model_state_sha256(reloaded.policy) != final_fingerprint
                or actor_forward_contract(reloaded.policy.config) != actor_forward_contract(policy.config)):
            raise RuntimeError("Reloaded standalone ACT weights or forward configuration do not match trained ACT.")

        summary = {
            "output_directory": str(output_directory),
            "final_act_checkpoint": str(final_act),
            "final_training_state": str(final_training_state),
            "episode_count": completed_episode_count,
            "valid_episode_count": completed_episode_count,
            "pick_place_episode_count": sum(e.get("task_mode") == "pick_place" for e in episodes),
            "task_success_count": sum(bool(e.get("task_success")) for e in episodes),
            "training_stage_summary": _training_stage_summary(episodes),
            "requested_minimum_episode_count": base_config.episode_count,
            "episode_attempt_count": attempt_counter[0],
            "failed_episode_attempt_count": len(failed_episodes),
            "maximum_consecutive_episode_failures_per_world": (
                base_config.max_consecutive_episode_failures
            ),
            "world_count": multiworld_config.world_count,
            "gazebo_gui": multiworld_config.gazebo_gui,
            "world_startup_max_attempts": (
                multiworld_config.world_startup_max_attempts
            ),
            "world_restart_max_attempts": (
                multiworld_config.world_restart_max_attempts
            ),
            "world_deferred_restart_max_attempts": (
                multiworld_config.world_deferred_restart_max_attempts
            ),
            "preflight_cleanup": preflight_cleanup_report.to_json_dict(),
            "discarded_environment_steps": discarded_environment_steps,
            "discarded_step_statistics": _discard_counters(discard_stats.snapshot()),
            "discarded_final_partial_transitions": (
                discarded_final_partial_transitions
            ),
            "total_collected_steps": total_collected_steps,
            "ppo_update_count": len(updates),
            "stage_skipped_update_count": stage_skipped_update_count,
            "success_gated_transition_count": success_gated_transition_count,
            **({"stage_success_skip_state": stage_success_skip.state_dict()} if stage_success_skip is not None else {}),
            "checkpoint_every_updates": base_config.checkpoint_every_updates,
            "saved_update_checkpoint_count": sum(
                bool(update.get("checkpoint_saved", False)) for update in updates
            ),
            "act_state_sha256_before": initial_fingerprint,
            "act_state_sha256_after": final_fingerprint,
            "act_parameters_changed": final_fingerprint != initial_fingerprint,
            "standalone_act_reload_verified": True,
            "critic_exported_with_deployment_act": False,
            "exploration_exported_with_deployment_act": False,
            **action_metadata,
            "tensorboard_enabled": base_config.tensorboard_enabled,
            "tensorboard_log_directory": tensorboard.log_directory_string,
            "tensorboard_url": tensorboard.url,
            "max_joint_delta_limit_enabled": False,
            "absolute_joint_limits_enabled": True,
            "worlds": [
                {
                    "world_index": slot.world_index,
                    "ros_domain_id": slot.client.config.ros_domain_id,
                    "gz_partition": slot.client.config.gz_partition,
                    "successful_episode_count": slot.successful_episode_count,
                    "valid_episode_count": slot.successful_episode_count,
                    "task_success_count": slot.task_success_count,
                    "failed_episode_count": slot.failed_episode_count,
                    "restart_count": slot.restart_count,
                    "episode_restart_count": slot.episode_restart_count,
                    "gazebo_log": str(slot.client.config.log_path),
                }
                for slot in slots
            ],
            "episodes": episodes,
            "failed_episodes": failed_episodes,
            "updates": updates,
        }
        _write_json(output_directory / "summary.json", summary)
        return summary
    except BaseException:
        try:
            _record_aborted_rollout(
                rollout, discard_stats=discard_stats, output_directory=output_directory,
                tensorboard=tensorboard, update_in_progress=update_in_progress,
            )
        except Exception as accounting_error:
            # Preserve the original failure (including KeyboardInterrupt),
            # especially when storage itself failed while recording an episode.
            print(f"[DISCARDED STEPS WARNING] Could not record pending rollout: {accounting_error}", flush=True)
        raise
    finally:
        for client in clients:
            try:
                client.shutdown()
            except BaseException as error:
                print(
                    "[WORLD SHUTDOWN WARNING] "
                    f"world={client.world_index}: {type(error).__name__}: {error}"
                )
        tensorboard.close()


def main() -> None:
    args = _parser().parse_args()
    base_config = _build_base_config(args)
    multiworld_config = MultiWorldConfig(
        world_count=args.world_count,
        base_ros_domain_id=args.base_ros_domain_id,
        gz_partition_prefix=args.gz_partition_prefix,
        gazebo_gui=args.gazebo_gui,
        world_startup_max_attempts=args.world_startup_max_attempts,
        world_restart_max_attempts=args.world_restart_max_attempts,
        world_deferred_restart_max_attempts=(
            args.world_deferred_restart_max_attempts
        ),
        preflight_cleanup_enabled=args.preflight_cleanup,
        preflight_cleanup_timeout_s=args.preflight_cleanup_timeout_s,
        s0_step_fraction=args.s0_step_fraction,
        stage_step_weights=args.stage_step_weights,
        stage_ratio_tolerance=args.stage_ratio_tolerance,
        stage_max_rollout_multiplier=args.stage_max_rollout_multiplier,
        stage_success_skip_enabled=args.stage_success_skip,
        stage_success_skip_threshold=args.stage_success_skip_threshold,
    )
    summary = run_act_ppo_multiworld_gazebo(base_config, multiworld_config)
    print("Direct ACT PPO multi-world Gazebo training completed.")
    for key in (
        "output_directory",
        "final_act_checkpoint",
        "final_training_state",
        "world_count",
        "episode_count",
        "requested_minimum_episode_count",
        "episode_attempt_count",
        "failed_episode_attempt_count",
        "total_collected_steps",
        "discarded_final_partial_transitions",
        "ppo_update_count",
        "checkpoint_every_updates",
        "saved_update_checkpoint_count",
        "tensorboard_log_directory",
        "tensorboard_url",
        "act_parameters_changed",
        "standalone_act_reload_verified",
    ):
        print(f"{key}={summary[key]}")


if __name__ == "__main__":
    main()
