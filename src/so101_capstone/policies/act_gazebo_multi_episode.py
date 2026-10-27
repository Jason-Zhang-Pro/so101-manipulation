"""Execute multiple ACT-controlled episodes in one Gazebo runtime.

The checkpoint, ACT policy adapter, and ROS/Gazebo runtime are initialized
once. Each episode still performs its own formal prepare/reset lifecycle.
Scored scenes manage target creation and deletion before and during episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from so101_capstone.environments import (
    ReachEnvironmentConfig,
    SimulationRuntimeConfig,
    create_ros_simulation_runtime,
)

from .act_checkpoint_loader import load_act_checkpoint
from .act_gazebo_episode import (
    _EvaluationSceneRuntimeProxy,
    _evaluation_metadata,
    _evaluation_reward,
    _evaluation_reward_path,
    _load_evaluation_scenes,
    _validate_evaluation_scene,
)
from .act_policy_adapter import ActPolicyAdapter
from .policy_episode_npz import PolicyEpisodeNpzSink
from .delayed_random_target import (
    DelayedRandomTargetEpisodeController,
    DelayedRandomTargetRuntimeProxy,
    RosTargetLifecycleBackend,
)
from .policy_multi_episode_runner import (
    PolicyMultiEpisodeResult,
    PolicyMultiEpisodeRunner,
)


@dataclass(frozen=True)
class ActGazeboMultiEpisodeResult:
    """Summary from one multi-episode ACT/Gazebo run."""

    checkpoint_path: Path
    device: torch.device
    run: PolicyMultiEpisodeResult
    output_directory: Path | None
    episode_npz_paths: tuple[Path, ...]
    target_manifest_paths: tuple[Path, ...]
    evaluation_metadata: dict[str, Any] | None = None


def _joint_delta_metadata(
    value: float | Sequence[float] | None,
) -> float | list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    return [
        float(item)
        for item in value
    ]


def _environment_config(
    max_episode_steps: int,
) -> ReachEnvironmentConfig:
    """Build the internal environment horizon for this rollout."""

    return ReachEnvironmentConfig(
        max_episode_steps=max_episode_steps,
    )


def _runtime_context(
    *,
    runtime_factory: Callable[..., Any] | None,
    runtime_config: SimulationRuntimeConfig,
    environment_config: ReachEnvironmentConfig,
    environment_randomization_enabled: bool = False,
    environment_randomization_config_path: Path | None = None,
    environment_randomization_seed: int = 0,
    reward_profile: str = "legacy",
    reward_config: Any = None,
    task_mode: str = "pick_place",
    observe_after_success_s: float = 0.0,
) -> Any:
    """Create the real runtime while preserving injected test factories."""

    if runtime_factory is None or reward_profile != "legacy":
        runtime_kwargs = {
            "runtime_config": runtime_config,
            "environment_config": environment_config,
        }
        if environment_randomization_enabled:
            runtime_kwargs.update(
                {
                    "environment_randomization_enabled": True,
                    "environment_randomization_config_path": (
                        environment_randomization_config_path
                    ),
                    "environment_randomization_seed": (
                        environment_randomization_seed
                    ),
                }
            )
        if reward_profile != "legacy":
            runtime_kwargs.update(reward_profile=reward_profile, reward_config=reward_config,
                                  task_mode=task_mode, observe_after_success_s=observe_after_success_s,
                                  environment_randomization_enabled=environment_randomization_enabled)
        factory = create_ros_simulation_runtime if runtime_factory is None else runtime_factory
        return factory(**runtime_kwargs)

    return runtime_factory(
        runtime_config=runtime_config,
    )


def run_act_gazebo_multi_episode(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    max_joint_delta_rad: float | Sequence[float] | None = None,
    episode_count: int,
    max_steps: int,
    environment_max_episode_steps: int = 600,
    startup_feedback_timeout_s: float = 30.0,
    output_directory: str | Path | None = None,
    checkpoint_loader: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    multi_episode_runner_factory: (
        Callable[..., Any] | None
    ) = None,
    npz_sink_factory: Callable[..., Any] | None = None,
    spawn_random_target_after_s: float | None = None,
    policy_hz: float = 30.0,
    target_backend_factory: Callable[..., Any] | None = None,
    target_controller_factory: Callable[..., Any] | None = None,
    delayed_runtime_factory: Callable[..., Any] | None = None,
    environment_randomization_enabled: bool = False,
    environment_randomization_config_path: str | Path | None = None,
    environment_randomization_seed: int = 0,
    retain_episode_transitions: bool = False,
    reward_profile: str = "legacy",
    reward_config: str | Path | None = None,
    task_mode: str = "pick_place",
    observe_after_success_s: float = 0.0,
    evaluation_scenes_path: str | Path | None = None,
) -> ActGazeboMultiEpisodeResult:
    """Load ACT once and execute several episodes in one runtime.

    Production evaluation returns lightweight per-episode summaries. Set
    ``retain_episode_transitions=True`` to request the legacy full history,
    including camera frames. NPZ archives have the same image-free schema
    in either mode. Injected runner factories keep their existing interface.
    """

    resolved_reward_path = (_evaluation_reward_path(checkpoint_path, reward_config)
                            if reward_profile == "pick_place_v2" else reward_config)
    reward = _evaluation_reward(reward_profile, resolved_reward_path, task_mode, observe_after_success_s)
    if evaluation_scenes_path is not None and reward is None:
        raise ValueError("Fixed evaluation scenes require pick_place_v2.")
    if reward is not None and reward.control_hz != policy_hz:
        raise ValueError("Evaluation policy_hz must match reward control_hz.")
    if evaluation_scenes_path is not None and spawn_random_target_after_s is not None:
        raise ValueError("Fixed evaluation scenes cannot be combined with random-target scheduling.")
    scenes = (() if reward is None else
              _load_evaluation_scenes(evaluation_scenes_path, episode_count=episode_count)
              if evaluation_scenes_path is not None else
              tuple(_validate_evaluation_scene({"name": f"episode_{index:06d}", "task_mode": task_mode,
                    "target_delay_s": 0.0 if spawn_random_target_after_s is None else spawn_random_target_after_s,
                    "seed": environment_randomization_seed + index}, require_fixed_target=False)
                    for index in range(episode_count)))
    if environment_randomization_enabled and any(scene["task_mode"] == "idle_completed" for scene in scenes):
        raise ValueError("idle_completed evaluation requires environment randomization disabled.")
    resolved_checkpoint_loader = (
        load_act_checkpoint
        if checkpoint_loader is None
        else checkpoint_loader
    )
    resolved_adapter_factory = (
        ActPolicyAdapter
        if adapter_factory is None
        else adapter_factory
    )
    resolved_runner_factory = (
        PolicyMultiEpisodeRunner
        if multi_episode_runner_factory is None
        else multi_episode_runner_factory
    )
    resolved_npz_sink_factory = (
        PolicyEpisodeNpzSink
        if npz_sink_factory is None
        else npz_sink_factory
    )
    resolved_target_backend_factory = (
        RosTargetLifecycleBackend
        if target_backend_factory is None
        else target_backend_factory
    )
    resolved_target_controller_factory = (
        DelayedRandomTargetEpisodeController
        if target_controller_factory is None
        else target_controller_factory
    )
    resolved_delayed_runtime_factory = (
        DelayedRandomTargetRuntimeProxy
        if delayed_runtime_factory is None
        else delayed_runtime_factory
    )

    resolved_checkpoint = (
        Path(checkpoint_path)
        .expanduser()
        .resolve()
    )

    loaded = resolved_checkpoint_loader(
        resolved_checkpoint,
        device=device,
    )
    evaluation_metadata = _evaluation_metadata(loaded, reward_profile, reward, task_mode, observe_after_success_s)
    evaluation_metadata["reward_config_source_path"] = None if resolved_reward_path is None else str(resolved_reward_path)
    if reward is not None:
        evaluation_metadata["scenes"] = list(scenes)
        evaluation_metadata["fixed_scene_list"] = evaluation_scenes_path is not None

    adapter = resolved_adapter_factory(
        loaded.policy,
        device=loaded.device,
        max_joint_delta_rad=max_joint_delta_rad,
    )

    resolved_output_directory = (
        None
        if output_directory is None
        else (
            Path(output_directory)
            .expanduser()
            .resolve()
        )
    )

    # A recoverable failure retries the same logical episode. Keep only its
    # latest sink, not failed attempts with no written archive.
    created_sinks: dict[int, Any] = {}

    if resolved_output_directory is None:
        transition_sink_factory = None
    else:
        resolved_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        def transition_sink_factory(
            episode_index: int,
        ) -> Any:
            sink = resolved_npz_sink_factory(
                resolved_output_directory
                / f"episode_{episode_index:06d}.npz",
                metadata={
                    "checkpoint": str(
                        resolved_checkpoint
                    ),
                    "device": str(loaded.device),
                    "episode_index": episode_index,
                    "episode_count": episode_count,
                    "max_steps": max_steps,
                    "environment_max_episode_steps": environment_max_episode_steps,
                    "max_joint_delta_rad": (
                        _joint_delta_metadata(
                            max_joint_delta_rad
                        )
                    ),
                    **({"evaluation": evaluation_metadata, "evaluation_scene": scenes[episode_index]} if reward is not None else {}),
                },
            )
            created_sinks[episode_index] = sink
            return sink

    environment_config = _environment_config(
        environment_max_episode_steps
    )

    runtime_config = SimulationRuntimeConfig(
        startup_feedback_timeout_s=(
            startup_feedback_timeout_s
        )
    )

    target_manifest_paths: tuple[Path, ...] = ()
    resolved_randomization_config = (
        None
        if environment_randomization_config_path is None
        else Path(environment_randomization_config_path).expanduser().resolve()
    )

    runner_kwargs = (
        {"retain_episode_transitions": retain_episode_transitions}
        if multi_episode_runner_factory is None
        else {}
    )
    if observe_after_success_s:
        runner_kwargs.update(observe_after_success_s=observe_after_success_s, policy_hz=policy_hz)

    with _runtime_context(
        runtime_factory=runtime_factory,
        runtime_config=runtime_config,
        environment_config=environment_config,
        environment_randomization_enabled=(
            environment_randomization_enabled
        ),
        environment_randomization_config_path=(
            resolved_randomization_config
        ),
        environment_randomization_seed=environment_randomization_seed,
        reward_profile=reward_profile,
        reward_config=reward,
        task_mode=task_mode,
        observe_after_success_s=observe_after_success_s,
    ) as runtime:
        if reward is not None:
            scene_runtime = _EvaluationSceneRuntimeProxy(runtime, scenes,
                n_action_steps=int(evaluation_metadata["n_action_steps"]), policy_hz=policy_hz,
                environment_randomization_enabled=environment_randomization_enabled)
            run_result = resolved_runner_factory(scene_runtime, adapter, **runner_kwargs).run(
                episode_count=episode_count, max_steps=max_steps,
                transition_sink_factory=transition_sink_factory,
                before_episode=scene_runtime.select_episode,
            )
        elif spawn_random_target_after_s is None:
            run_result = resolved_runner_factory(
                runtime,
                adapter,
                **runner_kwargs,
            ).run(
                episode_count=episode_count,
                max_steps=max_steps,
                transition_sink_factory=(
                    transition_sink_factory
                ),
            )
        else:
            with resolved_target_backend_factory() as target_backend:
                target_controller = (
                    resolved_target_controller_factory(
                        target_backend,
                        target_delay_s=(
                            spawn_random_target_after_s
                        ),
                        policy_hz=policy_hz,
                        output_directory=(
                            resolved_output_directory
                        ),
                    )
                )
                delayed_runtime = (
                    resolved_delayed_runtime_factory(
                        runtime,
                        target_controller,
                    )
                )
                run_result = resolved_runner_factory(
                    delayed_runtime,
                    adapter,
                    **runner_kwargs,
                ).run(
                    episode_count=episode_count,
                    max_steps=max_steps,
                    transition_sink_factory=(
                        transition_sink_factory
                    ),
                )
                target_manifest_paths = tuple(
                    target_controller.manifest_paths
                )

    episode_npz_paths: tuple[Path, ...]

    if resolved_output_directory is None:
        episode_npz_paths = ()
    else:
        paths = []

        for episode_index in range(run_result.episode_count):
            sink = created_sinks[episode_index]
            if sink.written_path is None:
                raise RuntimeError(
                    "An episode NPZ sink did not finalize."
                )
            paths.append(
                Path(sink.written_path).resolve()
            )

        episode_npz_paths = tuple(paths)

    if reward is not None and resolved_output_directory is not None:
        evaluation_payload = {"configuration": evaluation_metadata,
            "episodes": [{"scene": scenes[index], "evaluation": episode.final_info.get("evaluation", {}),
                          "step_count": episode.step_count, "task_outcome": episode.final_info.get("task_outcome")}
                         for index, episode in enumerate(run_result.episodes)]}
        (resolved_output_directory / "evaluation_summary.json").write_text(
            json.dumps(evaluation_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    return ActGazeboMultiEpisodeResult(
        checkpoint_path=resolved_checkpoint,
        device=torch.device(loaded.device),
        run=run_result,
        output_directory=resolved_output_directory,
        episode_npz_paths=episode_npz_paths,
        target_manifest_paths=target_manifest_paths,
        evaluation_metadata=evaluation_metadata,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load one local ACT checkpoint and execute multiple "
            "episodes in one deterministic Gazebo runtime."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--max-joint-delta-rad",
        type=float,
        default=None,
        help="Optional per-step joint delta limit in radians; disabled by default.",
    )
    parser.add_argument(
        "--episode-count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--environment-max-episode-steps",
        type=int,
        default=600,
        help=(
            "Internal environment episode horizon. "
            "Use 18000 for 10 simulated minutes at 30 Hz. "
            "The default is 600 steps for training-compatible runs."
        ),
    )
    parser.add_argument(
        "--startup-feedback-timeout-s",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Optional directory for episode_000000.npz, "
            "episode_000001.npz, and later archives. "
            "Camera images are excluded."
        ),
    )
    parser.add_argument(
        "--spawn-random-target-after-s",
        type=float,
        default=None,
        help=(
            "Start each episode without a target, then request "
            "one random target after this many policy seconds. "
            "Use 3.0 for the current fixed-delay experiment."
        ),
    )
    parser.add_argument(
        "--policy-hz",
        type=float,
        default=30.0,
        help=(
            "Policy step frequency used to convert the target "
            "delay into an exact step count."
        ),
    )
    parser.add_argument(
        "--environment-randomization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable episode-level environment randomization. Disabled by "
            "default and adds no randomization warm-up or image processing."
        ),
    )
    parser.add_argument(
        "--environment-randomization-config",
        type=Path,
        default=None,
        help="Optional environment-randomization JSON config path.",
    )
    parser.add_argument(
        "--environment-randomization-seed",
        type=int,
        default=0,
        help="Base seed for reproducible evaluation environments.",
    )
    parser.add_argument("--reward-profile", choices=("legacy", "pick_place_v2"), default="pick_place_v2")
    parser.add_argument("--reward-config", type=Path, default=None)
    parser.add_argument("--task-mode", choices=("pick_place", "idle_empty", "idle_completed"), default="pick_place")
    parser.add_argument("--observe-after-success-s", type=float, default=0.0)
    parser.add_argument("--evaluation-scenes", type=Path, default=None,
                        help="Versioned JSON list of explicit target/idle scenes; count must equal episode-count.")
    return parser


def main() -> None:
    args = _parser().parse_args()

    result = run_act_gazebo_multi_episode(
        args.checkpoint,
        device=args.device,
        max_joint_delta_rad=(
            args.max_joint_delta_rad
        ),
        episode_count=args.episode_count,
        max_steps=args.max_steps,
        environment_max_episode_steps=(
            args.environment_max_episode_steps
        ),
        startup_feedback_timeout_s=(
            args.startup_feedback_timeout_s
        ),
        output_directory=args.output_directory,
        spawn_random_target_after_s=(
            args.spawn_random_target_after_s
        ),
        policy_hz=args.policy_hz,
        environment_randomization_enabled=(
            args.environment_randomization
        ),
        environment_randomization_config_path=(
            args.environment_randomization_config
        ),
        environment_randomization_seed=(
            args.environment_randomization_seed
        ),
        reward_profile=args.reward_profile,
        reward_config=args.reward_config,
        task_mode=args.task_mode,
        observe_after_success_s=args.observe_after_success_s,
        evaluation_scenes_path=args.evaluation_scenes,
    )

    print("ACT Gazebo multi-episode run completed.")
    print(f"checkpoint={result.checkpoint_path}")
    print(f"device={result.device}")
    print(
        f"episode_count={result.run.episode_count}"
    )
    print(
        "total_step_count="
        f"{result.run.total_step_count}"
    )
    print(
        f"total_reward={result.run.total_reward:.6f}"
    )

    if result.output_directory is not None:
        print(
            "output_directory="
            f"{result.output_directory}"
        )

    for episode_index, episode in enumerate(
        result.run.episodes
    ):
        prefix = f"episode_{episode_index:06d}"
        print(
            f"{prefix}_step_count="
            f"{episode.step_count}"
        )
        print(
            f"{prefix}_total_reward="
            f"{episode.total_reward:.6f}"
        )
        print(
            f"{prefix}_terminated="
            f"{episode.terminated}"
        )
        print(
            f"{prefix}_truncated="
            f"{episode.truncated}"
        )
        print(
            f"{prefix}_reached_max_steps="
            f"{episode.reached_max_steps}"
        )
        if "evaluation" in episode.final_info:
            print(f"{prefix}_evaluation=" + json.dumps(episode.final_info["evaluation"], sort_keys=True))

        if episode_index < len(
            result.episode_npz_paths
        ):
            print(
                f"{prefix}_npz_path="
                f"{result.episode_npz_paths[episode_index]}"
            )

        if episode_index < len(
            result.target_manifest_paths
        ):
            print(
                f"{prefix}_target_manifest_path="
                f"{result.target_manifest_paths[episode_index]}"
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
