"""Execute one ACT-controlled episode in the deterministic Gazebo runtime.

This entry point performs real policy actions and can manage a scored target
scene and an episode archive. It never updates policy weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from so101_capstone.environments import (
    ReachEnvironmentConfig,
    SimulationRuntimeConfig,
    create_ros_simulation_runtime,
)

from .act_checkpoint_loader import load_act_checkpoint
from .act_policy_adapter import ActPolicyAdapter
from .act_temporal_config import validate_act_temporal_config
from .policy_episode_npz import PolicyEpisodeNpzSink
from .policy_episode_runner import (
    PolicyEpisodeResult,
    PolicyEpisodeRunner,
)
from .so101_sim_calibration import SIM_CALIBRATION_FILE
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.environments.target_cube_control import TargetCubeControlError


def _evaluation_reward_path(checkpoint_path: str | Path, explicit_path: str | Path | None) -> Path | None:
    """Prefer an explicit score definition, then the nearest training artifact."""
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    directory = checkpoint.parent if checkpoint.is_file() else checkpoint
    for ancestor in (directory, *directory.parents):
        candidate = ancestor / "resolved_reward.json"
        if candidate.is_file():
            return candidate
    return None


def _evaluation_randomization_from_environment() -> bool:
    value = os.environ.get("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED must be a boolean value.")


def _evaluation_reward(profile: str, path: str | Path | None, task_mode: str,
                       observe_after_success_s: float) -> PickPlaceRewardConfig | None:
    if profile not in ("legacy", "pick_place_v2"):
        raise ValueError("reward_profile must be legacy or pick_place_v2.")
    if task_mode not in ("pick_place", "idle_empty", "idle_completed"):
        raise ValueError("Unknown evaluation task_mode.")
    if not math.isfinite(observe_after_success_s) or observe_after_success_s < 0:
        raise ValueError("observe_after_success_s must be finite and non-negative.")
    if profile == "legacy":
        if path is not None or task_mode != "pick_place" or observe_after_success_s:
            raise ValueError("Reward config, task modes and monitoring require pick_place_v2.")
        return None
    reward = PickPlaceRewardConfig.load(path)
    if not math.isclose(reward.control_hz, 30.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("ACT Gazebo evaluation requires control_hz=30 to match the physics schedule.")
    if reward.grasp_evidence_mode != "contact":
        raise ValueError("Formal ACT evaluation requires contact evidence; proxy mode is for offline diagnostics only.")
    return reward


def _evaluation_metadata(loaded: Any, profile: str, reward: PickPlaceRewardConfig | None,
                         task_mode: str, observe_after_success_s: float) -> dict[str, Any]:
    config = getattr(loaded, "config", getattr(loaded.policy, "config", None))
    schedule = validate_act_temporal_config(config) if reward is not None or config is not None else {
        "chunk_size": None, "n_action_steps": None,
        "temporal_ensemble_coeff": None, "chunk_schedule": None,
    }
    return {
        "reward_profile": profile,
        "reward_config": None if reward is None else reward.to_dict(),
        "reward_config_hash": None if reward is None else reward.config_hash,
        "task_mode": task_mode,
        "observe_after_success_s": observe_after_success_s,
        **schedule,
        "calibration_file_sha256": hashlib.sha256(SIM_CALIBRATION_FILE.read_bytes()).hexdigest(),
        "deterministic_act": True,
    }


def _validate_evaluation_scene(scene: dict[str, Any], *, require_fixed_target: bool) -> dict[str, Any]:
    if not isinstance(scene, dict):
        raise ValueError("Each evaluation scene must be a JSON object.")
    allowed = {"name", "task_mode", "target_position_world_m", "target_quaternion_xyzw", "target_delay_s", "seed"}
    if set(scene) - allowed:
        raise ValueError(f"Unknown evaluation scene fields: {sorted(set(scene) - allowed)}")
    result = dict(scene)
    result.setdefault("name", "scene")
    result.setdefault("task_mode", "pick_place")
    result.setdefault("target_delay_s", 0.0)
    result.setdefault("seed", 0)
    if not isinstance(result["name"], str) or not result["name"].strip():
        raise ValueError("Evaluation scene name must be nonempty.")
    if result["task_mode"] not in ("pick_place", "idle_empty", "idle_completed"):
        raise ValueError("Unknown evaluation scene task_mode.")
    delay = float(result["target_delay_s"])
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("target_delay_s must be finite and non-negative.")
    result["target_delay_s"] = delay
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], int) or result["seed"] < 0:
        raise ValueError("Evaluation scene seed must be a non-negative integer.")
    position = result.get("target_position_world_m")
    if position is not None:
        vector = np.asarray(position, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("target_position_world_m must be a finite three-vector.")
        if result["task_mode"] != "pick_place":
            raise ValueError("Only pick_place scenes may supply a target position.")
        result["target_position_world_m"] = vector.tolist()
        quaternion = np.asarray(result.get("target_quaternion_xyzw", (0., 0., 0., 1.)), dtype=float)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-6):
            raise ValueError("target_quaternion_xyzw must be a normalized finite quaternion.")
        result["target_quaternion_xyzw"] = quaternion.tolist()
    elif require_fixed_target and result["task_mode"] == "pick_place":
        raise ValueError("Fixed pick_place evaluation scenes require target_position_world_m.")
    elif "target_quaternion_xyzw" in result:
        raise ValueError("target_quaternion_xyzw requires target_position_world_m.")
    return result


def _load_evaluation_scenes(path: str | Path, *, episode_count: int) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or set(payload) != {"schema_version", "scenes"}:
        raise ValueError("Evaluation scenes require schema_version=1 and a scenes list.")
    scenes = payload["scenes"]
    if not isinstance(scenes, list) or len(scenes) != episode_count:
        raise ValueError("The evaluation scene count must equal episode_count.")
    return tuple(_validate_evaluation_scene(scene, require_fixed_target=True) for scene in scenes)


class _EvaluationSceneRuntimeProxy:
    """Configure explicit evaluation scenes without altering ACT's inference cadence."""

    def __init__(self, runtime: Any, scenes: Sequence[dict[str, Any]], *, n_action_steps: int, policy_hz: float,
                 environment_randomization_enabled: bool = False):
        self.runtime, self.scenes = runtime, tuple(scenes)
        self.n_action_steps, self.policy_hz = n_action_steps, policy_hz
        self.episode_index = 0
        self.completed_steps = 0
        self.spawned = False
        self.environment_randomization_enabled = environment_randomization_enabled

    def select_episode(self, index: int) -> None:
        if not 0 <= index < len(self.scenes):
            raise ValueError("Evaluation scene index is out of range.")
        self.episode_index = index

    @staticmethod
    def _check_service(response: Any, operation: str) -> None:
        success = response.get("success", True) if isinstance(response, dict) else getattr(response, "success", True)
        if not success:
            raise TargetCubeControlError(f"Evaluation target service failed: {operation}.")

    def _spawn(self) -> None:
        scene = self.scenes[self.episode_index]
        position = scene.get("target_position_world_m")
        response = (self.runtime.spawn_random_target() if position is None else
                    self.runtime.spawn_target(tuple(position), quaternion_xyzw=tuple(scene["target_quaternion_xyzw"])))
        self._check_service(response, "spawn")
        self.spawned = True

    def prepare_episode(self) -> Any:
        scene = self.scenes[self.episode_index]
        self.completed_steps, self.spawned = 0, False
        self._check_service(self.runtime.delete_all_targets(), "delete")
        self.runtime.configure_task(scene["task_mode"], seed=scene["seed"], n_action_steps=self.n_action_steps)
        kwargs = {"environment_randomization_episode_key": scene["seed"]} if self.environment_randomization_enabled else {}
        prepared = self.runtime.prepare_episode(**kwargs)
        self.delay_steps = int(round(scene["target_delay_s"] * self.policy_hz))
        if scene["task_mode"] == "pick_place" and self.delay_steps == 0:
            self._spawn()
        return SimpleNamespace(observation=prepared.observation,
                               info={**prepared.info, "evaluation_scene": dict(scene), "evaluation_scene_index": self.episode_index})

    def step(self, action: Sequence[float]) -> Any:
        result = self.runtime.step(action)
        self.completed_steps += 1
        scene = self.scenes[self.episode_index]
        if (not result.terminated and not result.truncated and not self.spawned
                and scene["task_mode"] == "pick_place" and self.completed_steps == self.delay_steps):
            self._spawn()
        return result


@dataclass(frozen=True)
class ActGazeboEpisodeResult:
    """Summary and transitions from one ACT-controlled episode."""

    checkpoint_path: Path
    device: torch.device
    episode: PolicyEpisodeResult
    episode_npz_path: Path | None
    evaluation_metadata: dict[str, Any] | None = None


def run_act_gazebo_episode(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    max_joint_delta_rad: float | Sequence[float] | None = None,
    max_steps: int,
    startup_feedback_timeout_s: float = 30.0,
    checkpoint_loader: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    runner_factory: Callable[..., Any] | None = None,
    npz_sink_factory: Callable[..., Any] | None = None,
    output_npz_path: str | Path | None = None,
    reward_profile: str = "legacy",
    reward_config: str | Path | None = None,
    task_mode: str = "pick_place",
    observe_after_success_s: float = 0.0,
    target_position_world_m: Sequence[float] | None = None,
    target_quaternion_xyzw: Sequence[float] = (0., 0., 0., 1.),
    target_delay_s: float = 0.0,
    scene_seed: int = 0,
) -> ActGazeboEpisodeResult:
    """Load ACT and execute exactly one prepared policy episode."""

    resolved_reward_path = (_evaluation_reward_path(checkpoint_path, reward_config)
                            if reward_profile == "pick_place_v2" else reward_config)
    reward = _evaluation_reward(reward_profile, resolved_reward_path, task_mode, observe_after_success_s)
    randomization_enabled = reward is not None and _evaluation_randomization_from_environment()
    if randomization_enabled and task_mode == "idle_completed":
        raise ValueError("idle_completed evaluation requires environment randomization disabled.")
    if reward is None and (target_position_world_m is not None or target_delay_s):
        raise ValueError("Managed evaluation targets require pick_place_v2.")
    scene = None
    if reward is not None:
        scene_input = dict(task_mode=task_mode, target_delay_s=target_delay_s, seed=scene_seed)
        if target_position_world_m is not None:
            scene_input.update(target_position_world_m=target_position_world_m, target_quaternion_xyzw=target_quaternion_xyzw)
        scene = _validate_evaluation_scene(scene_input, require_fixed_target=False)
    resolved_checkpoint_loader = (
        load_act_checkpoint
        if checkpoint_loader is None
        else checkpoint_loader
    )
    resolved_runtime_factory = (
        create_ros_simulation_runtime
        if runtime_factory is None
        else runtime_factory
    )
    resolved_adapter_factory = (
        ActPolicyAdapter
        if adapter_factory is None
        else adapter_factory
    )
    resolved_runner_factory = (
        PolicyEpisodeRunner
        if runner_factory is None
        else runner_factory
    )
    resolved_npz_sink_factory = (
        PolicyEpisodeNpzSink
        if npz_sink_factory is None
        else npz_sink_factory
    )

    loaded = resolved_checkpoint_loader(
        checkpoint_path,
        device=device,
    )
    evaluation_metadata = _evaluation_metadata(loaded, reward_profile, reward, task_mode, observe_after_success_s)
    evaluation_metadata["reward_config_source_path"] = None if resolved_reward_path is None else str(resolved_reward_path)
    if scene is not None:
        evaluation_metadata["scene"] = scene

    adapter = resolved_adapter_factory(
        loaded.policy,
        device=loaded.device,
        max_joint_delta_rad=max_joint_delta_rad,
    )

    runtime_config = SimulationRuntimeConfig(
        startup_feedback_timeout_s=(
            startup_feedback_timeout_s
        )
    )

    transition_sink = None

    if output_npz_path is not None:
        transition_sink = resolved_npz_sink_factory(
            output_npz_path,
            metadata={
                "checkpoint": str(
                    Path(checkpoint_path)
                    .expanduser()
                    .resolve()
                ),
                "device": str(loaded.device),
                "max_steps": max_steps,
                "max_joint_delta_rad": (
                    None
                    if max_joint_delta_rad is None
                    else (
                        float(max_joint_delta_rad)
                        if isinstance(max_joint_delta_rad, (int, float))
                        else list(max_joint_delta_rad)
                    )
                ),
                **({"evaluation": evaluation_metadata} if reward is not None else {}),
            },
        )

    runtime_kwargs = {"runtime_config": runtime_config}
    if runtime_factory is None:
        runtime_kwargs["environment_config"] = ReachEnvironmentConfig(
            max_episode_steps=max_steps,
        )
    if reward is not None:
        runtime_kwargs.update(reward_profile=reward_profile, reward_config=reward, task_mode=task_mode,
                              observe_after_success_s=observe_after_success_s,
                              environment_randomization_enabled=randomization_enabled)
        runtime_kwargs["environment_config"] = ReachEnvironmentConfig(max_episode_steps=max_steps, control_hz=reward.control_hz)
    with resolved_runtime_factory(**runtime_kwargs) as runtime:
        runner_runtime = runtime
        runner_kwargs: dict[str, Any] = {}
        if reward is not None:
            assert scene is not None
            runner_runtime = _EvaluationSceneRuntimeProxy(runtime, [scene],
                n_action_steps=int(evaluation_metadata["n_action_steps"]), policy_hz=reward.control_hz,
                environment_randomization_enabled=randomization_enabled)
            if observe_after_success_s:
                runner_kwargs.update(observe_after_success_s=observe_after_success_s, policy_hz=reward.control_hz)
        if transition_sink is None:
            runner = resolved_runner_factory(
                runner_runtime,
                adapter,
                **runner_kwargs,
            )
        else:
            runner = resolved_runner_factory(
                runner_runtime,
                adapter,
                transition_sink=transition_sink,
                **runner_kwargs,
            )

        episode = runner.run_episode(
            max_steps=max_steps,
        )

    return ActGazeboEpisodeResult(
        checkpoint_path=(
            Path(checkpoint_path)
            .expanduser()
            .resolve()
        ),
        device=torch.device(loaded.device),
        episode=episode,
        episode_npz_path=(
            None
            if transition_sink is None
            else transition_sink.written_path
        ),
        evaluation_metadata=evaluation_metadata,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local ACT checkpoint and execute one real "
            "policy-controlled episode in deterministic Gazebo."
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
        help=(
            "Maximum absolute per-step change for each joint, "
            "in radians. Disabled when omitted; absolute joint limits remain."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--startup-feedback-timeout-s",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help=(
            "Optional compressed episode archive path. "
            "Joint state, actions, rewards, done flags, and "
            "metadata are stored; camera images are excluded."
        ),
    )
    parser.add_argument("--reward-profile", choices=("legacy", "pick_place_v2"), default="pick_place_v2")
    parser.add_argument("--reward-config", type=Path, default=None)
    parser.add_argument("--task-mode", choices=("pick_place", "idle_empty", "idle_completed"), default="pick_place")
    parser.add_argument("--observe-after-success-s", type=float, default=0.0)
    parser.add_argument("--target-position-world-m", nargs=3, type=float, default=None)
    parser.add_argument("--target-quaternion-xyzw", nargs=4, type=float, default=(0., 0., 0., 1.))
    parser.add_argument("--target-delay-s", type=float, default=0.0)
    parser.add_argument("--scene-seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()

    result = run_act_gazebo_episode(
        args.checkpoint,
        device=args.device,
        max_joint_delta_rad=(
            args.max_joint_delta_rad
        ),
        max_steps=args.max_steps,
        startup_feedback_timeout_s=(
            args.startup_feedback_timeout_s
        ),
        output_npz_path=args.output_npz,
        reward_profile=args.reward_profile,
        reward_config=args.reward_config,
        task_mode=args.task_mode,
        observe_after_success_s=args.observe_after_success_s,
        target_position_world_m=args.target_position_world_m,
        target_quaternion_xyzw=args.target_quaternion_xyzw,
        target_delay_s=args.target_delay_s,
        scene_seed=args.scene_seed,
    )

    episode = result.episode

    print("ACT Gazebo episode completed.")
    print(f"checkpoint={result.checkpoint_path}")
    print(f"device={result.device}")
    print(f"step_count={episode.step_count}")
    print(f"total_reward={episode.total_reward:.6f}")
    print(f"terminated={episode.terminated}")
    print(f"truncated={episode.truncated}")
    print(
        "reached_max_steps="
        f"{episode.reached_max_steps}"
    )
    if "evaluation" in episode.final_info:
        print("evaluation=" + json.dumps(episode.final_info["evaluation"], sort_keys=True))

    if result.episode_npz_path is not None:
        print(
            "episode_npz_path="
            f"{result.episode_npz_path}"
        )

    for key in (
        "episode_step",
        "target_active",
        "target_cube_name",
        "target_transition",
        "target_collection_cube_count",
        "step_end_sim_time_s",
    ):
        if key in episode.final_info:
            print(f"final_{key}={episode.final_info[key]}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
