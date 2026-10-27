"""Frozen-ACT plus Residual-PPO reward-based training in SO-101 Gazebo."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from so101_capstone.environments.simulation_runtime import (
    SimulationRuntimeConfig,
    create_ros_simulation_runtime,
)
from so101_capstone.policies.act_checkpoint_loader import (
    load_act_checkpoint,
)
from so101_capstone.policies.act_policy_adapter import (
    ActPolicyAdapter,
)
from so101_capstone.policies.so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS,
)

from .residual_ppo import (
    CompressedResidualObservation,
    ResidualActorCritic,
    ResidualPpoConfig,
    ResidualPpoTrainer,
    ResidualRolloutBuffer,
    compress_policy_observation,
    freeze_module,
    model_state_sha256,
)


FIRST_CHAIN_DEVICE = "cuda"
FIRST_CHAIN_EPISODE_COUNT = 6
FIRST_CHAIN_MAX_STEPS = 600
FIRST_CHAIN_TARGET_DELAY_S = 3.0
FIRST_CHAIN_ROLLOUT_STEPS = 600
FIRST_CHAIN_RESIDUAL_LIMIT_RAD = 0.1
FIRST_CHAIN_POLICY_HZ = 30.0
_DISABLED_DELTA_SENTINEL_RAD = 1_000_000.0

_GAZEBO_LOWER_RAD = np.asarray(
    [calibration.gazebo_lower_rad for calibration in SIM_JOINT_CALIBRATIONS],
    dtype=np.float64,
)
_GAZEBO_UPPER_RAD = np.asarray(
    [calibration.gazebo_upper_rad for calibration in SIM_JOINT_CALIBRATIONS],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ResidualPpoGazeboConfig:
    """Runtime and rollout configuration for one local training invocation."""

    act_checkpoint: Path
    device: str = FIRST_CHAIN_DEVICE
    episode_count: int = FIRST_CHAIN_EPISODE_COUNT
    max_steps_per_episode: int = FIRST_CHAIN_MAX_STEPS
    target_delay_s: float = FIRST_CHAIN_TARGET_DELAY_S
    rollout_steps: int = FIRST_CHAIN_ROLLOUT_STEPS
    residual_limit_rad: float = FIRST_CHAIN_RESIDUAL_LIMIT_RAD
    disable_max_joint_delta_limit: bool = True
    policy_hz: float = FIRST_CHAIN_POLICY_HZ
    startup_feedback_timeout_s: float = 30.0
    seed: int = 0
    output_directory: Path | None = None
    resume_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        checkpoint = Path(self.act_checkpoint).expanduser()
        object.__setattr__(self, "act_checkpoint", checkpoint)
        if self.output_directory is not None:
            object.__setattr__(
                self,
                "output_directory",
                Path(self.output_directory).expanduser(),
            )
        if self.resume_checkpoint is not None:
            object.__setattr__(
                self,
                "resume_checkpoint",
                Path(self.resume_checkpoint).expanduser(),
            )
        for name, value in (
            ("episode_count", self.episode_count),
            ("max_steps_per_episode", self.max_steps_per_episode),
            ("rollout_steps", self.rollout_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value, allow_zero in (
            ("target_delay_s", self.target_delay_s, True),
            ("residual_limit_rad", self.residual_limit_rad, False),
            ("policy_hz", self.policy_hz, False),
            ("startup_feedback_timeout_s", self.startup_feedback_timeout_s, False),
        ):
            result = float(value)
            invalid = not math.isfinite(result) or result < 0.0
            if not allow_zero:
                invalid = invalid or result == 0.0
            if invalid:
                requirement = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {requirement}.")
        if not isinstance(self.disable_max_joint_delta_limit, bool):
            raise ValueError("disable_max_joint_delta_limit must be a bool.")
        delay_steps_float = self.target_delay_s * self.policy_hz
        delay_steps = int(round(delay_steps_float))
        if not math.isclose(delay_steps_float, delay_steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "target_delay_s must map to an integer number of policy steps."
            )
        if delay_steps > self.max_steps_per_episode:
            raise ValueError(
                "target_delay_s occurs after max_steps_per_episode."
            )

    @property
    def target_delay_steps(self) -> int:
        return int(round(self.target_delay_s * self.policy_hz))

    @property
    def planned_transition_count(self) -> int:
        return self.episode_count * self.max_steps_per_episode

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("act_checkpoint", "output_directory", "resume_checkpoint"):
            value = result[key]
            result[key] = None if value is None else str(value)
        result["target_delay_steps"] = self.target_delay_steps
        result["planned_transition_count"] = self.planned_transition_count
        result["max_joint_delta_rad"] = None
        return result


def validate_first_chain_contract(config: ResidualPpoGazeboConfig) -> None:
    """Reject silent changes to the user-approved first-link design."""

    expected = {
        "device": FIRST_CHAIN_DEVICE,
        "episode_count": FIRST_CHAIN_EPISODE_COUNT,
        "max_steps_per_episode": FIRST_CHAIN_MAX_STEPS,
        "target_delay_s": FIRST_CHAIN_TARGET_DELAY_S,
        "rollout_steps": FIRST_CHAIN_ROLLOUT_STEPS,
        "residual_limit_rad": FIRST_CHAIN_RESIDUAL_LIMIT_RAD,
        "disable_max_joint_delta_limit": True,
        "policy_hz": FIRST_CHAIN_POLICY_HZ,
    }
    actual = {
        name: getattr(config, name)
        for name in expected
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, received {actual[name]!r}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "The first Residual-PPO chain is fixed and cannot be changed "
            "without explicit approval:\n- " + "\n- ".join(mismatches)
        )


def compose_residual_controller_command(
    base_action_rad: Sequence[float] | np.ndarray,
    residual_action_rad: Sequence[float] | np.ndarray,
    *,
    residual_limit_rad: float = FIRST_CHAIN_RESIDUAL_LIMIT_RAD,
) -> np.ndarray:
    """Add residual to ACT absolute target and apply only absolute limits."""

    base = np.asarray(base_action_rad, dtype=np.float64)
    residual = np.asarray(residual_action_rad, dtype=np.float64)
    if base.shape != (6,) or residual.shape != (6,):
        raise ValueError("base_action_rad and residual_action_rad must have shape (6,).")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(residual)):
        raise ValueError("Controller actions must contain only finite values.")
    if not math.isfinite(float(residual_limit_rad)) or residual_limit_rad <= 0.0:
        raise ValueError("residual_limit_rad must be positive and finite.")
    if np.any(np.abs(residual) > residual_limit_rad + 1e-6):
        raise ValueError("Residual action exceeds residual_limit_rad.")
    command = np.clip(base + residual, _GAZEBO_LOWER_RAD, _GAZEBO_UPPER_RAD)
    if command.shape != (6,) or not np.all(np.isfinite(command)):
        raise RuntimeError("Composed controller command is invalid.")
    result = command.copy()
    result.setflags(write=False)
    return result


def _observation_tensors(
    observation: CompressedResidualObservation,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.asarray(observation.top_image).copy())
        .unsqueeze(0)
        .to(device=device),
        torch.from_numpy(np.asarray(observation.gripper_image).copy())
        .unsqueeze(0)
        .to(device=device),
        torch.from_numpy(np.asarray(observation.joint_positions).copy())
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
        torch.from_numpy(np.asarray(observation.joint_velocities).copy())
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
    )


def _base_action_tensor(
    base_action: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(base_action, dtype=np.float32).copy())
        .unsqueeze(0)
        .to(device=device)
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, sort_keys=True)
        stream.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_PRETRAINED_RUN_PATTERN = re.compile(
    r"(?:^|/)act_so101_pick_test_(\d{8}_\d{6})(?:/|$)"
)


def _pretrained_run_timestamp(act_checkpoint: Path) -> str:
    """Extract the original ACT training run timestamp from its checkpoint path."""

    match = _PRETRAINED_RUN_PATTERN.search(Path(act_checkpoint).as_posix())
    if match is None:
        raise ValueError(
            "ACT checkpoint path must contain "
            "'act_so101_pick_test_YYYYMMDD_HHMMSS' so finetune lineage "
            "can be encoded in the output directory."
        )
    return match.group(1)


def _default_output_directory(act_checkpoint: Path) -> Path:
    finetune_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pretrained_timestamp = _pretrained_run_timestamp(act_checkpoint)
    return (
        Path("output_simulation")
        / "finetuned"
        / (
            f"residual_ppo_{finetune_timestamp}"
            f"_from_pretrained_{pretrained_timestamp}"
        )
    )


def _set_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _freeze_and_verify_act(policy: torch.nn.Module) -> str:
    freeze_module(policy)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("ACT policy still contains trainable parameters.")
    return model_state_sha256(policy)


def _spawn_result_payload(response: Any, *, completed_step: int) -> dict[str, Any]:
    success = bool(getattr(response, "success", True))
    message = str(getattr(response, "message", ""))
    if not success:
        raise RuntimeError(f"Target spawn service rejected the request: {message}")
    return {
        "requested": True,
        "request_completed_step": completed_step,
        "service_success": success,
        "service_message": message,
    }


def _delete_result_payload(response: Any, *, stage: str) -> dict[str, Any]:
    """Validate one authoritative target delete-all service response."""

    success = bool(getattr(response, "success", True))
    message = str(getattr(response, "message", ""))
    if not success:
        raise RuntimeError(
            f"Target delete-all service rejected cleanup {stage}: {message}"
        )
    return {
        "requested": True,
        "stage": stage,
        "service_success": success,
        "service_message": message,
    }


def _prepare_clean_episode(runtime: Any, *, episode_index: int) -> tuple[Any, dict[str, Any]]:
    """Delete all prior targets before invoking the preserving prepare lifecycle."""

    cleanup = _delete_result_payload(
        runtime.delete_all_targets(),
        stage=f"before episode {episode_index}",
    )
    prepared = runtime.prepare_episode()
    return prepared, cleanup


def _save_training_checkpoint(
    trainer: ResidualPpoTrainer,
    *,
    output_directory: Path,
    name: str,
    act_fingerprint: str,
    config: ResidualPpoGazeboConfig,
    metrics: dict[str, Any] | None,
) -> Path:
    path = output_directory / "checkpoints" / name
    return trainer.save_checkpoint(
        path,
        extra={
            "act_checkpoint": str(config.act_checkpoint.resolve()),
            "act_state_sha256": act_fingerprint,
            "gazebo_config": config.to_json_dict(),
            "metrics": metrics,
        },
    )


def run_residual_ppo_gazebo(
    config: ResidualPpoGazeboConfig,
    *,
    checkpoint_loader: Callable[..., Any] = load_act_checkpoint,
    runtime_factory: Callable[..., Any] = create_ros_simulation_runtime,
    adapter_factory: Callable[..., Any] = ActPolicyAdapter,
) -> dict[str, Any]:
    """Run the approved six-episode frozen-ACT Residual-PPO chain."""

    validate_first_chain_contract(config)
    _set_seed(config.seed)
    output_directory = (
        _default_output_directory(config.act_checkpoint)
        if config.output_directory is None
        else config.output_directory
    ).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    ppo_config = ResidualPpoConfig(
        residual_limit_rad=config.residual_limit_rad,
    )
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    _write_json(
        output_directory / "config.json",
        {
            "gazebo": config.to_json_dict(),
            "ppo": asdict(ppo_config),
            "act_base_action_source": "ActPolicyResult.mapped_gazebo_positions_rad",
            "max_joint_delta_limit_enabled": False,
            "absolute_joint_limits_enabled": True,
        },
    )

    loaded = checkpoint_loader(config.act_checkpoint, device=device)
    act_policy = loaded.policy
    act_fingerprint = _freeze_and_verify_act(act_policy)
    adapter = adapter_factory(
        act_policy,
        device=loaded.device,
        max_joint_delta_rad=_DISABLED_DELTA_SENTINEL_RAD,
    )

    residual_model = ResidualActorCritic(ppo_config).to(device)
    residual_model.eval()
    trainer = ResidualPpoTrainer(residual_model, config=ppo_config)
    if config.resume_checkpoint is not None:
        trainer.load_checkpoint(config.resume_checkpoint, map_location=device)

    rollout = ResidualRolloutBuffer(ppo_config)
    episode_records: list[dict[str, Any]] = []
    total_collected_steps = 0
    update_records: list[dict[str, Any]] = []

    runtime_config = SimulationRuntimeConfig(
        startup_feedback_timeout_s=config.startup_feedback_timeout_s,
    )

    with runtime_factory(runtime_config=runtime_config) as runtime:
        for episode_index in range(config.episode_count):
            prepared, cleanup_before_episode = _prepare_clean_episode(
                runtime,
                episode_index=episode_index,
            )
            adapter.reset()
            observation = prepared.observation
            episode_reward = 0.0
            terminated = False
            truncated = False
            spawn_payload: dict[str, Any] = {
                "requested": False,
                "request_completed_step": None,
                "service_success": None,
                "service_message": None,
            }
            last_info: dict[str, Any] = dict(prepared.info)
            completed_steps = 0

            if config.target_delay_steps == 0:
                spawn_payload = _spawn_result_payload(
                    runtime.spawn_random_target(),
                    completed_step=0,
                )

            for step_index in range(config.max_steps_per_episode):
                policy_result = adapter.select_action(observation)
                base_action = np.asarray(
                    policy_result.mapped_gazebo_positions_rad,
                    dtype=np.float64,
                )
                if base_action.shape != (6,) or not np.all(np.isfinite(base_action)):
                    raise RuntimeError("ACT mapped Gazebo action is invalid.")

                compressed = compress_policy_observation(
                    observation,
                    image_height=ppo_config.image_height,
                    image_width=ppo_config.image_width,
                )
                top, gripper, positions, velocities = _observation_tensors(
                    compressed,
                    device=device,
                )
                with torch.no_grad():
                    residual_tensor, log_probability_tensor, value_tensor = (
                        residual_model.act(
                            top,
                            gripper,
                            positions,
                            velocities,
                            _base_action_tensor(base_action, device=device),
                        )
                    )
                residual_action = (
                    residual_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
                )
                command = compose_residual_controller_command(
                    base_action,
                    residual_action,
                    residual_limit_rad=config.residual_limit_rad,
                )
                step_result = runtime.step(command)
                completed_steps = step_index + 1
                total_collected_steps += 1

                if (
                    not spawn_payload["requested"]
                    and completed_steps == config.target_delay_steps
                ):
                    spawn_payload = _spawn_result_payload(
                        runtime.spawn_random_target(),
                        completed_step=completed_steps,
                    )

                next_observation = step_result.observation
                next_compressed = compress_policy_observation(
                    next_observation,
                    image_height=ppo_config.image_height,
                    image_width=ppo_config.image_width,
                )
                next_top, next_gripper, next_positions, next_velocities = (
                    _observation_tensors(next_compressed, device=device)
                )
                with torch.no_grad():
                    next_value_tensor = residual_model.value(
                        next_top,
                        next_gripper,
                        next_positions,
                        next_velocities,
                    )

                terminated = bool(step_result.terminated)
                truncated = bool(step_result.truncated)
                reached_max_steps = completed_steps >= config.max_steps_per_episode
                episode_end = terminated or truncated or reached_max_steps
                rollout.add(
                    compressed,
                    base_action=base_action,
                    residual_action=residual_action,
                    log_probability=float(log_probability_tensor.item()),
                    value=float(value_tensor.item()),
                    reward=float(step_result.reward),
                    next_value=float(next_value_tensor.item()),
                    terminated=terminated,
                    episode_end=episode_end,
                )
                episode_reward += float(step_result.reward)
                last_info = dict(step_result.info)
                observation = next_observation
                if episode_end:
                    break

            episode_record = {
                "episode_index": episode_index,
                "step_count": completed_steps,
                "total_reward": episode_reward,
                "terminated": terminated,
                "truncated": truncated,
                "reached_max_steps": (
                    completed_steps >= config.max_steps_per_episode
                    and not terminated
                    and not truncated
                ),
                "target": spawn_payload,
                "cleanup_before_episode": cleanup_before_episode,
                "final_info": last_info,
                "rollout_buffer_size_after_episode": len(rollout),
            }
            episode_records.append(episode_record)
            _write_json(
                output_directory / "episodes" / f"episode_{episode_index:06d}.json",
                episode_record,
            )
            _append_jsonl(output_directory / "metrics.jsonl", {
                "type": "episode",
                **episode_record,
            })

            if len(rollout) >= config.rollout_steps:
                current_fingerprint = model_state_sha256(act_policy)
                if current_fingerprint != act_fingerprint:
                    raise RuntimeError("Frozen ACT parameters changed before PPO update.")
                metrics = trainer.update(rollout)
                metrics_payload = {
                    "type": "ppo_update",
                    "update_index": trainer.update_index,
                    "partial_rollout": False,
                    **metrics.to_dict(),
                }
                update_records.append(metrics_payload)
                _append_jsonl(output_directory / "metrics.jsonl", metrics_payload)
                _save_training_checkpoint(
                    trainer,
                    output_directory=output_directory,
                    name=f"update_{trainer.update_index:06d}.pt",
                    act_fingerprint=act_fingerprint,
                    config=config,
                    metrics=metrics_payload,
                )
                rollout.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        final_target_cleanup = _delete_result_payload(
            runtime.delete_all_targets(),
            stage="after final episode",
        )

    if len(rollout) > 0:
        if len(rollout) < 2:
            raise RuntimeError(
                "Final partial rollout contains fewer than two transitions."
            )
        current_fingerprint = model_state_sha256(act_policy)
        if current_fingerprint != act_fingerprint:
            raise RuntimeError("Frozen ACT parameters changed before final PPO update.")
        metrics = trainer.update(rollout)
        metrics_payload = {
            "type": "ppo_update",
            "update_index": trainer.update_index,
            "partial_rollout": True,
            **metrics.to_dict(),
        }
        update_records.append(metrics_payload)
        _append_jsonl(output_directory / "metrics.jsonl", metrics_payload)
        _save_training_checkpoint(
            trainer,
            output_directory=output_directory,
            name=f"update_{trainer.update_index:06d}.pt",
            act_fingerprint=act_fingerprint,
            config=config,
            metrics=metrics_payload,
        )
        rollout.clear()

    final_act_fingerprint = model_state_sha256(act_policy)
    if final_act_fingerprint != act_fingerprint:
        raise RuntimeError("Frozen ACT parameters changed during residual PPO training.")

    final_checkpoint = _save_training_checkpoint(
        trainer,
        output_directory=output_directory,
        name="final.pt",
        act_fingerprint=act_fingerprint,
        config=config,
        metrics=update_records[-1] if update_records else None,
    )
    summary = {
        "output_directory": str(output_directory),
        "final_checkpoint": str(final_checkpoint),
        "episode_count": len(episode_records),
        "total_collected_steps": total_collected_steps,
        "ppo_update_count": len(update_records),
        "act_state_sha256_before": act_fingerprint,
        "act_state_sha256_after": final_act_fingerprint,
        "act_frozen_verified": True,
        "max_joint_delta_limit_enabled": False,
        "absolute_joint_limits_enabled": True,
        "final_target_cleanup": final_target_cleanup,
        "episodes": episode_records,
        "updates": update_records,
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a bounded Residual-PPO policy over a frozen ACT base policy "
            "in the existing deterministic SO-101 Gazebo runtime."
        )
    )
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=FIRST_CHAIN_DEVICE)
    parser.add_argument("--episode-count", type=int, default=FIRST_CHAIN_EPISODE_COUNT)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=FIRST_CHAIN_MAX_STEPS,
    )
    parser.add_argument(
        "--target-delay-s",
        type=float,
        default=FIRST_CHAIN_TARGET_DELAY_S,
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=FIRST_CHAIN_ROLLOUT_STEPS,
    )
    parser.add_argument(
        "--residual-limit-rad",
        type=float,
        default=FIRST_CHAIN_RESIDUAL_LIMIT_RAD,
    )
    parser.add_argument(
        "--disable-max-joint-delta-limit",
        action="store_true",
        help=(
            "Required for the approved first chain. The ACT mapped target plus "
            "residual is clipped only to absolute Gazebo joint limits."
        ),
    )
    parser.add_argument("--startup-feedback-timeout-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = ResidualPpoGazeboConfig(
        act_checkpoint=args.act_checkpoint,
        device=args.device,
        episode_count=args.episode_count,
        max_steps_per_episode=args.max_steps_per_episode,
        target_delay_s=args.target_delay_s,
        rollout_steps=args.rollout_steps,
        residual_limit_rad=args.residual_limit_rad,
        disable_max_joint_delta_limit=args.disable_max_joint_delta_limit,
        startup_feedback_timeout_s=args.startup_feedback_timeout_s,
        seed=args.seed,
        output_directory=args.output_directory,
        resume_checkpoint=args.resume_checkpoint,
    )
    validate_first_chain_contract(config)
    summary = run_residual_ppo_gazebo(config)
    print("Residual PPO Gazebo training completed.")
    print(f"output_directory={summary['output_directory']}")
    print(f"final_checkpoint={summary['final_checkpoint']}")
    print(f"episode_count={summary['episode_count']}")
    print(f"total_collected_steps={summary['total_collected_steps']}")
    print(f"ppo_update_count={summary['ppo_update_count']}")
    print(f"act_frozen_verified={summary['act_frozen_verified']}")
    print("max_joint_delta_limit_enabled=False")


if __name__ == "__main__":
    main()
