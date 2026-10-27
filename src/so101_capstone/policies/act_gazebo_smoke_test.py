"""Safe ACT dry-run against the assembled SO-101 Gazebo runtime.

This module performs:

1. Local ACT checkpoint loading.
2. Existing deterministic runtime construction.
3. Existing HOME reset.
4. Collection of the runtime's first processed policy observation.
5. One ACT inference.
6. Logging of the proposed controller action.

It deliberately does not call ``runtime.step()``. Therefore the ACT action is
not published to the Gazebo position controller.

The HOME reset itself still publishes the configured HOME joint targets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from so101_capstone.environments import (
    PolicyObservation,
    SimulationRuntimeConfig,
    create_ros_simulation_runtime,
)

from .act_checkpoint_loader import (
    load_act_checkpoint,
)
from .act_policy_adapter import (
    ACT_IMAGE_CHANNELS,
    ACT_IMAGE_HEIGHT,
    ACT_IMAGE_WIDTH,
    ActPolicyAdapter,
)


_EXPECTED_IMAGE_SHAPE = (
    ACT_IMAGE_HEIGHT,
    ACT_IMAGE_WIDTH,
    ACT_IMAGE_CHANNELS,
)


def _readonly_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)

    if vector.shape != (6,):
        raise ValueError(
            f"{name} must have shape (6,)."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    result = vector.copy()
    result.setflags(write=False)
    return result


def validate_dry_run_observation(
    observation: PolicyObservation,
) -> None:
    """Validate the exact observation expected by the ACT checkpoint."""

    for name in (
        "joint_positions",
        "joint_velocities",
    ):
        value = np.asarray(
            getattr(observation, name)
        )

        if value.shape != (6,):
            raise ValueError(
                f"{name} must have shape (6,), "
                f"got {value.shape}."
            )

        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{name} must contain only finite values."
            )

    for name in (
        "top_image",
        "gripper_image",
    ):
        image = np.asarray(
            getattr(observation, name)
        )

        if image.shape != _EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"{name} must have shape "
                f"{_EXPECTED_IMAGE_SHAPE}, "
                f"got {image.shape}."
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"{name} must have dtype uint8, "
                f"got {image.dtype}."
            )


@dataclass(frozen=True)
class ActGazeboDryRunResult:
    """Data collected without publishing the ACT action."""

    checkpoint_path: Path
    device: torch.device
    joint_positions_rad: np.ndarray
    raw_policy_positions: np.ndarray
    clipped_policy_positions: np.ndarray
    proposed_controller_command_rad: np.ndarray
    top_image_shape: tuple[int, ...]
    gripper_image_shape: tuple[int, ...]
    reset_info: dict[str, Any]


def run_act_gazebo_dry_run(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    max_joint_delta_rad: float | Sequence[float] | None = None,
    startup_feedback_timeout_s: float = 30.0,
    checkpoint_loader: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
) -> ActGazeboDryRunResult:
    """Run one inference without executing the proposed ACT action."""

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

    loaded = resolved_checkpoint_loader(
        checkpoint_path,
        device=device,
    )

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

    with resolved_runtime_factory(
        runtime_config=runtime_config,
    ) as runtime:
        # This moves the robot to the configured HOME pose and leaves
        # Gazebo paused for deterministic stepping.
        reset_result = runtime.reset()

        # Clear ACT's queued action chunk at the episode boundary.
        adapter.reset()

        observation = reset_result.observation
        validate_dry_run_observation(observation)

        policy_result = adapter.select_action(
            observation
        )

        # Deliberately do not call:
        #
        # runtime.step(
        #     policy_result.controller_command()
        # )
        #
        # Therefore no ACT command is published in this dry-run.

        return ActGazeboDryRunResult(
            checkpoint_path=(
                Path(checkpoint_path)
                .expanduser()
                .resolve()
            ),
            device=torch.device(loaded.device),
            joint_positions_rad=_readonly_vector(
                observation.joint_positions,
                name="joint_positions_rad",
            ),
            raw_policy_positions=_readonly_vector(
                policy_result.raw_policy_positions,
                name="raw_policy_positions",
            ),
            clipped_policy_positions=_readonly_vector(
                policy_result.clipped_policy_positions,
                name="clipped_policy_positions",
            ),
            proposed_controller_command_rad=(
                _readonly_vector(
                    policy_result.controller_command(),
                    name=(
                        "proposed_controller_command_rad"
                    ),
                )
            ),
            top_image_shape=tuple(
                observation.top_image.shape
            ),
            gripper_image_shape=tuple(
                observation.gripper_image.shape
            ),
            reset_info=dict(reset_result.info),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reset the existing deterministic Gazebo runtime, "
            "run one ACT inference, and print the proposed "
            "action without executing it."
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
            "Optional per-step delta limit for the proposed command. "
            "Disabled by default; absolute joint limits remain."
        ),
    )
    parser.add_argument(
        "--startup-feedback-timeout-s",
        type=float,
        default=30.0,
    )

    return parser


def main() -> None:
    args = _parser().parse_args()

    result = run_act_gazebo_dry_run(
        args.checkpoint,
        device=args.device,
        max_joint_delta_rad=(
            args.max_joint_delta_rad
        ),
        startup_feedback_timeout_s=(
            args.startup_feedback_timeout_s
        ),
    )

    print("ACT Gazebo dry-run completed.")
    print(f"checkpoint={result.checkpoint_path}")
    print(f"device={result.device}")
    print(
        "top_image_shape="
        f"{result.top_image_shape}"
    )
    print(
        "gripper_image_shape="
        f"{result.gripper_image_shape}"
    )
    print(
        "joint_positions_rad="
        f"{np.round(result.joint_positions_rad, 6).tolist()}"
    )
    print(
        "raw_policy_positions="
        f"{np.round(result.raw_policy_positions, 6).tolist()}"
    )
    print(
        "clipped_policy_positions="
        f"{np.round(result.clipped_policy_positions, 6).tolist()}"
    )
    print(
        "proposed_controller_command_rad="
        f"{np.round(result.proposed_controller_command_rad, 6).tolist()}"
    )

    for key in (
        "reset_sim_time_s",
        "top_image_sequence",
        "gripper_image_sequence",
        "camera_skew_s",
        "target_active",
        "target_collection_cube_count",
    ):
        if key in result.reset_info:
            print(
                f"{key}={result.reset_info[key]}"
            )

    print(
        "[PASS] ACT inference completed without "
        "executing the policy action."
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
