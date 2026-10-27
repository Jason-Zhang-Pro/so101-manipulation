"""LeRobot ACT adapter for the SO-101 Gazebo environment.

The adapter presents simulation observations to ACT using the checkpoint's
training interface:

- ``observation.images.fixed``: RGB BCHW float32 image in [0, 1].
- ``observation.images.handeye``: RGB BCHW float32 image in [0, 1].
- ``observation.state``: six LeRobot-calibrated joint positions.

The ACT checkpoint returns six absolute LeRobot-calibrated joint targets.

The checkpoint performs its own mean/std input normalization and output
unnormalization. This adapter must not repeat those operations.

Joint velocities and a policy ``done`` output are not part of this checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch

from so101_capstone.environments.policy_action import (
    PolicyAction,
    parse_policy_action,
)
from so101_capstone.environments.policy_observation import (
    PolicyObservation,
)

from .so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS,
    SIM_POLICY_JOINT_COUNT,
    gazebo_positions_to_policy,
    policy_positions_to_gazebo,
)


ACT_FIXED_IMAGE_KEY = "observation.images.fixed"
ACT_HANDEYE_IMAGE_KEY = "observation.images.handeye"
ACT_STATE_KEY = "observation.state"

ACT_IMAGE_HEIGHT = 480
ACT_IMAGE_WIDTH = 640
ACT_IMAGE_CHANNELS = 3


class ActPolicyLike(Protocol):
    """Minimum LeRobot ACT interface required by this adapter."""

    def select_action(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return one batched six-dimensional action."""

    def reset(self) -> None:
        """Reset the policy action-chunk queue."""


def _readonly_six_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)

    if vector.shape != (SIM_POLICY_JOINT_COUNT,):
        raise ValueError(
            f"{name} must have shape "
            f"({SIM_POLICY_JOINT_COUNT},)."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    result = vector.copy()
    result.setflags(write=False)
    return result


def _validated_max_joint_delta(
    value: float | Sequence[float],
) -> np.ndarray:
    if np.isscalar(value):
        result = np.full(
            SIM_POLICY_JOINT_COUNT,
            float(value),
            dtype=float,
        )
    else:
        result = np.asarray(value, dtype=float)

    if result.shape != (SIM_POLICY_JOINT_COUNT,):
        raise ValueError(
            "max_joint_delta_rad must be a scalar or "
            "have shape (6,)."
        )

    if (
        not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError(
            "max_joint_delta_rad must contain only "
            "positive finite values."
        )

    result = result.copy()
    result.setflags(write=False)
    return result


def _validated_rgb_image(
    value: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    image = np.asarray(value)

    expected_shape = (
        ACT_IMAGE_HEIGHT,
        ACT_IMAGE_WIDTH,
        ACT_IMAGE_CHANNELS,
    )

    if image.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"{name} must have dtype uint8."
        )

    return np.ascontiguousarray(image).copy()


def _rgb_image_to_tensor(
    image: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert RGB HWC uint8 to BCHW float32 in [0, 1]."""

    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
        / 255.0
    )


_POLICY_LOWER = np.array(
    [
        calibration.policy_lower
        for calibration in SIM_JOINT_CALIBRATIONS
    ],
    dtype=float,
)

_POLICY_UPPER = np.array(
    [
        calibration.policy_upper
        for calibration in SIM_JOINT_CALIBRATIONS
    ],
    dtype=float,
)

_GAZEBO_LOWER_RAD = np.array(
    [
        calibration.gazebo_lower_rad
        for calibration in SIM_JOINT_CALIBRATIONS
    ],
    dtype=float,
)

_GAZEBO_UPPER_RAD = np.array(
    [
        calibration.gazebo_upper_rad
        for calibration in SIM_JOINT_CALIBRATIONS
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ActPolicyResult:
    """One ACT inference result and its safe controller action."""

    raw_policy_positions: np.ndarray
    clipped_policy_positions: np.ndarray
    mapped_gazebo_positions_rad: np.ndarray
    action: PolicyAction

    def controller_command(self) -> np.ndarray:
        """Return the final six-dimensional controller command."""

        return self.action.controller_command()


class ActPolicyAdapter:
    """Convert policy observations and ACT absolute actions."""

    def __init__(
        self,
        policy: ActPolicyLike,
        *,
        device: str | torch.device,
        max_joint_delta_rad: float | Sequence[float] | None = None,
    ) -> None:
        self.policy = policy
        self.device = torch.device(device)
        self.max_joint_delta_rad = (
            None
            if max_joint_delta_rad is None
            else _validated_max_joint_delta(max_joint_delta_rad)
        )

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable."
            )

    def reset(self) -> None:
        """Reset ACT's queued action chunk at episode reset."""

        self.policy.reset()

    def build_batch(
        self,
        observation: PolicyObservation,
    ) -> dict[str, torch.Tensor]:
        """Build one ACT inference batch.

        Joint velocities are deliberately excluded because this checkpoint
        uses a six-dimensional position-only state.
        """

        joint_positions_rad = _readonly_six_vector(
            observation.joint_positions,
            name="observation.joint_positions",
        )

        top_image = _validated_rgb_image(
            observation.top_image,
            name="observation.top_image",
        )

        gripper_image = _validated_rgb_image(
            observation.gripper_image,
            name="observation.gripper_image",
        )

        calibrated_state = gazebo_positions_to_policy(
            joint_positions_rad
        )

        state_tensor = (
            torch.from_numpy(
                calibrated_state.astype(
                    np.float32,
                    copy=True,
                )
            )
            .unsqueeze(0)
            .to(self.device)
        )

        return {
            ACT_FIXED_IMAGE_KEY: _rgb_image_to_tensor(
                top_image,
                device=self.device,
            ),
            ACT_HANDEYE_IMAGE_KEY: _rgb_image_to_tensor(
                gripper_image,
                device=self.device,
            ),
            ACT_STATE_KEY: state_tensor,
        }

    def map_policy_positions(
        self,
        observation: PolicyObservation,
        policy_positions: Sequence[float] | np.ndarray,
    ) -> ActPolicyResult:
        """Map externally supplied calibrated policy positions safely."""

        current_positions_rad = _readonly_six_vector(
            observation.joint_positions,
            name="observation.joint_positions",
        )

        raw_policy_positions = _readonly_six_vector(
            policy_positions,
            name="policy_positions",
        )

        clipped_policy_positions = np.clip(
            raw_policy_positions,
            _POLICY_LOWER,
            _POLICY_UPPER,
        )

        mapped_gazebo_positions_rad = (
            policy_positions_to_gazebo(
                clipped_policy_positions,
                clip=False,
            )
        )

        delta_limited_targets_rad = mapped_gazebo_positions_rad
        if self.max_joint_delta_rad is not None:
            delta_limited_targets_rad = np.clip(
                mapped_gazebo_positions_rad,
                current_positions_rad - self.max_joint_delta_rad,
                current_positions_rad + self.max_joint_delta_rad,
            )

        controller_targets_rad = np.clip(
            delta_limited_targets_rad,
            _GAZEBO_LOWER_RAD,
            _GAZEBO_UPPER_RAD,
        )

        action = parse_policy_action(
            controller_targets_rad
        )

        return ActPolicyResult(
            raw_policy_positions=_readonly_six_vector(
                raw_policy_positions,
                name="raw_policy_positions",
            ),
            clipped_policy_positions=_readonly_six_vector(
                clipped_policy_positions,
                name="clipped_policy_positions",
            ),
            mapped_gazebo_positions_rad=_readonly_six_vector(
                mapped_gazebo_positions_rad,
                name="mapped_gazebo_positions_rad",
            ),
            action=action,
        )

    def select_action(
        self,
        observation: PolicyObservation,
    ) -> ActPolicyResult:
        """Run ACT and produce a safe absolute Gazebo action."""

        batch = self.build_batch(observation)

        with torch.inference_mode():
            action_tensor = self.policy.select_action(
                batch
            )

        if not isinstance(action_tensor, torch.Tensor):
            raise TypeError(
                "ACT select_action must return a torch.Tensor."
            )

        expected_shape = (
            1,
            SIM_POLICY_JOINT_COUNT,
        )

        if tuple(action_tensor.shape) != expected_shape:
            raise ValueError(
                "ACT action must have shape "
                f"{expected_shape}."
            )

        if not bool(
            torch.isfinite(action_tensor).all().item()
        ):
            raise ValueError(
                "ACT action must contain only finite values."
            )

        return self.map_policy_positions(
            observation,
            action_tensor
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
            .numpy()[0],
        )
