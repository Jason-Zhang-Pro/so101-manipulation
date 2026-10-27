"""Assemble the complete raw policy observation.

The environment-level observation contains measured joint feedback and two
synchronized RGB frames. Model-specific normalization, tensor conversion,
channel ordering, resizing, and batching belong to a later policy adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image_cache import SynchronizedCameraFrames
from .policy_action import POLICY_JOINT_ACTION_SIZE
from .reach_env_core import PolicyStateObservation


def _readonly_state_vector(
    value: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)

    if vector.shape != (POLICY_JOINT_ACTION_SIZE,):
        raise ValueError(
            f"{name} must have shape "
            f"({POLICY_JOINT_ACTION_SIZE},)."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    result = vector.copy()
    result.setflags(write=False)
    return result


def _readonly_rgb_image(
    value: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    image = np.asarray(value)

    if image.dtype != np.uint8:
        raise ValueError(
            f"{name} must have dtype uint8."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape (height, width, 3)."
        )

    result = image.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PolicyObservation:
    """Complete raw observation visible to the policy."""

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    top_image: np.ndarray
    gripper_image: np.ndarray

    def state_vector(self) -> np.ndarray:
        """Return positions followed by velocities as a 12D vector."""

        return np.concatenate(
            (
                self.joint_positions,
                self.joint_velocities,
            )
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return independent arrays suitable for a policy adapter."""

        return {
            "joint_positions": self.joint_positions.copy(),
            "joint_velocities": self.joint_velocities.copy(),
            "top_image": self.top_image.copy(),
            "gripper_image": self.gripper_image.copy(),
        }


def build_policy_observation(
    state: PolicyStateObservation,
    camera_frames: SynchronizedCameraFrames,
) -> PolicyObservation:
    """Combine measured state and synchronized camera frames."""

    return PolicyObservation(
        joint_positions=_readonly_state_vector(
            state.joint_positions,
            name="joint_positions",
        ),
        joint_velocities=_readonly_state_vector(
            state.joint_velocities,
            name="joint_velocities",
        ),
        top_image=_readonly_rgb_image(
            camera_frames.top.image,
            name="top_image",
        ),
        gripper_image=_readonly_rgb_image(
            camera_frames.gripper.image,
            name="gripper_image",
        ),
    )
