"""Policy-action contract for the SO-101 environment.

The environment accepts six absolute joint-position targets in controller
order:

    shoulder_pan
    shoulder_lift
    elbow_flex
    wrist_flex
    wrist_roll
    gripper

Model-specific normalization and conversion into controller-space radians
belong to the policy adapter.

There is no policy completion or ``done`` output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .state_cache import MEASURED_JOINT_NAMES


POLICY_JOINT_NAMES = MEASURED_JOINT_NAMES
POLICY_JOINT_ACTION_SIZE = len(POLICY_JOINT_NAMES)
POLICY_ACTION_SIZE = POLICY_JOINT_ACTION_SIZE


@dataclass(frozen=True)
class PolicyAction:
    """Validated absolute joint targets for one control step."""

    joint_targets: np.ndarray

    def controller_command(self) -> np.ndarray:
        """Return an independent six-value controller command."""

        return self.joint_targets.copy()

    def joint_targets_by_name(self) -> dict[str, float]:
        """Return targets indexed by controller joint name."""

        return {
            name: float(value)
            for name, value in zip(
                POLICY_JOINT_NAMES,
                self.joint_targets,
            )
        }


def parse_policy_action(
    action: Sequence[float],
) -> PolicyAction:
    """Validate one six-dimensional absolute joint action."""

    action_array = np.asarray(
        action,
        dtype=float,
    )

    if action_array.shape != (POLICY_ACTION_SIZE,):
        raise ValueError(
            "Policy action must have shape "
            f"({POLICY_ACTION_SIZE},)."
        )

    if not np.all(np.isfinite(action_array)):
        raise ValueError(
            "Policy action must contain only finite values."
        )

    joint_targets = action_array.copy()
    joint_targets.setflags(write=False)

    return PolicyAction(
        joint_targets=joint_targets,
    )
