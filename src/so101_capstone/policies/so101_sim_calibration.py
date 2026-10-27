"""SO-101 simulation position calibration.

This module maps between:

- Gazebo joint positions in radians.
- LeRobot SO-101 calibrated policy positions.

The first five joints use the LeRobot range ``[-100, 100]``.
The gripper uses the LeRobot range ``[0, 100]``.

For the provisional simulation calibration, each Gazebo joint's current URDF
lower and upper limits are treated as the calibrated physical endpoints.

The episode HOME pose has no role in this mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


SIM_POLICY_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

SIM_POLICY_JOINT_COUNT = len(SIM_POLICY_JOINT_NAMES)


@dataclass(frozen=True)
class SimJointCalibration:
    """Linear calibration for one simulated joint."""

    name: str
    gazebo_lower_rad: float
    gazebo_upper_rad: float
    policy_lower: float
    policy_upper: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "Calibration joint name must not be empty."
            )

        values = (
            self.gazebo_lower_rad,
            self.gazebo_upper_rad,
            self.policy_lower,
            self.policy_upper,
        )

        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(
                "Calibration limits must be finite."
            )

        if self.gazebo_upper_rad <= self.gazebo_lower_rad:
            raise ValueError(
                "Gazebo upper limit must be greater than "
                "the lower limit."
            )

        if self.policy_upper <= self.policy_lower:
            raise ValueError(
                "Policy upper limit must be greater than "
                "the lower limit."
            )

    def gazebo_to_policy(
        self,
        position_rad: float,
        *,
        clip: bool = True,
    ) -> float:
        """Map one Gazebo position to LeRobot policy space."""

        position = float(position_rad)

        if not math.isfinite(position):
            raise ValueError(
                "Gazebo joint position must be finite."
            )

        if clip:
            position = min(
                self.gazebo_upper_rad,
                max(self.gazebo_lower_rad, position),
            )

        gazebo_fraction = (
            position - self.gazebo_lower_rad
        ) / (
            self.gazebo_upper_rad
            - self.gazebo_lower_rad
        )

        return (
            self.policy_lower
            + gazebo_fraction
            * (self.policy_upper - self.policy_lower)
        )

    def policy_to_gazebo(
        self,
        policy_position: float,
        *,
        clip: bool = True,
    ) -> float:
        """Map one LeRobot policy position to Gazebo radians."""

        position = float(policy_position)

        if not math.isfinite(position):
            raise ValueError(
                "Policy joint position must be finite."
            )

        if clip:
            position = min(
                self.policy_upper,
                max(self.policy_lower, position),
            )

        policy_fraction = (
            position - self.policy_lower
        ) / (
            self.policy_upper
            - self.policy_lower
        )

        return (
            self.gazebo_lower_rad
            + policy_fraction
            * (
                self.gazebo_upper_rad
                - self.gazebo_lower_rad
            )
        )


SIM_JOINT_CALIBRATIONS = (
    SimJointCalibration(
        name="shoulder_pan",
        gazebo_lower_rad=-1.91986,
        gazebo_upper_rad=1.91986,
        policy_lower=-100.0,
        policy_upper=100.0,
    ),
    SimJointCalibration(
        name="shoulder_lift",
        gazebo_lower_rad=-1.74533,
        gazebo_upper_rad=1.74533,
        policy_lower=-100.0,
        policy_upper=100.0,
    ),
    SimJointCalibration(
        name="elbow_flex",
        gazebo_lower_rad=-1.69,
        gazebo_upper_rad=1.69,
        policy_lower=-100.0,
        policy_upper=100.0,
    ),
    SimJointCalibration(
        name="wrist_flex",
        gazebo_lower_rad=-1.65806,
        gazebo_upper_rad=1.65806,
        policy_lower=-100.0,
        policy_upper=100.0,
    ),
    SimJointCalibration(
        name="wrist_roll",
        gazebo_lower_rad=-2.74385,
        gazebo_upper_rad=2.84121,
        policy_lower=-100.0,
        policy_upper=100.0,
    ),
    SimJointCalibration(
        name="gripper",
        gazebo_lower_rad=-0.174533,
        gazebo_upper_rad=1.74533,
        policy_lower=0.0,
        policy_upper=100.0,
    ),
)


if tuple(
    calibration.name
    for calibration in SIM_JOINT_CALIBRATIONS
) != SIM_POLICY_JOINT_NAMES:
    raise RuntimeError(
        "Simulation calibration order does not match "
        "SIM_POLICY_JOINT_NAMES."
    )


def _validated_position_vector(
    positions: Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        positions,
        dtype=float,
    )

    if vector.shape != (SIM_POLICY_JOINT_COUNT,):
        raise ValueError(
            f"{name} must have shape "
            f"({SIM_POLICY_JOINT_COUNT},)."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return vector.copy()


def gazebo_positions_to_policy(
    joint_positions_rad: Sequence[float],
    *,
    clip: bool = True,
) -> np.ndarray:
    """Convert six Gazebo positions to LeRobot policy positions.

    Input and output ordering is always ``SIM_POLICY_JOINT_NAMES``.
    """

    positions = _validated_position_vector(
        joint_positions_rad,
        name="joint_positions_rad",
    )

    result = np.array(
        [
            calibration.gazebo_to_policy(
                position,
                clip=clip,
            )
            for calibration, position in zip(
                SIM_JOINT_CALIBRATIONS,
                positions,
            )
        ],
        dtype=float,
    )

    result.setflags(write=False)
    return result


def policy_positions_to_gazebo(
    policy_positions: Sequence[float],
    *,
    clip: bool = True,
) -> np.ndarray:
    """Convert six LeRobot policy positions to Gazebo radians.

    Input and output ordering is always ``SIM_POLICY_JOINT_NAMES``.
    """

    positions = _validated_position_vector(
        policy_positions,
        name="policy_positions",
    )

    result = np.array(
        [
            calibration.policy_to_gazebo(
                position,
                clip=clip,
            )
            for calibration, position in zip(
                SIM_JOINT_CALIBRATIONS,
                positions,
            )
        ],
        dtype=float,
    )

    result.setflags(write=False)
    return result

# BEGIN SIMULATION CALIBRATION FILE LOADER
from dataclasses import replace as _dataclass_replace
import json as _json
from pathlib import Path as _Path


SIM_CALIBRATION_FILE = (
    _Path(__file__).resolve().parent
    / "calibration"
    / "so101_simulation.json"
)


def _load_sim_joint_calibrations_from_file():
    if not SIM_CALIBRATION_FILE.is_file():
        raise FileNotFoundError(
            "SO-101 simulation calibration file was not found: "
            f"{SIM_CALIBRATION_FILE}"
        )

    payload = _json.loads(
        SIM_CALIBRATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("schema_version") != 1:
        raise ValueError(
            "Unsupported SO-101 simulation calibration "
            "schema_version."
        )

    joint_order = tuple(
        payload.get("joint_order", ())
    )

    if joint_order != tuple(SIM_POLICY_JOINT_NAMES):
        raise ValueError(
            "Simulation calibration joint_order does not "
            "match the policy/controller order."
        )

    joints = payload.get("joints")

    if not isinstance(joints, dict):
        raise ValueError(
            "Simulation calibration must contain a joints object."
        )

    if set(joints) != set(SIM_POLICY_JOINT_NAMES):
        raise ValueError(
            "Simulation calibration joint names do not match "
            "the required SO-101 joint names."
        )

    replace = _dataclass_replace

    updated = [
        replace(
            calibration,
            gazebo_lower_rad=float(
                joints[joint_name]["range_min_rad"]
            ),
            gazebo_upper_rad=float(
                joints[joint_name]["range_max_rad"]
            ),
        )
        for joint_name, calibration in zip(
            SIM_POLICY_JOINT_NAMES,
            SIM_JOINT_CALIBRATIONS,
            strict=True,
        )
    ]
    updated = tuple(updated)


    return updated


SIM_JOINT_CALIBRATIONS = (
    _load_sim_joint_calibrations_from_file()
)
# END SIMULATION CALIBRATION FILE LOADER
