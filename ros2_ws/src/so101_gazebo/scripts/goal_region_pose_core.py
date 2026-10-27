"""Pure helpers for resolving the goal-region centre in the world frame."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Tuple


FLOAT_PATTERN = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class PoseSample:
    """One pose entry parsed from an Ignition Pose_V text message."""

    name: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


def _extract_nested_block(
    text: str,
    field_name: str,
) -> str | None:
    match = re.search(
        rf"\b{re.escape(field_name)}\s*\{{(.*?)\}}",
        text,
        re.DOTALL,
    )

    if match is None:
        return None

    return match.group(1)


def _extract_float(
    text: str,
    field_name: str,
    *,
    default: float,
) -> float:
    match = re.search(
        rf"\b{re.escape(field_name)}:\s*"
        rf"({FLOAT_PATTERN})",
        text,
    )

    if match is None:
        return default

    return float(match.group(1))


def parse_pose_block(block: str) -> PoseSample:
    """Parse one ``pose { ... }`` block from protobuf text output."""

    name_match = re.search(
        r'\bname:\s*"([^"]+)"',
        block,
    )

    if name_match is None:
        raise ValueError("Pose block does not contain an entity name")

    position = _extract_nested_block(block, "position") or ""
    orientation = _extract_nested_block(block, "orientation") or ""

    sample = PoseSample(
        name=name_match.group(1),
        x=_extract_float(position, "x", default=0.0),
        y=_extract_float(position, "y", default=0.0),
        z=_extract_float(position, "z", default=0.0),
        qx=_extract_float(orientation, "x", default=0.0),
        qy=_extract_float(orientation, "y", default=0.0),
        qz=_extract_float(orientation, "z", default=0.0),
        qw=_extract_float(orientation, "w", default=1.0),
    )

    values = (
        sample.x,
        sample.y,
        sample.z,
        sample.qx,
        sample.qy,
        sample.qz,
        sample.qw,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Pose block contains a non-finite value")

    return sample


def rotate_vector(
    quaternion: Tuple[float, float, float, float],
    vector: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Rotate a vector by a quaternion in ``x, y, z, w`` order."""

    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector

    norm = math.sqrt(
        qx * qx
        + qy * qy
        + qz * qz
        + qw * qw
    )

    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Parent pose contains an invalid quaternion")

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    dot_uv = qx * vx + qy * vy + qz * vz
    dot_uu = qx * qx + qy * qy + qz * qz

    cross_x = qy * vz - qz * vy
    cross_y = qz * vx - qx * vz
    cross_z = qx * vy - qy * vx

    scale_vector = qw * qw - dot_uu

    return (
        2.0 * dot_uv * qx
        + scale_vector * vx
        + 2.0 * qw * cross_x,
        2.0 * dot_uv * qy
        + scale_vector * vy
        + 2.0 * qw * cross_y,
        2.0 * dot_uv * qz
        + scale_vector * vz
        + 2.0 * qw * cross_z,
    )


def transform_point(
    parent_pose: PoseSample,
    local_pose: PoseSample,
) -> Tuple[float, float, float]:
    """Transform the child-link origin into the world frame."""

    rotated = rotate_vector(
        (
            parent_pose.qx,
            parent_pose.qy,
            parent_pose.qz,
            parent_pose.qw,
        ),
        (
            local_pose.x,
            local_pose.y,
            local_pose.z,
        ),
    )

    return (
        parent_pose.x + rotated[0],
        parent_pose.y + rotated[1],
        parent_pose.z + rotated[2],
    )
