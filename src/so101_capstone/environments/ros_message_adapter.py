"""Adapt ROS-like messages to the environment's pure-Python caches.

The functions use attribute-based access and therefore remain testable without
starting ROS nodes or importing rclpy.
"""

from __future__ import annotations

from typing import Any

from .image_cache import (
    ImageFrameSnapshot,
    RgbImageCache,
)
from .ros_image_conversion import (
    image_message_to_rgb_array,
    ros_stamp_to_seconds,
)
from .state_cache import (
    JointStateCache,
    JointStateSnapshot,
    TargetCubeSnapshot,
    TargetCubeStateCache,
)
from .target_cube_parser import (
    update_target_cube_cache_from_json,
)


def update_joint_cache_from_message(
    cache: JointStateCache,
    message: Any,
    *,
    received_monotonic_s: float | None = None,
) -> JointStateSnapshot:
    """Store one ROS-like ``sensor_msgs/JointState`` message."""

    try:
        names = message.name
        positions = message.position
        velocities = message.velocity
    except AttributeError as error:
        raise ValueError(
            "JointState message is missing name, position, "
            "or velocity."
        ) from error

    header = getattr(message, "header", None)
    stamp = None if header is None else getattr(header, "stamp", None)
    source_time = None if stamp is None else ros_stamp_to_seconds(stamp)
    return cache.update(
        names=names,
        positions=positions,
        velocities=velocities,
        received_monotonic_s=received_monotonic_s,
        source_sim_time_s=source_time,
    )


def update_target_cube_cache_from_message(
    cache: TargetCubeStateCache,
    message: Any,
    *,
    target_name: str | None = None,
    received_monotonic_s: float | None = None,
) -> TargetCubeSnapshot:
    """Store one ROS-like `/target_cube/states` String message."""

    try:
        payload = message.data
    except AttributeError as error:
        raise ValueError(
            "Target cube message is missing data."
        ) from error

    return update_target_cube_cache_from_json(
        cache,
        payload,
        target_name=target_name,
        received_monotonic_s=received_monotonic_s,
    )


def image_source_timestamp_s(
    message: Any,
) -> float | None:
    """Return an Image header timestamp when one is available."""

    header = getattr(message, "header", None)

    if header is None:
        return None

    stamp = getattr(header, "stamp", None)

    if stamp is None:
        return None

    return ros_stamp_to_seconds(stamp)


def update_rgb_image_cache_from_message(
    cache: RgbImageCache,
    message: Any,
    *,
    received_monotonic_s: float | None = None,
) -> ImageFrameSnapshot:
    """Convert and store one ROS-like ``sensor_msgs/Image`` message."""

    image = image_message_to_rgb_array(message)
    source_timestamp = image_source_timestamp_s(message)

    return cache.update(
        image,
        received_monotonic_s=received_monotonic_s,
        source_timestamp_s=source_timestamp,
    )
