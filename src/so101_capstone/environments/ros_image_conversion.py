"""Convert ROS-like Image messages to RGB NumPy arrays.

This module intentionally avoids cv_bridge because the current ROS Humble
cv_bridge binary is incompatible with NumPy 2.x in the active environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_SUPPORTED_ENCODINGS = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
}


def ros_stamp_to_seconds(stamp: Any) -> float:
    """Convert an object with ``sec`` and ``nanosec`` fields to seconds."""

    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "ROS stamp must provide integer sec and nanosec fields."
        ) from error

    if nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError(
            "ROS stamp nanosec must be within [0, 1_000_000_000)."
        )

    result = sec + nanosec * 1e-9

    if not np.isfinite(result):
        raise ValueError("ROS timestamp must be finite.")

    return float(result)


def image_message_to_rgb_array(message: Any) -> np.ndarray:
    """Convert a ROS-like sensor_msgs/Image message to HWC RGB uint8.

    Supported encodings:

    - rgb8
    - bgr8
    - rgba8
    - bgra8
    - mono8

    Row padding described by ``message.step`` is handled correctly.
    """

    try:
        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        encoding = str(message.encoding).lower()
        raw_data = message.data
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Image message is missing required fields."
        ) from error

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image height and width must be positive."
        )

    if encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(
            f"Unsupported image encoding: {encoding}."
        )

    channels = _SUPPORTED_ENCODINGS[encoding]
    minimum_row_bytes = width * channels

    if step < minimum_row_bytes:
        raise ValueError(
            "Image step is smaller than the encoded row width."
        )

    try:
        data = np.frombuffer(
            bytes(raw_data),
            dtype=np.uint8,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Image data must be bytes-like."
        ) from error

    required_bytes = height * step

    if data.size < required_bytes:
        raise ValueError(
            "Image data is shorter than height * step."
        )

    rows = data[:required_bytes].reshape(height, step)
    packed = rows[:, :minimum_row_bytes].reshape(
        height,
        width,
        channels,
    )

    if encoding == "rgb8":
        rgb = packed
    elif encoding == "bgr8":
        rgb = packed[:, :, [2, 1, 0]]
    elif encoding == "rgba8":
        rgb = packed[:, :, :3]
    elif encoding == "bgra8":
        rgb = packed[:, :, [2, 1, 0]]
    else:
        rgb = np.repeat(
            packed,
            repeats=3,
            axis=2,
        )

    result = np.ascontiguousarray(
        rgb,
        dtype=np.uint8,
    )
    result.setflags(write=False)
    return result
