from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments import (
    image_message_to_rgb_array,
    ros_stamp_to_seconds,
)


def image_message(
    *,
    height,
    width,
    encoding,
    step,
    data,
):
    return SimpleNamespace(
        height=height,
        width=width,
        encoding=encoding,
        step=step,
        data=data,
    )


def test_rgb8_conversion_handles_row_padding():
    message = image_message(
        height=1,
        width=2,
        encoding="rgb8",
        step=8,
        data=bytes([
            10, 20, 30,
            40, 50, 60,
            99, 99,
        ]),
    )

    rgb = image_message_to_rgb_array(message)

    np.testing.assert_array_equal(
        rgb,
        [[[10, 20, 30], [40, 50, 60]]],
    )
    assert rgb.dtype == np.uint8
    assert rgb.flags.writeable is False


def test_bgr8_is_reordered_to_rgb():
    message = image_message(
        height=1,
        width=1,
        encoding="bgr8",
        step=3,
        data=bytes([30, 20, 10]),
    )

    np.testing.assert_array_equal(
        image_message_to_rgb_array(message),
        [[[10, 20, 30]]],
    )


def test_rgba8_discards_alpha():
    message = image_message(
        height=1,
        width=1,
        encoding="rgba8",
        step=4,
        data=bytes([10, 20, 30, 255]),
    )

    np.testing.assert_array_equal(
        image_message_to_rgb_array(message),
        [[[10, 20, 30]]],
    )


def test_bgra8_reorders_channels_and_discards_alpha():
    message = image_message(
        height=1,
        width=1,
        encoding="bgra8",
        step=4,
        data=bytes([30, 20, 10, 255]),
    )

    np.testing.assert_array_equal(
        image_message_to_rgb_array(message),
        [[[10, 20, 30]]],
    )


def test_mono8_is_repeated_across_rgb_channels():
    message = image_message(
        height=1,
        width=2,
        encoding="mono8",
        step=2,
        data=bytes([10, 40]),
    )

    np.testing.assert_array_equal(
        image_message_to_rgb_array(message),
        [[[10, 10, 10], [40, 40, 40]]],
    )


def test_invalid_encoding_step_and_data_length_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        image_message_to_rgb_array(
            image_message(
                height=1,
                width=1,
                encoding="16UC1",
                step=2,
                data=bytes([0, 0]),
            )
        )

    with pytest.raises(ValueError, match="step"):
        image_message_to_rgb_array(
            image_message(
                height=1,
                width=2,
                encoding="rgb8",
                step=5,
                data=bytes(5),
            )
        )

    with pytest.raises(ValueError, match="shorter"):
        image_message_to_rgb_array(
            image_message(
                height=2,
                width=1,
                encoding="rgb8",
                step=3,
                data=bytes(3),
            )
        )


def test_ros_stamp_conversion():
    stamp = SimpleNamespace(
        sec=123,
        nanosec=500_000_000,
    )

    assert ros_stamp_to_seconds(stamp) == pytest.approx(
        123.5
    )

    with pytest.raises(ValueError, match="nanosec"):
        ros_stamp_to_seconds(
            SimpleNamespace(
                sec=1,
                nanosec=1_000_000_000,
            )
        )
