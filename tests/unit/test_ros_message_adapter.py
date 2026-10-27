import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments import (
    JointStateCache,
    RgbImageCache,
    TargetCubeStateCache,
    update_joint_cache_from_message,
    update_rgb_image_cache_from_message,
    update_target_cube_cache_from_message,
)


def test_joint_state_message_updates_cache_by_name():
    cache = JointStateCache()

    message = SimpleNamespace(
        name=[
            "gripper",
            "wrist_roll",
            "shoulder_pan",
            "elbow_flex",
            "shoulder_lift",
            "wrist_flex",
        ],
        position=[
            0.6,
            0.5,
            0.1,
            0.3,
            0.2,
            0.4,
        ],
        velocity=[
            1.6,
            1.5,
            1.1,
            1.3,
            1.2,
            1.4,
        ],
    )

    snapshot = update_joint_cache_from_message(
        cache,
        message,
        received_monotonic_s=10.0,
    )

    assert snapshot.positions == {
        "shoulder_pan": 0.1,
        "shoulder_lift": 0.2,
        "elbow_flex": 0.3,
        "wrist_flex": 0.4,
        "wrist_roll": 0.5,
        "gripper": 0.6,
    }

    assert snapshot.velocities == {
        "shoulder_pan": 1.1,
        "shoulder_lift": 1.2,
        "elbow_flex": 1.3,
        "wrist_flex": 1.4,
        "wrist_roll": 1.5,
        "gripper": 1.6,
    }


def test_target_cube_string_message_updates_selected_cube():
    cache = TargetCubeStateCache()

    payload = json.dumps(
        {
            "frame_id": "world",
            "cube_count": 2,
            "cubes": [
                {
                    "name": "target_cube_0001",
                    "entity_id": 101,
                    "position": {
                        "x": 0.20,
                        "y": 0.01,
                        "z": 0.765,
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                },
                {
                    "name": "target_cube_0002",
                    "entity_id": 102,
                    "position": {
                        "x": 0.30,
                        "y": 0.02,
                        "z": 0.765,
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                },
            ],
        }
    )

    snapshot = update_target_cube_cache_from_message(
        cache,
        SimpleNamespace(data=payload),
        target_name="target_cube_0002",
        received_monotonic_s=20.0,
    )

    assert snapshot.name == "target_cube_0002"

    np.testing.assert_allclose(
        snapshot.position,
        [0.30, 0.02, 0.765],
        atol=0.0,
    )


def test_image_message_is_converted_and_timestamped():
    cache = RgbImageCache(
        expected_shape=(1, 2, 3)
    )

    message = SimpleNamespace(
        height=1,
        width=2,
        encoding="bgr8",
        step=6,
        data=bytes(
            [
                30, 20, 10,
                60, 50, 40,
            ]
        ),
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=12,
                nanosec=250_000_000,
            )
        ),
    )

    snapshot = update_rgb_image_cache_from_message(
        cache,
        message,
        received_monotonic_s=30.0,
    )

    np.testing.assert_array_equal(
        snapshot.image,
        [
            [
                [10, 20, 30],
                [40, 50, 60],
            ]
        ],
    )

    assert snapshot.source_timestamp_s == pytest.approx(
        12.25
    )
    assert snapshot.received_monotonic_s == 30.0


def test_malformed_messages_are_rejected():
    with pytest.raises(ValueError, match="JointState"):
        update_joint_cache_from_message(
            JointStateCache(),
            SimpleNamespace(),
        )

    with pytest.raises(ValueError, match="data"):
        update_target_cube_cache_from_message(
            TargetCubeStateCache(),
            SimpleNamespace(),
        )

    with pytest.raises(ValueError, match="required fields"):
        update_rgb_image_cache_from_message(
            RgbImageCache(),
            SimpleNamespace(),
        )
