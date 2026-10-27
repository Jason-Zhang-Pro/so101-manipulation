import ast
from pathlib import Path
import re

import numpy as np

import so101_capstone.environments.image_cache as image_cache_module
from so101_capstone.environments.image_cache import (
    RgbImageCache,
    select_synchronized_camera_frames,
)


ROOT = Path(__file__).resolve().parents[2]


def _sensor_block(text, name, sensor_type):
    pattern = re.compile(
        rf'<sensor\b(?=[^>]*\bname="{re.escape(name)}")'
        rf'(?=[^>]*\btype="{re.escape(sensor_type)}")[^>]*>'
        rf'.*?</sensor>',
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    assert match is not None
    return match.group(0)


def test_unused_segmentation_is_off_and_rgb_is_45_hz():
    xacro = (
        ROOT
        / "ros2_ws/src/so101_description/urdf/so101.urdf.xacro"
    ).read_text(encoding="utf-8")

    for name in (
        "top_camera_segmentation_sensor",
        "gripper_camera_segmentation_sensor",
    ):
        block = _sensor_block(xacro, name, "segmentation")
        assert re.search(
            r"<always_on>\s*false\s*</always_on>",
            block,
        )

    for name in (
        "top_camera_sensor",
        "gripper_camera_sensor",
    ):
        block = _sensor_block(xacro, name, "camera")
        assert re.search(
            r"<always_on>\s*true\s*</always_on>",
            block,
        )
        assert re.search(
            r"<update_rate>\s*45\s*</update_rate>",
            block,
        )


def test_rgb_subscriptions_preserve_current_queue_depth():
    source = (
        ROOT
        / "src/so101_capstone/environments/ros_reach_node.py"
    ).read_text(encoding="utf-8")

    # Integer QoS depth keeps ROS's existing reliable/volatile defaults.
    # A previous draft of this test expected an unimplemented QoS change.
    subscriptions = {
        "_top_camera_subscription",
        "_gripper_camera_subscription",
    }
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr in subscriptions:
                assert isinstance(node.value, ast.Call)
                assert isinstance(node.value.args[-1], ast.Constant)
                assert node.value.args[-1].value == 10
                found.add(target.attr)
    assert found == subscriptions


def test_pair_search_copies_only_selected_two_frames(monkeypatch):
    top = RgbImageCache(max_history=8)
    gripper = RgbImageCache(max_history=8)

    for index in range(8):
        source_time = 10.0 + index * 0.01
        top.update(
            np.full((2, 3, 3), index, dtype=np.uint8),
            received_monotonic_s=20.0 + index * 0.01,
            source_timestamp_s=source_time,
        )
        gripper.update(
            np.full((2, 3, 3), index + 20, dtype=np.uint8),
            received_monotonic_s=20.001 + index * 0.01,
            source_timestamp_s=source_time + 0.001,
        )

    original_copy = image_cache_module._copy_snapshot
    copied_sequences = []

    def counting_copy(snapshot):
        copied_sequences.append(snapshot.sequence)
        return original_copy(snapshot)

    monkeypatch.setattr(
        image_cache_module,
        "_copy_snapshot",
        counting_copy,
    )

    pair = select_synchronized_camera_frames(
        top,
        gripper,
        now_monotonic_s=20.08,
        max_age_s=1.0,
        max_skew_s=0.01,
    )

    assert len(copied_sequences) == 2
    assert pair.top.image.flags.writeable is False
    assert pair.gripper.image.flags.writeable is False


def test_public_history_still_returns_independent_images():
    cache = RgbImageCache(max_history=2)

    returned = cache.update(
        np.zeros((2, 3, 3), dtype=np.uint8),
        received_monotonic_s=1.0,
        source_timestamp_s=1.0,
    )
    history = cache.history()

    assert len(history) == 1
    assert returned.image is not history[0].image
    assert np.shares_memory(
        returned.image,
        history[0].image,
    ) is False
