import numpy as np
import pytest

from so101_capstone.environments import (
    RgbImageCache,
    select_synchronized_camera_frames,
)


def rgb_frame(value, shape=(4, 6, 3)):
    return np.full(
        shape,
        value,
        dtype=np.uint8,
    )


def test_rgb_cache_stores_complete_frame_and_sequence():
    cache = RgbImageCache()

    snapshot = cache.update(
        rgb_frame(10),
        received_monotonic_s=5.0,
        source_timestamp_s=100.0,
    )

    assert snapshot.sequence == 1
    assert snapshot.received_monotonic_s == 5.0
    assert snapshot.source_timestamp_s == 100.0
    assert snapshot.image.shape == (4, 6, 3)
    assert snapshot.image.dtype == np.uint8
    assert snapshot.image.flags.writeable is False
    assert snapshot.age_s(5.1) == pytest.approx(0.1)


def test_cache_copies_input_and_returns_read_only_frames():
    cache = RgbImageCache()
    original = rgb_frame(20)

    cache.update(
        original,
        received_monotonic_s=1.0,
    )
    original[:] = 99

    latest = cache.latest()

    assert latest is not None
    assert np.all(latest.image == 20)

    with pytest.raises(ValueError):
        latest.image[0, 0, 0] = 1


def test_invalid_dtype_shape_and_expected_shape_are_rejected():
    cache = RgbImageCache(
        expected_shape=(4, 6, 3)
    )

    with pytest.raises(ValueError, match="uint8"):
        cache.update(
            np.zeros((4, 6, 3), dtype=np.float32)
        )

    with pytest.raises(ValueError, match="shape"):
        cache.update(
            np.zeros((4, 6), dtype=np.uint8)
        )

    with pytest.raises(ValueError, match="expected"):
        cache.update(
            np.zeros((5, 6, 3), dtype=np.uint8)
        )


def test_history_is_bounded():
    cache = RgbImageCache(max_history=2)

    cache.update(
        rgb_frame(1),
        received_monotonic_s=1.0,
    )
    cache.update(
        rgb_frame(2),
        received_monotonic_s=2.0,
    )
    cache.update(
        rgb_frame(3),
        received_monotonic_s=3.0,
    )

    history = cache.history()

    assert [frame.sequence for frame in history] == [2, 3]
    assert np.all(history[0].image == 2)
    assert np.all(history[1].image == 3)


def test_sync_selects_closest_source_timestamp_pair():
    top = RgbImageCache()
    gripper = RgbImageCache()

    top.update(
        rgb_frame(10),
        received_monotonic_s=20.00,
        source_timestamp_s=100.000,
    )
    top.update(
        rgb_frame(11),
        received_monotonic_s=20.03,
        source_timestamp_s=100.033,
    )

    gripper.update(
        rgb_frame(20),
        received_monotonic_s=20.01,
        source_timestamp_s=100.005,
    )
    gripper.update(
        rgb_frame(21),
        received_monotonic_s=20.06,
        source_timestamp_s=100.060,
    )

    pair = select_synchronized_camera_frames(
        top,
        gripper,
        now_monotonic_s=20.07,
        max_age_s=0.2,
        max_skew_s=0.05,
    )

    assert np.all(pair.top.image == 10)
    assert np.all(pair.gripper.image == 20)
    assert pair.skew_s == pytest.approx(0.005)
    assert pair.used_source_timestamps is True


def test_sync_falls_back_to_monotonic_receive_times():
    top = RgbImageCache()
    gripper = RgbImageCache()

    top.update(
        rgb_frame(10),
        received_monotonic_s=30.000,
    )
    gripper.update(
        rgb_frame(20),
        received_monotonic_s=30.012,
    )

    pair = select_synchronized_camera_frames(
        top,
        gripper,
        now_monotonic_s=30.02,
        max_age_s=0.2,
        max_skew_s=0.05,
    )

    assert pair.skew_s == pytest.approx(0.012)
    assert pair.used_source_timestamps is False


def test_sync_rejects_stale_frames():
    top = RgbImageCache()
    gripper = RgbImageCache()

    top.update(
        rgb_frame(10),
        received_monotonic_s=1.0,
    )
    gripper.update(
        rgb_frame(20),
        received_monotonic_s=1.0,
    )

    with pytest.raises(RuntimeError, match="fresh"):
        select_synchronized_camera_frames(
            top,
            gripper,
            now_monotonic_s=2.0,
            max_age_s=0.1,
        )


def test_sync_rejects_excessive_camera_skew():
    top = RgbImageCache()
    gripper = RgbImageCache()

    top.update(
        rgb_frame(10),
        received_monotonic_s=10.0,
        source_timestamp_s=50.000,
    )
    gripper.update(
        rgb_frame(20),
        received_monotonic_s=10.0,
        source_timestamp_s=50.100,
    )

    with pytest.raises(RuntimeError, match="skew"):
        select_synchronized_camera_frames(
            top,
            gripper,
            now_monotonic_s=10.01,
            max_age_s=0.2,
            max_skew_s=0.05,
        )


def test_sync_can_require_frames_newer_than_sequences():
    top_cache = RgbImageCache(max_history=4)
    gripper_cache = RgbImageCache(max_history=4)

    image = np.zeros((2, 3, 3), dtype=np.uint8)

    old_top = top_cache.update(
        image,
        received_monotonic_s=100.000,
        source_timestamp_s=50.000,
    )
    old_gripper = gripper_cache.update(
        image,
        received_monotonic_s=100.001,
        source_timestamp_s=50.001,
    )

    new_top = top_cache.update(
        image,
        received_monotonic_s=100.033,
        source_timestamp_s=50.033,
    )
    new_gripper = gripper_cache.update(
        image,
        received_monotonic_s=100.035,
        source_timestamp_s=50.035,
    )

    pair = select_synchronized_camera_frames(
        top_cache,
        gripper_cache,
        now_monotonic_s=100.04,
        max_age_s=0.2,
        max_skew_s=0.01,
        after_top_sequence=old_top.sequence,
        after_gripper_sequence=old_gripper.sequence,
    )

    assert pair.top.sequence == new_top.sequence
    assert pair.gripper.sequence == new_gripper.sequence
    assert pair.skew_s == pytest.approx(0.002)


def test_sync_rejects_when_no_frames_are_new_enough():
    top_cache = RgbImageCache(max_history=4)
    gripper_cache = RgbImageCache(max_history=4)

    image = np.zeros((2, 3, 3), dtype=np.uint8)

    top = top_cache.update(
        image,
        received_monotonic_s=100.0,
        source_timestamp_s=50.0,
    )
    gripper = gripper_cache.update(
        image,
        received_monotonic_s=100.0,
        source_timestamp_s=50.0,
    )

    with pytest.raises(
        RuntimeError,
        match="No fresh top-camera frame",
    ):
        select_synchronized_camera_frames(
            top_cache,
            gripper_cache,
            now_monotonic_s=100.01,
            after_top_sequence=top.sequence,
            after_gripper_sequence=gripper.sequence,
        )


@pytest.mark.parametrize(
    (
        "after_top_sequence",
        "after_gripper_sequence",
    ),
    [
        (-1, None),
        (None, True),
    ],
)
def test_sync_rejects_invalid_sequence_limits(
    after_top_sequence,
    after_gripper_sequence,
):
    top_cache = RgbImageCache(max_history=1)
    gripper_cache = RgbImageCache(max_history=1)

    with pytest.raises(ValueError, match="sequence"):
        select_synchronized_camera_frames(
            top_cache,
            gripper_cache,
            now_monotonic_s=100.0,
            after_top_sequence=after_top_sequence,
            after_gripper_sequence=after_gripper_sequence,
        )


def test_sync_filters_frames_by_source_timestamp_window():
    import numpy as np

    from so101_capstone.environments import (
        RgbImageCache,
        select_synchronized_camera_frames,
    )

    top_cache = RgbImageCache(max_history=8)
    gripper_cache = RgbImageCache(max_history=8)
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    # Delayed old messages: locally new, but from the previous step.
    top_cache.update(
        image,
        source_timestamp_s=9.990,
        received_monotonic_s=20.0,
    )
    gripper_cache.update(
        image,
        source_timestamp_s=9.990,
        received_monotonic_s=20.0,
    )

    # Valid messages generated inside the current step.
    valid_top = top_cache.update(
        image,
        source_timestamp_s=10.030,
        received_monotonic_s=20.1,
    )
    valid_gripper = gripper_cache.update(
        image,
        source_timestamp_s=10.030,
        received_monotonic_s=20.1,
    )

    selected = select_synchronized_camera_frames(
        top_cache,
        gripper_cache,
        now_monotonic_s=20.1,
        max_age_s=1.0,
        max_skew_s=0.01,
        min_source_timestamp_s=10.000,
        max_source_timestamp_s=10.033,
    )

    assert selected.top.sequence == valid_top.sequence
    assert (
        selected.gripper.sequence
        == valid_gripper.sequence
    )
    assert (
        selected.top.source_timestamp_s
        == pytest.approx(10.030)
    )


def test_source_timestamp_window_rejects_untimestamped_frames():
    import numpy as np

    from so101_capstone.environments import (
        RgbImageCache,
        select_synchronized_camera_frames,
    )

    top_cache = RgbImageCache(max_history=8)
    gripper_cache = RgbImageCache(max_history=8)
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    top_cache.update(
        image,
        received_monotonic_s=30.0,
    )
    gripper_cache.update(
        image,
        received_monotonic_s=30.0,
    )

    with pytest.raises(
        RuntimeError,
        match="top-camera",
    ):
        select_synchronized_camera_frames(
            top_cache,
            gripper_cache,
            now_monotonic_s=30.0,
            max_age_s=1.0,
            min_source_timestamp_s=5.0,
            max_source_timestamp_s=5.033,
        )
