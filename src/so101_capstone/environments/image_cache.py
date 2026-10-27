"""Thread-safe RGB image caches and two-camera synchronization.

ROS image callbacks will convert incoming images to RGB ``uint8`` arrays before
calling this module. This module itself does not depend on ROS.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Condition
from time import monotonic

import numpy as np


def _finite_time(
    value: float | None,
    *,
    name: str,
    use_monotonic_default: bool,
) -> float | None:
    if value is None:
        return monotonic() if use_monotonic_default else None

    result = float(value)

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _non_negative_finite(
    value: float,
    *,
    name: str,
) -> float:
    result = float(value)

    if not np.isfinite(result) or result < 0.0:
        raise ValueError(
            f"{name} must be finite and non-negative."
        )

    return result


@dataclass(frozen=True)
class ImageFrameSnapshot:
    """One complete RGB frame received from a camera."""

    image: np.ndarray
    sequence: int
    received_monotonic_s: float
    source_timestamp_s: float | None

    def age_s(
        self,
        now_monotonic_s: float | None = None,
    ) -> float:
        now = (
            monotonic()
            if now_monotonic_s is None
            else float(now_monotonic_s)
        )

        if not np.isfinite(now):
            raise ValueError(
                "now_monotonic_s must be finite."
            )

        return max(
            0.0,
            now - self.received_monotonic_s,
        )


@dataclass(frozen=True)
class SynchronizedCameraFrames:
    """One synchronized top/gripper RGB frame pair."""

    top: ImageFrameSnapshot
    gripper: ImageFrameSnapshot
    skew_s: float
    used_source_timestamps: bool


def _copy_snapshot(
    snapshot: ImageFrameSnapshot,
) -> ImageFrameSnapshot:
    image = snapshot.image.copy()
    image.setflags(write=False)

    return ImageFrameSnapshot(
        image=image,
        sequence=snapshot.sequence,
        received_monotonic_s=snapshot.received_monotonic_s,
        source_timestamp_s=snapshot.source_timestamp_s,
    )


class RgbImageCache:
    """Thread-safe bounded history of complete RGB frames."""

    def __init__(
        self,
        *,
        max_history: int = 8,
        expected_shape: Sequence[int] | None = None,
    ) -> None:
        if (
            isinstance(max_history, bool)
            or not isinstance(max_history, int)
            or max_history <= 0
        ):
            raise ValueError(
                "max_history must be a positive integer."
            )

        if expected_shape is None:
            validated_shape = None
        else:
            validated_shape = tuple(
                int(value)
                for value in expected_shape
            )

            if (
                len(validated_shape) != 3
                or validated_shape[0] <= 0
                or validated_shape[1] <= 0
                or validated_shape[2] != 3
            ):
                raise ValueError(
                    "expected_shape must be (height, width, 3)."
                )

        self._condition = Condition()
        self._history: deque[ImageFrameSnapshot] = deque(
            maxlen=max_history
        )
        self._expected_shape = validated_shape
        self._next_sequence = 1

    @property
    def expected_shape(self) -> tuple[int, int, int] | None:
        return self._expected_shape

    def update(
        self,
        image: np.ndarray,
        *,
        received_monotonic_s: float | None = None,
        source_timestamp_s: float | None = None,
    ) -> ImageFrameSnapshot:
        """Validate and store one complete RGB frame."""

        image_array = np.asarray(image)

        if image_array.dtype != np.uint8:
            raise ValueError(
                "RGB image must have dtype uint8."
            )

        if (
            image_array.ndim != 3
            or image_array.shape[2] != 3
        ):
            raise ValueError(
                "RGB image must have shape "
                "(height, width, 3)."
            )

        if (
            self._expected_shape is not None
            and image_array.shape != self._expected_shape
        ):
            raise ValueError(
                "RGB image shape does not match "
                f"expected shape {self._expected_shape}."
            )

        received_time = _finite_time(
            received_monotonic_s,
            name="received_monotonic_s",
            use_monotonic_default=True,
        )
        source_time = _finite_time(
            source_timestamp_s,
            name="source_timestamp_s",
            use_monotonic_default=False,
        )

        assert received_time is not None

        stored_image = image_array.copy()
        stored_image.setflags(write=False)

        with self._condition:
            snapshot = ImageFrameSnapshot(
                image=stored_image,
                sequence=self._next_sequence,
                received_monotonic_s=received_time,
                source_timestamp_s=source_time,
            )
            self._next_sequence += 1
            self._history.append(snapshot)
            self._condition.notify_all()

            return _copy_snapshot(snapshot)

    def latest(self) -> ImageFrameSnapshot | None:
        """Return an independent copy of the newest frame."""

        with self._condition:
            if not self._history:
                return None

            return _copy_snapshot(self._history[-1])

    def history(self) -> tuple[ImageFrameSnapshot, ...]:
        """Return independent copies from oldest to newest."""

        with self._condition:
            return tuple(
                _copy_snapshot(snapshot)
                for snapshot in self._history
            )

    def _history_refs_for_selection(
        self,
    ) -> tuple[ImageFrameSnapshot, ...]:
        """Return read-only frame references for internal pair selection.

        Stored image arrays are already read-only. Holding snapshot references
        is safe even if the bounded deque later evicts an entry. Public cache
        APIs continue returning independent image copies.
        """

        with self._condition:
            return tuple(self._history)

    def wait_for_sequence_greater_than(
        self,
        sequence: int,
        timeout_s: float,
    ) -> ImageFrameSnapshot | None:
        """Wait until a frame newer than ``sequence`` exists."""

        if sequence < 0:
            raise ValueError(
                "sequence must be non-negative."
            )

        timeout = _non_negative_finite(
            timeout_s,
            name="timeout_s",
        )
        deadline = monotonic() + timeout

        with self._condition:
            while (
                not self._history
                or self._history[-1].sequence <= sequence
            ):
                remaining = deadline - monotonic()

                if remaining <= 0.0:
                    return None

                self._condition.wait(timeout=remaining)

            return _copy_snapshot(self._history[-1])


def select_synchronized_camera_frames(
    top_cache: RgbImageCache,
    gripper_cache: RgbImageCache,
    *,
    now_monotonic_s: float,
    max_age_s: float = 0.2,
    max_skew_s: float = 0.05,
    after_top_sequence: int | None = None,
    after_gripper_sequence: int | None = None,
    min_source_timestamp_s: float | None = None,
    max_source_timestamp_s: float | None = None,
) -> SynchronizedCameraFrames:
    """Select the closest fresh top/gripper frame pair.

    Source timestamps are used when both frames provide them. Otherwise the
    local monotonic receive times are compared.
    """

    now = float(now_monotonic_s)

    if not np.isfinite(now):
        raise ValueError(
            "now_monotonic_s must be finite."
        )

    maximum_age = _non_negative_finite(
        max_age_s,
        name="max_age_s",
    )
    maximum_skew = _non_negative_finite(
        max_skew_s,
        name="max_skew_s",
    )

    minimum_source_timestamp = (
        None
        if min_source_timestamp_s is None
        else _non_negative_finite(
            min_source_timestamp_s,
            name="min_source_timestamp_s",
        )
    )
    maximum_source_timestamp = (
        None
        if max_source_timestamp_s is None
        else _non_negative_finite(
            max_source_timestamp_s,
            name="max_source_timestamp_s",
        )
    )

    if (
        minimum_source_timestamp is not None
        and maximum_source_timestamp is not None
        and minimum_source_timestamp
        > maximum_source_timestamp
    ):
        raise ValueError(
            "min_source_timestamp_s must not exceed "
            "max_source_timestamp_s."
        )

    def source_timestamp_is_allowed(
        frame: ImageFrameSnapshot,
    ) -> bool:
        if (
            minimum_source_timestamp is None
            and maximum_source_timestamp is None
        ):
            return True

        if frame.source_timestamp_s is None:
            return False

        source_time = float(frame.source_timestamp_s)

        if (
            minimum_source_timestamp is not None
            and source_time < minimum_source_timestamp
        ):
            return False

        if (
            maximum_source_timestamp is not None
            and source_time > maximum_source_timestamp
        ):
            return False

        return True

    sequence_limits = {
        "after_top_sequence": after_top_sequence,
        "after_gripper_sequence": after_gripper_sequence,
    }

    for name, value in sequence_limits.items():
        if value is None:
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer "
                "or None."
            )

    top_history = tuple(
        frame
        for frame in top_cache._history_refs_for_selection()
        if (
            frame.age_s(now) <= maximum_age
            and (
                after_top_sequence is None
                or frame.sequence > after_top_sequence
            )
            and source_timestamp_is_allowed(frame)
        )
    )
    gripper_history = tuple(
        frame
        for frame in gripper_cache._history_refs_for_selection()
        if (
            frame.age_s(now) <= maximum_age
            and (
                after_gripper_sequence is None
                or frame.sequence
                > after_gripper_sequence
            )
            and source_timestamp_is_allowed(frame)
        )
    )

    if not top_history:
        raise RuntimeError(
            "No fresh top-camera frame is available."
        )

    if not gripper_history:
        raise RuntimeError(
            "No fresh gripper-camera frame is available."
        )

    candidates: list[
        tuple[
            float,
            float,
            bool,
            ImageFrameSnapshot,
            ImageFrameSnapshot,
        ]
    ] = []

    for top_frame in top_history:
        for gripper_frame in gripper_history:
            use_source_timestamps = (
                top_frame.source_timestamp_s is not None
                and gripper_frame.source_timestamp_s is not None
            )

            if use_source_timestamps:
                top_time = float(
                    top_frame.source_timestamp_s
                )
                gripper_time = float(
                    gripper_frame.source_timestamp_s
                )
            else:
                top_time = top_frame.received_monotonic_s
                gripper_time = (
                    gripper_frame.received_monotonic_s
                )

            skew_s = abs(top_time - gripper_time)

            # Prefer the smallest skew. When skew is equal, prefer the more
            # recently received pair.
            pair_receive_time = max(
                top_frame.received_monotonic_s,
                gripper_frame.received_monotonic_s,
            )

            candidates.append(
                (
                    skew_s,
                    -pair_receive_time,
                    use_source_timestamps,
                    top_frame,
                    gripper_frame,
                )
            )

    (
        skew_s,
        _,
        used_source_timestamps,
        top_frame,
        gripper_frame,
    ) = min(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
        ),
    )

    if skew_s > maximum_skew:
        raise RuntimeError(
            "No synchronized camera pair is available: "
            f"minimum skew is {skew_s:.6f} s."
        )

    return SynchronizedCameraFrames(
        top=_copy_snapshot(top_frame),
        gripper=_copy_snapshot(gripper_frame),
        skew_s=float(skew_s),
        used_source_timestamps=used_source_timestamps,
    )
