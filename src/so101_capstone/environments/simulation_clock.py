"""Thread-safe cache for Gazebo simulation clock messages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Condition
from time import monotonic


# ROS timestamps have nanosecond resolution. Converting sec/nanosec fields
# and independently calculating expected times with binary floating-point
# arithmetic can differ by a few machine epsilon.
SIM_TIME_COMPARISON_TOLERANCE_S = 1e-9


def _non_negative_finite(
    value: float,
    *,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be finite and non-negative."
        )

    numeric = float(value)

    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"{name} must be finite and non-negative."
        )

    return numeric


def _validate_sequence(
    value: int,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer."
        )

    return value


@dataclass(frozen=True)
class SimulationClockSnapshot:
    """One received Gazebo simulation-clock sample."""

    sim_time_s: float
    sequence: int
    received_monotonic_s: float

    def age_s(
        self,
        now_monotonic_s: float | None = None,
    ) -> float:
        now = (
            monotonic()
            if now_monotonic_s is None
            else _non_negative_finite(
                now_monotonic_s,
                name="now_monotonic_s",
            )
        )

        return max(
            0.0,
            now - self.received_monotonic_s,
        )


def _copy_snapshot(
    snapshot: SimulationClockSnapshot,
) -> SimulationClockSnapshot:
    return SimulationClockSnapshot(
        sim_time_s=snapshot.sim_time_s,
        sequence=snapshot.sequence,
        received_monotonic_s=(
            snapshot.received_monotonic_s
        ),
    )


class SimulationClockCache:
    """Store and wait for Gazebo `/clock` updates."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._snapshot: SimulationClockSnapshot | None = None
        self._next_sequence = 1

    def update(
        self,
        sim_time_s: float,
        *,
        received_monotonic_s: float | None = None,
    ) -> SimulationClockSnapshot:
        """Store one simulation-clock update.

        Simulation time is allowed to move backwards because a Gazebo world
        reset may legitimately return the clock to zero.
        """

        validated_sim_time = _non_negative_finite(
            sim_time_s,
            name="sim_time_s",
        )

        received_time = (
            monotonic()
            if received_monotonic_s is None
            else _non_negative_finite(
                received_monotonic_s,
                name="received_monotonic_s",
            )
        )

        with self._condition:
            snapshot = SimulationClockSnapshot(
                sim_time_s=validated_sim_time,
                sequence=self._next_sequence,
                received_monotonic_s=received_time,
            )

            self._next_sequence += 1
            self._snapshot = snapshot
            self._condition.notify_all()

            return _copy_snapshot(snapshot)

    def latest(
        self,
    ) -> SimulationClockSnapshot | None:
        """Return an independent copy of the latest clock."""

        with self._condition:
            if self._snapshot is None:
                return None

            return _copy_snapshot(self._snapshot)

    def wait_for_sequence_greater_than(
        self,
        sequence: int,
        timeout_s: float,
    ) -> SimulationClockSnapshot | None:
        """Wait for a clock message newer than a local sequence."""

        validated_sequence = _validate_sequence(
            sequence,
            name="sequence",
        )
        timeout = _non_negative_finite(
            timeout_s,
            name="timeout_s",
        )
        deadline = monotonic() + timeout

        with self._condition:
            while (
                self._snapshot is None
                or self._snapshot.sequence
                <= validated_sequence
            ):
                remaining = deadline - monotonic()

                if remaining <= 0.0:
                    return None

                self._condition.wait(timeout=remaining)

            return _copy_snapshot(self._snapshot)

    def wait_until_at_least(
        self,
        target_sim_time_s: float,
        timeout_s: float,
        *,
        after_sequence: int | None = None,
    ) -> SimulationClockSnapshot | None:
        """Wait until a new clock reaches a target simulation time."""

        target = _non_negative_finite(
            target_sim_time_s,
            name="target_sim_time_s",
        )
        timeout = _non_negative_finite(
            timeout_s,
            name="timeout_s",
        )

        validated_after_sequence = (
            None
            if after_sequence is None
            else _validate_sequence(
                after_sequence,
                name="after_sequence",
            )
        )

        deadline = monotonic() + timeout

        with self._condition:
            while True:
                sequence_is_new = (
                    self._snapshot is not None
                    and (
                        validated_after_sequence is None
                        or self._snapshot.sequence
                        > validated_after_sequence
                    )
                )
                target_reached = (
                    self._snapshot is not None
                    and (
                        self._snapshot.sim_time_s >= target
                        or math.isclose(
                            self._snapshot.sim_time_s,
                            target,
                            rel_tol=0.0,
                            abs_tol=(
                                SIM_TIME_COMPARISON_TOLERANCE_S
                            ),
                        )
                    )
                )

                if sequence_is_new and target_reached:
                    return _copy_snapshot(
                        self._snapshot
                    )

                remaining = deadline - monotonic()

                if remaining <= 0.0:
                    return None

                self._condition.wait(timeout=remaining)
