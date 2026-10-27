"""Fixed-period scheduling for the SO-101 environment.

The scheduler uses absolute monotonic deadlines so policy inference and reward
calculation time do not accumulate as permanent control-period drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep

import numpy as np


@dataclass(frozen=True)
class FixedPeriodStepTiming:
    """Timing diagnostics for one completed control period."""

    step_index: int
    scheduled_monotonic_s: float
    actual_monotonic_s: float
    slept_s: float
    lateness_s: float
    missed_periods: int


class FixedPeriodScheduler:
    """Schedule environment steps against absolute monotonic deadlines."""

    def __init__(
        self,
        control_hz: float = 30.0,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        frequency = float(control_hz)

        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(
                "control_hz must be finite and positive."
            )

        self._control_hz = frequency
        self._control_period_s = 1.0 / frequency
        self._clock = clock
        self._sleeper = sleeper

        self._next_deadline_s: float | None = None
        self._step_index = 0

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def control_period_s(self) -> float:
        return self._control_period_s

    @property
    def started(self) -> bool:
        return self._next_deadline_s is not None

    def reset(
        self,
        start_monotonic_s: float | None = None,
    ) -> None:
        """Start a new schedule one period after the supplied time."""

        start_time = (
            float(self._clock())
            if start_monotonic_s is None
            else float(start_monotonic_s)
        )

        if not np.isfinite(start_time):
            raise ValueError(
                "start_monotonic_s must be finite."
            )

        self._next_deadline_s = (
            start_time + self._control_period_s
        )
        self._step_index = 0

    def wait_next(self) -> FixedPeriodStepTiming:
        """Wait for the next absolute control deadline.

        When execution is more than one complete period late, missed deadlines
        are skipped instead of issuing several immediate catch-up steps.
        """

        if self._next_deadline_s is None:
            raise RuntimeError(
                "reset() must be called before wait_next()."
            )

        scheduled_time = self._next_deadline_s
        before_wait = float(self._clock())

        if not np.isfinite(before_wait):
            raise RuntimeError(
                "Monotonic clock returned a non-finite value."
            )

        slept_s = 0.0

        if before_wait < scheduled_time:
            slept_s = scheduled_time - before_wait
            self._sleeper(slept_s)

        actual_time = float(self._clock())

        if not np.isfinite(actual_time):
            raise RuntimeError(
                "Monotonic clock returned a non-finite value."
            )

        lateness_s = max(
            0.0,
            actual_time - scheduled_time,
        )

        next_deadline = (
            scheduled_time + self._control_period_s
        )
        missed_periods = 0

        if actual_time >= next_deadline:
            missed_periods = (
                int(
                    (actual_time - next_deadline)
                    // self._control_period_s
                )
                + 1
            )
            next_deadline += (
                missed_periods * self._control_period_s
            )

        self._next_deadline_s = next_deadline
        self._step_index += 1

        return FixedPeriodStepTiming(
            step_index=self._step_index,
            scheduled_monotonic_s=scheduled_time,
            actual_monotonic_s=actual_time,
            slept_s=slept_s,
            lateness_s=lateness_s,
            missed_periods=missed_periods,
        )
