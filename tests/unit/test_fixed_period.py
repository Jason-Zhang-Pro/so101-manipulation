import pytest

from so101_capstone.environments import (
    FixedPeriodScheduler,
)


class FakeClock:
    def __init__(self, initial_time=0.0):
        self.time = float(initial_time)

    def __call__(self):
        return self.time

    def sleep(self, duration_s):
        self.time += float(duration_s)

    def advance(self, duration_s):
        self.time += float(duration_s)


def test_scheduler_uses_30_hz_period():
    clock = FakeClock(initial_time=10.0)

    scheduler = FixedPeriodScheduler(
        control_hz=30.0,
        clock=clock,
        sleeper=clock.sleep,
    )
    scheduler.reset()

    timing = scheduler.wait_next()

    assert scheduler.control_period_s == pytest.approx(
        1.0 / 30.0
    )
    assert timing.step_index == 1
    assert timing.scheduled_monotonic_s == pytest.approx(
        10.0 + 1.0 / 30.0
    )
    assert timing.actual_monotonic_s == pytest.approx(
        timing.scheduled_monotonic_s
    )
    assert timing.slept_s == pytest.approx(1.0 / 30.0)
    assert timing.lateness_s == pytest.approx(0.0)
    assert timing.missed_periods == 0


def test_processing_time_does_not_accumulate_schedule_drift():
    clock = FakeClock(initial_time=5.0)

    scheduler = FixedPeriodScheduler(
        control_hz=30.0,
        clock=clock,
        sleeper=clock.sleep,
    )
    scheduler.reset()

    first = scheduler.wait_next()

    # Simulate 10 ms of policy inference and environment processing.
    clock.advance(0.010)

    second = scheduler.wait_next()

    assert second.scheduled_monotonic_s == pytest.approx(
        first.scheduled_monotonic_s + 1.0 / 30.0
    )
    assert second.slept_s == pytest.approx(
        1.0 / 30.0 - 0.010
    )


def test_large_overrun_skips_missed_deadlines():
    clock = FakeClock(initial_time=0.0)

    scheduler = FixedPeriodScheduler(
        control_hz=30.0,
        clock=clock,
        sleeper=clock.sleep,
    )
    scheduler.reset()

    # Execution reaches 120 ms before the first wait.
    clock.advance(0.120)

    late = scheduler.wait_next()

    assert late.slept_s == pytest.approx(0.0)
    assert late.lateness_s == pytest.approx(
        0.120 - 1.0 / 30.0
    )
    assert late.missed_periods == 2

    next_timing = scheduler.wait_next()

    assert next_timing.scheduled_monotonic_s == pytest.approx(
        4.0 / 30.0
    )
    assert next_timing.actual_monotonic_s == pytest.approx(
        4.0 / 30.0
    )


def test_wait_requires_explicit_reset():
    clock = FakeClock()

    scheduler = FixedPeriodScheduler(
        clock=clock,
        sleeper=clock.sleep,
    )

    with pytest.raises(
        RuntimeError,
        match=r"reset\(\)",
    ):
        scheduler.wait_next()


def test_invalid_frequency_and_start_time_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        FixedPeriodScheduler(control_hz=0.0)

    clock = FakeClock()

    scheduler = FixedPeriodScheduler(
        clock=clock,
        sleeper=clock.sleep,
    )

    with pytest.raises(ValueError, match="finite"):
        scheduler.reset(start_monotonic_s=float("nan"))
