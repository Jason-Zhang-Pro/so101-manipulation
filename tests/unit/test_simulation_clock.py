import math

import pytest

from so101_capstone.environments import (
    SimulationClockCache,
)


def test_clock_cache_stores_sequence_and_time():
    cache = SimulationClockCache()

    first = cache.update(
        1.25,
        received_monotonic_s=10.0,
    )
    second = cache.update(
        1.251,
        received_monotonic_s=10.1,
    )

    assert first.sequence == 1
    assert first.sim_time_s == pytest.approx(1.25)
    assert second.sequence == 2
    assert second.sim_time_s == pytest.approx(1.251)

    latest = cache.latest()

    assert latest is not None
    assert latest.sequence == 2
    assert latest.received_monotonic_s == pytest.approx(
        10.1
    )


def test_wait_for_newer_sequence():
    cache = SimulationClockCache()

    cache.update(
        2.0,
        received_monotonic_s=10.0,
    )

    snapshot = cache.wait_for_sequence_greater_than(
        0,
        0.0,
    )

    assert snapshot is not None
    assert snapshot.sequence == 1

    assert (
        cache.wait_for_sequence_greater_than(
            1,
            0.0,
        )
        is None
    )


def test_wait_until_target_simulation_time():
    cache = SimulationClockCache()

    first = cache.update(
        3.0,
        received_monotonic_s=10.0,
    )
    cache.update(
        3.033,
        received_monotonic_s=10.1,
    )

    result = cache.wait_until_at_least(
        3.032,
        0.0,
        after_sequence=first.sequence,
    )

    assert result is not None
    assert result.sequence == 2
    assert result.sim_time_s == pytest.approx(3.033)


def test_target_wait_requires_new_sequence():
    cache = SimulationClockCache()

    latest = cache.update(
        4.0,
        received_monotonic_s=10.0,
    )

    result = cache.wait_until_at_least(
        3.0,
        0.0,
        after_sequence=latest.sequence,
    )

    assert result is None


def test_clock_reset_to_zero_is_allowed():
    cache = SimulationClockCache()

    cache.update(
        5.0,
        received_monotonic_s=10.0,
    )
    reset = cache.update(
        0.0,
        received_monotonic_s=10.1,
    )

    assert reset.sequence == 2
    assert reset.sim_time_s == 0.0


@pytest.mark.parametrize(
    "value",
    [
        -0.001,
        math.inf,
        math.nan,
        True,
    ],
)
def test_invalid_simulation_time_is_rejected(value):
    cache = SimulationClockCache()

    with pytest.raises(ValueError):
        cache.update(value)


@pytest.mark.parametrize(
    "timeout_s",
    [
        -1.0,
        math.nan,
    ],
)
def test_invalid_timeout_is_rejected(timeout_s):
    cache = SimulationClockCache()

    with pytest.raises(ValueError):
        cache.wait_for_sequence_greater_than(
            0,
            timeout_s,
        )

def test_wait_until_at_least_accepts_float_roundoff_below_target():
    from so101_capstone.environments.simulation_clock import (
        SimulationClockCache,
    )

    cache = SimulationClockCache()

    # Gazebo/ROS timestamp represented directly from nanoseconds.
    actual_time_s = 0.037
    cache.update(actual_time_s)

    # Independently calculated expected time is very slightly larger:
    # 0.037000000000000005 on standard binary floating point.
    expected_time_s = 0.003 + 34 * 0.001

    assert actual_time_s < expected_time_s

    snapshot = cache.wait_until_at_least(
        expected_time_s,
        0.0,
        after_sequence=0,
    )

    assert snapshot is not None
    assert snapshot.sim_time_s == actual_time_s


def test_wait_until_at_least_rejects_difference_beyond_tolerance():
    from so101_capstone.environments.simulation_clock import (
        SimulationClockCache,
    )

    cache = SimulationClockCache()
    cache.update(0.036999)

    snapshot = cache.wait_until_at_least(
        0.037,
        0.0,
        after_sequence=0,
    )

    assert snapshot is None
