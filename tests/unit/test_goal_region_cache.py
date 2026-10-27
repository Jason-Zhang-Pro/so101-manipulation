import math

import pytest

from so101_capstone.environments.simulation_runtime import (
    So101SimulationRuntime,
)
from so101_capstone.environments.state_cache import (
    GoalRegionCache,
    GoalRegionSnapshot,
)


def test_goal_region_cache_stores_world_frame_xyz():
    cache = GoalRegionCache()

    snapshot = cache.update_once(
        frame_id="world",
        x=0.078,
        y=-0.300,
        z=0.7515,
        received_monotonic_s=12.0,
    )

    assert snapshot.frame_id == "world"
    assert snapshot.x == pytest.approx(0.078)
    assert snapshot.y == pytest.approx(-0.300)
    assert snapshot.z == pytest.approx(0.7515)
    assert snapshot.received_monotonic_s == 12.0


def test_goal_region_cache_keeps_first_valid_value():
    cache = GoalRegionCache()

    first = cache.update_once(
        frame_id="world",
        x=0.078,
        y=-0.300,
        z=0.7515,
    )

    second = cache.update_once(
        frame_id="world",
        x=9.0,
        y=8.0,
        z=7.0,
    )

    assert second == first
    assert cache.latest() == first


def test_goal_region_cache_rejects_non_world_frame():
    cache = GoalRegionCache()

    with pytest.raises(
        ValueError,
        match="frame_id",
    ):
        cache.update_once(
            frame_id="lab_table",
            x=0.0,
            y=0.0,
            z=0.0,
        )


@pytest.mark.parametrize(
    "coordinates",
    [
        (math.nan, 0.0, 0.0),
        (0.0, math.inf, 0.0),
        (0.0, 0.0, -math.inf),
        (True, 0.0, 0.0),
    ],
)
def test_goal_region_cache_rejects_invalid_coordinates(
    coordinates,
):
    cache = GoalRegionCache()

    with pytest.raises(ValueError):
        cache.update_once(
            frame_id="world",
            x=coordinates[0],
            y=coordinates[1],
            z=coordinates[2],
        )


def test_goal_region_cache_wait_returns_cached_value():
    cache = GoalRegionCache()

    expected = cache.update_once(
        frame_id="world",
        x=0.078,
        y=-0.300,
        z=0.7515,
    )

    assert cache.wait_until_available(0.0) == expected


def test_goal_region_cache_wait_times_out_when_empty():
    cache = GoalRegionCache()

    assert cache.wait_until_available(0.001) is None


def test_runtime_exposes_frozen_goal_region_snapshot():
    snapshot = GoalRegionSnapshot(
        frame_id="world",
        x=0.078,
        y=-0.300,
        z=0.7515,
        received_monotonic_s=1.0,
    )

    runtime = So101SimulationRuntime(
        io_node=object(),
        control_node=object(),
        executor=object(),
        executor_thread=object(),
        world_controller=object(),
        cube_controller=object(),
        reset_manager=object(),
        coordinator=object(),
        goal_region_snapshot=snapshot,
    )

    assert runtime.goal_region_snapshot is snapshot
    assert runtime.goal_region_snapshot.x == pytest.approx(
        0.078
    )
