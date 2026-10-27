"""Tests for complete target-cube collection state caching."""

import pytest

from so101_capstone.environments.state_cache import (
    TargetCubeCollectionCache,
)


def test_empty_collection_is_a_valid_snapshot():
    cache = TargetCubeCollectionCache()

    snapshot = cache.update(
        (),
        source_sequence=10,
        source_sim_time_s=1.25,
        received_monotonic_s=5.0,
    )

    assert snapshot.cube_count == 0
    assert snapshot.cube_names == ()
    assert snapshot.sequence == 1
    assert snapshot.source_sequence == 10
    assert snapshot.source_sim_time_s == pytest.approx(1.25)
    assert snapshot.age_s(5.4) == pytest.approx(0.4)


def test_active_collection_replaces_empty_collection():
    cache = TargetCubeCollectionCache()

    empty = cache.update(
        (),
        source_sequence=20,
        source_sim_time_s=2.0,
        received_monotonic_s=10.0,
    )
    active = cache.update(
        ("target_cube_0001",),
        source_sequence=21,
        source_sim_time_s=2.033,
        received_monotonic_s=10.1,
    )

    latest = cache.latest()

    assert latest is not None
    assert latest.sequence > empty.sequence
    assert latest == active
    assert latest.cube_count == 1
    assert latest.cube_names == ("target_cube_0001",)


def test_wait_accepts_already_available_newer_collection():
    cache = TargetCubeCollectionCache()

    previous = cache.update(
        (),
        source_sequence=30,
        source_sim_time_s=3.0,
    )
    expected = cache.update(
        ("target_cube_0001",),
        source_sequence=31,
        source_sim_time_s=3.033,
    )

    received = cache.wait_for_sequence_greater_than(
        previous.sequence,
        0.0,
    )

    assert received == expected


@pytest.mark.parametrize(
    "cube_names",
    [
        ("",),
        ("target_cube_0001", "target_cube_0001"),
    ],
)
def test_invalid_cube_names_are_rejected(cube_names):
    cache = TargetCubeCollectionCache()

    with pytest.raises(ValueError):
        cache.update(cube_names)


@pytest.mark.parametrize(
    ("source_sequence", "source_sim_time_s"),
    [
        (-1, 0.0),
        (True, 0.0),
        (0, -0.001),
        (0, float("nan")),
        (0, True),
    ],
)
def test_invalid_source_metadata_is_rejected(
    source_sequence,
    source_sim_time_s,
):
    cache = TargetCubeCollectionCache()

    with pytest.raises(ValueError):
        cache.update(
            (),
            source_sequence=source_sequence,
            source_sim_time_s=source_sim_time_s,
        )
