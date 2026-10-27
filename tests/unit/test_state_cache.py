import numpy as np
import pytest

from so101_capstone.environments.state_cache import (
    MEASURED_JOINT_NAMES,
    JointStateCache,
    TargetCubeStateCache,
)


def scrambled_joint_feedback():
    names = [
        "gripper",
        "wrist_roll",
        "shoulder_pan",
        "elbow_flex",
        "shoulder_lift",
        "wrist_flex",
    ]
    positions = [
        0.6,
        0.5,
        0.1,
        0.3,
        0.2,
        0.4,
    ]
    velocities = [
        1.6,
        1.5,
        1.1,
        1.3,
        1.2,
        1.4,
    ]
    return names, positions, velocities


def test_joint_cache_reorders_complete_feedback():
    cache = JointStateCache()
    names, positions, velocities = scrambled_joint_feedback()

    snapshot = cache.update(
        names,
        positions,
        velocities,
        received_monotonic_s=10.0,
    )

    assert tuple(snapshot.positions) == MEASURED_JOINT_NAMES
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
    assert snapshot.sequence == 1
    assert snapshot.received_monotonic_s == 10.0
    assert snapshot.age_s(now_monotonic_s=10.25) == 0.25


def test_joint_cache_accepts_empty_velocity_array():
    cache = JointStateCache()
    names, positions, _ = scrambled_joint_feedback()

    snapshot = cache.update(
        names,
        positions,
        [],
    )

    assert snapshot.velocities is None


def test_joint_cache_rejects_missing_duplicate_and_nonfinite_data():
    cache = JointStateCache()
    names, positions, velocities = scrambled_joint_feedback()

    with pytest.raises(ValueError, match="gripper"):
        cache.update(
            names[1:],
            positions[1:],
            velocities[1:],
        )

    with pytest.raises(ValueError, match="duplicates"):
        cache.update(
            [*names, "gripper"],
            [*positions, 0.7],
        )

    invalid_positions = list(positions)
    invalid_positions[2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        cache.update(
            names,
            invalid_positions,
        )


def test_joint_cache_returns_independent_snapshots():
    cache = JointStateCache()
    names, positions, velocities = scrambled_joint_feedback()

    cache.update(
        names,
        positions,
        velocities,
    )

    first = cache.latest()
    assert first is not None
    first.positions["shoulder_pan"] = 99.0

    assert first.velocities is not None
    first.velocities["shoulder_pan"] = 99.0

    second = cache.latest()
    assert second is not None
    assert second.positions["shoulder_pan"] == 0.1

    assert second.velocities is not None
    assert second.velocities["shoulder_pan"] == 1.1


def test_joint_cache_sequence_distinguishes_new_feedback():
    cache = JointStateCache()
    names, positions, velocities = scrambled_joint_feedback()

    first = cache.update(
        names,
        positions,
        velocities,
    )

    assert (
        cache.wait_for_sequence_greater_than(
            sequence=0,
            timeout_s=0.0,
        )
        is not None
    )

    assert (
        cache.wait_for_sequence_greater_than(
            sequence=first.sequence,
            timeout_s=0.0,
        )
        is None
    )

    second = cache.update(
        names,
        positions,
        velocities,
    )

    assert second.sequence == first.sequence + 1


def test_target_cube_cache_normalizes_pose_and_copies_arrays():
    cache = TargetCubeStateCache()

    snapshot = cache.update(
        "target_cube_0001",
        [0.25, 0.05, 0.765],
        [0.0, 0.0, 0.0, 2.0],
        received_monotonic_s=20.0,
    )

    np.testing.assert_allclose(
        snapshot.position,
        [0.25, 0.05, 0.765],
        atol=0.0,
    )

    np.testing.assert_allclose(
        snapshot.quaternion_xyzw,
        [0.0, 0.0, 0.0, 1.0],
        atol=0.0,
    )

    assert snapshot.sequence == 1
    assert snapshot.age_s(
        now_monotonic_s=20.1,
    ) == pytest.approx(0.1)

    snapshot.position[0] = 99.0
    latest = cache.latest()

    assert latest is not None
    assert latest.position[0] == 0.25


def test_target_cube_cache_rejects_invalid_pose():
    cache = TargetCubeStateCache()

    with pytest.raises(ValueError, match="name"):
        cache.update(
            "",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )

    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        cache.update(
            "target_cube_0001",
            [0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )

    with pytest.raises(ValueError, match="non-zero"):
        cache.update(
            "target_cube_0001",
            [0.0, 0.0, 0.765],
            [0.0, 0.0, 0.0, 0.0],
        )


def test_target_cube_cache_preserves_source_metadata():
    cache = TargetCubeStateCache()

    snapshot = cache.update(
        "target_cube_0001",
        [0.2, 0.1, 0.765],
        [0.0, 0.0, 0.0, 1.0],
        source_sequence=123,
        source_sim_time_s=4.1,
        received_monotonic_s=20.0,
    )

    assert snapshot.sequence == 1
    assert snapshot.source_sequence == 123
    assert snapshot.source_sim_time_s == pytest.approx(4.1)

    copied = cache.latest()

    assert copied is not None
    assert copied.sequence == 1
    assert copied.source_sequence == 123
    assert copied.source_sim_time_s == pytest.approx(4.1)
