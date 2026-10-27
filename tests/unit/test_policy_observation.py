import numpy as np
import pytest

from so101_capstone.environments import (
    ImageFrameSnapshot,
    PolicyStateObservation,
    SynchronizedCameraFrames,
    build_policy_observation,
)


def image_snapshot(
    value,
    *,
    sequence,
    shape=(4, 6, 3),
    dtype=np.uint8,
):
    return ImageFrameSnapshot(
        image=np.full(
            shape,
            value,
            dtype=dtype,
        ),
        sequence=sequence,
        received_monotonic_s=10.0,
        source_timestamp_s=20.0,
    )


def synchronized_frames(
    *,
    top_shape=(4, 6, 3),
    gripper_shape=(4, 6, 3),
):
    return SynchronizedCameraFrames(
        top=image_snapshot(
            10,
            sequence=1,
            shape=top_shape,
        ),
        gripper=image_snapshot(
            20,
            sequence=2,
            shape=gripper_shape,
        ),
        skew_s=0.0,
        used_source_timestamps=True,
    )


def test_builds_complete_policy_observation():
    state = PolicyStateObservation(
        joint_positions=np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ),
        joint_velocities=np.array(
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        ),
    )

    observation = build_policy_observation(
        state,
        synchronized_frames(),
    )

    np.testing.assert_allclose(
        observation.joint_positions,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    np.testing.assert_allclose(
        observation.joint_velocities,
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    )

    assert observation.top_image.shape == (4, 6, 3)
    assert observation.gripper_image.shape == (4, 6, 3)
    assert observation.top_image.dtype == np.uint8
    assert observation.gripper_image.dtype == np.uint8
    assert np.all(observation.top_image == 10)
    assert np.all(observation.gripper_image == 20)


def test_state_vector_is_positions_followed_by_velocities():
    state = PolicyStateObservation(
        joint_positions=np.arange(6, dtype=float),
        joint_velocities=np.arange(
            10,
            16,
            dtype=float,
        ),
    )

    observation = build_policy_observation(
        state,
        synchronized_frames(),
    )

    np.testing.assert_allclose(
        observation.state_vector(),
        [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    )


def test_observation_arrays_are_read_only_and_independent():
    positions = np.zeros(6, dtype=float)
    velocities = np.ones(6, dtype=float)

    observation = build_policy_observation(
        PolicyStateObservation(
            joint_positions=positions,
            joint_velocities=velocities,
        ),
        synchronized_frames(),
    )

    positions[:] = 99.0
    velocities[:] = 99.0

    assert np.all(observation.joint_positions == 0.0)
    assert np.all(observation.joint_velocities == 1.0)

    with pytest.raises(ValueError):
        observation.top_image[0, 0, 0] = 1

    copied = observation.as_dict()
    copied["joint_positions"][0] = 77.0

    assert observation.joint_positions[0] == 0.0


def test_two_cameras_may_have_different_resolutions():
    state = PolicyStateObservation(
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
    )

    observation = build_policy_observation(
        state,
        synchronized_frames(
            top_shape=(240, 320, 3),
            gripper_shape=(480, 640, 3),
        ),
    )

    assert observation.top_image.shape == (240, 320, 3)
    assert observation.gripper_image.shape == (480, 640, 3)


def test_invalid_state_and_image_inputs_are_rejected():
    invalid_state = PolicyStateObservation(
        joint_positions=np.zeros(5),
        joint_velocities=np.zeros(6),
    )

    with pytest.raises(
        ValueError,
        match=r"shape \(6,\)",
    ):
        build_policy_observation(
            invalid_state,
            synchronized_frames(),
        )

    invalid_frames = SynchronizedCameraFrames(
        top=image_snapshot(
            10,
            sequence=1,
            dtype=np.float32,
        ),
        gripper=image_snapshot(
            20,
            sequence=2,
        ),
        skew_s=0.0,
        used_source_timestamps=True,
    )

    with pytest.raises(ValueError, match="uint8"):
        build_policy_observation(
            PolicyStateObservation(
                joint_positions=np.zeros(6),
                joint_velocities=np.zeros(6),
            ),
            invalid_frames,
        )
