import numpy as np
import pytest
import torch

from so101_capstone.environments.policy_action import (
    POLICY_ACTION_SIZE,
    PolicyAction,
)
from so101_capstone.environments.policy_observation import (
    PolicyObservation,
)
from so101_capstone.policies.act_policy_adapter import (
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_STATE_KEY,
    ActPolicyAdapter,
)
from so101_capstone.policies.so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS,
)


class RecordingPolicy:
    def __init__(self, action):
        self.action = torch.tensor(
            action,
            dtype=torch.float32,
        )
        self.batches = []
        self.reset_calls = 0

    def select_action(self, batch):
        self.batches.append(batch)
        return self.action.clone()

    def reset(self):
        self.reset_calls += 1


def gazebo_midpoints():
    return np.array(
        [
            (
                calibration.gazebo_lower_rad
                + calibration.gazebo_upper_rad
            )
            / 2.0
            for calibration in SIM_JOINT_CALIBRATIONS
        ],
        dtype=float,
    )


def make_observation(
    *,
    joint_positions=None,
    joint_velocities=None,
    top_pixel=(10, 20, 30),
    gripper_pixel=(40, 50, 60),
    top_shape=(480, 640, 3),
    gripper_shape=(480, 640, 3),
    image_dtype=np.uint8,
):
    if joint_positions is None:
        joint_positions = gazebo_midpoints()

    if joint_velocities is None:
        joint_velocities = np.zeros(6)

    top_image = np.empty(
        top_shape,
        dtype=image_dtype,
    )
    top_image[...] = top_pixel

    gripper_image = np.empty(
        gripper_shape,
        dtype=image_dtype,
    )
    gripper_image[...] = gripper_pixel

    return PolicyObservation(
        joint_positions=np.asarray(
            joint_positions,
            dtype=float,
        ),
        joint_velocities=np.asarray(
            joint_velocities,
            dtype=float,
        ),
        top_image=top_image,
        gripper_image=gripper_image,
    )


def make_adapter(
    action,
    *,
    max_joint_delta_rad=10.0,
):
    policy = RecordingPolicy(action)

    adapter = ActPolicyAdapter(
        policy,
        device="cpu",
        max_joint_delta_rad=max_joint_delta_rad,
    )

    return adapter, policy


def test_default_disables_delta_but_keeps_absolute_limits():
    adapter = ActPolicyAdapter(RecordingPolicy([[0.0] * 6]), device="cpu")
    observation = make_observation()
    result = adapter.map_policy_positions(observation, [1e9] * 6)
    expected = np.asarray([item.gazebo_upper_rad for item in SIM_JOINT_CALIBRATIONS])
    assert adapter.max_joint_delta_rad is None
    np.testing.assert_allclose(result.controller_command(), expected)
    assert np.any(np.abs(result.controller_command() - observation.joint_positions) > 0.05)
    with pytest.raises(ValueError, match="finite"):
        adapter.map_policy_positions(observation, [float("nan")] * 6)


def test_build_batch_matches_checkpoint_contract():
    adapter, _ = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )

    batch = adapter.build_batch(
        make_observation()
    )

    assert set(batch) == {
        ACT_FIXED_IMAGE_KEY,
        ACT_HANDEYE_IMAGE_KEY,
        ACT_STATE_KEY,
    }

    fixed = batch[ACT_FIXED_IMAGE_KEY]
    handeye = batch[ACT_HANDEYE_IMAGE_KEY]
    state = batch[ACT_STATE_KEY]

    assert fixed.shape == (1, 3, 480, 640)
    assert handeye.shape == (1, 3, 480, 640)
    assert state.shape == (1, 6)

    assert fixed.dtype == torch.float32
    assert handeye.dtype == torch.float32
    assert state.dtype == torch.float32

    torch.testing.assert_close(
        fixed[0, :, 0, 0],
        torch.tensor(
            [10.0, 20.0, 30.0],
            dtype=torch.float32,
        ) / 255.0,
    )

    torch.testing.assert_close(
        handeye[0, :, 0, 0],
        torch.tensor(
            [40.0, 50.0, 60.0],
            dtype=torch.float32,
        ) / 255.0,
    )

    torch.testing.assert_close(
        state,
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]],
            dtype=torch.float32,
        ),
        atol=1e-5,
        rtol=0.0,
    )


def test_joint_velocities_are_not_policy_inputs():
    adapter, _ = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )

    first = adapter.build_batch(
        make_observation(
            joint_velocities=np.zeros(6),
        )
    )

    second = adapter.build_batch(
        make_observation(
            joint_velocities=np.full(6, 999.0),
        )
    )

    assert set(first) == set(second)

    for key in first:
        torch.testing.assert_close(
            first[key],
            second[key],
        )


def test_absolute_action_maps_to_gazebo_limits():
    policy_action = [
        [
            -100.0,
            100.0,
            -100.0,
            100.0,
            -100.0,
            100.0,
        ]
    ]

    adapter, policy = make_adapter(
        policy_action,
        max_joint_delta_rad=10.0,
    )

    result = adapter.select_action(
        make_observation()
    )

    expected = np.array(
        [
            SIM_JOINT_CALIBRATIONS[0].gazebo_lower_rad,
            SIM_JOINT_CALIBRATIONS[1].gazebo_upper_rad,
            SIM_JOINT_CALIBRATIONS[2].gazebo_lower_rad,
            SIM_JOINT_CALIBRATIONS[3].gazebo_upper_rad,
            SIM_JOINT_CALIBRATIONS[4].gazebo_lower_rad,
            SIM_JOINT_CALIBRATIONS[5].gazebo_upper_rad,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        result.raw_policy_positions,
        policy_action[0],
        atol=1e-6,
    )

    np.testing.assert_allclose(
        result.mapped_gazebo_positions_rad,
        expected,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        result.controller_command(),
        expected,
        atol=1e-6,
    )

    assert isinstance(result.action, PolicyAction)
    assert POLICY_ACTION_SIZE == 6
    assert not hasattr(result.action, "done")
    assert not hasattr(result.action, "done_raw")
    assert len(policy.batches) == 1


def test_policy_output_is_clipped_before_mapping():
    adapter, _ = make_adapter(
        [[-200.0, 200.0, -300.0, 300.0, -400.0, 400.0]]
    )

    result = adapter.select_action(
        make_observation()
    )

    np.testing.assert_allclose(
        result.clipped_policy_positions,
        [
            -100.0,
            100.0,
            -100.0,
            100.0,
            -100.0,
            100.0,
        ],
        atol=0.0,
    )


def test_per_step_joint_delta_limit_is_applied():
    adapter, _ = make_adapter(
        [[100.0, -100.0, 100.0, -100.0, 100.0, 100.0]],
        max_joint_delta_rad=0.05,
    )

    result = adapter.select_action(
        make_observation(
            joint_positions=np.zeros(6),
        )
    )

    np.testing.assert_allclose(
        result.controller_command(),
        [0.05, -0.05, 0.05, -0.05, 0.05, 0.05],
        atol=1e-12,
    )


def test_reset_is_forwarded_to_policy():
    adapter, policy = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )

    adapter.reset()
    adapter.reset()

    assert policy.reset_calls == 2


@pytest.mark.parametrize(
    "action",
    [
        torch.zeros(6),
        torch.zeros((1, 7)),
        torch.zeros((2, 6)),
    ],
)
def test_invalid_action_shape_is_rejected(
    action,
):
    adapter, policy = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )
    policy.action = action

    with pytest.raises(
        ValueError,
        match="ACT action must have shape",
    ):
        adapter.select_action(
            make_observation()
        )


def test_nonfinite_action_is_rejected():
    adapter, _ = make_adapter(
        [[0.0, 0.0, np.nan, 0.0, 0.0, 50.0]]
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        adapter.select_action(
            make_observation()
        )


def test_non_tensor_action_is_rejected():
    adapter, policy = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )
    policy.action = None

    def invalid_select_action(batch):
        return np.zeros((1, 6))

    policy.select_action = invalid_select_action

    with pytest.raises(
        TypeError,
        match="torch.Tensor",
    ):
        adapter.select_action(
            make_observation()
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"top_shape": (240, 320, 3)},
            "top_image must have shape",
        ),
        (
            {"gripper_shape": (240, 320, 3)},
            "gripper_image must have shape",
        ),
        (
            {"image_dtype": np.float32},
            "dtype uint8",
        ),
    ],
)
def test_invalid_images_are_rejected(
    kwargs,
    message,
):
    adapter, _ = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        adapter.build_batch(
            make_observation(**kwargs)
        )


def test_invalid_joint_positions_are_rejected():
    adapter, _ = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )

    with pytest.raises(
        ValueError,
        match=r"shape \(6,\)",
    ):
        adapter.build_batch(
            make_observation(
                joint_positions=np.zeros(5),
            )
        )


@pytest.mark.parametrize(
    "max_delta",
    [
        0.0,
        -0.1,
        np.nan,
        [0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
    ],
)
def test_invalid_maximum_delta_is_rejected(
    max_delta,
):
    policy = RecordingPolicy(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )

    with pytest.raises(
        ValueError,
        match="max_joint_delta_rad",
    ):
        ActPolicyAdapter(
            policy,
            device="cpu",
            max_joint_delta_rad=max_delta,
        )


def test_result_arrays_are_read_only():
    adapter, _ = make_adapter(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]
    )

    result = adapter.select_action(
        make_observation()
    )

    with pytest.raises(ValueError):
        result.raw_policy_positions[0] = 1.0

    with pytest.raises(ValueError):
        result.action.joint_targets[0] = 1.0

    command = result.controller_command()
    command[0] = 99.0

    assert result.action.joint_targets[0] != 99.0
