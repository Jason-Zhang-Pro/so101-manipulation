import numpy as np
import pytest

from so101_capstone.environments import (
    POLICY_ACTION_SIZE,
    POLICY_JOINT_ACTION_SIZE,
    POLICY_JOINT_NAMES,
    PolicyAction,
    parse_policy_action,
)


def test_policy_action_layout_matches_controller_order():
    action = parse_policy_action(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]
    )

    assert POLICY_JOINT_ACTION_SIZE == 6
    assert POLICY_ACTION_SIZE == 6

    assert POLICY_JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    assert isinstance(action, PolicyAction)

    np.testing.assert_allclose(
        action.joint_targets,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        rtol=0.0,
        atol=0.0,
    )


def test_controller_command_is_an_independent_copy():
    action = parse_policy_action(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )

    command = action.controller_command()
    command[0] = 99.0

    assert action.joint_targets[0] == pytest.approx(0.1)

    assert action.joint_targets_by_name() == {
        "shoulder_pan": 0.1,
        "shoulder_lift": 0.2,
        "elbow_flex": 0.3,
        "wrist_flex": 0.4,
        "wrist_roll": 0.5,
        "gripper": 0.6,
    }


def test_action_storage_is_read_only_and_independent():
    source = np.arange(6, dtype=float)

    action = parse_policy_action(source)
    source[:] = 99.0

    np.testing.assert_allclose(
        action.joint_targets,
        np.arange(6, dtype=float),
    )

    with pytest.raises(ValueError):
        action.joint_targets[0] = 1.0


@pytest.mark.parametrize(
    "invalid_action",
    [
        [],
        [0.0],
        [0.0, 0.0, 0.0],
        [0.0] * 7,
        np.zeros((1, 6)),
    ],
)
def test_invalid_action_shape_is_rejected(
    invalid_action,
):
    with pytest.raises(
        ValueError,
        match=r"shape \(6,\)",
    ):
        parse_policy_action(invalid_action)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_nonfinite_action_is_rejected(
    invalid_value,
):
    action = np.zeros(6, dtype=float)
    action[2] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        parse_policy_action(action)
