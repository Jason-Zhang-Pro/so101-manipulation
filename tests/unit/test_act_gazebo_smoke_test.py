from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_capstone.environments import (
    PolicyObservation,
)
from so101_capstone.policies.act_gazebo_smoke_test import (
    run_act_gazebo_dry_run,
    validate_dry_run_observation,
)


def make_observation(
    *,
    top_shape=(480, 640, 3),
    gripper_shape=(480, 640, 3),
    image_dtype=np.uint8,
):
    return PolicyObservation(
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
        top_image=np.zeros(
            top_shape,
            dtype=image_dtype,
        ),
        gripper_image=np.zeros(
            gripper_shape,
            dtype=image_dtype,
        ),
    )


class FakeRuntime:
    def __init__(self, observation):
        self.observation = observation
        self.reset_calls = 0
        self.step_calls = 0
        self.enter_calls = 0
        self.exit_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ):
        self.exit_calls += 1
        return False

    def reset(self):
        self.reset_calls += 1

        return SimpleNamespace(
            observation=self.observation,
            info={
                "reset_sim_time_s": 1.0,
                "top_image_sequence": 10,
                "gripper_image_sequence": 20,
                "camera_skew_s": 0.001,
                "target_active": False,
            },
        )

    def step(self, _command):
        self.step_calls += 1
        raise AssertionError(
            "Dry-run must not call runtime.step()."
        )


class FakeAdapter:
    def __init__(
        self,
        policy,
        *,
        device,
        max_joint_delta_rad,
    ):
        self.policy = policy
        self.device = device
        self.max_joint_delta_rad = (
            max_joint_delta_rad
        )
        self.reset_calls = 0
        self.select_calls = 0

    def reset(self):
        self.reset_calls += 1

    def select_action(self, observation):
        self.select_calls += 1

        return SimpleNamespace(
            raw_policy_positions=np.array(
                [-1, -2, -3, -4, -5, 25],
                dtype=float,
            ),
            clipped_policy_positions=np.array(
                [-1, -2, -3, -4, -5, 25],
                dtype=float,
            ),
            controller_command=lambda: np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                dtype=float,
            ),
        )


def test_dry_run_infers_without_runtime_step(
    tmp_path,
):
    observation = make_observation()
    runtime = FakeRuntime(observation)
    adapters = []

    def checkpoint_loader(path, *, device):
        return SimpleNamespace(
            checkpoint_path=Path(path),
            device=torch.device("cpu"),
            policy=object(),
        )

    def runtime_factory(*, runtime_config):
        assert (
            runtime_config
            .startup_feedback_timeout_s
        ) == pytest.approx(12.0)

        return runtime

    def adapter_factory(*args, **kwargs):
        adapter = FakeAdapter(*args, **kwargs)
        adapters.append(adapter)
        return adapter

    checkpoint = tmp_path / "checkpoint"

    result = run_act_gazebo_dry_run(
        checkpoint,
        device="cpu",
        max_joint_delta_rad=0.05,
        startup_feedback_timeout_s=12.0,
        checkpoint_loader=checkpoint_loader,
        runtime_factory=runtime_factory,
        adapter_factory=adapter_factory,
    )

    assert runtime.reset_calls == 1
    assert runtime.step_calls == 0
    assert runtime.enter_calls == 1
    assert runtime.exit_calls == 1

    assert len(adapters) == 1
    assert adapters[0].reset_calls == 1
    assert adapters[0].select_calls == 1

    assert result.top_image_shape == (
        480,
        640,
        3,
    )
    assert result.gripper_image_shape == (
        480,
        640,
        3,
    )

    np.testing.assert_allclose(
        result.proposed_controller_command_rad,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            make_observation(
                top_shape=(600, 800, 3)
            ),
            "top_image must have shape",
        ),
        (
            make_observation(
                gripper_shape=(240, 320, 3)
            ),
            "gripper_image must have shape",
        ),
        (
            make_observation(
                image_dtype=np.float32
            ),
            "dtype uint8",
        ),
    ],
)
def test_invalid_policy_observation_is_rejected(
    observation,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        validate_dry_run_observation(
            observation
        )


def test_result_arrays_are_read_only(
    tmp_path,
):
    runtime = FakeRuntime(
        make_observation()
    )

    result = run_act_gazebo_dry_run(
        tmp_path / "checkpoint",
        device="cpu",
        max_joint_delta_rad=0.05,
        checkpoint_loader=(
            lambda path, device: SimpleNamespace(
                device=torch.device("cpu"),
                policy=object(),
            )
        ),
        runtime_factory=(
            lambda runtime_config: runtime
        ),
        adapter_factory=FakeAdapter,
    )

    with pytest.raises(ValueError):
        result.raw_policy_positions[0] = 9.0

    with pytest.raises(ValueError):
        result.proposed_controller_command_rad[0] = 9.0
