import pytest

import so101_capstone.policies.act_gazebo_multi_episode as module
from so101_capstone.environments import (
    ReachEnvironmentConfig,
    SimulationRuntimeConfig,
)


def required_cli_arguments():
    return [
        "--checkpoint",
        "/tmp/pretrained_model",
        "--max-joint-delta-rad",
        "1000000.0",
        "--episode-count",
        "1",
        "--max-steps",
        "600",
    ]


def test_cli_default_uses_training_compatible_600_step_horizon():
    args = module._parser().parse_args(required_cli_arguments())

    assert args.environment_max_episode_steps == 600


def test_cli_delta_limit_is_disabled_when_omitted():
    args = module._parser().parse_args([
        "--checkpoint", "/tmp/pretrained_model",
        "--episode-count", "1", "--max-steps", "600",
    ])
    assert args.max_joint_delta_rad is None


def test_cli_accepts_ten_minute_18000_step_test_horizon():
    args = module._parser().parse_args(
        required_cli_arguments()
        + ["--environment-max-episode-steps", "18000"]
    )

    assert args.environment_max_episode_steps == 18000


def test_environment_config_uses_requested_episode_horizon():
    config = module._environment_config(18000)

    assert isinstance(config, ReachEnvironmentConfig)
    assert config.max_episode_steps == 18000


def test_default_runtime_receives_environment_config(monkeypatch):
    calls = []
    sentinel = object()

    def fake_create_ros_simulation_runtime(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        module,
        "create_ros_simulation_runtime",
        fake_create_ros_simulation_runtime,
    )

    runtime_config = SimulationRuntimeConfig()
    environment_config = ReachEnvironmentConfig(
        max_episode_steps=18000,
    )

    result = module._runtime_context(
        runtime_factory=None,
        runtime_config=runtime_config,
        environment_config=environment_config,
    )

    assert result is sentinel
    assert calls == [
        {
            "runtime_config": runtime_config,
            "environment_config": environment_config,
        }
    ]


def test_injected_runtime_factory_keeps_existing_signature():
    calls = []
    sentinel = object()

    def runtime_factory(*, runtime_config):
        calls.append(runtime_config)
        return sentinel

    runtime_config = SimulationRuntimeConfig()
    environment_config = ReachEnvironmentConfig(
        max_episode_steps=18000,
    )

    result = module._runtime_context(
        runtime_factory=runtime_factory,
        runtime_config=runtime_config,
        environment_config=environment_config,
    )

    assert result is sentinel
    assert calls == [runtime_config]


@pytest.mark.parametrize("value", [0, -1])
def test_environment_horizon_rejects_non_positive_values(value):
    with pytest.raises(ValueError):
        module._environment_config(value)
