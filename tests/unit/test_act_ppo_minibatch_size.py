from pathlib import Path

import pytest

from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig,
    DEFAULT_MINIBATCH_SIZE,
    _act_ppo_config,
    _parser,
)


def test_minibatch_default_preserves_previous_behavior():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/pretrained_model"),
    )

    assert DEFAULT_MINIBATCH_SIZE == 128
    assert config.minibatch_size == 128
    assert _act_ppo_config(config).minibatch_size == 128


def test_minibatch_can_be_reduced_for_gpu_memory():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/pretrained_model"),
        minibatch_size=16,
    )

    assert _act_ppo_config(config).minibatch_size == 16


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_config_rejects_invalid_minibatch_size(value):
    with pytest.raises(ValueError, match="minibatch_size"):
        ActPpoGazeboConfig(
            act_checkpoint=Path("/tmp/pretrained_model"),
            minibatch_size=value,
        )


def test_cli_accepts_minibatch_size():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--minibatch-size",
            "16",
        ]
    )

    assert args.minibatch_size == 16
