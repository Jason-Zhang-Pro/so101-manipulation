from pathlib import Path

import pytest

from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig,
    DEFAULT_CHECKPOINT_EVERY_UPDATES,
    _parser,
    _should_save_update,
)


def test_checkpoint_interval_default_preserves_old_behavior():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/pretrained_model"),
    )

    assert DEFAULT_CHECKPOINT_EVERY_UPDATES == 1
    assert config.checkpoint_every_updates == 1
    assert _should_save_update(1, 1)
    assert _should_save_update(2, 1)


def test_checkpoint_interval_can_save_every_five_updates():
    expected = {
        1: False,
        2: False,
        3: False,
        4: False,
        5: True,
        6: False,
        9: False,
        10: True,
    }

    assert {
        index: _should_save_update(index, 5)
        for index in expected
    } == expected


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_config_rejects_invalid_checkpoint_interval(value):
    with pytest.raises(ValueError, match="checkpoint_every_updates"):
        ActPpoGazeboConfig(
            act_checkpoint=Path("/tmp/pretrained_model"),
            checkpoint_every_updates=value,
        )


@pytest.mark.parametrize(
    ("update_index", "interval"),
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        (1, 0),
        (1, -1),
        (1, True),
    ],
)
def test_checkpoint_decision_rejects_invalid_values(
    update_index,
    interval,
):
    with pytest.raises(ValueError):
        _should_save_update(update_index, interval)


def test_cli_accepts_checkpoint_interval():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--checkpoint-every-updates",
            "7",
        ]
    )

    assert args.checkpoint_every_updates == 7
