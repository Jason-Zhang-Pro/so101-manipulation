from pathlib import Path

import pytest

from so101_capstone.training.act_ppo_multiworld_gazebo import (
    MultiWorldConfig,
    _episode_minimum_reached,
    _parser,
    _rollout_threshold_reached,
)
from so101_capstone.training.act_ppo_multiworld_worker import (
    GazeboWorkerConfig,
    gazebo_launch_command,
)


def test_multiworld_assigns_independent_domains_and_partitions():
    config = MultiWorldConfig(
        world_count=4,
        base_ros_domain_id=20,
        gz_partition_prefix="gz_env_",
        gazebo_gui=True,
        world_startup_max_attempts=3,
        world_restart_max_attempts=3,
    )

    assert [config.domain_id(i) for i in range(4)] == [20, 21, 22, 23]
    assert [config.partition(i) for i in range(4)] == [
        "gz_env_20",
        "gz_env_21",
        "gz_env_22",
        "gz_env_23",
    ]


def test_rollout_threshold_is_minimum_and_all_overshoot_is_eligible():
    assert not _rollout_threshold_reached(1999, 2000)
    assert _rollout_threshold_reached(2000, 2000)
    assert _rollout_threshold_reached(2400, 2000)


def test_episode_count_is_minimum_and_parallel_wave_can_overshoot():
    assert not _episode_minimum_reached(100, 101)
    assert _episode_minimum_reached(101, 101)
    assert _episode_minimum_reached(104, 101)


def test_cli_exposes_multiworld_and_gui_controls():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--world-count",
            "4",
            "--base-ros-domain-id",
            "20",
            "--gz-partition-prefix",
            "gz_env_",
            "--no-gazebo-gui",
            "--world-startup-max-attempts",
            "3",
            "--world-restart-max-attempts",
            "3",
            "--world-deferred-restart-max-attempts",
            "4",
            "--preflight-cleanup",
            "--preflight-cleanup-timeout-s",
            "4.5",
        ]
    )

    assert args.world_count == 4
    assert args.base_ros_domain_id == 20
    assert args.gz_partition_prefix == "gz_env_"
    assert args.gazebo_gui is False
    assert args.world_startup_max_attempts == 3
    assert args.world_restart_max_attempts == 3
    assert args.world_deferred_restart_max_attempts == 4
    assert args.preflight_cleanup is True
    assert args.preflight_cleanup_timeout_s == 4.5


def test_worker_launch_command_forwards_gui_choice_to_ros_launch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(
        "SO101_ENVIRONMENT_RANDOMIZATION_ENABLED",
        raising=False,
    )
    config = GazeboWorkerConfig(
        world_index=0,
        ros_domain_id=20,
        gz_partition="gz_env_20",
        gazebo_gui=False,
        project_root=tmp_path,
        output_directory=tmp_path / "out",
        startup_feedback_timeout_s=30.0,
    )

    command = gazebo_launch_command(config)
    assert "gazebo_gui:=false" in command
    assert "environment_randomization:=false" in command


def test_worker_config_records_long_lived_world_launch_contract(tmp_path):
    config = GazeboWorkerConfig(
        world_index=2,
        ros_domain_id=22,
        gz_partition="gz_env_22",
        gazebo_gui=False,
        project_root=tmp_path,
        output_directory=tmp_path / "out",
        startup_feedback_timeout_s=30.0,
        startup_max_attempts=3,
        restart_max_attempts=3,
    )

    assert config.world_label == "world_02"
    assert config.log_path.name == "gazebo.log"
    assert config.payload()["gazebo_gui"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_count": 0},
        {"base_ros_domain_id": -1},
        {"world_count": 2, "base_ros_domain_id": 232},
        {"world_startup_max_attempts": 0},
        {"world_restart_max_attempts": -1},
        {"world_deferred_restart_max_attempts": 0},
        {"gazebo_gui": 1},
        {"preflight_cleanup_enabled": 1},
        {"preflight_cleanup_timeout_s": 0.0},
        {"s0_step_fraction": -0.1},
        {"s0_step_fraction": 1.1},
        {"s0_step_fraction": float("nan")},
        {"s0_step_fraction": True},
        {"stage_ratio_tolerance": -0.01},
        {"stage_ratio_tolerance": 0.6},
        {"stage_ratio_tolerance": float("inf")},
        {"stage_max_rollout_multiplier": 0.5},
        {"stage_max_rollout_multiplier": float("inf")},
    ],
)
def test_multiworld_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        MultiWorldConfig(**kwargs)
def test_managed_entry_rejects_in_episode_target_delete_for_legacy_profile():
    from so101_capstone.training.act_ppo_multiworld_gazebo import _build_base_config

    args = _parser().parse_args([
        "--act-checkpoint", "/tmp/act", "--reward-profile", "legacy",
        "--final-target-absent-s", "3.0",
    ])
    with pytest.raises(ValueError, match="in-episode target deletion"):
        _build_base_config(args)


def test_episode_session_tail_cleanup_is_independent_of_prepare_cleanup():
    from types import SimpleNamespace

    from so101_capstone.training.act_ppo import (
        ActPpoConfig,
        ActPpoRolloutBuffer,
    )
    from so101_capstone.training.act_ppo_multiworld_gazebo import (
        _EpisodeSession,
    )

    slot = SimpleNamespace(world_index=0)
    session = _EpisodeSession(
        slot=slot,
        buffer=ActPpoRolloutBuffer(ActPpoConfig()),
        observation=object(),
        last_info={},
        cleanup={"success": True, "message": "prepare cleanup"},
        home_reset={},
        attempt_index=1,
    )

    assert session.cleanup == {
        "success": True,
        "message": "prepare cleanup",
    }
    assert session.tail_target_cleanup["requested"] is False

def test_terminal_target_presence_uses_latest_environment_info():
    from so101_capstone.training.act_ppo_multiworld_gazebo import (
        _terminal_target_present,
    )

    assert _terminal_target_present(
        {"reward_state": 2, "target_active": True}
    ) is True
    assert _terminal_target_present(
        {"reward_state": 1, "target_active": False}
    ) is False

    # Never fall back to the historical "spawn was requested" meaning.
    assert _terminal_target_present({"reward_state": 1}) is False
