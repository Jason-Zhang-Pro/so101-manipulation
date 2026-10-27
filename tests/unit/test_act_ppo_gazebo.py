from pathlib import Path

import numpy as np
import pytest

from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig,
    _default_output_directory,
    _actor_update_enabled,
    _exploration_enabled,
    _exploration_stage,
    EpisodeFailureTracker,
    RecoverableEpisodeError,
    _map_policy_positions_for_episode,
    _recoverable_episode_category,
)


def test_defaults_use_six_twenty_second_episodes():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("checkpoint")
    )

    assert config.episode_count == 6
    assert config.max_steps_per_episode == 600
    assert config.rollout_steps == 600
    assert config.target_delay_steps == 90
    assert config.planned_transition_count == 3600
    assert config.to_json_dict()["max_joint_delta_rad"] is None


def test_relative_joint_delta_limit_must_remain_disabled():
    with pytest.raises(ValueError, match="requires"):
        ActPpoGazeboConfig(
            act_checkpoint=Path("checkpoint"),
            disable_max_joint_delta_limit=False,
        )


def test_default_output_records_pretrained_lineage():
    checkpoint = Path(
        "output_simulation/pretrained/"
        "act_so101_pick_test_20260803_143945/"
        "training_output/checkpoints/060000/"
        "pretrained_model"
    )

    output = _default_output_directory(checkpoint)

    assert output.parent == Path(
        "output_simulation/finetuned"
    )
    assert output.name.startswith("act_ppo_")
    assert output.name.endswith(
        "_from_pretrained_20260803_143945"
    )


@pytest.mark.parametrize(
    "info",
    [
        {"reward_state": 1},
        {"reward_state": 2, "robot_at_home": False},
        {"reward_state": 4, "robot_at_home": False},
        {"reward_state": 4, "robot_at_home": True},
    ],
)
def test_exploration_is_enabled_in_every_reward_stage_by_default(info):
    config = ActPpoGazeboConfig(act_checkpoint=Path("checkpoint"))
    assert _exploration_enabled(info, config) is True
    assert _actor_update_enabled(info, config) is True


def test_actor_switches_require_stochastic_samples():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("checkpoint"),
        state1_exploration_enabled=False,
        state1_actor_update_enabled=True,
        state2_exploration_enabled=True,
        state2_actor_update_enabled=False,
        state4_not_home_exploration_enabled=False,
        state4_not_home_actor_update_enabled=False,
        state4_home_exploration_enabled=True,
        state4_home_actor_update_enabled=True,
    )

    assert _exploration_enabled({"reward_state": 1}, config) is False
    assert _actor_update_enabled({"reward_state": 1}, config) is False
    assert _exploration_enabled({"reward_state": 2}, config) is True
    assert _actor_update_enabled({"reward_state": 2}, config) is False
    assert (
        _exploration_enabled(
            {"reward_state": 4, "robot_at_home": False}, config
        )
        is False
    )
    assert (
        _actor_update_enabled(
            {"reward_state": 4, "robot_at_home": False}, config
        )
        is False
    )
    assert (
        _exploration_enabled(
            {"reward_state": 4, "robot_at_home": True}, config
        )
        is True
    )
    assert (
        _actor_update_enabled(
            {"reward_state": 4, "robot_at_home": True}, config
        )
        is True
    )


@pytest.mark.parametrize(
    ("info", "expected_stage", "expected_log_std"),
    [
        ({"reward_state": 1}, "state1", -5.0),
        ({"reward_state": 2}, "state2", -4.0),
        (
            {"reward_state": 4, "robot_at_home": False},
            "state4_not_home",
            -4.0,
        ),
        (
            {"reward_state": 4, "robot_at_home": True},
            "state4_home",
            -5.0,
        ),
    ],
)
def test_exploration_stage_uses_configured_noise(
    info,
    expected_stage,
    expected_log_std,
):
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("checkpoint"),
        initial_log_std=-4.0,
        minimum_log_std=-6.0,
        maximum_log_std=-3.5,
    )

    stage, initial_log_std, offset = _exploration_stage(info, config)

    assert stage == expected_stage
    assert initial_log_std == pytest.approx(expected_log_std)
    assert offset == pytest.approx(
        expected_log_std - config.initial_log_std
    )


def test_stage_switches_reject_missing_reward_state():
    config = ActPpoGazeboConfig(act_checkpoint=Path("checkpoint"))
    with pytest.raises(ValueError, match="reward_state"):
        _exploration_enabled({}, config)
    with pytest.raises(ValueError, match="reward_state"):
        _actor_update_enabled({}, config)


@pytest.mark.parametrize(
    "name",
    [
        "state1_exploration_enabled",
        "state2_exploration_enabled",
        "state4_not_home_exploration_enabled",
        "state4_home_exploration_enabled",
        "state1_actor_update_enabled",
        "state2_actor_update_enabled",
        "state4_not_home_actor_update_enabled",
        "state4_home_actor_update_enabled",
    ],
)
def test_stage_switches_require_boolean_values(name):
    with pytest.raises(ValueError, match="must be a bool"):
        ActPpoGazeboConfig(
            act_checkpoint=Path("checkpoint"),
            **{name: 1},
        )


def test_default_recoverable_failure_limit_is_three():
    config = ActPpoGazeboConfig(act_checkpoint=Path("checkpoint"))
    assert config.max_consecutive_episode_failures == 3


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_recoverable_failure_limit_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        ActPpoGazeboConfig(
            act_checkpoint=Path("checkpoint"),
            max_consecutive_episode_failures=value,
        )


def test_episode_failure_tracker_stops_on_third_consecutive_failure():
    tracker = EpisodeFailureTracker(3)

    assert tracker.start_attempt() == 1
    assert tracker.record_failure() is False
    assert tracker.start_attempt() == 2
    assert tracker.record_failure() is False
    assert tracker.start_attempt() == 3
    assert tracker.record_failure() is True
    assert tracker.consecutive_failures == 3
    assert tracker.failure_count == 3


def test_valid_episode_resets_consecutive_failure_count():
    tracker = EpisodeFailureTracker(3)
    tracker.start_attempt()
    tracker.record_failure()
    tracker.start_attempt()
    tracker.record_success()

    assert tracker.consecutive_failures == 0
    assert tracker.failure_count == 1
    assert tracker.attempt_count == 2


def test_action_safety_value_error_becomes_recoverable_episode_error():
    class Adapter:
        def map_policy_positions(self, observation, policy_positions):
            raise ValueError("unsafe ACT action")

    with pytest.raises(
        RecoverableEpisodeError,
        match="unsafe ACT action",
    ) as captured:
        _map_policy_positions_for_episode(
            Adapter(),
            object(),
            object(),
        )

    assert captured.value.category == "act_action_safety"


def test_custom_recoverable_error_preserves_category():
    error = RecoverableEpisodeError("target_service", "failed")
    assert _recoverable_episode_category(error) == "target_service"


@pytest.mark.parametrize(
    ("info", "expected_stage"),
    [
        ({"reward_state": 1}, "state1"),
        ({"reward_state": 2}, "state2"),
        ({"reward_state": 3, "robot_at_home": False}, "state3"),
        ({"reward_state": 3, "robot_at_home": True}, "state3"),
        ({"reward_state": 4, "robot_at_home": False}, "state4_not_home"),
        ({"reward_state": 4, "robot_at_home": True}, "state4_home"),
    ],
)
def test_four_reward_states_map_to_five_ppo_stages(info, expected_stage):
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("checkpoint"),
        initial_log_std=-4.0,
        minimum_log_std=-24.0,
        maximum_log_std=-3.5,
    )
    stage, _, _ = _exploration_stage(info, config)
    assert stage == expected_stage


def test_new_state3_has_independent_exploration_and_actor_switches():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("checkpoint"),
        initial_log_std=-4.0,
        minimum_log_std=-24.0,
        maximum_log_std=-3.5,
        state3_exploration_log_std=-4.25,
        state3_exploration_enabled=False,
        state3_actor_update_enabled=False,
    )
    info = {"reward_state": 3, "robot_at_home": False}
    stage, initial_log_std, offset = _exploration_stage(info, config)
    assert stage == "state3"
    assert initial_log_std == pytest.approx(-4.25)
    assert offset == pytest.approx(-0.25)
    assert _exploration_enabled(info, config) is False
    assert _actor_update_enabled(info, config) is False
def test_final_target_absent_window_is_last_three_seconds():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/pretrained_model"),
        max_steps_per_episode=450,
        target_delay_s=3.0,
        final_target_absent_s=3.0,
        policy_hz=30.0,
    )

    assert config.target_delay_steps == 90
    assert config.final_target_absent_steps == 90
    assert config.final_target_delete_step == 360

    payload = config.to_json_dict()
    assert payload["final_target_absent_s"] == 3.0
    assert payload["final_target_absent_steps"] == 90
    assert payload["final_target_delete_step"] == 360


@pytest.mark.parametrize(
    "kwargs",
    [
        {"final_target_absent_s": -1.0},
        {"final_target_absent_s": float("inf")},
        {
            "max_steps_per_episode": 450,
            "target_delay_s": 3.0,
            "final_target_absent_s": 15.0,
        },
        {
            "max_steps_per_episode": 450,
            "target_delay_s": 12.0,
            "final_target_absent_s": 3.0,
        },
    ],
)
def test_final_target_absent_window_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ActPpoGazeboConfig(
            act_checkpoint=Path("/tmp/pretrained_model"),
            **kwargs,
        )


@pytest.mark.parametrize("state", [1, 2, 3, 4])
def test_deterministic_exploration_stages_never_update_actor(state):
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("checkpoint"),
        state1_exploration_enabled=False,
        state2_exploration_enabled=False,
        state3_exploration_enabled=False,
        state4_not_home_exploration_enabled=False,
        state4_home_exploration_enabled=False,
    )
    for at_home in (False, True):
        info = {"reward_state": state, "robot_at_home": at_home}
        assert not _actor_update_enabled(info, config)
