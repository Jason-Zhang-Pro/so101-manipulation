from pathlib import Path

import pytest

from so101_capstone.training.act_ppo import ActPpoConfig
from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig,
    _act_ppo_config,
    _parser,
)


def test_single_and_multiworld_cli_forward_target_kl():
    from so101_capstone.training.act_ppo_multiworld_gazebo import (
        _build_base_config,
        _parser as multiworld_parser,
    )

    arguments = ["--act-checkpoint", "/tmp/pretrained_model"]
    assert _parser().parse_args(arguments).target_kl == 0.02
    assert _build_base_config(multiworld_parser().parse_args(arguments)).target_kl == 0.02
    custom = arguments + ["--target-kl", "7.0"]
    assert _parser().parse_args(custom).target_kl == 7.0
    assert _act_ppo_config(_build_base_config(multiworld_parser().parse_args(custom))).target_kl == 7.0


def test_gazebo_defaults_match_act_ppo_defaults():
    gazebo = ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/pretrained_model"),
    )
    resolved = _act_ppo_config(gazebo)
    defaults = ActPpoConfig()

    assert resolved == defaults
    assert resolved.target_kl == 0.02
    assert resolved.entropy_coefficient == 0.0


@pytest.mark.parametrize("entry", ["single", "multiworld"])
def test_physical_exploration_profile_reaches_both_entries(monkeypatch, entry):
    import sys
    from collections import defaultdict
    from dataclasses import replace
    from so101_capstone.training import act_ppo_gazebo as single
    from so101_capstone.training import act_ppo_multiworld_gazebo as multi

    arguments = ["--act-checkpoint", "/tmp/act", "--reward-profile", "pick_place_v2",
                 "--exploration-profile", "final_good_act", "--initial-log-std", "-3",
                 "--minimum-log-std", "-9", "--maximum-log-std", "-2.5",
                 "--gripper-maximum-log-std", "-1.5", "--final-target-absent-s", "0"]
    captured = []
    if entry == "single":
        monkeypatch.setattr(sys, "argv", ["offline-act"] + arguments)
        monkeypatch.setattr(single, "run_act_ppo_gazebo",
                            lambda config: captured.append(config) or defaultdict(lambda: "offline"))
        single.main()
        base = captured[0]
    else:
        base = multi._build_base_config(multi._parser().parse_args(arguments))
    assert base.exploration_profile == "final_good_act"
    reward = single._resolved_reward(base)
    signature = single._execution_signature(base, reward)
    old_signature = single._execution_signature(replace(base, exploration_profile="stage_log_std"), reward)
    assert "exploration_profile" in signature
    assert "exploration_profile" not in old_signature
    assert signature != old_signature
    with pytest.raises(ValueError, match="requires pick_place_v2"):
        replace(base, reward_profile="legacy")


def test_all_training_side_settings_are_forwarded():
    gazebo = ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/pretrained_model"),
        minibatch_size=16,
        critic_image_height=80,
        critic_image_width=112,
        critic_hidden_dim=192,
        critic_learning_rate=2e-4,
        exploration_learning_rate=5e-5,
        gamma=0.98,
        gae_lambda=0.91,
        clip_ratio=0.15,
        target_kl=7.0,
        value_loss_coefficient=0.7,
        entropy_coefficient=0.002,
        mean_bounds_coefficient=0.03,
        max_gradient_norm=0.8,
        critic_max_gradient_norm=0.7,
        exploration_max_gradient_norm=0.6,
        ppo_epochs=3,
        initial_log_std=-2.0,
        minimum_log_std=-4.5,
        maximum_log_std=-0.1,
        state1_exploration_log_std=-4.0,
        state2_exploration_log_std=-3.0,
        state4_not_home_exploration_log_std=-3.2,
        state4_home_exploration_log_std=-4.2,
        actor_learning_rate=8e-6,
        actor_backbone_learning_rate=8e-7,
    )

    resolved = _act_ppo_config(gazebo)

    assert resolved.minibatch_size == 16
    assert resolved.critic_image_height == 80
    assert resolved.critic_image_width == 112
    assert resolved.critic_hidden_dim == 192
    assert resolved.critic_learning_rate == 2e-4
    assert resolved.exploration_learning_rate == 5e-5
    assert resolved.gamma == 0.98
    assert resolved.gae_lambda == 0.91
    assert resolved.clip_ratio == 0.15
    assert resolved.target_kl == 7.0
    assert resolved.value_loss_coefficient == 0.7
    assert resolved.entropy_coefficient == 0.002
    assert resolved.mean_bounds_coefficient == 0.03
    assert resolved.max_gradient_norm == 0.8
    assert resolved.critic_max_gradient_norm == 0.7
    assert resolved.exploration_max_gradient_norm == 0.6
    assert resolved.ppo_epochs == 3
    assert resolved.initial_log_std == -2.0
    assert resolved.minimum_log_std == -4.5
    assert resolved.maximum_log_std == -0.1
    assert resolved.actor_learning_rate == 8e-6
    assert resolved.actor_backbone_learning_rate == 8e-7


def test_cli_accepts_all_training_side_settings():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--minibatch-size",
            "16",
            "--critic-image-height",
            "80",
            "--critic-image-width",
            "112",
            "--critic-hidden-dim",
            "192",
            "--critic-learning-rate",
            "0.0002",
            "--gamma",
            "0.98",
            "--gae-lambda",
            "0.91",
            "--clip-ratio",
            "0.15",
            "--value-loss-coefficient",
            "0.7",
            "--entropy-coefficient",
            "0.002",
            "--max-gradient-norm",
            "0.8",
            "--ppo-epochs",
            "3",
            "--initial-log-std",
            "-2.0",
            "--minimum-log-std",
            "-4.5",
            "--maximum-log-std",
            "-0.1",
            "--actor-learning-rate",
            "0.000008",
            "--actor-backbone-learning-rate",
            "0.0000008",
        ]
    )

    assert args.minibatch_size == 16
    assert args.critic_image_height == 80
    assert args.critic_image_width == 112
    assert args.critic_hidden_dim == 192
    assert args.critic_learning_rate == 0.0002
    assert args.gamma == 0.98
    assert args.gae_lambda == 0.91
    assert args.clip_ratio == 0.15
    assert args.value_loss_coefficient == 0.7
    assert args.entropy_coefficient == 0.002
    assert args.max_gradient_norm == 0.8
    assert args.ppo_epochs == 3
    assert args.initial_log_std == -2.0
    assert args.minimum_log_std == -4.5
    assert args.maximum_log_std == -0.1
    assert args.actor_learning_rate == 0.000008
    assert args.actor_backbone_learning_rate == 0.0000008


def test_actor_learning_rates_default_to_pretrained_inheritance():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
        ]
    )

    assert args.actor_learning_rate is None
    assert args.actor_backbone_learning_rate is None


@pytest.mark.parametrize("entry", ["single", "multiworld"])
def test_entry_forwards_distribution_and_independent_optimizer_settings(monkeypatch, entry):
    from collections import defaultdict
    import sys
    from so101_capstone.training import act_ppo_gazebo as single
    from so101_capstone.training import act_ppo_multiworld_gazebo as multi

    arguments = ["--act-checkpoint", "/tmp/pretrained_model",
                 "--exploration-learning-rate", "0.00005",
                 "--mean-bounds-coefficient", "0.03",
                 "--max-gradient-norm", "0.8",
                 "--critic-max-gradient-norm", "0.7",
                 "--exploration-max-gradient-norm", "0.6"]
    captured = []
    if entry == "single":
        monkeypatch.setattr(sys, "argv", ["act-ppo"] + arguments)
        monkeypatch.setattr(single, "run_act_ppo_gazebo",
                            lambda config: captured.append(config) or defaultdict(lambda: "offline-test"))
        single.main()
        base = captured[0]
    else:
        base = multi._build_base_config(multi._parser().parse_args(arguments))
    resolved = _act_ppo_config(base)
    assert resolved.exploration_learning_rate == 5e-5
    assert resolved.mean_bounds_coefficient == 0.03
    assert resolved.max_gradient_norm == 0.8
    assert resolved.critic_max_gradient_norm == 0.7
    assert resolved.exploration_max_gradient_norm == 0.6


@pytest.mark.parametrize("name,default", [
    ("ACTOR_LEARNING_RATE", "2e-4"),
    ("ACTOR_BACKBONE_LEARNING_RATE", "2e-5"),
])
@pytest.mark.parametrize("value", [None, "", "0.000002"])
def test_launcher_preserves_explicit_empty_learning_rate_for_checkpoint_inheritance(name, default, value):
    import os
    import subprocess

    # Evaluate only the real assignment, never launch the training script.
    script = Path(__file__).resolve().parents[2] / "scripts" / "finetune_act_ppo_so101.sh"
    assignment = next(line for line in script.read_text().splitlines() if line.startswith(f"{name}="))
    environment = dict(os.environ)
    environment.pop(name, None)
    if value is not None:
        environment[name] = value
    result = subprocess.run(["bash", "-c", assignment + '\nprintf "%s" "$' + name + '"'],
                            env=environment, check=True, capture_output=True, text=True)
    assert result.stdout == (default if value is None else value)



def test_cli_accepts_per_state_exploration_settings():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--state1-exploration-log-std",
            "-5.2",
            "--state2-exploration-log-std",
            "-4.1",
            "--state4-not-home-exploration-log-std",
            "-4.2",
            "--state4-home-exploration-log-std",
            "-5.3",
        ]
    )

    assert args.state1_exploration_log_std == -5.2
    assert args.state2_exploration_log_std == -4.1
    assert args.state4_not_home_exploration_log_std == -4.2
    assert args.state4_home_exploration_log_std == -5.3


def test_cli_accepts_per_state_exploration_and_actor_update_switches():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--no-state1-exploration-enabled",
            "--state2-exploration-enabled",
            "--no-state4-not-home-exploration-enabled",
            "--state4-home-exploration-enabled",
            "--state1-actor-update-enabled",
            "--no-state2-actor-update-enabled",
            "--no-state4-not-home-actor-update-enabled",
            "--state4-home-actor-update-enabled",
        ]
    )

    assert args.state1_exploration_enabled is False
    assert args.state2_exploration_enabled is True
    assert args.state4_not_home_exploration_enabled is False
    assert args.state4_home_exploration_enabled is True
    assert args.state1_actor_update_enabled is True
    assert args.state2_actor_update_enabled is False
    assert args.state4_not_home_actor_update_enabled is False
    assert args.state4_home_actor_update_enabled is True


def test_per_state_switches_default_to_enabled():
    args = _parser().parse_args(
        ["--act-checkpoint", "/tmp/pretrained_model"]
    )

    assert args.state1_exploration_enabled is True
    assert args.state2_exploration_enabled is True
    assert args.state4_not_home_exploration_enabled is True
    assert args.state4_home_exploration_enabled is True
    assert args.state1_actor_update_enabled is True
    assert args.state2_actor_update_enabled is True
    assert args.state4_not_home_actor_update_enabled is True
    assert args.state4_home_actor_update_enabled is True


def test_cli_accepts_recoverable_episode_failure_limit():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--max-consecutive-episode-failures",
            "7",
        ]
    )
    assert args.max_consecutive_episode_failures == 7


def test_recoverable_episode_failure_limit_defaults_to_three():
    args = _parser().parse_args(
        ["--act-checkpoint", "/tmp/pretrained_model"]
    )
    assert args.max_consecutive_episode_failures == 3


def test_cli_accepts_new_state3_stage_settings():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--state3-exploration-log-std",
            "-4.25",
            "--no-state3-exploration-enabled",
            "--no-state3-actor-update-enabled",
            "--state4-not-home-exploration-log-std",
            "-4.35",
            "--state4-home-exploration-log-std",
            "-5.25",
        ]
    )
    assert args.state3_exploration_log_std == -4.25
    assert args.state3_exploration_enabled is False
    assert args.state3_actor_update_enabled is False
    assert args.state4_not_home_exploration_log_std == -4.35
    assert args.state4_home_exploration_log_std == -5.25


def test_new_state3_stage_switches_default_to_enabled():
    args = _parser().parse_args(["--act-checkpoint", "/tmp/pretrained_model"])
    assert args.state3_exploration_enabled is True
    assert args.state3_actor_update_enabled is True
def test_cli_accepts_final_target_absent_window():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--max-steps-per-episode",
            "450",
            "--target-delay-s",
            "3.0",
            "--final-target-absent-s",
            "3.0",
        ]
    )

    assert args.final_target_absent_s == 3.0


def test_new_cli_defaults_select_pick_place_without_target_deletion():
    from so101_capstone.training.act_ppo_multiworld_gazebo import _build_base_config
    args = _parser().parse_args(["--act-checkpoint", "/tmp/act"])
    config = _build_base_config(args)
    assert config.reward_profile == "pick_place_v2"
    assert config.final_target_absent_steps == 0
    assert config.gamma == .997
    assert config.task_mode == "mixed"


def test_pick_place_rejects_external_target_delete_and_gamma_mismatch(tmp_path):
    import json
    from so101_capstone.training.act_ppo_gazebo import _resolved_reward
    with pytest.raises(ValueError, match="deleting"):
        ActPpoGazeboConfig(act_checkpoint=Path("act"), reward_profile="pick_place_v2")
    path = tmp_path / "reward.json"
    path.write_text(json.dumps({"gamma": .9}))
    config = ActPpoGazeboConfig(act_checkpoint=Path("act"), reward_profile="pick_place_v2",
                               final_target_absent_s=0, reward_config=path)
    with pytest.raises(ValueError, match="gamma"):
        _resolved_reward(config)


def test_resume_signature_tracks_resolved_randomization_and_rejects_idle_completed(monkeypatch, tmp_path):
    import copy
    import json
    from dataclasses import replace
    from so101_capstone.environments.environment_randomization import EnvironmentRandomizationConfig
    from so101_capstone.training.act_ppo_gazebo import _execution_signature, _resolved_reward

    payload = copy.deepcopy(dict(EnvironmentRandomizationConfig.load().payload))
    for field in ("position", "size"):
        payload["goal_region"][field]["enabled"] = False
    path = tmp_path / "randomization.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "true")
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_CONFIG", str(path))
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_SEED", "123")
    config = ActPpoGazeboConfig(act_checkpoint=Path("act"), reward_profile="pick_place_v2",
                               final_target_absent_s=0)
    reward = _resolved_reward(config)
    first = _execution_signature(config, reward, world_count=2)
    signature = first["environment_randomization"]
    assert signature["config"] == payload
    assert len(signature["config_sha256"]) == 64
    assert signature["world_indices"] == [0, 1]
    assert signature["base_seed"] == 123
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_SEED", "124")
    assert first != _execution_signature(config, reward, world_count=2)
    assert first != _execution_signature(config, reward, world_count=3)
    with pytest.raises(ValueError, match="idle_completed"):
        _execution_signature(replace(config, task_mode="mixed"), reward)
    payload["goal_region"]["size"]["enabled"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fixed goal"):
        _execution_signature(config, reward)


def test_export_carries_exact_resolved_reward_gamma(tmp_path):
    import json
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    from so101_capstone.training.act_ppo_gazebo import _copy_reward_config_for_export
    reward = PickPlaceRewardConfig(gamma=.995)
    source = tmp_path / "resolved_reward.json"
    source.write_text(json.dumps(reward.to_dict()))
    standalone = tmp_path / "checkpoints" / "000001" / "pretrained_model"
    _copy_reward_config_for_export(tmp_path, standalone)
    restored = PickPlaceRewardConfig.load(standalone / "resolved_reward.json")
    assert restored.gamma == .995
    assert restored.config_hash == reward.config_hash
