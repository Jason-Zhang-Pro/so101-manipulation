from pathlib import Path

import pytest

from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig,
    _parser,
)
from so101_capstone.training.act_ppo_tensorboard import (
    DirectActPpoTensorBoard,
    _positive_port,
)


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


def _logger(writer):
    return DirectActPpoTensorBoard(
        writer=writer,
        log_directory=Path("/tmp/tensorboard"),
        url="http://127.0.0.1:6006/",
        server_pid=123,
    )


def test_direct_config_does_not_launch_tensorboard_by_default():
    config = ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/pretrained_model"),
    )

    assert config.tensorboard_enabled is False
    assert config.tensorboard_auto_open is False
    assert config.tensorboard_port == 6006


def test_update_logs_actual_stage_sample_ratio_at_update_index():
    writer = FakeWriter()
    _logger(writer).log_update({
        "update_index": 3,
        "stage_sampling": {"s0_steps": 1800, "s1_steps": 2400, "total_steps": 4200,
                           "target_s0_step_fraction": .4, "actual_s0_step_fraction": 3/7,
                           "ratio_error": 3/7 - .4, "ratio_tolerance": .05, "ratio_met": True},
    }, total_environment_steps=8000)
    scalars = {tag: (value, step) for tag, value, step in writer.scalars}
    assert scalars["stage_batch/s0_steps"] == (1800., 3)
    assert scalars["stage_batch/s1_steps"] == (2400., 3)
    assert scalars["stage_batch/actual_s0_step_fraction"] == (3/7, 3)
    assert scalars["stage_batch/target_s0_step_fraction"] == (.4, 3)


def test_discarded_steps_are_logged_without_waiting_for_ppo_update():
    writer = FakeWriter()
    logger = _logger(writer)
    snapshot = {
        "discarded_environment_steps": 311, "discarded_transition_count": 310,
        "gazebo_crash_discarded_environment_steps": 200,
        "gazebo_crash_discarded_transition_count": 199,
        "gazebo_crash_episode_count": 1, "failure_episode_count": 2,
        "unconfirmed_step_count": 1, "final_partial_rollout_discarded_transition_count": 0,
    }
    logger.log_discarded_steps(snapshot, event_index=2)
    assert writer.flush_count == 1
    assert set(writer.scalars) == {(f"discarded/{key}", float(value), 2) for key, value in snapshot.items()}
    DirectActPpoTensorBoard.disabled().log_discarded_steps(snapshot, event_index=2)


def test_cli_enables_tensorboard_and_browser_by_default():
    args = _parser().parse_args(
        ["--act-checkpoint", "/tmp/pretrained_model"]
    )

    assert args.tensorboard is True
    assert args.open_tensorboard is True
    assert args.tensorboard_port == 6006


def test_cli_can_disable_browser_without_disabling_logging():
    args = _parser().parse_args(
        [
            "--act-checkpoint",
            "/tmp/pretrained_model",
            "--no-open-tensorboard",
            "--tensorboard-port",
            "7007",
        ]
    )

    assert args.tensorboard is True
    assert args.open_tensorboard is False
    assert args.tensorboard_port == 7007


@pytest.mark.parametrize("value", [0, -1, 65536, True, 1.5])
def test_invalid_tensorboard_port_is_rejected(value):
    with pytest.raises(ValueError):
        if isinstance(value, int):
            ActPpoGazeboConfig(
                act_checkpoint=Path("/tmp/pretrained_model"),
                tensorboard_port=value,
            )
        else:
            _positive_port(value)


def test_every_update_is_written_and_flushed():
    writer = FakeWriter()
    logger = _logger(writer)

    logger.log_update(
        {
            "update_index": 3,
            "policy_loss": -0.25,
            "value_loss": 0.5,
            "entropy": 1.2,
            "approximate_kl": 0.01,
            "clip_fraction": 0.2,
            "gradient_norm": 4.0,
            "actor_learning_rate": 1e-5,
            "actor_backbone_learning_rate": 1e-6,
            "critic_learning_rate": 3e-4,
            "log_std_mean": -0.7,
            "transition_count": 1200,
            "minibatch_count": 40,
            "partial_rollout": False,
            "checkpoint_saved": False,
        },
        total_environment_steps=3600,
    )

    recorded = {(tag, step): value for tag, value, step in writer.scalars}
    assert recorded[("loss/policy_loss", 3)] == -0.25
    assert recorded[("loss/value_loss", 3)] == 0.5
    assert recorded[("progress/total_environment_steps", 3)] == 3600
    assert recorded[("progress/checkpoint_saved", 3)] == 0.0
    assert writer.flush_count == 1


def test_update_logs_analytic_kl_group_gradients_and_accepted_parameter_changes():
    writer = FakeWriter()
    values = {
        "mean_bounds_loss": 0.05,
        "pre_update_pre_clip_policy_kl": 0.0,
        "pre_clip_policy_kl": 0.01,
        "pre_clip_policy_kl_p95": 0.04,
        "pre_clip_policy_kl_max": 0.15,
        "actor_gradient_norm": 0.4,
        "critic_gradient_norm": 0.6,
        "exploration_gradient_norm": 0.1,
        "actor_parameter_relative_delta": 1e-5,
        "actor_parameter_changed_fraction": 0.1,
        "critic_parameter_relative_delta": 0.01,
        "critic_parameter_changed_fraction": 0.9,
        "exploration_parameter_relative_delta": 1e-4,
        "exploration_parameter_changed_fraction": 1.0,
        "exploration_learning_rate": 1e-4,
        "target_kl": 0.02,
        "configured_ppo_epochs": 4,
        "completed_ppo_epochs": 1,
        "early_stopped": True,
    }
    _logger(writer).log_update({"update_index": 2, **values}, total_environment_steps=1800)
    recorded = {(tag, step): value for tag, value, step in writer.scalars}
    assert recorded[("loss/mean_bounds_loss", 2)] == values["mean_bounds_loss"]
    for name in ("pre_update_pre_clip_policy_kl", "pre_clip_policy_kl",
                 "pre_clip_policy_kl_p95", "pre_clip_policy_kl_max", "target_kl"):
        assert recorded[(f"ppo/{name}", 2)] == values[name]
    for group in ("actor", "critic", "exploration"):
        for suffix in ("gradient_norm", "parameter_relative_delta", "parameter_changed_fraction"):
            name = f"{group}_{suffix}"
            assert recorded[(f"optimization/{name}", 2)] == values[name]
    assert recorded[("learning_rate/exploration", 2)] == 1e-4
    assert recorded[("ppo/configured_epochs", 2)] == 4
    assert recorded[("ppo/completed_epochs", 2)] == 1
    assert recorded[("ppo/early_stopped", 2)] == 1.0
    assert writer.flush_count == 1


@pytest.mark.parametrize("include_explicit_entropy", [False, True])
def test_entropy_alias_produces_only_one_explicitly_pre_clip_trace(include_explicit_entropy):
    writer = FakeWriter()
    payload = {"update_index": 1, "entropy": -10.0}
    if include_explicit_entropy:
        payload["pre_clip_gaussian_entropy"] = -9.0
    _logger(writer).log_update(payload, total_environment_steps=100)
    traces = [entry for entry in writer.scalars if "entropy" in entry[0]]
    assert traces == [("ppo/pre_clip_gaussian_entropy", -9.0 if include_explicit_entropy else -10.0, 1)]


def test_bounded_distribution_logs_canonical_kl_and_entropy_without_legacy_duplicates():
    writer = FakeWriter()
    payload = {"update_index": 4, "entropy": -1.0,
               "truncated_gaussian_entropy": -2.0, "pre_clip_gaussian_entropy": -3.0}
    for current, legacy in (
        ("pre_update_policy_kl", "pre_update_pre_clip_policy_kl"),
        ("policy_kl", "pre_clip_policy_kl"),
        ("policy_kl_p95", "pre_clip_policy_kl_p95"),
        ("policy_kl_max", "pre_clip_policy_kl_max"),
    ):
        payload[current], payload[legacy] = 0.1, 0.9
    _logger(writer).log_update(payload, total_environment_steps=1800)
    recorded = {(tag, step): value for tag, value, step in writer.scalars}
    assert recorded[("ppo/truncated_gaussian_entropy", 4)] == -2.0
    assert len([tag for tag, _ in recorded if "entropy" in tag]) == 1
    for key in ("pre_update_policy_kl", "policy_kl", "policy_kl_p95", "policy_kl_max"):
        assert recorded[(f"ppo/{key}", 4)] == 0.1
    assert not any("pre_clip" in tag for tag, _ in recorded)


def test_nonfinite_canonical_metrics_do_not_fall_back_to_legacy_aliases():
    writer = FakeWriter()
    _logger(writer).log_update({
        "update_index": 1, "policy_kl": float("inf"), "pre_clip_policy_kl": 0.1,
        "truncated_gaussian_entropy": float("nan"), "entropy": -1.0,
    }, total_environment_steps=1800)
    tags = [tag for tag, _, _ in writer.scalars]
    assert not any("policy_kl" in tag or "entropy" in tag for tag in tags)


@pytest.mark.parametrize("container", [tuple, list])
def test_each_exploration_axis_logs_its_standard_deviation_parameter_and_change(container):
    writer = FakeWriter()
    log_stds = container(-4.0 + 0.1 * joint for joint in range(6))
    deltas = container(1e-4 * (joint - 3) for joint in range(6))
    _logger(writer).log_update({
        "update_index": 4, "exploration_log_std": log_stds,
        "exploration_log_std_delta": deltas,
    }, total_environment_steps=3600)
    recorded = {(tag, step): value for tag, value, step in writer.scalars}
    for joint in range(6):
        assert recorded[(f"exploration/log_std/joint_{joint}", 4)] == log_stds[joint]
        assert recorded[(f"exploration/log_std_delta/joint_{joint}", 4)] == deltas[joint]


def test_update_logging_skips_nonfinite_diagnostics_and_incomplete_axis_arrays():
    writer = FakeWriter()
    _logger(writer).log_update({
        "update_index": 1, "pre_clip_policy_kl": float("inf"),
        "mean_bounds_loss": float("nan"), "pre_clip_gaussian_entropy": float("nan"),
        "exploration_log_std": [-4.0, float("nan"), -4.0, -4.0, -4.0, -4.0],
        "exploration_log_std_delta": [0.0],
    }, total_environment_steps=0)
    tags = [tag for tag, _, _ in writer.scalars]
    assert "ppo/pre_clip_policy_kl" not in tags
    assert "loss/mean_bounds_loss" not in tags
    assert "ppo/pre_clip_gaussian_entropy" not in tags
    assert "exploration/log_std/joint_1" not in tags
    assert "exploration/log_std/joint_5" in tags
    assert not any(tag.startswith("exploration/log_std_delta/") for tag in tags)


def test_episode_reward_and_status_are_written():
    writer = FakeWriter()
    logger = _logger(writer)

    logger.log_episode(
        {
            "episode_index": 1,
            "step_count": 400,
            "total_reward": 25.0,
            "terminated": True,
            "truncated": False,
            "reached_max_steps": False,
        }
    )

    recorded = {(tag, step): value for tag, value, step in writer.scalars}
    assert recorded[("episode/total_reward", 2)] == 25.0
    assert recorded[("episode/reward_per_step", 2)] == 0.0625
    assert recorded[("episode/terminated", 2)] == 1.0
    assert writer.flush_count == 1


def test_pick_place_metrics_do_not_confuse_valid_episode_with_task_success():
    writer = FakeWriter()
    _logger(writer).log_episode({
        "episode_index": 0, "step_count": 100, "total_reward": -2.,
        "valid": True, "task_mode": "pick_place", "truncated": True,
        "final_info": {"task_outcome": "running", "robot_at_home": True,
                       "robot_static": True, "home_error_rad": .01},
        "stage_reached": {"grasp_valid": 1},
        "reward_components_total": {"active_time": -1.},
        "action_diagnostics": {"idle": {"raw_mean_clamp_fraction": .2,
                                         "command_noise_rms_rad_per_joint": [0.] * 6}},
    })
    recorded = {tag: value for tag, value, step in writer.scalars}
    assert recorded["task/pick_place/success"] == 0
    assert recorded["task/stage_reached/grasp_valid"] == 1
    assert recorded["reward/active_time"] == -1
    assert recorded["action/idle/raw_mean_clamp_fraction"] == .2
