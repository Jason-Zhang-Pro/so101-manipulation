"""Keep backoff work and accepted per-phase movement visible in TensorBoard."""

import torch

from so101_capstone.training.act_ppo_diagnostics import phase_policy_diagnostics
from so101_capstone.training.act_ppo_tensorboard import DirectActPpoTensorBoard


class Writer:
    def __init__(self):
        self.scalars = {}
        self.flushes = 0

    def add_scalar(self, tag, value, step):
        self.scalars[(tag, step)] = value

    def flush(self):
        self.flushes += 1

    def close(self):
        pass


def test_backoff_and_phase_diagnostics_are_not_hidden_in_aggregate_kl():
    writer = Writer()
    logger = DirectActPpoTensorBoard(writer=writer, log_directory=None, url=None, server_pid=None)
    logger.log_update({
        "update_index": 4,
        "kl_backoff_attempt_count": 3,
        "kl_backoff_rejection_count": 2,
        "kl_backoff_retry_count": 2,
        "kl_backoff_at_min_scale": False,
        "policy_learning_rate_scale": 0.25,
        "phase_diagnostics": {
            "wait_home": {"transition_count": 20, "pre_clip_policy_kl": 0.03,
                          "clipped_mean_command_per_joint_rms_rad": {"gripper": 0.001, "wrist_roll": 0.002}},
            "approach": {"transition_count": 100, "pre_clip_policy_kl": 0.002,
                         "not_a_number": float("nan")},
        },
    }, total_environment_steps=7200)
    assert writer.scalars[("ppo/kl_backoff_retry_count", 4)] == 2
    assert writer.scalars[("learning_rate/policy_scale", 4)] == 0.25
    assert writer.scalars[("ppo/phase/wait_home/pre_clip_policy_kl", 4)] == 0.03
    assert writer.scalars[("ppo/phase/approach/pre_clip_policy_kl", 4)] == 0.002
    assert ("ppo/phase/approach/not_a_number", 4) not in writer.scalars
    assert writer.scalars[("ppo/phase/wait_home/clipped_mean_command_per_joint_rms_rad/gripper", 4)] == 0.001
    assert writer.scalars[("ppo/phase/wait_home/clipped_mean_command_per_joint_rms_rad/wrist_roll", 4)] == 0.002
    assert writer.flushes == 1


def test_old_metrics_without_phase_diagnostics_still_log():
    writer = Writer()
    logger = DirectActPpoTensorBoard(writer=writer, log_directory=None, url=None, server_pid=None)
    logger.log_update({"update_index": 1, "pre_clip_policy_kl": 0.01}, total_environment_steps=1800)
    assert writer.scalars[("ppo/pre_clip_policy_kl", 1)] == 0.01
    assert writer.flushes == 1


def test_bounded_phase_kl_suppresses_legacy_alias_without_losing_command_diagnostics():
    writer = Writer()
    logger = DirectActPpoTensorBoard(writer=writer, log_directory=None, url=None, server_pid=None)
    logger.log_update({
        "update_index": 5,
        "phase_diagnostics": {
            "approach": {"policy_kl": 0.2, "pre_clip_policy_kl": 0.8,
                         "policy_kl_p95": 0.3, "pre_clip_policy_kl_p95": 0.9,
                         "policy_kl_max": 0.4, "pre_clip_policy_kl_max": 1.0,
                         "clipped_mean_command_rms_rad": 0.02},
        },
    }, total_environment_steps=9000)
    for key, value in (("policy_kl", 0.2), ("policy_kl_p95", 0.3), ("policy_kl_max", 0.4)):
        assert writer.scalars[(f"ppo/phase/approach/{key}", 5)] == value
    assert not any("pre_clip" in tag for tag, _ in writer.scalars)
    assert writer.scalars[("ppo/phase/approach/clipped_mean_command_rms_rad", 5)] == 0.02


def test_phase_statistics_expose_canonical_kl_with_identical_compatibility_aliases():
    means = torch.tensor([[0., 0., 0., 0., 0., 50.], [1., 2., 3., 4., 5., 51.]])
    result = phase_policy_diagnostics(
        torch.tensor([0.1, 0.3]), means, means, torch.zeros((2, 0)),
    )["unknown"]
    for suffix in ("", "_p95", "_max"):
        assert result[f"policy_kl{suffix}"] == result[f"pre_clip_policy_kl{suffix}"]
    assert result["policy_kl"] == float(torch.tensor([0.1, 0.3]).double().mean())
    assert result["clipped_mean_command_rms_rad"] == 0.0
