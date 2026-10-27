"""Bounded KL retries preserve the old policy reference and full epoch state."""

import copy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from so101_capstone.policies.so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS, policy_positions_to_gazebo,
)
from so101_capstone.training.act_ppo import ActPpoConfig
from so101_capstone.training.act_ppo_diagnostics import phase_policy_diagnostics
from test_act_ppo_optimizer_numerics import collect, make_trainer


def assert_state_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("settings", [
    {"kl_backoff_max_retries": -1}, {"kl_backoff_max_retries": True},
    {"kl_backoff_max_retries": 1.5}, {"kl_backoff_factor": 1.},
    {"kl_backoff_factor": 0.}, {"kl_backoff_factor": float("nan")},
    {"kl_backoff_min_scale": 0.}, {"kl_backoff_min_scale": 1.1},
])
def test_invalid_backoff_config_rejected(settings):
    with pytest.raises(ValueError):
        ActPpoConfig(**settings)


def test_real_kl_retry_matches_one_small_lr_epoch_including_optimizer_critic_and_rng(monkeypatch):
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    settings = dict(actor_learning_rate=1., kl_backoff_max_retries=6, minibatch_size=1)
    trainer = make_trainer(**settings)
    rollout = collect(trainer)
    initial = trainer._epoch_rollback_snapshot()
    rng = trainer._rng_snapshot()
    permutations = []
    # A stochastic critic or future optimizer hook may use any of these RNGs.
    # Rejected attempts must not consume extra randomness in the accepted path.
    def consume_rng_on_step(target):
        original_step = target.optimizer.step
        def step(*args, **kwargs):
            torch.rand(3)
            np.random.random(3)
            random.random()
            return original_step(*args, **kwargs)
        monkeypatch.setattr(target.optimizer, "step", step)
    consume_rng_on_step(trainer)
    original_randperm = torch.randperm

    def record_permutation(*args, **kwargs):
        value = original_randperm(*args, **kwargs)
        permutations.append(value.clone())
        return value

    monkeypatch.setattr(torch, "randperm", record_permutation)
    actual = trainer.update(rollout)
    final_rng = trainer._rng_snapshot()
    assert actual.completed_ppo_epochs == 1
    assert not actual.early_stopped
    assert actual.kl_backoff_retry_count > 0
    assert actual.kl_backoff_attempt_count == actual.kl_backoff_retry_count + 1
    assert actual.kl_backoff_rejection_count == actual.kl_backoff_retry_count
    assert actual.kl_backoff_attempts[0]["pre_clip_policy_kl"] > actual.target_kl
    assert actual.pre_clip_policy_kl <= actual.target_kl
    assert all(torch.equal(permutations[0], item) for item in permutations)
    assert actual.actor_parameter_changed_fraction > 0
    assert actual.critic_parameter_changed_fraction > 0
    assert actual.critic_learning_rate == trainer.config.critic_learning_rate
    assert trainer.optimizer.param_groups[-2]["lr"] == trainer.config.critic_learning_rate
    assert actual.exploration_learning_rate == trainer.config.exploration_learning_rate * actual.policy_learning_rate_scale

    control = make_trainer(**settings)
    control._restore_epoch_rollback_snapshot(initial)
    control._set_policy_learning_rate_scale(actual.policy_learning_rate_scale)
    control._restore_rng_snapshot(rng)
    consume_rng_on_step(control)
    expected = control.update(rollout)
    assert expected.kl_backoff_attempt_count == 1
    assert_state_equal(trainer._epoch_rollback_snapshot(), control._epoch_rollback_snapshot())
    assert torch.equal(torch.get_rng_state(), final_rng["torch"])
    assert random.getstate() == final_rng["python"]
    np.testing.assert_array_equal(np.random.get_state()[1], final_rng["numpy"][1])
    assert actual.value_loss == expected.value_loss
    assert actual.pre_clip_policy_kl == expected.pre_clip_policy_kl
    assert actual.minibatch_count == 2  # Rejected steps never count as learning.
    # The accepted candidate and its saved diagnostics still compare to rollout means.
    batch = rollout.batch("cpu")
    manual = trainer._policy_ratio_diagnostics(batch, torch.arange(2))
    assert manual.pre_clip_policy_kl == actual.pre_clip_policy_kl
    assert actual.phase_diagnostics == manual.phase_diagnostics
    assert "unknown" in actual.phase_diagnostics


def test_all_rejected_epochs_restore_every_weight_and_optimizer_but_keep_smallest_lr(monkeypatch):
    trainer = make_trainer(kl_backoff_max_retries=8, kl_backoff_min_scale=.25, ppo_epochs=4)
    rollout = collect(trainer)
    before = trainer._epoch_rollback_snapshot()
    original = trainer._policy_ratio_diagnostics
    candidate_calls = 0

    def always_reject(batch, indices):
        nonlocal candidate_calls
        result = original(batch, indices)
        # Exact pre-update replay remains strict and genuine; candidates fail.
        if torch.equal(trainer.policy.model.actions.detach(), before["policy"]["model.actions"]):
            return result
        candidate_calls += 1
        return replace(result, pre_clip_policy_kl=.03)

    monkeypatch.setattr(trainer, "_policy_ratio_diagnostics", always_reject)
    metrics = trainer.update(rollout)
    assert metrics.early_stopped and metrics.completed_ppo_epochs == 0
    assert metrics.minibatch_count == 0
    assert metrics.kl_backoff_attempt_count == candidate_calls == 3
    assert metrics.kl_backoff_retry_count == 2
    assert metrics.kl_backoff_at_min_scale
    assert metrics.policy_learning_rate_scale == .25
    assert metrics.actor_parameter_changed_fraction == 0.
    assert metrics.pre_clip_policy_kl == 0.
    after = trainer._epoch_rollback_snapshot()
    for key in ("policy", "critic", "exploration"):
        assert_state_equal(before[key], after[key])
    before_optimizer = copy.deepcopy(before["optimizer"])
    for i, group in enumerate(before_optimizer["param_groups"]):
        if i != len(before_optimizer["param_groups"]) - 2:
            group["lr"] *= .25
    assert_state_equal(before_optimizer, after["optimizer"])
    assert all(parameter.grad is None for parameter in trainer._parameters())
    again = trainer.update(rollout)
    assert again.kl_backoff_attempt_count == 1  # Floor cannot spin forever.
    assert again.kl_backoff_at_min_scale
    assert again.policy_learning_rate_scale == .25


def test_precheck_mismatch_is_never_retried_or_hidden(monkeypatch):
    trainer = make_trainer(kl_backoff_max_retries=6)
    rollout = collect(trainer)
    rollout.log_probabilities[0] += 1.
    before = trainer._epoch_rollback_snapshot()
    with pytest.raises(RuntimeError, match="before|pre-update|mismatch|rollout"):
        trainer.update(rollout)
    assert trainer.policy_learning_rate_scale == 1.
    assert_state_equal(before, trainer._epoch_rollback_snapshot())


def test_numerical_failure_restores_state_without_repeating_invalid_computation(monkeypatch):
    trainer = make_trainer(kl_backoff_max_retries=6)
    rollout = collect(trainer)
    before = trainer._epoch_rollback_snapshot()
    original_step = trainer.optimizer.step

    def corrupt_step(*args, **kwargs):
        original_step(*args, **kwargs)
        with torch.no_grad():
            trainer.critic.value.fill_(float("nan"))

    monkeypatch.setattr(trainer.optimizer, "step", corrupt_step)
    metrics = trainer.update(rollout)
    assert metrics.kl_backoff_attempt_count == 1
    assert metrics.kl_backoff_attempts[0]["numerical_error"]
    assert metrics.early_stopped
    assert_state_equal(before, trainer._epoch_rollback_snapshot())


def test_scale_persists_in_checkpoint_and_resume_can_change_retry_controls(tmp_path):
    trainer = make_trainer(kl_backoff_max_retries=5)
    trainer._set_policy_learning_rate_scale(.125)
    trainer.update(collect(trainer))
    path = trainer.save_checkpoint(tmp_path / "backoff.pt")
    restored = make_trainer(kl_backoff_max_retries=2, kl_backoff_factor=.25,
                            kl_backoff_min_scale=1e-3)
    restored.load_checkpoint(path)
    assert restored.policy_learning_rate_scale == .125
    assert restored.actor_learning_rate == restored.config.actor_learning_rate * .125
    assert_state_equal(trainer._epoch_rollback_snapshot(), restored._epoch_rollback_snapshot())
    metrics = restored.update(collect(restored))
    assert metrics.policy_learning_rate_scale == .125
    assert metrics.actor_learning_rate == restored.actor_learning_rate


def test_old_schema3_defaults_to_scale_one_and_rejects_missing_scale_for_reduced_lr(tmp_path):
    original = make_trainer()
    payload = copy.deepcopy(original.checkpoint_payload())
    payload.pop("learning_rate_state")
    for name in ("kl_backoff_max_retries", "kl_backoff_factor", "kl_backoff_min_scale"):
        payload["config"].pop(name)
    path = tmp_path / "legacy-schema3.pt"
    torch.save(payload, path)
    restored = make_trainer(kl_backoff_max_retries=5)
    restored.load_checkpoint(path)
    assert restored.policy_learning_rate_scale == 1.
    payload["optimizer_state_dict"]["param_groups"][0]["lr"] *= .5
    torch.save(payload, path)
    before = restored._epoch_rollback_snapshot()
    with pytest.raises(ValueError, match="learning rates"):
        restored.load_checkpoint(path)
    assert_state_equal(before, restored._epoch_rollback_snapshot())


@pytest.mark.parametrize("mutation", ["scale", "base", "critic_lr"])
def test_inconsistent_checkpoint_lr_state_rejected_before_weight_mutation(tmp_path, mutation):
    source = make_trainer()
    payload = copy.deepcopy(source.checkpoint_payload())
    if mutation == "scale":
        payload["learning_rate_state"]["policy_scale"] = .5
    elif mutation == "base":
        payload["learning_rate_state"]["base_actor_learning_rate"] *= 2
    else:
        payload["optimizer_state_dict"]["param_groups"][-2]["lr"] *= .5
    payload["act_policy_state_dict"]["model.actions"].add_(10.)
    path = tmp_path / "corrupt.pt"
    torch.save(payload, path)
    target = make_trainer()
    before = target._epoch_rollback_snapshot()
    with pytest.raises(ValueError, match="learning rate"):
        target.load_checkpoint(path)
    assert_state_equal(before, target._epoch_rollback_snapshot())


def test_phase_diagnostics_use_clipped_calibrated_radians_and_unknown_legacy_context():
    old = torch.tensor([[0., 0., 0., 0., 0., 50.], [99., 0., 0., 0., 0., 100.]])
    new = torch.tensor([[1., 2., 3., 4., 5., 51.], [110., 0., 0., 0., 0., 110.]])
    context = torch.zeros((2, 93))
    context[0, 0], context[1, 2] = 1., 1.
    kl = torch.tensor([.01, .9])
    result = phase_policy_diagnostics(kl, old, new, context)
    assert set(result) == {"wait_home", "grasp_lift"}
    for i, phase in enumerate(("wait_home", "grasp_lift")):
        expected = np.asarray(policy_positions_to_gazebo(new[i].tolist())) - np.asarray(policy_positions_to_gazebo(old[i].tolist()))
        assert result[phase]["transition_count"] == 1
        assert result[phase]["clipped_mean_command_rms_rad"] == pytest.approx(np.sqrt(np.mean(expected ** 2)))
        for calibration, delta in zip(SIM_JOINT_CALIBRATIONS, expected, strict=True):
            assert result[phase]["clipped_mean_command_per_joint_rms_rad"][calibration.name] == pytest.approx(abs(delta))
    assert result["grasp_lift"]["clipped_mean_command_per_joint_rms_rad"]["gripper"] == 0.
    legacy = phase_policy_diagnostics(kl, old, new, torch.zeros((2, 0)))
    assert set(legacy) == {"unknown"}
    assert legacy["unknown"]["transition_count"] == 2
    invalid = phase_policy_diagnostics(kl, old, new, torch.zeros((2, 93)))
    assert set(invalid) == {"unknown"}


def test_retry_programming_failure_restores_epoch_start_including_rng_and_lr(monkeypatch):
    trainer = make_trainer(kl_backoff_max_retries=3)
    rollout = collect(trainer)
    before = trainer._epoch_rollback_snapshot()
    rng = trainer._rng_snapshot()
    original = trainer._policy_ratio_diagnostics
    calls = 0
    def fail_second_candidate(batch, indices):
        nonlocal calls
        calls += 1
        result = original(batch, indices)
        if calls == 2:
            return replace(result, pre_clip_policy_kl=.03)
        if calls == 3:
            raise RuntimeError("deliberate programming failure")
        return result
    monkeypatch.setattr(trainer, "_policy_ratio_diagnostics", fail_second_candidate)
    with pytest.raises(RuntimeError, match="deliberate programming"):
        trainer.update(rollout)
    assert trainer.policy_learning_rate_scale == 1.
    assert trainer.update_index == 0
    assert_state_equal(before, trainer._epoch_rollback_snapshot())
    assert torch.equal(torch.get_rng_state(), rng["torch"])


def test_original_rollout_reference_survives_multiple_epochs_and_retries(monkeypatch):
    trainer = make_trainer(actor_learning_rate=1., kl_backoff_max_retries=6, ppo_epochs=3)
    rollout = collect(trainer)
    original_means = torch.tensor(np.asarray(rollout.policy_action_means))
    original_stds = torch.tensor(np.asarray(rollout.policy_action_log_stds))
    original = trainer._policy_ratio_diagnostics
    observed = []
    def check_reference(batch, indices):
        assert torch.equal(batch.old_action_means, original_means)
        assert torch.equal(batch.old_action_log_stds, original_stds)
        result = original(batch, indices)
        observed.append(result.pre_clip_policy_kl)
        return result
    monkeypatch.setattr(trainer, "_policy_ratio_diagnostics", check_reference)
    metrics = trainer.update(rollout)
    assert metrics.completed_ppo_epochs == 3
    assert metrics.kl_backoff_attempt_count > 3
    assert metrics.pre_clip_policy_kl <= trainer.config.target_kl
    assert observed[0] == 0.
    assert metrics.pre_clip_policy_kl == observed[-1]


def test_backbone_group_scales_with_actor_and_exploration_but_not_critic():
    from so101_capstone.training.act_ppo import ActPpoTrainer
    from test_act_ppo_optimizer_numerics import ConstantAct, ConstantCritic
    policy = ConstantAct()
    policy.model.backbone = torch.nn.Linear(1, 1)
    config = ActPpoConfig()
    trainer = ActPpoTrainer(policy, ConstantCritic(config), config=config)
    original_rates = [group["lr"] for group in trainer.optimizer.param_groups]
    assert len(original_rates) == 4
    trainer._set_policy_learning_rate_scale(.1)
    actual = [group["lr"] for group in trainer.optimizer.param_groups]
    assert actual == pytest.approx([original_rates[0] * .1, original_rates[1] * .1,
                                    original_rates[2], original_rates[3] * .1])


def test_restore_warm_optimizer_never_aliases_or_mutates_epoch_snapshot():
    torch.manual_seed(25)
    trainer = make_trainer()
    assert trainer.update(collect(trainer)).completed_ppo_epochs == 1
    snapshot = trainer._epoch_rollback_snapshot()
    frozen = copy.deepcopy(snapshot)
    trainer._restore_epoch_rollback_snapshot(snapshot)
    saved_states = snapshot["optimizer"]["state"].values()
    for live, saved in zip(trainer.optimizer.state.values(), saved_states, strict=True):
        assert set(live) == {"step", "exp_avg", "exp_avg_sq"}
        for name, tensor in live.items():
            assert tensor.data_ptr() != saved[name].data_ptr(), name
    # Reusing the same snapshot must remain valid after arbitrary rejected work.
    assert trainer.update(collect(trainer)).completed_ppo_epochs == 1
    assert_state_equal(snapshot, frozen)
    trainer._restore_epoch_rollback_snapshot(snapshot)
    assert_state_equal(trainer._epoch_rollback_snapshot(), frozen)


def test_warm_adam_multiple_retries_equal_direct_small_lr_update(monkeypatch):
    torch.manual_seed(5)
    settings = dict(actor_learning_rate=1., kl_backoff_max_retries=6, minibatch_size=1)
    trainer = make_trainer(**settings)
    # Populate Adam moments with an accepted conservative update, then deliberately
    # try a large LR to exercise backoff on a nonempty epoch-start optimizer state.
    trainer._set_policy_learning_rate_scale(.01)
    assert trainer.update(collect(trainer)).completed_ppo_epochs == 1
    trainer._set_policy_learning_rate_scale(1.)
    rollout = collect(trainer)
    initial = trainer._epoch_rollback_snapshot()
    rng = trainer._rng_snapshot()
    assert initial["optimizer"]["state"]
    captured_epoch_snapshots = []
    original_snapshot = trainer._epoch_rollback_snapshot
    def capture_snapshot():
        snapshot = original_snapshot()
        captured_epoch_snapshots.append((snapshot, copy.deepcopy(snapshot)))
        return snapshot
    monkeypatch.setattr(trainer, "_epoch_rollback_snapshot", capture_snapshot)

    actual = trainer.update(rollout)
    assert actual.completed_ppo_epochs == 1
    assert actual.kl_backoff_retry_count >= 2
    for snapshot, frozen in captured_epoch_snapshots:
        assert_state_equal(snapshot, frozen)

    control = make_trainer(**settings)
    control._restore_epoch_rollback_snapshot(initial)
    control._set_policy_learning_rate_scale(actual.policy_learning_rate_scale)
    control._restore_rng_snapshot(rng)
    expected = control.update(rollout)
    assert expected.kl_backoff_attempt_count == 1
    assert_state_equal(original_snapshot(), control._epoch_rollback_snapshot())
    assert actual.value_loss == expected.value_loss
    assert actual.pre_clip_policy_kl == expected.pre_clip_policy_kl
    for state in trainer.optimizer.state.values():
        assert state["step"].item() == 4  # Two warmup + two accepted steps only.


def test_warm_adam_all_rejected_restores_moments_counters_and_snapshot(monkeypatch):
    torch.manual_seed(29)
    trainer = make_trainer(kl_backoff_max_retries=2)
    assert trainer.update(collect(trainer)).completed_ppo_epochs == 1
    rollout = collect(trainer)
    before = trainer._epoch_rollback_snapshot()
    original_snapshot = trainer._epoch_rollback_snapshot
    captured_snapshots = []
    def capture_snapshot():
        snapshot = original_snapshot()
        captured_snapshots.append((snapshot, copy.deepcopy(snapshot)))
        return snapshot
    monkeypatch.setattr(trainer, "_epoch_rollback_snapshot", capture_snapshot)
    original_diagnostic = trainer._policy_ratio_diagnostics
    calls = 0
    def reject_every_candidate(batch, indices):
        nonlocal calls
        calls += 1
        result = original_diagnostic(batch, indices)
        return result if calls == 1 else replace(result, pre_clip_policy_kl=.03)
    monkeypatch.setattr(trainer, "_policy_ratio_diagnostics", reject_every_candidate)

    metrics = trainer.update(rollout)
    assert metrics.completed_ppo_epochs == 0
    assert metrics.kl_backoff_rejection_count == 3
    assert metrics.minibatch_count == 0
    assert metrics.policy_learning_rate_scale == .25
    after = original_snapshot()
    for name in ("policy", "critic", "exploration"):
        assert_state_equal(before[name], after[name])
    assert_state_equal(before["optimizer"]["state"], after["optimizer"]["state"])
    for state in trainer.optimizer.state.values():
        assert state["step"].item() == 1
    for snapshot, frozen in captured_snapshots:
        assert_state_equal(snapshot, frozen)
