"""CPU behavior tests for executed ACT raw-output overflow regularization."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from lerobot.policies.act.bounded_actions import bound_action_means
from so101_capstone.policies.so101_sim_calibration import gazebo_positions_to_policy
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.training.act_ppo import (
    ActPpoConfig, ActPpoRolloutBuffer, ActPpoTrainer, actor_observation_tensors,
    evaluate_bounded_policy_actions, policy_action_bounds,
    predict_action_chunk_with_grad, predict_action_chunk_with_raw_grad,
    raw_action_bounds_statistics, sample_bounded_policy_actions,
)
from test_act_ppo_optimizer_numerics import ConstantCritic, collect, make_trainer
from test_act_ppo import TinyActPolicy
from test_act_ppo_temporal_ensemble import (
    TinyTemporalPolicy, collect as collect_temporal, observations,
)


RAW_METRICS = (
    "raw_mean_bounds_loss", "raw_mean_out_of_bounds_fraction",
    "raw_mean_overflow_normalized", "raw_mean_bound_derivative_mean",
)


def statistics(raw, softness=0.02):
    lower, upper = policy_action_bounds(device=raw.device, dtype=raw.dtype)
    return raw_action_bounds_statistics(raw, lower=lower, upper=upper, softness=softness)


def table_trainer(*, coeff=None, chunk_size=3, **settings):
    config = replace(ActPpoConfig(
        critic_image_height=8, critic_image_width=8, critic_hidden_dim=16,
        minibatch_size=2, ppo_epochs=1, actor_learning_rate=1e-3,
        raw_mean_bounds_coefficient=0.1,
    ), **settings)
    policy = TinyTemporalPolicy(chunk_size=chunk_size, coeff=coeff)
    with torch.no_grad():
        policy.model.predictions.zero_()
        policy.model.predictions[..., -1] = 50.
    return ActPpoTrainer(policy, ConstantCritic(config), config=config)


def non_temporal_rollout(trainer, action_indices=(0, 0)):
    rollout = ActPpoRolloutBuffer(trainer.config)
    for step, action_index in enumerate(action_indices):
        actor_observation, critic_observation = observations(step)
        reference = rollout.add_actor_observation(actor_observation)
        with torch.no_grad():
            chunk = predict_action_chunk_with_grad(
                trainer.policy, actor_observation_tensors(actor_observation, device="cpu"))
            mean = chunk[0, action_index]
            log_std = trainer.exploration.value()
            action, log_probability, sample = sample_bounded_policy_actions(
                mean, log_std, return_latent=True)
        last = step == len(action_indices) - 1
        rollout.add(
            critic_observation, actor_observation_index=reference, action_index=action_index,
            policy_action=action.numpy(), policy_action_latent=sample.numpy(),
            policy_action_mean=mean.numpy(), policy_action_log_std=log_std.detach().numpy(),
            log_probability=log_probability.item(), value=0., reward=0., next_value=0.,
            terminated=last, episode_end=last,
        )
    return rollout


@pytest.mark.parametrize("invalid", [-0.1, float("inf"), float("nan"), True])
def test_raw_bounds_coefficient_requires_finite_nonnegative_number(invalid):
    with pytest.raises(ValueError, match="raw_mean_bounds_coefficient"):
        ActPpoConfig(raw_mean_bounds_coefficient=invalid)


def test_endpoints_and_actual_home_have_zero_raw_penalty_and_zero_gradient():
    lower, upper = policy_action_bounds(device="cpu", dtype=torch.float64)
    home = torch.as_tensor(gazebo_positions_to_policy(
        PickPlaceRewardConfig().home_joint_positions_rad).copy(), dtype=torch.float64)
    near_limit = torch.tensor([99., -99., 98., -98., 0., 99.], dtype=torch.float64)
    raw = torch.stack((lower, upper, home, near_limit)).requires_grad_()
    measured = statistics(raw)
    assert all(measured[name].shape == (4,) for name in RAW_METRICS)
    for name in RAW_METRICS[:3]:
        torch.testing.assert_close(measured[name], torch.zeros(4, dtype=raw.dtype), atol=0, rtol=0)
    measured["raw_mean_bounds_loss"].sum().backward()
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw), atol=0, rtol=0)


def test_overflow_is_scaled_by_each_joint_range_and_has_inward_gradient():
    raw = torch.tensor([[120., -110., 0., 0., 0., 110.]], dtype=torch.float64, requires_grad=True)
    measured = statistics(raw)
    assert measured["raw_mean_bounds_loss"].item() == pytest.approx((.2**2 + .1**2 + .2**2) / 6)
    assert measured["raw_mean_out_of_bounds_fraction"].item() == pytest.approx(3 / 6)
    assert measured["raw_mean_overflow_normalized"].item() == pytest.approx(.5 / 6)
    measured["raw_mean_bounds_loss"].sum().backward()
    expected = torch.tensor([[2*.2/100/6, -2*.1/100/6, 0., 0., 0., 2*.2/50/6]], dtype=raw.dtype)
    torch.testing.assert_close(raw.grad, expected, atol=1e-14, rtol=1e-12)
    assert not any(measured[name].requires_grad for name in RAW_METRICS[1:])


def test_reported_bound_derivative_matches_actual_soft_bound_gradient():
    agent = make_trainer()
    raw = torch.tensor([[0., 99., -110., 120., -100., 110.]], dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(bound_action_means(raw, agent.policy.config).sum(), raw)[0]
    measured = statistics(raw, softness=agent.policy.config.action_bound_softness)
    torch.testing.assert_close(measured["raw_mean_bound_derivative_mean"], derivative.mean(-1))
    torch.testing.assert_close(statistics(raw, softness=None)["raw_mean_bound_derivative_mean"],
                               torch.ones(1, dtype=raw.dtype), atol=0, rtol=0)


def test_raw_and_bounded_predictions_share_one_forward_and_keep_raw_gradient():
    agent = make_trainer(first_joint=130.)
    batch = actor_observation_tensors(observations(0)[0], device="cpu")
    calls = []
    hook = agent.policy.model.register_forward_hook(lambda *_: calls.append(1))
    try:
        bounded, raw = predict_action_chunk_with_raw_grad(agent.policy, batch)
    finally:
        hook.remove()
    assert calls == [1]
    assert torch.all(raw[..., 0] == 130.)
    assert torch.all(bounded[..., 0] < 100.)
    torch.testing.assert_close(bounded, predict_action_chunk_with_grad(agent.policy, batch), atol=0, rtol=0)
    statistics(raw)["raw_mean_bounds_loss"].mean().backward()
    assert torch.all(agent.policy.model.actions.grad[:, 0] > 0)
    assert torch.count_nonzero(agent.policy.model.actions.grad[:, 1:]) == 0


def test_raw_penalty_uses_physical_means_with_differentiable_frozen_unnormalization():
    policy = TinyActPolicy(chunk_size=1)
    config = ActPpoConfig(raw_mean_bounds_coefficient=.1)
    agent = ActPpoTrainer(policy, ConstantCritic(config), config=config)
    with torch.no_grad():
        policy.model.linear.weight.zero_()
        policy.model.linear.bias.copy_(torch.tensor([60., 0., 0., 0., 0., 25.]))
    bounded, raw = predict_action_chunk_with_raw_grad(
        policy, actor_observation_tensors(observations(0)[0], device="cpu"))
    # This policy has frozen mean=0/std=2, with a @no_grad native unnormalizer.
    assert raw[0, 0, 0].item() == 120.
    assert raw[0, 0, -1].item() == 50.
    assert bounded[0, 0, 0] < 100.
    measured = statistics(raw)
    assert measured["raw_mean_bounds_loss"].item() == pytest.approx(.2**2 / 6, rel=1e-6)
    measured["raw_mean_bounds_loss"].sum().backward()
    assert policy.model.linear.bias.grad[0].item() == pytest.approx(2 * .2 / 100 / 6 * 2, rel=1e-6)
    assert all(parameter.grad is None for parameter in policy.unnormalize_outputs.parameters())


def test_non_temporal_regularization_uses_executed_action_indices_not_entire_chunk():
    agent = table_trainer()
    with torch.no_grad():
        agent.policy.model.predictions[..., 0] = 500.  # Unexecuted predictions must receive no gradient.
        agent.policy.model.predictions[0, 1, 0] = 120.
        agent.policy.model.predictions[1, 2, 0] = -130.
    rollout = non_temporal_rollout(agent, action_indices=(1, 2))
    batch = rollout.batch("cpu")
    indices = torch.tensor([1, 0])  # Shuffling must preserve the transition-to-chunk mapping.
    means, measured = agent._actor_means_and_raw_bounds(batch, indices)
    torch.testing.assert_close(means, agent._actor_means(batch, indices), atol=0, rtol=0)
    torch.testing.assert_close(measured["raw_mean_bounds_loss"],
                               torch.tensor([.3**2 / 6, .2**2 / 6]), atol=1e-8, rtol=1e-6)
    measured["raw_mean_bounds_loss"].sum().backward()
    grad = agent.policy.model.predictions.grad
    assert grad[0, 1, 0] > 0 and grad[1, 2, 0] < 0
    assert torch.count_nonzero(grad) == 2


@pytest.mark.parametrize("coeff", [0., .01, -.01])
def test_temporal_opposite_overflows_do_not_cancel_and_reach_each_historical_prediction(coeff):
    agent = table_trainer(coeff=coeff, chunk_size=3)
    with torch.no_grad():
        # Transition 2 uses exactly these three contributors, oldest first.
        agent.policy.model.predictions[0, 2, 0] = 120.
        agent.policy.model.predictions[1, 1, 0] = -120.
        agent.policy.model.predictions[2, 0, 0] = 130.
        # Finite but squaring this irrelevant raw tail would overflow float32.
        agent.policy.model.predictions[0, 0, 1] = 1e30
    rollout, expected_means = collect_temporal(agent, count=3)
    means, measured = agent._actor_means_and_raw_bounds(rollout.batch("cpu"), torch.tensor([2]))
    torch.testing.assert_close(means[0], expected_means[2], atol=0, rtol=0)
    weights = torch.exp(-coeff * torch.arange(3))
    weights /= weights.sum()
    expected_loss = (weights * torch.tensor([.2**2, .2**2, .3**2])).sum() / 6
    assert measured["raw_mean_bounds_loss"].item() == pytest.approx(expected_loss.item(), rel=2e-6)
    # Penalizing the averaged raw instead would see an in-range value here.
    assert abs(float((weights * torch.tensor([120., -120., 130.])).sum())) < 100.
    measured["raw_mean_bounds_loss"].sum().backward()
    grad = agent.policy.model.predictions.grad
    for index, (frame, action, overflow) in enumerate(((0, 2, .2), (1, 1, -.2), (2, 0, .3))):
        assert grad[frame, action, 0].item() == pytest.approx(float(weights[index]) * 2 * overflow / 100 / 6, rel=2e-6)
    assert torch.count_nonzero(grad) == 3


def test_positive_coefficient_reduces_actual_raw_overflow_with_zero_advantages():
    torch.manual_seed(101)
    agent = make_trainer(first_joint=130., actor_learning_rate=1e-2, raw_mean_bounds_coefficient=.1)
    rollout = collect(agent, rewards=(0., 0.))
    before = agent.policy.model.actions.detach().clone()
    exploration_before = agent.exploration.log_std.detach().clone()
    result = agent.update(rollout)
    assert result.completed_ppo_epochs == 1
    assert result.policy_loss == result.mean_bounds_loss == 0.
    assert result.raw_mean_bounds_loss > 0.
    assert result.raw_mean_out_of_bounds_fraction == pytest.approx(1 / 6)
    assert result.raw_mean_overflow_normalized == pytest.approx(.3 / 6, rel=1e-6)
    assert 0 < result.raw_mean_bound_derivative_mean < 1
    assert result.raw_mean_bounds_diagnostic_count == 2
    assert torch.all(agent.policy.model.actions[:, 0] < before[:, 0])
    torch.testing.assert_close(agent.policy.model.actions[:, 1:], before[:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(agent.exploration.log_std, exploration_before, atol=0, rtol=0)
    assert result.pre_update_max_abs_log_ratio == result.pre_update_pre_clip_policy_kl == 0.


def test_zero_coefficient_keeps_old_bounded_distribution_and_does_not_apply_raw_penalty(monkeypatch):
    torch.manual_seed(102)
    agent = make_trainer(first_joint=130., raw_mean_bounds_coefficient=0.)
    rollout = collect(agent, rewards=(0., 0.))
    batch = rollout.batch("cpu")
    means = agent._actor_means(batch, torch.arange(2))
    probability, _ = evaluate_bounded_policy_actions(
        means, batch.policy_actions, batch.old_action_log_stds,
        policy_action_latents=batch.policy_action_latents)
    torch.testing.assert_close(means, batch.old_action_means, atol=0, rtol=0)
    torch.testing.assert_close(probability, batch.old_log_probabilities, atol=0, rtol=0)
    before = agent.policy.model.actions.detach().clone()
    monkeypatch.setattr(agent, "_actor_means_and_raw_bounds",
                        lambda *args: pytest.fail("Disabled raw penalty must retain the old actor path"))
    result = agent.update(rollout)
    torch.testing.assert_close(agent.policy.model.actions, before, atol=0, rtol=0)
    assert result.raw_mean_bounds_diagnostic_count == 0
    assert result.raw_mean_bounds_loss == 0.
    assert result.pre_update_max_abs_log_ratio == 0.


def test_value_only_minibatch_does_not_regularize_actor_or_exploration():
    agent = make_trainer(first_joint=130., raw_mean_bounds_coefficient=100.)
    rollout = collect(agent, rewards=(1., 1.))
    rollout.actor_update_enabled[:] = [False, False]
    actor_before = agent.policy.model.actions.detach().clone()
    exploration_before = agent.exploration.log_std.detach().clone()
    critic_before = agent.critic.value.detach().clone()
    result = agent.update(rollout)
    assert result.raw_mean_bounds_diagnostic_count == 0
    assert result.raw_mean_bounds_loss == 0.
    torch.testing.assert_close(agent.policy.model.actions, actor_before, atol=0, rtol=0)
    torch.testing.assert_close(agent.exploration.log_std, exploration_before, atol=0, rtol=0)
    assert not torch.equal(agent.critic.value, critic_before)


def test_mixed_batch_regularizes_only_actor_enabled_transitions():
    agent = table_trainer()
    with torch.no_grad():
        agent.policy.model.predictions[0, 0, 0] = 120.
        agent.policy.model.predictions[1, 0, 0] = 180.
    rollout = non_temporal_rollout(agent)
    rollout.actor_update_enabled[:] = [True, False]
    before = agent.policy.model.predictions.detach().clone()
    result = agent.update(rollout)
    assert result.raw_mean_bounds_diagnostic_count == 1
    assert agent.policy.model.predictions[0, 0, 0] < before[0, 0, 0]
    torch.testing.assert_close(agent.policy.model.predictions[1], before[1], atol=0, rtol=0)


@pytest.mark.parametrize("accepted_epochs", [0, 1])
def test_rejected_epoch_discards_raw_statistics_and_restores_last_accepted_actor(monkeypatch, accepted_epochs):
    agent = make_trainer(first_joint=130., ppo_epochs=2, raw_mean_bounds_coefficient=.1)
    rollout = collect(agent, rewards=(0., 0.))
    expected = statistics(agent.policy.model.actions.detach())
    original = agent._policy_ratio_diagnostics
    candidates = []

    def reject_after_accepted_epochs(batch, indices):
        result = original(batch, indices)
        candidates.append(agent.policy.model.actions.detach().clone())
        # Call zero is the real precheck; inject a KL violation only after a real optimizer step.
        if len(candidates) == accepted_epochs + 2:
            return replace(result, pre_clip_policy_kl=agent.config.target_kl + 1.)
        return result

    monkeypatch.setattr(agent, "_policy_ratio_diagnostics", reject_after_accepted_epochs)
    metrics = agent.update(rollout)
    assert metrics.early_stopped
    assert metrics.completed_ppo_epochs == accepted_epochs
    assert metrics.kl_backoff_rejection_count == 1
    assert metrics.raw_mean_bounds_diagnostic_count == 2 * accepted_epochs
    for name in RAW_METRICS:
        assert getattr(metrics, name) == pytest.approx(expected[name].mean().item() if accepted_epochs else 0.)
    torch.testing.assert_close(agent.policy.model.actions, candidates[accepted_epochs], atol=0, rtol=0)


@pytest.mark.parametrize("legacy_missing_field", [False, True])
def test_checkpoint_can_resume_with_explicit_new_raw_coefficient_and_reports_change(tmp_path, capsys, legacy_missing_field):
    torch.manual_seed(103)
    source = make_trainer(first_joint=120., raw_mean_bounds_coefficient=0.)
    source.update(collect(source))  # Include populated optimizer state and counters.
    payload = source.checkpoint_payload()
    if legacy_missing_field:
        del payload["config"]["raw_mean_bounds_coefficient"]
    path = tmp_path / "raw_bounds_resume.pt"
    torch.save(payload, path)
    target = make_trainer(first_joint=-40., raw_mean_bounds_coefficient=.1)
    capsys.readouterr()
    target.load_checkpoint(path, map_location="cpu")
    log = capsys.readouterr().out
    assert "raw_mean_bounds_coefficient" in log and "0.1" in log
    assert target.config.raw_mean_bounds_coefficient == .1
    assert target.update_index == source.update_index
    assert target.total_environment_steps == source.total_environment_steps
    torch.testing.assert_close(target.policy.model.actions, source.policy.model.actions, atol=0, rtol=0)
    torch.testing.assert_close(target.exploration.log_std, source.exploration.log_std, atol=0, rtol=0)
    for parameter, state in source.optimizer.state.items():
        matching = next(p for p in target.optimizer.state if p.shape == parameter.shape)
        for key, value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(target.optimizer.state[matching][key], value, atol=0, rtol=0)


def test_old_checkpoint_missing_raw_coefficient_also_resumes_with_default_zero(tmp_path):
    source = make_trainer(first_joint=120.)
    payload = source.checkpoint_payload()
    del payload["config"]["raw_mean_bounds_coefficient"]
    path = tmp_path / "old_schema4.pt"
    torch.save(payload, path)
    target = make_trainer(first_joint=-40.)
    target.load_checkpoint(path, map_location="cpu")
    assert target.config.raw_mean_bounds_coefficient == 0.
    torch.testing.assert_close(target.policy.model.actions, source.policy.model.actions, atol=0, rtol=0)
