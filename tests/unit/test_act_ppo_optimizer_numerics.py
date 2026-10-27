"""Behavioral regressions for boundary correction and float32 PPO updates."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode
from so101_capstone.training.act_ppo import (
    ActPpoActorObservation, ActPpoConfig, ActPpoExploration, ActPpoRolloutBuffer, ActPpoTrainer,
    actor_observation_tensors, predict_action_chunk_with_grad, sample_bounded_policy_actions,
    evaluate_bounded_policy_actions,
)
from so101_capstone.training.residual_ppo import CompressedResidualObservation


class ConstantActCore(nn.Module):
    def __init__(self, first_joint):
        super().__init__()
        self.actions = nn.Parameter(torch.tensor([[first_joint, 0., 0., 0., 0., 50.]]).repeat(2, 1))

    def forward(self, batch):
        return self.actions.unsqueeze(0).expand(len(batch["observation.state"]), -1, -1), (None, None)


class ConstantAct(nn.Module):
    def __init__(self, first_joint=0.):
        super().__init__()
        self.config = SimpleNamespace(
            chunk_size=2, n_action_steps=2, image_features=[],
            output_features={"action": SimpleNamespace(type=FeatureType.ACTION)},
            normalization_mapping={FeatureType.ACTION: NormalizationMode.IDENTITY},
            optimizer_lr=1e-3, optimizer_lr_backbone=1e-4, optimizer_weight_decay=0.,
        )
        self.normalize_inputs = nn.Identity()
        self.model = ConstantActCore(first_joint)


class ConstantCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.value = nn.Parameter(torch.tensor(0.))

    def forward(self, top_images, *_):
        return self.value.expand(len(top_images))


def make_trainer(first_joint=0., **settings):
    cfg = replace(
        ActPpoConfig(critic_image_height=8, critic_image_width=8, critic_hidden_dim=16,
                     minibatch_size=2, ppo_epochs=1, initial_log_std=-4.,
                     actor_learning_rate=1e-3, actor_backbone_learning_rate=1e-4),
        **settings,
    )
    return ActPpoTrainer(ConstantAct(first_joint), ConstantCritic(cfg), config=cfg)


def collect(trainer, rewards=(0., 1.), offset=0.):
    pixels = np.zeros((3, 8, 8), dtype=np.uint8)
    states = np.zeros(6, dtype=np.float32)
    actor_observation = ActPpoActorObservation(pixels, pixels.copy(), states)
    critic_observation = CompressedResidualObservation(pixels, pixels.copy(), states, states.copy())
    rollout = ActPpoRolloutBuffer(trainer.config)
    reference = rollout.add_actor_observation(actor_observation)
    with torch.no_grad():
        means = predict_action_chunk_with_grad(
            trainer.policy, actor_observation_tensors(actor_observation, device="cpu")
        )
        log_std = trainer.exploration.with_offset(offset)
        actions, probabilities, samples = sample_bounded_policy_actions(means, log_std, return_latent=True)
    for index, reward in enumerate(rewards):
        rollout.add(
            critic_observation, actor_observation_index=reference, action_index=index,
            policy_action=actions[0, index].numpy(), policy_action_latent=samples[0, index].numpy(),
            policy_action_mean=means[0, index].detach().numpy(), policy_action_log_std=log_std.detach().numpy(),
            exploration_log_std_offset=offset, log_probability=probabilities[0, index].item(),
            value=0., next_value=0., reward=reward, terminated=index == 1, episode_end=index == 1,
        )
    return rollout


def test_exploration_updates_in_float32_with_independent_lr_even_when_actor_lr_is_tiny():
    torch.manual_seed(3)
    trainer = make_trainer(actor_learning_rate=2e-11, actor_backbone_learning_rate=1e-11,
                           exploration_learning_rate=1e-4)
    initial = trainer.exploration.log_std.detach().clone()
    optimizer_group = next(
        group for group in trainer.optimizer.param_groups
        if any(parameter is trainer.exploration.log_std for parameter in group["params"])
    )
    assert optimizer_group["lr"] == 1e-4

    metrics = trainer.update(collect(trainer))

    assert metrics.completed_ppo_epochs == 1
    assert not torch.equal(initial, trainer.exploration.log_std)
    assert metrics.exploration_parameter_changed_fraction > 0.
    assert metrics.exploration_parameter_relative_delta > 0.
    np.testing.assert_allclose(
        metrics.exploration_log_std_delta,
        (trainer.exploration.log_std.detach() - initial).numpy(), rtol=0., atol=0.,
    )
    assert metrics.pre_clip_policy_kl > 0.


@pytest.mark.parametrize("coefficient", [0., 10.])
def test_structural_bounds_hold_without_an_auxiliary_penalty(coefficient):
    torch.manual_seed(9)
    trainer = make_trainer(first_joint=101., actor_learning_rate=1e-2,
                           mean_bounds_coefficient=coefficient)
    before = trainer.policy.model.actions.detach().clone()
    initial_log_std = trainer.exploration.log_std.detach().clone()
    rollout = collect(trainer, rewards=(0., 0.))
    assert torch.equal(rollout.batch("cpu").advantages, torch.zeros(2))
    means = np.stack(rollout.policy_action_means)
    assert np.all(means[:, 0] < 100.)
    assert np.all(means[:, 0] > 98.)

    metrics = trainer.update(rollout)

    assert metrics.completed_ppo_epochs == 1
    assert metrics.policy_loss == 0.
    assert metrics.mean_bounds_loss == 0.
    assert torch.equal(trainer.policy.model.actions, before)
    assert torch.equal(trainer.policy.model.actions[:, 1:], before[:, 1:])
    assert torch.equal(trainer.exploration.log_std, initial_log_std)


def test_legal_endpoint_is_not_pushed_inward_by_bounds_auxiliary_loss():
    trainer = make_trainer(first_joint=100., mean_bounds_coefficient=10.)
    before = trainer.policy.model.actions.detach().clone()

    metrics = trainer.update(collect(trainer, rewards=(0., 0.)))

    assert metrics.mean_bounds_loss == 0.
    assert torch.equal(trainer.policy.model.actions, before)


def test_inward_advantage_updates_raw_actor_even_after_historical_mean_exceeded_limit():
    trainer = make_trainer(first_joint=109.5, actor_learning_rate=1e-2)
    rollout = collect(trainer, rewards=(1., 0.))
    # Fixed, legal actions scored under the actual collection distribution.
    # Positive advantage for the inward first action must move the network's
    # activation inward through the smooth bound, without an auxiliary penalty.
    for index, first_joint in enumerate((80., 99.)):
        action = torch.tensor([[first_joint, 0., 0., 0., 0., 50.]])
        sample = action.clone() / 100.
        sample[:, -1] = 0.
        mean = torch.from_numpy(rollout.policy_action_means[index]).unsqueeze(0)
        log_prob, _ = evaluate_bounded_policy_actions(
            mean, action, trainer.exploration.value(), policy_action_latents=sample,
        )
        rollout.policy_actions[index] = action[0].numpy()
        rollout.policy_action_latents[index] = sample[0].numpy()
        rollout.log_probabilities[index] = log_prob.item()
    before = trainer.policy.model.actions.detach().clone()

    metrics = trainer.update(rollout)

    assert metrics.completed_ppo_epochs == 1
    assert metrics.mean_bounds_loss == 0.
    assert trainer.policy.model.actions[0, 0] < before[0, 0]
    assert metrics.pre_update_max_abs_log_ratio == 0.


def test_actor_critic_and_exploration_gradients_are_clipped_independently(monkeypatch):
    torch.manual_seed(23)
    trainer = make_trainer(max_gradient_norm=.11, critic_max_gradient_norm=.22,
                           exploration_max_gradient_norm=.33)
    actor_ids = {id(p) for p in trainer.policy.parameters() if p.requires_grad}
    critic_ids = {id(p) for p in trainer.critic.parameters()}
    exploration_ids = {id(p) for p in trainer.exploration.parameters()}
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def observe_independent_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        calls.append(({id(p) for p in parameters}, max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe_independent_clip)
    metrics = trainer.update(collect(trainer))

    assert calls == [(actor_ids, .11), (critic_ids, .22), (exploration_ids, .33)]
    assert metrics.actor_gradient_norm > 0.
    assert metrics.critic_gradient_norm > 0.
    assert metrics.exploration_gradient_norm > 0.


def test_stage_log_std_offset_is_part_of_exact_old_distribution():
    torch.manual_seed(51)
    trainer = make_trainer()
    rollout = collect(trainer, offset=-.75)
    np.testing.assert_array_equal(rollout.policy_action_log_stds, np.full((2, 6), -4.75))

    metrics = trainer.update(rollout)

    assert metrics.pre_update_pre_clip_policy_kl == 0.
    assert metrics.pre_update_max_abs_log_ratio == 0.
    assert metrics.completed_ppo_epochs == 1


def test_gripper_can_explore_and_learn_above_arm_limit_with_exact_replay():
    torch.manual_seed(81)
    trainer = make_trainer(maximum_log_std=-3.5, gripper_maximum_log_std=-1.5)
    offset = np.asarray([0.] * 5 + [1.7], dtype=np.float32)
    rollout = collect(trainer, offset=offset)
    batch = rollout.batch("cpu")
    np.testing.assert_allclose(batch.old_action_log_stds.numpy(), np.asarray([[-4.] * 5 + [-2.3]] * 2), atol=1e-7)
    np.testing.assert_array_equal(batch.exploration_log_std_offsets.numpy(), np.stack([offset, offset]))
    before = trainer.exploration.log_std.detach().clone()
    strict = trainer._policy_ratio_diagnostics(batch, torch.arange(2))
    diagnostic = trainer._diagnostic_policy_ratio_stats(batch, torch.arange(2))
    assert strict.pre_clip_policy_kl == strict.max_abs_log_ratio == 0.
    assert diagnostic["max_abs_log_ratio"] == diagnostic["approximate_kl"] == 0.

    metrics = trainer.update(rollout)

    assert metrics.completed_ppo_epochs == 1
    assert metrics.pre_update_pre_clip_policy_kl == metrics.pre_update_max_abs_log_ratio == 0.
    assert trainer.exploration.log_std[-1] != before[-1]
    assert torch.all(trainer.exploration.with_offset(offset)[:5] <= -3.5)
    assert trainer.exploration.with_offset(offset)[-1] > -3.5
    assert metrics.effective_exploration_log_std_upper_bounds == (-3.5,) * 5 + (-1.5,)


def test_wider_gripper_cap_only_applies_after_stage_offset_without_polluting_home():
    cfg = ActPpoConfig(initial_log_std=-4., minimum_log_std=-6., maximum_log_std=-3.5,
                       gripper_maximum_log_std=-1.5)
    exploration = ActPpoExploration(cfg)
    with torch.no_grad():
        exploration.log_std.fill_(10.)
    base_maximum = torch.full((6,), -3.5)
    effective_maximum = torch.tensor([-3.5] * 5 + [-1.5])
    torch.testing.assert_close(exploration.value(), base_maximum, atol=0, rtol=0)
    torch.testing.assert_close(exploration.with_offset(10.), effective_maximum, atol=0, rtol=0)
    exploration.clamp_()
    torch.testing.assert_close(exploration.log_std, base_maximum, atol=0, rtol=0)
    torch.testing.assert_close(exploration.with_offset(-1.), torch.full((6,), -4.5), atol=0, rtol=0)
    torch.testing.assert_close(exploration.with_offset(torch.tensor([0.] * 5 + [1.7])),
                               torch.tensor([-3.5] * 5 + [-1.8]), atol=1e-6, rtol=0)
    torch.testing.assert_close(exploration.with_offset(-100.), torch.full((6,), -6.), atol=0, rtol=0)


def test_scalar_offsets_and_none_gripper_limit_preserve_previous_shared_behavior():
    cfg = ActPpoConfig(initial_log_std=-4., minimum_log_std=-6., maximum_log_std=-3.5)
    exploration = ActPpoExploration(cfg)
    for offset in (-100., -.75, 0., .5, 100.):
        old = (exploration.value() + offset).clamp(cfg.minimum_log_std, cfg.maximum_log_std)
        torch.testing.assert_close(exploration.with_offset(offset), old, atol=0, rtol=0)
    vector = torch.full((2, 6), -.75)
    torch.testing.assert_close(exploration.with_offset(vector), exploration.with_offset(-.75).expand(2, 6))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, -7.])
def test_invalid_gripper_exploration_upper_bound_is_rejected(invalid):
    with pytest.raises(ValueError, match="gripper_maximum_log_std"):
        ActPpoConfig(initial_log_std=-4., minimum_log_std=-6., maximum_log_std=-3.5,
                     gripper_maximum_log_std=invalid)


def test_resume_rejects_changed_gripper_exploration_upper_bound(tmp_path):
    source = make_trainer(maximum_log_std=-3.5, gripper_maximum_log_std=-1.5)
    checkpoint = source.save_checkpoint(tmp_path / "ppo.pt")
    changed = make_trainer(maximum_log_std=-3.5, gripper_maximum_log_std=-2.)
    before = changed.exploration.log_std.detach().clone()

    with pytest.raises(ValueError, match="config does not match"):
        changed.load_checkpoint(checkpoint)

    assert torch.equal(changed.exploration.log_std, before)
    assert not changed.optimizer.state


@pytest.mark.parametrize("offset", [np.zeros(5), np.zeros((2, 6)), np.asarray([0.] * 5 + [float("nan")])])
def test_invalid_per_transition_offset_is_rejected_atomically(offset):
    trainer = make_trainer()
    rollout = collect(trainer)
    before = rollout.checkpoint()
    pixels = np.zeros((3, 8, 8), dtype=np.uint8)
    critic_observation = CompressedResidualObservation(pixels, pixels.copy(), np.zeros(6), np.zeros(6))

    with pytest.raises(ValueError, match="exploration_log_std_offset"):
        rollout.add(
            critic_observation, actor_observation_index=0, action_index=0, policy_action=np.zeros(6),
            exploration_log_std_offset=offset, log_probability=0., value=0., reward=0., next_value=0.,
            terminated=True, episode_end=True,
        )

    assert rollout.checkpoint() == before
