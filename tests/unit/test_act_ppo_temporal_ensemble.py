"""Temporal PPO replay tests with tiny CPU actors, no weights or simulators."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from so101_capstone.training.act_ppo import (
    ActPpoActorObservation, ActPpoConfig, ActPpoRolloutBuffer, ActPpoTemporalHistory,
    ActPpoTrainer, ActPpoValueNetwork, actor_observation_tensors,
    predict_action_chunk_with_grad, sample_bounded_policy_actions,
)
from so101_capstone.training.residual_ppo import CompressedResidualObservation


class TableActor(nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.predictions = nn.Parameter(torch.randn(32, chunk_size, 6) * 2.)
        with torch.no_grad():
            self.predictions[..., -1] += 50.  # Interior physical gripper means.

    def forward(self, batch):
        ids = batch["observation.state"][:, 0].long()
        return self.predictions[ids], (None, None)


class TinyTemporalPolicy(nn.Module):
    def __init__(self, chunk_size=3, coeff=.01):
        super().__init__()
        self.config = SimpleNamespace(
            chunk_size=chunk_size, n_action_steps=1, temporal_ensemble_coeff=coeff,
            image_features=[], output_features={"action": SimpleNamespace(type=FeatureType.ACTION)},
            normalization_mapping={FeatureType.ACTION: NormalizationMode.IDENTITY},
            optimizer_lr=1e-4, optimizer_lr_backbone=1e-4, optimizer_weight_decay=0.,
        )
        self.normalize_inputs = nn.Identity()
        self.model = TableActor(chunk_size)


def config():
    return ActPpoConfig(critic_image_height=8, critic_image_width=8, critic_hidden_dim=16,
                        minibatch_size=2, ppo_epochs=1, actor_learning_rate=1e-4)


def observations(index):
    pixels = np.full((3, 8, 8), index, dtype=np.uint8)
    state = np.asarray([index, 0, 0, 0, 0, 0], dtype=np.float32)
    return (
        ActPpoActorObservation(pixels, pixels.copy(), state),
        CompressedResidualObservation(pixels, pixels.copy(), state, np.zeros(6, dtype=np.float32)),
    )


def collect(trainer, count=7):
    policy = trainer.policy
    history = ActPpoTemporalHistory(policy.config.temporal_ensemble_coeff, policy.config.chunk_size)
    rollout = ActPpoRolloutBuffer(trainer.config)
    sampled_means = []
    for index in range(count):
        observation, critic_observation = observations(index)
        reference = rollout.add_actor_observation(observation)
        with torch.no_grad():
            chunk = predict_action_chunk_with_grad(policy, actor_observation_tensors(observation, device="cpu"))
            mean = history.push(reference, chunk)
            sampled_means.append(mean[0, 0].clone())
            action, log_probability, latent = sample_bounded_policy_actions(
                mean, trainer.exploration.value(), return_latent=True,
            )
        rollout.add(
            critic_observation, actor_observation_index=reference, action_index=0,
            temporal_actor_references=history.actor_references,
            policy_action=action[0, 0].numpy(), policy_action_latent=latent[0, 0].numpy(),
            policy_action_mean=mean[0, 0].numpy(),
            policy_action_log_std=trainer.exploration.value().detach().numpy(),
            log_probability=log_probability[0, 0].item(), value=0., reward=float(index), next_value=0.,
            terminated=index == count - 1, episode_end=index == count - 1,
        )
    return rollout, torch.stack(sampled_means)


def trainer(coeff=.01, chunk_size=3):
    torch.manual_seed(29)
    cfg = config()
    return ActPpoTrainer(TinyTemporalPolicy(chunk_size, coeff), ActPpoValueNetwork(cfg), config=cfg)


@pytest.mark.parametrize("chunk_size,coeff", [(1, .01), (3, .01), (15, .01), (3, 0.), (3, -.01)])
def test_rollout_helper_matches_lerobot_and_keeps_episode_isolation(chunk_size, coeff):
    history = ActPpoTemporalHistory(coeff, chunk_size)
    other_world = ActPpoTemporalHistory(coeff, chunk_size)
    reference = ACTTemporalEnsembler(coeff, chunk_size)
    torch.manual_seed(4)
    for index in range(chunk_size + 5):
        chunk = torch.rand(1, chunk_size, 6)
        actual = history.push(index, chunk)
        torch.testing.assert_close(actual[:, 0], reference.update(chunk), atol=0, rtol=0)
        assert history.actor_references == tuple(range(max(0, index - chunk_size + 1), index + 1))
    assert other_world.actor_references == ()
    snapshot = history.actor_references
    history.push(None, torch.zeros(1, chunk_size, 6))
    assert snapshot[-1] is not None and history.actor_references[-1] is None
    history.reset()
    assert history.actor_references == ()
    chunk = torch.ones(1, chunk_size, 6)
    torch.testing.assert_close(history.push(0, chunk), chunk[:, :1], atol=0, rtol=0)


@pytest.mark.parametrize("chunk_size,coeff", [(1, .01), (3, .01), (15, .01), (3, -.01)])
def test_recomputed_means_and_both_kl_diagnostics_match_rollout_exactly(chunk_size, coeff):
    agent = trainer(coeff, chunk_size)
    rollout, expected = collect(agent, count=chunk_size + 3)
    batch = rollout.batch("cpu")
    indices = torch.arange(len(rollout))
    with torch.no_grad():
        actual = agent._actor_means(batch, indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Also exercises the activation-checkpoint branch and shuffled histories.
    torch.testing.assert_close(agent._actor_means(batch, indices.flip(0)), expected.flip(0), atol=0, rtol=0)
    strict = agent._policy_ratio_diagnostics(batch, indices)
    bounded = agent._diagnostic_policy_ratio_stats(batch, indices)
    assert strict.max_abs_log_ratio == strict.approximate_kl == 0.
    assert strict.pre_clip_policy_kl == strict.pre_clip_policy_kl_max == 0.
    assert bounded["max_abs_log_ratio"] == bounded["approximate_kl"] == 0.


def test_history_gradients_reach_every_contributing_old_frame_and_action():
    agent = trainer()
    rollout, _ = collect(agent, count=4)
    means = agent._actor_means(rollout.batch("cpu"), torch.tensor([3]))
    means.sum().backward()
    gradients = agent.policy.model.predictions.grad
    weights = torch.exp(-.01 * torch.arange(3))
    weights /= weights.sum()
    for order, (frame, action) in enumerate(((1, 2), (2, 1), (3, 0))):
        torch.testing.assert_close(gradients[frame, action], torch.full((6,), weights[order]))
    assert torch.count_nonzero(gradients).item() == 18
    assert gradients[0].abs().sum() == 0  # expired history receives no gradient.


def test_changing_only_historical_prediction_changes_new_ratio_and_kl():
    agent = trainer()
    rollout, expected = collect(agent, count=3)
    batch = rollout.batch("cpu")
    with torch.no_grad():
        agent.policy.model.predictions[0, 2] += 10.
        actual = agent._actor_means(batch, torch.tensor([2]))
    assert not torch.equal(actual[0], expected[2])
    assert agent._policy_ratio_diagnostics(batch, torch.tensor([2])).max_abs_log_ratio > 0
    assert agent._policy_ratio_diagnostics(batch, torch.tensor([2])).pre_clip_policy_kl > 0
    assert agent._diagnostic_policy_ratio_stats(batch, torch.tensor([2]))["approximate_kl"] > 0


def test_temporal_update_runs_gradient_checkpoint_branch_and_updates_actor():
    agent = trainer()
    rollout, _ = collect(agent, count=5)
    before = agent.policy.model.predictions.detach().clone()
    result = agent.update(rollout)
    assert result.pre_update_max_abs_log_ratio == 0.
    assert not torch.equal(before, agent.policy.model.predictions)


def test_temporal_refs_survive_extend_batch_checkpoint_rollback_clear():
    agent = trainer()
    episode, _ = collect(agent, count=4)
    combined = ActPpoRolloutBuffer(agent.config)
    combined.extend(episode)
    checkpoint = combined.checkpoint()
    combined.extend(episode)
    assert combined.temporal_actor_references[4:] == [(4,), (4, 5), (4, 5, 6), (5, 6, 7)]
    assert combined.batch("cpu").temporal_actor_references == tuple(combined.temporal_actor_references)
    assert combined.rollback(checkpoint) == 4
    assert combined.temporal_actor_references == episode.temporal_actor_references
    combined.clear()
    assert combined.temporal_actor_references == []


@pytest.mark.parametrize("history,action_index", [((None,), 0), ((1, 0), 0), ((0,), 0), ((0, 1), 1), ((), 0)])
def test_invalid_history_rejected_before_any_transition_is_appended(history, action_index):
    rollout = ActPpoRolloutBuffer(config())
    for index in range(2):
        rollout.add_actor_observation(observations(index)[0])
    before = rollout.checkpoint()
    with pytest.raises(ValueError, match="Temporal actor references"):
        rollout.add(
            observations(1)[1], actor_observation_index=1, action_index=action_index,
            temporal_actor_references=history, policy_action=np.zeros(6), log_probability=0.,
            value=0., reward=0., next_value=0., terminated=True, episode_end=True,
        )
    assert rollout.checkpoint() == before


def test_missing_history_and_non_temporal_policy_mismatch_are_rejected():
    agent = trainer()
    rollout, _ = collect(agent, count=2)
    batch = rollout.batch("cpu")
    with pytest.raises(ValueError, match="missing actor history"):
        agent._actor_means(replace(batch, temporal_actor_references=()), torch.tensor([1]))
    agent.policy.config.temporal_ensemble_coeff = None
    with pytest.raises(ValueError, match="non-temporal"):
        agent._actor_means(batch, torch.tensor([1]))


def test_temporal_checkpoint_signature_is_versioned_without_changing_non_temporal_signature():
    agent = trainer()
    assert agent.execution_signature["actor_execution"]["temporal_history_schema_version"] == 1
    assert "temporal_history_schema_version" not in trainer(coeff=None).execution_signature["actor_execution"]


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), True, -1000.])
def test_invalid_or_overflowing_temporal_weights_are_rejected(coefficient):
    with pytest.raises(ValueError):
        ActPpoTemporalHistory(coefficient, 3)


def assert_training_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_training_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected, strict=True):
            assert_training_state_equal(item, reference)
    else:
        assert actual == expected


def test_temporal_checkpoint_restores_updated_weights_optimizer_and_counters(tmp_path):
    source = trainer()
    rollout, _ = collect(source, count=5)
    source.update(rollout)
    assert source.optimizer.state  # Exercise populated Adam moments, not only defaults.
    checkpoint = source.save_checkpoint(tmp_path / "temporal.pt", extra={"test": "temporal"})
    restored = trainer()
    assert not torch.equal(restored.policy.model.predictions, source.policy.model.predictions)

    payload = restored.load_checkpoint(checkpoint, map_location="cpu")

    assert payload["extra"] == {"test": "temporal"}
    assert restored.update_index == source.update_index == 1
    assert restored.total_environment_steps == source.total_environment_steps == 5
    assert restored.execution_signature == source.execution_signature
    assert_training_state_equal(restored._epoch_rollback_snapshot(), source._epoch_rollback_snapshot())


def test_temporal_checkpoint_rejects_changed_coefficient_before_mutating_state(tmp_path):
    source = trainer(coeff=.01)
    with torch.no_grad():
        source.policy.model.predictions.add_(42.)
    checkpoint = source.save_checkpoint(tmp_path / "temporal.pt")
    restored = trainer(coeff=.02)
    restored.update_index = 7
    restored.total_environment_steps = 123
    before = restored._epoch_rollback_snapshot()

    with pytest.raises(ValueError, match="execution signature does not match"):
        restored.load_checkpoint(checkpoint, map_location="cpu")

    assert_training_state_equal(restored._epoch_rollback_snapshot(), before)
    assert (restored.update_index, restored.total_environment_steps) == (7, 123)


def test_temporal_checkpoint_rejects_schema_one_without_execution_signature(tmp_path):
    source = trainer()
    with torch.no_grad():
        source.policy.model.predictions.add_(42.)
    payload = source.checkpoint_payload()
    payload["schema_version"] = 1
    payload.pop("execution_signature")
    payload.pop("execution_signature_sha256")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(payload, checkpoint)
    restored = trainer()
    before = restored._epoch_rollback_snapshot()

    with pytest.raises(ValueError, match="[Ss]chema|[Ii]ncompatible|[Ll]egacy"):
        restored.load_checkpoint(checkpoint, map_location="cpu")

    assert_training_state_equal(restored._epoch_rollback_snapshot(), before)
    assert (restored.update_index, restored.total_environment_steps) == (0, 0)
