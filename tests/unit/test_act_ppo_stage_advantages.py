"""Stage advantage credit stays separate while GAE/value targets stay intact."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from so101_capstone.training.act_ppo import (
    ActPpoConfig,
    ActPpoRolloutBuffer,
    ActPpoTrainer,
    ActPpoValueNetwork,
    _execution_signature,
    actor_observation_tensors,
    predict_action_chunk_with_grad,
    sample_bounded_policy_actions,
)
from test_act_ppo import TinyActPolicy, actor_observation, critic_observation


def config(**overrides):
    values = dict(
        critic_image_height=16, critic_image_width=16, critic_hidden_dim=16,
        critic_task_context_dim=2, advantage_group_context_index=1,
        ppo_epochs=1, minibatch_size=4, gamma=.997, gae_lambda=.95,
    )
    return ActPpoConfig(**{**values, **overrides})


def buffer_for(groups, rewards, *, cfg=None, enabled=None, values=None,
               next_values=None, terminated=None, episode_end=None):
    cfg = config() if cfg is None else cfg
    buffer = ActPpoRolloutBuffer(cfg)
    reference = buffer.add_actor_observation(actor_observation())
    count = len(groups)
    enabled = [True] * count if enabled is None else enabled
    values = [0.] * count if values is None else values
    next_values = [0.] * count if next_values is None else next_values
    terminated = [True] * count if terminated is None else terminated
    episode_end = [True] * count if episode_end is None else episode_end
    for index, group in enumerate(groups):
        buffer.add(
            critic_observation(), actor_observation_index=reference,
            action_index=0, policy_action=np.zeros(6), log_probability=0.,
            critic_task_context=np.array([0., group], dtype=np.float32),
            actor_update_enabled=enabled[index], value=values[index],
            reward=float(rewards[index]), next_value=next_values[index],
            terminated=terminated[index], episode_end=episode_end[index],
        )
    return buffer


def test_update57_scales_no_longer_make_all_s1_advantages_negative():
    # Observed update 57 had 900 S0 / 1500 S1 transitions. These independent
    # terminal samples reproduce its stage means/scales, without loading logs.
    def samples(count, mean, std):
        x = np.linspace(-1., 1., count)
        return mean + std * (x - x.mean()) / x.std()

    raw = np.r_[samples(900, 1.7644808, .59315205),
                samples(1500, .016113702, .10275434)]
    groups = [0] * 900 + [1] * 1500
    legacy = buffer_for(groups, raw, cfg=config(advantage_group_context_index=None))
    grouped = buffer_for(groups, raw)
    old_batch, batch = legacy.batch("cpu"), grouped.batch("cpu")
    assert bool((old_batch.advantages[900:] < 0).all())
    for selected in (batch.advantages[:900], batch.advantages[900:]):
        assert float(selected.mean()) == pytest.approx(0., abs=1e-7)
        assert float(selected.std(unbiased=False)) == pytest.approx(1., abs=1e-7)
        assert bool((selected > 0).any()) and bool((selected < 0).any())
    torch.testing.assert_close(batch.returns, old_batch.returns, rtol=0, atol=0)
    diagnostics = grouped.advantage_normalization_diagnostics()
    assert diagnostics == batch.advantage_normalization
    for group, count in (("0", 900), ("1", 1500)):
        row = diagnostics["groups"][group]
        assert row["actor_transition_count"] == count
        assert row["normalized_positive_fraction"] == .5
        assert row["normalized_negative_fraction"] == .5
    json.dumps(diagnostics, allow_nan=False)


def test_value_only_outliers_cannot_change_either_actor_group():
    baseline = buffer_for([0, 0, 1, 1], [1., 3., 10., 20.]).batch("cpu")
    buffer = buffer_for([0, 0, 1, 1, 0, 1], [1., 3., 10., 20., 1e9, -1e9],
                        enabled=[True] * 4 + [False, False])
    batch = buffer.batch("cpu")
    torch.testing.assert_close(batch.advantages[:4], baseline.advantages, rtol=0, atol=0)
    assert batch.advantages[4:].tolist() == [0., 0.]
    assert batch.returns[4:].tolist() == [1e9, -1e9]
    for row in batch.advantage_normalization["groups"].values():
        assert row["actor_transition_count"] == 2
        assert row["value_only_transition_count"] == 1


@pytest.mark.parametrize("group", [0, 1])
def test_single_group_available_is_normalized_without_other_group(group):
    batch = buffer_for([group] * 3, [2., 3., 4.]).batch("cpu")
    assert batch.advantages.tolist() == pytest.approx([-np.sqrt(1.5), 0., np.sqrt(1.5)])
    missing = batch.advantage_normalization["groups"][str(1 - group)]
    assert missing["actor_transition_count"] == 0
    assert missing["raw_mean"] is None
    assert missing["normalized_std"] is None
    json.dumps(batch.advantage_normalization, allow_nan=False)


@pytest.mark.parametrize("rewards", [[5.], [5., 5.], [0., 1e-10]])
def test_singleton_constant_and_tiny_variance_groups_are_zero(rewards):
    batch = buffer_for([1] * len(rewards), rewards).batch("cpu")
    assert batch.advantages.tolist() == [0.] * len(rewards)
    assert batch.advantage_normalization["groups"]["1"]["degenerate"] is True
    assert bool(torch.isfinite(batch.advantages).all())


def test_value_only_batch_has_no_actor_statistics_and_preserves_returns():
    batch = buffer_for([0, 1], [5., -7.], enabled=[False, False]).batch("cpu")
    assert batch.advantages.tolist() == [0., 0.]
    assert batch.returns.tolist() == [5., -7.]
    assert batch.advantage_normalization["actor_transition_count"] == 0
    json.dumps(batch.advantage_normalization, allow_nan=False)


def test_grouping_does_not_change_truncation_bootstrap_or_gae_reset():
    kwargs = dict(values=[.5, .7, 1.], next_values=[.7, 3., 99.],
                  terminated=[False, False, True], episode_end=[False, True, True])
    cfg = config(gamma=.9, gae_lambda=.8)
    grouped = buffer_for([0, 0, 1], [1., 2., -1.], cfg=cfg, **kwargs)
    legacy = buffer_for([0, 0, 1], [1., 2., -1.],
                        cfg=replace(cfg, advantage_group_context_index=None), **kwargs)
    raw, returns = grouped._advantages_and_returns()
    np.testing.assert_allclose(raw, [4.01, 4., -2.], atol=1e-6)
    np.testing.assert_allclose(returns, [4.51, 4.7, -1.], atol=1e-6)
    torch.testing.assert_close(grouped.batch("cpu").returns, legacy.batch("cpu").returns,
                               rtol=0, atol=0)


def test_extend_and_rollback_preserve_groups_and_normalize_whole_batch_once():
    first = buffer_for([0, 1], [1., 10.])
    second = buffer_for([1, 0], [20., 3.])
    token = first.checkpoint()
    first.extend(second)
    expected = buffer_for([0, 1, 1, 0], [1., 10., 20., 3.]).batch("cpu")
    actual = first.batch("cpu")
    assert actual.actor_observation_indices.tolist() == [0, 0, 1, 1]
    torch.testing.assert_close(actual.advantages, expected.advantages, rtol=0, atol=0)
    assert first.rollback(token) == 2
    assert first.batch("cpu").advantages.tolist() == [0., 0.]


@pytest.mark.parametrize("index", [True, -1, 2, .5, "1"])
def test_config_rejects_invalid_context_index(index):
    with pytest.raises(ValueError, match="advantage_group_context_index"):
        config(advantage_group_context_index=index)


@pytest.mark.parametrize("flag", [-1., .5, 2., float("nan"), float("inf")])
def test_non_binary_context_rejected_before_recording(flag):
    with pytest.raises(ValueError, match="binary|finite"):
        buffer_for([flag], [1.])


def test_non_binary_context_rejected_even_when_actor_disabled_or_mutated():
    with pytest.raises(ValueError, match="binary"):
        buffer_for([.5], [1.], enabled=[False])
    buffer = buffer_for([0, 1], [1., 2.])
    buffer.critic_task_context[1][1] = .5
    with pytest.raises(ValueError, match="binary"):
        buffer.batch("cpu")


def test_global_mode_preserves_legacy_float32_arithmetic_and_value_only_inclusion():
    buffer = buffer_for([0, .5, 1], [-10., .123456, 1000.],
                        cfg=config(advantage_group_context_index=None),
                        enabled=[True, True, False])
    raw, _ = buffer._advantages_and_returns()
    expected = (raw - float(raw.mean())) / max(float(raw.std()), 1e-8)
    np.testing.assert_array_equal(buffer.batch("cpu").advantages.numpy(), expected)
    assert buffer.advantage_normalization_diagnostics() == {}


def test_only_opted_in_execution_signatures_change():
    from so101_capstone.training.act_ppo import configure_bounded_act_policy
    policy = TinyActPolicy()
    configure_bounded_act_policy(policy)
    global_config = config(advantage_group_context_index=None)
    signature = _execution_signature(policy, global_config, {})
    assert "advantage_normalization" not in signature
    grouped = _execution_signature(policy, config(), {})
    contract = grouped.pop("advantage_normalization")
    assert grouped == signature
    assert contract["context_index"] == 1
    assert contract["kind"] == "binary_context_actor_only_v1"
    other = _execution_signature(policy, config(advantage_group_context_index=0), {})
    assert other["advantage_normalization"] != contract


def test_old_checkpoint_missing_optional_field_still_resumes_global_only(tmp_path):
    global_config = config(advantage_group_context_index=None)
    trainer = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(global_config), config=global_config)
    payload = trainer.checkpoint_payload()
    payload["config"].pop("advantage_group_context_index")
    path = tmp_path / "legacy_global.pt"
    torch.save(payload, path)
    restored = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(global_config), config=global_config)
    restored.load_checkpoint(path)
    assert restored.execution_signature == trainer.execution_signature
    grouped_config = config()
    incompatible = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(grouped_config), config=grouped_config)
    with pytest.raises(ValueError, match="execution signature"):
        incompatible.load_checkpoint(path)


def test_update_exports_the_actor_group_statistics():
    torch.manual_seed(91)
    cfg = config()
    policy = TinyActPolicy()
    trainer = ActPpoTrainer(policy, ActPpoValueNetwork(cfg), config=cfg)
    buffer = ActPpoRolloutBuffer(cfg)
    observation = actor_observation()
    reference = buffer.add_actor_observation(observation)
    with torch.no_grad():
        means = predict_action_chunk_with_grad(policy, actor_observation_tensors(observation, device="cpu"))
        logs = trainer.exploration.value()
        actions, probabilities, latents = sample_bounded_policy_actions(means, logs, return_latent=True)
    for index, (group, reward) in enumerate(((0, 1.), (0, 3.), (1, 10.), (1, 20.))):
        action_index = index % 3
        buffer.add(
            critic_observation(), actor_observation_index=reference, action_index=action_index,
            policy_action=actions[0, action_index].numpy(),
            policy_action_latent=latents[0, action_index].numpy(),
            policy_action_mean=means[0, action_index].numpy(), policy_action_log_std=logs.numpy(),
            log_probability=float(probabilities[0, action_index]), critic_task_context=np.array([0., group]),
            value=0., reward=reward, next_value=0., terminated=True, episode_end=True,
        )
    expected = buffer.advantage_normalization_diagnostics()
    metrics = trainer.update(buffer)
    assert metrics.advantage_normalization == expected
    assert metrics.to_dict()["advantage_normalization"] == expected
