"""Whole-stage exclusion preserves episode GAE, observations and TE replay."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from so101_capstone.training.act_ppo import (
    ActPpoActorObservation, ActPpoConfig, ActPpoRolloutBuffer, ActPpoTrainer,
    ActPpoValueNetwork,
)
from so101_capstone.training.residual_ppo import CompressedResidualObservation


def configuration():
    return ActPpoConfig(
        critic_image_height=8, critic_image_width=8, critic_hidden_dim=16,
        critic_task_context_dim=2, critic_policy_context_dim=3,
        advantage_group_context_index=1, gamma=.5, gae_lambda=1.,
    )


def actor_observation(marker):
    image = np.full((3, 8, 8), marker, dtype=np.uint8)
    state = np.array([marker, 0., 0., 0., 0., 0.], dtype=np.float32)
    return ActPpoActorObservation(image, image.copy(), state)


def episode(group, rewards, *, marker, temporal=False, terminated=False, enabled=None):
    buffer = ActPpoRolloutBuffer(configuration())
    history = []
    if not temporal:
        reference = buffer.add_actor_observation(actor_observation(marker))
    for index, reward in enumerate(rewards):
        if temporal:
            reference = buffer.add_actor_observation(actor_observation(marker + index))
            history.append(reference)
        image = np.full((3, 8, 8), marker + index, dtype=np.uint8)
        critic = CompressedResidualObservation(image, image.copy(), np.full(6, marker), np.ones(6))
        last = index == len(rewards) - 1
        buffer.add(
            critic, actor_observation_index=reference, action_index=0 if temporal else index,
            temporal_actor_references=tuple(history) if temporal else None,
            policy_action=np.array([marker, 0., 0., 0., 0., 50.]),
            policy_action_latent=np.array([marker / 100, 0., 0., 0., 0., 0.]),
            policy_action_mean=np.array([marker, 0., 0., 0., 0., 50.]),
            policy_action_log_std=np.full(6, -3.), log_probability=-.5,
            critic_task_context=np.array([marker, group]),
            critic_policy_context=np.array([marker, index, group]),
            actor_update_enabled=True if enabled is None else enabled[index],
            exploration_log_std_offset=np.arange(6, dtype=np.float32) / 10.,
            exploration_target=np.full(6, marker),
            exploration_target_joint_mask=np.array([True, False] * 3),
            gripper_closing_only=bool(group), value=float(index + 2), reward=float(reward),
            next_value=float(index + 3), terminated=terminated and last, episode_end=last,
        )
    return buffer


def combined(*buffers):
    result = ActPpoRolloutBuffer(configuration())
    for buffer in buffers:
        result.extend(buffer)
    return result


def test_filter_removes_all_objective_data_and_compacts_only_retained_observations():
    source = combined(episode(0, [100., 200.], marker=3, enabled=[False, True]),
                      episode(1, [1., 2., 3.], marker=10, enabled=[False, True, True]),
                      episode(0, [300.], marker=20))
    source.add_actor_observation(actor_observation(30))  # Unused bootstrap observation.
    original_lengths = source.checkpoint()
    filtered = source.without_stage_groups({0})

    assert len(source) == 6 and source.checkpoint() == original_lengths
    assert len(filtered) == 3
    assert filtered.actor_refs == [0, 0, 0]
    assert len(filtered.actor_observations) == 1
    assert filtered.actor_observations[0] is source.actor_observations[1]
    assert not filtered.actor_observations[0].top_image.flags.writeable
    assert filtered.rewards == [1., 2., 3.]
    assert filtered.actor_update_enabled == [False, True, True]
    assert filtered.episode_end == [False, False, True]
    batch = filtered.batch("cpu")
    assert batch.critic_task_context[:, 1].tolist() == [1., 1., 1.]
    assert batch.joint_positions[:, 0].tolist() == [10., 10., 10.]
    assert batch.policy_actions[:, 0].tolist() == [10., 10., 10.]
    assert batch.exploration_log_std_offsets.shape == (3, 6)
    assert batch.advantages[0].item() == 0.  # Existing value-only semantics preserved.


def test_filter_preserves_each_retained_episode_gae_and_terminal_bootstrap():
    source = combined(episode(1, [1., 2.], marker=3, terminated=False),
                      episode(0, [1e6, -1e6], marker=10),
                      episode(1, [5., 6.], marker=20, terminated=True))
    expected_advantages, expected_returns = source._advantages_and_returns()
    filtered = source.without_stage_groups({0})
    advantages, returns = filtered._advantages_and_returns()
    np.testing.assert_array_equal(advantages, expected_advantages[[0, 1, 4, 5]])
    np.testing.assert_array_equal(returns, expected_returns[[0, 1, 4, 5]])
    assert filtered.episode_end == [False, True, False, True]
    assert filtered.terminated == [False, False, False, True]
    assert filtered.next_values == [3., 4., 3., 4.]
    assert filtered.actor_refs == [0, 0, 1, 1]
    np.testing.assert_array_equal(filtered.batch("cpu").returns.numpy(), returns)


def test_te_references_and_replayed_actor_means_are_preserved_after_compaction():
    from test_act_ppo_temporal_ensemble import TinyTemporalPolicy

    source = combined(episode(0, [100., 200.], marker=3, temporal=True),
                      episode(1, [1., 2., 3.], marker=10, temporal=True),
                      episode(0, [300.], marker=20, temporal=True))
    filtered = source.without_stage_groups({0})
    assert filtered.actor_refs == [0, 1, 2]
    assert filtered.temporal_actor_references == [(0,), (0, 1), (0, 1, 2)]
    assert len(filtered.actor_observations) == 3
    assert source.temporal_actor_references[2:5] == [(2,), (2, 3), (2, 3, 4)]
    policy = TinyTemporalPolicy(chunk_size=3, coeff=.01)
    trainer = ActPpoTrainer(policy, ActPpoValueNetwork(configuration()), config=configuration())
    with torch.no_grad():
        expected = trainer._actor_means(source.batch("cpu"), torch.tensor([2, 3, 4]))
        actual = trainer._actor_means(filtered.batch("cpu"), torch.arange(3))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_filter_copies_mutable_transition_storage_and_survives_source_clear():
    source = combined(episode(0, [1.], marker=3), episode(1, [2., 3.], marker=10))
    filtered = source.without_stage_groups({0})
    for name, values in vars(source).items():
        if name == "actor_observations" or not isinstance(values, list):
            continue
        for target, original in zip(getattr(filtered, name), values[1:]):
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(target, original)
                assert not np.shares_memory(target, original), name
    source.policy_action_means[1].fill(99.)
    source.top_images[1].fill(99)
    source.critic_task_context[1].fill(99.)
    assert filtered.policy_action_means[0][0] == 10.
    assert filtered.top_images[0][0, 0, 0] == 10
    assert filtered.critic_task_context[0].tolist() == [10., 1.]
    source.clear()
    assert len(filtered.batch("cpu").returns) == 2


def test_writable_actor_observations_are_copied_and_frozen():
    source = episode(1, [1.], marker=3)
    source.actor_observations[0] = actor_observation(7)
    filtered = source.without_stage_groups({0})
    source.actor_observations[0].top_image.fill(9)
    assert filtered.actor_observations[0].top_image[0, 0, 0] == 7
    assert not filtered.actor_observations[0].top_image.flags.writeable


@pytest.mark.parametrize("excluded,remaining", [({0}, [3.]), ({1}, [1., 2.]), ({0, 1}, []), (set(), [1., 2., 3.])])
def test_all_stage_selections_and_empty_result(excluded, remaining):
    source = combined(episode(0, [1., 2.], marker=3), episode(1, [3.], marker=10))
    filtered = source.without_stage_groups(excluded)
    assert filtered.rewards == remaining
    if not remaining:
        assert all(not values for values in vars(filtered).values() if isinstance(values, list))


def test_empty_source_produces_valid_empty_buffer_without_unused_observations():
    source = ActPpoRolloutBuffer(configuration())
    source.add_actor_observation(actor_observation(3))
    result = source.without_stage_groups({0, 1})
    assert len(result) == 0 and result.actor_observations == []
    assert len(source.actor_observations) == 1


@pytest.mark.parametrize("groups", [{2}, {-1}, {"S1"}, [False], [0, False], [1.0]])
def test_invalid_exclusions_are_rejected(groups):
    with pytest.raises(ValueError, match="integer 0 or 1"):
        episode(0, [1.], marker=3).without_stage_groups(groups)


def test_stage_filter_requires_explicit_context_group_index():
    buffer = ActPpoRolloutBuffer(replace(configuration(), advantage_group_context_index=None))
    with pytest.raises(ValueError, match="advantage_group_context_index"):
        buffer.without_stage_groups({0})


@pytest.mark.parametrize("corruption,match", [
    ("unfinished", "complete episodes"),
    ("mixed_stage", "multiple stage groups"),
    ("nonbinary", "binary stage context"),
    ("missing_field", "field lengths"),
    ("invalid_ref", "invalid actor reference"),
    ("early_terminal", "must end an episode"),
])
def test_malformed_rollout_is_rejected_without_mutating_source(corruption, match):
    source = episode(1, [1., 2.], marker=3)
    if corruption == "unfinished":
        source.episode_end[-1] = False
    elif corruption == "mixed_stage":
        source.critic_task_context[1][1] = 0.
    elif corruption == "nonbinary":
        source.critic_task_context[0][1] = .5
    elif corruption == "missing_field":
        source.policy_action_means.pop()
    elif corruption == "invalid_ref":
        source.actor_refs[0] = 999
    elif corruption == "early_terminal":
        source.terminated[0] = True
    original = source.checkpoint()
    with pytest.raises(ValueError, match=match):
        source.without_stage_groups({0})
    assert source.checkpoint() == original


def test_temporal_history_cannot_retain_an_excluded_episode_or_future_observation():
    source = combined(episode(0, [1.], marker=3, temporal=True),
                      episode(1, [2., 3.], marker=10, temporal=True))
    source.temporal_actor_references[1] = (0, 1)
    with pytest.raises(ValueError, match="crosses an episode boundary"):
        source.without_stage_groups({0})


def test_filtered_buffer_still_supports_extend_checkpoint_and_rollback():
    filtered = combined(episode(0, [1.], marker=3),
                        episode(1, [2., 3.], marker=10)).without_stage_groups({0})
    snapshot = filtered.checkpoint()
    assert filtered.extend(episode(1, [4.], marker=20)) == 1
    assert filtered.rollback(snapshot) == 1
    assert filtered.rewards == [2., 3.]
    assert filtered.actor_refs == [0, 0]
