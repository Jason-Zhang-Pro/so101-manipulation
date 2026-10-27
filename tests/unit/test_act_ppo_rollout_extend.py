import numpy as np
import pytest

from so101_capstone.training.act_ppo import (
    ActPpoActorObservation,
    ActPpoConfig,
    ActPpoRolloutBuffer,
)
from so101_capstone.training.residual_ppo import CompressedResidualObservation


def config():
    return ActPpoConfig(
        critic_image_height=16,
        critic_image_width=16,
        gamma=1.0,
        gae_lambda=1.0,
    )


def actor_observation(value):
    image = np.full((3, 8, 8), value, dtype=np.uint8)
    return ActPpoActorObservation(
        top_image=image,
        gripper_image=image.copy(),
        policy_state=np.full(6, float(value), dtype=np.float32),
    )


def critic_observation(value):
    image = np.full((3, 16, 16), value, dtype=np.uint8)
    return CompressedResidualObservation(
        top_image=image,
        gripper_image=image.copy(),
        joint_positions=np.full(6, float(value), dtype=np.float32),
        joint_velocities=np.zeros(6, dtype=np.float32),
    )


def episode_buffer(rewards, *, actor_value):
    rollout = ActPpoRolloutBuffer(config())
    reference = rollout.add_actor_observation(actor_observation(actor_value))
    for index, reward in enumerate(rewards):
        last = index == len(rewards) - 1
        rollout.add(
            critic_observation(actor_value + index),
            actor_observation_index=reference,
            action_index=index,
            policy_action=np.zeros(6),
            log_probability=-0.5,
            actor_update_enabled=True,
            exploration_log_std_offset=-0.25,
            gripper_closing_only=bool(actor_value),
            value=0.0,
            reward=float(reward),
            next_value=0.0,
            terminated=False,
            episode_end=last,
        )
    return rollout


def test_extend_remaps_actor_references_and_preserves_all_flags():
    combined = ActPpoRolloutBuffer(config())
    first = episode_buffer([1.0, 2.0], actor_value=0)
    second = episode_buffer([3.0, 4.0], actor_value=1)

    assert combined.extend(first) == 2
    assert combined.extend(second) == 2

    assert len(combined) == 4
    assert len(combined.actor_observations) == 2
    assert combined.actor_refs == [0, 0, 1, 1]
    assert combined.episode_end == [False, True, False, True]
    np.testing.assert_array_equal(combined.exploration_log_std_offsets, np.full((4, 6), -.25))
    assert combined.gripper_closing_only == [False, False, True, True]


def test_extend_keeps_gae_trajectories_separate_at_episode_boundaries():
    combined = ActPpoRolloutBuffer(config())
    combined.extend(episode_buffer([1.0, 2.0], actor_value=0))
    combined.extend(episode_buffer([10.0, 20.0], actor_value=1))

    advantages, returns = combined._advantages_and_returns()

    np.testing.assert_allclose(advantages, [3.0, 2.0, 30.0, 20.0])
    np.testing.assert_allclose(returns, advantages)


def test_extend_rejects_different_rollout_configuration():
    combined = ActPpoRolloutBuffer(config())
    other = ActPpoRolloutBuffer(
        ActPpoConfig(critic_image_height=8, critic_image_width=8)
    )

    with pytest.raises(ValueError, match="configurations differ"):
        combined.extend(other)


def test_critic_context_is_copied_through_extend_batch_and_episode_rollback():
    from dataclasses import replace
    cfg = replace(config(), critic_task_context_dim=3, critic_policy_context_dim=4)
    episode = ActPpoRolloutBuffer(cfg)
    ref = episode.add_actor_observation(actor_observation(0))
    task = np.asarray([1., 0., .25], dtype=np.float32)
    plan = np.asarray([0., 1., -.5, .5], dtype=np.float32)
    expected_task, expected_plan = task.copy(), plan.copy()
    episode.add(
        critic_observation(0), actor_observation_index=ref, action_index=0,
        policy_action=np.zeros(6), log_probability=0., value=0., reward=1., next_value=2.,
        terminated=False, episode_end=True,
        critic_task_context=task, critic_policy_context=plan,
    )
    task.fill(9.)
    plan.fill(9.)
    combined = ActPpoRolloutBuffer(cfg)
    combined.extend(episode)
    checkpoint = combined.checkpoint()
    combined.extend(episode)
    assert combined.rollback(checkpoint) == 1
    episode.critic_task_context[0].fill(7.)
    episode.critic_policy_context[0].fill(7.)
    batch = combined.batch("cpu")
    np.testing.assert_array_equal(batch.critic_task_context.numpy(), [expected_task])
    np.testing.assert_array_equal(batch.critic_policy_context.numpy(), [expected_plan])
    assert len(combined.critic_task_context) == len(combined.critic_policy_context) == len(combined) == 1


def test_context_validation_rejects_missing_or_nonfinite_without_appending_transition():
    from dataclasses import replace
    cfg = replace(config(), critic_task_context_dim=2, critic_policy_context_dim=2)
    rollout = ActPpoRolloutBuffer(cfg)
    reference = rollout.add_actor_observation(actor_observation(0))
    args = dict(
        actor_observation_index=reference, action_index=0, policy_action=np.zeros(6),
        log_probability=0., value=0., reward=1., next_value=0., terminated=True, episode_end=True,
    )
    checkpoint = rollout.checkpoint()
    with pytest.raises(ValueError, match="critic_task_context is required"):
        rollout.add(critic_observation(0), **args)
    with pytest.raises(ValueError, match="critic_policy_context must be a finite vector"):
        rollout.add(critic_observation(0), **args, critic_task_context=np.zeros(2), critic_policy_context=np.asarray([np.nan, 1.]))
    assert rollout.checkpoint() == checkpoint


def test_contextual_gae_bootstraps_truncation_but_not_success_or_next_episode():
    from dataclasses import replace
    cfg = replace(config(), gamma=.5, gae_lambda=1., critic_task_context_dim=1, critic_policy_context_dim=1)
    combined = ActPpoRolloutBuffer(cfg)
    for terminated in (False, True):
        episode = ActPpoRolloutBuffer(cfg)
        ref = episode.add_actor_observation(actor_observation(0))
        for index, (reward, value, next_value) in enumerate(((1., 10., 20.), (2., 20., 7.))):
            episode.add(
                critic_observation(0), actor_observation_index=ref, action_index=index,
                policy_action=np.zeros(6), log_probability=0., value=value, reward=reward,
                next_value=next_value, terminated=terminated and index == 1, episode_end=index == 1,
                critic_task_context=np.asarray([float(terminated)]), critic_policy_context=np.asarray([float(index)]),
            )
        combined.extend(episode)
    _, returns = combined._advantages_and_returns()
    # A hypothetical continuation V=7 matters at truncation; terminal success
    # zeroes it. Neither trajectory receives rewards from its neighbour.
    np.testing.assert_allclose(returns, [3.75, 5.5, 2., 2.])


def test_original_gaussian_statistics_survive_copy_extend_and_rollback():
    episode = ActPpoRolloutBuffer(config())
    reference = episode.add_actor_observation(actor_observation(0))
    mean = np.asarray([101., -102., 0., 0., 0., -1.], dtype=np.float32)
    sample = np.asarray([1.04, -1.1, .2, -.1, 0., -1.1], dtype=np.float64)
    logs = np.full(6, -4.25, dtype=np.float32)
    offset = np.asarray([-.5] * 5 + [1.7], dtype=np.float32)
    expected_mean, expected_sample, expected_logs = mean.copy(), sample.copy(), logs.copy()
    expected_offset = offset.copy()
    episode.add(
        critic_observation(0), actor_observation_index=reference, action_index=0,
        policy_action=np.asarray([100., -100., 20., -10., 0., 0.]),
        policy_action_latent=sample, policy_action_mean=mean, policy_action_log_std=logs,
        exploration_log_std_offset=offset,
        log_probability=1., value=0., reward=1., next_value=0., terminated=True, episode_end=True,
    )
    mean.fill(0.)
    sample.fill(0.)
    logs.fill(0.)
    offset.fill(0.)
    combined = ActPpoRolloutBuffer(config())
    combined.extend(episode)
    snapshot = combined.checkpoint()
    combined.extend(episode)
    assert combined.rollback(snapshot) == 1
    episode.policy_action_means[0].fill(0.)
    episode.policy_action_latents[0].fill(0.)
    episode.policy_action_log_stds[0].fill(0.)
    episode.exploration_log_std_offsets[0].fill(0.)
    batch = combined.batch("cpu")

    np.testing.assert_array_equal(batch.old_action_means.numpy(), [expected_mean])
    np.testing.assert_array_equal(batch.policy_action_latents.numpy(), [expected_sample])
    np.testing.assert_array_equal(batch.old_action_log_stds.numpy(), [expected_logs])
    np.testing.assert_array_equal(batch.exploration_log_std_offsets.numpy(), [expected_offset])
    combined.clear()
    assert combined.policy_action_means == combined.policy_action_log_stds == combined.policy_action_latents == []
