from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.constants import ACTION

from so101_capstone.training.act_ppo import (
    ActPpoActorObservation,
    ActPpoConfig,
    ActPpoRolloutBuffer,
    ActPpoTrainer,
    ActPpoValueNetwork,
    actor_observation_tensors,
    build_critic_policy_context,
    critic_policy_context_size,
    evaluate_bounded_policy_actions,
    policy_action_saturation_diagnostics,
    policy_action_bounds,
    predict_action_chunk_with_grad,
    sample_bounded_policy_actions,
)
from so101_capstone.training.residual_ppo import (
    CompressedResidualObservation,
    model_state_sha256,
)
from so101_capstone.environments.policy_observation import PolicyObservation
from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
from so101_capstone.policies.so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS,
)


class PassThrough(nn.Module):
    def forward(self, value):
        return value


class NoGradUnnormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.buffer_action = nn.ParameterDict(
            {
                "mean": nn.Parameter(
                    torch.zeros(6),
                    requires_grad=False,
                ),
                "std": nn.Parameter(
                    torch.full((6,), 2.0),
                    requires_grad=False,
                ),
            }
        )

    @torch.no_grad()
    def forward(self, batch):
        return {
            ACTION: (
                batch[ACTION] * self.buffer_action["std"]
                + self.buffer_action["mean"]
            )
        }


class TinyActCore(nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.linear = nn.Linear(6, chunk_size * 6)

    def forward(self, batch):
        actions = self.linear(
            batch["observation.state"]
        ).reshape(-1, self.chunk_size, 6)
        return actions, (None, None)


class TinyActPolicy(nn.Module):
    def __init__(self, chunk_size=3):
        super().__init__()
        self.config = SimpleNamespace(
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            image_features=[],
            output_features={
                ACTION: SimpleNamespace(
                    type=FeatureType.ACTION,
                )
            },
            normalization_mapping={
                FeatureType.ACTION: (
                    NormalizationMode.MEAN_STD
                )
            },
            optimizer_lr=1e-3,
            optimizer_lr_backbone=1e-3,
            optimizer_weight_decay=0.0,
        )
        self.normalize_inputs = PassThrough()
        self.unnormalize_outputs = NoGradUnnormalize()
        self.model = TinyActCore(chunk_size)

    def reset(self):
        return None


def config():
    return ActPpoConfig(
        critic_image_height=16,
        critic_image_width=16,
        critic_hidden_dim=32,
        ppo_epochs=1,
        minibatch_size=2,
        actor_learning_rate=1e-3,
        actor_backbone_learning_rate=1e-3,
    )


def actor_observation(value=0):
    image = np.full((3, 8, 8), value, dtype=np.uint8)
    return ActPpoActorObservation(
        top_image=image,
        gripper_image=image.copy(),
        policy_state=np.full(
            6,
            value / 10.0,
            dtype=np.float32,
        ),
    )


def critic_observation(value=0):
    image = np.full((3, 16, 16), value, dtype=np.uint8)
    return CompressedResidualObservation(
        top_image=image,
        gripper_image=image.copy(),
        joint_positions=np.full(
            6,
            value / 10.0,
            dtype=np.float32,
        ),
        joint_velocities=np.zeros(6, dtype=np.float32),
    )


def test_differentiable_action_unnormalization_preserves_actor_gradient():
    policy = TinyActPolicy()
    # A non-zero state is required to test the Linear weight gradient;
    # an all-zero input correctly produces zero dL/dW even when the
    # computation graph is intact.
    observation = actor_observation(value=1)

    actions = predict_action_chunk_with_grad(
        policy,
        actor_observation_tensors(
            observation,
            device="cpu",
        ),
    )
    actions.sum().backward()

    weight_gradient = policy.model.linear.weight.grad
    bias_gradient = policy.model.linear.bias.grad
    assert weight_gradient is not None
    assert bias_gradient is not None
    assert torch.any(weight_gradient != 0.0)
    assert torch.any(bias_gradient != 0.0)


def test_bounded_action_log_probability_round_trip():
    lower, upper = policy_action_bounds(device="cpu")
    means = ((lower + upper) / 2.0).reshape(1, 1, 6)
    means = means.expand(2, 3, 6).clone()
    log_std = torch.full((6,), -2.3)

    actions, original = sample_bounded_policy_actions(
        means,
        log_std,
        deterministic=True,
    )
    evaluated, entropy = evaluate_bounded_policy_actions(
        means,
        actions,
        log_std,
    )

    assert actions.shape == (2, 3, 6)
    assert original.shape == (2, 3)
    assert torch.all(actions >= lower)
    assert torch.all(actions <= upper)
    torch.testing.assert_close(
        original,
        evaluated,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(torch.isfinite(entropy))


def test_rollout_stores_one_chunk_observation_for_multiple_steps():
    rollout = ActPpoRolloutBuffer(config())
    reference = rollout.add_actor_observation(
        actor_observation()
    )

    for index in range(3):
        rollout.add(
            critic_observation(index),
            actor_observation_index=reference,
            action_index=index,
            policy_action=np.zeros(6),
            log_probability=0.0,
            actor_update_enabled=True,
            value=0.0,
            reward=1.0,
            next_value=0.0,
            terminated=index == 2,
            episode_end=index == 2,
        )

    batch = rollout.batch("cpu")
    assert batch.actor_top_images.shape[0] == 1
    assert batch.actor_observation_indices.tolist() == [0, 0, 0]
    assert batch.action_indices.tolist() == [0, 1, 2]


def test_ppo_update_changes_act_parameters():
    torch.manual_seed(3)
    ppo_config = config()
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(
        policy,
        critic,
        config=ppo_config,
    )
    rollout = ActPpoRolloutBuffer(ppo_config)
    observation = actor_observation()
    reference = rollout.add_actor_observation(observation)

    with torch.no_grad():
        means = predict_action_chunk_with_grad(
            policy,
            actor_observation_tensors(
                observation,
                device="cpu",
            ),
        )
        actions, log_probabilities, latents = (
            sample_bounded_policy_actions(
                means,
                trainer.exploration.value(),
                return_latent=True,
            )
        )

    for index in range(4):
        action_index = index % 3
        compact = critic_observation(index)
        with torch.no_grad():
            value = critic(
                torch.from_numpy(
                    compact.top_image.copy()
                ).unsqueeze(0),
                torch.from_numpy(
                    compact.gripper_image.copy()
                ).unsqueeze(0),
                torch.from_numpy(
                    compact.joint_positions.copy()
                ).unsqueeze(0),
                torch.from_numpy(
                    compact.joint_velocities.copy()
                ).unsqueeze(0),
            )
        rollout.add(
            compact,
            actor_observation_index=reference,
            action_index=action_index,
            policy_action=actions[
                0,
                action_index,
            ].numpy(),
            policy_action_latent=latents[0, action_index].numpy(),
            policy_action_mean=means[0, action_index].numpy(),
            policy_action_log_std=trainer.exploration.value().detach().numpy(),
            log_probability=float(
                log_probabilities[
                    0,
                    action_index,
                ].item()
            ),
            actor_update_enabled=True,
            value=float(value.item()),
            reward=float(index + 1),
            next_value=0.0,
            terminated=index == 3,
            episode_end=index == 3,
        )

    before = model_state_sha256(policy)
    metrics = trainer.update(rollout)
    after = model_state_sha256(policy)

    assert before != after
    assert metrics.transition_count == 4
    assert metrics.minibatch_count == 2
    assert trainer.update_index == 1


def test_deterministic_transitions_do_not_update_actor_or_exploration():
    torch.manual_seed(7)
    ppo_config = config()
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(
        policy,
        critic,
        config=ppo_config,
    )
    rollout = ActPpoRolloutBuffer(ppo_config)
    observation = actor_observation()
    reference = rollout.add_actor_observation(observation)

    with torch.no_grad():
        means = predict_action_chunk_with_grad(
            policy,
            actor_observation_tensors(
                observation,
                device="cpu",
            ),
        )
        actions, log_probabilities = (
            sample_bounded_policy_actions(
                means,
                trainer.exploration.value(),
                deterministic=True,
            )
        )

    for index in range(2):
        action_index = index
        rollout.add(
            critic_observation(index),
            actor_observation_index=reference,
            action_index=action_index,
            policy_action=actions[0, action_index].numpy(),
            log_probability=float(
                log_probabilities[0, action_index].item()
            ),
            actor_update_enabled=False,
            value=0.0,
            reward=1.0,
            next_value=0.0,
            terminated=index == 1,
            episode_end=index == 1,
        )

    actor_before = model_state_sha256(policy)
    exploration_before = (
        trainer.exploration.log_std.detach().clone()
    )
    trainer.update(rollout)
    actor_after = model_state_sha256(policy)

    assert actor_before == actor_after
    torch.testing.assert_close(
        trainer.exploration.log_std.detach(),
        exploration_before,
    )


def test_rollout_stores_exploration_log_std_offsets():
    rollout = ActPpoRolloutBuffer(config())
    reference = rollout.add_actor_observation(actor_observation())

    rollout.add(
        critic_observation(),
        actor_observation_index=reference,
        action_index=0,
        policy_action=np.zeros(6),
        log_probability=0.0,
        actor_update_enabled=True,
        exploration_log_std_offset=-1.25,
        value=0.0,
        reward=1.0,
        next_value=0.0,
        terminated=True,
        episode_end=True,
    )

    batch = rollout.batch("cpu")

    assert batch.actor_update_enabled.tolist() == [True]
    np.testing.assert_array_equal(batch.exploration_log_std_offsets.numpy(), np.full((1, 6), -1.25))


def test_rollout_rollback_discards_only_failed_episode_data():
    rollout = ActPpoRolloutBuffer(config())

    first_reference = rollout.add_actor_observation(
        actor_observation(value=1)
    )
    rollout.add(
        critic_observation(value=1),
        actor_observation_index=first_reference,
        action_index=0,
        policy_action=np.zeros(6),
        log_probability=0.0,
        actor_update_enabled=True,
        exploration_log_std_offset=0.0,
        value=0.0,
        reward=1.0,
        next_value=0.0,
        terminated=True,
        episode_end=True,
    )

    checkpoint = rollout.checkpoint()

    second_reference = rollout.add_actor_observation(
        actor_observation(value=2)
    )
    rollout.add(
        critic_observation(value=2),
        actor_observation_index=second_reference,
        action_index=0,
        policy_action=np.ones(6),
        log_probability=-1.0,
        actor_update_enabled=True,
        exploration_log_std_offset=-1.0,
        value=2.0,
        reward=2.0,
        next_value=3.0,
        terminated=False,
        episode_end=False,
    )

    assert rollout.rollback(checkpoint) == 1
    assert len(rollout) == 1

    batch = rollout.batch("cpu")
    assert batch.actor_top_images.shape[0] == 1
    assert batch.actor_observation_indices.tolist() == [0]
    assert rollout.rewards == pytest.approx([1.0])
    np.testing.assert_array_equal(batch.exploration_log_std_offsets.numpy(), np.zeros((1, 6)))


def test_rollout_rollback_rejects_foreign_checkpoint():
    rollout = ActPpoRolloutBuffer(config())
    with pytest.raises(ValueError, match="fields"):
        rollout.rollback({"rewards": 0})


class NoInferencePolicy:
    def select_action(self, batch):
        raise AssertionError("Policy inference must not be called.")

    def reset(self):
        return None


def test_adapter_maps_sampled_policy_action_without_inference():
    gazebo = np.asarray(
        [
            (item.gazebo_lower_rad + item.gazebo_upper_rad) / 2.0
            for item in SIM_JOINT_CALIBRATIONS
        ]
    )
    policy = np.asarray(
        [
            (item.policy_lower + item.policy_upper) / 2.0
            for item in SIM_JOINT_CALIBRATIONS
        ]
    )
    observation = PolicyObservation(
        joint_positions=gazebo,
        joint_velocities=np.zeros(6),
        top_image=np.zeros((480, 640, 3), dtype=np.uint8),
        gripper_image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    adapter = ActPolicyAdapter(
        NoInferencePolicy(),
        device="cpu",
        max_joint_delta_rad=1_000_000.0,
    )

    result = adapter.map_policy_positions(
        observation,
        policy,
    )

    assert result.action.controller_command().shape == (6,)
    np.testing.assert_allclose(
        result.clipped_policy_positions,
        policy,
    )


def test_rollout_stores_optional_exploration_target():
    rollout = ActPpoRolloutBuffer(config())
    reference = rollout.add_actor_observation(actor_observation())
    target = np.arange(6, dtype=np.float32)

    rollout.add(
        critic_observation(),
        actor_observation_index=reference,
        action_index=0,
        policy_action=np.zeros(6),
        log_probability=0.0,
        actor_update_enabled=True,
        exploration_log_std_offset=0.0,
        exploration_target=target,
        value=0.0,
        reward=1.0,
        next_value=0.0,
        terminated=True,
        episode_end=True,
    )

    batch = rollout.batch('cpu')
    assert batch.exploration_target_enabled.tolist() == [True]
    torch.testing.assert_close(
        batch.exploration_targets[0],
        torch.from_numpy(target),
    )
    assert batch.exploration_target_joint_masks[0].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_sampler_can_disable_arm_exploration_with_two_sided_gripper():
    torch.manual_seed(321)
    lower, upper = policy_action_bounds(device="cpu")
    mean = ((lower + upper) / 2.0).reshape(1, 1, 6)
    means = mean.expand(256, 8, 6).clone()
    log_std = torch.full((6,), -2.0)

    actions, sampled_log_prob = sample_bounded_policy_actions(
        means,
        log_std,
        gripper_closing_only=True,
        arm_exploration_enabled=False,
    )

    # Arm joints 1-5 receive no exploration perturbation.
    torch.testing.assert_close(
        actions[..., :5],
        means[..., :5],
        rtol=0.0,
        atol=1e-6,
    )

    # The legacy closing flag cannot reintroduce one-sided exploration.
    assert torch.any(actions[..., 5] > means[..., 5] + 1e-5)
    assert torch.any(actions[..., 5] < means[..., 5] - 1e-5)

    evaluated_log_prob, _ = evaluate_bounded_policy_actions(
        means,
        actions,
        log_std,
        gripper_closing_only=True,
    )
    torch.testing.assert_close(
        sampled_log_prob,
        evaluated_log_prob,
        atol=1e-4,
        rtol=1e-4,
    )


def test_legacy_closing_flag_log_probability_round_trip():
    torch.manual_seed(456)
    lower, upper = policy_action_bounds(device="cpu")
    means = ((lower + upper) / 2.0).reshape(1, 1, 6)
    means = means.expand(4, 3, 6).clone()
    log_std = torch.full((6,), -2.4)

    actions, sampled_log_prob = sample_bounded_policy_actions(
        means,
        log_std,
        gripper_closing_only=True,
    )
    evaluated_log_prob, entropy = evaluate_bounded_policy_actions(
        means,
        actions,
        log_std,
        gripper_closing_only=True,
    )

    torch.testing.assert_close(
        sampled_log_prob,
        evaluated_log_prob,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(torch.isfinite(entropy))


def test_rollout_stores_state3_gripper_closing_only_flag():
    rollout = ActPpoRolloutBuffer(config())
    reference = rollout.add_actor_observation(actor_observation())

    rollout.add(
        critic_observation(),
        actor_observation_index=reference,
        action_index=0,
        policy_action=np.zeros(6),
        log_probability=0.0,
        actor_update_enabled=True,
        exploration_log_std_offset=0.0,
        gripper_closing_only=True,
        value=0.0,
        reward=1.0,
        next_value=0.0,
        terminated=True,
        episode_end=True,
    )

    batch = rollout.batch("cpu")
    assert batch.gripper_closing_only.tolist() == [True]

def test_sampled_log_probability_matches_stored_original_sample():
    torch.manual_seed(29)
    means = torch.tensor(
        [[[99.0, -99.0, 95.0, -95.0, 90.0, 50.0]]],
        dtype=torch.float32,
    )
    log_std = torch.full((6,), -2.3)

    actions, sampled_log_probability, samples = sample_bounded_policy_actions(
        means,
        log_std,
        return_latent=True,
    )

    stored_actions = torch.from_numpy(
        actions.detach().cpu().numpy().astype(np.float32, copy=True)
    )
    evaluated_log_probability, _ = evaluate_bounded_policy_actions(
        means,
        stored_actions,
        log_std,
        policy_action_latents=samples.detach().double(),
    )

    torch.testing.assert_close(
        sampled_log_probability,
        evaluated_log_probability,
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("target_kl", [0.0, -0.1, float("inf"), float("nan")])
def test_target_kl_must_be_positive_and_finite(target_kl):
    with pytest.raises(ValueError, match="target_kl"):
        ActPpoConfig(target_kl=target_kl)


def _guard_test_rollout(ppo_config, policy, trainer, critic):
    rollout = ActPpoRolloutBuffer(ppo_config)
    observation = actor_observation()
    reference = rollout.add_actor_observation(observation)

    with torch.no_grad():
        means = predict_action_chunk_with_grad(
            policy,
            actor_observation_tensors(
                observation,
                device="cpu",
            ),
        )
        actions, log_probabilities, latents = sample_bounded_policy_actions(
            means,
            trainer.exploration.value(),
            return_latent=True,
        )

    for index in range(4):
        action_index = index % policy.config.chunk_size
        compact = critic_observation(index)
        with torch.no_grad():
            value = critic(
                torch.from_numpy(compact.top_image.copy()).unsqueeze(0),
                torch.from_numpy(compact.gripper_image.copy()).unsqueeze(0),
                torch.from_numpy(compact.joint_positions.copy()).unsqueeze(0),
                torch.from_numpy(compact.joint_velocities.copy()).unsqueeze(0),
            )
        rollout.add(
            compact,
            actor_observation_index=reference,
            action_index=action_index,
            policy_action=actions[0, action_index].numpy(),
            policy_action_latent=latents[0, action_index].numpy(),
            policy_action_mean=means[0, action_index].numpy(),
            policy_action_log_std=trainer.exploration.value().detach().numpy(),
            log_probability=float(
                log_probabilities[0, action_index].item()
            ),
            actor_update_enabled=True,
            value=float(value.item()),
            reward=float(index + 1),
            next_value=0.0,
            terminated=index == 3,
            episode_end=index == 3,
        )
    return rollout


def test_pre_update_log_probability_mismatch_stops_before_optimizer_step():
    torch.manual_seed(31)
    ppo_config = config()
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config)
    rollout = _guard_test_rollout(
        ppo_config,
        policy,
        trainer,
        critic,
    )
    rollout.log_probabilities[0] += 1.0

    before = model_state_sha256(policy)
    with pytest.raises(RuntimeError, match="pre-update"):
        trainer.update(rollout)
    after = model_state_sha256(policy)

    assert before == after
    assert trainer.update_index == 0
    assert trainer.total_environment_steps == 0


def test_target_kl_rolls_back_rejected_epoch_and_stops_remaining_epochs(
    capsys,
):
    torch.manual_seed(37)
    ppo_config = ActPpoConfig(
        critic_image_height=16,
        critic_image_width=16,
        critic_hidden_dim=32,
        ppo_epochs=3,
        minibatch_size=2,
        target_kl=1e-12,
        actor_learning_rate=1e-3,
        actor_backbone_learning_rate=1e-3,
    )
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config)
    rollout = _guard_test_rollout(
        ppo_config,
        policy,
        trainer,
        critic,
    )

    before_policy = model_state_sha256(policy)
    before_critic = {
        name: value.detach().clone()
        for name, value in critic.state_dict().items()
    }
    before_exploration = {
        name: value.detach().clone()
        for name, value in trainer.exploration.state_dict().items()
    }
    assert not trainer.optimizer.state

    metrics = trainer.update(rollout)
    output = capsys.readouterr().out

    assert metrics.pre_update_clip_fraction == pytest.approx(0.0)
    assert metrics.pre_update_max_abs_log_ratio < 0.05

    # After a rejected epoch, continuing at the same learning rate would repeat
    # the oversized update on the same data. Restore once and finish this batch.
    assert output.count("[PPO ROLLBACK]") == 1
    assert "[PPO CONTINUE]" not in output
    assert "[PPO EPOCH] 1/3 rejected" in output
    assert "[PPO EPOCH] 2/3" not in output
    assert "[PPO EPOCH] 3/3" not in output

    # completed_ppo_epochs now counts accepted epochs, not attempt index.
    assert metrics.completed_ppo_epochs == 0
    assert metrics.configured_ppo_epochs == 3
    assert metrics.minibatch_count == 0
    assert metrics.early_stopped is True
    assert metrics.policy_loss == pytest.approx(0.0)
    assert metrics.value_loss == pytest.approx(0.0)
    assert metrics.final_policy_approximate_kl == pytest.approx(
        metrics.pre_update_approximate_kl
    )

    # Every rejected epoch restores all trainable/optimizer state.
    assert model_state_sha256(policy) == before_policy
    for name, value in critic.state_dict().items():
        torch.testing.assert_close(value, before_critic[name])
    for name, value in trainer.exploration.state_dict().items():
        torch.testing.assert_close(value, before_exploration[name])
    assert not trainer.optimizer.state

    # The rollout is still consumed as one attempted PPO update.
    assert trainer.update_index == 1
    assert trainer.total_environment_steps == len(rollout)

def test_ppo_actor_recompute_helper_always_forwards_batch_one(monkeypatch):
    import so101_capstone.training.act_ppo as act_ppo_module

    torch.manual_seed(41)
    ppo_config = config()
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config)

    one = actor_observation_tensors(
        actor_observation(),
        device="cpu",
    )
    actor_batch = {
        key: torch.cat([value, value, value], dim=0)
        for key, value in one.items()
    }

    observed_batch_sizes = []

    def fake_predict(policy_arg, observation_batch):
        assert policy_arg is policy
        batch_size = int(
            observation_batch["observation.state"].shape[0]
        )
        observed_batch_sizes.append(batch_size)
        return torch.zeros(
            (batch_size, policy.config.chunk_size, 6),
            dtype=torch.float32,
        )

    monkeypatch.setattr(
        act_ppo_module,
        "predict_action_chunk_with_grad",
        fake_predict,
    )

    chunks = trainer._predict_action_chunks_singleton(actor_batch)

    assert chunks.shape == (3, policy.config.chunk_size, 6)
    assert observed_batch_sizes == [1, 1, 1]


@pytest.mark.parametrize("legacy_mode", ["home", "gripper_target", "closing", "arm_target_closing"])
def test_legacy_directional_controls_cannot_restrict_gaussian_exploration(legacy_mode):
    lower, upper = policy_action_bounds(device="cpu")
    means = ((lower + upper) / 2).reshape(1, 1, 6).expand(256, 8, 6).clone()
    target = means[0:1, 0:1].clone()
    target[..., :5] += torch.tensor([10., -10., 7., -7., 5.])
    # The gripper is exactly at its target in HOME mode: it must still explore.
    kwargs = {}
    if legacy_mode in ("home", "gripper_target", "arm_target_closing"):
        kwargs["exploration_target"] = target
        if legacy_mode == "gripper_target":
            target[..., 5] += 10
            kwargs["exploration_target_joint_mask"] = [False] * 5 + [True]
        elif legacy_mode == "arm_target_closing":
            kwargs["exploration_target_joint_mask"] = [True] * 5 + [False]
    if legacy_mode in ("closing", "arm_target_closing"):
        kwargs["gripper_closing_only"] = True
    log_std = torch.full((6,), -2.)
    torch.manual_seed(321)
    actions, log_prob = sample_bounded_policy_actions(means, log_std, **kwargs)
    torch.manual_seed(321)
    baseline, baseline_log_prob = sample_bounded_policy_actions(means, log_std)
    torch.testing.assert_close(actions, baseline, rtol=0, atol=0)
    torch.testing.assert_close(log_prob, baseline_log_prob, rtol=0, atol=0)
    for joint in range(6):
        assert torch.any(actions[..., joint] > means[..., joint])
        assert torch.any(actions[..., joint] < means[..., joint])
    # Moving the means across old actions must use the same full Gaussian.
    changed_means = means + 4.
    new_prob, entropy = evaluate_bounded_policy_actions(changed_means, actions, log_std, **kwargs)
    plain_prob, plain_entropy = evaluate_bounded_policy_actions(changed_means, actions, log_std)
    torch.testing.assert_close(new_prob, plain_prob, rtol=0, atol=0)
    torch.testing.assert_close(entropy, plain_entropy, rtol=0, atol=0)
    assert torch.isfinite(new_prob).all()


def test_ppo_keeps_normalization_fixed_while_training_actor():
    torch.manual_seed(3)
    policy = TinyActPolicy()
    policy.normalize_inputs = NoGradUnnormalize()
    # Use a passthrough forward while exposing realistic ParameterDict stats.
    policy.normalize_inputs.forward = lambda batch: batch
    policy.normalize_targets = NoGradUnnormalize()
    ppo_config = config()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config)
    stats = [p for name, p in policy.named_parameters() if not name.startswith("model.")]
    optimizer_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all(not p.requires_grad and id(p) not in optimizer_ids for p in stats)
    before_stats = [p.detach().clone() for p in stats]
    before_weights = policy.model.linear.bias.detach().clone()
    rollout = _guard_test_rollout(ppo_config, policy, trainer, critic)
    trainer.update(rollout)
    for old, current in zip(before_stats, stats, strict=True):
        torch.testing.assert_close(old, current, atol=0, rtol=0)
        assert current.grad is None
    assert not torch.equal(before_weights, policy.model.linear.bias)


def _assert_nested_state_equal(first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_state_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            _assert_nested_state_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("failure", ["gradient", "parameter", "optimizer", "candidate_action", "candidate_ratio", "train_logratio", "train_ratio", "unexpected"])
def test_failed_candidate_restores_every_training_state(monkeypatch, failure):
    from so101_capstone.training.act_ppo import ActPpoNumericalError
    torch.manual_seed(3)
    ppo_config = config()
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config)
    # Populate Adam moments before the failing update so restoration tests
    # pre-existing optimizer values, not only an empty state dictionary.
    trainer.update(_guard_test_rollout(ppo_config, policy, trainer, critic))
    rollout = _guard_test_rollout(ppo_config, policy, trainer, critic)
    before = trainer._epoch_rollback_snapshot()
    if failure == "gradient":
        policy.model.linear.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    elif failure in ("parameter", "optimizer", "unexpected"):
        original_step = trainer.optimizer.step
        def bad_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            if failure == "parameter":
                with torch.no_grad():
                    policy.model.linear.weight.fill_(float("nan"))
            elif failure == "optimizer":
                next(iter(trainer.optimizer.state.values()))["exp_avg"].fill_(float("inf"))
            else:
                raise RuntimeError("unrelated optimizer failure")
            return result
        monkeypatch.setattr(trainer.optimizer, "step", bad_step)
    elif failure in ("train_logratio", "train_ratio"):
        import so101_capstone.training.act_ppo as act_ppo_module
        original_evaluate = act_ppo_module.evaluate_bounded_policy_actions
        def nonfinite_training_ratio(*args, **kwargs):
            log_probability, entropy = original_evaluate(*args, **kwargs)
            if torch.is_grad_enabled():
                log_probability = log_probability + (float("nan") if failure == "train_logratio" else 1000.)
            return log_probability, entropy
        monkeypatch.setattr(act_ppo_module, "evaluate_bounded_policy_actions", nonfinite_training_ratio)
    else:
        original_diagnostics = trainer._policy_ratio_diagnostics
        calls = 0
        def bad_candidate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                if failure == "candidate_action":
                    with torch.no_grad():
                        policy.model.linear.bias.fill_(float("nan"))
                    return original_diagnostics(*args, **kwargs)
                raise ActPpoNumericalError("PPO ratio diagnostics produced non-finite ratio.")
            return original_diagnostics(*args, **kwargs)
        monkeypatch.setattr(trainer, "_policy_ratio_diagnostics", bad_candidate)
    if failure == "unexpected":
        with pytest.raises(RuntimeError, match="unrelated optimizer failure"):
            trainer.update(rollout)
    else:
        result = trainer.update(rollout)
        assert result.completed_ppo_epochs == 0
        assert result.minibatch_count == 0
    _assert_nested_state_equal(before, trainer._epoch_rollback_snapshot())
    assert all(p.grad is None for p in trainer._parameters())


def test_default_target_kl_uses_conservative_pre_clip_gaussian_budget():
    assert ActPpoConfig().target_kl == .02


def test_checkpoint_restores_cuda_rng_from_cpu_and_allows_only_kl_override(monkeypatch):
    import copy
    import so101_capstone.training.act_ppo as act_ppo_module
    ppo_config = config()
    trainer = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(ppo_config), config=ppo_config)
    payload = copy.deepcopy(trainer.checkpoint_payload())
    payload["config"]["target_kl"] = .1
    cpu_state = torch.get_rng_state()

    class MappedCudaState:
        def __init__(self):
            self.copied_to_cpu = False
        def detach(self):
            return self
        def cpu(self):
            self.copied_to_cpu = True
            return cpu_state

    states = [MappedCudaState(), MappedCudaState()]
    payload["cuda_rng_state_all"] = states
    monkeypatch.setattr(act_ppo_module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    received = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: received.extend(values))
    trainer.load_checkpoint("unused-checkpoint.pt", map_location="cuda")
    assert all(state.copied_to_cpu for state in states)
    assert len(received) == 2
    assert all(state.device.type == "cpu" for state in received)
    payload["config"]["critic_hidden_dim"] += 1
    with pytest.raises(ValueError, match="config does not match"):
        trainer.load_checkpoint("unused-checkpoint.pt")


def test_schema_three_rejects_unnamed_legacy_optimizer_before_loading(monkeypatch):
    import copy
    import so101_capstone.training.act_ppo as act_ppo_module
    ppo_config = config()
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config)
    # The old trainer's first group included all non-backbone policy params,
    # including the two fixed action statistics.
    old_optimizer = torch.optim.AdamW([
        {"params": list(policy.parameters())},
        {"params": list(critic.parameters())},
        {"params": list(trainer.exploration.parameters())},
    ])
    for index, group in enumerate(old_optimizer.param_groups):
        for offset, parameter in enumerate(group["params"]):
            parameter.grad = torch.full_like(parameter, float(index + offset + 1))
    old_optimizer.step()
    old_optimizer.zero_grad(set_to_none=True)
    payload = copy.deepcopy(trainer.checkpoint_payload())
    payload.pop("optimizer_parameter_names")
    payload["optimizer_state_dict"] = old_optimizer.state_dict()
    monkeypatch.setattr(act_ppo_module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    before = trainer._epoch_rollback_snapshot()
    with pytest.raises(ValueError, match="parameter names"):
        trainer.load_checkpoint("legacy-checkpoint.pt", map_location="cpu")
    _assert_nested_state_equal(before, trainer._epoch_rollback_snapshot())


def test_named_checkpoint_rejects_optimizer_parameter_reordering(monkeypatch):
    import copy
    import so101_capstone.training.act_ppo as act_ppo_module
    ppo_config = config()
    trainer = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(ppo_config), config=ppo_config)
    payload = copy.deepcopy(trainer.checkpoint_payload())
    payload["optimizer_parameter_names"][0].reverse()
    monkeypatch.setattr(act_ppo_module.torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match="parameter names"):
        trainer.load_checkpoint("reordered-checkpoint.pt")


def _latent_gaussian(means, log_std):
    """Independent reference distribution in the sampler's latent coordinates."""
    lower, upper = policy_action_bounds(device=means.device, dtype=means.dtype)
    centre, scale = (lower + upper) / 2, (upper - lower) / 2
    latent_mean = (means - centre) / scale
    std = log_std.to(device=means.device, dtype=torch.float64).exp().expand_as(means)
    return torch.distributions.Normal(latent_mean.double(), std), scale.double()


def test_boundary_sample_density_includes_truncation_normalizer():
    torch.manual_seed(922)
    lower, upper = policy_action_bounds(device="cpu")
    means = lower.reshape(1, 1, 6).expand(1024, 1, 6)
    log_std = torch.full((6,), -2.3)
    actions, log_probability, latent = sample_bounded_policy_actions(
        means, log_std, return_latent=True,
    )
    gaussian, _ = _latent_gaussian(means, log_std)
    normalizer = gaussian.cdf(torch.ones_like(latent)) - gaussian.cdf(-torch.ones_like(latent))
    reference = (gaussian.log_prob(latent.double()) - normalizer.log()).sum(-1)
    torch.testing.assert_close(log_probability, reference, rtol=1e-12, atol=1e-12)
    # At a lower limit exploration is inward, without folding half the mass
    # onto that limit or retaining out-of-range training samples.
    assert torch.all(latent.double() >= gaussian.mean)
    assert torch.count_nonzero(latent == -1.) == 0
    assert torch.all(actions >= lower) and torch.all(actions <= upper)
    # Sampling strength and actual commands do not change when requesting z.
    torch.manual_seed(922)
    legacy_actions, _ = sample_bounded_policy_actions(means, log_std)
    torch.testing.assert_close(actions, legacy_actions, rtol=0, atol=0)


def test_boundary_policy_ratio_matches_normalized_truncated_density():
    lower, _ = policy_action_bounds(device="cpu")
    means = lower.reshape(1, 1, 6).expand(2, 1, 6).clone()
    torch.manual_seed(123)
    log_std = torch.zeros(6)
    actions, old_probability, latent = sample_bounded_policy_actions(
        means, log_std, return_latent=True,
    )
    assert not torch.equal(actions[0], actions[1])
    assert old_probability[0].item() != old_probability[1].item()
    moved_means = means + .01
    new_probability, _ = evaluate_bounded_policy_actions(
        moved_means, actions, log_std, policy_action_latents=latent,
    )
    old_gaussian, _ = _latent_gaussian(means, log_std)
    new_gaussian, _ = _latent_gaussian(moved_means, log_std)
    old_z = old_gaussian.cdf(torch.ones_like(latent)) - old_gaussian.cdf(-torch.ones_like(latent))
    new_z = new_gaussian.cdf(torch.ones_like(latent)) - new_gaussian.cdf(-torch.ones_like(latent))
    expected_log_ratio = (new_gaussian.log_prob(latent.double()) - old_gaussian.log_prob(latent.double())
                          + old_z.log() - new_z.log()).sum(-1)
    torch.testing.assert_close(new_probability - old_probability, expected_log_ratio, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(new_probability).all()


def test_rollout_latents_survive_extend_and_failed_episode_rollback():
    ppo_config = config()
    segment = ActPpoRolloutBuffer(ppo_config)
    ref = segment.add_actor_observation(actor_observation())
    latent = np.arange(6, dtype=np.float64) + .1234567890123
    segment.add(
        critic_observation(), actor_observation_index=ref, action_index=0,
        policy_action=np.zeros(6), policy_action_latent=latent,
        log_probability=.1234567890123, value=0., reward=1., next_value=0.,
        terminated=True, episode_end=True,
    )
    rollout = ActPpoRolloutBuffer(ppo_config)
    rollout.extend(segment)
    checkpoint = rollout.checkpoint()
    rollout.extend(segment)
    rollout.rollback(checkpoint)
    batch = rollout.batch("cpu")
    torch.testing.assert_close(batch.policy_action_latents[0], torch.from_numpy(latent), rtol=0, atol=0)
    assert batch.old_log_probabilities.dtype == torch.float64
    assert batch.old_log_probabilities.item() == .1234567890123
    assert len(rollout.policy_action_latents) == 1


def test_ppo_precheck_uses_stored_bounded_samples_near_gripper_limit():
    torch.manual_seed(13)
    ppo_config = config()
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(ppo_config)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config)
    rollout = ActPpoRolloutBuffer(ppo_config)
    observation = actor_observation()
    ref = rollout.add_actor_observation(observation)
    with torch.no_grad():
        means = predict_action_chunk_with_grad(policy, actor_observation_tensors(observation, device="cpu"))
        actions, old_probability, latents = sample_bounded_policy_actions(means, trainer.exploration.value(), return_latent=True)
    for index in range(3):
        rollout.add(
            critic_observation(index), actor_observation_index=ref, action_index=index,
            policy_action=actions[0, index].numpy(), policy_action_latent=latents[0, index].numpy(),
            policy_action_mean=means[0, index].numpy(),
            policy_action_log_std=trainer.exploration.value().detach().numpy(),
            log_probability=old_probability[0, index].item(), value=0., reward=float(index), next_value=0.,
            terminated=index == 2, episode_end=index == 2,
        )
    diagnostics = trainer._policy_ratio_diagnostics(rollout.batch("cpu"), torch.arange(3))
    assert diagnostics.max_abs_log_ratio < 1e-6
    assert diagnostics.approximate_kl == 0.


def test_cached_plan_context_contains_only_unexecuted_means_and_detaches_actor():
    means = torch.tensor([
        [-80., -60., -40., -20., 0., 10.],
        [20., 40., 60., 80., 0., 75.],
        [900., 900., 900., 900., 900., 900.],
    ], requires_grad=True)
    context = build_critic_policy_context(means, action_index=1, n_action_steps=2)
    assert context.shape == (critic_policy_context_size(2),)
    assert context.dtype == np.float32
    assert not context.flags.writeable
    np.testing.assert_allclose(context[:2], [.5, .5])
    np.testing.assert_allclose(context[2:8], [.2, .4, .6, .8, 0., .5])
    np.testing.assert_array_equal(context[8:14], np.zeros(6))
    np.testing.assert_array_equal(context[14:], [1., 0.])
    # Executed means and unused ACT prediction tail cannot change the cache.
    with torch.no_grad():
        means[0].fill_(-900.)
        means[2].fill_(-900.)
    np.testing.assert_array_equal(
        context, build_critic_policy_context(means, action_index=1, n_action_steps=2),
    )
    assert means.grad is None
    exhausted = build_critic_policy_context(means, action_index=2, n_action_steps=2)
    np.testing.assert_array_equal(exhausted[:2], [1., 0.])
    np.testing.assert_array_equal(exhausted[2:], np.zeros(len(exhausted) - 2))


@pytest.mark.parametrize("kwargs", [
    {"action_index": -1, "n_action_steps": 2},
    {"action_index": 3, "n_action_steps": 2},
    {"action_index": True, "n_action_steps": 2},
    {"action_index": 0, "n_action_steps": 0},
    {"action_index": 0, "n_action_steps": 3},
])
def test_cached_plan_rejects_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        build_critic_policy_context(np.zeros((2, 6)), **kwargs)


def test_cached_plan_retains_different_preclip_means_at_same_execution_endpoint():
    endpoint = np.array([[100., 0., 0., 0., 0., 50.]])
    overflow = endpoint.copy()
    overflow[0, 0] = 110.
    first = build_critic_policy_context(endpoint, action_index=0, n_action_steps=1)
    second = build_critic_policy_context(overflow, action_index=0, n_action_steps=1)
    assert first[2] == pytest.approx(1.)
    assert second[2] == pytest.approx(1.1)


def test_critic_distinguishes_cached_plans_without_backpropagating_into_actor_inputs():
    from dataclasses import replace
    torch.manual_seed(101)
    cfg = replace(config(), critic_task_context_dim=3, critic_policy_context_dim=critic_policy_context_size(2))
    critic = ActPpoValueNetwork(cfg)
    images = torch.zeros(2, 3, 16, 16, dtype=torch.uint8)
    joints = torch.zeros(2, 6)
    first = np.zeros((2, 6))
    first[:, -1] = 50.
    second = first.copy()
    second[1, 0] = 80.
    cached_plans = torch.from_numpy(np.stack([
        build_critic_policy_context(first, action_index=0, n_action_steps=2),
        build_critic_policy_context(second, action_index=0, n_action_steps=2),
    ])).requires_grad_()
    task = torch.zeros(2, 3, requires_grad=True)
    values = critic(images, images, joints, joints, task, cached_plans)
    assert not torch.isclose(values[0], values[1], atol=1e-7)
    values.square().sum().backward()
    assert cached_plans.grad is None
    assert task.grad is None
    assert critic.state_encoder[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="task_context is required"):
        critic(images, images, joints, joints)
    with pytest.raises(ValueError, match="policy_context must have shape"):
        critic(images, images, joints, joints, task, torch.zeros(2, 1))


def test_saturation_diagnostics_use_policy_bounds_and_loaded_rad_calibration():
    means = np.zeros((2, 6))
    means[:, -1] = 50.
    actions = means + 1.
    stats = policy_action_saturation_diagnostics(means, actions)
    assert stats["raw_mean_out_of_bounds_fraction"] == 0.
    assert stats["raw_mean_clamp_fraction"] == 0.
    assert stats["bounded_action_saturation_fraction"] == 0.
    expected = [
        (joint.gazebo_upper_rad - joint.gazebo_lower_rad)
        / (joint.policy_upper - joint.policy_lower)
        for joint in SIM_JOINT_CALIBRATIONS
    ]
    np.testing.assert_allclose(stats["command_noise_rms_rad_per_joint"], expected)
    means[0, 0] = 100.
    means[0, 5] = -1.
    stats = policy_action_saturation_diagnostics(means)
    assert stats["raw_mean_out_of_bounds_fraction"] == pytest.approx(1 / 12)
    # Legal endpoints no longer enter an EPS clamp band.
    assert stats["raw_mean_clamp_fraction"] == pytest.approx(1 / 12)
    assert stats["raw_mean_clamp_fraction_per_joint"][5] == .5


@pytest.mark.parametrize("boundary_mean", [100., 101.])
def test_boundary_or_overflowed_mean_keeps_likelihood_gradient(boundary_mean):
    from lerobot.policies.act.bounded_actions import bound_action_means
    from so101_capstone.training.act_ppo import configure_bounded_act_policy
    raw = torch.tensor([[boundary_mean, 0., 0., 0., 0., 50.]], requires_grad=True)
    policy = TinyActPolicy()
    configure_bounded_act_policy(policy)
    means = bound_action_means(raw, policy.config)
    actions, _, latent = sample_bounded_policy_actions(
        means, torch.full((6,), -4.), deterministic=True, return_latent=True,
    )
    # A legal inward sample must propagate a likelihood gradient through the
    # smooth parameterization even when the raw network activation exceeds100.
    fixed_latent = latent.detach() - .1
    probability, _ = evaluate_bounded_policy_actions(
        means, actions.detach(), torch.full((6,), -4.), policy_action_latents=fixed_latent,
    )
    probability.sum().backward()
    assert raw.grad.abs().min() > 0.
    assert policy_action_saturation_diagnostics(means)["raw_mean_clamp_fraction"] == 0.


def test_contextual_ppo_update_uses_frozen_context_and_still_updates_act():
    from dataclasses import replace
    torch.manual_seed(41)
    cfg = replace(config(), critic_task_context_dim=2, critic_policy_context_dim=critic_policy_context_size(3))
    policy = TinyActPolicy()
    critic = ActPpoValueNetwork(cfg)
    trainer = ActPpoTrainer(policy, critic, config=cfg, execution_signature={"reward_profile": "pick_place_v2"})
    rollout = ActPpoRolloutBuffer(cfg)
    actor_obs = actor_observation(5)
    ref = rollout.add_actor_observation(actor_obs)
    with torch.no_grad():
        means = predict_action_chunk_with_grad(policy, actor_observation_tensors(actor_obs, device="cpu"))
        actions, log_probs, latents = sample_bounded_policy_actions(means, trainer.exploration.value(), return_latent=True)
    for index in range(3):
        rollout.add(
            critic_observation(index), actor_observation_index=ref, action_index=index,
            policy_action=actions[0, index].numpy(), policy_action_latent=latents[0, index].numpy(),
            policy_action_mean=means[0, index].numpy(),
            policy_action_log_std=trainer.exploration.value().detach().numpy(),
            log_probability=log_probs[0, index].item(), value=0., reward=float(index + 1), next_value=0.,
            terminated=index == 2, episode_end=index == 2,
            critic_task_context=np.asarray([index / 3, 1.], dtype=np.float32),
            critic_policy_context=build_critic_policy_context(means, action_index=index, n_action_steps=3),
        )
    before = model_state_sha256(policy)
    saved_context = np.stack(rollout.critic_policy_context).copy()
    seen = []
    hook = critic.register_forward_pre_hook(lambda module, inputs: seen.append((inputs[4].clone(), inputs[5].clone())))
    try:
        result = trainer.update(rollout)
    finally:
        hook.remove()
    assert result.completed_ppo_epochs == 1
    assert model_state_sha256(policy) != before
    assert seen and all(context.shape[1] == cfg.critic_policy_context_dim for _, context in seen)
    for task, plans in seen:
        for index, plan in zip(task[:, 0], plans):
            np.testing.assert_array_equal(plan.numpy(), saved_context[round(index.item() * 3)])
    np.testing.assert_array_equal(np.stack(rollout.critic_policy_context), saved_context)


def test_checkpoint_signature_roundtrip_and_rejects_environment_or_schedule_change(tmp_path):
    import copy
    cfg = config()
    contract = {"reward_profile": "pick_place_v2", "reward_hash": "abc", "groups": {"idle": -5.}}
    trainer = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(cfg), config=cfg, execution_signature=contract)
    path = trainer.save_checkpoint(tmp_path / "training_state.pt")
    signature = trainer.execution_signature
    contract["groups"]["idle"] = -2.
    assert trainer.execution_signature["environment"]["groups"]["idle"] == -5.
    matching_contract = copy.deepcopy(signature["environment"])
    restored = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(cfg), config=cfg, execution_signature=matching_contract)
    restored.load_checkpoint(path)
    assert model_state_sha256(restored.policy) == model_state_sha256(trainer.policy)
    assert restored.execution_signature_sha256 == trainer.execution_signature_sha256
    changed = TinyActPolicy()
    changed.config.n_action_steps = 2
    changed_trainer = ActPpoTrainer(changed, ActPpoValueNetwork(cfg), config=cfg, execution_signature=matching_contract)
    before = model_state_sha256(changed)
    with pytest.raises(ValueError, match="execution signature"):
        changed_trainer.load_checkpoint(path)
    assert model_state_sha256(changed) == before
    matching_contract["reward_hash"] = "different"
    changed_reward = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(cfg), config=cfg, execution_signature=matching_contract)
    with pytest.raises(ValueError, match="execution signature"):
        changed_reward.load_checkpoint(path)


@pytest.mark.parametrize("old_schema", [1, 2, 3])
def test_old_action_distribution_checkpoint_is_rejected_without_mutating_state(tmp_path, old_schema):
    import copy
    cfg = config()
    legacy = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(cfg), config=cfg)
    payload = copy.deepcopy(legacy.checkpoint_payload())
    payload["schema_version"] = old_schema
    if old_schema == 1:
        payload.pop("execution_signature")
        payload.pop("execution_signature_sha256")
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    before = legacy._epoch_rollback_snapshot()
    with pytest.raises(ValueError, match="[Ss]chema|[Ii]ncompatible|[Ll]egacy"):
        legacy.load_checkpoint(path)
    _assert_nested_state_equal(before, legacy._epoch_rollback_snapshot())


def test_bounded_endpoint_has_finite_truncated_density():
    lower, _ = policy_action_bounds(device="cpu")
    means = lower.unsqueeze(0)
    log_prob, entropy = evaluate_bounded_policy_actions(means, lower.unsqueeze(0), torch.full((6,), -4.))
    assert torch.isfinite(log_prob).all() and torch.isfinite(entropy).all()


@pytest.mark.parametrize("field", ["policy_action_latents", "policy_action_means", "policy_action_log_stds"])
def test_actor_update_requires_all_original_sampling_metadata_before_mutation(field):
    cfg = config()
    policy, critic = TinyActPolicy(), ActPpoValueNetwork(cfg)
    trainer = ActPpoTrainer(policy, critic, config=cfg)
    rollout = _guard_test_rollout(cfg, policy, trainer, critic)
    getattr(rollout, field)[0] = np.full(6, np.nan)
    before = trainer._epoch_rollback_snapshot()

    with pytest.raises(ValueError, match="requires original"):
        trainer.update(rollout)

    _assert_nested_state_equal(before, trainer._epoch_rollback_snapshot())
    assert trainer.update_index == trainer.total_environment_steps == 0


def test_actor_update_rejects_action_mismatched_with_original_sample():
    cfg = config()
    policy, critic = TinyActPolicy(), ActPpoValueNetwork(cfg)
    trainer = ActPpoTrainer(policy, critic, config=cfg)
    rollout = _guard_test_rollout(cfg, policy, trainer, critic)
    rollout.policy_actions[0][0] += 1.
    before = trainer._epoch_rollback_snapshot()

    with pytest.raises(ValueError, match="do not match saved bounded samples"):
        trainer.update(rollout)

    _assert_nested_state_equal(before, trainer._epoch_rollback_snapshot())


@pytest.mark.parametrize("field", ["policy_action_means", "policy_action_log_stds"])
def test_precheck_uses_original_distribution_statistics_not_only_sampled_log_probability(field):
    cfg = config()
    policy, critic = TinyActPolicy(), ActPpoValueNetwork(cfg)
    trainer = ActPpoTrainer(policy, critic, config=cfg)
    rollout = _guard_test_rollout(cfg, policy, trainer, critic)
    getattr(rollout, field)[0] += 1.

    with pytest.raises(RuntimeError, match="pre-update"):
        trainer.update(rollout)

    assert not trainer.optimizer.state


def test_analytic_kl_rejects_candidate_even_if_sampled_kl_is_zero(monkeypatch):
    from dataclasses import replace
    cfg = config()
    policy, critic = TinyActPolicy(), ActPpoValueNetwork(cfg)
    trainer = ActPpoTrainer(policy, critic, config=cfg)
    rollout = _guard_test_rollout(cfg, policy, trainer, critic)
    original = trainer._policy_ratio_diagnostics
    calls = 0

    def underestimated_sampled_kl(*args, **kwargs):
        nonlocal calls
        calls += 1
        diagnostics = original(*args, **kwargs)
        if calls == 2:
            return replace(diagnostics, approximate_kl=0., pre_clip_policy_kl=2. * cfg.target_kl)
        return diagnostics

    monkeypatch.setattr(trainer, "_policy_ratio_diagnostics", underestimated_sampled_kl)
    before = trainer._epoch_rollback_snapshot()
    result = trainer.update(rollout)

    assert result.early_stopped
    assert result.completed_ppo_epochs == result.minibatch_count == 0
    _assert_nested_state_equal(before, trainer._epoch_rollback_snapshot())


@pytest.mark.parametrize("changed_field,new_value", [("n_heads", 4), ("feedforward_activation", "gelu")])
def test_resume_rejects_shape_compatible_act_computation_change(tmp_path, changed_field, new_value):
    cfg = config()
    original_policy = TinyActPolicy()
    original_policy.config.n_heads = 8
    original_policy.config.feedforward_activation = "relu"
    original = ActPpoTrainer(original_policy, ActPpoValueNetwork(cfg), config=cfg)
    checkpoint = original.save_checkpoint(tmp_path / "ppo-schema4.pt")
    assert original.checkpoint_payload()["schema_version"] == 4
    changed_policy = TinyActPolicy()
    changed_policy.config.n_heads = 8
    changed_policy.config.feedforward_activation = "relu"
    setattr(changed_policy.config, changed_field, new_value)
    changed = ActPpoTrainer(changed_policy, ActPpoValueNetwork(cfg), config=cfg)
    before = changed._epoch_rollback_snapshot()

    with pytest.raises(ValueError, match="execution signature"):
        changed.load_checkpoint(checkpoint)

    _assert_nested_state_equal(before, changed._epoch_rollback_snapshot())
