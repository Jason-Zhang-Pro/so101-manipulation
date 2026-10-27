from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_capstone.training.residual_ppo import (
    CompressedResidualObservation,
    ResidualActorCritic,
    ResidualPpoConfig,
    ResidualPpoTrainer,
    ResidualRolloutBuffer,
    compress_policy_observation,
    freeze_module,
    model_state_sha256,
    resize_rgb_image,
)


def small_config(**overrides):
    values = {
        "image_height": 16,
        "image_width": 16,
        "hidden_dim": 32,
        "minibatch_size": 2,
        "ppo_epochs": 1,
        "residual_limit_rad": 0.1,
    }
    values.update(overrides)
    return ResidualPpoConfig(**values)


def image_batch(batch=2, height=16, width=16):
    return torch.randint(0, 256, (batch, 3, height, width), dtype=torch.uint8)


def state_batch(batch=2):
    return torch.zeros((batch, 6), dtype=torch.float32)


def compressed(value=0, *, config=None):
    config = small_config() if config is None else config
    image = np.full(
        (3, config.image_height, config.image_width),
        value,
        dtype=np.uint8,
    )
    vector = np.full(6, float(value), dtype=np.float32)
    return CompressedResidualObservation(
        top_image=image,
        gripper_image=image.copy(),
        joint_positions=vector,
        joint_velocities=np.zeros(6, dtype=np.float32),
    )


def test_actor_critic_outputs_bounded_residuals_and_finite_values():
    config = small_config()
    model = ResidualActorCritic(config)
    residual, log_probability, value = model.act(
        image_batch(),
        image_batch(),
        state_batch(),
        state_batch(),
        state_batch(),
    )

    assert residual.shape == (2, 6)
    assert log_probability.shape == (2,)
    assert value.shape == (2,)
    assert torch.all(torch.abs(residual) <= config.residual_limit_rad)
    assert torch.all(torch.isfinite(log_probability))
    assert torch.all(torch.isfinite(value))


def test_evaluate_actions_reproduces_deterministic_action_log_probability():
    config = small_config()
    model = ResidualActorCritic(config)
    inputs = (
        image_batch(),
        image_batch(),
        state_batch(),
        state_batch(),
        state_batch(),
    )
    residual, original_log_probability, _ = model.act(
        *inputs,
        deterministic=True,
    )
    evaluated_log_probability, entropy, value = model.evaluate_actions(
        *inputs,
        residual,
    )

    torch.testing.assert_close(
        evaluated_log_probability,
        original_log_probability,
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.all(torch.isfinite(entropy))
    assert torch.all(torch.isfinite(value))


def test_critic_does_not_require_a_future_act_base_action():
    model = ResidualActorCritic(small_config())
    value = model.value(
        image_batch(batch=1),
        image_batch(batch=1),
        state_batch(batch=1),
        state_batch(batch=1),
    )
    assert value.shape == (1,)


def test_resize_and_compress_preserve_uint8_contract():
    source = np.zeros((480, 640, 3), dtype=np.uint8)
    source[..., 1] = 200
    resized = resize_rgb_image(source, height=16, width=20)

    assert resized.shape == (3, 16, 20)
    assert resized.dtype == np.uint8
    assert resized.flags.writeable is False

    observation = SimpleNamespace(
        top_image=source,
        gripper_image=source.copy(),
        joint_positions=np.arange(6, dtype=float),
        joint_velocities=np.zeros(6),
    )
    compact = compress_policy_observation(
        observation,
        image_height=16,
        image_width=20,
    )
    assert compact.top_image.shape == (3, 16, 20)
    np.testing.assert_allclose(compact.joint_positions, np.arange(6))


def test_rollout_gae_stops_at_episode_boundary_and_bootstraps_truncation():
    config = small_config(gamma=1.0, gae_lambda=1.0)
    buffer = ResidualRolloutBuffer(config)
    buffer.add(
        compressed(0, config=config),
        base_action=np.zeros(6),
        residual_action=np.zeros(6),
        log_probability=0.0,
        value=1.0,
        reward=2.0,
        next_value=3.0,
        terminated=False,
        episode_end=True,
    )
    buffer.add(
        compressed(1, config=config),
        base_action=np.zeros(6),
        residual_action=np.zeros(6),
        log_probability=0.0,
        value=4.0,
        reward=5.0,
        next_value=999.0,
        terminated=True,
        episode_end=True,
    )

    advantages, returns = buffer._advantages_and_returns()
    np.testing.assert_allclose(advantages, [4.0, 1.0])
    np.testing.assert_allclose(returns, [5.0, 5.0])


def test_ppo_update_changes_only_residual_model_parameters():
    torch.manual_seed(1)
    config = small_config()
    model = ResidualActorCritic(config)
    trainer = ResidualPpoTrainer(model, config=config)
    buffer = ResidualRolloutBuffer(config)

    for index in range(4):
        observation = compressed(index, config=config)
        top = torch.from_numpy(observation.top_image.copy()).unsqueeze(0)
        gripper = torch.from_numpy(observation.gripper_image.copy()).unsqueeze(0)
        positions = torch.from_numpy(observation.joint_positions.copy()).unsqueeze(0)
        velocities = torch.from_numpy(observation.joint_velocities.copy()).unsqueeze(0)
        base = torch.zeros((1, 6))
        with torch.no_grad():
            action, log_probability, value = model.act(
                top,
                gripper,
                positions,
                velocities,
                base,
            )
        buffer.add(
            observation,
            base_action=np.zeros(6),
            residual_action=action.squeeze(0).numpy(),
            log_probability=float(log_probability.item()),
            value=float(value.item()),
            reward=float(index + 1),
            next_value=0.0,
            terminated=index == 3,
            episode_end=index == 3,
        )

    before = model_state_sha256(model)
    metrics = trainer.update(buffer)
    after = model_state_sha256(model)

    assert before != after
    assert metrics.transition_count == 4
    assert metrics.minibatch_count == 2
    assert trainer.update_index == 1


def test_freeze_module_disables_gradients_and_hash_detects_change():
    module = torch.nn.Linear(3, 2)
    first = model_state_sha256(module)
    freeze_module(module)
    assert module.training is False
    assert not any(parameter.requires_grad for parameter in module.parameters())

    with torch.no_grad():
        module.weight.add_(1.0)
    second = model_state_sha256(module)
    assert first != second


def test_checkpoint_round_trip(tmp_path):
    config = small_config()
    first_model = ResidualActorCritic(config)
    first_trainer = ResidualPpoTrainer(first_model, config=config)
    first_trainer.update_index = 3
    first_trainer.total_environment_steps = 600
    path = first_trainer.save_checkpoint(tmp_path / "checkpoint.pt")

    second_model = ResidualActorCritic(config)
    second_trainer = ResidualPpoTrainer(second_model, config=config)
    second_trainer.load_checkpoint(path, map_location="cpu")

    assert model_state_sha256(first_model) == model_state_sha256(second_model)
    assert second_trainer.update_index == 3
    assert second_trainer.total_environment_steps == 600


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_limit_rad": 0.0},
        {"gamma": 1.1},
        {"minibatch_size": 0},
        {"action_dim": 7},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ResidualPpoConfig(**kwargs)
