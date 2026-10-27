"""Actual ACT/project-loader/checkpoint boundary tests for bounded PPO.

These run a tiny transformer with the actual ResNet18 and project camera metadata.
Small synthetic image tensors reduce test cost; no camera/Gazebo process is used.
"""

import hashlib
import json

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from so101_capstone.policies.act_checkpoint_loader import (
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_STATE_KEY,
    load_act_checkpoint,
)
from so101_capstone.training.act_ppo import (
    ActPpoConfig,
    ActPpoTemporalHistory,
    ActPpoTrainer,
    ActPpoValueNetwork,
    predict_action_chunk_with_grad,
)
from so101_capstone.training.residual_ppo import model_state_sha256


def _file_hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}


def _legacy_act(temporal):
    config = ACTConfig(
        input_features={
            ACT_FIXED_IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            ACT_HANDEYE_IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            ACT_STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        device="cpu", pretrained_backbone_weights=None, push_to_hub=False,
        chunk_size=3, n_action_steps=1 if temporal is not None else 3,
        temporal_ensemble_coeff=temporal,
        use_vae=False, dim_model=16, n_heads=2, dim_feedforward=32,
        n_encoder_layers=1, n_decoder_layers=1, dropout=0.0,
    )
    policy = ACTPolicy(config, dataset_stats={
        "action": {"mean": torch.full((6,), 10.0), "std": torch.full((6,), 2.0)},
    })
    with torch.no_grad():
        policy.model.action_head.weight.zero_()
        policy.model.action_head.bias.copy_(
            (torch.tensor([109.5, -109.5, 95.0, 97.0, -98.0, 109.5]) - 10.0) / 2.0
        )
    return policy.eval()


def _observation():
    return {
        ACT_FIXED_IMAGE_KEY: torch.zeros(1, 3, 32, 32),
        ACT_HANDEYE_IMAGE_KEY: torch.zeros(1, 3, 32, 32),
        ACT_STATE_KEY: torch.zeros(1, 6),
    }


def _trainer(policy, config):
    return ActPpoTrainer(policy, ActPpoValueNetwork(config), config=config)


@pytest.mark.parametrize("temporal", [None, 0.01])
def test_actual_act_warm_start_export_resume_and_execution_contract(tmp_path, temporal):
    torch.manual_seed(41)
    source = tmp_path / "legacy_act"
    original = _legacy_act(temporal)
    original.save_pretrained(source)
    # A true historical file has no newly introduced config keys at all.
    config_path = source / "config.json"
    payload = json.loads(config_path.read_text())
    for key in ("action_bounds_lower", "action_bounds_upper", "action_bound_softness"):
        payload.pop(key)
    config_path.write_text(json.dumps(payload))
    source_hashes = _file_hashes(source)
    original_hash = model_state_sha256(original)
    del original

    loaded = load_act_checkpoint(source, device="cpu", strict=True)
    assert loaded.config.action_bounds_lower is None
    batch = _observation()
    assert loaded.policy.predict_action_chunk(batch)[0, 0, 0] == 109.5
    ppo_config = ActPpoConfig(critic_image_height=16, critic_image_width=16, critic_hidden_dim=16)
    trainer = _trainer(loaded.policy, ppo_config)
    assert model_state_sha256(trainer.policy) == original_hash
    assert _file_hashes(source) == source_hashes
    assert loaded.config.action_bounds_lower == [-100.0] * 5 + [0.0]

    chunk = predict_action_chunk_with_grad(trainer.policy, batch)
    native_chunk = trainer.policy.predict_action_chunk(batch)
    assert torch.equal(chunk.detach(), native_chunk)
    assert 98.0 < chunk[0, 0, 0] < 100.0
    assert torch.all(chunk[..., :5].abs() <= 100.0)
    assert torch.all((chunk[..., 5] >= 0.0) & (chunk[..., 5] <= 100.0))
    chunk.sum().backward()
    bias_gradient = trainer.policy.model.action_head.bias.grad
    assert torch.isfinite(bias_gradient).all() and torch.all(bias_gradient > 0)
    trainer.policy.zero_grad(set_to_none=True)

    # Native deployment and PPO collection agree across queue/TE boundaries.
    trainer.policy.reset()
    history = ActPpoTemporalHistory(temporal, 3) if temporal is not None else None
    for step in range(5):
        if history is not None:
            chunk = predict_action_chunk_with_grad(trainer.policy, batch)
            expected = history.push(step, chunk)[:, 0]
        else:
            expected = native_chunk[:, step % 3]
        assert torch.equal(trainer.policy.select_action(batch), expected)

    trainer.update_index = 7
    trainer.total_environment_steps = 1800
    # Match the export boundary: separate native ACT files and resumable PPO state.
    output = tmp_path / "output" / "checkpoints" / "update_000007"
    exported = {"standalone_act": output / "pretrained_model", "training_state": output / "training_state.pt"}
    trainer.policy.save_pretrained(exported["standalone_act"])
    trainer.save_checkpoint(exported["training_state"], extra={"test": "bounded_contract"})
    reloaded = load_act_checkpoint(exported["standalone_act"], device="cpu", strict=True)
    assert reloaded.config.action_bounds_lower == [-100.0] * 5 + [0.0]
    assert reloaded.config.action_bounds_upper == [100.0] * 6
    assert reloaded.config.action_bound_softness == 0.02
    assert torch.equal(reloaded.policy.predict_action_chunk(batch), native_chunk)
    resumed = _trainer(reloaded.policy, ppo_config)
    with torch.no_grad():
        resumed.policy.model.action_head.bias.add_(2.0)
    assert model_state_sha256(resumed.policy) != original_hash
    state = resumed.load_checkpoint(exported["training_state"], map_location="cpu")
    assert state["schema_version"] == 4
    assert resumed.update_index == 7 and resumed.total_environment_steps == 1800
    assert model_state_sha256(resumed.policy) == original_hash
    assert torch.equal(resumed.policy.predict_action_chunk(batch), native_chunk)

    # A changed mean transform cannot silently resume the old likelihood state.
    resumed.policy.config.action_bound_softness = 0.04
    incompatible = _trainer(resumed.policy, ppo_config)
    with torch.no_grad():
        incompatible.policy.model.action_head.bias.add_(3.0)
    before_actor = model_state_sha256(incompatible.policy)
    before_critic = model_state_sha256(incompatible.critic)
    before_exploration = incompatible.exploration.log_std.detach().clone()
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="execution signature"):
        incompatible.load_checkpoint(exported["training_state"], map_location="cpu")
    assert model_state_sha256(incompatible.policy) == before_actor
    assert model_state_sha256(incompatible.critic) == before_critic
    assert torch.equal(incompatible.exploration.log_std, before_exploration)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert incompatible.update_index == 0 and incompatible.total_environment_steps == 0
    assert _file_hashes(source) == source_hashes
