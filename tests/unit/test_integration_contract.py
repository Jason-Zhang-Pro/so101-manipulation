"""Regression checks for profile isolation and ACT logging compatibility."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("project_entry", ROOT / "scripts/project.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_profile_dry_run_cannot_start_a_forwarded_execute():
    with pytest.raises(ValueError, match="launcher"):
        entry.make_plan("primary", "train", ["--execute"])
    plan = entry.make_plan("capstone_ppo", "ppo-train")
    assert plan["execute"] is False
    assert plan["settings"]["ros_workspace"] == "ros2_ws"
    with pytest.raises(ValueError, match="unavailable"):
        entry.make_plan("primary", "ppo-train")


def test_each_profile_matches_its_existing_controller_and_uses_shared_policy_contract():
    profiles = json.loads((ROOT / "configs/runtime_profiles.json").read_text())
    for profile in profiles["profiles"].values():
        text = (ROOT / profile["controller_config"]).read_text()
        assert profile["controller"] + ":" in text
        assert f"update_rate: {profile['controller_hz']}" in text
    assert profiles["policy_contract"]["state_shape"] == [6]
    primary = entry.make_plan("primary", "deploy", execute=True)
    assert primary["command"][-1] == "--execute"
    assert "--execute" not in entry.make_plan("primary", "deploy")["command"]


def test_conflicting_ros_overlay_is_rejected_before_launch(monkeypatch):
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(ROOT / "ros2_ws/install/so101_description"))
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Conflicting overlay must not launch a child process")
    monkeypatch.setattr(entry.subprocess, "call", unexpected_launch)
    with pytest.raises(SystemExit) as error:
        entry.main(["--profile", "primary", "collect", "--execute"])
    assert error.value.code == 2


@pytest.mark.parametrize("bounded", [False, True])
def test_latent_logging_preserves_loss_gradient_and_default_model_contract(bounded):
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        normalization_mapping={"STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY},
        action_bounds_lower=[-100.] * 5 + [0.] if bounded else None,
        action_bounds_upper=[100.] * 6 if bounded else None,
        device="cpu", pretrained_backbone_weights=None, push_to_hub=False,
        chunk_size=3, n_action_steps=1, use_vae=True, latent_dim=4,
        dim_model=16, n_heads=2, dim_feedforward=32,
        n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1, dropout=0.0,
    )
    policy = ACTPolicy(config).train()
    batch = {"observation.state": torch.zeros(2, 6),
             "observation.environment_state": torch.zeros(2, 3),
             "action": torch.ones(2, 3, 6), "action_is_pad": torch.zeros(2, 3, dtype=torch.bool)}
    assert len(policy.model(batch)[1]) == 2
    assert len(policy.model(batch, return_latent=True)[1]) == 3
    torch.manual_seed(17)
    baseline, info = policy.forward(batch)
    baseline.backward()
    gradient = policy.model.action_head.weight.grad.clone()
    assert "_tensorboard_histograms" not in info
    policy.zero_grad()
    policy._log_latents = True
    torch.manual_seed(17)
    logged, info = policy.forward(batch)
    logged.backward()
    torch.testing.assert_close(logged, baseline)
    torch.testing.assert_close(policy.model.action_head.weight.grad, gradient)
    hist = info["_tensorboard_histograms"]
    assert hist["latent/z"].shape == (2, 4)
    assert all(not value.requires_grad for value in hist.values())
    assert len(policy.model(batch)[1]) == 2
