"""Native ACT inference/export and differentiable output limit regressions."""

import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.bounded_actions import bound_action_means, clamp_action_mean_roundoff
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler


LOWER = [-100.0] * 5 + [0.0]
UPPER = [100.0] * 6


def small_config(*, bounded=True, temporal=None, action_mode=NormalizationMode.IDENTITY, **kwargs):
    return ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        normalization_mapping={"STATE": NormalizationMode.IDENTITY, "ACTION": action_mode},
        action_bounds_lower=list(LOWER) if bounded else None,
        action_bounds_upper=list(UPPER) if bounded else None,
        device="cpu", pretrained_backbone_weights=None, push_to_hub=False,
        chunk_size=3, n_action_steps=1 if temporal is not None else 3,
        temporal_ensemble_coeff=temporal,
        use_vae=False, dim_model=16, n_heads=2, dim_feedforward=32,
        n_encoder_layers=1, n_decoder_layers=1, dropout=0.0,
        **kwargs,
    )


def observation():
    return {"observation.state": torch.zeros(1, 6), "observation.environment_state": torch.zeros(1, 3)}


def set_constant_output(policy, values):
    with torch.no_grad():
        policy.model.action_head.weight.zero_()
        policy.model.action_head.bias.copy_(torch.as_tensor(values))


def test_soft_bounds_preserve_home_and_interior_exactly():
    actions = torch.tensor([[-95.0, -98.0, 95.10715, 68.5987, -1.73639, 4.4901]])
    result = bound_action_means(actions, small_config())
    assert torch.equal(result, actions)
    legacy = small_config(bounded=False)
    assert bound_action_means(actions, legacy) is actions


def test_lightweight_actor_config_and_post_ensemble_roundoff_guard():
    actions = torch.tensor([[109.5, -109.5, 95.0, 0.0, 0.0, 101.0]])
    assert bound_action_means(actions, SimpleNamespace()) is actions
    assert clamp_action_mean_roundoff(actions, SimpleNamespace()) is actions
    config = SimpleNamespace(action_bounds_lower=LOWER, action_bounds_upper=UPPER)
    bounded = bound_action_means(actions, config)
    assert torch.equal(bounded, bound_action_means(actions, small_config()))
    # Never soft-bound an averaged action a second time.
    assert torch.equal(clamp_action_mean_roundoff(bounded, config), bounded)
    just_over_upper = torch.nextafter(torch.tensor(100.0), torch.tensor(float("inf")))
    rounded = torch.full((1, 6), just_over_upper)
    assert torch.equal(clamp_action_mean_roundoff(rounded, config), torch.full((1, 6), 100.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tail_retains_inward_gradient_and_finite_bounded_values(dtype):
    actions = torch.tensor([[109.5, -109.5, 1e6, -1e6, 0.0, 109.5]], dtype=dtype, requires_grad=True)
    bounded = bound_action_means(actions, small_config())
    assert bounded.dtype == actions.dtype
    assert torch.isfinite(bounded).all()
    assert torch.all(bounded >= torch.tensor(LOWER))
    assert torch.all(bounded <= torch.tensor(UPPER))
    bounded.sum().backward()
    assert torch.isfinite(actions.grad).all()
    assert torch.all(actions.grad > 0)
    # At 109.5 the old hard execution clip provided zero mean gradient.
    assert actions.grad[0, 0] == pytest.approx((2.0 / 13.5) ** 2, rel=1e-6)


def test_tail_is_continuously_differentiable_at_interior_boundary():
    config = small_config()
    actions = torch.tensor([[98.0, -98.0, 98.00001, -98.00001, 0.0, 99.0]], dtype=torch.float64, requires_grad=True)
    result = bound_action_means(actions, config)
    result.sum().backward()
    assert torch.allclose(actions.grad, torch.ones_like(actions), atol=1.1e-5, rtol=0)


@pytest.mark.parametrize("field,value", [
    ("action_bounds_upper", None),
    ("action_bounds_upper", [100.0] * 5),
    ("action_bounds_lower", [-100.0] * 5 + [100.0]),
    ("action_bounds_lower", [-100.0] * 5 + [float("nan")]),
    ("action_bound_softness", 0.0),
    ("action_bound_softness", 1.0),
    ("action_bound_softness", float("inf")),
])
def test_invalid_bound_config_is_rejected(field, value):
    config = small_config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate_action_bounds()


@pytest.mark.parametrize("values", [torch.zeros(2, 5), torch.full((2, 6), float("nan")), torch.zeros(2, 6, dtype=torch.int64)])
def test_invalid_predictions_are_rejected(values):
    with pytest.raises(ValueError):
        bound_action_means(values, small_config())


@pytest.mark.parametrize("temporal", [None, 0.01])
def test_native_select_action_bounds_predictions_before_queue_or_temporal_ensemble(temporal):
    policy = ACTPolicy(small_config(temporal=temporal))
    raw = [109.5, -109.5, 95.0, 0.0, 97.0, 120.0]
    set_constant_output(policy, raw)
    expected = policy.bound_action_means(torch.tensor([raw]))
    offline_ensemble = ACTTemporalEnsembler(temporal, 3) if temporal is not None else None
    for i in range(6):
        if temporal is not None:
            set_constant_output(policy, [x - i for x in raw])
            chunk = policy.predict_action_chunk(observation())
            expected = offline_ensemble.update(chunk)
        actual = policy.select_action(observation())
        assert torch.equal(actual, expected)
        assert torch.all(actual >= torch.tensor(LOWER))
        assert torch.all(actual <= torch.tensor(UPPER))


@pytest.mark.parametrize("bounded", [False, True])
def test_standalone_checkpoint_round_trip_preserves_limits_and_outputs(tmp_path, bounded):
    policy = ACTPolicy(small_config(bounded=bounded))
    set_constant_output(policy, [109.5, -109.5, 95.0, 0.0, 0.0, 120.0])
    before = policy.predict_action_chunk(observation())
    checkpoint = tmp_path / "act"
    policy.save_pretrained(checkpoint)
    if not bounded:
        # Simulate a historical checkpoint with none of the new config fields.
        config_path = checkpoint / "config.json"
        payload = json.loads(config_path.read_text())
        for field in ("action_bounds_lower", "action_bounds_upper", "action_bound_softness"):
            payload.pop(field)
        config_path.write_text(json.dumps(payload))
    loaded = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    assert torch.equal(loaded.predict_action_chunk(observation()), before)
    assert loaded.config.action_bounds_lower == (LOWER if bounded else None)
    if not bounded:
        assert before[0, 0, 0] == 109.5


@pytest.mark.parametrize("mode", [NormalizationMode.IDENTITY, NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX])
def test_bounded_bc_forward_matches_inference_and_keeps_gradients(mode):
    stats = {"action": {
        "mean": torch.tensor([10.0] * 6), "std": torch.tensor([2.0] * 6),
        "min": torch.tensor([-50.0] * 6), "max": torch.tensor([50.0] * 6),
    }}
    policy = ACTPolicy(small_config(action_mode=mode), dataset_stats=stats)
    set_constant_output(policy, [109.5] * 6)
    batch = observation() | {"action": torch.zeros(1, 3, 6), "action_is_pad": torch.zeros(1, 3, dtype=torch.bool)}
    predicted_physical = policy.predict_action_chunk(observation())
    expected_predictions = policy.normalize_targets({"action": predicted_physical})["action"]
    expected_targets = policy.normalize_targets(batch)["action"]
    expected_loss = torch.nn.functional.l1_loss(expected_predictions, expected_targets)
    loss, metrics = policy(batch)
    assert loss.detach() == pytest.approx(expected_loss.item(), abs=1e-7)
    assert metrics["l1_loss"] == pytest.approx(expected_loss.item(), abs=1e-7)
    loss.backward()
    assert torch.isfinite(policy.model.action_head.bias.grad).all()
    assert torch.all(policy.model.action_head.bias.grad > 0)
