from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101_capstone.training.act_ppo_checkpoint_contract import actor_forward_contract


class Mode(str, Enum):
    ACTION = "ACTION"
    MEAN_STD = "MEAN_STD"
    MIN_MAX = "MIN_MAX"


@dataclass
class Feature:
    type: str
    shape: tuple[int, ...]


@dataclass
class FakeActConfig:
    n_heads: int = 2
    dim_model: int = 8
    pre_norm: bool = False
    vision_backbone: str = "resnet18"
    n_encoder_layers: int = 2
    feedforward_activation: str = "relu"
    chunk_size: int = 15
    n_action_steps: int = 1
    temporal_ensemble_coeff: float | None = 0.01
    normalization_mapping: dict = field(default_factory=lambda: {Mode.ACTION: Mode.MEAN_STD})
    input_features: dict = field(default_factory=lambda: {
        "observation.images.top": Feature("VISUAL", (3, 480, 640)),
        "observation.images.wrist": Feature("VISUAL", (3, 480, 640)),
    })
    output_features: dict = field(default_factory=lambda: {"action": Feature("ACTION", (6,))})
    device: str = "cpu"
    pretrained_path: Path = Path("/checkpoint/original")
    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    kl_weight: float = 10.0
    use_amp: bool = False
    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool = False
    tags: list[str] | None = None
    license: str | None = None


def test_shape_compatible_attention_head_change_changes_contract():
    # Both head counts divide dim_model, and attention projection tensor sizes
    # are identical. Loading weights alone therefore cannot catch this change.
    before = FakeActConfig(n_heads=2)
    after = replace(before, n_heads=4)
    assert before.dim_model % before.n_heads == after.dim_model % after.n_heads == 0
    assert actor_forward_contract(before) != actor_forward_contract(after)


@pytest.mark.parametrize("field_name,value", [
    ("pre_norm", True), ("vision_backbone", "resnet34"),
    ("n_encoder_layers", 3), ("feedforward_activation", "gelu"),
    ("chunk_size", 30), ("n_action_steps", 2),
    ("temporal_ensemble_coeff", 0.02), ("use_amp", True),
    ("normalization_mapping", {Mode.ACTION: Mode.MIN_MAX}),
    ("output_features", {"action": Feature("ACTION", (7,))}),
])
def test_forward_setting_changes_contract(field_name, value):
    original = FakeActConfig()
    modified = replace(original, **{field_name: value})
    assert actor_forward_contract(original) != actor_forward_contract(modified)


def test_runtime_paths_optimizer_and_hub_metadata_do_not_change_contract():
    original = FakeActConfig()
    changed = replace(original, device="cuda:1", pretrained_path=Path("/moved/checkpoint"),
                      optimizer_lr=3e-5, optimizer_lr_backbone=7e-6,
                      optimizer_weight_decay=0.0, gradient_accumulation_steps=4,
                      kl_weight=1.0, push_to_hub=True, repo_id="test/new",
                      private=True, tags=["new"], license="apache-2.0")
    assert actor_forward_contract(original) == actor_forward_contract(changed)


def test_mapping_and_namespace_field_order_have_stable_json_representation():
    first = SimpleNamespace(chunk_size=15, n_heads=2, normalization_mapping={"STATE": "MEAN_STD", "ACTION": "MIN_MAX"})
    second = SimpleNamespace(normalization_mapping={"ACTION": "MIN_MAX", "STATE": "MEAN_STD"}, n_heads=2, chunk_size=15)
    assert json.dumps(actor_forward_contract(first)) == json.dumps(actor_forward_contract(second))


def test_input_camera_order_is_preserved_separately_from_sorted_json_keys():
    first = FakeActConfig()
    second = replace(first, input_features=dict(reversed(list(first.input_features.items()))))
    first_contract, second_contract = actor_forward_contract(first), actor_forward_contract(second)
    assert first_contract["config"] == second_contract["config"]
    assert first_contract["feature_order"] != second_contract["feature_order"]


def test_namespace_features_enum_and_path_are_plain_detached_json():
    config = SimpleNamespace(
        output_features={"action": SimpleNamespace(type=Mode.ACTION, shape=(6,))},
        normalization_mapping={Mode.ACTION: Mode.MEAN_STD},
        custom_asset=Path("asset/calibration.json"), custom_setting={"values": [1.0]},
    )
    contract = actor_forward_contract(config)
    assert contract["config"]["output_features"] == {"action": {"type": "ACTION", "shape": [6]}}
    assert contract["config"]["normalization_mapping"] == {"ACTION": "MEAN_STD"}
    assert contract["config"]["custom_asset"] == "asset/calibration.json"
    assert json.loads(json.dumps(contract, allow_nan=False)) == contract
    config.custom_setting["values"].append(2.0)
    assert contract["config"]["custom_setting"]["values"] == [1.0]


def test_unknown_dataclass_extension_changes_contract():
    first, second = FakeActConfig(), FakeActConfig()
    first.new_forward_option = {"scale": 1.0}
    second.new_forward_option = {"scale": 2.0}
    assert actor_forward_contract(first) != actor_forward_contract(second)


def test_plain_fake_config_supports_class_defaults():
    class FakeConfig:
        chunk_size = 15
        n_heads = 2

        def unused_method(self):
            return "not a configuration value"

    config = FakeConfig()
    config.n_action_steps = 1
    assert actor_forward_contract(config)["config"] == {"chunk_size": 15, "n_heads": 2, "n_action_steps": 1}


@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_nested_configuration_is_rejected(bad):
    with pytest.raises(ValueError, match="finite JSON"):
        actor_forward_contract(SimpleNamespace(new_forward_option={"scale": bad}))


@pytest.mark.parametrize("value", [{"a", "b"}, lambda: None])
def test_unknown_unserializable_values_fail_closed(value):
    with pytest.raises(ValueError, match="unsupported configuration type"):
        actor_forward_contract(SimpleNamespace(new_forward_option=value))


def test_real_act_config_covers_architecture_and_computed_camera_order():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig(device="cpu", chunk_size=15, n_action_steps=1,
                       temporal_ensemble_coeff=0.01, input_features={
                           "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                           "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                       }, output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))})
    contract = actor_forward_contract(config)
    assert contract["config"]["type"] == "act"
    assert contract["config"]["n_heads"] == config.n_heads
    assert contract["config"]["pre_norm"] == config.pre_norm
    assert contract["feature_order"]["image_features"] == list(config.image_features)
    changed = deepcopy(config)
    changed.n_heads //= 2
    assert actor_forward_contract(changed) != contract
