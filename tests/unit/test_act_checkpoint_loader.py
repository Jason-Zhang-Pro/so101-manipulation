import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import so101_capstone.policies.act_checkpoint_loader as loader


def write_checkpoint_stub(
    path: Path,
    *,
    checkpoint_type="act",
    include_config=True,
    include_model=True,
):
    path.mkdir(parents=True)

    if include_config:
        (path / loader.ACT_CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "type": checkpoint_type,
                    "example": 1,
                }
            ),
            encoding="utf-8",
        )

    if include_model:
        (path / loader.ACT_MODEL_FILENAME).write_bytes(
            b"stub"
        )


def feature(shape):
    return SimpleNamespace(shape=shape)


def valid_config():
    return SimpleNamespace(
        input_features={
            loader.ACT_FIXED_IMAGE_KEY: feature(
                (3, 480, 640)
            ),
            loader.ACT_HANDEYE_IMAGE_KEY: feature(
                (3, 480, 640)
            ),
            loader.ACT_STATE_KEY: feature((6,)),
        },
        output_features={
            "action": feature((6,)),
        },
        chunk_size=200,
        n_action_steps=50,
    )


def test_config_payload_removes_act_discriminator(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_stub(checkpoint)

    payload = loader._read_config_payload(
        checkpoint / loader.ACT_CONFIG_FILENAME
    )

    assert payload == {"example": 1}


def test_wrong_checkpoint_type_is_rejected(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_stub(
        checkpoint,
        checkpoint_type="diffusion",
    )

    with pytest.raises(
        ValueError,
        match='"type": "act"',
    ):
        loader._read_config_payload(
            checkpoint / loader.ACT_CONFIG_FILENAME
        )


def test_validate_accepts_current_checkpoint_contract():
    loader.validate_act_config(
        valid_config()
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda config: config.input_features.pop(
                loader.ACT_STATE_KEY
            ),
            "input features",
        ),
        (
            lambda config: setattr(
                config.input_features[
                    loader.ACT_FIXED_IMAGE_KEY
                ],
                "shape",
                (3, 240, 320),
            ),
            "observation.images.fixed",
        ),
        (
            lambda config: setattr(
                config.input_features[
                    loader.ACT_STATE_KEY
                ],
                "shape",
                (12,),
            ),
            "observation.state",
        ),
        (
            lambda config: setattr(
                config.output_features["action"],
                "shape",
                (7,),
            ),
            "ACT action",
        ),
        (
            lambda config: setattr(
                config,
                "n_action_steps",
                201,
            ),
            "n_action_steps",
        ),
    ],
)
def test_invalid_config_contract_is_rejected(
    change,
    message,
):
    config = valid_config()
    change(config)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        loader.validate_act_config(config)


def test_load_config_passes_payload_without_type(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_stub(checkpoint)

    expected_config = valid_config()
    captured = {}

    def fake_parse(payload, *, device):
        captured["payload"] = payload
        captured["device"] = device
        return expected_config

    monkeypatch.setattr(
        loader,
        "_parse_act_config_payload",
        fake_parse,
    )

    config = loader.load_act_config(
        checkpoint,
        device="cpu",
    )

    assert config is expected_config
    assert captured["payload"] == {"example": 1}
    assert captured["device"] == torch.device("cpu")


def test_missing_checkpoint_files_are_rejected(
    tmp_path,
):
    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        loader.load_act_checkpoint(
            missing_directory,
            device="cpu",
        )

    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_stub(
        checkpoint,
        include_model=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="model.safetensors",
    ):
        loader.load_act_checkpoint(
            checkpoint,
            device="cpu",
        )


def test_cuda_request_is_rejected_when_unavailable(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_stub(checkpoint)

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA",
    ):
        loader.load_act_checkpoint(
            checkpoint,
            device="cuda",
        )


def test_load_checkpoint_initializes_policy(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_stub(checkpoint)

    config = valid_config()
    calls = {}

    monkeypatch.setattr(
        loader,
        "load_act_config",
        lambda checkpoint_path, device: config,
    )

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(1)
            )
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    loaded_policy = FakePolicy()
    loaded_policy.train()

    class FakeActPolicy:
        @classmethod
        def from_pretrained(
            cls,
            checkpoint_path,
            **kwargs,
        ):
            calls["checkpoint_path"] = Path(
                checkpoint_path
            )
            calls["kwargs"] = kwargs
            return loaded_policy

    monkeypatch.setattr(
        loader,
        "ACTPolicy",
        FakeActPolicy,
    )

    loaded = loader.load_act_checkpoint(
        checkpoint,
        device="cpu",
        strict=False,
    )

    assert loaded.checkpoint_path == (
        checkpoint.resolve()
    )
    assert loaded.device == torch.device("cpu")
    assert loaded.config is config
    assert loaded.policy is loaded_policy

    assert loaded_policy.training is False
    assert loaded_policy.reset_calls == 1

    assert calls["checkpoint_path"] == (
        checkpoint.resolve()
    )
    assert calls["kwargs"]["config"] is config
    assert calls["kwargs"]["local_files_only"] is True
    assert calls["kwargs"]["strict"] is False


def test_unindexed_cuda_is_resolved_to_current_device(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_stub(checkpoint)

    config = valid_config()

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: 0,
    )

    captured = {}

    def fake_load_config(
        checkpoint_path,
        *,
        device,
    ):
        captured["device"] = device
        return config

    monkeypatch.setattr(
        loader,
        "load_act_config",
        fake_load_config,
    )

    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(
                    1,
                    device="meta",
                )
            )

        def to(self, device):
            captured["policy_device"] = device

            # Avoid requiring real CUDA in this unit test.
            self.weight = torch.nn.Parameter(
                torch.zeros(1)
            )
            return self

        def eval(self):
            self.training = False
            return self

        def reset(self):
            pass

        def parameters(self, recurse=True):
            return iter(
                [
                    SimpleNamespace(
                        device=torch.device("cuda:0")
                    )
                ]
            )

    fake_policy = FakePolicy()

    class FakeActPolicy:
        @classmethod
        def from_pretrained(
            cls,
            checkpoint_path,
            **kwargs,
        ):
            return fake_policy

    monkeypatch.setattr(
        loader,
        "ACTPolicy",
        FakeActPolicy,
    )

    loaded = loader.load_act_checkpoint(
        checkpoint,
        device="cuda",
    )

    # LeRobot configuration receives the generic CUDA device.
    assert captured["device"] == torch.device(
        "cuda"
    )
    assert captured["policy_device"] == torch.device(
        "cuda:0"
    )
    assert loaded.device == torch.device("cuda:0")
