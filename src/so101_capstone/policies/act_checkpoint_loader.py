"""Load a local LeRobot ACT checkpoint for SO-101 inference.

The installed LeRobot version cannot directly parse an ACT ``config.json``
through ``ACTConfig.from_pretrained`` because the top-level discriminator
field ``"type": "act"`` is passed to the concrete ACTConfig dataclass.

This module removes that discriminator before parsing while leaving the
checkpoint files unchanged.

Backbone download initialization is disabled because the complete trained
backbone parameters are restored from ``model.safetensors``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
from typing import Any

import draccus
import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from .act_policy_adapter import (
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_IMAGE_HEIGHT,
    ACT_IMAGE_WIDTH,
    ACT_STATE_KEY,
)
from .so101_sim_calibration import SIM_POLICY_JOINT_COUNT
from .act_temporal_config import validate_act_temporal_config


ACT_CONFIG_FILENAME = "config.json"
ACT_MODEL_FILENAME = "model.safetensors"

_EXPECTED_INPUT_KEYS = {
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_STATE_KEY,
}

_EXPECTED_OUTPUT_KEYS = {"action"}


@dataclass(frozen=True)
class LoadedActCheckpoint:
    """Loaded ACT policy and its validated configuration."""

    checkpoint_path: Path
    device: torch.device
    config: ACTConfig
    policy: ACTPolicy


def _checkpoint_path(
    checkpoint_path: str | Path,
) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"ACT checkpoint directory does not exist: {path}"
        )

    return path


def _required_file(
    checkpoint_path: Path,
    filename: str,
) -> Path:
    path = checkpoint_path / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"ACT checkpoint file does not exist: {path}"
        )

    return path


def _read_config_payload(
    config_path: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid ACT config JSON: {config_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "ACT config JSON must contain an object."
        )

    checkpoint_type = payload.pop("type", None)

    if checkpoint_type != "act":
        raise ValueError(
            "ACT config must contain top-level "
            '"type": "act".'
        )

    return payload


def _parse_act_config_payload(
    payload: dict[str, Any],
    *,
    device: torch.device,
) -> ACTConfig:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file)
            temporary_path = Path(temporary_file.name)

        with draccus.config_type("json"):
            config = draccus.parse(
                ACTConfig,
                temporary_path,
                args=[],
            )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return replace(
        config,
        device=str(device),
        pretrained_backbone_weights=None,
    )


def _feature_shape(
    feature: Any,
    *,
    name: str,
) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in feature.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"{name} has an invalid feature shape."
        ) from error

    return shape


def validate_act_config(
    config: ACTConfig,
) -> None:
    """Validate the checkpoint contract required by this project."""

    input_keys = set(config.input_features)
    output_keys = set(config.output_features)

    if input_keys != _EXPECTED_INPUT_KEYS:
        raise ValueError(
            "Unexpected ACT input features: "
            f"{sorted(input_keys)}"
        )

    if output_keys != _EXPECTED_OUTPUT_KEYS:
        raise ValueError(
            "Unexpected ACT output features: "
            f"{sorted(output_keys)}"
        )

    expected_image_shape = (
        3,
        ACT_IMAGE_HEIGHT,
        ACT_IMAGE_WIDTH,
    )

    for key in (
        ACT_FIXED_IMAGE_KEY,
        ACT_HANDEYE_IMAGE_KEY,
    ):
        shape = _feature_shape(
            config.input_features[key],
            name=key,
        )

        if shape != expected_image_shape:
            raise ValueError(
                f"{key} must have shape "
                f"{expected_image_shape}, got {shape}."
            )

    state_shape = _feature_shape(
        config.input_features[ACT_STATE_KEY],
        name=ACT_STATE_KEY,
    )

    if state_shape != (SIM_POLICY_JOINT_COUNT,):
        raise ValueError(
            f"{ACT_STATE_KEY} must have shape "
            f"({SIM_POLICY_JOINT_COUNT},), "
            f"got {state_shape}."
        )

    action_shape = _feature_shape(
        config.output_features["action"],
        name="action",
    )

    if action_shape != (SIM_POLICY_JOINT_COUNT,):
        raise ValueError(
            "ACT action must have shape "
            f"({SIM_POLICY_JOINT_COUNT},), "
            f"got {action_shape}."
        )

    validate_act_temporal_config(config)


def load_act_config(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> ACTConfig:
    """Load and validate a local ACT config without modifying it."""

    checkpoint = _checkpoint_path(checkpoint_path)
    config_path = _required_file(
        checkpoint,
        ACT_CONFIG_FILENAME,
    )

    resolved_device = torch.device(device)

    payload = _read_config_payload(config_path)
    config = _parse_act_config_payload(
        payload,
        device=resolved_device,
    )
    validate_act_config(config)

    return config


def load_act_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    strict: bool = False,
) -> LoadedActCheckpoint:
    """Load one local ACT policy for inference."""

    checkpoint = _checkpoint_path(checkpoint_path)

    _required_file(
        checkpoint,
        ACT_CONFIG_FILENAME,
    )
    _required_file(
        checkpoint,
        ACT_MODEL_FILENAME,
    )

    resolved_device = torch.device(device)

    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for ACT inference, "
                "but CUDA is unavailable."
            )

        if resolved_device.index is None:
            resolved_device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

    # LeRobot config validation accepts only generic device names such
    # as "cuda", while PyTorch runtime devices may be indexed as
    # "cuda:0". Keep those two representations separate.
    config_device = torch.device(
        resolved_device.type
    )

    config = load_act_config(
        checkpoint,
        device=config_device,
    )

    policy = ACTPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        strict=strict,
    )

    policy = policy.to(resolved_device)
    policy.eval()
    policy.reset()

    try:
        parameter_device = next(
            policy.parameters()
        ).device
    except StopIteration as error:
        raise RuntimeError(
            "Loaded ACT policy has no parameters."
        ) from error

    if parameter_device != resolved_device:
        raise RuntimeError(
            "ACT policy parameters are on "
            f"{parameter_device}, expected {resolved_device}."
        )

    return LoadedActCheckpoint(
        checkpoint_path=checkpoint,
        device=resolved_device,
        config=config,
        policy=policy,
    )
