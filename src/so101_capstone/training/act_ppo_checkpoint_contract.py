"""Stable ACT forward configuration for rejecting incompatible PPO resumes.

Parameter shapes alone do not identify an ACT computation: attention heads,
normalization and camera ordering can change without changing those shapes.
This module deliberately includes unknown configuration fields. Only known
runtime locations, optimizer settings and publishing metadata are omitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from numbers import Integral, Real
from pathlib import PurePath
from typing import Any


# Audited against the vendored ACTConfig and PreTrainedConfig. Keep this an
# explicit allowlist of omissions; a future optimizer_* field might affect
# computation and must not disappear solely because of its spelling.
_NON_FORWARD_FIELDS = frozenset({
    "device", "pretrained_path",
    "optimizer_lr", "optimizer_lr_backbone", "optimizer_weight_decay",
    "gradient_accumulation_steps", "kl_weight",
    "push_to_hub", "repo_id", "private", "tags", "license",
})


def _object_fields(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, type):
        raise ValueError("actor configuration must be an instance, not a class")
    result: dict[str, Any] = {}
    if is_dataclass(value):
        result.update((field.name, getattr(value, field.name)) for field in fields(value))
    else:
        # Lightweight test configurations sometimes put their defaults on the
        # class rather than in __dict__. Do not evaluate arbitrary properties.
        for cls in reversed(type(value).__mro__):
            for name, item in vars(cls).items():
                if name.startswith("_") or callable(item) or hasattr(item, "__get__"):
                    continue
                result[name] = item
    try:
        result.update(vars(value))
    except TypeError:
        if not is_dataclass(value):
            raise ValueError("actor configuration must expose named configuration fields") from None
    return result


def _json_key(key: Any, *, location: str) -> str:
    if isinstance(key, Enum):
        key = key.value
    if not isinstance(key, str):
        raise ValueError(f"{location} must use string or string-valued enum keys")
    return key


def _finite_json(value: Any, *, location: str) -> Any:
    if isinstance(value, Enum):
        return _finite_json(value.value, location=location)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{location} must contain finite JSON values")
        return result
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = _json_key(key, location=location)
            if name in result:
                raise ValueError(f"{location} has duplicate serialized key {name!r}")
            result[name] = _finite_json(item, location=f"{location}.{name}")
        return {name: result[name] for name in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, location=f"{location}[{index}]")
                for index, item in enumerate(value)]
    if not callable(value) and (is_dataclass(value) or hasattr(value, "__dict__")):
        return _finite_json(_object_fields(value), location=location)
    raise ValueError(f"{location} contains unsupported configuration type {type(value).__name__}")


def actor_forward_contract(policy_config: Any) -> dict[str, Any]:
    """Return an independent, finite JSON description of ACT computation.

The contract covers every stored dataclass/namespace field unless explicitly
excluded above, including future fields. ``use_amp`` and all architecture,
normalization, feature and temporal settings remain part of the contract.
Feature mapping order is recorded separately because native ACT iterates its
camera mapping when building model inputs; sorting JSON keys would hide a
camera swap. Ordinary mapping insertion order is otherwise immaterial.
"""
    try:
        values = _object_fields(policy_config)
        if not isinstance(policy_config, Mapping):
            # The real LeRobot configuration provides its registered policy
            # type through a property, rather than a dataclass field.
            policy_type = getattr(policy_config, "type", None)
            if policy_type is not None:
                values["type"] = policy_type
        values = {name: item for name, item in values.items() if name not in _NON_FORWARD_FIELDS}
        feature_order = {}
        for name in ("input_features", "output_features", "image_features"):
            value = (policy_config.get(name) if isinstance(policy_config, Mapping)
                     else getattr(policy_config, name, None))
            if isinstance(value, Mapping):
                feature_order[name] = [_json_key(key, location=name) for key in value]
        result = _finite_json({
            "schema_version": 1,
            "config": values,
            "feature_order": feature_order,
        }, location="actor_forward_contract")
        # Enforce the public promise of ordinary JSON values and detach every
        # container from the caller, including mutable future config fields.
        return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
    except RecursionError as error:
        raise ValueError("actor configuration must contain finite, non-cyclic JSON values") from error
