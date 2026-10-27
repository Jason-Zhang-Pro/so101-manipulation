"""PPO utilities that update ACT directly and export standalone ACT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Collection

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from lerobot.configs.types import NormalizationMode
from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.act.bounded_actions import bound_action_means

from so101_capstone.policies.act_policy_adapter import (
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_STATE_KEY,
)
from so101_capstone.policies.so101_sim_calibration import (
    SIM_CALIBRATION_FILE,
    SIM_JOINT_CALIBRATIONS,
)
from .residual_ppo import CompressedResidualObservation
from .act_ppo_action_distribution import (
    evaluate_policy_actions,
    truncated_gaussian_kl,
    mean_bounds_loss,
    normalized_action_means,
    sample_policy_actions,
)
from .act_ppo_checkpoint_contract import actor_forward_contract
from .act_ppo_diagnostics import phase_policy_diagnostics


ACTION_DIM = 6
CRITIC_POLICY_CONTEXT_SCHEMA_VERSION = 2
ACT_PPO_SAMPLING_SCHEMA_VERSION = 4
ACT_PPO_CHECKPOINT_SCHEMA_VERSION = 4
ACT_PPO_TEMPORAL_HISTORY_SCHEMA_VERSION = 1
_EPS = 1e-6
# _PPO_KL_GUARD_V1
_PRE_UPDATE_MAX_ABS_LOG_RATIO_TOLERANCE = 0.05
_POLICY_LOWER = np.asarray(
    [item.policy_lower for item in SIM_JOINT_CALIBRATIONS],
    dtype=np.float32,
)
_POLICY_UPPER = np.asarray(
    [item.policy_upper for item in SIM_JOINT_CALIBRATIONS],
    dtype=np.float32,
)


class ActPpoNumericalError(FloatingPointError):
    """A non-finite candidate update that can safely be rolled back."""


class ActPpoTemporalHistory:
    """Per-episode rollout ensemble plus references needed for PPO replay.

    Only policy means are ensembled; exploration is sampled afterwards. A
    bootstrap-only observation may use ``None`` but must never enter a saved
    transition. Create a separate instance for every world and episode.
    """

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        if isinstance(temporal_ensemble_coeff, bool) or not math.isfinite(float(temporal_ensemble_coeff)):
            raise ValueError("temporal_ensemble_coeff must be finite.")
        self.chunk_size = _positive_int(chunk_size, name="chunk_size")
        self._ensemble = ACTTemporalEnsembler(float(temporal_ensemble_coeff), chunk_size)
        if (not bool(torch.isfinite(self._ensemble.ensemble_weights).all())
                or not bool(torch.isfinite(self._ensemble.ensemble_weights_cumsum).all())):
            raise ValueError("Temporal ensemble weights overflow for this coefficient and chunk size.")
        self._actor_references: list[int | None] = []

    @property
    def actor_references(self) -> tuple[int | None, ...]:
        return tuple(self._actor_references)

    def reset(self) -> None:
        self._ensemble.reset()
        self._actor_references.clear()

    @torch.no_grad()
    def push(self, actor_ref: int | None, chunk: torch.Tensor) -> torch.Tensor:
        if actor_ref is not None and (
            isinstance(actor_ref, bool) or not isinstance(actor_ref, int) or actor_ref < 0
        ):
            raise ValueError("actor_ref must be a nonnegative integer or None.")
        if chunk.shape != (1, self.chunk_size, ACTION_DIM):
            raise ValueError("Temporal ensemble needs one complete ACT chunk.")
        _require_finite(chunk, name="temporal ACT chunk")
        fused = self._ensemble.update(chunk)
        _require_finite(fused, name="temporal ACT ensemble")
        # Convex averaging is bounded mathematically; only floating-point
        # roundoff can escape. Match native ACT's post-TE arithmetic guard.
        lower, upper = policy_action_bounds(device=fused.device, dtype=fused.dtype)
        fused = torch.maximum(torch.minimum(fused, upper), lower)
        self._actor_references.append(actor_ref)
        self._actor_references = self._actor_references[-self.chunk_size:]
        return fused.unsqueeze(1)


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ActPpoNumericalError(f"{name} contains non-finite values.")


def _require_finite_state(value: Any, *, name: str) -> None:
    if torch.is_tensor(value):
        _require_finite(value, name=name)
    elif isinstance(value, dict):
        for key, item in value.items():
            label = "parameter" if torch.is_tensor(key) else str(key)
            _require_finite_state(item, name=f"{name}.{label}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite_state(item, name=f"{name}.{index}")


def _positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class ActPpoConfig:
    """PPO settings; ACT architecture and default LR come from checkpoint."""

    critic_image_height: int = 96
    critic_image_width: int = 128
    critic_hidden_dim: int = 256
    critic_learning_rate: float = 3e-4
    critic_task_context_dim: int = 0
    critic_policy_context_dim: int = 0
    # Optional stage id in the critic context. None preserves legacy
    # whole-rollout normalization; enabled groups use actor transitions only.
    advantage_group_context_index: int | None = None
    advantage_group_count: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    # Retry a rejected epoch from its full snapshot at a smaller policy LR.
    # Zero preserves the previous rollback-and-stop behavior.
    kl_backoff_max_retries: int = 0
    kl_backoff_factor: float = 0.5
    kl_backoff_min_scale: float = 1e-4
    value_loss_coefficient: float = 0.5
    # Optional regularizer of the PRE-CLIP Gaussian, not executed-action entropy.
    entropy_coefficient: float = 0.0
    mean_bounds_coefficient: float = 0.1
    # Directly regularize physical ACT predictions before the soft action bound.
    # Zero preserves the historical objective and actor recomputation path.
    raw_mean_bounds_coefficient: float = 0.0
    max_gradient_norm: float = 0.5
    critic_max_gradient_norm: float = 0.5
    exploration_max_gradient_norm: float = 0.5
    exploration_learning_rate: float = 1e-4
    ppo_epochs: int = 4
    minibatch_size: int = 128
    initial_log_std: float = -2.3
    minimum_log_std: float = -5.0
    maximum_log_std: float = 0.0
    # Effective gripper upper bound AFTER stage offsets, not the learned base
    # parameter bound. The base remains within minimum/maximum_log_std so larger
    # manipulation noise cannot expand HOME exploration through a shared base.
    # None preserves the shared effective upper bound used by older callers.
    gripper_maximum_log_std: float | None = None
    actor_learning_rate: float | None = None
    actor_backbone_learning_rate: float | None = None

    def __post_init__(self) -> None:
        for name in ("critic_task_context_dim", "critic_policy_context_dim", "kl_backoff_max_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        index = self.advantage_group_context_index
        _positive_int(self.advantage_group_count, name="advantage_group_count")
        if index is not None and (
            isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < self.critic_task_context_dim
        ):
            raise ValueError("advantage_group_context_index must index the critic task context or be None.")
        for name in (
            "critic_image_height",
            "critic_image_width",
            "critic_hidden_dim",
            "ppo_epochs",
            "minibatch_size",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "critic_learning_rate",
            "clip_ratio",
            "target_kl",
            "value_loss_coefficient",
            "max_gradient_norm",
            "critic_max_gradient_norm",
            "exploration_max_gradient_norm",
            "exploration_learning_rate",
        ):
            _positive_float(getattr(self, name), name=name)
        if not 0.0 < _positive_float(self.kl_backoff_factor, name="kl_backoff_factor") < 1.0:
            raise ValueError("kl_backoff_factor must be within (0, 1).")
        if not 0.0 < _positive_float(self.kl_backoff_min_scale, name="kl_backoff_min_scale") <= 1.0:
            raise ValueError("kl_backoff_min_scale must be within (0, 1].")
        for name in ("gamma", "gae_lambda"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1].")
        for name in ("entropy_coefficient", "mean_bounds_coefficient", "raw_mean_bounds_coefficient"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "initial_log_std",
            "minimum_log_std",
            "maximum_log_std",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")
        if not (
            self.minimum_log_std
            <= self.initial_log_std
            <= self.maximum_log_std
        ):
            raise ValueError("initial_log_std is outside configured bounds.")
        if self.gripper_maximum_log_std is not None:
            maximum = self.gripper_maximum_log_std
            if isinstance(maximum, bool) or not math.isfinite(float(maximum)):
                raise ValueError("gripper_maximum_log_std must be finite or None.")
            if float(maximum) < self.minimum_log_std:
                raise ValueError("gripper_maximum_log_std must not be below minimum_log_std.")
        for name in (
            "actor_learning_rate",
            "actor_backbone_learning_rate",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive_float(value, name=name)


@dataclass(frozen=True)
class ActPpoActorObservation:
    top_image: np.ndarray
    gripper_image: np.ndarray
    policy_state: np.ndarray


def _image(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"{name} must have shape (3, height, width).")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8.")
    result = array.copy()
    result.setflags(write=False)
    return result


def _vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (ACTION_DIM,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite six-vector.")
    result = array.copy()
    result.setflags(write=False)
    return result


def actor_observation_from_batch(
    batch: dict[str, torch.Tensor],
) -> ActPpoActorObservation:
    expected = {
        ACT_FIXED_IMAGE_KEY,
        ACT_HANDEYE_IMAGE_KEY,
        ACT_STATE_KEY,
    }
    if set(batch) != expected:
        raise ValueError("ACT batch keys do not match checkpoint contract.")

    def to_image(key: str) -> np.ndarray:
        tensor = batch[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 4
            or tensor.shape[0] != 1
            or tensor.shape[1] != 3
        ):
            raise ValueError(f"{key} has an invalid shape.")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{key} contains non-finite values.")
        if bool(((tensor < 0.0) | (tensor > 1.0)).any().item()):
            raise ValueError(f"{key} must be within [0, 1].")
        return (
            tensor.detach()
            .cpu()
            .float()
            .squeeze(0)
            .mul(255.0)
            .round()
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .numpy()
        )

    state = batch[ACT_STATE_KEY]
    if tuple(state.shape) != (1, ACTION_DIM):
        raise ValueError("ACT state must have shape (1, 6).")
    return ActPpoActorObservation(
        top_image=_image(to_image(ACT_FIXED_IMAGE_KEY), name="top_image"),
        gripper_image=_image(
            to_image(ACT_HANDEYE_IMAGE_KEY),
            name="gripper_image",
        ),
        policy_state=_vector(
            state.detach().cpu().float().numpy()[0],
            name="policy_state",
        ),
    )


def actor_observation_tensors(
    observation: ActPpoActorObservation,
    *,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    target = torch.device(device)

    def to_tensor(value: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.asarray(value).copy())
            .unsqueeze(0)
            .to(device=target, dtype=torch.float32)
            / 255.0
        )

    return {
        ACT_FIXED_IMAGE_KEY: to_tensor(observation.top_image),
        ACT_HANDEYE_IMAGE_KEY: to_tensor(observation.gripper_image),
        ACT_STATE_KEY: (
            torch.from_numpy(np.asarray(observation.policy_state).copy())
            .unsqueeze(0)
            .to(device=target, dtype=torch.float32)
        ),
    }



def _unnormalize_actions_with_grad(
    policy: nn.Module,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Apply checkpoint action statistics without LeRobot's no-grad wrapper."""

    feature = policy.config.output_features[ACTION]
    mode = policy.config.normalization_mapping.get(
        feature.type,
        NormalizationMode.IDENTITY,
    )
    if mode is NormalizationMode.IDENTITY:
        return actions

    buffer_name = "buffer_" + ACTION.replace(".", "_")
    if not hasattr(policy.unnormalize_outputs, buffer_name):
        raise ValueError(
            "ACT checkpoint is missing action normalization statistics."
        )
    statistics = getattr(
        policy.unnormalize_outputs,
        buffer_name,
    )

    if mode is NormalizationMode.MEAN_STD:
        mean = statistics["mean"].detach().to(actions)
        standard_deviation = statistics["std"].detach().to(actions)
        if (
            bool(torch.isinf(mean).any().item())
            or bool(torch.isinf(standard_deviation).any().item())
        ):
            raise ValueError(
                "ACT action mean/std statistics are not initialized."
            )
        return actions * standard_deviation + mean

    if mode is NormalizationMode.MIN_MAX:
        minimum = statistics["min"].detach().to(actions)
        maximum = statistics["max"].detach().to(actions)
        if (
            bool(torch.isinf(minimum).any().item())
            or bool(torch.isinf(maximum).any().item())
        ):
            raise ValueError(
                "ACT action min/max statistics are not initialized."
            )
        return (
            (actions + 1.0)
            / 2.0
            * (maximum - minimum)
            + minimum
        )

    raise ValueError(
        f"Unsupported ACT action normalization mode: {mode}."
    )


def predict_action_chunk_with_grad(
    policy: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Differentiable equivalent of ACTPolicy.predict_action_chunk."""

    return predict_action_chunk_with_raw_grad(policy, batch)[0]


def predict_action_chunk_with_raw_grad(
    policy: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded and pre-bound physical predictions from one ACT forward.

    The raw tensor is already unnormalized into SO-101 policy units. Retaining
    it lets the auxiliary objective reach the actor without passing through the
    soft bound's small tail derivative; rollout likelihoods still use bounded.
    """

    processed = policy.normalize_inputs(batch)
    if policy.config.image_features:
        processed = dict(processed)
        processed[OBS_IMAGES] = [
            processed[key] for key in policy.config.image_features
        ]
    actions = policy.model(processed)[0]
    actions = _unnormalize_actions_with_grad(
        policy,
        actions,
    )
    # Same per-chunk parameterization as native ACT inference, before queueing
    # or temporal ensembling. Never bound an already ensembled mean a second time.
    _require_finite(actions, name="Raw ACT action chunk")
    raw_actions = actions
    actions = bound_action_means(actions, policy.config)
    expected = (
        actions.shape[0],
        int(policy.config.chunk_size),
        ACTION_DIM,
    )
    if tuple(actions.shape) != expected:
        raise ValueError(
            f"ACT action chunk has shape {tuple(actions.shape)}, "
            f"expected {expected}."
        )
    _require_finite(actions, name="ACT action chunk")
    return actions, raw_actions


def raw_action_bounds_statistics(
    raw_actions: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    softness: float | None = 0.02,
) -> dict[str, torch.Tensor]:
    """Per-action overflow loss and diagnostics, averaged across six joints.

    Only actual physical-range overflow is penalized. Legal actions in the
    soft-tail interval remain unpenalized, including the near-limit HOME pose.
    Diagnostics are detached; only the squared-overflow loss needs a gradient.
    ``softness=None`` describes a legacy policy without a mean-bound mapping.
    """
    normalized = normalized_action_means(raw_actions, lower=lower, upper=upper)
    overflow = torch.relu(normalized.abs() - 1.0)
    absolute = normalized.detach().abs()
    if softness is None:
        derivative = torch.ones_like(absolute)
    else:
        if isinstance(softness, bool) or not math.isfinite(float(softness)) or not 0 < softness < 1:
            raise ValueError("Action bound softness must lie in (0, 1), or be None.")
        # Derivative of the C1 rational tail in bounded_actions.py. The affine
        # normalization and inverse scale cancel, so this is d(bound)/d(raw).
        excess = torch.relu(absolute - (1.0 - softness))
        derivative = (softness / (softness + excess)).square()
    result = {
        "raw_mean_bounds_loss": overflow.square().mean(dim=-1),
        "raw_mean_out_of_bounds_fraction": (absolute > 1.0).to(raw_actions.dtype).mean(dim=-1),
        "raw_mean_overflow_normalized": overflow.detach().mean(dim=-1),
        "raw_mean_bound_derivative_mean": derivative.mean(dim=-1),
    }
    for name, value in result.items():
        _require_finite(value, name=name)
    return result


def configure_bounded_act_policy(policy: nn.Module) -> None:
    """Enable schema-4 SO-101 means for PPO without changing source files.

    Historical ACT checkpoints remain unmodified when loaded for evaluation.
    A new PPO warm start adds this serialized inference contract in memory;
    exported ACT then uses it without needing the critic or PPO sampler.
    """
    config = policy.config
    lower = getattr(config, "action_bounds_lower", None)
    upper = getattr(config, "action_bounds_upper", None)
    expected_lower, expected_upper = _POLICY_LOWER.tolist(), _POLICY_UPPER.tolist()
    if lower is None and upper is None:
        config.action_bounds_lower = expected_lower
        config.action_bounds_upper = expected_upper
        config.action_bound_softness = 0.02
    elif lower != expected_lower or upper != expected_upper:
        raise ValueError("ACT PPO requires SO-101 action bounds: five [-100, 100] joints and [0, 100] gripper.")
    # Also validate the softness of custom or lightweight configurations.
    softness = float(getattr(config, "action_bound_softness", 0.02))
    if not math.isfinite(softness) or not 0.0 < softness < 1.0:
        raise ValueError("ACT action_bound_softness must be finite and in (0, 1).")
    config.action_bound_softness = softness
    # Cached unbounded predictions must not cross this contract change.
    if callable(getattr(policy, "reset", None)):
        policy.reset()


def policy_action_bounds(
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(_POLICY_LOWER, device=device, dtype=dtype),
        torch.as_tensor(_POLICY_UPPER, device=device, dtype=dtype),
    )


def critic_policy_context_size(n_action_steps: int) -> int:
    """Size of the cached-plan context, independent of ACT's predicted horizon."""

    _positive_int(n_action_steps, name="n_action_steps")
    return 2 + (ACTION_DIM + 1) * n_action_steps


def _detached_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def build_critic_policy_context(
    action_means: torch.Tensor | np.ndarray,
    *,
    action_index: int,
    n_action_steps: int,
) -> np.ndarray:
    """Describe the remaining deterministic plan before sampling this action.

    Layout: index/N, remaining/N, N*6 normalized suffix means, N valid-row
    bits. The current unsampled mean is the first suffix row. Only the first
    N executable means are represented, even if ACT predicts a longer chunk.
    Padding and masks distinguish an exhausted cache from a zero-valued plan.
    Never pass sampled actions/latents here: a value baseline must not depend
    on the current action's random draw. The returned copy has no actor graph.

    An exhausted context is useful for diagnostics; collectors must generate
    the next chunk before bootstrapping a nonterminal boundary transition.
    """

    size = critic_policy_context_size(n_action_steps)
    if (
        isinstance(action_index, bool)
        or not isinstance(action_index, int)
        or not 0 <= action_index <= n_action_steps
    ):
        raise ValueError("action_index must be within [0, n_action_steps].")
    means = _detached_numpy(action_means).astype(np.float64, copy=False)
    if means.ndim == 3 and means.shape[0] == 1:
        means = means[0]
    if means.ndim != 2 or means.shape[1] != ACTION_DIM or means.shape[0] < n_action_steps:
        raise ValueError("action_means must contain at least n_action_steps six-vectors.")
    if not np.isfinite(means).all():
        raise ValueError("action_means must be finite.")
    centre = (_POLICY_UPPER + _POLICY_LOWER) / 2.0
    scale = (_POLICY_UPPER - _POLICY_LOWER) / 2.0
    # Schema 2 retains the Gaussian location. Means 100 and 110 execute the
    # same deterministic endpoint but have different stochastic transition
    # laws; clipping this feature would hide that distinction from the critic.
    normalized = (means - centre) / scale
    remaining = n_action_steps - action_index
    context = np.zeros(size, dtype=np.float32)
    context[:2] = (action_index / n_action_steps, remaining / n_action_steps)
    suffix = context[2 : 2 + ACTION_DIM * n_action_steps].reshape(n_action_steps, ACTION_DIM)
    suffix[:remaining] = normalized[action_index:n_action_steps]
    context[2 + ACTION_DIM * n_action_steps : 2 + ACTION_DIM * n_action_steps + remaining] = 1.0
    context.setflags(write=False)
    return context


def policy_action_saturation_diagnostics(
    action_means: torch.Tensor | np.ndarray,
    policy_actions: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Report overflow and effective command exploration in physical units.

    Optional actions must be the bounded actions actually sent to the adapter,
    in policy units. Their difference from the effective mean is converted to
    radians using the same loaded calibration as the adapter. This is command
    noise, not measured joint motion or an empirical noise standard deviation.
    """

    means = _detached_numpy(action_means).astype(np.float64, copy=False)
    if means.ndim < 1 or means.shape[-1] != ACTION_DIM or means.size == 0:
        raise ValueError("action_means must be a nonempty array of six-vectors.")
    if not np.isfinite(means).all():
        raise ValueError("action_means must be finite.")
    means = means.reshape(-1, ACTION_DIM)
    centre = (_POLICY_UPPER + _POLICY_LOWER) / 2.0
    scale = (_POLICY_UPPER - _POLICY_LOWER) / 2.0
    normalized = (means - centre) / scale
    outside = np.abs(normalized) > 1.0
    overflow = np.maximum(np.abs(normalized) - 1.0, 0.0)
    result: dict[str, Any] = {
        "sample_count": int(means.shape[0]),
        "raw_mean_out_of_bounds_fraction": float(outside.mean()),
        "raw_mean_out_of_bounds_fraction_per_joint": outside.mean(axis=0).tolist(),
        # Historical field names are retained for log readers. In schema 4,
        # these are bounded output means, not pre-parameterization activations.
        "raw_mean_clamp_fraction": float(outside.mean()),
        "raw_mean_clamp_fraction_per_joint": outside.mean(axis=0).tolist(),
        "raw_mean_overflow_normalized_per_joint": overflow.mean(axis=0).tolist(),
    }
    if policy_actions is not None:
        actions = _detached_numpy(policy_actions).astype(np.float64, copy=False)
        if actions.size != means.size or actions.shape[-1:] != (ACTION_DIM,):
            raise ValueError("policy_actions must have the same number of six-vectors as action_means.")
        actions = actions.reshape(-1, ACTION_DIM)
        if not np.isfinite(actions).all():
            raise ValueError("policy_actions must be finite.")
        normalized_actions = (actions - centre) / scale
        saturated = np.abs(normalized_actions) >= 1.0 - _EPS
        effective_mean = centre + scale * np.clip(normalized, -1.0, 1.0)
        radians_per_unit = np.asarray([
            (joint.gazebo_upper_rad - joint.gazebo_lower_rad)
            / (joint.policy_upper - joint.policy_lower)
            for joint in SIM_JOINT_CALIBRATIONS
        ])
        noise = (actions - effective_mean) * radians_per_unit
        result.update({
            "bounded_action_saturation_fraction": float(saturated.mean()),
            "bounded_action_saturation_fraction_per_joint": saturated.mean(axis=0).tolist(),
            "command_noise_rms_rad_per_joint": np.sqrt(np.mean(noise ** 2, axis=0)).tolist(),
        })
    return result


def _context_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    if value is None:
        if size:
            raise ValueError(f"{name} is required for this critic configuration.")
        return np.empty(0, dtype=np.float32)
    vector = _detached_numpy(value).astype(np.float32, copy=True)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return vector


def _json_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Copy a finite JSON contract without retaining mutable caller state."""

    if not isinstance(value, dict):
        raise ValueError("execution_signature must be a JSON object.")
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("execution_signature must contain finite JSON values.") from error


def _contract_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _execution_signature(
    policy: nn.Module,
    config: ActPpoConfig,
    environment_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    calibrations = {joint.name: asdict(joint) for joint in SIM_JOINT_CALIBRATIONS}
    signature = {
        "schema_version": 3,
        "actor_forward": actor_forward_contract(policy.config),
        "action_distribution": {
            "kind": "bounded_mean_truncated_gaussian",
            "sample_units": "normalized_half_range",
            "lower": _POLICY_LOWER.tolist(), "upper": _POLICY_UPPER.tolist(),
            "kl": "analytic_truncated_gaussian_old_to_new",
            "mean_parameterization": "identity_rational_tails_v1",
            "mean_softness": float(policy.config.action_bound_softness),
        },
        "resolved_optimizer": {
            "actor_lr": float(policy.config.optimizer_lr if config.actor_learning_rate is None else config.actor_learning_rate),
            "backbone_lr": float(policy.config.optimizer_lr_backbone if config.actor_backbone_learning_rate is None else config.actor_backbone_learning_rate),
            "actor_weight_decay": float(policy.config.optimizer_weight_decay),
        },
        "actor_execution": {
            "chunk_size": int(policy.config.chunk_size),
            "n_action_steps": int(policy.config.n_action_steps),
            "temporal_ensemble_coeff": getattr(policy.config, "temporal_ensemble_coeff", None),
        },
        "sampling_schema_version": ACT_PPO_SAMPLING_SCHEMA_VERSION,
        "policy_context_schema_version": CRITIC_POLICY_CONTEXT_SCHEMA_VERSION,
        "critic_task_context_dim": config.critic_task_context_dim,
        "critic_policy_context_dim": config.critic_policy_context_dim,
        "calibration_file_sha256": hashlib.sha256(SIM_CALIBRATION_FILE.read_bytes()).hexdigest(),
        "calibration_values_sha256": _contract_hash(calibrations),
        "environment": environment_contract or {},
    }
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        signature["actor_execution"]["temporal_history_schema_version"] = ACT_PPO_TEMPORAL_HISTORY_SCHEMA_VERSION
    if config.advantage_group_context_index is not None:
        signature["advantage_normalization"] = {
            "kind": ("binary_context_actor_only_v1" if config.advantage_group_count == 2
                     else "categorical_context_actor_only_v1"),
            "context_index": config.advantage_group_context_index,
            "minimum_group_size": 2, "minimum_std": 1e-8,
            "degenerate_group": "zero_actor_advantage",
        }
        if config.advantage_group_count != 2:
            signature["advantage_normalization"]["group_count"] = config.advantage_group_count
    return _json_contract(signature)


def sample_bounded_policy_actions(
    means: torch.Tensor,
    log_std: torch.Tensor,
    *,
    deterministic: bool = False,
    gripper_closing_only: bool = False,
    arm_exploration_enabled: bool = True,
    exploration_target: torch.Tensor | np.ndarray | None = None,
    exploration_target_joint_mask: torch.Tensor | np.ndarray | None = None,
    return_latent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a bounded Gaussian with its proper truncation normalizer.

    Formal PPO requires return_latent=True. The retained sample is the bounded
    draw in normalized action coordinates; no tail is folded onto endpoints.
    Legacy directional controls have no effect; explicit arm freezing is
    supported only for transitions excluded from actor updates.
    """
    if not isinstance(arm_exploration_enabled, bool):
        raise ValueError("arm_exploration_enabled must be a bool.")
    _require_finite(means, name="Action means")
    _require_finite(log_std, name="Exploration log standard deviation")
    lower, upper = policy_action_bounds(device=means.device, dtype=means.dtype)
    try:
        actions, log_prob, latent = sample_policy_actions(
            means, log_std, lower=lower, upper=upper, deterministic=deterministic,
        )
        if not arm_exploration_enabled and not deterministic:
            latent = latent.clone()
            latent[..., :5] = normalized_action_means(means, lower=lower, upper=upper)[..., :5]
            actions = (upper + lower) / 2 + (upper - lower) / 2 * latent
            log_prob, _ = evaluate_policy_actions(means, log_std, latent, lower=lower, upper=upper)
    except FloatingPointError as error:
        raise ActPpoNumericalError(str(error)) from error
    if return_latent:
        return actions, log_prob, latent
    return actions, log_prob


def _inverse_bounded_policy_actions(actions: torch.Tensor) -> torch.Tensor:
    """Affine inverse of a bounded command (formal replay retains the sample)."""
    value = actions.double()
    lower, upper = policy_action_bounds(device=value.device, dtype=value.dtype)
    if bool(((value < lower) | (value > upper)).any()):
        raise ValueError("Policy actions must be inside the action bounds.")
    return (value - (upper + lower) / 2) / ((upper - lower) / 2)


def evaluate_bounded_policy_actions(
    means: torch.Tensor,
    actions: torch.Tensor,
    log_std: torch.Tensor,
    *,
    gripper_closing_only: bool | torch.Tensor = False,
    exploration_target: torch.Tensor | np.ndarray | None = None,
    exploration_target_enabled: bool | torch.Tensor = True,
    exploration_target_joint_mask: torch.Tensor | np.ndarray | None = None,
    policy_action_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Likelihood and entropy of the normalized truncated action distribution."""
    if actions.shape != means.shape:
        raise ValueError("Actions and means must have identical shapes.")
    _require_finite(means, name="Action means")
    _require_finite(log_std, name="Exploration log standard deviation")
    _require_finite(actions, name="Policy actions")
    latent = (_inverse_bounded_policy_actions(actions.to(device=means.device))
              if policy_action_latents is None else policy_action_latents.detach())
    _require_finite(latent, name="Bounded policy action samples")
    lower, upper = policy_action_bounds(device=means.device, dtype=means.dtype)
    try:
        return evaluate_policy_actions(means, log_std, latent, lower=lower, upper=upper)
    except FloatingPointError as error:
        raise ActPpoNumericalError(str(error)) from error


class ActPpoExploration(nn.Module):
    def __init__(self, config: ActPpoConfig) -> None:
        super().__init__()
        self.config = config
        self.log_std = nn.Parameter(
            torch.full((ACTION_DIM,), config.initial_log_std)
        )

    def _maximum(self) -> torch.Tensor:
        maximum = torch.full_like(self.log_std, self.config.maximum_log_std)
        if self.config.gripper_maximum_log_std is not None:
            maximum[-1] = self.config.gripper_maximum_log_std
        return maximum

    def _clamp(self, value: torch.Tensor) -> torch.Tensor:
        return torch.minimum(value.clamp_min(self.config.minimum_log_std), self._maximum())

    def value(self) -> torch.Tensor:
        """Return the learned base, bounded identically for all six axes."""
        return self.log_std.clamp(self.config.minimum_log_std, self.config.maximum_log_std)

    def with_offset(self, offset: float | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Resolve scalar or (..., 6) stage offsets with identical axis bounds.

        Collectors, replay and KL diagnostics must all use this method. The
        gripper's wider cap applies only after stage offsets; the shared learned
        base remains tightly bounded so a grasp offset does not enlarge HOME
        noise. With no gripper override, scalars preserve the old behavior.
        """
        offset = torch.as_tensor(offset, device=self.log_std.device, dtype=self.log_std.dtype)
        if offset.ndim and offset.shape[-1] != ACTION_DIM:
            raise ValueError("Exploration offset must be scalar or end in dimension 6.")
        _require_finite(offset, name="Exploration log standard deviation offset")
        return self._clamp(self.value() + offset)

    def clamp_(self) -> None:
        with torch.no_grad():
            self.log_std.copy_(self.value())


def _initialise(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActPpoValueNetwork(nn.Module):
    """Training-only critic over current measurements and optional task/plan state."""

    def __init__(self, config: ActPpoConfig | None = None) -> None:
        super().__init__()
        self.config = ActPpoConfig() if config is None else config
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.Tanh(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(
                ACTION_DIM * 2
                + self.config.critic_task_context_dim
                + self.config.critic_policy_context_dim,
                128,
            ),
            nn.Tanh(),
        )
        self.body = nn.Sequential(
            nn.Linear(128 * 3, self.config.critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(
                self.config.critic_hidden_dim,
                self.config.critic_hidden_dim,
            ),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(self.config.critic_hidden_dim, 1)
        self.apply(_initialise)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _camera(self, value: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.critic_image_height,
            self.config.critic_image_width,
        )
        if (
            value.ndim != 4
            or value.shape[1] != 3
            or tuple(value.shape[-2:]) != expected
        ):
            raise ValueError(
                "Critic image shape must be "
                f"(batch, 3, {expected[0]}, {expected[1]})."
            )
        return self.camera_encoder(value.float() / 255.0)

    @staticmethod
    def _state(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != ACTION_DIM:
            raise ValueError(f"{name} must have shape (batch, 6).")
        return value.float()

    @staticmethod
    def _context(
        value: torch.Tensor | None,
        reference: torch.Tensor,
        *,
        size: int,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            if size:
                raise ValueError(f"{name} is required for this critic configuration.")
            return reference.new_empty((reference.shape[0], 0), dtype=torch.float32)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (reference.shape[0], size):
            raise ValueError(f"{name} must have shape (batch, {size}).")
        _require_finite(value, name=name)
        # The plan can originate from ACT, but value loss must not update ACT
        # through this auxiliary path. PPO owns the actor's gradients.
        return value.detach().to(device=reference.device, dtype=torch.float32)

    def forward(
        self,
        top_images: torch.Tensor,
        gripper_images: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        task_context: torch.Tensor | None = None,
        policy_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        top = self._camera(top_images)
        gripper = self._camera(gripper_images)
        state = self.state_encoder(
            torch.cat(
                (
                    self._state(joint_positions, name="joint_positions"),
                    self._state(joint_velocities, name="joint_velocities"),
                    self._context(
                        task_context, joint_positions,
                        size=self.config.critic_task_context_dim, name="task_context",
                    ),
                    self._context(
                        policy_context, joint_positions,
                        size=self.config.critic_policy_context_dim, name="policy_context",
                    ),
                ),
                dim=-1,
            )
        )
        hidden = self.body(torch.cat((top, gripper, state), dim=-1))
        return self.value_head(hidden).squeeze(-1)


@dataclass(frozen=True)
class ActPpoRolloutBatch:
    actor_top_images: torch.Tensor
    actor_gripper_images: torch.Tensor
    actor_policy_states: torch.Tensor
    actor_observation_indices: torch.Tensor
    action_indices: torch.Tensor
    policy_actions: torch.Tensor
    policy_action_latents: torch.Tensor
    old_action_means: torch.Tensor
    old_action_log_stds: torch.Tensor
    critic_top_images: torch.Tensor
    critic_gripper_images: torch.Tensor
    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    critic_task_context: torch.Tensor
    critic_policy_context: torch.Tensor
    old_log_probabilities: torch.Tensor
    actor_update_enabled: torch.Tensor
    exploration_log_std_offsets: torch.Tensor
    gripper_closing_only: torch.Tensor
    exploration_targets: torch.Tensor
    exploration_target_enabled: torch.Tensor
    exploration_target_joint_masks: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    temporal_actor_references: tuple[tuple[int, ...] | None, ...] = ()
    advantage_normalization: dict[str, Any] = field(default_factory=dict)


class ActPpoRolloutBuffer:
    """Unique chunk observations plus step-level PPO transitions."""

    def __init__(self, config: ActPpoConfig | None = None) -> None:
        self.config = ActPpoConfig() if config is None else config
        self.clear()

    def clear(self) -> None:
        self.actor_observations: list[ActPpoActorObservation] = []
        self.actor_refs: list[int] = []
        self.action_indices: list[int] = []
        self.temporal_actor_references: list[tuple[int, ...] | None] = []
        self.policy_actions: list[np.ndarray] = []
        self.policy_action_latents: list[np.ndarray | None] = []
        self.policy_action_means: list[np.ndarray | None] = []
        self.policy_action_log_stds: list[np.ndarray | None] = []
        self.top_images: list[np.ndarray] = []
        self.gripper_images: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.critic_task_context: list[np.ndarray] = []
        self.critic_policy_context: list[np.ndarray] = []
        self.log_probabilities: list[float] = []
        self.actor_update_enabled: list[bool] = []
        self.exploration_log_std_offsets: list[np.ndarray] = []
        self.gripper_closing_only: list[bool] = []
        self.exploration_targets: list[np.ndarray] = []
        self.exploration_target_enabled: list[bool] = []
        self.exploration_target_joint_masks: list[np.ndarray] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.next_values: list[float] = []
        self.terminated: list[bool] = []
        self.episode_end: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def extend(self, other: "ActPpoRolloutBuffer") -> int:
        """Append one complete rollout segment while remapping actor refs.

        Multi-world collection keeps each episode in a private buffer until
        that episode completes successfully. Appending whole episode buffers
        keeps transitions from each trajectory contiguous, which is required
        by the reverse GAE pass in :meth:`_advantages_and_returns`.
        """

        if not isinstance(other, ActPpoRolloutBuffer):
            raise TypeError("other must be an ActPpoRolloutBuffer.")
        if self.config != other.config:
            raise ValueError("Rollout buffer configurations differ.")
        transition_count = len(other)
        if transition_count == 0:
            return 0
        actor_offset = len(self.actor_observations)
        for observation in other.actor_observations:
            self.add_actor_observation(observation)
        self.actor_refs.extend(
            actor_offset + int(reference) for reference in other.actor_refs
        )
        self.action_indices.extend(int(value) for value in other.action_indices)
        self.temporal_actor_references.extend(
            None if refs is None else tuple(actor_offset + ref for ref in refs)
            for refs in other.temporal_actor_references
        )
        self.policy_actions.extend(value.copy() for value in other.policy_actions)
        for name in ("policy_action_latents", "policy_action_means", "policy_action_log_stds"):
            getattr(self, name).extend(None if value is None else value.copy() for value in getattr(other, name))
        self.top_images.extend(value.copy() for value in other.top_images)
        self.gripper_images.extend(value.copy() for value in other.gripper_images)
        self.positions.extend(value.copy() for value in other.positions)
        self.velocities.extend(value.copy() for value in other.velocities)
        self.critic_task_context.extend(value.copy() for value in other.critic_task_context)
        self.critic_policy_context.extend(value.copy() for value in other.critic_policy_context)
        self.log_probabilities.extend(float(value) for value in other.log_probabilities)
        self.actor_update_enabled.extend(bool(value) for value in other.actor_update_enabled)
        self.exploration_log_std_offsets.extend(
            value.copy() for value in other.exploration_log_std_offsets
        )
        self.gripper_closing_only.extend(bool(value) for value in other.gripper_closing_only)
        self.exploration_targets.extend(
            value.copy() for value in other.exploration_targets
        )
        self.exploration_target_enabled.extend(
            bool(value) for value in other.exploration_target_enabled
        )
        self.exploration_target_joint_masks.extend(
            value.copy() for value in other.exploration_target_joint_masks
        )
        self.values.extend(float(value) for value in other.values)
        self.rewards.extend(float(value) for value in other.rewards)
        self.next_values.extend(float(value) for value in other.next_values)
        self.terminated.extend(bool(value) for value in other.terminated)
        self.episode_end.extend(bool(value) for value in other.episode_end)
        return transition_count

    def without_stage_groups(self, excluded_groups: Collection[int]) -> "ActPpoRolloutBuffer":
        """Return complete episodes outside the excluded configured stage groups.

        Exclusion removes transitions from every objective, including critic,
        exploration and raw-action regularization; it is not an actor mask.
        GAE boundaries and truncation bootstrap values remain unchanged. Actor
        references, including TE histories, are compacted to retained episodes.

        Source lists are untouched. Mutable transition arrays are copied, while
        frozen actor observations with read-only arrays share their large image
        storage. Unreferenced actor images are neither copied nor retained.
        Callers must not use this operation on a partially collected episode.
        """
        requested = tuple(excluded_groups)
        if any(isinstance(group, (bool, np.bool_))
               or not isinstance(group, (int, np.integer)) or group not in range(self.config.advantage_group_count)
               for group in requested):
            raise ValueError("Excluded stage groups must contain only integer 0 or 1."
                             if self.config.advantage_group_count == 2 else
                             "Excluded stage groups must contain configured integer stage ids.")
        excluded = set(requested)
        context_index = self.config.advantage_group_context_index
        if context_index is None:
            raise ValueError("Stage filtering requires advantage_group_context_index.")
        fields = {name: values for name, values in vars(self).items()
                  if isinstance(values, list) and name != "actor_observations"}
        count = len(self)
        if any(len(values) != count for values in fields.values()):
            raise ValueError("Rollout transition field lengths do not match.")
        result = ActPpoRolloutBuffer(self.config)
        if not count:
            return result
        if not self.episode_end[-1]:
            raise ValueError("Stage filtering requires complete episodes; the final episode is incomplete.")

        retained = []
        start = 0
        episode_group = None
        episode_references: set[int] = set()
        for index in range(count):
            context = np.asarray(self.critic_task_context[index])
            if (context.shape != (self.config.critic_task_context_dim,)
                    or not np.isfinite(context).all()
                    or context[context_index] not in range(self.config.advantage_group_count)):
                raise ValueError("Stage filtering requires a finite binary stage context."
                                 if self.config.advantage_group_count == 2 else
                                 "Stage filtering requires a finite configured stage id.")
            group = int(context[context_index])
            if episode_group is None:
                episode_group = group
            elif group != episode_group:
                raise ValueError("Stage filtering cannot split an episode containing multiple stage groups.")
            reference = self.actor_refs[index]
            if (isinstance(reference, bool) or not isinstance(reference, (int, np.integer))
                    or not 0 <= reference < len(self.actor_observations)):
                raise ValueError("Stage filtering encountered an invalid actor reference.")
            episode_references.add(int(reference))
            history = self.temporal_actor_references[index]
            if history is not None:
                if (not isinstance(history, tuple) or not history
                        or any(isinstance(ref, bool) or not isinstance(ref, (int, np.integer))
                               for ref in history)
                        or history[-1] != reference or self.action_indices[index] != 0
                        or any(a >= b for a, b in zip(history, history[1:]))):
                    raise ValueError("Stage filtering encountered an invalid temporal actor history.")
                if any(ref not in episode_references for ref in history):
                    raise ValueError("Temporal actor history crosses an episode boundary or references a future observation.")
            if self.terminated[index] and not self.episode_end[index]:
                raise ValueError("Terminated rollout transitions must end an episode.")
            if self.episode_end[index]:
                if episode_group not in excluded:
                    retained.extend(range(start, index + 1))
                start = index + 1
                episode_group = None
                episode_references.clear()

        references = set(self.actor_refs[index] for index in retained)
        for index in retained:
            references.update(self.temporal_actor_references[index] or ())
        mapping = {reference: position for position, reference in enumerate(sorted(references))}
        for reference in sorted(references):
            observation = self.actor_observations[reference]
            arrays = (observation.top_image, observation.gripper_image, observation.policy_state)
            if all(not value.flags.writeable for value in arrays):
                result.actor_observations.append(observation)
            else:
                result.add_actor_observation(observation)
        for name, values in fields.items():
            if name == "actor_refs":
                copied = [mapping[values[index]] for index in retained]
            elif name == "temporal_actor_references":
                copied = [None if values[index] is None else tuple(mapping[ref] for ref in values[index])
                          for index in retained]
            else:
                copied = [values[index].copy() if isinstance(values[index], np.ndarray) else values[index]
                          for index in retained]
            setattr(result, name, copied)
        return result

    def checkpoint(self) -> dict[str, int]:
        """Snapshot every mutable rollout list before one episode attempt."""

        return {
            name: len(value)
            for name, value in vars(self).items()
            if isinstance(value, list)
        }

    def rollback(self, checkpoint: dict[str, int]) -> int:
        """Discard everything appended after a rollout checkpoint."""

        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint must be a dict[str, int].")

        lists = {
            name: value
            for name, value in vars(self).items()
            if isinstance(value, list)
        }
        if set(checkpoint) != set(lists):
            raise ValueError(
                "Rollout checkpoint fields do not match the buffer."
            )

        before = len(self)
        for name, value in lists.items():
            length = checkpoint[name]
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or length < 0
                or length > len(value)
            ):
                raise ValueError(
                    f"Invalid rollback length for {name}: {length}."
                )
            del value[length:]

        return before - len(self)

    def add_actor_observation(
        self,
        observation: ActPpoActorObservation,
    ) -> int:
        self.actor_observations.append(
            ActPpoActorObservation(
                top_image=_image(observation.top_image, name="top_image"),
                gripper_image=_image(
                    observation.gripper_image,
                    name="gripper_image",
                ),
                policy_state=_vector(
                    observation.policy_state,
                    name="policy_state",
                ),
            )
        )
        return len(self.actor_observations) - 1

    def add(
        self,
        critic_observation: CompressedResidualObservation,
        *,
        actor_observation_index: int,
        action_index: int,
        policy_action: np.ndarray,
        log_probability: float,
        policy_action_latent: np.ndarray | None = None,
        policy_action_mean: np.ndarray | None = None,
        policy_action_log_std: np.ndarray | None = None,
        temporal_actor_references: tuple[int, ...] | None = None,
        critic_task_context: np.ndarray | None = None,
        critic_policy_context: np.ndarray | None = None,
        actor_update_enabled: bool = True,
        exploration_log_std_offset: float | np.ndarray = 0.0,
        gripper_closing_only: bool = False,
        exploration_target: np.ndarray | None = None,
        exploration_target_joint_mask: np.ndarray | None = None,
        value: float,
        reward: float,
        next_value: float,
        terminated: bool,
        episode_end: bool,
    ) -> None:
        if not 0 <= actor_observation_index < len(self.actor_observations):
            raise ValueError("Invalid actor observation index.")
        if isinstance(action_index, bool) or action_index < 0:
            raise ValueError("action_index must be non-negative.")
        if temporal_actor_references is not None:
            if (
                not isinstance(temporal_actor_references, tuple)
                or not temporal_actor_references
                or any(isinstance(ref, bool) or not isinstance(ref, int)
                       or not 0 <= ref < len(self.actor_observations)
                       for ref in temporal_actor_references)
                or temporal_actor_references[-1] != actor_observation_index
                or any(a >= b for a, b in zip(temporal_actor_references, temporal_actor_references[1:]))
                or action_index != 0
            ):
                raise ValueError("Temporal actor references must be ordered, valid, end at the current observation, and use action_index=0.")
        expected = (
            3,
            self.config.critic_image_height,
            self.config.critic_image_width,
        )
        for name, image_value in (
            ("top_image", critic_observation.top_image),
            ("gripper_image", critic_observation.gripper_image),
        ):
            array = np.asarray(image_value)
            if array.shape != expected or array.dtype != np.uint8:
                raise ValueError(
                    f"{name} must have shape {expected} and dtype uint8."
                )
        for name, scalar in (
            ("log_probability", log_probability),
            ("value", value),
            ("reward", reward),
            ("next_value", next_value),
        ):
            if not math.isfinite(float(scalar)):
                raise ValueError(f"{name} must be finite.")
        offset_array = np.asarray(exploration_log_std_offset, dtype=np.float32)
        if offset_array.ndim == 0:
            offset_array = np.full(ACTION_DIM, offset_array.item(), dtype=np.float32)
        stored_offset = _vector(offset_array, name="exploration_log_std_offset").copy()
        if not isinstance(actor_update_enabled, bool):
            raise ValueError("actor_update_enabled must be a bool.")
        if not isinstance(gripper_closing_only, bool):
            raise ValueError("gripper_closing_only must be a bool.")
        if exploration_target is None:
            if exploration_target_joint_mask is not None:
                raise ValueError(
                    "exploration_target_joint_mask requires "
                    "exploration_target."
                )
            stored_exploration_target = np.zeros(
                ACTION_DIM,
                dtype=np.float32,
            )
            stored_exploration_target_joint_mask = np.zeros(
                ACTION_DIM,
                dtype=bool,
            )
            exploration_target_enabled = False
        else:
            stored_exploration_target = _vector(
                exploration_target,
                name="exploration_target",
            ).copy()
            if exploration_target_joint_mask is None:
                stored_exploration_target_joint_mask = np.ones(
                    ACTION_DIM,
                    dtype=bool,
                )
            else:
                stored_exploration_target_joint_mask = np.asarray(
                    exploration_target_joint_mask,
                    dtype=bool,
                )
                if stored_exploration_target_joint_mask.shape != (ACTION_DIM,):
                    raise ValueError(
                        "exploration_target_joint_mask must be a six-vector."
                    )
                stored_exploration_target_joint_mask = (
                    stored_exploration_target_joint_mask.copy()
                )
            exploration_target_enabled = True
        if terminated and not episode_end:
            raise ValueError("Terminated transitions must end the episode.")

        stored_task_context = _context_vector(
            critic_task_context, size=self.config.critic_task_context_dim,
            name="critic_task_context",
        )
        group_index = self.config.advantage_group_context_index
        if group_index is not None and stored_task_context[group_index] not in range(self.config.advantage_group_count):
            raise ValueError("Advantage group context must contain a binary 0/1 stage flag."
                             if self.config.advantage_group_count == 2 else
                             "Advantage group context must contain a configured integer stage id.")
        stored_policy_context = _context_vector(
            critic_policy_context, size=self.config.critic_policy_context_dim,
            name="critic_policy_context",
        )

        stored_action = _vector(policy_action, name="policy_action").copy()
        if policy_action_latent is None:
            # Value-only/GAE buffers need no policy metadata. Actor updates
            # reject missing samples rather than inventing one from a clip.
            stored_latent = None
        else:
            stored_latent = np.asarray(policy_action_latent, dtype=np.float64)
            if stored_latent.shape != (ACTION_DIM,) or not np.isfinite(stored_latent).all():
                raise ValueError("policy_action_latent must be a finite six-vector.")
        stored_mean = (None if policy_action_mean is None
                       else _vector(policy_action_mean, name="policy_action_mean").copy())
        stored_log_std = (None if policy_action_log_std is None
                          else _vector(policy_action_log_std, name="policy_action_log_std").copy())
        self.actor_refs.append(int(actor_observation_index))
        self.action_indices.append(int(action_index))
        self.temporal_actor_references.append(temporal_actor_references)
        self.policy_actions.append(stored_action)
        self.policy_action_latents.append(None if stored_latent is None else stored_latent.copy())
        self.policy_action_means.append(stored_mean)
        self.policy_action_log_stds.append(stored_log_std)
        self.top_images.append(np.asarray(critic_observation.top_image).copy())
        self.gripper_images.append(
            np.asarray(critic_observation.gripper_image).copy()
        )
        self.positions.append(
            np.asarray(critic_observation.joint_positions).copy()
        )
        self.velocities.append(
            np.asarray(critic_observation.joint_velocities).copy()
        )
        self.critic_task_context.append(stored_task_context)
        self.critic_policy_context.append(stored_policy_context)
        self.log_probabilities.append(float(log_probability))
        self.actor_update_enabled.append(actor_update_enabled)
        self.exploration_log_std_offsets.append(
            stored_offset
        )
        self.gripper_closing_only.append(gripper_closing_only)
        self.exploration_targets.append(stored_exploration_target)
        self.exploration_target_enabled.append(
            exploration_target_enabled
        )
        self.exploration_target_joint_masks.append(
            stored_exploration_target_joint_mask
        )
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.episode_end.append(bool(episode_end))

    def _advantages_and_returns(self) -> tuple[np.ndarray, np.ndarray]:
        if not self:
            raise ValueError("Cannot process an empty rollout.")
        advantages = np.zeros(len(self), dtype=np.float32)
        gae = 0.0
        for index in range(len(self) - 1, -1, -1):
            bootstrap = (
                0.0 if self.terminated[index] else self.next_values[index]
            )
            delta = (
                self.rewards[index]
                + self.config.gamma * bootstrap
                - self.values[index]
            )
            continuation = 0.0 if self.episode_end[index] else 1.0
            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * continuation
                * gae
            )
            advantages[index] = gae
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        return advantages, returns

    def _normalize_advantages(self, advantages: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        index = self.config.advantage_group_context_index
        if index is None:
            # Preserve historical arithmetic and inclusion of value-only
            # transitions for configurations that did not opt into grouping.
            return ((advantages - float(advantages.mean()))
                    / max(float(advantages.std()), 1e-8)), {}
        flags = np.stack(self.critic_task_context)[:, index]
        if not np.isin(flags, np.arange(self.config.advantage_group_count)).all():
            raise ValueError("Advantage group context must contain a binary 0/1 stage flag."
                             if self.config.advantage_group_count == 2 else
                             "Advantage group context must contain a configured integer stage id.")
        actor_mask = np.asarray(self.actor_update_enabled, dtype=bool)
        normalized = np.zeros_like(advantages)
        groups: dict[str, Any] = {}
        for group in range(self.config.advantage_group_count):
            selected = actor_mask & (flags == group)
            raw = advantages[selected].astype(np.float64)
            count = int(selected.sum())
            mean = float(raw.mean()) if count else None
            std = float(raw.std()) if count else None
            degenerate = count < 2 or std <= 1e-8
            if not degenerate:
                normalized[selected] = (raw - mean) / std
            scaled = normalized[selected].astype(np.float64)
            groups[str(group)] = {
                "actor_transition_count": count,
                "value_only_transition_count": int(((~actor_mask) & (flags == group)).sum()),
                "degenerate": degenerate,
                "raw_mean": mean, "raw_std": std,
                "raw_positive_fraction": float((raw > 0).mean()) if count else None,
                "raw_negative_fraction": float((raw < 0).mean()) if count else None,
                "normalized_mean": float(scaled.mean()) if count else None,
                "normalized_std": float(scaled.std()) if count else None,
                "normalized_positive_fraction": float((scaled > 0).mean()) if count else None,
                "normalized_negative_fraction": float((scaled < 0).mean()) if count else None,
            }
        return normalized, {
            "kind": ("binary_context_actor_only_v1" if self.config.advantage_group_count == 2
                     else "categorical_context_actor_only_v1"), "context_index": index,
            "actor_transition_count": int(actor_mask.sum()), "groups": groups,
        }

    def advantage_normalization_diagnostics(self) -> dict[str, Any]:
        """Report raw/grouped actor advantages without modifying the rollout."""
        advantages, _ = self._advantages_and_returns()
        return self._normalize_advantages(advantages)[1]

    def batch(self, device: str | torch.device) -> ActPpoRolloutBatch:
        advantages, returns = self._advantages_and_returns()
        advantages, advantage_diagnostics = self._normalize_advantages(advantages)
        target = torch.device(device)

        def tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(value, device=target, dtype=dtype)

        def policy_metadata(values: list[np.ndarray | None], dtype: torch.dtype) -> torch.Tensor:
            # None stays JSON-safe in value-only episode records. NaN exists
            # only in a batch, where update rejects it for actor transitions.
            return tensor(np.stack([np.full(ACTION_DIM, np.nan) if v is None else v for v in values]), dtype)

        return ActPpoRolloutBatch(
            actor_top_images=tensor(
                np.stack([item.top_image for item in self.actor_observations]),
                torch.uint8,
            ),
            actor_gripper_images=tensor(
                np.stack(
                    [item.gripper_image for item in self.actor_observations]
                ),
                torch.uint8,
            ),
            actor_policy_states=tensor(
                np.stack([item.policy_state for item in self.actor_observations]),
                torch.float32,
            ),
            actor_observation_indices=tensor(self.actor_refs, torch.long),
            action_indices=tensor(self.action_indices, torch.long),
            policy_actions=tensor(
                np.stack(self.policy_actions),
                torch.float32,
            ),
            policy_action_latents=policy_metadata(self.policy_action_latents, torch.float64),
            old_action_means=policy_metadata(self.policy_action_means, torch.float32),
            old_action_log_stds=policy_metadata(self.policy_action_log_stds, torch.float32),
            critic_top_images=tensor(
                np.stack(self.top_images),
                torch.uint8,
            ),
            critic_gripper_images=tensor(
                np.stack(self.gripper_images),
                torch.uint8,
            ),
            joint_positions=tensor(
                np.stack(self.positions),
                torch.float32,
            ),
            joint_velocities=tensor(
                np.stack(self.velocities),
                torch.float32,
            ),
            critic_task_context=tensor(np.stack(self.critic_task_context), torch.float32),
            critic_policy_context=tensor(np.stack(self.critic_policy_context), torch.float32),
            old_log_probabilities=tensor(
                self.log_probabilities,
                torch.float64,
            ),
            actor_update_enabled=tensor(
                self.actor_update_enabled,
                torch.bool,
            ),
            exploration_log_std_offsets=tensor(
                np.stack(self.exploration_log_std_offsets),
                torch.float32,
            ),
            gripper_closing_only=tensor(
                self.gripper_closing_only,
                torch.bool,
            ),
            exploration_targets=tensor(
                np.stack(self.exploration_targets),
                torch.float32,
            ),
            exploration_target_enabled=tensor(
                self.exploration_target_enabled,
                torch.bool,
            ),
            exploration_target_joint_masks=tensor(
                np.stack(self.exploration_target_joint_masks),
                torch.bool,
            ),
            advantages=tensor(advantages, torch.float32),
            returns=tensor(returns, torch.float32),
            temporal_actor_references=tuple(self.temporal_actor_references),
            advantage_normalization=advantage_diagnostics,
        )


@dataclass(frozen=True)
class ActPpoUpdateMetrics:
    transition_count: int
    minibatch_count: int
    configured_ppo_epochs: int
    completed_ppo_epochs: int
    early_stopped: bool
    target_kl: float
    pre_update_approximate_kl: float
    pre_update_clip_fraction: float
    pre_update_max_abs_log_ratio: float
    final_policy_approximate_kl: float
    final_policy_clip_fraction: float
    final_policy_max_abs_log_ratio: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    actor_learning_rate: float
    actor_backbone_learning_rate: float
    critic_learning_rate: float
    log_std_mean: float
    mean_bounds_loss: float = 0.0
    raw_mean_bounds_loss: float = 0.0
    raw_mean_out_of_bounds_fraction: float = 0.0
    raw_mean_overflow_normalized: float = 0.0
    raw_mean_bound_derivative_mean: float = 0.0
    # Accepted actor transition visits, including repeated visits across epochs.
    # Zero means raw diagnostics were not collected, not that no overflow exists.
    raw_mean_bounds_diagnostic_count: int = 0
    actor_gradient_norm: float = 0.0
    critic_gradient_norm: float = 0.0
    exploration_gradient_norm: float = 0.0
    exploration_learning_rate: float = 0.0
    pre_update_pre_clip_policy_kl: float = 0.0
    pre_clip_policy_kl: float = 0.0
    pre_clip_policy_kl_p95: float = 0.0
    pre_clip_policy_kl_max: float = 0.0
    actor_parameter_relative_delta: float = 0.0
    actor_parameter_changed_fraction: float = 0.0
    critic_parameter_relative_delta: float = 0.0
    critic_parameter_changed_fraction: float = 0.0
    exploration_parameter_relative_delta: float = 0.0
    exploration_parameter_changed_fraction: float = 0.0
    exploration_log_std: tuple[float, ...] = ()
    exploration_log_std_delta: tuple[float, ...] = ()
    effective_exploration_log_std_upper_bounds: tuple[float, ...] = ()
    kl_backoff_attempt_count: int = 0
    kl_backoff_rejection_count: int = 0
    kl_backoff_retry_count: int = 0
    policy_learning_rate_scale: float = 1.0
    kl_backoff_at_min_scale: bool = False
    kl_backoff_attempts: tuple[dict[str, Any], ...] = ()
    phase_diagnostics: dict[str, Any] = field(default_factory=dict)
    advantage_normalization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["truncated_gaussian_entropy"] = self.entropy
        result["action_distribution"] = "bounded_mean_truncated_gaussian"
        result["sampling_schema_version"] = ACT_PPO_SAMPLING_SCHEMA_VERSION
        # Old attribute names remain aliases for existing analysis clients;
        # schema and canonical names identify the actual bounded distribution.
        for name in ("pre_update_pre_clip_policy_kl", "pre_clip_policy_kl",
                     "pre_clip_policy_kl_p95", "pre_clip_policy_kl_max"):
            result[name.replace("pre_clip_", "")] = result[name]
        return result


@dataclass(frozen=True)
class _ActPpoRatioDiagnostics:
    actor_transition_count: int
    approximate_kl: float
    clip_fraction: float
    max_abs_log_ratio: float
    pre_clip_policy_kl: float = 0.0
    pre_clip_policy_kl_p95: float = 0.0
    pre_clip_policy_kl_max: float = 0.0
    phase_diagnostics: dict[str, Any] = field(default_factory=dict)


class ActPpoTrainer:
    def __init__(
        self,
        policy: nn.Module,
        critic: ActPpoValueNetwork,
        *,
        config: ActPpoConfig | None = None,
        execution_signature: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.critic = critic
        self.config = critic.config if config is None else config
        if self.config != critic.config:
            raise ValueError("Trainer and critic configurations differ.")
        configure_bounded_act_policy(policy)
        self._environment_execution_contract = (
            None if execution_signature is None else _json_contract(execution_signature)
        )
        # Snapshot the loaded calibration and execution contract now. A new
        # profile must never use a legacy checkpoint lacking that contract.
        self._execution_signature = _execution_signature(
            policy, self.config, self._environment_execution_contract,
        )
        try:
            self.device = next(policy.parameters()).device
        except StopIteration as error:
            raise ValueError("ACT has no parameters.") from error

        # ACT weights are trainable; dataset normalization statistics are fixed.
        normalization_parameters = {
            id(parameter)
            for name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
            for parameter in getattr(policy, name, nn.Identity()).parameters()
        }
        for parameter in policy.parameters():
            parameter.requires_grad_(id(parameter) not in normalization_parameters)
        policy.eval()
        critic.to(self.device).eval()
        self.exploration = ActPpoExploration(self.config).to(self.device)

        self.actor_learning_rate = (
            float(policy.config.optimizer_lr)
            if self.config.actor_learning_rate is None
            else float(self.config.actor_learning_rate)
        )
        self.actor_backbone_learning_rate = (
            float(policy.config.optimizer_lr_backbone)
            if self.config.actor_backbone_learning_rate is None
            else float(self.config.actor_backbone_learning_rate)
        )
        weight_decay = float(policy.config.optimizer_weight_decay)

        backbone = []
        other = []
        for name, parameter in policy.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone if name.startswith("model.backbone") else other).append(
                parameter
            )
        groups: list[dict[str, Any]] = [
            {
                "params": other,
                "lr": self.actor_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": list(critic.parameters()),
                "lr": self.config.critic_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": list(self.exploration.parameters()),
                "lr": self.config.exploration_learning_rate,
                "weight_decay": 0.0,
            },
        ]
        if backbone:
            groups.insert(
                1,
                {
                    "params": backbone,
                    "lr": self.actor_backbone_learning_rate,
                    "weight_decay": weight_decay,
                },
            )
        self.optimizer = torch.optim.AdamW(groups)
        self._base_actor_learning_rate = self.actor_learning_rate
        self._base_actor_backbone_learning_rate = self.actor_backbone_learning_rate
        self._base_group_learning_rates = tuple(float(group["lr"]) for group in groups)
        self._policy_group_indices = tuple(i for i in range(len(groups)) if i != len(groups) - 2)
        self.policy_learning_rate_scale = 1.0
        self.update_index = 0
        self.total_environment_steps = 0

    @property
    def execution_signature(self) -> dict[str, Any]:
        return copy.deepcopy(self._execution_signature)

    @property
    def execution_signature_sha256(self) -> str:
        return _contract_hash(self._execution_signature)

    def _set_policy_learning_rate_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("PPO policy learning-rate scale must be within (0, 1].")
        self.policy_learning_rate_scale = float(scale)
        for index, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self._base_group_learning_rates[index] * (
                scale if index in self._policy_group_indices else 1.0
            )
        self.actor_learning_rate = self._base_actor_learning_rate * scale
        self.actor_backbone_learning_rate = self._base_actor_backbone_learning_rate * scale

    def _learning_rate_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy_scale": self.policy_learning_rate_scale,
            "base_actor_learning_rate": self._base_actor_learning_rate,
            "base_actor_backbone_learning_rate": self._base_actor_backbone_learning_rate,
            "base_group_learning_rates": list(self._base_group_learning_rates),
        }

    def _checkpoint_policy_learning_rate_scale(self, payload: dict[str, Any]) -> float:
        """Validate runtime scale and optimizer LRs before mutating any weights."""
        state = payload.get("learning_rate_state")
        scale = 1.0 if state is None else state.get("policy_scale") if isinstance(state, dict) else None
        if (isinstance(scale, bool) or not isinstance(scale, (float, int))
                or not math.isfinite(scale) or not 0.0 < scale <= 1.0):
            raise ValueError("ACT PPO checkpoint learning-rate state is invalid.")
        if state is not None:
            expected = {**self._learning_rate_state(), "policy_scale": scale}
            if state != expected:
                raise ValueError("ACT PPO checkpoint base learning rates do not match.")
        groups = payload["optimizer_state_dict"]["param_groups"]
        if len(groups) != len(self._base_group_learning_rates) or any(
            not math.isclose(float(group["lr"]), base * (scale if index in self._policy_group_indices else 1.0), rel_tol=1e-12, abs_tol=0.0)
            for index, (group, base) in enumerate(zip(groups, self._base_group_learning_rates, strict=True))
        ):
            raise ValueError("ACT PPO checkpoint optimizer learning rates do not match runtime scale.")
        return float(scale)

    @staticmethod
    def _rng_snapshot() -> dict[str, Any]:
        state = {
            "torch": torch.get_rng_state().clone(),
            "numpy": copy.deepcopy(np.random.get_state()),
            "python": random.getstate(),
        }
        if torch.cuda.is_initialized():
            state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
        return state

    @staticmethod
    def _restore_rng_snapshot(state: dict[str, Any]) -> None:
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _actor_batch(
        self,
        batch: ActPpoRolloutBatch,
        indices: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        references = batch.actor_observation_indices[indices]
        unique_references, inverse = torch.unique(
            references,
            sorted=True,
            return_inverse=True,
        )
        return (
            {
                ACT_FIXED_IMAGE_KEY: (
                    batch.actor_top_images[unique_references].float()
                    / 255.0
                ),
                ACT_HANDEYE_IMAGE_KEY: (
                    batch.actor_gripper_images[unique_references].float()
                    / 255.0
                ),
                ACT_STATE_KEY: (
                    batch.actor_policy_states[unique_references]
                ),
            },
            inverse,
        )

    def _actor_means(
        self, batch: ActPpoRolloutBatch, indices: torch.Tensor,
    ) -> torch.Tensor:
        return self._actor_recomputed_outputs(batch, indices, include_raw_bounds=False)[0]

    def _actor_means_and_raw_bounds(
        self, batch: ActPpoRolloutBatch, indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute means and contributor-wise raw statistics in the same forwards."""
        return self._actor_recomputed_outputs(batch, indices, include_raw_bounds=True)

    def _raw_bounds_statistics(self, raw_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        lower, upper = policy_action_bounds(device=raw_actions.device, dtype=raw_actions.dtype)
        bounded = getattr(self.policy.config, "action_bounds_lower", None) is not None
        return raw_action_bounds_statistics(
            raw_actions, lower=lower, upper=upper,
            softness=(float(getattr(self.policy.config, "action_bound_softness", 0.02)) if bounded else None),
        )

    def _actor_recomputed_outputs(
        self, batch: ActPpoRolloutBatch, indices: torch.Tensor, *, include_raw_bounds: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Recompute the complete conditional mean under the current policy.

        For temporal ensembling the condition includes the observation history.
        Historical predictions must retain gradients: freezing them would change
        PPO's distribution after an update and invalidate both ratio and KL.
        Singleton ACT forwards preserve the rollout's numerical semantics.
        """
        coeff = getattr(self.policy.config, "temporal_ensemble_coeff", None)
        selected = indices.detach().cpu().tolist()
        histories = batch.temporal_actor_references
        if coeff is None:
            if histories and any(histories[index] is not None for index in selected):
                raise ValueError("Temporal rollout cannot be updated by a non-temporal policy.")
            actor_batch, inverse = self._actor_batch(batch, indices)
            action_indices = batch.action_indices[indices]
            if include_raw_bounds:
                chunks, raw_chunks = self._predict_action_chunks_and_raw_singleton(actor_batch)
                statistics = self._raw_bounds_statistics(raw_chunks[inverse, action_indices])
            else:
                chunks = self._predict_action_chunks_singleton(actor_batch)
                statistics = {}
            return chunks[inverse, action_indices], statistics
        if int(self.policy.config.n_action_steps) != 1:
            raise ValueError("Temporal PPO requires n_action_steps=1.")
        if len(histories) != int(batch.action_indices.numel()):
            raise ValueError("Temporal PPO rollout is missing actor history.")
        chosen: list[tuple[int, ...]] = []
        chunk_size = int(self.policy.config.chunk_size)
        for index in selected:
            refs = histories[index]
            if (not refs or len(refs) > chunk_size
                    or refs[-1] != int(batch.actor_observation_indices[index])
                    or int(batch.action_indices[index]) != 0):
                raise ValueError("Temporal PPO rollout has invalid actor history.")
            chosen.append(refs)
        unique = sorted({ref for refs in chosen for ref in refs})
        def predict_one(top: torch.Tensor, gripper: torch.Tensor, state: torch.Tensor):
            actor_batch = {
                ACT_FIXED_IMAGE_KEY: top.float() / 255.0,
                ACT_HANDEYE_IMAGE_KEY: gripper.float() / 255.0,
                ACT_STATE_KEY: state,
            }
            if include_raw_bounds:
                return self._predict_action_chunks_and_raw_singleton(actor_batch)
            return self._predict_action_chunks_singleton(actor_batch)

        predicted = []
        raw_predicted = []
        for reference in unique:
            inputs = (
                batch.actor_top_images[reference:reference + 1],
                batch.actor_gripper_images[reference:reference + 1],
                batch.actor_policy_states[reference:reference + 1],
            )
            # Histories can contain minibatch_size * chunk_size observations.
            # Save uint8 input views, not every backbone's activations or a
            # large float image copy; recompute each singleton on backward.
            chunk = (activation_checkpoint(predict_one, *inputs, use_reentrant=False)
                     if torch.is_grad_enabled() else predict_one(*inputs))
            if include_raw_bounds:
                chunk, raw_chunk = chunk
                raw_predicted.append(raw_chunk[0])
            predicted.append(chunk[0])
        chunks = torch.stack(predicted)
        positions = {ref: position for position, ref in enumerate(unique)}
        raw_statistics = {}
        statistic_positions = {}
        if include_raw_bounds:
            # Evaluate only predictions contributing to the selected actions;
            # unused ACT chunk positions cannot influence this objective.
            contributors = sorted({(positions[ref], len(refs) - age - 1)
                                   for refs in chosen for age, ref in enumerate(refs)})
            rows, columns = zip(*contributors)
            raw_chunks = torch.stack(raw_predicted)
            raw_statistics = self._raw_bounds_statistics(raw_chunks[list(rows), list(columns)])
            statistic_positions = {location: index for index, location in enumerate(contributors)}
        # LeRobot constructs these in default float32 on CPU and moves them to
        # the action device. Preserve both that precision and its multiply,
        # add, divide recurrence, rather than using a differently rounded sum.
        weights = torch.exp(-float(coeff) * torch.arange(chunk_size))
        sums = torch.cumsum(weights, dim=0).to(chunks.device)
        weights = weights.to(chunks.device)
        means = []
        statistics = {name: [] for name in raw_statistics}
        for refs in chosen:
            mean = chunks[positions[refs[0]], len(refs) - 1]
            # Penalize each prediction before weighting. Opposite overflowing
            # predictions must not cancel merely because TE's mean is legal.
            terms = {name: values[statistic_positions[positions[refs[0]], len(refs) - 1]]
                     for name, values in raw_statistics.items()}
            for age, ref in enumerate(refs[1:], start=1):
                mean = mean * sums[age - 1]
                mean = mean + chunks[positions[ref], len(refs) - age - 1] * weights[age]
                mean = mean / sums[age]
                for name, values in raw_statistics.items():
                    terms[name] = (terms[name] * sums[age - 1]
                                   + values[statistic_positions[positions[ref], len(refs) - age - 1]]
                                   * weights[age]) / sums[age]
            means.append(mean)
            for name, value in terms.items():
                statistics[name].append(value)
        fused = torch.stack(means)
        lower, upper = policy_action_bounds(device=fused.device, dtype=fused.dtype)
        return (torch.maximum(torch.minimum(fused, upper), lower),
                {name: torch.stack(values) for name, values in statistics.items()})

    def _parameters(self) -> list[nn.Parameter]:
        return [
            *self.policy.parameters(),
            *self.critic.parameters(),
            *self.exploration.parameters(),
        ]

    def _trainable_groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        return {
            group: [(name, p) for name, p in module.named_parameters() if p.requires_grad]
            for group, module in (("actor", self.policy), ("critic", self.critic), ("exploration", self.exploration))
        }

    def _parameter_update_diagnostics(
        self, baseline: dict[str, dict[str, torch.Tensor]], used: dict[str, set[str]],
    ) -> dict[str, float]:
        """Measure accepted changes; unused VAE/statistic parameters are excluded."""
        result = {}
        for group, parameters in self._trainable_groups().items():
            changed = count = 0
            delta_squared = base_squared = 0.0
            for name, p in parameters:
                if name not in used[group]:
                    continue
                old = baseline[group][name]
                current = p.detach().cpu()
                changed += int(torch.count_nonzero(current != old))
                count += p.numel()
                delta_squared += float((current.double() - old.double()).square().sum())
                base_squared += float(old.double().square().sum())
            result[f"{group}_parameter_changed_fraction"] = changed / max(count, 1)
            result[f"{group}_parameter_relative_delta"] = math.sqrt(delta_squared) / max(math.sqrt(base_squared), 1e-12)
        return result

    def _analytic_policy_kl(
        self, batch: ActPpoRolloutBatch, indices: torch.Tensor,
        means: torch.Tensor, log_std: torch.Tensor,
    ) -> torch.Tensor:
        lower, upper = policy_action_bounds(device=means.device, dtype=means.dtype)
        try:
            result = truncated_gaussian_kl(
                normalized_action_means(batch.old_action_means[indices], lower=lower, upper=upper),
                batch.old_action_log_stds[indices],
                normalized_action_means(means, lower=lower, upper=upper), log_std,
            )
        except FloatingPointError as error:
            raise ActPpoNumericalError(str(error)) from error
        _require_finite(result, name="Analytic truncated policy KL")
        return result

    # _PPO_EPOCH_ROLLBACK_V1
    @staticmethod
    def _rollback_clone_to_cpu(value: Any) -> Any:
        # Clone nested training state to CPU for one-epoch rollback.
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: ActPpoTrainer._rollback_clone_to_cpu(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                ActPpoTrainer._rollback_clone_to_cpu(item)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                ActPpoTrainer._rollback_clone_to_cpu(item)
                for item in value
            )
        return copy.deepcopy(value)

    def _epoch_rollback_snapshot(self) -> dict[str, Any]:
        # Keep one epoch-start snapshot on CPU, not duplicated on GPU.
        return {
            "policy": self._rollback_clone_to_cpu(
                self.policy.state_dict()
            ),
            "critic": self._rollback_clone_to_cpu(
                self.critic.state_dict()
            ),
            "exploration": self._rollback_clone_to_cpu(
                self.exploration.state_dict()
            ),
            "optimizer": self._rollback_clone_to_cpu(
                self.optimizer.state_dict()
            ),
        }

    def _restore_epoch_rollback_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> None:
        self.policy.load_state_dict(snapshot["policy"])
        self.critic.load_state_dict(snapshot["critic"])
        self.exploration.load_state_dict(snapshot["exploration"])
        # load_state_dict may reuse CPU tensors (including Adam's CPU step on
        # CUDA). Never hand it the reusable snapshot itself: later retries would
        # otherwise mutate the saved moments/counter and change their own origin.
        self.optimizer.load_state_dict(self._rollback_clone_to_cpu(snapshot["optimizer"]))
        self.optimizer.zero_grad(set_to_none=True)

    # _PPO_ACTOR_SINGLETON_FORWARD_V2
    def _predict_action_chunks_singleton(
        self,
        actor_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # CAPSTONE PERMANENT INVARIANT:
        # rollout ACT chunk generation uses one observation at a time.
        # PPO actor recomputation must preserve the same batch=1 forward
        # semantics. MINIBATCH_SIZE still applies to transitions, critic,
        # PPO losses, gradients, and optimizer.step().
        #
        # Real CUDA diagnosis on 2026-08-09 measured batch=1 vs batch=30
        # ACT mean drift of about 0.005267 policy units. Near bounded-action
        # saturation, this amplified into pre-update log-ratio magnitudes
        # in the hundreds despite unchanged policy parameters.

        if ACT_STATE_KEY not in actor_batch:
            raise KeyError(
                f"ACT actor batch is missing {ACT_STATE_KEY!r}."
            )

        batch_size = int(actor_batch[ACT_STATE_KEY].shape[0])
        if batch_size < 1:
            raise ValueError(
                "ACT singleton recompute needs at least one observation."
            )

        chunks: list[torch.Tensor] = []
        for index in range(batch_size):
            singleton_batch = {
                key: value[index : index + 1]
                for key, value in actor_batch.items()
            }
            chunk = predict_action_chunk_with_grad(
                self.policy,
                singleton_batch,
            )
            if int(chunk.shape[0]) != 1:
                raise RuntimeError(
                    "ACT singleton recompute returned a non-singleton "
                    f"batch: shape={tuple(chunk.shape)}."
                )
            chunks.append(chunk[0])

        return torch.stack(chunks, dim=0)

    def _predict_action_chunks_and_raw_singleton(
        self, actor_batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain pre-bound outputs without changing singleton ACT semantics."""
        if ACT_STATE_KEY not in actor_batch:
            raise KeyError(f"ACT actor batch is missing {ACT_STATE_KEY!r}.")
        batch_size = int(actor_batch[ACT_STATE_KEY].shape[0])
        if batch_size < 1:
            raise ValueError("ACT singleton recompute needs at least one observation.")
        bounded_chunks, raw_chunks = [], []
        for index in range(batch_size):
            singleton_batch = {key: value[index:index + 1] for key, value in actor_batch.items()}
            bounded, raw = predict_action_chunk_with_raw_grad(self.policy, singleton_batch)
            if int(bounded.shape[0]) != 1 or raw.shape != bounded.shape:
                raise RuntimeError("ACT raw singleton recompute returned inconsistent shapes.")
            bounded_chunks.append(bounded[0])
            raw_chunks.append(raw[0])
        return torch.stack(bounded_chunks), torch.stack(raw_chunks)

    # _PPO_KL_GROWTH_DIAGNOSTIC_V2_TWO_EPOCHS
    def _diagnostic_policy_ratio_stats(
        self,
        batch: ActPpoRolloutBatch,
        indices: torch.Tensor,
    ) -> dict[str, int | float]:
        # Overflow-safe post-step probability diagnostics.
        actor_transition_count = 0
        clipped_count = 0
        kl_sum = 0.0
        kl_is_infinite = False
        max_abs_log_ratio = 0.0
        max_log_ratio = -float("inf")
        min_log_ratio = float("inf")

        lower_clip_log = math.log(1.0 - self.config.clip_ratio)
        upper_clip_log = math.log(1.0 + self.config.clip_ratio)

        self.policy.eval()
        with torch.no_grad():
            for start in range(
                0,
                int(indices.numel()),
                self.config.minibatch_size,
            ):
                candidate_indices = indices[
                    start : start + self.config.minibatch_size
                ]
                actor_indices = candidate_indices[
                    batch.actor_update_enabled[candidate_indices]
                ]
                if not int(actor_indices.numel()):
                    continue

                means = self._actor_means(batch, actor_indices)
                exploration_log_std = self.exploration.with_offset(
                    batch.exploration_log_std_offsets[actor_indices]
                )
                new_log_prob, _ = evaluate_bounded_policy_actions(
                    means,
                    batch.policy_actions[actor_indices],
                    exploration_log_std,
                    policy_action_latents=batch.policy_action_latents[actor_indices],
                    gripper_closing_only=(
                        batch.gripper_closing_only[actor_indices]
                    ),
                    exploration_target=(
                        batch.exploration_targets[actor_indices]
                    ),
                    exploration_target_enabled=(
                        batch.exploration_target_enabled[actor_indices]
                    ),
                    exploration_target_joint_mask=(
                        batch.exploration_target_joint_masks[actor_indices]
                    ),
                )

                log_ratio = (
                    new_log_prob
                    - batch.old_log_probabilities[actor_indices]
                ).double()
                if not bool(torch.isfinite(log_ratio).all().item()):
                    return {
                        "actor_transition_count": actor_transition_count,
                        "approximate_kl": float("inf"),
                        "clip_fraction": 1.0,
                        "max_abs_log_ratio": float("inf"),
                        "max_log_ratio": float("inf"),
                        "min_log_ratio": -float("inf"),
                    }

                count = int(log_ratio.numel())
                actor_transition_count += count

                clipped_count += int(
                    (
                        (log_ratio < lower_clip_log)
                        | (log_ratio > upper_clip_log)
                    )
                    .sum()
                    .cpu()
                )

                local_max = float(log_ratio.max().cpu())
                local_min = float(log_ratio.min().cpu())
                max_log_ratio = max(max_log_ratio, local_max)
                min_log_ratio = min(min_log_ratio, local_min)
                max_abs_log_ratio = max(
                    max_abs_log_ratio,
                    abs(local_max),
                    abs(local_min),
                )

                if bool((log_ratio > 80.0).any().item()):
                    kl_is_infinite = True
                elif not kl_is_infinite:
                    kl_sum += float(
                        (torch.expm1(log_ratio) - log_ratio)
                        .sum()
                        .cpu()
                    )

        if actor_transition_count == 0:
            return {
                "actor_transition_count": 0,
                "approximate_kl": 0.0,
                "clip_fraction": 0.0,
                "max_abs_log_ratio": 0.0,
                "max_log_ratio": 0.0,
                "min_log_ratio": 0.0,
            }

        return {
            "actor_transition_count": actor_transition_count,
            "approximate_kl": (
                float("inf")
                if kl_is_infinite
                else kl_sum / actor_transition_count
            ),
            "clip_fraction": clipped_count / actor_transition_count,
            "max_abs_log_ratio": max_abs_log_ratio,
            "max_log_ratio": max_log_ratio,
            "min_log_ratio": min_log_ratio,
        }

    def _policy_ratio_diagnostics(
        self,
        batch: ActPpoRolloutBatch,
        all_indices: torch.Tensor,
    ) -> _ActPpoRatioDiagnostics:
        # Re-evaluate the complete actor-enabled rollout under current policy.

        actor_transition_count = 0
        kl_sum = 0.0
        clipped_count = 0
        max_abs_log_ratio = 0.0
        analytic_kls: list[torch.Tensor] = []
        diagnostic_old_means: list[torch.Tensor] = []
        diagnostic_new_means: list[torch.Tensor] = []
        diagnostic_contexts: list[torch.Tensor] = []

        self.policy.eval()
        with torch.no_grad():
            for start in range(
                0,
                int(all_indices.numel()),
                self.config.minibatch_size,
            ):
                indices = all_indices[
                    start : start + self.config.minibatch_size
                ]
                actor_indices = indices[
                    batch.actor_update_enabled[indices]
                ]
                if not int(actor_indices.numel()):
                    continue

                means = self._actor_means(batch, actor_indices)
                exploration_log_std = self.exploration.with_offset(
                    batch.exploration_log_std_offsets[actor_indices]
                )
                new_log_prob, _ = evaluate_bounded_policy_actions(
                    means,
                    batch.policy_actions[actor_indices],
                    exploration_log_std,
                    policy_action_latents=batch.policy_action_latents[actor_indices],
                    gripper_closing_only=(
                        batch.gripper_closing_only[actor_indices]
                    ),
                    exploration_target=(
                        batch.exploration_targets[actor_indices]
                    ),
                    exploration_target_enabled=(
                        batch.exploration_target_enabled[actor_indices]
                    ),
                    exploration_target_joint_mask=(
                        batch.exploration_target_joint_masks[actor_indices]
                    ),
                )

                log_ratio = (
                    new_log_prob
                    - batch.old_log_probabilities[actor_indices]
                ).double()
                if not bool(torch.isfinite(log_ratio).all().item()):
                    raise ActPpoNumericalError(
                        "PPO ratio diagnostics produced non-finite log_ratio."
                    )

                ratio = torch.exp(log_ratio)
                if not bool(torch.isfinite(ratio).all().item()):
                    raise ActPpoNumericalError(
                        "PPO ratio diagnostics produced non-finite ratio."
                    )

                approximate_kl = (ratio - 1.0) - log_ratio
                analytic_kls.append(self._analytic_policy_kl(
                    batch, actor_indices, means, exploration_log_std,
                ).cpu())
                diagnostic_old_means.append(batch.old_action_means[actor_indices].cpu())
                diagnostic_new_means.append(means.cpu())
                diagnostic_contexts.append(batch.critic_task_context[actor_indices].cpu())
                count = int(actor_indices.numel())
                actor_transition_count += count
                kl_sum += float(approximate_kl.sum().cpu())
                clipped_count += int(
                    (
                        torch.abs(ratio - 1.0)
                        > self.config.clip_ratio
                    )
                    .sum()
                    .cpu()
                )
                max_abs_log_ratio = max(
                    max_abs_log_ratio,
                    float(torch.abs(log_ratio).max().cpu()),
                )

        if actor_transition_count == 0:
            return _ActPpoRatioDiagnostics(
                actor_transition_count=0,
                approximate_kl=0.0,
                clip_fraction=0.0,
                max_abs_log_ratio=0.0,
            )

        analytic = torch.cat(analytic_kls)
        return _ActPpoRatioDiagnostics(
            actor_transition_count=actor_transition_count,
            approximate_kl=kl_sum / actor_transition_count,
            clip_fraction=clipped_count / actor_transition_count,
            max_abs_log_ratio=max_abs_log_ratio,
            pre_clip_policy_kl=float(analytic.mean()),
            pre_clip_policy_kl_p95=float(torch.quantile(analytic, .95)),
            pre_clip_policy_kl_max=float(analytic.max()),
            phase_diagnostics=phase_policy_diagnostics(
                analytic, torch.cat(diagnostic_old_means), torch.cat(diagnostic_new_means),
                torch.cat(diagnostic_contexts),
            ),
        )

    def update(
        self,
        rollout: ActPpoRolloutBuffer,
    ) -> ActPpoUpdateMetrics:
        if len(rollout) < 2:
            raise ValueError("PPO update needs at least two transitions.")
        batch = rollout.batch(self.device)
        count = len(rollout)
        all_indices = torch.arange(count, device=self.device)

        actor_mask = batch.actor_update_enabled
        for name, values in (("bounded samples", batch.policy_action_latents),
                             ("sampled ACT means", batch.old_action_means),
                             ("sampled log_std", batch.old_action_log_stds)):
            if not bool(torch.isfinite(values[actor_mask]).all()):
                raise ValueError(f"Actor rollout requires original {name}; collect a new schema-4 rollout.")
        lower, upper = policy_action_bounds(device=self.device)
        saved_samples = batch.policy_action_latents[actor_mask]
        if bool((saved_samples.abs() > 1.0).any()):
            raise ValueError("Actor rollout contains out-of-bounds samples; collect a new schema-4 rollout.")
        expected_actions = (lower + upper) / 2 + (upper - lower) / 2 * saved_samples
        if not torch.allclose(expected_actions, batch.policy_actions[actor_mask].double(), rtol=1e-6, atol=2e-5):
            raise ValueError("Actor rollout executed actions do not match saved bounded samples.")

        pre_update = self._policy_ratio_diagnostics(
            batch,
            all_indices,
        )
        print(
            "[PPO PRECHECK] "
            f"actor={pre_update.actor_transition_count} | "
            f"policy_KL={pre_update.pre_clip_policy_kl:.8g} | "
            f"clip={pre_update.clip_fraction:.6f} | "
            f"max|log_ratio|={pre_update.max_abs_log_ratio:.8g}",
            flush=True,
        )
        if (
            pre_update.actor_transition_count
            and (
                pre_update.max_abs_log_ratio
                > _PRE_UPDATE_MAX_ABS_LOG_RATIO_TOLERANCE
                or pre_update.clip_fraction > 0.0
                or pre_update.pre_clip_policy_kl_max > 1e-6
            )
        ):
            raise RuntimeError(
                "PPO pre-update log-prob consistency check failed: "
                f"max_abs_log_ratio={pre_update.max_abs_log_ratio:.8g}, "
                f"clip_fraction={pre_update.clip_fraction:.8g}, "
                "expected the current policy to reproduce rollout "
                "old_log_probabilities before any optimizer step."
            )

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
            "actor_gradient_norm": 0.0,
            "critic_gradient_norm": 0.0,
            "exploration_gradient_norm": 0.0,
            "mean_bounds_loss": 0.0,
            "raw_mean_bounds_loss": 0.0,
            "raw_mean_out_of_bounds_fraction": 0.0,
            "raw_mean_overflow_normalized": 0.0,
            "raw_mean_bound_derivative_mean": 0.0,
            "raw_mean_bounds_diagnostic_count": 0,
        }
        minibatches = 0
        completed_ppo_epochs = 0
        early_stopped = False
        final_policy_diagnostics = pre_update
        self.policy.eval()
        self.critic.train()
        parameter_baseline = {
            group: {name: p.detach().cpu().clone() for name, p in parameters}
            for group, parameters in self._trainable_groups().items()
        }
        used_parameters = {group: set() for group in parameter_baseline}
        initial_log_std = self.exploration.value().detach().cpu().clone()

        backoff_attempts: list[dict[str, Any]] = []
        for epoch_index in range(self.config.ppo_epochs):
            epoch_snapshot = self._epoch_rollback_snapshot()
            epoch_rng = self._rng_snapshot()
            epoch_start_scale = self.policy_learning_rate_scale
            epoch_start_totals = dict(totals)
            epoch_start_minibatches = minibatches
            epoch_start_diagnostics = final_policy_diagnostics

            for attempt_index in range(self.config.kl_backoff_max_retries + 1):
                epoch_used = {group: set() for group in used_parameters}
                numerical_rejection = None
                try:
                    permutation = all_indices[
                        torch.randperm(count, device=self.device)
                    ]
                    for start in range(0, count, self.config.minibatch_size):
                        indices = permutation[
                            start : start + self.config.minibatch_size
                        ]
                        actor_mask = batch.actor_update_enabled[indices]
                        actor_indices = indices[actor_mask]
                        actor_count = int(actor_indices.numel())

                        raw_statistics = {}
                        raw_bounds_loss = torch.zeros((), device=self.device)
                        if actor_count:
                            if self.config.raw_mean_bounds_coefficient > 0.0:
                                means, raw_statistics = self._actor_means_and_raw_bounds(batch, actor_indices)
                                raw_bounds_loss = raw_statistics["raw_mean_bounds_loss"].mean()
                            else:
                                # Preserve the old forward/monkeypatch contract when
                                # disabled, without an extra pass just for diagnostics.
                                means = self._actor_means(batch, actor_indices)
                            exploration_log_std = self.exploration.with_offset(
                                batch.exploration_log_std_offsets[actor_indices]
                            )
                            new_log_prob, entropy = (
                                evaluate_bounded_policy_actions(
                                    means,
                                    batch.policy_actions[actor_indices],
                                    exploration_log_std,
                                    policy_action_latents=batch.policy_action_latents[actor_indices],
                                    gripper_closing_only=(
                                        batch.gripper_closing_only[
                                            actor_indices
                                        ]
                                    ),
                                    exploration_target=(
                                        batch.exploration_targets[
                                            actor_indices
                                        ]
                                    ),
                                    exploration_target_enabled=(
                                        batch.exploration_target_enabled[
                                            actor_indices
                                        ]
                                    ),
                                    exploration_target_joint_mask=(
                                        batch.exploration_target_joint_masks[
                                            actor_indices
                                        ]
                                    ),
                                )
                            )
                            log_ratio = (
                                new_log_prob
                                - batch.old_log_probabilities[actor_indices]
                            )
                            _require_finite(log_ratio, name="PPO log ratio")
                            ratio = log_ratio.exp()
                            _require_finite(ratio, name="PPO ratio")
                            advantages = batch.advantages[actor_indices]
                            policy_loss = -torch.minimum(
                                ratio * advantages,
                                torch.clamp(
                                    ratio,
                                    1.0 - self.config.clip_ratio,
                                    1.0 + self.config.clip_ratio,
                                )
                                * advantages,
                            ).mean()
                            entropy_mean = entropy.mean()
                            bounds_loss = mean_bounds_loss(means, lower=lower, upper=upper)
                        else:
                            policy_loss = torch.zeros(
                                (),
                                device=self.device,
                            )
                            entropy_mean = torch.zeros(
                                (),
                                device=self.device,
                            )
                            bounds_loss = torch.zeros((), device=self.device)

                        new_value = self.critic(
                            batch.critic_top_images[indices],
                            batch.critic_gripper_images[indices],
                            batch.joint_positions[indices],
                            batch.joint_velocities[indices],
                            batch.critic_task_context[indices],
                            batch.critic_policy_context[indices],
                        )
                        value_loss = F.mse_loss(
                            new_value,
                            batch.returns[indices],
                        )
                        loss = (
                            policy_loss
                            + self.config.value_loss_coefficient * value_loss
                            + self.config.mean_bounds_coefficient * bounds_loss
                            + self.config.raw_mean_bounds_coefficient * raw_bounds_loss
                            - self.config.entropy_coefficient * entropy_mean
                        )

                        _require_finite(loss, name="PPO loss")
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        for parameter in self._parameters():
                            if parameter.grad is not None:
                                _require_finite(parameter.grad, name="PPO gradient")
                        norms = {}
                        limits = {"actor": self.config.max_gradient_norm,
                                  "critic": self.config.critic_max_gradient_norm,
                                  "exploration": self.config.exploration_max_gradient_norm}
                        for group, parameters in self._trainable_groups().items():
                            active = [(name, p) for name, p in parameters if p.grad is not None]
                            epoch_used[group].update(name for name, _ in active)
                            norms[group] = torch.nn.utils.clip_grad_norm_([p for _, p in active], limits[group]).to(self.device)
                            _require_finite(norms[group], name=f"{group} gradient norm")
                        gradient_norm = torch.stack(list(norms.values())).norm()
                        _require_finite(gradient_norm, name="PPO gradient norm")
                        self.optimizer.step()
                        for parameter in self._parameters():
                            if parameter.requires_grad:
                                _require_finite(parameter, name="PPO parameter")
                        _require_finite_state(self.optimizer.state, name="PPO optimizer")
                        self.exploration.clamp_()

                        with torch.no_grad():
                            if actor_count:
                                approximate_kl = (
                                    (ratio - 1.0) - log_ratio
                                ).mean()
                                clip_fraction = (
                                    (
                                        torch.abs(ratio - 1.0)
                                        > self.config.clip_ratio
                                    )
                                    .float()
                                    .mean()
                                )
                            else:
                                approximate_kl = torch.zeros(
                                    (),
                                    device=self.device,
                                )
                                clip_fraction = torch.zeros(
                                    (),
                                    device=self.device,
                                )
                        values = {
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                            "entropy": entropy_mean,
                            "approximate_kl": approximate_kl,
                            "clip_fraction": clip_fraction,
                            "gradient_norm": gradient_norm,
                            "actor_gradient_norm": norms["actor"],
                            "critic_gradient_norm": norms["critic"],
                            "exploration_gradient_norm": norms["exploration"],
                            "mean_bounds_loss": bounds_loss,
                        }
                        for name, value in values.items():
                            totals[name] += float(value.detach().cpu())
                        if raw_statistics:
                            # Weight diagnostics by actor transition visits. The
                            # epoch snapshot also rolls these sums/count back.
                            for name, value in raw_statistics.items():
                                totals[name] += float(value.detach().sum().cpu())
                            totals["raw_mean_bounds_diagnostic_count"] += actor_count
                        minibatches += 1
                    candidate_policy_diagnostics = (
                        self._policy_ratio_diagnostics(
                            batch,
                            all_indices,
                        )
                    )
                except FloatingPointError as error:
                    numerical_rejection = str(error)
                    candidate_policy_diagnostics = _ActPpoRatioDiagnostics(
                        actor_transition_count=pre_update.actor_transition_count,
                        approximate_kl=float("inf"),
                        clip_fraction=0.0,
                        max_abs_log_ratio=float("inf"),
                        pre_clip_policy_kl=float("inf"),
                    )
                except Exception:
                    # Programming/configuration failures must still propagate, but
                    # must never leave a partially applied candidate update behind.
                    self._restore_epoch_rollback_snapshot(epoch_snapshot)
                    self._restore_rng_snapshot(epoch_rng)
                    self._set_policy_learning_rate_scale(epoch_start_scale)
                    self.policy.eval()
                    self.critic.eval()
                    raise
                candidate_kl = float(
                    candidate_policy_diagnostics.pre_clip_policy_kl
                )
                candidate_rejected = (
                    not math.isfinite(candidate_kl)
                    or candidate_kl > self.config.target_kl
                )

                backoff_attempts.append({
                    "epoch": epoch_index + 1,
                    "attempt": attempt_index + 1,
                    "policy_learning_rate_scale": self.policy_learning_rate_scale,
                    "pre_clip_policy_kl": candidate_kl if math.isfinite(candidate_kl) else None,
                    "accepted": not candidate_rejected,
                    "numerical_error": numerical_rejection,
                    "phase_diagnostics": candidate_policy_diagnostics.phase_diagnostics,
                })
                if not candidate_rejected:
                    completed_ppo_epochs += 1
                    for group in used_parameters:
                        used_parameters[group].update(epoch_used[group])
                    final_policy_diagnostics = candidate_policy_diagnostics
                    print(
                        "[PPO EPOCH] "
                        f"{epoch_index + 1}/{self.config.ppo_epochs} accepted | "
                        f"attempt={attempt_index + 1} | "
                        f"policy_KL={candidate_kl:.8g} | "
                        f"clip={final_policy_diagnostics.clip_fraction:.6f} | "
                        f"max|log_ratio|={final_policy_diagnostics.max_abs_log_ratio:.8g} | "
                        f"policy_lr_scale={self.policy_learning_rate_scale:.8g}", flush=True,
                    )
                    break

                if numerical_rejection is not None:
                    print(f"[PPO NUMERICAL REJECTION] {numerical_rejection}", flush=True)
                print(
                    f"[PPO EPOCH] {epoch_index + 1}/{self.config.ppo_epochs} rejected | "
                    f"attempt={attempt_index + 1} | KL={candidate_kl:.8g}", flush=True,
                )
                attempted_scale = self.policy_learning_rate_scale
                self._restore_epoch_rollback_snapshot(epoch_snapshot)
                self._restore_rng_snapshot(epoch_rng)
                # Keep the smallest actually attempted scale even if every trial
                # fails. A new rollout then starts with this conservative LR.
                self._set_policy_learning_rate_scale(attempted_scale)
                totals = dict(epoch_start_totals)
                minibatches = epoch_start_minibatches
                final_policy_diagnostics = epoch_start_diagnostics
                print(
                    "[PPO ROLLBACK] "
                    f"epoch={epoch_index + 1}/{self.config.ppo_epochs} | "
                    f"attempt={attempt_index + 1} | candidate_KL={candidate_kl:.8g} | "
                    f"target_KL={self.config.target_kl:.8g} | "
                    "restored policy + critic + exploration + optimizer + RNG; "
                    "rejected epoch removed.", flush=True,
                )
                next_scale = max(
                    self.config.kl_backoff_min_scale,
                    attempted_scale * self.config.kl_backoff_factor,
                )
                can_retry = (
                    numerical_rejection is None
                    and attempt_index < self.config.kl_backoff_max_retries
                    and next_scale < attempted_scale
                )
                if can_retry:
                    self._set_policy_learning_rate_scale(next_scale)
                    print(
                        "[PPO BACKOFF] "
                        f"retry={attempt_index + 1}/{self.config.kl_backoff_max_retries} | "
                        f"policy_lr_scale={next_scale:.8g} | "
                        "same old rollout and shuffle; critic LR unchanged.", flush=True,
                    )
                    continue
                early_stopped = True
                print(
                    "[PPO STOP] rejected epoch rolled back; remaining epochs skipped. "
                    f"policy_lr_scale={attempted_scale:.8g}; "
                    f"at_min_scale={attempted_scale <= self.config.kl_backoff_min_scale}.",
                    flush=True,
                )
                break
            if early_stopped:
                break

        self.policy.eval()
        self.critic.eval()
        self.update_index += 1
        self.total_environment_steps += count
        return ActPpoUpdateMetrics(
            transition_count=count,
            minibatch_count=minibatches,
            configured_ppo_epochs=self.config.ppo_epochs,
            completed_ppo_epochs=completed_ppo_epochs,
            early_stopped=early_stopped,
            target_kl=self.config.target_kl,
            pre_update_approximate_kl=(
                pre_update.approximate_kl
            ),
            pre_update_clip_fraction=(
                pre_update.clip_fraction
            ),
            pre_update_max_abs_log_ratio=(
                pre_update.max_abs_log_ratio
            ),
            final_policy_approximate_kl=(
                final_policy_diagnostics.approximate_kl
            ),
            final_policy_clip_fraction=(
                final_policy_diagnostics.clip_fraction
            ),
            final_policy_max_abs_log_ratio=(
                final_policy_diagnostics.max_abs_log_ratio
            ),
            **{
                name: (
                    int(value) if name == "raw_mean_bounds_diagnostic_count"
                    else value / max(1, totals["raw_mean_bounds_diagnostic_count"])
                    if name.startswith("raw_mean_")
                    else value / minibatches if minibatches else 0.0
                )
                for name, value in totals.items()
            },
            actor_learning_rate=self.actor_learning_rate,
            actor_backbone_learning_rate=(
                self.actor_backbone_learning_rate
            ),
            critic_learning_rate=self.config.critic_learning_rate,
            exploration_learning_rate=self.config.exploration_learning_rate * self.policy_learning_rate_scale,
            kl_backoff_attempt_count=len(backoff_attempts),
            kl_backoff_rejection_count=sum(not item["accepted"] for item in backoff_attempts),
            kl_backoff_retry_count=sum(item["attempt"] > 1 for item in backoff_attempts),
            policy_learning_rate_scale=self.policy_learning_rate_scale,
            kl_backoff_at_min_scale=self.policy_learning_rate_scale <= self.config.kl_backoff_min_scale,
            kl_backoff_attempts=tuple(backoff_attempts),
            phase_diagnostics=final_policy_diagnostics.phase_diagnostics,
            advantage_normalization=batch.advantage_normalization,
            pre_update_pre_clip_policy_kl=pre_update.pre_clip_policy_kl,
            pre_clip_policy_kl=final_policy_diagnostics.pre_clip_policy_kl,
            pre_clip_policy_kl_p95=final_policy_diagnostics.pre_clip_policy_kl_p95,
            pre_clip_policy_kl_max=final_policy_diagnostics.pre_clip_policy_kl_max,
            **self._parameter_update_diagnostics(parameter_baseline, used_parameters),
            exploration_log_std=tuple(self.exploration.value().detach().cpu().tolist()),
            exploration_log_std_delta=tuple((self.exploration.value().detach().cpu() - initial_log_std).tolist()),
            effective_exploration_log_std_upper_bounds=tuple(self.exploration._maximum().detach().cpu().tolist()),
            log_std_mean=float(
                self.exploration.value().detach().mean().cpu()
            ),
        )

    def checkpoint_payload(
        self,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": ACT_PPO_CHECKPOINT_SCHEMA_VERSION,
            "config": asdict(self.config),
            "execution_signature": self.execution_signature,
            "execution_signature_sha256": self.execution_signature_sha256,
            "act_policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "exploration_state_dict": self.exploration.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_parameter_names": self._optimizer_parameter_names(),
            "learning_rate_state": self._learning_rate_state(),
            "update_index": self.update_index,
            "total_environment_steps": self.total_environment_steps,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if extra is not None:
            payload["extra"] = extra
        return payload

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(
                self.checkpoint_payload(extra=extra),
                temporary,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: Any = None,
    ) -> dict[str, Any]:
        payload = torch.load(
            Path(path).expanduser().resolve(),
            map_location=map_location,
            weights_only=False,
        )
        schema_version = payload.get("schema_version")
        if schema_version != ACT_PPO_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported ACT PPO checkpoint schema: the action distribution changed. "
                "Use ACT weights-only warm-start and collect a new rollout; "
                "legacy tanh or execution-clipped Gaussian states cannot be resumed."
            )
        saved_signature = payload.get("execution_signature")
        if (
            not isinstance(saved_signature, dict)
            or saved_signature != self._execution_signature
            or payload.get("execution_signature_sha256") != _contract_hash(saved_signature)
        ):
            raise ValueError(
                "ACT PPO checkpoint execution signature does not match; "
                "use ACT weights-only warm-start."
            )
        saved_config = payload.get("config")
        current_config = asdict(self.config)
        comparable_config = dict(saved_config) if isinstance(saved_config, dict) else None
        if comparable_config is not None:
            # Older global-normalization checkpoints predate this optional
            # field. A grouped checkpoint still requires its exact signature.
            comparable_config.setdefault("advantage_group_context_index", None)
            comparable_config.setdefault("advantage_group_count", 2)
            comparable_config.setdefault("raw_mean_bounds_coefficient", 0.0)
        # Guard controls may change on resume; architecture and learning-rate
        # bases remain strict; older training states require a weights-only start.
        override_fields = ("target_kl", "kl_backoff_max_retries", "kl_backoff_factor", "kl_backoff_min_scale",
                           "raw_mean_bounds_coefficient")
        if comparable_config is None or {
            **comparable_config, **{key: current_config[key] for key in override_fields}
        } != current_config:
            raise ValueError("ACT PPO checkpoint config does not match.")
        defaults = asdict(ActPpoConfig())
        for name in override_fields:
            previous = saved_config.get(name, defaults[name])
            if previous != current_config[name]:
                print(
                    f"[PPO RESUME] overriding checkpoint {name}={previous} "
                    f"with current {name}={current_config[name]}."
                    + (" Auxiliary raw-action bounds objective changed; policy distribution and optimizer state preserved."
                       if name == "raw_mean_bounds_coefficient" else ""), flush=True,
                )
        if payload.get("optimizer_parameter_names") != self._optimizer_parameter_names():
            raise ValueError("ACT PPO checkpoint optimizer parameter names do not match.")
        saved_lr_scale = self._checkpoint_policy_learning_rate_scale(payload)
        self.policy.load_state_dict(payload["act_policy_state_dict"])
        self.critic.load_state_dict(payload["critic_state_dict"])
        self.exploration.load_state_dict(
            payload["exploration_state_dict"]
        )
        self._load_checkpoint_optimizer(
            payload["optimizer_state_dict"], payload.get("optimizer_parameter_names")
        )
        self._set_policy_learning_rate_scale(saved_lr_scale)
        self.update_index = int(payload["update_index"])
        self.total_environment_steps = int(
            payload["total_environment_steps"]
        )
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        if (
            torch.cuda.is_available()
            and "cuda_rng_state_all" in payload
        ):
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in payload["cuda_rng_state_all"]]
            )
        self.policy.eval()
        self.critic.eval()
        return payload

    def _optimizer_parameter_names(self) -> list[list[str]]:
        names = {
            id(parameter): f"{prefix}.{name}"
            for prefix, module in (
                ("policy", self.policy), ("critic", self.critic), ("exploration", self.exploration)
            )
            for name, parameter in module.named_parameters()
        }
        return [[names[id(p)] for p in group["params"]] for group in self.optimizer.param_groups]

    def _load_checkpoint_optimizer(
        self, state: dict[str, Any], parameter_names: list[list[str]] | None = None
    ) -> None:
        """Only schema-4 named optimizer groups can resume this distribution."""
        groups = state["param_groups"]
        if parameter_names != self._optimizer_parameter_names():
            raise ValueError("ACT PPO checkpoint optimizer parameter names do not match.")
        if len(groups) != len(self.optimizer.param_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(groups, self.optimizer.param_groups, strict=True)
        ):
            raise ValueError("ACT PPO checkpoint optimizer groups do not match.")
        self.optimizer.load_state_dict(state)
