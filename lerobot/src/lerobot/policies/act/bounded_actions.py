"""Checkpoint-controlled, differentiable bounds for ACT action predictions."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lerobot.policies.act.configuration_act import ACTConfig


def _validated_limits(actions: Tensor, config: ACTConfig) -> tuple[list[float], list[float]] | None:
    """Also accept small policy config objects used by offline actor tests."""
    lower = getattr(config, "action_bounds_lower", None)
    upper = getattr(config, "action_bounds_upper", None)
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError("Both ACT action bounds must be supplied together.")
    if not lower or len(lower) != len(upper):
        raise ValueError("ACT action bounds must be nonempty and have the same length.")
    if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in zip(lower, upper)):
        raise ValueError("ACT action bounds must be finite and strictly increasing.")
    softness = float(getattr(config, "action_bound_softness", 0.02))
    if not math.isfinite(softness) or not 0 < softness < 1:
        raise ValueError("ACT action bound softness must be finite and between 0 and 1.")
    if not actions.is_floating_point() or actions.ndim < 1:
        raise ValueError("Bounded ACT actions must be a floating point tensor with an action dimension.")
    if actions.shape[-1] != len(lower):
        raise ValueError("ACT action dimension does not match the configured action bounds.")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("Bounded ACT actions must be finite.")
    return lower, upper


def clamp_action_mean_roundoff(actions: Tensor, config: ACTConfig) -> Tensor:
    """Guard arithmetic roundoff after averaging already-bounded action chunks.

    This is deliberately separate from the soft bound, which must only be
    applied once to each raw ACT prediction. Legacy checkpoint actions pass
    through unchanged.
    """
    limits = _validated_limits(actions, config)
    if limits is None:
        return actions
    lower, upper = (torch.as_tensor(values, device=actions.device, dtype=actions.dtype) for values in limits)
    return torch.maximum(torch.minimum(actions, upper), lower)


def bound_action_means(actions: Tensor, config: ACTConfig) -> Tensor:
    """Bound physical action predictions while preserving the interior exactly.

    In affine [-1, 1] coordinates the central ``1 - softness`` region is the
    identity. Outside it a C1 rational tail approaches the limit, with a nonzero
    analytic derivative at every finite input. Unlike clamping a network output,
    this leaves an inward gradient when a historical network predicts > 100.
    Bounds are opt-in so historical checkpoint inference is unchanged.
    """
    limits = _validated_limits(actions, config)
    if limits is None:
        return actions

    # Compute the small tail correction in float64. Preserve the input dtype and
    # the original interior values, including their exact floating point bits.
    lower, upper = (torch.as_tensor(values, device=actions.device, dtype=torch.float64) for values in limits)
    center = lower / 2 + upper / 2
    half_range = upper / 2 - lower / 2
    normalized = (actions.to(torch.float64) - center) / half_range
    softness = float(getattr(config, "action_bound_softness", 0.02))
    excess = (normalized.abs() - (1 - softness)).clamp_min(0)
    # Equivalent to 1-s + s*d/(s+d), without a potentially overflowing s*d.
    tail = normalized.sign() * (1 - softness * (softness / (softness + excess)))
    bounded = (center + half_range * tail).to(actions.dtype)
    # This only guards arithmetic roundoff after conversion; it never clamps the
    # raw network prediction or removes the tail's gradient at ordinary inputs.
    bounded = torch.maximum(torch.minimum(bounded, upper.to(actions.dtype)), lower.to(actions.dtype))
    return torch.where(normalized.abs() <= 1 - softness, actions, bounded)
