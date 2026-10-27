"""Bounded exploration in affine-normalized robot action coordinates.

The actor supplies a bounded Gaussian location; exploration is conditioned on
the fixed interval [-1, 1] in every joint. No out-of-range Gaussian tail is
folded into an execution endpoint. Collection and PPO replay use the same
truncated density, entropy and exact KL. Log densities use normalized action
coordinates; the constant affine Jacobian cancels from PPO ratios and KL.
Probability arithmetic is float64, retained samples use the actor dtype.
"""

from __future__ import annotations

import math

import torch


ACTION_DIM = 6
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SQRT_TWO = math.sqrt(2.0)


def _finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor.")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} must contain only finite values.")


def _means_tensor(means: torch.Tensor, *, name: str) -> None:
    _finite_tensor(means, name=name)
    if means.ndim < 1 or means.shape[-1] != ACTION_DIM:
        raise ValueError(f"{name} must end in dimension {ACTION_DIM}.")


def _broadcast_like(
    value: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _finite_tensor(value, name=name)
    converted = value.to(device=reference.device, dtype=dtype or reference.dtype)
    _finite_tensor(converted, name=name)
    try:
        return torch.broadcast_to(converted, reference.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} must broadcast to shape {tuple(reference.shape)}.") from error


def _coordinates(
    means: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _means_tensor(means, name="Action means")
    lower = _broadcast_like(lower, means, name="Action lower bounds")
    upper = _broadcast_like(upper, means, name="Action upper bounds")
    if not bool((upper > lower).all()):
        raise ValueError("Every action upper bound must exceed its lower bound.")
    centre = (upper + lower) / 2.0
    scale = (upper - lower) / 2.0
    _finite_tensor(centre, name="Action coordinate centre")
    _finite_tensor(scale, name="Action coordinate scale")
    if not bool((scale > 0.0).all()):
        raise ValueError("Every action coordinate scale must be positive.")
    return centre, scale


def normalized_action_means(
    means: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Affine coordinates, also usable to diagnose legacy unbounded means."""
    centre, scale = _coordinates(means, lower=lower, upper=upper)
    normalized = (means - centre) / scale
    _finite_tensor(normalized, name="Normalized action means")
    return normalized


def _log_scales(log_std: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    log_scales = _broadcast_like(
        log_std, means, name="Exploration log standard deviation", dtype=torch.float64
    )
    scales = log_scales.exp()
    _finite_tensor(scales, name="Exploration standard deviation")
    if not bool((scales > 0.0).all()):
        raise FloatingPointError("Exploration standard deviation must be positive.")
    return log_scales


def _bounded_locations(means: torch.Tensor, *, name: str) -> None:
    _means_tensor(means, name=name)
    if not bool((means.abs() <= 1.0).all()):
        raise ValueError(f"{name} must lie in [-1, 1]; bound the actor output before exploration.")


def _truncation(
    normalized_means: torch.Tensor, log_std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _bounded_locations(normalized_means, name="Normalized action means")
    means = normalized_means.double()
    logs = _log_scales(log_std, means)
    inverse_scales = (-logs).exp()
    _finite_tensor(inverse_scales, name="Inverse exploration standard deviation")
    alpha = (-1.0 - means) * inverse_scales
    beta = (1.0 - means) * inverse_scales
    erf_lower, erf_upper = torch.erf(alpha / _SQRT_TWO), torch.erf(beta / _SQRT_TWO)
    # alpha <= 0 <= beta, so this avoids cancellation between CDFs near 0.5.
    normalizer = 0.5 * (erf_upper - erf_lower)
    if not bool((normalizer > 0.0).all()):
        raise FloatingPointError("Truncated Gaussian normalizer must be positive.")
    return means, logs, alpha, beta, normalizer.log(), normalizer


def _standardized_moments(
    alpha: torch.Tensor, beta: torch.Tensor, normalizer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """E[z], E[z²], with a cancellation-free series for broad Gaussians."""
    phi_alpha = torch.exp(-0.5 * alpha.square() - 0.5 * _LOG_TWO_PI)
    phi_beta = torch.exp(-0.5 * beta.square() - 0.5 * _LOG_TWO_PI)
    first = (phi_alpha - phi_beta) / normalizer
    second = 1.0 + (alpha * phi_alpha - beta * phi_beta) / normalizer
    broad = torch.maximum(alpha.abs(), beta.abs()) <= 0.5
    if bool(broad.any()):
        a, b = alpha[broad], beta[broad]
        integrals = []
        # Integrate the Gaussian's convergent power series. On |z| <= 0.5,
        # the omitted terms are below float64 precision. Unlike 1+(aφa-bφb)/Z,
        # this does not subtract nearly equal numbers for the second moment.
        for order in (1, 2):
            integral = torch.zeros_like(a)
            coefficient = 1.0
            for k in range(12):
                power = 2 * k + order + 1
                integral = integral + coefficient * (b.pow(power) - a.pow(power)) / power
                coefficient *= -0.5 / (k + 1)
            integrals.append(integral * math.exp(-0.5 * _LOG_TWO_PI) / normalizer[broad])
        first, second = first.clone(), second.clone()
        first[broad], second[broad] = integrals
    return first, second


def _evaluate_normalized(
    normalized_means: torch.Tensor,
    log_std: torch.Tensor,
    policy_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _finite_tensor(policy_samples, name="Normalized policy samples")
    if policy_samples.shape != normalized_means.shape:
        raise ValueError("Normalized policy samples and action means must have identical shapes.")
    means, logs, alpha, beta, log_z, z = _truncation(normalized_means, log_std)
    samples = policy_samples.to(device=means.device, dtype=torch.float64)
    residual = (samples - means) * (-logs).exp()
    finite_log_prob = (-0.5 * residual.square() - logs - 0.5 * _LOG_TWO_PI - log_z).sum(-1)
    _finite_tensor(finite_log_prob, name="Policy log probability on finite samples")
    # An action outside the fixed support is impossible for every mean.
    log_prob = torch.where((samples.abs() <= 1.0).all(-1), finite_log_prob, -torch.inf)
    _, second = _standardized_moments(alpha, beta, z)
    entropy = (logs + 0.5 * _LOG_TWO_PI + log_z + 0.5 * second).sum(-1)
    _finite_tensor(entropy, name="Truncated Gaussian entropy")
    return log_prob, entropy


def sample_policy_actions(
    means: torch.Tensor,
    log_std: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bounded actions, log probability and retained normalized samples.

    Inverse-CDF sampling uses only mass inside the legal interval. Float32
    samples may round to an endpoint at tiny scales, but no outside tail is
    folded into it. Detach samples for fixed-action PPO replay. Deterministic
    evaluation returns the bounded Gaussian location exactly; near a bound
    this differs from the truncated distribution's expectation.
    """
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a bool.")
    centre, scale = _coordinates(means, lower=lower, upper=upper)
    normalized_means = (means - centre) / scale
    means64, logs, alpha, beta, _, _ = _truncation(normalized_means, log_std)
    if deterministic:
        samples = normalized_means
        actions = means
    else:
        erf_lower, erf_upper = torch.erf(alpha / _SQRT_TWO), torch.erf(beta / _SQRT_TWO)
        uniform = torch.rand_like(means64)
        # Avoid infinite erfinv at a floating-point endpoint, changing only
        # machine-precision tails of the probability interval.
        epsilon = torch.finfo(torch.float64).eps
        erf_sample = (erf_lower + uniform * (erf_upper - erf_lower)).clamp(-1.0 + epsilon, 1.0 - epsilon)
        sampled64 = means64 + logs.exp() * _SQRT_TWO * torch.erfinv(erf_sample)
        # Correct inverse-CDF numerical overshoot only, never an outside draw
        # from an unbounded Gaussian. Replay scores the final stored dtype.
        samples = sampled64.clamp(-1.0, 1.0).to(dtype=means.dtype)
        actions = centre + scale * samples
    _finite_tensor(actions, name="Executed policy actions")
    log_prob, _ = _evaluate_normalized(normalized_means, log_std, samples)
    return actions, log_prob, samples


def evaluate_policy_actions(
    means: torch.Tensor,
    log_std: torch.Tensor,
    policy_samples: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score fixed replay samples; outside support has log probability -inf."""
    normalized_means = normalized_action_means(means, lower=lower, upper=upper)
    return _evaluate_normalized(normalized_means, log_std, policy_samples)


def truncated_gaussian_kl(
    old_means_normalized: torch.Tensor,
    old_log_std: torch.Tensor,
    new_means_normalized: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """Exact KL(old || new), using old truncated first and second moments.

    Both distributions share fixed support [-1, 1]. Float64 and expm1 preserve
    an exact zero at identity and reduce cancellation for nearby scales. Tiny
    negative values from roundoff are rounded to zero after summing joints.
    """
    _bounded_locations(old_means_normalized, name="Old normalized action means")
    _bounded_locations(new_means_normalized, name="New normalized action means")
    if old_means_normalized.shape != new_means_normalized.shape:
        raise ValueError("Old and new normalized action means must have identical shapes.")
    old = old_means_normalized.to(device=new_means_normalized.device, dtype=torch.float64)
    old_means, old_logs, alpha, beta, old_log_z, old_z = _truncation(old, old_log_std)
    new_means, new_logs, _, _, new_log_z, _ = _truncation(new_means_normalized, new_log_std)
    first, second = _standardized_moments(alpha, beta, old_z)
    log_ratio = old_logs - new_logs
    delta = (old_means - new_means) * (-new_logs).exp()
    terms = -log_ratio + new_log_z - old_log_z + 0.5 * (
        torch.expm1(2.0 * log_ratio) * second
        + 2.0 * log_ratio.exp() * delta * first
        + delta.square()
    )
    kl = terms.sum(-1)
    _finite_tensor(kl, name="Truncated Gaussian policy KL")
    return kl.clamp_min(0.0)


def gaussian_kl(
    old_means_normalized: torch.Tensor,
    old_log_std: torch.Tensor,
    new_means_normalized: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    """Legacy Gaussian KL; this is not the bounded policy's update metric."""
    _means_tensor(old_means_normalized, name="Old normalized action means")
    _means_tensor(new_means_normalized, name="New normalized action means")
    if old_means_normalized.shape != new_means_normalized.shape:
        raise ValueError("Old and new normalized action means must have identical shapes.")
    old_means = old_means_normalized.to(device=new_means_normalized.device, dtype=torch.float64)
    new_means = new_means_normalized.double()
    old_logs = _log_scales(old_log_std, new_means)
    new_logs = _log_scales(new_log_std, new_means)
    log_ratio = old_logs - new_logs
    standardized_difference = (old_means - new_means) * (-new_logs).exp()
    # expm1 avoids cancellation of the variance term for nearly equal scales.
    terms = 0.5 * (
        torch.expm1(2.0 * log_ratio) - 2.0 * log_ratio + standardized_difference.square()
    )
    kl = terms.sum(dim=-1)
    _finite_tensor(kl, name="Gaussian policy KL")
    return kl


def mean_bounds_loss(
    means: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Legacy overflow diagnostic; structurally bounded actor means give zero."""
    normalized = normalized_action_means(means, lower=lower, upper=upper)
    loss = torch.relu(normalized.abs() - 1.0).square().mean()
    _finite_tensor(loss, name="Action mean bounds loss")
    return loss
