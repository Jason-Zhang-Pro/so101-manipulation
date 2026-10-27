import math

import numpy as np
import pytest
import torch

from so101_capstone.training.act_ppo_action_distribution import (
    evaluate_policy_actions,
    gaussian_kl,
    mean_bounds_loss,
    normalized_action_means,
    sample_policy_actions,
    truncated_gaussian_kl,
)


@pytest.fixture
def bounds():
    return {
        "lower": torch.tensor([-100.0] * 5 + [0.0]),
        "upper": torch.tensor([100.0] * 6),
    }


def _reference(mean, log_std):
    sigma = math.exp(log_std)
    alpha, beta = (-1 - mean) / sigma, (1 - mean) / sigma
    z = 0.5 * (math.erf(beta / math.sqrt(2)) - math.erf(alpha / math.sqrt(2)))
    return mean, sigma, math.log(z), alpha, beta


def _reference_logpdf(reference, samples):
    mean, sigma, log_z, _, _ = reference
    return -0.5 * ((samples - mean) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi) - log_z


def _integrate_reference(reference, function):
    mean, sigma, log_z, alpha, beta = reference
    # Integrate in standardized coordinates, so even a narrow endpoint law
    # is resolved. Omitted tails beyond 12 sigma are below 4e-33.
    low, high = max(alpha, -12), min(beta, 12)
    nodes, weights = np.polynomial.legendre.leggauss(160)
    standardized = (high + low) / 2 + (high - low) / 2 * nodes
    density = np.exp(-0.5 * standardized**2 - 0.5 * math.log(2 * math.pi) - log_z)
    return np.sum(weights * density * function(mean + sigma * standardized)) * (high - low) / 2


def test_sampling_replays_exact_stored_likelihood(bounds):
    torch.manual_seed(73)
    means = torch.tensor([99.0, -99.0, 0.0, 100.0, -100.0, 0.0]).expand(2, 3, 6)
    logs = torch.full((2, 1, 6), -3.5)
    with torch.no_grad():
        actions, old_log_prob, samples = sample_policy_actions(means, logs, **bounds)
    replayed, entropy = evaluate_policy_actions(means, logs, samples, **bounds)

    assert actions.shape == samples.shape == (2, 3, 6)
    assert replayed.shape == entropy.shape == (2, 3)
    assert samples.dtype == means.dtype
    assert replayed.dtype == entropy.dtype == torch.float64
    assert torch.equal(old_log_prob, replayed)
    assert torch.equal(torch.exp(replayed - old_log_prob), torch.ones_like(replayed))
    assert bool((actions >= bounds["lower"]).all())
    assert bool((actions <= bounds["upper"]).all())
    assert bool((samples.abs() <= 1.0).all())


@pytest.mark.parametrize("log_std", [-9.0, -4.0, -1.5, 0.0, 2.0, 6.0])
def test_likelihood_entropy_match_quadrature_including_bounds_and_broad_scale(bounds, log_std):
    means = torch.tensor([[100.0, -100.0, 0.0, 24.0, -99.0, 42.0]], dtype=torch.float64)
    logs = torch.full((6,), log_std, dtype=torch.float64)
    normalized = normalized_action_means(means, **bounds)
    samples = (normalized + torch.tensor([-0.0001, 0.0001, 0.0001, -0.0001, 0.0001, 0.0001])).clamp(-1, 1)
    references = [_reference(float(mu), log_std) for mu in normalized[0]]
    expected_logp = sum(_reference_logpdf(r, float(x)) for r, x in zip(references, samples[0], strict=True))
    expected_entropy = sum(_integrate_reference(r, lambda x: -_reference_logpdf(r, x)) for r in references)
    log_prob, entropy = evaluate_policy_actions(means, logs, samples, **bounds)

    assert float(log_prob) == pytest.approx(expected_logp, rel=1e-10, abs=1e-10)
    assert float(entropy) == pytest.approx(expected_entropy, rel=1e-10, abs=1e-10)


def test_outside_support_has_zero_probability_after_mean_update(bounds):
    means = torch.tensor([[99.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
    logs = torch.full((6,), -2.0)
    samples = torch.tensor([[1.0001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    old_logp, _ = evaluate_policy_actions(means, logs, samples, **bounds)
    new_logp, _ = evaluate_policy_actions(means - 1, logs, samples, **bounds)
    assert torch.isneginf(old_logp).all()
    assert torch.isneginf(new_logp).all()


@pytest.mark.parametrize("log_std", [-9.0, -4.0, -1.5])
def test_endpoint_exploration_is_inward_without_clipping_atom(bounds, log_std):
    torch.manual_seed(42)
    count = 8192
    means = torch.tensor([100.0, -100.0, 100.0, -100.0, 100.0, 0.0]).expand(count, 6)
    actions, _, samples = sample_policy_actions(means, torch.full((6,), log_std), **bounds)
    inward = torch.stack(
        [actions[:, 0] < 100.0, actions[:, 1] > -100.0, actions[:, 5] > 0.0], dim=-1
    ).float().mean(dim=0)
    # Float32 can round a few tiny-sigma samples to a boundary. There is no
    # 50% clipping atom: essentially every sampled movement goes inward.
    assert bool((inward > 0.999).all())
    assert bool((samples.abs() <= 1).all())
    sigma = math.exp(log_std)
    expected_std = sigma * math.sqrt(1 - 2 / math.pi)
    assert float(samples[:, 0].std()) == pytest.approx(expected_std, rel=0.04)
    assert float(samples[:, 0].mean()) == pytest.approx(1 - sigma * math.sqrt(2 / math.pi), abs=0.04 * sigma)


def test_deterministic_execution_preserves_bounded_location_exactly(bounds):
    means = torch.tensor([[-100.0, 95.10715, 0.1234567, 37.12345, 100.0, 0.0]])
    actions, _, samples = sample_policy_actions(means, torch.full((6,), -4.0), deterministic=True, **bounds)
    assert torch.equal(actions, means)
    assert torch.equal(samples, normalized_action_means(means, **bounds))


@pytest.mark.parametrize("joint,mean_value", [(0, 99.0), (1, -99.0), (5, 1.0), (5, 99.0)])
def test_location_and_scale_likelihood_gradients_match_finite_difference(bounds, joint, mean_value):
    means = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]], dtype=torch.float64)
    means[0, joint] = mean_value
    means.requires_grad_()
    samples = (normalized_action_means(means.detach(), **bounds) * 0.98 + 0.005).detach()
    logs = torch.full((6,), -3.0, dtype=torch.float64, requires_grad=True)
    log_prob, entropy = evaluate_policy_actions(means, logs, samples, **bounds)
    loss = log_prob.sum() + 0.01 * entropy.sum()
    mean_gradient, log_gradient = torch.autograd.grad(loss, (means, logs))
    step = 1e-5
    for tensor, expected in ((means, mean_gradient[0, joint]), (logs, log_gradient[joint])):
        positive, negative = tensor.detach().clone(), tensor.detach().clone()
        positive.reshape(-1)[joint] += step
        negative.reshape(-1)[joint] -= step
        high_args = (positive, logs) if tensor is means else (means, positive)
        low_args = (negative, logs) if tensor is means else (means, negative)
        high = evaluate_policy_actions(*high_args, samples, **bounds)
        low = evaluate_policy_actions(*low_args, samples, **bounds)
        numeric = ((high[0] + 0.01 * high[1]) - (low[0] + 0.01 * low[1])).sum() / (2 * step)
        assert abs(float(expected)) > 1e-5
        torch.testing.assert_close(expected, numeric, rtol=2e-7, atol=1e-8)


def test_float32_endpoint_locations_receive_finite_likelihood_gradient(bounds):
    means = torch.tensor([[100.0, -100.0, 100.0, -100.0, 0.0, 0.0]], requires_grad=True)
    samples = (normalized_action_means(means.detach(), **bounds) * 0.99 + 0.002).clamp(-1, 1)
    log_prob, _ = evaluate_policy_actions(means, torch.full((6,), -4.0), samples, **bounds)
    log_prob.sum().backward()
    assert means.grad.dtype == torch.float32
    assert bool(torch.isfinite(means.grad).all())
    assert bool((means.grad.abs() > 0).all())


def test_bounds_loss_keeps_legacy_diagnostic_and_bounded_means_have_zero(bounds):
    legal = torch.tensor([[-100.0, 100.0, 0.0, -20.0, 40.0, 0.0]], requires_grad=True)
    loss = mean_bounds_loss(legal, **bounds)
    loss.backward()
    assert float(loss) == 0
    assert torch.equal(legal.grad, torch.zeros_like(legal))
    invalid = torch.tensor([[110.0, 0.0, 0.0, 0.0, 0.0, 105.0]], dtype=torch.float64)
    assert float(mean_bounds_loss(invalid, **bounds)) == pytest.approx(2 * 0.1**2 / 6)


@pytest.mark.parametrize("old_log,new_log", [(-4.0, -3.8), (-1.5, -1.8), (0.0, 0.3), (2.0, 2.2)])
def test_truncated_kl_matches_numerical_integration(old_log, new_log):
    old_means = torch.tensor([[1.0, -1.0, 0.0, 0.3, -0.99, 0.98]], dtype=torch.float64)
    new_means = old_means * 0.985 + 0.001
    old_logs = torch.full((6,), old_log, dtype=torch.float64)
    new_logs = torch.full((6,), new_log, dtype=torch.float64)
    expected = 0
    for old_mu, new_mu in zip(old_means[0], new_means[0], strict=True):
        old, new = _reference(float(old_mu), old_log), _reference(float(new_mu), new_log)
        expected += _integrate_reference(old, lambda x: _reference_logpdf(old, x) - _reference_logpdf(new, x))
    actual = truncated_gaussian_kl(old_means, old_logs, new_means, new_logs)
    assert float(actual) == pytest.approx(expected, rel=2e-8, abs=1e-10)
    assert not torch.allclose(actual, truncated_gaussian_kl(new_means, new_logs, old_means, old_logs))


def test_truncated_kl_zero_identity_and_stable_near_identity():
    means = torch.tensor([[1., -1., 0., .98, -.98, .5]], dtype=torch.float64)
    logs = torch.full((6,), -4.0, dtype=torch.float64)
    assert torch.equal(truncated_gaussian_kl(means, logs, means, logs), torch.zeros(1, dtype=torch.float64))
    near = truncated_gaussian_kl(means, logs, means * (1 - 1e-7), logs + 1e-7)
    assert float(near) > 0
    assert float(near) < 1e-8
    interior = torch.zeros(3, 6, dtype=torch.float64)
    shifted = truncated_gaussian_kl(interior, logs, interior + 1e-7, logs)
    expected = 6 * 1e-14 / (2 * math.exp(-8))
    torch.testing.assert_close(shifted, torch.full((3,), expected, dtype=torch.float64), rtol=1e-12, atol=0)


def test_truncated_kl_matches_monte_carlo_log_ratio(bounds):
    torch.manual_seed(119)
    count = 40000
    old_means = torch.tensor([100.0, -100.0, 0.0, 50.0, -50.0, 0.0]).expand(count, 6)
    new_means = old_means + torch.tensor([-0.5, 0.2, -0.3, 0.0, -0.2, 0.4])
    old_logs = torch.full((6,), -3.0)
    new_logs = torch.full((6,), -2.9)
    with torch.no_grad():
        _, old_log_prob, samples = sample_policy_actions(old_means, old_logs, **bounds)
        new_log_prob, _ = evaluate_policy_actions(new_means, new_logs, samples, **bounds)
        exact = truncated_gaussian_kl(
            normalized_action_means(old_means[:1], **bounds), old_logs,
            normalized_action_means(new_means[:1], **bounds), new_logs,
        ).item()
    log_ratio = old_log_prob - new_log_prob
    standard_error = float(log_ratio.std()) / math.sqrt(count)
    assert abs(float(log_ratio.mean()) - exact) < 5 * standard_error


def test_kl_and_entropy_gradients_finite_for_broad_and_small_scales(bounds):
    for log_std in (-9.0, -1.5, 2.0, 6.0):
        old = torch.tensor([[1., -1., 0., .98, -.98, .5]], dtype=torch.float64)
        new = (old * 0.99).requires_grad_()
        logs = torch.full((6,), log_std, dtype=torch.float64, requires_grad=True)
        kl = truncated_gaussian_kl(old, logs.detach(), new, logs)
        kl.sum().backward()
        assert torch.isfinite(new.grad).all()
        assert torch.isfinite(logs.grad).all()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_means_and_samples_rejected(bounds, invalid):
    means = torch.zeros(1, 6)
    invalid_tensor = means.clone()
    invalid_tensor[0, 0] = invalid
    with pytest.raises(FloatingPointError):
        sample_policy_actions(invalid_tensor, torch.zeros(6), **bounds)
    with pytest.raises(FloatingPointError):
        evaluate_policy_actions(means, torch.zeros(6), invalid_tensor, **bounds)


@pytest.mark.parametrize("log_std", [float("nan"), float("inf"), 1000.0, -1000.0])
def test_invalid_exploration_scale_rejected(bounds, log_std):
    with pytest.raises(FloatingPointError):
        sample_policy_actions(torch.zeros(6), torch.full((6,), log_std), **bounds)


def test_unbounded_location_rejected_in_sampling_scoring_and_kl(bounds):
    invalid = torch.tensor([[101., 0., 0., 0., 0., 50.]])
    with pytest.raises(ValueError, match="bound the actor"):
        sample_policy_actions(invalid, torch.zeros(6), **bounds)
    with pytest.raises(ValueError, match="bound the actor"):
        evaluate_policy_actions(invalid, torch.zeros(6), torch.zeros_like(invalid), **bounds)
    with pytest.raises(ValueError, match="bound the actor"):
        truncated_gaussian_kl(torch.full((6,), 1.001), torch.zeros(6), torch.zeros(6), torch.zeros(6))


def test_invalid_shapes_and_bounds_rejected(bounds):
    with pytest.raises(ValueError, match="dimension 6"):
        sample_policy_actions(torch.zeros(2, 5), torch.zeros(6), **bounds)
    with pytest.raises(ValueError, match="broadcast"):
        sample_policy_actions(torch.zeros(2, 6), torch.zeros(3, 6), **bounds)
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_policy_actions(torch.zeros(2, 6), torch.zeros(6), torch.zeros(6), **bounds)
    with pytest.raises(ValueError, match="exceed"):
        sample_policy_actions(torch.zeros(6), torch.zeros(6), lower=bounds["upper"], upper=bounds["upper"])
    with pytest.raises(ValueError, match="identical shapes"):
        truncated_gaussian_kl(torch.zeros(2, 6), torch.zeros(6), torch.zeros(3, 6), torch.zeros(6))
    with pytest.raises(ValueError, match="empty"):
        normalized_action_means(torch.zeros(0, 6), **bounds)


def test_legacy_gaussian_kl_still_matches_torch_reference():
    old = torch.tensor([[1.1, -1.2, 0., .3, 0., -.4]], dtype=torch.float64)
    new = old + .01
    logs = torch.full((6,), -3., dtype=torch.float64)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(old, logs.exp()), torch.distributions.Normal(new, (logs + .2).exp())
    ).sum(-1)
    torch.testing.assert_close(gaussian_kl(old, logs, new, logs + .2), expected, rtol=1e-12, atol=1e-12)
