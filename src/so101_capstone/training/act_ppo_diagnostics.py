"""Read-only diagnostics for the old-rollout PPO policy comparison."""

from __future__ import annotations

import torch

from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS
from so101_capstone.rewards.task_state import TaskPhase


def phase_policy_diagnostics(
    analytic_kl: torch.Tensor,
    old_means: torch.Tensor,
    new_means: torch.Tensor,
    task_context: torch.Tensor,
) -> dict[str, dict]:
    """Group KL and clipped deterministic-command changes by recorded phase.

    These are commanded-position differences, not measured robot motion and
    not differences between random action samples. The affine slopes exactly
    match policy_positions_to_gazebo, including the gripper's [0, 100] range.
    Missing or invalid phase one-hot vectors remain explicitly unknown.
    """
    if not len(analytic_kl):
        return {}
    phases = tuple(phase.value for phase in TaskPhase)
    phase_indices = torch.full((len(analytic_kl),), -1, dtype=torch.long)
    if task_context.shape[-1] >= len(phases):
        one_hot = task_context[:, :len(phases)].detach().cpu()
        valid = ((one_hot == 0) | (one_hot == 1)).all(dim=-1) & (one_hot.sum(dim=-1) == 1)
        phase_indices[valid] = one_hot[valid].argmax(dim=-1)

    old_means, new_means = old_means.detach().cpu().double(), new_means.detach().cpu().double()
    lower = torch.tensor([c.policy_lower for c in SIM_JOINT_CALIBRATIONS], dtype=torch.float64)
    upper = torch.tensor([c.policy_upper for c in SIM_JOINT_CALIBRATIONS], dtype=torch.float64)
    slopes = torch.tensor([
        (c.gazebo_upper_rad - c.gazebo_lower_rad) / (c.policy_upper - c.policy_lower)
        for c in SIM_JOINT_CALIBRATIONS
    ], dtype=torch.float64)
    displacement = (new_means.clamp(lower, upper) - old_means.clamp(lower, upper)) * slopes
    analytic_kl = analytic_kl.detach().cpu().double()
    result = {}
    for index in torch.unique(phase_indices).tolist():
        mask = phase_indices == index
        kl, delta = analytic_kl[mask], displacement[mask]
        result[phases[index] if index >= 0 else "unknown"] = {
            "transition_count": int(mask.sum()),
            "policy_kl": float(kl.mean()),
            "policy_kl_p95": float(torch.quantile(kl, .95)),
            "policy_kl_max": float(kl.max()),
            # Compatibility field names only: schema 4 values describe the
            # bounded distribution, not a Gaussian before execution clipping.
            "pre_clip_policy_kl": float(kl.mean()),
            "pre_clip_policy_kl_p95": float(torch.quantile(kl, .95)),
            "pre_clip_policy_kl_max": float(kl.max()),
            "clipped_mean_command_rms_rad": float(delta.square().mean().sqrt()),
            "clipped_mean_command_max_abs_rad": float(delta.abs().max()),
            "clipped_mean_command_per_joint_rms_rad": {
                c.name: float(value) for c, value in zip(
                    SIM_JOINT_CALIBRATIONS, delta.square().mean(dim=0).sqrt(), strict=True,
                )
            },
        }
    return result
