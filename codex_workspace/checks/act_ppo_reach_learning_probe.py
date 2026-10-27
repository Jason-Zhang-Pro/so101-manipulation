"""Offline real ACT cadence/export/likelihood check; no optimizer or simulator.

Synthetic images exercise code compatibility only. This is not evidence of
reaching, grasping, task success, CUDA parity or real-time inference speed.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

import torch

from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
from so101_capstone.policies.act_policy_adapter import (
    ACT_FIXED_IMAGE_KEY, ACT_HANDEYE_IMAGE_KEY, ACT_STATE_KEY,
)
from so101_capstone.training.act_ppo import (
    ActPpoConfig, ActPpoTrainer, ActPpoValueNetwork,
    evaluate_bounded_policy_actions, predict_action_chunk_with_grad,
    sample_bounded_policy_actions,
)
from so101_capstone.training.act_ppo_gazebo import _configure_act_replanning
from so101_capstone.training.residual_ppo import model_state_sha256


def inputs(top: float, wrist: float, state: float) -> dict[str, torch.Tensor]:
    return {
        ACT_FIXED_IMAGE_KEY: torch.full((1, 3, 480, 640), top),
        ACT_HANDEYE_IMAGE_KEY: torch.full((1, 3, 480, 640), wrist),
        ACT_STATE_KEY: torch.full((1, 6), state),
    }


def native_sequence(policy, first_inputs, following_inputs):
    """Count actual model forwards while the native policy manages its queue."""
    forward_steps = []
    current_step = 0

    def record_forward(*_):
        forward_steps.append(current_step)

    policy.reset()
    hook = policy.model.register_forward_hook(record_forward)
    try:
        actions = []
        for current_step in range(1, 7):
            actions.append(policy.select_action(first_inputs if current_step == 1 else following_inputs).clone())
    finally:
        hook.remove()
    assert forward_steps == [1, 6], forward_steps
    return torch.stack(actions), forward_steps


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(271307)
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "outputs/train/act_so101_pick_test_20260807_160300/training_output/checkpoints/0005000/pretrained_model"
    loaded = load_act_checkpoint(checkpoint, device="cpu")
    policy = loaded.policy
    original_schedule = {
        "chunk_size": loaded.config.chunk_size,
        "n_action_steps": loaded.config.n_action_steps,
        "temporal_ensemble_coeff": loaded.config.temporal_ensemble_coeff,
    }
    original_weights = model_state_sha256(policy)
    assert original_schedule == {"chunk_size": 30, "n_action_steps": 15, "temporal_ensemble_coeff": None}
    _configure_act_replanning(loaded, 5)
    assert loaded.config.n_action_steps == policy.config.n_action_steps == policy.model.config.n_action_steps == 5
    assert loaded.config.chunk_size == policy.config.chunk_size == 30

    first_inputs = inputs(.5, .25, 0.)
    changed_inputs = inputs(.4, .35, .2)
    with torch.no_grad():
        first_chunk = policy.predict_action_chunk(first_inputs)
        changed_chunk = policy.predict_action_chunk(changed_inputs)
    native_actions, native_forward_steps = native_sequence(policy, first_inputs, changed_inputs)
    expected_actions = torch.cat((first_chunk[:, :5].transpose(0, 1), changed_chunk[:, :1].transpose(0, 1)))
    torch.testing.assert_close(native_actions, expected_actions, atol=0, rtol=0)

    cfg = ActPpoConfig(
        actor_learning_rate=1e-7, actor_backbone_learning_rate=1e-8,
        exploration_learning_rate=1e-4, initial_log_std=-4., minimum_log_std=-6.,
        maximum_log_std=-3.5, gripper_maximum_log_std=-1.5,
        critic_image_height=16, critic_image_width=16, critic_hidden_dim=32,
    )
    trainer = ActPpoTrainer(policy, ActPpoValueNetwork(cfg), config=cfg)
    initial_exploration = trainer.exploration.log_std.detach().clone()
    means = predict_action_chunk_with_grad(policy, first_inputs)
    torch.testing.assert_close(means.detach(), first_chunk, atol=0, rtol=0)
    manipulation_offset = torch.tensor([0.] * 5 + [1.7])
    effective_log_std = trainer.exploration.with_offset(manipulation_offset)
    torch.testing.assert_close(effective_log_std.detach(), torch.tensor([-4.] * 5 + [-2.3]), atol=1e-7, rtol=0)
    with torch.no_grad():
        actions, original_log_prob, original_samples = sample_bounded_policy_actions(
            means[:, :5], effective_log_std, return_latent=True,
        )
    replay_log_prob, entropy = evaluate_bounded_policy_actions(
        means[:, :5], actions, trainer.exploration.with_offset(manipulation_offset),
        policy_action_latents=original_samples.detach(),
    )
    torch.testing.assert_close(replay_log_prob.detach(), original_log_prob, atol=0, rtol=0)
    assert torch.isfinite(entropy).all()
    (-replay_log_prob.mean()).backward()
    gradient_parameters = [(name, parameter) for name, parameter in policy.named_parameters()
                           if parameter.grad is not None]
    assert gradient_parameters
    assert all(torch.isfinite(parameter.grad).all() for _, parameter in gradient_parameters)
    action_head_gradients = [(name, parameter) for name, parameter in gradient_parameters if "action_head" in name]
    assert action_head_gradients and any(torch.count_nonzero(parameter.grad) for _, parameter in action_head_gradients)
    assert trainer.exploration.log_std.grad is not None
    assert torch.isfinite(trainer.exploration.log_std.grad).all()
    assert torch.count_nonzero(trainer.exploration.log_std.grad) == 6
    assert not trainer.optimizer.state  # No Adam state can exist without an optimizer step.
    assert torch.equal(trainer.exploration.log_std.detach(), initial_exploration)
    assert model_state_sha256(policy) == original_weights

    with tempfile.TemporaryDirectory(prefix="act-ppo-reach-native-export-") as directory:
        policy.save_pretrained(directory)
        exported_config = json.loads((Path(directory) / "config.json").read_text())
        assert exported_config["n_action_steps"] == 5 and exported_config["chunk_size"] == 30
        reloaded = load_act_checkpoint(directory, device="cpu")
        assert reloaded.config.n_action_steps == 5 and reloaded.config.chunk_size == 30
        assert model_state_sha256(reloaded.policy) == original_weights
        reloaded_actions, reloaded_forward_steps = native_sequence(reloaded.policy, first_inputs, changed_inputs)
        torch.testing.assert_close(reloaded_actions, native_actions, atol=0, rtol=0)
    assert model_state_sha256(policy) == original_weights

    result = {
        "status": "passed", "checkpoint": str(checkpoint), "device": "cpu",
        "synthetic_images": True, "optimizer_steps": 0, "simulator_started": False,
        "original_schedule": original_schedule,
        "effective_schedule": {"chunk_size": 30, "n_action_steps": 5, "temporal_ensemble_coeff": None},
        "native_select_action_forward_steps": native_forward_steps,
        "exported_select_action_forward_steps": reloaded_forward_steps,
        "native_queue_expected_actions_max_abs_error": float((native_actions-expected_actions).abs().max()),
        "native_vs_differentiable_chunk_max_abs_error": float((means.detach()-first_chunk).abs().max()),
        "export_reload_native_actions_max_abs_error": float((reloaded_actions-native_actions).abs().max()),
        "manipulation_effective_log_std": effective_log_std.detach().tolist(),
        "sample_replay_max_abs_log_ratio": float((replay_log_prob.detach()-original_log_prob).abs().max()),
        "finite_actor_gradient_parameter_tensors": len(gradient_parameters),
        "nonzero_actor_gradient_parameter_tensors": sum(bool(torch.count_nonzero(parameter.grad)) for _, parameter in gradient_parameters),
        "action_head_gradient_nonzero": True,
        "exploration_gradient": trainer.exploration.log_std.grad.tolist(),
        "actor_weights_unchanged": True, "actor_weights_sha256": original_weights,
        "exploration_values_unchanged": True, "temporary_export_removed": True,
        "elapsed_s": round(time.monotonic()-started, 3),
        "limitations": [
            "Synthetic constant CPU images do not validate perceptual accuracy, processed camera realism or task success.",
            "Backward pass validates likelihood gradients only; no optimizer step or learning convergence was tested.",
            "Native select_action emits raw policy actions; execution bounds remain the deployment adapter's responsibility.",
            "No CUDA parity, memory/throughput limit, physics, hardware or environment was tested.",
        ],
    }
    destination = root / "codex_workspace/reports/2026-09-27_act_ppo_reach_learning_probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
