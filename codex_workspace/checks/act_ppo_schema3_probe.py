"""Offline CPU check of real ACT forward/gradient compatibility; no optimizer steps."""
import json
import tempfile
from pathlib import Path

import torch

from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
from so101_capstone.policies.act_policy_adapter import ACT_FIXED_IMAGE_KEY, ACT_HANDEYE_IMAGE_KEY, ACT_STATE_KEY
from so101_capstone.training.act_ppo import (
    ActPpoConfig, ActPpoTrainer, ActPpoValueNetwork,
    evaluate_bounded_policy_actions, predict_action_chunk_with_grad,
    sample_bounded_policy_actions,
)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1307)
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "outputs/train/act_so101_pick_test_20260807_160300/training_output/checkpoints/0005000/pretrained_model"
    loaded = load_act_checkpoint(checkpoint, device="cpu")
    policy = loaded.policy
    cfg = ActPpoConfig(actor_learning_rate=1e-7, actor_backbone_learning_rate=1e-8,
                       critic_image_height=16, critic_image_width=16, critic_hidden_dim=32)
    trainer = ActPpoTrainer(policy, ActPpoValueNetwork(cfg), config=cfg)
    batch = {
        ACT_FIXED_IMAGE_KEY: torch.full((1, 3, 480, 640), .5),
        ACT_HANDEYE_IMAGE_KEY: torch.full((1, 3, 480, 640), .25),
        ACT_STATE_KEY: torch.zeros(1, 6),
    }
    with torch.no_grad():
        native = policy.predict_action_chunk(batch)
    means = predict_action_chunk_with_grad(policy, batch)
    torch.testing.assert_close(native, means.detach(), rtol=0, atol=0)
    with torch.no_grad():
        actions, original_log_prob, samples = sample_bounded_policy_actions(
            means[:, :1], trainer.exploration.value(), return_latent=True,
        )
    replay_log_prob, _ = evaluate_bounded_policy_actions(
        means[:, :1], actions, trainer.exploration.value(), policy_action_latents=samples,
    )
    torch.testing.assert_close(replay_log_prob.detach(), original_log_prob, rtol=0, atol=0)
    (-replay_log_prob.mean()).backward()
    grad_parameters = [(name, p) for name, p in policy.named_parameters() if p.grad is not None]
    assert grad_parameters and all(torch.isfinite(p.grad).all() for _, p in grad_parameters)
    assert any(torch.count_nonzero(p.grad) for name, p in grad_parameters if "action_head" in name)
    assert torch.count_nonzero(trainer.exploration.log_std.grad)
    with tempfile.TemporaryDirectory(prefix="act-ppo-schema3-export-") as directory:
        policy.save_pretrained(directory)
        reloaded = load_act_checkpoint(directory, device="cpu")
        with torch.no_grad():
            exported_native = reloaded.policy.predict_action_chunk(batch)
        torch.testing.assert_close(exported_native, native, rtol=0, atol=0)
    result = {
        "checkpoint": str(checkpoint), "device": "cpu", "synthetic_images": True,
        "optimizer_steps": 0, "chunk_shape": list(means.shape),
        "native_forward_max_abs_error": float((native - means.detach()).abs().max()),
        "sample_replay_max_abs_log_ratio": float((replay_log_prob.detach() - original_log_prob).abs().max()),
        "finite_gradient_parameter_tensors": len(grad_parameters),
        "action_head_gradient_nonzero": True,
        "exploration_gradient": trainer.exploration.log_std.grad.tolist(),
        "checkpoint_schema": trainer.checkpoint_payload()["schema_version"],
        "actor_forward_contract_present": "actor_forward" in trainer.execution_signature,
        "export_reload_forward_max_abs_error": float((exported_native - native).abs().max()),
    }
    destination = root / "codex_workspace/reports/2026-09-27_act_ppo_schema3_real_act_probe.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
