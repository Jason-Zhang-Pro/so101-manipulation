#!/usr/bin/env python3
"""Write ACT VAE latent diagnostics to TensorBoard from saved checkpoints.

This is a sidecar for LeRobot training. It periodically loads the newest ACT
checkpoint, runs a small batch from the training dataset in train mode, and
writes latent mu/sigma/KL statistics to TensorBoard.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy


STEP_RE = re.compile(r"^\d{6}$")


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.parent.name)
    except ValueError:
        return -1


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None

    for candidate in ckpt_root.glob("*/pretrained_model"):
        if STEP_RE.match(candidate.parent.name) and (candidate / "config.json").exists():
            checkpoints.append(candidate)

    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def make_dataset(policy: ACTPolicy, repo_id: str, root: str | None, video_backend: str) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    return LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )


def prepare_batch(policy: ACTPolicy, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=device.type == "cuda")

    batch = policy.normalize_inputs(batch)
    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    batch = policy.normalize_targets(batch)
    return batch


@torch.no_grad()
def latent_stats(policy: ACTPolicy, batch: dict[str, torch.Tensor]) -> dict[str, float | torch.Tensor]:
    policy.train()
    _, (mu, log_sigma_x2) = policy.model(batch)
    if mu is None or log_sigma_x2 is None:
        raise RuntimeError("Policy did not return latent statistics. Is use_vae=true and action present?")

    sigma = torch.exp(0.5 * log_sigma_x2)
    kld_per_dim = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
    kld_per_sample = kld_per_dim.sum(dim=-1)

    return {
        "mu": mu.detach().cpu(),
        "sigma": sigma.detach().cpu(),
        "log_sigma_x2": log_sigma_x2.detach().cpu(),
        "z_sample": (mu + sigma * torch.randn_like(mu)).detach().cpu(),
        "z_mu_mean": float(mu.mean().item()),
        "z_mu_abs_mean": float(mu.abs().mean().item()),
        "z_mu_std": float(mu.std().item()),
        "z_sigma_mean": float(sigma.mean().item()),
        "z_sigma_std": float(sigma.std().item()),
        "z_logvar_mean": float(log_sigma_x2.mean().item()),
        "z_logvar_std": float(log_sigma_x2.std().item()),
        "kld_mean": float(kld_per_sample.mean().item()),
        "kld_per_dim_mean": float(kld_per_dim.mean().item()),
        "active_dims_abs_mu_gt_0_1": float((mu.abs().mean(dim=0) > 0.1).sum().item()),
        "active_dims_abs_mu_gt_0_05": float((mu.abs().mean(dim=0) > 0.05).sum().item()),
    }


def write_config(writer: SummaryWriter, policy: ACTPolicy, step: int) -> None:
    cfg = policy.config
    scalars = {
        "config/latent_dim": cfg.latent_dim,
        "config/n_vae_encoder_layers": cfg.n_vae_encoder_layers,
        "config/n_encoder_layers": cfg.n_encoder_layers,
        "config/n_decoder_layers": cfg.n_decoder_layers,
        "config/dim_model": cfg.dim_model,
        "config/n_heads": cfg.n_heads,
        "config/chunk_size": cfg.chunk_size,
        "config/n_action_steps": cfg.n_action_steps,
        "config/kl_weight": cfg.kl_weight,
        "config/temporal_ensemble_coeff": cfg.temporal_ensemble_coeff or 0.0,
    }
    for tag, value in scalars.items():
        writer.add_scalar(tag, value, step)

    writer.add_text(
        "config/act_layers",
        "\n".join(
            [
                f"latent_dim: {cfg.latent_dim}",
                f"vae_encoder_layers: {cfg.n_vae_encoder_layers}",
                f"transformer_encoder_layers: {cfg.n_encoder_layers}",
                f"transformer_decoder_layers: {cfg.n_decoder_layers}",
                f"dim_model: {cfg.dim_model}",
                f"heads: {cfg.n_heads}",
                f"chunk_size: {cfg.chunk_size}",
                f"n_action_steps: {cfg.n_action_steps}",
                f"kl_weight: {cfg.kl_weight}",
                f"temporal_ensemble_coeff: {cfg.temporal_ensemble_coeff}",
            ]
        ),
        step,
    )


def write_latent(writer: SummaryWriter, stats: dict[str, float | torch.Tensor], step: int, max_dims: int) -> None:
    for key, value in stats.items():
        if isinstance(value, float):
            writer.add_scalar(f"latent/{key}", value, step)

    mu = stats["mu"]
    sigma = stats["sigma"]
    z_sample = stats["z_sample"]
    if not isinstance(mu, torch.Tensor) or not isinstance(sigma, torch.Tensor) or not isinstance(z_sample, torch.Tensor):
        raise TypeError("latent tensors missing")

    writer.add_histogram("latent/z_mu_hist", mu, step)
    writer.add_histogram("latent/z_sigma_hist", sigma, step)
    writer.add_histogram("latent/z_sample_hist", z_sample, step)

    dims = min(max_dims, mu.shape[1])
    mu_dim = mu.mean(dim=0)
    sigma_dim = sigma.mean(dim=0)
    for idx in range(dims):
        writer.add_scalar(f"latent_dim/mu_{idx:02d}", float(mu_dim[idx]), step)
        writer.add_scalar(f"latent_dim/sigma_{idx:02d}", float(sigma_dim[idx]), step)

    writer.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="LeRobot output_dir containing checkpoints/")
    parser.add_argument("--logdir", required=True, help="TensorBoard logdir to write latent metrics into")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-dims", type=int, default=32)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    logdir = Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(logdir))
    seen_steps: set[int] = set()
    dataset = None
    dataloader = None

    try:
        while True:
            checkpoint = find_latest_checkpoint(run_dir)
            if checkpoint is None:
                print(f"waiting for checkpoint under {run_dir / 'checkpoints'}", flush=True)
            else:
                step = checkpoint_step(checkpoint)
                if step not in seen_steps:
                    print(f"loading checkpoint step {step}: {checkpoint}", flush=True)
                    policy = ACTPolicy.from_pretrained(checkpoint)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    policy.to(device)

                    if dataset is None:
                        dataset = make_dataset(policy, args.dataset_repo_id, args.dataset_root, args.video_backend)
                        dataloader = DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False,
                        )

                    write_config(writer, policy, step)

                    collected: list[dict[str, float | torch.Tensor]] = []
                    assert dataloader is not None
                    for idx, batch in enumerate(dataloader):
                        if idx >= args.num_batches:
                            break
                        prepared = prepare_batch(policy, batch, device)
                        collected.append(latent_stats(policy, prepared))

                    if collected:
                        tensor_keys = ["mu", "sigma", "log_sigma_x2", "z_sample"]
                        merged: dict[str, float | torch.Tensor] = {}
                        for key in tensor_keys:
                            merged[key] = torch.cat([item[key] for item in collected if isinstance(item[key], torch.Tensor)])
                        scalar_keys = [k for k, v in collected[0].items() if isinstance(v, float)]
                        for key in scalar_keys:
                            merged[key] = float(sum(float(item[key]) for item in collected) / len(collected))
                        write_latent(writer, merged, step, args.max_dims)
                        print(
                            "wrote latent "
                            f"step={step} z_mu_mean={merged['z_mu_mean']:.5f} "
                            f"z_mu_std={merged['z_mu_std']:.5f} "
                            f"kld={merged['kld_mean']:.5f}",
                            flush=True,
                        )
                    seen_steps.add(step)

            if args.once:
                break
            time.sleep(args.interval_s)
    finally:
        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
