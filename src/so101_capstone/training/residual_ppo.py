"""Residual PPO components for frozen-ACT reward-based simulation training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ACTION_DIM = 6
STATE_DIM = 6
DEFAULT_IMAGE_HEIGHT = 96
DEFAULT_IMAGE_WIDTH = 128
_LOG_PROB_EPS = 1e-6


def _positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def _probability(value: float, *, name: str, inclusive_one: bool = False) -> float:
    result = float(value)
    upper_ok = result <= 1.0 if inclusive_one else result < 1.0
    if not math.isfinite(result) or result < 0.0 or not upper_ok:
        bound = "[0, 1]" if inclusive_one else "[0, 1)"
        raise ValueError(f"{name} must be within {bound}.")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True)
class ResidualPpoConfig:
    """Trainable residual actor-critic and PPO update configuration."""

    action_dim: int = ACTION_DIM
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    residual_limit_rad: float = 0.1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    max_gradient_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 128
    hidden_dim: int = 256
    initial_log_std: float = -2.3

    def __post_init__(self) -> None:
        if self.action_dim != ACTION_DIM:
            raise ValueError("Residual PPO action_dim must be 6.")
        _positive_integer(self.image_height, name="image_height")
        _positive_integer(self.image_width, name="image_width")
        _positive_finite(self.residual_limit_rad, name="residual_limit_rad")
        _positive_finite(self.learning_rate, name="learning_rate")
        _probability(self.gamma, name="gamma", inclusive_one=True)
        _probability(self.gae_lambda, name="gae_lambda", inclusive_one=True)
        _positive_finite(self.clip_ratio, name="clip_ratio")
        _positive_finite(
            self.value_loss_coefficient,
            name="value_loss_coefficient",
        )
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be finite and non-negative.")
        _positive_finite(self.max_gradient_norm, name="max_gradient_norm")
        _positive_integer(self.ppo_epochs, name="ppo_epochs")
        _positive_integer(self.minibatch_size, name="minibatch_size")
        _positive_integer(self.hidden_dim, name="hidden_dim")
        if not math.isfinite(self.initial_log_std):
            raise ValueError("initial_log_std must be finite.")


@dataclass(frozen=True)
class CompressedResidualObservation:
    """CPU rollout representation used by the residual policy."""

    top_image: np.ndarray
    gripper_image: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray


def _readonly_vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (STATE_DIM,):
        raise ValueError(f"{name} must have shape ({STATE_DIM},).")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    result = vector.copy()
    result.setflags(write=False)
    return result


def resize_rgb_image(
    image: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize HWC uint8 RGB to compact CHW uint8 for rollout storage."""

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3).")
    if source.dtype != np.uint8:
        raise ValueError("RGB image must have dtype uint8.")
    _positive_integer(height, name="height")
    _positive_integer(width, name="width")

    tensor = (
        torch.from_numpy(np.ascontiguousarray(source).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
    )
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    result = (
        resized.round()
        .clamp_(0.0, 255.0)
        .to(dtype=torch.uint8)
        .squeeze(0)
        .cpu()
        .numpy()
        .copy()
    )
    result.setflags(write=False)
    return result


def compress_policy_observation(
    observation: Any,
    *,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
) -> CompressedResidualObservation:
    """Validate and compact one runtime policy observation."""

    return CompressedResidualObservation(
        top_image=resize_rgb_image(
            observation.top_image,
            height=image_height,
            width=image_width,
        ),
        gripper_image=resize_rgb_image(
            observation.gripper_image,
            height=image_height,
            width=image_width,
        ),
        joint_positions=_readonly_vector(
            observation.joint_positions,
            name="observation.joint_positions",
        ),
        joint_velocities=_readonly_vector(
            observation.joint_velocities,
            name="observation.joint_velocities",
        ),
    )


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualActorCritic(nn.Module):
    """Dual-camera Gaussian residual actor with a base-action-free critic."""

    def __init__(self, config: ResidualPpoConfig | None = None) -> None:
        super().__init__()
        self.config = ResidualPpoConfig() if config is None else config

        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.Tanh(),
        )
        self.actor_state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM * 3, 128),
            nn.Tanh(),
        )
        self.critic_state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM * 2, 128),
            nn.Tanh(),
        )
        self.actor_body = nn.Sequential(
            nn.Linear(128 * 3, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.critic_body = nn.Sequential(
            nn.Linear(128 * 3, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(self.config.hidden_dim, ACTION_DIM)
        self.critic_value = nn.Linear(self.config.hidden_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((ACTION_DIM,), self.config.initial_log_std)
        )

        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)
        nn.init.zeros_(self.critic_value.bias)

    @staticmethod
    def _normalise_images(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Image batch must have shape (batch, 3, height, width).")
        return images.to(dtype=torch.float32) / 255.0

    @staticmethod
    def _validate_state_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != STATE_DIM:
            raise ValueError(f"{name} must have shape (batch, {STATE_DIM}).")
        return value.to(dtype=torch.float32)

    def _camera_features(
        self,
        top_images: torch.Tensor,
        gripper_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top = self.camera_encoder(self._normalise_images(top_images))
        gripper = self.camera_encoder(self._normalise_images(gripper_images))
        return top, gripper

    def _actor_distribution(
        self,
        top_images: torch.Tensor,
        gripper_images: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> torch.distributions.Normal:
        top, gripper = self._camera_features(top_images, gripper_images)
        positions = self._validate_state_batch(joint_positions, name="joint_positions")
        velocities = self._validate_state_batch(joint_velocities, name="joint_velocities")
        base = self._validate_state_batch(base_actions, name="base_actions")
        state = self.actor_state_encoder(
            torch.cat((positions, velocities, base), dim=-1)
        )
        hidden = self.actor_body(torch.cat((top, gripper, state), dim=-1))
        mean = self.actor_mean(hidden)
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return torch.distributions.Normal(mean, log_std.exp())

    def value(
        self,
        top_images: torch.Tensor,
        gripper_images: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate V(s) without consuming or requiring a future ACT action."""

        top, gripper = self._camera_features(top_images, gripper_images)
        positions = self._validate_state_batch(joint_positions, name="joint_positions")
        velocities = self._validate_state_batch(joint_velocities, name="joint_velocities")
        state = self.critic_state_encoder(torch.cat((positions, velocities), dim=-1))
        hidden = self.critic_body(torch.cat((top, gripper, state), dim=-1))
        return self.critic_value(hidden).squeeze(-1)

    def _squashed_log_prob(
        self,
        distribution: torch.distributions.Normal,
        pre_tanh: torch.Tensor,
        squashed: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.config.residual_limit_rad
        correction = torch.log(scale * (1.0 - squashed.square()) + _LOG_PROB_EPS)
        return (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)

    def act(
        self,
        top_images: torch.Tensor,
        gripper_images: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        base_actions: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one bounded residual action, log probability, and value."""

        distribution = self._actor_distribution(
            top_images,
            gripper_images,
            joint_positions,
            joint_velocities,
            base_actions,
        )
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        residual = squashed * self.config.residual_limit_rad
        log_probability = self._squashed_log_prob(distribution, pre_tanh, squashed)
        value = self.value(
            top_images,
            gripper_images,
            joint_positions,
            joint_velocities,
        )
        return residual, log_probability, value

    def evaluate_actions(
        self,
        top_images: torch.Tensor,
        gripper_images: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        base_actions: torch.Tensor,
        residual_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored bounded residuals for PPO."""

        distribution = self._actor_distribution(
            top_images,
            gripper_images,
            joint_positions,
            joint_velocities,
            base_actions,
        )
        normalised = (
            residual_actions.to(dtype=torch.float32)
            / self.config.residual_limit_rad
        ).clamp(-1.0 + _LOG_PROB_EPS, 1.0 - _LOG_PROB_EPS)
        pre_tanh = torch.atanh(normalised)
        log_probability = self._squashed_log_prob(
            distribution,
            pre_tanh,
            normalised,
        )
        entropy = distribution.entropy().sum(dim=-1)
        value = self.value(
            top_images,
            gripper_images,
            joint_positions,
            joint_velocities,
        )
        return log_probability, entropy, value


@dataclass(frozen=True)
class ResidualRolloutBatch:
    top_images: torch.Tensor
    gripper_images: torch.Tensor
    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    base_actions: torch.Tensor
    residual_actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ResidualRolloutBuffer:
    """CPU rollout buffer with episode-aware GAE."""

    def __init__(self, config: ResidualPpoConfig | None = None) -> None:
        self.config = ResidualPpoConfig() if config is None else config
        self.clear()

    def clear(self) -> None:
        self._top_images: list[np.ndarray] = []
        self._gripper_images: list[np.ndarray] = []
        self._joint_positions: list[np.ndarray] = []
        self._joint_velocities: list[np.ndarray] = []
        self._base_actions: list[np.ndarray] = []
        self._residual_actions: list[np.ndarray] = []
        self._log_probabilities: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._next_values: list[float] = []
        self._terminated: list[bool] = []
        self._episode_end: list[bool] = []

    def __len__(self) -> int:
        return len(self._rewards)

    def add(
        self,
        observation: CompressedResidualObservation,
        *,
        base_action: Sequence[float] | np.ndarray,
        residual_action: Sequence[float] | np.ndarray,
        log_probability: float,
        value: float,
        reward: float,
        next_value: float,
        terminated: bool,
        episode_end: bool,
    ) -> None:
        expected_image_shape = (
            3,
            self.config.image_height,
            self.config.image_width,
        )
        for name, image in (
            ("top_image", observation.top_image),
            ("gripper_image", observation.gripper_image),
        ):
            array = np.asarray(image)
            if array.shape != expected_image_shape or array.dtype != np.uint8:
                raise ValueError(
                    f"{name} must have shape {expected_image_shape} and dtype uint8."
                )
        base = _readonly_vector(base_action, name="base_action")
        residual = _readonly_vector(residual_action, name="residual_action")
        if np.any(np.abs(residual) > self.config.residual_limit_rad + 1e-6):
            raise ValueError("residual_action exceeds residual_limit_rad.")
        scalars = {
            "log_probability": log_probability,
            "value": value,
            "reward": reward,
            "next_value": next_value,
        }
        for name, item in scalars.items():
            if not math.isfinite(float(item)):
                raise ValueError(f"{name} must be finite.")
        if terminated and not episode_end:
            raise ValueError("A terminated transition must end its episode.")

        self._top_images.append(np.asarray(observation.top_image).copy())
        self._gripper_images.append(np.asarray(observation.gripper_image).copy())
        self._joint_positions.append(np.asarray(observation.joint_positions).copy())
        self._joint_velocities.append(np.asarray(observation.joint_velocities).copy())
        self._base_actions.append(base.copy())
        self._residual_actions.append(residual.copy())
        self._log_probabilities.append(float(log_probability))
        self._values.append(float(value))
        self._rewards.append(float(reward))
        self._next_values.append(float(next_value))
        self._terminated.append(bool(terminated))
        self._episode_end.append(bool(episode_end))

    def _advantages_and_returns(self) -> tuple[np.ndarray, np.ndarray]:
        if not self:
            raise ValueError("Cannot compute advantages from an empty rollout.")
        advantages = np.zeros(len(self), dtype=np.float32)
        gae = 0.0
        for index in range(len(self) - 1, -1, -1):
            bootstrap = 0.0 if self._terminated[index] else self._next_values[index]
            delta = (
                self._rewards[index]
                + self.config.gamma * bootstrap
                - self._values[index]
            )
            continuation = 0.0 if self._episode_end[index] else 1.0
            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * continuation
                * gae
            )
            advantages[index] = gae
        returns = advantages + np.asarray(self._values, dtype=np.float32)
        return advantages, returns

    def batch(self, device: str | torch.device) -> ResidualRolloutBatch:
        advantages, returns = self._advantages_and_returns()
        mean = float(advantages.mean())
        standard_deviation = float(advantages.std())
        advantages = (advantages - mean) / max(standard_deviation, 1e-8)
        target_device = torch.device(device)

        def tensor(array: Any, *, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(array, dtype=dtype, device=target_device)

        return ResidualRolloutBatch(
            top_images=tensor(np.stack(self._top_images), dtype=torch.uint8),
            gripper_images=tensor(np.stack(self._gripper_images), dtype=torch.uint8),
            joint_positions=tensor(np.stack(self._joint_positions), dtype=torch.float32),
            joint_velocities=tensor(np.stack(self._joint_velocities), dtype=torch.float32),
            base_actions=tensor(np.stack(self._base_actions), dtype=torch.float32),
            residual_actions=tensor(np.stack(self._residual_actions), dtype=torch.float32),
            old_log_probabilities=tensor(self._log_probabilities, dtype=torch.float32),
            old_values=tensor(self._values, dtype=torch.float32),
            advantages=tensor(advantages, dtype=torch.float32),
            returns=tensor(returns, dtype=torch.float32),
        )


@dataclass(frozen=True)
class PpoUpdateMetrics:
    transition_count: int
    minibatch_count: int
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ResidualPpoTrainer:
    """Optimise only the residual actor-critic."""

    def __init__(
        self,
        model: ResidualActorCritic,
        *,
        config: ResidualPpoConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = model.config if config is None else config
        if self.config != model.config:
            raise ValueError("Trainer and model configurations must match.")
        self.optimizer = (
            torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            if optimizer is None
            else optimizer
        )
        self.update_index = 0
        self.total_environment_steps = 0

    def update(self, rollout: ResidualRolloutBuffer) -> PpoUpdateMetrics:
        if len(rollout) < 2:
            raise ValueError("PPO update requires at least two transitions.")
        device = next(self.model.parameters()).device
        batch = rollout.batch(device)
        transition_count = len(rollout)
        all_indices = torch.arange(transition_count, device=device)
        accumulators = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }
        minibatch_count = 0

        self.model.train()
        for _ in range(self.config.ppo_epochs):
            permutation = all_indices[torch.randperm(transition_count, device=device)]
            for start in range(0, transition_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                new_log_probability, entropy, new_value = self.model.evaluate_actions(
                    batch.top_images[indices],
                    batch.gripper_images[indices],
                    batch.joint_positions[indices],
                    batch.joint_velocities[indices],
                    batch.base_actions[indices],
                    batch.residual_actions[indices],
                )
                log_ratio = new_log_probability - batch.old_log_probabilities[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * batch.advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * batch.advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(new_value, batch.returns[indices])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_gradient_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                        .to(dtype=torch.float32)
                        .mean()
                    )

                values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy_mean,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for name, value in values.items():
                    accumulators[name] += float(value.detach().cpu())
                minibatch_count += 1

        self.model.eval()
        self.update_index += 1
        self.total_environment_steps += transition_count
        return PpoUpdateMetrics(
            transition_count=transition_count,
            minibatch_count=minibatch_count,
            **{
                name: total / minibatch_count
                for name, total in accumulators.items()
            },
        )

    def checkpoint_payload(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
        payload = self.checkpoint_payload(extra=extra)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination

    def load_checkpoint(self, path: str | Path, *, map_location: Any = None) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        payload = torch.load(source, map_location=map_location, weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported residual PPO checkpoint schema.")
        if payload.get("config") != asdict(self.config):
            raise ValueError("Residual PPO checkpoint configuration does not match.")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.update_index = int(payload["update_index"])
        self.total_environment_steps = int(payload["total_environment_steps"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        return payload


def model_state_sha256(module: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and values for freeze verification."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def freeze_module(module: nn.Module) -> None:
    """Put a base policy in inference mode and exclude it from optimisation."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def iter_minibatches(size: int, minibatch_size: int) -> Iterator[slice]:
    """Small public helper retained for deterministic unit tests."""

    _positive_integer(size, name="size")
    _positive_integer(minibatch_size, name="minibatch_size")
    for start in range(0, size, minibatch_size):
        yield slice(start, min(size, start + minibatch_size))
