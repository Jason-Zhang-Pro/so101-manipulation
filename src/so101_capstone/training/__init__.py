"""Reward-based training support for the SO-101 Capstone project."""

from .residual_ppo import (
    CompressedResidualObservation,
    PpoUpdateMetrics,
    ResidualActorCritic,
    ResidualPpoConfig,
    ResidualPpoTrainer,
    ResidualRolloutBatch,
    ResidualRolloutBuffer,
    compress_policy_observation,
    freeze_module,
    model_state_sha256,
    resize_rgb_image,
)

__all__ = [
    "CompressedResidualObservation",
    "PpoUpdateMetrics",
    "ResidualActorCritic",
    "ResidualPpoConfig",
    "ResidualPpoTrainer",
    "ResidualRolloutBatch",
    "ResidualRolloutBuffer",
    "compress_policy_observation",
    "freeze_module",
    "model_state_sha256",
    "resize_rgb_image",
]
