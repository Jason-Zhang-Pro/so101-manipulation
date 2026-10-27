"""Direct ACT PPO finetuning in deterministic SO-101 Gazebo.

PPO updates ACT itself. The final deployment artifact is a normal
standalone LeRobot ACT directory. The critic and exploration variance are
training-only and are saved separately for resume.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from functools import cached_property
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Callable, Mapping

import numpy as np
import torch

from so101_capstone.environments.episode_reset import EpisodeResetError
from so101_capstone.environments.gazebo_world_control import (
    GazeboWorldControlError,
)
from so101_capstone.environments.reach_env_core import (
    ReachEnvironmentConfig,
    ReachEnvironmentStateError,
)
from so101_capstone.environments.simulation_runtime import (
    SimulationRuntimeConfig,
    SimulationRuntimeError,
    create_ros_simulation_runtime,
)
from so101_capstone.environments.simulation_step_coordinator import (
    SimulationStepCoordinatorError,
)
from so101_capstone.environments.target_cube_control import (
    TargetCubeControlError,
)
from so101_capstone.policies.act_checkpoint_loader import (
    load_act_checkpoint,
)
from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
from so101_capstone.policies.act_temporal_config import validate_act_temporal_config
from so101_capstone.rewards import RewardState
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM, TASK_CONTEXT_SCHEMA_VERSION
from .act_ppo_curriculum import ReachCurriculum, ReachCurriculumConfig

from .act_ppo import (
    CRITIC_POLICY_CONTEXT_SCHEMA_VERSION,
    ActPpoConfig,
    ActPpoRolloutBuffer,
    ActPpoTemporalHistory,
    ActPpoTrainer,
    ActPpoValueNetwork,
    actor_observation_from_batch,
    actor_observation_tensors,
    predict_action_chunk_with_grad,
    sample_bounded_policy_actions,
    build_critic_policy_context,
    critic_policy_context_size,
    policy_action_saturation_diagnostics,
)
from .act_ppo_tensorboard import DirectActPpoTensorBoard
from .act_ppo_stage_schedule import (TRAINING_STAGE_SELECTIONS, selected_training_stages,
                                     uses_later_training_stages)
from .act_ppo_checkpoint_contract import actor_forward_contract
from .residual_ppo import (
    compress_policy_observation,
    model_state_sha256,
)


DEFAULT_DEVICE = "cuda"
DEFAULT_EPISODE_COUNT = 6
DEFAULT_MAX_CONSECUTIVE_EPISODE_FAILURES = 3
DEFAULT_MAX_STEPS_PER_EPISODE = 600
DEFAULT_TARGET_DELAY_S = 3.0
DEFAULT_FINAL_TARGET_ABSENT_S = 3.0
DEFAULT_ROLLOUT_STEPS = 600
DEFAULT_MINIBATCH_SIZE = 128
DEFAULT_CHECKPOINT_EVERY_UPDATES = 1
DEFAULT_POLICY_HZ = 30.0
_DISABLED_DELTA_SENTINEL_RAD = 1_000_000.0
_STAGE_TIME_FIELDS = (
    "s0_budget_s", "s0_home_hold_s", "s1_budget_s", "s1_reach_hold_s",
    "s2_budget_s", "s3_budget_s", "s4_budget_s",
)


@dataclass(frozen=True)
class ActPpoGazeboConfig:
    act_checkpoint: Path
    # Programmatic legacy callers retain their contract; the CLI selects v2.
    reward_profile: str = "legacy"
    reward_config: Path | None = None
    task_mode: str = "pick_place"
    idle_empty_probability: float = 0.1
    idle_completed_probability: float = 0.1
    randomize_target_delay: bool = True
    # Keep the previous contract for programmatic callers and old checkpoints.
    exploration_profile: str = "stage_log_std"
    idle_exploration_log_std: float = -5.0
    moving_exploration_log_std: float = -4.0
    manipulation_exploration_log_std: float = -4.0
    home_hold_exploration_log_std: float = -5.0
    moving_gripper_exploration_log_std: float | None = None
    manipulation_gripper_exploration_log_std: float | None = None
    gripper_maximum_log_std: float | None = None
    act_max_action_steps: int | None = None
    reach_curriculum: bool = False
    curriculum_window_size: int = 20
    curriculum_pass_rate: float = 0.7
    curriculum_rehearsal_probability: float = 0.2
    training_stages: str = "full"
    stage_config: Path | None = None
    s0_probability: float = 1.0 / 3.0
    # None inherits the stage JSON (or StageTrainingConfig defaults).
    s0_budget_s: float | None = None
    s0_home_hold_s: float | None = None
    s1_budget_s: float | None = None
    s1_reach_hold_s: float | None = None
    s2_budget_s: float | None = None
    s3_budget_s: float | None = None
    s4_budget_s: float | None = None
    # Retained when a mixed wave is narrowed to a single stage.
    enable_later_stage_context: bool = False
    stage_initialization_timeout_s: float | None = None
    device: str = DEFAULT_DEVICE
    episode_count: int = DEFAULT_EPISODE_COUNT
    max_consecutive_episode_failures: int = (
        DEFAULT_MAX_CONSECUTIVE_EPISODE_FAILURES
    )
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE
    target_delay_s: float = DEFAULT_TARGET_DELAY_S
    final_target_absent_s: float = DEFAULT_FINAL_TARGET_ABSENT_S
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE
    critic_image_height: int = 96
    critic_image_width: int = 128
    critic_hidden_dim: int = 256
    critic_learning_rate: float = 3e-4
    exploration_learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    kl_backoff_max_retries: int = 0
    kl_backoff_factor: float = 0.5
    kl_backoff_min_scale: float = 1e-4
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    mean_bounds_coefficient: float = 0.1
    raw_mean_bounds_coefficient: float = 0.0
    max_gradient_norm: float = 0.5
    critic_max_gradient_norm: float = 0.5
    exploration_max_gradient_norm: float = 0.5
    ppo_epochs: int = 4
    initial_log_std: float = -2.3
    minimum_log_std: float = -5.0
    maximum_log_std: float = 0.0
    state1_exploration_log_std: float = -5.0
    state2_exploration_log_std: float = -4.0
    state3_exploration_log_std: float = -4.0
    state4_not_home_exploration_log_std: float = -4.0
    state4_home_exploration_log_std: float = -5.0
    state1_exploration_enabled: bool = True
    state2_exploration_enabled: bool = True
    state3_exploration_enabled: bool = True
    state4_not_home_exploration_enabled: bool = True
    state4_home_exploration_enabled: bool = True
    state1_actor_update_enabled: bool = True
    state2_actor_update_enabled: bool = True
    state3_actor_update_enabled: bool = True
    state4_not_home_actor_update_enabled: bool = True
    state4_home_actor_update_enabled: bool = True
    actor_learning_rate: float | None = None
    actor_backbone_learning_rate: float | None = None
    checkpoint_every_updates: int = (
        DEFAULT_CHECKPOINT_EVERY_UPDATES
    )
    tensorboard_enabled: bool = False
    tensorboard_auto_open: bool = False
    tensorboard_port: int = 6006
    disable_max_joint_delta_limit: bool = True
    policy_hz: float = DEFAULT_POLICY_HZ
    startup_feedback_timeout_s: float = 30.0
    seed: int = 0
    output_directory: Path | None = None
    resume_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "act_checkpoint",
            Path(self.act_checkpoint).expanduser(),
        )
        for name in ("output_directory", "resume_checkpoint", "reward_config", "stage_config"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    Path(value).expanduser(),
                )
        if self.reward_profile not in {"legacy", "pick_place_v2"}:
            raise ValueError("Unknown reward profile.")
        if _assisted_grasp_enabled() and self.reward_profile != "pick_place_v2":
            raise ValueError("Assisted rigid grasp requires pick_place_v2 and its recorded task context.")
        if self.exploration_profile not in {"stage_log_std", "final_good_act", "final_good_act_near_target"}:
            raise ValueError("Unknown exploration profile.")
        if self.exploration_profile != "stage_log_std" and self.reward_profile != "pick_place_v2":
            raise ValueError("Physical-angle exploration requires pick_place_v2.")
        if self.training_stages != "full" and self.training_stages not in TRAINING_STAGE_SELECTIONS:
            raise ValueError("training_stages must be full, s0..s5, s0_s1 or s0_s5.")
        if uses_later_training_stages(self.training_stages):
            object.__setattr__(self, "enable_later_stage_context", True)
        if self.stage_initialization_timeout_s is not None and (
                isinstance(self.stage_initialization_timeout_s, bool)
                or not math.isfinite(self.stage_initialization_timeout_s)
                or self.stage_initialization_timeout_s <= 0):
            raise ValueError("stage_initialization_timeout_s must be positive and finite.")
        if (isinstance(self.raw_mean_bounds_coefficient, bool)
                or not math.isfinite(self.raw_mean_bounds_coefficient)
                or self.raw_mean_bounds_coefficient < 0):
            raise ValueError("raw_mean_bounds_coefficient must be finite and non-negative.")
        if (isinstance(self.s0_probability, bool) or not math.isfinite(self.s0_probability)
                or not 0 <= self.s0_probability <= 1):
            raise ValueError("s0_probability must be in [0, 1].")
        if self.training_stages != "full":
            if self.reward_profile != "pick_place_v2":
                raise ValueError("Stage training requires pick_place_v2.")
            if self.reach_curriculum:
                raise ValueError("Stage training uses a fixed central target sector; disable the full-task reach curriculum.")
            # Validate the merged durations before any simulation starts.
            _resolved_stage_config(self)
        elif self.stage_config is not None:
            raise ValueError("stage_config requires S0/S1 or later-stage training.")
        elif any(getattr(self, name) is not None for name in _STAGE_TIME_FIELDS):
            raise ValueError("Stage time overrides require S0/S1 or later-stage training.")
        if self.task_mode not in {"pick_place", "idle_empty", "idle_completed", "mixed"}:
            raise ValueError("Unknown task mode.")
        if self.reward_profile == "legacy" and self.task_mode != "pick_place":
            raise ValueError("Idle/mixed task modes require pick_place_v2.")
        if not isinstance(self.reach_curriculum, bool):
            raise ValueError("reach_curriculum must be a bool.")
        if self.reach_curriculum and self.reward_profile != "pick_place_v2":
            raise ValueError("Reach curriculum requires pick_place_v2.")
        if self.act_max_action_steps is not None and (
            isinstance(self.act_max_action_steps, bool)
            or not isinstance(self.act_max_action_steps, int) or self.act_max_action_steps < 1
        ):
            raise ValueError("act_max_action_steps must be a positive integer.")
        _curriculum_config(self)
        probabilities = (self.idle_empty_probability, self.idle_completed_probability)
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or sum(probabilities) > 1:
            raise ValueError("Idle scene probabilities must be nonnegative and sum to at most one.")
        if self.reward_profile == "pick_place_v2" and self.final_target_absent_s != 0:
            raise ValueError("pick_place_v2 requires final_target_absent_s=0; deleting an active target invalidates the task.")
        if self.reward_profile == "pick_place_v2":
            for group in ("state1", "state2", "state3", "state4_not_home", "state4_home"):
                expected_std = -5.0 if group in {"state1", "state4_home"} else -4.0
                if (getattr(self, f"{group}_exploration_log_std") != expected_std
                        or not getattr(self, f"{group}_exploration_enabled")
                        or not getattr(self, f"{group}_actor_update_enabled")):
                    raise ValueError("STATE1...STATE4 settings require reward_profile=legacy; use named v2 exploration groups.")
        for name in (
            "episode_count",
            "max_consecutive_episode_failures",
            "max_steps_per_episode",
            "rollout_steps",
            "minibatch_size",
            "checkpoint_every_updates",
            "tensorboard_port",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        for name, allow_zero in (
            ("target_delay_s", True),
            ("final_target_absent_s", True),
            ("policy_hz", False),
            ("startup_feedback_timeout_s", False),
        ):
            value = float(getattr(self, name))
            invalid = not math.isfinite(value) or value < 0.0
            if not allow_zero:
                invalid = invalid or value == 0.0
            if invalid:
                raise ValueError(f"{name} has an invalid value.")
        target_kl = float(self.target_kl)
        if not math.isfinite(target_kl) or target_kl <= 0.0:
            raise ValueError("target_kl must be positive and finite.")
        if (isinstance(self.kl_backoff_max_retries, bool)
                or not isinstance(self.kl_backoff_max_retries, int)
                or self.kl_backoff_max_retries < 0):
            raise ValueError("kl_backoff_max_retries must be a non-negative integer.")
        if (isinstance(self.kl_backoff_factor, bool)
                or not math.isfinite(self.kl_backoff_factor)
                or not 0 < self.kl_backoff_factor < 1):
            raise ValueError("kl_backoff_factor must be within (0, 1).")
        if (isinstance(self.kl_backoff_min_scale, bool)
                or not math.isfinite(self.kl_backoff_min_scale)
                or not 0 < self.kl_backoff_min_scale <= 1):
            raise ValueError("kl_backoff_min_scale must be within (0, 1].")
        for name in (
            "tensorboard_enabled",
            "tensorboard_auto_open",
            "state1_exploration_enabled",
            "state2_exploration_enabled",
            "state3_exploration_enabled",
            "state4_not_home_exploration_enabled",
            "state4_home_exploration_enabled",
            "state1_actor_update_enabled",
            "state2_actor_update_enabled",
            "state3_actor_update_enabled",
            "state4_not_home_actor_update_enabled",
            "state4_home_actor_update_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool.")
        if self.tensorboard_port > 65535:
            raise ValueError(
                "tensorboard_port must be within [1, 65535]."
            )
        if not self.disable_max_joint_delta_limit:
            raise ValueError(
                "Direct ACT PPO requires the relative joint-delta "
                "limit to remain disabled."
            )
        delay = self.target_delay_s * self.policy_hz
        rounded = int(round(delay))
        if not math.isclose(
            delay,
            rounded,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "target_delay_s must map to an integer policy step."
            )
        if self.training_stages == "full" and rounded > self.max_steps_per_episode:
            raise ValueError("Target delay occurs after the episode.")
        final_absent = (
            self.final_target_absent_s * self.policy_hz
        )
        final_absent_steps = int(round(final_absent))
        if not math.isclose(
            final_absent,
            final_absent_steps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "final_target_absent_s must map to an integer "
                "policy step."
            )
        if (
            final_absent_steps > 0
            and final_absent_steps >= self.max_steps_per_episode
        ):
            raise ValueError(
                "Final target-absent window must be shorter than "
                "the episode."
            )
        if final_absent_steps > 0:
            final_delete_step = (
                self.max_steps_per_episode - final_absent_steps
            )
            if rounded >= final_delete_step:
                raise ValueError(
                    "Target spawn must occur before the final "
                    "target-delete step."
                )

    @cached_property
    def critic_task_context_dim(self) -> int:
        """Resolve once, rather than rereading reward JSON on every control step."""
        if self.reward_profile == "legacy":
            return 0
        reward = _resolved_reward(self)
        dimension = TASK_CONTEXT_DIM
        if _assisted_grasp_enabled():
            from so101_capstone.rewards.task_state import GRASP_ATTACHMENT_CONTEXT_DIM
            dimension += GRASP_ATTACHMENT_CONTEXT_DIM
        if reward.reward_schema_version >= 5:
            from so101_capstone.rewards.pick_place_reward import REWARD_HISTORY_CONTEXT_DIM
            dimension += REWARD_HISTORY_CONTEXT_DIM
        stages = _resolved_stage_config(self)
        if stages is not None and stages.process_reward is not None:
            dimension += stages.process_reward.context_dim
        if stages is not None and stages.enable_later_stage_context:
            from so101_capstone.rewards.later_stage_training import later_stage_context_dim
            dimension += later_stage_context_dim()
        return dimension

    @property
    def resolved_stage_initialization_timeout_s(self) -> float:
        return (self.stage_initialization_timeout_s if self.stage_initialization_timeout_s is not None
                else 600.0 if self.enable_later_stage_context else 180.0)

    @property
    def target_delay_steps(self) -> int:
        return int(round(self.target_delay_s * self.policy_hz))

    @property
    def final_target_absent_steps(self) -> int:
        return int(
            round(self.final_target_absent_s * self.policy_hz)
        )

    @property
    def final_target_delete_step(self) -> int:
        return (
            self.max_steps_per_episode
            - self.final_target_absent_steps
        )

    @property
    def planned_transition_count(self) -> int:
        if self.training_stages != "full":
            stages = _resolved_stage_config(self)
            budget = max(stages.stage_budget_s(stage) for stage in selected_training_stages(self.training_stages))
            return self.episode_count * math.ceil(budget * self.policy_hz)
        return self.episode_count * self.max_steps_per_episode

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "act_checkpoint",
            "output_directory",
            "resume_checkpoint",
            "reward_config",
            "stage_config",
        ):
            value = result[name]
            result[name] = None if value is None else str(value)
        result["target_delay_steps"] = self.target_delay_steps
        result["final_target_absent_steps"] = (
            self.final_target_absent_steps
        )
        result["final_target_delete_step"] = (
            self.final_target_delete_step
        )
        result["planned_transition_count"] = (
            self.planned_transition_count
        )
        result["max_joint_delta_rad"] = None
        return result


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, sort_keys=True)
        stream.write("\n")


_PRETRAINED_PATTERN = re.compile(
    r"(?:^|/)act_so101_pick_test_(\d{8}_\d{6})(?:/|$)"
)


def _pretrained_run_timestamp(checkpoint: Path) -> str:
    match = _PRETRAINED_PATTERN.search(Path(checkpoint).as_posix())
    if match is None:
        raise ValueError(
            "ACT checkpoint path must contain "
            "act_so101_pick_test_YYYYMMDD_HHMMSS."
        )
    return match.group(1)


def _default_output_directory(checkpoint: Path) -> Path:
    finetune = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = _pretrained_run_timestamp(checkpoint)
    return (
        Path("output_simulation")
        / "finetuned"
        / f"act_ppo_{finetune}_from_pretrained_{source}"
    )


def _validate_stage_exploration_config(
    config: ActPpoGazeboConfig,
) -> None:
    if config.exploration_profile in {"final_good_act", "final_good_act_near_target"}:
        from .act_ppo_exploration_profile import validate_final_good_act_exploration_bounds
        validate_final_good_act_exploration_bounds(
            initial_log_std=config.initial_log_std,
            minimum_log_std=config.minimum_log_std,
            maximum_log_std=config.maximum_log_std,
            gripper_maximum_log_std=config.gripper_maximum_log_std,
        )
        return
    names = (
        "state1_exploration_log_std",
        "state2_exploration_log_std",
        "state3_exploration_log_std",
        "state4_not_home_exploration_log_std",
        "state4_home_exploration_log_std",
    ) if config.reward_profile == "legacy" else tuple(
        f"{group}_exploration_log_std" for group in ("idle", "moving", "manipulation", "home_hold")
    )
    for name in names:
        value = float(getattr(config, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if not config.minimum_log_std <= value <= config.maximum_log_std:
            raise ValueError(
                f"{name}={value} is outside configured log-std bounds."
            )
    gripper_maximum = (config.maximum_log_std if config.gripper_maximum_log_std is None
                       else config.gripper_maximum_log_std)
    for group in ("moving", "manipulation"):
        value = getattr(config, f"{group}_gripper_exploration_log_std")
        if value is not None and (not math.isfinite(value)
                                  or not config.minimum_log_std <= value <= gripper_maximum):
            raise ValueError(f"{group} gripper exploration is outside configured bounds.")
        if value is not None and config.reward_profile == "legacy":
            raise ValueError("Named gripper exploration requires pick_place_v2.")


def _act_ppo_config(
    config: ActPpoGazeboConfig,
    n_action_steps: int | None = None,
) -> ActPpoConfig:
    '''Resolve all Direct ACT PPO training-side settings.'''

    _validate_stage_exploration_config(config)
    stages = _resolved_stage_config(config)
    process = None if stages is None else stages.process_reward
    later_dimension = 0
    if stages is not None and stages.enable_later_stage_context:
        from so101_capstone.rewards.later_stage_training import later_stage_context_dim
        later_dimension = later_stage_context_dim()
    return ActPpoConfig(
        minibatch_size=config.minibatch_size,
        critic_image_height=config.critic_image_height,
        critic_image_width=config.critic_image_width,
        critic_hidden_dim=config.critic_hidden_dim,
        critic_learning_rate=config.critic_learning_rate,
        exploration_learning_rate=config.exploration_learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_ratio=config.clip_ratio,
        target_kl=config.target_kl,
        kl_backoff_max_retries=config.kl_backoff_max_retries,
        kl_backoff_factor=config.kl_backoff_factor,
        kl_backoff_min_scale=config.kl_backoff_min_scale,
        value_loss_coefficient=config.value_loss_coefficient,
        entropy_coefficient=config.entropy_coefficient,
        mean_bounds_coefficient=config.mean_bounds_coefficient,
        raw_mean_bounds_coefficient=config.raw_mean_bounds_coefficient,
        max_gradient_norm=config.max_gradient_norm,
        critic_max_gradient_norm=config.critic_max_gradient_norm,
        exploration_max_gradient_norm=config.exploration_max_gradient_norm,
        ppo_epochs=config.ppo_epochs,
        initial_log_std=config.initial_log_std,
        minimum_log_std=config.minimum_log_std,
        maximum_log_std=config.maximum_log_std,
        gripper_maximum_log_std=config.gripper_maximum_log_std,
        actor_learning_rate=config.actor_learning_rate,
        actor_backbone_learning_rate=config.actor_backbone_learning_rate,
        critic_task_context_dim=config.critic_task_context_dim,
        advantage_group_context_index=(config.critic_task_context_dim - later_dimension if later_dimension else
            config.critic_task_context_dim - process.context_dim
            if process is not None and process.schema_version >= 2 else None),
        advantage_group_count=6 if later_dimension else 2,
        critic_policy_context_dim=(
            critic_policy_context_size(n_action_steps)
            if config.reward_profile == "pick_place_v2" and n_action_steps is not None else 0
        ),
    )


def _resolved_reward(config: ActPpoGazeboConfig) -> PickPlaceRewardConfig | None:
    if config.reward_profile == "legacy":
        if config.reward_config is not None:
            raise ValueError("reward_config is only supported for pick_place_v2.")
        return None
    reward = PickPlaceRewardConfig.load(config.reward_config, gamma=config.gamma)
    if reward.grasp_evidence_mode != "contact":
        raise ValueError("Formal ACT-PPO training requires contact evidence; proxy is for offline diagnostics only.")
    if not math.isclose(reward.control_hz, config.policy_hz):
        raise ValueError("Reward and policy control rates must match.")
    return reward


def _randomization_signature(config: ActPpoGazeboConfig, *, world_count: int | None) -> dict[str, Any]:
    from so101_capstone.environments.environment_randomization import EnvironmentRandomizationConfig
    from so101_capstone.environments.simulation_runtime import _environment_bool, _environment_int

    if not _environment_bool("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", default=False):
        return {"enabled": False}
    randomization = EnvironmentRandomizationConfig.load(
        os.environ.get("SO101_ENVIRONMENT_RANDOMIZATION_CONFIG") or None)
    goal = randomization.payload["goal_region"]
    if any(bool(goal[field].get("enabled", False)) for field in ("position", "size")):
        raise ValueError("pick_place_v2 requires fixed goal position/size; disable their randomization.")
    if config.training_stages == "full" and (config.task_mode == "idle_completed" or (
        config.task_mode == "mixed" and config.idle_completed_probability > 0
    )):
        raise ValueError("idle_completed currently requires environment randomization disabled; "
                         "disable randomization or set idle_completed_probability=0 in mixed mode.")
    payload = dict(randomization.payload)
    return {
        "enabled": True, "config": payload,
        "config_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "base_seed": _environment_int("SO101_ENVIRONMENT_RANDOMIZATION_SEED", default=0),
        "world_indices": (list(range(world_count)) if world_count is not None else
                          [_environment_int("SO101_ENVIRONMENT_RANDOMIZATION_WORLD_INDEX", default=0)]),
        "episode_key": "global_attempt_index" if world_count is not None else "runtime_episode_counter",
        "sampling_schema": "blake2b_base_seed_world_index_episode_key_v1",
        "warmup_control_steps": 30,
    }


def _runtime_reward_kwargs(config: ActPpoGazeboConfig, reward: Any) -> dict[str, Any]:
    if reward is None:
        return {}
    return {"reward_profile": config.reward_profile, "reward_config": reward,
            "task_mode": "pick_place" if config.training_stages != "full" or config.task_mode == "mixed" else config.task_mode}


def _image_augmentation_signature(config: ActPpoGazeboConfig, *, world_count: int | None) -> dict[str, Any]:
    from so101_capstone.environments.environment_randomization import EnvironmentRandomizationConfig
    from so101_capstone.environments.simulation_runtime import _environment_bool, _environment_int
    if not _environment_bool("SO101_IMAGE_AUGMENTATION_ENABLED", default=False):
        return {"enabled": False}
    if _environment_bool("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", default=False):
        raise ValueError("Use either image-only augmentation or environment randomization, not both.")
    path = os.environ.get("SO101_IMAGE_AUGMENTATION_CONFIG") or (
        Path(__file__).resolve().parents[3] / "config" / "image_augmentation_s0_s1.json")
    payload = dict(EnvironmentRandomizationConfig.load(path).payload)
    return {
        "enabled": True, "config": payload,
        "config_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "base_seed": _environment_int("SO101_ENVIRONMENT_RANDOMIZATION_SEED", default=0),
        "world_indices": list(range(world_count)) if world_count is not None else [
            _environment_int("SO101_ENVIRONMENT_RANDOMIZATION_WORLD_INDEX", default=0)],
        "sampling_schema": "image_only_blake2b_base_seed_world_index_episode_key_v1",
        "environment_mutation": False,
        "ppo_replay": "reuse_collected_augmented_images_no_resampling",
    }


def _resolved_stage_config(config: ActPpoGazeboConfig):
    from so101_capstone.rewards.training_stages import StageTrainingConfig
    if config.training_stages == "full":
        return None
    overrides = {
        name: getattr(config, name) for name in _STAGE_TIME_FIELDS
        if getattr(config, name) is not None
    }
    if config.enable_later_stage_context:
        overrides["enable_later_stage_context"] = True
    return replace(StageTrainingConfig.load(config.stage_config), **overrides)


def _episode_step_limit(config: ActPpoGazeboConfig, task_mode: str, training_stage: str | None = None) -> int:
    stages = _resolved_stage_config(config)
    if stages is None:
        return config.max_steps_per_episode
    # The environment's simulation clock is authoritative. One guard step
    # accommodates floating-point/timing quantization without an early cut.
    stage = training_stage or (selected_training_stages(config.training_stages)[0]
        if len(selected_training_stages(config.training_stages)) == 1 else
        "S0" if task_mode == "idle_empty" else "S1")
    seconds = stages.stage_budget_s(stage)
    return math.ceil(seconds * config.policy_hz) + 1


def _sample_training_scene(config: ActPpoGazeboConfig, curriculum, *, seed, task_mode, info):
    if config.training_stages != "full":
        # Freeze the spatial curriculum at the central sector. Its old level
        # promotion eventually requires grasp/lift, outside this experiment.
        curriculum = ReachCurriculum()
    return _sample_curriculum_scene(curriculum, seed=seed, task_mode=task_mode, info=info)


def _exploration_profile_signature(config: ActPpoGazeboConfig) -> dict[str, Any]:
    if config.exploration_profile == "stage_log_std":
        return {}
    from .act_ppo_exploration_profile import (
        final_good_act_exploration_signature,
        final_good_act_near_target_exploration_signature,
    )
    signature = (final_good_act_near_target_exploration_signature()
                 if config.exploration_profile == "final_good_act_near_target"
                 else final_good_act_exploration_signature())
    return {"exploration_profile": signature}


def _assisted_grasp_enabled() -> bool:
    from so101_capstone.environments.simulation_runtime import _environment_bool
    return _environment_bool("SO101_ASSISTED_GRASP_ENABLED", default=False)


def _assisted_grasp_signature() -> dict[str, Any]:
    if not _assisted_grasp_enabled():
        # Omit the new field entirely to preserve unassisted checkpoint contracts.
        return {}
    from so101_capstone.rewards.task_state import GRASP_ATTACHMENT_CONTEXT_DIM
    return {"assisted_grasp": {
        "schema": 1,
        "parent_model": "so101",
        "parent_link": "gripper_link",
        "child_link": "cube_link",
        "release_delta_rad": math.radians(2.0),
        "release_reference": "measured_confirmation",
        "release_requires_measured_opening": True,
        "confirmation": "real_contact_grasp",
        "recorded_angles": "command_and_measured",
        "context_dim": GRASP_ATTACHMENT_CONTEXT_DIM,
    }}


def _execution_signature(config: ActPpoGazeboConfig, reward: Any, *, world_count: int | None = None) -> dict[str, Any]:
    if reward is None:
        return {}
    from so101_capstone.environments.task_measurements import TaskMeasurementConfig
    from so101_capstone.environments.simulation_step_coordinator import SimulationStepCoordinatorConfig
    measurement = TaskMeasurementConfig(**{
        name: getattr(reward, name) for name in (
            "goal_half_extents_xy_m", "goal_yaw_rad", "goal_support_offset_m",
            "source_support_z_m", "target_half_extents_m", "grasp_evidence_mode")
    }, max_sample_gap_s=reward.max_evidence_gap_s)
    stages = _resolved_stage_config(config)
    process_reward = None if stages is None else stages.process_reward
    process_signature = {}
    if process_reward is not None:
        process_signature = {"stage_process_reward": {
            "config": process_reward.to_dict(), "config_hash": process_reward.config_hash,
            "context_schema": process_reward.context_schema_version, "context_dim": process_reward.context_dim,
        }}
        if process_reward.schema_version >= 2:
            process_signature["stage_advantage_normalization"] = "binary_stage_actor_only_v1"
    later_context = stages is not None and stages.enable_later_stage_context
    if later_context:
        from so101_capstone.rewards.later_stage_training import later_stage_training_contract
        from so101_capstone.environments.stage_preparation import StagePreparationConfig
        process_signature["later_stage_training"] = later_stage_training_contract(stages, reward)
        process_signature["stage_preparation"] = asdict(StagePreparationConfig(**(stages.stage_preparation or {})))
        process_signature["stage_preparation_grasp_hold"] = {
            "reference": "certified_command", "offset_rad": 0.0,
        }
        if _assisted_grasp_enabled():
            # Override the general measured-opening release rule only during
            # these policy segments. Preparation still needs to place/release.
            process_signature["stage_grasp_attachment"] = {
                "schema": 1,
                "persistent_stages": ["S2", "S3"],
                "persistent_release_mode": "stage_persistent",
                "opening_commands_detach": False,
                "preparation_release_mode": "measured_opening",
                "other_stage_release_mode": "measured_opening",
            }
        process_signature["stage_advantage_normalization"] = "categorical_six_stage_actor_only_v1"
    return {
        "reward_profile": config.reward_profile, "reward_config": reward.to_dict(),
        "reward_config_hash": reward.config_hash,
        "task_context_dim": config.critic_task_context_dim,
        "task_context_schema": (TASK_CONTEXT_SCHEMA_VERSION + int(reward.reward_schema_version >= 5)
                                + (0 if process_reward is None else process_reward.context_schema_version)
                                + int(later_context)),
        **({"reward_history_context_schema": 1} if reward.reward_schema_version >= 5 else {}),
        **process_signature,
        **_assisted_grasp_signature(),
        "policy_context_schema": CRITIC_POLICY_CONTEXT_SCHEMA_VERSION,
        "chunk_schedule": "fixed_n_action_steps", "policy_hz": config.policy_hz,
        "task_mode": config.task_mode, "scene_seed": config.seed,
        "idle_probabilities": [config.idle_empty_probability, config.idle_completed_probability],
        "randomize_target_delay": config.randomize_target_delay,
        "target_delay_s": config.target_delay_s,
        "max_steps_per_episode": config.max_steps_per_episode,
        **({"stage_training": {
            "selection": config.training_stages, "s0_episode_probability": config.s0_probability,
            "config": stages.to_dict(),
            "boundary": ("s0_s5_stage_reward_sampling_cut_bootstrap_v1" if later_context else
                         "s0_original_s1_process_sampling_cut_bootstrap_v1" if process_reward is not None
                         else "full_objective_sampling_cut_bootstrap_v1"),
            "initialization": ("physical_preparation_certified_handoff_v1" if later_context else
                               "home_hold_settled_target_fresh_camera_v1"),
            "target_sampling": ReachCurriculum().sampling_contract(),
        }} if config.training_stages != "full" else {}),
        "reach_curriculum": {"enabled": config.reach_curriculum, **asdict(_curriculum_config(config)),
            "sampling_contract": ReachCurriculum(_curriculum_config(config)).sampling_contract()},
        "gripper_exploration": {group: getattr(config, f"{group}_gripper_exploration_log_std")
                                for group in ("moving", "manipulation")},
        "exploration": {group: getattr(config, f"{group}_exploration_log_std")
                        for group in ("idle", "moving", "manipulation", "home_hold")},
        **_exploration_profile_signature(config),
        "exploration_enabled": True, "actor_update_enabled": True,
        "measurement_contract": "source_timestamp_joint_target_contact_v1",
        "measurement_config": asdict(measurement),
        "step_coordinator_config": asdict(SimulationStepCoordinatorConfig()),
        "environment_randomization": _randomization_signature(config, world_count=world_count),
        **({"image_augmentation": _image_augmentation_signature(config, world_count=world_count)}
           if os.environ.get("SO101_IMAGE_AUGMENTATION_ENABLED", "false").strip().lower()
           not in {"false", "0", "no", "off", ""} else {}),
    }


def _episode_training_stage(config: ActPpoGazeboConfig, attempt: int, world_index: int = 0) -> str:
    stages = selected_training_stages(config.training_stages)
    if config.training_stages == "s0_s1":
        rng = random.Random(int(config.seed + attempt * 1009 + world_index * 100003))
        return "S0" if rng.random() < config.s0_probability else "S1"
    # Single-world all-stage calls use a deterministic round robin. Multiworld
    # overrides the selection for each actual-step-balanced wave.
    return stages[(max(1, attempt) - 1) % len(stages)]


def _episode_scene(config: ActPpoGazeboConfig, attempt: int, world_index: int = 0) -> tuple[str, int, int]:
    seed = int(config.seed + attempt * 1009 + world_index * 100003)
    rng = random.Random(seed)
    if config.training_stages != "full":
        stage = _episode_training_stage(config, attempt, world_index)
        return ("idle_empty" if stage == "S0" else "pick_place"), 0, seed
    mode = config.task_mode
    if mode == "mixed":
        draw = rng.random()
        mode = ("idle_empty" if draw < config.idle_empty_probability else
                "idle_completed" if draw < config.idle_empty_probability + config.idle_completed_probability else
                "pick_place")
    delay = config.target_delay_steps
    if config.reward_profile == "pick_place_v2" and config.randomize_target_delay:
        delay = rng.randint(0, delay)
    return mode, delay, seed


def _configure_act_replanning(loaded: Any, maximum_steps: int | None) -> None:
    """Cap the native ACT queue; exported deployment uses the same schedule."""
    if maximum_steps is None:
        return
    if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int) or maximum_steps < 1:
        raise ValueError("ACT replanning cap must be a positive integer.")
    validate_act_temporal_config(loaded.config)
    effective = min(int(loaded.config.n_action_steps), maximum_steps)
    loaded.config.n_action_steps = effective
    loaded.policy.config.n_action_steps = effective
    loaded.policy.model.config.n_action_steps = effective
    validate_act_temporal_config(loaded.policy.config)
    loaded.policy.reset()


def _curriculum_config(config: ActPpoGazeboConfig) -> ReachCurriculumConfig:
    return ReachCurriculumConfig(window_size=config.curriculum_window_size,
        pass_rate=config.curriculum_pass_rate, rehearsal_probability=config.curriculum_rehearsal_probability)


def _make_curriculum(config: ActPpoGazeboConfig, resume_payload: Mapping[str, Any] | None = None) -> ReachCurriculum | None:
    if not config.reach_curriculum:
        return None
    curriculum = ReachCurriculum(_curriculum_config(config))
    if resume_payload is not None:
        saved = resume_payload.get("extra", {}).get("collection_state", {}).get("reach_curriculum")
        if saved is None:
            raise ValueError("PPO resume is missing reach curriculum state; use ACT weights-only warm-start.")
        curriculum.load_state_dict(saved)
    return curriculum


def _sync_curriculum_state(trainer: Any, curriculum: ReachCurriculum | None) -> None:
    trainer.collection_state = {} if curriculum is None else {"reach_curriculum": curriculum.state_dict()}


def _sample_curriculum_scene(curriculum: ReachCurriculum | None, *, seed: int,
                             task_mode: str, info: Mapping[str, Any]) -> dict[str, Any]:
    if curriculum is None or task_mode != "pick_place":
        return {}
    # Geometry comes from the measured task, never a guessed goal position.
    return curriculum.sample_target(seed, **{name: info[name] for name in (
        "goal_center_world_m", "goal_half_extents_xy_m", "goal_yaw_rad",
        "target_half_extents_m", "source_support_z_m")})


def _spawn_episode_target(runtime: Any, scene: Mapping[str, Any]) -> Any:
    if not scene:
        return runtime.spawn_random_target()
    return runtime.spawn_target(scene["target_position_world_m"], scene["target_quaternion_xyzw"])


def _accumulate_task_metrics(metrics: dict[str, float], info: Mapping[str, Any]) -> None:
    for key in ("grasp_valid", "lifted_once", "placed_stable", "carried_into_goal", "robot_at_home", "pregrasp_ready"):
        value = info.get(key, False)
        if not isinstance(value, (bool, np.bool_, int, float, np.number)) or value not in (0, 1):
            raise ValueError(f"{key} must be finite boolean evidence (0 or 1).")
        metric = "precise_reach" if key == "pregrasp_ready" else key
        metrics[metric] = max(metrics.get(metric, 0.0), float(value))


def _critic_context(info: Mapping[str, Any], means: Any, index: int, n: int, config: ActPpoGazeboConfig) -> dict[str, np.ndarray]:
    if config.reward_profile == "legacy":
        return {}
    task = np.asarray(info["critic_task_context"], dtype=np.float32)
    if task.shape != (config.critic_task_context_dim,) or not np.isfinite(task).all():
        raise ValueError("Invalid critic task context from environment.")
    return {"critic_task_context": task,
            "critic_policy_context": build_critic_policy_context(means, action_index=index, n_action_steps=n)}


def _critic_value(critic: Any, compact: Any, contexts: dict[str, Any], device: torch.device) -> torch.Tensor:
    kwargs = {name.removeprefix("critic_"): torch.as_tensor(np.array(value, copy=True), device=device).unsqueeze(0)
              for name, value in contexts.items()}
    with torch.no_grad():
        return critic(*_critic_tensors(compact, device=device), **kwargs)


def _accumulate_action_diagnostics(totals: dict[str, Any], phase: str, means: Any, action: Any) -> None:
    diagnostics = policy_action_saturation_diagnostics(means, action)
    group = totals.setdefault(phase, {"sample_count": 0})
    group["sample_count"] += diagnostics.pop("sample_count")
    for name, value in diagnostics.items():
        array = np.asarray(value, dtype=float)
        if name == "command_noise_rms_rad_per_joint":
            array = array ** 2
        group[name] = np.asarray(group.get(name, np.zeros_like(array))) + array


def _finalize_action_diagnostics(totals: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for phase, values in totals.items():
        count = values["sample_count"]
        result[phase] = {"sample_count": count}
        for name, total in values.items():
            if name == "sample_count":
                continue
            value = np.asarray(total) / count
            if name == "command_noise_rms_rad_per_joint":
                value = np.sqrt(value)
            result[phase][name] = value.tolist()
    return result


class RecoverableEpisodeError(RuntimeError):
    """One recoverable failure that invalidates the current episode."""

    def __init__(self, category: str, message: str) -> None:
        normalized = str(category).strip()
        if not normalized:
            raise ValueError("Recoverable error category must not be empty.")
        self.category = normalized
        super().__init__(str(message))


@dataclass
class EpisodeFailureTracker:
    """Track attempts and enforce a consecutive recoverable-failure limit."""

    maximum_consecutive_failures: int
    attempt_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        value = self.maximum_consecutive_failures
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                "maximum_consecutive_failures must be a positive integer."
            )

    def start_attempt(self) -> int:
        self.attempt_count += 1
        return self.attempt_count

    def record_failure(self) -> bool:
        self.failure_count += 1
        self.consecutive_failures += 1
        return (
            self.consecutive_failures
            >= self.maximum_consecutive_failures
        )

    def record_success(self) -> None:
        self.consecutive_failures = 0


_RECOVERABLE_EPISODE_EXCEPTIONS = (
    RecoverableEpisodeError,
    EpisodeResetError,
    GazeboWorldControlError,
    ReachEnvironmentStateError,
    SimulationRuntimeError,
    SimulationStepCoordinatorError,
    TargetCubeControlError,
)


def _recoverable_episode_category(error: BaseException) -> str:
    """Convert a whitelisted runtime exception to a stable log category."""

    if isinstance(error, RecoverableEpisodeError):
        return error.category
    if isinstance(error, EpisodeResetError):
        return "episode_reset"
    if isinstance(error, TargetCubeControlError):
        return "target_service"
    if isinstance(error, GazeboWorldControlError):
        return "gazebo_control_or_timing"
    if isinstance(error, ReachEnvironmentStateError):
        return "environment_feedback"
    if isinstance(error, SimulationRuntimeError):
        return "simulation_runtime"
    if isinstance(error, SimulationStepCoordinatorError):
        message = str(error).lower()
        if "camera" in message or "image" in message:
            return "camera"
        if "joint" in message:
            return "joint_feedback"
        if (
            "clock" in message
            or "simulation-time" in message
            or "simulation time" in message
            or "sim_step" in message
            or "physics-step" in message
        ):
            return "gazebo_deterministic_timing"
        return "simulation_step"
    raise TypeError(
        f"Unsupported recoverable episode exception: {type(error).__name__}."
    )


def _map_policy_positions_for_episode(
    adapter: Any,
    observation: Any,
    policy_positions: np.ndarray,
) -> Any:
    """Convert ACT output while isolating action-safety validation failures."""

    try:
        return adapter.map_policy_positions(
            observation,
            policy_positions,
        )
    except (ValueError, FloatingPointError) as error:
        raise RecoverableEpisodeError(
            "act_action_safety",
            str(error),
        ) from error


def _cleanup_failed_episode(
    runtime: Any,
    *,
    attempt_index: int,
) -> dict[str, Any]:
    """Best-effort target cleanup before the next episode attempt."""

    stage = f"after failed episode attempt {attempt_index}"
    try:
        return _delete_payload(
            runtime.delete_all_targets(),
            stage=stage,
        )
    except _RECOVERABLE_EPISODE_EXCEPTIONS as error:
        return {
            "requested": True,
            "stage": stage,
            "service_success": False,
            "failure_category": _recoverable_episode_category(error),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        }


def _set_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _spawn_payload(response: Any, *, completed_step: int) -> dict[str, Any]:
    success = bool(getattr(response, "success", True))
    message = str(getattr(response, "message", ""))
    if not success:
        raise RecoverableEpisodeError(
            "target_service",
            f"Target spawn service rejected request: {message}",
        )
    return {
        "requested": True,
        "request_completed_step": completed_step,
        "service_success": success,
        "service_message": message,
    }


def _delete_payload(response: Any, *, stage: str) -> dict[str, Any]:
    success = bool(getattr(response, "success", True))
    message = str(getattr(response, "message", ""))
    if not success:
        raise RecoverableEpisodeError(
            "target_service",
            f"Target delete-all rejected cleanup {stage}: {message}",
        )
    return {
        "requested": True,
        "stage": stage,
        "service_success": success,
        "service_message": message,
    }


def _prepare_clean_episode(
    runtime: Any,
    *,
    episode_index: int,
    episode_attempt_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    cleanup = _delete_payload(
        runtime.delete_all_targets(),
        stage=f"before episode {episode_index}",
    )
    prepared = (runtime.prepare_episode() if episode_attempt_index is None else
                runtime.prepare_episode(environment_randomization_episode_key=episode_attempt_index))
    return prepared, cleanup


def _critic_tensors(
    observation: Any,
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        torch.from_numpy(np.asarray(observation.top_image).copy())
        .unsqueeze(0)
        .to(device=device),
        torch.from_numpy(
            np.asarray(observation.gripper_image).copy()
        )
        .unsqueeze(0)
        .to(device=device),
        torch.from_numpy(
            np.asarray(observation.joint_positions).copy()
        )
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
        torch.from_numpy(
            np.asarray(observation.joint_velocities).copy()
        )
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
    )


def _reward_state(info: Mapping[str, Any]) -> RewardState:
    try:
        return RewardState(int(info["reward_state"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Step info must contain a valid reward_state."
        ) from error


def _stage_name(info: Mapping[str, Any]) -> str:
    """Return the configured PPO stage name for one environment state."""

    training_stage = str(info.get("training_stage", "")).upper()
    if training_stage in {"S2", "S3", "S4", "S5"}:
        if training_stage == "S5":
            return "home_hold" if bool(info.get("stage_robot_at_home", info.get("robot_at_home", False))) else "moving"
        return {"S2": "manipulation", "S3": "moving", "S4": "manipulation"}[training_stage]
    if "task_phase" in info:
        phase = str(info.get("task_phase_name", info["task_phase"])).upper()
        groups = {"WAIT_HOME": "idle", "APPROACH": "moving", "GRASP_LIFT": "manipulation",
                  "TRANSPORT": "moving", "PLACE_RELEASE": "manipulation",
                  "RETURN_HOME": "moving", "HOME_HOLD": "home_hold"}
        if phase not in groups:
            raise ValueError(f"Unknown task phase: {phase}")
        return groups[phase]
    state = _reward_state(info)
    if state is RewardState.TARGET_ABSENT:
        return "state1"
    if state is RewardState.TARGET_APPROACH:
        return "state2"
    if state is RewardState.TARGET_GRASP_TRANSPORT:
        return "state3"
    if state is RewardState.TARGET_RELEASE_RETURN_HOME:
        if bool(info.get("robot_at_home", False)):
            return "state4_home"
        return "state4_not_home"
    raise ValueError(f"Unsupported reward state: {state!r}")


def _exploration_enabled(
    info: Mapping[str, Any],
    config: ActPpoGazeboConfig,
) -> bool:
    """Return whether rollout actions are sampled stochastically."""

    if config.reward_profile == "pick_place_v2":
        return True
    return bool(getattr(config, f"{_stage_name(info)}_exploration_enabled"))


def _actor_update_enabled(
    info: Mapping[str, Any],
    config: ActPpoGazeboConfig,
) -> bool:
    """Return whether this stage contributes to the PPO actor objective."""

    if config.reward_profile == "pick_place_v2":
        return True
    return _exploration_enabled(info, config) and bool(
        getattr(config, f"{_stage_name(info)}_actor_update_enabled")
    )


def _exploration_stage(
    info: Mapping[str, Any],
    config: ActPpoGazeboConfig,
) -> tuple[str, float, float | np.ndarray]:
    """Return stage name, initial log std, and shared-policy offset."""

    _validate_stage_exploration_config(config)
    name = _stage_name(info)
    if config.exploration_profile in {"final_good_act", "final_good_act_near_target"}:
        from .act_ppo_exploration_profile import (
            final_good_act_exploration_offset,
            final_good_act_near_target_exploration_offset,
        )
        selector = (final_good_act_near_target_exploration_offset
                    if config.exploration_profile == "final_good_act_near_target"
                    else final_good_act_exploration_offset)
        offset = selector(
            info, stage_name=name, initial_log_std=config.initial_log_std,
        )
        return name, float(config.initial_log_std + np.mean(offset)), offset
    initial_log_std = float(getattr(config, f"{name}_exploration_log_std"))
    offset = initial_log_std - float(config.initial_log_std)
    gripper = getattr(config, f"{name}_gripper_exploration_log_std", None)
    if config.reward_profile == "pick_place_v2" and gripper is not None:
        offset = np.full(6, offset, dtype=np.float32)
        offset[-1] = float(gripper) - float(config.initial_log_std)
    return name, initial_log_std, offset


def _should_save_update(
    update_index: int,
    checkpoint_every_updates: int,
) -> bool:
    """Return whether this PPO update should write an intermediate checkpoint."""

    for name, value in (
        ("update_index", update_index),
        ("checkpoint_every_updates", checkpoint_every_updates),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    return update_index % checkpoint_every_updates == 0


def _copy_reward_config_for_export(output_directory: Path, standalone: Path) -> None:
    resolved = output_directory / "resolved_reward.json"
    if resolved.is_file():
        standalone.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved, standalone / "resolved_reward.json")
    stages = output_directory / "resolved_stages.json"
    if stages.is_file():
        standalone.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stages, standalone / stages.name)


def _save_update(
    trainer: ActPpoTrainer,
    *,
    output_directory: Path,
    name: str,
    config: ActPpoGazeboConfig,
    metrics: dict[str, Any],
) -> dict[str, str]:
    directory = output_directory / "checkpoints" / name
    directory.mkdir(parents=True, exist_ok=False)
    training_state = trainer.save_checkpoint(
        directory / "training_state.pt",
        extra={
            "act_checkpoint": str(config.act_checkpoint.resolve()),
            "gazebo_config": config.to_json_dict(),
            "collection_state": getattr(trainer, "collection_state", {}),
            "metrics": metrics,
        },
    )
    standalone = directory / "pretrained_model"
    trainer.policy.save_pretrained(standalone)
    _copy_reward_config_for_export(output_directory, standalone)
    return {
        "training_state": str(training_state),
        "standalone_act": str(standalone),
    }


def _export_final(
    trainer: ActPpoTrainer,
    *,
    output_directory: Path,
    config: ActPpoGazeboConfig,
    metrics: dict[str, Any] | None,
) -> tuple[Path, Path]:
    standalone = output_directory / "pretrained_model"
    trainer.policy.save_pretrained(standalone)
    _copy_reward_config_for_export(output_directory, standalone)
    state = trainer.save_checkpoint(
        output_directory / "final_training_state.pt",
        extra={
            "act_checkpoint": str(config.act_checkpoint.resolve()),
            "gazebo_config": config.to_json_dict(),
            "collection_state": getattr(trainer, "collection_state", {}),
            "metrics": metrics,
            "final_standalone_act": str(standalone),
        },
    )
    return standalone, state


def run_act_ppo_gazebo(
    config: ActPpoGazeboConfig,
    *,
    checkpoint_loader: Callable[..., Any] = load_act_checkpoint,
    runtime_factory: Callable[..., Any] = create_ros_simulation_runtime,
    adapter_factory: Callable[..., Any] = ActPolicyAdapter,
) -> dict[str, Any]:
    reward = _resolved_reward(config)
    stage_config = _resolved_stage_config(config)
    execution_signature = _execution_signature(config, reward)
    _set_seed(config.seed)
    output_directory = (
        _default_output_directory(config.act_checkpoint)
        if config.output_directory is None
        else config.output_directory
    ).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    tensorboard = DirectActPpoTensorBoard.disabled()
    if config.tensorboard_enabled:
        tensorboard = DirectActPpoTensorBoard.start(
            output_directory / "tensorboard",
            preferred_port=config.tensorboard_port,
            auto_open=config.tensorboard_auto_open,
        )

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    loaded = checkpoint_loader(config.act_checkpoint, device=device)
    _configure_act_replanning(loaded, config.act_max_action_steps)
    policy = loaded.policy
    action_metadata = validate_act_temporal_config(loaded.config)
    temporal_coefficient = action_metadata["temporal_ensemble_coeff"]
    if temporal_coefficient is not None:
        execution_signature.update(action_metadata)

    policy.eval()
    policy.reset()

    adapter = adapter_factory(
        policy,
        device=loaded.device,
        max_joint_delta_rad=_DISABLED_DELTA_SENTINEL_RAD,
    )
    ppo_config = _act_ppo_config(config, int(loaded.config.n_action_steps))
    critic = ActPpoValueNetwork(ppo_config).to(device)
    trainer = ActPpoTrainer(policy, critic, config=ppo_config,
                            **({"execution_signature": execution_signature} if reward else {}))
    resume_payload = None
    if config.resume_checkpoint is not None:
        resume_payload = trainer.load_checkpoint(
            config.resume_checkpoint,
            map_location=device,
        )
    curriculum = _make_curriculum(config, resume_payload)
    _sync_curriculum_state(trainer, curriculum)
    initial_fingerprint = model_state_sha256(policy)

    if reward is not None:
        _write_json(output_directory / "resolved_reward.json", reward.to_dict())
    if stage_config is not None:
        _write_json(output_directory / "resolved_stages.json", stage_config.to_dict())
    _write_json(
        output_directory / "config.json",
        {
            "gazebo": config.to_json_dict(),
            "ppo": asdict(ppo_config),
            "resolved_reward": None if reward is None else reward.to_dict(),
            "execution_signature": execution_signature,
            "checkpoint_act_config": {
                **action_metadata,
                "action_bounds_lower": loaded.config.action_bounds_lower,
                "action_bounds_upper": loaded.config.action_bounds_upper,
                "action_bound_softness": loaded.config.action_bound_softness,
                "optimizer_lr": loaded.config.optimizer_lr,
                "optimizer_lr_backbone": (
                    loaded.config.optimizer_lr_backbone
                ),
                "optimizer_weight_decay": (
                    loaded.config.optimizer_weight_decay
                ),
                "use_vae": loaded.config.use_vae,
                "vision_backbone": loaded.config.vision_backbone,
            },
            "resolved_actor_learning_rate": (
                trainer.actor_learning_rate
            ),
            "resolved_actor_backbone_learning_rate": (
                trainer.actor_backbone_learning_rate
            ),
            "deployment_artifact": "standalone ACT pretrained_model",
            "critic_is_training_only": True,
            "exploration_is_training_only": True,
            "max_joint_delta_limit_enabled": False,
            "absolute_joint_limits_enabled": True,
        },
    )

    rollout = ActPpoRolloutBuffer(ppo_config)
    episodes: list[dict[str, Any]] = []
    failed_episodes: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    failure_tracker = EpisodeFailureTracker(
        config.max_consecutive_episode_failures
    )
    if curriculum is not None:
        failure_tracker.attempt_count = curriculum.attempt_count
    if stage_config is not None and resume_payload is not None:
        count = resume_payload.get("extra", {}).get("collection_state", {}).get("stage_attempt_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Stage resume requires a valid stage_attempt_count; use ACT weights-only warm-start.")
        failure_tracker.attempt_count = count
    if stage_config is not None:
        trainer.collection_state["stage_attempt_count"] = failure_tracker.attempt_count
    total_steps = 0
    discarded_environment_steps = 0
    final_cleanup = None
    runtime_config = SimulationRuntimeConfig(
        startup_feedback_timeout_s=(
            config.startup_feedback_timeout_s
        )
    )

    with runtime_factory(
        runtime_config=runtime_config,
        environment_config=ReachEnvironmentConfig(
            max_episode_steps=config.max_steps_per_episode,
        ),
        **_runtime_reward_kwargs(config, reward),
    ) as runtime:
        completed_episode_count = 0
        while completed_episode_count < config.episode_count:
            episode_index = completed_episode_count
            episode_attempt_index = failure_tracker.start_attempt()
            if stage_config is not None:
                trainer.collection_state["stage_attempt_count"] = episode_attempt_index
            if curriculum is not None:
                curriculum.attempt_count = episode_attempt_index
            rollout_checkpoint = rollout.checkpoint()
            total_steps_before_episode = total_steps
            completed_steps = 0
            try:
                task_mode, spawn_delay_steps, scene_seed = _episode_scene(config, episode_attempt_index)
                if config.reward_profile == "pick_place_v2":
                    runtime.configure_task(task_mode, seed=scene_seed, n_action_steps=int(loaded.config.n_action_steps))
                prepared, cleanup = _prepare_clean_episode(
                    runtime,
                    episode_index=episode_index,
                    **({"episode_attempt_index": episode_attempt_index} if stage_config is not None else {}),
                )
                policy.reset()
                observation = prepared.observation
                completed_steps = 0
                episode_reward = 0.0
                terminated = False
                truncated = False
                last_info = dict(prepared.info)
                curriculum_scene = _sample_training_scene(config, curriculum, seed=scene_seed,
                    task_mode=task_mode, info=last_info)
                exploration_enabled_steps = 0
                exploration_disabled_steps = 0
                actor_update_enabled_steps = 0
                actor_update_disabled_steps = 0
                exploration_mode_switches = 0
                spawn = {
                    "requested": False,
                    "request_completed_step": None,
                    "service_success": None,
                    "service_message": None,
                }
                tail_target_cleanup = {
                    "requested": False,
                    "request_completed_step": None,
                    "stage": "final target-absent window",
                    "service_success": None,
                    "service_message": None,
                }
                if spawn_delay_steps == 0 and task_mode == "pick_place":
                    spawn = _spawn_payload(
                        _spawn_episode_target(runtime, curriculum_scene),
                        completed_step=0,
                    )

                training_stage = None
                if stage_config is not None:
                    training_stage = _episode_training_stage(config, episode_attempt_index)
                    prepared = runtime.start_training_stage(training_stage, stage_config.to_dict())
                    observation, last_info = prepared.observation, dict(prepared.info)
                stage_initialization = last_info.get("stage_initialization")
                episode_step_limit = _episode_step_limit(config, task_mode, training_stage)

                means = None
                actor_reference = None
                action_index = 0
                temporal_history = (
                    ActPpoTemporalHistory(temporal_coefficient, int(loaded.config.chunk_size))
                    if temporal_coefficient is not None else None
                )
                chunk_stage = None
                task_metrics: dict[str, float] = {}
                reward_totals: dict[str, float] = {}
                action_diagnostics: dict[str, Any] = {}
                while completed_steps < episode_step_limit:
                    if means is None:
                        actor_observation = actor_observation_from_batch(adapter.build_batch(observation))
                        actor_reference = rollout.add_actor_observation(actor_observation)
                        with torch.no_grad():
                            means = predict_action_chunk_with_grad(
                                policy, actor_observation_tensors(actor_observation, device=device))
                            if temporal_history is not None:
                                means = temporal_history.push(actor_reference, means)
                        action_index = 0
                    # Freeze this step's history before next-state bootstrap advances it.
                    temporal_actor_references = (
                        temporal_history.actor_references if temporal_history is not None else None
                    )
                    exploration_stage, _, exploration_log_std_offset = _exploration_stage(last_info, config)
                    exploration_enabled = _exploration_enabled(last_info, config)
                    actor_update_enabled = _actor_update_enabled(last_info, config)
                    if chunk_stage is not None and chunk_stage != exploration_stage:
                        exploration_mode_switches += 1
                    chunk_stage = exploration_stage
                    exploration_log_std = trainer.exploration.with_offset(exploration_log_std_offset)
                    with torch.no_grad():
                        sampled, old_log_probabilities, sampled_latents = sample_bounded_policy_actions(
                            means[:, action_index:action_index + 1], exploration_log_std,
                            deterministic=not exploration_enabled, return_latent=True)
                    compact = compress_policy_observation(observation,
                        image_height=ppo_config.critic_image_height, image_width=ppo_config.critic_image_width)
                    contexts = _critic_context(last_info, means, action_index, int(loaded.config.n_action_steps), config)
                    value = _critic_value(critic, compact, contexts, device)
                    sampled_policy_action = sampled[0, 0].detach().cpu().double().numpy()
                    _accumulate_action_diagnostics(action_diagnostics, exploration_stage, means[:, action_index:action_index+1], sampled_policy_action)
                    policy_result = _map_policy_positions_for_episode(adapter, observation, sampled_policy_action)
                    step_result = runtime.step(policy_result.action.controller_command())
                    completed_steps += 1
                    total_steps += 1
                    exploration_enabled_steps += int(exploration_enabled)
                    exploration_disabled_steps += int(not exploration_enabled)
                    actor_update_enabled_steps += int(actor_update_enabled)
                    actor_update_disabled_steps += int(not actor_update_enabled)
                    episode_reward += float(step_result.reward)
                    terminated, truncated = bool(step_result.terminated), bool(step_result.truncated)
                    episode_end = terminated or truncated or completed_steps >= episode_step_limit
                    target_spawned_this_step = False
                    if (not episode_end and task_mode == "pick_place" and not spawn["requested"]
                            and completed_steps == spawn_delay_steps):
                        spawn = _spawn_payload(_spawn_episode_target(runtime, curriculum_scene), completed_step=completed_steps)
                        target_spawned_this_step = True
                    tail_target_deleted_this_step = False
                    if (not episode_end and config.final_target_absent_steps > 0
                            and not tail_target_cleanup["requested"]
                            and completed_steps == config.final_target_delete_step):
                        tail_target_cleanup = _delete_payload(runtime.delete_all_targets(), stage="final target-absent window")
                        tail_target_cleanup["request_completed_step"] = completed_steps
                        tail_target_deleted_this_step = True
                    next_observation = step_result.observation
                    next_info = dict(step_result.info)
                    next_stage, _, _ = _exploration_stage(next_info, config)
                    legacy_refresh = config.reward_profile == "legacy" and (
                        next_stage != exploration_stage or target_spawned_this_step or tail_target_deleted_this_step)
                    next_index = action_index + 1
                    next_means, next_reference = means, actor_reference
                    if not terminated and (next_index >= int(loaded.config.n_action_steps) or legacy_refresh):
                        next_actor_observation = actor_observation_from_batch(adapter.build_batch(next_observation))
                        with torch.no_grad():
                            next_means = predict_action_chunk_with_grad(
                                policy, actor_observation_tensors(next_actor_observation, device=device))
                        next_index = 0
                        # A truncated hypothetical continuation has no unexecuted actor transition.
                        next_reference = None if episode_end else rollout.add_actor_observation(next_actor_observation)
                        if temporal_history is not None:
                            with torch.no_grad():
                                next_means = temporal_history.push(next_reference, next_means)
                    next_value = 0.0
                    if not terminated:
                        next_compact = compress_policy_observation(next_observation,
                            image_height=ppo_config.critic_image_height, image_width=ppo_config.critic_image_width)
                        next_contexts = _critic_context(next_info, next_means, next_index, int(loaded.config.n_action_steps), config)
                        next_value = float(_critic_value(critic, next_compact, next_contexts, device).item())
                    rollout.add(compact, actor_observation_index=actor_reference, action_index=action_index,
                        temporal_actor_references=temporal_actor_references,
                        policy_action=sampled_policy_action, policy_action_latent=sampled_latents[0, 0].detach().cpu().numpy(),
                        policy_action_mean=means[0, action_index].detach().cpu().numpy(),
                        policy_action_log_std=exploration_log_std.detach().cpu().numpy(),
                        log_probability=float(old_log_probabilities[0, 0].item()),
                        actor_update_enabled=actor_update_enabled, exploration_log_std_offset=exploration_log_std_offset,
                        value=float(value.item()), reward=float(step_result.reward), next_value=next_value,
                        terminated=terminated, episode_end=episode_end, **contexts)
                    for key, component in next_info.get("reward_components", {}).items():
                        if isinstance(component, (int, float)):
                            reward_totals[key] = reward_totals.get(key, 0.0) + float(component)
                    _accumulate_task_metrics(task_metrics, next_info)
                    observation, last_info = next_observation, next_info
                    means, actor_reference, action_index = next_means, next_reference, next_index

                    if episode_end and curriculum is not None:
                        curriculum.observe({"valid": True, "task_mode": task_mode,
                            "episode_attempt_index": episode_attempt_index,
                            "task_success": last_info.get("task_outcome") == "success",
                            "stage_reached": task_metrics, "final_info": last_info, **curriculum_scene})
                        _sync_curriculum_state(trainer, curriculum)
                        _append_jsonl(output_directory / "metrics.jsonl",
                            {"type": "curriculum", "episode_attempt_index": episode_attempt_index,
                             **curriculum.diagnostics()})

                    if episode_end and (len(rollout) >= config.rollout_steps):
                        metrics = trainer.update(rollout)
                        payload = {
                            "type": "ppo_update",
                            "update_index": trainer.update_index,
                            "partial_rollout": False,
                            **metrics.to_dict(),
                        }
                        if _should_save_update(
                            trainer.update_index,
                            config.checkpoint_every_updates,
                        ):
                            payload["checkpoint_saved"] = True
                            payload.update(
                                _save_update(
                                    trainer,
                                    output_directory=output_directory,
                                    name=f"update_{trainer.update_index:06d}",
                                    config=config,
                                    metrics=payload,
                                )
                            )
                        else:
                            payload["checkpoint_saved"] = False
                        updates.append(payload)
                        tensorboard.log_update(
                            payload,
                            total_environment_steps=(
                                trainer.total_environment_steps
                            ),
                        )
                        _append_jsonl(
                            output_directory / "metrics.jsonl",
                            payload,
                        )
                        rollout.clear()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    if (
                        terminated
                        or truncated
                        or completed_steps >= episode_step_limit
                    ):
                        break

                episode_record = {
                    "episode_index": episode_index,
                    "episode_attempt_index": episode_attempt_index,
                    **curriculum_scene,
                    "valid": True,
                    "consecutive_failures_before_success": (
                        failure_tracker.consecutive_failures
                    ),
                    "step_count": completed_steps,
                    "task_mode": task_mode,
                    "task_success": last_info.get("task_outcome") == "success",
                    **({"stage_initialization": stage_initialization} if stage_initialization is not None else {}),
                    **({key: value for key, value in last_info.items()
                        if key.startswith("stage_") or key in ("training_stage", "start_certificate",
                            "assisted_initialization", "full_unassisted_success")}
                       if stage_config is not None else {}),
                    "stage_reached": task_metrics,
                    "reward_components_total": reward_totals,
                    "action_diagnostics": _finalize_action_diagnostics(action_diagnostics),
                    "total_reward": episode_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "reached_max_steps": (
                        completed_steps >= episode_step_limit
                        and not terminated
                        and not truncated
                    ),
                    "target": spawn,
                    "tail_target_cleanup": tail_target_cleanup,
                    "cleanup_before_episode": cleanup,
                    "final_info": last_info,
                    "rollout_buffer_size_after_episode": len(rollout),
                    "exploration_enabled_step_count": (
                        exploration_enabled_steps
                    ),
                    "exploration_disabled_step_count": (
                        exploration_disabled_steps
                    ),
                    "exploration_mode_switch_count": (
                        exploration_mode_switches
                    ),
                    "actor_update_enabled_step_count": (
                        actor_update_enabled_steps
                    ),
                    "actor_update_disabled_step_count": (
                        actor_update_disabled_steps
                    ),
                    "exploration_all_states_enabled": all(
                        (
                            config.state1_exploration_enabled,
                            config.state2_exploration_enabled,
                            config.state3_exploration_enabled,
                            config.state4_not_home_exploration_enabled,
                            config.state4_home_exploration_enabled,
                        )
                    ),
                    "actor_update_all_states_enabled": all(
                        (
                            config.state1_actor_update_enabled,
                            config.state2_actor_update_enabled,
                            config.state3_actor_update_enabled,
                            config.state4_not_home_actor_update_enabled,
                            config.state4_home_actor_update_enabled,
                        )
                    ),
                    "state1_exploration_enabled": (
                        config.state1_exploration_enabled
                    ),
                    "state2_exploration_enabled": (
                        config.state2_exploration_enabled
                    ),
                    "state4_not_home_exploration_enabled": (
                        config.state4_not_home_exploration_enabled
                    ),
                    "state4_home_exploration_enabled": (
                        config.state4_home_exploration_enabled
                    ),
                    "state1_actor_update_enabled": (
                        config.state1_actor_update_enabled
                    ),
                    "state2_actor_update_enabled": (
                        config.state2_actor_update_enabled
                    ),
                    "state4_not_home_actor_update_enabled": (
                        config.state4_not_home_actor_update_enabled
                    ),
                    "state4_home_actor_update_enabled": (
                        config.state4_home_actor_update_enabled
                    ),
                    "state1_initial_log_std": (
                        config.state1_exploration_log_std
                    ),
                    "state2_initial_log_std": (
                        config.state2_exploration_log_std
                    ),
                    "state4_not_home_initial_log_std": (
                        config.state4_not_home_exploration_log_std
                    ),
                    "state4_home_initial_log_std": (
                        config.state4_home_exploration_log_std
                    ),
                }
                if reward is not None:
                    episode_record = {key: value for key, value in episode_record.items()
                                      if not key.startswith("state") and "all_states" not in key}
                    episode_record["exploration_groups"] = execution_signature["exploration"]
                    episode_record["exploration_enabled"] = True
                    episode_record["actor_update_enabled"] = True
                episodes.append(episode_record)
                tensorboard.log_episode(episode_record)
                _write_json(
                    output_directory
                    / "episodes"
                    / f"episode_{episode_index:06d}.json",
                    episode_record,
                )
                _append_jsonl(
                    output_directory / "metrics.jsonl",
                    {"type": "episode", **episode_record},
                )
            except _RECOVERABLE_EPISODE_EXCEPTIONS as error:
                discarded_transition_count = rollout.rollback(
                    rollout_checkpoint
                )
                total_steps = total_steps_before_episode
                discarded_environment_steps += completed_steps
                policy.reset()
                cleanup_after_failure = _cleanup_failed_episode(
                    runtime,
                    attempt_index=episode_attempt_index,
                )
                failure_category = _recoverable_episode_category(error)
                abort_training = failure_tracker.record_failure()
                failure_record = {
                    "episode_index": episode_index,
                    "episode_attempt_index": episode_attempt_index,
                    "valid": False,
                    "failed_step": completed_steps,
                    "discarded_transition_count": (
                        discarded_transition_count
                    ),
                    "failure_category": failure_category,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "consecutive_episode_failures": (
                        failure_tracker.consecutive_failures
                    ),
                    "maximum_consecutive_episode_failures": (
                        config.max_consecutive_episode_failures
                    ),
                    "cleanup_after_failure": cleanup_after_failure,
                }
                failed_episodes.append(failure_record)
                _write_json(
                    output_directory
                    / "episode_failures"
                    / f"attempt_{episode_attempt_index:06d}.json",
                    failure_record,
                )
                _append_jsonl(
                    output_directory / "metrics.jsonl",
                    {"type": "episode_failure", **failure_record},
                )
                print(
                    "[RECOVERABLE EPISODE FAILURE] "
                    f"attempt={episode_attempt_index} "
                    f"episode={episode_index} "
                    f"category={failure_category} "
                    f"step={completed_steps} "
                    f"discarded={discarded_transition_count} "
                    "consecutive="
                    f"{failure_tracker.consecutive_failures}/"
                    f"{config.max_consecutive_episode_failures}: "
                    f"{error}"
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if abort_training:
                    raise RuntimeError(
                        "Direct ACT PPO stopped after "
                        f"{failure_tracker.consecutive_failures} "
                        "consecutive recoverable episode failures. "
                        "The latest completed checkpoint remains available."
                    ) from error
                continue

            failure_tracker.record_success()
            completed_episode_count += 1

        final_cleanup = _delete_payload(
            runtime.delete_all_targets(),
            stage="after final episode",
        )

    if len(rollout) > 0:
        if len(rollout) < 2:
            raise RuntimeError(
                "Final partial rollout has fewer than two transitions."
            )
        metrics = trainer.update(rollout)
        payload = {
            "type": "ppo_update",
            "update_index": trainer.update_index,
            "partial_rollout": True,
            **metrics.to_dict(),
        }
        if _should_save_update(
            trainer.update_index,
            config.checkpoint_every_updates,
        ):
            payload["checkpoint_saved"] = True
            payload.update(
                _save_update(
                    trainer,
                    output_directory=output_directory,
                    name=f"update_{trainer.update_index:06d}",
                    config=config,
                    metrics=payload,
                )
            )
        else:
            payload["checkpoint_saved"] = False
        updates.append(payload)
        tensorboard.log_update(
            payload,
            total_environment_steps=(
                trainer.total_environment_steps
            ),
        )
        _append_jsonl(
            output_directory / "metrics.jsonl",
            payload,
        )
        rollout.clear()

    final_fingerprint = model_state_sha256(policy)
    if final_fingerprint == initial_fingerprint:
        raise RuntimeError(
            "PPO completed without changing ACT parameters. "
            "The standalone ACT was not exported."
        )

    final_act, final_training_state = _export_final(
        trainer,
        output_directory=output_directory,
        config=config,
        metrics=updates[-1] if updates else None,
    )

    reloaded = checkpoint_loader(final_act, device="cpu")
    if (model_state_sha256(reloaded.policy) != final_fingerprint
            or actor_forward_contract(reloaded.policy.config) != actor_forward_contract(policy.config)):
        raise RuntimeError(
            "Reloaded standalone ACT weights or forward configuration do not match trained ACT."
        )

    summary = {
        "output_directory": str(output_directory),
        "final_act_checkpoint": str(final_act),
        "final_training_state": str(final_training_state),
        "episode_count": len(episodes),
        "valid_episode_count": len(episodes),
        "pick_place_episode_count": sum(e.get("task_mode") == "pick_place" for e in episodes),
        "task_success_count": sum(bool(e.get("task_success")) for e in episodes),
        "episode_attempt_count": failure_tracker.attempt_count,
        "failed_episode_attempt_count": (
            failure_tracker.failure_count
        ),
        "maximum_consecutive_episode_failures": (
            config.max_consecutive_episode_failures
        ),
        "discarded_environment_steps": (
            discarded_environment_steps
        ),
        "total_collected_steps": total_steps,
        "ppo_update_count": len(updates),
        "checkpoint_every_updates": (
            config.checkpoint_every_updates
        ),
        "saved_update_checkpoint_count": sum(
            bool(update.get("checkpoint_saved", False))
            for update in updates
        ),
        "act_state_sha256_before": initial_fingerprint,
        "act_state_sha256_after": final_fingerprint,
        "act_parameters_changed": (
            initial_fingerprint != final_fingerprint
        ),
        "standalone_act_reload_verified": True,
        "critic_exported_with_deployment_act": False,
        "exploration_exported_with_deployment_act": False,
        **action_metadata,
        "tensorboard_enabled": config.tensorboard_enabled,
        "tensorboard_log_directory": (
            tensorboard.log_directory_string
        ),
        "tensorboard_url": tensorboard.url,
        "max_joint_delta_limit_enabled": False,
        "absolute_joint_limits_enabled": True,
        "final_target_cleanup": final_cleanup,
        "episodes": episodes,
        "failed_episodes": failed_episodes,
        "updates": updates,
    }
    _write_json(output_directory / "summary.json", summary)
    tensorboard.close()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use PPO to update ACT directly and export a standalone ACT."
        )
    )
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--exploration-profile", choices=("stage_log_std", "final_good_act", "final_good_act_near_target"),
                        default="stage_log_std",
                        help="Use fixed normalized stage noise or distance-based physical-angle noise.")
    parser.add_argument("--training-stages", choices=("full", *TRAINING_STAGE_SELECTIONS), default="full")
    parser.add_argument("--stage-initialization-timeout-s", type=float,
                        help="Preparation wall-clock timeout; default 180s for S0/S1, 600s with later stages.")
    for stage_index in (2, 3, 4):
        parser.add_argument(f"--s{stage_index}-budget-s", type=float,
                            help="Stage duration in simulation seconds; S5 shares S0 budget.")
    parser.add_argument("--stage-config", type=Path)
    parser.add_argument("--s0-probability", type=float, default=1.0 / 3.0,
                        help="S0 episode probability in s0_s1; with 5s/10s budgets, 1/3 gives about 20%% steps if both run out.")
    parser.add_argument("--s0-budget-s", type=float,
                        help="S0 observation window in simulation seconds; overrides stage JSON.")
    parser.add_argument("--s0-home-hold-s", type=float,
                        help="Required trailing continuous HOME time in seconds; overrides stage JSON.")
    parser.add_argument("--s1-budget-s", type=float,
                        help="S1 maximum duration in simulation seconds; overrides stage JSON.")
    parser.add_argument("--s1-reach-hold-s", type=float,
                        help="Required continuous reach dwell in seconds; overrides stage JSON.")
    parser.add_argument("--act-max-action-steps", type=int,
                        help="Cap ACT's native execution queue; also saved in the exported ACT config.")
    parser.add_argument("--reach-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--curriculum-window-size", type=int, default=20)
    parser.add_argument("--curriculum-pass-rate", type=float, default=0.7)
    parser.add_argument("--curriculum-rehearsal-probability", type=float, default=0.2)
    parser.add_argument("--moving-gripper-exploration-log-std", type=float)
    parser.add_argument("--manipulation-gripper-exploration-log-std", type=float)
    parser.add_argument("--gripper-maximum-log-std", type=float)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--episode-count",
        type=int,
        default=DEFAULT_EPISODE_COUNT,
    )
    parser.add_argument(
        "--max-consecutive-episode-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_EPISODE_FAILURES,
        help=(
            "Invalidate and retry recoverable episode failures; stop the "
            "whole finetune when this many occur consecutively."
        ),
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_EPISODE,
    )
    parser.add_argument(
        "--target-delay-s",
        type=float,
        default=DEFAULT_TARGET_DELAY_S,
    )
    parser.add_argument(
        "--final-target-absent-s",
        type=float,
        default=0.0,
        help=(
            "Delete all target cubes this many seconds before "
            "the configured episode horizon, leaving a final "
            "no-target State1 training window. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
    )
    parser.add_argument(
        "--critic-image-height",
        type=int,
        default=96,
        help="Training-only critic image height.",
    )
    parser.add_argument(
        "--critic-image-width",
        type=int,
        default=128,
        help="Training-only critic image width.",
    )
    parser.add_argument(
        "--critic-hidden-dim",
        type=int,
        default=256,
        help="Training-only critic hidden dimension.",
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=3e-4,
        help="Critic optimizer learning rate.",
    )
    parser.add_argument(
        "--exploration-learning-rate",
        type=float,
        default=1e-4,
        help="Independent learning rate for exploration log standard deviations.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.997,
        help="Reward discount factor.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Generalized advantage estimation lambda.",
    )
    parser.add_argument(
        "--clip-ratio",
        type=float,
        default=0.2,
        help="PPO probability-ratio clipping range.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help=("Maximum accepted mean truncated Gaussian KL(old||new) per update; "
              "an exceeded limit restores the epoch, optionally retries at a smaller "
              "policy learning rate, then stops if no retry is accepted."),
    )
    parser.add_argument(
        "--kl-backoff-max-retries", type=int, default=0,
        help="Maximum smaller-step retries per rejected epoch; 0 disables retries.",
    )
    parser.add_argument(
        "--kl-backoff-factor", type=float, default=0.5,
        help="Multiply actor/backbone/exploration learning rates by this factor on retry.",
    )
    parser.add_argument(
        "--kl-backoff-min-scale", type=float, default=1e-4,
        help="Minimum policy learning-rate multiplier relative to configured rates.",
    )
    parser.add_argument(
        "--value-loss-coefficient",
        type=float,
        default=0.5,
        help="Value-loss coefficient in the PPO objective.",
    )
    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=0.0,
        help="Truncated Gaussian entropy bonus coefficient in normalized action coordinates.",
    )
    parser.add_argument(
        "--mean-bounds-coefficient",
        type=float,
        default=0.1,
        help="Legacy auxiliary coefficient; structurally bounded schema-4 means have zero bounds loss.",
    )
    parser.add_argument(
        "--raw-mean-bounds-coefficient",
        type=float,
        default=0.0,
        help=("Penalty on ACT predictions outside joint limits before the soft bound, "
              "normalized by each joint's half range. Only executed chunk/TE contributors "
              "are regularized; 0 preserves the previous objective. May be overridden "
              "on resume with an explicit objective-change log."),
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for ACT actor parameters, including the backbone.",
    )
    parser.add_argument(
        "--critic-max-gradient-norm",
        type=float,
        default=0.5,
        help="Independent maximum gradient norm for critic parameters.",
    )
    parser.add_argument(
        "--exploration-max-gradient-norm",
        type=float,
        default=0.5,
        help="Independent maximum gradient norm for exploration parameters.",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="Optimization passes over each rollout.",
    )
    parser.add_argument(
        "--initial-log-std",
        type=float,
        default=-2.3,
        help="Initial log standard deviation for action exploration.",
    )
    parser.add_argument(
        "--minimum-log-std",
        type=float,
        default=-5.0,
        help="Minimum clamped exploration log standard deviation.",
    )
    parser.add_argument(
        "--maximum-log-std",
        type=float,
        default=0.0,
        help="Maximum clamped exploration log standard deviation.",
    )
    parser.add_argument(
        "--state1-exploration-log-std",
        type=float,
        default=-5.0,
        help="Initial exploration log std in State 1.",
    )
    parser.add_argument(
        "--state2-exploration-log-std",
        type=float,
        default=-4.0,
        help="Initial exploration log std in State 2.",
    )
    parser.add_argument(
        "--state3-exploration-log-std",
        type=float,
        default=-4.0,
        help="Initial exploration log std in State 3 grasp/transport.",
    )
    parser.add_argument(
        "--state4-not-home-exploration-log-std",
        type=float,
        default=-4.0,
        help="Initial exploration log std in State 4 before HOME.",
    )
    parser.add_argument(
        "--state4-home-exploration-log-std",
        type=float,
        default=-5.0,
        help="Initial exploration log std in State 4 while HOME is held.",
    )
    parser.add_argument(
        "--state1-exploration-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic rollout exploration in State 1.",
    )
    parser.add_argument(
        "--state2-exploration-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic rollout exploration in State 2.",
    )
    parser.add_argument(
        "--state3-exploration-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic exploration in State 3 grasp/transport.",
    )
    parser.add_argument(
        "--state4-not-home-exploration-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic rollout exploration in State 4 before HOME.",
    )
    parser.add_argument(
        "--state4-home-exploration-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic rollout exploration in State 4 while HOME is held.",
    )
    parser.add_argument(
        "--state1-actor-update-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow State 1 transitions to update the ACT actor.",
    )
    parser.add_argument(
        "--state2-actor-update-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow State 2 transitions to update the ACT actor.",
    )
    parser.add_argument(
        "--state3-actor-update-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow State 3 transitions to update the ACT actor.",
    )
    parser.add_argument(
        "--state4-not-home-actor-update-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow State 4 pre-HOME transitions to update the ACT actor.",
    )
    parser.add_argument(
        "--state4-home-actor-update-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow State 4 HOME-hold transitions to update the ACT actor.",
    )
    parser.add_argument(
        "--actor-learning-rate",
        type=float,
        default=None,
        help=(
            "Optional ACT actor learning-rate override. When omitted, "
            "the pretrained ACT optimizer_lr is used."
        ),
    )
    parser.add_argument(
        "--actor-backbone-learning-rate",
        type=float,
        default=None,
        help=(
            "Optional ACT visual-backbone learning-rate override. "
            "When omitted, pretrained optimizer_lr_backbone is used."
        ),
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=DEFAULT_MINIBATCH_SIZE,
        help=(
            "Number of rollout transitions used in one PPO gradient "
            "minibatch. Lower values reduce GPU memory usage without "
            "changing rollout_steps."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_UPDATES,
        help=(
            "Save one intermediate ACT/training-state checkpoint every "
            "N PPO updates. PPO updates and metrics still run on every "
            "rollout. The final ACT and final training state are always "
            "saved regardless of this interval."
        ),
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write TensorBoard events for every PPO update. Enabled by "
            "default for command-line training; use --no-tensorboard to "
            "disable it."
        ),
    )
    parser.add_argument(
        "--open-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically start a TensorBoard server and open its URL in "
            "the browser. Use --no-open-tensorboard to keep logging without "
            "opening the browser."
        ),
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help=(
            "Preferred local TensorBoard port. If occupied, the next free "
            "port is selected automatically."
        ),
    )
    parser.add_argument(
        "--disable-max-joint-delta-limit",
        action="store_true",
        default=True,
        help=(
            "Compatibility flag: relative delta limiting is disabled by default. "
            "Absolute Gazebo joint limits remain active during PPO collection."
        ),
    )
    parser.add_argument(
        "--startup-feedback-timeout-s",
        type=float,
        default=30.0,
    )
    parser.add_argument("--reward-profile", choices=("legacy", "pick_place_v2"), default="pick_place_v2")
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--task-mode", choices=("pick_place", "idle_empty", "idle_completed", "mixed"), default="mixed")
    parser.add_argument("--idle-empty-probability", type=float, default=0.1)
    parser.add_argument("--idle-completed-probability", type=float, default=0.1)
    parser.add_argument("--randomize-target-delay", action=argparse.BooleanOptionalAction, default=True)
    for group, default in (("idle", -5.0), ("moving", -4.0), ("manipulation", -4.0), ("home-hold", -5.0)):
        parser.add_argument(f"--{group}-exploration-log-std", type=float, default=default)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def _reward_cli_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    result = {name: getattr(args, name) for name in (
        "reward_profile", "reward_config", "task_mode", "idle_empty_probability", "exploration_profile",
        "training_stages", "stage_config", "s0_probability", "stage_initialization_timeout_s",
        *_STAGE_TIME_FIELDS,
        "idle_completed_probability", "randomize_target_delay", "idle_exploration_log_std",
        "moving_exploration_log_std", "manipulation_exploration_log_std", "home_hold_exploration_log_std",
        "moving_gripper_exploration_log_std", "manipulation_gripper_exploration_log_std",
        "gripper_maximum_log_std", "act_max_action_steps", "reach_curriculum",
        "curriculum_window_size", "curriculum_pass_rate", "curriculum_rehearsal_probability")}
    if result["reward_profile"] == "legacy" and result["task_mode"] == "mixed":
        result["task_mode"] = "pick_place"
    return result


def main() -> None:
    args = _parser().parse_args()
    config = ActPpoGazeboConfig(
        act_checkpoint=args.act_checkpoint,
        **_reward_cli_kwargs(args),
        device=args.device,
        episode_count=args.episode_count,
        max_consecutive_episode_failures=(
            args.max_consecutive_episode_failures
        ),
        max_steps_per_episode=args.max_steps_per_episode,
        target_delay_s=args.target_delay_s,
        final_target_absent_s=args.final_target_absent_s,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        critic_image_height=args.critic_image_height,
        critic_image_width=args.critic_image_width,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_learning_rate=args.critic_learning_rate,
        exploration_learning_rate=args.exploration_learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        kl_backoff_max_retries=args.kl_backoff_max_retries,
        kl_backoff_factor=args.kl_backoff_factor,
        kl_backoff_min_scale=args.kl_backoff_min_scale,
        value_loss_coefficient=args.value_loss_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        mean_bounds_coefficient=args.mean_bounds_coefficient,
        raw_mean_bounds_coefficient=args.raw_mean_bounds_coefficient,
        max_gradient_norm=args.max_gradient_norm,
        critic_max_gradient_norm=args.critic_max_gradient_norm,
        exploration_max_gradient_norm=args.exploration_max_gradient_norm,
        ppo_epochs=args.ppo_epochs,
        initial_log_std=args.initial_log_std,
        minimum_log_std=args.minimum_log_std,
        maximum_log_std=args.maximum_log_std,
        state1_exploration_log_std=(
            args.state1_exploration_log_std
        ),
        state2_exploration_log_std=(
            args.state2_exploration_log_std
        ),
        state3_exploration_log_std=args.state3_exploration_log_std,
        state4_not_home_exploration_log_std=(
            args.state4_not_home_exploration_log_std
        ),
        state4_home_exploration_log_std=(
            args.state4_home_exploration_log_std
        ),
        state1_exploration_enabled=args.state1_exploration_enabled,
        state2_exploration_enabled=args.state2_exploration_enabled,
        state3_exploration_enabled=args.state3_exploration_enabled,
        state4_not_home_exploration_enabled=(
            args.state4_not_home_exploration_enabled
        ),
        state4_home_exploration_enabled=(
            args.state4_home_exploration_enabled
        ),
        state1_actor_update_enabled=args.state1_actor_update_enabled,
        state2_actor_update_enabled=args.state2_actor_update_enabled,
        state3_actor_update_enabled=args.state3_actor_update_enabled,
        state4_not_home_actor_update_enabled=(
            args.state4_not_home_actor_update_enabled
        ),
        state4_home_actor_update_enabled=(
            args.state4_home_actor_update_enabled
        ),
        actor_learning_rate=args.actor_learning_rate,
        actor_backbone_learning_rate=args.actor_backbone_learning_rate,
        checkpoint_every_updates=(
            args.checkpoint_every_updates
        ),
        tensorboard_enabled=args.tensorboard,
        tensorboard_auto_open=(
            args.open_tensorboard
        ),
        tensorboard_port=args.tensorboard_port,
        disable_max_joint_delta_limit=(
            args.disable_max_joint_delta_limit
        ),
        startup_feedback_timeout_s=(
            args.startup_feedback_timeout_s
        ),
        seed=args.seed,
        output_directory=args.output_directory,
        resume_checkpoint=args.resume_checkpoint,
    )
    summary = run_act_ppo_gazebo(config)
    print("Direct ACT PPO Gazebo training completed.")
    for key in (
        "output_directory",
        "final_act_checkpoint",
        "final_training_state",
        "episode_count",
        "episode_attempt_count",
        "failed_episode_attempt_count",
        "total_collected_steps",
        "ppo_update_count",
        "checkpoint_every_updates",
        "saved_update_checkpoint_count",
        "tensorboard_log_directory",
        "tensorboard_url",
        "act_parameters_changed",
        "standalone_act_reload_verified",
    ):
        print(f"{key}={summary[key]}")


if __name__ == "__main__":
    main()
