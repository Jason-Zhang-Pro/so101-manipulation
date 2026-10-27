"""Assemble and manage the deterministic SO-101 Gazebo runtime.

The ROS I/O node runs in a background executor so subscriptions continue to
update thread-safe caches while the main thread performs synchronous Gazebo
service calls and deterministic policy steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from so101_capstone.rewards import ReachRewardConfig

if TYPE_CHECKING:
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig

from .environment_randomization import (
    DEFAULT_IMAGE_AUGMENTATION_CONFIG,
    EnvironmentRandomizationConfig,
    EpisodeEnvironmentRandomizer,
    EpisodeImageAugmenter,
)
from .gazebo_environment_randomization import (
    GazeboEnvironmentRandomizationController,
)
from .episode_reset import (
    EpisodeResetConfig,
    EpisodeResetManager,
    EpisodeResetResult,
)
from .policy_observation import PolicyObservation
from .reach_env_core import (
    ReachEnvironmentConfig,
    So101ReachEnvironmentCore,
)
from .simulation_step import SimulationStepSchedule
from .simulation_step_coordinator import (
    CoordinatedSimulationStep,
    DeterministicSimulationStepCoordinator,
    SimulationStepCoordinatorConfig,
)
from .state_cache import GoalRegionSnapshot, MEASURED_JOINT_NAMES
from .target_cube_control import TargetCubeController
from .top_camera_processing import (
    TopCameraImageProcessor,
    TopCameraProcessingConfig,
    default_simulation_top_camera_processing_config,
)


class SimulationRuntimeError(RuntimeError):
    """Raised when the assembled simulation runtime cannot operate."""


def _positive_finite(
    value: float,
    *,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive finite number."
        )

    result = float(value)

    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a positive finite number."
        )

    return result


def _environment_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SimulationRuntimeError(
        f"{name} must be true or false, received {raw!r}."
    )


def _environment_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise SimulationRuntimeError(
            f"{name} must be an integer, received {raw!r}."
        ) from error
    return value


@dataclass(frozen=True)
class SimulationRuntimeConfig:
    """Startup and shutdown behavior for the assembled runtime."""

    startup_feedback_timeout_s: float = 30.0
    feedback_poll_period_s: float = 0.01
    shutdown_timeout_s: float = 5.0
    resume_world_on_close: bool = True

    def __post_init__(self) -> None:
        _positive_finite(
            self.startup_feedback_timeout_s,
            name="startup_feedback_timeout_s",
        )
        _positive_finite(
            self.feedback_poll_period_s,
            name="feedback_poll_period_s",
        )
        _positive_finite(
            self.shutdown_timeout_s,
            name="shutdown_timeout_s",
        )

        if not isinstance(
            self.resume_world_on_close,
            bool,
        ):
            raise ValueError(
                "resume_world_on_close must be a bool."
            )


@dataclass(frozen=True)
class PreparedSimulationEpisode:
    """Complete frame-0 state for one prepared episode."""

    home_reset: EpisodeResetResult
    observation: PolicyObservation
    info: dict[str, Any]


# Backward-compatible type name for existing callers.
SimulationRuntimeReset = PreparedSimulationEpisode


class So101SimulationRuntime:
    """Own the assembled deterministic simulation environment."""

    _REQUIRED_FEEDBACK = (
        ("simulation clock", "simulation_clock_cache"),
        ("joint feedback", "joint_state_cache"),
        (
            "target-cube collection feedback",
            "target_cube_collection_cache",
        ),
        ("top-camera image", "top_image_cache"),
        ("gripper-camera image", "gripper_image_cache"),
    )

    def __init__(
        self,
        *,
        io_node: Any,
        control_node: Any,
        executor: Any,
        executor_thread: Any,
        world_controller: Any,
        cube_controller: Any,
        reset_manager: EpisodeResetManager,
        coordinator: DeterministicSimulationStepCoordinator,
        goal_region_snapshot: GoalRegionSnapshot,
        config: SimulationRuntimeConfig | None = None,
        environment_randomizer: EpisodeEnvironmentRandomizer | None = None,
        image_augmenter: EpisodeImageAugmenter | None = None,
        gazebo_environment_randomization_controller: (
            GazeboEnvironmentRandomizationController | None
        ) = None,
        environment_randomization_warmup_steps: int = 30,
        canonical_goal_half_extent_x_m: float = 0.03,
        canonical_goal_half_extent_y_m: float = 0.03,
        shutdown_ros: Callable[[], None] | None = None,
    ) -> None:
        self.io_node = io_node
        self.control_node = control_node
        self.executor = executor
        self.executor_thread = executor_thread
        self.world_controller = world_controller
        self.cube_controller = cube_controller
        self.reset_manager = reset_manager
        self.coordinator = coordinator
        self._goal_region_snapshot = goal_region_snapshot
        self.environment_randomizer = environment_randomizer
        if environment_randomizer is not None and image_augmenter is not None:
            raise ValueError("Environment randomization and image-only augmentation cannot both be enabled.")
        self.image_augmenter = image_augmenter
        self.gazebo_environment_randomization_controller = (
            gazebo_environment_randomization_controller
        )
        if (
            isinstance(environment_randomization_warmup_steps, bool)
            or not isinstance(environment_randomization_warmup_steps, int)
            or environment_randomization_warmup_steps < 1
        ):
            raise ValueError(
                "environment_randomization_warmup_steps must be a positive integer."
            )
        self.environment_randomization_warmup_steps = (
            environment_randomization_warmup_steps
        )
        self._canonical_goal_center_world_m = (
            float(goal_region_snapshot.x),
            float(goal_region_snapshot.y),
            float(goal_region_snapshot.z),
        )
        self._canonical_goal_half_extent_x_m = float(
            canonical_goal_half_extent_x_m
        )
        self._canonical_goal_half_extent_y_m = float(
            canonical_goal_half_extent_y_m
        )
        self._environment_randomization_episode_counter = 0
        self.config = (
            SimulationRuntimeConfig()
            if config is None
            else config
        )
        self._shutdown_ros = shutdown_ros
        self._closed = False
        self._task_rng = np.random.default_rng()
        self._stage_start_available = False

    def configure_task(self, task_mode: str, *, seed: int | None = None,
                       n_action_steps: int = 1) -> None:
        """Select the next episode's task without changing policy observations."""
        self._ensure_open()
        if task_mode == "idle_completed" and self.environment_randomizer is not None:
            raise SimulationRuntimeError("Completed-idle preparation currently requires randomization disabled.")
        self.coordinator.environment.configure_task(task_mode, seed=seed, n_action_steps=n_action_steps)
        self._task_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])

    def _task_reset_targets(self, base_targets: Sequence[float]) -> tuple[float, ...]:
        """Small initial-pose variation; never move the HOME scoring reference."""
        environment = getattr(self.coordinator, "environment", None)
        if getattr(environment, "reward_profile", "legacy") != "pick_place_v2":
            return tuple(base_targets)
        from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS
        _validate_task_home_limits(environment.reward_config)
        targets = np.asarray(base_targets, dtype=float).copy()
        jitter = environment.reward_config.home_reset_jitter_rad
        targets[:5] += self._task_rng.uniform(-jitter, jitter, size=5)
        for index, calibration in enumerate(SIM_JOINT_CALIBRATIONS[:5]):
            targets[index] = np.clip(targets[index], calibration.gazebo_lower_rad, calibration.gazebo_upper_rad)
        return tuple(float(value) for value in targets)

    @property
    def goal_region_snapshot(
        self,
    ) -> GoalRegionSnapshot:
        """Return the world-start goal-region centre."""

        return self._goal_region_snapshot

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise SimulationRuntimeError(
                "Simulation runtime is already closed."
            )

    def _missing_feedback(self) -> tuple[str, ...]:
        missing = []

        for label, attribute_name in self._REQUIRED_FEEDBACK:
            cache = getattr(
                self.io_node,
                attribute_name,
                None,
            )

            if cache is None or cache.latest() is None:
                missing.append(label)

        return tuple(missing)

    def wait_until_ready(
        self,
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Wait until all feedback required by reset is available."""

        self._ensure_open()

        timeout = (
            self.config.startup_feedback_timeout_s
            if timeout_s is None
            else _positive_finite(
                timeout_s,
                name="timeout_s",
            )
        )
        deadline_s = monotonic() + timeout

        while True:
            missing = self._missing_feedback()

            if not missing:
                return

            remaining_s = deadline_s - monotonic()

            if remaining_s <= 0.0:
                raise SimulationRuntimeError(
                    "Timed out waiting for initial feedback: "
                    + ", ".join(missing)
                )

            sleep(
                min(
                    self.config.feedback_poll_period_s,
                    remaining_s,
                )
            )

    def _wait_for_evaluation_controller(self) -> dict[str, Any]:
        """Check the real controller before ZERO can be mistaken for a reset.

        Keep the freshly launched world running here: controller activation
        may need a physics update. The reset below owns the pause/step barrier.
        The separate control node is not in the background feedback executor.
        """
        from .controller_readiness import (
            ControllerReadinessError,
            wait_for_position_controller,
        )

        try:
            return wait_for_position_controller(
                self.control_node,
                command_subscription_count=self.io_node.joint_target_subscription_count,
                timeout_s=self.config.startup_feedback_timeout_s,
            )
        except ControllerReadinessError as error:
            raise SimulationRuntimeError(str(error)) from error

    def prepare_episode(
        self,
        *,
        now_monotonic_s: float | None = None,
        environment_randomization_episode_key: int | None = None,
        evaluation_initial_pose: str | None = None,
    ) -> PreparedSimulationEpisode:
        """Prepare HOME and return the new episode frame-0 state.

        With environment randomization disabled this intentionally follows
        the original reset path exactly. When enabled, one episode sample is
        applied before reset, followed by 30 discarded reset-only control
        frames and one frame-0 priming step.

        The explicit ``evaluation_initial_pose='zero'`` path holds the
        robot's zero-joint spawn pose instead of HOME. It excludes HOME
        jitter and all randomization while preserving the same stability,
        clock, and synchronized frame-0 checks. It first waits for the position
        controller to be active and connected, before reset pauses the world.
        """

        self._ensure_open()
        if getattr(self.coordinator, "grasp_attachment_controller", None) is not None:
            self.coordinator.environment.assert_attachment_reset_allowed()
        if evaluation_initial_pose not in (None, "zero"):
            raise ValueError("evaluation_initial_pose must be None or 'zero'.")
        if evaluation_initial_pose == "zero":
            environment = self.coordinator.environment
            if (getattr(environment, "reward_profile", "legacy") != "pick_place_v2"
                    or environment.task_mode != "pick_place"):
                raise ValueError("ZERO evaluation requires pick_place_v2 with task_mode='pick_place'.")
            if self.environment_randomizer is not None or self.image_augmenter is not None:
                raise ValueError("ZERO evaluation requires environment and image randomization disabled.")
        self._stage_start_available = False
        self.wait_until_ready()

        controller_readiness = None
        if evaluation_initial_pose == "zero":
            controller_readiness = self._wait_for_evaluation_controller()

        if self.image_augmenter is not None:
            self.image_augmenter.begin_episode(
                episode_key=self._next_randomization_episode_key(environment_randomization_episode_key),
            )

        # Master-switch fast path: no sampling, no Gazebo mutation, no image
        # randomization setup, and no extra warm-up steps.
        if self.environment_randomizer is None:
            previous_joint = self.io_node.joint_state_cache.latest()
            previous_target_collection = (
                self.io_node.target_cube_collection_cache.latest()
            )
            previous_top = self.io_node.top_image_cache.latest()
            previous_gripper = self.io_node.gripper_image_cache.latest()

            if previous_joint is None:
                raise SimulationRuntimeError(
                    "Joint feedback disappeared before episode preparation."
                )
            if previous_target_collection is None:
                raise SimulationRuntimeError(
                    "Target-cube collection feedback disappeared before "
                    "episode preparation."
                )
            if previous_target_collection.source_sequence is None:
                raise SimulationRuntimeError(
                    "Target-cube collection source sequence is missing before "
                    "episode preparation."
                )
            if previous_top is None:
                raise SimulationRuntimeError(
                    "Top-camera feedback disappeared before episode preparation."
                )
            if previous_gripper is None:
                raise SimulationRuntimeError(
                    "Gripper-camera feedback disappeared before episode preparation."
                )

            task_profile = getattr(getattr(self.coordinator, "environment", None), "reward_profile", "legacy")
            if task_profile == "pick_place_v2":
                reset_targets = ((0.0,) * len(MEASURED_JOINT_NAMES)
                                 if evaluation_initial_pose == "zero" else
                                 self._task_reset_targets(self.reset_manager.config.home_joint_positions))
                home_reset = self.reset_manager.reset_to_home(home_joint_positions_override=reset_targets)
            else:
                home_reset = self.reset_manager.reset_to_home()
            if (getattr(getattr(self.coordinator, "environment", None), "reward_profile", "legacy") == "pick_place_v2"
                    and self.coordinator.environment.task_mode == "idle_completed"):
                cfg = self.coordinator.environment.reward_config
                center = self.coordinator.environment.goal_region_center_world_m
                response = self.spawn_target((center[0], center[1],
                    center[2] + cfg.goal_support_offset_m + cfg.target_half_extents_m[2] + 0.003))
                target_name = json.loads(response.message)["name"]
                _, settled_clock, _ = self.reset_manager.hold_targets_for_steps(
                    reset_targets,
                    control_steps=max(30, round(cfg.control_hz)),
                )
                deadline = monotonic() + self.config.startup_feedback_timeout_s
                while True:
                    target = self.io_node.target_cube_cache.latest()
                    collection = self.io_node.target_cube_collection_cache.latest()
                    if target is not None and target.name == target_name and collection is not None and target_name in collection.cube_names:
                        break
                    if monotonic() >= deadline:
                        raise SimulationRuntimeError("Timed out confirming the completed-idle target before frame zero.")
                    sleep(self.config.feedback_poll_period_s)
                home_reset = replace(home_reset, clock_state=settled_clock)
            observation, info = self.coordinator.prepare_episode_from_latest(
                after_joint_sequence=previous_joint.sequence,
                after_target_collection_sequence=(
                    previous_target_collection.sequence
                ),
                after_target_collection_source_sequence=(
                    previous_target_collection.source_sequence
                ),
                after_top_sequence=previous_top.sequence,
                after_gripper_sequence=previous_gripper.sequence,
                episode_clock_state=home_reset.clock_state,
                now_monotonic_s=now_monotonic_s,
                **({"reset_command_joint_targets_rad": reset_targets}
                   if getattr(getattr(self.coordinator, "environment", None), "reward_profile", "legacy") == "pick_place_v2" else {}),
            )
            if task_profile == "pick_place_v2":
                info = dict(info)
                info["reset_joint_targets_rad"] = list(reset_targets)
                info["reset_home_offset_rad"] = (np.asarray(reset_targets) - self.coordinator.environment.reward_config.home_joint_positions_rad).tolist()
                if evaluation_initial_pose == "zero":
                    info["evaluation_initial_pose"] = "zero"
                    info["reset_pose_name"] = "ZERO"
                    info["home_initialization_assisted"] = False
                    info["evaluation_controller_readiness"] = controller_readiness
            if self.image_augmenter is not None:
                info = dict(info)
                info["image_augmentation"] = self.image_augmenter.record()
            self._stage_start_available = evaluation_initial_pose is None
            return PreparedSimulationEpisode(
                home_reset=home_reset, observation=observation, info=info
            )

        controller = self.gazebo_environment_randomization_controller
        if controller is None:
            raise SimulationRuntimeError(
                "Environment randomizer is enabled without a Gazebo controller."
            )

        episode_key = self._next_randomization_episode_key(environment_randomization_episode_key)

        sample = self.environment_randomizer.begin_episode(
            episode_key=episode_key,
            goal_center_world_m=self._canonical_goal_center_world_m,
            goal_half_extent_x_m=self._canonical_goal_half_extent_x_m,
            goal_half_extent_y_m=self._canonical_goal_half_extent_y_m,
            home_joint_positions_rad=self.reset_manager.config.home_joint_positions,
        )
        controller.apply_episode(sample)

        goal_center = self.environment_randomizer.goal_center_world_m()
        goal_half_x, goal_half_y = self.environment_randomizer.goal_half_extents_m()
        self.coordinator.environment.set_episode_goal_region(
            goal_center,
            half_extent_x_m=goal_half_x,
            half_extent_y_m=goal_half_y,
        )

        randomized_home = self.environment_randomizer.randomized_home_joint_positions(
            self.reset_manager.config.home_joint_positions
        )
        if getattr(self.coordinator.environment, "reward_profile", "legacy") == "pick_place_v2":
            randomized_home = self._task_reset_targets(randomized_home)
        home_reset = self.reset_manager.reset_to_home(
            home_joint_positions_override=randomized_home
        )

        # Exactly 30 discarded 30 Hz environment frames. No policy action,
        # reward transition, or observation randomizer is invoked here.
        self.reset_manager.hold_targets_for_steps(
            randomized_home,
            control_steps=self.environment_randomization_warmup_steps,
        )

        after_joint = self.io_node.joint_state_cache.latest()
        after_top = self.io_node.top_image_cache.latest()
        after_gripper = self.io_node.gripper_image_cache.latest()
        if after_joint is None or after_top is None or after_gripper is None:
            raise SimulationRuntimeError(
                "Feedback disappeared after environment-randomization warm-up."
            )

        # One extra reset-only step generates the formal post-warm-up frame-0.
        # The three sequence markers above force policy-visible joint / top /
        # gripper data to be newer than all discarded warm-up observations.
        _, frame0_clock, _ = self.reset_manager.hold_targets_for_steps(
            randomized_home,
            control_steps=1,
        )
        observation, info = self.coordinator.prepare_episode_from_latest(
            after_joint_sequence=after_joint.sequence,
            after_target_collection_sequence=None,
            after_target_collection_source_sequence=None,
            after_top_sequence=after_top.sequence,
            after_gripper_sequence=after_gripper.sequence,
            episode_clock_state=frame0_clock,
            now_monotonic_s=now_monotonic_s,
            **({"reset_command_joint_targets_rad": randomized_home}
               if getattr(self.coordinator.environment, "reward_profile", "legacy") == "pick_place_v2" else {}),
        )
        info = dict(info)
        info["environment_randomization"] = self.environment_randomizer.record()
        if getattr(self.coordinator.environment, "reward_profile", "legacy") == "pick_place_v2":
            info["reset_joint_targets_rad"] = list(randomized_home)
            info["reset_home_offset_rad"] = (np.asarray(randomized_home) - self.coordinator.environment.reward_config.home_joint_positions_rad).tolist()
        info["environment_randomization_warmup_steps"] = (
            self.environment_randomization_warmup_steps
        )
        self._stage_start_available = True
        return PreparedSimulationEpisode(
            home_reset=home_reset, observation=observation, info=info
        )

    def evaluation_measurements(self) -> dict[str, Any]:
        """Export the evidence-aligned physical snapshot for evaluation reports.

        This does not add privileged measurements to the actor observation.
        The process-isolated worker calls it only in explicit evaluation mode.
        """
        self._ensure_open()
        snapshot = getattr(self.coordinator.environment, "_task_snapshot", None)
        if snapshot is None:
            raise SimulationRuntimeError("Evaluation measurements require a prepared task snapshot.")
        return asdict(snapshot.measurements)

    def _next_randomization_episode_key(self, episode_key: int | None) -> int:
        if episode_key is None:
            episode_key = self._environment_randomization_episode_counter
            self._environment_randomization_episode_counter += 1
        elif isinstance(episode_key, bool) or not isinstance(episode_key, int) or episode_key < 0:
            raise ValueError("environment_randomization_episode_key must be a non-negative integer or None.")
        return episode_key

    def start_training_stage(
        self,
        training_stage: str,
        stage_config: dict[str, Any],
    ) -> PreparedSimulationEpisode:
        """Prepare and validate a stage after reset/spawn, before ACT control.

        Preparation commands create no policy/reward transitions. Later stages
        use deterministic physical steps with task history retained; their time
        is excluded from the new stage's sampling budget.
        """
        self._ensure_open()
        if training_stage in {"S2", "S3", "S4", "S5"}:
            try:
                return self._start_later_training_stage(training_stage, stage_config)
            except Exception as error:
                if not hasattr(error, "stage_initialization"):
                    error.stage_initialization = {
                        "valid": False, "training_stage": training_stage,
                        "phase": "INITIALIZATION", "reason": str(error),
                        "policy_steps": 0,
                    }
                raise
        environment = self.coordinator.environment
        if training_stage not in {"S0", "S1"}:
            raise ValueError("Training stage must be S0, S1, S2, S3, S4 or S5.")
        if getattr(environment, "reward_profile", "legacy") != "pick_place_v2":
            raise ValueError("Training stage initialization requires pick_place_v2.")
        expected_mode = "idle_empty" if training_stage == "S0" else "pick_place"
        if environment.task_mode != expected_mode:
            raise ValueError("Training stage and prepared task mode disagree.")
        if not self._stage_start_available:
            raise SimulationRuntimeError(
                "Stage initialization requires an unused prepared episode."
            )
        # A failed or partially completed initialization cannot be retried on
        # a scene whose physical preparation has already been consumed.
        self._stage_start_available = False
        self.wait_until_ready()
        initial_clock = self.io_node.simulation_clock_cache.latest()
        if initial_clock is None:
            raise SimulationRuntimeError("No clock before stage initialization.")
        started_wall_s = monotonic()
        reward_config = environment.reward_config
        home = tuple(reward_config.home_joint_positions_rad)
        warmup_steps = max(30, math.ceil(reward_config.control_hz))
        joint, clock, physics_steps = self.reset_manager.hold_targets_for_steps(
            home, control_steps=warmup_steps,
        )
        discarded_steps = warmup_steps
        stable_samples = 0
        required_samples = self.reset_manager.config.stable_sample_count
        max_error = max_velocity = float("inf")
        for attempt in range(self.reset_manager.config.max_control_steps):
            if joint.velocities is None:
                raise SimulationRuntimeError("Stage initialization requires joint velocities.")
            positions = np.asarray([joint.positions[name] for name in MEASURED_JOINT_NAMES])
            velocities = np.asarray([joint.velocities[name] for name in MEASURED_JOINT_NAMES])
            max_error = float(np.max(np.abs(positions - home)))
            max_velocity = float(np.max(np.abs(velocities)))
            valid_home = (
                np.isfinite(positions).all() and np.isfinite(velocities).all()
                and max_error <= reward_config.home_joint_tolerance_rad
                and max_velocity <= reward_config.home_velocity_tolerance_rad_s
            )
            stable_samples = stable_samples + 1 if valid_home else 0
            if stable_samples >= required_samples:
                break
            if attempt + 1 < self.reset_manager.config.max_control_steps:
                joint, clock, count = self.reset_manager.hold_targets_for_steps(
                    home, control_steps=1,
                )
                discarded_steps += 1
                physics_steps += count
        else:
            raise SimulationRuntimeError(
                "HOME did not stabilize before stage initialization: "
                f"error={max_error:.6f} rad, velocity={max_velocity:.6f} rad/s."
            )

        markers = {
            name: getattr(self.io_node, name).latest()
            for name in (
                "joint_state_cache", "target_cube_collection_cache",
                "top_image_cache", "gripper_image_cache",
            )
        }
        if any(value is None for value in markers.values()):
            raise SimulationRuntimeError("Feedback disappeared during stage warm-up.")
        collection = markers["target_cube_collection_cache"]
        if collection.source_sequence is None:
            raise SimulationRuntimeError("Stage target collection lacks a source sequence.")
        # More than one target-pose period also supplies fresh synchronized
        # cameras after all discarded warm-up frames, without camera reuse.
        prime_steps = max(4, math.ceil(reward_config.control_hz * reward_config.max_evidence_gap_s) + 1)
        joint, clock, count = self.reset_manager.hold_targets_for_steps(
            home, control_steps=prime_steps,
        )
        discarded_steps += prime_steps
        physics_steps += count
        environment.configure_training_stage(training_stage, stage_config)
        observation, info = self.coordinator.prepare_episode_from_latest(
            after_joint_sequence=markers["joint_state_cache"].sequence,
            after_target_collection_sequence=collection.sequence,
            after_target_collection_source_sequence=collection.source_sequence,
            after_top_sequence=markers["top_image_cache"].sequence,
            after_gripper_sequence=markers["gripper_image_cache"].sequence,
            episode_clock_state=clock,
            reset_command_joint_targets_rad=home,
        )
        info = dict(info)
        self._validate_stage_start_info(training_stage, info, initial_clock.sim_time_s)
        # Task evidence may be aligned to an older target/contact sample.
        # Also certify the newest joints ACT will actually observe at handoff.
        observed_positions = np.asarray(observation.joint_positions, dtype=float)
        observed_velocities = np.asarray(observation.joint_velocities, dtype=float)
        if (observed_positions.shape != (6,) or observed_velocities.shape != (6,)
                or not np.isfinite(observed_positions).all()
                or not np.isfinite(observed_velocities).all()):
            raise SimulationRuntimeError("Stage handoff joint observation is invalid.")
        observed_error = float(np.max(np.abs(observed_positions - home)))
        observed_velocity = float(np.max(np.abs(observed_velocities)))
        if (observed_error > reward_config.home_joint_tolerance_rad
                or observed_velocity > reward_config.home_velocity_tolerance_rad_s):
            raise SimulationRuntimeError("Latest stage handoff HOME state is not stable.")
        certificate = {
            "certificate": "s0_home_ready" if training_stage == "S0" else "s1_source_target_ready",
            "valid": True,
            "training_stage": training_stage,
            "source": "home_hold_and_sensor_settle",
            # S0/S1 start at the ordinary HOME reset; no reach/grasp skill
            # was supplied by a controller. Later held-object stages differ.
            "assisted_initialization": False,
            "discarded_control_steps": discarded_steps,
            "discarded_physics_steps": physics_steps,
            "initialization_start_sim_time_s": initial_clock.sim_time_s,
            "start_sim_time_s": clock.sim_time_s,
            "initialization_sim_duration_s": clock.sim_time_s - initial_clock.sim_time_s,
            "initialization_wall_duration_s": monotonic() - started_wall_s,
            "home_stable_samples": stable_samples,
            "home_max_position_error_rad": observed_error,
            "home_max_velocity_rad_s": observed_velocity,
            "target_name": info.get("target_name"),
            "target_static": info.get("target_static", False),
            "source_supported": training_stage == "S1",
            "camera_source_times_s": list(info["camera_source_times_s"]),
            "contact_measurement_valid": bool(info["contact_measurement_valid"]),
        }
        info["stage_initialization"] = certificate
        info["reset_joint_targets_rad"] = list(home)
        info["reset_home_offset_rad"] = [0.0] * len(home)
        if self.environment_randomizer is not None:
            info["environment_randomization"] = self.environment_randomizer.record()
        if getattr(self, "image_augmenter", None) is not None:
            info["image_augmentation"] = self.image_augmenter.record()
        return PreparedSimulationEpisode(
            home_reset=EpisodeResetResult(
                joint_state=joint, clock_state=clock, control_steps=discarded_steps,
                physics_steps=physics_steps, stable_sample_count=stable_samples,
                max_position_error_rad=observed_error,
                max_velocity_rad_s=observed_velocity,
            ),
            observation=observation,
            info=info,
        )

    def _start_later_training_stage(
        self, training_stage: str, stage_config: dict[str, Any],
    ) -> PreparedSimulationEpisode:
        """Physically build a later-stage start, retaining real task history.

        Every controller action uses the ordinary deterministic feedback cycle.
        The core's preparation mode prevents rewards or episode steps from
        entering PPO. No target is moved by a service or deleted here.
        """
        from so101_capstone.rewards.training_stages import StageTrainingConfig
        from .stage_preparation import (
            StagePreparationConfig, StagePreparationController, StagePreparationError,
        )

        # Reject invalid controller/configuration input before consuming a world.
        resolved = StageTrainingConfig.from_dict(stage_config)
        controller_config = StagePreparationConfig(**(resolved.stage_preparation or {}))
        started_wall_s = monotonic()
        source = self.start_training_stage("S1", stage_config)
        environment = self.coordinator.environment
        environment.begin_stage_preparation()
        initial = environment.task_snapshot
        if initial is None:
            raise SimulationRuntimeError("No measured source state before stage preparation.")
        physics_steps = source.home_reset.physics_steps
        last_step = None

        def advance(targets):
            nonlocal physics_steps, last_step
            last_step = self.coordinator.step(targets)
            physics_steps += last_step.physics_steps
            if (not last_step.info.get("preparation_only")
                    or last_step.info.get("episode_step") != 0
                    or last_step.reward != 0):
                raise SimulationRuntimeError("Preparation leaked into a policy/reward step.")
            if last_step.terminated or last_step.truncated:
                raise SimulationRuntimeError("Physical preparation ended before a valid handoff.")
            snapshot = environment.task_snapshot
            if snapshot is None:
                raise SimulationRuntimeError("Preparation feedback has no task snapshot.")
            return snapshot

        try:
            report = StagePreparationController(
                environment.reward_config, controller_config,
            ).run(training_stage, initial, advance)
        except StagePreparationError as error:
            failure = SimulationRuntimeError(
                f"{training_stage} preparation failed: phase={error.phase}, "
                f"preparation_steps={error.steps}, reason={error.reason}; "
                "this world must be rebuilt before retrying."
            )
            failure.stage_initialization = {
                "valid": False, "training_stage": training_stage,
                "phase": error.phase, "reason": error.reason,
                "preparation_control_steps": error.steps,
                "discarded_control_steps": source.home_reset.control_steps + error.steps,
                "discarded_physics_steps": physics_steps, "policy_steps": 0,
                "preparation_config": asdict(controller_config),
            }
            raise failure from error
        if last_step is None:
            raise SimulationRuntimeError("Later-stage preparation produced no measured control step.")
        try:
            observation, info = self.coordinator.handoff_training_stage(
                training_stage, stage_config,
                command_joint_targets_rad=report.last_command_joint_targets_rad,
            )
            info = dict(info)
            if (info.get("episode_step") != 0 or not info.get("frame_zero")
                    or info.get("training_stage") != training_stage
                    or info.get("stage_elapsed_s") != 0):
                raise SimulationRuntimeError("Later-stage handoff did not produce a fresh frame zero.")
        except Exception as error:
            error.stage_initialization = {
                "valid": False, "training_stage": training_stage,
                "phase": "HANDOFF", "reason": str(error), "policy_steps": 0,
                "preparation_control_steps": report.steps,
                "discarded_control_steps": source.home_reset.control_steps + report.steps,
                "discarded_physics_steps": physics_steps,
            }
            raise
        # Record controller work separately from actor sampling and failure loss.
        certificate = {
            **report.certificate,
            "valid": True, "training_stage": training_stage,
            "source": "measured_stage_preparation_controller",
            "assisted_initialization": True,
            "discarded_control_steps": source.home_reset.control_steps + report.steps,
            "preparation_control_steps": report.steps,
            "discarded_physics_steps": physics_steps,
            "phases": list(report.phases),
            "initialization_start_sim_time_s": source.info["stage_initialization"]["initialization_start_sim_time_s"],
            "start_sim_time_s": last_step.end_clock.sim_time_s,
            "initialization_wall_duration_s": monotonic() - started_wall_s,
            "camera_source_times_s": list(info.get("camera_source_times_s", ())),
            "target_name": info.get("target_name"),
            "preparation_config": asdict(controller_config),
        }
        certificate["initialization_sim_duration_s"] = (
            certificate["start_sim_time_s"] - certificate["initialization_start_sim_time_s"])
        info["stage_initialization"] = certificate
        info["reset_joint_targets_rad"] = list(report.last_command_joint_targets_rad)
        info["preparation_only"] = False
        if self.environment_randomizer is not None:
            info["environment_randomization"] = self.environment_randomizer.record()
        if getattr(self, "image_augmenter", None) is not None:
            info["image_augmentation"] = self.image_augmenter.record()
        joint = last_step.joint_state
        q = np.asarray([joint.positions[name] for name in MEASURED_JOINT_NAMES])
        velocity = np.asarray([joint.velocities[name] for name in MEASURED_JOINT_NAMES])
        return PreparedSimulationEpisode(
            home_reset=EpisodeResetResult(
                joint_state=joint, clock_state=last_step.end_clock,
                control_steps=certificate["discarded_control_steps"], physics_steps=physics_steps,
                stable_sample_count=0,
                max_position_error_rad=float(np.max(np.abs(q - report.last_command_joint_targets_rad))),
                max_velocity_rad_s=float(np.max(np.abs(velocity))),
            ), observation=observation, info=info,
        )

    def _validate_stage_start_info(
        self, training_stage: str, info: dict[str, Any], request_sim_time_s: float,
    ) -> None:
        """Reject invalid starts before publishing any PPO-visible frame zero."""
        if not info.get("frame_zero") or info.get("episode_step") != 0:
            raise SimulationRuntimeError("Stage initialization did not produce frame zero.")
        if not (info.get("robot_at_home") and info.get("robot_static")):
            raise SimulationRuntimeError("Stage initial HOME state is not stable.")
        if not (info.get("evidence_valid") and info.get("contact_measurement_valid")):
            raise SimulationRuntimeError("Stage initial sensor/contact evidence is invalid.")
        stamps = info.get("camera_source_times_s", ())
        if len(stamps) != 2 or any(
            stamp is None or not math.isfinite(float(stamp))
            or float(stamp) <= request_sim_time_s for stamp in stamps
        ):
            raise SimulationRuntimeError("Stage cameras must be newer than initialization.")
        count = info.get("target_collection_cube_count")
        present = bool(info.get("physical_target_present"))
        if training_stage == "S0":
            if present or count != 0:
                raise SimulationRuntimeError("S0 requires an empty target scene.")
            return
        if not present or count != 1 or not info.get("target_name"):
            raise SimulationRuntimeError("S1 requires exactly one physical target.")
        if not info.get("target_motion_valid") or not info.get("target_static"):
            raise SimulationRuntimeError("S1 target has not been measured to settle.")
        reward_config = self.coordinator.environment.reward_config
        bottom = float(info.get("target_bottom_z_m", float("nan")))
        source = float(info.get("source_support_z_m", float("nan")))
        if not math.isfinite(bottom - source) or abs(bottom - source) > reward_config.support_height_tolerance_m:
            raise SimulationRuntimeError("S1 target is not supported at the source.")
        if info.get("target_in_goal") or info.get("grasp_valid"):
            raise SimulationRuntimeError("S1 target must be ungrasped outside the goal.")

    def reset(
        self,
        *,
        now_monotonic_s: float | None = None,
    ) -> PreparedSimulationEpisode:
        """Backward-compatible alias for :meth:`prepare_episode`."""

        return self.prepare_episode(
            now_monotonic_s=now_monotonic_s,
        )

    def step(
        self,
        joint_targets: Sequence[float],
        *,
        now_monotonic_s: float | None = None,
    ) -> CoordinatedSimulationStep:
        """Execute one deterministic policy/environment step."""

        self._ensure_open()

        self._stage_start_available = False
        result = self.coordinator.step(
            joint_targets,
            now_monotonic_s=now_monotonic_s,
        )
        image_augmenter = getattr(self, "image_augmenter", None)
        if self.environment_randomizer is None and image_augmenter is None:
            return result
        info = dict(result.info)
        if self.environment_randomizer is not None:
            info["environment_randomization"] = self.environment_randomizer.record()
        if image_augmenter is not None:
            info["image_augmentation"] = image_augmenter.record()
        return replace(result, info=info)

    def randomize_spawned_target(self, model_name: str) -> None:
        """Queue this episode's target appearance before its first render."""

        if self.environment_randomizer is None:
            return
        controller = self.gazebo_environment_randomization_controller
        if controller is None:
            raise SimulationRuntimeError(
                "Environment randomizer is enabled without a Gazebo controller."
            )
        sample = self.environment_randomizer.current_sample
        if sample is None:
            raise SimulationRuntimeError(
                "Target randomization was requested before episode sampling."
            )
        target = sample.scene["target_color"]
        if not bool(target["enabled"]):
            return
        controller.apply_target_instance(
            str(model_name),
            target["rgb"],
        )

    def spawn_random_target(self) -> Any:
        """Request one randomly positioned target cube."""

        self._ensure_open()
        response = self.cube_controller.spawn_random()
        if self.environment_randomizer is not None and bool(
            getattr(response, "success", False)
        ):
            try:
                payload = json.loads(str(getattr(response, "message", "")))
                model_name = str(payload["name"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise SimulationRuntimeError(
                    "Target spawned but its model name could not be recovered "
                    "for appearance randomization."
                ) from error
            self.randomize_spawned_target(model_name)
        return response

    def spawn_target(self, position_world_m, quaternion_xyzw=(0.0, 0.0, 0.0, 1.0)) -> Any:
        """Spawn a requested fixed scenario; policy sees it on later measured frames."""
        self._ensure_open()
        response = self.cube_controller.spawn_at(position_world_m, quaternion_xyzw)
        if self.environment_randomizer is not None:
            self.randomize_spawned_target(json.loads(response.message)["name"])
        return response

    def delete_all_targets(self) -> Any:
        """Delete every managed target without ending the episode."""

        self._ensure_open()
        if getattr(self.coordinator, "grasp_attachment_controller", None) is not None:
            self.coordinator.environment.assert_attachment_reset_allowed()
        return self.cube_controller.delete_all()

    def close(self) -> None:
        """Stop callbacks, destroy nodes, and release ROS resources."""

        if self._closed:
            return

        self._closed = True
        errors: list[tuple[str, BaseException]] = []

        def attempt(
            label: str,
            action: Callable[[], Any],
        ) -> None:
            try:
                action()
            except BaseException as error:
                errors.append((label, error))

        if self.config.resume_world_on_close:
            attempt(
                "resume Gazebo",
                self.world_controller.resume,
            )

        attempt(
            "shutdown executor",
            lambda: self.executor.shutdown(
                timeout_sec=self.config.shutdown_timeout_s
            ),
        )
        attempt(
            "join executor thread",
            lambda: self.executor_thread.join(
                timeout=self.config.shutdown_timeout_s
            ),
        )

        is_alive = getattr(
            self.executor_thread,
            "is_alive",
            None,
        )

        if callable(is_alive) and is_alive():
            errors.append(
                (
                    "join executor thread",
                    RuntimeError(
                        "Executor thread did not stop."
                    ),
                )
            )

        attempt(
            "destroy I/O node",
            self.io_node.destroy_node,
        )
        attempt(
            "destroy control node",
            self.control_node.destroy_node,
        )

        if self._shutdown_ros is not None:
            attempt(
                "shutdown ROS",
                self._shutdown_ros,
            )

        if errors:
            details = "; ".join(
                f"{label}: {error}"
                for label, error in errors
            )
            raise SimulationRuntimeError(
                f"Runtime cleanup failed: {details}"
            ) from errors[0][1]

    def __enter__(self) -> So101SimulationRuntime:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception: Any,
        traceback: Any,
    ) -> bool:
        try:
            self.close()
        except Exception:
            if exception_type is None:
                raise

        return False


def _validate_task_home_limits(reward_config) -> None:
    from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS
    for value, calibration in zip(reward_config.home_joint_positions_rad, SIM_JOINT_CALIBRATIONS):
        if not calibration.gazebo_lower_rad <= value <= calibration.gazebo_upper_rad:
            raise ValueError(f"Reward HOME {calibration.name} exceeds its physical joint limits.")


def create_ros_simulation_runtime(
    *,
    runtime_config: SimulationRuntimeConfig | None = None,
    environment_config: ReachEnvironmentConfig | None = None,
    reward_config: ReachRewardConfig | PickPlaceRewardConfig | None = None,
    reset_config: EpisodeResetConfig | None = None,
    coordinator_config: (
        SimulationStepCoordinatorConfig | None
    ) = None,
    top_camera_config: (
        TopCameraProcessingConfig | None
    ) = None,
    environment_randomization_enabled: bool | None = None,
    environment_randomization_config_path: str | Path | None = None,
    environment_randomization_seed: int | None = None,
    environment_randomization_world_index: int | None = None,
    environment_randomization_warmup_steps: int = 30,
    image_augmentation_enabled: bool | None = None,
    image_augmentation_config_path: str | Path | None = None,
    ros_args: list[str] | None = None,
    reward_profile: str = "legacy",
    task_mode: str = "pick_place",
    observe_after_success_s: float = 0.0,
) -> So101SimulationRuntime:
    """Create the complete ROS/Gazebo simulation runtime.

    The caller must run Python with ``PYTHONNOUSERSITE=1`` so ROS Humble,
    system NumPy, and system OpenCV use compatible binary interfaces.
    """

    if reward_profile not in {"legacy", "pick_place_v2"}:
        raise ValueError("Unknown reward_profile.")
    assisted_grasp_enabled = _environment_bool("SO101_ASSISTED_GRASP_ENABLED", default=False)
    if assisted_grasp_enabled and reward_profile != "pick_place_v2":
        raise ValueError("Rigid grasp assistance requires pick_place_v2 contact evidence.")
    if reward_profile == "pick_place_v2":
        from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
        reward_config = PickPlaceRewardConfig() if reward_config is None else reward_config
        if not isinstance(reward_config, PickPlaceRewardConfig):
            raise TypeError("pick_place_v2 requires a resolved PickPlaceRewardConfig.")
        if not math.isclose(reward_config.control_hz, 30.0, abs_tol=1e-9):
            raise ValueError("pick_place_v2 control_hz must match the physical 30 Hz schedule.")
        _validate_task_home_limits(reward_config)

    randomization_enabled = (
        _environment_bool("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", default=False)
        if environment_randomization_enabled is None else bool(environment_randomization_enabled)
    )
    image_only_enabled = (
        _environment_bool("SO101_IMAGE_AUGMENTATION_ENABLED", default=False)
        if image_augmentation_enabled is None else bool(image_augmentation_enabled)
    )
    if randomization_enabled and image_only_enabled:
        raise SimulationRuntimeError(
            "Environment randomization and image-only augmentation cannot both be enabled."
        )

    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        from .gazebo_world_control import (
            GazeboWorldController,
        )
        from .ros_reach_node import So101ReachRosNode
    except ImportError as error:
        raise SimulationRuntimeError(
            "ROS 2 runtime dependencies could not be imported. "
            "Source ROS Humble and run with "
            "PYTHONNOUSERSITE=1."
        ) from error

    owns_ros_context = not rclpy.ok()

    resolved_runtime_config = (
        SimulationRuntimeConfig()
        if runtime_config is None
        else runtime_config
    )

    resolved_reset_config = (
        EpisodeResetConfig()
        if reset_config is None
        else reset_config
    )
    if reward_profile == "pick_place_v2":
        # Reset and scoring share exactly one physical HOME pose.
        if reset_config is None:
            resolved_reset_config = replace(resolved_reset_config,
                home_joint_positions=tuple(reward_config.home_joint_positions_rad))
        elif tuple(reset_config.home_joint_positions) != tuple(reward_config.home_joint_positions_rad):
            raise ValueError("Reset HOME and reward HOME must agree.")

    if randomization_enabled or image_only_enabled:
        randomization_seed = (
            _environment_int(
                "SO101_ENVIRONMENT_RANDOMIZATION_SEED", default=0
            )
            if environment_randomization_seed is None
            else int(environment_randomization_seed)
        )
        randomization_world_index = (
            _environment_int(
                "SO101_ENVIRONMENT_RANDOMIZATION_WORLD_INDEX", default=0
            )
            if environment_randomization_world_index is None
            else int(environment_randomization_world_index)
        )
    if randomization_enabled:
        randomization_config_path = environment_randomization_config_path
        if randomization_config_path is None:
            randomization_config_path = os.environ.get("SO101_ENVIRONMENT_RANDOMIZATION_CONFIG") or None
        randomization_config = EnvironmentRandomizationConfig.load(
            randomization_config_path
        )
        if reward_profile == "pick_place_v2":
            goal_randomization = randomization_config.payload["goal_region"]
            if any(bool(goal_randomization[field].get("enabled", False)) for field in ("position", "size")):
                raise SimulationRuntimeError("pick_place_v2 requires fixed goal position/size until physical collision randomization is validated.")
            if task_mode == "idle_completed":
                raise SimulationRuntimeError("Completed-idle preparation currently requires randomization disabled.")
        environment_randomizer = EpisodeEnvironmentRandomizer(
            randomization_config,
            base_seed=randomization_seed,
            world_index=randomization_world_index,
        )
        gazebo_randomization_controller = (
            GazeboEnvironmentRandomizationController()
        )
    else:
        environment_randomizer = None
        gazebo_randomization_controller = None

    image_augmenter = None
    if image_only_enabled:
        image_config_path = image_augmentation_config_path
        if image_config_path is None:
            image_config_path = os.environ.get("SO101_IMAGE_AUGMENTATION_CONFIG") or DEFAULT_IMAGE_AUGMENTATION_CONFIG
        image_augmenter = EpisodeImageAugmenter(
            EnvironmentRandomizationConfig.load(image_config_path),
            base_seed=randomization_seed, world_index=randomization_world_index,
        )

    if owns_ros_context:
        rclpy.init(args=ros_args)

    io_node = None
    control_node = None
    executor = None
    executor_thread = None

    try:
        io_node = (So101ReachRosNode(enable_task_contacts=True,
                                   **({"enable_grasp_attachment": True} if assisted_grasp_enabled else {}))
                   if reward_profile == "pick_place_v2" else So101ReachRosNode())
        control_node = Node(
            "so101_simulation_control"
        )

        executor = SingleThreadedExecutor()
        executor.add_node(io_node)

        executor_thread = Thread(
            target=executor.spin,
            name="so101_ros_feedback_executor",
            daemon=True,
        )
        executor_thread.start()

        goal_region_snapshot = (
            io_node.goal_region_cache.wait_until_available(
                resolved_runtime_config.startup_feedback_timeout_s
            )
        )

        if goal_region_snapshot is None:
            raise SimulationRuntimeError(
                "Timed out waiting for the world-frame "
                "goal-region centre."
            )

        schedule = SimulationStepSchedule(
            control_hz=30.0,
            physics_step_s=0.001,
        )
        world_controller = GazeboWorldController(
            control_node
        )
        cube_controller = TargetCubeController(
            control_node
        )
        grasp_attachment_controller = None
        if assisted_grasp_enabled:
            from .gazebo_grasp_attachment import GazeboGraspAttachmentController
            grasp_attachment_controller = GazeboGraspAttachmentController(
                status_provider=io_node.grasp_attachment_state_cache.wait_for_sim_time)
            grasp_attachment_controller.refresh_status()
        environment = So101ReachEnvironmentCore(
            config=environment_config,
            reward_config=reward_config,
            reward_profile=reward_profile,
            task_mode=task_mode,
            observe_after_success_s=observe_after_success_s,
            grasp_attachment_controller=grasp_attachment_controller,
            goal_region_center_world_m=(
                goal_region_snapshot.x,
                goal_region_snapshot.y,
                goal_region_snapshot.z,
            ),
        )
        resolved_top_camera_config = (
            default_simulation_top_camera_processing_config()
            if top_camera_config is None
            else top_camera_config
        )
        top_camera_processor = TopCameraImageProcessor(
            resolved_top_camera_config
        )

        reset_manager = EpisodeResetManager(
            world_controller=world_controller,
            schedule=schedule,
            simulation_clock_cache=(
                io_node.simulation_clock_cache
            ),
            joint_cache=io_node.joint_state_cache,
            publish_joint_targets=(
                io_node.publish_joint_targets
            ),
            config=resolved_reset_config,
        )

        coordinator = (
            DeterministicSimulationStepCoordinator(
                environment=environment,
                world_stepper=world_controller,
                schedule=schedule,
                simulation_clock_cache=(
                    io_node.simulation_clock_cache
                ),
                joint_cache=io_node.joint_state_cache,
                cube_collection_cache=(
                    io_node.target_cube_collection_cache
                ),
                cube_cache=io_node.target_cube_cache,
                top_image_cache=io_node.top_image_cache,
                gripper_image_cache=(
                    io_node.gripper_image_cache
                ),
                publish_joint_targets=(
                    io_node.publish_joint_targets
                ),
                top_camera_processor=(
                    top_camera_processor
                ),
                observation_randomizer=(
                    image_augmenter.apply_observation if image_augmenter is not None
                    else (None if environment_randomizer is None
                          else environment_randomizer.apply_observation)
                ),
                config=coordinator_config,
                contact_cache=getattr(io_node, "contact_cache", None),
                grasp_attachment_controller=grasp_attachment_controller,
            )
        )

        shutdown_ros = None

        if owns_ros_context:
            def shutdown_ros() -> None:
                if rclpy.ok():
                    rclpy.shutdown()

        return So101SimulationRuntime(
            io_node=io_node,
            control_node=control_node,
            executor=executor,
            executor_thread=executor_thread,
            world_controller=world_controller,
            cube_controller=cube_controller,
            reset_manager=reset_manager,
            coordinator=coordinator,
            goal_region_snapshot=goal_region_snapshot,
            config=resolved_runtime_config,
            environment_randomizer=environment_randomizer,
            image_augmenter=image_augmenter,
            gazebo_environment_randomization_controller=(
                gazebo_randomization_controller
            ),
            environment_randomization_warmup_steps=(
                environment_randomization_warmup_steps
            ),
            canonical_goal_half_extent_x_m=(
                environment.reward_config.goal_half_extents_xy_m[0] if reward_profile == "pick_place_v2"
                else environment.reward_config.goal_region_half_extent_x_m
            ),
            canonical_goal_half_extent_y_m=(
                environment.reward_config.goal_half_extents_xy_m[1] if reward_profile == "pick_place_v2"
                else environment.reward_config.goal_region_half_extent_y_m
            ),
            shutdown_ros=shutdown_ros,
        )

    except BaseException:
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=1.0)
            except BaseException:
                pass

        if executor_thread is not None:
            try:
                executor_thread.join(timeout=1.0)
            except BaseException:
                pass

        if io_node is not None:
            try:
                io_node.destroy_node()
            except BaseException:
                pass

        if control_node is not None:
            try:
                control_node.destroy_node()
            except BaseException:
                pass

        if owns_ros_context and rclpy.ok():
            rclpy.shutdown()

        raise
