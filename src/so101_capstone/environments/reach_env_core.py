"""Pure-Python core of the SO-101 Gazebo reaching environment.

This module combines measured feedback, forward kinematics, and reaching
reward calculation. ROS subscriptions, control command publication, image
synchronization, and fixed-period scheduling belong to a later wrapper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np

from so101_capstone.rewards import (
    ReachRewardConfig,
    ReachRewardInput,
    StatefulReachReward,
)
from so101_capstone.robot import compute_grasp_point_pose

from .state_cache import (
    MEASURED_JOINT_NAMES,
    JointStateSnapshot,
    TargetCubeSnapshot,
)


class ReachEnvironmentStateError(RuntimeError):
    """Raised when feedback cannot form a valid environment step."""


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReachEnvironmentConfig:
    """Timing and episode limits for the reaching environment."""

    control_hz: float = 30.0
    max_episode_steps: int = 600
    max_joint_state_age_s: float = 0.2
    max_cube_state_age_s: float = 0.3

    def __post_init__(self) -> None:
        numeric_values = {
            "control_hz": self.control_hz,
            "max_joint_state_age_s": self.max_joint_state_age_s,
            "max_cube_state_age_s": self.max_cube_state_age_s,
        }

        for name, value in numeric_values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive.")

        if self.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive."
            )

        if self.max_joint_state_age_s < 0.0:
            raise ValueError(
                "max_joint_state_age_s must be non-negative."
            )

        if self.max_cube_state_age_s < 0.0:
            raise ValueError(
                "max_cube_state_age_s must be non-negative."
            )

    @property
    def control_period_s(self) -> float:
        return 1.0 / self.control_hz


@dataclass(frozen=True)
class PolicyStateObservation:
    """State portion of the policy observation.

    The later ROS wrapper will add top-camera and gripper-camera RGB images.
    Joint ordering is always ``MEASURED_JOINT_NAMES``.
    """

    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "joint_positions": self.joint_positions.copy(),
            "joint_velocities": self.joint_velocities.copy(),
        }


@dataclass(frozen=True)
class ReachPrivilegedState:
    """Simulation truth used for reward, success, and diagnostics."""

    grasp_position: np.ndarray
    grasp_quaternion_xyzw: np.ndarray
    grasp_rotation: np.ndarray
    target_position: np.ndarray
    target_quaternion_xyzw: np.ndarray
    target_delta: np.ndarray
    distance_m: float


class So101ReachEnvironmentCore:
    """State-transition core for the SO-101 reaching task."""

    def __init__(
        self,
        config: ReachEnvironmentConfig | None = None,
        reward_config: ReachRewardConfig | None = None,
        *,
        goal_region_center_world_m: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        reward_profile: str = "legacy",
        task_mode: str = "pick_place",
        observe_after_success_s: float = 0.0,
        grasp_attachment_controller: Any = None,
    ) -> None:
        self.config = (
            ReachEnvironmentConfig()
            if config is None
            else config
        )
        if reward_profile not in {"legacy", "pick_place_v2"}:
            raise ValueError("Unknown reward_profile.")
        if not np.isfinite(observe_after_success_s) or observe_after_success_s < 0:
            raise ValueError("observe_after_success_s must be finite and non-negative.")
        self.reward_profile = reward_profile
        if grasp_attachment_controller is not None and reward_profile != "pick_place_v2":
            raise ValueError("Rigid grasp assistance requires pick_place_v2 contact evidence.")
        self.grasp_attachment_controller = grasp_attachment_controller
        self.observe_after_success_s = float(observe_after_success_s)
        self.task_mode = task_mode
        self._home_hold_required_s = None
        self._task_episode_index = 0
        self._task_snapshot = None
        self._monitor_start_s = None
        self._previous_command = None
        self._stage_evaluator = None
        self._stage_process_reward = None
        self._stage_process_context_enabled = False
        self._stage_process_config = None
        self._later_stage = None
        self._stage_preparing = False
        self._stage_config = None
        self._last_stage_info = {}
        self._last_task_feedback = None
        self._stage_reward_ledger = None
        self.reward_config = (
            ReachRewardConfig()
            if reward_config is None
            else reward_config
        )

        goal_center = np.asarray(
            goal_region_center_world_m,
            dtype=float,
        )
        if goal_center.shape != (3,):
            raise ValueError(
                "goal_region_center_world_m must have shape (3,)."
            )
        if not np.all(np.isfinite(goal_center)):
            raise ValueError(
                "goal_region_center_world_m must contain only "
                "finite values."
            )

        self._goal_region_center_world_m = _readonly_copy(
            goal_center
        )
        if reward_profile == "legacy":
            if task_mode != "pick_place" or observe_after_success_s:
                raise ValueError("Task modes and success monitoring require pick_place_v2.")
            if not isinstance(self.reward_config, ReachRewardConfig):
                raise TypeError("Legacy profile requires ReachRewardConfig.")
            self._reward_tracker = StatefulReachReward(self.reward_config)
        else:
            from so101_capstone.rewards.pick_place_reward import PickPlaceReward, PickPlaceRewardConfig
            from so101_capstone.rewards.task_state import TaskStateTracker
            from .task_measurements import TaskMeasurementBuilder, TaskMeasurementConfig
            self.reward_config = PickPlaceRewardConfig() if reward_config is None else reward_config
            if not isinstance(self.reward_config, PickPlaceRewardConfig):
                raise TypeError("pick_place_v2 requires PickPlaceRewardConfig.")
            if grasp_attachment_controller is not None and self.reward_config.grasp_evidence_mode != "contact":
                raise ValueError("Rigid grasp assistance requires real bilateral contact evidence.")
            if not np.isclose(self.reward_config.control_hz, self.config.control_hz):
                raise ValueError("Reward and environment control frequencies must match.")
            measurement_config = TaskMeasurementConfig(**{
                name: getattr(self.reward_config, name)
                for name in ("goal_half_extents_xy_m", "goal_yaw_rad", "goal_support_offset_m",
                             "source_support_z_m", "target_half_extents_m", "grasp_evidence_mode")
            }, max_sample_gap_s=self.reward_config.max_evidence_gap_s)
            self._measurement_builder = TaskMeasurementBuilder(measurement_config)
            self._task_tracker = TaskStateTracker(self.reward_config)
            self._pick_place_reward = PickPlaceReward(self.reward_config)
            self.configure_task(task_mode)

        self._episode_active = False
        self._episode_step = 0
        self._target_cube_name: str | None = None
        self._last_joint_sequence = 0
        self._last_cube_sequence = 0
        self._last_cube_source_sequence: int | None = None
        self._last_cube_source_sim_time_s: float | None = None

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    @property
    def episode_step(self) -> int:
        return self._episode_step

    @property
    def target_active(self) -> bool:
        return self._target_cube_name is not None

    @property
    def task_snapshot(self):
        """Latest measured physical history, including preparation contacts."""
        return self._task_snapshot

    @property
    def stage_preparing(self) -> bool:
        return self._stage_preparing

    def sync_grasp_attachment_stage(self) -> None:
        """Keep release semantics tied to formal policy collection only."""
        if self.grasp_attachment_controller is None:
            return
        adapter = self._later_stage if self._later_stage is not None else self._stage_evaluator
        stage = None if self._stage_preparing or adapter is None else adapter.stage
        self.grasp_attachment_controller.configure_training_stage(stage)

    def assert_attachment_reset_allowed(self) -> None:
        """Never erase a physical constraint through a logical episode reset."""
        controller = self.grasp_attachment_controller
        if controller is None:
            return
        metadata = controller.metadata()
        if metadata.get("active") or metadata.get("state") != "detached":
            raise ReachEnvironmentStateError(
                "Cannot reset/delete a target while grasp attachment is active or unconfirmed; rebuild the world."
            )

    def annotate_grasp_attachment(self, info: dict[str, Any]) -> dict[str, Any]:
        """Expose a paused attach acknowledgement without scoring another step.

        Reward/history and stage contexts were already computed for this step.
        Only the base task context and attachment metadata change at the same
        boundary; the tracker, preparation callbacks and returned info share
        the newly annotated snapshot.
        """
        controller = self.grasp_attachment_controller
        if controller is None:
            return info
        if self._task_snapshot is None:
            raise ReachEnvironmentStateError("Attachment annotation requires a measured task snapshot.")
        from so101_capstone.rewards.task_state import encode_task_context
        old_base_size = len(encode_task_context(self._task_snapshot))
        context = list(info["critic_task_context"])
        if len(context) < old_base_size:
            raise ReachEnvironmentStateError("Attachment annotation received a truncated critic task context.")
        previous_snapshot = self._task_snapshot
        m = previous_snapshot.measurements
        annotated = replace(m, **controller.measurement_fields(m.sim_time_s, m.target_name))
        try:
            self._task_snapshot = self._task_tracker.annotate_attachment_measurements(annotated)
        except ValueError as error:
            raise ReachEnvironmentStateError(str(error)) from error
        # Reward objects enforce an exact previous-snapshot chain. They have
        # already scored this transition, so update only matching references
        # after the tracker verified this is metadata-only. Never reset their
        # potentials, dwell, record bonuses or episode ledger. In preparation
        # the dormant full-task reward may still point at frame zero; leave it.
        reward_views = (self._pick_place_reward, self._stage_process_reward,
                        None if self._later_stage is None else self._later_stage.reward)
        for reward_view in reward_views:
            if reward_view is not None and reward_view._previous == previous_snapshot:
                reward_view._previous = self._task_snapshot
        snapshot_info = self._task_snapshot.to_info()
        result = dict(info)
        result.update({key: value for key, value in snapshot_info.items()
                       if key not in {"critic_task_context", "task_context_schema_version"}})
        result["critic_task_context"] = snapshot_info["critic_task_context"] + context[old_base_size:]
        result["task_snapshot"] = snapshot_info
        result["grasp_attachment"] = controller.metadata()
        self._attachment_stage_labels(result)
        return result

    def _attachment_stage_labels(self, info: dict[str, Any]) -> None:
        if self.grasp_attachment_controller is not None and "stage_success" in info:
            used = self.grasp_attachment_controller.metadata()["ever_used"]
            info["stage_unassisted_success"] = bool(info["stage_success"] and not used)
            info["stage_success_physics"] = "rigid_grasp_assisted" if used else "unassisted"

    def begin_stage_preparation(self) -> None:
        if self.reward_profile != "pick_place_v2" or self._task_snapshot is None:
            raise ReachEnvironmentStateError("stage preparation requires a measured task reset")
        if self._task_snapshot.outcome != "running" or self.task_mode != "pick_place":
            raise ReachEnvironmentStateError("stage preparation requires a running pick_place task")
        self._stage_preparing = True
        self._stage_evaluator = self._stage_process_reward = self._later_stage = None
        self.sync_grasp_attachment_stage()
        self._last_stage_info = {}
        self._episode_step = 0
        self._episode_active = True

    def handoff_training_stage(self, stage: str, config=None, *, command_joint_targets_rad=None):
        """Start policy collection without clearing grasp/release provenance."""
        if not self._stage_preparing or self._last_task_feedback is None:
            raise ReachEnvironmentStateError("handoff requires completed physical preparation")
        if stage not in {"S2", "S3", "S4", "S5"}:
            raise ValueError("assisted handoff stage must be S2 through S5")
        self.configure_training_stage(stage, config)
        try:
            self._last_stage_info = self._later_stage.reset(self._task_snapshot, ledger=self._stage_reward_ledger)
        except ValueError as error:
            raise ReachEnvironmentStateError(str(error)) from error
        self._pick_place_reward.reset(self._task_snapshot)
        self._stage_reward_ledger = self._later_stage.reward.ledger
        self._episode_step = 0
        self._episode_active = True
        self._stage_preparing = False
        self.sync_grasp_attachment_stage()
        self._previous_command = (None if command_joint_targets_rad is None else tuple(command_joint_targets_rad))
        joint, cube, ages, reused = self._last_task_feedback
        info = self._task_info(self._task_snapshot, joint_state=joint, cube_state=cube,
                               joint_age_s=ages[0], cube_age_s=ages[1], cube_state_reused=reused)
        info.update(self._last_stage_info)
        info.update(reward_components={}, training_terminated=False, preparation_only=False,
                    terminated=False, truncated=False, stage_handoff=True)
        self._attachment_stage_labels(info)
        return self._build_policy_observation(joint), info

    @property
    def goal_region_center_world_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._goal_region_center_world_m)

    def configure_task(self, task_mode: str, *, seed: int | None = None,
                       n_action_steps: int = 1) -> None:
        """Configure the next reset; no task is silently changed mid-transition."""
        if task_mode not in {"pick_place", "idle_empty", "idle_completed"}:
            raise ValueError("Unknown task_mode.")
        if self.reward_profile == "legacy":
            if task_mode != "pick_place":
                raise ValueError("Idle task modes require pick_place_v2.")
            return
        self.task_mode = task_mode
        self._stage_evaluator = None
        self._stage_process_reward = None
        self._stage_process_context_enabled = False
        self._stage_process_config = None
        self._later_stage = None
        self._stage_preparing = False
        self._stage_config = None
        self._last_stage_info = {}
        self.sync_grasp_attachment_stage()
        low, high = self.reward_config.resolve_home_hold_bounds(n_action_steps)
        self._home_hold_required_s = float(np.random.default_rng(seed).uniform(low, high))

    def configure_training_stage(self, stage: str | None, config=None) -> None:
        """Configure a sampling segment before reset; never alter task outcomes."""
        if stage is None:
            self._stage_evaluator = None
            self._stage_process_reward = None
            self._stage_process_context_enabled = False
            self._stage_process_config = None
            self._later_stage = None
            self._stage_config = None
            self._last_stage_info = {}
            self.sync_grasp_attachment_stage()
            return
        from so101_capstone.rewards.training_stages import StageEvaluator, StageTrainingConfig
        if self.reward_profile != "pick_place_v2" or self.observe_after_success_s:
            raise ValueError("Stage training requires pick_place_v2 without success monitoring.")
        expected_mode = {"S0": "idle_empty", **{f"S{i}": "pick_place" for i in range(1, 6)}}.get(stage)
        if expected_mode is None or self.task_mode != expected_mode:
            raise ValueError("Training stage and physical task mode disagree.")
        resolved = StageTrainingConfig() if config is None else config
        if isinstance(resolved, dict):
            resolved = StageTrainingConfig.from_dict(resolved)
        stage_process_reward = None
        if stage == "S1" and resolved.process_reward is not None:
            from so101_capstone.rewards.stage_process_reward import StageReachReward
            reward_class = StageReachReward
            if resolved.process_reward.schema_version == 4:
                from so101_capstone.rewards.s1_lift_approach_reward import S1LiftApproachReward
                reward_class = S1LiftApproachReward
            stage_process_reward = reward_class(resolved.process_reward, resolved, self.reward_config)
        self._stage_config = resolved
        self._later_stage = None
        if stage in {"S2", "S3", "S4", "S5"}:
            from so101_capstone.rewards.later_stage_training import LaterStageTraining
            if not resolved.enable_later_stage_context:
                raise ValueError("S2--S5 requires enable_later_stage_context for reward history")
            self._later_stage = LaterStageTraining(stage, resolved, self.reward_config)
            self._stage_evaluator = None
        else:
            self._stage_evaluator = StageEvaluator(stage, resolved, self.reward_config)
        self._stage_process_reward = stage_process_reward
        self._stage_process_context_enabled = resolved.process_reward is not None
        self._stage_process_config = resolved.process_reward
        # Keep the full-task HOME requirement deterministic and independent of
        # the sampling-stage dwell, which may be shorter than the task minimum.
        self._home_hold_required_s = self.reward_config.home_hold_min_s
        self.sync_grasp_attachment_stage()

    def set_episode_goal_region(
        self,
        center_world_m: tuple[float, float, float],
        *,
        half_extent_x_m: float,
        half_extent_y_m: float,
    ) -> None:
        """Set the visual-matched goal geometry before the next reset."""

        center = np.asarray(center_world_m, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_world_m must be a finite 3-vector.")
        half_x = float(half_extent_x_m)
        half_y = float(half_extent_y_m)
        if not np.isfinite(half_x) or half_x <= 0.0:
            raise ValueError("half_extent_x_m must be finite and positive.")
        if not np.isfinite(half_y) or half_y <= 0.0:
            raise ValueError("half_extent_y_m must be finite and positive.")

        if self.reward_profile == "pick_place_v2":
            expected = np.asarray(self.reward_config.goal_half_extents_xy_m)
            if not np.allclose((half_x, half_y), expected) or not np.allclose(center, self._goal_region_center_world_m):
                raise ValueError("pick_place_v2 requires fixed, physical-matched goal geometry.")
            return
        self._goal_region_center_world_m = _readonly_copy(center)
        self.reward_config = replace(
            self.reward_config,
            goal_region_half_extent_x_m=half_x,
            goal_region_half_extent_y_m=half_y,
        )
        self._reward_tracker = StatefulReachReward(self.reward_config)

    def _clear_target_tracking(self) -> None:
        self._target_cube_name = None
        self._last_cube_sequence = 0
        self._last_cube_source_sequence = None
        self._last_cube_source_sim_time_s = None

    def _validate_cube_source_order(
        self,
        cube_state: TargetCubeSnapshot,
    ) -> None:
        """Reject out-of-order Gazebo cube-state metadata."""

        source_sequence = cube_state.source_sequence
        source_sim_time_s = cube_state.source_sim_time_s

        if (
            source_sequence is not None
            and self._last_cube_source_sequence is not None
            and source_sequence
            < self._last_cube_source_sequence
        ):
            raise ReachEnvironmentStateError(
                "Target cube source sequence moved backwards."
            )

        if (
            source_sim_time_s is not None
            and self._last_cube_source_sim_time_s is not None
            and source_sim_time_s
            < self._last_cube_source_sim_time_s - 1e-12
        ):
            raise ReachEnvironmentStateError(
                "Target cube simulation time moved backwards."
            )

    def _cube_state_reused(
        self,
        cube_state: TargetCubeSnapshot,
    ) -> bool:
        """Determine whether this step reused one source state."""

        if (
            cube_state.source_sequence is not None
            and self._last_cube_source_sequence is not None
        ):
            return (
                cube_state.source_sequence
                == self._last_cube_source_sequence
            )

        return (
            cube_state.sequence
            == self._last_cube_sequence
        )

    def _validate_freshness(
        self,
        joint_state: JointStateSnapshot,
        cube_state: TargetCubeSnapshot | None,
        now_monotonic_s: float,
        *,
        step_end_sim_time_s: float | None = None,
    ) -> tuple[float, float | None]:
        now = float(now_monotonic_s)

        if not np.isfinite(now):
            raise ValueError(
                "now_monotonic_s must be finite."
            )

        if step_end_sim_time_s is not None and (
            not np.isfinite(step_end_sim_time_s) or step_end_sim_time_s < 0.0
        ):
            raise ValueError("step_end_sim_time_s must be finite and non-negative.")

        joint_age_s = joint_state.age_s(now)

        if joint_age_s > self.config.max_joint_state_age_s:
            raise ReachEnvironmentStateError(
                "Joint feedback is stale: "
                f"{joint_age_s:.6f} s old."
            )

        if cube_state is None:
            return joint_age_s, None

        if step_end_sim_time_s is None:
            # Standalone/non-deterministic callers retain the wall-time guard.
            cube_age_s = cube_state.age_s(now)
        else:
            # Paused learning advances wall time but not the physical scene.
            # The deterministic caller supplies the verified step-end clock;
            # keep the original reception timestamp and age the source pose.
            if cube_state.source_sim_time_s is None:
                raise ReachEnvironmentStateError(
                    "Target cube pose source timestamp is required for simulation-time freshness."
                )
            cube_age_s = step_end_sim_time_s - cube_state.source_sim_time_s
            if cube_age_s < -1e-6:
                raise ReachEnvironmentStateError(
                    "Target cube pose is newer than the completed simulation step."
                )
            cube_age_s = max(0.0, cube_age_s)

        if cube_age_s > self.config.max_cube_state_age_s:
            raise ReachEnvironmentStateError(
                "Target cube feedback is stale: "
                f"{cube_age_s:.6f} s old."
            )

        return joint_age_s, cube_age_s

    def _build_policy_observation(
        self,
        joint_state: JointStateSnapshot,
    ) -> PolicyStateObservation:
        if joint_state.velocities is None:
            raise ReachEnvironmentStateError(
                "Joint velocities are required for the policy "
                "observation."
            )

        positions = np.array(
            [
                joint_state.positions[name]
                for name in MEASURED_JOINT_NAMES
            ],
            dtype=float,
        )
        velocities = np.array(
            [
                joint_state.velocities[name]
                for name in MEASURED_JOINT_NAMES
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(positions)):
            raise ReachEnvironmentStateError(
                "Joint positions contain invalid values."
            )

        if not np.all(np.isfinite(velocities)):
            raise ReachEnvironmentStateError(
                "Joint velocities contain invalid values."
            )

        return PolicyStateObservation(
            joint_positions=_readonly_copy(positions),
            joint_velocities=_readonly_copy(velocities),
        )

    def _build_privileged_state(
        self,
        joint_state: JointStateSnapshot,
        cube_state: TargetCubeSnapshot,
    ) -> ReachPrivilegedState:
        grasp_pose = compute_grasp_point_pose(
            joint_state.positions
        )

        target_delta = (
            cube_state.position - grasp_pose.position
        )
        distance_m = float(
            np.linalg.norm(target_delta)
        )

        return ReachPrivilegedState(
            grasp_position=_readonly_copy(
                grasp_pose.position
            ),
            grasp_quaternion_xyzw=_readonly_copy(
                grasp_pose.quaternion_xyzw
            ),
            grasp_rotation=_readonly_copy(
                grasp_pose.transform[:3, :3]
            ),
            target_position=_readonly_copy(
                cube_state.position
            ),
            target_quaternion_xyzw=_readonly_copy(
                cube_state.quaternion_xyzw
            ),
            target_delta=_readonly_copy(target_delta),
            distance_m=distance_m,
        )

    def _base_info(
        self,
        *,
        joint_state: JointStateSnapshot,
        cube_state: TargetCubeSnapshot | None,
        joint_age_s: float,
        cube_age_s: float | None,
        privileged_state: ReachPrivilegedState | None,
        cube_state_reused: bool,
    ) -> dict[str, Any]:
        return {
            "episode_step": self._episode_step,
            "target_active": self.target_active,
            "target_cube_name": self._target_cube_name,
            "joint_sequence": joint_state.sequence,
            "cube_sequence": (
                None
                if cube_state is None
                else cube_state.sequence
            ),
            "cube_source_sequence": (
                None
                if cube_state is None
                else cube_state.source_sequence
            ),
            "cube_source_sim_time_s": (
                None
                if cube_state is None
                else cube_state.source_sim_time_s
            ),
            "joint_state_age_s": joint_age_s,
            "cube_state_age_s": cube_age_s,
            "cube_state_reused": cube_state_reused,
            "control_hz": self.config.control_hz,
            "control_period_s": self.config.control_period_s,
            "privileged_state": privileged_state,
        }

    def reset_from_feedback(
        self,
        joint_state: JointStateSnapshot,
        cube_state: TargetCubeSnapshot | None = None,
        *,
        now_monotonic_s: float,
        step_end_sim_time_s: float | None = None,
        joint_history: Sequence[JointStateSnapshot] = (),
        cube_history: Sequence[TargetCubeSnapshot] = (),
        contact_cache: Any = None,
        camera_source_times_s: tuple[float | None, float | None] = (None, None),
        command_joint_targets_rad: Sequence[float] | None = None,
    ) -> tuple[PolicyStateObservation, dict[str, Any]]:
        """Start an episode, with or without an active target cube."""

        self.assert_attachment_reset_allowed()
        if self.reward_profile == "pick_place_v2":
            return self._reset_task_from_feedback(
                joint_state, cube_state, now_monotonic_s=now_monotonic_s,
                step_end_sim_time_s=step_end_sim_time_s, joint_history=joint_history,
                cube_history=cube_history, contact_cache=contact_cache,
                camera_source_times_s=camera_source_times_s,
                command_joint_targets_rad=command_joint_targets_rad,
            )

        joint_age_s, cube_age_s = self._validate_freshness(
            joint_state,
            cube_state,
            now_monotonic_s,
        )
        observation = self._build_policy_observation(
            joint_state
        )
        privileged_state = None

        self._episode_active = True
        self._episode_step = 0
        self._last_joint_sequence = joint_state.sequence

        if cube_state is None:
            self._clear_target_tracking()
            self._reward_tracker.reset(
                joint_positions_rad=(
                    observation.joint_positions
                ),
            )
            reward_state = 1
        else:
            privileged_state = self._build_privileged_state(
                joint_state,
                cube_state,
            )
            self._target_cube_name = cube_state.name
            self._last_cube_sequence = cube_state.sequence
            self._last_cube_source_sequence = (
                cube_state.source_sequence
            )
            self._last_cube_source_sim_time_s = (
                cube_state.source_sim_time_s
            )
            self._reward_tracker.reset(
                joint_positions_rad=(
                    observation.joint_positions
                ),
                target_name=cube_state.name,
                target_position_world_m=(
                    privileged_state.target_position
                ),
                grasp_position_world_m=(
                    privileged_state.grasp_position
                ),
                grasp_rotation_world=(
                    privileged_state.grasp_rotation
                ),
                goal_region_center_world_m=(
                    self._goal_region_center_world_m
                ),
            )
            # reset() establishes an existing target's State-2 baseline;
            # the first reward step resolves contact or goal transitions.
            reward_state = 2

        info = self._base_info(
            joint_state=joint_state,
            cube_state=cube_state,
            joint_age_s=joint_age_s,
            cube_age_s=cube_age_s,
            privileged_state=privileged_state,
            cube_state_reused=False,
        )
        info.update(
            {
                "reward_state": reward_state,
                "goal_region_center_world_m": (
                    self._goal_region_center_world_m.copy()
                ),
            }
        )

        return observation, info

    def step_from_feedback(
        self,
        joint_state: JointStateSnapshot,
        cube_state: TargetCubeSnapshot | None = None,
        *,
        now_monotonic_s: float,
        step_end_sim_time_s: float | None = None,
        control_dt_s: float | None = None,
        joint_history: Sequence[JointStateSnapshot] = (),
        cube_history: Sequence[TargetCubeSnapshot] = (),
        contact_cache: Any = None,
        camera_source_times_s: tuple[float | None, float | None] = (None, None),
        command_joint_targets_rad: Sequence[float] | None = None,
    ) -> tuple[
        PolicyStateObservation,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Evaluate one 30 Hz policy step with an optional target."""

        if self.reward_profile == "pick_place_v2":
            return self._step_task_from_feedback(
                joint_state, cube_state, now_monotonic_s=now_monotonic_s,
                step_end_sim_time_s=step_end_sim_time_s, control_dt_s=control_dt_s,
                joint_history=joint_history, cube_history=cube_history,
                contact_cache=contact_cache, camera_source_times_s=camera_source_times_s,
                command_joint_targets_rad=command_joint_targets_rad,
            )

        if not self._episode_active:
            raise ReachEnvironmentStateError(
                "reset_from_feedback() must be called before step."
            )

        if joint_state.sequence <= self._last_joint_sequence:
            raise ReachEnvironmentStateError(
                "A newer joint-state message is required "
                "for each environment step."
            )

        joint_age_s, cube_age_s = self._validate_freshness(
            joint_state,
            cube_state,
            now_monotonic_s,
            step_end_sim_time_s=step_end_sim_time_s,
        )
        observation = self._build_policy_observation(
            joint_state
        )

        privileged_state = None
        cube_state_reused = False
        target_transition = "inactive"

        if cube_state is None:
            if self.target_active:
                target_transition = "removed"
                self._clear_target_tracking()
        else:
            target_was_active = self.target_active
            target_switched = (
                target_was_active
                and cube_state.name
                != self._target_cube_name
            )

            if not target_was_active:
                target_transition = "activated"
                self._target_cube_name = cube_state.name
            elif target_switched:
                target_transition = "switched"
                self._target_cube_name = cube_state.name
                self._last_cube_sequence = 0
                self._last_cube_source_sequence = None
                self._last_cube_source_sim_time_s = None
            else:
                target_transition = "unchanged"
                self._validate_cube_source_order(
                    cube_state
                )
                cube_state_reused = (
                    self._cube_state_reused(cube_state)
                )

            privileged_state = self._build_privileged_state(
                joint_state,
                cube_state,
            )
            self._last_cube_sequence = cube_state.sequence
            self._last_cube_source_sequence = (
                cube_state.source_sequence
            )
            self._last_cube_source_sim_time_s = (
                cube_state.source_sim_time_s
            )

        grasp_pose = compute_grasp_point_pose(
            joint_state.positions
        )
        reward_result = self._reward_tracker.step(
            ReachRewardInput(
                joint_positions_rad=(
                    observation.joint_positions
                ),
                grasp_position_world_m=(
                    grasp_pose.position
                ),
                grasp_rotation_world=(
                    grasp_pose.transform[:3, :3]
                ),
                goal_region_center_world_m=(
                    self._goal_region_center_world_m
                ),
                target_name=(
                    None
                    if cube_state is None
                    else cube_state.name
                ),
                target_position_world_m=(
                    None
                    if cube_state is None
                    else cube_state.position
                ),
            )
        )

        self._episode_step += 1
        terminated = reward_result.success
        truncated = (
            not terminated
            and self._episode_step
            >= self.config.max_episode_steps
        )

        info = self._base_info(
            joint_state=joint_state,
            cube_state=cube_state,
            joint_age_s=joint_age_s,
            cube_age_s=cube_age_s,
            privileged_state=privileged_state,
            cube_state_reused=cube_state_reused,
        )
        info.update(
            {
                "cube_state_age_clock": (
                    "monotonic" if step_end_sim_time_s is None else "simulation"
                ),
                "cube_received_monotonic_age_s": (
                    None if cube_state is None else cube_state.age_s(now_monotonic_s)
                ),
                "distance_m": reward_result.distance_m,
                "progress_m": reward_result.distance_improvement_m,
                "alignment_error_deg": (
                    reward_result.alignment_error_deg
                ),
                "reward_state": int(reward_result.state),
                "reward_state_name": (
                    reward_result.state.name
                ),
                "reward_target_age_steps": (
                    reward_result.target_age_steps
                ),
                "robot_at_home": (
                    reward_result.robot_at_home
                ),
                "target_in_goal_region": (
                    reward_result.target_in_goal_region
                ),
                "success_home_hold_count": (
                    reward_result.success_home_hold_count
                ),
                "success_home_hold_required_steps": (
                    self.reward_config.success_home_hold_steps
                ),
                "success_home_hold_remaining_steps": max(
                    0,
                    self.reward_config.success_home_hold_steps
                    - reward_result.success_home_hold_count,
                ),
                "reward_components": (
                    reward_result.components
                ),
                "reward_events": reward_result.events,
                "success": reward_result.success,
                "target_transition": target_transition,
                "goal_region_center_world_m": (
                    self._goal_region_center_world_m.copy()
                ),
            }
        )

        self._last_joint_sequence = joint_state.sequence

        if terminated or truncated:
            self._episode_active = False

        return (
            observation,
            float(reward_result.reward),
            terminated,
            truncated,
            info,
        )

    def _task_measurements(self, joint_state, cube_state, *, step_end_sim_time_s,
                           joint_history, cube_history, contact_cache, camera_source_times_s,
                           command_joint_targets_rad=None):
        from .task_measurements import TaskMeasurementError
        if step_end_sim_time_s is None:
            raise ReachEnvironmentStateError("pick_place_v2 requires explicit control simulation time.")
        try:
            measurements = self._measurement_builder.update(
                joint_state, cube_state, control_sim_time_s=step_end_sim_time_s,
                joint_history=joint_history, cube_history=cube_history,
                contact_cache=contact_cache, camera_source_times_s=camera_source_times_s,
                goal_center_world_m=self._goal_region_center_world_m,
                command_joint_targets_rad=command_joint_targets_rad,
            )
            if self.grasp_attachment_controller is not None:
                measurements = replace(measurements, **self.grasp_attachment_controller.measurement_fields(
                    measurements.sim_time_s, measurements.target_name))
            return measurements
        except (TaskMeasurementError, ValueError) as error:
            raise ReachEnvironmentStateError(str(error)) from error

    def _task_info(self, snapshot, *, joint_state, cube_state, joint_age_s, cube_age_s,
                   cube_state_reused=False):
        privileged = None if cube_state is None else self._build_privileged_state(joint_state, cube_state)
        info = self._base_info(
            joint_state=joint_state, cube_state=cube_state, joint_age_s=joint_age_s,
            cube_age_s=cube_age_s, privileged_state=privileged, cube_state_reused=cube_state_reused,
        )
        info.update(snapshot.to_info())
        info.update(self._measurement_builder.last_info)
        if self.grasp_attachment_controller is not None:
            info["grasp_attachment"] = self.grasp_attachment_controller.metadata()
        if self.reward_config.reward_schema_version >= 5:
            from so101_capstone.rewards.pick_place_reward import REWARD_HISTORY_CONTEXT_SCHEMA_VERSION
            history = self._pick_place_reward.critic_history_context()
            info["critic_task_context"] = np.concatenate((info["critic_task_context"], history)).tolist()
            info["task_context_schema_version"] += 1
            info["reward_history_context_schema_version"] = REWARD_HISTORY_CONTEXT_SCHEMA_VERSION
        if self._stage_process_context_enabled:
            process_config = self._stage_process_config
            # S0 keeps its original reward and reserves the same critic shape
            # as S1 when both stages share one training value network.
            context = (np.zeros(process_config.context_dim, dtype=np.float32)
                       if self._stage_process_reward is None else self._stage_process_reward.critic_context())
            info["critic_task_context"] = np.concatenate((info["critic_task_context"], context)).tolist()
            info["task_context_schema_version"] += process_config.context_schema_version
            info["stage_process_context_schema_version"] = process_config.context_schema_version
        if self._stage_config is not None and self._stage_config.enable_later_stage_context:
            from so101_capstone.rewards.later_stage_training import (
                LATER_STAGE_CONTEXT_SCHEMA_VERSION, later_stage_critic_context, later_stage_context_dim,
            )
            stage = self._last_stage_info.get("training_stage")
            context = (np.zeros(later_stage_context_dim(), dtype=np.float32) if stage is None else
                       later_stage_critic_context(stage,
                           elapsed_s=float(self._last_stage_info.get("stage_elapsed_s", 0.0)),
                           budget_s=self._stage_config.stage_budget_s(stage), snapshot=snapshot,
                           adapter=self._later_stage))
            info["critic_task_context"] = np.concatenate((info["critic_task_context"], context)).tolist()
            info["task_context_schema_version"] += LATER_STAGE_CONTEXT_SCHEMA_VERSION
            info["later_stage_context_schema_version"] = LATER_STAGE_CONTEXT_SCHEMA_VERSION
        info.update(
            reward_profile=self.reward_profile,
            home_joint_tolerance_rad=self.reward_config.home_joint_tolerance_rad,
            task_snapshot=snapshot.to_info(),
            success=snapshot.outcome == "success",
            reward_events=snapshot.events,
            distance_m=snapshot.target_distance_m if cube_state is not None else None,
            target_in_goal_region=snapshot.target_in_goal,
            goal_region_center_world_m=self._goal_region_center_world_m.copy(),
            monitor_continuing=self._monitor_start_s is not None,
            measurement_config=asdict(self._measurement_builder.config),
            measurement_schema_version=1,
        )
        return info

    def _reset_task_from_feedback(self, joint_state, cube_state, *, now_monotonic_s,
                                   step_end_sim_time_s, joint_history, cube_history,
                                   contact_cache, camera_source_times_s, command_joint_targets_rad):
        if self._later_stage is not None:
            raise ReachEnvironmentStateError("later stages require physical preparation and handoff; reset loses provenance")
        self._stage_preparing = False
        ages = self._validate_freshness(joint_state, cube_state, now_monotonic_s,
                                        step_end_sim_time_s=step_end_sim_time_s)
        observation = self._build_policy_observation(joint_state)
        initial_command = tuple(observation.joint_positions if command_joint_targets_rad is None
                                else command_joint_targets_rad)
        prehistory = ()
        if self._stage_evaluator is not None and cube_state is not None:
            # The runtime held HOME and observed the newly spawned target
            # settle before handing control to ACT. Retain only recent source
            # samples for velocity estimation, never task/event history.
            # One lag window for source/contact alignment plus one velocity
            # pair before the selected pose. The builder separately validates
            # present evidence age and each inter-sample gap.
            lower = step_end_sim_time_s - 3 * self.reward_config.max_evidence_gap_s
            by_time = {item.source_sim_time_s: item for item in (*cube_history, cube_state)
                       if item.name == cube_state.name and item.source_sim_time_s is not None
                       and lower <= item.source_sim_time_s <= step_end_sim_time_s}
            prehistory = tuple(by_time[t] for t in sorted(by_time))
            # A gap early in preparation need not invalidate the newest
            # continuous samples. Never interpolate velocity across that gap.
            for index in range(len(prehistory) - 1, 0, -1):
                gap = prehistory[index].source_sim_time_s - prehistory[index - 1].source_sim_time_s
                if gap > self.reward_config.max_evidence_gap_s + 1e-9:
                    prehistory = prehistory[index:]
                    break
        from .task_measurements import TaskMeasurementError
        try:
            self._measurement_builder.reset(target_history=prehistory)
        except (TaskMeasurementError, ValueError) as error:
            raise ReachEnvironmentStateError(str(error)) from error
        measurements = self._task_measurements(
            joint_state, cube_state, step_end_sim_time_s=step_end_sim_time_s,
            joint_history=joint_history, cube_history=cube_history,
            contact_cache=contact_cache, camera_source_times_s=camera_source_times_s,
            command_joint_targets_rad=initial_command,
        )
        self._task_episode_index += 1
        try:
            snapshot = self._task_tracker.reset(
                measurements, task_mode=self.task_mode,
                task_instance_id=f"episode_{self._task_episode_index}",
                home_hold_required_s=self._home_hold_required_s,
            )
        except ValueError as error:
            raise ReachEnvironmentStateError(str(error)) from error
        self._pick_place_reward.reset(snapshot)
        self._stage_reward_ledger = None
        try:
            stage_info = {} if self._stage_evaluator is None else self._stage_evaluator.reset(snapshot)
            if self._stage_process_reward is not None:
                self._stage_process_reward.reset(snapshot, stage_info)
        except ValueError as error:
            raise ReachEnvironmentStateError(str(error)) from error
        self._last_stage_info = stage_info
        self._last_task_feedback = (joint_state, cube_state, ages, False)
        self._task_snapshot = snapshot
        self._episode_active = True
        self._episode_step = 0
        self._monitor_start_s = None
        self._previous_command = initial_command
        self._last_joint_sequence = joint_state.sequence
        self._clear_target_tracking()
        if cube_state is not None:
            self._target_cube_name = cube_state.name
            self._last_cube_sequence = cube_state.sequence
            self._last_cube_source_sequence = cube_state.source_sequence
            self._last_cube_source_sim_time_s = cube_state.source_sim_time_s
        info = self._task_info(snapshot, joint_state=joint_state, cube_state=cube_state,
                               joint_age_s=ages[0], cube_age_s=ages[1])
        info.update(reward_components={}, training_terminated=False)
        info.update(stage_info)
        info["initial_command_source"] = "measured_joint_fallback" if command_joint_targets_rad is None else "reset_hold_command"
        self._attachment_stage_labels(info)
        return observation, info

    def _step_task_from_feedback(self, joint_state, cube_state, *, now_monotonic_s,
                                 step_end_sim_time_s, control_dt_s, joint_history,
                                 cube_history, contact_cache, camera_source_times_s,
                                 command_joint_targets_rad):
        from so101_capstone.rewards.pick_place_reward import PickPlaceRewardInput
        if not self._episode_active or self._task_snapshot is None:
            raise ReachEnvironmentStateError("reset_from_feedback() must be called before step.")
        if joint_state.sequence <= self._last_joint_sequence:
            raise ReachEnvironmentStateError("A newer joint-state message is required for each step.")
        ages = self._validate_freshness(joint_state, cube_state, now_monotonic_s,
                                        step_end_sim_time_s=step_end_sim_time_s)
        previous = self._task_snapshot
        if step_end_sim_time_s is None:
            raise ReachEnvironmentStateError("pick_place_v2 requires explicit control simulation time.")
        dt = step_end_sim_time_s - previous.sim_time_s
        if dt <= 0 or (control_dt_s is not None and not np.isclose(dt, control_dt_s, atol=1e-6)):
            raise ReachEnvironmentStateError("Task transition dt must match the physical control interval.")
        reused = False
        if cube_state is not None and self.target_active:
            self._validate_cube_source_order(cube_state)
            reused = self._cube_state_reused(cube_state)
        measurements = self._task_measurements(
            joint_state, cube_state, step_end_sim_time_s=step_end_sim_time_s,
            joint_history=joint_history, cube_history=cube_history,
            contact_cache=contact_cache, camera_source_times_s=camera_source_times_s,
            command_joint_targets_rad=command_joint_targets_rad,
        )
        try:
            snapshot = (self._task_tracker.update(measurements, allow_success=False)
                        if self._stage_preparing or self._later_stage is not None
                        else self._task_tracker.update(measurements))
        except ValueError as error:
            raise ReachEnvironmentStateError(str(error)) from error
        if self._stage_preparing:
            if not (measurements.evidence_valid and measurements.joint_measurement_valid
                    and measurements.target_measurement_valid and measurements.target_motion_valid
                    and measurements.contact_valid):
                raise ReachEnvironmentStateError("invalid preparation sensor evidence; discard world")
            self._task_snapshot = snapshot
            self._last_joint_sequence = joint_state.sequence
            if cube_state is not None:
                self._last_cube_sequence = cube_state.sequence
                self._last_cube_source_sequence = cube_state.source_sequence
                self._last_cube_source_sim_time_s = cube_state.source_sim_time_s
            self._last_task_feedback = (joint_state, cube_state, ages, reused)
            self._previous_command = None if command_joint_targets_rad is None else tuple(command_joint_targets_rad)
            info = self._task_info(snapshot, joint_state=joint_state, cube_state=cube_state,
                                  joint_age_s=ages[0], cube_age_s=ages[1], cube_state_reused=reused)
            failed = snapshot.outcome == "task_failure"
            info.update(preparation_only=True, reward_components={}, training_terminated=False,
                        terminated=failed, truncated=False, episode_step=0)
            if failed:
                self._episode_active = False
            return self._build_policy_observation(joint_state), 0.0, failed, False, info
        self._episode_step += 1
        true_terminal = snapshot.outcome in {"success", "task_failure"}
        try:
            stage_info = {} if self._stage_evaluator is None else self._stage_evaluator.update(snapshot)
        except ValueError as error:
            raise ReachEnvironmentStateError(str(error)) from error
        truncated = not true_terminal and (
            bool(stage_info.get("stage_cut")) if self._stage_evaluator is not None
            else False if self._later_stage is not None else self._episode_step >= self.config.max_episode_steps)
        already_monitoring = self._monitor_start_s is not None
        if already_monitoring:
            reward, components = 0.0, {}
            reward_info = {"potential_before": None, "potential_after": None}
        else:
            transition = PickPlaceRewardInput(
                prev_snapshot=previous, next_snapshot=snapshot, dt_s=dt,
                terminated=true_terminal, truncated=truncated,
                command_joint_targets_rad=command_joint_targets_rad,
                previous_command_joint_targets_rad=self._previous_command,
                termination_reason=snapshot.failure_reason or ("success" if true_terminal else None),
            )
            if self._later_stage is not None:
                try:
                    result, stage_info = self._later_stage.step(transition)
                except ValueError as error:
                    raise ReachEnvironmentStateError(str(error)) from error
                truncated = bool(stage_info["stage_cut"]) and not true_terminal
            else:
                result = (self._pick_place_reward.step(transition) if self._stage_process_reward is None
                          else self._stage_process_reward.step(transition, stage_info))
            reward, components = result.reward, result.components
            reward_info = result.to_info()
        if snapshot.success_event and self.observe_after_success_s > 0:
            self._monitor_start_s = step_end_sim_time_s
        terminated = true_terminal
        if self._monitor_start_s is not None:
            # The successful training transition is closed. The remaining
            # physical execution is evaluation only; no reset or rule action.
            terminated = step_end_sim_time_s - self._monitor_start_s >= self.observe_after_success_s
            truncated = False
        self._task_snapshot = snapshot
        self._last_joint_sequence = joint_state.sequence
        self._target_cube_name = None if cube_state is None else cube_state.name
        if cube_state is not None:
            self._last_cube_sequence = cube_state.sequence
            self._last_cube_source_sequence = cube_state.source_sequence
            self._last_cube_source_sim_time_s = cube_state.source_sim_time_s
        self._previous_command = None if command_joint_targets_rad is None else tuple(command_joint_targets_rad)
        self._last_stage_info = stage_info
        self._last_task_feedback = (joint_state, cube_state, ages, reused)
        info = self._task_info(snapshot, joint_state=joint_state, cube_state=cube_state,
                               joint_age_s=ages[0], cube_age_s=ages[1], cube_state_reused=reused)
        info.update(reward_info)
        info.update(stage_info)
        info.update(reward_components=components, training_terminated=true_terminal and not already_monitoring,
                    terminated=terminated, truncated=truncated,
                    monitor_continuing=self._monitor_start_s is not None and not terminated,
                    monitor_elapsed_s=0.0 if self._monitor_start_s is None else step_end_sim_time_s-self._monitor_start_s,
                    monitoring_only=already_monitoring,
                    target_transition="activated" if previous.target_name is None and snapshot.target_name else "unchanged")
        if terminated or truncated:
            self._episode_active = False
        self._attachment_stage_labels(info)
        return self._build_policy_observation(joint_state), float(reward), terminated, truncated, info
