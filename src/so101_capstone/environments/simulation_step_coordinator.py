"""Coordinate one deterministic Gazebo policy step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from time import monotonic, sleep
from typing import Any, Callable, Protocol, Sequence

from .image_cache import (
    RgbImageCache,
    SynchronizedCameraFrames,
    select_synchronized_camera_frames,
)
from .policy_observation import (
    PolicyObservation,
    build_policy_observation,
)
from .reach_env_core import (
    PolicyStateObservation,
    So101ReachEnvironmentCore,
)
from .simulation_clock import (
    SIM_TIME_COMPARISON_TOLERANCE_S,
    SimulationClockCache,
    SimulationClockSnapshot,
)
from .top_camera_processing import TopCameraImageProcessor
from .simulation_step import SimulationStepSchedule
from .state_cache import (
    JointStateCache,
    JointStateSnapshot,
    TargetCubeCollectionCache,
    TargetCubeCollectionSnapshot,
    TargetCubeSnapshot,
    TargetCubeStateCache,
)


class SimulationStepCoordinatorError(RuntimeError):
    """Raised when a deterministic step cannot collect new feedback."""


class WorldStepper(Protocol):
    def step(self, physics_steps: int) -> Any:
        """Advance Gazebo by an integer number of physics steps."""


@dataclass(frozen=True)
class SimulationStepCoordinatorConfig:
    """Timeout and camera constraints for one simulation step."""

    feedback_timeout_s: float = 1.0
    max_image_age_s: float = 0.5
    max_image_skew_s: float = 0.05
    max_sim_step_error_s: float = 1e-6
    # Newly consumed pairs must meet this lag; bounded reuse below is a
    # separate fallback when a new pair misses its callback deadline.
    max_camera_lag_from_step_end_s: float = 0.05

    max_camera_reuse_steps: int = 6
    max_camera_reuse_lag_s: float = 0.30
    max_camera_callback_grace_s: float = 0.015
    def __post_init__(self) -> None:
        values = {
            "feedback_timeout_s": self.feedback_timeout_s,
            "max_image_age_s": self.max_image_age_s,
            "max_image_skew_s": self.max_image_skew_s,
            "max_sim_step_error_s": (
                self.max_sim_step_error_s
            ),
            "max_camera_lag_from_step_end_s": (
                self.max_camera_lag_from_step_end_s
            ),
        }

        for name, value in values.items():
            numeric = float(value)

            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite.")

        if self.feedback_timeout_s <= 0.0:
            raise ValueError(
                "feedback_timeout_s must be positive."
            )

        if self.max_image_age_s < 0.0:
            raise ValueError(
                "max_image_age_s must be non-negative."
            )

        if self.max_image_skew_s < 0.0:
            raise ValueError(
                "max_image_skew_s must be non-negative."
            )

        if self.max_sim_step_error_s < 0.0:
            raise ValueError(
                "max_sim_step_error_s must be "
                "non-negative."
            )

        if self.max_camera_lag_from_step_end_s < 0.0:
            raise ValueError(
                "max_camera_lag_from_step_end_s must "
                "be non-negative."
            )
        if (
            isinstance(self.max_camera_reuse_steps, bool)
            or not isinstance(
                self.max_camera_reuse_steps,
                int,
            )
            or self.max_camera_reuse_steps < 0
        ):
            raise ValueError(
                "max_camera_reuse_steps must be a "
                "non-negative integer."
            )

        reuse_lag_s = float(
            self.max_camera_reuse_lag_s
        )

        if (
            not math.isfinite(reuse_lag_s)
            or reuse_lag_s < 0.0
        ):
            raise ValueError(
                "max_camera_reuse_lag_s must be finite "
                "and non-negative."
            )

        callback_grace_s = float(
            self.max_camera_callback_grace_s
        )

        if (
            not math.isfinite(callback_grace_s)
            or callback_grace_s < 0.0
        ):
            raise ValueError(
                "max_camera_callback_grace_s must be finite "
                "and non-negative."
            )


@dataclass(frozen=True)
class CoordinatedSimulationStep:
    """Complete output collected after one deterministic world step."""

    observation: PolicyObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    physics_steps: int
    start_clock: SimulationClockSnapshot
    end_clock: SimulationClockSnapshot
    joint_state: JointStateSnapshot
    cube_collection_state: TargetCubeCollectionSnapshot
    cube_state: TargetCubeSnapshot | None
    camera_frames: SynchronizedCameraFrames


def _validated_now(value: float | None) -> float:
    now = monotonic() if value is None else float(value)

    if not math.isfinite(now):
        raise ValueError(
            "now_monotonic_s must be finite."
        )

    return now


class DeterministicSimulationStepCoordinator:
    """Publish, advance Gazebo, and collect one new observation.

    ROS callbacks must be processed concurrently by an executor while this
    object waits on the thread-safe caches.
    """

    def __init__(
        self,
        *,
        environment: So101ReachEnvironmentCore,
        world_stepper: WorldStepper,
        schedule: SimulationStepSchedule,
        simulation_clock_cache: SimulationClockCache,
        joint_cache: JointStateCache,
        cube_collection_cache: TargetCubeCollectionCache,
        cube_cache: TargetCubeStateCache,
        top_image_cache: RgbImageCache,
        gripper_image_cache: RgbImageCache,
        publish_joint_targets: Callable[
            [Sequence[float]],
            None,
        ],
        top_camera_processor: (
            TopCameraImageProcessor | None
        ) = None,
        observation_randomizer: (
            Callable[[PolicyObservation], PolicyObservation] | None
        ) = None,
        config: SimulationStepCoordinatorConfig | None = None,
        contact_cache: Any = None,
        grasp_attachment_controller: Any = None,
    ) -> None:
        self.environment = environment
        existing_attachment = getattr(environment, "grasp_attachment_controller", None)
        if (existing_attachment is not None and grasp_attachment_controller is not None
                and existing_attachment is not grasp_attachment_controller):
            raise ValueError("Core and coordinator must share one attachment controller per world.")
        self.grasp_attachment_controller = (existing_attachment if grasp_attachment_controller is None
                                            else grasp_attachment_controller)
        if self.grasp_attachment_controller is not None:
            if getattr(environment, "reward_profile", "legacy") != "pick_place_v2":
                raise ValueError("Rigid grasp assistance requires pick_place_v2 contact evidence.")
            environment.grasp_attachment_controller = self.grasp_attachment_controller
            environment.sync_grasp_attachment_stage()
        self.contact_cache = contact_cache
        self.world_stepper = world_stepper
        self.schedule = schedule
        self.simulation_clock_cache = (
            simulation_clock_cache
        )
        self.joint_cache = joint_cache
        self.cube_collection_cache = (
            cube_collection_cache
        )
        self.cube_cache = cube_cache
        self.top_image_cache = top_image_cache
        self.gripper_image_cache = gripper_image_cache
        self.publish_joint_targets = publish_joint_targets
        self.top_camera_processor = top_camera_processor
        self.observation_randomizer = observation_randomizer
        self.config = (
            SimulationStepCoordinatorConfig()
            if config is None
            else config
        )
        self._last_camera_frames: (
            SynchronizedCameraFrames | None
        ) = None
        self._consecutive_camera_reuse_steps = 0

    def _task_feedback_kwargs(self, camera_frames: SynchronizedCameraFrames) -> dict[str, Any]:
        if getattr(self.environment, "reward_profile", "legacy") != "pick_place_v2":
            return {}
        return {
            "joint_history": self.joint_cache.history(),
            "cube_history": self.cube_cache.history(),
            "contact_cache": self.contact_cache,
            "camera_source_times_s": (
                camera_frames.top.source_timestamp_s,
                camera_frames.gripper.source_timestamp_s,
            ),
        }

    def _build_policy_observation(
        self,
        state: PolicyStateObservation,
        camera_frames: SynchronizedCameraFrames,
    ) -> PolicyObservation:
        """Build policy input while preserving raw cached frames."""

        policy_frames = camera_frames

        if self.top_camera_processor is not None:
            processed_top = (
                self.top_camera_processor.process(
                    camera_frames.top.image
                )
            )

            policy_frames = replace(
                camera_frames,
                top=replace(
                    camera_frames.top,
                    image=processed_top,
                ),
            )

        observation = build_policy_observation(
            state,
            policy_frames,
        )
        if self.observation_randomizer is None:
            return observation
        return self.observation_randomizer(observation)

    @staticmethod
    def _required_latest(
        cache: Any,
        *,
        label: str,
    ) -> Any:
        snapshot = cache.latest()

        if snapshot is None:
            raise SimulationStepCoordinatorError(
                f"No {label} is available."
            )

        return snapshot

    @staticmethod
    def _remaining_time(deadline_s: float) -> float:
        return max(0.0, deadline_s - monotonic())

    def _wait_for_new_joint(
        self,
        *,
        previous_sequence: int,
        deadline_s: float,
    ) -> JointStateSnapshot:
        snapshot = (
            self.joint_cache.wait_for_sequence_greater_than(
                previous_sequence,
                self._remaining_time(deadline_s),
            )
        )

        if snapshot is None:
            raise SimulationStepCoordinatorError(
                "Timed out waiting for new joint feedback."
            )

        return snapshot

    @staticmethod
    def _required_collection_source_sequence(
        snapshot: TargetCubeCollectionSnapshot,
    ) -> int:
        source_sequence = snapshot.source_sequence

        if source_sequence is None:
            raise SimulationStepCoordinatorError(
                "Target-cube collection source sequence "
                "is missing."
            )

        return source_sequence

    def _target_feedback_for_step(
        self,
        *,
        previous_collection: TargetCubeCollectionSnapshot,
        deadline_s: float,
    ) -> tuple[
        TargetCubeCollectionSnapshot,
        TargetCubeSnapshot | None,
    ]:
        """Use the newest available target state without requiring 30 Hz updates.

        The target manager publishes at a lower rate than policy stepping.
        Reuse an unchanged source snapshot, but preserve the existing strict
        matching barrier whenever a newer collection source is available.
        """

        previous_source_sequence = (
            self._required_collection_source_sequence(
                previous_collection
            )
        )
        # The ROS callback updates collection and detail caches separately.
        # Even when reusing a source, another callback can run between our
        # reads. Use the same bounded matching loop as fresh feedback.
        collection, cube = self._wait_for_new_target_feedback(
            previous_local_sequence=previous_collection.sequence - 1,
            previous_source_sequence=(
                previous_source_sequence
            ),
            deadline_s=deadline_s,
            allow_unchanged_source=True,
        )
        if (
            collection.source_sequence == previous_source_sequence
            and collection.cube_names != previous_collection.cube_names
        ):
            raise SimulationStepCoordinatorError(
                "Target-cube collection changed without "
                "advancing its source sequence."
            )
        return collection, cube

    def _wait_for_new_target_feedback(
        self,
        *,
        previous_local_sequence: int,
        previous_source_sequence: int,
        deadline_s: float,
        allow_unchanged_source: bool = False,
    ) -> tuple[
        TargetCubeCollectionSnapshot,
        TargetCubeSnapshot | None,
    ]:
        """Wait for one new collection and matching cube details."""

        collection_local_sequence = (
            previous_local_sequence
        )

        while True:
            collection = (
                self.cube_collection_cache
                .wait_for_sequence_greater_than(
                    collection_local_sequence,
                    self._remaining_time(deadline_s),
                )
            )

            if collection is None:
                raise SimulationStepCoordinatorError(
                    "Timed out waiting for a new Gazebo "
                    "target-cube source collection state."
                )

            collection_local_sequence = (
                collection.sequence
            )
            collection_source_sequence = (
                self._required_collection_source_sequence(
                    collection
                )
            )

            if (
                collection_source_sequence
                < previous_source_sequence
            ):
                raise SimulationStepCoordinatorError(
                    "Target-cube collection source sequence "
                    "moved backwards."
                )

            if (
                collection_source_sequence
                == previous_source_sequence
                and not allow_unchanged_source
            ):
                continue

            if collection.cube_count == 0:
                return collection, None

            cube_state = self.cube_cache.latest()
            cube_local_sequence = (
                0
                if cube_state is None
                else cube_state.sequence
            )

            while True:
                if cube_state is None:
                    cube_state = (
                        self.cube_cache
                        .wait_for_sequence_greater_than(
                            0,
                            self._remaining_time(
                                deadline_s
                            ),
                        )
                    )
                elif cube_state.source_sequence is None:
                    raise SimulationStepCoordinatorError(
                        "Target-cube source sequence is "
                        "missing."
                    )
                elif (
                    cube_state.source_sequence
                    < collection_source_sequence
                ):
                    cube_state = (
                        self.cube_cache
                        .wait_for_sequence_greater_than(
                            cube_local_sequence,
                            self._remaining_time(
                                deadline_s
                            ),
                        )
                    )
                elif (
                    cube_state.source_sequence
                    > collection_source_sequence
                ):
                    # A newer callback has already updated the
                    # detailed cache. Wait for its corresponding
                    # collection update.
                    break
                else:
                    if (
                        cube_state.name
                        not in collection.cube_names
                    ):
                        raise SimulationStepCoordinatorError(
                            "Detailed target cube is not "
                            "present in its collection state."
                        )

                    return collection, cube_state

                if cube_state is None:
                    raise SimulationStepCoordinatorError(
                        "Timed out waiting for detailed "
                        "target-cube feedback."
                    )

                cube_local_sequence = (
                    cube_state.sequence
                )

    @staticmethod
    def _validated_optional_sequence(
        value: int | None,
        *,
        name: str,
    ) -> int | None:
        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer or None."
            )

        return value

    def prepare_episode_from_latest(
        self,
        *,
        after_joint_sequence: int | None = None,
        after_target_collection_sequence: int | None = None,
        after_target_collection_source_sequence: (
            int | None
        ) = None,
        after_top_sequence: int | None = None,
        after_gripper_sequence: int | None = None,
        episode_clock_state: (
            SimulationClockSnapshot | None
        ) = None,
        now_monotonic_s: float | None = None,
        reset_command_joint_targets_rad: Sequence[float] | None = None,
    ) -> tuple[PolicyObservation, dict[str, Any]]:
        """Initialize episode state from fresh post-HOME feedback.

        This method does not publish an action, advance a policy/reward step,
        or alter target entities. When sequence markers are provided, frame-0
        must use feedback newer than the pre-reset snapshots.
        """

        if self.grasp_attachment_controller is not None:
            self.environment.assert_attachment_reset_allowed()
        self._last_preparation_step = None

        joint_sequence = self._validated_optional_sequence(
            after_joint_sequence,
            name="after_joint_sequence",
        )
        target_collection_sequence = (
            self._validated_optional_sequence(
                after_target_collection_sequence,
                name="after_target_collection_sequence",
            )
        )
        target_collection_source_sequence = (
            self._validated_optional_sequence(
                after_target_collection_source_sequence,
                name=(
                    "after_target_collection_source_sequence"
                ),
            )
        )

        if (
            target_collection_sequence is None
        ) != (
            target_collection_source_sequence is None
        ):
            raise ValueError(
                "Target-collection local and source sequence "
                "markers must be provided together."
            )

        top_sequence = self._validated_optional_sequence(
            after_top_sequence,
            name="after_top_sequence",
        )
        gripper_sequence = self._validated_optional_sequence(
            after_gripper_sequence,
            name="after_gripper_sequence",
        )

        deadline_s = (
            monotonic()
            + self.config.feedback_timeout_s
        )

        if joint_sequence is None:
            joint_state = self._required_latest(
                self.joint_cache,
                label="joint feedback",
            )
        else:
            joint_state = (
                self.joint_cache
                .wait_for_sequence_greater_than(
                    joint_sequence,
                    self._remaining_time(deadline_s),
                )
            )

            if joint_state is None:
                raise SimulationStepCoordinatorError(
                    "Timed out waiting for post-reset "
                    "joint feedback."
                )

        if target_collection_sequence is None:
            cube_collection_state = self._required_latest(
                self.cube_collection_cache,
                label="target-cube collection feedback",
            )
            cube_collection_state, cube_state = self._wait_for_new_target_feedback(
                previous_local_sequence=cube_collection_state.sequence - 1,
                previous_source_sequence=self._required_collection_source_sequence(
                    cube_collection_state
                ),
                deadline_s=deadline_s,
                allow_unchanged_source=True,
            )
        else:
            assert target_collection_source_sequence is not None

            (
                cube_collection_state,
                cube_state,
            ) = self._wait_for_new_target_feedback(
                previous_local_sequence=(
                    target_collection_sequence
                ),
                previous_source_sequence=(
                    target_collection_source_sequence
                ),
                deadline_s=deadline_s,
            )

        clock_state = (
            self._required_latest(
                self.simulation_clock_cache,
                label="simulation clock",
            )
            if episode_clock_state is None
            else episode_clock_state
        )

        if top_sequence is not None:
            top_frame = (
                self.top_image_cache
                .wait_for_sequence_greater_than(
                    top_sequence,
                    self._remaining_time(deadline_s),
                )
            )

            if top_frame is None:
                raise SimulationStepCoordinatorError(
                    "Timed out waiting for a post-reset "
                    "top-camera frame."
                )

        if gripper_sequence is not None:
            gripper_frame = (
                self.gripper_image_cache
                .wait_for_sequence_greater_than(
                    gripper_sequence,
                    self._remaining_time(deadline_s),
                )
            )

            if gripper_frame is None:
                raise SimulationStepCoordinatorError(
                    "Timed out waiting for a post-reset "
                    "gripper-camera frame."
                )

        # Preserve legacy reset semantics. The new task profile additionally
        # binds frame zero to the current physical reset clock: a late callback
        # carrying an old or pre-rebase future image is not a fresh observation.
        minimum_camera_source_time_s = None
        maximum_camera_source_time_s = None
        if getattr(self.environment, "reward_profile", "legacy") == "pick_place_v2":
            minimum_camera_source_time_s = clock_state.sim_time_s - self.config.max_camera_lag_from_step_end_s
            maximum_camera_source_time_s = clock_state.sim_time_s + self.config.max_sim_step_error_s

        now = _validated_now(now_monotonic_s)

        while True:
            try:
                camera_frames = (
                    select_synchronized_camera_frames(
                        self.top_image_cache,
                        self.gripper_image_cache,
                        now_monotonic_s=now,
                        max_age_s=(
                            self.config.max_image_age_s
                        ),
                        max_skew_s=(
                            self.config.max_image_skew_s
                        ),
                        after_top_sequence=top_sequence,
                        after_gripper_sequence=(
                            gripper_sequence
                        ),
                        min_source_timestamp_s=(
                            minimum_camera_source_time_s
                        ),
                        max_source_timestamp_s=(
                            maximum_camera_source_time_s
                        ),
                    )
                )
                break
            except RuntimeError as error:
                remaining_s = self._remaining_time(
                    deadline_s
                )

                if remaining_s <= 0.0:
                    raise SimulationStepCoordinatorError(
                        "Timed out waiting for synchronized "
                        "post-reset camera frames."
                    ) from error

                sleep(min(0.001, remaining_s))

                if now_monotonic_s is None:
                    now = _validated_now(None)

        self._last_camera_frames = camera_frames
        self._consecutive_camera_reuse_steps = 0
        state_observation, core_info = (
            self.environment.reset_from_feedback(
                joint_state,
                cube_state,
                now_monotonic_s=now,
                **({"step_end_sim_time_s": clock_state.sim_time_s,
                    "command_joint_targets_rad": reset_command_joint_targets_rad,
                    **self._task_feedback_kwargs(camera_frames)}
                   if getattr(self.environment, "reward_profile", "legacy") == "pick_place_v2" else {}),
            )
        )

        self.schedule.reset()

        observation = self._build_policy_observation(
            state_observation,
            camera_frames,
        )

        info = dict(core_info)
        if getattr(self.environment, "reward_profile", "legacy") == "pick_place_v2":
            info["sensor_sync_config"] = asdict(self.config)
        info.update(
            {
                "episode_prepared": True,
                "frame_zero": True,
                "top_image_sequence": (
                    camera_frames.top.sequence
                ),
                "gripper_image_sequence": (
                    camera_frames.gripper.sequence
                ),
                "camera_skew_s": camera_frames.skew_s,
                "target_collection_sequence": (
                    cube_collection_state.sequence
                ),
                "target_collection_source_sequence": (
                    cube_collection_state.source_sequence
                ),
                "target_collection_source_sim_time_s": (
                    cube_collection_state
                    .source_sim_time_s
                ),
                "target_collection_cube_count": (
                    cube_collection_state.cube_count
                ),
                "target_collection_cube_names": (
                    cube_collection_state.cube_names
                ),
                "reset_sim_time_s": clock_state.sim_time_s,
                "reset_clock_sequence": (
                    clock_state.sequence
                ),
            }
        )

        return observation, info

    def reset_from_latest(
        self,
        *,
        now_monotonic_s: float | None = None,
    ) -> tuple[PolicyObservation, dict[str, Any]]:
        """Backward-compatible preparation without freshness markers."""

        return self.prepare_episode_from_latest(
            now_monotonic_s=now_monotonic_s,
        )

    def handoff_training_stage(self, stage: str, config=None, *,
                               command_joint_targets_rad=None):
        """Reuse verified final preparation feedback without a physical reset.

        The final controller step already synchronizes cameras/contacts/poses.
        No simulated time may pass between that measurement and the boundary.
        In particular, reset_from_feedback must not erase its real grasp trace.
        """
        previous = getattr(self, "_last_preparation_step", None)
        if previous is None or not previous.info.get("preparation_only"):
            raise SimulationStepCoordinatorError("stage handoff requires a measured preparation step")
        if previous.terminated or previous.truncated:
            raise SimulationStepCoordinatorError("cannot hand off a failed preparation")
        clock = self._required_latest(self.simulation_clock_cache, label="simulation clock")
        if abs(clock.sim_time_s - previous.end_clock.sim_time_s) > 1e-9:
            raise SimulationStepCoordinatorError("simulation advanced after the preparation boundary")
        if previous.info["camera_lag_from_step_end_s"] > self.config.max_camera_lag_from_step_end_s + SIM_TIME_COMPARISON_TOLERANCE_S:
            raise SimulationStepCoordinatorError("preparation camera frames are too old for frame zero")
        state, core_info = self.environment.handoff_training_stage(
            stage, config, command_joint_targets_rad=command_joint_targets_rad)
        observation = self._build_policy_observation(state, previous.camera_frames)
        info = dict(previous.info)
        info.update(core_info)
        info.update(episode_prepared=True, frame_zero=True, reset_sim_time_s=clock.sim_time_s,
                    reset_clock_sequence=clock.sequence, initial_command_source="stage_preparation_hold_command")
        self.schedule.reset()
        self._consecutive_camera_reuse_steps = 0
        self._last_preparation_step = None
        return observation, info

    def step(
        self,
        joint_targets: Sequence[float],
        *,
        now_monotonic_s: float | None = None,
    ) -> CoordinatedSimulationStep:
        """Execute one deterministic control and feedback cycle."""

        if self.grasp_attachment_controller is not None:
            self.environment.sync_grasp_attachment_stage()
            self.grasp_attachment_controller.before_step(joint_targets, self.environment.task_snapshot)
        previous_joint = self._required_latest(
            self.joint_cache,
            label="joint feedback",
        )
        previous_cube_collection = self._required_latest(
            self.cube_collection_cache,
            label="target-cube collection feedback",
        )
        previous_top = self._required_latest(
            self.top_image_cache,
            label="top-camera frame",
        )
        previous_gripper = self._required_latest(
            self.gripper_image_cache,
            label="gripper-camera frame",
        )
        previous_clock = self._required_latest(
            self.simulation_clock_cache,
            label="simulation clock",
        )

        previous_cube_source_sequence = (
            self._required_collection_source_sequence(
                previous_cube_collection
            )
        )

        self.publish_joint_targets(joint_targets)

        physics_steps = self.schedule.next_step_count()
        expected_sim_delta_s = (
            physics_steps
            * self.schedule.physics_step_s
        )
        expected_end_sim_time_s = (
            previous_clock.sim_time_s
            + expected_sim_delta_s
        )

        self.world_stepper.step(physics_steps)

        deadline_s = (
            monotonic()
            + self.config.feedback_timeout_s
        )

        end_clock = (
            self.simulation_clock_cache.wait_until_at_least(
                expected_end_sim_time_s,
                self._remaining_time(deadline_s),
                after_sequence=previous_clock.sequence,
            )
        )

        if end_clock is None:
            raise SimulationStepCoordinatorError(
                "Timed out waiting for the simulation clock "
                "to reach the requested step end."
            )

        actual_sim_delta_s = (
            end_clock.sim_time_s
            - previous_clock.sim_time_s
        )
        sim_step_error_s = (
            actual_sim_delta_s
            - expected_sim_delta_s
        )

        if (
            abs(sim_step_error_s)
            > self.config.max_sim_step_error_s
        ):
            raise SimulationStepCoordinatorError(
                "Gazebo simulation-time delta does not match "
                "the requested physics-step count: "
                f"expected {expected_sim_delta_s:.9f} s, "
                f"received {actual_sim_delta_s:.9f} s."
            )

        joint_state = self._wait_for_new_joint(
            previous_sequence=previous_joint.sequence,
            deadline_s=deadline_s,
        )

        (
            cube_collection_state,
            cube_state,
        ) = self._target_feedback_for_step(
            previous_collection=(
                previous_cube_collection
            ),
            deadline_s=deadline_s,
        )
        target_collection_updated = (
            self._required_collection_source_sequence(
                cube_collection_state
            )
            > previous_cube_source_sequence
        )

        if cube_state is not None:
            source_time = cube_state.source_sim_time_s
            if source_time is None:
                raise SimulationStepCoordinatorError(
                    "Target-cube pose source timestamp is missing."
                )
            # Manager publication/callback times do not make an old Gazebo
            # pose fresh. Simulation time stays fixed during paused learning.
            pose_lag_s = end_clock.sim_time_s - source_time
            if pose_lag_s > self.environment.config.max_cube_state_age_s:
                raise SimulationStepCoordinatorError(
                    "Target-cube pose source is stale: "
                    f"lag={pose_lag_s:.6f} s."
                )
            if pose_lag_s < -self.config.max_sim_step_error_s:
                raise SimulationStepCoordinatorError(
                    "Target-cube pose is newer than the completed simulation step."
                )

        now = _validated_now(now_monotonic_s)

        timestamp_tolerance_s = (
            self.config.max_sim_step_error_s
        )
        if self._last_camera_frames is None:
            raise SimulationStepCoordinatorError(
                "No previously consumed synchronized camera "
                "pair is available."
            )

        consumed_top_sequence = (
            self._last_camera_frames.top.sequence
        )
        consumed_gripper_sequence = (
            self._last_camera_frames.gripper.sequence
        )

        # Use the same fixed numerical tolerance here and in the final lag
        # check: end - (end - limit) can round slightly above the limit.
        minimum_camera_source_time_s = max(
            0.0,
            end_clock.sim_time_s
            - self.config.max_camera_lag_from_step_end_s
            - SIM_TIME_COMPARISON_TOLERANCE_S,
        )
        maximum_camera_source_time_s = (
            end_clock.sim_time_s
            + timestamp_tolerance_s
        )

        camera_pair_reused = False
        camera_callback_grace_used = False

        while True:
            try:
                camera_frames = (
                    select_synchronized_camera_frames(
                        self.top_image_cache,
                        self.gripper_image_cache,
                        now_monotonic_s=now,
                        max_age_s=(
                            self.config.max_image_age_s
                        ),
                        max_skew_s=(
                            self.config.max_image_skew_s
                        ),
                        after_top_sequence=(
                            consumed_top_sequence
                        ),
                        after_gripper_sequence=(
                            consumed_gripper_sequence
                        ),
                        min_source_timestamp_s=(
                            minimum_camera_source_time_s
                        ),
                        max_source_timestamp_s=(
                            maximum_camera_source_time_s
                        ),
                    )
                )
                self._last_camera_frames = camera_frames
                self._consecutive_camera_reuse_steps = 0
                break
            except RuntimeError as error:
                # The simulator step has completed, but the
                # Gazebo-to-ROS image callback can arrive a few
                # milliseconds later in wall time. Allow one
                # short callback grace period, retry once, and
                # only then fall back to bounded frame reuse.
                if (
                    not camera_callback_grace_used
                    and self.config.max_camera_callback_grace_s
                    > 0.0
                ):
                    sleep(
                        self.config.max_camera_callback_grace_s
                    )
                    camera_callback_grace_used = True

                    if now_monotonic_s is None:
                        now = _validated_now(None)

                    continue

                if self._last_camera_frames is None:
                    raise SimulationStepCoordinatorError(
                        "No new synchronized camera pair is "
                        "available after the callback grace "
                        "period, and no previous pair exists "
                        "for bounded reuse."
                    ) from error

                next_reuse_count = (
                    self._consecutive_camera_reuse_steps
                    + 1
                )

                if (
                    next_reuse_count
                    > self.config.max_camera_reuse_steps
                ):
                    raise SimulationStepCoordinatorError(
                        "Camera frame reuse limit exceeded: "
                        f"attempted {next_reuse_count} consecutive "
                        "steps, configured maximum is "
                        f"{self.config.max_camera_reuse_steps}."
                    ) from error

                camera_frames = self._last_camera_frames
                self._consecutive_camera_reuse_steps = (
                    next_reuse_count
                )
                camera_pair_reused = True
                break
        top_source_time = (
            camera_frames.top.source_timestamp_s
        )
        gripper_source_time = (
            camera_frames.gripper.source_timestamp_s
        )

        if (
            top_source_time is None
            or gripper_source_time is None
        ):
            raise SimulationStepCoordinatorError(
                "Gazebo camera source timestamps are required "
                "for deterministic simulation stepping."
            )

        camera_times = {
            "top": float(top_source_time),
            "gripper": float(gripper_source_time),
        }

        for camera_name, camera_time_s in camera_times.items():
            if (
                not camera_pair_reused
                and camera_time_s
                < minimum_camera_source_time_s
                - timestamp_tolerance_s
            ):
                raise SimulationStepCoordinatorError(
                    f"{camera_name} camera frame predates "
                    "the allowed camera-lag window."
                )

            if (
                camera_time_s
                > end_clock.sim_time_s
                + timestamp_tolerance_s
            ):
                raise SimulationStepCoordinatorError(
                    f"{camera_name} camera frame is newer "
                    "than the completed simulation step."
                )

        top_camera_lag_s = (
            end_clock.sim_time_s
            - float(top_source_time)
        )
        gripper_camera_lag_s = (
            end_clock.sim_time_s
            - float(gripper_source_time)
        )
        camera_lag_from_step_end_s = max(
            top_camera_lag_s,
            gripper_camera_lag_s,
        )

        maximum_allowed_camera_lag_s = (
            self.config.max_camera_reuse_lag_s
            if camera_pair_reused
            else self.config.max_camera_lag_from_step_end_s
        )

        if (
            camera_lag_from_step_end_s
            > maximum_allowed_camera_lag_s
            + SIM_TIME_COMPARISON_TOLERANCE_S
        ):
            label = (
                "Reused camera frames"
                if camera_pair_reused
                else "Camera frames"
            )
            raise SimulationStepCoordinatorError(
                f"{label} are too old relative to the "
                "completed simulation step: "
                f"lag={camera_lag_from_step_end_s:.17g} s, "
                f"limit={maximum_allowed_camera_lag_s:.17g} s, "
                f"excess={camera_lag_from_step_end_s - maximum_allowed_camera_lag_s:.17g} s, "
                f"comparison_tolerance={SIM_TIME_COMPARISON_TOLERANCE_S:.17g} s, "
                f"step_end={end_clock.sim_time_s:.17g} s, "
                f"top_source={top_source_time:.17g} s, "
                f"gripper_source={gripper_source_time:.17g} s."
            )

        (
            state_observation,
            reward,
            terminated,
            truncated,
            core_info,
        ) = self._step_environment_with_attachment(
            joint_state,
            cube_state,
            now_monotonic_s=now,
            step_end_sim_time_s=end_clock.sim_time_s,
            **({"control_dt_s": actual_sim_delta_s,
                "command_joint_targets_rad": tuple(joint_targets),
                **self._task_feedback_kwargs(camera_frames)}
               if getattr(self.environment, "reward_profile", "legacy") == "pick_place_v2" else {}),
        )

        observation = self._build_policy_observation(
            state_observation,
            camera_frames,
        )

        info = dict(core_info)
        if getattr(self.environment, "reward_profile", "legacy") == "pick_place_v2":
            info["sensor_sync_config"] = asdict(self.config)
        info.update(
            {
                "physics_steps": physics_steps,
                "scheduled_sim_time_s": (
                    self.schedule.elapsed_sim_time_s
                ),
                "schedule_timing_error_s": (
                    self.schedule.timing_error_s
                ),
                "step_start_sim_time_s": (
                    previous_clock.sim_time_s
                ),
                "step_end_sim_time_s": (
                    end_clock.sim_time_s
                ),
                "expected_sim_delta_s": (
                    expected_sim_delta_s
                ),
                "actual_sim_delta_s": (
                    actual_sim_delta_s
                ),
                "sim_step_error_s": sim_step_error_s,
                "top_camera_lag_from_step_end_s": (
                    top_camera_lag_s
                ),
                "gripper_camera_lag_from_step_end_s": (
                    gripper_camera_lag_s
                ),
                "camera_lag_from_step_end_s": (
                    camera_lag_from_step_end_s
                ),
                "camera_pair_reused": (
                    camera_pair_reused
                ),
                "consecutive_camera_reuse_steps": (
                    self._consecutive_camera_reuse_steps
                ),
                "max_camera_reuse_steps": (
                    self.config.max_camera_reuse_steps
                ),
                "max_camera_reuse_lag_s": (
                    self.config.max_camera_reuse_lag_s
                ),
                "max_camera_lag_from_step_end_s": (
                    self.config.max_camera_lag_from_step_end_s
                ),
                "max_camera_callback_grace_s": (
                    self.config.max_camera_callback_grace_s
                ),
                "top_image_sequence": (
                    camera_frames.top.sequence
                ),
                "gripper_image_sequence": (
                    camera_frames.gripper.sequence
                ),
                "top_image_source_timestamp_s": (
                    camera_frames.top.source_timestamp_s
                ),
                "gripper_image_source_timestamp_s": (
                    camera_frames.gripper.source_timestamp_s
                ),
                "camera_skew_s": camera_frames.skew_s,
                "target_collection_sequence": (
                    cube_collection_state.sequence
                ),
                "target_collection_source_sequence": (
                    cube_collection_state.source_sequence
                ),
                "target_collection_source_sim_time_s": (
                    cube_collection_state
                    .source_sim_time_s
                ),
                "target_collection_cube_count": (
                    cube_collection_state.cube_count
                ),
                "target_collection_cube_names": (
                    cube_collection_state.cube_names
                ),
                "target_collection_updated": (
                    target_collection_updated
                ),
            }
        )

        result = CoordinatedSimulationStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            physics_steps=physics_steps,
            start_clock=previous_clock,
            end_clock=end_clock,
            joint_state=joint_state,
            cube_collection_state=(
                cube_collection_state
            ),
            cube_state=cube_state,
            camera_frames=camera_frames,
        )
        if info.get("preparation_only"):
            self._last_preparation_step = result
        return result

    def _step_environment_with_attachment(self, *args, **kwargs):
        """Check constraint evidence and annotate attach at this step boundary."""
        controller = self.grasp_attachment_controller
        if controller is not None and controller.metadata()["active"]:
            controller.refresh_status(minimum_sim_time_s=kwargs["step_end_sim_time_s"])
        result = self.environment.step_from_feedback(*args, **kwargs)
        if controller is None:
            return result
        command = kwargs.get("command_joint_targets_rad")
        if command is None:
            raise SimulationStepCoordinatorError("Rigid grasp assistance requires the executed joint command.")
        controller.after_step(self.environment.task_snapshot, command)
        observation, reward, terminated, truncated, info = result
        return observation, reward, terminated, truncated, self.environment.annotate_grasp_attachment(info)
