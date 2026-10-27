"""Episode reset coordination for the SO-101 Gazebo environment.

Each reset first moves all measured joints to ZERO, waits for consecutive
stable feedback, then moves to the configured HOME pose and waits for
stability again. Only after both phases complete is the deterministic
33/33/34 policy-step schedule restarted for frame-0.

Target-cube deletion, delayed spawning, observation construction, and
Gymnasium integration belong to later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import monotonic, sleep
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .simulation_clock import (
    SimulationClockCache,
    SimulationClockSnapshot,
)
from .simulation_step import SimulationStepSchedule
from .state_cache import (
    JointStateCache,
    JointStateSnapshot,
    MEASURED_JOINT_NAMES,
)


class EpisodeResetError(RuntimeError):
    """Raised when the robot cannot complete an episode reset."""


class ResetWorldController(Protocol):
    """Gazebo operations required during episode reset."""

    def pause(self) -> Any:
        """Pause continuous simulation."""

    def resume(self) -> Any:
        """Resume continuous simulation."""

    def step(self, physics_steps: int) -> Any:
        """Advance paused simulation by integer physics steps."""


@dataclass(frozen=True)
class EpisodeResetConfig:
    """HOME pose and per-phase stability requirements for episode reset."""

    home_joint_positions: tuple[float, ...] = (
                                                      0.0,
                                                      -1.710422666954443,
                                                      1.605702911834783,
                                                      1.134464013796314,
                                                      0.0,
                                                      -0.087266462599716,
                                                  )
    position_tolerance_rad: float = 0.02
    velocity_tolerance_rad_s: float = 0.05
    stable_sample_count: int = 5
    max_control_steps: int = 300
    feedback_timeout_s: float = 1.0
    pause_clock_settle_s: float = 0.05
    max_sim_step_error_s: float = 1e-6

    def __post_init__(self) -> None:
        home = np.asarray(
            self.home_joint_positions,
            dtype=float,
        )

        if home.shape != (len(MEASURED_JOINT_NAMES),):
            raise ValueError(
                "home_joint_positions must contain exactly "
                f"{len(MEASURED_JOINT_NAMES)} values."
            )

        if not np.all(np.isfinite(home)):
            raise ValueError(
                "home_joint_positions must contain only finite values."
            )

        finite_values = {
            "position_tolerance_rad": self.position_tolerance_rad,
            "velocity_tolerance_rad_s": (
                self.velocity_tolerance_rad_s
            ),
            "feedback_timeout_s": self.feedback_timeout_s,
            "pause_clock_settle_s": self.pause_clock_settle_s,
            "max_sim_step_error_s": self.max_sim_step_error_s,
        }

        for name, raw_value in finite_values.items():
            value = float(raw_value)

            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if self.position_tolerance_rad < 0.0:
            raise ValueError(
                "position_tolerance_rad must be non-negative."
            )

        if self.velocity_tolerance_rad_s < 0.0:
            raise ValueError(
                "velocity_tolerance_rad_s must be non-negative."
            )

        if self.feedback_timeout_s <= 0.0:
            raise ValueError(
                "feedback_timeout_s must be positive."
            )

        if self.pause_clock_settle_s < 0.0:
            raise ValueError(
                "pause_clock_settle_s must be non-negative."
            )

        if (
            self.pause_clock_settle_s
            >= self.feedback_timeout_s
        ):
            raise ValueError(
                "pause_clock_settle_s must be less than "
                "feedback_timeout_s."
            )

        if self.max_sim_step_error_s < 0.0:
            raise ValueError(
                "max_sim_step_error_s must be non-negative."
            )

        if (
            isinstance(self.stable_sample_count, bool)
            or not isinstance(self.stable_sample_count, int)
            or self.stable_sample_count < 1
        ):
            raise ValueError(
                "stable_sample_count must be a positive integer."
            )

        if (
            isinstance(self.max_control_steps, bool)
            or not isinstance(self.max_control_steps, int)
            or self.max_control_steps < 1
        ):
            raise ValueError(
                "max_control_steps must be a positive integer."
            )


@dataclass(frozen=True)
class EpisodeResetResult:
    """Measured state after a successful ZERO-then-HOME reset."""

    joint_state: JointStateSnapshot
    clock_state: SimulationClockSnapshot
    control_steps: int
    physics_steps: int
    stable_sample_count: int
    max_position_error_rad: float
    max_velocity_rad_s: float


class EpisodeResetManager:
    """Move the SO-101 to ZERO, then HOME, with measured stability."""

    def __init__(
        self,
        *,
        world_controller: ResetWorldController,
        schedule: SimulationStepSchedule,
        simulation_clock_cache: SimulationClockCache,
        joint_cache: JointStateCache,
        publish_joint_targets: Callable[
            [Sequence[float]],
            None,
        ],
        config: EpisodeResetConfig | None = None,
    ) -> None:
        self.world_controller = world_controller
        self.schedule = schedule
        self.simulation_clock_cache = simulation_clock_cache
        self.joint_cache = joint_cache
        self.publish_joint_targets = publish_joint_targets
        self.config = (
            EpisodeResetConfig()
            if config is None
            else config
        )

    @staticmethod
    def _required_latest(
        cache: Any,
        *,
        label: str,
    ) -> Any:
        snapshot = cache.latest()

        if snapshot is None:
            raise EpisodeResetError(
                f"No {label} is available before reset."
            )

        return snapshot

    @staticmethod
    def _remaining_time(deadline_s: float) -> float:
        return max(0.0, deadline_s - monotonic())

    def _wait_for_paused_clock_barrier(
        self,
    ) -> SimulationClockSnapshot:
        """Wait until Gazebo simulation time stops advancing.

        Gazebo may continue publishing `/clock` while paused. Those messages
        can increase the local cache sequence even though their simulation
        timestamp is unchanged, so stability must be based on ``sim_time_s``
        rather than message sequence.
        """

        snapshot = self._required_latest(
            self.simulation_clock_cache,
            label="simulation clock",
        )

        settle_duration_s = (
            self.config.pause_clock_settle_s
        )

        if settle_duration_s == 0.0:
            return snapshot

        deadline_s = (
            monotonic()
            + self.config.feedback_timeout_s
        )
        stable_since_s = monotonic()
        stable_sim_time_s = snapshot.sim_time_s

        while True:
            remaining_s = self._remaining_time(
                deadline_s
            )

            if remaining_s <= 0.0:
                raise EpisodeResetError(
                    "Timed out waiting for Gazebo simulation time "
                    "to settle after pausing. "
                    f"Last simulation time: "
                    f"{snapshot.sim_time_s:.9f} s; "
                    f"last clock sequence: {snapshot.sequence}."
                )

            sleep(min(0.005, remaining_s))

            current = self._required_latest(
                self.simulation_clock_cache,
                label="simulation clock",
            )
            now_s = monotonic()

            if (
                current.sim_time_s
                < stable_sim_time_s - 1e-12
            ):
                raise EpisodeResetError(
                    "Simulation time moved backwards while "
                    "establishing the paused-clock barrier."
                )

            if (
                current.sim_time_s
                > stable_sim_time_s + 1e-12
            ):
                stable_sim_time_s = current.sim_time_s
                stable_since_s = now_s

            snapshot = current

            if (
                now_s - stable_since_s
                >= settle_duration_s
            ):
                return snapshot

    def _measure_stability(
        self,
        joint_state: JointStateSnapshot,
        joint_targets: np.ndarray,
    ) -> tuple[float, float]:
        if joint_state.velocities is None:
            raise EpisodeResetError(
                "Joint velocities are required for HOME stability checks."
            )

        positions = np.asarray(
            [
                joint_state.positions[name]
                for name in MEASURED_JOINT_NAMES
            ],
            dtype=float,
        )
        velocities = np.asarray(
            [
                joint_state.velocities[name]
                for name in MEASURED_JOINT_NAMES
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(positions)):
            raise EpisodeResetError(
                "Joint positions contain invalid values during reset."
            )

        if not np.all(np.isfinite(velocities)):
            raise EpisodeResetError(
                "Joint velocities contain invalid values during reset."
            )

        max_position_error = float(
            np.max(np.abs(positions - joint_targets))
        )
        max_velocity = float(
            np.max(np.abs(velocities))
        )

        return max_position_error, max_velocity

    def _move_to_targets(
        self,
        joint_targets: np.ndarray,
        *,
        phase_label: str,
        clock_rebase_used: bool,
    ) -> tuple[
        JointStateSnapshot,
        SimulationClockSnapshot,
        int,
        int,
        int,
        float,
        float,
        bool,
    ]:
        """Move one reset phase to stable measured joint feedback."""

        stable_samples = 0
        phase_physics_steps = 0
        last_position_error = math.inf
        last_velocity = math.inf

        for control_step in range(
            1,
            self.config.max_control_steps + 1,
        ):
            previous_joint = self._required_latest(
                self.joint_cache,
                label="joint feedback",
            )
            previous_clock = self._required_latest(
                self.simulation_clock_cache,
                label="simulation clock",
            )

            self.publish_joint_targets(
                joint_targets.copy()
            )

            physics_steps = self.schedule.next_step_count()
            expected_sim_delta_s = (
                physics_steps
                * self.schedule.physics_step_s
            )
            expected_end_sim_time_s = (
                previous_clock.sim_time_s
                + expected_sim_delta_s
            )

            self.world_controller.step(physics_steps)
            phase_physics_steps += physics_steps

            deadline_s = (
                monotonic()
                + self.config.feedback_timeout_s
            )

            end_clock = (
                self.simulation_clock_cache
                .wait_until_at_least(
                    expected_end_sim_time_s,
                    self._remaining_time(deadline_s),
                    after_sequence=(
                        previous_clock.sequence
                    ),
                )
            )

            if end_clock is None:
                raise EpisodeResetError(
                    "Timed out waiting for /clock during "
                    f"{phase_label} reset."
                )

            actual_sim_delta_s = (
                end_clock.sim_time_s
                - previous_clock.sim_time_s
            )
            sim_step_error_s = (
                actual_sim_delta_s
                - expected_sim_delta_s
            )

            sim_step_mismatch = (
                abs(sim_step_error_s)
                > self.config.max_sim_step_error_s
            )
            single_tick_clock_rebase = (
                sim_step_error_s
                > self.config.max_sim_step_error_s
                and abs(
                    sim_step_error_s
                    - self.schedule.physics_step_s
                )
                <= self.config.max_sim_step_error_s
            )

            if sim_step_mismatch:
                if (
                    single_tick_clock_rebase
                    and not clock_rebase_used
                ):
                    # A delayed /clock callback can expose one
                    # already-completed 1 ms physics tick at the
                    # boundary between consecutive episode resets.
                    #
                    # Reset is outside policy/reward stepping, so
                    # accept this once across the complete
                    # ZERO-then-HOME reset and rebase subsequent
                    # timing on the newly observed clock.
                    clock_rebase_used = True
                else:
                    raise EpisodeResetError(
                        "Gazebo simulation-time delta did not match "
                        "the requested reset physics steps during "
                        f"{phase_label}: "
                        f"start={previous_clock.sim_time_s:.9f} s, "
                        f"end={end_clock.sim_time_s:.9f} s, "
                        f"expected_delta="
                        f"{expected_sim_delta_s:.9f} s, "
                        f"actual_delta={actual_sim_delta_s:.9f} s, "
                        f"error={sim_step_error_s:.9f} s."
                    )

            joint_state = (
                self.joint_cache
                .wait_for_sequence_greater_than(
                    previous_joint.sequence,
                    self._remaining_time(deadline_s),
                )
            )

            if joint_state is None:
                raise EpisodeResetError(
                    "Timed out waiting for new joint feedback "
                    f"during {phase_label} reset."
                )

            (
                last_position_error,
                last_velocity,
            ) = self._measure_stability(
                joint_state,
                joint_targets,
            )

            stable_now = (
                last_position_error
                <= self.config.position_tolerance_rad
                and last_velocity
                <= self.config.velocity_tolerance_rad_s
            )

            if stable_now:
                stable_samples += 1
            else:
                stable_samples = 0

            if (
                stable_samples
                >= self.config.stable_sample_count
            ):
                return (
                    joint_state,
                    end_clock,
                    control_step,
                    phase_physics_steps,
                    stable_samples,
                    last_position_error,
                    last_velocity,
                    clock_rebase_used,
                )

        raise EpisodeResetError(
            f"Robot did not stabilize at {phase_label} within "
            f"{self.config.max_control_steps} control steps. "
            f"Last position error: "
            f"{last_position_error:.6f} rad; "
            f"last velocity: {last_velocity:.6f} rad/s."
        )

    def hold_targets_for_steps(
        self,
        joint_targets: Sequence[float],
        *,
        control_steps: int,
    ) -> tuple[JointStateSnapshot, SimulationClockSnapshot, int]:
        """Hold fixed reset-only targets for an exact number of 30 Hz steps.

        The caller must already own a paused Gazebo world. These steps never
        create policy actions or reward transitions. They are used only for
        post-randomization sensor warm-up and frame-0 priming.
        """

        targets = np.asarray(joint_targets, dtype=float)
        if targets.shape != (len(MEASURED_JOINT_NAMES),):
            raise ValueError(
                "joint_targets must contain exactly "
                f"{len(MEASURED_JOINT_NAMES)} values."
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError("joint_targets must contain only finite values.")
        if (
            isinstance(control_steps, bool)
            or not isinstance(control_steps, int)
            or control_steps < 1
        ):
            raise ValueError("control_steps must be a positive integer.")

        last_joint = self._required_latest(
            self.joint_cache, label="joint feedback"
        )
        last_clock = self._required_latest(
            self.simulation_clock_cache, label="simulation clock"
        )
        total_physics_steps = 0

        try:
            for _ in range(control_steps):
                previous_joint = self._required_latest(
                    self.joint_cache, label="joint feedback"
                )
                previous_clock = self._required_latest(
                    self.simulation_clock_cache, label="simulation clock"
                )

                self.publish_joint_targets(targets.copy())
                physics_steps = self.schedule.next_step_count()
                expected_delta_s = physics_steps * self.schedule.physics_step_s
                expected_end_s = previous_clock.sim_time_s + expected_delta_s
                self.world_controller.step(physics_steps)
                total_physics_steps += physics_steps

                deadline_s = monotonic() + self.config.feedback_timeout_s
                last_clock = self.simulation_clock_cache.wait_until_at_least(
                    expected_end_s,
                    self._remaining_time(deadline_s),
                    after_sequence=previous_clock.sequence,
                )
                if last_clock is None:
                    raise EpisodeResetError(
                        "Timed out waiting for /clock during reset-only hold."
                    )

                actual_delta_s = last_clock.sim_time_s - previous_clock.sim_time_s
                if abs(actual_delta_s - expected_delta_s) > self.config.max_sim_step_error_s:
                    raise EpisodeResetError(
                        "Gazebo simulation-time delta did not match the requested "
                        "reset-only hold physics steps: "
                        f"expected_delta={expected_delta_s:.9f} s, "
                        f"actual_delta={actual_delta_s:.9f} s."
                    )

                last_joint = self.joint_cache.wait_for_sequence_greater_than(
                    previous_joint.sequence,
                    self._remaining_time(deadline_s),
                )
                if last_joint is None:
                    raise EpisodeResetError(
                        "Timed out waiting for joint feedback during reset-only hold."
                    )

            return last_joint, last_clock, total_physics_steps
        finally:
            # Policy frame-0 always starts from a fresh 33/33/34 schedule.
            self.schedule.reset()

    def reset_to_home(
        self,
        *,
        home_joint_positions_override: Sequence[float] | None = None,
    ) -> EpisodeResetResult:
        """Pause Gazebo, move ZERO then HOME, and wait for stability.

        Both ZERO and HOME are reset-only phases: they do not create policy
        steps or reward transitions. A successful reset intentionally leaves
        Gazebo paused so the deterministic policy coordinator can take
        ownership of simulation stepping. If reset fails, Gazebo is resumed
        before the error is propagated.
        """

        zero_targets = np.zeros(
            len(MEASURED_JOINT_NAMES),
            dtype=float,
        )
        home_targets = np.asarray(
            self.config.home_joint_positions
            if home_joint_positions_override is None
            else home_joint_positions_override,
            dtype=float,
        )
        if home_targets.shape != (len(MEASURED_JOINT_NAMES),):
            raise ValueError(
                "home_joint_positions_override must contain exactly "
                f"{len(MEASURED_JOINT_NAMES)} values."
            )
        if not np.all(np.isfinite(home_targets)):
            raise ValueError(
                "home_joint_positions_override must contain only finite values."
            )

        self._required_latest(
            self.joint_cache,
            label="joint feedback",
        )
        self._required_latest(
            self.simulation_clock_cache,
            label="simulation clock",
        )
        self.schedule.reset()

        paused = False

        try:
            self.world_controller.pause()
            paused = True

            # Drain queued clock callbacks from the continuous-running
            # period before establishing the deterministic reset baseline.
            self._wait_for_paused_clock_barrier()

            (
                _zero_joint_state,
                _zero_end_clock,
                zero_control_steps,
                zero_physics_steps,
                _zero_stable_samples,
                _zero_position_error,
                _zero_velocity,
                clock_rebase_used,
            ) = self._move_to_targets(
                zero_targets,
                phase_label="ZERO",
                clock_rebase_used=False,
            )

            (
                home_joint_state,
                home_end_clock,
                home_control_steps,
                home_physics_steps,
                home_stable_samples,
                home_position_error,
                home_velocity,
                _clock_rebase_used,
            ) = self._move_to_targets(
                home_targets,
                phase_label="HOME",
                clock_rebase_used=clock_rebase_used,
            )

            # Drain any final delayed /clock callbacks before handing the
            # paused world to the policy coordinator.
            final_clock = (
                self._wait_for_paused_clock_barrier()
            )
            final_clock_advance_s = (
                final_clock.sim_time_s
                - home_end_clock.sim_time_s
            )

            if (
                final_clock_advance_s
                < -self.config.max_sim_step_error_s
                or final_clock_advance_s
                > (
                    self.schedule.physics_step_s
                    + self.config.max_sim_step_error_s
                )
            ):
                raise EpisodeResetError(
                    "Gazebo simulation time changed unexpectedly "
                    "while establishing the final HOME reset "
                    "clock baseline: "
                    f"step_end={home_end_clock.sim_time_s:.9f} s, "
                    f"final={final_clock.sim_time_s:.9f} s, "
                    f"advance={final_clock_advance_s:.9f} s."
                )

            # Frame-0 owns a fresh 33/33/34 schedule regardless of how many
            # reset-only steps ZERO and HOME required.
            self.schedule.reset()

            return EpisodeResetResult(
                joint_state=home_joint_state,
                clock_state=final_clock,
                control_steps=(
                    zero_control_steps
                    + home_control_steps
                ),
                physics_steps=(
                    zero_physics_steps
                    + home_physics_steps
                ),
                stable_sample_count=home_stable_samples,
                max_position_error_rad=(
                    home_position_error
                ),
                max_velocity_rad_s=home_velocity,
            )

        except Exception as error:
            self.schedule.reset()

            if paused:
                try:
                    self.world_controller.resume()
                except Exception as resume_error:
                    raise EpisodeResetError(
                        "Episode reset failed and Gazebo could not "
                        "be resumed."
                    ) from resume_error

            if isinstance(error, EpisodeResetError):
                raise

            raise EpisodeResetError(
                "Unexpected error during episode ZERO-then-HOME reset."
            ) from error
