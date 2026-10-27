"""Stateful pick-and-place reward for the SO-101 Gazebo task.

One reward step is one 30 Hz policy/environment step. It is not a Gazebo
physics step. All calculations use measured joint feedback, measured target
pose, and forward-kinematics output.

Every tunable reward value is collected in :class:`ReachRewardConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Mapping, Sequence

import numpy as np


_DEG_PER_RAD = 180.0 / math.pi
_DEFAULT_HOME_JOINT_POSITIONS = (
    0.0,
    -1.710422666954443,
    1.605702911834783,
    1.134464013796314,
    0.0,
    -0.087266462599716,
)


class RewardState(IntEnum):
    """Four mutually exclusive task states.

    State transitions:
    - State 1 -> State 2 when a target appears or switches.
    - State 2 -> State 3 when the compare point enters the target radius.
    - State 2 -> State 4 when the target is already in the goal region.
    - State 3 -> State 2 when the compare point leaves the target radius.
    - State 3 -> State 4 when the target enters the goal region.
    - State 4 -> State 2 when the target leaves the goal region.
    - Any state -> State 1 when the target disappears.

    Legacy enum aliases are retained so existing callers using the old
    outside-goal / inside-goal names continue to import successfully.
    """

    TARGET_ABSENT = 1
    TARGET_APPROACH = 2
    TARGET_GRASP_TRANSPORT = 3
    TARGET_RELEASE_RETURN_HOME = 4

    # Backward-compatible names from the former three-state machine.
    TARGET_OUTSIDE_GOAL = TARGET_APPROACH
    TARGET_INSIDE_GOAL = TARGET_RELEASE_RETURN_HOME


@dataclass(frozen=True)
class ReachRewardConfig:
    """All configurable geometry, thresholds, and reward values.

    One reward step is one 30 Hz policy/environment step.

    Joint positions supplied to the reward are measured in radians. Parameters
    ending in ``_deg`` or ``_per_deg`` use degrees. Distance parameters use
    metres.

    Reward-state summary:
    - State 1: no target; return to / hold HOME.
    - State 2: approach and align with the target before compare-point contact.
    - State 3: maintain contact, close the gripper, lift, and transport.
    - State 4: target is inside the goal region; return to / hold HOME.
    """

    # ======================================================================
    # Shared task geometry / tolerances
    # ======================================================================

    # Per-joint motion deadbands retained for reward/config compatibility.
    joint_motion_tolerance_deg: float = 1.0
    gripper_motion_tolerance_deg: float = 1.0

    # Success requires every one of the six joints, including the gripper,
    # to be within this absolute HOME tolerance.
    home_joint_tolerance_deg: float = 1.0

    # State2 -> State3 when the 3D compare-point/target-centre distance is
    # <= target_contact_radius_m.
    target_contact_radius_m: float = 0.02

    # Target is considered lifted once:
    # target_z >= target_resting_center_z_m + target_lift_height_m.
    target_lift_height_m: float = 0.02
    target_resting_center_z_m: float = 0.765

    # Goal membership currently uses target-centre X/Y only.
    goal_region_half_extent_x_m: float = 0.03
    goal_region_half_extent_y_m: float = 0.03

    # Local axis used by the State2 ground-plane alignment reward.
    alignment_axis_local_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # Sign conventions used for gripper opening/closing progress.
    gripper_open_direction: float = 1.0
    gripper_close_direction: float = -1.0

    numeric_epsilon: float = 1e-12

    # ======================================================================
    # State 1: target absent -> return to / hold HOME
    # ======================================================================
    #
    # For each of the six joints:
    #   effective_error_i =
    #       max(abs(q_i - HOME_i)_deg - state1_home_tolerance_deg, 0)
    #
    # Let e = mean(effective_error_i).
    #
    #   e <= state1_home_error_scale_deg:
    #       reward = +state1_stationary_reward
    #                * (1 - (e / state1_home_error_scale_deg)^2)
    #
    #   state1_home_error_scale_deg < e
    #       <= state1_home_negative_start_deg:
    #       reward = 0
    #
    #   e > state1_home_negative_start_deg:
    #       reward = -state1_stationary_reward
    #                * ((e - state1_home_negative_start_deg)
    #                   / state1_home_error_scale_deg)^2
    #
    # State4 currently reuses this same HOME shaping curve.
    state1_stationary_reward: float = 1
    state1_home_tolerance_deg: float = 1.0
    state1_home_error_scale_deg: float = 2.0
    state1_home_negative_start_deg: float = 5.0

    # Legacy compatibility only; no longer used by the current State1 logic.
    state1_motion_penalty_per_deg: float = 0.1

    # ======================================================================
    # State 2: approach / alignment before contact
    # ======================================================================

    # ----------------------------------------------------------------------
    # State2 3D distance shaping
    # ----------------------------------------------------------------------

    # Applied on EVERY State2 step, including the first State2 reward step:
    #   distance_penalty = -distance_penalty_weight * distance_m^2
    distance_penalty_weight: float = 1.0

    # During the FIRST State2 phase only, reward each new historical best
    # compare-point distance:
    #   improvement = previous_best_distance - new_best_distance
    #   reward = distance_improvement_weight * improvement^2
    distance_improvement_weight: float = 100.0

    # On consecutive State2 -> State2 steps only, penalize moving farther away:
    #   increase = current_distance - previous_distance
    #   penalty = -state2_distance_increase_penalty_per_m * increase
    # The component is zero when distance does not increase.
    state2_distance_increase_penalty_per_m: float = 1000.0

    # Legacy compatibility only. The early_approach reward/event is disabled
    # and these two fields do not affect current reward output.
    early_approach_window_steps: int = 30
    early_approach_bonus: float = 1.0

    # ----------------------------------------------------------------------
    # State2 local +Z -> target alignment
    # ----------------------------------------------------------------------
    #
    # G = grasp/compare reference point.
    # local +Z points out of the gripper mouth.
    # B = G - reference_offset * local_Z_world.
    # theta = angle(local_Z_world, target_center - B).
    #
    # theta > threshold:
    #   fixed negative reward every State2 step.
    #
    # first theta <= threshold:
    #   one +success_bonus per target.
    state2_local_z_target_reference_offset_m: float = 0.04
    state2_local_z_target_threshold_deg: float = 3.0
    state2_local_z_target_misaligned_penalty_per_step: float = 0.1
    state2_local_z_target_success_bonus: float = 5.0

    # ----------------------------------------------------------------------
    # State2 wrist-roll penalty
    # ----------------------------------------------------------------------
    #
    # Every State2 step, penalize counterclockwise wrist-roll displacement
    # from HOME:
    #   penalty = -per_deg * counterclockwise_degrees
    state2_counterclockwise_wrist_roll_penalty_per_deg: float = 1.0

    # ----------------------------------------------------------------------
    # State2 gripper opening / 45-degree target
    # ----------------------------------------------------------------------

    # During the first State2 phase and only while target_age_steps <= this
    # window, reward NEW maximum opening relative to HOME. The accumulated
    # reward is capped per target.
    gripper_opening_window_steps: int = 90
    gripper_opening_reward_per_deg: float = 0.1
    gripper_opening_reward_cap: float = 4.5

    # Every State2 step, penalize absolute opening error from the 45-degree
    # target:
    #   penalty = -gripper_target_opening_penalty_per_deg
    #             * abs(opening_deg - gripper_target_opening_deg)
    gripper_target_opening_deg: float = 45.0
    gripper_target_opening_penalty_per_deg: float = 1.0

    # Legacy compatibility only; no longer used by current State2 logic.
    gripper_still_reward: float = 0.01
    gripper_motion_penalty: float = 0.01

    # ----------------------------------------------------------------------
    # State2 local +X ground-plane alignment
    # ----------------------------------------------------------------------
    #
    # Progress/success are active only during the FIRST State2 phase.
    #
    # Every new alignment improvement bucket:
    #   +alignment_progress_reward_per_bucket
    #
    # First entry into <= alignment_success_threshold_deg:
    #   +alignment_success_bonus once.
    #
    # After success, leaving beyond alignment_exit_threshold_deg:
    #   -alignment_exit_penalty each qualifying exit.
    alignment_progress_bucket_deg: float = 1.0
    alignment_progress_reward_per_bucket: float = 0.4
    alignment_success_threshold_deg: float = 3.0
    alignment_success_bonus: float = 5.0
    alignment_exit_threshold_deg: float = 3.0
    alignment_exit_penalty: float = 2.0

    # ----------------------------------------------------------------------
    # State2 <-> State3 contact transition
    # ----------------------------------------------------------------------

    # First State2 -> State3 contact entry per target.
    target_contact_bonus: float = 5.0

    # Every State3 -> State2 contact loss.
    target_contact_exit_penalty: float = 10.0

    # ======================================================================
    # State 3: grasp / hold / lift / transport
    # ======================================================================

    # Every State3 step, including the State2 -> State3 transition frame.
    state3_hold_reward: float = 0.1

    # Reward NEW maximum closing progress from the gripper angle recorded on
    # first contact. Total closing reward is capped across the episode.
    gripper_closing_reward_per_deg: float = 0.1
    gripper_closing_reward_cap: float = 4.0

    # First valid lift per target:
    #   +target_lift_bonus
    #
    # After a valid lift, first drop below the lift threshold while not in
    # the goal region:
    #   -target_drop_penalty
    target_lift_bonus: float = 2.0
    target_drop_penalty: float = 2.0

    # ======================================================================
    # State3 -> State4 / State4 -> State2 goal transitions
    # ======================================================================

    # First State3 -> State4 goal entry per target.
    goal_entry_bonus: float = 10.0

    # Every State4 -> State2 goal exit.
    goal_exit_penalty: float = 11.0

    # ======================================================================
    # State 4: target in goal -> return to / hold HOME
    # ======================================================================
    #
    # Current State4 reward uses the SAME HOME shaping curve as State1,
    # including state1_stationary_reward, state1_home_tolerance_deg,
    # state1_home_error_scale_deg, and state1_home_negative_start_deg.
    #
    # The following three State4 fields are retained only for compatibility;
    # the old gripper-opening reward and fixed not-HOME penalty are disabled.
    state4_gripper_open_reward_per_deg: float = 0.2
    state4_gripper_open_reward_cap: float = 3.0
    state4_not_home_penalty: float = 0.1

    # Success requires State4 + strict HOME for this many consecutive 30 Hz
    # reward steps, then adds success_bonus.
    success_home_hold_steps: int = 90
    success_bonus: float = 50.0

    # HOME used by State1 shaping, State4 shaping, success checking, and
    # State2 wrist/gripper reference calculations.
    home_joint_positions: tuple[float, ...] = (
        _DEFAULT_HOME_JOINT_POSITIONS
    )

    def __post_init__(self) -> None:
        finite_nonnegative = {
            "joint_motion_tolerance_deg": self.joint_motion_tolerance_deg,
            "gripper_motion_tolerance_deg": (
                self.gripper_motion_tolerance_deg
            ),
            "home_joint_tolerance_deg": self.home_joint_tolerance_deg,
            "target_contact_radius_m": self.target_contact_radius_m,
            "target_lift_height_m": self.target_lift_height_m,
            "target_resting_center_z_m": self.target_resting_center_z_m,
            "goal_region_half_extent_x_m": (
                self.goal_region_half_extent_x_m
            ),
            "goal_region_half_extent_y_m": (
                self.goal_region_half_extent_y_m
            ),
            "numeric_epsilon": self.numeric_epsilon,
            "state1_stationary_reward": self.state1_stationary_reward,
            "state1_motion_penalty_per_deg": (
                self.state1_motion_penalty_per_deg
            ),
            "state1_home_tolerance_deg": (
                self.state1_home_tolerance_deg
            ),
            "state1_home_error_scale_deg": (
                self.state1_home_error_scale_deg
            ),
            "state1_home_negative_start_deg": (
                self.state1_home_negative_start_deg
            ),
            "early_approach_bonus": self.early_approach_bonus,
            "distance_penalty_weight": self.distance_penalty_weight,
            "distance_improvement_weight": (
                self.distance_improvement_weight
            ),
            "state2_distance_increase_penalty_per_m": (
                self.state2_distance_increase_penalty_per_m
            ),
            "state2_local_z_target_reference_offset_m": (
                self.state2_local_z_target_reference_offset_m
            ),
            "state2_local_z_target_threshold_deg": (
                self.state2_local_z_target_threshold_deg
            ),
            "state2_local_z_target_misaligned_penalty_per_step": (
                self.state2_local_z_target_misaligned_penalty_per_step
            ),
            "state2_local_z_target_success_bonus": (
                self.state2_local_z_target_success_bonus
            ),
            "state2_counterclockwise_wrist_roll_penalty_per_deg": (
                self.state2_counterclockwise_wrist_roll_penalty_per_deg
            ),
            "gripper_opening_reward_per_deg": (
                self.gripper_opening_reward_per_deg
            ),
            "gripper_opening_reward_cap": (
                self.gripper_opening_reward_cap
            ),
            "gripper_still_reward": self.gripper_still_reward,
            "gripper_motion_penalty": self.gripper_motion_penalty,
            "gripper_target_opening_deg": (
                self.gripper_target_opening_deg
            ),
            "gripper_target_opening_penalty_per_deg": (
                self.gripper_target_opening_penalty_per_deg
            ),
            "alignment_progress_bucket_deg": (
                self.alignment_progress_bucket_deg
            ),
            "alignment_progress_reward_per_bucket": (
                self.alignment_progress_reward_per_bucket
            ),
            "alignment_success_threshold_deg": (
                self.alignment_success_threshold_deg
            ),
            "alignment_success_bonus": self.alignment_success_bonus,
            "alignment_exit_threshold_deg": (
                self.alignment_exit_threshold_deg
            ),
            "alignment_exit_penalty": self.alignment_exit_penalty,
            "target_contact_bonus": self.target_contact_bonus,
            "target_contact_exit_penalty": (
                self.target_contact_exit_penalty
            ),
            "state3_hold_reward": self.state3_hold_reward,
            "gripper_closing_reward_per_deg": (
                self.gripper_closing_reward_per_deg
            ),
            "gripper_closing_reward_cap": (
                self.gripper_closing_reward_cap
            ),
            "target_lift_bonus": self.target_lift_bonus,
            "target_drop_penalty": self.target_drop_penalty,
            "goal_entry_bonus": self.goal_entry_bonus,
            "goal_exit_penalty": self.goal_exit_penalty,
            "state4_gripper_open_reward_per_deg": (
                self.state4_gripper_open_reward_per_deg
            ),
            "state4_gripper_open_reward_cap": (
                self.state4_gripper_open_reward_cap
            ),
            "state4_not_home_penalty": (
                self.state4_not_home_penalty
            ),
            "success_bonus": self.success_bonus,
        }

        for name, raw_value in finite_nonnegative.items():
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )

        positive_values = {
            "state1_home_error_scale_deg": (
                self.state1_home_error_scale_deg
            ),
            "state1_home_negative_start_deg": (
                self.state1_home_negative_start_deg
            ),
            "target_contact_radius_m": self.target_contact_radius_m,
            "state2_local_z_target_reference_offset_m": (
                self.state2_local_z_target_reference_offset_m
            ),
            "state2_local_z_target_threshold_deg": (
                self.state2_local_z_target_threshold_deg
            ),
            "target_lift_height_m": self.target_lift_height_m,
            "goal_region_half_extent_x_m": (
                self.goal_region_half_extent_x_m
            ),
            "goal_region_half_extent_y_m": (
                self.goal_region_half_extent_y_m
            ),
            "alignment_progress_bucket_deg": (
                self.alignment_progress_bucket_deg
            ),
            "alignment_success_threshold_deg": (
                self.alignment_success_threshold_deg
            ),
            "alignment_exit_threshold_deg": (
                self.alignment_exit_threshold_deg
            ),
        }
        for name, value in positive_values.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")

        if (
            self.state1_home_negative_start_deg
            < self.state1_home_error_scale_deg
        ):
            raise ValueError(
                "state1_home_negative_start_deg must be greater than or "
                "equal to state1_home_error_scale_deg."
            )

        if (
            self.alignment_exit_threshold_deg
            < self.alignment_success_threshold_deg
        ):
            raise ValueError(
                "alignment_exit_threshold_deg must be greater than or "
                "equal to alignment_success_threshold_deg."
            )

        integer_values = {
            "early_approach_window_steps": (
                self.early_approach_window_steps
            ),
            "gripper_opening_window_steps": (
                self.gripper_opening_window_steps
            ),
            "success_home_hold_steps": (
                self.success_home_hold_steps
            ),
        }
        for name, value in integer_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        for name, value in {
            "gripper_open_direction": self.gripper_open_direction,
            "gripper_close_direction": self.gripper_close_direction,
        }.items():
            direction = float(value)
            if not math.isfinite(direction) or abs(direction) != 1.0:
                raise ValueError(f"{name} must be either -1.0 or 1.0.")

        axis = _validated_vector(
            self.alignment_axis_local_xyz,
            name="alignment_axis_local_xyz",
        )
        if np.linalg.norm(axis) <= self.numeric_epsilon:
            raise ValueError(
                "alignment_axis_local_xyz must be non-zero."
            )

        home = _validated_joint_positions(
            self.home_joint_positions,
            name="home_joint_positions",
        )
        if home.shape != (6,):
            raise ValueError(
                "home_joint_positions must contain exactly six values."
            )


@dataclass(frozen=True)
class ReachRewardInput:
    """Measured inputs for one policy/environment step."""

    joint_positions_rad: Sequence[float]
    grasp_position_world_m: Sequence[float]
    grasp_rotation_world: Sequence[Sequence[float]]
    goal_region_center_world_m: Sequence[float]
    target_name: str | None = None
    target_position_world_m: Sequence[float] | None = None


@dataclass(frozen=True)
class ReachRewardResult:
    """Reward value, state, events, and inspectable components."""

    reward: float
    state: RewardState
    success: bool
    distance_m: float | None
    distance_improvement_m: float | None
    alignment_error_deg: float | None
    target_age_steps: int
    robot_at_home: bool
    target_in_goal_region: bool
    success_home_hold_count: int
    components: dict[str, float]
    events: tuple[str, ...]


_COMPONENT_NAMES = (
    "state1_stationary",
    "state1_motion",
    "state1_home_error",
    "early_approach",
    "distance_penalty",
    "distance_improvement",
    "distance_increase",
    "state2_local_z_target_misaligned",
    "state2_local_z_target_success",
    "state2_counterclockwise_wrist_roll",
    "gripper_opening",
    "gripper_stability",
    "alignment_progress",
    "alignment_success",
    "alignment_exit",
    "target_contact",
    "target_contact_exit",
    "state3_hold",
    "gripper_closing",
    "target_lift",
    "target_drop",
    "goal_entry",
    "goal_exit",
    "state4_gripper_opening",
    "state4_not_home",
    "success",
)


def _validated_joint_positions(
    value: Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validated_vector(
    value: Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validated_rotation(
    value: Sequence[Sequence[float]],
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3):
        raise ValueError(
            "grasp_rotation_world must have shape (3, 3)."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "grasp_rotation_world must contain only finite values."
        )
    return result


def _empty_components() -> dict[str, float]:
    return {name: 0.0 for name in _COMPONENT_NAMES}


def _add_component(
    components: dict[str, float],
    name: str,
    value: float,
) -> None:
    components[name] += float(value)


class StatefulReachReward:
    """Stateful four-state reward machine for one episode."""

    def __init__(
        self,
        config: ReachRewardConfig | None = None,
    ) -> None:
        self.config = (
            ReachRewardConfig()
            if config is None
            else config
        )
        self._home = np.asarray(
            self.config.home_joint_positions,
            dtype=float,
        )
        axis = np.asarray(
            self.config.alignment_axis_local_xyz,
            dtype=float,
        )
        self._alignment_axis_local = axis / np.linalg.norm(axis)
        self._episode_initialized = False
        self._previous_joint_positions: np.ndarray | None = None
        self._previous_state = RewardState.TARGET_ABSENT
        self._active_target_name: str | None = None
        self._clear_episode_progress()
        self._clear_target_progress()

    @property
    def active_target_name(self) -> str | None:
        return self._active_target_name

    def _clear_episode_progress(self) -> None:
        # Closing reward remains capped across the whole episode.
        self._contact_closing_reward_total = 0.0

    def _clear_target_progress(self) -> None:
        self._target_age_steps = 0
        self._state2_entry_pending = True
        self._first_state2_phase_active = True

        self._initial_distance_m: float | None = None
        self._best_distance_m: float | None = None
        self._previous_distance_m: float | None = None

        # The +5 local-Z alignment bonus is once per target.
        self._state2_local_z_target_success_awarded = False

        self._maximum_home_opening_deg = 0.0
        self._home_opening_reward_total = 0.0

        self._initial_alignment_error_deg: float | None = None
        self._best_alignment_error_deg: float | None = None
        self._alignment_progress_buckets_awarded = 0
        self._alignment_success_awarded = False
        self._alignment_inside_success_band = False

        # Contact bonus is once per target.
        self._contact_bonus_awarded = False
        self._contact_entry_gripper_rad: float | None = None
        self._maximum_contact_closing_deg = 0.0

        self._lift_bonus_awarded = False
        self._lift_cycle_armed = False

        self._goal_entry_awarded = False
        self._state4_entry_gripper_rad: float | None = None
        self._maximum_state4_opening_deg = 0.0
        self._state4_opening_reward_total = 0.0
        self._success_home_hold_count = 0

    def reset(
        self,
        *,
        joint_positions_rad: Sequence[float],
        target_name: str | None = None,
        target_position_world_m: Sequence[float] | None = None,
        grasp_position_world_m: Sequence[float] | None = None,
        grasp_rotation_world: Sequence[Sequence[float]] | None = None,
        goal_region_center_world_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Reset all episode and target-specific reward memory."""

        joints = _validated_joint_positions(
            joint_positions_rad,
            name="joint_positions_rad",
        )
        if joints.shape != self._home.shape:
            raise ValueError(
                "joint_positions_rad must match home_joint_positions."
            )

        self._episode_initialized = True
        self._previous_joint_positions = joints.copy()
        self._previous_state = RewardState.TARGET_ABSENT
        self._active_target_name = None
        self._clear_episode_progress()
        self._clear_target_progress()

        if target_name is not None:
            if (
                target_position_world_m is None
                or grasp_position_world_m is None
                or grasp_rotation_world is None
            ):
                raise ValueError(
                    "Target, grasp position, and grasp rotation are all "
                    "required when reset starts with a target."
                )
            self._activate_target(
                target_name=target_name,
                target_position_world_m=target_position_world_m,
                grasp_position_world_m=grasp_position_world_m,
                grasp_rotation_world=grasp_rotation_world,
                gripper_rad=float(joints[-1]),
            )
            _validated_vector(
                goal_region_center_world_m,
                name="goal_region_center_world_m",
            )

            # reset() establishes the existing target as the State-2
            # frame-zero baseline.  The first formal policy/reward step is
            # therefore allowed to transition immediately to State 3 or
            # State 4 from measured feedback; reset itself consumes no
            # policy step and produces no reward.
            self._previous_state = RewardState.TARGET_APPROACH
            self._state2_entry_pending = False

    def _activate_target(
        self,
        *,
        target_name: str,
        target_position_world_m: Sequence[float],
        grasp_position_world_m: Sequence[float],
        grasp_rotation_world: Sequence[Sequence[float]],
        gripper_rad: float,
    ) -> None:
        normalized_name = str(target_name).strip()
        if not normalized_name:
            raise ValueError("target_name must not be empty.")

        target = _validated_vector(
            target_position_world_m,
            name="target_position_world_m",
        )
        grasp = _validated_vector(
            grasp_position_world_m,
            name="grasp_position_world_m",
        )
        rotation = _validated_rotation(grasp_rotation_world)

        self._active_target_name = normalized_name
        self._clear_target_progress()

        distance = float(np.linalg.norm(target - grasp))
        alignment = self._alignment_error_deg(
            grasp_position=grasp,
            grasp_rotation=rotation,
            target_position=target,
        )
        self._initial_distance_m = distance
        self._best_distance_m = distance
        self._previous_distance_m = distance
        self._initial_alignment_error_deg = alignment
        self._best_alignment_error_deg = alignment

        # The baseline is fixed by the physical table and 30 mm cube geometry.
        _ = gripper_rad

    def _target_in_goal(
        self,
        target_position: np.ndarray,
        goal_center: np.ndarray,
    ) -> bool:
        return bool(
            abs(target_position[0] - goal_center[0])
            <= self.config.goal_region_half_extent_x_m
            and abs(target_position[1] - goal_center[1])
            <= self.config.goal_region_half_extent_y_m
        )

    def _state_from_observation(
        self,
        *,
        force_state2_entry: bool,
        target_in_goal: bool,
        compare_point_in_range: bool,
    ) -> RewardState:
        """Resolve the strict four-state transition contract."""

        if force_state2_entry:
            return RewardState.TARGET_APPROACH

        if (
            self._previous_state
            is RewardState.TARGET_RELEASE_RETURN_HOME
        ):
            if target_in_goal:
                return RewardState.TARGET_RELEASE_RETURN_HOME
            return RewardState.TARGET_APPROACH

        if (
            self._previous_state
            is RewardState.TARGET_GRASP_TRANSPORT
        ):
            if target_in_goal:
                return RewardState.TARGET_RELEASE_RETURN_HOME
            if compare_point_in_range:
                return RewardState.TARGET_GRASP_TRANSPORT
            return RewardState.TARGET_APPROACH

        # State 2 may jump directly to State 4.  This is needed when a
        # target is already inside the goal region when it appears.  A newly
        # activated target still spends its activation step in State 2
        # because force_state2_entry is handled above.
        if target_in_goal:
            return RewardState.TARGET_RELEASE_RETURN_HOME

        if compare_point_in_range:
            return RewardState.TARGET_GRASP_TRANSPORT

        return RewardState.TARGET_APPROACH

    def _alignment_error_deg(
        self,
        *,
        grasp_position: np.ndarray,
        grasp_rotation: np.ndarray,
        target_position: np.ndarray,
    ) -> float:
        """Return compare-axis deviation from the global ground plane.

        ``grasp_rotation`` is the rotation of the existing offset compare
        pose supplied by the environment.  With the default local +X axis,
        an error of 0 degrees means the compare-point X axis is parallel to
        the ground plane; 90 degrees means it is vertical.
        """

        _ = grasp_position
        _ = target_position
        axis_world = grasp_rotation @ self._alignment_axis_local
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm <= self.config.numeric_epsilon:
            raise ValueError(
                "The transformed alignment axis is zero."
            )
        axis_world = axis_world / axis_norm

        # The ground plane normal is global +Z.  Parallel-to-plane error is
        # therefore asin(|axis dot normal|), giving 0 deg when horizontal
        # and 90 deg when vertical.  The absolute value makes +X and -X
        # equivalent for plane parallelism.
        vertical_component = float(
            np.clip(abs(axis_world[2]), 0.0, 1.0)
        )
        return float(math.degrees(math.asin(vertical_component)))

    def _robot_at_home(self, joints: np.ndarray) -> bool:
        errors_deg = np.abs(joints - self._home) * _DEG_PER_RAD
        return bool(
            np.all(
                errors_deg
                <= self.config.home_joint_tolerance_deg
                + self.config.numeric_epsilon
            )
        )

    def _state1_reward(
        self,
        *,
        joints: np.ndarray,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        # Reward proximity to HOME instead of low frame-to-frame motion.
        #
        # Each of the six joints, including the gripper, gets its own
        # state1_home_tolerance_deg deadband. Only error beyond that
        # tolerance contributes to the mean effective HOME error.
        home_error_deg = np.abs(joints - self._home) * _DEG_PER_RAD
        effective_home_error_deg = np.maximum(
            home_error_deg - self.config.state1_home_tolerance_deg,
            0.0,
        )
        mean_effective_home_error_deg = float(
            np.mean(effective_home_error_deg)
        )

        # Piecewise State-1 HOME shaping:
        # - 0 .. home_error_scale: +max_reward -> 0 quadratically.
        # - home_error_scale .. negative_start: 0.
        # - beyond negative_start: negative reward grows quadratically.
        if (
            mean_effective_home_error_deg
            <= self.config.state1_home_error_scale_deg
        ):
            reward = self.config.state1_stationary_reward * (
                1.0
                - (
                    mean_effective_home_error_deg
                    / self.config.state1_home_error_scale_deg
                )
                ** 2
            )
        elif (
            mean_effective_home_error_deg
            <= self.config.state1_home_negative_start_deg
        ):
            reward = 0.0
        else:
            negative_error_deg = (
                mean_effective_home_error_deg
                - self.config.state1_home_negative_start_deg
            )
            reward = -self.config.state1_stationary_reward * (
                negative_error_deg
                / self.config.state1_home_error_scale_deg
            ) ** 2

        if (
            mean_effective_home_error_deg
            <= self.config.numeric_epsilon
        ):
            _add_component(
                components,
                "state1_stationary",
                reward,
            )
            events.append("state1_stationary")
        else:
            _add_component(
                components,
                "state1_home_error",
                reward,
            )
            events.append("state1_home_error")

    def _state2_local_z_target_error_deg(
        self,
        *,
        grasp_position: np.ndarray,
        grasp_rotation: np.ndarray,
        target_position: np.ndarray,
    ) -> float:
        # Return local +Z -> target angle from a point inside the gripper.
        # local +Z points out of the gripper mouth.
        local_z_world = grasp_rotation @ np.asarray(
            (0.0, 0.0, 1.0),
            dtype=float,
        )
        local_z_norm = float(np.linalg.norm(local_z_world))
        if local_z_norm <= self.config.numeric_epsilon:
            raise ValueError(
                "The transformed grasp local +Z axis is zero."
            )
        local_z_world = local_z_world / local_z_norm

        # Move the angular reference point inside along local -Z:
        # B = G - offset * local_Z_world.
        reference_point = (
            grasp_position
            - self.config.state2_local_z_target_reference_offset_m
            * local_z_world
        )

        target_direction = target_position - reference_point
        target_direction_norm = float(np.linalg.norm(target_direction))
        if target_direction_norm <= self.config.numeric_epsilon:
            raise ValueError(
                "State-2 local +Z target direction is zero."
            )
        target_direction = target_direction / target_direction_norm

        cosine = float(
            np.clip(
                np.dot(local_z_world, target_direction),
                -1.0,
                1.0,
            )
        )
        return float(math.degrees(math.acos(cosine)))

    def _reward_state2_local_z_target_alignment(
        self,
        *,
        grasp_position: np.ndarray,
        grasp_rotation: np.ndarray,
        target_position: np.ndarray,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        # Apply local +Z target alignment reward on every State-2 step.
        error_deg = self._state2_local_z_target_error_deg(
            grasp_position=grasp_position,
            grasp_rotation=grasp_rotation,
            target_position=target_position,
        )

        inside = bool(
            error_deg
            <= self.config.state2_local_z_target_threshold_deg
            + self.config.numeric_epsilon
        )

        if inside:
            if not self._state2_local_z_target_success_awarded:
                self._state2_local_z_target_success_awarded = True
                _add_component(
                    components,
                    "state2_local_z_target_success",
                    self.config.state2_local_z_target_success_bonus,
                )
                events.append("state2_local_z_target_success")
            return

        penalty = (
            -self.config
            .state2_local_z_target_misaligned_penalty_per_step
        )
        if penalty < -self.config.numeric_epsilon:
            _add_component(
                components,
                "state2_local_z_target_misaligned",
                penalty,
            )
            events.append("state2_local_z_target_misaligned")

    def _reward_state2_counterclockwise_wrist_roll(
        self,
        *,
        wrist_roll_rad: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        # Penalize every State-2 step spent counterclockwise of HOME.
        home_wrist_roll_rad = float(self._home[4])
        counterclockwise_deg = max(
            0.0,
            (
                float(wrist_roll_rad)
                - home_wrist_roll_rad
            )
            * _DEG_PER_RAD,
        )
        penalty = (
            -self.config
            .state2_counterclockwise_wrist_roll_penalty_per_deg
            * counterclockwise_deg
        )
        if penalty < -self.config.numeric_epsilon:
            _add_component(
                components,
                "state2_counterclockwise_wrist_roll",
                penalty,
            )
            events.append(
                "state2_counterclockwise_wrist_roll"
            )

    def _reward_new_home_opening(
        self,
        *,
        gripper_rad: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        home_gripper = float(self._home[-1])
        opening_deg = max(
            0.0,
            (
                gripper_rad - home_gripper
            )
            * self.config.gripper_open_direction
            * _DEG_PER_RAD,
        )
        new_opening_deg = max(
            0.0,
            opening_deg - self._maximum_home_opening_deg,
        )
        self._maximum_home_opening_deg = max(
            self._maximum_home_opening_deg,
            opening_deg,
        )

        available = max(
            0.0,
            self.config.gripper_opening_reward_cap
            - self._home_opening_reward_total,
        )
        reward = min(
            available,
            (
                new_opening_deg
                * self.config.gripper_opening_reward_per_deg
            ),
        )
        if reward > self.config.numeric_epsilon:
            self._home_opening_reward_total += reward
            _add_component(
                components,
                "gripper_opening",
                reward,
            )
            events.append("gripper_opening_progress")

    def _reward_state2_gripper_target_opening(
        self,
        *,
        gripper_rad: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        """Penalize State-2 gripper opening error from 45 degrees."""

        home_gripper = float(self._home[-1])
        opening_deg = max(
            0.0,
            (
                gripper_rad - home_gripper
            )
            * self.config.gripper_open_direction
            * _DEG_PER_RAD,
        )
        error_deg = abs(
            opening_deg - self.config.gripper_target_opening_deg
        )
        penalty = (
            -self.config.gripper_target_opening_penalty_per_deg
            * error_deg
        )
        if penalty < -self.config.numeric_epsilon:
            _add_component(
                components,
                "gripper_stability",
                penalty,
            )
            events.append("gripper_opening_target_error")

    def _reward_state2_distance_increase(
        self,
        *,
        distance_m: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        # Penalize moving farther away on consecutive State-2 steps.
        if self._previous_distance_m is None:
            return

        increase_m = (
            float(distance_m) - float(self._previous_distance_m)
        )
        if increase_m <= self.config.numeric_epsilon:
            return

        penalty = (
            -self.config.state2_distance_increase_penalty_per_m
            * increase_m
        )
        _add_component(
            components,
            "distance_increase",
            penalty,
        )
        events.append("distance_increase")

    def _reward_distance_progress(
        self,
        *,
        distance_m: float,
        components: dict[str, float],
        events: list[str],
    ) -> float:
        assert self._initial_distance_m is not None
        assert self._best_distance_m is not None

        # early_approach was intentionally removed. Record-based
        # distance improvement remains active during the first State2 phase.
        improvement = 0.0
        if (
            distance_m
            < self._best_distance_m
            - self.config.numeric_epsilon
        ):
            improvement = self._best_distance_m - distance_m
            self._best_distance_m = distance_m
            reward = (
                self.config.distance_improvement_weight
                * improvement
                * improvement
            )
            _add_component(
                components,
                "distance_improvement",
                reward,
            )
            events.append("distance_record")

        return float(improvement)

    def _reward_first_state2_alignment(
        self,
        *,
        alignment_error_deg: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        """Reward alignment progress/success only in first State 2."""

        assert self._initial_alignment_error_deg is not None
        assert self._best_alignment_error_deg is not None

        if (
            alignment_error_deg
            < self._best_alignment_error_deg
            - self.config.numeric_epsilon
        ):
            self._best_alignment_error_deg = alignment_error_deg

        total_improvement_deg = max(
            0.0,
            self._initial_alignment_error_deg
            - self._best_alignment_error_deg,
        )
        eligible_buckets = int(
            math.floor(
                (
                    total_improvement_deg
                    + self.config.numeric_epsilon
                )
                / self.config.alignment_progress_bucket_deg
            )
        )
        new_buckets = max(
            0,
            eligible_buckets
            - self._alignment_progress_buckets_awarded,
        )
        if new_buckets:
            self._alignment_progress_buckets_awarded += new_buckets
            reward = (
                new_buckets
                * self.config.alignment_progress_reward_per_bucket
            )
            _add_component(
                components,
                "alignment_progress",
                reward,
            )
            events.append("alignment_progress")

        inside_success_band = bool(
            alignment_error_deg
            <= self.config.alignment_success_threshold_deg
            + self.config.numeric_epsilon
        )
        if not self._alignment_success_awarded and inside_success_band:
            self._alignment_success_awarded = True
            self._alignment_inside_success_band = True
            _add_component(
                components,
                "alignment_success",
                self.config.alignment_success_bonus,
            )
            events.append("alignment_success")

    def _reward_state2_alignment_exit(
        self,
        *,
        alignment_error_deg: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        """Apply alignment-exit tracking on every State-2 step."""

        if not self._alignment_success_awarded:
            return

        inside_success_band = bool(
            alignment_error_deg
            <= self.config.alignment_success_threshold_deg
            + self.config.numeric_epsilon
        )
        if inside_success_band:
            self._alignment_inside_success_band = True
            return

        if (
            self._alignment_inside_success_band
            and alignment_error_deg
            > self.config.alignment_exit_threshold_deg
            + self.config.numeric_epsilon
        ):
            self._alignment_inside_success_band = False
            _add_component(
                components,
                "alignment_exit",
                -self.config.alignment_exit_penalty,
            )
            events.append("alignment_exit")

    def _reward_state2_contact_entry(
        self,
        *,
        gripper_rad: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        """Reward State-2 -> State-3 contact once per target."""

        if self._contact_bonus_awarded:
            return

        self._contact_bonus_awarded = True
        self._contact_entry_gripper_rad = gripper_rad
        _add_component(
            components,
            "target_contact",
            self.config.target_contact_bonus,
        )
        events.append("target_contact")

    def _reward_state3_contact_exit(
        self,
        *,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        """Penalize every State-3 -> State-2 contact loss."""

        _add_component(
            components,
            "target_contact_exit",
            -self.config.target_contact_exit_penalty,
        )
        events.append("target_contact_exit")

    def _reward_state3_closing(
        self,
        *,
        gripper_rad: float,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        """Reward new gripper closing progress while State 3 is active."""

        if self._contact_entry_gripper_rad is None:
            return

        closing_deg = max(
            0.0,
            (
                gripper_rad
                - self._contact_entry_gripper_rad
            )
            * self.config.gripper_close_direction
            * _DEG_PER_RAD,
        )
        new_closing_deg = max(
            0.0,
            closing_deg - self._maximum_contact_closing_deg,
        )
        self._maximum_contact_closing_deg = max(
            self._maximum_contact_closing_deg,
            closing_deg,
        )
        available = max(
            0.0,
            self.config.gripper_closing_reward_cap
            - self._contact_closing_reward_total,
        )
        reward = min(
            available,
            (
                new_closing_deg
                * self.config.gripper_closing_reward_per_deg
            ),
        )
        if reward > self.config.numeric_epsilon:
            self._contact_closing_reward_total += reward
            _add_component(
                components,
                "gripper_closing",
                reward,
            )
            events.append("gripper_closing_progress")

    def _reward_lift_and_drop(
        self,
        *,
        target_z_m: float,
        target_in_goal: bool,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        lift_threshold = (
            self.config.target_resting_center_z_m
            + self.config.target_lift_height_m
        )
        lifted = bool(
            target_z_m
            >= lift_threshold
            - self.config.numeric_epsilon
        )

        if lifted:
            self._lift_cycle_armed = True
            if not self._lift_bonus_awarded:
                self._lift_bonus_awarded = True
                _add_component(
                    components,
                    "target_lift",
                    self.config.target_lift_bonus,
                )
                events.append("target_lift")

        elif self._lift_cycle_armed and not target_in_goal:
            self._lift_cycle_armed = False
            _add_component(
                components,
                "target_drop",
                -self.config.target_drop_penalty,
            )
            events.append("target_drop")

    def _reward_state4_home(
        self,
        *,
        joints: np.ndarray,
        components: dict[str, float],
        events: list[str],
    ) -> None:
        # State4 uses the same HOME shaping as State1.
        #
        # Each of the six joints, including the gripper, gets the same
        # state1_home_tolerance_deg deadband. Only error beyond that
        # tolerance contributes to the mean effective HOME error.
        home_error_deg = np.abs(joints - self._home) * _DEG_PER_RAD
        effective_home_error_deg = np.maximum(
            home_error_deg - self.config.state1_home_tolerance_deg,
            0.0,
        )
        mean_effective_home_error_deg = float(
            np.mean(effective_home_error_deg)
        )

        if (
            mean_effective_home_error_deg
            <= self.config.state1_home_error_scale_deg
        ):
            reward = self.config.state1_stationary_reward * (
                1.0
                - (
                    mean_effective_home_error_deg
                    / self.config.state1_home_error_scale_deg
                )
                ** 2
            )
        elif (
            mean_effective_home_error_deg
            <= self.config.state1_home_negative_start_deg
        ):
            reward = 0.0
        else:
            negative_error_deg = (
                mean_effective_home_error_deg
                - self.config.state1_home_negative_start_deg
            )
            reward = -self.config.state1_stationary_reward * (
                negative_error_deg
                / self.config.state1_home_error_scale_deg
            ) ** 2

        _add_component(
            components,
            "state4_not_home",
            reward,
        )
        events.append("state4_home_reward")

    def step(
        self,
        observation: ReachRewardInput,
    ) -> ReachRewardResult:
        """Compute reward for one measured 30 Hz policy step."""

        if not self._episode_initialized:
            raise RuntimeError(
                "StatefulReachReward.reset() must be called before step()."
            )

        joints = _validated_joint_positions(
            observation.joint_positions_rad,
            name="joint_positions_rad",
        )
        if joints.shape != self._home.shape:
            raise ValueError(
                "joint_positions_rad must match home_joint_positions."
            )
        grasp = _validated_vector(
            observation.grasp_position_world_m,
            name="grasp_position_world_m",
        )
        rotation = _validated_rotation(
            observation.grasp_rotation_world
        )
        goal = _validated_vector(
            observation.goal_region_center_world_m,
            name="goal_region_center_world_m",
        )

        target_name = (
            None
            if observation.target_name is None
            else str(observation.target_name).strip()
        )
        target_position = None
        if target_name is not None:
            if not target_name:
                raise ValueError("target_name must not be empty.")
            if observation.target_position_world_m is None:
                raise ValueError(
                    "target_position_world_m is required when a target "
                    "name is present."
                )
            target_position = _validated_vector(
                observation.target_position_world_m,
                name="target_position_world_m",
            )
        elif observation.target_position_world_m is not None:
            raise ValueError(
                "target_name is required when target_position_world_m "
                "is present."
            )

        components = _empty_components()
        events: list[str] = []
        robot_at_home = self._robot_at_home(joints)
        distance_m: float | None = None
        distance_improvement_m: float | None = None
        alignment_error_deg: float | None = None
        target_in_goal = False

        if target_name is None:
            current_state = RewardState.TARGET_ABSENT
            if self._active_target_name is not None:
                self._active_target_name = None
                self._clear_target_progress()
                events.append("target_removed")
            self._state1_reward(
                joints=joints,
                components=components,
                events=events,
            )
        else:
            assert target_position is not None
            target_changed = target_name != self._active_target_name
            if target_changed:
                switched = self._active_target_name is not None
                self._activate_target(
                    target_name=target_name,
                    target_position_world_m=target_position,
                    grasp_position_world_m=grasp,
                    grasp_rotation_world=rotation,
                    gripper_rad=float(joints[-1]),
                )
                events.append(
                    "target_switched"
                    if switched
                    else "target_activated"
                )

            self._target_age_steps += 1
            target_in_goal = self._target_in_goal(
                target_position,
                goal,
            )
            distance_m = float(np.linalg.norm(target_position - grasp))
            compare_point_in_range = bool(
                distance_m
                <= self.config.target_contact_radius_m
                + self.config.numeric_epsilon
            )
            force_state2_entry = bool(
                target_changed or self._state2_entry_pending
            )
            current_state = self._state_from_observation(
                force_state2_entry=force_state2_entry,
                target_in_goal=target_in_goal,
                compare_point_in_range=compare_point_in_range,
            )
            self._state2_entry_pending = False

            distance_improvement_m = 0.0
            alignment_error_deg = self._alignment_error_deg(
                grasp_position=grasp,
                grasp_rotation=rotation,
                target_position=target_position,
            )

            state2_to_state3 = bool(
                not target_changed
                and self._previous_state is RewardState.TARGET_APPROACH
                and current_state is RewardState.TARGET_GRASP_TRANSPORT
            )
            state3_to_state2 = bool(
                not target_changed
                and self._previous_state
                is RewardState.TARGET_GRASP_TRANSPORT
                and current_state is RewardState.TARGET_APPROACH
            )
            state2_to_state4 = bool(
                not target_changed
                and self._previous_state is RewardState.TARGET_APPROACH
                and current_state
                is RewardState.TARGET_RELEASE_RETURN_HOME
            )
            state3_to_state4 = bool(
                not target_changed
                and self._previous_state
                is RewardState.TARGET_GRASP_TRANSPORT
                and current_state
                is RewardState.TARGET_RELEASE_RETURN_HOME
            )
            state4_to_state2 = bool(
                not target_changed
                and self._previous_state
                is RewardState.TARGET_RELEASE_RETURN_HOME
                and current_state is RewardState.TARGET_APPROACH
            )

            # State-2 terminal contact reward: once per target.
            if state2_to_state3:
                self._reward_state2_contact_entry(
                    gripper_rad=float(joints[-1]),
                    components=components,
                    events=events,
                )

            # State-3 terminal contact-loss penalty.
            if state3_to_state2:
                self._reward_state3_contact_exit(
                    components=components,
                    events=events,
                )

            # State-3 terminal goal-entry reward: once per target.
            if state3_to_state4:
                if not self._goal_entry_awarded:
                    self._goal_entry_awarded = True
                    _add_component(
                        components,
                        "goal_entry",
                        self.config.goal_entry_bonus,
                    )
                    events.append("goal_entry")
                if self._state4_entry_gripper_rad is None:
                    self._state4_entry_gripper_rad = float(joints[-1])

            # State-4 goal-exit penalty may fire repeatedly.
            if state4_to_state2:
                _add_component(
                    components,
                    "goal_exit",
                    -self.config.goal_exit_penalty,
                )
                events.append("goal_exit")

            # Lift/drop belongs to State 3, including its exit frames.
            if (
                current_state is RewardState.TARGET_GRASP_TRANSPORT
                or self._previous_state
                is RewardState.TARGET_GRASP_TRANSPORT
            ):
                self._reward_lift_and_drop(
                    target_z_m=float(target_position[2]),
                    target_in_goal=target_in_goal,
                    components=components,
                    events=events,
                )

            if current_state is RewardState.TARGET_APPROACH:
                self._reward_state2_local_z_target_alignment(
                    grasp_position=grasp,
                    grasp_rotation=rotation,
                    target_position=target_position,
                    components=components,
                    events=events,
                )

                if (
                    not target_changed
                    and self._previous_state
                    is RewardState.TARGET_APPROACH
                ):
                    self._reward_state2_distance_increase(
                        distance_m=distance_m,
                        components=components,
                        events=events,
                    )

                self._reward_state2_counterclockwise_wrist_roll(
                    wrist_roll_rad=float(joints[4]),
                    components=components,
                    events=events,
                )

                # Distance penalty is active from the first State2
                # reward step; there is no longer a 30-step grace window.
                _add_component(
                    components,
                    "distance_penalty",
                    (
                        -self.config.distance_penalty_weight
                        * distance_m
                        * distance_m
                    ),
                )

                self._reward_state2_gripper_target_opening(
                    gripper_rad=float(joints[-1]),
                    components=components,
                    events=events,
                )

                if self._first_state2_phase_active:
                    distance_improvement_m = (
                        self._reward_distance_progress(
                            distance_m=distance_m,
                            components=components,
                            events=events,
                        )
                    )

                    if (
                        self._target_age_steps
                        <= self.config.gripper_opening_window_steps
                    ):
                        self._reward_new_home_opening(
                            gripper_rad=float(joints[-1]),
                            components=components,
                            events=events,
                        )

                    self._reward_first_state2_alignment(
                        alignment_error_deg=alignment_error_deg,
                        components=components,
                        events=events,
                    )

                self._reward_state2_alignment_exit(
                    alignment_error_deg=alignment_error_deg,
                    components=components,
                    events=events,
                )

            elif current_state is RewardState.TARGET_GRASP_TRANSPORT:
                _add_component(
                    components,
                    "state3_hold",
                    self.config.state3_hold_reward,
                )
                if (
                    self.config.state3_hold_reward
                    > self.config.numeric_epsilon
                ):
                    events.append("state3_hold")

                self._reward_state3_closing(
                    gripper_rad=float(joints[-1]),
                    components=components,
                    events=events,
                )

            else:
                assert (
                    current_state
                    is RewardState.TARGET_RELEASE_RETURN_HOME
                )
                self._reward_state4_home(
                    joints=joints,
                    components=components,
                    events=events,
                )

            if state2_to_state3 or state2_to_state4:
                self._first_state2_phase_active = False

            self._previous_distance_m = distance_m

        home_hold_active = bool(
            current_state is RewardState.TARGET_RELEASE_RETURN_HOME
            and robot_at_home
        )
        if home_hold_active:
            self._success_home_hold_count += 1
            if self._success_home_hold_count == 1:
                events.append("success_home_hold_started")
        else:
            if self._success_home_hold_count > 0:
                events.append("success_home_hold_reset")
            self._success_home_hold_count = 0

        success = bool(
            self._success_home_hold_count
            >= self.config.success_home_hold_steps
        )
        if success:
            _add_component(
                components,
                "success",
                self.config.success_bonus,
            )
            events.append("success")

        reward = float(sum(components.values()))
        self._previous_joint_positions = joints.copy()
        self._previous_state = current_state

        return ReachRewardResult(
            reward=reward,
            state=current_state,
            success=success,
            distance_m=distance_m,
            distance_improvement_m=distance_improvement_m,
            alignment_error_deg=alignment_error_deg,
            target_age_steps=self._target_age_steps,
            robot_at_home=robot_at_home,
            target_in_goal_region=target_in_goal,
            success_home_hold_count=(
                self._success_home_hold_count
            ),
            components=components,
            events=tuple(events),
        )


def compute_reach_reward(
    tracker: StatefulReachReward,
    observation: ReachRewardInput,
) -> ReachRewardResult:
    """Compatibility entry point for one stateful reward step."""

    if not isinstance(tracker, StatefulReachReward):
        raise TypeError("tracker must be a StatefulReachReward.")
    return tracker.step(observation)
