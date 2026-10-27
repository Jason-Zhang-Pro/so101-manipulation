"""Physical S2--S5 starts, separate from ACT actions and PPO transitions.

The runtime supplies one deterministic 30 Hz physics step and keeps ONE live
TaskStateTracker throughout preparation. This controller only sends bounded
joint targets. It never teleports an object, invents contact, resets tracker
history, or rewards its own preparation. This is a guarded waypoint controller,
not a general collision planner; unreachable or stalled preparation fails
closed and must be recorded separately from policy episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Callable

import numpy as np

from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS
from so101_capstone.robot.so101_forward_kinematics import ACTIVE_JOINT_NAMES, compute_grasp_point_pose
from so101_capstone.robot.so101_inverse_kinematics import IKConfig, InverseKinematicsError, solve_grasp_point_ik
from so101_capstone.rewards.later_stage_reward import live_grasp, live_carry, placement_valid, relative_slip_m, object_speed
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.s2_grasp_lift_reward import S2RewardConfig
from so101_capstone.rewards.s3_transport_reward import S3RewardConfig
from so101_capstone.rewards.s4_place_release_reward import S4PlaceReleaseRewardConfig
from so101_capstone.rewards.task_state import TaskSnapshot, object_geometry


CONTROL_DT_S = 1.0 / 30.0
# Gazebo's existing 1 ms physics clock implements 30 Hz as 33/33/34 ms.
MIN_CONTROL_DT_S = 0.033
MAX_CONTROL_DT_S = 0.034


@dataclass(frozen=True)
class StagePreparationConfig:
    joint_speed_rad_s: float = 0.5
    gripper_speed_rad_s: float = 0.8
    cartesian_speed_m_s: float = 0.06
    cartesian_tolerance_m: float = 0.003
    joint_tolerance_rad: float = 0.035
    max_tracking_error_rad: float = 0.25
    phase_timeout_s: float = 20.0
    total_timeout_s: float = 120.0
    source_settle_s: float = 0.1
    source_hover_clearance_m: float = 0.08
    pregrasp_offset_z_m: float = 0.006
    pregrasp_distance_m: float = 0.02
    carry_clearance_m: float = 0.06
    goal_hover_clearance_m: float = 0.04
    empty_raise_clearance_m: float = 0.08
    floor_clearance_m: float = 0.004
    axis_tolerance_rad: float = math.radians(20)

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{f.name} must be positive and finite")
        if self.joint_speed_rad_s > 1 or self.gripper_speed_rad_s > 1.5 or self.cartesian_speed_m_s > .15:
            raise ValueError("preparation speed exceeds its conservative configuration ceiling")
        if self.total_timeout_s < self.phase_timeout_s or self.total_timeout_s > 600:
            raise ValueError("total_timeout_s must cover a phase and be at most 600 seconds")
        if self.phase_timeout_s < CONTROL_DT_S or self.cartesian_tolerance_m >= self.pregrasp_distance_m:
            raise ValueError("invalid preparation time or Cartesian tolerances")
        if self.pregrasp_offset_z_m + self.cartesian_tolerance_m >= self.pregrasp_distance_m:
            raise ValueError("pregrasp offset and tolerance must fit strictly inside the S2 radius")
        if self.pregrasp_distance_m > .02:
            raise ValueError("S2 preparation may not relax the 2 cm near-target boundary")
        if not 0 < self.axis_tolerance_rad < math.pi / 2:
            raise ValueError("preparation axis tolerance must be below pi/2")
        if self.goal_hover_clearance_m < .03 or self.goal_hover_clearance_m > .05:
            raise ValueError("goal hover must leave 3--5 cm for the S4 descent")
        if self.carry_clearance_m < self.goal_hover_clearance_m:
            raise ValueError("transport clearance may not be below destination hover")
        if self.source_hover_clearance_m < .08 or self.empty_raise_clearance_m < .08:
            raise ValueError("source/empty-tool preparation needs at least 8 cm clearance")


class StagePreparationError(RuntimeError):
    def __init__(self, phase: str, steps: int, reason: str):
        self.phase, self.steps, self.reason = phase, steps, reason
        self.stage_initialization = dict(phase=phase, preparation_control_steps=steps, reason=reason)
        super().__init__(f"stage preparation {phase} failed after {steps} steps: {reason}")


@dataclass(frozen=True)
class StagePreparationReport:
    stage: str
    final_snapshot: TaskSnapshot
    last_command_joint_targets_rad: tuple[float, ...]
    steps: int
    phases: tuple[str, ...]
    certificate: dict[str, float | bool]


class StagePreparationController:
    """Build a later-stage start from an empty gripper and a source object.

    ``advance(command)`` must execute one real control step (the existing
    33/33/34 ms schedule) and return a measured snapshot. Sensor evidence can
    legitimately repeat between frames; only fresh evidence certifies holds.
    The callback must not collect PPO transitions or recreate/reset the target.
    One instance can be reused serially; instances may not be run concurrently.
    """

    def __init__(self, reward_config: PickPlaceRewardConfig,
                 config: StagePreparationConfig | None = None):
        self.reward_config = reward_config
        self.config = config or StagePreparationConfig()
        self.s2, self.s3, self.s4 = S2RewardConfig(), S3RewardConfig(), S4PlaceReleaseRewardConfig()
        self.lower = np.array([j.gazebo_lower_rad for j in SIM_JOINT_CALIBRATIONS])
        self.upper = np.array([j.gazebo_upper_rad for j in SIM_JOINT_CALIBRATIONS])
        self.open_gripper = reward_config.home_joint_positions_rad[-1] + .5 * (
            reward_config.approach_opening_min_rad + reward_config.approach_opening_max_rad)
        self.closed_gripper = reward_config.home_joint_positions_rad[-1]
        if not self.lower[-1] <= self.closed_gripper < self.open_gripper <= self.upper[-1]:
            raise ValueError("configured preparation gripper positions exceed calibrated limits")

    def _fail(self, reason: str):
        raise StagePreparationError(self.phase, self.steps, reason)

    def _phase(self, name: str):
        self.phase = name
        self.phases.append(name)
        self.phase_start = self.steps

    def _validate(self, s: TaskSnapshot):
        if not isinstance(s, TaskSnapshot):
            self._fail("advance must return a TaskSnapshot")
        m = s.measurements
        if not (m.evidence_valid and m.joint_measurement_valid and m.target_measurement_valid
                and m.target_motion_valid and m.contact_valid):
            self._fail("invalid joint/target/motion/contact evidence")
        if m.grasp_evidence_mode != "contact":
            self._fail("physical preparation requires actual contact evidence")
        if s.outcome != "running" or m.failure_reason or s.failure_reason:
            self._fail(f"task no longer running: {s.failure_reason or m.failure_reason or s.outcome}")
        if s.target_name is None or s.task_mode != "pick_place":
            self._fail("preparation requires a source pick_place target")
        if np.any(np.asarray(m.joint_positions_rad) < self.lower - .02) or np.any(np.asarray(m.joint_positions_rad) > self.upper + .02):
            self._fail("measured joint position outside calibrated range")

    def _tick(self, target, *, require_grasp=False, allow_contact=True):
        c = self.config
        if self.steps >= math.ceil(c.total_timeout_s / CONTROL_DT_S):
            self._fail("total preparation timeout")
        if self.steps - self.phase_start >= math.ceil(c.phase_timeout_s / CONTROL_DT_S):
            self._fail("phase timeout (unreachable, stalled, or unmet physical certificate)")
        target = np.asarray(target, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all() or np.any(target < self.lower) or np.any(target > self.upper):
            self._fail("invalid/out-of-bounds controller target")
        velocity = np.array([c.joint_speed_rad_s] * 5 + [c.gripper_speed_rad_s])
        measured = np.asarray(self.snapshot.measurements.joint_positions_rad)
        if np.max(np.abs(measured[:5] - self.command[:5])) > c.max_tracking_error_rad:
            self._fail("arm feedback tracking error: motion may be blocked")
        delta = target - self.command
        # Scale the arm vector together, preserving the IK segment direction;
        # component-wise clipping can bend a tiny Cartesian step into the table.
        delta[:5] *= min(1.0, c.joint_speed_rad_s * MIN_CONTROL_DT_S /
                         max(np.max(np.abs(delta[:5])), 1e-12))
        delta[-1] = np.clip(delta[-1], -velocity[-1] * MIN_CONTROL_DT_S, velocity[-1] * MIN_CONTROL_DT_S)
        command = self.command + delta
        # This guards the calibrated reference point along the short joint
        # segment; it is not a guarantee of all-link collision freedom.
        for fraction in (.25, .5, .75, 1.0):
            q = self.command + fraction * (command - self.command)
            position = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q[:5]))).position
            if position[2] < self.floor_z - 1e-6:
                self._fail("planned grasp-reference segment crosses the support-plane guard")
        previous = self.snapshot
        self.steps += 1
        try:
            s = self.advance(tuple(float(v) for v in command))
        except StagePreparationError:
            raise
        except Exception as exc:
            self._fail(f"physics advance failed: {type(exc).__name__}: {exc}")
        self._validate(s)
        if (s.task_instance_id, s.target_name) != self.identity:
            self._fail("physical episode/target identity changed during preparation")
        dt = s.sim_time_s - previous.sim_time_s
        if not MIN_CONTROL_DT_S - 1e-6 <= dt <= MAX_CONTROL_DT_S + 1e-6:
            self._fail("advance did not preserve deterministic 30 Hz stepping")
        evidence_dt = s.evidence_sim_time_s - previous.evidence_sim_time_s
        if not -1e-9 <= evidence_dt <= self.reward_config.max_evidence_gap_s + 1e-9 or (
                s.sim_time_s - s.evidence_sim_time_s > self.reward_config.max_evidence_gap_s + 1e-9):
            self._fail("stale or discontinuous preparation evidence")
        self.snapshot, self.command = s, command
        if require_grasp and not live_grasp(s):
            self._fail("grasp lost during preparation; rebuild instead of inventing a held start")
        if not allow_contact and (s.measurements.left_contact or s.measurements.right_contact or live_grasp(s)):
            self._fail("unexpected target contact during open-gripper positioning")
        return s

    def _hold(self, predicate, duration: float, *, require_grasp=False, command=None):
        hold = 0.0
        previous_ok = bool(predicate(self.snapshot))
        hold_boundary = self.snapshot.sim_time_s
        target = self.command.copy() if command is None else np.asarray(command).copy()
        attempt = self.snapshot.grasp_attempt_id
        while True:
            old = self.snapshot
            s = self._tick(target, require_grasp=require_grasp)
            current_ok = bool(predicate(s))
            evidence_dt = s.evidence_sim_time_s - old.evidence_sim_time_s
            if not current_ok or attempt != s.grasp_attempt_id:
                hold = 0.0
                hold_boundary = s.sim_time_s
            elif evidence_dt > 1e-9 and previous_ok:
                hold += min(evidence_dt, max(0.0, s.evidence_sim_time_s - hold_boundary))
            if evidence_dt > 1e-9 or not current_ok:
                previous_ok = current_ok
            attempt = s.grasp_attempt_id
            if current_ok and hold + 1e-9 >= duration:
                return

    def _move(self, position, *, axis=None, require_grasp=False, allow_contact=True):
        position = np.asarray(position, dtype=float)
        axis = np.asarray(self.snapshot.measurements.grasp_rotation_world)[:, 2] if axis is None else np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        while True:
            m = self.snapshot.measurements
            current = np.asarray(m.grasp_position_world_m)
            delta = position - current
            axis_error = math.acos(np.clip(np.asarray(m.grasp_rotation_world)[:, 2] @ axis, -1, 1))
            if np.linalg.norm(delta) <= self.config.cartesian_tolerance_m and axis_error <= self.config.axis_tolerance_rad:
                return
            self._move_step(delta, axis, require_grasp=require_grasp, allow_contact=allow_contact)

    def _move_step(self, translation, axis, *, require_grasp=False, allow_contact=True):
        """Advance one small Cartesian segment; remeasure before planning more."""
        measured = np.asarray(self.snapshot.measurements.grasp_position_world_m)
        goal = measured + np.asarray(translation, dtype=float)
        # Task measurements align joints to target/contact evidence, sometimes
        # at 10 Hz. Advance the bounded command path at every 30 Hz tick rather
        # than restarting the same segment three times from repeated evidence.
        # Completion still uses measured poses; _tick enforces tracking error.
        planned = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, self.command[:5]))).position
        translation = goal - planned
        waypoint = planned + translation * min(
            1.0, self.config.cartesian_speed_m_s * CONTROL_DT_S /
            max(np.linalg.norm(translation), 1e-12))
        # Accept up to 1 cm, but first seek a precise fit: the 2 mm waypoint
        # itself lies inside that tolerance even before the arm has moved.
        cfg = IKConfig(position_tolerance_m=.01, preferred_position_tolerance_m=.0005,
                       axis_tolerance_rad=self.config.axis_tolerance_rad, multistart=False)
        try:
            solution = solve_grasp_point_ik(waypoint, self.command[:5], desired_axis_world=axis, config=cfg)
        except InverseKinematicsError as exc:
            self._fail(str(exc))
        target = np.array((*solution.joint_positions_rad, self.command[-1]))
        self._tick(target, require_grasp=require_grasp, allow_contact=allow_contact)

    def _move_object(self, desired_bottom: float, xy=None):
        # A source-specific tilted axis cannot generally remain fixed while
        # this five-joint arm travels to the side goal. Keep the configured
        # downward cone instead, allowing the local IK branch to turn. Since
        # that turns the held object too, close the loop on its measured bottom
        # and center on EVERY step, rather than translating a stale tip target.
        initial = self.snapshot.measurements
        target_xy = np.asarray(initial.target_position_world_m[:2] if xy is None else xy, dtype=float)
        axis = np.asarray(self.reward_config.desired_grasp_axis_world, dtype=float)
        axis /= np.linalg.norm(axis)
        while True:
            m = self.snapshot.measurements
            _, bottom, _ = object_geometry(m)
            delta = np.array((*(target_xy - np.asarray(m.target_position_world_m[:2])),
                              desired_bottom - bottom))
            axis_error = math.acos(np.clip(np.asarray(m.grasp_rotation_world)[:, 2] @ axis, -1, 1))
            if np.linalg.norm(delta) <= self.config.cartesian_tolerance_m and axis_error <= self.config.axis_tolerance_rad:
                return
            self._move_step(delta, axis, require_grasp=True)

    def _source_ready(self, s):
        m, r = s.measurements, self.reward_config
        _, bottom, _ = object_geometry(m)
        linear, angular = object_speed(s)
        return bool(not live_grasp(s) and not s.delivery_valid and not (m.left_contact or m.right_contact)
                    and abs(bottom - m.source_support_z_m) <= r.support_height_tolerance_m
                    and linear <= r.object_linear_velocity_tolerance_m_s
                    and angular <= r.object_angular_velocity_tolerance_rad_s)

    def _s2_start(self, s):
        m = s.measurements
        distance = float(np.linalg.norm(np.asarray(m.grasp_position_world_m) - m.target_position_world_m))
        opening = m.joint_positions_rad[-1] - self.reward_config.home_joint_positions_rad[-1]
        return bool(self._source_ready(s) and distance < self.config.pregrasp_distance_m
                    and self.reward_config.approach_opening_min_rad <= opening <= self.reward_config.approach_opening_max_rad)

    def _s2_complete(self, s):
        c, m = self.s2, s.measurements
        _, bottom, _ = object_geometry(m)
        linear, angular = object_speed(s)
        return bool(live_grasp(s) and bottom - m.source_support_z_m + 1e-9 >= c.lift_clearance_m
                    and bottom - s.grasp_origin_bottom_z_m + 1e-9 >= c.lift_rise_m
                    and relative_slip_m(s) <= c.stable_slip_m
                    and linear <= c.stable_linear_speed_m_s and angular <= c.stable_angular_speed_rad_s)

    def _s3_complete(self, s):
        return bool(live_carry(s) and object_geometry(s.measurements)[0])

    def _s4_start(self, s):
        inside, bottom, _ = object_geometry(s.measurements)
        clearance = bottom - s.measurements.goal_support_z_m
        return bool(live_carry(s) and inside and .03 - 1e-9 <= clearance <= .05 + 1e-9)

    def _s4_complete(self, s):
        return placement_valid(s, self.reward_config, separation_m=self.s4.release_separation_m)

    def _release_ready_to_raise(self, s):
        m = s.measurements
        if not (s.valid_release and s.delivery_valid and not live_grasp(s)):
            return False
        if m.grasp_attachment_enabled:
            # After detachment, a resting cube may still touch the fixed finger.
            # Withdraw the opened empty tool to remove that contact, then keep
            # the existing strict two-finger/separation/stability final check.
            return bool(not m.grasp_attachment_active and
                        abs(m.joint_positions_rad[-1] - self.open_gripper)
                        <= self.config.joint_tolerance_rad)
        return not (m.left_contact or m.right_contact)

    def run(self, stage: str, initial: TaskSnapshot,
            advance: Callable[[tuple[float, ...]], TaskSnapshot]) -> StagePreparationReport:
        stage = str(stage).upper()
        if stage not in ("S2", "S3", "S4", "S5"):
            raise ValueError("StagePreparationController supports S2--S5; S0/S1 use existing HOME preparation")
        if not callable(advance):
            raise TypeError("advance must be callable")
        self.snapshot, self.advance = initial, advance
        self.steps, self.phases, self.phase_start = 0, [], 0
        self._phase("SOURCE_SETTLE")
        self._validate(initial)
        self.identity = (initial.task_instance_id, initial.target_name)
        self.command = np.clip(np.asarray(initial.measurements.joint_positions_rad), self.lower, self.upper)
        self.floor_z = initial.measurements.source_support_z_m + self.config.floor_clearance_m
        if live_grasp(initial) or initial.delivery_valid or initial.grasp_attempt_id:
            self._fail("start must be a fresh source episode, not a reset of certified carry history")
        self._hold(self._source_ready, self.config.source_settle_s)
        self._phase("RAISE_FROM_HOME")
        point = np.asarray(self.snapshot.measurements.grasp_position_world_m).copy()
        point[2] = max(point[2], self.snapshot.measurements.source_support_z_m + self.config.source_hover_clearance_m)
        self._move(point, allow_contact=False)
        self._phase("OPEN_GRIPPER")
        opened = self.command.copy()
        opened[-1] = self.open_gripper
        self._hold(lambda s: abs(s.measurements.joint_positions_rad[-1] - self.open_gripper) <= self.config.joint_tolerance_rad,
                   CONTROL_DT_S, command=opened)
        self._phase("ABOVE_SOURCE")
        m = self.snapshot.measurements
        source = np.asarray(m.target_position_world_m).copy()
        source[2] = m.source_support_z_m + self.config.source_hover_clearance_m
        self._move(source, axis=self.reward_config.desired_grasp_axis_world, allow_contact=False)
        self._phase("DESCEND_OPEN")
        point = np.asarray(self.snapshot.measurements.target_position_world_m).copy()
        point[2] += self.config.pregrasp_offset_z_m
        self._move(point, axis=self.reward_config.desired_grasp_axis_world, allow_contact=False)
        self._hold(self._s2_start, self.config.source_settle_s)
        if stage == "S2":
            return self._report(stage)
        self._phase("CLOSE_AND_CERTIFY")
        closed = self.command.copy()
        closed[-1] = self.closed_gripper
        while not live_grasp(self.snapshot):
            self._tick(closed)
        # Keep the exact command that established real grasp, with no extra
        # closure. Carrying moves preserve it until an explicit opening action.
        self._phase("LIFT_OBJECT")
        bottom = max(self.snapshot.measurements.source_support_z_m + self.s2.lift_clearance_m,
                     self.snapshot.grasp_origin_bottom_z_m + self.s2.lift_rise_m) + .005
        self._move_object(bottom)
        self._phase("VERIFY_LIFT")
        self._hold(self._s2_complete, self.s2.stable_hold_s, require_grasp=True)
        if stage == "S3":
            return self._report(stage)
        self._phase("TRANSPORT_CLEARANCE")
        m = self.snapshot.measurements
        self._move_object(max(m.source_support_z_m, m.goal_support_z_m) + self.config.carry_clearance_m)
        self._phase("CARRY_TO_GOAL")
        m = self.snapshot.measurements
        self._move_object(max(m.source_support_z_m, m.goal_support_z_m) + self.config.carry_clearance_m,
                          m.goal_center_world_m[:2])
        self._phase("GOAL_HOVER")
        self._move_object(self.snapshot.measurements.goal_support_z_m + self.config.goal_hover_clearance_m)
        self._hold(self._s4_start, self.s3.transport_hold_s, require_grasp=True)
        if stage == "S4":
            return self._report(stage)
        self._phase("LOWER_TO_SUPPORT")
        self.floor_z = max(self.floor_z, self.snapshot.measurements.goal_support_z_m + self.config.floor_clearance_m)
        self._move_object(self.snapshot.measurements.goal_support_z_m + .001)
        self._hold(lambda s: object_speed(s)[0] <= self.reward_config.release_linear_velocity_max_m_s
                   and object_speed(s)[1] <= self.reward_config.release_angular_velocity_max_rad_s,
                   self.config.source_settle_s, require_grasp=True)
        self._phase("RELEASE_OBJECT")
        opened = self.command.copy()
        opened[-1] = self.open_gripper
        self._hold(self._release_ready_to_raise, CONTROL_DT_S, command=opened)
        self._phase("RAISE_EMPTY_GRIPPER")
        point = np.asarray(self.snapshot.measurements.grasp_position_world_m).copy()
        point[2] = (self.snapshot.measurements.goal_support_z_m + self.config.empty_raise_clearance_m
                    + self.config.cartesian_tolerance_m)
        self._move(point)
        self._phase("VERIFY_PLACEMENT")
        self._hold(lambda s: self._s4_complete(s) and s.measurements.grasp_position_world_m[2]
                   + 1e-9 >= s.measurements.goal_support_z_m + self.config.empty_raise_clearance_m,
                   self.s4.place_hold_s)
        return self._report(stage)

    def _report(self, stage):
        s, m = self.snapshot, self.snapshot.measurements
        inside, bottom, _ = object_geometry(m)
        certificate = dict(grasp_valid=live_grasp(s), lifted=bool(s.attempt_lifted),
                           inside_goal=inside, valid_release=bool(s.valid_release),
                           delivery_valid=bool(s.delivery_valid),
                           object_source_clearance_m=bottom - m.source_support_z_m,
                           object_grasp_rise_m=bottom - s.grasp_origin_bottom_z_m if s.grasp_attempt_id else 0.0,
                           object_goal_clearance_m=bottom - m.goal_support_z_m,
                           grasp_target_distance_m=s.target_distance_m,
                           grasp_goal_clearance_m=m.grasp_position_world_m[2] - m.goal_support_z_m,
                           s2_start=self._s2_start(s), s2_complete=self._s2_complete(s),
                           s3_complete=self._s3_complete(s), s4_start=self._s4_start(s),
                           s4_complete=self._s4_complete(s))
        self._phase("READY")
        return StagePreparationReport(stage, s, tuple(float(v) for v in self.command),
                                      self.steps, tuple(self.phases), certificate)
