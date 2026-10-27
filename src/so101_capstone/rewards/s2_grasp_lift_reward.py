"""Candidate S2 with two parts: CLOSE -> LIFT.

Used by later_stage_training. S1's distance-only endpoint is not a grasp certificate.
CLOSE includes fine alignment and measured contact confirmation, not merely a
closed gripper angle. LIFT starts only with a currently certified grasp and
finishes with the object held steadily; losing that grasp returns to CLOSE.
The object bottom must clear the source surface by 3 cm AND rise by 5 cm
relative to this grasp's acquisition height, then remain stable for 0.1 s.
These are S2's own thresholds, independent of the shared task tracker's lift
milestone. All milestones use measured contact/attempt history. No action,
gripper override, reset or retry controller is implemented in this file.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .later_stage_reward import (
    LaterRewardTuning, LaterStageReward, Milestone, StageSignals, live_grasp,
    attachment_command_reward, object_speed, quality, relative_slip_m, unit, validate_numbers,
)
from .pick_place_reward import PickPlaceRewardConfig
from .task_state import TaskSnapshot, object_geometry


@dataclass(frozen=True)
class S2RewardConfig:
    alignment_distance_m: float = 0.012
    alignment_tolerance_rad: float = math.radians(20)
    distance_scale_m: float = 0.01
    orientation_scale_rad: float = math.radians(15)
    lift_clearance_m: float = 0.03
    lift_rise_m: float = 0.05
    stable_slip_m: float = 0.005
    stable_linear_speed_m_s: float = 0.015
    stable_angular_speed_rad_s: float = 0.15
    stable_hold_s: float = 0.1
    alignment_bonus: float = 3.0
    dual_contact_bonus: float = 2.0
    grasp_bonus: float = 5.0
    lift_bonus: float = 5.0
    stable_hold_bonus: float = 5.0
    attachment_close_margin_scale_rad: float = math.radians(2.0)
    # Below the default 0.1/s time cost, so holding alone does not pay to stall.
    attachment_close_reward_per_s: float = 0.05
    attachment_open_cost_per_s: float = 0.2

    def __post_init__(self) -> None:
        validate_numbers(self)
        if self.alignment_tolerance_rad >= math.pi:
            raise ValueError("alignment tolerance must be smaller than pi")


class S2GraspLiftReward(LaterStageReward):
    stage = "S2"

    def __init__(self, reward_config: PickPlaceRewardConfig,
                 config: S2RewardConfig | None = None, tuning: LaterRewardTuning | None = None):
        self.config = config or S2RewardConfig()
        c = self.config
        if c.alignment_distance_m > reward_config.grasp_candidate_distance_m:
            raise ValueError("S2 alignment distance must fit the grasp candidate radius")
        if c.stable_slip_m > reward_config.relative_grasp_tolerance_m:
            raise ValueError("stable holding may not allow more slip than grasp certification")
        super().__init__(reward_config, tuning=tuning,
            weights=dict(centering=2, orientation=1, aperture=1, contact=1, lift=3, stability=2),
            milestones=(Milestone("aligned", c.alignment_bonus),
                        Milestone("dual_contact", c.dual_contact_bonus),
                        # CLOSE completion; keep the original once-only ledger key.
                        Milestone("grasped", c.grasp_bonus),
                        Milestone("lifted", c.lift_bonus),
                        # LIFT completion, in addition to the shared S2 completion bonus.
                        Milestone("stable_hold", c.stable_hold_bonus, c.stable_hold_s)),
            completion_goal="stable_hold")

    def _command_reward(self, snapshot: TaskSnapshot, *, covered_s: float,
                        fresh: bool, failed: bool) -> tuple[dict[str, float], dict[str, float]]:
        c = self.config
        return attachment_command_reward(snapshot, stage=self.stage,
            margin_scale_rad=c.attachment_close_margin_scale_rad,
            close_reward_per_s=c.attachment_close_reward_per_s,
            open_cost_per_s=c.attachment_open_cost_per_s,
            covered_s=covered_s, fresh=fresh, failed=failed)

    def signals(self, s: TaskSnapshot) -> StageSignals:
        m, c, r = s.measurements, self.config, self.reward_config
        target, grasp = np.asarray(m.target_position_world_m), np.asarray(m.grasp_position_world_m)
        delta = target - grasp
        distance = float(np.linalg.norm(delta))
        axis = np.asarray(m.grasp_rotation_world)[:, 2]
        direction = delta + r.approach_reference_offset_m * axis
        length = float(np.linalg.norm(direction))
        bearing = math.pi if length < 1e-9 else math.acos(float(np.clip(
            np.dot(axis, direction / length), -1, 1)))
        axis_error = math.acos(float(np.clip(np.dot(axis, r.desired_grasp_axis_world), -1, 1)))
        angle = max(bearing, axis_error)
        opening = m.joint_positions_rad[-1] - r.home_joint_positions_rad[-1]
        opening_error = max(r.approach_opening_min_rad - opening,
                            opening - r.approach_opening_max_rad, 0.0)
        held = live_grasp(s)
        _, bottom, _ = object_geometry(m, r.goal_margin_m)
        # Both absolute clearance and lift relative to THIS acquisition matter.
        clearance = bottom - m.source_support_z_m
        rise = bottom - s.grasp_origin_bottom_z_m if held else 0.0
        clearance_fraction = unit(clearance / c.lift_clearance_m) if held else 0.0
        rise_fraction = unit(rise / c.lift_rise_m) if held else 0.0
        # Both objectives give continuous feedback at their own scale. Keep
        # the existing lift-channel weight and once-only record ledger.
        lift_fraction = 0.5 * (clearance_fraction + rise_fraction)
        # The old task tracker has a separate 2 cm / .1 s lift certificate.
        # Do not add that delay before this stage's own .1 s stable dwell.
        lifted = bool(held and clearance + 1e-9 >= c.lift_clearance_m
                      and rise + 1e-9 >= c.lift_rise_m)
        slip = relative_slip_m(s) if held else r.relative_grasp_tolerance_m
        if not math.isfinite(slip):
            slip = r.relative_grasp_tolerance_m
        linear, angular = object_speed(s)
        stable = bool(lifted and slip <= c.stable_slip_m and
                      linear <= c.stable_linear_speed_m_s and angular <= c.stable_angular_speed_rad_s)
        near = distance <= c.alignment_distance_m and angle <= c.alignment_tolerance_rad
        aligned = held or (near and opening_error <= 1e-9)
        dual = bool(held or (m.contacts_simultaneous and m.left_contact and m.right_contact
                            and distance <= r.grasp_candidate_distance_m))
        # Blend open-jaw preparation into closure only near an aligned object.
        close_gate = unit((r.pregrasp_distance_tolerance_m - distance) /
                          (r.pregrasp_distance_tolerance_m - r.grasp_close_full_distance_m))
        close_gate *= unit(1 - angle / r.pregrasp_orientation_tolerance_rad)
        aperture = ((1 - close_gate) * quality(opening_error, r.approach_opening_scale_rad)
                    + close_gate * math.exp(-max(opening, 0) / r.approach_opening_max_rad))
        contact_fraction = unit(s.contact_duration_s / r.grasp_confirm_s) if dual else 0.0
        stability = (quality(max(0, slip - c.stable_slip_m), c.stable_slip_m)
                     * quality(max(0, linear - c.stable_linear_speed_m_s), c.stable_linear_speed_m_s)
                     * quality(max(0, angular - c.stable_angular_speed_rad_s), c.stable_angular_speed_rad_s))
        qualities = dict(
            centering=1.0 if held else quality(max(0, distance - c.alignment_distance_m), c.distance_scale_m),
            orientation=1.0 if held else quality(max(0, angle - c.alignment_tolerance_rad), c.orientation_scale_rad),
            aperture=1.0 if held else aperture,
            contact=1.0 if held else contact_fraction,
            lift=lift_fraction, stability=stability if lifted else 0.0)
        # The main two-part sequence follows current physical certification,
        # never an old milestone payment or a gripper-angle command. Fine
        # alignment/contact and lift/stability guidance remain as subgoals.
        phase = "LIFT" if held else "CLOSE"
        return StageSignals(phase, qualities,
            dict(aligned=aligned, dual_contact=dual, grasped=held, lifted=lifted, stable_hold=stable),
            dict(distance_m=distance, bearing_error_rad=bearing, axis_error_rad=axis_error,
                 opening_rad=opening, object_clearance_m=clearance, attempt_rise_m=rise,
                 object_clearance_required_m=c.lift_clearance_m, attempt_rise_required_m=c.lift_rise_m,
                 object_clearance_fraction=clearance_fraction, attempt_rise_fraction=rise_fraction,
                 relative_slip_m=slip, object_linear_speed_m_s=linear,
                 object_angular_speed_rad_s=angular, grasp_attempt_id=float(s.grasp_attempt_id),
                 close_alignment_ready=float(aligned), close_dual_contact=float(dual),
                 close_complete=float(held), lift_height_confirmed=float(lifted),
                 lift_stable_candidate=float(stable)))
