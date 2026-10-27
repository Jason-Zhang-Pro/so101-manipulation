"""Candidate S3: secure clearance -> carry the full footprint into the goal.

Used by later_stage_training. Progress follows the OBJECT, gated by the current
certified grasp/lift. A previous successful lift or a pushed-in object cannot
unlock transport reward. Dropping returns RECOVER_GRASP diagnostics; deciding
whether/where to retry belongs to the training lifecycle, never this reward.
Success requires the current valid carry and the entire object footprint inside
the goal for 0.1 s. Centering, hover height and low speed remain process guidance,
not completion gates. Geometric membership includes the boundary, without the
full-task tracker's additional goal margin.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .later_stage_reward import (
    LaterRewardTuning, LaterStageReward, Milestone, StageSignals, live_carry,
    attachment_command_reward, object_speed, quality, relative_slip_m, unit, validate_numbers,
)
from .pick_place_reward import PickPlaceRewardConfig
from .task_state import TaskSnapshot, object_geometry, quaternion_rotation


@dataclass(frozen=True)
class S3RewardConfig:
    hover_min_clearance_m: float = 0.02
    hover_max_clearance_m: float = 0.05
    goal_center_tolerance_m: float = 0.01
    distance_scale_m: float = 0.04
    height_scale_m: float = 0.02
    stable_slip_m: float = 0.005
    hover_linear_speed_m_s: float = 0.02
    hover_angular_speed_rad_s: float = 0.15
    transport_hold_s: float = 0.1
    clearance_bonus: float = 3.0
    arrival_bonus: float = 5.0
    centered_bonus: float = 3.0
    hover_bonus: float = 5.0  # Existing terminal bonus now pays transported.
    attachment_close_margin_scale_rad: float = math.radians(2.0)
    # Below the default 0.1/s time cost, so holding alone does not pay to stall.
    attachment_close_reward_per_s: float = 0.05
    attachment_open_cost_per_s: float = 0.2

    def __post_init__(self) -> None:
        validate_numbers(self)
        if self.hover_max_clearance_m <= self.hover_min_clearance_m:
            raise ValueError("hover interval must be ordered")


class S3TransportReward(LaterStageReward):
    stage = "S3"

    def __init__(self, reward_config: PickPlaceRewardConfig,
                 config: S3RewardConfig | None = None, tuning: LaterRewardTuning | None = None):
        self.config = config or S3RewardConfig()
        c = self.config
        if c.hover_min_clearance_m < reward_config.transport_clearance_m:
            raise ValueError("hover minimum must retain certified transport clearance")
        if c.stable_slip_m > reward_config.relative_grasp_tolerance_m:
            raise ValueError("hover slip tolerance exceeds grasp certification")
        super().__init__(reward_config, tuning=tuning,
            weights=dict(clearance=2, transport=4, centering=2, stability=2),
            milestones=(Milestone("safe_clearance", c.clearance_bonus),
                        Milestone("arrived", c.arrival_bonus),
                        Milestone("centered", c.centered_bonus),
                        Milestone("transported", c.hover_bonus, c.transport_hold_s)),
            completion_goal="transported")

    def _command_reward(self, snapshot: TaskSnapshot, *, covered_s: float,
                        fresh: bool, failed: bool) -> tuple[dict[str, float], dict[str, float]]:
        c = self.config
        return attachment_command_reward(snapshot, stage=self.stage,
            margin_scale_rad=c.attachment_close_margin_scale_rad,
            close_reward_per_s=c.attachment_close_reward_per_s,
            open_cost_per_s=c.attachment_open_cost_per_s,
            covered_s=covered_s, fresh=fresh, failed=failed)

    def signals(self, s: TaskSnapshot) -> StageSignals:
        c, r, m = self.config, self.reward_config, s.measurements
        inside, bottom, distance = object_geometry(m)
        margin_inside, _, _ = object_geometry(m, r.goal_margin_m)
        held = live_carry(s)
        # live_carry already certifies a current source/legal-recovery grasp and
        # a real lift in THIS attempt. The tracker's carried_into_goal flag also
        # requires extra height/margin, so it cannot gate this stage's new goal.
        transported = bool(held and inside)
        # Clear both planes while transporting, including a raised goal pad.
        clearance = bottom - max(m.source_support_z_m, m.goal_support_z_m)
        goal_clearance = bottom - m.goal_support_z_m
        safe = bool(held and clearance >= c.hover_min_clearance_m - 1e-9)
        arrived = bool(safe and margin_inside and s.carried_into_goal)
        centered = bool(arrived and distance <= c.goal_center_tolerance_m)
        height_error = max(c.hover_min_clearance_m - goal_clearance,
                           goal_clearance - c.hover_max_clearance_m, 0.0)
        slip = relative_slip_m(s) if held else r.relative_grasp_tolerance_m
        linear, angular = object_speed(s)
        stable = bool(centered and height_error <= 1e-9 and slip <= c.stable_slip_m
                      and linear <= c.hover_linear_speed_m_s and angular <= c.hover_angular_speed_rad_s)
        height_quality = quality(height_error, c.height_scale_m)
        clearance_gate = unit(clearance / c.hover_min_clearance_m) if held else 0.0
        carry_gate = float(held) * clearance_gate
        # Deceleration matters close to the destination, not while crossing it.
        proximity = quality(max(0, distance - c.goal_center_tolerance_m), c.distance_scale_m)
        cy, sy = math.cos(m.goal_yaw_rad), math.sin(m.goal_yaw_rad)
        goal_rotation = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
        relative = goal_rotation.T @ (np.asarray(m.target_position_world_m) - m.goal_center_world_m)
        projected_half = np.abs(goal_rotation.T @ quaternion_rotation(m.target_quaternion_xyzw)) @ m.target_half_extents_m
        overflow = float(np.max(np.maximum(0.0, np.abs(relative[:2]) + projected_half[:2]
                                           + r.goal_margin_m - m.goal_half_extents_xy_m)))
        speed_quality = (quality(max(0, linear - c.hover_linear_speed_m_s), c.hover_linear_speed_m_s)
                         * quality(max(0, angular - c.hover_angular_speed_rad_s), c.hover_angular_speed_rad_s))
        slip_quality = quality(max(0, slip - c.stable_slip_m), c.stable_slip_m)
        qualities = dict(clearance=float(held) * height_quality,
                         transport=carry_gate * proximity,
                         centering=carry_gate * proximity * quality(overflow, c.distance_scale_m),
                         stability=carry_gate * proximity * height_quality * slip_quality * speed_quality)
        phase = ("RECOVER_GRASP" if not held else "HOLD_INSIDE_GOAL" if transported else
                 "ESTABLISH_CLEARANCE" if not safe else
                 "CARRY_TO_GOAL")
        # Infinity represents a missing anchor internally; metrics stay finite.
        shown_slip = min(slip, r.relative_grasp_tolerance_m * 10)
        return StageSignals(phase, qualities,
            dict(safe_clearance=safe, arrived=arrived, centered=centered, transported=transported),
            dict(object_goal_xy_m=distance, object_clearance_m=clearance,
                 footprint_overflow_m=overflow,
                 goal_clearance_m=goal_clearance, object_inside_goal=float(inside),
                 transport_candidate=float(transported),
                 transport_hold_required_s=c.transport_hold_s,
                 stable_hover_candidate=float(stable),
                 relative_slip_m=shown_slip, object_linear_speed_m_s=linear,
                 object_angular_speed_rad_s=angular))
