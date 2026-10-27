"""Standalone S4 reward: LOWER -> RELEASE -> RAISE the empty gripper.

This module only measures existing TaskSnapshot evidence. It does not move the
arm, create a stage initializer, authorize a release, or change task outcomes.
Use the same episode reward ledger through stage handoffs and recovery visits.
The later_stage_training adapter enables this reward for S4. Numerical
guidance and physical preparation still require Gazebo validation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .later_stage_reward import (
    LaterRewardTuning,
    LaterStageReward,
    Milestone,
    StageSignals,
    live_carry,
    object_speed,
    placement_valid,
    quality,
    unit,
    validate_numbers,
)
from .pick_place_reward import PickPlaceRewardConfig
from .task_state import TaskSnapshot, object_geometry


@dataclass(frozen=True)
class S4PlaceReleaseRewardConfig:
    goal_center_tolerance_m: float = 0.010
    alignment_scale_m: float = 0.030
    descent_scale_m: float = 0.020
    linear_speed_scale_m_s: float = 0.050
    angular_speed_scale_rad_s: float = 0.400
    release_separation_m: float = 0.050
    place_hold_s: float = 0.1
    # Measured grasp-reference height above the goal support plane, not an
    # object lift or a distance-only withdrawal. Candidate defaults match S1.
    raise_clearance_m: float = 0.080
    raise_hold_s: float = 0.1
    aligned_bonus: float = 3.0
    lowered_bonus: float = 3.0
    released_bonus: float = 5.0
    withdrawn_bonus: float = 3.0
    placed_bonus: float = 5.0
    raised_bonus: float = 5.0

    def __post_init__(self) -> None:
        validate_numbers(self)


class S4PlaceReleaseReward(LaterStageReward):
    """Lower the held object, release it, then raise the empty gripper.

    Real release provenance remains certified by TaskStateTracker. S4 owns its
    0.1 s stable-placement dwell at 5 cm separation. Earlier alignment/descent
    qualities stay achieved after a
    legal release, so raising the gripper cannot erase placement progress.
    A thrown/pushed-in object has no valid delivery provenance and earns none
    of the post-release qualities or milestones. Alignment and slowing down
    remain small goals inside LOWER; separation and settling are inside RAISE.
    Empty-tool height remains a rewarded subgoal, not a success requirement.
    No automatic stage switch or robot command is implemented here.
    """

    stage = "S4"

    def __init__(self, reward_config: PickPlaceRewardConfig,
                 config: S4PlaceReleaseRewardConfig | None = None,
                 tuning: LaterRewardTuning | None = None):
        self.config = config or S4PlaceReleaseRewardConfig()
        c = self.config
        super().__init__(
            reward_config,
            weights={"alignment": 1.0, "descent": 1.5, "release_speed": 0.8,
                     "release": 1.0, "withdrawal": 1.0, "settling": 1.0,
                     "empty_raise": 1.5},
            milestones=(
                Milestone("aligned", c.aligned_bonus),
                Milestone("lowered", c.lowered_bonus),
                Milestone("released", c.released_bonus),
                Milestone("withdrawn", c.withdrawn_bonus),
                Milestone("placed", c.placed_bonus, c.place_hold_s),
                Milestone("raised", c.raised_bonus, c.raise_hold_s),
            ),
            completion_goal="placed",
            tuning=tuning,
        )

    def signals(self, s: TaskSnapshot) -> StageSignals:
        c, r, m = self.config, self.reward_config, s.measurements
        inside, bottom, center_error = object_geometry(m, r.goal_margin_m)
        clearance = bottom - m.goal_support_z_m
        linear_speed, angular_speed = object_speed(s)
        carrying = live_carry(s)
        arrived = bool(carrying and s.carried_into_goal and inside)
        released = bool(s.valid_release and s.delivery_valid and inside
                        and not s.grasp_attempt_active and not s.grasp_valid)
        no_finger_contact = not (m.left_contact or m.right_contact)
        # Losing bilateral contact may certify a legal release while one
        # finger still touches. Finish RELEASE only after both fingers clear.
        release_complete = released and no_finger_contact
        release_height_ok = (
            -r.support_height_tolerance_m <= clearance <= r.release_max_clearance_m
        )
        release_speed_ok = (linear_speed <= r.release_linear_velocity_max_m_s
                            and angular_speed <= r.release_angular_velocity_max_rad_s)
        centered = center_error <= c.goal_center_tolerance_m
        lowered = bool(arrived and centered and release_height_ok and release_speed_ok)
        withdrawn = bool(release_complete
                         and s.target_distance_m + 1e-9 >= c.release_separation_m)
        # Current evidence only: the shared stage milestone times this at
        # 0.1 s, independently of the full-task tracker's 0.4 s / 6 cm gate.
        placed = placement_valid(s, r, separation_m=c.release_separation_m)
        tool_clearance = float(m.grasp_position_world_m[2] - m.goal_support_z_m)
        raise_eligible = bool(release_complete and s.target_supported)
        raised = bool(placed and tool_clearance + 1e-9 >= c.raise_clearance_m)

        # The legal-height interval gives dense descent guidance without
        # encouraging penetration below the physical support surface.
        height_error = max(-r.support_height_tolerance_m - clearance,
                           clearance - r.release_max_clearance_m, 0.0)
        alignment = (1.0 if released else
                     quality(max(center_error - c.goal_center_tolerance_m, 0.0),
                             c.alignment_scale_m) if carrying else 0.0)
        descent = (1.0 if released else
                   quality(height_error, c.descent_scale_m) if arrived else 0.0)
        release_speed = (1.0 if released else
                         0.5 * (quality(max(linear_speed - r.release_linear_velocity_max_m_s, 0.0),
                                        c.linear_speed_scale_m_s)
                                + quality(max(angular_speed - r.release_angular_velocity_max_rad_s, 0.0),
                                          c.angular_speed_scale_rad_s)) if arrived else 0.0)
        opening = m.joint_positions_rad[-1] - r.home_joint_positions_rad[-1]
        # Small measured opening improvements guide the legal-release action;
        # opening aloft or outside the goal earns no such credit. Certification
        # still comes from the tracker, never from jaw angle alone.
        release_quality = (1.0 if release_complete else 0.5 if released else
                           0.25 * unit(opening / r.approach_opening_min_rad) if lowered else 0.0)
        # Releasing just one finger is not withdrawal, and a loose airborne
        # object cannot earn stable-placement progress merely by opening jaws.
        withdrawal = (unit(s.target_distance_m / c.release_separation_m)
                      if released and no_finger_contact else 0.0)
        # Timing progress is paid by the placed milestone's dwell ledger.
        # This instantaneous quality must not reuse the old tracker timer.
        settling = float(placed)
        # Continuous first-visit bands pay small upward improvements. Moving
        # sideways alone cannot satisfy this objective, nor can lifting the
        # object along with the fingers or raising before a legal release.
        empty_raise = unit(tool_clearance / c.raise_clearance_m) if raise_eligible else 0.0
        if release_complete:
            phase = "RAISE"
        elif lowered or released:
            phase = "RELEASE"
        else:
            phase = "LOWER"
        return StageSignals(
            phase=phase,
            qualities={"alignment": alignment, "descent": descent,
                       "release_speed": release_speed, "release": release_quality,
                       "withdrawal": withdrawal, "settling": settling,
                       "empty_raise": empty_raise},
            goals={"aligned": bool((arrived and centered) or released),
                   "lowered": bool(lowered or released), "released": release_complete,
                   "withdrawn": withdrawn, "placed": placed, "raised": raised},
            metrics={"goal_center_error_m": center_error,
                     "object_bottom_clearance_m": clearance,
                     "object_linear_speed_m_s": linear_speed,
                     "object_angular_speed_rad_s": angular_speed,
                     "opening_rad": opening,
                     "withdrawal_distance_m": s.target_distance_m,
                     "place_stable_duration_s": s.place_stable_duration_s,
                     "placement_candidate": float(placed),
                     "place_hold_required_s": c.place_hold_s,
                     "release_separation_required_m": c.release_separation_m,
                     "grasp_clearance_m": tool_clearance,
                     "raise_clearance_required_m": c.raise_clearance_m,
                     "raise_hold_required_s": c.raise_hold_s,
                     "raise_candidate": float(raised),
                     "release_complete": float(release_complete),
                     "recovery_required": float(not carrying and not released),
                     "inside_goal": float(inside), "live_carry": float(carrying),
                     "valid_delivery": float(released)},
        )
