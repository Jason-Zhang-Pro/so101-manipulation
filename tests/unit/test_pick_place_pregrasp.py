"""Physical pregrasp guidance must improve geometry without certifying success."""

from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import (
    PickPlaceReward, PickPlaceRewardConfig, PickPlaceRewardInput,
    TaskMeasurements, TaskPhase, TaskStateTracker, TASK_CONTEXT_DIM,
)
from so101_capstone.rewards.task_state import TASK_CONTEXT_SCHEMA_VERSION, encode_task_context


def measurement(t=0.0, *, distance=0.01, opening=0.6, aligned=True, target=True, **changes):
    cfg = PickPlaceRewardConfig()
    joints = list(cfg.home_joint_positions_rad)
    joints[-1] += opening
    # Local +Z points down toward a table object; an alternate orientation
    # puts it sideways. No fingertip positions are fabricated by this fixture.
    rotation = np.diag([1.0, -1.0, -1.0]) if aligned else np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    values = dict(
        sim_time_s=t, joint_positions_rad=joints, joint_velocities_rad_s=(0.,) * 6,
        grasp_position_world_m=(0., 0., 0.765 + distance), grasp_rotation_world=rotation,
        target_name="cube" if target else None,
        target_position_world_m=(0., 0., 0.765) if target else None,
        target_motion_valid=target, contact_valid=True,
    )
    values.update(changes)
    return TaskMeasurements(**values)


def snapshot(**changes):
    return TaskStateTracker(PickPlaceRewardConfig()).reset(measurement(**changes))


def test_no_contact_near_target_keeps_precise_distance_signal():
    reward = PickPlaceReward(PickPlaceRewardConfig())
    snapshots = [snapshot(distance=distance) for distance in (0.044, 0.03, 0.02, 0.015, 0.008, 0.004, 0.)]
    potentials = [reward.potential(s) for s in snapshots]
    assert all(s.phase is TaskPhase.GRASP_LIFT and not s.grasp_valid for s in snapshots)
    assert all(right > left for left, right in zip(potentials, potentials[1:]))
    assert potentials[-1] - potentials[0] > 0.5


def test_approach_candidate_boundary_has_no_phase_bonus_or_potential_jump():
    reward = PickPlaceReward(PickPlaceRewardConfig())
    outside, inside = snapshot(distance=0.045001), snapshot(distance=0.044999)
    assert outside.phase is TaskPhase.APPROACH
    assert inside.phase is TaskPhase.GRASP_LIFT
    assert 0 < reward.potential(inside)-reward.potential(outside) < 0.001
    assert reward.potential(inside) == reward.potential(replace(inside, phase=TaskPhase.APPROACH))


def test_far_reach_still_has_nonzero_distance_signal():
    reward = PickPlaceReward(PickPlaceRewardConfig())
    assert reward.potential(snapshot(distance=.35)) > reward.potential(snapshot(distance=.40)) > 0


def test_alignment_and_opening_are_visible_before_any_contact():
    reward = PickPlaceReward(PickPlaceRewardConfig())
    aligned = snapshot(distance=.025, opening=.6)
    sideways = snapshot(distance=.025, opening=.6, aligned=False)
    closed = snapshot(distance=.025, opening=0.)
    assert aligned.approach_orientation_score > sideways.approach_orientation_score
    assert reward.potential(aligned) > reward.potential(sideways)
    assert reward.potential(aligned) > reward.potential(closed)


def test_close_guidance_is_local_and_does_not_reward_empty_or_sideways_closing():
    reward = PickPlaceReward(PickPlaceRewardConfig())
    far_open, far_closed = snapshot(distance=.08), snapshot(distance=.08, opening=0.)
    side_open = snapshot(distance=.019, aligned=False)
    side_closed = snapshot(distance=.019, opening=0., aligned=False)
    empty_open, empty_closed = snapshot(target=False), snapshot(target=False, opening=0.)
    assert far_open.grasp_close_gate == side_open.grasp_close_gate == empty_open.grasp_close_gate == 0
    assert reward.potential(far_closed) < reward.potential(far_open)
    assert reward.potential(side_closed) < reward.potential(side_open)
    # Empty scenes use HOME, with no closure contribution from pregrasp.
    assert empty_open.approach_score == empty_closed.approach_score == 0
    assert empty_open.grasp_closure_score == empty_closed.grasp_closure_score == 0


def test_centered_closure_improves_potential_without_losing_an_opening_bonus():
    reward = PickPlaceReward(PickPlaceRewardConfig())
    states = [snapshot(distance=.005, opening=opening) for opening in (.9, .7, .45, .3, .1, 0.)]
    values = [reward.potential(s) for s in states]
    assert all(s.grasp_close_gate == pytest.approx(1.) for s in states)
    assert all(right > left for left, right in zip(values, values[1:]))
    assert all(not s.grasp_valid and not s.attempt_lifted and s.outcome == "running" for s in states)
    assert values[-1] < 2.6  # A confirmed grasp remains more valuable.


def test_partial_close_gate_is_continuous_at_both_distance_endpoints():
    reward = PickPlaceReward(PickPlaceRewardConfig())
    for opening in (.9, .45, 0.):
        for boundary in (.02, .008):
            outside = reward.potential(snapshot(distance=boundary+1e-7, opening=opening))
            inside = reward.potential(snapshot(distance=boundary-1e-7, opening=opening))
            assert abs(inside-outside) < 1e-4


def test_measured_dual_contact_hands_off_smoothly_but_one_finger_does_not_certify():
    cfg = PickPlaceRewardConfig()
    reward = PickPlaceReward(cfg)
    tracker = TaskStateTracker(cfg)
    start = tracker.reset(measurement(opening=.2))
    first = tracker.update(measurement(.05, opening=.2, left_contact=True))
    assert not first.grasp_valid and first.contact_score == 0
    partial = tracker.update(measurement(.10, opening=.2, left_contact=True, right_contact=True, contacts_simultaneous=True))
    confirmed = tracker.update(measurement(.15, opening=.2, left_contact=True, right_contact=True, contacts_simultaneous=True))
    assert partial.contact_score == pytest.approx(.5) and not partial.grasp_valid
    assert confirmed.grasp_valid and not confirmed.attempt_lifted
    assert reward.potential(start) < reward.potential(partial) < reward.potential(confirmed)
    assert reward.potential(confirmed) == pytest.approx(2.6)


def test_precise_reach_requires_measured_dwell_and_is_not_task_success():
    tracker = TaskStateTracker(PickPlaceRewardConfig())
    assert not tracker.reset(measurement()).pregrasp_ready
    for t in (.05, .10):
        s = tracker.update(measurement(t))
        assert not s.pregrasp_ready
    s = tracker.update(measurement(.15))
    assert s.pregrasp_ready and s.pregrasp_ready_duration_s == pytest.approx(.15)
    assert not s.grasp_valid and not s.success_event and s.outcome == "running"
    assert s.to_info()["pregrasp_ready"] is True


@pytest.mark.parametrize("changes", [
    {"distance": .021}, {"opening": .2}, {"distance": .019, "aligned": False},
    {"evidence_valid": False}, {"joint_measurement_valid": False},
    {"target_measurement_valid": False}, {"target_motion_valid": False},
    {"contact_valid": False},
])
def test_precise_reach_dwell_resets_when_geometry_or_evidence_breaks(changes):
    tracker = TaskStateTracker(PickPlaceRewardConfig())
    tracker.reset(measurement())
    tracker.update(measurement(.05))
    tracker.update(measurement(.10))
    interrupted = tracker.update(measurement(.15, **changes))
    assert not interrupted.pregrasp_ready and interrupted.pregrasp_ready_duration_s == 0
    resumed = tracker.update(measurement(.20))
    assert not resumed.pregrasp_ready and resumed.pregrasp_ready_duration_s == pytest.approx(.05)


def test_precise_reach_does_not_count_republished_timestamps_or_unknown_gaps():
    tracker = TaskStateTracker(PickPlaceRewardConfig())
    tracker.reset(measurement())
    tracker.update(measurement(.05))
    s = tracker.update(measurement(.10, evidence_sim_time_s=.05))
    assert s.pregrasp_ready_duration_s == pytest.approx(.05)
    assert not s.pregrasp_ready
    s = tracker.update(measurement(.50))
    assert s.pregrasp_ready_duration_s == 0 and not s.pregrasp_ready


def test_geometry_info_and_critic_schema_expose_pregrasp_without_changing_actor():
    s = snapshot()
    info = s.to_info()
    assert TASK_CONTEXT_SCHEMA_VERSION == 2 and TASK_CONTEXT_DIM == 93
    assert encode_task_context(s).shape == (93,)
    for field in ("goal_center_world_m", "goal_half_extents_xy_m", "goal_yaw_rad",
                  "target_half_extents_m", "source_support_z_m"):
        assert info[field] == getattr(s.measurements, field)


@pytest.mark.parametrize("distance", [0., .005])
@pytest.mark.parametrize("rotation", [
    np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]),
    np.eye(3),
], ids=["sideways", "upside_down"])
def test_wrong_world_orientation_cannot_pass_at_center_or_five_millimetres(distance, rotation):
    cfg = PickPlaceRewardConfig()
    tracker, reward = TaskStateTracker(cfg), PickPlaceReward(cfg)
    options = {"distance": distance, "grasp_rotation_world": rotation}
    opened = tracker.reset(measurement(**options))
    for t in (.05, .10, .15, .20):
        state = tracker.update(measurement(t, **options))
        assert not state.pregrasp_ready and state.pregrasp_ready_duration_s == 0
        assert state.grasp_close_gate == 0 and state.approach_orientation_score < 1
    closed = snapshot(opening=0., **options)
    assert reward.potential(closed) < reward.potential(opened)


def test_desired_world_axis_is_explicit_normalized_and_affects_certification():
    down_cfg = PickPlaceRewardConfig()
    up_cfg = PickPlaceRewardConfig(desired_grasp_axis_world=(0., 0., 2.))
    assert up_cfg.desired_grasp_axis_world == (0., 0., 1.)
    assert up_cfg.config_hash != down_cfg.config_hash
    tracker = TaskStateTracker(up_cfg)
    tracker.reset(measurement(distance=0., grasp_rotation_world=np.eye(3)))
    for t in (.05, .10, .15):
        state = tracker.update(measurement(t, distance=0., grasp_rotation_world=np.eye(3)))
    assert state.pregrasp_ready and state.grasp_close_gate == 1
    assert not state.grasp_valid


def test_home_guidance_improves_for_any_joint_even_far_outside_previous_cutoff():
    cfg = PickPlaceRewardConfig()
    home = np.asarray(cfg.home_joint_positions_rad)
    errors = np.array([1., 1.2, 1.4, 1.6, 1.8, 2.])
    original = snapshot(target=False, joint_positions_rad=home+errors)
    assert 0 < original.home_score < 1 and not original.robot_at_home
    reward = PickPlaceReward(cfg)
    for joint in range(6):
        closer = errors.copy()
        closer[joint] -= .1
        improved = snapshot(target=False, joint_positions_rad=home+closer)
        assert improved.home_score > original.home_score
        assert reward.potential(improved) > reward.potential(original)
        assert not improved.robot_at_home
    inside = snapshot(target=False, joint_positions_rad=home+cfg.home_joint_tolerance_rad/2)
    assert inside.home_score == 1 and inside.robot_at_home
    moving = snapshot(target=False, joint_positions_rad=home,
                      joint_velocities_rad_s=(.1,) * 6)
    assert moving.home_score == 1 and not moving.robot_static


def test_certified_transport_has_guidance_beyond_the_previous_distance_cutoff():
    cfg = PickPlaceRewardConfig()
    tracker = TaskStateTracker(cfg)
    goal = (.60, 0., .753)
    tracker.reset(measurement(goal_center_world_m=goal, opening=.2))

    def contact(t, x=0., z=.765):
        return measurement(t, opening=.2, target_position_world_m=(x, 0., z),
                           grasp_position_world_m=(x, 0., z), goal_center_world_m=goal,
                           left_contact=True, right_contact=True, contacts_simultaneous=True)

    tracker.update(contact(.05))
    tracker.update(contact(.10))
    tracker.update(contact(.15, z=.805))
    far = tracker.update(contact(.20, z=.805))
    closer = tracker.update(contact(.25, x=.05, z=.805))
    assert far.phase is closer.phase is TaskPhase.TRANSPORT
    assert far.grasp_valid and far.attempt_lifted
    assert 0 < far.transport_score < closer.transport_score < 1


def test_support_height_guidance_does_not_go_flat_above_five_centimetres():
    far = snapshot(target_position_world_m=(0., 0., .968))
    closer = snapshot(target_position_world_m=(0., 0., .868))
    assert 0 < far.support_score < closer.support_score < 1
    assert not far.target_supported and not closer.target_supported


def test_pregrasp_shaping_still_telescopes_on_an_open_close_loop():
    cfg = PickPlaceRewardConfig()
    tracker, reward = TaskStateTracker(cfg), PickPlaceReward(cfg)
    initial = tracker.reset(measurement(distance=.03))
    reward.reset(initial)
    previous, discounted = initial, 0.
    sequence = [(0.005, .6), (.005, .1), (.005, .6), (.03, .6)]
    for step, (distance, opening) in enumerate(sequence):
        current = tracker.update(measurement((step+1)*.05, distance=distance, opening=opening))
        result = reward.step(PickPlaceRewardInput(previous, current, .05))
        discounted += cfg.gamma**step * result.components["potential_shaping"]
        previous = current
    assert discounted == pytest.approx(-reward.potential(initial)+cfg.gamma**len(sequence)*reward.potential(previous))
    assert discounted <= 0


@pytest.mark.parametrize("changes", [
    {"reward_schema_version": 2}, {"approach_fine_distance_scale_m": 0.},
    {"approach_fine_distance_scale_m": .3}, {"pregrasp_distance_tolerance_m": .05},
    {"grasp_close_full_distance_m": .02}, {"pregrasp_confirm_s": 0.},
    {"pregrasp_orientation_tolerance_rad": math.pi+.1}, {"approach_opening_min_rad": 0.},
    {"desired_grasp_axis_world": (0., 0., 0.)},
    {"desired_grasp_axis_world": (0., 0., math.nan)},
    {"desired_grasp_axis_world": (0., -1.)},
])
def test_invalid_pregrasp_configuration_rejected(changes):
    with pytest.raises(ValueError):
        PickPlaceRewardConfig(**changes)
