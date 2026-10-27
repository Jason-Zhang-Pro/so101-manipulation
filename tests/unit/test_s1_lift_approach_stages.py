"""S1 lift/approach phases share one clock and require measured progress."""
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards.s1_lift_approach import S1LiftApproachConfig, S1LiftApproachTracker
from so101_capstone.rewards.stage_process_reward import StageProcessRewardConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_training_stages import StageTrace


def stages(**overrides):
    process = StageProcessRewardConfig(
        schema_version=4, s1_shaping_scale=.1, s1_distance_cost_per_s=.4,
        s1_best_distance_bonus=8., s1_distance_band_edges_m=(.08, .02),
        s1_distance_band_bonuses=(48.,), s1_lift_approach=S1LiftApproachConfig(),
    )
    return StageTrainingConfig(process_reward=process, **overrides)


class LiftTrace(StageTrace):
    def __init__(self, stage="S1", **kwargs):
        super().__init__(stage, config=kwargs.pop("config", stages()), **kwargs)

    def measurement(self, **overrides):
        # Valid HOME joints but a low grasp reference point: geometry is
        # injected here so boundary tests do not depend on FK calibration.
        return super().measurement(**{"grasp_position_world_m": (.1, 0., .78), **overrides})

    def lifted(self, **overrides):
        return self.step(**{"grasp_position_world_m": (.1, 0., .83), **overrides})

    def reached(self, **overrides):
        q = list(self.reward.home_joint_positions_rad)
        q[4] += self.config.process_reward.s1_lift_approach.roll_target_rad
        q[-1] += .7
        return self.step(**{"grasp_position_world_m": (0., 0., .775),
                            "joint_positions_rad": q, **overrides})

    def finish_lift(self):
        for _ in range(3):
            self.lifted()
        assert self.info["stage_s1_phase"] == "APPROACH"


def test_configuration_roundtrips_and_defaults_have_signed_clockwise_rotation():
    cfg = S1LiftApproachConfig()
    assert cfg.roll_target_rad == pytest.approx(-math.pi / 2)
    assert cfg.lift_clearance_m == .08
    assert S1LiftApproachConfig.from_dict(cfg.to_dict()) == cfg
    resolved = stages()
    assert StageTrainingConfig.from_dict(resolved.to_dict()) == resolved


@pytest.mark.parametrize("values", [
    {"lift_clearance_m": 0}, {"lift_hold_s": True}, {"lift_band_step_m": .1},
    {"roll_target_rad": 0}, {"roll_target_rad": math.inf}, {"roll_target_rad": 4},
    {"roll_tolerance_rad": math.pi / 2}, {"roll_band_step_rad": math.pi},
    {"approach_inner_xy_m": .06}, {"approach_final_dz_m": .1},
    {"corridor_tolerance_m": .1}, {"unexpected": 1},
])
def test_configuration_rejects_invalid_ranges(values):
    with pytest.raises(ValueError):
        S1LiftApproachConfig.from_dict(values)


def test_configuration_rejects_nonobject_and_incompatible_stage_budget_or_dz():
    with pytest.raises(ValueError, match="JSON object"):
        S1LiftApproachConfig.from_dict([])
    with pytest.raises(ValueError, match="single S1 budget"):
        stages(s1_budget_s=.2)
    with pytest.raises(ValueError, match="final approach dz"):
        stages(s1_reach_dz_min_m=.015)


def test_low_initial_point_starts_lift_without_success_or_clock_reset():
    trace = LiftTrace()
    assert trace.info["stage_s1_phase"] == "LIFT"
    assert not trace.info["stage_s1_lift_completed"]
    assert trace.info["stage_s1_lift_hold_s"] == 0
    assert trace.info["stage_s1_clearance_m"] == pytest.approx(.03)
    assert not trace.info["stage_reach_candidate"]
    trace.finish_lift()
    assert trace.info["stage_elapsed_s"] == pytest.approx(.15)
    assert trace.info["stage_s1_lift_hold_s"] == pytest.approx(.1)
    assert not trace.info["stage_cut"]


def test_height_threshold_uses_measured_table_plane_and_exact_eight_centimetres():
    trace = LiftTrace()
    for _ in range(5):
        info = trace.step(grasp_position_world_m=(.1, 0., .829))
        assert info["stage_s1_phase"] == "LIFT"
        assert info["stage_s1_lift_hold_s"] == 0
    trace.finish_lift()
    tracker = S1LiftApproachTracker(S1LiftApproachConfig(), trace.reward)
    shifted = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements,
        source_support_z_m=1., target_position_world_m=(0., 0., 1.015),
        grasp_position_world_m=(.1, 0., 1.08)))
    data = tracker.reset(shifted, trace.info)
    assert data["stage_s1_clearance_m"] == pytest.approx(.08)
    assert data["stage_s1_required_clearance_m"] == pytest.approx(.08)


def test_initially_high_point_still_requires_fresh_measured_hold():
    trace = LiftTrace()
    initial = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements,
        grasp_position_world_m=(.1, 0., .83)))
    trace.info = trace.evaluator.reset(initial)
    assert not trace.info["stage_s1_lift_completed"]
    assert not trace.lifted()["stage_s1_lift_completed"]
    assert trace.lifted()["stage_s1_lift_completed"]


def test_duplicate_evidence_does_not_complete_lift_or_credit_missing_interval():
    trace = LiftTrace()
    trace.lifted()
    trace.lifted()
    assert trace.info["stage_s1_lift_hold_s"] == pytest.approx(.05)
    info = trace.lifted(evidence_sim_time_s=.1)
    assert info["stage_s1_lift_hold_s"] == pytest.approx(.05)
    assert not info["stage_s1_lift_completed"]
    # The next fresh endpoint covers only its latest control interval.
    assert trace.lifted()["stage_s1_lift_hold_s"] == pytest.approx(.1)


def test_long_gap_and_height_regression_restart_lift_hold():
    trace = LiftTrace()
    trace.lifted()
    trace.lifted()
    info = trace.lifted(dt=.2)
    assert info["stage_s1_lift_hold_s"] == 0
    assert not info["stage_s1_lift_completed"]
    trace.lifted()
    assert trace.step()["stage_s1_lift_hold_s"] == 0
    trace.lifted()
    assert trace.info["stage_s1_lift_hold_s"] == 0


@pytest.mark.parametrize("overrides", [
    {"left_contact": True}, {"target_linear_velocity_m_s": (.05, 0., 0.)},
    {"target_position_world_m": (.1, 0., .765)},
    {"target_position_world_m": (0., 0., .79)},
])
def test_lift_requires_original_contact_free_supported_source_conditions(overrides):
    trace = LiftTrace()
    for _ in range(4):
        info = trace.lifted(**overrides)
        assert not info["stage_s1_lift_completed"]
        assert info["stage_s1_lift_hold_s"] == 0


def test_direct_reach_without_lift_cannot_pass_even_with_correct_roll():
    trace = LiftTrace()
    for _ in range(6):
        info = trace.reached()
        assert not info["stage_reach_candidate"]
        assert not info["stage_success"]
        assert info["stage_reach_hold_s"] == 0


def test_completed_lift_is_latched_but_does_not_disable_corridor():
    trace = LiftTrace()
    trace.finish_lift()
    info = trace.step(grasp_position_world_m=(.1, 0., .775))
    assert info["stage_s1_lift_completed"]
    assert info["stage_s1_phase"] == "APPROACH"
    assert not info["stage_s1_corridor_safe"]
    assert info["stage_s1_corridor_deficit_m"] == pytest.approx(.055)
    trace.reached()
    assert trace.info["stage_s1_lift_completed"]


def test_roll_error_is_relative_to_home_signed_and_not_wrapped():
    trace = LiftTrace()
    cfg = trace.config.process_reward.s1_lift_approach
    assert trace.info["stage_s1_roll_error_rad"] == pytest.approx(math.pi / 2)
    q = list(trace.reward.home_joint_positions_rad)
    q[4] -= cfg.roll_target_rad
    assert trace.step(joint_positions_rad=q)["stage_s1_roll_error_rad"] == pytest.approx(math.pi)
    q[4] = trace.reward.home_joint_positions_rad[4] + cfg.roll_target_rad + 2 * math.pi
    assert trace.step(joint_positions_rad=q)["stage_s1_roll_error_rad"] == pytest.approx(2 * math.pi)
    custom = replace(trace.reward, home_joint_positions_rad=(0., 0., 0., 0., .2, 0.))
    tracker = S1LiftApproachTracker(cfg, custom)
    snapshot = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements,
        joint_positions_rad=(0., 0., 0., 0., .2 + cfg.roll_target_rad, 0.)))
    assert tracker.reset(snapshot, trace.info)["stage_s1_roll_error_rad"] == pytest.approx(0)


def test_roll_tolerance_is_required_throughout_reach_dwell():
    trace = LiftTrace()
    trace.finish_lift()
    q = list(trace.reward.home_joint_positions_rad)
    q[-1] += .7
    for _ in range(4):
        assert not trace.reached(joint_positions_rad=q)["stage_reach_candidate"]
    trace.reached()
    trace.reached()
    assert trace.info["stage_reach_hold_s"] == pytest.approx(.05)
    assert trace.reached(joint_positions_rad=q)["stage_reach_hold_s"] == 0
    for _ in range(3):
        assert not trace.reached()["stage_success"]
    assert trace.reached()["stage_success"]
    assert trace.info["stage_result_reason"] == "reach_hold"


@pytest.mark.parametrize("xy,required", [(.07, .08), (.06, .08), (.035, .0525), (.01, .025), (0., .025)])
def test_approach_corridor_descends_continuously_over_target(xy, required):
    trace = LiftTrace()
    info = trace.step(grasp_position_world_m=(xy, 0., .75 + required))
    assert info["stage_s1_required_clearance_m"] == pytest.approx(required)
    assert info["stage_s1_corridor_safe"]
    assert info["stage_s1_corridor_deficit_m"] == pytest.approx(0)


def test_parallel_metric_uses_grasp_local_x_world_vertical_component():
    trace = LiftTrace()
    assert trace.info["stage_s1_parallel_error_rad"] == pytest.approx(0)
    # A proper rotation whose local X points directly upward.
    matrix = np.asarray(((0., 0., -1.), (0., 1., 0.), (1., 0., 0.)))
    info = trace.step(grasp_rotation_world=matrix)
    assert info["stage_s1_parallel_error_rad"] == pytest.approx(math.pi / 2)


def test_physical_downstream_success_cannot_bypass_lift_latch():
    trace = LiftTrace()
    trace.reached()
    snapshot = replace(trace.snapshot, grasp_valid=True, grasp_attempt_active=True,
        attempt_lifted=True, contact_duration_s=1., lift_duration_s=1.,
        grasp_origin_bottom_z_m=.75, measurements=replace(trace.snapshot.measurements,
            target_position_world_m=(0., 0., .82), grasp_position_world_m=(0., 0., .83),
            left_contact=True, right_contact=True, contacts_simultaneous=True))
    metrics = trace.evaluator._metrics(snapshot)
    metrics.update(trace.evaluator._s1_lift_approach.metrics(snapshot, metrics))
    assert not trace.evaluator._strong_downstream(snapshot, metrics)


def test_s1_times_out_at_ten_seconds_without_restarting_at_phase_transition():
    trace = LiftTrace()
    for _ in range(100):
        assert not trace.step()["stage_cut"]
    trace.finish_lift()
    while trace.time < 10 - 1e-9:
        trace.lifted()
    assert trace.info["stage_elapsed_s"] == pytest.approx(10)
    assert trace.info["stage_s1_lift_completed"]
    assert trace.info["stage_result"] == "TIMEOUT"
    assert trace.snapshot.outcome == "running"


def test_s0_uses_original_five_second_evaluation_and_no_s1_subphase():
    original, new = StageTrace("S0"), LiftTrace("S0")
    for _ in range(100):
        previous, current = original.step(), new.step()
        assert current == previous
    assert current["stage_elapsed_s"] == pytest.approx(5)
    assert current["stage_result"] == "SUCCESS"
    assert not any(key.startswith("stage_s1_") for key in current)
