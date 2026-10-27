"""Measured stage boundaries must not fabricate reward terminals or success."""
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.rewards import PickPlaceRewardConfig, TaskMeasurements, TaskStateTracker
from so101_capstone.rewards.training_stages import StageEvaluator, StageTrainingConfig


class StageTrace:
    def __init__(self, stage="S1", *, config=None, reward_config=None):
        self.reward = reward_config or PickPlaceRewardConfig()
        self.config = config or StageTrainingConfig()
        self.stage = stage
        self.time = 0.0
        self.tracker = TaskStateTracker(self.reward)
        self.evaluator = StageEvaluator(stage, self.config, self.reward)
        self.snapshot = self.tracker.reset(self.measurement(),
                                           "idle_empty" if stage == "S0" else "pick_place",
                                           task_instance_id="stages")
        self.info = self.evaluator.reset(self.snapshot)

    def measurement(self, **overrides):
        values = dict(
            sim_time_s=self.time,
            joint_positions_rad=self.reward.home_joint_positions_rad,
            joint_velocities_rad_s=(0.0,) * 6,
            grasp_position_world_m=(0, 0, 1.1),
            grasp_rotation_world=np.diag((1.0, -1.0, -1.0)),
            target_name="cube" if self.stage == "S1" else None,
            target_position_world_m=(0, 0, .765) if self.stage == "S1" else None,
            target_motion_valid=self.stage == "S1", contact_valid=True,
            goal_center_world_m=(.25, 0, .753),
        )
        values.update(overrides)
        return TaskMeasurements(**values)

    def step(self, *, dt=.05, reach=False, **overrides):
        self.time += dt
        if reach:
            q = list(self.reward.home_joint_positions_rad)
            q[-1] += .7
            values = dict(grasp_position_world_m=(0, 0, .775), joint_positions_rad=q)
            values.update(overrides)
            overrides = values
        self.snapshot = self.tracker.update(self.measurement(**overrides))
        self.info = self.evaluator.update(self.snapshot)
        return self.info


def test_config_roundtrip_default_file_and_frozen(tmp_path):
    cfg = StageTrainingConfig()
    path = tmp_path / "stages.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert StageTrainingConfig.load(path) == cfg
    assert StageTrainingConfig.load().config_hash == cfg.config_hash
    project_config = Path(__file__).resolve().parents[2] / "config" / "act_ppo_stages.json"
    assert StageTrainingConfig.load(project_config) == cfg
    with pytest.raises(FrozenInstanceError):
        cfg.s0_budget_s = 10


@pytest.mark.parametrize("values", [
    {"stage_schema_version": 2}, {"stage_schema_version": True},
    {"s0_budget_s": 0}, {"s0_budget_s": 2}, {"s1_budget_s": .1},
    {"s1_reach_xy_m": .03}, {"s1_reach_dz_min_m": -.01},
    {"s1_reach_dz_min_m": .03}, {"s1_opening_min_rad": 1.0},
    {"s1_orientation_tolerance_rad": math.inf}, {"s1_reach_hold_s": True},
    {"calibration_status": ""}, {"unknown": 1},
])
def test_config_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        StageTrainingConfig.from_dict(values)


def test_config_rejects_nonobject_json(tmp_path):
    path = tmp_path / "stages.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        StageTrainingConfig.load(path)


def test_only_implemented_stages_are_accepted():
    with pytest.raises(ValueError, match="S0 or S1"):
        StageEvaluator("S2", StageTrainingConfig(), PickPlaceRewardConfig())


def test_s0_runs_entire_budget_and_requires_trailing_home_hold():
    trace = StageTrace("S0")
    for _ in range(99):
        info = trace.step()
        assert info["stage_result"] == "RUNNING"
    info = trace.step()
    assert info["stage_elapsed_s"] == pytest.approx(5)
    assert info["stage_result"] == "SUCCESS"
    assert info["stage_success"] and info["stage_cut"]
    assert not info["full_unassisted_success"]
    assert trace.snapshot.outcome == "running"
    assert not trace.snapshot.success_event
    with pytest.raises(RuntimeError, match="closed"):
        trace.step()


def test_s0_earlier_home_dwell_does_not_compensate_for_late_motion():
    trace = StageTrace("S0")
    for _ in range(42):
        trace.step()
    trace.step(joint_velocities_rad_s=(.1, 0, 0, 0, 0, 0))
    for _ in range(57):
        info = trace.step()
    assert info["stage_elapsed_s"] == pytest.approx(5)
    assert info["stage_home_hold_s"] < 3
    assert info["stage_timeout"] and info["stage_cut"]
    assert not info["stage_success"]


def test_home_thresholds_come_from_reward_config_including_gripper():
    cfg = replace(PickPlaceRewardConfig(), home_joint_tolerance_rad=.003,
                  home_velocity_tolerance_rad_s=.004)
    trace = StageTrace("S0", reward_config=cfg)
    q = list(cfg.home_joint_positions_rad)
    q[-1] += .004
    initial = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements,
                                                          joint_positions_rad=q))
    with pytest.raises(ValueError, match="home_position_valid"):
        trace.evaluator.reset(initial)
    moving = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements,
                                                         joint_velocities_rad_s=(.005, 0, 0, 0, 0, 0)))
    with pytest.raises(ValueError, match="home_velocity_valid"):
        trace.evaluator.reset(moving)


@pytest.mark.parametrize("overrides", [
    {"target_linear_velocity_m_s": (.1, 0, 0)},
    {"target_angular_velocity_rad_s": (0, 0, 1)},
    {"target_position_world_m": (0, 0, .80)},
    {"target_position_world_m": (.08, 0, .765)},
    {"left_contact": True},
])
def test_s1_reset_rejects_unsettled_unsupported_displaced_or_gripped_target(overrides):
    trace = StageTrace()
    initial = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements, **overrides))
    with pytest.raises(ValueError, match="initial state"):
        trace.evaluator.reset(initial)


def test_s1_reset_rejects_old_grasp_history_and_missing_source_anchor():
    trace = StageTrace()
    with pytest.raises(ValueError, match="no_prior_grasp"):
        trace.evaluator.reset(replace(trace.snapshot, lifted_once=True))
    with pytest.raises(ValueError, match="target_at_source"):
        trace.evaluator.reset(replace(trace.snapshot, source_position_xy_m=None))


def test_s1_requires_all_reach_conditions_over_same_fresh_interval():
    trace = StageTrace()
    for _ in range(3):
        info = trace.step(reach=True)
        assert info["stage_result"] == "RUNNING"
    info = trace.step(reach=True)
    assert info["stage_result"] == "SUCCESS"
    assert info["stage_reach_hold_s"] == pytest.approx(.15)
    assert info["stage_result_reason"] == "reach_hold"
    assert info["stage_cut"] and not info["full_unassisted_success"]
    assert trace.snapshot.outcome == "running"


@pytest.mark.parametrize("overrides", [
    {"grasp_position_world_m": (0, 0, .79)},
    {"grasp_position_world_m": (.012, 0, .77)},
    {"grasp_position_world_m": (0, 0, .76)},
    {"grasp_rotation_world": np.eye(3)},
    {"target_linear_velocity_m_s": (.04, 0, 0)},
    {"target_position_world_m": (0, 0, .78), "grasp_position_world_m": (0, 0, .79)},
    {"left_contact": True},
])
def test_reach_proximity_alone_cannot_certify_stage(overrides):
    trace = StageTrace()
    for _ in range(5):
        info = trace.step(reach=True, **overrides)
        assert not info["stage_reach_candidate"]
        assert not info["stage_success"]
        assert info["stage_reach_hold_s"] == 0


def test_wrong_opening_resets_reach_dwell_and_does_not_accumulate_disjoint_windows():
    trace = StageTrace()
    trace.step(reach=True)
    trace.step(reach=True)
    assert trace.info["stage_reach_hold_s"] == pytest.approx(.05)
    trace.step(reach=True, joint_positions_rad=trace.reward.home_joint_positions_rad)
    assert trace.info["stage_reach_hold_s"] == 0
    trace.step(reach=True)
    assert trace.info["stage_reach_hold_s"] == 0
    trace.step(reach=True)
    trace.step(reach=True)
    assert not trace.info["stage_success"]
    assert trace.step(reach=True)["stage_success"]


def test_duplicate_evidence_never_advances_dwell():
    trace = StageTrace()
    trace.step(reach=True)
    trace.step(reach=True)
    before = trace.info["stage_reach_hold_s"]
    info = trace.step(reach=True, evidence_sim_time_s=.1)
    assert not info["stage_evidence_fresh"]
    assert info["stage_reach_hold_s"] == before
    assert not info["stage_success"]


def test_evidence_gap_restarts_dwell_instead_of_covering_missing_time():
    trace = StageTrace()
    trace.step(reach=True)
    trace.step(reach=True)
    info = trace.step(dt=.2, reach=True)
    assert info["stage_reach_hold_s"] == 0
    assert not info["stage_success"]
    for _ in range(2):
        assert not trace.step(reach=True)["stage_success"]
    assert trace.step(reach=True)["stage_success"]


@pytest.mark.parametrize("flag", ["evidence_valid", "joint_measurement_valid",
                                 "target_measurement_valid", "target_motion_valid", "contact_valid"])
def test_invalid_measurements_raise_instead_of_reporting_policy_failure(flag):
    trace = StageTrace()
    with pytest.raises(ValueError, match="invalid stage sensor evidence"):
        trace.step(**{flag: False})


def test_stale_valid_flags_cannot_certify_stage():
    trace = StageTrace()
    with pytest.raises(ValueError, match="stale"):
        trace.step(dt=.2, reach=True, evidence_sim_time_s=0)


def test_timeout_is_external_cut_and_never_full_task_failure():
    trace = StageTrace(config=replace(StageTrainingConfig(), s1_budget_s=.2))
    for _ in range(4):
        info = trace.step()
    assert info["stage_result"] == "TIMEOUT"
    assert info["stage_cut"] and not info["stage_success"]
    assert trace.snapshot.outcome == "running"
    assert not trace.snapshot.failure_reason


def test_failure_overrides_milestone_and_preserves_original_terminal_semantics():
    trace = StageTrace()
    for _ in range(3):
        trace.step(reach=True)
    info = trace.step(reach=True, failure_reason="collision")
    assert info["stage_result"] == "FAILURE"
    assert not info["stage_cut"]
    assert not info["stage_success"] and not info["stage_timeout"]
    assert info["stage_result_reason"] == "collision"
    assert trace.snapshot.outcome == "task_failure"


def test_real_current_grasp_and_lift_can_pass_without_open_jaw_reach_dwell():
    trace = StageTrace()
    q = list(trace.reward.home_joint_positions_rad)
    q[-1] += .27
    for _ in range(3):
        info = trace.step(grasp_position_world_m=(0, 0, .765), joint_positions_rad=q,
                          left_contact=True, right_contact=True, contacts_simultaneous=True)
    assert not info["stage_success"]
    for _ in range(2):
        info = trace.step(target_position_world_m=(0, 0, .805), grasp_position_world_m=(0, 0, .805),
                          joint_positions_rad=q, left_contact=True, right_contact=True,
                          contacts_simultaneous=True)
    assert info["stage_success"]
    assert info["stage_result_reason"] == "grasp_lift"
    assert info["stage_downstream_grasp_lift"]
    assert info["stage_reach_hold_s"] == 0
    assert not info["full_unassisted_success"]


def test_historical_lift_flag_without_current_grasp_is_not_success():
    trace = StageTrace()
    trace.time = .05
    snapshot = trace.tracker.update(trace.measurement())
    snapshot = replace(snapshot, lifted_once=True, attempt_lifted=True,
                       contact_duration_s=.2, lift_duration_s=.2)
    info = trace.evaluator.update(snapshot)
    assert not info["stage_success"]
    assert not info["stage_downstream_grasp_lift"]


def test_start_certificate_is_present_in_every_info_and_cannot_be_mutated_by_caller():
    trace = StageTrace()
    assert trace.info["start_valid"]
    assert trace.info["start_certificate"]["start_valid"]
    assert not trace.info["assisted_initialization"]
    trace.info["start_certificate"]["checks"]["home_position_valid"] = False
    assert trace.step()["start_certificate"]["checks"]["home_position_valid"] is True


def test_whole_segment_joint_statistics_retain_earlier_motion_on_all_six_axes():
    trace = StageTrace("S0")
    errors = np.arange(1, 7) * .01
    velocities = np.arange(1, 7) * -.1
    q = np.asarray(trace.reward.home_joint_positions_rad) + errors
    trace.step(joint_positions_rad=q, joint_velocities_rad_s=velocities)
    info = trace.step()
    assert info["stage_home_error_rad"] == 0
    assert info["stage_measurement_sample_count"] == 3  # Initial + two fresh samples.
    metrics = info["stage_metrics"]
    np.testing.assert_allclose(metrics["stage_home_error_max_per_joint_rad"], errors)
    np.testing.assert_allclose(metrics["stage_home_error_sample_rms_per_joint_rad"], errors / np.sqrt(3))
    np.testing.assert_allclose(metrics["stage_joint_velocity_max_per_joint_rad_s"], np.abs(velocities))
    np.testing.assert_allclose(metrics["stage_joint_velocity_sample_rms_per_joint_rad_s"], np.abs(velocities) / np.sqrt(3))
    assert info["stage_home_static_fraction"] == 0  # Neither interval has two stable endpoints.


def test_reused_evidence_is_excluded_from_samples_and_cannot_be_recredited_as_stable_time():
    trace = StageTrace("S0")
    trace.step()
    info = trace.step(evidence_sim_time_s=.05)
    assert info["stage_measurement_sample_count"] == 2
    assert info["stage_home_static_evidence_s"] == pytest.approx(.05)
    assert info["stage_home_static_fraction"] == pytest.approx(.5)
    info = trace.step()
    assert info["stage_measurement_sample_count"] == 3
    # The fresh .15 stamp may not retroactively fill the reused .05-.10 step.
    assert info["stage_home_static_evidence_s"] == pytest.approx(.1)
    assert info["stage_home_static_fraction"] == pytest.approx(2 / 3)
    assert info["stage_evidence_coverage_fraction"] == pytest.approx(2 / 3)


def test_gap_endpoints_count_as_measurements_but_missing_time_never_counts_as_stability():
    trace = StageTrace("S0")
    trace.step()
    info = trace.step(dt=.2)
    assert info["stage_measurement_sample_count"] == 3
    assert info["stage_fresh_evidence_s"] == pytest.approx(.05)
    assert info["stage_home_static_fraction"] == pytest.approx(.2)
    info = trace.step()
    assert info["stage_fresh_evidence_s"] == pytest.approx(.1)
    assert info["stage_home_static_fraction"] == pytest.approx(1 / 3)


def test_fresh_stable_whole_window_has_full_coverage():
    trace = StageTrace("S0")
    for _ in range(100):
        info = trace.step()
    assert info["stage_home_static_evidence_s"] == pytest.approx(5)
    assert info["stage_home_static_fraction"] == pytest.approx(1)
    assert info["stage_evidence_coverage_fraction"] == pytest.approx(1)
    assert info["stage_success"]  # Statistics leave the previous success criterion unchanged.
    assert info["stage_min_reach_distance_m"] is None
    assert info["stage_min_reach_xy_m"] is None
    json.dumps(info, allow_nan=False)


def test_reach_window_statistics_preserve_closest_distance_and_candidate_time():
    trace = StageTrace()
    initial = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements,
                       grasp_position_world_m=(.1, .05, 1.1)))
    trace.evaluator.reset(initial)
    close = (.008, 0, .775)
    trace.step(reach=True, grasp_position_world_m=close)
    trace.step(reach=True, grasp_position_world_m=close)
    info = trace.step(grasp_position_world_m=(.1, .05, 1.1))
    assert info["stage_min_reach_distance_m"] == pytest.approx(np.hypot(.008, .01))
    assert info["stage_min_reach_xy_m"] == pytest.approx(.008)
    assert info["stage_reach_candidate_evidence_s"] == pytest.approx(.05)
    assert info["stage_reach_hold_s"] == 0
    assert not info["stage_success"]


def test_statistics_reset_between_segments_and_do_not_expose_internal_arrays():
    trace = StageTrace("S0")
    info = trace.step(joint_velocities_rad_s=(1, 0, 0, 0, 0, 0))
    info["stage_metrics"]["stage_joint_velocity_max_per_joint_rad_s"][0] = 200
    assert trace.step()["stage_joint_velocity_max_per_joint_rad_s"][0] == 1
    info = trace.evaluator.reset(trace.snapshot)
    assert info["stage_measurement_sample_count"] == 1
    assert info["stage_joint_velocity_max_per_joint_rad_s"] == [0.] * 6
    assert info["stage_fresh_evidence_s"] == 0
    assert info["stage_home_static_fraction"] == 0
