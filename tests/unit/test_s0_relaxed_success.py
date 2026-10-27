"""Relaxed S0 success is an opt-in stage contract, not a new reset/reward contract."""
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import PickPlaceRewardConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_training_stages import StageTrace


def relaxed(**overrides):
    values = dict(s0_home_joint_tolerance_rad=math.radians(8),
                  s0_home_velocity_tolerance_rad_s=.3,
                  s0_home_hold_s=1., s0_success_mode="home_hold")
    values.update(overrides)
    return StageTrainingConfig(**values)


def displaced(trace, joint, error):
    q = list(trace.reward.home_joint_positions_rad)
    q[joint] += error
    return q


def test_default_schema_and_hash_are_byte_compatible_with_legacy():
    cfg = StageTrainingConfig()
    assert cfg.config_hash == "918facc6e4d89d41747cea931d3443ec8d048688c71161950a1ba04b6a829784"
    assert not {"s0_home_joint_tolerance_rad", "s0_home_velocity_tolerance_rad_s", "s0_success_mode"} & cfg.to_dict().keys()
    assert StageTrainingConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.s0_home_hold_s == 3
    assert cfg.s0_success_mode == "window_end"


def test_new_contract_roundtrips_and_changes_resume_hash():
    cfg = relaxed()
    assert cfg.to_dict()["s0_success_mode"] == "home_hold"
    assert StageTrainingConfig.from_dict(cfg.to_dict()) == cfg
    assert len({cfg.config_hash, replace(cfg, s0_home_hold_s=2).config_hash,
                replace(cfg, s0_success_mode="window_end").config_hash,
                replace(cfg, s0_home_joint_tolerance_rad=.1).config_hash,
                replace(cfg, s0_home_velocity_tolerance_rad_s=.2).config_hash}) == 5


@pytest.mark.parametrize("name", ["s0_home_joint_tolerance_rad", "s0_home_velocity_tolerance_rad_s"])
@pytest.mark.parametrize("value", [True, False, 0, -1, math.nan, math.inf, -math.inf, "0.3"])
def test_stage_thresholds_reject_nonpositive_nonfinite_or_boolean(name, value):
    with pytest.raises(ValueError, match=name):
        relaxed(**{name: value})


@pytest.mark.parametrize("mode", [None, True, "", "early", 1])
def test_invalid_success_mode_rejected(mode):
    with pytest.raises(ValueError, match="s0_success_mode"):
        relaxed(s0_success_mode=mode)


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("sign", [-1, 1])
def test_each_joint_including_gripper_accepts_exact_position_and_speed_boundary(joint, sign):
    trace = StageTrace("S0", config=relaxed())
    q = displaced(trace, joint, sign * math.radians(8))
    velocity = [0.] * 6
    velocity[joint] = sign * .3
    for _ in range(19):
        info = trace.step(joint_positions_rad=q, joint_velocities_rad_s=velocity)
        assert not info["stage_success"]
    info = trace.step(joint_positions_rad=q, joint_velocities_rad_s=velocity)
    assert info["stage_success"] and info["stage_cut"]
    assert info["stage_elapsed_s"] == pytest.approx(1)
    assert info["stage_home_hold_s"] == pytest.approx(1)
    assert info["stage_s0_home_joint_tolerance_rad"] == pytest.approx(math.radians(8))
    assert info["stage_s0_home_velocity_tolerance_rad_s"] == .3
    assert info["stage_s0_home_hold_required_s"] == 1
    assert info["stage_s0_success_mode"] == "home_hold"
    assert trace.snapshot.outcome == "running"
    assert not trace.snapshot.success_event


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("criterion", ["position", "velocity"])
def test_any_joint_over_threshold_clears_hold_and_requires_new_continuous_second(joint, criterion):
    trace = StageTrace("S0", config=relaxed())
    for _ in range(19):
        trace.step()
    if criterion == "position":
        info = trace.step(joint_positions_rad=displaced(trace, joint, math.radians(8) + 1e-5))
    else:
        velocity = [0.] * 6
        velocity[joint] = -.30001
        info = trace.step(joint_velocities_rad_s=velocity)
    assert info["stage_home_hold_s"] == 0 and not info["stage_success"]
    for _ in range(20):
        assert not trace.step()["stage_success"]
    assert trace.step()["stage_success"]


@pytest.mark.parametrize("criterion", ["position", "velocity"])
def test_relaxed_stage_success_does_not_relax_initialization_certificate(criterion):
    trace = StageTrace("S0", config=relaxed())
    values = ({"joint_positions_rad": displaced(trace, 5, math.radians(2))}
              if criterion == "position" else {"joint_velocities_rad_s": (.1,) * 6})
    initial = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements, **values))
    with pytest.raises(ValueError, match=f"home_{criterion}_valid"):
        trace.evaluator.reset(initial)


def test_omitted_thresholds_still_use_reward_thresholds_for_early_success():
    reward = replace(PickPlaceRewardConfig(), home_joint_tolerance_rad=.01,
                     home_velocity_tolerance_rad_s=.02)
    cfg = relaxed(s0_home_joint_tolerance_rad=None, s0_home_velocity_tolerance_rad_s=None)
    trace = StageTrace("S0", config=cfg, reward_config=reward)
    info = trace.step(joint_velocities_rad_s=(.03,) * 6)
    assert not info["stage_robot_static"]
    assert info["stage_home_hold_s"] == 0
    assert info["stage_s0_home_joint_tolerance_rad"] == .01
    assert info["stage_s0_home_velocity_tolerance_rad_s"] == .02


def test_repeated_evidence_cannot_be_added_later_as_new_hold_time():
    trace = StageTrace("S0", config=relaxed())
    for _ in range(18):
        trace.step()
    info = trace.step(evidence_sim_time_s=.9)
    assert not info["stage_evidence_fresh"]
    assert info["stage_home_hold_s"] == pytest.approx(.9)
    info = trace.step()
    assert info["stage_home_hold_s"] == pytest.approx(.95)
    assert not info["stage_success"]
    assert trace.step()["stage_success"]


def test_evidence_gap_clears_early_hold_without_crediting_missing_interval():
    trace = StageTrace("S0", config=relaxed())
    for _ in range(19):
        trace.step()
    info = trace.step(dt=.2)
    assert not info["stage_evidence_fresh"]
    assert info["stage_home_hold_s"] == 0
    for _ in range(19):
        assert not trace.step()["stage_success"]
    assert trace.step()["stage_success"]


def test_invalid_evidence_cannot_certify_early_success():
    trace = StageTrace("S0", config=relaxed())
    for _ in range(19):
        trace.step()
    with pytest.raises(ValueError, match="invalid stage sensor evidence"):
        trace.step(joint_measurement_valid=False)


@pytest.mark.parametrize("last_dt, succeeds", [(.05, True), (.06, False)])
def test_success_at_deadline_allowed_but_late_measurement_cannot_retroactively_succeed(last_dt, succeeds):
    trace = StageTrace("S0", config=relaxed(s0_budget_s=1))
    for _ in range(19):
        trace.step()
    info = trace.step(dt=last_dt)
    assert info["stage_success"] is succeeds
    assert info["stage_timeout"] is not succeeds
    assert info["stage_cut"]


def test_physical_failure_has_priority_over_simultaneous_hold_completion():
    trace = StageTrace("S0", config=relaxed())
    for _ in range(19):
        trace.step()
    trace.time += .05
    snapshot = trace.tracker.update(trace.measurement())
    info = trace.evaluator.update(replace(snapshot, outcome="task_failure", failure_reason="collision"))
    assert info["stage_home_hold_s"] == pytest.approx(1)
    assert info["stage_result"] == "FAILURE"
    assert not info["stage_success"] and not info["stage_cut"]
    assert info["stage_result_reason"] == "collision"


def test_window_end_mode_preserves_full_window_even_with_relaxed_thresholds():
    trace = StageTrace("S0", config=relaxed(s0_success_mode="window_end"))
    for _ in range(99):
        assert not trace.step()["stage_success"]
    assert trace.step()["stage_success"]


def test_s1_observations_and_success_are_unchanged_by_s0_settings():
    baseline = StageTrace("S1")
    adjusted = StageTrace("S1", config=relaxed())
    assert baseline.info == adjusted.info
    for _ in range(4):
        assert baseline.step(reach=True) == adjusted.step(reach=True)
    assert adjusted.info["stage_success"]
    assert "stage_s0_success_mode" not in adjusted.info


def test_early_success_is_external_environment_truncation_not_task_terminal():
    from so101_capstone.environments.reach_env_core import ReachEnvironmentConfig, So101ReachEnvironmentCore
    from test_task_measurements import joint, contacts, GOAL

    core = So101ReachEnvironmentCore(ReachEnvironmentConfig(), PickPlaceRewardConfig(),
        goal_region_center_world_m=GOAL, reward_profile="pick_place_v2", task_mode="idle_empty")
    core.configure_training_stage("S0", relaxed())
    stream = contacts(*np.arange(31) / 30)
    core.reset_from_feedback(joint(0), now_monotonic_s=0, step_end_sim_time_s=0, contact_cache=stream)
    for step in range(1, 31):
        _, _, terminated, truncated, info = core.step_from_feedback(
            joint(step / 30, step + 1), now_monotonic_s=0, step_end_sim_time_s=step / 30,
            control_dt_s=1 / 30, contact_cache=stream)
        assert not terminated
        assert truncated is (step == 30)
    assert info["stage_success"] and info["stage_elapsed_s"] == pytest.approx(1)
    assert info["task_outcome"] == "running"
    assert not info["success"] and not info["full_unassisted_success"]
    assert info["reward_components"]["success"] == 0
    assert info["reward_components"]["failure"] == 0
