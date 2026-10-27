"""S0 position-only success must not fabricate a static-robot measurement."""
from dataclasses import replace
import math

import pytest

from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_training_stages import StageTrace


def position_only(**overrides):
    values = dict(s0_home_joint_tolerance_rad=math.radians(8),
                  s0_home_hold_s=1., s0_success_mode="position_hold")
    values.update(overrides)
    return StageTrainingConfig(**values)


def positions(trace, joint, offset):
    q = list(trace.reward.home_joint_positions_rad)
    q[joint] += offset
    return q


def test_position_hold_success_ignores_velocity_but_retains_real_static_metrics():
    trace = StageTrace("S0", config=position_only())
    for index in range(20):
        q = [home + (-1 if index % 2 else 1) * math.radians(2)
             for home in trace.reward.home_joint_positions_rad]
        info = trace.step(joint_positions_rad=q, joint_velocities_rad_s=(2.,) * 6)
        assert info["stage_success"] is (index == 19)
        assert info["stage_home_hold_candidate"]
        assert info["stage_robot_at_home"]
        assert not info["stage_robot_static"]
        assert info["stage_max_joint_velocity_rad_s"] == 2
        assert info["stage_home_static_evidence_s"] == 0
        assert info["stage_home_static_fraction"] == 0
        assert info["stage_s0_home_velocity_required"] is False
        assert info["stage_s0_home_velocity_tolerance_rad_s"] == trace.reward.home_velocity_tolerance_rad_s
    assert info["stage_cut"] and info["stage_result_reason"] == "position_hold"
    assert info["stage_home_hold_s"] == pytest.approx(1)
    assert info["stage_elapsed_s"] == pytest.approx(1)
    assert trace.snapshot.outcome == "running" and not trace.snapshot.success_event


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_eight_degree_boundary_on_each_joint_including_gripper_is_excluded(joint, sign):
    trace = StageTrace("S0", config=position_only())
    q = positions(trace, joint, sign * math.radians(8))
    for _ in range(21):
        info = trace.step(joint_positions_rad=q)
        assert not info["stage_robot_at_home"]
        assert not info["stage_home_hold_candidate"]
        assert info["stage_home_hold_s"] == 0
        assert not info["stage_success"]


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("sign", [-1, 1])
def test_measurably_inside_strict_boundary_is_accepted(joint, sign):
    trace = StageTrace("S0", config=position_only())
    q = positions(trace, joint, sign * (math.radians(8) - 1e-7))
    for _ in range(20):
        info = trace.step(joint_positions_rad=q, joint_velocities_rad_s=(.5,) * 6)
    assert info["stage_success"]


@pytest.mark.parametrize("joint", range(6))
def test_single_joint_leaving_position_band_resets_continuous_hold(joint):
    trace = StageTrace("S0", config=position_only())
    for _ in range(19):
        trace.step(joint_velocities_rad_s=(.5,) * 6)
    info = trace.step(joint_positions_rad=positions(trace, joint, math.radians(8.1)))
    assert not info["stage_home_hold_candidate"]
    assert info["stage_home_hold_s"] == 0
    for _ in range(20):
        assert not trace.step(joint_velocities_rad_s=(.5,) * 6)["stage_success"]
    assert trace.step(joint_velocities_rad_s=(.5,) * 6)["stage_success"]


def test_velocity_variation_does_not_restart_position_hold_or_relabel_static_time():
    trace = StageTrace("S0", config=position_only())
    for _ in range(10):
        trace.step()
    for _ in range(10):
        info = trace.step(joint_velocities_rad_s=(.5,) * 6)
    assert info["stage_success"]
    assert info["stage_home_hold_s"] == pytest.approx(1)
    assert info["stage_home_static_evidence_s"] == pytest.approx(.5)
    assert info["stage_home_static_fraction"] == pytest.approx(.5)


def test_explicit_velocity_tolerance_still_only_affects_static_diagnostics():
    trace = StageTrace("S0", config=position_only(s0_home_velocity_tolerance_rad_s=.1))
    for _ in range(20):
        info = trace.step(joint_velocities_rad_s=(.2,) * 6)
    assert info["stage_success"]
    assert not info["stage_robot_static"]
    assert info["stage_s0_home_velocity_required"] is False


def test_duplicate_evidence_cannot_count_twice_or_be_caught_up_later():
    trace = StageTrace("S0", config=position_only())
    for _ in range(18):
        trace.step(joint_velocities_rad_s=(.5,) * 6)
    info = trace.step(evidence_sim_time_s=.9, joint_velocities_rad_s=(.5,) * 6)
    assert info["stage_home_hold_s"] == pytest.approx(.9)
    assert not info["stage_evidence_fresh"]
    info = trace.step(joint_velocities_rad_s=(.5,) * 6)
    assert info["stage_home_hold_s"] == pytest.approx(.95)
    assert not info["stage_success"]
    assert trace.step(joint_velocities_rad_s=(.5,) * 6)["stage_success"]


def test_evidence_gap_clears_position_hold():
    trace = StageTrace("S0", config=position_only())
    for _ in range(19):
        trace.step()
    info = trace.step(dt=.2)
    assert info["stage_home_hold_s"] == 0
    assert not info["stage_success"]
    for _ in range(19):
        assert not trace.step()["stage_success"]
    assert trace.step()["stage_success"]


def test_invalid_joint_evidence_cannot_certify_success():
    trace = StageTrace("S0", config=position_only())
    for _ in range(19):
        trace.step()
    with pytest.raises(ValueError, match="invalid stage sensor evidence"):
        trace.step(joint_measurement_valid=False)


@pytest.mark.parametrize("last_dt, success", [(.05, True), (.06, False)])
def test_hold_must_complete_by_deadline(last_dt, success):
    trace = StageTrace("S0", config=position_only(s0_budget_s=1.))
    for _ in range(19):
        trace.step(joint_velocities_rad_s=(.5,) * 6)
    info = trace.step(dt=last_dt, joint_velocities_rad_s=(.5,) * 6)
    assert info["stage_success"] is success
    assert info["stage_timeout"] is not success
    assert info["stage_cut"]


def test_physical_failure_beats_simultaneous_position_success():
    trace = StageTrace("S0", config=position_only())
    for _ in range(19):
        trace.step()
    trace.time += .05
    snapshot = trace.tracker.update(trace.measurement(joint_velocities_rad_s=(.5,) * 6))
    info = trace.evaluator.update(replace(snapshot, outcome="task_failure", failure_reason="collision"))
    assert info["stage_home_hold_s"] == pytest.approx(1)
    assert info["stage_result"] == "FAILURE"
    assert not info["stage_success"] and not info["stage_cut"]


@pytest.mark.parametrize("criterion", ["position", "velocity"])
def test_position_only_success_does_not_relax_initial_state_certificate(criterion):
    trace = StageTrace("S0", config=position_only())
    values = ({"joint_positions_rad": positions(trace, 5, math.radians(2))}
              if criterion == "position" else {"joint_velocities_rad_s": (.1,) * 6})
    initial = replace(trace.snapshot, measurements=replace(trace.snapshot.measurements, **values))
    with pytest.raises(ValueError, match=f"home_{criterion}_valid"):
        trace.evaluator.reset(initial)


def test_legacy_hash_and_both_existing_modes_remain_distinct():
    assert StageTrainingConfig().config_hash == "918facc6e4d89d41747cea931d3443ec8d048688c71161950a1ba04b6a829784"
    cfg = position_only()
    assert StageTrainingConfig.from_dict(cfg.to_dict()) == cfg
    assert len({cfg.config_hash, replace(cfg, s0_success_mode="home_hold").config_hash,
                replace(cfg, s0_success_mode="window_end").config_hash}) == 3
    trace = StageTrace("S0", config=replace(cfg, s0_success_mode="home_hold"))
    for _ in range(20):
        info = trace.step(joint_velocities_rad_s=(.5,) * 6)
    assert info["stage_s0_home_velocity_required"] is True
    assert not info["stage_home_hold_candidate"] and not info["stage_success"]
    assert info["stage_home_hold_s"] == 0


def test_s1_complete_info_and_success_do_not_change():
    baseline = StageTrace("S1")
    adjusted = StageTrace("S1", config=position_only())
    assert baseline.info == adjusted.info
    for _ in range(4):
        assert baseline.step(reach=True) == adjusted.step(reach=True)
    assert adjusted.info["stage_success"]
