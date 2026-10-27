from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.rewards import (
    PickPlaceReward, PickPlaceRewardConfig, PickPlaceRewardInput,
    TaskMeasurements, TaskPhase, TaskSnapshot,
)


def snapshot(t=0.0, *, target=False, phase=TaskPhase.WAIT_HOME, mode="pick_place", **kwargs):
    cfg = PickPlaceRewardConfig()
    m = TaskMeasurements(t, cfg.home_joint_positions_rad, (0,) * 6, (0, 0, 1), np.eye(3),
                         target_name="cube" if target else None,
                         target_position_world_m=(0, 0, 0.765) if target else None,
                         target_motion_valid=target, contact_valid=True)
    return TaskSnapshot(m, "episode", mode, phase, home_score=1, **kwargs)


def advance(prev, *, dt=1/30, **changes):
    return replace(prev, measurements=replace(prev.measurements, sim_time_s=prev.sim_time_s+dt), **changes)


def test_config_round_trip_hash_gamma_contract_and_candidate_status(tmp_path):
    cfg = PickPlaceRewardConfig.load(Path(__file__).parents[2] / "config" / "pick_place_reward.json", gamma=.997)
    assert cfg.calibration_status == "candidate_unvalidated" and cfg.failure_cost == 10
    assert PickPlaceRewardConfig.from_dict(cfg.to_dict()).config_hash == cfg.config_hash
    assert replace(cfg, failure_cost=5).config_hash != cfg.config_hash
    path = tmp_path / "reward.json"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert PickPlaceRewardConfig.load(path).to_dict() == cfg.to_dict()
    with pytest.raises(ValueError, match="gamma"):
        PickPlaceRewardConfig.load(path, gamma=.99)
    assert PickPlaceRewardConfig.load(None, gamma=.995).gamma == .995
    with pytest.raises(ValueError, match="unknown"):
        PickPlaceRewardConfig.from_dict({"typo": 2})


@pytest.mark.parametrize("updates", [
    {"gamma": math.nan}, {"gamma": 0}, {"home_hold_min_s": 2},
    {"home_hold_max_s": 2}, {"grasp_confirm_s": 0}, {"failure_cost": -1},
    {"home_position_scales_rad": (1, 1)}, {"velocity_scales_rad_s": (0,) * 6},
    {"release_max_clearance_m": .03}, {"grasp_evidence_mode": "auto"},
    {"home_joint_positions_rad": (math.inf,) * 6}, {"shaping_enabled": 1},
    {"control_hz": "30"}, {"failure_cost": True}, {"home_reset_jitter_rad": -.1},
])
def test_invalid_config_is_rejected(updates):
    with pytest.raises(ValueError):
        PickPlaceRewardConfig.from_dict(updates)


def test_home_hold_bounds_follow_executed_chunk_and_keep_random_window():
    cfg = PickPlaceRewardConfig()
    assert cfg.resolve_home_hold_bounds(1) == (3, 5)
    low, high = cfg.resolve_home_hold_bounds(100)
    assert low == pytest.approx(200/30) and high-low == pytest.approx(2)
    low, high = replace(cfg, home_hold_max_s=3).resolve_home_hold_bounds(100)
    assert high-low >= 1/cfg.control_hz-1e-9
    with pytest.raises(ValueError):
        cfg.resolve_home_hold_bounds(0)


def test_stationary_home_is_never_positive_and_has_zero_base_cost():
    cfg = PickPlaceRewardConfig()
    evaluator = PickPlaceReward(cfg)
    prev = snapshot(mode="idle_empty")
    evaluator.reset(prev)
    for _ in range(10):
        nxt = advance(prev)
        result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
        assert result.reward == pytest.approx((cfg.gamma-1)*2)
        assert sum(v for k, v in result.components.items() if k != "potential_shaping") == 0
        prev = nxt


def test_physical_target_spawn_does_not_reset_potential():
    evaluator = PickPlaceReward(PickPlaceRewardConfig())
    prev = snapshot()
    evaluator.reset(prev)
    nxt = snapshot(1/30, target=True, phase=TaskPhase.APPROACH, approach_score=.2)
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
    assert result.potential_before == 2
    assert result.potential_after == pytest.approx(.4)
    assert result.components["potential_shaping"] == pytest.approx(.997*.4-2)


@pytest.mark.parametrize("outcome,bonus", [("success", 100), ("task_failure", -10)])
def test_true_terminal_zeroes_potential_and_includes_final_correction(outcome, bonus):
    cfg = PickPlaceRewardConfig()
    evaluator = PickPlaceReward(cfg)
    prev = snapshot(target=True, phase=TaskPhase.HOME_HOLD, home_stable_duration_s=2.9)
    evaluator.reset(prev)
    nxt = advance(prev, outcome=outcome, success_event=outcome == "success", home_stable_duration_s=3)
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30, terminated=True))
    assert result.potential_after == 0
    assert result.reward == pytest.approx(bonus-.01-result.potential_before)
    with pytest.raises(RuntimeError, match="reset"):
        evaluator.step(PickPlaceRewardInput(nxt, advance(nxt), 1/30, terminated=True))


def test_truncation_keeps_successor_potential_and_never_adds_failure():
    evaluator = PickPlaceReward(PickPlaceRewardConfig())
    prev = snapshot(target=True, phase=TaskPhase.TRANSPORT, transport_score=.8)
    evaluator.reset(prev)
    nxt = advance(prev, transport_score=.9)
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30, truncated=True))
    assert result.potential_after == pytest.approx(5.8)
    assert result.components["failure"] == 0 and result.truncated and not result.terminated


def test_base_reward_ablation_really_disables_all_potential_terms():
    evaluator = PickPlaceReward(PickPlaceRewardConfig(
        shaping_enabled=False, remaining_task_cost_per_s=0.,
    ))
    prev = snapshot(target=True, phase=TaskPhase.GRASP_LIFT, lift_score=.8)
    evaluator.reset(prev)
    nxt = advance(prev, lift_score=1)
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
    assert result.potential_before == result.potential_after == 0
    assert result.reward == pytest.approx(-.01)


def test_home_cost_uses_q_error_and_actual_velocity_not_absolute_action():
    cfg = PickPlaceRewardConfig()
    prev = snapshot(mode="idle_empty")
    evaluator = PickPlaceReward(cfg)
    evaluator.reset(prev)
    command = cfg.home_joint_positions_rad  # Nonzero absolute HOME is valid.
    nxt = advance(prev)
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30, command_joint_targets_rad=command, previous_command_joint_targets_rad=command))
    assert result.components["home_position"] == result.components["home_velocity"] == 0
    prev = nxt
    nxt = advance(prev)
    nxt = replace(nxt, measurements=replace(nxt.measurements, joint_positions_rad=(0,) * 6, joint_velocities_rad_s=(.5,) * 6))
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
    assert -.05 <= result.components["home_position"] < 0
    assert -.02 <= result.components["home_velocity"] < 0


def test_return_home_does_not_receive_strong_home_cost():
    cfg = PickPlaceRewardConfig()
    prev = snapshot(target=True, phase=TaskPhase.RETURN_HOME)
    nxt = advance(prev)
    nxt = replace(nxt, measurements=replace(nxt.measurements, joint_positions_rad=(0,) * 6, joint_velocities_rad_s=(.5,) * 6))
    evaluator = PickPlaceReward(cfg)
    evaluator.reset(prev)
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
    assert result.components["home_position"] == result.components["home_velocity"] == 0


def test_costs_scale_with_simulated_dt_but_discount_stays_per_control_step():
    cfg = PickPlaceRewardConfig()
    prev = snapshot(target=True, phase=TaskPhase.APPROACH, approach_score=.5)
    evaluator = PickPlaceReward(cfg)
    evaluator.reset(prev)
    nxt = advance(prev, dt=.1)
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, .1))
    assert result.components["active_time"] == pytest.approx(-.03)
    assert result.components["potential_shaping"] == pytest.approx(cfg.gamma-1)


def test_command_cost_requires_real_two_commands_and_is_bounded():
    cfg = PickPlaceRewardConfig(command_change_cost_per_step=.005)
    evaluator = PickPlaceReward(cfg)
    prev = snapshot()
    evaluator.reset(prev)
    nxt = advance(prev)
    with pytest.raises(ValueError, match="actual current and previous"):
        evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
    result = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30,
                            command_joint_targets_rad=(10,) * 6,
                            previous_command_joint_targets_rad=(0,) * 6))
    assert result.components["command_change"] == -.005


def test_duplicate_and_discontinuous_transitions_are_rejected():
    evaluator = PickPlaceReward(PickPlaceRewardConfig())
    prev = snapshot()
    nxt = advance(prev)
    evaluator.reset(prev)
    evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
    with pytest.raises(ValueError, match="previous snapshot"):
        evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30))
    with pytest.raises(ValueError, match="dt"):
        evaluator.step(PickPlaceRewardInput(nxt, advance(nxt), 1))
    with pytest.raises(ValueError, match="identity"):
        evaluator.step(PickPlaceRewardInput(nxt, replace(advance(nxt), task_instance_id="other"), 1/30))


@pytest.mark.parametrize("terminated,truncated,outcome", [(True,True,"success"),(False,False,"success"),(True,False,"running")])
def test_inconsistent_termination_cannot_enter_reward(terminated, truncated, outcome):
    evaluator = PickPlaceReward(PickPlaceRewardConfig())
    prev = snapshot()
    evaluator.reset(prev)
    nxt = advance(prev, outcome=outcome)
    with pytest.raises(ValueError):
        evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30, terminated=terminated, truncated=truncated))


def test_potential_is_nonnegative_bounded_in_every_phase():
    evaluator = PickPlaceReward(PickPlaceRewardConfig())
    for phase in TaskPhase:
        for score in (0, .25, .7, 1):
            s = snapshot(target=True, phase=phase, approach_score=score, contact_score=score,
                         lift_score=score, transport_score=score, support_score=score,
                         separation_score=score, home_stable_duration_s=score*3,
                         place_stable_duration_s=score*.4, valid_release=bool(score))
            assert 0 <= evaluator.potential(s) <= 10
            assert evaluator.potential(s, terminal=True) == 0
