"""Weighted collection uses accepted transitions without truncating episodes."""

import math

import pytest

from so101_capstone.training.act_ppo_stage_schedule import (
    BalancedStageWaveScheduler,
    parse_stage_step_weights,
)


STAGES = tuple(f"S{index}" for index in range(6))
WEIGHTS = "1:3:2:2:1:1"
FRACTIONS = dict(zip(STAGES, (.1, .3, .2, .2, .1, .1), strict=True))


def scheduler(**kwargs):
    return BalancedStageWaveScheduler(
        **{"selection": "s0_s5", "rollout_steps": 2400, "stage_step_weights": WEIGHTS, **kwargs}
    )


def plan(s, max_worlds=4, **budgets):
    values = {stage: 120 for stage in STAGES}
    values.update(budgets)
    return s.plan_wave(max_worlds, s0_episode_steps=values["S0"], s1_episode_steps=values["S1"],
                       stage_episode_steps=values)


@pytest.mark.parametrize("value", [None, 123, True, "", "1:2", "1:1:1:1:1:1:1", "1:1::1:1:1",
                                   "1:1:0:1:1:1", "1:1:-1:1:1:1", "1:1:nan:1:1:1",
                                   "1:1:inf:1:1:1", "1:1:1e999:1:1:1", "1:1:a:1:1:1",
                                   "1:1:1e-999:1:1:1"])
def test_invalid_weight_values_fail_without_silent_default(value):
    with pytest.raises(ValueError, match="six positive finite numbers"):
        parse_stage_step_weights(value)
    with pytest.raises(ValueError, match="six positive finite numbers"):
        scheduler(stage_step_weights=value)


def test_weights_accept_decimals_and_spaces_and_define_effective_step_targets():
    assert parse_stage_step_weights(" .5 : 1.5:1:1:.5:.5 ") == (.5, 1.5, 1., 1., .5, .5)
    s = scheduler()
    assert s.target_stage_fractions == FRACTIONS
    assert s.diagnostics()["target_stage_fractions"] == FRACTIONS
    assert [2400 * value for value in s.target_stage_fractions.values()] == [240, 720, 480, 480, 240, 240]
    assert s.contract()["schema"] == "weighted_stage_wave_actual_steps_v1"


def test_equivalent_ratio_scales_have_identical_resume_contracts():
    expected = scheduler().contract()
    assert scheduler(stage_step_weights="2:6:4:4:2:2").contract() == expected
    assert scheduler(stage_step_weights=".5:1.5:1:1:.5:.5").contract() == expected
    assert scheduler(stage_step_weights=".1:.3:.2:.2:.1:.1").contract() == expected
    assert scheduler(stage_step_weights=".7:2.1:1.4:1.4:.7:.7").contract() == expected
    assert scheduler(stage_step_weights="1:2:2:2:1:1").contract() != expected


def test_large_finite_weights_do_not_overflow_their_sum():
    s = scheduler(stage_step_weights="1e308:5e307:5e307:1e308:5e307:5e307")
    assert sum(s.target_stage_fractions.values()) == pytest.approx(1)
    assert s.target_stage_fractions["S0"] == pytest.approx(.25)
    assert all(math.isfinite(value) and value > 0 for value in s.target_stage_fractions.values())
    assert plan(s)["world_count"] > 0


def test_skipped_stages_are_removed_then_remaining_weights_renormalized():
    s = scheduler(active_stages=frozenset({"S1", "S2", "S3", "S4"}))
    assert s.target_stage_fractions == {"S1": 3 / 8, "S2": 2 / 8, "S3": 2 / 8, "S4": 1 / 8}
    assert s.s0_step_fraction == 0
    assert s.rollout_steps == 2400
    assert s.contract()["stages"] == ["S1", "S2", "S3", "S4"]
    with pytest.raises(ValueError, match="not active"):
        s.observe("S0", 200)
    for stage, count in zip(s.stages, (900, 600, 600, 300), strict=True):
        s.observe(stage, count)
    assert s.ready
    assert s.diagnostics()["s0_steps"] == 0
    assert s.diagnostics()["s5_steps"] == 0


def test_exact_weighted_counts_are_ready_and_empty_reset_restores_same_contract():
    s = scheduler(ratio_tolerance=0)
    contract = s.contract()
    for stage, count in zip(STAGES, (240, 720, 480, 480, 240, 240), strict=True):
        s.observe(stage, count)
    assert s.ready
    assert s.diagnostics()["ratio_error"] == 0
    with pytest.raises(RuntimeError, match="ready for PPO"):
        s.next_stage()
    s.reset()
    assert s.total_steps == 0 and not s.ready
    assert s.next_stage() == "S0"
    assert s.contract() == contract


def test_below_threshold_or_absent_stage_cannot_be_ready_even_with_perfect_or_loose_ratio():
    short = scheduler(ratio_tolerance=0)
    for stage, count in zip(STAGES, (120, 360, 240, 240, 120, 120), strict=True):
        short.observe(stage, count)
    assert short.diagnostics()["ratio_met"]
    assert not short.ready
    absent = scheduler(ratio_tolerance=.5)
    absent.observe("S1", 2400)
    assert not absent.ready
    assert not absent.diagnostics()["required_stages_present"]


def test_ratio_tolerance_is_absolute_percentage_points_for_every_active_stage():
    s = scheduler(ratio_tolerance=.02)
    for stage, count in zip(STAGES, (240, 768, 432, 480, 240, 240), strict=True):
        s.observe(stage, count)
    assert s.ready
    outside = scheduler(ratio_tolerance=.02)
    for stage, count in zip(STAGES, (240, 769, 431, 480, 240, 240), strict=True):
        outside.observe(stage, count)
    assert not outside.ready


def test_next_stage_uses_progress_toward_weighted_share_and_deterministic_ties():
    s = scheduler(rollout_steps=10000)
    for stage, count in zip(STAGES, (100, 180, 160, 200, 100, 100), strict=True):
        s.observe(stage, count)
    # S1 has more raw steps than S2 but less progress toward its larger share.
    assert s.next_stage() == "S1"
    s.observe("S1", 60)
    assert s.next_stage() == "S1"  # S1 and S2 both represent 800 weighted steps.
    s.observe("S1", 1)
    assert s.next_stage() == "S2"


def test_wave_size_catches_up_weighted_counts_and_keeps_complete_episodes():
    s = scheduler()
    first = plan(s)
    assert first["stage"] == "S0" and first["world_count"] == 2
    assert first["planning_status"] == "weighted_share_complete_episode_wave"
    s.observe("S0", 120)
    s.observe("S0", 120)
    second = plan(s)
    assert second["stage"] == "S1" and second["world_count"] == 4
    assert second["planned_additional_episodes"]["S1"] == 4


def test_oversized_completed_episode_expands_other_targets_instead_of_discarding_it():
    s = scheduler(rollout_steps=1000)
    s.observe("S0", 300)
    wave = plan(s, max_worlds=20, S1=100)
    # S0 represents a 3000-step total at its 10% share, so S1 aims for 900.
    assert wave["stage"] == "S1" and wave["world_count"] == 9
    assert s.diagnostics()["s0_steps"] == 300
    assert not s.ready


def test_short_successes_and_unequal_episode_lengths_fill_real_steps_until_ready():
    s = scheduler(ratio_tolerance=.03)
    actual_lengths = {"S0": 30, "S1": 60, "S2": 90, "S3": 120, "S4": 60, "S5": 30}
    episodes = {stage: [] for stage in STAGES}
    for _ in range(100):
        if s.ready:
            break
        wave = plan(s, S0=150, S1=300, S2=300, S3=300, S4=300, S5=150)
        stage = wave["stage"]
        assert 1 <= wave["world_count"] <= 4
        # Whole wave must be accepted even if one episode reaches the threshold.
        for _ in range(wave["world_count"]):
            count = actual_lengths[stage]
            s.observe(stage, count)
            episodes[stage].append(count)
    assert s.ready
    assert s.total_steps >= 2400
    for stage in STAGES:
        assert s.diagnostics()["stage_steps"][stage] == sum(episodes[stage])
        assert s.diagnostics()[f"{stage.lower()}_episode_count"] == len(episodes[stage])
        assert abs(s.diagnostics()["stage_fractions"][stage] - FRACTIONS[stage]) <= .03 + 1e-12


def test_collection_cap_does_not_authorize_an_imbalanced_update():
    s = scheduler(rollout_steps=60, max_rollout_multiplier=1)
    s.observe("S0", 90)
    assert not s.ready
    # observe accepts complete in-flight episodes, even beyond the cap.
    s.observe("S0", 30)
    assert s.total_steps == 120
    with pytest.raises(RuntimeError, match="collection cap"):
        plan(s)


def test_default_equal_contract_is_backward_compatible_and_scale_independent():
    s = BalancedStageWaveScheduler(selection="s0_s5", rollout_steps=2400)
    assert s.contract() == {
        "schema": "equal_stage_wave_actual_steps_v1", "selection": "s0_s5", "stages": list(STAGES),
        "target_step_fractions": {stage: 1 / 6 for stage in STAGES}, "ratio_tolerance": .05,
        "max_rollout_multiplier": 4., "rollout_step_threshold": 2400,
        "world_count_semantics": "maximum_available_worlds_per_wave",
        "tail_planning": "least_actual_steps_complete_episode_wave_actual_counts_decide_readiness",
    }
    assert scheduler(stage_step_weights="7:7:7:7:7:7").contract() == s.contract()
    assert plan(s)["planning_status"] == "equal_share_complete_episode_wave"
    assert plan(s)["world_count"] == 4


def test_single_active_stage_keeps_total_threshold_and_equal_contract():
    s = scheduler(active_stages=frozenset({"S3"}))
    assert s.target_stage_fractions == {"S3": 1.}
    assert s.contract()["schema"] == "equal_stage_wave_actual_steps_v1"
    s.observe("S3", 2399)
    assert not s.ready
    s.observe("S3", 1)
    assert s.ready
