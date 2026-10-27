"""Whole-stage waves must be balanced by accepted steps, not episode counts."""
import math

import pytest

from so101_capstone.training.act_ppo_stage_schedule import StageWaveScheduler


def scheduler(**kwargs):
    return StageWaveScheduler(**{"selection": "s0_s1", "rollout_steps": 1800, **kwargs})


@pytest.mark.parametrize("tolerance,expected_steps,expected_waves", [
    (.05, 4200, ["S1", "S0", "S0", "S1", "S0"]),
    (.01, 6000, ["S1", "S0", "S0", "S1", "S0", "S1", "S0"]),
])
@pytest.mark.parametrize("rollout_steps", [1800, 2400])
def test_full_600_1200_step_waves_wait_for_actual_ratio(tolerance, expected_steps, expected_waves, rollout_steps):
    plan = scheduler(ratio_tolerance=tolerance, rollout_steps=rollout_steps)
    waves = []
    while not plan.ready:
        stage = plan.next_stage()
        waves.append(stage)
        plan.observe(stage, 600 if stage == "S0" else 1200)
    assert waves == expected_waves
    assert plan.total_steps == expected_steps
    info = plan.diagnostics()
    assert info["total_steps"] == info["s0_steps"] + info["s1_steps"]
    assert info["ratio_met"] and info["ready"]
    assert info["ratio_error"] <= tolerance + 1e-12


def test_early_s1_success_counts_only_executed_steps_without_padding():
    plan = scheduler()
    accepted = []
    while not plan.ready:
        stage = plan.next_stage()
        for _ in range(4):
            steps = 150 if stage == "S0" else 40
            plan.observe(stage, steps)
            accepted.append((stage, steps))
    info = plan.diagnostics()
    assert info["s1_steps"] == sum(steps for stage, steps in accepted if stage == "S1")
    assert info["s0_steps"] == sum(steps for stage, steps in accepted if stage == "S0")
    assert info["total_steps"] == sum(steps for _, steps in accepted)
    assert info["total_steps"] == 2800
    assert info["s1_steps"] == 1600  # Forty real short episodes, not 40 * 300.
    assert abs(info["actual_s0_step_fraction"] - .4) <= .05


def test_raw_minimum_step_quotas_are_not_sufficient_for_ratio_readiness():
    plan = scheduler(s0_step_fraction=.2)
    plan.observe("S0", 600)
    plan.observe("S1", 1440)
    assert plan.total_steps > 1800
    assert not plan.ready
    assert plan.next_stage() == "S1"


def test_even_wide_tolerance_requires_every_positive_target_stage():
    plan = scheduler(ratio_tolerance=.5)
    plan.observe("S1", 1800)
    assert plan.diagnostics()["ratio_met"]
    assert not plan.diagnostics()["required_stages_present"]
    assert not plan.ready
    assert plan.next_stage() == "S0"


@pytest.mark.parametrize("selection,provided_p,effective_p,stage", [
    ("s0", .4, 1., "S0"), ("s1", .4, 0., "S1"),
    ("s0_s1", 1., 1., "S0"), ("s0_s1", 0., 0., "S1"),
])
def test_single_stage_selection_and_endpoint_fractions(selection, provided_p, effective_p, stage):
    plan = scheduler(selection=selection, s0_step_fraction=provided_p)
    assert plan.contract()["target_s0_step_fraction"] == effective_p
    assert plan.next_stage() == stage
    with pytest.raises(ValueError, match="not a target stage"):
        plan.observe("S1" if stage == "S0" else "S0", 1)
    plan.observe(stage, 1800)
    assert plan.ready
    assert plan.diagnostics()["ratio_error"] == 0


@pytest.mark.parametrize("fraction,expected", [(.4, "S1"), (.5, "S0"), (.6, "S0")])
def test_empty_and_exact_ratio_ties_are_deterministic(fraction, expected):
    plan = scheduler(s0_step_fraction=fraction)
    assert plan.next_stage() == expected
    plan.observe("S0", round(10 * fraction))
    plan.observe("S1", round(10 * (1 - fraction)))
    assert plan.next_stage() == expected


def test_cap_is_checked_before_next_wave_and_reports_ratio_deficit():
    plan = scheduler(max_rollout_multiplier=1)
    plan.observe("S1", 1200)
    assert plan.next_stage() == "S0"
    plan.observe("S0", 600)
    assert not plan.ready
    with pytest.raises(RuntimeError, match="S0=600, S1=1200.*cap=1800"):
        plan.next_stage()


def test_started_complete_wave_can_overshoot_cap_and_still_finish_ready():
    plan = scheduler(rollout_steps=100, max_rollout_multiplier=1)
    plan.observe("S1", 60)
    assert plan.next_stage() == "S0"
    plan.observe("S0", 30)
    plan.observe("S0", 10)
    assert plan.ready
    # The wave was already started: its remaining complete episodes survive.
    plan.observe("S0", 9)
    assert plan.total_steps == 109 and plan.ready
    with pytest.raises(RuntimeError, match="ready for PPO"):
        plan.next_stage()


def test_ready_midwave_may_change_before_all_wave_segments_are_accepted():
    plan = scheduler(rollout_steps=100)
    plan.observe("S1", 60)
    plan.observe("S0", 40)
    assert plan.ready
    plan.observe("S0", 100)
    assert not plan.ready
    assert plan.next_stage() == "S1"


def test_noninteger_multiplier_cap_rounds_up():
    plan = scheduler(rollout_steps=101, max_rollout_multiplier=1.01)
    assert plan.diagnostics()["max_steps_before_next_wave"] == math.ceil(101 * 1.01) == 103


def test_ratio_boundary_uses_small_numerical_tolerance():
    plan = scheduler(rollout_steps=100, s0_step_fraction=.45, ratio_tolerance=.05)
    plan.observe("S0", 40)
    plan.observe("S1", 60)
    assert plan.ready


def test_reset_clears_only_batch_counts_and_contract_never_contains_counts():
    plan = scheduler()
    contract = plan.contract()
    plan.observe("S0", 1200)
    plan.observe("S1", 1800)
    assert plan.ready
    assert plan.contract() == contract
    plan.reset()
    info = plan.diagnostics()
    assert info["s0_steps"] == info["s1_steps"] == info["total_steps"] == 0
    assert info["actual_s0_step_fraction"] is None and info["ratio_error"] is None
    assert not info["ready"] and not info["ratio_met"]
    assert plan.next_stage() == "S1"
    assert plan.contract() == contract
    assert not any(key in contract for key in ("s0_steps", "s1_steps", "total_steps"))


@pytest.mark.parametrize("kwargs", [
    {"selection": "full"}, {"selection": "S0"},
    {"rollout_steps": 0}, {"rollout_steps": -1}, {"rollout_steps": True}, {"rollout_steps": 1.5},
    {"s0_step_fraction": -.01}, {"s0_step_fraction": 1.01}, {"s0_step_fraction": math.nan},
    {"s0_step_fraction": True}, {"s0_step_fraction": "0.4"},
    {"ratio_tolerance": -.01}, {"ratio_tolerance": .51}, {"ratio_tolerance": math.inf},
    {"ratio_tolerance": True},
    {"max_rollout_multiplier": .99}, {"max_rollout_multiplier": math.inf},
    {"max_rollout_multiplier": True},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        scheduler(**kwargs)


@pytest.mark.parametrize("stage,steps", [("s0", 1), ("S2", 1), ("S0", 0),
                                         ("S0", -1), ("S1", True), ("S1", 2.5), ("S1", "3")])
def test_invalid_observation_does_not_change_counts(stage, steps):
    plan = scheduler()
    before = plan.diagnostics()
    with pytest.raises(ValueError):
        plan.observe(stage, steps)
    assert plan.diagnostics() == before


def test_variable_world_tail_reaches_2400_without_full_wave_overshoot():
    plan = scheduler(rollout_steps=2400)
    first = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
    assert first["planned_additional_episodes"] == {"S0": 6, "S1": 5}
    assert first["estimated_final_steps"] == 2400
    assert first["estimated_final_s0_fraction"] == pytest.approx(.375)
    assert not first["planning_fallback"]
    assert not plan.ready and plan.total_steps == 0  # Forecasts are not samples.
    waves = []
    while not plan.ready:
        wave = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
        waves.append((wave["stage"], wave["world_count"]))
        for _ in range(wave["world_count"]):
            plan.observe(wave["stage"], 150 if wave["stage"] == "S0" else 300)
    assert waves == [("S1", 4), ("S0", 4), ("S0", 2), ("S1", 1)]
    assert plan.total_steps == 2400
    assert plan.diagnostics()["s0_steps"] == 900
    assert plan.diagnostics()["s1_steps"] == 1500


@pytest.mark.parametrize("max_worlds", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("rollout_steps,expected_total", [(1800, 1950), (2400, 2400), (3000, 3000)])
def test_planning_adapts_to_world_cap_and_rollout_threshold(max_worlds, rollout_steps, expected_total):
    plan = scheduler(rollout_steps=rollout_steps)
    while not plan.ready:
        wave = plan.plan_wave(max_worlds, s0_episode_steps=150, s1_episode_steps=300)
        assert 1 <= wave["world_count"] <= max_worlds
        for _ in range(wave["world_count"]):
            plan.observe(wave["stage"], 150 if wave["stage"] == "S0" else 300)
    assert plan.total_steps == expected_total
    assert plan.diagnostics()["ratio_error"] <= .05


def test_tail_replans_with_short_s1_measured_mean_without_padding():
    plan = scheduler(rollout_steps=2400)
    for _ in range(4):
        plan.observe("S1", 40)
    wave = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
    assert wave["estimated_episode_steps"] == {"S0": 150., "S1": 40.}
    assert wave["planned_additional_episodes"] == {"S0": 7, "S1": 30}
    assert wave["estimated_final_steps"] == 2410
    assert plan.total_steps == 160
    while not plan.ready:
        wave = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
        for _ in range(wave["world_count"]):
            plan.observe(wave["stage"], 150 if wave["stage"] == "S0" else 40)
    assert plan.total_steps == 2410
    assert plan.diagnostics()["s1_steps"] == 1360


def test_tail_forecast_keeps_noninteger_episode_means():
    plan = scheduler(rollout_steps=20)
    for stage, steps in (("S0", 1), ("S0", 2), ("S1", 2), ("S1", 3)):
        plan.observe(stage, steps)
    wave = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
    assert wave["estimated_episode_steps"] == {"S0": 1.5, "S1": 2.5}
    assert wave["planned_additional_episodes"] == {"S0": 3, "S1": 3}
    assert wave["estimated_final_steps"] == 20
    assert plan.total_steps == 8 and not plan.ready


def test_tail_can_use_other_stage_when_deficit_preference_needs_no_more_episodes():
    plan = scheduler(rollout_steps=95)
    plan.observe("S0", 35)
    for _ in range(11):
        plan.observe("S1", 5)
    assert plan.next_stage() == "S0"
    wave = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
    assert wave["stage"] == "S1" and wave["world_count"] == 1
    assert wave["planned_additional_episodes"] == {"S0": 0, "S1": 1}
    assert wave["estimated_final_steps"] == 95


def test_infeasible_forecast_falls_back_until_actual_collection_cap():
    plan = scheduler(rollout_steps=10, ratio_tolerance=0, max_rollout_multiplier=1)
    wave = plan.plan_wave(1, s0_episode_steps=3, s1_episode_steps=7)
    assert wave["planning_fallback"]
    assert wave["planning_status"] == "fallback_no_feasible_tail"
    assert (wave["stage"], wave["world_count"]) == ("S1", 1)
    plan.observe("S1", 7)
    wave = plan.plan_wave(1, s0_episode_steps=3, s1_episode_steps=7)
    assert wave["planning_fallback"]
    assert (wave["stage"], wave["world_count"]) == ("S0", 1)
    plan.observe("S0", 3)
    with pytest.raises(RuntimeError, match="collection cap"):
        plan.plan_wave(1, s0_episode_steps=3, s1_episode_steps=7)


def test_large_forecast_search_is_bounded_and_falls_back():
    plan = scheduler(rollout_steps=1_000_000)
    wave = plan.plan_wave(4, s0_episode_steps=1, s1_episode_steps=1)
    assert wave["planning_status"] == "fallback_search_bound"
    assert wave["world_count"] == 4


def test_reset_forgets_batch_episode_means_but_keeps_versioned_contract():
    plan = scheduler()
    plan.observe("S0", 12)
    plan.observe("S1", 7)
    before = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
    assert before["estimated_episode_steps"] == {"S0": 12., "S1": 7.}
    contract = plan.contract()
    plan.reset()
    after = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
    assert after["estimated_episode_steps"] == {"S0": 150., "S1": 300.}
    assert plan.diagnostics()["s0_episode_count"] == plan.diagnostics()["s1_episode_count"] == 0
    assert plan.contract() == contract
    assert contract["schema"] == "fixed_stage_wave_actual_steps_v2_variable_world_count"
    assert contract["world_count_semantics"] == "maximum_available_worlds_per_wave"


@pytest.mark.parametrize("selection,expected", [("s0", "S0"), ("s1", "S1")])
def test_single_stage_tail_uses_fewer_worlds_and_keeps_actual_ready_guard(selection, expected):
    plan = scheduler(selection=selection, rollout_steps=350)
    wave = plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)
    assert wave["stage"] == expected
    assert wave["world_count"] == (3 if selection == "s0" else 2)
    for _ in range(wave["world_count"]):
        plan.observe(expected, 150 if expected == "S0" else 300)
    assert plan.ready
    with pytest.raises(RuntimeError, match="ready for PPO"):
        plan.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300)


@pytest.mark.parametrize("values", [
    {"max_worlds": 0}, {"max_worlds": True}, {"max_worlds": 2.5},
    {"s0_episode_steps": 0}, {"s0_episode_steps": False},
    {"s1_episode_steps": -1}, {"s1_episode_steps": 1.5},
])
def test_invalid_wave_inputs_leave_counts_unchanged(values):
    plan = scheduler()
    kwargs = {"max_worlds": 4, "s0_episode_steps": 150, "s1_episode_steps": 300, **values}
    before = plan.diagnostics()
    with pytest.raises(ValueError):
        plan.plan_wave(**kwargs)
    assert plan.diagnostics() == before
