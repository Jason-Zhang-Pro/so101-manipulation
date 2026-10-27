"""Weighted stage contracts reach collection, success gates, CLI and resume.

The collection harness fakes simulation responses and the numerical optimizer,
while retaining production world scheduling, episode persistence and PPO gates.
"""
from dataclasses import replace
import sys

import pytest

from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from test_act_ppo_stage_success_skip_integration import run_collection
from test_act_ppo_stage_wave_integration import stage_config


WEIGHTS = "1:3:2:2:1:1"
EXPECTED_FRACTIONS = {"S0": .1, "S1": .3, "S2": .2, "S3": .2, "S4": .1, "S5": .1}
EXPECTED_STEPS = {0: 2, 1: 6, 2: 4, 3: 4, 4: 2, 5: 2}


def weighted_collection(tmp_path, monkeypatch, **kwargs):
    # Reuse the simulation-free harness without changing its equal-share tests.
    config_type = multi.MultiWorldConfig
    with monkeypatch.context() as scoped:
        scoped.setattr(multi, "MultiWorldConfig", lambda **values: config_type(
            **values, stage_step_weights=WEIGHTS))
        return run_collection(tmp_path, scoped, selection="s0_s5", rollout_steps=20, **kwargs)


def test_multiworld_scheduler_requires_weighted_actual_steps_before_update(tmp_path):
    base = replace(stage_config(tmp_path, rollout_steps=60), training_stages="s0_s5")
    worlds = multi.MultiWorldConfig(stage_step_weights=WEIGHTS, stage_ratio_tolerance=0.)
    scheduler = multi._stage_wave_scheduler(base, worlds)
    assert scheduler.diagnostics()["target_stage_fractions"] == pytest.approx(EXPECTED_FRACTIONS)
    for index in range(6):
        scheduler.observe(f"S{index}", 10)
    assert scheduler.total_steps == 60 and not scheduler.ready
    # Completing whole episodes can overshoot the requested minimum. The
    # resulting 100 real steps, rather than the initial 60, meet this ratio.
    scheduler.observe("S1", 20)
    scheduler.observe("S2", 10)
    scheduler.observe("S3", 10)
    assert scheduler.total_steps == 100 and scheduler.ready


def test_real_collection_uses_weights_and_records_actual_sampling_contract(tmp_path, monkeypatch):
    run = weighted_collection(tmp_path, monkeypatch, success_for=lambda *_: False)
    updates = [event for event in run.events if event[0] == "ppo"]
    assert [event[2] for event in updates] == [EXPECTED_STEPS]
    assert run.summary["total_collected_steps"] == 20
    assert run.summary["episode_count"] == 10
    sampling = run.summary["updates"][0]["stage_sampling"]
    assert sampling["target_stage_fractions"] == pytest.approx(EXPECTED_FRACTIONS)
    assert sampling["stage_fractions"] == pytest.approx(EXPECTED_FRACTIONS)
    assert sampling["ready"]


def test_success_gates_filter_then_renormalize_and_refill_remaining_stages(tmp_path, monkeypatch):
    run = weighted_collection(
        tmp_path, monkeypatch, minimum_episodes=25,
        success_for=lambda stage, count, total: count <= {"S0": 1, "S2": 2}.get(stage, 0),
    )
    updates = [event for event in run.events if event[0] == "ppo"]
    assert [event[2] for event in updates] == [
        {1: 6, 3: 4, 4: 2, 5: 2},
        {1: 12, 3: 8, 4: 4, 5: 4},
        EXPECTED_STEPS,
    ]
    # The skipped batch still exceeds the full 20-step threshold: a 3:2:1:1
    # ratio with complete two-step episodes requires 28 real collected steps.
    assert run.summary["total_collected_steps"] == 68
    assert run.summary["episode_count"] == 34
    assert run.trainer.total_environment_steps == 62
    assert run.summary["discarded_environment_steps"] == 0
    sampling = run.summary["updates"][1]["stage_sampling"]
    assert sampling["target_stage_fractions"] == pytest.approx(
        {"S1": 3 / 7, "S3": 2 / 7, "S4": 1 / 7, "S5": 1 / 7})
    assert sampling["total_steps"] == 28 and sampling["ready"]


def test_all_weighted_stages_successful_preserves_one_worldless_skip_cycle(tmp_path, monkeypatch):
    run = weighted_collection(tmp_path, monkeypatch, minimum_episodes=11,
                              success_for=lambda stage, count, total: total <= 10)
    assert [event[2] for event in run.events if event[0] == "ppo"] == [EXPECTED_STEPS]
    assert run.summary["total_collected_steps"] == 40
    assert run.summary["episode_count"] == 20
    assert run.summary["stage_skipped_update_count"] == 1
    assert run.summary["success_gated_transition_count"] == 20
    assert sum(record["type"] == "stage_sampling_skipped" for record in run.records) == 1
    assert sum(event[0] == "prepare" for event in run.events) == 20


def test_resume_signature_uses_proportions_not_weight_units_or_legacy_s0_fraction(tmp_path, monkeypatch):
    monkeypatch.delenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", raising=False)
    monkeypatch.delenv("SO101_IMAGE_AUGMENTATION_ENABLED", raising=False)
    base = replace(stage_config(tmp_path, rollout_steps=20), training_stages="s0_s5")
    worlds = multi.MultiWorldConfig(stage_step_weights=WEIGHTS)
    reward = single._resolved_reward(base)
    signature = multi._multiworld_execution_signature(base, reward, worlds)
    for ratio in ("2:6:4:4:2:2", "0.1:0.3:0.2:0.2:0.1:0.1"):
        assert multi._multiworld_execution_signature(
            base, reward, replace(worlds, stage_step_weights=ratio)) == signature
    assert multi._multiworld_execution_signature(
        base, reward, replace(worlds, s0_step_fraction=.9)) == signature
    assert multi._multiworld_execution_signature(
        base, reward, replace(worlds, stage_step_weights="1:1:1:1:1:1")) != signature


def test_cli_passes_requested_weights_to_runtime_before_worlds_start(monkeypatch):
    class ReachedRuntime(Exception):
        pass

    captured = []

    def capture(base, worlds):
        captured.append((base, worlds))
        raise ReachedRuntime

    monkeypatch.setattr(sys, "argv", ["act_ppo_multiworld_gazebo", "--act-checkpoint", "/tmp/actor",
                                    "--training-stages", "s0_s5", "--stage-step-weights", WEIGHTS])
    monkeypatch.setattr(multi, "run_act_ppo_multiworld_gazebo", capture)
    with pytest.raises(ReachedRuntime):
        multi.main()
    base, worlds = captured[0]
    assert worlds.stage_step_weights == WEIGHTS
    assert multi._stage_wave_scheduler(base, worlds).contract()["target_step_fractions"] == pytest.approx(
        EXPECTED_FRACTIONS)


@pytest.mark.parametrize("invalid", ["", "1:3:2:2:1", "1:3:2:2:1:1:1", "1:3:0:2:1:1",
                                     "1:3:-2:2:1:1", "1:3:nan:2:1:1", "1:3:inf:2:1:1",
                                     "1:3:no:2:1:1", "1,3,2,2,1,1"])
def test_invalid_weight_configuration_and_cli_fail_before_training(monkeypatch, invalid):
    with pytest.raises(ValueError):
        multi.MultiWorldConfig(stage_step_weights=invalid)

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed stage weights reached the training entry.")

    monkeypatch.setattr(sys, "argv", ["act_ppo_multiworld_gazebo", "--act-checkpoint", "/tmp/actor",
                                    "--training-stages", "s0_s5", "--stage-step-weights", invalid])
    monkeypatch.setattr(multi, "run_act_ppo_multiworld_gazebo", forbidden)
    with pytest.raises((ValueError, SystemExit)):
        multi.main()
