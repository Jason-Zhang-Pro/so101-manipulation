"""All-stage collection keeps physical preparation outside on-policy trajectories."""
from dataclasses import replace
from pathlib import Path

import pytest

from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo_stage_schedule import BalancedStageWaveScheduler
from so101_capstone.training.act_ppo_stage_success_skip import StageSuccessSkipController
from test_act_ppo_stage_collection import config, GEOMETRY
from test_act_ppo_stage_cli import build_config
from test_act_ppo_stage_success_skip_integration import run_collection


@pytest.mark.parametrize("selection", ["s2", "s3", "s4", "s5", "s0_s5"])
@pytest.mark.parametrize("multiworld", [False, True])
def test_new_cli_selections_and_context_survive_single_wave_narrowing(selection, multiworld):
    base = build_config(["--training-stages", selection, "--s2-budget-s", "8",
                         "--s3-budget-s", "9", "--s4-budget-s", "11"], multiworld=multiworld)
    resolved = single._resolved_stage_config(base)
    assert resolved.enable_later_stage_context
    assert [resolved.stage_budget_s(stage) for stage in ("S2", "S3", "S4", "S5")] == [8, 9, 11, 5]
    ppo = single._act_ppo_config(base, 2)
    assert ppo.advantage_group_count == 6
    assert ppo.advantage_group_context_index is not None
    assert base.resolved_stage_initialization_timeout_s == 600
    for stage in ("s0", "s1", "s2", "s3", "s4", "s5"):
        wave = replace(base, training_stages=stage)
        assert wave.critic_task_context_dim == base.critic_task_context_dim
        assert single._resolved_stage_config(wave).enable_later_stage_context
        assert single._episode_training_stage(wave, 1) == stage.upper()
        assert single._episode_step_limit(wave, "idle_empty" if stage == "s0" else "pick_place") == int(resolved.stage_budget_s(stage.upper()) * 30) + 1


def test_legacy_context_and_initialization_timeout_unchanged():
    from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM
    base = config(training_stages="s0_s1")
    assert base.critic_task_context_dim == TASK_CONTEXT_DIM
    assert single._act_ppo_config(base, 2).advantage_group_count == 2
    assert base.resolved_stage_initialization_timeout_s == 180
    assert replace(base, stage_initialization_timeout_s=250).resolved_stage_initialization_timeout_s == 250


@pytest.mark.parametrize("stage", ["S2", "S3", "S4", "S5"])
def test_later_prepare_spawns_source_then_calls_actual_stage_before_any_policy_sample(monkeypatch, stage):
    monkeypatch.delenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", raising=False)
    events = []

    class Client:
        world_index = 2
        def prepare_episode_async(self, attempt, **kwargs):
            assert kwargs["task_mode"] == "pick_place"
            events.append("prepare")
            self.response = {"type": "episode_prepared", "observation": "unprepared", "info": GEOMETRY,
                             "cleanup": {}, "home_reset": {}}
        def spawn_target_async(self, position, quaternion):
            events.append("spawn")
            self.response = {"type": "target_spawned", "success": True, "message": "{}"}
        def start_training_stage_async(self, name, cfg):
            assert name == stage and cfg["enable_later_stage_context"]
            events.append("physical_prepare")
            self.response = {"type": "stage_started", "observation": "frame_zero",
                             "info": {"training_stage": stage, "stage_elapsed_s": 0.,
                                      "stage_initialization": {"preparation_control_steps": 250}}}
        def receive(self, **kwargs):
            if self.response["type"] == "stage_started":
                assert kwargs["timeout_s"] == 600
            return self.response
        def observation_from_metadata(self, metadata):
            return metadata

    base = config(training_stages=stage.lower())
    sessions, failed = multi._prepare_sessions([multi._WorldSlot(client=Client())], base_config=base,
        ppo_config=single._act_ppo_config(base, 2), attempt_counter=[0])
    assert not failed and len(sessions) == 1
    session = sessions[0]
    assert events == ["prepare", "spawn", "physical_prepare"]
    assert session.observation == "frame_zero" and session.training_stage == stage
    assert len(session.buffer) == session.completed_steps == 0
    assert session.buffer.actor_observations == []
    assert session.stage_initialization["stage_initialization"]["preparation_control_steps"] == 250


def test_balancing_uses_actual_steps_and_short_successes_without_padding():
    scheduler = BalancedStageWaveScheduler(selection="s0_s5", rollout_steps=60, ratio_tolerance=.02)
    budgets = {f"S{i}": 300 for i in range(6)}
    visited = set()
    for _ in range(200):
        plan = scheduler.plan_wave(4, s0_episode_steps=150, s1_episode_steps=300,
                                   stage_episode_steps=budgets)
        stage = plan["stage"]
        visited.add(stage)
        for _ in range(plan["world_count"]):
            scheduler.observe(stage, 1 if stage == "S2" else 5)
        if scheduler.ready:
            break
    assert scheduler.ready and visited == set(budgets)
    diag = scheduler.diagnostics()
    assert diag["total_steps"] >= 60
    assert diag["s2_episode_count"] == diag["s2_steps"]
    assert all(abs(value - 1/6) <= .02 for value in diag["stage_fractions"].values())


def test_skip_any_later_stage_and_refill_with_remaining_stages():
    base = config(training_stages="s0_s5", rollout_steps=60)
    worlds = multi.MultiWorldConfig(world_count=4, stage_success_skip_enabled=True)
    controller = multi._stage_success_skip_controller(base, worlds)
    for stage in ("S0", "S1", "S2", "S3", "S4", "S5"):
        for index in range(5):
            controller.observe({"training_stage": stage, "valid": True,
                                "stage_success": stage in {"S0", "S2", "S5"} and index < 4})
    decision = controller.finish_batch()
    assert decision.exclude_update_stages == {"S0", "S2", "S5"}
    scheduler = multi._stage_wave_scheduler(base, worlds, controller.current_skipped_stages)
    assert set(scheduler.stages) == {"S1", "S3", "S4"}
    for stage in scheduler.stages:
        scheduler.observe(stage, 20)
    assert scheduler.ready and scheduler.total_steps == 60
    for stage in controller.active_stages:
        controller.observe({"training_stage": stage, "valid": True, "stage_success": False})
    controller.finish_batch()
    assert controller.active_stages == {f"S{i}" for i in range(6)}


def test_all_six_skipped_consumes_exactly_one_empty_batch():
    controller = StageSuccessSkipController("s0_s5")
    for stage in controller.active_stages:
        controller.observe({"training_stage": stage, "valid": True, "stage_success": True})
    controller.finish_batch()
    assert not controller.active_stages
    assert controller.consume_empty_batch().empty_batch
    assert controller.active_stages == {f"S{i}" for i in range(6)}
    with pytest.raises(RuntimeError):
        controller.consume_empty_batch()


def test_all_six_real_collection_boundary_filters_later_stage_and_closes_worlds(tmp_path, monkeypatch):
    run = run_collection(tmp_path, monkeypatch, selection="s0_s5", rollout_steps=12,
                         minimum_episodes=7, success_for=lambda stage, count, total: stage in {"S0", "S2"})
    updates = [event for event in run.events if event[0] == "ppo"]
    assert len(updates) == 2
    assert updates[0][2] == {1: 2, 3: 2, 4: 2, 5: 2}
    assert updates[1][1] >= 12 and set(updates[1][2]) == {1, 3, 4, 5}
    decisions = [record for record in run.records if record["type"] == "stage_success_skip_decision"]
    assert decisions[0]["next_skip_stages"] == ["S0", "S2"]
    assert decisions[1]["next_skip_stages"] == []
    assert set(run.summary["training_stage_summary"]) == {f"S{i}" for i in range(6)}


@pytest.mark.parametrize("stage,group", [("S2", "manipulation"), ("S3", "moving"),
                                        ("S4", "manipulation"), ("S5", "moving")])
def test_exploration_uses_local_stage_not_preparation_physical_phase(stage, group):
    import numpy as np
    base = config(training_stages=stage.lower(), exploration_profile="final_good_act",
                  initial_log_std=-3., minimum_log_std=-9., maximum_log_std=-2.5,
                  gripper_maximum_log_std=-1.5)
    info = {"training_stage": stage, "task_phase": "APPROACH", "target_distance_m": .01,
            "home_error_rad": .2, "home_joint_tolerance_rad": .14, "robot_at_home": False}
    name, _, offset = single._exploration_stage(info, base)
    assert name == group and offset.shape == (6,) and np.isfinite(offset).all()
    if stage == "S5":
        near = {**info, "stage_home_error_rad": .01}
        assert np.all(single._exploration_stage(near, base)[2] < offset)


def test_preparation_failure_metadata_survives_worker_without_wasting_policy_steps():
    from so101_capstone.training.act_ppo_multiworld_worker import _error_payload
    from so101_capstone.training.act_ppo import ActPpoRolloutBuffer
    error = RuntimeError("failed physical grasp")
    error.stage_initialization = {"phase": "close", "preparation_control_steps": 178,
                                  "reason": "no bilateral contact"}
    response = _error_payload(error, operation="start_training_stage")
    class Client:
        world_index = 0
        def start_training_stage_async(self, stage, config):
            assert stage == "S3"
        def receive(self, **kwargs):
            return response
    base = config(training_stages="s3")
    session = multi._EpisodeSession(slot=multi._WorldSlot(client=Client()),
        buffer=ActPpoRolloutBuffer(single._act_ppo_config(base, 2)),
        observation=None, last_info={}, cleanup={}, home_reset={}, attempt_index=1,
        training_stage="S3")
    ready, failures = multi._start_training_stage_sessions([session], stage_config={})
    assert not ready and len(failures) == 1
    failure = failures[0]
    assert failure.category == "stage_initialization"
    assert failure.stage_initialization == error.stage_initialization
    assert failure.completed_steps == failure.discarded_transition_count == 0


def test_new_contract_records_reward_and_preparation_defaults_and_rejects_old_shape():
    old = config(training_stages="s0_s1")
    new = config(training_stages="s0_s5")
    reward = single._resolved_reward(new)
    signature = single._execution_signature(new, reward)
    assert signature["later_stage_training"]["stages"]["S2"]["config"]["lift_clearance_m"] == .03
    assert signature["stage_preparation"]["joint_speed_rad_s"] == .5
    assert signature["task_context_dim"] > old.critic_task_context_dim
    with pytest.raises(ValueError, match="context"):
        single._critic_context({"critic_task_context": [0.] * old.critic_task_context_dim},
                               None, 0, 2, new)


def test_worker_timeout_records_known_stage_and_unknown_preparation_steps():
    from so101_capstone.training.act_ppo import ActPpoRolloutBuffer
    from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerCommunicationError
    class Client:
        world_index = 0
        def start_training_stage_async(self, stage, config):
            pass
        def receive(self, **kwargs):
            raise GazeboWorkerCommunicationError("stage preparation worker timeout")
    base = config(training_stages="s5")
    session = multi._EpisodeSession(slot=multi._WorldSlot(client=Client()),
        buffer=ActPpoRolloutBuffer(single._act_ppo_config(base, 1)),
        observation=None, last_info={}, cleanup={}, home_reset={}, attempt_index=1,
        training_stage="S5")
    ready, failures = multi._start_training_stage_sessions([session], stage_config={})
    assert not ready and len(failures) == 1
    assert failures[0].stage_initialization["training_stage"] == "S5"
    assert failures[0].stage_initialization["preparation_control_steps"] is None
    assert failures[0].stage_initialization["policy_steps"] == 0
    assert failures[0].completed_steps == 0


def test_formal_json_context_matches_real_core_handoff_and_first_training_step(monkeypatch):
    import test_later_stage_training as core_tests
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    from so101_capstone.rewards.training_stages import StageTrainingConfig
    root = Path(__file__).resolve().parents[2]
    reward = PickPlaceRewardConfig.load(root / "config/pick_place_reward.json")
    stage_path = root / "config/act_ppo_stages_s0_s5.json"
    stages = StageTrainingConfig.load(stage_path)
    original_trace = core_tests.Trace
    monkeypatch.setattr(core_tests, "Trace", lambda: original_trace(config=reward))
    stream = core_tests.CoreTrace(monkeypatch)
    for _ in range(4):
        stream.frame(contact=True)
    for _ in range(4):
        stream.frame(contact=True, position=(0., 0., .82))
    base = config(training_stages="s0_s5", stage_config=stage_path,
                  reward_config=root / "config/pick_place_reward.json")
    ppo = single._act_ppo_config(base, 1)
    snapshot = stream.core.task_snapshot
    _, info = stream.core.handoff_training_stage("S3", stages,
        command_joint_targets_rad=snapshot.measurements.joint_positions_rad)
    assert len(info["critic_task_context"]) == ppo.critic_task_context_dim
    assert info["critic_task_context"][ppo.advantage_group_context_index] == 3
    _, _, terminal, truncated, info = stream.frame(contact=True)
    assert not terminal and not truncated and info["episode_step"] == 1
    assert len(info["critic_task_context"]) == ppo.critic_task_context_dim
