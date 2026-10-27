"""Schema-4 S1 stays one measured collection segment across lift and approach."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.rewards.s1_lift_approach_reward import S1LiftApproachReward
from so101_capstone.rewards.task_state import TaskMeasurements
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo import ActPpoRolloutBuffer, ActPpoTrainer, ActPpoValueNetwork
from so101_capstone.training.residual_ppo import model_state_sha256
from test_act_ppo import TinyActPolicy
from test_act_ppo_stage_wave_integration import FakeClient
from test_stage_process_integration import (
    DT, PROJECT, REWARD, add_transition, core_for, reset_core, small_ppo_config,
    step_core, training_config,
)
from test_task_measurements import GOAL


NEW_PROFILE = "act_ppo_stages_s1_lift_approach.json"
GUARD_PROFILE = "act_ppo_stages_s1_posture_guard.json"
COMPLETION_PROFILE = "act_ppo_stages_s1_lift_completion.json"
OLD_PROFILE = "act_ppo_stages_s1_progress.json"


def stages(profile=NEW_PROFILE):
    return StageTrainingConfig.load(PROJECT / "config" / profile)


def install_measured_lift(monkeypatch, core):
    """Replace sensor geometry only; real tracking/evaluation/reward still run."""
    def measurements(joint_state, cube_state, *, step_end_sim_time_s, **_):
        tick = round((step_end_sim_time_s - .06) / DT)
        joints = list(joint_state.positions.values())
        if tick > 0:
            joints[4] = REWARD.home_joint_positions_rad[4] - np.pi / 2
            joints[5] = REWARD.home_joint_positions_rad[5] + .7
        return TaskMeasurements(
            sim_time_s=step_end_sim_time_s, joint_positions_rad=joints,
            joint_velocities_rad_s=(0.,) * 6,
            grasp_position_world_m=(.1, 0., .79 if tick == 0 else .85),
            grasp_rotation_world=np.diag((1., -1., -1.)),
            target_name=cube_state.name, target_position_world_m=(.2, 0., .765),
            target_motion_valid=True, contact_valid=True, goal_center_world_m=GOAL,
        )

    monkeypatch.setattr(core, "_task_measurements", measurements)


def test_new_profile_roundtrips_and_preserves_the_previous_s1_schema_three():
    previous, current = stages(OLD_PROFILE), stages()
    assert previous.process_reward.schema_version == 3
    assert previous.process_reward.context_dim == 14
    assert "s1_lift_approach" not in previous.process_reward.to_dict()
    assert current.process_reward.schema_version == 4
    assert current.process_reward.context_dim == 26
    assert replace(current, process_reward=previous.process_reward,
                   s0_home_hold_s=previous.s0_home_hold_s,
                   s0_home_joint_tolerance_rad=None,
                   s0_home_velocity_tolerance_rad_s=None, s0_success_mode="window_end") == previous
    assert current.s0_home_hold_s == 1. and current.s0_success_mode == "position_hold"
    restored = StageTrainingConfig.from_dict(json.loads(json.dumps(current.to_dict())))
    assert restored == current and restored.config_hash == current.config_hash
    assert current.config_hash != previous.config_hash
    assert current.process_reward.s1_lift_approach.lift_clearance_m == .08
    assert (current.s0_budget_s, current.s1_budget_s) == (5., 10.)


@pytest.mark.parametrize("stage", ["S0", "S1"])
@pytest.mark.parametrize("profile,process_dim", [(NEW_PROFILE, 26), (GUARD_PROFILE, 34), (COMPLETION_PROFILE, 34)])
def test_versioned_history_reaches_real_environment_buffer_and_critic(tmp_path, stage, profile, process_dim):
    current = stages(profile)
    task_dim = 126 + process_dim
    config = training_config(tmp_path, current, stage)
    if profile in (GUARD_PROFILE, COMPLETION_PROFILE):
        config = replace(config, exploration_profile="final_good_act_near_target",
                         initial_log_std=-3., minimum_log_std=-9.,
                         maximum_log_std=-2.5, gripper_maximum_log_std=-1.5)
    core = core_for(stage, current)
    _, initial = reset_core(core, stage)
    _, reward, terminated, truncated, info = step_core(
        core, DT if stage == "S0" else .06 + DT, 4, stage=stage,
    )
    assert isinstance(core._stage_process_reward, S1LiftApproachReward) == (stage == "S1")
    for item in (initial, info):
        assert len(item["critic_task_context"]) == config.critic_task_context_dim == task_dim
        assert item["stage_process_context_schema_version"] == current.process_reward.context_schema_version
        assert item["task_context_schema_version"] == 3 + current.process_reward.context_schema_version
        assert item["critic_task_context"][126] == (stage == "S1")
        if stage == "S0":
            np.testing.assert_array_equal(item["critic_task_context"][126:], np.zeros(process_dim))
            assert "stage_s1_phase" not in item
        else:
            assert item["stage_s1_phase"] == "LIFT"
            assert item["stage_metrics"]["stage_s1_phase"] == "LIFT"
    if stage == "S1":
        np.testing.assert_array_equal(info["critic_task_context"][126:], core._stage_process_reward.critic_context())
        assert "s1_lift_progress" in info["reward_components"]
        assert "s1_roll_progress" in info["reward_components"]
    ppo = small_ppo_config(config)
    assert ppo.advantage_group_context_index == 126
    signature = single._execution_signature(config, REWARD)
    assert signature["task_context_dim"] == task_dim
    assert signature["task_context_schema"] == info["task_context_schema_version"]
    assert signature["stage_process_reward"]["context_dim"] == process_dim
    assert signature["stage_process_reward"]["config_hash"] == current.process_reward.config_hash
    if profile in (GUARD_PROFILE, COMPLETION_PROFILE):
        assert signature["exploration_profile"]["profile"] == "final_good_act_near_target"
    assert signature["stage_advantage_normalization"] == "binary_stage_actor_only_v1"
    context = single._critic_context(info, np.zeros((1, 6)), 0, 1, config)
    buffer = ActPpoRolloutBuffer(ppo)
    add_transition(buffer, context, reward=reward, terminated=terminated)
    batch = buffer.batch("cpu")
    assert batch.critic_task_context.shape == (1, task_dim)
    assert batch.advantages[0].item() == 0
    assert batch.returns[0].item() == pytest.approx(reward + REWARD.gamma * 7, abs=1e-5)
    assert not terminated and not truncated


@pytest.mark.parametrize("profile,process_dim", [(NEW_PROFILE, 26), (GUARD_PROFILE, 34), (COMPLETION_PROFILE, 34)])
def test_s0_keeps_previous_rewards_until_its_configured_stage_cut(profile, process_dim):
    previous = core_for("S0", stages(OLD_PROFILE))
    current_stages = stages(profile)
    current = core_for("S0", current_stages)
    reset_core(previous, "S0")
    reset_core(current, "S0")
    limit = 150 if current_stages.s0_success_mode == "window_end" else 30
    for tick in range(1, limit + 1):
        offset, velocity = ((.1, .2) if tick <= 10 else (0., 0.))
        old = step_core(previous, tick * DT, tick + 1, stage="S0", offset=offset, velocity=velocity)
        new = step_core(current, tick * DT, tick + 1, stage="S0", offset=offset, velocity=velocity)
        assert new[1:3] == old[1:3]  # Same reward and true-terminal semantics.
        assert new[3] == (tick == limit)
        assert old[3] == (tick == 150)
        assert new[4]["reward_components"] == old[4]["reward_components"]
        np.testing.assert_array_equal(new[4]["critic_task_context"][:126], old[4]["critic_task_context"][:126])
        np.testing.assert_array_equal(new[4]["critic_task_context"][126:], np.zeros(process_dim))
        assert new[4]["stage_elapsed_s"] == pytest.approx(tick * DT)
    assert new[4]["stage_success"] and new[3] and not new[2]


@pytest.mark.parametrize("profile", [NEW_PROFILE, GUARD_PROFILE, COMPLETION_PROFILE])
def test_lift_to_approach_flows_through_one_300_step_s1_without_budget_restart(monkeypatch, profile):
    core = core_for("S1", stages(profile))
    install_measured_lift(monkeypatch, core)
    _, initial = reset_core(core, "S1")
    assert initial["stage_s1_phase"] == "LIFT"
    certificate = initial["start_certificate"]
    phases = []
    for tick in range(1, 301):
        _, reward, terminated, truncated, info = step_core(core, .06 + tick * DT, tick + 3, stage="S1")
        phases.append(info["stage_s1_phase"])
        if profile == COMPLETION_PROFILE:
            assert info["reward_components"]["s1_lift_completed"] == (5. if tick == 4 else 0.)
        assert info["training_stage"] == "S1"
        assert info["stage_budget_s"] == 10.
        assert info["stage_elapsed_s"] == pytest.approx(tick * DT)
        assert info["start_certificate"] == certificate
        assert not terminated and truncated == (tick == 300)
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        np.testing.assert_array_equal(info["critic_task_context"][126:], core._stage_process_reward.critic_context())
    assert phases[:3] == ["LIFT"] * 3
    assert phases[3:] == ["APPROACH"] * 297
    assert info["stage_s1_lift_completed"]
    assert info["stage_result"] == "TIMEOUT" and not info["stage_success"]
    assert info["stage_s1_roll_error_rad"] == pytest.approx(0.)


@pytest.mark.parametrize("profile", [GUARD_PROFILE, COMPLETION_PROFILE])
def test_distance_success_immediately_cuts_real_environment_and_bootstraps(tmp_path, monkeypatch, profile):
    current = stages(profile)
    config = training_config(tmp_path, current, "distance_success")
    core = core_for("S1", current)

    def measurements(joint_state, cube_state, *, step_end_sim_time_s, **_):
        tick = round((step_end_sim_time_s - .06) / DT)
        position = ((.1, 0., .79) if tick == 0 else
                    (.1, 0., .85) if tick <= 4 else (.215, 0., .765))
        # Final 1.5 cm horizontal offset violates the old XY/corridor tests.
        # HOME wrist/opening and an upward axis also fail old posture gates.
        return TaskMeasurements(
            sim_time_s=step_end_sim_time_s,
            joint_positions_rad=REWARD.home_joint_positions_rad,
            joint_velocities_rad_s=(0.,) * 6,
            grasp_position_world_m=position, grasp_rotation_world=np.eye(3),
            target_name=cube_state.name, target_position_world_m=(.2, 0., .765),
            target_motion_valid=True, contact_valid=True, goal_center_world_m=GOAL,
        )

    monkeypatch.setattr(core, "_task_measurements", measurements)
    reset_core(core, "S1")
    for tick in range(1, 6):
        _, reward, terminated, truncated, info = step_core(core, .06 + tick * DT, tick + 3, stage="S1")
        assert not terminated
        assert truncated == (tick == 5)
    assert info["stage_s1_lift_completed"]
    assert not info["stage_s1_corridor_safe"]
    assert info["stage_reach_xy_m"] > current.s1_reach_xy_m
    assert info["stage_s1_roll_error_rad"] > current.process_reward.s1_lift_approach.roll_tolerance_rad
    assert info["stage_success"] and info["stage_result_reason"] == "reach_distance"
    assert info["stage_reach_hold_s"] == 0
    assert info["reward_components"]["s1_success"] == 10
    assert np.isfinite(reward)
    context = single._critic_context(info, np.zeros((1, 6)), 0, 1, config)
    buffer = ActPpoRolloutBuffer(small_ppo_config(config))
    add_transition(buffer, context, reward=reward, terminated=terminated)
    assert buffer.batch("cpu").returns[0].item() == pytest.approx(reward + REWARD.gamma * 7, abs=1e-5)


@pytest.mark.parametrize("source_profile,target_profile", [
    (OLD_PROFILE, NEW_PROFILE), (NEW_PROFILE, OLD_PROFILE),
    (NEW_PROFILE, GUARD_PROFILE), (GUARD_PROFILE, NEW_PROFILE),
])
def test_reward_versions_cannot_mix_rollouts_or_resume_before_mutating_models(tmp_path, source_profile, target_profile):
    source = training_config(tmp_path, stages(source_profile), "source")
    target = training_config(tmp_path, stages(target_profile), "target")
    source_ppo, target_ppo = small_ppo_config(source), small_ppo_config(target)
    assert source_ppo.critic_task_context_dim == 126 + stages(source_profile).process_reward.context_dim
    assert target_ppo.critic_task_context_dim == 126 + stages(target_profile).process_reward.context_dim
    assert source_ppo.critic_task_context_dim != target_ppo.critic_task_context_dim
    assert source_ppo.advantage_group_context_index == target_ppo.advantage_group_context_index == 126
    with pytest.raises(ValueError, match="configurations differ"):
        ActPpoRolloutBuffer(target_ppo).extend(ActPpoRolloutBuffer(source_ppo))

    def trainer(config, training):
        return ActPpoTrainer(TinyActPolicy(chunk_size=1), ActPpoValueNetwork(config),
                            config=config, execution_signature=single._execution_signature(training, REWARD))

    original = trainer(source_ppo, source)
    path = original.save_checkpoint(tmp_path / "training_state.pt")
    matching = trainer(source_ppo, source)
    matching.load_checkpoint(path)
    assert model_state_sha256(matching.policy) == model_state_sha256(original.policy)
    assert model_state_sha256(matching.critic) == model_state_sha256(original.critic)
    incompatible = trainer(target_ppo, target)
    modules = (incompatible.policy, incompatible.critic, incompatible.exploration)
    before = tuple(model_state_sha256(module) for module in modules)
    with pytest.raises(ValueError, match="execution signature"):
        incompatible.load_checkpoint(path)
    assert tuple(model_state_sha256(module) for module in modules) == before


@pytest.mark.parametrize("stage,mode,limit", [("S0", "idle_empty", 151), ("S1", "pick_place", 301)])
@pytest.mark.parametrize("profile,process_dim", [(NEW_PROFILE, 26), (GUARD_PROFILE, 34), (COMPLETION_PROFILE, 34)])
def test_single_and_multiworld_prepare_one_stage_with_original_5_10_second_budgets(tmp_path, stage, mode, limit, profile, process_dim):
    base = replace(training_config(tmp_path, stages(profile), stage), training_stages=stage.lower())
    assert single._episode_step_limit(base, mode) == limit
    ppo = single._act_ppo_config(base, 1)
    events = []

    class Client(FakeClient):
        def start_training_stage_async(self, training_stage, stage_config):
            assert stage_config["s0_budget_s"] == 5.
            assert stage_config["s1_budget_s"] == 10.
            assert stage_config["process_reward"]["schema_version"] == 4
            super().start_training_stage_async(training_stage, stage_config)

    clients = [Client(SimpleNamespace(world_index=index), events) for index in range(2)]
    for client in clients:
        client.active = True
    sessions, failures = multi._prepare_sessions(
        [multi._WorldSlot(client=client) for client in clients], base_config=base,
        ppo_config=ppo, attempt_counter=[0],
    )
    assert failures == [] and len(sessions) == 2
    assert [event for event in events if event[0] == "stage"] == [
        ("stage", 0, stage), ("stage", 1, stage),
    ]
    assert all(session.max_steps == limit and session.completed_steps == 0 for session in sessions)
    assert [event for event in events if event[0] == "spawn"] == ([('spawn', 0), ('spawn', 1)] if stage == "S1" else [])
    signature = multi._multiworld_execution_signature(base, REWARD, multi.MultiWorldConfig(world_count=2))
    assert signature["stage_training"]["config"]["s1_budget_s"] == 10.
    assert signature["task_context_dim"] == 126 + process_dim
