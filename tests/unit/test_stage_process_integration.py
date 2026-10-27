"""Stage reward selection, checkpoint contracts, and real measured-stage cuts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.environments.reach_env_core import (
    ReachEnvironmentConfig,
    So101ReachEnvironmentCore,
)
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.stage_process_reward import StageProcessRewardConfig
from so101_capstone.rewards.task_state import TaskMeasurements
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training.act_ppo import ActPpoRolloutBuffer, ActPpoValueNetwork
from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig,
    _act_ppo_config,
    _critic_context,
    _execution_signature,
)
from test_act_ppo import actor_observation, critic_observation
from test_task_measurements import GOAL, contacts, cube, joint


PROJECT = Path(__file__).resolve().parents[2]
REWARD = PickPlaceRewardConfig(reward_schema_version=5)
DT = 1 / 30


def dense_stages(**kwargs):
    return StageTrainingConfig(process_reward=StageProcessRewardConfig(), **kwargs)


def training_config(tmp_path, stages, name):
    reward_path = tmp_path / f"{name}_reward.json"
    reward_path.write_text(json.dumps(REWARD.to_dict()))
    stage_path = tmp_path / f"{name}_stages.json"
    stage_path.write_text(json.dumps(stages.to_dict()))
    return ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "actor", reward_profile="pick_place_v2",
        reward_config=reward_path, stage_config=stage_path,
        training_stages="s0_s1", gamma=REWARD.gamma, final_target_absent_s=0,
    )


def core_for(stage, stages):
    core = So101ReachEnvironmentCore(
        ReachEnvironmentConfig(max_episode_steps=2), REWARD,
        reward_profile="pick_place_v2", goal_region_center_world_m=GOAL,
        task_mode="idle_empty" if stage == "S0" else "pick_place",
    )
    core.configure_training_stage(stage, stages)
    return core


def reset_core(core, stage):
    if stage == "S0":
        return core.reset_from_feedback(
            joint(0), now_monotonic_s=0, step_end_sim_time_s=0,
            contact_cache=contacts(0),
        )
    return core.reset_from_feedback(
        joint(.06, 3), cube(.06, 3), now_monotonic_s=0,
        step_end_sim_time_s=.06, cube_history=(cube(0), cube(.03, 2)),
        contact_cache=contacts(0, .03, .06), camera_source_times_s=(.06, .06),
    )


def step_core(core, t, seq, *, stage, offset=0., velocity=0., opening=None):
    measured = joint(t, seq, offset=offset)
    if opening is not None:
        positions = dict(measured.positions)
        gripper = tuple(positions)[-1]
        positions[gripper] = REWARD.home_joint_positions_rad[-1] + opening
        measured = replace(measured, positions=positions)
    measured = replace(measured, velocities=dict.fromkeys(measured.positions, velocity))
    return core.step_from_feedback(
        measured, cube(t, seq) if stage == "S1" else None,
        now_monotonic_s=0, step_end_sim_time_s=t, control_dt_s=DT,
        contact_cache=contacts(max(0., t-DT), t),
    )


def small_ppo_config(config):
    return replace(_act_ppo_config(config, 1), critic_image_height=16,
                   critic_image_width=16, critic_hidden_dim=32)


def add_transition(buffer, context, *, reward, terminated=False, episode_end=True):
    ref = buffer.add_actor_observation(actor_observation())
    buffer.add(
        critic_observation(), actor_observation_index=ref, action_index=0,
        policy_action=np.zeros(6), log_probability=0., value=2., reward=reward,
        next_value=7., terminated=terminated, episode_end=episode_end, **context,
    )


def test_legacy_stage_serialization_and_hash_are_exactly_preserved():
    original = json.loads((PROJECT / "config/act_ppo_stages.json").read_text())
    expected_hash = hashlib.sha256(json.dumps(
        original, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    for values in (original, dict(original, process_reward=None)):
        config = StageTrainingConfig.from_dict(values)
        assert config.process_reward is None
        assert config.to_dict() == original
        assert "process_reward" not in config.to_dict()
        assert config.config_hash == expected_hash


def test_nested_process_config_roundtrips_and_project_thresholds_are_unchanged():
    old = StageTrainingConfig.load(PROJECT / "config/act_ppo_stages.json")
    dense = StageTrainingConfig.load(PROJECT / "config/act_ppo_stages_dense.json")
    assert isinstance(dense.process_reward, StageProcessRewardConfig)
    encoded = dense.to_dict()
    assert StageTrainingConfig.from_dict(json.loads(json.dumps(encoded))) == dense
    assert StageTrainingConfig.from_dict(encoded).config_hash == dense.config_hash
    assert encoded.pop("process_reward") == dense.process_reward.to_dict()
    assert encoded == old.to_dict()
    assert dense.config_hash != old.config_hash


@pytest.mark.parametrize("process", [False, [], {"unknown": 1}, {"s1_xy_scale_m": 0}])
def test_nested_invalid_process_configuration_is_rejected(process):
    with pytest.raises(ValueError):
        StageTrainingConfig.from_dict({"process_reward": process})


def test_s1_replaces_full_reward_and_context_reaches_the_ppo_contract(tmp_path, monkeypatch):
    stages = dense_stages()
    config = training_config(tmp_path, stages, "dense")
    core = core_for("S1", stages)
    _, initial = reset_core(core, "S1")

    def no_full_reward(_transition):
        pytest.fail("S1 must replace, rather than add to, the full task reward")

    monkeypatch.setattr(core._pick_place_reward, "step", no_full_reward)
    _, reward, terminated, truncated, after = step_core(core, .06+DT, 4, stage="S1")
    assert not terminated and not truncated
    assert reward == pytest.approx(sum(after["reward_components"].values()))
    assert "s1_distance_coarse_progress" in after["reward_components"]
    assert "reach_step" not in after["reward_components"]
    assert "reach_x_alignment_regression" not in after["reward_components"]
    assert "remaining_task" not in after["reward_components"]
    assert after["potential_after"] > REWARD.potential_max
    assert initial["critic_task_context"][93:126] == after["critic_task_context"][93:126]
    for info in (initial, after):
        assert len(info["critic_task_context"]) == config.critic_task_context_dim == 138
        assert info["task_context_schema_version"] == 4
        assert info["stage_process_context_schema_version"] == 1
        assert info["critic_task_context"][-12:-10] == [1., 1.]
        contexts = _critic_context(info, np.zeros((1, 6)), 0, 1, config)
        assert contexts["critic_task_context"].shape == (138,)
    np.testing.assert_array_equal(after["critic_task_context"][-12:], core._stage_process_reward.critic_context())
    assert _act_ppo_config(config, 1).critic_task_context_dim == 138
    buffer = ActPpoRolloutBuffer(small_ppo_config(config))
    add_transition(buffer, contexts, reward=reward, episode_end=False)
    np.testing.assert_array_equal(buffer.critic_task_context[0], after["critic_task_context"])


def test_s0_original_rewards_match_component_for_component_with_zero_context_padding():
    original, dense = core_for("S0", StageTrainingConfig()), core_for("S0", dense_stages())
    _, old_initial = reset_core(original, "S0")
    _, new_initial = reset_core(dense, "S0")
    assert new_initial["critic_task_context"][:-12] == old_initial["critic_task_context"]
    np.testing.assert_array_equal(new_initial["critic_task_context"][-12:], np.zeros(12))
    for tick in range(1, 151):
        offset, speed = ((.1, .2) if tick == 1 else (.05, .1) if tick == 2 else (0., 0.))
        old = step_core(original, tick*DT, tick+1, stage="S0", offset=offset, velocity=speed)
        new = step_core(dense, tick*DT, tick+1, stage="S0", offset=offset, velocity=speed)
        assert new[1:4] == old[1:4]
        old_info, new_info = old[4], new[4]
        assert new_info["reward_components"] == old_info["reward_components"]
        assert new_info["critic_task_context"][:-12] == old_info["critic_task_context"]
        np.testing.assert_array_equal(new_info["critic_task_context"][-12:], np.zeros(12))
    assert new_info["stage_success"] and new_info["truncated"]
    assert not new_info["terminated"] and not new_info["success"]
    assert new_info["reward_components"]["success"] == 0
    assert new_info["potential_after"] > 0


@pytest.mark.parametrize("reset_configuration", ["stage_none", "task"])
def test_removing_stage_configuration_removes_process_context(reset_configuration):
    core = core_for("S0", dense_stages())
    _, before = reset_core(core, "S0")
    assert len(before["critic_task_context"]) == 138
    if reset_configuration == "stage_none":
        core.configure_training_stage(None)
    else:
        core.configure_task("idle_empty")
    _, after = reset_core(core, "S0")
    assert len(after["critic_task_context"]) == 126
    assert after["task_context_schema_version"] == 3
    assert "stage_process_context_schema_version" not in after
    assert "training_stage" not in after
    assert core._stage_process_reward is None


def test_real_s1_success_cuts_with_new_potential_and_gae_bootstrap(tmp_path, monkeypatch):
    stages = dense_stages()
    config = training_config(tmp_path, stages, "success")
    core = core_for("S1", stages)

    def measured_pose(joint_state, cube_state, *, step_end_sim_time_s, **_):
        # Inject aligned sensor geometry only. The real tracker, stage evaluator,
        # environment reward selection, and PPO boundary handling remain active.
        joints = tuple(joint_state.positions.values())
        opened = joints[-1]-REWARD.home_joint_positions_rad[-1] > .45
        return TaskMeasurements(
            sim_time_s=step_end_sim_time_s, joint_positions_rad=joints,
            joint_velocities_rad_s=(0.,)*6,
            grasp_position_world_m=(.2, 0., .775 if opened else 1.1),
            grasp_rotation_world=np.diag((1., -1., -1.)),
            target_name=cube_state.name, target_position_world_m=(.2, 0., .765),
            target_motion_valid=True, contact_valid=True, goal_center_world_m=GOAL,
        )

    monkeypatch.setattr(core, "_task_measurements", measured_pose)
    reset_core(core, "S1")
    for tick in range(1, 7):
        _, reward, terminated, truncated, info = step_core(
            core, .06+tick*DT, tick+3, stage="S1", opening=.7,
        )
        assert not terminated
        assert truncated == (tick == 6)
    assert info["stage_success"] and info["stage_result_reason"] == "reach_hold"
    assert info["task_outcome"] == "running" and not info["success"]
    assert info["reward_components"]["s1_success"] == stages.process_reward.s1_success_bonus
    assert info["reward_components"]["success"] == 0
    assert info["potential_after"] == pytest.approx(sum(stages.process_reward.s1_potential_weights))
    contexts = _critic_context(info, np.zeros((1, 6)), 0, 1, config)
    buffer = ActPpoRolloutBuffer(small_ppo_config(config))
    add_transition(buffer, contexts, reward=reward, terminated=terminated, episode_end=truncated)
    advantages, returns = buffer._advantages_and_returns()
    assert returns[0] == pytest.approx(reward+REWARD.gamma*7.)
    assert advantages[0] == pytest.approx(reward+REWARD.gamma*7.-2.)


def test_old_new_signatures_and_critic_shapes_cannot_mix_replay(tmp_path):
    old_config = training_config(tmp_path, StageTrainingConfig(), "old")
    dense = dense_stages()
    new_config = training_config(tmp_path, dense, "new")
    old_signature = _execution_signature(old_config, REWARD)
    new_signature = _execution_signature(new_config, REWARD)
    assert "stage_process_reward" not in old_signature
    assert "process_reward" not in old_signature["stage_training"]["config"]
    assert old_signature["stage_training"]["boundary"] == "full_objective_sampling_cut_bootstrap_v1"
    assert old_signature["task_context_dim"] == 126
    assert old_signature["task_context_schema"] == 3
    assert new_signature["task_context_dim"] == 138
    assert new_signature["task_context_schema"] == 4
    assert new_signature["stage_process_reward"]["config_hash"] == dense.process_reward.config_hash
    weights = list(dense.process_reward.s1_potential_weights)
    weights[0] += 1.
    changed = replace(dense, process_reward=replace(dense.process_reward, s1_potential_weights=weights))
    changed_config = training_config(tmp_path, changed, "changed")
    assert _execution_signature(changed_config, REWARD) != new_signature
    old_ppo, new_ppo = small_ppo_config(old_config), small_ppo_config(new_config)
    old_critic, new_critic = ActPpoValueNetwork(old_ppo), ActPpoValueNetwork(new_ppo)
    assert new_critic.state_encoder[0].in_features == old_critic.state_encoder[0].in_features+12
    old_info = {"critic_task_context": [0.]*126}
    with pytest.raises(ValueError, match="critic task context"):
        _critic_context(old_info, np.zeros((1, 6)), 0, 1, new_config)
    old_contexts = _critic_context(old_info, np.zeros((1, 6)), 0, 1, old_config)
    buffer = ActPpoRolloutBuffer(new_ppo)
    with pytest.raises(ValueError, match="critic_task_context"):
        add_transition(buffer, old_contexts, reward=0.)
    assert len(buffer) == 0
