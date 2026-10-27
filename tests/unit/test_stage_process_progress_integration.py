"""The new S1 task signal reaches both collectors without altering S0 rewards."""

from dataclasses import replace
import json

import numpy as np
import pytest

from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training.act_ppo import ActPpoRolloutBuffer, ActPpoTrainer, ActPpoValueNetwork
from so101_capstone.training.act_ppo_gazebo import _act_ppo_config, _critic_context, _execution_signature
from so101_capstone.training.residual_ppo import model_state_sha256
from test_act_ppo import TinyActPolicy
from test_stage_process_integration import (
    DT, PROJECT, REWARD, add_transition, core_for, reset_core, small_ppo_config,
    step_core, training_config,
)


V2_PROFILE = "act_ppo_stages_s1_progress_v2.json"
V3_PROFILE = "act_ppo_stages_s1_progress.json"
PROFILES = (V2_PROFILE, V3_PROFILE)
# Recorded in the completed schema-2 run's serialized execution contract.
V2_PROCESS_HASH = "7279879690482889d51d857e61e27d689b5898eb9ac4996a2020c65ab13936e1"
V2_STAGES_HASH = "f9e59053b07b39c979e5641fadda5e884b82069a303524504fe5ab13cbf2e3cf"


def progress_stages(profile=V3_PROFILE):
    return StageTrainingConfig.load(PROJECT / "config" / profile)


def test_new_defaults_preserve_all_stage_success_thresholds_and_old_config():
    old = StageTrainingConfig.load(PROJECT / "config/act_ppo_stages_dense.json")
    new = progress_stages()
    assert replace(new, process_reward=old.process_reward) == old
    assert old.process_reward.context_dim == 12
    assert new.process_reward.context_dim == 14
    assert old.process_reward.schema_version == 1
    previous = progress_stages(V2_PROFILE)
    assert previous.process_reward.schema_version == 2
    assert new.process_reward.schema_version == 3
    assert replace(new, process_reward=previous.process_reward) == previous
    script = (PROJECT / "scripts/finetune_act_ppo_so101.sh").read_text()
    assert 'STAGE_CONFIG="${STAGE_CONFIG:-$PROJECT_ROOT/config/act_ppo_stages_s1_lift_approach.json}"' in script


def test_preserved_schema_two_profile_matches_recorded_hashes_and_roundtrips():
    previous = progress_stages(V2_PROFILE)
    payload = previous.to_dict()
    restored = StageTrainingConfig.from_dict(json.loads(json.dumps(payload)))
    assert restored == previous
    assert restored.to_dict() == payload
    assert previous.config_hash == restored.config_hash == V2_STAGES_HASH
    assert previous.process_reward.config_hash == V2_PROCESS_HASH
    assert previous.process_reward.context_dim == 14
    assert previous.process_reward.context_schema_version == 2
    assert progress_stages().config_hash != V2_STAGES_HASH


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("stage", ["S0", "S1"])
def test_new_history_and_stage_group_reach_critic_batch_and_resume_contract(tmp_path, stage, profile):
    stages = progress_stages(profile)
    config = training_config(tmp_path, stages, stage)
    core = core_for(stage, stages)
    _, initial = reset_core(core, stage)
    t = DT if stage == "S0" else .06 + DT
    _, reward, terminated, truncated, info = step_core(core, t, 4, stage=stage)
    for data in (initial, info):
        assert len(data["critic_task_context"]) == config.critic_task_context_dim == 140
        assert data["task_context_schema_version"] == 3 + stages.process_reward.context_schema_version
        assert data["stage_process_context_schema_version"] == stages.process_reward.context_schema_version
        assert data["critic_task_context"][-14] == (stage == "S1")
        if stage == "S0":
            np.testing.assert_array_equal(data["critic_task_context"][-14:], np.zeros(14))
    if stage == "S1":
        np.testing.assert_array_equal(info["critic_task_context"][-14:], core._stage_process_reward.critic_context())
        assert "s1_best_distance" in info["reward_components"]
        assert "s1_distance_remaining" in info["reward_components"]
    ppo = _act_ppo_config(config, 1)
    assert ppo.advantage_group_context_index == 126
    signature = _execution_signature(config, REWARD)
    assert signature["stage_process_reward"]["context_schema"] == stages.process_reward.context_schema_version
    assert signature["stage_process_reward"]["context_dim"] == 14
    assert signature["task_context_schema"] == info["task_context_schema_version"]
    assert signature["stage_advantage_normalization"] == "binary_stage_actor_only_v1"
    contexts = _critic_context(info, np.zeros((1, 6)), 0, 1, config)
    buffer = ActPpoRolloutBuffer(small_ppo_config(config))
    add_transition(buffer, contexts, reward=reward, terminated=terminated)
    batch = buffer.batch("cpu")
    assert batch.critic_task_context.shape == (1, 140)
    # A one-sample group cannot supply relative actor credit, but must keep its
    # value target and nonterminal bootstrap intact.
    assert batch.advantages[0].item() == 0
    assert batch.returns[0].item() == pytest.approx(reward + REWARD.gamma * 7, abs=1e-5)
    assert not terminated and not truncated


@pytest.mark.parametrize("profile", PROFILES)
def test_s0_rewards_and_success_are_unchanged_for_the_whole_five_seconds(profile):
    old_stages = StageTrainingConfig.load(PROJECT / "config/act_ppo_stages_dense.json")
    old, new = core_for("S0", old_stages), core_for("S0", progress_stages(profile))
    reset_core(old, "S0")
    reset_core(new, "S0")
    for tick in range(1, 151):
        offset, velocity = ((.1, .2) if tick <= 10 else (0., 0.))
        previous = step_core(old, tick * DT, tick + 1, stage="S0", offset=offset, velocity=velocity)
        current = step_core(new, tick * DT, tick + 1, stage="S0", offset=offset, velocity=velocity)
        assert current[1:4] == previous[1:4]
        assert current[4]["reward_components"] == previous[4]["reward_components"]
        np.testing.assert_array_equal(current[4]["critic_task_context"][:-14], previous[4]["critic_task_context"][:-12])
    assert current[4]["stage_success"] and current[3] and not current[2]


@pytest.mark.parametrize("profile", PROFILES)
def test_old_reward_contract_remains_global_and_cannot_mix_new_rollout(tmp_path, profile):
    old = training_config(tmp_path, StageTrainingConfig.load(PROJECT / "config/act_ppo_stages_dense.json"), "old")
    new = training_config(tmp_path, progress_stages(profile), "new")
    assert _act_ppo_config(old, 1).advantage_group_context_index is None
    signature = _execution_signature(old, REWARD)
    assert signature["task_context_dim"] == 138
    assert signature["task_context_schema"] == 4
    assert "stage_advantage_normalization" not in signature
    assert signature != _execution_signature(new, REWARD)
    with pytest.raises(ValueError, match="configurations differ"):
        ActPpoRolloutBuffer(small_ppo_config(new)).extend(ActPpoRolloutBuffer(small_ppo_config(old)))


@pytest.mark.parametrize("source_profile,target_profile", [(V2_PROFILE, V3_PROFILE), (V3_PROFILE, V2_PROFILE)])
def test_shape_compatible_reward_versions_cannot_resume_each_other(tmp_path, source_profile, target_profile):
    source = training_config(tmp_path, progress_stages(source_profile), "source")
    target = training_config(tmp_path, progress_stages(target_profile), "target")
    source_ppo, target_ppo = small_ppo_config(source), small_ppo_config(target)
    # Both versions use 140 critic inputs and the same advantage-group index.
    # The environment execution signature must reject incompatible rewards.
    assert source_ppo == target_ppo
    source_signature = _execution_signature(source, REWARD)
    target_signature = _execution_signature(target, REWARD)
    assert source_signature != target_signature

    def trainer(config, signature):
        return ActPpoTrainer(TinyActPolicy(chunk_size=1), ActPpoValueNetwork(config),
                            config=config, execution_signature=signature)

    original = trainer(source_ppo, source_signature)
    path = original.save_checkpoint(tmp_path / "training_state.pt")
    matching = trainer(source_ppo, source_signature)
    matching.load_checkpoint(path)
    assert matching.execution_signature_sha256 == original.execution_signature_sha256
    assert model_state_sha256(matching.policy) == model_state_sha256(original.policy)
    assert model_state_sha256(matching.critic) == model_state_sha256(original.critic)

    incompatible = trainer(target_ppo, target_signature)
    modules = (incompatible.policy, incompatible.critic, incompatible.exploration)
    before = tuple(model_state_sha256(module) for module in modules)
    with pytest.raises(ValueError, match="execution signature"):
        incompatible.load_checkpoint(path)
    assert tuple(model_state_sha256(module) for module in modules) == before


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("clear", ["task", "stage"])
def test_leaving_stage_mode_clears_versioned_history(clear, profile):
    core = core_for("S1", progress_stages(profile))
    reset_core(core, "S1")
    if clear == "task":
        core.configure_task("pick_place")
    else:
        core.configure_training_stage(None)
    _, info = reset_core(core, "S1")
    assert len(info["critic_task_context"]) == 126
    assert "stage_process_context_schema_version" not in info
    assert core._stage_process_config is None
