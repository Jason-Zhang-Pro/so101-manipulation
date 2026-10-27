"""Assisted physics must be explicit and incompatible with unassisted PPO resume."""

from dataclasses import replace
import math
from pathlib import Path
import subprocess

import pytest
import torch

from so101_capstone.training.act_ppo import ActPpoTrainer, ActPpoValueNetwork
from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig, _act_ppo_config, _execution_signature, _resolved_reward,
)
from so101_capstone.training.residual_ppo import model_state_sha256
from test_act_ppo import TinyActPolicy
from test_act_ppo_s0_s5_launcher import project, run_launcher, parse_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_physics_environment(monkeypatch):
    monkeypatch.delenv("SO101_ASSISTED_GRASP_ENABLED", raising=False)
    monkeypatch.setenv("SO101_IMAGE_AUGMENTATION_ENABLED", "false")
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")


def launch_config(stages="s0_s5"):
    return ActPpoGazeboConfig(
        act_checkpoint=Path("/unused/act"), reward_profile="pick_place_v2",
        reward_config=ROOT / "config/pick_place_reward.json", training_stages=stages,
        stage_config=None if stages == "full" else ROOT / "config" / (
            "act_ppo_stages_s0_s5.json" if stages == "s0_s5"
            else "act_ppo_stages_s1_lift_approach.json"),
        final_target_absent_s=0,
        critic_image_height=16, critic_image_width=16, critic_hidden_dim=32,
    )


@pytest.mark.parametrize("stages", ["full", "s0_s1", "s0_s5"])
def test_assisted_context_grows_before_stage_tail_and_signature_is_explicit(monkeypatch, stages):
    natural = launch_config(stages)
    old = _act_ppo_config(natural, 1)
    old_signature = _execution_signature(natural, _resolved_reward(natural))
    monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", "true")
    assisted = replace(natural)
    new = _act_ppo_config(assisted, 1)
    new_signature = _execution_signature(assisted, _resolved_reward(assisted))
    assert new.critic_task_context_dim == old.critic_task_context_dim + 6
    if old.advantage_group_context_index is not None:
        assert new.advantage_group_context_index == old.advantage_group_context_index + 6
    else:
        assert new.advantage_group_context_index is None
    assert new.advantage_group_count == old.advantage_group_count
    assert new_signature.pop("assisted_grasp") == {
        "schema": 1, "parent_model": "so101", "parent_link": "gripper_link",
        "child_link": "cube_link", "release_delta_rad": math.radians(2.0),
        "release_reference": "measured_confirmation", "release_requires_measured_opening": True,
        "confirmation": "real_contact_grasp", "recorded_angles": "command_and_measured",
        "context_dim": 6,
    }
    new_signature["task_context_dim"] -= 6
    if stages == "s0_s5":
        assert new_signature.pop("stage_grasp_attachment") == {
            "schema": 1, "persistent_stages": ["S2", "S3"],
            "persistent_release_mode": "stage_persistent", "opening_commands_detach": False,
            "preparation_release_mode": "measured_opening", "other_stage_release_mode": "measured_opening",
        }
    assert new_signature == old_signature


@pytest.mark.parametrize("disabled", ["false", "0", "no", "off"])
def test_disabled_flag_preserves_absent_flag_contract(monkeypatch, disabled):
    cfg = launch_config()
    expected = _execution_signature(cfg, _resolved_reward(cfg))
    monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", disabled)
    changed = replace(cfg)
    actual = _execution_signature(changed, _resolved_reward(changed))
    assert actual == expected and "assisted_grasp" not in actual


def test_legacy_assisted_training_fails_before_creating_an_untracked_physics_run(monkeypatch):
    natural = ActPpoGazeboConfig(act_checkpoint=Path("/unused/act"))
    assert natural.critic_task_context_dim == 0
    monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", "true")
    with pytest.raises(ValueError, match="Assisted rigid grasp requires pick_place_v2"):
        replace(natural)


@pytest.mark.parametrize("source_assisted", [False, True])
def test_resume_requires_matching_assisted_physics_before_loading_weights(
    monkeypatch, tmp_path, source_assisted,
):
    def make_trainer(enabled):
        monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", str(enabled).lower())
        cfg = launch_config()
        ppo = _act_ppo_config(cfg, 1)
        return ActPpoTrainer(
            TinyActPolicy(chunk_size=1), ActPpoValueNetwork(ppo), config=ppo,
            execution_signature=_execution_signature(cfg, _resolved_reward(cfg)),
        )

    original = make_trainer(source_assisted)
    path = original.save_checkpoint(tmp_path / "training_state.pt")
    matching = make_trainer(source_assisted)
    matching.load_checkpoint(path, map_location="cpu")
    assert model_state_sha256(matching.critic) == model_state_sha256(original.critic)
    incompatible = make_trainer(not source_assisted)
    modules = incompatible.policy, incompatible.critic, incompatible.exploration
    before = tuple(model_state_sha256(module) for module in modules)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="execution signature"):
        incompatible.load_checkpoint(path, map_location="cpu")
    assert tuple(model_state_sha256(module) for module in modules) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_six_stage_launcher_enables_assistance_and_clears_stale_internal_flag(project):
    stale = {"ASSISTED_GRASP_ENABLED": "false", "SO101_ASSISTED_GRASP_ENABLED": "false"}
    printed = parse_config(run_launcher(project, "--print-config", overrides=stale), inherited=stale)
    assert printed["ASSISTED_GRASP_ENABLED"] == "true"
    assert "SO101_ASSISTED_GRASP_ENABLED" not in printed
    result = run_launcher(project, overrides=stale)
    assert result.returncode == 0, result.stderr
    dispatched = dict(item.split("=", 1) for item in
                      result.stdout.removeprefix("TRAINER_STUB\n").split("\0") if item)
    assert dispatched["ASSISTED_GRASP_ENABLED"] == "true"
    assert "SO101_ASSISTED_GRASP_ENABLED" not in dispatched


@pytest.mark.parametrize("old_behavior", ["command_only_release", "extra_grasp_tightening",
                                          "angle_release_in_s2_s3", "command_angle_reward_reference"])
def test_resume_rejects_obsolete_grasp_behavior(monkeypatch, tmp_path, old_behavior):
    monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", "true")
    cfg = launch_config()
    ppo = _act_ppo_config(cfg, 1)
    safe_contract = _execution_signature(cfg, _resolved_reward(cfg))
    old_contract = _execution_signature(cfg, _resolved_reward(cfg))
    if old_behavior == "command_only_release":
        old_contract["assisted_grasp"].pop("release_reference")
        old_contract["assisted_grasp"].pop("release_requires_measured_opening")
    elif old_behavior == "extra_grasp_tightening":
        # Earlier preparation used a fixed extra -2 degrees without recording
        # that behavior in its numeric configuration. Do not silently resume it.
        old_contract.pop("stage_preparation_grasp_hold")
    elif old_behavior == "angle_release_in_s2_s3":
        old_contract.pop("stage_grasp_attachment")
    else:
        old_contract["later_stage_training"]["stages"]["S2"]["attachment_command_reward"]["reference"] = "command_at_attachment"

    def trainer(contract):
        return ActPpoTrainer(
            TinyActPolicy(chunk_size=1), ActPpoValueNetwork(ppo), config=ppo,
            execution_signature=contract,
        )

    previous = trainer(old_contract)
    checkpoint = previous.save_checkpoint(tmp_path / "old_grasp_state.pt")
    current = trainer(safe_contract)
    modules = current.policy, current.critic, current.exploration
    before = tuple(model_state_sha256(module) for module in modules)
    with pytest.raises(ValueError, match="execution signature"):
        current.load_checkpoint(checkpoint, map_location="cpu")
    assert tuple(model_state_sha256(module) for module in modules) == before


@pytest.mark.parametrize("external,internal,expected", [
    (None, "true", "false"), ("true", "false", "true"), ("FALSE", "true", "false"),
])
def test_common_launcher_resolves_flag_and_overwrites_stale_internal_value(external, internal, expected):
    # Execute only the actual setting/validation/export blocks, never conda/ROS/training.
    source = (ROOT / "scripts/finetune_act_ppo_so101.sh").read_text()
    assignment = next(line for line in source.splitlines() if line.startswith("ASSISTED_GRASP_ENABLED="))
    validation_start = source.index('case "${ASSISTED_GRASP_ENABLED,,}" in')
    validation = source[validation_start:source.index("esac", validation_start) + len("esac")]
    export = next(line for line in source.splitlines() if line.startswith("export SO101_ASSISTED_GRASP_ENABLED="))
    env = {"SO101_ASSISTED_GRASP_ENABLED": internal}
    if external is not None:
        env["ASSISTED_GRASP_ENABLED"] = external
    result = subprocess.run(["/bin/bash", "-e", "-c", assignment + "\n" + validation + "\n" +
                             export + '\nprintf "%s" "$SO101_ASSISTED_GRASP_ENABLED"'],
                            env=env, text=True, capture_output=True, check=True, timeout=10)
    assert result.stdout == expected
