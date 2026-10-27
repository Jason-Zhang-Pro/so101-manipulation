"""Read-only six-stage evaluation plans and their opt-in command-line entry."""
from __future__ import annotations

import builtins
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.training import act_stage_evaluation as evaluation


REPOSITORY = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path):
    project = tmp_path / "project with spaces"
    config = project / "config"
    config.mkdir(parents=True)
    for name in ("act_ppo_stages_s0_s5.json", "pick_place_reward.json"):
        shutil.copyfile(REPOSITORY / "config" / name, config / name)
    run = project / "output_simulation" / "finetuned" / "source_run"
    checkpoint = run / "checkpoints" / "update_000005" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    model = {
        "type": "act",
        "chunk_size": 15,
        "n_action_steps": 1,
        "temporal_ensemble_coeff": .01,
        "action_bounds_lower": [-100.] * 5 + [0.],
        "action_bounds_upper": [100.] * 6,
    }
    _write(checkpoint / "config.json", model)
    (checkpoint / "model.safetensors").write_bytes(b"not real weights: planning must not load")
    return SimpleNamespace(
        project=project, checkpoint=checkpoint, run=run, model=model,
        output=tmp_path / "evaluation",
        stages=_read(config / "act_ppo_stages_s0_s5.json"),
        reward=_read(config / "pick_place_reward.json"),
    )


def _plan(a, **kwargs):
    kwargs.setdefault("output_directory", a.output)
    return evaluation.build_plan(a.checkpoint, project_root=a.project, **kwargs)


def _assisted_signature():
    return {
        "schema": 1, "parent_model": "so101", "parent_link": "gripper_link",
        "child_link": "cube_link", "release_delta_rad": math.radians(2),
        "release_reference": "measured_confirmation", "release_requires_measured_opening": True,
        "confirmation": "real_contact_grasp", "recorded_angles": "command_and_measured",
        "context_dim": 6,
    }


def _save_six_stage(a, *, with_run=False, assisted=True, persistent=True):
    stages = dict(a.stages, s0_budget_s=6., s2_budget_s=12.)
    reward = dict(a.reward, gamma=.99)
    _write(a.checkpoint / "resolved_stages.json", stages)
    _write(a.checkpoint / "resolved_reward.json", reward)
    if with_run:
        signature = {}
        if assisted:
            signature["assisted_grasp"] = _assisted_signature()
            if persistent:
                signature["stage_grasp_attachment"] = {
                    "schema": 1, "persistent_stages": ["S2", "S3"],
                }
        _write(a.run / "config.json", {
            "gazebo": {}, "resolved_stages": stages, "resolved_reward": reward,
            "execution_signature": signature,
            "multiworld": {"worlds": [{"ros_domain_id": 20}, {"ros_domain_id": 21}]},
        })


def test_planning_reads_only_configs_without_loading_weights_starting_world_or_writing(
        artifacts, monkeypatch):
    a = artifacts
    before = {path: path.read_bytes() for path in a.project.rglob("*") if path.is_file()}
    original_open, original_import = Path.open, builtins.__import__

    def checked_open(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+"), "planning tried to write a file"
        assert path.name not in {"model.safetensors", "training_state.pt"}, "planning read weights"
        return original_open(path, mode, *args, **kwargs)

    def checked_import(name, *args, **kwargs):
        assert not name.endswith(("act_checkpoint_loader", "act_ppo_multiworld_worker"))
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("planning tried to create a directory, process, or execute a policy")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", checked_open)
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(builtins, "__import__", checked_import)
        patch.setattr(subprocess, "Popen", forbidden)
        patch.setattr(evaluation, "execute_plan", forbidden)
        plan = _plan(a)
    assert not a.output.exists()
    assert before == {path: path.read_bytes() for path in a.project.rglob("*") if path.is_file()}
    assert plan["stages"] == ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert plan["total_episode_count"] == 6
    assert plan["policy_hz"] == 30
    assert plan["gazebo_gui"] is True
    assert plan["config_source"] == "current_project"
    assert plan["checkpoint_config"] == a.model
    assert plan["execution"] == {
        "chunk_size": 15, "n_action_steps": 1, "temporal_ensemble_coeff": .01,
        "chunk_schedule": "temporal_ensemble_every_step",
    }
    assert all(plan[key] is False for key in (
        "exploration_enabled", "optimizer_enabled", "image_augmentation_enabled",
        "environment_randomization_enabled",
    ))


def test_subset_order_episode_counts_fixed_scene_and_seed_wrapping(artifacts):
    plan = _plan(artifacts, stages=("s5", "s0", "s2"), episodes_per_stage=2, seed=2**32 - 1)
    assert plan["stages"] == ["S5", "S0", "S2"]
    assert plan["total_episode_count"] == 6
    assert [case["name"] for case in plan["cases"]] == [
        f"{stage}/episode_{episode:03d}" for stage in ("S5", "S0", "S2") for episode in (1, 2)
    ]
    assert [case["scene_seed"] for case in plan["cases"]] == [2**32 - 1, 0] * 3
    assert [case["task_mode"] for case in plan["cases"]] == [
        "pick_place", "pick_place", "idle_empty", "idle_empty", "pick_place", "pick_place",
    ]


@pytest.mark.parametrize("stages", [(), ("S6",), ("S0", "S0"), ("s2", "S2"), ("S0,S1",)])
def test_rejects_empty_duplicate_or_unknown_stages(artifacts, stages):
    with pytest.raises(ValueError, match="distinct stages"):
        _plan(artifacts, stages=stages)
    assert not artifacts.output.exists()


def test_saved_six_stage_rules_and_reward_are_preferred(artifacts):
    _save_six_stage(artifacts)
    plan = _plan(artifacts)
    assert plan["config_source"] == "checkpoint"
    assert plan["source_run"] is None
    assert plan["stage_config_path"] == str(artifacts.checkpoint / "resolved_stages.json")
    assert plan["resolved_stages"]["s0_budget_s"] == 6.
    assert plan["resolved_stages"]["s2_budget_s"] == 12.
    assert plan["resolved_reward"]["gamma"] == .99
    assert plan["assisted_grasp_enabled"] is True
    assert any("metadata unavailable" in warning for warning in plan["warnings"])


def test_old_s0_s1_export_falls_back_to_current_six_stage_rules_and_reward(artifacts):
    old = dict(artifacts.stages, enable_later_stage_context=False, s0_budget_s=9.)
    _write(artifacts.checkpoint / "resolved_stages.json", old)
    _write(artifacts.checkpoint / "resolved_reward.json", dict(artifacts.reward, gamma=.9))
    plan = _plan(artifacts)
    assert plan["config_source"] == "current_project"
    assert plan["resolved_stages"]["s0_budget_s"] == artifacts.stages["s0_budget_s"]
    assert plan["resolved_reward"]["gamma"] == PickPlaceRewardConfig.from_dict(artifacts.reward).gamma
    assert any("No saved six-stage" in warning for warning in plan["warnings"])


def test_explicit_configs_override_checkpoint_rules(artifacts):
    _save_six_stage(artifacts, with_run=True)
    plan = _plan(
        artifacts, stage_config=artifacts.project / "config/act_ppo_stages_s0_s5.json",
        reward_config=artifacts.project / "config/pick_place_reward.json", assisted_grasp=False,
    )
    assert plan["config_source"] == "explicit"
    assert plan["resolved_stages"]["s0_budget_s"] == 5.
    assert plan["resolved_reward"]["gamma"] == PickPlaceRewardConfig.from_dict(artifacts.reward).gamma
    assert plan["assisted_grasp_enabled"] is False


@pytest.mark.parametrize("assisted", [False, True])
def test_source_run_inherits_physics_without_relying_on_ambient_environment(
        artifacts, monkeypatch, assisted):
    _save_six_stage(artifacts, with_run=True, assisted=assisted)
    monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", str(not assisted).lower())
    plan = _plan(artifacts)
    assert plan["source_run"] == str(artifacts.run)
    assert plan["assisted_grasp_enabled"] is assisted
    assert plan["physics_mode"] == ("rigid_grasp_assisted" if assisted else "unassisted")
    assert not any("predates" in warning for warning in plan["warnings"])
    overridden = _plan(artifacts, assisted_grasp=not assisted)
    assert overridden["assisted_grasp_enabled"] is not assisted


def test_older_assisted_run_and_randomized_run_are_explicitly_described(artifacts):
    _save_six_stage(artifacts, with_run=True, persistent=False)
    source = _read(artifacts.run / "config.json")
    source["execution_signature"]["environment_randomization"] = {"enabled": True}
    _write(artifacts.run / "config.json", source)
    plan = _plan(artifacts)
    assert any("predates persistent S2/S3" in warning for warning in plan["warnings"])
    assert any("physical randomization" in warning for warning in plan["warnings"])
    assert plan["environment_randomization_enabled"] is False


def test_final_run_export_inherits_unassisted_physics_and_protects_source_domain(artifacts):
    final_export = artifacts.run / "pretrained_model"
    shutil.move(str(artifacts.checkpoint), final_export)
    artifacts.checkpoint = final_export
    _save_six_stage(artifacts, with_run=True, assisted=False)
    plan = _plan(artifacts)
    assert plan["source_run"] == str(artifacts.run)
    assert plan["config_source"] == "checkpoint"
    assert plan["assisted_grasp_enabled"] is False
    assert plan["resolved_stages"]["s2_budget_s"] == 12.
    with pytest.raises(ValueError, match="overlaps"):
        _plan(artifacts, ros_domain_id=20)
    with pytest.raises(ValueError, match="new directory"):
        _plan(artifacts, output_directory=artifacts.run / "evaluation")


def test_unsupported_saved_assisted_grasp_contract_requires_explicit_resolution(artifacts):
    _save_six_stage(artifacts, with_run=True)
    source = _read(artifacts.run / "config.json")
    source["execution_signature"]["assisted_grasp"]["release_reference"] = "old_command_angle"
    _write(artifacts.run / "config.json", source)
    with pytest.raises(ValueError, match="Unsupported assisted-grasp"):
        _plan(artifacts)
    assert _plan(artifacts, assisted_grasp=False)["assisted_grasp_enabled"] is False


@pytest.mark.parametrize("mismatch", ["stages", "reward"])
def test_source_run_rejects_mixed_checkpoint_artifacts(artifacts, mismatch):
    _save_six_stage(artifacts, with_run=True)
    source = _read(artifacts.run / "config.json")
    key = "resolved_stages" if mismatch == "stages" else "resolved_reward"
    source[key]["s0_budget_s" if mismatch == "stages" else "gamma"] = .123
    _write(artifacts.run / "config.json", source)
    with pytest.raises(ValueError, match="differ"):
        _plan(artifacts)


@pytest.mark.parametrize("domain", [20, 21])
def test_source_training_domain_cannot_be_reused(artifacts, domain):
    _save_six_stage(artifacts, with_run=True)
    with pytest.raises(ValueError, match="overlaps"):
        _plan(artifacts, ros_domain_id=domain)


@pytest.mark.parametrize("kwargs", [
    {"episodes_per_stage": 0}, {"episodes_per_stage": True}, {"episodes_per_stage": 1.5},
    {"ros_domain_id": -1}, {"ros_domain_id": 233}, {"ros_domain_id": True},
    {"seed": -1}, {"seed": 2**32}, {"seed": False},
    {"startup_timeout_s": 0}, {"startup_timeout_s": float("nan")},
    {"preparation_timeout_s": float("inf")}, {"preparation_timeout_s": True},
    {"target_position": (.26, 0.)}, {"target_position": (.26, float("nan"), .768)},
    {"target_position": (.26, 0., float("inf"))}, {"assisted_grasp": "false"},
])
def test_invalid_runtime_settings_fail_before_creating_output(artifacts, kwargs):
    with pytest.raises(ValueError):
        _plan(artifacts, **kwargs)
    assert not artifacts.output.exists()


@pytest.mark.parametrize("change", [
    {"chunk_size": 0}, {"n_action_steps": 16},
    {"temporal_ensemble_coeff": .01, "n_action_steps": 2},
    {"temporal_ensemble_coeff": float("nan")},
])
def test_invalid_act_temporal_settings_are_not_silently_rewritten(artifacts, change):
    _write(artifacts.checkpoint / "config.json", dict(artifacts.model, **change))
    before = (artifacts.checkpoint / "config.json").read_bytes()
    with pytest.raises(ValueError, match="ACT"):
        _plan(artifacts)
    assert (artifacts.checkpoint / "config.json").read_bytes() == before


def test_te_off_fixed_chunk_execution_and_native_bounds_are_preserved(artifacts):
    model = dict(artifacts.model, temporal_ensemble_coeff=None, n_action_steps=3)
    _write(artifacts.checkpoint / "config.json", model)
    plan = _plan(artifacts)
    assert plan["execution"]["chunk_schedule"] == "fixed_n_action_steps"
    assert plan["execution"]["n_action_steps"] == 3
    assert plan["execution"]["chunk_size"] == 15
    assert plan["checkpoint_config"]["action_bounds_lower"] == model["action_bounds_lower"]
    assert plan["checkpoint_config"]["action_bounds_upper"] == model["action_bounds_upper"]
    assert _read(artifacts.checkpoint / "config.json") == model


def test_budget_overrides_preserve_s0_s5_link_and_noninteger_durations(artifacts):
    plan = _plan(artifacts, budget_overrides={
        "s0_budget_s": 7., "s1_budget_s": 8., "s2_budget_s": 9.1,
        "s3_budget_s": 11., "s4_budget_s": 12.,
    })
    assert [case["budget_s"] for case in plan["cases"]] == [7., 8., 9.1, 11., 12., 7.]
    assert all(case["maximum_steps"] >= case["budget_s"] * 30 for case in plan["cases"])
    assert _read(artifacts.project / "config/act_ppo_stages_s0_s5.json") == artifacts.stages


@pytest.mark.parametrize("override", [
    {"s5_budget_s": 8.}, {"s2_budget_s": 0.}, {"s3_budget_s": float("inf")},
    {"s4_budget_s": True}, {"s0_budget_s": .5}, {"s1_budget_s": .01},
])
def test_invalid_budgets_and_holds_outside_budget_are_rejected(artifacts, override):
    with pytest.raises(ValueError):
        _plan(artifacts, budget_overrides=override)


@pytest.mark.parametrize("kind", ["existing", "checkpoint", "inside_checkpoint", "source_run", "inside_run"])
def test_output_must_be_new_and_outside_source_artifacts(artifacts, kind):
    _save_six_stage(artifacts, with_run=True)
    destination = {
        "existing": artifacts.output,
        "checkpoint": artifacts.checkpoint,
        "inside_checkpoint": artifacts.checkpoint / "eval",
        "source_run": artifacts.run,
        "inside_run": artifacts.run / "eval",
    }[kind]
    if kind == "existing":
        destination.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        _plan(artifacts, output_directory=destination)


def test_default_output_is_unique_but_not_created(artifacts):
    first = _plan(artifacts, output_directory=None)
    second = _plan(artifacts, output_directory=None)
    assert first["output_directory"] != second["output_directory"]
    assert first["gz_partition"] != second["gz_partition"]
    assert not Path(first["output_directory"]).exists()
    assert not Path(second["output_directory"]).exists()


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors"])
def test_missing_required_checkpoint_file_is_reported(artifacts, missing):
    (artifacts.checkpoint / missing).unlink()
    with pytest.raises(FileNotFoundError):
        _plan(artifacts)


def test_non_act_checkpoint_is_rejected(artifacts):
    _write(artifacts.checkpoint / "config.json", dict(artifacts.model, type="diffusion"))
    with pytest.raises(ValueError, match="ACT pretrained_model"):
        _plan(artifacts)


@pytest.mark.parametrize("invalid", [{"control_hz": 20.}, {"grasp_evidence_mode": "geometry"}])
def test_reward_requires_30hz_and_real_contact_evidence(artifacts, invalid):
    path = artifacts.project / "config/pick_place_reward.json"
    _write(path, dict(artifacts.reward, **invalid))
    with pytest.raises(ValueError):
        _plan(artifacts)


def test_explicit_s0_s1_config_is_not_accepted_as_six_stage_contract(artifacts):
    path = artifacts.project / "config/old_stages.json"
    _write(path, dict(artifacts.stages, enable_later_stage_context=False))
    with pytest.raises(ValueError, match="enable_later_stage_context"):
        _plan(artifacts, stage_config=path)


def _cli_args(a, *extra):
    return ["stage-evaluation", "--checkpoint", str(a.checkpoint),
            "--project-root", str(a.project), "--output-directory", str(a.output), *extra]


def test_cli_default_prints_plan_only_and_accepts_subset_budget_options(artifacts, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _cli_args(
        artifacts, "--stages", "s2", "s5", "--episodes", "2", "--headless",
        "--s0-budget-s", "7", "--s2-budget-s", "12", "--assisted-grasp", "false",
    ))
    monkeypatch.setattr(evaluation, "execute_plan", lambda plan: pytest.fail("dry run executed"))
    evaluation.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["stages"] == ["S2", "S5"] and plan["total_episode_count"] == 4
    assert [case["budget_s"] for case in plan["cases"]] == [12., 12., 7., 7.]
    assert plan["gazebo_gui"] is False and plan["assisted_grasp_enabled"] is False
    assert not artifacts.output.exists()


@pytest.mark.parametrize("complete,evaluated,exit_code", [
    (True, True, None), (False, True, 1), (True, False, 1),
])
def test_cli_execute_requires_explicit_flag_and_reports_incomplete_evaluation(
        artifacts, monkeypatch, complete, evaluated, exit_code):
    monkeypatch.setattr(sys, "argv", _cli_args(artifacts, "--stages", "S3", "--execute"))
    called = []

    def fake_execute(plan):
        called.append(plan)
        return {"complete": complete, "episodes": [{"evaluated": evaluated}]}

    monkeypatch.setattr(evaluation, "execute_plan", fake_execute)
    if exit_code is None:
        evaluation.main()
    else:
        with pytest.raises(SystemExit) as error:
            evaluation.main()
        assert error.value.code == exit_code
    assert len(called) == 1 and called[0]["stages"] == ["S3"]
    assert not artifacts.output.exists()


def test_shell_wrapper_passes_options_and_environment_without_running_real_python(tmp_path):
    project = tmp_path / "wrapper project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "evaluate_act_stages.sh"
    shutil.copyfile(REPOSITORY / "scripts/evaluate_act_stages.sh", wrapper)
    setup = project / "ros2_ws/install/setup.bash"
    setup.parent.mkdir(parents=True)
    setup.write_text("# stub workspace setup\n", encoding="utf-8")
    conda = project / "conda.bash"
    conda.write_text('conda() { test "$1" = activate; }\n', encoding="utf-8")
    ros = project / "ros.bash"
    ros.write_text("# stub ROS setup\n", encoding="utf-8")
    binary = project / "stub_bin"
    binary.mkdir()
    fake_python = binary / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'no_user_site': os.environ.get('PYTHONNOUSERSITE'), "
        "'pythonpath': os.environ.get('PYTHONPATH')}))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = dict(os.environ, EVAL_CONDA_SETUP=str(conda), EVAL_ROS_SETUP=str(ros),
               PATH=str(binary) + os.pathsep + os.environ["PATH"], PYTHONPATH="existing_path")
    subprocess.run(["bash", "-n", str(wrapper)], check=True, capture_output=True, text=True, env=env)
    options = ["--checkpoint", "/a checkpoint/pretrained_model", "--stages", "S2", "S4", "--headless"]
    result = subprocess.run(["bash", str(wrapper), *options], check=True, capture_output=True, text=True, env=env)
    forwarded = json.loads(result.stdout)
    assert forwarded["argv"] == ["-u", "-m", "so101_capstone.training.act_stage_evaluation", *options]
    assert forwarded["cwd"] == str(project)
    assert forwarded["no_user_site"] == "1"
    assert forwarded["pythonpath"] == f"{project}/src:{project}/lerobot/src:existing_path"
