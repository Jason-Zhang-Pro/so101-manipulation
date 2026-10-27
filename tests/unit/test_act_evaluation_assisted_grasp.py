"""Evaluation physics follows its saved plan, not a previous terminal setting."""

from dataclasses import asdict
import json
import math
import os
from types import SimpleNamespace

import pytest

from so101_capstone.training import act_home_reach_evaluation as home
from so101_capstone.training import act_paired_evaluation as paired
from test_act_home_reach_evaluation import artifacts
from test_act_paired_evaluation import source_run


ATTACHMENT_CONTRACT = {
    "schema": 1, "parent_model": "so101", "parent_link": "gripper_link",
    "child_link": "cube_link", "release_delta_rad": math.radians(2.0),
    "release_reference": "measured_confirmation", "release_requires_measured_opening": True,
    "confirmation": "real_contact_grasp", "recorded_angles": "command_and_measured",
    "context_dim": 6,
}
FLAG = "SO101_ASSISTED_GRASP_ENABLED"


def paired_plan(source_run, enabled):
    run, updated, output = source_run
    source = json.loads((run / "config.json").read_text())
    if enabled:
        source["execution_signature"]["assisted_grasp"] = dict(ATTACHMENT_CONTRACT)
        (run / "config.json").write_text(json.dumps(source))
    return paired.build_plan(run, updated, output, project_root=run.parent)


@pytest.mark.parametrize("enabled", [False, True])
def test_paired_plan_records_source_physics_despite_opposite_environment(source_run, monkeypatch, enabled):
    monkeypatch.setenv(FLAG, str(not enabled).lower())
    plan = paired_plan(source_run, enabled)
    assert plan["assisted_grasp_enabled"] is enabled
    assert plan["physics_mode"] == ("rigid_grasp_assisted" if enabled else "unassisted")


@pytest.mark.parametrize("contract", [None, {}, {**ATTACHMENT_CONTRACT, "schema": 2},
                                       {**ATTACHMENT_CONTRACT, "parent_link": "moving_jaw"},
                                       {**ATTACHMENT_CONTRACT, "release_reference": "command_confirmation"},
                                       {**ATTACHMENT_CONTRACT, "release_requires_measured_opening": False},
                                       {key: value for key, value in ATTACHMENT_CONTRACT.items()
                                        if key not in {"release_reference", "release_requires_measured_opening"}}])
def test_paired_rejects_unsupported_attachment_contract(contract):
    with pytest.raises(ValueError, match="Unsupported assisted-grasp"):
        paired._assisted_grasp_from_signature({"assisted_grasp": contract})


def test_home_reach_plan_is_explicitly_unassisted(artifacts, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    project, checkpoint, output = artifacts
    plan = home.build_plan(checkpoint, project_root=project, output_directory=output)
    assert plan["assisted_grasp_enabled"] is False
    assert plan["physics_mode"] == "unassisted"


def install_execution_stubs(monkeypatch, module, *, expected, fail):
    import torch
    from so101_capstone.policies import act_checkpoint_loader, act_policy_adapter
    from so101_capstone.training import act_ppo_gazebo, act_ppo_multiworld_worker

    observed = []
    def load(*args, **kwargs):
        observed.append(os.environ.get(FLAG))
        assert os.environ[FLAG] == str(expected).lower()
        if fail:
            raise RuntimeError("offline injected load failure")
        return SimpleNamespace(policy=SimpleNamespace(eval=lambda: None), device="cpu",
                               config=SimpleNamespace(n_action_steps=1))

    class Worker:
        def __init__(self, config):
            assert os.environ[FLAG] == str(expected).lower()
            self.config = config
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def start(self):
            return {"type": "ready"}
        def stop_world(self):
            return {"type": "world_stopped", "world_index": 0}

    monkeypatch.setattr(module, "assert_resources_unused", lambda plan: None)
    monkeypatch.setattr(act_checkpoint_loader, "load_act_checkpoint", load)
    monkeypatch.setattr(act_policy_adapter, "ActPolicyAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(act_ppo_multiworld_worker, "GazeboWorkerClient", Worker)
    monkeypatch.setattr(act_ppo_gazebo, "_configure_act_replanning", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(module, "_run_episode", lambda *args, **kwargs: {
        "complete_timeline": True, "minimum_target_distance_m": .01})
    return observed


@pytest.mark.parametrize("kind,source_enabled", [("paired", False), ("paired", True), ("home", False)])
@pytest.mark.parametrize("original", [None, "true", "false"])
@pytest.mark.parametrize("fail", [False, True])
def test_execute_binds_physics_and_restores_environment_on_success_or_failure(
    monkeypatch, source_run, artifacts, kind, source_enabled, original, fail,
):
    from so101_capstone.environments.simulation_step_coordinator import SimulationStepCoordinatorConfig
    if original is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, original)
    if kind == "paired":
        module = paired
        plan = paired_plan(source_run, source_enabled)
        plan["execution_signature"]["step_coordinator_config"] = asdict(SimulationStepCoordinatorConfig())
        plan["cases"] = [plan["cases"][0]]
        plan["variants"] = [plan["variants"][0]]
    else:
        module = home
        project, checkpoint, output = artifacts
        plan = home.build_plan(checkpoint, project_root=project, output_directory=output, device="cpu")
    observed = install_execution_stubs(monkeypatch, module, expected=source_enabled, fail=fail)
    if fail:
        with pytest.raises(RuntimeError, match="offline injected load failure"):
            module.execute_plan(plan)
    else:
        result = module.execute_plan(plan)
        assert result["assisted_grasp_enabled"] is source_enabled
        assert result["physics_mode"] == plan["physics_mode"]
    assert observed == [str(source_enabled).lower()]
    assert os.environ.get(FLAG) == original


def test_home_reach_refuses_a_manually_changed_assisted_plan(artifacts, monkeypatch):
    project, checkpoint, output = artifacts
    plan = home.build_plan(checkpoint, project_root=project, output_directory=output)
    plan["assisted_grasp_enabled"] = True
    with pytest.raises(ValueError, match="unassisted physics only"):
        home.execute_plan(plan)
    assert not output.exists()


def test_paired_refuses_plan_and_source_physics_mismatch(source_run):
    plan = paired_plan(source_run, True)
    plan["assisted_grasp_enabled"] = False
    with pytest.raises(ValueError, match="differs from the saved source contract"):
        paired.execute_plan(plan)
    assert not source_run[2].exists()
