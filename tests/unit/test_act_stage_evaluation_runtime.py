"""Exercise independent stage inference and persistence without Gazebo/models."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training import act_stage_evaluation as evaluation


def read(path):
    return json.loads(path.read_text())


def observation(index):
    return SimpleNamespace(joint_positions=np.zeros(6), joint_velocities=np.zeros(6), step=index)


class Worker:
    def __init__(self, stage="S2", result="SUCCESS", *, failure_at=None, bad_start=False, spawn_ok=True):
        self.stage, self.result = stage, result
        self.failure_at, self.bad_start, self.spawn_ok = failure_at, bad_start, spawn_ok
        self.calls = []
        self.index = 0

    def info(self):
        return dict(training_stage=self.stage, start_valid=True, episode_step=self.index,
                    stage_elapsed_s=self.index / 30, sim_time_s=50 + self.index / 30,
                    frame_zero=self.index == 0, evidence_valid=True, task_evidence_available=True,
                    target_motion_valid=self.stage != "S0",
                    physical_target_present=self.stage != "S0", distance_m=.1 - self.index * .01,
                    stage_result=self.result if self.index >= 3 else "RUNNING",
                    stage_success=self.index >= 3 and self.result == "SUCCESS",
                    stage_metrics={}, reward_components={})

    def receive(self, *, timeout_s):
        self.calls.append(("receive", timeout_s))
        if self.response == "fail":
            raise RuntimeError("lost feedback")
        return self.response

    def prepare_episode_async(self, index, *, task_mode, scene_seed):
        self.calls.append(("prepare", task_mode, scene_seed))
        self.response = dict(type="episode_prepared", info={"sim_time_s": 7}, observation={})

    def spawn_target_async(self, position, quaternion):
        self.calls.append(("spawn", position))
        self.response = dict(type="target_spawned", success=self.spawn_ok)

    def start_training_stage_async(self, stage, config):
        self.calls.append(("handoff", stage))
        assert stage == self.stage
        info = self.info()
        if self.bad_start:
            info["episode_step"] = 5
        self.response = dict(type="stage_started", info=info, observation={"step": 0},
                             initialization=dict(valid=True, preparation_steps=560))

    def observation_from_metadata(self, metadata):
        return observation(metadata["step"])

    def step_async(self, command):
        self.index += 1
        self.calls.append(("step", self.index))
        self.response = "fail" if self.index == self.failure_at else dict(
            type="step_result", observation={"step": self.index}, reward=1,
            terminated=self.index >= 3 and self.result == "FAILURE",
            truncated=self.index >= 3 and self.result in {"SUCCESS", "TIMEOUT"},
            info=self.info())


class Adapter:
    def __init__(self, *args, **kwargs):
        self.resets = 0
        self.selected = []

    def reset(self):
        self.resets += 1

    def select_action(self, obs):
        self.selected.append(obs.step)
        return SimpleNamespace(raw_policy_positions=np.zeros(6), clipped_policy_positions=np.zeros(6),
                               action=SimpleNamespace(joint_targets=np.zeros(6)))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "_save_frame", lambda *args: None)
    plan = dict(resolved_reward=PickPlaceRewardConfig().to_dict(),
                resolved_stages=StageTrainingConfig(enable_later_stage_context=True).to_dict(),
                startup_feedback_timeout_s=60, preparation_timeout_s=600,
                target_position_world_m=[.26, 0, .768], target_quaternion_xyzw=[0, 0, 0, 1])
    case = dict(name="S2/episode_001", stage="S2", episode_index=1, task_mode="pick_place",
                scene_seed=1009, budget_s=8, maximum_steps=242)
    return plan, case, tmp_path / "case"


@pytest.mark.parametrize("stage", evaluation.STAGES)
@pytest.mark.parametrize("result", ["SUCCESS", "TIMEOUT", "FAILURE"])
def test_all_stage_policy_starts_after_preparation_and_uses_authoritative_result(setup, stage, result):
    plan, case, destination = setup
    case.update(stage=stage, task_mode="idle_empty" if stage == "S0" else "pick_place")
    worker, adapter = Worker(stage, result), Adapter()
    summary = evaluation._run_episode(worker, adapter, case, plan, destination)
    rows = [json.loads(line) for line in (destination / "steps.jsonl").read_text().splitlines()]
    assert summary["stage_result"] == result and summary["evaluated"]
    assert summary["success"] == (result == "SUCCESS")
    assert summary["completed_steps"] == 3 and summary["total_reward"] == 3
    assert summary["home"]["final_joint_abs_error_deg"] is not None
    if stage != "S0":
        assert summary["target"]["minimum_distance_m"] == pytest.approx(.07)
    assert rows[-1]["elapsed_s"] == pytest.approx(.1)  # No 560 preparation steps in this time.
    assert [row["step_index"] for row in rows] == [1, 2, 3]
    assert adapter.resets == 1 and adapter.selected == [0, 1, 2]
    assert len([call for call in worker.calls if call[0] == "spawn"]) == (stage != "S0")
    assert ("receive", 600) in worker.calls
    assert read(destination / "initialization.json")["initialization"]["preparation_steps"] == 560
    assert read(destination / "summary.json") == summary


@pytest.mark.parametrize("failure", ["prepare_certificate", "spawn", "feedback", "no_stage_terminal"])
def test_failure_preserves_confirmed_records_and_never_calls_it_success(setup, failure):
    plan, case, destination = setup
    worker = Worker(bad_start=failure == "prepare_certificate", spawn_ok=failure != "spawn",
                    failure_at=3 if failure == "feedback" else None,
                    result="RUNNING" if failure == "no_stage_terminal" else "SUCCESS")
    case["maximum_steps"] = 4
    with pytest.raises(RuntimeError):
        evaluation._run_episode(worker, Adapter(), case, plan, destination)
    summary = read(destination / "summary.json")
    assert summary["stage_result"] == "ERROR" and not summary["evaluated"] and not summary["success"]
    expected = 2 if failure == "feedback" else 4 if failure == "no_stage_terminal" else 0
    assert summary["completed_steps"] == expected
    assert len((destination / "steps.jsonl").read_text().splitlines()) == expected
    assert (destination / "failure.json").is_file()


def test_wrong_stage_after_handoff_cannot_count_success(setup):
    plan, case, destination = setup
    worker = Worker()
    step = worker.step_async

    def wrong(command):
        step(command)
        worker.response["info"]["training_stage"] = "S3"

    worker.step_async = wrong
    with pytest.raises(RuntimeError, match="requested stage"):
        evaluation._run_episode(worker, Adapter(), case, plan, destination)
    assert read(destination / "summary.json")["stage_result"] == "ERROR"


def test_execute_closes_after_persistence_restores_environment_and_retries_next_case(setup, monkeypatch, tmp_path):
    import torch
    from so101_capstone.policies import act_checkpoint_loader, act_policy_adapter
    from so101_capstone.training import act_ppo_multiworld_worker

    plan, case, _ = setup
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fake checkpoint")
    output = tmp_path / "output"
    second = dict(case, name="S3/episode_001", stage="S3")
    plan.update(project_root=str(tmp_path), checkpoint=str(checkpoint), output_directory=str(output),
                device="cpu", gazebo_gui=True, ros_domain_id=125, gz_partition="stage-test",
                assisted_grasp_enabled=True, physics_mode="rigid_grasp_assisted",
                protocol="independent_s0_s5_policy", stages=["S2", "S3"], cases=[case, second])
    monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", "false")
    monkeypatch.setenv("SO101_IMAGE_AUGMENTATION_ENABLED", "true")
    monkeypatch.delenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", raising=False)
    monkeypatch.setattr(evaluation, "assert_resources_unused", lambda plan: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    policy = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(act_checkpoint_loader, "load_act_checkpoint",
                        lambda *a, **k: SimpleNamespace(policy=policy, device="cpu",
                                                       config=SimpleNamespace(n_action_steps=1)))
    monkeypatch.setattr(act_policy_adapter, "ActPolicyAdapter", Adapter)
    clients = []

    class Client(Worker):
        def __init__(self, config):
            index = len(clients)
            super().__init__(stage=plan["cases"][index]["stage"], failure_at=2 if index == 0 else None)
            self.config, self.closed = config, False
            clients.append(self)
            assert config.evaluation_mode is False
            assert config.gazebo_gui is True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            assert self.closed

        def start(self):
            return dict(type="ready")

        def stop_world(self):
            directory = output / plan["cases"][clients.index(self)]["name"]
            assert (directory / "summary.json").is_file()
            assert (directory / "steps.jsonl").is_file()
            self.closed = True
            return dict(type="world_stopped", world_index=0)

    monkeypatch.setattr(act_ppo_multiworld_worker, "GazeboWorkerClient", Client)
    result = evaluation.execute_plan(plan)
    assert result["complete"] and len(clients) == 2
    assert [row["stage_result"] for row in result["episodes"]] == ["ERROR", "SUCCESS"]
    assert all(row["world_cleanup_confirmed"] for row in result["episodes"])
    assert read(output / "summary.json") == result
    import os
    assert os.environ["SO101_ASSISTED_GRASP_ENABLED"] == "false"
    assert os.environ["SO101_IMAGE_AUGMENTATION_ENABLED"] == "true"
    assert "SO101_ENVIRONMENT_RANDOMIZATION_ENABLED" not in os.environ
