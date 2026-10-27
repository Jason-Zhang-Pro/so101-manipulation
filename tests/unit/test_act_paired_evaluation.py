"""Paired evaluator contracts: exact source scenes, isolation and real worker protocol."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.training.act_paired_evaluation import (
    _run_episode, assert_resources_unused, build_plan, summarize_steps,
)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def source_run(tmp_path):
    run = tmp_path / "training"
    initial, updated = tmp_path / "initial", tmp_path / "updated"
    model = {"chunk_size": 15, "n_action_steps": 1, "temporal_ensemble_coeff": .01}
    for path in (initial, updated):
        _write(path / "config.json", model)
        (path / "model.safetensors").write_bytes(b"not_loaded_by_plan")
    reward = {"home_joint_positions_rad": [0.] * 6}
    config = {"gazebo": {"act_checkpoint": str(initial), "act_max_action_steps": None,
                          "reward_profile": "pick_place_v2", "seed": 0, "max_steps_per_episode": 600,
                          "startup_feedback_timeout_s": 30.},
              "resolved_reward": reward,
              "execution_signature": {**model, "environment_randomization": {"enabled": False}, "policy_hz": 30.},
              "multiworld": {"worlds": [{"ros_domain_id": 20}, {"ros_domain_id": 21}]}}
    _write(run / "config.json", config)
    _write(run / "resolved_reward.json", reward)
    modes = ["idle_completed", "idle_empty", "pick_place", "pick_place", "pick_place"]
    for index, mode in enumerate(modes):
        record = {"episode_index": index, "episode_attempt_index": index + 1, "world_index": 1,
                  "task_mode": mode, "valid": True,
                  "target": {"request_completed_step": 2 if mode == "pick_place" else None}}
        if mode == "pick_place":
            record.update(scene_seed=(index + 1) * 1009 + 100003,
                          target_position_world_m=[.24 + index * .001, .02, .768],
                          target_quaternion_xyzw=[0, 0, 0, 1])
        _write(run / "episodes" / f"episode_{index:06d}.json", record)
    return run, updated, tmp_path / "eval"


def test_plan_reuses_exact_seed_delay_pose_and_five_cases_without_creating_output(source_run):
    run, updated, output = source_run
    plan = build_plan(run, updated, output, project_root=run.parent)
    assert plan["total_episode_count"] == 10
    assert not output.exists()
    assert [case["task_mode"] for case in plan["cases"]] == ["idle_empty", "idle_completed"] + ["pick_place"] * 3
    case = plan["cases"][2]
    assert case["scene_seed"] == 103030
    assert case["target_delay_steps"] == 2
    assert case["target_position_world_m"] == [.242, .02, .768]
    assert plan["resolved_reward"] == json.loads((run / "resolved_reward.json").read_text())
    assert not plan["exploration_enabled"] and not plan["optimizer_enabled"]


def test_plan_rejects_training_domain_and_existing_destination(source_run):
    run, updated, output = source_run
    with pytest.raises(ValueError, match="overlaps"):
        build_plan(run, updated, output, project_root=run.parent, ros_domain_id=20)
    output.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        build_plan(run, updated, output, project_root=run.parent)


def test_plan_rejects_reward_drift_and_supports_six_episode_smoke(source_run):
    run, updated, output = source_run
    plan = build_plan(run, updated, output, project_root=run.parent, reach_scenes=1)
    assert plan["total_episode_count"] == 6
    _write(run / "resolved_reward.json", {"changed": True})
    with pytest.raises(ValueError, match="reward differ"):
        build_plan(run, updated, output, project_root=run.parent)


def test_conflict_check_refuses_even_nonmanaged_process_and_does_not_signal():
    plan = {"ros_domain_id": 120, "gz_partition": "unique_eval"}
    assert_resources_unused(plan, [SimpleNamespace(pid=123, environment={"ROS_DOMAIN_ID": "20"})])
    with pytest.raises(RuntimeError, match="PID 123"):
        assert_resources_unused(plan, [SimpleNamespace(pid=123, environment={"ROS_DOMAIN_ID": "120"})])
    with pytest.raises(RuntimeError, match="PID 124"):
        assert_resources_unused(plan, [SimpleNamespace(pid=124, environment={"GZ_PARTITION": "unique_eval"})])


def _observation(step):
    return SimpleNamespace(joint_positions=np.full(6, step * .001), joint_velocities=np.zeros(6),
                           top_image=np.full((3, 4, 3), step % 256, dtype=np.uint8),
                           gripper_image=np.zeros((3, 4, 3), dtype=np.uint8))


class WorkerStub:
    """Use actual worker wire response types, not evaluator-invented response wrappers."""
    def __init__(self):
        self.step = 0
        self.spawned = []
        self.response = None

    def prepare_episode_async(self, attempt, *, task_mode, scene_seed):
        assert (attempt, task_mode, scene_seed) == (3, "pick_place", 103030)
        self.response = {"type": "episode_prepared", "observation": {"step": 0}, "info": {}}

    def receive(self, *, timeout_s):
        assert self.response is not None
        result, self.response = self.response, None
        return result

    def observation_from_metadata(self, metadata):
        return _observation(metadata["step"])

    def spawn_target_async(self, xyz, quat):
        self.spawned.append((self.step, xyz, quat))
        self.response = {"type": "target_spawned", "success": True, "message": "created"}

    def step_async(self, q):
        assert len(q) == 6
        self.step += 1
        info = {"physical_target_present": bool(self.spawned), "evidence_valid": True,
                "distance_m": abs(self.step - 5) * .001 + .015, "grasp_close_gate": 1. if self.step == 5 else 0.,
                "pregrasp_ready": self.step == 5, "task_outcome": "running",
                "critic_task_context": [0.] * 38 + [.01, .02, -.03] + [0.] * 52}
        self.response = {"type": "step_result", "observation": {"step": self.step},
                         "reward": -.01, "terminated": False, "truncated": self.step == 600, "info": info}


class AdapterStub:
    def __init__(self):
        self.reset_count = 0
        self.seen_steps = []

    def reset(self):
        self.reset_count += 1

    def select_action(self, observation):
        self.seen_steps.append(observation.joint_positions[0])
        return SimpleNamespace(raw_policy_positions=np.zeros(6), action=SimpleNamespace(joint_targets=np.zeros(6)))


def test_complete_episode_uses_real_protocol_resets_history_spawns_at_exact_step_and_saves_frames(source_run):
    run, updated, output = source_run
    plan = build_plan(run, updated, output, project_root=run.parent)
    worker, adapter = WorkerStub(), AdapterStub()
    summary = _run_episode(worker, adapter, plan["cases"][2], plan, output / "episode")
    assert worker.spawned == [(2, [.242, .02, .768], [0, 0, 0, 1])]
    assert adapter.reset_count == 1 and len(adapter.seen_steps) == 600
    assert summary["step_count"] == 600 and summary["truncated"]
    assert summary["minimum_target_distance_m"] == .015
    assert summary["nearest_step_index"] == 5
    assert summary["stage_reached"]["precise_reach"]
    assert summary["nearest_geometry"]["aligned_target_minus_grasp_xyz_m"] == [.01, .02, -.03]
    assert len((output / "episode" / "steps.jsonl").read_text().splitlines()) == 600
    for name in ("initial", "nearest", "final"):
        for camera in ("top", "gripper"):
            assert (output / "episode" / f"{name}_{camera}.png").is_file()


def test_preplaced_object_is_not_counted_as_learned_place():
    records = [{"step_index": 1, "joint_positions_after_rad": [0.] * 6,
                "reward": 0., "terminated": False, "truncated": True,
                "info": {"placed_stable": True}}]
    summary = summarize_steps(records, {"home_joint_positions_rad": [0.] * 6}, "idle_completed")
    assert summary["stage_reached"]["placed_stable"]
    assert not summary["learned_place_observed"]
    assert summary["minimum_target_distance_m"] is None


def test_invalid_target_evidence_is_not_used_for_nearest_distance():
    records = [{"step_index": 1, "joint_positions_after_rad": [0.] * 6,
                "reward": 0., "terminated": False, "truncated": True,
                "info": {"physical_target_present": True, "target_name": "cube", "evidence_valid": False,
                         "distance_m": .001}}]
    summary = summarize_steps(records, {"home_joint_positions_rad": [0.] * 6}, "pick_place")
    assert summary["minimum_target_distance_m"] is None


def test_step_failure_preserves_completed_rows_and_records_failure(source_run):
    run, updated, output = source_run
    plan = build_plan(run, updated, output, project_root=run.parent)
    worker = WorkerStub()
    original = worker.step_async
    def step_with_error(q):
        if worker.step == 3:
            worker.response = {"type": "operation_error", "operation": "step", "error_message": "feedback timeout"}
        else:
            original(q)
    worker.step_async = step_with_error
    with pytest.raises(RuntimeError, match="feedback timeout"):
        _run_episode(worker, AdapterStub(), plan["cases"][2], plan, output / "episode")
    assert len((output / "episode" / "steps.jsonl").read_text().splitlines()) == 3
    failure = json.loads((output / "episode" / "failure.json").read_text())
    assert failure["completed_steps"] == 3
    assert not (output / "episode" / "summary.json").exists()


def test_real_privileged_dataclass_survives_initial_step_and_geometry_serialization(source_run):
    from so101_capstone.environments.reach_env_core import ReachPrivilegedState
    from enum import Enum
    class RecordingPhase(Enum):
        APPROACH = "approach"
    privileged = ReachPrivilegedState(
        grasp_position=np.array([.24, .02, .79]),
        grasp_quaternion_xyzw=np.array([0., 0., 0., 1.]),
        grasp_rotation=np.eye(3),
        target_position=np.array([.24, .02, .765]),
        target_quaternion_xyzw=np.array([0., 0., 0., 1.]),
        target_delta=np.array([0., 0., -.025]),
        distance_m=np.float64(.025))
    run, updated, output = source_run
    plan = build_plan(run, updated, output, project_root=run.parent)
    plan["max_steps_per_episode"] = 6
    worker = WorkerStub()
    original = worker.receive
    def receive_real_privileged(*, timeout_s):
        result = original(timeout_s=timeout_s)
        if "info" in result:
            result["info"].update(privileged_state=privileged, recorded_phase=RecordingPhase.APPROACH)
        return result
    worker.receive = receive_real_privileged
    summary = _run_episode(worker, AdapterStub(), plan["cases"][2], plan, output / "episode")
    assert summary["step_count"] == 6
    for filename in ("initial.json", "nearest.json", "final.json", "summary.json"):
        assert json.loads((output / "episode" / filename).read_text())
    initial = json.loads((output / "episode" / "initial.json").read_text())
    assert initial["info"]["privileged_state"]["target_delta"] == [0., 0., -.025]
    assert initial["info"]["recorded_phase"] == "approach"
    row = json.loads((output / "episode" / "steps.jsonl").read_text().splitlines()[0])
    assert row["info"]["privileged_state"]["grasp_rotation"] == np.eye(3).tolist()


@pytest.mark.parametrize("stop_response", [
    None,
    {"type": "operation_error", "operation": "stop_world", "error_message": "cleanup failed"},
    {"type": "shutdown_complete", "world_index": 0},
    {"type": "world_stopped", "world_index": 1},
    {"type": "world_stopped"},
    {"type": "world_stopped", "world_index": 0},
])
def test_execute_requires_matching_world_cleanup_ack_before_counting_episode(
    source_run, monkeypatch, stop_response,
):
    """Legacy workers can return errors/None instead of validating stop responses."""
    from dataclasses import asdict
    import torch
    from so101_capstone.environments.simulation_step_coordinator import SimulationStepCoordinatorConfig
    from so101_capstone.policies import act_checkpoint_loader, act_policy_adapter
    from so101_capstone.training import act_paired_evaluation as evaluation
    from so101_capstone.training import act_ppo_gazebo, act_ppo_multiworld_worker

    run, updated, output = source_run
    plan = build_plan(run, updated, output, project_root=run.parent)
    plan["execution_signature"]["step_coordinator_config"] = asdict(SimulationStepCoordinatorConfig())
    plan["cases"] = [plan["cases"][2]]
    plan["variants"] = [plan["variants"][0]]
    plan["max_steps_per_episode"] = 6
    clients = []

    class LegacyWorker(WorkerStub):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.closed = False
            self.stop_calls = 0
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def start(self):
            return {"type": "ready"}

        def stop_world(self):
            self.stop_calls += 1
            return stop_response

    loaded = SimpleNamespace(policy=SimpleNamespace(eval=lambda: None), device="cpu",
                             config=SimpleNamespace(n_action_steps=1))
    monkeypatch.setattr(evaluation, "assert_resources_unused", lambda plan: None)
    monkeypatch.setattr(act_checkpoint_loader, "load_act_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(act_policy_adapter, "ActPolicyAdapter", lambda *args, **kwargs: AdapterStub())
    monkeypatch.setattr(act_ppo_gazebo, "_configure_act_replanning", lambda *args: None)
    monkeypatch.setattr(act_ppo_multiworld_worker, "GazeboWorkerClient", LegacyWorker)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")

    successful_stop = stop_response == {"type": "world_stopped", "world_index": 0}
    if successful_stop:
        comparison = evaluation.execute_plan(plan)
        assert comparison["complete"] and len(comparison["episodes"]) == 1
        assert not (output / "failure.json").exists()
    else:
        with pytest.raises(RuntimeError, match="Expected"):
            evaluation.execute_plan(plan)
        failure = json.loads((output / "failure.json").read_text())
        assert failure["completed_episodes"] == 0
        assert not (output / "comparison.json").exists()

    # A cleanup failure never discards the already recorded trajectory evidence.
    episode = output / "initial" / "pick_place_0"
    assert json.loads((episode / "summary.json").read_text())["step_count"] == 6
    assert len((episode / "steps.jsonl").read_text().splitlines()) == 6
    assert clients[0].stop_calls == 1 and clients[0].closed
