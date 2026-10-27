"""The continuous ZERO/HOME/target protocol, without ROS, weights or Gazebo."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.evaluation import home_reach_metrics
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.training import act_home_reach_evaluation as evaluation


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def artifacts(tmp_path):
    project = tmp_path / "project"
    checkpoint = project / "checkpoints" / "test_model"
    checkpoint.mkdir(parents=True)
    model = {"type": "act", "chunk_size": 15, "n_action_steps": 1,
             "temporal_ensemble_coeff": .01}
    (checkpoint / "config.json").write_text(json.dumps(model), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"planning does not load these weights")
    (project / "config").mkdir()
    reward = PickPlaceRewardConfig().to_dict()
    (project / "config" / "pick_place_reward.json").write_text(json.dumps(reward), encoding="utf-8")
    return project, checkpoint, tmp_path / "evaluation"


@pytest.fixture
def plan(artifacts):
    project, checkpoint, output = artifacts
    return evaluation.build_plan(checkpoint, project_root=project, output_directory=output)


def test_plan_preserves_te_and_execution_config_without_creating_output(artifacts):
    project, checkpoint, output = artifacts
    original_config = (checkpoint / "config.json").read_bytes()
    result = evaluation.build_plan(checkpoint, project_root=project, output_directory=output)
    assert result["execution"] == {"chunk_size": 15, "n_action_steps": 1, "temporal_ensemble_coeff": .01}
    assert result["gazebo_gui"] is True
    assert result["initial_pose"] == "zero" and result["task_mode"] == "pick_place"
    assert result["policy_hz"] == 30
    assert result["target_delay_steps"] == 90 and result["total_steps"] == 390
    assert result["ros_domain_id"] == 120
    assert result["controller_readiness_required"] is True
    assert result["schema_version"] == 2
    assert result["target_position_world_m"] == [.25, 0., .768]
    for key in ("exploration_enabled", "image_augmentation_enabled", "environment_randomization_enabled", "optimizer_enabled"):
        assert result[key] is False
    assert not output.exists()
    assert (checkpoint / "config.json").read_bytes() == original_config


def test_checkpoint_reward_is_preferred_but_explicit_override_is_supported(artifacts):
    project, checkpoint, output = artifacts
    saved = PickPlaceRewardConfig(gamma=.99).to_dict()
    reward_path = checkpoint / "resolved_reward.json"
    reward_path.write_text(json.dumps(saved), encoding="utf-8")
    result = evaluation.build_plan(checkpoint, project_root=project, output_directory=output, gui=False)
    assert result["resolved_reward"] == saved
    assert result["reward_config_path"] == str(reward_path)
    assert result["gazebo_gui"] is False
    override = project / "config/pick_place_reward.json"
    result = evaluation.build_plan(checkpoint, project_root=project, output_directory=output, reward_config=override)
    assert result["resolved_reward"]["gamma"] == .997


@pytest.mark.parametrize("destination_kind", ["existing", "checkpoint", "inside_checkpoint"])
def test_plan_refuses_existing_output_or_checkpoint_contents(artifacts, destination_kind):
    project, checkpoint, output = artifacts
    if destination_kind == "existing":
        output.mkdir()
    elif destination_kind == "checkpoint":
        output = checkpoint
    else:
        output = checkpoint / "evaluation"
    with pytest.raises(ValueError, match="new directory"):
        evaluation.build_plan(checkpoint, project_root=project, output_directory=output)


@pytest.mark.parametrize("kwargs", [
    {"ros_domain_id": -1}, {"ros_domain_id": 233}, {"episodes": 0},
    {"seed": -1}, {"seed": 2**32}, {"startup_timeout_s": 0},
    {"startup_timeout_s": float("inf")}, {"target_position": (.25, float("nan"), .768)},
    {"target_position": (.25, 0.)},
])
def test_plan_rejects_invalid_runtime_parameters(artifacts, kwargs):
    project, checkpoint, output = artifacts
    with pytest.raises(ValueError):
        evaluation.build_plan(checkpoint, project_root=project, output_directory=output, **kwargs)
    assert not output.exists()


def test_plan_rejects_non_act_missing_weights_and_non_30hz(artifacts):
    project, checkpoint, output = artifacts
    config = checkpoint / "config.json"
    config.write_text('{"type":"diffusion"}', encoding="utf-8")
    with pytest.raises(ValueError, match="ACT"):
        evaluation.build_plan(checkpoint, project_root=project, output_directory=output)
    config.write_text('{"type":"act"}', encoding="utf-8")
    (checkpoint / "resolved_reward.json").write_text(
        json.dumps(PickPlaceRewardConfig(control_hz=20.).to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="30 Hz"):
        evaluation.build_plan(checkpoint, project_root=project, output_directory=output)
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.build_plan(checkpoint, project_root=project, output_directory=output)


def _observation(step):
    return SimpleNamespace(joint_positions=np.full(6, step / 1000.), joint_velocities=np.zeros(6),
                           step=step)


class FakeWorker:
    """Only expose the commands permitted by this continuous evaluation."""
    def __init__(self, *, terminal_step=None, fail_step=None, spawn_success=True,
                 initial_pose="zero", clock_drift_step=None, prepare_error=False,
                 controller_readiness=True):
        self.step = 0
        self.calls = []
        self.response = None
        self.terminal_step = terminal_step
        self.fail_step = fail_step
        self.spawn_success = spawn_success
        self.initial_pose = initial_pose
        self.clock_drift_step = clock_drift_step
        self.prepare_error = prepare_error
        self.spawned = False
        self.controller_readiness = controller_readiness

    def info(self):
        # Use the actual deterministic 33/33/34 ms pattern, after a nonzero
        # setup clock, so the runner must subtract the measured origin.
        cycles, remainder = divmod(self.step, 3)
        sim_time = 7. + cycles / 10. + remainder * .033
        if self.step == self.clock_drift_step:
            sim_time += .01
        return {"sim_time_s": sim_time, "evaluation_initial_pose": self.initial_pose,
                "evaluation_controller_readiness": {
                    "ready": self.controller_readiness, "controller_name": "position_controller",
                    "controller_state": "active" if self.controller_readiness else "inactive",
                    "command_subscription_count": 1 if self.controller_readiness else 0,
                    "waited_wall_s": 2.5,
                },
                "stub_step": self.step, "stub_target_present": self.spawned,
                "task_outcome": "running", "failure_reason": "safety_stop" if self.step == self.terminal_step else None}

    def prepare_episode_async(self, attempt, *, task_mode, scene_seed):
        self.calls.append(("prepare", attempt, task_mode, scene_seed))
        if self.prepare_error:
            self.response = {"type": "operation_error", "error_message": "zero preparation timeout"}
        else:
            self.response = {"type": "episode_prepared", "observation": {"step": 0}, "info": self.info()}

    def receive(self, *, timeout_s):
        assert timeout_s > 0
        assert self.response is not None
        response, self.response = self.response, None
        return response

    def observation_from_metadata(self, metadata):
        return _observation(metadata["step"])

    def spawn_target_async(self, xyz, quat):
        self.calls.append(("spawn", self.step, list(xyz), list(quat)))
        self.spawned = self.spawn_success
        self.response = {"type": "target_spawned", "success": self.spawn_success, "message": "stub service result"}

    def step_async(self, targets):
        self.calls.append(("step", self.step + 1, targets.tolist()))
        if self.step + 1 == self.fail_step:
            self.response = {"type": "operation_error", "error_message": "synchronized camera timeout"}
            return
        self.step += 1
        self.response = {"type": "step_result", "observation": {"step": self.step}, "info": self.info(),
                         "reward": -.01, "terminated": self.step == self.terminal_step,
                         "truncated": self.step == 390}


class FakeAdapter:
    def __init__(self):
        self.reset_count = 0
        self.seen_steps = []

    def reset(self):
        self.reset_count += 1

    def select_action(self, observation):
        self.seen_steps.append(observation.step)
        # A different command each step verifies that no pre-target HOLD action
        # silently substitutes for the policy output.
        positions = np.full(6, (observation.step + 1) / 1000.)
        return SimpleNamespace(raw_policy_positions=positions, action=SimpleNamespace(joint_targets=positions))


@pytest.fixture
def monitors(monkeypatch):
    frames, summaries = [], []

    def measure(info, reward_config):
        step = info["stub_step"]
        return {"evidence_valid": True, "at_home": step >= 30,
                "evidence_sim_time_s": info["sim_time_s"],
                "target_present": info["stub_target_present"],
                "target_evidence_valid": info["stub_target_present"],
                "distance_m": .1 - step / 10000. if info["stub_target_present"] else None,
                "home_error_rad": .001 if step >= 30 else .5,
                "task_phase": "APPROACH" if info["stub_target_present"] else "WAIT_HOME"}

    def summarize(rows, *, initial, reward_config, target_delay_s):
        summaries.append((rows, initial, reward_config, target_delay_s))
        return {"diagnostic_rows": len(rows)}

    monkeypatch.setattr(home_reach_metrics, "measure", measure)
    monkeypatch.setattr(home_reach_metrics, "summarize", summarize)
    monkeypatch.setattr(evaluation, "_save_frame", lambda destination, name, observation, metadata:
                        frames.append((name, observation.step, metadata)))
    return frames, summaries


def test_complete_timeline_has_390_policy_actions_and_one_spawn_after_90_confirmed_steps(plan, tmp_path, monitors):
    worker, adapter = FakeWorker(), FakeAdapter()
    destination = tmp_path / "episode"
    result = evaluation._run_episode(worker, adapter, plan, destination, episode_index=2)
    assert worker.calls[0] == ("prepare", 2, "pick_place", plan["seed"])
    assert adapter.reset_count == 1 and adapter.seen_steps == list(range(390))
    step_calls = [call for call in worker.calls if call[0] == "step"]
    assert len(step_calls) == 390
    for step, call in enumerate(step_calls, 1):
        assert call == ("step", step, [step / 1000.] * 6)
    assert [call for call in worker.calls if call[0] == "spawn"] == [
        ("spawn", 90, [.25, 0., .768], [0., 0., 0., 1.])]
    spawn_index = next(i for i, call in enumerate(worker.calls) if call[0] == "spawn")
    assert worker.calls[spawn_index - 1][0:2] == ("step", 90)
    assert worker.calls[spawn_index + 1][0:2] == ("step", 91)
    assert result["complete_timeline"] and result["completed_steps"] == 390 and result["elapsed_s"] == 13.
    assert result["end_reason"] == "time_budget" and result["error"] is None
    rows = _jsonl(destination / "steps.jsonl")
    assert [row["step_index"] for row in rows] == list(range(1, 391))
    assert rows[89]["measured_sim_elapsed_s"] == pytest.approx(3.)
    assert rows[-1]["measured_sim_elapsed_s"] == pytest.approx(13.)
    assert rows[89]["diagnostics"]["target_present"] is False
    assert rows[90]["diagnostics"]["target_present"] is True
    assert rows[0]["joint_positions_before_rad"] == [0.] * 6
    assert rows[-1]["joint_positions_after_rad"] == [.39] * 6
    events = _jsonl(destination / "events.jsonl")
    assert events[0]["event"] == "controller_ready_before_test"
    assert events[0]["readiness"]["waited_wall_s"] == 2.5
    assert events[1]["event"] == "policy_started" and events[1]["elapsed_s"] == 0.
    assert result["controller_readiness"]["controller_state"] == "active"
    request = next(event for event in events if event["event"] == "target_requested")
    assert request["completed_steps"] == 90 and request["elapsed_s"] == 3.
    feedback = next(event for event in events if event["event"] == "first_target_feedback")
    assert feedback["elapsed_s"] == pytest.approx(91 / 30.)
    frames, summaries = monitors
    assert [(name, step) for name, step, _ in frames] == [
        ("initial", 0), ("first_home", 30), ("before_target", 90),
        ("first_target_feedback", 91), ("final", 390), ("nearest_target", 390)]
    assert summaries[0][3] == 3.
    assert _json(destination / "summary.json") == result


def test_early_environment_terminal_persists_partial_timeline(plan, tmp_path, monitors):
    worker, adapter = FakeWorker(terminal_step=125), FakeAdapter()
    destination = tmp_path / "episode"
    result = evaluation._run_episode(worker, adapter, plan, destination, episode_index=1)
    assert not result["complete_timeline"] and result["completed_steps"] == 125
    assert result["end_reason"] == "safety_stop" and result["error"] is None
    assert len(adapter.seen_steps) == 125
    assert len(_jsonl(destination / "steps.jsonl")) == 125
    assert _json(destination / "summary.json")["diagnostic_rows"] == 125
    assert not (destination / "failure.json").exists()


def test_step_failure_preserves_only_confirmed_steps_and_partial_summary(plan, tmp_path, monitors):
    worker, adapter = FakeWorker(fail_step=8), FakeAdapter()
    destination = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="camera timeout"):
        evaluation._run_episode(worker, adapter, plan, destination, episode_index=1)
    assert len(adapter.seen_steps) == 8
    assert len(_jsonl(destination / "steps.jsonl")) == 7
    result = _json(destination / "summary.json")
    assert not result["complete_timeline"] and result["completed_steps"] == 7
    assert result["end_reason"] == "runtime_error" and result["diagnostic_rows"] == 7
    assert _json(destination / "failure.json")["completed_steps"] == 7
    assert ("final", 7) in [(name, step) for name, step, _ in monitors[0]]
    assert not any(call[0] == "spawn" for call in worker.calls)


def test_target_creation_failure_is_logged_after_exactly_90_policy_steps(plan, tmp_path, monitors):
    worker, adapter = FakeWorker(spawn_success=False), FakeAdapter()
    destination = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="Target creation failed"):
        evaluation._run_episode(worker, adapter, plan, destination, episode_index=1)
    assert len(adapter.seen_steps) == 90 and worker.step == 90
    result = _json(destination / "summary.json")
    assert result["completed_steps"] == 90 and not result["complete_timeline"]
    assert result["target_spawn"]["success"] is False
    events = _jsonl(destination / "events.jsonl")
    assert events[-2]["event"] == "target_requested"
    assert events[-1]["event"] == "target_service_result" and not events[-1]["response"]["success"]
    assert len(_jsonl(destination / "steps.jsonl")) == 90


def test_prepare_failure_still_writes_empty_partial_summary(plan, tmp_path, monitors):
    worker, adapter = FakeWorker(prepare_error=True), FakeAdapter()
    destination = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="zero preparation timeout"):
        evaluation._run_episode(worker, adapter, plan, destination, episode_index=1)
    assert adapter.reset_count == 0 and not adapter.seen_steps
    assert not _jsonl(destination / "steps.jsonl")
    result = _json(destination / "summary.json")
    assert result["completed_steps"] == 0 and not result["complete_timeline"]
    assert result["error"]["completed_steps"] == 0
    assert monitors == ([], [])


def test_worker_must_confirm_zero_pose_before_policy_is_called(plan, tmp_path, monitors):
    worker, adapter = FakeWorker(initial_pose="home"), FakeAdapter()
    destination = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="ZERO initial pose"):
        evaluation._run_episode(worker, adapter, plan, destination, episode_index=1)
    assert adapter.reset_count == 0 and not adapter.seen_steps
    assert _json(destination / "summary.json")["completed_steps"] == 0


@pytest.mark.parametrize("readiness", [
    None,
    {},
    {"ready": False, "controller_name": "position_controller", "controller_state": "inactive", "command_subscription_count": 1},
    {"ready": True, "controller_name": "position_controller", "controller_state": "inactive", "command_subscription_count": 1},
    {"ready": True, "controller_name": "position_controller", "controller_state": "active", "command_subscription_count": 0},
    {"ready": True, "controller_name": "joint_state_broadcaster", "controller_state": "active", "command_subscription_count": 1},
])
def test_missing_or_incomplete_controller_readiness_prevents_policy_and_clock_start(plan, tmp_path, monitors, readiness):
    worker, adapter = FakeWorker(), FakeAdapter()
    original_info = worker.info

    def info():
        value = original_info()
        if readiness is None:
            value.pop("evaluation_controller_readiness")
        else:
            value["evaluation_controller_readiness"] = readiness
        return value

    worker.info = info
    destination = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="active, connected position controller"):
        evaluation._run_episode(worker, adapter, plan, destination, episode_index=1)
    assert adapter.reset_count == 0 and adapter.seen_steps == []
    assert not any(call[0] in {"step", "spawn"} for call in worker.calls)
    assert _jsonl(destination / "steps.jsonl") == []
    assert _jsonl(destination / "events.jsonl") == []
    assert _json(destination / "summary.json")["complete_timeline"] is False


def test_clock_divergence_aborts_with_confirmed_transition_saved(plan, tmp_path, monitors):
    worker = FakeWorker(clock_drift_step=13)
    destination = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="clock diverged"):
        evaluation._run_episode(worker, FakeAdapter(), plan, destination, episode_index=1)
    assert len(_jsonl(destination / "steps.jsonl")) == 13
    result = _json(destination / "summary.json")
    assert result["completed_steps"] == 13 and not result["complete_timeline"]
    assert "clock diverged" in result["error"]["message"]
