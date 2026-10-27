"""Evaluation-only worker options, exercised without launching ROS or Gazebo."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.training import act_ppo_multiworld_worker as worker


def config(tmp_path, **kwargs):
    return worker.GazeboWorkerConfig(
        world_index=0, ros_domain_id=120, gz_partition="unit_eval", gazebo_gui=False,
        project_root=tmp_path, output_directory=tmp_path, startup_feedback_timeout_s=2.,
        reward_profile="pick_place_v2", reward_config=PickPlaceRewardConfig().to_dict(),
        **kwargs,
    )


def make_host(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")
    monkeypatch.setenv("SO101_IMAGE_AUGMENTATION_ENABLED", "false")
    host = worker._WorkerRuntimeHost(
        config(tmp_path, **kwargs), top_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
        gripper_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
    )
    monkeypatch.setattr(host, "publish_observation", lambda observation: observation)
    return host


def fake_runtime():
    calls = []
    observation = {"joint_positions": (0.,) * 6, "joint_velocities": (0.,) * 6}
    measurements = {"sim_time_s": 10., "evidence_sim_time_s": 9.96, "joint_positions_rad": (0.,) * 6}
    prepared = SimpleNamespace(observation=observation, info={"evaluation_initial_pose": "zero"},
                               home_reset=SimpleNamespace(control_steps=10, physics_steps=333,
                                                          stable_sample_count=5, max_position_error_rad=.001,
                                                          max_velocity_rad_s=.002))
    result = SimpleNamespace(observation=observation, info={"step": 1}, reward=0.,
                             terminated=False, truncated=False)

    def prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return prepared

    runtime = SimpleNamespace(
        configure_task=lambda *args, **kwargs: calls.append(("configure", args, kwargs)),
        prepare_episode=prepare, evaluation_measurements=lambda: dict(measurements),
        step=lambda action: result,
        start_training_stage=lambda **kwargs: pytest.fail("Evaluation must not initialize HOME"),
    )
    return runtime, calls, observation, measurements


@pytest.mark.parametrize("evaluation_mode", [False, True])
def test_evaluation_options_round_trip_and_default_is_disabled(tmp_path, evaluation_mode):
    original = config(tmp_path, evaluation_mode=evaluation_mode,
                      observe_after_success_s=13. if evaluation_mode else 0.)
    restored = worker.GazeboWorkerConfig.from_payload(original.payload())
    assert restored == original
    legacy_payload = config(tmp_path).payload()
    legacy_payload.pop("evaluation_mode")
    legacy_payload.pop("observe_after_success_s")
    legacy = worker.GazeboWorkerConfig.from_payload(legacy_payload)
    assert legacy.evaluation_mode is False
    assert legacy.observe_after_success_s == 0


@pytest.mark.parametrize("seconds", [-1., float("nan"), float("inf"), True])
def test_invalid_success_monitor_duration_is_rejected(tmp_path, seconds):
    with pytest.raises(ValueError, match="finite and non-negative"):
        config(tmp_path, evaluation_mode=True, observe_after_success_s=seconds)


def test_evaluation_flags_reject_unsafe_training_combinations(tmp_path):
    with pytest.raises(ValueError, match="evaluation worker"):
        config(tmp_path, observe_after_success_s=13.)
    with pytest.raises(ValueError, match="must be a bool"):
        config(tmp_path, evaluation_mode=1)
    with pytest.raises(ValueError, match="require pick_place_v2"):
        replace(config(tmp_path, evaluation_mode=True), reward_profile="legacy", reward_config=None)


def test_evaluation_prepares_zero_and_exports_aligned_evidence_without_actor_changes(tmp_path, monkeypatch):
    host = make_host(tmp_path, monkeypatch, evaluation_mode=True)
    host.runtime, calls, observation, measurements = fake_runtime()
    host._fresh_world_available = True
    response = host.prepare_episode(scene_seed=42)
    assert calls == [("configure", ("pick_place",), {"seed": 42, "n_action_steps": 1}),
                     ("prepare", {"evaluation_initial_pose": "zero"})]
    assert response["observation"] is observation
    assert response["info"]["evaluation_measurements"] == measurements
    assert "home_reset" not in response
    assert response["initial_pose_reset"]["stable_sample_count"] == 5
    assert response["cleanup"]["requested"] is False
    assert "evaluation_measurements" not in observation
    stepped = host.step([0.] * 6)
    assert stepped["observation"] is observation
    assert stepped["info"]["evaluation_measurements"] == measurements
    with pytest.raises(RuntimeError, match="freshly started"):
        host.prepare_episode()
    with pytest.raises(ValueError, match="cannot initialize HOME"):
        host.start_training_stage(training_stage="S0", stage_config={})


def test_training_worker_keeps_default_preparation_and_omits_privileged_export(tmp_path, monkeypatch):
    host = make_host(tmp_path, monkeypatch)
    host.runtime, calls, _, _ = fake_runtime()
    host.runtime.evaluation_measurements = lambda: pytest.fail("Training must not request evaluation evidence")
    host._fresh_world_available = True
    response = host.prepare_episode(scene_seed=42)
    assert calls[-1] == ("prepare", {})
    assert "home_reset" in response
    assert "initial_pose_reset" not in response
    assert "evaluation_measurements" not in response["info"]
    assert "evaluation_measurements" not in host.step([0.] * 6)["info"]


@pytest.mark.parametrize("task_mode", ["idle_empty", "idle_completed"])
def test_evaluation_worker_cannot_prepare_incompatible_task_mode(tmp_path, monkeypatch, task_mode):
    host = make_host(tmp_path, monkeypatch, evaluation_mode=True)
    host.runtime, calls, _, _ = fake_runtime()
    host._fresh_world_available = True
    with pytest.raises(ValueError, match="task_mode='pick_place'"):
        host.prepare_episode(task_mode=task_mode)
    assert calls == []
    assert host._fresh_world_available is False


@pytest.mark.parametrize("flag", ["SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "SO101_IMAGE_AUGMENTATION_ENABLED"])
def test_evaluation_rejects_randomization_before_launch(tmp_path, monkeypatch, flag):
    host = make_host(tmp_path, monkeypatch, evaluation_mode=True)
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not launch"))
    monkeypatch.setattr(host, "stop_world", lambda: pytest.fail("Must not mutate worlds"))
    with pytest.raises(ValueError, match="randomization disabled"):
        host._launch_once(reason="unit-test", attempt=1)


@pytest.mark.parametrize("evaluation_mode", [False, True])
def test_only_evaluation_passes_success_monitoring_to_runtime_factory(tmp_path, monkeypatch, evaluation_mode):
    from so101_capstone.environments import simulation_runtime

    host = make_host(tmp_path, monkeypatch, evaluation_mode=evaluation_mode,
                     observe_after_success_s=13. if evaluation_mode else 0.)
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(wait_until_ready=lambda **kwargs: None)

    monkeypatch.setattr(simulation_runtime, "create_ros_simulation_runtime", create)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(host, "stop_world", lambda: None)
    try:
        host._launch_once(reason="unit-test", attempt=1)
        if evaluation_mode:
            assert captured["observe_after_success_s"] == 13.
        else:
            assert "observe_after_success_s" not in captured
        assert host._fresh_world_available is True
    finally:
        if host.log_stream is not None:
            host.log_stream.close()
