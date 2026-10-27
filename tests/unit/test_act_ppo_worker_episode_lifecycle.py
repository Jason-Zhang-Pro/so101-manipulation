"""Worker episode ownership tests; no ROS or Gazebo processes are started."""

from types import SimpleNamespace

import pytest

from so101_capstone.training import act_ppo_multiworld_worker as worker


def make_host(tmp_path, monkeypatch, *, reward_profile="legacy"):
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")
    config = worker.GazeboWorkerConfig(
        world_index=2,
        ros_domain_id=22,
        gz_partition="gz_env_22",
        gazebo_gui=False,
        project_root=tmp_path,
        output_directory=tmp_path,
        startup_feedback_timeout_s=2.,
        reward_profile=reward_profile,
        reward_config={} if reward_profile == "pick_place_v2" else None,
    )
    host = worker._WorkerRuntimeHost(
        config,
        top_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
        gripper_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
    )
    monkeypatch.setattr(host, "publish_observation", lambda observation: observation)
    return host


def prepared_episode():
    return SimpleNamespace(
        observation={"sample": "fresh"},
        info={"episode": 0},
        home_reset=SimpleNamespace(
            control_steps=30,
            physics_steps=1000,
            stable_sample_count=3,
            max_position_error_rad=0.,
            max_velocity_rad_s=0.,
        ),
    )


@pytest.mark.parametrize("reward_profile", ["legacy", "pick_place_v2"])
def test_prepare_uses_fresh_world_once_without_removing_targets(
    tmp_path, monkeypatch, reward_profile,
):
    host = make_host(tmp_path, monkeypatch, reward_profile=reward_profile)
    calls = []
    host.runtime = SimpleNamespace(
        configure_task=lambda *args, **kwargs: calls.append((args, kwargs)),
        prepare_episode=lambda **kwargs: prepared_episode(),
        delete_all_targets=lambda: pytest.fail("Live targets must not be removed"),
    )
    host._fresh_world_available = True

    result = host.prepare_episode(task_mode="idle_completed", scene_seed=42)

    assert result["cleanup"]["requested"] is False
    assert result["cleanup"]["strategy"] == "fresh_world"
    assert result["cleanup"]["success"] is True
    if reward_profile == "pick_place_v2":
        assert calls == [(("idle_completed",), {"seed": 42, "n_action_steps": 1})]
    else:
        assert calls == []
    with pytest.raises(RuntimeError, match="freshly started Gazebo world"):
        host.prepare_episode()


@pytest.mark.parametrize("failure_phase", ["configure", "prepare"])
def test_partial_preparation_also_consumes_world(tmp_path, monkeypatch, failure_phase):
    host = make_host(tmp_path, monkeypatch, reward_profile="pick_place_v2")
    calls = []

    def configure(*args, **kwargs):
        calls.append("configure")
        if failure_phase == "configure":
            raise ValueError("partial scene")

    def prepare(**kwargs):
        calls.append("prepare")
        raise ValueError("partial scene")

    host.runtime = SimpleNamespace(configure_task=configure, prepare_episode=prepare)
    host._fresh_world_available = True
    with pytest.raises(ValueError, match="partial scene"):
        host.prepare_episode()
    assert host._fresh_world_available is False
    completed_calls = list(calls)
    with pytest.raises(RuntimeError, match="freshly started Gazebo world"):
        host.prepare_episode()
    assert calls == completed_calls


def test_randomization_episode_key_is_not_reset_by_world_recreation(tmp_path, monkeypatch):
    host = make_host(tmp_path, monkeypatch)
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "true")
    keys = []

    def prepare(**kwargs):
        keys.append(kwargs["environment_randomization_episode_key"])
        return prepared_episode()

    for episode_key in (4, 7):
        # A replacement runtime starts a new local counter, while the parent
        # continues to supply its globally unique attempt index.
        host.runtime = SimpleNamespace(prepare_episode=prepare)
        host._fresh_world_available = True
        host.prepare_episode(episode_attempt_index=episode_key)
    assert keys == [4, 7]


def test_stop_world_closes_runtime_without_removing_targets(tmp_path, monkeypatch):
    host = make_host(tmp_path, monkeypatch)
    events = []
    host.runtime = SimpleNamespace(
        close=lambda: events.append("close"),
        delete_all_targets=lambda: pytest.fail("Stopping must not remove live targets"),
    )
    host._fresh_world_available = True
    monkeypatch.setattr(host, "_terminate_launch_process", lambda: events.append("terminate"))
    monkeypatch.setattr(
        worker,
        "_cleanup_reserved_world_resources",
        lambda config: {"scan_count": 2, "matched": 0, "terminated": 0, "killed": 0},
    )

    host.stop_world()

    assert events == ["close", "terminate"]
    assert host.runtime is None
    assert host._fresh_world_available is False


def test_worker_rejects_delete_rpc_and_forwards_episode_restart_reason(tmp_path, monkeypatch):
    host = make_host(tmp_path, monkeypatch)
    events = []

    def start_world(**kwargs):
        events.append(kwargs)
        return {"type": "ready"}

    monkeypatch.setattr(host, "start_world", start_world)
    monkeypatch.setattr(host, "stop_world", lambda: None)
    host.runtime = SimpleNamespace(
        delete_all_targets=lambda: pytest.fail("Delete RPC reached runtime"),
    )
    monkeypatch.setattr(worker, "_WorkerRuntimeHost", lambda *args, **kwargs: host)
    monkeypatch.setattr(worker, "_runtime_error_types", lambda: ())
    requests = iter([
        {"command": "delete_all_targets"},
        {"command": "restart_world", "reason": "episode_complete"},
        {"command": "restart_world"},
        {"command": "shutdown"},
    ])
    replies = []
    connection = SimpleNamespace(recv=lambda: next(requests), send=replies.append, close=lambda: None)
    for name in ("ROS_DOMAIN_ID", "GZ_PARTITION", "IGN_PARTITION"):
        monkeypatch.setenv(name, "unit-test")

    worker._worker_main(connection, host.config.payload(), None, None)

    assert replies[1]["type"] == "operation_error"
    assert replies[1]["operation"] == "delete_all_targets"
    assert "Target deletion is disabled" in replies[1]["error_message"]
    assert replies[1]["recoverable"] is False
    assert [item["reason"] for item in events] == [
        "initial_startup", "episode_complete", "runtime_restart",
    ]


@pytest.mark.parametrize("reason", ["runtime_restart", "episode_complete"])
def test_client_forwards_restart_reason_without_replacing_worker(reason):
    client = object.__new__(worker.GazeboWorkerClient)
    client.process = SimpleNamespace(is_alive=lambda: True)
    requests = []
    client.request_async = lambda command, **kwargs: requests.append((command, kwargs))

    mode = client.begin_restart(reason=reason)

    assert mode == "runtime_restart"
    assert requests == [("restart_world", {"reason": reason})]


def test_failed_launch_does_not_rearm_episode_preparation(tmp_path, monkeypatch):
    from so101_capstone.environments import simulation_runtime

    host = make_host(tmp_path, monkeypatch)
    host._fresh_world_available = True
    monkeypatch.setattr(host, "stop_world", lambda: None)
    monkeypatch.setattr(
        worker.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(poll=lambda: 1),
    )
    monkeypatch.setattr(
        simulation_runtime,
        "create_ros_simulation_runtime",
        lambda **kwargs: SimpleNamespace(wait_until_ready=lambda **kwargs: None),
    )
    try:
        with pytest.raises(RuntimeError, match="exited before"):
            host._launch_once(reason="episode_complete", attempt=1)
        assert host._fresh_world_available is False
        assert "episode_complete: launch attempt 1" in host.config.log_path.read_text()
    finally:
        if host.log_stream is not None:
            host.log_stream.close()
