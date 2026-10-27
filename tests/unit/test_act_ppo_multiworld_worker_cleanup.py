from multiprocessing import RawArray
from types import SimpleNamespace
from dataclasses import replace

import pytest

import so101_capstone.training.act_ppo_multiworld_worker as worker_module
from so101_capstone.training.act_ppo_multiworld_worker import (
    GazeboWorkerConfig,
    _WorkerRuntimeHost,
    _cleanup_reserved_world_resources,
)


IMAGE_BYTES = 480 * 640 * 3


def make_config(tmp_path):
    return GazeboWorkerConfig(
        world_index=2,
        ros_domain_id=22,
        gz_partition="gz_env_22",
        gazebo_gui=True,
        project_root=tmp_path,
        output_directory=tmp_path / "out",
        startup_feedback_timeout_s=30.0,
        launch_shutdown_timeout_s=10.0,
    )


def make_report(*, matched=0, terminated=0, killed=0):
    return SimpleNamespace(
        matched_processes=tuple(object() for _ in range(matched)),
        terminated_processes=tuple(object() for _ in range(terminated)),
        killed_processes=tuple(object() for _ in range(killed)),
    )


def test_scoped_cleanup_requires_two_consecutive_empty_scans(tmp_path):
    calls = []
    reports = iter(
        [
            make_report(matched=3, terminated=3, killed=1),
            make_report(),
            make_report(),
        ]
    )
    sleeps = []

    def fake_cleanup(**kwargs):
        calls.append(kwargs)
        return next(reports)

    summary = _cleanup_reserved_world_resources(
        make_config(tmp_path),
        cleanup_fn=fake_cleanup,
        sleep_fn=sleeps.append,
    )

    assert summary == {
        "scan_count": 3,
        "matched": 3,
        "terminated": 3,
        "killed": 1,
    }
    assert len(calls) == 3
    assert all(
        call["target_ros_domain_ids"] == (22,)
        for call in calls
    )
    assert all(
        call["target_gz_partitions"] == ("gz_env_22",)
        for call in calls
    )
    assert all(call["timeout_s"] == 5.0 for call in calls)
    assert len(sleeps) == 2


def test_scoped_cleanup_fails_if_resource_never_goes_quiet(tmp_path):
    def fake_cleanup(**_kwargs):
        return make_report(matched=1, terminated=1)

    with pytest.raises(RuntimeError, match="stable empty state"):
        _cleanup_reserved_world_resources(
            make_config(tmp_path),
            cleanup_fn=fake_cleanup,
            sleep_fn=lambda _seconds: None,
            quiet_scans_required=2,
            max_scans=4,
        )


def test_stop_world_runs_scoped_cleanup_even_without_launch_parent(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_scoped_cleanup(config):
        calls.append(config)
        return {
            "scan_count": 2,
            "matched": 4,
            "terminated": 4,
            "killed": 1,
        }

    monkeypatch.setattr(
        worker_module,
        "_cleanup_reserved_world_resources",
        fake_scoped_cleanup,
    )

    host = _WorkerRuntimeHost(
        make_config(tmp_path),
        top_buffer=RawArray("B", IMAGE_BYTES),
        gripper_buffer=RawArray("B", IMAGE_BYTES),
    )

    host.stop_world()

    assert calls == [host.config]
    assert host.last_cleanup_summary == {
        "scan_count": 2,
        "matched": 4,
        "terminated": 4,
        "killed": 1,
    }


def test_stop_world_cleanup_failure_is_not_silently_ignored(
    tmp_path,
    monkeypatch,
):
    def fail_cleanup(_config):
        raise RuntimeError("old world still present")

    monkeypatch.setattr(
        worker_module,
        "_cleanup_reserved_world_resources",
        fail_cleanup,
    )

    host = _WorkerRuntimeHost(
        make_config(tmp_path),
        top_buffer=RawArray("B", IMAGE_BYTES),
        gripper_buffer=RawArray("B", IMAGE_BYTES),
    )

    with pytest.raises(RuntimeError, match="old world still present"):
        host.stop_world()


def test_worker_horizon_survives_payload_and_reaches_runtime(tmp_path, monkeypatch):
    import so101_capstone.environments.simulation_runtime as runtime_module

    config = replace(make_config(tmp_path), max_episode_steps=1200)
    config = GazeboWorkerConfig.from_payload(config.payload())
    assert config.max_episode_steps == 1200
    captured = {}
    runtime = SimpleNamespace(wait_until_ready=lambda **kwargs: None)

    def create_runtime(**kwargs):
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(runtime_module, "create_ros_simulation_runtime", create_runtime)
    monkeypatch.setattr(worker_module.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(_WorkerRuntimeHost, "stop_world", lambda self: None)
    monkeypatch.setattr(_WorkerRuntimeHost, "_write_log_header", lambda self, message: None)
    monkeypatch.delenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", raising=False)
    config.log_path.parent.mkdir(parents=True)
    host = _WorkerRuntimeHost(config, top_buffer=bytearray(IMAGE_BYTES), gripper_buffer=bytearray(IMAGE_BYTES))
    try:
        assert host._fresh_world_available is False
        host._launch_once(reason="unit-test", attempt=1)
        assert captured["environment_config"].max_episode_steps == 1200
        assert captured["runtime_config"].resume_world_on_close is False
        assert host._fresh_world_available is True
    finally:
        if host.log_stream is not None:
            host.log_stream.close()
