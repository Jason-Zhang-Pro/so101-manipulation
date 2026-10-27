"""Crash attribution tests: temporary text logs and fake workers only."""

from types import SimpleNamespace

import pytest

from so101_capstone.training.act_ppo_gazebo_crash import read_gazebo_crash_diagnostics
from so101_capstone.training import act_ppo_multiworld_worker as worker


@pytest.mark.parametrize("line", [
    "[ign gazebo-1] Segmentation fault (Address not mapped to object [0x123])",
    "[ign gazebo-1] Aborted (core dumped)",
    "[gz sim-1] Bus error (core dumped)",
    "[gzserver-2] Floating point exception",
    "[gazebo-1] Illegal instruction",
    "[ign gazebo-1] received signal SIGSEGV",
    "[ign gazebo-1] received signal 11",
    "[ERROR] [ign gazebo-1]: process has died [pid 12, exit code -11, cmd 'ign gazebo']",
    "[ERROR] [ign gazebo-1]: process has died [pid 12, exit code 134, cmd 'ign gazebo']",
])
def test_confirms_gazebo_fatal_signal(tmp_path, line):
    path = tmp_path / "gazebo.log"
    path.write_text(line + "\n")
    result = read_gazebo_crash_diagnostics(path, launch_log_offset=0)
    assert result["confirmed"] is True
    assert result["status"] == "confirmed"
    assert result["evidence"][0]["line"] == line


@pytest.mark.parametrize("line", [
    "Camera frames are too old relative to the completed simulation step",
    "[ERROR] [ros2 launch-1]: process has died [exit code -11]",
    "[ERROR] [ign gazebo-1]: process has died [pid 12, exit code 1]",
    "[ERROR] [ign gazebo-1]: process has died [pid 12, exit code -15]",
    "[ERROR] [ign gazebo-1]: process has died [pid 12, exit code -9]",
    "[ign gazebo-1] received signal SIGTERM",
    "[ign gazebo-1] received signal 15",
    "[ign gazebo-1] received signal SIGKILL",
    "[ign gazebo-1] [INFO] operation aborted during shutdown",
    "[controller_manager-4] Segmentation fault",
    "[WORLD CLEANUP] world=2 terminated=7 killed=0",
    "Segmentation fault",  # No process attribution.
])
def test_does_not_promote_unrelated_failures_or_cleanup(tmp_path, line):
    path = tmp_path / "gazebo.log"
    path.write_text(line + "\n")
    result = read_gazebo_crash_diagnostics(path, launch_log_offset=0)
    assert result["confirmed"] is False
    assert result["status"] == "unconfirmed"
    assert result["evidence"] == []


def test_launch_boundary_excludes_previous_crashed_episode(tmp_path):
    path = tmp_path / "gazebo.log"
    old = "[ign gazebo-1] Segmentation fault\n"
    path.write_text(old + "episode_complete: launch attempt 1\n")
    result = read_gazebo_crash_diagnostics(path, launch_log_offset=len(old.encode()))
    assert result["confirmed"] is False
    assert result["launch_log_offset"] == len(old.encode())
    assert read_gazebo_crash_diagnostics(path)["status"] == "unknown"


def test_log_read_is_bounded_and_still_captures_recent_signal(tmp_path):
    path = tmp_path / "gazebo.log"
    path.write_text("old log\n" * 1000 + "[ign gazebo-1] Segmentation fault\n")
    result = read_gazebo_crash_diagnostics(path, launch_log_offset=0, max_read_bytes=128)
    assert result["confirmed"] is True
    assert result["truncated"] is True
    assert result["bytes_read"] == 128


def test_unreadable_or_rotated_logs_remain_unknown(tmp_path):
    path = tmp_path / "missing.log"
    assert read_gazebo_crash_diagnostics(path, launch_log_offset=0)["reason"] == "log_read_failed"
    path.write_text("new")
    result = read_gazebo_crash_diagnostics(path, launch_log_offset=100)
    assert result["confirmed"] is False
    assert result["reason"] == "log_truncated_since_launch"


def make_client(tmp_path):
    config = worker.GazeboWorkerConfig(
        world_index=2,
        ros_domain_id=22,
        gz_partition="gz_env_22",
        gazebo_gui=False,
        project_root=tmp_path,
        output_directory=tmp_path,
        startup_feedback_timeout_s=2.,
    )
    # No multiprocessing resources or subprocesses are needed for these tests.
    client = object.__new__(worker.GazeboWorkerClient)
    client.config = config
    client.launch_log_offset = 0
    client.launch_pid = None
    client.process = SimpleNamespace(is_alive=lambda: True)
    config.log_path.parent.mkdir(parents=True)
    config.log_path.write_text("world started\n")
    return client


def test_timeout_evidence_is_captured_before_forced_cleanup(tmp_path):
    client = make_client(tmp_path)
    client.parent_connection = SimpleNamespace(poll=lambda timeout: False)
    events = []

    def kill():
        events.append("kill")
        with client.config.log_path.open("a") as stream:
            stream.write("[ign gazebo-1] Segmentation fault\n")

    client._hard_kill = kill
    with pytest.raises(worker.GazeboWorkerCommunicationError) as caught:
        client._receive(timeout_s=1.)
    assert caught.value.gazebo_diagnostics["confirmed"] is False
    assert events == ["kill"]
    # A fatal error occurring during our shutdown must not rewrite attribution.
    assert client.gazebo_diagnostics()["confirmed"] is True


@pytest.mark.parametrize("failure", ["recv", "poll", "send", "dead"])
def test_communication_failure_keeps_current_crash_evidence(tmp_path, failure):
    client = make_client(tmp_path)
    with client.config.log_path.open("a") as stream:
        stream.write("[ign gazebo-1] Segmentation fault\n")
    client._hard_kill = lambda: client.config.log_path.write_text("cleanup\n")

    def fail(*args):
        raise EOFError("disconnected")

    client.parent_connection = SimpleNamespace(
        poll=fail if failure == "poll" else lambda timeout: True,
        recv=fail,
        send=fail,
    )
    if failure == "dead":
        client.process = None
    with pytest.raises(worker.GazeboWorkerCommunicationError) as caught:
        if failure in {"send", "dead"}:
            client._send({"command": "step"})
        else:
            client._receive(timeout_s=1.)
    assert caught.value.gazebo_diagnostics["confirmed"] is True


def test_ready_refreshes_launch_offset_and_planned_restart_excludes_old_log(tmp_path):
    client = make_client(tmp_path)
    old = "[ign gazebo-1] Segmentation fault\n"
    client.config.log_path.write_text(old)
    sent = []
    client._send = sent.append
    client.request_async("restart_world", reason="episode_complete")
    assert client.launch_log_offset == len(old.encode())
    assert client.gazebo_diagnostics()["confirmed"] is False
    response = {"type": "ready", "launch_pid": 123, "launch_log_offset": 999}
    client.parent_connection = SimpleNamespace(poll=lambda timeout: True, recv=lambda: response)
    assert client._receive(timeout_s=1.) is response
    assert client.launch_log_offset == 999
    assert client.launch_pid == 123


def test_worker_operation_error_reports_evidence_before_cleanup(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client.config.log_path.write_text("[ign gazebo-1] Segmentation fault\n")
    host = worker._WorkerRuntimeHost(
        client.config,
        top_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
        gripper_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
    )
    host.launch_log_offset = 0
    host.start_world = lambda **kwargs: {"type": "ready", "launch_log_offset": 0}
    host.stop_world = lambda: client.config.log_path.write_text("cleaned\n")

    def failing_step(targets):
        raise RuntimeError("clock timeout")

    host.step = failing_step
    monkeypatch.setattr(worker, "_WorkerRuntimeHost", lambda *args, **kwargs: host)
    monkeypatch.setattr(worker, "_runtime_error_types", lambda: (RuntimeError,))
    requests = iter([{"command": "step", "joint_targets": []}, {"command": "shutdown"}])
    replies = []
    connection = SimpleNamespace(recv=lambda: next(requests), send=replies.append, close=lambda: None)
    for name in ("ROS_DOMAIN_ID", "GZ_PARTITION", "IGN_PARTITION"):
        monkeypatch.setenv(name, "unit-test")
    worker._worker_main(connection, client.config.payload(), None, None)
    assert replies[1]["type"] == "operation_error"
    assert replies[1]["launch_log_offset"] == 0
    assert replies[1]["gazebo_diagnostics"]["confirmed"] is True
    assert client.config.log_path.read_text() == "cleaned\n"


def test_communication_error_original_constructor_is_compatible():
    error = worker.GazeboWorkerCommunicationError("missing")
    assert str(error) == "missing"
    assert error.gazebo_diagnostics is None
