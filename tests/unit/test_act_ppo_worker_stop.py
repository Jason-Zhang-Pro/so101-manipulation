"""Stop acknowledgements and deadlines without ROS, Gazebo, or child processes."""

from types import SimpleNamespace

import pytest

from so101_capstone.training import act_ppo_multiworld_worker as worker


def make_client(tmp_path, *, alive=True, response=None, launch_timeout_s=10.0):
    client = object.__new__(worker.GazeboWorkerClient)
    client.config = worker.GazeboWorkerConfig(
        world_index=2,
        ros_domain_id=22,
        gz_partition="gz_env_22",
        gazebo_gui=False,
        project_root=tmp_path,
        output_directory=tmp_path,
        startup_feedback_timeout_s=2.0,
        launch_shutdown_timeout_s=launch_timeout_s,
    )
    events = []
    if response is None:
        response = {"type": "world_stopped", "world_index": 2, "launch_pid": None}

    def poll(timeout_s):
        events.append(("poll", timeout_s))
        return True

    def receive():
        events.append(("receive",))
        return response

    client.process = SimpleNamespace(
        is_alive=lambda: alive,
        join=lambda **kwargs: events.append(("join", kwargs)),
    )
    client.parent_connection = SimpleNamespace(
        send=lambda payload: events.append(("send", payload)),
        poll=poll,
        recv=receive,
        close=lambda: events.append(("close",)),
    )
    client.launch_pid = 1234
    client.launch_log_offset = 0
    client._closed = False
    client.gazebo_diagnostics = lambda: {"source": "unit-test"}
    client._hard_kill = lambda: events.append(("hard_kill",))
    return client, events


def test_begin_stop_only_dispatches_cleanup(tmp_path):
    client, events = make_client(tmp_path)

    assert client.begin_stop() is None

    assert events == [("send", {"command": "stop_world"})]
    assert client.launch_pid == 1234
    assert client.alive


def test_stop_confirms_cleanup_and_reuses_worker_for_restart(tmp_path):
    client, events = make_client(tmp_path)
    process = client.process

    response = client.stop_world()

    assert response == {"type": "world_stopped", "world_index": 2, "launch_pid": None}
    assert events == [
        ("send", {"command": "stop_world"}),
        ("poll", 84.0),
        ("receive",),
    ]
    assert client.launch_pid is None
    assert client.process is process
    assert client.alive
    assert client.begin_restart(reason="episode_complete") == "runtime_restart"
    assert events[-1] == (
        "send", {"command": "restart_world", "reason": "episode_complete"},
    )


@pytest.mark.parametrize(
    "response",
    [
        {"type": "operation_error", "error_message": "cleanup still has processes"},
        {"type": "ready", "world_index": 2},
        {"type": "shutdown_complete", "world_index": 2},
        {"type": "world_stopped", "world_index": 1},
        {"type": "world_stopped", "world_index": "2"},
        {"type": "world_stopped"},
        {},
    ],
)
def test_stop_rejects_error_or_wrong_acknowledgement(tmp_path, response):
    client, events = make_client(tmp_path, response=response)

    with pytest.raises(worker.GazeboWorkerCommunicationError, match="cleanup was not confirmed") as raised:
        client.stop_world()

    assert repr(response) in str(raised.value)
    assert raised.value.gazebo_diagnostics == {"source": "unit-test"}
    if response.get("type") != "shutdown_complete":
        assert client.launch_pid == 1234
    assert ("hard_kill",) not in events


def test_dead_worker_does_not_count_as_confirmed_cleanup(tmp_path):
    client, events = make_client(tmp_path, alive=False)

    with pytest.raises(worker.GazeboWorkerCommunicationError, match="cleanup cannot be confirmed"):
        client.stop_world()

    assert events == []
    assert client.launch_pid == 1234


@pytest.mark.parametrize("launch_timeout_s", [0.5, 10.0, 60.0])
def test_stop_waits_for_entire_host_cleanup_budget(tmp_path, launch_timeout_s):
    client, events = make_client(tmp_path, launch_timeout_s=launch_timeout_s)
    scoped_timeout_s = min(5.0, launch_timeout_s)
    # Runtime executor shutdown + thread join, launch TERM + KILL, and all
    # eight scoped TERM/KILL/settle/scan intervals must fit before timeout.
    host_wait_budget_s = (
        2 * 5.0
        + launch_timeout_s + 2.0
        + 8 * (scoped_timeout_s + min(1.0, scoped_timeout_s) + 0.25 + 0.25)
    )

    client.stop_world()

    timeout_s = next(event[1] for event in events if event[0] == "poll")
    assert timeout_s >= host_wait_budget_s + 10.0


def test_shutdown_uses_same_cleanup_budget(tmp_path):
    client, events = make_client(
        tmp_path,
        response={"type": "shutdown_complete", "world_index": 2},
    )

    client.shutdown()

    assert events[:3] == [
        ("send", {"command": "shutdown"}),
        ("poll", 84.0),
        ("receive",),
    ]


def test_stop_timeout_remains_a_communication_failure(tmp_path):
    client, events = make_client(tmp_path)
    client.parent_connection.poll = lambda timeout_s: False

    with pytest.raises(worker.GazeboWorkerCommunicationError, match="timed out after 84.0 s"):
        client.stop_world()

    assert events == [("send", {"command": "stop_world"}), ("hard_kill",)]
