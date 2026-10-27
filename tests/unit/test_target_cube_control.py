"""Tests for persistent target-cube Trigger service clients."""

from types import SimpleNamespace

import pytest

from so101_capstone.environments.target_cube_control import (
    DELETE_ALL_SERVICE,
    SPAWN_RANDOM_SERVICE,
    TargetCubeControlError,
    TargetCubeController,
)


class FakeTrigger:
    class Request:
        pass


class FakeFuture:
    def __init__(
        self,
        *,
        response=None,
        done=True,
        error=None,
    ):
        self._response = response
        self._done = done
        self._error = error

    def done(self):
        return self._done

    def exception(self):
        return self._error

    def result(self):
        return self._response


class FakeClient:
    def __init__(
        self,
        *,
        available=True,
        future=None,
    ):
        self.available = available
        self.future = (
            FakeFuture(
                response=SimpleNamespace(
                    success=True,
                    message="ok",
                )
            )
            if future is None
            else future
        )
        self.wait_timeouts = []
        self.requests = []

    def wait_for_service(self, *, timeout_sec):
        self.wait_timeouts.append(timeout_sec)
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        return self.future


class FakeNode:
    def __init__(
        self,
        spawn_client,
        delete_client,
    ):
        self.spawn_client = spawn_client
        self.delete_client = delete_client
        self.created_clients = []

    def create_client(
        self,
        service_type,
        service_name,
    ):
        self.created_clients.append(
            (service_type, service_name)
        )

        if service_name == SPAWN_RANDOM_SERVICE:
            return self.spawn_client

        if service_name == DELETE_ALL_SERVICE:
            return self.delete_client

        raise AssertionError(
            f"Unexpected service name: {service_name}"
        )


def make_controller(
    *,
    spawn_available=True,
    delete_available=True,
    spawn_future=None,
    delete_future=None,
):
    spawn_client = FakeClient(
        available=spawn_available,
        future=spawn_future,
    )
    delete_client = FakeClient(
        available=delete_available,
        future=delete_future,
    )
    node = FakeNode(
        spawn_client,
        delete_client,
    )
    spin_calls = []

    def fake_spin(
        spin_node,
        future,
        *,
        timeout_sec,
    ):
        spin_calls.append(
            (spin_node, future, timeout_sec)
        )

    controller = TargetCubeController(
        node,
        trigger_service_type=FakeTrigger,
        spin_until_future_complete=fake_spin,
    )

    return (
        controller,
        node,
        spawn_client,
        delete_client,
        spin_calls,
    )


def test_controller_creates_two_persistent_clients():
    controller, node, _, _, _ = make_controller()

    assert controller.spawn_service_name == (
        SPAWN_RANDOM_SERVICE
    )
    assert controller.delete_all_service_name == (
        DELETE_ALL_SERVICE
    )

    assert node.created_clients == [
        (FakeTrigger, SPAWN_RANDOM_SERVICE),
        (FakeTrigger, DELETE_ALL_SERVICE),
    ]


def test_fixed_spawn_uses_argument_vector_and_returns_requested_pose(tmp_path, monkeypatch):
    import json
    import so101_capstone.environments.target_cube_control as module
    controller, *_ = make_controller()
    sdf = tmp_path / "target model.sdf"
    sdf.write_text("<sdf/>")
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="data: true", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    result = controller.spawn_at([.2, -.1, .768], [0, 0, 0, 2], model_sdf_path=sdf)
    payload = json.loads(result.message)
    assert result.success and payload["name"].startswith("target_cube_")
    assert payload["requested_position"] == {"x": .2, "y": -.1, "z": .768}
    assert payload["requested_quaternion_xyzw"] == [0, 0, 0, 1]
    assert calls[0][0][:4] == ["ign", "service", "-s", "/world/empty/create"]
    assert "shell" not in calls[0][1]
    with pytest.raises(ValueError):
        controller.spawn_at([.2, 0, float("nan")], model_sdf_path=sdf)


def test_fixed_spawn_rejection_is_not_reported_as_success(tmp_path, monkeypatch):
    import so101_capstone.environments.target_cube_control as module
    controller, *_ = make_controller()
    sdf = tmp_path / "target.sdf"
    sdf.write_text("<sdf/>")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="data: false", stderr=""))
    with pytest.raises(TargetCubeControlError, match="rejected"):
        controller.spawn_at([.2, 0, .768], model_sdf_path=sdf)


def test_spawn_random_calls_expected_service():
    (
        controller,
        node,
        spawn_client,
        delete_client,
        spin_calls,
    ) = make_controller()

    response = controller.spawn_random()

    assert response.success is True
    assert response.message == "ok"
    assert len(spawn_client.requests) == 1
    assert isinstance(
        spawn_client.requests[0],
        FakeTrigger.Request,
    )
    assert delete_client.requests == []
    assert spawn_client.wait_timeouts == [5.0]
    assert spin_calls == [
        (
            node,
            spawn_client.future,
            5.0,
        )
    ]


def test_delete_all_calls_expected_service():
    (
        controller,
        _node,
        spawn_client,
        delete_client,
        _spin_calls,
    ) = make_controller()

    response = controller.delete_all()

    assert response.success is True
    assert len(delete_client.requests) == 1
    assert isinstance(
        delete_client.requests[0],
        FakeTrigger.Request,
    )
    assert spawn_client.requests == []


@pytest.mark.parametrize(
    ("method_name", "spawn_available", "delete_available"),
    [
        ("spawn_random", False, True),
        ("delete_all", True, False),
    ],
)
def test_unavailable_service_is_reported(
    method_name,
    spawn_available,
    delete_available,
):
    (
        controller,
        _node,
        _spawn_client,
        _delete_client,
        _spin_calls,
    ) = make_controller(
        spawn_available=spawn_available,
        delete_available=delete_available,
    )

    with pytest.raises(
        TargetCubeControlError,
        match="unavailable",
    ):
        getattr(controller, method_name)()


def test_service_timeout_is_reported():
    future = FakeFuture(
        response=None,
        done=False,
    )
    (
        controller,
        _node,
        _spawn_client,
        _delete_client,
        _spin_calls,
    ) = make_controller(
        spawn_future=future,
    )

    with pytest.raises(
        TargetCubeControlError,
        match="timed out",
    ):
        controller.spawn_random()


def test_service_exception_is_reported():
    future = FakeFuture(
        response=None,
        error=RuntimeError("transport failure"),
    )
    (
        controller,
        _node,
        _spawn_client,
        _delete_client,
        _spin_calls,
    ) = make_controller(
        delete_future=future,
    )

    with pytest.raises(
        TargetCubeControlError,
        match="request failed",
    ):
        controller.delete_all()


def test_missing_response_is_reported():
    future = FakeFuture(
        response=None,
    )
    (
        controller,
        _node,
        _spawn_client,
        _delete_client,
        _spin_calls,
    ) = make_controller(
        spawn_future=future,
    )

    with pytest.raises(
        TargetCubeControlError,
        match="no response",
    ):
        controller.spawn_random()


def test_rejected_response_includes_service_message():
    future = FakeFuture(
        response=SimpleNamespace(
            success=False,
            message="Maximum managed cube count reached",
        )
    )
    (
        controller,
        _node,
        _spawn_client,
        _delete_client,
        _spin_calls,
    ) = make_controller(
        spawn_future=future,
    )

    with pytest.raises(
        TargetCubeControlError,
        match="Maximum managed cube count reached",
    ):
        controller.spawn_random()


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("spawn_service_name", ""),
        ("spawn_service_name", "target_cube/spawn_random"),
        ("delete_all_service_name", ""),
        ("service_timeout_s", 0.0),
        ("service_timeout_s", float("nan")),
        ("call_timeout_s", -1.0),
        ("call_timeout_s", True),
    ],
)
def test_invalid_configuration_is_rejected(
    keyword,
    value,
):
    kwargs = {
        "trigger_service_type": FakeTrigger,
        "spin_until_future_complete": (
            lambda *_args, **_kwargs: None
        ),
        keyword: value,
    }

    node = FakeNode(
        FakeClient(),
        FakeClient(),
    )

    with pytest.raises(ValueError):
        TargetCubeController(
            node,
            **kwargs,
        )


def test_spawn_and_delete_service_names_must_differ():
    node = FakeNode(
        FakeClient(),
        FakeClient(),
    )

    with pytest.raises(
        ValueError,
        match="must differ",
    ):
        TargetCubeController(
            node,
            spawn_service_name="/same_service",
            delete_all_service_name="/same_service",
            trigger_service_type=FakeTrigger,
            spin_until_future_complete=(
                lambda *_args, **_kwargs: None
            ),
        )
