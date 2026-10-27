from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("ros_gz_interfaces.srv")

import so101_capstone.environments.gazebo_world_control as module
from so101_capstone.environments.gazebo_world_control import (
    MAX_MULTI_STEP,
    WORLD_CONTROL_SERVICE,
    GazeboWorldControlError,
    GazeboWorldController,
)
from ros_gz_interfaces.srv import ControlWorld


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
                response=SimpleNamespace(success=True)
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
    def __init__(self, client):
        self.client = client
        self.service_type = None
        self.service_name = None

    def create_client(self, service_type, service_name):
        self.service_type = service_type
        self.service_name = service_name
        return self.client


def make_controller(
    monkeypatch,
    *,
    available=True,
    future=None,
):
    client = FakeClient(
        available=available,
        future=future,
    )
    node = FakeNode(client)

    monkeypatch.setattr(
        module.rclpy,
        "spin_until_future_complete",
        lambda *_args, **_kwargs: None,
    )

    controller = GazeboWorldController(node)

    return controller, node, client


def test_controller_creates_expected_persistent_client(
    monkeypatch,
):
    controller, node, _ = make_controller(monkeypatch)

    assert controller.service_name == WORLD_CONTROL_SERVICE
    assert node.service_type is ControlWorld
    assert node.service_name == WORLD_CONTROL_SERVICE


def test_pause_request(monkeypatch):
    controller, _, client = make_controller(monkeypatch)

    response = controller.pause()
    request = client.requests[0]

    assert response.success is True
    assert request.world_control.pause is True
    assert request.world_control.multi_step == 0


def test_resume_request(monkeypatch):
    controller, _, client = make_controller(monkeypatch)

    controller.resume()
    request = client.requests[0]

    assert request.world_control.pause is False
    assert request.world_control.multi_step == 0


def test_step_request(monkeypatch):
    controller, _, client = make_controller(monkeypatch)

    controller.step(33)
    request = client.requests[0]

    assert request.world_control.pause is True
    assert request.world_control.multi_step == 33


@pytest.mark.parametrize(
    "physics_steps",
    [
        0,
        -1,
        True,
        MAX_MULTI_STEP + 1,
    ],
)
def test_invalid_physics_step_count_is_rejected(
    monkeypatch,
    physics_steps,
):
    controller, _, client = make_controller(monkeypatch)

    with pytest.raises(ValueError, match="physics_steps"):
        controller.step(physics_steps)

    assert client.requests == []


def test_unavailable_service_is_reported(monkeypatch):
    controller, _, client = make_controller(
        monkeypatch,
        available=False,
    )

    with pytest.raises(
        GazeboWorldControlError,
        match="unavailable",
    ):
        controller.pause()

    assert client.requests == []


def test_service_call_timeout_is_reported(monkeypatch):
    future = FakeFuture(
        response=None,
        done=False,
    )
    controller, _, _ = make_controller(
        monkeypatch,
        future=future,
    )

    with pytest.raises(
        GazeboWorldControlError,
        match="timed out",
    ):
        controller.step(33)


def test_service_exception_is_reported(monkeypatch):
    future = FakeFuture(
        response=None,
        error=RuntimeError("transport failure"),
    )
    controller, _, _ = make_controller(
        monkeypatch,
        future=future,
    )

    with pytest.raises(
        GazeboWorldControlError,
        match="request failed",
    ):
        controller.pause()


def test_rejected_request_is_reported(monkeypatch):
    future = FakeFuture(
        response=SimpleNamespace(success=False),
    )
    controller, _, _ = make_controller(
        monkeypatch,
        future=future,
    )

    with pytest.raises(
        GazeboWorldControlError,
        match="rejected",
    ):
        controller.step(33)
