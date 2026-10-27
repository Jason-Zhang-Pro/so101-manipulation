from collections import deque
from types import SimpleNamespace
import sys

import pytest

from so101_capstone.environments import controller_readiness as readiness
from so101_capstone.environments.state_cache import MEASURED_JOINT_NAMES


INTERFACES = [f"{name}/position" for name in MEASURED_JOINT_NAMES]


def controller_response(state="active", interfaces=None):
    return SimpleNamespace(controller=[SimpleNamespace(
        name="position_controller", state=state,
        claimed_interfaces=INTERFACES if interfaces is None else interfaces,
    )])


class FakeFuture:
    def __init__(self, response=None, *, error=None, never_complete=False):
        self.response = response
        self.error = error
        self.never_complete = never_complete
        self.completed = False
        self.cancelled = False

    def done(self):
        return self.completed

    def exception(self):
        return self.error

    def result(self):
        return self.response

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def service(monkeypatch):
    class Service:
        def __init__(self):
            self.now = 0.0
            self.available = True
            self.availability = deque()
            self.responses = deque([controller_response()])
            self.subscribers = 1
            self.created = []
            self.destroyed = []
            self.futures = []
            self.spin_calls = []
            self.query_count = 0
            self.response_delay = 0.01
            self.node = SimpleNamespace(create_client=self.create_client, destroy_client=self.destroy_client)

        def sleep(self, duration):
            assert duration >= 0.0
            self.now += duration

        def create_client(self, service_type, name):
            self.created.append((service_type, name))
            return self

        def destroy_client(self, client):
            self.destroyed.append(client)

        def wait_for_service(self, *, timeout_sec):
            assert timeout_sec > 0.0
            available = self.availability.popleft() if self.availability else self.available
            if not available:
                self.now += timeout_sec
            return available

        def call_async(self, request):
            self.query_count += 1
            response = self.responses.popleft() if len(self.responses) > 1 else self.responses[0]
            future = response if isinstance(response, FakeFuture) else FakeFuture(response)
            self.futures.append(future)
            return future

        def spin(self, node, future, *, timeout_sec):
            assert node is self.node
            assert timeout_sec > 0.0
            self.spin_calls.append(timeout_sec)
            if future.never_complete:
                self.now += timeout_sec
            else:
                self.now += self.response_delay
                future.completed = True

        def count(self):
            return self.subscribers() if callable(self.subscribers) else self.subscribers

        def wait(self, timeout=0.3):
            return readiness.wait_for_position_controller(
                self.node, command_subscription_count=self.count, timeout_s=timeout,
            )

    instance = Service()
    monkeypatch.setattr(readiness, "monotonic", lambda: instance.now)
    monkeypatch.setattr(readiness, "sleep", instance.sleep)
    monkeypatch.setattr(readiness, "time", lambda: 1_800_000_000.0 + instance.now)
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(spin_until_future_complete=instance.spin))
    service_type = SimpleNamespace(Request=lambda: object())
    monkeypatch.setitem(sys.modules, "controller_manager_msgs", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "controller_manager_msgs.srv", SimpleNamespace(ListControllers=service_type))
    return instance


def test_waits_for_service_loaded_controller_and_active_interfaces(service):
    service.availability.extend([False, True])
    service.responses = deque([
        SimpleNamespace(controller=[]),
        controller_response("inactive", []),
        controller_response(interfaces=list(reversed(INTERFACES))),
    ])
    result = service.wait(timeout=0.5)
    assert result["ready"] is True
    assert result["controller_state"] == "active"
    assert result["controller_name"] == "position_controller"
    assert result["service_available"] is True
    assert result["claimed_interfaces"] == sorted(INTERFACES)
    assert result["missing_interfaces"] == []
    assert result["command_subscription_count"] == 1
    assert result["waited_wall_s"] == pytest.approx(service.now)
    assert result["checked_wall_time_s"] == pytest.approx(1_800_000_000 + service.now)
    assert service.query_count == 3
    assert service.created[0][1] == "/controller_manager/list_controllers"
    assert service.destroyed == [service]


def test_active_controller_still_waits_for_command_subscription(service):
    service.subscribers = lambda: int(service.query_count >= 3)
    result = service.wait()
    assert result["ready"] is True
    assert service.query_count == 3


@pytest.mark.parametrize("state,interfaces,subscribers,expected", [
    ("inactive", INTERFACES, 1, "inactive"),
    ("active", INTERFACES[:-1], 1, "gripper/position"),
    ("active", [], 1, "missing_interfaces"),
    ("active", [name.replace("/position", "/velocity") for name in INTERFACES], 1, "gripper/position"),
    ("active", INTERFACES, 0, '"command_subscription_count": 0'),
])
def test_never_accepts_incomplete_command_path(service, state, interfaces, subscribers, expected):
    service.responses = deque([controller_response(state, interfaces)])
    service.subscribers = subscribers
    with pytest.raises(readiness.ControllerReadinessError, match="Timed out") as caught:
        service.wait(timeout=0.2)
    assert expected in str(caught.value)
    assert caught.value.last_status["ready"] is False
    assert service.destroyed == [service]
    assert service.now == pytest.approx(0.2)


def test_service_unavailable_consumes_one_total_deadline(service):
    service.available = False
    with pytest.raises(readiness.ControllerReadinessError) as caught:
        service.wait(timeout=0.16)
    assert caught.value.last_status["service_available"] is False
    assert caught.value.last_status["controller_state"] == "not_observed"
    assert service.query_count == 0
    assert service.now == pytest.approx(0.16)
    assert service.destroyed == [service]


def test_unanswered_response_is_cancelled_and_client_destroyed(service):
    pending = FakeFuture(never_complete=True)
    service.responses = deque([pending])
    service.availability.extend([False, True])
    with pytest.raises(readiness.ControllerReadinessError, match="ListControllers response"):
        service.wait(timeout=0.25)
    assert service.spin_calls == [pytest.approx(0.15)]
    assert service.now == pytest.approx(0.25)
    assert pending.cancelled is True
    assert service.destroyed == [service]


@pytest.mark.parametrize("future,expected", [
    (FakeFuture(error=RuntimeError("service failed")), "service failed"),
    (FakeFuture(response=None), "returned no response"),
])
def test_service_errors_fail_closed_and_cleanup(service, future, expected):
    service.responses = deque([future])
    with pytest.raises(readiness.ControllerReadinessError, match=expected):
        service.wait()
    assert service.destroyed == [service]


def test_ready_response_after_deadline_cannot_start_evaluation(service):
    service.response_delay = 0.31
    with pytest.raises(readiness.ControllerReadinessError, match="Timed out"):
        service.wait(timeout=0.3)
    assert service.destroyed == [service]


def test_wrong_controller_name_cannot_satisfy_readiness(service):
    response = controller_response()
    response.controller[0].name = "joint_state_broadcaster"
    service.responses = deque([response])
    with pytest.raises(readiness.ControllerReadinessError) as caught:
        service.wait()
    assert caught.value.last_status["controller_state"] == "not_loaded"


def test_additional_claimed_interfaces_are_allowed(service):
    service.responses = deque([controller_response(interfaces=INTERFACES + ["extra/position"])])
    assert service.wait()["ready"] is True


def test_graph_inspection_failure_is_reported_and_client_destroyed(service):
    def broken_count():
        raise RuntimeError("graph unavailable")

    service.subscribers = broken_count
    with pytest.raises(readiness.ControllerReadinessError, match="graph unavailable"):
        service.wait()
    assert service.destroyed == [service]


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_timeout_must_be_positive_finite_before_creating_client(service, timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        service.wait(timeout)
    assert not service.created
