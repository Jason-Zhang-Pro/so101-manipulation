"""Check the command path before a zero-pose evaluation starts its clock.

The caller must keep the world running while controllers are being activated.
This helper only reads controller status and ROS graph discovery; it never
activates controllers, publishes commands, or changes the simulation clock.
The supplied control node must not already be spun by another executor.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math
from time import monotonic, sleep, time
from typing import Any

from .state_cache import MEASURED_JOINT_NAMES


CONTROLLER_LIST_SERVICE = "/controller_manager/list_controllers"
POSITION_CONTROLLER_NAME = "position_controller"
_POLL_PERIOD_S = 0.05


class ControllerReadinessError(RuntimeError):
    """The position command path could not be confirmed before evaluation."""

    def __init__(self, message: str, *, last_status: dict[str, Any]) -> None:
        self.last_status = dict(last_status)
        super().__init__(f"{message} Last status: {json.dumps(last_status, sort_keys=True)}")


def wait_for_position_controller(
    control_node: Any,
    *,
    command_subscription_count: Callable[[], int],
    timeout_s: float,
) -> dict[str, Any]:
    """Require an active controller owning all six position interfaces.

    Controller services and command subscription discovery share one wall-time
    deadline. A zero-position joint-state message alone cannot establish that
    published policy actions will be received by the position controller.
    """

    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be positive and finite.")

    # Import only on the explicit evaluation path, preserving non-ROS imports
    # and the dependencies of callers that do not request this check.
    import rclpy
    from controller_manager_msgs.srv import ListControllers

    started = monotonic()
    deadline = started + timeout
    required = sorted(f"{name}/position" for name in MEASURED_JOINT_NAMES)
    status: dict[str, Any] = {
        "ready": False,
        "service_name": CONTROLLER_LIST_SERVICE,
        "service_available": False,
        "controller_name": POSITION_CONTROLLER_NAME,
        "controller_state": "not_observed",
        "required_interfaces": required,
        "claimed_interfaces": [],
        "missing_interfaces": required.copy(),
        "command_subscription_count": 0,
    }
    client = None
    future = None

    def fail(reason: str) -> ControllerReadinessError:
        status["waited_wall_s"] = max(0.0, monotonic() - started)
        status["checked_wall_time_s"] = time()
        return ControllerReadinessError(reason, last_status=status)

    try:
        client = control_node.create_client(ListControllers, CONTROLLER_LIST_SERVICE)
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise fail("Timed out waiting for the position controller command path.")
            status["command_subscription_count"] = int(command_subscription_count())
            status["service_available"] = bool(
                client.wait_for_service(timeout_sec=min(_POLL_PERIOD_S, remaining))
            )
            if not status["service_available"]:
                # wait_for_service normally consumes the interval. Also yield
                # when graph shutdown or a fake client returns immediately.
                sleep(min(_POLL_PERIOD_S, max(0.0, deadline - monotonic())))
                continue
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise fail("Timed out before querying the position controller state.")
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(control_node, future, timeout_sec=remaining)
            if not future.done():
                raise fail("Timed out waiting for the controller-manager ListControllers response.")
            error = future.exception()
            if error is not None:
                raise fail(f"Controller-manager ListControllers request failed: {error}") from error
            response = future.result()
            future = None
            if response is None:
                raise fail("Controller-manager ListControllers returned no response.")

            controller = next(
                (item for item in response.controller if item.name == POSITION_CONTROLLER_NAME),
                None,
            )
            status["controller_state"] = "not_loaded" if controller is None else str(controller.state)
            status["claimed_interfaces"] = [] if controller is None else sorted(
                str(interface) for interface in controller.claimed_interfaces
            )
            status["missing_interfaces"] = sorted(set(required) - set(status["claimed_interfaces"]))
            status["command_subscription_count"] = int(command_subscription_count())
            if monotonic() >= deadline:
                raise fail("Timed out waiting for the position controller command path.")
            if (status["controller_state"] == "active"
                    and not status["missing_interfaces"]
                    and status["command_subscription_count"] > 0):
                status["ready"] = True
                status["waited_wall_s"] = max(0.0, monotonic() - started)
                status["checked_wall_time_s"] = time()
                return status
            sleep(min(_POLL_PERIOD_S, max(0.0, deadline - monotonic())))
    except ControllerReadinessError:
        raise
    except Exception as error:
        raise fail(f"Failed to check the position controller command path: {error}") from error
    finally:
        try:
            if future is not None and not future.done():
                future.cancel()
        finally:
            if client is not None:
                control_node.destroy_client(client)
