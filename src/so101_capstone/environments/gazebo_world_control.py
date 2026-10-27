"""Persistent ROS 2 client for deterministic Gazebo world control."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import ControlWorld


WORLD_CONTROL_SERVICE = "/world/empty/control"
MAX_MULTI_STEP = (2**32) - 1


class GazeboWorldControlError(RuntimeError):
    """Raised when Gazebo world control cannot be completed."""


def _positive_timeout(
    value: float,
    *,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive finite number."
        )

    timeout = float(value)

    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(
            f"{name} must be a positive finite number."
        )

    return timeout


class GazeboWorldController:
    """Control Gazebo through one persistent ROS service client."""

    def __init__(
        self,
        node: Node,
        *,
        service_name: str = WORLD_CONTROL_SERVICE,
        service_timeout_s: float = 5.0,
        call_timeout_s: float = 5.0,
    ) -> None:
        normalized_service_name = str(service_name).strip()

        if not normalized_service_name:
            raise ValueError(
                "World-control service name must not be empty."
            )

        self._node = node
        self._service_name = normalized_service_name
        self._service_timeout_s = _positive_timeout(
            service_timeout_s,
            name="service_timeout_s",
        )
        self._call_timeout_s = _positive_timeout(
            call_timeout_s,
            name="call_timeout_s",
        )

        self._client = self._node.create_client(
            ControlWorld,
            self._service_name,
        )

    @property
    def service_name(self) -> str:
        return self._service_name

    @property
    def service_timeout_s(self) -> float:
        return self._service_timeout_s

    @property
    def call_timeout_s(self) -> float:
        return self._call_timeout_s

    def wait_until_available(self) -> None:
        """Wait for the ROS-to-Gazebo service bridge."""

        available = self._client.wait_for_service(
            timeout_sec=self._service_timeout_s
        )

        if not available:
            raise GazeboWorldControlError(
                "Gazebo world-control service is unavailable: "
                f"{self._service_name}"
            )

    def _call(
        self,
        *,
        pause: bool,
        multi_step: int,
    ) -> ControlWorld.Response:
        self.wait_until_available()

        request = ControlWorld.Request()
        request.world_control.pause = pause
        request.world_control.multi_step = multi_step

        future = self._client.call_async(request)

        rclpy.spin_until_future_complete(
            self._node,
            future,
            timeout_sec=self._call_timeout_s,
        )

        if not future.done():
            raise GazeboWorldControlError(
                "Gazebo world-control request timed out."
            )

        error = future.exception()

        if error is not None:
            raise GazeboWorldControlError(
                "Gazebo world-control request failed."
            ) from error

        response = future.result()

        if response is None:
            raise GazeboWorldControlError(
                "Gazebo world-control service returned "
                "no response."
            )

        if not response.success:
            raise GazeboWorldControlError(
                "Gazebo rejected the world-control request."
            )

        return response

    def pause(self) -> ControlWorld.Response:
        """Pause simulation without advancing physics."""

        return self._call(
            pause=True,
            multi_step=0,
        )

    def resume(self) -> ControlWorld.Response:
        """Resume continuous simulation."""

        return self._call(
            pause=False,
            multi_step=0,
        )

    def step(
        self,
        physics_steps: int,
    ) -> ControlWorld.Response:
        """Advance a paused simulation by integer physics steps."""

        if (
            isinstance(physics_steps, bool)
            or not isinstance(physics_steps, int)
            or physics_steps < 1
            or physics_steps > MAX_MULTI_STEP
        ):
            raise ValueError(
                "physics_steps must be an integer from "
                f"1 to {MAX_MULTI_STEP}."
            )

        return self._call(
            pause=True,
            multi_step=physics_steps,
        )
