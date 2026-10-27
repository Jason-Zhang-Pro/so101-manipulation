"""Persistent ROS 2 clients for target-cube lifecycle services."""

from __future__ import annotations

import math
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from uuid import uuid4
from typing import Any, Callable

import numpy as np


SPAWN_RANDOM_SERVICE = "/target_cube/spawn_random"
DELETE_ALL_SERVICE = "/target_cube/delete_all"


class TargetCubeControlError(RuntimeError):
    """Raised when a target-cube service call cannot complete."""


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


def _service_name(
    value: str,
    *,
    name: str,
) -> str:
    normalized = str(value).strip()

    if not normalized:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not normalized.startswith("/"):
        raise ValueError(
            f"{name} must be an absolute ROS service name."
        )

    return normalized


class TargetCubeController:
    """Call target-cube spawn and delete-all Trigger services.

    ROS dependencies are imported lazily so the module's validation and
    service-control logic can be unit tested outside a sourced ROS
    environment.
    """

    def __init__(
        self,
        node: Any,
        *,
        spawn_service_name: str = SPAWN_RANDOM_SERVICE,
        delete_all_service_name: str = DELETE_ALL_SERVICE,
        service_timeout_s: float = 5.0,
        call_timeout_s: float = 5.0,
        trigger_service_type: Any | None = None,
        spin_until_future_complete: Callable[..., Any] | None = None,
    ) -> None:
        self._node = node
        self._spawn_service_name = _service_name(
            spawn_service_name,
            name="spawn_service_name",
        )
        self._delete_all_service_name = _service_name(
            delete_all_service_name,
            name="delete_all_service_name",
        )

        if (
            self._spawn_service_name
            == self._delete_all_service_name
        ):
            raise ValueError(
                "Spawn and delete-all service names must differ."
            )

        self._service_timeout_s = _positive_timeout(
            service_timeout_s,
            name="service_timeout_s",
        )
        self._call_timeout_s = _positive_timeout(
            call_timeout_s,
            name="call_timeout_s",
        )

        if trigger_service_type is None:
            from std_srvs.srv import Trigger

            trigger_service_type = Trigger

        if spin_until_future_complete is None:
            import rclpy

            spin_until_future_complete = (
                rclpy.spin_until_future_complete
            )

        self._trigger_service_type = trigger_service_type
        self._spin_until_future_complete = (
            spin_until_future_complete
        )

        self._spawn_client = self._node.create_client(
            self._trigger_service_type,
            self._spawn_service_name,
        )
        self._delete_all_client = self._node.create_client(
            self._trigger_service_type,
            self._delete_all_service_name,
        )

    @property
    def spawn_service_name(self) -> str:
        return self._spawn_service_name

    @property
    def delete_all_service_name(self) -> str:
        return self._delete_all_service_name

    @property
    def service_timeout_s(self) -> float:
        return self._service_timeout_s

    @property
    def call_timeout_s(self) -> float:
        return self._call_timeout_s

    def _call(
        self,
        *,
        client: Any,
        service_name: str,
    ) -> Any:
        available = client.wait_for_service(
            timeout_sec=self._service_timeout_s
        )

        if not available:
            raise TargetCubeControlError(
                "Target-cube service is unavailable: "
                f"{service_name}"
            )

        request = self._trigger_service_type.Request()
        future = client.call_async(request)

        self._spin_until_future_complete(
            self._node,
            future,
            timeout_sec=self._call_timeout_s,
        )

        if not future.done():
            raise TargetCubeControlError(
                "Target-cube service request timed out: "
                f"{service_name}"
            )

        error = future.exception()

        if error is not None:
            raise TargetCubeControlError(
                "Target-cube service request failed: "
                f"{service_name}"
            ) from error

        response = future.result()

        if response is None:
            raise TargetCubeControlError(
                "Target-cube service returned no response: "
                f"{service_name}"
            )

        if not bool(response.success):
            message = str(
                getattr(response, "message", "")
            ).strip()

            detail = (
                f": {message}"
                if message
                else ""
            )

            raise TargetCubeControlError(
                "Target-cube service rejected the request: "
                f"{service_name}{detail}"
            )

        return response

    def spawn_random(self) -> Any:
        """Spawn one randomly positioned target cube."""

        return self._call(
            client=self._spawn_client,
            service_name=self._spawn_service_name,
        )

    def delete_all(self) -> Any:
        """Delete every target cube managed by the cube manager."""

        return self._call(
            client=self._delete_all_client,
            service_name=self._delete_all_service_name,
        )

    def spawn_at(self, position_world_m, quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                 *, model_sdf_path: str | Path | None = None, world_name: str = "empty") -> Any:
        """Create a deterministic target using the manager's native Gazebo API.

        The existing manager discovers all ``target_cube_`` model names from
        dynamic poses, so deletion and feedback use its ordinary registry.
        The service reply only confirms creation; callers must await feedback.
        """
        position = np.asarray(position_world_m, dtype=float)
        quaternion = np.asarray(quaternion_xyzw, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_world_m must be a finite 3-vector.")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) < 1e-12:
            raise ValueError("quaternion_xyzw must be a finite nonzero 4-vector.")
        quaternion /= np.linalg.norm(quaternion)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", world_name):
            raise ValueError("Invalid Gazebo world name.")
        if model_sdf_path is None:
            from ament_index_python.packages import get_package_share_directory
            model_sdf_path = Path(get_package_share_directory("so101_gazebo")) / "models/target_cube/model.sdf"
        model_path = Path(model_sdf_path).resolve()
        if not model_path.is_file():
            raise TargetCubeControlError(f"Target model does not exist: {model_path}")
        name = "target_cube_fixed_" + uuid4().hex
        request = (
            f"sdf_filename: {json.dumps(str(model_path))} name: {json.dumps(name)} "
            "allow_renaming: false pose: { position: { "
            + " ".join(f"{axis}: {value:.17g}" for axis, value in zip(("x", "y", "z"), position))
            + " } orientation: { "
            + " ".join(f"{axis}: {value:.17g}" for axis, value in zip(("x", "y", "z", "w"), quaternion))
            + " } }"
        )
        try:
            completed = subprocess.run(
                ["ign", "service", "-s", f"/world/{world_name}/create", "--reqtype",
                 "ignition.msgs.EntityFactory", "--reptype", "ignition.msgs.Boolean",
                 "--timeout", str(round(self._call_timeout_s * 1000)), "--req", request],
                capture_output=True, text=True, timeout=self._call_timeout_s + 1.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TargetCubeControlError("Deterministic target creation failed.") from error
        output = completed.stdout.strip() or completed.stderr.strip()
        if completed.returncode != 0 or re.search(r"\bdata:\s*true\b", output) is None:
            raise TargetCubeControlError(f"Gazebo rejected deterministic target creation: {output}")
        return SimpleNamespace(success=True, message=json.dumps({
            "name": name, "requested_position": dict(zip(("x", "y", "z"), position.tolist())),
            "requested_quaternion_xyzw": quaternion.tolist(),
        }))
