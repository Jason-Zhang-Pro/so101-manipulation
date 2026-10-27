"""Apply one sampled environment randomization to Gazebo Fortress."""

from __future__ import annotations

import json
import math
import subprocess
from typing import Any, Mapping

from .environment_randomization import (
    EnvironmentRandomizationError,
    EpisodeRandomizationSample,
)


class GazeboEnvironmentRandomizationController:
    """Send episode-scoped randomization to the world system plugin."""

    def __init__(
        self,
        *,
        world_name: str = "empty",
        timeout_ms: int = 5000,
    ) -> None:
        name = str(world_name).strip()
        if not name:
            raise ValueError("world_name must not be empty.")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
            raise ValueError("timeout_ms must be a positive integer.")
        self.world_name = name
        self.timeout_ms = timeout_ms
        self.service_name = (
            f"/world/{self.world_name}/so101_environment_randomization/apply"
        )

    @staticmethod
    def _rgb(values: Any) -> str:
        result = tuple(float(value) for value in values)
        if len(result) != 3 or not all(math.isfinite(value) for value in result):
            raise EnvironmentRandomizationError("RGB values must be a finite 3-vector.")
        return ",".join(f"{value:.12g}" for value in result)

    @staticmethod
    def _vec3(values: Any) -> str:
        result = tuple(float(value) for value in values)
        if len(result) != 3 or not all(math.isfinite(value) for value in result):
            raise EnvironmentRandomizationError("Vector values must be a finite 3-vector.")
        return ",".join(f"{value:.12g}" for value in result)

    def _payload(self, sample: EpisodeRandomizationSample) -> str:
        lines: list[str] = []

        scene = sample.scene
        lines.append(
            "background="
            + self._rgb(scene["background_color"]["rgb"])
        )
        lines.append(
            "ground="
            + self._rgb(scene["ground_color"]["rgb"])
        )
        lines.append(
            "table="
            + self._rgb(scene["table_color"]["rgb"])
        )
        lines.append(
            "target="
            + self._rgb(scene["target_color"]["rgb"])
        )

        goal = sample.goal_region
        lines.append(
            "goal_position_enabled="
            + ("1" if bool(goal["position"]["enabled"]) else "0")
        )
        lines.append(
            "goal_color_enabled="
            + ("1" if bool(goal["color"]["enabled"]) else "0")
        )
        lines.append(
            "goal_size_enabled="
            + ("1" if bool(goal["size"]["enabled"]) else "0")
        )
        lines.append(
            "goal_color="
            + self._rgb(goal["color"]["rgb"])
        )
        lines.append(
            "goal_center="
            + self._vec3(goal["position"]["center_world_m"])
        )
        lines.append(
            "goal_delta_xy="
            + ",".join(
                f"{float(value):.12g}"
                for value in goal["position"]["delta_xy_m"]
            )
        )
        lines.append(
            "goal_half_extents="
            + f"{float(goal['size']['half_extent_x_m']):.12g},"
            + f"{float(goal['size']['half_extent_y_m']):.12g}"
        )

        for camera_name, sensor_name in (
            ("top_camera", "top_camera_sensor"),
            ("gripper_camera", "gripper_camera_sensor"),
        ):
            pose = sample.cameras[camera_name]["pose"]
            lines.append(
                f"camera.{sensor_name}="
                + self._vec3(pose["position_delta_m"])
                + ","
                + self._vec3(pose["rotation_delta_deg"])
            )

        for light_name, values in sample.lights.items():
            lines.append(
                f"light.{light_name}="
                + f"{float(values['intensity_scale']):.12g},"
                + self._rgb(values["diffuse_rgb"])
                + ","
                + self._vec3(values["position_xyz_m"])
                + ","
                + ("1" if bool(values["cast_shadows"]) else "0")
            )

        return "\n".join(lines)

    def _call(self, payload: str) -> None:
        request_text = f"data: {json.dumps(payload)}"
        command = [
            "ign",
            "service",
            "-s",
            self.service_name,
            "--reqtype",
            "ignition.msgs.StringMsg",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            str(self.timeout_ms),
            "--req",
            request_text,
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = completed.stdout or ""
        if completed.returncode != 0 or "data: true" not in output.lower():
            raise EnvironmentRandomizationError(
                "Gazebo environment-randomization service failed: "
                f"service={self.service_name!r}, "
                f"returncode={completed.returncode}, "
                f"output={output.strip()!r}"
            )

    def apply_episode(self, sample: EpisodeRandomizationSample) -> None:
        """Queue one complete episode command before reset/warm-up."""
        self._call(self._payload(sample))

    def apply_target_instance(
        self,
        model_name: str,
        rgb: Any,
    ) -> None:
        """Queue target color before the next paused-world render step."""
        name = str(model_name).strip()
        if not name or "," in name or "\n" in name:
            raise EnvironmentRandomizationError(
                "Target model name is invalid for randomization."
            )
        self._call(f"target_instance={name},{self._rgb(rgb)}")
