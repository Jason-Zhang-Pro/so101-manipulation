"""Episode-scoped environment and policy-observation randomization.

The master switch lives outside this module. When disabled, callers should
not instantiate :class:`EpisodeEnvironmentRandomizer`, which keeps the
existing simulation path free of randomization work.

One randomization sample is drawn per episode. Scene, task, camera geometry,
and fixed image-transform parameters remain constant throughout that episode.
RGB image noise and joint-state measurement noise are intentionally fresh on
every policy observation while their configured amplitudes remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .policy_observation import PolicyObservation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENVIRONMENT_RANDOMIZATION_CONFIG = (
    PROJECT_ROOT / "config" / "environment_randomization.json"
)
DEFAULT_IMAGE_AUGMENTATION_CONFIG = PROJECT_ROOT / "config" / "image_augmentation_s0_s1.json"

_BASE_BACKGROUND_RGB = (0.42, 0.46, 0.50)
_BASE_GROUND_RGB = (0.20, 0.24, 0.28)
_BASE_TABLE_RGB = (0.92, 0.92, 0.90)
_BASE_TARGET_RGB = (0.70, 0.015, 0.010)
_BASE_GOAL_RGB = (0.0, 0.0, 0.0)
_BASE_LIGHT_RGB = (0.225, 0.225, 0.225)

_BASE_LIGHT_POSITIONS = {
    "fill_light_front_left": (-0.50, 0.85, 2.50),
    "fill_light_front_right": (-0.50, -0.85, 2.50),
    "fill_light_back_left": (1.10, 0.85, 2.50),
    "fill_light_back_right": (1.10, -0.85, 2.50),
}


class EnvironmentRandomizationError(RuntimeError):
    """Raised when randomization configuration or application is invalid."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentRandomizationError(f"{name} must be an object.")
    return value


def _enabled(node: Mapping[str, Any]) -> bool:
    value = node.get("enabled", False)
    if not isinstance(value, bool):
        raise EnvironmentRandomizationError("Randomization enabled fields must be bool.")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise EnvironmentRandomizationError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise EnvironmentRandomizationError(f"{name} must be finite.")
    return result


def _range2(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EnvironmentRandomizationError(f"{name} must contain [min, max].")
    if len(value) != 2:
        raise EnvironmentRandomizationError(f"{name} must contain exactly 2 values.")
    low = _finite_float(value[0], name=f"{name}[0]")
    high = _finite_float(value[1], name=f"{name}[1]")
    if low > high:
        raise EnvironmentRandomizationError(f"{name} min must not exceed max.")
    return low, high


def _amplitude3(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EnvironmentRandomizationError(f"{name} must contain 3 values.")
    if len(value) != 3:
        raise EnvironmentRandomizationError(f"{name} must contain exactly 3 values.")
    result = tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    if any(item < 0.0 for item in result):
        raise EnvironmentRandomizationError(f"{name} amplitudes must be non-negative.")
    return result  # type: ignore[return-value]


def _clamp_rgb(values: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in values)  # type: ignore[return-value]


def _sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low == high:
        return low
    return float(rng.uniform(low, high))


def _sample_delta3(
    rng: np.random.Generator,
    amplitudes: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        0.0 if amplitude == 0.0 else float(rng.uniform(-amplitude, amplitude))
        for amplitude in amplitudes
    )  # type: ignore[return-value]


def _sample_rgb_delta(
    rng: np.random.Generator,
    base_rgb: Sequence[float],
    amplitudes: tuple[float, float, float],
) -> tuple[float, float, float]:
    delta = _sample_delta3(rng, amplitudes)
    return _clamp_rgb(tuple(float(base) + change for base, change in zip(base_rgb, delta)))


def _derive_seed(base_seed: int, world_index: int, episode_key: int) -> int:
    payload = f"{int(base_seed)}:{int(world_index)}:{int(episode_key)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class EnvironmentRandomizationConfig:
    """Validated JSON-backed randomization configuration."""

    payload: Mapping[str, Any]
    source_path: Path

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
    ) -> "EnvironmentRandomizationConfig":
        resolved = (
            DEFAULT_ENVIRONMENT_RANDOMIZATION_CONFIG
            if path is None
            else Path(path).expanduser().resolve()
        )
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except OSError as error:
            raise EnvironmentRandomizationError(
                f"Could not read environment-randomization config: {resolved}"
            ) from error
        except json.JSONDecodeError as error:
            raise EnvironmentRandomizationError(
                f"Invalid JSON in environment-randomization config: {resolved}: {error}"
            ) from error

        root = _mapping(payload, name="environment randomization config")
        cls._validate(root)
        return cls(payload=root, source_path=resolved)

    @staticmethod
    def _validate(payload: Mapping[str, Any]) -> None:
        for required in (
            "scene",
            "lights",
            "top_camera",
            "gripper_camera",
            "joint_state",
            "home_initial_pose",
            "goal_region",
        ):
            _mapping(payload.get(required), name=required)

        scene = _mapping(payload["scene"], name="scene")
        for name in ("background_color", "ground_color", "table_color", "target_color"):
            node = _mapping(scene.get(name), name=f"scene.{name}")
            _enabled(node)
            _amplitude3(node.get("rgb_delta", (0.0, 0.0, 0.0)), name=f"scene.{name}.rgb_delta")

        lights = _mapping(payload["lights"], name="lights")
        for light_name in _BASE_LIGHT_POSITIONS:
            light = _mapping(lights.get(light_name), name=f"lights.{light_name}")
            for feature_name in ("intensity", "color", "position", "shadows"):
                feature = _mapping(
                    light.get(feature_name),
                    name=f"lights.{light_name}.{feature_name}",
                )
                _enabled(feature)
            _range2(light["intensity"].get("scale_range", (1.0, 1.0)), name=f"{light_name}.intensity.scale_range")
            _amplitude3(light["color"].get("rgb_gain_amplitude", (0.0, 0.0, 0.0)), name=f"{light_name}.color.rgb_gain_amplitude")
            _amplitude3(light["position"].get("xyz_amplitude_m", (0.0, 0.0, 0.0)), name=f"{light_name}.position.xyz_amplitude_m")
            probability = _finite_float(light["shadows"].get("probability_on", 0.0), name=f"{light_name}.shadows.probability_on")
            if not 0.0 <= probability <= 1.0:
                raise EnvironmentRandomizationError(
                    f"{light_name}.shadows.probability_on must be within [0, 1]."
                )

        for camera_name in ("top_camera", "gripper_camera"):
            camera = _mapping(payload[camera_name], name=camera_name)
            pose = _mapping(camera.get("pose"), name=f"{camera_name}.pose")
            _enabled(pose)
            _amplitude3(pose.get("position_amplitude_m", (0.0, 0.0, 0.0)), name=f"{camera_name}.pose.position_amplitude_m")
            _amplitude3(pose.get("rotation_amplitude_deg", (0.0, 0.0, 0.0)), name=f"{camera_name}.pose.rotation_amplitude_deg")

            image = _mapping(camera.get("image"), name=f"{camera_name}.image")
            for feature_name in (
                "brightness",
                "contrast",
                "gamma",
                "saturation",
                "white_balance_rgb_gain",
                "rgb_noise",
                "blur",
                "translation",
                "distortion",
                "crop",
            ):
                feature = _mapping(
                    image.get(feature_name),
                    name=f"{camera_name}.image.{feature_name}",
                )
                _enabled(feature)

        joint_state = _mapping(payload["joint_state"], name="joint_state")
        for feature_name in ("position_noise", "velocity_noise"):
            feature = _mapping(joint_state.get(feature_name), name=f"joint_state.{feature_name}")
            _enabled(feature)

        home = _mapping(payload["home_initial_pose"], name="home_initial_pose")
        _enabled(home)
        max_deg = _finite_float(home.get("toward_zero_max_deg", 0.0), name="home_initial_pose.toward_zero_max_deg")
        if max_deg < 0.0:
            raise EnvironmentRandomizationError(
                "home_initial_pose.toward_zero_max_deg must be non-negative."
            )

        goal = _mapping(payload["goal_region"], name="goal_region")
        for feature_name in ("position", "color", "size"):
            feature = _mapping(goal.get(feature_name), name=f"goal_region.{feature_name}")
            _enabled(feature)


@dataclass(frozen=True)
class EpisodeRandomizationSample:
    """All fixed random values for one episode."""

    base_seed: int
    world_index: int
    episode_key: int
    derived_seed: int
    config_path: str
    scene: Mapping[str, Any]
    lights: Mapping[str, Any]
    cameras: Mapping[str, Any]
    joint_state: Mapping[str, Any]
    home_initial_pose: Mapping[str, Any]
    goal_region: Mapping[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "base_seed": self.base_seed,
                "world_index": self.world_index,
                "episode_key": self.episode_key,
                "derived_seed": self.derived_seed,
                "config_path": self.config_path,
                "scene": self.scene,
                "lights": self.lights,
                "cameras": self.cameras,
                "joint_state": self.joint_state,
                "home_initial_pose": self.home_initial_pose,
                "goal_region": self.goal_region,
            }
        )


class EpisodeEnvironmentRandomizer:
    """Sample one environment per episode and randomize policy observations."""

    def __init__(
        self,
        config: EnvironmentRandomizationConfig,
        *,
        base_seed: int = 0,
        world_index: int = 0,
    ) -> None:
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise ValueError("base_seed must be an integer.")
        if isinstance(world_index, bool) or not isinstance(world_index, int) or world_index < 0:
            raise ValueError("world_index must be a non-negative integer.")

        self.config = config
        self.base_seed = base_seed
        self.world_index = world_index
        self.current_sample: EpisodeRandomizationSample | None = None
        self._frame_rng: np.random.Generator | None = None
        self._geometry_cache: dict[tuple[str, tuple[int, int], float, float], tuple[np.ndarray, np.ndarray]] = {}

    def begin_episode(
        self,
        *,
        episode_key: int,
        goal_center_world_m: Sequence[float],
        goal_half_extent_x_m: float,
        goal_half_extent_y_m: float,
        home_joint_positions_rad: Sequence[float],
    ) -> EpisodeRandomizationSample:
        if isinstance(episode_key, bool) or not isinstance(episode_key, int) or episode_key < 0:
            raise ValueError("episode_key must be a non-negative integer.")

        goal_center = np.asarray(goal_center_world_m, dtype=float)
        if goal_center.shape != (3,) or not np.all(np.isfinite(goal_center)):
            raise ValueError("goal_center_world_m must be a finite 3-vector.")

        home = np.asarray(home_joint_positions_rad, dtype=float)
        if home.ndim != 1 or home.size == 0 or not np.all(np.isfinite(home)):
            raise ValueError("home_joint_positions_rad must be a finite vector.")

        derived_seed = _derive_seed(self.base_seed, self.world_index, episode_key)
        rng = np.random.default_rng(derived_seed)

        payload = self.config.payload
        scene_cfg = _mapping(payload["scene"], name="scene")

        scene_values: dict[str, Any] = {}
        scene_bases = {
            "background_color": _BASE_BACKGROUND_RGB,
            "ground_color": _BASE_GROUND_RGB,
            "table_color": _BASE_TABLE_RGB,
            "target_color": _BASE_TARGET_RGB,
        }
        for name, base in scene_bases.items():
            cfg = _mapping(scene_cfg[name], name=f"scene.{name}")
            if _enabled(cfg):
                color = _sample_rgb_delta(
                    rng,
                    base,
                    _amplitude3(cfg.get("rgb_delta", (0.0, 0.0, 0.0)), name=f"scene.{name}.rgb_delta"),
                )
            else:
                color = tuple(base)
            scene_values[name] = {
                "enabled": _enabled(cfg),
                "rgb": color,
            }

        lights_cfg = _mapping(payload["lights"], name="lights")
        lights_values: dict[str, Any] = {}
        for light_name, base_position in _BASE_LIGHT_POSITIONS.items():
            cfg = _mapping(lights_cfg[light_name], name=f"lights.{light_name}")

            intensity_cfg = _mapping(cfg["intensity"], name=f"{light_name}.intensity")
            intensity_scale = (
                _sample_range(
                    rng,
                    _range2(
                        intensity_cfg.get("scale_range", (1.0, 1.0)),
                        name=f"{light_name}.intensity.scale_range",
                    ),
                )
                if _enabled(intensity_cfg)
                else 1.0
            )

            color_cfg = _mapping(cfg["color"], name=f"{light_name}.color")
            if _enabled(color_cfg):
                gain_amp = _amplitude3(
                    color_cfg.get("rgb_gain_amplitude", (0.0, 0.0, 0.0)),
                    name=f"{light_name}.color.rgb_gain_amplitude",
                )
                gains = tuple(
                    1.0 if amp == 0.0 else float(rng.uniform(1.0 - amp, 1.0 + amp))
                    for amp in gain_amp
                )
                diffuse = _clamp_rgb(
                    tuple(channel * gain for channel, gain in zip(_BASE_LIGHT_RGB, gains))
                )
            else:
                diffuse = _BASE_LIGHT_RGB

            position_cfg = _mapping(cfg["position"], name=f"{light_name}.position")
            position_delta = (
                _sample_delta3(
                    rng,
                    _amplitude3(
                        position_cfg.get("xyz_amplitude_m", (0.0, 0.0, 0.0)),
                        name=f"{light_name}.position.xyz_amplitude_m",
                    ),
                )
                if _enabled(position_cfg)
                else (0.0, 0.0, 0.0)
            )
            position = tuple(
                float(base) + float(delta)
                for base, delta in zip(base_position, position_delta)
            )

            shadows_cfg = _mapping(cfg["shadows"], name=f"{light_name}.shadows")
            if _enabled(shadows_cfg):
                probability_on = _finite_float(
                    shadows_cfg.get("probability_on", 0.0),
                    name=f"{light_name}.shadows.probability_on",
                )
                cast_shadows = bool(rng.random() < probability_on)
            else:
                cast_shadows = False

            lights_values[light_name] = {
                "intensity_enabled": _enabled(intensity_cfg),
                "intensity_scale": intensity_scale,
                "color_enabled": _enabled(color_cfg),
                "diffuse_rgb": diffuse,
                "position_enabled": _enabled(position_cfg),
                "position_xyz_m": position,
                "shadows_enabled": _enabled(shadows_cfg),
                "cast_shadows": cast_shadows,
            }

        cameras_values: dict[str, Any] = {}
        for camera_name in ("top_camera", "gripper_camera"):
            camera_cfg = _mapping(payload[camera_name], name=camera_name)
            pose_cfg = _mapping(camera_cfg["pose"], name=f"{camera_name}.pose")
            if _enabled(pose_cfg):
                position_delta = _sample_delta3(
                    rng,
                    _amplitude3(
                        pose_cfg.get("position_amplitude_m", (0.0, 0.0, 0.0)),
                        name=f"{camera_name}.pose.position_amplitude_m",
                    ),
                )
                rotation_delta_deg = _sample_delta3(
                    rng,
                    _amplitude3(
                        pose_cfg.get("rotation_amplitude_deg", (0.0, 0.0, 0.0)),
                        name=f"{camera_name}.pose.rotation_amplitude_deg",
                    ),
                )
            else:
                position_delta = (0.0, 0.0, 0.0)
                rotation_delta_deg = (0.0, 0.0, 0.0)

            image_cfg = _mapping(camera_cfg["image"], name=f"{camera_name}.image")
            image_values = self._sample_image_parameters(rng, image_cfg, camera_name=camera_name)

            cameras_values[camera_name] = {
                "pose": {
                    "enabled": _enabled(pose_cfg),
                    "position_delta_m": position_delta,
                    "rotation_delta_deg": rotation_delta_deg,
                },
                "image": image_values,
            }

        joint_cfg = _mapping(payload["joint_state"], name="joint_state")
        position_noise_cfg = _mapping(joint_cfg["position_noise"], name="joint_state.position_noise")
        velocity_noise_cfg = _mapping(joint_cfg["velocity_noise"], name="joint_state.velocity_noise")
        joint_values = {
            "position_noise": {
                "enabled": _enabled(position_noise_cfg),
                "std_deg": (
                    _sample_range(
                        rng,
                        _range2(
                            position_noise_cfg.get("std_deg_range", (0.0, 0.0)),
                            name="joint_state.position_noise.std_deg_range",
                        ),
                    )
                    if _enabled(position_noise_cfg)
                    else 0.0
                ),
            },
            "velocity_noise": {
                "enabled": _enabled(velocity_noise_cfg),
                "std_rad_s": (
                    _sample_range(
                        rng,
                        _range2(
                            velocity_noise_cfg.get("std_rad_s_range", (0.0, 0.0)),
                            name="joint_state.velocity_noise.std_rad_s_range",
                        ),
                    )
                    if _enabled(velocity_noise_cfg)
                    else 0.0
                ),
            },
        }

        home_cfg = _mapping(payload["home_initial_pose"], name="home_initial_pose")
        if _enabled(home_cfg):
            max_delta_deg = _finite_float(
                home_cfg.get("toward_zero_max_deg", 0.0),
                name="home_initial_pose.toward_zero_max_deg",
            )
            max_home_rad = float(np.max(np.abs(home)))
            if max_home_rad <= 1e-12 or max_delta_deg <= 0.0:
                max_fraction = 0.0
            else:
                max_fraction = min(1.0, math.radians(max_delta_deg) / max_home_rad)
            toward_zero_fraction = (
                0.0 if max_fraction == 0.0 else float(rng.uniform(0.0, max_fraction))
            )
        else:
            max_delta_deg = 0.0
            toward_zero_fraction = 0.0

        home_values = {
            "enabled": _enabled(home_cfg),
            "toward_zero_max_deg": max_delta_deg,
            "toward_zero_fraction": toward_zero_fraction,
        }

        goal_cfg = _mapping(payload["goal_region"], name="goal_region")
        position_cfg = _mapping(goal_cfg["position"], name="goal_region.position")
        color_cfg = _mapping(goal_cfg["color"], name="goal_region.color")
        size_cfg = _mapping(goal_cfg["size"], name="goal_region.size")

        if _enabled(position_cfg):
            xy_amp = position_cfg.get("xy_amplitude_m", (0.0, 0.0))
            if not isinstance(xy_amp, Sequence) or isinstance(xy_amp, (str, bytes)) or len(xy_amp) != 2:
                raise EnvironmentRandomizationError(
                    "goal_region.position.xy_amplitude_m must contain 2 values."
                )
            x_amp = _finite_float(xy_amp[0], name="goal_region.position.xy_amplitude_m[0]")
            y_amp = _finite_float(xy_amp[1], name="goal_region.position.xy_amplitude_m[1]")
            if x_amp < 0.0 or y_amp < 0.0:
                raise EnvironmentRandomizationError(
                    "goal-region position amplitudes must be non-negative."
                )
            delta_xy = (
                0.0 if x_amp == 0.0 else float(rng.uniform(-x_amp, x_amp)),
                0.0 if y_amp == 0.0 else float(rng.uniform(-y_amp, y_amp)),
            )
        else:
            delta_xy = (0.0, 0.0)

        center = (
            float(goal_center[0] + delta_xy[0]),
            float(goal_center[1] + delta_xy[1]),
            float(goal_center[2]),
        )

        if _enabled(color_cfg):
            goal_rgb = _sample_rgb_delta(
                rng,
                _BASE_GOAL_RGB,
                _amplitude3(
                    color_cfg.get("rgb_delta", (0.0, 0.0, 0.0)),
                    name="goal_region.color.rgb_delta",
                ),
            )
        else:
            goal_rgb = _BASE_GOAL_RGB

        if _enabled(size_cfg):
            half_x = _sample_range(
                rng,
                _range2(
                    size_cfg.get(
                        "half_extent_x_range_m",
                        (goal_half_extent_x_m, goal_half_extent_x_m),
                    ),
                    name="goal_region.size.half_extent_x_range_m",
                ),
            )
            half_y = _sample_range(
                rng,
                _range2(
                    size_cfg.get(
                        "half_extent_y_range_m",
                        (goal_half_extent_y_m, goal_half_extent_y_m),
                    ),
                    name="goal_region.size.half_extent_y_range_m",
                ),
            )
            if half_x <= 0.0 or half_y <= 0.0:
                raise EnvironmentRandomizationError(
                    "goal-region half extents must remain positive."
                )
        else:
            half_x = float(goal_half_extent_x_m)
            half_y = float(goal_half_extent_y_m)

        goal_values = {
            "position": {
                "enabled": _enabled(position_cfg),
                "center_world_m": center,
                "delta_xy_m": delta_xy,
            },
            "color": {
                "enabled": _enabled(color_cfg),
                "rgb": goal_rgb,
            },
            "size": {
                "enabled": _enabled(size_cfg),
                "half_extent_x_m": half_x,
                "half_extent_y_m": half_y,
            },
        }

        sample = EpisodeRandomizationSample(
            base_seed=self.base_seed,
            world_index=self.world_index,
            episode_key=episode_key,
            derived_seed=derived_seed,
            config_path=str(self.config.source_path),
            scene=scene_values,
            lights=lights_values,
            cameras=cameras_values,
            joint_state=joint_values,
            home_initial_pose=home_values,
            goal_region=goal_values,
        )
        self.current_sample = sample
        self._frame_rng = np.random.default_rng(derived_seed ^ 0x9E3779B97F4A7C15)
        self._geometry_cache.clear()
        return sample

    @staticmethod
    def _sample_image_parameters(
        rng: np.random.Generator,
        image_cfg: Mapping[str, Any],
        *,
        camera_name: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        scalar_features = {
            "brightness": ("gain_range", 1.0),
            "contrast": ("gain_range", 1.0),
            "gamma": ("gamma_range", 1.0),
            "saturation": ("gain_range", 1.0),
            "blur": ("sigma_px_range", 0.0),
        }
        for feature_name, (range_name, disabled_value) in scalar_features.items():
            cfg = _mapping(image_cfg[feature_name], name=f"{camera_name}.image.{feature_name}")
            enabled = _enabled(cfg)
            value = (
                _sample_range(
                    rng,
                    _range2(
                        cfg.get(range_name, (disabled_value, disabled_value)),
                        name=f"{camera_name}.image.{feature_name}.{range_name}",
                    ),
                )
                if enabled
                else disabled_value
            )
            if feature_name == "blur" and value < 0.0:
                raise EnvironmentRandomizationError("blur sigma must be non-negative.")
            if feature_name in ("brightness", "contrast", "gamma", "saturation") and value <= 0.0:
                raise EnvironmentRandomizationError(
                    f"{camera_name} {feature_name} sampled a non-positive value."
                )
            result[feature_name] = {"enabled": enabled, "value": value}

        wb_cfg = _mapping(
            image_cfg["white_balance_rgb_gain"],
            name=f"{camera_name}.image.white_balance_rgb_gain",
        )
        wb_enabled = _enabled(wb_cfg)
        channel_ranges = wb_cfg.get(
            "channel_gain_ranges",
            ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        )
        if not isinstance(channel_ranges, Sequence) or len(channel_ranges) != 3:
            raise EnvironmentRandomizationError(
                f"{camera_name}.image.white_balance_rgb_gain.channel_gain_ranges "
                "must contain three [min, max] ranges."
            )
        wb_gains = tuple(
            _sample_range(
                rng,
                _range2(channel_ranges[index], name=f"{camera_name}.white_balance.channel_{index}"),
            )
            if wb_enabled
            else 1.0
            for index in range(3)
        )
        if any(value <= 0.0 for value in wb_gains):
            raise EnvironmentRandomizationError("white-balance gains must be positive.")
        result["white_balance_rgb_gain"] = {
            "enabled": wb_enabled,
            "rgb_gains": wb_gains,
        }

        noise_cfg = _mapping(image_cfg["rgb_noise"], name=f"{camera_name}.image.rgb_noise")
        noise_enabled = _enabled(noise_cfg)
        noise_std = (
            _sample_range(
                rng,
                _range2(
                    noise_cfg.get("std_8bit_range", (0.0, 0.0)),
                    name=f"{camera_name}.image.rgb_noise.std_8bit_range",
                ),
            )
            if noise_enabled
            else 0.0
        )
        if noise_std < 0.0:
            raise EnvironmentRandomizationError("RGB-noise std must be non-negative.")
        result["rgb_noise"] = {
            "enabled": noise_enabled,
            "std_8bit": noise_std,
        }

        translation_cfg = _mapping(
            image_cfg["translation"],
            name=f"{camera_name}.image.translation",
        )
        translation_enabled = _enabled(translation_cfg)
        x_amp = _finite_float(
            translation_cfg.get("x_amplitude_px", 0.0),
            name=f"{camera_name}.image.translation.x_amplitude_px",
        )
        y_amp = _finite_float(
            translation_cfg.get("y_amplitude_px", 0.0),
            name=f"{camera_name}.image.translation.y_amplitude_px",
        )
        if x_amp < 0.0 or y_amp < 0.0:
            raise EnvironmentRandomizationError("image translation amplitudes must be non-negative.")
        result["translation"] = {
            "enabled": translation_enabled,
            "x_px": (
                float(rng.uniform(-x_amp, x_amp))
                if translation_enabled and x_amp > 0.0
                else 0.0
            ),
            "y_px": (
                float(rng.uniform(-y_amp, y_amp))
                if translation_enabled and y_amp > 0.0
                else 0.0
            ),
        }

        distortion_cfg = _mapping(
            image_cfg["distortion"],
            name=f"{camera_name}.image.distortion",
        )
        distortion_enabled = _enabled(distortion_cfg)
        result["distortion"] = {
            "enabled": distortion_enabled,
            "delta_k1": (
                _sample_range(
                    rng,
                    _range2(
                        distortion_cfg.get("delta_k1_range", (0.0, 0.0)),
                        name=f"{camera_name}.image.distortion.delta_k1_range",
                    ),
                )
                if distortion_enabled
                else 0.0
            ),
            "delta_k2": (
                _sample_range(
                    rng,
                    _range2(
                        distortion_cfg.get("delta_k2_range", (0.0, 0.0)),
                        name=f"{camera_name}.image.distortion.delta_k2_range",
                    ),
                )
                if distortion_enabled
                else 0.0
            ),
        }

        crop_cfg = _mapping(image_cfg["crop"], name=f"{camera_name}.image.crop")
        crop_enabled = _enabled(crop_cfg)
        crop_scale = (
            _sample_range(
                rng,
                _range2(
                    crop_cfg.get("scale_range", (1.0, 1.0)),
                    name=f"{camera_name}.image.crop.scale_range",
                ),
            )
            if crop_enabled
            else 1.0
        )
        if not 0.5 <= crop_scale <= 1.0:
            raise EnvironmentRandomizationError(
                f"{camera_name} crop scale must remain within [0.5, 1.0]."
            )
        cx_amp = _finite_float(
            crop_cfg.get("center_x_amplitude_px", 0.0),
            name=f"{camera_name}.image.crop.center_x_amplitude_px",
        )
        cy_amp = _finite_float(
            crop_cfg.get("center_y_amplitude_px", 0.0),
            name=f"{camera_name}.image.crop.center_y_amplitude_px",
        )
        if cx_amp < 0.0 or cy_amp < 0.0:
            raise EnvironmentRandomizationError("crop center amplitudes must be non-negative.")
        result["crop"] = {
            "enabled": crop_enabled,
            "scale": crop_scale,
            "center_x_offset_px": (
                float(rng.uniform(-cx_amp, cx_amp))
                if crop_enabled and cx_amp > 0.0
                else 0.0
            ),
            "center_y_offset_px": (
                float(rng.uniform(-cy_amp, cy_amp))
                if crop_enabled and cy_amp > 0.0
                else 0.0
            ),
        }

        return result

    def randomized_home_joint_positions(
        self,
        canonical_home_joint_positions_rad: Sequence[float],
    ) -> np.ndarray:
        sample = self._require_sample()
        home = np.asarray(canonical_home_joint_positions_rad, dtype=float)
        fraction = float(sample.home_initial_pose["toward_zero_fraction"])
        result = home * (1.0 - fraction)
        return result.astype(float, copy=True)

    def goal_center_world_m(self) -> tuple[float, float, float]:
        sample = self._require_sample()
        return tuple(float(value) for value in sample.goal_region["position"]["center_world_m"])  # type: ignore[return-value]

    def goal_half_extents_m(self) -> tuple[float, float]:
        sample = self._require_sample()
        size = sample.goal_region["size"]
        return (
            float(size["half_extent_x_m"]),
            float(size["half_extent_y_m"]),
        )

    def record(self) -> dict[str, Any]:
        return self._require_sample().to_json_dict()

    def _require_sample(self) -> EpisodeRandomizationSample:
        if self.current_sample is None:
            raise EnvironmentRandomizationError(
                "begin_episode() must be called before using randomization."
            )
        return self.current_sample

    def apply_observation(self, observation: PolicyObservation) -> PolicyObservation:
        """Apply policy-only image and joint measurement randomization."""

        sample = self._require_sample()
        if self._frame_rng is None:
            raise EnvironmentRandomizationError("Frame RNG is unavailable.")

        positions = np.asarray(observation.joint_positions, dtype=float).copy()
        velocities = np.asarray(observation.joint_velocities, dtype=float).copy()

        position_noise = sample.joint_state["position_noise"]
        if bool(position_noise["enabled"]) and float(position_noise["std_deg"]) > 0.0:
            positions += self._frame_rng.normal(
                0.0,
                math.radians(float(position_noise["std_deg"])),
                size=positions.shape,
            )

        velocity_noise = sample.joint_state["velocity_noise"]
        if bool(velocity_noise["enabled"]) and float(velocity_noise["std_rad_s"]) > 0.0:
            velocities += self._frame_rng.normal(
                0.0,
                float(velocity_noise["std_rad_s"]),
                size=velocities.shape,
            )

        top = self._apply_image(
            np.asarray(observation.top_image),
            sample.cameras["top_camera"]["image"],
            camera_name="top_camera",
        )
        gripper = self._apply_image(
            np.asarray(observation.gripper_image),
            sample.cameras["gripper_camera"]["image"],
            camera_name="gripper_camera",
        )

        positions.setflags(write=False)
        velocities.setflags(write=False)
        top.setflags(write=False)
        gripper.setflags(write=False)

        return PolicyObservation(
            joint_positions=positions,
            joint_velocities=velocities,
            top_image=top,
            gripper_image=gripper,
        )

    @staticmethod
    def _image_transform_needed(parameters: Mapping[str, Any]) -> bool:
        return any(
            bool(parameters[name]["enabled"])
            for name in (
                "brightness",
                "contrast",
                "gamma",
                "saturation",
                "white_balance_rgb_gain",
                "rgb_noise",
                "blur",
                "translation",
                "distortion",
                "crop",
            )
        )

    def _apply_image(
        self,
        image: np.ndarray,
        parameters: Mapping[str, Any],
        *,
        camera_name: str,
    ) -> np.ndarray:
        source = np.asarray(image)
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise EnvironmentRandomizationError(
                f"{camera_name} image must be uint8 HxWx3."
            )

        if not self._image_transform_needed(parameters):
            return source.copy()

        cv2 = None
        if any(
            bool(parameters[name]["enabled"])
            for name in ("crop", "distortion", "translation", "blur")
        ):
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise EnvironmentRandomizationError(
                    "OpenCV is required for geometric / blur image randomization."
                ) from error
            cv2 = cv2_module

        working = source.copy()
        height, width = working.shape[:2]

        crop = parameters["crop"]
        if bool(crop["enabled"]):
            scale = float(crop["scale"])
            crop_w = max(2, min(width, int(round(width * scale))))
            crop_h = max(2, min(height, int(round(height * scale))))
            center_x = (width - 1) * 0.5 + float(crop["center_x_offset_px"])
            center_y = (height - 1) * 0.5 + float(crop["center_y_offset_px"])
            x0 = int(round(center_x - (crop_w - 1) * 0.5))
            y0 = int(round(center_y - (crop_h - 1) * 0.5))
            x0 = min(max(0, x0), width - crop_w)
            y0 = min(max(0, y0), height - crop_h)
            assert cv2 is not None
            working = cv2.resize(
                working[y0:y0 + crop_h, x0:x0 + crop_w],
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        distortion = parameters["distortion"]
        if bool(distortion["enabled"]) and (
            abs(float(distortion["delta_k1"])) > 0.0
            or abs(float(distortion["delta_k2"])) > 0.0
        ):
            cache_key = (
                camera_name,
                (height, width),
                float(distortion["delta_k1"]),
                float(distortion["delta_k2"]),
            )
            maps = self._geometry_cache.get(cache_key)
            if maps is None:
                yy, xx = np.indices((height, width), dtype=np.float32)
                cx = (width - 1) * 0.5
                cy = (height - 1) * 0.5
                scale_xy = float(max(width, height))
                nx = (xx - cx) / scale_xy
                ny = (yy - cy) / scale_xy
                r2 = nx * nx + ny * ny
                factor = (
                    1.0
                    + float(distortion["delta_k1"]) * r2
                    + float(distortion["delta_k2"]) * r2 * r2
                )
                map_x = (cx + nx * factor * scale_xy).astype(np.float32)
                map_y = (cy + ny * factor * scale_xy).astype(np.float32)
                maps = (map_x, map_y)
                self._geometry_cache[cache_key] = maps
            assert cv2 is not None
            working = cv2.remap(
                working,
                maps[0],
                maps[1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        translation = parameters["translation"]
        if bool(translation["enabled"]) and (
            abs(float(translation["x_px"])) > 0.0
            or abs(float(translation["y_px"])) > 0.0
        ):
            matrix = np.asarray(
                [
                    [1.0, 0.0, float(translation["x_px"])],
                    [0.0, 1.0, float(translation["y_px"])],
                ],
                dtype=np.float32,
            )
            assert cv2 is not None
            working = cv2.warpAffine(
                working,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        blur = parameters["blur"]
        if bool(blur["enabled"]) and float(blur["value"]) > 1e-9:
            assert cv2 is not None
            working = cv2.GaussianBlur(
                working,
                (0, 0),
                sigmaX=float(blur["value"]),
                sigmaY=float(blur["value"]),
                borderType=cv2.BORDER_REFLECT_101,
            )

        pixels = working.astype(np.float32) / 255.0

        contrast = parameters["contrast"]
        if bool(contrast["enabled"]):
            pixels = (pixels - 0.5) * float(contrast["value"]) + 0.5

        saturation = parameters["saturation"]
        if bool(saturation["enabled"]):
            gray = (
                0.299 * pixels[..., 0:1]
                + 0.587 * pixels[..., 1:2]
                + 0.114 * pixels[..., 2:3]
            )
            pixels = gray + float(saturation["value"]) * (pixels - gray)

        white_balance = parameters["white_balance_rgb_gain"]
        if bool(white_balance["enabled"]):
            gains = np.asarray(white_balance["rgb_gains"], dtype=np.float32).reshape(1, 1, 3)
            pixels = pixels * gains

        gamma = parameters["gamma"]
        if bool(gamma["enabled"]):
            pixels = np.power(
                np.clip(pixels, 0.0, 1.0),
                float(gamma["value"]),
            )

        brightness = parameters["brightness"]
        if bool(brightness["enabled"]):
            pixels = pixels * float(brightness["value"])

        noise = parameters["rgb_noise"]
        if bool(noise["enabled"]) and float(noise["std_8bit"]) > 0.0:
            pixels += self._frame_rng.normal(
                0.0,
                float(noise["std_8bit"]) / 255.0,
                size=pixels.shape,
            ).astype(np.float32)

        return np.clip(np.rint(pixels * 255.0), 0.0, 255.0).astype(np.uint8)


class EpisodeImageAugmenter:
    """Independent image-only pipeline with no scene or joint randomization.

    The existing JSON schema supplies the two ``camera.image`` sections.
    Every other section is ignored, even if it contains enabled features.
    Only the existing pixel-transform implementation is reused; the full
    environment sampler and its physical mutation hooks are never called.
    """

    def __init__(self, config: EnvironmentRandomizationConfig, *,
                 base_seed: int = 0, world_index: int = 0) -> None:
        self._transformer = EpisodeEnvironmentRandomizer(
            config, base_seed=base_seed, world_index=world_index,
        )
        self.config, self.base_seed, self.world_index = config, base_seed, world_index
        self._record: dict[str, Any] | None = None

    def begin_episode(self, *, episode_key: int) -> dict[str, Any]:
        if isinstance(episode_key, bool) or not isinstance(episode_key, int) or episode_key < 0:
            raise ValueError("episode_key must be a non-negative integer.")
        derived_seed = _derive_seed(self.base_seed, self.world_index, episode_key)
        rng = np.random.default_rng(derived_seed)
        cameras = {
            name: {"image": EpisodeEnvironmentRandomizer._sample_image_parameters(
                rng, self.config.payload[name]["image"], camera_name=name,
            )}
            for name in ("top_camera", "gripper_camera")
        }
        self._transformer._frame_rng = np.random.default_rng(derived_seed ^ 0x9E3779B97F4A7C15)
        self._transformer._geometry_cache.clear()
        self._record = {
            "enabled": True, "mode": "image_only", "base_seed": self.base_seed,
            "world_index": self.world_index, "episode_key": episode_key,
            "derived_seed": derived_seed, "config_path": str(self.config.source_path),
            "cameras": cameras,
        }
        return self.record()

    def record(self) -> dict[str, Any]:
        if self._record is None:
            raise EnvironmentRandomizationError("begin_episode() must precede image augmentation.")
        return _jsonable(self._record)

    def apply_observation(self, observation: PolicyObservation) -> PolicyObservation:
        record = self.record()
        top, gripper = (
            self._transformer._apply_image(
                np.asarray(pixels), record["cameras"][name]["image"], camera_name=name,
            )
            for name, pixels in (("top_camera", observation.top_image),
                                 ("gripper_camera", observation.gripper_image))
        )
        positions = np.asarray(observation.joint_positions).copy()
        velocities = np.asarray(observation.joint_velocities).copy()
        for value in (positions, velocities, top, gripper):
            value.setflags(write=False)
        return PolicyObservation(joint_positions=positions, joint_velocities=velocities,
                                 top_image=top, gripper_image=gripper)
