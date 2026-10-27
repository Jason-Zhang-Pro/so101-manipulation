"""Delayed random-target lifecycle for multi-episode policy runs.

An episode starts before a target exists. A random target is requested after
an exact number of completed policy steps. Target appearance is confirmed by
the first non-empty /target_cube/states update after the spawn request.

No /clock timestamp is required. The persisted record keeps the request step,
service result, confirmation flag, confirmation step, target name, and initial
position.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TriggerResult:
    """Result from one Trigger service request."""

    success: bool
    message: str


@dataclass(frozen=True)
class TargetSnapshot:
    """Minimal target information recovered from the states topic."""

    name: str | None
    position: tuple[float, float, float] | None


@dataclass
class DelayedTargetEpisodeRecord:
    """Persisted target lifecycle record for one episode."""

    episode_index: int
    policy_hz: float
    target_delay_s: float
    target_delay_steps: int
    target_spawn_request_step: int | None = None
    target_spawn_service_success: bool | None = None
    target_spawn_service_message: str | None = None
    target_spawn_confirmed: bool = False
    target_spawn_confirmed_step: int | None = None
    target_name: str | None = None
    target_initial_position: tuple[float, float, float] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)

        if self.target_initial_position is not None:
            result["target_initial_position"] = list(
                self.target_initial_position
            )

        return result


class DelayedRandomTargetEpisodeController:
    """Delete old targets, then spawn one after a fixed episode delay."""

    def __init__(
        self,
        backend: Any,
        *,
        target_delay_s: float,
        policy_hz: float = 30.0,
        output_directory: str | Path | None = None,
    ) -> None:
        if not math.isfinite(target_delay_s) or target_delay_s < 0.0:
            raise ValueError(
                "target_delay_s must be finite and non-negative."
            )

        if not math.isfinite(policy_hz) or policy_hz <= 0.0:
            raise ValueError(
                "policy_hz must be finite and positive."
            )

        delay_steps_float = target_delay_s * policy_hz
        delay_steps = int(round(delay_steps_float))

        if not math.isclose(
            delay_steps_float,
            delay_steps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "target_delay_s must map to an integer number of "
                "policy steps."
            )

        self._backend = backend
        self.target_delay_s = float(target_delay_s)
        self.policy_hz = float(policy_hz)
        self.target_delay_steps = delay_steps
        self.output_directory = (
            None
            if output_directory is None
            else Path(output_directory).expanduser().resolve()
        )

        self._record: DelayedTargetEpisodeRecord | None = None
        self._manifest_paths: list[Path] = []
        self._current_step = 0

    @property
    def manifest_paths(self) -> tuple[Path, ...]:
        return tuple(self._manifest_paths)

    @property
    def current_record(self) -> DelayedTargetEpisodeRecord | None:
        return self._record

    def clear_before_episode(self, episode_index: int) -> None:
        del episode_index

        result = self._backend.delete_all()

        if not result.success:
            raise RuntimeError(
                "Failed to delete target cubes before episode: "
                f"{result.message}"
            )

        # A successful Trigger response is the authoritative
        # deletion confirmation. Do not wait for the
        # simulation-time target-state publisher while Gazebo
        # is paused between episodes.

    def start_episode(self, episode_index: int) -> None:
        self._backend.poll(timeout_s=0.0)

        self._current_step = 0
        self._record = DelayedTargetEpisodeRecord(
            episode_index=episode_index,
            policy_hz=self.policy_hz,
            target_delay_s=self.target_delay_s,
            target_delay_steps=self.target_delay_steps,
        )
        self._persist_record()

        if self.target_delay_steps == 0:
            self._request_target(step_count=0)
            self._backend.poll(timeout_s=0.0)
            self._confirm_target_if_available()

    def after_step(self, completed_step_count: int) -> None:
        if self._record is None:
            raise RuntimeError(
                "Delayed target episode has not been started."
            )

        if completed_step_count <= self._current_step:
            raise ValueError(
                "completed_step_count must increase monotonically."
            )

        self._current_step = completed_step_count
        self._backend.poll(timeout_s=0.0)

        if (
            completed_step_count >= self.target_delay_steps
            and self._record.target_spawn_request_step is None
        ):
            self._request_target(
                step_count=completed_step_count
            )

        confirmation_poll_timeout_s = (
            0.05
            if (
                self._record.target_spawn_request_step
                is not None
                and not self._record.target_spawn_confirmed
            )
            else 0.0
        )
        self._backend.poll(
            timeout_s=confirmation_poll_timeout_s
        )
        self._confirm_target_if_available()

    def _request_target(self, *, step_count: int) -> None:
        if self._record is None:
            raise RuntimeError(
                "Delayed target episode has not been started."
            )

        self._record.target_spawn_request_step = step_count
        self._persist_record()

        result = self._backend.spawn_random()
        self._record.target_spawn_service_success = result.success
        self._record.target_spawn_service_message = result.message

        if not result.success:
            self._persist_record()
            raise RuntimeError(
                "Failed to spawn a random target: "
                f"{result.message}"
            )

        service_target = _parse_spawn_service_target(
            result.message
        )
        self._record.target_spawn_confirmed = True
        self._record.target_spawn_confirmed_step = step_count

        if service_target is not None:
            self._record.target_name = service_target.name
            self._record.target_initial_position = (
                service_target.position
            )

        self._persist_record()

    def _confirm_target_if_available(self) -> None:
        if self._record is None:
            return

        if self._record.target_spawn_request_step is None:
            return

        if self._record.target_spawn_service_success is not True:
            return

        if self._record.target_spawn_confirmed:
            return

        targets = self._backend.current_targets

        if not targets:
            return

        target = targets[0]
        self._record.target_spawn_confirmed = True
        self._record.target_spawn_confirmed_step = (
            self._current_step
        )
        self._record.target_name = target.name
        self._record.target_initial_position = target.position
        self._persist_record()

    def _persist_record(self) -> None:
        if self._record is None or self.output_directory is None:
            return

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        path = (
            self.output_directory
            / (
                f"episode_{self._record.episode_index:06d}"
                "_target.json"
            )
        )
        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(
                self._record.to_json_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)

        if path not in self._manifest_paths:
            self._manifest_paths.append(path)


class DelayedRandomTargetRuntimeProxy:
    """Intercept episode boundaries and completed policy steps."""

    def __init__(
        self,
        runtime: Any,
        controller: DelayedRandomTargetEpisodeController,
    ) -> None:
        self._runtime = runtime
        self._controller = controller
        self._episode_index = -1
        self._completed_step_count = 0
        self._randomized_target_name: str | None = None

    def _randomize_spawned_target_if_ready(self) -> None:
        record = getattr(
            self._controller,
            "current_record",
            None,
        )
        if record is None or record.target_name is None:
            return
        if record.target_name == self._randomized_target_name:
            return
        callback = getattr(self._runtime, "randomize_spawned_target", None)
        if callback is not None:
            callback(record.target_name)
        self._randomized_target_name = record.target_name

    def prepare_episode(self) -> Any:
        self._episode_index += 1
        self._controller.clear_before_episode(
            self._episode_index
        )
        self._randomized_target_name = None
        frame = self._runtime.prepare_episode()
        self._completed_step_count = 0
        self._controller.start_episode(
            self._episode_index
        )
        self._randomize_spawned_target_if_ready()
        return frame

    def step(self, action: Any) -> Any:
        result = self._runtime.step(action)
        self._completed_step_count += 1
        self._controller.after_step(
            self._completed_step_count
        )
        self._randomize_spawned_target_if_ready()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)


class RosTargetLifecycleBackend:
    """ROS 2 backend for target Trigger services and state confirmation."""

    def __init__(
        self,
        *,
        delete_service: str = "/target_cube/delete_all",
        spawn_service: str = "/target_cube/spawn_random",
        states_topic: str = "/target_cube/states",
        service_timeout_s: float = 5.0,
        state_timeout_s: float = 5.0,
    ) -> None:
        self.delete_service = delete_service
        self.spawn_service = spawn_service
        self.states_topic = states_topic
        self.service_timeout_s = float(service_timeout_s)
        self.state_timeout_s = float(state_timeout_s)

        self._rclpy: Any = None
        self._node: Any = None
        self._trigger_type: Any = None
        self._delete_client: Any = None
        self._spawn_client: Any = None
        self._current_targets: tuple[TargetSnapshot, ...] | None = None

    def __enter__(self) -> "RosTargetLifecycleBackend":
        import rclpy
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        if not rclpy.ok():
            raise RuntimeError(
                "rclpy is not initialized. Create the simulation "
                "runtime before the target backend."
            )

        self._rclpy = rclpy
        self._trigger_type = Trigger
        self._node = rclpy.create_node(
            f"so101_delayed_target_{os.getpid()}"
        )
        self._delete_client = self._node.create_client(
            Trigger,
            self.delete_service,
        )
        self._spawn_client = self._node.create_client(
            Trigger,
            self.spawn_service,
        )
        self._node.create_subscription(
            String,
            self.states_topic,
            self._states_callback,
            10,
        )

        for name, client in (
            (self.delete_service, self._delete_client),
            (self.spawn_service, self._spawn_client),
        ):
            if not client.wait_for_service(
                timeout_sec=self.service_timeout_s
            ):
                raise RuntimeError(
                    f"Target service is unavailable: {name}"
                )

        self.poll(timeout_s=0.0)
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception: Any,
        traceback: Any,
    ) -> bool:
        del exception_type, exception, traceback

        if self._node is not None:
            self._node.destroy_node()

        self._node = None
        return False

    @property
    def current_targets(self) -> tuple[TargetSnapshot, ...]:
        return self._current_targets or ()

    def poll(self, *, timeout_s: float = 0.0) -> None:
        if self._node is None:
            raise RuntimeError(
                "Target backend is not active."
            )

        self._rclpy.spin_once(
            self._node,
            timeout_sec=timeout_s,
        )

    def delete_all(self) -> TriggerResult:
        return self._call_trigger(self._delete_client)

    def spawn_random(self) -> TriggerResult:
        return self._call_trigger(self._spawn_client)

    def wait_until_empty(self) -> None:
        deadline = time.monotonic() + self.state_timeout_s

        while time.monotonic() < deadline:
            self.poll(timeout_s=0.05)

            if (
                self._current_targets is not None
                and not self._current_targets
            ):
                return

        raise TimeoutError(
            "Timed out waiting for an empty target collection."
        )

    def _call_trigger(self, client: Any) -> TriggerResult:
        request = self._trigger_type.Request()
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(
            self._node,
            future,
            timeout_sec=self.service_timeout_s,
        )

        if not future.done():
            raise TimeoutError(
                "Timed out waiting for target service response."
            )

        response = future.result()

        if response is None:
            raise RuntimeError(
                "Target service returned no response."
            )

        return TriggerResult(
            success=bool(response.success),
            message=str(response.message),
        )

    def _states_callback(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return

        self._current_targets = _parse_target_snapshots(
            payload
        )


def _parse_spawn_service_target(
    message: str,
) -> TargetSnapshot | None:
    """Parse the target manager's successful Trigger message."""

    try:
        payload = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, Mapping):
        return None

    position = _coerce_xyz(
        payload.get("requested_position")
    )

    if position is None:
        position = _extract_position(payload)

    return TargetSnapshot(
        name=_extract_name(
            payload,
            fallback=None,
        ),
        position=position,
    )


def _parse_target_snapshots(
    payload: Any,
) -> tuple[TargetSnapshot, ...]:
    entries: list[tuple[str | None, Any]] = []

    if isinstance(payload, list):
        entries = [(None, item) for item in payload]
    elif isinstance(payload, Mapping):
        for key in (
            "targets",
            "cubes",
            "states",
            "items",
            "models",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                entries = [
                    (None, item)
                    for item in value
                ]
                break

            if isinstance(value, Mapping):
                entries = [
                    (str(name), item)
                    for name, item in value.items()
                ]
                break
        else:
            if _looks_like_target_entry(payload):
                entries = [(None, payload)]
            elif all(
                isinstance(value, Mapping)
                for value in payload.values()
            ):
                entries = [
                    (str(name), item)
                    for name, item in payload.items()
                ]

    snapshots = []

    for fallback_name, entry in entries:
        if not isinstance(entry, Mapping):
            continue

        snapshots.append(
            TargetSnapshot(
                name=_extract_name(
                    entry,
                    fallback=fallback_name,
                ),
                position=_extract_position(entry),
            )
        )

    return tuple(snapshots)


def _looks_like_target_entry(
    entry: Mapping[str, Any],
) -> bool:
    return any(
        key in entry
        for key in (
            "name",
            "model_name",
            "entity_name",
            "position",
            "pose",
            "x",
            "y",
            "z",
        )
    )


def _extract_name(
    entry: Mapping[str, Any],
    *,
    fallback: str | None,
) -> str | None:
    for key in (
        "name",
        "model_name",
        "entity_name",
        "target_name",
    ):
        value = entry.get(key)

        if value is not None:
            return str(value)

    return fallback


def _extract_position(
    entry: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    candidates: list[Any] = [
        entry.get("position"),
        entry.get("pose"),
    ]

    pose = entry.get("pose")

    if isinstance(pose, Mapping):
        candidates.insert(0, pose.get("position"))

    candidates.append(entry)

    for candidate in candidates:
        result = _coerce_xyz(candidate)

        if result is not None:
            return result

    return None


def _coerce_xyz(
    value: Any,
) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        if all(key in value for key in ("x", "y", "z")):
            try:
                return (
                    float(value["x"]),
                    float(value["y"]),
                    float(value["z"]),
                )
            except (TypeError, ValueError):
                return None

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 3
    ):
        try:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
            )
        except (TypeError, ValueError):
            return None

    return None
