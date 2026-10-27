"""Thread-safe caches for robot, target-cube, and goal-region state.

The caches store snapshots received from ROS callbacks. They do not subscribe
to ROS topics themselves, send control commands, compute forward kinematics, or
calculate rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from threading import Condition
from time import monotonic
from typing import Sequence

import numpy as np

from so101_capstone.robot import ACTIVE_JOINT_NAMES


MEASURED_JOINT_NAMES = (*ACTIVE_JOINT_NAMES, "gripper")


def _received_time_or_now(
    received_monotonic_s: float | None,
) -> float:
    value = (
        monotonic()
        if received_monotonic_s is None
        else float(received_monotonic_s)
    )

    if not np.isfinite(value):
        raise ValueError("Received time must be finite.")

    return value


def _validate_timeout(timeout_s: float) -> float:
    value = float(timeout_s)

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "Timeout must be finite and non-negative."
        )

    return value


@dataclass(frozen=True)
class JointStateSnapshot:
    """One complete measured joint-state message."""

    positions: dict[str, float]
    velocities: dict[str, float] | None
    sequence: int
    received_monotonic_s: float
    source_sim_time_s: float | None = None

    def age_s(
        self,
        now_monotonic_s: float | None = None,
    ) -> float:
        now = (
            monotonic()
            if now_monotonic_s is None
            else float(now_monotonic_s)
        )

        if not np.isfinite(now):
            raise ValueError("Current time must be finite.")

        return max(0.0, now - self.received_monotonic_s)


@dataclass(frozen=True)
class TargetCubeSnapshot:
    """One measured target-cube pose in the task world frame."""

    name: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    sequence: int
    received_monotonic_s: float
    source_sequence: int | None = None
    source_sim_time_s: float | None = None

    def age_s(
        self,
        now_monotonic_s: float | None = None,
    ) -> float:
        now = (
            monotonic()
            if now_monotonic_s is None
            else float(now_monotonic_s)
        )

        if not np.isfinite(now):
            raise ValueError("Current time must be finite.")

        return max(0.0, now - self.received_monotonic_s)


def _copy_joint_snapshot(
    snapshot: JointStateSnapshot,
) -> JointStateSnapshot:
    return JointStateSnapshot(
        positions=dict(snapshot.positions),
        velocities=(
            None
            if snapshot.velocities is None
            else dict(snapshot.velocities)
        ),
        sequence=snapshot.sequence,
        received_monotonic_s=snapshot.received_monotonic_s,
        source_sim_time_s=snapshot.source_sim_time_s,
    )


def _copy_cube_snapshot(
    snapshot: TargetCubeSnapshot,
) -> TargetCubeSnapshot:
    return TargetCubeSnapshot(
        name=snapshot.name,
        position=snapshot.position.copy(),
        quaternion_xyzw=snapshot.quaternion_xyzw.copy(),
        sequence=snapshot.sequence,
        received_monotonic_s=snapshot.received_monotonic_s,
        source_sequence=snapshot.source_sequence,
        source_sim_time_s=snapshot.source_sim_time_s,
    )


class JointStateCache:
    """Thread-safe cache of the latest complete measured joint state."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._snapshot: JointStateSnapshot | None = None
        self._next_sequence = 1
        self._history: deque[JointStateSnapshot] = deque(maxlen=512)

    def update(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float] | None = None,
        *,
        received_monotonic_s: float | None = None,
        source_sim_time_s: float | None = None,
    ) -> JointStateSnapshot:
        """Validate and store one complete joint-state message."""

        joint_names = tuple(str(name) for name in names)

        if len(set(joint_names)) != len(joint_names):
            raise ValueError(
                "Joint names must not contain duplicates."
            )

        position_array = np.asarray(positions, dtype=float)

        if position_array.shape != (len(joint_names),):
            raise ValueError(
                "Joint name and position arrays must have "
                "the same one-dimensional length."
            )

        if not np.all(np.isfinite(position_array)):
            raise ValueError(
                "Joint positions must contain only finite values."
            )

        positions_by_name = {
            name: float(position)
            for name, position in zip(
                joint_names,
                position_array,
            )
        }

        missing = [
            name
            for name in MEASURED_JOINT_NAMES
            if name not in positions_by_name
        ]

        if missing:
            raise ValueError(
                "Missing required joint positions: "
                + ", ".join(missing)
            )

        ordered_positions = {
            name: positions_by_name[name]
            for name in MEASURED_JOINT_NAMES
        }

        ordered_velocities: dict[str, float] | None = None

        if velocities is not None:
            velocity_array = np.asarray(velocities, dtype=float)

            # ROS JointState velocity may legally be an empty array.
            if velocity_array.size != 0:
                if velocity_array.shape != (len(joint_names),):
                    raise ValueError(
                        "Joint name and velocity arrays must "
                        "have the same one-dimensional length."
                    )

                if not np.all(np.isfinite(velocity_array)):
                    raise ValueError(
                        "Joint velocities must contain only "
                        "finite values."
                    )

                velocities_by_name = {
                    name: float(velocity)
                    for name, velocity in zip(
                        joint_names,
                        velocity_array,
                    )
                }

                ordered_velocities = {
                    name: velocities_by_name[name]
                    for name in MEASURED_JOINT_NAMES
                }

        received_time = _received_time_or_now(
            received_monotonic_s
        )

        if source_sim_time_s is not None:
            source_sim_time_s = float(source_sim_time_s)
            if not np.isfinite(source_sim_time_s) or source_sim_time_s < 0:
                raise ValueError("Joint source simulation time must be finite and non-negative.")

        with self._condition:
            snapshot = JointStateSnapshot(
                positions=ordered_positions,
                velocities=ordered_velocities,
                sequence=self._next_sequence,
                received_monotonic_s=received_time,
                source_sim_time_s=source_sim_time_s,
            )
            if self._history and source_sim_time_s is not None:
                previous_time = self._history[-1].source_sim_time_s
                if previous_time is not None and source_sim_time_s < previous_time:
                    self._history.clear()
            self._history.append(snapshot)
            self._next_sequence += 1
            self._snapshot = snapshot
            self._condition.notify_all()

            return _copy_joint_snapshot(snapshot)

    def history(self) -> tuple[JointStateSnapshot, ...]:
        """Return bounded physical history for interpolation, never extrapolation."""
        with self._condition:
            return tuple(_copy_joint_snapshot(item) for item in self._history)

    def latest(self) -> JointStateSnapshot | None:
        """Return an independent copy of the latest snapshot."""

        with self._condition:
            if self._snapshot is None:
                return None

            return _copy_joint_snapshot(self._snapshot)

    def wait_for_sequence_greater_than(
        self,
        sequence: int,
        timeout_s: float,
    ) -> JointStateSnapshot | None:
        """Wait for measured feedback newer than ``sequence``."""

        if sequence < 0:
            raise ValueError(
                "Sequence must be non-negative."
            )

        timeout = _validate_timeout(timeout_s)
        deadline = monotonic() + timeout

        with self._condition:
            while (
                self._snapshot is None
                or self._snapshot.sequence <= sequence
            ):
                remaining = deadline - monotonic()

                if remaining <= 0.0:
                    return None

                self._condition.wait(timeout=remaining)

            return _copy_joint_snapshot(self._snapshot)


class TargetCubeStateCache:
    """Thread-safe cache of the current episode target-cube pose."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._snapshot: TargetCubeSnapshot | None = None
        self._next_sequence = 1
        self._history: deque[TargetCubeSnapshot] = deque(maxlen=256)

    def update(
        self,
        name: str,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
        *,
        source_sequence: int | None = None,
        source_sim_time_s: float | None = None,
        received_monotonic_s: float | None = None,
    ) -> TargetCubeSnapshot:
        """Validate and store one target-cube pose."""

        cube_name = str(name).strip()

        if not cube_name:
            raise ValueError(
                "Target cube name must not be empty."
            )

        position_array = np.asarray(position, dtype=float)

        if position_array.shape != (3,):
            raise ValueError(
                "Target cube position must have shape (3,)."
            )

        if not np.all(np.isfinite(position_array)):
            raise ValueError(
                "Target cube position must contain only "
                "finite values."
            )

        quaternion = np.asarray(
            quaternion_xyzw,
            dtype=float,
        )

        if quaternion.shape != (4,):
            raise ValueError(
                "Target cube quaternion must have shape (4,)."
            )

        if not np.all(np.isfinite(quaternion)):
            raise ValueError(
                "Target cube quaternion must contain only "
                "finite values."
            )

        quaternion_norm = float(np.linalg.norm(quaternion))

        if quaternion_norm <= 1e-12:
            raise ValueError(
                "Target cube quaternion must have non-zero norm."
            )

        normalized_quaternion = quaternion / quaternion_norm

        if source_sequence is None:
            validated_source_sequence = None
        elif (
            isinstance(source_sequence, bool)
            or not isinstance(source_sequence, int)
            or source_sequence < 0
        ):
            raise ValueError(
                "Source sequence must be a non-negative integer."
            )
        else:
            validated_source_sequence = source_sequence

        if source_sim_time_s is None:
            validated_source_sim_time_s = None
        else:
            if isinstance(source_sim_time_s, bool):
                raise ValueError(
                    "Source simulation time must be finite "
                    "and non-negative."
                )

            validated_source_sim_time_s = float(
                source_sim_time_s
            )

            if (
                not np.isfinite(validated_source_sim_time_s)
                or validated_source_sim_time_s < 0.0
            ):
                raise ValueError(
                    "Source simulation time must be finite "
                    "and non-negative."
                )

        received_time = _received_time_or_now(
            received_monotonic_s
        )

        with self._condition:
            snapshot = TargetCubeSnapshot(
                name=cube_name,
                position=position_array.copy(),
                quaternion_xyzw=normalized_quaternion.copy(),
                sequence=self._next_sequence,
                received_monotonic_s=received_time,
                source_sequence=validated_source_sequence,
                source_sim_time_s=(
                    validated_source_sim_time_s
                ),
            )
            if self._history:
                previous = self._history[-1]
                if previous.name != snapshot.name or (
                    previous.source_sim_time_s is not None
                    and snapshot.source_sim_time_s is not None
                    and snapshot.source_sim_time_s < previous.source_sim_time_s
                ):
                    self._history.clear()
            # A manager timer publication is not a new physical pose.
            if not self._history or (
                snapshot.source_sim_time_s != self._history[-1].source_sim_time_s
            ):
                self._history.append(snapshot)
            self._next_sequence += 1
            self._snapshot = snapshot
            self._condition.notify_all()

            return _copy_cube_snapshot(snapshot)

    def history(self) -> tuple[TargetCubeSnapshot, ...]:
        with self._condition:
            return tuple(_copy_cube_snapshot(item) for item in self._history)

    def latest(self) -> TargetCubeSnapshot | None:
        """Return an independent copy of the latest snapshot."""

        with self._condition:
            if self._snapshot is None:
                return None

            return _copy_cube_snapshot(self._snapshot)

    def wait_for_sequence_greater_than(
        self,
        sequence: int,
        timeout_s: float,
    ) -> TargetCubeSnapshot | None:
        """Wait for a target-cube update newer than ``sequence``."""

        if sequence < 0:
            raise ValueError(
                "Sequence must be non-negative."
            )

        timeout = _validate_timeout(timeout_s)
        deadline = monotonic() + timeout

        with self._condition:
            while (
                self._snapshot is None
                or self._snapshot.sequence <= sequence
            ):
                remaining = deadline - monotonic()

                if remaining <= 0.0:
                    return None

                self._condition.wait(timeout=remaining)

            return _copy_cube_snapshot(self._snapshot)


@dataclass(frozen=True)
class TargetCubeCollectionSnapshot:
    """One complete `/target_cube/states` collection update.

    Unlike ``TargetCubeSnapshot``, this snapshot remains valid when no target
    cube exists. It is used to distinguish a new empty world state from stale
    target-cube feedback left over from an earlier episode.
    """

    cube_names: tuple[str, ...]
    sequence: int
    received_monotonic_s: float
    source_sequence: int | None = None
    source_sim_time_s: float | None = None

    @property
    def cube_count(self) -> int:
        return len(self.cube_names)

    def age_s(
        self,
        now_monotonic_s: float | None = None,
    ) -> float:
        now = (
            monotonic()
            if now_monotonic_s is None
            else float(now_monotonic_s)
        )

        if not np.isfinite(now):
            raise ValueError("Current time must be finite.")

        return max(0.0, now - self.received_monotonic_s)


def _copy_cube_collection_snapshot(
    snapshot: TargetCubeCollectionSnapshot,
) -> TargetCubeCollectionSnapshot:
    return TargetCubeCollectionSnapshot(
        cube_names=tuple(snapshot.cube_names),
        sequence=snapshot.sequence,
        received_monotonic_s=snapshot.received_monotonic_s,
        source_sequence=snapshot.source_sequence,
        source_sim_time_s=snapshot.source_sim_time_s,
    )


class TargetCubeCollectionCache:
    """Thread-safe cache of complete target-cube collection states."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._snapshot: TargetCubeCollectionSnapshot | None = None
        self._next_sequence = 1

    def update(
        self,
        cube_names: Sequence[str],
        *,
        source_sequence: int | None = None,
        source_sim_time_s: float | None = None,
        received_monotonic_s: float | None = None,
    ) -> TargetCubeCollectionSnapshot:
        """Store one complete `/target_cube/states` update."""

        normalized_names = tuple(
            str(name).strip()
            for name in cube_names
        )

        if any(not name for name in normalized_names):
            raise ValueError(
                "Target cube names must not be empty."
            )

        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(
                "Target cube names must not contain duplicates."
            )

        if source_sequence is None:
            validated_source_sequence = None
        elif (
            isinstance(source_sequence, bool)
            or not isinstance(source_sequence, int)
            or source_sequence < 0
        ):
            raise ValueError(
                "Source sequence must be a non-negative integer."
            )
        else:
            validated_source_sequence = source_sequence

        if source_sim_time_s is None:
            validated_source_sim_time_s = None
        else:
            if isinstance(source_sim_time_s, bool):
                raise ValueError(
                    "Source simulation time must be finite "
                    "and non-negative."
                )

            validated_source_sim_time_s = float(
                source_sim_time_s
            )

            if (
                not np.isfinite(validated_source_sim_time_s)
                or validated_source_sim_time_s < 0.0
            ):
                raise ValueError(
                    "Source simulation time must be finite "
                    "and non-negative."
                )

        received_time = _received_time_or_now(
            received_monotonic_s
        )

        with self._condition:
            snapshot = TargetCubeCollectionSnapshot(
                cube_names=normalized_names,
                sequence=self._next_sequence,
                received_monotonic_s=received_time,
                source_sequence=validated_source_sequence,
                source_sim_time_s=(
                    validated_source_sim_time_s
                ),
            )
            self._next_sequence += 1
            self._snapshot = snapshot
            self._condition.notify_all()

            return _copy_cube_collection_snapshot(snapshot)

    def latest(
        self,
    ) -> TargetCubeCollectionSnapshot | None:
        """Return an independent copy of the latest collection state."""

        with self._condition:
            if self._snapshot is None:
                return None

            return _copy_cube_collection_snapshot(
                self._snapshot
            )

    def wait_for_sequence_greater_than(
        self,
        sequence: int,
        timeout_s: float,
    ) -> TargetCubeCollectionSnapshot | None:
        """Wait for a collection update newer than ``sequence``."""

        if sequence < 0:
            raise ValueError(
                "Sequence must be non-negative."
            )

        timeout = _validate_timeout(timeout_s)
        deadline = monotonic() + timeout

        with self._condition:
            while (
                self._snapshot is None
                or self._snapshot.sequence <= sequence
            ):
                remaining = deadline - monotonic()

                if remaining <= 0.0:
                    return None

                self._condition.wait(timeout=remaining)

            return _copy_cube_collection_snapshot(
                self._snapshot
            )

@dataclass(frozen=True)
class GoalRegionSnapshot:
    """Immutable goal-region centre in the Gazebo world frame."""

    frame_id: str
    x: float
    y: float
    z: float
    received_monotonic_s: float


def _copy_goal_region_snapshot(
    snapshot: GoalRegionSnapshot,
) -> GoalRegionSnapshot:
    return GoalRegionSnapshot(
        frame_id=snapshot.frame_id,
        x=snapshot.x,
        y=snapshot.y,
        z=snapshot.z,
        received_monotonic_s=snapshot.received_monotonic_s,
    )


class GoalRegionCache:
    """Thread-safe one-shot cache for the static goal-region centre."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._snapshot: GoalRegionSnapshot | None = None

    def update_once(
        self,
        *,
        frame_id: str,
        x: float,
        y: float,
        z: float,
        received_monotonic_s: float | None = None,
    ) -> GoalRegionSnapshot:
        """Store the first valid world-frame centre and ignore later updates."""

        normalized_frame_id = str(frame_id).strip()

        if normalized_frame_id != "world":
            raise ValueError(
                "Goal-region frame_id must be 'world'."
            )

        raw_values = (x, y, z)

        if any(isinstance(value, bool) for value in raw_values):
            raise ValueError(
                "Goal-region coordinates must be finite numbers."
            )

        coordinates = tuple(
            float(value)
            for value in raw_values
        )

        if not all(np.isfinite(value) for value in coordinates):
            raise ValueError(
                "Goal-region coordinates must be finite numbers."
            )

        received_time = _received_time_or_now(
            received_monotonic_s
        )

        with self._condition:
            if self._snapshot is None:
                self._snapshot = GoalRegionSnapshot(
                    frame_id=normalized_frame_id,
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                    received_monotonic_s=received_time,
                )
                self._condition.notify_all()

            return _copy_goal_region_snapshot(
                self._snapshot
            )

    def latest(self) -> GoalRegionSnapshot | None:
        """Return an independent copy of the cached centre."""

        with self._condition:
            if self._snapshot is None:
                return None

            return _copy_goal_region_snapshot(
                self._snapshot
            )

    def wait_until_available(
        self,
        timeout_s: float,
    ) -> GoalRegionSnapshot | None:
        """Wait until the first valid centre has been cached."""

        timeout = _validate_timeout(timeout_s)
        deadline = monotonic() + timeout

        with self._condition:
            while self._snapshot is None:
                remaining = deadline - monotonic()

                if remaining <= 0.0:
                    return None

                self._condition.wait(timeout=remaining)

            return _copy_goal_region_snapshot(
                self._snapshot
            )
