"""Preflight cleanup for SO-101 multi-world Gazebo training.

The cleanup is intentionally scoped to processes that both:

1. use one of the ROS domains / Gazebo transport partitions reserved by the
   pending finetune run, and
2. look like a managed SO-101 / Gazebo simulation or Direct ACT PPO process.

This avoids a blanket ``pkill gz`` / ``pkill python`` while still reclaiming
stale worlds and ROS nodes from interrupted runs before new workers start.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import signal
import time
from typing import Callable, Iterable, Mapping


_MANAGED_COMMAND_MARKERS = (
    "so101_capstone.training.act_ppo_multiworld_gazebo",
    "so101_capstone.training.act_ppo_gazebo",
    "so101_bringup sim.launch.py",
    "ros2 launch so101_bringup",
    "ign gazebo",
    "gz sim",
    "parameter_bridge",
    "robot_state_publisher",
    "target_cube_manager.py",
    "goal_region_pose_publisher.py",
    "joint_state_broadcaster",
    "position_controller",
    "controller_manager/spawner",
    "controller_manager spawner",
    "ros_gz_sim create",
    "/ros_gz_sim/create",
    "ign topic",
    "gz topic",
)


class MultiWorldPreflightCleanupError(RuntimeError):
    """Raised when stale managed simulation processes cannot be cleared."""


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    ppid: int
    command: str
    environment: Mapping[str, str]


@dataclass(frozen=True)
class ProcessCleanupRecord:
    pid: int
    ppid: int
    command: str
    ros_domain_id: str | None
    gz_partition: str | None
    ign_partition: str | None


@dataclass(frozen=True)
class PreflightCleanupReport:
    enabled: bool
    target_ros_domain_ids: tuple[int, ...]
    target_gz_partitions: tuple[str, ...]
    matched_processes: tuple[ProcessCleanupRecord, ...]
    terminated_processes: tuple[ProcessCleanupRecord, ...]
    killed_processes: tuple[ProcessCleanupRecord, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "target_ros_domain_ids": list(self.target_ros_domain_ids),
            "target_gz_partitions": list(self.target_gz_partitions),
            "matched_processes": [asdict(item) for item in self.matched_processes],
            "terminated_processes": [asdict(item) for item in self.terminated_processes],
            "killed_processes": [asdict(item) for item in self.killed_processes],
        }


def _read_environment(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def _read_command(path: Path) -> str:
    raw = path.read_bytes()
    parts = [item.decode(errors="replace") for item in raw.split(b"\0") if item]
    return " ".join(parts)


def _read_ppid(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    close = text.rfind(")")
    if close < 0:
        raise ValueError("Malformed /proc stat record.")
    fields = text[close + 1 :].strip().split()
    if len(fields) < 2:
        raise ValueError("Malformed /proc stat record.")
    return int(fields[1])


def scan_processes(proc_root: Path = Path("/proc")) -> list[ProcessSnapshot]:
    snapshots: list[ProcessSnapshot] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as error:
        raise MultiWorldPreflightCleanupError(
            f"Could not scan process table at {proc_root}: {error}"
        ) from error

    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            command = _read_command(entry / "cmdline")
            if not command:
                continue
            environment = _read_environment(entry / "environ")
            ppid = _read_ppid(entry / "stat")
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
            continue
        snapshots.append(
            ProcessSnapshot(
                pid=pid,
                ppid=ppid,
                command=command,
                environment=environment,
            )
        )
    return snapshots


def _ancestor_pids(snapshots: Iterable[ProcessSnapshot], *, start_pid: int) -> set[int]:
    by_pid = {item.pid: item for item in snapshots}
    result: set[int] = set()
    current = int(start_pid)
    while current > 0 and current not in result:
        result.add(current)
        snapshot = by_pid.get(current)
        if snapshot is None or snapshot.ppid <= 0 or snapshot.ppid == current:
            break
        current = snapshot.ppid
    return result


def _resource_matches(snapshot: ProcessSnapshot, *, target_domains: set[str], target_partitions: set[str]) -> bool:
    environment = snapshot.environment
    return (
        environment.get("ROS_DOMAIN_ID") in target_domains
        or environment.get("GZ_PARTITION") in target_partitions
        or environment.get("IGN_PARTITION") in target_partitions
    )


def _command_is_managed(command: str) -> bool:
    normalized = " ".join(str(command).lower().split())
    return any(marker in normalized for marker in _MANAGED_COMMAND_MARKERS)


def managed_process_matches(
    snapshot: ProcessSnapshot,
    *,
    target_ros_domain_ids: Iterable[int],
    target_gz_partitions: Iterable[str],
    excluded_pids: Iterable[int] = (),
) -> bool:
    if snapshot.pid in set(int(value) for value in excluded_pids):
        return False
    target_domains = {str(int(value)) for value in target_ros_domain_ids}
    target_partitions = {str(value) for value in target_gz_partitions}
    return _resource_matches(
        snapshot,
        target_domains=target_domains,
        target_partitions=target_partitions,
    ) and _command_is_managed(snapshot.command)


def _descendants(snapshots: Iterable[ProcessSnapshot], root_pids: set[int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for item in snapshots:
        children.setdefault(item.ppid, []).append(item.pid)
    result = set(root_pids)
    pending = list(root_pids)
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _record(snapshot: ProcessSnapshot) -> ProcessCleanupRecord:
    environment = snapshot.environment
    return ProcessCleanupRecord(
        pid=snapshot.pid,
        ppid=snapshot.ppid,
        command=snapshot.command,
        ros_domain_id=environment.get("ROS_DOMAIN_ID"),
        gz_partition=environment.get("GZ_PARTITION"),
        ign_partition=environment.get("IGN_PARTITION"),
    )


def _process_is_active(pid: int, proc_root: Path) -> bool:
    stat_path = proc_root / str(pid) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False
    close = text.rfind(")")
    if close < 0:
        return True
    fields = text[close + 1 :].strip().split()
    if not fields:
        return True
    return fields[0] != "Z"


def cleanup_stale_multiworld_processes(
    *,
    target_ros_domain_ids: Iterable[int],
    target_gz_partitions: Iterable[str],
    timeout_s: float = 5.0,
    proc_root: Path = Path("/proc"),
    current_pid: int | None = None,
    signal_process: Callable[[int, int], None] = os.kill,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PreflightCleanupReport:
    timeout = float(timeout_s)
    if timeout <= 0.0:
        raise ValueError("timeout_s must be positive.")
    domains = tuple(int(value) for value in target_ros_domain_ids)
    partitions = tuple(str(value) for value in target_gz_partitions)
    if not domains:
        raise ValueError("At least one ROS domain is required.")
    if len(set(domains)) != len(domains):
        raise ValueError("ROS domain IDs must be unique.")
    if any(not 0 <= value <= 232 for value in domains):
        raise ValueError("ROS domain IDs must be within [0, 232].")
    if not partitions or any(not value for value in partitions):
        raise ValueError("Gazebo partitions must be non-empty.")

    snapshots = scan_processes(proc_root)
    own_pid = os.getpid() if current_pid is None else int(current_pid)
    excluded = _ancestor_pids(snapshots, start_pid=own_pid)
    roots = {
        item.pid
        for item in snapshots
        if managed_process_matches(
            item,
            target_ros_domain_ids=domains,
            target_gz_partitions=partitions,
            excluded_pids=excluded,
        )
    }
    # Kill only processes that themselves match the managed simulation
    # signature. Do not recursively kill arbitrary descendants (for
    # example, a browser opened by an old TensorBoard process).
    target_pids = roots - excluded
    by_pid = {item.pid: item for item in snapshots}
    matched = tuple(_record(by_pid[pid]) for pid in sorted(target_pids) if pid in by_pid)
    if not target_pids:
        return PreflightCleanupReport(
            enabled=True,
            target_ros_domain_ids=domains,
            target_gz_partitions=partitions,
            matched_processes=(),
            terminated_processes=(),
            killed_processes=(),
        )

    terminated: list[ProcessCleanupRecord] = []
    for pid in sorted(target_pids, reverse=True):
        snapshot = by_pid.get(pid)
        try:
            signal_process(pid, signal.SIGTERM)
            if snapshot is not None:
                terminated.append(_record(snapshot))
        except ProcessLookupError:
            continue
        except PermissionError as error:
            raise MultiWorldPreflightCleanupError(
                f"Permission denied terminating stale process {pid}."
            ) from error

    deadline = time.monotonic() + timeout
    remaining = {pid for pid in target_pids if _process_is_active(pid, proc_root)}
    while remaining and time.monotonic() < deadline:
        sleep_fn(min(0.05, max(0.0, deadline - time.monotonic())))
        remaining = {pid for pid in remaining if _process_is_active(pid, proc_root)}

    killed: list[ProcessCleanupRecord] = []
    for pid in sorted(remaining, reverse=True):
        snapshot = by_pid.get(pid)
        try:
            signal_process(pid, signal.SIGKILL)
            if snapshot is not None:
                killed.append(_record(snapshot))
        except ProcessLookupError:
            continue
        except PermissionError as error:
            raise MultiWorldPreflightCleanupError(
                f"Permission denied killing stale process {pid}."
            ) from error

    if remaining:
        final_deadline = time.monotonic() + min(1.0, timeout)
        still_active = set(remaining)
        while still_active and time.monotonic() < final_deadline:
            sleep_fn(0.05)
            still_active = {pid for pid in still_active if _process_is_active(pid, proc_root)}
        if still_active:
            raise MultiWorldPreflightCleanupError(
                "Stale managed simulation processes remained after SIGKILL: "
                + ", ".join(str(pid) for pid in sorted(still_active))
            )

    sleep_fn(0.25)
    return PreflightCleanupReport(
        enabled=True,
        target_ros_domain_ids=domains,
        target_gz_partitions=partitions,
        matched_processes=matched,
        terminated_processes=tuple(terminated),
        killed_processes=tuple(killed),
    )
