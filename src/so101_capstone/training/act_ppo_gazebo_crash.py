"""Conservative, bounded crash evidence from a single Gazebo launch log.

Timeouts and a failed ros2 launch wrapper are not evidence that Gazebo crashed.
Only Gazebo-tagged fatal signals are classified as confirmed. Call this on a
failure, before stopping the world; it performs no IPC and no process control.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_GAZEBO_PROCESS = re.compile(
    r"\[(?:ign(?:ition)?[ _-]+gazebo|gz[ _-]+sim|gzserver|gazebo)(?:-\d+)?\]",
    re.IGNORECASE,
)
_FATAL_MESSAGE = re.compile(
    r"^\s*(?:Segmentation fault|Bus error|Floating point exception|Illegal instruction)\b"
    r"|^\s*Aborted\s*(?:\(core dumped\))?\s*$"
    r"|\b(?:signal\s*[:=]?\s*(?:SIGSEGV|SIGABRT|SIGBUS|SIGFPE|SIGILL|11|6|7|8|4))\b",
    re.IGNORECASE,
)
_EXIT_CODE = re.compile(r"\bexit code\s*[:=]?\s*(-?\d+)\b", re.IGNORECASE)
_FATAL_EXIT_CODES = {-11, -6, -7, -8, -4, 139, 134, 135, 136, 132}


def read_gazebo_crash_diagnostics(
    log_path: str | Path,
    *,
    launch_log_offset: int | None = None,
    max_read_bytes: int = 262144,
) -> dict[str, Any]:
    """Return evidence, never promoting missing logs or timeouts to crashes.

    ``launch_log_offset`` is the byte offset recorded immediately before this
    launch. Without it we cannot safely attribute an appended log to the current
    episode, so historical crash lines are deliberately not read. Very large
    launches are inspected only at the tail, and the result reports truncation.
    A false ``confirmed`` means no confirmation, not proof of a healthy world.
    """

    result: dict[str, Any] = {
        "confirmed": False,
        "evidence": [],
        "log_path": str(log_path),
        "launch_log_offset": launch_log_offset,
        "status": "unknown",
    }
    if launch_log_offset is None:
        result["reason"] = "launch_log_offset_unavailable"
        return result
    if (
        isinstance(launch_log_offset, bool)
        or not isinstance(launch_log_offset, int)
        or launch_log_offset < 0
        or isinstance(max_read_bytes, bool)
        or not isinstance(max_read_bytes, int)
        or max_read_bytes < 1
    ):
        result["reason"] = "invalid_log_read_bounds"
        return result
    try:
        with Path(log_path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            if size < launch_log_offset:
                result["reason"] = "log_truncated_since_launch"
                return result
            start = max(launch_log_offset, size - max_read_bytes)
            stream.seek(start)
            raw = stream.read(min(max_read_bytes, size - start))
    except (OSError, ValueError) as error:
        result["reason"] = "log_read_failed"
        result["read_error"] = f"{type(error).__name__}: {error}"
        return result
    result.update(
        read_start_offset=start,
        read_end_offset=start + len(raw),
        bytes_read=len(raw),
        truncated=start > launch_log_offset,
        status="unconfirmed",
    )
    if start > launch_log_offset:
        # Do not classify a line whose process prefix was truncated away.
        raw = raw.partition(b"\n")[2]
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = _ANSI_ESCAPE.sub("", line)
        process = _GAZEBO_PROCESS.search(line)
        if process is None:
            continue
        code = _EXIT_CODE.search(line)
        fatal_exit = (
            "process has died" in line.lower()
            and code is not None
            and int(code.group(1)) in _FATAL_EXIT_CODES
        )
        if _FATAL_MESSAGE.search(line[process.end():]) or fatal_exit:
            result["evidence"].append({
                "kind": "fatal_exit" if fatal_exit else "fatal_signal",
                "line": line[:2048],
            })
            if len(result["evidence"]) >= 8:
                break
    if result["evidence"]:
        result["confirmed"] = True
        result["status"] = "confirmed"
    return result
