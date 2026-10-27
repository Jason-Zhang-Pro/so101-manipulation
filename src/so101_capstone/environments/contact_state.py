"""Timestamped finger contacts, independent of ROS imports.

An empty message is evidence of no contact. Missing messages are unknown.
The cache never extrapolates the last contact through a sensor outage.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import re
from threading import Condition
from typing import Any

FIXED_FINGER_CONTACT_TOPIC = "/so101/contacts/fixed_finger"
MOVING_FINGER_CONTACT_TOPIC = "/so101/contacts/moving_finger"


@dataclass(frozen=True)
class ContactSample:
    sim_time_s: float
    collision_pairs: tuple[tuple[str, str], ...]

    def touches(self, target_name: str | None) -> bool:
        if target_name is None:
            return False
        # Exact scoped model tokens prevent target_cube_1 matching _10.
        return any(
            target_name in re.split(r"::|/", name)
            for pair in self.collision_pairs for name in pair
        )


@dataclass(frozen=True)
class ContactEvidence:
    valid: bool
    left_contact: bool = False
    right_contact: bool = False
    contacts_simultaneous: bool = False
    sim_time_s: float | None = None
    covered_duration_s: float = 0.0
    simultaneous_duration_s: float = 0.0
    reason: str | None = None


class FingerContactCache:
    """Bounded two-channel history, with a physical clock reset boundary."""

    def __init__(self, max_samples: int = 1024) -> None:
        if max_samples < 2:
            raise ValueError("At least two contact history samples are required.")
        self._condition = Condition()
        self._samples: dict[str, deque[ContactSample]] = {
            side: deque(maxlen=max_samples) for side in ("left", "right")
        }

    def update(self, side: str, *, sim_time_s: float,
               collision_pairs: tuple[tuple[str, str], ...] = ()) -> None:
        if side not in self._samples:
            raise ValueError("Contact side must be left or right.")
        timestamp = float(sim_time_s)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Contact source time must be finite and non-negative.")
        pairs = tuple((str(a), str(b)) for a, b in collision_pairs)
        with self._condition:
            history = self._samples[side]
            if history and timestamp < history[-1].sim_time_s - 1e-9:
                # Clear both sides so contacts from before a reset cannot pair.
                for samples in self._samples.values():
                    samples.clear()
            if history and abs(timestamp - history[-1].sim_time_s) <= 1e-9:
                history[-1] = ContactSample(timestamp, pairs)
            else:
                history.append(ContactSample(timestamp, pairs))
            self._condition.notify_all()

    def update_from_message(self, side: str, message: Any) -> None:
        try:
            stamp = message.header.stamp
            nanosec = int(stamp.nanosec)
            if not 0 <= nanosec < 1_000_000_000:
                raise ValueError("Invalid contact nanoseconds.")
            timestamp = int(stamp.sec) + nanosec * 1e-9
            pairs = tuple(
                (item.collision1.name, item.collision2.name)
                for item in message.contacts
            )
        except (AttributeError, TypeError) as error:
            raise ValueError("Contact message requires a source stamp and collision names.") from error
        self.update(side, sim_time_s=timestamp, collision_pairs=pairs)

    def history(self) -> dict[str, tuple[ContactSample, ...]]:
        with self._condition:
            return {side: tuple(values) for side, values in self._samples.items()}

    @staticmethod
    def _at(samples: tuple[ContactSample, ...], time_s: float,
            target_name: str | None, max_gap_s: float) -> tuple[bool, bool]:
        before = after = None
        for sample in samples:
            if abs(sample.sim_time_s - time_s) <= 1e-8:
                return True, sample.touches(target_name)
            if sample.sim_time_s < time_s:
                before = sample
            elif sample.sim_time_s > time_s:
                after = sample
                break
        if before is None or after is None or after.sim_time_s - before.sim_time_s > max_gap_s:
            return False, False
        first, last = before.touches(target_name), after.touches(target_name)
        # A transition between the bracketing samples has unknown exact time.
        return (first == last), first and last

    def evidence(self, *, target_name: str | None, end_time_s: float,
                 start_time_s: float | None = None,
                 max_gap_s: float = 0.12) -> ContactEvidence:
        if max_gap_s <= 0 or not math.isfinite(max_gap_s):
            raise ValueError("Contact sample gap must be positive and finite.")
        histories = self.history()
        end = float(end_time_s)
        start = end if start_time_s is None else float(start_time_s)
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("Contact interval must be finite and ordered.")
        points = sorted({start, end, *(
            sample.sim_time_s for samples in histories.values() for sample in samples
            if start < sample.sim_time_s < end
        )})
        statuses: list[tuple[bool, bool]] = []
        for time_s in points:
            left_ok, left = self._at(histories["left"], time_s, target_name, max_gap_s)
            right_ok, right = self._at(histories["right"], time_s, target_name, max_gap_s)
            if not left_ok or not right_ok:
                return ContactEvidence(False, sim_time_s=end, reason="missing_or_ambiguous_contact_interval")
            statuses.append((left, right))
        if any(b - a > max_gap_s for a, b in zip(points, points[1:])):
            return ContactEvidence(False, sim_time_s=end, reason="contact_sample_gap")
        overlap = sum(
            points[index + 1] - points[index]
            for index in range(len(points) - 1)
            if all(statuses[index]) and all(statuses[index + 1])
        )
        left, right = statuses[-1]
        simultaneous = all(all(status) for status in statuses)
        return ContactEvidence(True, left, right, simultaneous, end,
                               end - start, overlap)
