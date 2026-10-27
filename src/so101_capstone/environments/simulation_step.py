"""Deterministic mapping from policy steps to simulation physics steps."""

from __future__ import annotations

from fractions import Fraction
import math


def _positive_fraction(
    value: float,
    *,
    name: str,
) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number.")

    numeric_value = float(value)

    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")

    return Fraction(str(value))


class SimulationStepSchedule:
    """Schedule integer physics steps for a fixed simulation-time rate.

    For 30 Hz control with a 1 ms physics step, the generated sequence is:

        33, 33, 34, 33, 33, 34, ...

    The schedule uses exact rational arithmetic, so cumulative timing does not
    drift during long episodes.
    """

    def __init__(
        self,
        *,
        control_hz: float = 30.0,
        physics_step_s: float = 0.001,
    ) -> None:
        self._control_hz = _positive_fraction(
            control_hz,
            name="control_hz",
        )
        self._physics_step_s = _positive_fraction(
            physics_step_s,
            name="physics_step_s",
        )
        self._control_period_s = (
            Fraction(1, 1) / self._control_hz
        )

        physics_hz = (
            Fraction(1, 1) / self._physics_step_s
        )

        if physics_hz < self._control_hz:
            raise ValueError(
                "Physics update frequency must be at least "
                "the control frequency."
            )

        self._control_step_count = 0
        self._physics_step_count = 0

    @property
    def control_hz(self) -> float:
        return float(self._control_hz)

    @property
    def physics_step_s(self) -> float:
        return float(self._physics_step_s)

    @property
    def control_step_count(self) -> int:
        return self._control_step_count

    @property
    def physics_step_count(self) -> int:
        return self._physics_step_count

    @property
    def elapsed_sim_time_s(self) -> float:
        return float(
            self._physics_step_count
            * self._physics_step_s
        )

    @property
    def ideal_elapsed_sim_time_s(self) -> float:
        return float(
            self._control_step_count
            * self._control_period_s
        )

    @property
    def timing_error_s(self) -> float:
        """Actual stepped time minus the ideal control-grid time."""

        return (
            self.elapsed_sim_time_s
            - self.ideal_elapsed_sim_time_s
        )

    def reset(self) -> None:
        self._control_step_count = 0
        self._physics_step_count = 0

    def next_step_count(self) -> int:
        """Return the physics steps required for the next control step."""

        next_control_step = self._control_step_count + 1

        ideal_total_physics_steps = (
            next_control_step
            * self._control_period_s
            / self._physics_step_s
        )

        target_total_physics_steps = (
            ideal_total_physics_steps.numerator
            // ideal_total_physics_steps.denominator
        )

        step_count = (
            target_total_physics_steps
            - self._physics_step_count
        )

        if step_count < 1:
            raise RuntimeError(
                "Control step did not advance simulation time."
            )

        self._control_step_count = next_control_step
        self._physics_step_count = (
            target_total_physics_steps
        )

        return step_count
