"""Run one policy-controlled episode against the deterministic runtime.

This module owns only the policy/environment interaction loop. Target
generation, target deletion, multi-episode orchestration, persistence, and
policy optimization remain separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Sequence

import numpy as np

from so101_capstone.environments import (
    PolicyAction,
    PolicyObservation,
    parse_policy_action,
)


class PolicyEpisodeActionSafetyError(ValueError):
    '''Policy output could not be converted to a safe controller action.'''


class PolicyEpisodeActionResult(Protocol):
    """Minimum policy output required by the runner."""

    def controller_command(self) -> Sequence[float]:
        """Return one six-dimensional absolute controller command."""


class PolicyEpisodePolicy(Protocol):
    """Minimum policy lifecycle required for one episode."""

    def reset(self) -> None:
        """Clear policy state and any queued action chunk."""

    def select_action(
        self,
        observation: PolicyObservation,
    ) -> PolicyEpisodeActionResult:
        """Select the action for the current observation."""


class PolicyEpisodeRuntimeStep(Protocol):
    """Runtime step data consumed by the runner."""

    observation: PolicyObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class PolicyEpisodeRuntime(Protocol):
    """Minimum deterministic runtime interface required by the runner."""

    def prepare_episode(self) -> Any:
        """Return the prepared frame-0 observation and metadata."""

    def step(
        self,
        joint_targets: Sequence[float],
    ) -> PolicyEpisodeRuntimeStep:
        """Execute one policy/environment step."""


def _validated_max_steps(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            "max_steps must be a positive integer."
        )

    return value


def _validated_reward(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(
            "Runtime reward must be a finite number."
        )

    reward = float(value)

    if not math.isfinite(reward):
        raise ValueError(
            "Runtime reward must be a finite number."
        )

    return reward


@dataclass(frozen=True)
class PolicyEpisodeState:
    """Image-free state retained by long-running evaluation archives."""

    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    @classmethod
    def from_observation(cls, observation: PolicyObservation) -> PolicyEpisodeState:
        arrays = []
        for value in (observation.joint_positions, observation.joint_velocities):
            array = np.asarray(value, dtype=float).copy()
            array.setflags(write=False)
            arrays.append(array)
        return cls(*arrays)


@dataclass(frozen=True)
class PolicyTransition:
    """One complete environment transition."""

    step_index: int
    observation: PolicyObservation | PolicyEpisodeState
    action: PolicyAction
    reward: float
    next_observation: PolicyObservation | PolicyEpisodeState
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError(
                "step_index must be a non-negative integer."
            )

        if not isinstance(self.terminated, bool):
            raise ValueError(
                "terminated must be a bool."
            )

        if not isinstance(self.truncated, bool):
            raise ValueError(
                "truncated must be a bool."
            )

        object.__setattr__(
            self,
            "reward",
            _validated_reward(self.reward),
        )
        object.__setattr__(
            self,
            "info",
            dict(self.info),
        )

    @property
    def done(self) -> bool:
        """Return whether the environment ended on this transition."""

        return self.terminated or self.truncated


class PolicyTransitionSink(Protocol):
    """Receive completed transitions from one policy episode."""

    def reset(self) -> None:
        """Clear state at the start of a newly prepared episode."""

    def append(
        self,
        transition: PolicyTransition,
    ) -> None:
        """Record one completed environment transition."""

    def finalize(
        self,
        episode: PolicyEpisodeResult,
    ) -> None:
        """Finalize one completed policy episode."""



class InMemoryPolicyTransitionSink:
    """Keep completed transitions in memory for later consumers."""

    def __init__(self) -> None:
        self._transitions: list[PolicyTransition] = []

    def reset(self) -> None:
        """Clear transitions from the previous episode."""

        self._transitions.clear()

    def append(
        self,
        transition: PolicyTransition,
    ) -> None:
        """Append one completed transition."""

        if not isinstance(transition, PolicyTransition):
            raise TypeError(
                "transition must be a PolicyTransition."
            )

        expected_step_index = len(self._transitions)

        if transition.step_index != expected_step_index:
            raise ValueError(
                "Transition step_index must be contiguous: "
                f"expected {expected_step_index}, "
                f"received {transition.step_index}."
            )

        self._transitions.append(transition)

    @property
    def transitions(self) -> tuple[PolicyTransition, ...]:
        """Return an immutable transition snapshot."""

        return tuple(self._transitions)

    def finalize(
        self,
        episode: PolicyEpisodeResult,
    ) -> None:
        """Keep the in-memory snapshot unchanged after completion."""

        if len(self._transitions) != len(
            episode.transitions
        ) or not all(
            stored is finalized
            for stored, finalized in zip(
                self._transitions,
                episode.transitions,
                strict=True,
            )
        ):
            raise ValueError(
                "In-memory transitions do not match "
                "the finalized episode."
            )



@dataclass(frozen=True)
class PolicyEpisodeResult:
    """Complete result of one policy-controlled episode."""

    initial_observation: PolicyObservation | PolicyEpisodeState
    initial_info: dict[str, Any]
    transitions: tuple[PolicyTransition, ...]
    final_observation: PolicyObservation | PolicyEpisodeState
    final_info: dict[str, Any]
    terminated: bool
    truncated: bool
    reached_max_steps: bool
    total_reward: float

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError(
                "Policy episode must contain at least one transition."
            )

        for name in (
            "terminated",
            "truncated",
            "reached_max_steps",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"{name} must be a bool."
                )

        stop_reasons = sum(
            (
                self.terminated,
                self.truncated,
                self.reached_max_steps,
            )
        )

        if stop_reasons != 1:
            raise ValueError(
                "Exactly one episode stop reason must be true."
            )

        final_transition = self.transitions[-1]

        if self.final_observation is not final_transition.next_observation:
            raise ValueError(
                "final_observation must be the final transition's "
                "next_observation."
            )

        if self.terminated != final_transition.terminated:
            raise ValueError(
                "terminated does not match the final transition."
            )

        if self.truncated != final_transition.truncated:
            raise ValueError(
                "truncated does not match the final transition."
            )

        object.__setattr__(
            self,
            "transitions",
            tuple(self.transitions),
        )
        object.__setattr__(
            self,
            "initial_info",
            dict(self.initial_info),
        )
        object.__setattr__(
            self,
            "final_info",
            dict(self.final_info),
        )
        object.__setattr__(
            self,
            "total_reward",
            _validated_reward(self.total_reward),
        )

    @property
    def step_count(self) -> int:
        """Return the number of executed runtime steps."""

        return len(self.transitions)


class PolicyEpisodeRunner:
    """Execute one complete policy/environment episode."""

    def __init__(
        self,
        runtime: PolicyEpisodeRuntime,
        policy: PolicyEpisodePolicy,
        *,
        transition_sink: PolicyTransitionSink | None = None,
        retain_observation_images: bool = True,
        observe_after_success_s: float = 0.0,
        policy_hz: float = 30.0,
    ) -> None:
        self.runtime = runtime
        self.policy = policy
        self.transition_sink = transition_sink
        if not isinstance(retain_observation_images, bool):
            raise ValueError("retain_observation_images must be a bool.")
        self.retain_observation_images = retain_observation_images
        if not math.isfinite(observe_after_success_s) or observe_after_success_s < 0:
            raise ValueError("observe_after_success_s must be finite and non-negative.")
        if not math.isfinite(policy_hz) or policy_hz <= 0:
            raise ValueError("policy_hz must be finite and positive.")
        self.monitor_step_budget = math.ceil(observe_after_success_s * policy_hz)

    def run_episode(
        self,
        *,
        max_steps: int,
    ) -> PolicyEpisodeResult:
        """Prepare, reset policy state, and execute one episode.

        The final next-observation is recorded in the final transition.
        The policy is not queried again after a terminal, truncated, or
        max-step transition.
        """

        step_limit = _validated_max_steps(max_steps)

        prepared = self.runtime.prepare_episode()
        initial_observation = prepared.observation
        initial_info = dict(prepared.info)

        if self.transition_sink is not None:
            self.transition_sink.reset()

        # This is the episode boundary for ACT's action-chunk queue.
        self.policy.reset()

        current_observation = initial_observation
        recorded_initial = (
            initial_observation
            if self.retain_observation_images
            else PolicyEpisodeState.from_observation(initial_observation)
        )
        recorded_current = recorded_initial
        transitions: list[PolicyTransition] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        first_success_step: int | None = None
        first_success_time_s: float | None = None
        task_metrics_enabled = False
        monitor_steps = monitor_home_violations = monitor_object_violations = monitor_regrasp_steps = 0
        monitor_peak_speed = monitor_max_home_error = 0.0
        monitor_velocity_sum_squares = np.zeros(6, dtype=float)
        idle_steps = idle_home_violations = idle_object_violations = idle_regrasp_steps = 0
        idle_peak_speed = idle_max_home_error = 0.0
        idle_velocity_sum_squares = np.zeros(6, dtype=float)

        for step_index in range(step_limit + self.monitor_step_budget):
            # Only a real success event can extend the task budget. The core
            # owns monitor termination and keeps ACT's queue uninterrupted.
            if step_index >= step_limit and first_success_step is None:
                break
            policy_result = self.policy.select_action(
                current_observation
            )
            try:
                action = parse_policy_action(
                    policy_result.controller_command()
                )
            except (ValueError, FloatingPointError) as error:
                raise PolicyEpisodeActionSafetyError(
                    f"Policy action validation failed: {error}"
                ) from error

            runtime_step = self.runtime.step(
                action.controller_command()
            )
            reward = _validated_reward(
                runtime_step.reward
            )
            info = runtime_step.info
            task_metrics_enabled = task_metrics_enabled or "task_outcome" in info or "success_event" in info
            task_mode = info.get("task_mode", initial_info.get("task_mode"))
            if task_mode in ("idle_empty", "idle_completed"):
                idle_steps += 1
                idle_home_violations += int(any(info.get(key) is False for key in ("robot_at_home", "robot_static")))
                velocities = np.asarray(runtime_step.observation.joint_velocities, dtype=float)
                idle_velocity_sum_squares += velocities ** 2
                idle_peak_speed = max(idle_peak_speed, float(np.abs(velocities).max()))
                idle_max_home_error = max(idle_max_home_error, float(info.get("home_error_rad", 0.0)))
                if task_mode == "idle_completed":
                    # The evaluator rebuilds its placement dwell after reset;
                    # that confirmation delay is not a physical disturbance.
                    idle_object_violations += int(any(info.get(key) is False for key in
                        ("target_in_goal", "target_supported", "target_static")))
                    idle_regrasp_steps += int(bool(info.get("grasp_valid", False)))
            if first_success_step is None and bool(info.get("success_event", False)):
                first_success_step = step_index + 1
                timestamp = info.get("sim_time_s", info.get("step_end_sim_time_s"))
                first_success_time_s = None if timestamp is None else float(timestamp)
            elif first_success_step is not None:
                monitor_steps += 1
                monitor_home_violations += int(any(info.get(key) is False for key in ("robot_at_home", "robot_static")))
                monitor_object_violations += int(any(info.get(key) is False for key in ("placed_stable", "target_in_goal", "target_supported", "target_static")))
                monitor_regrasp_steps += int(bool(info.get("grasp_valid", False)))
                velocities = np.asarray(runtime_step.observation.joint_velocities, dtype=float)
                monitor_velocity_sum_squares += velocities ** 2
                monitor_peak_speed = max(monitor_peak_speed, float(np.abs(velocities).max()))
                monitor_max_home_error = max(monitor_max_home_error, float(info.get("home_error_rad", 0.0)))

            recorded_next = (
                runtime_step.observation
                if self.retain_observation_images
                else PolicyEpisodeState.from_observation(runtime_step.observation)
            )
            transition = PolicyTransition(
                step_index=step_index,
                observation=recorded_current,
                action=action,
                reward=reward,
                next_observation=recorded_next,
                terminated=runtime_step.terminated,
                truncated=runtime_step.truncated,
                info=runtime_step.info,
            )
            transitions.append(transition)

            if self.transition_sink is not None:
                self.transition_sink.append(transition)

            total_reward += reward

            current_observation = runtime_step.observation
            recorded_current = recorded_next
            terminated = runtime_step.terminated
            truncated = runtime_step.truncated

            if terminated or truncated:
                break

        reached_max_steps = (
            not terminated
            and not truncated
        )

        final_transition = transitions[-1]

        final_info = dict(final_transition.info)
        if task_metrics_enabled:
            final_info["evaluation"] = {
                "task_success": first_success_step is not None,
                "first_success_step": first_success_step,
                "first_success_sim_time_s": first_success_time_s,
                "monitor_step_count": monitor_steps,
                "monitor_home_violation_steps": monitor_home_violations,
                "monitor_object_violation_steps": monitor_object_violations,
                "monitor_regrasp_steps": monitor_regrasp_steps,
                "monitor_peak_joint_speed_rad_s": monitor_peak_speed,
                "monitor_max_home_error_rad": monitor_max_home_error,
                "monitor_joint_velocity_rms_rad_s": (
                    np.sqrt(monitor_velocity_sum_squares / max(1, monitor_steps)).tolist()
                ),
                "monitor_completed": bool(first_success_step is not None and self.monitor_step_budget and terminated and not final_info.get("monitor_continuing", False)),
                "idle_step_count": idle_steps,
                "idle_home_violation_steps": idle_home_violations,
                "idle_object_violation_steps": idle_object_violations,
                "idle_regrasp_steps": idle_regrasp_steps,
                "idle_peak_joint_speed_rad_s": idle_peak_speed,
                "idle_max_home_error_rad": idle_max_home_error,
                "idle_joint_velocity_rms_rad_s": (
                    np.sqrt(idle_velocity_sum_squares / max(1, idle_steps)).tolist()
                ),
            }

        result = PolicyEpisodeResult(
            initial_observation=recorded_initial,
            initial_info=initial_info,
            transitions=tuple(transitions),
            final_observation=recorded_current,
            final_info=final_info,
            terminated=terminated,
            truncated=truncated,
            reached_max_steps=reached_max_steps,
            total_reward=total_reward,
        )

        if self.transition_sink is not None:
            self.transition_sink.finalize(result)

        return result
