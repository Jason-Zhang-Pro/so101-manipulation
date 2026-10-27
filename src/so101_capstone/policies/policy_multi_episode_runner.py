"""Run multiple policy episodes in one runtime process.

This module reuses one initialized runtime and one loaded policy across
episodes. Each inner episode still owns its formal boundary:
prepare_episode(), one policy reset, policy steps, final transition, and sink
finalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from so101_capstone.environments.episode_reset import (
    EpisodeResetError,
)
from so101_capstone.environments.gazebo_world_control import (
    GazeboWorldControlError,
)
from so101_capstone.environments.reach_env_core import (
    ReachEnvironmentStateError,
)
from so101_capstone.environments.simulation_runtime import (
    SimulationRuntimeError,
)
from so101_capstone.environments.simulation_step_coordinator import (
    SimulationStepCoordinatorError,
)
from so101_capstone.environments.target_cube_control import (
    TargetCubeControlError,
)

from .policy_episode_runner import (
    PolicyEpisodeActionSafetyError,
    PolicyEpisodePolicy,
    PolicyEpisodeResult,
    PolicyEpisodeRunner,
    PolicyEpisodeRuntime,
    PolicyTransitionSink,
)


DEFAULT_MAX_CONSECUTIVE_EPISODE_FAILURES = 3

_DELAYED_TARGET_SERVICE_FAILURE_PREFIXES = (
    "Failed to delete target cubes before episode:",
    "Failed to spawn a random target:",
)


def _recoverable_episode_category(
    error: Exception,
) -> str | None:
    '''Return a stable category only for explicitly recoverable failures.'''

    if isinstance(error, PolicyEpisodeActionSafetyError):
        return "act_action_safety"
    if isinstance(error, EpisodeResetError):
        return "episode_reset"
    if isinstance(error, TargetCubeControlError):
        return "target_service"
    if isinstance(error, GazeboWorldControlError):
        return "gazebo_control_or_timing"
    if isinstance(error, ReachEnvironmentStateError):
        return "environment_feedback"
    if isinstance(error, SimulationRuntimeError):
        return "simulation_runtime"
    if isinstance(error, SimulationStepCoordinatorError):
        message = str(error).lower()

        if "camera" in message or "image" in message:
            return "camera"
        if "joint" in message:
            return "joint_feedback"
        if (
            "clock" in message
            or "simulation-time" in message
            or "simulation time" in message
            or "sim_step" in message
            or "physics-step" in message
        ):
            return "gazebo_deterministic_timing"
        return "simulation_step"

    if isinstance(error, RuntimeError):
        message = str(error)
        if message.startswith(
            _DELAYED_TARGET_SERVICE_FAILURE_PREFIXES
        ):
            return "target_service"

    return None


class PolicyEpisodeSinkFactory(Protocol):
    """Create a distinct transition sink for one episode index."""

    def __call__(
        self,
        episode_index: int,
    ) -> PolicyTransitionSink:
        """Return the sink for one zero-based episode index."""


def _positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return value


@dataclass(frozen=True)
class PolicyEpisodeSummary:
    """Completed episode metrics without retaining frames or transitions."""

    step_count: int
    total_reward: float
    terminated: bool
    truncated: bool
    reached_max_steps: bool
    initial_info: dict[str, Any]
    final_info: dict[str, Any]

    @classmethod
    def from_episode(cls, episode: PolicyEpisodeResult) -> PolicyEpisodeSummary:
        return cls(
            step_count=episode.step_count,
            total_reward=episode.total_reward,
            terminated=episode.terminated,
            truncated=episode.truncated,
            reached_max_steps=episode.reached_max_steps,
            initial_info=dict(episode.initial_info),
            final_info=dict(episode.final_info),
        )


@dataclass(frozen=True)
class PolicyMultiEpisodeResult:
    """Results from a completed multi-episode run."""

    episodes: tuple[PolicyEpisodeResult | PolicyEpisodeSummary, ...]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError(
                "Multi-episode result must contain "
                "at least one episode."
            )

        object.__setattr__(
            self,
            "episodes",
            tuple(self.episodes),
        )

    @property
    def episode_count(self) -> int:
        """Return the number of completed episodes."""

        return len(self.episodes)

    @property
    def total_step_count(self) -> int:
        """Return the total number of runtime steps."""

        return sum(
            episode.step_count
            for episode in self.episodes
        )

    @property
    def total_reward(self) -> float:
        """Return the reward sum across all episodes."""

        return float(
            sum(
                episode.total_reward
                for episode in self.episodes
            )
        )


class PolicyMultiEpisodeRunner:
    """Run several formal episodes using one runtime and policy."""

    def __init__(
        self,
        runtime: PolicyEpisodeRuntime,
        policy: PolicyEpisodePolicy,
        *,
        episode_runner_factory: Callable[..., PolicyEpisodeRunner] = (
            PolicyEpisodeRunner
        ),
        retain_episode_transitions: bool = True,
        observe_after_success_s: float = 0.0,
        policy_hz: float = 30.0,
    ) -> None:
        self.runtime = runtime
        self.policy = policy
        self.episode_runner_factory = (
            episode_runner_factory
        )
        if not isinstance(retain_episode_transitions, bool):
            raise ValueError("retain_episode_transitions must be a bool.")
        self.retain_episode_transitions = retain_episode_transitions
        self.observe_after_success_s = observe_after_success_s
        self.policy_hz = policy_hz

    def run(
        self,
        *,
        episode_count: int,
        max_steps: int,
        transition_sink_factory: (
            PolicyEpisodeSinkFactory | None
        ) = None,
        max_consecutive_episode_failures: int = (
            DEFAULT_MAX_CONSECUTIVE_EPISODE_FAILURES
        ),
        before_episode: Callable[[int], None] | None = None,
    ) -> PolicyMultiEpisodeResult:
        '''Run the requested number of valid episodes.

        Explicitly recoverable runtime failures invalidate only the current
        attempt. The logical episode index is retried, so invalid attempts do
        not count toward ``episode_count`` and cannot enter the returned
        result. Any non-whitelisted exception is propagated immediately.
        '''

        count = _positive_integer(
            episode_count,
            name="episode_count",
        )
        step_limit = _positive_integer(
            max_steps,
            name="max_steps",
        )
        failure_limit = _positive_integer(
            max_consecutive_episode_failures,
            name="max_consecutive_episode_failures",
        )

        episodes: list[PolicyEpisodeResult | PolicyEpisodeSummary] = []
        attempt_index = 0
        consecutive_failures = 0

        while len(episodes) < count:
            episode_index = len(episodes)
            attempt_index += 1

            transition_sink = (
                None
                if transition_sink_factory is None
                else transition_sink_factory(
                    episode_index
                )
            )

            runner_kwargs = (
                {}
                if self.retain_episode_transitions
                else {"retain_observation_images": False}
            )
            if self.observe_after_success_s:
                runner_kwargs.update(observe_after_success_s=self.observe_after_success_s, policy_hz=self.policy_hz)
            if transition_sink is None:
                episode_runner = (
                    self.episode_runner_factory(
                        self.runtime,
                        self.policy,
                        **runner_kwargs,
                    )
                )
            else:
                episode_runner = (
                    self.episode_runner_factory(
                        self.runtime,
                        self.policy,
                        transition_sink=transition_sink,
                        **runner_kwargs,
                    )
                )

            try:
                if before_episode is not None:
                    before_episode(episode_index)
                episode = episode_runner.run_episode(
                    max_steps=step_limit,
                )
            except Exception as error:
                category = _recoverable_episode_category(
                    error
                )
                if category is None:
                    raise

                if transition_sink is not None:
                    transition_sink.reset()

                # Clear queued policy state before retrying. A failure in
                # reset itself remains fatal rather than being hidden.
                self.policy.reset()

                consecutive_failures += 1
                print(
                    "[RECOVERABLE EPISODE FAILURE] "
                    f"attempt={attempt_index} "
                    f"episode={episode_index} "
                    f"category={category} "
                    f"consecutive={consecutive_failures}/"
                    f"{failure_limit} "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

                if consecutive_failures >= failure_limit:
                    raise RuntimeError(
                        "Policy multi-episode run stopped after "
                        f"{consecutive_failures} consecutive "
                        "recoverable episode failures."
                    ) from error

                continue

            episodes.append(
                episode
                if self.retain_episode_transitions
                else PolicyEpisodeSummary.from_episode(episode)
            )
            if not self.retain_episode_transitions:
                release = getattr(transition_sink, "release_transitions", None)
                if release is not None:
                    release()
                del episode
            consecutive_failures = 0

        return PolicyMultiEpisodeResult(
            episodes=tuple(episodes),
        )
