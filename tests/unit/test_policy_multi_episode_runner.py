from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments import (
    PolicyObservation,
)
from so101_capstone.policies import (
    PolicyMultiEpisodeResult,
    PolicyMultiEpisodeRunner,
)


def observation(value):
    return PolicyObservation(
        joint_positions=np.full(
            6,
            value,
            dtype=float,
        ),
        joint_velocities=np.zeros(
            6,
            dtype=float,
        ),
        top_image=np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        ),
        gripper_image=np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        ),
    )


class FakePolicyResult:
    def __init__(self, value):
        self.value = value

    def controller_command(self):
        return np.full(
            6,
            self.value,
            dtype=float,
        )


class FakePolicy:
    def __init__(self, events):
        self.events = events
        self.reset_calls = 0
        self.select_calls = 0

    def reset(self):
        self.events.append("policy.reset")
        self.reset_calls += 1

    def select_action(self, current_observation):
        self.events.append("policy.select")
        self.select_calls += 1
        return FakePolicyResult(
            self.select_calls * 0.01
        )


class FakeRuntime:
    def __init__(self, events):
        self.events = events
        self.prepare_calls = 0
        self.step_calls = 0
        self.current_episode = -1

    def prepare_episode(self):
        self.current_episode += 1
        self.prepare_calls += 1
        self.events.append(
            f"runtime.prepare.{self.current_episode}"
        )
        return SimpleNamespace(
            observation=observation(
                self.current_episode * 10
            ),
            info={
                "episode_index": self.current_episode,
                "episode_step": 0,
            },
        )

    def step(self, command):
        self.step_calls += 1
        self.events.append(
            f"runtime.step.{self.current_episode}"
        )
        return SimpleNamespace(
            observation=observation(
                self.current_episode * 10 + 1
            ),
            reward=float(
                self.current_episode + 1
            ),
            terminated=True,
            truncated=False,
            info={
                "episode_index": self.current_episode,
                "episode_step": 1,
            },
        )


class RecordingSink:
    def __init__(self, episode_index):
        self.episode_index = episode_index
        self.transitions = []
        self.reset_calls = 0
        self.finalized_episode = None

    def reset(self):
        self.reset_calls += 1
        self.transitions.clear()

    def append(self, transition):
        self.transitions.append(transition)

    def finalize(self, episode):
        self.finalized_episode = episode


def test_compact_multi_episode_does_not_retain_frames_or_archived_transitions(tmp_path):
    import gc
    import weakref
    from so101_capstone.policies import PolicyEpisodeNpzSink, PolicyEpisodeSummary

    images = []

    class Runtime(FakeRuntime):
        def prepare_episode(self):
            prepared = super().prepare_episode()
            images.extend([weakref.ref(prepared.observation.top_image), weakref.ref(prepared.observation.gripper_image)])
            return prepared

        def step(self, action):
            result = super().step(action)
            images.extend([weakref.ref(result.observation.top_image), weakref.ref(result.observation.gripper_image)])
            return result

    sinks = []

    def sink_factory(index):
        sink = PolicyEpisodeNpzSink(tmp_path / f"episode_{index}.npz")
        sinks.append(sink)
        return sink

    result = PolicyMultiEpisodeRunner(Runtime([]), FakePolicy([]), retain_episode_transitions=False).run(
        episode_count=5, max_steps=2, transition_sink_factory=sink_factory,
    )
    gc.collect()
    assert result.episode_count == 5
    assert result.total_step_count == 5
    assert result.total_reward == 15.0
    assert all(isinstance(item, PolicyEpisodeSummary) for item in result.episodes)
    assert all(image() is None for image in images)
    assert all(sink.transitions == () and sink.written_path.is_file() for sink in sinks)


def test_multi_episode_runner_reuses_runtime_and_resets_policy_each_episode():
    events = []
    runtime = FakeRuntime(events)
    policy = FakePolicy(events)
    sinks = []

    def sink_factory(episode_index):
        sink = RecordingSink(episode_index)
        sinks.append(sink)
        return sink

    result = PolicyMultiEpisodeRunner(
        runtime,
        policy,
    ).run(
        episode_count=3,
        max_steps=5,
        transition_sink_factory=sink_factory,
    )

    assert isinstance(
        result,
        PolicyMultiEpisodeResult,
    )
    assert result.episode_count == 3
    assert result.total_step_count == 3
    assert result.total_reward == pytest.approx(6.0)

    assert runtime.prepare_calls == 3
    assert runtime.step_calls == 3
    assert policy.reset_calls == 3
    assert policy.select_calls == 3

    assert events == [
        "runtime.prepare.0",
        "policy.reset",
        "policy.select",
        "runtime.step.0",
        "runtime.prepare.1",
        "policy.reset",
        "policy.select",
        "runtime.step.1",
        "runtime.prepare.2",
        "policy.reset",
        "policy.select",
        "runtime.step.2",
    ]

    assert [
        sink.episode_index
        for sink in sinks
    ] == [0, 1, 2]

    for sink, episode in zip(
        sinks,
        result.episodes,
        strict=True,
    ):
        assert sink.reset_calls == 1
        assert len(sink.transitions) == 1
        assert sink.finalized_episode is episode


def test_multi_episode_runner_without_sink_factory():
    events = []
    runtime = FakeRuntime(events)
    policy = FakePolicy(events)

    result = PolicyMultiEpisodeRunner(
        runtime,
        policy,
    ).run(
        episode_count=2,
        max_steps=1,
    )

    assert result.episode_count == 2
    assert runtime.prepare_calls == 2
    assert policy.reset_calls == 2


@pytest.mark.parametrize(
    ("episode_count", "max_steps"),
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        (1, 0),
        (1, -1),
        (1, True),
        (1.5, 1),
        (1, 1.5),
    ],
)
def test_invalid_counts_are_rejected_before_runtime_use(
    episode_count,
    max_steps,
):
    events = []
    runtime = FakeRuntime(events)
    policy = FakePolicy(events)

    with pytest.raises(
        ValueError,
        match=(
            "episode_count"
            if (
                isinstance(episode_count, bool)
                or episode_count != 1
            )
            else "max_steps"
        ),
    ):
        PolicyMultiEpisodeRunner(
            runtime,
            policy,
        ).run(
            episode_count=episode_count,
            max_steps=max_steps,
        )

    assert events == []
    assert runtime.prepare_calls == 0
    assert policy.reset_calls == 0


def test_multi_episode_runner_stops_immediately_on_failure():
    events = []
    runtime = FakeRuntime(events)
    policy = FakePolicy(events)

    class FailingRunner:
        created = 0

        def __init__(self, runtime_arg, policy_arg):
            self.index = FailingRunner.created
            FailingRunner.created += 1

        def run_episode(self, *, max_steps):
            if self.index == 1:
                raise RuntimeError("episode failed")
            return SimpleNamespace(
                step_count=1,
                total_reward=1.0,
            )

    with pytest.raises(
        RuntimeError,
        match="episode failed",
    ):
        PolicyMultiEpisodeRunner(
            runtime,
            policy,
            episode_runner_factory=FailingRunner,
        ).run(
            episode_count=3,
            max_steps=2,
        )

    assert FailingRunner.created == 2


class _RecoveryTestPolicy:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def _recovery_test_result(value):
    from types import SimpleNamespace

    return SimpleNamespace(
        step_count=1,
        total_reward=float(value),
    )


def _recovery_test_factory(outcomes, attempts):
    class Factory:
        def __call__(
            self,
            runtime,
            policy,
            *,
            transition_sink=None,
        ):
            del runtime, policy
            outcome = outcomes.pop(0)
            attempts.append(transition_sink)

            class Runner:
                def run_episode(self, *, max_steps):
                    assert max_steps == 1
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome

            return Runner()

    return Factory()


def _camera_failure():
    from so101_capstone.environments.simulation_step_coordinator import (
        SimulationStepCoordinatorError,
    )

    return SimulationStepCoordinatorError(
        "Reused camera frames are too old relative to the "
        "completed simulation step."
    )


def test_recoverable_camera_attempt_is_retried_without_counting():
    outcomes = [
        _camera_failure(),
        _recovery_test_result(1),
        _recovery_test_result(2),
    ]
    attempts = []
    sink_indices = []
    policy = _RecoveryTestPolicy()

    result = PolicyMultiEpisodeRunner(
        object(),
        policy,
        episode_runner_factory=_recovery_test_factory(
            outcomes,
            attempts,
        ),
    ).run(
        episode_count=2,
        max_steps=1,
        transition_sink_factory=lambda index: (
            sink_indices.append(index) or RecordingSink(index)
        ),
    )

    assert result.episode_count == 2
    assert result.total_reward == pytest.approx(3.0)
    assert sink_indices == [0, 0, 1]
    assert len(attempts) == 3
    assert attempts[0].reset_calls == 1
    assert policy.reset_calls == 1
    assert outcomes == []


def test_valid_episode_resets_consecutive_failure_counter():
    outcomes = [
        _camera_failure(),
        _recovery_test_result(1),
        _camera_failure(),
        _camera_failure(),
        _recovery_test_result(2),
    ]
    attempts = []
    policy = _RecoveryTestPolicy()

    result = PolicyMultiEpisodeRunner(
        object(),
        policy,
        episode_runner_factory=_recovery_test_factory(
            outcomes,
            attempts,
        ),
    ).run(
        episode_count=2,
        max_steps=1,
    )

    assert result.episode_count == 2
    assert policy.reset_calls == 3
    assert len(attempts) == 5
    assert outcomes == []


def test_third_consecutive_recoverable_failure_stops_run():
    outcomes = [
        _camera_failure(),
        _camera_failure(),
        _camera_failure(),
    ]
    attempts = []
    policy = _RecoveryTestPolicy()

    with pytest.raises(
        RuntimeError,
        match=(
            "stopped after 3 consecutive recoverable "
            "episode failures"
        ),
    ):
        PolicyMultiEpisodeRunner(
            object(),
            policy,
            episode_runner_factory=_recovery_test_factory(
                outcomes,
                attempts,
            ),
        ).run(
            episode_count=1,
            max_steps=1,
        )

    assert policy.reset_calls == 3
    assert len(attempts) == 3


def test_delayed_target_service_failure_is_retried():
    outcomes = [
        RuntimeError(
            "Failed to spawn a random target: service rejected"
        ),
        _recovery_test_result(1),
    ]
    attempts = []
    policy = _RecoveryTestPolicy()

    result = PolicyMultiEpisodeRunner(
        object(),
        policy,
        episode_runner_factory=_recovery_test_factory(
            outcomes,
            attempts,
        ),
    ).run(
        episode_count=1,
        max_steps=1,
    )

    assert result.episode_count == 1
    assert policy.reset_calls == 1
    assert len(attempts) == 2


def test_typed_policy_action_safety_failure_is_retried():
    from so101_capstone.policies.policy_episode_runner import (
        PolicyEpisodeActionSafetyError,
    )

    outcomes = [
        PolicyEpisodeActionSafetyError("invalid action"),
        _recovery_test_result(1),
    ]
    attempts = []
    policy = _RecoveryTestPolicy()

    result = PolicyMultiEpisodeRunner(
        object(),
        policy,
        episode_runner_factory=_recovery_test_factory(
            outcomes,
            attempts,
        ),
    ).run(
        episode_count=1,
        max_steps=1,
    )

    assert result.episode_count == 1
    assert policy.reset_calls == 1
    assert len(attempts) == 2


def test_unexpected_runtime_error_remains_fatal():
    outcomes = [RuntimeError("unexpected model failure")]
    attempts = []
    policy = _RecoveryTestPolicy()

    with pytest.raises(
        RuntimeError,
        match="unexpected model failure",
    ):
        PolicyMultiEpisodeRunner(
            object(),
            policy,
            episode_runner_factory=_recovery_test_factory(
                outcomes,
                attempts,
            ),
        ).run(
            episode_count=1,
            max_steps=1,
        )

    assert policy.reset_calls == 0
    assert len(attempts) == 1


def test_invalid_consecutive_failure_limits_are_rejected():
    policy = _RecoveryTestPolicy()

    for value in (0, -1, True):
        with pytest.raises(
            ValueError,
            match="max_consecutive_episode_failures",
        ):
            PolicyMultiEpisodeRunner(
                object(),
                policy,
            ).run(
                episode_count=1,
                max_steps=1,
                max_consecutive_episode_failures=value,
            )

def test_raw_value_error_from_episode_runner_remains_fatal():
    outcomes = [ValueError("policy internal value error")]
    attempts = []
    policy = _RecoveryTestPolicy()

    with pytest.raises(
        ValueError,
        match="policy internal value error",
    ):
        PolicyMultiEpisodeRunner(
            object(),
            policy,
            episode_runner_factory=_recovery_test_factory(
                outcomes,
                attempts,
            ),
        ).run(
            episode_count=1,
            max_steps=1,
        )

    assert policy.reset_calls == 0
    assert len(attempts) == 1
