from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments import (
    PolicyObservation,
    parse_policy_action,
)
from so101_capstone.policies import (
    InMemoryPolicyTransitionSink,
    PolicyEpisodeRunner,
    PolicyTransition,
)


def observation(value):
    return PolicyObservation(
        joint_positions=np.full(6, value, dtype=float),
        joint_velocities=np.zeros(6, dtype=float),
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
    def __init__(self, command):
        self.command = np.asarray(command, dtype=float)

    def controller_command(self):
        return self.command.copy()


class FakePolicy:
    def __init__(self, commands, events):
        self.commands = list(commands)
        self.events = events
        self.reset_calls = 0
        self.selected_observations = []

    def reset(self):
        self.reset_calls += 1
        self.events.append("policy.reset")

    def select_action(self, current_observation):
        index = len(self.selected_observations)
        self.selected_observations.append(current_observation)
        self.events.append(f"policy.select.{index}")
        return FakePolicyResult(self.commands[index])


class FakeRuntime:
    def __init__(self, initial_observation, steps, events):
        self.initial_observation = initial_observation
        self.steps = list(steps)
        self.events = events
        self.prepare_calls = 0
        self.commands = []

    def prepare_episode(self):
        self.prepare_calls += 1
        self.events.append("runtime.prepare")
        return SimpleNamespace(
            observation=self.initial_observation,
            info={
                "episode_step": 0,
                "mutable": [],
            },
        )

    def step(self, command):
        index = len(self.commands)
        self.commands.append(np.asarray(command, dtype=float))
        self.events.append(f"runtime.step.{index}")
        return self.steps[index]


def runtime_step(
    next_observation,
    reward,
    *,
    terminated=False,
    truncated=False,
    info=None,
):
    return SimpleNamespace(
        observation=next_observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=(
            {"episode_step": 1}
            if info is None
            else info
        ),
    )


def command(value):
    return np.full(6, value, dtype=float)


def test_success_monitor_keeps_act_running_without_reset_and_separates_violations():
    def info(step, **changes):
        return {"episode_step": step, "sim_time_s": float(step), "task_outcome": "success" if step >= 2 else "running",
                "success_event": step == 2, "monitoring_only": step > 2,
                "monitor_continuing": 2 <= step < 4, "robot_at_home": True, "robot_static": True,
                "placed_stable": True, "target_in_goal": True, "target_supported": True,
                "target_static": True, "grasp_valid": False, "home_error_rad": .01, **changes}
    steps = [
        runtime_step(observation(1), -.1, info=info(1)),
        runtime_step(observation(2), 100., info=info(2)),
        runtime_step(observation(3), 0., info=info(3, robot_at_home=False, robot_static=False,
                                                      target_in_goal=False, grasp_valid=True, home_error_rad=.2)),
        runtime_step(observation(4), 0., terminated=True, info=info(4)),
    ]
    runtime = FakeRuntime(observation(0), steps, [])
    policy = FakePolicy([command(.1), command(.2), command(.3), command(.4)], [])
    sink = InMemoryPolicyTransitionSink()
    result = PolicyEpisodeRunner(runtime, policy, transition_sink=sink,
        observe_after_success_s=2., policy_hz=1.).run_episode(max_steps=2)
    assert result.step_count == 4
    assert policy.reset_calls == runtime.prepare_calls == 1
    assert len(policy.selected_observations) == 4
    assert len(sink.transitions) == 4
    np.testing.assert_allclose(runtime.commands[2], command(.3))
    assert result.total_reward == pytest.approx(99.9)
    metrics = result.final_info["evaluation"]
    assert metrics["task_success"]
    assert metrics["first_success_step"] == 2
    assert metrics["first_success_sim_time_s"] == 2.
    assert metrics["monitor_step_count"] == 2
    assert metrics["monitor_home_violation_steps"] == 1
    assert metrics["monitor_object_violation_steps"] == 1
    assert metrics["monitor_regrasp_steps"] == 1
    assert metrics["monitor_max_home_error_rad"] == .2
    assert metrics["monitor_completed"]


def test_monitor_budget_does_not_extend_an_unsuccessful_or_idle_episode():
    runtime = FakeRuntime(observation(0), [
        runtime_step(observation(1), 0., info={"task_mode": "idle_empty", "task_outcome": "running"}),
        runtime_step(observation(2), 0., info={"task_mode": "idle_empty", "task_outcome": "running"}),
    ], [])
    policy = FakePolicy([command(.1), command(.2)], [])
    result = PolicyEpisodeRunner(runtime, policy, observe_after_success_s=30.).run_episode(max_steps=2)
    assert result.step_count == 2
    assert result.reached_max_steps
    assert not result.final_info["evaluation"]["task_success"]
    assert result.final_info["evaluation"]["monitor_step_count"] == 0


@pytest.mark.parametrize("task_mode", ["idle_empty", "idle_completed"])
def test_idle_statistics_measure_the_whole_episode_without_a_success_event(task_mode):
    steps = []
    for index, velocity in enumerate((1., -2.)):
        state = observation(index)
        state = PolicyObservation(joint_positions=state.joint_positions,
            joint_velocities=np.full(6, velocity), top_image=state.top_image, gripper_image=state.gripper_image)
        steps.append(runtime_step(state, -.1, truncated=index == 1, info={
            "task_mode": task_mode, "task_outcome": "running", "success_event": False,
            "robot_at_home": index == 0, "robot_static": index == 0,
            "home_error_rad": .1 * (index + 1), "placed_stable": False,
            "target_in_goal": index == 0, "target_supported": True,
            "target_static": index == 0, "grasp_valid": index == 1,
        }))
    result = PolicyEpisodeRunner(FakeRuntime(observation(0), steps, []),
        FakePolicy([command(.1), command(.2)], []), observe_after_success_s=5.).run_episode(max_steps=2)
    metrics = result.final_info["evaluation"]
    assert metrics["idle_step_count"] == 2
    assert metrics["idle_home_violation_steps"] == 1
    assert metrics["idle_object_violation_steps"] == int(task_mode == "idle_completed")
    assert metrics["idle_regrasp_steps"] == int(task_mode == "idle_completed")
    assert metrics["idle_peak_joint_speed_rad_s"] == 2.
    assert metrics["idle_max_home_error_rad"] == .2
    np.testing.assert_allclose(metrics["idle_joint_velocity_rms_rad_s"], np.sqrt(2.5))
    assert metrics["monitor_step_count"] == 0
    assert not metrics["task_success"]


def test_compact_history_preserves_live_images_and_npz_state(tmp_path):
    from so101_capstone.policies import PolicyEpisodeNpzSink, PolicyEpisodeState, write_policy_episode_npz

    first, second, final = observation(0), observation(1), observation(2)
    runtime = FakeRuntime(first, [runtime_step(second, 1.25), runtime_step(final, 2.5, terminated=True)], [])
    policy = FakePolicy([command(0.1), command(0.2)], [])
    sink = PolicyEpisodeNpzSink(tmp_path / "compact.npz")
    result = PolicyEpisodeRunner(runtime, policy, transition_sink=sink, retain_observation_images=False).run_episode(max_steps=5)
    assert policy.selected_observations == [first, second]
    assert isinstance(result.initial_observation, PolicyEpisodeState)
    assert all(not hasattr(item.observation, "top_image") for item in result.transitions)
    assert result.step_count == 2
    assert result.total_reward == 3.75
    full = PolicyEpisodeRunner(
        FakeRuntime(first, [runtime_step(second, 1.25), runtime_step(final, 2.5, terminated=True)], []),
        FakePolicy([command(0.1), command(0.2)], []),
    ).run_episode(max_steps=5)
    full_path = write_policy_episode_npz(full, tmp_path / "full.npz")
    with np.load(sink.written_path, allow_pickle=False) as archive:
        with np.load(full_path, allow_pickle=False) as full_archive:
            assert archive.files == full_archive.files
            for key in archive.files:
                np.testing.assert_array_equal(archive[key], full_archive[key])
        np.testing.assert_array_equal(archive["observation_joint_positions"], [first.joint_positions, second.joint_positions])
        np.testing.assert_array_equal(archive["next_observation_joint_positions"], [second.joint_positions, final.joint_positions])
        np.testing.assert_array_equal(archive["reward"], [1.25, 2.5])
        assert archive["episode_terminated"].item()


def test_runner_orders_prepare_reset_select_step_and_records_terminal():
    events = []
    first = observation(0)
    second = observation(1)
    final = observation(2)

    runtime = FakeRuntime(
        first,
        [
            runtime_step(
                second,
                1.25,
                info={"episode_step": 1},
            ),
            runtime_step(
                final,
                2.5,
                terminated=True,
                info={"episode_step": 2},
            ),
        ],
        events,
    )
    policy = FakePolicy(
        [command(0.1), command(0.2)],
        events,
    )

    result = PolicyEpisodeRunner(
        runtime,
        policy,
    ).run_episode(max_steps=10)

    assert events == [
        "runtime.prepare",
        "policy.reset",
        "policy.select.0",
        "runtime.step.0",
        "policy.select.1",
        "runtime.step.1",
    ]
    assert runtime.prepare_calls == 1
    assert policy.reset_calls == 1
    assert policy.selected_observations == [first, second]
    assert all(
        item is not final
        for item in policy.selected_observations
    )

    assert result.initial_observation is first
    assert result.final_observation is final
    assert result.step_count == 2
    assert result.terminated is True
    assert result.truncated is False
    assert result.reached_max_steps is False
    assert result.total_reward == pytest.approx(3.75)

    assert all(
        isinstance(item, PolicyTransition)
        for item in result.transitions
    )
    assert result.transitions[0].observation is first
    assert result.transitions[0].next_observation is second
    assert result.transitions[1].observation is second
    assert result.transitions[1].next_observation is final
    assert result.transitions[1].done is True

    np.testing.assert_allclose(
        runtime.commands,
        [command(0.1), command(0.2)],
    )


def test_runner_stops_at_max_steps_without_extra_policy_call():
    events = []
    observations = [
        observation(index)
        for index in range(4)
    ]

    runtime = FakeRuntime(
        observations[0],
        [
            runtime_step(
                observations[index + 1],
                1.0,
                info={"episode_step": index + 1},
            )
            for index in range(3)
        ],
        events,
    )
    policy = FakePolicy(
        [command(0.1), command(0.2), command(0.3)],
        events,
    )

    result = PolicyEpisodeRunner(
        runtime,
        policy,
    ).run_episode(max_steps=3)

    assert len(policy.selected_observations) == 3
    assert result.step_count == 3
    assert result.terminated is False
    assert result.truncated is False
    assert result.reached_max_steps is True
    assert result.final_observation is observations[3]


def test_runner_preserves_environment_truncation():
    events = []
    first = observation(0)
    final = observation(1)

    runtime = FakeRuntime(
        first,
        [
            runtime_step(
                final,
                -0.5,
                truncated=True,
                info={"episode_step": 1},
            )
        ],
        events,
    )
    policy = FakePolicy([command(0.1)], events)

    result = PolicyEpisodeRunner(
        runtime,
        policy,
    ).run_episode(max_steps=5)

    assert result.terminated is False
    assert result.truncated is True
    assert result.reached_max_steps is False
    assert result.step_count == 1


def test_runner_copies_episode_and_transition_info():
    events = []
    first = observation(0)
    final = observation(1)
    step_info = {
        "episode_step": 1,
        "mutable": [],
    }

    runtime = FakeRuntime(
        first,
        [
            runtime_step(
                final,
                1.0,
                terminated=True,
                info=step_info,
            )
        ],
        events,
    )
    policy = FakePolicy([command(0.1)], events)

    result = PolicyEpisodeRunner(
        runtime,
        policy,
    ).run_episode(max_steps=5)

    step_info["new"] = "changed"

    assert "new" not in result.final_info
    assert "new" not in result.transitions[0].info


@pytest.mark.parametrize(
    "max_steps",
    [0, -1, True, 1.5],
)
def test_invalid_max_steps_is_rejected_before_prepare(
    max_steps,
):
    events = []
    runtime = FakeRuntime(
        observation(0),
        [],
        events,
    )
    policy = FakePolicy([], events)

    with pytest.raises(
        ValueError,
        match="max_steps",
    ):
        PolicyEpisodeRunner(
            runtime,
            policy,
        ).run_episode(max_steps=max_steps)

    assert events == []


@pytest.mark.parametrize(
    "invalid_command",
    [
        np.zeros(5),
        np.zeros(7),
        np.array([0, 0, 0, np.nan, 0, 0]),
    ],
)
def test_invalid_policy_command_is_rejected_before_runtime_step(
    invalid_command,
):
    events = []
    runtime = FakeRuntime(
        observation(0),
        [],
        events,
    )
    policy = FakePolicy(
        [invalid_command],
        events,
    )

    with pytest.raises(ValueError):
        PolicyEpisodeRunner(
            runtime,
            policy,
        ).run_episode(max_steps=1)

    assert runtime.commands == []


def test_in_memory_sink_receives_every_completed_transition():
    events = []
    first = observation(0)
    second = observation(1)
    final = observation(2)

    runtime = FakeRuntime(
        first,
        [
            runtime_step(
                second,
                1.0,
                info={"episode_step": 1},
            ),
            runtime_step(
                final,
                2.0,
                terminated=True,
                info={"episode_step": 2},
            ),
        ],
        events,
    )
    policy = FakePolicy(
        [command(0.1), command(0.2)],
        events,
    )
    sink = InMemoryPolicyTransitionSink()

    result = PolicyEpisodeRunner(
        runtime,
        policy,
        transition_sink=sink,
    ).run_episode(max_steps=5)

    assert sink.transitions == result.transitions
    assert len(sink.transitions) == 2
    assert sink.transitions[-1].done is True


def test_in_memory_sink_resets_and_requires_contiguous_steps():
    first = observation(0)
    second = observation(1)

    transition = PolicyTransition(
        step_index=0,
        observation=first,
        action=parse_policy_action(command(0.1)),
        reward=1.0,
        next_observation=second,
        terminated=False,
        truncated=False,
        info={},
    )

    sink = InMemoryPolicyTransitionSink()
    sink.append(transition)
    assert sink.transitions == (transition,)

    sink.reset()
    assert sink.transitions == ()

    invalid = PolicyTransition(
        step_index=1,
        observation=first,
        action=parse_policy_action(command(0.1)),
        reward=1.0,
        next_observation=second,
        terminated=False,
        truncated=False,
        info={},
    )

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        sink.append(invalid)

def test_runner_finalizes_sink_after_final_transition():
    events = []
    first = observation(0)
    final = observation(1)

    class RecordingSink:
        def __init__(self):
            self.transitions = []
            self.finalized_episode = None

        def reset(self):
            events.append("sink.reset")
            self.transitions.clear()

        def append(self, transition):
            events.append(
                f"sink.append.{transition.step_index}"
            )
            self.transitions.append(transition)

        def finalize(self, episode):
            events.append("sink.finalize")
            self.finalized_episode = episode

    runtime = FakeRuntime(
        first,
        [
            runtime_step(
                final,
                1.0,
                terminated=True,
                info={"episode_step": 1},
            )
        ],
        events,
    )
    policy = FakePolicy([command(0.1)], events)
    sink = RecordingSink()

    result = PolicyEpisodeRunner(
        runtime,
        policy,
        transition_sink=sink,
    ).run_episode(max_steps=5)

    assert events == [
        "runtime.prepare",
        "sink.reset",
        "policy.reset",
        "policy.select.0",
        "runtime.step.0",
        "sink.append.0",
        "sink.finalize",
    ]
    assert sink.transitions == list(
        result.transitions
    )
    assert sink.finalized_episode is result
