"""Tests for deterministic SO-101 HOME episode reset."""

import numpy as np
import pytest

from so101_capstone.environments.episode_reset import (
    EpisodeResetConfig,
    EpisodeResetError,
    EpisodeResetManager,
)
from so101_capstone.environments.simulation_clock import (
    SimulationClockCache,
)
from so101_capstone.environments.simulation_step import (
    SimulationStepSchedule,
)
from so101_capstone.environments.state_cache import (
    JointStateCache,
    MEASURED_JOINT_NAMES,
)


class FakeResetWorld:
    def __init__(
        self,
        *,
        clock_cache,
        joint_cache,
        position_samples,
        velocity_samples,
    ):
        self.clock_cache = clock_cache
        self.joint_cache = joint_cache
        self.position_samples = list(position_samples)
        self.velocity_samples = list(velocity_samples)
        self.sim_time_s = 0.0
        self.pause_calls = 0
        self.resume_calls = 0
        self.step_calls = []

    def pause(self):
        self.pause_calls += 1

    def resume(self):
        self.resume_calls += 1

    def step(self, physics_steps):
        self.step_calls.append(physics_steps)
        self.sim_time_s += physics_steps * 0.001

        sample_index = min(
            len(self.step_calls) - 1,
            len(self.position_samples) - 1,
        )

        self.clock_cache.update(self.sim_time_s)
        self.joint_cache.update(
            MEASURED_JOINT_NAMES,
            self.position_samples[sample_index],
            self.velocity_samples[sample_index],
        )


def build_manager(
    position_samples,
    velocity_samples,
    *,
    stable_sample_count=2,
    max_control_steps=10,
):
    clock_cache = SimulationClockCache()
    joint_cache = JointStateCache()
    schedule = SimulationStepSchedule()

    clock_cache.update(0.0)
    joint_cache.update(
        MEASURED_JOINT_NAMES,
        np.full(6, 0.2),
        np.zeros(6),
    )

    world = FakeResetWorld(
        clock_cache=clock_cache,
        joint_cache=joint_cache,
        position_samples=position_samples,
        velocity_samples=velocity_samples,
    )
    published_commands = []

    manager = EpisodeResetManager(
        world_controller=world,
        schedule=schedule,
        simulation_clock_cache=clock_cache,
        joint_cache=joint_cache,
        publish_joint_targets=lambda command: (
            published_commands.append(
                np.asarray(command, dtype=float)
            )
        ),
        config=EpisodeResetConfig(
            home_joint_positions=(0.0,) * 6,
            position_tolerance_rad=0.01,
            velocity_tolerance_rad_s=0.05,
            stable_sample_count=stable_sample_count,
            max_control_steps=max_control_steps,
            pause_clock_settle_s=0.0,
        ),
    )

    return manager, world, schedule, published_commands


def test_zero_evaluation_override_keeps_nonzero_home_reference_and_publishes_only_zero():
    from dataclasses import replace

    manager, world, schedule, commands = build_manager(
        [np.zeros(6)] * 4, [np.zeros(6)] * 4,
    )
    canonical_home = EpisodeResetConfig().home_joint_positions
    assert np.any(np.asarray(canonical_home) != 0)
    manager.config = replace(manager.config, home_joint_positions=canonical_home)

    result = manager.reset_to_home(home_joint_positions_override=(0.0,) * 6)

    assert manager.config.home_joint_positions == canonical_home
    assert result.control_steps == 4
    assert result.stable_sample_count == 2
    assert len(commands) == 4
    for command in commands:
        np.testing.assert_array_equal(command, np.zeros(6))
    assert world.pause_calls == 1
    assert world.resume_calls == 0
    assert [schedule.next_step_count() for _ in range(3)] == [33, 33, 34]


def test_paused_clock_barrier_accepts_repeated_same_time_messages(
    monkeypatch,
):
    import so101_capstone.environments.episode_reset as module

    clock_cache = SimulationClockCache()
    joint_cache = JointStateCache()

    clock_cache.update(12.5)
    joint_cache.update(
        MEASURED_JOINT_NAMES,
        np.zeros(6),
        np.zeros(6),
    )

    simulated_monotonic_s = 0.0

    def fake_monotonic():
        return simulated_monotonic_s

    def fake_sleep(duration_s):
        nonlocal simulated_monotonic_s
        simulated_monotonic_s += duration_s
        clock_cache.update(12.5)

    monkeypatch.setattr(module, "monotonic", fake_monotonic)
    monkeypatch.setattr(module, "sleep", fake_sleep)

    manager = EpisodeResetManager(
        world_controller=object(),
        schedule=SimulationStepSchedule(),
        simulation_clock_cache=clock_cache,
        joint_cache=joint_cache,
        publish_joint_targets=lambda _targets: None,
        config=EpisodeResetConfig(
            pause_clock_settle_s=0.02,
            feedback_timeout_s=1.0,
        ),
    )

    initial = clock_cache.latest()
    assert initial is not None

    settled = manager._wait_for_paused_clock_barrier()

    assert settled.sim_time_s == pytest.approx(12.5)
    assert settled.sequence > initial.sequence
    assert simulated_monotonic_s >= 0.02


def test_reset_moves_home_and_restarts_schedule():
    manager, world, schedule, published_commands = build_manager(
        position_samples=[
            np.full(6, 0.05),
            np.full(6, 0.005),
            np.full(6, 0.004),
        ],
        velocity_samples=[
            np.full(6, 0.10),
            np.full(6, 0.01),
            np.full(6, 0.01),
        ],
    )

    result = manager.reset_to_home()

    assert world.pause_calls == 1
    assert world.resume_calls == 0
    assert world.step_calls == [33, 33, 34, 33, 33]
    assert result.control_steps == 5
    assert result.physics_steps == 166
    assert result.stable_sample_count == 2
    assert result.max_position_error_rad == pytest.approx(0.004)
    assert result.max_velocity_rad_s == pytest.approx(0.01)

    assert len(published_commands) == 5

    for command in published_commands:
        np.testing.assert_allclose(command, np.zeros(6))

    assert schedule.control_step_count == 0
    assert schedule.physics_step_count == 0
    assert schedule.next_step_count() == 33



def test_reset_moves_to_zero_before_configured_home():
    clock_cache = SimulationClockCache()
    joint_cache = JointStateCache()
    schedule = SimulationStepSchedule()

    clock_cache.update(0.0)
    joint_cache.update(
        MEASURED_JOINT_NAMES,
        np.full(6, 0.2),
        np.zeros(6),
    )

    home = np.asarray(
        [0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
        dtype=float,
    )

    world = FakeResetWorld(
        clock_cache=clock_cache,
        joint_cache=joint_cache,
        position_samples=[
            np.full(6, 0.05),
            np.full(6, 0.005),
            np.full(6, 0.004),
            home + 0.05,
            home + 0.005,
            home + 0.004,
        ],
        velocity_samples=[
            np.full(6, 0.10),
            np.full(6, 0.01),
            np.full(6, 0.01),
            np.full(6, 0.10),
            np.full(6, 0.01),
            np.full(6, 0.01),
        ],
    )
    published_commands = []

    manager = EpisodeResetManager(
        world_controller=world,
        schedule=schedule,
        simulation_clock_cache=clock_cache,
        joint_cache=joint_cache,
        publish_joint_targets=lambda command: (
            published_commands.append(
                np.asarray(command, dtype=float)
            )
        ),
        config=EpisodeResetConfig(
            home_joint_positions=tuple(home),
            position_tolerance_rad=0.01,
            velocity_tolerance_rad_s=0.05,
            stable_sample_count=2,
            max_control_steps=10,
            pause_clock_settle_s=0.0,
        ),
    )

    result = manager.reset_to_home()

    assert world.pause_calls == 1
    assert world.resume_calls == 0
    assert world.step_calls == [33, 33, 34, 33, 33, 34]
    assert result.control_steps == 6
    assert result.physics_steps == 200
    assert result.stable_sample_count == 2
    assert result.max_position_error_rad == pytest.approx(0.004)
    assert result.max_velocity_rad_s == pytest.approx(0.01)

    assert len(published_commands) == 6

    for command in published_commands[:3]:
        np.testing.assert_allclose(command, np.zeros(6))

    for command in published_commands[3:]:
        np.testing.assert_allclose(command, home)

    assert schedule.control_step_count == 0
    assert schedule.physics_step_count == 0
    assert schedule.next_step_count() == 33

def test_failed_reset_resumes_gazebo():
    manager, world, schedule, _ = build_manager(
        position_samples=[
            np.full(6, 0.2),
            np.full(6, 0.2),
        ],
        velocity_samples=[
            np.zeros(6),
            np.zeros(6),
        ],
        stable_sample_count=2,
        max_control_steps=2,
    )

    with pytest.raises(
        EpisodeResetError,
        match="did not stabilize",
    ):
        manager.reset_to_home()

    assert world.pause_calls == 1
    assert world.resume_calls == 1
    assert world.step_calls == [33, 33]
    assert schedule.next_step_count() == 33


def test_reset_requires_joint_velocity_feedback():
    manager, world, schedule, _ = build_manager(
        position_samples=[np.zeros(6)],
        velocity_samples=[()],
        max_control_steps=1,
    )

    with pytest.raises(
        EpisodeResetError,
        match="velocities",
    ):
        manager.reset_to_home()

    assert world.resume_calls == 1
    assert schedule.next_step_count() == 33


@pytest.mark.parametrize(
    "home_joint_positions",
    [
        (0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, float("nan")),
    ],
)
def test_invalid_home_joint_positions_are_rejected(
    home_joint_positions,
):
    with pytest.raises(ValueError):
        EpisodeResetConfig(
            home_joint_positions=home_joint_positions
        )


def _inject_clock_ticks_before_first_reset_step(
    world,
    *,
    tick_count,
):
    original_step = world.step
    injected = False

    def step_with_delayed_clock(physics_steps):
        nonlocal injected

        if not injected:
            injected = True
            world.sim_time_s += tick_count * 0.001
            world.clock_cache.update(world.sim_time_s)

        original_step(physics_steps)

    world.step = step_with_delayed_clock


def test_reset_rebases_one_delayed_clock_tick():
    manager, world, schedule, _ = build_manager(
        position_samples=[
            np.full(6, 0.05),
            np.full(6, 0.005),
            np.full(6, 0.004),
        ],
        velocity_samples=[
            np.full(6, 0.10),
            np.full(6, 0.01),
            np.full(6, 0.01),
        ],
    )
    _inject_clock_ticks_before_first_reset_step(
        world,
        tick_count=1,
    )

    result = manager.reset_to_home()

    assert result.control_steps == 5
    assert result.physics_steps == 166
    assert result.clock_state.sim_time_s == pytest.approx(0.167)
    assert schedule.next_step_count() == 33
    assert world.resume_calls == 0


def test_reset_rejects_two_delayed_clock_ticks():
    manager, world, _, _ = build_manager(
        position_samples=[
            np.full(6, 0.05),
            np.full(6, 0.005),
            np.full(6, 0.004),
        ],
        velocity_samples=[
            np.full(6, 0.10),
            np.full(6, 0.01),
            np.full(6, 0.01),
        ],
    )
    _inject_clock_ticks_before_first_reset_step(
        world,
        tick_count=2,
    )

    with pytest.raises(
        EpisodeResetError,
        match="simulation-time delta",
    ):
        manager.reset_to_home()

    assert world.resume_calls == 1
