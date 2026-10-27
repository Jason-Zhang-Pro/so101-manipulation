from types import SimpleNamespace

import pytest
import numpy as np
from dataclasses import replace

import so101_capstone.environments.simulation_runtime as module
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.environments.simulation_runtime import (
    PreparedSimulationEpisode,
    SimulationRuntimeConfig,
    SimulationRuntimeError,
    SimulationRuntimeReset,
    So101SimulationRuntime,
)


class FakeCache:
    def __init__(self, value=None):
        self.value = value

    def latest(self):
        return self.value


def test_task_home_jitter_is_seeded_arm_only_and_does_not_move_scoring_home():
    cfg = PickPlaceRewardConfig()
    first, second = make_runtime(), make_runtime()
    for system in (first, second):
        system.coordinator.environment = SimpleNamespace(reward_profile="pick_place_v2",
            reward_config=cfg, configure_task=lambda *args, **kwargs: None)
        system.runtime.configure_task("idle_empty", seed=14)
    a = first.runtime._task_reset_targets(cfg.home_joint_positions_rad)
    b = second.runtime._task_reset_targets(cfg.home_joint_positions_rad)
    assert a == b
    assert a[-1] == cfg.home_joint_positions_rad[-1]
    assert 0 < np.max(np.abs(np.array(a[:5])-cfg.home_joint_positions_rad[:5])) <= cfg.home_reset_jitter_rad
    assert first.coordinator.environment.reward_config.home_joint_positions_rad == cfg.home_joint_positions_rad


def test_task_home_outside_physical_limits_and_wrong_rate_rejected_before_ros_startup():
    cfg = PickPlaceRewardConfig()
    home = list(cfg.home_joint_positions_rad)
    home[0] = 10
    with pytest.raises(ValueError, match="physical joint limits"):
        module.create_ros_simulation_runtime(reward_profile="pick_place_v2",
            reward_config=replace(cfg, home_joint_positions_rad=tuple(home)))
    with pytest.raises(ValueError, match="30 Hz"):
        module.create_ros_simulation_runtime(reward_profile="pick_place_v2",
            reward_config=replace(cfg, control_hz=20))


class FakeNode:
    def __init__(self, *, ready=True):
        value = object() if ready else None

        self.simulation_clock_cache = FakeCache(value)
        self.joint_state_cache = FakeCache(
            SimpleNamespace(sequence=11)
            if ready
            else None
        )
        self.target_cube_collection_cache = FakeCache(
            SimpleNamespace(
                sequence=16,
                source_sequence=26,
            )
            if ready
            else None
        )
        self.top_image_cache = FakeCache(
            SimpleNamespace(sequence=21)
            if ready
            else None
        )
        self.gripper_image_cache = FakeCache(
            SimpleNamespace(sequence=31)
            if ready
            else None
        )
        self.destroy_calls = 0

    def destroy_node(self):
        self.destroy_calls += 1


class FakeWorldController:
    def __init__(self):
        self.resume_calls = 0

    def resume(self):
        self.resume_calls += 1


class FakeCubeController:
    def __init__(self):
        self.spawn_calls = 0
        self.delete_all_calls = 0

    def spawn_random(self):
        self.spawn_calls += 1
        return "spawned"

    def delete_all(self):
        self.delete_all_calls += 1
        return "deleted-all"


class FakeResetManager:
    def __init__(self):
        self.calls = 0
        self.clock_state = SimpleNamespace(
            sim_time_s=4.2,
            sequence=41,
        )

    def reset_to_home(self):
        self.calls += 1
        return SimpleNamespace(
            clock_state=self.clock_state,
        )


class FakeCoordinator:
    def __init__(self):
        self.prepare_calls = []
        self.step_calls = []

    def prepare_episode_from_latest(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return "observation", {"episode_step": 0}

    def step(
        self,
        joint_targets,
        *,
        now_monotonic_s=None,
    ):
        self.step_calls.append(
            (list(joint_targets), now_monotonic_s)
        )
        return "step-result"


class FakeExecutor:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, *, timeout_sec):
        self.shutdown_calls.append(timeout_sec)


class FakeThread:
    def __init__(self):
        self.join_calls = []
        self.alive = False

    def join(self, *, timeout):
        self.join_calls.append(timeout)

    def is_alive(self):
        return self.alive


def make_runtime(
    *,
    ready=True,
    config=None,
):
    io_node = FakeNode(ready=ready)
    control_node = FakeNode()
    world = FakeWorldController()
    cube = FakeCubeController()
    reset_manager = FakeResetManager()
    coordinator = FakeCoordinator()
    executor = FakeExecutor()
    thread = FakeThread()
    shutdown_calls = []

    runtime = So101SimulationRuntime(
        io_node=io_node,
        control_node=control_node,
        executor=executor,
        executor_thread=thread,
        world_controller=world,
        cube_controller=cube,
        reset_manager=reset_manager,
        coordinator=coordinator,
        goal_region_snapshot=SimpleNamespace(
            frame_id="world",
            x=0.078,
            y=-0.300,
            z=0.7515,
        ),
        config=config,
        shutdown_ros=lambda: shutdown_calls.append(True),
    )

    return SimpleNamespace(
        runtime=runtime,
        io_node=io_node,
        control_node=control_node,
        world=world,
        cube=cube,
        reset_manager=reset_manager,
        coordinator=coordinator,
        executor=executor,
        thread=thread,
        shutdown_calls=shutdown_calls,
    )


def test_prepare_episode_builds_fresh_frame_zero_without_target_action():
    system = make_runtime()
    system.runtime._wait_for_evaluation_controller = lambda: pytest.fail("Default training reset must not use the evaluation gate")

    result = system.runtime.prepare_episode(
        now_monotonic_s=12.5
    )

    assert isinstance(result, PreparedSimulationEpisode)
    assert result.home_reset.clock_state.sim_time_s == pytest.approx(
        4.2
    )
    assert result.observation == "observation"
    assert result.info == {"episode_step": 0}
    assert system.reset_manager.calls == 1
    assert system.coordinator.prepare_calls == [
        {
            "after_joint_sequence": 11,
            "after_target_collection_sequence": 16,
            "after_target_collection_source_sequence": 26,
            "after_top_sequence": 21,
            "after_gripper_sequence": 31,
            "episode_clock_state": (
                system.reset_manager.clock_state
            ),
            "now_monotonic_s": 12.5,
        }
    ]
    assert system.cube.spawn_calls == 0
    assert system.cube.delete_all_calls == 0


def make_zero_evaluation_runtime():
    system = make_runtime()
    reward = PickPlaceRewardConfig()
    system.coordinator.environment = SimpleNamespace(
        reward_profile="pick_place_v2", task_mode="pick_place", reward_config=reward,
    )
    calls = []
    events = []

    def controller_ready():
        events.append("controller_ready")
        return {"ready": True, "controller_name": "position_controller", "controller_state": "active",
                "command_subscription_count": 1, "waited_wall_s": 2.5}

    system.runtime._wait_for_evaluation_controller = controller_ready
    system.preparation_events = events

    def reset_to_home(**kwargs):
        events.append("zero_reset")
        calls.append(kwargs)
        return SimpleNamespace(clock_state=system.reset_manager.clock_state)

    system.reset_manager.reset_to_home = reset_to_home
    system.reset_calls = calls
    return system


def test_zero_evaluation_preparation_never_jitters_or_moves_to_home(monkeypatch):
    system = make_zero_evaluation_runtime()
    monkeypatch.setattr(system.runtime, "_task_reset_targets",
                        lambda *args: pytest.fail("ZERO initialization must not jitter HOME"))
    result = system.runtime.prepare_episode(evaluation_initial_pose="zero", now_monotonic_s=12.5)
    zeros = (0.0,) * 6
    assert system.reset_calls == [{"home_joint_positions_override": zeros}]
    assert system.coordinator.prepare_calls == [{
        "after_joint_sequence": 11, "after_target_collection_sequence": 16,
        "after_target_collection_source_sequence": 26, "after_top_sequence": 21,
        "after_gripper_sequence": 31, "episode_clock_state": system.reset_manager.clock_state,
        "now_monotonic_s": 12.5, "reset_command_joint_targets_rad": zeros,
    }]
    assert result.info["evaluation_initial_pose"] == "zero"
    assert result.info["reset_pose_name"] == "ZERO"
    assert result.info["home_initialization_assisted"] is False
    assert result.info["evaluation_controller_readiness"]["ready"] is True
    assert result.info["evaluation_controller_readiness"]["waited_wall_s"] == 2.5
    assert system.preparation_events == ["controller_ready", "zero_reset"]
    assert result.info["reset_joint_targets_rad"] == list(zeros)
    np.testing.assert_allclose(result.info["reset_home_offset_rad"],
                               -np.asarray(system.coordinator.environment.reward_config.home_joint_positions_rad))
    assert system.runtime._stage_start_available is False
    assert system.cube.spawn_calls == system.cube.delete_all_calls == 0


def test_zero_evaluation_waits_before_reset_and_initial_observation():
    system = make_zero_evaluation_runtime()
    prepare_frame = system.coordinator.prepare_episode_from_latest

    def frame_zero(**kwargs):
        assert system.preparation_events == ["controller_ready", "zero_reset"]
        system.preparation_events.append("frame_zero")
        return prepare_frame(**kwargs)

    system.coordinator.prepare_episode_from_latest = frame_zero
    system.runtime.prepare_episode(evaluation_initial_pose="zero")
    assert system.preparation_events == ["controller_ready", "zero_reset", "frame_zero"]


def test_zero_evaluation_controller_timeout_never_resets_or_prepares_frame_zero():
    system = make_zero_evaluation_runtime()

    def timeout():
        raise SimulationRuntimeError("Timed out waiting for active position_controller")

    system.runtime._wait_for_evaluation_controller = timeout
    with pytest.raises(SimulationRuntimeError, match="active position_controller"):
        system.runtime.prepare_episode(evaluation_initial_pose="zero")
    assert system.reset_calls == []
    assert system.coordinator.prepare_calls == []
    assert system.runtime._stage_start_available is False
    assert system.world.resume_calls == 0


def test_evaluation_gate_uses_separate_control_node_and_the_command_publisher(monkeypatch):
    from so101_capstone.environments import controller_readiness

    system = make_runtime(config=SimulationRuntimeConfig(startup_feedback_timeout_s=7.5))
    system.io_node.joint_target_subscription_count = lambda: 1
    checked = {"ready": True, "controller_state": "active"}

    def wait(node, *, command_subscription_count, timeout_s):
        assert node is system.control_node
        assert command_subscription_count() == 1
        assert timeout_s == 7.5
        assert system.reset_manager.calls == 0
        return checked

    monkeypatch.setattr(controller_readiness, "wait_for_position_controller", wait)
    assert system.runtime._wait_for_evaluation_controller() is checked


def test_evaluation_gate_converts_service_failure_to_runtime_error(monkeypatch):
    from so101_capstone.environments import controller_readiness

    system = make_runtime()
    system.io_node.joint_target_subscription_count = lambda: 0

    def fail(*args, **kwargs):
        raise controller_readiness.ControllerReadinessError(
            "Controller query timed out", last_status={"controller_state": "inactive"})

    monkeypatch.setattr(controller_readiness, "wait_for_position_controller", fail)
    with pytest.raises(SimulationRuntimeError, match="Controller query timed out.*inactive") as caught:
        system.runtime._wait_for_evaluation_controller()
    assert isinstance(caught.value.__cause__, controller_readiness.ControllerReadinessError)


@pytest.mark.parametrize("invalid_pose", ["home", "spawn", "ZERO", 0, False])
def test_evaluation_rejects_unknown_initial_pose_before_reset(invalid_pose):
    system = make_zero_evaluation_runtime()
    with pytest.raises(ValueError, match="evaluation_initial_pose"):
        system.runtime.prepare_episode(evaluation_initial_pose=invalid_pose)
    assert system.reset_calls == []


@pytest.mark.parametrize("invalid_setting", ["legacy", "idle_empty", "idle_completed", "physical", "images"])
def test_zero_evaluation_rejects_incompatible_modes_before_reset(invalid_setting):
    system = make_zero_evaluation_runtime()
    if invalid_setting == "legacy":
        system.coordinator.environment.reward_profile = "legacy"
    elif invalid_setting in {"idle_empty", "idle_completed"}:
        system.coordinator.environment.task_mode = invalid_setting
    elif invalid_setting == "physical":
        system.runtime.environment_randomizer = object()
    else:
        system.runtime.image_augmenter = object()
    with pytest.raises(ValueError, match="ZERO evaluation"):
        system.runtime.prepare_episode(evaluation_initial_pose="zero")
    assert system.reset_calls == []


def test_evaluation_measurements_export_aligned_snapshot_without_replacing_observation():
    from so101_capstone.rewards.task_state import TaskMeasurements

    system = make_zero_evaluation_runtime()
    measurements = TaskMeasurements(
        sim_time_s=4.2, evidence_sim_time_s=4.1,
        joint_positions_rad=(0.,) * 6, joint_velocities_rad_s=(0.,) * 6,
        grasp_position_world_m=(.2, 0., .8),
        grasp_rotation_world=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
    )
    system.coordinator.environment._task_snapshot = SimpleNamespace(measurements=measurements)
    exported = system.runtime.evaluation_measurements()
    assert exported["sim_time_s"] == 4.2
    assert exported["evidence_sim_time_s"] == 4.1
    assert exported["grasp_position_world_m"] == (.2, 0., .8)
    exported["joint_positions_rad"] = (99.,) * 6
    assert measurements.joint_positions_rad == (0.,) * 6
    system.coordinator.environment._task_snapshot = None
    with pytest.raises(SimulationRuntimeError, match="prepared task snapshot"):
        system.runtime.evaluation_measurements()


def test_reset_is_prepare_episode_compatibility_alias():
    system = make_runtime()

    result = system.runtime.reset(
        now_monotonic_s=13.0
    )

    assert isinstance(result, SimulationRuntimeReset)
    assert system.reset_manager.calls == 1
    assert system.coordinator.prepare_calls[0][
        "now_monotonic_s"
    ] == 13.0


def test_step_and_target_operations_delegate_independently():
    system = make_runtime()

    step_result = system.runtime.step(
        [0.0] * 6,
        now_monotonic_s=13.0,
    )
    spawn_result = (
        system.runtime.spawn_random_target()
    )
    delete_result = (
        system.runtime.delete_all_targets()
    )

    assert step_result == "step-result"
    assert system.coordinator.step_calls == [
        ([0.0] * 6, 13.0)
    ]
    assert spawn_result == "spawned"
    assert delete_result == "deleted-all"
    assert system.cube.spawn_calls == 1
    assert system.cube.delete_all_calls == 1


def test_wait_until_ready_reports_missing_feedback(
    monkeypatch,
):
    system = make_runtime(
        ready=False,
        config=SimulationRuntimeConfig(
            startup_feedback_timeout_s=0.03,
            feedback_poll_period_s=0.01,
        ),
    )

    current_time = 0.0

    def fake_monotonic():
        return current_time

    def fake_sleep(duration):
        nonlocal current_time
        current_time += duration

    monkeypatch.setattr(
        module,
        "monotonic",
        fake_monotonic,
    )
    monkeypatch.setattr(
        module,
        "sleep",
        fake_sleep,
    )

    with pytest.raises(
        SimulationRuntimeError,
        match="simulation clock",
    ):
        system.runtime.wait_until_ready()


def test_close_releases_every_owned_resource():
    system = make_runtime()

    system.runtime.close()

    assert system.runtime.closed
    assert system.world.resume_calls == 1
    assert system.executor.shutdown_calls == [5.0]
    assert system.thread.join_calls == [5.0]
    assert system.io_node.destroy_calls == 1
    assert system.control_node.destroy_calls == 1
    assert system.shutdown_calls == [True]

    system.runtime.close()

    assert system.world.resume_calls == 1
    assert system.io_node.destroy_calls == 1


def test_close_can_leave_world_paused():
    system = make_runtime(
        config=SimulationRuntimeConfig(
            resume_world_on_close=False
        )
    )

    system.runtime.close()

    assert system.world.resume_calls == 0


def test_closed_runtime_rejects_operations():
    system = make_runtime()
    system.runtime.close()

    with pytest.raises(
        SimulationRuntimeError,
        match="already closed",
    ):
        system.runtime.prepare_episode()

    with pytest.raises(
        SimulationRuntimeError,
        match="already closed",
    ):
        system.runtime.step([0.0] * 6)

    with pytest.raises(
        SimulationRuntimeError,
        match="already closed",
    ):
        system.runtime.spawn_random_target()

    with pytest.raises(
        SimulationRuntimeError,
        match="already closed",
    ):
        system.runtime.delete_all_targets()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"startup_feedback_timeout_s": 0.0},
        {"feedback_poll_period_s": -1.0},
        {"shutdown_timeout_s": float("nan")},
        {"resume_world_on_close": 1},
    ],
)
def test_invalid_runtime_configuration_is_rejected(
    kwargs,
):
    with pytest.raises(ValueError):
        SimulationRuntimeConfig(**kwargs)
