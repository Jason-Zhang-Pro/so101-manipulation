"""Stage handoff checks with real task evidence and no Gazebo processes."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments.contact_state import FingerContactCache
from so101_capstone.environments.episode_reset import EpisodeResetConfig
from so101_capstone.environments.reach_env_core import (
    ReachEnvironmentConfig,
    ReachEnvironmentStateError,
    So101ReachEnvironmentCore,
)
from so101_capstone.environments.simulation_runtime import (
    SimulationRuntimeConfig,
    SimulationRuntimeError,
    So101SimulationRuntime,
)
from so101_capstone.environments.state_cache import (
    JointStateSnapshot,
    MEASURED_JOINT_NAMES,
    TargetCubeSnapshot,
)
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training import act_ppo_multiworld_worker as worker


class Cache:
    def __init__(self):
        self.values = []

    def latest(self):
        return self.values[-1] if self.values else None

    def history(self):
        return tuple(self.values)


def make_runtime(stage="S1", *, moving_target=False, unstable_home=False,
                 missing_contacts=False, stale_cameras=False, source_bottom=.750,
                 target_period_steps=1, contact_delay_steps=0):
    reward = PickPlaceRewardConfig()
    core = So101ReachEnvironmentCore(
        reward_profile="pick_place_v2", reward_config=reward,
        task_mode="idle_empty" if stage == "S0" else "pick_place",
        goal_region_center_world_m=(.078, -.3, .7515),
        config=ReachEnvironmentConfig(max_episode_steps=1),
    )
    runtime = object.__new__(So101SimulationRuntime)
    runtime._closed = False
    runtime._stage_start_available = True
    runtime.config = SimulationRuntimeConfig()
    runtime.environment_randomizer = None
    io = SimpleNamespace(**{name: Cache() for name in (
        "simulation_clock_cache", "joint_state_cache", "target_cube_cache",
        "target_cube_collection_cache", "top_image_cache", "gripper_image_cache",
    )})
    runtime.io_node = io
    contacts = FingerContactCache()
    holds, prepares, events = [], [], []
    configure = core.configure_training_stage

    def configure_stage(*args):
        events.append("configure_stage")
        configure(*args)

    core.configure_training_stage = configure_stage

    def advance():
        sequence = len(io.joint_state_cache.values) + 1
        t = sequence / 30
        positions = np.asarray(reward.home_joint_positions_rad).copy()
        if unstable_home:
            positions[0] += .1
        joint = JointStateSnapshot(
            dict(zip(MEASURED_JOINT_NAMES, positions)),
            dict.fromkeys(MEASURED_JOINT_NAMES, 0.), sequence, 0., t,
        )
        io.joint_state_cache.values.append(joint)
        io.simulation_clock_cache.values.append(SimpleNamespace(sequence=sequence, sim_time_s=t))
        names = () if stage == "S0" else ("target_cube_1",)
        io.target_cube_collection_cache.values.append(SimpleNamespace(
            sequence=sequence, source_sequence=sequence,
            cube_count=len(names), cube_names=names, source_sim_time_s=t,
        ))
        if names and (sequence == 1 or sequence % target_period_steps == target_period_steps - 1):
            io.target_cube_cache.values.append(TargetCubeSnapshot(
                names[0], np.array([.26 + (.01 * sequence if moving_target else 0), 0., source_bottom + .015]),
                np.array([0., 0., 0., 1.]), sequence, 0., sequence, t,
            ))
        for name in ("top_image_cache", "gripper_image_cache"):
            getattr(io, name).values.append(SimpleNamespace(
                sequence=sequence, source_timestamp_s=0. if stale_cameras else t,
            ))
        if not missing_contacts:
            for side in ("left", "right"):
                contacts.update(side, sim_time_s=max(0., t - contact_delay_steps / 30))

    def hold(targets, *, control_steps):
        np.testing.assert_allclose(targets, reward.home_joint_positions_rad)
        holds.append(control_steps)
        events.append("hold")
        for _ in range(control_steps):
            advance()
        return io.joint_state_cache.latest(), io.simulation_clock_cache.latest(), control_steps * 33

    def prepare(**kwargs):
        events.append("prepare")
        prepares.append(kwargs)
        # Assert that runtime requested post-warmup frames for every stream.
        assert io.joint_state_cache.latest().sequence > kwargs["after_joint_sequence"]
        assert io.top_image_cache.latest().sequence > kwargs["after_top_sequence"]
        assert io.gripper_image_cache.latest().sequence > kwargs["after_gripper_sequence"]
        assert io.target_cube_collection_cache.latest().source_sequence > kwargs["after_target_collection_source_sequence"]
        clock = kwargs["episode_clock_state"]
        stamps = tuple(getattr(io, name).latest().source_timestamp_s
                       for name in ("top_image_cache", "gripper_image_cache"))
        observation, info = core.reset_from_feedback(
            io.joint_state_cache.latest(), io.target_cube_cache.latest(),
            now_monotonic_s=0., step_end_sim_time_s=clock.sim_time_s,
            joint_history=io.joint_state_cache.history(),
            cube_history=io.target_cube_cache.history(), contact_cache=contacts,
            camera_source_times_s=stamps,
            command_joint_targets_rad=kwargs["reset_command_joint_targets_rad"],
        )
        collection = io.target_cube_collection_cache.latest()
        info.update(frame_zero=True, target_collection_cube_count=collection.cube_count)
        return observation, info

    advance()
    runtime.reset_manager = SimpleNamespace(
        config=replace(EpisodeResetConfig(), max_control_steps=6),
        hold_targets_for_steps=hold,
    )
    runtime.coordinator = SimpleNamespace(
        environment=core, prepare_episode_from_latest=prepare,
        step=lambda *args, **kwargs: pytest.fail("Initialization must not produce PPO transitions"),
    )
    return SimpleNamespace(runtime=runtime, core=core, holds=holds, prepares=prepares,
                           events=events, io=io)


@pytest.mark.parametrize("stage", ["S0", "S1"])
def test_stage_handoff_warms_sensors_and_starts_fresh_zero_budget(stage):
    system = make_runtime(stage)
    initial_time = system.io.simulation_clock_cache.latest().sim_time_s
    result = system.runtime.start_training_stage(stage, StageTrainingConfig().to_dict())

    assert system.holds[0] == 30
    assert system.events[-2:] == ["configure_stage", "prepare"]
    assert result.info["episode_step"] == 0
    assert result.info["stage_elapsed_s"] == 0
    assert result.info["stage_result"] == "RUNNING"
    assert result.info["reward_components"] == {}
    assert result.info["home_stable_duration_s"] == 0
    certificate = result.info["stage_initialization"]
    assert certificate["valid"] is True
    assert certificate["training_stage"] == stage
    assert certificate["discarded_control_steps"] == sum(system.holds)
    assert certificate["initialization_start_sim_time_s"] == initial_time
    assert certificate["start_sim_time_s"] > initial_time
    assert certificate["home_stable_samples"] == 5
    assert min(certificate["camera_source_times_s"]) > initial_time
    assert result.home_reset.control_steps == sum(system.holds)
    if stage == "S1":
        # This uses the real measurement builder. Its history prewarm must
        # establish measured zero velocity instead of inventing staticity.
        assert result.info["target_motion_valid"] is True
        assert result.info["target_static"] is True
        assert result.info["start_certificate"]["start_valid"] is True
    with pytest.raises(SimulationRuntimeError, match="unused prepared episode"):
        system.runtime.start_training_stage(stage, StageTrainingConfig().to_dict())


@pytest.mark.parametrize("kwargs, message", [
    ({"moving_target": True}, "target_static"),
    ({"source_bottom": .8}, "target_source_supported"),
    ({"missing_contacts": True}, "invalid stage sensor evidence"),
    ({"stale_cameras": True}, "cameras must be newer"),
    ({"unstable_home": True}, "HOME did not stabilize"),
])
def test_invalid_stage_preparation_never_returns_a_training_observation(kwargs, message):
    system = make_runtime(**kwargs)
    with pytest.raises((SimulationRuntimeError, ReachEnvironmentStateError), match=message):
        system.runtime.start_training_stage("S1", StageTrainingConfig().to_dict())
    assert system.runtime._stage_start_available is False
    with pytest.raises(SimulationRuntimeError, match="unused prepared episode"):
        system.runtime.start_training_stage("S1", StageTrainingConfig().to_dict())


def test_stage_cannot_start_with_wrong_mode_or_without_preparation():
    system = make_runtime("S0")
    with pytest.raises(ValueError, match="mode disagree"):
        system.runtime.start_training_stage("S1", {})
    with pytest.raises(ValueError, match="Training stage must"):
        system.runtime.start_training_stage("S6", {})
    system.runtime._stage_start_available = False
    with pytest.raises(SimulationRuntimeError, match="unused prepared episode"):
        system.runtime.start_training_stage("S0", {})


def test_stage_validation_uses_source_support_not_goal_support():
    system = make_runtime()
    result = system.runtime.start_training_stage("S1", StageTrainingConfig().to_dict())
    valid = dict(result.info, target_supported=False)
    system.runtime._validate_stage_start_info("S1", valid, 0.)
    invalid = dict(valid, target_bottom_z_m=.753, source_support_z_m=.700, target_supported=True)
    with pytest.raises(SimulationRuntimeError, match="supported at the source"):
        system.runtime._validate_stage_start_info("S1", invalid, 0.)


def test_stage_handoff_accepts_a_fresh_10hz_target_with_camera_and_joint_lag():
    system = make_runtime(target_period_steps=3)
    result = system.runtime.start_training_stage("S1", StageTrainingConfig().to_dict())
    info = result.info
    assert info["target_motion_valid"] is True
    assert info["sim_time_s"] - info["target_source_sim_time_s"] == pytest.approx(2 / 30)
    assert info["stage_elapsed_s"] == 0


def test_stage_handoff_retains_motion_pair_before_delayed_contact_evidence():
    system = make_runtime(target_period_steps=2, contact_delay_steps=2)
    result = system.runtime.start_training_stage("S1", StageTrainingConfig().to_dict())
    info = result.info
    assert info["target_motion_valid"] is True
    assert info["contact_measurement_valid"] is True
    assert info["sim_time_s"] - info["evidence_sim_time_s"] == pytest.approx(.1)
    assert info["home_stable_duration_s"] == 0


def test_stage_handoff_checks_latest_joints_as_well_as_aligned_evidence():
    system = make_runtime()
    prepare = system.runtime.coordinator.prepare_episode_from_latest

    def unstable_latest(**kwargs):
        observation, info = prepare(**kwargs)
        assert info["robot_at_home"] and info["robot_static"]
        return replace(observation, joint_positions=observation.joint_positions + .1), info

    system.runtime.coordinator.prepare_episode_from_latest = unstable_latest
    with pytest.raises(SimulationRuntimeError, match="Latest stage handoff HOME"):
        system.runtime.start_training_stage("S1", StageTrainingConfig().to_dict())


@pytest.mark.parametrize("stage,count,present", [("S0", 1, True), ("S1", 0, False), ("S1", 2, True)])
def test_stage_validation_rejects_target_collection_mismatch(stage, count, present):
    system = make_runtime(stage)
    result = system.runtime.start_training_stage(stage, StageTrainingConfig().to_dict())
    invalid = dict(result.info, target_collection_cube_count=count, physical_target_present=present)
    with pytest.raises(SimulationRuntimeError, match="target scene|physical target"):
        system.runtime._validate_stage_start_info(stage, invalid, 0.)


def test_worker_and_client_forward_stage_handoff_without_a_policy_step():
    client = object.__new__(worker.GazeboWorkerClient)
    requests = []
    client.request_async = lambda command, **kwargs: requests.append((command, kwargs))
    payload = StageTrainingConfig().to_dict()
    client.start_training_stage_async("S1", payload)
    assert requests == [("start_training_stage", {"training_stage": "S1", "stage_config": payload})]

    system = make_runtime()
    host = object.__new__(worker._WorkerRuntimeHost)
    host.config = SimpleNamespace(evaluation_mode=False)
    host.runtime = system.runtime
    host.publish_observation = lambda observation: observation
    reply = host.start_training_stage(training_stage="S1", stage_config=payload)
    assert reply["type"] == "stage_started"
    assert reply["info"]["episode_step"] == 0
    assert reply["initialization"] == reply["info"]["stage_initialization"]


def test_worker_dispatches_start_stage_command(tmp_path, monkeypatch):
    config = worker.GazeboWorkerConfig(
        world_index=0, ros_domain_id=50, gz_partition="test_stage_50", gazebo_gui=False,
        project_root=tmp_path, output_directory=tmp_path, startup_feedback_timeout_s=1.,
    )
    calls, replies = [], []

    def start_stage(**kwargs):
        calls.append(kwargs)
        return {"type": "stage_started"}

    host = SimpleNamespace(
        start_world=lambda **kwargs: {"type": "ready"},
        start_training_stage=start_stage, stop_world=lambda: None,
    )
    monkeypatch.setattr(worker, "_WorkerRuntimeHost", lambda *args, **kwargs: host)
    monkeypatch.setattr(worker, "_runtime_error_types", lambda: ())
    for name in ("ROS_DOMAIN_ID", "GZ_PARTITION", "IGN_PARTITION"):
        monkeypatch.setenv(name, "test")
    requests = iter([
        {"command": "start_training_stage", "training_stage": "S0", "stage_config": {}},
        {"command": "shutdown"},
    ])
    connection = SimpleNamespace(recv=lambda: next(requests), send=replies.append, close=lambda: None)
    worker._worker_main(connection, config.payload(), None, None)
    assert calls == [{"training_stage": "S0", "stage_config": {}}]
    assert [reply["type"] for reply in replies] == ["ready", "stage_started", "shutdown_complete"]
