import numpy as np
import pytest

from so101_capstone.environments import (
    DeterministicSimulationStepCoordinator,
    JointStateCache,
    RgbImageCache,
    SimulationClockCache,
    SimulationStepCoordinatorConfig,
    SimulationStepCoordinatorError,
    So101ReachEnvironmentCore,
    TargetCubeCollectionCache,
    TargetCubeStateCache,
)
from so101_capstone.environments.simulation_step import (
    SimulationStepSchedule,
)


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def add_joint(cache, *, received_time):
    return cache.update(
        JOINT_NAMES,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        received_monotonic_s=received_time,
    )


def add_cube(
    cache,
    *,
    source_sequence,
    source_time,
    received_time,
):
    return cache.update(
        "target_cube_0001",
        [0.30, 0.0, 0.765],
        [0.0, 0.0, 0.0, 1.0],
        source_sequence=source_sequence,
        source_sim_time_s=source_time,
        received_monotonic_s=received_time,
    )


def add_cube_collection(
    cache,
    *,
    active,
    source_sequence,
    source_time,
    received_time,
):
    return cache.update(
        (
            ("target_cube_0001",)
            if active
            else ()
        ),
        source_sequence=source_sequence,
        source_sim_time_s=source_time,
        received_monotonic_s=received_time,
    )


def add_images(
    top_cache,
    gripper_cache,
    *,
    source_time,
    received_time,
):
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    top_cache.update(
        image,
        source_timestamp_s=source_time - 0.002,
        received_monotonic_s=received_time,
    )
    gripper_cache.update(
        image,
        source_timestamp_s=source_time,
        received_monotonic_s=received_time,
    )


class RecordingTopCameraProcessor:
    def __init__(self):
        self.inputs = []

    def process(self, image):
        self.inputs.append(image.copy())

        output = np.full(
            (1, 2, 3),
            90 + len(self.inputs),
            dtype=np.uint8,
        )
        output.setflags(write=False)
        return output


class FakeWorldStepper:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def step(self, physics_steps):
        self.calls.append(physics_steps)
        self.callback(physics_steps)


def make_system(
    *,
    repeat_cube_source=False,
    advance_clock=True,
    clock_step_multiplier=1.0,
    initial_cube_active=True,
    spawn_cube_on_update=None,
    top_camera_processor=None,
    camera_update_every=1,
    max_camera_reuse_steps=6,
    max_camera_reuse_lag_s=0.18,
):
    joint_cache = JointStateCache()
    cube_collection_cache = (
        TargetCubeCollectionCache()
    )
    cube_cache = TargetCubeStateCache()
    clock_cache = SimulationClockCache()
    top_cache = RgbImageCache(max_history=8)
    gripper_cache = RgbImageCache(max_history=8)

    clock_cache.update(
        1.0,
        received_monotonic_s=10.0,
    )
    add_joint(joint_cache, received_time=10.0)
    add_cube_collection(
        cube_collection_cache,
        active=initial_cube_active,
        source_sequence=10,
        source_time=1.0,
        received_time=10.0,
    )

    if initial_cube_active:
        add_cube(
            cube_cache,
            source_sequence=10,
            source_time=1.0,
            received_time=10.0,
        )

    add_images(
        top_cache,
        gripper_cache,
        source_time=1.0,
        received_time=10.0,
    )

    update_count = 0
    feedback_sim_time = 1.0

    def update_feedback(physics_steps):
        nonlocal update_count
        nonlocal feedback_sim_time

        update_count += 1
        feedback_sim_time += physics_steps * 0.001

        if advance_clock:
            clock_cache.update(
                1.0
                + (
                    feedback_sim_time - 1.0
                )
                * clock_step_multiplier,
                received_monotonic_s=(
                    10.0 + update_count * 0.1
                ),
            )

        add_joint(
            joint_cache,
            received_time=10.0 + update_count * 0.1,
        )
        source_sequence = (
            10
            if repeat_cube_source
            else 10 + update_count
        )
        cube_active = (
            initial_cube_active
            or (
                spawn_cube_on_update is not None
                and update_count
                >= spawn_cube_on_update
            )
        )

        add_cube_collection(
            cube_collection_cache,
            active=cube_active,
            source_sequence=source_sequence,
            source_time=feedback_sim_time,
            received_time=10.0 + update_count * 0.1,
        )

        if cube_active:
            add_cube(
                cube_cache,
                source_sequence=source_sequence,
                source_time=feedback_sim_time,
                received_time=(
                    10.0 + update_count * 0.1
                ),
            )

        if update_count % camera_update_every == 0:
            add_images(
                top_cache,
                gripper_cache,
                source_time=feedback_sim_time,
                received_time=10.0 + update_count * 0.1,
            )

    world = FakeWorldStepper(update_feedback)
    published = []

    coordinator = DeterministicSimulationStepCoordinator(
        environment=So101ReachEnvironmentCore(),
        world_stepper=world,
        schedule=SimulationStepSchedule(),
        simulation_clock_cache=clock_cache,
        joint_cache=joint_cache,
        cube_collection_cache=(
            cube_collection_cache
        ),
        cube_cache=cube_cache,
        top_image_cache=top_cache,
        gripper_image_cache=gripper_cache,
        publish_joint_targets=published.append,
        top_camera_processor=top_camera_processor,
        config=SimulationStepCoordinatorConfig(
            feedback_timeout_s=0.01,
            max_camera_reuse_steps=(
                max_camera_reuse_steps
            ),
            max_camera_reuse_lag_s=(
                max_camera_reuse_lag_s
            ),
        ),
    )

    return coordinator, world, published


def test_reset_builds_complete_policy_observation():
    coordinator, _, _ = make_system()

    observation, info = coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    assert observation.joint_positions.shape == (6,)
    assert observation.joint_velocities.shape == (6,)
    assert observation.top_image.shape == (2, 3, 3)
    assert observation.gripper_image.shape == (2, 3, 3)
    assert info["top_image_sequence"] == 1
    assert info["gripper_image_sequence"] == 1


def test_reset_and_step_process_only_policy_top_image():
    processor = RecordingTopCameraProcessor()
    coordinator, _, _ = make_system(
        top_camera_processor=processor,
    )

    reset_observation, _ = coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    assert len(processor.inputs) == 1
    assert processor.inputs[0].shape == (2, 3, 3)
    assert reset_observation.top_image.shape == (1, 2, 3)
    assert np.all(reset_observation.top_image == 91)
    assert reset_observation.gripper_image.shape == (2, 3, 3)

    result = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )

    assert len(processor.inputs) == 2
    assert result.observation.top_image.shape == (1, 2, 3)
    assert np.all(result.observation.top_image == 92)

    # CoordinatedSimulationStep retains the unprocessed cache frame.
    assert result.camera_frames.top.image.shape == (2, 3, 3)
    assert np.all(result.camera_frames.top.image == 0)

    assert result.observation.gripper_image.shape == (2, 3, 3)
    assert np.all(result.observation.gripper_image == 0)


def test_step_publishes_advances_and_collects_new_feedback():
    coordinator, world, published = make_system()

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    targets = [0.0] * 6
    result = coordinator.step(
        targets,
        now_monotonic_s=10.1,
    )

    assert published == [targets]
    assert world.calls == [33]
    assert result.physics_steps == 33
    assert result.joint_state.sequence == 2
    assert result.cube_state.source_sequence == 11
    assert result.info[
        "target_collection_updated"
    ] is True
    assert result.camera_frames.top.sequence == 2
    assert result.camera_frames.gripper.sequence == 2
    assert result.info["physics_steps"] == 33
    assert result.info["step_start_sim_time_s"] == pytest.approx(
        1.0
    )
    assert result.info["step_end_sim_time_s"] == pytest.approx(
        1.033
    )
    assert result.info["actual_sim_delta_s"] == pytest.approx(
        0.033
    )
    assert result.info["sim_step_error_s"] == pytest.approx(
        0.0
    )
    assert result.info[
        "camera_lag_from_step_end_s"
    ] == pytest.approx(0.002)


def test_three_steps_follow_33_33_34_schedule():
    coordinator, world, _ = make_system()

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    for index in range(3):
        coordinator.step(
            [0.0] * 6,
            now_monotonic_s=10.1 + index * 0.1,
        )

    assert world.calls == [33, 33, 34]
    assert coordinator.schedule.elapsed_sim_time_s == pytest.approx(
        0.1
    )
    assert coordinator.schedule.timing_error_s == pytest.approx(
        0.0
    )


def test_repeated_target_source_is_reused_without_blocking():
    coordinator, _, _ = make_system(
        repeat_cube_source=True,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    result = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )

    assert result.cube_state is not None
    assert result.cube_state.source_sequence == 10
    assert result.info[
        "target_collection_source_sequence"
    ] == 10
    assert result.info[
        "target_collection_updated"
    ] is False


def test_reset_requires_initial_feedback():
    coordinator, _, _ = make_system()
    coordinator.joint_cache = JointStateCache()

    with pytest.raises(
        SimulationStepCoordinatorError,
        match="joint feedback",
    ):
        coordinator.reset_from_latest(
            now_monotonic_s=10.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feedback_timeout_s": 0.0},
        {"max_image_age_s": -1.0},
        {"max_image_skew_s": -1.0},
        {"max_sim_step_error_s": -1.0},
        {
            "max_camera_lag_from_step_end_s": -1.0
        },
        {"max_camera_reuse_steps": -1},
        {"max_camera_reuse_steps": 1.5},
        {"max_camera_reuse_steps": True},
        {"max_camera_reuse_lag_s": -1.0},
        {"max_camera_callback_grace_s": -1.0},
    ],
)
def test_invalid_coordinator_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SimulationStepCoordinatorConfig(**kwargs)



def test_step_requires_simulation_clock_to_advance():
    coordinator, _, _ = make_system(
        advance_clock=False,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    with pytest.raises(
        SimulationStepCoordinatorError,
        match="simulation clock",
    ):
        coordinator.step(
            [0.0] * 6,
            now_monotonic_s=10.1,
        )


def test_step_rejects_incorrect_actual_simulation_delta():
    coordinator, _, _ = make_system(
        clock_step_multiplier=2.0,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    with pytest.raises(
        SimulationStepCoordinatorError,
        match="simulation-time delta",
    ):
        coordinator.step(
            [0.0] * 6,
            now_monotonic_s=10.1,
        )


def test_reset_ignores_stale_cube_when_collection_is_empty():
    coordinator, _, _ = make_system(
        initial_cube_active=False,
    )

    # Simulate detailed feedback left over from an earlier episode.
    add_cube(
        coordinator.cube_cache,
        source_sequence=9,
        source_time=0.9,
        received_time=9.9,
    )

    _observation, info = coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    assert info["target_active"] is False
    assert info["target_cube_name"] is None
    assert info["target_collection_cube_count"] == 0


def test_step_without_cube_returns_state1_stationary_reward():
    coordinator, world, published = make_system(
        initial_cube_active=False,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    result = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )

    assert world.calls == [33]
    assert published == [[0.0] * 6]
    assert result.cube_state is None
    assert result.cube_collection_state.cube_count == 0
    assert result.reward == pytest.approx(
        sum(result.info["reward_components"].values())
    )
    assert result.info["reward_components"]["state1_home_error"] == pytest.approx(
        result.reward
    )
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["target_active"] is False
    assert result.info["target_collection_cube_count"] == 0


def test_delayed_cube_spawn_activates_target():
    coordinator, _, _ = make_system(
        initial_cube_active=False,
        spawn_cube_on_update=2,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    first = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )
    second = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.2,
    )

    assert first.cube_state is None
    assert first.info["target_active"] is False
    assert first.reward == pytest.approx(
        sum(first.info["reward_components"].values())
    )
    assert first.info["reward_components"]["state1_home_error"] == pytest.approx(
        first.reward
    )

    assert second.cube_state is not None
    assert second.cube_collection_state.cube_count == 1
    assert second.info["target_active"] is True
    assert second.info["progress_m"] == pytest.approx(0.0)
    assert second.info["target_cube_name"] == (
        "target_cube_0001"
    )


def test_empty_collection_reuses_unchanged_source():
    coordinator, _, _ = make_system(
        repeat_cube_source=True,
        initial_cube_active=False,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    result = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )

    assert result.cube_state is None
    assert result.cube_collection_state.cube_count == 0
    assert result.info[
        "target_collection_source_sequence"
    ] == 10
    assert result.info[
        "target_collection_updated"
    ] is False


def test_prepare_episode_uses_fresh_post_reset_feedback():
    coordinator, world, published = make_system()

    previous_joint = coordinator.joint_cache.latest()
    previous_target_collection = (
        coordinator.cube_collection_cache.latest()
    )
    previous_top = coordinator.top_image_cache.latest()
    previous_gripper = coordinator.gripper_image_cache.latest()
    reset_clock = coordinator.simulation_clock_cache.latest()

    assert previous_joint is not None
    assert previous_target_collection is not None
    assert (
        previous_target_collection.source_sequence
        is not None
    )
    assert previous_top is not None
    assert previous_gripper is not None
    assert reset_clock is not None

    add_joint(
        coordinator.joint_cache,
        received_time=10.05,
    )
    add_cube_collection(
        coordinator.cube_collection_cache,
        active=True,
        source_sequence=11,
        source_time=1.0,
        received_time=10.05,
    )
    add_cube(
        coordinator.cube_cache,
        source_sequence=11,
        source_time=1.0,
        received_time=10.05,
    )
    add_images(
        coordinator.top_image_cache,
        coordinator.gripper_image_cache,
        source_time=1.0,
        received_time=10.05,
    )

    observation, info = (
        coordinator.prepare_episode_from_latest(
            after_joint_sequence=previous_joint.sequence,
            after_target_collection_sequence=(
                previous_target_collection.sequence
            ),
            after_target_collection_source_sequence=(
                previous_target_collection.source_sequence
            ),
            after_top_sequence=previous_top.sequence,
            after_gripper_sequence=(
                previous_gripper.sequence
            ),
            episode_clock_state=reset_clock,
            now_monotonic_s=10.05,
        )
    )

    assert observation.joint_positions.shape == (6,)
    assert info["episode_prepared"] is True
    assert info["frame_zero"] is True
    assert info["episode_step"] == 0
    assert info["joint_sequence"] == 2
    assert info["target_collection_sequence"] == 2
    assert info["target_collection_source_sequence"] == 11
    assert info["top_image_sequence"] == 2
    assert info["gripper_image_sequence"] == 2
    assert world.calls == []
    assert published == []
    assert coordinator.schedule.next_step_count() == 33


@pytest.mark.parametrize("camera_source_time", [.5, 4.0])
def test_task_profile_frame_zero_rejects_late_callback_with_old_or_future_source(camera_source_time):
    coordinator, _, _ = make_system()
    coordinator.environment.reward_profile = "pick_place_v2"
    add_images(coordinator.top_image_cache, coordinator.gripper_image_cache,
               source_time=camera_source_time, received_time=10.05)
    with pytest.raises(SimulationStepCoordinatorError, match="post-reset camera frames"):
        coordinator.prepare_episode_from_latest(after_top_sequence=1,
            after_gripper_sequence=1, now_monotonic_s=10.05)


def test_prepare_episode_accepts_multiple_target_collection():
    coordinator, _, _ = make_system()

    coordinator.cube_collection_cache.update(
        (
            "target_cube_0001",
            "target_cube_0002",
        ),
        source_sequence=11,
        source_sim_time_s=1.0,
        received_monotonic_s=10.0,
    )
    coordinator.cube_cache.update(
        "target_cube_0002",
        [0.32, 0.0, 0.765],
        [0.0, 0.0, 0.0, 1.0],
        source_sequence=11,
        source_sim_time_s=1.0,
        received_monotonic_s=10.0,
    )

    _observation, info = (
        coordinator.prepare_episode_from_latest(
            now_monotonic_s=10.0,
        )
    )

    assert info["target_collection_cube_count"] == 2
    assert info["target_collection_cube_names"] == (
        "target_cube_0001",
        "target_cube_0002",
    )
    assert info["target_cube_name"] == "target_cube_0002"
    assert info["episode_step"] == 0


@pytest.mark.parametrize(
    "keyword",
    (
        "after_joint_sequence",
        "after_top_sequence",
        "after_gripper_sequence",
    ),
)
def test_prepare_episode_rejects_invalid_sequence_marker(
    keyword,
):
    coordinator, _, _ = make_system()

    with pytest.raises(ValueError):
        coordinator.prepare_episode_from_latest(
            **{keyword: -1},
            now_monotonic_s=10.0,
        )



def test_prepare_episode_requires_target_sequence_marker_pair():
    coordinator, _, _ = make_system()

    with pytest.raises(
        ValueError,
        match="must be provided together",
    ):
        coordinator.prepare_episode_from_latest(
            after_target_collection_sequence=1,
            now_monotonic_s=10.0,
        )

def test_prepare_episode_accepts_post_reset_sequence_after_time_rebase():
    coordinator, _, _ = make_system()

    previous_joint = coordinator.joint_cache.latest()
    previous_target_collection = (
        coordinator.cube_collection_cache.latest()
    )
    previous_top = coordinator.top_image_cache.latest()
    previous_gripper = coordinator.gripper_image_cache.latest()
    reset_clock = coordinator.simulation_clock_cache.latest()

    assert previous_joint is not None
    assert previous_target_collection is not None
    assert (
        previous_target_collection.source_sequence
        is not None
    )
    assert previous_top is not None
    assert previous_gripper is not None
    assert reset_clock is not None

    add_joint(
        coordinator.joint_cache,
        received_time=10.05,
    )
    add_cube_collection(
        coordinator.cube_collection_cache,
        active=True,
        source_sequence=11,
        source_time=0.25,
        received_time=10.05,
    )
    add_cube(
        coordinator.cube_cache,
        source_sequence=11,
        source_time=0.25,
        received_time=10.05,
    )
    add_images(
        coordinator.top_image_cache,
        coordinator.gripper_image_cache,
        source_time=0.25,
        received_time=10.05,
    )

    observation, info = (
        coordinator.prepare_episode_from_latest(
            after_joint_sequence=(
                previous_joint.sequence
            ),
            after_target_collection_sequence=(
                previous_target_collection.sequence
            ),
            after_target_collection_source_sequence=(
                previous_target_collection.source_sequence
            ),
            after_top_sequence=previous_top.sequence,
            after_gripper_sequence=(
                previous_gripper.sequence
            ),
            episode_clock_state=reset_clock,
            now_monotonic_s=10.05,
        )
    )

    assert observation.joint_positions.shape == (6,)
    assert info["episode_prepared"] is True
    assert info["frame_zero"] is True
    assert info["joint_sequence"] == 2
    assert info["top_image_sequence"] == 2
    assert info["gripper_image_sequence"] == 2
    assert info["reset_sim_time_s"] == pytest.approx(
        reset_clock.sim_time_s
    )

    latest_top = coordinator.top_image_cache.latest()
    latest_gripper = (
        coordinator.gripper_image_cache.latest()
    )

    assert latest_top is not None
    assert latest_gripper is not None
    assert (
        latest_top.source_timestamp_s
        < reset_clock.sim_time_s
    )
    assert (
        latest_gripper.source_timestamp_s
        < reset_clock.sim_time_s
    )

def test_camera_pair_reuse_is_bounded_and_reported():
    coordinator, _, _ = make_system(
        camera_update_every=3,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    first = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )
    second = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.2,
    )
    third = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.3,
    )

    assert first.info["camera_pair_reused"] is True
    assert first.info[
        "consecutive_camera_reuse_steps"
    ] == 1
    assert first.camera_frames.top.sequence == 1
    assert first.camera_frames.gripper.sequence == 1

    assert second.info["camera_pair_reused"] is True
    assert second.info[
        "consecutive_camera_reuse_steps"
    ] == 2
    assert second.camera_frames.top.sequence == 1
    assert second.camera_frames.gripper.sequence == 1

    assert third.info["camera_pair_reused"] is False
    assert third.info[
        "consecutive_camera_reuse_steps"
    ] == 0
    assert third.camera_frames.top.sequence == 2
    assert third.camera_frames.gripper.sequence == 2


def test_camera_pair_reuse_limit_is_enforced():
    coordinator, _, _ = make_system(
        camera_update_every=999,
        max_camera_reuse_steps=2,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    first = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )
    second = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.2,
    )

    assert first.info[
        "consecutive_camera_reuse_steps"
    ] == 1
    assert second.info[
        "consecutive_camera_reuse_steps"
    ] == 2

    with pytest.raises(
        SimulationStepCoordinatorError,
        match="reuse limit exceeded",
    ):
        coordinator.step(
            [0.0] * 6,
            now_monotonic_s=10.3,
        )


def test_camera_pair_reuse_lag_limit_is_enforced():
    coordinator, _, _ = make_system(
        camera_update_every=999,
        max_camera_reuse_steps=6,
        max_camera_reuse_lag_s=0.02,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    with pytest.raises(
        SimulationStepCoordinatorError,
        match="Reused camera frames are too old",
    ):
        coordinator.step(
            [0.0] * 6,
            now_monotonic_s=10.1,
        )

def test_camera_pair_reuse_uses_single_callback_grace(
    monkeypatch,
):
    import so101_capstone.environments.simulation_step_coordinator as coordinator_module

    coordinator, _, _ = make_system(
        camera_update_every=999,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    sleep_calls = []

    monkeypatch.setattr(
        coordinator_module,
        "sleep",
        sleep_calls.append,
    )

    result = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )

    assert sleep_calls == pytest.approx([0.015])
    assert result.info["camera_pair_reused"] is True
    assert result.info[
        "consecutive_camera_reuse_steps"
    ] == 1
    assert result.info[
        "max_camera_callback_grace_s"
    ] == pytest.approx(0.015)

def test_delayed_interstep_camera_pair_is_consumed():
    coordinator, _, _ = make_system(
        camera_update_every=999,
    )

    coordinator.reset_from_latest(
        now_monotonic_s=10.0,
    )

    first = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.1,
    )

    assert first.info["camera_pair_reused"] is True
    assert first.camera_frames.top.sequence == 1
    assert first.camera_frames.gripper.sequence == 1

    # Simulate a camera pair generated during the first
    # simulation step but delivered after that policy step
    # already fell back to frame reuse.
    add_images(
        coordinator.top_image_cache,
        coordinator.gripper_image_cache,
        source_time=1.020,
        received_time=10.15,
    )

    second = coordinator.step(
        [0.0] * 6,
        now_monotonic_s=10.2,
    )

    assert second.info["camera_pair_reused"] is False
    assert second.info[
        "consecutive_camera_reuse_steps"
    ] == 0
    assert second.camera_frames.top.sequence == 2
    assert second.camera_frames.gripper.sequence == 2
    assert second.info[
        "camera_lag_from_step_end_s"
    ] == pytest.approx(0.048)


def test_target_callback_between_collection_and_detail_reads_is_retried(monkeypatch):
    from time import monotonic

    coordinator, _, _ = make_system()
    previous = coordinator.cube_collection_cache.latest()
    original_latest = coordinator.cube_cache.latest
    updated = False

    def callback_during_read():
        nonlocal updated
        if not updated:
            updated = True
            add_cube_collection(
                coordinator.cube_collection_cache, active=True,
                source_sequence=11, source_time=1.033, received_time=10.1,
            )
            add_cube(
                coordinator.cube_cache, source_sequence=11,
                source_time=1.033, received_time=10.1,
            )
        return original_latest()

    monkeypatch.setattr(coordinator.cube_cache, "latest", callback_during_read)
    collection, cube = coordinator._target_feedback_for_step(
        previous_collection=previous, deadline_s=monotonic() + 0.1,
    )
    assert collection.source_sequence == cube.source_sequence == 11


def test_permanently_mismatched_target_feedback_still_times_out():
    from time import monotonic

    coordinator, _, _ = make_system()
    previous = coordinator.cube_collection_cache.latest()
    add_cube_collection(
        coordinator.cube_collection_cache, active=True,
        source_sequence=11, source_time=1.033, received_time=10.1,
    )
    with pytest.raises(SimulationStepCoordinatorError, match="detailed target-cube"):
        coordinator._target_feedback_for_step(
            previous_collection=previous, deadline_s=monotonic() + 0.001,
        )


def test_republished_old_target_pose_is_rejected_despite_fresh_callback():
    coordinator, world, _ = make_system()
    coordinator.reset_from_latest(now_monotonic_s=10.0)
    original_callback = world.callback

    def stale_pose(physics_steps):
        original_callback(physics_steps)
        for cache in (coordinator.cube_cache, coordinator.cube_collection_cache):
            if cache is coordinator.cube_cache:
                add_cube(cache, source_sequence=11, source_time=0.7, received_time=10.1)
            else:
                add_cube_collection(
                    cache, active=True, source_sequence=11,
                    source_time=0.7, received_time=10.1,
                )

    world.callback = stale_pose
    with pytest.raises(SimulationStepCoordinatorError, match="pose source is stale"):
        coordinator.step([0.0] * 6, now_monotonic_s=10.1)


def test_paused_learning_wall_time_does_not_age_the_pose_source():
    coordinator, world, _ = make_system()
    coordinator.reset_from_latest(now_monotonic_s=10.0)
    def feedback_after_long_wall_pause(physics_steps):
        # No target callback has run since frame-zero: neither pose nor
        # collection reception timestamps are artificially refreshed.
        coordinator.simulation_clock_cache.update(
            1.0 + physics_steps * 0.001, received_monotonic_s=1010.1
        )
        add_joint(coordinator.joint_cache, received_time=1010.1)
        add_images(
            coordinator.top_image_cache, coordinator.gripper_image_cache,
            source_time=1.033, received_time=1010.1,
        )

    world.callback = feedback_after_long_wall_pause
    result = coordinator.step([0.0] * 6, now_monotonic_s=1010.1)
    assert result.info["cube_source_sim_time_s"] == pytest.approx(1.0)
    assert result.info["step_end_sim_time_s"] == pytest.approx(1.033)
    assert result.cube_state.received_monotonic_s == 10.0
    assert result.info["cube_state_age_s"] == pytest.approx(0.033)
    assert result.info["cube_state_age_clock"] == "simulation"
    assert result.info["cube_received_monotonic_age_s"] == pytest.approx(1000.1)


def test_new_camera_lag_limit_does_not_disable_bounded_reuse():
    from dataclasses import replace

    coordinator, _, _ = make_system(camera_update_every=999)
    coordinator.config = replace(
        coordinator.config, max_camera_lag_from_step_end_s=0.01
    )
    coordinator.reset_from_latest(now_monotonic_s=10.0)
    add_images(
        coordinator.top_image_cache, coordinator.gripper_image_cache,
        source_time=1.010, received_time=10.05,
    )
    result = coordinator.step([0.0] * 6, now_monotonic_s=10.1)
    # The new pair has 25 ms lag and must not count as a fresh pair under
    # a 10 ms limit. The already-consumed pair remains within reuse limits.
    assert result.info["camera_pair_reused"] is True
    assert result.camera_frames.top.sequence == 1
    assert result.info["max_camera_lag_from_step_end_s"] == pytest.approx(0.01)


@pytest.mark.parametrize("start_time", [1.0, 10.0, 1000.0])
@pytest.mark.parametrize("edge_direction", [-1, 0, 1])
def test_new_camera_pair_at_lag_boundary_survives_float_roundoff(
    start_time, edge_direction,
):
    import math

    coordinator, world, _ = make_system(initial_cube_active=False)
    coordinator.simulation_clock_cache.update(
        start_time, received_monotonic_s=10.0,
    )
    add_cube_collection(
        coordinator.cube_collection_cache, active=False,
        source_sequence=10, source_time=start_time, received_time=10.0,
    )
    add_images(
        coordinator.top_image_cache, coordinator.gripper_image_cache,
        source_time=start_time, received_time=10.0,
    )
    coordinator.reset_from_latest(now_monotonic_s=10.0)

    def feedback_at_boundary(physics_steps):
        end_time = start_time + physics_steps * 0.001
        coordinator.simulation_clock_cache.update(
            end_time, received_monotonic_s=10.1,
        )
        add_joint(coordinator.joint_cache, received_time=10.1)
        add_cube_collection(
            coordinator.cube_collection_cache, active=False,
            source_sequence=11, source_time=end_time, received_time=10.1,
        )
        source_time = end_time - 0.05
        if edge_direction:
            source_time = math.nextafter(source_time, edge_direction * math.inf)
        for cache in (coordinator.top_image_cache, coordinator.gripper_image_cache):
            cache.update(
                np.zeros((2, 3, 3), dtype=np.uint8),
                source_timestamp_s=source_time, received_monotonic_s=10.1,
            )

    world.callback = feedback_at_boundary
    result = coordinator.step([0.0] * 6, now_monotonic_s=10.1)
    assert result.info["camera_pair_reused"] is False
    assert result.info["camera_lag_from_step_end_s"] == pytest.approx(0.05)
    assert world.calls == [33]


@pytest.mark.parametrize("excess_s", [1e-8, 0.001])
def test_new_camera_pair_beyond_numerical_tolerance_still_uses_bounded_reuse(
    excess_s,
):
    coordinator, world, _ = make_system(camera_update_every=999)
    coordinator.reset_from_latest(now_monotonic_s=10.0)
    original_callback = world.callback

    def stale_new_pair(physics_steps):
        original_callback(physics_steps)
        end_time = coordinator.simulation_clock_cache.latest().sim_time_s
        for cache in (coordinator.top_image_cache, coordinator.gripper_image_cache):
            cache.update(
                np.zeros((2, 3, 3), dtype=np.uint8),
                source_timestamp_s=end_time - 0.05 - excess_s,
                received_monotonic_s=10.1,
            )

    world.callback = stale_new_pair
    result = coordinator.step([0.0] * 6, now_monotonic_s=10.1)
    assert result.info["camera_pair_reused"] is True
    assert result.camera_frames.top.sequence == 1
    assert result.camera_frames.gripper.sequence == 1


@pytest.mark.parametrize("lag_limit", [0.05, 0.30])
@pytest.mark.parametrize("excess_s", [0.0, 1e-8, 0.001])
def test_reused_camera_lag_boundary_and_real_excess(lag_limit, excess_s):
    coordinator, _, _ = make_system(
        camera_update_every=999, max_camera_reuse_lag_s=lag_limit,
    )
    # The next deterministic step ends at 1.033 s. Both cameras carry the
    # same source time so the boundary is not shifted by the fixture's skew.
    source_time = 1.033 - lag_limit - excess_s
    for cache in (coordinator.top_image_cache, coordinator.gripper_image_cache):
        cache.update(
            np.zeros((2, 3, 3), dtype=np.uint8),
            source_timestamp_s=source_time, received_monotonic_s=10.0,
        )
    coordinator.reset_from_latest(now_monotonic_s=10.0)

    if excess_s:
        with pytest.raises(
            SimulationStepCoordinatorError, match="Reused camera frames are too old",
        ) as caught:
            coordinator.step([0.0] * 6, now_monotonic_s=10.1)
        for diagnostic in ("excess=", "comparison_tolerance=", "step_end=", "top_source=", "gripper_source="):
            assert diagnostic in str(caught.value)
    else:
        result = coordinator.step([0.0] * 6, now_monotonic_s=10.1)
        assert result.info["camera_pair_reused"] is True
        assert result.info["camera_lag_from_step_end_s"] == pytest.approx(lag_limit)
