"""Image augmentation is independent of scene and proprioception changes."""

from dataclasses import dataclass
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments import simulation_runtime as runtime_module
from so101_capstone.environments.environment_randomization import (
    DEFAULT_ENVIRONMENT_RANDOMIZATION_CONFIG,
    DEFAULT_IMAGE_AUGMENTATION_CONFIG,
    EnvironmentRandomizationConfig,
    EnvironmentRandomizationError,
    EpisodeEnvironmentRandomizer,
    EpisodeImageAugmenter,
)
from so101_capstone.environments.policy_observation import PolicyObservation
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerConfig, gazebo_launch_command
from test_simulation_runtime import make_runtime
from test_act_ppo_stage_initialization import make_runtime as make_stage_runtime


def image_config(tmp_path, *, noise=False):
    # Deliberately leave the physical features enabled. The independent
    # pipeline must never even sample these parts of a full-schema config.
    payload = json.loads(DEFAULT_ENVIRONMENT_RANDOMIZATION_CONFIG.read_text())
    for camera in ("top_camera", "gripper_camera"):
        for feature in payload[camera]["image"].values():
            feature["enabled"] = False
        payload[camera]["image"]["brightness"] = {"enabled": True, "gain_range": [1.2, 1.2]}
        payload[camera]["image"]["rgb_noise"] = {"enabled": noise, "std_8bit_range": [3., 3.]}
    path = tmp_path / "image_config.json"
    path.write_text(json.dumps(payload))
    return EnvironmentRandomizationConfig.load(path)


def observation():
    return PolicyObservation(
        joint_positions=np.arange(6, dtype=np.float32),
        joint_velocities=np.arange(6, dtype=np.float32) / 10,
        top_image=np.full((12, 16, 3), 100, dtype=np.uint8),
        gripper_image=np.full((12, 16, 3), 50, dtype=np.uint8),
    )


def test_image_only_does_not_sample_environment_or_change_joints(tmp_path, monkeypatch):
    monkeypatch.setattr(EpisodeEnvironmentRandomizer, "begin_episode",
                        lambda *args, **kwargs: pytest.fail("Full environment sampler was called"))
    augmenter = EpisodeImageAugmenter(image_config(tmp_path), base_seed=7, world_index=2)
    record = augmenter.begin_episode(episode_key=9)
    original = observation()
    changed = augmenter.apply_observation(original)
    np.testing.assert_array_equal(changed.top_image, 120)
    np.testing.assert_array_equal(changed.gripper_image, 60)
    np.testing.assert_array_equal(original.top_image, 100)
    np.testing.assert_array_equal(changed.joint_positions, original.joint_positions)
    np.testing.assert_array_equal(changed.joint_velocities, original.joint_velocities)
    assert changed.joint_positions.dtype == original.joint_positions.dtype
    assert not changed.top_image.flags.writeable
    assert not changed.joint_positions.flags.writeable
    assert record["mode"] == "image_only"
    assert record["episode_key"] == 9
    assert not set(record).intersection({"scene", "lights", "joint_state", "home_initial_pose", "goal_region"})
    assert all(set(camera) == {"image"} for camera in record["cameras"].values())


def test_image_seed_uses_world_and_global_episode_key_and_noise_changes_per_frame(tmp_path):
    config = image_config(tmp_path, noise=True)
    first = EpisodeImageAugmenter(config, base_seed=37, world_index=2)
    restarted = EpisodeImageAugmenter(config, base_seed=37, world_index=2)
    other_world = EpisodeImageAugmenter(config, base_seed=37, world_index=3)
    a, b, c = (item.begin_episode(episode_key=19) for item in (first, restarted, other_world))
    assert a == b
    assert a["derived_seed"] != c["derived_seed"]
    one = first.apply_observation(observation())
    repeated = restarted.apply_observation(observation())
    two = first.apply_observation(observation())
    np.testing.assert_array_equal(one.top_image, repeated.top_image)
    assert not np.array_equal(one.top_image, two.top_image)
    assert first.record() == a
    next_record = first.begin_episode(episode_key=20)
    assert next_record["derived_seed"] != a["derived_seed"]


def test_image_augmentation_requires_an_episode(tmp_path):
    augmenter = EpisodeImageAugmenter(image_config(tmp_path))
    with pytest.raises(EnvironmentRandomizationError, match="begin_episode"):
        augmenter.apply_observation(observation())
    for key in (-1, True, 1.5):
        with pytest.raises(ValueError, match="episode_key"):
            augmenter.begin_episode(episode_key=key)


def test_image_only_distortion_works_without_a_full_environment_sample(tmp_path):
    config = image_config(tmp_path)
    for camera in ("top_camera", "gripper_camera"):
        config.payload[camera]["image"]["distortion"] = {
            "enabled": True, "delta_k1_range": [.04, .04], "delta_k2_range": [.02, .02],
        }
    augmenter = EpisodeImageAugmenter(config)
    augmenter.begin_episode(episode_key=1)
    assert augmenter._transformer.current_sample is None
    for _ in range(2):
        result = augmenter.apply_observation(observation())
        assert result.top_image.shape == observation().top_image.shape
    assert len(augmenter._transformer._geometry_cache) == 2


def test_full_image_only_preset_applies_all_enabled_transforms():
    augmenter = EpisodeImageAugmenter(EnvironmentRandomizationConfig.load(DEFAULT_IMAGE_AUGMENTATION_CONFIG))
    augmenter.begin_episode(episode_key=3)
    original = observation()
    result = augmenter.apply_observation(original)
    assert result.top_image.shape == original.top_image.shape
    assert result.gripper_image.dtype == np.uint8
    assert not np.array_equal(result.top_image, original.top_image)
    np.testing.assert_array_equal(result.joint_positions, original.joint_positions)


def test_image_only_prepare_keeps_existing_physical_reset_path(tmp_path):
    system = make_runtime()
    system.runtime.image_augmenter = EpisodeImageAugmenter(image_config(tmp_path), base_seed=5)
    system.runtime.gazebo_environment_randomization_controller = SimpleNamespace(
        apply_episode=lambda *args: pytest.fail("Scene mutation in image-only mode"),
    )
    assert system.runtime.environment_randomizer is None
    result = system.runtime.prepare_episode(environment_randomization_episode_key=12)
    assert result.info["image_augmentation"]["episode_key"] == 12
    assert "environment_randomization" not in result.info
    assert "environment_randomization_warmup_steps" not in result.info
    assert system.reset_manager.calls == 1
    assert system.cube.spawn_calls == system.cube.delete_all_calls == 0
    first_seed = result.info["image_augmentation"]["derived_seed"]
    second = system.runtime.prepare_episode(environment_randomization_episode_key=13)
    assert second.info["image_augmentation"]["derived_seed"] != first_seed


def test_image_record_persists_through_stage_handoff_and_policy_step(tmp_path):
    system = make_stage_runtime("S1")
    augmenter = EpisodeImageAugmenter(image_config(tmp_path), base_seed=5, world_index=1)
    expected = augmenter.begin_episode(episode_key=27)
    system.runtime.image_augmenter = augmenter
    result = system.runtime.start_training_stage("S1", StageTrainingConfig().to_dict())
    assert result.info["image_augmentation"] == expected
    assert "environment_randomization" not in result.info

    @dataclass
    class Step:
        info: dict

    system.runtime.coordinator.step = lambda *args, **kwargs: Step({"episode_step": 1})
    stepped = system.runtime.step([0.] * 6)
    assert stepped.info["image_augmentation"] == expected
    assert stepped.info["episode_step"] == 1


def test_factory_rejects_both_modes_before_loading_ros(monkeypatch):
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "true")
    monkeypatch.setenv("SO101_IMAGE_AUGMENTATION_ENABLED", "true")
    with pytest.raises(runtime_module.SimulationRuntimeError, match="cannot both"):
        runtime_module.create_ros_simulation_runtime()


def test_image_only_worker_launch_keeps_gazebo_environment_switch_off(tmp_path, monkeypatch):
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")
    monkeypatch.setenv("SO101_IMAGE_AUGMENTATION_ENABLED", "true")
    config = GazeboWorkerConfig(
        world_index=0, ros_domain_id=50, gz_partition="test_images_50", gazebo_gui=False,
        project_root=tmp_path, output_directory=tmp_path, startup_feedback_timeout_s=1.,
    )
    assert "environment_randomization:=false" in gazebo_launch_command(config)


def test_factory_builds_an_image_pipeline_without_a_gazebo_randomizer(tmp_path, monkeypatch):
    config = image_config(tmp_path)
    system = make_runtime()
    system.io_node.goal_region_cache = SimpleNamespace(wait_until_available=lambda timeout: SimpleNamespace(x=.1, y=.2, z=.75))
    system.io_node.target_cube_cache = SimpleNamespace(latest=lambda: None)
    system.io_node.publish_joint_targets = lambda targets: None
    ros = SimpleNamespace(ok=lambda: True)
    executor = SimpleNamespace(add_node=lambda node: None, spin=lambda: None)
    monkeypatch.setitem(sys.modules, "rclpy", ros)
    monkeypatch.setitem(sys.modules, "rclpy.executors", SimpleNamespace(SingleThreadedExecutor=lambda: executor))
    monkeypatch.setitem(sys.modules, "rclpy.node", SimpleNamespace(Node=lambda name: system.control_node))
    monkeypatch.setitem(sys.modules, "so101_capstone.environments.ros_reach_node", SimpleNamespace(So101ReachRosNode=lambda **kwargs: system.io_node))
    monkeypatch.setitem(sys.modules, "so101_capstone.environments.gazebo_world_control", SimpleNamespace(GazeboWorldController=lambda node: system.world))
    monkeypatch.setattr(runtime_module, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(runtime_module, "TargetCubeController", lambda node: system.cube)
    monkeypatch.setattr(runtime_module, "EpisodeResetManager", lambda **kwargs: system.reset_manager)
    monkeypatch.setattr(runtime_module, "TopCameraImageProcessor", lambda config: SimpleNamespace())
    monkeypatch.setattr(runtime_module, "DeterministicSimulationStepCoordinator", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(runtime_module, "GazeboEnvironmentRandomizationController",
                        lambda: pytest.fail("Gazebo randomizer must not be constructed"))
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")
    monkeypatch.setenv("SO101_IMAGE_AUGMENTATION_ENABLED", "true")
    monkeypatch.setenv("SO101_IMAGE_AUGMENTATION_CONFIG", str(config.source_path))
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_SEED", "71")
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_WORLD_INDEX", "4")
    runtime = runtime_module.create_ros_simulation_runtime()
    assert runtime.environment_randomizer is None
    assert runtime.gazebo_environment_randomization_controller is None
    assert runtime.coordinator.observation_randomizer.__self__ is runtime.image_augmenter
    record = runtime.image_augmenter.begin_episode(episode_key=8)
    assert (record["base_seed"], record["world_index"], record["episode_key"]) == (71, 4, 8)
    transformed = runtime.coordinator.observation_randomizer(observation())
    np.testing.assert_array_equal(transformed.top_image, 120)
