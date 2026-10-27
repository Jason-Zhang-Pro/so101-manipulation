import copy
import json
import math
from pathlib import Path

import numpy as np

from so101_capstone.environments.environment_randomization import (
    EnvironmentRandomizationConfig,
    EpisodeEnvironmentRandomizer,
)
from so101_capstone.environments.policy_observation import PolicyObservation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "environment_randomization.json"
HOME = np.asarray(
    (
        0.0,
        -1.710422666954443,
        1.605702911834783,
        1.134464013796314,
        0.0,
        -0.087266462599716,
    ),
    dtype=float,
)


def _sample(randomizer, *, episode_key=17):
    return randomizer.begin_episode(
        episode_key=episode_key,
        goal_center_world_m=(-0.35, 0.20, 0.7515),
        goal_half_extent_x_m=0.03,
        goal_half_extent_y_m=0.03,
        home_joint_positions_rad=HOME,
    )


def test_default_config_exposes_every_requested_independent_switch():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    for name in (
        "background_color",
        "ground_color",
        "table_color",
        "target_color",
    ):
        assert isinstance(payload["scene"][name]["enabled"], bool)

    for light in payload["lights"].values():
        for name in ("intensity", "color", "position", "shadows"):
            assert isinstance(light[name]["enabled"], bool)

    for camera_name in ("top_camera", "gripper_camera"):
        camera = payload[camera_name]
        assert isinstance(camera["pose"]["enabled"], bool)
        for name in (
            "brightness",
            "contrast",
            "gamma",
            "saturation",
            "white_balance_rgb_gain",
            "rgb_noise",
            "blur",
            "translation",
            "distortion",
            "crop",
        ):
            assert isinstance(camera["image"][name]["enabled"], bool)

    assert isinstance(payload["joint_state"]["position_noise"]["enabled"], bool)
    assert isinstance(payload["joint_state"]["velocity_noise"]["enabled"], bool)
    assert isinstance(payload["home_initial_pose"]["enabled"], bool)
    for name in ("position", "color", "size"):
        assert isinstance(payload["goal_region"][name]["enabled"], bool)


def test_episode_sample_is_reproducible_and_world_specific():
    config = EnvironmentRandomizationConfig.load(CONFIG_PATH)
    first = EpisodeEnvironmentRandomizer(config, base_seed=23, world_index=2)
    second = EpisodeEnvironmentRandomizer(config, base_seed=23, world_index=2)
    other_world = EpisodeEnvironmentRandomizer(config, base_seed=23, world_index=3)

    sample_a = _sample(first, episode_key=101)
    sample_b = _sample(second, episode_key=101)
    sample_c = _sample(other_world, episode_key=101)

    assert sample_a.to_json_dict() == sample_b.to_json_dict()
    assert sample_a.derived_seed != sample_c.derived_seed


def test_randomized_home_moves_only_toward_zero_and_stays_within_cap():
    config = EnvironmentRandomizationConfig.load(CONFIG_PATH)
    randomizer = EpisodeEnvironmentRandomizer(config, base_seed=7, world_index=0)
    _sample(randomizer)

    randomized = randomizer.randomized_home_joint_positions(HOME)
    delta_deg = np.rad2deg(randomized - HOME)

    for canonical, value in zip(HOME, randomized):
        if canonical > 0.0:
            assert 0.0 <= value <= canonical
        elif canonical < 0.0:
            assert canonical <= value <= 0.0
        else:
            assert value == 0.0

    configured_cap = float(
        config.payload["home_initial_pose"]["toward_zero_max_deg"]
    )
    assert float(np.max(np.abs(delta_deg))) <= configured_cap + 1e-9


def test_goal_position_and_size_record_are_the_reward_values():
    config = EnvironmentRandomizationConfig.load(CONFIG_PATH)
    randomizer = EpisodeEnvironmentRandomizer(config, base_seed=5, world_index=1)
    _sample(randomizer)

    center = randomizer.goal_center_world_m()
    half_x, half_y = randomizer.goal_half_extents_m()
    record = randomizer.record()["goal_region"]

    assert tuple(record["position"]["center_world_m"]) == center
    assert record["size"]["half_extent_x_m"] == half_x
    assert record["size"]["half_extent_y_m"] == half_y


def test_rgb_and_joint_noise_change_per_frame_but_episode_parameters_do_not(tmp_path):
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    # Keep this unit test independent of OpenCV by turning geometric/color
    # transforms off and forcing only frame-level noise to nonzero values.
    for camera_name in ("top_camera", "gripper_camera"):
        for name, node in payload[camera_name]["image"].items():
            node["enabled"] = name == "rgb_noise"
        payload[camera_name]["image"]["rgb_noise"]["std_8bit_range"] = [2.0, 2.0]

    payload["joint_state"]["position_noise"] = {
        "enabled": True,
        "std_deg_range": [0.2, 0.2],
    }
    payload["joint_state"]["velocity_noise"] = {
        "enabled": True,
        "std_rad_s_range": [0.01, 0.01],
    }

    path = tmp_path / "randomization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = EnvironmentRandomizationConfig.load(path)
    randomizer = EpisodeEnvironmentRandomizer(config, base_seed=31, world_index=0)
    sample = _sample(randomizer)
    fixed_record = sample.to_json_dict()

    observation = PolicyObservation(
        joint_positions=HOME.copy(),
        joint_velocities=np.zeros(6, dtype=float),
        top_image=np.full((480, 640, 3), 128, dtype=np.uint8),
        gripper_image=np.full((480, 640, 3), 128, dtype=np.uint8),
    )

    first = randomizer.apply_observation(observation)
    second = randomizer.apply_observation(observation)

    assert first.top_image.shape == observation.top_image.shape
    assert first.top_image.dtype == np.uint8
    assert not np.array_equal(first.top_image, second.top_image)
    assert not np.array_equal(first.gripper_image, second.gripper_image)
    assert not np.array_equal(first.joint_positions, second.joint_positions)
    assert not np.array_equal(first.joint_velocities, second.joint_velocities)
    assert randomizer.record() == fixed_record
