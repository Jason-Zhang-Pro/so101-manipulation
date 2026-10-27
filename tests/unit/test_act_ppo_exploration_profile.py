"""Exercise the launch profile through the real PPO configuration and sampler.

These are offline command-distribution checks, not evidence of task performance.
Only selected variable assignments are evaluated; the training script is never
sourced and no simulator or policy checkpoint is loaded.
"""

from dataclasses import replace
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.act.bounded_actions import bound_action_means

from so101_capstone.policies.so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS,
    policy_positions_to_gazebo,
)
from so101_capstone.training.act_ppo import (
    ActPpoExploration,
    evaluate_bounded_policy_actions,
    policy_action_bounds,
    sample_bounded_policy_actions,
)
from so101_capstone.training.act_ppo_gazebo import _act_ppo_config, _exploration_stage
from so101_capstone.training.act_ppo_multiworld_gazebo import _build_base_config, _parser


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "finetune_act_ppo_so101.sh"
VARIABLE_FLAGS = {
    "REWARD_PROFILE": "--reward-profile",
    "EXPLORATION_PROFILE": "--exploration-profile",
    "IDLE_EXPLORATION_LOG_STD": "--idle-exploration-log-std",
    "MOVING_EXPLORATION_LOG_STD": "--moving-exploration-log-std",
    "MANIPULATION_EXPLORATION_LOG_STD": "--manipulation-exploration-log-std",
    "HOME_HOLD_EXPLORATION_LOG_STD": "--home-hold-exploration-log-std",
    "MOVING_GRIPPER_EXPLORATION_LOG_STD": "--moving-gripper-exploration-log-std",
    "MANIPULATION_GRIPPER_EXPLORATION_LOG_STD": "--manipulation-gripper-exploration-log-std",
    "GRIPPER_MAXIMUM_LOG_STD": "--gripper-maximum-log-std",
    "INITIAL_LOG_STD": "--initial-log-std",
    "MINIMUM_LOG_STD": "--minimum-log-std",
    "MAXIMUM_LOG_STD": "--maximum-log-std",
    "ACTOR_LEARNING_RATE": "--actor-learning-rate",
    "ACTOR_BACKBONE_LEARNING_RATE": "--actor-backbone-learning-rate",
    "CRITIC_LEARNING_RATE": "--critic-learning-rate",
    "EXPLORATION_LEARNING_RATE": "--exploration-learning-rate",
    "PPO_EPOCHS": "--ppo-epochs",
    "TARGET_KL": "--target-kl",
}


@pytest.fixture(scope="module")
def launch_profile():
    script = SCRIPT.read_text()
    assignments = []
    for variable in VARIABLE_FLAGS:
        matches = re.findall(rf"^{variable}=.*$", script, re.MULTILINE)
        assert len(matches) == 1, f"Expected one launch default for {variable}"
        assignments.append(matches[0])
    environment = {name: value for name, value in os.environ.items()
                   if name not in VARIABLE_FLAGS and name != "BASH_ENV"}
    print_values = 'printf "%s\\0" ' + " ".join(f'"${name}"' for name in VARIABLE_FLAGS)
    result = subprocess.run(
        ["bash", "-c", "\n".join([*assignments, print_values])],
        env=environment, check=True, capture_output=True,
    )
    values = result.stdout.decode().rstrip("\0").split("\0")
    assert len(values) == len(VARIABLE_FLAGS)
    arguments = ["--act-checkpoint", "/tmp/offline_exploration_profile",
                 "--final-target-absent-s", "0"]
    for flag, value in zip(VARIABLE_FLAGS.values(), values):
        arguments.extend([flag, value])
    return _build_base_config(_parser().parse_args(arguments))


def _effective_log_std(config, phase, *, learned_base=None, distance=.1, home_error=.5):
    exploration = ActPpoExploration(_act_ppo_config(config, 5))
    if learned_base is not None:
        with torch.no_grad():
            exploration.log_std.fill_(learned_base)
        exploration.clamp_()
    info = {"task_phase": phase, "target_distance_m": distance,
            "home_error_rad": home_error, "home_joint_tolerance_rad": math.radians(1)}
    name, _, offset = _exploration_stage(info, config)
    return name, exploration.with_offset(offset).detach()


def _previous_profile(config):
    return replace(
        config, exploration_profile="stage_log_std", initial_log_std=-4., maximum_log_std=-3.5,
        idle_exploration_log_std=-7., home_hold_exploration_log_std=-7.,
        moving_exploration_log_std=-4., manipulation_exploration_log_std=-4.,
        moving_gripper_exploration_log_std=-3.,
        manipulation_gripper_exploration_log_std=-2.3,
    )


def _command_positions(actions):
    return np.stack([policy_positions_to_gazebo(row) for row in actions.detach().numpy()])


def test_aggressive_launch_defaults_reach_real_ppo_config(launch_profile):
    ppo = _act_ppo_config(launch_profile, 5)
    assert launch_profile.reward_profile == "pick_place_v2"
    assert launch_profile.exploration_profile == "final_good_act"
    assert ppo.initial_log_std == -3.
    assert ppo.minimum_log_std == -9.
    assert ppo.maximum_log_std == -2.5
    assert ppo.gripper_maximum_log_std == -1.5
    assert ppo.maximum_log_std - ppo.initial_log_std == .5
    assert ppo.actor_learning_rate == 2e-4
    assert ppo.actor_backbone_learning_rate == 2e-5
    assert ppo.critic_learning_rate == 2e-4
    assert ppo.exploration_learning_rate == 2e-4
    assert ppo.ppo_epochs == 2
    assert ppo.target_kl == .02


@pytest.mark.parametrize("distance,sigma", [(.2, .12), (.1, .08), (.03, .05)])
@pytest.mark.parametrize("learned_shift", [0., .1])
def test_launch_physical_profile_survives_real_exploration_bounds(launch_profile, distance, sigma, learned_shift):
    _, logs = _effective_log_std(
        launch_profile, "APPROACH", distance=distance,
        learned_base=launch_profile.initial_log_std + learned_shift,
    )
    spans = np.asarray([(j.gazebo_upper_rad-j.gazebo_lower_rad)/2
                        for j in SIM_JOINT_CALIBRATIONS])
    np.testing.assert_allclose(spans * logs.exp().numpy(), sigma * math.exp(learned_shift), rtol=1e-6)


@pytest.mark.parametrize("phase", ["WAIT_HOME", "RETURN_HOME", "HOME_HOLD"])
def test_launch_home_profile_settles_at_real_tolerance(launch_profile, phase):
    _, logs = _effective_log_std(launch_profile, phase, home_error=math.radians(1))
    spans = np.asarray([(j.gazebo_upper_rad-j.gazebo_lower_rad)/2
                        for j in SIM_JOINT_CALIBRATIONS])
    np.testing.assert_allclose(spans * logs.exp().numpy(), .001, rtol=1e-6)


@pytest.mark.parametrize(
    "phase,group,arm,gripper,max_arm,max_gripper",
    [
        ("WAIT_HOME", "idle", -6., -6., -5.5, -5.5),
        ("APPROACH", "moving", -3., -2.5, -2.5, -2.),
        ("GRASP_LIFT", "manipulation", -3.5, -2., -3., -1.5),
        ("TRANSPORT", "moving", -3., -2.5, -2.5, -2.),
        ("PLACE_RELEASE", "manipulation", -3.5, -2., -3., -1.5),
        ("RETURN_HOME", "moving", -3., -2.5, -2.5, -2.),
        ("HOME_HOLD", "home_hold", -7., -7., -6.5, -6.5),
    ],
)
def test_stage_offsets_and_shared_base_cap_preserve_actual_bounds(
    launch_profile, phase, group, arm, gripper, max_arm, max_gripper,
):
    launch_profile = replace(launch_profile, exploration_profile="stage_log_std")
    name, initial = _effective_log_std(launch_profile, phase)
    capped_name, maximum = _effective_log_std(launch_profile, phase, learned_base=100.)
    assert name == capped_name == group
    torch.testing.assert_close(initial, torch.tensor([arm] * 5 + [gripper]))
    torch.testing.assert_close(maximum, torch.tensor([max_arm] * 5 + [max_gripper]))


@pytest.mark.parametrize(
    "phase,arm_multiplier,gripper_multiplier",
    [
        ("WAIT_HOME", math.exp(1.), math.exp(1.)),
        ("APPROACH", math.exp(1.), math.exp(.5)),
        ("GRASP_LIFT", math.exp(.5), math.exp(.3)),
        ("HOME_HOLD", 1., 1.),
    ],
)
def test_same_seed_midpoint_command_rms_matches_profile_change_and_replays(
    launch_profile, phase, arm_multiplier, gripper_multiplier,
):
    launch_profile = replace(launch_profile, exploration_profile="stage_log_std")
    means = torch.tensor([0.] * 5 + [50.], dtype=torch.float64).expand(4096, 6)
    centre = policy_positions_to_gazebo(means[0].numpy())
    rms = []
    for config in (_previous_profile(launch_profile), launch_profile):
        _, logs = _effective_log_std(config, phase)
        torch.manual_seed(732)
        actions, sampled_log_prob, latents = sample_bounded_policy_actions(
            means, logs, return_latent=True,
        )
        replay, _ = evaluate_bounded_policy_actions(
            means, actions, logs, policy_action_latents=latents,
        )
        torch.testing.assert_close(replay, sampled_log_prob, rtol=0., atol=0.)
        # Midpoint noise is entirely inside the bounds for this fixed sample.
        assert bool((latents.abs() < 1.).all())
        commands = _command_positions(actions)
        rms.append(np.sqrt(np.mean((commands - centre) ** 2, axis=0)))
        # Use runtime calibration, including the gripper's 0..100 policy range.
        half_spans = (
            policy_positions_to_gazebo([100.] * 6)
            - policy_positions_to_gazebo([-100.] * 5 + [0.])
        ) / 2.
        expected_sigma = half_spans * logs.double().exp().numpy()
        np.testing.assert_allclose(rms[-1], expected_sigma, rtol=.035)
    np.testing.assert_allclose(
        rms[1] / rms[0], [arm_multiplier] * 5 + [gripper_multiplier], rtol=1e-6,
    )


@pytest.mark.parametrize("phase", ["WAIT_HOME", "APPROACH", "GRASP_LIFT", "HOME_HOLD"])
def test_bounded_profile_preserves_inward_exploration_and_exact_replay(launch_profile, phase):
    raw_means = torch.tensor([110., -110., 100., -100., 120., -5.]).expand(1024, 6)
    bounds_config = SimpleNamespace(action_bounds_lower=[-100.] * 5 + [0.],
                                   action_bounds_upper=[100.] * 6, action_bound_softness=.02)
    means = bound_action_means(raw_means, bounds_config)
    _, logs = _effective_log_std(launch_profile, phase)
    torch.manual_seed(951)
    actions, sampled_log_prob, latents = sample_bounded_policy_actions(
        means, logs, return_latent=True,
    )
    replay, _ = evaluate_bounded_policy_actions(
        means, actions, logs, policy_action_latents=latents,
    )
    lower, upper = policy_action_bounds(device=actions.device, dtype=actions.dtype)
    assert bool((latents.abs() <= 1.).all())
    assert bool(((means >= lower) & (means <= upper)).all())
    assert bool(((actions >= lower) & (actions <= upper)).all())
    outward_sign = torch.tensor([1., -1., 1., -1., 1., -1.])
    assert bool((((actions - means) * outward_sign < 0).float().mean(dim=0) > .4).all())
    assert bool((actions.std(dim=0) > 0.).all())
    # Allow float32 rounding at a limit, but reject the old tail-clipping pileup.
    endpoint_fraction = ((actions == lower) | (actions == upper)).float().mean(dim=0)
    assert bool((endpoint_fraction < .01).all())
    torch.testing.assert_close(replay, sampled_log_prob, rtol=0., atol=0.)
    commands = _command_positions(actions)
    # Calibration mapping is the deployed command transformation, not exp(log_std) radians.
    lower_rad = np.array([joint.gazebo_lower_rad for joint in SIM_JOINT_CALIBRATIONS])
    upper_rad = np.array([joint.gazebo_upper_rad for joint in SIM_JOINT_CALIBRATIONS])
    assert np.all(commands >= lower_rad - 1e-12)
    assert np.all(commands <= upper_rad + 1e-12)
