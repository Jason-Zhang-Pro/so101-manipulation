"""Disabling wrist guidance leaves target alignment and movement rewards intact."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.rewards.s1_lift_approach import S1LiftApproachConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_s1_lift_approach_reward import PROFILE, Trace, posture_step
from test_s1_lift_approach_stages import LiftTrace


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
ROTATION_COMPONENTS = (
    "s1_roll_progress", "s1_parallel_progress", "s1_roll_remaining",
    "s1_parallel_regression", "s1_wrist_roll_regression",
)


def disabled_stages(*, posture=True, qualified=False):
    stages = StageTrainingConfig.load(PROFILE)
    sequence = replace(stages.process_reward.s1_lift_approach,
                       wrist_rotation_reward_enabled=False)
    if not posture:
        sequence = replace(sequence, posture_regression=None)
    stages = replace(stages, process_reward=replace(stages.process_reward,
                                                   s1_lift_approach=sequence))
    if qualified:
        stages = replace(stages, s1_success_mode="qualified_reach", s1_reach_hold_s=.15)
    return stages


def phase_trace(phase):
    trace = Trace(disabled_stages())
    if phase == "APPROACH":
        trace.lift()
    assert trace.trace.info["stage_s1_phase"] == phase
    return trace


def assert_rotation_zero(result):
    for key in ROTATION_COMPONENTS:
        assert result.components[key] == 0., key
    bands = {key: value for key, value in result.components.items()
             if key.startswith("s1_roll_band_")}
    assert bands
    assert all(value == 0. for value in bands.values())


@pytest.mark.parametrize("phase", ["LIFT", "APPROACH"])
@pytest.mark.parametrize("channel", [2, 3], ids=["parallel", "wrist_roll"])
def test_rotation_improvement_regression_and_hold_cannot_change_any_reward(phase, channel):
    control, turning = phase_trace(phase), phase_trace(phase)
    # Keep bearing, downward axis, position and all evidence identical. These
    # measured channels isolate the reward from kinematic position changes.
    baseline = [30., 30., 30., 60.]
    for angle in (baseline[channel], 15., 50., 50., 0., 75.):
        angles = baseline.copy()
        angles[channel] = angle
        expected = posture_step(control, baseline)
        actual = posture_step(turning, angles)
        assert_rotation_zero(actual)
        assert actual.components == pytest.approx(expected.components)
        assert actual.reward == pytest.approx(expected.reward)
        assert actual.potential_before == pytest.approx(expected.potential_before)
        assert actual.potential_after == pytest.approx(expected.potential_after)
        assert turning.trace.info["stage_s1_phase"] == phase


@pytest.mark.parametrize("phase", ["LIFT", "APPROACH"])
@pytest.mark.parametrize("kind", ["duplicate", "gap", "contact", "downstream", "failure", "failed_lift"])
def test_disabled_rotation_has_no_carried_cost_on_bad_evidence_or_terminal_steps(phase, kind):
    control, turning = phase_trace(phase), phase_trace(phase)
    for angles in ([30., 30., 5., 5.], [30., 30., 70., 120.]):
        expected = posture_step(control, [30., 30., 30., 60.])
        actual = posture_step(turning, angles)
        assert_rotation_zero(actual)
        assert actual.reward == pytest.approx(expected.reward)
    kwargs = {}
    if kind == "duplicate":
        kwargs["evidence_sim_time_s"] = turning.trace.snapshot.evidence_sim_time_s
    elif kind == "gap":
        kwargs["dt"] = .2
    elif kind == "contact":
        kwargs["left_contact"] = True
    elif kind == "downstream":
        kwargs.update(downstream=True, left_contact=True)
    else:
        kwargs["failure_reason"] = "collision"
        if kind == "failed_lift":
            kwargs["downstream"] = True
    expected = posture_step(control, [30., 30., 30., 60.], **kwargs)
    actual = posture_step(turning, [30., 30., 0., 0.], **kwargs)
    assert_rotation_zero(actual)
    assert actual.components == pytest.approx(expected.components)
    assert actual.reward == pytest.approx(expected.reward)
    if kind in {"failure", "failed_lift"}:
        assert actual.terminated
        assert actual.components["failure"] == -turning.trace.reward.failure_cost


@pytest.mark.parametrize("phase", ["LIFT", "APPROACH"])
@pytest.mark.parametrize("channel,name", [(0, "bearing"), (1, "axis")])
def test_target_alignment_still_rewards_progress_and_penalizes_regression(phase, channel, name):
    trace = phase_trace(phase)
    angles = [60., 60., 30., 60.]
    posture_step(trace, angles)
    angles[channel] = 30.
    improved = posture_step(trace, angles)
    assert improved.components[f"s1_{name}_progress"] > 0.
    assert_rotation_zero(improved)
    angles[channel] = 75.
    regressed = posture_step(trace, angles)
    assert regressed.components[f"s1_{name}_progress"] < 0.
    assert regressed.components[f"s1_{name}_regression"] < 0.
    assert_rotation_zero(regressed)


def test_qualified_reach_accepts_unturned_wrist_but_still_requires_target_alignment():
    trace = LiftTrace(config=disabled_stages(qualified=True))
    trace.finish_lift()
    joints = list(trace.reward.home_joint_positions_rad)
    joints[-1] += .7
    angle = math.radians(60)
    tilted = np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                       [-math.sin(angle), 0., math.cos(angle)]]) @ np.diag([1., -1., -1.])
    bad = trace.reached(joint_positions_rad=joints, grasp_rotation_world=tilted)
    assert bad["stage_s1_roll_error_rad"] > trace.config.process_reward.s1_lift_approach.roll_tolerance_rad
    assert bad["stage_orientation_error_rad"] > trace.config.s1_orientation_tolerance_rad
    assert not bad["stage_reach_candidate"]
    assert bad["stage_reach_hold_s"] == 0.
    for _ in range(4):
        good = trace.reached(joint_positions_rad=joints)
    assert good["stage_s1_roll_error_rad"] == pytest.approx(math.pi / 2)
    assert good["stage_reach_candidate"]
    assert good["stage_success"]
    assert good["stage_result_reason"] == "reach_hold"


@pytest.mark.parametrize("posture,dimension", [(True, 34), (False, 26)])
def test_disabled_flag_roundtrips_without_changing_critic_shape(posture, dimension):
    disabled = disabled_stages(posture=posture)
    enabled = replace(disabled, process_reward=replace(disabled.process_reward,
        s1_lift_approach=replace(disabled.process_reward.s1_lift_approach,
                               wrist_rotation_reward_enabled=True)))
    serialized = disabled.to_dict()
    assert serialized["process_reward"]["s1_lift_approach"]["wrist_rotation_reward_enabled"] is False
    assert StageTrainingConfig.from_dict(serialized) == disabled
    assert disabled.config_hash != enabled.config_hash
    assert disabled.process_reward.config_hash != enabled.process_reward.config_hash
    assert disabled.process_reward.context_dim == enabled.process_reward.context_dim == dimension
    assert disabled.process_reward.context_schema_version == enabled.process_reward.context_schema_version
    assert Trace(disabled).reward.critic_context().shape == Trace(enabled).reward.critic_context().shape == (dimension,)


def test_default_and_explicit_enabled_flag_preserve_historical_hashes():
    original = StageTrainingConfig.load(PROFILE)
    assert original.process_reward.s1_lift_approach.wrist_rotation_reward_enabled is True
    assert "wrist_rotation_reward_enabled" not in S1LiftApproachConfig().to_dict()
    assert original.config_hash == "9564d5364c87362865a670a0d64c059d49dbbe683f1d12e301523891396fa93e"
    assert original.process_reward.config_hash == "e7fdf130e32c24188d5f5a441f1b9132b413b70390ea0e7ecce235c71af90624"
    explicit = original.to_dict()
    explicit["process_reward"]["s1_lift_approach"]["wrist_rotation_reward_enabled"] = True
    restored = StageTrainingConfig.from_dict(explicit)
    assert restored.to_dict() == original.to_dict()
    assert restored.config_hash == original.config_hash


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_rotation_flag_requires_a_boolean(value):
    with pytest.raises(ValueError, match="wrist_rotation_reward_enabled"):
        S1LiftApproachConfig.from_dict({"wrist_rotation_reward_enabled": value})


@pytest.mark.parametrize("profile", ["act_ppo_stages_s0_s5.json", "act_ppo_stages_s1_lift_approach.json",
                                     "act_ppo_stages_s1_lift_completion.json"])
def test_current_profiles_disable_wrist_guidance(profile):
    stages = StageTrainingConfig.load(CONFIG_DIR / profile)
    assert stages.process_reward.s1_lift_approach.wrist_rotation_reward_enabled is False
