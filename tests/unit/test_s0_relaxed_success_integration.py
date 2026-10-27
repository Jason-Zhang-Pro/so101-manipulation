"""S0's relaxed milestone cuts sampling without declaring full-task success."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.environments.reach_env_core import (
    ReachEnvironmentConfig, So101ReachEnvironmentCore,
)
from so101_capstone.environments.state_cache import MEASURED_JOINT_NAMES
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training.act_ppo_stage_success_skip import StageSuccessSkipController
from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig, _resolved_stage_config, _execution_signature,
)
from test_task_measurements import contacts, joint, GOAL


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("act_ppo_stages_s1_lift_approach.json", "act_ppo_stages_s1_lift_completion.json")


def run_s0(*, successful):
    reward = PickPlaceRewardConfig()
    stages = StageTrainingConfig.load(ROOT / "config" / PROFILES[-1])
    core = So101ReachEnvironmentCore(
        ReachEnvironmentConfig(max_episode_steps=2), reward,
        goal_region_center_world_m=GOAL, reward_profile="pick_place_v2", task_mode="idle_empty",
    )
    core.configure_training_stage("S0", stages)
    stream = contacts(*np.arange(151) / 30)
    _, initial = core.reset_from_feedback(joint(0), now_monotonic_s=0,
        step_end_sim_time_s=0, contact_cache=stream)
    assert initial["start_valid"]
    for step in range(1, 151):
        feedback = replace(joint(step / 30, step + 1, offset=math.radians(6 if successful else 9)),
            velocities=dict.fromkeys(MEASURED_JOINT_NAMES, .8))
        _, _, terminated, truncated, info = core.step_from_feedback(feedback,
            now_monotonic_s=0, step_end_sim_time_s=step / 30, control_dt_s=1 / 30,
            contact_cache=stream)
        if terminated or truncated:
            return reward, info, terminated, truncated
    raise AssertionError("S0 must finish within its configured budget")


def test_s0_success_ends_sampling_early_but_keeps_task_reward_and_bootstrap():
    reward, info, terminated, truncated = run_s0(successful=True)
    assert info["stage_success"] and info["stage_result"] == "SUCCESS"
    assert info["stage_elapsed_s"] == pytest.approx(1.)
    assert info["stage_home_hold_s"] == pytest.approx(1.)
    assert truncated and not terminated
    assert info["task_outcome"] == "running"
    assert not info["success"] and not info["full_unassisted_success"]
    assert not info["robot_at_home"] and not info["robot_static"]
    assert info["stage_robot_at_home"] and not info["stage_robot_static"]
    assert info["stage_home_static_fraction"] == 0.
    assert info["reward_components"]["success"] == 0
    assert info["reward_components"]["failure"] == 0
    assert reward.home_joint_tolerance_rad == pytest.approx(math.radians(1))
    assert reward.home_velocity_tolerance_rad_s == .03
    assert info["home_joint_tolerance_rad"] == reward.home_joint_tolerance_rad


@pytest.mark.parametrize("successes,total,qualified", [(4, 5, True), (4, 6, False)])
def test_formal_s0_results_feed_the_existing_eighty_percent_skip(successes, total, qualified):
    _, passed, _, _ = run_s0(successful=True)
    _, failed, terminated, truncated = run_s0(successful=False)
    assert failed["stage_result"] == "TIMEOUT" and not failed["stage_success"]
    assert failed["stage_elapsed_s"] == pytest.approx(5.)
    assert truncated and not terminated
    controller = StageSuccessSkipController("s0_s1", threshold=.8)
    for index in range(total):
        controller.observe({"valid": True, "training_stage": "S0",
                            "stage_success": (passed if index < successes else failed)["stage_success"]})
    controller.observe({"valid": True, "training_stage": "S1", "stage_success": False})
    result = controller.finish_batch()
    assert result.success_rates["S0"] == successes / total
    assert result.exclude_update_stages == (frozenset({"S0"}) if qualified else frozenset())
    assert result.next_skip_stages == result.exclude_update_stages


@pytest.mark.parametrize("profile", PROFILES)
def test_current_profiles_and_resolved_checkpoint_signature_use_new_s0_contract(profile):
    base = ActPpoGazeboConfig(act_checkpoint=Path("/tmp/actor"), reward_profile="pick_place_v2",
        training_stages="s0_s1", final_target_absent_s=0, stage_config=ROOT / "config" / profile)
    stage = _resolved_stage_config(base)
    assert stage.s0_success_mode == "position_hold"
    assert stage.s0_home_hold_s == 1.
    assert stage.s0_budget_s == 5.
    assert stage.s0_home_joint_tolerance_rad == pytest.approx(math.radians(8))
    assert stage.s0_home_velocity_tolerance_rad_s is None
    signature = _execution_signature(base, PickPlaceRewardConfig())
    assert signature["stage_training"]["config"] == stage.to_dict()
    legacy = replace(stage, s0_success_mode="window_end", s0_home_hold_s=3.,
        s0_home_joint_tolerance_rad=None, s0_home_velocity_tolerance_rad_s=None)
    assert stage.config_hash != legacy.config_hash
    assert stage.process_reward == legacy.process_reward


def test_saved_command_no_longer_overrides_the_one_second_hold_with_three():
    script = (ROOT / "codex_workspace/commands/finetune_good40_posture_distance.sh").read_text()
    assert "export S0_HOME_HOLD_S=1.0" in script
    assert "export S0_HOME_HOLD_S=3.0" not in script
