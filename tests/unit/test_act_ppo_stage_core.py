from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from so101_capstone.environments.reach_env_core import ReachEnvironmentConfig, So101ReachEnvironmentCore
from so101_capstone.environments.task_measurements import TaskMeasurementBuilder, TaskMeasurementError
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training.act_ppo_gazebo import (
    ActPpoGazeboConfig, _episode_scene, _execution_signature, _resolved_reward,
    _episode_step_limit, _sample_training_scene, _parser, _reward_cli_kwargs,
)
from test_task_measurements import joint, cube, contacts, GOAL


def config(**kw):
    return ActPpoGazeboConfig(act_checkpoint=Path("/tmp/actor"), reward_profile="pick_place_v2",
                             training_stages="s0_s1", final_target_absent_s=0, **kw)


def test_stage_scene_overrides_old_modes_and_delays_and_is_reproducible():
    cfg = config(task_mode="idle_completed", target_delay_s=3)
    first = [_episode_scene(cfg, i) for i in range(100)]
    assert first == [_episode_scene(cfg, i) for i in range(100)]
    assert {mode for mode, _, _ in first} == {"idle_empty", "pick_place"}
    assert {delay for _, delay, _ in first} == {0}
    assert _episode_step_limit(cfg, "idle_empty") == 151
    assert _episode_step_limit(cfg, "pick_place") == 301


def test_cli_and_checkpoint_contract_record_stage_thresholds(tmp_path):
    path = tmp_path / "stages.json"
    import json
    path.write_text(json.dumps(StageTrainingConfig().to_dict()))
    args = _parser().parse_args(["--act-checkpoint", "/tmp/actor", "--training-stages", "s0_s1",
                                "--stage-config", str(path), "--s0-probability", "0.4"])
    cfg = ActPpoGazeboConfig(act_checkpoint=args.act_checkpoint, final_target_absent_s=0,
                            **_reward_cli_kwargs(args))
    signature = _execution_signature(cfg, _resolved_reward(cfg))
    assert signature["stage_training"]["config"]["s1_budget_s"] == 10
    assert cfg.to_json_dict()["stage_config"] == str(path)
    path.write_text(json.dumps(replace(StageTrainingConfig(), s1_budget_s=11).to_dict()))
    assert signature != _execution_signature(cfg, _resolved_reward(cfg))
    with pytest.raises(ValueError, match="disable"):
        config(reach_curriculum=True)


def test_explicit_motion_prehistory_makes_first_stage_frame_measured():
    builder = TaskMeasurementBuilder()
    builder.reset(target_history=(cube(0), cube(.03, 2)))
    result = builder.update(joint(.06, 3), cube(.06, 3), control_sim_time_s=.06,
                            contact_cache=contacts(0, .03, .06), goal_center_world_m=GOAL)
    assert result.evidence_valid and result.target_motion_valid
    assert result.target_linear_velocity_m_s == (0., 0., 0.)
    builder.reset()
    result = builder.update(joint(.06, 3), cube(.06, 3), control_sim_time_s=.06,
                            cube_history=(cube(0), cube(.03, 2)),
                            contact_cache=contacts(.06), goal_center_world_m=GOAL)
    assert not result.target_motion_valid  # Ordinary resets keep old semantics.


@pytest.mark.parametrize("history", [(cube(0), cube(.2)), (cube(.03), cube(0)),
                                     (cube(0), cube(.03, name="different"))])
def test_bad_motion_prehistory_rejected(history):
    with pytest.raises(TaskMeasurementError):
        TaskMeasurementBuilder().reset(target_history=history)


@pytest.mark.parametrize("budget, hold", [(5, 3), (2, 1)])
def test_s0_full_window_cut_retains_full_task_reward_potential(budget, hold):
    reward = PickPlaceRewardConfig()
    # The old episode cap cannot end S0 before its declared observation window.
    core = So101ReachEnvironmentCore(ReachEnvironmentConfig(max_episode_steps=2), reward,
              goal_region_center_world_m=GOAL, reward_profile="pick_place_v2", task_mode="idle_empty")
    core.configure_training_stage("S0", StageTrainingConfig(s0_budget_s=budget, s0_home_hold_s=hold))
    steps = budget * 30
    stream = contacts(*np.arange(steps + 1) / 30)
    _, info = core.reset_from_feedback(joint(0), now_monotonic_s=0, step_end_sim_time_s=0,
                                      contact_cache=stream)
    assert info["stage_result"] == "RUNNING"
    for step in range(1, steps + 1):
        _, _, terminated, truncated, info = core.step_from_feedback(joint(step/30, step+1),
                 now_monotonic_s=0, step_end_sim_time_s=step/30, control_dt_s=1/30,
                 contact_cache=stream)
        assert not terminated
        assert truncated == (step == steps)
    assert info["stage_success"]
    assert info["task_outcome"] == "running"
    assert not info["success"] and not info["full_unassisted_success"]
    assert info["potential_after"] > 0
    assert info["reward_components"]["success"] == 0
    assert info["reward_components"]["failure"] == 0


@pytest.mark.parametrize("s0_hold", [1, 3, 4])
def test_s1_initial_frame_uses_recent_same_target_history(s0_hold):
    reward = PickPlaceRewardConfig()
    core = So101ReachEnvironmentCore(reward_config=reward, goal_region_center_world_m=GOAL,
                                    reward_profile="pick_place_v2")
    core.configure_training_stage("S1", StageTrainingConfig(s0_home_hold_s=s0_hold))
    assert core._home_hold_required_s == reward.home_hold_min_s
    _, info = core.reset_from_feedback(joint(.06, 3), cube(.06, 3), now_monotonic_s=0,
          step_end_sim_time_s=.06, cube_history=(cube(0), cube(.03, 2)),
          contact_cache=contacts(0, .03, .06), camera_source_times_s=(.06, .06))
    assert info["start_valid"] and info["target_motion_valid"]
    assert info["stage_elapsed_s"] == 0
    assert info["stage_result"] == "RUNNING"
