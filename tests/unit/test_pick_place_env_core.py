from dataclasses import replace

import numpy as np
import pytest

from so101_capstone.environments.contact_state import FingerContactCache
from so101_capstone.environments.reach_env_core import ReachEnvironmentConfig, ReachEnvironmentStateError, So101ReachEnvironmentCore
from so101_capstone.environments.state_cache import JointStateSnapshot, MEASURED_JOINT_NAMES, TargetCubeSnapshot
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM

CFG = PickPlaceRewardConfig()
GOAL = (.078, -.3, .7515)


def joint(t, seq, velocity=0):
    return JointStateSnapshot(dict(zip(MEASURED_JOINT_NAMES, CFG.home_joint_positions_rad)),
                              dict.fromkeys(MEASURED_JOINT_NAMES, velocity), seq, t, t)


def cube(t, seq, position=(.25, 0, .765), name="target_cube_1"):
    return TargetCubeSnapshot(name, np.array(position), np.array([0., 0., 0., 1.]), seq, t, seq, t)


def feedback(core, t, seq, *, target=None, velocity=0, reset=False):
    contacts = FingerContactCache()
    for stamp in (max(0, t-.05), t):
        contacts.update("left", sim_time_s=stamp)
        contacts.update("right", sim_time_s=stamp)
    fn = core.reset_from_feedback if reset else core.step_from_feedback
    return fn(joint(t, seq, velocity), target, now_monotonic_s=t,
              step_end_sim_time_s=t, contact_cache=contacts)


def make_core(**kwargs):
    return So101ReachEnvironmentCore(reward_profile="pick_place_v2", reward_config=CFG,
                                    goal_region_center_world_m=GOAL, **kwargs)


def test_profile_requires_matching_explicit_config_and_legacy_stays_default():
    assert So101ReachEnvironmentCore().reward_profile == "legacy"
    with pytest.raises(TypeError, match="Legacy"):
        So101ReachEnvironmentCore(reward_config=CFG)
    with pytest.raises(ValueError, match="Task modes"):
        So101ReachEnvironmentCore(task_mode="idle_empty")


def test_reset_context_and_measured_velocity_feed_the_same_task_snapshot():
    core = make_core(task_mode="idle_empty")
    _, initial = feedback(core, 0, 1, reset=True)
    assert len(initial["critic_task_context"]) == TASK_CONTEXT_DIM
    _, reward, terminated, truncated, info = feedback(core, .05, 2, velocity=.2)
    assert not terminated and not truncated
    assert info["robot_at_home"] and not info["robot_static"]
    assert info["task_phase"] == "wait_home" and not info["success"]
    assert info["reward_components"]["home_velocity"] < 0
    assert np.isfinite(reward)
    assert info["task_snapshot"]["robot_static"] == info["robot_static"]
    assert info["measurement_config"]["goal_half_extents_xy_m"] == (.05, .05)


def test_delayed_target_appears_once_and_removal_is_invalid_sampling():
    core = make_core()
    feedback(core, 0, 1, reset=True)
    _, _, _, _, info = feedback(core, .05, 2, target=cube(.05, 1))
    assert info["physical_target_present"]
    assert info["target_transition"] == "activated"
    assert not info["target_motion_valid"]
    with pytest.raises(ReachEnvironmentStateError, match="disappeared"):
        feedback(core, .1, 3)


def test_completed_idle_requires_physical_object_before_reset_and_no_success_bonus():
    core = make_core(task_mode="idle_completed", config=ReachEnvironmentConfig(max_episode_steps=2))
    with pytest.raises(ReachEnvironmentStateError, match="initially supported"):
        feedback(core, 0, 1, reset=True)
    placed = (GOAL[0], GOAL[1], .768)
    feedback(core, 0, 1, target=cube(0, 1, placed), reset=True)
    feedback(core, .05, 2, target=cube(.05, 2, placed))
    _, _, terminated, truncated, info = feedback(core, .1, 3, target=cube(.1, 3, placed))
    assert not terminated and truncated
    assert not info["success"] and info["reward_components"]["success"] == 0


def test_clock_dt_and_missing_source_stamp_fail_closed():
    core = make_core()
    with pytest.raises(ReachEnvironmentStateError, match="simulation time"):
        core.reset_from_feedback(joint(0, 1), now_monotonic_s=0)
    feedback(core, 0, 1, reset=True)
    with pytest.raises(ReachEnvironmentStateError, match="dt"):
        core.step_from_feedback(joint(.05, 2), now_monotonic_s=.05,
                                step_end_sim_time_s=.05, control_dt_s=.1)


def test_monitor_continues_physics_without_reset_or_duplicate_terminal_reward(monkeypatch):
    core = make_core(observe_after_success_s=.1)
    feedback(core, 0, 1, reset=True)
    original_update = core._task_tracker.update
    calls = 0

    def certified_success(measurements):
        # Task-state criteria have their own trajectory tests. This controlled
        # event verifies environment termination/monitor wiring in isolation.
        nonlocal calls
        snapshot = original_update(measurements)
        calls += 1
        return replace(snapshot, outcome="success", success_event=calls == 1)

    monkeypatch.setattr(core._task_tracker, "update", certified_success)
    _, _, terminated, _, first = feedback(core, .05, 2)
    assert not terminated and first["training_terminated"] and first["success_event"]
    _, reward, terminated, _, continuing = feedback(core, .1, 3, velocity=.2)
    assert reward == 0 and not terminated and continuing["monitoring_only"]
    assert not continuing["robot_static"]
    _, reward, terminated, _, final = feedback(core, .16, 4)
    assert reward == 0 and terminated and not final["success_event"]
    assert not core.episode_active


def test_home_confirmation_window_covers_two_chunks_and_is_seeded():
    first, second = make_core(), make_core()
    first.configure_task("pick_place", seed=7, n_action_steps=100)
    second.configure_task("pick_place", seed=7, n_action_steps=100)
    assert first._home_hold_required_s == second._home_hold_required_s
    assert first._home_hold_required_s >= 200/30


def test_command_change_cost_uses_reset_hold_command_for_first_transition():
    core = So101ReachEnvironmentCore(reward_profile="pick_place_v2",
        reward_config=replace(CFG, command_change_cost_per_step=1.0),
        task_mode="idle_empty", goal_region_center_world_m=GOAL)
    contacts = FingerContactCache()
    for t in (0, .05):
        for side in ("left", "right"):
            contacts.update(side, sim_time_s=t)
    command = np.asarray(CFG.home_joint_positions_rad).copy()
    command[0] += .01
    _, reset_info = core.reset_from_feedback(joint(0, 1), now_monotonic_s=0,
        step_end_sim_time_s=0, contact_cache=contacts, command_joint_targets_rad=command)
    assert reset_info["initial_command_source"] == "reset_hold_command"
    _, _, _, _, info = core.step_from_feedback(joint(.05, 2), now_monotonic_s=.05,
        step_end_sim_time_s=.05, contact_cache=contacts, command_joint_targets_rad=command)
    assert info["reward_components"]["command_change"] == 0
    assert tuple(core._task_snapshot.measurements.executed_command_joint_targets_rad) == tuple(command)


@pytest.mark.parametrize("schema,history_size", [(3, 0), (4, 0), (5, 33)])
def test_reward_history_context_reaches_ppo_after_reset_and_step(tmp_path, schema, history_size):
    import json
    from so101_capstone.training.act_ppo_gazebo import (
        ActPpoGazeboConfig, _act_ppo_config, _critic_context, _execution_signature,
    )
    reward_config = PickPlaceRewardConfig(reward_schema_version=schema)
    path = tmp_path / "reward.json"
    path.write_text(json.dumps(reward_config.to_dict()))
    config = ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "act", reward_profile="pick_place_v2",
        reward_config=path, gamma=reward_config.gamma, final_target_absent_s=0,
    )
    core = So101ReachEnvironmentCore(
        reward_profile="pick_place_v2", reward_config=reward_config,
        goal_region_center_world_m=GOAL, task_mode="idle_empty",
    )
    _, initial = feedback(core, 0, 1, reset=True)
    _, _, _, _, after = feedback(core, 1/30, 2, velocity=.2)
    for info in (initial, after):
        assert len(info["critic_task_context"]) == TASK_CONTEXT_DIM + history_size
        assert info["task_context_schema_version"] == 2 + int(schema == 5)
        assert info["home_joint_tolerance_rad"] == reward_config.home_joint_tolerance_rad
        context = _critic_context(info, np.zeros((1, 6)), 0, 1, config)
        assert context["critic_task_context"].shape == (config.critic_task_context_dim,)
        assert np.isfinite(context["critic_task_context"]).all()
    if history_size:
        assert initial["critic_task_context"] != after["critic_task_context"]
        np.testing.assert_array_equal(after["critic_task_context"][-history_size:],
                                      core._pick_place_reward.critic_history_context())
        assert after["reward_history_context_schema_version"] == 1
    else:
        assert "reward_history_context_schema_version" not in after
    assert _act_ppo_config(config, 1).critic_task_context_dim == TASK_CONTEXT_DIM + history_size
    signature = _execution_signature(config, reward_config)
    assert signature["task_context_dim"] == TASK_CONTEXT_DIM + history_size
    assert signature["task_context_schema"] == after["task_context_schema_version"]
    assert ("reward_history_context_schema" in signature) == bool(history_size)
    # Stale worker info must fail instead of silently training a critic with a
    # different reward-memory contract.
    malformed = dict(after, critic_task_context=[0.] * (TASK_CONTEXT_DIM + history_size + 1))
    with pytest.raises(ValueError, match="critic task context"):
        _critic_context(malformed, np.zeros((1, 6)), 0, 1, config)
