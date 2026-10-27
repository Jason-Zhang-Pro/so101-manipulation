
import math
import numpy as np
import pytest

from so101_capstone.rewards.reach_reward import (
    ReachRewardConfig, ReachRewardInput, RewardState, StatefulReachReward
)

I=np.eye(3)

def obs(cfg, *, target_name="a", target=(0.20,0.0,0.765), grasp=(0.0,0.0,0.765), goal=(0.0,0.20,0.0), joints=None, rotation=I):
    if joints is None:
        joints=cfg.home_joint_positions
    return ReachRewardInput(
        joint_positions_rad=joints,
        grasp_position_world_m=grasp,
        grasp_rotation_world=rotation,
        goal_region_center_world_m=goal,
        target_name=target_name,
        target_position_world_m=None if target_name is None else target,
    )

def tracker(cfg=None):
    cfg=ReachRewardConfig() if cfg is None else cfg
    t=StatefulReachReward(cfg)
    t.reset(joint_positions_rad=cfg.home_joint_positions)
    return cfg,t

def test_state2_can_jump_directly_to_state4_for_target_already_in_goal():
    cfg,t=tracker()
    # New target always spends its activation step in State2.
    r=t.step(obs(cfg,target=(0,0.20,0.765),grasp=(0.2,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH

    # On the next measured step the same target is still in the goal, so
    # State2 may jump directly to State4 without requiring State3.
    r=t.step(obs(cfg,target=(0,0.20,0.765),grasp=(0.2,0,0.765)))
    assert r.state is RewardState.TARGET_RELEASE_RETURN_HOME

    # goal_entry belongs to State3, so a direct State2 -> State4 transition
    # must not award the +10 goal-entry bonus.
    assert r.components["goal_entry"] == 0

    # Leaving State4 returns to State2, but this is not the target's first
    # State2 phase anymore.
    r=t.step(obs(cfg,target=(0.2,0,0.765),grasp=(0.0,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH
    assert r.components["early_approach"] == 0
    assert r.components["distance_improvement"] == 0


def test_first_state2_distance_once():
    cfg=ReachRewardConfig(
        gripper_target_opening_penalty_per_deg=0.0,
        state2_counterclockwise_wrist_roll_penalty_per_deg=0.0,
        distance_penalty_weight=0.0,
        alignment_progress_reward_per_bucket=0.0,
        alignment_success_bonus=0.0,
        gripper_opening_reward_per_deg=0.0,
    )
    t=StatefulReachReward(cfg); t.reset(joint_positions_rad=cfg.home_joint_positions)
    r=t.step(obs(cfg, target=(0.2,0,0.765), grasp=(0,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH
    r=t.step(obs(cfg, target=(0.2,0,0.765), grasp=(0.05,0,0.765)))
    assert r.components["early_approach"] == pytest.approx(0.0)
    assert r.components["distance_improvement"] > 0
    # enter state3
    r=t.step(obs(cfg,target=(0.2,0,0.765),grasp=(0.2,0,0.765)))
    assert r.state is RewardState.TARGET_GRASP_TRANSPORT
    # Leave to State2 at 3 cm, strictly outside the inclusive 2 cm radius.
    r=t.step(obs(cfg,target=(0.2,0,0.765),grasp=(0.17,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH
    assert r.components["early_approach"] == 0
    assert r.components["distance_improvement"] == 0

def test_gripper_45_penalty():
    cfg=ReachRewardConfig(
        state2_counterclockwise_wrist_roll_penalty_per_deg=0.0,
        distance_penalty_weight=0.0,
        early_approach_bonus=0.0,
        distance_improvement_weight=0.0,
        gripper_opening_reward_per_deg=0.0,
        alignment_progress_reward_per_bucket=0.0,
        alignment_success_bonus=0.0,
        gripper_target_opening_deg=45.0,
        gripper_target_opening_penalty_per_deg=1.0,
    )
    t=StatefulReachReward(cfg); t.reset(joint_positions_rad=cfg.home_joint_positions)
    joints=list(cfg.home_joint_positions)
    joints[-1]+=math.radians(30)
    r=t.step(obs(cfg,joints=joints))
    assert r.state is RewardState.TARGET_APPROACH
    assert r.components["gripper_stability"] == pytest.approx(-15.0)
    joints[-1]=cfg.home_joint_positions[-1]+math.radians(45)
    r=t.step(obs(cfg,joints=joints))
    assert r.components["gripper_stability"] == pytest.approx(0.0)

def test_contact_once_per_target():
    cfg,t=tracker()
    t.step(obs(cfg))
    r=t.step(obs(cfg,grasp=(0.2,0,0.765)))
    assert r.state is RewardState.TARGET_GRASP_TRANSPORT
    assert r.components["target_contact"] == pytest.approx(5)
    r=t.step(obs(cfg,grasp=(0.15,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH
    assert r.components["target_contact_exit"] == pytest.approx(-10)
    r=t.step(obs(cfg,grasp=(0.2,0,0.765)))
    assert r.state is RewardState.TARGET_GRASP_TRANSPORT
    assert r.components["target_contact"] == 0
    # switch target forces state2
    r=t.step(obs(cfg,target_name="b",target=(0.3,0,0.765),grasp=(0.3,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH
    r=t.step(obs(cfg,target_name="b",target=(0.3,0,0.765),grasp=(0.3,0,0.765)))
    assert r.components["target_contact"] == pytest.approx(5)

def test_goal_entry_once_and_exit_repeat():
    cfg,t=tracker()
    t.step(obs(cfg))
    t.step(obs(cfg,grasp=(0.2,0,0.765))) # state3
    r=t.step(obs(cfg,target=(0,0.2,0.765),grasp=(0,0.2,0.765)))
    assert r.state is RewardState.TARGET_RELEASE_RETURN_HOME
    assert r.components["goal_entry"] == pytest.approx(10)
    r=t.step(obs(cfg,target=(0.2,0,0.765),grasp=(0.2,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH
    assert r.components["goal_exit"] == pytest.approx(-11)
    t.step(obs(cfg,target=(0.2,0,0.765),grasp=(0.2,0,0.765))) # state3
    r=t.step(obs(cfg,target=(0,0.2,0.765),grasp=(0,0.2,0.765)))
    assert r.state is RewardState.TARGET_RELEASE_RETURN_HOME
    assert r.components["goal_entry"] == 0
    r=t.step(obs(cfg,target=(0.2,0,0.765),grasp=(0.2,0,0.765)))
    assert r.components["goal_exit"] == pytest.approx(-11)

def test_state4_names():
    cfg=ReachRewardConfig()
    assert cfg.state4_gripper_open_reward_per_deg == pytest.approx(0.2)
    assert cfg.state4_gripper_open_reward_cap == pytest.approx(3)
    assert cfg.state4_not_home_penalty == pytest.approx(0.1)
    # result has new component keys
    _,t=tracker(cfg)
    r=t.step(obs(cfg,target_name=None))
    assert "state4_gripper_opening" in r.components
    assert "state4_not_home" in r.components
    assert "state3_gripper_opening" not in r.components
    assert "state3_not_home" not in r.components

def test_lift_only_state3():
    cfg=ReachRewardConfig(
        gripper_target_opening_penalty_per_deg=0.0,
        state2_counterclockwise_wrist_roll_penalty_per_deg=0.0,
    )
    t=StatefulReachReward(cfg); t.reset(joint_positions_rad=cfg.home_joint_positions)
    # Lifted cube while State2 must not get lift bonus.
    r=t.step(obs(cfg,target=(0.2,0,0.80),grasp=(0,0,0.765)))
    assert r.state is RewardState.TARGET_APPROACH
    assert r.components["target_lift"] == 0
    # Enter state3 while lifted.
    r=t.step(obs(cfg,target=(0.2,0,0.80),grasp=(0.2,0,0.80)))
    assert r.state is RewardState.TARGET_GRASP_TRANSPORT
    assert r.components["target_lift"] == pytest.approx(2)
