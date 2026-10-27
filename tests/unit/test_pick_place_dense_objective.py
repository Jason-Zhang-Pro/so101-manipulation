"""The schema-4 remaining-task cost is a bounded, explicit objective change."""
from dataclasses import replace
import pytest

from so101_capstone.rewards import (
    PickPlaceReward, PickPlaceRewardConfig, PickPlaceRewardInput, TaskPhase,
)
from test_pick_place_reward import snapshot, advance
from test_pick_place_reward_schema import state


def static_result(s, config=None, **flags):
    reward = PickPlaceReward(config or PickPlaceRewardConfig())
    reward.reset(s)
    nxt = advance(s)
    return reward.step(PickPlaceRewardInput(s, nxt, 1/30, **flags))


def remaining(s, config=None):
    return static_result(s, config).components["remaining_task"]


@pytest.mark.parametrize("invalid", [-.1, float("nan"), float("inf"), True, "0.3"])
def test_dense_cost_config_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="remaining_task_cost"):
        PickPlaceRewardConfig(remaining_task_cost_per_s=invalid)


def test_schema4_default_and_zero_ablation_are_explicit_and_round_trip():
    cfg = PickPlaceRewardConfig()
    assert cfg.remaining_task_cost_per_s == .3
    assert cfg.to_dict()["remaining_task_cost_per_s"] == .3
    assert PickPlaceRewardConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    disabled = replace(cfg, remaining_task_cost_per_s=0.)
    assert disabled.config_hash != cfg.config_hash
    assert remaining(state(disabled), disabled) == 0.
    assert cfg.success_definition_version == disabled.success_definition_version == 2


def test_schema3_hash_and_entire_reward_result_remain_historical():
    cfg = PickPlaceRewardConfig(reward_schema_version=3)
    assert cfg.remaining_task_cost_per_s is None
    assert "remaining_task_cost_per_s" not in cfg.to_dict()
    assert cfg.config_hash == "3aa8fbbb178c56fbe675320fe8d4ab2971bb9db86ae02ab54f9303bd865b762e"
    assert PickPlaceRewardConfig.from_dict(cfg.to_dict()).config_hash == cfg.config_hash
    previous = snapshot(target=True, phase=TaskPhase.APPROACH, approach_score=.2)
    current = advance(previous, approach_score=.7)
    reward = PickPlaceReward(cfg)
    reward.reset(previous)
    result = reward.step(PickPlaceRewardInput(previous,current,1/30))
    expected = {
        "reward": cfg.gamma*1.4-.4-.01,
        "reward_components": {"potential_shaping":cfg.gamma*1.4-.4,
           "success":0.,"failure":0.,"active_time":-.01,"home_position":0.,
           "home_velocity":0.,"moving_velocity":-0.,"command_change":0.},
        "potential_before":.4,"potential_after":1.4,"terminated":False,"truncated":False,
    }
    assert result.to_info() == expected
    for explicit in (0.,.3,1.):
        with pytest.raises(ValueError, match="schema 3"):
            PickPlaceRewardConfig(reward_schema_version=3,remaining_task_cost_per_s=explicit)


def test_stationary_closer_alignment_and_opening_have_lower_remaining_cost():
    cfg = PickPlaceRewardConfig(shaping_enabled=False)
    distances = [.40,.35,.20,.08,.045,.025,.02,.015,.008,.005]
    costs=[remaining(state(cfg,distance=d,opening=.6),cfg) for d in distances]
    assert all(b>a for a,b in zip(costs,costs[1:]))
    assert all(-.01<=v<0 for v in costs)
    assert remaining(state(cfg,distance=.025,opening=.6),cfg) > remaining(state(cfg,distance=.025,opening=1.8),cfg)
    from test_pick_place_pregrasp import snapshot as pregrasp_snapshot
    assert remaining(pregrasp_snapshot(distance=.025,aligned=True),cfg) > remaining(pregrasp_snapshot(distance=.025,aligned=False),cfg)
    # Remaining downstream work never disappears just because the FK is close.
    assert remaining(state(cfg,distance=0.,opening=0.),cfg) == pytest.approx(-.008)


def test_local_close_gate_guides_closure_but_never_certifies_grasp_or_success():
    cfg=PickPlaceRewardConfig()
    states=[state(cfg,distance=.005,opening=a) for a in (1.8,1.,.6,.3,0.)]
    costs=[remaining(s,cfg) for s in states]
    assert all(b>a for a,b in zip(costs,costs[1:]))
    assert all(not s.grasp_valid and not s.attempt_lifted and not s.success_event for s in states)
    assert remaining(state(cfg,distance=.08,opening=.6),cfg) > remaining(state(cfg,distance=.08,opening=0.),cfg)


def test_dense_progress_is_independent_of_potential_scale_and_shaping_ablation():
    cfg=PickPlaceRewardConfig()
    s=state(cfg,distance=.03,opening=.7)
    expected=remaining(s,cfg)
    for changed in (replace(cfg,shaping_enabled=False),replace(cfg,potential_max=0.),replace(cfg,potential_max=1000.)):
        assert remaining(s,changed)==expected
    both_off=replace(cfg,shaping_enabled=False,remaining_task_cost_per_s=0.)
    assert static_result(s,both_off).reward==pytest.approx(-.01)


def test_success_failure_and_truncation_do_not_confuse_terminal_potential_with_progress():
    cfg=PickPlaceRewardConfig()
    initial=snapshot(target=True,phase=TaskPhase.APPROACH,approach_score=.5)
    regular=static_result(initial,cfg)
    truncated=static_result(initial,cfg,truncated=True)
    assert truncated.components==regular.components
    assert truncated.potential_after==1.
    for outcome,expected_cost in (("success",0.),("task_failure",regular.components["remaining_task"])):
        reward=PickPlaceReward(cfg);reward.reset(initial)
        final=advance(initial,outcome=outcome,success_event=outcome=="success")
        result=reward.step(PickPlaceRewardInput(initial,final,1/30,terminated=True))
        assert result.potential_after==0.
        assert result.components["remaining_task"]==expected_cost
        assert result.components[outcome if outcome=="success" else "failure"]==(100 if outcome=="success" else -10)


def test_only_active_pick_place_has_remaining_task_cost_and_spawn_uses_next_state():
    cfg=PickPlaceRewardConfig()
    empty=snapshot()
    assert remaining(empty,cfg)==0.
    assert remaining(snapshot(target=True,mode="idle_completed"),cfg)==0.
    assert remaining(snapshot(mode="idle_empty"),cfg)==0.
    appeared=snapshot(1/30,target=True,phase=TaskPhase.APPROACH,approach_score=.4)
    reward=PickPlaceReward(cfg);reward.reset(empty)
    result=reward.step(PickPlaceRewardInput(empty,appeared,1/30))
    assert result.components["remaining_task"]==pytest.approx(-.01*(1-.8/10))
    assert result.components["active_time"]==0. # Existing active-time convention uses prev.


def test_missing_or_invalid_evidence_does_not_waive_cost_or_invent_progress_and_reset_clears_memory():
    cfg=PickPlaceRewardConfig()
    reward=PickPlaceReward(cfg)
    initial=snapshot(target=True,phase=TaskPhase.APPROACH,approach_score=.5)
    reward.reset(initial)
    valid=advance(initial)
    expected=reward.step(PickPlaceRewardInput(initial,valid,1/30)).components["remaining_task"]
    # Fabricated high score must not be credited while measurements are unknown.
    invalid=advance(valid,approach_score=1.)
    invalid=replace(invalid,measurements=replace(invalid.measurements,evidence_valid=False))
    result=reward.step(PickPlaceRewardInput(valid,invalid,1/30))
    assert result.components["remaining_task"]==expected
    # The tracker normally rejects disappearance; the reward itself still cannot
    # create a zero-cost escape if a missing identity is handed to it.
    missing=advance(invalid,phase=TaskPhase.WAIT_HOME)
    missing=replace(missing,measurements=replace(missing.measurements,target_name=None,target_position_world_m=None,target_motion_valid=False))
    assert reward.step(PickPlaceRewardInput(invalid,missing,1/30)).components["remaining_task"]==expected
    missing_again=advance(missing)
    assert reward.step(PickPlaceRewardInput(missing,missing_again,1/30)).components["remaining_task"]==expected
    empty=snapshot()
    reward.reset(empty)
    assert reward.step(PickPlaceRewardInput(empty,advance(empty),1/30)).components["remaining_task"]==0.


def test_stale_contact_or_target_evidence_cannot_buy_dense_progress():
    cfg=PickPlaceRewardConfig()
    for changes in ({"target_motion_valid":False},{"target_measurement_valid":False},
                    {"contact_valid":False},{"joint_measurement_valid":False},
                    {"evidence_sim_time_s":0.}):
        initial=snapshot(target=True,phase=TaskPhase.APPROACH,approach_score=.5)
        reward=PickPlaceReward(cfg);reward.reset(initial)
        nxt=advance(initial,approach_score=1.)
        nxt=replace(nxt,measurements=replace(nxt.measurements,**changes))
        result=reward.step(PickPlaceRewardInput(initial,nxt,1/30))
        assert result.components["remaining_task"]==pytest.approx(-.009)


def test_valid_regression_is_charged_and_cost_scales_with_simulated_time():
    cfg=PickPlaceRewardConfig()
    initial=snapshot(target=True,phase=TaskPhase.APPROACH,approach_score=.8)
    reward=PickPlaceReward(cfg);reward.reset(initial)
    worse=advance(initial,dt=.1,approach_score=.2)
    result=reward.step(PickPlaceRewardInput(initial,worse,.1))
    assert result.components["remaining_task"]==pytest.approx(-.3*.1*(1-.4/10))
    assert result.components["remaining_task"] < 3*remaining(initial,cfg)


def test_stage_handoffs_never_reset_remaining_cost_upward():
    cfg=PickPlaceRewardConfig()
    reach=snapshot(target=True,phase=TaskPhase.GRASP_LIFT,approach_score=1.)
    grasped=replace(reach,contact_score=1.,lift_score=0.)
    lifted=replace(grasped,lift_score=1.)
    transported=replace(lifted,phase=TaskPhase.TRANSPORT,transport_score=0.)
    arrived=replace(transported,transport_score=1.)
    placed=replace(arrived,phase=TaskPhase.PLACE_RELEASE)
    released=replace(placed,support_score=1.,valid_release=True,separation_score=1.,place_stable_duration_s=.4)
    returning=replace(released,phase=TaskPhase.RETURN_HOME,home_score=0.,home_stable_duration_s=0.)
    home=replace(returning,phase=TaskPhase.HOME_HOLD,home_score=1.,home_stable_duration_s=3.)
    states=[reach,grasped,lifted,transported,arrived,placed,released,returning,home]
    costs=[remaining(s,cfg) for s in states]
    assert all(b>=a-1e-15 for a,b in zip(costs,costs[1:]))
    assert costs[-1]==0.
    # The geometric APPROACH->GRASP candidate boundary shares one formula.
    outside=state(cfg,distance=.045001,opening=.6)
    inside=state(cfg,distance=.044999,opening=.6)
    assert 0 < remaining(inside,cfg)-remaining(outside,cfg) < 1e-6


def test_close_open_loop_and_holding_near_target_never_generate_positive_dense_income():
    cfg=PickPlaceRewardConfig()
    reward=PickPlaceReward(cfg)
    previous=state(cfg,distance=.005,opening=1.8)
    reward.reset(previous)
    total=0.
    for i,opening in enumerate((1.4,.7,.2,.7,1.4,1.8)):
        current=state(cfg,distance=.005,opening=opening,time=(i+1)/30)
        result=reward.step(PickPlaceRewardInput(previous,current,1/30))
        assert result.components["remaining_task"]<0.
        total+=cfg.gamma**i*result.reward
        previous=current
    assert total<0.
    assert static_result(state(cfg,distance=0.,opening=0.),cfg).reward<0.


def test_budget_and_objective_change_are_visible_even_without_potential_shaping():
    cfg=PickPlaceRewardConfig(shaping_enabled=False)
    far=static_result(state(cfg,distance=.4,opening=.6),cfg)
    near=static_result(state(cfg,distance=.005,opening=.6),cfg)
    assert near.reward>far.reward
    # Limited bound: target-active approach, dt=1/30, no HOME/command costs.
    worst=(cfg.active_time_cost_per_s+cfg.remaining_task_cost_per_s)/30/(1-cfg.gamma)
    assert worst==pytest.approx(6.666666666666661)
    assert worst<cfg.failure_cost
