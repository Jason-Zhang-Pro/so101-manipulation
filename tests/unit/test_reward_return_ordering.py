"""Discounted return audits, including terminal and truncation boundary terms."""
from dataclasses import replace

import numpy as np
import pytest

from so101_capstone.rewards import (
    PickPlaceReward, PickPlaceRewardConfig, PickPlaceRewardInput,
    TaskMeasurements, TaskPhase, TaskSnapshot,
)


def run_trace(length, *, outcome="success", cfg=None, start_potential_score=.4):
    cfg = cfg or PickPlaceRewardConfig()
    m = TaskMeasurements(0, cfg.home_joint_positions_rad, (0,) * 6, (0, 0, 1), np.eye(3),
                         target_name="cube", target_position_world_m=(0, 0, .765),
                         target_motion_valid=True, contact_valid=True)
    initial = TaskSnapshot(m, "return-audit", "pick_place", TaskPhase.APPROACH,
                           approach_score=start_potential_score, home_score=1)
    evaluator = PickPlaceReward(cfg)
    evaluator.reset(initial)
    prev = initial
    shaped_return = base_return = 0.0
    last = None
    for i in range(length):
        final = i == length-1
        p = (i+1)/length
        phase = TaskPhase.APPROACH if p < .2 else TaskPhase.GRASP_LIFT if p < .4 else TaskPhase.TRANSPORT if p < .6 else TaskPhase.PLACE_RELEASE if p < .8 else TaskPhase.HOME_HOLD
        nxt = replace(prev, measurements=replace(m, sim_time_s=(i+1)/30), phase=phase,
                      approach_score=p, contact_score=p, lift_score=p, transport_score=p,
                      support_score=p, separation_score=p, valid_release=p >= .8,
                      place_stable_duration_s=.4*p, home_stable_duration_s=3*p,
                      outcome=outcome if final else "running", success_event=bool(final and outcome == "success"))
        last = evaluator.step(PickPlaceRewardInput(prev, nxt, 1/30,
                                                  terminated=final and outcome != "running",
                                                  truncated=final and outcome == "running"))
        shaped_return += cfg.gamma**i*last.reward
        base_return += cfg.gamma**i*sum(value for key,value in last.components.items() if key != "potential_shaping")
        prev = nxt
    return shaped_return, base_return, evaluator.potential(initial), last.potential_after


@pytest.mark.parametrize("outcome", ["success", "task_failure"])
@pytest.mark.parametrize("length", [1, 17, 300])
def test_whole_episode_telescope_preserves_base_objective(outcome, length):
    shaped, base, initial, final = run_trace(length, outcome=outcome)
    assert final == 0
    assert shaped == pytest.approx(base-initial, abs=1e-10)


def test_truncated_prefix_retains_boundary_and_bootstrap_cancels_it():
    cfg = PickPlaceRewardConfig()
    n = 53
    shaped, base, initial, final = run_trace(n, outcome="running", cfg=cfg)
    assert final > 0
    assert shaped == pytest.approx(base-initial+cfg.gamma**n*final, abs=1e-10)
    arbitrary_base_value = 12.5
    shaped_value = arbitrary_base_value-final
    assert shaped+cfg.gamma**n*shaped_value == pytest.approx(base+cfg.gamma**n*arbitrary_base_value-initial)


@pytest.mark.parametrize("schema", [3, 4])
def test_candidate_budget_fast_success_slow_success_wait_and_failure(schema):
    cfg = PickPlaceRewardConfig(reward_schema_version=schema)
    fast = run_trace(300, cfg=cfg)[0]
    slow = run_trace(600, cfg=cfg)[0]
    initial = .8
    remaining_cost = 0. if schema == 3 else cfg.remaining_task_cost_per_s/30*(1-initial/10)
    wait_forever = (-cfg.active_time_cost_per_s/30-remaining_cost)/(1-cfg.gamma)-initial
    immediate_failure = run_trace(1, outcome="task_failure", cfg=cfg)[0]
    assert fast > slow > wait_forever > immediate_failure


@pytest.mark.parametrize("failure_cost", [5, 10, 20])
def test_attempt_vs_wait_probability_threshold_is_explicit(failure_cost):
    # This closed form assumes the historical constant active-time objective.
    # Schema 4 has an additional state-dependent cost audited above.
    cfg = PickPlaceRewardConfig(reward_schema_version=3, failure_cost=failure_cost)
    success = run_trace(300, outcome="success", cfg=cfg)[0]
    failure = run_trace(300, outcome="task_failure", cfg=cfg)[0]
    wait = -.01/(1-cfg.gamma)-.8
    threshold = (failure_cost-.01*cfg.gamma/(1-cfg.gamma))/(100+failure_cost)
    assert threshold*success+(1-threshold)*failure == pytest.approx(wait, abs=1e-9)
    p = threshold+.02
    assert p*success+(1-p)*failure > wait
    p = max(0, threshold-.01)
    assert p*success+(1-p)*failure < wait


def test_perfect_shaped_value_cancels_potential_in_td_residual_and_gae():
    gamma, lam = .997, .99
    base_reward = np.array([-.01, -.01, -.01, 100-.01])
    phi = np.array([.8, 2.5, 6.0, 9.5, 0.0])
    base_value = np.array([55.0, 56.0, 60.0, 90.0, 0.0])
    shaped_reward = base_reward+gamma*phi[1:]-phi[:-1]
    shaped_value = base_value-phi
    base_delta = base_reward+gamma*base_value[1:]-base_value[:-1]
    shaped_delta = shaped_reward+gamma*shaped_value[1:]-shaped_value[:-1]
    np.testing.assert_allclose(shaped_delta, base_delta, atol=1e-12)
    def gae(delta):
        result = np.zeros_like(delta)
        tail = 0.0
        for i in reversed(range(len(delta))):
            tail = delta[i]+gamma*lam*tail
            result[i] = tail
        return result
    np.testing.assert_allclose(gae(shaped_delta), gae(base_delta), atol=1e-12)


def test_potential_does_not_make_endless_constant_positive_score_profitable():
    cfg = PickPlaceRewardConfig()
    phi = 9.0
    reward = cfg.gamma*phi-phi-.01
    assert reward < 0
    assert reward/(1-cfg.gamma) == pytest.approx(-.01/(1-cfg.gamma)-phi)
