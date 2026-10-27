"""A certified 8 cm / 0.1 s lift pays once, independently of final reach."""
from dataclasses import replace
import math
from pathlib import Path

import pytest

from so101_capstone.rewards.s1_lift_approach import S1LiftApproachConfig
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_s1_lift_approach_reward import DT, Trace
from test_s1_lift_approach_stages import LiftTrace


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PROFILE = CONFIG_DIR / "act_ppo_stages_s1_lift_completion.json"
PREVIOUS_PROFILE = CONFIG_DIR / "act_ppo_stages_s1_posture_guard.json"
BONUS_KEY = "s1_lift_completed"


def trace():
    return Trace(StageTrainingConfig.load(PROFILE))


def test_s1_profile_adds_bonus_disables_rotation_and_preserves_critic_and_old_hashes():
    current = StageTrainingConfig.load(PROFILE)
    previous = StageTrainingConfig.load(PREVIOUS_PROFILE)
    # The current profile also opts into relaxed S0 early success. Isolate
    # the S1 contract here; S0 boundaries have dedicated integration tests.
    serialized = replace(current, s0_home_hold_s=previous.s0_home_hold_s,
        s0_home_joint_tolerance_rad=None, s0_home_velocity_tolerance_rad_s=None,
        s0_success_mode="window_end").to_dict()
    assert serialized["process_reward"]["s1_lift_approach"].pop("lift_completion_bonus") == 5.
    assert serialized["process_reward"]["s1_lift_approach"].pop("wrist_rotation_reward_enabled") is False
    assert serialized == previous.to_dict()
    assert StageTrainingConfig.from_dict(current.to_dict()) == current
    assert current.config_hash != previous.config_hash
    assert current.process_reward.config_hash != previous.process_reward.config_hash
    assert current.process_reward.context_dim == previous.process_reward.context_dim == 34
    assert current.process_reward.context_schema_version == previous.process_reward.context_schema_version == 5
    assert previous.config_hash == "9564d5364c87362865a670a0d64c059d49dbbe683f1d12e301523891396fa93e"
    assert previous.process_reward.config_hash == "e7fdf130e32c24188d5f5a441f1b9132b413b70390ea0e7ecce235c71af90624"
    explicit_zero = previous.to_dict()
    explicit_zero["process_reward"]["s1_lift_approach"]["lift_completion_bonus"] = 0.
    restored = StageTrainingConfig.from_dict(explicit_zero)
    assert restored.to_dict() == previous.to_dict()
    assert restored.config_hash == previous.config_hash
    assert "lift_completion_bonus" not in S1LiftApproachConfig().to_dict()


@pytest.mark.parametrize("bonus", [-1., math.nan, math.inf, -math.inf, True, False, "5", None])
def test_invalid_completion_bonus_is_rejected(bonus):
    with pytest.raises(ValueError, match="lift_completion_bonus"):
        S1LiftApproachConfig.from_dict({"lift_completion_bonus": bonus})


def test_exact_height_and_full_fresh_dwell_pay_once_without_replacing_progress():
    current = trace()
    # Isolate the completion bonus from the current profile's rotation switch.
    previous = Trace(replace(current.stages, process_reward=replace(current.stages.process_reward,
        s1_lift_approach=replace(current.stages.process_reward.s1_lift_approach, lift_completion_bonus=0.0))))
    payments = []
    for index in range(9):
        result = current.step(**current.pose(clearance=.08))
        old_result = previous.step(**previous.pose(clearance=.08))
        expected = 5. if index == 3 else 0.
        payments.append(result.components[BONUS_KEY])
        assert result.components[BONUS_KEY] == expected
        assert result.reward - old_result.reward == pytest.approx(expected)
        without_bonus = {key: value for key, value in result.components.items() if key != BONUS_KEY}
        assert without_bonus == pytest.approx(old_result.components)
        assert current.reward.critic_context() == pytest.approx(previous.reward.critic_context())
        if index < 3:
            assert not current.trace.info["stage_s1_lift_completed"]
        else:
            assert current.trace.info["stage_s1_lift_completed"]
            assert current.trace.info["stage_s1_lift_hold_s"] == pytest.approx(.1)
    assert sum(payments) == 5.


def test_subthreshold_height_and_incomplete_dwell_do_not_pay():
    tr = trace()
    for _ in range(6):
        result = tr.step(**tr.pose(clearance=.0799))
        assert result.components[BONUS_KEY] == 0.
        assert tr.trace.info["stage_s1_lift_hold_s"] == 0.
    for _ in range(3):
        result = tr.step(**tr.pose(clearance=.08))
        assert result.components[BONUS_KEY] == 0.
    assert tr.trace.info["stage_s1_lift_hold_s"] == pytest.approx(2 * DT)
    assert tr.step(**tr.pose(clearance=.08)).components[BONUS_KEY] == 5.


def test_dipping_before_completion_restarts_continuous_hold():
    tr = trace()
    for _ in range(3):
        assert tr.step(**tr.pose(clearance=.08)).components[BONUS_KEY] == 0.
    assert tr.step(**tr.pose(clearance=.079)).components[BONUS_KEY] == 0.
    assert tr.trace.info["stage_s1_lift_hold_s"] == 0.
    for _ in range(3):
        assert tr.step(**tr.pose(clearance=.08)).components[BONUS_KEY] == 0.
    assert tr.step(**tr.pose(clearance=.08)).components[BONUS_KEY] == 5.


def test_descending_and_lifting_again_cannot_repeat_completion_bonus():
    tr = trace()
    assert tr.lift().components[BONUS_KEY] == 5.
    for height in [.03, .08, .08, .08, .08, .03, .09, .09, .09, .09]:
        result = tr.step(**tr.pose(clearance=height))
        assert result.components[BONUS_KEY] == 0.
        assert tr.trace.info["stage_s1_lift_completed"]


@pytest.mark.parametrize("kind", ["duplicate", "gap", "contact"])
def test_unqualified_evidence_cannot_complete_lift_and_recovery_pays_only_once(kind):
    tr = trace()
    for _ in range(2):
        tr.step(**tr.pose(clearance=.08))
    if kind == "duplicate":
        result = tr.step(**tr.pose(clearance=.08),
                         evidence_sim_time_s=tr.trace.snapshot.evidence_sim_time_s)
    elif kind == "gap":
        result = tr.step(dt=.2, **tr.pose(clearance=.08))
    else:
        result = tr.step(**tr.pose(clearance=.08), left_contact=True)
    assert result.components[BONUS_KEY] == 0.
    assert not tr.trace.info["stage_s1_lift_completed"]
    payments = [tr.step(**tr.pose(clearance=.08)).components[BONUS_KEY] for _ in range(6)]
    assert sum(payments) == 5.
    assert payments.count(5.) == 1


def test_physical_failure_on_completion_step_never_receives_bonus():
    tr = trace()
    for _ in range(3):
        tr.step(**tr.pose(clearance=.08))
    result = tr.step(**tr.pose(clearance=.08), failure_reason="collision")
    assert tr.trace.info["stage_s1_lift_completed"]
    assert tr.trace.info["stage_result"] == "FAILURE"
    assert result.terminated and not result.truncated
    assert result.components[BONUS_KEY] == 0.
    assert result.components["failure"] == -tr.trace.reward.failure_cost


def test_reward_reset_allows_a_new_episode_to_earn_one_completion_bonus():
    tr = trace()
    assert tr.lift().components[BONUS_KEY] == 5.
    tr.trace = LiftTrace(config=tr.stages)
    tr.reward.reset(tr.trace.snapshot, tr.trace.info)
    assert tr.lift().components[BONUS_KEY] == 5.
    assert tr.step(**tr.pose(clearance=.08)).components[BONUS_KEY] == 0.


def test_final_reach_keeps_its_separate_success_bonus_and_bootstrap_boundary():
    tr = trace()
    lifted = tr.lift()
    assert lifted.components[BONUS_KEY] == 5.
    assert lifted.components["s1_success"] == 0.
    assert not lifted.truncated and not lifted.terminated
    reached = tr.step(**tr.pose(x=0., clearance=.025))
    assert tr.trace.info["stage_result_reason"] == "reach_distance"
    assert reached.components["s1_success"] == 10.
    assert reached.components[BONUS_KEY] == 0.
    assert reached.truncated and not reached.terminated
    assert reached.potential_after > 0.


@pytest.mark.parametrize("budget, expected", [(6 * DT, 5.), (5.5 * DT, 0.)])
def test_completion_at_deadline_pays_but_a_late_guard_step_cannot_pay(budget, expected):
    tr = Trace(replace(StageTrainingConfig.load(PROFILE), s1_budget_s=budget))
    for _ in range(2):
        tr.step(**tr.pose(clearance=.03))
    for _ in range(4):
        result = tr.step(**tr.pose(clearance=.08))
    assert tr.trace.info["stage_s1_lift_completed"]
    assert tr.trace.info["stage_elapsed_s"] == pytest.approx(6 * DT)
    assert tr.trace.info["stage_timeout"]
    assert result.truncated and not result.terminated
    assert result.components[BONUS_KEY] == expected
