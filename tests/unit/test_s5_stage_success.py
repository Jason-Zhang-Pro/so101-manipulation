"""S5 reuses S0 HOME success, with genuine grasp/carry/release evidence."""
from dataclasses import replace
import math
from pathlib import Path

import pytest

from so101_capstone.rewards.pick_place_reward import PickPlaceRewardInput
from so101_capstone.rewards.s5_return_home_reward import (
    S5ReturnHomeReward, S5ReturnHomeRewardConfig,
)
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.rewards.s0_home_success import S0HomeSuccess
from test_task_state import Trace
from test_training_stages import StageTrace


DT = 1 / 30


def current_stage_config(profile="act_ppo_stages_s1_lift_completion.json"):
    return StageTrainingConfig.load(Path(__file__).resolve().parents[2] / "config" / profile)


class S5Trace:
    def __init__(self, config=None, *, stage_config=None):
        self.trace = Trace()
        self.trace.acquire_and_lift()
        self.trace.step((.25, 0., .808), contact=True, repeat=3, at_home=False, dt=DT)
        self.trace.step((.25, 0., .768), contact=True, at_home=False, dt=DT)
        self.tool = (.305, 0., .768)  # 5.5 cm: deliberately below the legacy 6 cm gate.
        self.trace.step(contact=False, at_home=False, dt=DT,
                        grasp_position_world_m=self.tool)
        assert self.trace.snapshot.valid_release and self.trace.snapshot.delivery_valid
        assert not self.trace.snapshot.placed_stable
        self.stage_config = stage_config if stage_config is not None else current_stage_config()
        self.reward = S5ReturnHomeReward(self.trace.config, config,
                                       stage_config=self.stage_config)
        self.reward.reset(self.trace.snapshot)

    def joints(self, error_deg=7.0):
        return [q + math.radians(error_deg) for q in self.trace.config.home_joint_positions_rad]

    def step(self, *, dt=DT, joints=None, speeds=None, **measurements):
        previous = self.trace.snapshot
        current = self.trace.step(contact=False, at_home=False, dt=dt,
            joint_positions_rad=self.joints() if joints is None else joints,
            joint_velocities_rad_s=(.4,) * 6 if speeds is None else speeds,
            grasp_position_world_m=self.tool, **measurements)
        return self.reward.step(PickPlaceRewardInput(previous, current, dt,
            terminated=current.outcome != "running", truncated=False))

    def hold(self, count=31, **measurements):
        return [self.step(**measurements) for _ in range(count)]


def test_s5_success_uses_six_position_errors_and_one_second_without_velocity_gate():
    t = S5Trace()
    results = t.hold()
    assert results[-1].stage_complete
    assert results[-1].metrics["home_held_hold_s"] == pytest.approx(1.0)
    assert results[-1].metrics["home_velocity_required"] == 0
    assert results[-1].metrics["home_max_error_rad"] == pytest.approx(math.radians(7))
    assert sum(r.stage_success_event for r in results) == 1
    assert sum(r.components["s5_home_held_complete"] for r in results) == t.reward.config.home_held_bonus
    assert sum(r.components["s5_stage_complete"] for r in results) == t.reward.tuning.stage_completion_bonus
    assert all(r.components["s5_home_static_complete"] == 0 for r in results)
    # The draft does not silently use the full task's 6 cm/0.4 s placement or
    # 1 degree/low-speed/3 second HOME success requirements.
    assert not t.trace.snapshot.placed_stable
    assert not t.trace.snapshot.robot_at_home and not t.trace.snapshot.robot_static
    assert t.trace.snapshot.home_stable_duration_s == 0
    assert t.trace.snapshot.outcome == "running"


def test_s5_exact_hold_boundary_does_not_round_point999_seconds_up_to_success():
    t = S5Trace()
    assert t.step().metrics["home_held_hold_s"] == 0
    for _ in range(19):
        assert not t.step(dt=.05).stage_complete
    before = t.step(dt=.049)
    assert before.metrics["home_held_hold_s"] == pytest.approx(.999)
    assert not before.stage_complete
    complete = t.step(dt=.001)
    assert complete.metrics["home_held_hold_s"] == pytest.approx(1.0)
    assert complete.stage_complete


@pytest.mark.parametrize("profile", ["act_ppo_stages_s1_lift_completion.json",
                                    "act_ppo_stages_s1_lift_approach.json"])
def test_s5_home_contract_is_the_same_resolved_configuration_used_by_s0(profile):
    config = current_stage_config(profile)
    s0 = StageTrace("S0", config=config)
    s5 = S5Trace(stage_config=config)
    assert isinstance(s0.evaluator._s0_home, S0HomeSuccess)
    assert isinstance(s5.reward.home_success, S0HomeSuccess)
    assert s0.evaluator._s0_home.config is config
    assert s5.reward.home_success.config is config
    assert s5.reward.home_success.joint_tolerance_rad == s0.evaluator._s0_home.joint_tolerance_rad
    assert s5.reward.home_success.hold_required_s == s0.evaluator._s0_home.hold_required_s
    assert s5.reward.home_success.velocity_required == s0.evaluator._s0_home.velocity_required


def paired_home_traces(config):
    s0 = StageTrace("S0", config=config)
    s5 = S5Trace(stage_config=config)
    # Both stage clocks start with the same certified HOME endpoint.
    s5.step(joints=s5.trace.config.home_joint_positions_rad, speeds=(0.,) * 6)
    s5.reward.reset(s5.trace.snapshot, ledger=s5.reward.ledger)
    return s0, s5


def step_pair(s0, s5, *, dt=.05, error_deg=2.5, speeds=(.4,) * 6):
    joints = s5.joints(error_deg)
    zero = s0.step(dt=dt, joint_positions_rad=joints, joint_velocities_rad_s=speeds)
    five = s5.step(dt=dt, joints=joints, speeds=speeds)
    assert five.metrics["home_held_hold_s"] == pytest.approx(zero["stage_home_hold_s"])
    assert five.stage_complete == zero["stage_success"]
    return zero, five


def test_one_changed_s0_configuration_updates_s5_threshold_and_hold_without_s5_overrides():
    config = replace(current_stage_config(), s0_home_joint_tolerance_rad=math.radians(3),
                     s0_home_hold_s=.2)
    s0, s5 = paired_home_traces(config)
    zero, five = step_pair(s0, s5, error_deg=3)
    assert not zero["stage_home_hold_candidate"] and not five.metrics["home_hold_candidate"]
    assert five.metrics["home_position_tolerance_rad"] == math.radians(3)
    assert five.metrics["home_hold_required_s"] == .2
    for index in range(5):
        zero, five = step_pair(s0, s5)
        assert five.stage_complete == (index == 4)
    assert zero["stage_result"] == "SUCCESS"
    assert five.metrics["home_velocity_required"] == 0


def test_s0_velocity_mode_and_threshold_are_also_used_by_s5():
    config = replace(current_stage_config(), s0_success_mode="home_hold",
                     s0_home_joint_tolerance_rad=math.radians(3),
                     s0_home_velocity_tolerance_rad_s=.01, s0_home_hold_s=.2)
    s0, s5 = paired_home_traces(config)
    for _ in range(5):
        zero, five = step_pair(s0, s5, speeds=(.02,) * 6)
        assert zero["stage_home_hold_s"] == 0
        assert five.metrics["home_velocity_required"] == 1
        assert not five.stage_complete
    for index in range(5):
        zero, five = step_pair(s0, s5, speeds=(0.,) * 6)
        assert five.stage_complete == (index == 4)


def test_s5_window_end_mode_waits_for_the_same_s0_budget():
    config = replace(current_stage_config(), s0_success_mode="window_end",
                     s0_budget_s=.4, s0_home_hold_s=.2)
    s0, s5 = paired_home_traces(config)
    for index in range(8):
        zero, five = step_pair(s0, s5, speeds=(0.,) * 6)
        assert five.stage_complete == (index == 7)
        if index == 4:
            assert five.metrics["home_held_hold_s"] >= .2
            assert not five.stage_complete
    assert zero["stage_result"] == "SUCCESS"


def test_s5_early_success_and_late_home_hold_use_the_same_s0_deadline():
    config = replace(current_stage_config(), s0_budget_s=.3, s0_home_hold_s=.2)
    s0, s5 = paired_home_traces(config)
    for index in range(4):
        zero, five = step_pair(s0, s5)
        assert five.stage_complete == (index == 3)
    assert zero["stage_elapsed_s"] < config.s0_budget_s

    s0, s5 = paired_home_traces(config)
    for _ in range(3):
        zero, five = step_pair(s0, s5, error_deg=15)
    for _ in range(3):
        zero, five = step_pair(s0, s5)
    assert zero["stage_result"] == "TIMEOUT"
    assert s5.reward.home_success.decision()[0] == "TIMEOUT"
    assert not five.stage_complete
    # Finishing a long-enough hold after collection's deadline cannot make
    # the already timed-out S5 segment successful retroactively.
    for _ in range(5):
        assert not s5.step(dt=.05, joints=s5.joints(2.5)).stage_complete


def test_s5_placement_gate_cannot_accumulate_s0_home_time_while_object_is_moving():
    config = replace(current_stage_config(), s0_home_joint_tolerance_rad=math.radians(3),
                     s0_home_hold_s=.2)
    s0, s5 = paired_home_traces(config)
    for _ in range(4):
        zero = s0.step(dt=.05, joint_positions_rad=s5.joints(2.5),
                       joint_velocities_rad_s=(.4,) * 6)
        five = s5.step(dt=.05, joints=s5.joints(2.5),
                       target_linear_velocity_m_s=(.02, 0., 0.))
        assert five.metrics["home_held_hold_s"] == 0
        assert not five.stage_complete
    assert zero["stage_success"]
    for index in range(5):
        five = s5.step(dt=.05, joints=s5.joints(2.5))
        assert five.stage_complete == (index == 4)


def test_s5_requires_the_resolved_s0_configuration_and_has_no_independent_home_thresholds():
    with pytest.raises(TypeError, match="stage_config"):
        S5ReturnHomeReward(Trace().config)
    for name in ("home_position_tolerance_rad", "home_hold_min_s"):
        with pytest.raises(TypeError):
            S5ReturnHomeRewardConfig(**{name: .1})


@pytest.mark.parametrize("joint,sign", [(0, -1), (0, 1), (5, -1), (5, 1)])
def test_s0_and_s5_home_dwell_match_with_reused_evidence_and_position_excursion(joint, sign):
    config_path = Path(__file__).resolve().parents[2] / "config" / "act_ppo_stages_s1_lift_completion.json"
    s0 = StageTrace("S0", config=StageTrainingConfig.load(config_path))
    s5 = S5Trace()
    # Establish the same HOME baseline before timing either stage.
    s5.step(joints=s5.trace.config.home_joint_positions_rad)
    s5.reward.reset(s5.trace.snapshot, ledger=s5.reward.ledger)
    joints = s5.joints()
    for _ in range(18):
        zero = s0.step(dt=DT, joint_positions_rad=joints, joint_velocities_rad_s=(.4,) * 6)
        five = s5.step(joints=joints)
        assert five.metrics["home_held_hold_s"] == pytest.approx(zero["stage_home_hold_s"])
        assert five.stage_complete == zero["stage_success"]
    assert five.metrics["home_held_hold_s"] == pytest.approx(.6)
    before_potential = s5.reward.reward_state()["potential"]
    # Reused feedback cannot add time, but crossing the strict 8 degree
    # boundary must still cancel the existing continuous hold in both stages.
    joints[joint] = s5.trace.config.home_joint_positions_rad[joint] + sign * math.radians(8)
    zero = s0.step(dt=DT, joint_positions_rad=joints,
                   evidence_sim_time_s=s0.snapshot.evidence_sim_time_s)
    five = s5.step(joints=joints, evidence_sim_time_s=s5.trace.snapshot.evidence_sim_time_s)
    assert zero["stage_home_hold_s"] == five.metrics["home_held_hold_s"] == 0
    assert not zero["stage_success"] and not five.stage_complete
    assert five.potential_after == before_potential
    assert all(value == 0 for key, value in five.components.items()
               if "_band_" in key or key.endswith("_complete") or key.endswith("_dwell"))
    # A full new second is needed after recovery, independent of velocity.
    for index in range(31):
        speeds = ((.5 if index % 2 else 2.),) * 6
        zero = s0.step(dt=DT, joint_positions_rad=s5.joints(), joint_velocities_rad_s=speeds)
        five = s5.step(speeds=speeds)
        assert five.metrics["home_held_hold_s"] == pytest.approx(zero["stage_home_hold_s"])
        assert five.stage_complete == zero["stage_success"] == (index == 30)


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("sign", [-1, 1])
def test_s5_exact_eight_degree_boundary_fails_for_every_joint_including_gripper(joint, sign):
    t = S5Trace()
    joints = t.joints()
    joints[joint] = (t.trace.config.home_joint_positions_rad[joint]
                     + sign * t.reward.home_success.joint_tolerance_rad)
    results = t.hold(joints=joints)
    assert not any(r.stage_complete for r in results)
    assert all(r.metrics["home_hold_candidate"] == 0 for r in results)
    assert results[-1].metrics["home_held_hold_s"] == 0
    joints[joint] = (t.trace.config.home_joint_positions_rad[joint]
                     + sign * math.radians(7.999))
    assert t.hold(joints=joints)[-1].stage_complete


def test_s5_position_excursion_restarts_dwell_instead_of_accumulating_disjoint_holds():
    t = S5Trace()
    partial = t.hold(count=20)[-1]
    assert 0 < partial.metrics["home_held_hold_s"] < 1
    joints = t.joints()
    joints[-1] = t.trace.config.home_joint_positions_rad[-1] + math.radians(8)
    assert t.step(joints=joints).metrics["home_held_hold_s"] == 0
    results = t.hold()
    assert not any(r.stage_complete for r in results[:-1])
    assert results[-1].stage_complete


@pytest.mark.parametrize("interruption", [
    {"left_contact": True},
    {"right_contact": True},
    {"target_linear_velocity_m_s": (.02, 0., 0.)},
    {"target_angular_velocity_rad_s": (0., 0., .2)},
])
def test_s5_placement_interruption_resets_home_hold(interruption):
    t = S5Trace()
    assert 0 < t.hold(count=20)[-1].metrics["home_held_hold_s"] < 1
    broken = t.step(**interruption)
    assert not broken.stage_complete
    assert broken.metrics["home_held_hold_s"] == 0
    results = t.hold()
    assert not any(r.stage_complete for r in results[:-1])
    assert results[-1].stage_complete


def test_s5_distance_below_five_cm_resets_placement_dwell():
    t = S5Trace()
    t.hold(count=20)
    t.tool = (.299, 0., .768)
    broken = t.step()
    assert broken.metrics["home_held_hold_s"] == 0
    assert not broken.stage_complete
    t.tool = (.305, 0., .768)
    assert t.hold()[-1].stage_complete


def test_s5_disturbing_placed_object_revokes_success_and_does_not_repay_completion():
    t = S5Trace()
    first = t.hold()[-1]
    assert first.stage_success_event
    broken = t.step(left_contact=True)
    assert not broken.stage_complete
    results = t.hold()
    assert results[-1].stage_complete
    assert not any(r.stage_success_event for r in results)
    assert all(r.components["s5_home_held_complete"] == 0 for r in results)


@pytest.mark.parametrize("name", ["release_separation_m"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_s5_invalid_success_config_is_rejected(name, value):
    with pytest.raises(ValueError):
        S5ReturnHomeRewardConfig(**{name: value})
