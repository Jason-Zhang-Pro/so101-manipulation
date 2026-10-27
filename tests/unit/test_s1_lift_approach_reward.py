"""Evidence-qualified small progress guides lift, wrist turn and descent."""
from pathlib import Path
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import PickPlaceRewardInput
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.rewards.stage_process_reward import StageProcessRewardConfig
from so101_capstone.rewards.s1_lift_approach import S1PostureRegressionConfig
from so101_capstone.rewards.s1_lift_approach_reward import S1LiftApproachReward, APPROACH_INDICES, NAMES, _band_credit
from test_s1_lift_approach_stages import LiftTrace


PROFILE = Path(__file__).resolve().parents[2] / "config/act_ppo_stages_s1_posture_guard.json"
DT = 1 / 30


class Trace:
    def __init__(self, config=None):
        self.stages = config or StageTrainingConfig.load(PROFILE)
        self.trace = LiftTrace(config=self.stages)
        self.reward = S1LiftApproachReward(self.stages.process_reward, self.stages, self.trace.reward)
        self.reward.reset(self.trace.snapshot, self.trace.info)

    def pose(self, *, x=.1, clearance=.03, roll=0., opening=.7, tilt=0.):
        joints = list(self.trace.reward.home_joint_positions_rad)
        joints[4] += roll
        joints[-1] += opening
        a = math.radians(tilt)
        rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                             [-math.sin(a), 0, math.cos(a)]]) @ np.diag([1., -1., -1.])
        return dict(grasp_position_world_m=(x, 0., .75 + clearance),
                    joint_positions_rad=joints, grasp_rotation_world=rotation)

    def step(self, *, dt=DT, **measurements):
        prev = self.trace.snapshot
        info = self.trace.step(dt=dt, **measurements)
        return self.reward.step(PickPlaceRewardInput(
            prev, self.trace.snapshot, dt,
            terminated=self.trace.snapshot.outcome != "running",
            truncated=bool(info["stage_cut"])), info)

    def lift(self, **overrides):
        result = None
        for _ in range(4):
            result = self.step(**self.pose(clearance=.08, **overrides))
        assert self.trace.info["stage_s1_phase"] == "APPROACH"
        return result


def credit(result, name):
    return sum(value for key, value in result.components.items() if key.startswith(f"s1_{name}_band_"))


def test_decimal_band_boundary_does_not_create_zero_width_nan_interval():
    with np.errstate(divide="raise", invalid="raise"):
        values = _band_credit(.07, .07, .01, 2.)
    assert values.shape == (7,)
    assert values == pytest.approx(np.full(7, 2.))


def test_dense_cost_budget_is_bounded_below_failure_without_saturating_progress():
    tr = Trace()
    assert tr.reward.state_cost_scale == pytest.approx(.4)
    cost_keys = ("s1_time", "s1_remaining", "s1_distance_remaining",
                 "s1_lift_remaining", "s1_roll_remaining", "s1_corridor_remaining")
    tr.lift()
    total = 0.
    while not tr.trace.info["stage_cut"]:
        result = tr.step(**tr.pose(x=.1, clearance=0., roll=1.), left_contact=True)
        cost = sum(result.components[key] for key in cost_keys)
        assert -.9 * DT - 1e-12 <= cost <= 0
        total += cost
    assert total >= -9. - 1e-9
    assert -9. > -tr.trace.reward.failure_cost
    with pytest.raises(ValueError, match="exhausts"):
        S1LiftApproachReward(replace(tr.stages.process_reward, s1_time_cost_per_s=1.),
                            tr.stages, tr.trace.reward)


def test_schema_four_roundtrip_and_preserved_old_profile():
    new = StageTrainingConfig.load(PROFILE)
    assert StageTrainingConfig.from_dict(new.to_dict()) == new
    assert new.process_reward.context_dim == 34
    assert new.process_reward.context_schema_version == 5
    historical = new.to_dict()
    historical.pop("s1_success_mode")
    historical["s1_reach_hold_s"] = .15
    historical["process_reward"]["s1_lift_approach"].pop("posture_regression")
    historical = StageTrainingConfig.from_dict(historical)
    assert historical.process_reward.context_dim == 26
    assert historical.process_reward.context_schema_version == 4
    assert "posture_regression" not in historical.to_dict()["process_reward"]["s1_lift_approach"]
    assert historical.config_hash == "4a0ef2292039ce21c87c5f96d8a6a184b6d36327cd7577284c71d9355c48d00a"
    assert historical.process_reward.config_hash == "067306ff8b80f3cbf5e0152dbea53ad179ec20a5154dcd264df9eb843a09a414"
    old = StageTrainingConfig.load(PROFILE.with_name("act_ppo_stages_s1_progress.json"))
    assert old.process_reward.schema_version == 3
    assert "s1_lift_approach" not in old.process_reward.to_dict()
    assert old.config_hash == "4db8bd904766556fbe5170f289b3c6634f6d250cd323d677c46df84d6cd7012e"
    assert new.config_hash != old.config_hash


def test_new_geometry_cannot_silently_enter_old_process_versions():
    new = StageTrainingConfig.load(PROFILE).process_reward.to_dict()
    for version in (1, 2, 3):
        with pytest.raises(ValueError, match="requires schema_version=4"):
            StageProcessRewardConfig.from_dict({**new, "schema_version": version})
    with pytest.raises(ValueError, match="requires s1_lift_approach"):
        StageProcessRewardConfig.from_dict({**new, "s1_lift_approach": None})


def test_every_millimetre_upwards_gets_positive_credit_without_distance_conflict():
    tr = Trace()
    before_distance = tr.trace.info["stage_reach_distance_m"]
    for height in np.arange(.031, .0801, .001):
        result = tr.step(**tr.pose(clearance=float(height)))
        assert credit(result, "lift") == pytest.approx(.4)
        assert credit(result, "distance") == result.components["s1_best_distance"] == 0
        assert result.components["s1_distance_remaining"] == 0
        assert result.components["s1_height_progress"] == 0
        assert result.reward > 0
    assert tr.trace.info["stage_reach_distance_m"] > before_distance


def test_height_oscillation_and_overshoot_cannot_repeat_lift_credit():
    tr = Trace()
    first = tr.step(**tr.pose(clearance=.075))
    assert credit(first, "lift") == pytest.approx(18.)
    for _ in range(3):
        down = tr.step(**tr.pose(clearance=.06))
        up = tr.step(**tr.pose(clearance=.075))
        assert credit(down, "lift") == credit(up, "lift") == 0
        assert down.components["s1_lift_progress"] < 0 < up.components["s1_lift_progress"]
    assert credit(tr.step(**tr.pose(clearance=.15)), "lift") == pytest.approx(2.)


def test_clockwise_turn_rewards_each_degree_and_penalizes_wrong_direction():
    tr = Trace()
    wrong = tr.step(**tr.pose(roll=math.radians(10)))
    assert credit(wrong, "roll") == 0
    assert wrong.components["s1_roll_progress"] < 0
    tr.step(**tr.pose())
    total = 0
    for angle in range(1, 91):
        result = tr.step(**tr.pose(roll=-math.radians(angle)))
        assert credit(result, "roll") == pytest.approx(.2)
        total += credit(result, "roll")
    assert total == pytest.approx(18.)
    held = tr.step(**tr.pose(roll=-math.pi / 2))
    assert credit(held, "roll") == held.components["s1_roll_progress"] == 0
    assert held.components["s1_roll_remaining"] == 0
    drift = tr.step(**tr.pose(roll=-math.pi / 3))
    assert drift.components["s1_roll_progress"] < 0
    assert drift.components["s1_roll_remaining"] < 0
    assert credit(tr.step(**tr.pose(roll=-math.pi / 2)), "roll") == 0


def test_target_alignment_and_parallelism_are_guided_during_lift():
    tr = Trace()
    tr.step(**tr.pose(tilt=60))
    result = tr.step(**tr.pose(tilt=59))
    assert tr.trace.info["stage_s1_phase"] == "LIFT"
    assert result.components["s1_axis_progress"] > 0
    assert result.components["s1_parallel_progress"] > 0


def test_phase_switch_rebases_approach_without_fake_progress_bonus():
    tr = Trace()
    switched = tr.lift()
    assert tr.trace.info["stage_elapsed_s"] == pytest.approx(4 * DT)
    assert not switched.truncated
    assert credit(switched, "distance") == switched.components["s1_best_distance"] == 0
    phase_deltas = sum(switched.components[f"s1_{NAMES[index]}_progress"] for index in APPROACH_INDICES)
    assert phase_deltas > 0
    assert phase_deltas + switched.components["s1_phase_rebase"] == pytest.approx(0)
    assert switched.reward < 0
    # A valid endpoint above the sloping corridor retains the old fine bands.
    tr.step(**tr.pose(x=.055, clearance=.08))
    near = tr.step(**tr.pose(x=.045, clearance=.075))
    assert tr.trace.info["stage_s1_corridor_safe"]
    assert credit(near, "distance") > 0
    assert near.components["s1_best_distance"] > 0


def test_early_low_sideways_shortcut_cannot_collect_distance_bands():
    tr = Trace()
    tr.lift()
    low = tr.step(**tr.pose(x=.07, clearance=.025))
    assert not tr.trace.info["stage_s1_corridor_safe"]
    assert credit(low, "distance") == low.components["s1_best_distance"] == 0
    assert low.components["s1_corridor_progress"] < 0
    assert low.components["s1_corridor_remaining"] < 0
    recovered = tr.step(**tr.pose(x=.07, clearance=.08))
    assert recovered.components["s1_corridor_progress"] > 0


@pytest.mark.parametrize("kind", ["contact", "gap", "duplicate"])
def test_bad_evidence_never_pays_lift_or_turn_credit(kind):
    tr = Trace()
    pose = tr.pose(clearance=.06, roll=-math.pi / 4)
    if kind == "contact":
        first = tr.step(**pose, left_contact=True)
    elif kind == "gap":
        first = tr.step(dt=.2, **pose)
    else:
        first = tr.step(**pose, evidence_sim_time_s=tr.trace.snapshot.evidence_sim_time_s)
    assert credit(first, "lift") == credit(first, "roll") == 0
    recovered = tr.step(**pose)
    assert credit(recovered, "lift") == credit(recovered, "roll") == 0
    repeated = tr.step(**pose)
    assert credit(repeated, "lift") == credit(repeated, "roll") == 0


def test_duplicate_geometry_cannot_improve_reward_cost_or_critic_history():
    tr = Trace()
    first = tr.step(**tr.pose(clearance=.04))
    before = tr.reward.critic_context().copy()
    result = tr.step(**tr.pose(clearance=.08, roll=-math.pi / 2),
                     evidence_sim_time_s=tr.trace.snapshot.evidence_sim_time_s)
    for name in ("s1_lift_remaining", "s1_roll_remaining", "s1_remaining"):
        assert result.components[name] == pytest.approx(first.components[name])
    after = tr.reward.critic_context()
    assert after.shape == (34,)
    assert after[20:22] == pytest.approx(before[20:22])


def test_lift_then_reach_success_has_one_bonus_and_keeps_bootstrap():
    tr = Trace()
    tr.lift(roll=-math.pi / 2)
    for _ in range(7):
        if tr.trace.info["stage_cut"]:
            break
        result = tr.step(**tr.pose(x=0, clearance=.025, roll=-math.pi / 2))
    assert tr.trace.info["stage_success"]
    assert result.truncated and not result.terminated
    assert result.components["s1_success"] == 10
    assert result.potential_after > 0
    with pytest.raises(RuntimeError):
        tr.step(**tr.pose())


def test_real_failure_clears_potential_and_withholds_new_records():
    tr = Trace()
    result = tr.step(**tr.pose(clearance=.08, roll=-math.pi / 2), failure_reason="collision")
    assert result.terminated and not result.truncated
    assert result.potential_after == 0
    assert credit(result, "lift") == credit(result, "roll") == 0
    assert result.components["failure"] == -tr.trace.reward.failure_cost


def test_timeout_does_not_restart_budget_after_lift():
    tr = Trace()
    tr.lift()
    while not tr.trace.info["stage_cut"]:
        result = tr.step(**tr.pose(clearance=.08))
    assert tr.trace.info["stage_elapsed_s"] == pytest.approx(10.)
    assert tr.trace.info["stage_timeout"]
    assert result.truncated and not result.terminated
    assert tr.reward.critic_context()[18] == 1


POSTURE_METRICS = ("stage_bearing_error_rad", "stage_axis_error_rad",
                   "stage_s1_parallel_error_rad", "stage_s1_roll_error_rad")
REGRESSION_KEYS = tuple(f"s1_{name}_regression" for name in ("bearing", "axis", "parallel", "wrist_roll"))


def posture_step(tr, angles, *, dt=DT, downstream=False, **measurements):
    """Isolate the four geometry channels with certified-metric test values."""
    prev = tr.trace.snapshot
    info = dict(tr.trace.step(dt=dt, **measurements))
    info.update(dict(zip(POSTURE_METRICS, np.deg2rad(angles))))
    info["stage_downstream_grasp_lift"] = downstream
    return tr.reward.step(PickPlaceRewardInput(
        prev, tr.trace.snapshot, dt, terminated=tr.trace.snapshot.outcome != "running",
        truncated=bool(info["stage_cut"])), info)


def regression_values(result):
    return np.asarray([result.components[key] for key in REGRESSION_KEYS])


def test_posture_regression_charges_held_bad_pose_every_step_and_recovery_reduces_cost():
    tr = Trace()
    assert tr.reward.critic_context()[26:30] == pytest.approx([-1.] * 4)
    posture_step(tr, [30, 30, 30, 30])
    assert tr.reward.critic_context()[26:30] == pytest.approx(np.full(4, 1 / 6))
    bad = posture_step(tr, [35.5] * 4)
    held = posture_step(tr, [35.5] * 4)
    assert regression_values(bad) == pytest.approx(np.full(4, -.02 * DT * .5))
    assert regression_values(held) == pytest.approx(regression_values(bad))
    recovering = posture_step(tr, [33.] * 4)
    assert regression_values(recovering) == pytest.approx(np.full(4, -.02 * DT * .25))
    recovered = posture_step(tr, [30.] * 4)
    assert regression_values(recovered) == pytest.approx(np.zeros(4))
    assert tr.reward.critic_context()[26:30] == pytest.approx(np.full(4, 1 / 6))


@pytest.mark.parametrize("channel", range(4))
def test_posture_channels_are_independent(channel):
    tr = Trace()
    posture_step(tr, [30] * 4)
    angles = [30.] * 4
    angles[channel] = 40.5
    result = posture_step(tr, angles)
    expected = np.zeros(4)
    expected[channel] = -.02 * DT
    assert regression_values(result) == pytest.approx(expected)


def test_posture_regression_target_tolerances_and_half_degree_slack():
    tr = Trace()
    posture_step(tr, [0, 0, 0, 0])
    for angles in ([19, 19, 2, 14], [20.5, 20.5, 3.5, 15.5]):
        assert regression_values(posture_step(tr, angles)) == pytest.approx(np.zeros(4), abs=1e-15)
    result = posture_step(tr, [25.5, 25.5, 8.5, 20.5])
    assert regression_values(result) == pytest.approx(np.full(4, -.02 * DT * .5))


@pytest.mark.parametrize("kind", ["contact", "moving_target", "gap", "duplicate", "terminated", "failed_lift"])
def test_invalid_posture_cannot_improve_record_or_erase_existing_cost(kind):
    tr = Trace()
    posture_step(tr, [30] * 4)
    last = posture_step(tr, [40.5] * 4)
    history = tr.reward.critic_context()[26:].copy()
    kwargs = {}
    if kind == "contact":
        kwargs["left_contact"] = True
    elif kind == "moving_target":
        kwargs["target_linear_velocity_m_s"] = (.05, 0., 0.)
    elif kind == "gap":
        kwargs["dt"] = .2
    elif kind == "duplicate":
        kwargs["evidence_sim_time_s"] = tr.trace.snapshot.evidence_sim_time_s
    else:
        kwargs["failure_reason"] = "collision"
        if kind == "failed_lift":
            kwargs["downstream"] = True
    invalid = posture_step(tr, [0] * 4, **kwargs)
    assert tr.reward.critic_context()[26:] == pytest.approx(history)
    assert regression_values(invalid) / kwargs.get("dt", DT) == pytest.approx(regression_values(last) / DT)


def test_posture_records_reset_and_valid_downstream_lift_clears_penalty():
    tr = Trace()
    posture_step(tr, [30] * 4)
    posture_step(tr, [40.5] * 4)
    downstream = posture_step(tr, [60] * 4, downstream=True, left_contact=True)
    assert regression_values(downstream) == pytest.approx(np.zeros(4))
    assert tr.reward.critic_context()[26:30] == pytest.approx(np.full(4, 1 / 6))
    fresh_trace = LiftTrace(config=tr.stages)
    tr.reward.reset(fresh_trace.snapshot, fresh_trace.info)
    assert tr.reward.critic_context()[26:30] == pytest.approx([-1.] * 4)
    assert tr.reward.critic_context()[30:] == pytest.approx(np.zeros(4))


def test_posture_regression_integrates_control_time_with_bounded_combined_dense_cost():
    def accrue(dt):
        tr = Trace()
        posture_step(tr, [30] * 4)
        values = np.zeros(4)
        for _ in range(round(1 / dt)):
            values += regression_values(posture_step(tr, [180] * 4, dt=dt))
        return values
    assert accrue(1 / 30) == pytest.approx(accrue(1 / 60))
    assert accrue(1 / 30) == pytest.approx(np.full(4, -.02))
    tr = Trace()
    posture_step(tr, [30] * 4)
    tr.lift()
    cost_keys = ("s1_time", "s1_remaining", "s1_distance_remaining", "s1_lift_remaining",
                 "s1_roll_remaining", "s1_corridor_remaining", *REGRESSION_KEYS)
    total = 0.
    while not tr.trace.info["stage_cut"]:
        result = posture_step(tr, [180] * 4)
        cost = sum(result.components[key] for key in cost_keys)
        assert -.98 * DT - 1e-12 <= cost <= 0
        total += cost
    assert total >= -9.8 - 1e-9
    invalid = replace(tr.stages.process_reward.s1_lift_approach,
                      posture_regression=S1PostureRegressionConfig(cost_per_s=.03))
    with pytest.raises(ValueError, match="remaining dense cost"):
        S1LiftApproachReward(replace(tr.stages.process_reward, s1_lift_approach=invalid),
                            tr.stages, tr.trace.reward)


def test_disabling_posture_regression_preserves_every_old_reward_component_and_context_slot():
    current = StageTrainingConfig.load(PROFILE)
    legacy = replace(current, process_reward=replace(current.process_reward,
        s1_lift_approach=replace(current.process_reward.s1_lift_approach, posture_regression=None)))
    old, new = Trace(legacy), Trace(current)
    for angles in ([30] * 4, [40.5] * 4, [33] * 4, [20] * 4):
        before, after = posture_step(old, angles), posture_step(new, angles)
        assert not set(REGRESSION_KEYS).intersection(before.components)
        assert before.components == {k: v for k, v in after.components.items() if k not in REGRESSION_KEYS}
        assert before.reward == pytest.approx(after.reward - sum(regression_values(after)))
        assert old.reward.critic_context() == pytest.approx(new.reward.critic_context()[:26])


@pytest.mark.parametrize("values", [[], {"unknown": 1}, {"slack_rad": 0}, {"cost_per_s": True},
                                  {"error_scale_rad": float("nan")}, {"parallel_tolerance_rad": math.pi},
                                  {"slack_rad": .2, "error_scale_rad": .1}])
def test_posture_configuration_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        S1PostureRegressionConfig.from_dict(values)
