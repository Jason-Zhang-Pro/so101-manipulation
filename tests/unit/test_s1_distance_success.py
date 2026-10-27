"""The optional S1 milestone is a certified lift followed by strict 2 cm reach."""
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import PickPlaceRewardInput
from so101_capstone.rewards.s1_lift_approach_reward import S1LiftApproachReward
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_s1_lift_approach_stages import LiftTrace, stages


def distance_stages(**overrides):
    return replace(stages(), s1_success_mode="lift_then_distance", s1_reach_hold_s=0., **overrides)


def trace():
    return LiftTrace(config=distance_stages())


def close_pose(distance=.019):
    # XY deliberately exceeds the old 1 cm condition; opening and wrist stay
    # at HOME, so a pass cannot accidentally come from the old qualified pose.
    return {"grasp_position_world_m": (distance, 0., .765)}


def test_close_target_cannot_pass_before_certified_lift():
    tr = trace()
    info = tr.step(**close_pose(.001))
    assert info["stage_s1_phase"] == "LIFT"
    assert not info["stage_s1_lift_completed"]
    assert not info["stage_reach_candidate"]
    assert not info["stage_success"]


def test_79_mm_clearance_never_completes_80_mm_lift():
    tr = trace()
    for _ in range(6):
        info = tr.step(grasp_position_world_m=(.1, 0., .829))
        assert not info["stage_s1_lift_completed"]
    assert not tr.step(**close_pose(.001))["stage_success"]


def test_lift_still_requires_a_tenth_second_of_qualified_evidence():
    tr = trace()
    assert not tr.lifted()["stage_s1_lift_completed"]
    assert not tr.lifted()["stage_s1_lift_completed"]
    assert tr.info["stage_s1_lift_hold_s"] == pytest.approx(.05)
    info = tr.step(**close_pose(.001))
    assert not info["stage_success"]
    assert info["stage_s1_lift_hold_s"] == 0.
    tr.finish_lift()
    assert tr.info["stage_s1_lift_hold_s"] == pytest.approx(.1)
    info = tr.step(**close_pose())
    assert info["stage_success"] and info["stage_cut"]
    assert info["stage_result_reason"] == "reach_distance"
    assert info["stage_reach_hold_s"] == 0.
    assert tr.snapshot.outcome == "running"


@pytest.mark.parametrize("distance", [.020, .02001, .025])
def test_distance_threshold_is_strict_and_above_boundary_stays_running(distance):
    tr = trace()
    tr.finish_lift()
    info = tr.step(**close_pose(distance))
    assert info["stage_reach_distance_m"] == pytest.approx(distance)
    assert not info["stage_reach_candidate"]
    assert not info["stage_success"]
    assert info["stage_result"] == "RUNNING"


@pytest.mark.parametrize("distance", [.01999, .01, .0])
def test_first_fresh_endpoint_inside_2_cm_passes_without_reach_dwell(distance):
    tr = trace()
    tr.finish_lift()
    info = tr.step(dt=1 / 30, **close_pose(distance))
    assert info["stage_success"] and info["stage_cut"]
    assert info["stage_result_reason"] == "reach_distance"
    assert info["stage_reach_candidate"]
    assert info["stage_reach_hold_s"] == 0.


@pytest.mark.parametrize("violation", ["orientation", "opening", "roll", "negative_dz", "corridor",
                                        "contact", "moving_target", "displaced_target", "unsupported_target"])
def test_old_pose_and_source_gates_do_not_block_post_lift_distance_success(violation):
    tr = trace()
    tr.finish_lift()
    pose = close_pose(.015)
    if violation == "orientation":
        pose["grasp_rotation_world"] = np.eye(3)
    elif violation in ("opening", "roll"):
        joints = list(tr.reward.home_joint_positions_rad)
        if violation == "opening":
            joints[-1] += 1.4
        else:
            joints[4] += math.pi / 2
        pose["joint_positions_rad"] = joints
    elif violation == "negative_dz":
        pose["grasp_position_world_m"] = (.005, 0., .755)
    elif violation == "contact":
        pose["left_contact"] = True
    elif violation == "moving_target":
        pose["target_linear_velocity_m_s"] = (.2, 0., 0.)
    elif violation == "displaced_target":
        pose.update(target_position_world_m=(.1, 0., .765), grasp_position_world_m=(.115, 0., .765))
    elif violation == "unsupported_target":
        pose.update(target_position_world_m=(0., 0., .82), grasp_position_world_m=(.015, 0., .82))
    info = tr.step(**pose)
    if violation == "corridor":
        assert not info["stage_s1_corridor_safe"]
    assert info["stage_success"]
    assert info["stage_result_reason"] == "reach_distance"


def test_lift_eligibility_is_unchanged_by_distance_only_success_mode():
    tr = trace()
    for _ in range(5):
        assert not tr.lifted(left_contact=True)["stage_s1_lift_completed"]
    assert not tr.step(**close_pose())["stage_success"]


def test_physical_failure_takes_precedence_over_a_close_endpoint():
    tr = trace()
    tr.finish_lift()
    info = tr.step(**close_pose(), failure_reason="collision")
    assert info["stage_result"] == "FAILURE"
    assert not info["stage_success"]
    assert tr.snapshot.outcome == "task_failure"


@pytest.mark.parametrize("kind", ["duplicate", "gap"])
def test_reused_or_gap_evidence_cannot_certify_distance_success(kind):
    tr = trace()
    tr.finish_lift()
    kwargs = {"evidence_sim_time_s": tr.snapshot.evidence_sim_time_s} if kind == "duplicate" else {"dt": .2}
    info = tr.step(**close_pose(), **kwargs)
    assert not info["stage_evidence_fresh"]
    assert not info["stage_success"]
    assert tr.step(**close_pose())["stage_success"]


def test_invalid_sensor_evidence_is_rejected_without_certifying_a_pass():
    tr = trace()
    tr.finish_lift()
    with pytest.raises(ValueError, match="invalid stage sensor evidence"):
        tr.step(**close_pose(), evidence_valid=False)
    assert not tr.info["stage_success"]


def test_endpoint_after_ten_seconds_cannot_retroactively_pass():
    tr = trace()
    tr.finish_lift()
    while tr.time < 9.94:
        tr.lifted()
    assert tr.time == pytest.approx(9.95)
    info = tr.step(dt=.1, **close_pose())
    assert info["stage_evidence_fresh"]
    assert info["stage_elapsed_s"] == pytest.approx(10.05)
    assert info["stage_timeout"] and not info["stage_success"]


def test_fresh_close_endpoint_at_exact_deadline_can_pass():
    tr = trace()
    tr.finish_lift()
    while tr.time < 9.94:
        tr.lifted()
    info = tr.step(**close_pose())
    assert info["stage_elapsed_s"] == pytest.approx(10.)
    assert info["stage_success"]


def test_downstream_grasp_lift_does_not_bypass_2_cm_distance(monkeypatch):
    tr = trace()
    tr.finish_lift()
    monkeypatch.setattr(tr.evaluator, "_strong_downstream", lambda *_: True)
    info = tr.step(**close_pose(.04))
    assert info["stage_downstream_grasp_lift"]
    assert not info["stage_success"]
    assert info["stage_result"] == "RUNNING"


def test_zero_reach_dwell_keeps_process_reward_and_context_finite():
    tr = trace()
    reward = S1LiftApproachReward(tr.config.process_reward, tr.config, tr.reward)
    reward.reset(tr.snapshot, tr.info)
    for point in ((.1, 0., .83), (.1, 0., .83), (.1, 0., .83), (.019, 0., .765)):
        prev = tr.snapshot
        info = tr.step(grasp_position_world_m=point)
        with np.errstate(divide="raise", invalid="raise"):
            result = reward.step(PickPlaceRewardInput(
                prev, tr.snapshot, .05, terminated=False, truncated=bool(info["stage_cut"])), info)
        assert math.isfinite(result.reward)
        assert np.isfinite(list(result.components.values())).all()
        assert np.isfinite(reward.critic_context()).all()
    assert info["stage_success"]
    assert result.truncated and not result.terminated


def test_new_success_mode_roundtrips_while_default_legacy_hashes_stay_unchanged():
    cfg = distance_stages()
    assert StageTrainingConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.to_dict()["s1_success_mode"] == "lift_then_distance"
    default = StageTrainingConfig()
    assert default.s1_success_mode == "qualified_reach"
    assert "s1_success_mode" not in default.to_dict()
    assert default.config_hash == "918facc6e4d89d41747cea931d3443ec8d048688c71161950a1ba04b6a829784"
    assert "s1_success_mode" not in stages().to_dict()
    assert stages().config_hash == "ed5aaa3fe7c42888087bbc1fdefeae3112d24e58b5e8b7bbc8397f7234a2b047"
    with pytest.raises(ValueError):
        replace(stages(), s1_reach_hold_s=0.)
