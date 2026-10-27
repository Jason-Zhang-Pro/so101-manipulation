"""Evidence-clock and geometric contracts for repeatable HOME/reach monitoring."""
from copy import deepcopy
import math

import pytest

from so101_capstone.evaluation.home_reach_metrics import measure, summarize


REWARD = {
    "home_joint_positions_rad": [0, 0, 0, 0, 0, -0.1],
    "home_joint_tolerance_rad": 0.02,
    "home_velocity_tolerance_rad_s": 0.03,
    "max_evidence_gap_s": 0.12,
    "desired_grasp_axis_world": [0, 0, -1],
    "approach_reference_offset_m": 0.04,
}


def info(time=10.0, *, target=False, distance=0.06):
    return {"evaluation_measurements": {
        "sim_time_s": time, "evidence_sim_time_s": time,
        "joint_positions_rad": [0, 0, 0, 0, 0, -0.1],
        "joint_velocities_rad_s": [0] * 6,
        "grasp_position_world_m": [0.25, 0, 0.75 + distance],
        "grasp_rotation_world": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "target_name": "cube" if target else None,
        "target_position_world_m": [0.25, 0, 0.75] if target else None,
        "evidence_valid": True, "joint_measurement_valid": True,
        "target_measurement_valid": True, "target_motion_valid": True,
        "contact_valid": True, "left_contact": False, "right_contact": False,
    }, "task_snapshot": {"task_phase": "approach" if target else "wait_home"}}


def row(elapsed, *, target=False, distance=0.06, edit=None):
    value = info(10 + elapsed, target=target, distance=distance)
    if edit:
        value["evaluation_measurements"].update(edit)
    return {"step_index": round(elapsed * 30), "elapsed_s": elapsed,
            "diagnostics": measure(value, REWARD)}


def test_geometry_matches_stage_metrics_and_separates_position_velocity():
    value = info(target=True)
    value["evaluation_measurements"]["grasp_position_world_m"] = [0.28, 0.04, 0.77]
    value["evaluation_measurements"]["joint_positions_rad"][-1] = 0.5
    value["evaluation_measurements"]["joint_velocities_rad_s"][2] = -0.04
    result = measure(value, REWARD)
    assert result["evidence_valid"] and result["target_evidence_valid"]
    assert result["distance_m"] == pytest.approx(math.sqrt(0.03**2 + 0.04**2 + 0.02**2))
    assert result["xy_distance_m"] == pytest.approx(0.05)
    assert result["dz_m"] == pytest.approx(0.02)
    assert result["bearing_error_deg"] == pytest.approx(math.degrees(math.atan2(0.05, 0.06)))
    assert result["axis_error_deg"] == 0
    assert result["opening_rad"] == pytest.approx(0.6)
    assert result["home_joint_errors_rad"] == pytest.approx([0, 0, 0, 0, 0, 0.6])
    assert not result["at_home"] and not result["robot_static"]


def test_no_target_retains_home_metrics_without_invented_reach_distance():
    result = measure(info(), REWARD)
    assert result["at_home"] and result["robot_static"]
    assert not result["target_present"] and not result["target_evidence_valid"]
    assert result["distance_m"] is None and result["bearing_error_deg"] is None


@pytest.mark.parametrize("edit", [
    {"evidence_valid": False}, {"joint_measurement_valid": False},
    {"evidence_sim_time_s": 9.8}, {"evidence_sim_time_s": 10.1},
    {"joint_positions_rad": [float("nan")] * 6},
    {"joint_velocities_rad_s": [0] * 5},
])
def test_invalid_or_stale_evidence_cannot_certify_home_or_target(edit):
    value = info(target=True)
    value["evaluation_measurements"].update(edit)
    result = measure(value, REWARD)
    assert not result["evidence_valid"] and not result["target_evidence_valid"]
    assert not result["at_home"] and not result["robot_static"]


@pytest.mark.parametrize("edit", [{"target_motion_valid": False}, {"target_measurement_valid": False},
                                 {"grasp_rotation_world": None},
                                 {"grasp_rotation_world": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}])
def test_invalid_target_geometry_does_not_erase_good_joint_evidence(edit):
    value = info(target=True)
    value["evaluation_measurements"].update(edit)
    result = measure(value, REWARD)
    assert result["evidence_valid"] and result["at_home"]
    assert not result["target_evidence_valid"]


def test_duplicate_evidence_adds_neither_samples_nor_stable_duration():
    samples = [row(.1), row(.15, edit={"evidence_sim_time_s": 10.1}), row(.2)]
    result = summarize(samples, initial=measure(info(), REWARD), reward_config=REWARD)
    assert result["evidence"]["valid_unique_samples"] == 3
    assert result["evidence"]["duplicate_samples"] == 1
    assert result["before_target"]["longest_home_and_static_duration_s"] == pytest.approx(.2)
    assert result["before_target"]["first_at_home_evidence_elapsed_s"] == 0


@pytest.mark.parametrize("middle", [
    row(.15, edit={"evidence_valid": False}),
    row(.15, edit={"evidence_sim_time_s": 10.05}),
    row(.15, edit={"joint_velocities_rad_s": [.04] * 6}),
])
def test_bad_or_unstable_intermediate_evidence_breaks_hold(middle):
    samples = [row(.1), deepcopy(middle), row(.2), row(.3)]
    result = summarize(samples, initial=measure(info(), REWARD), reward_config=REWARD)
    assert result["before_target"]["longest_home_and_static_duration_s"] == pytest.approx(.1)


def test_long_gap_keeps_measured_endpoint_but_does_not_credit_missing_time():
    samples = [row(.1), row(.6), row(.7)]
    result = summarize(samples, initial=measure(info(), REWARD), reward_config=REWARD)
    assert result["evidence"]["valid_unique_samples"] == 4
    assert result["evidence"]["long_evidence_gaps"] == 1
    assert result["before_target"]["longest_home_and_static_duration_s"] == pytest.approx(.1)


def test_boundary_and_reach_summary_never_equate_two_cm_with_success():
    samples = [row(3), row(3.1, target=True, distance=.08), row(3.2, target=True, distance=.01),
               row(3.25, target=True, distance=.001, edit={"evidence_sim_time_s": 13.2}),
               row(3.3, target=True, distance=.04)]
    result = summarize(samples, initial=measure(info(), REWARD), reward_config=REWARD)
    assert result["before_target"]["home_and_static_at_target_request"]
    after = result["after_target"]
    assert after["first_valid_target_feedback_elapsed_s"] == 3.1
    assert after["first_valid_target_evidence_elapsed_s"] == pytest.approx(3.1)
    assert after["minimum_distance_m"] == pytest.approx(.01)
    assert after["nearest"]["step_index"] == 96
    assert after["final_distance_m"] == pytest.approx(.04)
    assert after["tail_mean_distance_m"] == pytest.approx((.08 + .01 + .04) / 3)
    assert after["distance_threshold_sample_counts"] == {"le_8_cm": 3, "le_6_cm": 2, "le_4_cm": 2, "le_2_cm": 1}
    assert not result["formal_stage_success_evaluated"]
    assert not result["distance_threshold_counts_are_successes"]


def test_missing_boundary_and_invalid_final_remain_unknown():
    samples = [row(2.9), row(3.1, target=True), row(13, target=True, edit={"evidence_valid": False})]
    result = summarize(samples, initial=measure(info(), REWARD), reward_config=REWARD)
    assert result["before_target"]["home_and_static_at_target_request"] is None
    assert not result["before_target"]["boundary_measurement_available"]
    assert result["after_target"]["final_distance_m"] is None
    assert result["after_target"]["tail_mean_distance_m"] is None


def test_first_home_uses_evidence_clock_relative_to_initial_not_control_clock():
    initial_info = info()
    initial_info["evaluation_measurements"]["joint_positions_rad"][1] = .5
    samples = [row(.1, edit={"evidence_sim_time_s": 10.08})]
    result = summarize(samples, initial=measure(initial_info, REWARD), reward_config=REWARD)
    assert result["before_target"]["first_at_home_evidence_elapsed_s"] == pytest.approx(.08)


def test_initial_camera_lag_does_not_shift_policy_timeline_origin():
    initial_info = info()
    initial_info["evaluation_measurements"].update(
        evidence_sim_time_s=9.95, joint_positions_rad=[0, .5, 0, 0, 0, -.1])
    samples = [row(.25, edit={"evidence_sim_time_s": 10.2})]
    result = summarize(samples, initial=measure(initial_info, REWARD), reward_config=REWARD)
    assert result["before_target"]["first_at_home_evidence_elapsed_s"] == pytest.approx(.2)
    initial_info["evaluation_measurements"]["joint_positions_rad"][1] = 0
    result = summarize([], initial=measure(initial_info, REWARD), reward_config=REWARD)
    assert result["before_target"]["first_at_home_evidence_elapsed_s"] == 0
    result = summarize([row(.1, edit={"evidence_sim_time_s": 10.05})],
                       initial=measure(initial_info, REWARD), reward_config=REWARD)
    assert result["before_target"]["longest_home_and_static_duration_s"] == pytest.approx(.05)


def test_origin_falls_back_to_initial_evidence_time_for_older_diagnostics():
    initial_diagnostics = measure(info(), REWARD)
    initial_diagnostics.pop("sim_time_s")
    initial_diagnostics["at_home"] = False
    result = summarize([row(.1)], initial=initial_diagnostics, reward_config=REWARD)
    assert result["before_target"]["first_at_home_evidence_elapsed_s"] == pytest.approx(.1)
