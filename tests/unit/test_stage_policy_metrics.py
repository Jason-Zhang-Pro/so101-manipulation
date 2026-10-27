"""Offline reporting must not invent stage success or stale sensor progress."""
from copy import deepcopy
import math

import pytest

from so101_capstone.evaluation.stage_policy_metrics import aggregate_results, summarize_episode


CONFIG = {"home_joint_positions_rad": [0, 0, 0, 0, 0, -0.1], "max_evidence_gap_s": 0.12}


def row(t=0.1, *, stage="S1", result="RUNNING", error_deg=6.0, distance=0.06):
    return {
        "step_index": round(t * 30),
        "reward": 1.25,
        "joint_positions_after_rad": [q + math.radians(error_deg) for q in CONFIG["home_joint_positions_rad"]],
        "joint_velocities_after_rad_s": [0.01, -0.02, 0.03, -0.04, 0.05, -0.06],
        "terminated": result in {"SUCCESS", "FAILURE"}, "truncated": result == "TIMEOUT",
        "info": {
            "training_stage": stage, "stage_result": result, "stage_elapsed_s": t,
            "stage_result_reason": "criterion_from_environment" if result != "RUNNING" else None,
            "sim_time_s": 10.0 + t, "evidence_sim_time_s": 10.0 + t,
            "evidence_valid": True, "task_evidence_available": True,
            "physical_target_present": True, "target_name": "target",
            "target_motion_valid": True, "distance_m": distance,
            "stage_metrics": {"stage_s1_clearance_m": 0.04, "stage_s1_lift_hold_s": 0.0},
            "task_snapshot": {"grasp_valid": False, "valid_release": False, "events": []},
        },
    }


def summarize(records, stage="S1", **kwargs):
    return summarize_episode(records, stage=stage, initial_info={},
                             reward_config=CONFIG, **kwargs)


@pytest.mark.parametrize("stage", [f"S{i}" for i in range(6)])
@pytest.mark.parametrize("result", ["SUCCESS", "TIMEOUT", "FAILURE"])
def test_authoritative_stage_result_is_preserved(stage, result):
    # Even impossible-looking diagnostics cannot override the runtime decision.
    value = summarize([row(stage=stage, result=result, error_deg=120, distance=10)], stage)
    assert value["stage_result"] == result
    assert value["evaluated"]
    assert value["success"] == (result == "SUCCESS")
    assert value["completed_steps"] == 1
    assert value["total_reward"] == 1.25
    assert value["stage_result_reason"] == "criterion_from_environment"
    assert value["error"] is None


@pytest.mark.parametrize("change", ["wrong_stage", "nonterminal", "running", "exception", "not_started"])
def test_non_authoritative_success_never_counts(change):
    sample = row(result="SUCCESS", error_deg=0, distance=0.001)
    kwargs = {}
    if change == "wrong_stage":
        sample["info"]["training_stage"] = "S2"
    elif change == "nonterminal":
        sample["terminated"] = False
    elif change == "running":
        sample["info"]["stage_result"] = "RUNNING"
    elif change == "exception":
        kwargs["error"] = "Gazebo disconnected"
    else:
        kwargs["policy_started"] = False
    result = summarize([sample], **kwargs)
    assert result["stage_result"] == "ERROR"
    assert not result["evaluated"]
    assert not result["success"]
    assert result["error"]


def test_no_policy_samples_does_not_treat_preparation_as_stage_success():
    initial = row(result="SUCCESS")["info"]
    result = summarize_episode([], stage="S1", initial_info=initial, reward_config=CONFIG,
                               policy_started=False, error="Preparation could not certify start")
    assert result["stage_result"] == "ERROR"
    assert result["completed_steps"] == 0
    assert result["total_reward"] == 0
    assert result["target"]["minimum_distance_m"] is None
    assert result["home"]["final_mae_deg"] is None
    assert not result["s1_lift"]["completed"]
    assert result["final_info"] == initial
    result["final_info"]["training_stage"] = "changed"
    assert initial["training_stage"] == "S1"


def test_measured_home_statistics_include_gripper_and_tail_one_second():
    records = [row(0.1, error_deg=12), row(1.2, error_deg=6), row(1.7, error_deg=0, result="TIMEOUT")]
    # A gripper-only 6-degree final error: six-joint MAE is one degree.
    records[-1]["joint_positions_after_rad"][-1] += math.radians(6)
    result = summarize(records)
    home = result["home"]
    assert home["mean_mae_deg"] == pytest.approx(19 / 3)
    assert home["final_mae_deg"] == pytest.approx(1)
    assert home["final_max_abs_error_deg"] == pytest.approx(6)
    assert home["final_joint_abs_error_deg"] == pytest.approx([0, 0, 0, 0, 0, 6])
    assert home["tail_valid_unique_samples"] == 2
    assert home["tail_mean_mae_deg"] == pytest.approx(3.5)
    assert home["mean_joint_abs_velocity_rad_s"] == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert home["final_joint_abs_velocity_rad_s"] == pytest.approx(home["mean_joint_abs_velocity_rad_s"])
    assert result["total_reward"] == 3.75


@pytest.mark.parametrize("edit", [
    {"evidence_valid": False}, {"task_evidence_available": False},
    {"target_motion_valid": False}, {"evidence_sim_time_s": 9.0},
    {"evidence_sim_time_s": 15.0},
    {"target_name": None, "physical_target_present": False},
])
def test_invalid_target_evidence_cannot_supply_minimum_or_final_distance(edit):
    first, last = row(distance=0.06), row(0.2, distance=0.001, result="TIMEOUT")
    last["info"].update(edit)
    result = summarize([first, last])
    assert result["target"]["minimum_distance_m"] == 0.06
    assert result["target"]["final_distance_m"] is None
    assert result["target"]["last_valid_distance_m"] == 0.06


def test_duplicate_and_backwards_evidence_cannot_improve_nearest():
    records = [row(0.2), row(0.2, distance=0.001), row(0.1, distance=0.0001, result="TIMEOUT")]
    result = summarize(records)
    assert result["evidence"]["duplicate"] == 1
    assert result["evidence"]["backwards"] == 1
    assert result["target"]["valid_unique_samples"] == 1
    assert result["target"]["minimum_distance_m"] == 0.06
    assert result["target"]["final_distance_m"] is None
    assert result["home"]["final_mae_deg"] is None


def test_lift_completion_and_reward_subgoals_are_recorded_without_success_inference():
    records = [row(), row(0.2), row(0.3, result="TIMEOUT")]
    records[1]["info"]["stage_metrics"].update(
        stage_s1_lift_completed=True, stage_s1_lift_hold_s=0.1, stage_s1_clearance_m=0.085)
    records[2]["info"]["stage_metrics"].update(stage_s1_lift_completed=True, stage_s1_clearance_m=0.06)
    records[1]["info"]["reward_completed_subgoals"] = ["lift_complete"]
    records[2]["info"]["reward_completed_subgoals"] = ["lift_complete", "another_goal"]
    result = summarize(records)
    assert result["s1_lift"] == {
        "completed": True, "first_completion_elapsed_s": 0.2,
        "maximum_clearance_m": 0.085, "maximum_hold_s": 0.1,
    }
    assert result["reward_completed_subgoals"] == ["lift_complete", "another_goal"]
    assert result["stage_metrics"] == records[-1]["info"]["stage_metrics"]
    assert result["stage_result"] == "TIMEOUT"


def test_attachment_release_and_drop_evidence_are_distinguished():
    records = [row(stage="S4"), row(0.2, stage="S4"), row(0.3, stage="S4", result="FAILURE")]
    records[0]["info"]["grasp_attachment"] = {"active": True, "events": ["grasp_attachment_attached"]}
    records[0]["info"]["task_snapshot"].update(grasp_valid=True, events=["grasp_confirmed"])
    records[0]["info"]["reward_events"] = ["grasp_confirmed"]  # Same event, not two grasps.
    records[1]["info"]["task_snapshot"].update(valid_release=True, events=["valid_release", "grasp_lost"])
    records[1]["info"]["grasp_attachment"] = {"active": False, "events": ["grasp_attachment_detached"]}
    result = summarize(records[:2], "S4")
    assert result["state_events"]["grasp_lost_events"] == 1
    assert not result["state_events"]["dropped_observed"]
    records[2]["info"]["task_snapshot"].update(failure_reason="object_below_workspace")
    result = summarize(records, "S4")
    events = result["state_events"]
    assert events["grasp_confirmed_events"] == 1
    assert events["attachment_attached_events"] == 1
    assert events["attachment_detached_events"] == 1
    assert events["observed_attachment_active"]
    assert events["valid_release_events"] == 1
    assert events["dropped_events"] == 1
    assert events["dropped_observed"]


def test_optional_serialized_measurements_are_supported():
    sample = row(result="SUCCESS")
    sample["info"]["evaluation_measurements"] = {
        "sim_time_s": 10.1, "evidence_sim_time_s": 10.1,
        "evidence_valid": True, "joint_measurement_valid": True,
        "joint_positions_rad": CONFIG["home_joint_positions_rad"],
        "joint_velocities_rad_s": [0] * 6,
        "target_name": "target", "target_measurement_valid": True, "target_motion_valid": True,
        "target_position_world_m": [0, 0, 0], "grasp_position_world_m": [0.03, 0.04, 0],
    }
    result = summarize([sample])
    assert result["home"]["final_mae_deg"] == 0
    assert result["target"]["minimum_distance_m"] == pytest.approx(0.05)


def test_missing_diagnostics_remain_unknown_but_authoritative_decision_is_valid():
    sample = row(result="SUCCESS")
    sample["info"] = {key: value for key, value in sample["info"].items()
                      if key in {"training_stage", "stage_result", "stage_elapsed_s"}}
    result = summarize([sample])
    assert result["success"]
    assert result["home"]["final_mae_deg"] is None
    assert result["target"]["minimum_distance_m"] is None


def test_aggregate_uses_separate_denominators_and_does_not_trust_success_boolean():
    episodes = [
        summarize([row(stage="S0", result="SUCCESS")], "S0"),
        summarize([row(stage="S0", result="TIMEOUT")], "S0"),
        summarize([], "S0", policy_started=False, error="prep failed"),
        summarize([row(stage="S1", result="FAILURE")]),
        summarize([row(stage="S1", result="SUCCESS")], error="sensor disconnected"),
    ]
    episodes[-1]["success"] = True
    result = aggregate_results(episodes, ["S0", "S1", "S2"])
    assert result["S0"] == {
        "attempted": 3, "evaluated": 2, "success": 1, "timeout": 1, "failure": 0,
        "errors": 1, "preparation_errors": 1,
        "success_rate_evaluated": 0.5, "success_rate_all_attempts": 1 / 3,
    }
    assert result["S1"]["evaluated"] == 1
    assert result["S1"]["failure"] == 1
    assert result["S1"]["success"] == 0
    assert result["S1"]["errors"] == 1
    assert result["S2"]["success_rate_evaluated"] is None
    assert result["S2"]["success_rate_all_attempts"] is None
    assert aggregate_results([], "s0")["S0"]["attempted"] == 0
    assert aggregate_results([], []) == {}


def test_pure_summary_does_not_modify_inputs():
    records = [row(result="SUCCESS")]
    before = deepcopy(records)
    summarize(records)
    assert records == before


def test_structured_runtime_error_is_preserved():
    error = {"operation": "policy_step", "type": "RuntimeError", "message": "world crashed"}
    result = summarize([row()], error=error)
    assert result["error"] == error
    assert result["error"] is not error
    assert result["stage_result"] == "ERROR"
