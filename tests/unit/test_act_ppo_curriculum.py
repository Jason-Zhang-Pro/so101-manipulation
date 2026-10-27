"""Pure geometric/statistical curriculum tests; no physical reachability claim."""

from copy import deepcopy
import json
import math

import pytest

from so101_capstone.training.act_ppo_curriculum import (
    ReachCurriculum, ReachCurriculumConfig, target_overlaps_goal,
)


GOAL = (0.078, -0.300, 0.7515)


def episode(index, level=0, *, precise=True, grasp=False, lift=False, **overrides):
    return {"episode_attempt_index": index, "task_mode": "pick_place", "valid": True,
            "sampled_level": level, "curriculum_rehearsal": False,
            "stage_reached": {"precise_reach": precise, "grasp_valid": grasp, "lifted_once": lift},
            **overrides}


def promote(curriculum, level):
    for _ in range(curriculum.config.window_size):
        curriculum.observe(episode(curriculum.allocate_attempt(), level,
                                   grasp=level > 0, lift=level > 0))
    assert curriculum.level == level + 1


def test_at_least_twenty_current_level_episodes_and_seventy_percent_pass_required():
    curriculum = ReachCurriculum()
    for index in range(1, 20):
        curriculum.observe(episode(index, precise=index <= 13))
    assert curriculum.level == 0
    assert not curriculum.observe(episode(20, precise=False))["promoted"]
    assert curriculum.diagnostics()["curriculum_window_pass_count"] == 13
    # Replace failed observations until the rolling window reaches 14/20.
    for index in range(21, 34):
        assert not curriculum.observe(episode(index))["promoted"]
    result = curriculum.observe(episode(34))
    assert result["promoted"] and curriculum.level == 1
    assert result["curriculum_window_count"] == 0


def test_second_level_requires_grasp_and_lift_in_same_episode_not_separate_episodes():
    curriculum = ReachCurriculum()
    promote(curriculum, 0)
    for index in range(21, 41):
        curriculum.observe(episode(index, 1, grasp=index % 2 == 0, lift=index % 2 == 1))
    assert curriculum.level == 1
    assert curriculum.diagnostics()["curriculum_window_pass_count"] == 0
    for index in range(41, 61):
        curriculum.observe(episode(index, 1, grasp=True, lift=True))
    assert curriculum.level == 2
    for index in range(61, 85):
        curriculum.observe(episode(index, 2, task_success=True,
                                   final_info={"grasp_evidence_mode": "contact"}))
    assert curriculum.level == 2


def test_smooth_contact_grasp_and_lift_can_pass_reach_without_open_gripper_dwell():
    curriculum = ReachCurriculum()
    for index in range(1, 21):
        curriculum.observe(episode(index, precise=False, grasp=True, lift=True,
                                   final_info={"grasp_evidence_mode": "contact"}))
    assert curriculum.level == 1


def test_contact_without_lift_cannot_skip_reach_frontier():
    curriculum = ReachCurriculum()
    for index in range(1, 21):
        curriculum.observe(episode(index, precise=False, grasp=True, lift=False))
    assert curriculum.level == 0


@pytest.mark.parametrize("level", [0, 1])
def test_formal_complete_task_can_pass_easier_frontiers(level):
    curriculum = ReachCurriculum()
    if level:
        promote(curriculum, 0)
    for _ in range(20):
        curriculum.observe(episode(curriculum.allocate_attempt(), level,
                                   precise=False, task_success=True,
                                   final_info={"grasp_evidence_mode": "contact", "formal_success": True}))
    assert curriculum.level == level + 1


@pytest.mark.parametrize("extra", [{}, {"final_info": {"grasp_evidence_mode": "contact", "formal_success": False}}])
def test_bare_or_nonformal_success_label_cannot_skip_frontier(extra):
    curriculum = ReachCurriculum()
    for index in range(1, 21):
        curriculum.observe(episode(index, precise=False, task_success=True, **extra))
    assert curriculum.level == 0


@pytest.mark.parametrize("overrides", [
    {"valid": False}, {"valid_episode": False}, {"type": "episode_failure"},
    {"task_mode": "idle_empty"}, {"task_mode": "idle_completed"},
    {"curriculum_probe": True}, {"grasp_evidence_mode": "proxy"},
    {"final_info": {"grasp_evidence_mode": "proxy"}},
    {"grasp_evidence_mode": "contact", "final_info": {"grasp_evidence_mode": "proxy"}},
    {"grasp_evidence_mode": "proxy", "final_info": {"grasp_evidence_mode": "contact"}},
    {"sampled_level": 2}, {"curriculum_rehearsal": True},
])
def test_failed_idle_probe_proxy_and_rehearsal_episodes_never_unlock_current_level(overrides):
    curriculum = ReachCurriculum()
    for index in range(1, 41):
        curriculum.observe(episode(index, **overrides))
    assert curriculum.level == 0
    assert curriculum.diagnostics()["curriculum_window_count"] == 0


def test_duplicate_attempt_and_old_wave_results_do_not_advance_new_frontier():
    curriculum = ReachCurriculum()
    for _ in range(30):
        curriculum.observe(episode(1))
    assert curriculum.diagnostics()["curriculum_window_count"] == 1
    for index in range(2, 21):
        curriculum.observe(episode(index))
    assert curriculum.level == 1
    curriculum.observe(episode(21, 0))
    assert curriculum.diagnostics()["curriculum_window_count"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2, "true", None])
def test_nonboolean_milestones_reject_without_mutating_state(value):
    curriculum = ReachCurriculum()
    before = curriculum.state_dict()
    with pytest.raises(ValueError, match="precise_reach"):
        curriculum.observe(episode(1, precise=value))
    assert curriculum.state_dict() == before


def test_numeric_boolean_milestones_are_supported():
    curriculum = ReachCurriculum()
    for index in range(1, 21):
        curriculum.observe(episode(index, precise=1.0, grasp=0., lift=0))
    assert curriculum.level == 1


def test_final_info_contact_evidence_allows_measured_milestones():
    curriculum = ReachCurriculum()
    for index in range(1, 21):
        curriculum.observe(episode(index, final_info={"grasp_evidence_mode": "contact"}))
    assert curriculum.level == 1


def test_invalid_final_info_rejects_without_mutating_state():
    curriculum = ReachCurriculum()
    before = curriculum.state_dict()
    with pytest.raises(ValueError, match="final_info"):
        curriculum.observe(episode(1, final_info=None))
    assert curriculum.state_dict() == before


@pytest.mark.parametrize("level,inner,outer,angle_degrees", [(0, .24, .28, 10), (1, .22, .31, 25), (2, .20, .35, 45)])
def test_seeded_samples_stay_in_candidate_sectors_on_table_with_fixed_yaw(level, inner, outer, angle_degrees):
    curriculum = ReachCurriculum(ReachCurriculumConfig(rehearsal_probability=0))
    for old_level in range(level):
        promote(curriculum, old_level)
    before = curriculum.state_dict()
    for seed in range(200):
        scene = curriculum.sample_target(seed, goal_center_world_m=GOAL)
        assert scene == curriculum.sample_target(seed, goal_center_world_m=GOAL)
        x, y, z = scene["target_position_world_m"]
        assert inner <= math.hypot(x, y) <= outer
        assert abs(math.degrees(math.atan2(y, x))) <= angle_degrees
        assert z == pytest.approx(.768)
        assert scene["target_quaternion_xyzw"] == [0., 0., 0., 1.]
        assert scene["sampled_level"] == level
        assert not target_overlaps_goal((x, y), (.015, .015), GOAL[:2], (.05, .05), 0.)
    assert curriculum.state_dict() == before


def test_rehearsal_draws_only_earlier_levels_deterministically():
    curriculum = ReachCurriculum()
    promote(curriculum, 0)
    promote(curriculum, 1)
    counts = [0, 0, 0]
    for seed in range(1000):
        scene = curriculum.sample_target(seed, goal_center_world_m=GOAL)
        counts[scene["sampled_level"]] += 1
        assert scene["curriculum_rehearsal"] == (scene["sampled_level"] < 2)
    assert counts[0] > 0 and counts[1] > 0
    assert .15 < (counts[0] + counts[1]) / 1000 < .25


def test_goal_overlap_rejects_partial_touching_and_rotated_footprints():
    assert target_overlaps_goal((.06, 0), (.015, .015), (0, 0), (.05, .05), 0.)
    assert target_overlaps_goal((.065, 0), (.015, .015), (0, 0), (.05, .05), 0.)
    assert not target_overlaps_goal((.066, 0), (.015, .015), (0, 0), (.05, .05), 0.)
    assert target_overlaps_goal((.08, 0), (.015, .015), (0, 0), (.05, .05), math.pi / 4)
    assert not target_overlaps_goal((.09, 0), (.015, .015), (0, 0), (.05, .05), math.pi / 4)


def test_overlapping_goal_is_resampled_and_impossible_geometry_fails_closed():
    curriculum = ReachCurriculum()
    for seed in range(100):
        scene = curriculum.sample_target(seed, goal_center_world_m=(.26, 0, .75), goal_half_extents_xy_m=(.002, .002))
        assert not target_overlaps_goal(scene["target_position_world_m"][:2], (.015, .015),
                                        (.26, 0), (.002, .002), 0.)
    with pytest.raises(ValueError, match="non-overlapping"):
        curriculum.sample_target(0, goal_center_world_m=(0, 0, .75), goal_half_extents_xy_m=(1., 1.))


def test_resume_restores_frontier_history_attempt_ids_and_seeded_rehearsal():
    curriculum = ReachCurriculum()
    promote(curriculum, 0)
    for _ in range(9):
        curriculum.observe(episode(curriculum.allocate_attempt(), 1, grasp=True, lift=True))
    curriculum.allocate_attempt()  # A failed preparation still consumes its identifier.
    restored = ReachCurriculum()
    restored.load_state_dict(json.loads(json.dumps(curriculum.state_dict())))
    assert restored.state_dict() == curriculum.state_dict()
    for seed in range(30):
        assert restored.sample_target(seed, goal_center_world_m=GOAL) == curriculum.sample_target(seed, goal_center_world_m=GOAL)
    assert restored.allocate_attempt() == curriculum.allocate_attempt() == 31
    assert restored.observe(episode(21, 1, grasp=True, lift=True))["ignored_reason"] == "duplicate_attempt"


def test_sampling_contract_is_detached_json_and_explicitly_versions_geometry():
    curriculum = ReachCurriculum()
    contract = curriculum.sampling_contract()
    assert json.loads(json.dumps(contract, allow_nan=False)) == contract
    assert [(sector["inner_radius_m"], sector["outer_radius_m"])
            for sector in contract["sectors"]] == [(.24, .28), (.22, .31), (.20, .35)]
    assert contract["target_quaternion_xyzw"] == [0., 0., 0., 1.]
    assert "uniform_radius_squared" in contract["position_sampling"]
    assert "partial_overlap_and_touch" in contract["goal_exclusion"]
    contract["sectors"][0]["outer_radius_m"] = .9
    assert curriculum.sampling_contract()["sectors"][0]["outer_radius_m"] == .28


@pytest.mark.parametrize("change", ["range", "sampler_version", "missing"])
def test_resume_rejects_changed_or_missing_sampling_contract_without_mutation(change):
    curriculum = ReachCurriculum()
    curriculum.observe(episode(1))
    before = curriculum.state_dict()
    saved = json.loads(json.dumps(before))
    if change == "range":
        saved["sampling_contract"]["sectors"][0]["outer_radius_m"] = .3
    elif change == "sampler_version":
        saved["sampling_contract"]["schema_version"] += 1
    else:
        del saved["sampling_contract"]
    with pytest.raises(ValueError, match="schema|sampling contract"):
        curriculum.load_state_dict(saved)
    assert curriculum.state_dict() == before


@pytest.mark.parametrize("mutate", [
    lambda state: state.update(schema_version=0),
    lambda state: state["config"].update(pass_rate=.9),
    lambda state: state.update(level=3),
    lambda state: state.update(attempt_count=-1),
    lambda state: state.update(window=[float("nan")]),
    lambda state: state.update(seen_attempts=[1, 1]),
    lambda state: state.update(level_episode_counts=[21, 0, 0]),
    lambda state: state.update(unknown=1),
])
def test_invalid_state_is_rejected_atomically(mutate):
    curriculum = ReachCurriculum()
    curriculum.observe(episode(1))
    before = curriculum.state_dict()
    invalid = deepcopy(before)
    mutate(invalid)
    with pytest.raises(ValueError):
        curriculum.load_state_dict(invalid)
    assert curriculum.state_dict() == before


@pytest.mark.parametrize("overrides", [
    {"window_size": 19}, {"window_size": True}, {"window_size": 20.5},
    {"pass_rate": 0}, {"pass_rate": 1.1}, {"pass_rate": float("nan")},
    {"rehearsal_probability": -.1}, {"rehearsal_probability": 1},
    {"max_spawn_attempts": 0}, {"max_spawn_attempts": True},
])
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        ReachCurriculumConfig(**overrides)


@pytest.mark.parametrize("overrides", [
    {"seed": -1}, {"seed": True}, {"goal_center_world_m": (0, float("nan"), 0)},
    {"goal_center_world_m": (0, 0)}, {"goal_half_extents_xy_m": (0, .1)},
    {"goal_yaw_rad": float("inf")}, {"target_half_extents_m": (.1, .1, -1)},
    {"source_support_z_m": float("nan")}, {"spawn_clearance_m": -.01},
])
def test_invalid_spawn_inputs_are_rejected(overrides):
    arguments = {"seed": 0, "goal_center_world_m": GOAL, **overrides}
    with pytest.raises(ValueError):
        ReachCurriculum().sample_target(**arguments)
