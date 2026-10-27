"""Independent numeric/trajectory checks for schema-5 history rewards."""

from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.rewards import (
    PickPlaceReward, PickPlaceRewardConfig, PickPlaceRewardInput,
    TaskMeasurements, TaskPhase, TaskStateTracker,
)


DT = 1 / 30


def reach_state(cfg, t=0., *, distance=.30, x_degrees=0., z_degrees=20.,
                opening=.7, **evidence):
    """Real geometry with independently controlled X-plane and Z-bearing errors.

    z_degrees must be >= x_degrees: schema 5 additionally requires the local
    Z axis to point down. No reward-private methods or history are mocked.
    """
    angle = math.radians(x_degrees)
    ry = np.array([[math.cos(angle), 0., math.sin(angle)],
                   [0., 1., 0.], [-math.sin(angle), 0., math.cos(angle)]])
    rotation = ry @ np.diag([1., -1., -1.])
    local_x, local_z = rotation[:, 0], rotation[:, 2]
    beta = math.radians(z_degrees)
    offset = cfg.approach_reference_offset_m
    length = offset * math.cos(beta) + math.sqrt(
        distance**2 - (offset * math.sin(beta))**2
    )
    grasp = np.array([0., 0., .9])
    target = grasp - offset * local_z + length * (
        math.cos(beta) * local_z + math.sin(beta) * local_x
    )
    q = list(cfg.home_joint_positions_rad)
    q[-1] += opening
    values = dict(target_name="cube", target_position_world_m=target,
                  target_motion_valid=True, contact_valid=True)
    values.update(evidence)
    measured = TaskMeasurements(t, q, (0.,) * 6, grasp, rotation, **values)
    snapshot = TaskStateTracker(cfg).reset(measured, task_instance_id="schema5")
    return replace(snapshot, phase=TaskPhase.APPROACH)


def home_state(cfg, t=0., *, error_degrees=5., phase=TaskPhase.WAIT_HOME,
               velocity=0., **evidence):
    error = math.radians(error_degrees)
    q = np.asarray(cfg.home_joint_positions_rad) + error
    measured = TaskMeasurements(t, q, (velocity,) * 6, (0., 0., 1.),
                                np.eye(3), contact_valid=True, **evidence)
    mode = "idle_empty" if phase is TaskPhase.WAIT_HOME else "pick_place"
    snapshot = TaskStateTracker(cfg).reset(
        measured, mode, task_instance_id="schema5"
    )
    return replace(snapshot, phase=phase)


def step(evaluator, previous, current, **flags):
    return evaluator.step(PickPlaceRewardInput(
        previous, current, current.sim_time_s - previous.sim_time_s, **flags
    ))


def evaluator_at(initial, cfg):
    evaluator = PickPlaceReward(cfg)
    evaluator.reset(initial)
    return evaluator


def test_distance_new_best_and_no_progress_branches_are_exclusive():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg)
    evaluator = evaluator_at(previous, cfg)
    cases = [(.29, 1., 0.), (.29, 0., -100 * .29**2),
             (.30, 0., -9.), (.285, .5, 0.)]
    for index, (distance, progress, stagnation) in enumerate(cases, 1):
        current = reach_state(cfg, index * DT, distance=distance)
        result = step(evaluator, previous, current)
        assert result.components["reach_progress"] == pytest.approx(progress)
        assert result.components["reach_stagnation"] == pytest.approx(stagnation)
        assert result.components["reach_step"] == pytest.approx(-.5)
        previous = current


@pytest.mark.parametrize("axis, fixed, values", [
    ("x", {"z_degrees": 30.}, [2.49, 2.01, 1.99, 3.01, 3.01]),
    ("z", {"x_degrees": 0.}, [20.49, 20.01, 19.99, 21.01, 21.01]),
])
def test_half_degree_floor_progress_and_persistent_regression(axis, fixed, values):
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    kwargs = {**fixed, f"{axis}_degrees": values[0]}
    previous = reach_state(cfg, **kwargs)
    evaluator = evaluator_at(previous, cfg)
    for index, (angle, gain, penalty) in enumerate(zip(
        values[1:], [0., .1, 0., 0.], [0., 0., -1.5, -1.5]
    ), 1):
        current = reach_state(cfg, index * DT, **{**fixed, f"{axis}_degrees": angle})
        result = step(evaluator, previous, current)
        assert result.components[f"reach_{axis}_alignment_progress"] == pytest.approx(gain)
        assert result.components[f"reach_{axis}_alignment_regression"] == pytest.approx(penalty)
        previous = current


def test_z_perfect_bonus_is_once_and_reset_inside_band_is_not_a_free_bonus():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg, z_degrees=4.1)
    evaluator = evaluator_at(previous, cfg)
    for index, (angle, bonus) in enumerate([(2.9, 5.), (4.1, 0.), (2.9, 0.)], 1):
        current = reach_state(cfg, index * DT, z_degrees=angle)
        assert step(evaluator, previous, current).components["reach_z_alignment_perfect"] == bonus
        previous = current
    evaluator.reset(reach_state(cfg, z_degrees=2.9))
    result = step(evaluator, reach_state(cfg, z_degrees=2.9), reach_state(cfg, DT, z_degrees=2.9))
    assert result.components["reach_z_alignment_perfect"] == 0.


@pytest.mark.parametrize("distance, angle, penalty", [
    (.059, 10.1, -5.), (.061, 10.1, 0.), (.059, 9.9, 0.),
])
def test_close_range_misalignment_penalty_has_explicit_gates(distance, angle, penalty):
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg, distance=distance, z_degrees=angle)
    evaluator = evaluator_at(previous, cfg)
    current = reach_state(cfg, DT, distance=distance, z_degrees=angle)
    assert step(evaluator, previous, current).components["reach_close_alignment"] == penalty


def test_threshold_rewards_are_once_and_multi_crossing_is_times_one_point_five():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg, distance=.30)
    evaluator = evaluator_at(previous, cfg)
    current = reach_state(cfg, DT, distance=.14)
    assert step(evaluator, previous, current).components["reach_threshold"] == pytest.approx((.1+.2+.3)*1.5)
    previous, current = current, reach_state(cfg, 2*DT, distance=.30)
    assert step(evaluator, previous, current).components["reach_threshold"] == 0.
    previous, current = current, reach_state(cfg, 3*DT, distance=.14)
    assert step(evaluator, previous, current).components["reach_threshold"] == 0.
    # Reset starts a genuinely new reward history.
    previous = reach_state(cfg, distance=.30)
    evaluator.reset(previous)
    current = reach_state(cfg, DT, distance=.21)
    assert step(evaluator, previous, current).components["reach_threshold"] == pytest.approx(.1)


def test_initially_inside_distance_thresholds_does_not_allow_exit_reentry_farming():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg, distance=.14)
    evaluator = evaluator_at(previous, cfg)
    for index, distance in enumerate([.30, .14, .30, .14], 1):
        current = reach_state(cfg, index*DT, distance=distance)
        assert step(evaluator, previous, current).components["reach_threshold"] == 0.
        previous = current


@pytest.mark.parametrize("invalid", [
    {"evidence_valid": False}, {"target_measurement_valid": False},
    {"joint_measurement_valid": False}, {"target_motion_valid": False},
    {"evidence_sim_time_s": 0.},
])
def test_invalid_or_reused_evidence_cannot_write_best_or_award_bonuses(invalid):
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    initial = reach_state(cfg)
    evaluator = evaluator_at(initial, cfg)
    bogus = reach_state(cfg, DT, distance=.01, z_degrees=2., **invalid)
    result = step(evaluator, initial, bogus)
    for name in ("reach_progress", "reach_threshold", "reach_z_alignment_perfect",
                 "reach_x_alignment_progress", "reach_z_alignment_progress",
                 "reach_gripper_progress"):
        assert result.components[name] == 0.
    assert result.components["reach_stagnation"] == pytest.approx(-9.)
    recovered = reach_state(cfg, 2*DT, distance=.29, z_degrees=2.)
    result = step(evaluator, bogus, recovered)
    assert result.components["reach_progress"] == pytest.approx(1.)
    assert result.components["reach_threshold"] == 0.
    assert result.components["reach_z_alignment_perfect"] == 5.


def test_repeated_close_cannot_reaward_the_same_best_closure():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg, distance=.005, z_degrees=0., opening=1.2)
    evaluator = evaluator_at(previous, cfg)
    progress = []
    regression = []
    for index, opening in enumerate([.8, 1.2, .8], 1):
        current = reach_state(cfg, index*DT, distance=.005, z_degrees=0., opening=opening)
        result = step(evaluator, previous, current)
        progress.append(result.components["reach_gripper_progress"])
        regression.append(result.components["reach_gripper_regression"])
        previous = current
    assert progress[0] > 0.
    assert progress[1:] == [0., 0.]
    assert regression[1] < 0.


def test_closing_history_survives_gate_switches_and_far_closed_pose_does_not_block_it():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg, distance=.10, z_degrees=0., opening=0.)
    evaluator = evaluator_at(previous, cfg)
    observed = []
    for index, (distance, opening) in enumerate([
        (.005, 1.2), (.005, .8), (.10, 1.2), (.005, .8),
    ], 1):
        current = reach_state(cfg, index*DT, distance=distance, z_degrees=0., opening=opening)
        result = step(evaluator, previous, current)
        observed.append(result.components["reach_gripper_progress"])
        previous = current
    assert observed[0] == 0.  # First close-gated observation is a baseline.
    assert observed[1] > 0.  # Actual later closing improves its own history.
    assert observed[3] == 0.  # Leaving/re-entering the gate cannot reset it.


def test_first_valid_geometry_after_invalid_reset_only_establishes_history():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    initial = reach_state(cfg, evidence_valid=False)
    evaluator = evaluator_at(initial, cfg)
    first = reach_state(cfg, DT, distance=.01, z_degrees=2.)
    result = step(evaluator, initial, first)
    for name in ("reach_progress", "reach_threshold", "reach_z_alignment_perfect",
                 "reach_x_alignment_progress", "reach_z_alignment_progress"):
        assert result.components[name] == 0.
    second = reach_state(cfg, 2*DT, distance=.009, z_degrees=2.)
    assert step(evaluator, first, second).components["reach_progress"] == pytest.approx(.1)


@pytest.mark.parametrize("dt", [DT, 2*DT])
def test_process_costs_scale_with_time_but_measured_gains_do_not(dt):
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    initial = reach_state(cfg)
    evaluator = evaluator_at(initial, cfg)
    still = reach_state(cfg, dt)
    result = step(evaluator, initial, still)
    assert result.components["reach_step"] == pytest.approx(-.5*dt*30)
    assert result.components["reach_stagnation"] == pytest.approx(-9*dt*30)
    closer = reach_state(cfg, 2*dt, distance=.29)
    result = step(evaluator, still, closer)
    assert result.components["reach_progress"] == pytest.approx(1.)


def test_home_history_improvement_stagnation_regression_and_reset():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = home_state(cfg, error_degrees=5.)
    evaluator = evaluator_at(previous, cfg)
    current = home_state(cfg, DT, error_degrees=4.)
    result = step(evaluator, previous, current)
    assert result.components["home_progress"] == pytest.approx(10*math.radians(1.)/.5)
    assert result.components["home_stagnation"] == 0.
    previous, current = current, home_state(cfg, 2*DT, error_degrees=4.)
    result = step(evaluator, previous, current)
    e = math.radians(3.)/.5
    assert result.components["home_progress"] == 0.
    assert result.components["home_stagnation"] == pytest.approx(-.5*e*e)
    previous, current = current, home_state(cfg, 3*DT, error_degrees=5.)
    result = step(evaluator, previous, current)
    assert result.components["home_regression"] == pytest.approx(-2*(math.radians(1.)/.5)**2)
    initial = home_state(cfg, error_degrees=5.)
    evaluator.reset(initial)
    result = step(evaluator, initial, home_state(cfg, DT, error_degrees=4.))
    assert result.components["home_progress"] == pytest.approx(10*math.radians(1.)/.5)


@pytest.mark.parametrize("invalid", [{"evidence_valid": False}, {"evidence_sim_time_s": 0.}])
def test_home_invalid_or_reused_observation_cannot_create_false_history(invalid):
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    initial = home_state(cfg, error_degrees=5.)
    evaluator = evaluator_at(initial, cfg)
    bogus_home = home_state(cfg, DT, error_degrees=0., **invalid)
    result = step(evaluator, initial, bogus_home)
    assert result.components["home_progress"] == 0.
    assert result.components["home_stagnation"] == pytest.approx(-.5*(math.radians(4.)/.5)**2)
    assert result.components["home_unsettled"] == -.5
    recovered = home_state(cfg, 2*DT, error_degrees=4.)
    result = step(evaluator, bogus_home, recovered)
    assert result.components["home_progress"] == pytest.approx(10*math.radians(1.)/.5)


def test_history_context_is_versioned_complete_and_resets_without_external_aliasing():
    from so101_capstone.rewards.pick_place_reward import (
        REWARD_HISTORY_CONTEXT_DIM, REWARD_HISTORY_CONTEXT_SCHEMA_VERSION,
    )

    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    initial = reach_state(cfg)
    evaluator = evaluator_at(initial, cfg)
    baseline = evaluator.critic_history_context()
    assert REWARD_HISTORY_CONTEXT_SCHEMA_VERSION == 1
    assert baseline.shape == (REWARD_HISTORY_CONTEXT_DIM,) == (33,)
    assert baseline.dtype == np.float32 and np.isfinite(baseline).all()
    current = reach_state(cfg, DT, distance=.14, z_degrees=2.)
    step(evaluator, initial, current)
    changed = evaluator.critic_history_context()
    assert not np.array_equal(changed, baseline)
    changed[:] = 123.
    assert not np.array_equal(evaluator.critic_history_context(), changed)
    evaluator.reset(initial)
    np.testing.assert_array_equal(evaluator.critic_history_context(), baseline)
    for schema in (3, 4):
        historical = PickPlaceReward(PickPlaceRewardConfig(reward_schema_version=schema))
        historical.reset(initial)
        assert historical.critic_history_context().shape == (0,)


@pytest.mark.parametrize("phase", [TaskPhase.WAIT_HOME, TaskPhase.HOME_HOLD])
@pytest.mark.parametrize("error_degrees", [0., .5, .999])
def test_home_hold_inside_tolerance_has_no_stagnation_incentive_to_leave(phase, error_degrees):
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = home_state(cfg, error_degrees=error_degrees, phase=phase)
    evaluator = evaluator_at(previous, cfg)
    for index in range(1, 5):
        current = home_state(cfg, index*DT, error_degrees=error_degrees, phase=phase)
        result = step(evaluator, previous, current)
        for name in ("home_progress", "home_stagnation", "home_regression", "home_unsettled"):
            assert result.components[name] == pytest.approx(0.)
        previous = current


def test_schema5_terminal_and_truncation_keep_the_existing_contract():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    for outcome, terminated, truncated, expected in [
        ("success", True, False, 100.), ("task_failure", True, False, -10.),
        ("running", False, True, 0.),
    ]:
        previous = reach_state(cfg)
        evaluator = evaluator_at(previous, cfg)
        current = replace(reach_state(cfg, DT, distance=.29), outcome=outcome,
                          success_event=outcome == "success")
        result = step(evaluator, previous, current, terminated=terminated, truncated=truncated)
        assert result.components["success"] + result.components["failure"] == expected
        assert result.terminated is terminated and result.truncated is truncated
        assert result.potential_after == (0. if terminated else evaluator.potential(current))
        with pytest.raises(RuntimeError, match="reset"):
            step(evaluator, current, reach_state(cfg, 2*DT))


def test_schema5_does_not_accept_an_uncertified_success_event():
    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    previous = reach_state(cfg)
    evaluator = evaluator_at(previous, cfg)
    current = replace(reach_state(cfg, DT), outcome="success", success_event=False)
    with pytest.raises(ValueError, match="success event"):
        step(evaluator, previous, current, terminated=True)


@pytest.mark.parametrize("outcome", ["success", "task_failure"])
def test_schema5_preserves_contact_certified_physical_outcomes(outcome):
    from test_task_state import Trace

    cfg = PickPlaceRewardConfig(reward_schema_version=5)
    trace = Trace(config=cfg)
    evaluator = evaluator_at(trace.snapshot, cfg)
    results = []

    def physical_step(position=None, **kwargs):
        previous = trace.snapshot
        current = trace.step(position, **kwargs)
        result = step(evaluator, previous, current,
                      terminated=current.outcome in ("success", "task_failure"))
        results.append(result)
        return current

    for _ in range(3):
        physical_step((0., 0., .765), contact=True, at_home=False)
    for _ in range(3):
        current = physical_step((0., 0., .805), contact=True, at_home=False)
    assert current.grasp_valid and current.attempt_lifted
    if outcome == "task_failure":
        current = physical_step((0., 0., .5), at_home=False)
        assert current.outcome == "task_failure"
        assert results[-1].components["failure"] == -10.
        assert all(result.components["success"] == 0. for result in results)
        return

    for _ in range(2):
        physical_step((.25, 0., .808), contact=True, at_home=False)
    physical_step((.25, 0., .768), contact=True, at_home=False)
    current = physical_step(at_home=False)
    assert current.valid_release and current.delivery_valid
    for _ in range(10):
        current = physical_step(at_home=False)
    assert current.placed_stable
    for _ in range(65):
        current = physical_step()
        if current.outcome == "success":
            break
    assert current.outcome == "success" and current.to_info()["formal_success"]
    assert results[-1].components["success"] == 100.
    assert all(result.components["success"] == 0. for result in results[:-1])
