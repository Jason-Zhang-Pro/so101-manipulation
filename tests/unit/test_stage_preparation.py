"""Preparation controller/certificates with the real TaskStateTracker.

The waypoint actuator is stubbed in semantic traces, independently of the IK
roundtrip tests. These are not Gazebo contact/dynamics validation.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.environments.stage_preparation import (
    CONTROL_DT_S, StagePreparationConfig, StagePreparationController, StagePreparationError,
)
from so101_capstone.rewards import PickPlaceRewardConfig, TaskStateTracker, TaskMeasurements
from so101_capstone.robot.so101_forward_kinematics import ACTIVE_JOINT_NAMES, compute_grasp_point_pose


class WaypointTraceController(StagePreparationController):
    def _move(self, position, *, axis=None, require_grasp=False, allow_contact=True):
        self.planned_point = np.asarray(position).copy()
        self._tick(self.command, require_grasp=require_grasp, allow_contact=allow_contact)

    def _move_object(self, desired_bottom, xy=None):
        # These semantic contact traces intentionally stub the actuator. The
        # separate kinematic-chain tests exercise the real per-step controller.
        from so101_capstone.rewards.task_state import object_geometry
        m = self.snapshot.measurements
        _, bottom, _ = object_geometry(m)
        delta = np.array((0., 0., desired_bottom - bottom))
        if xy is not None:
            delta[:2] = np.asarray(xy) - np.asarray(m.target_position_world_m[:2])
        self._move(np.asarray(m.grasp_position_world_m) + delta, require_grasp=True)


class PhysicsTrace:
    def __init__(self, *, config=None, contacts=True, drop_phase=None, stale=False):
        self.reward = PickPlaceRewardConfig()
        self.controller = WaypointTraceController(self.reward, config)
        self.tracker = TaskStateTracker(self.reward)
        self.t, self.contacts, self.drop_phase, self.stale = 0., contacts, drop_phase, stale
        self.q = self.reward.home_joint_positions_rad
        self.grasp = np.array((.174, 0., .776))
        self.position = np.array((0., 0., .765))
        self.contact = False
        self.commands = []
        self.initial = self.tracker.reset(self.measurement(), "pick_place", task_instance_id="preparation_trace")

    def measurement(self):
        return TaskMeasurements(sim_time_s=self.t, evidence_sim_time_s=0 if self.stale else self.t,
            joint_positions_rad=self.q, joint_velocities_rad_s=(0,) * 6,
            grasp_position_world_m=self.grasp, grasp_rotation_world=np.diag((1, -1, -1)),
            target_name="source_cube", target_position_world_m=self.position, target_motion_valid=True,
            contact_valid=True, left_contact=self.contact, right_contact=self.contact,
            contacts_simultaneous=self.contact)

    def advance(self, command):
        self.commands.append(command)
        self.q = command
        c = self.controller
        new_grasp = getattr(c, "planned_point", self.grasp).copy()
        if self.contact:
            self.position += new_grasp - self.grasp
        self.grasp = new_grasp
        if c.phase == "CLOSE_AND_CERTIFY" and self.contacts and command[-1] <= .2:
            self.contact = True
        if c.phase == "RELEASE_OBJECT" and command[-1] > .35:
            self.contact = False
        if c.phase == self.drop_phase:
            self.contact = False
            self.position[2] = .765
        self.t += CONTROL_DT_S
        return self.tracker.update(self.measurement())

    def run(self, stage):
        return self.controller.run(stage, self.initial, self.advance)


@pytest.mark.parametrize("stage,expected", [("s2", "s2_start"), ("s3", "s2_complete"),
                                         ("s4", "s4_start"), ("s5", "s4_complete")])
def test_physical_chain_certifies_each_start_with_one_continuous_tracker(stage, expected):
    trace = PhysicsTrace()
    report = trace.run(stage)
    assert report.stage == stage.upper()
    assert report.certificate[expected]
    assert report.steps == len(trace.commands) > 0
    assert report.phases[-1] == "READY"
    assert report.final_snapshot.task_instance_id == "preparation_trace"
    assert report.final_snapshot.target_name == "source_cube"
    if stage == "s2":
        assert not report.final_snapshot.grasp_valid
        assert "CLOSE_AND_CERTIFY" not in report.phases
    else:
        assert report.final_snapshot.grasp_attempt_id == 1
    if stage == "s5":
        assert report.certificate["valid_release"] and report.certificate["delivery_valid"]
        assert report.certificate["grasp_goal_clearance_m"] >= .08
        assert not report.final_snapshot.grasp_valid
    velocities = np.abs(np.diff(np.asarray([trace.initial.measurements.joint_positions_rad] + trace.commands), axis=0)) / CONTROL_DT_S
    assert (velocities[:, :5] <= trace.controller.config.joint_speed_rad_s + 1e-9).all()
    assert (velocities[:, 5] <= trace.controller.config.gripper_speed_rad_s + 1e-9).all()


def test_closed_empty_gripper_never_certifies_s3_and_times_out_without_teleport():
    trace = PhysicsTrace(contacts=False, config=StagePreparationConfig(phase_timeout_s=2))
    with pytest.raises(StagePreparationError, match="phase timeout") as exc:
        trace.run("S3")
    assert exc.value.phase == "CLOSE_AND_CERTIFY"
    assert trace.controller.snapshot.grasp_attempt_id == 0
    assert np.allclose(trace.position, (0, 0, .765))


@pytest.mark.parametrize("phase", ["LIFT_OBJECT", "CARRY_TO_GOAL", "LOWER_TO_SUPPORT"])
def test_actual_grasp_loss_fails_preparation_instead_of_fabricating_success(phase):
    trace = PhysicsTrace(drop_phase=phase)
    with pytest.raises(StagePreparationError, match="grasp lost") as exc:
        trace.run("S5")
    assert exc.value.phase == phase


def test_noisy_false_contacts_need_the_real_continuous_grasp_certificate():
    trace = PhysicsTrace(contacts=False, config=StagePreparationConfig(phase_timeout_s=2))
    original = trace.measurement
    def one_sided():
        # Return a tracker-generated single-finger measurement. Merely closing
        # jaws or touching one side never creates the bilateral certificate.
        return replace(original(), left_contact=trace.controller.phase == "CLOSE_AND_CERTIFY",
                       right_contact=False, contacts_simultaneous=False)
    trace.measurement = one_sided
    with pytest.raises(StagePreparationError) as exc:
        trace.run("S3")
    assert exc.value.phase == "CLOSE_AND_CERTIFY"


def test_stale_feedback_stops_before_preparation_can_claim_hold():
    trace = PhysicsTrace(stale=True)
    with pytest.raises(StagePreparationError, match="stale"):
        trace.run("S2")


def test_33_33_34_ms_steps_reuse_10hz_evidence_without_counting_duplicate_frames():
    trace = PhysicsTrace()
    controller = trace.controller
    controller.snapshot, controller.steps, controller.phase_start = trace.initial, 0, 0
    controller.phase, controller.phases = "SOURCE_SETTLE", ["SOURCE_SETTLE"]
    controller.command = np.asarray(trace.q)
    controller.floor_z = .754
    controller.identity = (trace.initial.task_instance_id, trace.initial.target_name)
    durations = [.033, .033, .034]
    evidence_times = []
    def advance(command):
        trace.q = command
        trace.t += durations[len(evidence_times) % 3]
        evidence = math.floor((trace.t + 1e-8) * 10) / 10
        evidence_times.append(evidence)
        return trace.tracker.update(replace(trace.measurement(), evidence_sim_time_s=evidence))
    controller.advance = advance
    controller._hold(controller._source_ready, .1)
    assert controller.steps == 3
    assert evidence_times == [0, 0, .1]
    assert controller.snapshot.sim_time_s == pytest.approx(.1)


def test_preparation_error_exposes_machine_readable_failure_record():
    trace = PhysicsTrace(stale=True)
    with pytest.raises(StagePreparationError) as exc:
        trace.run("S2")
    record = exc.value.stage_initialization
    assert record["phase"] == "SOURCE_SETTLE"
    assert record["preparation_control_steps"] == exc.value.steps
    assert "stale" in record["reason"]


def test_initial_proxy_or_missing_motion_evidence_is_rejected_without_commands():
    trace = PhysicsTrace()
    for measurement in (replace(trace.initial.measurements, grasp_evidence_mode="proxy"),
                        replace(trace.initial.measurements, target_motion_valid=False)):
        with pytest.raises(StagePreparationError):
            trace.controller.run("S3", replace(trace.initial, measurements=measurement), trace.advance)
    assert trace.commands == []


def test_real_waypoint_guard_rejects_below_support_path_before_advance():
    trace = PhysicsTrace()
    controller = StagePreparationController(trace.reward)
    controller.snapshot, controller.steps, controller.phase_start = trace.initial, 0, 0
    controller.phase, controller.phases = "PATH_GUARD", []
    controller.command = np.asarray(trace.q)
    controller.floor_z = 2.0
    controller.advance = trace.advance
    with pytest.raises(StagePreparationError, match="support-plane guard"):
        controller._tick(controller.command)
    assert trace.commands == []


def test_s2_closed_loop_commands_reach_source_using_actual_fk_without_waypoint_stub():
    reward = PickPlaceRewardConfig()
    controller = StagePreparationController(reward)
    tracker = TaskStateTracker(reward)
    commands, heights = [], []
    time = 0.0
    def measurement(q):
        pose = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q)))
        return TaskMeasurements(sim_time_s=time, joint_positions_rad=q,
            joint_velocities_rad_s=(0,) * 6, grasp_position_world_m=pose.position,
            grasp_rotation_world=pose.transform[:3, :3], target_name="cube",
            target_position_world_m=(.18, 0, .765), target_motion_valid=True, contact_valid=True)
    initial = tracker.reset(measurement(reward.home_joint_positions_rad), "pick_place")
    def advance(command):
        nonlocal time
        time += CONTROL_DT_S
        commands.append(command)
        m = measurement(command)
        heights.append(m.grasp_position_world_m[2])
        return tracker.update(m)
    report = controller.run("S2", initial, advance)
    assert report.certificate["s2_start"]
    assert max(heights) >= .83 - controller.config.cartesian_tolerance_m
    assert min(heights) >= .75 + controller.config.floor_clearance_m
    assert report.final_snapshot.target_distance_m < .02
    assert report.final_snapshot.measurements.target_position_world_m == (.18, 0, .765)
    assert report.final_snapshot.grasp_attempt_id == 0
    assert len(commands) < 600


@pytest.mark.parametrize("kwargs", [dict(joint_speed_rad_s=10), dict(gripper_speed_rad_s=2),
    dict(cartesian_speed_m_s=.2), dict(phase_timeout_s=0), dict(total_timeout_s=601),
    dict(pregrasp_offset_z_m=.02), dict(goal_hover_clearance_m=.02), dict(axis_tolerance_rad=math.pi),
    dict(joint_speed_rad_s=True), dict(joint_tolerance_rad=float("nan")), dict(empty_raise_clearance_m=.04)])
def test_invalid_preparation_configuration(kwargs):
    with pytest.raises(ValueError):
        StagePreparationConfig(**kwargs)


def test_unknown_stage_rejected_before_any_motion():
    trace = PhysicsTrace()
    with pytest.raises(ValueError, match="supports S2--S5"):
        trace.run("S6")
    assert not trace.commands
