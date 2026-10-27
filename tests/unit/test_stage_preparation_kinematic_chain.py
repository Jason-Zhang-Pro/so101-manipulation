"""Real FK/IK preparation over the actual source sector and side goal.

The arm is an ideal position servo; contact acquisition, attachment and settling
are deterministic test fixtures. A held cube follows the full rigid transform,
including its orientation and changing projected bottom. These tests catch
unreachable waypoint/axis constraints, not Gazebo contact or friction problems.
No controller movement method is overridden.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from so101_capstone.environments.stage_preparation import StagePreparationController
from so101_capstone.rewards import PickPlaceRewardConfig, TaskMeasurements, TaskStateTracker
from so101_capstone.rewards.task_state import object_geometry
from so101_capstone.robot.so101_forward_kinematics import (
    ACTIVE_JOINT_NAMES, compute_grasp_point_pose, _quaternion_xyzw_from_rotation,
)


class RigidCarryTrace:
    def __init__(self, radius, angle, evidence_period_steps=1):
        self.reward = PickPlaceRewardConfig()
        self.controller = StagePreparationController(self.reward)
        self.tracker = TaskStateTracker(self.reward)
        self.time = 0.
        self.commands = []
        self.records = []
        self.q = np.asarray(self.reward.home_joint_positions_rad)
        self.position = np.array((radius * math.cos(angle), radius * math.sin(angle), .765))
        self.rotation = np.eye(3)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.contact = False
        self.released = False
        self.relative_position = self.relative_rotation = None
        self.initial = self.tracker.reset(self.measurement(), "pick_place")
        self.cached_measurement = self.initial.measurements
        self.evidence_period_steps = evidence_period_steps

    def measurement(self):
        pose = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, self.q)))
        return TaskMeasurements(
            sim_time_s=self.time, joint_positions_rad=self.q,
            joint_velocities_rad_s=(0,) * 6, grasp_position_world_m=pose.position,
            grasp_rotation_world=pose.transform[:3, :3], target_name="source_cube",
            target_position_world_m=self.position,
            target_quaternion_xyzw=_quaternion_xyzw_from_rotation(self.rotation),
            target_linear_velocity_m_s=self.linear_velocity,
            target_angular_velocity_rad_s=self.angular_velocity,
            target_motion_valid=True, contact_valid=True,
            left_contact=self.contact, right_contact=self.contact,
            contacts_simultaneous=self.contact,
            goal_center_world_m=(.078, -.300, .7515), goal_support_z_m=.753,
        )

    def advance(self, command):
        dt = (.033, .033, .034)[len(self.commands) % 3]
        self.time += dt
        self.q = np.asarray(command)
        self.commands.append(self.q.copy())
        pose = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, self.q)))
        rotation = pose.transform[:3, :3]
        before_position, before_rotation = self.position.copy(), self.rotation.copy()
        if self.controller.phase == "CLOSE_AND_CERTIFY" and self.q[-1] <= .2 and not self.contact:
            self.contact = True
            self.relative_position = rotation.T @ (self.position - pose.position)
            self.relative_rotation = rotation.T @ self.rotation
        if self.contact:
            self.position = pose.position + rotation @ self.relative_position
            self.rotation = rotation @ self.relative_rotation
        if self.controller.phase == "RELEASE_OBJECT" and self.q[-1] > .35:
            self.contact = False
            self.released = True
        if self.released:
            # Ideal slow tabletop settling follows release; do not teleport or
            # instantaneously reorient the cube, which would correctly fail the
            # tracker's speed condition at the loss-of-contact boundary.
            bottom = self.position[2] - np.abs(self.rotation[2]) @ np.array((.015,) * 3)
            self.position[2] -= min(max(0., bottom - .753), .01 * dt)
        self.linear_velocity = (self.position - before_position) / dt
        angle = math.acos(float(np.clip((np.trace(self.rotation @ before_rotation.T) - 1) / 2, -1., 1.)))
        self.angular_velocity = np.array((angle / dt, 0., 0.))
        if len(self.commands) % self.evidence_period_steps == 0:
            self.cached_measurement = self.measurement()
        aligned = replace(self.cached_measurement, sim_time_s=self.time,
                          evidence_sim_time_s=self.cached_measurement.sim_time_s)
        snapshot = self.tracker.update(aligned, allow_success=False)
        self.records.append((self.controller.phase, snapshot))
        return snapshot


@pytest.mark.parametrize("radius", [.24, .26, .28])
@pytest.mark.parametrize("angle_degrees", [-10, 0, 10])
@pytest.mark.parametrize("evidence_period_steps", [1, 3], ids=["30hz_evidence", "10hz_evidence"])
def test_actual_sector_to_side_goal_chain_uses_measured_object_bottom(radius, angle_degrees, evidence_period_steps):
    trace = RigidCarryTrace(radius, math.radians(angle_degrees), evidence_period_steps)
    report = trace.controller.run("S5", trace.initial, trace.advance)
    assert report.certificate["s4_complete"]
    assert report.certificate["valid_release"]
    assert report.certificate["delivery_valid"]
    assert report.final_snapshot.grasp_attempt_id == 1
    assert report.steps == len(trace.commands) < 900
    records = dict(trace.records)  # Last measured state of each phase.
    for phase, required in (("VERIFY_LIFT", .805), ("TRANSPORT_CLEARANCE", .813),
                            ("CARRY_TO_GOAL", .813), ("GOAL_HOVER", .793),
                            ("LOWER_TO_SUPPORT", .754)):
        snapshot = records[phase]
        _, bottom, _ = object_geometry(snapshot.measurements)
        assert bottom == pytest.approx(required, abs=.003)
        assert snapshot.grasp_valid
    at_goal = records["CARRY_TO_GOAL"].measurements
    assert np.linalg.norm(np.asarray(at_goal.target_position_world_m[:2]) - (.078, -.300)) < .003
    # Rigid orientation really changed while carrying. A fixture that merely
    # translates an upright cube would hide the bottom-height correction bug.
    assert max(np.linalg.norm(np.asarray(s.measurements.target_quaternion_xyzw)[:3])
               for phase, s in trace.records if phase == "CARRY_TO_GOAL") > .02
    assert all(s.target_name == "source_cube" for _, s in trace.records)
