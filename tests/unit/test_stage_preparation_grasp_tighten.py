"""Hold the certified command without extra closure, using real task certificates."""
from dataclasses import replace
import math

import numpy as np
import pytest

from so101_capstone.environments.stage_preparation import (
    MIN_CONTROL_DT_S,
    StagePreparationError,
)
from so101_capstone.rewards.later_stage_reward import live_grasp
from test_stage_preparation import PhysicsTrace


class ApertureTrace(PhysicsTrace):
    """Keep measured jaw angle above its command, as an object can do."""

    def __init__(self, **kwargs):
        self.phase_commands = []
        self.certified_command = None
        self.certified_measurement = None
        super().__init__(**kwargs)

    def measurement(self):
        m = super().measurement()
        if self.contact:
            joints = list(m.joint_positions_rad)
            joints[-1] += math.radians(5)
            return replace(m, joint_positions_rad=tuple(joints))
        return m

    def advance(self, command):
        phase = self.controller.phase
        self.phase_commands.append((phase, command))
        snapshot = super().advance(command)
        if phase == "CLOSE_AND_CERTIFY" and live_grasp(snapshot):
            if self.certified_command is None:
                self.certified_command = command[-1]
                self.certified_measurement = snapshot.measurements.joint_positions_rad[-1]
        return snapshot


@pytest.mark.parametrize("stage", ["S3", "S4", "S5"])
def test_keep_certified_command_without_tightening_until_release(stage):
    trace = ApertureTrace()
    report = trace.run(stage)
    expected = trace.certified_command
    assert trace.certified_measurement - trace.certified_command == pytest.approx(math.radians(5))
    assert "TIGHTEN_GRASP" not in report.phases
    assert report.phases.index("LIFT_OBJECT") == report.phases.index("CLOSE_AND_CERTIFY") + 1
    carrying_phases = {"LIFT_OBJECT", "VERIFY_LIFT", "TRANSPORT_CLEARANCE", "CARRY_TO_GOAL",
                       "GOAL_HOVER", "LOWER_TO_SUPPORT"}
    held = [q[-1] for phase, q in trace.phase_commands if phase in carrying_phases]
    assert held and np.allclose(held, expected, atol=1e-12, rtol=0)
    commands = np.asarray([trace.initial.measurements.joint_positions_rad, *trace.commands])
    assert np.all(np.abs(np.diff(commands[:, -1])) <=
                  trace.controller.config.gripper_speed_rad_s * MIN_CONTROL_DT_S + 1e-12)
    if stage == "S5":
        assert report.certificate["s4_complete"]
        assert any(q[-1] > expected for phase, q in trace.phase_commands if phase == "RELEASE_OBJECT")


def test_grasp_loss_on_first_lift_step_still_stops_preparation():
    trace = ApertureTrace(drop_phase="LIFT_OBJECT")
    with pytest.raises(StagePreparationError, match="grasp lost") as exc:
        trace.run("S3")
    assert exc.value.phase == "LIFT_OBJECT"
    assert "TIGHTEN_GRASP" not in trace.controller.phases
    assert sum(phase == "LIFT_OBJECT" for phase, _ in trace.phase_commands) == 1


def test_s2_start_stays_open_without_certifying_or_tightening():
    trace = ApertureTrace()
    report = trace.run("S2")
    assert report.certificate["s2_start"]
    assert trace.certified_command is None
    assert "TIGHTEN_GRASP" not in report.phases
    assert "CLOSE_AND_CERTIFY" not in report.phases
    assert report.last_command_joint_targets_rad[-1] == pytest.approx(trace.controller.open_gripper)
