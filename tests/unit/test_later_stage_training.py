"""Real sensor-history reward handoffs; no simulator or hardware required."""
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from so101_capstone.environments.reach_env_core import (
    ReachEnvironmentConfig, ReachEnvironmentStateError, So101ReachEnvironmentCore,
)
from so101_capstone.environments.state_cache import MEASURED_JOINT_NAMES, JointStateSnapshot, TargetCubeSnapshot
from so101_capstone.environments.simulation_step_coordinator import (
    DeterministicSimulationStepCoordinator, SimulationStepCoordinatorError,
)
from so101_capstone.rewards.later_stage_training import (
    LaterStageTraining, later_stage_context_dim, later_stage_context_fields,
    later_stage_critic_context, later_stage_training_contract,
)
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardInput
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM
from test_task_state import Trace


DT = 1 / 30
STAGES = StageTrainingConfig(enable_later_stage_context=True, s0_home_hold_s=1.,
    s0_home_joint_tolerance_rad=math.radians(8), s0_success_mode="position_hold")


def prepare(stage):
    trace = Trace()
    if stage == "S2":
        trace.step(grasp_position_world_m=(0., 0., .783), dt=DT, at_home=False)
    else:
        trace.acquire_and_lift()
        trace.step((0., 0., .82), contact=True, dt=DT, repeat=4, at_home=False)
        if stage in {"S4", "S5"}:
            trace.step((.25, 0., .82), contact=True, dt=DT, repeat=4, at_home=False)
        if stage == "S5":
            trace.step((.25, 0., .768), contact=True, dt=DT, at_home=False)
            trace.step(contact=False, dt=DT, repeat=5, at_home=False)
    return trace


def update(adapter, trace, **kwargs):
    previous = trace.snapshot
    current = trace.step(dt=DT, **kwargs)
    return adapter.step(PickPlaceRewardInput(previous, current, DT,
                         terminated=current.outcome == "task_failure"))


@pytest.mark.parametrize("stage", ["S2", "S3", "S4", "S5"])
def test_start_certificate_preserves_real_provenance_and_starts_empty_dwell(stage):
    trace = prepare(stage)
    before = trace.snapshot
    adapter = LaterStageTraining(stage, STAGES, trace.config)
    info = adapter.reset(before)
    assert info["start_valid"] and info["assisted_initialization"]
    assert info["stage_result"] == "RUNNING" and info["stage_elapsed_s"] == 0
    assert trace.snapshot is before
    assert all(v == 0 for v in adapter.reward.reward_state()["dwell_s"].values())
    context = later_stage_critic_context(stage, elapsed_s=0, budget_s=adapter.budget_s,
                                        snapshot=before, adapter=adapter)
    assert context.shape == (later_stage_context_dim(),)
    assert context[0] == int(stage[1]) and np.isfinite(context).all()
    assert len(set(later_stage_context_fields())) == later_stage_context_dim()


@pytest.mark.parametrize("stage", ["S3", "S4", "S5"])
def test_joint_pose_without_real_grasp_or_release_history_cannot_start_later_stage(stage):
    trace = Trace()
    adapter = LaterStageTraining(stage, STAGES, trace.config)
    with pytest.raises(ValueError, match="initial state"):
        adapter.reset(trace.snapshot)


def test_s2_real_grasp_then_5cm_lift_cuts_with_bootstrap_not_task_success():
    trace = prepare("S2")
    adapter = LaterStageTraining("S2", STAGES, trace.config)
    adapter.reset(trace.snapshot)
    for _ in range(3):
        result, info = update(adapter, trace, contact=True, at_home=False)
    for _ in range(4):
        result, info = update(adapter, trace, position=(0., 0., .82), contact=True, at_home=False)
    assert info["stage_success"] and info["stage_cut"]
    assert result.truncated and not result.terminated
    assert result.components["s2_stage_complete"] == 20
    assert result.potential_after > 0  # sampling boundary does not erase value potential
    assert trace.snapshot.outcome == "running"
    with pytest.raises(RuntimeError, match="handoff"):
        update(adapter, trace, contact=True, at_home=False)


def test_s3_hold_inside_and_s4_release_use_their_own_success_contract():
    trace = prepare("S3")
    adapter = LaterStageTraining("S3", STAGES, trace.config)
    adapter.reset(trace.snapshot)
    for _ in range(4):
        result, info = update(adapter, trace, position=(.25, 0., .82), contact=True, at_home=False)
    assert info["stage_success"] and result.truncated
    # S4 is a separate stage baseline, preserving exactly the carried object.
    adapter = LaterStageTraining("S4", STAGES, trace.config)
    adapter.reset(trace.snapshot)
    update(adapter, trace, position=(.25, 0., .768), contact=True, at_home=False)
    for _ in range(4):
        result, info = update(adapter, trace, contact=False, at_home=False)
    assert info["stage_success"] and result.truncated
    assert not trace.snapshot.placed_stable  # full task's old .4s does not delay S4 .1s
    assert result.potential_after > 0


def test_s5_uses_same_home_contract_and_placement_remains_required():
    trace = prepare("S5")
    adapter = LaterStageTraining("S5", STAGES, trace.config)
    adapter.reset(trace.snapshot)
    for _ in range(31):
        result, info = update(adapter, trace, at_home=True, joint_velocities_rad_s=(.5,) * 6)
    assert info["stage_success"] and result.truncated
    assert adapter.reward.home_success.config is STAGES
    assert trace.snapshot.outcome == "running"


def test_timeout_and_physical_failure_have_distinct_return_semantics():
    trace = prepare("S2")
    adapter = LaterStageTraining("S2", replace(STAGES, s2_budget_s=DT * 2), trace.config)
    adapter.reset(trace.snapshot)
    update(adapter, trace, at_home=False)
    result, info = update(adapter, trace, at_home=False)
    assert info["stage_timeout"] and result.truncated and not result.terminated
    trace = prepare("S2")
    adapter = LaterStageTraining("S2", STAGES, trace.config)
    adapter.reset(trace.snapshot)
    result, info = update(adapter, trace, position=(0., 0., .5), at_home=False)
    assert info["stage_result"] == "FAILURE" and not info["stage_cut"]
    assert result.terminated and not result.truncated and result.potential_after == 0


def test_late_hold_completion_does_not_pay_success_after_budget():
    trace = prepare("S3")
    adapter = LaterStageTraining("S3", replace(STAGES, s3_budget_s=.12), trace.config)
    adapter.reset(trace.snapshot)
    for _ in range(4):
        result, info = update(adapter, trace, position=(.25, 0., .82), contact=True, at_home=False)
    assert info["stage_timeout"] and not result.stage_complete
    assert result.components["s3_stage_complete"] == result.components["s3_transported_complete"] == 0
    assert result.reward == pytest.approx(sum(result.components.values()))


def test_tracker_success_gate_preserves_contact_history_and_never_suppresses_failure():
    trace = prepare("S5")
    for _ in range(140):
        trace.t += DT
        trace.snapshot = trace.tracker.update(trace.measurement(), allow_success=False)
    assert trace.snapshot.placed_stable and trace.snapshot.home_stable_duration_s >= 3
    assert trace.snapshot.outcome == "running" and trace.snapshot.delivery_valid
    trace.t += DT
    trace.snapshot = trace.tracker.update(trace.measurement(failure_reason="forced_sensor_failure"),
                                           allow_success=False)
    assert trace.snapshot.outcome == "task_failure"


class CoreTrace:
    """Exercise real core/tracker/rewards; stub only sensor acquisition/FK input."""
    def __init__(self, monkeypatch):
        self.trace = Trace()
        self.core = So101ReachEnvironmentCore(ReachEnvironmentConfig(max_episode_steps=1), self.trace.config,
            reward_profile="pick_place_v2", goal_region_center_world_m=(.25, 0., .753))
        self.time = 0.
        self.seq = 0
        self.position = (0., 0., .765)
        self.measurement = self.trace.measurement()
        monkeypatch.setattr(self.core, "_task_measurements", lambda *a, **kw: self.measurement)
        self.frame(reset=True)
        self.core.begin_stage_preparation()

    def frame(self, *, reset=False, contact=False, position=None, grasp=None, **extra):
        if not reset:
            self.time += DT
        self.seq += 1
        if position is not None:
            self.position = position
        self.measurement = self.trace.measurement(sim_time_s=self.time, target_position_world_m=self.position,
            grasp_position_world_m=grasp or (self.position if contact else (0., 0., 1.1)),
            left_contact=contact, right_contact=contact, contacts_simultaneous=contact, **extra)
        q = self.measurement.joint_positions_rad
        joint = JointStateSnapshot(dict(zip(MEASURED_JOINT_NAMES, q)), dict.fromkeys(MEASURED_JOINT_NAMES, 0.),
                                   self.seq, self.time, self.time)
        cube = TargetCubeSnapshot("cube", np.asarray(self.position), np.asarray((0., 0., 0., 1.)),
                                   self.seq, self.time, self.seq, self.time)
        fn = self.core.reset_from_feedback if reset else self.core.step_from_feedback
        return fn(joint, cube, now_monotonic_s=self.time, step_end_sim_time_s=self.time,
                  command_joint_targets_rad=q)


def test_core_preparation_ignores_policy_cap_and_handoff_preserves_grasp_history(monkeypatch):
    stream = CoreTrace(monkeypatch)
    core = stream.core
    monkeypatch.setattr(core._pick_place_reward, "step", lambda *a: pytest.fail("preparation scored reward"))
    for _ in range(4):
        _, reward, terminal, truncated, info = stream.frame(contact=True)
        assert reward == 0 and info["preparation_only"] and info["episode_step"] == 0
        assert not terminal and not truncated
    for _ in range(4):
        stream.frame(contact=True, position=(0., 0., .82))
    before = core.task_snapshot
    assert before.grasp_valid and before.attempt_lifted
    _, info = core.handoff_training_stage("S3", STAGES, command_joint_targets_rad=before.measurements.joint_positions_rad)
    assert core.task_snapshot is before and core.episode_step == 0
    assert info["stage_elapsed_s"] == 0 and info["reward_components"] == {}
    assert len(info["critic_task_context"]) == TASK_CONTEXT_DIM + later_stage_context_dim()
    for step in range(1, 5):
        _, reward, terminal, truncated, info = stream.frame(contact=True, position=(.25, 0., .82))
        assert info["episode_step"] == step
    assert truncated and not terminal and info["stage_success"]
    assert set(info["reward_components"]) and all(k.startswith("s3_") for k in info["reward_components"])


def test_later_reset_rejected_and_failed_preparation_cannot_handoff(monkeypatch):
    stream = CoreTrace(monkeypatch)
    _, _, terminal, _, info = stream.frame(position=(0., 0., .5))
    assert terminal and info["preparation_only"] and info["episode_step"] == 0
    with pytest.raises(ReachEnvironmentStateError, match="running"):
        stream.core.begin_stage_preparation()
    with pytest.raises(ReachEnvironmentStateError, match="initial state"):
        stream.core.handoff_training_stage("S2", STAGES)
    core = So101ReachEnvironmentCore(reward_profile="pick_place_v2")
    core.configure_training_stage("S3", STAGES)
    with pytest.raises(ReachEnvironmentStateError, match="provenance"):
        core.reset_from_feedback(None, now_monotonic_s=0)


def test_config_preserves_old_hash_payload_and_stage_budget_and_contract():
    old = StageTrainingConfig().to_dict()
    assert all(key not in old for key in ("s2_budget_s", "s3_budget_s", "s4_budget_s", "stage_preparation", "enable_later_stage_context"))
    cfg = replace(STAGES, s0_budget_s=8, s2_budget_s=12)
    assert cfg.stage_budget_s("S5") == 8 and cfg.stage_budget_s("S2") == 12
    assert cfg == StageTrainingConfig.from_dict(cfg.to_dict())
    contract = later_stage_training_contract(cfg, Trace().config)
    assert contract["stages"]["S2"]["config"]["lift_rise_m"] == .05
    assert contract["stages"]["S5"]["budget_s"] == 8


def test_coordinator_handoff_reuses_synced_frame_without_reset(monkeypatch):
    coordinator = object.__new__(DeterministicSimulationStepCoordinator)
    clock = SimpleNamespace(sim_time_s=10., sequence=4)
    coordinator.simulation_clock_cache = SimpleNamespace(latest=lambda: clock)
    coordinator.config = SimpleNamespace(max_camera_lag_from_step_end_s=.04)
    info = dict(preparation_only=True, camera_lag_from_step_end_s=.02, episode_step=0)
    frames = object()
    coordinator._last_preparation_step = SimpleNamespace(info=info, terminated=False, truncated=False,
                                                        end_clock=clock, camera_frames=frames)
    coordinator.environment = SimpleNamespace(handoff_training_stage=lambda *a, **kw: (
        "state", dict(preparation_only=False, training_stage="S3", episode_step=0)))
    calls = []
    coordinator.schedule = SimpleNamespace(reset=lambda: calls.append("schedule"))
    monkeypatch.setattr(coordinator, "_build_policy_observation", lambda state, camera: (state, camera))
    observation, info = coordinator.handoff_training_stage("S3", STAGES)
    assert observation == ("state", frames) and info["frame_zero"] and not info["preparation_only"]
    assert calls == ["schedule"] and coordinator._last_preparation_step is None
    with pytest.raises(SimulationStepCoordinatorError, match="preparation"):
        coordinator.handoff_training_stage("S3", STAGES)


@pytest.mark.parametrize("stale", ["clock", "camera", "failure"])
def test_coordinator_rejects_unverified_handoff(stale):
    coordinator = object.__new__(DeterministicSimulationStepCoordinator)
    clock = SimpleNamespace(sim_time_s=10., sequence=4)
    coordinator.simulation_clock_cache = SimpleNamespace(latest=lambda: clock)
    coordinator.config = SimpleNamespace(max_camera_lag_from_step_end_s=.04)
    coordinator._last_preparation_step = SimpleNamespace(
        info=dict(preparation_only=True, camera_lag_from_step_end_s=.1 if stale == "camera" else .02),
        terminated=stale == "failure", truncated=False,
        end_clock=SimpleNamespace(sim_time_s=9. if stale == "clock" else 10.))
    with pytest.raises(SimulationStepCoordinatorError):
        coordinator.handoff_training_stage("S3", STAGES)
