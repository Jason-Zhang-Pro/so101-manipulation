"""Runtime orchestration; physical preparation and core evidence have separate tests."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from so101_capstone.environments import stage_preparation as prep
from so101_capstone.environments.simulation_runtime import SimulationRuntimeError
from so101_capstone.rewards.training_stages import StageTrainingConfig
from test_act_ppo_stage_initialization import make_runtime


def system_with_preparation(monkeypatch, *, fail=False, leak=False, bad_handoff=False):
    system = make_runtime()
    events, steps = [], []

    def advance(targets):
        assert system.core.stage_preparing
        events.append("physical_step")
        end = system.io.simulation_clock_cache.latest().sim_time_s + (len(steps) + 1) / 30
        result = SimpleNamespace(
            info={"preparation_only": True, "episode_step": 0}, reward=1. if leak else 0.,
            terminated=False, truncated=False, physics_steps=33,
            end_clock=SimpleNamespace(sim_time_s=end),
            joint_state=system.io.joint_state_cache.latest(),
        )
        steps.append(result)
        return result

    class Controller:
        def __init__(self, reward, config):
            assert reward is system.core.reward_config

        def run(self, stage, initial, advance):
            assert initial is system.core.task_snapshot
            command = tuple(system.core.reward_config.home_joint_positions_rad)
            advance(command)
            advance(command)
            if fail:
                raise prep.StagePreparationError("CLOSE_AND_CERTIFY", 2, "no physical contact")
            return prep.StagePreparationReport(stage, initial, command, 2,
                                               ("SOURCE_SETTLE", "MEASURED_READY"), {"measured": True})

    def handoff(stage, config, *, command_joint_targets_rad):
        events.append("handoff")
        assert len(steps) == 2
        if bad_handoff:
            raise ValueError("invalid measured certificate")
        return {"test_observation": True}, {
            "frame_zero": True, "episode_step": 0, "stage_elapsed_s": 0,
            "training_stage": stage, "camera_source_times_s": [1., 1.],
            "target_name": "target_cube_1",
        }

    monkeypatch.setattr(prep, "StagePreparationController", Controller)
    system.runtime.coordinator.step = advance
    system.runtime.coordinator.handoff_training_stage = handoff
    return system, events, steps


@pytest.mark.parametrize("stage", ["S2", "S3", "S4", "S5"])
def test_runtime_physically_prepares_then_hands_off_at_step_zero(monkeypatch, stage):
    system, events, steps = system_with_preparation(monkeypatch)
    result = system.runtime.start_training_stage(stage, StageTrainingConfig().to_dict())
    assert events == ["physical_step", "physical_step", "handoff"]
    assert result.info["episode_step"] == result.info["stage_elapsed_s"] == 0
    assert result.info["preparation_only"] is False
    certificate = result.info["stage_initialization"]
    assert certificate["training_stage"] == stage and certificate["valid"]
    assert certificate["assisted_initialization"] and certificate["measured"]
    assert certificate["preparation_control_steps"] == 2
    assert certificate["discarded_control_steps"] == sum(system.holds) + 2
    assert result.home_reset.physics_steps == sum(system.holds) * 33 + 66
    assert result.home_reset.clock_state is steps[-1].end_clock
    assert "preparation_config" in certificate
    with pytest.raises(SimulationRuntimeError, match="unused prepared episode"):
        system.runtime.start_training_stage(stage, {})


def test_preparation_failure_is_separate_from_policy_wasted_steps(monkeypatch):
    system, events, _ = system_with_preparation(monkeypatch, fail=True)
    with pytest.raises(SimulationRuntimeError, match="no physical contact") as caught:
        system.runtime.start_training_stage("S3", {})
    assert "handoff" not in events
    metadata = caught.value.stage_initialization
    assert not metadata["valid"]
    assert metadata["training_stage"] == "S3"
    assert metadata["phase"] == "CLOSE_AND_CERTIFY"
    assert metadata["policy_steps"] == 0
    assert metadata["preparation_control_steps"] == 2
    assert not system.runtime._stage_start_available


def test_runtime_rejects_preparation_reward_leak(monkeypatch):
    system, events, _ = system_with_preparation(monkeypatch, leak=True)
    with pytest.raises(SimulationRuntimeError, match="leaked"):
        system.runtime.start_training_stage("S2", {})
    assert "handoff" not in events


def test_failed_measured_handoff_keeps_discarded_preparation_accounting(monkeypatch):
    system, _, _ = system_with_preparation(monkeypatch, bad_handoff=True)
    with pytest.raises(ValueError, match="certificate") as caught:
        system.runtime.start_training_stage("S4", {})
    assert caught.value.stage_initialization["phase"] == "HANDOFF"
    assert caught.value.stage_initialization["policy_steps"] == 0
    assert caught.value.stage_initialization["preparation_control_steps"] == 2


def test_invalid_preparation_config_rejected_before_consuming_world(monkeypatch):
    system, events, _ = system_with_preparation(monkeypatch)
    config = replace(StageTrainingConfig(), stage_preparation={"joint_speed_rad_s": 10.})
    with pytest.raises(ValueError, match="speed"):
        system.runtime.start_training_stage("S2", config.to_dict())
    assert events == [] and not system.holds
    assert system.runtime._stage_start_available


def test_incorrect_frame_zero_keeps_preparation_failure_accounting(monkeypatch):
    system, _, _ = system_with_preparation(monkeypatch)
    handoff = system.runtime.coordinator.handoff_training_stage
    def wrong_frame(*args, **kwargs):
        observation, info = handoff(*args, **kwargs)
        return observation, {**info, "episode_step": 2}
    system.runtime.coordinator.handoff_training_stage = wrong_frame
    with pytest.raises(SimulationRuntimeError, match="frame zero") as caught:
        system.runtime.start_training_stage("S4", {})
    assert caught.value.stage_initialization["phase"] == "HANDOFF"
    assert caught.value.stage_initialization["preparation_control_steps"] == 2
    assert caught.value.stage_initialization["policy_steps"] == 0
