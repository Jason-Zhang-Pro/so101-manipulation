"""Lost-step integration tests without starting Gazebo or a worker process."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_capstone.training import act_ppo_multiworld_gazebo as module
from so101_capstone.training.act_ppo import ActPpoActorObservation, ActPpoRolloutBuffer
from so101_capstone.training.act_ppo_discard_stats import DiscardedStepStats
from so101_capstone.training.act_ppo_gazebo import ActPpoGazeboConfig, _act_ppo_config
from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerCommunicationError


def diagnostics(confirmed):
    return {"confirmed": confirmed, "status": "confirmed" if confirmed else "not_found",
            "evidence": ["[ign gazebo-1] Segmentation fault"] if confirmed else []}


@pytest.mark.parametrize("mode,completed,transitions,unknown", [
    ("timeout", 2, 2, 1), ("step_error", 2, 2, 1), ("spawn_failure", 2, 1, 0),
])
def test_failure_counters_distinguish_complete_transitions_and_unconfirmed_step(
    tmp_path, monkeypatch, mode, completed, transitions, unknown,
):
    base = ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "unused", max_steps_per_episode=4,
        final_target_absent_s=0., target_delay_s=0.,
        critic_image_height=8, critic_image_width=8,
    )
    ppo = _act_ppo_config(base, 4)
    observation = SimpleNamespace(
        top_image=np.zeros((8, 8, 3), np.uint8), gripper_image=np.zeros((8, 8, 3), np.uint8),
        joint_positions=np.zeros(6), joint_velocities=np.zeros(6),
    )

    class Client:
        world_index = 0
        calls = 0

        def gazebo_diagnostics(self):
            return diagnostics(True)

        def step_async(self, command):
            self.calls += 1

        def spawn_random_target_async(self):
            raise GazeboWorkerCommunicationError("spawn pipe failed", gazebo_diagnostics=diagnostics(True))

        def receive(self, **kwargs):
            if self.calls == 3:
                if mode == "timeout":
                    raise GazeboWorkerCommunicationError("timeout", gazebo_diagnostics=diagnostics(True))
                return {"type": "operation_error", "error_type": "SimulationStepCoordinatorError",
                        "error_message": "camera feedback missing", "gazebo_diagnostics": diagnostics(True)}
            return {"type": "step_result", "reward": 1., "terminated": False,
                    "truncated": False, "info": {}, "observation": {}}

        def observation_from_metadata(self, metadata):
            return observation

    buffer = ActPpoRolloutBuffer(ppo)
    reference = buffer.add_actor_observation(ActPpoActorObservation(
        top_image=np.zeros((3, 8, 8), np.uint8),
        gripper_image=np.zeros((3, 8, 8), np.uint8), policy_state=np.zeros(6),
    ))
    session = module._EpisodeSession(
        slot=module._WorldSlot(client=Client(), available=True), buffer=buffer,
        observation=observation, last_info={}, cleanup={}, home_reset={}, attempt_index=7,
        task_mode="pick_place" if mode == "spawn_failure" else "idle_empty", spawn_delay_steps=2,
        chunk_means=torch.zeros((1, 4, 6)), chunk_actor_reference=reference,
        chunk_stage="test", chunk_executable=4,
        sampled_actions=np.zeros((4, 6)), sampled_action_latents=np.zeros((4, 6)),
        sampled_log_probabilities=np.zeros(4),
        sampled_action_log_std=np.full(6, ppo.initial_log_std),
    )
    monkeypatch.setattr(module, "_critic_context", lambda *args: {})
    monkeypatch.setattr(module, "_critic_value", lambda *args: torch.tensor([0.]))
    monkeypatch.setattr(module, "_exploration_stage", lambda *args: ("test", True, 0.))
    monkeypatch.setattr(module, "_accumulate_action_diagnostics", lambda *args: None)
    monkeypatch.setattr(module, "_map_policy_positions_for_episode", lambda *args: SimpleNamespace(
        action=SimpleNamespace(controller_command=lambda: np.zeros(6))))
    reported = []
    successful, failures = module._run_sessions(
        [session], adapter=object(), policy=object(), critic=object(), trainer=object(),
        device=torch.device("cpu"), loaded_config=SimpleNamespace(n_action_steps=4),
        base_config=base, ppo_config=ppo,
        on_failure=reported.append,
    )
    assert successful == [] and len(failures) == 1
    assert reported == failures
    failure = failures[0]
    assert failure.completed_steps == completed
    assert failure.discarded_transition_count == transitions
    assert failure.unconfirmed_step_count == unknown
    stats = DiscardedStepStats(tmp_path, world_indices=[0])
    record = module._record_failure(failure, output_directory=tmp_path,
                                   maximum_consecutive_failures=3, discard_stats=stats)
    saved = json.loads((tmp_path / "discarded_steps.json").read_text())
    assert saved["discarded_environment_steps"] == completed
    assert saved["gazebo_crash_discarded_environment_steps"] == completed
    assert saved["gazebo_crash_discarded_transition_count"] == transitions
    assert saved["unconfirmed_step_count"] == unknown
    assert record["failure_category"] == "gazebo_crash"
    failure_file = json.loads((tmp_path / record["failure_record_path"]).read_text())
    assert failure_file["gazebo_crash"]["confirmed"] is True


def test_captured_pre_cleanup_evidence_is_not_replaced_by_later_shutdown_crash(tmp_path):
    client = SimpleNamespace(world_index=0, gazebo_diagnostics=lambda: diagnostics(True))
    evidence = module._gazebo_failure_diagnostics(client, {"gazebo_diagnostics": diagnostics(False)})
    failure = module._EpisodeFailure(
        slot=module._WorldSlot(client=client), attempt_index=1, completed_steps=12,
        discarded_transition_count=12, category="camera", error_type="Timeout",
        message="late image", unconfirmed_step_count=1, gazebo_diagnostics=evidence,
    )
    stats = DiscardedStepStats(tmp_path)
    record = module._record_failure(failure, output_directory=tmp_path,
                                   maximum_consecutive_failures=3, discard_stats=stats)
    assert record["failure_category"] == "camera"
    assert stats.snapshot()["gazebo_crash_discarded_environment_steps"] == 0
    assert stats.snapshot()["discarded_environment_steps"] == 12


def test_world_crashes_before_prepare_are_recorded_with_zero_policy_steps(tmp_path, monkeypatch):
    base = ActPpoGazeboConfig(act_checkpoint=tmp_path / "unused", final_target_absent_s=0.)
    class Client:
        world_index = 2
        def prepare_episode_async(self, *args, **kwargs):
            raise GazeboWorkerCommunicationError("disconnected", gazebo_diagnostics=diagnostics(True))
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")
    sessions, failures = module._prepare_sessions(
        [module._WorldSlot(client=Client())], ppo_config=_act_ppo_config(base, 4),
        base_config=base, attempt_counter=[0],
    )
    assert sessions == [] and len(failures) == 1
    assert failures[0].completed_steps == failures[0].unconfirmed_step_count == 0
    assert failures[0].gazebo_diagnostics["confirmed"] is True


@pytest.mark.parametrize("interrupt_other_world", [False, True])
def test_wave_persists_failure_before_other_world_finishes_and_deduplicates(
    tmp_path, monkeypatch, interrupt_other_world,
):
    base = ActPpoGazeboConfig(act_checkpoint=tmp_path / "unused", final_target_absent_s=0.)
    stats = DiscardedStepStats(tmp_path)
    slot = module._WorldSlot(client=SimpleNamespace(world_index=0), available=True)
    failure = module._EpisodeFailure(
        slot=slot, attempt_index=3, completed_steps=20, discarded_transition_count=20,
        category="worker_communication", error_type="Timeout", message="dead world",
        unconfirmed_step_count=1, gazebo_diagnostics=diagnostics(True),
    )
    monkeypatch.setattr(module, "_prepare_sessions", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(module, "_stop_failed_worlds", lambda *args: None)
    def collect(*args, on_failure, **kwargs):
        on_failure(failure)
        # Another world has not finished; the first world's failure must
        # already survive an interruption of the learner here.
        assert json.loads((tmp_path / "discarded_steps.json").read_text())["discarded_environment_steps"] == 20
        assert (tmp_path / "episode_failures/attempt_000003_world_00.json").is_file()
        if interrupt_other_world:
            raise KeyboardInterrupt()
        return [], [failure]
    monkeypatch.setattr(module, "_run_sessions", collect)
    kwargs = dict(
        adapter=object(), policy=object(), critic=object(), trainer=object(),
        device=torch.device("cpu"), loaded_config=object(), base_config=base,
        multiworld_config=module.MultiWorldConfig(world_restart_max_attempts=0),
        ppo_config=_act_ppo_config(base, 4), output_directory=tmp_path,
        attempt_counter=[0], discard_stats=stats,
    )
    if interrupt_other_world:
        with pytest.raises(KeyboardInterrupt):
            module._run_wave([slot], **kwargs)
    else:
        _, records, discarded = module._run_wave([slot], **kwargs)
        assert len(records) == 1 and discarded == 20
    assert stats.snapshot()["failure_episode_count"] == 1
    assert slot.failed_episode_count == 1


@pytest.mark.parametrize("update_in_progress", [False, True])
def test_aborted_learner_accounts_pending_rollout_without_guessing_crash_cause(
    tmp_path, update_in_progress,
):
    from so101_capstone.training.act_ppo_tensorboard import DirectActPpoTensorBoard
    buffer = ActPpoRolloutBuffer()
    buffer.rewards.extend([1.] * 8)  # This test isolates pending-length accounting.
    stats = DiscardedStepStats(tmp_path)
    module._record_aborted_rollout(
        buffer, discard_stats=stats, output_directory=tmp_path,
        tensorboard=DirectActPpoTensorBoard.disabled(), update_in_progress=update_in_progress,
    )
    saved = json.loads((tmp_path / "discarded_steps.json").read_text())
    assert saved["aborted_rollout_discarded_transition_count"] == (0 if update_in_progress else 8)
    assert saved["interrupted_update_transition_count"] == (8 if update_in_progress else 0)
    assert saved["discarded_transition_count"] == (0 if update_in_progress else 8)
    assert saved["gazebo_crash_discarded_transition_count"] == 0
