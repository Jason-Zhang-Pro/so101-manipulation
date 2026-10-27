"""Exercise stage preparation and PPO truncation at the real collector boundary."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_capstone.policies.act_policy_adapter import (
    ACT_FIXED_IMAGE_KEY, ACT_HANDEYE_IMAGE_KEY, ACT_STATE_KEY,
)
from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM
from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo import (
    ActPpoExploration, ActPpoRolloutBuffer, configure_bounded_act_policy, policy_action_bounds,
)
from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerCommunicationError


GEOMETRY = dict(goal_center_world_m=[.2, .2, .75], goal_half_extents_xy_m=[.05, .05],
                goal_yaw_rad=0., target_half_extents_m=[.015] * 3, source_support_z_m=.75)


def config(**kwargs):
    return single.ActPpoGazeboConfig(
        act_checkpoint=Path("/tmp/stage-act"), reward_profile="pick_place_v2",
        final_target_absent_s=0., target_delay_s=0., critic_image_height=8,
        critic_image_width=8, **kwargs,
    )


@pytest.mark.parametrize("stage,mode", [("S0", "idle_empty"), ("S1", "pick_place")])
def test_preparation_returns_post_initialization_frame_zero_without_policy_samples(monkeypatch, stage, mode):
    monkeypatch.delenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", raising=False)
    events = []

    class Client:
        world_index = 0

        def prepare_episode_async(self, attempt, **kwargs):
            events.append("prepare")
            assert kwargs["task_mode"] == mode
            self.response = {"type": "episode_prepared", "observation": "empty_frame",
                             "info": GEOMETRY, "cleanup": {}, "home_reset": {}}

        def spawn_target_async(self, position, quaternion):
            events.append("spawn")
            self.response = {"type": "target_spawned", "success": True, "message": "{}"}

        def start_training_stage_async(self, training_stage, stage_config):
            events.append("stage_prepare")
            assert training_stage == stage
            assert stage_config["s0_budget_s"] == 5.
            assert ("spawn" in events) == (stage == "S1")
            self.response = {
                "type": "stage_started", "observation": "ready_frame",
                "info": {"training_stage": stage, "stage_elapsed_s": 0.,
                         "stage_initialization": {"valid": True, "discarded_control_steps": 40},
                         "start_certificate": {"start_valid": True, "training_stage": stage}},
            }

        def receive(self, **kwargs):
            return self.response

        def observation_from_metadata(self, metadata):
            return metadata

    base = config(training_stages=stage.lower())
    sessions, failures = multi._prepare_sessions(
        [multi._WorldSlot(client=Client())], base_config=base,
        ppo_config=single._act_ppo_config(base, 2), attempt_counter=[0],
    )
    assert failures == [] and len(sessions) == 1
    session = sessions[0]
    assert session.observation == "ready_frame"
    assert session.completed_steps == 0 and len(session.buffer) == 0
    assert len(session.buffer.actor_observations) == 0
    assert session.last_info["stage_elapsed_s"] == 0.
    assert session.stage_initialization["stage_initialization"]["discarded_control_steps"] == 40
    assert session.max_steps == single._episode_step_limit(base, mode)
    assert events == (["prepare", "stage_prepare"] if stage == "S0" else ["prepare", "spawn", "stage_prepare"])


@pytest.mark.parametrize("failure", ["send", "receive", "operation_error", "wrong_stage"])
def test_stage_preparation_failure_has_zero_policy_steps_and_does_not_block_other_response(failure):
    events = []
    base = config()
    ppo = single._act_ppo_config(base, 2)

    class Client:
        def __init__(self, index):
            self.world_index = index

        def start_training_stage_async(self, stage, stage_config):
            events.append(("send", self.world_index))
            if self.world_index == 0 and failure == "send":
                raise GazeboWorkerCommunicationError("send failed")

        def receive(self, **kwargs):
            events.append(("receive", self.world_index))
            if self.world_index == 0 and failure == "receive":
                raise GazeboWorkerCommunicationError("receive failed")
            if self.world_index == 0 and failure == "operation_error":
                return {"type": "operation_error", "error_message": "target did not settle"}
            return {"type": "stage_started", "observation": "ready",
                    "info": {"training_stage": "S1" if self.world_index == 0 else "S0"}}

        def observation_from_metadata(self, metadata):
            return metadata

    sessions = [multi._EpisodeSession(
        slot=multi._WorldSlot(client=Client(index)), buffer=ActPpoRolloutBuffer(ppo),
        observation="old", last_info={}, cleanup={}, home_reset={},
        attempt_index=index + 1, task_mode="idle_empty",
    ) for index in range(2)]
    reported = []
    ready, failed = multi._start_training_stage_sessions(sessions, stage_config={}, on_failure=reported.append)
    assert ready == [sessions[1]] and len(failed) == 1 and reported == failed
    assert failed[0].category == "stage_initialization"
    assert failed[0].completed_steps == failed[0].discarded_transition_count == failed[0].unconfirmed_step_count == 0
    assert ("receive", 1) in events
    assert events[:2] == [("send", 0), ("send", 1)]
    assert all(len(session.buffer) == 0 for session in sessions)


def observation(index):
    return SimpleNamespace(top_image=np.full((8, 8, 3), index, np.uint8),
                           gripper_image=np.zeros((8, 8, 3), np.uint8),
                           joint_positions=np.zeros(6), joint_velocities=np.zeros(6))


@pytest.mark.parametrize("terminal,stage_result", [(False, "SUCCESS"), (False, "TIMEOUT"), (True, "FAILURE")])
def test_stage_cuts_bootstrap_from_final_frame_and_gae_never_crosses_segments(monkeypatch, terminal, stage_result):
    base = config(training_stages="s1", max_steps_per_episode=2)
    ppo = single._act_ppo_config(base, 2)

    def info(index):
        return {"training_stage": "S1", "stage_result": stage_result if index == 4 else "RUNNING",
                "stage_success": index == 4 and stage_result == "SUCCESS", "stage_elapsed_s": index / 30.,
                "stage_budget_s": 10., "task_phase": "approach",
                "task_outcome": "task_failure" if index == 4 and terminal else "running",
                "critic_task_context": [float(index)] * TASK_CONTEXT_DIM,
                "start_certificate": {"training_stage": "S1", "start_valid": True}}

    class Client:
        world_index = 0
        steps = 0

        def step_async(self, command):
            self.steps += 1

        def receive(self, **kwargs):
            return {"type": "step_result", "observation": self.steps, "reward": 1.,
                    "terminated": terminal and self.steps == 4,
                    "truncated": not terminal and self.steps == 4, "info": info(self.steps)}

        def observation_from_metadata(self, index):
            return observation(index)

    session = multi._EpisodeSession(
        slot=multi._WorldSlot(client=Client()), buffer=ActPpoRolloutBuffer(ppo),
        observation=observation(0), last_info=info(0), cleanup={}, home_reset={},
        attempt_index=1, spawn={"requested": True},
    )
    forward_frames = []

    def predict(policy, batch):
        forward_frames.append(int(round(float(batch[ACT_FIXED_IMAGE_KEY][0, 0, 0, 0]) * 255)))
        lower, upper = policy_action_bounds(device="cpu")
        return ((lower + upper) / 2).expand(1, 2, 6).clone()

    monkeypatch.setattr(multi, "predict_action_chunk_with_grad", predict)
    monkeypatch.setattr(multi, "_map_policy_positions_for_episode", lambda *args: SimpleNamespace(
        action=SimpleNamespace(controller_command=lambda: np.zeros(6))))

    def batch(obs):
        return {ACT_FIXED_IMAGE_KEY: torch.tensor(obs.top_image.transpose(2, 0, 1)[None]).float() / 255,
                ACT_HANDEYE_IMAGE_KEY: torch.zeros(1, 3, 8, 8), ACT_STATE_KEY: torch.zeros(1, 6)}

    class Critic:
        def __call__(self, *args, task_context, policy_context):
            return task_context[:, 0]

    done, failed = multi._run_sessions(
        [session], adapter=SimpleNamespace(build_batch=batch), policy=object(), critic=Critic(),
        trainer=SimpleNamespace(exploration=ActPpoExploration(ppo)), device=torch.device("cpu"),
        loaded_config=SimpleNamespace(n_action_steps=2), base_config=base, ppo_config=ppo,
    )
    assert failed == [] and done == [session]
    assert session.completed_steps == 4  # Global two-step cap cannot end a ten-second S1.
    assert session.buffer.next_values[-1] == (0. if terminal else 4.)
    assert session.buffer.episode_end == [False, False, False, True]
    assert session.buffer.terminated == [False, False, False, terminal]
    assert forward_frames == ([0, 2] if terminal else [0, 2, 4])
    assert len(session.buffer.actor_observations) == 2
    original_advantages, _ = session.buffer._advantages_and_returns()
    combined = ActPpoRolloutBuffer(ppo)
    combined.extend(session.buffer)
    combined.extend(session.buffer)
    combined.rewards[4:] = [10000.] * 4
    advantages, _ = combined._advantages_and_returns()
    np.testing.assert_allclose(advantages[:4], original_advantages)
    assert original_advantages[-1] == pytest.approx(1. + ppo.gamma * (0. if terminal else 4.) - 3.)
    record = multi._episode_record(session, episode_index=0, base_config=base, rollout_size_after_episode=4)
    assert record["training_stage"] == "S1" and record["stage_result"] == stage_result
    assert record["stage_success"] == (stage_result == "SUCCESS")
    assert record["task_success"] is False
    assert record["start_certificate"]["start_valid"] is True
    summary = multi._training_stage_summary([record])
    assert summary["S1"]["stage_success_count"] == int(stage_result == "SUCCESS")
    assert summary["S1"]["stage_timeout_count"] == int(stage_result == "TIMEOUT")
    assert summary["S1"]["task_failure_count"] == int(terminal)


def test_stage_resume_restores_next_scene_seed_and_preserves_other_collection_state():
    base = config(training_stages="s0_s1")
    trainer = SimpleNamespace(collection_state={"unrelated_state": {"version": 1}})
    assert multi._initial_attempt_count(base, None, None) == 0
    multi._sync_stage_attempt_count(trainer, base, 47)
    payload = {"extra": {"collection_state": trainer.collection_state}}
    restored = multi._initial_attempt_count(base, None, payload)
    assert restored == 47
    assert trainer.collection_state["unrelated_state"] == {"version": 1}
    assert single._episode_scene(base, restored + 1, 2) == single._episode_scene(base, 48, 2)
    assert single._episode_scene(base, restored + 1, 2)[2] != single._episode_scene(base, 1, 2)[2]


@pytest.mark.parametrize("saved", [{}, {"stage_attempt_count": -1}, {"stage_attempt_count": True},
                                  {"stage_attempt_count": 2.5}, {"stage_attempt_count": "3"}])
def test_stage_resume_rejects_missing_or_invalid_attempt_counter(saved):
    with pytest.raises(ValueError, match="stage_attempt_count"):
        multi._initial_attempt_count(config(training_stages="s1"), None,
                                     {"extra": {"collection_state": saved}})


def test_full_task_curriculum_resume_and_state_are_unchanged():
    base = config()
    curriculum = SimpleNamespace(attempt_count=23)
    assert multi._initial_attempt_count(base, curriculum, {"extra": {}}) == 23
    assert multi._initial_attempt_count(base, None, None) == 0
    trainer = SimpleNamespace(collection_state={"reach_curriculum": {"attempt_count": 23}})
    original = trainer.collection_state
    multi._sync_stage_attempt_count(trainer, base, 99)
    assert trainer.collection_state is original
    assert original == {"reach_curriculum": {"attempt_count": 23}}


def test_real_main_loop_restores_stage_counter_and_updates_it_on_failed_wave(tmp_path, monkeypatch):
    """Exercise the actual try/finally boundary without launching ROS workers."""
    policy = torch.nn.Linear(1, 1)
    policy.reset = lambda: None
    act_config = SimpleNamespace(
        temporal_ensemble_coeff=None, n_action_steps=2, chunk_size=2,
        optimizer_lr=3e-8, optimizer_lr_backbone=3e-9,
        optimizer_weight_decay=0., use_vae=True, vision_backbone="fake",
    )
    loaded = SimpleNamespace(policy=policy, config=act_config, device=torch.device("cpu"))
    policy.config = loaded.config
    trainers = []

    class Trainer:
        actor_learning_rate = 3e-8
        actor_backbone_learning_rate = 3e-9

        def __init__(self, policy, *args, **kwargs):
            configure_bounded_act_policy(policy)
            self.update_index = 4
            trainers.append(self)

        def load_checkpoint(self, *args, **kwargs):
            return {"extra": {"collection_state": {"stage_attempt_count": 17}}}

    class Client:
        def __init__(self, config):
            self.config, self.world_index = config, config.world_index

        def begin_start(self):
            pass

        def finish_start(self):
            return {"type": "ready"}

        def shutdown(self):
            pass

    def failed_wave(slots, *, attempt_counter, trainer, **kwargs):
        assert attempt_counter == [17]
        assert trainer.collection_state == {"stage_attempt_count": 17}
        attempt_counter[0] += 2  # Both attempted scenes failed preparation.
        raise multi._WorldShutdownError("simulated cleanup failure")

    monkeypatch.setattr(multi, "load_act_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(multi, "ActPolicyAdapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(multi, "ActPpoValueNetwork", lambda config: torch.nn.Identity())
    monkeypatch.setattr(multi, "ActPpoTrainer", Trainer)
    monkeypatch.setattr(multi, "GazeboWorkerClient", Client)
    monkeypatch.setattr(multi, "_run_wave", failed_wave)
    monkeypatch.setattr(multi, "_terminal_print_block", lambda *args, **kwargs: None)
    base = config(training_stages="s0_s1", device="cpu", tensorboard_enabled=False,
                  output_directory=tmp_path / "run", resume_checkpoint=tmp_path / "checkpoint.pt",
                  episode_count=2)
    with pytest.raises(multi._WorldShutdownError, match="simulated cleanup failure"):
        multi.run_act_ppo_multiworld_gazebo(base, multi.MultiWorldConfig(
            world_count=1, preflight_cleanup_enabled=False))
    assert trainers[0].collection_state == {"stage_attempt_count": 19}
