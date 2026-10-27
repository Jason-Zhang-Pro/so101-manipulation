"""Temporal collector contracts with in-process worlds; no simulator is started."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from so101_capstone.policies.act_policy_adapter import (
    ACT_FIXED_IMAGE_KEY, ACT_HANDEYE_IMAGE_KEY, ACT_STATE_KEY,
)
from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM
from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo import (
    ActPpoExploration, ActPpoRolloutBuffer, build_critic_policy_context,
    configure_bounded_act_policy, policy_action_bounds,
)
from so101_capstone.training.act_ppo_episode_records import persist_episode_steps


COEFF = 0.01
CHUNK = 3
STEPS = 5


def observation(frame):
    return SimpleNamespace(top_image=np.full((8, 8, 3), frame, np.uint8),
                           gripper_image=np.zeros((8, 8, 3), np.uint8),
                           joint_positions=np.zeros(6), joint_velocities=np.zeros(6))


def info(step):
    phases = ["WAIT_HOME", "APPROACH", "GRASP_LIFT", "TRANSPORT", "HOME_HOLD", "HOME_HOLD"]
    return {"task_phase": phases[step], "critic_task_context": [float(step)] * TASK_CONTEXT_DIM,
            "task_outcome": "success" if step == STEPS else "running"}


def build_batch(obs):
    return {ACT_FIXED_IMAGE_KEY: torch.from_numpy(obs.top_image.transpose(2, 0, 1)[None]).float() / 255,
            ACT_HANDEYE_IMAGE_KEY: torch.zeros(1, 3, 8, 8), ACT_STATE_KEY: torch.zeros(1, 6)}


def chunk(frame):
    lower, upper = policy_action_bounds(device="cpu")
    return ((lower + upper) / 2 + frame * .1 + torch.arange(CHUNK).view(1, CHUNK, 1) * .7).clone()


def expected_means(start, count):
    oracle = ACTTemporalEnsembler(COEFF, CHUNK)
    return [oracle.update(chunk(start + index)).unsqueeze(1) for index in range(count)]


def context_value(means):
    return float(build_critic_policy_context(means, action_index=0, n_action_steps=1)[2])


class Critic(nn.Module):
    def forward(self, *args, task_context, policy_context):
        # Expose the actual fused action mean used for bootstrap, not only step index.
        return policy_context[:, 2]


def patch_prediction(monkeypatch, module):
    frames, sampled_means = [], []

    def predict(policy, batch):
        frame = int(round(float(batch[ACT_FIXED_IMAGE_KEY][0, 0, 0, 0]) * 255))
        frames.append(frame)
        return chunk(frame)

    original_sample = module.sample_bounded_policy_actions

    def sample(means, *args, **kwargs):
        sampled_means.append(means.clone())
        return original_sample(means, *args, **kwargs)

    monkeypatch.setattr(module, "predict_action_chunk_with_grad", predict)
    monkeypatch.setattr(module, "sample_bounded_policy_actions", sample)
    monkeypatch.setattr(module, "_map_policy_positions_for_episode", lambda *args: SimpleNamespace(
        action=SimpleNamespace(controller_command=lambda: np.zeros(6))))
    return frames, sampled_means


def config(**kwargs):
    return single.ActPpoGazeboConfig(act_checkpoint=Path("unused"), reward_profile="pick_place_v2",
        final_target_absent_s=0, max_steps_per_episode=STEPS, target_delay_s=0,
        critic_image_height=8, critic_image_width=8, **kwargs)


@pytest.mark.parametrize("terminal", [False, True])
def test_multiworld_temporal_histories_are_isolated_and_bootstrap_uses_fused_mean(monkeypatch, tmp_path, terminal):
    base = config()
    ppo = single._act_ppo_config(base, 1)
    frames, means = patch_prediction(monkeypatch, multi)

    class Client:
        def __init__(self, world):
            self.world_index, self.steps = world, 0

        def step_async(self, command):
            self.steps += 1

        def receive(self, **kwargs):
            return {"type": "step_result", "observation": self.world_index * 20 + self.steps,
                    "reward": 1., "terminated": terminal and self.steps == STEPS,
                    "truncated": not terminal and self.steps == STEPS, "info": info(self.steps)}

        def observation_from_metadata(self, frame):
            return observation(frame)

    def session(world, attempt):
        return multi._EpisodeSession(slot=multi._WorldSlot(client=Client(world)),
            buffer=ActPpoRolloutBuffer(ppo), observation=observation(world * 20), last_info=info(0),
            cleanup={}, home_reset={}, attempt_index=attempt, spawn={"requested": True})

    sessions = [session(0, 1), session(1, 2)]
    kwargs = dict(adapter=SimpleNamespace(build_batch=build_batch), policy=object(), critic=Critic(),
        trainer=SimpleNamespace(exploration=ActPpoExploration(ppo)),
        device=torch.device("cpu"), loaded_config=SimpleNamespace(chunk_size=CHUNK,
            n_action_steps=1, temporal_ensemble_coeff=COEFF), base_config=base, ppo_config=ppo)
    done, failed = multi._run_sessions(sessions, **kwargs)
    assert failed == [] and len(done) == 2
    for world, session_result in enumerate(sessions):
        expected = expected_means(world * 20, STEPS + (not terminal))
        for actual, wanted in zip(means[world::2], expected[:STEPS], strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        buffer = session_result.buffer
        np.testing.assert_allclose(buffer.policy_action_means,
                                   [value[0, 0].numpy() for value in expected[:STEPS]], rtol=0, atol=0)
        np.testing.assert_allclose(buffer.policy_action_log_stds,
                                   [[value] * 6 for value in [-5., -4., -4., -4., -5.]])
        assert buffer.actor_refs == list(range(STEPS))
        assert buffer.action_indices == [0] * STEPS
        assert len(buffer.actor_observations) == STEPS
        assert buffer.temporal_actor_references == [(0,), (0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4)]
        record_path = tmp_path / f"world_{world}_steps.jsonl"
        persist_episode_steps(record_path, buffer, STEPS)
        records = [json.loads(line) for line in record_path.read_text().splitlines()]
        assert [item["temporal_actor_references"] for item in records] == [
            list(refs) for refs in buffer.temporal_actor_references]
        assert buffer.values == pytest.approx([context_value(x) for x in expected[:STEPS]])
        next_values = [context_value(x) for x in expected[1:STEPS]]
        next_values.append(0. if terminal else context_value(expected[-1]))
        assert buffer.next_values == pytest.approx(next_values)
    assert len(frames) == 2 * (STEPS + (not terminal))
    assert sessions[0].temporal_history is not sessions[1].temporal_history
    # A replacement episode starts with no contributions from the old world session.
    replacement = session(0, 3)
    old_count = len(means)
    multi._run_sessions([replacement], **kwargs)
    torch.testing.assert_close(means[old_count], expected_means(0, 1)[0], rtol=0, atol=0)
    assert replacement.buffer.temporal_actor_references[0] == (0,)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("fail_first", [False, True])
def test_single_temporal_entry_resets_history_between_episodes_and_records_contract(monkeypatch, tmp_path, terminal, fail_first):
    frames, means = patch_prediction(monkeypatch, single)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def reset(self):
            pass

    class Runtime:
        attempt = 0

        def configure_task(self, mode, **kwargs):
            assert kwargs["n_action_steps"] == 1

        def delete_all_targets(self):
            return SimpleNamespace(success=True, message="deleted")

        spawn_random_target = delete_all_targets

        def prepare_episode(self):
            self.index = 0
            self.attempt += 1
            self.start = 40 if fail_first and self.attempt == 1 else 0
            return SimpleNamespace(observation=observation(self.start), info=info(0))

        def step(self, command):
            self.index += 1
            if fail_first and self.attempt == 1 and self.index == 2:
                raise single.RecoverableEpisodeError("feedback", "simulated feedback failure")
            return SimpleNamespace(observation=observation(self.start + self.index), info=info(self.index), reward=1.,
                terminated=terminal and self.index == STEPS, truncated=not terminal and self.index == STEPS)

    @contextmanager
    def runtime_factory(**kwargs):
        yield Runtime()

    captured = {}

    class Trainer:
        actor_learning_rate = actor_backbone_learning_rate = .001
        update_index = total_environment_steps = 0
        def __init__(self, policy, critic, **kwargs):
            configure_bounded_act_policy(policy)
            self.policy = policy
            self.exploration = ActPpoExploration(kwargs["config"])
            captured["signature"] = kwargs["execution_signature"]

        def update(self, buffer):
            self.update_index += 1
            self.total_environment_steps += len(buffer)
            with torch.no_grad():
                self.policy.weight.add_(1.)
            captured["histories"] = list(buffer.temporal_actor_references)
            captured["next_values"] = list(buffer.next_values)
            return SimpleNamespace(to_dict=lambda: {})

    loaded = SimpleNamespace(policy=Policy(), device=torch.device("cpu"), config=SimpleNamespace(
        chunk_size=CHUNK, n_action_steps=1, temporal_ensemble_coeff=COEFF,
        optimizer_lr=.001, optimizer_lr_backbone=.001, optimizer_weight_decay=0.,
        use_vae=True, vision_backbone="fake"))
    loaded.policy.config = loaded.config
    monkeypatch.setattr(single, "ActPpoTrainer", Trainer)
    monkeypatch.setattr(single, "ActPpoValueNetwork", lambda cfg: Critic())
    monkeypatch.setattr(single, "_save_update", lambda *args, **kwargs: {})
    monkeypatch.setattr(single, "_export_final", lambda *args, **kwargs: (tmp_path / "act", tmp_path / "state.pt"))
    result = single.run_act_ppo_gazebo(config(output_directory=tmp_path / "out", device="cpu",
        episode_count=2, rollout_steps=100), checkpoint_loader=lambda *args, **kwargs: loaded,
        runtime_factory=runtime_factory, adapter_factory=lambda *args, **kwargs: SimpleNamespace(build_batch=build_batch))
    expected = expected_means(0, STEPS + (not terminal))
    expected_samples = (expected_means(40, 2) if fail_first else []) + expected[:STEPS] * 2
    for actual, wanted in zip(means, expected_samples, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert captured["histories"] == [(0,), (0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4),
                                      (5,), (5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9)]
    assert captured["signature"]["chunk_schedule"] == "temporal_ensemble_every_step"
    assert captured["signature"]["temporal_ensemble_coeff"] == COEFF
    assert result["temporal_ensemble_coeff"] == COEFF
    assert result["n_action_steps"] == 1
    assert len(frames) == 2 * (STEPS + (not terminal)) + (2 if fail_first else 0)
    assert len(result["failed_episodes"]) == int(fail_first)
    for last in (STEPS - 1, 2 * STEPS - 1):
        assert captured["next_values"][last] == pytest.approx(0. if terminal else context_value(expected[-1]))
