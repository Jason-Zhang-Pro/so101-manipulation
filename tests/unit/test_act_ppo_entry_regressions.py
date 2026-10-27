"""Entry-point regressions using in-process fakes, never live simulators."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from so101_capstone.policies.act_policy_adapter import (
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_STATE_KEY,
)
from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training import act_ppo_multiworld_worker as worker
from so101_capstone.training.act_ppo import (
    ActPpoExploration, ActPpoRolloutBuffer, configure_bounded_act_policy, policy_action_bounds,
)
from so101_capstone.training.residual_ppo import model_state_sha256


class StopBeforeSimulation(RuntimeError):
    pass


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.]))

    def reset(self):
        pass


def _fake_loaded(policy):
    policy.config = SimpleNamespace(
        temporal_ensemble_coeff=None,
        n_action_steps=30,
        chunk_size=30,
        optimizer_lr=1e-3,
        optimizer_lr_backbone=1e-3,
        optimizer_weight_decay=0.,
        use_vae=True,
        vision_backbone="fake",
    )
    return SimpleNamespace(policy=policy, device=torch.device("cpu"), config=policy.config)


def _patch_entry_training_objects(monkeypatch, module, policy, events):
    class FakeTrainer:
        actor_learning_rate = 1e-3
        actor_backbone_learning_rate = 1e-3

        def __init__(self, policy, critic, *, config):
            configure_bounded_act_policy(policy)
            self.policy = policy

        def load_checkpoint(self, path, *, map_location):
            assert Path(path).name == "resume.pt"
            assert map_location == torch.device("cpu")
            with torch.no_grad():
                self.policy.weight.fill_(7.)
            events.append("resumed")

    def fingerprint(policy):
        events.append(("fingerprint", model_state_sha256(policy)))
        return model_state_sha256(policy)

    monkeypatch.setattr(module, "ActPpoValueNetwork", lambda config: nn.Identity())
    monkeypatch.setattr(module, "ActPpoTrainer", FakeTrainer)
    monkeypatch.setattr(module, "model_state_sha256", fingerprint)
    monkeypatch.setattr(module, "_set_seed", lambda seed: None)
    loaded = _fake_loaded(policy)
    return loaded


def test_single_entry_resumes_before_initial_fingerprint_and_passes_1200_horizon(monkeypatch, tmp_path):
    policy = FakePolicy()
    original = model_state_sha256(policy)
    events = []
    loaded = _patch_entry_training_objects(monkeypatch, single, policy, events)
    received = []

    def runtime_factory(**kwargs):
        received.append(kwargs)
        raise StopBeforeSimulation()

    config = single.ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "act",
        resume_checkpoint=tmp_path / "resume.pt",
        output_directory=tmp_path / "single-output",
        device="cpu",
        max_steps_per_episode=1200,
    )
    with pytest.raises(StopBeforeSimulation):
        single.run_act_ppo_gazebo(
            config,
            checkpoint_loader=lambda *args, **kwargs: loaded,
            runtime_factory=runtime_factory,
            adapter_factory=lambda *args, **kwargs: object(),
        )
    assert events == ["resumed", ("fingerprint", model_state_sha256(policy))]
    assert events[1][1] != original
    assert received[0]["environment_config"].max_episode_steps == 1200


def test_multi_entry_resumes_before_initial_fingerprint_and_sets_each_worker_horizon(monkeypatch, tmp_path):
    policy = FakePolicy()
    original = model_state_sha256(policy)
    events = []
    loaded = _patch_entry_training_objects(monkeypatch, multi, policy, events)
    monkeypatch.setattr(multi, "load_act_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(multi, "ActPolicyAdapter", lambda *args, **kwargs: object())
    clients = []

    class FakeClient:
        def __init__(self, config):
            self.config = config
            self.world_index = config.world_index
            self.closed = False
            clients.append(self)

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(multi, "GazeboWorkerClient", FakeClient)

    def stop_before_workers(slots, **kwargs):
        assert len(slots) == 2
        raise StopBeforeSimulation()

    monkeypatch.setattr(multi, "_start_clients_concurrently", stop_before_workers)
    base = single.ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "act",
        resume_checkpoint=tmp_path / "resume.pt",
        output_directory=tmp_path / "multi-output",
        device="cpu",
        max_steps_per_episode=1200,
        final_target_absent_s=0,
    )
    worlds = multi.MultiWorldConfig(world_count=2, preflight_cleanup_enabled=False)
    with pytest.raises(StopBeforeSimulation):
        multi.run_act_ppo_multiworld_gazebo(base, worlds)
    assert events == ["resumed", ("fingerprint", model_state_sha256(policy))]
    assert events[1][1] != original
    assert [client.config.max_episode_steps for client in clients] == [1200, 1200]
    assert all(client.closed for client in clients)
    assert clients[0].config.ros_domain_id != clients[1].config.ros_domain_id


def test_worker_payload_and_launch_forward_1200_horizon_to_runtime(monkeypatch, tmp_path):
    from so101_capstone.environments import simulation_runtime

    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")
    original = worker.GazeboWorkerConfig(
        world_index=0,
        ros_domain_id=71,
        gz_partition="unit-test-world",
        gazebo_gui=False,
        project_root=tmp_path,
        output_directory=tmp_path,
        startup_feedback_timeout_s=2.,
        max_episode_steps=1200,
    )
    config = worker.GazeboWorkerConfig.from_payload(original.payload())
    host = worker._WorkerRuntimeHost(
        config,
        top_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
        gripper_buffer=bytearray(worker._POLICY_IMAGE_BYTES),
    )
    monkeypatch.setattr(host, "stop_world", lambda: None)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(poll=lambda: None))
    received = []
    ready = []

    def runtime_factory(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(wait_until_ready=lambda **kw: ready.append(kw))

    monkeypatch.setattr(simulation_runtime, "create_ros_simulation_runtime", runtime_factory)
    try:
        host._launch_once(reason="unit test", attempt=1)
    finally:
        if host.log_stream is not None:
            host.log_stream.close()
    assert received[0]["environment_config"].max_episode_steps == 1200
    assert received[0]["runtime_config"].resume_world_on_close is False
    assert ready == [{"timeout_s": 2.}]


_STAGES = [
    ("state1", {"reward_state": 1}),
    ("state2", {"reward_state": 2}),
    ("state3", {"reward_state": 3}),
    ("state4_not_home", {"reward_state": 4, "robot_at_home": False}),
    ("state4_home", {"reward_state": 4, "robot_at_home": True}),
]


@pytest.mark.parametrize("stage,info", _STAGES)
@pytest.mark.parametrize("explore", [True, False])
def test_refresh_chunk_uses_full_gaussian_and_excludes_deterministic_actor_samples(monkeypatch, stage, info, explore):
    base = single.ActPpoGazeboConfig(
        act_checkpoint=Path("unused-act"),
        max_steps_per_episode=1200,
        **{f"{stage}_exploration_enabled": explore, f"{stage}_actor_update_enabled": True},
    )
    ppo_config = single._act_ppo_config(base)
    actor_batch = {
        ACT_FIXED_IMAGE_KEY: torch.zeros(1, 3, 8, 8),
        ACT_HANDEYE_IMAGE_KEY: torch.zeros(1, 3, 8, 8),
        ACT_STATE_KEY: torch.zeros(1, 6),
    }
    session = multi._EpisodeSession(
        slot=multi._WorldSlot(client=SimpleNamespace(world_index=0)),
        buffer=ActPpoRolloutBuffer(ppo_config),
        observation=object(),
        last_info=info,
        cleanup={},
        home_reset={},
        attempt_index=1,
        completed_steps=16,
    )
    lower, upper = policy_action_bounds(device="cpu")
    means = ((lower + upper) / 2).expand(1, 30, 6).clone()
    monkeypatch.setattr(multi, "predict_action_chunk_with_grad", lambda *args, **kwargs: means)
    sampler = multi.sample_bounded_policy_actions
    calls = []

    def record_sampler(means, log_std, **kwargs):
        calls.append(kwargs)
        return sampler(means, log_std, **kwargs)

    monkeypatch.setattr(multi, "sample_bounded_policy_actions", record_sampler)
    multi._refresh_chunk(
        session,
        adapter=SimpleNamespace(build_batch=lambda observation: actor_batch),
        policy=object(),
        trainer=SimpleNamespace(exploration=ActPpoExploration(ppo_config)),
        device=torch.device("cpu"),
        loaded_config=SimpleNamespace(n_action_steps=30),
        base_config=base,
        ppo_config=ppo_config,
    )
    assert calls == [{"deterministic": not explore, "return_latent": True}]
    assert session.chunk_exploration_enabled is explore
    assert session.chunk_actor_update_enabled is explore
    assert single._actor_update_enabled(info, base) is explore
    assert session.chunk_executable == 30  # No State3 15-step split.
    assert session.sampled_actions.shape == (30, 6)
    assert session.sampled_action_latents.shape == (30, 6)
    assert session.sampled_action_log_std.shape == (6,)
    if not explore:
        torch.testing.assert_close(torch.tensor(session.sampled_actions).float(), means[0])


@pytest.mark.parametrize("terminal", [True, False])
def test_pick_place_collector_keeps_chunk_across_phases_and_bootstraps_correct_cache(monkeypatch, terminal):
    import numpy as np
    from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM

    base = single.ActPpoGazeboConfig(
        act_checkpoint=Path("unused"), reward_profile="pick_place_v2", final_target_absent_s=0,
        max_steps_per_episode=4, target_delay_s=0, manipulation_exploration_log_std=-3.5,
        critic_image_height=8, critic_image_width=8,
    )
    ppo = single._act_ppo_config(base, n_action_steps=2)
    phases = ["WAIT_HOME", "APPROACH", "GRASP_LIFT", "TRANSPORT", "HOME_HOLD"]

    def info(index):
        return {"task_phase": phases[index], "critic_task_context": [float(index)] * TASK_CONTEXT_DIM,
                "task_outcome": "success" if index == 4 and terminal else "running"}

    def observation(index):
        return SimpleNamespace(top_image=np.full((8, 8, 3), index, np.uint8),
                               gripper_image=np.zeros((8, 8, 3), np.uint8),
                               joint_positions=np.zeros(6), joint_velocities=np.zeros(6))

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
        observation=observation(0), last_info=info(0), cleanup={}, home_reset={}, attempt_index=1,
        spawn={"requested": True},
    )
    forward_frames = []

    def predict(policy, batch):
        forward_frames.append(int(round(float(batch[ACT_FIXED_IMAGE_KEY][0, 0, 0, 0]) * 255)))
        lower, upper = policy_action_bounds(device="cpu")
        return ((lower + upper) / 2 + len(forward_frames)).expand(1, 2, 6).clone()

    monkeypatch.setattr(multi, "predict_action_chunk_with_grad", predict)
    monkeypatch.setattr(multi, "_map_policy_positions_for_episode", lambda *args: SimpleNamespace(
        action=SimpleNamespace(controller_command=lambda: np.zeros(6))))
    sample_shapes = []
    original_sampler = multi.sample_bounded_policy_actions

    def sample(means, std, **kwargs):
        sample_shapes.append(tuple(means.shape))
        return original_sampler(means, std, **kwargs)

    monkeypatch.setattr(multi, "sample_bounded_policy_actions", sample)

    def batch(obs):
        return {ACT_FIXED_IMAGE_KEY: torch.tensor(obs.top_image.transpose(2, 0, 1)[None]).float() / 255,
                ACT_HANDEYE_IMAGE_KEY: torch.zeros(1, 3, 8, 8), ACT_STATE_KEY: torch.zeros(1, 6)}

    class Critic:
        def __call__(self, *args, task_context, policy_context):
            return policy_context[:, 0] + task_context[:, 0]

    done, failed = multi._run_sessions(
        [session], adapter=SimpleNamespace(build_batch=batch), policy=object(), critic=Critic(),
        trainer=SimpleNamespace(exploration=ActPpoExploration(ppo)),
        device=torch.device("cpu"), loaded_config=SimpleNamespace(n_action_steps=2), base_config=base, ppo_config=ppo,
    )
    assert failed == [] and done == [session]
    assert sample_shapes == [(1, 1, 6)] * 4
    assert forward_frames == ([0, 2] if terminal else [0, 2, 4])
    assert session.buffer.actor_refs == [0, 0, 1, 1]
    assert session.buffer.action_indices == [0, 1, 0, 1]
    assert len(session.buffer.actor_observations) == 2  # No phantom transition at truncation.
    assert session.buffer.values == [0., 1.5, 2., 3.5]
    assert session.buffer.next_values == [1.5, 2., 3.5, 0. if terminal else 4.]
    np.testing.assert_allclose(session.buffer.exploration_log_std_offsets,
                               [[value] * 6 for value in [-2.7, -1.7, -1.2, -1.7]])
    np.testing.assert_allclose(session.buffer.policy_action_means,
                               [[1., 1., 1., 1., 1., 51.]] * 2 + [[2., 2., 2., 2., 2., 52.]] * 2)
    np.testing.assert_allclose(session.buffer.policy_action_log_stds,
                               [[value] * 6 for value in [-5., -4., -3.5, -4.]])
    assert session.buffer.actor_update_enabled == [True] * 4
    assert [x[0] for x in session.buffer.critic_task_context] == [0., 1., 2., 3.]


def test_pick_place_worker_payload_preserves_resolved_reward(tmp_path):
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig

    reward = PickPlaceRewardConfig(gamma=.995, failure_cost=5.)
    config = worker.GazeboWorkerConfig(
        world_index=0, ros_domain_id=20, gz_partition="unit-test", gazebo_gui=False,
        project_root=tmp_path, output_directory=tmp_path, startup_feedback_timeout_s=1.,
        reward_profile="pick_place_v2", reward_config=reward.to_dict(), n_action_steps=30,
    )
    restored = worker.GazeboWorkerConfig.from_payload(config.payload())
    assert PickPlaceRewardConfig.from_dict(restored.reward_config).config_hash == reward.config_hash
    assert restored.n_action_steps == 30


def test_single_pick_place_entry_stores_chunk_start_refs_and_terminal_zero(monkeypatch, tmp_path):
    import numpy as np
    from contextlib import contextmanager
    from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM

    phases = ["WAIT_HOME", "APPROACH", "GRASP_LIFT", "HOME_HOLD"]

    def info(index):
        return {"task_phase": phases[index], "critic_task_context": [float(index)] * TASK_CONTEXT_DIM,
                "task_outcome": "success" if index == 3 else "running"}

    def observation(index):
        return SimpleNamespace(top_image=np.full((8, 8, 3), index, np.uint8),
                               gripper_image=np.zeros((8, 8, 3), np.uint8),
                               joint_positions=np.zeros(6), joint_velocities=np.zeros(6))

    class Runtime:
        index = 0
        configured = None

        def configure_task(self, mode, **kwargs):
            self.configured = (mode, kwargs)

        def delete_all_targets(self):
            return SimpleNamespace(success=True, message="deleted")

        def spawn_random_target(self):
            return SimpleNamespace(success=True, message="spawned")

        def prepare_episode(self):
            return SimpleNamespace(observation=observation(0), info=info(0))

        def step(self, command):
            self.index += 1
            return SimpleNamespace(observation=observation(self.index), info=info(self.index), reward=1.,
                                   terminated=self.index == 3, truncated=False)

    runtime = Runtime()

    @contextmanager
    def runtime_factory(**kwargs):
        assert kwargs["reward_profile"] == "pick_place_v2"
        yield runtime

    policy = FakePolicy()
    loaded = _fake_loaded(policy)
    loaded.config.n_action_steps = loaded.config.chunk_size = 2
    forward_count = []

    def predict(*args):
        forward_count.append(1)
        lower, upper = policy_action_bounds(device="cpu")
        return ((lower + upper) / 2).expand(1, 2, 6).clone()

    captured = {}

    class Trainer:
        actor_learning_rate = actor_backbone_learning_rate = .001
        update_index = total_environment_steps = 0
        def __init__(self, policy, critic, **kwargs):
            configure_bounded_act_policy(policy)
            self.policy = policy
            self.exploration = ActPpoExploration(kwargs["config"])
            assert kwargs["execution_signature"]["reward_profile"] == "pick_place_v2"

        def update(self, buffer):
            self.update_index += 1
            self.total_environment_steps += len(buffer)
            with torch.no_grad():
                self.policy.weight.add_(1.)
            captured.update(refs=list(buffer.actor_refs), indices=list(buffer.action_indices),
                            next_values=list(buffer.next_values), tasks=[x[0] for x in buffer.critic_task_context])
            np.testing.assert_allclose(buffer.policy_action_means, [[0., 0., 0., 0., 0., 50.]] * 3)
            np.testing.assert_allclose(buffer.policy_action_log_stds,
                                       [[value] * 6 for value in [-5., -4., -4.]])
            return SimpleNamespace(to_dict=lambda: {})

    class Critic(nn.Module):
        def forward(self, *args, task_context, policy_context):
            return task_context[:, 0] + policy_context[:, 0]

    def batch(obs):
        return {ACT_FIXED_IMAGE_KEY: torch.tensor(obs.top_image.transpose(2, 0, 1)[None]).float() / 255,
                ACT_HANDEYE_IMAGE_KEY: torch.zeros(1, 3, 8, 8), ACT_STATE_KEY: torch.zeros(1, 6)}

    monkeypatch.setattr(single, "ActPpoTrainer", Trainer)
    monkeypatch.setattr(single, "ActPpoValueNetwork", lambda config: Critic())
    monkeypatch.setattr(single, "predict_action_chunk_with_grad", predict)
    monkeypatch.setattr(single, "_map_policy_positions_for_episode", lambda *args: SimpleNamespace(
        action=SimpleNamespace(controller_command=lambda: np.zeros(6))))
    monkeypatch.setattr(single, "_save_update", lambda *args, **kwargs: {})
    monkeypatch.setattr(single, "_export_final", lambda *args, **kwargs: (tmp_path / "act", tmp_path / "state.pt"))
    config = single.ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "act", output_directory=tmp_path / "run", device="cpu",
        reward_profile="pick_place_v2", final_target_absent_s=0, target_delay_s=0,
        max_steps_per_episode=3, episode_count=1, rollout_steps=10,
        critic_image_height=8, critic_image_width=8,
    )
    result = single.run_act_ppo_gazebo(config, checkpoint_loader=lambda *args, **kwargs: loaded,
        runtime_factory=runtime_factory, adapter_factory=lambda *args, **kwargs: SimpleNamespace(build_batch=batch))
    assert captured == {"refs": [0, 0, 1], "indices": [0, 1, 0], "next_values": [1.5, 2., 0.], "tasks": [0., 1., 2.]}
    assert len(forward_count) == 2
    assert result["task_success_count"] == 1
    assert result["valid_episode_count"] == 1
    assert runtime.configured[1]["n_action_steps"] == 2
