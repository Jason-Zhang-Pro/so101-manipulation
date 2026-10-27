"""Regression checks at the reward/curriculum/sampler/deployment boundaries."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_capstone.training.act_ppo import ActPpoExploration, evaluate_bounded_policy_actions
from so101_capstone.training.act_ppo_curriculum import ReachCurriculum
from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerClient


def config(**kwargs):
    return single.ActPpoGazeboConfig(act_checkpoint=Path("/tmp/act"), reward_profile="pick_place_v2",
        gamma=.997, final_target_absent_s=0., initial_log_std=-4., minimum_log_std=-6., maximum_log_std=-3.5,
        gripper_maximum_log_std=-1.5, moving_gripper_exploration_log_std=-3.,
        manipulation_gripper_exploration_log_std=-2.3, **kwargs)


GEOMETRY = dict(goal_center_world_m=[.2, .2, .75], goal_half_extents_xy_m=[.05, .05],
                goal_yaw_rad=0., target_half_extents_m=[.015] * 3, source_support_z_m=.75)


@pytest.mark.parametrize("steps,cap,temporal,expected", [(30, 5, None, 5), (3, 5, None, 3),
    (1, 5, .01, 1), (30, None, None, 30)])
def test_replanning_changes_exported_native_config_and_keeps_chunk_horizon(steps, cap, temporal, expected):
    from collections import deque
    cfg = SimpleNamespace(n_action_steps=steps, chunk_size=60, temporal_ensemble_coeff=temporal)
    class Policy:
        config = cfg
        model = SimpleNamespace(config=cfg)
        def reset(self):
            self.queue = deque(maxlen=self.config.n_action_steps)
    policy = Policy()
    policy.reset()
    single._configure_act_replanning(SimpleNamespace(config=cfg, policy=policy), cap)
    assert cfg.n_action_steps == policy.config.n_action_steps == policy.queue.maxlen == expected
    assert cfg.chunk_size == 60 and cfg.temporal_ensemble_coeff == temporal


def test_resume_preserves_curriculum_attempts_and_next_scene():
    cfg = config(reach_curriculum=True)
    curriculum = single._make_curriculum(cfg)
    curriculum.attempt_count = 37
    scene = single._sample_curriculum_scene(curriculum, seed=2026, task_mode="pick_place", info=GEOMETRY)
    trainer = SimpleNamespace()
    single._sync_curriculum_state(trainer, curriculum)
    resumed = single._make_curriculum(cfg, {"extra": {"collection_state": trainer.collection_state}})
    assert resumed.attempt_count == 37
    assert resumed.allocate_attempt() == 38
    assert single._sample_curriculum_scene(resumed, seed=2026, task_mode="pick_place", info=GEOMETRY) == scene
    with pytest.raises(ValueError, match="missing reach curriculum state"):
        single._make_curriculum(cfg, {"extra": {}})
    with pytest.raises(ValueError, match="config"):
        single._make_curriculum(replace(cfg, curriculum_pass_rate=.8),
            {"extra": {"collection_state": trainer.collection_state}})


@pytest.mark.parametrize("mode", ["idle_empty", "idle_completed"])
def test_curriculum_never_spawns_active_target_in_idle(mode):
    assert single._sample_curriculum_scene(ReachCurriculum(), seed=1, task_mode=mode, info={}) == {}


def test_worker_explicit_spawn_protocol_preserves_coordinates():
    messages = []
    client = object.__new__(GazeboWorkerClient)
    client.request_async = lambda command, **kw: messages.append((command, kw))
    client.spawn_target_async([.26, -.02, .768], [0., 0., 0., 1.])
    assert messages == [("spawn_target", dict(position_world_m=[.26, -.02, .768],
                                               quaternion_xyzw=[0., 0., 0., 1.]))]


@pytest.mark.parametrize("delay", [0., 1.])
def test_multiworld_prepare_and_delayed_spawn_use_same_seeded_target(monkeypatch, delay):
    monkeypatch.delenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", raising=False)
    class Client:
        world_index = 2
        explicit = []
        def prepare_episode_async(self, attempt, **kwargs):
            self.prepared = (attempt, kwargs)
        def receive(self, **kwargs):
            if self.explicit:
                return {"type": "target_spawned", "success": True, "message": "{}"}
            return {"type": "episode_prepared", "observation": None, "info": GEOMETRY,
                    "cleanup": {}, "home_reset": {}}
        def observation_from_metadata(self, _):
            return None
        def spawn_target_async(self, position, quaternion):
            self.explicit.append((position, quaternion))
        def spawn_random_target_async(self):
            raise AssertionError("Curriculum must not use the independent random target manager")
    client = Client()
    client.explicit = []
    curriculum = ReachCurriculum()
    cfg = config(reach_curriculum=True, target_delay_s=delay, randomize_target_delay=False)
    counter = [40]
    sessions, failures = multi._prepare_sessions([multi._WorldSlot(client=client)],
        ppo_config=single._act_ppo_config(cfg, 5), base_config=cfg,
        attempt_counter=counter, curriculum=curriculum)
    assert not failures and len(sessions) == 1 and curriculum.attempt_count == 41
    session = sessions[0]
    if delay:
        assert not client.explicit
        multi._spawn_session_target(session)
    assert client.explicit == [(session.curriculum_scene["target_position_world_m"], [0., 0., 0., 1.])]
    record = multi._episode_record(session, episode_index=0, base_config=cfg, rollout_size_after_episode=0)
    assert record["sampled_level"] == 0 and record["scene_seed"] == client.prepared[1]["scene_seed"]


@pytest.mark.parametrize("phase,expected", [("WAIT_HOME", -5.), ("APPROACH", -3.), ("GRASP_LIFT", -2.3)])
def test_collector_axis_noise_is_exactly_replayable(phase, expected):
    cfg = config()
    ppo = single._act_ppo_config(cfg, 5)
    exploration = ActPpoExploration(ppo)
    session = multi._EpisodeSession(slot=SimpleNamespace(), buffer=None, observation=None,
        last_info={"task_phase": phase}, cleanup={}, home_reset={}, attempt_index=1,
        chunk_means=torch.tensor([[[0., 0., 0., 0., 0., 50.]]]).expand(1, 5, 6))
    multi._sample_session_action(session, trainer=SimpleNamespace(exploration=exploration), base_config=cfg, ppo_config=ppo)
    assert session.sampled_action_log_std[-1] == pytest.approx(expected)
    std = exploration.with_offset(session.chunk_log_std_offset)
    replay, _ = evaluate_bounded_policy_actions(session.chunk_means[:, :1],
        torch.tensor(session.sampled_actions).unsqueeze(0), std,
        policy_action_latents=torch.tensor(session.sampled_action_latents).unsqueeze(0))
    np.testing.assert_allclose(replay.detach().numpy()[0], session.sampled_log_probabilities, atol=1e-12)


def test_precise_reach_does_not_count_as_grasp_or_full_task_success():
    metrics = {}
    single._accumulate_task_metrics(metrics, {"pregrasp_ready": True})
    assert metrics["precise_reach"] == 1 and metrics["grasp_valid"] == metrics["lifted_once"] == 0


@pytest.mark.parametrize("value", [float("nan"), 2., "false"])
def test_invalid_stage_evidence_cannot_be_promoted_to_success(value):
    with pytest.raises(ValueError, match="finite boolean"):
        single._accumulate_task_metrics({}, {"pregrasp_ready": value})


@pytest.mark.parametrize("parser", [single._parser, multi._parser])
def test_new_controls_reach_both_cli_config_paths(parser):
    args = parser().parse_args(["--act-checkpoint", "/tmp/act", "--reach-curriculum",
        "--act-max-action-steps", "5", "--manipulation-gripper-exploration-log-std", "-2.3",
        "--gripper-maximum-log-std", "-1.5"])
    kwargs = single._reward_cli_kwargs(args)
    assert kwargs["reach_curriculum"] and kwargs["act_max_action_steps"] == 5
    assert kwargs["manipulation_gripper_exploration_log_std"] == -2.3
    assert kwargs["gripper_maximum_log_std"] == -1.5
