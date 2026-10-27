"""Exercise stage scheduling at real collection, persistence and PPO boundaries."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo import (
    ActPpoActorObservation, ActPpoRolloutBuffer, configure_bounded_act_policy,
)
from so101_capstone.training.residual_ppo import CompressedResidualObservation


GEOMETRY = dict(goal_center_world_m=[.2, .2, .75], goal_half_extents_xy_m=[.05, .05],
                goal_yaw_rad=0., target_half_extents_m=[.015] * 3, source_support_z_m=.75)


def stage_config(tmp_path, **kwargs):
    return single.ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "fake-act", output_directory=tmp_path / "run",
        training_stages="s0_s1", reward_profile="pick_place_v2", device="cpu",
        final_target_absent_s=0., target_delay_s=0., critic_image_height=8,
        critic_image_width=8, tensorboard_enabled=False, **kwargs,
    )


def episode_buffer(ppo_config, step_count, world_index=0):
    buffer = ActPpoRolloutBuffer(ppo_config)
    actor_ref = buffer.add_actor_observation(ActPpoActorObservation(
        top_image=np.full((3, 8, 8), world_index, np.uint8),
        gripper_image=np.zeros((3, 8, 8), np.uint8),
        policy_state=np.zeros(6, np.float32),
    ))
    for step in range(step_count):
        buffer.add(
            CompressedResidualObservation(
                top_image=np.zeros((3, 8, 8), np.uint8),
                gripper_image=np.zeros((3, 8, 8), np.uint8),
                joint_positions=np.zeros(6, np.float32),
                joint_velocities=np.zeros(6, np.float32),
            ),
            actor_observation_index=actor_ref, action_index=0,
            policy_action=np.zeros(6), log_probability=-.5,
            value=0., reward=1., next_value=.5, terminated=False,
            episode_end=step == step_count - 1,
            critic_task_context=np.zeros(ppo_config.critic_task_context_dim),
            critic_policy_context=np.zeros(ppo_config.critic_policy_context_dim),
        )
    return buffer


class FakeClient:
    """Fake simulation protocol; scheduling and process lifecycle stay real."""

    def __init__(self, config, events):
        self.config = config
        self.world_index = config.world_index
        self.events = events
        self.active = False

    def begin_start(self):
        self.events.append(("start", self.world_index))

    def finish_start(self):
        self.active = True
        return {"type": "ready"}

    def begin_stop(self):
        self.events.append(("stop", self.world_index))

    def finish_stop(self):
        self.active = False
        return {"type": "world_stopped", "world_index": self.world_index}

    def begin_restart(self, *, reason="runtime_restart"):
        assert not self.active
        self.events.append(("restart", self.world_index, reason))
        return "runtime_restart"

    def finish_restart(self, mode):
        assert mode == "runtime_restart"
        self.active = True
        return {"type": "ready", "attempts_used": 1}

    def shutdown(self):
        self.active = False
        self.events.append(("shutdown", self.world_index))

    def prepare_episode_async(self, attempt, *, task_mode, scene_seed):
        assert self.active
        self.events.append(("prepare", self.world_index, attempt, task_mode, scene_seed))
        self.response = {"type": "episode_prepared", "observation": "before_stage",
                         "info": GEOMETRY, "cleanup": {}, "home_reset": {}}

    def spawn_target_async(self, position, quaternion):
        self.events.append(("spawn", self.world_index))
        self.response = {"type": "target_spawned", "success": True, "message": "{}"}

    def start_training_stage_async(self, training_stage, stage_config):
        self.events.append(("stage", self.world_index, training_stage))
        self.response = {
            "type": "stage_started", "observation": "frame_zero",
            "info": {"training_stage": training_stage, "stage_elapsed_s": 0.,
                     "start_certificate": {"start_valid": True, "training_stage": training_stage}},
        }

    def receive(self, **kwargs):
        return self.response

    def observation_from_metadata(self, metadata):
        return metadata


@pytest.mark.parametrize("stage,mode", [("S0", "idle_empty"), ("S1", "pick_place")])
def test_real_wave_retry_preserves_stage_and_advances_attempt_and_scene_seeds(tmp_path, monkeypatch, stage, mode):
    base = stage_config(tmp_path)
    ppo = single._act_ppo_config(base, 2)
    events = []
    clients = [FakeClient(SimpleNamespace(world_index=index), events) for index in range(2)]
    for client in clients:
        client.active = True
    slots = [multi._WorldSlot(client=client, available=True) for client in clients]
    calls = []

    def run_sessions(sessions, *, base_config, on_failure, **kwargs):
        calls.append([session.attempt_index for session in sessions])
        assert base_config.training_stages == stage.lower()
        assert all(session.observation == "frame_zero" for session in sessions)
        assert all(session.task_mode == mode for session in sessions)
        assert all(session.last_info["training_stage"] == stage for session in sessions)
        completed = sessions if len(calls) == 2 else sessions[:1]
        for session in completed:
            session.buffer = episode_buffer(ppo, 2, session.slot.world_index)
            session.completed_steps = 2
            session.episode_reward = 2.
        if len(calls) == 2:
            return completed, []
        failed = multi._EpisodeFailure(
            slot=sessions[1].slot, attempt_index=sessions[1].attempt_index,
            completed_steps=0, discarded_transition_count=0,
            category="camera", error_type="Timeout", message="camera timeout",
        )
        on_failure(failed)  # The normal callback plus returned failure must count once.
        return completed, [failed]

    monkeypatch.setattr(multi, "_run_sessions", run_sessions)
    monkeypatch.setattr(multi, "_terminal_print_block", lambda *args, **kwargs: None)
    attempt_counter = [40]
    completed, failed, discarded = multi._run_wave(
        slots, adapter=None, policy=None, critic=None, trainer=None,
        device=torch.device("cpu"), loaded_config=None, base_config=base,
        multiworld_config=multi.MultiWorldConfig(world_count=2, world_restart_max_attempts=1),
        ppo_config=ppo, output_directory=tmp_path, attempt_counter=attempt_counter,
        wave_training_stage=stage,
    )
    assert calls == [[41, 42], [43]]
    assert attempt_counter == [43]
    assert [session.attempt_index for session in completed] == [41, 43]
    assert [record["episode_attempt_index"] for record in failed] == [42]
    assert slots[1].failed_episode_count == 1 and discarded == 0
    assert base.training_stages == "s0_s1"
    assert [event for event in events if event[0] == "prepare"] == [
        ("prepare", world, attempt, mode, base.seed + attempt * 1009 + world * 100003)
        for world, attempt in [(0, 41), (1, 42), (1, 43)]
    ]
    assert [event for event in events if event[0] == "stage"] == [
        ("stage", world, stage) for world in [0, 1, 1]
    ]
    assert [event[1] for event in events if event[0] == "restart"] == [1]
    assert [event[1] for event in events if event[0] == "spawn"] == ([0, 1, 1] if stage == "S1" else [])


@pytest.mark.parametrize("invalid", ["missing", "wrong_count", "unready"])
def test_update_guard_rejects_missing_inconsistent_or_unbalanced_stage_counts(tmp_path, invalid):
    base = stage_config(tmp_path, rollout_steps=16)
    worlds = multi.MultiWorldConfig(world_count=4)
    scheduler = multi._stage_wave_scheduler(base, worlds)
    if invalid == "missing":
        scheduler = None
    elif invalid == "wrong_count":
        scheduler.observe("S0", 12)
        scheduler.observe("S1", 16)
        assert scheduler.ready
    else:
        scheduler.observe("S0", 8)
        scheduler.observe("S1", 8)
        assert scheduler.total_steps == 16 and not scheduler.ready
    rollout = episode_buffer(single._act_ppo_config(base, 2), 16)
    with pytest.raises(RuntimeError, match="requested actual step ratio"):
        multi._perform_update(
            rollout, trainer=object(), output_directory=tmp_path,
            base_config=base, multiworld_config=worlds, tensorboard=object(),
            rollout_index=1, stage_scheduler=scheduler,
        )


@pytest.mark.parametrize("change", [
    {"s0_step_fraction": .5}, {"stage_ratio_tolerance": .025},
    {"stage_max_rollout_multiplier": 8.}, {"world_count": 2},
])
def test_resume_signature_covers_stage_sampling_contract(tmp_path, monkeypatch, change):
    monkeypatch.delenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", raising=False)
    monkeypatch.delenv("SO101_IMAGE_AUGMENTATION_ENABLED", raising=False)
    base = stage_config(tmp_path, rollout_steps=16)
    worlds = multi.MultiWorldConfig(world_count=4)
    reward = single._resolved_reward(base)
    signature = multi._multiworld_execution_signature(base, reward, worlds)
    assert multi._multiworld_execution_signature(base, reward, replace(worlds, **change)) != signature


@pytest.mark.parametrize("change", [
    {"s0_budget_s": 6.}, {"s0_home_hold_s": 2.},
    {"s1_budget_s": 12.}, {"s1_reach_hold_s": .2}, {"rollout_steps": 32},
])
def test_resume_signature_covers_stage_time_and_rollout_threshold(tmp_path, change):
    base = stage_config(tmp_path, rollout_steps=16)
    worlds = multi.MultiWorldConfig(world_count=4)
    reward = single._resolved_reward(base)
    signature = multi._multiworld_execution_signature(base, reward, worlds)
    assert multi._multiworld_execution_signature(replace(base, **change), reward, worlds) != signature


def test_unused_episode_probability_does_not_change_wave_resume_signature(tmp_path):
    base = stage_config(tmp_path, rollout_steps=16)
    worlds = multi.MultiWorldConfig(world_count=4)
    reward = single._resolved_reward(base)
    signature = multi._multiworld_execution_signature(base, reward, worlds)
    assert "s0_episode_probability" not in signature["stage_training"]
    assert multi._multiworld_execution_signature(replace(base, s0_probability=.99), reward, worlds) == signature


@pytest.mark.parametrize("minimum_episodes,update_count", [(4, 1), (12, 2)])
def test_real_main_loop_finishes_balanced_batches_with_smaller_tail_waves(
    tmp_path, monkeypatch, minimum_episodes, update_count,
):
    """Keep real planning, stage preparation, recording, update gate and reset."""
    events, clients, trainers = [], [], []
    collected_slots = {}
    policy = torch.nn.Linear(1, 1)
    policy.reset = lambda: None
    loaded = SimpleNamespace(policy=policy, device=torch.device("cpu"), config=SimpleNamespace(
        temporal_ensemble_coeff=None, n_action_steps=2, chunk_size=2,
        optimizer_lr=3e-8, optimizer_lr_backbone=3e-9,
        optimizer_weight_decay=0., use_vae=True, vision_backbone="fake",
    ))
    policy.config = loaded.config

    class Client(FakeClient):
        def __init__(self, config):
            super().__init__(config, events)
            clients.append(self)

    class Metrics:
        def __init__(self, transition_count):
            self.transition_count = transition_count

        def __getattr__(self, name):
            return 0.

        def to_dict(self):
            return {"transition_count": self.transition_count}

    class Trainer:
        actor_learning_rate = 3e-8
        actor_backbone_learning_rate = 3e-9

        def __init__(self, policy, critic, *, config, **kwargs):
            configure_bounded_act_policy(policy)
            self.config = config
            self.update_index = self.total_environment_steps = 0
            self.collection_state = {}
            self.exploration = SimpleNamespace(value=lambda: torch.tensor([-4.]))
            trainers.append(self)

        def update(self, rollout):
            assert all(not client.active for client in clients)
            assert len(rollout) == 32
            assert rollout.episode_end.count(True) == 11
            events.append(("ppo", len(rollout), dict(self.collection_state)))
            self.update_index += 1
            self.total_environment_steps += len(rollout)
            if self.update_index == 1:
                # Model a world whose retry allowance was exhausted in the
                # consumed batch. The main loop must reset every slot before
                # selecting any smaller subset for the next batch.
                collected_slots[3].consecutive_episode_failures = 3
                collected_slots[3].quarantined = True
            with torch.no_grad():
                policy.weight.add_(1.)
            return Metrics(len(rollout))

    def run_sessions(sessions, *, base_config, ppo_config, **kwargs):
        stage = base_config.training_stages.upper()
        count = 2 if stage == "S0" else 4
        selected_indices = {session.slot.world_index for session in sessions}
        collected_slots.update({session.slot.world_index: session.slot for session in sessions})
        assert {client.world_index for client in clients if client.active} == selected_indices
        assert 1 <= len(sessions) <= 4
        assert all(session.last_info["training_stage"] == stage for session in sessions)
        events.append(("collect", stage, len(sessions)))
        for session in sessions:
            session.buffer = episode_buffer(ppo_config, count, session.slot.world_index)
            session.completed_steps = count
            session.episode_reward = float(count)
            session.truncated = True
            session.last_info.update({"stage_result": "TIMEOUT", "stage_timeout": True,
                                      "stage_success": False, "task_outcome": "running"})
        return sessions, []

    def export(trainer, *, output_directory, **kwargs):
        assert all(not client.active for client in clients)
        events.append(("export",))
        return output_directory / "final", output_directory / "state.pt"

    real_prepare = multi._prepare_stage_worlds_for_wave

    def prepare(slots, **kwargs):
        if kwargs["starting_new_rollout"]:
            # Check before launch/recovery could reset only selected slots.
            assert all(slot.consecutive_episode_failures == 0 for slot in slots)
            assert all(not slot.quarantined for slot in slots)
            events.append(("new_batch_failure_budgets_reset",))
        return real_prepare(slots, **kwargs)

    monkeypatch.setattr(multi, "GazeboWorkerClient", Client)
    monkeypatch.setattr(multi, "ActPpoTrainer", Trainer)
    monkeypatch.setattr(multi, "ActPpoValueNetwork", lambda config: torch.nn.Identity())
    monkeypatch.setattr(multi, "ActPolicyAdapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(multi, "load_act_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(multi, "_run_sessions", run_sessions)
    monkeypatch.setattr(multi, "_export_final", export)
    monkeypatch.setattr(multi, "_prepare_stage_worlds_for_wave", prepare)
    monkeypatch.setattr(multi, "_terminal_print_block", lambda *args, **kwargs: None)
    base = stage_config(
        tmp_path, episode_count=minimum_episodes, rollout_steps=32,
        checkpoint_every_updates=100,
        s0_budget_s=2 / 30, s0_home_hold_s=1 / 30,
        s1_budget_s=4 / 30, s1_reach_hold_s=1 / 30,
    )
    worlds = multi.MultiWorldConfig(world_count=4, world_restart_max_attempts=0,
                                   preflight_cleanup_enabled=False)
    summary = multi.run_act_ppo_multiworld_gazebo(base, worlds)
    expected_waves = [("collect", "S1", 4), ("collect", "S0", 4),
                      ("collect", "S0", 2), ("collect", "S1", 1)]
    assert [event for event in events if event[0] == "collect"] == expected_waves * update_count
    assert summary["requested_minimum_episode_count"] == minimum_episodes
    assert summary["episode_count"] == 11 * update_count
    assert summary["ppo_update_count"] == update_count
    assert summary["total_collected_steps"] == 32 * update_count
    assert summary["discarded_final_partial_transitions"] == 0
    assert sum(event[0] == "new_batch_failure_budgets_reset" for event in events) == update_count - 1
    assert all(slot.consecutive_episode_failures == 0 for slot in collected_slots.values())
    assert trainers[0].collection_state == {"stage_attempt_count": 11 * update_count}
    assert [event[2] for event in events if event[0] == "ppo"] == [
        {"stage_attempt_count": 11 * index} for index in range(1, update_count + 1)
    ]
    for update in summary["updates"]:
        assert update["actual_rollout_transition_count"] == 32
        assert update["partial_rollout"] is False
        balance = update["stage_sampling"]
        assert balance["s0_steps"] == 12 and balance["s1_steps"] == 20
        assert balance["actual_s0_step_fraction"] == .375
        assert balance["ready"] is True
    records = [json.loads(line) for line in (base.output_directory / "metrics.jsonl").read_text().splitlines()]
    sampling = [record for record in records if record["type"] == "stage_wave_sampling"]
    assert [record["total_steps"] for record in sampling] == [16, 24, 28, 32] * update_count
    # Every accepted episode is complete and appears exactly once in storage.
    assert len({record["episode_attempt_index"] for record in summary["episodes"]}) == 11 * update_count
    for episode in summary["episodes"]:
        rows = [json.loads(line) for line in
                (base.output_directory / episode["step_records"]["path"]).read_text().splitlines()]
        assert len(rows) == episode["recorded_transition_count"]
        assert rows[-1]["episode_end"] is True
        assert rows[-1]["next_values"] == .5
    last_collect = max(index for index, event in enumerate(events) if event[0] == "collect")
    assert not any(event[0] in {"start", "restart"} for event in events[last_collect + 1:])
    assert all(not client.active for client in clients)


@pytest.mark.parametrize("failed_start", [False, True])
def test_stage_world_preparation_starts_only_selected_worlds_and_replaces_failed_launches(
    tmp_path, failed_start,
):
    events = []
    clients = [FakeClient(SimpleNamespace(world_index=index), events) for index in range(4)]
    if failed_start:
        clients[0].finish_start = lambda: {"type": "startup_failed"}
    slots = [multi._WorldSlot(client=client) for client in clients]
    ready = multi._prepare_stage_worlds_for_wave(
        slots, world_count=2, output_directory=tmp_path,
        multiworld_config=multi.MultiWorldConfig(world_count=4, world_restart_max_attempts=0),
        starting_new_rollout=False,
    )
    expected = [1, 2] if failed_start else [0, 1]
    assert [slot.world_index for slot in ready] == expected
    assert [client.world_index for client in clients if client.active] == expected
    assert [event[1] for event in events if event[0] == "start"] == ([0, 1, 2] if failed_start else [0, 1])
    assert slots[3].startup_response is None
    assert slots[0].quarantined is failed_start


@pytest.mark.parametrize("saved_attempt_count", [0, 7, 47])
def test_partial_world_selection_replays_from_saved_attempt_count(tmp_path, saved_attempt_count):
    events = []
    clients = [FakeClient(SimpleNamespace(world_index=index), events) for index in range(4)]
    slots = [multi._WorldSlot(client=client, successful_episode_count=100 - index,
                             failed_episode_count=index)
             for index, client in enumerate(clients)]
    ready = multi._prepare_stage_worlds_for_wave(
        slots, world_count=2, output_directory=tmp_path,
        multiworld_config=multi.MultiWorldConfig(world_count=4, world_restart_max_attempts=0),
        starting_new_rollout=False, selection_offset=saved_attempt_count,
    )
    assert [slot.world_index for slot in ready] == [
        saved_attempt_count % 4, (saved_attempt_count + 1) % 4,
    ]
    assert sum(client.active for client in clients) == 2
