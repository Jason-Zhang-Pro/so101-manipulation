"""Stage success skipping across real collection and checkpoint boundaries.

Only the simulator responses and numerical learner are fakes. World selection,
episode recording, stage scheduling, filtering, and checkpoint orchestration
are the production implementation.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo import configure_bounded_act_policy
from test_act_ppo_stage_wave_integration import FakeClient, episode_buffer, stage_config


def run_collection(tmp_path, monkeypatch, *, success_for, minimum_episodes=1,
                   selection="s0_s1", rollout_steps=20, resume=None,
                   rejected_updates=()):
    events, clients, trainers, saved = [], [], [], []
    stage_episode_counts = Counter()
    policy = torch.nn.Linear(1, 1)
    policy.reset = lambda: None
    policy.save_pretrained = lambda directory: Path(directory).mkdir(parents=True, exist_ok=True)
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
        def __init__(self, count, accepted):
            self.transition_count = count
            self.completed_ppo_epochs = int(accepted)
            self.configured_ppo_epochs = 1
            self.early_stopped = not accepted

        def __getattr__(self, name):
            return 0.

        def to_dict(self):
            return vars(self).copy()

    class Trainer:
        actor_learning_rate = 3e-8
        actor_backbone_learning_rate = 3e-9

        def __init__(self, policy, critic, *, config, **kwargs):
            configure_bounded_act_policy(policy)
            self.policy, self.config = policy, config
            self.update_index = self.total_environment_steps = 0
            self.collection_state = {}
            self.exploration = SimpleNamespace(value=lambda: torch.tensor([-4.]))
            trainers.append(self)

        def update(self, rollout):
            assert len(rollout) > 0, "Empty filtered batches must not enter PPO."
            assert all(not client.active for client in clients)
            index = self.config.advantage_group_context_index
            assert index is not None
            groups = Counter(int(context[index]) for context in rollout.critic_task_context)
            events.append(("ppo", len(rollout), dict(groups), deepcopy(self.collection_state)))
            self.update_index += 1
            self.total_environment_steps += len(rollout)
            accepted = self.update_index not in rejected_updates
            if accepted:
                with torch.no_grad():
                    policy.weight.add_(1.)
            return Metrics(len(rollout), accepted)

        def save_checkpoint(self, path, *, extra):
            assert all(not client.active for client in clients)
            payload = deepcopy({"update_index": self.update_index,
                                "total_environment_steps": self.total_environment_steps,
                                "extra": extra})
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload))
            saved.append((destination, payload))
            events.append(("checkpoint", self.update_index, deepcopy(extra["collection_state"])))
            return destination

        def load_checkpoint(self, path, **kwargs):
            payload = json.loads(Path(path).read_text())
            self.update_index = payload["update_index"]
            self.total_environment_steps = payload["total_environment_steps"]
            events.append(("resume", deepcopy(payload["extra"]["collection_state"])))
            return payload

    def run_sessions(sessions, *, base_config, ppo_config, **kwargs):
        stage = base_config.training_stages.upper()
        indices = {session.slot.world_index for session in sessions}
        assert indices == {client.world_index for client in clients if client.active}
        assert 1 <= len(sessions) <= 4
        events.append(("collect", stage, len(sessions)))
        for session in sessions:
            stage_episode_counts[stage] += 1
            total = sum(stage_episode_counts.values())
            success = bool(success_for(stage, stage_episode_counts[stage], total))
            session.buffer = episode_buffer(ppo_config, 2, session.slot.world_index)
            index = ppo_config.advantage_group_context_index
            assert index is not None
            for context in session.buffer.critic_task_context:
                context[index] = int(stage[1:])
            session.completed_steps = 2
            session.episode_reward = 2.
            session.terminated, session.truncated = success, not success
            session.last_info.update({"stage_result": "SUCCESS" if success else "TIMEOUT",
                                      "stage_timeout": not success, "stage_success": success,
                                      "task_outcome": "running"})
        return sessions, []

    monkeypatch.setattr(multi, "GazeboWorkerClient", Client)
    monkeypatch.setattr(multi, "ActPpoTrainer", Trainer)
    monkeypatch.setattr(multi, "ActPpoValueNetwork", lambda config: torch.nn.Identity())
    monkeypatch.setattr(multi, "ActPolicyAdapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(multi, "load_act_checkpoint", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(multi, "_run_sessions", run_sessions)
    monkeypatch.setattr(multi, "_terminal_print_block", lambda *args, **kwargs: None)
    base = replace(stage_config(
        tmp_path, episode_count=minimum_episodes, rollout_steps=rollout_steps,
        stage_config=Path(__file__).resolve().parents[2] / "config/act_ppo_stages_s1_progress.json",
        checkpoint_every_updates=1,
        s0_budget_s=2 / 30, s0_home_hold_s=1 / 30,
        s1_budget_s=2 / 30, s1_reach_hold_s=1 / 30,
    ), training_stages=selection, resume_checkpoint=resume)
    worlds = multi.MultiWorldConfig(
        world_count=4, world_restart_max_attempts=0, preflight_cleanup_enabled=False,
        s0_step_fraction=.5, stage_ratio_tolerance=0.,
        stage_success_skip_enabled=True, stage_success_skip_threshold=.8,
    )
    summary = multi.run_act_ppo_multiworld_gazebo(base, worlds)
    records = [json.loads(line) for line in
               (base.output_directory / "metrics.jsonl").read_text().splitlines()]
    assert all(not client.active for client in clients)
    # Every valid trajectory remains in the audit output, including those
    # intentionally withheld from every PPO objective.
    for episode in summary["episodes"]:
        rows = [json.loads(line) for line in
                (base.output_directory / episode["step_records"]["path"]).read_text().splitlines()]
        assert len(rows) == episode["recorded_transition_count"] == 2
        assert rows[-1]["episode_end"] is True
    return SimpleNamespace(summary=summary, records=records, events=events,
                           saved=saved, trainer=trainers[0], base=base, worlds=worlds)


@pytest.mark.parametrize("s0_successes,expected_groups", [(3, {0: 10, 1: 10}), (4, {1: 10})])
def test_gate_uses_current_complete_episode_success_rate_at_eighty_percent(
    tmp_path, monkeypatch, s0_successes, expected_groups,
):
    run = run_collection(tmp_path, monkeypatch,
                         success_for=lambda stage, count, total: stage == "S0" and count <= s0_successes)
    updates = [event for event in run.events if event[0] == "ppo"]
    assert len(updates) == 1
    assert updates[0][2] == expected_groups
    assert run.summary["episode_count"] == 10
    assert run.summary["total_collected_steps"] == 20
    assert run.trainer.total_environment_steps == sum(expected_groups.values())
    assert run.summary["discarded_environment_steps"] == 0


def test_next_batch_collects_full_threshold_from_remaining_stage_then_restores_both(
    tmp_path, monkeypatch,
):
    run = run_collection(tmp_path, monkeypatch, minimum_episodes=21,
                         success_for=lambda stage, count, total: stage == "S0" and count <= 4)
    updates = [event for event in run.events if event[0] == "ppo"]
    assert [event[2] for event in updates] == [{1: 10}, {1: 20}, {0: 10, 1: 10}]
    assert run.summary["total_collected_steps"] == 60
    assert run.summary["episode_count"] == 30
    assert run.summary["ppo_update_count"] == 3
    assert run.trainer.total_environment_steps == 50


def test_kl_rejection_still_consumes_cooldown_batch(tmp_path, monkeypatch):
    run = run_collection(tmp_path, monkeypatch, minimum_episodes=11, rejected_updates=(1,),
                         success_for=lambda stage, count, total: stage == "S0" and count <= 4)
    updates = [event for event in run.events if event[0] == "ppo"]
    assert [event[2] for event in updates] == [{1: 10}, {1: 20}]
    assert run.summary["updates"][0]["completed_ppo_epochs"] == 0
    assert run.summary["ppo_update_count"] == 2


def test_all_stages_successful_skips_empty_update_and_one_worldless_cycle(tmp_path, monkeypatch):
    run = run_collection(tmp_path, monkeypatch, minimum_episodes=11,
                         success_for=lambda stage, count, total: total <= 10)
    updates = [event for event in run.events if event[0] == "ppo"]
    assert len(updates) == 1 and updates[0][2] == {0: 10, 1: 10}
    assert run.summary["episode_count"] == 20
    assert run.summary["total_collected_steps"] == 40
    assert run.summary["ppo_update_count"] == 1
    assert len([record for record in run.records if record["type"] == "stage_update_skipped"]) == 1
    assert len([record for record in run.records if record["type"] == "stage_sampling_skipped"]) == 1
    assert run.summary["stage_skipped_update_count"] == 1
    assert run.summary["success_gated_transition_count"] == 20
    assert sum(event[0] == "prepare" for event in run.events) == 20


@pytest.mark.parametrize("selection,group", [("s0", 0), ("s1", 1)])
def test_single_stage_resumes_after_one_empty_cooldown_cycle(tmp_path, monkeypatch, selection, group):
    run = run_collection(tmp_path, monkeypatch, selection=selection, rollout_steps=10,
                         minimum_episodes=6, success_for=lambda stage, count, total: count <= 5)
    updates = [event for event in run.events if event[0] == "ppo"]
    assert len(updates) == 1 and updates[0][2] == {group: 10}
    assert run.summary["episode_count"] == 10
    assert run.summary["ppo_update_count"] == 1


def test_saved_checkpoint_restores_next_skip_but_not_previous_batch_counts(tmp_path, monkeypatch):
    first = run_collection(tmp_path / "first", monkeypatch,
                           success_for=lambda stage, count, total: stage == "S0" and count <= 4)
    checkpoint, payload = next((path, payload) for path, payload in first.saved
                               if path.parent.name == "update_000001")
    assert "stage_success_skip" in payload["extra"]["collection_state"]
    resumed = run_collection(tmp_path / "resumed", monkeypatch, resume=checkpoint,
                             success_for=lambda stage, count, total: False)
    updates = [event for event in resumed.events if event[0] == "ppo"]
    assert len(updates) == 1 and updates[0][2] == {1: 20}
    assert resumed.summary["episode_count"] == 10
    assert resumed.summary["total_collected_steps"] == 20
    assert [row["training_stage"] for row in resumed.summary["episodes"]] == ["S1"] * 10
    assert resumed.trainer.update_index == 2
    assert resumed.trainer.collection_state["stage_attempt_count"] == 20


def test_success_only_finish_saves_unchanged_policy_and_pending_cooldown(tmp_path, monkeypatch):
    run = run_collection(tmp_path, monkeypatch, success_for=lambda stage, count, total: True)
    assert not [event for event in run.events if event[0] == "ppo"]
    assert run.summary["ppo_update_count"] == 0
    assert run.summary["act_parameters_changed"] is False
    assert run.summary["stage_skipped_update_count"] == 1
    assert run.trainer.update_index == run.trainer.total_environment_steps == 0
    assert run.trainer.collection_state["stage_success_skip"]["current_skipped_stages"] == ["S0", "S1"]
    assert run.trainer.collection_state["stage_success_skip"]["collection_batch_index"] == 1
    path, payload = next((path, payload) for path, payload in run.saved
                         if path.name == "final_training_state.pt")
    assert path.is_file()
    assert payload["extra"]["collection_state"] == run.trainer.collection_state


def test_resume_all_skipped_consumes_one_worldless_cycle_before_fresh_collection(tmp_path, monkeypatch):
    first = run_collection(tmp_path / "first", monkeypatch,
                           success_for=lambda stage, count, total: True)
    checkpoint = next(path for path, payload in first.saved if path.name == "final_training_state.pt")
    resumed = run_collection(tmp_path / "resumed", monkeypatch, resume=checkpoint,
                             success_for=lambda stage, count, total: False)
    assert [event[2] for event in resumed.events if event[0] == "ppo"] == [{0: 10, 1: 10}]
    skipped = [record for record in resumed.records if record["type"] == "stage_sampling_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["batch_index"] == 1
    assert sum(event[0] == "prepare" for event in resumed.events) == 10
    assert resumed.trainer.collection_state["stage_success_skip"]["collection_batch_index"] == 3
