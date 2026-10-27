"""Exercise the real record/copy barrier with fake worker process controls."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from so101_capstone.training.act_ppo import (
    ActPpoActorObservation,
    ActPpoConfig,
    ActPpoRolloutBuffer,
    configure_bounded_act_policy,
)
from so101_capstone.training.act_ppo_gazebo import ActPpoGazeboConfig
from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerCommunicationError
from so101_capstone.training.residual_ppo import CompressedResidualObservation


def episode_buffer(world_index):
    buffer = ActPpoRolloutBuffer(ActPpoConfig(
        critic_image_height=8, critic_image_width=8,
    ))
    actor_ref = buffer.add_actor_observation(ActPpoActorObservation(
        top_image=np.full((3, 8, 8), 10 + world_index, np.uint8),
        gripper_image=np.full((3, 8, 8), 20 + world_index, np.uint8),
        policy_state=np.full(6, world_index, np.float32),
    ))
    for step_index in range(2):
        buffer.add(
            CompressedResidualObservation(
                top_image=np.full((3, 8, 8), 30 + world_index, np.uint8),
                gripper_image=np.full((3, 8, 8), 40 + world_index, np.uint8),
                joint_positions=np.full(6, step_index, np.float32),
                joint_velocities=np.zeros(6, np.float32),
            ),
            actor_observation_index=actor_ref,
            action_index=step_index,
            policy_action=np.full(6, world_index / 10),
            log_probability=-.5,
            value=0., reward=world_index * 10 + step_index + 1., next_value=0.,
            terminated=False, episode_end=step_index == 1,
        )
    return buffer


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class FakeClient:
    def __init__(self, world_index, events):
        self.world_index = world_index
        self.events = events
        self.before_begin = lambda: None
        self.begin_failure = False
        self.ready = True
        self.active = True

    def begin_stop(self):
        self.before_begin()
        self.events.append(("stop_begin", self.world_index))

    def finish_stop(self):
        self.events.append(("stop_finish", self.world_index))
        self.active = False
        return {"type": "world_stopped", "world_index": self.world_index}

    def begin_restart(self, *, reason="runtime_restart"):
        assert not self.active
        self.before_begin()
        self.events.append(("begin", self.world_index, reason))
        if self.begin_failure:
            raise GazeboWorkerCommunicationError("worker unavailable")
        return "runtime_restart"

    def finish_restart(self, mode):
        assert mode == "runtime_restart"
        self.events.append(("finish", self.world_index))
        self.active = self.ready
        return {"type": "ready" if self.ready else "restart_failed", "attempts_used": 1}


def make_wave(tmp_path, monkeypatch, world_count):
    events = []
    sessions = []
    for world_index in range(world_count):
        buffer = episode_buffer(world_index)
        slot = multi._WorldSlot(
            client=FakeClient(world_index, events), available=True,
            consecutive_episode_failures=2, failed_episode_count=3,
        )
        sessions.append(multi._EpisodeSession(
            slot=slot, buffer=buffer, observation=None,
            last_info={"reward_state": 4, "task_outcome": "success"},
            cleanup={"requested": False, "strategy": "fresh_world"},
            home_reset={}, attempt_index=world_index + 17,
            completed_steps=2, episode_reward=sum(buffer.rewards),
        ))
    rollout = ActPpoRolloutBuffer(sessions[0].buffer.config)
    base = ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "unused-act", output_directory=tmp_path,
        device="cpu", max_steps_per_episode=2,
        target_delay_s=0., final_target_absent_s=0.,
    )
    tensorboard_records = []
    tensorboard = SimpleNamespace(log_episode=tensorboard_records.append)
    synced = []
    original_sync = multi.sync_record

    def track_sync(path):
        original_sync(path)
        synced.append(path.relative_to(tmp_path).as_posix())

    monkeypatch.setattr(multi, "sync_record", track_sync)
    monkeypatch.setattr(multi, "_terminal_print_block", lambda *args, **kwargs: None)
    kwargs = dict(
        rollout=rollout, base_config=base, output_directory=tmp_path,
        first_episode_index=5, tensorboard=tensorboard,
    )

    def assert_all_records_and_transitions_are_present():
        assert len(rollout) == world_count * 2
        assert rollout.actor_refs == [index for index in range(world_count) for _ in range(2)]
        assert rollout.episode_end == [False, True] * world_count
        episode_metrics = [row for row in read_jsonl(tmp_path / "metrics.jsonl") if row["type"] == "episode"]
        assert len(episode_metrics) == world_count
        assert len(tensorboard_records) == world_count
        assert len(synced) == world_count * 2
        for index in range(world_count):
            step_path = tmp_path / "episode_steps" / f"episode_{index + 5:06d}.jsonl"
            rows = read_jsonl(step_path)
            assert [row["step_index"] for row in rows] == [1, 2]
            assert [row["rewards"] for row in rows] == [index * 10 + 1., index * 10 + 2.]
            assert [row["episode_end"] for row in rows] == [False, True]
            episode_path = tmp_path / "episodes" / f"episode_{index + 5:06d}.json"
            record = json.loads(episode_path.read_text())
            assert record["world_index"] == index
            assert record["step_count"] == record["recorded_transition_count"] == 2
            assert record["step_records"]["step_count"] == 2
            assert record["step_records"]["path"] == str(step_path.relative_to(tmp_path))
            assert record["rollout_buffer_size_after_episode"] == 2 * (index + 1)
            assert f"episodes/episode_{index + 5:06d}.json" in synced
            assert episode_metrics[index]["episode_index"] == index + 5
        assert synced.count("metrics.jsonl") == world_count

    for session in sessions:
        session.slot.client.before_begin = assert_all_records_and_transitions_are_present
    return sessions, kwargs, events, assert_all_records_and_transitions_are_present


@pytest.mark.parametrize("world_count", [1, 3])
def test_worlds_stop_only_after_all_records_and_parent_copies(tmp_path, monkeypatch, world_count):
    sessions, kwargs, events, barrier = make_wave(tmp_path, monkeypatch, world_count)
    for session in sessions:
        def check_then_mutate_replaced_runtime_buffer(session=session):
            barrier()
            session.buffer.top_images[0].fill(255)
            session.buffer.gripper_images[0].fill(255)
            # Actor snapshots are read-only by contract. Force this test's
            # source arrays writable to verify the parent also owns copies.
            session.buffer.actor_observations[0].top_image.setflags(write=True)
            session.buffer.actor_observations[0].gripper_image.setflags(write=True)
            session.buffer.actor_observations[0].top_image.fill(255)
            session.buffer.actor_observations[0].gripper_image.fill(255)
        session.slot.client.before_begin = check_then_mutate_replaced_runtime_buffer

    records = multi._complete_episode_wave(list(reversed(sessions)), **kwargs)

    assert [record["world_index"] for record in records] == list(range(world_count))
    assert events == (
        [("stop_begin", index) for index in range(world_count)]
        + [("stop_finish", index) for index in range(world_count)]
    )
    rollout = kwargs["rollout"]
    for index, session in enumerate(sessions):
        np.testing.assert_array_equal(rollout.top_images[index * 2], 30 + index)
        np.testing.assert_array_equal(rollout.gripper_images[index * 2], 40 + index)
        np.testing.assert_array_equal(rollout.actor_observations[index].top_image, 10 + index)
        np.testing.assert_array_equal(rollout.actor_observations[index].gripper_image, 20 + index)
        assert session.slot.available is False
        assert session.slot.pending_episode_restart is True
        assert session.slot.episode_restart_count == 0
        assert session.slot.consecutive_episode_failures == 2
        assert session.slot.failed_episode_count == 3


@pytest.mark.parametrize("failure", [
    "step_write", "episode_write", "episode_sync", "metrics_write", "metrics_sync", "invalid_last_step",
])
def test_recording_failure_never_rebuilds_any_world(tmp_path, monkeypatch, failure):
    sessions, kwargs, events, _ = make_wave(tmp_path, monkeypatch, 2)
    if failure == "invalid_last_step":
        sessions[1].buffer.episode_end[-1] = False
        expected_error = ValueError
    else:
        method, failing_call = {
            "step_write": ("persist_episode_steps", 2),
            "episode_write": ("_write_json", 2),
            "episode_sync": ("sync_record", 3),
            "metrics_write": ("_append_jsonl", 2),
            "metrics_sync": ("sync_record", 4),
        }[failure]
        original = getattr(multi, method)
        calls = 0

        def fail_second_world(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == failing_call:
                raise OSError("record storage failed")
            return original(*args, **kwargs)

        monkeypatch.setattr(multi, method, fail_second_world)
        expected_error = OSError

    with pytest.raises(expected_error):
        multi._complete_episode_wave(sessions, **kwargs)

    assert events == []
    assert all(session.slot.episode_restart_count == 0 for session in sessions)
    if failure == "invalid_last_step":
        assert len(kwargs["rollout"]) == 0
        assert not (tmp_path / "episodes").exists()
    else:
        assert not any(row.get("event") == "episode_stop_begin" for row in read_jsonl(tmp_path / "metrics.jsonl"))


@pytest.mark.parametrize("failure", ["begin", "ready"])
def test_restart_failure_preserves_records_rollout_and_episode_failure_budget(tmp_path, monkeypatch, failure):
    sessions, kwargs, events, barrier = make_wave(tmp_path, monkeypatch, 2)
    failing_client = sessions[0].slot.client
    failing_client.begin_failure = failure == "begin"
    failing_client.ready = failure != "ready"
    extend_calls = []
    original_extend = kwargs["rollout"].extend

    def extend_once(buffer):
        extend_calls.append(buffer)
        return original_extend(buffer)

    monkeypatch.setattr(kwargs["rollout"], "extend", extend_once)
    records = multi._complete_episode_wave(sessions, **kwargs)
    assert all(not session.slot.client.active for session in sessions)
    events.clear()
    multi._prepare_worlds_for_wave(
        [session.slot for session in sessions], output_directory=tmp_path,
        multiworld_config=multi.MultiWorldConfig(world_restart_max_attempts=0),
        starting_new_rollout=True,
    )

    barrier()
    assert len(records) == 2
    assert extend_calls == [session.buffer for session in sessions]
    assert sessions[0].slot.available is False
    assert sessions[0].slot.quarantined is False
    assert sessions[0].slot.episode_restart_count == 0
    assert sessions[1].slot.available is True
    assert all(session.slot.consecutive_episode_failures == 2 for session in sessions)
    assert all(session.slot.failed_episode_count == 3 for session in sessions)
    assert events[:2] == [("begin", 0, "episode_complete"), ("begin", 1, "episode_complete")]
    expected_finish = [("finish", 1)] if failure == "begin" else [("finish", 0), ("finish", 1)]
    assert events[2:] == expected_finish + [("stop_begin", 0), ("stop_finish", 0)]

    # Retrying only the failed world's recovery must never reaccept the saved
    # episode, duplicate metrics, or debit its episode failure budget.
    failing_client.begin_failure = False
    failing_client.ready = True
    multi._recover_slots_concurrently(
        [sessions[0].slot], output_directory=tmp_path,
        reset_episode_failure_budget=False, quarantine_on_failure=False,
    )
    barrier()
    assert sessions[0].slot.available is True
    assert len(kwargs["rollout"]) == 4
    assert len(extend_calls) == 2
    assert sessions[0].slot.consecutive_episode_failures == 2
    assert sessions[0].slot.failed_episode_count == 3


@pytest.mark.parametrize("world_count", [1, 3])
def test_last_wave_records_every_step_without_starting_replacement_worlds(tmp_path, monkeypatch, world_count):
    sessions, kwargs, events, barrier = make_wave(tmp_path, monkeypatch, world_count)

    records = multi._complete_episode_wave(sessions, **kwargs)

    barrier()
    assert len(records) == world_count
    assert events == (
        [("stop_begin", index) for index in range(world_count)]
        + [("stop_finish", index) for index in range(world_count)]
    )
    assert all(session.slot.episode_restart_count == 0 for session in sessions)
    assert all(not session.slot.client.active for session in sessions)


@pytest.mark.parametrize("restart_limit", [0, 2])
@pytest.mark.parametrize("starting_new_rollout", [False, True])
def test_planned_replacements_start_at_next_wave_with_or_without_ppo(
    tmp_path, monkeypatch, restart_limit, starting_new_rollout,
):
    sessions, kwargs, events, barrier = make_wave(tmp_path, monkeypatch, 3)
    multi._complete_episode_wave(sessions, **kwargs)
    assert not any(event[0] == "begin" for event in events)
    assert all(not session.slot.client.active for session in sessions)
    # This is also the state in which the learner may run PPO. The next wave
    # owns startup even when positive-N failure recovery disables deferral.
    events.clear()
    multi._prepare_worlds_for_wave(
        [session.slot for session in sessions], output_directory=tmp_path,
        multiworld_config=multi.MultiWorldConfig(world_restart_max_attempts=restart_limit),
        starting_new_rollout=starting_new_rollout,
    )
    barrier()
    assert events == (
        [("begin", index, "episode_complete") for index in range(3)]
        + [("finish", index) for index in range(3)]
    )
    assert all(session.slot.available and not session.slot.pending_episode_restart for session in sessions)
    assert all(session.slot.episode_restart_count == 1 for session in sessions)
    assert all(session.slot.consecutive_episode_failures == 2 for session in sessions)


@pytest.mark.parametrize("failure_phase", ["begin", "finish"])
@pytest.mark.parametrize("cleanup_succeeds", [False, True])
def test_unconfirmed_stop_requires_scoped_cleanup_and_preserves_recorded_data(
    tmp_path, monkeypatch, failure_phase, cleanup_succeeds,
):
    sessions, kwargs, events, barrier = make_wave(tmp_path, monkeypatch, 2)
    client = sessions[0].slot.client
    client.config = SimpleNamespace(world_index=0)
    client._hard_kill = lambda: events.append(("hard_kill", 0))

    def fail_stop():
        raise GazeboWorkerCommunicationError("no stop confirmation")

    monkeypatch.setattr(client, f"{failure_phase}_stop", fail_stop)

    def scoped_cleanup(config):
        # All other in-flight stop responses must be consumed even on error.
        assert ("stop_finish", 1) in events
        events.append(("scoped_cleanup", config.world_index))
        if not cleanup_succeeds:
            raise RuntimeError("resources still present")
        client.active = False
        return {"scan_count": 2}

    monkeypatch.setattr(multi, "_cleanup_reserved_world_resources", scoped_cleanup)
    if cleanup_succeeds:
        multi._complete_episode_wave(sessions, **kwargs)
        assert all(session.slot.pending_episode_restart for session in sessions)
    else:
        with pytest.raises(RuntimeError, match="shutdown could not be confirmed"):
            multi._complete_episode_wave(sessions, **kwargs)
        assert not any(session.slot.pending_episode_restart for session in sessions)
    barrier()
    assert not any(event[0] == "begin" for event in events)


def test_failed_world_cleanup_error_carries_other_completed_episodes(tmp_path, monkeypatch):
    sessions, kwargs, _, _ = make_wave(tmp_path, monkeypatch, 2)
    failed_slot = sessions[1].slot
    failure = multi._EpisodeFailure(
        slot=failed_slot, attempt_index=19, completed_steps=0,
        discarded_transition_count=0, category="camera", error_type="Timeout",
        message="post-reset camera timeout",
    )
    monkeypatch.setattr(multi, "_prepare_sessions", lambda *a, **kw: ([], []))
    monkeypatch.setattr(multi, "_run_sessions", lambda *a, **kw: ([sessions[0]], [failure]))

    def fail_cleanup(*args):
        raise multi._WorldShutdownError("world1 still present")

    monkeypatch.setattr(multi, "_stop_failed_worlds", fail_cleanup)
    with pytest.raises(multi._WorldShutdownError) as captured:
        multi._run_wave(
            [session.slot for session in sessions], adapter=None, policy=None,
            critic=None, trainer=None, device=torch.device("cpu"), loaded_config=None,
            base_config=kwargs["base_config"], multiworld_config=multi.MultiWorldConfig(),
            ppo_config=kwargs["rollout"].config, output_directory=tmp_path, attempt_counter=[0],
        )
    assert captured.value.completed_sessions == [sessions[0]]


def test_failed_startup_world_is_closed_while_ready_world_remains_available(tmp_path):
    events = []
    clients = [FakeClient(index, events) for index in range(2)]
    for client in clients:
        client.begin_start = lambda: None
    clients[0].finish_start = lambda: {"type": "ready"}
    clients[1].finish_start = lambda: {"type": "startup_failed"}
    slots = [multi._WorldSlot(client=client) for client in clients]
    multi._start_clients_concurrently(slots, output_directory=tmp_path)
    assert slots[0].available and clients[0].active
    assert not slots[1].available and slots[1].quarantined and not clients[1].active
    assert events == [("stop_begin", 1), ("stop_finish", 1)]


@pytest.mark.parametrize("world_count", [1, 3])
@pytest.mark.parametrize("waves_per_update", [1, 2])
def test_real_main_loop_never_creates_worlds_before_ppo_or_after_final_wave(
    tmp_path, monkeypatch, world_count, waves_per_update,
):
    _exercise_main_loop(tmp_path, monkeypatch, world_count, waves_per_update)


def test_real_main_loop_records_completed_episodes_before_cleanup_abort(tmp_path, monkeypatch):
    _exercise_main_loop(tmp_path, monkeypatch, 2, 1, cleanup_abort=True)


def _exercise_main_loop(tmp_path, monkeypatch, world_count, waves_per_update, *, cleanup_abort=False):
    """Run the real scheduler and persistence with fake simulation and optimizer."""
    events = []
    clients = []
    output = tmp_path / "run"
    policy = torch.nn.Linear(1, 1)
    policy.reset = lambda: None
    config = SimpleNamespace(
        temporal_ensemble_coeff=None, n_action_steps=2, chunk_size=2,
        optimizer_lr=3e-8, optimizer_lr_backbone=3e-9,
        optimizer_weight_decay=0., use_vae=True, vision_backbone="fake",
    )
    loaded = SimpleNamespace(policy=policy, config=config, device=torch.device("cpu"))
    policy.config = loaded.config

    class Client(FakeClient):
        def __init__(self, config):
            super().__init__(config.world_index, events)
            self.config = config
            self.active = False
            clients.append(self)

        def begin_start(self):
            events.append(("initial", self.world_index))

        def finish_start(self):
            self.active = True
            return {"type": "ready"}

        def shutdown(self):
            self.active = False
            events.append(("shutdown", self.world_index))

    class Trainer:
        actor_learning_rate = 3e-8
        actor_backbone_learning_rate = 3e-9

        def __init__(self, policy, critic, *, config):
            configure_bounded_act_policy(policy)
            self.update_index = 0

    def collect(slots, *, attempt_counter, **kw):
        events.append(("collect", attempt_counter[0]))
        assert all(slot.client.active and slot.available for slot in slots)
        sessions = []
        for slot in slots:
            attempt_counter[0] += 1
            buffer = episode_buffer(slot.world_index)
            slot.successful_episode_count += 1
            sessions.append(multi._EpisodeSession(
                slot=slot, buffer=buffer, observation=None, last_info={},
                cleanup={}, home_reset={}, attempt_index=attempt_counter[0],
                completed_steps=2, episode_reward=sum(buffer.rewards),
            ))
        if cleanup_abort:
            error = multi._WorldShutdownError("failed world cleanup")
            error.completed_sessions = sessions[:1]
            raise error
        return sessions, [], 0

    def update(rollout, *, trainer, **kw):
        assert all(not client.active for client in clients)
        assert len(rollout) == world_count * 2 * waves_per_update
        # The actual recorder must have written full episodes before PPO.
        assert len(list((output / "episode_steps").glob("*.jsonl"))) >= world_count * waves_per_update
        events.append(("ppo", len(rollout)))
        trainer.update_index += 1
        with torch.no_grad():
            policy.weight.add_(1.)
        return {"update_index": trainer.update_index}

    def export(*args, **kwargs):
        assert all(not client.active for client in clients)
        events.append(("export",))
        return output / "final", output / "state.pt"

    monkeypatch.setattr(multi, "GazeboWorkerClient", Client)
    monkeypatch.setattr(multi, "ActPpoTrainer", Trainer)
    monkeypatch.setattr(multi, "ActPpoValueNetwork", lambda config: torch.nn.Identity())
    monkeypatch.setattr(multi, "ActPolicyAdapter", lambda *a, **kw: None)
    monkeypatch.setattr(multi, "load_act_checkpoint", lambda *a, **kw: loaded)
    monkeypatch.setattr(multi, "_run_wave", collect)
    monkeypatch.setattr(multi, "_perform_update", update)
    monkeypatch.setattr(multi, "_export_final", export)
    base = ActPpoGazeboConfig(
        act_checkpoint=tmp_path / "fake", output_directory=output,
        device="cpu", tensorboard_enabled=False, episode_count=world_count * 2,
        max_steps_per_episode=2, rollout_steps=world_count * 2 * waves_per_update,
        target_delay_s=0., final_target_absent_s=0., critic_image_height=8, critic_image_width=8,
    )
    worlds = multi.MultiWorldConfig(world_count=world_count, world_restart_max_attempts=0,
                                   preflight_cleanup_enabled=False)
    if cleanup_abort:
        with pytest.raises(multi._WorldShutdownError, match="failed world cleanup"):
            multi.run_act_ppo_multiworld_gazebo(base, worlds)
        rows = read_jsonl(output / "episode_steps" / "episode_000000.jsonl")
        assert len(rows) == 2 and rows[-1]["episode_end"]
        assert not any(event[0] in {"begin", "ppo", "export"} for event in events)
    else:
        summary = multi.run_act_ppo_multiworld_gazebo(base, worlds)
        assert summary["episode_count"] == world_count * 2
        assert summary["ppo_update_count"] == 2 // waves_per_update
        assert summary["total_collected_steps"] == world_count * 4
        assert len([event for event in events if event[0] == "begin"]) == world_count
        assert len([event for event in events if event[0] == "stop_finish"]) == world_count * 2
        first_restart = next(i for i, event in enumerate(events) if event[0] == "begin")
        if waves_per_update == 1:
            first_update = next(i for i, event in enumerate(events) if event[0] == "ppo")
            assert first_restart > first_update
        last_collect = max(i for i, event in enumerate(events) if event[0] == "collect")
        assert not any(event[0] == "begin" for event in events[last_collect + 1:])
    assert all(not client.active for client in clients)
