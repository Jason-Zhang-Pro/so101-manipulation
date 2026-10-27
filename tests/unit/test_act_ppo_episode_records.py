import json
import os
import stat

import numpy as np
import pytest

from so101_capstone.training.act_ppo import (
    ActPpoActorObservation,
    ActPpoConfig,
    ActPpoRolloutBuffer,
)
import so101_capstone.training.act_ppo_episode_records as records
from so101_capstone.training.residual_ppo import CompressedResidualObservation


def complete_episode():
    buffer = ActPpoRolloutBuffer(ActPpoConfig(
        critic_image_height=8, critic_image_width=8,
        critic_task_context_dim=2, critic_policy_context_dim=2,
    ))
    reference = buffer.add_actor_observation(ActPpoActorObservation(
        top_image=np.full((3, 8, 8), 217, np.uint8),
        gripper_image=np.full((3, 8, 8), 219, np.uint8),
        policy_state=np.arange(6, dtype=np.float32),
    ))
    for index in range(3):
        buffer.add(
            CompressedResidualObservation(
                top_image=np.full((3, 8, 8), 221, np.uint8),
                gripper_image=np.full((3, 8, 8), 223, np.uint8),
                joint_positions=np.full(6, index, np.float32),
                joint_velocities=np.full(6, index / 10, np.float32),
            ),
            actor_observation_index=reference, action_index=index,
            policy_action=np.full(6, index / 10),
            policy_action_latent=np.full(6, index / 20),
            log_probability=-index / 3, value=index / 4,
            reward=index + .5, next_value=(index + 1) / 4,
            terminated=False, episode_end=index == 2,
            critic_task_context=np.asarray([index, 1.]),
            critic_policy_context=np.asarray([index, .5]),
        )
    return buffer


def test_complete_episode_persists_all_nonimage_transitions(tmp_path):
    buffer = complete_episode()
    path = tmp_path / "episode_000001.steps.jsonl"
    result = records.persist_episode_steps(path, buffer, 3)

    assert result == {"path": str(path), "step_count": 3, "images_included": False}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step_index"] for row in rows] == [1, 2, 3]
    assert [row["rewards"] for row in rows] == [.5, 1.5, 2.5]
    assert [row["episode_end"] for row in rows] == [False, False, True]
    assert [row["next_values"] for row in rows] == [.25, .5, .75]
    for index, row in enumerate(rows):
        assert row["sampling_schema_version"] == 4
        assert row["critic_policy_context_schema_version"] == 2
        assert row["policy_action_means"] is None
        assert row["policy_action_log_stds"] is None
        for name, values in vars(buffer).items():
            if isinstance(values, list) and name not in {
                "actor_observations", "top_images", "gripper_images",
            }:
                np.testing.assert_array_equal(row[name], values[index])
        assert row["actor_policy_state"] == list(range(6))
        assert "actor_observations" not in row
        assert "top_images" not in row and "gripper_images" not in row
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("completed_steps", [0, -1, True, 3.0, 2, 4])
def test_rejects_invalid_or_mismatched_step_count(tmp_path, completed_steps):
    path = tmp_path / "steps.jsonl"
    with pytest.raises(ValueError):
        records.persist_episode_steps(path, complete_episode(), completed_steps)
    assert not path.exists()


@pytest.mark.parametrize("field", ["positions", "policy_actions", "top_images", "values", "actor_refs"])
def test_rejects_missing_transition_in_any_field(tmp_path, field):
    buffer = complete_episode()
    getattr(buffer, field).pop()
    with pytest.raises(ValueError, match=field):
        records.persist_episode_steps(tmp_path / "steps.jsonl", buffer, 3)


@pytest.mark.parametrize("reference", [-1, 1, True, .5])
def test_rejects_invalid_actor_reference(reference):
    buffer = complete_episode()
    buffer.actor_refs[1] = reference
    with pytest.raises(ValueError, match="Invalid actor reference"):
        records.validate_complete_episode(buffer, 3)


@pytest.mark.parametrize("field,index,value", [
    ("episode_end", 2, False), ("episode_end", 0, True), ("terminated", 1, True),
])
def test_rejects_missing_or_premature_episode_end(field, index, value):
    buffer = complete_episode()
    getattr(buffer, field)[index] = value
    with pytest.raises(ValueError):
        records.validate_complete_episode(buffer, 3)


def test_rejects_foreign_buffer():
    with pytest.raises(TypeError, match="ActPpoRolloutBuffer"):
        records.validate_complete_episode(object(), 3)


def test_partial_json_write_preserves_existing_record_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "steps.jsonl"
    path.write_text("previous record\n")
    original_dump = json.dump

    def interrupted_dump(record, stream, **kwargs):
        if record["step_index"] == 2:
            raise OSError("disk full")
        return original_dump(record, stream, **kwargs)

    monkeypatch.setattr(records.json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="disk full"):
        records.persist_episode_steps(path, complete_episode(), 3)
    assert path.read_text() == "previous record\n"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("fail_call", [1, 2])
def test_file_or_directory_sync_failure_cannot_report_success(tmp_path, monkeypatch, fail_call):
    path = tmp_path / "steps.jsonl"
    path.write_text("previous record\n")
    original_sync = os.fsync
    calls = []

    def failing_sync(descriptor):
        calls.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        if len(calls) == fail_call:
            raise OSError("sync failed")
        return original_sync(descriptor)

    monkeypatch.setattr(records.os, "fsync", failing_sync)
    with pytest.raises(OSError, match="sync failed"):
        records.persist_episode_steps(path, complete_episode(), 3)
    assert calls == ([False] if fail_call == 1 else [False, True])
    if fail_call == 1:
        assert path.read_text() == "previous record\n"
    assert list(tmp_path.iterdir()) == [path]


def test_sync_record_flushes_file_and_parent_directory(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    path.write_text("{}\n")
    calls = []
    original_sync = os.fsync

    def observe_sync(descriptor):
        calls.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        return original_sync(descriptor)

    monkeypatch.setattr(records.os, "fsync", observe_sync)
    records.sync_record(path)
    assert calls == [False, True]


def test_sync_record_propagates_missing_or_failed_file(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        records.sync_record(tmp_path / "missing.json")
    path = tmp_path / "summary.json"
    path.write_text("{}\n")

    def failing_sync(_descriptor):
        raise OSError("sync failed")

    monkeypatch.setattr(records.os, "fsync", failing_sync)
    with pytest.raises(OSError, match="sync failed"):
        records.sync_record(path)
