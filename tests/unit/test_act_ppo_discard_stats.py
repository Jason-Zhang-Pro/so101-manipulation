import json
import os
from pathlib import Path

import pytest

import so101_capstone.training.act_ppo_discard_stats as accounting


def failure(*, world=0, attempt=1, completed=10, buffered=9, crash=False,
            category="camera", unconfirmed=1):
    return {
        "world_index": world,
        "episode_attempt_index": attempt,
        "failed_step": completed,
        "discarded_transition_count": buffered,
        "failure_category": category,
        "gazebo_crash": {"confirmed": crash},
        "unconfirmed_step_count": unconfirmed,
    }


def test_initial_summary_is_saved_beside_training_outputs_with_zero_counts(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path, [0, 1, 2])
    saved = json.loads((tmp_path / "discarded_steps.json").read_text())
    assert saved == stats.snapshot()
    assert saved["schema_version"] == 1
    assert saved["updated_at_utc"].endswith("+00:00")
    assert saved["failure_episode_count"] == 0
    assert saved["gazebo_crash_episode_count"] == 0
    assert saved["discarded_environment_steps"] == 0
    assert saved["discarded_transition_count"] == 0
    assert set(saved["by_world"]) == {"0", "1", "2"}
    assert saved["failure_events"] == []
    assert list(tmp_path.iterdir()) == [tmp_path / "discarded_steps.json"]


def test_failed_episode_counts_separate_crashes_buffered_and_unconfirmed_steps(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path, [0, 1])
    stats.record_failure(failure(crash=True, category="worker"))
    stats.record_failure(failure(world=1, attempt=2, completed=20, buffered=20, unconfirmed=0))
    result = stats.record_failure(failure(attempt=3, completed=4, buffered=4, category="worker"))
    assert result["discarded_environment_steps"] == 34
    assert result["discarded_transition_count"] == 33
    assert result["failure_discarded_transition_count"] == 33
    assert result["gazebo_crash_discarded_environment_steps"] == 10
    assert result["gazebo_crash_discarded_transition_count"] == 9
    assert result["gazebo_crash_episode_count"] == 1
    assert result["failure_episode_count"] == 3
    assert result["unconfirmed_step_count"] == 2
    assert result["gazebo_crash_unconfirmed_step_count"] == 1
    assert result["by_world"]["0"]["discarded_environment_steps"] == 14
    assert result["by_world"]["1"]["discarded_transition_count"] == 20
    assert result["by_failure_category"]["worker"]["discarded_transition_count"] == 13
    assert result["by_failure_category"]["camera"]["gazebo_crash_episode_count"] == 0
    event = result["failure_events"][0]
    assert event["failure_record_path"] == "episode_failures/attempt_000001_world_00.json"
    assert event["discarded_environment_steps"] == 10
    assert event["discarded_transition_count"] == 9
    assert event["unconfirmed_step_count"] == 1
    assert json.loads(stats.path.read_text()) == result


@pytest.mark.parametrize("diagnostics", [None, {}, {"confirmed": False}, {"confirmed": "true"}, {"confirmed": 1}])
def test_unconfirmed_crash_evidence_is_never_classified_as_a_crash(tmp_path, diagnostics):
    stats = accounting.DiscardedStepStats(tmp_path)
    record = failure(category="gazebo_crash")
    record["gazebo_crash"] = diagnostics
    result = stats.record_failure(record)
    assert result["failure_episode_count"] == 1
    assert result["discarded_environment_steps"] == 10
    assert result["gazebo_crash_episode_count"] == 0
    assert result["gazebo_crash_discarded_environment_steps"] == 0


def test_crash_before_first_step_records_crash_without_inventing_executed_steps(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path)
    result = stats.record_failure(failure(completed=0, buffered=0, crash=True, unconfirmed=1))
    assert result["gazebo_crash_episode_count"] == 1
    assert result["failure_episode_count"] == 1
    assert result["gazebo_crash_discarded_environment_steps"] == 0
    assert result["discarded_transition_count"] == 0
    assert result["unconfirmed_step_count"] == 1


def test_duplicate_event_and_reloaded_summary_are_idempotent(tmp_path):
    record = failure(crash=True)
    stats = accounting.DiscardedStepStats(tmp_path)
    saved = stats.record_failure(record)
    assert stats.record_failure(record) == saved
    restarted = accounting.DiscardedStepStats(tmp_path)
    assert restarted.record_failure(record) == saved
    result = restarted.record_failure(failure(attempt=2))
    assert result["failure_episode_count"] == 2
    assert result["discarded_transition_count"] == 18


def test_conflicting_duplicate_is_rejected_without_changing_record(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path)
    saved = stats.record_failure(failure())
    with pytest.raises(ValueError, match="Conflicting failure event"):
        stats.record_failure(failure(completed=20))
    assert stats.snapshot() == saved
    assert json.loads(stats.path.read_text()) == saved


def test_final_partial_rollout_has_separate_reason_and_no_crash_attribution(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path)
    stats.record_failure(failure(crash=True))
    result = stats.record_final_partial(100)
    assert result["discarded_transition_count"] == 109
    assert result["failure_discarded_transition_count"] == 9
    assert result["final_partial_rollout_discarded_transition_count"] == 100
    assert result["final_partial_rollout_reason"] == "training_end_partial_rollout"
    assert result["discarded_environment_steps"] == 10
    assert result["gazebo_crash_discarded_transition_count"] == 9
    assert result["gazebo_crash_discarded_environment_steps"] == 10
    assert result["failure_episode_count"] == 1
    assert result["by_world"]["0"]["discarded_transition_count"] == 9
    assert len(result["failure_events"]) == 1
    assert stats.record_final_partial(100) == result
    result = stats.record_final_partial(101)
    assert result["discarded_transition_count"] == 110
    assert result["gazebo_crash_discarded_transition_count"] == 9


def test_saved_episodes_and_planned_world_rebuilds_without_failure_keep_zero(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path, [0, 1])
    result = stats.record_final_partial(0)
    assert result["discarded_environment_steps"] == 0
    assert result["discarded_transition_count"] == 0
    assert result["failure_episode_count"] == 0
    assert result["gazebo_crash_episode_count"] == 0


def test_snapshot_does_not_expose_mutable_internal_state(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path)
    result = stats.record_failure(failure())
    result["by_world"]["0"]["discarded_environment_steps"] = 999
    result["failure_events"].clear()
    assert stats.snapshot()["by_world"]["0"]["discarded_environment_steps"] == 10
    assert len(stats.snapshot()["failure_events"]) == 1


@pytest.mark.parametrize("field,value", [
    ("world_index", -1), ("episode_attempt_index", True), ("failed_step", 1.5),
    ("failed_step", -1), ("discarded_transition_count", -1),
    ("discarded_transition_count", 11), ("unconfirmed_step_count", 2),
    ("unconfirmed_step_count", True), ("failure_category", ""), ("gazebo_crash", "crash"),
])
def test_invalid_failure_record_cannot_change_counts(tmp_path, field, value):
    stats = accounting.DiscardedStepStats(tmp_path)
    before = stats.snapshot()
    record = failure()
    record[field] = value
    with pytest.raises(ValueError):
        stats.record_failure(record)
    assert stats.snapshot() == before
    assert json.loads(stats.path.read_text()) == before


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_final_rollout_count_is_rejected(tmp_path, count):
    stats = accounting.DiscardedStepStats(tmp_path)
    with pytest.raises(ValueError):
        stats.record_final_partial(count)
    assert stats.snapshot()["discarded_transition_count"] == 0


def test_partial_json_failure_preserves_previous_file_and_retry_counts_once(tmp_path, monkeypatch):
    stats = accounting.DiscardedStepStats(tmp_path)
    before = stats.snapshot()
    original_dump = json.dump

    def disk_full(payload, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("disk full")

    monkeypatch.setattr(accounting.json, "dump", disk_full)
    with pytest.raises(OSError, match="disk full"):
        stats.record_failure(failure())
    assert stats.snapshot() == before
    assert json.loads(stats.path.read_text()) == before
    assert list(tmp_path.iterdir()) == [stats.path]
    monkeypatch.setattr(accounting.json, "dump", original_dump)
    assert stats.record_failure(failure())["failure_episode_count"] == 1


def test_replace_failure_preserves_previous_file_and_cleans_temporary(tmp_path, monkeypatch):
    stats = accounting.DiscardedStepStats(tmp_path)
    before = stats.snapshot()

    def rejected_replace(self, target):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", rejected_replace)
    with pytest.raises(OSError, match="replace failed"):
        stats.record_failure(failure())
    assert stats.snapshot() == before
    assert json.loads(stats.path.read_text()) == before
    assert list(tmp_path.iterdir()) == [stats.path]


@pytest.mark.parametrize("failed_sync", [1, 2, 3])
def test_sync_failures_propagate_without_advancing_memory_or_deduplication(tmp_path, monkeypatch, failed_sync):
    stats = accounting.DiscardedStepStats(tmp_path)
    before = stats.snapshot()
    original_sync = os.fsync
    calls = 0

    def failing_sync(descriptor):
        nonlocal calls
        calls += 1
        if calls == failed_sync:
            raise OSError("sync failed")
        return original_sync(descriptor)

    monkeypatch.setattr(accounting.os, "fsync", failing_sync)
    with pytest.raises(OSError, match="sync failed"):
        stats.record_failure(failure(crash=True))
    assert stats.snapshot() == before
    assert list(tmp_path.iterdir()) == [stats.path]
    if failed_sync == 1:
        assert json.loads(stats.path.read_text()) == before
    # A post-replacement fsync failure may leave the new file visible, but is
    # still an error. Retrying must persist the event once, never double-count.
    monkeypatch.setattr(accounting.os, "fsync", original_sync)
    result = stats.record_failure(failure(crash=True))
    assert result["failure_episode_count"] == 1
    assert result["gazebo_crash_episode_count"] == 1
    assert json.loads(stats.path.read_text()) == result


def test_aborted_rollout_counts_confirmed_unused_transitions_without_crash_attribution(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path)
    stats.record_failure(failure(crash=True))
    result = stats.record_aborted_rollout(80)
    assert result["aborted_rollout_discarded_transition_count"] == 80
    assert result["aborted_rollout_reason"] == "training_aborted_partial_rollout"
    assert result["discarded_transition_count"] == 89
    assert result["failure_discarded_transition_count"] == 9
    assert result["final_partial_rollout_discarded_transition_count"] == 0
    assert result["interrupted_update_transition_count"] == 0
    assert result["gazebo_crash_discarded_transition_count"] == 9
    assert result["gazebo_crash_discarded_environment_steps"] == 10
    assert result["discarded_environment_steps"] == 10
    assert result["by_world"]["0"]["discarded_transition_count"] == 9
    assert result["by_failure_category"]["camera"]["discarded_transition_count"] == 9
    assert len(result["failure_events"]) == 1
    assert stats.record_aborted_rollout(80) == result
    assert json.loads(stats.path.read_text()) == result


def test_interrupted_update_is_separate_from_confirmed_discard_and_crash_counts(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path)
    stats.record_failure(failure(crash=True))
    result = stats.record_aborted_rollout(1800, update_in_progress=True)
    assert result["interrupted_update_transition_count"] == 1800
    assert result["interrupted_update_reason"] == "training_aborted_during_ppo_update"
    assert result["aborted_rollout_discarded_transition_count"] == 0
    assert result["discarded_transition_count"] == 9
    assert result["gazebo_crash_discarded_transition_count"] == 9
    assert result["gazebo_crash_episode_count"] == 1
    assert result["failure_episode_count"] == 1
    assert stats.record_aborted_rollout(1800, update_in_progress=True) == result
    assert json.loads(stats.path.read_text()) == result


@pytest.mark.parametrize("order", [
    ("failure", "final", "abort"),
    ("failure", "abort", "final"),
    ("abort", "failure", "final"),
    ("abort", "final", "failure"),
    ("final", "failure", "abort"),
    ("final", "abort", "failure"),
])
def test_discard_total_keeps_all_reason_subtotals_regardless_of_record_order(tmp_path, order):
    stats = accounting.DiscardedStepStats(tmp_path)
    actions = {
        "failure": lambda: stats.record_failure(failure(crash=True)),
        "final": lambda: stats.record_final_partial(100),
        "abort": lambda: stats.record_aborted_rollout(80),
    }
    for reason in order:
        actions[reason]()
    result = stats.record_aborted_rollout(1800, update_in_progress=True)
    assert result["discarded_transition_count"] == 189
    assert result["failure_discarded_transition_count"] == 9
    assert result["final_partial_rollout_discarded_transition_count"] == 100
    assert result["aborted_rollout_discarded_transition_count"] == 80
    assert result["interrupted_update_transition_count"] == 1800
    assert stats.record_aborted_rollout(81)["discarded_transition_count"] == 190
    assert stats.record_final_partial(101)["discarded_transition_count"] == 191


def test_old_schema_one_adds_zero_abort_fields_without_losing_saved_failures(tmp_path):
    stats = accounting.DiscardedStepStats(tmp_path)
    previous = stats.record_failure(failure(crash=True))
    for key in (
        "aborted_rollout_discarded_transition_count", "aborted_rollout_reason",
        "interrupted_update_transition_count", "interrupted_update_reason",
    ):
        previous.pop(key)
    stats.path.write_text(json.dumps(previous))
    restarted = accounting.DiscardedStepStats(tmp_path)
    loaded = restarted.snapshot()
    assert loaded["aborted_rollout_discarded_transition_count"] == 0
    assert loaded["interrupted_update_transition_count"] == 0
    assert loaded["discarded_transition_count"] == 9
    assert restarted.record_failure(failure(crash=True))["failure_episode_count"] == 1
    assert json.loads(stats.path.read_text()) == loaded
    result = restarted.record_aborted_rollout(80)
    assert result["discarded_transition_count"] == 89
    assert result["gazebo_crash_discarded_transition_count"] == 9


@pytest.mark.parametrize("count", [-1, True, 1.5])
@pytest.mark.parametrize("update_in_progress", [False, True])
def test_invalid_aborted_count_cannot_change_stats(tmp_path, count, update_in_progress):
    stats = accounting.DiscardedStepStats(tmp_path)
    before = stats.snapshot()
    with pytest.raises(ValueError):
        stats.record_aborted_rollout(count, update_in_progress=update_in_progress)
    assert stats.snapshot() == before


def test_aborted_write_failure_does_not_claim_persisted_counts(tmp_path, monkeypatch):
    stats = accounting.DiscardedStepStats(tmp_path)
    before = stats.snapshot()

    def failed_sync(path):
        raise OSError("sync failed")

    monkeypatch.setattr(accounting, "sync_record", failed_sync)
    with pytest.raises(OSError, match="sync failed"):
        stats.record_aborted_rollout(80)
    assert stats.snapshot() == before
