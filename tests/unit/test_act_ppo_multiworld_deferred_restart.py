from pathlib import Path

import pytest

import so101_capstone.training.act_ppo_multiworld_gazebo as module
from so101_capstone.training.act_ppo_multiworld_gazebo import (
    MultiWorldConfig,
    _WorldSlot,
    _deferred_world_restart_enabled,
    _recover_slots_concurrently,
    _worker_restart_attempt_limit,
)


class FakeClient:
    def __init__(self, world_index, responses):
        self.world_index = world_index
        self.responses = list(responses)
        self.begin_calls = 0
        self.finish_calls = 0
        self.stop_calls = 0

    def begin_stop(self):
        self.stop_calls += 1

    def finish_stop(self):
        return {"type": "world_stopped", "world_index": self.world_index}

    def begin_restart(self):
        self.begin_calls += 1
        return "runtime_restart"

    def finish_restart(self, mode):
        assert mode == "runtime_restart"
        self.finish_calls += 1
        return self.responses.pop(0)


def test_zero_means_deferred_not_unlimited():
    config = MultiWorldConfig(
        world_restart_max_attempts=0,
        world_deferred_restart_max_attempts=3,
    )
    assert _deferred_world_restart_enabled(config) is True
    assert _worker_restart_attempt_limit(config) == 3


def test_positive_limit_preserves_immediate_mode():
    config = MultiWorldConfig(
        world_restart_max_attempts=2,
        world_deferred_restart_max_attempts=3,
    )
    assert _deferred_world_restart_enabled(config) is False
    assert _worker_restart_attempt_limit(config) == 2


@pytest.mark.parametrize("value", [-1, -2])
def test_negative_immediate_restart_limit_is_rejected(value):
    with pytest.raises(ValueError, match="non-negative"):
        MultiWorldConfig(world_restart_max_attempts=value)


@pytest.mark.parametrize("value", [0, -1])
def test_deferred_restart_limit_must_be_positive(value):
    with pytest.raises(ValueError, match="positive"):
        MultiWorldConfig(world_deferred_restart_max_attempts=value)


def test_failed_boundary_rebuild_stays_pending_not_quarantined(tmp_path, monkeypatch):
    client = FakeClient(1, [{"type": "restart_failed", "world_index": 1}])
    slot = _WorldSlot(client=client)
    slot.available = False
    slot.quarantined = False
    events = []
    monkeypatch.setattr(
        module,
        "_log_world_event",
        lambda _directory, payload: events.append(payload),
    )

    _recover_slots_concurrently(
        [slot],
        output_directory=Path(tmp_path),
        reset_episode_failure_budget=False,
        quarantine_on_failure=False,
    )

    assert client.begin_calls == 1
    assert client.finish_calls == 1
    assert slot.available is False
    assert slot.quarantined is False
    assert slot.restart_count == 0
    failed = next(event for event in events if event["event"] == "restart_failed")
    assert failed["quarantine_on_failure"] is False
    assert client.stop_calls == 1
    assert events[-1]["event"] == "world_stopped"


def test_successful_boundary_rebuild_rejoins_next_wave(tmp_path, monkeypatch):
    client = FakeClient(2, [{"type": "ready", "world_index": 2}])
    slot = _WorldSlot(client=client)
    slot.available = False
    slot.quarantined = False
    slot.consecutive_episode_failures = 1
    monkeypatch.setattr(module, "_log_world_event", lambda *_args: None)

    _recover_slots_concurrently(
        [slot],
        output_directory=Path(tmp_path),
        reset_episode_failure_budget=False,
        quarantine_on_failure=False,
    )

    assert slot.available is True
    assert slot.quarantined is False
    assert slot.restart_count == 1
    assert slot.consecutive_episode_failures == 1
