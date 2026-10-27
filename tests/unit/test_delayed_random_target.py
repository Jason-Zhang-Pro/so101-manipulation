import json
from pathlib import Path

import pytest

from so101_capstone.policies.delayed_random_target import (
    DelayedRandomTargetEpisodeController,
    DelayedRandomTargetRuntimeProxy,
    TargetSnapshot,
    TriggerResult,
    _parse_target_snapshots,
)


class FakeBackend:
    def __init__(self):
        self.events = []
        self.current_targets = ()
        self.spawn_calls = 0
        self.delete_calls = 0

    def poll(self, *, timeout_s=0.0):
        self.events.append(("poll", timeout_s))

    def delete_all(self):
        self.delete_calls += 1
        self.events.append("delete_all")
        self.current_targets = ()
        return TriggerResult(True, "deleted")

    def wait_until_empty(self):
        self.events.append("wait_until_empty")

    def spawn_random(self):
        self.spawn_calls += 1
        self.events.append("spawn_random")
        return TriggerResult(True, "spawned")


def test_fixed_three_second_delay_requests_at_step_90(
    tmp_path,
):
    backend = FakeBackend()
    controller = DelayedRandomTargetEpisodeController(
        backend,
        target_delay_s=3.0,
        policy_hz=30.0,
        output_directory=tmp_path,
    )

    controller.clear_before_episode(0)
    controller.start_episode(0)

    for step in range(1, 90):
        controller.after_step(step)

    assert backend.spawn_calls == 0

    controller.after_step(90)

    assert backend.spawn_calls == 1
    record = controller.current_record
    assert record is not None
    assert record.target_delay_steps == 90
    assert record.target_spawn_request_step == 90
    assert record.target_spawn_service_success is True
    assert record.target_spawn_confirmed is True
    assert record.target_spawn_confirmed_step == 90
    assert record.target_name is None
    assert record.target_initial_position is None

    manifest = (
        tmp_path
        / "episode_000000_target.json"
    )
    assert controller.manifest_paths == (
        manifest.resolve(),
    )

    payload = json.loads(
        manifest.read_text(encoding="utf-8")
    )
    assert payload["target_delay_s"] == 3.0
    assert payload["target_delay_steps"] == 90
    assert payload["target_spawn_request_step"] == 90
    assert payload["target_spawn_service_success"] is True
    assert payload["target_spawn_confirmed"] is True
    assert payload["target_spawn_confirmed_step"] == 90
    assert payload["target_name"] is None
    assert payload["target_initial_position"] is None
    assert "episode_start_sim_time_ns" not in payload
    assert "target_spawn_request_sim_time_ns" not in payload
    assert "target_spawn_confirmed_sim_time_ns" not in payload



def test_service_success_confirms_without_states_topic():
    backend = FakeBackend()
    controller = DelayedRandomTargetEpisodeController(
        backend,
        target_delay_s=0.0,
        policy_hz=30.0,
    )

    controller.start_episode(0)
    record = controller.current_record

    assert record is not None
    assert record.target_spawn_service_success is True
    assert record.target_spawn_confirmed is True
    assert record.target_spawn_confirmed_step == 0

    backend.current_targets = (
        TargetSnapshot(
            name="cube",
            position=(0.1, 0.2, 0.768),
        ),
    )
    controller.after_step(1)

    assert record.target_spawn_confirmed is True
    assert record.target_spawn_confirmed_step == 0



def test_runtime_proxy_starts_episode_before_delay_steps():
    events = []

    class Runtime:
        def prepare_episode(self):
            events.append("runtime.prepare")
            return "frame-0"

        def step(self, action):
            events.append(("runtime.step", action))
            return f"frame-{action}"

    class Controller:
        def clear_before_episode(self, episode_index):
            events.append(("target.clear", episode_index))

        def start_episode(self, episode_index):
            events.append(("target.start", episode_index))

        def after_step(self, completed_step_count):
            events.append(("target.after", completed_step_count))

    proxy = DelayedRandomTargetRuntimeProxy(
        Runtime(),
        Controller(),
    )

    assert proxy.prepare_episode() == "frame-0"
    assert proxy.step("a") == "frame-a"
    assert proxy.step("b") == "frame-b"

    assert events == [
        ("target.clear", 0),
        "runtime.prepare",
        ("target.start", 0),
        ("runtime.step", "a"),
        ("target.after", 1),
        ("runtime.step", "b"),
        ("target.after", 2),
    ]


@pytest.mark.parametrize(
    ("payload", "expected_name", "expected_position"),
    [
        (
            {
                "targets": [
                    {
                        "name": "cube_a",
                        "position": {
                            "x": 0.1,
                            "y": 0.2,
                            "z": 0.3,
                        },
                    }
                ]
            },
            "cube_a",
            (0.1, 0.2, 0.3),
        ),
        (
            {
                "cube_b": {
                    "pose": {
                        "position": [
                            -0.1,
                            0.4,
                            0.768,
                        ]
                    }
                }
            },
            "cube_b",
            (-0.1, 0.4, 0.768),
        ),
    ],
)
def test_parse_target_snapshots(
    payload,
    expected_name,
    expected_position,
):
    snapshots = _parse_target_snapshots(payload)

    assert len(snapshots) == 1
    assert snapshots[0].name == expected_name
    assert snapshots[0].position == pytest.approx(
        expected_position
    )


def test_delay_must_match_integer_policy_steps():
    with pytest.raises(
        ValueError,
        match="integer number",
    ):
        DelayedRandomTargetEpisodeController(
            FakeBackend(),
            target_delay_s=0.01,
            policy_hz=30.0,
        )

def test_confirmed_target_does_not_wait_for_states_topic():
    backend = FakeBackend()
    controller = DelayedRandomTargetEpisodeController(
        backend,
        target_delay_s=0.0,
        policy_hz=30.0,
    )

    controller.start_episode(0)
    backend.events.clear()
    controller.after_step(1)

    poll_events = [
        event
        for event in backend.events
        if isinstance(event, tuple)
        and event[0] == "poll"
    ]

    assert poll_events
    assert all(
        timeout_s == 0.0
        for _, timeout_s in poll_events
    )


def test_successful_spawn_service_confirms_target_immediately():
    backend = FakeBackend()

    def spawn_random():
        backend.spawn_calls += 1
        return TriggerResult(
            True,
            (
                '{"name": "target_cube_0001", '
                '"entity_id": 106, '
                '"requested_position": {'
                '"x": 0.095, "y": -0.357, "z": 0.768}}'
            ),
        )

    backend.spawn_random = spawn_random

    controller = DelayedRandomTargetEpisodeController(
        backend,
        target_delay_s=3.0,
        policy_hz=30.0,
    )
    controller.start_episode(0)

    for step in range(1, 91):
        controller.after_step(step)

    record = controller.current_record
    assert record is not None
    assert record.target_spawn_service_success is True
    assert record.target_spawn_confirmed is True
    assert record.target_spawn_confirmed_step == 90
    assert record.target_name == "target_cube_0001"
    assert record.target_initial_position == pytest.approx(
        (0.095, -0.357, 0.768)
    )

def test_clear_before_episode_does_not_wait_for_states_topic():
    class Backend(FakeBackend):
        def wait_until_empty(self):
            raise AssertionError(
                "clear_before_episode must not wait for "
                "the target states topic."
            )

    backend = Backend()
    controller = DelayedRandomTargetEpisodeController(
        backend,
        target_delay_s=3.0,
        policy_hz=30.0,
    )

    controller.clear_before_episode(0)

    assert backend.delete_calls == 1
    assert backend.events == ["delete_all"]


def test_clear_before_episode_rejects_delete_failure():
    class Backend(FakeBackend):
        def delete_all(self):
            self.delete_calls += 1
            self.events.append("delete_all")
            return TriggerResult(False, "delete failed")

        def wait_until_empty(self):
            raise AssertionError(
                "wait_until_empty must not be called."
            )

    backend = Backend()
    controller = DelayedRandomTargetEpisodeController(
        backend,
        target_delay_s=3.0,
        policy_hz=30.0,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to delete target cubes",
    ):
        controller.clear_before_episode(0)

    assert backend.delete_calls == 1
    assert backend.events == ["delete_all"]
