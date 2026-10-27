from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from so101_capstone.policies.act_gazebo_multi_episode import (
    ActGazeboMultiEpisodeResult,
    run_act_gazebo_multi_episode,
)


class FakeRuntime:
    def __init__(self, events):
        self.events = events
        self.enter_calls = 0
        self.exit_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        self.events.append("runtime.enter")
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ):
        self.exit_calls += 1
        self.events.append("runtime.exit")
        return False


class FakeAdapter:
    def __init__(
        self,
        policy,
        *,
        device,
        max_joint_delta_rad,
    ):
        self.policy = policy
        self.device = device
        self.max_joint_delta_rad = (
            max_joint_delta_rad
        )


class FakeSink:
    def __init__(
        self,
        output_path,
        *,
        metadata,
    ):
        self.output_path = Path(
            output_path
        ).resolve()
        self.metadata = metadata
        self.written_path = self.output_path


class FakeMultiRunner:
    def __init__(
        self,
        runtime,
        policy,
        captured,
        events,
    ):
        self.runtime = runtime
        self.policy = policy
        self.captured = captured
        self.events = events

    def run(
        self,
        *,
        episode_count,
        max_steps,
        transition_sink_factory,
    ):
        self.events.append("runner.run")
        self.captured["episode_count"] = (
            episode_count
        )
        self.captured["max_steps"] = max_steps
        self.captured[
            "transition_sink_factory"
        ] = transition_sink_factory

        episodes = []

        for episode_index in range(
            episode_count
        ):
            if transition_sink_factory is not None:
                transition_sink_factory(
                    episode_index
                )

            episodes.append(
                SimpleNamespace(
                    step_count=max_steps,
                    total_reward=float(
                        episode_index + 1
                    ),
                    terminated=False,
                    truncated=False,
                    reached_max_steps=True,
                )
            )

        return SimpleNamespace(
            episodes=tuple(episodes),
            episode_count=episode_count,
            total_step_count=(
                episode_count * max_steps
            ),
            total_reward=float(
                sum(
                    range(
                        1,
                        episode_count + 1,
                    )
                )
            ),
        )


@pytest.mark.parametrize("temporal", [False, True])
def test_fixed_scene_list_retries_same_scene_and_does_not_spawn_idle_completed(tmp_path, temporal):
    import json
    import numpy as np
    from so101_capstone.environments import PolicyObservation, parse_policy_action
    from so101_capstone.environments.simulation_step_coordinator import SimulationStepCoordinatorError

    def observation():
        return PolicyObservation(joint_positions=np.zeros(6), joint_velocities=np.zeros(6),
            top_image=np.zeros((4, 6, 3), dtype=np.uint8), gripper_image=np.zeros((4, 6, 3), dtype=np.uint8))

    class Runtime(FakeRuntime):
        attempt = 0
        configured = []
        spawned = []
        def configure_task(self, mode, **kwargs):
            self.mode = mode
            self.configured.append((mode, kwargs))
        def delete_all_targets(self):
            return SimpleNamespace(success=True)
        def prepare_episode(self):
            self.attempt += 1
            self.steps = 0
            return SimpleNamespace(observation=observation(), info={"task_mode": self.mode})
        def spawn_target(self, position, **kwargs):
            self.spawned.append((self.attempt, self.steps, position, kwargs))
            return SimpleNamespace(success=True)
        def spawn_random_target(self):
            pytest.fail("Explicit evaluation must not request a random target")
        def step(self, action):
            self.steps += 1
            if self.attempt == 1 and self.steps == 2:
                raise SimulationStepCoordinatorError("camera failed")
            return SimpleNamespace(observation=observation(), reward=0., terminated=self.steps == 2,
                truncated=False, info={"task_mode": self.mode, "task_outcome": "running", "success_event": False})

    class Adapter(FakeAdapter):
        def reset(self):
            if temporal:
                self.policy.reset()
        def select_action(self, observation):
            if temporal:
                self.policy.select_action({})
            return parse_policy_action([0.] * 6)

    scenes = [{"name": "left", "task_mode": "pick_place", "target_position_world_m": [.2, -.1, .765],
               "target_delay_s": 1 / 30, "seed": 12},
              {"name": "completed", "task_mode": "idle_completed", "seed": 15}]
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"schema_version": 1, "scenes": scenes}))
    runtime = Runtime([])
    config = SimpleNamespace(chunk_size=15, n_action_steps=1 if temporal else 5,
                             temporal_ensemble_coeff=.01 if temporal else None)
    if temporal:
        from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

        class TinyTemporalPolicy:
            # Reuse the real ACT reset/select_action and temporal averaging,
            # supplying synthetic predictions without a model or checkpoint.
            select_action = ACTPolicy.select_action
            def __init__(self):
                self.config = config
                self.temporal_ensembler = ACTTemporalEnsembler(.01, 15)
                self.reset_count = 0
                self.first_predictions = []
                self.prediction_counts = []
            def eval(self):
                return self
            def reset(self):
                self.reset_count += 1
                self.local_step = 0
                ACTPolicy.reset(self)
                assert self.temporal_ensembler.ensembled_actions is None
                self.prediction_counts.append(0)
            def predict_action_chunk(self, batch):
                if not self.local_step:
                    self.first_predictions.append(self.temporal_ensembler.ensembled_actions)
                self.local_step += 1
                self.prediction_counts[-1] += 1
                return torch.full((1, 15, 6), float(self.reset_count * 10 + self.local_step))
        policy = TinyTemporalPolicy()
    else:
        policy = SimpleNamespace(config=config)
    captured = {}
    def runtime_factory(**kwargs):
        captured.update(kwargs)
        return runtime
    result = run_act_gazebo_multi_episode(tmp_path / "act", device="cpu", episode_count=2, max_steps=5,
        reward_profile="pick_place_v2", evaluation_scenes_path=path, output_directory=tmp_path / "result",
        checkpoint_loader=lambda *args, **kwargs: SimpleNamespace(policy=policy, device="cpu"),
        runtime_factory=runtime_factory, adapter_factory=Adapter)
    assert runtime.attempt == 3
    assert [mode for mode, _ in runtime.configured] == ["pick_place", "pick_place", "idle_completed"]
    assert [options["seed"] for _, options in runtime.configured] == [12, 12, 15]
    assert all(options["n_action_steps"] == (1 if temporal else 5) for _, options in runtime.configured)
    assert [record[:3] for record in runtime.spawned] == [(1, 1, (.2, -.1, .765)), (2, 1, (.2, -.1, .765))]
    assert result.run.episode_count == 2
    assert [episode.initial_info["evaluation_scene"]["name"] for episode in result.run.episodes] == ["left", "completed"]
    assert result.evaluation_metadata["fixed_scene_list"]
    assert captured["reward_profile"] == "pick_place_v2"
    summary = json.loads((tmp_path / "result" / "evaluation_summary.json").read_text())
    assert len(summary["episodes"]) == 2
    assert summary["configuration"]["reward_config_hash"]
    assert len(result.episode_npz_paths) == 2
    assert summary["configuration"]["chunk_schedule"] == (
        "temporal_ensemble_every_step" if temporal else "fixed_n_action_steps")
    if temporal:
        # Existing recovery clears state immediately after failure as well as
        # at the next prepare boundary. That extra reset performs no inference.
        assert policy.reset_count == 4
        assert policy.prediction_counts == [2, 0, 2, 2]
        assert policy.first_predictions == [None, None, None]
        assert summary["configuration"]["temporal_ensemble_coeff"] == .01


def test_fixed_scene_loader_rejects_random_pose_and_count_mismatch(tmp_path):
    import json
    from so101_capstone.policies.act_gazebo_episode import _load_evaluation_scenes
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"schema_version": 1, "scenes": [{"task_mode": "pick_place"}]}))
    with pytest.raises(ValueError, match="require target_position"):
        _load_evaluation_scenes(path, episode_count=1)
    with pytest.raises(ValueError, match="scene count"):
        _load_evaluation_scenes(path, episode_count=2)


def test_fixed_completed_idle_rejects_randomization_before_checkpoint_load(tmp_path):
    import json
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"schema_version": 1, "scenes": [{"task_mode": "idle_completed"}]}))
    with pytest.raises(ValueError, match="idle_completed.*randomization disabled"):
        run_act_gazebo_multi_episode(tmp_path / "act", device="cpu", episode_count=1, max_steps=5,
            reward_profile="pick_place_v2", evaluation_scenes_path=path, environment_randomization_enabled=True,
            checkpoint_loader=lambda *args, **kwargs: pytest.fail("must fail before ACT load"))


def test_new_profile_explicit_randomization_disabled_is_forwarded_despite_ambient_setting(tmp_path, monkeypatch):
    from so101_capstone.policies.act_gazebo_multi_episode import _runtime_context
    from so101_capstone.environments import ReachEnvironmentConfig, SimulationRuntimeConfig
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "true")
    captured = {}
    _runtime_context(runtime_factory=lambda **kwargs: captured.update(kwargs),
        runtime_config=SimulationRuntimeConfig(), environment_config=ReachEnvironmentConfig(),
        reward_profile="pick_place_v2", reward_config=PickPlaceRewardConfig(), environment_randomization_enabled=False)
    assert captured["environment_randomization_enabled"] is False


@pytest.mark.parametrize("reuse_retry_sink", [False, True])
def test_real_runner_retry_archives_only_valid_attempts(tmp_path, reuse_retry_sink):
    import json
    import numpy as np
    from so101_capstone.environments import PolicyObservation, parse_policy_action
    from so101_capstone.environments.simulation_step_coordinator import SimulationStepCoordinatorError
    from so101_capstone.policies import PolicyEpisodeNpzSink, PolicyEpisodeSummary

    def observation(value):
        return PolicyObservation(
            joint_positions=np.full(6, value, dtype=float),
            joint_velocities=np.zeros(6),
            top_image=np.zeros((4, 6, 3), dtype=np.uint8),
            gripper_image=np.zeros((4, 6, 3), dtype=np.uint8),
        )

    class Runtime(FakeRuntime):
        attempt = 0

        def prepare_episode(self):
            self.attempt += 1
            self.steps = 0
            return SimpleNamespace(observation=observation(0), info={"attempt": self.attempt})

        def step(self, action):
            self.steps += 1
            if self.attempt == 1 and self.steps == 2:
                raise SimulationStepCoordinatorError("camera frame expired")
            return SimpleNamespace(
                observation=observation(self.steps), reward=1.0,
                terminated=self.steps == 3, truncated=False,
                info={"attempt": self.attempt, "episode_step": self.steps},
            )

    class Adapter(FakeAdapter):
        def reset(self):
            pass

        def select_action(self, current_observation):
            assert current_observation.top_image.shape == (4, 6, 3)
            return parse_policy_action([0.0] * 6)

    sinks = []
    by_path = {}

    def sink_factory(path, *, metadata):
        if reuse_retry_sink and path in by_path:
            return by_path[path]
        sink = PolicyEpisodeNpzSink(path, metadata=metadata)
        by_path[path] = sink
        sinks.append(sink)
        return sink

    runtime = Runtime([])
    result = run_act_gazebo_multi_episode(
        tmp_path / "checkpoint", device="cpu", episode_count=2, max_steps=5,
        output_directory=tmp_path / "episodes",
        checkpoint_loader=lambda *args, **kwargs: SimpleNamespace(policy=object(), device="cpu"),
        runtime_factory=lambda **kwargs: runtime,
        adapter_factory=Adapter, npz_sink_factory=sink_factory,
    )
    assert runtime.attempt == 3
    assert result.run.total_step_count == 6
    assert result.run.total_reward == 6.0
    assert all(isinstance(episode, PolicyEpisodeSummary) for episode in result.run.episodes)
    assert len(result.episode_npz_paths) == 2
    assert len(set(result.episode_npz_paths)) == 2
    assert all(sink.transitions == () for sink in sinks)
    for index, path in enumerate(result.episode_npz_paths):
        with np.load(path, allow_pickle=False) as archive:
            assert len(archive["reward"]) == 3
            assert json.loads(archive["final_info_json"].item())["attempt"] == index + 2
            assert json.loads(archive["metadata_json"].item())["max_joint_delta_rad"] is None


def test_multi_episode_entry_loads_and_initializes_once(
    tmp_path,
):
    events = []
    captured = {}
    checkpoint = tmp_path / "checkpoint"
    output_directory = tmp_path / "episodes"
    loaded_policy = object()

    def checkpoint_loader(path, *, device):
        events.append("checkpoint.load")
        captured["checkpoint"] = Path(path)
        captured["requested_device"] = device
        return SimpleNamespace(
            policy=loaded_policy,
            device=torch.device("cpu"),
        )

    def adapter_factory(
        policy,
        *,
        device,
        max_joint_delta_rad,
    ):
        events.append("adapter.create")
        adapter = FakeAdapter(
            policy,
            device=device,
            max_joint_delta_rad=max_joint_delta_rad,
        )
        captured["adapter"] = adapter
        return adapter

    runtime = FakeRuntime(events)

    def runtime_factory(*, runtime_config):
        captured["runtime_config"] = (
            runtime_config
        )
        return runtime

    def runner_factory(runtime_arg, policy_arg):
        captured["runner_runtime"] = runtime_arg
        captured["runner_policy"] = policy_arg
        return FakeMultiRunner(
            runtime_arg,
            policy_arg,
            captured,
            events,
        )

    result = run_act_gazebo_multi_episode(
        checkpoint,
        device="cpu",
        max_joint_delta_rad=0.03,
        episode_count=2,
        max_steps=3,
        startup_feedback_timeout_s=12.0,
        output_directory=output_directory,
        checkpoint_loader=checkpoint_loader,
        runtime_factory=runtime_factory,
        adapter_factory=adapter_factory,
        multi_episode_runner_factory=(
            runner_factory
        ),
        npz_sink_factory=FakeSink,
    )

    assert isinstance(
        result,
        ActGazeboMultiEpisodeResult,
    )
    assert result.checkpoint_path == (
        checkpoint.resolve()
    )
    assert result.device == torch.device("cpu")
    assert result.output_directory == (
        output_directory.resolve()
    )
    assert result.episode_npz_paths == (
        (
            output_directory
            / "episode_000000.npz"
        ).resolve(),
        (
            output_directory
            / "episode_000001.npz"
        ).resolve(),
    )
    assert result.target_manifest_paths == ()

    assert events == [
        "checkpoint.load",
        "adapter.create",
        "runtime.enter",
        "runner.run",
        "runtime.exit",
    ]
    assert runtime.enter_calls == 1
    assert runtime.exit_calls == 1
    assert captured["checkpoint"] == (
        checkpoint.resolve()
    )
    assert captured["requested_device"] == "cpu"

    adapter = captured["adapter"]
    assert adapter.policy is loaded_policy
    assert adapter.device == torch.device("cpu")
    assert adapter.max_joint_delta_rad == (
        pytest.approx(0.03)
    )

    assert captured["runner_runtime"] is runtime
    assert captured["runner_policy"] is adapter
    assert captured["episode_count"] == 2
    assert captured["max_steps"] == 3
    assert (
        captured["runtime_config"]
        .startup_feedback_timeout_s
    ) == pytest.approx(12.0)


def test_multi_episode_entry_without_output_directory(
    tmp_path,
):
    events = []
    captured = {}
    runtime = FakeRuntime(events)

    result = run_act_gazebo_multi_episode(
        tmp_path / "checkpoint",
        device="cpu",
        max_joint_delta_rad=0.03,
        episode_count=1,
        max_steps=2,
        checkpoint_loader=(
            lambda path, device: SimpleNamespace(
                policy=object(),
                device=torch.device("cpu"),
            )
        ),
        runtime_factory=(
            lambda runtime_config: runtime
        ),
        adapter_factory=FakeAdapter,
        multi_episode_runner_factory=(
            lambda runtime_arg, policy_arg: (
                FakeMultiRunner(
                    runtime_arg,
                    policy_arg,
                    captured,
                    events,
                )
            )
        ),
    )

    assert result.output_directory is None
    assert result.episode_npz_paths == ()
    assert result.target_manifest_paths == ()
    assert captured[
        "transition_sink_factory"
    ] is None


def test_multi_episode_runtime_closes_on_failure(
    tmp_path,
):
    events = []
    runtime = FakeRuntime(events)

    class FailingRunner:
        def run(
            self,
            *,
            episode_count,
            max_steps,
            transition_sink_factory,
        ):
            raise RuntimeError(
                "multi-episode failed"
            )

    with pytest.raises(
        RuntimeError,
        match="multi-episode failed",
    ):
        run_act_gazebo_multi_episode(
            tmp_path / "checkpoint",
            device="cpu",
            max_joint_delta_rad=0.03,
            episode_count=2,
            max_steps=3,
            checkpoint_loader=(
                lambda path, device: SimpleNamespace(
                    policy=object(),
                    device=torch.device("cpu"),
                )
            ),
            runtime_factory=(
                lambda runtime_config: runtime
            ),
            adapter_factory=FakeAdapter,
            multi_episode_runner_factory=(
                lambda runtime_arg, policy_arg: (
                    FailingRunner()
                )
            ),
        )

    assert runtime.enter_calls == 1
    assert runtime.exit_calls == 1


def test_multi_episode_module_help_has_no_import_warning():
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()

    paths = [
        str(project_root / "src"),
        str(project_root / "lerobot" / "src"),
    ]

    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])

    environment["PYTHONPATH"] = os.pathsep.join(
        paths
    )
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            (
                "so101_capstone.policies."
                "act_gazebo_multi_episode"
            ),
            "--help",
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "RuntimeWarning" not in completed.stderr
    assert "found in sys.modules" not in completed.stderr

def test_multi_episode_entry_wraps_runtime_for_delayed_target(
    tmp_path,
):
    events = []
    captured = {}
    runtime = FakeRuntime(events)
    output_directory = tmp_path / "episodes"

    class Backend:
        def __enter__(self):
            events.append("target_backend.enter")
            return self

        def __exit__(
            self,
            exception_type,
            exception,
            traceback,
        ):
            events.append("target_backend.exit")
            return False

    class Controller:
        def __init__(
            self,
            backend,
            *,
            target_delay_s,
            policy_hz,
            output_directory,
        ):
            captured["target_backend"] = backend
            captured["target_delay_s"] = target_delay_s
            captured["policy_hz"] = policy_hz
            captured["target_output_directory"] = (
                output_directory
            )
            self.manifest_paths = (
                output_directory
                / "episode_000000_target.json",
            )

    class RuntimeProxy:
        def __init__(self, wrapped, controller):
            captured["wrapped_runtime"] = wrapped
            captured["target_controller"] = controller

    proxy_holder = {}

    def delayed_runtime_factory(wrapped, controller):
        proxy = RuntimeProxy(wrapped, controller)
        proxy_holder["proxy"] = proxy
        return proxy

    def runner_factory(runtime_arg, policy_arg):
        captured["runner_runtime"] = runtime_arg
        return FakeMultiRunner(
            runtime_arg,
            policy_arg,
            captured,
            events,
        )

    result = run_act_gazebo_multi_episode(
        tmp_path / "checkpoint",
        device="cpu",
        max_joint_delta_rad=0.03,
        episode_count=1,
        max_steps=120,
        output_directory=output_directory,
        spawn_random_target_after_s=3.0,
        policy_hz=30.0,
        checkpoint_loader=(
            lambda path, device: SimpleNamespace(
                policy=object(),
                device=torch.device("cpu"),
            )
        ),
        runtime_factory=(
            lambda runtime_config: runtime
        ),
        adapter_factory=FakeAdapter,
        multi_episode_runner_factory=(
            runner_factory
        ),
        target_backend_factory=Backend,
        target_controller_factory=Controller,
        delayed_runtime_factory=(
            delayed_runtime_factory
        ),
        npz_sink_factory=FakeSink,
    )

    assert captured["target_delay_s"] == pytest.approx(3.0)
    assert captured["policy_hz"] == pytest.approx(30.0)
    assert captured["wrapped_runtime"] is runtime
    assert captured["runner_runtime"] is proxy_holder["proxy"]
    assert result.target_manifest_paths == (
        (
            output_directory
            / "episode_000000_target.json"
        ).resolve(),
    )
    assert events == [
        "runtime.enter",
        "target_backend.enter",
        "runner.run",
        "target_backend.exit",
        "runtime.exit",
    ]
