from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from so101_capstone.policies.act_gazebo_episode import (
    ActGazeboEpisodeResult,
    run_act_gazebo_episode,
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
        self.max_joint_delta_rad = max_joint_delta_rad


class FakeRunner:
    def __init__(self, runtime, policy, events, episode):
        self.runtime = runtime
        self.policy = policy
        self.events = events
        self.episode = episode
        self.max_steps = None

    def run_episode(self, *, max_steps):
        self.max_steps = max_steps
        self.events.append("runner.run")
        return self.episode


@pytest.mark.parametrize("randomization", [False, True])
def test_new_profile_forwards_resolved_reward_and_explicit_scene_without_changing_policy(tmp_path, monkeypatch, randomization):
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", str(randomization).lower())
    events, captured = [], {}

    class Runtime(FakeRuntime):
        def delete_all_targets(self):
            events.append("delete")
            return SimpleNamespace(success=True)
        def configure_task(self, mode, **kwargs):
            captured["task"] = (mode, kwargs)
            events.append("configure")
        def prepare_episode(self, **kwargs):
            events.append("prepare")
            captured["prepare_kwargs"] = kwargs
            return SimpleNamespace(observation=object(), info={"physical_target_present": False})
        def spawn_target(self, position, **kwargs):
            captured["target"] = (position, kwargs)
            events.append("spawn")
            return SimpleNamespace(success=True)

    class Runner:
        def __init__(self, runtime, policy, **kwargs):
            self.runtime = runtime
            captured["runner_kwargs"] = kwargs
        def run_episode(self, *, max_steps):
            prepared = self.runtime.prepare_episode()
            captured["initial_info"] = prepared.info
            return SimpleNamespace(step_count=1)

    runtime = Runtime(events)
    loaded_policy = SimpleNamespace(config=SimpleNamespace(chunk_size=15, n_action_steps=5, temporal_ensemble_coeff=None))
    def runtime_factory(**kwargs):
        captured["runtime_kwargs"] = kwargs
        return runtime
    result = run_act_gazebo_episode(tmp_path / "act", device="cpu", max_steps=1200,
        reward_profile="pick_place_v2", task_mode="pick_place", observe_after_success_s=10.,
        target_position_world_m=(.2, -.1, .765), scene_seed=17,
        checkpoint_loader=lambda *args, **kwargs: SimpleNamespace(policy=loaded_policy, device="cpu"),
        runtime_factory=runtime_factory, adapter_factory=FakeAdapter, runner_factory=Runner)
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    assert isinstance(captured["runtime_kwargs"]["reward_config"], PickPlaceRewardConfig)
    assert captured["runtime_kwargs"]["observe_after_success_s"] == 10.
    assert captured["runtime_kwargs"]["environment_config"].max_episode_steps == 1200
    assert captured["runtime_kwargs"]["environment_randomization_enabled"] is randomization
    assert captured["prepare_kwargs"] == ({"environment_randomization_episode_key": 17} if randomization else {})
    assert captured["task"] == ("pick_place", {"seed": 17, "n_action_steps": 5})
    assert captured["target"] == ((.2, -.1, .765), {"quaternion_xyzw": (0., 0., 0., 1.)})
    assert events.index("configure") < events.index("prepare") < events.index("spawn")
    assert captured["initial_info"]["physical_target_present"] is False
    assert result.evaluation_metadata["n_action_steps"] == 5
    assert result.evaluation_metadata["deterministic_act"]
    assert captured["runner_kwargs"] == {"observe_after_success_s": 10., "policy_hz": 30.}


def test_formal_evaluation_rejects_proxy_reward_before_loading_checkpoint(tmp_path):
    from so101_capstone.policies.act_gazebo_episode import _evaluation_reward
    path = tmp_path / "proxy.json"
    path.write_text('{"grasp_evidence_mode": "proxy"}')
    with pytest.raises(ValueError, match="requires contact evidence"):
        _evaluation_reward("pick_place_v2", path, "pick_place", 0.)


@pytest.mark.parametrize("coefficient,steps", [
    (float("nan"), 1), (float("inf"), 1), (True, 1), (.01, 5), (-1., 1),
])
def test_formal_evaluation_rejects_invalid_temporal_schedule_before_runtime(tmp_path, coefficient, steps):
    policy = SimpleNamespace(config=SimpleNamespace(
        chunk_size=100, n_action_steps=steps, temporal_ensemble_coeff=coefficient))
    with pytest.raises(ValueError, match="temporal"):
        run_act_gazebo_episode(tmp_path, device="cpu", max_steps=3, reward_profile="pick_place_v2",
            checkpoint_loader=lambda *args, **kwargs: SimpleNamespace(policy=policy, device="cpu"),
            runtime_factory=lambda **kwargs: pytest.fail("invalid config must fail before runtime"))


@pytest.mark.parametrize("monitor", [False, True])
def test_formal_temporal_evaluation_uses_real_select_action_and_clears_initial_history(tmp_path, monkeypatch, monitor):
    import numpy as np
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from so101_capstone.environments import PolicyObservation

    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "false")
    observation = PolicyObservation(joint_positions=np.zeros(6), joint_velocities=np.zeros(6),
        top_image=np.zeros((480, 640, 3), dtype=np.uint8),
        gripper_image=np.zeros((480, 640, 3), dtype=np.uint8))

    class TinyTemporalPolicy(ACTPolicy):
        # Exercise the real inference/reset methods without building a network
        # or loading checkpoint weights. Predict deterministic synthetic chunks.
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(chunk_size=3, n_action_steps=1, temporal_ensemble_coeff=.01)
            self.temporal_ensembler = ACTTemporalEnsembler(.01, 3)
            self.reset_count = self.chunk_calls = 0
            self.selected = []
        def reset(self):
            self.reset_count += 1
            super().reset()
        def predict_action_chunk(self, batch):
            self.chunk_calls += 1
            return torch.full((1, 3, 6), float(self.chunk_calls * 10))
        def select_action(self, batch):
            action = super().select_action(batch)
            self.selected.append(action.clone())
            return action

    class Runtime(FakeRuntime):
        def configure_task(self, mode, **kwargs):
            self.task = (mode, kwargs)
        def delete_all_targets(self):
            return SimpleNamespace(success=True)
        def prepare_episode(self):
            self.step_count = 0
            return SimpleNamespace(observation=observation, info={})
        def step(self, action):
            self.step_count += 1
            return SimpleNamespace(observation=observation, reward=0.,
                terminated=self.step_count == (4 if monitor else 2), truncated=False,
                info={"success_event": self.step_count == 2})

    policy = TinyTemporalPolicy()
    policy.temporal_ensembler.update(torch.full((1, 3, 6), 999.))
    runtime = Runtime([])
    result = run_act_gazebo_episode(tmp_path / "act", device="cpu", max_steps=2,
        reward_profile="pick_place_v2", task_mode="idle_empty",
        observe_after_success_s=2 / 30 if monitor else 0.,
        checkpoint_loader=lambda *args, **kwargs: SimpleNamespace(policy=policy, device="cpu"),
        runtime_factory=lambda **kwargs: runtime)
    assert policy.reset_count == 1
    assert policy.chunk_calls == runtime.step_count == result.episode.step_count == (4 if monitor else 2)
    assert torch.allclose(policy.selected[0], torch.full((1, 6), 10.))
    weight = torch.exp(torch.tensor(-.01))
    expected = (10. + weight * 20.) / (1. + weight)
    assert torch.allclose(policy.selected[1], torch.full((1, 6), expected))
    assert runtime.task == ("idle_empty", {"seed": 0, "n_action_steps": 1})
    assert result.evaluation_metadata["chunk_schedule"] == "temporal_ensemble_every_step"
    assert result.evaluation_metadata["temporal_ensemble_coeff"] == .01
    assert result.evaluation_metadata["reward_config"]["control_hz"] == 30.
    if monitor:
        assert result.episode.final_info["evaluation"]["monitor_step_count"] == 2
        # The success event must not reset the temporal ensemble before the
        # monitoring steps; the next action still includes earlier predictions.
        assert torch.all(policy.selected[2] < 30.)


@pytest.mark.parametrize("source", ["run", "checkpoint", "explicit"])
def test_evaluation_reuses_resolved_training_gamma_and_explicit_override(tmp_path, source):
    import json
    training_run = tmp_path / "run"
    checkpoint = training_run / "checkpoints" / "000005" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    automatic = training_run / "resolved_reward.json"
    automatic.write_text(json.dumps({"gamma": .995}))
    checkpoint_reward = checkpoint / "resolved_reward.json"
    if source != "run":
        checkpoint_reward.write_text(json.dumps({"gamma": .993}))
    explicit = tmp_path / "explicit_reward.json"
    explicit.write_text(json.dumps({"gamma": .99}))
    captured = {}
    runtime = FakeRuntime([])
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=5, n_action_steps=5, temporal_ensemble_coeff=None))
    def runtime_factory(**kwargs):
        captured.update(kwargs)
        return runtime
    result = run_act_gazebo_episode(checkpoint, device="cpu", max_steps=5, reward_profile="pick_place_v2",
        reward_config=explicit if source == "explicit" else None,
        checkpoint_loader=lambda *args, **kwargs: SimpleNamespace(policy=policy, device="cpu"),
        runtime_factory=runtime_factory, adapter_factory=FakeAdapter,
        runner_factory=lambda *args: FakeRunner(runtime, policy, [], object()))
    expected_gamma, expected_path = {"run": (.995, automatic), "checkpoint": (.993, checkpoint_reward),
                                     "explicit": (.99, explicit)}[source]
    assert captured["reward_config"].gamma == expected_gamma
    assert result.evaluation_metadata["reward_config"]["gamma"] == expected_gamma
    assert result.evaluation_metadata["reward_config_source_path"] == str(expected_path)


def test_evaluation_rejects_physics_rate_mismatch_before_checkpoint_load(tmp_path):
    path = tmp_path / "reward.json"
    path.write_text('{"control_hz": 20}')
    with pytest.raises(ValueError, match="control_hz=30"):
        run_act_gazebo_episode(tmp_path, device="cpu", max_steps=5, reward_profile="pick_place_v2",
            reward_config=path, checkpoint_loader=lambda *args, **kwargs: pytest.fail("must fail before ACT load"))


def test_single_completed_idle_rejects_environment_randomization_before_checkpoint_load(tmp_path, monkeypatch):
    monkeypatch.setenv("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "true")
    with pytest.raises(ValueError, match="idle_completed.*randomization disabled"):
        run_act_gazebo_episode(tmp_path, device="cpu", max_steps=5, reward_profile="pick_place_v2",
            task_mode="idle_completed", checkpoint_loader=lambda *args, **kwargs: pytest.fail("must fail before ACT load"))


def test_new_evaluation_cli_accepts_fixed_pose_and_monitor_window():
    from so101_capstone.policies.act_gazebo_episode import _parser
    args = _parser().parse_args(["--checkpoint", "/tmp/act", "--max-steps", "900",
        "--target-position-world-m", ".2", "-.1", ".765", "--observe-after-success-s", "20"])
    assert args.reward_profile == "pick_place_v2"
    assert args.target_position_world_m == [.2, -.1, .765]
    assert args.observe_after_success_s == 20.


def test_default_runtime_receives_requested_horizon_and_unlimited_delta(tmp_path, monkeypatch):
    import so101_capstone.policies.act_gazebo_episode as module

    captured = {}
    runtime = FakeRuntime([])

    def create_runtime(**kwargs):
        captured.update(kwargs)
        return runtime

    def adapter_factory(policy, **kwargs):
        captured["delta"] = kwargs["max_joint_delta_rad"]
        return FakeAdapter(policy, **kwargs)

    monkeypatch.setattr(module, "create_ros_simulation_runtime", create_runtime)
    module.run_act_gazebo_episode(
        tmp_path / "checkpoint",
        device="cpu",
        max_steps=1200,
        checkpoint_loader=lambda *args, **kwargs: SimpleNamespace(policy=object(), device="cpu"),
        adapter_factory=adapter_factory,
        runner_factory=lambda *args: FakeRunner(runtime, object(), [], object()),
    )
    assert captured["environment_config"].max_episode_steps == 1200
    assert captured["delta"] is None
    args = module._parser().parse_args(["--checkpoint", str(tmp_path), "--max-steps", "1200"])
    assert args.max_joint_delta_rad is None


def test_act_gazebo_episode_wires_checkpoint_adapter_runtime_and_runner(
    tmp_path,
):
    events = []
    checkpoint = tmp_path / "checkpoint"
    loaded_policy = object()
    episode = SimpleNamespace(
        step_count=3,
        total_reward=1.5,
        terminated=False,
        truncated=False,
        reached_max_steps=True,
        final_info={"episode_step": 3},
    )
    captured = {}

    def checkpoint_loader(path, *, device):
        captured["checkpoint_path"] = Path(path)
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
        adapter = FakeAdapter(
            policy,
            device=device,
            max_joint_delta_rad=max_joint_delta_rad,
        )
        captured["adapter"] = adapter
        return adapter

    runtime = FakeRuntime(events)

    def runtime_factory(*, runtime_config):
        captured["runtime_config"] = runtime_config
        return runtime

    def runner_factory(runtime_arg, policy_arg):
        runner = FakeRunner(
            runtime_arg,
            policy_arg,
            events,
            episode,
        )
        captured["runner"] = runner
        return runner

    result = run_act_gazebo_episode(
        checkpoint,
        device="cpu",
        max_joint_delta_rad=0.05,
        max_steps=3,
        startup_feedback_timeout_s=12.0,
        checkpoint_loader=checkpoint_loader,
        runtime_factory=runtime_factory,
        adapter_factory=adapter_factory,
        runner_factory=runner_factory,
    )

    assert isinstance(result, ActGazeboEpisodeResult)
    assert result.checkpoint_path == checkpoint.resolve()
    assert result.device == torch.device("cpu")
    assert result.episode is episode
    assert result.episode_npz_path is None

    assert captured["checkpoint_path"] == checkpoint
    assert captured["requested_device"] == "cpu"

    adapter = captured["adapter"]
    assert adapter.policy is loaded_policy
    assert adapter.device == torch.device("cpu")
    assert adapter.max_joint_delta_rad == pytest.approx(0.05)

    assert (
        captured["runtime_config"]
        .startup_feedback_timeout_s
    ) == pytest.approx(12.0)

    runner = captured["runner"]
    assert runner.runtime is runtime
    assert runner.policy is adapter
    assert runner.max_steps == 3

    assert runtime.enter_calls == 1
    assert runtime.exit_calls == 1
    assert events == [
        "runtime.enter",
        "runner.run",
        "runtime.exit",
    ]


def test_runtime_context_closes_when_runner_raises(
    tmp_path,
):
    events = []
    runtime = FakeRuntime(events)

    class FailingRunner:
        def run_episode(self, *, max_steps):
            raise RuntimeError("episode failed")

    with pytest.raises(
        RuntimeError,
        match="episode failed",
    ):
        run_act_gazebo_episode(
            tmp_path / "checkpoint",
            device="cpu",
            max_joint_delta_rad=0.05,
            max_steps=1,
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
            runner_factory=(
                lambda runtime_arg, policy_arg: FailingRunner()
            ),
        )

    assert runtime.enter_calls == 1
    assert runtime.exit_calls == 1


def test_module_cli_help_has_no_duplicate_import_warning():
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_paths = [
        str(project_root / "src"),
        str(project_root / "lerobot" / "src"),
    ]

    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])

    environment["PYTHONPATH"] = os.pathsep.join(
        python_paths
    )
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "so101_capstone.policies.act_gazebo_episode",
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

def test_act_episode_can_persist_npz_through_transition_sink(
    tmp_path,
):
    events = []
    runtime = FakeRuntime(events)
    output_path = tmp_path / "episode.npz"
    captured = {}
    episode = SimpleNamespace(
        step_count=1,
        total_reward=0.5,
        terminated=False,
        truncated=False,
        reached_max_steps=True,
        final_info={"episode_step": 1},
    )

    class FakeNpzSink:
        def __init__(
            self,
            path,
            *,
            metadata,
        ):
            self.output_path = Path(path).resolve()
            self.metadata = metadata
            self.written_path = self.output_path
            captured["sink"] = self

    class SinkAwareRunner:
        def __init__(
            self,
            runtime_arg,
            policy_arg,
            *,
            transition_sink,
        ):
            captured["runtime"] = runtime_arg
            captured["policy"] = policy_arg
            captured["transition_sink"] = transition_sink

        def run_episode(self, *, max_steps):
            captured["max_steps"] = max_steps
            return episode

    result = run_act_gazebo_episode(
        tmp_path / "checkpoint",
        device="cpu",
        max_joint_delta_rad=0.05,
        max_steps=1,
        output_npz_path=output_path,
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
        runner_factory=SinkAwareRunner,
        npz_sink_factory=FakeNpzSink,
    )

    sink = captured["sink"]

    assert sink.output_path == output_path.resolve()
    assert sink.metadata["checkpoint"].endswith(
        "checkpoint"
    )
    assert sink.metadata["device"] == "cpu"
    assert sink.metadata["max_steps"] == 1
    assert sink.metadata[
        "max_joint_delta_rad"
    ] == pytest.approx(0.05)
    assert captured["transition_sink"] is sink
    assert result.episode_npz_path == output_path.resolve()
