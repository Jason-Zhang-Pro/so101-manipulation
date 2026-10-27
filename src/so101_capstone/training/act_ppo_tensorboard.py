"""TensorBoard logging and automatic browser launch for Direct ACT PPO."""

from __future__ import annotations

import atexit
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Mapping
import webbrowser


_UPDATE_TAGS = {
    "policy_loss": "loss/policy_loss",
    "value_loss": "loss/value_loss",
    "mean_bounds_loss": "loss/mean_bounds_loss",
    "approximate_kl": "ppo/approximate_kl",
    "target_kl": "ppo/target_kl",
    "kl_backoff_attempt_count": "ppo/kl_backoff_attempt_count",
    "kl_backoff_rejection_count": "ppo/kl_backoff_rejection_count",
    "kl_backoff_retry_count": "ppo/kl_backoff_retry_count",
    "kl_backoff_at_min_scale": "ppo/kl_backoff_at_min_scale",
    "policy_learning_rate_scale": "learning_rate/policy_scale",
    "configured_ppo_epochs": "ppo/configured_epochs",
    "completed_ppo_epochs": "ppo/completed_epochs",
    "early_stopped": "ppo/early_stopped",
    "clip_fraction": "ppo/clip_fraction",
    "gradient_norm": "optimization/gradient_norm",
    "actor_gradient_norm": "optimization/actor_gradient_norm",
    "critic_gradient_norm": "optimization/critic_gradient_norm",
    "exploration_gradient_norm": "optimization/exploration_gradient_norm",
    "actor_parameter_relative_delta": "optimization/actor_parameter_relative_delta",
    "actor_parameter_changed_fraction": "optimization/actor_parameter_changed_fraction",
    "critic_parameter_relative_delta": "optimization/critic_parameter_relative_delta",
    "critic_parameter_changed_fraction": "optimization/critic_parameter_changed_fraction",
    "exploration_parameter_relative_delta": "optimization/exploration_parameter_relative_delta",
    "exploration_parameter_changed_fraction": "optimization/exploration_parameter_changed_fraction",
    "actor_learning_rate": "learning_rate/actor",
    "actor_backbone_learning_rate": "learning_rate/actor_backbone",
    "critic_learning_rate": "learning_rate/critic",
    "exploration_learning_rate": "learning_rate/exploration",
    "log_std_mean": "exploration/log_std_mean",
    "transition_count": "rollout/transition_count",
    "minibatch_count": "rollout/minibatch_count",
}

# Schema 4 reports KL for the actual bounded distribution. Older payloads used
# an unclipped Gaussian and retain their historical labels when read directly.
_POLICY_KL_ALIASES = {
    "pre_update_policy_kl": "pre_update_pre_clip_policy_kl",
    "policy_kl": "pre_clip_policy_kl",
    "policy_kl_p95": "pre_clip_policy_kl_p95",
    "policy_kl_max": "pre_clip_policy_kl_max",
}


def _positive_port(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("preferred_port must be an integer.")
    if not 1 <= value <= 65535:
        raise ValueError("preferred_port must be within [1, 65535].")
    return value


def _available_port(preferred_port: int) -> int:
    start = _positive_port(preferred_port)
    for port in range(start, min(start + 100, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No free TensorBoard port was found from {start} to "
        f"{min(start + 99, 65535)}."
    )


def _wait_for_server(
    process: subprocess.Popen[bytes],
    *,
    port: int,
    timeout_s: float = 8.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "TensorBoard server exited during startup with code "
                f"{return_code}. Check tensorboard_server.log."
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.2)
            try:
                client.connect(("127.0.0.1", port))
            except OSError:
                time.sleep(0.1)
                continue
            return
    raise RuntimeError(
        f"TensorBoard did not become available on port {port} within "
        f"{timeout_s:.1f} seconds."
    )


def _open_url(url: str) -> bool:
    try:
        if webbrowser.open_new_tab(url):
            return True
    except Exception:
        pass

    opener = shutil.which("xdg-open")
    if opener is None:
        return False
    try:
        subprocess.Popen(
            [opener, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    return True


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


class DirectActPpoTensorBoard:
    """Write one TensorBoard point for every PPO update and episode."""

    def __init__(
        self,
        *,
        writer: Any | None,
        log_directory: Path | None,
        url: str | None,
        server_pid: int | None,
    ) -> None:
        self._writer = writer
        self.log_directory = log_directory
        self.url = url
        self.server_pid = server_pid
        self._closed = False
        if writer is not None:
            atexit.register(self.close)

    @classmethod
    def disabled(cls) -> "DirectActPpoTensorBoard":
        return cls(
            writer=None,
            log_directory=None,
            url=None,
            server_pid=None,
        )

    @classmethod
    def start(
        cls,
        log_directory: str | Path,
        *,
        preferred_port: int = 6006,
        auto_open: bool = True,
    ) -> "DirectActPpoTensorBoard":
        if not isinstance(auto_open, bool):
            raise ValueError("auto_open must be a bool.")

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard support is unavailable in the active Python "
                "environment. Install tensorboard in the lerobot conda "
                "environment."
            ) from error

        directory = Path(log_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(directory), flush_secs=1)
        writer.add_text("run/status", "Direct ACT PPO training started", 0)
        writer.flush()

        port = _available_port(preferred_port)
        url = f"http://127.0.0.1:{port}/"
        server_log_path = directory.parent / "tensorboard_server.log"
        with server_log_path.open("ab", buffering=0) as server_log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(directory),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--reload_interval",
                    "1",
                ],
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )

        try:
            _wait_for_server(process, port=port)
        except Exception:
            writer.close()
            raise

        opened = False
        if auto_open:
            opened = _open_url(url)

        print(f"TensorBoard log directory: {directory}", flush=True)
        print(f"TensorBoard URL: {url}", flush=True)
        if auto_open and not opened:
            print(
                "TensorBoard browser could not be opened automatically; "
                "open the URL above manually.",
                flush=True,
            )

        return cls(
            writer=writer,
            log_directory=directory,
            url=url,
            server_pid=process.pid,
        )

    @property
    def log_directory_string(self) -> str | None:
        if self.log_directory is None:
            return None
        return str(self.log_directory)

    def log_stage_success_skip(self, payload: Mapping[str, Any]) -> None:
        """Collection-batch diagnostics, including rounds with no PPO update."""
        if self._writer is None:
            return
        batch_index = int(payload["batch_index"])
        for key in ("collected_transition_count", "update_transition_count", "excluded_transition_count", "empty_batch"):
            value = _finite_scalar(payload.get(key))
            if value is not None:
                self._writer.add_scalar(f"stage_skip/{key}", value, batch_index)
        for stage, rate in payload.get("success_rates", {}).items():
            value = _finite_scalar(rate)
            if value is not None:
                self._writer.add_scalar(f"stage_skip/{stage}/success_rate", value, batch_index)
        for stage in ("S0", "S1", "S2", "S3", "S4", "S5"):
            for key in ("exclude_update_stages", "next_skip_stages"):
                self._writer.add_scalar(f"stage_skip/{stage}/{key}",
                                       float(stage in payload.get(key, [])), batch_index)
        self._writer.flush()

    def log_update(
        self,
        payload: Mapping[str, Any],
        *,
        total_environment_steps: int,
    ) -> None:
        if self._writer is None:
            return
        update_index = int(payload["update_index"])

        for key, tag in _UPDATE_TAGS.items():
            value = _finite_scalar(payload.get(key))
            if value is not None:
                self._writer.add_scalar(tag, value, update_index)

        # Raw diagnostics describe actor transition presentations in accepted
        # minibatches (including repeated epochs), not fresh environment steps.
        # Disabled collection must not look like a verified zero overflow rate.
        raw_count = _finite_scalar(payload.get("raw_mean_bounds_diagnostic_count"))
        if raw_count is not None:
            self._writer.add_scalar("optimization/raw_bounds_diagnostic_count", raw_count, update_index)
        if raw_count is not None and raw_count > 0:
            for key, tag in (
                ("raw_mean_bounds_loss", "loss/raw_mean_bounds_loss"),
                ("raw_mean_out_of_bounds_fraction", "action/raw_mean_out_of_bounds_fraction"),
                ("raw_mean_overflow_normalized", "action/raw_mean_overflow_normalized"),
                ("raw_mean_bound_derivative_mean", "action/raw_mean_bound_derivative_mean"),
            ):
                value = _finite_scalar(payload.get(key))
                if value is not None:
                    self._writer.add_scalar(tag, value, update_index)

        for current, legacy in _POLICY_KL_ALIASES.items():
            key = current if current in payload else legacy
            value = _finite_scalar(payload.get(key))
            if value is not None:
                self._writer.add_scalar(f"ppo/{key}", value, update_index)

        stage_sampling = payload.get("stage_sampling", {})
        if isinstance(stage_sampling, Mapping):
            for key in ("s0_steps", "s1_steps", "s2_steps", "s3_steps", "s4_steps", "s5_steps", "total_steps", "target_s0_step_fraction",
                        "actual_s0_step_fraction", "ratio_error", "ratio_tolerance", "ratio_met"):
                value = _finite_scalar(stage_sampling.get(key))
                if value is not None:
                    self._writer.add_scalar(f"stage_batch/{key}", value, update_index)

        # Emit exactly one distribution-specific entropy trace. Do not relabel
        # archived Gaussian metrics as bounded-distribution entropy.
        entropy_key = ("truncated_gaussian_entropy" if "truncated_gaussian_entropy" in payload
                       else "pre_clip_gaussian_entropy")
        entropy = _finite_scalar(payload.get(entropy_key, payload.get("entropy")))
        if entropy is not None:
            self._writer.add_scalar(f"ppo/{entropy_key}", entropy, update_index)
        for key, tag in (("exploration_log_std", "exploration/log_std"),
                         ("exploration_log_std_delta", "exploration/log_std_delta")):
            values = payload.get(key)
            if isinstance(values, (tuple, list)) and len(values) == 6:
                for joint, item in enumerate(values):
                    value = _finite_scalar(item)
                    if value is not None:
                        self._writer.add_scalar(f"{tag}/joint_{joint}", value, update_index)

        # Phase diagnostics describe the final accepted policy against the same
        # rollout reference. Rejected candidates remain in the structured log.
        phases = payload.get("phase_diagnostics", {})
        if isinstance(phases, Mapping):
            for phase, diagnostics in phases.items():
                if not isinstance(diagnostics, Mapping):
                    continue
                suppressed_aliases = {
                    legacy for current, legacy in _POLICY_KL_ALIASES.items()
                    if current in diagnostics
                }
                for name, item in diagnostics.items():
                    if name in suppressed_aliases:
                        continue
                    tag = f"ppo/phase/{phase}/{name}"
                    if isinstance(item, Mapping):
                        for joint, scalar in item.items():
                            value = _finite_scalar(scalar)
                            if value is not None:
                                self._writer.add_scalar(f"{tag}/{joint}", value, update_index)
                    elif isinstance(item, (tuple, list)) and len(item) == 6:
                        for joint, scalar in enumerate(item):
                            value = _finite_scalar(scalar)
                            if value is not None:
                                self._writer.add_scalar(f"{tag}/joint_{joint}", value, update_index)
                    else:
                        value = _finite_scalar(item)
                        if value is not None:
                            self._writer.add_scalar(tag, value, update_index)

        self._writer.add_scalar(
            "progress/total_environment_steps",
            int(total_environment_steps),
            update_index,
        )
        self._writer.add_scalar(
            "progress/partial_rollout",
            float(bool(payload.get("partial_rollout", False))),
            update_index,
        )
        self._writer.add_scalar(
            "progress/checkpoint_saved",
            float(bool(payload.get("checkpoint_saved", False))),
            update_index,
        )
        self._writer.flush()

    def log_discarded_steps(self, snapshot: Mapping[str, Any], *, event_index: int) -> None:
        """Log cumulative lost work independently of successful PPO updates."""
        if self._writer is None:
            return
        for key in (
            "discarded_environment_steps", "discarded_transition_count",
            "gazebo_crash_discarded_environment_steps", "gazebo_crash_discarded_transition_count",
            "gazebo_crash_episode_count", "failure_episode_count", "unconfirmed_step_count",
            "final_partial_rollout_discarded_transition_count",
            "aborted_rollout_discarded_transition_count", "interrupted_update_transition_count",
        ):
            value = _finite_scalar(snapshot.get(key))
            if value is not None:
                self._writer.add_scalar(f"discarded/{key}", value, event_index)
        self._writer.flush()

    def log_episode(self, episode: Mapping[str, Any]) -> None:
        if self._writer is None:
            return
        episode_step = int(episode["episode_index"]) + 1
        step_count = int(episode["step_count"])
        total_reward = float(episode["total_reward"])

        self._writer.add_scalar(
            "episode/total_reward",
            total_reward,
            episode_step,
        )
        self._writer.add_scalar(
            "episode/step_count",
            step_count,
            episode_step,
        )
        if step_count > 0:
            self._writer.add_scalar(
                "episode/reward_per_step",
                total_reward / step_count,
                episode_step,
            )
        for key in ("terminated", "truncated", "reached_max_steps"):
            self._writer.add_scalar(
                f"episode/{key}",
                float(bool(episode.get(key, False))),
                episode_step,
            )
        info = episode.get("final_info", {})
        training_stage = info.get("training_stage")
        if training_stage in {"S0", "S1", "S2", "S3", "S4", "S5"}:
            for key in ("stage_success", "stage_timeout", "stage_elapsed_s", "stage_budget_s"):
                value = _finite_scalar(info.get(key))
                if value is not None:
                    self._writer.add_scalar(f"stage/{training_stage}/{key}", value, episode_step)
            self._writer.add_scalar(f"stage/{training_stage}/reward_per_step",
                                    total_reward / max(1, step_count), episode_step)
            for key, item in info.get("stage_metrics", {}).items():
                values = item if isinstance(item, (tuple, list)) else [item]
                for index, scalar in enumerate(values):
                    value = _finite_scalar(scalar)
                    if value is not None:
                        suffix = f"/joint_{index}" if isinstance(item, (tuple, list)) else ""
                        self._writer.add_scalar(f"stage/{training_stage}/{key}{suffix}", value, episode_step)
        for key in ("curriculum_level", "sampled_level", "curriculum_rehearsal"):
            value = _finite_scalar(episode.get(key))
            if value is not None:
                self._writer.add_scalar(f"curriculum/{key}", value, episode_step)
        if "task_outcome" in info:
            mode = episode.get("task_mode", info.get("task_mode", "pick_place"))
            self._writer.add_scalar(f"task/{mode}/success", float(info["task_outcome"] == "success"), episode_step)
            self._writer.add_scalar(f"task/{mode}/failure", float(info["task_outcome"] == "task_failure"), episode_step)
            for key in ("robot_at_home", "robot_static", "home_stable_duration_s", "place_stable_duration_s",
                        "home_error_rad", "max_joint_velocity_rad_s", "target_distance_m",
                        "approach_distance_score", "approach_orientation_score", "approach_opening_score",
                        "grasp_close_gate", "grasp_closure_score", "pregrasp_ready_duration_s"):
                value = _finite_scalar(info.get(key))
                if value is not None:
                    self._writer.add_scalar(f"task/{key}", value, episode_step)
            for key, value in episode.get("stage_reached", {}).items():
                self._writer.add_scalar(f"task/stage_reached/{key}", float(value), episode_step)
            for key, value in episode.get("reward_components_total", {}).items():
                self._writer.add_scalar(f"reward/{key}", float(value), episode_step)
            for phase, diagnostics in episode.get("action_diagnostics", {}).items():
                for name, value in diagnostics.items():
                    if isinstance(value, list):
                        for joint, scalar in enumerate(value):
                            self._writer.add_scalar(f"action/{phase}/{name}/joint_{joint}", float(scalar), episode_step)
                    else:
                        self._writer.add_scalar(f"action/{phase}/{name}", float(value), episode_step)
        self._writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
