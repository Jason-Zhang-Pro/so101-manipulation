"""Repeatable ACT-only test: ZERO -> autonomous HOME -> target at 3s -> stop at 13s.

All times are simulation times. Planning is read-only; --execute starts Gazebo.
This continuous test deliberately does not invoke the HOME-initialized S0/S1
training protocol, PPO exploration, augmentation, or an optimizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid
from typing import Any

from .act_paired_evaluation import (
    _json_default, _read_json, _require, _save_frame, _write_json,
    assert_resources_unused,
)

POLICY_HZ = 30
TARGET_DELAY_STEPS = 90
TOTAL_STEPS = 390


def build_plan(checkpoint: Path, *, project_root: Path, output_directory: Path | None = None,
               reward_config: Path | None = None, device: str = "cuda", gui: bool = True,
               ros_domain_id: int = 120, episodes: int = 1, seed: int = 1009,
               target_position: tuple[float, float, float] = (.25, 0., .768),
               startup_timeout_s: float = 60.) -> dict[str, Any]:
    """Resolve artifacts without loading a model or starting ROS/Gazebo."""
    checkpoint, project_root = checkpoint.expanduser().resolve(), project_root.resolve()
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if not 0 <= ros_domain_id <= 232 or episodes < 1 or not 0 <= seed < 2**32:
        raise ValueError("Invalid domain, episode count, or seed.")
    if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
        raise ValueError("startup timeout must be positive and finite.")
    if len(target_position) != 3 or not all(math.isfinite(v) for v in target_position):
        raise ValueError("Target position must contain three finite world coordinates.")
    model_config = _read_json(checkpoint / "config.json")
    if model_config.get("type") != "act":
        raise ValueError("An ACT pretrained_model directory is required.")
    # Fine-tuned policies carry their scoring reference. Original supervised
    # checkpoints use the current project reward unless explicitly overridden.
    reward_path = reward_config or (checkpoint / "resolved_reward.json")
    if reward_config is None and not reward_path.is_file():
        reward_path = project_root / "config/pick_place_reward.json"
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    resolved_reward = PickPlaceRewardConfig.load(reward_path).to_dict()
    if resolved_reward["control_hz"] != POLICY_HZ:
        raise ValueError("This evaluator requires the existing 30 Hz control schedule.")
    token = uuid.uuid4().hex[:8]
    output = (output_directory or project_root / "output_simulation/evaluation" /
              f"home_reach_{datetime.now():%Y%m%d_%H%M%S}_{token}").expanduser().resolve()
    if output.exists() or output == checkpoint or checkpoint in output.parents:
        raise ValueError("Output must be a new directory outside the checkpoint.")
    return {
        "schema_version": 2, "protocol": "zero_home_3s_target_10s",
        "project_root": str(project_root), "checkpoint": str(checkpoint),
        "checkpoint_config": model_config, "reward_config_path": str(reward_path.resolve()),
        "resolved_reward": resolved_reward, "output_directory": str(output),
        "device": device, "gazebo_gui": gui, "ros_domain_id": ros_domain_id,
        "gz_partition": f"so101_home_reach_eval_{token}", "episodes": episodes, "seed": seed,
        "target_position_world_m": list(target_position), "target_quaternion_xyzw": [0., 0., 0., 1.],
        "policy_hz": POLICY_HZ, "target_delay_steps": TARGET_DELAY_STEPS,
        "total_steps": TOTAL_STEPS, "startup_feedback_timeout_s": startup_timeout_s,
        "initial_pose": "zero", "task_mode": "pick_place",
        "controller_readiness_required": True,
        "exploration_enabled": False, "image_augmentation_enabled": False,
        "environment_randomization_enabled": False, "optimizer_enabled": False,
        "assisted_grasp_enabled": False, "physics_mode": "unassisted",
        "execution": {key: model_config.get(key) for key in
                      ("chunk_size", "n_action_steps", "temporal_ensemble_coeff")},
        "timing": "Wait for active position controller and matched command subscriber, then settle at ZERO before t=0; "
                  "spawn request after 90 steps; stop after 390 steps. "
                  "Camera/target feedback latency is recorded separately. No HOME assistance after t=0.",
    }


def _emit(stream: Any, event: str, **data: Any) -> None:
    row = {"event": event, **data}
    text = json.dumps(row, allow_nan=False, default=_json_default)
    stream.write(text + "\n")
    print(text, flush=True)


def _sync(stream: Any) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _run_episode(client: Any, adapter: Any, plan: dict[str, Any], destination: Path,
                 *, episode_index: int) -> dict[str, Any]:
    from so101_capstone.evaluation.home_reach_metrics import measure, summarize

    destination.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    prepared = None
    initial = None
    observation = None
    nearest_observation = nearest_row = None
    spawn = None
    error_info = None
    end_reason = "time_budget"
    with (destination / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as steps, \
         (destination / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events:
        try:
            client.prepare_episode_async(episode_index, task_mode="pick_place", scene_seed=plan["seed"])
            # Controller discovery/activation is setup time, before the old
            # bounded reset + camera preparation budget and before step zero.
            prepared = _require(client.receive(timeout_s=120 + plan["startup_feedback_timeout_s"]), "episode_prepared")
            _write_json(destination / "prepared.json", prepared)
            observation = client.observation_from_metadata(prepared["observation"])
            initial = measure(prepared["info"], plan["resolved_reward"])
            if prepared["info"].get("evaluation_initial_pose") != "zero":
                raise RuntimeError("Worker did not confirm the evaluation ZERO initial pose.")
            readiness = prepared["info"].get("evaluation_controller_readiness") or {}
            if (readiness.get("ready") is not True
                    or readiness.get("controller_name") != "position_controller"
                    or readiness.get("controller_state") != "active"
                    or readiness.get("command_subscription_count", 0) < 1):
                raise RuntimeError("Worker did not confirm an active, connected position controller; test clock not started.")
            _emit(events, "controller_ready_before_test", readiness=readiness)
            origin = float(prepared["info"]["sim_time_s"])
            _save_frame(destination, "initial", observation, initial)
            _emit(events, "policy_started", elapsed_s=0., initial=initial)
            adapter.reset()
            first_target_feedback = False
            first_home = False
            for completed in range(plan["total_steps"]):
                if completed == plan["target_delay_steps"]:
                    _save_frame(destination, "before_target", observation, rows[-1]["diagnostics"])
                    # This service request does not replace an ACT action or
                    # execute any hidden settling/hold steps inside the test.
                    _emit(events, "target_requested", completed_steps=completed,
                          elapsed_s=completed / POLICY_HZ, position_world_m=plan["target_position_world_m"])
                    _sync(events)
                    client.spawn_target_async(plan["target_position_world_m"], plan["target_quaternion_xyzw"])
                    spawn = _require(client.receive(timeout_s=30), "target_spawned")
                    _emit(events, "target_service_result", elapsed_s=completed / POLICY_HZ, response=spawn)
                    if not spawn["success"]:
                        raise RuntimeError(f"Target creation failed: {spawn}")
                action = adapter.select_action(observation)
                client.step_async(action.action.joint_targets)
                response = _require(client.receive(timeout_s=30), "step_result")
                next_observation = client.observation_from_metadata(response["observation"])
                info = response["info"]
                diagnostics = measure(info, plan["resolved_reward"])
                row = {
                    "step_index": completed + 1, "elapsed_s": (completed + 1) / POLICY_HZ,
                    "measured_sim_elapsed_s": float(info["sim_time_s"]) - origin,
                    "raw_policy_positions": action.raw_policy_positions.tolist(),
                    "controller_targets_rad": action.action.joint_targets.tolist(),
                    "joint_positions_before_rad": observation.joint_positions.tolist(),
                    "joint_positions_after_rad": next_observation.joint_positions.tolist(),
                    "joint_velocities_after_rad_s": next_observation.joint_velocities.tolist(),
                    "reward": response["reward"], "terminated": response["terminated"],
                    "truncated": response["truncated"], "diagnostics": diagnostics, "info": info,
                }
                steps.write(json.dumps(row, allow_nan=False, default=_json_default) + "\n")
                rows.append(row)
                observation = next_observation
                # The 33/33/34 ms schedule differs from nominal i/30 by <1ms.
                if abs(row["measured_sim_elapsed_s"] - row["elapsed_s"]) > .002:
                    raise RuntimeError("Simulation clock diverged from the deterministic 30 Hz test timeline.")
                if diagnostics["evidence_valid"] and diagnostics["at_home"] and not first_home:
                    first_home = True
                    _save_frame(destination, "first_home", observation, diagnostics)
                    _emit(events, "first_home_position", elapsed_s=row["elapsed_s"], diagnostics=diagnostics)
                if diagnostics["target_present"] and not first_target_feedback:
                    first_target_feedback = True
                    _save_frame(destination, "first_target_feedback", observation, diagnostics)
                    _emit(events, "first_target_feedback", elapsed_s=row["elapsed_s"],
                          evidence_sim_time_s=diagnostics["evidence_sim_time_s"],
                          camera_source_times_s=info.get("camera_source_times_s"))
                if diagnostics["target_evidence_valid"] and diagnostics["distance_m"] is not None:
                    if nearest_row is None or diagnostics["distance_m"] < nearest_row["diagnostics"]["distance_m"]:
                        nearest_observation, nearest_row = observation, row
                if (completed + 1) % POLICY_HZ == 0:
                    _sync(steps)
                    _emit(events, "progress", step=completed + 1, elapsed_s=row["elapsed_s"],
                          home_error_rad=diagnostics["home_error_rad"],
                          distance_m=diagnostics["distance_m"], task_phase=diagnostics["task_phase"])
                if response["terminated"] or response["truncated"]:
                    if completed + 1 < plan["total_steps"]:
                        end_reason = str(info.get("failure_reason") or info.get("task_outcome") or "early_terminal")
                    break
        except BaseException as error:
            end_reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "runtime_error"
            error_info = {"type": type(error).__name__, "message": str(error), "completed_steps": len(rows)}
            _write_json(destination / "failure.json", error_info)
            raise
        finally:
            # Persist all confirmed transitions and partial outcomes BEFORE
            # returning to the caller that closes this episode's Gazebo world.
            _sync(steps)
            _sync(events)
            metrics = {} if initial is None else summarize(
                rows, initial=initial, reward_config=plan["resolved_reward"],
                target_delay_s=plan["target_delay_steps"] / POLICY_HZ)
            summary = {
                "episode_index": episode_index, "complete_timeline": len(rows) == plan["total_steps"] and error_info is None,
                "end_reason": end_reason, "completed_steps": len(rows),
                "elapsed_s": len(rows) / POLICY_HZ, "target_spawn": spawn,
                "controller_readiness": None if prepared is None else prepared["info"].get("evaluation_controller_readiness"),
                "error": error_info, **metrics,
            }
            _write_json(destination / "summary.json", summary)
            if observation is not None:
                _save_frame(destination, "final", observation, rows[-1]["diagnostics"] if rows else initial)
            if nearest_row is not None:
                _save_frame(destination, "nearest_target", nearest_observation, nearest_row["diagnostics"])
    return summary


def execute_plan(plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
    from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
    from .act_ppo_multiworld_worker import GazeboWorkerClient, GazeboWorkerConfig

    if plan.get("assisted_grasp_enabled", False) is not False:
        raise ValueError("The HOME/reach evaluator uses unassisted physics only.")
    assert_resources_unused(plan)
    output = Path(plan["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    _write_json(output / "resolved_reward.json", plan["resolved_reward"])
    original_env = {key: os.environ.get(key) for key in
                    ("SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "SO101_IMAGE_AUGMENTATION_ENABLED",
                     "SO101_ASSISTED_GRASP_ENABLED")}
    completed = []
    try:
        for key in original_env:
            os.environ[key] = "false"
        torch.set_num_threads(4)
        loaded = load_act_checkpoint(Path(plan["checkpoint"]), device=plan["device"])
        loaded.policy.eval()
        adapter = ActPolicyAdapter(loaded.policy, device=loaded.device, max_joint_delta_rad=None)
        digest = hashlib.sha256()
        with (Path(plan["checkpoint"]) / "model.safetensors").open("rb") as model:
            for chunk in iter(lambda: model.read(1024 * 1024), b""):
                digest.update(chunk)
        _write_json(output / "model_identity.json", {"checkpoint": plan["checkpoint"], "sha256": digest.hexdigest()})
        for index in range(1, plan["episodes"] + 1):
            random.seed(plan["seed"])
            np.random.seed(plan["seed"])
            torch.manual_seed(plan["seed"])
            destination = output / f"episode_{index:03d}"
            config = GazeboWorkerConfig(
                world_index=0, ros_domain_id=plan["ros_domain_id"], gz_partition=plan["gz_partition"],
                gazebo_gui=plan["gazebo_gui"], project_root=Path(plan["project_root"]),
                output_directory=output / f"world_run_{index:03d}",
                startup_feedback_timeout_s=plan["startup_feedback_timeout_s"], startup_max_attempts=1,
                restart_max_attempts=1, max_episode_steps=plan["total_steps"], reward_profile="pick_place_v2",
                reward_config=plan["resolved_reward"], n_action_steps=int(loaded.config.n_action_steps),
                evaluation_mode=True, observe_after_success_s=plan["total_steps"] / POLICY_HZ,
            )
            # The client owns only the reserved evaluation world. It is closed
            # after _run_episode's finally block has persisted measurements.
            with GazeboWorkerClient(config) as client:
                _require(client.start(), "ready")
                with torch.inference_mode():
                    summary = _run_episode(client, adapter, plan, destination, episode_index=index)
                stopped = _require(client.stop_world(), "world_stopped")
                _write_json(destination / "world_closed.json", stopped)
            completed.append(summary)
            _write_json(output / "summary.json", {"complete": False, "episodes": completed})
    except BaseException as error:
        _write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                             "completed_episodes": len(completed)})
        raise
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    result = {"complete": all(row["complete_timeline"] for row in completed), "episodes": completed,
              "assisted_grasp_enabled": False, "physics_mode": "unassisted"}
    _write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2, default=_json_default), flush=True)
    print(f"Evaluation output: {output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="ACT pretrained_model directory")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action="store_true", help="GUI is enabled by default")
    parser.add_argument("--ros-domain-id", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=1, help="Repeat the same scene in fresh worlds")
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--target-position", type=float, nargs=3, default=(.25, 0., .768), metavar=("X", "Y", "Z"))
    parser.add_argument("--startup-timeout-s", type=float, default=60.)
    parser.add_argument("--execute", action="store_true", help="Start bounded simulation inference")
    args = parser.parse_args()
    plan = build_plan(args.checkpoint, project_root=args.project_root, output_directory=args.output_directory,
                      reward_config=args.reward_config, device=args.device, gui=not args.headless,
                      ros_domain_id=args.ros_domain_id, episodes=args.episodes, seed=args.seed,
                      target_position=tuple(args.target_position), startup_timeout_s=args.startup_timeout_s)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute and not execute_plan(plan)["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
