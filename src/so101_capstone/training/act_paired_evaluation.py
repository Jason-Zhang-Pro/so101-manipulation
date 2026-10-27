"""Paired deterministic ACT evaluation in separately reserved Gazebo resources.

The default command only prints a plan. --execute runs inference, never PPO.
Scenes and reward are copied from an existing run; each policy sees matching
scene seeds, target spawn delays/poses, camera processing and action scheduling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid
from typing import Any


_STAGE_KEYS = ("precise_reach", "pregrasp_ready", "grasp_valid", "lifted_once",
               "carried_into_goal", "placed_stable", "robot_at_home")


def _assisted_grasp_from_signature(signature: dict[str, Any]) -> bool:
    """Resolve source physics explicitly; never infer it from ambient environment."""
    if "assisted_grasp" not in signature:
        return False
    expected = {
        "schema": 1, "parent_model": "so101", "parent_link": "gripper_link",
        "child_link": "cube_link", "release_delta_rad": math.radians(2.0),
        "release_reference": "measured_confirmation", "release_requires_measured_opening": True,
        "confirmation": "real_contact_grasp", "recorded_angles": "command_and_measured",
        "context_dim": 6,
    }
    if signature["assisted_grasp"] != expected:
        raise ValueError("Unsupported assisted-grasp execution contract; explicit reconciliation required.")
    return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    # Worker measurements may contain NumPy scalars/arrays; do not stringify
    # them because downstream metrics must retain their numeric structure.
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def build_plan(run_directory: Path, updated_checkpoint: Path, output_directory: Path,
               *, project_root: Path, ros_domain_id: int = 120,
               device: str = "cpu", reach_scenes: int = 3) -> dict[str, Any]:
    """Build reproducible cases without importing Torch, ROS or loading weights."""
    if not 0 <= ros_domain_id <= 232:
        raise ValueError("ROS domain must be within [0, 232].")
    if not 1 <= reach_scenes <= 3:
        raise ValueError("reach_scenes must be in [1, 3].")
    run_directory = run_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise ValueError("Evaluation output must be a new directory.")
    if output_directory == run_directory or run_directory in output_directory.parents:
        raise ValueError("Evaluation output must be outside the source training run.")
    config = _read_json(run_directory / "config.json")
    reward = _read_json(run_directory / "resolved_reward.json")
    if reward != config["resolved_reward"]:
        raise ValueError("Run config and saved resolved reward differ.")
    signature = config["execution_signature"]
    assisted_grasp = _assisted_grasp_from_signature(signature)
    if signature["environment_randomization"].get("enabled"):
        raise ValueError("Paired evaluator currently requires randomization-disabled source runs.")
    if config["gazebo"]["reward_profile"] != "pick_place_v2":
        raise ValueError("Paired evaluator requires pick_place_v2 task measurements.")
    if signature["policy_hz"] != 30.0:
        raise ValueError("Worker interface currently has a fixed 30 Hz control schedule.")
    reserved = {int(item["ros_domain_id"]) for item in config["multiworld"]["worlds"]}
    if ros_domain_id in reserved:
        raise ValueError("Evaluation domain overlaps source training resources.")
    initial = Path(config["gazebo"]["act_checkpoint"]).resolve()
    updated = updated_checkpoint.resolve()
    for checkpoint in (initial, updated):
        for name in ("config.json", "model.safetensors"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        model_config = _read_json(checkpoint / "config.json")
        for name in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff"):
            value = model_config.get(name)
            if name == "n_action_steps" and config["gazebo"].get("act_max_action_steps") is not None:
                value = min(value, config["gazebo"]["act_max_action_steps"])
            if value != signature[name]:
                raise ValueError(f"Checkpoint {name} does not match run execution: {checkpoint}")
    records = [_read_json(path) for path in sorted((run_directory / "episodes").glob("episode_*.json"))]
    cases = []
    for mode, count in (("idle_empty", 1), ("idle_completed", 1), ("pick_place", reach_scenes)):
        matching = [record for record in records if record.get("valid", True) and record["task_mode"] == mode]
        if len(matching) < count:
            raise ValueError(f"Not enough completed source episodes for {mode}.")
        for ordinal, record in enumerate(matching[:count]):
            seed = int(config["gazebo"]["seed"] + record["episode_attempt_index"] * 1009
                       + record["world_index"] * 100003)
            if "scene_seed" in record and record["scene_seed"] != seed:
                raise ValueError("Saved scene seed disagrees with source run convention.")
            case = {"name": f"{mode}_{ordinal}", "task_mode": mode, "scene_seed": seed,
                    "source_episode_index": record["episode_index"],
                    "source_episode_attempt_index": record["episode_attempt_index"],
                    "target_delay_steps": record["target"]["request_completed_step"],
                    "target_position_world_m": record.get("target_position_world_m"),
                    "target_quaternion_xyzw": record.get("target_quaternion_xyzw")}
            if mode == "pick_place" and (case["target_delay_steps"] is None or case["target_position_world_m"] is None):
                raise ValueError("Pick/place source case lacks its exact target spawn contract.")
            cases.append(case)
    checkpoint_tag = updated.parent.name
    update_number = checkpoint_tag.removeprefix("update_")
    updated_name = f"update_{int(update_number)}" if checkpoint_tag.startswith("update_") and update_number.isdigit() else "updated"
    return {"schema_version": 1, "source_run": str(run_directory), "output_directory": str(output_directory),
            "project_root": str(project_root.resolve()), "device": device,
            "ros_domain_id": ros_domain_id, "gz_partition": f"so101_paired_eval_{uuid.uuid4().hex}",
            "gazebo_gui": False, "exploration_enabled": False, "optimizer_enabled": False,
            "assisted_grasp_enabled": assisted_grasp,
            "physics_mode": "rigid_grasp_assisted" if assisted_grasp else "unassisted",
            "inference": "standalone ACT native select_action; eval mode; episode-reset temporal history",
            "max_steps_per_episode": config["gazebo"]["max_steps_per_episode"],
            "act_max_action_steps": config["gazebo"].get("act_max_action_steps"),
            "startup_feedback_timeout_s": config["gazebo"]["startup_feedback_timeout_s"],
            "execution_signature": signature, "resolved_reward": reward,
            "resolved_reward_sha256": hashlib.sha256(json.dumps(reward, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "variants": [{"name": "initial", "checkpoint": str(initial)},
                         {"name": updated_name, "checkpoint": str(updated)}],
            "cases": cases, "total_episode_count": 2 * len(cases),
            "limitations": ["Same seeds do not guarantee bitwise-identical Gazebo rendering or reset feedback.",
                            "This small fixed-scene comparison is not a generalization success-rate estimate.",
                            "Original saved reward is used even if repository reward defaults have changed."]}


def assert_resources_unused(plan: dict[str, Any], snapshots: Any = None) -> None:
    """Refuse conflicts; never invoke preflight process cleanup for evaluation."""
    if snapshots is None:
        from .act_ppo_multiworld_cleanup import scan_processes
        snapshots = scan_processes()
    for item in snapshots:
        env = item.environment
        if (env.get("ROS_DOMAIN_ID") == str(plan["ros_domain_id"])
                or env.get("GZ_PARTITION") == plan["gz_partition"]
                or env.get("IGN_PARTITION") == plan["gz_partition"]):
            raise RuntimeError(f"Evaluation resource already in use by PID {item.pid}; refusing to start.")


def _require(response: Any, expected: str) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("type") != expected:
        raise RuntimeError(f"Expected {expected}; worker returned {response!r}")
    return response


def summarize_steps(records: list[dict[str, Any]], reward_config: dict[str, Any],
                    task_mode: str) -> dict[str, Any]:
    """Keep physical outcomes separate from reward and preplaced goal objects."""
    import math
    home = reward_config["home_joint_positions_rad"]
    tail = records[-min(100, len(records)):]
    errors = [[abs(q - reference) for q, reference in zip(row["joint_positions_after_rad"], home)] for row in tail]
    distances = [(row["step_index"], row["info"]["distance_m"]) for row in records
                 if row["info"].get("physical_target_present", False) and row["info"].get("evidence_valid", True)
                 and row["info"].get("distance_m") is not None]
    nearest = min(distances, key=lambda item: item[1]) if distances else None
    return {"step_count": len(records), "total_reward": sum(row["reward"] for row in records),
            "minimum_target_distance_m": None if nearest is None else nearest[1],
            "nearest_step_index": None if nearest is None else nearest[0],
            "tail_mean_absolute_home_error_deg_per_joint": [math.degrees(sum(row[j] for row in errors) / len(errors)) for j in range(6)],
            "stage_reached": {key: any(bool(row["info"].get("pregrasp_ready" if key == "precise_reach" else key, False)) for row in records) for key in _STAGE_KEYS},
            "task_success": records[-1]["info"].get("task_outcome") == "success",
            "nearest_geometry": None if nearest is None else frame_geometry(records[nearest[0] - 1], reward_config),
            "learned_place_observed": task_mode == "pick_place" and any(bool(row["info"].get("placed_stable")) for row in records),
            "close_gate_positive_steps": sum(row["info"].get("grasp_close_gate", 0) > 0 for row in records),
            "final_info": records[-1]["info"], "terminated": records[-1]["terminated"],
            "truncated": records[-1]["truncated"]}


def frame_geometry(row: dict[str, Any], reward_config: dict[str, Any]) -> dict[str, Any]:
    """Include both aligned task context and latest-feedback FK, without conflating them."""
    info = row["info"]
    context = info.get("critic_task_context", [])
    return {"step_index": row["step_index"], "distance_m": info.get("distance_m"),
            "aligned_target_minus_grasp_xyz_m": context[38:41] if len(context) >= 41 else None,
            "latest_feedback_privileged_state": info.get("privileged_state"),
            "joint_positions_rad": row["joint_positions_after_rad"],
            "opening_from_latest_joint_feedback_rad": row["joint_positions_after_rad"][-1] - reward_config["home_joint_positions_rad"][-1],
            "evidence_sim_time_s": info.get("evidence_sim_time_s"),
            "joint_source_sim_time_s": info.get("joint_source_sim_time_s"),
            "camera_source_times_s": info.get("camera_source_times_s")}


def _save_frame(destination: Path, name: str, observation: Any, metadata: Any) -> None:
    from PIL import Image
    Image.fromarray(observation.top_image).save(destination / f"{name}_top.png")
    Image.fromarray(observation.gripper_image).save(destination / f"{name}_gripper.png")
    _write_json(destination / f"{name}.json", metadata)


def _run_episode(client: Any, adapter: Any, case: dict[str, Any], plan: dict[str, Any],
                 destination: Path) -> dict[str, Any]:
    import numpy as np
    destination.mkdir(parents=True)
    client.prepare_episode_async(case["source_episode_attempt_index"], task_mode=case["task_mode"], scene_seed=case["scene_seed"])
    prepared = _require(client.receive(timeout_s=90), "episode_prepared")
    observation = client.observation_from_metadata(prepared["observation"])
    initial_observation = observation
    _save_frame(destination, "initial", observation, {"joint_positions_rad": observation.joint_positions.tolist(), "info": prepared["info"]})
    nearest_observation = None
    nearest_row = None
    adapter.reset()
    records = []
    spawned = False
    try:
        with (destination / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            for step in range(plan["max_steps_per_episode"]):
                if case["task_mode"] == "pick_place" and step == case["target_delay_steps"]:
                    client.spawn_target_async(case["target_position_world_m"], case["target_quaternion_xyzw"])
                    spawn = _require(client.receive(timeout_s=30), "target_spawned")
                    if not spawn["success"]:
                        raise RuntimeError(f"Target spawn failed: {spawn}")
                    spawned = True
                action = adapter.select_action(observation)
                client.step_async(action.action.joint_targets)
                response = _require(client.receive(timeout_s=30), "step_result")
                next_observation = client.observation_from_metadata(response["observation"])
                row = {"step_index": step + 1, "raw_policy_positions": action.raw_policy_positions.tolist(),
                       "controller_targets_rad": action.action.joint_targets.tolist(),
                       "joint_positions_before_rad": observation.joint_positions.tolist(),
                       "joint_positions_after_rad": next_observation.joint_positions.tolist(),
                       "joint_velocities_after_rad_s": next_observation.joint_velocities.tolist(),
                       "reward": response["reward"], "terminated": response["terminated"],
                       "truncated": response["truncated"], "info": response["info"]}
                stream.write(json.dumps(row, allow_nan=False, default=_json_default) + "\n")
                records.append(row)
                distance = row["info"].get("distance_m")
                if (row["info"].get("physical_target_present", False) and row["info"].get("evidence_valid", True)
                        and distance is not None and (nearest_row is None or distance < nearest_row["info"]["distance_m"])):
                    nearest_observation, nearest_row = next_observation, row
                observation = next_observation
                if step % 150 == 149:
                    print(json.dumps({"case": case["name"], "step": step + 1,
                                      "distance_m": response["info"].get("distance_m")}), flush=True)
                if response["terminated"] or response["truncated"]:
                    break
            stream.flush()
            os.fsync(stream.fileno())
        np.savez_compressed(destination / "observations.npz", initial_top=initial_observation.top_image,
                            initial_gripper=initial_observation.gripper_image,
                            initial_q=initial_observation.joint_positions, final_top=observation.top_image,
                            final_gripper=observation.gripper_image, final_q=observation.joint_positions)
        _save_frame(destination, "final", observation, frame_geometry(records[-1], plan["resolved_reward"]))
        if nearest_row is not None:
            _save_frame(destination, "nearest", nearest_observation, frame_geometry(nearest_row, plan["resolved_reward"]))
        summary = {"case": case, "prepared": prepared, "target_spawned": spawned,
                   **summarize_steps(records, plan["resolved_reward"], case["task_mode"])}
        _write_json(destination / "summary.json", summary)
        return summary
    except BaseException as error:
        _write_json(destination / "failure.json", {"type": type(error).__name__, "message": str(error), "completed_steps": len(records)})
        raise


def execute_plan(plan: dict[str, Any]) -> dict[str, Any]:
    # Imports remain below explicit --execute so planning needs no ML/ROS stack.
    import numpy as np
    import torch
    from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
    from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
    from so101_capstone.environments.simulation_step_coordinator import SimulationStepCoordinatorConfig
    from .act_ppo_gazebo import _configure_act_replanning
    from .act_ppo_multiworld_worker import GazeboWorkerClient, GazeboWorkerConfig

    assisted_grasp = _assisted_grasp_from_signature(plan["execution_signature"])
    if plan.get("assisted_grasp_enabled", assisted_grasp) is not assisted_grasp:
        raise ValueError("Evaluation assisted-grasp mode differs from the saved source contract.")
    if asdict(SimulationStepCoordinatorConfig()) != plan["execution_signature"]["step_coordinator_config"]:
        raise ValueError("Current camera/timing contract differs from saved training; explicit reconciliation required.")
    assert_resources_unused(plan)
    output = Path(plan["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    _write_json(output / "resolved_reward.json", plan["resolved_reward"])
    original_env = {key: os.environ.get(key) for key in (
        "SO101_ENVIRONMENT_RANDOMIZATION_ENABLED", "SO101_ASSISTED_GRASP_ENABLED")}
    completed = []
    try:
        os.environ["SO101_ENVIRONMENT_RANDOMIZATION_ENABLED"] = "false"
        os.environ["SO101_ASSISTED_GRASP_ENABLED"] = str(assisted_grasp).lower()
        torch.set_num_threads(4)
        for variant in plan["variants"]:
            loaded = load_act_checkpoint(Path(variant["checkpoint"]), device=plan["device"])
            _configure_act_replanning(loaded, plan["act_max_action_steps"])
            loaded.policy.eval()
            adapter = ActPolicyAdapter(loaded.policy, device=loaded.device, max_joint_delta_rad=None)
            config = GazeboWorkerConfig(world_index=0, ros_domain_id=plan["ros_domain_id"],
                gz_partition=plan["gz_partition"], gazebo_gui=False, project_root=Path(plan["project_root"]),
                output_directory=output / variant["name"], startup_feedback_timeout_s=plan["startup_feedback_timeout_s"],
                startup_max_attempts=1, restart_max_attempts=1, max_episode_steps=plan["max_steps_per_episode"],
                reward_profile="pick_place_v2", reward_config=plan["resolved_reward"],
                n_action_steps=int(loaded.config.n_action_steps))
            with GazeboWorkerClient(config) as client:
                for index, case in enumerate(plan["cases"]):
                    random.seed(case["scene_seed"])
                    np.random.seed(case["scene_seed"] % 2**32)
                    torch.manual_seed(case["scene_seed"])
                    if index == 0:
                        _require(client.start(), "ready")
                    else:
                        _require(client.restart(), "ready")
                    print(json.dumps({"variant": variant["name"], "case": case["name"], "status": "started"}), flush=True)
                    summary = _run_episode(client, adapter, case, plan, output / variant["name"] / case["name"])
                    stopped = _require(client.stop_world(), "world_stopped")
                    if stopped.get("world_index") != config.world_index:
                        raise RuntimeError(
                            f"Expected world {config.world_index} cleanup acknowledgement; "
                            f"worker returned {stopped!r}"
                        )
                    completed.append({"variant": variant["name"], **summary})
                    _write_json(output / "comparison.json", {"complete": False, "episodes": completed})
                    print(json.dumps({"variant": variant["name"], "case": case["name"], "status": "completed",
                                      "minimum_distance_m": summary["minimum_target_distance_m"]}), flush=True)
            del adapter, loaded
            if plan["device"].startswith("cuda"):
                torch.cuda.empty_cache()
    except BaseException as error:
        _write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error), "completed_episodes": len(completed)})
        raise
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    comparison = {"complete": True, "episodes": completed,
                  "assisted_grasp_enabled": assisted_grasp,
                  "physics_mode": "rigid_grasp_assisted" if assisted_grasp else "unassisted"}
    _write_json(output / "comparison.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--updated-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--ros-domain-id", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reach-scenes", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--execute", action="store_true", help="Explicitly start isolated simulation inference.")
    args = parser.parse_args()
    plan = build_plan(args.run_directory, args.updated_checkpoint, args.output_directory,
                      project_root=args.project_root, ros_domain_id=args.ros_domain_id,
                      device=args.device, reach_scenes=args.reach_scenes)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.execute:
        execute_plan(plan)


if __name__ == "__main__":
    main()
