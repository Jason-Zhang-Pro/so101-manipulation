"""Bounded, inference-only comparison with the saved S0/S1 stage lifecycle.

Without --execute this only validates local JSON/files and prints a plan.
Source conda lerobot, ROS Humble and the project workspace before execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "lerobot" / "src")]

def _read_json(path: Path) -> dict:
    # Do not import the training package during planning: its __init__ imports
    # Torch, even though the paired evaluator itself has lazy runtime imports.
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def build_plan(run_directory: Path, updated_checkpoint: Path, output_directory: Path,
               *, device: str = "cuda") -> dict:
    run = run_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError("Output must be a new directory outside the source training run.")
    config = _read_json(run / "config.json")
    reward = _read_json(run / "resolved_reward.json")
    stages = _read_json(run / "resolved_stages.json")
    if reward != config["resolved_reward"] or stages != config["resolved_stages"]:
        raise ValueError("Saved resolved reward/stages differ from the run configuration.")
    gazebo, signature = config["gazebo"], config["execution_signature"]
    act_contract = dict(config.get("checkpoint_act_config", signature))
    for key in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff"):
        if key not in act_contract:
            raise ValueError(f"Saved ACT execution contract is missing {key}.")
    effective_steps = act_contract["n_action_steps"]
    if gazebo.get("act_max_action_steps") is not None:
        effective_steps = min(effective_steps, gazebo["act_max_action_steps"])
    act_contract["n_action_steps"] = effective_steps
    if gazebo["reward_profile"] != "pick_place_v2" or gazebo["training_stages"] != "s0_s1":
        raise ValueError("This evaluator requires a saved pick_place_v2 S0/S1 run.")
    if signature["policy_hz"] != 30.0 or reward["control_hz"] != 30.0:
        raise ValueError("Stage evaluation requires the saved deterministic 30 Hz schedule.")
    if signature["environment_randomization"].get("enabled"):
        raise ValueError("This fixed-scene evaluation requires a physical-randomization-disabled run.")
    domain = 120
    if domain in {world["ros_domain_id"] for world in config["multiworld"]["worlds"]}:
        raise ValueError("Evaluation domain 120 overlaps the training run.")
    initial = Path(gazebo["act_checkpoint"]).expanduser().resolve()
    updated = updated_checkpoint.expanduser().resolve()
    if initial == updated:
        raise ValueError("Initial and updated checkpoints must be distinct.")
    variants = []
    for role, checkpoint in (("initial", initial), ("updated", updated)):
        for name in ("config.json", "model.safetensors"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        model = _read_json(checkpoint / "config.json")
        for key in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff"):
            value = model.get(key)
            if key == "n_action_steps" and gazebo.get("act_max_action_steps") is not None:
                value = min(value, gazebo["act_max_action_steps"])
            if value != act_contract[key]:
                raise ValueError(f"Checkpoint {key} differs from the saved execution contract: {checkpoint}")
        variants.append({"name": f"{role}_{checkpoint.parent.name}", "checkpoint": str(checkpoint)})
    cases = []
    for stage, mode in (("S0", "idle_empty"), ("S1", "pick_place")):
        budget = float(stages[f"{stage.lower()}_budget_s"])
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Saved stage budgets must be finite and positive.")
        if not math.isclose(budget, 5.0 if stage == "S0" else 10.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("This bounded evaluator requires saved S0/S1 budgets of 5/10 seconds.")
        steps = math.ceil(budget * 30.0)
        cases.append({
            "name": stage, "training_stage": stage, "task_mode": mode,
            "scene_seed": 1009, "source_episode_attempt_index": 0,
            "budget_s": budget, "expected_budget_steps": steps, "maximum_steps": steps + 1,
            "target_position_world_m": [.2476828788, -.0010926351, .768] if stage == "S1" else None,
            "target_quaternion_xyzw": [0., 0., 0., 1.] if stage == "S1" else None,
        })
    return {
        "schema_version": 1, "source_run": str(run), "project_root": str(PROJECT_ROOT),
        "output_directory": str(output), "device": device,
        "ros_domain_id": domain, "gz_partition": f"so101_stage_eval_{uuid.uuid4().hex}",
        "gazebo_gui": False, "exploration_enabled": False, "optimizer_enabled": False,
        "environment_randomization_enabled": False, "image_augmentation_enabled": False,
        "inference": "standalone ACT eval/select_action; no PPO exploration or optimizer",
        "initialization": "saved StageEvaluator and start_training_stage sensor-certified HOME lifecycle",
        "act_max_action_steps": gazebo.get("act_max_action_steps"),
        "effective_act_execution": {key: act_contract[key] for key in (
            "chunk_size", "n_action_steps", "temporal_ensemble_coeff")},
        "startup_feedback_timeout_s": 60.0, "execution_signature": signature,
        "resolved_reward": reward, "resolved_stages": stages,
        "resolved_reward_sha256": _digest(reward), "resolved_stages_sha256": _digest(stages),
        "variants": variants, "cases": cases, "total_episode_count": 4,
        "limitations": [
            "The prior ordinary-CLI tests used a different initialization and no StageEvaluator; their metrics are not paired with these tests.",
            "Only the two variants in this run share the same stage initialization, fixed target, seed, and disabled image augmentation.",
            "One S0 and one fixed-target S1 per model are diagnostic trajectories, not a generalization success-rate estimate.",
            "Identical seeds do not guarantee bitwise-identical Gazebo rendering or feedback.",
            "The 60-second startup feedback timeout is an evaluation wall-clock allowance; the saved physical/camera timing contract remains unchanged.",
        ],
    }


def _episode(client, adapter, case: dict, plan: dict, destination: Path) -> dict:
    records = []
    prepared = started = None
    try:
        client.prepare_episode_async(case["source_episode_attempt_index"],
                                     task_mode=case["task_mode"], scene_seed=case["scene_seed"])
        prepared = _require(client.receive(timeout_s=90), "episode_prepared")
        if case["training_stage"] == "S1":
            client.spawn_target_async(case["target_position_world_m"], case["target_quaternion_xyzw"])
            spawned = _require(client.receive(timeout_s=30), "target_spawned")
            if not spawned.get("success", False):
                raise RuntimeError(f"Fixed target spawn failed: {spawned}")
        client.start_training_stage_async(case["training_stage"], plan["resolved_stages"])
        started = _require(client.receive(timeout_s=90), "stage_started")
        if started["info"].get("training_stage") != case["training_stage"]:
            raise RuntimeError("Worker started the wrong training stage.")
        observation = client.observation_from_metadata(started["observation"])
        initial_row = {"step_index": 0, "info": started["info"],
                       "joint_positions_after_rad": observation.joint_positions.tolist()}
        _save_frame(destination, "initial", observation,
                    {**frame_geometry(initial_row, plan["resolved_reward"]), "info": started["info"]})
        _write_json(destination / "initialization.json", {"prepared": prepared, "stage_started": started})
        adapter.reset()
        nearest_observation = nearest_row = None
        with (destination / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            try:
                for index in range(case["maximum_steps"]):
                    action = adapter.select_action(observation)
                    client.step_async(action.action.joint_targets)
                    response = _require(client.receive(timeout_s=30), "step_result")
                    next_observation = client.observation_from_metadata(response["observation"])
                    row = {
                        "step_index": index + 1,
                        "raw_policy_positions": action.raw_policy_positions.tolist(),
                        "clipped_policy_positions": action.clipped_policy_positions.tolist(),
                        "controller_targets_rad": action.action.joint_targets.tolist(),
                        "joint_positions_before_rad": observation.joint_positions.tolist(),
                        "joint_positions_after_rad": next_observation.joint_positions.tolist(),
                        "joint_velocities_after_rad_s": next_observation.joint_velocities.tolist(),
                        "reward": response["reward"], "terminated": response["terminated"],
                        "truncated": response["truncated"], "info": response["info"],
                    }
                    stream.write(json.dumps(row, allow_nan=False, default=_json_default) + "\n")
                    records.append(row)
                    distance = row["info"].get("stage_reach_distance_m")
                    if (distance is not None and row["info"].get("evidence_valid", False)
                            and (nearest_row is None or distance < nearest_row["info"]["stage_reach_distance_m"])):
                        nearest_observation, nearest_row = next_observation, row
                    observation = next_observation
                    if index % 30 == 29 or response["terminated"] or response["truncated"]:
                        print(json.dumps({"stage": case["training_stage"], "step": index + 1,
                                          "stage_result": row["info"].get("stage_result"),
                                          "distance_m": distance}), flush=True)
                    if response["terminated"] or response["truncated"]:
                        break
            finally:
                stream.flush()
                os.fsync(stream.fileno())
        if not records or not (records[-1]["terminated"] or records[-1]["truncated"]):
            raise RuntimeError("Stage did not end within its bounded step allowance.")
        final = records[-1]["info"]
        if final.get("stage_result") not in {"SUCCESS", "TIMEOUT", "FAILURE"}:
            raise RuntimeError("Episode stopped without an authoritative stage result.")
        _save_frame(destination, "final", observation, frame_geometry(records[-1], plan["resolved_reward"]))
        if nearest_row is not None:
            _save_frame(destination, "nearest", nearest_observation,
                        frame_geometry(nearest_row, plan["resolved_reward"]))
        summary = {
            "case": case, "initialization": started["initialization"],
            **summarize_steps(records, plan["resolved_reward"], case["task_mode"]),
            "training_stage": case["training_stage"], "stage_result": final["stage_result"],
            "stage_success": final.get("stage_success", False),
            "stage_result_reason": final.get("stage_result_reason"),
            "stage_elapsed_s": final.get("stage_elapsed_s"),
            "stage_metrics": final.get("stage_metrics", {}),
            "stage_home_hold_s": final.get("stage_home_hold_s"),
            "stage_reach_hold_s": final.get("stage_reach_hold_s"),
            "maximum_steps": case["maximum_steps"],
        }
        _write_json(destination / "summary.json", summary)
        return summary
    except BaseException as error:
        _write_json(destination / "failure.json", {
            "type": type(error).__name__, "message": str(error), "completed_steps": len(records),
            "case": case, "prepared": prepared, "stage_started": started,
            "last_record": records[-1] if records else None,
        })
        raise


def execute_plan(plan: dict) -> dict:
    global _json_default, _require, _save_frame, _write_json, frame_geometry, summarize_steps
    import numpy as np
    import torch
    from so101_capstone.training.act_paired_evaluation import (
        _json_default, _require, _save_frame, _write_json,
        assert_resources_unused, frame_geometry, summarize_steps,
    )
    from so101_capstone.environments.simulation_step_coordinator import SimulationStepCoordinatorConfig
    from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
    from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
    from so101_capstone.training.act_ppo_gazebo import _configure_act_replanning
    from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerClient, GazeboWorkerConfig

    if asdict(SimulationStepCoordinatorConfig()) != plan["execution_signature"]["step_coordinator_config"]:
        raise ValueError("Current camera/timing contract differs from the saved training run.")
    assert_resources_unused(plan)
    output = Path(plan["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    _write_json(output / "resolved_reward.json", plan["resolved_reward"])
    _write_json(output / "resolved_stages.json", plan["resolved_stages"])
    os.environ["SO101_ENVIRONMENT_RANDOMIZATION_ENABLED"] = "false"
    os.environ["SO101_IMAGE_AUGMENTATION_ENABLED"] = "false"
    torch.set_num_threads(4)
    completed = []
    current_case = current_variant = None
    try:
        for variant in plan["variants"]:
            current_variant = variant["name"]
            loaded = load_act_checkpoint(Path(variant["checkpoint"]), device=plan["device"])
            _configure_act_replanning(loaded, plan["act_max_action_steps"])
            loaded.policy.eval()
            adapter = ActPolicyAdapter(loaded.policy, device=loaded.device, max_joint_delta_rad=None)
            for case in plan["cases"]:
                current_case = case["name"]
                destination = output / variant["name"] / case["name"]
                destination.mkdir(parents=True, exist_ok=False)
                random.seed(case["scene_seed"])
                np.random.seed(case["scene_seed"] % 2**32)
                torch.manual_seed(case["scene_seed"])
                # Every case gets a new process/world, including across models.
                config = GazeboWorkerConfig(
                    world_index=0, ros_domain_id=plan["ros_domain_id"], gz_partition=plan["gz_partition"],
                    gazebo_gui=False, project_root=Path(plan["project_root"]), output_directory=destination,
                    startup_feedback_timeout_s=plan["startup_feedback_timeout_s"],
                    startup_max_attempts=1, restart_max_attempts=1,
                    max_episode_steps=case["maximum_steps"], reward_profile="pick_place_v2",
                    reward_config=plan["resolved_reward"], n_action_steps=int(loaded.config.n_action_steps),
                )
                with GazeboWorkerClient(config) as client:
                    _require(client.start(), "ready")
                    print(json.dumps({"variant": variant["name"], "stage": case["name"],
                                      "status": "started"}), flush=True)
                    summary = _episode(client, adapter, case, plan, destination)
                    stopped = _require(client.stop_world(), "world_stopped")
                    if stopped.get("world_index") != 0:
                        raise RuntimeError("Evaluation world cleanup acknowledgement is inconsistent.")
                summary["world_cleanup_confirmed"] = True
                _write_json(destination / "summary.json", summary)
                completed.append({"variant": variant["name"], **summary})
                _write_json(output / "comparison.json", {"complete": False, "episodes": completed})
                print(json.dumps({"variant": variant["name"], "stage": case["name"],
                                  "status": "completed", "result": summary["stage_result"]}), flush=True)
            del adapter, loaded
            if plan["device"].startswith("cuda"):
                torch.cuda.empty_cache()
    except BaseException as error:
        _write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "variant": current_variant, "stage": current_case, "completed_episodes": len(completed),
        })
        _write_json(output / "comparison.json", {"complete": False, "episodes": completed})
        raise
    comparison = {"complete": True, "episodes": completed}
    _write_json(output / "comparison.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--updated-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="Start four bounded isolated simulation episodes.")
    args = parser.parse_args()
    plan = build_plan(args.run_directory, args.updated_checkpoint, args.output_directory, device=args.device)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.execute:
        execute_plan(plan)


if __name__ == "__main__":
    main()
