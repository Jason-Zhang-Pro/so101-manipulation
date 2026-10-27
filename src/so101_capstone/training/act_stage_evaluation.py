"""Independent ACT policy tests for S0--S5, using the training stage lifecycle.

Planning does not load weights or start a world. --execute runs inference only:
one fresh world per stage/episode, physical preparation, then native ACT actions.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any
import uuid

from .act_paired_evaluation import (
    _assisted_grasp_from_signature, _json_default, _read_json, _require,
    _save_frame, _write_json, assert_resources_unused,
)

STAGES = tuple(f"S{i}" for i in range(6))
POLICY_HZ = 30


def _source_run(checkpoint: Path) -> tuple[Path | None, dict]:
    # Read only the standard export's enclosing run, never load training_state.pt.
    if checkpoint.name == "pretrained_model":
        run = checkpoint.parents[2] if checkpoint.parent.parent.name == "checkpoints" else checkpoint.parent
        if (run / "config.json").is_file():
            data = _read_json(run / "config.json")
            if isinstance(data.get("gazebo"), dict):
                return run, data
    return None, {}


def build_plan(
    checkpoint: Path, *, project_root: Path, stages: tuple[str, ...] = STAGES,
    episodes_per_stage: int = 1, output_directory: Path | None = None,
    stage_config: Path | None = None, reward_config: Path | None = None,
    assisted_grasp: bool | None = None, device: str = "cuda", gui: bool = True,
    ros_domain_id: int = 125, seed: int = 1009,
    target_position: tuple[float, float, float] = (.26, 0., .768),
    startup_timeout_s: float = 60., preparation_timeout_s: float = 600.,
    budget_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Resolve exact test settings without mutating checkpoints or the workspace."""
    from so101_capstone.policies.act_temporal_config import validate_act_temporal_config
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    from so101_capstone.rewards.training_stages import StageTrainingConfig

    checkpoint, project_root = checkpoint.expanduser().resolve(), project_root.expanduser().resolve()
    stages = tuple(str(stage).upper() for stage in stages)
    if not stages or len(set(stages)) != len(stages) or any(s not in STAGES for s in stages):
        raise ValueError("Select distinct stages from S0 through S5.")
    if (isinstance(episodes_per_stage, bool) or not isinstance(episodes_per_stage, int)
            or episodes_per_stage < 1):
        raise ValueError("episodes_per_stage must be a positive integer.")
    if (isinstance(ros_domain_id, bool) or not isinstance(ros_domain_id, int)
            or not 0 <= ros_domain_id <= 232):
        raise ValueError("ROS domain must be within [0, 232].")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer within [0, 2**32).")
    for value in (startup_timeout_s, preparation_timeout_s):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Timeouts must be positive finite seconds.")
    if len(target_position) != 3 or not all(math.isfinite(v) for v in target_position):
        raise ValueError("Target position must contain three finite world coordinates.")
    if assisted_grasp is not None and type(assisted_grasp) is not bool:
        raise ValueError("assisted_grasp must be a bool or None.")
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    model = _read_json(checkpoint / "config.json")
    if model.get("type") != "act":
        raise ValueError("An ACT pretrained_model directory is required.")
    execution = validate_act_temporal_config(SimpleNamespace(**model))
    run, saved = _source_run(checkpoint)
    warnings = []
    saved_stages = checkpoint / "resolved_stages.json"
    saved_stage_payload = _read_json(saved_stages) if saved_stages.is_file() else {}
    # Old S0/S1 exports cannot supply later-stage start/success contracts.
    inherit_stages = stage_config is None and saved_stage_payload.get("enable_later_stage_context") is True
    stage_path = (stage_config or (saved_stages if inherit_stages else
                                  project_root / "config/act_ppo_stages_s0_s5.json")).expanduser().resolve()
    resolved = StageTrainingConfig.load(stage_path)
    if not resolved.enable_later_stage_context:
        raise ValueError("Stage evaluation requires enable_later_stage_context=true in the six-stage configuration.")
    if not inherit_stages and stage_config is None:
        warnings.append("No saved six-stage configuration: using current project S0--S5 rules.")
    budget_overrides = dict(budget_overrides or {})
    if any(key not in {"s0_budget_s", "s1_budget_s", "s2_budget_s", "s3_budget_s", "s4_budget_s"}
           for key in budget_overrides):
        raise ValueError("Budget overrides support S0--S4; S5 shares S0.")
    resolved = replace(resolved, **budget_overrides)
    saved_reward = checkpoint / "resolved_reward.json"
    reward_path = (reward_config or (saved_reward if inherit_stages and saved_reward.is_file() else
                                    project_root / "config/pick_place_reward.json")).expanduser().resolve()
    reward = PickPlaceRewardConfig.load(reward_path)
    if reward.control_hz != POLICY_HZ or reward.grasp_evidence_mode != "contact":
        raise ValueError("Stage evaluation requires 30 Hz and contact-certified grasp evidence.")
    signature = saved.get("execution_signature", {})
    if inherit_stages and saved:
        # Catch mismatched artifact bundles, without requiring reward hashes to
        # match current code: this evaluator explicitly uses current stage code.
        if saved.get("resolved_stages") != saved_stage_payload:
            raise ValueError("Checkpoint stages differ from the source run's resolved stages.")
        if reward_config is None and saved_reward.is_file() and saved.get("resolved_reward") != _read_json(saved_reward):
            raise ValueError("Checkpoint reward differs from the source run's resolved reward.")
    if saved:
        reserved = {int(w["ros_domain_id"]) for w in saved.get("multiworld", {}).get("worlds", [])}
        if ros_domain_id in reserved:
            raise ValueError("Evaluation ROS domain overlaps the source training domains.")
    if assisted_grasp is None:
        assisted_grasp = _assisted_grasp_from_signature(signature) if inherit_stages and saved else True
    if inherit_stages and saved and assisted_grasp and not signature.get("stage_grasp_attachment"):
        warnings.append("Source run predates persistent S2/S3 grasp; this test uses current stage attachment rules.")
    if inherit_stages and not saved:
        warnings.append("Source run metadata unavailable: assisted grasp defaults to current six-stage mode (enabled).")
    if signature.get("environment_randomization", {}).get("enabled"):
        warnings.append("Source run used physical randomization; evaluation uses the fixed scene specified here.")
    token = uuid.uuid4().hex[:8]
    output = (output_directory or project_root / "output_simulation/evaluation" /
              f"stages_{datetime.now():%Y%m%d_%H%M%S}_{token}").expanduser().resolve()
    if (output.exists() or output == checkpoint or checkpoint in output.parents
            or (run is not None and (output == run or run in output.parents))):
        raise ValueError("Output must be a new directory outside the checkpoint and source training run.")
    cases = []
    for stage in stages:
        for ordinal in range(1, episodes_per_stage + 1):
            budget = resolved.stage_budget_s(stage)
            cases.append(dict(name=f"{stage}/episode_{ordinal:03d}", stage=stage,
                              episode_index=ordinal, scene_seed=(seed + ordinal - 1) % 2**32,
                              task_mode="idle_empty" if stage == "S0" else "pick_place",
                              budget_s=budget, maximum_steps=math.ceil(budget * POLICY_HZ) + 2))
    return dict(
        schema_version=1, protocol="independent_s0_s5_policy",
        project_root=str(project_root), checkpoint=str(checkpoint),
        checkpoint_config=model, source_run=None if run is None else str(run),
        stage_config_path=str(stage_path), reward_config_path=str(reward_path),
        config_source="checkpoint" if inherit_stages else "explicit" if stage_config else "current_project",
        resolved_stages=resolved.to_dict(), resolved_reward=reward.to_dict(),
        budget_overrides=budget_overrides, stages=list(stages), cases=cases,
        episodes_per_stage=episodes_per_stage, total_episode_count=len(cases),
        output_directory=str(output), device=device, gazebo_gui=gui,
        ros_domain_id=ros_domain_id, gz_partition=f"so101_stage_eval_{token}", seed=seed,
        target_position_world_m=list(target_position), target_quaternion_xyzw=[0., 0., 0., 1.],
        startup_feedback_timeout_s=startup_timeout_s, preparation_timeout_s=preparation_timeout_s,
        policy_hz=POLICY_HZ, execution=execution,
        assisted_grasp_enabled=assisted_grasp,
        physics_mode="rigid_grasp_assisted" if assisted_grasp else "unassisted",
        attachment_policy="current stage rules: persistent S2/S3; measured opening elsewhere",
        implementation="current project stage, preparation and physics code; saved numeric configuration when available",
        exploration_enabled=False, optimizer_enabled=False, image_augmentation_enabled=False,
        environment_randomization_enabled=False, warnings=warnings,
        limitations=["Each stage starts independently after preparation; this is not a full-task success rate.",
                     "S0 starts at HOME, so it tests holding HOME rather than returning from an arbitrary pose.",
                     "Repeated episodes use the specified fixed target; this is not a random-scene generalization test.",
                     "Saved numeric settings do not restore historical implementations or camera/physics binaries."],
    )


def _emit(stream: Any, event: str, **data: Any) -> None:
    text = json.dumps(dict(event=event, **data), allow_nan=False, default=_json_default)
    stream.write(text + "\n")
    print(text, flush=True)


def _sync(stream: Any) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _run_episode(client: Any, adapter: Any, case: dict, plan: dict, destination: Path) -> dict:
    """Persist partial measurements before the caller closes the owned world."""
    from so101_capstone.evaluation.stage_policy_metrics import summarize_episode

    destination.mkdir(parents=True, exist_ok=False)
    records = []
    initial_info = {}
    observation = None
    nearest = None
    nearest_observation = None
    error_info = None
    policy_started = False
    operation = "prepare"
    with (destination / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as steps, \
         (destination / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events:
        try:
            _emit(events, "preparing", stage=case["stage"], budget_s=case["budget_s"])
            client.prepare_episode_async(case["episode_index"], task_mode=case["task_mode"],
                                         scene_seed=case["scene_seed"])
            response = client.receive(timeout_s=120 + plan["startup_feedback_timeout_s"])
            _write_json(destination / "prepared.json", response)
            _require(response, "episode_prepared")
            if case["stage"] != "S0":
                operation = "spawn_target"
                client.spawn_target_async(plan["target_position_world_m"], plan["target_quaternion_xyzw"])
                response = client.receive(timeout_s=30)
                _write_json(destination / "target_spawn.json", response)
                _require(response, "target_spawned")
                if not response.get("success"):
                    raise RuntimeError("Target creation failed.")
            operation = "stage_initialization"
            client.start_training_stage_async(case["stage"], plan["resolved_stages"])
            response = client.receive(timeout_s=plan["preparation_timeout_s"])
            _write_json(destination / "initialization.json", response)
            started = _require(response, "stage_started")
            initial_info = started["info"]
            if (initial_info.get("training_stage") != case["stage"]
                    or initial_info.get("start_valid") is not True
                    or initial_info.get("frame_zero") is not True
                    or initial_info.get("episode_step") != 0
                    or abs(float(initial_info.get("stage_elapsed_s", -1))) > 1e-9
                    or started.get("initialization", {}).get("valid") is not True):
                raise RuntimeError("Worker did not certify the requested stage at policy step zero.")
            observation = client.observation_from_metadata(started["observation"])
            _save_frame(destination, "initial", observation, initial_info)
            adapter.reset()
            origin = float(initial_info["sim_time_s"])
            policy_started = True
            operation = "policy_step"
            _emit(events, "policy_started", stage=case["stage"], initialization=started["initialization"])
            for index in range(1, case["maximum_steps"] + 1):
                action = adapter.select_action(observation)
                client.step_async(action.action.joint_targets)
                response = _require(client.receive(timeout_s=30), "step_result")
                after = client.observation_from_metadata(response["observation"])
                info = response["info"]
                row = dict(
                    step_index=index, elapsed_s=float(info["sim_time_s"]) - origin,
                    raw_policy_positions=action.raw_policy_positions.tolist(),
                    clipped_policy_positions=action.clipped_policy_positions.tolist(),
                    controller_targets_rad=action.action.joint_targets.tolist(),
                    joint_positions_before_rad=observation.joint_positions.tolist(),
                    joint_positions_after_rad=after.joint_positions.tolist(),
                    joint_velocities_after_rad_s=after.joint_velocities.tolist(),
                    reward=response["reward"], terminated=response["terminated"],
                    truncated=response["truncated"], info=info,
                )
                steps.write(json.dumps(row, allow_nan=False, default=_json_default) + "\n")
                records.append(row)
                observation = after
                if (info.get("training_stage") != case["stage"] or info.get("preparation_only")
                        or info.get("episode_step") != index):
                    raise RuntimeError("Policy feedback does not match the requested stage/step.")
                if abs(row["elapsed_s"] - index / POLICY_HZ) > .002:
                    raise RuntimeError("Simulation time diverged from the deterministic 30 Hz timeline.")
                distance = info.get("distance_m")
                if (info.get("physical_target_present") and info.get("evidence_valid")
                        and distance is not None and (nearest is None or distance < nearest["info"]["distance_m"])):
                    nearest, nearest_observation = row, after
                if index % POLICY_HZ == 0 or response["terminated"] or response["truncated"]:
                    _sync(steps)
                    _emit(events, "progress", stage=case["stage"], step=index, elapsed_s=row["elapsed_s"],
                          stage_result=info.get("stage_result"), distance_m=distance,
                          stage_metrics=info.get("stage_metrics", {}),
                          home_hold_s=info.get("stage_home_hold_s"))
                if response["terminated"] or response["truncated"]:
                    if info.get("stage_result") not in {"SUCCESS", "TIMEOUT", "FAILURE"}:
                        raise RuntimeError("Worker ended without an authoritative stage result.")
                    break
            else:
                raise RuntimeError("Stage did not terminate within its bounded step allowance.")
        except BaseException as error:
            error_info = dict(type=type(error).__name__, message=str(error), operation=operation,
                              completed_steps=len(records))
            _write_json(destination / "failure.json", error_info)
            _emit(events, "error", stage=case["stage"], **error_info)
            raise
        finally:
            _sync(steps)
            _sync(events)
            summary = summarize_episode(records, stage=case["stage"], initial_info=initial_info,
                                        reward_config=plan["resolved_reward"], error=error_info,
                                        policy_started=policy_started)
            summary["case"] = case
            _write_json(destination / "summary.json", summary)
            # A screenshot error must not prevent preservation of confirmed data.
            if observation is not None:
                try:
                    _save_frame(destination, "final", observation, records[-1]["info"] if records else initial_info)
                    if nearest is not None:
                        _save_frame(destination, "nearest_target", nearest_observation, nearest)
                except Exception as error:
                    _write_json(destination / "frame_error.json", dict(type=type(error).__name__, message=str(error)))
    return summary


def execute_plan(plan: dict) -> dict:
    import numpy as np
    import torch
    from so101_capstone.evaluation.stage_policy_metrics import aggregate_results, summarize_episode
    from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
    from so101_capstone.policies.act_policy_adapter import ActPolicyAdapter
    from .act_ppo_multiworld_worker import GazeboWorkerClient, GazeboWorkerConfig

    assert_resources_unused(plan)
    output = Path(plan["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    _write_json(output / "resolved_reward.json", plan["resolved_reward"])
    _write_json(output / "resolved_stages.json", plan["resolved_stages"])
    overrides = dict(SO101_ENVIRONMENT_RANDOMIZATION_ENABLED="false",
                     SO101_IMAGE_AUGMENTATION_ENABLED="false",
                     SO101_ASSISTED_GRASP_ENABLED=str(plan["assisted_grasp_enabled"]).lower())
    original = {key: os.environ.get(key) for key in overrides}
    completed = []
    finished = False

    def persist():
        result = dict(complete=finished, protocol=plan["protocol"], checkpoint=plan["checkpoint"],
                      assisted_grasp_enabled=plan["assisted_grasp_enabled"],
                      physics_mode=plan["physics_mode"], episodes=completed,
                      stages=aggregate_results(completed, plan["stages"]))
        _write_json(output / "summary.json", result)
        return result

    try:
        os.environ.update(overrides)
        torch.set_num_threads(4)
        loaded = load_act_checkpoint(Path(plan["checkpoint"]), device=plan["device"])
        loaded.policy.eval()
        adapter = ActPolicyAdapter(loaded.policy, device=loaded.device, max_joint_delta_rad=None)
        digest = hashlib.sha256()
        with (Path(plan["checkpoint"]) / "model.safetensors").open("rb") as model:
            for chunk in iter(lambda: model.read(1024 * 1024), b""):
                digest.update(chunk)
        _write_json(output / "model_identity.json", dict(sha256=digest.hexdigest(), checkpoint=plan["checkpoint"]))
        for ordinal, case in enumerate(plan["cases"], 1):
            random.seed(case["scene_seed"])
            np.random.seed(case["scene_seed"])
            torch.manual_seed(case["scene_seed"])
            destination = output / case["name"]
            config = GazeboWorkerConfig(
                world_index=0, ros_domain_id=plan["ros_domain_id"], gz_partition=plan["gz_partition"],
                gazebo_gui=plan["gazebo_gui"], project_root=Path(plan["project_root"]),
                output_directory=output / "world_runs" / f"case_{ordinal:03d}",
                startup_feedback_timeout_s=plan["startup_feedback_timeout_s"],
                startup_max_attempts=1, restart_max_attempts=1,
                max_episode_steps=case["maximum_steps"], reward_profile="pick_place_v2",
                reward_config=plan["resolved_reward"], n_action_steps=int(loaded.config.n_action_steps),
                # evaluation_mode=True is the older ZERO/HOME continuous test
                # and forbids stage handoff. This path shares training starts.
                evaluation_mode=False,
            )
            cleanup_confirmed = False
            try:
                with GazeboWorkerClient(config) as client:
                    _require(client.start(), "ready")
                    try:
                        with torch.inference_mode():
                            _run_episode(client, adapter, case, plan, destination)
                    finally:
                        # _run_episode has saved all completed steps and summary.
                        stopped = _require(client.stop_world(), "world_stopped")
                        if stopped.get("world_index") != 0:
                            raise RuntimeError("Evaluation cleanup acknowledged the wrong world.")
                        cleanup_confirmed = True
                        _write_json(destination / "world_closed.json", stopped)
            except Exception as error:
                failure = dict(type=type(error).__name__, message=str(error))
                _write_json(destination / "execution_error.json", failure)
                if not (destination / "summary.json").exists():
                    summary = summarize_episode([], stage=case["stage"], initial_info={},
                                                reward_config=plan["resolved_reward"],
                                                error=failure, policy_started=False)
                    summary["case"] = case
                    _write_json(destination / "summary.json", summary)
                if not cleanup_confirmed:
                    # Never start another world while ownership/cleanup is uncertain.
                    raise
            finally:
                if (destination / "summary.json").exists():
                    summary = _read_json(destination / "summary.json")
                    summary["world_cleanup_confirmed"] = cleanup_confirmed
                    completed.append(summary)
                    _write_json(destination / "summary.json", summary)
                    persist()
            print(json.dumps(dict(stage=case["stage"], episode=case["episode_index"],
                                  result=summary["stage_result"], output=str(destination))), flush=True)
        finished = True
    except BaseException as error:
        _write_json(output / "failure.json", dict(type=type(error).__name__, message=str(error),
                                                completed_cases=len(completed)))
        raise
    finally:
        try:
            persist()
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    result = persist()
    print(json.dumps(result["stages"], indent=2, default=_json_default), flush=True)
    print(f"Evaluation output: {output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="ACT pretrained_model directory")
    parser.add_argument("--stages", nargs="+", type=str.upper, choices=STAGES, default=STAGES)
    parser.add_argument("--episodes-per-stage", "--episodes", type=int, default=1)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--stage-config", type=Path)
    parser.add_argument("--reward-config", type=Path)
    parser.add_argument("--assisted-grasp", choices=("true", "false"), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action="store_true", help="GUI is enabled by default")
    parser.add_argument("--ros-domain-id", type=int, default=125)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--target-position", type=float, nargs=3, default=(.26, 0., .768))
    parser.add_argument("--startup-timeout-s", type=float, default=60.)
    parser.add_argument("--preparation-timeout-s", type=float, default=600.)
    for stage in STAGES[:5]:
        parser.add_argument(f"--{stage.lower()}-budget-s", type=float)
    parser.add_argument("--execute", action="store_true", help="Start bounded Gazebo policy tests")
    args = parser.parse_args()
    budgets = {f"{s.lower()}_budget_s": getattr(args, f"{s.lower()}_budget_s") for s in STAGES[:5]
               if getattr(args, f"{s.lower()}_budget_s") is not None}
    plan = build_plan(
        args.checkpoint, project_root=args.project_root, stages=tuple(args.stages),
        episodes_per_stage=args.episodes_per_stage, output_directory=args.output_directory,
        stage_config=args.stage_config, reward_config=args.reward_config,
        assisted_grasp=None if args.assisted_grasp is None else args.assisted_grasp == "true",
        device=args.device, gui=not args.headless, ros_domain_id=args.ros_domain_id, seed=args.seed,
        target_position=tuple(args.target_position), startup_timeout_s=args.startup_timeout_s,
        preparation_timeout_s=args.preparation_timeout_s, budget_overrides=budgets,
    )
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        result = execute_plan(plan)
        if not result["complete"] or any(not row["evaluated"] for row in result["episodes"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
