"""Bounded, isolated simulator geometry probe; never ACT or PPO training.

First --make-plan offline (system Python has scipy). Later --execute-plan with
the project's sourced ROS/conda environment. Planning alone never starts ROS.
This is a diagnostic trajectory, not a demonstration injected into training.
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
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from so101_capstone.rewards import PickPlaceRewardConfig
from so101_capstone.robot.so101_forward_kinematics import ACTIVE_JOINT_NAMES, GRASP_POINT_OFFSET, compute_grasp_point_pose
import probe_grasp_geometry_reward as static_probe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REWARD = ROOT / "output_simulation/finetuned/act_ppo_20260927_153051_from_0005000/resolved_reward.json"


def serial(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=serial)+"\n")


def pose(q):
    return compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q)))


def finger_model():
    urdf = ET.parse(ROOT / "ros2_ws/src/so101_description/urdf/so101.urdf.xacro").getroot()
    fixed = urdf.find('./link[@name="gripper_link"]/collision[@name="generated_wrist_roll_follower_so101_v1_part_01"]')
    moving = urdf.find('./link[@name="moving_jaw_so101_v1_link"]/collision[@name="generated_moving_jaw_so101_v1_part_01"]')
    parent_from_reference = static_probe.origin(urdf.find('./joint[@name="gripper_frame_joint"]')) @ static_probe.transform(GRASP_POINT_OFFSET, (0.,)*3)
    return (static_probe.transformed(static_probe.mesh_vertices(fixed), static_probe.origin(fixed)),
            static_probe.mesh_vertices(moving), static_probe.origin(urdf.find('./joint[@name="gripper"]')),
            static_probe.origin(moving), np.linalg.inv(parent_from_reference))


def minimum_finger_z(q, model):
    fixed, moving, joint, collision, reference_from_parent = model
    world_from_parent = pose(q).transform @ reference_from_parent
    moving_parent = static_probe.transformed(moving, joint @ static_probe.transform((0.,)*3, (0., 0., q[-1])) @ collision)
    return float(static_probe.transformed(np.concatenate([fixed, moving_parent]), world_from_parent)[:, 2].min())


def file_hashes():
    paths = ["ros2_ws/src/so101_description/urdf/so101.urdf.xacro",
             "ros2_ws/src/so101_gazebo/models/target_cube/model.sdf",
             "ros2_ws/src/so101_gazebo/models/lab_table/model.sdf",
             "src/so101_capstone/robot/so101_forward_kinematics.py",
             "src/so101_capstone/policies/calibration/so101_simulation.json",
             "ros2_ws/src/so101_description/meshes/collision_generated_ultralow/wrist_roll_follower_so101_v1__functional_finger__part_01.stl",
             "ros2_ws/src/so101_description/meshes/collision_generated_ultralow/moving_jaw_so101_v1__functional_finger__part_01.stl"]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}


def make_plan(reward_path):
    from scipy.optimize import least_squares

    cfg = PickPlaceRewardConfig.from_dict(json.loads(reward_path.read_text()))
    if cfg.reward_schema_version != 3:
        raise ValueError("This diagnostic requires saved schema 3 reward for the historical geometry comparison")
    calibration = json.loads((ROOT / "src/so101_capstone/policies/calibration/so101_simulation.json").read_text())
    limits = [calibration["joints"][name] for name in calibration["joint_order"]]
    lower = np.array([item["range_min_rad"] for item in limits])
    upper = np.array([item["range_max_rad"] for item in limits])
    home = np.array(cfg.home_joint_positions_rad)
    jaw_open = home[-1] + .70
    target = np.array([.26, 0., .765])
    waypoints, commands = [], []
    previous = home.copy()
    hz, maximum_speed = 30., .25

    def append(stage, destination, minimum_s=0.):
        nonlocal previous
        count = max(1, math.ceil(minimum_s*hz), math.ceil(np.max(np.abs(destination-previous))/maximum_speed*hz))
        for index in range(1, count+1):
            commands.append({"stage": stage, "q_rad": (previous+(destination-previous)*index/count).tolist()})
        previous = np.array(destination)

    def ik(point, start):
        def residual(q):
            current = pose(q)
            return np.r_[(current.position-point)/.01, (current.transform[:3, 2]-[0., 0., -1.])/.1]
        starts = [start[:5], np.array([-.01, .11, .397, 1.06, -.116]), (lower[:5]+upper[:5])/2]
        candidates = []
        for seed in starts:
            candidates.append(least_squares(residual, np.clip(seed, lower[:5]+1e-7, upper[:5]-1e-7),
                              bounds=(lower[:5], upper[:5]), max_nfev=300, ftol=1e-9, xtol=1e-9, gtol=1e-9))
            if np.linalg.norm(candidates[-1].fun) < .005:
                break
        result = min(candidates, key=lambda item: np.linalg.norm(item.fun))
        reached = pose(result.x)
        error = float(np.linalg.norm(reached.position-point))
        angle = math.degrees(math.acos(float(np.clip(-reached.transform[2, 2], -1., 1.))))
        if error > .001 or angle > 2.:
            raise RuntimeError(f"IK failed strict diagnostic tolerance at {point}: {error} m, {angle} deg")
        q = np.r_[result.x, jaw_open]
        waypoints.append({"target_reference_m": point.tolist(), "q_rad": q.tolist(), "position_error_m": error,
                          "axis_error_deg": angle, "solver_success": bool(result.success)})
        return q

    append("home_hold", home, 1.)
    append("open_at_home", np.r_[home[:5], jaw_open])
    above = target + [0., 0., .08]
    append("move_above", ik(above, previous))
    append("above_hold", previous, 1.)
    # Successive IK points only 2 mm apart; final replay remains bounded in q.
    for z in np.linspace(above[2], target[2], 41)[1:]:
        append("descend", ik(np.array([target[0], target[1], z]), previous), .2)
    append("preclose_hold", previous, .5)
    preclose_q = previous.copy()
    append("close", np.r_[previous[:5], home[-1]])
    append("contact_hold", previous, .5)
    for z in np.linspace(target[2], target[2]+.03, 16)[1:]:
        next_q = ik(np.array([target[0], target[1], z]), previous)
        next_q[-1] = home[-1]  # Runtime replaces with the measured certified closing command.
        append("lift", next_q, .2)
    append("lift_hold", previous, 1.)
    q_all = np.array([home.tolist()] + [entry["q_rad"] for entry in commands])
    model = finger_model()
    min_z = min(minimum_finger_z(q, model) for q in q_all)
    if not (np.all(q_all >= lower) and np.all(q_all <= upper)):
        raise ValueError("Trajectory exceeds calibrated limits")
    if min_z < cfg.source_support_z_m-.0005:
        raise ValueError(f"Commanded finger mesh crosses tabletop: minimum z={min_z}")
    stage_counts = {stage: sum(entry["stage"] == stage for entry in commands) for stage in dict.fromkeys(row["stage"] for row in commands)}
    return {"schema_version": 1, "purpose": "dynamic geometry probe, no ACT, optimizer, or training integration",
            "resolved_reward": cfg.to_dict(), "reward_hash": cfg.config_hash, "source_reward": str(reward_path.resolve()),
            "project_root": str(ROOT), "source_geometry_hashes": file_hashes(), "scene_seed": 20260927,
            "cube_spawn_position_world_m": [.26, 0., .768], "expected_resting_center_world_m": target.tolist(),
            "cube_quaternion_xyzw": [0., 0., 0., 1.], "initial_opening_rad": .70,
            "max_joint_command_speed_rad_s": maximum_speed, "control_hz": hz,
            "settle_steps": 30, "max_episode_steps": 1500, "max_runtime_wall_s": 1800.,
            "home_q_rad": home.tolist(), "joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist(),
            "preclose_q_rad": preclose_q.tolist(), "waypoints": waypoints, "commands": commands,
            "offline": {"command_steps": len(commands), "total_steps_including_settle": len(commands)+30,
                        "command_duration_s": (len(commands)+30)/hz, "stage_counts": stage_counts,
                        "maximum_joint_command_speed_rad_s": float(np.max(np.abs(np.diff(q_all, axis=0)))*hz),
                        "minimum_commanded_finger_mesh_z_m": min_z,
                        "maximum_ik_position_error_m": max(row["position_error_m"] for row in waypoints),
                        "maximum_ik_axis_error_deg": max(row["axis_error_deg"] for row in waypoints)},
            "abort_thresholds": {"arm_tracking_error_rad": .20, "tracking_bad_steps": 15,
                                 "joint_velocity_rad_s": 4., "target_xy_displacement_m": .025,
                                 "target_below_table_m": .003, "finger_mesh_below_table_m": .002,
                                 "measurement_bad_steps": 4, "preclose_distance_m": .01},
            "limitations": ["IK and finger-table mesh tests do not certify arm/body self-collision or a collision-free path.",
                            "General arm/body contact forces are not exposed by this worker; anomalies are inferred from measured motion, task failure, and Gazebo logs.",
                            "Small lift is attempted only after real grasp_valid with valid contact evidence; failure does not relax reward thresholds.",
                            "No manual dataset grasp was certified from action/state alone; this is an independently specified diagnostic trajectory."]}


def require(response, kind):
    if response.get("type") != kind:
        raise RuntimeError(f"Expected {kind}, received {response}")
    return response


def execute_plan(plan, output, domain):
    from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerClient, GazeboWorkerConfig
    from so101_capstone.training.act_paired_evaluation import assert_resources_unused
    from PIL import Image

    if domain != 121:
        raise ValueError("This prepared diagnostic reserves domain 121 only; never target evaluation domain 120")
    if plan["source_geometry_hashes"] != file_hashes():
        raise ValueError("Geometry changed since planning; regenerate and review the plan")
    cfg = PickPlaceRewardConfig.from_dict(plan["resolved_reward"])
    if cfg.reward_schema_version != 3 or cfg.config_hash != plan["reward_hash"]:
        raise ValueError("Invalid saved reward contract")
    partition = f"so101_grasp_geometry_{uuid.uuid4().hex}"
    assert_resources_unused({"ros_domain_id": domain, "gz_partition": partition})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    write(output / "resolved_reward.json", cfg.to_dict())
    write(output / "resources.json", {"ros_domain_id": domain, "gz_partition": partition})
    os.environ["SO101_ENVIRONMENT_RANDOMIZATION_ENABLED"] = "false"
    config = GazeboWorkerConfig(world_index=0, ros_domain_id=domain, gz_partition=partition, gazebo_gui=False,
            project_root=ROOT, output_directory=output, startup_feedback_timeout_s=45., startup_max_attempts=1,
            restart_max_attempts=1, max_episode_steps=plan["max_episode_steps"], reward_profile="pick_place_v2",
            reward_config=cfg.to_dict(), n_action_steps=1)
    thresholds = plan["abort_thresholds"]
    model = finger_model()
    rows, outcome, reason = [], "not_started", None
    captured_grip = None
    bad_tracking = bad_measurement = 0
    baseline_target = None
    start_wall = time.monotonic()
    previous_stage = None
    try:
        with GazeboWorkerClient(config) as client:
            require(client.start(), "ready")
            client.prepare_episode_async(0, task_mode="pick_place", scene_seed=plan["scene_seed"])
            prepared = require(client.receive(timeout_s=90), "episode_prepared")
            write(output / "prepared.json", prepared)
            client.spawn_target_async(plan["cube_spawn_position_world_m"], plan["cube_quaternion_xyzw"])
            spawn = require(client.receive(timeout_s=30), "target_spawned")
            if not spawn["success"]:
                raise RuntimeError(f"Target spawn failed: {spawn}")
            commands = [{"stage": "target_settle", "q_rad": plan["home_q_rad"]}] * plan["settle_steps"] + plan["commands"]
            with (output / "steps.jsonl").open("x", buffering=1) as stream:
                for index, command in enumerate(commands):
                    stage = command["stage"]
                    if time.monotonic()-start_wall > plan["max_runtime_wall_s"]:
                        reason = "wall_timeout"
                        break
                    if stage == "close" and previous_stage != "close":
                        last = rows[-1]["info"]
                        last_context = last.get("critic_task_context", [])
                        opening = rows[-1]["opening_rad"]
                        if (len(last_context) != 93 or not all(last_context[i] > .5 for i in (34, 35, 36, 37))
                                or last.get("distance_m", math.inf) > thresholds["preclose_distance_m"]
                                or last.get("grasp_close_gate", 0.) <= 0.
                                or not cfg.approach_opening_min_rad <= opening <= cfg.approach_opening_max_rad):
                            reason = "preclose_position_not_reached"
                            break
                    if stage in ("contact_hold", "lift", "lift_hold") and captured_grip is None:
                        reason = "no_certified_grasp_after_bounded_closing"
                        break
                    if stage == "lift" and previous_stage != "lift" and not rows[-1]["info"].get("grasp_valid", False):
                        reason = "grasp_lost_before_lift"
                        break
                    q = np.array(command["q_rad"])
                    if captured_grip is not None and stage in ("close", "contact_hold", "lift", "lift_hold"):
                        q[-1] = captured_grip
                    client.step_async(q)
                    response = require(client.receive(timeout_s=30), "step_result")
                    observation = client.observation_from_metadata(response["observation"])
                    info = response["info"]
                    context = info.get("critic_task_context", [])
                    valid = len(context) == 93 and all(context[i] > .5 for i in (34, 35, 36, 37))
                    privileged = info.get("privileged_state")
                    target = None if privileged is None else np.asarray(privileged.target_position)
                    if stage != "target_settle" and baseline_target is None and target is not None:
                        baseline_target = target.copy()
                    measured_q = observation.joint_positions
                    tip_z = minimum_finger_z(measured_q, model)
                    row = {"step": index+1, "stage": stage, "command_q_rad": q.tolist(),
                           "measured_q_rad": measured_q.tolist(), "measured_velocity_rad_s": observation.joint_velocities.tolist(),
                           "opening_rad": float(measured_q[-1]-cfg.home_joint_positions_rad[-1]),
                           "left_contact": bool(context[71]) if len(context) == 93 else None,
                           "right_contact": bool(context[72]) if len(context) == 93 else None,
                           "simultaneous_contact": bool(context[73]) if len(context) == 93 else None,
                           "minimum_finger_mesh_z_from_measured_q_m": tip_z,
                           "target_position_world_m": target, "target_displacement_from_settled_m": None if target is None or baseline_target is None else target-baseline_target,
                           "wall_elapsed_s": time.monotonic()-start_wall, "reward": response["reward"],
                           "terminated": response["terminated"], "truncated": response["truncated"], "info": info}
                    stream.write(json.dumps(row, allow_nan=False, default=serial)+"\n")
                    rows.append(row)
                    if stage != previous_stage or index % 60 == 0:
                        Image.fromarray(observation.top_image).save(output / f"step_{index+1:04d}_{stage}_top.png")
                        Image.fromarray(observation.gripper_image).save(output / f"step_{index+1:04d}_{stage}_gripper.png")
                        print(json.dumps({"step": index+1, "stage": stage, "distance_m": info.get("distance_m"),
                                          "grasp_valid": info.get("grasp_valid"), "lifted_once": info.get("lifted_once")}), flush=True)
                    previous_stage = stage
                    if (stage == "close" and valid and info.get("grasp_valid", False)
                            and all(context[i] > .5 for i in (71, 72, 73))):
                        captured_grip = float(q[-1])
                    bad_measurement = 0 if valid or stage == "target_settle" else bad_measurement+1
                    bad_tracking = bad_tracking+1 if np.max(np.abs(measured_q[:5]-q[:5])) > thresholds["arm_tracking_error_rad"] else 0
                    if response["terminated"] or response["truncated"]:
                        reason = f"environment_end:{info.get('failure_reason')}"
                    elif bad_measurement >= thresholds["measurement_bad_steps"]:
                        reason = "invalid_measurement_or_contact_evidence"
                    elif bad_tracking >= thresholds["tracking_bad_steps"]:
                        reason = "sustained_arm_tracking_error"
                    elif np.max(np.abs(observation.joint_velocities)) > thresholds["joint_velocity_rad_s"]:
                        reason = "unexpected_joint_velocity"
                    elif tip_z < cfg.source_support_z_m-thresholds["finger_mesh_below_table_m"]:
                        reason = "measured_finger_mesh_table_penetration"
                    elif target is not None and info.get("target_bottom_z_m", target[2]-cfg.target_half_extents_m[2]) < cfg.source_support_z_m-thresholds["target_below_table_m"]:
                        reason = "target_below_table"
                    elif target is not None and baseline_target is not None and np.linalg.norm((target-baseline_target)[:2]) > thresholds["target_xy_displacement_m"]:
                        reason = "unexpected_target_xy_motion"
                    elif stage in ("lift", "lift_hold") and not info.get("grasp_valid", False):
                        reason = "grasp_lost_during_lift"
                    if reason:
                        break
            final_info = {} if not rows else rows[-1]["info"]
            outcome = "lift_evidence_observed" if reason is None and final_info.get("lifted_once", False) else "diagnostic_failed"
            write(output / "gazebo_diagnostics.json", client.gazebo_diagnostics())
            client.stop_world()
    except BaseException as error:
        outcome, reason = "infrastructure_error", f"{type(error).__name__}: {error}"
        write(output / "exception.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write(output / "summary.json", {"outcome": outcome, "reason": reason, "completed_steps": len(rows),
             "captured_closing_command_q_rad": captured_grip, "actual_target_settled_position_m": baseline_target,
             "grasp_observed": any(row["info"].get("grasp_valid", False) for row in rows),
             "lift_observed": any(row["info"].get("lifted_once", False) for row in rows),
             "task_success": False, "scope": "Grasp/lift geometry probe only; no transport/place/HOME task was attempted.",
             "final_info": {} if not rows else rows[-1]["info"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-plan", type=Path)
    mode.add_argument("--execute-plan", type=Path)
    parser.add_argument("--reward-config", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--ros-domain-id", type=int, default=121)
    args = parser.parse_args()
    if args.make_plan:
        plan = make_plan(args.reward_config)
        write(args.make_plan, plan)
        print(json.dumps({"plan": str(args.make_plan), "offline": plan["offline"], "limitations": plan["limitations"]}, indent=2))
    else:
        if args.output_directory is None:
            parser.error("--execute-plan requires a new --output-directory")
        execute_plan(json.loads(args.execute_plan.read_text()), args.output_directory.resolve(), args.ros_domain_id)


if __name__ == "__main__":
    main()
