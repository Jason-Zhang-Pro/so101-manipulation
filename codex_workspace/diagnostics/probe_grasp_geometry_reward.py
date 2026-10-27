"""Offline geometry/shaping audit. No policy load, ROS, physics, or run writes.

From the repository root:
  PYTHONPATH=src python codex_workspace/diagnostics/probe_grasp_geometry_reward.py

The default run is historical training data, not checkpoint-21 evaluation.
Mesh axis-aligned bounds are diagnostic bounds, never contact certification.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

from so101_capstone.rewards import PickPlaceReward, PickPlaceRewardConfig, TaskMeasurements, TaskStateTracker
from so101_capstone.robot.so101_forward_kinematics import (
    ACTIVE_JOINT_NAMES, GRASP_POINT_OFFSET, compute_grasp_point_pose,
)

ROOT = Path(__file__).resolve().parents[2]


def transform(xyz, rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rz @ ry @ rx, xyz
    return result


def origin(element):
    node = element.find("origin")
    return transform([float(v) for v in node.get("xyz", "0 0 0").split()],
                     [float(v) for v in node.get("rpy", "0 0 0").split()])


def mesh_vertices(collision):
    mesh = collision.find("geometry/mesh")
    path = ROOT / mesh.get("filename").replace("package://so101_description/", "ros2_ws/src/so101_description/")
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + count * 50, "Expected binary STL"
    vertices = np.array([struct.unpack_from("<12fH", data, 84 + index*50)[3:12] for index in range(count)]).reshape(-1, 3)
    scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
    return vertices * scale


def transformed(vertices, matrix):
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]


def bounds(vertices):
    return {"min_m": vertices.min(axis=0).tolist(), "max_m": vertices.max(axis=0).tolist()}


def geometry(config):
    urdf = ET.parse(ROOT / "ros2_ws/src/so101_description/urdf/so101.urdf.xacro").getroot()
    frame = origin(urdf.find('./joint[@name="gripper_frame_joint"]'))
    reference = (frame @ np.r_[GRASP_POINT_OFFSET, 1.])[:3]
    fixed = urdf.find('./link[@name="gripper_link"]/collision[@name="generated_wrist_roll_follower_so101_v1_part_01"]')
    moving = urdf.find('./link[@name="moving_jaw_so101_v1_link"]/collision[@name="generated_moving_jaw_so101_v1_part_01"]')
    jaw_joint = urdf.find('./joint[@name="gripper"]')
    fixed_bounds = bounds(transformed(mesh_vertices(fixed), origin(fixed)))
    jaw_bounds = []
    for opening in (0., .45, .70, .90, 1.6525244887227628):
        angle = config.home_joint_positions_rad[-1] + opening
        matrix = origin(jaw_joint) @ transform((0.,)*3, (0., 0., angle)) @ origin(moving)
        jaw_bounds.append({"opening_rad": opening, "joint_q_rad": angle, **bounds(transformed(mesh_vertices(moving), matrix))})
    table = ET.parse(ROOT / "ros2_ws/src/so101_gazebo/models/lab_table/model.sdf").getroot()
    top = table.find('.//collision[@name="tabletop_collision"]')
    table_z = float(top.findtext("pose").split()[2]) + float(top.findtext("geometry/box/size").split()[2])/2
    cube = ET.parse(ROOT / "ros2_ws/src/so101_gazebo/models/target_cube/model.sdf").getroot()
    cube_size = [float(v) for v in cube.findtext('.//collision/geometry/box/size').split()]
    calibration = json.loads((ROOT / "src/so101_capstone/policies/calibration/so101_simulation.json").read_text())
    home = []
    for name, q in zip(calibration["joint_order"], config.home_joint_positions_rad):
        cal = calibration["joints"][name]
        u = cal["policy_min"] + (q-cal["range_min_rad"])/(cal["range_max_rad"]-cal["range_min_rad"])*(cal["policy_max"]-cal["policy_min"])
        home.append({"joint": name, "q_rad": q, "q_deg": math.degrees(q), "policy_units": u, "inside_calibrated_range": cal["range_min_rad"] <= q <= cal["range_max_rad"]})
    comparisons = []
    for q in (config.home_joint_positions_rad, (.1613821089, .0190884639, .3369787931, 1.0501406193, .0211986024, 1.5652580261)):
        manual = transform((0., 0., .7524), (0.,)*3)
        for name, angle in zip(ACTIVE_JOINT_NAMES, q):
            joint = urdf.find(f'./joint[@name="{name}"]')
            assert joint.find("axis").get("xyz") == "0 0 1"
            manual = manual @ origin(joint) @ transform((0.,)*3, (0., 0., angle))
        manual = manual @ frame @ transform(GRASP_POINT_OFFSET, (0.,)*3)
        production = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q))).transform
        comparisons.append(float(np.max(np.abs(manual-production))))
    return {"reference_in_gripper_link_m": reference.tolist(), "fixed_finger_bounds_in_gripper_link": fixed_bounds,
            "moving_finger_bounds_in_gripper_link": jaw_bounds, "cube_size_m": cube_size,
            "table_surface_z_m": table_z, "cube_rest_center_z_m": table_z+cube_size[2]/2,
            "home": home, "urdf_chain_vs_production_fk_max_absolute_errors": comparisons,
            "warning": "Bounds are not fingertip aperture or a contact simulation. Reference pose is independent of gripper q; moving finger is not."}


def inspect_run(run, config):
    counts = dict(steps=0, valid_active=0, opening_ge_1_7=0, old_opening_score_zero=0,
                  near_4_5_cm=0, near_4_5_cm_opening_ge_1_7=0, near_2_cm=0, positive_close_gate=0,
                  any_finger_contact=0, pregrasp_ready=0)
    best = None
    files = sorted((run / "episode_steps").glob("*.jsonl"))
    for file in files:
        for line in file.open():
            row = json.loads(line)
            counts["steps"] += 1
            context = row["critic_task_context"]
            assert len(context) == 93, "Only historical task context schema 2 is decoded"
            if not (context[7] > .5 and context[34] > .5 and context[35] > .5 and context[36] > .5 and context[37] > .5):
                continue
            counts["valid_active"] += 1
            distance = float(np.linalg.norm(context[38:41]))
            opening = float(row["positions"][-1] - config.home_joint_positions_rad[-1])
            counts["opening_ge_1_7"] += opening >= 1.7
            counts["old_opening_score_zero"] += context[83] == 0.
            counts["near_4_5_cm"] += distance <= .045
            counts["near_4_5_cm_opening_ge_1_7"] += distance <= .045 and opening >= 1.7
            counts["near_2_cm"] += distance <= .02
            counts["positive_close_gate"] += context[84] > 0.
            counts["any_finger_contact"] += max(context[71:73]) > 0.
            counts["pregrasp_ready"] += context[26] > .5
            if best is None or distance < best["distance_m"]:
                best = {"episode": file.stem, "step": row["step_index"], "distance_m": distance,
                        "target_minus_reference_m": context[38:41], "positions_rad": row["positions"],
                        "opening_rad_from_recorded_q": opening, "opening_score_recorded": context[83],
                        "orientation_error_deg_from_recorded_score": -math.log(context[82])*90,
                        "close_gate_recorded": context[84]}
    return {"files": len(files), **counts, "best": best,
            "note": "Training episodes 0..62 precede updates 1..21 (policy updates 0..20). q may be timestamp-offset from task evidence; recorded opening score is authoritative for that context."}


def perturbation_probe(run_stats):
    best = run_stats["best"]
    q = np.array(best["positions_rad"])
    pose = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q)))
    target = pose.position + np.array(best["target_minus_reference_m"])
    base = TaskMeasurements(0., q, (0.,)*6, pose.position, pose.transform[:3, :3], target_name="cube",
                            target_position_world_m=target, target_motion_valid=True, contact_valid=True,
                            goal_center_world_m=(.078, -.300, .753))
    perturbations = {"unchanged": (np.zeros(3), 0.), "descend_1_mm": (np.array([0., 0., -.001]), 0.),
                     "align_x_1_mm": (np.array([.001, 0., 0.]), 0.), "align_y_1_mm": (np.array([0., .001, 0.]), 0.),
                     "narrow_gripper_0_01_rad": (np.zeros(3), -.01)}
    probes = {}
    for schema in (3, 4):
        cfg = PickPlaceRewardConfig(reward_schema_version=schema)
        reward = PickPlaceReward(cfg)
        before = reward.potential(TaskStateTracker(cfg).reset(base))
        results = {}
        for name, (shift, grip) in perturbations.items():
            next_q = q.copy()
            next_q[-1] += grip
            snapshot = TaskStateTracker(cfg).reset(replace(base, joint_positions_rad=next_q, grasp_position_world_m=pose.position+shift))
            after = reward.potential(snapshot)
            results[name] = {"potential": after, "delta_potential": after-before,
                             "one_step_shaping": cfg.gamma*after-before,
                             "reward_advantage_vs_stationary": cfg.gamma*(after-before),
                             "distance_m": snapshot.target_distance_m, "orientation_score": snapshot.approach_orientation_score,
                             "opening_score": snapshot.approach_opening_score, "close_gate": snapshot.grasp_close_gate}
        probes[str(schema)] = results
    return {"values": probes, "note": "Controlled local reward differences, using measured task displacement and q-derived rotation. Cartesian shifts are hypothetical, not a feasible commanded action or PPO backpropagation through the environment."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "output_simulation/finetuned/act_ppo_20260927_153051_from_0005000")
    parser.add_argument("--output", type=Path, default=ROOT / "codex_workspace/reports/2026-09-27_grasp_geometry_reward_probe.json")
    args = parser.parse_args()
    legacy = PickPlaceRewardConfig.from_dict(json.loads((args.run / "resolved_reward.json").read_text()))
    current = PickPlaceRewardConfig()
    run_stats = inspect_run(args.run, legacy)
    result = {"historical_reward_hash": legacy.config_hash, "current_reward_hash": current.config_hash,
              "geometry": geometry(current), "run": run_stats, "local_reward_probe": perturbation_probe(run_stats),
              "source_sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
                  "ros2_ws/src/so101_description/urdf/so101.urdf.xacro",
                  "src/so101_capstone/robot/so101_forward_kinematics.py",
                  "src/so101_capstone/rewards/task_state.py",
                  "src/so101_capstone/rewards/pick_place_reward.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "run": run_stats, "local_reward_probe": result["local_reward_probe"]}, indent=2))


if __name__ == "__main__":
    main()
