"""Offline bounded IK checks only; never sends robot commands or starts Gazebo.

Run from the project root with its existing system scientific Python:
PYTHONPATH=src /usr/bin/python3 -s codex_workspace/checks/act_ppo_curriculum_fk_reachability.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

from so101_capstone.robot.so101_forward_kinematics import (
    ACTIVE_JOINT_NAMES, DEFAULT_WORLD_FROM_BASE_TRANSLATION, GRASP_POINT_OFFSET,
    compute_grasp_point_pose,
)
from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS


def main():
    root = Path(__file__).resolve().parents[2]
    reward = json.loads((root / "config/pick_place_reward.json").read_text())
    home = np.asarray(reward["home_joint_positions_rad"][:5])
    lower = np.asarray([j.gazebo_lower_rad for j in SIM_JOINT_CALIBRATIONS[:5]])
    upper = np.asarray([j.gazebo_upper_rad for j in SIM_JOINT_CALIBRATIONS[:5]])
    downward = np.asarray([0., 0., -1.])
    rng = np.random.default_rng(20260927)
    position_tolerance = .02
    orientation_tolerance_deg = 20.
    sectors = ((.24, .28, 10.), (.22, .31, 25.), (.20, .35, 45.))
    cases = []
    previous = (lower + upper) / 2

    def pose(q):
        return compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q)))

    for level, (inner, outer, angle_max) in enumerate(sectors):
        for radius in (inner, (inner + outer) / 2, outer):
            for angle_deg in (-angle_max, 0., angle_max):
                angle = math.radians(angle_deg)
                target = np.asarray([radius * math.cos(angle), radius * math.sin(angle), .765])

                def residual(q):
                    current = pose(q)
                    return np.concatenate(((current.position - target) / position_tolerance,
                                           (current.transform[:3, 2] - downward) / math.radians(20.)))

                starts = [previous.copy(), home.copy(), (lower + upper) / 2]
                starts += [rng.uniform(lower, upper) for _ in range(8)]
                candidates = []
                for start in starts:
                    result = least_squares(residual, np.clip(start, lower + 1e-8, upper - 1e-8),
                                           bounds=(lower, upper), max_nfev=400,
                                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
                    reached = pose(result.x)
                    distance = float(np.linalg.norm(reached.position - target))
                    angular_error = float(np.degrees(np.arccos(np.clip(reached.transform[:3, 2] @ downward, -1, 1))))
                    feasible = distance <= position_tolerance and angular_error <= orientation_tolerance_deg
                    candidates.append({
                        "joint_positions_rad": result.x.tolist(),
                        "position_world_m": reached.position.tolist(),
                        "gripper_positive_z_world": reached.transform[:3, 2].tolist(),
                        "position_error_m": distance,
                        "orientation_error_deg": angular_error,
                        "within_joint_limits": bool(np.all(result.x >= lower) and np.all(result.x <= upper)),
                        "minimum_joint_limit_margin_rad": float(np.min(np.minimum(result.x - lower, upper - result.x))),
                        "meets_geometric_thresholds": feasible,
                        "solver_success": bool(result.success),
                        "residual_norm": float(np.linalg.norm(result.fun)),
                        "function_evaluations": int(result.nfev),
                    })
                    if distance < 1e-5 and angular_error < .05:
                        break
                best = min(candidates, key=lambda row: (not row["meets_geometric_thresholds"], row["residual_norm"]))
                previous = np.asarray(best["joint_positions_rad"])
                case = {"level": level, "radius_m": radius, "angle_deg": angle_deg,
                        "target_world_m": target.tolist(), "starts_attempted": len(candidates), **best}
                cases.append(case)
                print(json.dumps({"level": level, "radius_m": radius, "angle_deg": angle_deg,
                                  "position_error_mm": best["position_error_m"] * 1000,
                                  "orientation_error_deg": best["orientation_error_deg"],
                                  "feasible": best["meets_geometric_thresholds"]}), flush=True)

    levels = []
    for level in range(3):
        selected = [row for row in cases if row["level"] == level]
        levels.append({"level": level, "sample_count": len(selected),
                       "geometrically_feasible_count": sum(row["meets_geometric_thresholds"] for row in selected),
                       "maximum_position_error_m": max(row["position_error_m"] for row in selected),
                       "maximum_orientation_error_deg": max(row["orientation_error_deg"] for row in selected)})
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "bounded multistart scipy.optimize.least_squares using production FK and calibration limits",
        "scope": "kinematics only: no simulator, physical collision, contact, torque, trajectory or success validation",
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "target_resting_center_z_m": .765,
        "position_tolerance_m": position_tolerance,
        "orientation_tolerance_deg": orientation_tolerance_deg,
        "desired_gripper_positive_z_world": downward.tolist(),
        "world_from_base_translation": DEFAULT_WORLD_FROM_BASE_TRANSLATION.tolist(),
        "grasp_point_offset": GRASP_POINT_OFFSET.tolist(),
        "joint_names": list(ACTIVE_JOINT_NAMES), "joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist(),
        "level_summaries": levels, "cases": cases,
    }
    destination = root / "codex_workspace/reports/2026-09-27_curriculum_fk_reachability.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(destination), "level_summaries": levels}), flush=True)


if __name__ == "__main__":
    main()
