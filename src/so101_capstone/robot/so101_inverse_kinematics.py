"""Deterministic bounded IK for simulation preparation, using the existing FK.

Only grasp-reference position and its +Z direction are constrained. This is
not a six-DOF pose solver or a collision planner; callers must check the path
and actual feedback. An unreachable request raises instead of returning a
clipped approximation as if it had succeeded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np

from so101_capstone.policies.so101_sim_calibration import SIM_JOINT_CALIBRATIONS
from .so101_forward_kinematics import ACTIVE_JOINT_NAMES, compute_grasp_point_pose


@dataclass(frozen=True)
class IKConfig:
    position_tolerance_m: float = 0.002
    axis_tolerance_rad: float = math.radians(20)
    axis_weight_m: float = 0.08
    finite_difference_rad: float = 1e-5
    max_iteration_delta_rad: float = 0.15
    max_iterations: int = 120
    multistart: bool = True
    # Keep small Cartesian commands moving when acceptance is wider than a
    # waypoint. Search this precision first, then consider an acceptable fit.
    preferred_position_tolerance_m: float | None = None

    def __post_init__(self):
        for name in ("position_tolerance_m", "axis_tolerance_rad", "axis_weight_m",
                     "finite_difference_rad", "max_iteration_delta_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.axis_tolerance_rad >= math.pi:
            raise ValueError("axis_tolerance_rad must be below pi")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.multistart, bool):
            raise ValueError("multistart must be boolean")
        preferred = self.preferred_position_tolerance_m
        if preferred is not None and (
                isinstance(preferred, bool) or not isinstance(preferred, Real)
                or not math.isfinite(preferred) or not 0 < preferred <= self.position_tolerance_m):
            raise ValueError("preferred_position_tolerance_m must be positive and no larger than position_tolerance_m")


@dataclass(frozen=True)
class IKResult:
    joint_positions_rad: tuple[float, ...]
    position_error_m: float
    axis_error_rad: float
    iterations: int


class InverseKinematicsError(ValueError):
    pass


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have {size} finite elements")
    return result


def solve_grasp_point_ik(position_world_m: Sequence[float],
                         seed_joint_positions_rad: Sequence[float], *,
                         desired_axis_world: Sequence[float] = (0, 0, -1),
                         config: IKConfig | None = None) -> IKResult:
    """Solve within calibrated limits. Seed accepts five arm or six joints."""
    c = config or IKConfig()
    target = _vector(position_world_m, 3, "position_world_m")
    seed = np.asarray(seed_joint_positions_rad, dtype=float)
    if seed.shape not in ((5,), (6,)) or not np.isfinite(seed).all():
        raise ValueError("seed_joint_positions_rad must have five or six finite elements")
    seed = seed[:5].copy()
    lower = np.array([j.gazebo_lower_rad for j in SIM_JOINT_CALIBRATIONS[:5]])
    upper = np.array([j.gazebo_upper_rad for j in SIM_JOINT_CALIBRATIONS[:5]])
    if np.any(seed < lower) or np.any(seed > upper):
        raise ValueError("IK seed exceeds calibrated joint limits")
    axis = _vector(desired_axis_world, 3, "desired_axis_world")
    if np.linalg.norm(axis) < 1e-9:
        raise ValueError("desired_axis_world must be nonzero")
    axis = axis / np.linalg.norm(axis)

    def evaluate(q):
        pose = compute_grasp_point_pose(dict(zip(ACTIVE_JOINT_NAMES, q)))
        current_axis = pose.transform[:3, 2]
        angle = float(math.acos(np.clip(current_axis @ axis, -1, 1)))
        # This arm has only five joints. Once direction is inside the allowed
        # cone, do not sacrifice position accuracy to chase an unnecessarily
        # exact axis at a joint limit. Leave a margin inside the acceptance cone.
        direction_scale = max(0.0, angle - .8 * c.axis_tolerance_rad) / max(angle, 1e-12)
        residual = np.concatenate((pose.position - target,
                                   c.axis_weight_m * direction_scale * (current_axis - axis)))
        return residual, float(np.linalg.norm(pose.position - target)), angle

    starts = [seed]
    if c.multistart:
        # Fixed seeds keep the result reproducible and avoid introducing an RNG
        # dependency into episode preparation. Never exceed calibrated limits.
        starts += [np.clip(np.array(v), lower, upper) for v in
                   ((0, -1.4, 1.4, 0.8, -1.57), (0, -0.5, 1.0, 1.0, -1.57),
                    (0, 0.5, -1.0, -1.0, 1.57), (0, 0, 0, 0, 0))]
    best = (math.inf, math.inf)
    best_acceptable = None
    preferred_tolerance = (c.position_tolerance_m if c.preferred_position_tolerance_m is None
                           else c.preferred_position_tolerance_m)
    iterations = 0
    for start in starts:
        q, damping = start.copy(), 1e-4
        for _ in range(c.max_iterations):
            iterations += 1
            residual, position_error, axis_error = evaluate(q)
            if position_error <= preferred_tolerance and axis_error <= c.axis_tolerance_rad:
                return IKResult(tuple(float(v) for v in q), position_error, axis_error, iterations)
            if (position_error <= c.position_tolerance_m and axis_error <= c.axis_tolerance_rad
                    and (best_acceptable is None or (position_error, axis_error) < best_acceptable[:2])):
                best_acceptable = (position_error, axis_error, tuple(float(v) for v in q))
            if position_error + c.axis_weight_m * axis_error < best[0] + c.axis_weight_m * best[1]:
                best = (position_error, axis_error)
            jacobian = np.empty((6, 5))
            for j in range(5):
                step = c.finite_difference_rad if q[j] + c.finite_difference_rad <= upper[j] else -c.finite_difference_rad
                shifted = q.copy()
                shifted[j] += step
                jacobian[:, j] = (evaluate(shifted)[0] - residual) / step
            delta = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(5), -jacobian.T @ residual)
            delta *= min(1.0, c.max_iteration_delta_rad / max(np.max(np.abs(delta)), 1e-12))
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                candidate = np.clip(q + scale * delta, lower, upper)
                if np.linalg.norm(evaluate(candidate)[0]) < np.linalg.norm(residual) - 1e-12:
                    q, accepted = candidate, True
                    damping = max(1e-8, damping * .5)
                    break
            if not accepted:
                damping = min(1.0, damping * 10)
                if damping >= 1.0:
                    break
    if best_acceptable is not None:
        position_error, axis_error, joints = best_acceptable
        return IKResult(joints, position_error, axis_error, iterations)
    raise InverseKinematicsError(
        f"No verified bounded IK solution: best position error={best[0]:.6f} m, "
        f"axis error={math.degrees(best[1]):.3f} deg")
