"""Deterministic target-position curriculum; every episode remains a full task.

These sectors are geometric candidates inside the existing spawn workspace,
not claims of validated robot reachability. A level changes only the initial
target distribution: no grasp, lift, release, success or termination is faked.
Stronger measured downstream evidence also passes an easier frontier: a real
grasp followed by lift need not pause open at pregrasp, and a complete task
with explicit formal contact provenance can pass either training frontier.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import random
from typing import Any, Mapping, Sequence


_SCHEMA_VERSION = 1
# Inner radius, outer radius, symmetric angle limit in radians.
_SECTORS = ((0.24, 0.28, math.radians(10)),
            (0.22, 0.31, math.radians(25)),
            (0.20, 0.35, math.radians(45)))


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return float(value)


def _vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a finite {size}-vector.") from error
    if len(result) != size:
        raise ValueError(f"{name} must be a finite {size}-vector.")
    return tuple(_finite(item, name) for item in result)


def _flag(value: Any, name: str) -> bool:
    # Episode stage summaries currently use numeric 0/1; reject NaN and all
    # other numbers instead of letting bool(NaN) fabricate a passed milestone.
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and math.isfinite(value) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be bool or finite numeric 0/1.")


@dataclass(frozen=True)
class ReachCurriculumConfig:
    window_size: int = 20
    pass_rate: float = 0.7
    rehearsal_probability: float = 0.2
    max_spawn_attempts: int = 128

    def __post_init__(self) -> None:
        window = _integer(self.window_size, "window_size", minimum=20)
        rate = _finite(self.pass_rate, "pass_rate")
        if not 0 < rate <= 1:
            raise ValueError("pass_rate must be in (0, 1].")
        rehearsal = _finite(self.rehearsal_probability, "rehearsal_probability")
        if not 0 <= rehearsal < 1:
            raise ValueError("rehearsal_probability must be in [0, 1).")
        attempts = _integer(self.max_spawn_attempts, "max_spawn_attempts", minimum=1)
        # Persist ordinary JSON scalars even when programmatic callers pass
        # integer/real scalar subclasses.
        object.__setattr__(self, "window_size", window)
        object.__setattr__(self, "pass_rate", rate)
        object.__setattr__(self, "rehearsal_probability", rehearsal)
        object.__setattr__(self, "max_spawn_attempts", attempts)


def target_overlaps_goal(
    target_xy: Sequence[float], target_half_extents_xy_m: Sequence[float],
    goal_xy: Sequence[float], goal_half_extents_xy_m: Sequence[float],
    goal_yaw_rad: float,
) -> bool:
    """Rectangle SAT: even partial overlap or touching is rejected.

    The curriculum spawns upright cubes at yaw zero; the goal may be rotated.
    """
    tx, ty = _vector(target_xy, 2, "target_xy")
    thx, thy = _vector(target_half_extents_xy_m, 2, "target_half_extents_xy_m")
    gx, gy = _vector(goal_xy, 2, "goal_xy")
    ghx, ghy = _vector(goal_half_extents_xy_m, 2, "goal_half_extents_xy_m")
    if min(thx, thy, ghx, ghy) <= 0:
        raise ValueError("Target and goal half extents must be positive.")
    yaw = _finite(goal_yaw_rad, "goal_yaw_rad")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    for ax, ay in ((1., 0.), (0., 1.), (cosine, sine), (-sine, cosine)):
        distance = abs((tx - gx) * ax + (ty - gy) * ay)
        target_radius = thx * abs(ax) + thy * abs(ay)
        goal_radius = ghx * abs(ax * cosine + ay * sine) + ghy * abs(-ax * sine + ay * cosine)
        if distance > target_radius + goal_radius + 1e-12:
            return False
    return True


class ReachCurriculum:
    def __init__(self, config: ReachCurriculumConfig | None = None) -> None:
        self.config = ReachCurriculumConfig() if config is None else config
        if not isinstance(self.config, ReachCurriculumConfig):
            raise ValueError("config must be ReachCurriculumConfig.")
        self.level = 0
        self.attempt_count = 0
        self._window: deque[bool] = deque(maxlen=self.config.window_size)
        self._seen_attempts: set[int] = set()
        self._level_episode_counts = [0, 0, 0]
        self._rehearsal_episode_count = 0

    def allocate_attempt(self) -> int:
        self.attempt_count += 1
        return self.attempt_count

    def sampling_contract(self) -> dict[str, Any]:
        """Version the geometry and sampling semantics independently of progress."""
        return {
            "schema_version": 1,
            "sectors": [
                {"level": level, "inner_radius_m": inner, "outer_radius_m": outer,
                 "angle_min_rad": -angle, "angle_max_rad": angle}
                for level, (inner, outer, angle) in enumerate(_SECTORS)
            ],
            "workspace_center_xy_m": [0., 0.],
            "position_sampling": "uniform_radius_squared_and_uniform_angle_v1",
            "random_generator": "python_random_local_integer_seed",
            "target_quaternion_xyzw": [0., 0., 0., 1.],
            "goal_exclusion": "rotated_rectangle_sat_reject_partial_overlap_and_touch_v1",
            "goal_exclusion_tolerance_m": 1e-12,
            "spawn_height": "source_support_z_plus_target_half_height_plus_clearance",
            "default_spawn_clearance_m": .003,
            "rehearsal_sampling": "bernoulli_then_uniform_among_earlier_levels_v1",
        }

    def sample_target(
        self, seed: int, *, goal_center_world_m: Sequence[float],
        goal_half_extents_xy_m: Sequence[float] = (.05, .05), goal_yaw_rad: float = 0.,
        target_half_extents_m: Sequence[float] = (.015, .015, .015),
        source_support_z_m: float = .75, spawn_clearance_m: float = .003,
    ) -> dict[str, Any]:
        seed = _integer(seed, "seed")
        goal = _vector(goal_center_world_m, 3, "goal_center_world_m")
        goal_half = _vector(goal_half_extents_xy_m, 2, "goal_half_extents_xy_m")
        target_half = _vector(target_half_extents_m, 3, "target_half_extents_m")
        if min(*goal_half, *target_half) <= 0:
            raise ValueError("Target and goal half extents must be positive.")
        yaw = _finite(goal_yaw_rad, "goal_yaw_rad")
        support = _finite(source_support_z_m, "source_support_z_m")
        clearance = _finite(spawn_clearance_m, "spawn_clearance_m")
        if clearance < 0:
            raise ValueError("spawn_clearance_m must be nonnegative.")
        rng = random.Random(seed)
        rehearsal = self.level > 0 and rng.random() < self.config.rehearsal_probability
        sampled_level = rng.randrange(self.level) if rehearsal else self.level
        inner, outer, angle_limit = _SECTORS[sampled_level]
        for _ in range(self.config.max_spawn_attempts):
            radius = math.sqrt(rng.uniform(inner * inner, outer * outer))
            angle = rng.uniform(-angle_limit, angle_limit)
            xy = (radius * math.cos(angle), radius * math.sin(angle))
            if not target_overlaps_goal(xy, target_half[:2], goal[:2], goal_half, yaw):
                return {
                    "curriculum_level": self.level,
                    "sampled_level": sampled_level,
                    "curriculum_rehearsal": rehearsal,
                    "scene_seed": seed,
                    "target_position_world_m": [*xy, support + target_half[2] + clearance],
                    "target_quaternion_xyzw": [0., 0., 0., 1.],
                    "workspace_validation": "geometric_only_not_physical_reachability",
                }
        raise ValueError("Curriculum target sector has no sampled non-overlapping goal position; check geometry.")

    def observe(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise ValueError("Episode record must be a mapping.")
        previous = self.level
        valid = _flag(record.get("valid_episode", record.get("valid", True)), "valid_episode")
        if not valid or record.get("type") == "episode_failure" or record.get("task_mode") != "pick_place":
            return {**self.diagnostics(), "promoted": False, "ignored_reason": "not_valid_pick_place"}
        final_info = record.get("final_info", {})
        if not isinstance(final_info, Mapping):
            raise ValueError("final_info must be a mapping.")
        # Collectors retain environment evidence in final_info. Never let a
        # top-level default (or conflicting contact label) hide an explicit
        # proxy/unknown evidence mode in either location.
        if (record.get("grasp_evidence_mode", "contact") != "contact"
                or final_info.get("grasp_evidence_mode", "contact") != "contact"):
            return {**self.diagnostics(), "promoted": False, "ignored_reason": "non_contact_evidence"}
        if _flag(record.get("curriculum_probe", False), "curriculum_probe"):
            return {**self.diagnostics(), "promoted": False, "ignored_reason": "evaluation_probe"}
        attempt = _integer(record.get("episode_attempt_index"), "episode_attempt_index", minimum=1)
        if attempt in self._seen_attempts:
            return {**self.diagnostics(), "promoted": False, "ignored_reason": "duplicate_attempt"}
        sampled_level = _integer(record.get("sampled_level"), "sampled_level")
        if sampled_level > 2:
            raise ValueError("sampled_level must be 0, 1 or 2.")
        rehearsal = _flag(record.get("curriculum_rehearsal", False), "curriculum_rehearsal")
        stages = record.get("stage_reached", {})
        if not isinstance(stages, Mapping):
            raise ValueError("stage_reached must be a mapping.")
        precise = _flag(stages.get("precise_reach", record.get("precise_reach", False)), "precise_reach")
        grasp = _flag(stages.get("grasp_valid", record.get("grasp_valid", False)), "grasp_valid")
        lift = _flag(stages.get("lifted_once", record.get("lifted_once", False)), "lifted_once")
        success = _flag(record.get("task_success", False), "task_success")
        # A smooth successful grasp may never dwell open at pregrasp. Stronger
        # measured downstream evidence can satisfy the easier frontier. A bare
        # task-success label needs explicit formal contact provenance.
        explicit_contact = (record.get("grasp_evidence_mode") == "contact"
                            or final_info.get("grasp_evidence_mode") == "contact")
        formal_success = _flag(final_info.get("formal_success", record.get("formal_success", True)),
                               "formal_success")
        complete_success = success and explicit_contact and formal_success
        grasp_and_lift = grasp and lift
        self._seen_attempts.add(attempt)
        self.attempt_count = max(self.attempt_count, attempt)
        if rehearsal:
            self._rehearsal_episode_count += 1
        if rehearsal or sampled_level != self.level:
            return {**self.diagnostics(), "promoted": False, "ignored_reason": "not_current_frontier"}
        self._level_episode_counts[self.level] += 1
        passed = ((precise or grasp_and_lift or complete_success) if self.level == 0
                  else (grasp_and_lift or complete_success) if self.level == 1
                  else complete_success)
        self._window.append(passed)
        if (self.level < 2 and len(self._window) == self.config.window_size
                and sum(self._window) >= math.ceil(self.config.pass_rate * self.config.window_size)):
            self.level += 1
            self._window.clear()
        return {**self.diagnostics(), "promoted": self.level != previous, "previous_level": previous}

    def diagnostics(self) -> dict[str, Any]:
        count = len(self._window)
        return {
            "curriculum_level": self.level,
            "curriculum_attempt_count": self.attempt_count,
            "curriculum_window_count": count,
            "curriculum_window_pass_count": sum(self._window),
            "curriculum_window_pass_rate": sum(self._window) / count if count else 0.,
            "curriculum_level_episode_counts": self._level_episode_counts.copy(),
            "curriculum_rehearsal_episode_count": self._rehearsal_episode_count,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "config": asdict(self.config),
            "sampling_contract": self.sampling_contract(),
            "level": self.level,
            "attempt_count": self.attempt_count,
            "window": list(self._window),
            "seen_attempts": sorted(self._seen_attempts),
            "level_episode_counts": self._level_episode_counts.copy(),
            "rehearsal_episode_count": self._rehearsal_episode_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("Curriculum state fields do not match this schema.")
        if _integer(state["schema_version"], "schema_version") != _SCHEMA_VERSION or state["config"] != asdict(self.config):
            raise ValueError("Curriculum state schema/config does not match.")
        if state["sampling_contract"] != self.sampling_contract():
            raise ValueError("Curriculum sampling contract does not match; target geometry or sampling changed.")
        level = _integer(state["level"], "level")
        attempts = _integer(state["attempt_count"], "attempt_count")
        if level > 2:
            raise ValueError("level must be 0, 1 or 2.")
        if not isinstance(state["window"], list) or len(state["window"]) > self.config.window_size:
            raise ValueError("Curriculum window is invalid.")
        window = [_flag(value, "window result") for value in state["window"]]
        if not isinstance(state["seen_attempts"], list):
            raise ValueError("seen_attempts must be a list.")
        seen = [_integer(value, "seen attempt", minimum=1) for value in state["seen_attempts"]]
        if len(set(seen)) != len(seen) or any(value > attempts for value in seen):
            raise ValueError("Curriculum attempt history is inconsistent.")
        if not isinstance(state["level_episode_counts"], list) or len(state["level_episode_counts"]) != 3:
            raise ValueError("level_episode_counts must contain three integers.")
        counts = [_integer(value, "level episode count") for value in state["level_episode_counts"]]
        rehearsal = _integer(state["rehearsal_episode_count"], "rehearsal_episode_count")
        if (sum(counts) + rehearsal > len(seen) or counts[level] < len(window)
                or any(counts[index] for index in range(level + 1, 3))
                or any(counts[index] < self.config.window_size for index in range(level))):
            raise ValueError("Curriculum counters are inconsistent.")
        self.level, self.attempt_count = level, attempts
        self._window = deque(window, maxlen=self.config.window_size)
        self._seen_attempts = set(seen)
        self._level_episode_counts = counts
        self._rehearsal_episode_count = rehearsal
