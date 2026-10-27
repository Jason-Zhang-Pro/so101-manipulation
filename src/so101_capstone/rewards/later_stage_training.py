"""Training adapter for measured S2--S5 rewards and assisted start boundaries.

Preparation never earns reward. Its physical grasp/carry/release history must
survive handoff; only reward records and the stage budget start at that point.
Stage success/timeout are sampling cuts with bootstrap, physical failure is
terminal. Actor observations remain images/joints; stage history is critic-only.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np

from .later_stage_reward import EpisodeRewardLedger, live_carry, placement_valid
from .pick_place_reward import PickPlaceRewardConfig, PickPlaceRewardInput
from .s2_grasp_lift_reward import S2GraspLiftReward
from .s3_transport_reward import S3TransportReward
from .s4_place_release_reward import S4PlaceReleaseReward
from .s5_return_home_reward import S5ReturnHomeReward
from .task_state import TaskSnapshot, object_geometry
from .training_stages import StageTrainingConfig


LATER_STAGE_CONTEXT_SCHEMA_VERSION = 1
PHASES = ("CLOSE", "LIFT", "RECOVER_GRASP", "HOLD_INSIDE_GOAL", "ESTABLISH_CLEARANCE",
          "CARRY_TO_GOAL", "LOWER", "RELEASE", "RAISE", "HOME_HOLD", "SETTLE_HOME",
          "RETURN_HOME", "WITHDRAW", "PLACEMENT_RECOVERY_REQUIRED")


def make_later_reward(stage: str, config: StageTrainingConfig,
                      reward_config: PickPlaceRewardConfig):
    factories = {"S2": S2GraspLiftReward, "S3": S3TransportReward,
                 "S4": S4PlaceReleaseReward, "S5": S5ReturnHomeReward}
    if stage not in factories:
        raise ValueError("later reward stage must be S2 through S5")
    kwargs = {"stage_config": config} if stage == "S5" else {}
    return factories[stage](reward_config, **kwargs)


@lru_cache(maxsize=1)
def later_stage_context_fields() -> tuple[str, ...]:
    """Stable schema: stage index first, then every mutable reward quantity.

    Other-stage slots are zero. Identity strings are validated at boundaries;
    the current-attempt loss-paid flag is sufficient for the only dynamic
    ledger key, because grasp attempt IDs increase and cannot recur.
    """
    fields = ["stage_index", "elapsed_s", "budget_s", "evidence_age_s", "potential",
              "current_attempt_loss_paid", "stage_complete_paid"]
    fields += [f"phase/{name}" for name in PHASES]
    defaults, reward = StageTrainingConfig(), PickPlaceRewardConfig()
    for stage in ("S2", "S3", "S4", "S5"):
        instance = make_later_reward(stage, defaults, reward)
        for key in sorted(instance.weights):
            fields += [f"{stage}/quality/{key}", f"{stage}/best/{key}"]
        for key in sorted(instance.milestones):
            fields += [f"{stage}/goal/{key}", f"{stage}/dwell/{key}",
                       f"{stage}/dwell_best/{key}", f"{stage}/paid/{key}"]
    fields += [f"home/{name}" for name in (
        "initialized", "last_candidate", "eligible", "hold_s", "fresh", "ready", "elapsed_s",
        "start_age_s", "last_sim_age_s", "last_evidence_age_s", "decision_success", "decision_timeout",
        "home_error_rad", "max_joint_velocity_rad_s", "at_home", "robot_static", "candidate")]
    return tuple(fields)


def later_stage_context_dim() -> int:
    return len(later_stage_context_fields())


def later_stage_training_contract(config: StageTrainingConfig,
                                  reward_config: PickPlaceRewardConfig) -> dict[str, object]:
    """Persist reward defaults beside stage overrides for safe resume checks."""
    stages = {}
    for stage in ("S2", "S3", "S4", "S5"):
        reward = make_later_reward(stage, config, reward_config)
        stages[stage] = dict(config=asdict(reward.config), tuning=asdict(reward.tuning),
                             budget_s=config.stage_budget_s(stage), completion_goal=reward.completion_goal,
                             milestones=[asdict(m) for m in reward.milestones.values()])
        if stage in {"S2", "S3"}:
            stages[stage]["attachment_command_reward"] = {
                "schema": 1,
                "reference": "frozen_measured_angle_at_attachment",
                "action": "executed_gripper_command",
                "positive_condition": "command_strictly_less_than_reference",
                "penalty_condition": "command_greater_than_or_equal_to_reference",
                "requires": "current_valid_matching_rigid_attachment",
                "timing": "fresh_covered_simulation_time",
            }
    return dict(schema_version=1, context_schema=LATER_STAGE_CONTEXT_SCHEMA_VERSION,
                context_fields=list(later_stage_context_fields()), stages=stages,
                s5_home_contract=config.to_dict())


def later_stage_critic_context(stage: str, *, elapsed_s: float, budget_s: float,
                              snapshot: TaskSnapshot, adapter=None) -> np.ndarray:
    values = dict(stage_index=float(stage[1]), elapsed_s=elapsed_s, budget_s=budget_s,
                  evidence_age_s=snapshot.sim_time_s - snapshot.evidence_sim_time_s)
    if adapter is not None:
        state = adapter.reward.reward_state()
        prefix = stage.lower()
        values.update(potential=state["potential"],
                      current_attempt_loss_paid=float(f"grasp_lost_attempt_{snapshot.grasp_attempt_id}" in state["paid"]),
                      stage_complete_paid=float(f"{prefix}_stage_complete" in state["paid"]))
        values[f"phase/{state['phase']}"] = 1.0
        for key, value in state["qualities"].items():
            values[f"{stage}/quality/{key}"] = value
            values[f"{stage}/best/{key}"] = state["best"][f"{prefix}_{key}"]
        for key, value in state["previous_goals"].items():
            values[f"{stage}/goal/{key}"] = float(value)
            values[f"{stage}/dwell/{key}"] = state["dwell_s"][key]
            values[f"{stage}/dwell_best/{key}"] = state["best"].get(f"{prefix}_{key}_dwell", 0.0)
            values[f"{stage}/paid/{key}"] = float(f"{prefix}_{key}_complete" in state["paid"])
        home = state.get("home_success", {})
        for key, value in home.items():
            if isinstance(value, (bool, int, float)):
                values[f"home/{key}"] = float(value)
        for key in ("start_time", "last_sim_time", "last_evidence_time"):
            if home.get(key) is not None:
                label = {"start_time": "start_age_s", "last_sim_time": "last_sim_age_s",
                         "last_evidence_time": "last_evidence_age_s"}[key]
                values[f"home/{label}"] = snapshot.sim_time_s - home[key]
        for key, value in (home.get("measurement") or {}).items():
            values[f"home/{key}"] = float(value)
        values["home/decision_success"] = float(home.get("decision", (None,))[0] == "SUCCESS")
        values["home/decision_timeout"] = float(home.get("decision", (None,))[0] == "TIMEOUT")
    result = np.asarray([values.get(key, 0.0) for key in later_stage_context_fields()], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite later-stage critic context")
    return result


class LaterStageTraining:
    def __init__(self, stage: str, config: StageTrainingConfig,
                 reward_config: PickPlaceRewardConfig):
        self.stage, self.config, self.reward_config = stage, config, reward_config
        self.reward = make_later_reward(stage, config, reward_config)
        self.budget_s = config.stage_budget_s(stage)
        self._initialized = False

    def reset(self, snapshot: TaskSnapshot, *, ledger: EpisodeRewardLedger | None = None) -> dict[str, object]:
        self.reward._validate_snapshot(snapshot)
        m = snapshot.measurements
        inside, bottom, _ = object_geometry(m)
        checks = dict(task_running=snapshot.outcome == "running", sensor_evidence_valid=True)
        if self.stage == "S2":
            source = np.asarray(snapshot.source_position_xy_m)
            checks.update(no_prior_grasp=not (snapshot.grasp_attempt_active or snapshot.lifted_once
                           or m.left_contact or m.right_contact),
                          target_at_source=source.shape == (2,) and np.linalg.norm(
                              np.asarray(m.target_position_world_m[:2]) - source) <= self.reward_config.source_adjustment_tolerance_m,
                          target_supported=abs(bottom - m.source_support_z_m) <= self.reward_config.support_height_tolerance_m,
                          target_static=snapshot.target_static,
                          near_target=snapshot.target_distance_m <= self.config.s1_reach_distance_m + 1e-9,
                          outside_goal=not inside)
        elif self.stage == "S3":
            s2 = S2GraspLiftReward(self.reward_config).signals(snapshot)
            checks.update(current_lift=live_carry(snapshot), lifted_start=s2.goals["stable_hold"],
                          outside_goal=not inside)
        elif self.stage == "S4":
            checks.update(current_lift=live_carry(snapshot), full_footprint_inside=inside,
                          carried_entry=snapshot.carried_into_goal)
        else:
            checks.update(valid_placement=placement_valid(snapshot, self.reward_config,
                          separation_m=self.reward.config.release_separation_m))
        if not all(checks.values()):
            raise ValueError(f"invalid {self.stage} initial state: " + ", ".join(k for k, v in checks.items() if not v))
        self.reward.reset(snapshot, ledger=ledger or EpisodeRewardLedger.for_snapshot(snapshot))
        self._start_s = snapshot.sim_time_s
        self._certificate = dict(start_valid=True, training_stage=self.stage, checks=checks,
                                 sim_time_s=snapshot.sim_time_s, evidence_sim_time_s=snapshot.evidence_sim_time_s)
        self._result, self._reason = "RUNNING", None
        self._initialized = True
        return self.info(snapshot)

    def info(self, snapshot: TaskSnapshot, result=None) -> dict[str, object]:
        metrics = dict(self.reward.signals(snapshot).metrics if result is None else result.metrics)
        home = {}
        if self.stage == "S5":
            measured = self.reward.home_success.measure(snapshot)
            home = dict(stage_home_error_rad=measured.home_error_rad,
                        stage_max_joint_velocity_rad_s=measured.max_joint_velocity_rad_s,
                        stage_robot_at_home=measured.at_home, stage_robot_static=measured.robot_static,
                        stage_home_hold_s=self.reward.home_success.hold_s,
                        home_joint_tolerance_rad=self.reward.home_success.joint_tolerance_rad)
        return dict(training_stage=self.stage, stage_result=self._result,
                    stage_success=self._result == "SUCCESS", stage_timeout=self._result == "TIMEOUT",
                    stage_cut=self._result in {"SUCCESS", "TIMEOUT"}, stage_result_reason=self._reason,
                    stage_elapsed_s=snapshot.sim_time_s - self._start_s, stage_budget_s=self.budget_s,
                    start_valid=True, start_certificate=self._certificate,
                    assisted_initialization=True, full_unassisted_success=False,
                    stage_metrics=metrics, later_stage_reward_state=self.reward.reward_state(), **home)

    def step(self, transition: PickPlaceRewardInput):
        if not self._initialized or self._result != "RUNNING":
            raise RuntimeError("later stage requires a fresh handoff before sampling")
        snapshot = transition.next_snapshot
        elapsed = snapshot.sim_time_s - self._start_s
        result = self.reward.step(transition)
        if snapshot.outcome == "task_failure":
            self._result, self._reason = "FAILURE", snapshot.failure_reason
        elif result.stage_complete and elapsed <= self.budget_s + 1e-9:
            self._result, self._reason = "SUCCESS", self.reward.completion_goal
        elif elapsed + 1e-9 >= self.budget_s:
            self._result, self._reason = "TIMEOUT", "stage_budget_exhausted"
            if result.stage_complete:
                # A post-deadline sample cannot earn a completion bonus while
                # the episode is reported as a timeout. Ordinary process
                # feedback remains tied to the measured physical interval.
                components = dict(result.components)
                components[f"{self.stage.lower()}_stage_complete"] = 0.0
                components[f"{self.stage.lower()}_{self.reward.completion_goal}_complete"] = 0.0
                result = replace(result, reward=float(sum(components.values())), components=components,
                                 stage_complete=False, stage_success_event=False,
                                 completed_subgoals=tuple(k for k in result.completed_subgoals
                                                         if k != self.reward.completion_goal))
        cut = self._result in {"SUCCESS", "TIMEOUT"}
        if cut:
            self.reward._closed = True
            result = replace(result, truncated=True)
        return result, self.info(snapshot, result)
