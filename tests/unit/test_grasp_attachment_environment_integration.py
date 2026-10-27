"""Attachment integration using real caches/core/coordinator and fake physics.

The fake backend and sensor trace test ordering, accounting and evidence
plumbing. They do not establish Gazebo constraint or friction performance.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from so101_capstone.environments.contact_state import FingerContactCache
from so101_capstone.environments.gazebo_grasp_attachment import GazeboGraspAttachmentError, GraspAttachmentStateCache
from so101_capstone.environments.image_cache import RgbImageCache
from so101_capstone.environments.reach_env_core import ReachEnvironmentStateError, So101ReachEnvironmentCore
from so101_capstone.environments.simulation_clock import SimulationClockCache
from so101_capstone.environments.simulation_runtime import create_ros_simulation_runtime
from so101_capstone.environments.simulation_step import SimulationStepSchedule
from so101_capstone.environments.simulation_step_coordinator import (
    DeterministicSimulationStepCoordinator, SimulationStepCoordinatorConfig,
)
from so101_capstone.environments.state_cache import (
    JointStateCache, MEASURED_JOINT_NAMES, TargetCubeCollectionCache, TargetCubeStateCache,
)
from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
from so101_capstone.rewards.task_state import TASK_CONTEXT_DIM, GRASP_ATTACHMENT_CONTEXT_DIM
from so101_capstone.robot import compute_grasp_point_pose

from test_gazebo_grasp_attachment import controller


class AttachmentHarness:
    def __init__(self, *, preparing=False):
        self.attachment_cache = GraspAttachmentStateCache()
        self.client, self.backend, _ = controller(status_provider=self.attachment_cache.wait_for_sim_time)
        self.reward = PickPlaceRewardConfig(reward_schema_version=5)
        self.q = tuple(self.reward.home_joint_positions_rad[:-1]) + (.1236,)
        self.command = tuple(self.reward.home_joint_positions_rad[:-1]) + (.0333,)
        self.target = compute_grasp_point_pose(dict(zip(MEASURED_JOINT_NAMES, self.q))).position.copy()
        self.target[2] -= .006
        # The synthetic source plane supports the trace's object exactly.
        self.reward = replace(self.reward, source_support_z_m=float(self.target[2] - .015))
        self.core = So101ReachEnvironmentCore(
            reward_profile="pick_place_v2", reward_config=self.reward,
            goal_region_center_world_m=(.078, -.3, .7515),
            grasp_attachment_controller=self.client)
        self.joint = JointStateCache()
        self.cubes = TargetCubeCollectionCache()
        self.cube = TargetCubeStateCache()
        self.clock = SimulationClockCache()
        self.contacts = FingerContactCache()
        self.top, self.gripper = RgbImageCache(), RgbImageCache()
        self.t, self.seq = 1., 0
        self.contact = False
        self.order, self.steps, self.published = [], [], []
        self._feedback()
        self.coordinator = DeterministicSimulationStepCoordinator(
            environment=self.core, world_stepper=SimpleNamespace(step=self._physics_step),
            schedule=SimulationStepSchedule(), simulation_clock_cache=self.clock,
            joint_cache=self.joint, cube_collection_cache=self.cubes, cube_cache=self.cube,
            top_image_cache=self.top, gripper_image_cache=self.gripper,
            publish_joint_targets=self._publish, contact_cache=self.contacts,
            grasp_attachment_controller=self.client,
            config=SimulationStepCoordinatorConfig(feedback_timeout_s=.01))
        self.coordinator.reset_from_latest(now_monotonic_s=100.)
        if preparing:
            self.core.begin_stage_preparation()
        self.contact = True

    def _publish(self, command):
        self.order.append(("publish", self.client.metadata()["active"]))
        self.published.append(tuple(command))

    def _physics_step(self, count):
        self.order.append(("physics", self.client.metadata()["active"]))
        self.steps.append(count)
        self.t += count * .001
        self._feedback()

    def _feedback(self):
        self.seq += 1
        self.backend.status["sim_time_s"] = self.t
        self.attachment_cache.update(self.backend.status)
        self.joint.update(MEASURED_JOINT_NAMES, self.q, (0.,) * 6,
                          received_monotonic_s=100., source_sim_time_s=self.t)
        self.cube.update("target_cube_1", self.target, (0., 0., 0., 1.),
                         source_sequence=self.seq, source_sim_time_s=self.t, received_monotonic_s=100.)
        self.cubes.update(("target_cube_1",), source_sequence=self.seq,
                          source_sim_time_s=self.t, received_monotonic_s=100.)
        self.clock.update(self.t, received_monotonic_s=100.)
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.top.update(image, source_timestamp_s=self.t, received_monotonic_s=100.)
        self.gripper.update(image, source_timestamp_s=self.t, received_monotonic_s=100.)
        for side in ("left", "right"):
            self.contacts.update(side, sim_time_s=self.t,
                collision_pairs=(("robot_finger", "target_cube_1::link::collision"),) if self.contact else ())

    def step(self, command=None):
        return self.coordinator.step(self.command if command is None else command, now_monotonic_s=100.)

    def attach(self):
        for _ in range(10):
            result = self.step()
            if self.client.metadata()["active"]:
                return result
        pytest.fail("Trace did not produce a real tracker contact certificate")


@pytest.mark.parametrize("preparing", [False, True])
def test_first_attach_annotates_current_snapshot_context_without_an_extra_step(preparing):
    h = AttachmentHarness(preparing=preparing)
    result = h.attach()
    m = h.core.task_snapshot.measurements
    assert m.grasp_attachment_active and m.grasp_attachment_ever_used
    assert result.info["task_snapshot"]["grasp_attachment_active"]
    assert result.info["grasp_attachment"]["active"]
    assert result.info["grasp_attachment"]["events"] == ["grasp_attachment_attached"]
    assert h.core._task_tracker.snapshot is h.core.task_snapshot
    assert m.sim_time_s == result.end_clock.sim_time_s == h.t
    assert m.executed_command_joint_targets_rad == h.command
    assert m.grasp_attachment_command_rad == h.command[-1]
    assert m.grasp_attachment_measured_rad == h.q[-1]
    base_end = TASK_CONTEXT_DIM + GRASP_ATTACHMENT_CONTEXT_DIM
    assert result.info["critic_task_context"][TASK_CONTEXT_DIM] == 1.
    assert result.info["critic_task_context"][TASK_CONTEXT_DIM + 1] == 1.
    np.testing.assert_array_equal(result.info["critic_task_context"][base_end:],
                                  h.core._pick_place_reward.critic_history_context())
    assert len(h.steps) == len(h.published)
    assert h.steps == ([33, 33, 34] * 4)[:len(h.steps)]
    assert sum(h.steps) * .001 == pytest.approx(h.t - 1.)
    assert result.info["episode_step"] == (0 if preparing else len(h.steps))
    if preparing:
        assert result.reward == 0 and result.info["preparation_only"]


def test_zero_contacts_after_attach_are_not_fabricated_and_attempt_stays_valid():
    h = AttachmentHarness()
    h.attach()
    h.contact = False
    # First contact transition interval may be ambiguous. Real contact cache
    # then publishes fresh, unambiguously empty samples, not a sensor outage.
    h.step()
    result = h.step()
    m = h.core.task_snapshot.measurements
    assert m.contact_valid and not m.left_contact and not m.right_contact
    assert m.grasp_attachment_active and h.core.task_snapshot.grasp_valid
    assert result.info["success_physics"] == "rigid_grasp_assisted"


def test_open_action_runs_attached_then_releases_at_measured_clearance_without_extra_physics():
    h = AttachmentHarness()
    h.attach()
    opened = h.command[:-1] + (h.q[-1] + h.client.release_delta_rad,)
    # Previous measured snapshot is not open; the next actual feedback is.
    h.q = h.q[:-1] + (opened[-1],)
    previous_steps = len(h.steps)
    result = h.step(opened)
    assert h.order[-2:] == [("publish", True), ("physics", True)]
    assert not result.info["grasp_attachment_active"]
    assert result.info["grasp_attachment_ever_used"]
    assert result.info["grasp_attachment"]["events"] == ["grasp_attachment_detached"]
    assert h.backend.attachments == h.backend.detachments == 1
    assert not result.info["valid_release"]  # The object was not delivered into goal.
    assert len(h.steps) == previous_steps + 1
    assert result.info["grasp_attachment"]["release_reference"] == "measured_confirmation"


def test_closing_action_cancels_requested_release_in_the_real_coordinator_path():
    h = AttachmentHarness()
    h.attach()
    opened = h.command[:-1] + (h.q[-1] + h.client.release_delta_rad,)
    waiting = h.step(opened)
    assert waiting.info["grasp_attachment"]["release_requested"]
    assert waiting.info["grasp_attachment_active"]
    h.q = h.q[:-1] + (opened[-1],)
    canceled = h.step(h.command)
    assert canceled.info["grasp_attachment_active"]
    assert not canceled.info["grasp_attachment"]["release_requested"]
    # New opening command with previously measured clearance can detach
    # before the action is published, without inserting a physics substep.
    released = h.step(opened)
    assert h.order[-2:] == [("publish", False), ("physics", False)]
    assert not released.info["grasp_attachment_active"]


def test_lost_backend_constraint_is_detected_before_reward_update():
    h = AttachmentHarness()
    h.attach()
    count = h.core.episode_step
    h.backend.status.update(state="detached", joint_entity=0)
    with pytest.raises(GazeboGraspAttachmentError, match="unexpectedly changed"):
        h.step()
    # The physical attempt happened, but no invalid transition was scored.
    assert h.core.episode_step == count
    assert len(h.steps) == count + 1


def test_active_attachment_blocks_logical_reset_before_cache_or_tracker_mutation():
    h = AttachmentHarness()
    h.attach()
    before = h.core.task_snapshot
    with pytest.raises(ReachEnvironmentStateError, match="Cannot reset"):
        h.coordinator.reset_from_latest(now_monotonic_s=100.)
    assert h.core.task_snapshot is before
    with pytest.raises(ReachEnvironmentStateError, match="Cannot reset"):
        h.core.reset_from_feedback(h.joint.latest(), h.cube.latest(), now_monotonic_s=100.,
                                   step_end_sim_time_s=h.t)
    assert h.core.task_snapshot is before


def test_annotation_preserves_stage_tail_and_marks_stage_result_as_assisted():
    h = AttachmentHarness()
    result = h.attach()
    suffix = [11., 22., 33., 44.]
    info = {**result.info, "critic_task_context": result.info["critic_task_context"] + suffix,
            "stage_success": True, "training_stage": "S2", "stage_result": "SUCCESS",
            "task_context_schema_version": 999}
    before_step = h.core.episode_step
    updated = h.core.annotate_grasp_attachment(info)
    assert updated["critic_task_context"][-4:] == suffix
    assert updated["task_context_schema_version"] == 999
    assert updated["stage_success"] and not updated["stage_unassisted_success"]
    assert updated["stage_success_physics"] == "rigid_grasp_assisted"
    assert updated["stage_result"] == "SUCCESS"
    assert h.core.episode_step == before_step


def test_s2_policy_grasp_continues_reward_chain_for_two_steps_after_attachment():
    from so101_capstone.rewards.training_stages import StageTrainingConfig
    from so101_capstone.rewards.later_stage_training import later_stage_context_dim
    h = AttachmentHarness(preparing=True)
    h.contact = False
    h.step()
    h.step()
    config = StageTrainingConfig(enable_later_stage_context=True)
    _, initial = h.coordinator.handoff_training_stage("S2", config, command_joint_targets_rad=h.command)
    assert initial["episode_step"] == 0 and not initial["grasp_attachment_active"]
    h.contact = True
    attached = h.attach()
    reward = h.core._later_stage.reward
    assert reward._previous is h.core.task_snapshot
    assert attached.info["critic_task_context"][-later_stage_context_dim()] == 2.
    paid_before = set(reward.ledger.paid)
    rewards = [h.step(), h.step()]
    assert reward._previous is h.core.task_snapshot
    assert paid_before <= reward.ledger.paid
    assert all(np.isfinite(result.reward) for result in rewards)
    assert all(result.info["grasp_attachment_active"] for result in rewards)
    assert h.backend.attachments == 1


def test_full_policy_keeps_exact_reward_continuity_after_attachment_without_reset():
    h = AttachmentHarness()
    h.attach()
    reward = h.core._pick_place_reward
    assert reward._previous is h.core.task_snapshot
    first = h.step()
    second = h.step()
    assert np.isfinite(first.reward) and np.isfinite(second.reward)
    assert reward._previous is h.core.task_snapshot
    assert second.info["episode_step"] == len(h.steps)
    assert h.backend.attachments == 1
    assert len(h.backend.calls) == 1  # No per-step ign service subprocess.


def test_core_rejects_legacy_and_proxy_modes_for_attachment():
    client, _, _ = controller()
    with pytest.raises(ValueError, match="pick_place_v2"):
        So101ReachEnvironmentCore(grasp_attachment_controller=client)
    with pytest.raises(ValueError, match="bilateral"):
        So101ReachEnvironmentCore(reward_profile="pick_place_v2",
                                 reward_config=PickPlaceRewardConfig(grasp_evidence_mode="proxy"),
                                 grasp_attachment_controller=client)


def test_runtime_flag_rejects_legacy_before_importing_or_starting_ros(monkeypatch):
    monkeypatch.setenv("SO101_ASSISTED_GRASP_ENABLED", "true")
    with pytest.raises(ValueError, match="pick_place_v2"):
        create_ros_simulation_runtime()


def test_runtime_preparation_and_deletion_check_attachment_before_external_calls():
    from test_simulation_runtime import make_runtime
    system = make_runtime()
    h = AttachmentHarness()
    h.attach()
    system.coordinator.environment = h.core
    system.coordinator.grasp_attachment_controller = h.client
    with pytest.raises(ReachEnvironmentStateError, match="Cannot reset"):
        system.runtime.prepare_episode()
    with pytest.raises(ReachEnvironmentStateError, match="Cannot reset"):
        system.runtime.delete_all_targets()


def test_ros_callback_feeds_stream_cache_and_records_parse_failures():
    pytest.importorskip("rclpy")
    from so101_capstone.environments.ros_reach_node import So101ReachRosNode, GRASP_ATTACHMENT_STATE_TOPIC
    from test_gazebo_grasp_attachment import FakeBackend
    import json
    cache = GraspAttachmentStateCache()
    errors = []
    node = SimpleNamespace(grasp_attachment_state_cache=cache,
                           get_logger=lambda: SimpleNamespace(error=errors.append))
    assert GRASP_ATTACHMENT_STATE_TOPIC == "/so101/grasp_attachment_state"
    So101ReachRosNode._grasp_attachment_callback(node, SimpleNamespace(data=json.dumps(FakeBackend().status)))
    assert cache.wait_for_sim_time(1., 0)["state"] == "detached"
    So101ReachRosNode._grasp_attachment_callback(node, SimpleNamespace(data="bad-json"))
    assert len(errors) == 1
    with pytest.raises(GazeboGraspAttachmentError):
        cache.wait_for_sim_time(1., 0)
