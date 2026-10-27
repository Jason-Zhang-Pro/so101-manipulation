"""One isolated, bounded physics check of certified attach/carry/release.

No ACT/optimizer, no target deletion, no interaction with other worlds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--domain", type=int, default=127)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    token = uuid.uuid4().hex[:8]
    output = root / "codex_workspace/reports" / f"grasp_attachment_live_{datetime.now():%Y%m%d_%H%M%S}_{token}"
    plan = dict(ros_domain_id=args.domain, gz_partition=f"so101_grasp_check_{token}",
                output_directory=str(output), gazebo_gui=False, world_count=1,
                assisted_grasp_enabled=True, optimizer_enabled=False, policy_enabled=False)
    print(json.dumps(plan), flush=True)
    if not args.execute:
        return
    from so101_capstone.training.act_paired_evaluation import assert_resources_unused, _write_json, _json_default
    from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerConfig, _WorkerRuntimeHost
    from so101_capstone.environments.stage_preparation import StagePreparationConfig, StagePreparationController
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    from so101_capstone.rewards.training_stages import StageTrainingConfig
    from so101_capstone.rewards.later_stage_reward import relative_slip_m

    assert_resources_unused(plan)
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), GZ_PARTITION=plan["gz_partition"],
                      IGN_PARTITION=plan["gz_partition"], SO101_ASSISTED_GRASP_ENABLED="true",
                      SO101_ENVIRONMENT_RANDOMIZATION_ENABLED="false",
                      SO101_IMAGE_AUGMENTATION_ENABLED="false")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    reward = PickPlaceRewardConfig.load(root / "config/pick_place_reward.json")
    stages = StageTrainingConfig.load(root / "config/act_ppo_stages_s0_s5.json")
    cfg = GazeboWorkerConfig(world_index=0, ros_domain_id=args.domain, gz_partition=plan["gz_partition"],
        gazebo_gui=False, project_root=root, output_directory=output,
        startup_feedback_timeout_s=45, startup_max_attempts=1, restart_max_attempts=1,
        max_episode_steps=301, reward_profile="pick_place_v2", reward_config=reward.to_dict())
    host = _WorkerRuntimeHost(cfg, top_buffer=bytearray(480 * 640 * 3),
                              gripper_buffer=bytearray(480 * 640 * 3))
    counters = dict(steps=0, attached_steps=0, max_attached_slip_m=0.0)
    events = []
    class LoggedPreparation(StagePreparationController):
        def _phase(self, name):
            super()._phase(name)
            print(json.dumps(dict(phase=name, steps=self.steps)), flush=True)
        def _move_step(self, translation, axis, **kwargs):
            try:
                return super()._move_step(translation, axis, **kwargs)
            except Exception as error:
                _write_json(output / "failed_motion.json", dict(phase=self.phase,
                    preparation_steps=self.steps, error=str(error), translation_m=list(translation),
                    axis_world=list(axis), seed_command_rad=self.command.tolist(), snapshot=asdict(self.snapshot)))
                raise
    try:
        started = host.start_world(max_attempts=1, reason="attachment_verification")
        _write_json(output / "started.json", started)
        if started["type"] != "ready":
            raise RuntimeError("isolated world failed to start")
        host.prepare_episode(episode_attempt_index=1, task_mode="pick_place", scene_seed=1009)
        runtime = host._require_runtime()
        response = runtime.spawn_target((.26, 0., .768), (0., 0., 0., 1.))
        if not response.success:
            raise RuntimeError("target spawn failed")
        source = runtime.start_training_stage("S1", stages.to_dict())
        _write_json(output / "S1_start.json", source.info)
        environment = runtime.coordinator.environment
        environment.begin_stage_preparation()
        def advance(command):
            result = runtime.coordinator.step(command)
            snapshot = environment.task_snapshot
            counters["steps"] += 1
            metadata = result.info.get("grasp_attachment", {})
            with (output / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=counters["steps"], command_rad=list(command),
                    grasp_attachment=metadata, snapshot=asdict(snapshot)),
                    allow_nan=False, default=_json_default) + "\n")
            if metadata.get("active"):
                counters["attached_steps"] += 1
                counters["max_attached_slip_m"] = max(counters["max_attached_slip_m"], relative_slip_m(snapshot))
            if metadata.get("events"):
                events.append(dict(step=counters["steps"], **metadata))
                print(json.dumps(events[-1]), flush=True)
            if counters["steps"] % 100 == 0:
                print(json.dumps(counters), flush=True)
            if (not result.info.get("preparation_only") or result.info.get("episode_step") != 0
                    or result.reward != 0 or result.terminated or result.truncated):
                raise RuntimeError("preparation physics/protocol failed")
            return snapshot
        controller = LoggedPreparation(reward, StagePreparationConfig(**(stages.stage_preparation or {})))
        report = controller.run("S5", environment.task_snapshot, advance)
        _, info = runtime.coordinator.handoff_training_stage("S5", stages.to_dict(),
            command_joint_targets_rad=report.last_command_joint_targets_rad)
        _write_json(output / "S5_start.json", info)
        assert counters["attached_steps"] > 0
        assert environment.task_snapshot.measurements.grasp_attachment_ever_used
        assert not environment.task_snapshot.measurements.grasp_attachment_active
        for index in range(2):
            result = runtime.coordinator.step(report.last_command_joint_targets_rad)
            assert result.info["training_stage"] == "S5" and not result.info.get("preparation_only")
            assert not result.info["grasp_attachment"]["active"]
            _write_json(output / f"S5_step_{index + 1}.json", result.info)
        _write_json(output / "summary.json", dict(success=True, counters=counters, events=events,
            certificate=report.certificate, scope="assisted preparation only, not learned-policy evaluation"))
        print(f"PASSED: {output}", flush=True)
    except BaseException as error:
        _write_json(output / "failure.json", dict(type=type(error).__name__, message=str(error), counters=counters, events=events))
        raise
    finally:
        host.stop_world()
        _write_json(output / "cleanup.json", host.last_cleanup_summary)


if __name__ == "__main__":
    main()
