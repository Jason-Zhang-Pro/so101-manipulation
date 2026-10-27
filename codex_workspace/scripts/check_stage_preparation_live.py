"""Bounded one-world preparation check; no model loading or optimizer."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[f"S{i}" for i in range(6)], default="S2")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--domain", type=int, default=126)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    token = uuid.uuid4().hex[:8]
    output = root / "codex_workspace/reports" / f"stage_start_{args.stage}_{datetime.now():%Y%m%d_%H%M%S}_{token}"
    plan = dict(stage=args.stage, ros_domain_id=args.domain, gz_partition=f"so101_stage_check_{token}",
                output_directory=str(output), optimizer_enabled=False, policy_check_steps=2)
    print(json.dumps(plan), flush=True)
    if not args.execute:
        return
    from so101_capstone.training.act_paired_evaluation import assert_resources_unused, _write_json
    from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerConfig, GazeboWorkerClient
    from so101_capstone.rewards.training_stages import StageTrainingConfig
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    assert_resources_unused(plan)
    os.environ["SO101_ENVIRONMENT_RANDOMIZATION_ENABLED"] = "false"
    os.environ["SO101_IMAGE_AUGMENTATION_ENABLED"] = "false"
    os.environ["SO101_ASSISTED_GRASP_ENABLED"] = "true"
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    reward = PickPlaceRewardConfig.load(root / "config/pick_place_reward.json")
    stages = StageTrainingConfig.load(root / "config/act_ppo_stages_s0_s5.json")
    cfg = GazeboWorkerConfig(world_index=0, ros_domain_id=args.domain, gz_partition=plan["gz_partition"],
        gazebo_gui=False, project_root=root, output_directory=output,
        startup_feedback_timeout_s=45, startup_max_attempts=1, restart_max_attempts=1,
        max_episode_steps=301, reward_profile="pick_place_v2", reward_config=reward.to_dict())

    def require(client, response, kind, name):
        _write_json(output / f"{name}.json", response)
        print(json.dumps(dict(event=name, type=response.get("type"),
                             error=response.get("error_message"))), flush=True)
        if response.get("type") != kind:
            raise RuntimeError(f"{name} failed; see {output / (name + '.json')}")
        return response

    try:
        with GazeboWorkerClient(cfg) as client:
            require(client, client.start(), "ready", "started")
            client.prepare_episode_async(1, task_mode="idle_empty" if args.stage == "S0" else "pick_place", scene_seed=1009)
            require(client, client.receive(timeout_s=150), "episode_prepared", "prepared")
            if args.stage != "S0":
                client.spawn_target_async((.26, 0., .768), (0., 0., 0., 1.))
                spawned = require(client, client.receive(timeout_s=30), "target_spawned", "spawned")
                if not spawned.get("success"):
                    raise RuntimeError("Source target spawn failed")
            client.start_training_stage_async(args.stage, stages.to_dict())
            started = require(client, client.receive(timeout_s=600), "stage_started", "stage_started")
            info = started["info"]
            assert info["training_stage"] == args.stage and info["episode_step"] == 0
            assert info["stage_elapsed_s"] == 0 and info["frame_zero"]
            assert info["start_valid"] and info["stage_initialization"]["valid"]
            command = info["reset_joint_targets_rad"]
            for index in range(1, 3):
                client.step_async(command)
                result = require(client, client.receive(timeout_s=30), "step_result", f"step_{index}")
                assert result["info"]["episode_step"] == index
                assert result["info"]["training_stage"] == args.stage
                assert not result["info"].get("preparation_only")
                if result["terminated"] or result["truncated"]:
                    break
            require(client, client.stop_world(), "world_stopped", "closed")
        _write_json(output / "summary.json", {**plan, "success": True})
        print(f"PASSED {args.stage}: {output}", flush=True)
    except BaseException as error:
        _write_json(output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
