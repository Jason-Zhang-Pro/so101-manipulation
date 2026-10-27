"""Interactive, single-world view of the existing S0--S5 preparation path.

No ACT model or optimizer is created. The official controller is unchanged.
One target is spawned once and is retained until the world is closed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import uuid


class InspectionClosed(Exception):
    pass


class Panel:
    def __init__(self, output: Path, idle_timeout: float):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.window = tk.Tk()
        self.window.title("SO-101 | S0–S5 起点目视检查")
        self.window.geometry("660x380")
        self.window.minsize(600, 340)
        self.closed = False
        self.next_requested = False
        self.idle_timeout = idle_timeout
        self.alive = lambda: True
        self.status = tk.StringVar(value="正在启动一个 Gazebo 世界……")
        self.details = tk.StringVar(value="自动准备演示，不运行 ACT 或 PPO。")
        frame = ttk.Frame(self.window, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=self.status, font=("sans", 17, "bold"),
                  wraplength=610).pack(anchor="w", pady=(0, 14))
        ttk.Label(frame, textvariable=self.details, wraplength=610,
                  justify="left").pack(anchor="w", fill="x", pady=(0, 16))
        self.next_button = ttk.Button(frame, text="继续下一段", command=self._next,
                                      state="disabled")
        self.next_button.pack(side="left", anchor="s", padx=(0, 14))
        ttk.Button(frame, text="结束检查并关闭世界", command=self._close).pack(side="left", anchor="s")
        ttk.Label(frame, text=f"日志：{output.name}\n停留超过 {idle_timeout / 60:g} 分钟后自动结束。",
                  wraplength=300).pack(side="right", anchor="s")
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.pump()

    def _next(self):
        self.next_requested = True

    def _close(self):
        self.closed = True

    def pump(self):
        if self.closed:
            raise InspectionClosed("用户结束检查")
        try:
            self.window.update_idletasks()
            self.window.update()
        except self.tk.TclError as error:
            raise InspectionClosed("检查面板已关闭") from error
        if self.closed:
            raise InspectionClosed("用户结束检查")
        if not self.alive():
            raise InspectionClosed("Gazebo launch 已退出")

    def moving(self, message):
        self.status.set(message)
        self.details.set("正在执行现有自动准备控制器。可在 Gazebo 中观察运动；中间不会重建世界。")
        self.next_button.configure(state="disabled")
        self.pump()

    def pause(self, title, details, *, terminal=False):
        self.status.set(title)
        self.details.set(details + "\n\n世界已暂停，请勿点击 Gazebo 的播放或重置。"
                         + ("\n观察完毕后点击“结束检查并关闭世界”。" if terminal
                            else "\n观察完毕后点击“继续下一段”。"))
        self.next_requested = False
        self.next_button.configure(state="disabled" if terminal else "normal")
        deadline = time.monotonic() + self.idle_timeout
        while terminal or not self.next_requested:
            self.pump()
            if time.monotonic() >= deadline:
                raise InspectionClosed("目视检查停留时间已到")
            time.sleep(.05)
        self.next_button.configure(state="disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--domain", type=int, default=126)
    parser.add_argument("--idle-timeout", type=float, default=1800)
    args = parser.parse_args()
    if not 0 <= args.domain <= 232 or not 0 < args.idle_timeout <= 3600:
        parser.error("domain must be 0..232 and idle-timeout must be 0..3600 seconds")
    root = Path(__file__).resolve().parents[2]
    token = uuid.uuid4().hex[:8]
    output = root / "codex_workspace/reports" / f"stage_gui_{datetime.now():%Y%m%d_%H%M%S}_{token}"
    plan = dict(ros_domain_id=args.domain, gz_partition=f"so101_stage_gui_{token}",
                output_directory=str(output), gazebo_gui=True, world_count=1,
                rebuild_between_stages=False, policy_enabled=False, optimizer_enabled=False,
                scope="visual preparation checkpoints, not learned stage performance")
    plan["assisted_grasp_enabled"] = True
    print(json.dumps(plan), flush=True)
    if not args.execute:
        return

    import numpy as np
    from so101_capstone.training.act_paired_evaluation import assert_resources_unused, _write_json, _json_default
    from so101_capstone.training.act_ppo_multiworld_worker import GazeboWorkerConfig, _WorkerRuntimeHost
    from so101_capstone.environments.stage_preparation import StagePreparationConfig, StagePreparationController
    from so101_capstone.rewards.pick_place_reward import PickPlaceRewardConfig
    from so101_capstone.rewards.training_stages import StageTrainingConfig

    assert_resources_unused(plan)
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), GZ_PARTITION=plan["gz_partition"],
                      IGN_PARTITION=plan["gz_partition"],
                      SO101_ENVIRONMENT_RANDOMIZATION_ENABLED="false",
                      SO101_IMAGE_AUGMENTATION_ENABLED="false",
                      SO101_ASSISTED_GRASP_ENABLED="true")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    panel = Panel(output, args.idle_timeout)
    reward = PickPlaceRewardConfig.load(root / "config/pick_place_reward.json")
    stages = StageTrainingConfig.load(root / "config/act_ppo_stages_s0_s5.json")
    cfg = GazeboWorkerConfig(world_index=0, ros_domain_id=args.domain, gz_partition=plan["gz_partition"],
        gazebo_gui=True, project_root=root, output_directory=output,
        startup_feedback_timeout_s=45, startup_max_attempts=1, restart_max_attempts=1,
        max_episode_steps=301, reward_profile="pick_place_v2", reward_config=reward.to_dict())
    host = _WorkerRuntimeHost(cfg, top_buffer=bytearray(480 * 640 * 3),
                              gripper_buffer=bytearray(480 * 640 * 3))
    status = "starting"

    def record(event, **data):
        item = dict(event=event, wall_time=datetime.now().isoformat(), **data)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)

    class VisualPreparation(StagePreparationController):
        def _phase(self, name):
            super()._phase(name)
            record("phase", phase=name, preparation_steps=self.steps)
            view = {
                "CLOSE_AND_CERTIFY": ("S2 起点：开夹接近目标", "下一段：闭夹、确认真实抓持、抬起并稳定。"),
                "TRANSPORT_CLEARANCE": ("S3 起点：已抓持并抬起", "下一段：持物搬运到放置区上方。"),
                "LOWER_TO_SUPPORT": ("S4 起点：持物悬停于放置区", "下一段：降低物体、松夹爪、空夹爪抬起。"),
            }.get(name)
            if view:
                _write_json(output / f"{name}_snapshot.json", asdict(self.snapshot))
                panel.pause(*view)
            panel.moving(f"自动准备：{name}")

        def _move_step(self, translation, axis, **kwargs):
            try:
                return super()._move_step(translation, axis, **kwargs)
            except Exception as error:
                _write_json(output / "failed_motion.json", dict(
                    phase=self.phase, preparation_steps=self.steps, error=str(error),
                    translation_m=np.asarray(translation).tolist(), axis_world=np.asarray(axis).tolist(),
                    seed_command_rad=self.command.tolist(), snapshot=asdict(self.snapshot)))
                raise

    try:
        started = host.start_world(max_attempts=1, reason="initial_startup")
        _write_json(output / "started.json", started)
        if started["type"] != "ready":
            raise RuntimeError("Gazebo 启动失败，请查看 started.json")
        panel.alive = lambda: host.launch_process is not None and host.launch_process.poll() is None
        prepared = host.prepare_episode(episode_attempt_index=1, task_mode="pick_place", scene_seed=1009)
        _write_json(output / "prepared.json", prepared)
        record("view", stage="S0", kind="visual_home_only")
        panel.pause("S0 起点：HOME，无目标", "这里展示 HOME 起点。为保持同一物理世界，此处不运行 S0 成功率评估。\n下一段：生成一个目标，展示 S1 起点。")
        runtime = host._require_runtime()
        response = runtime.spawn_target((.26, 0., .768), (0., 0., 0., 1.))
        if not getattr(response, "success", False):
            raise RuntimeError("目标生成失败")
        source = runtime.start_training_stage("S1", stages.to_dict())
        _write_json(output / "S1_start.json", source.info)
        record("view", stage="S1", kind="certified_source_start")
        panel.pause("S1 起点：HOME，目标已生成", "下一段：抬高夹爪、开夹、从上方接近目标，准备 S2 起点。\n这段动作由自动准备控制器执行。")
        environment = runtime.coordinator.environment
        environment.begin_stage_preparation()
        initial = environment.task_snapshot
        if initial is None:
            raise RuntimeError("准备前缺少实际测量")

        def advance(command):
            panel.pump()
            result = runtime.coordinator.step(command)
            if (not result.info.get("preparation_only") or result.info.get("episode_step") != 0
                    or result.reward != 0 or result.terminated or result.truncated):
                raise RuntimeError("准备步骤违反训练隔离或物理终止条件")
            snapshot = environment.task_snapshot
            with (output / "preparation_trace.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(command_rad=list(command), snapshot=asdict(snapshot),
                                             grasp_attachment=result.info.get("grasp_attachment")),
                                        allow_nan=False, default=_json_default) + "\n")
            return snapshot

        controller = VisualPreparation(reward, StagePreparationConfig(**(stages.stage_preparation or {})))
        report = controller.run("S5", initial, advance)
        observation, info = runtime.coordinator.handoff_training_stage(
            "S5", stages.to_dict(), command_joint_targets_rad=report.last_command_joint_targets_rad)
        _write_json(output / "S5_start.json", info)
        record("view", stage="S5", kind="certified_placed_start", certificate=report.certificate)
        status = "all_preparation_checkpoints_reached"
        panel.pause("S5 起点：物体已放置，空夹爪抬起", "S0–S5 起点展示完成，期间只有一个世界和一个目标。\nS5 训练从这里学习回到 HOME，本工具没有运行 ACT。", terminal=True)
    except InspectionClosed as error:
        record("inspection_closed", reason=str(error))
    except Exception as error:
        status = "preparation_failed"
        _write_json(output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        record("failure", error=str(error))
        try:
            panel.pause("准备失败：保留现场供目视检查", str(error), terminal=True)
        except InspectionClosed as closed:
            record("inspection_closed", reason=str(closed))
    finally:
        try:
            panel.status.set("正在关闭本次检查世界……")
            panel.window.update()
        except panel.tk.TclError:
            pass
        try:
            host.stop_world()
        finally:
            _write_json(output / "summary.json", dict(status=status, cleanup=host.last_cleanup_summary))
            try:
                panel.window.destroy()
            except panel.tk.TclError:
                pass
        print(f"CLOSED: {output}", flush=True)


if __name__ == "__main__":
    main()
