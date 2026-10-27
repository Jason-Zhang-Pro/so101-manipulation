# ACT PPO：记录后关闭 Gazebo，采集前再创建世界

正式入口 `scripts/finetune_act_ppo_so101.sh` 使用
`so101_capstone.training.act_ppo_multiworld_gazebo`。从本次修改起，
`WORLD_COUNT=1` 和多世界均采用同一生命周期，不需要额外开关：

1. 每个新启动的世界只准备一次 episode，准备时不调用 target 删除服务。
2. episode 内保留 target，直到正常终止、截断或达到最大步数。
3. 等当前 wave 的完整 episode 采集结束，确认每个实际完成的 step 都有完整
   PPO transition，actor 引用有效，最后一个 transition 标记 episode 结束。
4. 将完整 episode 复制到主进程的 PPO rollout，保存逐步 JSONL 和 episode
   汇总，将文件与目录同步到磁盘。所有成功 episode 均通过这一步后才关闭世界。
5. 关闭旧世界并确认该 worker 的 ROS domain / Gazebo partition 内受管进程
   清理完成，此时不创建替代世界。主进程中的 ACT、critic、优化器及 rollout 保留。
6. 检查累计 rollout 是否达到阈值。达到则执行 PPO 更新并按配置保存 checkpoint；
   不足则保留数据，继续下一轮采集。PPO 期间没有托管的 Gazebo 世界，worker
   通信进程可保持存活，但不持有 ROS runtime、Gazebo server 或 GUI。
7. 仅在下一 wave 确定需要采集时，创建替代世界，完成 ready、reset 和相机同步
   检查后开始 episode。最后一轮也关闭旧世界，但不创建无人使用的新世界。

即：`记录并同步 → 关闭世界 → PPO（数据足够时）→ 下一 wave 前创建 → reset → 采集`。

准备过程失败的世界也不能再次 prepare，必须先重建。step/文件完整性检查失败
会终止本次训练，不会继续计划重建或开始下一轮。退出时仍会关闭受管进程。
正常 episode 保存后的重建失败只影响下一轮世界可用性，不丢弃或重复加入
已记录的 PPO transitions，也不计作该 episode 采集失败。
关闭操作必须收到世界编号匹配的 `world_stopped`。worker 已退出、失联或拒绝
关闭时，父进程先停止该 worker，再按原有 domain/partition 范围确认资源清空；
清理仍失败则终止训练，不进入 PPO。启动或恢复失败的世界也经过此清理屏障。

## 配置与日志

- `FINAL_TARGET_ABSENT_S=0` 必须保留；非零值在启动世界前报错，不再支持
  managed PPO 中的 episode 尾部删除窗口，包括 legacy reward。
- `GAZEBO_GUI=true` 仍显示 GUI；采集结束后窗口关闭，PPO 期间没有窗口，
  下一轮采集前重新出现。
- `WORLD_RESTART_MAX_ATTEMPTS` / `WORLD_DEFERRED_RESTART_MAX_ATTEMPTS` 仍管理
  重建尝试次数；设为 0 的原延迟故障恢复语义保留，不会关闭正常 episode 后的重建。
- 30 Hz policy/environment/reward 步进、reward 数值、KL、相机几何、关节限位
  和已有机器人回 HOME reset 流程没有改动。环境随机化模块没有改动。

每轮输出目录新增：

```text
episode_steps/episode_000000.jsonl
episodes/episode_000000.json
metrics.jsonl
worlds/world_00/gazebo.log
```

逐步文件 `step_index` 从 1 开始，每行包含对应 action/latent、joint state、
critic context、reward/value/next_value、log probability、actor 引用与终止标志。
episode 汇总的 `step_records` 给出文件路径和行数；`recorded_transition_count`
应等于 `step_count`。主进程内的训练图像仍按原流程保留到 PPO 更新；逐步 JSONL
不额外保存 RGB，不是完整数据集或可恢复未更新 rollout 的检查点。

终端先显示 `EPISODE RECORDED` 和 `EPISODE WORLDS CLOSED`，随后按需执行 PPO。
下轮开始前显示 `PREPARE WORLDS FOR NEXT WAVE` 与
`WORLD RECOVERY ... reason=episode_complete`。
`metrics.jsonl` 中的 `episode_stop_begin`、`world_stopped` 与稍后的
`restart_ready` / `restart_failed` 记录关闭和按需启动的顺序。
Gazebo 日志追加 `episode_complete: launch attempt ...`，不覆盖旧日志。

该行为不要求更换 ACT 权重或丢弃已完成 PPO 更新的训练检查点。
原有 checkpoint 对 reward、观测、策略和探索配置的兼容性校验继续生效。
正在运行的父进程不会自动切换到新调度；下一次启动才使用修改后的流程。

## 范围与验证

本次修改覆盖正式 shell 和上面所列的托管 Python 入口。较早的
`so101_capstone.training.act_ppo_gazebo` 直接 Python 入口连接外部已启动的世界，
不管理该进程，仍保留旧清理行为；不要用它代替这里的防崩溃运行流程。
policy evaluation、Residual PPO 和手动 target 管理工具不在本次修改范围内。

训练耗时会增加，尤其每轮只有 20 秒仿真而 Gazebo 启动较慢时。此次改动绕开
运行中动态删除 target 的已知触发路径，没有修补系统 DART 库，也不保证消除
其它 Gazebo、相机或通信故障。

验证分为：Python 单元测试、shell 静态语法检查、Gazebo live smoke。
本次不自动启动 Gazebo 或训练；实际多次重建、GUI 和相机重新就绪仍需后续
单世界/多世界 live smoke 验证，不能用单元测试结果替代。
