# ACT PPO 作废 step 统计

正式微调脚本的单世界和多世界均自动启用，不需要新参数。统计与模型、checkpoint
和原有日志保存在同一次训练的 output 目录中：

```text
<本次 output>/
  discarded_steps.json       # 实时累计统计，失败事件发生后原子更新并同步到磁盘
  episode_failures/          # 每次失败的 world、attempt、时间、计数与 crash 证据
  metrics.jsonl             # episode_failure 和 discarded_steps 事件
  tensorboard/              # 启用 TensorBoard 时的 discarded/* 曲线
  checkpoints/
```

累计文件在训练开始时即创建（初始为 0），不依赖训练正常结束才保存。
发现 episode 失败后即写入，其他世界继续采集时也可查看，不必等整个 wave 完成。
正常结束的 `summary.json` 也包含 `discarded_step_statistics`。已有历史运行输出
不会被自动回填或改写；这些记录用于修改后的新运行。

## 计数含义

| 字段 | 含义 |
| --- | --- |
| `discarded_environment_steps` | 失败 episode 中，训练主进程已收到成功结果、但整段 episode 被作废的 policy/environment steps 总数 |
| `gazebo_crash_discarded_environment_steps` | 上述 step 中，失败时有当前 Gazebo launch 崩溃证据的子集 |
| `failure_discarded_transition_count` | 失败 episode 的 PPO buffer 实际丢弃的 transition 总数 |
| `gazebo_crash_discarded_transition_count` | 其中有 Gazebo crash 证据的 transition 数 |
| `failure_episode_count` | 失败 episode 尝试次数，重试是新的 attempt |
| `gazebo_crash_episode_count` | 有 crash 证据的失败 episode 次数；不是整个系统所有进程崩溃次数 |
| `unconfirmed_step_count` | 尝试发送但未收到有效 step_result 的 step 请求数，不加入已完成 step 总数 |
| `gazebo_crash_unconfirmed_step_count` | 上一项中有 crash 证据的子集 |
| `final_partial_rollout_discarded_transition_count` | 正常训练结束时，未凑满 PPO update 阈值而丢弃的 transition；与 crash 分开 |
| `aborted_rollout_discarded_transition_count` | 训练器受控异常退出时，主 rollout 中尚未进入 PPO 更新的数据；不直接归因于 crash |
| `interrupted_update_transition_count` | PPO 更新过程被中断时涉及的数据量，可能已部分使用，不加入确认作废总数 |
| `discarded_transition_count` | 失败 episode + 正常结束尾部 + 受控异常退出时未更新 rollout 的确认作废总量 |

例如：一个 600-step episode 已成功收到 300 个 step，第 301 个请求没有返回，
当前 Gazebo 日志确认 segmentation fault，则累计增加 **300 个作废 step** 和
**1 个未确认 step 请求**。不会把未执行的剩余步数加进去，也不会假定第 301 步
已经完成。

少数 target spawn 失败发生在 step_result 已收到、但该步尚未写入 PPO buffer
之间，此时已完成 step 与 buffer transition 数会差 1，因此两者分别记录。

每个失败事件按 `(world_index, episode_attempt_index)` 去重。`by_world` 和
`by_failure_category` 给出失败 episode 分项；正常/异常退出尾部不按 world 或 crash
归因。`failure_events` 保存明细文件引用。

## Gazebo crash 归因

失败时只检查当前 launch 的 Gazebo 日志范围，最多读取末尾 256 KiB。
识别 Gazebo 标记的 segmentation fault、abort 等致命信号及对应信号退出码。
明细保存匹配日志、路径与读取区间；保留原始错误类型和原始失败类别。

- 相机超时、普通通信超时不会自动认定为 Gazebo crash。
- 每次世界重建更新日志起点，避免把上一次 crash 算到下一回合。
- 通信失败时在强制清理进程前取证，清理后的新日志不会覆盖已采集的证据。
- `confirmed=false` 表示没有确认，不代表证明 Gazebo 没有崩溃。缺少日志或读取
  范围被截断时，统计可能低于实际 crash 数；明细保留相应状态。
- episode 完整记录后的计划重建不会产生作废计数。初次启动失败尚无 policy
  step，也不会凭空计入损失。

统计针对训练器已识别并记录的失败 episode，以及主 rollout 的受控退出。
Ctrl+C 等能够进入异常处理的退出会尝试记录主 rollout；系统掉电、SIGKILL 或
写盘故障不能保证留下最后一次事件。尚在当前 wave 内、未交给主 rollout 的
数据也不能视为已完成持久化；不要把该文件理解为任意中断或硬件故障的完整审计。

终端每次 episode 失败显示 `wasted steps`、`unconfirmed`、`Gazebo crash`，以及
累计 `wasted total` / `crash wasted`。每轮 `[TRAIN]` 行也显示累计值。

本次仅做 Python 单元与集成测试（模拟 worker 返回和日志），未启动 Gazebo、训练
或主动制造真实崩溃；实际崩溃场景的 live 验证仍待进行。
