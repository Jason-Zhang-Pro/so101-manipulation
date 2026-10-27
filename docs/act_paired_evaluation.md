# 配对 ACT 仿真评估

`so101_capstone.training.act_paired_evaluation` 比较初始 ACT 与指定的更新后 ACT。
它只运行模型推理，不创建 critic、optimizer 或 PPO rollout，也不更改原训练目录。

默认选取原训练中最早的无目标 HOME、目标已在目标区的 HOME，以及三个抓放场景。
每对复用原始 scene seed、目标生成位置/姿态和生成时刻，并为每集重建独立 world。
使用原训练的 `resolved_reward.json`、600 步等运行设置和 ACT 原生执行队列/temporal ensembling；
没有训练时的高斯探索。当前只支持 30 Hz、未启用环境随机化的 `pick_place_v2` 运行。

相同 seed 不保证 Gazebo 渲染、传感器回调和 HOME reset 反馈逐位相同。
这组固定场景用于定位行为变化，不是泛化成功率评测。

## 启动

先确认主机上评估使用的 ROS domain 未被占用。默认 domain 为 `120`，
Gazebo partition 每次随机生成。程序会拒绝已有资源冲突，不对原训练 domains 做预清理。
worker 自身的启动和退出清理仅针对此次独占的 domain/partition。

在项目根目录执行下面命令。脚本会加载 conda 和 ROS 环境，保留 ROS source
设置的 PYTHONPATH；覆盖该路径会令 `rclpy` 和 ROS 消息包无法导入。

```bash
bash scripts/evaluate_act_pair.sh \
  --run-directory output_simulation/finetuned/act_ppo_20260927_153051_from_0005000 \
  --updated-checkpoint output_simulation/finetuned/act_ppo_20260927_153051_from_0005000/checkpoints/update_000021/pretrained_model \
  --output-directory output_simulation/evaluations/paired_act_initial_vs_update21_new_run \
  --device cuda \
  --execute
```

默认使用 `$HOME/miniconda3` 下的 `lerobot` 环境和 ROS Humble。
可用 `EVAL_CONDA_SETUP`、`EVAL_CONDA_ENV`、`EVAL_ROS_SETUP` 指定其他安装位置。

省略 `--execute` 只打印计划，不导入 ML/ROS 模块、不加载权重、不启动 Gazebo。
输出目录必须不存在，也不能位于原训练目录内。重复评估应指定新的输出目录。
`--reach-scenes 1` 将评估缩短为两种 HOME 和一个 reach 场景，共六集。
`--device cpu` 可以把策略推理放在 CPU；相机渲染仍可能使用 GPU。

## 结果与异常

- `plan.json`：来源模型、原始场景、执行设置、奖励原文与 SHA-256。
- `comparison.json`：已完成 episode 的汇总，只有全部完成后 `complete` 才为 `true`。
- 每个 episode 的 `steps.jsonl`：逐步关节/动作、原始 task info、reward、终止标志。
- 每个 episode 的 `summary.json`：最近距离、抓取/抬升/放置证据、末尾 100 帧 HOME 各关节误差。
- `initial`、`nearest`、`final` 的 top/gripper PNG 及 JSON：处理后的相机输入与几何证据。
- `worlds/world_00/gazebo.log`：每个模型的独立 worker 启动及清理日志。
- `failure.json`：错误与已完成 episode/step 数；部分结果不会当作完成评估。

几何 JSON 分开保存时间对齐后的 task-context XYZ 与最新 joint feedback 的 FK。
`opening_from_latest_joint_feedback_rad` 是最新夹爪关节相对保存 HOME 的估计开度，
其时间戳可能略晚于奖励使用的对齐证据。目标预先放在目标区的 idle_completed
场景不会被计为模型学会了放置。
