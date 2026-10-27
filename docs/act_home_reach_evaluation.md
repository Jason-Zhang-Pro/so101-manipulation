# ACT：从初始零位回 HOME，再接近目标

入口：`scripts/evaluate_act_home_reach.sh`。这是可重复调用的连续推理测试，默认打开 Gazebo GUI；不会训练或加载 PPO optimizer。省略 `--execute` 只查看计划。

## 固定时间线

1. 创建独立世界，先通过 `/controller_manager/list_controllers` 确认 `position_controller` 已激活、已接管六个关节的 position 接口，并确认动作发布端已有匹配订阅者。此时保持新世界运行，让控制器完成激活；等待超时则中止，不能开始测试计时。然后使用机械臂出生时的六关节零位，准备阶段将零位稳定下来并确认相机反馈，随后交给 ACT，设为 `t=0`。不会预先命令 HOME，也不会调用 S0/S1 的 HOME warm-up。控制器等待、零位准备和相机同步均不占用前 3 秒。
2. `0–3 s`：无目标，完全由 ACT 决定动作，检查是否自行回 HOME。
3. 完成第 90 个控制 step 后发送目标创建请求，与是否到 HOME 无关。记录服务响应与首次物理反馈。相机源时间单独记录，不能据此声称已做图像目标检测。
4. 继续 300 step，到 `t=13 s` 停止。因此目标请求后观察 10 秒，正常总共 390 step。均为仿真时间；启动、推理和渲染可能使实际等待更久。
5. 先写完每步记录、summary 和截图，再关闭该世界；episode 内不删除 target。运行异常或手动 Ctrl+C 时保留已确认的步骤及失败原因，并退出测试世界。

保持原有 30 Hz 确定性物理步进、相机处理、绝对关节限位。测试不加 PPO 探索，不做图像增强或物理环境随机化。动作 chunk、执行步数和 temporal ensembling 完整继承所选模型；不通过命令隐藏地覆盖它们。

## 启动

在项目根目录执行，换成要测试的 `pretrained_model` 路径：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"
bash scripts/evaluate_act_home_reach.sh \
  --checkpoint "/absolute/path/to/pretrained_model" \
  --execute
```

可追加：

- `--episodes 3`：相同 seed 和目标重复三次，每次新建世界。
- `--headless`：关闭 GUI。
- `--device cpu`：使用 CPU，默认 CUDA。
- `--target-position 0.25 0.0 0.768`：目标世界坐标，单位米。这也是默认固定测试点，位于现有目标采样范围。固定位置方便比较，不代表随机目标成功率。
- `--seed 1009`：默认 seed。同 seed 不保证 Gazebo 位级重现。
- `--output-directory /absolute/new/directory`：输出到新的目录，不覆盖已有结果。
- `--ros-domain-id 120`：默认独立 domain 120，partition 每次唯一。若资源已占用则拒绝启动，不清理其他训练进程。
- `--reward-config /absolute/path/reward.json`：覆盖评分参考；默认优先使用模型保存的 `resolved_reward.json`，监督模型没有此文件时使用项目 `config/pick_place_reward.json`。比较不同模型时应核对评分参考一致。

## 结果

默认输出：`output_simulation/evaluation/home_reach_<时间>_<唯一编号>/`。启动时打印完整计划和输出路径。

- `plan.json`、`model_identity.json`、`resolved_reward.json`：模型配置、权重 SHA256、时间线、目标位置和评分参考。
- `summary.json`：各次测试汇总。
- `episode_001/steps.jsonl`：每一个已完成 step 的动作、实际关节、对齐的世界测量、诊断、奖励和传感器时效。
- `episode_001/events.jsonl`：控制器就绪确认、开始、目标请求/响应、首次 HOME、首次目标反馈，以及每秒进度。`prepared.json` 和 `summary.json` 也保存控制器状态、接管接口、订阅连接数及等待耗时。计划 schema 2 表示包含此控制器检查；旧 schema 1 测试未保证控制器先于计时激活。
- `episode_001/summary.json`：是否完整执行时间线、回 HOME 与接近目标统计。
- `episode_001/*_top.png`、`*_gripper.png`：初始、目标生成前、首次到 HOME（若发生）、首次目标反馈、最近目标、最终画面，配套 JSON 保存测量值。
- `episode_001/world_closed.json`：正常关闭世界的确认。
- `world_run_001/worlds/world_00/gazebo.log`：Gazebo 启动、运行、退出和异常日志。
- `failure.json`：异常退出原因（如发生）。Ctrl+C/通信异常只能记录已收到确认的 step，不声称最后未确认的物理请求没有执行。

HOME 统计包含首次到达、目标出现前是否处于 HOME，以及最长连续静止时间。使用 reward 中原有 HOME 参考、位置和速度容差，并排除失效、过时、重复测量对持续时间的虚增。前 3 秒包含回 HOME 的过程，不能将“到过 HOME”称为训练协议要求的“保持 HOME 三秒成功”。

接近目标统计区分最小距离、最终距离、末段平均距离、XY/高度偏差、朝向、夹爪开口与接触/抓取状态。`manipulation` 或距离小于 2 cm 本身均不等于成功抓取。本测试不提前停在 S1 阈值，完整观察后续行为；物理任务失败或运行故障可以提前终止，报告会明确标记未完成 13 秒。

离线单测只验证时间线、记录、诊断和初始化分支；Gazebo GUI、相机时序和真实模型行为仍需实际运行确认。
