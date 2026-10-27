# 三仓库整合说明

本地分支 `codex/integrate-act-ppo` 以 `so101-manipulation` 为主体，纳入
`czhengxi151-pixel/lerobot` 的 `thesis-act-so101` 分支及 `so101-capstone`。
来源提交见 [integration_sources.json](integration_sources.json)。
本版本含来自私有仓库的源码；公开推送前需由项目所有者确认发布范围。

## 已确认的接口选择

- 保留两套配置，通过统一入口显式选择，不重算或覆盖原标定。
- LeRobot 只保留根目录 `lerobot/` 一份运行源码。
- 保留 Capstone 的动作限位、梯度累积和 PPO 接口，加入 ACT 潜变量监测。
- 对外为六维关节位置状态、六维绝对位置动作及 fixed/handeye 两路 RGB 图像。
- 策略前五轴范围为 [-100,100]，夹爪为 [0,100]；控制器接收弧度。

| 配置 | 工作区 | 控制器 | 控制器更新 | 默认策略执行 |
|---|---|---|---|---|
| primary | ros | forward_position_controller | 200 Hz | 15 Hz |
| capstone_ppo | ros2_ws | position_controller | 100 Hz | 30 Hz |

两套 `so101_description` 属于两个独立构建工作区。不要从仓库根目录递归
`colcon build`，不要在同一终端叠加 source 两套工作区。统一入口会拒绝检测到的冲突叠加。
配置清单在 `configs/runtime_profiles.json`，实际运行细节仍由对应入口处理。

## 统一入口

以下命令只打印计划，不导入 ROS/PyTorch，也不启动训练或硬件：

```bash
python scripts/project.py --profile primary collect
python scripts/project.py --profile primary train -- --run act_integrated
python scripts/project.py --profile primary deploy -- --checkpoint /path/to/pretrained_model
python scripts/project.py --profile capstone_ppo ppo-train
python scripts/project.py --profile capstone_ppo evaluate -- --checkpoint /path/to/pretrained_model
```

`--` 后传递给原有入口；在 `--` 前增加 `--execute` 才会调度实际命令。
ACT 的 train、deploy 等原有入口仍会检查数据与模型。PPO 训练要求先设置
`PRETRAINED_MODEL`；默认历史模型不随仓库提供。完整 S0–S5 启动器是新训练，
并非完整 PPO 断点恢复。评估默认采用辅助抓持，必须与训练条件匹配。

## 安装和 ROS

仿真运行目标环境仍为 Ubuntu 22.04 / ROS Humble / Gazebo Fortress / Python 3.10。
Windows 本地检查只用于离线验证。先按 [Capstone 环境说明](publication/setup_and_usage.md)
准备匹配依赖，再在专用环境安装：

```bash
python -m pip install --no-deps -e ./lerobot
python -m pip install --no-deps -e .
```

以上不安装依赖。依赖快照不是经过新机器验证的完整锁文件。
PPO 通用脚本支持 `PPO_CONDA_SETUP`、`PPO_CONDA_ENV`、`PPO_ROS_SETUP` 覆盖。
primary 的历史设置见 [setup.md](setup.md)；PPO 工作区仅在 `ros2_ws/` 内构建。

原主体 mesh 仍为外部输入，不能仅因 Capstone 含同名 mesh 就宣称两套几何一致。
离线 smoke 默认报告缺失 mesh 数量；`--require-meshes` 会严格检查全部引用。

## TensorBoard 与模型兼容

`LEROBOT_TENSORBOARD_LOGDIR` 优先，兼容原有 `LEROBOT_TENSORBOARD_DIR`；
二者均未设置时，`LEROBOT_TENSORBOARD=1` 写入当前训练输出目录。
潜变量仅在 TensorBoard 日志更新点捕获实际训练 forward 的 z，不增加新模型参数。
默认模型内部仍返回 `(actions, (mu, log_variance))`；显式
`return_latent=True` 才增加 z，避免破坏 PPO 与旧调用者。

梯度累积的训练 step/checkpoint step 继续按 optimizer update 计数。
标量按 microbatch 样本数加权；潜变量直方图来自该 update 的最后一个 microbatch。
旧 `patches/lerobot-local.patch` 及发布补丁仅用于溯源，不能重复应用到整合源码。

## 保留的实验边界

主体的 reach PPO、传统控制、实机 ACT 结果均保留。Capstone 的 residual PPO
与历史配置也保留，当前入口为 `scripts/finetune_act_ppo_s0_s5.sh`。
默认刚性抓持辅助、自动准备阶段起点的成功不能替代自然摩擦或连续完整抓放验证。
两份来源 README 作为历史快照保留在 `capstone_source_README.md` 和
`thesis_act_source_README.md`，其中相对链接按各自原仓库解释。

## 检查

```bash
python scripts/smoke_check.py
python scripts/test_act_workflow.py
python -m pytest tests/unit
```

pytest 需要完整测试依赖；Linux shell、ROS、Gazebo 与实机检查是不同层级。
当前实际执行结果与限制见 [integration_validation.md](integration_validation.md)。
