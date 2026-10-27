# SO-101 Capstone：ACT 与 Gazebo 分阶段强化学习

本项目围绕 SO-101 Leader / Follower、顶部与夹爪双相机，提供遥操作数据采集相关代码、ACT 监督训练、Gazebo 推理评估，以及直接更新 ACT 的 PPO 训练实现。当前主要开发入口是 **S0–S5 分阶段采样、同一 ACT policy 联合更新**。

这是从本地项目整理的源码发布副本。**不包含采集数据、训练输出、模型权重、checkpoint、运行日志、ROS 构建产物或本地缓存**。运行训练或评估前，需要自行准备数据或模型、安装依赖并构建 ROS 工作区。整理发布副本没有修改原工作项目。

## 当前实现

- 30 Hz 确定性环境步进，双相机与关节观测、动作限位、reset 和反馈检查。
- ACT 监督训练、chunk 动作执行及 temporal ensembling 支持。
- 直接 ACT-PPO：多世界采样、KL 检查与回退、checkpoint、TensorBoard、作废 step 记录。
- S0–S5 独立准备起点、按有效 step 权重分配采样、阶段成功率达标后的更新过滤与下一轮采样跳过。
- S0–S5 分别评估 policy，以及单世界自动准备起点目视检查。
- 独立环境随机化模块；当前六阶段完整启动器关闭物理环境随机化，开启训练图像增强。

旧 Residual PPO 和早期单世界入口仍作为历史代码保留，不能据此替代当前六阶段启动入口。

## 六个阶段

| 阶段 | 自动准备的起点 | 训练目标概述 |
|---|---|---|
| S0 | 无目标，HOME 姿势 | 六关节含夹爪的 HOME 误差各自 `<8°`，连续保持至少 1 秒；不检查速度 |
| S1 | HOME，源区目标已生成 | 抓取参考点抬到桌面上方 8 cm 并保持 0.1 秒，再接近至距目标中心 `<2 cm` |
| S2 | 夹爪在目标附近，尚未完成抓持 | 闭夹、确认抓持并稳定抬起 |
| S3 | 已抓持并抬起 | 持物搬运至放置区 |
| S4 | 持物位于放置区上方 | 降低、释放、撤开夹爪并确认物体稳定放置 |
| S5 | 物体已放稳，夹爪已撤开 | 放置持续有效，同时满足与 S0 共享的 HOME 成功判定 |

准确的证据条件、奖励分项、保持时间及异常处理以源码和生效配置为准，参见[六阶段训练与准备文档](docs/act_ppo_s0_s5_training.md)。完整启动器的阶段预算为 S0/S5 5 秒、S1 10 秒、S2–S4 各 8 秒，准备时间不计入阶段预算或 PPO 样本。

完整六阶段启动器默认使用**仿真刚性抓持辅助**：真实有效抓持确认后连接目标与固定夹指；S2/S3 采样中保持连接，S4 使用张开指令和实测角度决定释放。这种辅助任务的成功不等于已学会自然摩擦防滑。S0 从 HOME 开始，主要检验保持能力；各阶段独立成功也不等于连续完成整个抓放任务。

## 主要入口

| 用途 | 入口 |
|---|---|
| ACT 监督训练 | `scripts/train_act_so101.sh`；含原机器固定路径，跨目录使用下述安装文档中的等效 CLI |
| 当前完整 S0–S5 PPO | `scripts/finetune_act_ppo_s0_s5.sh` |
| 通用直接 ACT-PPO | `scripts/finetune_act_ppo_so101.sh` |
| 分别测试 S0–S5 | `scripts/evaluate_act_stages.sh` |
| 自主回 HOME 后出现目标的连续评估 | `scripts/evaluate_act_home_reach.sh` |
| S0–S5 自动准备起点 GUI | `codex_workspace/scripts/show_stage_preparation_gui.py` |

所有使用方法和前置条件见 **[环境准备与运行说明](docs/publication/setup_and_usage.md)**。不要直接运行默认六阶段命令：默认引用的历史 good40 模型不包含在发布内容中，必须显式设置 `PRETRAINED_MODEL`。

例如，在依赖和外部模型准备完成后，从克隆根目录仅查看配置：

```bash
PRETRAINED_MODEL="/absolute/path/to/pretrained_model" \
STAGE_STEP_WEIGHTS="1:3:2:2:1:1" \
bash scripts/finetune_act_ppo_s0_s5.sh --print-config
```

`--print-config` 不启动训练或 Gazebo，但仍要求模型配置和权重真实存在。去掉该选项会启动训练。

## 目录

```text
src/so101_capstone/    环境、policy、奖励、PPO、评估及运动学
ros2_ws/src/          ROS 2 packages、模型、场景、C++ 插件和 launch
lerobot/             本地定制 LeRobot 源码快照，已随项目保存
config/              当前及历史实验配置
scripts/             训练、评估和辅助入口
tests/               Capstone 离线测试及其他验证入口
docs/                项目文档；publication/ 为此次发布整理说明
codex_workspace/     选取的辅助脚本、启动命令和补丁说明
reportcontent/       实验图表/场景的生成脚本，产物与数据另行准备
tools/               模型与碰撞等工具
hardware/            硬件使用说明及非私密配置
```

`lerobot/` 的本地修改直接包含在源码中，不是指向其他仓库的子模块，也不需要重复应用旧补丁。上游许可证保留；来源、基线和变更见 [LeRobot 源码说明](docs/publication/lerobot_vendor.md)。

## 环境与验证边界

开发环境为 Ubuntu 22.04、ROS 2 Humble、Gazebo Fortress、Python 3.10 的 Conda `lerobot` 环境及 CUDA GPU。原机器使用 RTX 5080。源码发布不意味着在新机器完成了安装、构建或训练验证；当前安装版本记录也不是依赖锁文件。

离线单元测试、Gazebo live smoke、真实硬件验证是不同层级。文档中的历史通过记录不能证明新克隆已通过，也不能代替当前环境验证。真实机械臂还需要自己的设备地址、标定与操作准备。

原项目较早的 README 存在“训练代码仍是骨架”等过时描述，已保留为[原 README 快照](docs/publication/README_original_snapshot.md)。其他历史文档中的绝对路径、参数和实验结论需要结合当前源码核对；优先参考本页、发布说明和当前六阶段文档。

本源码目录没有 `.git`。原主仓库的 93 个提交和 4 个标签另存于准备包的 `local_history/capstone.bundle`，用于本地历史恢复；旧历史含运行报告等内容，因此没有混入这份纯源码目录。
