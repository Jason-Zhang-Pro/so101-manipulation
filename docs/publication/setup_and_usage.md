# 环境准备与运行说明

以下是源码核对后的使用范例。本次整理没有执行安装、训练、Gazebo 或硬件操作，也没有完成新机器安装验证。除特别说明，命令均从克隆根目录运行；`/absolute/path/...` 需替换为实际路径。

## 环境前置

原开发环境为 Ubuntu 22.04、ROS 2 Humble、Gazebo Fortress / ignition-gazebo6、Conda `lerobot` / Python 3.10.18、NVIDIA RTX 5080。构建需要 colcon、ament_cmake 和 C++17 编译器。

ROS 运行需要 `ros_gz_sim`、`ros_gz_bridge`、`ros_gz_interfaces`、xacro、`robot_state_publisher`、`controller_manager` 及相应位置控制器。机器人 Xacro 使用 `gz_ros2_control/GazeboSimSystem` 和 `gz_ros2_control-system`，需安装与 Humble/Fortress 相匹配的插件。

Gazebo CMake 还要求 `ignition-gazebo6`、`ignition-msgs8`、`ignition-plugin1`、`ignition-rendering6`、`ignition-transport11` 开发组件。现有 `package.xml` 没有完整声明这些构建依赖，不能把一次 `rosdep install` 视为已经安装全部依赖。无 GUI 训练仍需要双相机渲染所需的图形环境。

整理时从原 Conda 环境读取的版本如下。这是**已安装包元数据快照，不是 lockfile，也不是完整依赖列表或新环境重装验证结果**：

| 包 | 版本 |
|---|---|
| LeRobot | 0.3.4，本地定制源码 |
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| torchcodec / numpy | 0.5 / 2.2.6 |
| tensorboard / pytest | 2.21.0 / 8.4.2 |
| gymnasium / safetensors | 0.29.1 / 0.8.0 |
| opencv-python-headless | 5.0.0.93 |
| datasets / draccus | 3.6.0 / 0.10.0 |
| av / matplotlib | 17.1.0 / 3.10.9 |
| feetech-servo-sdk / Pillow | 1.0.0 / 12.3.0 |
| diffusers / huggingface-hub | 0.39.0 / 1.25.1 |

声明依赖见 [`lerobot/pyproject.toml`](../../lerobot/pyproject.toml)。安装时需核对 CUDA、视频解码与 ROS Python 兼容性，不要用自动升级原工作环境代替环境恢复。

整合版本已补齐根目录 `pyproject.toml`，可安装项目包。在依赖已经准备好的专用环境里，注册本地 LeRobot 源码及项目：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
python -m pip install --no-deps -e ./lerobot
python -m pip install --no-deps -e .
```

`--no-deps` 不安装依赖，上述不是完整环境安装命令。当前本地定制源码已经包含，无需重复应用旧补丁。Capstone 通过以下方式导入：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
```

## ROS 构建

在匹配的系统依赖准备完毕后，使用正确的系统 Python/ROS 构建环境：

```bash
source /opt/ros/humble/setup.bash
(
  cd ros2_ws
  colcon build --packages-select so101_description so101_gazebo so101_bringup
)
source "$PWD/ros2_ws/install/setup.bash"
```

发布不含 `build/`、`install/`、`log/`；必须本机重新构建。修改模型、插件、launch 或安装资源后也需重建。上述命令未在新机器执行。

## 外部数据与模型

监督训练需要符合本项目双相机与六维动作约定的完整 LeRobot 数据集，包括 metadata、数据表和对应视频。历史 ID `tiger/so101_pick_test` 只是原项目标识，不承诺该数据集可从公共 Hub 下载。

ACT 和 PPO 模型均不发布。当前加载器与阶段评估使用实际的：

```text
pretrained_model/
├── config.json
└── model.safetensors
```

应保留模型真实配置与归一化状态。完整 PPO 恢复还需 critic、optimizer、训练状态和兼容的执行签名；仅有 ACT 权重是新训练的起点。

独立取出的模型可能无法找到原 run 的 `resolved_stages.json`、`resolved_reward.json` 或辅助抓持配置。评估时可显式指定当前配置，不能把这种测试说成恢复了原实验。

## ACT 监督训练

原 `scripts/train_act_so101.sh` 固定了原机器的 LeRobot、输出、TensorBoard 转换程序路径，以及 `~/miniconda3` 和环境名。数据预检查固定到 `~/.cache/huggingface/lerobot/tiger/so101_pick_test`。脚本内路径不能通过同名外部 `export` 覆盖。

在其他目录可使用下面的等效核心训练命令，保留原脚本当前超参并显式指定数据路径：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

python -m lerobot.scripts.train \
  --dataset.repo_id="tiger/so101_pick_test" \
  --dataset.root="/absolute/path/to/local_dataset" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}}}' \
  --policy.type=act \
  --output_dir="$PWD/outputs/train/my_act_run" \
  --job_name=my_act_run \
  --policy.device=cuda \
  --batch_size=30 \
  --num_workers=16 \
  --steps=1000000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --policy.chunk_size=15 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.dropout=0.1 \
  --policy.kl_weight=10.0 \
  --policy.optimizer_lr=1e-5 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

每次使用新的输出目录。这条命令会开始长时间训练，不含原 shell 的 CUDA 运算预检、参数快照、自动 TensorBoard 转换、浏览器及进程清理包装。图像增强为亮度和对比度变化；TE 是推理时融合，不改变监督训练损失。

## 六阶段直接 ACT-PPO

完整入口 `scripts/finetune_act_ppo_s0_s5.sh` 自动确定根目录，调用通用 `scripts/finetune_act_ppo_so101.sh`。底层仍固定加载 `~/miniconda3/etc/profile.d/conda.sh`、环境 `lerobot` 和 `/opt/ros/humble/setup.bash`；本机布局不同时，需要使用者在自己的副本适配环境加载位置。

先查看配置：

```bash
PRETRAINED_MODEL="/absolute/path/to/pretrained_model" \
WORLD_COUNT=4 \
GAZEBO_GUI=false \
STAGE_STEP_WEIGHTS="1:3:2:2:1:1" \
bash scripts/finetune_act_ppo_s0_s5.sh --print-config
```

此模式不启动 Conda、ROS、Gazebo 或训练，但会检查模型文件真实存在。开始训练时使用同样的命令，去掉 `--print-config`。需要画面时设 `GAZEBO_GUI=true`。

完整脚本仅允许环境覆盖 `PRETRAINED_MODEL`、`WORLD_COUNT`、`GAZEBO_GUI`、`EPISODE_COUNT`、`STAGE_STEP_WEIGHTS`。其他参数固定在脚本配置表中；不能假定外部 `export TARGET_KL=...` 或 `MINIBATCH_SIZE=...` 可以覆盖。

| 当前完整脚本设置 | 值 |
|---|---|
| 世界数 / GUI | 4 / 关闭 |
| S0:S1:S2:S3:S4:S5 有效步数权重 | 1:3:2:2:1:1 |
| 阶段预算 | S0/S5 5 秒、S1 10 秒、S2–S4 各 8 秒 |
| rollout / minibatch / PPO epochs | 2400 / 30 / 2 |
| target KL | 50，启用回退 |
| actor / backbone LR | 3.125e-6 / 3.125e-7 |
| critic / exploration LR | 2e-4 / 3.125e-6 |
| checkpoint / TensorBoard | 每 5 次 update 保存 / 每次 update 记录 |
| 图像增强 / 物理随机化 | 开启 / 关闭 |
| 仿真刚性抓持辅助 | 开启 |

比例为成功阶段过滤前的有效步数目标。完整 episode 和容差可能使实际步数超过 rollout。阶段本轮成功率达 80% 时，过滤其本轮更新样本，下一轮跳过该阶段采样一次，由其余阶段补足步数。共享 ACT 不被冻结，其他阶段更新仍可能影响已达标阶段。

默认历史 good40 模型不随源码提供，必须显式覆盖。完整脚本清除 `RESUME_CHECKPOINT`，只加载指定 ACT 权重，新建 critic、optimizer、探索状态和输出目录，从 update 1 开始，**不是完整 PPO 断点恢复**。完整恢复需通过通用入口配置并满足执行签名兼容要求。

`ACT_MAX_ACTION_STEPS=1` 不等同于关闭 TE，以实际 checkpoint 和启动解析结果为准。上述监督训练模型保存 TE=0.01，不能假设它等同于历史关闭 TE 的 PPO 起点。

启动器默认预清理其 ROS domain/partition 的遗留进程（domain 从 20 起），不要与其他任务共用这些资源。世界结束后记录结果并关闭，需要后续采样时再创建；产物写入新 `output_simulation/finetuned/...`。

详见[六阶段训练与准备文档](../act_ppo_s0_s5_training.md)。

## 分别评估 S0–S5

模型、依赖和 ROS 构建就绪后，先打印计划：

```bash
bash scripts/evaluate_act_stages.sh \
  --checkpoint "/absolute/path/to/pretrained_model" \
  --stage-config "$PWD/config/act_ppo_stages_s0_s5.json" \
  --reward-config "$PWD/config/pick_place_reward.json" \
  --assisted-grasp true
```

追加 `--execute` 才启动实际 GUI 评估；可追加 `--stages S2 S3`、`--episodes-per-stage 3`、`--headless`。默认计划不创建结果目录、不启动 Gazebo，但 shell 仍会先加载 Conda、ROS 及 `ros2_ws/install/setup.bash`。评估脚本支持 `EVAL_CONDA_SETUP`、`EVAL_CONDA_ENV`、`EVAL_ROS_SETUP` 覆盖。

本例使用当前配置。若希望自动读取完整六阶段模型原 run 的配置，可省略 `--stage-config` 和 `--reward-config`，并核对打印计划中的配置来源。当前 JSON 和未覆盖的通用入口中 S2–S4 各为 10 秒；要显式复现完整训练脚本的 8 秒预算，可增加：

```text
--s2-budget-s 8 --s3-budget-s 8 --s4-budget-s 8
```

S5 跟随 S0 预算。评估不做 PPO、不加探索噪声、不开训练图像增强，按 checkpoint 配置执行 ACT。每例使用新世界，准备失败和策略失败单独记录。默认固定目标 `(0.26, 0, 0.768)`；可用 `--target-position X Y Z` 改变。固定目标重复测试不代表随机泛化成功率。

默认 domain 125，资源冲突拒绝启动；结果写入新 `output_simulation/evaluation/stages_.../`。辅助抓持成功不能代表自然摩擦防滑已经学会，各阶段独立成功也不能代表连续任务成功。

## 自动准备起点 GUI

此工具展示自动控制器准备起点，不是 ACT policy 测试；同一世界依次展示各阶段：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
source /opt/ros/humble/setup.bash
source "$PWD/ros2_ws/install/setup.bash"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

# 仅计划
python codex_workspace/scripts/show_stage_preparation_gui.py

# 实际 GUI
python codex_workspace/scripts/show_stage_preparation_gui.py --execute
```

默认独立 domain 126，启用辅助抓持，现场记录进入 `codex_workspace/reports/`。自动准备和正式 S2/S3 采样的连接释放规则不同，详见六阶段文档。

## 离线测试与验证层级

核对解释器和依赖后，项目单元测试入口为：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$PWD/src:$PWD/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/unit
```

此处是命令范例，本次发布整理没有运行全套测试。部分历史路径或资源相关测试在新克隆上仍需实际核对。LeRobot 二进制 fixtures 被排除，完整上游测试可能需另行获取对应资源。

必须区分静态检查、离线单元测试、Gazebo live smoke 和真实硬件验证。离线逻辑通过不证明真实碰撞、抓持和物理准备成功；历史通过记录也不证明新机器已通过。真实硬件需要独立设备、标定和操作准备。
