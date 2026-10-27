# SO-101 Capstone Project

## End-to-End Robotic Arm Manipulation via Reinforcement Learning and Vision-Language-Action Models

本仓库是 SO-101 Capstone 项目的正式工作目录，目标是建立一套基于双机械臂、双相机、Gazebo 仿真和机器学习方法的端到端机器人操作系统。

项目主要研究方向包括：

* SO-101 Leader / Follower 双机械臂控制
* Top camera 与 Gripper camera 双视角视觉输入
* 真实机器人与 Gazebo 正式仿真环境
* Imitation Learning
* Reinforcement Learning
* Vision-Language-Action（VLA）模型
* Sim-to-Real transfer

当前阶段的主要工作是完善和记录正式 Gazebo 抓取任务环境，并在此基础上继续开发数据采集、训练、评估和真实机器人部署流程。

---

## 1. 当前项目状态

### 已实现并验证

以下功能已经实现，并经过命令检查或人工确认：

* SO-101 Leader 和 Follower 均可正常连接
* 两台机械臂均已完成 calibration
* Leader 可以实时控制 Follower
* 顶部相机和夹爪相机可以同时稳定运行
* SO-101 模型可以在 Gazebo Fortress 中正常生成
* `ros2_control` 可以正常启动
* `joint_state_broadcaster` 可以发布关节状态
* `position_controller` 可以控制六个关节
* 顶部 RGB 相机可发布 ROS 2 图像与 CameraInfo
* 夹爪 RGB 相机可发布 ROS 2 图像与 CameraInfo
* 实验桌、机械臂位置、灯光和材质已经完成视觉检查
* 动态目标方块可以随机生成
* 可以实时获取方块名称、Gazebo entity ID、位置和姿态
* 可以按名称删除单个方块
* 可以一次删除全部方块
* Target cube manager 已集成到正式仿真 launch file

### 当前仍属于项目骨架的部分

以下目录目前主要用于定义未来项目结构，尚不代表相关功能已经完整实现：

```text
configs/
scripts/
src/so101_capstone/
tests/
tools/
```

训练、数据处理、policy、evaluation 和 experiment 入口必须在实际实现并验证后，才能被视为正式可用功能。

---

## 2. 系统环境

当前已经验证的开发环境：

| Component               | Version / Description |
| ----------------------- | --------------------- |
| Operating System        | Ubuntu 22.04          |
| ROS                     | ROS 2 Humble          |
| Simulator               | Gazebo Fortress       |
| Main scripting language | Python                |
| Robot framework         | LeRobot               |
| ROS domain              | `ROS_DOMAIN_ID=20`    |
| Gazebo partition        | `gz_env_20`           |

正式项目根目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone
```

正式 ROS 2 workspace：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws
```

---

## 3. Gazebo 仿真快速启动

打开一个新终端：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20

ros2 launch so101_bringup sim.launch.py
```

正式启动文件：

```text
ros2_ws/src/so101_bringup/launch/sim.launch.py
```

该 launch file 会启动：

* Gazebo Fortress
* 正式 simulation world
* SO-101 robot model
* `robot_state_publisher`
* Gazebo-to-ROS camera bridge
* Target cube manager
* `joint_state_broadcaster`
* `position_controller`

---

## 4. 机械臂生成位置

SO-101 在 Gazebo 中的正式生成坐标为：

```text
x = 0.0
y = 0.0
z = 0.7524
```

其中 `z=0.7524` 用于补偿 base mesh 原点约 `-0.0024 m` 的垂直偏移，使机械臂底座在视觉和碰撞上正确落在高度为 `0.750 m` 的桌面上。

除非重新验证完整模型与桌面关系，否则不要修改该高度。

---

## 5. 仿真启动后的基本验证

### 5.1 检查相机 topics

```bash
ros2 topic list | grep -E 'top_camera|gripper_camera'
```

当前正式 RGB camera interfaces：

```text
/top_camera/image_raw
/top_camera/camera_info
/gripper_camera/image_raw
/gripper_camera/camera_info
```

检查顶部相机频率：

```bash
ros2 topic hz /top_camera/image_raw
```

检查夹爪相机频率：

```bash
ros2 topic hz /gripper_camera/image_raw
```

### 5.2 检查 controllers

```bash
ros2 control list_controllers
```

预期启动的 controllers：

```text
joint_state_broadcaster
position_controller
```

### 5.3 检查关节状态

```bash
ros2 topic echo /joint_states --once
```

---

## 6. 动态目标方块

Target cube manager 会随正式 Gazebo launch 自动启动，不需要单独运行 Python 节点。

### 6.1 生成一个随机方块

```bash
ros2 service call \
  /target_cube/spawn_random \
  std_srvs/srv/Trigger \
  "{}"
```

### 6.2 查看所有方块状态

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

### 6.3 查看单个方块实时位姿

```bash
ros2 topic echo \
  /target_cube/pose/target_cube_0001 \
  --once
```

### 6.4 按名称删除方块

```bash
ros2 topic pub \
  --once \
  /target_cube/delete_name \
  std_msgs/msg/String \
  "{data: 'target_cube_0001'}"
```

### 6.5 查看删除结果

```bash
ros2 topic echo \
  /target_cube/delete_result \
  --once \
  --full-length
```

### 6.6 删除全部方块

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

### 6.7 当前方块参数

| Parameter               |                     Value |
| ----------------------- | ------------------------: |
| Cube size               | `0.030 × 0.030 × 0.030 m` |
| Mass                    |                `0.020 kg` |
| Table surface height    |                 `0.750 m` |
| Spawn centre height     |                 `0.768 m` |
| Inner spawn radius      |                  `0.10 m` |
| Outer spawn radius      |                  `0.40 m` |
| Spawn angle             |          `-90°` to `+90°` |
| Minimum centre distance |                  `0.05 m` |
| Maximum managed count   |                      `20` |
| State publication rate  |                   `10 Hz` |

随机位置使用以机械臂底座世界坐标 `(0, 0)` 为中心的前半圆环区域，并按面积均匀采样。

---

## 7. 真实机器人硬件

### 7.1 SO-101 机械臂

| Device   | Serial number | Persistent path                                                |
| -------- | ------------- | -------------------------------------------------------------- |
| Follower | `5C4C124039`  | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00` |
| Leader   | `5C82107443`  | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107443-if00` |

### 7.2 相机

| Camera                    | Persistent path                                                         |
| ------------------------- | ----------------------------------------------------------------------- |
| Gripper / Hand-eye camera | `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0`              |
| Top / Fixed camera        | `/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0` |

当前常用设置：

```text
Resolution: 640 × 480
Frame rate: 15 FPS 或 30 FPS
```

### 7.3 USB 连接要求

双相机曾经因为 USB 扩展坞带宽不足而无法稳定同时运行。

当前已验证方案：

* 两个相机直接连接主板 USB 接口
* 不通过带宽不足的 USB hub
* 使用 `/dev/v4l/by-id/` 固定路径
* 不依赖可能变化的 `/dev/video0`、`/dev/video2`

机械臂串口同样应使用：

```text
/dev/serial/by-id/
```

不要依赖可能变化的：

```text
/dev/ttyUSB0
/dev/ttyUSB1
```

本机的原始设备检查记录位于：

```text
hardware/device_ids/IDlist.txt
```

该文件包含 machine-specific information，因此已被 `.gitignore` 排除。

---

## 8. 项目目录结构

```text
capstone/
├── configs/
│   ├── camera/
│   ├── experiments/
│   ├── reward/
│   ├── robot/
│   ├── task/
│   └── training/
├── data/
│   ├── metadata/
│   ├── processed/
│   ├── raw/
│   │   ├── real/
│   │   └── simulation/
│   └── samples/
├── docs/
│   ├── migration/
│   │   ├── backups/
│   │   ├── old_cap_robot_assets.txt
│   │   └── so101_urdf_source.txt
│   ├── gazebo_simulation.md
│   ├── project_structure.md
│   ├── real_robot_setup.md
│   ├── target_cube_manager.md
│   └── troubleshooting.md
├── hardware/
│   ├── device_ids/
│   │   └── IDlist.txt
│   └── README.md
├── huggingface/
├── lerobot/
├── notebooks/
│   └── analysis/
├── outputs/
│   ├── checkpoints/
│   ├── evaluations/
│   ├── logs/
│   ├── runs/
│   ├── tensorboard/
│   └── videos/
├── references/
│   └── SO-ARM101_original/
├── ros2_ws/
│   ├── README.md
│   └── src/
│       ├── so101_bringup/
│       │   └── README.md
│       ├── so101_description/
│       │   └── README.md
│       └── so101_gazebo/
│           └── README.md
├── scripts/
│   ├── data/
│   ├── evaluation/
│   ├── setup/
│   ├── simulation/
│   └── training/
├── src/
│   └── so101_capstone/
│       ├── common/
│       ├── control/
│       ├── datasets/
│       ├── environments/
│       ├── evaluation/
│       ├── perception/
│       ├── policies/
│       ├── rewards/
│       ├── robot/
│       ├── simulation/
│       ├── tasks/
│       └── __init__.py
├── tests/
│   ├── hardware/
│   ├── integration/
│   ├── simulation/
│   └── unit/
├── tools/
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 9. 主要目录职责

### `ros2_ws/`

正式 ROS 2 / Gazebo workspace。

当前已经实现并验证的主要运行功能都位于此目录。

### `lerobot/`

正式 LeRobot source tree 和运行目录，用于：

* 真实机械臂控制
* Leader-Follower teleoperation
* 相机读取
* 未来 imitation learning
* 数据采集
* Policy training 与 evaluation

该目录属于外部源码树，因此被主 Capstone repository 的 `.gitignore` 排除。

### `huggingface/`

Hugging Face cache 目录。

该目录不是项目源代码，不应提交到 Git。

### `references/SO-ARM101_original/`

SO-ARM101 官方教程与原始参考资料的只读归档。

其中包括：

* 官方教程
* 原始源码与工具
* 技术参数
* 机械图纸
* 3D model files
* 官方 distribution files

该目录应保持 read-only，不应作为正式运行目录，也不应直接在其中开发 Capstone 功能。

### `docs/migration/`

记录模型、URDF、mesh 和旧项目资源迁移过程。

该目录中的路径可能是历史路径，只用于说明 provenance，不代表当前正式项目路径。

### `configs/`

计划用于项目级配置，例如：

* Camera configuration
* Robot configuration
* Reward configuration
* Task configuration
* Training configuration
* Experiment configuration

当前主要为目录骨架。

### `scripts/`

计划用于可执行入口，例如：

* Simulation
* Data collection
* Training
* Evaluation
* Environment setup

当前主要为目录骨架。

### `src/so101_capstone/`

计划用于项目自己的 Python package，包括：

* Robot interfaces
* Perception
* Control
* Simulation
* Environments
* Tasks
* Rewards
* Policies
* Datasets
* Evaluation

当前尚未实现完整功能。

### `tests/`

计划用于：

* Unit tests
* Integration tests
* Simulation tests
* Hardware tests

当前测试结构尚未填充。

### `outputs/`

用于保存：

* Checkpoints
* Evaluation results
* Logs
* Training runs
* TensorBoard data
* Videos

该目录被 Git 忽略。

---

## 10. ROS 2 packages

### 10.1 `so101_description`

职责：

* SO-101 Xacro model
* Robot links 和 joints
* Mesh assets
* Camera links 和 sensors
* Joint limits
* `ros2_control` interfaces
* Gazebo visual materials

正式 Xacro：

```text
ros2_ws/src/so101_description/urdf/so101.urdf.xacro
```

安装后的副本：

```text
ros2_ws/install/so101_description/share/so101_description/urdf/so101.urdf.xacro
```

### 10.2 `so101_gazebo`

职责：

* Gazebo world
* Table model
* Target cube model
* Controller configuration
* Target cube manager configuration
* Dynamic target cube manager node

主要文件：

```text
ros2_ws/src/so101_gazebo/worlds/empty.sdf
ros2_ws/src/so101_gazebo/config/controllers.yaml
ros2_ws/src/so101_gazebo/config/target_cube.yaml
ros2_ws/src/so101_gazebo/scripts/target_cube_manager.py
```

### 10.3 `so101_bringup`

职责：

* 启动完整仿真系统
* 生成 robot model
* 启动 Gazebo
* 配置 resource path 和 plugin path
* 启动 ROS-Gazebo bridges
* 启动 controllers
* 启动 target cube manager

正式启动文件：

```text
ros2_ws/src/so101_bringup/launch/sim.launch.py
```

---

## 11. 相机配置

### Top camera

| Parameter       |          Value |
| --------------- | -------------: |
| Resolution      |    `640 × 480` |
| Update rate     |        `30 Hz` |
| Horizontal FOV  | `2.792527 rad` |
| Approximate FOV |         `160°` |

ROS 2 topics：

```text
/top_camera/image_raw
/top_camera/camera_info
```

### Gripper camera

| Parameter       |          Value |
| --------------- | -------------: |
| Resolution      |    `640 × 480` |
| Update rate     |        `30 Hz` |
| Horizontal FOV  | `0.872665 rad` |
| Approximate FOV |          `50°` |

ROS 2 topics：

```text
/gripper_camera/image_raw
/gripper_camera/camera_info
```

Gripper camera sensor pose：

```xml
<pose>0.007 -0.0015 0.0001 0 -1.570796 1.570796</pose>
```

Xacro 中还定义了 segmentation sensors，但当前正式 `sim.launch.py` 只 bridge RGB image 和 CameraInfo topics。

因此，在 segmentation bridge 被正式加入并验证之前，不应把 segmentation ROS 2 topics 记录为已启用功能。

---

## 12. 世界与实验桌

正式 world file：

```text
ros2_ws/src/so101_gazebo/worlds/empty.sdf
```

实验桌模型：

```text
ros2_ws/src/so101_gazebo/models/lab_table/
```

桌面尺寸：

```text
Length:    1.200 m
Width:     0.600 m
Thickness: 0.034 m
Top Z:     0.750 m
```

桌子在世界中的 pose：

```xml
<pose>0.278 0.05 0 0 0 1.570796</pose>
```

世界坐标约定：

```text
+X = 向桌面中心方向
+Y = 机械臂左侧
+Z = 向上
```

---

## 13. 灯光设计

当前场景使用一个 directional light：

```xml
<light type="directional" name="main_overhead_light">
```

以前曾测试 point light，但在 ROS camera image 中产生明显的十字形或球状渲染伪影，因此已经删除。

除非有明确的视觉验证需求，否则不要重新加入 point light。

---

## 14. 材质设计说明

机械臂当前正式外观：

* 3D printed structure：白色 PLA
* STS3215 servo：深黑色塑料
* Top camera：黑色塑料
* Gripper camera：黑色塑料
* Gripper camera mount：白色 PLA

Gazebo Fortress 对 mesh visual 的全局 URDF material reference 存在显示问题。

只使用：

```xml
<material name="3d_printed"/>
```

可能导致白色 mesh 回退为黑色。

因此，Xacro 中部分 visual 使用 inline material：

```xml
<material name="3d_printed">
  <color rgba="0.88 0.88 0.85 1.0"/>
</material>
```

这些 inline materials 是有意保留的，不应删除。

---

## 15. Xacro 修改要求

不要使用以下方式重写整份 Xacro：

```python
xml.etree.ElementTree.ElementTree.write()
```

该方法以前导致：

```xml
xmlns:xacro="http://www.ros.org/wiki/xacro"
```

被改为：

```xml
xmlns:ns0="http://www.ros.org/wiki/xacro"
```

同时破坏：

* Comments
* Indentation
* Namespace readability
* Git diff quality

对 Xacro 进行小范围修改时，应优先使用 precise text replacement，并在修改后检查：

```bash
git diff --check
```

---

## 16. 不应随意修改的已验证配置

以下配置已经经过验证，不应在没有明确原因和重新测试的情况下修改：

* Joint definitions
* Joint origins
* Joint limits
* Mesh paths
* `ros2_control`
* `controllers.yaml`
* Camera poses
* Camera FOV
* Camera resolution
* Camera topics
* Robot spawn height
* Table pose
* World lighting
* Target cube spawn region
* Target cube deletion logic
* Entity ID tracking logic

---

## 17. Git 忽略内容

当前 `.gitignore` 排除：

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
lerobot/
huggingface/
references/
outputs/
data/raw/
data/processed/
__pycache__/
*.pyc
*.pyo
*.swp
*.save
hardware/device_ids/IDlist.txt
docs/migration/backups/
```

不要随意 `git add -f` 以下内容：

* Build products
* Runtime logs
* Cache files
* Raw datasets
* Model checkpoints
* External repositories
* Machine-specific device scans
* Temporary backup files

---

## 18. Git 工作流程

修改前：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git branch --show-current
```

修改后：

```bash
git diff
git diff --check
git status --short
```

建议每次：

1. 只修改一个明确的功能或文档
2. 检查完整 Git diff
3. 运行相关验证命令
4. 确认没有临时文件或缓存
5. 用户确认后再 commit

不要把不相关的代码、仿真、硬件和文档修改合并到同一个 commit。

---

## 19. 文档索引

以下文档已经建立并完成初步检查：

### 项目与子系统

* [项目结构](docs/project_structure.md)
* [真实 SO-101 系统设置](docs/real_robot_setup.md)
* [Gazebo 正式仿真系统](docs/gazebo_simulation.md)
* [Target Cube Manager](docs/target_cube_manager.md)
* [故障排查](docs/troubleshooting.md)
* [真实硬件映射](hardware/README.md)
* [ROS 2 / Gazebo Workspace](ros2_ws/README.md)

### ROS 2 packages

* [`so101_description`](ros2_ws/src/so101_description/README.md)
* [`so101_gazebo`](ros2_ws/src/so101_gazebo/README.md)
* [`so101_bringup`](ros2_ws/src/so101_bringup/README.md)

历史迁移来源与 provenance 记录位于：

```text
docs/migration/
```

新增功能完成并验证后，应同步更新对应技术文档和本索引。

---

## 20. 当前开发优先级

1. 完善项目 README 和技术文档
2. 保持当前 Gazebo baseline 稳定
3. 完善正式抓取任务环境
4. 定义稳定的数据采集接口
5. 建立 imitation learning workflow
6. 开发 reinforcement learning environment
7. 研究 VLA policy integration
8. 在真实 SO-101 平台上评估 sim-to-real transfer

尚未实现或验证的训练与 policy 功能，不应记录为已经可以运行。
