# Gazebo 正式仿真系统

本文档说明 Capstone 项目当前正式 SO-101 Gazebo Fortress 仿真环境的整体结构、启动流程、坐标关系、ROS 2 接口、验证方法和已知限制。

项目根目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone
```

ROS 2 workspace：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws
```

正式启动命令：

```bash
ros2 launch so101_bringup sim.launch.py
```

本文档描述的是当前已经建立并验证的 Gazebo baseline。除非有明确需求并完成重新验证，否则不要随意修改 robot model、controller、camera、world、table、lighting 或 target cube manager。

---

## 1. 系统目标

当前 Gazebo 环境用于：

- 模拟 SO-101 Follower arm
- 模拟顶部固定相机
- 模拟夹爪相机
- 提供六关节 position control
- 发布 joint states
- 提供桌面抓取环境
- 随机生成动态目标方块
- 实时跟踪方块名称、entity ID 和 pose
- 删除单个或全部方块
- 为后续 imitation learning、reinforcement learning、VLA 和 sim-to-real 提供正式仿真基础

当前阶段主要目标是保持环境稳定、接口清晰、结构可复现。

---

## 2. 已验证环境

| Component | Environment |
|---|---|
| Operating system | Ubuntu 22.04 |
| ROS | ROS 2 Humble |
| Simulator | Gazebo Fortress |
| Render engine | Ogre 2 |
| Control | `ros2_control` |
| Main custom node language | Python |
| ROS domain | `ROS_DOMAIN_ID=20` |
| Gazebo partition | `gz_env_20` |

当前正式环境变量：

```bash
export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

所有参与同一仿真的终端应使用相同的 ROS domain 和 Gazebo partition。

---

## 3. ROS 2 package 架构

当前 workspace 包含三个正式 packages：

```text
ros2_ws/src/
├── so101_description/
├── so101_gazebo/
└── so101_bringup/
```

### `so101_description`

负责：

- SO-101 Xacro
- Robot links 和 joints
- Mesh assets
- Joint limits
- Top camera 和 gripper camera
- RGB 与 segmentation sensors
- Gazebo visual materials
- `ros2_control` interfaces

### `so101_gazebo`

负责：

- Gazebo world
- Laboratory table
- Target cube model
- Controller YAML
- Target cube configuration
- Target cube manager Python node

### `so101_bringup`

负责：

- 读取和组合前两个 packages
- 启动 Gazebo
- 生成 SO-101
- 启动 bridge
- 启动 target cube manager
- 启动 controllers

依赖关系：

```text
so101_description
        │
        ▼
so101_gazebo
        │
        ▼
so101_bringup
```

---

## 4. 正式文件位置

### Robot model

```text
ros2_ws/src/so101_description/urdf/so101.urdf.xacro
```

### Controllers

```text
ros2_ws/src/so101_gazebo/config/controllers.yaml
```

### Target cube configuration

```text
ros2_ws/src/so101_gazebo/config/target_cube.yaml
```

### World

```text
ros2_ws/src/so101_gazebo/worlds/empty.sdf
```

### Table model

```text
ros2_ws/src/so101_gazebo/models/lab_table/
```

### Target cube model

```text
ros2_ws/src/so101_gazebo/models/target_cube/
```

### Target cube manager

```text
ros2_ws/src/so101_gazebo/scripts/target_cube_manager.py
```

### Formal launch

```text
ros2_ws/src/so101_bringup/launch/sim.launch.py
```

---

## 5. Source 与 install 的关系

Source files 位于：

```text
ros2_ws/src/
```

构建后，ROS 2 正式运行通常读取：

```text
ros2_ws/install/
```

因此修改 source 后必须重新构建并重新 source。

示例：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

如果修改后没有生效，首先检查是否仍在使用旧的 `install/` 副本。

---

## 6. 每次启动前

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

确认 packages：

```bash
ros2 pkg prefix so101_description
ros2 pkg prefix so101_gazebo
ros2 pkg prefix so101_bringup
```

---

## 7. 正式启动

```bash
ros2 launch so101_bringup sim.launch.py
```

Launch file 会：

1. 查找 package share directories
2. 检查 Xacro、controller、world 和 target cube configuration
3. 解析 Xacro
4. 设置 Gazebo resource paths
5. 设置 Gazebo plugin paths
6. 启动 Gazebo Fortress
7. 启动 `robot_state_publisher`
8. 启动 ROS-Gazebo sensor bridge
9. 生成 SO-101 entity
10. 启动 target cube manager
11. 启动 `joint_state_broadcaster`
12. 启动 `position_controller`

---

## 8. 启动时间顺序

当前 `sim.launch.py` 使用以下延时：

| Launch time | Action |
|---:|---|
| Immediately | Gazebo、`robot_state_publisher`、sensor bridge |
| `2.0 s` | Spawn SO-101 |
| `2.0 s` | Start target cube manager |
| `5.0 s` | Start `joint_state_broadcaster` |
| `8.0 s` | Start `position_controller` |

这些延时用于等待 Gazebo、robot entity 和 controller manager 初始化。

当前延时已经验证可以正常启动，不应随意缩短。

---

## 9. World 坐标约定

当前 world 坐标约定：

```text
+X = 向桌面中心 / 工作区前方
+Y = 机械臂左侧
+Z = 向上
```

机械臂底座世界坐标中心：

```text
(0, 0)
```

Target cube 随机生成区域也以该点为中心。

---

## 10. Laboratory table

Table model：

```text
so101_gazebo/models/lab_table/
```

主要尺寸：

| Parameter | Value |
|---|---:|
| Length | `1.200 m` |
| Width | `0.600 m` |
| Thickness | `0.034 m` |
| Top surface | `z = 0.750 m` |
| Leg height | `0.716 m` |

World pose：

```xml
<pose>0.278 0.05 0 0 0 1.570796</pose>
```

该 pose 已根据机械臂位置和桌面边缘关系调整。

当前布局实现：

- 机械臂位于桌面近边
- 桌面前方为抓取区域
- 桌面左右空间符合当前实验布置
- 机械臂底座正确落在桌面上

---

## 11. SO-101 生成位置

Robot entity name：

```text
so101
```

正式 spawn position：

```text
x = 0.0
y = 0.0
z = 0.7524
```

桌面上表面为：

```text
z = 0.7500
```

Robot spawn height 比桌面高 `0.0024 m`，用于补偿 base mesh 原点约 `-0.0024 m` 的偏移。

不要将 spawn height 改为 `0.7500`，除非重新验证：

- Base visual
- Base collision
- Table contact
- Robot height
- Camera observations
- Grasping workspace

---

## 12. Robot link tree

主要 link tree：

```text
world
└── base_link
    ├── shoulder_link
    │   └── upper_arm_link
    │       └── lower_arm_link
    │           └── wrist_link
    │               └── gripper_link
    │                   ├── moving_jaw_so101_v1_link
    │                   ├── gripper_camera_mount_link
    │                   └── gripper_frame_link
    │                       └── gripper_camera_link
    └── top_camera_link
```

主要 controlled joints：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

---

## 13. ros2_control

SO-101 Xacro 使用：

```text
gz_ros2_control/GazeboSimSystem
```

每个 controlled joint 提供：

```text
Command:
  position

State:
  position
  velocity
```

Gazebo control plugin 从：

```text
so101_gazebo/config/controllers.yaml
```

加载配置。

---

## 14. Controllers

当前 controllers：

```text
joint_state_broadcaster
position_controller
```

Controller manager update rate：

```text
100 Hz
```

`position_controller` 类型：

```text
forward_command_controller/ForwardCommandController
```

Interface：

```text
position
```

控制顺序：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

该 joint 顺序应在后续控制、数据采集和 policy interface 中保持一致。

---

## 15. Controller 验证

```bash
ros2 control list_controllers
```

预期：

```text
joint_state_broadcaster
position_controller
```

两者应为：

```text
active
```

检查 hardware interfaces：

```bash
ros2 control list_hardware_interfaces
```

检查 joint states：

```bash
ros2 topic echo /joint_states --once
```

---

## 16. Top camera

当前 Top camera：

| Parameter | Value |
|---|---:|
| Resolution | `640 × 480` |
| Update rate | `30 Hz` |
| Horizontal FOV | `2.792527 rad` |
| Approximate FOV | `160°` |

ROS 2 topics：

```text
/top_camera/image_raw
/top_camera/camera_info
```

---

## 17. Gripper camera

当前 Gripper camera：

| Parameter | Value |
|---|---:|
| Resolution | `640 × 480` |
| Update rate | `30 Hz` |
| Horizontal FOV | `0.872665 rad` |
| Approximate FOV | `50°` |

Sensor pose：

```xml
<pose>0.007 -0.0015 0.0001 0 -1.570796 1.570796</pose>
```

ROS 2 topics：

```text
/gripper_camera/image_raw
/gripper_camera/camera_info
```

---

## 18. Camera bridge

当前正式 bridge 包含：

```text
/clock
/top_camera/image_raw
/top_camera/camera_info
/gripper_camera/image_raw
/gripper_camera/camera_info
```

检查：

```bash
ros2 topic list | grep -E 'clock|top_camera|gripper_camera'
```

检查频率：

```bash
ros2 topic hz /top_camera/image_raw
```

```bash
ros2 topic hz /gripper_camera/image_raw
```

---

## 19. Segmentation sensor 状态

Xacro 中已定义：

```text
/top_camera/segmentation
/top_camera/segmentation/camera_info
/gripper_camera/segmentation
/gripper_camera/segmentation/camera_info
```

但当前正式 launch 未 bridge segmentation topics。

因此当前状态：

```text
Sensor definition: 已存在
Gazebo sensor: 已配置
Formal ROS 2 bridge: 尚未加入
Formal ROS 2 verification: 尚未完成
```

在正式 bridge 和验证完成前，不应把 segmentation ROS 2 topics 写成现有可用接口。

---

## 20. Scene 与 lighting

当前 scene：

```xml
<scene>
  <ambient>0.35 0.35 0.35 1</ambient>
  <background>0.70 0.72 0.75 1</background>
  <shadows>true</shadows>
</scene>
```

当前主光源：

```text
main_overhead_light
```

类型：

```text
directional
```

方向：

```text
-0.25 -0.30 -0.92
```

---

## 21. 为什么不使用 point light

以前测试 point light 时，ROS camera image 中出现：

- 十字形 artefact
- 球状光源 artefact

因此正式 world 已删除 point light，只保留 directional light。

任何 lighting 修改都必须同时检查：

- Gazebo GUI
- Top camera image
- Gripper camera image
- Shadow
- Brightness
- Exposure consistency

---

## 22. Robot materials

当前视觉设计：

| Component | Appearance |
|---|---|
| 3D printed parts | 白色 PLA |
| STS3215 servos | 深黑色塑料 |
| Top camera | 黑色塑料 |
| Gripper camera | 黑色塑料 |
| Gripper camera mount | 白色 PLA |

Gazebo Fortress 对全局 URDF material reference 存在显示问题。

因此部分 mesh visual 有意使用 inline material color。

不要删除这些 inline materials，否则白色 mesh 可能显示为黑色。

---

## 23. Target cube model

Target cube：

| Parameter | Value |
|---|---:|
| Size | `0.030 × 0.030 × 0.030 m` |
| Mass | `0.020 kg` |
| Type | Dynamic |
| Appearance | Green matte PLA |
| Origin | Cube centre |

Table surface：

```text
z = 0.750 m
```

Spawn clearance：

```text
0.003 m
```

Spawn centre height：

```text
0.768 m
```

稳定后中心高度约：

```text
0.765 m
```

---

## 24. Target cube spawn region

Spawn centre：

```text
world (0, 0)
```

Region：

```text
Front half-annulus
```

参数：

| Parameter | Value |
|---|---:|
| Inner radius | `0.10 m` |
| Outer radius | `0.40 m` |
| Angle min | `-90°` |
| Angle max | `+90°` |
| Minimum centre distance | `0.05 m` |
| Maximum spawn attempts | `100` |
| Maximum cube count | `20` |

Random sampling 使用面积均匀采样。

---

## 25. Target cube manager

Manager：

```text
so101_gazebo/scripts/target_cube_manager.py
```

Node：

```text
target_cube_manager
```

Manager 自动随正式 simulation launch 启动。

主要功能：

- Random spawn
- Unique naming
- Entity ID discovery
- Pose tracking
- JSON state publication
- Per-cube pose publication
- Delete by name
- Delete all
- Deleted pose suppression

---

## 26. Gazebo 与 ROS 2 数据流

```text
Gazebo Fortress
    │
    ├── Robot dynamics
    ├── Camera sensors
    ├── Entity create/remove
    ├── Dynamic pose stream
    │
    ├──────── ros_gz_bridge ────────► ROS 2
    │                                 ├── /clock
    │                                 ├── /top_camera/image_raw
    │                                 ├── /top_camera/camera_info
    │                                 ├── /gripper_camera/image_raw
    │                                 └── /gripper_camera/camera_info
    │
    ├──────── gz_ros2_control ──────► controller_manager
    │                                 ├── joint_state_broadcaster
    │                                 └── position_controller
    │
    └──────── ign topic/service ────► target_cube_manager
                                      ├── spawn service
                                      ├── delete service
                                      ├── states topic
                                      └── per-cube pose topics
```

---

## 27. Target cube ROS 2 interfaces

Spawn：

```text
/target_cube/spawn_random
std_srvs/srv/Trigger
```

Delete all：

```text
/target_cube/delete_all
std_srvs/srv/Trigger
```

Delete by name：

```text
/target_cube/delete_name
std_msgs/msg/String
```

Delete result：

```text
/target_cube/delete_result
std_msgs/msg/String
```

All states：

```text
/target_cube/states
std_msgs/msg/String
```

Per-cube pose：

```text
/target_cube/pose/target_cube_XXXX
geometry_msgs/msg/PoseStamped
```

---

## 28. Target cube quick test

Spawn：

```bash
ros2 service call \
  /target_cube/spawn_random \
  std_srvs/srv/Trigger \
  "{}"
```

View states：

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

View pose：

```bash
ros2 topic echo \
  /target_cube/pose/target_cube_0001 \
  --once
```

Delete one：

```bash
ros2 topic pub \
  --once \
  /target_cube/delete_name \
  std_msgs/msg/String \
  "{data: 'target_cube_0001'}"
```

Delete all：

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

---

## 29. Deleted pose suppression

Gazebo 可能在 entity 删除后短时间继续发布旧 pose。

Manager 当前使用约：

```text
2.0 s
```

的 ignore window，防止：

- 已删除 cube 被旧 pose 重新加入状态
- Per-cube pose publisher 被重新创建
- Delete result 与 states 不一致

该逻辑已经用于解决实际出现的 stale pose 问题，不应随意删除。

---

## 30. 启动后验证清单

### Nodes

```bash
ros2 node list
```

重点确认：

```text
/robot_state_publisher
/sensor_bridge
/target_cube_manager
/controller_manager
```

### Controllers

```bash
ros2 control list_controllers
```

### Joint states

```bash
ros2 topic echo /joint_states --once
```

### Cameras

```bash
ros2 topic list | grep -E 'top_camera|gripper_camera'
```

### Cube services

```bash
ros2 service list | grep target_cube
```

### Cube topics

```bash
ros2 topic list | grep target_cube
```

---

## 31. Xacro 验证

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash

xacro \
  src/so101_description/urdf/so101.urdf.xacro \
  controllers_file:="$(pwd)/src/so101_gazebo/config/controllers.yaml" \
  > /tmp/so101_test.urdf

check_urdf /tmp/so101_test.urdf
```

该验证已成功完成。

---

## 32. Build

构建三个正式 packages：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash

colcon build \
  --packages-select \
  so101_description \
  so101_gazebo \
  so101_bringup

source install/setup.bash
```

如果该完整命令尚未在当前文档批次后重新运行，应在最终提交前执行一次。

---

## 33. Clean rebuild

只有在 build cache 或 install copy 明显异常时才执行：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

rm -rf build install log

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

该操作不会删除：

```text
src/
```

但会重新生成全部 build outputs。

---

## 34. 常见问题

### 修改 source 后没有生效

执行：

```bash
colcon build
source install/setup.bash
```

确认 package prefix：

```bash
ros2 pkg prefix so101_bringup
```

### Camera topics 不出现

检查：

```bash
ros2 node list | grep sensor_bridge
ign topic -l | grep -E 'top_camera|gripper_camera'
```

### Controllers 不 active

等待 launch 至少 `8 s` 后：

```bash
ros2 control list_controllers
```

### Table 或 cube model 找不到

检查安装目录：

```bash
find \
  install/so101_gazebo/share/so101_gazebo/models \
  -maxdepth 3 \
  -type f \
  | sort
```

### 其他终端看不到节点

确认相同：

```text
ROS_DOMAIN_ID=20
GZ_PARTITION=gz_env_20
IGN_PARTITION=gz_env_20
```

---

## 35. 当前已知限制

当前应明确记录：

- Segmentation sensors 已定义，但正式 ROS 2 bridge 尚未加入
- `target_cube/model.config` description 仍写 `40 mm`，实际模型为 `30 mm`
- Training、policy 和 data collection interface 尚未作为正式 Gazebo workflow 完成
- 当前主要 controller 为 position controller
- World 文件名仍为 `empty.sdf`，但场景已包含 table、light 和 ground
- 当前仿真 baseline 主要用于单机械臂桌面抓取任务

---

## 36. 不应随意修改的配置

以下内容已经验证：

- Robot Xacro
- Joint names
- Joint origins
- Joint limits
- Mesh paths
- `ros2_control`
- Controllers
- Camera poses
- Camera FOV
- Camera resolution
- Camera topics
- Robot spawn height
- Table pose
- Table height
- World systems
- Directional light
- Cube size
- Cube mass
- Cube spawn region
- Entity ID deletion
- Stale pose suppression
- Startup delays

任何修改都必须重新执行：

1. Build
2. Xacro / URDF validation
3. Gazebo startup
4. Controller check
5. Camera check
6. Cube manager check
7. Manual visual inspection

---

## 37. Xacro 修改注意事项

不要使用：

```python
xml.etree.ElementTree.ElementTree.write()
```

重写整个 Xacro。

该方法曾导致：

- `xmlns:xacro` 变成 `xmlns:ns0`
- Comments 被破坏
- Indentation 变化
- Git diff 大范围变化

小范围修改应使用精确文本替换或小型 patch。

---

## 38. Git 注意事项

不要提交：

```text
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
__pycache__/
```

文档完成前可以暂不 commit。

最终统一检查：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```
