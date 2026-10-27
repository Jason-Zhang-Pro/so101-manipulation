# SO-101 ROS 2 / Gazebo Workspace

本目录是 Capstone 项目的正式 ROS 2 与 Gazebo 工作区，负责 SO-101 机械臂模型、Gazebo 场景、相机、`ros2_control`、controllers、正式启动流程以及动态目标方块管理。

项目根目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone
```

ROS 2 工作区：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws
```

> 本文档只描述已经建立并验证的 ROS 2 / Gazebo 系统。
> LeRobot、真实机械臂操作、训练、数据采集和 policy workflow 将在其他项目文档中单独说明。

---

## 1. 已验证环境

| Component | Current environment |
|---|---|
| Operating system | Ubuntu 22.04 |
| ROS distribution | ROS 2 Humble |
| Simulator | Gazebo Fortress |
| Build system | `colcon` + `ament_cmake` |
| Main custom node language | Python |
| ROS domain | `ROS_DOMAIN_ID=20` |
| Gazebo partition | `gz_env_20` |

当前正式环境变量：

```bash
export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

`GZ_PARTITION` 和 `IGN_PARTITION` 使用相同值，是为了兼容当前 Gazebo Fortress / Ignition 工具命名。

---

## 2. 工作区结构

```text
ros2_ws/
├── build/                       # colcon 生成，不提交 Git
├── install/                     # colcon 生成，不提交 Git
├── log/                         # colcon 生成，不提交 Git
└── src/
    ├── so101_description/
    │   ├── meshes/
    │   ├── urdf/
    │   ├── CMakeLists.txt
    │   └── package.xml
    ├── so101_gazebo/
    │   ├── config/
    │   ├── models/
    │   ├── scripts/
    │   ├── worlds/
    │   ├── CMakeLists.txt
    │   └── package.xml
    └── so101_bringup/
        ├── launch/
        ├── CMakeLists.txt
        └── package.xml
```

生成目录：

```text
build/
install/
log/
```

均由 `colcon` 创建，不应手动编辑，也不应提交到 Git。

---

## 3. ROS 2 packages

### 3.1 `so101_description`

负责：

- SO-101 正式 Xacro 模型
- Robot links 和 joints
- Mesh assets
- Joint limits
- Top camera 与 gripper camera links
- RGB 和 segmentation camera sensors
- `ros2_control` hardware interfaces
- Gazebo visual materials

正式 Xacro：

```text
src/so101_description/urdf/so101.urdf.xacro
```

安装后的副本：

```text
install/so101_description/share/so101_description/urdf/so101.urdf.xacro
```

---

### 3.2 `so101_gazebo`

负责：

- Gazebo world
- Laboratory table model
- Dynamic target cube model
- Controller configuration
- Target cube manager configuration
- Target cube manager Python node

主要文件：

```text
src/so101_gazebo/worlds/empty.sdf
src/so101_gazebo/config/controllers.yaml
src/so101_gazebo/config/target_cube.yaml
src/so101_gazebo/scripts/target_cube_manager.py
src/so101_gazebo/models/lab_table/
src/so101_gazebo/models/target_cube/
```

---

### 3.3 `so101_bringup`

负责正式仿真系统的统一启动。

正式 launch file：

```text
src/so101_bringup/launch/sim.launch.py
```

该 launch file 会启动：

- Gazebo Fortress
- 正式 world
- `robot_state_publisher`
- SO-101 robot entity
- ROS-Gazebo sensor bridge
- Target cube manager
- `joint_state_broadcaster`
- `position_controller`

---

## 4. 初次构建

进入工作区：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"
```

加载 ROS 2 Humble：

```bash
source /opt/ros/humble/setup.bash
```

构建工作区：

```bash
colcon build
```

构建成功后加载本工作区：

```bash
source install/setup.bash
```

每次修改以下内容后都需要重新构建：

- Xacro
- Mesh installation rules
- Gazebo world
- Gazebo models
- YAML configuration
- Launch files
- Installed Python scripts
- `CMakeLists.txt`
- `package.xml`

源文件位于 `src/` 中。运行时通常使用 `install/` 下的安装副本，因此修改源文件但未重新构建，可能导致 Gazebo 继续加载旧版本。

---

## 5. 每次使用工作区

打开新终端后执行：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

可以使用以下命令确认 package 已被发现：

```bash
ros2 pkg prefix so101_description
ros2 pkg prefix so101_gazebo
ros2 pkg prefix so101_bringup
```

---

## 6. 正式启动仿真

```bash
ros2 launch so101_bringup sim.launch.py
```

启动文件会自动设置 Gazebo resource path 和 plugin path，包括：

- ROS 2 package share directories
- `so101_gazebo/models`
- `/opt/ros/humble/lib`
- 已有的 Gazebo resource 和 plugin path

正常情况下，不需要手动把模型目录复制到用户级 Gazebo model folder。

---

## 7. 正式 world

World source：

```text
src/so101_gazebo/worlds/empty.sdf
```

World name：

```text
empty
```

当前 world 包含：

- Physics system
- User commands system
- Scene broadcaster
- Contact system
- Sensors system using Ogre 2
- Ground plane
- Laboratory table
- Directional overhead light

当前坐标约定：

```text
+X = 向桌面中心 / 工作区前方
+Y = 机械臂左侧
+Z = 向上
```

---

## 8. 桌子模型

模型目录：

```text
src/so101_gazebo/models/lab_table/
```

主要尺寸：

| Parameter | Value |
|---|---:|
| Tabletop length | `1.200 m` |
| Tabletop width | `0.600 m` |
| Tabletop thickness | `0.034 m` |
| Tabletop top surface | `z = 0.750 m` |
| Leg cross-section | `0.050 × 0.050 m` |
| Leg height | `0.716 m` |

World pose：

```xml
<pose>0.278 0.05 0 0 0 1.570796</pose>
```

该位置已经根据机械臂底座和桌面边缘关系进行调整并完成视觉检查。

---

## 9. SO-101 生成位置

SO-101 entity name：

```text
so101
```

正式生成位置：

```text
x = 0.0
y = 0.0
z = 0.7524
```

`z=0.7524` 用于补偿 base mesh 原点约 `-0.0024 m` 的偏移，使机械臂底座正确落在 `z=0.750 m` 的桌面上。

不要把该高度当作普通场景参数随意修改。

---

## 10. Controllers

Controller configuration：

```text
src/so101_gazebo/config/controllers.yaml
```

Controller manager update rate：

```text
100 Hz
```

当前 controllers：

```text
joint_state_broadcaster
position_controller
```

`position_controller` 控制以下六个 joints：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

Command interface：

```text
position
```

State interfaces：

```text
position
velocity
```

检查 controller 状态：

```bash
ros2 control list_controllers
```

检查 hardware interfaces：

```bash
ros2 control list_hardware_interfaces
```

检查关节状态：

```bash
ros2 topic echo /joint_states --once
```

---

## 11. 相机与 ROS 2 bridge

### Top camera

| Parameter | Value |
|---|---:|
| Resolution | `640 × 480` |
| Update rate | `30 Hz` |
| Horizontal FOV | `2.792527 rad` |
| Approximate FOV | `160°` |

已桥接 ROS 2 topics：

```text
/top_camera/image_raw
/top_camera/camera_info
```

### Gripper camera

| Parameter | Value |
|---|---:|
| Resolution | `640 × 480` |
| Update rate | `30 Hz` |
| Horizontal FOV | `0.872665 rad` |
| Approximate FOV | `50°` |

已桥接 ROS 2 topics：

```text
/gripper_camera/image_raw
/gripper_camera/camera_info
```

Gripper camera sensor pose：

```xml
<pose>0.007 -0.0015 0.0001 0 -1.570796 1.570796</pose>
```

检查相机 topics：

```bash
ros2 topic list | grep -E 'top_camera|gripper_camera'
```

检查发布频率：

```bash
ros2 topic hz /top_camera/image_raw
ros2 topic hz /gripper_camera/image_raw
```

显示图像时可以使用已安装的 ROS 2 image viewer，例如：

```bash
ros2 run rqt_image_view rqt_image_view
```

### Segmentation sensors

Xacro 中已经定义：

```text
/top_camera/segmentation
/gripper_camera/segmentation
```

以及对应 CameraInfo topics。

但是当前正式 `sim.launch.py` 只 bridge RGB images 和 RGB CameraInfo。Segmentation topics 在加入正式 bridge 并重新验证之前，不应视为当前 ROS 2 可用接口。

---

## 12. 动态目标方块

Target cube model：

```text
src/so101_gazebo/models/target_cube/model.sdf
```

Manager configuration：

```text
src/so101_gazebo/config/target_cube.yaml
```

Manager node：

```text
src/so101_gazebo/scripts/target_cube_manager.py
```

Manager 会随正式 simulation launch 自动启动。

### 当前参数

| Parameter | Value |
|---|---:|
| Cube size | `0.030 × 0.030 × 0.030 m` |
| Mass | `0.020 kg` |
| Table surface | `z = 0.750 m` |
| Spawn clearance | `0.003 m` |
| Spawn centre height | `z = 0.768 m` |
| Inner radius | `0.10 m` |
| Outer radius | `0.40 m` |
| Angular range | `-90°` to `+90°` |
| Minimum centre distance | `0.05 m` |
| Maximum managed count | `20` |
| State publish rate | `10 Hz` |

随机位置使用以前半圆环为边界的面积均匀采样。

### ROS 2 interfaces

生成一个方块：

```bash
ros2 service call \
  /target_cube/spawn_random \
  std_srvs/srv/Trigger \
  "{}"
```

查看所有状态：

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

查看单个方块 pose：

```bash
ros2 topic echo \
  /target_cube/pose/target_cube_0001 \
  --once
```

按名称删除：

```bash
ros2 topic pub \
  --once \
  /target_cube/delete_name \
  std_msgs/msg/String \
  "{data: 'target_cube_0001'}"
```

查看删除结果：

```bash
ros2 topic echo \
  /target_cube/delete_result \
  --once \
  --full-length
```

删除全部方块：

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

Manager 内部通过：

- `ign service` 创建和删除 entity
- `ign topic` 长期读取 `/world/empty/dynamic_pose/info`
- Gazebo entity ID 执行删除
- `rclpy` 暴露 ROS 2 services 和 topics

---

## 13. 启动后验证清单

### 13.1 检查节点

```bash
ros2 node list
```

应能看到与以下功能对应的节点：

```text
robot_state_publisher
sensor_bridge
target_cube_manager
controller_manager
```

实际名称可能带有 `/` 前缀。

### 13.2 检查 topics

```bash
ros2 topic list
```

重点确认：

```text
/clock
/joint_states
/top_camera/image_raw
/top_camera/camera_info
/gripper_camera/image_raw
/gripper_camera/camera_info
/target_cube/states
/target_cube/delete_name
/target_cube/delete_result
```

单个方块生成后还会出现：

```text
/target_cube/pose/target_cube_XXXX
```

### 13.3 检查 services

```bash
ros2 service list | grep target_cube
```

预期：

```text
/target_cube/spawn_random
/target_cube/delete_all
```

### 13.4 检查 controllers

```bash
ros2 control list_controllers
```

预期为 active：

```text
joint_state_broadcaster
position_controller
```

---

## 14. Xacro 验证

进入 workspace：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"
```

生成临时 URDF：

```bash
xacro \
  src/so101_description/urdf/so101.urdf.xacro \
  controllers_file:="$(pwd)/src/so101_gazebo/config/controllers.yaml" \
  > /tmp/so101_test.urdf
```

检查 URDF：

```bash
check_urdf /tmp/so101_test.urdf
```

临时文件位于 `/tmp`，不应提交到项目仓库。

---

## 15. 重新构建

普通修改后：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

只有在安装副本或 build cache 明显异常时，才执行 clean rebuild：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

rm -rf build install log

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

`build/`、`install/` 和 `log/` 都是生成目录。删除它们不会删除 `src/` 中的源代码，但 clean rebuild 会重新生成整个工作区。

---

## 16. 常见检查

### 修改后没有生效

确认已经重新构建并重新 source：

```bash
colcon build
source install/setup.bash
```

确认加载的是当前 workspace：

```bash
ros2 pkg prefix so101_bringup
```

### Gazebo 找不到 table 或 target cube model

检查 package 是否已构建：

```bash
ros2 pkg prefix so101_gazebo
```

检查安装目录：

```bash
find install/so101_gazebo/share/so101_gazebo/models -maxdepth 3 -type f
```

### Camera topics 不出现

检查 bridge node：

```bash
ros2 node list | grep sensor_bridge
```

检查 Gazebo sensor system 和 topics：

```bash
ign topic -l | grep -E 'top_camera|gripper_camera'
```

### Controllers 未启动

等待 launch file 中的 controller startup delay 完成，然后检查：

```bash
ros2 control list_controllers
```

检查 controller manager：

```bash
ros2 node list | grep controller_manager
```

### ROS 2 看不到其他终端的节点

确认所有终端使用相同：

```bash
export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

---

## 17. 已验证配置保护原则

以下内容已经验证，不应随意修改：

- Joint names
- Joint origins
- Joint limits
- Mesh paths
- `ros2_control` interfaces
- `controllers.yaml`
- Camera poses
- Camera FOV
- Camera resolutions
- Camera topic names
- Robot spawn height
- Table pose
- Directional light configuration
- Target cube spawn region
- Target cube deletion logic
- Gazebo entity ID tracking

任何修改都应：

1. 只改一个明确组件
2. 检查 Git diff
3. 重新构建
4. 运行相关验证命令
5. 人工检查 Gazebo 模型、控制或相机
6. 用户确认后再 commit

---

## 18. Xacro 和材质注意事项

不要使用 `xml.etree.ElementTree.ElementTree.write()` 重写完整 Xacro。

该操作曾导致：

- `xmlns:xacro` 被改成 `xmlns:ns0`
- Comments 和 indentation 被破坏
- Git diff 大范围变化
- Xacro 可读性下降

小范围修改应优先使用精确文本替换。

部分 visual 使用 inline material 是有意设计。Gazebo Fortress 对全局 URDF material reference 的渲染并不稳定，删除 inline color 可能使白色 3D printed parts 显示为黑色。

---

## 19. 灯光注意事项

当前 world 使用单个 directional light。

以前测试 point light 时，ROS camera image 中出现明显十字形或球状 artefact，因此 point light 已被删除。

在重新加入额外 light source 前，必须同时检查：

- Gazebo GUI
- Top camera ROS image
- Gripper camera ROS image
- Shadow quality
- Exposure 和亮度一致性

---

## 20. Git 注意事项

以下目录不提交：

```text
build/
install/
log/
```

检查工作区：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
```

文档完善期间可以暂不 commit，但最终提交前必须确认：

- 没有 build output
- 没有 Python cache
- 没有编辑器 backup
- 没有临时 URDF
- 没有不相关代码修改
