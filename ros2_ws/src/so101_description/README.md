# so101_description

`so101_description` 是 Capstone 项目中 SO-101 机械臂描述 package，负责保存正式 Xacro、mesh、link / joint 结构、相机 sensor、Gazebo 材质以及 `ros2_control` 接口。

Package 路径：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws/src/so101_description
```

正式 Xacro：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws/src/so101_description/urdf/so101.urdf.xacro
```

该 package 已用于当前正式 Gazebo 仿真，并已经完成模型、控制和相机的人工检查。

---

## 1. Package 职责

本 package 负责：

- SO-101 robot description
- Xacro / URDF 结构
- Robot links 与 joints
- Joint origins 与 joint limits
- Mesh assets
- Top camera link 和 sensor
- Gripper camera mount、link 和 sensor
- RGB camera sensors
- Segmentation camera sensors
- `ros2_control` command / state interfaces
- Gazebo visual materials
- Gazebo control plugin configuration

本 package 不负责：

- Gazebo world
- Table model
- Target cube model
- Controller YAML
- Dynamic target cube manager
- 完整 simulation launch

这些内容分别位于：

```text
so101_gazebo
so101_bringup
```

---

## 2. 目录结构

```text
so101_description/
├── meshes/
│   ├── base_motor_holder_so101_v1.stl
│   ├── base_so101_v2.stl
│   ├── gripper_camera_mount.stl
│   ├── motor_holder_so101_base_v1.stl
│   ├── motor_holder_so101_wrist_v1.stl
│   ├── moving_jaw_so101_v1.stl
│   ├── rotation_pitch_so101_v1.stl
│   ├── sts3215_03a_no_horn_v1.stl
│   ├── sts3215_03a_v1.stl
│   ├── under_arm_so101_v1.stl
│   ├── upper_arm_so101_v1.stl
│   ├── waveshare_mounting_plate_so101_v2.stl
│   ├── wrist_roll_follower_so101_v1.stl
│   └── wrist_roll_pitch_so101_v2.stl
├── urdf/
│   └── so101.urdf.xacro
├── CMakeLists.txt
├── package.xml
└── README.md
```

`config/`、`rviz/`、`include/` 和 `src/` 目录由 package 结构保留，但当前正式模型主要使用 `urdf/` 和 `meshes/`。

---

## 3. 正式模型文件

Source Xacro：

```text
src/so101_description/urdf/so101.urdf.xacro
```

构建后的安装副本：

```text
install/so101_description/share/so101_description/urdf/so101.urdf.xacro
```

Gazebo 正式运行时通过 ROS 2 package share directory 读取安装副本。

因此修改 source Xacro 后，必须重新构建：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
colcon build --packages-select so101_description
source install/setup.bash
```

如果同时修改了 controller、Gazebo package 或 launch file，应重新构建相关 packages。

---

## 4. Xacro 参数

当前 Xacro 定义：

```xml
<xacro:arg name="controllers_file" default=""/>
```

该参数由：

```text
so101_bringup/launch/sim.launch.py
```

传入，并指向：

```text
so101_gazebo/config/controllers.yaml
```

Xacro 中 Gazebo control plugin 使用：

```xml
<parameters>$(arg controllers_file)</parameters>
```

不要删除或改名该参数，否则正式 launch 无法按当前方式加载 controller configuration。

---

## 5. Robot link tree

当前主要 link tree：

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

主要 links：

```text
world
base_link
shoulder_link
upper_arm_link
lower_arm_link
wrist_link
gripper_link
moving_jaw_so101_v1_link
gripper_frame_link
gripper_camera_mount_link
gripper_camera_link
top_camera_link
```

---

## 6. Robot joints

主要 actuated joints：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

固定 joints 包括：

```text
world_to_base
gripper_frame_joint
gripper_camera_mount_joint
gripper_camera_joint
top_camera_joint
```

这六个 actuated joints 与：

```text
so101_gazebo/config/controllers.yaml
```

中的 `position_controller` 保持一致。

不要单独修改 joint name。Joint name 一旦改变，还必须同步检查：

- `controllers.yaml`
- `ros2_control`
- Command topics
- Joint state consumers
- Training / policy interfaces
- Data collection code
- Gazebo control plugin

---

## 7. ros2_control

当前 Xacro 包含：

```xml
<ros2_control name="GazeboSimSystem" type="system">
```

Hardware plugin：

```xml
<plugin>gz_ros2_control/GazeboSimSystem</plugin>
```

六个 joints 均提供：

```text
Command interface:
  position

State interfaces:
  position
  velocity
```

当前 Gazebo plugin：

```xml
<plugin
  filename="gz_ros2_control-system"
  name="gz_ros2_control::GazeboSimROS2ControlPlugin">
```

Controller parameters 通过 Xacro argument 加载。

该部分已经验证可以：

- 启动 controller manager
- 启动 `joint_state_broadcaster`
- 启动 `position_controller`
- 获取 joint states
- 控制六个关节

除非重新验证完整 control chain，否则不要修改：

- Hardware plugin
- Command interfaces
- State interfaces
- Joint names
- Controller parameter path

---

## 8. Mesh assets

当前正式 Xacro 使用的 SO-101 mesh 均保存在：

```text
so101_description/meshes/
```

当前 package 中共有 14 个 mesh 文件，其中包括 13 个原始 SO-101 结构 / 舵机 mesh 和 1 个项目新增的夹爪相机支架：

```text
gripper_camera_mount.stl
```

Mesh URI 应使用 ROS package 路径，例如：

```text
package://so101_description/meshes/<filename>.stl
```

正式项目不应继续从旧项目目录或 `references/` 直接加载 mesh。

---

## 9. 模型来源与迁移记录

当前模型最初参考了旧项目中的 SO-101 calibrated URDF：

```text
/media/tiger/New Volume/ubuntufile/old_cap_robot/cap_robot/
robot_models/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

迁移记录位于：

```text
docs/migration/so101_urdf_source.txt
docs/migration/old_cap_robot_assets.txt
```

`so101_urdf_source.txt` 中记录的 destination 是迁移当时的：

```text
ros2_ws/src/so101_description/urdf/so101.urdf
```

当前正式文件已经转换为：

```text
ros2_ws/src/so101_description/urdf/so101.urdf.xacro
```

因此 migration 文件是历史 provenance 记录，不代表当前正式运行路径。

官方 SO-ARM101 教程和原始资料保存在：

```text
references/SO-ARM101_original/
```

该目录是官方教程的只读归档，不是正式 runtime directory。

---

## 10. 材质设计

当前机械臂视觉设计：

| Component | Appearance |
|---|---|
| 3D printed structure | 白色 PLA |
| STS3215 servos | 深黑色塑料 |
| Top camera | 黑色塑料 |
| Gripper camera body | 黑色塑料 |
| Gripper camera lens | 深黑色 |
| Gripper camera mount | 白色 PLA |

全局材质定义包括：

```xml
<material name="3d_printed">
  <color rgba="0.88 0.88 0.85 1.0"/>
</material>
```

```xml
<material name="sts3215">
  <color rgba="0.03 0.03 0.03 1.0"/>
</material>
```

```xml
<material name="camera_black_plastic">
  <color rgba="0.03 0.03 0.03 1.0"/>
</material>
```

---

## 11. 为什么保留 inline material

Gazebo Fortress 对 mesh visual 的全局 URDF material reference 存在显示问题。

只写：

```xml
<material name="3d_printed"/>
```

时，部分本应为白色的 mesh 可能回退为黑色。

因此部分 visual 中有意使用 inline material：

```xml
<material name="3d_printed">
  <color rgba="0.88 0.88 0.85 1.0"/>
</material>
```

以及：

```xml
<material name="sts3215">
  <color rgba="0.03 0.03 0.03 1.0"/>
</material>
```

这些 inline colors 是当前 Gazebo Fortress 渲染兼容方案，不是重复或无用配置。

不要为了“精简 XML”删除它们。

---

## 12. Top camera

Top camera link：

```text
top_camera_link
```

固定 joint：

```text
top_camera_joint
```

RGB sensor：

```text
top_camera_sensor
```

当前参数：

| Parameter | Value |
|---|---:|
| Type | Camera |
| Resolution | `640 × 480` |
| Update rate | `30 Hz` |
| Horizontal FOV | `2.792527 rad` |
| Approximate FOV | `160°` |

Gazebo topic：

```text
/top_camera/image_raw
```

CameraInfo topic：

```text
/top_camera/camera_info
```

正式 `sim.launch.py` 已将以上 RGB topics bridge 到 ROS 2。

---

## 13. Gripper camera

相关 links：

```text
gripper_camera_mount_link
gripper_camera_link
```

相关 fixed joints：

```text
gripper_camera_mount_joint
gripper_camera_joint
```

RGB sensor：

```text
gripper_camera_sensor
```

当前参数：

| Parameter | Value |
|---|---:|
| Type | Camera |
| Resolution | `640 × 480` |
| Update rate | `30 Hz` |
| Horizontal FOV | `0.872665 rad` |
| Approximate FOV | `50°` |

Sensor pose：

```xml
<pose>0.007 -0.0015 0.0001 0 -1.570796 1.570796</pose>
```

Gazebo topic：

```text
/gripper_camera/image_raw
```

CameraInfo topic：

```text
/gripper_camera/camera_info
```

正式 `sim.launch.py` 已将以上 RGB topics bridge 到 ROS 2。

---

## 14. Segmentation sensors

Xacro 中还定义：

```text
top_camera_segmentation_sensor
gripper_camera_segmentation_sensor
```

对应 topics：

```text
/top_camera/segmentation
/top_camera/segmentation/camera_info
/gripper_camera/segmentation
/gripper_camera/segmentation/camera_info
```

这些 sensors 存在于 Xacro 中，但当前正式：

```text
so101_bringup/launch/sim.launch.py
```

只 bridge RGB image 和 RGB CameraInfo。

因此当前文档状态应区分：

- Segmentation sensor 已定义
- Segmentation ROS 2 bridge 尚未作为正式已验证接口记录

不要仅因为 Xacro 中存在 topic，就声称 ROS 2 中一定可以直接读取。

---

## 15. Xacro 验证

进入 ROS 2 workspace：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"
```

加载 ROS 2：

```bash
source /opt/ros/humble/setup.bash
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

该验证已经成功完成。

临时生成文件位于：

```text
/tmp/so101_test.urdf
```

不应复制回 source tree，也不应提交 Git。

---

## 16. 构建验证

只构建 description package：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash

colcon build --packages-select so101_description

source install/setup.bash
```

确认 package：

```bash
ros2 pkg prefix so101_description
```

确认安装后的 Xacro：

```bash
test -f \
  install/so101_description/share/so101_description/urdf/so101.urdf.xacro \
  && echo "[OK] Installed Xacro found" \
  || echo "[MISSING] Installed Xacro not found"
```

确认安装后的 meshes：

```bash
find \
  install/so101_description/share/so101_description/meshes \
  -maxdepth 1 \
  -type f \
  -name '*.stl' \
  | sort
```

---

## 17. 仿真验证

完整仿真启动：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20

ros2 launch so101_bringup sim.launch.py
```

启动后确认：

- Robot 可以生成
- Robot 位于桌面正确位置
- 白色与黑色材质显示正确
- Top camera 图像正常
- Gripper camera 图像正常
- Joint states 正常
- Controllers 正常
- Robot 可被 position controller 控制

用户已经完成以上模型、控制和相机的人工检查。

---

## 18. Xacro 修改原则

### 不要使用 XML writer 重写整个文件

不要使用：

```python
xml.etree.ElementTree.ElementTree.write()
```

该方法曾将：

```xml
xmlns:xacro="http://www.ros.org/wiki/xacro"
```

改为：

```xml
xmlns:ns0="http://www.ros.org/wiki/xacro"
```

并破坏：

- Xacro namespace readability
- Comments
- Indentation
- XML formatting
- Git diff quality

### 推荐方式

小范围修改应优先使用：

- 精确文本替换
- 手动编辑明确片段
- 小范围 patch
- 修改后检查 `git diff`

修改后至少运行：

```bash
git diff --check
```

以及 Xacro / URDF 验证命令。

---

## 19. 不应随意修改的内容

以下配置已经验证：

- Link names
- Joint names
- Joint origins
- Joint limits
- Mesh URI
- Visual origins
- Collision origins
- Camera links
- Camera poses
- Camera FOV
- Camera resolution
- Camera topic names
- `ros2_control`
- Hardware plugin
- Command interfaces
- State interfaces
- Inline materials

修改这些内容可能影响：

- Robot geometry
- Controller startup
- Joint command mapping
- Camera observations
- Training data compatibility
- Simulation-to-real consistency
- Existing ROS 2 interfaces

任何修改都必须有明确目的，并重新执行构建、Xacro、控制和视觉验证。

---

## 20. 与其他 packages 的关系

```text
so101_description
        │
        ▼
so101_gazebo
        │
        ▼
so101_bringup
```

其中：

- `so101_description` 提供 robot model
- `so101_gazebo` 提供 world、models 和 controllers
- `so101_bringup` 将两者组合并启动完整 simulation

`so101_gazebo/package.xml` 当前声明依赖：

```text
so101_description
```

`so101_bringup/package.xml` 当前声明依赖：

```text
so101_description
so101_gazebo
```

---

## 21. Git 注意事项

不要提交：

- `ros2_ws/build/`
- `ros2_ws/install/`
- `ros2_ws/log/`
- `/tmp/so101_test.urdf`
- Python cache
- 编辑器 backup
- 临时 Xacro 副本

文档完善期间暂不 commit。最终提交前统一检查：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```
