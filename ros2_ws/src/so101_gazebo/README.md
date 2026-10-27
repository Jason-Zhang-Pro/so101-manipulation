# so101_gazebo

`so101_gazebo` 是 Capstone 项目中负责 Gazebo Fortress 场景与运行资源的 ROS 2 package。

本 package 保存：

- 正式 Gazebo world
- Laboratory table model
- Dynamic target cube model
- `ros2_control` controller configuration
- Target cube manager configuration
- Target cube manager Python node

Package 路径：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws/src/so101_gazebo
```

本 package 已集成到正式启动文件：

```bash
ros2 launch so101_bringup sim.launch.py
```

---

## 1. Package 职责

本 package 负责：

- Gazebo world 资源
- Gazebo model 资源
- Controller YAML
- Dynamic object 配置
- Target cube 生成、状态读取和删除
- Python manager node 的安装
- 向其他 package 提供 Gazebo runtime resources

本 package 不负责：

- SO-101 Xacro 和 mesh
- Robot link / joint 定义
- 完整仿真启动顺序
- 真实机械臂设备连接
- LeRobot teleoperation
- 训练和 policy workflow

这些内容分别位于：

```text
so101_description
so101_bringup
hardware/
lerobot/
```

---

## 2. 目录结构

```text
so101_gazebo/
├── config/
│   ├── controllers.yaml
│   └── target_cube.yaml
├── models/
│   ├── lab_table/
│   │   ├── model.config
│   │   └── model.sdf
│   └── target_cube/
│       ├── model.config
│       └── model.sdf
├── scripts/
│   └── target_cube_manager.py
├── worlds/
│   └── empty.sdf
├── CMakeLists.txt
├── package.xml
└── README.md
```

`launch/`、`include/` 和 `src/` 目录由 package 结构保留，但当前正式功能主要位于：

```text
config/
models/
scripts/
worlds/
```

---

## 3. 安装内容

`CMakeLists.txt` 当前安装：

```text
config/
launch/
worlds/
models/
```

到：

```text
share/so101_gazebo/
```

同时安装：

```text
scripts/target_cube_manager.py
```

到：

```text
lib/so101_gazebo/
```

因此正式运行时使用的是 `install/` 下的安装副本。

修改以下内容后必须重新构建：

- `controllers.yaml`
- `target_cube.yaml`
- `empty.sdf`
- `model.config`
- `model.sdf`
- `target_cube_manager.py`
- `CMakeLists.txt`
- `package.xml`

---

## 4. Package dependencies

当前 `package.xml` 声明：

```text
so101_description
rclpy
ament_index_python
geometry_msgs
std_msgs
std_srvs
```

其中：

- `so101_description` 提供 SO-101 robot model
- `rclpy` 用于 target cube manager
- `ament_index_python` 用于查找安装后的 package share directory
- `geometry_msgs` 用于发布 `PoseStamped`
- `std_msgs` 用于名称删除请求和 JSON 状态
- `std_srvs` 用于 spawn / delete-all Trigger services

Gazebo、bridge 和 controller runtime 的完整启动依赖由 `so101_bringup` 统一声明。

---

## 5. 正式 world

World source：

```text
src/so101_gazebo/worlds/empty.sdf
```

World name：

```text
empty
```

尽管文件名是 `empty.sdf`，该 world 当前并非真正空场景。它已经包含：

- Physics system
- User commands system
- Scene broadcaster
- Contact system
- Sensors system
- Ground plane
- Laboratory table
- Directional light
- Scene background 和 ambient light

文件名保留是为了与当前正式 launch 和 manager 配置保持一致。

---

## 6. Physics

当前 physics 配置：

```xml
<physics name="1ms" type="ignored">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
</physics>
```

关键参数：

| Parameter | Value |
|---|---:|
| Maximum step size | `0.001 s` |
| Target real-time factor | `1.0` |

不要在没有性能和接触稳定性测试的情况下随意修改 physics step size。

---

## 7. Gazebo systems

当前 world 加载：

```text
Physics
UserCommands
SceneBroadcaster
Contact
Sensors
```

对应 plugin：

```xml
ignition-gazebo-physics-system
ignition-gazebo-user-commands-system
ignition-gazebo-scene-broadcaster-system
ignition-gazebo-contact-system
ignition-gazebo-sensors-system
```

Sensors system 使用：

```xml
<render_engine>ogre2</render_engine>
```

这些 systems 支持：

- Robot dynamics
- Entity creation / removal
- Scene state publication
- Collision contact
- Camera rendering

Target cube manager 依赖 UserCommands system 提供 entity create / remove services。

---

## 8. Scene settings

当前 scene：

```xml
<scene>
  <ambient>0.35 0.35 0.35 1</ambient>
  <background>0.70 0.72 0.75 1</background>
  <shadows>true</shadows>
</scene>
```

当前设计目标：

- Indoor laboratory appearance
- 中性灰背景
- 保留阴影
- 避免过暗或过曝
- 保持 top camera 和 gripper camera 画面可用

---

## 9. Directional light

当前 world 只使用一个主要方向光：

```xml
<light type="directional" name="main_overhead_light">
```

配置：

```xml
<pose>0.25 0.30 2.00 0 0 0</pose>
<diffuse>0.72 0.72 0.70 1</diffuse>
<specular>0.08 0.08 0.08 1</specular>
<direction>-0.25 -0.30 -0.92</direction>
```

其作用是模拟从前上方、略偏机械臂左侧到达工作区的主光源。

### 为什么没有 point light

以前测试 point light 时，在 ROS camera image 中出现明显十字形或球状 rendering artefact。

因此 point light 已被删除。

除非重新进行：

- Gazebo GUI 检查
- Top camera ROS image 检查
- Gripper camera ROS image 检查
- Shadow 和 exposure 检查

否则不要重新加入 point light。

---

## 10. Ground plane

World 包含 static ground plane：

```text
ground_plane
```

主要配置：

- Normal：`0 0 1`
- Size：`100 × 100 m`
- 灰色材质
- Collision enabled
- Semantic label：`1`

Ground plane 的 label plugin 用于避免 segmentation output 完全被视为无标签背景。

---

## 11. Laboratory table

模型目录：

```text
src/so101_gazebo/models/lab_table/
```

文件：

```text
model.config
model.sdf
```

模型名称：

```text
lab_table
```

类型：

```text
static
```

---

## 12. Table dimensions

桌面参数：

| Parameter | Value |
|---|---:|
| Length | `1.200 m` |
| Width | `0.600 m` |
| Thickness | `0.034 m` |
| Top surface | `z = 0.750 m` |
| Bottom surface | `z = 0.716 m` |
| Centre height | `z = 0.733 m` |

桌腿参数：

| Parameter | Value |
|---|---:|
| Cross-section | `0.050 × 0.050 m` |
| Height | `0.716 m` |
| Centre height | `z = 0.358 m` |

模型坐标约定：

```text
Model origin = 桌面中心正下方的地面点
X = 桌面长度方向
Y = 桌面宽度方向
Z = 向上
```

---

## 13. Table pose in world

正式 world 中：

```xml
<include>
  <uri>model://lab_table</uri>
  <name>lab_table</name>
  <pose>0.278 0.05 0 0 0 1.570796</pose>
</include>
```

当前 pose 已经实现：

- 机械臂靠近桌面近边
- 桌面左右可用空间符合当前实验布局
- 桌面近边位于机械臂底座中心后方约 `0.022 m`
- 机械臂 base 正确落在桌面上

不要在没有重新检查 robot reachability、camera view 和 table edge relationship 的情况下修改该 pose。

---

## 14. Table appearance

桌面使用 matte white appearance，近似白色 IKEA LAGKAPTEN 风格。

桌面 material：

```text
Ambient:  0.72 0.72 0.70 1
Diffuse:  0.92 0.92 0.90 1
Specular: 0.03 0.03 0.03 1
```

桌腿使用深色材质。

桌面 collision 使用：

```text
mu  = 0.8
mu2 = 0.8
```

这些值当前用于支持目标方块稳定落在桌面上。

---

## 15. Controllers

Controller configuration：

```text
src/so101_gazebo/config/controllers.yaml
```

当前 controller manager update rate：

```text
100 Hz
```

定义：

```text
joint_state_broadcaster
position_controller
```

`position_controller` 类型：

```text
forward_command_controller/ForwardCommandController
```

Interface：

```text
position
```

控制 joints：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

这些 joint names 必须与 `so101_description` Xacro 中的 `ros2_control` joints 完全一致。

---

## 16. Controller validation

仿真启动后：

```bash
ros2 control list_controllers
```

预期：

```text
joint_state_broadcaster
position_controller
```

检查 hardware interfaces：

```bash
ros2 control list_hardware_interfaces
```

检查 joint states：

```bash
ros2 topic echo /joint_states --once
```

Controller 已验证可以启动并控制机械臂。

不要只修改 `controllers.yaml` 而不同步检查 Xacro 中：

- Joint names
- Command interfaces
- State interfaces

---

## 17. Target cube model

模型目录：

```text
src/so101_gazebo/models/target_cube/
```

文件：

```text
model.config
model.sdf
```

模型名称：

```text
target_cube
```

类型：

```text
dynamic
```

用途：

- Grasping target
- Object tracking
- Random spawn experiments
- Future training and evaluation tasks

---

## 18. Target cube physical properties

当前正式参数：

| Parameter | Value |
|---|---:|
| Size | `0.030 × 0.030 × 0.030 m` |
| Mass | `0.020 kg` |
| Model origin | Cube centre |
| Appearance | Matte green PLA |
| Static | `false` |

Inertia：

```text
ixx = 0.000003
iyy = 0.000003
izz = 0.000003
```

Collision friction：

```text
mu  = 0.8
mu2 = 0.8
```

Bounce：

```text
restitution_coefficient = 0.0
```

Contact parameters：

```text
kp        = 1000000.0
kd        = 1.0
max_vel   = 0.1
min_depth = 0.0005
```

这些参数已经用于当前桌面落体和抓取测试，不应随意调整。

---

## 19. Cube metadata inconsistency

当前：

```text
models/target_cube/model.config
```

中的 description 仍写有：

```text
Dynamic 40 mm test cube
```

但正式模型和 manager 配置均为：

```text
0.030 m
```

即实际是：

```text
30 mm cube
```

文档与实验参数应以：

```text
model.sdf
target_cube.yaml
```

为准。

该元数据错误应在独立的小修改中修正，不要与 README 或其他代码修改混在一起。

---

## 20. Target cube configuration

配置文件：

```text
src/so101_gazebo/config/target_cube.yaml
```

Node namespace：

```text
target_cube_manager
```

World：

```text
empty
```

Dynamic pose topic：

```text
/world/empty/dynamic_pose/info
```

Model name prefix：

```text
target_cube_
```

Name digits：

```text
4
```

示例：

```text
target_cube_0001
target_cube_0002
```

---

## 21. Spawn height

参数：

```text
Cube size:          0.030 m
Table surface:      0.750 m
Spawn clearance:    0.003 m
Cube half-height:   0.015 m
```

计算：

```text
spawn_z
= table_surface_z + cube_size / 2 + spawn_clearance
= 0.750 + 0.015 + 0.003
= 0.768 m
```

方块自由下落并稳定后，中心高度约为：

```text
0.765 m
```

---

## 22. Spawn region

随机生成区域：

```text
Centre:       world (0, 0)
Inner radius: 0.10 m
Outer radius: 0.40 m
Angle min:    -90°
Angle max:    +90°
```

即以机械臂底座为中心、沿世界 `+X` 方向的前半圆环。

采样使用面积均匀方式，而不是简单对 radius 做均匀采样。

同时限制：

```text
Minimum centre distance: 0.05 m
Maximum spawn attempts:  100
Maximum cube count:      20
```

---

## 23. Target cube manager

Python node：

```text
src/so101_gazebo/scripts/target_cube_manager.py
```

安装后的 executable：

```text
lib/so101_gazebo/target_cube_manager.py
```

Node name：

```text
target_cube_manager
```

该 node 会由：

```text
so101_bringup/launch/sim.launch.py
```

自动启动。

不需要在正式仿真中手动再启动第二个 manager instance。

---

## 24. Manager implementation

Manager 主要使用：

- `rclpy`
- `subprocess`
- 长期运行的 `ign topic`
- 按需执行的 `ign service`
- Thread-safe in-memory state
- Dynamic per-cube pose publishers

创建 entity：

```text
/world/empty/create
```

删除 entity：

```text
/world/empty/remove
```

实时 pose source：

```text
/world/empty/dynamic_pose/info
```

删除时使用 Gazebo entity ID，而不是只依赖名称。

---

## 25. ROS 2 interfaces

生成一个随机方块：

```text
/target_cube/spawn_random
std_srvs/srv/Trigger
```

删除全部方块：

```text
/target_cube/delete_all
std_srvs/srv/Trigger
```

按名称请求删除：

```text
/target_cube/delete_name
std_msgs/msg/String
```

删除结果：

```text
/target_cube/delete_result
std_msgs/msg/String
```

所有方块状态：

```text
/target_cube/states
std_msgs/msg/String
```

单个方块 pose：

```text
/target_cube/pose/target_cube_XXXX
geometry_msgs/msg/PoseStamped
```

---

## 26. Spawn one cube

```bash
ros2 service call \
  /target_cube/spawn_random \
  std_srvs/srv/Trigger \
  "{}"
```

调用成功后，manager 会：

1. 检查当前管理数量
2. 在前半圆环中采样位置
3. 检查最小中心间距
4. 分配唯一名称
5. 调用 Gazebo create service
6. 等待 entity 出现在 dynamic pose stream
7. 保存 entity ID 和 pose
8. 发布状态

---

## 27. View all cube states

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

该 topic 使用 JSON string，包含当前 manager 追踪的方块信息，例如：

- Name
- Entity ID
- Position
- Orientation

实际字段应以 node 当前输出为准。

---

## 28. View one cube pose

```bash
ros2 topic echo \
  /target_cube/pose/target_cube_0001 \
  --once
```

只有 manager 已经发现并管理该方块后，对应动态 publisher 才会存在。

---

## 29. Delete one cube

```bash
ros2 topic pub \
  --once \
  /target_cube/delete_name \
  std_msgs/msg/String \
  "{data: 'target_cube_0001'}"
```

查看结果：

```bash
ros2 topic echo \
  /target_cube/delete_result \
  --once \
  --full-length
```

Manager 会根据已记录的 entity ID 调用 Gazebo remove service。

---

## 30. Delete all cubes

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

该 service 会依次删除当前 manager 追踪的所有 target cubes。

---

## 31. Deleted pose suppression

Gazebo dynamic pose stream 可能在 entity 删除后短时间继续包含旧 pose。

Manager 已实现删除后的临时 ignore window：

```text
2.0 s
```

用途：

- 防止刚删除的方块被旧 pose 再次写回本地状态
- 防止重新创建已删除方块的 pose publisher
- 保持 delete result 与 states 一致

不要在不了解 stale pose 行为的情况下删除该逻辑。

---

## 32. Manager parameter summary

| Parameter | Current value |
|---|---:|
| `world_name` | `empty` |
| `dynamic_pose_topic` | `/world/empty/dynamic_pose/info` |
| `model_name_prefix` | `target_cube_` |
| `name_digits` | `4` |
| `cube_size` | `0.030` |
| `table_surface_z` | `0.750` |
| `spawn_clearance` | `0.003` |
| `spawn_center_x` | `0.0` |
| `spawn_center_y` | `0.0` |
| `spawn_inner_radius` | `0.10` |
| `spawn_outer_radius` | `0.40` |
| `spawn_angle_min` | `-1.57079632679` |
| `spawn_angle_max` | `1.57079632679` |
| `minimum_center_distance` | `0.050` |
| `maximum_spawn_attempts` | `100` |
| `maximum_cube_count` | `20` |
| `state_publish_rate` | `10.0 Hz` |

---

## 33. Build

只构建本 package：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash

colcon build --packages-select so101_gazebo

source install/setup.bash
```

由于本 package 依赖 `so101_description`，在 description 尚未构建或同时修改时，可以使用：

```bash
colcon build \
  --packages-select \
  so101_description \
  so101_gazebo
```

---

## 34. Installed resource check

确认 package：

```bash
ros2 pkg prefix so101_gazebo
```

检查安装后的 world：

```bash
test -f \
  install/so101_gazebo/share/so101_gazebo/worlds/empty.sdf \
  && echo "[OK] World found" \
  || echo "[MISSING] World not found"
```

检查安装后的 models：

```bash
find \
  install/so101_gazebo/share/so101_gazebo/models \
  -maxdepth 3 \
  -type f \
  | sort
```

检查 manager executable：

```bash
test -x \
  install/so101_gazebo/lib/so101_gazebo/target_cube_manager.py \
  && echo "[OK] Target cube manager installed" \
  || echo "[MISSING] Target cube manager not installed"
```

---

## 35. Full simulation validation

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20

ros2 launch so101_bringup sim.launch.py
```

验证：

```bash
ros2 control list_controllers
ros2 topic list | grep -E 'top_camera|gripper_camera'
ros2 service list | grep target_cube
```

生成测试方块：

```bash
ros2 service call \
  /target_cube/spawn_random \
  std_srvs/srv/Trigger \
  "{}"
```

查看状态：

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

删除全部测试方块：

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

---

## 36. 不应随意修改的配置

以下内容已经验证：

- World name `empty`
- Physics step size
- Gazebo systems
- Ogre 2 render engine
- Table dimensions
- Table world pose
- Table surface height
- Directional light
- Controller joint names
- Controller interface
- Cube size
- Cube mass
- Cube friction and contact
- Spawn height calculation
- Spawn half-annulus
- Minimum centre distance
- Maximum cube count
- Entity ID deletion
- Deleted pose suppression
- Dynamic pose topic

修改任意一项都可能影响：

- Robot-table alignment
- Camera observations
- Controller startup
- Object stability
- Grasping behaviour
- Training data distribution
- Manager state consistency

---

## 37. Git 注意事项

不要提交：

- `ros2_ws/build/`
- `ros2_ws/install/`
- `ros2_ws/log/`
- Python `__pycache__/`
- 临时 manager 输出
- 临时 world 副本
- 运行时生成的数据

文档完善期间暂不 commit。

最终统一检查：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```
