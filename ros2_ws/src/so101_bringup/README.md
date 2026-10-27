# so101_bringup

`so101_bringup` 是 Capstone 项目中负责统一启动 SO-101 Gazebo 仿真的 ROS 2 package。

Package 路径：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws/src/so101_bringup
```

正式启动文件：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws/src/so101_bringup/launch/sim.launch.py
```

正式启动命令：

```bash
ros2 launch so101_bringup sim.launch.py
```

该 package 不保存机械臂模型或 Gazebo world，而是负责把：

- `so101_description`
- `so101_gazebo`
- Gazebo Fortress
- ROS 2 nodes
- Controllers
- Camera bridges
- Target cube manager

组合成完整的正式仿真系统。

---

## 1. Package 职责

本 package 负责：

- 查找已安装的 ROS 2 package resources
- 读取 SO-101 Xacro
- 向 Xacro 传入 controller configuration
- 启动 Gazebo Fortress
- 加载正式 world
- 启动 `robot_state_publisher`
- 在 Gazebo 中生成 SO-101
- Bridge simulation clock
- Bridge 两路 RGB camera
- Bridge 两路 RGB CameraInfo
- 启动 target cube manager
- 启动 `joint_state_broadcaster`
- 启动 `position_controller`
- 设置 Gazebo resource path
- 设置 Gazebo plugin path

本 package 不负责：

- 定义 robot links 和 joints
- 保存 mesh
- 保存 world 或 models
- 实现 target cube manager
- 真实机械臂连接
- LeRobot teleoperation
- 训练与 policy workflow

---

## 2. 目录结构

```text
so101_bringup/
├── launch/
│   └── sim.launch.py
├── CMakeLists.txt
├── package.xml
└── README.md
```

`config/`、`include/` 和 `src/` 目录由 package 结构保留，但当前正式功能集中在：

```text
launch/sim.launch.py
```

---

## 3. Package dependencies

当前 `package.xml` 声明以下 runtime dependencies：

```text
ament_index_python
launch
launch_ros
xacro
robot_state_publisher
ros_gz_sim
ros_gz_bridge
controller_manager
so101_description
so101_gazebo
```

各依赖职责：

| Dependency | Purpose |
|---|---|
| `ament_index_python` | 查找 package share directory |
| `launch` | ROS 2 launch framework |
| `launch_ros` | 启动 ROS 2 nodes |
| `xacro` | 解析 SO-101 Xacro |
| `robot_state_publisher` | 发布 robot transforms |
| `ros_gz_sim` | 启动 Gazebo 和生成 entity |
| `ros_gz_bridge` | Bridge Gazebo 与 ROS 2 topics |
| `controller_manager` | 启动 controller spawners |
| `so101_description` | 提供 Xacro 和 meshes |
| `so101_gazebo` | 提供 world、models、controllers 和 cube manager |

---

## 4. 安装内容

`CMakeLists.txt` 当前安装：

```text
launch/
config/
```

到：

```text
share/so101_bringup/
```

修改 `sim.launch.py` 后必须重新构建：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash

colcon build --packages-select so101_bringup

source install/setup.bash
```

正式运行时，ROS 2 使用安装后的 launch file：

```text
install/so101_bringup/share/so101_bringup/launch/sim.launch.py
```

如果修改 source launch file 后没有重新构建，运行时可能继续使用旧安装副本。

---

## 5. Launch argument

当前 launch file 声明：

```text
use_sim_time
```

默认值：

```text
true
```

用途：

- 让 `robot_state_publisher` 使用 Gazebo simulation clock
- 保持 ROS 2 node 时间与仿真时间一致

默认启动：

```bash
ros2 launch so101_bringup sim.launch.py
```

显式指定：

```bash
ros2 launch so101_bringup sim.launch.py use_sim_time:=true
```

当前正式仿真应保持：

```text
use_sim_time = true
```

---

## 6. 读取的正式资源

Launch file 通过 `get_package_share_directory()` 查找：

```text
so101_description
so101_gazebo
ros_gz_sim
```

随后读取：

```text
so101_description/urdf/so101.urdf.xacro
so101_gazebo/config/controllers.yaml
so101_gazebo/worlds/empty.sdf
so101_gazebo/config/target_cube.yaml
```

启动前会检查这些文件是否存在。

如果缺少任何 required file，launch 会抛出：

```text
FileNotFoundError
```

因此遇到 required simulation file not found 时，应先检查：

- 对应 package 是否成功构建
- `install/` 中是否存在安装副本
- `CMakeLists.txt` 是否安装了对应目录
- 当前终端是否 source 了正确 workspace

---

## 7. Xacro processing

Launch file 使用：

```python
xacro.process_file()
```

读取：

```text
so101.urdf.xacro
```

并传入：

```text
controllers_file
```

该 mapping 指向安装后的：

```text
so101_gazebo/config/controllers.yaml
```

处理结果转换为 XML string，并作为：

```text
robot_description
```

传给 `robot_state_publisher`。

该流程保证 robot model 与正式 controller configuration 使用同一启动入口。

---

## 8. Gazebo resource paths

Launch file 构造 Gazebo resource path，包括：

```text
so101_description 所在 package share 父目录
so101_gazebo/models
已有 GZ_SIM_RESOURCE_PATH
已有 IGN_GAZEBO_RESOURCE_PATH
```

随后同时设置：

```text
GZ_SIM_RESOURCE_PATH
IGN_GAZEBO_RESOURCE_PATH
```

用途：

- 让 Gazebo 找到 ROS 2 package resources
- 让 Gazebo 找到 `lab_table`
- 让 Gazebo 找到 `target_cube`
- 保留用户原有 Gazebo resource paths
- 兼容 Gazebo / Ignition 命名

正常情况下，不需要手动复制 models 到：

```text
~/.gazebo/models
```

---

## 9. Gazebo plugin paths

Launch file 构造 plugin path：

```text
/opt/ros/humble/lib
已有 GZ_SIM_SYSTEM_PLUGIN_PATH
```

随后同时设置：

```text
GZ_SIM_SYSTEM_PLUGIN_PATH
IGN_GAZEBO_SYSTEM_PLUGIN_PATH
```

用途：

- 让 Gazebo 找到 ROS 2 安装的 system plugins
- 保留已有 plugin path
- 兼容 Gazebo / Ignition environment variable naming

不要在不了解 plugin discovery 的情况下删除：

```text
/opt/ros/humble/lib
```

---

## 10. Gazebo startup

Launch file include：

```text
ros_gz_sim/launch/gz_sim.launch.py
```

并传入：

```text
-r "<world_file>"
```

其中：

- `-r` 表示启动后立即运行 simulation
- `<world_file>` 指向安装后的 `empty.sdf`

正式 world source：

```text
so101_gazebo/worlds/empty.sdf
```

World name：

```text
empty
```

---

## 11. robot_state_publisher

Launch 启动：

```text
package:    robot_state_publisher
executable: robot_state_publisher
name:       robot_state_publisher
```

Parameters：

```text
robot_description
use_sim_time
```

作用：

- 发布 robot kinematic transforms
- 让 ROS 2 系统获得 SO-101 link tree
- 使用 Gazebo simulation time

检查节点：

```bash
ros2 node list | grep robot_state_publisher
```

检查 robot description：

```bash
ros2 param get /robot_state_publisher robot_description
```

该输出可能很长。

---

## 12. SO-101 entity spawn

Robot spawn 使用：

```text
package:    ros_gz_sim
executable: create
name:       spawn_so101
```

Entity name：

```text
so101
```

Source：

```text
robot_description
```

Spawn position：

```text
x = 0.0
y = 0.0
z = 0.7524
```

Launch arguments：

```text
-name so101
-topic robot_description
-x 0.0
-y 0.0
-z 0.7524
```

---

## 13. Spawn height

正式生成高度：

```text
z = 0.7524
```

该高度用于补偿 base mesh 原点约：

```text
-0.0024 m
```

桌面上表面：

```text
z = 0.750 m
```

因此该 spawn height 使机械臂底座视觉上正确落在桌面上。

不要将其简化为：

```text
z = 0.7500
```

也不要在没有重新检查以下内容时修改：

- Base mesh origin
- Robot collision
- Table surface
- Robot-table contact
- Camera view
- Grasping workspace

---

## 14. Sensor bridge

Launch 启动：

```text
package:    ros_gz_bridge
executable: parameter_bridge
name:       sensor_bridge
```

当前 bridge interfaces：

```text
/clock
/top_camera/image_raw
/top_camera/camera_info
/gripper_camera/image_raw
/gripper_camera/camera_info
```

对应 message types：

| Topic | ROS 2 type | Gazebo type |
|---|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | `gz.msgs.Clock` |
| `/top_camera/image_raw` | `sensor_msgs/msg/Image` | `gz.msgs.Image` |
| `/top_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | `gz.msgs.CameraInfo` |
| `/gripper_camera/image_raw` | `sensor_msgs/msg/Image` | `gz.msgs.Image` |
| `/gripper_camera/camera_info` | `sensor_msgs/msg/CameraInfo` | `gz.msgs.CameraInfo` |

检查 bridge node：

```bash
ros2 node list | grep sensor_bridge
```

检查 ROS 2 topics：

```bash
ros2 topic list | grep -E 'clock|top_camera|gripper_camera'
```

---

## 15. Segmentation bridge status

`so101_description` Xacro 中已经定义 segmentation sensors：

```text
/top_camera/segmentation
/top_camera/segmentation/camera_info
/gripper_camera/segmentation
/gripper_camera/segmentation/camera_info
```

但当前 `sim.launch.py` 没有 bridge 这些 topics。

因此当前正式状态是：

- Segmentation sensors 已定义
- RGB sensors 已 bridge
- RGB CameraInfo 已 bridge
- Segmentation topics 尚未作为正式 ROS 2 bridge 接口启用

不要仅根据 Xacro 中的 topic 定义，声称 segmentation 已由正式 launch bridge。

---

## 16. Target cube manager startup

Launch 启动：

```text
package:    so101_gazebo
executable: target_cube_manager.py
name:       target_cube_manager
```

Parameters：

```text
so101_gazebo/config/target_cube.yaml
```

Manager 会自动启动，不需要在另一个终端再次执行同一个 node。

检查节点：

```bash
ros2 node list | grep target_cube_manager
```

检查 services：

```bash
ros2 service list | grep target_cube
```

检查 topics：

```bash
ros2 topic list | grep target_cube
```

---

## 17. joint_state_broadcaster startup

Launch 使用 controller manager spawner：

```text
package:    controller_manager
executable: spawner
controller: joint_state_broadcaster
```

Controller manager：

```text
/controller_manager
```

Timeout：

```text
60 s
```

启动后检查：

```bash
ros2 control list_controllers
```

预期：

```text
joint_state_broadcaster
```

状态应为：

```text
active
```

---

## 18. position_controller startup

Launch 使用 controller manager spawner：

```text
package:    controller_manager
executable: spawner
controller: position_controller
```

Controller manager：

```text
/controller_manager
```

Timeout：

```text
60 s
```

启动后检查：

```bash
ros2 control list_controllers
```

预期：

```text
position_controller
```

状态应为：

```text
active
```

该 controller 的 joint 与 interface 由：

```text
so101_gazebo/config/controllers.yaml
```

定义。

---

## 19. Launch timing

当前 `sim.launch.py` 使用 `TimerAction` 控制部分启动顺序：

| Time after launch | Action |
|---:|---|
| Immediately | Gazebo、`robot_state_publisher`、sensor bridge |
| `2.0 s` | Spawn SO-101 |
| `2.0 s` | Start target cube manager |
| `5.0 s` | Spawn `joint_state_broadcaster` |
| `8.0 s` | Spawn `position_controller` |

这些延时用于给 Gazebo、robot entity 和 controller manager 留出初始化时间。

当前设置已经能够启动正式系统。

不要在没有观察 startup logs 和 controller readiness 的情况下随意缩短延时。

---

## 20. 启动流程概览

```text
ros2 launch so101_bringup sim.launch.py
            │
            ├── Locate package resources
            ├── Validate required files
            ├── Process SO-101 Xacro
            ├── Set Gazebo resource paths
            ├── Set Gazebo plugin paths
            │
            ├── Start Gazebo Fortress
            ├── Start robot_state_publisher
            ├── Start sensor_bridge
            │
            ├── After 2 s: spawn SO-101
            ├── After 2 s: start target_cube_manager
            ├── After 5 s: start joint_state_broadcaster
            └── After 8 s: start position_controller
```

---

## 21. 每次启动前的环境

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

正式启动：

```bash
ros2 launch so101_bringup sim.launch.py
```

所有需要互相通信的终端应使用相同的：

```text
ROS_DOMAIN_ID
GZ_PARTITION
IGN_PARTITION
```

---

## 22. 启动后验证

### Nodes

```bash
ros2 node list
```

重点检查：

```text
/robot_state_publisher
/sensor_bridge
/target_cube_manager
/controller_manager
```

`spawn_so101` 和 controller spawner 是启动期间运行的进程，不一定长期保留为节点。

### Controllers

```bash
ros2 control list_controllers
```

预期：

```text
joint_state_broadcaster
position_controller
```

### Joint states

```bash
ros2 topic echo /joint_states --once
```

### Camera topics

```bash
ros2 topic list | grep -E 'top_camera|gripper_camera'
```

预期：

```text
/top_camera/image_raw
/top_camera/camera_info
/gripper_camera/image_raw
/gripper_camera/camera_info
```

### Camera frequency

```bash
ros2 topic hz /top_camera/image_raw
```

```bash
ros2 topic hz /gripper_camera/image_raw
```

### Target cube manager

```bash
ros2 service list | grep target_cube
ros2 topic list | grep target_cube
```

---

## 23. Target cube quick test

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

Delete all：

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

---

## 24. Build

只构建 bringup：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash

colcon build --packages-select so101_bringup

source install/setup.bash
```

如果同时修改了三个 packages：

```bash
colcon build \
  --packages-select \
  so101_description \
  so101_gazebo \
  so101_bringup
```

---

## 25. Installed launch check

确认 package：

```bash
ros2 pkg prefix so101_bringup
```

确认安装后的 launch：

```bash
test -f \
  install/so101_bringup/share/so101_bringup/launch/sim.launch.py \
  && echo "[OK] Installed launch found" \
  || echo "[MISSING] Installed launch not found"
```

查看安装后的 launch：

```bash
sed -n '1,280p' \
  install/so101_bringup/share/so101_bringup/launch/sim.launch.py
```

如果 source 与 install 内容不同，应重新构建。

---

## 26. 常见问题

### 26.1 修改 launch 后没有生效

重新构建并 source：

```bash
colcon build --packages-select so101_bringup
source install/setup.bash
```

确认 package prefix：

```bash
ros2 pkg prefix so101_bringup
```

---

### 26.2 Required simulation file not found

检查：

```bash
ros2 pkg prefix so101_description
ros2 pkg prefix so101_gazebo
```

检查安装文件：

```bash
test -f \
  install/so101_description/share/so101_description/urdf/so101.urdf.xacro

test -f \
  install/so101_gazebo/share/so101_gazebo/config/controllers.yaml

test -f \
  install/so101_gazebo/share/so101_gazebo/worlds/empty.sdf

test -f \
  install/so101_gazebo/share/so101_gazebo/config/target_cube.yaml
```

---

### 26.3 Gazebo 找不到 table 或 target cube

检查 resource path 和安装后的 models：

```bash
find \
  install/so101_gazebo/share/so101_gazebo/models \
  -maxdepth 3 \
  -type f \
  | sort
```

不要把问题临时绕过为手动复制 model，先确认 package 安装规则和 workspace source。

---

### 26.4 Robot 没有生成

检查 launch log 中：

```text
spawn_so101
```

并检查 Gazebo 是否已经正常加载 world。

可查看 Gazebo entity list：

```bash
ign model --list
```

该命令依赖当前 Gazebo / Ignition CLI 环境。

---

### 26.5 Controllers 没有 active

等待至少 `8 s` 后再检查：

```bash
ros2 control list_controllers
```

检查 controller manager：

```bash
ros2 node list | grep controller_manager
```

检查 source 与 install 的 controller YAML 是否一致。

---

### 26.6 Camera topics 不出现

检查：

```bash
ros2 node list | grep sensor_bridge
```

```bash
ign topic -l | grep -E 'top_camera|gripper_camera'
```

如果 Gazebo topics 存在但 ROS 2 topics 不存在，重点检查 bridge。

如果 Gazebo topics 也不存在，重点检查：

- Sensors system
- Xacro camera sensors
- Robot entity 是否生成
- Ogre 2 rendering

---

### 26.7 其他终端看不到节点

确保每个终端都设置：

```bash
export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

并 source：

```bash
source /opt/ros/humble/setup.bash
source "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws/install/setup.bash"
```

---

## 27. 不应随意修改的 launch 配置

以下内容已经验证：

- Required file paths
- Xacro mapping name
- World file
- Gazebo resource paths
- Gazebo plugin paths
- Entity name `so101`
- Robot spawn coordinates
- RGB bridge topics
- Target cube manager parameters
- Controller names
- Controller manager path
- Controller timeout
- Startup delays
- `use_sim_time=true`

修改这些内容可能影响：

- Model discovery
- Plugin discovery
- Robot spawn
- Camera topics
- Controller startup
- Target cube manager
- Existing verification commands
- Training and data interfaces

---

## 28. 修改原则

修改 `sim.launch.py` 时：

1. 只修改一个明确功能
2. 不顺带修改 Xacro、world 或 controller
3. 重新构建 `so101_bringup`
4. 重新 source workspace
5. 启动完整 simulation
6. 检查 nodes、topics、controllers 和 manager
7. 人工检查 robot 和 cameras
8. 最后检查 Git diff

不要为了格式化而无意义重写整个 launch file。

---

## 29. Git 注意事项

不要提交：

- `ros2_ws/build/`
- `ros2_ws/install/`
- `ros2_ws/log/`
- Python `__pycache__/`
- Launch 临时副本
- Runtime logs

文档完善期间暂不 commit。

最终统一检查：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```
