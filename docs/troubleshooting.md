# Capstone Troubleshooting

本文档记录 SO-101 Capstone 项目中已经实际出现过，或当前正式系统中需要重点防止的问题。

范围包括：

- ROS 2 workspace
- Gazebo Fortress
- SO-101 model
- `ros2_control`
- Cameras
- Target Cube Manager
- Real robot hardware
- LeRobot
- Git 和项目文件管理

本文档优先记录已经确认的原因和处理方法。尚未完成定位的问题不应被写成确定结论。

项目根目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone
```

ROS 2 workspace：

```bash
/media/tiger/New Volume/ubuntufile/capstone/ros2_ws
```

---

## 1. 基本诊断原则

出现问题时，按以下顺序检查：

1. 当前目录是否正确
2. ROS 2 Humble 是否已 source
3. 当前 workspace 是否已 source
4. 环境变量是否一致
5. Source 是否重新 build
6. `install/` 是否仍是旧副本
7. ROS 2 nodes、topics、services 是否存在
8. Gazebo topics 和 entities 是否存在
9. 是否有重复进程
10. 是否误用了动态设备路径
11. 是否有 USB bandwidth 问题
12. 是否存在未预期的 Git 修改

避免一开始就同时修改多个 package 或系统配置。

---

# Part A — ROS 2 Workspace

## 2. `ros2` 找不到 package

### 症状

```text
Package 'so101_bringup' not found
```

或：

```bash
ros2 pkg prefix so101_bringup
```

失败。

### 常见原因

- 没有 source ROS 2 Humble
- 没有 source 当前 workspace
- Package 尚未构建
- 当前终端 source 了其他 workspace
- `install/` 不存在或不完整

### 处理

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

确认：

```bash
ros2 pkg prefix so101_description
ros2 pkg prefix so101_gazebo
ros2 pkg prefix so101_bringup
```

---

## 3. 修改 source 后运行结果没有变化

### 症状

已经修改：

```text
ros2_ws/src/
```

中的 Xacro、world、YAML、launch 或 Python script，但 Gazebo 仍表现为旧版本。

### 原因

正式运行通常读取：

```text
ros2_ws/install/
```

而不是直接读取 `src/`。

### 处理

重新构建并重新 source：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

确认安装副本：

```bash
ros2 pkg prefix so101_bringup
```

检查 source 与 install：

```bash
diff -u \
  src/so101_bringup/launch/sim.launch.py \
  install/so101_bringup/share/so101_bringup/launch/sim.launch.py
```

如果没有输出，两个文件一致。

---

## 4. 需要 clean rebuild 吗

普通修改通常只需：

```bash
colcon build
```

只有在以下情况才考虑删除生成目录：

- `install/` 明显损坏
- Package installation rules 改变后状态异常
- Build cache 导致持续错误
- Source 和 install 无法恢复一致
- Package discovery 异常

Clean rebuild：

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

但不要在未保存 source 修改时进行无关文件清理。

---

## 5. 不同终端看不到同一组 ROS 2 nodes

### 症状

一个终端可以看到仿真 nodes，另一个终端看不到。

### 原因

不同终端的：

```text
ROS_DOMAIN_ID
```

不一致，或某个终端未 source workspace。

### 处理

每个终端执行：

```bash
source /opt/ros/humble/setup.bash
source "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws/install/setup.bash"

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

检查：

```bash
echo "$ROS_DOMAIN_ID"
echo "$GZ_PARTITION"
echo "$IGN_PARTITION"
```

预期：

```text
20
gz_env_20
gz_env_20
```

---

# Part B — Gazebo Startup

## 6. 正式仿真无法启动

正式命令：

```bash
ros2 launch so101_bringup sim.launch.py
```

启动前确认：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20
```

检查 required files：

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

## 7. `Required simulation file not found`

### 原因

`sim.launch.py` 启动前会检查：

- Xacro
- Controllers
- World
- Target cube configuration

如果安装副本缺失，会抛出 `FileNotFoundError`。

### 处理

重新构建相关 packages：

```bash
colcon build \
  --packages-select \
  so101_description \
  so101_gazebo \
  so101_bringup

source install/setup.bash
```

检查 `CMakeLists.txt` 是否安装对应目录。

不要通过把文件手动复制到 `install/` 来长期绕过问题，因为下次 build 可能覆盖该修改。

---

## 8. Gazebo 找不到 table model

### 症状

World 启动，但 `lab_table` 不出现，或 console 显示 model URI 找不到。

### 检查

```bash
find \
  install/so101_gazebo/share/so101_gazebo/models/lab_table \
  -maxdepth 2 \
  -type f \
  | sort
```

应包含：

```text
model.config
model.sdf
```

### 处理

重新构建：

```bash
colcon build --packages-select so101_gazebo so101_bringup
source install/setup.bash
```

正式 launch 已设置：

```text
GZ_SIM_RESOURCE_PATH
IGN_GAZEBO_RESOURCE_PATH
```

正常情况下不需要把 model 复制到：

```text
~/.gazebo/models
```

---

## 9. Gazebo 找不到 target cube model

### 检查

```bash
find \
  install/so101_gazebo/share/so101_gazebo/models/target_cube \
  -maxdepth 2 \
  -type f \
  | sort
```

应包含：

```text
model.config
model.sdf
```

Manager 默认从安装后的 package share path 自动查找 `model.sdf`。

如果 source 中存在但 install 中不存在，应检查 `so101_gazebo/CMakeLists.txt` 和重新 build。

---

## 10. Robot 没有生成

### 检查

查看 launch output 中：

```text
spawn_so101
```

检查 Gazebo entity：

```bash
ign model --list
```

检查 `robot_state_publisher`：

```bash
ros2 node list | grep robot_state_publisher
```

### 可能原因

- Xacro 解析失败
- Robot description 未发布
- Gazebo 尚未准备好
- Resource path 错误
- Spawn process 失败
- Source 与 install 不一致

### 不要随意修改

当前 robot spawn delay：

```text
2.0 s
```

当前 spawn height：

```text
z = 0.7524
```

这些值已经验证。

---

## 11. Robot 生成后不在桌面正确位置

### 正式参数

桌面上表面：

```text
z = 0.7500
```

Robot spawn height：

```text
z = 0.7524
```

差值用于补偿 base mesh 原点约：

```text
-0.0024 m
```

### 处理原则

不要直接把 spawn height 改成 `0.7500`。

先检查：

- Source launch
- Installed launch
- Table pose
- Base mesh origin
- 是否加载了错误 model
- 是否存在旧 Gazebo instance

---

# Part C — Xacro and Robot Model

## 12. Xacro namespace 被改成 `ns0`

### 已发生问题

使用：

```python
xml.etree.ElementTree.ElementTree.write()
```

重写整份 Xacro 后：

```xml
xmlns:xacro="http://www.ros.org/wiki/xacro"
```

被改成：

```xml
xmlns:ns0="http://www.ros.org/wiki/xacro"
```

同时还会破坏：

- Comments
- Indentation
- XML readability
- Git diff

### 处理

不要使用 XML writer 重写整个 Xacro。

小范围修改应使用：

- 精确文本替换
- 小型 patch
- 手动修改明确片段

修改后运行：

```bash
git diff --check
```

以及：

```bash
xacro \
  src/so101_description/urdf/so101.urdf.xacro \
  controllers_file:="$(pwd)/src/so101_gazebo/config/controllers.yaml" \
  > /tmp/so101_test.urdf

check_urdf /tmp/so101_test.urdf
```

---

## 13. 白色机械臂零件显示为黑色

### 已发生问题

仅引用全局 URDF material：

```xml
<material name="3d_printed"/>
```

时，Gazebo Fortress 中部分 white mesh 回退为黑色。

### 原因

Gazebo Fortress 对部分 mesh visual 的全局 URDF material reference 显示不稳定。

### 已验证处理

在相关 visual 中保留 inline material：

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

不要为了减少重复 XML 删除这些 inline colors。

---

## 14. Xacro 可以解析，但 Gazebo 控制器无法启动

检查 Xacro 参数：

```xml
<xacro:arg name="controllers_file" default=""/>
```

检查 plugin：

```xml
<parameters>$(arg controllers_file)</parameters>
```

检查 launch 是否传入正确 mapping。

检查安装后的 controller YAML：

```bash
test -f \
  install/so101_gazebo/share/so101_gazebo/config/controllers.yaml
```

确认 joint names 在以下位置一致：

- Xacro joints
- `ros2_control`
- `controllers.yaml`

---

# Part D — Controllers

## 15. Controllers 未启动

等待 launch 至少：

```text
8 s
```

再检查：

```bash
ros2 control list_controllers
```

预期：

```text
joint_state_broadcaster
position_controller
```

状态应为：

```text
active
```

---

## 16. `controller_manager` 不存在

检查：

```bash
ros2 node list | grep controller_manager
```

可能原因：

- `gz_ros2_control` plugin 未加载
- Xacro 中 plugin 配置错误
- Controller YAML path 错误
- Robot entity 未生成
- Gazebo plugin path 错误

正式 launch 已设置：

```text
GZ_SIM_SYSTEM_PLUGIN_PATH
IGN_GAZEBO_SYSTEM_PLUGIN_PATH
```

并包含：

```text
/opt/ros/humble/lib
```

不要随意删除该 plugin path。

---

## 17. `joint_state_broadcaster` active，但没有 `/joint_states`

检查：

```bash
ros2 topic list | grep joint_states
```

检查 broadcaster：

```bash
ros2 control list_controllers
```

检查 hardware interfaces：

```bash
ros2 control list_hardware_interfaces
```

确认六个 joints 都具有 state interfaces：

```text
position
velocity
```

---

## 18. Position controller 无法控制某个关节

检查 `controllers.yaml` 中 joint 顺序：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

确认与 Xacro 完全一致。

不要只修改某一个文件中的 joint name。

任何 joint name 变更还会影响：

- Training observations
- Policy actions
- Dataset metadata
- Control scripts
- Evaluation code

---

# Part E — Cameras

## 19. ROS 2 camera topics 不出现

检查 bridge：

```bash
ros2 node list | grep sensor_bridge
```

检查 ROS 2 topics：

```bash
ros2 topic list | grep -E 'top_camera|gripper_camera'
```

检查 Gazebo topics：

```bash
ign topic -l | grep -E 'top_camera|gripper_camera'
```

### 判断

- Gazebo topic 存在、ROS 2 topic 不存在：重点检查 bridge
- Gazebo topic 也不存在：重点检查 Xacro sensor、robot spawn、Sensors system 和 Ogre 2

---

## 20. Camera topic 存在但没有数据

检查频率：

```bash
ros2 topic hz /top_camera/image_raw
```

```bash
ros2 topic hz /gripper_camera/image_raw
```

检查 Gazebo 是否正在运行而不是 paused。

检查 simulation clock：

```bash
ros2 topic echo /clock --once
```

检查 camera sensor update rate 是否仍为：

```text
30 Hz
```

---

## 21. Point light 导致十字形或球状伪影

### 已发生问题

在 world 中加入 point light 后，ROS camera image 中出现明显：

- 十字形 artefact
- 球状 artefact

### 已验证处理

删除 point light，只保留：

```text
main_overhead_light
```

类型：

```text
directional
```

不要只在 Gazebo GUI 中检查 lighting。

任何 light 修改都必须同时查看：

- Top camera ROS image
- Gripper camera ROS image
- Gazebo GUI

---

## 22. Segmentation topics 在 Xacro 中存在，但 ROS 2 中看不到

### 原因

Segmentation sensors 已定义，但正式 `sim.launch.py` 当前没有 bridge：

```text
/top_camera/segmentation
/gripper_camera/segmentation
```

### 当前状态

```text
Gazebo sensor definition: 已存在
Formal ROS 2 bridge: 尚未加入
```

因此该情况不一定是故障。

在正式 bridge 加入并验证前，不应把 segmentation topics 视为现有 ROS 2 接口。

---

# Part F — Target Cube Manager

## 23. `target_cube_manager` node 不存在

检查：

```bash
ros2 node list | grep target_cube_manager
```

检查安装 executable：

```bash
test -x \
  install/so101_gazebo/lib/so101_gazebo/target_cube_manager.py
```

重新构建：

```bash
colcon build --packages-select so101_gazebo so101_bringup
source install/setup.bash
```

正式 manager 已由 `sim.launch.py` 自动启动。

不要再手动启动第二个 instance。

---

## 24. Spawn service 不存在

检查：

```bash
ros2 service list | grep target_cube
```

预期：

```text
/target_cube/spawn_random
/target_cube/delete_all
```

如果不存在，先确认 manager node 是否运行。

---

## 25. Spawn request 失败

可能原因：

- 已达到 `20` 个方块上限
- `100` 次尝试内找不到满足距离的位置
- Gazebo create service 不可用
- Target cube SDF 未安装
- World name 不是 `empty`
- New entity 未及时出现在 dynamic pose stream

检查 model：

```bash
test -f \
  install/so101_gazebo/share/so101_gazebo/models/target_cube/model.sdf
```

检查 Gazebo services：

```bash
ign service -l | grep '/world/empty/create'
```

---

## 26. Cube 已生成，但 `/target_cube/states` 中没有

检查 dynamic pose topic：

```bash
ign topic -l | grep '/world/empty/dynamic_pose/info'
```

检查 manager node。

检查方块名称是否以：

```text
target_cube_
```

开头。

---

## 27. 删除后状态重新出现

### 已发生问题

Gazebo entity 删除后，dynamic pose stream 可能短时间继续发布旧 pose。

### 已验证处理

Manager 使用：

```text
2.0 s
```

deleted pose ignore window。

不要删除该逻辑。

如果问题再次出现，检查：

- Entity 是否真的删除
- Ignore window 是否仍存在
- Name matching
- Local state removal
- Pose reader delay

---

## 28. Delete-by-name 无效

确认名称完全一致：

```text
target_cube_0001
```

检查当前 states：

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

检查 delete result：

```bash
ros2 topic echo \
  /target_cube/delete_result \
  --once \
  --full-length
```

Manager 通过名称查找 entity ID，再使用 ID 删除。

---

## 29. 方块尺寸文档不一致

### 当前问题

`model.config` description 写：

```text
40 mm
```

但正式 `model.sdf` 和 `target_cube.yaml` 都是：

```text
30 mm
```

### 正确值

```text
0.030 × 0.030 × 0.030 m
```

在 metadata 修正前，以：

- `model.sdf`
- `target_cube.yaml`

为准。

---

# Part G — Real Robot and Cameras

## 30. `/dev/ttyUSB*` 编号变化

### 原因

Linux 动态分配 serial device number。

### 处理

使用：

```text
/dev/serial/by-id/
```

Follower：

```text
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00
```

Leader：

```text
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107443-if00
```

---

## 31. `/dev/video*` 编号变化

### 原因

Linux 根据 camera 枚举顺序分配 `/dev/videoN`。

### 处理

使用：

```text
/dev/v4l/by-id/
```

Gripper camera：

```text
/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
```

Top camera：

```text
/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0
```

---

## 32. 两个真实相机无法同时运行

### 已发生问题

通过 USB 扩展坞连接双相机时，出现 bandwidth 不足。

### 已验证处理

两个相机直接连接主板 USB。

优先测试：

```text
640 × 480 @ 15 FPS
```

检查：

```bash
lsusb -t
```

不要在未排除 USB bandwidth 问题前修改 LeRobot 或 OpenCV 源码。

---

## 33. 相机 device busy

先找到 persistent path 对应的动态路径：

```bash
readlink -f \
  /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
```

再检查占用进程：

```bash
fuser -v /dev/video0
```

实际 `/dev/videoN` 应以 `readlink -f` 输出为准。

关闭重复 camera viewer、LeRobot process 或测试程序。

---

## 34. 机械臂 serial port 被占用

先解析真实设备：

```bash
readlink -f \
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00
```

检查：

```bash
fuser -v /dev/ttyUSB0
```

实际 `/dev/ttyUSBN` 应以 `readlink -f` 输出为准。

不要同时运行多个 teleoperation 或 robot control process。

---

## 35. Leader 和 Follower 角色互换

固定映射：

```text
Follower: 5C4C124039
Leader:   5C82107443
```

如果行为异常，检查 command 中两个 port 是否写反。

不要根据 `/dev/ttyUSB0` 或 `/dev/ttyUSB1` 判断角色。

---

## 36. LeRobot import 使用了错误路径

激活环境：

```bash
conda activate lerobot
```

进入：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/lerobot"
```

检查：

```bash
python -c "import lerobot; print(lerobot.__file__)"
```

确认 import path 指向正式 Capstone LeRobot 目录。

此前已确认：

```text
PYTHONNOUSERSITE=1
```

该设置用于减少用户级 Python package 干扰。

---

# Part H — Git and Files

## 37. 不应提交的生成文件

当前 `.gitignore` 已排除：

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
hardware/device_ids/IDlist.txt
docs/migration/backups/
```

以及 Python 和编辑器临时文件。

不要使用 `git add -f` 强制提交这些内容，除非有明确理由。

---

## 38. Python cache 出现在 Git status

删除：

```bash
find . \
  -type d \
  -name "__pycache__" \
  -prune \
  -exec rm -rf {} +
```

检查：

```bash
git status --short
```

`.gitignore` 已忽略：

```text
__pycache__/
*.py[cod]
```

---

## 39. 编辑器 backup 出现在项目中

当前忽略：

```text
*~
*.save
*.swp
```

不要把临时 backup 当成正式 source。

重要的历史版本应通过 Git 保存，而不是复制成：

```text
file.py.backup
file.py.old
file.py.save
```

---

## 40. `references/` 的用途混淆

```text
references/SO-ARM101_original/
```

是 SO-ARM101 官方教程和原始资料的只读归档。

它不是：

- 正式 ROS 2 workspace
- 当前运行目录
- 应直接修改的 source tree

正式运行代码位于：

```text
ros2_ws/
lerobot/
src/
```

---

## 41. Migration 文件中的路径与当前路径不同

例如：

```text
docs/migration/so101_urdf_source.txt
```

记录迁移当时 destination 为：

```text
so101.urdf
```

当前正式文件已经是：

```text
so101.urdf.xacro
```

Migration files 是 provenance / history，不覆盖当前正式路径。

---

## 42. `git diff` 过大

如果只修改很小内容但 diff 很大，优先检查：

- XML writer 是否重写整个文件
- Line ending 是否改变
- Formatter 是否重排全文件
- Namespace prefix 是否变化
- Indentation 是否全局变化

不要直接提交异常大的 diff。

先恢复，再使用精确 patch。

---

## 43. 最终检查命令

文档和代码全部完成后：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```

检查：

- 是否只有预期文件
- 是否没有 cache
- 是否没有 build outputs
- 是否没有 backup
- 是否没有无关代码修改
- Markdown 是否位于正确目录

---

## 44. 当前仍需后续确认的内容

以下内容不应在未确认前写成确定解决方案：

- 完整 LeRobot production teleoperation command
- Calibration exact file path
- Segmentation ROS 2 bridge
- Training environment interface
- Dataset collection workflow
- Policy deployment workflow
- Target cube `model.config` metadata 修复
- ROS package `TODO` descriptions
- 主机层 ACPI / GPE 日志异常的最终根因

这些问题应在实际检查或独立修复后，再更新本文档。
