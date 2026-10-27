# Target Cube Manager

本文档说明 Capstone 项目中动态目标方块系统的用途、配置、ROS 2 接口、Gazebo 交互方式、状态管理和验证方法。

该系统已经集成到正式仿真启动流程中：

```bash
ros2 launch so101_bringup sim.launch.py
```

正常启动正式仿真时，不需要再单独运行第二个 `target_cube_manager` 实例。

---

## 1. 相关文件

Target cube model：

```text
ros2_ws/src/so101_gazebo/models/target_cube/model.config
ros2_ws/src/so101_gazebo/models/target_cube/model.sdf
```

Manager configuration：

```text
ros2_ws/src/so101_gazebo/config/target_cube.yaml
```

Manager node：

```text
ros2_ws/src/so101_gazebo/scripts/target_cube_manager.py
```

正式启动文件：

```text
ros2_ws/src/so101_bringup/launch/sim.launch.py
```

---

## 2. 系统功能

当前 Target Cube Manager 支持：

- 随机生成单个目标方块
- 为每个方块分配唯一名称
- 同时管理多个方块
- 获取 Gazebo entity ID
- 实时跟踪位置和姿态
- 发布所有方块状态
- 为每个方块动态创建独立 pose topic
- 按名称删除单个方块
- 一次删除全部方块
- 使用 entity ID 调用 Gazebo 删除服务
- 抑制删除后短时间内残留的旧 pose
- 限制方块总数
- 限制方块之间的最小中心距离

---

## 3. Target cube model

模型名称：

```text
target_cube
```

模型类型：

```text
dynamic
```

当前物理参数：

| Parameter | Value |
|---|---:|
| Size | `0.030 × 0.030 × 0.030 m` |
| Mass | `0.020 kg` |
| Model origin | Cube centre |
| Appearance | Green matte PLA |
| Static | `false` |

Inertia：

```text
ixx = 0.000003
iyy = 0.000003
izz = 0.000003
ixy = 0.0
ixz = 0.0
iyz = 0.0
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

Contact：

```text
kp        = 1000000.0
kd        = 1.0
max_vel   = 0.1
min_depth = 0.0005
```

---

## 4. Model metadata inconsistency

当前：

```text
models/target_cube/model.config
```

中的 description 仍包含：

```text
Dynamic 40 mm test cube
```

但实际模型尺寸在：

```text
model.sdf
target_cube.yaml
```

中均为：

```text
0.030 m
```

因此当前正式方块是：

```text
30 mm cube
```

文档、实验和训练配置应以 `model.sdf` 和 `target_cube.yaml` 为准。

该 `40 mm` 描述属于 metadata error，后续应作为独立的小修改修正，不应与其他代码或 README 修改混在一起。

---

## 5. Manager node

Node name：

```text
target_cube_manager
```

Source：

```text
ros2_ws/src/so101_gazebo/scripts/target_cube_manager.py
```

安装后的 executable：

```text
ros2_ws/install/so101_gazebo/lib/so101_gazebo/target_cube_manager.py
```

正式 `sim.launch.py` 会在启动约 `2.0 s` 后运行该 node。

---

## 6. 主要依赖

Python node 使用：

```text
rclpy
ament_index_python
geometry_msgs
std_msgs
std_srvs
subprocess
threading
json
```

主要用途：

| Dependency | Purpose |
|---|---|
| `rclpy` | ROS 2 node、service、publisher、subscription、timer |
| `ament_index_python` | 查找安装后的 model path |
| `geometry_msgs` | 发布 `PoseStamped` |
| `std_msgs` | 名称删除请求、JSON string 状态和结果 |
| `std_srvs` | Spawn 与 delete-all Trigger services |
| `subprocess` | 调用 `ign service` 和运行 `ign topic` |
| `threading` | 后台读取 Gazebo dynamic pose stream |
| `json` | 序列化状态与操作结果 |

---

## 7. Gazebo world 与 pose source

World name：

```text
empty
```

Dynamic pose topic：

```text
/world/empty/dynamic_pose/info
```

Manager 启动长期运行的 `ign topic` process，持续读取该 Gazebo topic。

该数据源用于：

- 发现新生成的方块
- 获取 Gazebo entity ID
- 更新方块 position
- 更新方块 orientation
- 判断方块是否仍存在
- 发布 ROS 2 状态

---

## 8. Gazebo services

创建 entity：

```text
/world/empty/create
```

删除 entity：

```text
/world/empty/remove
```

Manager 通过 `subprocess` 调用：

```text
ign service
```

创建时使用 target cube SDF 和生成 pose。

删除时使用已经记录的 Gazebo entity ID，而不是只依赖 entity name。

---

## 9. Model path

配置参数：

```text
model_sdf_path
```

当前 YAML 中没有显式设置该参数。

当参数为空时，manager 使用：

```text
get_package_share_directory("so101_gazebo")
```

自动定位安装后的：

```text
models/target_cube/model.sdf
```

因此正式运行依赖 `so101_gazebo` 已正确构建并安装 models directory。

---

## 10. Naming

Model name prefix：

```text
target_cube_
```

Digits：

```text
4
```

示例：

```text
target_cube_0001
target_cube_0002
target_cube_0003
```

每个当前管理的方块应具有唯一名称。

名称用于：

- 用户查看
- Per-cube pose topic
- Delete-by-name request
- 状态 JSON
- 日志和实验记录

实际删除操作最终使用 entity ID。

---

## 11. Spawn height calculation

当前参数：

```text
cube_size        = 0.030 m
table_surface_z  = 0.750 m
spawn_clearance  = 0.003 m
```

计算：

```text
spawn_z
= table_surface_z + cube_size / 2 + spawn_clearance
= 0.750 + 0.015 + 0.003
= 0.768 m
```

因此方块创建时：

```text
Cube centre z = 0.768 m
```

方块底部比桌面高：

```text
0.003 m
```

在重力作用下稳定落到桌面后，中心高度约：

```text
0.765 m
```

---

## 12. Spawn region

Spawn centre：

```text
world (0.0, 0.0)
```

该点是机械臂底座中心。

当前生成区域：

```text
世界 +X 方向的前半圆环
```

参数：

| Parameter | Value |
|---|---:|
| Centre X | `0.0 m` |
| Centre Y | `0.0 m` |
| Inner radius | `0.10 m` |
| Outer radius | `0.40 m` |
| Minimum angle | `-π/2` |
| Maximum angle | `+π/2` |
| Angular range | `-90° to +90°` |

World 坐标约定：

```text
+X = 向桌面中心 / 工作区前方
+Y = 机械臂左侧
+Z = 向上
```

---

## 13. Area-uniform sampling

Manager 的随机生成应在半圆环面积内均匀分布。

不能简单使用：

```text
r ~ Uniform(r_min, r_max)
```

因为这会使靠近内半径的单位面积采样密度更高。

当前设计按照面积进行均匀采样，使半圆环中不同位置的单位面积概率一致。

该分布会影响：

- 训练目标位置分布
- Evaluation coverage
- Reachability statistics
- Sim-to-real task design

不要在没有实验设计依据的情况下更改采样方法。

---

## 14. Cube separation

Minimum centre distance：

```text
0.050 m
```

Manager 在生成新位置时，会与当前已管理方块的中心位置比较。

如果候选位置与现有方块距离过近，则重新采样。

Maximum spawn attempts：

```text
100
```

如果在最大尝试次数内找不到满足条件的位置，spawn request 应返回失败，而不是强制生成重叠方块。

---

## 15. Maximum cube count

最大管理数量：

```text
20
```

当当前 managed cube count 达到上限时，不应继续生成新方块。

该限制用于避免：

- Scene 中 entity 数量无限增长
- ROS 2 publisher 数量持续增加
- Spawn region 过度拥挤
- 状态管理负担增大
- 随机采样频繁失败

---

## 16. State publish rate

当前状态发布频率：

```text
10.0 Hz
```

Manager 使用 timer 周期性发布：

- 所有方块状态
- 单个方块 pose

该频率不同于 camera frequency 和 controller update rate。

当前主要频率：

| Interface | Rate |
|---|---:|
| Controller manager | `100 Hz` |
| Cameras | `30 Hz` |
| Target cube state publication | `10 Hz` |

---

## 17. Internal cube state

每个方块状态包含：

```text
name
entity_id
x
y
z
qx
qy
qz
qw
last_seen
```

含义：

| Field | Meaning |
|---|---|
| `name` | 唯一方块名称 |
| `entity_id` | Gazebo entity ID |
| `x`, `y`, `z` | World position |
| `qx`, `qy`, `qz`, `qw` | Quaternion orientation |
| `last_seen` | 最后一次从 pose stream 观察到的时间 |

---

## 18. ROS 2 interface summary

| Interface | Type | Direction | Purpose |
|---|---|---|---|
| `/target_cube/spawn_random` | `std_srvs/srv/Trigger` | Service | 随机生成一个方块 |
| `/target_cube/delete_all` | `std_srvs/srv/Trigger` | Service | 删除全部 managed cubes |
| `/target_cube/delete_name` | `std_msgs/msg/String` | Subscription | 按名称请求删除 |
| `/target_cube/delete_result` | `std_msgs/msg/String` | Publisher | 发布删除结果 |
| `/target_cube/states` | `std_msgs/msg/String` | Publisher | 发布全部方块 JSON 状态 |
| `/target_cube/pose/target_cube_XXXX` | `geometry_msgs/msg/PoseStamped` | Publisher | 发布单个方块 pose |

---

## 19. Spawn service

Service：

```text
/target_cube/spawn_random
```

Type：

```text
std_srvs/srv/Trigger
```

调用：

```bash
ros2 service call \
  /target_cube/spawn_random \
  std_srvs/srv/Trigger \
  "{}"
```

典型流程：

1. 检查当前 cube count
2. 生成下一个唯一名称
3. 在前半圆环中采样位置
4. 检查 minimum centre distance
5. 读取 target cube SDF
6. 调用 Gazebo create service
7. 等待新 entity 出现在 dynamic pose stream
8. 获取 entity ID
9. 保存本地状态
10. 返回 Trigger response

---

## 20. All-states topic

Topic：

```text
/target_cube/states
```

Type：

```text
std_msgs/msg/String
```

查看一次：

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

消息内容使用 JSON string，表示当前 manager 追踪的方块。

状态至少用于记录：

- Name
- Entity ID
- Position
- Orientation

具体 JSON key 应以当前 node 输出为准。

---

## 21. Per-cube pose topics

Topic pattern：

```text
/target_cube/pose/target_cube_XXXX
```

Type：

```text
geometry_msgs/msg/PoseStamped
```

示例：

```bash
ros2 topic echo \
  /target_cube/pose/target_cube_0001 \
  --once
```

Manager 会在发现方块后动态创建对应 publisher。

因此在方块生成前，该 topic 可能不存在。

查看当前 pose topics：

```bash
ros2 topic list | grep '/target_cube/pose/'
```

---

## 22. Delete by name

Request topic：

```text
/target_cube/delete_name
```

Type：

```text
std_msgs/msg/String
```

调用：

```bash
ros2 topic pub \
  --once \
  /target_cube/delete_name \
  std_msgs/msg/String \
  "{data: 'target_cube_0001'}"
```

Manager 根据名称查找本地状态，再取得对应 entity ID，并调用 Gazebo remove service。

---

## 23. Delete result

Topic：

```text
/target_cube/delete_result
```

Type：

```text
std_msgs/msg/String
```

查看一次：

```bash
ros2 topic echo \
  /target_cube/delete_result \
  --once \
  --full-length
```

Result 使用 string / JSON 形式发布。

具体字段应以当前 node 输出为准，不应在其他程序中假设未记录的固定 schema。

---

## 24. Delete-all service

Service：

```text
/target_cube/delete_all
```

Type：

```text
std_srvs/srv/Trigger
```

调用：

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

Manager 会处理当前已管理的所有 cubes。

Delete-all 只针对 manager 当前追踪的 target cube entities，不应被理解为删除 world 中所有 Gazebo entities。

---

## 25. Why deletion uses entity ID

Gazebo entity name 主要用于识别和用户接口。

正式删除使用：

```text
entity_id
```

优点：

- 删除目标更明确
- 避免仅依赖名称匹配
- 与 dynamic pose stream 返回的实体一致
- 便于确认 manager 删除的是已跟踪 entity

因此状态管理必须保留：

```text
name → entity_id
```

映射。

---

## 26. Stale pose problem

实际测试中发现：

Gazebo entity 删除后，`dynamic_pose/info` 可能短时间继续出现该 entity 的旧 pose。

如果 manager 立即接受这些旧数据，可能导致：

- 已删除名称重新出现在 states 中
- 已删除方块的 pose publisher 被重新创建
- 用户误以为删除失败
- Local state 与 Gazebo scene 不一致

---

## 27. Deleted pose suppression

Manager 当前维护删除后的临时 ignore state。

Ignore duration：

```text
2.0 s
```

在该时间窗口内，已删除方块名称的旧 pose 不会重新写回本地状态。

该逻辑用于解决实际观察到的 stale pose 问题。

不要在没有重新测试 Gazebo pose stream 行为的情况下删除或缩短该机制。

---

## 28. Threading and state protection

Manager 使用后台 process / thread 持续读取 Gazebo pose stream。

同时 ROS 2 callbacks 可能执行：

- Spawn
- Delete by name
- Delete all
- Timer publication

因此本地状态需要 thread-safe access。

当前 node 使用 lock 保护 shared state。

修改 manager 时，必须避免：

- 在无锁状态下同时修改 `_states`
- Spawn 和 delete race condition
- Publisher 创建与 state 删除冲突
- Pose thread 在 shutdown 后继续运行

---

## 29. Manager automatic startup

正式 `sim.launch.py` 启动：

```text
target_cube_manager.py
```

并加载：

```text
target_cube.yaml
```

启动延时：

```text
2.0 s
```

正式启动后检查：

```bash
ros2 node list | grep target_cube_manager
```

不要同时手动运行第二个 manager，因为可能造成：

- 重复 services
- 重复 publishers
- 名称分配冲突
- 两套本地 state
- 重复 Gazebo create / remove requests

---

## 30. Configuration summary

| Parameter | Current value |
|---|---:|
| `world_name` | `empty` |
| `dynamic_pose_topic` | `/world/empty/dynamic_pose/info` |
| `model_name_prefix` | `target_cube_` |
| `name_digits` | `4` |
| `model_sdf_path` | Empty; auto-resolved |
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
| Deleted pose ignore duration | `2.0 s` |

---

## 31. Start and verify

Start simulation：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20

ros2 launch so101_bringup sim.launch.py
```

Check node：

```bash
ros2 node list | grep target_cube_manager
```

Check services：

```bash
ros2 service list | grep target_cube
```

Check topics：

```bash
ros2 topic list | grep target_cube
```

---

## 32. Functional test sequence

### Step 1: Spawn

```bash
ros2 service call \
  /target_cube/spawn_random \
  std_srvs/srv/Trigger \
  "{}"
```

### Step 2: Check states

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

### Step 3: Check pose

根据实际名称，例如：

```bash
ros2 topic echo \
  /target_cube/pose/target_cube_0001 \
  --once
```

### Step 4: Delete one

```bash
ros2 topic pub \
  --once \
  /target_cube/delete_name \
  std_msgs/msg/String \
  "{data: 'target_cube_0001'}"
```

### Step 5: Check deletion result

```bash
ros2 topic echo \
  /target_cube/delete_result \
  --once \
  --full-length
```

### Step 6: Confirm state removal

```bash
ros2 topic echo \
  /target_cube/states \
  --once \
  --full-length
```

### Step 7: Delete remaining cubes

```bash
ros2 service call \
  /target_cube/delete_all \
  std_srvs/srv/Trigger \
  "{}"
```

---

## 33. Multiple-cube test

连续调用 spawn service：

```bash
for i in 1 2 3 4 5; do
  ros2 service call \
    /target_cube/spawn_random \
    std_srvs/srv/Trigger \
    "{}"
done
```

检查：

- 名称是否唯一
- Entity ID 是否不同
- Position 是否位于前半圆环
- 中心间距是否不小于 `0.05 m`
- Per-cube pose topics 是否出现
- Delete-all 是否全部清理

---

## 34. Build

只构建 Gazebo package：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/ros2_ws"

source /opt/ros/humble/setup.bash

colcon build --packages-select so101_gazebo

source install/setup.bash
```

如果同时修改 launch：

```bash
colcon build \
  --packages-select \
  so101_gazebo \
  so101_bringup
```

---

## 35. Installed files check

Check configuration：

```bash
test -f \
  install/so101_gazebo/share/so101_gazebo/config/target_cube.yaml \
  && echo "[OK] Target cube configuration installed" \
  || echo "[MISSING] Target cube configuration"
```

Check model：

```bash
test -f \
  install/so101_gazebo/share/so101_gazebo/models/target_cube/model.sdf \
  && echo "[OK] Target cube model installed" \
  || echo "[MISSING] Target cube model"
```

Check executable：

```bash
test -x \
  install/so101_gazebo/lib/so101_gazebo/target_cube_manager.py \
  && echo "[OK] Target cube manager installed" \
  || echo "[MISSING] Target cube manager"
```

---

## 36. Common problems

### Manager node 不存在

检查：

```bash
ros2 node list | grep target_cube_manager
```

如果不存在：

- 检查 `sim.launch.py`
- 检查 `so101_gazebo` 是否已构建
- 检查 executable 是否安装
- 检查 launch output 中的 Python error

### Spawn service 不存在

检查：

```bash
ros2 service list | grep target_cube
```

确认 manager node 已启动。

### Spawn 返回失败

可能原因：

- 达到 maximum cube count
- 找不到满足最小间距的位置
- Gazebo create service 不可用
- Model SDF path 不存在
- World name 不匹配
- New entity 未在等待时间内出现

### Cube 生成但 states 中没有

检查：

```bash
ign topic -l | grep dynamic_pose
```

确认：

```text
/world/empty/dynamic_pose/info
```

存在，并检查 manager pose reader。

### 删除后方块状态重新出现

该问题应由 deleted pose suppression 处理。

如果再次发生，检查：

- Ignore duration
- Name matching
- Local state removal
- Pose stream delay
- Gazebo entity 是否实际删除

### Delete-by-name 无效

检查：

- 名称是否完全一致
- 方块是否仍在 manager states 中
- `delete_result` topic
- 对应 entity ID 是否存在

---

## 37. Interface stability

后续 training、evaluation 或 data collection code 如果依赖 manager，应优先依赖已记录接口：

```text
/target_cube/spawn_random
/target_cube/delete_all
/target_cube/delete_name
/target_cube/delete_result
/target_cube/states
/target_cube/pose/target_cube_XXXX
```

如果未来修改：

- Topic name
- Service name
- JSON schema
- Naming format
- Pose frame
- Spawn distribution

必须同步更新：

- 本文档
- Training environment
- Dataset metadata
- Evaluation scripts
- Tests
- Top-level README 中的示例

---

## 38. 不应随意修改的内容

以下行为已经验证：

- 30 mm cube
- `0.020 kg` mass
- Table height `0.750 m`
- Spawn height `0.768 m`
- Front half-annulus
- Area-uniform sampling
- `0.05 m` minimum distance
- `20` maximum cubes
- Unique four-digit naming
- Entity ID deletion
- Dynamic pose tracking
- `10 Hz` state publication
- `2.0 s` deleted pose suppression
- Automatic startup through `sim.launch.py`

任何修改都应重新执行：

1. Build
2. Single-cube spawn
3. Multiple-cube spawn
4. State publication
5. Per-cube pose
6. Delete by name
7. Delete all
8. Stale pose check
9. Manual Gazebo inspection

---

## 39. Git 注意事项

不要提交：

- `ros2_ws/build/`
- `ros2_ws/install/`
- `ros2_ws/log/`
- Python `__pycache__/`
- Runtime JSON output
- 临时 SDF
- 测试生成的方块记录

文档完善期间暂不 commit。

最终统一检查：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```
