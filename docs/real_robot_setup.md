# 真实 SO-101 系统设置

本文档记录 Capstone 项目当前真实 SO-101 Leader / Follower 双机械臂与双相机系统的已验证设置、固定设备路径、连接要求、检查方法和使用原则。

项目根目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone
```

LeRobot 目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone/lerobot
```

本文件只记录已经确认的真实硬件状态。未完整核对的命令不会被写成正式可运行命令。

---

## 1. 系统组成

当前真实系统包括：

- 1 台 SO-101 Leader arm
- 1 台 SO-101 Follower arm
- 1 个 Gripper / Hand-eye camera
- 1 个 Top / Fixed camera
- Ubuntu 22.04 主机
- LeRobot runtime environment

当前已经验证：

- Leader arm 可以连接
- Follower arm 可以连接
- 两台机械臂均已完成 calibration
- Leader 可以控制 Follower
- Gripper camera 可以读取
- Top camera 可以读取
- 两路相机可以同时稳定运行
- 双相机直接连接主板 USB 后工作正常

---

## 2. LeRobot 环境

当前 Conda environment：

```text
lerobot
```

进入环境：

```bash
conda activate lerobot
```

当前项目中 LeRobot 正式目录：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone/lerobot"
```

此前已确认：

```text
Python 3.10
PYTHONNOUSERSITE=1
```

检查 Python：

```bash
python --version
```

检查 LeRobot import：

```bash
python -c "import lerobot; print(lerobot.__file__)"
```

检查当前工作目录：

```bash
pwd
```

正式运行时应确认路径属于：

```text
/media/tiger/New Volume/ubuntufile/capstone/lerobot
```

---

## 3. Follower arm

| Field | Value |
|---|---|
| Role | Follower |
| Serial number | `5C4C124039` |
| Persistent path | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00` |

Follower 是被 Leader 或 policy 控制的机械臂。

不要依赖：

```text
/dev/ttyUSB0
/dev/ttyUSB1
```

正式配置应使用完整的 `/dev/serial/by-id/` path。

---

## 4. Leader arm

| Field | Value |
|---|---|
| Role | Leader |
| Serial number | `5C82107443` |
| Persistent path | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107443-if00` |

Leader 用于人工 teleoperation。

Leader 和 Follower 的 serial number 不应互换。

---

## 5. Gripper camera

| Field | Value |
|---|---|
| Role | Gripper / Hand-eye camera |
| Persistent path | `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0` |
| Previously observed dynamic path | `/dev/video0` |

该相机安装在夹爪附近，用于近距离目标和抓取观察。

正式配置必须使用：

```text
/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
```

不要把 `/dev/video0` 写死到项目配置中。

---

## 6. Top camera

| Field | Value |
|---|---|
| Role | Top / Fixed camera |
| Persistent path | `/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0` |
| Previously observed dynamic path | `/dev/video2` |

该相机提供固定第三人称 workspace view。

正式配置必须使用：

```text
/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0
```

不要把 `/dev/video2` 写死到项目配置中。

---

## 7. 当前相机配置

已稳定使用：

| Parameter | Value |
|---|---|
| Width | `640` |
| Height | `480` |
| FPS | `15` 或 `30` |
| Camera count | `2` |

优先稳定配置：

```text
640 × 480 @ 15 FPS
```

已测试更高帧率：

```text
640 × 480 @ 30 FPS
```

修改相机参数后必须重新验证：

- 两路相机是否同时打开
- 是否掉帧
- 是否出现 device busy
- 是否发生 USB disconnect
- LeRobot display 是否稳定
- 数据采集是否持续写入

---

## 8. USB bandwidth 问题

### 已发生问题

双相机曾通过 USB 扩展坞连接。

该连接方式造成 bandwidth 不足，表现为：

- 两路相机不能稳定同时运行
- 某一路无法打开
- Camera stream 中断
- 应用层看起来像 OpenCV 或 LeRobot 错误

### 已验证解决方案

将两个相机直接连接到主板 USB 接口后，双相机可以稳定运行。

### 使用原则

- 两个相机优先直连主板 USB
- 避免把两路 camera stream 接到同一个低带宽 hub
- 出现双相机问题时先检查 USB topology
- 必要时先降低 FPS 到 `15`
- 单独测试每个相机
- 不要在未排除 USB bandwidth 前修改 LeRobot 源码

---

## 9. 一次检查四个固定设备

```bash
while IFS='|' read -r name path; do
    if [ -e "$path" ]; then
        printf '[OK]      %-12s %s\n' "$name" "$path"
        printf '          -> %s\n' "$(readlink -f "$path")"
    else
        printf '[MISSING] %-12s %s\n' "$name" "$path"
    fi
done <<'EOF'
Follower|/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00
Leader|/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107443-if00
GripperCam|/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
TopCam|/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0
EOF
```

正常情况下应看到四个：

```text
[OK]
```

---

## 10. 检查 serial devices

```bash
ls -l /dev/serial/by-id/
```

Follower：

```bash
test -e \
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00 \
  && echo "[OK] Follower found" \
  || echo "[MISSING] Follower not found"
```

Leader：

```bash
test -e \
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107443-if00 \
  && echo "[OK] Leader found" \
  || echo "[MISSING] Leader not found"
```

---

## 11. 检查 camera devices

```bash
ls -l /dev/v4l/by-id/
```

Gripper camera：

```bash
test -e \
  /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  && echo "[OK] Gripper camera found" \
  || echo "[MISSING] Gripper camera not found"
```

Top camera：

```bash
test -e \
  /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0 \
  && echo "[OK] Top camera found" \
  || echo "[MISSING] Top camera not found"
```

---

## 12. 检查 camera formats

Gripper camera：

```bash
v4l2-ctl \
  --device=/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  --list-formats-ext
```

Top camera：

```bash
v4l2-ctl \
  --device=/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0 \
  --list-formats-ext
```

如果系统没有 `v4l2-ctl`，需要使用时再确认 `v4l-utils` 是否安装。

---

## 13. 检查 USB topology

```bash
lsusb -t
```

重点检查：

- 两个相机是否连接在同一个 hub 下
- 链路速度
- USB 2 / USB 3
- 是否与其他高带宽设备共享 controller
- 是否经过外接 hub

列出 USB devices：

```bash
lsusb
```

---

## 14. Calibration 状态

当前已经确认：

```text
Leader calibration: completed
Follower calibration: completed
```

Calibration data 的具体存储位置和文件名，应在下一次正式运行前通过当前 LeRobot 配置实际检查后再补充到本文档。

不要在未检查当前版本 LeRobot 行为时猜测 calibration file path。

---

## 15. Teleoperation 状态

当前已经确认：

```text
Leader can control Follower
```

现有 teleoperation 使用：

- Leader serial path
- Follower serial path
- Gripper camera persistent path
- Top camera persistent path
- `640 × 480`
- `15 FPS` 或 `30 FPS`
- LeRobot `teleoperate` module

但当前完整 production command 尚未在本轮文档检查中逐项从实际命令历史或配置文件重新核对。

因此本文档暂不写入可能缺少参数的完整 teleoperation command。

下一次正式运行时，应保存并补充以下内容：

- 当前 LeRobot version
- Robot type argument
- Teleoperator type argument
- Follower port argument
- Leader port argument
- Camera dictionary syntax
- Display option
- FPS option
- Calibration references
- 完整成功运行命令

---

## 16. Teleoperation 前检查清单

1. 启动 Ubuntu
2. 连接 Leader
3. 连接 Follower
4. 两个相机直连主板 USB
5. 开启机械臂电源
6. 运行四设备检查
7. 激活 `lerobot` Conda environment
8. 进入正式 LeRobot 目录
9. 确认 Python 和 import path
10. 确认没有其他进程占用 serial port 或 camera
11. 使用已验证的 teleoperation command
12. 先低速、小范围移动
13. 确认 Leader / Follower 角色没有互换

---

## 17. 安全原则

真实机械臂操作前：

- 保持 workspace 无障碍物
- 确保电缆不会被关节拉扯
- 确保夹爪附近没有手指
- 首次运行时使用小范围运动
- 避免突然大幅度移动 Leader
- 发现异常立即停止程序或断开机械臂电源
- 不要在未知 calibration 状态下高速运行
- 不要同时运行多个 teleoperation 或 robot control process

本文档不替代实验室安全规范。

---

## 18. Real system 与 Gazebo 的区别

真实系统使用：

```text
/dev/serial/by-id/
/dev/v4l/by-id/
```

Gazebo 使用：

```text
ROS 2 topics
Gazebo entities
ros2_control
```

Gazebo 相机 topics：

```text
/top_camera/image_raw
/top_camera/camera_info
/gripper_camera/image_raw
/gripper_camera/camera_info
```

不要把真实 camera device path 写入 Gazebo launch。

不要把 Gazebo ROS 2 topic 当作真实 camera path。

---

## 19. LeRobot 与 ROS 2 的边界

当前项目中：

### LeRobot

主要负责：

- Real robot connection
- Leader-Follower teleoperation
- Real camera input
- Dataset collection
- Imitation learning workflows
- Future policy training / evaluation

### ROS 2 / Gazebo

主要负责：

- Formal simulation
- Robot model
- `ros2_control`
- Simulation cameras
- World and object management
- Future simulation environment interface

两套系统属于同一 Capstone 项目，但运行入口和设备接口不同。

---

## 20. 本机设备记录

原始 machine-specific notes：

```text
hardware/device_ids/IDlist.txt
```

该文件：

- 被 `.gitignore` 排除
- 用于本机快速检查
- 不作为团队正式文档
- 不替代 `hardware/README.md`
- 不替代本文档

稳定映射应同时记录在：

```text
hardware/README.md
docs/real_robot_setup.md
```

---

## 21. 常见问题

### Serial path 不存在

检查：

```bash
ls -l /dev/serial/by-id/
```

可能原因：

- 机械臂未连接
- USB cable 问题
- Controller 未上电
- USB device 重新枚举
- 使用了错误 serial number

### Camera path 不存在

检查：

```bash
ls -l /dev/v4l/by-id/
```

可能原因：

- 相机未连接
- USB hub 或 cable 问题
- 相机枚举失败
- 路径拼写错误

### 两个相机无法同时打开

优先：

1. 取消 USB hub
2. 两个相机直连主板
3. 使用 `640 × 480 @ 15 FPS`
4. 单独测试每个相机
5. 检查 `lsusb -t`
6. 关闭占用 camera 的其他程序

### Device busy

查找占用 camera 的进程：

```bash
fuser -v /dev/video0
```

实际 dynamic device number 应通过 `readlink -f` 确认。

查找占用 serial port 的进程：

```bash
fuser -v /dev/ttyUSB0
```

不要先假设实际设备一定是 `video0` 或 `ttyUSB0`。

### Leader 和 Follower 行为异常

检查：

- Serial number 是否对应正确角色
- Port arguments 是否互换
- Calibration 是否加载
- 是否同时有第二个控制进程
- Leader motion 是否过大
- 电源和 USB connection 是否稳定

---

## 22. 需要后续补充的已知空缺

以下信息目前尚未在本轮文件检查中完整确认，因此不应猜测：

- 完整 production teleoperation command
- 当前 LeRobot commit / version
- Calibration file exact path
- Calibration file naming
- Real data collection command
- Dataset output path
- Real policy evaluation command
- Camera pixel format
- 实际稳定运行时使用 15 FPS 还是 30 FPS 作为默认值

下一次成功运行真实系统时，应把这些内容从实际命令和输出中补充到本文档。

---

## 23. 修改硬件配置的要求

只有在实际硬件变化后才修改：

- Leader serial number
- Follower serial number
- Camera ID
- Camera role
- USB connection strategy
- Resolution
- Default FPS

修改前必须：

1. 运行 persistent path 检查
2. 确认实际硬件角色
3. 验证 Leader 控制 Follower
4. 验证双相机同时运行
5. 保存成功命令
6. 更新 `hardware/README.md`
7. 更新本文档

---

## 24. Git 注意事项

不要提交：

- `hardware/device_ids/IDlist.txt`
- Camera captures
- Raw datasets
- Calibration backup files，除非明确决定纳入版本控制
- LeRobot cache
- Hugging Face cache
- 临时日志
- `__pycache__/`

文档完善期间暂不 commit。

最终统一检查：

```bash
cd "/media/tiger/New Volume/ubuntufile/capstone"

git status --short
git diff --check
git diff
```
