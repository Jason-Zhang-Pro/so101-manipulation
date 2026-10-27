# SO-101 Hardware Setup

本目录记录 Capstone 项目中真实 SO-101 机械臂、相机和固定设备路径。

这里保存的是项目实际使用并已验证的硬件映射。运行真实机器人、LeRobot teleoperation、数据采集或相机测试时，应优先使用本文档中的 persistent device paths，而不是可能随重启或插拔发生变化的 `/dev/ttyUSB*` 和 `/dev/video*` 编号。

项目根目录：

```bash
/media/tiger/New Volume/ubuntufile/capstone
```

---

## 1. 当前硬件组成

真实机器人系统包括：

- 1 台 SO-101 Leader arm
- 1 台 SO-101 Follower arm
- 1 个 Gripper / Hand-eye camera
- 1 个 Top / Fixed camera

当前已经验证：

- Leader 和 Follower 均可正常连接
- 两台机械臂均已完成 calibration
- Leader 可以控制 Follower
- 两路相机可以同时运行
- 两路相机在直接连接主板 USB 后能够稳定工作

---

## 2. 机械臂固定设备映射

### Follower arm

| Field | Value |
|---|---|
| Role | Follower |
| Serial number | `5C4C124039` |
| Persistent device path | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00` |

### Leader arm

| Field | Value |
|---|---|
| Role | Leader |
| Serial number | `5C82107443` |
| Persistent device path | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107443-if00` |

运行真实机械臂时应使用完整的 `/dev/serial/by-id/` 路径。

不要依赖：

```text
/dev/ttyUSB0
/dev/ttyUSB1
```

这些编号可能因为以下原因变化：

- 重新启动系统
- 更换 USB 接口
- 改变插入顺序
- 同时连接其他 USB serial devices
- USB 设备短暂断开并重新连接

---

## 3. 相机固定设备映射

### Gripper / Hand-eye camera

| Field | Value |
|---|---|
| Role | Gripper camera / Hand-eye camera |
| Persistent device path | `/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0` |
| Previously observed dynamic path | `/dev/video0` |

### Top / Fixed camera

| Field | Value |
|---|---|
| Role | Top camera / Fixed camera |
| Persistent device path | `/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0` |
| Previously observed dynamic path | `/dev/video2` |

运行 LeRobot 或相机测试时，应使用完整的 `/dev/v4l/by-id/` 路径。

不要只依赖：

```text
/dev/video0
/dev/video2
```

`/dev/videoN` 编号可能在重启、重新插拔或增加其他 video devices 后变化。

---

## 4. 当前相机配置

当前已经稳定使用的图像配置：

| Parameter | Value |
|---|---|
| Resolution | `640 × 480` |
| Frame rate | `15 FPS` 或 `30 FPS` |
| Camera count | 2 |

在需要优先保证双相机稳定性时，可以先使用：

```text
640 × 480 @ 15 FPS
```

在 USB 带宽、处理速度和应用需求允许时，可以使用：

```text
640 × 480 @ 30 FPS
```

修改分辨率或帧率后，应重新验证两路相机能否同时稳定运行。

---

## 5. USB 连接要求

### 已发生的问题

两路相机曾通过 USB 扩展坞连接。

该连接方式出现 USB bandwidth 不足，导致双相机无法稳定同时工作。

### 已验证方案

将两个相机直接连接到主板 USB 接口后，以下配置可以稳定运行：

- Gripper camera
- Top camera
- `640 × 480`
- `15 FPS` 或 `30 FPS`

### 使用原则

- 两个相机优先直接连接主板 USB
- 避免将两路高带宽 camera stream 接入同一个低带宽 USB hub
- 如果使用 USB hub，应确认其供电、USB generation 和上行带宽
- 相机不稳定时，先降低 FPS，再检查 USB topology
- 不要把相机故障直接归因于 LeRobot 或 OpenCV，先排除 USB bandwidth 问题

---

## 6. 一次检查四个固定设备

执行：

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

该命令还会通过 `readlink -f` 显示当前实际映射，例如：

```text
/dev/ttyUSB*
/dev/video*
```

正式配置仍应使用 persistent path，而不是 `readlink` 显示的动态路径。

---

## 7. 单独检查机械臂 serial devices

列出当前 serial-by-id：

```bash
ls -l /dev/serial/by-id/
```

只检查 Follower：

```bash
test -e \
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00 \
  && echo "[OK] Follower found" \
  || echo "[MISSING] Follower not found"
```

只检查 Leader：

```bash
test -e \
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107443-if00 \
  && echo "[OK] Leader found" \
  || echo "[MISSING] Leader not found"
```

---

## 8. 单独检查相机 devices

列出当前 V4L persistent paths：

```bash
ls -l /dev/v4l/by-id/
```

只检查 Gripper camera：

```bash
test -e \
  /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  && echo "[OK] Gripper camera found" \
  || echo "[MISSING] Gripper camera not found"
```

只检查 Top camera：

```bash
test -e \
  /dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0 \
  && echo "[OK] Top camera found" \
  || echo "[MISSING] Top camera not found"
```

---

## 9. 检查相机支持的格式

可以使用 `v4l2-ctl` 检查每个相机支持的格式、分辨率和帧率。

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

如果系统没有 `v4l2-ctl`，不要仅为查看 README 而修改环境。需要使用时再确认是否安装 `v4l-utils`。

---

## 10. 检查 USB topology

当双相机出现掉帧、无法打开或其中一路中断时，可以检查 USB topology：

```bash
lsusb -t
```

重点检查：

- 两个相机是否位于同一个 hub 下
- 当前链路是 USB 2 还是 USB 3
- 是否有其他高带宽设备共享同一控制器
- 是否通过外接 hub
- 是否出现速率明显低于预期的链路

也可以列出 USB devices：

```bash
lsusb
```

---

## 11. 常见问题

### 11.1 某个 `/dev/videoN` 发生变化

原因：

- Linux 根据设备发现顺序分配动态编号
- 插拔顺序改变
- 新增其他 camera 或 virtual video device

处理：

使用：

```text
/dev/v4l/by-id/
```

不要把 `/dev/videoN` 写死到正式配置中。

---

### 11.2 某个 `/dev/ttyUSB*` 发生变化

原因：

- USB serial device 枚举顺序改变
- Leader 和 Follower 插入顺序变化
- 其他 USB serial device 占用编号

处理：

使用：

```text
/dev/serial/by-id/
```

并根据 SO-101 controller serial number 区分 Leader 和 Follower。

---

### 11.3 两个相机不能同时打开

优先检查：

1. 是否通过 USB hub
2. 两个相机是否共享同一低带宽链路
3. 是否使用过高的 resolution 或 FPS
4. 单个相机能否分别运行
5. `lsusb -t` 中的 USB topology
6. 直接连接主板 USB 后是否恢复

当前项目曾确认该问题由 USB 扩展坞带宽不足导致。

---

### 11.4 相机存在但应用无法打开

检查：

```bash
ls -l /dev/v4l/by-id/
readlink -f /dev/v4l/by-id/<camera-name>
```

然后确认：

- 设备路径拼写正确
- 使用的是 `video-index0`
- 没有其他程序独占设备
- 请求的分辨率和 FPS 受相机支持
- 当前用户具有访问 video device 的权限

---

### 11.5 机械臂路径存在但无法连接

检查：

```bash
ls -l /dev/serial/by-id/
readlink -f /dev/serial/by-id/<device-name>
```

然后确认：

- Leader / Follower 没有互换
- serial number 与本文档一致
- 没有其他程序占用 serial port
- USB 连接稳定
- 使用了正确的 LeRobot environment
- 机械臂电源和 controller 已正常连接

---

## 12. 本机原始设备记录

本机还保留一份原始设备检查记录：

```text
hardware/device_ids/IDlist.txt
```

该文件：

- 用于保存 machine-specific device notes
- 被项目 `.gitignore` 排除
- 不作为正式共享文档
- 不应替代本 README 中的稳定硬件映射

需要更新正式设备映射时，应先实际检查设备，再修改本 README。

---

## 13. 真实系统与 Gazebo 的区别

本文档中的设备路径只适用于真实硬件。

Gazebo 仿真相机通过 ROS 2 topics 提供：

```text
/top_camera/image_raw
/top_camera/camera_info
/gripper_camera/image_raw
/gripper_camera/camera_info
```

Gazebo 仿真不使用：

```text
/dev/v4l/by-id/
/dev/serial/by-id/
```

真实机器人运行说明与 Gazebo 仿真说明必须保持分离，避免将 hardware device path 与 ROS simulation topic 混用。

---

## 14. 修改硬件映射的要求

只有在实际硬件发生变化并重新确认后，才修改本文件，例如：

- 更换 Leader controller
- 更换 Follower controller
- 更换相机
- 相机获得新的唯一 ID
- 正式 USB connection strategy 改变

修改前应运行设备检查命令，并确认：

- Serial number
- Persistent path
- Device role
- Camera role
- 双相机同时运行状态

不要根据单次出现的 `/dev/videoN` 或 `/dev/ttyUSBN` 编号更新正式映射。
