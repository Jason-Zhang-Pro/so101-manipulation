import os
import subprocess
import time
import threading
from typing import Optional
import xml.etree.ElementTree as ET

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import tf2_ros


CUBE_GRID_CENTERS = np.array(
    [(x, y) for x in (0.432, 0.512, 0.592) for y in (-0.274, -0.214, -0.154, -0.094, -0.034)],
    dtype=np.float32,
)

try:
    from ros_gz_interfaces.msg import Contacts

    ROS_GZ_CONTACTS_AVAILABLE = True
except Exception:
    Contacts = None
    ROS_GZ_CONTACTS_AVAILABLE = False

try:
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GzNode

    GZ_TRANSPORT_AVAILABLE = True
except Exception:
    Pose_V = None
    GzNode = None
    GZ_TRANSPORT_AVAILABLE = False


def _normalize_namespace(namespace):
    namespace = (namespace or "").strip()
    if not namespace or namespace == "/":
        return ""
    return "/" + namespace.strip("/")


class SO101ROSBridge(Node):
    def __init__(self, joint_names, env_id=0, ros_namespace=""):
        node_name = f"so101_ros_gym_bridge_{env_id}"
        namespace = _normalize_namespace(ros_namespace)
        super().__init__(node_name, namespace=namespace or None)

        self.joint_names = list(joint_names)
        self.ros_namespace = namespace
        self.latest_joint_state = None
        self.joint_state_lock = threading.Lock()
        self.has_robot_cube_contact = False
        self.has_robot_table_contact = False
        self.contact_lock = threading.Lock()
        self.last_cube_contact_time = 0.0
        self.last_table_contact_time = 0.0

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            self._topic("forward_position_controller/commands"),
            10,
        )

        self.joint_sub = self.create_subscription(
            JointState,
            self._topic("joint_states"),
            self._joint_state_callback,
            10,
        )

        if ROS_GZ_CONTACTS_AVAILABLE:
            self.cube_contact_sub = self.create_subscription(
                Contacts,
                "/world/grasp_world/model/cube/link/link/sensor/cube_contact_sensor/contact",
                self._cube_contact_callback,
                10,
            )
            self.table_contact_sub = self.create_subscription(
                Contacts,
                "/world/grasp_world/model/table/link/table_link/sensor/table_contact_sensor/contact",
                self._table_contact_callback,
                10,
            )
        else:
            print("[WARN] ros_gz_interfaces.msg.Contacts unavailable; contact bridge disabled.")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

    def _topic(self, topic_name):
        if self.ros_namespace:
            return f"{self.ros_namespace}/{topic_name.lstrip('/')}"
        return f"/{topic_name.lstrip('/')}"

    def _joint_state_callback(self, msg):
        with self.joint_state_lock:
            self.latest_joint_state = msg

    @staticmethod
    def _contact_names(contact):
        return (
            f"{contact.collision1.name} {contact.collision2.name}"
        ).lower()

    def _cube_contact_callback(self, msg):
        now = time.time()
        has_contact = any(
            "cube" in self._contact_names(contact)
            and "so101" in self._contact_names(contact)
            for contact in msg.contacts
        )

        with self.contact_lock:
            self.has_robot_cube_contact = bool(has_contact)
            if has_contact:
                self.last_cube_contact_time = now

    def _table_contact_callback(self, msg):
        now = time.time()
        active_robot_links = (
            "shoulder_link",
            "upper_arm_link",
            "lower_arm_link",
            "wrist_link",
            "gripper_link",
            "moving_jaw",
        )
        has_contact = False

        for contact in msg.contacts:
            names = self._contact_names(contact)
            if "table" not in names or "so101" not in names:
                continue
            if "base_link" in names or "cube" in names:
                continue
            if not any(link_name in names for link_name in active_robot_links):
                continue

            has_contact = True
            break

        with self.contact_lock:
            self.has_robot_table_contact = bool(has_contact)
            if has_contact:
                self.last_table_contact_time = now

    def get_contact_flags(self, stale_after=0.2):
        now = time.time()
        with self.contact_lock:
            cube_contact = (
                self.has_robot_cube_contact
                and now - self.last_cube_contact_time <= stale_after
            )
            table_contact = (
                self.has_robot_table_contact
                and now - self.last_table_contact_time <= stale_after
            )

        return cube_contact, table_contact

    def send_joint_command(self, q):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q]
        self.cmd_pub.publish(msg)

    def get_joint_positions_velocities(self):
        with self.joint_state_lock:
            msg = self.latest_joint_state

        if msg is None:
            return None, None

        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))

        positions = []
        velocities = []

        for name in self.joint_names:
            if name not in name_to_pos:
                return None, None

            positions.append(float(name_to_pos[name]))
            velocities.append(float(name_to_vel.get(name, 0.0)))

        return (
            np.array(positions, dtype=np.float32),
            np.array(velocities, dtype=np.float32),
        )

    def lookup_transform(self, target_frame, source_frame):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
        except Exception:
            return None


class GazeboPoseBridge:
    def __init__(self, world_name="grasp_world", subscribe_pose=False):
        self.world_name = world_name
        self.pose_topic = f"/world/{world_name}/pose/info"
        self.latest_model_positions = {}
        self.pose_lock = threading.Lock()
        self.node = None
        self.pose_process = None
        self.pose_thread = None
        self.last_set_pose_status = ""

        if not subscribe_pose:
            return

        if GZ_TRANSPORT_AVAILABLE:
            try:
                self.node = GzNode()
                self.node.subscribe(Pose_V, self.pose_topic, self._pose_callback)
            except Exception as exc:
                self.node = None
                print(f"[WARN] Gazebo Python pose bridge unavailable: {exc}")

        # The Python gz transport subscription may not dispatch callbacks in
        # this ROS/Gym process. Keep the CLI echo reader as the reliable source.
        self._start_ign_pose_echo()

    def _pose_callback(self, msg):
        positions = {}
        for pose in msg.pose:
            name = pose.name
            if not name:
                continue
            positions[name] = np.array(
                [pose.position.x, pose.position.y, pose.position.z],
                dtype=np.float32,
            )

        with self.pose_lock:
            self.latest_model_positions.update(positions)

    def _start_ign_pose_echo(self):
        try:
            self.pose_process = subprocess.Popen(
                ["ign", "topic", "-e", "-t", self.pose_topic],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.pose_process = None
            print(f"[WARN] Could not start ign pose echo bridge: {exc}")
            return

        self.pose_thread = threading.Thread(
            target=self._read_ign_pose_echo,
            daemon=True,
        )
        self.pose_thread.start()

    def _read_ign_pose_echo(self):
        current_name = None
        current_pos = None
        pose_depth = 0
        in_position = False
        position_depth = 0

        for raw_line in self.pose_process.stdout:
            line = raw_line.strip()

            if line == "pose {":
                current_name = None
                current_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
                pose_depth = 1
                in_position = False
                position_depth = 0
                continue

            if pose_depth <= 0:
                continue

            if line.startswith("name:"):
                current_name = line.split(":", 1)[1].strip().strip('"')
            elif line == "position {":
                in_position = True
                position_depth = 0
            elif in_position and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                if key in current_pos:
                    try:
                        current_pos[key] = float(value.strip())
                    except ValueError:
                        pass

            open_count = line.count("{")
            close_count = line.count("}")

            if in_position:
                position_depth += open_count - close_count
                if position_depth <= 0:
                    in_position = False

            pose_depth += open_count - close_count

            if pose_depth <= 0 and current_name:
                with self.pose_lock:
                    self.latest_model_positions[current_name] = np.array(
                        [current_pos["x"], current_pos["y"], current_pos["z"]],
                        dtype=np.float32,
                    )

    def get_model_position(self, model_name):
        with self.pose_lock:
            pos = self.latest_model_positions.get(model_name)

        if pos is None:
            return None

        return pos.copy()

    def set_model_position(self, model_name, position, attempts=5):
        req = (
            f'name: "{model_name}" '
            f'position {{ x: {float(position[0])} y: {float(position[1])} z: {float(position[2])} }} '
            'orientation { w: 1.0 }'
        )

        cmd = [
            "ign",
            "service",
            "-s",
            f"/world/{self.world_name}/set_pose",
            "--reqtype",
            "ignition.msgs.Pose",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            req,
        ]

        for attempt in range(max(1, attempts)):
            try:
                completed = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3.0,
                )
            except Exception as exc:
                completed = None
                self.last_set_pose_status = f"exception={type(exc).__name__}: {exc}"

            if (
                completed is not None
                and completed.returncode == 0
                and "data: true" in completed.stdout
            ):
                self.last_set_pose_status = "data:true"
                return True

            if completed is not None:
                stdout = completed.stdout.strip().replace("\n", " ")
                stderr = completed.stderr.strip().replace("\n", " ")
                self.last_set_pose_status = (
                    f"returncode={completed.returncode} stdout={stdout!r} "
                    f"stderr={stderr!r}"
                )

            if attempt + 1 < attempts:
                time.sleep(0.08)

        return False

    def close(self):
        if self.pose_process is not None:
            try:
                self.pose_process.terminate()
                self.pose_process.wait(timeout=1.0)
            except Exception:
                try:
                    self.pose_process.kill()
                except Exception:
                    pass


def _quaternion_to_matrix(q):
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)

    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)

    scale = 2.0 / norm
    xx = x * x * scale
    yy = y * y * scale
    zz = z * z * scale
    xy = x * y * scale
    xz = x * z * scale
    yz = y * z * scale
    wx = w * x * scale
    wy = w * y * scale
    wz = w * z * scale

    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float32,
    )


def _rpy_to_matrix(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)

    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c

    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _make_transform(xyz, rpy):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = _rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    transform[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return transform


def _parse_vector(raw, default):
    if raw is None:
        return np.array(default, dtype=np.float32)
    return np.array([float(item) for item in raw.split()], dtype=np.float32)


def _load_robot_description_root(path="robot_description.yaml"):
    try:
        text = open(path, "r", encoding="utf-8").read()
    except OSError:
        return None

    start = text.find("<?xml")
    if start < 0:
        start = text.find("<robot")
    if start < 0:
        return None

    xml_text = text[start:]
    lines = [line[6:] if line.startswith("      ") else line for line in xml_text.splitlines()]

    try:
        return ET.fromstring("\n".join(lines))
    except ET.ParseError:
        return None


class SO101GazeboROSEnv(gym.Env):
    """
    Gazebo + ROS2 control version of the original PyBullet SO101ReachEnv.

    Control path:
        Gym action
        -> incremental joint target
        -> /forward_position_controller/commands
        -> ros2_control
        -> Ignition Gazebo robot

    State path:
        /joint_states
        -> joint position / velocity
        -> approximate FK for first working version

    Important:
    - Start robot_state_publisher first.
    - Start ign gazebo second.
    - Load these controllers:
        joint_state_broadcaster
        forward_position_controller
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 20}

    def __init__(
        self,
        gui=False,
        max_steps=180,
        action_scale=0.030,
        use_done_action=True,
        use_reachable_sampling=True,
        control_dt=0.07,
        env_id=0,
        ros_namespace="",
        use_tf_kinematics=True,
        use_gazebo_cube_pose=True,
        subscribe_gazebo_pose=True,
        gazebo_world_name="grasp_world",
        randomize_cube=True,
        fixed_cube_position=None,
    ):
        super().__init__()

        self.gui = gui
        self.env_id = env_id
        self.ros_namespace = _normalize_namespace(ros_namespace)
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.use_done_action = use_done_action
        self.use_reachable_sampling = use_reachable_sampling
        self.control_dt = control_dt
        self.use_tf_kinematics = use_tf_kinematics
        self.use_gazebo_cube_pose = use_gazebo_cube_pose
        self.subscribe_gazebo_pose = subscribe_gazebo_pose
        self.gazebo_world_name = gazebo_world_name
        self.randomize_cube = randomize_cube
        self.fixed_cube_position = (
            np.array(fixed_cube_position, dtype=np.float32)
            if fixed_cube_position is not None
            else np.array([0.512, -0.154, 0.7655], dtype=np.float32)
        )

        # Must match so101_controllers.yaml order.
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        self.joint_lower_limits = np.array(
            [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533],
            dtype=np.float32,
        )
        self.joint_upper_limits = np.array(
            [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
            dtype=np.float32,
        )

        self.home_pose = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        # Approximate geometry for first Gazebo ROS version.
        # This keeps observation/reward shape close to your PyBullet env.
        self.base_pos = np.array([0.2, -0.15, 0.75], dtype=np.float32)
        self.L1 = 0.11
        self.L2 = 0.12
        self.L3 = 0.09

        self.obj_offset_local = np.array([-0.025, 0.0, -0.008], dtype=np.float32)

        # Reward parameters copied from PyBullet env.
        self.success_threshold = 0.02
        self.success_hold_required = 3
        self.success_min_gripper_open_norm = 0.45

        self.progress_reward_weight = 70.0
        self.distance_penalty_weight = 3.5
        self.no_improve_penalty_weight = 6.0
        self.step_penalty = 0.035
        self.success_bonus = 45.0
        self.timeout_penalty = -5.0

        self.light_contact_penalty = 0.20
        self.medium_contact_penalty = 1.00
        self.bad_contact_penalty = 4.00
        self.table_contact_penalty = 4.0

        self.cube_move_threshold = 0.010
        self.cube_move_fail_threshold = 0.040
        self.anti_push_weight = 120.0

        self.safe_approach_distance = 0.060
        self.safe_approach_weight = 1.5

        self.near_collision_distance = 0.020
        self.near_collision_penalty_weight = 1.0

        self.radial_error_weight = 0.8
        self.radial_progress_weight = 3.5

        self.gripper_too_closed_penalty = 0.35
        self.gripper_over_open_penalty = 0.03

        self.done_success_bonus = 20.0
        self.done_failure_penalty = -35.0
        self.done_action_threshold = 0.7

        self.action_penalty_weight = 0.003
        self.action_smoothing = 0.20

        self.step_count = 0
        self.success_hold_steps = 0

        self.target_joint_positions = self.home_pose.copy()
        self.last_joint_positions = self.home_pose.copy()
        self.current_joint_velocities = np.zeros_like(self.home_pose)
        self.prev_action = None

        self.prev_distance = None
        self.prev_radial_error = None
        self.prev_gripper_open_norm = None

        # First ROS version: cube state is managed internally.
        # Later you can replace this with Gazebo pose subscription/service.
        self.cube_pos = None
        self.cube_initial_pos = None
        self._warned_tf_fallback = False
        self._warned_cube_fallback = False
        self._fk_joint_chain = []
        self._build_urdf_fk_chain()

        self.np_random = np.random.default_rng()

        self._start_ros()
        self.gz_pose_bridge = GazeboPoseBridge(
            self.gazebo_world_name,
            subscribe_pose=self.subscribe_gazebo_pose,
        )

        initial_obs, _ = self.reset()

        base_action_dim = len(self.joint_names)
        action_dim = base_action_dim + (1 if self.use_done_action else 0)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_obs.shape,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # ROS setup
    # ---------------------------------------------------------

    def _start_ros(self):
        if not rclpy.ok():
            rclpy.init(args=None)

        self.bridge = SO101ROSBridge(
            self.joint_names,
            env_id=self.env_id,
            ros_namespace=self.ros_namespace,
        )

        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.bridge)

        self.spin_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.spin_thread.start()

        self._wait_for_joint_states(timeout=5.0)

    def _wait_for_joint_states(self, timeout=5.0):
        start = time.time()

        while time.time() - start < timeout:
            q, dq = self.bridge.get_joint_positions_velocities()
            if q is not None:
                return True
            time.sleep(0.05)

        ns_prefix = self.ros_namespace or ""
        topic = f"{ns_prefix}/joint_states" if ns_prefix else "/joint_states"
        print(f"[WARN] No {topic} received. Is Gazebo and joint_state_broadcaster running?")
        return False

    def close(self):
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown()
            if hasattr(self, "spin_thread") and self.spin_thread.is_alive():
                self.spin_thread.join(timeout=1.0)
            if hasattr(self, "bridge"):
                self.bridge.destroy_node()
            if hasattr(self, "gz_pose_bridge"):
                self.gz_pose_bridge.close()
        except Exception:
            pass

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.success_hold_steps = 0

        self.target_joint_positions = self.home_pose.copy()
        self.last_joint_positions = self.target_joint_positions.copy()
        self.current_joint_velocities = np.zeros_like(self.home_pose)

        # Send home pose several times so controller settles.
        for _ in range(20):
            self.bridge.send_joint_command(self.target_joint_positions)
            time.sleep(0.03)

        # Wait a little and read real joint state.
        time.sleep(0.3)
        q, dq = self.bridge.get_joint_positions_velocities()
        if q is not None:
            self.target_joint_positions = q.copy()
            self.last_joint_positions = q.copy()
            self.current_joint_velocities = dq.copy()

        self.cube_pos = np.array(self._sample_cube_position(), dtype=np.float32)
        if self.use_gazebo_cube_pose and self.gz_pose_bridge is not None:
            desired_cube_pos = self.cube_pos.copy()
            cube_set = False
            verified_cube_pos = None
            last_diff = None

            for set_attempt in range(3):
                cube_set = self.gz_pose_bridge.set_model_position(
                    "cube",
                    desired_cube_pos,
                    attempts=1,
                ) or cube_set

                for _ in range(60):
                    time.sleep(0.05)
                    real_cube_pos = self.gz_pose_bridge.get_model_position("cube")
                    if real_cube_pos is None:
                        continue

                    verified_cube_pos = real_cube_pos
                    last_diff = float(np.linalg.norm(real_cube_pos - desired_cube_pos))
                    if last_diff <= 0.005:
                        break

                if last_diff is not None and last_diff <= 0.005:
                    break

            if verified_cube_pos is not None:
                self.cube_pos = verified_cube_pos.astype(np.float32)

            pose_verified = (
                verified_cube_pos is not None
                and np.linalg.norm(self.cube_pos - desired_cube_pos) <= 0.005
            )

            if not pose_verified:
                print(
                    "[WARN] Gazebo cube pose was not verified at sampled position; "
                    "using latest Gazebo cube pose for Gym state. "
                    f"desired={desired_cube_pos.tolist()} "
                    f"latest={None if verified_cube_pos is None else verified_cube_pos.tolist()} "
                    f"diff={last_diff} "
                    f"set_status={self.gz_pose_bridge.last_set_pose_status}",
                    flush=True,
                )

        self.cube_initial_pos = self.cube_pos.copy()

        self.prev_distance = self._compute_distance_to_cube()
        self.prev_radial_error, _ = self._get_axis_alignment_metrics()
        self.prev_gripper_open_norm = self._get_gripper_open_norm()

        action_dim = len(self.joint_names) + (1 if self.use_done_action else 0)
        self.prev_action = np.zeros(action_dim, dtype=np.float32)

        return self._get_observation(), {}

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------

    def _build_urdf_fk_chain(self):
        root = _load_robot_description_root()
        if root is None:
            print("[WARN] Could not parse robot_description.yaml; reachable cube sampling will use fallback range.")
            return

        joints_by_parent = {}

        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue

            origin = joint.find("origin")
            axis = joint.find("axis")
            name = joint.attrib.get("name", "")

            joint_data = {
                "name": name,
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib.get("link", ""),
                "child": child.attrib.get("link", ""),
                "xyz": _parse_vector(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0]),
                "rpy": _parse_vector(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0]),
                "axis": _parse_vector(axis.attrib.get("xyz") if axis is not None else None, [0, 0, 1]),
            }

            joints_by_parent.setdefault(joint_data["parent"], []).append(joint_data)

        start_link = "base_link"
        target_link = "gripper_frame_link"

        def find_chain(link, chain, visited):
            if link == target_link:
                return chain
            if link in visited:
                return None

            visited.add(link)

            for joint in joints_by_parent.get(link, []):
                result = find_chain(joint["child"], chain + [joint], visited)
                if result is not None:
                    return result

            return None

        chain = find_chain(start_link, [], set())

        if chain is None:
            print("[WARN] Could not build base_link -> gripper_frame_link FK chain.")
            return

        self._fk_joint_chain = chain

    def _compute_compare_point_for_joint_positions(self, q):
        if not self._fk_joint_chain:
            return None

        q_by_name = {
            joint_name: float(joint_position)
            for joint_name, joint_position in zip(self.joint_names, q)
        }

        transform = np.eye(4, dtype=np.float32)

        for joint in self._fk_joint_chain:
            transform = transform @ _make_transform(joint["xyz"], joint["rpy"])

            if joint["type"] in ("revolute", "continuous"):
                joint_angle = q_by_name.get(joint["name"], 0.0)
                joint_rotation = np.eye(4, dtype=np.float32)
                joint_rotation[:3, :3] = _axis_angle_to_matrix(joint["axis"], joint_angle)
                transform = transform @ joint_rotation

        ee_pos = self.base_pos + transform[:3, 3]
        ee_rot = transform[:3, :3]
        return ee_pos + ee_rot @ self.obj_offset_local

    def _sample_cube_position(self):
        if not self.randomize_cube:
            return self.fixed_cube_position.tolist()

        cube_xy = CUBE_GRID_CENTERS[self.np_random.integers(len(CUBE_GRID_CENTERS))]
        return [float(cube_xy[0]), float(cube_xy[1]), 0.7655]

    # ---------------------------------------------------------
    # State helpers
    # ---------------------------------------------------------

    def get_joint_positions(self):
        q, _ = self.bridge.get_joint_positions_velocities()
        if q is None:
            return self.target_joint_positions.astype(np.float32)
        return q.astype(np.float32)

    def get_joint_velocities(self):
        _, dq = self.bridge.get_joint_positions_velocities()
        if dq is None:
            return self.current_joint_velocities.astype(np.float32)
        return dq.astype(np.float32)

    def _approx_fk(self):
        q = self.get_joint_positions()

        pan = float(q[0])
        shoulder = float(q[1])
        elbow = float(q[2])
        wrist = float(q[3])

        a1 = shoulder
        a2 = shoulder + elbow
        a3 = shoulder + elbow + wrist

        r = (
            self.L1 * np.cos(a1)
            + self.L2 * np.cos(a2)
            + self.L3 * np.cos(a3)
        )
        z = (
            self.L1 * np.sin(a1)
            + self.L2 * np.sin(a2)
            + self.L3 * np.sin(a3)
        )

        x = self.base_pos[0] + r * np.cos(pan)
        y = self.base_pos[1] + r * np.sin(pan)
        z = self.base_pos[2] + z

        return np.array([x, y, z], dtype=np.float32)

    def _get_link_pose_from_tf(self, frame_name):
        if not self.use_tf_kinematics:
            return None, None

        transform = self.bridge.lookup_transform("base_link", frame_name)
        if transform is None:
            if not self._warned_tf_fallback:
                print("[WARN] TF lookup failed; using approximate FK fallback.")
                self._warned_tf_fallback = True
            return None, None

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        pos = self.base_pos + np.array(
            [translation.x, translation.y, translation.z],
            dtype=np.float32,
        )
        rot = _quaternion_to_matrix(rotation)

        return pos, rot

    def get_compare_point_position(self):
        ee_pos, ee_rot = self._get_link_pose_from_tf("gripper_frame_link")
        if ee_pos is not None:
            return ee_pos + ee_rot @ self.obj_offset_local

        return self._approx_fk() + self.obj_offset_local

    def get_grasp_center_position(self):
        fixed_jaw_pos, _ = self._get_link_pose_from_tf("gripper_link")
        moving_jaw_pos, _ = self._get_link_pose_from_tf("moving_jaw_so101_v1_link")

        if fixed_jaw_pos is not None and moving_jaw_pos is not None:
            return 0.5 * (fixed_jaw_pos + moving_jaw_pos)

        return self._approx_fk()

    def _get_cube_position(self):
        if self.use_gazebo_cube_pose and self.gz_pose_bridge is not None:
            cube_pos = self.gz_pose_bridge.get_model_position("cube")
            if cube_pos is not None:
                self.cube_pos = cube_pos
                return cube_pos.astype(np.float32)

            if not self._warned_cube_fallback:
                print("[WARN] Gazebo cube pose not received; using internal cube position fallback.")
                self._warned_cube_fallback = True

        return self.cube_pos.astype(np.float32)

    def _get_cube_move_distance(self):
        if self.cube_initial_pos is None:
            return 0.0
        return float(np.linalg.norm(self.cube_pos[:2] - self.cube_initial_pos[:2]))

    def _get_observation(self):
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()
        compare_pos = self.get_compare_point_position()
        grasp_pos = self.get_grasp_center_position()
        cube_pos = self._get_cube_position()
        relative_pos = cube_pos - compare_pos

        radial_error, axial_offset = self._get_axis_alignment_metrics()
        gripper_open_norm = self._get_gripper_open_norm()
        cube_move = self._get_cube_move_distance()

        has_cube_contact = float(self._has_robot_cube_contact())
        has_table_contact = float(self._has_robot_table_contact())

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                compare_pos,
                grasp_pos,
                cube_pos,
                relative_pos,
                np.array(
                    [
                        radial_error,
                        axial_offset,
                        gripper_open_norm,
                        cube_move,
                        has_cube_contact,
                        has_table_contact,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        return obs

    def _compute_distance_to_cube(self):
        compare_pos = self.get_compare_point_position()
        cube_pos = self._get_cube_position()
        return float(np.linalg.norm(compare_pos - cube_pos))

    def _get_relative_position_to_cube(self):
        compare_pos = self.get_compare_point_position()
        cube_pos = self._get_cube_position()
        return cube_pos - compare_pos

    def _get_gripper_z_axis(self):
        _, ee_rot = self._get_link_pose_from_tf("gripper_frame_link")
        if ee_rot is not None:
            axis = ee_rot[:, 2]
            norm = float(np.linalg.norm(axis))
            if norm > 1e-8:
                return axis / norm

        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _get_axis_alignment_metrics(self):
        compare_pos = self.get_compare_point_position()
        cube_pos = self._get_cube_position()
        z_axis = self._get_gripper_z_axis()

        vector_to_cube = cube_pos - compare_pos
        axial_offset = float(np.dot(vector_to_cube, z_axis))
        perpendicular = vector_to_cube - axial_offset * z_axis
        radial_error = float(np.linalg.norm(perpendicular))

        return radial_error, axial_offset

    def _get_gripper_open_norm(self):
        idx = self.joint_names.index("gripper")
        pos = self.get_joint_positions()[idx]
        low = float(self.joint_lower_limits[idx])
        high = float(self.joint_upper_limits[idx])

        if abs(high - low) < 1e-8:
            return 1.0

        return float(np.clip((pos - low) / (high - low), 0.0, 1.0))

    # ---------------------------------------------------------
    # Contact helpers
    # ---------------------------------------------------------

    def _has_robot_cube_contact(self):
        cube_contact, _ = self.bridge.get_contact_flags()
        return cube_contact

    def _has_robot_table_contact(self):
        _, table_contact = self.bridge.get_contact_flags()
        return table_contact

    def _get_min_distance_robot_cube(self):
        if self._has_robot_cube_contact():
            return 0.0
        return self._compute_distance_to_cube()

    # ---------------------------------------------------------
    # Scripted post-success grasp helper
    # ---------------------------------------------------------

    def execute_scripted_grasp(
        self,
        close_position=0.0,
        close_steps=25,
        lift_delta=0.18,
        lift_steps=25,
        command_dt=0.04,
    ):
        """
        Close the gripper after the learned reach policy succeeds.

        This is intentionally not used during training. It is a deterministic
        post-success routine for play/demo runs, so PPO only learns reaching.
        """
        q = self.get_joint_positions().copy()
        target = q.copy()

        gripper_idx = self.joint_names.index("gripper")
        shoulder_idx = self.joint_names.index("shoulder_lift")

        gripper_close = float(
            np.clip(
                close_position,
                self.joint_lower_limits[gripper_idx],
                self.joint_upper_limits[gripper_idx],
            )
        )

        start_gripper = float(q[gripper_idx])

        for alpha in np.linspace(0.0, 1.0, close_steps):
            target[gripper_idx] = (1.0 - alpha) * start_gripper + alpha * gripper_close
            self.bridge.send_joint_command(target)
            time.sleep(command_dt)

        q = self.get_joint_positions().copy()
        target = q.copy()
        start_shoulder = float(q[shoulder_idx])
        shoulder_lift = float(
            np.clip(
                start_shoulder + lift_delta,
                self.joint_lower_limits[shoulder_idx],
                self.joint_upper_limits[shoulder_idx],
            )
        )

        for alpha in np.linspace(0.0, 1.0, lift_steps):
            target[shoulder_idx] = (1.0 - alpha) * start_shoulder + alpha * shoulder_lift
            target[gripper_idx] = gripper_close
            self.bridge.send_joint_command(target)
            time.sleep(command_dt)

        self.target_joint_positions = target.copy()

    # ---------------------------------------------------------
    # Reward
    # ---------------------------------------------------------

    def _is_success_ready(self, distance, gripper_open_norm):
        cube_move = self._get_cube_move_distance()

        return (
            distance <= self.success_threshold
            and gripper_open_norm >= self.success_min_gripper_open_norm
            and cube_move <= self.cube_move_threshold
        )

    def _compute_reward(self, action_without_done):
        relative_pos = self._get_relative_position_to_cube()
        distance = float(np.linalg.norm(relative_pos))

        previous_distance = distance if self.prev_distance is None else self.prev_distance
        progress = previous_distance - distance

        radial_error, axial_offset = self._get_axis_alignment_metrics()
        previous_radial_error = radial_error if self.prev_radial_error is None else self.prev_radial_error
        radial_progress = previous_radial_error - radial_error

        gripper_open_norm = self._get_gripper_open_norm()
        cube_move = self._get_cube_move_distance()

        has_cube_contact = self._has_robot_cube_contact()
        has_table_contact = self._has_robot_table_contact()
        min_robot_cube_distance = self._get_min_distance_robot_cube()

        reward = 0.0
        rc = {}

        progress_reward = self.progress_reward_weight * progress
        reward += progress_reward
        rc["progress_reward"] = float(progress_reward)

        distance_penalty = -self.distance_penalty_weight * distance
        reward += distance_penalty
        rc["distance_penalty"] = float(distance_penalty)

        no_improve_penalty = 0.0
        if progress <= 0.0:
            no_improve_penalty = -self.no_improve_penalty_weight * (distance ** 2)
            reward += no_improve_penalty
        rc["no_improve_penalty"] = float(no_improve_penalty)

        safe_approach_reward = 0.0
        if not has_cube_contact and distance < self.safe_approach_distance:
            safe_approach_reward = self.safe_approach_weight * (
                self.safe_approach_distance - distance
            )
            reward += safe_approach_reward
        rc["safe_approach_reward"] = float(safe_approach_reward)

        radial_progress_reward = self.radial_progress_weight * radial_progress
        radial_error_penalty = -self.radial_error_weight * radial_error
        reward += radial_progress_reward + radial_error_penalty

        rc["radial_progress_reward"] = float(radial_progress_reward)
        rc["radial_error_penalty"] = float(radial_error_penalty)
        rc["radial_error"] = float(radial_error)
        rc["axial_offset"] = float(axial_offset)

        near_collision_penalty = 0.0
        if min_robot_cube_distance < self.near_collision_distance:
            near_collision_penalty = -self.near_collision_penalty_weight * (
                self.near_collision_distance - min_robot_cube_distance
            )
            reward += near_collision_penalty
        rc["near_collision_penalty"] = float(near_collision_penalty)
        rc["min_robot_cube_distance"] = float(min_robot_cube_distance)

        contact_penalty = 0.0
        if has_cube_contact:
            if cube_move < 0.005 and distance < 0.035:
                contact_penalty = -self.light_contact_penalty
            elif cube_move < 0.015:
                contact_penalty = -self.medium_contact_penalty
            else:
                contact_penalty = -self.bad_contact_penalty
            reward += contact_penalty
        rc["contact_penalty"] = float(contact_penalty)

        table_penalty = 0.0
        if has_table_contact:
            table_penalty = -self.table_contact_penalty
            reward += table_penalty
        rc["table_penalty"] = float(table_penalty)

        anti_push_penalty = 0.0
        if cube_move > self.cube_move_threshold:
            anti_push_penalty = -self.anti_push_weight * (
                cube_move - self.cube_move_threshold
            )
            reward += anti_push_penalty
        rc["anti_push_penalty"] = float(anti_push_penalty)
        rc["cube_move"] = float(cube_move)

        gripper_reward = 0.0
        if gripper_open_norm < self.success_min_gripper_open_norm:
            gripper_reward -= self.gripper_too_closed_penalty * (
                self.success_min_gripper_open_norm - gripper_open_norm
            )

        if gripper_open_norm > 0.95:

            gripper_reward -= self.gripper_over_open_penalty

        reward += gripper_reward
        rc["gripper_reward"] = float(gripper_reward)
        rc["gripper_open_norm"] = float(gripper_open_norm)

        action_penalty = -self.action_penalty_weight * float(np.linalg.norm(action_without_done))
        reward += action_penalty
        rc["action_penalty"] = float(action_penalty)

        reward -= self.step_penalty
        rc["step_penalty"] = float(-self.step_penalty)

        rc["has_cube_contact"] = float(has_cube_contact)
        rc["has_table_contact"] = float(has_table_contact)

        success_ready = self._is_success_ready(distance, gripper_open_norm)
        if success_ready:
            self.success_hold_steps += 1
        else:
            self.success_hold_steps = 0

        success = self.success_hold_steps >= self.success_hold_required

        success_bonus = 0.0
        if success:
            success_bonus = self.success_bonus
            reward += success_bonus
        rc["success_bonus"] = float(success_bonus)
        rc["success_ready"] = float(success_ready)

        early_fail = cube_move > self.cube_move_fail_threshold

        self.prev_distance = distance
        self.prev_radial_error = radial_error
        self.prev_gripper_open_norm = gripper_open_norm

        return reward, distance, progress, success, early_fail, rc

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        expected_dim = len(self.joint_names) + (1 if self.use_done_action else 0)
        if len(action) != expected_dim:
            raise ValueError(f"Action must have length {expected_dim}, got {len(action)}")

        if self.use_done_action:
            done_declaration = float(action[-1])
        else:
            done_declaration = -1.0

        smoothed_full_action = (
            self.action_smoothing * self.prev_action
            + (1.0 - self.action_smoothing) * action
        )

        if self.use_done_action:
            smoothed_joint_action = smoothed_full_action[:-1]
            done_declaration = float(smoothed_full_action[-1])
        else:
            smoothed_joint_action = smoothed_full_action

        distance_now = self._compute_distance_to_cube()
        gripper_open_norm = self._get_gripper_open_norm()

        if self.use_done_action and done_declaration >= self.done_action_threshold:
            self.step_count += 1

            success_ready = self._is_success_ready(distance_now, gripper_open_norm)
            success = bool(success_ready)
            reward = self.done_success_bonus if success else self.done_failure_penalty
            obs = self._get_observation()

            info = {
                "distance_to_cube": distance_now,
                "progress": 0.0,
                "success": success,
                "success_hold_steps": self.success_hold_steps,
                "step_count": self.step_count,
                "target_joint_positions": self.target_joint_positions.copy(),
                "has_cube_contact": self._has_robot_cube_contact(),
                "has_table_contact": self._has_robot_table_contact(),
                "cube_move": self._get_cube_move_distance(),
                "gripper_open_norm": gripper_open_norm,
                "compare_point": self.get_compare_point_position(),
                "cube_position": self._get_cube_position(),
                "early_fail": False,
                "reward_components": {
                    "policy_declared_done": 1.0,
                    "done_reward": float(reward),
                    "success_ready": float(success_ready),
                    "gripper_open_norm": float(gripper_open_norm),
                },
            }

            return obs, float(reward), True, False, info

        if distance_now < 0.08:
            arm_scale = 0.35
            wrist_roll_scale = 1.35
            gripper_scale = 0.40
        elif distance_now < 0.14:
            arm_scale = 0.55
            wrist_roll_scale = 1.20
            gripper_scale = 0.45
        else:
            arm_scale = 1.00
            wrist_roll_scale = 1.00
            gripper_scale = 0.45

        scaled_action = smoothed_joint_action.copy()
        scaled_action *= arm_scale

        wrist_roll_idx = self.joint_names.index("wrist_roll")
        scaled_action[wrist_roll_idx] = smoothed_joint_action[wrist_roll_idx] * wrist_roll_scale

        gripper_idx = self.joint_names.index("gripper")
        scaled_action[gripper_idx] = smoothed_joint_action[gripper_idx] * gripper_scale

        self.prev_action = smoothed_full_action.copy()

        joint_delta = scaled_action * self.action_scale
        joint_delta[wrist_roll_idx] *= 1.8
        joint_delta[gripper_idx] *= 0.8

        self.target_joint_positions += joint_delta
        self.target_joint_positions = np.clip(
            self.target_joint_positions,
            self.joint_lower_limits,
            self.joint_upper_limits,
        )

        self.bridge.send_joint_command(self.target_joint_positions)
        time.sleep(self.control_dt)

        q, dq = self.bridge.get_joint_positions_velocities()
        if q is not None:
            self.current_joint_velocities = dq.copy()

        self.step_count += 1

        obs = self._get_observation()
        reward, distance, progress, success, early_fail, reward_components = self._compute_reward(
            scaled_action
        )

        terminated = success or early_fail
        truncated = self.step_count >= self.max_steps

        timeout_penalty = 0.0
        if truncated and not success:
            timeout_penalty = self.timeout_penalty
            reward += timeout_penalty

        info = {
            "distance_to_cube": distance,
            "progress": progress,
            "success": success,
            "success_hold_steps": self.success_hold_steps,
            "step_count": self.step_count,
            "target_joint_positions": self.target_joint_positions.copy(),
            "has_cube_contact": self._has_robot_cube_contact(),
            "has_table_contact": self._has_robot_table_contact(),
            "cube_move": self._get_cube_move_distance(),
            "gripper_open_norm": self._get_gripper_open_norm(),
            "compare_point": self.get_compare_point_position(),
            "cube_position": self._get_cube_position(),
            "early_fail": early_fail,
            "reward_components": reward_components,
            "timeout_penalty": timeout_penalty,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info


if __name__ == "__main__":
    env = SO101GazeboROSEnv(gui=True, max_steps=180)

    try:
        obs, _ = env.reset()
        print("Observation shape:", obs.shape)
        print("Controlled joints:", env.joint_names)
        print("Action dim:", env.action_space.shape)

        for _ in range(env.max_steps):
            action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape[0])
            obs, reward, terminated, truncated, info = env.step(action)
            rc = info.get("reward_components", {})

            print(
                f"step={info['step_count']:03d} "
                f"reward={reward:.3f} "
                f"dist={info['distance_to_cube']:.3f} "
                f"progress={info['progress']:.3f} "
                f"contact={info['has_cube_contact']} "
                f"table={info['has_table_contact']} "
                f"cube_move={info['cube_move']:.3f} "
                f"min_dist={rc.get('min_robot_cube_distance', 0):.3f} "
                f"contact_pen={rc.get('contact_penalty', 0):.3f} "
                f"anti_push={rc.get('anti_push_penalty', 0):.3f} "
                f"safe={rc.get('safe_approach_reward', 0):.3f} "
                f"success={info['success']} "
                f"early_fail={info['early_fail']}"
            )

            if terminated or truncated:
                print("Episode finished.")
                break

    finally:
        env.close()
