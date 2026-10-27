import os
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _boolean_value(value: str, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be true or false, received {value!r}.")


def _environment_randomized_world(base_world_file: str) -> str:
    source = Path(base_world_file).read_text(encoding="utf-8")
    anchor = 'filename="ignition-gazebo-scene-broadcaster-system"'
    if source.count(anchor) != 1:
        raise RuntimeError(
            "Could not locate a unique SceneBroadcaster plugin in the installed world."
        )

    anchor_index = source.index(anchor)
    plugin_start = source.rfind("<plugin", 0, anchor_index)
    if plugin_start < 0:
        raise RuntimeError(
            "Could not locate the SceneBroadcaster plugin start tag."
        )

    line_start = source.rfind("\n", 0, plugin_start) + 1
    indent = source[line_start:plugin_start]
    if indent.strip():
        raise RuntimeError(
            "Unexpected text before the SceneBroadcaster plugin start tag."
        )

    plugin = (
        f'{indent}<plugin\n'
        f'{indent}  filename="libso101-environment-randomization-system.so"\n'
        f'{indent}  name="so101::gazebo::EnvironmentRandomizationSystem"/>\n\n'
    )
    rendered = source[:line_start] + plugin + source[line_start:]

    domain = os.environ.get("ROS_DOMAIN_ID", "default")
    path = Path("/tmp") / (
        f"so101_empty_randomized_domain_{domain}_pid_{os.getpid()}.sdf"
    )
    path.write_text(rendered, encoding="utf-8")
    return str(path)


def _gazebo_args(world_file: str, gazebo_gui: str) -> str:
    value = str(gazebo_gui).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return f'-r "{world_file}"'
    if value in {"false", "0", "no", "off"}:
        return f'-s -r "{world_file}"'
    raise ValueError(
        "gazebo_gui must be true or false, "
        f"received {gazebo_gui!r}."
    )


def _launch_gazebo(context, *, ros_gz_sim_share: str, world_file: str):
    gazebo_gui = LaunchConfiguration("gazebo_gui").perform(context)
    environment_randomization = _boolean_value(
        LaunchConfiguration("environment_randomization").perform(context),
        name="environment_randomization",
    )
    selected_world_file = (
        _environment_randomized_world(world_file)
        if environment_randomization
        else world_file
    )
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    ros_gz_sim_share,
                    "launch",
                    "gz_sim.launch.py",
                )
            ),
            launch_arguments={
                "gz_args": _gazebo_args(selected_world_file, gazebo_gui),
            }.items(),
        )
    ]


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    description_share = get_package_share_directory("so101_description")
    gazebo_share = get_package_share_directory("so101_gazebo")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    xacro_file = os.path.join(
        description_share,
        "urdf",
        "so101.urdf.xacro",
    )

    controllers_file = os.path.join(
        gazebo_share,
        "config",
        "controllers.yaml",
    )

    world_file = os.path.join(
        gazebo_share,
        "worlds",
        "empty.sdf",
    )

    target_cube_config = os.path.join(
        gazebo_share,
        "config",
        "target_cube.yaml",
    )

    for required_file in (
        xacro_file,
        controllers_file,
        world_file,
        target_cube_config,
    ):
        if not os.path.isfile(required_file):
            raise FileNotFoundError(
                f"Required simulation file not found: {required_file}"
            )

    robot_description = xacro.process_file(
        xacro_file,
        mappings={
            "controllers_file": controllers_file,
        },
    ).toxml()

    resource_paths = [
        str(Path(description_share).parent),
        os.path.join(gazebo_share, "models"),
        os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
    ]
    gazebo_resource_path = os.pathsep.join(
        path for path in resource_paths if path
    )

    plugin_paths = [
        "/opt/ros/humble/lib",
        os.path.join(get_package_prefix("so101_gazebo"), "lib"),
        os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", ""),
    ]
    gazebo_plugin_path = os.pathsep.join(
        path for path in plugin_paths if path
    )

    gz_sim = OpaqueFunction(
        function=_launch_gazebo,
        kwargs={
            "ros_gz_sim_share": ros_gz_sim_share,
            "world_file": world_file,
        },
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    spawn_robot = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                name="spawn_so101",
                output="screen",
                arguments=[
                    "-name",
                    "so101",
                    "-topic",
                    "robot_description",
                    "-x",
                    "0.0",
                    "-y",
                    "0.0",
                    "-z",
                    "0.7524",
                ],
            )
        ],
    )

    sensor_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="sensor_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/top_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/top_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/gripper_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/gripper_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/so101/contacts/fixed_finger@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
            "/so101/contacts/moving_finger@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
            "/world/empty/so101/grasp_attachment/state@std_msgs/msg/String[gz.msgs.StringMsg",
        ],
        remappings=[
            ("/world/empty/so101/grasp_attachment/state", "/so101/grasp_attachment_state"),
        ],
    )

    world_control_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="world_control_bridge",
        output="screen",
        arguments=[
            (
                "/world/empty/control@"
                "ros_gz_interfaces/srv/ControlWorld@"
                "ignition.msgs.WorldControl@"
                "ignition.msgs.Boolean"
            ),
        ],
    )

    goal_region_pose_publisher = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="so101_gazebo",
                executable="goal_region_pose_publisher.py",
                name="goal_region_pose_publisher",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "world_pose_topic": (
                            "/world/empty/pose/info"
                        ),
                        "table_model_name": "lab_table",
                        "goal_link_name": (
                            "goal_region_plate_link"
                        ),
                        "output_topic": (
                            "/goal_region/center"
                        ),
                    }
                ],
            )
        ],
    )

    target_cube_manager = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="so101_gazebo",
                executable="target_cube_manager.py",
                name="target_cube_manager",
                output="screen",
                parameters=[
                    target_cube_config,
                    {"use_sim_time": use_sim_time},
                ],
            )
        ],
    )

    joint_state_broadcaster = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "joint_state_broadcaster",
                    "--controller-manager",
                    "/controller_manager",
                    "--controller-manager-timeout",
                    "60",
                ],
            )
        ],
    )

    position_controller = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "position_controller",
                    "--controller-manager",
                    "/controller_manager",
                    "--controller-manager-timeout",
                    "60",
                ],
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use Gazebo simulation time",
            ),
            DeclareLaunchArgument(
                "gazebo_gui",
                default_value="true",
                description=(
                    "Open the Gazebo GUI. Set false to run server-only "
                    "with the Fortress -s option."
                ),
            ),
            DeclareLaunchArgument(
                "environment_randomization",
                default_value=os.environ.get(
                    "SO101_ENVIRONMENT_RANDOMIZATION_ENABLED",
                    "false",
                ),
                description=(
                    "Load the SO-101 environment-randomization world plugin "
                    "when true. The original world is used unchanged when false."
                ),
            ),
            SetEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH",
                gazebo_resource_path,
            ),
            SetEnvironmentVariable(
                "IGN_GAZEBO_RESOURCE_PATH",
                gazebo_resource_path,
            ),
            SetEnvironmentVariable(
                "GZ_SIM_SYSTEM_PLUGIN_PATH",
                gazebo_plugin_path,
            ),
            SetEnvironmentVariable(
                "IGN_GAZEBO_SYSTEM_PLUGIN_PATH",
                gazebo_plugin_path,
            ),
            gz_sim,
            robot_state_publisher,
            spawn_robot,
            sensor_bridge,
            world_control_bridge,
            goal_region_pose_publisher,
            target_cube_manager,
            joint_state_broadcaster,
            position_controller,
        ]
    )
