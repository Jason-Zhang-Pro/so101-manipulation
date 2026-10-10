# Setup and Verification

## Offline verification

Run `python3 scripts/smoke_check.py` from the repository root. This is the verified
minimal path and requires no ROS, GPU, hardware or datasets.

## Runtime environment

The working project used Ubuntu 22.04, ROS 2 Humble, Ignition Gazebo, MoveIt 2,
PyTorch, OpenCV, LeRobot and Stable-Baselines3. Exact dependency versions were not
fully captured. A clean-machine runtime is therefore not yet certified.
The ACT utilities depend on the older LeRobot interfaces used by the working
project; installing an arbitrary latest release is not sufficient evidence of
compatibility. Dataset metadata uses LeRobot v2.1.

Build ROS packages from the repository root:

```bash
source /opt/ros/humble/setup.bash
cd ros
colcon build --symlink-install
source install/setup.bash
```

The Gazebo `ign_ros2_control-system` plugin and controller manager are also needed.
Source ROS in every terminal that imports `rclpy`. Use a Python environment that
can import both ROS Python modules and the learning dependencies.

Prepare relocatable runtime worlds:

```bash
python3 scripts/prepare_world.py
```

This generates `outputs/runtime_worlds/` and inserts the local controller path and
robot URI at runtime. Start Gazebo with the generated world only after setting up
the relevant ROS plugin search paths. A complete bringup launcher from the
separate working ROS workspace is not included in this snapshot.

MoveIt configuration entry point:

```bash
ros2 launch so101_moveit_config move_group.launch.py
```

ACT Gazebo playback accepts `--policy-path` and `--dataset-root`. Install the
compatible LeRobot checkout in your environment (`pip install -e /path/to/lerobot`)
or set `LEROBOT_SRC` explicitly. Models and datasets are external inputs.

## Runtime status

Source compilation, XML/reference checks, offline validation and figure rendering
were tested. ROS build, Gazebo dynamics, real-robot motion and fresh policy
training were not executed as part of assembling this publication draft.
