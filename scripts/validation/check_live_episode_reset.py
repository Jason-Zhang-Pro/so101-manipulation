"""Live validation of deterministic SO-101 HOME episode reset."""

from __future__ import annotations

from threading import Thread

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from so101_capstone.environments.episode_reset import (
    EpisodeResetConfig,
    EpisodeResetManager,
)
from so101_capstone.environments.gazebo_world_control import (
    GazeboWorldController,
)
from so101_capstone.environments.ros_reach_node import (
    So101ReachRosNode,
)
from so101_capstone.environments.simulation_step import (
    SimulationStepSchedule,
)
from so101_capstone.environments.target_cube_control import (
    TargetCubeController,
)
from so101_capstone.environments.state_cache import (
    MEASURED_JOINT_NAMES,
)


FEEDBACK_TIMEOUT_S = 5.0


def require_initial_feedback(
    io_node: So101ReachRosNode,
) -> None:
    """Wait for the feedback required by HOME reset."""

    clock = (
        io_node.simulation_clock_cache
        .wait_for_sequence_greater_than(
            0,
            FEEDBACK_TIMEOUT_S,
        )
    )

    if clock is None:
        raise RuntimeError(
            "No initial /clock feedback received."
        )

    joint = (
        io_node.joint_state_cache
        .wait_for_sequence_greater_than(
            0,
            FEEDBACK_TIMEOUT_S,
        )
    )

    if joint is None:
        raise RuntimeError(
            "No initial /joint_states feedback received."
        )

    cube_collection = (
        io_node.target_cube_collection_cache
        .wait_for_sequence_greater_than(
            0,
            FEEDBACK_TIMEOUT_S,
        )
    )

    if cube_collection is None:
        raise RuntimeError(
            "No initial target-cube collection feedback received."
        )

    if cube_collection.source_sequence is None:
        raise RuntimeError(
            "Initial cube collection has no source sequence."
        )

    if cube_collection.source_sim_time_s is None:
        raise RuntimeError(
            "Initial cube collection has no simulation time."
        )

    if joint.velocities is None:
        raise RuntimeError(
            "Initial joint feedback has no velocities."
        )

    positions = tuple(
        joint.positions[name]
        for name in MEASURED_JOINT_NAMES
    )
    velocities = tuple(
        joint.velocities[name]
        for name in MEASURED_JOINT_NAMES
    )

    print("initial clock sequence:", clock.sequence)
    print("initial simulation time:", clock.sim_time_s)
    print("initial joint sequence:", joint.sequence)
    print("initial joint positions:", positions)
    print("initial joint velocities:", velocities)
    print(
        "initial cube collection sequence:",
        cube_collection.sequence,
    )
    print(
        "initial cube source sequence:",
        cube_collection.source_sequence,
    )
    print(
        "initial cube simulation time:",
        cube_collection.source_sim_time_s,
    )
    print(
        "initial cube names:",
        cube_collection.cube_names,
    )


def expected_physics_steps(
    control_steps: int,
) -> int:
    """Calculate the exact 33/33/34 total for N control steps."""

    schedule = SimulationStepSchedule(
        control_hz=30.0,
        physics_step_s=0.001,
    )

    return sum(
        schedule.next_step_count()
        for _ in range(control_steps)
    )


rclpy.init()

io_node = So101ReachRosNode()

# Keep synchronous Gazebo service calls outside the I/O executor node.
control_node = rclpy.create_node(
    "so101_episode_reset_control_client"
)

executor = MultiThreadedExecutor(num_threads=2)
executor.add_node(io_node)

executor_thread = Thread(
    target=executor.spin,
    name="so101_episode_reset_executor",
    daemon=True,
)
executor_thread.start()

world_controller = GazeboWorldController(
    control_node,
    service_timeout_s=5.0,
    call_timeout_s=5.0,
)

cube_controller = TargetCubeController(
    control_node,
    service_timeout_s=5.0,
    call_timeout_s=5.0,
)

schedule = SimulationStepSchedule(
    control_hz=30.0,
    physics_step_s=0.001,
)

config = EpisodeResetConfig(
    home_joint_positions=(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    position_tolerance_rad=0.01,
    velocity_tolerance_rad_s=0.05,
    stable_sample_count=5,
    max_control_steps=300,
    feedback_timeout_s=2.0,
    max_sim_step_error_s=1e-6,
)

try:
    require_initial_feedback(io_node)

    manager = EpisodeResetManager(
        world_controller=world_controller,
        cube_controller=cube_controller,
        schedule=schedule,
        simulation_clock_cache=(
            io_node.simulation_clock_cache
        ),
        joint_cache=io_node.joint_state_cache,
        cube_collection_cache=(
            io_node.target_cube_collection_cache
        ),
        publish_joint_targets=(
            io_node.publish_joint_targets
        ),
        config=config,
    )

    print()
    print("===== Running HOME episode reset =====")

    result = manager.reset_to_home()

    final_positions = np.asarray(
        [
            result.joint_state.positions[name]
            for name in MEASURED_JOINT_NAMES
        ],
        dtype=float,
    )
    final_velocities = np.asarray(
        [
            result.joint_state.velocities[name]
            for name in MEASURED_JOINT_NAMES
        ],
        dtype=float,
    )

    expected_total_steps = expected_physics_steps(
        result.control_steps
    )

    print("control steps:", result.control_steps)
    print("physics steps:", result.physics_steps)
    print(
        "expected physics steps:",
        expected_total_steps,
    )
    print(
        "stable sample count:",
        result.stable_sample_count,
    )
    print(
        "empty cube collection sequence:",
        result.cube_collection_state.sequence,
    )
    print(
        "empty cube source sequence:",
        result.cube_collection_state.source_sequence,
    )
    print(
        "empty cube simulation time:",
        result.cube_collection_state.source_sim_time_s,
    )
    print(
        "empty cube count:",
        result.cube_collection_state.cube_count,
    )
    print(
        "max position error:",
        result.max_position_error_rad,
    )
    print(
        "max velocity:",
        result.max_velocity_rad_s,
    )
    print(
        "final simulation time:",
        result.clock_state.sim_time_s,
    )
    print("final joint positions:", tuple(final_positions))
    print("final joint velocities:", tuple(final_velocities))
    print(
        "schedule control count after reset:",
        schedule.control_step_count,
    )
    print(
        "schedule physics count after reset:",
        schedule.physics_step_count,
    )

    if result.physics_steps != expected_total_steps:
        raise RuntimeError(
            "Reset physics-step total does not match "
            "the 33/33/34 schedule."
        )

    if result.cube_collection_state.cube_count != 0:
        raise RuntimeError(
            "Target cubes were not deleted during reset."
        )

    if (
        result.stable_sample_count
        < config.stable_sample_count
    ):
        raise RuntimeError(
            "HOME stability sample requirement was not met."
        )

    if (
        result.max_position_error_rad
        > config.position_tolerance_rad
    ):
        raise RuntimeError(
            "Final HOME position error exceeds tolerance."
        )

    if (
        result.max_velocity_rad_s
        > config.velocity_tolerance_rad_s
    ):
        raise RuntimeError(
            "Final HOME velocity exceeds tolerance."
        )

    if (
        schedule.control_step_count != 0
        or schedule.physics_step_count != 0
    ):
        raise RuntimeError(
            "Policy schedule was not reset after HOME reset."
        )

    first_policy_physics_steps = (
        schedule.next_step_count()
    )

    print(
        "first policy physics steps:",
        first_policy_physics_steps,
    )

    if first_policy_physics_steps != 33:
        raise RuntimeError(
            "First policy step after reset is not 33."
        )

    # Restore the unused policy schedule to its clean initial state.
    schedule.reset()

    print()
    print(
        "[OK] live HOME episode reset validation completed"
    )

finally:
    try:
        world_controller.resume()
        print("[OK] Gazebo resumed")
    except Exception as error:
        print(
            "[ERROR] Gazebo resume failed:",
            error,
        )

    executor.shutdown()
    executor_thread.join()

    if executor_thread.is_alive():
        raise RuntimeError(
            "ROS executor thread did not stop cleanly."
        )

    executor.remove_node(io_node)

    io_node.destroy_node()
    control_node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()
