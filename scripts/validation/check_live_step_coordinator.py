"""Live integration check for one deterministic SO-101 simulation step."""

from __future__ import annotations

from threading import Thread
from time import monotonic

import rclpy
from rclpy.executors import MultiThreadedExecutor

from so101_capstone.environments import (
    POLICY_JOINT_NAMES,
    DeterministicSimulationStepCoordinator,
    SimulationStepCoordinatorConfig,
    So101ReachEnvironmentCore,
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


FEEDBACK_TIMEOUT_S = 5.0


def require_initial_feedback(io_node: So101ReachRosNode) -> None:
    """Wait until all required feedback streams are available."""

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

    cube = (
        io_node.target_cube_cache
        .wait_for_sequence_greater_than(
            0,
            FEEDBACK_TIMEOUT_S,
        )
    )
    if cube is None:
        raise RuntimeError(
            "No initial /target_cube/states feedback received. "
            "Confirm that a target cube exists."
        )

    if cube.source_sequence is None:
        raise RuntimeError(
            "Initial target-cube state has no source sequence."
        )

    top = (
        io_node.top_image_cache
        .wait_for_sequence_greater_than(
            0,
            FEEDBACK_TIMEOUT_S,
        )
    )
    if top is None:
        raise RuntimeError(
            "No initial top-camera frame received."
        )

    gripper = (
        io_node.gripper_image_cache
        .wait_for_sequence_greater_than(
            0,
            FEEDBACK_TIMEOUT_S,
        )
    )
    if gripper is None:
        raise RuntimeError(
            "No initial gripper-camera frame received."
        )

    print("initial clock sequence:", clock.sequence)
    print("initial simulation time:", clock.sim_time_s)
    print("initial joint sequence:", joint.sequence)
    print("initial cube local sequence:", cube.sequence)
    print("initial cube source sequence:", cube.source_sequence)
    print("initial cube sim time:", cube.source_sim_time_s)
    print("initial top image sequence:", top.sequence)
    print("initial gripper image sequence:", gripper.sequence)


rclpy.init()

io_node = So101ReachRosNode()

# Use a separate node for synchronous service calls. The I/O node remains
# inside the background executor while Gazebo service calls are in progress.
control_node = rclpy.create_node(
    "so101_world_control_client"
)

executor = MultiThreadedExecutor(num_threads=2)
executor.add_node(io_node)

executor_thread = Thread(
    target=executor.spin,
    name="so101_ros_executor",
    daemon=True,
)
executor_thread.start()

world_controller = GazeboWorldController(
    control_node,
    service_timeout_s=5.0,
    call_timeout_s=5.0,
)

try:
    require_initial_feedback(io_node)

    world_controller.pause()
    print("[OK] Gazebo paused")

    coordinator = DeterministicSimulationStepCoordinator(
        environment=So101ReachEnvironmentCore(),
        world_stepper=world_controller,
        schedule=SimulationStepSchedule(
            control_hz=30.0,
            physics_step_s=0.001,
        ),
        simulation_clock_cache=(
            io_node.simulation_clock_cache
        ),
        joint_cache=io_node.joint_state_cache,
        cube_collection_cache=(
            io_node.target_cube_collection_cache
        ),
        cube_cache=io_node.target_cube_cache,
        top_image_cache=io_node.top_image_cache,
        gripper_image_cache=(
            io_node.gripper_image_cache
        ),
        publish_joint_targets=(
            io_node.publish_joint_targets
        ),
        config=SimulationStepCoordinatorConfig(
            feedback_timeout_s=2.0,
            max_image_age_s=0.5,
            max_image_skew_s=0.05,
        ),
    )

    initial_observation, initial_info = (
        coordinator.reset_from_latest(
            now_monotonic_s=monotonic(),
        )
    )

    latest_joint = io_node.joint_state_cache.latest()

    if latest_joint is None:
        raise RuntimeError(
            "Joint feedback disappeared before stepping."
        )

    # Publish the currently measured positions as the next target.
    # This requests position holding rather than intentional movement.
    hold_targets = [
        latest_joint.positions[name]
        for name in POLICY_JOINT_NAMES
    ]

    print("hold targets:", hold_targets)
    print(
        "initial observation shapes:",
        initial_observation.joint_positions.shape,
        initial_observation.top_image.shape,
        initial_observation.gripper_image.shape,
    )
    print(
        "reset cube source sequence:",
        initial_info["cube_source_sequence"],
    )

    expected_step_counts = (33, 33, 34)
    results = []

    for step_index, expected_physics_steps in enumerate(
        expected_step_counts,
        start=1,
    ):
        latest_joint = io_node.joint_state_cache.latest()

        if latest_joint is None:
            raise RuntimeError(
                "Joint feedback disappeared before stepping."
            )

        step_hold_targets = [
            latest_joint.positions[name]
            for name in POLICY_JOINT_NAMES
        ]

        result = coordinator.step(step_hold_targets)
        results.append(result)

        print()
        print(
            f"===== Coordinated step {step_index} ====="
        )
        print("physics steps:", result.physics_steps)
        print(
            "scheduled simulation time:",
            result.info["scheduled_sim_time_s"],
        )
        print(
            "schedule timing error:",
            result.info["schedule_timing_error_s"],
        )
        print(
            "step start simulation time:",
            result.info["step_start_sim_time_s"],
        )
        print(
            "step end simulation time:",
            result.info["step_end_sim_time_s"],
        )
        print(
            "expected simulation delta:",
            result.info["expected_sim_delta_s"],
        )
        print(
            "actual simulation delta:",
            result.info["actual_sim_delta_s"],
        )
        print(
            "simulation step error:",
            result.info["sim_step_error_s"],
        )
        print(
            "camera lag from step end:",
            result.info[
                "camera_lag_from_step_end_s"
            ],
        )
        print(
            "camera skew:",
            result.camera_frames.skew_s,
        )
        print(
            "cube source sequence:",
            result.cube_state.source_sequence,
        )
        print(
            "top/gripper image sequences:",
            result.camera_frames.top.sequence,
            result.camera_frames.gripper.sequence,
        )

        if result.physics_steps != expected_physics_steps:
            raise RuntimeError(
                f"Step {step_index} used "
                f"{result.physics_steps} physics steps; "
                f"expected {expected_physics_steps}."
            )

        expected_delta_s = (
            expected_physics_steps * 0.001
        )

        if (
            abs(
                result.info["actual_sim_delta_s"]
                - expected_delta_s
            )
            > 1e-9
        ):
            raise RuntimeError(
                f"Step {step_index} actual simulation "
                "delta is incorrect."
            )

    actual_step_counts = tuple(
        result.physics_steps
        for result in results
    )
    cumulative_actual_sim_time_s = sum(
        result.info["actual_sim_delta_s"]
        for result in results
    )
    total_clock_delta_s = (
        results[-1].info["step_end_sim_time_s"]
        - results[0].info["step_start_sim_time_s"]
    )

    print()
    print("===== Three-step summary =====")
    print("physics-step sequence:", actual_step_counts)
    print(
        "scheduled cumulative simulation time:",
        coordinator.schedule.elapsed_sim_time_s,
    )
    print(
        "cumulative schedule timing error:",
        coordinator.schedule.timing_error_s,
    )
    print(
        "summed actual simulation time:",
        cumulative_actual_sim_time_s,
    )
    print(
        "clock start-to-end delta:",
        total_clock_delta_s,
    )

    if actual_step_counts != expected_step_counts:
        raise RuntimeError(
            "Physics-step sequence is not 33, 33, 34."
        )

    if (
        abs(
            coordinator.schedule.elapsed_sim_time_s
            - 0.1
        )
        > 1e-12
    ):
        raise RuntimeError(
            "Scheduled cumulative simulation time "
            "is not 0.1 s."
        )

    if (
        abs(coordinator.schedule.timing_error_s)
        > 1e-12
    ):
        raise RuntimeError(
            "Cumulative schedule timing error "
            "did not return to zero."
        )

    if (
        abs(cumulative_actual_sim_time_s - 0.1)
        > 1e-9
    ):
        raise RuntimeError(
            "Summed actual simulation time "
            "is not 0.1 s."
        )

    if abs(total_clock_delta_s - 0.1) > 1e-9:
        raise RuntimeError(
            "Gazebo clock did not advance exactly 0.1 s."
        )

    print()
    print(
        "[OK] live 33/33/34 deterministic "
        "coordinator validation completed"
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

    # Do not destroy ROS entities while the executor thread may still
    # be using subscriptions, clients, timers, or wait-set handles.
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
