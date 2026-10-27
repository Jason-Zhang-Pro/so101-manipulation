"""Validate policy stepping before a target cube is spawned."""

from __future__ import annotations

from threading import Thread
from time import monotonic, sleep

import rclpy
from rclpy.executors import MultiThreadedExecutor

from so101_capstone.environments import (
    POLICY_JOINT_NAMES,
    DeterministicSimulationStepCoordinator,
    SimulationStepCoordinatorConfig,
    So101ReachEnvironmentCore,
    TargetCubeCollectionSnapshot,
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
from so101_capstone.rewards import ReachRewardConfig


FEEDBACK_TIMEOUT_S = 5.0
PAUSED_CLOCK_SETTLE_S = 0.05
EXPECTED_STEP_COUNTS = (33, 33, 34)
BASE_STEP_REWARD = -0.001


def remaining_time(deadline_s: float) -> float:
    return max(0.0, deadline_s - monotonic())


def require_initial_feedback(
    io_node: So101ReachRosNode,
) -> TargetCubeCollectionSnapshot:
    """Wait for streams required by an empty-scene reset."""

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

    collection = (
        io_node.target_cube_collection_cache
        .wait_for_sequence_greater_than(
            0,
            FEEDBACK_TIMEOUT_S,
        )
    )
    if collection is None:
        raise RuntimeError(
            "No initial target-cube collection received."
        )

    if collection.source_sequence is None:
        raise RuntimeError(
            "Initial target collection has no source sequence."
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
    print(
        "initial collection sequence:",
        collection.sequence,
    )
    print(
        "initial collection source sequence:",
        collection.source_sequence,
    )
    print(
        "initial collection cube names:",
        collection.cube_names,
    )
    print("initial top image sequence:", top.sequence)
    print(
        "initial gripper image sequence:",
        gripper.sequence,
    )

    return collection


def wait_for_new_empty_collection(
    io_node: So101ReachRosNode,
    previous: TargetCubeCollectionSnapshot,
) -> TargetCubeCollectionSnapshot:
    """Wait until a newer source confirms zero managed cubes."""

    if previous.source_sequence is None:
        raise RuntimeError(
            "Previous collection source sequence is missing."
        )

    deadline_s = monotonic() + FEEDBACK_TIMEOUT_S
    local_sequence = previous.sequence

    while True:
        snapshot = (
            io_node.target_cube_collection_cache
            .wait_for_sequence_greater_than(
                local_sequence,
                remaining_time(deadline_s),
            )
        )

        if snapshot is None:
            raise RuntimeError(
                "Timed out waiting for an empty target "
                "collection."
            )

        local_sequence = snapshot.sequence

        if snapshot.source_sequence is None:
            raise RuntimeError(
                "Target collection source sequence is missing."
            )

        if (
            snapshot.source_sequence
            < previous.source_sequence
        ):
            raise RuntimeError(
                "Target collection source sequence "
                "moved backwards."
            )

        if (
            snapshot.source_sequence
            == previous.source_sequence
        ):
            continue

        if snapshot.cube_count == 0:
            return snapshot


def wait_for_paused_clock(
    io_node: So101ReachRosNode,
) -> None:
    """Wait until Gazebo simulation time stops advancing."""

    snapshot = io_node.simulation_clock_cache.latest()

    if snapshot is None:
        raise RuntimeError(
            "No simulation clock is available."
        )

    deadline_s = monotonic() + FEEDBACK_TIMEOUT_S
    stable_since_s = monotonic()
    stable_sim_time_s = snapshot.sim_time_s

    while True:
        if monotonic() >= deadline_s:
            raise RuntimeError(
                "Timed out waiting for paused Gazebo clock."
            )

        sleep(0.005)

        current = io_node.simulation_clock_cache.latest()

        if current is None:
            raise RuntimeError(
                "Simulation clock disappeared."
            )

        if (
            current.sim_time_s
            > stable_sim_time_s + 1e-12
        ):
            stable_sim_time_s = current.sim_time_s
            stable_since_s = monotonic()

        if (
            monotonic() - stable_since_s
            >= PAUSED_CLOCK_SETTLE_S
        ):
            print(
                "paused simulation time:",
                current.sim_time_s,
            )
            return


rclpy.init()

io_node = So101ReachRosNode()
control_node = rclpy.create_node(
    "so101_delayed_target_validation"
)

executor = MultiThreadedExecutor(num_threads=2)
executor.add_node(io_node)

executor_thread = Thread(
    target=executor.spin,
    name="so101_delayed_target_executor",
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

gazebo_paused = False

try:
    initial_collection = require_initial_feedback(
        io_node
    )

    print()
    print("===== Delete previous target cubes =====")

    cube_controller.delete_all()

    empty_collection = wait_for_new_empty_collection(
        io_node,
        initial_collection,
    )

    print(
        "confirmed empty collection source sequence:",
        empty_collection.source_sequence,
    )
    print(
        "confirmed empty cube count:",
        empty_collection.cube_count,
    )

    world_controller.pause()
    gazebo_paused = True
    print("[OK] Gazebo paused")

    wait_for_paused_clock(io_node)

    coordinator = DeterministicSimulationStepCoordinator(
        environment=So101ReachEnvironmentCore(
            reward_config=ReachRewardConfig(
                success_distance_m=1e-6,
            )
        ),
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

    observation, reset_info = (
        coordinator.reset_from_latest(
            now_monotonic_s=monotonic(),
        )
    )

    print()
    print("===== Empty-scene reset =====")
    print(
        "observation shapes:",
        observation.joint_positions.shape,
        observation.top_image.shape,
        observation.gripper_image.shape,
    )
    print(
        "reset target active:",
        reset_info["target_active"],
    )
    print(
        "reset collection cube count:",
        reset_info[
            "target_collection_cube_count"
        ],
    )

    if reset_info["target_active"]:
        raise RuntimeError(
            "Target unexpectedly active after empty reset."
        )

    if (
        reset_info["target_collection_cube_count"]
        != 0
    ):
        raise RuntimeError(
            "Reset collection is not empty."
        )

    results = []
    activation_step = None

    for step_index, expected_physics_steps in enumerate(
        EXPECTED_STEP_COUNTS,
        start=1,
    ):
        if step_index == 2:
            print()
            print("===== Request delayed cube spawn =====")
            cube_controller.spawn_random()
            print("[OK] spawn_random request succeeded")

        latest_joint = io_node.joint_state_cache.latest()

        if latest_joint is None:
            raise RuntimeError(
                "Joint feedback disappeared before step."
            )

        hold_targets = [
            latest_joint.positions[name]
            for name in POLICY_JOINT_NAMES
        ]

        result = coordinator.step(hold_targets)
        results.append(result)

        target_active = bool(
            result.info["target_active"]
        )

        print()
        print(
            f"===== Coordinated step {step_index} ====="
        )
        print("physics steps:", result.physics_steps)
        print("reward:", result.reward)
        print("target active:", target_active)
        print(
            "collection cube count:",
            result.cube_collection_state.cube_count,
        )
        print(
            "collection source sequence:",
            result.cube_collection_state.source_sequence,
        )
        print(
            "cube source sequence:",
            (
                None
                if result.cube_state is None
                else result.cube_state.source_sequence
            ),
        )
        print(
            "progress:",
            result.info["progress_m"],
        )
        print(
            "step simulation delta:",
            result.info["actual_sim_delta_s"],
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
                f"Step {step_index} simulation delta "
                "is incorrect."
            )

        if step_index == 1:
            if result.cube_state is not None:
                raise RuntimeError(
                    "First policy step unexpectedly "
                    "contains a target cube."
                )

            if target_active:
                raise RuntimeError(
                    "Target active before delayed spawn."
                )

            if (
                abs(result.reward - BASE_STEP_REWARD)
                > 1e-12
            ):
                raise RuntimeError(
                    "Empty-scene step did not return "
                    "the base reward."
                )

        if target_active and activation_step is None:
            activation_step = step_index

            if result.cube_state is None:
                raise RuntimeError(
                    "Target active without detailed cube state."
                )

            if (
                result.info["progress_m"] is None
                or abs(result.info["progress_m"]) > 1e-12
            ):
                raise RuntimeError(
                    "First target state did not initialize "
                    "progress to zero."
                )

        if (
            activation_step is not None
            and not target_active
        ):
            raise RuntimeError(
                "Target became inactive after activation."
            )

    actual_step_counts = tuple(
        result.physics_steps
        for result in results
    )
    total_clock_delta_s = (
        results[-1].end_clock.sim_time_s
        - results[0].start_clock.sim_time_s
    )

    print()
    print("===== Delayed-target summary =====")
    print("physics-step sequence:", actual_step_counts)
    print("target activation step:", activation_step)
    print(
        "total clock delta:",
        total_clock_delta_s,
    )
    print(
        "schedule elapsed simulation time:",
        coordinator.schedule.elapsed_sim_time_s,
    )

    if actual_step_counts != EXPECTED_STEP_COUNTS:
        raise RuntimeError(
            "Physics-step sequence is not 33, 33, 34."
        )

    if activation_step not in (2, 3):
        raise RuntimeError(
            "Target was not activated after spawn request."
        )

    if abs(total_clock_delta_s - 0.1) > 1e-9:
        raise RuntimeError(
            "Gazebo clock did not advance exactly 0.1 s."
        )

    if (
        abs(
            coordinator.schedule.elapsed_sim_time_s
            - 0.1
        )
        > 1e-12
    ):
        raise RuntimeError(
            "Schedule elapsed time is not 0.1 s."
        )

    print()
    print(
        "[OK] live delayed-target policy stepping "
        "validation completed"
    )

finally:
    if gazebo_paused:
        try:
            world_controller.resume()
            gazebo_paused = False
            print("[OK] Gazebo resumed")
        except Exception as error:
            print(
                "[ERROR] Gazebo resume failed:",
                error,
            )

    try:
        current_collection = (
            io_node.target_cube_collection_cache.latest()
        )

        if (
            current_collection is not None
            and current_collection.cube_count > 0
        ):
            cube_controller.delete_all()
            cleaned = wait_for_new_empty_collection(
                io_node,
                current_collection,
            )
            print(
                "[OK] test cube deleted; cube count:",
                cleaned.cube_count,
            )
    except Exception as error:
        print(
            "[ERROR] Test cube cleanup failed:",
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
