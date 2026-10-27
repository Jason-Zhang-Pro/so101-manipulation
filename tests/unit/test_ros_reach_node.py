import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("std_msgs.msg")
pytest.importorskip("rosgraph_msgs.msg")

from so101_capstone.environments.ros_reach_node import (
    CONTROLLER_COMMAND_TOPIC,
    GRIPPER_CAMERA_TOPIC,
    JOINT_STATE_TOPIC,
    TARGET_CUBE_STATES_TOPIC,
    TOP_CAMERA_TOPIC,
    controller_command_message,
)


def test_ros_topic_contract_is_fixed():
    assert JOINT_STATE_TOPIC == "/joint_states"
    assert TARGET_CUBE_STATES_TOPIC == (
        "/target_cube/states"
    )
    assert TOP_CAMERA_TOPIC == (
        "/top_camera/image_raw"
    )
    assert GRIPPER_CAMERA_TOPIC == (
        "/gripper_camera/image_raw"
    )
    assert CONTROLLER_COMMAND_TOPIC == (
        "/position_controller/commands"
    )


def test_controller_message_preserves_six_joint_order():
    message = controller_command_message(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )

    assert list(message.data) == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]


def test_command_readiness_uses_the_publisher_matched_subscription_count():
    from types import SimpleNamespace
    from so101_capstone.environments.ros_reach_node import So101ReachRosNode

    node = SimpleNamespace(_command_publisher=SimpleNamespace(get_subscription_count=lambda: 2))
    assert So101ReachRosNode.joint_target_subscription_count(node) == 2


def test_invalid_controller_commands_are_rejected():
    with pytest.raises(
        ValueError,
        match=r"shape \(6,\)",
    ):
        controller_command_message(
            [0.0, 0.0, 0.0]
        )

    with pytest.raises(ValueError, match="finite"):
        controller_command_message(
            [
                0.0,
                0.0,
                np.nan,
                0.0,
                0.0,
                0.0,
            ]
        )



def test_target_cube_callback_forwards_source_metadata():
    import json
    from types import SimpleNamespace

    from so101_capstone.environments.state_cache import (
        TargetCubeCollectionCache,
        TargetCubeStateCache,
    )
    from so101_capstone.environments.ros_reach_node import (
        So101ReachRosNode,
    )

    class FakeLogger:
        def warning(self, _message):
            raise AssertionError(
                "Callback unexpectedly rejected the message."
            )

    fake_node = SimpleNamespace(
        _target_cube_name=None,
        target_cube_collection_cache=(
            TargetCubeCollectionCache()
        ),
        target_cube_cache=TargetCubeStateCache(),
        get_logger=lambda: FakeLogger(),
    )

    message = SimpleNamespace(
        data=json.dumps(
            {
                "frame_id": "world",
                "sequence": 321,
                "sim_time_s": 10.5,
                "cube_count": 1,
                "cubes": [
                    {
                        "name": "target_cube_0001",
                        "entity_id": 101,
                        "position": {
                            "x": 0.2,
                            "y": 0.1,
                            "z": 0.765,
                        },
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0,
                        },
                    }
                ],
            }
        )
    )

    So101ReachRosNode._target_cube_states_callback(
        fake_node,
        message,
    )

    snapshot = fake_node.target_cube_cache.latest()
    collection = (
        fake_node.target_cube_collection_cache.latest()
    )

    assert snapshot is not None
    assert snapshot.name == "target_cube_0001"
    assert snapshot.sequence == 1
    assert snapshot.source_sequence == 321
    assert snapshot.source_sim_time_s == pytest.approx(10.5)

    assert collection is not None
    assert collection.cube_count == 1
    assert collection.cube_names == (
        "target_cube_0001",
    )
    assert collection.source_sequence == 321
    assert collection.source_sim_time_s == pytest.approx(
        10.5
    )




def test_empty_target_cube_callback_updates_collection_only():
    import json
    from types import SimpleNamespace

    from so101_capstone.environments.ros_reach_node import (
        So101ReachRosNode,
    )
    from so101_capstone.environments.state_cache import (
        TargetCubeCollectionCache,
        TargetCubeStateCache,
    )

    class FakeLogger:
        def warning(self, message):
            raise AssertionError(message)

    cube_cache = TargetCubeStateCache()
    cube_cache.update(
        name="target_cube_0001",
        position=(0.2, 0.1, 0.765),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_sequence=400,
        source_sim_time_s=20.0,
        received_monotonic_s=5.0,
    )

    fake_node = SimpleNamespace(
        _target_cube_name=None,
        target_cube_collection_cache=(
            TargetCubeCollectionCache()
        ),
        target_cube_cache=cube_cache,
        get_logger=lambda: FakeLogger(),
    )

    previous_cube = cube_cache.latest()

    message = SimpleNamespace(
        data=json.dumps(
            {
                "frame_id": "world",
                "sequence": 401,
                "sim_time_s": 20.033,
                "cube_count": 0,
                "cubes": [],
            }
        )
    )

    So101ReachRosNode._target_cube_states_callback(
        fake_node,
        message,
    )

    collection = (
        fake_node.target_cube_collection_cache.latest()
    )
    detailed_cube = fake_node.target_cube_cache.latest()

    assert collection is not None
    assert collection.cube_count == 0
    assert collection.cube_names == ()
    assert collection.source_sequence == 401
    assert collection.source_sim_time_s == pytest.approx(
        20.033
    )

    # The detailed pose cache is intentionally retained. Reset logic
    # must use the collection cache to determine target existence.
    assert detailed_cube is not None
    assert previous_cube is not None
    assert detailed_cube.name == previous_cube.name
    np.testing.assert_array_equal(
        detailed_cube.position,
        previous_cube.position,
    )
    np.testing.assert_array_equal(
        detailed_cube.quaternion_xyzw,
        previous_cube.quaternion_xyzw,
    )
    assert detailed_cube.sequence == previous_cube.sequence
    assert (
        detailed_cube.received_monotonic_s
        == previous_cube.received_monotonic_s
    )
    assert (
        detailed_cube.source_sequence
        == previous_cube.source_sequence
    )
    assert (
        detailed_cube.source_sim_time_s
        == previous_cube.source_sim_time_s
    )

def test_clock_callback_updates_simulation_cache():
    from types import SimpleNamespace

    from so101_capstone.environments import (
        SimulationClockCache,
    )
    from so101_capstone.environments.ros_reach_node import (
        CLOCK_TOPIC,
        So101ReachRosNode,
    )

    class FakeLogger:
        def error(self, message):
            raise AssertionError(message)

    fake_node = SimpleNamespace(
        simulation_clock_cache=SimulationClockCache(),
        get_logger=lambda: FakeLogger(),
    )

    message = SimpleNamespace(
        clock=SimpleNamespace(
            sec=12,
            nanosec=345_000_000,
        )
    )

    So101ReachRosNode._clock_callback(
        fake_node,
        message,
    )

    snapshot = (
        fake_node.simulation_clock_cache.latest()
    )

    assert CLOCK_TOPIC == "/clock"
    assert snapshot is not None
    assert snapshot.sequence == 1
    assert snapshot.sim_time_s == pytest.approx(
        12.345
    )

def test_runtime_target_selection_is_stable_with_multiple_cubes():
    import json

    from so101_capstone.environments.ros_reach_node import (
        _select_runtime_target_cube,
    )
    from so101_capstone.environments.target_cube_parser import (
        parse_target_cube_states_json,
    )

    def cube(name, entity_id):
        return {
            "name": name,
            "entity_id": entity_id,
            "position": {
                "x": 0.25,
                "y": 0.0,
                "z": 0.765,
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "w": 1.0,
            },
        }

    states = parse_target_cube_states_json(
        json.dumps(
            {
                "frame_id": "world",
                "cube_count": 2,
                "cubes": [
                    cube("target_cube_0002", 102),
                    cube("target_cube_0001", 101),
                ],
            }
        )
    )

    initial = _select_runtime_target_cube(
        states,
        configured_target_name=None,
        current_target_name=None,
    )
    assert initial.name == "target_cube_0001"

    preserved = _select_runtime_target_cube(
        states,
        configured_target_name=None,
        current_target_name="target_cube_0002",
    )
    assert preserved.name == "target_cube_0002"

    configured = _select_runtime_target_cube(
        states,
        configured_target_name="target_cube_0001",
        current_target_name="target_cube_0002",
    )
    assert configured.name == "target_cube_0001"

    remaining = parse_target_cube_states_json(
        json.dumps(
            {
                "frame_id": "world",
                "cube_count": 1,
                "cubes": [
                    cube("target_cube_0002", 102),
                ],
            }
        )
    )

    switched = _select_runtime_target_cube(
        remaining,
        configured_target_name=None,
        current_target_name="target_cube_0001",
    )
    assert switched.name == "target_cube_0002"
