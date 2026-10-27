"""Exercise the manager's stream/publication logic without starting ROS."""

import ast
import io
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def manager_module(monkeypatch):
    path = (
        Path(__file__).resolve().parents[2]
        / "ros2_ws/src/so101_gazebo/scripts/target_cube_manager.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ros_modules = ("rclpy", "ament_index_python", "geometry_msgs", "std_msgs", "std_srvs")
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module.startswith(ros_modules)
        )
        and not (
            isinstance(node, ast.Import)
            and any(alias.name.startswith(ros_modules) for alias in node.names)
        )
    ]
    module = ModuleType("_target_manager_unit")
    module.Node = object
    module.Trigger = SimpleNamespace(Request=object, Response=object)
    module.String = SimpleNamespace
    module.PoseStamped = lambda: SimpleNamespace(
        header=SimpleNamespace(),
        pose=SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def make_manager(module):
    manager = module.TargetCubeManager.__new__(module.TargetCubeManager)
    manager.model_name_prefix = "target_cube_"
    manager._states = {}
    manager._deleted_until = {}
    manager._pose_publishers = {}
    manager._state_lock = threading.Lock()
    manager._state_sequence = 0
    manager._stop_event = threading.Event()
    manager._pose_process = SimpleNamespace(poll=lambda: None)
    manager.get_logger = lambda: SimpleNamespace(error=lambda message: None)
    clock_value = SimpleNamespace(
        nanoseconds=42_000_000_000,
        to_msg=lambda: SimpleNamespace(sec=42, nanosec=0),
    )
    manager.get_clock = lambda: SimpleNamespace(now=lambda: clock_value)
    messages = []
    manager._states_publisher = SimpleNamespace(publish=messages.append)
    poses = []
    manager.create_publisher = lambda *args: SimpleNamespace(publish=poses.append)
    return manager, messages, poses


POSE = '''pose {
  name: "target_cube_0001"
  id: 21
  position {
    x: 0.2
    y: 0.1
    z: 0.765
  }
  orientation {
    w: 1
  }
}
'''


def test_stream_preserves_gazebo_stamp_instead_of_reader_time(manager_module):
    manager, _, _ = make_manager(manager_module)
    manager._pose_process = SimpleNamespace(
        stdout=io.StringIO(
            "header {\n  stamp {\n    sec: 7\n    nsec: 125000000\n  }\n}\n" + POSE
        ),
        poll=lambda: 0,
    )
    manager._pose_reader_loop()
    state = manager._states["target_cube_0001"]
    assert state.source_sim_time_s == pytest.approx(7.125)


def test_republication_does_not_advance_cached_pose_source_time(manager_module):
    manager, messages, poses = make_manager(manager_module)
    manager._process_pose_block(POSE, source_sim_time_s=7.125)
    manager._publish_states()
    manager._publish_states()
    first, second = [json.loads(message.data) for message in messages]
    assert second["sequence"] == first["sequence"] + 1
    assert first["sim_time_s"] == second["sim_time_s"] == pytest.approx(7.125)
    assert poses[-1].header.stamp.sec == 7
    assert poses[-1].header.stamp.nanosec == 125000000


def test_reader_exit_stops_refreshing_feedback(manager_module):
    manager, messages, _ = make_manager(manager_module)
    manager._process_pose_block(POSE, source_sim_time_s=7.125)
    manager._pose_process = SimpleNamespace(poll=lambda: 1)
    manager._publish_states()
    assert messages == []
    assert manager._state_sequence == 0


def test_missing_source_stamp_cannot_be_replaced_with_current_time(manager_module):
    manager, messages, _ = make_manager(manager_module)
    manager._process_pose_block(POSE)
    manager._publish_states()
    assert messages == []


def test_empty_registry_still_advances_reset_feedback(manager_module):
    manager, messages, _ = make_manager(manager_module)
    manager._publish_states()
    assert json.loads(messages[0].data)["sim_time_s"] == pytest.approx(42.0)
    assert json.loads(messages[0].data)["cube_count"] == 0


@pytest.mark.parametrize("header, expected", [
    ("header { stamp { } }", 0.0),
    ("header { stamp { sec: 1 nsec: 1 } }", 1.000000001),
    ("header { stamp { nsec: 1000000000 } }", None),
    ("header { stamp { sec: -1 } }", None),
    ("header { }", None),
])
def test_header_timestamp_validation(manager_module, header, expected):
    assert manager_module.TargetCubeManager._pose_header_timestamp_s(header) == expected
