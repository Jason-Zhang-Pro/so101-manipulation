import ast
from pathlib import Path

import pytest


def load_gazebo_args_function():
    root = Path(__file__).resolve().parents[2]
    path = root / "ros2_ws/src/so101_bringup/launch/sim.launch.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_gazebo_args"
        ),
        None,
    )
    assert function is not None, "sim.launch.py must define _gazebo_args"
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {}
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["_gazebo_args"]


def test_gazebo_gui_true_preserves_normal_gui_launch_arguments():
    gazebo_args = load_gazebo_args_function()
    assert gazebo_args("/tmp/world.sdf", "true") == (
        '-r "/tmp/world.sdf"'
    )


def test_gazebo_gui_false_uses_fortress_server_only_flag():
    gazebo_args = load_gazebo_args_function()
    assert gazebo_args("/tmp/world.sdf", "false") == (
        '-s -r "/tmp/world.sdf"'
    )


def test_gazebo_gui_rejects_unknown_value():
    gazebo_args = load_gazebo_args_function()
    with pytest.raises(ValueError, match="gazebo_gui"):
        gazebo_args("/tmp/world.sdf", "maybe")
