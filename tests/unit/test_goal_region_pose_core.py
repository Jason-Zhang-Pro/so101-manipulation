import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = (
    PROJECT_ROOT
    / "ros2_ws"
    / "src"
    / "so101_gazebo"
    / "scripts"
)

sys.path.insert(0, str(SCRIPT_DIRECTORY))

from goal_region_pose_core import (  # noqa: E402
    PoseSample,
    parse_pose_block,
    transform_point,
)


def test_parse_pose_block_defaults_omitted_zero_fields():
    sample = parse_pose_block(
        """
pose {
  name: "lab_table"
  id: 8
  position {
    x: 0.278
    y: 0.05
  }
  orientation {
    w: 1
  }
}
"""
    )

    assert sample.name == "lab_table"
    assert sample.x == 0.278
    assert sample.y == 0.05
    assert sample.z == 0.0
    assert sample.qx == 0.0
    assert sample.qy == 0.0
    assert sample.qz == 0.0
    assert sample.qw == 1.0


def test_real_table_and_goal_link_resolve_expected_world_point():
    table = parse_pose_block(
        """
pose {
  name: "lab_table"
  id: 8
  position {
    x: 0.278
    y: 0.05
  }
  orientation {
    z: 0.70710666564709435
    w: 0.70710689672598181
  }
}
"""
    )

    goal = parse_pose_block(
        """
pose {
  name: "goal_region_plate_link"
  id: 20
  position {
    x: -0.35
    y: 0.20000000000000007
    z: 0.75150000000000006
  }
  orientation {
    w: 1
  }
}
"""
    )

    x, y, z = transform_point(table, goal)

    assert math.isclose(x, 0.078, abs_tol=1.0e-6)
    assert math.isclose(y, -0.300, abs_tol=1.0e-6)
    assert math.isclose(z, 0.7515, abs_tol=1.0e-9)


def test_transform_point_applies_parent_translation():
    parent = PoseSample(
        name="parent",
        x=1.0,
        y=2.0,
        z=3.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0,
    )
    child = PoseSample(
        name="child",
        x=0.1,
        y=0.2,
        z=0.3,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0,
    )

    assert transform_point(parent, child) == (
        1.1,
        2.2,
        3.3,
    )


def test_transform_point_rejects_zero_parent_quaternion():
    parent = PoseSample(
        name="parent",
        x=0.0,
        y=0.0,
        z=0.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=0.0,
    )
    child = PoseSample(
        name="child",
        x=1.0,
        y=0.0,
        z=0.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=1.0,
    )

    try:
        transform_point(parent, child)
    except ValueError as error:
        assert "quaternion" in str(error)
    else:
        raise AssertionError(
            "Expected an invalid quaternion to be rejected"
        )
