from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
XACRO = (
    ROOT
    / "ros2_ws"
    / "src"
    / "so101_description"
    / "urdf"
    / "so101.urdf.xacro"
)


def _root() -> ET.Element:
    return ET.parse(XACRO).getroot()


def _link(name: str) -> ET.Element:
    link = _root().find(f"./link[@name='{name}']")
    assert link is not None
    return link


def test_gripper_camera_mount_collision_matches_visual_mesh():
    link = _link("gripper_camera_mount_link")
    collisions = link.findall("./collision")
    assert len(collisions) == 1

    collision = collisions[0]
    assert collision.get("name") == "gripper_camera_mount_collision"

    origin = collision.find("./origin")
    assert origin is not None
    assert origin.get("xyz") == "0 0 0"
    assert origin.get("rpy") == "0 0 0"

    mesh = collision.find("./geometry/mesh")
    assert mesh is not None
    assert (
        mesh.get("filename")
        == "package://so101_description/meshes/gripper_camera_mount.stl"
    )
    assert mesh.get("scale") == "0.001 0.001 0.001"


def test_gripper_camera_body_collision_matches_visual_geometry():
    link = _link("gripper_camera_link")
    collision = link.find("./collision[@name='camera_body_collision']")
    assert collision is not None

    origin = collision.find("./origin")
    assert origin is not None
    assert origin.get("xyz") == "0 0 -0.001"
    assert origin.get("rpy") == "0 0 0"

    box = collision.find("./geometry/box")
    assert box is not None
    assert box.get("size") == "0.024 0.024 0.002"


def test_gripper_camera_lens_collision_matches_visual_geometry():
    link = _link("gripper_camera_link")
    collision = link.find("./collision[@name='camera_lens_collision']")
    assert collision is not None

    origin = collision.find("./origin")
    assert origin is not None
    assert origin.get("xyz") == "0 0 0.014"
    assert origin.get("rpy") == "0 0 0"

    cylinder = collision.find("./geometry/cylinder")
    assert cylinder is not None
    assert cylinder.get("radius") == "0.008"
    assert cylinder.get("length") == "0.028"


def test_gripper_camera_has_exactly_two_named_collisions():
    link = _link("gripper_camera_link")
    names = [collision.get("name") for collision in link.findall("./collision")]
    assert names == ["camera_body_collision", "camera_lens_collision"]
