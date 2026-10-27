from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
WORLD = (
    ROOT
    / "ros2_ws"
    / "src"
    / "so101_gazebo"
    / "worlds"
    / "empty.sdf"
)


def _world() -> ET.Element:
    root = ET.parse(WORLD).getroot()
    world = root.find("./world")
    assert world is not None
    return world


def test_bluegray_ambient_world_appearance():
    world = _world()

    scene = world.find("./scene")
    assert scene is not None
    assert scene.findtext("./ambient") == "0.27 0.27 0.27 1"
    assert scene.findtext("./background") == "0.42 0.46 0.50 1"
    assert scene.findtext("./shadows") == "false"

    directional = [
        light
        for light in world.findall("./light")
        if light.get("type") == "directional"
    ]
    assert directional == []

    ground = world.find("./model[@name='ground_plane']")
    assert ground is not None
    material = ground.find("./link/visual/material")
    assert material is not None
    assert material.findtext("./ambient") == "0.20 0.24 0.28 1"
    assert material.findtext("./diffuse") == "0.20 0.24 0.28 1"
    assert material.findtext("./specular") == "0.02 0.02 0.02 1"


def test_lab_table_include_remains_present():
    world = _world()
    includes = world.findall("./include")
    table = [
        include
        for include in includes
        if include.findtext("./name") == "lab_table"
    ]
    assert len(table) == 1
    assert table[0].findtext("./uri") == "model://lab_table"
