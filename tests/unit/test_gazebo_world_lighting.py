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


def test_world_uses_uniform_ambient_only_lighting():
    world = _world()
    scene = world.find("./scene")
    assert scene is not None

    ambient = scene.findtext("./ambient")
    background = scene.findtext("./background")
    shadows = scene.findtext("./shadows")

    assert ambient == "0.27 0.27 0.27 1"
    assert background == "0.42 0.46 0.50 1"
    assert shadows == "false"


def test_world_has_no_directional_light():
    world = _world()
    directional_lights = [
        light
        for light in world.findall("./light")
        if light.get("type") == "directional"
    ]
    assert directional_lights == []


def test_main_overhead_light_is_removed():
    world = _world()
    assert world.find("./light[@name='main_overhead_light']") is None
