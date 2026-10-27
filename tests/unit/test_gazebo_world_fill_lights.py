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

EXPECTED_LIGHTS = {
    "fill_light_front_left": "-0.50 0.85 2.50 0 0 0",
    "fill_light_front_right": "-0.50 -0.85 2.50 0 0 0",
    "fill_light_back_left": "1.10 0.85 2.50 0 0 0",
    "fill_light_back_right": "1.10 -0.85 2.50 0 0 0",
}


def _world() -> ET.Element:
    root = ET.parse(WORLD).getroot()
    world = root.find("./world")
    assert world is not None
    return world


def test_scene_uses_daylight_fill_lighting_without_shadows():
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


def test_four_named_shadowless_point_fill_lights_are_present():
    world = _world()
    lights = {
        light.get("name"): light
        for light in world.findall("./light")
        if light.get("name") in EXPECTED_LIGHTS
    }
    assert set(lights) == set(EXPECTED_LIGHTS)

    for name, expected_pose in EXPECTED_LIGHTS.items():
        light = lights[name]
        assert light.get("type") == "point"
        assert light.findtext("./pose") == expected_pose
        assert light.findtext("./cast_shadows") == "false"
        assert light.findtext("./diffuse") == "0.225 0.225 0.225 1"
        assert light.findtext("./specular") == "0.02 0.02 0.02 1"

        attenuation = light.find("./attenuation")
        assert attenuation is not None
        assert attenuation.findtext("./range") == "10"
        assert attenuation.findtext("./constant") == "1.00"
        assert attenuation.findtext("./linear") == "0.10"
        assert attenuation.findtext("./quadratic") == "0.025"


def test_fill_light_names_are_unique():
    world = _world()
    names = [
        light.get("name")
        for light in world.findall("./light")
        if light.get("name") in EXPECTED_LIGHTS
    ]
    assert len(names) == 4
    assert len(set(names)) == 4
