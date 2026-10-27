from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "ros2_ws/src/so101_description/urdf/so101.urdf.xacro"


def test_two_contact_sensors_and_bridge_topics_match():
    root = ET.parse(URDF).getroot()
    sensors = [s for s in root.findall("gazebo/sensor") if s.get("type") == "contact"]
    assert len(sensors) == 2
    topics = {s.findtext("contact/topic") for s in sensors}
    assert topics == {"/so101/contacts/raw/fixed_finger", "/so101/contacts/raw/moving_finger"}
    launch = (ROOT / "ros2_ws/src/so101_bringup/launch/sim.launch.py").read_text()
    for topic in topics:
        assert f"{topic.replace('/raw/', '/')}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts" in launch
        assert f"{topic}@ros_gz_interfaces/msg/Contacts" not in launch
    world = ET.parse(ROOT / "ros2_ws/src/so101_gazebo/worlds/empty.sdf").getroot()
    plugins = {p.get("filename") for p in world.findall("world/plugin")}
    assert "ignition-gazebo-contact-system" in plugins
    assert "so101-contact-evidence-system" in plugins


@pytest.mark.skipif(not Path("/opt/ros/humble/setup.bash").exists() or shutil.which("ign") is None,
                    reason="Fortress/xacro conversion tools are not installed")
def test_generated_sdf_sensors_reference_actual_collision_names(tmp_path):
    # URDF fixed-joint reduction changes collision names. This is a static
    # conversion, not a Gazebo simulation; no world/controller is started.
    urdf = subprocess.run(["bash", "-c", 'source /opt/ros/humble/setup.bash && xacro "$1"',
                           "contact_sensor_test", str(URDF)], capture_output=True, text=True, timeout=15, check=True).stdout
    path = tmp_path / "robot.urdf"
    path.write_text(urdf)
    sdf = subprocess.run(["ign", "sdf", "-p", str(path)], capture_output=True, text=True, timeout=15, check=True).stdout
    model = ET.fromstring(sdf)
    seen = set()
    for link in model.findall(".//link"):
        collisions = {item.get("name") for item in link.findall("collision")}
        for sensor in link.findall("sensor"):
            if sensor.get("type") != "contact":
                continue
            assert sensor.findtext("contact/collision") in collisions
            seen.add(sensor.get("name"))
    assert seen == {"fixed_finger_contact", "moving_finger_contact"}
