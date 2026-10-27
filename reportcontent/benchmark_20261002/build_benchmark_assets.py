#!/usr/bin/env python3
"""Build isolated, matched-runtime benchmark assets; never start a simulator.

The baseline reconstructs pre-optimization collision/camera/lighting assets on
the current common scene, ROS launch, controllers, and contact instrumentation.
It is not a checkout of the historical training stack. Generated SDF contact
names must be verified before the baseline is launched.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASELINE = "1eb21fc9687de67283d1d2194f308a88db4ab7ee"
UPSTREAM = Path("/media/tiger/New Volume/ubuntufile/old_cap_robot/cap_robot/robot_models(original)/SO-ARM100/Simulation/SO101")
PACKAGES = ("so101_description", "so101_gazebo", "so101_bringup")
URDF_REL = Path("share/so101_description/urdf/so101.urdf.xacro")
WORLD_REL = Path("share/so101_gazebo/worlds/empty.sdf")
LAUNCH_REL = Path("share/so101_bringup/launch/sim.launch.py")
CONTACTS = {
    "fixed_finger_contact": {
        "link": "gripper_link", "mesh": "wrist_roll_follower_so101_v1.stl",
        "marker": "benchmark_original_fixed_jaw",
    },
    "moving_finger_contact": {
        "link": "moving_jaw_so101_v1_link", "mesh": "moving_jaw_so101_v1.stl",
        "marker": "benchmark_original_moving_jaw",
    },
}
ET.register_namespace("xacro", "http://www.ros.org/wiki/xacro")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)


def write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root, space="  ")
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n")


def canonical(node: ET.Element) -> list:
    return [node.tag, dict(sorted(node.attrib.items())), (node.text or "").strip(),
            [canonical(child) for child in node]]


def prefix(variant: str) -> Path:
    return OUT / "assets" / variant


def immutable_inputs() -> dict[str, str]:
    result = {}
    for package in PACKAGES:
        for item in (ROOT / "ros2_ws/install" / package / "share" / package).rglob("*"):
            if item.is_file() and "__pycache__" not in item.parts:
                result[str(item)] = sha(item)
    result[str(Path.home() / ".ignition/gazebo/6/gui.config")] = sha(Path.home() / ".ignition/gazebo/6/gui.config")
    for item in (ROOT / "ros2_ws/install/so101_gazebo/lib").glob("*.so"):
        result[str(item)] = sha(item)
    return result


def copy_prefix(variant: str) -> None:
    target = prefix(variant)
    if target.is_symlink():
        raise ValueError("Benchmark prefix itself must not be a symlink")
    for package in PACKAGES:
        source = ROOT / "ros2_ws/install" / package / "share" / package
        dest = target / "share" / package
        shutil.copytree(source, dest, symlinks=False, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("meshes", "__pycache__", "*.pyc"))
        marker = target / "share/ament_index/resource_index/packages" / package
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    links = {
        target / "share/so101_description/meshes": ROOT / "ros2_ws/install/so101_description/share/so101_description/meshes",
        target / "lib": ROOT / "ros2_ws/install/so101_gazebo/lib",
    }
    for link, source in links.items():
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError(f"Unexpected preexisting symlink: {link}")
        elif link.exists():
            raise ValueError(f"Refusing to replace an existing directory: {link}")
        else:
            link.symlink_to(source.resolve(), target_is_directory=True)


def add_gui_config() -> dict:
    source = Path.home() / ".ignition/gazebo/6/gui.config"
    text = source.read_text()
    text, width_count = re.subn(r"(<window>\s*<width>)\d+(</width>)", r"\g<1>640\2", text, count=1)
    text, height_count = re.subn(r"(<window>\s*<width>\d+</width>\s*<height>)\d+(</height>)", r"\g<1>480\2", text, count=1)
    text, camera_count = re.subn(r"<camera_pose>[^<]+</camera_pose>",
                               "<camera_pose>1.30 -1.30 1.65 0 0.42 2.32</camera_pose>", text, count=1)
    if (width_count, height_count, camera_count) != (1, 1, 1):
        raise ValueError("Expected one main window width, height, and camera pose")
    # Preserve every existing GUI plugin and menu; only suppress first-run dialog.
    text = text.replace('name="quick_start" show_again="true"', 'name="quick_start" show_again="false"')
    ET.fromstring("<root>" + re.sub(r"<\?xml[^>]+\?>", "", text) + "</root>")
    dest = OUT / "gui.config"
    dest.write_text(text)
    return {"source": str(source), "source_sha256": sha(source), "copy": str(dest),
            "sha256": sha(dest), "width": 640, "height": 480,
            "camera_pose": [1.30, -1.30, 1.65, 0.0, 0.42, 2.32]}


def upstream_geometry_proof() -> dict:
    """Prove the baseline arm retains every unmodified upstream mesh instance."""
    upstream_urdf = UPSTREAM / "so101_new_calib.urdf"
    baseline_urdf = prefix("baseline") / URDF_REL
    original = ET.parse(upstream_urdf).getroot()
    baseline = ET.parse(baseline_urdf).getroot()

    def instances(robot: ET.Element) -> list:
        result = []
        for link in robot.findall("link"):
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None:
                    continue
                name = Path(mesh.get("filename")).name
                if name == "gripper_camera_mount.stl":
                    continue  # Additional camera hardware, never a replacement arm mesh.
                origin = collision.find("origin")
                result.append({"link": link.get("name"), "mesh": name,
                               "scale": mesh.get("scale", "1 1 1"),
                               "origin": dict(origin.attrib) if origin is not None else {}})
        return result

    original_instances = instances(original)
    baseline_instances = instances(baseline)
    assert original_instances == baseline_instances, "Baseline arm collision instances differ from upstream"
    records = []
    for name in sorted({entry["mesh"] for entry in original_instances}):
        source = UPSTREAM / "assets" / name
        current = prefix("baseline") / "share/so101_description/meshes" / name
        source_bytes, current_bytes = source.read_bytes(), current.read_bytes()
        historical = git_blob("ros2_ws/src/so101_description/meshes/" + name)
        faces = struct.unpack("<I", source_bytes[80:84])[0]
        assert len(source_bytes) == 84 + 50 * faces
        assert source_bytes == current_bytes == historical, f"Baseline original mesh was changed: {name}"
        records.append({"name": name, "upstream_path": str(source), "baseline_resolved_path": str(current.resolve()),
                        "upstream_sha256": hashlib.sha256(source_bytes).hexdigest(),
                        "baseline_sha256": hashlib.sha256(current_bytes).hexdigest(),
                        "historical_1eb21fc_sha256": hashlib.sha256(historical).hexdigest(),
                        "byte_identical": True, "triangles": faces,
                        "robot_instance_count": sum(entry["mesh"] == name for entry in original_instances)})
    assert len(records) == 13 and len(original_instances) == 17
    return {"source_urdf": str(upstream_urdf), "source_urdf_sha256": sha(upstream_urdf),
            "unique_original_arm_meshes": len(records), "arm_collision_instances": len(original_instances),
            "unique_triangles": sum(record["triangles"] for record in records),
            "instance_triangles": sum(record["triangles"] * record["robot_instance_count"] for record in records),
            "all_mesh_bytes_identical_to_upstream_and_1eb21fc": True,
            "all_collision_links_origins_scales_and_order_identical": True,
            "meshes": records, "instances": original_instances,
            "extra_camera_geometry": "Unmodified 3516-face camera mount plus camera box/cylinder primitives are added scene hardware; no upstream arm geometry is replaced by those primitives."}


def patch_common_launch(path: Path) -> None:
    text = path.read_text()
    old = "        return f'-r \"{world_file}\"'"
    new = ("        gui_config = os.environ.get(\"SO101_BENCHMARK_GUI_CONFIG\", \"\")\n"
           "        suffix = f' --gui-config \"{gui_config}\"' if gui_config else \"\"\n"
           "        return f'-r \"{world_file}\"' + suffix")
    if text.count(old) != 1:
        raise ValueError("Common launch GUI hook did not match exactly once")
    path.write_text(text.replace(old, new))


def sensors(urdf: ET.Element) -> list[dict]:
    return [{"name": s.get("name"), "type": s.get("type"),
             "always_on": s.findtext("always_on"), "hz": s.findtext("update_rate"),
             "width": s.findtext("camera/image/width"), "height": s.findtext("camera/image/height"),
             "contact_collision": s.findtext("contact/collision")}
            for s in urdf.findall(".//sensor")]


def validate_assets(variant: str) -> dict:
    base = prefix(variant)
    robot = ET.parse(base / URDF_REL).getroot()
    world = ET.parse(base / WORLD_REL).getroot()
    meshes = []
    for name in sorted({m.get("filename") for m in robot.findall(".//mesh")}):
        if not name.startswith("package://"):
            raise ValueError(f"Unexpected mesh URI: {name}")
        package, rel = name[len("package://"):].split("/", 1)
        resolved = base / "share" / package / rel
        if not resolved.is_file():
            raise ValueError(f"Missing mesh: {resolved}")
        meshes.append({"uri": name, "resolved": str(resolved.resolve()), "sha256": sha(resolved)})
    sensor_list = sensors(robot)
    by_name = {s["name"]: s for s in sensor_list}
    expected = ("1280", "960", "60") if variant == "baseline" else ("800", "600", "45")
    assert tuple(by_name["top_camera_sensor"][k] for k in ("width", "height", "hz")) == expected
    assert tuple(by_name["gripper_camera_sensor"][k] for k in ("width", "height", "hz")) == ("640", "480", expected[2])
    assert all(name in by_name for name in CONTACTS)
    assert any("ContactEvidenceSystem" in p.get("name", "") for p in world.findall("world/plugin"))
    assert all(by_name[n]["always_on"] == "false" for n in ("top_camera_segmentation_sensor", "gripper_camera_segmentation_sensor"))
    return {"meshes": meshes, "sensors": sensor_list,
            "scene_shadows": world.findtext("world/scene/shadows"),
            "lights": [{"name": p.get("name"), "type": p.get("type"),
                        "cast_shadows": p.findtext("cast_shadows")} for p in world.findall("world/light")],
            "plugins": [dict(p.attrib) for p in world.findall("world/plugin")]}


def refresh_manifest(manifest: dict) -> None:
    manifest["original_arm_geometry_proof"] = upstream_geometry_proof()
    for variant in ("baseline", "latest"):
        base = prefix(variant)
        item = manifest["variants"][variant]
        item["asset_validation"] = validate_assets(variant)
        item["files_sha256"] = {str(p.relative_to(base)): sha(p) for p in base.rglob("*")
                                if p.is_file() and not p.is_symlink()}
    for name, relative in (("urdf", URDF_REL), ("world", WORLD_REL), ("launch", LAUNCH_REL)):
        before = (prefix("baseline") / relative).read_text().splitlines(keepends=True)
        after = (prefix("latest") / relative).read_text().splitlines(keepends=True)
        (OUT / f"baseline_vs_latest_{name}.diff").write_text("".join(difflib.unified_diff(before, after, fromfile=f"baseline/{relative}", tofile=f"latest/{relative}")))
    (OUT / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build() -> None:
    before = immutable_inputs()
    for variant in ("baseline", "latest"):
        copy_prefix(variant)
        patch_common_launch(prefix(variant) / LAUNCH_REL)
    gui = add_gui_config()
    latest_robot = ET.parse(prefix("latest") / URDF_REL).getroot()
    baseline_robot = ET.fromstring(git_blob("ros2_ws/src/so101_description/urdf/so101.urdf.xacro"))
    for name, mapping in CONTACTS.items():
        link = baseline_robot.find(f"link[@name='{mapping['link']}']")
        candidates = [c for c in link.findall("collision") if c.find("geometry/mesh") is not None
                      and c.find("geometry/mesh").get("filename", "").endswith("/" + mapping["mesh"])]
        if len(candidates) != 1:
            raise ValueError(f"Original collision not unique: {name}")
        candidates[0].set("name", mapping["marker"])
        element = next(g for g in latest_robot.findall("gazebo") if g.find(f"sensor[@name='{name}']") is not None)
        appended = copy.deepcopy(element)
        appended.find("sensor/contact/collision").text = mapping["marker"]
        baseline_robot.append(appended)
    write_xml(prefix("baseline") / URDF_REL, baseline_robot)
    current_world = ET.parse(prefix("latest") / WORLD_REL).getroot()
    old_world = ET.fromstring(git_blob("ros2_ws/src/so101_gazebo/worlds/empty.sdf"))
    selected = current_world.find("world")
    for child in list(selected):
        if child.tag in {"scene", "light"}:
            selected.remove(child)
    for child in old_world.find("world"):
        if child.tag in {"scene", "light"}:
            selected.append(copy.deepcopy(child))
    write_xml(prefix("baseline") / WORLD_REL, current_world)
    checks = {}
    for tag in ("joint", "material", "ros2_control"):
        checks[tag] = [canonical(e) for e in baseline_robot.findall(tag)] == [canonical(e) for e in latest_robot.findall(tag)]
    for tag in ("visual", "inertial"):
        checks[tag] = {l.get("name"): [canonical(e) for e in l.findall(tag)] for l in baseline_robot.findall("link")} == {l.get("name"): [canonical(e) for e in l.findall(tag)] for l in latest_robot.findall("link")}
    assert all(checks.values()), checks
    after = immutable_inputs()
    assert before == after, "Production installation or user GUI configuration changed during build"
    manifest = {
        "schema_version": 1, "historical_asset_revision": BASELINE,
        "current_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "definition": "Reconstructed legacy collision/camera/lighting assets under matched current runtime, scene and contact instrumentation; not the historical software stack.",
        "contact_semantics": "Baseline observes whole original jaws; latest observes functional finger collisions. Both have two contact sensors and the same evidence plugin, but their contact geometry semantics differ.",
        "common_assets": ["current table and goal location", "current target model/config/manager", "current ground material", "current controller config", "current native/custom plugins", "current launch with common benchmark GUI override"],
        "structural_equality": checks, "production_input_hashes": before,
        "linked_assets_read_only_by_convention": ["per-variant meshes symlink", "per-variant lib symlink; do not write through these"],
        "gui": gui, "variants": {},
        "launch_environment": {"AMENT_PREFIX_PATH": "<variant prefix>:$AMENT_PREFIX_PATH", "SO101_BENCHMARK_GUI_CONFIG": str(OUT / "gui.config"), "SO101_ENVIRONMENT_RANDOMIZATION_ENABLED": "false", "ROS_DOMAIN_ID": "unique per world", "IGN_PARTITION": "unique per world", "GZ_PARTITION": "same private value as IGN_PARTITION"},
    }
    for variant in ("baseline", "latest"):
        manifest["variants"][variant] = {"prefix": str(prefix(variant)), "contact_sdf_verification": {"status": "pending", "must_verify_before_launch": True}}
    refresh_manifest(manifest)
    print(json.dumps({"manifest": str(OUT / "asset_manifest.json"), "production_unchanged": True, "contact_sdf_verification": "pending"}, indent=2))


def verify_sdf(variant: str, sdf_path: Path, bind: bool) -> None:
    """Validate converted SDF; optionally bind baseline Xacro to resolved names.

    Caller expands the Xacro and runs `ign sdf -p` (parser only), then passes
    its saved SDF here. After --bind, regenerate SDF and call without --bind.
    """
    manifest_path = OUT / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    root = ET.parse(sdf_path).getroot()
    records = []
    for name, mapping in CONTACTS.items():
        matches = [(link, sensor) for link in root.findall(".//link")
                   for sensor in link.findall("sensor") if sensor.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one converted sensor {name}; found {len(matches)}")
        link, sensor = matches[0]
        actual_names = [c.get("name") for c in link.findall("collision")]
        requested = sensor.findtext("contact/collision")
        if variant == "baseline":
            candidates = [n for n in actual_names if mapping["marker"] in n]
            if len(candidates) != 1:
                raise ValueError(f"Cannot resolve original jaw collision for {name}: {actual_names}")
            resolved = candidates[0]
        else:
            resolved = requested
        if resolved not in actual_names:
            raise ValueError(f"Contact target is absent on sensor parent link: {name} -> {resolved}")
        if not bind and requested != resolved:
            raise ValueError(f"Contact needs rebinding: {requested} -> {resolved}; use --bind then regenerate SDF")
        records.append({"sensor": name, "parent_link": link.get("name"), "requested": requested, "resolved": resolved, "valid": requested == resolved})
    if bind:
        if variant != "baseline":
            raise ValueError("Only the reconstructed baseline can be rebound")
        path = prefix(variant) / URDF_REL
        robot = ET.parse(path).getroot()
        for record in records:
            sensor = robot.find(f".//sensor[@name='{record['sensor']}']")
            sensor.find("contact/collision").text = record["resolved"]
        write_xml(path, robot)
    manifest["variants"][variant]["contact_sdf_verification"] = {
        "status": "rebound_needs_regeneration" if bind else "verified",
        "must_verify_before_launch": bind, "sdf": str(sdf_path.resolve()), "sdf_sha256": sha(sdf_path),
        "urdf_sha256": sha(prefix(variant) / URDF_REL), "sensors": records,
    }
    refresh_manifest(manifest)
    print(json.dumps(manifest["variants"][variant]["contact_sdf_verification"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    verify = commands.add_parser("verify-sdf")
    verify.add_argument("--variant", choices=("baseline", "latest"), required=True)
    verify.add_argument("--sdf", type=Path, required=True)
    verify.add_argument("--bind", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        verify_sdf(args.variant, args.sdf, args.bind)


if __name__ == "__main__":
    main()
