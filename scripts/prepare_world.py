#!/usr/bin/env python3
"""Materialize world assets and current checkout paths for local Gazebo use."""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/runtime_worlds")
    args = parser.parse_args()
    dest = args.output.expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f"Choose a new output directory: {dest}")
    shutil.copytree(ROOT / "simulation/worlds", dest)
    controller = ROOT / "simulation/config/controllers.yaml"
    for name in ("so101.sdf", "so101.urdf"):
        path = dest / name
        tree = ET.parse(path)
        for node in tree.findall(".//parameters"):
            node.text = str(controller)
        tree.write(path, encoding="utf-8", xml_declaration=True)
    path = dest / "grasp_world.sdf"
    tree = ET.parse(path)
    for node in tree.findall(".//include/uri"):
        if node.text == "so101.sdf":
            node.text = (dest / "so101.sdf").as_uri()
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(dest / "grasp_world.sdf")


if __name__ == "__main__":
    main()
