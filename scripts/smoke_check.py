#!/usr/bin/env python3
"""Validate the publication draft offline without importing runtime modules."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_mesh(reference, path):
    prefix = "package://so101_description/"
    if reference.startswith(prefix):
        workspace = "ros2_ws" if "ros2_ws" in path.relative_to(ROOT).parts else "ros"
        return ROOT / workspace / "src/so101_description" / reference[len(prefix):]
    if reference.startswith("file://"):
        return Path(reference[7:])
    return path.parent / reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-meshes", action="store_true")
    args = parser.parse_args()
    counts = {"python": 0, "xml": 0, "mesh_references": 0}
    # Walk only source directories; never scan a local Python installation.
    excluded = {".git", ".venv", "__pycache__", "outputs", "build", "install", "log", ".pytest_cache"}
    import os
    files = []
    for parent, directories, names in os.walk(ROOT):
        directories[:] = [d for d in directories if d not in excluded]
        files.extend(Path(parent) / name for name in names)
    for path in (p for p in files if p.suffix == ".py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        counts["python"] += 1
    paths = [p for p in files if p.suffix in {".urdf", ".sdf", ".xacro"} or p.name == "package.xml"]
    missing = []
    for path in paths:
        if "outputs" in path.relative_to(ROOT).parts:
            continue
        tree = ET.parse(path)
        counts["xml"] += 1
        for mesh in tree.findall(".//mesh"):
            ref = mesh.get("filename") or mesh.findtext("uri")
            if ref:
                target = resolve_mesh(ref, path)
                if not target.is_file():
                    missing.append({"reference": ref, "source": str(path.relative_to(ROOT))})
                counts["mesh_references"] += 1
    data = json.loads((ROOT / "configs/reported_results.json").read_text())
    if data["seen"]["successes"] != 128 or data["seen"]["trials"] != 150:
        raise RuntimeError("Unexpected seen-position result fixture")
    for result in data["lighting"]:
        if not 0 <= result["successes"] <= result["trials"]:
            raise RuntimeError("Invalid trial count")
    required = ["README.md", "THIRD_PARTY_NOTICES.md", "docs/source_manifest.json",
                "assets/experiment_results.png", "assets/simulation_results.png", "assets/system_pipeline.png"]
    for name in required:
        if not (ROOT / name).is_file():
            raise RuntimeError(f"Missing deliverable {name}")
    if args.require_meshes and missing:
        raise RuntimeError(f"Missing {len(missing)} external mesh references; first: {missing[0]}")
    print(json.dumps({"status": "PASS", **counts, "missing_external_mesh_references": len(missing),
                      "runtime_assets_complete": not missing, "hardware_connected": False}, indent=2))


if __name__ == "__main__":
    main()
