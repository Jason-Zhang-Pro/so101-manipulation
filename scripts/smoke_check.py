#!/usr/bin/env python3
"""Validate the publication draft offline without importing runtime modules."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_mesh(reference, path):
    prefix = "package://so101_description/"
    if reference.startswith(prefix):
        return ROOT / "ros/src/so101_description" / reference[len(prefix):]
    if reference.startswith("file://"):
        return Path(reference[7:])
    return path.parent / reference


def main():
    counts = {"python": 0, "xml": 0, "mesh_references": 0}
    for path in ROOT.rglob("*.py"):
        if "outputs" in path.relative_to(ROOT).parts:
            continue
        compile(path.read_text(), str(path), "exec")
        counts["python"] += 1
    paths = [*ROOT.rglob("*.urdf"), *ROOT.rglob("*.sdf"), *ROOT.rglob("*.xacro"), *ROOT.rglob("package.xml")]
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
                    raise RuntimeError(f"Missing mesh {ref} in {path.relative_to(ROOT)}")
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
    print(json.dumps({"status": "PASS", **counts, "hardware_connected": False}, indent=2))


if __name__ == "__main__":
    main()
