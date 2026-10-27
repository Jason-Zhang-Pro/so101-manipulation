#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET


DEFAULT_ROOT = Path("/media/tiger/New Volume/ubuntufile/capstone")
XACRO_REL = Path("ros2_ws/src/so101_description/urdf/so101.urdf.xacro")
MESH_ROOT_REL = Path("ros2_ws/src/so101_description/meshes")
DEST_SUBDIR = Path("collision_generated_ultralow")

# Keep the separately validated camera-mount collision unchanged during the
# first solver stress test. Camera body/lens are already primitives.
DEFAULT_SKIPS = {"gripper_camera_mount.stl"}

PACKAGE_PREFIX = "package://so101_description/meshes/"
COLLISION_BLOCK_RE = re.compile(
    r"(?P<indent>^[ \t]*)<collision(?P<attrs>[^>]*)>"
    r"(?P<body>.*?)"
    r"^[ \t]*</collision>",
    re.MULTILINE | re.DOTALL,
)
MESH_FILENAME_RE = re.compile(
    r'(<mesh\b[^>]*\bfilename=")'
    r'(?P<filename>[^"]+)'
    r'(")',
    re.DOTALL,
)
VISUAL_BLOCK_RE = re.compile(
    r"<visual(?:\s[^>]*)?>.*?</visual>",
    re.DOTALL,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Install generated ultra-low SO-101 collision proxy meshes and "
            "patch only collision mesh references in the source Xacro."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        required=True,
        help=(
            "Final proxy directory containing meshes/ and "
            "collision_replacement_manifest.json."
        ),
    )
    parser.add_argument(
        "--include-camera-mount",
        action="store_true",
        help=(
            "Also replace gripper_camera_mount.stl collision. Default is to "
            "keep it unchanged for the first stress test."
        ),
    )
    return parser.parse_args()


def collision_mesh_uri(block: str) -> str | None:
    match = MESH_FILENAME_RE.search(block)
    if match is None:
        return None
    return match.group("filename")


def basename_from_package_uri(uri: str) -> str | None:
    if not uri.startswith(PACKAGE_PREFIX):
        return None
    return Path(uri[len(PACKAGE_PREFIX):]).name


def replace_mesh_filename(block: str, new_uri: str) -> str:
    matches = list(MESH_FILENAME_RE.finditer(block))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one <mesh filename=...> in collision block; "
            f"found {len(matches)}."
        )
    match = matches[0]
    return (
        block[:match.start("filename")]
        + new_uri
        + block[match.end("filename"):]
    )


def add_generated_name(block: str, suffix: str) -> str:
    opening = re.match(
        r"(?P<indent>^[ \t]*)<collision(?P<attrs>[^>]*)>",
        block,
        re.MULTILINE,
    )
    if opening is None:
        raise RuntimeError("Could not parse collision opening tag.")

    attrs = opening.group("attrs")
    name_match = re.search(r'\bname="([^"]*)"', attrs)
    if name_match is not None:
        old_name = name_match.group(1)
        new_attrs = (
            attrs[:name_match.start(1)]
            + f"{old_name}_{suffix}"
            + attrs[name_match.end(1):]
        )
    else:
        new_attrs = attrs + f' name="generated_{suffix}"'

    return (
        block[:opening.start("attrs")]
        + new_attrs
        + block[opening.end("attrs"):]
    )


def count_visual_meshes(text: str) -> Counter:
    counter = Counter()
    for block in VISUAL_BLOCK_RE.findall(text):
        for match in MESH_FILENAME_RE.finditer(block):
            counter[match.group("filename")] += 1
    return counter


def count_collision_meshes(text: str) -> Counter:
    counter = Counter()
    for match in COLLISION_BLOCK_RE.finditer(text):
        uri = collision_mesh_uri(match.group(0))
        if uri is not None:
            counter[uri] += 1
    return counter


def patch_xacro(
    original_text: str,
    manifest: dict,
    skipped_basenames: set[str],
):
    generated = manifest.get("generated_meshes")
    if not isinstance(generated, dict) or not generated:
        raise RuntimeError(
            "Manifest has no non-empty generated_meshes mapping."
        )

    replaced_use_count = Counter()
    replacement_part_count = Counter()

    pieces = []
    cursor = 0

    for match in COLLISION_BLOCK_RE.finditer(original_text):
        block = match.group(0)
        uri = collision_mesh_uri(block)
        basename = basename_from_package_uri(uri) if uri else None

        pieces.append(original_text[cursor:match.start()])

        if (
            basename is None
            or basename in skipped_basenames
            or basename not in generated
        ):
            pieces.append(block)
            cursor = match.end()
            continue

        record = generated[basename]
        proxy_files = record.get("proxy_files")
        if not isinstance(proxy_files, list) or not proxy_files:
            raise RuntimeError(
                f"{basename}: manifest proxy_files is empty."
            )

        replacement_blocks = []
        for part_index, proxy_filename in enumerate(proxy_files):
            new_uri = (
                PACKAGE_PREFIX
                + DEST_SUBDIR.as_posix()
                + "/"
                + proxy_filename
            )
            new_block = replace_mesh_filename(block, new_uri)

            # Multi-part collision entries must have unique names when the
            # source entry was named. For unnamed entries, also assign unique
            # generated names to make generated URDF inspection unambiguous.
            if len(proxy_files) > 1:
                stem = Path(basename).stem
                new_block = add_generated_name(
                    new_block,
                    f"{stem}_part_{part_index:02d}",
                )

            replacement_blocks.append(new_block)

        pieces.append("\n".join(replacement_blocks))
        cursor = match.end()

        replaced_use_count[basename] += 1
        replacement_part_count[basename] = len(proxy_files)

    pieces.append(original_text[cursor:])
    return (
        "".join(pieces),
        replaced_use_count,
        replacement_part_count,
    )


def validate_manifest_files(proxy_dir: Path, manifest: dict):
    generated = manifest.get("generated_meshes", {})
    mesh_dir = proxy_dir / "meshes"
    missing = []
    for basename, record in generated.items():
        for filename in record.get("proxy_files", []):
            path = mesh_dir / filename
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing generated proxy files:\n  "
            + "\n  ".join(missing)
        )


def main():
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    proxy_dir = args.proxy_dir.expanduser().resolve()
    xacro = root / XACRO_REL
    mesh_root = root / MESH_ROOT_REL
    manifest_path = proxy_dir / "collision_replacement_manifest.json"

    if not xacro.is_file():
        raise FileNotFoundError(xacro)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not (proxy_dir / "meshes").is_dir():
        raise FileNotFoundError(proxy_dir / "meshes")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_files(proxy_dir, manifest)

    skipped = set()
    if not args.include_camera_mount:
        skipped.update(DEFAULT_SKIPS)

    original_text = xacro.read_text(encoding="utf-8")
    original_visuals = count_visual_meshes(original_text)
    original_collisions = count_collision_meshes(original_text)

    patched_text, replaced_uses, part_counts = patch_xacro(
        original_text,
        manifest,
        skipped,
    )

    if not replaced_uses:
        raise RuntimeError(
            "No collision mesh references matched the manifest. "
            "Nothing was changed."
        )

    # Strong invariant: visual mesh references must remain byte-for-byte
    # unchanged as a multiset.
    patched_visuals = count_visual_meshes(patched_text)
    if patched_visuals != original_visuals:
        raise RuntimeError(
            "Safety check failed: visual mesh references changed."
        )

    # Ensure skipped source collision remains untouched.
    patched_collisions = count_collision_meshes(patched_text)
    for basename in skipped:
        source_uri = PACKAGE_PREFIX + basename
        if patched_collisions[source_uri] != original_collisions[source_uri]:
            raise RuntimeError(
                f"Safety check failed: skipped collision {basename} changed."
            )

    # Ensure every replaced source URI disappeared from collision geometry.
    for basename in replaced_uses:
        source_uri = PACKAGE_PREFIX + basename
        if patched_collisions[source_uri] != 0:
            raise RuntimeError(
                f"Safety check failed: original collision URI remains: "
                f"{source_uri}"
            )

    # Parse as XML before touching project source.
    ET.fromstring(patched_text)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(
        f"/tmp/so101_collision_apply_backup_{timestamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(xacro, backup_dir / xacro.name)

    dest_dir = mesh_root / DEST_SUBDIR
    if dest_dir.exists():
        shutil.copytree(
            dest_dir,
            backup_dir / DEST_SUBDIR.name,
        )
        shutil.rmtree(dest_dir)

    shutil.copytree(proxy_dir / "meshes", dest_dir)

    # Keep the manifest with the installed collision assets.
    shutil.copy2(
        manifest_path,
        dest_dir / "collision_replacement_manifest.json",
    )

    temp_xacro = xacro.with_suffix(
        xacro.suffix + ".collision_proxy_tmp"
    )
    temp_xacro.write_text(patched_text, encoding="utf-8")
    temp_xacro.replace(xacro)

    # Re-parse the on-disk source after atomic replacement.
    ET.parse(xacro)

    print("SO-101 ultra-low collision proxies installed.")
    print(f"Source Xacro: {xacro}")
    print(f"Installed proxy directory: {dest_dir}")
    print(f"Backup: {backup_dir}")
    print()
    print("Replaced collision mesh uses:")
    for basename in sorted(replaced_uses):
        print(
            f"  {basename}: uses={replaced_uses[basename]}, "
            f"parts/use={part_counts[basename]}"
        )

    if skipped:
        print()
        print("Intentionally unchanged in this first stress test:")
        for basename in sorted(skipped):
            print(f"  {basename}")

    generated_uri_count = sum(
        count
        for uri, count in patched_collisions.items()
        if uri.startswith(
            PACKAGE_PREFIX + DEST_SUBDIR.as_posix() + "/"
        )
    )

    print()
    print(
        f"Generated collision mesh references in Xacro: "
        f"{generated_uri_count}"
    )
    print(
        f"Visual mesh references unchanged: "
        f"{sum(original_visuals.values())}"
    )
    print("No joint, inertial, controller, reward, policy, or visual change.")
    print("No git add/commit was performed.")
    print()
    print("Next: rebuild so101_description before launching Gazebo.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
