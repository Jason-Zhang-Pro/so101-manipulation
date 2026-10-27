#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_PROJECT_ROOT = Path("/media/tiger/New Volume/ubuntufile/capstone")
PACKAGE_PREFIX = "package://so101_description/meshes/"
FIXED_MESH = "wrist_roll_follower_so101_v1.stl"
MOVING_MESH = "moving_jaw_so101_v1.stl"
CUBE_SIZE_M = 0.03


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render the assembled SO-101 gripper using the original visual "
            "meshes, generated ultra-low collision proxies, and a 30 mm cube."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=None,
        help=(
            "Collision-proxy generation directory. If omitted, automatically "
            "uses the newest /tmp/so101_collision_ultralow_* directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <proxy-dir>/gripper_assembled_preview"
        ),
    )
    parser.add_argument(
        "--cube-size-mm",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--contact-tolerance-mm",
        type=float,
        default=0.35,
        help=(
            "Approximate proxy-to-cube distance used to identify the first "
            "moving-jaw contact while searching from open toward closed."
        ),
    )
    parser.add_argument(
        "--search-samples",
        type=int,
        default=701,
    )
    return parser.parse_args()


def newest_proxy_dir() -> Path:
    candidates = [
        path
        for path in Path("/tmp").glob("so101_collision_ultralow_*")
        if path.is_dir()
        and (path / "collision_replacement_manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No /tmp/so101_collision_ultralow_* directory with a manifest "
            "was found. Pass --proxy-dir explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {text!r}")
    return np.asarray(values, dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]],
        dtype=float,
    )
    ry = np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]],
        dtype=float,
    )
    rz = np.array(
        [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]],
        dtype=float,
    )
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("Joint axis cannot be zero.")
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one = 1.0 - c
    return np.array(
        [
            [c + x*x*one, x*y*one - z*s, x*z*one + y*s],
            [y*x*one + z*s, c + y*y*one, y*z*one - x*s],
            [z*x*one - y*s, z*y*one + x*s, c + z*z*one],
        ],
        dtype=float,
    )


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rpy_matrix(np.asarray(rpy, dtype=float))
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    return matrix


def joint_transform(origin_xyz, origin_rpy, axis, q) -> np.ndarray:
    base = transform(origin_xyz, origin_rpy)
    motion = np.eye(4)
    motion[:3, :3] = axis_angle_matrix(axis, q)
    return base @ motion


def parse_origin(element: ET.Element | None):
    if element is None:
        return np.zeros(3), np.zeros(3)
    return vec(element.get("xyz")), vec(element.get("rpy"))


def mesh_entry(link: ET.Element, mesh_basename: str, kind: str):
    for element in link.findall(f"./{kind}"):
        mesh = element.find("./geometry/mesh")
        if mesh is None:
            continue
        filename = mesh.get("filename", "")
        if Path(filename).name != mesh_basename:
            continue
        xyz, rpy = parse_origin(element.find("./origin"))
        scale_text = mesh.get("scale")
        scale = (
            vec(scale_text, default=(1, 1, 1))
            if scale_text
            else np.ones(3)
        )
        return {
            "filename": filename,
            "origin_xyz": xyz,
            "origin_rpy": rpy,
            "scale": scale,
        }
    raise RuntimeError(
        f"Could not find {kind} mesh {mesh_basename!r} "
        f"in link {link.get('name')!r}."
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise RuntimeError(f"No meshes in {path}")
        loaded = trimesh.util.concatenate(meshes)
    loaded.remove_unreferenced_vertices()
    return loaded


def apply_scale(mesh: trimesh.Trimesh, scale: np.ndarray):
    result = mesh.copy()
    matrix = np.eye(4)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale
    result.apply_transform(matrix)
    return result


def transformed(mesh: trimesh.Trimesh, matrix: np.ndarray):
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def manifest_proxy_files(proxy_dir: Path, mesh_name: str) -> list[Path]:
    manifest_path = proxy_dir / "collision_replacement_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["generated_meshes"][mesh_name]
    paths = [
        proxy_dir / "meshes" / filename
        for filename in record["proxy_files"]
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def source_mesh_path(project_root: Path, uri: str) -> Path:
    if not uri.startswith(PACKAGE_PREFIX):
        raise ValueError(f"Unexpected mesh URI: {uri}")
    return (
        project_root
        / "ros2_ws"
        / "src"
        / "so101_description"
        / "meshes"
        / uri[len(PACKAGE_PREFIX):]
    )


def parse_model(project_root: Path):
    xacro = (
        project_root
        / "ros2_ws"
        / "src"
        / "so101_description"
        / "urdf"
        / "so101.urdf.xacro"
    )
    root = ET.parse(xacro).getroot()

    gripper = root.find("./link[@name='gripper_link']")
    moving = root.find("./link[@name='moving_jaw_so101_v1_link']")
    if gripper is None or moving is None:
        raise RuntimeError("Required gripper links are missing.")

    fixed_visual = mesh_entry(gripper, FIXED_MESH, "visual")
    fixed_collision = mesh_entry(gripper, FIXED_MESH, "collision")
    moving_visual = mesh_entry(moving, MOVING_MESH, "visual")
    moving_collision = mesh_entry(moving, MOVING_MESH, "collision")

    gripper_joint = root.find("./joint[@name='gripper']")
    if gripper_joint is None:
        raise RuntimeError("gripper joint is missing.")
    j_xyz, j_rpy = parse_origin(gripper_joint.find("./origin"))
    axis_element = gripper_joint.find("./axis")
    axis = vec(
        axis_element.get("xyz") if axis_element is not None else "1 0 0"
    )
    limit = gripper_joint.find("./limit")
    if limit is None:
        raise RuntimeError("gripper joint limit is missing.")
    lower = float(limit.get("lower"))
    upper = float(limit.get("upper"))

    frame_joint = root.find("./joint[@name='gripper_frame_joint']")
    if frame_joint is None:
        raise RuntimeError("gripper_frame_joint is missing.")
    frame_xyz, frame_rpy = parse_origin(frame_joint.find("./origin"))

    return {
        "xacro": xacro,
        "fixed_visual": fixed_visual,
        "fixed_collision": fixed_collision,
        "moving_visual": moving_visual,
        "moving_collision": moving_collision,
        "joint_xyz": j_xyz,
        "joint_rpy": j_rpy,
        "joint_axis": axis,
        "joint_lower": lower,
        "joint_upper": upper,
        "frame_xyz": frame_xyz,
        "frame_rpy": frame_rpy,
    }


def build_geometry(project_root: Path, proxy_dir: Path, model: dict):
    fixed_visual = apply_scale(
        load_mesh(
            source_mesh_path(
                project_root,
                model["fixed_visual"]["filename"],
            )
        ),
        model["fixed_visual"]["scale"],
    )
    fixed_visual_local = transformed(
        fixed_visual,
        transform(
            model["fixed_visual"]["origin_xyz"],
            model["fixed_visual"]["origin_rpy"],
        ),
    )

    moving_visual = apply_scale(
        load_mesh(
            source_mesh_path(
                project_root,
                model["moving_visual"]["filename"],
            )
        ),
        model["moving_visual"]["scale"],
    )
    moving_visual_link_local = transformed(
        moving_visual,
        transform(
            model["moving_visual"]["origin_xyz"],
            model["moving_visual"]["origin_rpy"],
        ),
    )

    fixed_proxies = []
    for proxy_path in manifest_proxy_files(proxy_dir, FIXED_MESH):
        proxy = apply_scale(
            load_mesh(proxy_path),
            model["fixed_collision"]["scale"],
        )
        proxy = transformed(
            proxy,
            transform(
                model["fixed_collision"]["origin_xyz"],
                model["fixed_collision"]["origin_rpy"],
            ),
        )
        fixed_proxies.append(proxy)

    moving_proxies_link_local = []
    for proxy_path in manifest_proxy_files(proxy_dir, MOVING_MESH):
        proxy = apply_scale(
            load_mesh(proxy_path),
            model["moving_collision"]["scale"],
        )
        proxy = transformed(
            proxy,
            transform(
                model["moving_collision"]["origin_xyz"],
                model["moving_collision"]["origin_rpy"],
            ),
        )
        moving_proxies_link_local.append(proxy)

    return {
        "fixed_visual": fixed_visual_local,
        "moving_visual_link": moving_visual_link_local,
        "fixed_proxies": fixed_proxies,
        "moving_proxies_link": moving_proxies_link_local,
    }


def moving_parent_transform(model: dict, q: float):
    return joint_transform(
        model["joint_xyz"],
        model["joint_rpy"],
        model["joint_axis"],
        q,
    )


def moving_at(mesh, model, q):
    return transformed(mesh, moving_parent_transform(model, q))


def cube_mesh(center: np.ndarray, size: float):
    cube = trimesh.creation.box(extents=(size, size, size))
    cube.apply_translation(center)
    return cube


def point_to_aabb_distance(points: np.ndarray, center, half):
    delta = np.abs(points - center[None, :]) - half
    outside = np.maximum(delta, 0.0)
    return np.linalg.norm(outside, axis=1)


def sampled_points(mesh: trimesh.Trimesh):
    # Vertices plus triangle centroids provide a deterministic and dependency-
    # free approximation for contact search. This is diagnostic only.
    vertices = mesh.vertices
    centroids = mesh.triangles_center
    return np.vstack([vertices, centroids])


def proxy_cube_distance(parts, cube_center, cube_size):
    half = cube_size * 0.5
    minimum = math.inf
    for part in parts:
        distances = point_to_aabb_distance(
            sampled_points(part),
            cube_center,
            half,
        )
        minimum = min(minimum, float(distances.min()))
    return minimum


def find_first_contact_q(model, geometry, cube_center, cube_size, tolerance):
    # Project convention: larger gripper joint angle is opening, lower angle
    # is closing. Search from upper limit toward lower limit.
    upper = model["joint_upper"]
    lower = model["joint_lower"]

    qs = np.linspace(upper, lower, 701)
    best_q = upper
    best_distance = math.inf
    first_contact = None

    for q in qs:
        parts = [
            moving_at(part, model, float(q))
            for part in geometry["moving_proxies_link"]
        ]
        distance = proxy_cube_distance(
            parts,
            cube_center,
            cube_size,
        )
        if distance < best_distance:
            best_distance = distance
            best_q = float(q)
        if distance <= tolerance:
            first_contact = (float(q), distance)
            break

    return first_contact, (best_q, best_distance)


def face_subset(mesh: trimesh.Trimesh, max_faces=3000):
    faces = mesh.faces
    if len(faces) <= max_faces:
        return mesh.vertices[faces]
    indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
    return mesh.vertices[faces[indices]]


def add_mesh(ax, mesh, *, facecolor, edgecolor, alpha, linewidth, max_faces=3000):
    collection = Poly3DCollection(
        face_subset(mesh, max_faces=max_faces),
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.add_collection3d(collection)


def plot_pose(
    model,
    geometry,
    cube,
    q,
    output_path,
    label,
    distance_mm,
):
    moving_visual = moving_at(
        geometry["moving_visual_link"],
        model,
        q,
    )
    moving_proxies = [
        moving_at(part, model, q)
        for part in geometry["moving_proxies_link"]
    ]

    all_meshes = [
        geometry["fixed_visual"],
        moving_visual,
        cube,
        *geometry["fixed_proxies"],
        *moving_proxies,
    ]
    points = np.vstack([mesh.vertices for mesh in all_meshes])
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = max(float((maximum - minimum).max()), 0.04) * 0.60

    fig = plt.figure(figsize=(14, 11))
    views = [
        ("isometric", 22, 38),
        ("front", 0, -90),
        ("side", 0, 0),
        ("top", 90, -90),
    ]

    for index, (view_name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")

        # Dark original visual mesh.
        add_mesh(
            ax,
            geometry["fixed_visual"],
            facecolor=(0.14, 0.14, 0.14),
            edgecolor=(0.04, 0.04, 0.04),
            alpha=0.30,
            linewidth=0.16,
        )
        add_mesh(
            ax,
            moving_visual,
            facecolor=(0.22, 0.22, 0.22),
            edgecolor=(0.04, 0.04, 0.04),
            alpha=0.30,
            linewidth=0.16,
        )

        # Collision proxies: fixed red, moving blue.
        for part in geometry["fixed_proxies"]:
            add_mesh(
                ax,
                part,
                facecolor=(0.90, 0.12, 0.10),
                edgecolor=(0.65, 0.02, 0.02),
                alpha=0.26,
                linewidth=0.45,
            )
        for part in moving_proxies:
            add_mesh(
                ax,
                part,
                facecolor=(0.10, 0.35, 0.90),
                edgecolor=(0.02, 0.12, 0.65),
                alpha=0.28,
                linewidth=0.45,
            )

        # 30 mm cube.
        add_mesh(
            ax,
            cube,
            facecolor=(0.10, 0.70, 0.20),
            edgecolor=(0.02, 0.35, 0.05),
            alpha=0.38,
            linewidth=0.80,
            max_faces=100,
        )

        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

    fig.suptitle(
        f"{label}\n"
        f"gripper joint = {math.degrees(q):.2f} deg, "
        f"moving-proxy to cube approx. distance = {distance_mm:.2f} mm\n"
        "dark grey = original visual, red = fixed proxy, "
        "blue = moving proxy, green = 30 mm cube"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def write_html(output_dir: Path, poses):
    rows = []
    for pose in poses:
        rows.append(
            "<tr>"
            f"<td>{pose['label']}</td>"
            f"<td>{pose['q_deg']:.2f}°</td>"
            f"<td>{pose['distance_mm']:.3f} mm</td>"
            f"<td><a href=\"{pose['file']}\">open preview</a></td>"
            "</tr>"
        )
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SO-101 assembled gripper collision preview</title>
<style>
body { font-family: sans-serif; margin: 28px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #bbb; padding: 7px; text-align: left; }
th { background: #eee; }
</style>
</head>
<body>
<h1>SO-101 assembled gripper collision preview</h1>
<p>
Dark grey = original visual meshes; red = fixed-jaw collision proxy;
blue = moving-jaw collision proxy; green = 30 mm cube.
</p>
<p>
The cube centre is the current URDF <code>gripper_frame_link</code> origin
expressed in <code>gripper_link</code>. This is a geometric inspection,
not a Gazebo physics simulation.
</p>
<table>
<thead><tr><th>Pose</th><th>Joint</th><th>Approx. moving-proxy gap</th><th>Preview</th></tr></thead>
<tbody>
""" + "\n".join(rows) + """
</tbody></table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    proxy_dir = (
        args.proxy_dir.resolve()
        if args.proxy_dir is not None
        else newest_proxy_dir()
    )
    if not (proxy_dir / "collision_replacement_manifest.json").is_file():
        raise FileNotFoundError(
            proxy_dir / "collision_replacement_manifest.json"
        )

    output_dir = (
        args.output.resolve()
        if args.output is not None
        else proxy_dir / "gripper_assembled_preview"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = parse_model(project_root)
    geometry = build_geometry(project_root, proxy_dir, model)

    cube_size = args.cube_size_mm / 1000.0
    # The current gripper_frame_joint is fixed directly under gripper_link,
    # so its translation is the cube reference centre in gripper_link.
    cube_center = model["frame_xyz"].copy()
    cube = cube_mesh(cube_center, cube_size)

    fixed_distance = proxy_cube_distance(
        geometry["fixed_proxies"],
        cube_center,
        cube_size,
    )

    tolerance = args.contact_tolerance_mm / 1000.0

    # Respect the user-selected search density.
    global np
    qs = np.linspace(
        model["joint_upper"],
        model["joint_lower"],
        max(3, int(args.search_samples)),
    )
    first_contact = None
    best_q = model["joint_upper"]
    best_distance = math.inf

    for q in qs:
        moving_parts = [
            moving_at(part, model, float(q))
            for part in geometry["moving_proxies_link"]
        ]
        distance = proxy_cube_distance(
            moving_parts,
            cube_center,
            cube_size,
        )
        if distance < best_distance:
            best_q = float(q)
            best_distance = distance
        if distance <= tolerance:
            first_contact = (float(q), distance)
            break

    if first_contact is None:
        grasp_q = best_q
        grasp_distance = best_distance
        grasp_label = "closest-to-cube pose (no tolerance contact found)"
    else:
        grasp_q, grasp_distance = first_contact
        grasp_label = "auto grasp: first proxy contact while closing"

    q_open = model["joint_upper"]
    q_45 = min(
        max(math.radians(45.0), model["joint_lower"]),
        model["joint_upper"],
    )

    pose_specs = [
        ("fully_open", "fully open", q_open),
        ("45deg", "45 degree gripper joint", q_45),
        ("auto_grasp", grasp_label, grasp_q),
    ]

    poses = []
    for filename_stem, label, q in pose_specs:
        moving_parts = [
            moving_at(part, model, q)
            for part in geometry["moving_proxies_link"]
        ]
        distance = proxy_cube_distance(
            moving_parts,
            cube_center,
            cube_size,
        )
        filename = f"{filename_stem}.png"
        plot_pose(
            model,
            geometry,
            cube,
            q,
            output_dir / filename,
            label,
            distance * 1000.0,
        )
        poses.append(
            {
                "label": label,
                "q_rad": q,
                "q_deg": math.degrees(q),
                "distance_mm": distance * 1000.0,
                "file": filename,
            }
        )

    report = {
        "project_root": str(project_root),
        "source_xacro": str(model["xacro"]),
        "proxy_dir": str(proxy_dir),
        "output_dir": str(output_dir),
        "cube_size_mm": args.cube_size_mm,
        "cube_center_in_gripper_link_m": cube_center.tolist(),
        "cube_reference": "gripper_frame_link origin",
        "gripper_joint": {
            "lower_rad": model["joint_lower"],
            "upper_rad": model["joint_upper"],
            "lower_deg": math.degrees(model["joint_lower"]),
            "upper_deg": math.degrees(model["joint_upper"]),
            "origin_xyz": model["joint_xyz"].tolist(),
            "origin_rpy": model["joint_rpy"].tolist(),
            "axis": model["joint_axis"].tolist(),
        },
        "fixed_proxy_to_cube_approx_distance_mm": fixed_distance * 1000.0,
        "contact_tolerance_mm": args.contact_tolerance_mm,
        "poses": poses,
        "notes": [
            "This tool does not modify the project or proxy meshes.",
            "Approximate gap uses proxy vertices + triangle centroids versus "
            "the cube AABB; use the rendered overlay for final judgement.",
            "Auto-grasp is the first sampled joint angle, moving from the "
            "URDF upper limit toward the lower limit, whose moving proxy "
            "comes within the configured contact tolerance of the cube.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_html(output_dir, poses)

    print("SO-101 assembled gripper preview generated.")
    print(f"Source Xacro: {model['xacro']}")
    print(f"Proxy directory: {proxy_dir}")
    print(f"Output directory: {output_dir}")
    print(
        "Gripper joint range: "
        f"{math.degrees(model['joint_lower']):.2f} deg .. "
        f"{math.degrees(model['joint_upper']):.2f} deg"
    )
    print(
        "Cube centre (gripper_link frame): "
        + " ".join(f"{value:.6f}" for value in cube_center)
    )
    print(
        "Fixed proxy -> cube approximate gap: "
        f"{fixed_distance * 1000.0:.3f} mm"
    )
    if first_contact is not None:
        print(
            "Auto-grasp first-contact joint: "
            f"{math.degrees(grasp_q):.3f} deg "
            f"(approx gap {grasp_distance * 1000.0:.3f} mm)"
        )
    else:
        print(
            "No moving-proxy contact found within tolerance; "
            "using closest sampled pose: "
            f"{math.degrees(grasp_q):.3f} deg "
            f"(approx gap {grasp_distance * 1000.0:.3f} mm)"
        )

    print(f"Open: xdg-open '{output_dir / 'index.html'}'")
    print(f"Folder: xdg-open '{output_dir}'")
    print("No project source file was modified.")


if __name__ == "__main__":
    main()
