#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import coacd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROJECT_ROOT = Path("/media/tiger/New Volume/ubuntufile/capstone")
PACKAGE_PREFIX = "package://so101_description/meshes/"
FIXED_MESH = "wrist_roll_follower_so101_v1.stl"
CUBE_SIZE = 0.03


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--proxy-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def newest_proxy_dir():
    candidates = [
        p for p in Path("/tmp").glob("so101_collision_ultralow_*")
        if p.is_dir() and (p / "collision_replacement_manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No /tmp/so101_collision_ultralow_* directory with manifest found."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def vec(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return np.asarray(default, dtype=float)
    values = [float(v) for v in text.split()]
    if len(values) != 3:
        raise ValueError(text)
    return np.asarray(values, dtype=float)


def rpy_matrix(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=float)
    ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=float)
    rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=float)
    return rz @ ry @ rx


def transform(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = rpy_matrix(rpy)
    m[:3, 3] = xyz
    return m


def parse_origin(element):
    if element is None:
        return np.zeros(3), np.zeros(3)
    return vec(element.get("xyz")), vec(element.get("rpy"))


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        pieces = [
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not pieces:
            raise RuntimeError(f"No mesh geometry: {path}")
        mesh = trimesh.util.concatenate(pieces)
    mesh.remove_unreferenced_vertices()
    return mesh


def apply_scale(mesh, scale):
    result = mesh.copy()
    m = np.eye(4)
    m[0,0], m[1,1], m[2,2] = scale
    result.apply_transform(m)
    return result


def transformed(mesh, matrix):
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def mesh_points(mesh):
    return np.vstack([mesh.vertices, mesh.triangles_center])


def aabb_distance(points, center, half):
    delta = np.abs(points - center[None, :]) - half
    outside = np.maximum(delta, 0.0)
    return np.linalg.norm(outside, axis=1)


def mesh_cube_gap(mesh, cube_center, cube_size):
    return float(aabb_distance(
        mesh_points(mesh),
        cube_center,
        cube_size * 0.5,
    ).min())


def simplify_convex(mesh, max_vertices):
    # CoACD preprocesses open / non-manifold meshes itself, which is useful
    # here because our functional body/finger split is deliberately an open
    # triangle submesh. This also avoids trimesh.convex_hull -> SciPy.
    cmesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        cmesh,
        threshold=1.0,
        max_convex_hull=1,
        preprocess_mode="auto",
        preprocess_resolution=20,
        resolution=800,
        mcts_nodes=8,
        mcts_iterations=40,
        mcts_max_depth=2,
        merge=True,
        decimate=True,
        max_ch_vertex=int(max_vertices),
        pca=False,
        apx_mode="ch",
        seed=0,
    )
    if not parts:
        raise RuntimeError("CoACD returned no convex hull.")
    vertices, faces = parts[0]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def parse_model(project_root):
    xacro = (
        project_root / "ros2_ws/src/so101_description/urdf/so101.urdf.xacro"
    )
    root = ET.parse(xacro).getroot()

    gripper = root.find("./link[@name='gripper_link']")
    if gripper is None:
        raise RuntimeError("gripper_link not found")

    collision_record = None
    for collision in gripper.findall("./collision"):
        mesh = collision.find("./geometry/mesh")
        if mesh is None:
            continue
        if Path(mesh.get("filename", "")).name == FIXED_MESH:
            xyz, rpy = parse_origin(collision.find("./origin"))
            collision_record = {
                "uri": mesh.get("filename"),
                "scale": vec(mesh.get("scale"), (1,1,1))
                if mesh.get("scale") else np.ones(3),
                "xyz": xyz,
                "rpy": rpy,
            }
            break
    if collision_record is None:
        raise RuntimeError(f"Collision mesh {FIXED_MESH} not found")

    frame_joint = root.find("./joint[@name='gripper_frame_joint']")
    if frame_joint is None:
        raise RuntimeError("gripper_frame_joint not found")
    frame_xyz, frame_rpy = parse_origin(frame_joint.find("./origin"))

    return xacro, collision_record, frame_xyz, frame_rpy


def source_path(project_root, uri):
    if not uri.startswith(PACKAGE_PREFIX):
        raise ValueError(uri)
    return (
        project_root
        / "ros2_ws/src/so101_description/meshes"
        / uri[len(PACKAGE_PREFIX):]
    )


def to_gripper_frame(raw_mesh, record):
    scaled = apply_scale(raw_mesh, record["scale"])
    return transformed(scaled, transform(record["xyz"], record["rpy"]))


def to_raw_local(mesh_gripper, record):
    result = mesh_gripper.copy()
    result.apply_transform(np.linalg.inv(transform(record["xyz"], record["rpy"])))
    inv_scale = np.eye(4)
    inv_scale[0,0] = 1.0 / record["scale"][0]
    inv_scale[1,1] = 1.0 / record["scale"][1]
    inv_scale[2,2] = 1.0 / record["scale"][2]
    result.apply_transform(inv_scale)
    return result


def generate_candidate(original, cube_center, finger_fraction, overlap_fraction):
    points = mesh_points(original)
    centroid = original.vertices.mean(axis=0)
    direction = cube_center - centroid
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise RuntimeError("Cube centre coincides with mesh centroid.")
    direction /= norm

    projection = (points - centroid[None, :]) @ direction
    low = float(projection.min())
    high = float(projection.max())
    span = max(high - low, 1e-9)

    cut = float(np.quantile(projection, 1.0 - finger_fraction))
    overlap = span * overlap_fraction

    # Avoid trimesh.convex_hull here because that path requires SciPy.
    # Instead split the ORIGINAL triangle mesh by triangle-centroid position,
    # then let CoACD itself convert each open / non-manifold submesh into one
    # aggressively simplified convex hull.
    triangle_centers = original.triangles_center
    triangle_projection = (
        triangle_centers - centroid[None, :]
    ) @ direction

    body_face_mask = triangle_projection <= cut + overlap
    finger_face_mask = triangle_projection >= cut - overlap

    body_face_indices = np.flatnonzero(body_face_mask)
    finger_face_indices = np.flatnonzero(finger_face_mask)

    if len(body_face_indices) < 4 or len(finger_face_indices) < 4:
        raise RuntimeError("Candidate split produced too few triangle faces.")

    body = original.submesh(
        [body_face_indices],
        append=True,
        repair=False,
    )
    finger = original.submesh(
        [finger_face_indices],
        append=True,
        repair=False,
    )

    body.remove_unreferenced_vertices()
    finger.remove_unreferenced_vertices()

    body = simplify_convex(body, max_vertices=24)
    finger = simplify_convex(finger, max_vertices=24)

    return body, finger, {
        "finger_fraction": float(finger_fraction),
        "overlap_fraction": float(overlap_fraction),
        "cut_projection": cut,
        "direction": direction.tolist(),
    }


def face_subset(mesh, max_faces=2800):
    faces = mesh.faces
    if len(faces) <= max_faces:
        return mesh.vertices[faces]
    ids = np.linspace(0, len(faces)-1, max_faces, dtype=int)
    return mesh.vertices[faces[ids]]


def add_mesh(ax, mesh, color, alpha, edge, linewidth):
    poly = Poly3DCollection(
        face_subset(mesh),
        facecolor=color,
        edgecolor=edge,
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.add_collection3d(poly)


def cube_mesh(center):
    cube = trimesh.creation.box(extents=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    cube.apply_translation(center)
    return cube


def render_candidate(original, body, finger, cube, output, title):
    all_points = np.vstack([
        original.vertices, body.vertices, finger.vertices, cube.vertices
    ])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2.0
    extent = max(float((all_points.max(axis=0)-all_points.min(axis=0)).max()), 0.04) * 0.62

    fig = plt.figure(figsize=(14,11))
    views = [
        ("isometric", 24, 35),
        ("front", 0, -90),
        ("side", 0, 0),
        ("top", 90, -90),
    ]
    for i, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2,2,i, projection="3d")
        add_mesh(ax, original, (0.15,0.15,0.15), 0.28, (0.03,0.03,0.03), 0.16)
        add_mesh(ax, body, (0.85,0.12,0.10), 0.30, (0.60,0.02,0.02), 0.50)
        add_mesh(ax, finger, (0.12,0.35,0.90), 0.32, (0.02,0.12,0.60), 0.50)
        add_mesh(ax, cube, (0.10,0.72,0.18), 0.38, (0.02,0.32,0.05), 0.80)
        ax.set_xlim(center[0]-extent, center[0]+extent)
        ax.set_ylim(center[1]-extent, center[1]+extent)
        ax.set_zlim(center[2]-extent, center[2]+extent)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
    fig.suptitle(
        title
        + "\nDark grey=original STL, red=filled body hull, "
          "blue=functional fixed-finger hull, green=30 mm cube"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def score_candidate(original_gap, body_gap, finger_gap):
    # Main objective: preserve the original grasp-contact position.
    score = abs(finger_gap - original_gap)

    # The body should not become the first thing that touches the cube.
    desired_body_clearance = max(original_gap + 0.002, finger_gap + 0.002)
    if body_gap < desired_body_clearance:
        score += (desired_body_clearance - body_gap) * 15.0

    # Strong penalty if the body intersects / touches the cube.
    if body_gap < 0.00025:
        score += 0.050

    return float(score)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    proxy_dir = args.proxy_dir.resolve() if args.proxy_dir else newest_proxy_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else Path.home() / "下载" / f"so101_collision_functional_wrist_{timestamp}"
    )

    if output_dir.exists():
        raise FileExistsError(output_dir)

    # Clone the existing proxy result so the previous assembled-preview tool
    # can be run directly against the recommended candidate.
    shutil.copytree(proxy_dir, output_dir)
    candidates_dir = output_dir / "functional_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    xacro, record, cube_center, _cube_rpy = parse_model(project_root)
    raw = load_mesh(source_path(project_root, record["uri"]))
    original = to_gripper_frame(raw, record)
    cube = cube_mesh(cube_center)
    original_gap = mesh_cube_gap(original, cube_center, CUBE_SIZE)

    candidate_specs = [
        (0.18, 0.025),
        (0.24, 0.030),
        (0.30, 0.035),
        (0.36, 0.040),
        (0.42, 0.045),
    ]

    results = []
    for index, (finger_fraction, overlap_fraction) in enumerate(candidate_specs):
        body, finger, metadata = generate_candidate(
            original,
            cube_center,
            finger_fraction,
            overlap_fraction,
        )
        body_gap = mesh_cube_gap(body, cube_center, CUBE_SIZE)
        finger_gap = mesh_cube_gap(finger, cube_center, CUBE_SIZE)
        combined_gap = min(body_gap, finger_gap)
        score = score_candidate(original_gap, body_gap, finger_gap)

        name = f"candidate_{index:02d}"
        render_candidate(
            original,
            body,
            finger,
            cube,
            candidates_dir / f"{name}.png",
            (
                f"{name}: finger_fraction={finger_fraction:.2f}, "
                f"overlap={overlap_fraction:.3f}\n"
                f"original gap={original_gap*1000:.2f} mm, "
                f"body gap={body_gap*1000:.2f} mm, "
                f"finger gap={finger_gap*1000:.2f} mm"
            ),
        )

        results.append({
            "name": name,
            "finger_fraction": finger_fraction,
            "overlap_fraction": overlap_fraction,
            "body_gap_mm": body_gap * 1000.0,
            "finger_gap_mm": finger_gap * 1000.0,
            "combined_gap_mm": combined_gap * 1000.0,
            "score": score,
            "body_faces": int(len(body.faces)),
            "finger_faces": int(len(finger.faces)),
            "body": body,
            "finger": finger,
            "metadata": metadata,
        })

    recommended = min(results, key=lambda r: r["score"])
    body_raw = to_raw_local(recommended["body"], record)
    finger_raw = to_raw_local(recommended["finger"], record)

    mesh_dir = output_dir / "meshes"

    # Remove only the old generated wrist-roll-follower proxies from this
    # cloned output directory. Other generated collision proxies are kept.
    manifest_path = output_dir / "collision_replacement_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_files = manifest["generated_meshes"][FIXED_MESH]["proxy_files"]
    for filename in old_files:
        (mesh_dir / filename).unlink(missing_ok=True)

    body_name = "wrist_roll_follower_so101_v1__functional_body__part_00.stl"
    finger_name = "wrist_roll_follower_so101_v1__functional_finger__part_01.stl"
    body_raw.export(mesh_dir / body_name)
    finger_raw.export(mesh_dir / finger_name)

    manifest["generated_meshes"][FIXED_MESH] = {
        "source": str(source_path(project_root, record["uri"])),
        "profile": "functional_2hull_filled_body_plus_fixed_finger",
        "proxy_files": [body_name, finger_name],
        "functional_partition": {
            "recommended_candidate": recommended["name"],
            "finger_fraction": recommended["finger_fraction"],
            "overlap_fraction": recommended["overlap_fraction"],
            "original_cube_gap_mm": original_gap * 1000.0,
            "body_cube_gap_mm": recommended["body_gap_mm"],
            "finger_cube_gap_mm": recommended["finger_gap_mm"],
            "body_faces": recommended["body_faces"],
            "finger_faces": recommended["finger_faces"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Replace the old single-part preview for this mesh with the recommended
    # functional-partition preview, while keeping all candidate images.
    render_candidate(
        original,
        recommended["body"],
        recommended["finger"],
        cube,
        output_dir / "previews" / "wrist_roll_follower_so101_v1.png",
        (
            "RECOMMENDED functional wrist_roll_follower proxy\n"
            f"{recommended['name']}, original gap={original_gap*1000:.2f} mm, "
            f"body gap={recommended['body_gap_mm']:.2f} mm, "
            f"finger gap={recommended['finger_gap_mm']:.2f} mm"
        ),
    )

    serializable = []
    for result in results:
        serializable.append({
            key: value
            for key, value in result.items()
            if key not in {"body", "finger"}
        })
    report = {
        "source_xacro": str(xacro),
        "base_proxy_dir": str(proxy_dir),
        "output_dir": str(output_dir),
        "original_cube_gap_mm": original_gap * 1000.0,
        "cube_center_in_gripper_link_m": cube_center.tolist(),
        "candidate_results": serializable,
        "recommended": {
            key: value
            for key, value in recommended.items()
            if key not in {"body", "finger"}
        },
        "note": (
            "Project source files are unchanged. This cloned proxy directory "
            "uses a functional two-hull wrist_roll_follower: one convex body "
            "that fills the servo recess, plus one fixed gripping-finger hull."
        ),
    }
    (output_dir / "functional_wrist_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "SO-101 functional wrist_roll_follower collision proxy",
        "",
        f"Base proxy directory: {proxy_dir}",
        f"Recommended: {recommended['name']}",
        f"Original fixed-jaw/cube approximate gap: {original_gap*1000:.3f} mm",
        f"Body/cube approximate gap: {recommended['body_gap_mm']:.3f} mm",
        f"Finger/cube approximate gap: {recommended['finger_gap_mm']:.3f} mm",
        f"Body faces: {recommended['body_faces']}",
        f"Finger faces: {recommended['finger_faces']}",
        "",
        "The servo mounting recess is intentionally filled by the body hull.",
        "The fixed gripping finger remains a separate convex hull.",
        "No project source file was modified.",
        "",
        "Next inspection:",
        "  1. Open functional_candidates/",
        "  2. Open previews/wrist_roll_follower_so101_v1.png",
        "  3. Run preview_so101_gripper_assembled_collision.py against this output directory.",
    ]
    (output_dir / "FUNCTIONAL_WRIST_README.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("SO-101 functional wrist_roll_follower proxy generated.")
    print(f"Base proxy directory: {proxy_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Recommended candidate: {recommended['name']}")
    print(f"Original fixed-jaw/cube gap: {original_gap*1000:.3f} mm")
    print(f"Body/cube gap: {recommended['body_gap_mm']:.3f} mm")
    print(f"Finger/cube gap: {recommended['finger_gap_mm']:.3f} mm")
    print(f"Body faces: {recommended['body_faces']}")
    print(f"Finger faces: {recommended['finger_faces']}")
    print()
    print(f"Open candidates: xdg-open '{candidates_dir}'")
    print(
        "Open recommended: xdg-open "
        f"'{output_dir / 'previews' / 'wrist_roll_follower_so101_v1.png'}'"
    )
    print()
    print("No project source file was modified.")


if __name__ == "__main__":
    main()
