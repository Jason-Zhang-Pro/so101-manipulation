#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
MOVING_MESH = "moving_jaw_so101_v1.stl"
MOVING_LINK = "moving_jaw_so101_v1_link"
GRIPPER_JOINT = "gripper"
CUBE_SIZE = 0.03


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument(
        "--proxy-dir",
        type=Path,
        required=True,
        help=(
            "Existing proxy directory to clone. Use the functional-wrist "
            "directory so the fixed-jaw improvement is retained."
        ),
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--contact-tolerance-mm", type=float, default=0.35)
    p.add_argument("--search-samples", type=int, default=1001)
    return p.parse_args()


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


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array([
        [c+x*x*t, x*y*t-z*s, x*z*t+y*s],
        [y*x*t+z*s, c+y*y*t, y*z*t-x*s],
        [z*x*t-y*s, z*y*t+x*s, c+z*z*t],
    ], dtype=float)


def transform(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = rpy_matrix(rpy)
    m[:3, 3] = xyz
    return m


def joint_transform(xyz, rpy, axis, q):
    base = transform(xyz, rpy)
    motion = np.eye(4)
    motion[:3, :3] = axis_angle_matrix(axis, q)
    return base @ motion


def parse_origin(element):
    if element is None:
        return np.zeros(3), np.zeros(3)
    return vec(element.get("xyz")), vec(element.get("rpy"))


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not parts:
            raise RuntimeError(f"No mesh geometry: {path}")
        mesh = trimesh.util.concatenate(parts)
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


def simplify_convex(mesh, max_vertices=24):
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
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True,
    )


def source_path(project_root, uri):
    if not uri.startswith(PACKAGE_PREFIX):
        raise ValueError(uri)
    return (
        project_root
        / "ros2_ws/src/so101_description/meshes"
        / uri[len(PACKAGE_PREFIX):]
    )


def parse_model(project_root):
    xacro = (
        project_root
        / "ros2_ws/src/so101_description/urdf/so101.urdf.xacro"
    )
    root = ET.parse(xacro).getroot()

    moving_link = root.find(f"./link[@name='{MOVING_LINK}']")
    if moving_link is None:
        raise RuntimeError(f"{MOVING_LINK} not found")

    collision_record = None
    for collision in moving_link.findall("./collision"):
        mesh = collision.find("./geometry/mesh")
        if mesh is None:
            continue
        if Path(mesh.get("filename", "")).name == MOVING_MESH:
            xyz, rpy = parse_origin(collision.find("./origin"))
            collision_record = {
                "uri": mesh.get("filename"),
                "scale": (
                    vec(mesh.get("scale"), (1,1,1))
                    if mesh.get("scale")
                    else np.ones(3)
                ),
                "xyz": xyz,
                "rpy": rpy,
            }
            break
    if collision_record is None:
        raise RuntimeError(
            f"Collision mesh {MOVING_MESH} not found in {MOVING_LINK}"
        )

    joint = root.find(f"./joint[@name='{GRIPPER_JOINT}']")
    if joint is None:
        raise RuntimeError(f"{GRIPPER_JOINT} joint not found")
    joint_xyz, joint_rpy = parse_origin(joint.find("./origin"))
    axis_el = joint.find("./axis")
    axis = vec(axis_el.get("xyz") if axis_el is not None else "1 0 0")
    limit = joint.find("./limit")
    if limit is None:
        raise RuntimeError("gripper joint limit not found")
    lower = float(limit.get("lower"))
    upper = float(limit.get("upper"))

    frame_joint = root.find("./joint[@name='gripper_frame_joint']")
    if frame_joint is None:
        raise RuntimeError("gripper_frame_joint not found")
    frame_xyz, frame_rpy = parse_origin(frame_joint.find("./origin"))

    return {
        "xacro": xacro,
        "collision": collision_record,
        "joint_xyz": joint_xyz,
        "joint_rpy": joint_rpy,
        "joint_axis": axis,
        "joint_lower": lower,
        "joint_upper": upper,
        "cube_center": frame_xyz,
    }


def collision_mesh_link_local(raw_mesh, record):
    scaled = apply_scale(raw_mesh, record["scale"])
    return transformed(
        scaled,
        transform(record["xyz"], record["rpy"]),
    )


def link_local_to_raw(mesh_link, record):
    result = mesh_link.copy()
    result.apply_transform(
        np.linalg.inv(transform(record["xyz"], record["rpy"]))
    )
    inv_scale = np.eye(4)
    inv_scale[0,0] = 1.0 / record["scale"][0]
    inv_scale[1,1] = 1.0 / record["scale"][1]
    inv_scale[2,2] = 1.0 / record["scale"][2]
    result.apply_transform(inv_scale)
    return result


def moving_to_gripper(mesh_link, model, q):
    return transformed(
        mesh_link,
        joint_transform(
            model["joint_xyz"],
            model["joint_rpy"],
            model["joint_axis"],
            q,
        ),
    )


def cube_mesh(center):
    cube = trimesh.creation.box(
        extents=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    )
    cube.apply_translation(center)
    return cube


def original_contact_search(original_link, model, tolerance, samples):
    qs = np.linspace(
        model["joint_upper"],
        model["joint_lower"],
        max(5, samples),
    )
    best = (model["joint_upper"], math.inf)
    first = None

    for q in qs:
        placed = moving_to_gripper(original_link, model, float(q))
        gap = mesh_cube_gap(
            placed,
            model["cube_center"],
            CUBE_SIZE,
        )
        if gap < best[1]:
            best = (float(q), gap)
        if gap <= tolerance:
            first = (float(q), gap)
            break

    return first, best


def cube_center_in_moving_link(model, q):
    world_from_link = joint_transform(
        model["joint_xyz"],
        model["joint_rpy"],
        model["joint_axis"],
        q,
    )
    point = np.ones(4)
    point[:3] = model["cube_center"]
    local = np.linalg.inv(world_from_link) @ point
    return local[:3]


def generate_candidate(
    original_link,
    cube_center_link,
    finger_fraction,
    overlap_fraction,
):
    points = mesh_points(original_link)
    centroid = original_link.vertices.mean(axis=0)
    direction = cube_center_link - centroid
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise RuntimeError("Cube centre coincides with moving mesh centroid.")
    direction /= norm

    projection = (points - centroid[None, :]) @ direction
    low = float(projection.min())
    high = float(projection.max())
    span = max(high - low, 1e-9)

    cut = float(np.quantile(
        projection,
        1.0 - finger_fraction,
    ))
    overlap = span * overlap_fraction

    centers = original_link.triangles_center
    center_projection = (
        centers - centroid[None, :]
    ) @ direction

    body_ids = np.flatnonzero(
        center_projection <= cut + overlap
    )
    finger_ids = np.flatnonzero(
        center_projection >= cut - overlap
    )

    if len(body_ids) < 4 or len(finger_ids) < 4:
        raise RuntimeError("Candidate split produced too few faces.")

    body = original_link.submesh(
        [body_ids],
        append=True,
        repair=False,
    )
    finger = original_link.submesh(
        [finger_ids],
        append=True,
        repair=False,
    )
    body.remove_unreferenced_vertices()
    finger.remove_unreferenced_vertices()

    body = simplify_convex(body, 24)
    finger = simplify_convex(finger, 24)

    return body, finger, {
        "finger_fraction": float(finger_fraction),
        "overlap_fraction": float(overlap_fraction),
        "cut_projection": cut,
        "direction": direction.tolist(),
    }


def candidate_contact_search(body, finger, model, tolerance, samples):
    qs = np.linspace(
        model["joint_upper"],
        model["joint_lower"],
        max(5, samples),
    )
    first = None
    best = None
    min_body_gap_before_contact = math.inf

    for q in qs:
        body_world = moving_to_gripper(body, model, float(q))
        finger_world = moving_to_gripper(finger, model, float(q))
        body_gap = mesh_cube_gap(
            body_world,
            model["cube_center"],
            CUBE_SIZE,
        )
        finger_gap = mesh_cube_gap(
            finger_world,
            model["cube_center"],
            CUBE_SIZE,
        )
        combined = min(body_gap, finger_gap)

        if best is None or combined < best["combined_gap"]:
            best = {
                "q": float(q),
                "combined_gap": combined,
                "body_gap": body_gap,
                "finger_gap": finger_gap,
            }

        min_body_gap_before_contact = min(
            min_body_gap_before_contact,
            body_gap,
        )

        if combined <= tolerance:
            first = {
                "q": float(q),
                "combined_gap": combined,
                "body_gap": body_gap,
                "finger_gap": finger_gap,
                "touching_part": (
                    "finger"
                    if finger_gap <= body_gap
                    else "body"
                ),
                "min_body_gap_before_contact": (
                    min_body_gap_before_contact
                ),
            }
            break

    return first, best


def score_candidate(
    original_q,
    candidate_first,
    candidate_best,
    model,
):
    if candidate_first is None:
        # No contact at all is unacceptable.
        return 1000.0 + candidate_best["combined_gap"]

    q_error = abs(candidate_first["q"] - original_q)
    score = q_error

    # Body must not be the first contacting component.
    if candidate_first["touching_part"] != "finger":
        score += 10.0

    # Keep body away from cube by at least 2 mm during approach.
    body_clearance = candidate_first["min_body_gap_before_contact"]
    if body_clearance < 0.002:
        score += (0.002 - body_clearance) * 100.0 + 1.0

    # Prefer a finger contact whose geometric gap is near zero.
    score += abs(candidate_first["finger_gap"]) * 3.0

    return float(score)


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


def render_candidate(
    original_world,
    body_world,
    finger_world,
    cube,
    output,
    title,
):
    points = np.vstack([
        original_world.vertices,
        body_world.vertices,
        finger_world.vertices,
        cube.vertices,
    ])
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = max(
        float((points.max(axis=0)-points.min(axis=0)).max()),
        0.04,
    ) * 0.62

    fig = plt.figure(figsize=(14,11))
    views = [
        ("isometric", 24, 35),
        ("front", 0, -90),
        ("side", 0, 0),
        ("top", 90, -90),
    ]

    for i, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2,2,i, projection="3d")
        add_mesh(
            ax, original_world,
            (0.15,0.15,0.15), 0.28,
            (0.03,0.03,0.03), 0.16,
        )
        add_mesh(
            ax, body_world,
            (0.85,0.12,0.10), 0.30,
            (0.60,0.02,0.02), 0.50,
        )
        add_mesh(
            ax, finger_world,
            (0.12,0.35,0.90), 0.32,
            (0.02,0.12,0.60), 0.50,
        )
        add_mesh(
            ax, cube,
            (0.10,0.72,0.18), 0.38,
            (0.02,0.32,0.05), 0.80,
        )

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
        + "\nDark grey=original moving jaw, "
          "red=filled body hull, "
          "blue=functional finger hull, "
          "green=30 mm cube"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    proxy_dir = args.proxy_dir.expanduser().resolve()

    manifest_src = proxy_dir / "collision_replacement_manifest.json"
    if not manifest_src.is_file():
        raise FileNotFoundError(manifest_src)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else (
            Path.home()
            / "下载"
            / f"so101_collision_functional_gripper_{timestamp}"
        )
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)

    shutil.copytree(proxy_dir, output_dir)
    candidates_dir = output_dir / "moving_jaw_functional_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    model = parse_model(project_root)
    record = model["collision"]

    raw = load_mesh(source_path(project_root, record["uri"]))
    original_link = collision_mesh_link_local(raw, record)

    tolerance = args.contact_tolerance_mm / 1000.0
    original_first, original_best = original_contact_search(
        original_link,
        model,
        tolerance,
        args.search_samples,
    )
    if original_first is None:
        reference_q = original_best[0]
        reference_gap = original_best[1]
        reference_kind = "closest_pose_no_tolerance_contact"
    else:
        reference_q, reference_gap = original_first
        reference_kind = "first_contact"

    cube_center_link = cube_center_in_moving_link(
        model,
        reference_q,
    )
    cube = cube_mesh(model["cube_center"])

    candidate_specs = [
        (0.16, 0.020),
        (0.22, 0.025),
        (0.28, 0.030),
        (0.34, 0.035),
        (0.40, 0.040),
        (0.46, 0.045),
    ]

    results = []

    for index, (finger_fraction, overlap_fraction) in enumerate(
        candidate_specs
    ):
        body, finger, metadata = generate_candidate(
            original_link,
            cube_center_link,
            finger_fraction,
            overlap_fraction,
        )

        first, best = candidate_contact_search(
            body,
            finger,
            model,
            tolerance,
            args.search_samples,
        )
        score = score_candidate(
            reference_q,
            first,
            best,
            model,
        )

        q_preview = (
            first["q"]
            if first is not None
            else best["q"]
        )
        original_world = moving_to_gripper(
            original_link,
            model,
            q_preview,
        )
        body_world = moving_to_gripper(
            body,
            model,
            q_preview,
        )
        finger_world = moving_to_gripper(
            finger,
            model,
            q_preview,
        )

        name = f"candidate_{index:02d}"

        if first is None:
            contact_text = (
                f"NO contact; closest gap "
                f"{best['combined_gap']*1000:.2f} mm"
            )
            touching_part = "none"
            q_deg = math.degrees(best["q"])
            body_gap_mm = best["body_gap"] * 1000.0
            finger_gap_mm = best["finger_gap"] * 1000.0
            min_body_gap_mm = math.nan
        else:
            contact_text = (
                f"first={first['touching_part']}, "
                f"body={first['body_gap']*1000:.2f} mm, "
                f"finger={first['finger_gap']*1000:.2f} mm"
            )
            touching_part = first["touching_part"]
            q_deg = math.degrees(first["q"])
            body_gap_mm = first["body_gap"] * 1000.0
            finger_gap_mm = first["finger_gap"] * 1000.0
            min_body_gap_mm = (
                first["min_body_gap_before_contact"] * 1000.0
            )

        render_candidate(
            original_world,
            body_world,
            finger_world,
            cube,
            candidates_dir / f"{name}.png",
            (
                f"{name}: fraction={finger_fraction:.2f}, "
                f"overlap={overlap_fraction:.3f}, "
                f"q={q_deg:.2f} deg\n"
                f"{contact_text}"
            ),
        )

        results.append({
            "name": name,
            "finger_fraction": finger_fraction,
            "overlap_fraction": overlap_fraction,
            "score": score,
            "first_touching_part": touching_part,
            "first_contact_q_deg": q_deg,
            "body_gap_mm_at_first_contact": body_gap_mm,
            "finger_gap_mm_at_first_contact": finger_gap_mm,
            "minimum_body_gap_before_contact_mm": min_body_gap_mm,
            "body_faces": int(len(body.faces)),
            "finger_faces": int(len(finger.faces)),
            "metadata": metadata,
            "body": body,
            "finger": finger,
        })

    recommended = min(results, key=lambda r: r["score"])

    body_raw = link_local_to_raw(
        recommended["body"],
        record,
    )
    finger_raw = link_local_to_raw(
        recommended["finger"],
        record,
    )

    mesh_dir = output_dir / "meshes"
    manifest_path = (
        output_dir
        / "collision_replacement_manifest.json"
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    old_files = manifest["generated_meshes"][MOVING_MESH]["proxy_files"]
    for filename in old_files:
        (mesh_dir / filename).unlink(missing_ok=True)

    body_name = (
        "moving_jaw_so101_v1__functional_body__part_00.stl"
    )
    finger_name = (
        "moving_jaw_so101_v1__functional_finger__part_01.stl"
    )
    body_raw.export(mesh_dir / body_name)
    finger_raw.export(mesh_dir / finger_name)

    manifest["generated_meshes"][MOVING_MESH] = {
        "source": str(
            source_path(project_root, record["uri"])
        ),
        "profile": (
            "functional_2hull_filled_body_plus_moving_finger"
        ),
        "proxy_files": [body_name, finger_name],
        "functional_partition": {
            "recommended_candidate": recommended["name"],
            "finger_fraction": recommended["finger_fraction"],
            "overlap_fraction": recommended["overlap_fraction"],
            "reference_contact_kind": reference_kind,
            "original_contact_q_deg": math.degrees(reference_q),
            "original_gap_mm": reference_gap * 1000.0,
            "candidate_first_touching_part": (
                recommended["first_touching_part"]
            ),
            "candidate_first_contact_q_deg": (
                recommended["first_contact_q_deg"]
            ),
            "body_gap_mm_at_first_contact": (
                recommended["body_gap_mm_at_first_contact"]
            ),
            "finger_gap_mm_at_first_contact": (
                recommended["finger_gap_mm_at_first_contact"]
            ),
            "minimum_body_gap_before_contact_mm": (
                recommended[
                    "minimum_body_gap_before_contact_mm"
                ]
            ),
            "body_faces": recommended["body_faces"],
            "finger_faces": recommended["finger_faces"],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    q_preview = math.radians(
        recommended["first_contact_q_deg"]
    )
    original_world = moving_to_gripper(
        original_link,
        model,
        q_preview,
    )
    body_world = moving_to_gripper(
        recommended["body"],
        model,
        q_preview,
    )
    finger_world = moving_to_gripper(
        recommended["finger"],
        model,
        q_preview,
    )

    render_candidate(
        original_world,
        body_world,
        finger_world,
        cube,
        output_dir / "previews" / "moving_jaw_so101_v1.png",
        (
            "RECOMMENDED functional moving_jaw proxy\n"
            f"{recommended['name']}; "
            f"original contact={math.degrees(reference_q):.2f} deg; "
            f"new contact={recommended['first_contact_q_deg']:.2f} deg; "
            f"first part={recommended['first_touching_part']}"
        ),
    )

    serializable = []
    for r in results:
        serializable.append({
            k: v for k, v in r.items()
            if k not in {"body", "finger"}
        })

    report = {
        "source_xacro": str(model["xacro"]),
        "base_proxy_dir": str(proxy_dir),
        "output_dir": str(output_dir),
        "cube_center_in_gripper_link_m": (
            model["cube_center"].tolist()
        ),
        "contact_tolerance_mm": args.contact_tolerance_mm,
        "original_reference": {
            "kind": reference_kind,
            "q_rad": reference_q,
            "q_deg": math.degrees(reference_q),
            "gap_mm": reference_gap * 1000.0,
        },
        "candidate_results": serializable,
        "recommended": {
            k: v for k, v in recommended.items()
            if k not in {"body", "finger"}
        },
        "note": (
            "Project source files are unchanged. The cloned proxy "
            "directory now retains the existing functional fixed jaw "
            "and replaces moving_jaw with two functional convex hulls: "
            "a filled body and a separate moving gripping finger."
        ),
    }
    (
        output_dir / "functional_moving_jaw_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    readme = [
        "SO-101 functional moving_jaw collision proxy",
        "",
        f"Base proxy directory: {proxy_dir}",
        f"Recommended: {recommended['name']}",
        (
            "Original moving-jaw contact angle: "
            f"{math.degrees(reference_q):.3f} deg"
        ),
        (
            "New moving-jaw first-contact angle: "
            f"{recommended['first_contact_q_deg']:.3f} deg"
        ),
        (
            "First contacting new part: "
            f"{recommended['first_touching_part']}"
        ),
        (
            "Body gap at first contact: "
            f"{recommended['body_gap_mm_at_first_contact']:.3f} mm"
        ),
        (
            "Finger gap at first contact: "
            f"{recommended['finger_gap_mm_at_first_contact']:.3f} mm"
        ),
        (
            "Minimum body gap before contact: "
            f"{recommended['minimum_body_gap_before_contact_mm']:.3f} mm"
        ),
        f"Body faces: {recommended['body_faces']}",
        f"Finger faces: {recommended['finger_faces']}",
        "",
        "No project source file was modified.",
    ]
    (
        output_dir / "FUNCTIONAL_MOVING_JAW_README.txt"
    ).write_text(
        "\n".join(readme),
        encoding="utf-8",
    )

    print("SO-101 functional moving_jaw proxy generated.")
    print(f"Base proxy directory: {proxy_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Recommended candidate: {recommended['name']}")
    print(
        "Original moving-jaw first-contact angle: "
        f"{math.degrees(reference_q):.3f} deg"
    )
    print(
        "New first-contact angle: "
        f"{recommended['first_contact_q_deg']:.3f} deg"
    )
    print(
        "First contacting part: "
        f"{recommended['first_touching_part']}"
    )
    print(
        "Body gap at first contact: "
        f"{recommended['body_gap_mm_at_first_contact']:.3f} mm"
    )
    print(
        "Finger gap at first contact: "
        f"{recommended['finger_gap_mm_at_first_contact']:.3f} mm"
    )
    print(
        "Minimum body gap before contact: "
        f"{recommended['minimum_body_gap_before_contact_mm']:.3f} mm"
    )
    print(f"Body faces: {recommended['body_faces']}")
    print(f"Finger faces: {recommended['finger_faces']}")
    print()
    print(
        f"Open candidates: xdg-open '{candidates_dir}'"
    )
    print(
        "Open recommended: xdg-open "
        f"'{output_dir / 'previews' / 'moving_jaw_so101_v1.png'}'"
    )
    print()
    print("No project source file was modified.")


if __name__ == "__main__":
    main()
