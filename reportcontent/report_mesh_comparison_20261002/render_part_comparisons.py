#!/usr/bin/env python3
"""Render the exact URDF visual STL / collision STL pairs for a report.

This is an offline, CPU-only renderer. It never imports ROS, starts Gazebo,
connects to a running simulation, changes meshes, or runs mesh simplification.
All input triangles are passed to Matplotlib's Poly3DCollection unchanged.

Run with the existing lerobot environment. Use --check-only for a lightweight
path / triangle-count / URDF-placement check without rendering any figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Bound numerical-library CPU use before importing NumPy / Trimesh.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros2_ws/src/so101_description"
URDF = PACKAGE / "urdf/so101.urdf.xacro"
OUTPUT = Path(__file__).resolve().parent

BLUE = "#3976ad"
ORANGE = "#d98632"
CYAN = "#299b95"
TEXT = "#25354A"
MUTED = "#667487"

PARTS = (
    {"key": "upper_arm", "title": "Upper arm",
     "stem": "upper_arm_so101_v1", "source_faces": 26068,
     "proxies": (("ultralow_1hull__part_00", "Body", 42),)},
    {"key": "wrist_roll_follower", "title": "Wrist-roll follower",
     "stem": "wrist_roll_follower_so101_v1", "source_faces": 28796,
     "proxies": (("functional_body__part_00", "Body", 44),
                 ("functional_finger__part_01", "Finger", 42))},
    {"key": "moving_jaw", "title": "Moving jaw",
     "stem": "moving_jaw_so101_v1", "source_faces": 28270,
     "proxies": (("functional_body__part_00", "Body", 44),
                 ("functional_finger__part_01", "Finger", 44))},
)


def placement(element: ET.Element) -> np.ndarray:
    """Compare exact URDF origin and scale; both sides share mesh coordinates."""
    origin = element.find("origin")
    mesh = element.find("geometry/mesh")
    assert mesh is not None
    xyz = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
    rpy = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
    return np.fromstring(f"{xyz} {rpy} {mesh.get('scale', '1 1 1')}", sep=" ")


def load_mesh(path: Path, expected_faces: int) -> trimesh.Trimesh:
    # process=False preserves every face, including coincident STL vertices.
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a single triangle mesh: {path}")
    if mesh.faces.shape != (expected_faces, 3):
        raise ValueError(f"Triangle count changed for {path.name}: "
                         f"expected {expected_faces}, got {len(mesh.faces)}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Non-finite vertex in {path}")
    return mesh


def input_record(path: Path, mesh: trimesh.Trimesh) -> dict:
    return {"path": str(path.relative_to(ROOT)), "triangles": len(mesh.faces),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bounds_m": mesh.bounds.tolist()}


def load_pairs() -> list[dict]:
    tree = ET.parse(URDF).getroot()
    pairs = []
    for spec in PARTS:
        source_path = PACKAGE / "meshes" / f"{spec['stem']}.stl"
        source_uri = "package://so101_description/" + str(source_path.relative_to(PACKAGE))
        matches = [(link, visual) for link in tree.findall("link")
                   for visual in link.findall("visual")
                   if visual.find("geometry/mesh") is not None
                   and visual.find("geometry/mesh").get("filename") == source_uri]
        if len(matches) != 1:
            raise ValueError(f"Expected one visual in current URDF for {source_uri}")
        link, visual = matches[0]
        source = load_mesh(source_path, spec["source_faces"])
        proxies, records = [], []
        for suffix, label, expected_faces in spec["proxies"]:
            proxy_path = PACKAGE / "meshes/collision_generated_ultralow" / f"{spec['stem']}__{suffix}.stl"
            uri = "package://so101_description/" + str(proxy_path.relative_to(PACKAGE))
            collisions = [collision for collision in link.findall("collision")
                          if collision.find("geometry/mesh") is not None
                          and collision.find("geometry/mesh").get("filename") == uri]
            if len(collisions) != 1:
                raise ValueError(f"Current URDF does not reference expected proxy: {uri}")
            if not np.allclose(placement(visual), placement(collisions[0]), rtol=0, atol=1e-12):
                raise ValueError(f"Source and collision origin / scale differ for {uri}")
            if not np.allclose(placement(visual)[6:], 1.0, rtol=0, atol=1e-12):
                raise ValueError("Non-unit mesh scale needs explicit handling.")
            mesh = load_mesh(proxy_path, expected_faces)
            proxies.append((label, mesh))
            records.append({"component": label, **input_record(proxy_path, mesh)})
        all_bounds = np.concatenate([source.bounds, *(mesh.bounds for _, mesh in proxies)])
        bounds = np.stack((all_bounds.min(axis=0), all_bounds.max(axis=0)))
        pairs.append({**spec, "source": source, "meshes": proxies,
                      "bounds": bounds, "record": {
                          "part": spec["key"], "urdf_link": link.get("name"),
                          "urdf_origin_xyz_rpy_scale": placement(visual).tolist(),
                          "source": input_record(source_path, source), "collision": records,
                          "shared_bounds_m": bounds.tolist(),
                          "reduction_percent": 100 * (1 - sum(len(m.faces) for _, m in proxies)
                                                       / len(source.faces)),
                      }})
    return pairs


def triangle_collection(mesh: trimesh.Trimesh, color: str, *, proxy: bool):
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    assert len(triangles) == len(mesh.faces)  # No sampling or simplification.
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], np.finfo(float).eps)
    light = np.asarray((0.25, -0.55, 0.80))
    light /= np.linalg.norm(light)
    shade = 0.56 + 0.44 * np.abs(normals @ light)
    colors = np.asarray(to_rgb(color))[None, :] * shade[:, None]
    collection = Poly3DCollection(
        triangles * 1000.0, facecolors=colors,
        edgecolors=(0.13, 0.19, 0.24, 0.65 if proxy else 0.15),
        linewidths=0.46 if proxy else 0.10, antialiaseds=True,
        zsort="average", rasterized=False,
    )
    return collection


def draw_part(ax, pair: dict, *, simplified: bool, span_m: float,
              elevation: float, azimuth: float) -> None:
    ax.set_proj_type("ortho")
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_box_aspect((1, 1, 1), zoom=1.22)
    center = pair["bounds"].mean(axis=0) * 1000.0
    half = span_m * 1000.0 / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_axis_off()
    ax.set_facecolor("white")
    if simplified:
        for label, mesh in pair["meshes"]:
            ax.add_collection3d(triangle_collection(mesh, CYAN if label == "Finger" else ORANGE,
                                                    proxy=True))
    else:
        ax.add_collection3d(triangle_collection(pair["source"], BLUE, proxy=False))


def face_label(pair: dict, simplified: bool) -> str:
    if not simplified:
        return f"{len(pair['source'].faces):,} triangles"
    total = sum(len(mesh.faces) for _, mesh in pair["meshes"])
    if len(pair["meshes"]) == 1:
        return f"{total:,} triangles  |  1 convex hull"
    counts = " + ".join(f"{len(mesh.faces)} {label.lower()}" for label, mesh in pair["meshes"])
    return f"{counts}  =  {total} triangles"


def add_footer(fig, *, scale_mode: str, fontsize: float = 10.5) -> None:
    from matplotlib.patches import Patch
    # Physical spacing keeps the shorter individual figures as legible as the
    # three-row sheet. Matplotlib's legend anchor marks its lower boundary.
    height_inches = float(fig.get_size_inches()[1])
    footer_note_y = 0.17 / height_inches
    footer_detail_y = 0.40 / height_inches
    legend_bottom_y = 0.57 / height_inches
    fig.legend(handles=[Patch(facecolor=BLUE, label="Original mesh"),
                        Patch(facecolor=ORANGE, label="Collision body"),
                        Patch(facecolor=CYAN, label="Finger / contact hull")],
               loc="lower center", bbox_to_anchor=(0.5, legend_bottom_y), ncol=3,
               frameon=False, fontsize=fontsize, handlelength=1.25,
               columnspacing=2.0, labelcolor=TEXT)
    scope = "All rows" if scale_mode == "global" else "Each pair"
    fig.text(0.5, footer_detail_y, f"{scope}: identical scale and orthographic view. All mesh triangles retained.",
             ha="center", va="center", color=MUTED, fontsize=fontsize - 0.5)
    fig.text(0.5, footer_note_y, "Collision geometry simplified; visual meshes unchanged.",
             ha="center", va="center", color=TEXT, fontsize=fontsize, weight="medium")


def render(pairs: list[dict], args) -> list[Path]:
    # Avoid GUI / GPU imports and leave the training process untouched.
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    global_span = max(float(np.ptp(pair["bounds"], axis=0).max()) for pair in pairs) * 1.08
    spans = [global_span if args.scale_mode == "global"
             else float(np.ptp(pair["bounds"], axis=0).max()) * 1.08 for pair in pairs]
    written = []
    fig = plt.figure(figsize=(12.0, 14.8))
    fig.suptitle("Collision Mesh Simplification", y=0.985, fontsize=22,
                 color=TEXT, weight="bold")
    fig.text(0.265, 0.945, "ORIGINAL GEOMETRY", ha="center", color=BLUE,
             fontsize=12.5, weight="bold")
    fig.text(0.755, 0.945, "SIMPLIFIED COLLISION GEOMETRY", ha="center", color="#AE691A",
             fontsize=12.5, weight="bold")
    for row, (pair, span) in enumerate(zip(pairs, spans)):
        y = 0.642 - 0.271 * row
        for column in range(2):
            ax = fig.add_axes([0.022 + column * 0.49, y, 0.466, 0.278], projection="3d")
            draw_part(ax, pair, simplified=bool(column), span_m=span,
                      elevation=args.elevation, azimuth=args.azimuth)
            ax.text2D(0.5, 0.972, pair["title"], transform=ax.transAxes,
                      ha="center", color=TEXT, fontsize=14, weight="bold")
            ax.text2D(0.5, 0.917, face_label(pair, bool(column)), transform=ax.transAxes,
                      ha="center", color=MUTED, fontsize=10.5)
    add_footer(fig, scale_mode=args.scale_mode)
    for extension in ("png", "pdf"):
        path = args.output_dir / f"collision_mesh_part_comparison.{extension}"
        fig.savefig(path, dpi=args.dpi)
        written.append(path)
    plt.close(fig)

    # Render and release individual pairs sequentially to bound memory usage.
    for pair, span in zip(pairs, spans):
        fig = plt.figure(figsize=(11.2, 6.1))
        fig.suptitle(pair["title"] + " — Collision Mesh Comparison", y=0.975,
                     fontsize=19, color=TEXT, weight="bold")
        for column in range(2):
            ax = fig.add_axes([0.014 + column * 0.495, 0.135, 0.477, 0.72], projection="3d")
            draw_part(ax, pair, simplified=bool(column), span_m=span,
                      elevation=args.elevation, azimuth=args.azimuth)
            title = "Simplified collision geometry" if column else "Original geometry"
            ax.text2D(0.5, 0.98, title, transform=ax.transAxes, ha="center",
                      color=TEXT, fontsize=13, weight="bold")
            ax.text2D(0.5, 0.915, face_label(pair, bool(column)), transform=ax.transAxes,
                      ha="center", color=MUTED, fontsize=11)
        add_footer(fig, scale_mode="pair", fontsize=10)
        path = args.output_dir / f"{pair['key']}_collision_comparison.png"
        fig.savefig(path, dpi=args.dpi)
        written.append(path)
        plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--elevation", type=float, default=25.0)
    parser.add_argument("--azimuth", type=float, default=-45.0)
    parser.add_argument("--scale-mode", choices=("pair", "global"), default="pair",
                        help="Pair: maximize each row, keeping its two meshes identical in scale; "
                             "global: identical physical scale across all three rows.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    pairs = load_pairs()
    records = [pair["record"] for pair in pairs]
    if args.check_only:
        print(json.dumps({"urdf": str(URDF), "parts": records}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = render(pairs, args)
    provenance = {"renderer": "Matplotlib Agg / CPU", "geometry_processing": "none",
                  "triangle_sampling": "none", "projection": "orthographic",
                  "elevation_deg": args.elevation, "azimuth_deg": args.azimuth,
                  "scale_mode": args.scale_mode, "dpi": args.dpi,
                  "urdf": str(URDF.relative_to(ROOT)),
                  "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
                  "parts": records, "artifacts": [str(path) for path in paths]}
    manifest = args.output_dir / "part_comparison_provenance.json"
    manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": [str(path) for path in (*paths, manifest)]}, indent=2))


if __name__ == "__main__":
    main()
