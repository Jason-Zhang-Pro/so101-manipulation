#!/usr/bin/env python3
"""Render actual source/current collision triangles, without a simulator or GPU."""
from pathlib import Path
import argparse
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MESHDIR = ROOT / 'ros2_ws/src/so101_description/meshes'
URDF = ROOT / 'ros2_ws/src/so101_description/urdf/so101.urdf.xacro'
PROXYDIR = MESHDIR / 'collision_generated_ultralow'
BLUE, ORANGE, TEAL = '#3976ad', '#d98632', '#299b95'
POSE_DEG = dict(shoulder_pan=0, shoulder_lift=-35, elbow_flex=45,
                wrist_flex=20, wrist_roll=0, gripper=25)


def vec(text, default=(0, 0, 0)):
    return np.array([float(x) for x in text.split()] if text else default)


def origin(element):
    matrix = np.eye(4)
    if element is None:
        return matrix
    r, p, y = vec(element.get('rpy'))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    matrix[:3, :3] = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    matrix[:3, 3] = vec(element.get('xyz'))
    return matrix


def assemble():
    robot = ET.parse(URDF).getroot()
    manifest = json.loads((PROXYDIR / 'collision_replacement_manifest.json').read_text())
    proxy_to_source = {Path(p).name: source for source, record in manifest['generated_meshes'].items() for p in record['proxy_files']}
    frames = {'world': np.eye(4)}
    pending = list(robot.findall('joint'))
    while pending:
        progressed = False
        for joint in list(pending):
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in frames:
                continue
            motion = np.eye(4)
            if joint.get('type') != 'fixed':
                q = math.radians(POSE_DEG.get(joint.get('name'), 0))
                axis = vec(joint.find('axis').get('xyz'), (1, 0, 0))
                motion = trimesh.transformations.rotation_matrix(q, axis)
            frames[child] = frames[parent] @ origin(joint.find('origin')) @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('Unresolved joint tree')
    originals, current, records, seen = [], [], [], set()
    cache = {}

    def transformed(path, matrix, scale):
        if path not in cache:
            cache[path] = trimesh.load(path, force='mesh', process=False)
        mesh = cache[path]
        vertices = (mesh.vertices * scale) @ matrix[:3, :3].T + matrix[:3, 3]
        return vertices[mesh.faces]

    for link in robot.findall('link'):
        for collision in link.findall('collision'):
            element = collision.find('geometry/mesh')
            if element is None or 'collision_generated_ultralow/' not in element.get('filename', ''):
                continue  # Unchanged camera geometry is outside this comparison.
            filename = Path(element.get('filename')).name
            source = proxy_to_source[filename]
            local = origin(collision.find('origin'))
            matrix = frames[link.get('name')] @ local
            scale = vec(element.get('scale'), (1, 1, 1))
            triangles = transformed(PROXYDIR / filename, matrix, scale)
            current.append((triangles, TEAL if 'functional_finger' in filename else ORANGE))
            key = (link.get('name'), source, tuple(local.ravel()), tuple(scale))
            if key not in seen:
                original = transformed(MESHDIR / source, matrix, scale)
                originals.append((original, BLUE))
                records.append(dict(link=link.get('name'), source=source, original_triangles=len(original), proxies=[]))
                seen.add(key)
            record = next(r for r in records if r['link'] == link.get('name') and r['source'] == source)
            record['proxies'].append(dict(filename=filename, triangles=len(triangles)))
    return originals, current, records


def project(parts, azimuth=-62, elevation=20):
    a, e = np.radians([azimuth, elevation])
    camera = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.])
    up = np.cross(camera, right)
    basis = np.stack([right, up], axis=1)
    light = camera + right * -.45 + up * .85
    light /= np.linalg.norm(light)
    triangles = np.concatenate([p[0] for p in parts])
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-15)
    normals *= np.where(normals @ camera < 0, -1, 1)[:, None]
    shade = .43 + .57 * np.clip(normals @ light, 0, 1)
    colors = np.concatenate([np.broadcast_to(to_rgb(c), (len(t), 3)) for t, c in parts]) * shade[:, None]
    depth_order = np.argsort(triangles.mean(axis=1) @ camera, kind='stable')
    return (triangles @ basis)[depth_order], colors[depth_order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpi', type=int, default=260)
    args = parser.parse_args()
    original, current, records = assemble()
    n0, n1 = sum(len(t) for t, _ in original), sum(len(t) for t, _ in current)
    assert (n0, n1) == (398884, 832), (n0, n1)
    projected = [project(original), project(current)]
    all_points = np.concatenate([p[0].reshape(-1, 2) for p in projected])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    size = hi - lo
    lo, hi = lo - size * .05, hi + size * .05
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.2))
    fig.subplots_adjust(left=.025, right=.975, bottom=.16, top=.80, wspace=.05)
    for i, (ax, (tri, colors)) in enumerate(zip(axes, projected)):
        ax.add_collection(PolyCollection(tri, facecolors=colors, edgecolors='none' if i == 0 else '#77552f', linewidths=0 if i == 0 else .35, antialiaseds=i != 0, rasterized=True))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(('Original mesh', 'Current collision mesh')[i], fontsize=19, weight='bold', pad=20, color=(BLUE, ORANGE)[i])
        ax.text(.5, -.035, f'{(n0,n1)[i]:,} triangles  |  {(17,19)[i]} mesh instances', transform=ax.transAxes, ha='center', fontsize=12, color='#334155')
    fig.suptitle('SO-101  |  Collision mesh simplification', fontsize=22, x=.5, y=.965, weight='bold', color='#172b3b')
    fig.text(.5, .885, f'{100*(1-n1/n0):.2f}% fewer triangles  ·  Identical pose, view and scale', ha='center', fontsize=13, color='#475569')
    fig.text(.5, .065, 'Robot collision meshes only; unchanged cameras omitted. Visual meshes remain unchanged.', ha='center', fontsize=11, color='#526174')
    fig.text(.5, .033, 'All source triangles retained in this figure. Teal: separate finger collision components.', ha='center', fontsize=10, color='#64748b')
    png, pdf = OUT / 'robot_collision_comparison.png', OUT / 'robot_collision_comparison.pdf'
    fig.savefig(png, dpi=args.dpi, facecolor='white')
    fig.savefig(pdf, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    metadata = dict(scope='Robot STL collision meshes; unchanged cameras omitted', urdf=str(URDF), original_triangles=n0, current_triangles=n1, reduction_percent=100*(1-n1/n0), joint_pose_degrees=POSE_DEG, orthographic_camera=dict(azimuth=-62, elevation=20), records=records, rasterized_all_triangles=True, visual_meshes_unchanged=True)
    (OUT / 'robot_mesh_statistics.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(png=str(png), pdf=str(pdf), original_triangles=n0, current_triangles=n1), indent=2))


if __name__ == '__main__':
    main()
