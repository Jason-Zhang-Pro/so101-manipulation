#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path('/media/tiger/New Volume/ubuntufile/capstone')
PACKAGE_PREFIX = 'package://so101_description/meshes/'
GRASP_CRITICAL_MESHES = {
    'moving_jaw_so101_v1.stl',
    'wrist_roll_follower_so101_v1.stl',
}
DEPENDENCIES = {
    'numpy': 'numpy',
    'trimesh': 'trimesh>=4.4',
    'matplotlib': 'matplotlib>=3.5',
    'coacd': 'coacd==1.0.11',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--install-deps', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def ensure_dependencies(install):
    missing = []
    for module, requirement in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(requirement)
    if not missing:
        return
    if not install:
        quoted = ' '.join(f'"{x}"' for x in missing)
        raise SystemExit(
            'Missing Python dependencies:\n  ' + '\n  '.join(missing)
            + '\n\nRe-run with --install-deps, or install manually:\n'
            + f'  {sys.executable} -m pip install {quoted}'
        )
    print('Installing missing dependencies:')
    for item in missing:
        print('  ' + item)
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)


def import_deps():
    import numpy as np
    import trimesh
    import coacd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    return np, trimesh, coacd, plt, Poly3DCollection


def resolve_paths(project_root):
    description_root = project_root / 'ros2_ws' / 'src' / 'so101_description'
    return description_root / 'urdf' / 'so101.urdf.xacro', description_root / 'meshes'


def parse_collision_mesh_uses(xacro_path):
    root = ET.parse(xacro_path).getroot()
    uses = []
    for link in root.findall('./link'):
        link_name = link.get('name', '')
        for idx, collision in enumerate(link.findall('./collision')):
            mesh = collision.find('./geometry/mesh')
            if mesh is None:
                continue
            filename = mesh.get('filename')
            if not filename or not filename.startswith(PACKAGE_PREFIX):
                continue
            origin = collision.find('./origin')
            uses.append({
                'link': link_name,
                'collision_index': idx,
                'collision_name': collision.get('name'),
                'mesh_uri': filename,
                'mesh_relative': filename[len(PACKAGE_PREFIX):],
                'scale': mesh.get('scale'),
                'origin_xyz': origin.get('xyz') if origin is not None else '0 0 0',
                'origin_rpy': origin.get('rpy') if origin is not None else '0 0 0',
            })
    return uses


def load_mesh(trimesh, path):
    loaded = trimesh.load(path, force='mesh', process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, 'faces')]
        if not geoms:
            raise RuntimeError(f'No triangle geometry in {path}')
        loaded = trimesh.util.concatenate(geoms)
    if loaded.faces is None or len(loaded.faces) == 0:
        raise RuntimeError(f'Mesh has no faces: {path}')
    loaded.remove_unreferenced_vertices()
    return loaded


def profile(mesh_name, seed):
    if mesh_name in GRASP_CRITICAL_MESHES:
        return 'grasp_critical_2hull', {
            'threshold': 0.12,
            'max_convex_hull': 2,
            'preprocess_mode': 'auto',
            'preprocess_resolution': 35,
            'resolution': 1500,
            'mcts_nodes': 10,
            'mcts_iterations': 80,
            'mcts_max_depth': 3,
            'merge': True,
            'decimate': True,
            'max_ch_vertex': 32,
            'pca': False,
            'apx_mode': 'ch',
            'seed': seed,
        }
    return 'ultralow_1hull', {
        'threshold': 1.0,
        'max_convex_hull': 1,
        'preprocess_mode': 'auto',
        'preprocess_resolution': 25,
        'resolution': 1000,
        'mcts_nodes': 10,
        'mcts_iterations': 60,
        'mcts_max_depth': 2,
        'merge': True,
        'decimate': True,
        'max_ch_vertex': 24,
        'pca': False,
        'apx_mode': 'ch',
        'seed': seed,
    }


def decompose(coacd, trimesh, source_mesh, mesh_name, seed):
    profile_name, kwargs = profile(mesh_name, seed)
    cmesh = coacd.Mesh(source_mesh.vertices, source_mesh.faces)
    raw = coacd.run_coacd(cmesh, **kwargs)
    parts = []
    for vertices, faces in raw:
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        part.remove_unreferenced_vertices()
        if len(part.faces):
            parts.append(part)
    if not parts:
        raise RuntimeError(f'CoACD returned no usable parts for {mesh_name}')
    return profile_name, kwargs, parts


def combined_bounds(np, parts):
    mins = np.vstack([p.bounds[0] for p in parts])
    maxs = np.vstack([p.bounds[1] for p in parts])
    return np.vstack([mins.min(axis=0), maxs.max(axis=0)])


def safe_volume(mesh):
    try:
        value = float(abs(mesh.volume))
        if math.isfinite(value):
            return value
    except Exception:
        return None
    return None


def preview(np, plt, Poly3DCollection, source_mesh, parts, output_path, title):
    faces = source_mesh.faces
    if len(faces) > 2500:
        ids = np.linspace(0, len(faces)-1, 2500, dtype=int)
        faces = faces[ids]
    source_triangles = source_mesh.vertices[faces]

    fig = plt.figure(figsize=(14, 11))
    views = [('isometric', 24, 35), ('front', 0, -90), ('side', 0, 0), ('top', 90, -90)]
    all_points = np.vstack([source_mesh.vertices, *[p.vertices for p in parts]])
    center = all_points.mean(axis=0)
    radius = max(float((all_points.max(axis=0)-all_points.min(axis=0)).max()) * 0.58, 1e-9)
    palette = [(0.85,0.10,0.10), (0.10,0.35,0.85), (0.10,0.65,0.30), (0.75,0.45,0.05)]

    for i, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2,2,i, projection='3d')
        original = Poly3DCollection(source_triangles, alpha=0.38,
                                    facecolor=(0.18,0.18,0.18),
                                    edgecolor=(0.08,0.08,0.08,0.70), linewidth=0.22)
        ax.add_collection3d(original)
        for j, part in enumerate(parts):
            triangles = part.vertices[part.faces]
            color = palette[j % len(palette)]
            proxy = Poly3DCollection(triangles, alpha=0.26,
                                     facecolor=color, edgecolor=color, linewidth=0.45)
            ax.add_collection3d(proxy)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    fig.suptitle(title + '\nGrey = original STL, coloured = generated collision proxy')
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_html(rows, output_dir):
    body = []
    for r in rows:
        body.append(
            '<tr>'
            f"<td>{r['mesh']}</td><td>{r['profile']}</td><td>{r['original_faces']}</td>"
            f"<td>{r['proxy_parts']}</td><td>{r['proxy_faces_total']}</td>"
            f"<td>{r['face_reduction_percent']:.1f}%</td>"
            f"<td><a href=\"previews/{r['preview_file']}\">preview</a></td></tr>"
        )
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>SO-101 collision proxies</title>
<style>body{font-family:sans-serif;margin:28px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:7px;text-align:left}th{background:#eee}</style></head><body>
<h1>SO-101 ultra-low collision proxies</h1>
<p>Inspection only. Formal Xacro and original STL files were not modified.</p>
<p>Ordinary STL: maximum 1 convex hull. Grasp-critical STL: maximum 2 convex hulls.</p>
<table><thead><tr><th>Mesh</th><th>Profile</th><th>Original faces</th><th>Proxy parts</th><th>Proxy faces</th><th>Reduction</th><th>Overlay</th></tr></thead><tbody>''' + '\n'.join(body) + '</tbody></table></body></html>'
    (output_dir / 'index.html').write_text(html, encoding='utf-8')


def replacement_plan(uses, generated):
    lines = ['SO-101 ULTRA-LOW COLLISION REPLACEMENT PLAN', '', 'Preview only; source Xacro was NOT modified.', '']
    for use in uses:
        mesh_name = Path(use['mesh_relative']).name
        result = generated[mesh_name]
        lines.append(f"link={use['link']} source={mesh_name} profile={result['profile']} proxy_parts={len(result['proxy_files'])}")
        lines.append(f"  origin xyz=\"{use['origin_xyz']}\" rpy=\"{use['origin_rpy']}\"")
        if use['scale']:
            lines.append(f"  preserve scale=\"{use['scale']}\"")
        for f in result['proxy_files']:
            lines.append('  -> ' + f)
        lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    ensure_dependencies(args.install_deps)
    np, trimesh, coacd, plt, Poly3DCollection = import_deps()

    project_root = args.project_root.resolve()
    xacro_path, mesh_root = resolve_paths(project_root)
    if not xacro_path.is_file():
        raise FileNotFoundError(xacro_path)
    if not mesh_root.is_dir():
        raise FileNotFoundError(mesh_root)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output.resolve() if args.output else Path('/tmp') / f'so101_collision_ultralow_{timestamp}'
    mesh_output = output_dir / 'meshes'
    preview_output = output_dir / 'previews'
    mesh_output.mkdir(parents=True, exist_ok=False)
    preview_output.mkdir(parents=True, exist_ok=True)

    uses = parse_collision_mesh_uses(xacro_path)
    mesh_names = sorted({Path(u['mesh_relative']).name for u in uses})
    if not mesh_names:
        raise RuntimeError('No package collision meshes found in Xacro.')

    print(f'Source Xacro: {xacro_path}')
    print(f'Collision mesh uses: {len(uses)}')
    print(f'Unique collision STL files: {len(mesh_names)}')
    print(f'Output: {output_dir}')
    print('Profile: ordinary=max 1 hull; grasp-critical=max 2 hulls')
    print()

    generated = {}
    rows = []
    for n, mesh_name in enumerate(mesh_names, 1):
        source_path = mesh_root / mesh_name
        if not source_path.is_file():
            raise FileNotFoundError(f'Xacro references missing mesh: {source_path}')
        print(f'[{n}/{len(mesh_names)}] {mesh_name}')
        source_mesh = load_mesh(trimesh, source_path)
        profile_name, kwargs, parts = decompose(coacd, trimesh, source_mesh, mesh_name, args.seed)

        files = []
        for i, part in enumerate(parts):
            name = f'{source_path.stem}__{profile_name}__part_{i:02d}.stl'
            part.export(mesh_output / name)
            files.append(name)

        preview_name = source_path.stem + '.png'
        preview(np, plt, Poly3DCollection, source_mesh, parts, preview_output / preview_name,
                f'{mesh_name} — {profile_name}')

        original_faces = int(len(source_mesh.faces))
        proxy_faces = int(sum(len(p.faces) for p in parts))
        reduction = 100.0 * (1.0 - proxy_faces / original_faces) if original_faces else 0.0
        pbounds = combined_bounds(np, parts)
        obounds = np.asarray(source_mesh.bounds)

        generated[mesh_name] = {
            'source': str(source_path),
            'profile': profile_name,
            'coacd_parameters': kwargs,
            'proxy_files': files,
        }
        rows.append({
            'mesh': mesh_name,
            'profile': profile_name,
            'original_vertices': int(len(source_mesh.vertices)),
            'original_faces': original_faces,
            'proxy_parts': len(parts),
            'proxy_vertices_total': int(sum(len(p.vertices) for p in parts)),
            'proxy_faces_total': proxy_faces,
            'face_reduction_percent': reduction,
            'original_bounds': obounds.tolist(),
            'proxy_bounds': pbounds.tolist(),
            'bounds_delta_raw_units': (pbounds - obounds).tolist(),
            'original_volume_raw_units3': safe_volume(source_mesh),
            'proxy_volume_raw_units3': sum(v for v in (safe_volume(p) for p in parts) if v is not None),
            'preview_file': preview_name,
        })
        print(f'    {original_faces} faces -> {len(parts)} part(s), {proxy_faces} faces ({reduction:.1f}% fewer)')

    manifest = {
        'generated_at': datetime.now().isoformat(),
        'project_root': str(project_root),
        'source_xacro': str(xacro_path),
        'note': 'Inspection only. Source Xacro and original STL files were not modified.',
        'grasp_critical_meshes': sorted(GRASP_CRITICAL_MESHES),
        'generated_meshes': generated,
        'collision_uses': uses,
    }
    (output_dir / 'report.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (output_dir / 'collision_replacement_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    fields = ['mesh','profile','original_vertices','original_faces','proxy_parts','proxy_vertices_total','proxy_faces_total','face_reduction_percent','preview_file']
    with (output_dir / 'report.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r[k] for k in fields})

    (output_dir / 'replacement_plan.txt').write_text(replacement_plan(uses, generated), encoding='utf-8')
    (output_dir / 'README.txt').write_text(textwrap.dedent(f'''\
SO-101 ultra-low collision proxy generation
===========================================

Source:
  {xacro_path}

This run DID NOT modify:
  - so101.urdf.xacro
  - original STL files
  - joints / inertial parameters / controllers
  - reward / PPO code

Simplification policy:
  - ordinary collision STL: max 1 convex hull
  - moving_jaw_so101_v1.stl: max 2 convex hulls
  - wrist_roll_follower_so101_v1.stl: max 2 convex hulls
  - existing primitive collisions are untouched

Inspect:
  index.html
  report.csv
  previews/*.png

Generated collision STL files:
  meshes/

Do NOT copy these into the formal robot model yet. First inspect the overlays.
'''), encoding='utf-8')
    write_html(rows, output_dir)

    print('\nGeneration complete.')
    print(f'Output directory: {output_dir}')
    print(f"Open report: xdg-open '{output_dir / 'index.html'}'")
    print(f"Open folder: xdg-open '{output_dir}'")
    print('No project source file was modified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
