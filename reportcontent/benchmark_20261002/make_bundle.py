#!/usr/bin/python3
"""Package final reports, measured data, provenance and reproduction scripts."""
from pathlib import Path
import hashlib
import zipfile

root = Path(__file__).resolve().parent
destination = root / 'so101_benchmark_results_zh_en.zip'
files = set()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.py', '.sh', '.json', '.diff', '.urdf', '.sdf', '.config', '.txt'}:
        files.add(path)
for folder in ['reports', 'trials', 'independent_free_run', 'excluded_preliminary',
               'pilot_latest', 'pilot_latest_v2', 'pilot_gui4']:
    for path in (root / folder).rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix in {
            '.json', '.jsonl', '.log', '.html', '.csv', '.png', '.svg', '.pdf', '.txt'}:
            files.add(path)
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(files):
        archive.write(path, str(Path('so101_benchmark_20261002') / path.relative_to(root)))
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
(root / (destination.name + '.sha256')).write_text(f'{digest}  {destination.name}\n')
print(f'{destination}\nfiles={len(files)}\nMiB={destination.stat().st_size / 2**20:.2f}\nSHA256={digest}')
