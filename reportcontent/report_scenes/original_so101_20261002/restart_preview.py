"""Restart the report preview after checking its private partition."""
from pathlib import Path
import json
import os
import runpy
import signal
import time

folder = Path(__file__).resolve().parent
metadata = json.loads((folder / "scene_manifest.json").read_text())
state = json.loads((folder / "preview_process.json").read_text())
partition = metadata["partition"]

def is_preview(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
        return f'IGN_PARTITION={partition}'.encode() in raw
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False

pids = [int(path.name) for path in Path('/proc').iterdir()
        if path.name.isdigit() and is_preview(int(path.name))]
root_pid = state['pid']
if is_preview(root_pid):
    os.kill(root_pid, signal.SIGINT)
deadline = time.monotonic() + 5
while time.monotonic() < deadline and any(is_preview(pid) for pid in pids):
    time.sleep(0.1)
for pid in pids:
    if is_preview(pid):
        os.kill(pid, signal.SIGTERM)
time.sleep(0.3)
if any(is_preview(pid) for pid in pids):
    raise SystemExit('Previous preview still stopping; no duplicate launched.')
print('Stopped only private report partition PIDs:', pids)
runpy.run_path(str(folder / 'launch_preview.py'), run_name='__main__')
