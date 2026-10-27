"""Stop only the latest-world report instance, never training worlds."""
from pathlib import Path
import json
import os
import signal
import time

folder=Path(__file__).resolve().parent
meta=json.loads((folder/'scene_manifest.json').read_text())
state=json.loads((folder/'preview_process.json').read_text())
partition=meta['partition']
assert partition=='report_latest_so101_20261002_78f1'
def owned(pid):
    try:
        return f'IGN_PARTITION={partition}'.encode() in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    except OSError:return False
pids=[int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and owned(int(p.name))]
for pid in (state['gui_pid'],state['pid']):
    if owned(pid):os.kill(pid,signal.SIGINT)
deadline=time.monotonic()+6
while time.monotonic()<deadline and any(owned(p) for p in pids):time.sleep(.1)
for pid in pids:
    if owned(pid):os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+3
while time.monotonic()<deadline and any(owned(p) for p in pids):time.sleep(.1)
remaining=[p for p in pids if owned(p)]
result={'partition':partition,'closed_pids':pids,'remaining_pids':remaining,
        'background_training_pid':meta['training_pid_at_precheck'],
        'background_training_still_present':Path('/proc',str(meta['training_pid_at_precheck'])).exists()}
(folder/'closed.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if remaining:raise SystemExit('Some report processes still stopping.')
