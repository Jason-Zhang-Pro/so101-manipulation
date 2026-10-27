#!/usr/bin/python3
"""Read-only verification of this benchmark's process cleanup."""
import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import psutil

root = Path(__file__).resolve().parent
remaining = []
training = []
for process in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        env = process.environ()
        partition = env.get('IGN_PARTITION', env.get('GZ_PARTITION', ''))
        if partition.startswith('so101bench_'):
            remaining.append({'pid': process.pid, 'name': process.info['name'], 'partition': partition})
        cmd = ' '.join(process.info['cmdline'] or [])
        if 'so101_capstone.training.act_ppo_multiworld_gazebo' in cmd:
            training.append({'pid': process.pid, 'name': process.info['name']})
    except psutil.Error:
        pass
owned_pids = set()
for dataset in ['trials', 'independent_free_run', 'excluded_preliminary']:
    for path in (root / dataset).glob('*/result.json'):
        data = json.loads(path.read_text())
        for world in data.get('world_specs', []):
            owned_pids.update(world[key] for key in ['launch_pid', 'worker_pid'] if key in world)
alive = []
for pid in owned_pids:
    try:
        process = psutil.Process(pid)
        alive.append({'pid': pid, 'name': process.name(), 'status': process.status()})
    except psutil.NoSuchProcess:
        pass
result = {
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'benchmark_partition_processes_remaining': remaining,
    'recorded_launch_or_worker_pids_still_alive': alive,
    'act_ppo_training_processes': training,
    'system_available_mib': psutil.virtual_memory().available / 2**20,
    'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(),
}
(root / 'verification_host_cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
