"""Launch only the isolated, paused original-model report preview."""
from pathlib import Path
import json
import os
import subprocess

folder = Path(__file__).resolve().parent
metadata = json.loads((folder / "scene_manifest.json").read_text())
env = os.environ.copy()
env.update({
    "IGN_PARTITION": metadata["partition"],
    "GZ_PARTITION": metadata["partition"],
    "ROS_DOMAIN_ID": str(metadata["ros_domain_id"]),
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "__GLX_VENDOR_LIBRARY_NAME": "mesa",
    "GALLIUM_DRIVER": "llvmpipe",
    "LP_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "CUDA_VISIBLE_DEVICES": "",
    "QT_QPA_PLATFORM": "xcb",
})
if not env.get("DISPLAY"):
    raise SystemExit("No desktop DISPLAY available; preview was not launched.")
command = [
    "nice", "-n", "15", "/usr/bin/ign", "gazebo", "-v", "3", "-z", "15",
    "--gui-config", str(folder / "gui.config"),
    str(folder / "original_so101.sdf"),
]
with (folder / "gazebo.log").open("ab", buffering=0) as log:
    process = subprocess.Popen(
        command, env=env, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
(folder / "preview_process.json").write_text(json.dumps({
    "pid": process.pid, "process_group": process.pid,
    "partition": metadata["partition"], "command": command,
}, indent=2) + "\n")
print(json.dumps({"preview_pid": process.pid, "partition": metadata["partition"],
                  "log": str(folder / "gazebo.log")}, indent=2))
