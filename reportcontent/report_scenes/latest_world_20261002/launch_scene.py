"""Close only the original report preview; open the installed current world."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time

folder = Path(__file__).resolve().parent
root = folder.parents[2]
old = folder.parent / 'original_so101_20261002'
old_meta = json.loads((old / 'scene_manifest.json').read_text())
old_state = json.loads((old / 'preview_process.json').read_text())
meta = json.loads((folder / 'scene_manifest.json').read_text())

def owns(pid, partition):
    try:
        return f'IGN_PARTITION={partition}'.encode() in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
    except OSError:
        return False

old_pids = [int(p.name) for p in Path('/proc').iterdir()
            if p.name.isdigit() and owns(int(p.name), old_meta['partition'])]
if owns(old_state['pid'], old_meta['partition']):
    os.kill(old_state['pid'], signal.SIGINT)
deadline = time.monotonic() + 5
while time.monotonic() < deadline and any(owns(pid, old_meta['partition']) for pid in old_pids):
    time.sleep(.1)
for pid in old_pids:
    if owns(pid, old_meta['partition']):
        os.kill(pid, signal.SIGTERM)
print('Closed original preview only:', old_pids, flush=True)
env = os.environ.copy()
env.update({'IGN_PARTITION': meta['partition'], 'GZ_PARTITION': meta['partition'],
            'ROS_DOMAIN_ID': str(meta['ros_domain_id']), 'ROS_LOCALHOST_ONLY': '1',
            'SO101_ENVIRONMENT_RANDOMIZATION_ENABLED': 'false',
            'SO101_IMAGE_AUGMENTATION_ENABLED': 'false',
            'ROS_LOG_DIR': str(folder / 'ros_logs'), 'PYTHONNOUSERSITE': '1',
            'LIBGL_ALWAYS_SOFTWARE': '1', '__GLX_VENDOR_LIBRARY_NAME': 'mesa',
            'GALLIUM_DRIVER': 'llvmpipe', 'LP_NUM_THREADS': '2',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': '',
            'QT_QPA_PLATFORM': 'xcb'})
def launch(name, command):
    with (folder / (name+'.log')).open('ab', buffering=0) as log:
        p = subprocess.Popen(['nice','-n','15',*command],env=env,cwd=root,
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                             start_new_session=True)
    print(name, p.pid, flush=True)
    return p.pid
setup = 'source /opt/ros/humble/setup.bash\nsource ros2_ws/install/setup.bash\n'
server = launch('gazebo', ['/bin/bash','-c',setup+'exec /opt/ros/humble/bin/ros2 launch so101_bringup sim.launch.py gazebo_gui:=false environment_randomization:=false'])
gui = launch('gui', ['/usr/bin/ign','gazebo','-g','-v','2','--gui-config',str(folder/'gui.config')])
prepare = launch('prepare', ['/bin/bash','-c',setup+'exec /usr/bin/python3 "$1"','prepare',str(folder/'prepare_scene.py')])
(folder/'preview_process.json').write_text(json.dumps({'pid':server,'gui_pid':gui,'prepare_pid':prepare,'partition':meta['partition']},indent=2)+'\n')
