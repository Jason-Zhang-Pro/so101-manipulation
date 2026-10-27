"""Set measured HOME, spawn one random target, settle and pause the report world."""
from pathlib import Path
import json
import math
import os
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import ListControllers
from ros_gz_interfaces.srv import ControlWorld

folder = Path(__file__).resolve().parent
meta = json.loads((folder/'scene_manifest.json').read_text())
assert os.environ.get('IGN_PARTITION') == meta['partition']
assert os.environ.get('GZ_PARTITION') == meta['partition']
assert os.environ.get('ROS_DOMAIN_ID') == str(meta['ros_domain_id'])
assert meta['ros_domain_id'] not in (20,21,22,23)
names = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
home = meta['home_joint_positions_rad']
rclpy.init()
node = rclpy.create_node('report_latest_scene_preparation')
state = {'joint': None, 'sim': 0., 'cubes': None}
targets = [0.] * 6
pub = node.create_publisher(Float64MultiArray,'/position_controller/commands',10)
def send():
    pub.publish(Float64MultiArray(data=targets))
node.create_timer(.1, send)
node.create_subscription(JointState,'/joint_states',lambda msg:state.update(joint=msg),qos_profile_sensor_data)
node.create_subscription(Clock,'/clock',lambda msg:state.update(sim=msg.clock.sec+msg.clock.nanosec*1e-9),qos_profile_sensor_data)
node.create_subscription(String,'/target_cube/states',lambda msg:state.update(cubes=json.loads(msg.data)),10)
def service(kind,name,request,timeout=90):
    c=node.create_client(kind,name)
    if not c.wait_for_service(timeout_sec=timeout):raise RuntimeError('Missing service '+name)
    f=c.call_async(request)
    end=time.monotonic()+timeout
    while not f.done() and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
    if not f.done():raise RuntimeError('Service timed out '+name)
    response=f.result()
    if response is None:raise RuntimeError('Empty service response '+name)
    return response
def errors():
    msg=state['joint']
    if msg is None or not all(n in msg.name for n in names) or len(msg.velocity)!=len(msg.name):return None
    pos=[msg.position[msg.name.index(n)] for n in names]
    vel=[msg.velocity[msg.name.index(n)] for n in names]
    return max(abs(a-b) for a,b in zip(pos,targets)),max(abs(v) for v in vel),pos,vel
def settle(label,need_cube=False):
    end=time.monotonic()+150
    stable_start=None
    while time.monotonic()<end:
        rclpy.spin_once(node,timeout_sec=.05)
        e=errors()
        cubes=state['cubes']
        valid=e is not None and e[0]<=math.radians(1) and e[1]<=.03
        if need_cube:
            valid=valid and cubes is not None and cubes.get('cube_count')==1
            if valid:valid=.762 <= cubes['cubes'][0]['position']['z'] <= .769
        if valid:
            if stable_start is None:stable_start=state['sim']
            if state['sim']-stable_start>=1.:
                print(label, {'position_error_rad':e[0],'speed_rad_s':e[1],'sim_time_s':state['sim']},flush=True)
                return e
        else:stable_start=None
    raise RuntimeError('Could not establish stable '+label+'; last feedback '+repr(errors()))
try:
    end=time.monotonic()+100
    while True:
        controllers=service(ListControllers,'/controller_manager/list_controllers',ListControllers.Request())
        if any(c.name=='position_controller' and c.state=='active' for c in controllers.controller):break
        if time.monotonic()>end:raise RuntimeError('Position controller did not activate')
        rclpy.spin_once(node,timeout_sec=.2)
    settle('ZERO confirmed')
    targets[:]=home
    settle('HOME confirmed')
    if state['cubes'] is not None and state['cubes'].get('cube_count',0)!=0:raise RuntimeError('Fresh report world unexpectedly already contains a target')
    result=service(Trigger,'/target_cube/spawn_random',Trigger.Request(),timeout=20)
    if not result.success:raise RuntimeError(result.message)
    print('Random target created:',result.message,flush=True)
    e=settle('HOME and one settled target confirmed',need_cube=True)
    request=ControlWorld.Request()
    request.world_control.pause=True
    response=service(ControlWorld,'/world/empty/control',request,timeout=15)
    if not response.success:raise RuntimeError('Report-world pause rejected')
    evidence={'home_target_rad':home,'joint_positions_rad':e[2],'joint_velocities_rad_s':e[3],
              'max_home_error_rad':e[0],'max_joint_speed_rad_s':e[1],
              'target_spawn':json.loads(result.message),'target_feedback':state['cubes'],
              'paused':True,'sim_time_s':state['sim'],'partition':meta['partition']}
    (folder/'ready.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('READY: latest world paused at HOME with exactly one random target.',flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
