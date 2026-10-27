"""Capture the isolated display world's two actual pre-normalization ACT images."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import os
import time
import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from so101_capstone.environments.gazebo_world_control import GazeboWorldController
from so101_capstone.environments.image_cache import RgbImageCache, select_synchronized_camera_frames
from so101_capstone.environments.ros_message_adapter import update_rgb_image_cache_from_message
from so101_capstone.environments.top_camera_processing import TopCameraImageProcessor, default_simulation_top_camera_processing_config

folder=Path(__file__).resolve().parent
meta=json.loads((folder/'scene_manifest.json').read_text())
assert os.environ.get('ROS_DOMAIN_ID')==str(meta['ros_domain_id'])=='231'
assert os.environ.get('IGN_PARTITION')==os.environ.get('GZ_PARTITION')==meta['partition']
assert os.environ.get('SO101_IMAGE_AUGMENTATION_ENABLED')=='false'
assert os.environ.get('SO101_ENVIRONMENT_RANDOMIZATION_ENABLED')=='false'
config=default_simulation_top_camera_processing_config()
processor=TopCameraImageProcessor(config)
top=RgbImageCache(max_history=8,expected_shape=config.input_shape)
gripper=RgbImageCache(max_history=8,expected_shape=(480,640,3))
state={'sim_time_s':None,'joints':None,'targets':None}
rclpy.init()
node=rclpy.create_node('report_capture_policy_images')
node.create_subscription(Image,'/top_camera/image_raw',lambda msg:update_rgb_image_cache_from_message(top,msg),qos_profile_sensor_data)
node.create_subscription(Image,'/gripper_camera/image_raw',lambda msg:update_rgb_image_cache_from_message(gripper,msg),qos_profile_sensor_data)
node.create_subscription(Clock,'/clock',lambda msg:state.update(sim_time_s=msg.clock.sec+msg.clock.nanosec*1e-9),qos_profile_sensor_data)
node.create_subscription(JointState,'/joint_states',lambda msg:state.update(joints={'names':list(msg.name),'positions_rad':list(msg.position),'velocities_rad_s':list(msg.velocity)}),qos_profile_sensor_data)
node.create_subscription(String,'/target_cube/states',lambda msg:state.update(targets=json.loads(msg.data)),10)
control=GazeboWorldController(node,service_timeout_s=15,call_timeout_s=15)
pair=None
steps=0
try:
    control.pause()
    # Allow subscription discovery while this private world remains paused.
    until=time.monotonic()+2
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.1)
    for _ in range(6):
        control.step(34)
        steps+=34
        until=time.monotonic()+10
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.05)
            if state['sim_time_s'] is None:continue
            try:
                candidate=select_synchronized_camera_frames(top,gripper,
                    now_monotonic_s=time.monotonic(),max_age_s=.5,max_skew_s=.05,
                    min_source_timestamp_s=state['sim_time_s']-.05,
                    max_source_timestamp_s=state['sim_time_s']+1e-6)
            except (ValueError, RuntimeError):
                continue
            if state['joints'] is not None and state['targets'] is not None:
                pair=candidate
                break
        if pair is not None:break
    if pair is None:raise RuntimeError('No fresh synchronized camera pair from report world.')
    control.pause()
    out=folder/('policy_images_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    top_policy=processor.process(pair.top.image)
    gripper_policy=pair.gripper.image
    assert top_policy.shape==gripper_policy.shape==(480,640,3)
    for name,array in [('top_camera_policy',top_policy),('gripper_camera_policy',gripper_policy)]:
        path=out/(name+'.png')
        PILImage.fromarray(array).save(path)
        assert np.array_equal(np.array(PILImage.open(path)),array)
    np.savez_compressed(out/'raw_camera_frames.npz',top_rgb=pair.top.image,gripper_rgb=pair.gripper.image)
    evidence={'partition':meta['partition'],'ros_domain_id':231,
        'source':'live isolated report world, not background training',
        'top_processing':asdict(config),'image_augmentation_enabled':False,
        'environment_randomization_enabled':False,'output_rgb_shape':[480,640,3],
        'top_source_time_s':pair.top.source_timestamp_s,
        'gripper_source_time_s':pair.gripper.source_timestamp_s,
        'camera_skew_s':pair.skew_s,'extra_physics_steps_for_capture':steps,
        'paused_after_capture':True,**state}
    (out/'capture.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (folder/'latest_policy_capture.json').write_text(json.dumps({'directory':str(out)},indent=2)+'\n')
    print(json.dumps({'directory':str(out),'camera_skew_s':pair.skew_s,
                      'extra_sim_time_s':steps*.001,'training_pid_exists':Path('/proc',str(meta['training_pid_at_precheck'])).exists()},indent=2),flush=True)
finally:
    try:control.pause()
    finally:
        node.destroy_node()
        rclpy.shutdown()
