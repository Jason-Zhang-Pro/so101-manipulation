from so101_capstone.training.act_ppo_multiworld_cleanup import (
    ProcessSnapshot,
    managed_process_matches,
)


def process(pid, command, *, domain=None, partition=None):
    environment = {}
    if domain is not None:
        environment["ROS_DOMAIN_ID"] = str(domain)
    if partition is not None:
        environment["GZ_PARTITION"] = partition
        environment["IGN_PARTITION"] = partition
    return ProcessSnapshot(pid=pid, ppid=1, command=command, environment=environment)


def test_managed_sim_process_in_reserved_domain_is_selected():
    snapshot = process(123, "ros2 launch so101_bringup sim.launch.py gazebo_gui:=true", domain=20, partition="gz_env_20")
    assert managed_process_matches(snapshot, target_ros_domain_ids=(20, 21), target_gz_partitions=("gz_env_20", "gz_env_21"))


def test_target_manager_in_reserved_partition_is_selected():
    snapshot = process(124, "/usr/bin/python3 target_cube_manager.py", domain=21, partition="gz_env_21")
    assert managed_process_matches(snapshot, target_ros_domain_ids=(20, 21), target_gz_partitions=("gz_env_20", "gz_env_21"))


def test_same_command_in_unrelated_domain_is_not_selected():
    snapshot = process(125, "ros2 launch so101_bringup sim.launch.py", domain=30, partition="gz_env_30")
    assert not managed_process_matches(snapshot, target_ros_domain_ids=(20, 21), target_gz_partitions=("gz_env_20", "gz_env_21"))


def test_unrelated_process_in_reserved_domain_is_not_selected():
    snapshot = process(126, "python3 unrelated_data_analysis.py", domain=20, partition="gz_env_20")
    assert not managed_process_matches(snapshot, target_ros_domain_ids=(20, 21), target_gz_partitions=("gz_env_20", "gz_env_21"))


def test_current_process_can_be_explicitly_excluded():
    snapshot = process(127, "python -m so101_capstone.training.act_ppo_multiworld_gazebo", domain=20, partition="gz_env_20")
    assert not managed_process_matches(snapshot, target_ros_domain_ids=(20,), target_gz_partitions=("gz_env_20",), excluded_pids=(127,))
