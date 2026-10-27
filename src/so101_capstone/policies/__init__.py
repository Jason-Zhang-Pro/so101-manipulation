"""Policy adapters and simulation calibration for SO-101."""

from importlib import import_module as _import_module


# Calibration is also used by the ROS environment, which does not require
# ACT's Torch/LeRobot dependencies. Resolve every public export on demand.
_EXPORT_MODULES = {
    name: module
    for module, names in {
        ".act_checkpoint_loader": (
            "ACT_CONFIG_FILENAME", "ACT_MODEL_FILENAME", "LoadedActCheckpoint",
            "load_act_checkpoint", "load_act_config", "validate_act_config",
        ),
        ".act_policy_adapter": (
            "ACT_FIXED_IMAGE_KEY", "ACT_HANDEYE_IMAGE_KEY", "ACT_IMAGE_CHANNELS",
            "ACT_IMAGE_HEIGHT", "ACT_IMAGE_WIDTH", "ACT_STATE_KEY",
            "ActPolicyAdapter", "ActPolicyLike", "ActPolicyResult",
        ),
        ".policy_episode_runner": (
            "InMemoryPolicyTransitionSink", "PolicyEpisodeActionResult", "PolicyEpisodePolicy",
            "PolicyEpisodeResult", "PolicyEpisodeState", "PolicyEpisodeRunner",
            "PolicyEpisodeRuntime", "PolicyEpisodeRuntimeStep", "PolicyTransition", "PolicyTransitionSink",
        ),
        ".policy_multi_episode_runner": (
            "PolicyEpisodeSinkFactory", "PolicyEpisodeSummary", "PolicyMultiEpisodeResult", "PolicyMultiEpisodeRunner",
        ),
        ".policy_episode_npz": (
            "POLICY_EPISODE_NPZ_SCHEMA_VERSION", "PolicyEpisodeNpzSink", "write_policy_episode_npz",
        ),
        ".so101_sim_calibration": (
            "SIM_JOINT_CALIBRATIONS", "SIM_POLICY_JOINT_COUNT", "SIM_POLICY_JOINT_NAMES",
            "SimJointCalibration", "gazebo_positions_to_policy", "policy_positions_to_gazebo",
        ),
        ".act_gazebo_episode": ("ActGazeboEpisodeResult", "run_act_gazebo_episode"),
        ".act_gazebo_multi_episode": ("ActGazeboMultiEpisodeResult", "run_act_gazebo_multi_episode"),
    }.items()
    for name in names
}

__all__ = [
    "ACT_CONFIG_FILENAME",
    "ACT_FIXED_IMAGE_KEY",
    "ACT_HANDEYE_IMAGE_KEY",
    "ACT_IMAGE_CHANNELS",
    "ACT_IMAGE_HEIGHT",
    "ACT_IMAGE_WIDTH",
    "ACT_MODEL_FILENAME",
    "ACT_STATE_KEY",
    "ActGazeboEpisodeResult",
    "ActGazeboMultiEpisodeResult",
    "ActPolicyAdapter",
    "ActPolicyLike",
    "ActPolicyResult",
    "InMemoryPolicyTransitionSink",
    "LoadedActCheckpoint",
    "POLICY_EPISODE_NPZ_SCHEMA_VERSION",
    "PolicyEpisodeActionResult",
    "PolicyEpisodeNpzSink",
    "PolicyEpisodeSinkFactory",
    "PolicyEpisodePolicy",
    "PolicyEpisodeResult",
    "PolicyEpisodeState",
    "PolicyEpisodeSummary",
    "PolicyMultiEpisodeResult",
    "PolicyMultiEpisodeRunner",
    "PolicyEpisodeRunner",
    "PolicyEpisodeRuntime",
    "PolicyEpisodeRuntimeStep",
    "PolicyTransition",
    "PolicyTransitionSink",
    "SIM_JOINT_CALIBRATIONS",
    "SIM_POLICY_JOINT_COUNT",
    "SIM_POLICY_JOINT_NAMES",
    "SimJointCalibration",
    "gazebo_positions_to_policy",
    "load_act_checkpoint",
    "run_act_gazebo_episode",
    "run_act_gazebo_multi_episode",
    "load_act_config",
    "policy_positions_to_gazebo",
    "validate_act_config",
    "write_policy_episode_npz",
]


def __getattr__(name: str):
    """Keep the package API without loading unrelated runtime dependencies."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
