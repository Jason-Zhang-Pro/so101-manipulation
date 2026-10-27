#!/usr/bin/env bash
# Inference-only paired evaluation; omit --execute to print its plan.
set -e

EVAL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_CONDA_SETUP="${EVAL_CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
EVAL_ROS_SETUP="${EVAL_ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "$EVAL_CONDA_SETUP"
conda activate "${EVAL_CONDA_ENV:-lerobot}"
source "$EVAL_ROS_SETUP"
source "$EVAL_PROJECT_ROOT/ros2_ws/install/setup.bash"
export PYTHONNOUSERSITE=1
# Keep ROS message/rclpy paths installed by the setup files.
export PYTHONPATH="$EVAL_PROJECT_ROOT/src:$EVAL_PROJECT_ROOT/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$EVAL_PROJECT_ROOT"
exec python -u -m so101_capstone.training.act_paired_evaluation "$@"
