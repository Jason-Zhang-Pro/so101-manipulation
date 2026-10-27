#!/usr/bin/env bash
# Independent S0--S5 ACT tests; GUI by default, --execute starts simulation.
set -e

EVAL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_CONDA_SETUP="${EVAL_CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$EVAL_CONDA_SETUP"
conda activate "${EVAL_CONDA_ENV:-lerobot}"
source "${EVAL_ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "$EVAL_PROJECT_ROOT/ros2_ws/install/setup.bash"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EVAL_PROJECT_ROOT/src:$EVAL_PROJECT_ROOT/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$EVAL_PROJECT_ROOT"
exec python -u -m so101_capstone.training.act_stage_evaluation "$@"
