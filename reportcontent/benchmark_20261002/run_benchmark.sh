#!/bin/bash
set -e
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$benchmark_dir/../.." && pwd)"
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
source "$project_dir/ros2_ws/install/setup.bash"
export PYTHONPATH="$project_dir/src:${PYTHONPATH:-}"
exec /usr/bin/python3 "$benchmark_dir/run_benchmark.py" "$@"
