#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/tiger/New Volume/ubuntufile/capstone"
BASE_POLICY_TIMESTAMP="20260803_143945"
CHECKPOINT="$ROOT/output_simulation/pretrained/act_so101_pick_test_${BASE_POLICY_TIMESTAMP}/training_output/checkpoints/060000/pretrained_model"

FINETUNE_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIRECTORY="$ROOT/output_simulation/finetuned/residual_ppo_${FINETUNE_TIMESTAMP}_from_pretrained_${BASE_POLICY_TIMESTAMP}"

cd "$ROOT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot

# ROS setup scripts may read variables that are not defined yet. Temporarily
# disable nounset only while sourcing the ROS environments.
set +u
source /opt/ros/humble/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src:$ROOT/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID=20
export GZ_PARTITION=gz_env_20
export IGN_PARTITION=gz_env_20

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "[ERROR] ACT checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$ROOT/output_simulation/finetuned"
echo "[INFO] base_policy_timestamp=$BASE_POLICY_TIMESTAMP"
echo "[INFO] output_directory=$OUTPUT_DIRECTORY"

python -m so101_capstone.training.residual_ppo_gazebo \
  "$@" \
  --act-checkpoint "$CHECKPOINT" \
  --device cuda \
  --episode-count 6 \
  --max-steps-per-episode 300 \
  --target-delay-s 3.0 \
  --rollout-steps 600 \
  --residual-limit-rad 0.1 \
  --disable-max-joint-delta-limit \
  --output-directory "$OUTPUT_DIRECTORY"
