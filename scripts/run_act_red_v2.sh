#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export PYTHONPATH="$PROJECT_DIR/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
MODEL_RUN_DIR="${MODEL_RUN_DIR:-$PROJECT_DIR/outputs/train/act_red_v2_c50_te001_b12_80k_20260806_000623}"
CHECKPOINT="${1:-last}"
POLICY_DIR="$MODEL_RUN_DIR/checkpoints/$CHECKPOINT/pretrained_model"

ROBOT_PORT="${ROBOT_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C124039-if00}"
ROBOT_ID="${ROBOT_ID:-my_awesome_follower_arm}"
FIXED_CAMERA="${FIXED_CAMERA:-/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0}"
HANDEYE_CAMERA="${HANDEYE_CAMERA:-/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0}"
EPISODE_SECONDS="${EPISODE_SECONDS:-20}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5}"

for required_path in \
  "$PYTHON_BIN" \
  "$POLICY_DIR/config.json" \
  "$POLICY_DIR/model.safetensors" \
  "$ROBOT_PORT" \
  "$FIXED_CAMERA" \
  "$HANDEYE_CAMERA"
do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path is missing: $required_path" >&2
    exit 1
  fi
done

RESOLVED_CHECKPOINT="$(basename "$(dirname "$(readlink -f "$POLICY_DIR")")")"
RUN_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
MODEL_NAME="$(basename "$MODEL_RUN_DIR")"
RUN_NAME="eval_${MODEL_NAME}_${RESOLVED_CHECKPOINT}_${RUN_STAMP}"
OUTPUT_ROOT="$PROJECT_DIR/jasonz/$RUN_NAME"

echo "Policy:     $POLICY_DIR"
echo "Checkpoint: $RESOLVED_CHECKPOINT"
echo "Robot:      $ROBOT_PORT"
echo "Fixed cam:  $FIXED_CAMERA"
echo "Hand cam:   $HANDEYE_CAMERA"
echo "Output:     $OUTPUT_ROOT"
echo
echo "The arm will move under policy control. Keep the power switch within reach."

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.cameras="{fixed: {type: opencv, index_or_path: '$FIXED_CAMERA', width: 640, height: 480, fps: 30}, handeye: {type: opencv, index_or_path: '$HANDEYE_CAMERA', width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id="jasonz/$RUN_NAME" \
  --dataset.root="$OUTPUT_ROOT" \
  --dataset.single_task="Pick up the red cube" \
  --dataset.fps=30 \
  --dataset.episode_time_s="$EPISODE_SECONDS" \
  --dataset.num_episodes=1 \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --policy.path="$POLICY_DIR" \
  --policy.device=cuda \
  --display_data=true
