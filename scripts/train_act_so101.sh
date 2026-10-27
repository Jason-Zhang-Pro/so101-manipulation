#!/usr/bin/env bash
set -Eeuo pipefail


# ============================================================
# 1. 环境配置
# ============================================================

# 加载 Conda 并激活 LeRobot 环境。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot

# 防止加载 ~/.local 中其他 Python 环境的软件包。
export PYTHONNOUSERSITE=1

# LeRobot 源代码目录。
LEROBOT_ROOT="/media/tiger/New Volume/ubuntufile/capstone/lerobot"

# 所有训练实验的根目录。
OUTPUT_ROOT="/media/tiger/New Volume/ubuntufile/capstone/outputs/train"

# 当前使用的本地 LeRobot 数据集。
# /media/tiger/New Volume/ubuntufile/capstone/huggingface/lerobot/tiger/so101_pick_test
DATASET_REPO_ID="tiger/so101_pick_test"

# 将 LeRobot 文本日志转换成 TensorBoard event 的程序。
TENSORBOARD_CONVERTER="/media/tiger/New Volume/ubuntufile/capstone/scripts/lerobot_log_to_tensorboard.py"


# ============================================================
# 2. 实验目录和文件
# ============================================================

# 实验基础名称。
RUN_BASE_NAME="act_so101_pick_test"

# 当前时间戳，例如：20260803_143600。
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 完整实验名称。
RUN_NAME="${RUN_BASE_NAME}_${RUN_TIMESTAMP}"

# 本次实验主目录。
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# LeRobot 模型、checkpoint 和训练状态保存目录。
TRAIN_OUTPUT_DIR="${RUN_DIR}/training_output"

# 本次实验的参数记录文件。
PARAM_FILE="${RUN_DIR}/training_parameters.txt"

# LeRobot 完整训练日志。
LOG_FILE="${RUN_DIR}/train.log"

# TensorBoard event 文件目录。
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

# TensorBoard 日志转换程序自身的日志。
TENSORBOARD_WRITER_LOG="${RUN_DIR}/tensorboard_writer.log"

# TensorBoard 服务器自身的日志。
TENSORBOARD_SERVER_LOG="${RUN_DIR}/tensorboard_server.log"

# TensorBoard 本地地址。
TENSORBOARD_HOST="127.0.0.1"
TENSORBOARD_PORT="6006"
TENSORBOARD_URL="http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT}"


# ============================================================
# 3. 通用训练参数
# ============================================================

# 每次优化更新使用的样本数量。
# 数值越大通常 GPU 利用率越高，但显存占用也越高。
BATCH_SIZE=30

# 数据加载后台进程数量。
# 用于读取 Parquet 文件和解码双相机视频。
NUM_WORKERS=16

# 总训练更新次数。
STEPS=1000000

# 每隔多少个 step 输出一次训练日志。
# TensorBoard 也会按照这个频率获得新的数据点。
LOG_FREQ=10

# 每隔多少个 step 保存一次 checkpoint。
SAVE_FREQ=5000


# ============================================================
# 4. ACT 和图像增强参数
# ============================================================

# ACT 每次预测的完整动作序列长度。
# 数据集为 30 FPS 时，15 个动作约覆盖 0.5 秒。
CHUNK_SIZE=15

# 每次模型推理后实际使用多少个动作，再重新观察和推理。
# Temporal ensembling 要求每个控制步都推理，因此必须为 1。
N_ACTION_STEPS=1

# 推理时融合不同时间对同一控制步的预测；不改变监督训练的损失。
# 设为 null 可关闭；后续 ACT-PPO 和评估应使用与 checkpoint 一致的配置。
TEMPORAL_ENSEMBLE_COEFF=0.01

# 训练图片仅做温和的亮度、对比度增强，不改变相机视野或畸变几何。
# 完整替换默认变换集合，不启用色相、饱和度、锐度变换。
IMAGE_TRANSFORMS_ENABLE=true
IMAGE_TRANSFORMS_MAX_NUM_TRANSFORMS=2
IMAGE_TRANSFORMS_RANDOM_ORDER=false
IMAGE_TRANSFORMS_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}}}'

# Transformer dropout 比例。
DROPOUT=0.1

# CVAE KL divergence 损失权重。
KL_WEIGHT=10.0

# ACT 主网络学习率。
LEARNING_RATE=1e-5


# ============================================================
# 5. 启动前检查
# ============================================================

# 确保动作执行数量没有超过动作块长度。
if (( N_ACTION_STEPS > CHUNK_SIZE )); then
    echo "[ERROR] N_ACTION_STEPS 不能大于 CHUNK_SIZE"
    echo "N_ACTION_STEPS=$N_ACTION_STEPS"
    echo "CHUNK_SIZE=$CHUNK_SIZE"
    exit 1
fi

if [[ "$TEMPORAL_ENSEMBLE_COEFF" != "null" ]] && (( N_ACTION_STEPS != 1 )); then
    echo "[ERROR] 开启 temporal ensembling 时 N_ACTION_STEPS 必须为 1"
    exit 1
fi

# 检查必要命令。
for REQUIRED_COMMAND in python lerobot-train tensorboard git tee; do
    if ! command -v "$REQUIRED_COMMAND" >/dev/null 2>&1; then
        echo "[ERROR] 找不到命令：$REQUIRED_COMMAND"
        exit 1
    fi
done

# 检查 LeRobot 源代码目录。
if [ ! -d "$LEROBOT_ROOT" ]; then
    echo "[ERROR] 找不到 LeRobot 目录："
    echo "$LEROBOT_ROOT"
    exit 1
fi

# 检查数据集。
DATASET_LOCAL_PATH="$HOME/.cache/huggingface/lerobot/$DATASET_REPO_ID"

if [ ! -f "$DATASET_LOCAL_PATH/meta/info.json" ]; then
    echo "[ERROR] 找不到数据集 meta/info.json："
    echo "$DATASET_LOCAL_PATH/meta/info.json"
    exit 1
fi

# 检查 TensorBoard 日志转换程序。
if [ ! -f "$TENSORBOARD_CONVERTER" ]; then
    echo "[ERROR] 找不到 TensorBoard 日志转换程序："
    echo "$TENSORBOARD_CONVERTER"
    exit 1
fi

# 防止覆盖相同时间戳的实验。
mkdir -p "$OUTPUT_ROOT"

if [ -e "$RUN_DIR" ]; then
    echo "[ERROR] 实验目录已经存在："
    echo "$RUN_DIR"
    exit 1
fi

# 创建本次实验需要的目录。
mkdir -p "$RUN_DIR"
mkdir -p "$TENSORBOARD_DIR"


# ============================================================
# 6. 检查 CUDA
# ============================================================

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("[ERROR] PyTorch 无法使用 CUDA")

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("[OK] CUDA 实际运算测试通过")
PY


# ============================================================
# 7. 收集软件和硬件版本
# ============================================================

PYTHON_VERSION="$(python --version 2>&1)"

PYTORCH_VERSION="$(
python - <<'PY'
import torch
print(torch.__version__)
PY
)"

PYTORCH_CUDA_VERSION="$(
python - <<'PY'
import torch
print(torch.version.cuda)
PY
)"

GPU_NAME="$(
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
PY
)"

GPU_CAPABILITY="$(
python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"

SUPPORTED_CUDA_ARCHITECTURES="$(
python - <<'PY'
import torch
print(",".join(torch.cuda.get_arch_list()))
PY
)"

LEROBOT_GIT_COMMIT="$(
git -C "$LEROBOT_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown"
)"

LEROBOT_GIT_BRANCH="$(
git -C "$LEROBOT_ROOT" branch --show-current 2>/dev/null || echo "unknown"
)"


# ============================================================
# 8. 写入本次训练参数
# ============================================================

cat > "$PARAM_FILE" <<EOF
SO-101 ACT Training Parameters
============================================================

Experiment
------------------------------------------------------------
run_name=$RUN_NAME
run_timestamp=$RUN_TIMESTAMP
run_directory=$RUN_DIR
training_output_directory=$TRAIN_OUTPUT_DIR
training_log=$LOG_FILE

TensorBoard
------------------------------------------------------------
tensorboard_directory=$TENSORBOARD_DIR
tensorboard_writer_log=$TENSORBOARD_WRITER_LOG
tensorboard_server_log=$TENSORBOARD_SERVER_LOG
tensorboard_url=$TENSORBOARD_URL

Dataset
------------------------------------------------------------
dataset.repo_id=$DATASET_REPO_ID
dataset.local_path=$DATASET_LOCAL_PATH
dataset.image_transforms.enable=$IMAGE_TRANSFORMS_ENABLE
dataset.image_transforms.max_num_transforms=$IMAGE_TRANSFORMS_MAX_NUM_TRANSFORMS
dataset.image_transforms.random_order=$IMAGE_TRANSFORMS_RANDOM_ORDER
dataset.image_transforms.tfs=$IMAGE_TRANSFORMS_TFS

Policy
------------------------------------------------------------
policy.type=act
policy.device=cuda
policy.chunk_size=$CHUNK_SIZE
policy.n_action_steps=$N_ACTION_STEPS
policy.temporal_ensemble_coeff=$TEMPORAL_ENSEMBLE_COEFF
policy.dropout=$DROPOUT
policy.kl_weight=$KL_WEIGHT
policy.optimizer_lr=$LEARNING_RATE

Training
------------------------------------------------------------
batch_size=$BATCH_SIZE
num_workers=$NUM_WORKERS
steps=$STEPS
log_freq=$LOG_FREQ
save_checkpoint=true
save_freq=$SAVE_FREQ
wandb.enable=false
policy.push_to_hub=false

Environment
------------------------------------------------------------
python_version=$PYTHON_VERSION
pytorch_version=$PYTORCH_VERSION
pytorch_cuda_runtime=$PYTORCH_CUDA_VERSION
gpu_name=$GPU_NAME
gpu_capability=$GPU_CAPABILITY
supported_cuda_architectures=$SUPPORTED_CUDA_ARCHITECTURES
lerobot_root=$LEROBOT_ROOT
lerobot_git_branch=$LEROBOT_GIT_BRANCH
lerobot_git_commit=$LEROBOT_GIT_COMMIT

Status
------------------------------------------------------------
training_status=initializing
EOF


# ============================================================
# 9. 后台进程及退出处理
# ============================================================

TB_WRITER_PID=""
TB_SERVER_PID=""
TRAIN_STATUS="running"

cleanup_background_processes() {
    local EXIT_CODE=$?

    # 关闭日志转换程序。
    if [ -n "$TB_WRITER_PID" ]; then
        kill "$TB_WRITER_PID" 2>/dev/null || true
        wait "$TB_WRITER_PID" 2>/dev/null || true
    fi

    # 关闭 TensorBoard 服务器。
    if [ -n "$TB_SERVER_PID" ]; then
        kill "$TB_SERVER_PID" 2>/dev/null || true
        wait "$TB_SERVER_PID" 2>/dev/null || true
    fi

    # 如果没有正常完成，记录中断或失败状态。
    if [ "$TRAIN_STATUS" = "running" ]; then
        if [ "$EXIT_CODE" -eq 130 ] || [ "$EXIT_CODE" -eq 143 ]; then
            FINAL_STATUS="interrupted"
        else
            FINAL_STATUS="failed"
        fi

        {
            echo
            echo "training_end_time=$(date --iso-8601=seconds)"
            echo "training_status=$FINAL_STATUS"
            echo "exit_code=$EXIT_CODE"
        } >> "$PARAM_FILE"
    fi
}

# 脚本退出时统一清理后台程序。
trap cleanup_background_processes EXIT

# Ctrl+C 统一返回退出码 130。
trap 'exit 130' INT

# 收到终止信号时返回退出码 143。
trap 'exit 143' TERM


# ============================================================
# 10. 显示训练配置
# ============================================================

echo "============================================================"
echo "训练名称：$RUN_NAME"
echo "数据集：$DATASET_REPO_ID"
echo "主目录：$RUN_DIR"
echo "模型输出：$TRAIN_OUTPUT_DIR"
echo "参数记录：$PARAM_FILE"
echo "训练日志：$LOG_FILE"
echo "TensorBoard：$TENSORBOARD_URL"
echo
echo "Batch size：$BATCH_SIZE"
echo "Num workers：$NUM_WORKERS"
echo "Steps：$STEPS"
echo "Log frequency：$LOG_FREQ"
echo "Save frequency：$SAVE_FREQ"
echo "Chunk size：$CHUNK_SIZE"
echo "Action steps：$N_ACTION_STEPS"
echo "Temporal ensemble coefficient：$TEMPORAL_ENSEMBLE_COEFF"
echo "Image transforms enabled：$IMAGE_TRANSFORMS_ENABLE"
echo "Image transforms：$IMAGE_TRANSFORMS_TFS"
echo "Dropout：$DROPOUT"
echo "KL weight：$KL_WEIGHT"
echo "Learning rate：$LEARNING_RATE"
echo "============================================================"


# ============================================================
# 11. 启动 TensorBoard 日志转换程序
# ============================================================

touch "$LOG_FILE"

python "$TENSORBOARD_CONVERTER" \
    --log-file "$LOG_FILE" \
    --log-dir "$TENSORBOARD_DIR" \
    > "$TENSORBOARD_WRITER_LOG" 2>&1 &

TB_WRITER_PID=$!

sleep 1

if ! kill -0 "$TB_WRITER_PID" 2>/dev/null; then
    echo "[ERROR] TensorBoard 日志转换程序启动失败"
    cat "$TENSORBOARD_WRITER_LOG"
    exit 1
fi


# ============================================================
# 12. 启动 TensorBoard 服务器
# ============================================================

tensorboard \
    --logdir "$TENSORBOARD_DIR" \
    --host "$TENSORBOARD_HOST" \
    --port "$TENSORBOARD_PORT" \
    --reload_interval 2 \
    > "$TENSORBOARD_SERVER_LOG" 2>&1 &

TB_SERVER_PID=$!

sleep 3

if ! kill -0 "$TB_SERVER_PID" 2>/dev/null; then
    echo "[ERROR] TensorBoard 服务器启动失败"
    echo "可能是 6006 端口已经被占用。"
    cat "$TENSORBOARD_SERVER_LOG"
    exit 1
fi

echo "[OK] TensorBoard 已启动：$TENSORBOARD_URL"

# 在 Ubuntu 桌面环境中自动打开浏览器。
if command -v xdg-open >/dev/null 2>&1; then
    (
        sleep 1
        xdg-open "$TENSORBOARD_URL" >/dev/null 2>&1 || true
    ) &
fi


# ============================================================
# 13. 开始训练
# ============================================================

cd "$LEROBOT_ROOT"

TRAIN_START_TIME="$(date --iso-8601=seconds)"

{
    echo
    echo "training_start_time=$TRAIN_START_TIME"
    echo "training_status=running"
} >> "$PARAM_FILE"

lerobot-train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.image_transforms.enable="$IMAGE_TRANSFORMS_ENABLE" \
    --dataset.image_transforms.max_num_transforms="$IMAGE_TRANSFORMS_MAX_NUM_TRANSFORMS" \
    --dataset.image_transforms.random_order="$IMAGE_TRANSFORMS_RANDOM_ORDER" \
    --dataset.image_transforms.tfs="$IMAGE_TRANSFORMS_TFS" \
    --policy.type=act \
    --output_dir="$TRAIN_OUTPUT_DIR" \
    --job_name="$RUN_NAME" \
    --policy.device=cuda \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --steps="$STEPS" \
    --log_freq="$LOG_FREQ" \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --policy.chunk_size="$CHUNK_SIZE" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --policy.temporal_ensemble_coeff="$TEMPORAL_ENSEMBLE_COEFF" \
    --policy.dropout="$DROPOUT" \
    --policy.kl_weight="$KL_WEIGHT" \
    --policy.optimizer_lr="$LEARNING_RATE" \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    2>&1 | tee "$LOG_FILE"


# ============================================================
# 14. 记录训练完成状态
# ============================================================

TRAIN_END_TIME="$(date --iso-8601=seconds)"
TRAIN_STATUS="completed"

{
    echo
    echo "training_end_time=$TRAIN_END_TIME"
    echo "training_status=completed"
    echo "exit_code=0"
} >> "$PARAM_FILE"

echo
echo "============================================================"
echo "[OK] 训练完成"
echo "实验目录：$RUN_DIR"
echo "参数文件：$PARAM_FILE"
echo "训练日志：$LOG_FILE"
echo "模型目录：$TRAIN_OUTPUT_DIR"
echo "TensorBoard 数据：$TENSORBOARD_DIR"
echo "============================================================"
