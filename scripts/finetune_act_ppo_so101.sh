#!/usr/bin/env bash

# ============================================================
# 1. Pretrained ACT 模型
# ============================================================

# 所有下列参数都支持启动命令中的环境变量覆盖。
# 未提供环境变量时，使用脚本中写明的默认值。
#
# 示例：
# EPISODE_COUNT=20 TASK_MODE=mixed ./scripts/finetune_act_ppo_so101.sh

REWARD_PROFILE="${REWARD_PROFILE:-pick_place_v2}"
# full、单阶段s0..s5、原S0/S1混合s0_s1，或全部六阶段s0_s5。
TRAINING_STAGES="${TRAINING_STAGES:-s0_s1}"
STAGE_CONFIG="${STAGE_CONFIG:-}"
# 旧单episode随机选择概率；multi-world s0_s1调度不再使用此值。
# 新调度直接控制每次PPO batch中的有效ACT步数比例，无需按阶段时长换算。
if [[ "$TRAINING_STAGES" == "s0_s1" && -v S0_PROBABILITY ]]; then
    echo "提示：multi-world s0_s1不再使用S0_PROBABILITY；请设置S0_STEP_FRACTION控制每次update的实际步数比例。" >&2
fi
S0_PROBABILITY="${S0_PROBABILITY:-0.3333333333333333}"
S0_STEP_FRACTION="${S0_STEP_FRACTION:-0.4}"
# 六阶段有效采样步数权重，顺序S0:S1:S2:S3:S4:S5；仅s0_s5使用。
STAGE_STEP_WEIGHTS="${STAGE_STEP_WEIGHTS-1:1:1:1:1:1}"
# 绝对比例容差：0.05表示±5个百分点，不是相对误差5%。
STAGE_RATIO_TOLERANCE="${STAGE_RATIO_TOLERANCE:-0.05}"
# 达到此倍数的rollout目标仍未满足比例时，在下一wave前停止并报错。
STAGE_MAX_ROLLOUT_MULTIPLIER="${STAGE_MAX_ROLLOUT_MULTIPLIER:-4}"
# 本批某阶段有效episode成功率达标：不更新该阶段，下一完整采样批跳过一次。
# 跳过时其余阶段仍补满ROLLOUT_STEPS；再下一批恢复原比例。
STAGE_SUCCESS_SKIP_ENABLED="${STAGE_SUCCESS_SKIP_ENABLED:-true}"
STAGE_SUCCESS_SKIP_THRESHOLD="${STAGE_SUCCESS_SKIP_THRESHOLD:-0.8}"
# 仿真秒：非空值覆盖 STAGE_CONFIG，未设置或留空则继承 JSON。
# 默认 JSON：S0 最长5秒；六轴HOME误差<8度连续1秒即成功，不检查速度。
# S1 时间与通过条件继续读取所选 JSON。
S0_BUDGET_S="${S0_BUDGET_S:-}"
S0_HOME_HOLD_S="${S0_HOME_HOLD_S:-}"
S1_BUDGET_S="${S1_BUDGET_S:-}"
S1_REACH_HOLD_S="${S1_REACH_HOLD_S:-}"
S2_BUDGET_S="${S2_BUDGET_S:-}"
S3_BUDGET_S="${S3_BUDGET_S:-}"
S4_BUDGET_S="${S4_BUDGET_S:-}"
# S5 HOME预算继承S0；准备步骤不计入上述阶段预算或PPO样本。
STAGE_INITIALIZATION_TIMEOUT_S="${STAGE_INITIALIZATION_TIMEOUT_S:-}"
REWARD_CONFIG="${REWARD_CONFIG:-}"
TASK_MODE="${TASK_MODE:-mixed}"
IDLE_EMPTY_PROBABILITY="${IDLE_EMPTY_PROBABILITY:-0.1}"
IDLE_COMPLETED_PROBABILITY="${IDLE_COMPLETED_PROBABILITY:-0.1}"
# 保留 ACT 绝对位置动作；借鉴 final_good_act 的距离分档，按物理弧度设置噪声。
# stage_log_std 可显式选择旧固定阶段配置；以下 LOG_STD 组仅在该模式下生效。
EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-final_good_act}"
IDLE_EXPLORATION_LOG_STD="${IDLE_EXPLORATION_LOG_STD:--6.0}"
MOVING_EXPLORATION_LOG_STD="${MOVING_EXPLORATION_LOG_STD:--3.0}"
MANIPULATION_EXPLORATION_LOG_STD="${MANIPULATION_EXPLORATION_LOG_STD:--3.5}"
HOME_HOLD_EXPLORATION_LOG_STD="${HOME_HOLD_EXPLORATION_LOG_STD:--7.0}"
# 初始 ACT 只有不准确 reach：训练课程与独立夹爪探索，均为待仿真标定起点。
REACH_CURRICULUM="${REACH_CURRICULUM:-false}"
CURRICULUM_WINDOW_SIZE="${CURRICULUM_WINDOW_SIZE:-20}"
CURRICULUM_PASS_RATE="${CURRICULUM_PASS_RATE:-0.7}"
CURRICULUM_REHEARSAL_PROBABILITY="${CURRICULUM_REHEARSAL_PROBABILITY:-0.2}"
MOVING_GRIPPER_EXPLORATION_LOG_STD="${MOVING_GRIPPER_EXPLORATION_LOG_STD:--2.5}"
MANIPULATION_GRIPPER_EXPLORATION_LOG_STD="${MANIPULATION_GRIPPER_EXPLORATION_LOG_STD:--2.0}"
GRIPPER_MAXIMUM_LOG_STD="${GRIPPER_MAXIMUM_LOG_STD:--1.5}"
# 限制原生执行队列至最多5步（30Hz下约0.167秒），已启用TE的1步配置保持1步。
# 显式留空继承 checkpoint；此设置写入导出的 ACT config。
ACT_MAX_ACTION_STEPS="${ACT_MAX_ACTION_STEPS-5}"
if [[ "$REWARD_PROFILE" == "pick_place_v2" ]]; then
    for legacy_group in STATE1 STATE2 STATE3 STATE4_NOT_HOME STATE4_HOME; do
        for suffix in EXPLORATION_ENABLED ACTOR_UPDATE_ENABLED EXPLORATION_LOG_STD; do
            legacy_name="${legacy_group}_${suffix}"
            if [[ -v "$legacy_name" ]]; then
                echo "错误：$legacy_name 仅适用于 REWARD_PROFILE=legacy；新版使用 IDLE/MOVING/MANIPULATION/HOME_HOLD。" >&2
                exit 2
            fi
        done
    done
fi

PRETRAINED_MODEL="${PRETRAINED_MODEL:-/media/tiger/New Volume/ubuntufile/capstone/codex_workspace/checkpoints/act_0005000_te_off_zdtwineg}"


# ============================================================
# 2. Episode 与 rollout 设置
# ============================================================

# 有效采集 episode 总数的最低目标（不是抓取放置成功数）。
# 多 world 下按全部 world 合计；最后一轮已经启动的完整 episode 都会跑完，
# 因此最终有效 episode 数允许超过这个值。
# s0_s1模式还会补齐当前batch的步数与阶段比例，再执行最后一次update。
EPISODE_COUNT="${EPISODE_COUNT:-100}"

# 同一个 world 连续多少次 episode 未完整完成后，
# 本 rollout 暂停使用该 world；下一次 PPO update 后再重新尝试。
MAX_CONSECUTIVE_EPISODE_FAILURES="${MAX_CONSECUTIVE_EPISODE_FAILURES:-3}"

# ============================================================
# 2A. Multi-world Gazebo
# ============================================================

# 独立 Gazebo world 的并行上限。
# s0_s1按当前batch实际步数及各阶段episode均长规划下一wave，只启动选中的world；
# 接近达标时可只启动1或2个world，每个选中world仍运行完整episode。
# 每个 world 使用独立 ROS_DOMAIN_ID 和 Gazebo partition。
# 每轮完整 episode 的所有 steps 和汇总记录落盘后重建 world。
# WORLD_COUNT=1 也使用相同流程；不在存活 world 中删除 target。
WORLD_COUNT="${WORLD_COUNT:-1}"

# true：每个 world 打开一个 Gazebo GUI。
# false：每个 world 使用 Fortress -s server-only 模式。
GAZEBO_GUI="${GAZEBO_GUI:-true}"

# world 0 使用 BASE_ROS_DOMAIN_ID，后续 world 依次 +1。
BASE_ROS_DOMAIN_ID="${BASE_ROS_DOMAIN_ID:-20}"
GZ_PARTITION_PREFIX="${GZ_PARTITION_PREFIX:-gz_env_}"

# 初次启动一个 world 时，连续启动失败最多尝试多少次。
WORLD_STARTUP_MAX_ATTEMPTS="${WORLD_STARTUP_MAX_ATTEMPTS:-3}"

# finetune 运行中某个 world 的 episode 失败后：
#   >0：保持原行为，在当前 wave 内立即最多重建 N 次并重跑该 world。
#    0：当前 wave 不重建；等其他 world 完成本轮后，到下一 wave 边界再尝试。
WORLD_RESTART_MAX_ATTEMPTS="${WORLD_RESTART_MAX_ATTEMPTS:-3}"

# WORLD_RESTART_MAX_ATTEMPTS=0 时，每个 wave 边界对 pending world
# 最多连续重建多少次。仍失败则健康 world 继续，下一个 wave 边界再试。
WORLD_DEFERRED_RESTART_MAX_ATTEMPTS="${WORLD_DEFERRED_RESTART_MAX_ATTEMPTS:-3}"

# 启动本次 finetune 前，清理将占用 domain/partition 中遗留的
# SO-101 finetune / Gazebo / ROS simulation 进程。
PREFLIGHT_CLEANUP_ENABLED="${PREFLIGHT_CLEANUP_ENABLED:-true}"

# 先发送 SIGTERM；超过该墙钟时间仍未退出的遗留进程会收到 SIGKILL。
PREFLIGHT_CLEANUP_TIMEOUT_S="${PREFLIGHT_CLEANUP_TIMEOUT_S:-5.0}"

# full 模式的 episode 上限。S0/S1 使用阶段秒数预算（默认5/10秒）。
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-1200}"

# PPO update 的最低 transition 阈值，不是固定 batch size。
#
# 只在一整轮“本wave选中world的完整episode”结束后检查。
# 如果 4 个 world 各完成 600 steps，而 ROLLOUT_STEPS=2000，
# 则实际 2400 transitions 全部用于一次 PPO update，不截断、不留尾巴。
# s0_s1还要求本batch实际S0步数比例落入目标±容差；
# 下一wave根据实际计数重新选择同一个阶段和不超过WORLD_COUNT的world数量。
# 均长预测只用于规划，是否update以完整episode的实际有效步数为准。
# full模式仍由所有当前可用world开始完整episode。
# 到episode目标后，s0_s1会补齐当前batch；full模式仍丢弃不足阈值的尾批。
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2400}"

# 一个 PPO update 中，每次送入 GPU 做梯度计算的数据量
#
# RTX 5080 16 GB 建议先用 16
# 稳定后可以尝试 24 或 32
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16}"

# 同一批 rollout 数据重复优化多少轮
#
# 先扩大探索和初始学习率，保留2个epoch；KL超限回滚并缩小步长。
PPO_EPOCHS="${PPO_EPOCHS:-2}"


# ============================================================
# 3. PPO 核心参数
# ============================================================

# 奖励折扣因子
# 越接近 1，越重视较远未来的奖励
GAMMA="${GAMMA:-0.997}"

# GAE 参数
# 控制 advantage 估计的偏差与方差平衡
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"

# PPO ratio clipping 范围
# 限制一次 update 对 policy 的修改幅度
CLIP_RATIO="${CLIP_RATIO:-0.2}"
# 全 rollout 的有界截断 Gaussian KL(old||new) 均值上限。
# 超限时恢复 epoch，降低策略学习率后有界重试；仍超限则回滚并结束 update。
# 较小的策略步长会保留到后续 update 和 checkpoint；critic 学习率不变。
TARGET_KL="${TARGET_KL:-0.02}"
KL_BACKOFF_MAX_RETRIES="${KL_BACKOFF_MAX_RETRIES:-8}"
KL_BACKOFF_FACTOR="${KL_BACKOFF_FACTOR:-0.5}"
KL_BACKOFF_MIN_SCALE="${KL_BACKOFF_MIN_SCALE:-1e-4}"

# critic value loss 在总 loss 中的权重
VALUE_LOSS_COEFFICIENT="${VALUE_LOSS_COEFFICIENT:-0.5}"

# 有界截断 Gaussian 的 entropy bonus（仿射归一化动作坐标）。
# 先关闭额外熵奖励，探索幅度仍由下面的分阶段标准差控制。
ENTROPY_COEFFICIENT="${ENTROPY_COEFFICIENT:-0}"
MEAN_BOUNDS_COEFFICIENT="${MEAN_BOUNDS_COEFFICIENT:-0.1}"
# 保留旧参数兼容；新有界均值使该辅助越界损失为0。
# 限幅前 ACT 原始预测的越界辅助损失；按各关节半量程归一化。
# 0.1 是本次启用的初始权重，需结合 raw 越界日志和任务表现评估；设0可关闭。
RAW_MEAN_BOUNDS_COEFFICIENT="${RAW_MEAN_BOUNDS_COEFFICIENT:-0.1}"

# ACT、critic、探索参数分别裁剪，避免一组的大梯度压制另一组。
MAX_GRADIENT_NORM="${MAX_GRADIENT_NORM:-0.5}"
CRITIC_MAX_GRADIENT_NORM="${CRITIC_MAX_GRADIENT_NORM:-0.5}"
EXPLORATION_MAX_GRADIENT_NORM="${EXPLORATION_MAX_GRADIENT_NORM:-0.5}"


# ============================================================
# 4. Critic 配置
# ============================================================

# critic 使用的图像尺寸
#
# 只影响训练时的 value network。
# 不会改变 ACT 本身看到的图像尺寸。
CRITIC_IMAGE_HEIGHT="${CRITIC_IMAGE_HEIGHT:-96}"
CRITIC_IMAGE_WIDTH="${CRITIC_IMAGE_WIDTH:-128}"

# critic 隐藏层宽度
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-256}"

# critic optimizer 学习率
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-0.0002}"
EXPLORATION_LEARNING_RATE="${EXPLORATION_LEARNING_RATE:-0.0002}"


# ============================================================
# 5. ACT actor 学习率
# ============================================================

# 留空：
# 从 pretrained ACT config.json 继承 optimizer_lr
#
# 设置具体数值：
# 覆盖 pretrained ACT 的 actor 学习率
#
# 采用 final_good_act 的 2e-4 起点；真实有效 LR 仍由当前 KL 回退决定。
# 使用 ${VAR-default} 保留显式空字符串，让 Python 继承 checkpoint。
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE-2e-4}"

# 留空：
# 从 pretrained ACT config.json 继承 optimizer_lr_backbone
#
# 设置具体数值：
# 覆盖视觉 backbone 学习率
#
# 旧 MLP 没有图像 backbone；沿用 capstone 比 actor 小十倍的参数组比例。
# 显式留空仍继承 checkpoint 的 backbone 学习率。
ACTOR_BACKBONE_LEARNING_RATE="${ACTOR_BACKBONE_LEARNING_RATE-2e-5}"


# ============================================================
# 6. PPO 分 State 探索与 ACT actor update
# ============================================================

# 每个 State 都可以独立控制：
#
# EXPLORATION_ENABLED=true
#   rollout 使用 stochastic PPO action。
# EXPLORATION_ENABLED=false
#   rollout 直接使用 ACT mean action，不加入随机探索。
#
# ACTOR_UPDATE_ENABLED=true
#   该 State 的 transition 参与 PPO actor / entropy update。
# ACTOR_UPDATE_ENABLED=false
#   该 State 的 transition 不更新 ACT actor，但仍参与 critic/value update。
#
# 各 PPO stage 开关默认全部 true。

# State 1：无 target
STATE1_EXPLORATION_ENABLED="${STATE1_EXPLORATION_ENABLED:-true}"
STATE1_ACTOR_UPDATE_ENABLED="${STATE1_ACTOR_UPDATE_ENABLED:-true}"

# State 2：target 尚未进入 goal
STATE2_EXPLORATION_ENABLED="${STATE2_EXPLORATION_ENABLED:-true}"
STATE2_ACTOR_UPDATE_ENABLED="${STATE2_ACTOR_UPDATE_ENABLED:-true}"

# State 3：compare point 已进入 target 范围，正在 grasp / transport
STATE3_EXPLORATION_ENABLED="${STATE3_EXPLORATION_ENABLED:-true}"
STATE3_ACTOR_UPDATE_ENABLED="${STATE3_ACTOR_UPDATE_ENABLED:-true}"

# State 3：target 已在 goal，但 robot 尚未回 HOME
STATE4_NOT_HOME_EXPLORATION_ENABLED="${STATE4_NOT_HOME_EXPLORATION_ENABLED:-true}"
STATE4_NOT_HOME_ACTOR_UPDATE_ENABLED="${STATE4_NOT_HOME_ACTOR_UPDATE_ENABLED:-true}"

# State 3：robot 已回 HOME，正在累计保持帧
STATE4_HOME_EXPLORATION_ENABLED="${STATE4_HOME_EXPLORATION_ENABLED:-true}"
STATE4_HOME_ACTOR_UPDATE_ENABLED="${STATE4_HOME_ACTOR_UPDATE_ENABLED:-true}"

# ------------------------------------------------------------
# Exploration 噪声大小
# ------------------------------------------------------------

# 共享可学习 exploration 的初始 log standard deviation
# 截断前的尺度参数使用仿射归一化单位：
# arm 的 policy-unit std = 100 * exp(log_std)，gripper = 50 * exp(log_std)。
# 动作只在合法范围内采样；靠近边界时向内探索，实际标准差受截断影响。
#
# final_good_act 模式把物理 sigma 换算成逐步6维偏移；初始 -3 表示学习倍率为1。
# 改变初始基值不改变物理初始噪声。PPO仍可学习倍率；上界限制保持。
# stage_log_std 模式继续使用顶部固定阶段组；下方STATE开关仅legacy。
INITIAL_LOG_STD="${INITIAL_LOG_STD:--3.0}"

# 所有轴有效 exploration log std 下限；HOME 可向更静止的动作学习。
# -9.0 的手臂标准差约0.0123 policy units；S0初始-6约0.2479 units。
MINIMUM_LOG_STD="${MINIMUM_LOG_STD:--9.0}"

# 共享可学习基础值与机械臂实际 exploration log std 上限
# 夹爪增加阶段偏移后另受 GRIPPER_MAXIMUM_LOG_STD 限制。
# -2.5 对应归一化标准差约0.0821，手臂约8.21 policy units。
MAXIMUM_LOG_STD="${MAXIMUM_LOG_STD:--2.5}"

# State 1：无 target；保持很小噪声，使这些 transition 仍可更新 ACT
# -5.0 对应标准差约 0.0067
STATE1_EXPLORATION_LOG_STD="${STATE1_EXPLORATION_LOG_STD:--5.0}"

# State 2：target 尚未进入 goal
# -4.0 对应标准差约 0.0183
STATE2_EXPLORATION_LOG_STD="${STATE2_EXPLORATION_LOG_STD:--4.0}"

# State 3：grasp / transport；所有关节使用双向随机探索
# -4.0 对应标准差约 0.0183
STATE3_EXPLORATION_LOG_STD="${STATE3_EXPLORATION_LOG_STD:--4.0}"

# State 3：target 已在 goal，但 robot 尚未回 HOME
# -4.0 对应标准差约 0.0183
STATE4_NOT_HOME_EXPLORATION_LOG_STD="${STATE4_NOT_HOME_EXPLORATION_LOG_STD:--4.0}"

# State 3：robot 已回 HOME，正在累计 90 帧保持
# -5.0 对应标准差约 0.0067
STATE4_HOME_EXPLORATION_LOG_STD="${STATE4_HOME_EXPLORATION_LOG_STD:--5.0}"


# ============================================================
# 7. Target 与随机种子
# ============================================================

# episode 开始后多少秒生成 target
#
# 3 秒 × 30 Hz = 前 90 个 steps 没有 target
TARGET_DELAY_S="${TARGET_DELAY_S:-3.0}"
# 必须为 0：target 保留到 episode 完成，随后通过整 world 重建清理。
# 非零设置由托管训练入口拒绝，不能恢复旧的 episode 内删除行为。
FINAL_TARGET_ABSENT_S="${FINAL_TARGET_ABSENT_S:-0.0}"

# PPO、PyTorch、NumPy 等训练侧随机种子
#
# 课程开启时也控制方块位置；课程关闭时manager仍独立随机采样。
SEED="${SEED:-0}"


# ============================================================
# 8. Checkpoint 与 TensorBoard
# ============================================================

# 每多少次 PPO update 保存一次中间 checkpoint
#
# 1  = update 1、2、3……
# 5  = update 5、10、15……
# 10 = update 10、20、30……
#
# 无论这个值是多少，最终 pretrained_model 和
# final_training_state.pt 都会保存。
CHECKPOINT_EVERY_UPDATES="${CHECKPOINT_EVERY_UPDATES:-5}"

# TensorBoard 首选端口
# 如果端口被占用，程序会自动尝试后续端口。
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

# 等待 ROS joint/camera feedback 的最长墙钟时间
STARTUP_FEEDBACK_TIMEOUT_S="${STARTUP_FEEDBACK_TIMEOUT_S:-30.0}"


# ============================================================
# 9. 项目与运行设备
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
DEVICE="${DEVICE:-cuda}"

# Optional simulation-only rigid attachment after a real contact-confirmed grasp.
# Keep existing launchers on natural contact physics unless explicitly enabled.
ASSISTED_GRASP_ENABLED="${ASSISTED_GRASP_ENABLED:-false}"

# Physical environment randomization. When false, scene/pose/joint mutations
# and their extra reset warm-up are skipped. Image augmentation has its own
# independent switch below.
ENVIRONMENT_RANDOMIZATION="${ENVIRONMENT_RANDOMIZATION:-false}"
ENVIRONMENT_RANDOMIZATION_CONFIG="${ENVIRONMENT_RANDOMIZATION_CONFIG:-$PROJECT_ROOT/config/environment_randomization.json}"
ENVIRONMENT_RANDOMIZATION_SEED="${ENVIRONMENT_RANDOMIZATION_SEED:-0}"
# 只处理策略看到的两路图片，不改变Gazebo场景、灯光、相机位姿或关节测量。
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-true}"
IMAGE_AUGMENTATION_CONFIG="${IMAGE_AUGMENTATION_CONFIG:-$PROJECT_ROOT/config/image_augmentation_s0_s1.json}"

if [[ "$REWARD_PROFILE" == "pick_place_v2" ]]; then
    REWARD_CONFIG="${REWARD_CONFIG:-$PROJECT_ROOT/config/pick_place_reward.json}"
fi
if [[ "$TRAINING_STAGES" != "full" ]]; then
    case "$TRAINING_STAGES" in
        s2|s3|s4|s5|s0_s5) STAGE_CONFIG="${STAGE_CONFIG:-$PROJECT_ROOT/config/act_ppo_stages_s0_s5.json}";;
        *) STAGE_CONFIG="${STAGE_CONFIG:-$PROJECT_ROOT/config/act_ppo_stages_s1_lift_approach.json}";;
    esac
fi
cd "$PROJECT_ROOT"


# ============================================================
# 10. Python / ROS 环境
# ============================================================

source "${PPO_CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${PPO_CONDA_ENV:-lerobot}"

source "${PPO_ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "$PROJECT_ROOT/ros2_ws/install/setup.bash"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

# Parent process itself does not own a Gazebo runtime, but preserve the
# project baseline environment. Worker i overrides these with its own
# BASE_ROS_DOMAIN_ID+i and matching partition.
export ROS_DOMAIN_ID="$BASE_ROS_DOMAIN_ID"
export GZ_PARTITION="${GZ_PARTITION_PREFIX}${BASE_ROS_DOMAIN_ID}"
export IGN_PARTITION="$GZ_PARTITION"

# 减少 CUDA 显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${ASSISTED_GRASP_ENABLED,,}" in
    true|false) ;;
    *)
        echo "错误：ASSISTED_GRASP_ENABLED 必须设置为 true 或 false，当前值：$ASSISTED_GRASP_ENABLED"
        exit 1
        ;;
esac

case "${ENVIRONMENT_RANDOMIZATION,,}" in
    true|false)
        ;;
    *)
        echo "错误：ENVIRONMENT_RANDOMIZATION 必须设置为 true 或 false，当前值：$ENVIRONMENT_RANDOMIZATION"
        exit 1
        ;;
esac

if [[ "${ENVIRONMENT_RANDOMIZATION,,}" == "true" ]] &&
   [[ ! -f "$ENVIRONMENT_RANDOMIZATION_CONFIG" ]]; then
    echo "错误：找不到 environment randomization config："
    echo "$ENVIRONMENT_RANDOMIZATION_CONFIG"
    exit 1
fi

if ! [[ "$ENVIRONMENT_RANDOMIZATION_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "错误：ENVIRONMENT_RANDOMIZATION_SEED 必须是整数，当前值：$ENVIRONMENT_RANDOMIZATION_SEED"
    exit 1
fi

export SO101_ENVIRONMENT_RANDOMIZATION_ENABLED="$ENVIRONMENT_RANDOMIZATION"
export SO101_ENVIRONMENT_RANDOMIZATION_CONFIG="$ENVIRONMENT_RANDOMIZATION_CONFIG"
export SO101_ENVIRONMENT_RANDOMIZATION_SEED="$ENVIRONMENT_RANDOMIZATION_SEED"
export SO101_IMAGE_AUGMENTATION_ENABLED="$IMAGE_AUGMENTATION"
export SO101_IMAGE_AUGMENTATION_CONFIG="$IMAGE_AUGMENTATION_CONFIG"
export SO101_ASSISTED_GRASP_ENABLED="${ASSISTED_GRASP_ENABLED,,}"


# ============================================================
# 11. 自动生成输出目录
# ============================================================

FINETUNE_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MODEL_TAG="$(basename "$(dirname "$PRETRAINED_MODEL")")"
MODEL_TAG="$(
    printf '%s' "$MODEL_TAG" |
    tr ' /' '__' |
    tr -cd '[:alnum:]_.-'
)"

if [[ -z "$MODEL_TAG" ]]; then
    MODEL_TAG="custom_pretrained"
fi

OUTPUT_DIRECTORY="$PROJECT_ROOT/output_simulation/finetuned/act_ppo_${FINETUNE_TIMESTAMP}_from_${MODEL_TAG}"


# ============================================================
# 12. 检查 pretrained ACT
# ============================================================

if [[ ! -d "$PRETRAINED_MODEL" ]]; then
    echo "错误：找不到 pretrained model："
    echo "$PRETRAINED_MODEL"
    exit 1
fi

if [[ ! -f "$PRETRAINED_MODEL/config.json" ]]; then
    echo "错误：pretrained model 中没有 config.json："
    echo "$PRETRAINED_MODEL"
    exit 1
fi

if [[ ! -f "$PRETRAINED_MODEL/model.safetensors" ]] &&
   [[ ! -f "$PRETRAINED_MODEL/pytorch_model.bin" ]]; then
    echo "错误：pretrained model 中没有模型权重文件："
    echo "$PRETRAINED_MODEL"
    exit 1
fi


# ============================================================
# 13. 检查 CUDA
# ============================================================

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("错误：CUDA 不可用。")

properties = torch.cuda.get_device_properties(0)

print("CUDA device:", torch.cuda.get_device_name(0))
print(
    "CUDA total memory:",
    round(properties.total_memory / 1024**3, 2),
    "GiB",
)
PY


# ============================================================
# 14. 检查 Gazebo / ROS 启动工具
# ============================================================

# Gazebo 不再要求手动提前启动；multi-world trainer 会自行启动和关闭。
if ! command -v ros2 >/dev/null 2>&1; then
    echo "错误：找不到 ros2 命令。"
    exit 1
fi

if ! command -v ign >/dev/null 2>&1; then
    echo "错误：找不到 ign 命令。"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/output_simulation/finetuned"


# ============================================================
# 15. 可选 CLI 参数
# ============================================================

ACTOR_LEARNING_RATE_ARGS=()
STATE_SWITCH_ARGS=()
GAZEBO_GUI_ARGS=()
PREFLIGHT_CLEANUP_ARGS=()

append_boolean_optional_arg() {
    local value="${1,,}"
    local enabled_flag="$2"
    local disabled_flag="$3"
    local variable_name="$4"

    case "$value" in
        true)
            STATE_SWITCH_ARGS+=("$enabled_flag")
            ;;
        false)
            STATE_SWITCH_ARGS+=("$disabled_flag")
            ;;
        *)
            echo "错误：$variable_name 必须设置为 true 或 false，当前值：$1"
            exit 1
            ;;
    esac
}

case "${PREFLIGHT_CLEANUP_ENABLED,,}" in
    true)
        PREFLIGHT_CLEANUP_ARGS+=(--preflight-cleanup)
        ;;
    false)
        PREFLIGHT_CLEANUP_ARGS+=(--no-preflight-cleanup)
        ;;
    *)
        echo "错误：PREFLIGHT_CLEANUP_ENABLED 必须设置为 true 或 false，当前值：$PREFLIGHT_CLEANUP_ENABLED"
        exit 1
        ;;
esac

case "${GAZEBO_GUI,,}" in
    true)
        GAZEBO_GUI_ARGS+=(--gazebo-gui)
        ;;
    false)
        GAZEBO_GUI_ARGS+=(--no-gazebo-gui)
        ;;
    *)
        echo "错误：GAZEBO_GUI 必须设置为 true 或 false，当前值：$GAZEBO_GUI"
        exit 1
        ;;
esac

append_boolean_optional_arg \
    "$STATE1_EXPLORATION_ENABLED" \
    --state1-exploration-enabled \
    --no-state1-exploration-enabled \
    STATE1_EXPLORATION_ENABLED
append_boolean_optional_arg \
    "$STATE2_EXPLORATION_ENABLED" \
    --state2-exploration-enabled \
    --no-state2-exploration-enabled \
    STATE2_EXPLORATION_ENABLED
append_boolean_optional_arg \
    "$STATE3_EXPLORATION_ENABLED" \
    --state3-exploration-enabled \
    --no-state3-exploration-enabled \
    STATE3_EXPLORATION_ENABLED
append_boolean_optional_arg \
    "$STATE4_NOT_HOME_EXPLORATION_ENABLED" \
    --state4-not-home-exploration-enabled \
    --no-state4-not-home-exploration-enabled \
    STATE4_NOT_HOME_EXPLORATION_ENABLED
append_boolean_optional_arg \
    "$STATE4_HOME_EXPLORATION_ENABLED" \
    --state4-home-exploration-enabled \
    --no-state4-home-exploration-enabled \
    STATE4_HOME_EXPLORATION_ENABLED

append_boolean_optional_arg \
    "$STATE1_ACTOR_UPDATE_ENABLED" \
    --state1-actor-update-enabled \
    --no-state1-actor-update-enabled \
    STATE1_ACTOR_UPDATE_ENABLED
append_boolean_optional_arg \
    "$STATE2_ACTOR_UPDATE_ENABLED" \
    --state2-actor-update-enabled \
    --no-state2-actor-update-enabled \
    STATE2_ACTOR_UPDATE_ENABLED
append_boolean_optional_arg \
    "$STATE3_ACTOR_UPDATE_ENABLED" \
    --state3-actor-update-enabled \
    --no-state3-actor-update-enabled \
    STATE3_ACTOR_UPDATE_ENABLED
append_boolean_optional_arg \
    "$STATE4_NOT_HOME_ACTOR_UPDATE_ENABLED" \
    --state4-not-home-actor-update-enabled \
    --no-state4-not-home-actor-update-enabled \
    STATE4_NOT_HOME_ACTOR_UPDATE_ENABLED
append_boolean_optional_arg \
    "$STATE4_HOME_ACTOR_UPDATE_ENABLED" \
    --state4-home-actor-update-enabled \
    --no-state4-home-actor-update-enabled \
    STATE4_HOME_ACTOR_UPDATE_ENABLED

if [[ -n "$ACTOR_LEARNING_RATE" ]]; then
    ACTOR_LEARNING_RATE_ARGS+=(
        --actor-learning-rate
        "$ACTOR_LEARNING_RATE"
    )
fi

if [[ -n "$ACTOR_BACKBONE_LEARNING_RATE" ]]; then
    ACTOR_LEARNING_RATE_ARGS+=(
        --actor-backbone-learning-rate
        "$ACTOR_BACKBONE_LEARNING_RATE"
    )
fi


# ============================================================
# 16. 显示本次训练配置
# ============================================================

echo
echo "============================================================"
echo "Direct ACT PPO multi-world finetune"
echo "============================================================"
echo "Pretrained ACT:"
echo "  $PRETRAINED_MODEL"
echo
echo "Output directory:"
echo "  $OUTPUT_DIRECTORY"
echo
echo "World count upper limit:        $WORLD_COUNT"
echo "Gazebo GUI per world:           $GAZEBO_GUI"
echo "World lifecycle:               record all episode steps, then rebuild"
echo "Base ROS domain ID:             $BASE_ROS_DOMAIN_ID"
echo "Gazebo partition prefix:        $GZ_PARTITION_PREFIX"
echo "Assisted rigid grasp:           $SO101_ASSISTED_GRASP_ENABLED (simulation attachment; not natural friction grasp)"
echo "Environment randomization:      $ENVIRONMENT_RANDOMIZATION"
echo "Image augmentation only:        $IMAGE_AUGMENTATION"
echo "Image augmentation config:      $IMAGE_AUGMENTATION_CONFIG"
if [[ "${ENVIRONMENT_RANDOMIZATION,,}" == "true" ]]; then
    echo "Randomization config:           $ENVIRONMENT_RANDOMIZATION_CONFIG"
    echo "Randomization base seed:        $ENVIRONMENT_RANDOMIZATION_SEED"
fi
echo "World startup max attempts:     $WORLD_STARTUP_MAX_ATTEMPTS"
echo "World restart max attempts:     $WORLD_RESTART_MAX_ATTEMPTS"
echo "Deferred restart max attempts:  $WORLD_DEFERRED_RESTART_MAX_ATTEMPTS"
echo "Preflight cleanup enabled:      $PREFLIGHT_CLEANUP_ENABLED"
echo "Preflight cleanup timeout:      $PREFLIGHT_CLEANUP_TIMEOUT_S s"
echo
echo "Episode count minimum:          $EPISODE_COUNT"
echo "Max consecutive episode fails:  $MAX_CONSECUTIVE_EPISODE_FAILURES"
echo "Maximum steps per episode:      $MAX_STEPS_PER_EPISODE"
echo "Rollout transitions per update: $ROLLOUT_STEPS"
if [[ "$TRAINING_STAGES" == "s0_s1" ]]; then
    echo "Wave stage:                     same stage in selected worlds"
    echo "Wave world count:               planned up to $WORLD_COUNT; tail waves may use 1 or 2"
    echo "S0 actual step target/tolerance: $S0_STEP_FRACTION / $STAGE_RATIO_TOLERANCE (absolute)"
    echo "Stage rollout maximum multiple: $STAGE_MAX_ROLLOUT_MULTIPLIER (checked between waves)"
fi
echo "PPO minibatch size:             $MINIBATCH_SIZE"
echo "PPO epochs:                     $PPO_EPOCHS"
echo
echo "Gamma:                          $GAMMA"
echo "GAE lambda:                     $GAE_LAMBDA"
echo "Clip ratio:                     $CLIP_RATIO"
echo "Truncated Gaussian KL limit:    $TARGET_KL"
echo "KL rejection:                   restore epoch, retry with smaller policy LR"
echo "KL retry limit/factor:          $KL_BACKOFF_MAX_RETRIES / $KL_BACKOFF_FACTOR"
echo "KL minimum policy LR scale:     $KL_BACKOFF_MIN_SCALE"
echo "Value loss coefficient:         $VALUE_LOSS_COEFFICIENT"
echo "Latent entropy coefficient:     $ENTROPY_COEFFICIENT"
echo "Mean bounds coefficient:        $MEAN_BOUNDS_COEFFICIENT"
echo "Raw ACT bounds coefficient:     $RAW_MEAN_BOUNDS_COEFFICIENT"
echo "Actor maximum gradient norm:    $MAX_GRADIENT_NORM"
echo "Critic maximum gradient norm:   $CRITIC_MAX_GRADIENT_NORM"
echo "Exploration max gradient norm:  $EXPLORATION_MAX_GRADIENT_NORM"
echo
echo "Critic image size:              ${CRITIC_IMAGE_WIDTH}x${CRITIC_IMAGE_HEIGHT}"
echo "Critic hidden dimension:        $CRITIC_HIDDEN_DIM"
echo "Critic learning rate:           $CRITIC_LEARNING_RATE"
echo "Exploration learning rate:      $EXPLORATION_LEARNING_RATE"
echo
echo "Initial log std:                $INITIAL_LOG_STD"
echo "Minimum log std:                $MINIMUM_LOG_STD"
echo "Maximum log std:                $MAXIMUM_LOG_STD"
echo "ACT execution queue cap:        ${ACT_MAX_ACTION_STEPS:-inherit checkpoint}"
if [[ "$REWARD_PROFILE" == "pick_place_v2" ]]; then
    echo "Reach position curriculum:      $REACH_CURRICULUM"
    echo "Curriculum window / pass rate:  $CURRICULUM_WINDOW_SIZE / $CURRICULUM_PASS_RATE"
    echo "Moving gripper log std:         $MOVING_GRIPPER_EXPLORATION_LOG_STD"
    echo "Manipulation gripper log std:   $MANIPULATION_GRIPPER_EXPLORATION_LOG_STD"
    echo "Gripper effective log std cap:  $GRIPPER_MAXIMUM_LOG_STD"
    echo "Idle initial log std:           $IDLE_EXPLORATION_LOG_STD"
    echo "Moving initial log std:         $MOVING_EXPLORATION_LOG_STD"
    echo "Manipulation initial log std:   $MANIPULATION_EXPLORATION_LOG_STD"
    echo "HOME hold initial log std:      $HOME_HOLD_EXPLORATION_LOG_STD"
    echo "Exploration / ACT update:       enabled in every task phase"
else
    echo "State 1 exploration enabled:    $STATE1_EXPLORATION_ENABLED"
    echo "State 1 actor update enabled:   $STATE1_ACTOR_UPDATE_ENABLED"
    echo "State 1 initial log std:        $STATE1_EXPLORATION_LOG_STD"
    echo "State 2 exploration enabled:    $STATE2_EXPLORATION_ENABLED"
    echo "State 2 actor update enabled:   $STATE2_ACTOR_UPDATE_ENABLED"
    echo "State 2 initial log std:        $STATE2_EXPLORATION_LOG_STD"
    echo "State 3 exploration enabled:    $STATE3_EXPLORATION_ENABLED"
    echo "State 3 actor update enabled:   $STATE3_ACTOR_UPDATE_ENABLED"
    echo "State 3 initial log std:        $STATE3_EXPLORATION_LOG_STD"
    echo "State 4 not-HOME exploration:   $STATE4_NOT_HOME_EXPLORATION_ENABLED"
    echo "State 4 not-HOME actor update:  $STATE4_NOT_HOME_ACTOR_UPDATE_ENABLED"
    echo "State 4 not-HOME log std:       $STATE4_NOT_HOME_EXPLORATION_LOG_STD"
    echo "State 4 HOME exploration:       $STATE4_HOME_EXPLORATION_ENABLED"
    echo "State 4 HOME actor update:      $STATE4_HOME_ACTOR_UPDATE_ENABLED"
    echo "State 4 HOME log std:           $STATE4_HOME_EXPLORATION_LOG_STD"
fi
echo
echo "Actor learning rate:            ${ACTOR_LEARNING_RATE:-inherit pretrained}"
echo "Actor backbone learning rate:   ${ACTOR_BACKBONE_LEARNING_RATE:-inherit pretrained}"
echo
echo "Target delay:                   $TARGET_DELAY_S s"
echo "Final target-absent window:     $FINAL_TARGET_ABSENT_S s"
echo "Checkpoint every updates:       $CHECKPOINT_EVERY_UPDATES"
echo "TensorBoard preferred port:     $TENSORBOARD_PORT"
echo "Seed:                           $SEED"
echo "Device:                         $DEVICE"
echo
echo "TensorBoard logging:            enabled"
echo "TensorBoard browser opening:    enabled"
echo "Exploration control:            independent per reward state"
echo "ACT actor update control:       independent per reward state"
echo "Maximum joint delta limit:      disabled"
echo "============================================================"
echo


# ============================================================
# 17. Direct ACT PPO finetune
# ============================================================

REWARD_ARGS=(--reward-profile "$REWARD_PROFILE" --task-mode "$TASK_MODE")
REWARD_ARGS+=(--training-stages "$TRAINING_STAGES" --s0-probability "$S0_PROBABILITY")
if [[ -n "$STAGE_CONFIG" ]]; then REWARD_ARGS+=(--stage-config "$STAGE_CONFIG"); fi
if [[ -n "$S0_BUDGET_S" ]]; then REWARD_ARGS+=(--s0-budget-s "$S0_BUDGET_S"); fi
if [[ -n "$S0_HOME_HOLD_S" ]]; then REWARD_ARGS+=(--s0-home-hold-s "$S0_HOME_HOLD_S"); fi
if [[ -n "$S1_BUDGET_S" ]]; then REWARD_ARGS+=(--s1-budget-s "$S1_BUDGET_S"); fi
if [[ -n "$S1_REACH_HOLD_S" ]]; then REWARD_ARGS+=(--s1-reach-hold-s "$S1_REACH_HOLD_S"); fi
if [[ -n "$S2_BUDGET_S" ]]; then REWARD_ARGS+=(--s2-budget-s "$S2_BUDGET_S"); fi
if [[ -n "$S3_BUDGET_S" ]]; then REWARD_ARGS+=(--s3-budget-s "$S3_BUDGET_S"); fi
if [[ -n "$S4_BUDGET_S" ]]; then REWARD_ARGS+=(--s4-budget-s "$S4_BUDGET_S"); fi
if [[ -n "$STAGE_INITIALIZATION_TIMEOUT_S" ]]; then REWARD_ARGS+=(--stage-initialization-timeout-s "$STAGE_INITIALIZATION_TIMEOUT_S"); fi
if [[ -n "$ACT_MAX_ACTION_STEPS" ]]; then REWARD_ARGS+=(--act-max-action-steps "$ACT_MAX_ACTION_STEPS"); fi
if [[ -n "$REWARD_CONFIG" ]]; then REWARD_ARGS+=(--reward-config "$REWARD_CONFIG"); fi
EXPLORATION_ARGS=()
STAGE_SCHEDULE_ARGS=(--s0-step-fraction "$S0_STEP_FRACTION"
    --stage-ratio-tolerance "$STAGE_RATIO_TOLERANCE"
    --stage-max-rollout-multiplier "$STAGE_MAX_ROLLOUT_MULTIPLIER")
if [[ "$TRAINING_STAGES" == "s0_s5" ]]; then
    STAGE_SCHEDULE_ARGS+=(--stage-step-weights "$STAGE_STEP_WEIGHTS")
fi
STAGE_SUCCESS_SKIP_ARGS=()
if [[ "$TRAINING_STAGES" != "full" ]]; then
    case "$STAGE_SUCCESS_SKIP_ENABLED" in
        true) STAGE_SUCCESS_SKIP_ARGS+=(--stage-success-skip);;
        false) STAGE_SUCCESS_SKIP_ARGS+=(--no-stage-success-skip);;
        *) echo "错误：STAGE_SUCCESS_SKIP_ENABLED 必须为 true 或 false。" >&2; exit 2;;
    esac
    STAGE_SUCCESS_SKIP_ARGS+=(--stage-success-skip-threshold "$STAGE_SUCCESS_SKIP_THRESHOLD")
    echo "Stage success skip: $STAGE_SUCCESS_SKIP_ENABLED; threshold: $STAGE_SUCCESS_SKIP_THRESHOLD; skip one full batch, refill steps with remaining stages"
fi
if [[ "$REWARD_PROFILE" == "pick_place_v2" ]]; then
    case "$REACH_CURRICULUM" in
        true) REWARD_ARGS+=(--reach-curriculum);;
        false) REWARD_ARGS+=(--no-reach-curriculum);;
        *) echo "错误：REACH_CURRICULUM 必须为 true 或 false。" >&2; exit 2;;
    esac
    REWARD_ARGS+=(--curriculum-window-size "$CURRICULUM_WINDOW_SIZE"
        --curriculum-pass-rate "$CURRICULUM_PASS_RATE"
        --curriculum-rehearsal-probability "$CURRICULUM_REHEARSAL_PROBABILITY")
    FILTERED_SWITCH_ARGS=()
    for flag in "${STATE_SWITCH_ARGS[@]}"; do
        case "$flag" in --state*|--no-state*) ;; *) FILTERED_SWITCH_ARGS+=("$flag");; esac
    done
    STATE_SWITCH_ARGS=("${FILTERED_SWITCH_ARGS[@]}")
    EXPLORATION_ARGS=(--exploration-profile "$EXPLORATION_PROFILE"
        --idle-exploration-log-std "$IDLE_EXPLORATION_LOG_STD"
        --moving-exploration-log-std "$MOVING_EXPLORATION_LOG_STD"
        --manipulation-exploration-log-std "$MANIPULATION_EXPLORATION_LOG_STD"
        --moving-gripper-exploration-log-std "$MOVING_GRIPPER_EXPLORATION_LOG_STD"
        --manipulation-gripper-exploration-log-std "$MANIPULATION_GRIPPER_EXPLORATION_LOG_STD"
        --gripper-maximum-log-std "$GRIPPER_MAXIMUM_LOG_STD"
        --home-hold-exploration-log-std "$HOME_HOLD_EXPLORATION_LOG_STD"
        --idle-empty-probability "$IDLE_EMPTY_PROBABILITY"
        --idle-completed-probability "$IDLE_COMPLETED_PROBABILITY")
else
    EXPLORATION_ARGS=(--state1-exploration-log-std "$STATE1_EXPLORATION_LOG_STD"
        --state2-exploration-log-std "$STATE2_EXPLORATION_LOG_STD"
        --state3-exploration-log-std "$STATE3_EXPLORATION_LOG_STD"
        --state4-not-home-exploration-log-std "$STATE4_NOT_HOME_EXPLORATION_LOG_STD"
        --state4-home-exploration-log-std "$STATE4_HOME_EXPLORATION_LOG_STD")
fi
echo "Reward profile: $REWARD_PROFILE; config: $REWARD_CONFIG; task mode: $TASK_MODE"
if [[ "$REWARD_PROFILE" == "pick_place_v2" ]]; then
    echo "Exploration profile: $EXPLORATION_PROFILE (final_good_act uses physical-angle noise; stage LOG_STD overrides apply only to stage_log_std)"
fi
echo "Training stages: $TRAINING_STAGES; config: $STAGE_CONFIG"
if [[ "$TRAINING_STAGES" == "s0_s1" ]]; then
    echo "S0_PROBABILITY is unused in multi-world s0_s1; effective S0_STEP_FRACTION=$S0_STEP_FRACTION"
fi
if [[ "$TRAINING_STAGES" == "s0_s5" ]]; then
    echo "Stage step weights (S0:S1:S2:S3:S4:S5): $STAGE_STEP_WEIGHTS; ratio tolerance: $STAGE_RATIO_TOLERANCE (absolute)"
    echo "Skipped stages are removed for one batch; remaining step weights are renormalized and fill ROLLOUT_STEPS"
    echo "Later-stage starts: physically prepared and certified before ACT collection; fresh critic context required"
fi
echo "Later budgets (s): S2=${S2_BUDGET_S:-inherit JSON}, S3=${S3_BUDGET_S:-inherit JSON}, S4=${S4_BUDGET_S:-inherit JSON}; S5 shares S0"
echo "Stage preparation timeout: ${STAGE_INITIALIZATION_TIMEOUT_S:-180s S0/S1, 600s later-stage context}"
echo "Stage time overrides (s): S0 budget=${S0_BUDGET_S:-inherit JSON}, HOME hold=${S0_HOME_HOLD_S:-inherit JSON}; S1 budget=${S1_BUDGET_S:-inherit JSON}, reach hold=${S1_REACH_HOLD_S:-inherit JSON}"

python -m so101_capstone.training.act_ppo_multiworld_gazebo \
    --act-checkpoint "$PRETRAINED_MODEL" \
    "${REWARD_ARGS[@]}" \
    "${STAGE_SCHEDULE_ARGS[@]}" \
    "${STAGE_SUCCESS_SKIP_ARGS[@]}" \
    "${EXPLORATION_ARGS[@]}" \
    --device "$DEVICE" \
    --world-count "$WORLD_COUNT" \
    --base-ros-domain-id "$BASE_ROS_DOMAIN_ID" \
    --gz-partition-prefix "$GZ_PARTITION_PREFIX" \
    "${GAZEBO_GUI_ARGS[@]}" \
    --world-startup-max-attempts "$WORLD_STARTUP_MAX_ATTEMPTS" \
    --world-restart-max-attempts "$WORLD_RESTART_MAX_ATTEMPTS" \
    --world-deferred-restart-max-attempts "$WORLD_DEFERRED_RESTART_MAX_ATTEMPTS" \
    "${PREFLIGHT_CLEANUP_ARGS[@]}" \
    --preflight-cleanup-timeout-s "$PREFLIGHT_CLEANUP_TIMEOUT_S" \
    --episode-count "$EPISODE_COUNT" \
    --max-consecutive-episode-failures "$MAX_CONSECUTIVE_EPISODE_FAILURES" \
    --max-steps-per-episode "$MAX_STEPS_PER_EPISODE" \
    --target-delay-s "$TARGET_DELAY_S" \
    --final-target-absent-s "$FINAL_TARGET_ABSENT_S" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --minibatch-size "$MINIBATCH_SIZE" \
    --ppo-epochs "$PPO_EPOCHS" \
    --critic-image-height "$CRITIC_IMAGE_HEIGHT" \
    --critic-image-width "$CRITIC_IMAGE_WIDTH" \
    --critic-hidden-dim "$CRITIC_HIDDEN_DIM" \
    --critic-learning-rate "$CRITIC_LEARNING_RATE" \
    --exploration-learning-rate "$EXPLORATION_LEARNING_RATE" \
    --gamma "$GAMMA" \
    --gae-lambda "$GAE_LAMBDA" \
    --clip-ratio "$CLIP_RATIO" \
    --target-kl "$TARGET_KL" \
    --kl-backoff-max-retries "$KL_BACKOFF_MAX_RETRIES" \
    --kl-backoff-factor "$KL_BACKOFF_FACTOR" \
    --kl-backoff-min-scale "$KL_BACKOFF_MIN_SCALE" \
    --value-loss-coefficient "$VALUE_LOSS_COEFFICIENT" \
    --entropy-coefficient "$ENTROPY_COEFFICIENT" \
    --mean-bounds-coefficient "$MEAN_BOUNDS_COEFFICIENT" \
    --raw-mean-bounds-coefficient "$RAW_MEAN_BOUNDS_COEFFICIENT" \
    --max-gradient-norm "$MAX_GRADIENT_NORM" \
    --critic-max-gradient-norm "$CRITIC_MAX_GRADIENT_NORM" \
    --exploration-max-gradient-norm "$EXPLORATION_MAX_GRADIENT_NORM" \
    --initial-log-std "$INITIAL_LOG_STD" \
    --minimum-log-std "$MINIMUM_LOG_STD" \
    --maximum-log-std "$MAXIMUM_LOG_STD" \
    "${STATE_SWITCH_ARGS[@]}" \
    --checkpoint-every-updates "$CHECKPOINT_EVERY_UPDATES" \
    --tensorboard-port "$TENSORBOARD_PORT" \
    --startup-feedback-timeout-s "$STARTUP_FEEDBACK_TIMEOUT_S" \
    --seed "$SEED" \
    --disable-max-joint-delta-limit \
    --output-directory "$OUTPUT_DIRECTORY" \
    "${ACTOR_LEARNING_RATE_ARGS[@]}"
